"""Compile the canonical scene graph into the Figma-facing design schema v2."""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .schema import DesignDoc, Layer, SCHEMA_VERSION, dump, validate_design
from .text_analysis import fit_text_box
from .raster_clusters import is_intentional_raster_cluster

# Candidate keys that ``_compile`` already routes to a concrete Layer field.  Anything
# else on a reconstruct entity (e.g. an image ``ref`` or a future mask spec another
# stage attaches) is unknown to the dataclass, so it is preserved under
# ``meta['passthrough']`` rather than silently dropped on the way to design.json.
_CONSUMED_CANDIDATE_KEYS = frozenset({
    "id", "target", "box", "meta", "z_index", "z", "visible_box", "ink_box",
    "rotation", "opacity", "blend_mode", "effects", "constraints", "component",
    "layout", "children", "text", "style", "text_runs", "fill", "stroke",
    "shape_kind", "path", "svg", "src", "radius", "paths", "mask", "name", "role",
})


def _truncate(value, length=28):
    value = " ".join(str(value or "").split())
    return value if len(value) <= length else value[: length - 1] + "…"


def _name(candidate):
    meta = candidate.get("meta") or {}
    # Names are part of the deliverable, not cosmetic metadata.  When the scene/VLM
    # knows an asset's identity, preserve it verbatim so a designer sees e.g. "X logo"
    # rather than a pile of anonymous `Image — cutout` layers.  Fall back only when no
    # semantic label survived detection.
    explicit = (candidate.get("name") or meta.get("semantic_name") or
                meta.get("layer_name") or meta.get("vlm_name") or meta.get("label"))
    if explicit:
        return _truncate(explicit, 56)
    role = str(meta.get("role") or "").strip()
    target = candidate.get("target")
    if target == "text":
        return f'{(role or "Text").title()} — "{_truncate(candidate.get("text"))}"'
    if target == "image":
        if meta.get("wordmark"):
            return f'Logo — {_truncate(candidate.get("text") or "wordmark")} (raster crop)'
        if meta.get("substitution") or meta.get("low_fidelity"):
            return f'Text (fallback) — "{_truncate(candidate.get("text"))}"'
        # These are deliberately image-filled native nodes, so they remain trivially
        # swappable in Figma even when a complex asset cannot safely be vectorized.
        return f'{(role or "Image").title()} — swappable crop'
    if target == "icon":
        return f'{(role or "Icon").title()} — vector'
    if target == "group":
        return f'{(role or "Frame").title()}'
    return (role or "Shape").title()


def _resolve(path: Optional[str], run_dir: str) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.normpath(os.path.join(run_dir, path))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return None


def _stage_asset(src: Optional[str], layer_id: str, run_dir: str, warnings: list) -> Optional[str]:
    resolved = _resolve(src, run_dir)
    if not resolved:
        warnings.append({"code": "missing-asset", "layer_id": layer_id, "path": src})
        return None
    # Existing-but-truncated assets are just as unusable as missing files.  Detect them
    # before copying so preview/Figma never receive a poisoned checkpoint.
    try:
        if os.path.getsize(resolved) <= 0:
            raise ValueError("empty file")
        if os.path.splitext(resolved)[1].lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"
        }:
            from PIL import Image
            with Image.open(resolved) as image:
                image.verify()
    except (OSError, ValueError, SyntaxError) as exc:
        warnings.append({
            "code": "corrupt-asset", "layer_id": layer_id, "path": src,
            "detail": str(exc),
        })
        return None
    assets = os.path.join(run_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    # Assets already in the run are canonical; do not duplicate them on every rebuild.
    if os.path.commonpath([os.path.abspath(resolved), os.path.abspath(assets)]) == os.path.abspath(assets):
        return os.path.relpath(resolved, run_dir)
    base = os.path.basename(resolved)
    destination = os.path.join(assets, f"{layer_id}_{base}")
    if os.path.abspath(resolved) != os.path.abspath(destination):
        shutil.copyfile(resolved, destination)
    return os.path.relpath(destination, run_dir)


def _surface_fill(candidate):
    fill = candidate.get("fill")
    if fill is not None:
        return fill
    style = candidate.get("style") or {}
    fills = style.get("fills")
    if isinstance(fills, list) and fills:
        return fills[0]
    if style.get("fill") is not None:
        return style.get("fill")
    if style.get("color"):
        return {"kind": "flat", "color": style["color"]}
    return None


def _semantic_z(candidate, target):
    """Return a stable fallback when upstream stages emit the placeholder z=0."""
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or candidate.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"} or meta.get("source") == "inpaint":
        return -1_000_000
    band = str(meta.get("z_band") or "").lower()
    band_z = {
        "background": -1_000_000, "plate": -1_000_000,
        "content": 20, "scene": 20, "foreground": 30,
        "overlay": 40, "chrome": 50, "ui": 50,
    }.get(band)
    if band_z is not None:
        return band_z
    if target == "text":
        return 40
    if target == "icon":
        return 35
    if target == "image":
        return 30 if role not in {"background", "photo-fragment"} else 25
    if target in {"shape", "group"}:
        return 20
    return 10


def _compile(candidate: dict, run_dir: str, warnings: list) -> Layer:
    target = candidate.get("target")
    layer_id = str(candidate.get("id") or "layer")
    box = dict(candidate.get("box") or {"x": 0, "y": 0, "w": 1, "h": 1})
    meta = dict(candidate.get("meta") or {})
    # The Figma importer understands the complete style object (multiple paints,
    # strokes and effects).  Keep it intact for every editable layer instead of
    # reducing non-text layers to their first fill before export.
    source_style = dict(candidate.get("style") or {})
    source_effects = candidate.get("effects")
    if source_effects is None:
        source_effects = source_style.get("effects")
    if candidate.get("z_index") is not None:
        z_raw = candidate.get("z_index")
    elif candidate.get("z") is not None:
        z_raw = candidate.get("z")
    elif meta.get("z") is not None:
        z_raw = meta.get("z")
    else:
        # Missing z is not an explicit paint-order instruction.  Route it through the
        # semantic stack below so a gradient/background shape stays behind its image,
        # and icons/text stay above that image.  The old image default (10) silently
        # put unannotated photos behind native gradient surfaces.
        z_raw = None
    # Fusion assigns OCR a small ``z=1`` merely to distinguish it from its
    # detected shell.  It is not a final paint order: native button/card shapes
    # receive semantic z=20 and would otherwise cover their own CTA. Preserve
    # genuinely explicit text z-orders (>1), but promote the fusion placeholder
    # to the normal front text band.
    text_placeholder_z = target == "text" and z_raw in (None, 0, 1, "0", "0.0", "1", "1.0")
    z_index = float(_semantic_z(candidate, target) if text_placeholder_z or z_raw in (None, 0, "0", "0.0") else z_raw)
    if meta.get("substitution"):
        warnings.append({"code": "text-fidelity-fallback", "layer_id": layer_id, **meta["substitution"]})
    common = {
        "id": layer_id,
        "name": _name(candidate),
        "box": box,
        "z_index": z_index,
        "visible_box": candidate.get("visible_box") or candidate.get("ink_box"),
        "rotation": float(candidate.get("rotation", 0) or 0),
        "opacity": float(candidate.get("opacity", 1) if candidate.get("opacity") is not None else 1),
        "blend_mode": str(candidate.get("blend_mode") or "NORMAL"),
        "effects": list(source_effects) if isinstance(source_effects, list) else [],
        "meta": {**meta, "z": z_index, "source_id": layer_id},
        "constraints": dict(candidate.get("constraints") or {}),
        "component": dict(candidate.get("component") or {}),
        "layout": dict(candidate.get("layout") or {}),
    }
    passthrough = {key: value for key, value in candidate.items()
                   if key not in _CONSUMED_CANDIDATE_KEYS}
    if passthrough:
        common["meta"]["passthrough"] = passthrough

    if target == "group":
        children = []
        for child in candidate.get("children") or []:
            try:
                children.append(_compile(child, run_dir, warnings))
            except Exception as exc:
                warnings.append({
                    "code": "layer-compile-error", "layer_id": child.get("id"),
                    "detail": str(exc),
                })
        children.sort(key=lambda child: child.z_index)
        return Layer(
            type="group",
            children=children,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            radius=candidate.get("radius") or source_style.get("radius"),
            style=source_style,
            shape_kind="frame",
            **common,
        )

    if target == "text":
        style = source_style
        fill = candidate.get("fill") or style.pop("fill", None)
        stroke = candidate.get("stroke") or style.pop("stroke", None)
        text_value = str(candidate.get("text") or "")
        # Fit against ink/painted bounds when available so Python preview and the Figma
        # plugin agree on the same target box (plugin uses visible_box in fitTextToVisibleBox).
        fit_box = dict(
            candidate.get("visible_box") or candidate.get("ink_box") or common["box"]
        )
        fitted_box, auto_resize, style_patch = fit_text_box(text_value, style, fit_box)
        common["box"] = fitted_box
        style.update(style_patch)
        style.setdefault("autoResize", auto_resize)
        style["preFitted"] = True
        style["fit"] = False
        return Layer(
            type="text",
            text=text_value,
            style=style,
            text_runs=list(candidate.get("text_runs") or []),
            fill=fill,
            stroke=stroke,
            **common,
        )

    if target == "shape":
        return Layer(
            type="shape",
            shape_kind=candidate.get("shape_kind") or "rect",
            path=candidate.get("path"),
            svg=candidate.get("svg"),
            src=_stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
                if candidate.get("src") else None,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            radius=candidate.get("radius") or source_style.get("radius"),
            style=source_style,
            **common,
        )

    if target == "icon":
        paths = list(candidate.get("paths") or [])
        svg = candidate.get("svg")
        path = candidate.get("path") or (paths[0].get("d") if len(paths) == 1 else None)
        return Layer(
            type="shape",
            shape_kind="path",
            path=path,
            svg=svg,
            src=_stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
                if candidate.get("src") else None,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            style=source_style,
            meta={**common.pop("meta"), "vector_paths": paths},
            **common,
        )

    # Unknown candidates route conservatively to an alpha raster, never a fake gray box.
    src = _stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
    layer_meta = common.pop("meta")
    if not src:
        layer_meta["compiler_error"] = "missing image asset"
    mask = dict(candidate.get("mask") or {}) if isinstance(candidate.get("mask"), dict) else None
    if mask and mask.get("src"):
        mask["src"] = _stage_asset(mask.get("src"), f"{layer_id}_mask", run_dir, warnings)
    return Layer(type="image", src=src, mask=mask, style=source_style, meta=layer_meta, **common)


def _count_layers(layers):
    return sum(1 + _count_layers(layer.children) for layer in layers)


def _count_editable(layers):
    return sum((1 if layer.type in ("text", "shape", "group") else 0) +
               _count_editable(layer.children) for layer in layers)


_LEGITIMATE_RASTER_ROLES = frozenset({
    "background", "photo", "image", "product", "product-cluster", "person",
    "people", "face", "hand", "avatar", "profile", "profile-photo", "thumbnail",
    "illustration", "package", "logo", "wordmark", "brand", "logotype",
})


def _leaf_accounting(layers):
    """Describe real foreground material without letting wrapper groups inflate editability.

    The historical editable ratio counted every FRAME/GROUP as editable, even when that frame
    contained only one raster screenshot. Keep the old metric for compatibility, but publish a
    leaf-only accounting contract that acceptance QA can audit honestly.
    """
    out = {
        "foreground_leaf_count": 0,
        "native_leaf_count": 0,
        "raster_leaf_count": 0,
        "intentional_raster_cluster_count": 0,
        "fallback_raster_count": 0,
        "unexplained_raster_count": 0,
        "unexplained_raster_ids": [],
    }

    def visit(layer):
        children = list(layer.children or [])
        if children:
            for child in children:
                visit(child)
            return
        meta = layer.meta or {}
        role = str(meta.get("role") or "").strip().lower().replace("_", "-")
        if role == "background":
            return
        out["foreground_leaf_count"] += 1
        if layer.type in ("text", "shape"):
            out["native_leaf_count"] += 1
            return
        if layer.type != "image":
            return
        out["raster_leaf_count"] += 1
        intentional = bool(meta.get("intentional_raster_cluster")) or is_intentional_raster_cluster(role)
        if intentional:
            out["intentional_raster_cluster_count"] += 1
        # "fallback" flags (fallback/raster_fallback/vector_fallback/substitution/low_fidelity)
        # are set by routing/vectorize exactly when they GIVE UP on producing a native layer.
        # They are therefore NOT a legitimate reason on their own -- treating them as
        # self-justifying let every give-up silently launder itself into "explained" and
        # never increment unexplained_raster_count. A raster leaf only counts as explained
        # when it has an INDEPENDENT legitimate reason: a genuine legitimate role (photo,
        # product, person, logo, ...), an intentional raster cluster, or a wordmark. A bare
        # semantic_name is a label, not a reason, and does not grant a free pass either.
        fallback = bool(
            meta.get("fallback") or meta.get("raster_fallback") or meta.get("vector_fallback")
            or meta.get("substitution") or meta.get("low_fidelity")
        )
        if fallback:
            out["fallback_raster_count"] += 1
        legitimate = bool(intentional or role in _LEGITIMATE_RASTER_ROLES or meta.get("wordmark"))
        explained = legitimate
        if fallback and not explained:
            out["unexplained_raster_count"] += 1
            out["unexplained_raster_ids"].append(str(layer.id))

    for root in layers:
        visit(root)
    denominator = max(1, out["foreground_leaf_count"])
    out["native_leaf_ratio"] = round(out["native_leaf_count"] / denominator, 4)
    out["unexplained_raster_ids"] = sorted(out["unexplained_raster_ids"])
    return out


def build(candidates: list, canvas: dict, run_dir: str, base_src: str | None = None,
          doc_id: str = "doc", name: str = "design", kept_in_photo: Optional[list] = None) -> DesignDoc:
    """Build schema v2.

    ``base_src`` must be a reconstructed clean plate. Refusing the normalized/original source
    here prevents the old duplicate-elements architecture from silently returning.
    """
    os.makedirs(run_dir, exist_ok=True)
    warnings = []
    if base_src and candidates and os.path.basename(base_src).lower() in ("normalized.png", "original.png"):
        raise ValueError("refusing untouched source as rebuilt background; run reconstruct/inpaint first")

    layers = []
    if base_src:
        base_rel = _stage_asset(base_src, "background", run_dir, warnings)
        layers.append(Layer(
            id="background", type="image", name="Background — clean plate",
            box={"x": 0, "y": 0, "w": canvas["w"], "h": canvas["h"]},
            z_index=-1_000_000, src=base_rel,
            constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
            meta={"source": "inpaint", "role": "background", "z": -1_000_000},
        ))

    kept = list(kept_in_photo or [])
    for candidate in candidates:
        if candidate.get("target") == "drop":
            if candidate.get("text"):
                kept.append(str(candidate["text"]).strip())
            continue
        try:
            layers.append(_compile(candidate, run_dir, warnings))
        except Exception as exc:
            # One malformed entity must not hide all other editable layers. Omit only the
            # broken entity and make the partial compilation a hard structural QA failure.
            warnings.append({
                "code": "layer-compile-error", "layer_id": candidate.get("id"),
                "detail": str(exc),
            })
    layers.sort(key=lambda layer: layer.z_index)
    total = _count_layers(layers)
    editable = _count_editable(layers)
    leaf_accounting = _leaf_accounting(layers)
    doc = DesignDoc(
        id=doc_id,
        name=name,
        canvas={"w": canvas["w"], "h": canvas["h"]},
        schema_version=SCHEMA_VERSION,
        layers=layers,
        kept_in_photo=sorted(set(x for x in kept if x)),
        meta={
            "layer_count": total,
            "root_layer_count": len(layers),
            # Legacy metric, kept only for back-compat: it counts every FRAME/GROUP as
            # "editable", so a wrapper frame around a single raster image inflates this
            # number even though nothing inside is actually editable. `native_leaf_ratio`
            # below (leaf-only, no wrapper credit) is the honest metric acceptance gates on.
            "editable_ratio": round(editable / max(1, total), 4),
            "native_leaf_ratio": leaf_accounting["native_leaf_ratio"],
            "leaf_accounting": leaf_accounting,
            "warnings": warnings,
            "compiler": "scene-graph-v2",
            "coordinate_space": "local",
        },
    )
    schema_errors = validate_design(doc)
    if schema_errors:
        # Same list object backs doc.meta["warnings"], so this mutation is visible in
        # the document we're about to dump without reconstructing it.
        warnings.extend({"code": "invalid-schema", "detail": msg} for msg in schema_errors)

    dump(doc, os.path.join(run_dir, "design.json"))
    dump({"ok": not warnings, "warnings": warnings, "layer_count": total},
         os.path.join(run_dir, "design_preflight.json"))
    return doc
