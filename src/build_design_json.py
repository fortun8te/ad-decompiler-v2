"""Compile the canonical scene graph into the Figma-facing design schema v2."""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .schema import DesignDoc, Layer, SCHEMA_VERSION, dump, validate_design


def _truncate(value, length=28):
    value = " ".join(str(value or "").split())
    return value if len(value) <= length else value[: length - 1] + "…"


def _name(candidate):
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").strip()
    target = candidate.get("target")
    if target == "text":
        return f'{(role or "Text").title()} — "{_truncate(candidate.get("text"))}"'
    if target == "image":
        if meta.get("wordmark"):
            return f'Logo — {_truncate(candidate.get("text") or "wordmark")}'
        if meta.get("substitution") or meta.get("low_fidelity"):
            return f'Text (fallback) — "{_truncate(candidate.get("text"))}"'
        return f'{(role or "Image").title()} — cutout'
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
    default_z = {"image": 10, "shape": 20, "group": 20, "icon": 30, "text": 40}
    if candidate.get("z_index") is not None:
        z_raw = candidate.get("z_index")
    elif candidate.get("z") is not None:
        z_raw = candidate.get("z")
    elif meta.get("z") is not None:
        z_raw = meta.get("z")
    else:
        z_raw = default_z.get(target, 10)
    # Fusion/layout currently use zero as an unset placeholder. Treat only that
    # placeholder as absent; explicit positive/negative ordering remains intact.
    z_index = float(_semantic_z(candidate, target) if z_raw in (None, 0, "0", "0.0") else z_raw)
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
        "effects": list(candidate.get("effects") or []),
        "meta": {**meta, "z": z_index, "source_id": layer_id},
        "constraints": dict(candidate.get("constraints") or {}),
        "component": dict(candidate.get("component") or {}),
        "layout": dict(candidate.get("layout") or {}),
    }

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
            fill=_surface_fill(candidate),
            stroke=candidate.get("stroke"),
            radius=candidate.get("radius") or (candidate.get("style") or {}).get("radius"),
            shape_kind="frame",
            **common,
        )

    if target == "text":
        style = dict(candidate.get("style") or {})
        fill = candidate.get("fill") or style.pop("fill", None)
        stroke = candidate.get("stroke") or style.pop("stroke", None)
        return Layer(
            type="text",
            text=str(candidate.get("text") or ""),
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
            radius=candidate.get("radius") or (candidate.get("style") or {}).get("radius"),
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
    return Layer(type="image", src=src, mask=mask, meta=layer_meta, **common)


def _count_layers(layers):
    return sum(1 + _count_layers(layer.children) for layer in layers)


def _count_editable(layers):
    return sum((1 if layer.type in ("text", "shape", "group") else 0) +
               _count_editable(layer.children) for layer in layers)


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
            "editable_ratio": round(editable / max(1, total), 4),
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
