"""Fuse SAM 3, residual-CC, and Qwen observations into canonical elements.

The existing merge stage compares only bounding boxes.  This module instead materializes
full-canvas masks, performs semantic mask-aware NMS, and gives each source observation to
exactly one canonical element.  Near-identical observations collapse; a small object inside
a real container remains a separate child rather than disappearing as a duplicate.

No model is imported here.  NumPy and Pillow are loaded lazily inside :func:`fuse`, keeping
module import CPU-safe and cheap.

Public API::

    fused = fuse(
        sam3=sam3_manifest_or_elements,
        residual=residual_elements,
        qwen=qwen_layers,
        canvas={"w": 1080, "h": 1350},
        cfg=cfg,
        run_dir="runs/example",
    )

The returned list uses canonical IDs and saves box-local masks under
``fused_elements/<id>.png`` when ``run_dir`` is supplied.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional
from .raster_clusters import INTENTIONAL_RASTER_CLUSTER_ROLES, normalized_role


DEFAULTS = {
    "mask_iou": 0.68,
    "bbox_iou": 0.78,
    "containment": 0.92,
    "similar_area_ratio": 0.70,
    # Must stay >= similar_area_ratio: the duplicate check in _duplicate() keeps a pair
    # separate (not merged) whenever containment >= containment threshold and
    # area_ratio < similar_area_ratio. If nested_max_area_ratio were lower than
    # similar_area_ratio, pairs with area_ratio in that gap would be kept separate but
    # never linked as parent/child by the parent-link pass below, i.e. they'd ship as
    # two unrelated overlapping top-level elements (duplicate ownership).
    "nested_max_area_ratio": 0.70,
    "nested_containment": 0.90,
    # ── overlap / z-order robustness ────────────────────────────────────────────────
    # A residual observation and the SAM mask that was box-refined FROM it are the same
    # object by construction; merge them on provenance even when the sparse residual CC
    # and the solid SAM mask disagree on IoU (benchmark 009: E015/E016, E007/E008).
    "link_residual_refine": True,
    "link_min_containment": 0.50,
    # Same-role masks nested inside a bigger same-role mask are fragments of one object
    # (009: whole share icon E017 + its glyph pieces E018/E023), not meaningful children;
    # collapse them into the containing cluster instead of shipping overlapping siblings.
    "absorb_fragments": True,
    "absorb_containment": 0.90,
    # Thin fill slivers (residual CCs split off a button plate by OCR text dilation)
    # inside a model-masked button-like shell are absorbed as fill fragments
    # (009: the white strips around the 'Volgend' pill).
    "sliver_absorb": True,
    "sliver_containment": 0.75,
    "sliver_max_thickness": 10,
    "sliver_min_aspect": 4.0,
    "sliver_parent_roles": ("button", "badge", "sticker", "pill"),
    # Anti-aliased fringes push child masks 1-2 px past a model parent mask; tolerate
    # that when testing child-in-parent containment (absorption + parent linking).
    "containment_dilate_px": 2,
    # Residual-CC roles come from hardcoded solidity/edge-density heuristics; a family
    # conflict against a model label must not block geometric dedup (an element must
    # never ship as both photo-fragment and shape).
    "residual_family_advisory": True,
    # ── canonical product geometry (013/135/067) ────────────────────────────────────
    # Near-identical masks are the same physical object regardless of how the two
    # observations were labelled. Without this escape hatch the product-over-photo-panel
    # carve-out in _semantic_compatible blocked a box-refine "photo" and a text-prompt
    # "product" with mask IoU 0.99 from merging (135: E004/E005 shipped as dual
    # near-duplicate full-frame groups).
    "identity_iou": 0.90,
    # A raster cluster with no semantic (text-prompt) evidence whose pixels are mostly
    # covered by the union of semantically-detected product masks is a duplicate
    # observation of already-owned pixels, not new content (067: box-refine "photo" strip
    # that was exactly the union of three distinct jars; 135: residual fragment swarm
    # straddling the two bars). Distinct products themselves are never eligible.
    "product_shadow": True,
    "product_shadow_containment": 0.75,
    # Full-bleed low-score photo bands backed only by residual/box-refine evidence are
    # background chunks the residual detector re-emitted as elements (013: the y491-989
    # gradient band that shipped as an opaque "Photo" group plate ABOVE the product and
    # decapitated the bag; 021: empty junk groups). The plate already owns those pixels.
    "suppress_junk_bands": True,
    "junk_band_min_width_frac": 0.92,
    "junk_band_min_coverage": 0.10,
    "junk_band_max_score": 0.50,
    # A logo/wordmark detection fully nested inside a product cutout is the product's
    # own printed label ink (013: the "ü snacks" pill and bear-face lockup printed on
    # the grüns bag), not a separate design element. Shipping it separately makes
    # downstream reconstruction re-render it (solid-disc "badge" over the bag) and
    # feeds the peel gate artwork-only overlap pairs. Absorb it into the parent and
    # flag the parent (meta.printed_lockup) so peel lifts the product off the plate.
    "absorb_printed_artwork": True,
    "printed_artwork_min_containment": 0.95,
    "printed_artwork_max_area_ratio": 0.5,
    # The invariant, generalized: ANY non-product, non-text child that rides inside a
    # product cutout (icon / checkmark-cross glyph / nutrition panel / small printed
    # photo or shape — not just logos and flavor bars) is that product's own packaging
    # ink and must ride the cutout, never ship as a separate element that reconstruct /
    # peel would punch out of the packaging (091 on-pack photos+icon, 094 on-pack chip,
    # 135's seven printed glyphs/shapes). Runs after the logo + shell absorbers and
    # sweeps up everything they leave behind. A decoration that straddles the product
    # edge so no nesting link formed still folds when its BOX sits mostly inside the
    # product (box_containment) — a label sticking 20% over the edge must not punch the
    # bag. Distinct SKUs (product-instance roles) and anything larger than
    # max_area_ratio of the product are never decorations.
    "absorb_product_decorations": True,
    "product_decoration_min_containment": 0.80,
    "product_decoration_box_containment": 0.80,
    "product_decoration_max_area_ratio": 0.60,
    # Adjacent product parts that share a text-prompt (jar+lid, dual chocolate bars)
    # are one physical owner when their hulls nearly touch. Multi-SKU layouts with
    # larger gaps (067's three jars) stay separate — gap is relative to object size.
    "product_cluster_merge": True,
    "product_cluster_gap_px": 4,
    "product_cluster_gap_frac": 0.18,
    "product_cluster_min_area_ratio": 0.08,
    # Swiss-cheese SAM halves can leave a mask gap while boxes still heavily overlap
    # (135 dual protein bars). Box IoU above this merges even when hulls do not touch.
    "product_cluster_box_iou": 0.22,
    # SAM packaging masks often have Swiss-cheese interior holes (stroke art, gummy
    # cuts). Seal them so the product alpha is a tight solid silhouette.
    "seal_product_holes": True,
    "product_hole_close_px": 5,
    # Flavor bars / on-pack chips SAM labels as buttons (002 VANILLE SMAAK yellow
    # highlight) are packaging ink, not CTA chrome — fold into the product owner.
    "absorb_packaging_shells": True,
    "packaging_shell_min_containment": 0.92,
    "packaging_shell_max_area_ratio": 0.18,
    "packaging_shell_min_aspect": 3.2,
    "canonical_prefix": "E",
}

_GRAPHIC_ROLES = {"logo", "icon", "arrow", "badge", "symbol", "pictogram", "sticker",
                  # list-row glyphs (src/icon_detect.py): ✓ / ✗ / ? marks
                  "verified", "checkmark", "cross", "question-mark"}
_LIST_ROW_ICON_ROLES = frozenset({
    "verified", "checkmark", "check", "check-mark", "check_mark", "tick",
    "cross", "question-mark", "question_mark",
})
_CHECKLIST_CARD_ROLES = frozenset({
    "card", "panel", "frame", "container", "plate", "shape", "badge", "",
})
_RASTER_ROLES = {
    "photo",
    "product",
    "person",
    "face",
    "hand",
    "object",
    "illustration",
    "package",
} | set(INTENTIONAL_RASTER_CLUSTER_ROLES)
# Distinct product cutouts (box vs earplugs) — merge only near-identical masks.
_PRODUCT_INSTANCE_ROLES = frozenset({
    "product", "package", "packaging", "object", "bottle", "box", "jar", "can",
    "pouch", "carton", "product-cluster", "packshot", "tube", "canister", "device",
    "sachet", "pill-cloud", "body-progression", "body-morph",
})
# Side-by-side comparison portraits / panels (Huel, MONTE, HiStrips, Wavy) must stay
# two separate frames — same-family raster agreement alone must not fuse them.
_COMPARISON_PANEL_ROLES = frozenset({
    "photo", "person", "people", "portrait", "face", "panel", "comparison-panel",
    "comparison-column", "photo-panel", "image-panel", "photo-fragment", "photo_fragment",
})
# Photo/portrait panels a seam-straddling product may overlap — never absorb/nest the
# product under these (Wavy tube across Before/After sections).
_PHOTO_PANEL_ROLES = frozenset({
    "photo", "image", "background", "panel", "portrait", "lifestyle",
    "comparison-panel", "comparison-column", "photo-panel", "image-panel",
})
_SHAPE_ROLES = {"shape", "button", "card", "container", "background", "frame"}
_CONTAINER_ROLES = {"shape", "button", "card", "container", "badge", "frame", "background"}
_STRUCTURAL_FIELDS = (
    "structure_group_id", "repeat_group_id", "panel_set_id", "grid_group_id",
    "comparison_group_id", "chart_group_id", "row_index", "column_index",
)


def _np():
    import numpy as np

    return np


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("elements", "layers", "observations"):
            if isinstance(value.get(key), list):
                return value[key]
        return []
    return list(value)


def _valid_box(box: Any) -> bool:
    return bool(
        isinstance(box, dict)
        and float(box.get("w", 0) or 0) > 0
        and float(box.get("h", 0) or 0) > 0
    )


def _clip_box(box: dict, width: int, height: int) -> dict:
    x0 = max(0, min(width, int(round(float(box.get("x", 0))))))
    y0 = max(0, min(height, int(round(float(box.get("y", 0))))))
    x1 = max(x0, min(width, int(round(float(box.get("x", 0)) + float(box.get("w", 0))))))
    y1 = max(y0, min(height, int(round(float(box.get("y", 0)) + float(box.get("h", 0))))))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _rect_mask(box: dict, width: int, height: int):
    np = _np()
    out = np.zeros((height, width), dtype=bool)
    b = _clip_box(box, width, height)
    if b["w"] and b["h"]:
        out[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]] = True
    return out


def _mask_box(mask) -> dict:
    np = _np()
    ys, xs = np.where(mask)
    if not xs.size:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def _bbox_iou(a: dict, b: dict) -> float:
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def _mask_metrics(a, b) -> dict:
    inter = int((a & b).sum())
    aa, ab = int(a.sum()), int(b.sum())
    union = aa + ab - inter
    small, large = min(aa, ab), max(aa, ab)
    return {
        "iou": inter / union if union else 0.0,
        "containment": inter / small if small else 0.0,
        "area_ratio": small / large if large else 0.0,
        "inter": inter,
    }


def _role(item: dict, source: str, canvas: dict) -> str:
    role = item.get("role") or (item.get("meta") or {}).get("role")
    if role:
        role = normalized_role(role)
    if not role and source == "qwen":
        hint = str(item.get("kind_hint") or "").strip().lower()
        role = None if hint in ("", "unknown", "object") else hint
    if not role:
        role = {
            "icon": "icon",
            "shape": "shape",
            "photo-fragment": "photo",
        }.get(item.get("kind"), "object")
    box = item.get("box") or {}
    area_frac = (float(box.get("w", 0)) * float(box.get("h", 0))) / max(
        1, int(canvas["w"]) * int(canvas["h"])
    )
    if source == "qwen" and role in ("object", "photo") and area_frac >= 0.82:
        return "background"
    return role


def _kind(role: str, fallback=None) -> str:
    if fallback in ("shape", "icon", "photo-fragment"):
        return fallback
    if role in _GRAPHIC_ROLES:
        return "icon"
    if role in _SHAPE_ROLES:
        return "shape"
    return "photo-fragment"


def _family(role: str) -> str:
    if role in _GRAPHIC_ROLES:
        return "graphic"
    if role in _RASTER_ROLES:
        return "raster"
    if role in _SHAPE_ROLES:
        return "shape"
    return "unknown"


def _heuristic_label(obs: dict) -> bool:
    """True when the observation's role came from residual-CC threshold heuristics."""
    return obs["source"] == "residual" or obs.get("mode") in (
        "residual-fallback",
        "box-refine-fallback",
    )


def _semantic_compatible(a: dict, b: dict, mask_iou: float, opts: Optional[dict] = None) -> bool:
    fa, fb = _family(a["role"]), _family(b["role"])
    ra = str(a.get("role") or "").lower().replace("_", "-")
    rb = str(b.get("role") or "").lower().replace("_", "-")
    # Near-identical masks are one physical object no matter what the two sources called
    # it — a box-refined residual labelled "photo" and a text-prompt "product" with IoU
    # 0.99 must collapse (135). This must outrank every role carve-out below; the
    # carve-outs exist to keep DISTINCT overlapping objects apart, and near-identity is
    # incompatible with distinctness.
    if mask_iou >= float((opts or {}).get("identity_iou", DEFAULTS["identity_iou"])):
        return True
    # Sibling product cutouts (box + earplugs) must stay separate unless masks agree
    # almost completely — same-family alone is not enough to collapse them.
    if (
        ra in _PRODUCT_INSTANCE_ROLES
        and rb in _PRODUCT_INSTANCE_ROLES
        and mask_iou < 0.85
    ):
        return False
    # Two comparison photo panels / portraits: keep separate unless nearly identical.
    if (
        ra in _COMPARISON_PANEL_ROLES
        and rb in _COMPARISON_PANEL_ROLES
        and mask_iou < 0.85
    ):
        return False
    # Product/package over a photo panel (Wavy tube across section seams) must never
    # collapse into the panel raster — keep a discrete high-z cutout.
    if (
        (ra in _PRODUCT_INSTANCE_ROLES and rb in _PHOTO_PANEL_ROLES)
        or (rb in _PRODUCT_INSTANCE_ROLES and ra in _PHOTO_PANEL_ROLES)
    ):
        return False
    if fa == fb or "unknown" in (fa, fb):
        return True
    # Residual-CC kinds are solidity/edge-density guesses, not semantics; when one side of
    # the pair is heuristic, geometry alone decides and the model label wins the cluster.
    if (opts or {}).get("residual_family_advisory", True) and (
        _heuristic_label(a) or _heuristic_label(b)
    ):
        return True
    # Very strong mask agreement beats a weak source label (e.g. residual shape vs SAM badge).
    return mask_iou >= 0.88


def _score(item: dict, source: str) -> float:
    raw = item.get("score")
    if raw is None:
        raw = item.get("confidence")
    if raw is None and source == "residual":
        # coverage is not confidence; retain a conservative deterministic prior.
        raw = 0.38
    if raw is None and source == "qwen":
        raw = 0.50
    try:
        return max(0.0, min(1.0, float(raw if raw is not None else 0.40)))
    except (TypeError, ValueError):
        return 0.40


def _source_label(obs: dict) -> str:
    mode = obs.get("mode")
    return f"{obs['source']}:{mode}" if mode else obs["source"]


def _priority(obs: dict) -> float:
    source, mode = obs["source"], obs.get("mode")
    if source == "sam3" and mode in ("residual-fallback", "box-refine-fallback"):
        base = 1.15
    elif source == "sam3" and mode == "box-refine":
        base = 4.0
    elif source == "sam3" and mode == "text-prompt":
        base = 3.6
    elif source == "sam3" and mode == "box-refine-small":
        # Padded second-pass refinement of a small residual: model evidence, but the
        # prompt geometry was loose, so a text-prompt whole-object mask outranks it.
        base = 3.4
    elif source == "sam3":
        base = 2.8
    elif source == "qwen":
        base = 2.3
    elif source == "residual":
        base = 1.5
    else:
        base = 1.0
    if obs.get("mask_quality") == "box":
        base -= 0.35
    return base + obs["score"] * 0.2


def _candidate_paths(item: dict, source: str, base_dirs: list[str]) -> list[str]:
    raw = [item.get("mask_path"), item.get("mask_src")]
    if isinstance(item.get("mask"), dict):
        raw.append(item["mask"].get("src"))
    if source == "qwen":
        raw.extend([item.get("png"), item.get("src")])
    if source == "residual" and item.get("id"):
        raw.append(os.path.join("elements", f"{item['id']}.png"))
    paths = []
    for value in raw:
        if not value:
            continue
        value = str(value)
        options = [value] if os.path.isabs(value) else [os.path.join(b, value) for b in base_dirs]
        for path in options:
            norm = os.path.abspath(os.path.expanduser(path))
            if norm not in paths:
                paths.append(norm)
    return paths


def _array(value):
    np = _np()
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _place_local_mask(arr, box: dict, width: int, height: int):
    np = _np()
    from PIL import Image

    b = _clip_box(box, width, height)
    full = np.zeros((height, width), dtype=bool)
    if not b["w"] or not b["h"]:
        return full
    if arr.shape != (b["h"], b["w"]):
        arr = np.asarray(
            Image.fromarray((arr > 0).astype(np.uint8) * 255).resize(
                (b["w"], b["h"]), Image.Resampling.NEAREST
            )
        )
    full[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]] = arr > 0
    return full


def _materialize_mask(item: dict, source: str, canvas: dict, base_dirs: list[str]):
    np = _np()
    width, height = int(canvas["w"]), int(canvas["h"])
    raw = _array(item.get("_mask"))
    if raw is not None and raw.size:
        raw = np.squeeze(raw)
        if raw.shape == (height, width):
            return raw > 0, "mask"
        if raw.ndim == 2:
            return _place_local_mask(raw, item.get("box") or {}, width, height), "mask"

    for path in _candidate_paths(item, source, base_dirs):
        if not os.path.exists(path):
            continue
        try:
            from PIL import Image

            with Image.open(path) as im:
                if source == "qwen":
                    if "A" not in im.getbands():
                        continue
                    arr = np.asarray(im.convert("RGBA"))[:, :, 3]
                else:
                    arr = np.asarray(im.convert("L"))
            if arr.shape == (height, width):
                return arr > 8, "mask"
            if source == "qwen":
                arr = np.asarray(
                    Image.fromarray(arr.astype(np.uint8)).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )
                return arr > 8, "mask-rescaled"
            return _place_local_mask(arr, item.get("box") or {}, width, height), "mask"
        except Exception:
            continue
    return _rect_mask(item.get("box") or {}, width, height), "box"


def _obs_descriptor(obs: dict) -> dict:
    prov = obs.get("raw_provenance") or {}
    out = {
        "key": obs["key"],
        "source": obs["source"],
        "id": obs["source_id"],
        "role": obs["role"],
        "score": round(obs["score"], 4),
        "mask_quality": obs.get("mask_quality"),
    }
    for key in ("mode", "prompt", "residual_id", "input_box", "model_box"):
        value = obs.get(key, prov.get(key))
        if value is not None:
            out[key] = value
    asset = obs.get("asset")
    if asset:
        out["asset"] = asset
    return out


def _normalize(item: dict, source: str, canvas: dict, base_dirs: list[str], ordinal: int):
    if not isinstance(item, dict) or not _valid_box(item.get("box")):
        return None
    prov = item.get("provenance") or {}
    source_id = str(item.get("id") or f"{source}-{ordinal}")
    role = _role(item, source, canvas)
    mask, quality = _materialize_mask(item, source, canvas, base_dirs)
    if int(mask.sum()) <= 0:
        return None
    box = _mask_box(mask)
    mode = prov.get("mode")
    actual_source = source
    if source == "sam3" and str(item.get("source", "")).startswith("residual"):
        # It still belongs to the SAM observation stream, but should not outrank real SAM.
        mode = mode or "residual-fallback"
    structural = {
        key: item.get(key, prov.get(key))
        for key in _STRUCTURAL_FIELDS
        if item.get(key, prov.get(key)) not in (None, "")
    }
    return {
        "key": f"{source}:{source_id}",
        "source": actual_source,
        "source_id": source_id,
        "role": role,
        "kind": _kind(role, item.get("kind")),
        "score": _score(item, source),
        "box": box,
        "mask": mask,
        "mask_quality": quality,
        "mode": mode,
        "prompt": prov.get("prompt"),
        "raw_provenance": prov,
        "asset": item.get("png") or item.get("src") or item.get("asset_src"),
        "structural": structural,
        # Detector-provided metadata (overlay corner_radius / fill / text_ids, …)
        # rides along so the canonical element can keep it (overlay-cv proposals).
        "meta": dict(item.get("meta") or {}) or None,
    }


def _duplicate(a: dict, b: dict, opts: dict) -> tuple[bool, dict]:
    metrics = _mask_metrics(a["mask"], b["mask"])
    nested = (
        metrics["containment"] >= opts["containment"]
        and metrics["area_ratio"] < opts["similar_area_ratio"]
    )
    if nested:
        return False, metrics
    mask_duplicate = metrics["iou"] >= opts["mask_iou"]
    containment_duplicate = (
        metrics["containment"] >= opts["containment"]
        and metrics["area_ratio"] >= opts["similar_area_ratio"]
    )
    bbox_duplicate = (
        a["mask_quality"] == "box"
        and b["mask_quality"] == "box"
        and _bbox_iou(a["box"], b["box"]) >= opts["bbox_iou"]
    )
    compatible = _semantic_compatible(a, b, metrics["iou"], opts)
    return compatible and (mask_duplicate or containment_duplicate or bbox_duplicate), metrics


_REFINE_LINK_MODES = ("box-refine", "box-refine-small")


def _linked_refine_cluster(clusters: list, obs: dict, opts: dict):
    """Find the cluster holding the SAM mask that was box-refined FROM this residual.

    ``sam3_detect`` prompts SAM with every residual box and records ``residual_id`` in the
    accepted refinement's provenance.  That pair is one object by construction, yet a sparse
    residual CC (anti-aliased ring/strokes) against the solid SAM mask often lands in the
    IoU band that the nested rule keeps separate — shipping the same element twice.
    Provenance identity plus a weak geometric sanity floor merges them.
    """
    if obs["source"] != "residual":
        return None
    target = str(obs["source_id"])
    for cluster in clusters:
        for member in cluster["members"]:
            if member["source"] != "sam3" or member.get("mode") not in _REFINE_LINK_MODES:
                continue
            prov = member.get("raw_provenance") or {}
            if prov.get("residual_id") is None or str(prov["residual_id"]) != target:
                continue
            metrics = _mask_metrics(obs["mask"], member["mask"])
            if (
                metrics["containment"] >= float(opts["link_min_containment"])
                or metrics["iou"] >= float(opts["mask_iou"])
            ):
                return cluster, metrics
    return None


def _dilate(mask, px: int):
    """Cheap 4-neighbourhood binary dilation via numpy shifts (no scipy dependency)."""
    if px <= 0:
        return mask
    out = mask.copy()
    for _ in range(int(px)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _containment_in(child_mask, parent_mask) -> float:
    """Fraction of the child mask covered by the (possibly dilated) parent mask."""
    child_area = int(child_mask.sum())
    if not child_area:
        return 0.0
    return int((child_mask & parent_mask).sum()) / child_area


def _is_sliver(box: dict, opts: dict) -> bool:
    w, h = int(box.get("w", 0)), int(box.get("h", 0))
    if w <= 0 or h <= 0:
        return False
    if min(w, h) > int(opts.get("sliver_max_thickness", 10)):
        return False
    return max(w, h) / max(1, min(w, h)) >= float(opts.get("sliver_min_aspect", 4.0))


def _absorb_fragment_clusters(clusters: list, opts: dict) -> list:
    """Collapse fragment clusters into the cluster that visually contains them.

    Two evidence-backed fragment classes (runs/golden-optimized-check/009):

    * same-role/kind masks nested inside a bigger mask that is NOT a meaningful parent —
      glyph pieces of one icon detected alongside the whole icon;
    * thin fill slivers inside a model-masked button-like shell — residual CCs of the
      button plate split apart by OCR text dilation (the white strips around 'Volgend').

    The containing cluster's winner mask is preserved untouched, so downstream corner
    radius inference (reconstruct._corner_radius) keeps a clean, tight shape mask.
    Meaningful nesting (icon in a button, badge on a photo) is skipped here and recorded
    by the parent-link pass instead.
    """
    absorb_enabled = bool(opts.get("absorb_fragments", True))
    sliver_enabled = bool(opts.get("sliver_absorb", True))
    if not (absorb_enabled or sliver_enabled) or len(clusters) < 2:
        return clusters
    dilate_px = int(opts.get("containment_dilate_px", 0))
    sliver_roles = {normalized_role(r) for r in (opts.get("sliver_parent_roles") or ())}
    areas = [int(c["winner"]["mask"].sum()) for c in clusters]
    order = sorted(range(len(clusters)), key=lambda i: areas[i])
    alive = [True] * len(clusters)
    dilated = {}
    for child_i in order:
        child = clusters[child_i]
        cw = child["winner"]
        best = None
        for parent_i in order:
            if parent_i == child_i or not alive[parent_i] or areas[parent_i] <= areas[child_i]:
                continue
            pw = clusters[parent_i]["winner"]
            if parent_i not in dilated:
                dilated[parent_i] = _dilate(pw["mask"], dilate_px)
            containment = _containment_in(cw["mask"], dilated[parent_i])
            same_role = (
                normalized_role(pw["role"]) == normalized_role(cw["role"])
                and pw["kind"] == cw["kind"]
            )
            fragment = (
                absorb_enabled
                and same_role
                and not _meaningful_parent(pw, cw)
                and containment >= float(opts["absorb_containment"])
            )
            sliver = (
                sliver_enabled
                and _is_sliver(cw["box"], opts)
                and normalized_role(pw["role"]) in sliver_roles
                and pw["source"] == "sam3"
                and pw.get("mask_quality") == "mask"
                and containment >= float(opts["sliver_containment"])
            )
            if not (fragment or sliver):
                continue
            reason = "absorbed-fragment" if fragment else "absorbed-sliver"
            if best is None or areas[parent_i] < areas[best[0]]:
                best = (parent_i, containment, reason)
        if best is None:
            continue
        parent_i, containment, reason = best
        parent = clusters[parent_i]
        metrics = _mask_metrics(cw["mask"], parent["winner"]["mask"])
        parent["members"].extend(child["members"])
        parent["merges"].extend(child["merges"])
        parent["merges"].append(
            {
                "key": cw["key"],
                "mask_iou": round(metrics["iou"], 4),
                "containment": round(containment, 4),
                "area_ratio": round(metrics["area_ratio"], 4),
                "reason": reason,
            }
        )
        alive[child_i] = False
    return [cluster for keep, cluster in zip(alive, clusters) if keep]


def _has_text_prompt(cluster: dict) -> bool:
    return any(m.get("mode") == "text-prompt" for m in cluster["members"])


def _cluster_max_score(cluster: dict) -> float:
    return max(float(m.get("score", 0) or 0) for m in cluster["members"])


def _promote_product_identity(clusters: list) -> None:
    """Give a cluster its semantic product identity when the winner label is heuristic.

    Box-refine/residual observation roles come from residual-CC heuristics
    (_role_from_residual), not semantics. When such an observation wins a cluster that
    also holds a text-prompt product observation of the same object (near-identical
    masks, merged via identity_iou), the canonical element must ship as a product, not
    as a "photo" (135: the choco bar would otherwise lose its product identity to the
    box-refine observation that seeded the cluster).
    """
    for cluster in clusters:
        winner = cluster["winner"]
        if normalized_role(winner["role"]) in _PRODUCT_INSTANCE_ROLES:
            continue
        if winner.get("mode") == "text-prompt":
            continue  # semantic label already; do not override
        best = None
        for member in cluster["members"]:
            if member.get("mode") != "text-prompt":
                continue
            if normalized_role(member["role"]) not in _PRODUCT_INSTANCE_ROLES:
                continue
            if best is None or member["score"] > best["score"]:
                best = member
        if best is not None:
            winner["role"] = best["role"]
            winner["kind"] = best["kind"]
            cluster["merges"].append(
                {"key": best["key"], "reason": "product-identity-promoted"}
            )


def _suppress_product_shadows(clusters: list, opts: dict) -> list:
    """Absorb no-evidence raster clusters whose pixels product masks already own.

    A cluster with no text-prompt observation of its own, whose mask is mostly covered
    by the union of semantically-detected product masks, is a duplicate observation of
    already-owned pixels (067: one box-refine "photo" strip == the union of three jars;
    135: the residual fragment swarm straddling both bars). Members are absorbed into
    the product cluster with the largest pixel overlap so provenance is preserved.

    Distinct products (product-instance roles), graphic roles (logo/badge/icon on a
    package), and anything with its own text-prompt evidence are never eligible, so
    multi-SKU layouts (067's three jars) keep every real product.
    """
    if not opts.get("product_shadow", True) or len(clusters) < 2:
        return clusters
    np = _np()
    product_idx = [
        i
        for i, c in enumerate(clusters)
        if normalized_role(c["winner"]["role"]) in _PRODUCT_INSTANCE_ROLES
        and _has_text_prompt(c)
    ]
    if not product_idx:
        return clusters
    threshold = float(opts.get("product_shadow_containment", 0.75))
    product_set = set(product_idx)
    alive = [True] * len(clusters)
    for i, cluster in enumerate(clusters):
        if i in product_set:
            continue
        winner = cluster["winner"]
        role = normalized_role(winner["role"])
        if role in _PRODUCT_INSTANCE_ROLES or role in _GRAPHIC_ROLES:
            continue
        if winner["kind"] != "photo-fragment":
            continue
        if _has_text_prompt(cluster):
            continue
        mask = winner["mask"]
        area = int(mask.sum())
        if not area:
            continue
        covered = np.zeros_like(mask)
        best = None
        for pi in product_idx:
            pmask = clusters[pi]["winner"]["mask"]
            inter = int((mask & pmask).sum())
            if inter:
                covered |= mask & pmask
            if best is None or inter > best[0]:
                best = (inter, pi)
        frac = int(covered.sum()) / area
        if frac >= threshold and best is not None and best[0] > 0:
            parent = clusters[best[1]]
            parent["members"].extend(cluster["members"])
            parent["merges"].extend(cluster["merges"])
            parent["merges"].append(
                {
                    "key": winner["key"],
                    "containment": round(frac, 4),
                    "reason": "absorbed-product-shadow",
                }
            )
            alive[i] = False
    return [c for keep, c in zip(alive, clusters) if keep]


def _cluster_product_prompts(cluster: dict) -> set[str]:
    """Normalized text-prompt strings that tagged this cluster as a product instance."""
    prompts = set()
    for member in cluster["members"]:
        if member.get("mode") != "text-prompt":
            continue
        if normalized_role(member.get("role")) not in _PRODUCT_INSTANCE_ROLES:
            continue
        prompt = str(member.get("prompt") or "").strip().lower()
        if prompt:
            prompts.add(prompt)
    return prompts


def _hull_gap_budget(box_a: dict, box_b: dict, opts: dict) -> int:
    """Max mask gap (px) still considered 'one product' for cluster merge."""
    floor = int(opts.get("product_cluster_gap_px", 4))
    frac = float(opts.get("product_cluster_gap_frac", 0.12))
    sides = []
    for box in (box_a, box_b):
        w, h = int(box.get("w", 0) or 0), int(box.get("h", 0) or 0)
        if w > 0 and h > 0:
            sides.append(min(w, h))
    if not sides:
        return max(0, floor)
    return max(floor, int(round(frac * min(sides))))


def _masks_hull_adjacent(a, b, gap_px: int) -> bool:
    """True when masks already touch/overlap or their dilated hulls meet within gap_px."""
    if gap_px < 0:
        return False
    if int((a & b).sum()) > 0:
        return True
    if gap_px == 0:
        return False
    # Dilate the smaller mask only — cheaper, same contact test.
    aa, ab = int(a.sum()), int(b.sum())
    if aa <= ab:
        return bool(((_dilate(a, gap_px) & b).sum()))
    return bool(((_dilate(b, gap_px) & a).sum()))


def _merge_product_clusters(clusters: list, opts: dict) -> list:
    """Collapse adjacent same-prompt product parts into one owner (135 dual bars / jar+lid).

    Distinct SKUs with larger gaps (067's spaced jars) stay separate because the gap
    budget scales with object size. Only clusters that already carry product-instance
    identity *and* a shared text-prompt string are eligible.
    """
    if not opts.get("product_cluster_merge", True) or len(clusters) < 2:
        return clusters
    min_ratio = float(opts.get("product_cluster_min_area_ratio", 0.08))
    product_idx = [
        i
        for i, c in enumerate(clusters)
        if normalized_role(c["winner"]["role"]) in _PRODUCT_INSTANCE_ROLES
        and _cluster_product_prompts(c)
    ]
    if len(product_idx) < 2:
        return clusters

    parent = list(range(len(clusters)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    box_iou_min = float(opts.get("product_cluster_box_iou", 0.22))
    for ai, a_i in enumerate(product_idx):
        a = clusters[a_i]
        a_prompts = _cluster_product_prompts(a)
        a_area = max(1, int(a["winner"]["mask"].sum()))
        for b_i in product_idx[ai + 1 :]:
            b = clusters[b_i]
            shared = a_prompts & _cluster_product_prompts(b)
            if not shared:
                continue
            b_area = max(1, int(b["winner"]["mask"].sum()))
            ratio = min(a_area, b_area) / max(a_area, b_area)
            if ratio < min_ratio:
                continue
            gap = _hull_gap_budget(a["winner"]["box"], b["winner"]["box"], opts)
            adjacent = _masks_hull_adjacent(
                a["winner"]["mask"], b["winner"]["mask"], gap,
            )
            if not adjacent:
                # Masks may be perforated / gapped while the painted boxes still overlap
                # as one packshot (135). Spaced multi-SKU layouts (067) stay near-zero IoU.
                ab, bb = a["winner"]["box"], b["winner"]["box"]
                ax0, ay0 = float(ab["x"]), float(ab["y"])
                ax1, ay1 = ax0 + float(ab["w"]), ay0 + float(ab["h"])
                bx0, by0 = float(bb["x"]), float(bb["y"])
                bx1, by1 = bx0 + float(bb["w"]), by0 + float(bb["h"])
                iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
                ih = max(0.0, min(ay1, by1) - max(ay0, by0))
                inter = iw * ih
                union_a = (
                    float(ab["w"]) * float(ab["h"])
                    + float(bb["w"]) * float(bb["h"])
                    - inter
                )
                iou = inter / max(1.0, union_a)
                if iou < box_iou_min:
                    continue
            union(a_i, b_i)

    groups: dict[int, list[int]] = {}
    for i in product_idx:
        groups.setdefault(find(i), []).append(i)
    absorb_into: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # Keep the highest-priority winner; union every sibling mask into it.
        root = max(members, key=lambda i: (_priority(clusters[i]["winner"]),
                                           int(clusters[i]["winner"]["mask"].sum())))
        for i in members:
            if i != root:
                absorb_into[i] = root

    if not absorb_into:
        return clusters

    alive = [True] * len(clusters)
    for child_i, root_i in absorb_into.items():
        # Path-compress through any chain (A→B, B→C) created by the grouping above.
        while root_i in absorb_into:
            root_i = absorb_into[root_i]
        child = clusters[child_i]
        parent_c = clusters[root_i]
        shared = sorted(_cluster_product_prompts(parent_c) & _cluster_product_prompts(child))
        parent_c["winner"]["mask"] = parent_c["winner"]["mask"] | child["winner"]["mask"]
        parent_c["winner"]["box"] = _mask_box(parent_c["winner"]["mask"])
        parent_c["members"].extend(child["members"])
        parent_c["merges"].extend(child["merges"])
        metrics = _mask_metrics(child["winner"]["mask"], parent_c["winner"]["mask"])
        parent_c["merges"].append(
            {
                "key": child["winner"]["key"],
                "mask_iou": round(metrics["iou"], 4),
                "containment": round(metrics["containment"], 4),
                "area_ratio": round(metrics["area_ratio"], 4),
                "reason": "product-cluster-hull-merge",
                "shared_prompts": shared,
            }
        )
        alive[child_i] = False
    return [c for keep, c in zip(alive, clusters) if keep]


def _fill_mask_holes(mask):
    """Fill interior background components not reachable from the silhouette border."""
    np = _np()
    box = _mask_box(mask)
    if not box["w"] or not box["h"]:
        return mask
    # Work on a 1px-padded crop so exterior background always touches the crop border.
    y0, x0, bh, bw = box["y"], box["x"], box["h"], box["w"]
    crop = np.zeros((bh + 2, bw + 2), dtype=bool)
    crop[1:-1, 1:-1] = mask[y0 : y0 + bh, x0 : x0 + bw]
    h, w = crop.shape
    reachable = np.zeros_like(crop, dtype=bool)
    stack = [(0, 0)]
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or reachable[y, x] or crop[y, x]:
            continue
        reachable[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    filled = crop | (~crop & ~reachable)
    out = mask.copy()
    out[y0 : y0 + bh, x0 : x0 + bw] = filled[1:-1, 1:-1]
    return out


def _seal_product_masks(clusters: list, opts: dict) -> None:
    """Close thin notches and fill Swiss-cheese holes on product winner masks."""
    if not opts.get("seal_product_holes", True):
        return
    close_px = int(opts.get("product_hole_close_px", 3))
    for cluster in clusters:
        winner = cluster["winner"]
        if normalized_role(winner["role"]) not in _PRODUCT_INSTANCE_ROLES:
            continue
        if winner.get("mask_quality") == "box":
            continue  # rectangular fallback — nothing to seal
        sealed = winner["mask"]
        if close_px > 0:
            # Morphological close ≈ dilate then erode via dual dilate on complement.
            sealed = _dilate(sealed, close_px)
            # Erode: dilate the complement, invert.
            sealed = ~_dilate(~sealed, close_px)
        sealed = _fill_mask_holes(sealed)
        if int(sealed.sum()) <= 0:
            continue
        winner["mask"] = sealed
        winner["box"] = _mask_box(sealed)


_JUNK_BAND_ROLES = frozenset({"photo", "image", "object", "photo-fragment"})


def _suppress_junk_bands(clusters: list, opts: dict, canvas: dict) -> tuple[list, list]:
    """Drop full-bleed, low-score photo bands that have no semantic evidence.

    The residual detector re-emits leftover background as full-width "photo" bands;
    box-refine then rubber-stamps them at scores below the text-prompt bar. Such a band
    ships as an opaque group plate that can z-order ABOVE a real product and decapitate
    it (013: the y491-989 gradient band over the gruns bag), or as an empty junk group
    (021). The background plate already owns these pixels — dropping the cluster leaves
    them exactly where they belong. Real full-bleed photos keep their element: any
    text-prompt observation or a score at/above junk_band_max_score exempts the cluster.
    """
    if not opts.get("suppress_junk_bands", True):
        return clusters, []
    width, height = int(canvas["w"]), int(canvas["h"])
    min_width = float(opts.get("junk_band_min_width_frac", 0.92)) * width
    min_area = float(opts.get("junk_band_min_coverage", 0.10)) * width * height
    max_score = float(opts.get("junk_band_max_score", 0.50))
    kept, dropped = [], []
    for cluster in clusters:
        winner = cluster["winner"]
        role = normalized_role(winner["role"])
        score = _cluster_max_score(cluster)
        if (
            winner["kind"] == "photo-fragment"
            and role in _JUNK_BAND_ROLES
            and not _has_text_prompt(cluster)
            and float(winner["box"]["w"]) >= min_width
            and int(winner["mask"].sum()) >= min_area
            and score < max_score
        ):
            dropped.append(
                {
                    "key": winner["key"],
                    "role": winner["role"],
                    "box": dict(winner["box"]),
                    "score": round(score, 4),
                    "members": [m["key"] for m in cluster["members"]],
                    "reason": "junk-band-no-semantic-evidence",
                }
            )
        else:
            kept.append(cluster)
    return kept, dropped


#: Brand lettering / lockup roles that read as printed ink when nested in a product.
_PRINTED_ARTWORK_ROLES = frozenset({"logo", "wordmark", "brandmark", "monogram"})
#: On-pack highlight bars / chips SAM often labels as CTA chrome (002 flavor bar).
_PACKAGING_SHELL_ROLES = frozenset({
    "button", "badge", "chip", "pill", "sticker", "shape", "seal", "tag",
})


def _absorb_into_product_parent(
    results: list,
    opts: dict,
    *,
    roles: frozenset,
    min_containment: float,
    max_ratio: float,
    reason: str,
    role_filter=None,
) -> tuple[list, list]:
    """Fold nested child roles into a product parent; record printed_lockup decorations."""
    by_id = {r["id"]: r for r in results}
    absorbed = []
    for r in results:
        role = str(r.get("role") or "").lower()
        if role not in roles:
            continue
        parent = by_id.get(str(r.get("parent_id") or ""))
        rel = next((link for link in (r.get("relationships") or [])
                    if link.get("type") == "nested-in"
                    and link.get("target") == r.get("parent_id")), None)
        if parent is None or rel is None:
            continue
        if str(parent.get("role") or "").lower() not in _PRODUCT_INSTANCE_ROLES:
            continue
        if float(rel.get("containment") or 0.0) < min_containment:
            continue
        if float(rel.get("area_ratio") or 1.0) > max_ratio:
            continue
        if role_filter is not None and not role_filter(r, parent, rel):
            continue
        absorbed.append({
            "id": r["id"], "role": role, "parent": parent["id"],
            "containment": rel.get("containment"), "area_ratio": rel.get("area_ratio"),
            "reason": reason,
        })
        meta = dict(parent.get("meta") or {})
        meta["printed_lockup"] = True
        decorations = list(meta.get("absorbed_decorations") or [])
        decorations.append({"id": r["id"], "role": role, "box": dict(r.get("box") or {})})
        meta["absorbed_decorations"] = decorations
        parent["meta"] = meta
    if not absorbed:
        return results, []
    gone = {a["id"] for a in absorbed}
    kept = [r for r in results if r["id"] not in gone]
    for r in kept:
        if str(r.get("parent_id") or "") in gone:
            r["parent_id"] = by_id[str(r["parent_id"])].get("parent_id")
            r["relationships"] = [link for link in (r.get("relationships") or [])
                                  if not (link.get("type") == "nested-in"
                                          and link.get("target") in gone)]
    return kept, absorbed


def _box_inside_frac(inner: dict, outer: dict) -> float:
    """Fraction of ``inner`` covered by ``outer`` (axis-aligned boxes)."""
    ix0 = float(inner.get("x", 0) or 0)
    iy0 = float(inner.get("y", 0) or 0)
    ix1 = ix0 + float(inner.get("w", 0) or 0)
    iy1 = iy0 + float(inner.get("h", 0) or 0)
    ox0 = float(outer.get("x", 0) or 0)
    oy0 = float(outer.get("y", 0) or 0)
    ox1 = ox0 + float(outer.get("w", 0) or 0)
    oy1 = oy0 + float(outer.get("h", 0) or 0)
    iw = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    ih = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    area = max(0.0, float(inner.get("w", 0) or 0) * float(inner.get("h", 0) or 0))
    return (iw * ih) / area if area > 0 else 0.0


def _absorb_list_icons_into_cards(results: list, opts: dict) -> tuple[list, list]:
    """Fold checklist ✓/✗ chips into their hosting card before peel (066).

    icon_detect emits one chip per row. Peeling those chips out of a white card
    destroys the card raster; merge/reconstruct cannot restore source ink once the
    under-layer is hole-punched. Absorbing here keeps a single card element.
    """
    if not results:
        return results, []
    min_icons = int(opts.get("list_icon_card_min_icons", 2))
    min_inside = float(opts.get("list_icon_card_min_inside", 0.85))
    cards = []
    icons = []
    for item in results:
        role = normalized_role(item.get("role") or (item.get("meta") or {}).get("role"))
        kind = str(item.get("kind") or "").lower()
        box = item.get("box") or {}
        area = float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
        if area <= 0:
            continue
        if role in _LIST_ROW_ICON_ROLES or (
            kind == "icon" and role in _LIST_ROW_ICON_ROLES | {""}
            and (item.get("meta") or {}).get("icon_cv")
        ):
            icons.append(item)
            continue
        if role in _CHECKLIST_CARD_ROLES or kind in {"shape", "card"}:
            cards.append(item)
    if not cards or not icons:
        return results, []
    absorbed = []
    gone = set()
    for card in cards:
        card_box = card.get("box") or {}
        card_area = float(card_box.get("w", 0) or 0) * float(card_box.get("h", 0) or 0)
        if card_area <= 0:
            continue
        nested = []
        for icon in icons:
            if icon.get("id") in gone:
                continue
            ibox = icon.get("box") or {}
            iarea = float(ibox.get("w", 0) or 0) * float(ibox.get("h", 0) or 0)
            if iarea <= 0 or iarea / card_area > 0.20 or card_area < iarea * 8:
                continue
            if _box_inside_frac(ibox, card_box) >= min_inside:
                nested.append(icon)
        if len(nested) < min_icons:
            continue
        meta = card.setdefault("meta", {})
        # Icons stay in the card raster (avoid hole-punch), but checklist *copy*
        # must remain editable TEXT — never mark the card as a chrome text bake.
        meta["checklist_editable"] = True
        meta.pop("checklist_raster_chip", None)
        meta.setdefault("absorbed_list_icons", [])
        for icon in nested:
            gone.add(icon.get("id"))
            meta["absorbed_list_icons"].append(icon.get("id"))
            absorbed.append({
                "id": icon.get("id"),
                "host": card.get("id"),
                "reason": "list-icon-absorbed-into-card",
                "role": icon.get("role"),
            })
    if not gone:
        return results, []
    kept = [r for r in results if r.get("id") not in gone]
    return kept, absorbed


def _absorb_printed_artwork(results: list, opts: dict) -> tuple[list, list]:
    """Fold printed-on-product artwork into its parent product (013 grüns bag).

    A canonical element with a brand-artwork role (``_PRINTED_ARTWORK_ROLES``) whose
    mask is ≥ ``printed_artwork_min_containment`` inside a *product* parent and small
    relative to it is the product's own label ink — SAM's "logo" prompt firing on the
    packaging print.  It is dropped as a separate element (the parent raster already
    carries those pixels; peel's ink discipline never punches them out) and recorded
    on the parent as ``meta.printed_lockup`` + ``meta.absorbed_decorations`` so the
    peel gate can lift the product off the plate.  Logos over non-product surfaces
    (floating brand marks on photos/backgrounds) are untouched — those are genuine
    overlay layers.
    """
    return _absorb_into_product_parent(
        results,
        opts,
        roles=_PRINTED_ARTWORK_ROLES,
        min_containment=float(opts.get("printed_artwork_min_containment", 0.95)),
        max_ratio=float(opts.get("printed_artwork_max_area_ratio", 0.5)),
        reason="printed-on-product-artwork",
    )


def _absorb_packaging_shells(results: list, opts: dict) -> tuple[list, list]:
    """Fold on-pack SAM 'button'/chip false positives into the product (002 flavor bar)."""
    if not opts.get("absorb_packaging_shells", True):
        return results, []
    min_aspect = float(opts.get("packaging_shell_min_aspect", 3.2))

    def _eligible(child: dict, parent: dict, rel: dict) -> bool:
        box = child.get("box") or {}
        w = float(box.get("w", 0) or 0)
        h = float(box.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            return False
        aspect = max(w, h) / max(1.0, min(w, h))
        role = str(child.get("role") or "").lower()
        # Wide/thin flavor bars and seals are packaging; square-ish badges/stickers too.
        # Tall CTA buttons (aspect < threshold) on a pack face stay separate.
        if role in {"badge", "chip", "pill", "sticker", "seal", "tag"}:
            return True
        if role in {"button", "shape"} and aspect >= min_aspect:
            return True
        return False

    return _absorb_into_product_parent(
        results,
        opts,
        roles=_PACKAGING_SHELL_ROLES,
        min_containment=float(opts.get("packaging_shell_min_containment", 0.92)),
        max_ratio=float(opts.get("packaging_shell_max_area_ratio", 0.18)),
        reason="printed-on-product-shell",
        role_filter=_eligible,
    )


def _absorb_product_decorations(results: list, opts: dict) -> tuple[list, list]:
    """Fold EVERY remaining non-product child that rides inside a product cutout.

    The invariant made total: one SKU = one alpha, so nothing inside a product's mask
    ships as a separate design element.  ``_absorb_printed_artwork`` and
    ``_absorb_packaging_shells`` cover the two special cases SAM most often mislabels
    (brand lockups, on-pack flavor bars); this pass runs after them and sweeps up
    whatever they left — checkmark / cross list glyphs, generic icons, small printed
    photo scraps, and shapes too narrow for the flavor-bar aspect gate (091 / 094 /
    135).  Each such child is the product's own label ink; emitting it separately makes
    reconstruct hole-punch it out of the packaging and hands peel an inside-the-product
    occlusion pair — the exact "peeling inside products" the mandate forbids.

    A child is absorbed into a product parent when either:
      * it is nested (its ``nested-in`` link, i.e. mask containment ≥ nested_containment),
        or
      * it has no product link but its BOX sits ≥ ``product_decoration_box_containment``
        inside a product — a label that straddles the product edge (containment dips
        under the nesting threshold) must still ride the cutout, never punch the bag.

    Distinct SKUs (product-instance roles) and full-bleed backgrounds are never
    decorations; a child larger than ``product_decoration_max_area_ratio`` of the
    product is a container, not printed ink, and is left alone.  Iterated to a fixed
    point so a glyph nested in a panel that is itself nested in the product folds in
    once its intermediate parent is absorbed (135: ✓/✗ chips under a nutrition panel
    under the bar).
    """
    if not opts.get("absorb_product_decorations", True):
        return results, []
    min_cont = float(opts.get("product_decoration_min_containment", 0.80))
    box_cont = float(opts.get("product_decoration_box_containment", 0.80))
    max_ratio = float(opts.get("product_decoration_max_area_ratio", 0.60))
    absorbed: list = []
    for _ in range(8):  # fixed-point: absorbing a panel reparents its glyphs to the product
        by_id = {r["id"]: r for r in results}
        product_ids = {
            r["id"] for r in results
            if str(r.get("role") or "").lower() in _PRODUCT_INSTANCE_ROLES
        }
        if not product_ids:
            break
        gone: dict = {}
        for r in results:
            if r["id"] in product_ids:
                continue
            role = str(r.get("role") or "").lower()
            if role in _PRODUCT_INSTANCE_ROLES or role == "background":
                continue
            target = None
            containment = None
            area_ratio = None
            parent = by_id.get(str(r.get("parent_id") or ""))
            if parent is not None and parent["id"] in product_ids:
                rel = next((link for link in (r.get("relationships") or [])
                            if link.get("type") == "nested-in"
                            and link.get("target") == parent["id"]), None)
                cont = (float(rel["containment"]) if rel and rel.get("containment") is not None
                        else _box_inside_frac(r.get("box") or {}, parent.get("box") or {}))
                if cont >= min_cont:
                    target, containment = parent, cont
                    if rel and rel.get("area_ratio") is not None:
                        area_ratio = float(rel["area_ratio"])
            if target is None:
                # Edge-straddling decoration with no product link: fold on box containment.
                best = None
                for pid in product_ids:
                    frac = _box_inside_frac(r.get("box") or {}, by_id[pid].get("box") or {})
                    if frac >= box_cont and (best is None or frac > best[1]):
                        best = (by_id[pid], frac)
                if best is not None:
                    target, containment = best[0], best[1]
            if target is None:
                continue
            if area_ratio is None:
                pbox = target.get("box") or {}
                cbox = r.get("box") or {}
                parea = float(pbox.get("w", 0) or 0) * float(pbox.get("h", 0) or 0)
                carea = float(cbox.get("w", 0) or 0) * float(cbox.get("h", 0) or 0)
                area_ratio = (carea / parea) if parea > 0 else 1.0
            if area_ratio > max_ratio:
                continue  # a container the size of the product, not printed ink
            gone[r["id"]] = (target, role, containment, area_ratio)
        if not gone:
            break
        for cid, (target, role, containment, area_ratio) in gone.items():
            child = by_id[cid]
            meta = dict(target.get("meta") or {})
            meta["printed_lockup"] = True
            decorations = list(meta.get("absorbed_decorations") or [])
            decorations.append({"id": cid, "role": role, "box": dict(child.get("box") or {})})
            meta["absorbed_decorations"] = decorations
            target["meta"] = meta
            absorbed.append({
                "id": cid, "role": role, "parent": target["id"],
                "containment": round(float(containment), 4) if containment is not None else None,
                "area_ratio": round(float(area_ratio), 4),
                "reason": "printed-on-product-decoration",
            })
        goneset = set(gone)
        kept = [r for r in results if r["id"] not in goneset]
        for r in kept:
            pid = str(r.get("parent_id") or "")
            if pid in goneset:
                r["parent_id"] = by_id[pid].get("parent_id")
                r["relationships"] = [link for link in (r.get("relationships") or [])
                                      if not (link.get("type") == "nested-in"
                                              and link.get("target") in goneset)]
        results = kept
    return results, absorbed


def _infer_canvas(sam3, residual, qwen) -> dict:
    if isinstance(sam3, dict) and isinstance(sam3.get("source"), dict):
        src = sam3["source"]
        if src.get("w") and src.get("h"):
            return {"w": int(src["w"]), "h": int(src["h"])}
    max_x = max_y = 1
    for item in _as_list(sam3) + _as_list(residual) + _as_list(qwen):
        box = item.get("box") or {}
        max_x = max(max_x, int(float(box.get("x", 0)) + float(box.get("w", 0))))
        max_y = max(max_y, int(float(box.get("y", 0)) + float(box.get("h", 0))))
    return {"w": max_x, "h": max_y}


def _meaningful_parent(parent: dict, child: dict) -> bool:
    # Shape/card/button-style containers, and raster-role containers (a photo, product
    # shot, or person that visually contains a nested logo/badge/icon/product), are both
    # valid parents. Without the raster branch, a photo containing a nested product mask
    # could pass the containment/area-ratio gate above but never actually get linked,
    # shipping as two unrelated overlapping top-level elements.
    parent_role = normalized_role(parent["role"])
    child_role = normalized_role(child["role"])
    # Seam-straddling product tubes (Wavy across Before/After) stay top-level vs photo
    # panels. Fully inset products inside a photo may still nest for ownership.
    if child_role in _PRODUCT_INSTANCE_ROLES and parent["kind"] in {
        "photo-fragment", "photo", "image",
    }:
        if parent_role not in _CONTAINER_ROLES:
            metrics = _mask_metrics(child["mask"], parent["mask"])
            if metrics["containment"] < 0.85:
                return False
    if parent_role in _CONTAINER_ROLES or parent_role in _RASTER_ROLES or parent["kind"] == "photo-fragment":
        return parent_role != child_role or parent["kind"] != child["kind"]
    return parent["kind"] == "shape" and child["kind"] != "shape"


def _write_mask(mask, box: dict, path: str) -> None:
    np = _np()
    from PIL import Image

    crop = mask[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(crop.astype(np.uint8) * 255).save(path)


# ── star-rating rows → one pixel-exact chip PER STAR ──────────────────────────────────
# sam3 detects "star rating row" / "trustpilot stars" as ONE element (role "rating"), so
# the whole ★★★★★ run shipped as a single raster blob: you could not restyle 4-of-5, and
# a partial/half star was frozen into the strip. The user's ask is explicit — "crop out
# each star and rasterize it, ensuring the accompanying text remains clear" — so a rating
# row splits into its connected components, each becoming its own alpha cutout chip.
# Raster-first, never vectorized: a traced star is exactly the "random lines / overly
# complex graphics" class (vectorize._DEFAULT_SCORE_MIN carries a "star" tier that this
# keeps us out of). The adjacent review copy is a separate text candidate and is never
# touched here.
#
# Fails CLOSED: anything that does not look like a clean row of similar, separated glyphs
# (touching stars, a fused strip, one blob, wild size spread) keeps the single original
# chip. A wrong split would shatter real artwork into debris — worse than one honest chip.
_RATING_ROLES = frozenset({"rating", "star_rating", "stars", "star-rating"})
_RATING_SPLIT_DEFAULTS = {
    "enabled": True,
    "min_stars": 2,
    "max_stars": 10,
    # A star is roughly as tall as it is wide; components far from the row height are
    # specks/AA debris or a fused pair, not glyphs.
    "min_component_frac": 0.25,   # of the row's median component area
    "max_component_frac": 2.5,
    "min_component_px": 12,
    # Every component must sit on the row's baseline band (a rating row is horizontal).
    "max_center_dev_frac": 0.35,  # of row height
}


def _rating_split_opts(opts) -> dict:
    merged = dict(_RATING_SPLIT_DEFAULTS)
    supplied = (opts or {}).get("rating_split")
    if isinstance(supplied, dict):
        merged.update(supplied)
    return merged


def _split_rating_clusters(clusters: list, opts) -> list:
    """Replace each star-rating cluster with one cluster per star glyph (fails closed)."""
    rcfg = _rating_split_opts(opts)
    if not bool(rcfg.get("enabled", True)):
        return clusters
    np = _np()
    try:
        import cv2
    except Exception:
        return clusters  # no CC backend: keep the honest single chip
    out = []
    for cluster in clusters:
        winner = cluster.get("winner") or {}
        role = str(winner.get("role") or "").lower().replace("-", "_")
        mask = winner.get("mask")
        if role not in _RATING_ROLES or mask is None:
            out.append(cluster)
            continue
        try:
            binary = (np.asarray(mask) > 0).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        except Exception:
            out.append(cluster)
            continue
        rows = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(rcfg["min_component_px"]):
                continue  # AA speck
            rows.append((label, area))
        if not (int(rcfg["min_stars"]) <= len(rows) <= int(rcfg["max_stars"])):
            out.append(cluster)
            continue
        areas = np.asarray([area for _, area in rows], dtype=np.float32)
        median = float(np.median(areas))
        if median <= 0:
            out.append(cluster)
            continue
        # Uniformity: real stars in one row are near-identical glyphs.
        if not np.all((areas >= float(rcfg["min_component_frac"]) * median)
                      & (areas <= float(rcfg["max_component_frac"]) * median)):
            out.append(cluster)
            continue
        # Horizontality: every glyph shares the row's vertical centre.
        row_box = _mask_box(binary)
        centres_y = np.asarray([centroids[label][1] for label, _ in rows], dtype=np.float32)
        row_centre = row_box["y"] + row_box["h"] / 2.0
        if row_box["h"] <= 0 or np.any(
            np.abs(centres_y - row_centre) > float(rcfg["max_center_dev_frac"]) * row_box["h"]
        ):
            out.append(cluster)
            continue
        # Ordered left→right so ids/z read like the row does.
        ordered = sorted(rows, key=lambda item: float(centroids[item[0]][0]))
        for index, (label, _area) in enumerate(ordered):
            star_mask = labels == label
            star_winner = dict(winner)
            star_winner["mask"] = star_mask
            star_winner["box"] = _mask_box(star_mask)
            star_winner["role"] = "star"
            star_winner["kind"] = "icon"
            star_meta = dict(winner.get("meta") or {})
            star_meta.update({
                "rating_star_index": index,
                "rating_star_count": len(ordered),
                "rating_split_from": str(winner.get("key") or winner.get("id") or "rating"),
                # Raster-first contract: an exact alpha cutout, never a traced path.
                "icon_chip": True,
                "intentional_raster_cluster": True,
            })
            star_winner["meta"] = star_meta
            out.append({
                "winner": star_winner,
                "members": [star_winner],
                "merges": list(cluster.get("merges") or []),
            })
    return out


def fuse(
    sam3=None,
    residual=None,
    qwen=None,
    canvas: Optional[dict] = None,
    cfg: Optional[dict] = None,
    run_dir: Optional[str] = None,
) -> list[dict]:
    """Return one canonical element per visible instance.

    Mask duplicates collapse across sources.  High-containment observations with materially
    different areas are retained, and container relationships are recorded after canonical
    IDs are assigned.
    """
    cfg = cfg or {}
    opts = dict(DEFAULTS)
    opts.update(cfg.get("element_fusion") or {})
    canvas = dict(canvas or _infer_canvas(sam3, residual, qwen))
    canvas = {"w": max(1, int(canvas["w"])), "h": max(1, int(canvas["h"]))}

    base_dirs = []
    for value in (run_dir, cfg.get("base_dir"), os.getcwd()):
        if value:
            path = os.path.abspath(os.path.expanduser(str(value)))
            if path not in base_dirs:
                base_dirs.append(path)

    observations = []
    for source, values in (
        ("sam3", _as_list(sam3)),
        ("residual", _as_list(residual)),
        ("qwen", _as_list(qwen)),
    ):
        for ordinal, item in enumerate(values):
            obs = _normalize(item, source, canvas, base_dirs, ordinal)
            if obs:
                observations.append(obs)

    # Residual-fallback SAM observations must never replace the deterministic residual stream.
    # If SAM emitted nothing for a residual id, the residual observation still survives above.

    # Identity dedup happens before geometric clustering. If the same observation was supplied
    # twice, only its strongest copy can enter a canonical element.
    unique = {}
    for obs in observations:
        old = unique.get(obs["key"])
        if old is None or _priority(obs) > _priority(old):
            unique[obs["key"]] = obs
    observations = sorted(
        unique.values(),
        key=lambda o: (-_priority(o), o["box"]["y"], o["box"]["x"], o["key"]),
    )

    clusters = []
    for obs in observations:
        match = None
        match_score = -1.0
        match_metrics = None
        match_reason = None
        # Provenance identity beats geometry: a residual observation always joins the
        # cluster holding the SAM mask that was box-refined from it.
        if opts.get("link_residual_refine", True):
            linked = _linked_refine_cluster(clusters, obs, opts)
            if linked is not None:
                match, match_metrics = linked
                match_reason = "residual-refine-link"
        if match is None:
            for cluster in clusters:
                duplicate, metrics = _duplicate(obs, cluster["winner"], opts)
                if duplicate and metrics["iou"] > match_score:
                    match, match_score, match_metrics = cluster, metrics["iou"], metrics
        if match is None:
            clusters.append({"winner": obs, "members": [obs], "merges": []})
        else:
            match["members"].append(obs)
            merge_record = {
                "key": obs["key"],
                "mask_iou": round(match_metrics["iou"], 4),
                "containment": round(match_metrics["containment"], 4),
                "area_ratio": round(match_metrics["area_ratio"], 4),
            }
            if match_reason:
                merge_record["reason"] = match_reason
            match["merges"].append(merge_record)

    # Fragment absorption: contained same-role pieces and button-fill slivers collapse
    # into the containing cluster before canonical IDs are assigned.
    clusters = _absorb_fragment_clusters(clusters, opts)

    # Canonical product geometry: a heuristic-labelled winner takes the product identity
    # of a merged text-prompt observation; full-bleed no-evidence background bands drop;
    # duplicate raster coverage of product pixels is absorbed into the products;
    # adjacent same-prompt parts (jar+lid / dual bars) collapse to one owner; product
    # alphas are sealed so Swiss-cheese SAM holes do not survive into peel/cutouts.
    _promote_product_identity(clusters)
    # A ★★★★★ row is N glyphs, not one object: split before IDs are assigned so each star
    # earns its own canonical id, box, alpha mask and chip asset.
    clusters = _split_rating_clusters(clusters, opts)
    clusters, suppressed_bands = _suppress_junk_bands(clusters, opts, canvas)
    clusters = _suppress_product_shadows(clusters, opts)
    clusters = _merge_product_clusters(clusters, opts)
    _seal_product_masks(clusters, opts)

    # Stable IDs are based on final geometry, not model/source order.
    clusters.sort(
        key=lambda c: (
            c["winner"]["box"]["y"],
            c["winner"]["box"]["x"],
            -int(c["winner"]["mask"].sum()),
            c["winner"]["role"],
        )
    )
    prefix = str(opts["canonical_prefix"])
    results = []
    for index, cluster in enumerate(clusters):
        winner = cluster["winner"]
        cid = f"{prefix}{index:03d}"
        mask = winner["mask"]
        box = _mask_box(mask)
        descriptors = []
        seen = set()
        for member in cluster["members"]:
            if member["key"] in seen:
                continue
            seen.add(member["key"])
            descriptors.append(_obs_descriptor(member))
        sources = sorted({_source_label(member) for member in cluster["members"]})
        cross_source_bonus = 0.025 * max(0, len({m["source"] for m in cluster["members"]}) - 1)
        score = min(1.0, max(m["score"] for m in cluster["members"]) + cross_source_bonus)
        assets = []
        for member in cluster["members"]:
            if member.get("asset") and member["asset"] not in assets:
                assets.append(member["asset"])
        # Structural IDs are useful only when every observation that supplied a value
        # agrees. A detector disagreement must fall back to absolute/raster fidelity,
        # never silently manufacture a shared panel or chart relationship.
        structural = {}
        for field in _STRUCTURAL_FIELDS:
            values = {
                member.get("structural", {}).get(field)
                for member in cluster["members"]
                if member.get("structural", {}).get(field) not in (None, "")
            }
            if len(values) == 1:
                structural[field] = values.pop()
        rel = os.path.join("fused_elements", f"{cid}.png")
        path = os.path.join(run_dir, rel) if run_dir else None
        if path:
            _write_mask(mask, box, path)
        member_meta = {}
        for member in cluster["members"]:
            if member.get("meta"):
                member_meta = {**member["meta"], **member_meta}
        if winner.get("meta"):
            member_meta.update(winner["meta"])
        results.append(
            {
                "id": cid,
                **({"meta": member_meta} if member_meta else {}),
                "box": box,
                "kind": winner["kind"],
                "role": winner["role"],
                "score": round(score, 4),
                "area": float(mask.sum()),
                "coverage": round(float(mask.sum()) / (canvas["w"] * canvas["h"]), 6),
                "source": "fused",
                "mask": {"kind": "alpha", "src": rel} if run_dir else {"kind": "alpha"},
                "mask_src": rel if run_dir else None,
                "mask_path": os.path.abspath(path) if path else None,
                "asset_src": assets[0] if assets else None,
                "asset_candidates": assets,
                "parent_id": None,
                "relationships": [],
                **structural,
                "provenance": {
                    "sources": sources,
                    "observations": descriptors,
                    "nms": {
                        "observation_count": len(descriptors),
                        "merged_count": max(0, len(descriptors) - 1),
                        "merges": cluster["merges"],
                    },
                },
            }
        )

    # Preserve meaningful nesting. The smallest containing candidate wins as the direct parent;
    # nested masks were never eligible for NMS suppression above.  Containment is tested
    # against a slightly dilated parent so a child's anti-aliased fringe (1-2 px past the
    # model parent mask) cannot break a real icon-in-button relationship.
    link_dilate_px = int(opts.get("containment_dilate_px", 0))
    dilated_parents = {}
    for child_i, child_cluster in enumerate(clusters):
        child_mask = child_cluster["winner"]["mask"]
        child_area = int(child_mask.sum())
        best = None
        for parent_i, parent_cluster in enumerate(clusters):
            if child_i == parent_i:
                continue
            parent_mask = parent_cluster["winner"]["mask"]
            parent_area = int(parent_mask.sum())
            if parent_area <= child_area:
                continue
            if parent_i not in dilated_parents:
                dilated_parents[parent_i] = _dilate(parent_mask, link_dilate_px)
            containment = _containment_in(child_mask, dilated_parents[parent_i])
            area_ratio = child_area / parent_area
            if (
                containment >= opts["nested_containment"]
                and area_ratio <= opts["nested_max_area_ratio"]
                and _meaningful_parent(parent_cluster["winner"], child_cluster["winner"])
            ):
                if best is None or parent_area < best[0]:
                    best = (parent_area, parent_i, containment, area_ratio)
        if best is not None:
            _, parent_i, containment, area_ratio = best
            parent_id = results[parent_i]["id"]
            results[child_i]["parent_id"] = parent_id
            results[child_i]["relationships"].append(
                {
                    "type": "nested-in",
                    "target": parent_id,
                    "containment": round(containment, 4),
                    "area_ratio": round(area_ratio, 4),
                }
            )

    # Printed-on-product artwork folds into its parent product (needs the nesting
    # links above): the "logo" prompt firing on packaging print must not ship as a
    # separate element (013's bag lockups) — see _absorb_printed_artwork.
    # Packaging highlight shells (002 VANILLE bar mislabeled as button) fold the same way.
    absorbed_artwork = []
    absorbed_shells = []
    if opts.get("absorb_printed_artwork", True):
        results, absorbed_artwork = _absorb_printed_artwork(results, opts)
    if opts.get("absorb_packaging_shells", True):
        results, absorbed_shells = _absorb_packaging_shells(results, opts)

    # Post-fusion glyph/chart refinement (src/icon_detect.py): ✓ / ✗ / ? list glyphs
    # become role-tagged icon elements attached to their text rows, stacked duplicate
    # fragments collapse into one icon, and a gridline-verified chart region is
    # re-roled to "chart" (intentional raster cluster).  Additive and advisory —
    # a refinement failure must never break fusion.
    if run_dir and (cfg.get("icon_detect") or {}).get("enabled", True):
        try:
            from . import icon_detect
            results = icon_detect.refine(results, canvas=canvas, cfg=cfg, run_dir=run_dir)
        except Exception:
            pass

    # 066: checklist cards owning many ✓/✗ chips must keep those pixels. Emitting the
    # chips as peelable siblings punches holes that later stages cannot heal.
    absorbed_list_icons = []
    if opts.get("absorb_list_icons_into_cards", True):
        results, absorbed_list_icons = _absorb_list_icons_into_cards(results, opts)

    # Generalized invariant (runs LAST, after icon_detect re-roles ✓/✗ glyphs and the
    # checklist-card absorber): every remaining non-product child that rides inside a
    # product cutout folds into the product so nothing inside a SKU is ever emitted as a
    # separate element that reconstruct/peel would punch out of the packaging.
    absorbed_decorations = []
    if opts.get("absorb_product_decorations", True):
        results, absorbed_decorations = _absorb_product_decorations(results, opts)

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "fused_elements.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        report = {
            "kind": "fusion-diagnostics",
            "suppressed_junk_bands": suppressed_bands,
            "absorbed_printed_artwork": absorbed_artwork,
            "absorbed_packaging_shells": absorbed_shells,
            "absorbed_product_decorations": absorbed_decorations,
            "absorbed_list_icons": absorbed_list_icons,
            "counts": {
                "canonical": len(results),
                "suppressed_junk_bands": len(suppressed_bands),
                "absorbed_printed_artwork": len(absorbed_artwork),
                "absorbed_packaging_shells": len(absorbed_shells),
                "absorbed_product_decorations": len(absorbed_decorations),
                "absorbed_list_icons": len(absorbed_list_icons),
            },
        }
        with open(os.path.join(run_dir, "fusion_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return results


fuse_elements = fuse
