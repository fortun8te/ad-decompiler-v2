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
    "canonical_prefix": "E",
}

_GRAPHIC_ROLES = {"logo", "icon", "arrow", "badge", "symbol", "pictogram", "sticker"}
_RASTER_ROLES = {
    "photo",
    "product",
    "person",
    "face",
    "hand",
    "object",
    "illustration",
    "package",
}
_SHAPE_ROLES = {"shape", "button", "card", "container", "background", "frame"}
_CONTAINER_ROLES = {"shape", "button", "card", "container", "badge", "frame", "background"}


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
        role = str(role).strip().lower().replace(" ", "-")
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


def _semantic_compatible(a: dict, b: dict, mask_iou: float) -> bool:
    fa, fb = _family(a["role"]), _family(b["role"])
    if fa == fb or "unknown" in (fa, fb):
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
    compatible = _semantic_compatible(a, b, metrics["iou"])
    return compatible and (mask_duplicate or containment_duplicate or bbox_duplicate), metrics


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
    if parent["role"] in _CONTAINER_ROLES or parent["role"] in _RASTER_ROLES or parent["kind"] == "photo-fragment":
        return parent["role"] != child["role"] or parent["kind"] != child["kind"]
    return parent["kind"] == "shape" and child["kind"] != "shape"


def _write_mask(mask, box: dict, path: str) -> None:
    np = _np()
    from PIL import Image

    crop = mask[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(crop.astype(np.uint8) * 255).save(path)


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
        for cluster in clusters:
            duplicate, metrics = _duplicate(obs, cluster["winner"], opts)
            if duplicate and metrics["iou"] > match_score:
                match, match_score, match_metrics = cluster, metrics["iou"], metrics
        if match is None:
            clusters.append({"winner": obs, "members": [obs], "merges": []})
        else:
            match["members"].append(obs)
            match["merges"].append(
                {
                    "key": obs["key"],
                    "mask_iou": round(match_metrics["iou"], 4),
                    "containment": round(match_metrics["containment"], 4),
                    "area_ratio": round(match_metrics["area_ratio"], 4),
                }
            )

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
        rel = os.path.join("fused_elements", f"{cid}.png")
        path = os.path.join(run_dir, rel) if run_dir else None
        if path:
            _write_mask(mask, box, path)
        results.append(
            {
                "id": cid,
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
    # nested masks were never eligible for NMS suppression above.
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
            metrics = _mask_metrics(child_mask, parent_mask)
            if (
                metrics["containment"] >= opts["nested_containment"]
                and metrics["area_ratio"] <= opts["nested_max_area_ratio"]
                and _meaningful_parent(parent_cluster["winner"], child_cluster["winner"])
            ):
                if best is None or parent_area < best[0]:
                    best = (parent_area, parent_i, metrics)
        if best is not None:
            _, parent_i, metrics = best
            parent_id = results[parent_i]["id"]
            results[child_i]["parent_id"] = parent_id
            results[child_i]["relationships"].append(
                {
                    "type": "nested-in",
                    "target": parent_id,
                    "containment": round(metrics["containment"], 4),
                    "area_ratio": round(metrics["area_ratio"], 4),
                }
            )

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "fused_elements.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    return results


fuse_elements = fuse
