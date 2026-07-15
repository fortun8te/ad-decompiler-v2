"""vlm_element_propose.py — optional VLM box proposals before SAM box-refine.

After deterministic residual-CC detection, tile the normalized ad (overlapping grid
and/or residual-gap crops) and ask a local vision-language model for coarse element
boxes. Accepted proposals are merged into the residual list as role-tagged entries for
SAM geometric refinement. Disabled by default and never raises: LM Studio outages or
parse failures leave the residual list unchanged.
"""
from __future__ import annotations

import copy
import json
import re

from src import vlm_client

_DEFAULT_GRID = 4
_DEFAULT_OVERLAP = 0.25
_DEFAULT_PASSES = 2
_DEFAULT_MAX_TOKENS = 400
_DEFAULT_IOU = 0.55
_DEFAULT_GAP_COVERAGE = 0.35
_DEFAULT_MAX_TILES = 20
_DEFAULT_MAX_PROPOSALS = 32
_DEFAULT_LIGHTWEIGHT_GRID = 2
_DEFAULT_LIGHTWEIGHT_MAX_TILES = 8
_DEFAULT_LIGHTWEIGHT_OVERLAP = 0.15
_DEFAULT_LIGHTWEIGHT_BELOW_SAM = 3

_LABELS = frozenset({
    "product", "icon", "button", "badge", "person", "panel", "comparison_panel",
    "screenshot", "ui_panel", "receipt", "chart", "graph", "table",
    "nutrition_panel", "diagram", "infographic", "product_cluster",
})
_LABEL_TO_KIND = {
    "product": "photo-fragment",
    "person": "photo-fragment",
    "icon": "icon",
    "button": "shape",
    "badge": "icon",
    "screenshot": "photo-fragment",
    "ui_panel": "photo-fragment",
    "receipt": "photo-fragment",
    "chart": "photo-fragment",
    "graph": "photo-fragment",
    "table": "photo-fragment",
    "nutrition_panel": "photo-fragment",
    "diagram": "photo-fragment",
    "infographic": "photo-fragment",
    "product_cluster": "photo-fragment",
    "panel": "photo-fragment",
    "comparison_panel": "photo-fragment",
}

_PROMPT = (
    "This crop is from a digital advertisement. List every distinct non-text visual element "
    "you can see (product, person, icon, button, badge, image panel in a multi-panel layout, "
    "before/after comparison panel, screenshot, UI panel, receipt, chart, "
    "graph, table, nutrition panel, diagram, infographic, inseparable product cluster). "
    "Ignore ordinary background and typography.\n\n"
    "Reply with ONLY valid JSON — a JSON array, no markdown, no explanation:\n"
    '[{"label": "product|icon|button|badge|person|panel|comparison_panel|screenshot|ui_panel|receipt|chart|graph|table|nutrition_panel|diagram|infographic|product_cluster", '
    '"approx_box_fraction": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}]\n\n'
    "approx_box_fraction uses 0-1 coordinates relative to THIS crop (x,y = top-left, w,h = size). "
    "Use an empty array [] when no elements are visible."
)

_PROPOSAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "approx_box_fraction"],
        "properties": {
            "label": {"type": "string", "enum": sorted(_LABELS)},
            "approx_box_fraction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["x", "y", "w", "h"],
                "properties": {
                    key: {"type": "number", "minimum": 0, "maximum": 1}
                    for key in ("x", "y", "w", "h")
                },
            },
        },
    },
}


def _vlm_cfg(cfg: dict) -> dict:
    root = (cfg or {}).get("vlm") or {}
    ep = root.get("element_propose") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
    }
    merged.update({k: v for k, v in ep.items() if k != "enabled"})
    return merged


def _iou(a: dict, b: dict) -> float:
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _clip_box(box: dict, width: int, height: int) -> dict | None:
    x0 = max(0, int(box.get("x", 0)))
    y0 = max(0, int(box.get("y", 0)))
    x1 = min(width, x0 + max(0, int(box.get("w", 0))))
    y1 = min(height, y0 + max(0, int(box.get("h", 0))))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return {"x": x0, "y": y0, "w": w, "h": h}


def _parse_fraction_box(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x", 0))
        y = float(raw.get("y", 0))
        w = float(raw.get("w", 0))
        h = float(raw.get("h", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0 - x, max(0.0, w))
    h = min(1.0 - y, max(0.0, h))
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _parse_proposals(raw: str) -> list[dict] | None:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if label not in _LABELS:
            continue
        frac = _parse_fraction_box(item.get("approx_box_fraction") or item.get("box"))
        if frac is None:
            continue
        out.append({"label": label, "approx_box_fraction": frac})
    return out


def _proposal_signature(proposals: list[dict]) -> tuple:
    sig = []
    for p in proposals:
        frac = p["approx_box_fraction"]
        sig.append((
            p["label"],
            round(frac["x"], 2),
            round(frac["y"], 2),
            round(frac["w"], 2),
            round(frac["h"], 2),
        ))
    return tuple(sig)


# The VLM is not byte-deterministic even at temperature=0 (structured JSON label choice can
# vary between identical calls). Requiring an exact signature match between the two passes
# would make consensus fail almost every time on a real model, so agreement is judged on
# geometry instead: same element count, and every box in one pass has a positional match in
# the other pass. Labels are allowed to differ.
_BOX_MATCH_IOU = 0.6
_BOX_MATCH_CENTER_FRACTION = 0.05  # of tile diagonal, as a center-distance tolerance


def _box_center(frac: dict) -> tuple[float, float]:
    return (frac["x"] + frac["w"] / 2.0, frac["y"] + frac["h"] / 2.0)


def _fraction_box_iou(a: dict, b: dict) -> float:
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _boxes_match(a: dict, b: dict) -> bool:
    if _fraction_box_iou(a, b) >= _BOX_MATCH_IOU:
        return True
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    # Fraction boxes are 0-1 relative to the tile, so the tile diagonal is sqrt(2).
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return dist <= _BOX_MATCH_CENTER_FRACTION * (2 ** 0.5)


def _proposals_agree(a: list[dict], b: list[dict]) -> bool:
    """True when both passes propose the same number of elements and each box in `a`
    has an unmatched positional counterpart in `b` (IoU or center-distance tolerance).
    Labels need not match."""
    if len(a) != len(b):
        return False
    remaining = list(b)
    for item in a:
        match_idx = next(
            (idx for idx, other in enumerate(remaining)
             if _boxes_match(item["approx_box_fraction"], other["approx_box_fraction"])),
            None,
        )
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return True


def _two_pass_proposals(
    crop: bytes,
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    passes: int,
) -> tuple[list[dict] | None, str | None]:
    parsed_passes: list[list[dict] | None] = []
    for _ in range(max(1, passes)):
        try:
            raw = vlm_client.ask_vlm(
                crop,
                _PROMPT,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                response_schema=_PROPOSAL_SCHEMA,
            )
        except Exception:
            return None, "vlm_error"
        parsed = _parse_proposals(raw)
        if parsed is None:
            return None, "vlm_parse_error"
        parsed_passes.append(parsed)
    first = parsed_passes[0]
    for other in parsed_passes[1:]:
        if not _proposals_agree(first, other):
            return None, "vlm_disagreement"
    return first, None


def _grid_tiles(width: int, height: int, grid: int, overlap: float) -> list[dict]:
    n = max(1, int(grid))
    overlap = min(0.75, max(0.0, float(overlap)))
    stride_x = width / n
    stride_y = height / n
    pad_x = stride_x * overlap
    pad_y = stride_y * overlap
    tiles: list[dict] = []
    for row in range(n):
        for col in range(n):
            x0 = max(0, int(round(col * stride_x - pad_x)))
            y0 = max(0, int(round(row * stride_y - pad_y)))
            x1 = min(width, int(round((col + 1) * stride_x + pad_x)))
            y1 = min(height, int(round((row + 1) * stride_y + pad_y)))
            if x1 > x0 and y1 > y0:
                tiles.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return tiles


def _tile_coverage_by_residual(tile: dict, residual: list[dict]) -> float:
    area = max(1, tile["w"] * tile["h"])
    covered = 0
    for item in residual:
        box = item.get("box")
        if not box:
            continue
        ix = max(0, min(tile["x"] + tile["w"], box["x"] + box["w"]) - max(tile["x"], box["x"]))
        iy = max(0, min(tile["y"] + tile["h"], box["y"] + box["h"]) - max(tile["y"], box["y"]))
        covered += ix * iy
    return covered / area


def _gap_tiles(width: int, height: int, residual: list[dict], grid: int, overlap: float,
               max_coverage: float) -> list[dict]:
    return [
        tile for tile in _grid_tiles(width, height, grid, overlap)
        if _tile_coverage_by_residual(tile, residual) <= max_coverage
    ]


def _fraction_to_pixel(frac: dict, tile: dict, canvas_w: int, canvas_h: int) -> dict | None:
    box = {
        "x": tile["x"] + int(round(frac["x"] * tile["w"])),
        "y": tile["y"] + int(round(frac["y"] * tile["h"])),
        "w": int(round(frac["w"] * tile["w"])),
        "h": int(round(frac["h"] * tile["h"])),
    }
    return _clip_box(box, canvas_w, canvas_h)


def _dedupe_boxes(boxes: list[dict], iou_thresh: float) -> list[dict]:
    kept: list[dict] = []
    for candidate in boxes:
        if any(_iou(candidate["box"], other["box"]) >= iou_thresh for other in kept):
            continue
        kept.append(candidate)
    return kept


def _build_element(proposal: dict, canvas_w: int, canvas_h: int, tile_idx: int) -> dict | None:
    box = proposal.get("box")
    if not box:
        return None
    label = proposal["label"]
    area = float(box["w"] * box["h"])
    canvas_area = max(1, canvas_w * canvas_h)
    return {
        "box": box,
        "kind": _LABEL_TO_KIND.get(label, "photo-fragment"),
        "role": label,
        "area": area,
        "coverage": round(area / canvas_area, 4),
        "source": "vlm-propose",
        "meta": {"vlm_element": {"label": label, "tile": tile_idx}},
    }


def _should_use_lightweight_grid(ep: dict, sam_element_count: int | None) -> bool:
    if ep.get("lightweight_grid"):
        return True
    threshold = ep.get("lightweight_grid_below_sam_count")
    if threshold is None or sam_element_count is None:
        return False
    return int(sam_element_count) < int(threshold)


def _effective_ep_cfg(ep: dict, sam_element_count: int | None) -> dict:
    if not _should_use_lightweight_grid(ep, sam_element_count):
        return ep
    lightweight = {
        "grid": int(ep.get("lightweight_grid_size", _DEFAULT_LIGHTWEIGHT_GRID)),
        "max_tiles": int(ep.get("lightweight_max_tiles", _DEFAULT_LIGHTWEIGHT_MAX_TILES)),
        "overlap": float(ep.get("lightweight_overlap", _DEFAULT_LIGHTWEIGHT_OVERLAP)),
        "tile_mode": "grid",
    }
    return {**ep, **lightweight}


class _ResidualWithNotice(list):
    """A residual list that also carries a `vlm_degraded` notice.

    Subclassing list keeps every existing caller/test working unmodified
    (`out == residual`, iteration, indexing, len() all behave like a plain
    list), while a caller that wants to observe the degradation can read
    `.vlm_degraded` off the returned object instead of it being silently
    dropped."""

    vlm_degraded: dict | None = None


def _degraded_residual(residual: list[dict], stats: dict) -> list[dict]:
    """Return `residual` unchanged, but loudly annotated when tiles degraded.

    Historically enrich_residual "never raised" and silently returned the
    unchanged residual on vlm_error/vlm_disagreement, which made LM Studio
    outages or persistent disagreement invisible to callers. This keeps the
    non-raising contract but makes the degradation observable."""
    notes = stats.get("notes") or []
    if not notes:
        return residual
    out = _ResidualWithNotice(residual)
    out.vlm_degraded = {
        "reason": notes[0] if len(notes) == 1 else ",".join(notes),
        "tile_count": stats.get("tiles", 0),
    }
    return out


def enrich_residual(
    image_path: str,
    residual: list[dict],
    cfg: dict,
    sam_element_count: int | None = None,
) -> list[dict]:
    """Return residual proposals augmented with optional VLM box hints. Never raises."""
    ep = ((cfg or {}).get("vlm") or {}).get("element_propose") or {}
    if not ep.get("enabled", False):
        return residual
    ep = _effective_ep_cfg(ep, sam_element_count)

    vcfg = _vlm_cfg(cfg)
    vcfg.update({k: v for k, v in ep.items() if k != "enabled"})
    base_url = str(vcfg.get("base_url") or vlm_client._DEFAULT_BASE_URL)
    model = str(vcfg.get("model") or vlm_client._DEFAULT_MODEL)
    timeout_s = float(vcfg.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S)
    max_tokens = int(vcfg.get("max_tokens") or _DEFAULT_MAX_TOKENS)
    passes = int(vcfg.get("passes", _DEFAULT_PASSES))
    grid = int(vcfg.get("grid", _DEFAULT_GRID))
    overlap = float(vcfg.get("overlap", _DEFAULT_OVERLAP))
    iou_thresh = float(vcfg.get("iou_thresh", _DEFAULT_IOU))
    gap_coverage = float(vcfg.get("gap_coverage", _DEFAULT_GAP_COVERAGE))
    max_tiles = int(vcfg.get("max_tiles", _DEFAULT_MAX_TILES))
    max_proposals = int(vcfg.get("max_proposals", _DEFAULT_MAX_PROPOSALS))
    tile_mode = str(vcfg.get("tile_mode", "both")).strip().lower()

    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
    except Exception:
        return residual

    width, height = image.size
    base_residual = list(residual or [])
    tiles: list[dict] = []
    if tile_mode in {"grid", "both"}:
        tiles.extend(_grid_tiles(width, height, grid, overlap))
    if tile_mode in {"gaps", "both"}:
        gap = _gap_tiles(width, height, base_residual, grid, overlap, gap_coverage)
        seen = {(t["x"], t["y"], t["w"], t["h"]) for t in tiles}
        for tile in gap:
            key = (tile["x"], tile["y"], tile["w"], tile["h"])
            if key not in seen:
                tiles.append(tile)
                seen.add(key)
    if not tiles:
        return residual
    if len(tiles) > max_tiles:
        tiles = tiles[:max_tiles]

    raw_proposals: list[dict] = []
    stats = {"tiles": len(tiles), "accepted": 0, "skipped": 0, "notes": []}

    for tile_idx, tile in enumerate(tiles):
        crop = vlm_client.crop_box_bytes(image, tile, padding=0)
        if crop is None:
            stats["skipped"] += 1
            continue
        proposals, note = _two_pass_proposals(
            crop,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            passes=passes,
        )
        if note:
            stats["skipped"] += 1
            if note not in stats["notes"]:
                stats["notes"].append(note)
            continue
        if not proposals:
            continue
        stats["accepted"] += 1
        for proposal in proposals:
            box = _fraction_to_pixel(proposal["approx_box_fraction"], tile, width, height)
            if box is None:
                continue
            raw_proposals.append({"label": proposal["label"], "box": box, "tile": tile_idx})

    if not raw_proposals:
        return _degraded_residual(residual, stats)

    existing_boxes = [item["box"] for item in base_residual if item.get("box")]
    filtered: list[dict] = []
    for proposal in raw_proposals:
        if any(_iou(proposal["box"], box) >= iou_thresh for box in existing_boxes):
            continue
        filtered.append(proposal)
    filtered = _dedupe_boxes(filtered, iou_thresh)
    if max_proposals > 0:
        filtered = filtered[:max_proposals]
    if not filtered:
        return _degraded_residual(residual, stats)

    out = copy.deepcopy(base_residual)
    used_ids = {str(item.get("id", "")) for item in out}
    next_idx = 0
    for proposal in filtered:
        element = _build_element(proposal, width, height, proposal["tile"])
        if element is None:
            continue
        while f"VP{next_idx}" in used_ids:
            next_idx += 1
        element["id"] = f"VP{next_idx}"
        used_ids.add(element["id"])
        next_idx += 1
        if not any(e.get("source") == "vlm-propose" for e in out):
            element.setdefault("meta", {})["vlm_element_propose"] = stats
        out.append(element)

    return out
