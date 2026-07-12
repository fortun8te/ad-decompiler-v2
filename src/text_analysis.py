"""Text geometry, typography, and hierarchy enrichment.

This module deliberately sits *after* OCR.  OCR owns character recognition;
``analyze_text`` turns its line/word boxes into the richer observations needed by
the Figma compiler:

* painted (visible-ink) bounds, baseline, rotation, and sampled text colour
* conservative typography estimates and ranked candidates
* paragraph/text-block grouping, alignment, roles, hierarchy, and shared styles
* optional, bounded local-font retrieval by rendering the recognized string

Heavy/optional dependencies are imported lazily.  With font matching disabled
(the default), the implementation needs only Pillow and NumPy and has a reliable
geometry-only fallback when the source image cannot be opened.

Public API::

    enriched = analyze_text(image_path, ocr_result, cfg)

Configuration is read from ``cfg["text_analysis"]``.  Local matching accepts a
boolean or mapping under ``font_matching``::

    text_analysis:
      font_matching:
        enabled: true
        max_fonts: 48
        max_lines: 12
        top_k: 5
        font_dirs: []       # optional; platform font dirs are used otherwise
        font_files: []      # useful for a controlled/private font catalogue
        families: []        # optional filename/family filter

The returned mapping remains OCR-shaped: all original top-level keys and line
fields are preserved, while ``lines`` are enriched and ``blocks``, ``styles``,
``sections`` and ``hierarchy`` are added.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import math
import os
import re
import statistics
import time
from typing import Any, Iterable, Optional


_FONT_DISCOVERY_CACHE: dict[tuple, list[dict]] = {}
_FONT_META_CACHE: dict[str, dict] = {}
_FONT_MATCH_CACHE: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_FONT_MATCH_CACHE_LIMIT = 128

_DEFAULT_FAMILIES = ["Inter", "Arial", "Helvetica", "Roboto", "DejaVu Sans"]
_DEFAULT_LOCAL_SCORE_THRESHOLD = 0.55
_GOOGLE_FONTS_CACHE_DIRS = [
    "~/.cache/google-fonts",
    "~/.local/share/fonts/google-fonts",
    "~/.fonts/google-fonts",
]
_CTA_RE = re.compile(
    r"\b(shop|buy|order|get|try|learn|discover|download|book|join|start|sign up|"
    r"subscribe|claim|apply|contact|swipe|tap|click)(\s+now|\s+today)?\b",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    r"(?:[$€£¥]\s?\d|\d(?:[.,]\d{1,2})?\s?(?:usd|eur|gbp|dollars?|euros?))",
    re.IGNORECASE,
)
_OFFER_RE = re.compile(r"(?:\b\d{1,3}\s?%|\bsave\b|\boff\b|\bfree\b)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generic helpers


def _text_cfg(cfg: Optional[dict]) -> dict:
    cfg = cfg or {}
    value = cfg.get("text_analysis") or {}
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_box(box: Optional[dict]) -> dict:
    box = box or {}
    return {
        "x": _num(box.get("x")),
        "y": _num(box.get("y")),
        "w": max(0.0, _num(box.get("w"))),
        "h": max(0.0, _num(box.get("h"))),
    }


def _union_boxes(boxes: Iterable[dict]) -> dict:
    boxes = [_clean_box(b) for b in boxes]
    if not boxes:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _box_center(box: dict) -> tuple[float, float]:
    b = _clean_box(box)
    return b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0


def _horizontal_overlap(a: dict, b: dict) -> float:
    a, b = _clean_box(a), _clean_box(b)
    overlap = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    denom = max(1.0, min(a["w"], b["w"]))
    return overlap / denom


def _quad_rotation(quad: Any) -> float:
    try:
        p0, p1 = quad[0], quad[1]
        angle = math.degrees(math.atan2(float(p1[1]) - float(p0[1]),
                                        float(p1[0]) - float(p0[0])))
    except (TypeError, ValueError, IndexError):
        return 0.0
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return round(angle, 3)


def _rgb_hex(rgb: Iterable[float]) -> str:
    vals = [max(0, min(255, int(round(float(v))))) for v in rgb]
    while len(vals) < 3:
        vals.append(0)
    return "#%02x%02x%02x" % tuple(vals[:3])


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = str(value or "#000000").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return (0, 0, 0)


def _colour_distance(a: str, b: str) -> float:
    aa, bb = _hex_rgb(a), _hex_rgb(b)
    return math.sqrt(sum((aa[i] - bb[i]) ** 2 for i in range(3)))


def _median(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(v) for v in values]
    return float(statistics.median(values)) if values else float(default)


# ---------------------------------------------------------------------------
# Painted text geometry


def _load_rgb(path: str):
    try:
        from PIL import Image
        import numpy as np

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        return None


def _otsu(values) -> float:
    """Small NumPy-only Otsu implementation for uint8-like values."""
    import numpy as np

    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probabilities = hist / total
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * np.arange(256))
    total_mu = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros(256, dtype=np.float64)
    valid = denom > 1e-12
    between[valid] = ((total_mu * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(np.argmax(between))


def _minority_luminance_mask(crop):
    lum = crop[..., 0] * 0.2126 + crop[..., 1] * 0.7152 + crop[..., 2] * 0.0722
    threshold = _otsu(lum)
    dark = lum < threshold
    light = lum > threshold
    candidates = []
    for mask in (dark, light):
        ratio = float(mask.mean())
        if 0.002 <= ratio <= 0.65:
            candidates.append((ratio, mask))
    return min(candidates, key=lambda item: item[0])[1] if candidates else dark


def _clean_ink_mask(mask):
    """Remove tiny specks when OpenCV is present; otherwise return unchanged."""
    try:
        import cv2
        import numpy as np

        raw = mask.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        keep = np.zeros_like(raw)
        min_area = max(2, int(raw.size * 0.0004))
        for idx in range(1, count):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == idx] = 1
        return keep.astype(bool) if keep.any() else mask
    except Exception:
        return mask


def _ink_mask(crop):
    """Return (mask, confidence) using border-estimated background contrast."""
    import numpy as np

    if crop is None or crop.size == 0:
        return None, 0.0
    h, w = crop.shape[:2]
    border_width = max(1, min(3, h // 5, w // 5))
    borders = np.concatenate([
        crop[:border_width].reshape(-1, 3),
        crop[-border_width:].reshape(-1, 3),
        crop[:, :border_width].reshape(-1, 3),
        crop[:, -border_width:].reshape(-1, 3),
    ])
    bg = np.median(borders.astype(np.float32), axis=0)
    delta = np.sqrt(np.sum((crop.astype(np.float32) - bg) ** 2, axis=2))
    scaled = np.clip(delta, 0, 255)
    threshold = max(10.0, _otsu(scaled))
    mask = delta > threshold
    ratio = float(mask.mean())
    if ratio < 0.002 or ratio > 0.68:
        mask = _minority_luminance_mask(crop)
        ratio = float(mask.mean())
    mask = _clean_ink_mask(mask)
    ratio = float(mask.mean())
    contrast = float(np.median(delta[mask])) if mask.any() else 0.0
    confidence = min(1.0, max(0.0, contrast / 80.0))
    if not 0.002 <= ratio <= 0.68:
        confidence *= 0.25
    return mask, round(confidence, 4)


def _fallback_geometry(line: dict) -> tuple[dict, dict, float, None]:
    box = _clean_box(line.get("box"))
    rotation = _quad_rotation(line.get("quad"))
    baseline_y = box["y"] + box["h"] * 0.82
    slope = math.tan(math.radians(rotation))
    baseline = {
        "x0": round(box["x"], 3),
        "y0": round(baseline_y, 3),
        "x1": round(box["x"] + box["w"], 3),
        "y1": round(baseline_y + slope * box["w"], 3),
        "confidence": 0.2,
    }
    return box, baseline, 0.0, None


_FLAT_FILL_BLACK = {"kind": "flat", "color": "#000000"}
_GRADIENT_COLOR_DISTANCE = 36.0
_STROKE_COLOR_DISTANCE = 55.0
_STROKE_MIN_RIM_FRAC = 0.08
_STROKE_MAX_RIM_STD = 45.0


def _erode_mask(mask):
    """1px binary erosion; uses OpenCV when available, a NumPy fallback otherwise."""
    import numpy as np

    try:
        import cv2

        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    except Exception:
        core = np.asarray(mask).astype(bool)
        eroded = np.zeros_like(core)
        if core.shape[0] < 3 or core.shape[1] < 3:
            return eroded
        eroded[1:-1, 1:-1] = (
            core[1:-1, 1:-1] & core[:-2, 1:-1] & core[2:, 1:-1] & core[1:-1, :-2] & core[1:-1, 2:]
        )
        return eroded


def _dominant_axis_gradient(crop, mask) -> Optional[dict]:
    """Detect a simple 2-stop linear gradient by comparing the median ink colour at
    opposite ends of the mask along whichever axis (vertical/horizontal) shows the
    larger colour split. Returns None for ordinary flat-fill text."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if ys.size < 24:
        return None
    h, w = mask.shape[:2]
    candidates = []
    y_lo = np.percentile(ys, 20)
    y_hi = np.percentile(ys, 80)
    row_idx = np.arange(h)[:, None]
    top = mask & (row_idx <= y_lo)
    bottom = mask & (row_idx >= y_hi)
    if top.any() and bottom.any():
        top_rgb = np.median(crop[top].astype(np.float32), axis=0)
        bottom_rgb = np.median(crop[bottom].astype(np.float32), axis=0)
        dist = math.sqrt(float(np.sum((top_rgb - bottom_rgb) ** 2)))
        candidates.append((dist, 90.0, top_rgb, bottom_rgb))
    x_lo = np.percentile(xs, 20)
    x_hi = np.percentile(xs, 80)
    col_idx = np.arange(w)[None, :]
    left = mask & (col_idx <= x_lo)
    right = mask & (col_idx >= x_hi)
    if left.any() and right.any():
        left_rgb = np.median(crop[left].astype(np.float32), axis=0)
        right_rgb = np.median(crop[right].astype(np.float32), axis=0)
        dist = math.sqrt(float(np.sum((left_rgb - right_rgb) ** 2)))
        candidates.append((dist, 0.0, left_rgb, right_rgb))
    if not candidates:
        return None
    dist, angle, start_rgb, end_rgb = max(candidates, key=lambda item: item[0])
    if dist < _GRADIENT_COLOR_DISTANCE:
        return None
    return {
        "kind": "linear",
        "angle": angle,
        "stops": [
            {"offset": 0.0, "color": _rgb_hex(start_rgb)},
            {"offset": 1.0, "color": _rgb_hex(end_rgb)},
        ],
    }


def _boundary_depth(mask):
    """Per-pixel integer distance to the nearest non-ink pixel (0 = on the mask's own
    edge). Uses OpenCV's exact distance transform when available; otherwise repeated
    1px erosion (dependency-free, adequate for typical small text crops)."""
    import numpy as np

    m = np.asarray(mask).astype(bool)
    try:
        import cv2

        return cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    except Exception:
        depth = np.zeros(m.shape, dtype=np.float32)
        current = m
        d = 0.0
        for _ in range(64):
            if not current.any():
                break
            depth[current] = d
            eroded = _erode_mask(current)
            if not eroded.any() or np.array_equal(eroded, current):
                break
            current = eroded
            d += 1.0
        return depth


def _stroke_from_mask(crop, mask) -> Optional[tuple[dict, str]]:
    """Sample a distinct outline-band colour around the glyph interior.

    Buckets ink pixels by depth-from-edge and finds the largest colour jump between
    consecutive depth rings — the boundary between an outline stroke band and the
    fill interior — regardless of the stroke's actual pixel width. Returns
    (stroke_dict, interior_fill_hex) or None when no separately coloured rim exists.
    """
    import numpy as np

    total = int(mask.sum())
    if total < 40:
        return None
    depth = _boundary_depth(mask)
    int_depth = depth.astype(np.int32)
    max_depth = int(int_depth[mask].max()) if mask.any() else 0
    if max_depth < 2:
        return None
    # A ring with only a handful of pixels (e.g. the tip of a single bowl/serif at
    # the deepest depth) produces an unreliable median that can masquerade as a
    # sharp colour "jump" against its neighbour. Require enough pixels per ring
    # before trusting it as a boundary candidate.
    min_ring_pixels = max(8, int(total * 0.01))
    profiles = []
    for d in range(min(max_depth, 40) + 1):
        ring = mask & (int_depth == d)
        if not ring.any():
            continue
        profiles.append((d, np.median(crop[ring].astype(np.float32), axis=0), int(ring.sum())))
    if len(profiles) < 3:
        return None
    reliable = [p for p in profiles if p[2] >= min_ring_pixels]
    if len(reliable) < 2:
        return None
    best_jump, boundary_depth = 0.0, None
    for i in range(1, len(reliable)):
        d0, rgb0, _ = reliable[i - 1]
        d1, rgb1, _ = reliable[i]
        jump = math.sqrt(float(np.sum((rgb0 - rgb1) ** 2)))
        if jump > best_jump:
            best_jump, boundary_depth = jump, d1
    if boundary_depth is None or best_jump < _STROKE_COLOR_DISTANCE:
        return None
    rim = mask & (int_depth < boundary_depth)
    interior = mask & (int_depth >= boundary_depth)
    if not rim.any() or not interior.any() or int(rim.sum()) < max(8, int(total * _STROKE_MIN_RIM_FRAC)):
        return None
    # Depth 0 sits on the mask's own outer edge and is contaminated by anti-aliased
    # blends with the background; sample the stroke colour just inside that ring.
    sample_lo = 1 if boundary_depth > 1 else 0
    rim_sample = mask & (int_depth >= sample_lo) & (int_depth < boundary_depth)
    if not rim_sample.any():
        rim_sample = rim
    rim_pixels = crop[rim_sample].astype(np.float32)
    rim_rgb = np.median(rim_pixels, axis=0)
    interior_rgb = np.median(crop[interior].astype(np.float32), axis=0)
    # Median absolute deviation, not stddev: thin-stroke regions (e.g. an 'F' stem)
    # can leak a minority of interior-coloured pixels into the rim sample, which
    # would blow up a plain stddev even though the rim colour itself is clean.
    consistency = float(np.median(np.abs(rim_pixels - rim_rgb), axis=0).mean())
    if consistency > _STROKE_MAX_RIM_STD:
        return None
    return {"kind": "flat", "color": _rgb_hex(rim_rgb), "width": round(float(boundary_depth), 1)}, _rgb_hex(interior_rgb)


def _paint_from_mask(crop, mask, fallback_hex: str) -> dict:
    """Best-effort fill/stroke description for the painted ink, in addition to the
    single flattened colour used elsewhere for backward compatibility."""
    fill = {"kind": "flat", "color": fallback_hex}
    stroke = None
    gradient = _dominant_axis_gradient(crop, mask)
    if gradient is not None:
        fill = gradient
    stroke_result = _stroke_from_mask(crop, mask)
    if stroke_result is not None:
        stroke, interior_hex = stroke_result
        if gradient is None:
            fill = {"kind": "flat", "color": interior_hex}
    return {"fill": fill, "stroke": stroke}


def _measure_shear_angle(mask) -> Optional[float]:
    """Estimate the ink mask's horizontal shear (italic slant) in degrees.

    Cross-correlates the column-ink profile of the mask's top half against its
    bottom half; the horizontal shift that best aligns the two profiles is the
    average per-row horizontal drift, converted to an angle from vertical. This
    aggregates over every column (not a single row's noisy centroid), so it stays
    stable across ordinary multi-letter, multi-word lines. A fallback signal
    independent of local font-file matching."""
    import numpy as np

    tight = _tight_mask(mask)
    if tight is None:
        return None
    h, w = tight.shape
    if h < 10 or w < 12:
        return None
    mid = h // 2
    top_profile = tight[:mid].sum(axis=0).astype(np.float64)
    bottom_profile = tight[mid:].sum(axis=0).astype(np.float64)
    if top_profile.sum() < 4 or bottom_profile.sum() < 4:
        return None
    top_profile = top_profile - top_profile.mean()
    bottom_profile = bottom_profile - bottom_profile.mean()
    min_overlap = max(6, int(w * 0.4))
    max_shift = max(1, w - min_overlap)
    best_shift, best_score = 0, None
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a, b = top_profile[: w - shift], bottom_profile[shift:]
        else:
            s = -shift
            a, b = top_profile[s:], bottom_profile[: w - s]
        if a.size < min_overlap:
            continue
        score = float(np.dot(a, b))
        if best_score is None or score > best_score:
            best_score, best_shift = score, shift
    if best_score is None or best_score <= 0 or best_shift == 0:
        return None
    dy = max(1.0, h / 2.0)
    return round(math.degrees(math.atan(best_shift / dy)), 2)


def _painted_geometry(image, line: dict) -> tuple[dict, dict, str, float, Any, dict]:
    import numpy as np

    box = _clean_box(line.get("box"))
    if image is None or box["w"] <= 0 or box["h"] <= 0:
        painted, baseline, confidence, mask = _fallback_geometry(line)
        paint = {"fill": dict(_FLAT_FILL_BLACK), "stroke": None}
        return painted, baseline, "#000000", confidence, mask, paint

    ih, iw = image.shape[:2]
    x0 = max(0, min(iw, int(math.floor(box["x"]))))
    y0 = max(0, min(ih, int(math.floor(box["y"]))))
    x1 = max(x0, min(iw, int(math.ceil(box["x"] + box["w"]))))
    y1 = max(y0, min(ih, int(math.ceil(box["y"] + box["h"]))))
    crop = image[y0:y1, x0:x1]
    mask, confidence = _ink_mask(crop)
    if mask is None or not mask.any():
        painted, baseline, fallback_conf, _ = _fallback_geometry(line)
        paint = {"fill": dict(_FLAT_FILL_BLACK), "stroke": None}
        return painted, baseline, "#000000", fallback_conf, None, paint

    ys, xs = np.nonzero(mask)
    lx0, ly0 = int(xs.min()), int(ys.min())
    lx1, ly1 = int(xs.max()) + 1, int(ys.max()) + 1
    painted = {
        "x": float(x0 + lx0),
        "y": float(y0 + ly0),
        "w": float(max(1, lx1 - lx0)),
        "h": float(max(1, ly1 - ly0)),
    }
    ink_pixels = crop[mask]
    if ink_pixels.size:
        # Anti-aliased edge pixels are blends with the background and vary with
        # the letter shapes in a word.  Sample the most background-distant ink
        # quartile so two lines with the same authored fill get the same colour.
        ch, cw = crop.shape[:2]
        bw = max(1, min(3, ch // 5, cw // 5))
        border = np.concatenate([
            crop[:bw].reshape(-1, 3), crop[-bw:].reshape(-1, 3),
            crop[:, :bw].reshape(-1, 3), crop[:, -bw:].reshape(-1, 3),
        ]).astype(np.float32)
        bg = np.median(border, axis=0)
        distances = np.sqrt(np.sum((ink_pixels.astype(np.float32) - bg) ** 2, axis=1))
        cutoff = np.percentile(distances, 70)
        core = ink_pixels[distances >= cutoff]
        rgb = np.median((core if core.size else ink_pixels).astype(np.float32), axis=0)
    else:
        rgb = [0, 0, 0]

    # The lower-percentile column bottoms reject sparse descenders while closely
    # following the cap/x-height baseline on ordinary Latin text.
    bottoms = []
    for column in range(mask.shape[1]):
        rows = np.nonzero(mask[:, column])[0]
        if rows.size:
            bottoms.append(float(rows.max()))
    baseline_local = float(np.percentile(bottoms, 68)) if bottoms else float(ly1 - 1)
    rotation = _quad_rotation(line.get("quad"))
    slope = math.tan(math.radians(rotation))
    baseline_y = y0 + baseline_local
    baseline = {
        "x0": round(painted["x"], 3),
        "y0": round(baseline_y, 3),
        "x1": round(painted["x"] + painted["w"], 3),
        "y1": round(baseline_y + slope * painted["w"], 3),
        "confidence": confidence,
    }
    tight_mask = mask[ly0:ly1, lx0:lx1]
    paint = _paint_from_mask(crop, mask, _rgb_hex(rgb))
    colour = paint["fill"]["stops"][0]["color"] if paint["fill"].get("kind") == "linear" else paint["fill"]["color"]
    return painted, baseline, colour, confidence, tight_mask, paint


# ---------------------------------------------------------------------------
# Typography estimates and optional font matching


def _estimate_weight(mask, painted_box: dict) -> int:
    if mask is None:
        return 400
    try:
        density = float(mask.mean())
    except Exception:
        return 400
    # Density alone is intentionally conservative; exact weight is resolved by
    # the later Figma render-fit loop.
    if density >= 0.46:
        return 700
    if density >= 0.34:
        return 600
    if density <= 0.12 and painted_box.get("h", 0) >= 10:
        return 300
    return 400


def _style_name(weight: int, italic: bool = False) -> str:
    if weight >= 700:
        base = "Bold"
    elif weight >= 600:
        base = "Semi Bold"
    elif weight >= 500:
        base = "Medium"
    elif weight <= 300:
        base = "Light"
    else:
        base = "Regular"
    return f"{base} Italic" if italic else base


def _weight_candidates(weight: int) -> list[dict]:
    allowed = [300, 400, 500, 600, 700, 800]
    ordered = sorted(allowed, key=lambda value: (abs(value - weight), value))[:3]
    return [
        {"value": value, "score": round(max(0.15, 1.0 - abs(value - weight) / 400.0), 3)}
        for value in ordered
    ]


def _size_candidates(size: float) -> list[dict]:
    values = [size, size * 0.94, size * 1.07]
    scores = [0.75, 0.58, 0.55]
    return [
        {"value": round(max(1.0, value), 2), "score": score}
        for value, score in zip(values, scores)
    ]


def _estimate_tracking(text: str, painted_box: dict, font_size: float) -> float:
    chars = len(text.replace(" ", ""))
    if chars < 4 or font_size <= 0:
        return 0.0
    spaces = text.count(" ")
    expected = chars * font_size * 0.52 + spaces * font_size * 0.28
    tracking = (painted_box["w"] - expected) / max(1, chars - 1)
    return round(max(-font_size * 0.08, min(font_size * 0.20, tracking)), 3)


def _fallback_font_candidates(weight: int, options: dict, top_k: int, italic: bool = False) -> list[dict]:
    families = options.get("fallback_families") or options.get("families") or _DEFAULT_FAMILIES
    if isinstance(families, str):
        families = [families]
    out = []
    for index, family in enumerate(families):
        out.append({
            "family": str(family),
            "style": _style_name(weight, italic=italic),
            "weight": int(weight),
            "score": round(max(0.25, 0.62 - index * 0.07), 3),
            "source": "fallback",
        })
        if len(out) >= top_k:
            break
    return out


def _typography_profile(geo: dict) -> dict:
    shear = geo.get("shear_angle")
    return {
        "weight": int(geo.get("weight", 400)),
        "italic": bool(shear is not None and abs(shear) >= 6.0),
        "font_size": float(geo.get("font_size", 16.0)),
    }


def _meta_alignment_adjustment(meta: dict, profile: dict) -> float:
    weight_delta = abs(int(meta.get("weight", 400)) - profile["weight"])
    adjustment = max(0.0, 1.0 - weight_delta / 500.0) * 0.12
    meta_italic = "italic" in str(meta.get("style") or "").lower()
    if meta_italic == profile["italic"]:
        adjustment += 0.08
    else:
        adjustment -= 0.06
    return adjustment


def _google_fonts_cache_dirs(options: dict) -> list[str]:
    explicit = options.get("google_fonts_cache") or options.get("google_fonts_dir")
    dirs = []
    if explicit:
        if isinstance(explicit, str):
            explicit = [explicit]
        dirs.extend(os.path.expanduser(path) for path in explicit)
    for path in _GOOGLE_FONTS_CACHE_DIRS:
        expanded = os.path.expanduser(path)
        if expanded not in dirs:
            dirs.append(expanded)
    return [path for path in dirs if os.path.isdir(path)]


def _discover_google_fonts(options: dict) -> list[dict]:
    dirs = _google_fonts_cache_dirs(options)
    if not dirs:
        return []
    cache_options = dict(options)
    cache_options["font_dirs"] = dirs
    cache_options["font_files"] = []
    return _discover_fonts(cache_options)


def _candidate_key(item: dict) -> tuple:
    return (
        str(item.get("family", "")).lower(),
        str(item.get("style", "")).lower(),
        int(item.get("weight", 400) or 400),
        str(item.get("source", "")).lower(),
    )


def _merge_font_candidates(*groups: Iterable[dict], top_k: int) -> list[dict]:
    merged = []
    seen = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            key = _candidate_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
            if len(merged) >= top_k:
                return merged
    return merged


def _font_options(config: dict) -> dict:
    raw = config.get("font_matching", False)
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("enabled", True)
        return out
    return {"enabled": False}


def _platform_font_dirs() -> list[str]:
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Library", "Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.join(home, ".fonts"),
        os.path.join(home, ".local", "share", "fonts"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def _name_record(font, name_id: int) -> Optional[str]:
    try:
        records = font["name"].names
    except Exception:
        return None
    for record in records:
        if record.nameID != name_id:
            continue
        try:
            return record.toUnicode().strip()
        except Exception:
            try:
                return record.string.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
    return None


def _font_metadata(path: str) -> dict:
    cached = _FONT_META_CACHE.get(path)
    if cached is not None:
        return dict(cached)
    stem = os.path.splitext(os.path.basename(path))[0]
    family = re.sub(r"[-_](thin|light|regular|medium|semibold|semi-bold|bold|black|italic).*$",
                    "", stem, flags=re.IGNORECASE).replace("_", " ")
    lower = stem.lower()
    italic = "italic" in lower or "oblique" in lower
    if "black" in lower:
        weight = 900
    elif "extra" in lower and "bold" in lower:
        weight = 800
    elif "semi" in lower and "bold" in lower:
        weight = 600
    elif "bold" in lower:
        weight = 700
    elif "medium" in lower:
        weight = 500
    elif "light" in lower:
        weight = 300
    elif "thin" in lower:
        weight = 200
    else:
        weight = 400
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(path, lazy=True, fontNumber=0)
        family = _name_record(font, 16) or _name_record(font, 1) or family
        subfamily = _name_record(font, 17) or _name_record(font, 2)
        if subfamily:
            italic = italic or "italic" in subfamily.lower() or "oblique" in subfamily.lower()
        try:
            weight = int(font["OS/2"].usWeightClass)
        except Exception:
            pass
        font.close()
    except Exception:
        pass
    meta = {
        "path": path,
        "family": family.strip() or stem,
        "weight": max(100, min(900, int(weight))),
        "style": _style_name(weight, italic),
    }
    _FONT_META_CACHE[path] = dict(meta)
    return meta


def _discover_fonts(options: dict) -> list[dict]:
    explicit = options.get("font_files") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    dirs = options.get("font_dirs") if "font_dirs" in options else _platform_font_dirs()
    if isinstance(dirs, str):
        dirs = [dirs]
    family_filter = options.get("families") or []
    if isinstance(family_filter, str):
        family_filter = [family_filter]
    key = (tuple(sorted(map(os.path.abspath, explicit))),
           tuple(sorted(map(os.path.abspath, dirs))),
           tuple(str(v).lower() for v in family_filter),
           int(options.get("scan_limit", 3000)))
    if key in _FONT_DISCOVERY_CACHE:
        return [dict(item) for item in _FONT_DISCOVERY_CACHE[key]]

    paths = [os.path.abspath(os.path.expanduser(path)) for path in explicit if os.path.isfile(os.path.expanduser(path))]
    scan_limit = max(1, int(options.get("scan_limit", 3000)))
    for root in dirs:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue
        for current, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename.lower().endswith((".ttf", ".otf", ".ttc")):
                    paths.append(os.path.join(current, filename))
                    if len(paths) >= scan_limit:
                        break
            if len(paths) >= scan_limit:
                break
        if len(paths) >= scan_limit:
            break

    unique = []
    seen = set()
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            unique.append(real)
            seen.add(real)

    metas = [_font_metadata(path) for path in unique]
    filters = [str(value).lower() for value in family_filter]
    if filters:
        metas = [m for m in metas if any(value in (m["family"] + " " + m["path"]).lower()
                                             for value in filters)]

    preferred = ["inter", "arial", "helvetica", "roboto", "dejavu", "liberation", "noto"]

    def rank(meta):
        haystack = (meta["family"] + " " + meta["path"]).lower()
        pref = next((idx for idx, value in enumerate(preferred) if value in haystack), len(preferred))
        return pref, meta["family"].lower(), meta["weight"], meta["path"]

    metas.sort(key=rank)
    _FONT_DISCOVERY_CACHE[key] = [dict(item) for item in metas]
    return metas


def _tight_mask(mask):
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    if arr.size == 0 or not arr.any():
        return None
    ys, xs = np.nonzero(arr)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_font_mask(text: str, path: str, size: float):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        font = ImageFont.truetype(path, max(1, int(round(size))))
        probe = Image.new("L", (8, 8), 0)
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        canvas = Image.new("L", (width + 8, height + 8), 0)
        ImageDraw.Draw(canvas).text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
        return _tight_mask(np.asarray(canvas) > 32)
    except Exception:
        return None


def _resize_mask(mask, width: int = 160, height: int = 64):
    from PIL import Image
    import numpy as np

    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _font_similarity(source, rendered) -> float:
    import numpy as np

    source, rendered = _tight_mask(source), _tight_mask(rendered)
    if source is None or rendered is None:
        return 0.0
    src_ratio = source.shape[1] / max(1.0, source.shape[0])
    rnd_ratio = rendered.shape[1] / max(1.0, rendered.shape[0])
    ratio_penalty = min(1.0, abs(math.log(max(1e-4, src_ratio / max(1e-4, rnd_ratio)))) / 1.2)
    aa, bb = _resize_mask(source), _resize_mask(rendered)
    mae = float(np.mean(np.abs(aa - bb)))
    a_bin, b_bin = aa >= 0.35, bb >= 0.35
    union = float(np.logical_or(a_bin, b_bin).sum())
    iou = float(np.logical_and(a_bin, b_bin).sum()) / union if union else 0.0
    density_penalty = min(1.0, abs(float(source.mean()) - float(rendered.mean())) * 3.0)
    loss = 0.48 * mae + 0.30 * (1.0 - iou) + 0.17 * ratio_penalty + 0.05 * density_penalty
    return round(max(0.0, min(1.0, 1.0 - loss)), 4)


def _match_fonts(text: str, source_mask, estimated_size: float, options: dict,
                 profile: Optional[dict] = None, fonts: Optional[list[dict]] = None,
                 source_label: str = "local-render") -> list[dict]:
    import numpy as np

    tight = _tight_mask(source_mask)
    if tight is None or not text.strip():
        return []
    profile = profile or {"weight": 400, "italic": False, "font_size": estimated_size}
    max_fonts = max(1, min(256, int(options.get("max_fonts", 48))))
    top_k = max(1, min(12, int(options.get("top_k", 5))))
    if fonts is None:
        fonts = _discover_fonts(options)[:max_fonts]
    else:
        fonts = fonts[:max_fonts]
    if not fonts:
        return []

    fingerprint_mask = _resize_mask(tight, 48, 20)
    fingerprint = hashlib.sha1((fingerprint_mask * 255).astype(np.uint8).tobytes()).hexdigest()[:16]
    cache_key = (text, fingerprint, round(float(estimated_size), 1),
                 tuple((item["path"], item["weight"]) for item in fonts), top_k, source_label,
                 profile.get("weight"), profile.get("italic"))
    if not options.get("repair_pass") and not options.get("force_rematch"):
        cached = _FONT_MATCH_CACHE.get(cache_key)
        if cached is not None:
            _FONT_MATCH_CACHE.move_to_end(cache_key)
            return [dict(item) for item in cached]

    scored = []
    target_height = tight.shape[0]
    for meta in fonts:
        rendered = _render_font_mask(text, meta["path"], estimated_size)
        if rendered is None:
            continue
        adjusted = estimated_size * target_height / max(1.0, rendered.shape[0])
        rendered = _render_font_mask(text, meta["path"], adjusted)
        score = _font_similarity(tight, rendered)
        score = round(max(0.0, min(1.0, score + _meta_alignment_adjustment(meta, profile))), 4)
        scored.append({
            "family": meta["family"],
            "style": meta["style"],
            "weight": meta["weight"],
            "score": score,
            "source": source_label,
            "path": meta["path"],
        })
    scored.sort(key=lambda item: (-item["score"], item["family"], item["weight"]))
    deduped = []
    seen = set()
    for item in scored:
        key = (item["family"].lower(), item["style"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    if not options.get("repair_pass") and not options.get("force_rematch"):
        _FONT_MATCH_CACHE[cache_key] = [dict(item) for item in deduped]
        _FONT_MATCH_CACHE.move_to_end(cache_key)
        while len(_FONT_MATCH_CACHE) > _FONT_MATCH_CACHE_LIMIT:
            _FONT_MATCH_CACHE.popitem(last=False)
    return deduped


def _resolve_font_candidates(text: str, source_mask, geo: dict, options: dict) -> list[dict]:
    top_k = max(1, min(12, int(options.get("top_k", 5))))
    profile = _typography_profile(geo)
    estimated_size = profile["font_size"]
    if options.get("repair_pass") or options.get("force_rematch"):
        _FONT_MATCH_CACHE.clear()

    local = _match_fonts(text, source_mask, estimated_size, options, profile=profile)
    google = []
    if _google_fonts_cache_dirs(options):
        google_fonts = _discover_google_fonts(options)[:max(1, min(256, int(options.get("max_fonts", 48))))]
        if google_fonts:
            google = _match_fonts(
                text, source_mask, estimated_size, options, profile=profile,
                fonts=google_fonts, source_label="google-cache",
            )
    fallback_slots = max(0, top_k - len(local) - len(google))
    fallback = _fallback_font_candidates(
        profile["weight"], options, max(1, fallback_slots or top_k), italic=profile["italic"],
    )
    return _merge_font_candidates(local, google, fallback, top_k=top_k)


def local_score_threshold(cfg: Optional[dict]) -> float:
    options = _font_options(_text_cfg(cfg))
    raw = options.get("local_score_threshold", _DEFAULT_LOCAL_SCORE_THRESHOLD)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_LOCAL_SCORE_THRESHOLD


def needs_vlm_font_judge(ocr_result: dict, cfg: Optional[dict] = None) -> bool:
    """True when font matching ran but the best local/google render score is weak."""
    if not _font_options(_text_cfg(cfg)).get("enabled"):
        return False
    threshold = local_score_threshold(cfg)
    for line in ocr_result.get("lines") or []:
        style = line.get("style") or {}
        render_candidates = [
            item for item in (style.get("fontCandidates") or [])
            if isinstance(item, dict) and item.get("source") in {"local-render", "google-cache"}
        ]
        if not render_candidates:
            continue
        top_score = max(float(item.get("score", 0.0) or 0.0) for item in render_candidates)
        if top_score < threshold:
            return True
    return False


def _pre_font_signals(line: dict, painted: dict, mask, config: dict) -> dict:
    """Signals available cheaply, before any (expensive) font-file rendering: the
    estimated size/weight and a glyph-shear (italic) measurement from the ink mask
    alone. Used both to assemble the final style and to cluster same-style lines
    before font matching runs."""
    cap_ratio = max(0.45, min(0.90, _num(config.get("cap_height_ratio"), 0.72)))
    font_size = max(1.0, min(512.0, painted["h"] / cap_ratio if painted["h"] else line["box"]["h"] * 0.9))
    weight = _estimate_weight(mask, painted)
    shear_angle = _measure_shear_angle(mask)
    return {"font_size": font_size, "weight": weight, "shear_angle": shear_angle}


def _style_cluster_key(geo: dict, colour: str) -> tuple:
    """Coarse style bucket usable before font matching: same-style lines (same
    rounded size/weight/slant/colour) should share one matched font instead of
    each independently spending the font-matching budget."""
    rgb = _hex_rgb(colour)
    colour_bucket = tuple(int(round(value / 16.0) * 16) for value in rgb)
    size_bucket = round(geo["font_size"] / 2.0) * 2
    weight_bucket = int(round(geo["weight"] / 100.0) * 100)
    shear = geo.get("shear_angle")
    italic_bucket = bool(shear is not None and abs(shear) >= 6.0)
    return (size_bucket, weight_bucket, italic_bucket, colour_bucket)


def _base_style(line: dict, painted: dict, colour: str, ink_confidence: float,
                mask, config: dict, font_options: dict, geo: dict,
                preset_candidates: Optional[list[dict]] = None, paint: Optional[dict] = None) -> dict:
    font_size = geo["font_size"]
    weight = geo["weight"]
    shear_angle = geo.get("shear_angle")
    top_k = max(1, min(12, int(font_options.get("top_k", 5))))
    candidates = list(preset_candidates) if preset_candidates else []
    if not candidates:
        candidates = _fallback_font_candidates(
            geo["weight"], font_options, top_k,
            italic=bool(geo.get("shear_angle") is not None and abs(geo["shear_angle"]) >= 6.0),
        )
    chosen = candidates[0]
    weight = int(chosen.get("weight", weight)) if chosen.get("source") == "local-render" else weight
    match_italic = "italic" in str(chosen.get("style") or "").lower()
    measured_italic = shear_angle is not None and abs(shear_angle) >= _num(config.get("italic_shear_deg"), 6.0)
    italic = match_italic or measured_italic
    style = chosen.get("style") or _style_name(weight)
    if italic and "italic" not in style.lower():
        style = _style_name(weight, italic=True)
    primary_is_italic = "italic" in style.lower()
    alt_style = _style_name(weight, italic=not primary_is_italic)
    return {
        "fontFamily": chosen.get("family", "Inter"),
        "fontSize": round(font_size, 2),
        "fontWeight": weight,
        "fontStyle": style,
        "fontCandidates": candidates,
        "fontSizeCandidates": _size_candidates(font_size),
        "fontWeightCandidates": _weight_candidates(weight),
        "fontStyleCandidates": [
            {"value": style, "score": 0.72},
            {"value": alt_style, "score": 0.24},
        ],
        "italicShearDeg": shear_angle,
        "color": colour,
        "colorRGB": list(_hex_rgb(colour)),
        "align": "LEFT",
        "lineHeight": round(max(font_size * 1.15, painted["h"]), 2),
        "letterSpacing": _estimate_tracking(line.get("text", ""), painted, font_size),
        "confidence": round(min(_num(line.get("conf"), 0.5), max(0.25, ink_confidence)), 4),
        "fill": (paint or {}).get("fill") or {"kind": "flat", "color": colour},
        "stroke": (paint or {}).get("stroke"),
    }


# ---------------------------------------------------------------------------
# Roles, paragraph grouping, alignment, hierarchy and repeated styles


def _assign_roles(lines: list[dict], canvas: dict) -> None:
    sizes = [line["style"]["fontSize"] for line in lines]
    median_size = _median(sizes, 16.0)
    max_size = max(sizes, default=median_size)
    canvas_h = max(1.0, _num(canvas.get("h"), 1.0))
    for line in lines:
        text = str(line.get("text") or "").strip()
        size = line["style"]["fontSize"]
        y_ratio = line["box"]["y"] / canvas_h
        words = text.split()
        if _PRICE_RE.search(text):
            role = "price"
        elif _OFFER_RE.search(text) and len(words) <= 8:
            role = "offer"
        elif _CTA_RE.search(text) and len(words) <= 8:
            role = "cta"
        elif (
            size >= max(median_size * 1.45, max_size * 0.84)
            or (text.isupper() and len(words) >= 3 and y_ratio <= 0.35)
        ) and len(words) <= 18:
            role = "headline"
        elif size >= median_size * 1.18 and len(words) <= 24:
            role = "subheadline"
        elif len(text) >= 52 or len(words) >= 9:
            role = "body"
        elif y_ratio <= 0.13 and len(words) <= 5 and size <= median_size * 1.05:
            role = "eyebrow"
        elif y_ratio >= 0.86 and size <= median_size:
            role = "footer"
        elif size <= median_size * 0.78:
            role = "caption"
        else:
            role = "body"
        level = {
            "headline": 1,
            "subheadline": 2,
            "price": 2,
            "offer": 2,
            "eyebrow": 2,
            "body": 3,
            "cta": 3,
            "caption": 4,
            "footer": 4,
        }.get(role, 3)
        line["role"] = role
        line["hierarchy"] = {"level": level, "parent_id": None}


def _compatible_roles(a: str, b: str) -> bool:
    if a == b:
        return True
    families = [
        {"headline", "subheadline"},
        {"body", "caption"},
        {"offer", "price"},
    ]
    return any(a in family and b in family for family in families)


def _can_join(previous: dict, current: dict, config: dict) -> bool:
    a, b = previous["box"], current["box"]
    acy, bcy = _box_center(a)[1], _box_center(b)[1]
    if abs(acy - bcy) < min(a["h"], b["h"]) * 0.55:
        return False  # same visual row: usually separate columns/elements
    gap = b["y"] - (a["y"] + a["h"])
    max_gap = max(a["h"], b["h"]) * _num(config.get("paragraph_gap_factor"), 1.25)
    if gap < -min(a["h"], b["h"]) * 0.25 or gap > max_gap:
        return False
    size_ratio = max(previous["style"]["fontSize"], current["style"]["fontSize"]) / max(
        1.0, min(previous["style"]["fontSize"], current["style"]["fontSize"])
    )
    if size_ratio > _num(config.get("paragraph_size_ratio"), 1.35):
        return False
    if _colour_distance(previous["style"]["color"], current["style"]["color"]) > _num(
            config.get("paragraph_color_distance"), 90.0):
        return False
    if not _compatible_roles(previous["role"], current["role"]):
        return False
    aligned = (
        _horizontal_overlap(a, b) >= 0.22
        or abs(a["x"] - b["x"]) <= max(a["h"], b["h"]) * 1.15
        or abs(_box_center(a)[0] - _box_center(b)[0]) <= max(a["h"], b["h"]) * 1.5
        or abs((a["x"] + a["w"]) - (b["x"] + b["w"])) <= max(a["h"], b["h"]) * 1.15
    )
    return aligned


def _infer_alignment(lines: list[dict], canvas_w: float) -> str:
    if len(lines) == 1:
        box = lines[0]["box"]
        center = _box_center(box)[0]
        if box["w"] < canvas_w * 0.88 and abs(center - canvas_w / 2.0) <= canvas_w * 0.055:
            return "CENTER"
        if box["x"] >= canvas_w * 0.52 and abs((box["x"] + box["w"]) - canvas_w) <= canvas_w * 0.08:
            return "RIGHT"
        return "LEFT"
    lefts = [line["box"]["x"] for line in lines]
    rights = [line["box"]["x"] + line["box"]["w"] for line in lines]
    centers = [_box_center(line["box"])[0] for line in lines]
    spread = {
        "LEFT": statistics.pstdev(lefts),
        "RIGHT": statistics.pstdev(rights),
        "CENTER": statistics.pstdev(centers),
    }
    return min(spread, key=spread.get)


def _make_blocks(lines: list[dict], canvas: dict, config: dict) -> list[dict]:
    ordered = sorted(lines, key=lambda line: (line["box"]["y"], line["box"]["x"]))
    groups: list[list[dict]] = []
    for line in ordered:
        # Read top-to-bottom *within each text column*. A simple ``groups[-1]`` pass
        # cross-merges copy in a two-column layout because OCR is normally ordered by y,x.
        candidates = []
        for index, group in enumerate(groups):
            previous = group[-1]
            if not _can_join(previous, line, config):
                continue
            a, b = previous["box"], line["box"]
            gap = max(0.0, b["y"] - (a["y"] + a["h"]))
            left_delta = abs(a["x"] - b["x"])
            center_delta = abs(_box_center(a)[0] - _box_center(b)[0])
            # Column alignment is more meaningful than a marginally smaller vertical gap.
            score = (left_delta + 0.45 * center_delta + 0.35 * gap) / max(1.0, a["h"], b["h"])
            candidates.append((score, index))
        if candidates:
            _, index = min(candidates, key=lambda item: (item[0], item[1]))
            groups[index].append(line)
        else:
            groups.append([line])

    canvas_w = max(1.0, _num(canvas.get("w"), 1.0))
    blocks = []
    groups.sort(key=lambda group: (group[0]["box"]["y"], group[0]["box"]["x"]))
    for index, group in enumerate(groups):
        block_id = f"B{index}"
        alignment = _infer_alignment(group, canvas_w)
        baselines = [line["baseline"]["y0"] for line in group]
        deltas = [baselines[i + 1] - baselines[i] for i in range(len(baselines) - 1)]
        median_size = _median((line["style"]["fontSize"] for line in group), 16.0)
        line_height = max(median_size * 0.92, _median(deltas, median_size * 1.15))
        role_line = min(group, key=lambda line: line["hierarchy"]["level"])
        role = role_line["role"]
        level = role_line["hierarchy"]["level"]
        for line in group:
            line["block_id"] = block_id
            line["style"]["align"] = alignment
            line["style"]["lineHeight"] = round(line_height, 2)
            line["hierarchy"]["parent_id"] = block_id
        # Propagate the fidelity gate onto the block. A block is only as trustworthy as its
        # worst line: if any member line was flagged low_fidelity, the block must carry that
        # flag (and that line's fallback crop/reason) so downstream routing sees it — blocks,
        # not lines, are what merge_layers._text_sources actually emits as candidates.
        worst_line = min(
            group,
            key=lambda line: (line.get("meta") or {}).get("fidelity_confidence", 1.0),
        )
        worst_meta = worst_line.get("meta") or {}
        block_meta = {
            "fidelity_confidence": worst_meta.get("fidelity_confidence"),
            "low_fidelity": bool(worst_meta.get("low_fidelity")),
        }
        if worst_meta.get("low_fidelity"):
            block_meta["fidelity_reason"] = worst_meta.get("fidelity_reason")
            if worst_meta.get("fallback_src"):
                block_meta["fallback_src"] = worst_meta["fallback_src"]
            if worst_meta.get("substitution"):
                block_meta["substitution"] = worst_meta["substitution"]
        blocks.append({
            "id": block_id,
            "type": "paragraph" if len(group) > 1 else "text",
            "line_ids": [line["id"] for line in group],
            "text": "\n".join(line.get("text", "") for line in group),
            "box": _union_boxes(line["box"] for line in group),
            "painted_box": _union_boxes(line["painted_box"] for line in group),
            "alignment": alignment,
            "line_height": round(line_height, 2),
            "role": role,
            "hierarchy": {"level": level, "parent_id": None},
            "style_id": None,
            "meta": block_meta,
        })
    return blocks


def _make_sections(blocks: list[dict], canvas: dict) -> list[dict]:
    if not blocks:
        return []
    canvas_h = max(1.0, _num(canvas.get("h"), 1.0))
    threshold = max(24.0, canvas_h * 0.09)
    ordered = sorted(blocks, key=lambda block: (block["box"]["y"], block["box"]["x"]))
    groups = [[ordered[0]]]
    for block in ordered[1:]:
        previous = groups[-1][-1]["box"]
        gap = block["box"]["y"] - (previous["y"] + previous["h"])
        if gap <= threshold:
            groups[-1].append(block)
        else:
            groups.append([block])
    sections = []
    for index, group in enumerate(groups):
        section_id = f"S{index}"
        for block in group:
            block["section_id"] = section_id
            block["hierarchy"]["parent_id"] = section_id
        role_block = min(group, key=lambda block: block["hierarchy"]["level"])
        sections.append({
            "id": section_id,
            "type": "text-section",
            "block_ids": [block["id"] for block in group],
            "box": _union_boxes(block["box"] for block in group),
            "role": role_block["role"],
            "hierarchy": {"level": 0, "parent_id": "text-root"},
        })
    return sections


def _style_key(style: dict) -> tuple:
    rgb = _hex_rgb(style.get("color", "#000000"))
    colour_bucket = tuple(int(round(value / 8.0) * 8) for value in rgb)
    return (
        str(style.get("fontFamily", "Inter")).lower(),
        round(_num(style.get("fontSize"))),
        int(round(_num(style.get("fontWeight"), 400) / 100.0) * 100),
        str(style.get("fontStyle", "Regular")).lower(),
        colour_bucket,
        str(style.get("align", "LEFT")),
        round(_num(style.get("lineHeight"))),
        round(_num(style.get("letterSpacing")) * 2.0) / 2.0,
    )


_DEFAULT_FIDELITY_MIN_CONFIDENCE = 0.30


def _save_fallback_crop(image, mask, painted_box: dict, run_dir: Optional[str], line_id: str) -> Optional[str]:
    """Persist the original painted pixels (ink mask baked into the alpha channel)
    so a low-fidelity line can render as a masked-pixel fallback layer instead of
    guessed text. Returns a run_dir-relative path, or None if it cannot be saved
    (missing run_dir/image — the caller still records the substitution note)."""
    if image is None or mask is None or not run_dir:
        return None
    try:
        import numpy as np
        from PIL import Image

        x0 = max(0, int(round(painted_box["x"])))
        y0 = max(0, int(round(painted_box["y"])))
        w = max(1, int(round(painted_box["w"])))
        h = max(1, int(round(painted_box["h"])))
        x1 = min(image.shape[1], x0 + w)
        y1 = min(image.shape[0], y0 + h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = image[y0:y1, x0:x1]
        m = np.asarray(mask)
        if m.shape[:2] != crop.shape[:2]:
            return None
        alpha = (m.astype(np.uint8) * 255)
        rgba = np.dstack([crop, alpha]).astype(np.uint8)
        out_dir = os.path.join(run_dir, "text_fallback")
        os.makedirs(out_dir, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(line_id)) or "line"
        path = os.path.join(out_dir, f"{safe_id}.png")
        Image.fromarray(rgba, mode="RGBA").save(path)
        return os.path.relpath(path, run_dir)
    except Exception:
        return None


def _assign_style_ids(lines: list[dict], blocks: list[dict]) -> list[dict]:
    by_key: "OrderedDict[tuple, dict]" = OrderedDict()
    for line in lines:
        key = _style_key(line["style"])
        if key not in by_key:
            style_id = f"TS{len(by_key)}"
            public = {
                name: copy.deepcopy(value)
                for name, value in line["style"].items()
                if name not in {"fontCandidates", "fontSizeCandidates", "fontWeightCandidates",
                                "fontStyleCandidates", "confidence"}
            }
            by_key[key] = {"id": style_id, **public, "usage": []}
        entry = by_key[key]
        entry["usage"].append(line["id"])
        line["style_id"] = entry["id"]
    by_line = {line["id"]: line for line in lines}
    for block in blocks:
        ids = [by_line[line_id]["style_id"] for line_id in block["line_ids"]]
        block["style_id"] = max(set(ids), key=ids.count) if ids else None
    styles = list(by_key.values())
    for style in styles:
        style["repeated"] = len(style["usage"]) > 1
    return styles


# ---------------------------------------------------------------------------
# Public API


def analyze_text(img_path: str, ocr_result: dict, cfg: Optional[dict] = None) -> dict:
    """Enrich an OCR result without mutating it.

    Parameters
    ----------
    img_path:
        Source image used by OCR.  A missing/unreadable image is allowed; the
        function falls back to OCR geometry and conservative typography.
    ocr_result:
        Existing OCR-shaped mapping (``engine/source/ms/lines``).
    cfg:
        Pipeline configuration; options live under ``text_analysis``.

    Returns
    -------
    dict
        Original OCR mapping plus enriched ``lines`` and top-level ``blocks``,
        ``styles``, ``sections`` and ``hierarchy``.
    """
    started = time.perf_counter()
    result = copy.deepcopy(ocr_result or {})
    config = _text_cfg(cfg)
    font_options = _font_options(config)
    image = _load_rgb(img_path)

    source = dict(result.get("source") or {})
    if image is not None:
        source.setdefault("w", int(image.shape[1]))
        source.setdefault("h", int(image.shape[0]))
    source.setdefault("path", img_path)
    source.setdefault("w", 0)
    source.setdefault("h", 0)
    result["source"] = source
    canvas = {"w": source.get("w", 0), "h": source.get("h", 0)}

    run_dir = None
    if isinstance(cfg, dict):
        run_dir = cfg.get("run_dir")
    fidelity_min_confidence = _num(config.get("fidelity_min_confidence"), _DEFAULT_FIDELITY_MIN_CONFIDENCE)

    raw_lines = result.get("lines") or []
    max_match_lines = max(0, min(100, int(font_options.get("max_lines", 12))))

    # Pass 1: cheap per-line geometry (painted bounds, ink mask, paint/shear signals).
    # No font-file rendering happens here.
    prepared = []
    masks: dict[str, Any] = {}
    for index, raw in enumerate(raw_lines):
        line = copy.deepcopy(raw)
        line.setdefault("id", f"L{index}")
        line["box"] = _clean_box(line.get("box"))
        painted, baseline, colour, ink_confidence, mask, paint = _painted_geometry(image, line)
        line["painted_box"] = painted
        line["baseline"] = baseline
        line["rotation"] = _quad_rotation(line.get("quad"))
        line["rotation_deg"] = line["rotation"]
        line["ink_confidence"] = ink_confidence
        masks[line["id"]] = mask
        geo = _pre_font_signals(line, painted, mask, config)
        prepared.append({
            "line": line, "painted": painted, "colour": colour,
            "ink_confidence": ink_confidence, "mask": mask, "paint": paint, "geo": geo,
        })

    # Pass 2: font matching by style-cluster representative. Lines that share a
    # coarse style bucket (size/weight/slant/colour) reuse one representative's
    # match instead of each independently spending the match budget — so a
    # >max_lines-line ad still gets a real font on every same-style line, not
    # just the first `max_lines` lines encountered.
    preset_by_index: dict[int, list[dict]] = {}
    match_count = 0
    if font_options.get("enabled") and prepared:
        clusters: "OrderedDict[tuple, list[int]]" = OrderedDict()
        for i, item in enumerate(prepared):
            key = _style_cluster_key(item["geo"], item["colour"])
            clusters.setdefault(key, []).append(i)
        min_ink = _num(font_options.get("min_ink_confidence"), 0.25)
        for idxs in clusters.values():
            if match_count >= max_match_lines:
                break
            representative = max(idxs, key=lambda i: prepared[i]["ink_confidence"])
            match_count += 1
            rep_item = prepared[representative]
            if rep_item["ink_confidence"] < min_ink:
                continue
            candidates = _resolve_font_candidates(
                rep_item["line"].get("text", ""), rep_item["mask"],
                rep_item["geo"], font_options,
            )
            if candidates:
                for i in idxs:
                    preset_by_index[i] = candidates

    # Pass 3: assemble the final per-line style, and gate low-fidelity lines
    # (poor ink isolation or a poor font/effect match) to a masked-pixel
    # fallback instead of emitting a guessed font rendering downstream.
    enriched = []
    for i, item in enumerate(prepared):
        line = item["line"]
        preset = preset_by_index.get(i)
        line["style"] = _base_style(
            line, item["painted"], item["colour"], item["ink_confidence"], item["mask"],
            config, font_options, item["geo"], preset_candidates=preset, paint=item["paint"],
        )

        font_confidence = None
        candidates = line["style"].get("fontCandidates") or []
        render_candidates = [
            item for item in candidates
            if isinstance(item, dict) and item.get("source") in {"local-render", "google-cache"}
        ]
        if render_candidates:
            font_confidence = max(float(item.get("score", 0.0) or 0.0) for item in render_candidates)
        fidelity_confidence = item["ink_confidence"] if font_confidence is None else min(
            item["ink_confidence"], font_confidence
        )
        low_fidelity = fidelity_confidence < fidelity_min_confidence
        meta = line.setdefault("meta", {})
        meta["fidelity_confidence"] = round(fidelity_confidence, 4)
        meta["low_fidelity"] = bool(low_fidelity)
        if low_fidelity:
            reasons = []
            if item["ink_confidence"] < fidelity_min_confidence:
                reasons.append(f"ink_confidence:{item['ink_confidence']:.2f}<{fidelity_min_confidence:.2f}")
            if font_confidence is not None and font_confidence < fidelity_min_confidence:
                reasons.append(f"font_confidence:{font_confidence:.2f}<{fidelity_min_confidence:.2f}")
            reason = "; ".join(reasons) or "low-confidence font/effect match"
            meta["fidelity_reason"] = reason
            fallback_src = _save_fallback_crop(image, item["mask"], item["painted"], run_dir, line["id"])
            if fallback_src:
                meta["fallback_src"] = fallback_src
            meta["substitution"] = {
                "from": "text", "to": "masked-pixel-fallback",
                "reason": reason, "confidence": meta["fidelity_confidence"],
            }

        enriched.append(line)

    _assign_roles(enriched, canvas)
    blocks = _make_blocks(enriched, canvas, config)
    sections = _make_sections(blocks, canvas)
    styles = _assign_style_ids(enriched, blocks)

    result["lines"] = enriched
    result["blocks"] = blocks
    result["styles"] = styles
    result["sections"] = sections
    result["hierarchy"] = {
        "id": "text-root",
        "type": "text-root",
        "children": [section["id"] for section in sections],
    }
    result["text_analysis"] = {
        "version": 1,
        "ms": round((time.perf_counter() - started) * 1000.0, 2),
        "font_matching": bool(font_options.get("enabled")),
        "font_matches_attempted": match_count,
        "image_available": image is not None,
    }
    return result


def run_text_analysis(img_path: str, ocr_result: dict, cfg: Optional[dict] = None) -> dict:
    """Stage-style alias for :func:`analyze_text`."""
    return analyze_text(img_path, ocr_result, cfg)


__all__ = ["analyze_text", "run_text_analysis", "needs_vlm_font_judge", "local_score_threshold"]
