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
        google_fonts_cache: ~/.cache/google-fonts   # optional on-disk OFL corpus

Matched families are always relabelled to a Figma-loadable Google Fonts family
of the same class (Figma can natively load Google Fonts but not local
Windows-only faces), so the emitted ``fontFamily`` is always editable in Figma
while all measured styling is preserved.  See ``_figma_google_family`` and the
``GOOGLE_FONTS_FAMILIES`` / ``_LOCAL_TO_GOOGLE`` tables.

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

_DEFAULT_FAMILIES = ["Inter", "Roboto", "Open Sans", "Lato", "Montserrat"]
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
    """Return the text baseline angle from an OCR quadrilateral.

    OCR providers do not agree on quad winding.  In particular, some emit the
    short side first, which made ordinary horizontal lines look vertical.  The
    text direction is the longer of the two pairs of opposite edges.
    """
    try:
        points = [(float(point[0]), float(point[1])) for point in quad[:4]]
        edges = []
        for index in range(4):
            x0, y0 = points[index]
            x1, y1 = points[(index + 1) % 4]
            dx, dy = x1 - x0, y1 - y0
            edges.append((math.hypot(dx, dy), dx, dy))
        # Opposite edges describe the same text direction.  Use their longer
        # pair and average their directed vectors after making them agree.
        pair = (edges[0], edges[2]) if edges[0][0] + edges[2][0] >= edges[1][0] + edges[3][0] else (edges[1], edges[3])
        _, dx, dy = pair[0]
        _, odx, ody = pair[1]
        if dx * odx + dy * ody < 0:
            odx, ody = -odx, -ody
        angle = math.degrees(math.atan2(dy + ody, dx + odx))
    except (TypeError, ValueError, IndexError):
        return 0.0
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return round(angle, 3)


_DEFAULT_ROTATION_SNAP_DEG = 2.5


def _snap_rotation(angle: float, snap_deg: float) -> float:
    """Snap a near-zero baseline angle to exactly horizontal.

    OCR quads on perfectly horizontal ad copy routinely wobble by a degree or
    two; rendering that wobble skews text that the source paints straight.  A
    genuinely rotated element keeps its angle (|angle| >= snap_deg).
    """
    if snap_deg > 0 and abs(float(angle)) < float(snap_deg):
        return 0.0
    return float(angle)


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


def _fallback_geometry(line: dict, snap_deg: float = 0.0) -> tuple[dict, dict, float, None]:
    box = _clean_box(line.get("box"))
    rotation = _snap_rotation(_quad_rotation(line.get("quad")), snap_deg)
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

    core = _erode_mask(mask)
    if int(core.sum()) >= max(24, int(np.asarray(mask).sum() * .35)):
        mask = core
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
    outside = ~mask
    if outside.any():
        background_rgb = np.median(crop[outside].astype(np.float32), axis=0)
        axis = interior_rgb - background_rgb
        norm = float(np.dot(axis, axis))
        if norm > 1:
            blend = float(np.dot(rim_rgb - background_rgb, axis) / norm)
            predicted = background_rgb + max(0.0, min(1.0, blend)) * axis
            # A normal anti-aliased edge is simply a blend between the glyph fill and
            # its background. It is not an authored outline and must not become a Figma
            # stroke that visibly fattens the reconstructed text.
            if 0.03 < blend < 0.97 and float(np.linalg.norm(rim_rgb - predicted)) < 18.0:
                return None
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
    angle = math.degrees(math.atan(best_shift / dy))
    # The signal is intended to identify italic shear, not compensate for a
    # rotated text box.  Large values are overwhelmingly cross-correlation
    # artefacts (or box rotation) and would incorrectly force an italic font.
    if abs(angle) > 20.0:
        return None
    return round(angle, 2)


def _painted_geometry(image, line: dict, snap_deg: float = 0.0) -> tuple[dict, dict, str, float, Any, dict]:
    import numpy as np

    box = _clean_box(line.get("box"))
    if image is None or box["w"] <= 0 or box["h"] <= 0:
        painted, baseline, confidence, mask = _fallback_geometry(line, snap_deg)
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
        painted, baseline, fallback_conf, _ = _fallback_geometry(line, snap_deg)
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
    rotation = _snap_rotation(_quad_rotation(line.get("quad")), snap_deg)
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


def _native_text_decoration(mask, text: str) -> tuple[Optional[str], Optional[dict]]:
    """Recognize only an unmistakable continuous underline/strike rule.

    Short glyph bars (E, T, hyphens) must not turn into Figma text decoration.  We
    therefore require a nearly continuous run spanning most of the painted text width.
    Anything ambiguous remains part of the exact text fallback/plate pixels.
    """
    if mask is None or not str(text or "").strip() or "_" in str(text):
        return None, None
    try:
        import numpy as np

        ink = np.asarray(mask, dtype=bool)
        if ink.ndim != 2 or min(ink.shape) < 3:
            return None, None
        h, w = ink.shape
        if w < 12:
            return None, None
        longest = []
        for row in ink:
            padded = np.pad(row.astype(np.int8), (1, 1))
            edges = np.diff(padded)
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            longest.append(int(np.max(ends - starts)) if len(starts) else 0)
        longest = np.asarray(longest)
        strong = longest >= max(10, int(round(w * 0.72)))
        runs = []
        start = None
        for idx, value in enumerate(np.r_[strong, False]):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx - 1))
                start = None
        if not runs:
            return None, None
        y0, y1 = max(runs, key=lambda pair: longest[pair[0]:pair[1] + 1].max())
        thickness = y1 - y0 + 1
        if thickness > max(3, int(round(h * 0.13))):
            return None, None
        centre = (y0 + y1) / 2.0 / max(1.0, h - 1)
        if centre >= 0.80:
            kind = "UNDERLINE"
        elif 0.34 <= centre <= 0.66:
            kind = "STRIKETHROUGH"
        else:
            return None, None
        confidence = min(1.0, float(longest[y0:y1 + 1].max()) / max(1.0, w))
        return kind, {
            "source": "continuous-source-rule", "confidence": round(confidence, 4),
            "relative_y": round(centre, 4), "thickness_px": thickness,
            "mask_rows": [int(y0), int(y1)],
        }
    except Exception:
        return None, None


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


# ---------------------------------------------------------------------------
# Text-box fitting for the Figma compiler
#
# A design.json text layer is rendered into a fixed box.  When the box is narrower
# than the rendered glyph run the text clips on the right (the ad9 defect).
# ``fit_text_box`` returns a box that fully contains the text plus a Figma
# ``autoResize`` hint: single-line labels grow width from their alignment anchor;
# fixed-width wrapped paragraphs keep their width and shrink fontSize/letterSpacing
# until every line fits, growing height instead.


def _fit_font(style: dict, font_size: float):
    try:
        from PIL import ImageFont
    except Exception:
        return None
    candidates = style.get("fontCandidates") or []
    paths = [c.get("path") for c in candidates
             if isinstance(c, dict) and c.get("path") and os.path.exists(c["path"])]
    paths += ["arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
    size = max(1, int(round(font_size)))
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def _line_advance(font, line: str, tracking: float) -> float:
    """Rendered width of one line honoring per-glyph tracking (letterSpacing)."""
    if not line:
        return 0.0
    try:
        width = sum(font.getlength(ch) for ch in line)
    except Exception:
        width = len(line) * 0.5 * float(getattr(font, "size", 12) or 12)
    return width + tracking * max(0, len(line) - 1)


def _glyph_height(font, lines: list[str], fallback: float) -> float:
    """Visible glyph height, excluding the font's usually-large line gap.

    OCR boxes describe painted ink, while ``getmetrics`` describes a line cell.  Fitting
    the latter into an ink box over-shrinks type; growing the box to fit it caused the
    opposite ad9 failure.  Pillow's glyph bounds are the closest deterministic proxy used
    by both the design builder and preview renderer.
    """
    heights = []
    for line in lines:
        if not line:
            continue
        try:
            bounds = font.getbbox(line)
            heights.append(max(0.0, float(bounds[3] - bounds[1])))
        except Exception:
            pass
    return max(heights, default=max(1.0, float(fallback)))


def fit_text_box(text: str, style: dict, box: dict) -> tuple[dict, str, dict]:
    """Return ``(box, auto_resize, style_patch)`` sized so ``text`` cannot clip.

    The input box is painted source geometry, not a disposable layout suggestion.  Font
    substitutions therefore shrink to that geometry instead of expanding it by hundreds
    of pixels. ``auto_resize`` remains a Figma hint (``WIDTH`` for labels, ``HEIGHT`` for
    paragraphs); ``style_patch`` carries the fitted size/tracking used by both preview and
    Figma.
    """
    fitted = dict(box or {})
    text = str(text or "")
    lines = text.split("\n")
    font_size = _num(style.get("fontSize"), max(1.0, _num(fitted.get("h"), 12.0)))
    if font_size <= 0:
        font_size = max(1.0, _num(fitted.get("h"), 12.0))
    # CODIA-PARITY POLICY: letterSpacing is ALWAYS 0 in the emitted style. Codia ships
    # tracking 0 on every text node; fitted tracking was measurement noise that made
    # renders worse than the naive choice (spec §2/§7). Width error is absorbed by the
    # box/fontSize, never by tracking. Measurement below therefore uses 0 as well so
    # the fitted fontSize reflects the final (untracked) render.
    tracking = 0.0
    line_height = _num(style.get("lineHeight"), font_size * 1.2) or font_size * 1.2
    align = str(style.get("align", "LEFT")).upper()
    font = _fit_font(style, font_size)
    if font is None:
        return fitted, "NONE", {"letterSpacing": 0.0}
    bounded_tracking = 0.0
    widths = [_line_advance(font, line, bounded_tracking) for line in lines]
    content_w = max(widths + [0.0])
    line_count = max(1, len(lines))
    patch: dict = {}
    avail_w = max(1.0, _num(fitted.get("w"), content_w or 1.0))
    glyph_h = _glyph_height(font, lines, font_size)
    content_h = (line_count - 1) * line_height + glyph_h
    avail_h = max(1.0, _num(fitted.get("h"), content_h or 1.0))
    width_scale = min(1.0, avail_w / max(1.0, content_w))
    # Preserve measured inter-line spacing: only the glyph portion scales.  This keeps
    # OCR-derived baselines stable while making substituted glyphs fit the painted box.
    glyph_room = max(1.0, avail_h - (line_count - 1) * line_height)
    height_scale = min(1.0, glyph_room / max(1.0, glyph_h))
    target_scale = min(width_scale, height_scale)
    if target_scale < 0.999:
        new_size = max(1.0, font_size * target_scale)
        patch["fontSize"] = round(new_size, 2)
        # Multiline OCR boxes carry a measured line height for the original font.
        # Keeping that absolute value after shrinking a substitute font is a common
        # source of clipped final lines in both the preview and Figma.
        if line_count > 1:
            patch["lineHeight"] = round(max(new_size, line_height * target_scale), 2)

    # At very small sizes Pillow rounds to whole-pixel font sizes, so the linear estimate
    # can still overshoot by a pixel.  A short bounded refinement keeps the contract exact
    # without opening the box; tracking is never used to chase the painted width.
    for _ in range(4):
        effective_size = patch.get("fontSize", font_size)
        fit_font = _fit_font({**style, **patch}, effective_size) or font
        current = max((_line_advance(fit_font, line, 0.0) for line in lines), default=0.0)
        if current <= avail_w + 0.25:
            break
        ratio = max(0.5, min(0.99, avail_w / max(1.0, current)))
        new_size = max(1.0, effective_size * ratio)
        patch["fontSize"] = round(new_size, 2)

    # Emit the tracking policy explicitly so preview, plugin and parity all read 0.
    if _num(style.get("letterSpacing"), 0.0) != 0.0:
        patch["letterSpacing"] = 0.0
    return fitted, "WIDTH" if line_count <= 1 else "HEIGHT", patch


def _fallback_font_candidates(weight: int, options: dict, top_k: int, italic: bool = False) -> list[dict]:
    families = options.get("fallback_families") or options.get("families") or _DEFAULT_FAMILIES
    if isinstance(families, str):
        families = [families]
    out = []
    for index, family in enumerate(families):
        # Fallbacks must also be Figma-loadable: remap any configured non-Google
        # family (e.g. Arial/Helvetica) to its Google equivalent by name.
        gfam, kind = _figma_google_family(family)
        entry = {
            "family": gfam,
            "style": _style_name(weight, italic=italic),
            "weight": int(weight),
            "score": round(max(0.25, 0.62 - index * 0.07), 3),
            "source": "fallback",
            "figma_loadable": True,
            "figma_font_source": kind,
        }
        if gfam != str(family):
            entry["local_family"] = str(family)
        out.append(entry)
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


# ---------------------------------------------------------------------------
# Google Fonts inventory + license-clean local->Google mapping
#
# Figma natively loads any Google Fonts family but NOT local Windows-only fonts,
# so matching to a Google family is strictly better for editability: the emitted
# ``fontFamily`` is one Figma can actually render. Two license-clean sources
# feed this (neither depends on the non-commercial Lens weights rejected in
# docs/FONT-MATCHER-EVAL.md; Google Fonts are OFL/Apache = free/commercial-OK):
#
#   * an on-disk Google Fonts corpus under the ``google_fonts_cache`` path —
#     matched natively when present (see ``_discover_google_fonts``); AND
#   * the curated inventory + local->Google mapping below, which needs NO font
#     files on disk: it substitutes the *reported* family for a Figma-loadable
#     Google equivalent of the SAME CLASS while the local .ttf is still used to
#     render and score the fit, so all styling (size/weight/tracking/leading/
#     colour) is preserved.
#
# One-time OFL corpus install (optional — matching works without it via the
# mapping): ``git clone --depth 1 https://github.com/google/fonts \
# ~/.cache/google-fonts`` then leave ``google_fonts_cache`` at its default.

# Curated Google Fonts families covering the bulk of ad typography. Used to
# (a) recognise a matched family that is ALREADY a Google font (emitted
# unchanged and preferred in ranking) and (b) validate every mapping target
# below. Not exhaustive: an on-disk corpus is matched in full regardless.
GOOGLE_FONTS_FAMILIES = (
    # grotesque / geometric / humanist sans
    "Inter", "Roboto", "Open Sans", "Lato", "Montserrat", "Poppins", "Raleway",
    "Nunito", "Nunito Sans", "Work Sans", "Source Sans 3", "PT Sans", "Noto Sans",
    "Rubik", "Karla", "Mulish", "Manrope", "DM Sans", "Barlow", "Barlow Condensed",
    "Archivo", "Libre Franklin", "Josefin Sans", "Jost", "Questrial", "Fira Sans",
    "Cabin", "Quicksand", "Comfortaa", "Dosis", "Titillium Web", "Heebo",
    "Assistant", "Hind", "Catamaran", "Oxygen", "Signika", "Exo 2", "Saira",
    "Chivo", "Figtree", "Sora", "Outfit", "Plus Jakarta Sans", "Kanit", "Prompt",
    "Space Grotesk", "IBM Plex Sans", "Schibsted Grotesk", "Albert Sans",
    "Roboto Condensed",
    # display / condensed headline
    "Oswald", "Bebas Neue", "Anton", "Teko", "Abril Fatface",
    # serif
    "Playfair Display", "Merriweather", "PT Serif", "Noto Serif", "Lora",
    "Bitter", "Crimson Text", "Cormorant Garamond", "EB Garamond",
    "Libre Baskerville", "Source Serif 4", "DM Serif Display", "IBM Plex Serif",
    "Bree Serif", "Marcellus", "Roboto Serif",
    # slab
    "Roboto Slab", "Zilla Slab",
    # monospace
    "Roboto Mono", "Space Mono", "JetBrains Mono", "Fira Code", "IBM Plex Mono",
    "Source Code Pro",
    # script / handwriting
    "Dancing Script", "Pacifico", "Great Vibes", "Caveat", "Sacramento",
    "Lobster", "Comic Neue",
    # metric-compatible OFL substitutes for the common local-only faces
    "Arimo", "Tinos", "Cousine", "Carlito", "Caladea", "Gelasio",
)

_GOOGLE_FONTS_NORM = {re.sub(r"\s+", "", name.lower()): name for name in GOOGLE_FONTS_FAMILIES}


def _norm_family(name: Any) -> str:
    return re.sub(r"\s+", "", str(name or "").lower())


# Family -> glyph class by NAME, for the document-level consensus gate.  This is a
# name lookup (not a font-file probe) so it works on the already-relabelled Google
# family names carried on each line's style, where the local .ttf path may be
# absent.  Only true serif and script/handwriting families are enumerated; every
# other family (grotesque/geometric/humanist sans AND condensed *display* faces
# like Oswald/Anton/Bebas Neue) is treated as ``SANS`` so the two-tier policy is
# preserved — a distinctive display headline is never force-unified to body sans.
_SERIF_FAMILY_NAMES = (
    "Playfair Display", "Merriweather", "PT Serif", "Noto Serif", "Lora", "Bitter",
    "Crimson Text", "Cormorant Garamond", "EB Garamond", "Libre Baskerville",
    "Source Serif 4", "DM Serif Display", "IBM Plex Serif", "Bree Serif",
    "Marcellus", "Roboto Serif", "Roboto Slab", "Zilla Slab",
    # metric-compatible OFL serif substitutes for local faces
    "Tinos", "Gelasio", "Caladea",
    # common local serif names that may survive unmapped
    "Times New Roman", "Times", "Georgia", "Garamond", "Cambria", "Baskerville",
    "Book Antiqua", "Palatino", "Palatino Linotype", "Constantia",
)
_SCRIPT_FAMILY_NAMES = (
    "Dancing Script", "Pacifico", "Great Vibes", "Caveat", "Sacramento",
    "Lobster", "Comic Neue", "Gabriola", "Segoe Script", "Brush Script MT",
    "Mistral", "Freestyle Script", "Inkfree",
)
_SERIF_FAMILY_NORM = {_norm_family(n) for n in _SERIF_FAMILY_NAMES}
_SCRIPT_FAMILY_NORM = {_norm_family(n) for n in _SCRIPT_FAMILY_NAMES}


def _family_class(family: Any, path: Optional[str] = None) -> str:
    """Coarse sans/serif/script class for a family, for the consensus gate.

    Prefers the font FILE's PANOSE class (font_fit.classify_font_file) when a
    real, on-disk ``path`` is available; otherwise falls back to a name lookup.
    Returns one of ``"sans"``/``"serif"``/``"script"``; anything not positively
    serif or script is reported ``"sans"`` (the safe default that keeps display
    headline faces out of the forbidden set).
    """
    if path and os.path.isfile(str(path)):
        try:
            from src import font_fit

            cls = font_fit.classify_font_file(path)
            if cls in (font_fit.SERIF, font_fit.SCRIPT):
                return cls
            if cls == font_fit.SANS:
                return "sans"
        except Exception:
            pass
    norm = _norm_family(family)
    if norm in _SERIF_FAMILY_NORM:
        return "serif"
    if norm in _SCRIPT_FAMILY_NORM:
        return "script"
    return "sans"


# Closest Google-Fonts equivalent for common local-only (Windows/macOS) faces.
# Every target is the SAME CLASS as its source and appears in
# GOOGLE_FONTS_FAMILIES. The metric-compatible OFL substitutes are used where
# they exist (Arimo=Arial, Tinos=Times, Cousine=Courier, Carlito=Calibri,
# Caladea=Cambria, Gelasio=Georgia) so the substitution changes as little as
# possible about the rendered line.
_LOCAL_TO_GOOGLE = {
    # sans-serif
    "arial": "Arimo", "arialmt": "Arimo", "helvetica": "Arimo",
    "helveticaneue": "Arimo", "liberationsans": "Arimo",
    "calibri": "Carlito", "segoeui": "Inter",
    "candara": "Open Sans", "corbel": "Open Sans", "tahoma": "Open Sans",
    "verdana": "Open Sans", "lucidasans": "Open Sans", "lucidagrande": "Open Sans",
    "trebuchetms": "Fira Sans", "trebuchet": "Fira Sans",
    "gadugi": "Inter", "leelawadeeui": "Inter", "malgungothic": "Inter",
    "dejavusans": "Inter", "notosans": "Noto Sans",
    "franklingothic": "Libre Franklin", "franklingothicmedium": "Libre Franklin",
    "centurygothic": "Jost", "futura": "Jost", "gillsans": "Lato",
    "bahnschrift": "Barlow Condensed",
    "impact": "Anton", "haettenschweiler": "Anton",
    # serif
    "cambria": "Caladea", "cambriamath": "Caladea", "constantia": "PT Serif",
    "georgia": "Gelasio", "timesnewroman": "Tinos", "times": "Tinos",
    "liberationserif": "Tinos",
    "garamond": "EB Garamond", "bookantiqua": "PT Serif", "palatino": "PT Serif",
    "palatinolinotype": "PT Serif", "baskerville": "Libre Baskerville",
    "baskervilleoldface": "Libre Baskerville",
    "rockwell": "Zilla Slab", "notoserif": "Noto Serif",
    # monospace
    "consolas": "Source Code Pro", "couriernew": "Cousine", "courier": "Cousine",
    "lucidaconsole": "Cousine", "cascadiacode": "JetBrains Mono",
    "cascadiamono": "JetBrains Mono",
    # script / handwriting
    "gabriola": "Dancing Script", "segoescript": "Dancing Script",
    "brushscriptmt": "Pacifico", "brushscript": "Pacifico",
    "comicsansms": "Comic Neue", "inkfree": "Caveat", "mistral": "Great Vibes",
    "freestylescript": "Great Vibes",
}

# Same-class Google default when a local face is neither a known Google family
# nor in the explicit map — keyed by the class the font FILE reports
# (font_fit.classify_font_file), so an unknown local sans still emits a sans.
_CLASS_DEFAULT_GOOGLE = {
    "sans": "Inter", "serif": "PT Serif", "script": "Dancing Script",
    "decorative": "Oswald", "text": "Inter",
}


def _figma_google_family(family: Any, path: Optional[str] = None,
                         source: Optional[str] = None) -> tuple[str, str]:
    """Map a matched family to a Figma-loadable Google Fonts family (same class).

    Returns ``(google_family, kind)`` where ``kind`` is ``native-google`` (the
    match is already a Google family — emitted unchanged, preferred in ranking),
    ``mapped-local`` (a known local-only face swapped for its closest same-class
    Google equivalent) or ``mapped-class`` (an unknown local face swapped for the
    same-class Google default). Only the *family name* changes; callers keep the
    local ``path`` for rendering, so every styling attribute is preserved.
    """
    name = str(family or "").strip()
    norm = _norm_family(name)
    # A match from the on-disk OFL corpus, or whose name is already curated, is
    # Figma-loadable as-is.
    if source == "google-cache":
        return (name or "Inter"), "native-google"
    if norm in _GOOGLE_FONTS_NORM:
        return _GOOGLE_FONTS_NORM[norm], "native-google"
    if norm in _LOCAL_TO_GOOGLE:
        return _LOCAL_TO_GOOGLE[norm], "mapped-local"
    cls = None
    if path:
        try:
            from src import font_fit

            cls = font_fit.classify_font_file(path)
        except Exception:
            cls = None
    return _CLASS_DEFAULT_GOOGLE.get(cls or "sans", "Inter"), "mapped-class"


def _relabel_google_families(candidates: list) -> list:
    """Relabel each candidate's reported ``family`` to a Figma-loadable Google
    family (same class), preserving the local ``path`` and every other field so
    styling and fit evidence are untouched. Records ``local_family`` when the
    name changed and ``figma_font_source`` (the mapping kind); marks
    ``figma_loadable``. Order and count are preserved (no dedup) so callers that
    inspect the candidate chain — and its per-source diversity — see it intact.
    """
    out = []
    for cand in candidates or []:
        if not isinstance(cand, dict):
            out.append(cand)
            continue
        item = dict(cand)
        gfam = item.pop("_google_family", None)
        kind = item.pop("_google_kind", None)
        if gfam is None:
            gfam, kind = _figma_google_family(item.get("family"), item.get("path"), item.get("source"))
        if gfam and str(item.get("family")) != gfam:
            item["local_family"] = item.get("family")
            item["family"] = gfam
        if kind:
            item["figma_font_source"] = kind
        item["figma_loadable"] = True
        out.append(item)
    return out


def _google_fonts_cache_dirs(options: dict) -> list[str]:
    explicit = options.get("google_fonts_cache") or options.get("google_fonts_dir")
    dirs = []
    if explicit:
        if isinstance(explicit, str):
            explicit = [explicit]
        dirs.extend(os.path.expanduser(path) for path in explicit)
    # A caller that pins an explicit ``font_files`` universe has chosen the exact
    # candidate set deliberately; do NOT inject the ambient default OFL corpus on
    # top of it (that would smuggle Inter/other Google faces into a match that was
    # meant to consider only the given files, defeating the class/fit gate). An
    # explicit ``google_fonts_cache`` above is still honored. The normal
    # auto-discovery path (no ``font_files``) keeps the ambient corpus, so
    # corpus-primary matching is unchanged.
    if not options.get("font_files"):
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
    fonts = _discover_fonts(cache_options)
    # Bound the corpus to the curated inventory unless the caller overrides it,
    # so the match set stays the common-ad families rather than the entire
    # (huge) google/fonts tree. Fail open: if the filter would empty the corpus,
    # keep everything present so an unfamiliar-but-installed OFL family can match.
    if options.get("google_fonts_curated", True):
        allow = options.get("google_fonts_families") or GOOGLE_FONTS_FAMILIES
        allow_norm = {_norm_family(name) for name in allow}
        curated = [meta for meta in fonts if _norm_family(meta.get("family")) in allow_norm]
        if curated:
            fonts = curated
    return fonts


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
    # Empty means "no extra directories", not "disable every installed system font".
    dirs = options.get("font_dirs") or _platform_font_dirs()
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

    # Inventory preference order. This ranking decides which families survive the
    # downstream ``max_fonts`` cut (after class filtering), so it must cover the
    # staples ads actually use — the old 7-name list let the cut degenerate to the
    # alphabetical head of C:\Windows\Fonts (Calibri/Cambria/Candara/"Dodo"…) and
    # Segoe UI never even entered the match (benchmark 009: every line matched a
    # different arbitrary system font). Order ≈ how often the family (or a close
    # metric twin) appears in ad creative; Inter first (Figma default, Chirp-alike).
    preferred = [
        "inter", "segoe ui", "helvetica", "arial", "roboto", "open sans", "lato",
        "montserrat", "poppins", "source sans", "sf pro", "verdana", "tahoma",
        "trebuchet", "franklin gothic", "futura", "century gothic", "gill sans",
        "calibri", "candara", "corbel", "georgia", "garamond", "times new roman",
        "cambria", "playfair", "merriweather", "baskerville", "bahnschrift",
        "impact", "oswald", "bebas", "haettenschweiler", "rockwell", "courier new",
        "consolas", "comic sans", "segoe script", "brush script",
        "dejavu", "liberation", "noto",
    ]

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


def _resolve_font_candidates(text: str, source_mask, geo: dict, options: dict,
                             render_fit: Optional[dict] = None) -> tuple[list[dict], dict]:
    """Rank font candidates for one style-cluster representative.

    Returns ``(candidates, evidence)``.  Before shape matching, the source ink
    is classed serif/sans/script and the candidate inventory is hard-filtered by
    that class (``font_matching.class_gate``); after matching, the top candidates
    are render-and-fit refined against the ink mask (``text_analysis.render_fit``)
    so the emitted ranking reflects fitted pixel evidence, not aspect-blind
    shape scores.
    """
    top_k = max(1, min(12, int(options.get("top_k", 5))))
    profile = _typography_profile(geo)
    estimated_size = profile["font_size"]
    evidence: dict[str, Any] = {}
    if options.get("repair_pass") or options.get("force_rematch"):
        _FONT_MATCH_CACHE.clear()

    max_fonts = max(1, min(256, int(options.get("max_fonts", 48))))
    source_class = None
    if options.get("class_gate", True):
        try:
            from src import font_fit

            class_info = font_fit.classify_source(text, source_mask, estimated_size, options)
            gate_min = _num(options.get("class_gate_min_confidence"), 0.5)
            text_gate = _num(options.get("class_gate_text_min_confidence"), 0.5)
            cls = class_info.get("class")
            words = len(text.split())
            glyphs = len(text.replace(" ", ""))
            # A genuine script SOURCE is a short wordmark-like run; multi-word or long
            # copy classed "script" is almost always an ornate serif/display headline
            # that fits the script reference deceptively well (052's Gabriola headline).
            # Distrust it: a real script wordmark is routed as artwork before this.
            script_plausible = cls == font_fit.SCRIPT and words <= 1 and glyphs <= 5
            if cls == font_fit.SCRIPT and not script_plausible:
                source_class = font_fit.TEXT
            elif cls and _num(class_info.get("confidence")) >= gate_min:
                # A confident sans/serif (or wordmark-short script) call hard-filters
                # to that class.
                source_class = cls
            elif _num(class_info.get("text_confidence")) >= text_gate and cls != font_fit.SCRIPT:
                # Sans-vs-serif undecided but the source is clearly plain text:
                # keep both text classes, exclude only script/decorative faces so a
                # swash face can never win plain body/headline copy (the Gabriola
                # failure) while any clean same-class substitute stays eligible.
                source_class = font_fit.TEXT
            # Numeric / stat strings ("257", "21K", "121K", "66") carry too few, too
            # simple glyphs for the class vote to fire, so a swash/serif display face
            # wins the shape match by luck (benchmark 009: '666'/'89' -> Dancing
            # Script, '257'/'21K' -> Caladea serif — where Codia ships plain Inter).
            # These are effectively never set in a script/decorative face; when the
            # class call stays undecided, still exclude script/decorative so a clean
            # same-class face is picked. A confident script call above is untouched.
            if source_class is None and cls != font_fit.SCRIPT:
                compact = text.replace(" ", "")
                numericish = bool(compact) and (
                    sum(ch.isdigit() for ch in compact) / len(compact) >= 0.5)
                if numericish:
                    source_class = font_fit.TEXT
            evidence["class_gate"] = class_info
            evidence["source_class"] = source_class
        except Exception:
            source_class = None

    def _gated(fonts: list[dict]) -> list[dict]:
        if not source_class or not fonts:
            return fonts[:max_fonts]
        from src import font_fit

        return font_fit.filter_fonts_by_class(fonts, source_class)[:max_fonts]

    local = _match_fonts(text, source_mask, estimated_size, options, profile=profile,
                         fonts=_gated(_discover_fonts(options)))
    google = []
    if _google_fonts_cache_dirs(options):
        google_fonts = _gated(_discover_google_fonts(options))
        if google_fonts:
            google = _match_fonts(
                text, source_mask, estimated_size, options, profile=profile,
                fonts=google_fonts, source_label="google-cache",
            )
    fallback_slots = max(0, top_k - len(local) - len(google))
    fallback = _fallback_font_candidates(
        profile["weight"], options, max(1, fallback_slots or top_k), italic=profile["italic"],
    )
    # Corpus-primary: the on-disk Google/OFL corpus is the reference the render-fit
    # ranks against, so a plausible VISUAL match — chunky display sans -> Anton/Oswald/
    # Archivo Black, geometric -> Poppins/Montserrat, editorial serif -> Playfair —
    # can win over a generic local remap that flattens everything to Arimo/Carlito.
    # The corpus is merged FIRST and a wider pre-refine pool keeps both corpus and
    # local candidates so refine_candidates render-fits and ranks BOTH by fitted
    # pixel evidence (with a small Figma-loadable-Google tie bonus). Curation, the
    # class gate and the short-string discount already bound and vet the corpus set.
    # Corpus-primary applies to AUTO-DISCOVERED platform fonts (the generic-remap
    # problem: everything collapses to Arimo/Carlito). An explicit ``font_files`` list
    # is a deliberate caller choice and is respected as-is (local-first, unchanged).
    corpus_primary = bool(google) and not options.get("font_files")
    pool_k = max(top_k, int(_num(options.get("fit_pool"), 8))) if corpus_primary else top_k
    if corpus_primary:
        merged = _merge_font_candidates(google, local, fallback, top_k=pool_k)
    else:
        merged = _merge_font_candidates(local, google, fallback, top_k=top_k)

    # Tag Figma-loadability BEFORE ranking so refine_candidates can prefer a
    # genuine Google match over a local-only face that must be remapped; stash the
    # resolved Google family so the relabel after ranking is a pure name swap.
    for cand in merged:
        if not isinstance(cand, dict):
            continue
        gfam, kind = _figma_google_family(cand.get("family"), cand.get("path"), cand.get("source"))
        cand["google_native"] = kind == "native-google"
        cand["_google_family"] = gfam
        cand["_google_kind"] = kind

    fit_opts = render_fit if render_fit is not None else {"enabled": True}
    if corpus_primary:
        # Fit enough of the widened pool that the best corpus AND best local faces are
        # both render-fitted before ranking (default only fits the first 3).
        fit_opts = dict(fit_opts)
        fit_opts.setdefault("max_candidates", max(6, int(_num(fit_opts.get("max_candidates"), 6))))
    if fit_opts.get("enabled", True):
        try:
            from src import font_fit

            merged, fit_evidence = font_fit.refine_candidates(
                text, source_mask, merged, estimated_size, fit_opts,
            )
            evidence["render_fit"] = fit_evidence
        except Exception:
            pass
    # Emit the top_k by the fitted (visual) ranking; the wider pool above was only to
    # give the corpus a fair render-fit against the local candidates.
    if corpus_primary and len(merged) > top_k:
        merged = merged[:top_k]

    # Relabel the reported family to a Figma-loadable Google equivalent AFTER
    # ranking (local path kept for rendering, so styling/fit are untouched). Every
    # emitted fontFamily is now one Figma can natively load.
    merged = _relabel_google_families(merged)
    return merged, evidence


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
    built = {
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
    # The preview renderer draws candidates[0]; declared weight/style must match it
    # (also covers a google-cache top candidate whose weight differs from the coarse
    # ink-density estimate above).
    _reconcile_style_weight(built)
    return built


def _reconcile_style_weight(style: dict) -> None:
    """Keep ``fontWeight``/``fontStyle`` consistent with the rendered top candidate.

    render_preview draws the preview from ``fontCandidates[0].path`` while the Figma
    export node reads ``fontWeight``/``fontStyle``. When a later pass (font consensus,
    per-line refit) swaps the top candidate to a different weight *without* updating
    these fields, the preview renders one weight while the exported node claims
    another. In the benchmark this desynced 25 text layers — a family's bold headline
    face was rendered onto regular body copy (and vice-versa), roughly doubling
    rendered-ink and slicing otherwise-editable text. The top renderable candidate is
    the single source of truth for what is actually drawn, so mirror it here.
    (test_text_analysis asserts exactly this invariant: candidates[0].weight ==
    fontWeight.)
    """
    if not isinstance(style, dict):
        return
    cands = style.get("fontCandidates") or []
    top = cands[0] if cands and isinstance(cands[0], dict) else None
    if not top or top.get("source") not in {"local-render", "google-cache"}:
        return
    weight = top.get("weight")
    if isinstance(weight, (int, float)):
        weight = int(weight)
        style["fontWeight"] = weight
        # Preserve an already-italic style label; only correct the weight token.
        italic = "italic" in str(style.get("fontStyle") or "").lower()
        top_style = str(top.get("style") or "")
        if top_style and (("italic" in top_style.lower()) == italic):
            style["fontStyle"] = top_style
        else:
            style["fontStyle"] = _style_name(weight, italic=italic)
        style["fontWeightCandidates"] = _weight_candidates(weight)


def _render_fit_options(config: dict) -> dict:
    """Normalize ``text_analysis.render_fit`` via font_fit (bool or mapping, default ON)."""
    try:
        from src import font_fit

        return font_fit.fit_options(config)
    except Exception:
        return {"enabled": False}


def _apply_line_render_fit(line: dict, mask, painted: dict, render_fit_options: dict) -> Optional[dict]:
    """Refine one line's emitted size/tracking by fitting its chosen font to its
    own ink mask (cluster representatives share matched candidates, but every
    line has its own painted geometry).  Applies the fitted ``fontSize`` and
    ``letterSpacing`` when the fit passes ``min_score``; always records the fit
    evidence on ``meta.render_fit`` so downstream fidelity gating can see a bad
    fit even when nothing was applied.  Returns the fit mapping or ``None``.
    """
    if not render_fit_options.get("enabled", True) or mask is None:
        return None
    style = line.get("style") or {}
    candidates = style.get("fontCandidates") or []
    chosen = candidates[0] if candidates and isinstance(candidates[0], dict) else None
    if not chosen or not chosen.get("path") or not os.path.exists(str(chosen["path"])):
        return None
    try:
        from src import font_fit

        fit = font_fit.fit_line(
            line.get("text", ""), chosen["path"], mask,
            _num(style.get("fontSize"), 16.0), render_fit_options,
        )
    except Exception:
        return None
    if fit is None:
        return None
    min_score = _num(render_fit_options.get("min_score"), 0.30)
    applied = fit["score"] >= min_score
    if applied:
        new_size = _num(fit.get("fontSize"), style.get("fontSize"))
        style["fontSize"] = round(new_size, 2)
        style["letterSpacing"] = round(_num(fit.get("letterSpacing")), 3)
        style["lineHeight"] = round(max(new_size * 1.15, _num(painted.get("h"))), 2)
        style["fontSizeCandidates"] = _size_candidates(new_size)
    line.setdefault("meta", {})["render_fit"] = {
        "family": chosen.get("family"),
        "score": fit["score"],
        "fontSize": fit["fontSize"],
        "letterSpacing": fit["letterSpacing"],
        "applied": applied,
    }
    return fit


def _apply_font_consensus(prepared: list[dict], render_fit_options: dict,
                          font_options: dict) -> Optional[dict]:
    """Document-level font family consensus across all matched lines.

    Votes each line's chosen family weighted by ink area x fit score, then
    re-fits outlier lines against the winning family's font file. A line adopts
    the consensus only when its consensus fit passes ``render_fit.min_score``
    AND comes within ``tolerance`` of (or beats) its own fit — so a genuinely
    different family (serif headline over sans body, a script wordmark) keeps
    its own match: the fit itself is the guard, no class bookkeeping needed.
    Lines whose own fit is already exact-font territory (``strong_keep``) are
    never touched. Returns an evidence dict for the run artifact, or None.
    """
    opts = (font_options or {}).get("consensus") or {}
    if opts.get("enabled", True) is False:
        return None
    if len(prepared) < int(_num(opts.get("min_lines"), 3)):
        return None

    votes: dict[str, dict] = {}
    for item in prepared:
        line = item.get("line") or {}
        style = line.get("style") or {}
        family = style.get("fontFamily")
        fit_meta = (line.get("meta") or {}).get("render_fit") or {}
        score = fit_meta.get("score")
        if not family or score is None:
            continue
        painted = item.get("painted") or {}
        area = max(1.0, _num(painted.get("w"), 1.0) * _num(painted.get("h"), 1.0))
        candidate = next(
            (c for c in (style.get("fontCandidates") or [])
             if isinstance(c, dict) and str(c.get("family")) == str(family) and c.get("path")),
            None,
        )
        entry = votes.setdefault(str(family), {"weight": 0.0, "lines": 0,
                                               "candidate": candidate, "best_fit": 0.0})
        entry["weight"] += area * max(_num(score, 0.0), 0.05)
        entry["lines"] += 1
        if _num(score, 0.0) > entry["best_fit"]:
            entry["best_fit"] = _num(score, 0.0)
            if candidate:
                entry["candidate"] = candidate
    if not votes:
        return None
    total = sum(entry["weight"] for entry in votes.values())
    family, info = max(votes.items(), key=lambda kv: kv[1]["weight"])
    share = (info["weight"] / total) if total > 0 else 0.0

    # Class consistency: when almost all voting weight is a single class (sans is
    # the overwhelming case for ad body/label copy), the document clearly wants ONE
    # family for that class.  We then (a) lower the share bar so a dominant-but-not
    # -majority sans family still unifies the block, and (b) raise the pull
    # (tolerance) so more per-line matches fold into it — the scattered
    # Archivo/Poppins/Inter/Albert-Sans zoo on benchmark 002 is exactly this.
    class_weight: dict[str, float] = {}
    for fam, entry in votes.items():
        cand = entry.get("candidate") or {}
        cls = _family_class(fam, cand.get("path"))
        class_weight[cls] = class_weight.get(cls, 0.0) + entry["weight"]
    sans_share = (class_weight.get("sans", 0.0) / total) if total > 0 else 0.0
    consensus_class = _family_class(family, (info.get("candidate") or {}).get("path"))
    sans_consistency = _num(opts.get("sans_consistency"), 0.70)
    class_consistent = consensus_class == "sans" and sans_share >= sans_consistency
    # Dominance: the winning family clearly leads the runner-up, not a near-tie among
    # several families (that is not a "dominant sans" and must not be forced through).
    others = sorted((e["weight"] for f, e in votes.items() if f != family), reverse=True)
    second_weight = others[0] if others else 0.0
    dominant = info["weight"] >= max(second_weight, 1e-6) * _num(opts.get("dominance_ratio"), 1.4)

    evidence: dict[str, Any] = {
        "family": family, "share": round(share, 3),
        "lines_voting": info["lines"], "applied": False, "refit": [],
        "consensus_class": consensus_class, "sans_share": round(sans_share, 3),
        "class_consistent": class_consistent, "dominant": dominant,
    }
    candidate = info.get("candidate")
    path = str((candidate or {}).get("path") or "")
    min_share = _num(opts.get("min_share"), 0.30)
    if class_consistent and dominant:
        # A dominant, consistent sans still unifies the block when its share sits just
        # under the default floor (each body line otherwise keeps its own scattered pick).
        min_share = min(min_share, _num(opts.get("consistent_min_share"), 0.20))
    if share < min_share or not path or not os.path.exists(path):
        return evidence

    tolerance = _num(opts.get("tolerance"), 0.10)
    if class_consistent:
        tolerance = max(tolerance, _num(opts.get("consistent_tolerance"), 0.18))
    strong_keep = _num(opts.get("strong_keep"), 0.72)
    min_score = _num(render_fit_options.get("min_score"), 0.30)
    try:
        from src import font_fit
    except Exception:
        return evidence
    for item in prepared:
        line = item.get("line") or {}
        style = line.get("style") or {}
        if str(style.get("fontFamily")) == family:
            continue
        # Family consensus must NOT flatten weight. The stored representative
        # candidate carries one fixed weight (the family's loudest, best-fitting
        # voter — often a bold headline). Promoting it onto a line of a different
        # weight class renders the wrong stroke thickness: a bold face on regular
        # body copy doubles rendered ink (region_ssim collapses, the raster-slice
        # gate then fires), and a regular face on a bold headline renders too light.
        # Only unify the FAMILY across same-weight lines; a genuinely different
        # weight keeps its own correct-weight match (benchmark 025/091/002:
        # weight-flattened body/headlines drove the editable-text collapse).
        # HARD class gate: a serif/script family that leaked onto a sans-consensus
        # document (benchmark 002: EB Garamond on the sans body line
        # "zoetstof: sucralose") is a cross-class defect, never a real accent. When
        # the document is a consistent sans, such a line is FORBIDDEN its own family
        # and must fold into the consensus sans — even if the consensus refit scores
        # worse or the weight differs. A genuinely distinctive serif/display headline
        # is still protected below by the ``strong_keep`` guard (a real Playfair-class
        # headline scores in exact-font territory and is left untouched).
        line_family = str(style.get("fontFamily") or "")
        line_cand = next(
            (c for c in (style.get("fontCandidates") or [])
             if isinstance(c, dict) and str(c.get("family")) == line_family and c.get("path")),
            None,
        )
        line_class = _family_class(line_family, (line_cand or {}).get("path"))
        forbidden = class_consistent and line_class in ("serif", "script")

        rep_weight = candidate.get("weight") if isinstance(candidate, dict) else None
        line_weight = style.get("fontWeight")
        max_weight_delta = _num(opts.get("max_weight_delta"), 200)
        if (not forbidden and isinstance(rep_weight, (int, float))
                and isinstance(line_weight, (int, float))
                and abs(int(rep_weight) - int(line_weight)) >= max_weight_delta):
            continue
        meta = line.setdefault("meta", {})
        own = _num((meta.get("render_fit") or {}).get("score"), 0.0)
        if own >= strong_keep:
            continue  # near-exact per-line match outranks consistency (incl. real serif)
        mask = item.get("font_mask")
        fit = None
        if mask is not None:
            try:
                fit = font_fit.fit_line(
                    line.get("text", ""), path, mask,
                    _num(style.get("fontSize"), 16.0), render_fit_options,
                )
            except Exception:
                fit = None
        new_score = _num(fit.get("score"), 0.0) if fit is not None else 0.0
        # Adopt on evidence when the consensus fit is within tolerance of (or beats)
        # the line's own match AND does not push an editable line below the raster bar.
        adopt = (
            fit is not None
            and new_score >= max(min_score, own - tolerance)
            and not (own >= min_score and new_score < min_score)
        )
        previous = line_family
        if adopt:
            new_size = _num(fit.get("fontSize"), _num(style.get("fontSize"), 16.0))
            style["fontFamily"] = family
            style["fontSize"] = round(new_size, 2)
            style["letterSpacing"] = round(_num(fit.get("letterSpacing")), 3)
            style["lineHeight"] = round(
                max(new_size * 1.15, _num((item.get("painted") or {}).get("h"))), 2)
            style["fontSizeCandidates"] = _size_candidates(new_size)
            promoted = dict(candidate)
            promoted["fit"] = {"score": new_score}
            style["fontCandidates"] = [promoted] + [
                c for c in (style.get("fontCandidates") or [])
                if not (isinstance(c, dict) and str(c.get("family")) == family)
            ]
            # The renderer draws candidates[0]; keep the declared weight/style in sync
            # so the exported Figma node matches the promoted face.
            _reconcile_style_weight(style)
            meta["render_fit"] = {
                "family": family, "score": new_score, "fontSize": fit.get("fontSize"),
                "letterSpacing": fit.get("letterSpacing"), "applied": True, "consensus": True,
            }
            item["line_fit"] = fit
            evidence["refit"].append({
                "id": line.get("id"), "from": previous, "to": family,
                "own_score": round(own, 3), "new_score": round(new_score, 3),
            })
        elif forbidden:
            # No confident refit to promote geometry from, but a serif/script must
            # not survive on sans body copy. Relabel the FAMILY to the consensus sans
            # (keeping the line's own size and weight — we are not confident enough to
            # move geometry), so a Figma-loadable sans is emitted instead of the leak.
            style["fontFamily"] = family
            promoted = dict(candidate)
            promoted["family"] = family
            if isinstance(line_weight, (int, float)):
                promoted["weight"] = int(line_weight)
            promoted["fit"] = {"score": new_score} if fit is not None else {}
            style["fontCandidates"] = [promoted] + [
                c for c in (style.get("fontCandidates") or [])
                if not (isinstance(c, dict) and str(c.get("family")) in (family, previous))
            ]
            meta["render_fit"] = {
                "family": family, "score": round(new_score, 4) if fit is not None else round(own, 4),
                "fontSize": _num(style.get("fontSize"), 16.0),
                "letterSpacing": _num(style.get("letterSpacing")),
                "applied": True, "consensus": True, "forbidden_class": line_class,
            }
            evidence["refit"].append({
                "id": line.get("id"), "from": previous, "to": family,
                "own_score": round(own, 3), "new_score": round(new_score, 3),
                "forbidden_class": line_class,
            })
    evidence["applied"] = True
    return evidence


def _enrich_word_styles(image, line: dict, config: dict) -> None:
    """Attach conservative, pixel-evidenced style overrides to OCR words.

    A single OCR line can contain e.g. ``ONLY $19`` where the price changes colour,
    size or weight.  Treating the whole line as one style loses that design.  We do
    *not* independently guess a font for every word: the proven line family remains
    the base and only large, measurable paint/geometry differences become overrides.
    No image (or weak/ambiguous evidence) means no word runs downstream.
    """
    if image is None or not line.get("words") or not line.get("style"):
        return
    base = line["style"]
    base_size = max(1.0, _num(base.get("fontSize"), 16.0))
    base_weight = int(round(_num(base.get("fontWeight"), 400)))
    base_colour = str(base.get("color") or "#000000")
    min_conf = _num(config.get("word_style_min_ink_confidence"), 0.42)
    colour_delta = _num(config.get("word_style_color_distance"), 58.0)
    # Per-word SIZE is the "weird scaling" defect (benchmark 002: "per 100g" split into
    # per=12.5px + 100g=31px; ingredient lines fragmented). A single word measured
    # inside a normal line is noisy — glyph composition alone (a short lowercase token
    # lacks caps/ascenders; a digit run is taller) shifts the tight-ink height enough to
    # read as a spurious size. So a per-word size override now demands the same standard
    # the weight guard already got: a raised ratio, high ink confidence, INDEPENDENT
    # corroboration (contrast/weight), a sane cap on how far a word may exceed its line,
    # and — for the always-noisier "smaller than the line" direction — a standalone token.
    # When in doubt we keep the line uniform rather than emit a mis-scaled word.
    size_ratio_gate = _num(config.get("word_style_size_ratio"), 1.35)
    size_min_conf = _num(config.get("word_style_size_min_ink_confidence"), 0.60)
    size_max_up_ratio = _num(config.get("word_style_size_max_up_ratio"), 1.5)
    # Per-word weight is estimated from stroke density, which is the noisiest signal —
    # same-weight lines jitter ±100-150. Codia only splits on CLEAR contrast (its real
    # splits are ~400 apart: 121K/700 vs weergaven/300). Keep the delta well above the
    # noise floor and require higher ink confidence for a weight change specifically, so
    # a heavy-inked regular word can't fragment an otherwise-uniform line.
    weight_delta = int(_num(config.get("word_style_weight_delta"), 260))
    weight_min_conf = _num(config.get("word_style_weight_min_ink_confidence"), 0.60)

    def _styleable(candidate: Any) -> bool:
        # A lone punctuation mark or a 1-char sliver (",", "->", a stray digit) carries
        # no reliable per-word style and must never become its own Figma run.
        txt = str((candidate or {}).get("text") or "").strip()
        return len(txt) >= 2 and any(ch.isalnum() for ch in txt)

    valid_word_count = sum(
        1 for w in (line.get("words") or []) if isinstance(w, dict) and _styleable(w)
    )

    for raw_word in line.get("words") or []:
        if not isinstance(raw_word, dict) or not _styleable(raw_word):
            continue
        word = raw_word
        word["box"] = _clean_box(word.get("box"))
        painted, _baseline, colour, ink_conf, mask, paint = _painted_geometry(image, word)
        if ink_conf < min_conf:
            continue
        geo = _pre_font_signals(word, painted, mask, config)
        measured_size = max(1.0, _num(geo.get("font_size"), base_size))
        measured_weight = int(round(_num(geo.get("weight"), base_weight)))
        ratio = max(measured_size, base_size) / max(1.0, min(measured_size, base_size))
        colour_changed = _colour_distance(colour, base_colour) >= colour_delta
        weight_changed = (abs(measured_weight - base_weight) >= weight_delta
                          and ink_conf >= weight_min_conf)
        # A per-word size override needs an independent reason to believe the word is a
        # genuinely different run (a contrasting colour or a real weight change), not just
        # a different measured height. A single, emphasised token can stand alone; a word
        # mid-line among peers cannot carry a size split on its own.
        standalone = valid_word_count <= 1 or (colour_changed and valid_word_count <= 2)
        corroborated = colour_changed or weight_changed
        if measured_size >= base_size:
            # Word BIGGER than its line (the "blew up 100g" case): allow only a bounded
            # jump unless it is a clearly standalone emphasised token.
            size_changed = (
                ratio >= size_ratio_gate and ink_conf >= size_min_conf and corroborated
                and (measured_size <= base_size * size_max_up_ratio or standalone)
            )
        else:
            # Word SMALLER than its line: dominated by glyph-composition noise (a short
            # lowercase token lacks caps/ascenders, so its tight-ink height under-reads).
            # Demand the strongest evidence — contrast AND weight AND the word being the
            # line's ONLY token — otherwise keep the line uniform. This is what suppresses
            # the "per 100g" fragmentation: any multi-word line never splits downward.
            size_changed = (
                ratio >= size_ratio_gate and ink_conf >= size_min_conf
                and colour_changed and weight_changed and valid_word_count <= 1
            )
        shear = geo.get("shear_angle")
        italic_changed = bool(shear is not None and abs(shear) >= _num(config.get("italic_shear_deg"), 6.0)) \
            != ("italic" in str(base.get("fontStyle") or "").lower())
        if not any((colour_changed, size_changed, weight_changed, italic_changed)):
            continue
        style = copy.deepcopy(base)
        # Keep the line's matched family/candidates. Only promote signals that passed
        # their own strong gate, avoiding anti-aliasing noise becoming Figma runs.
        if colour_changed:
            style["color"] = colour
            style["colorRGB"] = list(_hex_rgb(colour))
            style["fill"] = (paint or {}).get("fill") or {"kind": "flat", "color": colour}
            style["stroke"] = (paint or {}).get("stroke")
        if size_changed:
            style["fontSize"] = round(measured_size, 2)
            style["lineHeight"] = round(max(measured_size * 1.15, painted["h"]), 2)
        if weight_changed:
            style["fontWeight"] = measured_weight
            style["fontStyle"] = _style_name(
                measured_weight, italic="italic" in str(style.get("fontStyle") or "").lower()
            )
        if italic_changed:
            italic = bool(shear is not None and abs(shear) >= _num(config.get("italic_shear_deg"), 6.0))
            style["fontStyle"] = _style_name(int(style.get("fontWeight", base_weight)), italic=italic)
        word["style"] = style
        word["style_evidence"] = {
            "source": "word-pixels", "confidence": round(float(ink_conf), 4),
            "changed": [name for name, changed in (
                ("color", colour_changed), ("size", size_changed),
                ("weight", weight_changed), ("italic", italic_changed),
            ) if changed],
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
        rotations = [float(line.get("rotation_deg", 0.0) or 0.0) for line in group]
        snap_deg = _num(config.get("rotation_snap_deg"), _DEFAULT_ROTATION_SNAP_DEG)
        if len(group) == 1:
            rotation = rotations[0]
        else:
            # A paragraph rotates only when every member line agrees on the same
            # substantial angle.  One malformed OCR quad (or a mix of snapped and
            # wobbly lines) must never skew copy the source paints horizontal.
            agreeing = [angle for angle in rotations if abs(angle) >= max(0.01, snap_deg)]
            if len(agreeing) == len(rotations) and rotations and (
                max(rotations) - min(rotations)
            ) <= 3.0:
                rotation = _median(rotations)
            else:
                rotation = 0.0
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
            # Authored line breaks are part of the design: the block's text keeps
            # one explicit "\n" per detected source line so no later stage can
            # re-wrap the paragraph differently from the source.
            "text": "\n".join(line.get("text", "") for line in group),
            # Per-line geometry rides along so downstream consumers can place or
            # verify each authored line exactly, not just the union box.
            "line_geometry": [
                {
                    "id": line["id"],
                    "text": line.get("text", ""),
                    "box": dict(line["box"]),
                    "painted_box": dict(line["painted_box"]),
                    "baseline": dict(line["baseline"]),
                    "rotation_deg": float(line.get("rotation_deg", 0.0) or 0.0),
                }
                for line in group
            ],
            "box": _union_boxes(line["box"] for line in group),
            "painted_box": _union_boxes(line["painted_box"] for line in group),
            "alignment": alignment,
            "line_height": round(line_height, 2),
            "rotation": round(rotation, 3),
            "rotation_deg": round(rotation, 3),
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
# Fidelity is a "will this render look right" signal, not "is this the exact
# font".  A legible line set in a plausible SAME-CLASS face (any clean sans for a
# sans line, any serif for a serif headline) stays editable text — its exact fit
# IoU is floored above the raster bar so accurate styling, not font identity,
# decides editability.  Only a wrong-CLASS render (a script/decorative face over
# plain text — the case that actually looks broken) is capped below the bar, and
# illegible ink still slices via the ink-confidence term.
_DEFAULT_FIDELITY_SAME_CLASS_FLOOR = 0.50
_DEFAULT_FIDELITY_WRONG_CLASS_CAP = 0.28


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
    fidelity_same_class_floor = _num(config.get("fidelity_same_class_floor"), _DEFAULT_FIDELITY_SAME_CLASS_FLOOR)
    fidelity_wrong_class_cap = _num(config.get("fidelity_wrong_class_cap"), _DEFAULT_FIDELITY_WRONG_CLASS_CAP)

    raw_lines = result.get("lines") or []
    max_match_lines = max(0, min(100, int(font_options.get("max_lines", 12))))
    snap_deg = _num(config.get("rotation_snap_deg"), _DEFAULT_ROTATION_SNAP_DEG)
    render_fit_options = _render_fit_options(config)

    # Pass 1: cheap per-line geometry (painted bounds, ink mask, paint/shear signals).
    # No font-file rendering happens here.
    prepared = []
    masks: dict[str, Any] = {}
    for index, raw in enumerate(raw_lines):
        line = copy.deepcopy(raw)
        line.setdefault("id", f"L{index}")
        line["box"] = _clean_box(line.get("box"))
        painted, baseline, colour, ink_confidence, mask, paint = _painted_geometry(
            image, line, snap_deg=snap_deg,
        )
        line["painted_box"] = painted
        line["baseline"] = baseline
        raw_rotation = _quad_rotation(line.get("quad"))
        line["rotation"] = _snap_rotation(raw_rotation, snap_deg)
        line["rotation_deg"] = line["rotation"]
        if line["rotation"] != raw_rotation:
            line.setdefault("meta", {})["rotation_raw_deg"] = raw_rotation
        line["ink_confidence"] = ink_confidence
        masks[line["id"]] = mask
        decoration, decoration_evidence = _native_text_decoration(mask, line.get("text", ""))
        font_mask = mask
        if decoration_evidence and mask is not None:
            try:
                font_mask = mask.copy()
                row0, row1 = decoration_evidence["mask_rows"]
                font_mask[row0:row1 + 1, :] = False
            except Exception:
                font_mask = mask
        geo = _pre_font_signals(line, painted, font_mask, config)
        prepared.append({
            "line": line, "painted": painted, "colour": colour,
            "ink_confidence": ink_confidence, "mask": mask, "font_mask": font_mask,
            "paint": paint, "geo": geo, "decoration": decoration,
            "decoration_evidence": decoration_evidence,
        })

    # Pass 2: font matching by style-cluster representative. Lines that share a
    # coarse style bucket (size/weight/slant/colour) reuse one representative's
    # match instead of each independently spending the match budget — so a
    # >max_lines-line ad still gets a real font on every same-style line, not
    # just the first `max_lines` lines encountered.
    preset_by_index: dict[int, list[dict]] = {}
    match_evidence_by_index: dict[int, dict] = {}
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
            candidates, match_evidence = _resolve_font_candidates(
                rep_item["line"].get("text", ""), rep_item["font_mask"],
                rep_item["geo"], font_options, render_fit=render_fit_options,
            )
            if candidates:
                for i in idxs:
                    preset_by_index[i] = candidates
                    if match_evidence:
                        match_evidence_by_index[i] = match_evidence

    # Pass 3: assemble the final per-line style, and gate low-fidelity lines
    # (poor ink isolation or a poor font/effect match) to a masked-pixel
    # fallback instead of emitting a guessed font rendering downstream.
    enriched = []
    for i, item in enumerate(prepared):
        line = item["line"]
        preset = preset_by_index.get(i)
        line["style"] = _base_style(
            line, item["painted"], item["colour"], item["ink_confidence"], item["font_mask"],
            config, font_options, item["geo"], preset_candidates=preset, paint=item["paint"],
        )
        decoration = item["decoration"]
        decoration_evidence = item["decoration_evidence"]
        if decoration:
            line["style"]["textDecoration"] = decoration
            line.setdefault("meta", {})["text_decoration_evidence"] = decoration_evidence
        match_evidence = match_evidence_by_index.get(i)
        if match_evidence:
            line.setdefault("meta", {})["font_match"] = copy.deepcopy(match_evidence)
        item["line_fit"] = _apply_line_render_fit(line, item["font_mask"], item["painted"], render_fit_options)
        _enrich_word_styles(image, line, config)
        enriched.append(line)

    # Pass 3.5: document-level font consensus. Ads use one or two families, but
    # per-line matching picks each line's own best shape/fit match, scattering a
    # single-family ad across arbitrary system fonts (benchmark 009: Lucida,
    # Candara, Calibri, Courier New, Cambria, Malgun on ONE Chirp-set post) —
    # every mediocre match then fits at ~0.4, renders mismatched ink, and the
    # raster-slice gate correctly converts it to pixels. Voting across lines and
    # re-fitting outliers against the winning family both restores consistency
    # and lifts fit scores, keeping text editable.
    consensus_evidence = _apply_font_consensus(prepared, render_fit_options, font_options)
    if consensus_evidence:
        result["font_consensus"] = consensus_evidence

    # Pass 3.6: fidelity gating (after consensus so adopted re-fits count).
    for i, item in enumerate(prepared):
        line = item["line"]
        line_fit = item.get("line_fit")

        font_confidence = None
        candidates = line["style"].get("fontCandidates") or []
        render_candidates = [
            item for item in candidates
            if isinstance(item, dict) and item.get("source") in {"local-render", "google-cache"}
        ]
        # Fitted-render scores are pixel evidence at the emitted size/tracking and
        # therefore trump aspect-blind shape-match scores: a swash face that
        # shape-matched at 0.85 but fits the ink at 0.2 must gate to the
        # masked-pixel fallback, not ship as confident text.
        fit_scores = [
            float(candidate["fit"].get("score", 0.0) or 0.0)
            for candidate in render_candidates
            if isinstance(candidate.get("fit"), dict)
        ]
        if line_fit is not None:
            fit_scores.append(float(line_fit.get("score", 0.0) or 0.0))
        if fit_scores:
            font_confidence = max(fit_scores)
        elif render_candidates:
            font_confidence = max(float(item.get("score", 0.0) or 0.0) for item in render_candidates)

        # Reframe fidelity as "will the render look right", biased toward keeping
        # text editable. A legible line in a plausible SAME-CLASS font clears the
        # bar even when the exact typeface is unknown — accurate styling
        # (size/weight/tracking/leading/colour, applied above) is what makes a
        # substitute read correctly, so a modest fit IoU is floored rather than
        # letting it rasterize editable copy. Only a wrong-CLASS render (a
        # script/decorative face over plain text — the case that looks broken) is
        # capped below the bar; illegible ink still slices via ink_confidence.
        chosen_class = None
        wrong_class = False
        styled_font_confidence = font_confidence
        if font_confidence is not None:
            try:
                from src import font_fit

                chosen = render_candidates[0] if render_candidates else None
                chosen_path = chosen.get("path") if isinstance(chosen, dict) else None
                if chosen_path:
                    chosen_class = font_fit.classify_font_file(chosen_path)
                src_class = ((line.get("meta") or {}).get("font_match") or {}).get(
                    "class_gate") or {}
                src_is_script = src_class.get("class") == font_fit.SCRIPT
                wrong_class = (
                    chosen_class in (font_fit.SCRIPT, font_fit.DECORATIVE) and not src_is_script
                )
            except Exception:
                wrong_class = False
            if wrong_class:
                styled_font_confidence = min(font_confidence, fidelity_wrong_class_cap)
            else:
                styled_font_confidence = max(font_confidence, fidelity_same_class_floor)

        fidelity_confidence = item["ink_confidence"] if styled_font_confidence is None else min(
            item["ink_confidence"], styled_font_confidence
        )
        low_fidelity = fidelity_confidence < fidelity_min_confidence
        meta = line.setdefault("meta", {})
        meta["fidelity_confidence"] = round(fidelity_confidence, 4)
        meta["low_fidelity"] = bool(low_fidelity)
        if low_fidelity:
            reasons = []
            if item["ink_confidence"] < fidelity_min_confidence:
                reasons.append(f"ink_confidence:{item['ink_confidence']:.2f}<{fidelity_min_confidence:.2f}")
            if wrong_class:
                reasons.append(f"wrong-font-class:{chosen_class}-for-plain-text")
            elif font_confidence is not None and font_confidence < fidelity_min_confidence:
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
    # Preserve a machine-checkable audit trail for optional VLM proofreading.  A
    # correction without the original OCR text is not evidence and is therefore
    # reported as invalid instead of being silently trusted downstream.
    vlm = result.get("vlm_proofread")
    if isinstance(vlm, dict):
        corrected = [line for line in enriched if line.get("vlm_corrected")]
        invalid = [line.get("id") for line in corrected if not line.get("ocr_text")]
        result["text_analysis_vlm_evidence"] = {
            "present": True,
            "lines_checked": int(vlm.get("lines_checked", 0) or 0),
            "lines_corrected": int(vlm.get("lines_corrected", 0) or 0),
            "ensemble_disagreement_checked": int(vlm.get("ensemble_disagreement_checked", 0) or 0),
            "invalid_corrections": invalid,
            "valid": not invalid and int(vlm.get("lines_corrected", 0) or 0) == len(corrected),
            "fail_closed": bool(invalid),
        }
    else:
        result["text_analysis_vlm_evidence"] = {
            "present": False, "valid": True, "fail_closed": False,
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


__all__ = ["analyze_text", "run_text_analysis", "needs_vlm_font_judge", "local_score_threshold",
           "fit_text_box", "GOOGLE_FONTS_FAMILIES"]
