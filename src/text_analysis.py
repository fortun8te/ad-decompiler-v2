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


# ── exterior plate prior (ink/plate polarity adjudication) ────────────────────
# A band outside the line box is trusted as "the plate" only when this fraction of it
# sits within _PLATE_RING_TOLERANCE of its own median; a band crossing a plate boundary
# is bimodal, falls below it, and the prior abstains (see _exterior_plate_prior).
_PLATE_RING_UNIFORMITY = 0.5
_PLATE_RING_TOLERANCE = 30.0
_PLATE_RING_MIN_PIXELS = 20
# The elected ink must sit at least this much closer to the true plate than the rejected
# class before we call it an inversion — well beyond AA/JPEG noise, so a merely darker
# median never trips it.
_PLATE_INVERSION_MARGIN = 40.0


def _resolve_ink_polarity(crop, mask, plate_prior):
    """Reverse ``mask`` when it demonstrably elected the PLATE as ink.

    Uses ``plate_prior`` (pixels outside the box — the only ones a glyph-tight box cannot
    contaminate) purely as an adjudicator between the two luminance classes: ink is the
    class FARTHER from the true plate.

    Fires only on a true INVERSION — the elected mask's colour belongs to the plate class
    rather than the ink class. It deliberately does NOT act when the elected mask merely
    sits between the two (the signature of a correct mask that includes anti-aliased edge
    pixels: 101's body copy elects #2a2a2a where the pure glyph core is #080808). Nudging
    those would churn every line's fill and reshape ink mass for no correctness gain, so
    a mask that is already on the ink side is left exactly as found.
    """
    import numpy as np

    if plate_prior is None or mask is None or not mask.any() or not (~mask).any():
        return mask
    plate = np.asarray(plate_prior[0], dtype=np.float32)
    minority = _minority_luminance_mask(crop)
    if minority is None or not minority.any() or not (~minority).any():
        return mask
    c_minority = np.median(crop[minority].astype(np.float32), axis=0)
    c_majority = np.median(crop[~minority].astype(np.float32), axis=0)
    d_minority = float(np.linalg.norm(c_minority - plate))
    d_majority = float(np.linalg.norm(c_majority - plate))
    # Both classes equally far from the plate ⇒ no confident call (e.g. two-tone art).
    if abs(d_majority - d_minority) < _PLATE_INVERSION_MARGIN:
        return mask
    if d_majority > d_minority:
        ink_mask, c_ink, c_plate = ~minority, c_majority, c_minority
    else:
        ink_mask, c_ink, c_plate = minority, c_minority, c_majority
    current = np.median(crop[mask].astype(np.float32), axis=0)
    to_plate = float(np.linalg.norm(current - c_plate))
    to_ink = float(np.linalg.norm(current - c_ink))
    if to_plate + _PLATE_INVERSION_MARGIN < to_ink:
        return ink_mask
    return mask


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


def _exterior_plate_prior(image, box) -> Optional[tuple]:
    """Median colour of a band strictly OUTSIDE ``box``, when that band is uniform.

    Both polarity heuristics inside ``_ink_mask`` estimate the plate from pixels INSIDE
    the crop, so a glyph-tight box gives them no clean plate to compare against. The
    pixels just outside the box are the only ones guaranteed to be plate — *if* the band
    does not cross a plate boundary.

    That "if" is the whole design. ``_collar_box``'s docstring records that widening the
    LINE sampling window by a collar is a net regression precisely because a line box's
    collar can leave its plate. So this returns the band's median ONLY when the band is
    demonstrably a single colour (>= ``_PLATE_RING_UNIFORMITY`` of it near its own
    median); a band straddling two plates is bimodal, fails that test, and we abstain
    rather than guess. Crucially the band is used only to ADJUDICATE polarity, never as
    the crop, so geometry and mass stay measured on the tight box.

    Returns ``(plate_rgb, uniformity)`` or ``None``.
    """
    import numpy as np

    if image is None:
        return None
    box = _clean_box(box)
    ih, iw = image.shape[:2]
    pad = max(3.0, min(16.0, min(box["w"], box["h"]) * 0.25))
    ox0, oy0 = int(max(0, box["x"] - pad)), int(max(0, box["y"] - pad))
    ox1, oy1 = int(min(iw, box["x"] + box["w"] + pad)), int(min(ih, box["y"] + box["h"] + pad))
    ix0, iy0 = int(max(0, box["x"])), int(max(0, box["y"]))
    ix1, iy1 = int(min(iw, box["x"] + box["w"])), int(min(ih, box["y"] + box["h"]))
    outer = image[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return None
    keep = np.ones(outer.shape[:2], dtype=bool)
    keep[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = False
    ring = outer[keep]
    if ring.size == 0 or len(ring) < _PLATE_RING_MIN_PIXELS:
        return None
    ring = ring.astype(np.float32)
    plate = np.median(ring, axis=0)
    uniformity = float((np.linalg.norm(ring - plate, axis=1) < _PLATE_RING_TOLERANCE).mean())
    if uniformity < _PLATE_RING_UNIFORMITY:
        return None
    return plate, uniformity


def _ink_mask(crop, plate_prior=None):
    """Return (mask, confidence) using border-estimated background contrast.

    Glyph-tight OCR boxes can contaminate the border estimate with ink. In that
    case the contrast mask selects the white plate and text colour inference flips
    black copy to white-on-white (002's KRACHTSPORT headline). Prefer the minority
    luminance class when the estimated border is demonstrably closer to that class.

    That minority rule assumes ink is the SMALLER luminance class, which an ultra-heavy
    display headline breaks: 067's red "WE'RE SAYING GOODBYE" is 54% of its own tight box,
    so the plate is the minority and both heuristics elect the plate as ink (#f6f6f6
    instead of #fb0202) and the line renders invisible. ``plate_prior`` — from
    ``_exterior_plate_prior`` — breaks that tie with pixels the box cannot contaminate,
    and is applied only to REVERSE a demonstrated inversion (the elected ink sits on the
    true plate while the rejected class does not), never to retune a shade.
    """
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
    else:
        minority = _minority_luminance_mask(crop)
        if minority is not None and minority.any() and (~minority).any():
            majority = ~minority
            dist_minority = float(np.linalg.norm(
                np.median(crop[minority].astype(np.float32), axis=0) - bg,
            ))
            dist_majority = float(np.linalg.norm(
                np.median(crop[majority].astype(np.float32), axis=0) - bg,
            ))
            if dist_minority + 12.0 < dist_majority:
                mask = minority
                ratio = float(mask.mean())
    mask = _resolve_ink_polarity(crop, mask, plate_prior)
    mask = _clean_ink_mask(mask)
    ratio = float(mask.mean())
    if mask.any() and (~mask).any():
        plate = np.median(crop[~mask].astype(np.float32), axis=0)
        delta = np.sqrt(np.sum((crop.astype(np.float32) - plate) ** 2, axis=2))
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
# Body/headline copy should stay plain editable text. Thin "understroke" bands from
# AA / peel edges become exploded stroke layers downstream; only keep clear authored
# outlines (wider rim + strong fill↔stroke split).
_PLAIN_EDITABLE_ROLES = frozenset({
    "body", "headline", "subheadline", "caption", "footer", "disclaimer",
    "eyebrow", "offer", "cta",
})
_PLAIN_TEXT_STROKE_MIN_WIDTH = 2.5
_PLAIN_TEXT_STROKE_MIN_DISTANCE = 85.0


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
    # Cap outline width so a thick band cannot plate over the glyph fill in Figma.
    # Authored outlines sit outside the fill; CENTER/INSIDE covers letters.
    width = round(min(float(boundary_depth), 8.0), 1)
    fill_hex = _rgb_hex(interior_rgb)
    stroke_hex = _rgb_hex(rim_rgb)
    # Thin bands need a stronger fill↔stroke split; weak jumps are almost always AA.
    if width <= 2.0 and _colour_distance(stroke_hex, fill_hex) < _STROKE_COLOR_DISTANCE + 20:
        return None
    return {
        "kind": "flat",
        "color": stroke_hex,
        "width": width,
        "align": "OUTSIDE",
        "strokeAlign": "OUTSIDE",
    }, fill_hex


def _stroke_is_authored_outline(stroke: Optional[dict], fill_hex: Optional[str]) -> bool:
    """True when a detected stroke looks like a real marketing outline, not AA peel."""
    if not isinstance(stroke, dict):
        return False
    try:
        width = float(stroke.get("width", stroke.get("weight", 0)) or 0)
    except (TypeError, ValueError):
        return False
    if width < _PLAIN_TEXT_STROKE_MIN_WIDTH:
        return False
    stroke_hex = stroke.get("color") or stroke.get("paint")
    if isinstance(stroke_hex, dict):
        stroke_hex = stroke_hex.get("color")
    if not stroke_hex or not fill_hex:
        return width >= _PLAIN_TEXT_STROKE_MIN_WIDTH + 0.5
    return _colour_distance(str(stroke_hex), str(fill_hex)) >= _PLAIN_TEXT_STROKE_MIN_DISTANCE


def _prefer_plain_editable_text(lines: list[dict]) -> None:
    """Drop weak stroke/understroke effects on body/headline so text stays editable.

    Effects that fail a strong local outline match are left off the style (plate /
    slice paths handle lockups elsewhere) instead of inventing peel stroke layers.
    """
    for line in lines:
        role = str(line.get("role") or (line.get("meta") or {}).get("role") or "").lower()
        if role and role not in _PLAIN_EDITABLE_ROLES:
            continue
        style = line.get("style")
        if not isinstance(style, dict):
            continue
        stroke = style.get("stroke")
        if not stroke:
            continue
        fill = style.get("fill") if isinstance(style.get("fill"), dict) else {}
        fill_hex = fill.get("color") if isinstance(fill, dict) else None
        if fill_hex is None:
            fill_hex = style.get("color")
        if _stroke_is_authored_outline(stroke, fill_hex):
            continue
        style["stroke"] = None
        meta = line.setdefault("meta", {})
        meta["plain_text_stroke_suppressed"] = True
        meta["suppressed_stroke"] = stroke
        # Word-level stroke copies of the same weak rim would re-explode runs.
        for word in line.get("words") or []:
            if not isinstance(word, dict):
                continue
            wstyle = word.get("style")
            if isinstance(wstyle, dict) and wstyle.get("stroke"):
                wstyle["stroke"] = None


def _shadow_from_mask(crop, mask) -> Optional[dict]:
    """Detect an authored *offset* drop-shadow under glyph ink.

    Ink isolation usually absorbs the dark satellite into the same mask as the
    fill, so concentric outline logic cannot see it. Split ink by luminance into
    a bright fill cluster and a darker satellite; when the dark centroid is
    clearly shifted (and not a concentric outline ring), emit a DROP_SHADOW.
    Fail closed on unimodal ink, busy plates, and near-zero offsets.
    """
    import numpy as np

    m = np.asarray(mask).astype(bool)
    total = int(m.sum())
    if total < 80 or min(m.shape) < 12:
        return None
    outside = ~m
    if int(outside.sum()) < 30:
        return None
    bg_samples = crop[outside].astype(np.float32)
    if float(np.max(np.std(bg_samples, axis=0))) > 36.0:
        return None
    lum = crop.astype(np.float32).mean(axis=2)
    ink_lum = lum[m]
    if ink_lum.size < 80:
        return None
    p20, p50, p80 = np.percentile(ink_lum, [20, 50, 80])
    # Need a real light/dark split inside the ink (fill vs shadow), not AA noise.
    if float(p80 - p20) < 55.0:
        return None
    split = float((p20 + p80) / 2.0)
    dark = m & (lum <= split)
    bright = m & (lum >= max(split, p50))
    dark_n, bright_n = int(dark.sum()), int(bright.sum())
    if dark_n < max(24, int(total * 0.12)) or bright_n < max(24, int(total * 0.18)):
        return None
    # Shadow satellite should not dominate the glyph fill.
    if dark_n > bright_n * 1.35:
        return None
    ys_d, xs_d = np.nonzero(dark)
    ys_b, xs_b = np.nonzero(bright)
    dx = float(xs_d.mean() - xs_b.mean())
    dy = float(ys_d.mean() - ys_b.mean())
    if abs(dx) < 2.0 and abs(dy) < 2.0:
        return None
    if abs(dx) > 24.0 or abs(dy) > 24.0:
        return None
    dark_rgb = np.median(crop[dark].astype(np.float32), axis=0)
    bright_rgb = np.median(crop[bright].astype(np.float32), axis=0)
    # Dark cluster must actually be darker / distinct from the fill.
    if float(np.mean(dark_rgb)) >= float(np.mean(bright_rgb)) - 25.0:
        return None
    if float(np.linalg.norm(dark_rgb - bright_rgb)) < 40.0:
        return None
    # Anti-aliased edges of ordinary flat text form a "bright" pseudo-cluster that
    # is just a blend between the true ink and the plate behind it (101's checklist:
    # dark slate ink on a white card read as grey fill + dark "shadow", shipping
    # near-invisible grey text). A real fill+shadow pair has a fill colour OFF the
    # ink<->plate blend line; reject bright clusters that sit on it.
    bg_rgb = np.median(bg_samples, axis=0)
    axis = dark_rgb - bg_rgb
    norm = float(np.dot(axis, axis))
    if norm > 1.0:
        t = float(np.dot(bright_rgb - bg_rgb, axis) / norm)
        predicted = bg_rgb + max(0.0, min(1.0, t)) * axis
        if 0.03 < t < 0.97 and float(np.linalg.norm(bright_rgb - predicted)) < 24.0:
            return None
    # Symmetric guard for INVERTED copy (white text on a dark card, 009's tweet body):
    # the "dark satellite" is just the grey anti-aliased collar between the white fill
    # and the black background, i.e. it sits ON the background<->fill blend line. A real
    # drop shadow is a distinct ink offset OFF that line. Without this the collar ships a
    # light-grey DROP_SHADOW that renders as a lighter plate/scrim beside every line.
    axis_fill = bright_rgb - bg_rgb
    norm_fill = float(np.dot(axis_fill, axis_fill))
    if norm_fill > 1.0:
        t_dark = float(np.dot(dark_rgb - bg_rgb, axis_fill) / norm_fill)
        predicted_dark = bg_rgb + max(0.0, min(1.0, t_dark)) * axis_fill
        if 0.03 < t_dark < 0.97 and float(np.linalg.norm(dark_rgb - predicted_dark)) < 24.0:
            return None
    opacity = max(0.22, min(0.72, (255.0 - float(np.mean(dark_rgb))) / 255.0 * 0.85))
    radius = max(2.0, min(14.0, 0.55 * (abs(dx) + abs(dy)) + 1.5))
    return {
        "type": "DROP_SHADOW",
        "color": _rgb_hex(dark_rgb),
        "opacity": round(opacity, 3),
        "offset": {"x": int(round(dx)), "y": int(round(dy))},
        "radius": round(radius, 2),
        "spread": 0.0,
        "visible": True,
    }


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance for an (r, g, b) triple in 0..255."""
    channels = []
    for value in rgb:
        c = max(0.0, min(1.0, float(value) / 255.0))
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex_colour: str, plate_rgb) -> float:
    """WCAG contrast ratio between a hex fill and the plate's median RGB."""
    l1 = _relative_luminance(_hex_rgb(hex_colour))
    l2 = _relative_luminance(plate_rgb)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _is_ink_plate_blend(hex_colour: str, fallback_hex: str, plate_rgb) -> bool:
    """True when a colour sits ON the blend segment between the robust ink sample
    and the plate — the signature of anti-aliased edge pixels, not authored paint."""
    import numpy as np

    ink = np.asarray(_hex_rgb(fallback_hex), dtype=np.float32)
    plate = np.asarray(plate_rgb, dtype=np.float32)
    probe = np.asarray(_hex_rgb(hex_colour), dtype=np.float32)
    axis = ink - plate
    norm = float(np.dot(axis, axis))
    if norm <= 1.0:
        return False
    t = float(np.dot(probe - plate, axis) / norm)
    predicted = plate + max(0.0, min(1.0, t)) * axis
    return 0.02 < t < 0.98 and float(np.linalg.norm(probe - predicted)) < 20.0


def _paint_from_mask(crop, mask, fallback_hex: str) -> dict:
    """Best-effort fill/stroke/effect description for the painted ink, in addition to the
    single flattened colour used elsewhere for backward compatibility."""
    fill = {"kind": "flat", "color": fallback_hex}
    stroke = None
    effects: list[dict] = []
    plate_rgb = None
    try:
        import numpy as np

        outside = ~np.asarray(mask).astype(bool)
        if outside.any():
            plate_rgb = np.median(crop[outside].astype(np.float32), axis=0)
    except Exception:
        plate_rgb = None
    gradient = _dominant_axis_gradient(crop, mask)
    if gradient is not None and plate_rgb is not None:
        # Thin small text defeats the eroded-core sampling and the "gradient" is just
        # anti-aliasing: every stop lies on the ink<->plate blend line (101's
        # "durability" shipped a grey AA pseudo-gradient over #010101 ink). An
        # authored gradient has at least one stop off that line.
        stops = [s.get("color") for s in gradient.get("stops") or [] if s.get("color")]
        if stops and all(_is_ink_plate_blend(c, fallback_hex, plate_rgb) for c in stops):
            gradient = None
    if gradient is not None:
        fill = gradient
    stroke_result = _stroke_from_mask(crop, mask)
    if stroke_result is not None:
        stroke, interior_hex = stroke_result
        if gradient is None:
            fill = {"kind": "flat", "color": interior_hex}
    else:
        # Only hunt for offset shadows when there is no concentric outline band —
        # outline + shadow double-counting invents noisy Figma effects.
        shadow = _shadow_from_mask(crop, mask)
        if shadow is not None:
            effects.append(shadow)
            # Shadow satellites pollute axis gradients; keep a flat fill from the
            # bright ink cluster so the glyph colour stays readable.
            try:
                import numpy as np

                lum = crop.astype(np.float32).mean(axis=2)
                bright = mask & (lum >= float(np.percentile(lum[mask], 60)))
                if bright.any():
                    fill = {
                        "kind": "flat",
                        "color": _rgb_hex(np.median(crop[bright].astype(np.float32), axis=0)),
                    }
            except Exception:
                pass
    # Contrast sanity (no-outline paths only — knockout/outline text legitimately
    # matches its plate): an effect/gradient override must not ship a fill that
    # blends into its own plate when the robust ink-core sample contrasts fine.
    # 101's checklist emitted #9b9b9b on a white card (ghost text) while the
    # ink-core sample was #010101; re-sample in that case and drop the effects
    # derived from the same misread ink split.
    if stroke is None and plate_rgb is not None:
        if fill.get("kind") == "linear":
            fill_colours = [s.get("color") for s in fill.get("stops") or [] if s.get("color")]
        else:
            fill_colours = [fill.get("color")] if fill.get("color") else []
        if fill_colours:
            worst = min(_contrast_ratio(c, plate_rgb) for c in fill_colours)
            fallback_contrast = _contrast_ratio(fallback_hex, plate_rgb)
            if worst < 3.0 and fallback_contrast >= worst * 1.6:
                fill = {"kind": "flat", "color": fallback_hex}
                effects = []
    return {"fill": fill, "stroke": stroke, "effects": effects}


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


_FOREIGN_INK_CHROMA = 45.0
_FOREIGN_INK_MIN_FRAC = 0.04
_FOREIGN_INK_MIN_GLYPH_FRAC = 0.30
_FOREIGN_INK_MAX_BG_CHROMA = 22.0
_FOREIGN_INK_MIN_RULE_CHROMA = 100.0


def _detect_foreign_strike_ink(crop, mask, known_strike: bool = False):
    """Detect a saturated foreign-hue rule (a hand-drawn strike/underline) laid over
    achromatic glyph ink and return ``(glyph_mask, info)``.

    A struck word (091) carries a saturated red diagonal over black copy. Both the
    fill sampler and the density-based weight estimate otherwise ingest that red — the
    struck words render dark red and their inflated ink density fakes a Bold weight.
    The strike is *chromatic* (high R/G/B spread) while body/headline glyphs are
    *achromatic* (near-grey), so classifying ink by chroma cleanly separates them even
    when the strike is nearly half the ink and heavily anti-aliased (a colour-axis /
    minority test fails there because the blended median is itself reddish).

    Stripping the chromatic rule fixes fill and weight at the source and lets the
    strike be re-emitted as a decoration. Genuinely coloured text (a whole red word, a
    blue headline) has little or no achromatic glyph mass and so fails the glyph-mass
    gate — it is returned unchanged. A compact chromatic blob (coloured price digits
    mid-line) fails the elongation / span gates. ``info`` is ``{"color","box","angle"}``
    (box in crop-local pixel coords), or the mask is returned untouched with
    ``info=None``.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total < 40 or min(m.shape[:2]) < 4:
        return mask, None
    ch, cw = crop.shape[:2]
    # A genuine strike is drawn over text on a clean (achromatic) plate; the strike ink
    # is the only chroma present. Text on a COLOURED plate/photo (a product can, an
    # orange CTA button, a lifestyle photo) also yields chromatic ink pixels — from the
    # plate bleeding through the glyph mask — but there the BACKGROUND itself is
    # coloured. Refuse to treat plate-bleed as a strike: require an achromatic border.
    bw = max(1, min(3, ch // 5, cw // 5))
    border = np.concatenate([
        crop[:bw].reshape(-1, 3), crop[-bw:].reshape(-1, 3),
        crop[:, :bw].reshape(-1, 3), crop[:, -bw:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(border, axis=0)
    if float(bg.max() - bg.min()) > _FOREIGN_INK_MAX_BG_CHROMA:
        return mask, None
    ink = crop[m].astype(np.float32)
    chroma = ink.max(axis=1) - ink.min(axis=1)
    foreign_ink = chroma > _FOREIGN_INK_CHROMA
    foreign_frac = float(foreign_ink.mean())
    glyph_frac = 1.0 - foreign_frac
    min_frac = _FOREIGN_INK_MIN_FRAC
    if foreign_frac < min_frac or glyph_frac < _FOREIGN_INK_MIN_GLYPH_FRAC:
        return mask, None
    foreign_pixels = ink[foreign_ink]
    # The rule ink must be strongly saturated (a red/blue marker), not the mildly
    # chromatic ink of type photographed on a tinted surface.
    if float(np.median(foreign_pixels.max(axis=1) - foreign_pixels.min(axis=1))) < _FOREIGN_INK_MIN_RULE_CHROMA:
        return mask, None
    # The rule must be one hue, not a scatter of coloured noise: most foreign pixels
    # should share a dominant colour channel (a red scribble is R-dominant throughout).
    dominant = np.argmax(foreign_pixels, axis=1)
    counts = np.bincount(dominant, minlength=3)
    if float(counts.max()) / max(1.0, float(foreign_pixels.shape[0])) < 0.6:
        return mask, None
    ys, xs = np.nonzero(m)
    foreign_full = np.zeros(m.shape[:2], dtype=bool)
    foreign_full[ys[foreign_ink], xs[foreign_ink]] = True
    fy, fx = np.nonzero(foreign_full)
    if fy.size < 12:
        return mask, None
    points = np.column_stack([fx.astype(np.float32), fy.astype(np.float32)])
    centre = points.mean(axis=0)
    covariance = np.cov(points - centre, rowvar=False)
    try:
        evals, evecs = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return mask, None
    major = float(max(evals.max(), 1e-3))
    minor = float(max(evals.min(), 1e-3))
    # A strike is elongated (thin perpendicular to its run) even when drawn diagonally.
    elong_gate = 2.5 if known_strike else 4.0
    if major / minor < elong_gate:
        return mask, None
    axis_vec = evecs[:, int(np.argmax(evals))]
    proj = (points - centre) @ axis_vec
    span = float(proj.max() - proj.min())
    span_gate = cw * (0.25 if known_strike else 0.40)
    if span < max(16.0, span_gate):
        return mask, None
    glyph_mask = m & ~foreign_full
    glyph_count = int(glyph_mask.sum())
    if glyph_count < max(40, int(total * _FOREIGN_INK_MIN_GLYPH_FRAC)):
        return mask, None
    # Classify the rule by its vertical position over the GLYPH ink: a strikethrough
    # crosses the middle, an underline rides the bottom. This keeps a coloured
    # underline (002's "€49") from being emitted as a strike.
    gy, _gx = np.nonzero(glyph_mask)
    gy0, gy1 = float(gy.min()), float(gy.max())
    denom = max(1.0, gy1 - gy0)
    rel_centre = (float(centre[1]) - gy0) / denom
    if rel_centre <= 0.68:
        kind = "strikethrough"
    elif rel_centre >= 0.74:
        kind = "underline"
    else:
        kind = "strikethrough"
    colour = _rgb_hex(np.median(foreign_pixels, axis=0))
    box = {
        "x": int(fx.min()), "y": int(fy.min()),
        "w": int(fx.max() - fx.min() + 1), "h": int(fy.max() - fy.min() + 1),
    }
    angle = math.degrees(math.atan2(float(axis_vec[1]), float(axis_vec[0])))
    # Endpoints and mean thickness of the actual swipe, along its own (PCA) axis. A
    # hand-drawn strike is a long diagonal marker stroke that overshoots the glyph run
    # (091's swipe starts LEFT of "Foggy" and rises across it); a flat mid-height rule
    # spanning the text box is a visibly different mark. These let the caller re-emit
    # the rule as a real vector at its measured angle/length/weight instead.
    if axis_vec[0] < 0:
        axis_vec = -axis_vec
        proj = -proj
    p0 = centre + axis_vec * float(proj.min())
    p1 = centre + axis_vec * float(proj.max())
    thickness = float(fy.size) / max(1.0, span)
    return glyph_mask, {
        "color": colour, "box": box, "angle": round(angle, 2), "kind": kind,
        "p0": [float(p0[0]), float(p0[1])], "p1": [float(p1[0]), float(p1[1])],
        "span": round(span, 2), "thickness": round(thickness, 2),
    }


def _collar_box(box: dict, image=None) -> dict:
    """Grow a tight WORD box by a narrow exterior collar for plate sampling.

    ``_painted_geometry`` estimates the background from the crop's BORDER RING. A box
    that hugs the glyphs puts that ring on ink, so the ink/plate polarity can invert
    (002: black all-caps words touching their crop border read as white runs). A word
    sits INSIDE its line's plate, so a few pixels of collar reliably buys real plate.

    NOT usable on the LINE path, though the same polarity inversion happens there
    (067's red headline samples the plate white and renders invisible): a line box's
    collar can cross a plate boundary entirely, so padding it merely moves the
    inversion around. Measured over the 15 replayable fixtures, collaring the line
    path rescued 7 lines (067 'WE'RE SAYING GOODBYE' #f6f6f6 -> #fb0202) but pushed 12
    others to near-white (101 'NOT ALL TPU TUBES' #000000 -> #ffffff, 066 'OUR
    COMPETITOR', 094 'CAFFEINE', 091 'neuton') — a net regression. The line-path fix
    belongs in ``_ink_mask``'s existing minority-luminance guard, which already owns
    this exact case, not in the sampling window.

    The pad scales with the box so it never swallows a neighbour, and is clamped to
    the image so the crop stays in bounds.
    """
    box = _clean_box(box)
    pad = max(3.0, min(16.0, min(box["w"], box["h"]) * 0.25))
    x0, y0 = box["x"] - pad, box["y"] - pad
    x1, y1 = box["x"] + box["w"] + pad, box["y"] + box["h"] + pad
    if image is not None:
        ih, iw = image.shape[:2]
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(iw), x1), min(float(ih), y1)
    return {"x": x0, "y": y0, "w": max(1.0, x1 - x0), "h": max(1.0, y1 - y0)}


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
    mask, confidence = _ink_mask(crop, plate_prior=_exterior_plate_prior(image, box))
    if mask is None or not mask.any():
        painted, baseline, fallback_conf, _ = _fallback_geometry(line, snap_deg)
        paint = {"fill": dict(_FLAT_FILL_BLACK), "stroke": None}
        return painted, baseline, "#000000", fallback_conf, None, paint

    # Strip a foreign-hue strike/underline rule (091's red scribbles) BEFORE any
    # geometry, colour, or density sampling so the struck word keeps its true black
    # fill and regular weight, and record it so the decoration is re-emitted. OCR may
    # already have flagged the line (meta.strikethrough) — trust that as a strong prior.
    known_strike = bool((line.get("meta") or {}).get("strikethrough"))
    glyph_mask, strike_info = _detect_foreign_strike_ink(crop, mask, known_strike=known_strike)
    if strike_info is not None:
        # Always strip the foreign rule ink so fill colour and ink-density weight are
        # measured from the glyphs alone (fixes the dark-red fill + fake-Bold weight).
        mask = glyph_mask
        meta = line.setdefault("meta", {})
        if strike_info.get("color") and not meta.get("strike_ink_color"):
            meta["strike_ink_color"] = strike_info["color"]
        # Only a mid-height rule authors a STRIKETHROUGH here; a coloured underline is
        # left to the dedicated price/native-rule paths (avoids 002's "€49" underline
        # being emitted as a strike).
        if strike_info.get("kind") == "strikethrough" and not meta.get("strikethrough"):
            meta["strikethrough"] = True
            if not meta.get("strikethrough_box"):
                b = strike_info["box"]
                meta["strikethrough_box"] = {
                    "x": x0 + b["x"], "y": y0 + b["y"], "w": b["w"], "h": b["h"],
                }
        # Record the swipe's measured geometry in IMAGE coords so the caller can emit it
        # as a precise vector rule rather than a flat box-width line.
        if strike_info.get("p0") and strike_info.get("p1"):
            meta["strike_ink_shape"] = {
                "kind": strike_info.get("kind"),
                "x0": round(float(x0 + strike_info["p0"][0]), 2),
                "y0": round(float(y0 + strike_info["p0"][1]), 2),
                "x1": round(float(x0 + strike_info["p1"][0]), 2),
                "y1": round(float(y0 + strike_info["p1"][1]), 2),
                "color": strike_info.get("color"),
                "thickness": strike_info.get("thickness"),
                "angle": strike_info.get("angle"),
                "span": strike_info.get("span"),
            }

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
        # (Foreign strike ink is already removed from ``mask`` above.)
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


def _strike_span_fraction(strike_box, painted_box) -> Optional[list]:
    """[start, end] x-fraction of a strike within its text box, or None.

    OCR reports the strike's bounding box (``strikethrough_box``) in image
    coordinates; expressing it as a fraction of the painted text box lets the
    renderer strike only the struck words regardless of later re-fit/reposition.
    A near-full-width strike returns None so the whole line is struck cleanly.
    """
    if not isinstance(strike_box, dict) or not isinstance(painted_box, dict):
        return None
    try:
        sx, sw = float(strike_box["x"]), float(strike_box["w"])
        px, pw = float(painted_box["x"]), float(painted_box["w"])
    except (TypeError, ValueError, KeyError):
        return None
    if pw <= 0 or sw <= 0:
        return None
    f0 = max(0.0, min(1.0, (sx - px) / pw))
    f1 = max(0.0, min(1.0, (sx + sw - px) / pw))
    if f1 <= f0:
        return None
    if f0 <= 0.04 and f1 >= 0.96:
        return None
    return [round(f0, 4), round(f1, 4)]


def _hand_swipe_rule(meta: dict, painted_box: Optional[dict]) -> Optional[dict]:
    """A hand-drawn strike's measured ink as a vector rule, or None to use a flat rule.

    ``meta.strike_ink_shape`` is written by ``_detect_foreign_strike_ink`` when a
    saturated foreign-hue stroke is found over achromatic glyphs — i.e. an annotation
    drawn on top of the type, not a typographic decoration.  Such a mark differs from a
    text-decoration line in three measurable ways, any one of which makes the flat rule
    a visibly wrong reproduction:

      * it runs at its own angle (091's swipe rises ~10° across "Foggy");
      * it is several times thicker than a decoration line (font_size * 0.06);
      * it overshoots the glyph run instead of stopping at the box edge, which
        ``_strike_span_fraction`` cannot express (it clamps to [0,1] of the box).

    Returns a ``native_decoration_shapes`` entry (the same schema
    ``_native_colored_price_rules`` emits) when the measured geometry is trustworthy:
    a real span, a sane thickness, and endpoints.  Returns None for a short/degenerate
    detection so the caller keeps the conservative flat rule.
    """
    shape = meta.get("strike_ink_shape") if isinstance(meta, dict) else None
    if not isinstance(shape, dict) or shape.get("kind") != "strikethrough":
        return None
    try:
        x0, y0 = float(shape["x0"]), float(shape["y0"])
        x1, y1 = float(shape["x1"]), float(shape["y1"])
        thickness = float(shape.get("thickness") or 0.0)
        span = float(shape.get("span") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1, thickness, span)):
        return None
    # A degenerate/short detection carries no better information than the flat rule.
    box = _clean_box(painted_box)
    if span < max(16.0, 0.20 * max(1.0, box["w"])) or thickness < 1.0:
        return None
    colour = shape.get("color")
    if not colour:
        return None
    return {
        "kind": "strikethrough",
        "x0": round(x0, 2), "y0": round(y0, 2), "x1": round(x1, 2), "y1": round(y1, 2),
        "color": colour,
        "thickness": round(max(1.0, min(24.0, thickness)), 2),
        "confidence": round(min(1.0, span / max(1.0, box["w"])), 4),
        "source": "hand-swipe-ink",
    }


def _native_text_decoration(mask, text: str) -> tuple[Optional[str], Optional[dict]]:
    """Recognize only an unmistakable continuous underline/strike rule.

    Short glyph bars (E, T, hyphens) must not turn into Figma text decoration.  We
    therefore require a nearly continuous run spanning most of the painted text width.
    Anything ambiguous remains part of the exact text fallback/plate pixels.
    """
    if mask is None or not str(text or "").strip() or "_" in str(text):
        return None, None
    # A stray 1-2 char OCR fragment ("I &") whose glyph bars span the box must never
    # become a strikethrough; a real struck token has at least two letters/digits.
    if sum(1 for ch in str(text) if ch.isalnum()) < 2:
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


def _native_colored_price_rules(image, line: dict) -> list[dict]:
    """Recover authored coloured strike/underline rules around a price as vectors.

    OCR commonly merges the separator arrow into a price line while the smaller
    per-price observations are later deduped.  Saturated red rules are strong pixel
    evidence and should survive that dedup as editable line shapes, not disappear into
    a raster slice.  The detector is intentionally narrow: a currency token plus one
    long, saturated red component that is either diagonal through the text or horizontal
    at its lower edge.
    """
    if image is None or not re.search(r"[€$£]\s*\d", str(line.get("text") or "")):
        return []
    try:
        import cv2
        import numpy as np

        box = _clean_box(line.get("box"))
        ih, iw = image.shape[:2]
        x0 = max(0, min(iw, int(math.floor(box["x"]))))
        y0 = max(0, min(ih, int(math.floor(box["y"]))))
        x1 = max(x0, min(iw, int(math.ceil(box["x"] + box["w"]))))
        y1 = max(y0, min(ih, int(math.ceil(box["y"] + box["h"]))))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            return []
        rgb = crop.astype(np.int16)
        red = (
            (rgb[:, :, 0] >= 120)
            & (rgb[:, :, 0] - rgb[:, :, 1] >= 45)
            & (rgb[:, :, 0] - rgb[:, :, 2] >= 45)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
        rules = []
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            cw = int(stats[idx, cv2.CC_STAT_WIDTH])
            ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if area < 12 or max(cw, ch) < max(18, int(crop.shape[1] * 0.30)):
                continue
            ys, xs = np.nonzero(labels == idx)
            points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            centre = points.mean(axis=0)
            covariance = np.cov(points - centre, rowvar=False)
            values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, int(np.argmax(values))]
            if axis[0] < 0:
                axis = -axis
            projection = (points - centre) @ axis
            p0 = centre + axis * float(projection.min())
            p1 = centre + axis * float(projection.max())
            span = float(projection.max() - projection.min())
            if span < max(18.0, crop.shape[1] * 0.30):
                continue
            angle = math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))
            relative_y = float(centre[1]) / max(1.0, crop.shape[0] - 1)
            if abs(angle) <= 12.0 and relative_y >= 0.72:
                kind = "underline"
            elif 10.0 <= abs(angle) <= 45.0 and 0.20 <= relative_y <= 0.85:
                kind = "strikethrough"
            else:
                continue
            colour = _rgb_hex(np.median(crop[labels == idx].astype(np.float32), axis=0))
            thickness = max(2.0, min(8.0, area / max(1.0, span)))
            rules.append({
                "kind": kind,
                "x0": round(float(x0 + p0[0]), 2),
                "y0": round(float(y0 + p0[1]), 2),
                "x1": round(float(x0 + p1[0]), 2),
                "y1": round(float(y0 + p1[1]), 2),
                "color": colour,
                "thickness": round(thickness, 2),
                "confidence": round(min(1.0, span / max(1.0, crop.shape[1])), 4),
                "source": "saturated-price-rule",
            })
        return sorted(rules, key=lambda item: (item["kind"], item["x0"], item["y0"]))
    except Exception:
        return []


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
    # the later Figma render-fit loop. Near-solid ink maps to ExtraBold (800) so
    # UI chrome like Codia's "Post" (Inter ExtraBold) is not stuck at Bold 700.
    if density >= 0.58:
        return 800
    if density >= 0.46:
        return 700
    if density >= 0.34:
        return 600
    if density <= 0.12 and painted_box.get("h", 0) >= 10:
        return 300
    return 400


def _style_name(weight: int, italic: bool = False) -> str:
    if weight >= 800:
        base = "Extra Bold"
    elif weight >= 700:
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
    try:
        target_weight = float(style.get("fontWeight") or 400)
    except (TypeError, ValueError):
        target_weight = 400.0
    usable = [
        c for c in candidates
        if isinstance(c, dict) and c.get("path") and os.path.exists(c["path"])
    ]

    def _cand_weight(candidate: dict) -> float:
        try:
            return float(candidate.get("weight") or 400)
        except (TypeError, ValueError):
            return 400.0

    # Prefer a file whose weight matches the declared node weight; a Regular path
    # must not measure Bold text (weight-split siblings inherit the parent list).
    usable.sort(key=lambda c: abs(_cand_weight(c) - target_weight))
    paths = []
    for candidate in usable:
        if abs(_cand_weight(candidate) - target_weight) <= 150:
            paths.append(candidate["path"])
    if target_weight >= 600:
        paths += ["arialbd.ttf"]
    paths += [c["path"] for c in usable if c["path"] not in paths]
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
    # Height fit scales the WHOLE block — glyph AND leading — by one factor, because a
    # font's natural line spacing is proportional to its size. The earlier approach held
    # line_height fixed and subtracted it whole (glyph_room = avail_h - gaps*line_height),
    # then shrank only the glyph into what was left. For display headlines whose synthetic
    # leading (~1.2*fontSize) approaches the painted per-line pitch, that left almost no
    # room for the glyph and collapsed the size: 013's "We NEVER / do this!" (ink ~150px)
    # shrank to ~40% of ink height. Scaling leading with the glyph keeps multi-line fits at
    # ink height and still shrinks only when the content genuinely overflows the box.
    # (Single-line content_h == glyph_h, so this is identical to the old path there.)
    height_scale = min(1.0, avail_h / max(1.0, content_h))
    target_scale = min(width_scale, height_scale)
    if target_scale < 0.999:
        new_size = max(1.0, font_size * target_scale)
        patch["fontSize"] = round(new_size, 2)
        # OCR boxes carry a measured line height for the original font. Keeping that
        # absolute value after shrinking a substitute font is a common source of
        # clipped final lines in both the preview and Figma. Single-line nodes need
        # the rescale just as much: an OCR fontSize over-measure (013 "snacks" fs
        # 59.7 → fitted 28) otherwise leaves lineHeight at 2.5x the fitted size and
        # the generous emitted box balloons to ~2.7x the ink height.
        patch["lineHeight"] = round(max(new_size * 1.12, line_height * target_scale), 2)

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

    # Hard floor: lineHeight must never sit below fontSize (preview/Figma clip ascenders).
    effective_size = float(patch.get("fontSize", font_size) or font_size)
    effective_lh = float(patch.get("lineHeight", line_height) or line_height)
    if effective_size > 0 and effective_lh < effective_size * 1.05:
        patch["lineHeight"] = round(effective_size * 1.12, 2)

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


_FAMILY_PATH_CACHE: dict = {}


def _resolve_family_path(family: Any, weight: int, italic: bool,
                         options: Optional[dict] = None) -> Optional[str]:
    """On-disk file for a DECLARED family, at the closest weight/slant.

    The platform-UI prior (and the design-time family stamp) OVERRIDE a line's
    family with a declared one. Keeping the outvoted candidate's ``path`` then
    leaves design.json and the preview drawing different fonts — and, worse,
    leaves the emitted ``fontSize`` fitted to a face Figma will never load
    (009's tweet body fits Lato-Medium at 34.54 where its own twin line, which
    resolved real Inter, fits 35.77; Figma then renders the declared Inter 6%
    narrow). Resolving the declared family to a real file keeps label, path, fit
    and preview on ONE font.

    Returns ``None`` when the family is not installed, so callers can fall back to
    dropping the stale path rather than lying about it.
    """
    name = _norm_family(family)
    if not name:
        return None
    key = (name, int(weight), bool(italic))
    if key in _FAMILY_PATH_CACHE:
        return _FAMILY_PATH_CACHE[key]
    opts = dict(options or {})
    # The declared family is a NAME, not a member of any caller-pinned universe:
    # resolve it against the ambient corpus (curated OFL + installed system faces).
    opts.pop("font_files", None)
    fonts = list(_discover_google_fonts(opts))
    try:
        fonts.extend(_discover_fonts({"font_dirs": _platform_font_dirs()}))
    except Exception:
        pass
    matches = [meta for meta in fonts if _norm_family(meta.get("family")) == name]
    if not matches:
        _FAMILY_PATH_CACHE[key] = None
        return None

    def rank(meta: dict) -> tuple:
        path = str(meta.get("path") or "")
        try:
            from src import font_fit

            axes = font_fit.variable_axes(path)
        except Exception:
            axes = {}
        wght = axes.get("wght") if axes else None
        if wght and float(wght[0]) <= weight <= float(wght[2]):
            distance = 0.0          # a variable face dials to the exact weight
        else:
            distance = abs(float(meta.get("weight") or 400) - weight)
        is_italic = "italic" in str(meta.get("style") or "").lower()
        return (0 if is_italic == bool(italic) else 1, distance, path)

    best = str(sorted(matches, key=rank)[0].get("path") or "") or None
    if best and not os.path.exists(best):
        best = None
    _FAMILY_PATH_CACHE[key] = best
    return best


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
    # HARD italic gate: shape similarity can prefer an Italic variant on upright ink
    # (013: "We NEVER do this!" shipped Poppins 800 ITALIC — the soft -0.06 alignment
    # penalty was outscored by incidental shape overlap). When the measured shear is
    # decisively upright (<4°), italic/oblique variants are excluded outright; when
    # decisively slanted (>8°), upright variants are excluded. The ambiguous 4-8°
    # band keeps both and lets rendered-fit evidence decide.
    shear = profile.get("shear_angle")
    if shear is None and profile.get("italic") is not None:
        shear = 10.0 if profile.get("italic") else 0.0
    if shear is not None:
        def _is_italic_variant(m):
            s = str(m.get("style") or "") + " " + str(m.get("path") or "")
            return "italic" in s.lower() or "oblique" in s.lower()
        if abs(float(shear)) < 4.0:
            fonts = [m for m in fonts if not _is_italic_variant(m)] or fonts
        elif abs(float(shear)) > 8.0:
            fonts = [m for m in fonts if _is_italic_variant(m)] or fonts
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


# Vertical extents as a fraction of the CAP height, for a typical sans (cap 0.72em):
# ascenders reach slightly above the caps, the x-height is ~0.52em, descenders drop
# ~0.21em below the baseline. Used to predict how much of the em a line's ink SHOULD
# span given the glyphs it actually contains.
_ASCENDER_OF_CAP = 1.04
_XHEIGHT_OF_CAP = 0.72
_DESCENDER_OF_CAP = 0.29
_ASCENDER_CHARS = frozenset("bdfhklt")
_DESCENDER_CHARS = frozenset("gjpqy")
# 'i'/'j' carry a TITTLE (the dot), which clears x-height and lands at roughly cap
# height — below the true ascender. Ignoring it models "mini pumps" (101) as
# x-height-only ink when its ink really spans tittle→descender, inflating the fitted
# size by 26% (30.25 against a column of 24.0) and ejecting the row from its peers.
# Measured against 101's checklist column (ink_h 22 whose true size is 24.0): tittle
# as x-height +26.1%, as ascender -4.3%, as CAP height -1.3%.
_TITTLE_CHARS = frozenset("ij")


def _expected_ink_ratio(text: str, cap_ratio: float) -> float:
    """Fraction of the em that THIS line's ink bbox should span, from its own glyphs.

    Sizing a line as ``ink_height / cap_ratio`` silently assumes every line's ink spans
    exactly the caps. It does not: ink runs from the tallest glyph to the lowest, so the
    same font size measures ~0.72em all-caps, ~0.96em with both an ascender and a
    descender, and only ~0.52em for x-height-only copy like "no". The fixed ratio turns
    that glyph mix into phantom size differences — 009's tweet body is one size in the
    source yet emitted 37.0…50.0 — which is exactly the "similar elements should share a
    scale" complaint. Measuring the ratio per line removes the cause instead of snapping
    peers to a median afterwards.
    """
    text = str(text or "")
    letters = [ch for ch in text if ch.isalpha() or ch.isdigit()]
    if not letters:
        return cap_ratio
    has_ascender = any(ch in _ASCENDER_CHARS for ch in text if ch.islower())
    has_cap = any(ch.isupper() or ch.isdigit() for ch in text)
    has_tittle = any(ch in _TITTLE_CHARS for ch in text if ch.islower())
    has_descender = any(ch in _DESCENDER_CHARS for ch in text if ch.islower())
    if has_ascender:
        top = cap_ratio * _ASCENDER_OF_CAP
    elif has_cap or has_tittle:
        # A tittle tops out at ~cap height: 'mini pumps' is NOT x-height-only ink.
        top = cap_ratio
    else:
        top = cap_ratio * _XHEIGHT_OF_CAP
    bottom = cap_ratio * _DESCENDER_OF_CAP if has_descender else 0.0
    return max(0.2, top + bottom)


def _pre_font_signals(line: dict, painted: dict, mask, config: dict) -> dict:
    """Signals available cheaply, before any (expensive) font-file rendering: the
    estimated size/weight and a glyph-shear (italic) measurement from the ink mask
    alone. Used both to assemble the final style and to cluster same-style lines
    before font matching runs."""
    cap_ratio = max(0.45, min(0.90, _num(config.get("cap_height_ratio"), 0.72)))
    # The glyph-mix correction reads a MEASURED ink bbox. Without a mask the height is
    # the OCR line box, which already includes ascender/descender room for every line
    # regardless of glyphs, so correcting it would invent size differences instead of
    # removing them (and split a column's lines across style clusters).
    ink_ratio = (
        _expected_ink_ratio(line.get("text"), cap_ratio)
        if mask is not None and getattr(mask, "any", bool)() else cap_ratio
    )
    font_size = max(1.0, min(512.0, painted["h"] / ink_ratio if painted["h"] else line["box"]["h"] * 0.9))
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
        # Codia parity: emitted letterSpacing is always 0. Heuristic tracking is
        # retained only as a diagnostic on meta when render-fit records it.
        "letterSpacing": 0.0,
        "confidence": round(min(_num(line.get("conf"), 0.5), max(0.25, ink_confidence)), 4),
        "fill": (paint or {}).get("fill") or {"kind": "flat", "color": colour},
        "stroke": (paint or {}).get("stroke"),
        "effects": list((paint or {}).get("effects") or []),
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
    """Refine one line's emitted size by fitting its chosen font to its own ink
    mask (cluster representatives share matched candidates, but every line has
    its own painted geometry).  Applies the fitted ``fontSize`` when the fit
    passes ``min_score``; ``letterSpacing`` stays 0 (Codia parity) while the
    fitted tracking is recorded on ``meta.render_fit`` for diagnostics.
    Returns the fit mapping or ``None``.
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

        # Only a face we resolved BY NAME gets its variable axis driven. A
        # matcher-chosen face was scored and fitted at its file's own default
        # instance, so re-rendering it at some other weight would fit a face nothing
        # upstream evaluated — that perturbs fitted sizes, which shifts peer-cluster
        # medians, which flips 066's rows to the cluster majority weight and costs
        # text recall. `None` keeps the default instance, exactly as before.
        fit = font_fit.fit_line(
            line.get("text", ""), chosen["path"], mask,
            _num(style.get("fontSize"), 16.0), render_fit_options,
            weight=(_num(chosen.get("weight"), 400.0)
                    if chosen.get("family_resolved") else None),
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
        style["letterSpacing"] = 0.0
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


def _handwriting_crop_bytes(image, painted: dict, padding: int = 6) -> Optional[bytes]:
    """PNG bytes of a line's painted box cut from the ORIGINAL image, for the VLM.

    Uses ``vlm_client.crop_box_bytes`` so the crop/encoding convention matches every
    other VLM stage (vlm_ocr_judge et al).
    """
    if image is None or not isinstance(painted, dict):
        return None
    try:
        from PIL import Image

        from src import vlm_client

        return vlm_client.crop_box_bytes(Image.fromarray(image), painted, padding)
    except Exception:
        return None


def _apply_handwriting_gate(prepared: list[dict], image, cfg: Optional[dict],
                            run_dir: Optional[str]) -> dict:
    """Rasterize genuinely hand-lettered lines to pixel-exact chips (HARD spec §11).

    A hand-drawn/marker/script word has no library equivalent, so emitting it in the
    nearest sans is the most visible reconstruction failure there is (091's marker
    "Sharp" rendered as Barlow Condensed).  Rather than guess a font, chip the ORIGINAL
    ink — but ONLY for lines the VLM positively confirms as hand-drawn, because
    rasterizing typeset copy is the worse error (see ``handwriting`` module docstring:
    the cheap stats put "Sharp" on the *typeset* side of every threshold, so nothing
    cheaper than the VLM may make this call).

    The chip reuses the existing fidelity-fallback contract — ``meta.low_fidelity`` +
    ``meta.fallback_src`` become an image node in ``routing._text_fidelity_fallback`` —
    and stamps ``meta.handwriting``/``ocr_text``/``font_attempted``/``renderback_score``
    so the chip stays greppable and editable-aware downstream.

    Returns an evidence mapping for ``result["handwriting"]``.
    """
    try:
        from src import handwriting
    except Exception as exc:  # pragma: no cover - import guard
        return {"enabled": False, "note": f"import-error:{type(exc).__name__}"}
    if not handwriting.enabled(cfg):
        return {"enabled": False, "note": "disabled"}

    evidence: dict = {"enabled": True, "candidates": [], "checked": 0,
                      "rasterized": [], "kept_native": []}

    # Stage A on every line (cheap: no font rendering, no VLM).
    staged: list[tuple] = []
    for item in prepared:
        line = item["line"]
        stats = handwriting.ink_stats(item.get("font_mask"))
        item["hw_stats"] = stats
        is_candidate, signals = handwriting.stage_a_candidate(line, stats, cfg)
        item["hw_stage_a"] = signals
        if is_candidate:
            staged.append((line["id"], item["painted"]))
            evidence["candidates"].append(line["id"])

    if not staged:
        return evidence

    # Stage B on a bounded, ink-area-ranked subset.
    chosen = set(handwriting.select_candidates(staged, cfg))
    by_id = {item["line"]["id"]: item for item in prepared}
    for line_id in chosen:
        item = by_id.get(line_id)
        if item is None:
            continue
        line = item["line"]
        crop = _handwriting_crop_bytes(image, item["painted"])
        decision = handwriting.decide(line, item.get("hw_stats"), crop, cfg)
        evidence["checked"] += 1
        meta = line.setdefault("meta", {})
        vlm = decision.get("vlm") or {}
        meta["handwriting_evidence"] = {
            "reason": decision.get("reason"),
            "renderback_score": decision.get("renderback_score"),
            "font_attempted": decision.get("font_attempted"),
            "stroke_width_cv": (item.get("hw_stats") or {}).get("stroke_width_cv"),
            "vlm": {k: vlm.get(k) for k in ("available", "handwritten", "style",
                                            "confidence", "note")} if vlm else None,
        }
        if decision.get("handwriting"):
            # Positive VLM identification is worth recording even when the line stays
            # native: it is text-side evidence that this ink is drawn, not set, which
            # the scene-intent/merge side consumes when deciding in-image vs overlay.
            meta["handwriting"] = True
        if not decision.get("rasterize"):
            evidence["kept_native"].append({"id": line_id, "text": line.get("text"),
                                            "reason": decision.get("reason")})
            continue

        meta["ocr_text"] = line.get("text")
        meta["font_attempted"] = decision.get("font_attempted")
        meta["renderback_score"] = decision.get("renderback_score")
        reason = (f"handwriting: {decision.get('reason')} "
                  f"(font_attempted={decision.get('font_attempted')}, "
                  f"renderback={decision.get('renderback_score')})")
        meta["low_fidelity"] = True
        meta["fidelity_reason"] = reason
        fallback_src = _save_fallback_crop(image, item["mask"], item["painted"], run_dir,
                                           line["id"])
        if fallback_src:
            meta["fallback_src"] = fallback_src
        meta["substitution"] = {
            "from": "text", "to": "handwriting-chip", "reason": reason,
            "confidence": meta.get("fidelity_confidence"),
        }
        evidence["rasterized"].append({
            "id": line_id, "text": line.get("text"),
            "font_attempted": decision.get("font_attempted"),
            "renderback_score": decision.get("renderback_score"),
            "vlm_style": vlm.get("style"), "vlm_confidence": vlm.get("confidence"),
        })
    return evidence


def _platform_ui_cfg(cfg: Optional[dict]) -> dict:
    """Resolve the text_analysis mapping, accepting a bare options dict in tests."""
    tcfg = _text_cfg(cfg)
    if tcfg:
        return tcfg
    if isinstance(cfg, dict) and (
        "platform_ui_prior" in cfg or "platform_ui_family" in cfg or "platform_ui" in cfg
    ):
        return cfg
    return {}


def _platform_ui_prior_enabled(cfg: Optional[dict]) -> bool:
    """Whether body/UI copy should default to Inter (social / platform screenshots)."""
    tcfg = _platform_ui_cfg(cfg)
    if "platform_ui_prior" in tcfg:
        return bool(tcfg.get("platform_ui_prior"))
    policy = ((cfg or {}).get("routing") or {}).get("text_policy") or {}
    if policy.get("platform_ui_prior") is not None:
        return bool(policy.get("platform_ui_prior"))
    if str(policy.get("default_family") or "").lower() == "inter":
        return True
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()
    return archetype == "social_screenshot"


def _platform_ui_family(cfg: Optional[dict]) -> str:
    tcfg = _platform_ui_cfg(cfg)
    family = tcfg.get("platform_ui_family")
    if not family:
        policy = ((cfg or {}).get("routing") or {}).get("text_policy") or {}
        family = policy.get("default_family")
    return str(family or "Inter")


def _apply_platform_ui_font_prior(prepared: list[dict], cfg: Optional[dict],
                                  render_fit_options: dict) -> Optional[dict]:
    """Force Inter (or configured family) on platform-UI sans lines.

    Keeps a line's own family only when it is a confident non-sans (serif/script
    display) match — body/meta/stat lines on social chrome always become Inter.
    letterSpacing is forced to 0 (Codia parity).
    """
    if not _platform_ui_prior_enabled(cfg):
        return None
    family = _platform_ui_family(cfg)
    tcfg = _platform_ui_cfg(cfg)
    font_options = _font_options(tcfg)
    strong_keep = _num((tcfg.get("platform_ui") or {}).get("strong_keep"), 0.72)
    applied = []
    skipped = []
    for item in prepared:
        line = item.get("line") or {}
        style = line.get("style") or {}
        current = str(style.get("fontFamily") or "")
        if not current:
            continue
        def _force_words_to_family() -> None:
            for word in line.get("words") or []:
                if not isinstance(word, dict):
                    continue
                wstyle = word.get("style")
                if not isinstance(wstyle, dict):
                    continue
                w_current = str(wstyle.get("fontFamily") or "")
                if w_current and _norm_family(w_current) != _norm_family(family):
                    w_class = _family_class(
                        w_current,
                        ((wstyle.get("fontCandidates") or [{}])[0] or {}).get("path"),
                    )
                    if w_class in ("serif", "script"):
                        continue
                wstyle["fontFamily"] = family
                wstyle["letterSpacing"] = 0.0
                wcands = [c for c in (wstyle.get("fontCandidates") or []) if isinstance(c, dict)]
                if wcands:
                    w_promoted = dict(wcands[0])
                    w_promoted["family"] = family
                    wstyle["fontCandidates"] = [w_promoted] + wcands[1:]
                _reconcile_style_weight(wstyle)

        if _norm_family(current) == _norm_family(family):
            style["letterSpacing"] = 0.0
            _force_words_to_family()
            continue
        candidates = [c for c in (style.get("fontCandidates") or []) if isinstance(c, dict)]
        top = candidates[0] if candidates else {}
        line_class = _family_class(current, top.get("path"))
        fit_meta = (line.get("meta") or {}).get("render_fit") or {}
        score = _num(fit_meta.get("score"), 0.0)
        if isinstance(top.get("fit"), dict):
            score = max(score, _num(top["fit"].get("score"), 0.0))
        score = max(score, _num(top.get("score"), 0.0))
        # Distinctive display faces keep their match when the fit is strong.
        if line_class in ("serif", "script") and score >= strong_keep:
            skipped.append({"id": line.get("id"), "family": current, "class": line_class,
                            "score": score, "reason": "strong-non-sans"})
            continue
        style["fontFamily"] = family
        style["letterSpacing"] = 0.0
        weight = int(round(_num(style.get("fontWeight"), 400)))
        italic = "italic" in str(style.get("fontStyle") or "").lower()
        style_label = _style_name(weight, italic=italic)
        inter_cand = next(
            (c for c in candidates
             if _norm_family(c.get("family")) == _norm_family(family)),
            None,
        )
        if inter_cand is not None:
            promoted = dict(inter_cand)
            promoted["weight"] = int(round(_num(promoted.get("weight"), weight)))
            rest = [c for c in candidates if c is not inter_cand]
            style["fontCandidates"] = [promoted] + rest
        elif top:
            # The prior REPLACES the family, so the outvoted face's path must go with
            # it. Keeping that path made design.json and the preview draw different
            # fonts (009's tweet body: family "Inter", path Lato-Medium.ttf) and — the
            # real defect — left the emitted fontSize fitted to a face Figma never
            # loads, so the DELIVERABLE renders ~6% narrow. Resolve the declared family
            # to a real file and re-fit against it; only when it is not installed do we
            # fall back to the documented relabel (keep path, record local_family).
            promoted = dict(top)
            promoted["family"] = family
            promoted["weight"] = weight
            promoted["style"] = style_label
            promoted["source"] = promoted.get("source") or "platform-ui-prior"
            promoted["figma_loadable"] = True
            resolved = _resolve_family_path(family, weight, italic, font_options)
            if resolved:
                promoted["path"] = resolved
                promoted.pop("local_family", None)
                promoted.pop("fit", None)      # the old face's fit is not this face's
                # We picked this file BY NAME for a declared weight, so a variable
                # face must be dialled to it. Only such candidates carry the flag:
                # a matcher-chosen face was fitted at its file's own default instance
                # (Archivo[wdth,wght] defaults to wght 600, not the 400 its OS/2
                # record reports), so touching its axis would re-render something
                # nothing upstream evaluated and shift 066's peer clusters.
                promoted["family_resolved"] = True
            else:
                promoted["local_family"] = current
            style["fontCandidates"] = [promoted] + candidates[1:]
            if resolved:
                refit = _apply_line_render_fit(
                    line, item.get("font_mask"), item.get("painted"), render_fit_options,
                )
                if refit is not None:
                    item["line_fit"] = refit
        else:
            style["fontCandidates"] = [{
                "family": family, "style": style_label, "weight": weight,
                "score": max(score, 0.55), "source": "platform-ui-prior",
                "figma_loadable": True,
            }]
        _reconcile_style_weight(style)
        meta = line.setdefault("meta", {})
        meta["platform_ui_prior"] = {
            "from": current, "to": family, "previous_class": line_class,
        }
        # Word-level styles inherit forensic faces; force them too so weight-split
        # design nodes (121K / engagement counts) stay on the platform family.
        _force_words_to_family()
        applied.append(line.get("id"))
    if not applied and not skipped:
        return None
    return {
        "family": family,
        "applied": bool(applied),
        "applied_lines": applied,
        "skipped": skipped,
        "enabled": True,
    }


def _unify_block_families(enriched: list[dict], prepared: list[dict],
                          render_fit_options: dict, font_options: dict) -> list[dict]:
    """Same-block font coherence: lines of ONE text block share one family.

    Document consensus (below) deliberately caps fit regression so a genuine display
    face survives the body font — but that per-line independence lets two lines of the
    SAME headline ship different families (benchmark 094: "Caffeine-Free Energy" in
    Lato 800 beside "Boost" in Carlito 700). Within a block, visual coherence outranks
    a small per-line fit delta: refit minority lines to the block's dominant family
    with a looser regression allowance. Same weight guard as doc consensus (a bold
    line never adopts a regular representative), and a line already in exact-font
    territory keeps its match — blocks mixing a true display face with body copy are
    separated by _make_blocks' style clustering before this runs.
    """
    opts = (font_options or {}).get("block_unify") or {}
    if opts.get("enabled", True) is False:
        return []
    try:
        from src import font_fit
    except Exception:
        return []
    by_line_id = {id(item.get("line")): item for item in prepared}
    blocks: dict[str, list[dict]] = {}
    for line in enriched:
        bid = line.get("block_id")
        if bid and (line.get("style") or {}).get("fontFamily"):
            blocks.setdefault(str(bid), []).append(line)
    max_regression = _num(opts.get("max_fit_regression"), 0.18)
    strong_keep = _num(opts.get("strong_keep"), 0.72)
    max_weight_delta = _num(opts.get("max_weight_delta"), 200)
    min_score = _num(render_fit_options.get("min_score"), 0.30)
    changes: list[dict] = []
    for bid, lines in blocks.items():
        if len(lines) < 2:
            continue
        fams = {str((l.get("style") or {}).get("fontFamily")) for l in lines}
        if len(fams) < 2:
            continue
        # Dominant family by ink area x fit among the block's own lines.
        weight: dict[str, float] = {}
        rep: dict[str, dict] = {}
        for l in lines:
            style = l.get("style") or {}
            fam = str(style.get("fontFamily"))
            box = l.get("painted_box") or l.get("box") or {}
            area = max(1.0, _num(box.get("w"), 1.0) * _num(box.get("h"), 1.0))
            fit = _num(((l.get("meta") or {}).get("render_fit") or {}).get("score"), 0.2)
            weight[fam] = weight.get(fam, 0.0) + area * max(fit, 0.05)
            cand = next((c for c in (style.get("fontCandidates") or [])
                         if isinstance(c, dict) and str(c.get("family")) == fam and c.get("path")), None)
            if cand and fam not in rep:
                rep[fam] = cand
        target = max(weight, key=weight.get)
        cand = rep.get(target)
        path = str((cand or {}).get("path") or "")
        if not path or not os.path.exists(path):
            continue
        for l in lines:
            style = l.get("style") or {}
            if str(style.get("fontFamily")) == target:
                continue
            own = _num(((l.get("meta") or {}).get("render_fit") or {}).get("score"), 0.0)
            if own >= strong_keep:
                continue
            rep_weight = (cand or {}).get("weight")
            line_weight = style.get("fontWeight")
            if (isinstance(rep_weight, (int, float)) and isinstance(line_weight, (int, float))
                    and abs(int(rep_weight) - int(line_weight)) >= max_weight_delta):
                continue
            item = by_line_id.get(id(l))
            mask = item.get("font_mask") if item else None
            if mask is None:
                continue
            try:
                fit = font_fit.fit_line(l.get("text", ""), path, mask,
                                        _num(style.get("fontSize"), 16.0), render_fit_options)
            except Exception:
                continue
            new_score = _num((fit or {}).get("score"), 0.0)
            if fit is None or (own - new_score) > max_regression:
                continue
            if own >= min_score and new_score < min_score:
                continue
            previous = str(style.get("fontFamily"))
            new_size = _num(fit.get("fontSize"), _num(style.get("fontSize"), 16.0))
            style["fontFamily"] = target
            style["fontSize"] = round(new_size, 2)
            style["letterSpacing"] = 0.0
            meta = l.setdefault("meta", {})
            meta.setdefault("render_fit", {})["score"] = new_score
            meta["block_font_unified"] = {"from": previous, "to": target,
                                          "own": round(own, 3), "new": round(new_score, 3)}
            changes.append({"block": bid, "line": l.get("id"), "from": previous,
                            "to": target, "own": round(own, 3), "new": round(new_score, 3)})
    return changes


def _unify_repeated_row_labels(enriched: list[dict]) -> list[dict]:
    """Snap repeated same-pattern labels sharing a horizontal row to one style.

    Chart axis / category labels (107: WEEK 1 … WEEK 5) are short, identically
    formatted lines on a common baseline. Independent per-line font matching lets one
    of them (WEEK 3) fall into a different style cluster and render a visibly different
    font or size — the row then looks jittery. Group by text pattern (digits and
    letters normalised) plus baseline row, and force every member of a group of 3+ to
    the group's majority family / median size / majority weight so the row reads as one
    system. Deliberately narrow: only short labels, only rows of 3+ identical patterns.
    """
    from collections import Counter

    def _pattern(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", str(text or "").strip())
        return re.sub(r"\d+", "#", collapsed)

    by_pattern: dict[str, list[tuple]] = {}
    for line in enriched:
        style = line.get("style") or {}
        text = str(line.get("text") or "").strip()
        if not text or len(text) > 24 or not style.get("fontFamily"):
            continue
        pattern = _pattern(text)
        if len(pattern) < 2 or not any(ch.isalnum() for ch in pattern):
            continue
        base = line.get("baseline") or {}
        by = _num(base.get("y0"), _num((line.get("box") or {}).get("y"), 0.0))
        by_pattern.setdefault(pattern, []).append((by, line))

    # Cluster each pattern's lines into horizontal rows by baseline proximity (a shared
    # axis row has near-identical baselines). The tolerance uses the row's own median
    # size, NOT each line's own (mis)measured size, so a jittery size can't split the row.
    groups: list[tuple[str, list[dict]]] = []
    for pattern, items in by_pattern.items():
        items.sort(key=lambda it: it[0])
        rows: list[list[tuple]] = []
        for by, line in items:
            if rows and abs(by - rows[-1][0][0]) <= 28.0:
                rows[-1].append((by, line))
            else:
                rows.append([(by, line)])
        for row in rows:
            groups.append((pattern, [ln for _by, ln in row]))

    changes: list[dict] = []
    for pattern, members in groups:
        if len(members) < 3:
            continue
        fam = Counter(str((m.get("style") or {}).get("fontFamily")) for m in members).most_common(1)[0][0]
        weight = Counter(int(_num((m.get("style") or {}).get("fontWeight"), 400)) for m in members).most_common(1)[0][0]
        sizes = sorted(_num((m.get("style") or {}).get("fontSize"), 16.0) for m in members)
        med_size = sizes[len(sizes) // 2]
        rep = next((m for m in members if str((m.get("style") or {}).get("fontFamily")) == fam), members[0])
        rep_candidates = (rep.get("style") or {}).get("fontCandidates")
        for m in members:
            style = m.get("style") or {}
            changed = []
            if str(style.get("fontFamily")) != fam:
                style["fontFamily"] = fam
                if rep_candidates:
                    style["fontCandidates"] = copy.deepcopy(rep_candidates)
                changed.append("family")
            if abs(_num(style.get("fontSize"), 16.0) - med_size) > 0.5:
                style["fontSize"] = round(med_size, 2)
                changed.append("size")
            if int(_num(style.get("fontWeight"), 400)) != weight:
                style["fontWeight"] = weight
                changed.append("weight")
            style["letterSpacing"] = 0.0
            if changed:
                m.setdefault("meta", {})["row_label_unified"] = {
                    "pattern": pattern, "family": fam, "size": round(med_size, 2),
                    "changed": changed,
                }
                changes.append({"text": m.get("text"), "pattern": pattern, "changed": changed})
    return changes


# Sibling lines whose fitted sizes already agree this closely are one scale that drifted,
# not a hierarchy. Kept tight: a real step (headline over body, 013's headline vs subhead)
# is far larger than this, so it splits into its own cluster and survives untouched.
_PEER_SIZE_TOLERANCE = 0.15
# How far a peer's composition-normalised ink must stand out from its cluster before its
# weight is treated as authored rather than mis-measured. Same threshold, same evidence
# and same measured margin as the per-word gate in _enrich_word_styles.
_PEER_WEIGHT_INK_RATIO = 1.30
# A COLUMN is peers that the block grouper could not see (see _unify_column_text_scale).
# Left edges must agree to within this fraction of the median line height, and the stack
# must be vertically contiguous within this multiple of it — enough to bridge the row gaps
# of a checklist (101: ~2x line height) without ever reaching an unrelated element.
# All three gates are measured against the LOCAL pair of lines being joined, never a
# document-wide median: on a poster the median line is a 74px display glyph, and
# tolerances scaled from it (a 55px left slop, a 222px gap) chain unrelated headline
# fragments into "columns" — 088 grouped 'BLACK' at y=812 with 'BLACK F' 327px below it.
_COLUMN_LEFT_TOLERANCE = 0.35
_COLUMN_GAP_FACTOR = 1.5
# A column is a stack of same-size rows. Comparing the RAW OCR box heights (not the
# fitted sizes the pass is about to rewrite) keeps a display fragment from joining body
# copy it happens to sit above.
_COLUMN_HEIGHT_RATIO = 1.35
# Column peers get a looser size band than block peers because the ANALYTIC size model's
# own residual is that big. Measured on 101's checklist (a column that is one size in the
# source): the glyph model nails ascender-only rows to 0.2% but still under-reads
# 'durability' by 13% (its 'y' descender is shallower than the 'p' the one-constant
# descender model assumes), so two rows of identical authored size legitimately measure
# 1.151 apart. 0.15 cannot see them as peers; 0.20 can.
_COLUMN_SIZE_TOLERANCE = 0.20
# ...and the hierarchy guard moves to the CLUSTER's total span, which is the stronger
# statement: no unified column may span more than 20%, i.e. exactly the measurement noise
# floor above. A real authored step (091's 1.222) exceeds it, so it can never be absorbed
# no matter how its members chain.
_COLUMN_SPREAD_CAP = 1.20


def _snap_peer_cluster(cluster: list[dict], label: dict, changes: list[dict]) -> None:
    """Snap one cluster of same-scale peers to a single size/family/weight.

    Family and weight move only on a STRICT majority — never on a tie. A majority alone
    is NOT enough to touch weight, though: a designer bolding one line of a three-line
    stack is outvoted 2-1 by definition, which is how 135's headline lost its authored
    Bold ('UPFRONT 3-WEEKSE' 400 / '50% KORTING OP' 800 / 'BIJNA' 400 -> all 400). So a
    member also has to FAIL to stand out in the ink before its weight is overruled,
    measured on composition-normalised density against its peers — the same evidence a
    per-word weight change needs, for the same reason (a line's raw density moves with
    its glyph mix: no-descender copy sits in a short, dense box and over-reads).
    Measured, the two cases separate cleanly:
        025's list rows   peerR 0.87 / 1.00 / 1.03  -> nothing stands out, unify to 400
        135's headline    peerR 0.59 / 1.51 / 1.00  -> '50% KORTING OP' is really Bold
    """
    from collections import Counter

    if len(cluster) < 2:
        return
    sizes = sorted(_num((m.get("style") or {}).get("fontSize"), 16.0) for m in cluster)
    med_size = sizes[len(sizes) // 2]
    fams = Counter(str((m.get("style") or {}).get("fontFamily")) for m in cluster)
    weights = Counter(int(_num((m.get("style") or {}).get("fontWeight"), 400)) for m in cluster)
    fam, fam_n = fams.most_common(1)[0]
    weight, weight_n = weights.most_common(1)[0]
    inks = sorted(d for d in (_num((m.get("meta") or {}).get("ink_density_norm"), 0.0)
                              for m in cluster) if d > 0)
    med_ink = inks[len(inks) // 2] if inks else None
    for member in cluster:
        style = member.get("style") or {}
        changed = []
        own_ink = _num((member.get("meta") or {}).get("ink_density_norm"), 0.0)
        ink_ratio = (own_ink / med_ink) if (med_ink and own_ink > 0) else None
        stands_out = ink_ratio is not None and (
            ink_ratio >= _PEER_WEIGHT_INK_RATIO or ink_ratio <= 1.0 / _PEER_WEIGHT_INK_RATIO
        )
        if abs(_num(style.get("fontSize"), 16.0) - med_size) > 0.5:
            style["fontSize"] = round(med_size, 2)
            style["fontSizeCandidates"] = _size_candidates(med_size)
            changed.append("size")
        if fam_n * 2 > len(cluster) and str(style.get("fontFamily")) != fam:
            style["fontFamily"] = fam
            changed.append("family")
        own_weight = int(_num(style.get("fontWeight"), 400))
        if (weight_n * 2 > len(cluster) and own_weight != weight and not stands_out):
            style["fontWeight"] = weight
            style["fontWeightCandidates"] = _weight_candidates(weight)
            changed.append("weight")
        if changed:
            member.setdefault("meta", {})["peer_scale_unified"] = {
                **label, "size": round(med_size, 2), "changed": changed,
            }
            changes.append({"text": member.get("text"), "changed": changed, **label})


def _unify_peer_text_scale(enriched: list[dict]) -> list[dict]:
    """Snap sibling lines of one block that measure the same scale to a single size.

    Sizes are fitted per line, so lines a reader sees as one element drift apart: 016's
    '21+ vitamins' / '& minerals' callout emits 29.4 and 26.8, and 101's checklist rows
    scatter. Group by block, cluster the block's lines by fitted size, and snap each
    cluster of 2+ to its median.

    The clustering IS the hierarchy guard, and it is why this cannot flatten a design:
    peers are only ever lines already within ``_PEER_SIZE_TOLERANCE`` of each other, so a
    genuine size step (013's headline over its subhead, 025's card headings) exceeds the
    band, lands in its own cluster, and is never merged with the copy beside it. Family
    and weight move only on a strict majority — never on a tie — so a deliberate bold lead
    line inside a paragraph keeps its contrast.
    """
    from collections import Counter

    by_block: dict[str, list[dict]] = {}
    for line in enriched:
        style = line.get("style") or {}
        text = str(line.get("text") or "").strip()
        # Sub-glyph OCR speckle ('-', '- -', '.') measures nonsense sizes; never let it
        # vote on, or be dragged by, a real peer group.
        if not style.get("fontFamily") or len(text) < 2 or not any(c.isalnum() for c in text):
            continue
        block_id = str(line.get("block_id") or "")
        if block_id:
            by_block.setdefault(block_id, []).append(line)

    changes: list[dict] = []
    for block_id, members in by_block.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: _num((m.get("style") or {}).get("fontSize"), 16.0))
        clusters: list[list[dict]] = []
        for member in members:
            size = _num((member.get("style") or {}).get("fontSize"), 16.0)
            if clusters:
                base = _num((clusters[-1][0].get("style") or {}).get("fontSize"), 16.0)
                if size <= base * (1.0 + _PEER_SIZE_TOLERANCE):
                    clusters[-1].append(member)
                    continue
            clusters.append([member])
        for cluster in clusters:
            _snap_peer_cluster(cluster, {"block": block_id}, changes)
    return changes


def _column_groups(usable: list[dict]) -> list[list[dict]]:
    """Stack lines into left-aligned columns of same-size rows (see _unify_column_text_scale)."""
    ordered = sorted(usable, key=lambda l: ((l.get("box") or {}).get("y", 0.0)))
    columns: list[list[dict]] = []
    for line in ordered:
        box = line.get("box") or {}
        colour = str((line.get("style") or {}).get("color") or "#000000")
        placed = False
        for column in columns:
            last = column[-1]
            lbox = last.get("box") or {}
            tall = max(_num(lbox.get("h"), 16.0), _num(box.get("h"), 16.0))
            short = max(1e-6, min(_num(lbox.get("h"), 16.0), _num(box.get("h"), 16.0)))
            if tall / short > _COLUMN_HEIGHT_RATIO:
                continue
            if abs(_num(lbox.get("x"), 0.0) - _num(box.get("x"), 0.0)) > max(
                    2.0, tall * _COLUMN_LEFT_TOLERANCE):
                continue
            if _colour_distance(colour, str((last.get("style") or {}).get("color") or "#000000")) > 60.0:
                continue
            gap = _num(box.get("y"), 0.0) - (_num(lbox.get("y"), 0.0) + _num(lbox.get("h"), 0.0))
            # An overlapping/same-row neighbour is a different element, not the next row.
            if gap < -0.25 * short or gap > tall * _COLUMN_GAP_FACTOR:
                continue
            # A column never has ANOTHER line living between two of its rows — the same
            # veto _make_blocks uses. Without it a chain hops over the row that really
            # follows: 013's pouch label (whose OCR boxes are ~2x their glyphs and
            # overlap each other) interleaved into 'COMPREHENSIVE NUTRITION'+'Greens'
            # and 'Superfoods'+'Gummies', which then flattened a small letter-spaced
            # eyebrow into the display text below it.
            if _interleaved(last, line, ordered):
                continue
            column.append(line)
            placed = True
            break
        if not placed:
            columns.append([line])
    return columns


def _unify_column_text_scale(enriched: list[dict]) -> list[dict]:
    """Snap lines that form one left-aligned COLUMN to a single scale.

    _unify_peer_text_scale can only see peers that share a ``block_id``, and the block
    grouper vetoes joins on ROLE (``_can_join`` -> ``_compatible_roles``). Roles are
    semantic labels derived from regex/size/position, so a uniform column shatters on
    contents alone: 101's checklist emits '50% thicker for better' as an *offer* (the
    offer regex fires on "50%") and 'repairs & sealant use' as a *footer* (it sits below
    y=0.86), stranding each in a singleton block that the block pass — which needs 2+
    members — can never reach. The rows then render at 20.9/23.0/24.1 with weights
    jittering 350/400, which is the visible "text scatter".

    A column is a VISUAL fact, not a semantic one: lines sharing a left edge and a colour
    in a contiguous vertical run are peers whatever role the labeller gave them. Grouping
    on that recovers exactly the peers the role veto hid, and the size clustering is still
    the hierarchy guard — a genuine step (091's 1.222) exceeds the band, lands in its own
    cluster and is never merged with the copy beside it.
    """
    usable = []
    for line in enriched:
        style = line.get("style") or {}
        text = str(line.get("text") or "").strip()
        if not style.get("fontFamily") or len(text) < 2 or not any(c.isalnum() for c in text):
            continue
        box = line.get("box") or {}
        if not box.get("h"):
            continue
        usable.append(line)
    if len(usable) < 2:
        return []

    columns = _column_groups(usable)
    changes: list[dict] = []
    for column in columns:
        if len(column) < 2:
            continue
        members = sorted(column, key=lambda m: _num((m.get("style") or {}).get("fontSize"), 16.0))
        clusters: list[list[dict]] = []
        for member in members:
            size = _num((member.get("style") or {}).get("fontSize"), 16.0)
            if clusters:
                current = [_num((m.get("style") or {}).get("fontSize"), 16.0) for m in clusters[-1]]
                # Chain against the cluster's running MEDIAN, not its smallest member: a
                # min-anchored chain splits a uniform column on a hair (101 emitted 24.1
                # against a 24.03 cutoff drawn from a 20.9 outlier). The spread cap keeps
                # a median-anchored chain from walking a gradient into one bucket.
                med = sorted(current)[len(current) // 2]
                spread = max(current + [size]) / max(1e-6, min(current + [size]))
                if size <= med * (1.0 + _COLUMN_SIZE_TOLERANCE) and spread <= _COLUMN_SPREAD_CAP:
                    clusters[-1].append(member)
                    continue
            clusters.append([member])
        for cluster in clusters:
            _snap_peer_cluster(cluster, {"column": round(_num((cluster[0].get("box") or {}).get("x"), 0.0), 1)},
                               changes)
    return changes


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
    # Absolute cap on how much fit a line may LOSE by folding into the consensus
    # family, independent of the (deliberately loose) class-consistent tolerance.
    # The loosened tolerance is meant to gather a *scattered* sans zoo whose members
    # all fit the winner about equally (002's body labels regress <=0.045 or improve);
    # it must never fold a line whose own face fits markedly better — the squared
    # display headline "KRACHTSPORT BUNDEL" fits Archivo at 0.478 but the body's Lato
    # at only 0.315 (a 0.163 regression), which is exactly the wrong-font headline the
    # audit flagged. Cap the regression so genuine display faces survive while every
    # near-tie body line still unifies.
    max_fit_regression = _num(opts.get("max_fit_regression"), 0.10)
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
        # "Forbidden" must mean a serif family LEAKED onto sans ink — not that the ink
        # itself is serif. The matched family's class alone cannot tell those apart, so
        # ask the per-line class gate, which classes the SOURCE ink by fitting canonical
        # reference faces to it. Without this the rule converts every genuine serif
        # headline in a sans-majority ad into a sans (091's serif "Foggy and Steady" ->
        # Noto Sans): the intended escape hatch (``strong_keep``, 0.72) is unreachable
        # for such a line because render-back IoU falls off with string length — a
        # 16-glyph headline tops out near 0.16 no matter how right the face is.
        gate = ((line.get("meta") or {}).get("font_match") or {}).get("class_gate") or {}
        source_class = str(gate.get("class") or "")
        source_gate_conf = _num(gate.get("confidence"), 0.0)
        protect_min_conf = _num(opts.get("source_class_protect_min_confidence"), 0.20)
        source_is_non_sans = (source_class in ("serif", "script")
                              and source_gate_conf >= protect_min_conf)
        forbidden = (class_consistent and line_class in ("serif", "script")
                     and not source_is_non_sans)
        if source_is_non_sans and consensus_class == "sans" and line_class in ("serif", "script"):
            # The ink is positively non-sans and the line already matched that class.
            # A sans consensus must not overwrite it on a fit-score difference that is
            # noise at these magnitudes (091 L25: own serif 0.161 vs consensus sans
            # 0.131 — the consensus is *worse* and still won).
            meta = line.setdefault("meta", {})
            meta["font_consensus_skipped"] = {
                "reason": "source-class-protected", "source_class": source_class,
                "confidence": round(source_gate_conf, 3), "kept": line_family,
            }
            evidence.setdefault("class_protected", []).append({
                "id": line.get("id"), "kept": line_family, "source_class": source_class,
                "confidence": round(source_gate_conf, 3),
            })
            continue

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
            and (own - new_score) <= max_fit_regression
            and not (own >= min_score and new_score < min_score)
        )
        previous = line_family
        if adopt:
            new_size = _num(fit.get("fontSize"), _num(style.get("fontSize"), 16.0))
            style["fontFamily"] = family
            style["fontSize"] = round(new_size, 2)
            style["letterSpacing"] = 0.0
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


# Connective tissue, EN + NL (the benchmark's two languages). A designer emphasises a
# phrase, a number or a brand; a lone bold "we"/"to"/"en" mid-sentence is measurement
# noise. Deliberately excludes anything that can carry authored emphasis on its own
# (no nouns, no verbs, no numerals) — see _enrich_word_styles' jitter clamp.
_FUNCTION_WORDS = {
    # EN
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "if", "in", "is",
    "it", "of", "on", "or", "so", "than", "that", "the", "then", "to", "up", "us",
    "we", "with", "you", "your", "our", "my", "me", "he", "she", "they", "this",
    # NL
    "aan", "al", "als", "bij", "dat", "de", "die", "dit", "een", "en", "er", "het",
    "hun", "ik", "in", "is", "je", "met", "na", "naar", "niet", "nu", "of", "om",
    "ook", "op", "te", "tot", "uit", "uw", "van", "voor", "waar", "wij", "zijn",
}


def _short_function_word(text: str) -> bool:
    """True for a token that cannot plausibly carry authored bold on its own.

    Either a 1-2 character token (too few glyphs for a density estimate to mean
    anything) or a known function word.
    """
    token = str(text or "").strip().strip(".,:;!?'\"()[]").lower()
    if not token:
        return False
    return len(token) <= 2 or token in _FUNCTION_WORDS


# RELEASING a word from its line's italic needs far stronger evidence than asserting
# italic on an upright line, because a word re-measures the SAME ink as its line with a
# slightly different tight mask. Across the bench, single-token lines (word ink ==
# line ink, so any delta is pure measurement noise) disagree by a median of 1.58° and
# up to 3.50°: 013 'We NEVER' reads -6.75 as a line and -5.68 as a word, 107 'Orange'
# -6.34 vs -4.76. With one symmetric 6.0° gate both words fall to the upright side of
# their own italic line and get relabelled upright while still carrying the line's
# ITALIC font file — the preview resolves the FILE and looks correct while Figma
# resolves the STYLE NAME and ships the headline upright. Only a word measuring at most
# this angle is decisively upright; unmeasurable (None) is NOT evidence of upright —
# _measure_shear_angle returns None for thin/short masks AND for genuinely upright ink
# alike (091 'MGNAT' line -6.34 → word None, 107 'DAILY HYDRATION' -8.43 → None).
_ITALIC_RELEASE_DEG = 3.0


def _word_italic_state(shear, base_italic: bool, config: dict) -> bool:
    """Is this word italic? Hysteresis around its LINE's slant (see above)."""
    assert_gate = _num(config.get("italic_shear_deg"), 6.0)
    if shear is None:
        return base_italic          # unmeasurable: keep the line's slant, don't invent one
    magnitude = abs(shear)
    if base_italic:
        release = min(_ITALIC_RELEASE_DEG, assert_gate)
        return magnitude > release
    return magnitude >= assert_gate


def path_is_italic(path) -> Optional[bool]:
    """Does this FILE actually draw italic? Ask the font, not its filename.

    ``_font_metadata`` reads the real name records (with a filename fallback), so
    this is right where naming conventions lie: ``calibri.ttf`` and ``segoeui.ttf``
    END IN "i.ttf" without being italic, while ``Candarali.ttf`` IS italic without
    containing "italic". Returns None when there is no path to judge.
    """
    if not path:
        return None
    try:
        return "italic" in str(_font_metadata(str(path)).get("style") or "").lower()
    except Exception:
        return None


def _match_candidate_slant(style: dict, italic: bool, config: dict) -> None:
    """Keep ``fontCandidates[0]`` on the slant the word now DECLARES.

    Word overrides deliberately inherit the LINE's candidates (same face), so a word
    that flips slant would otherwise declare one slant while carrying the other's
    file. The mismatch is invisible in the preview — which resolves the FILE and
    draws the right slant — but Figma resolves the STYLE NAME, so the DELIVERABLE
    ships the wrong slant. Resolve the declared family at the new slant; if it is not
    installed, drop a contradicting path rather than lie about it (the same fallback
    ``build_design_json._promote_weight_candidate`` uses for weight mismatches).
    """
    cands = [dict(c) for c in (style.get("fontCandidates") or []) if isinstance(c, dict)]
    if not cands:
        return
    top = cands[0]
    weight = int(round(_num(style.get("fontWeight"), 400.0)))
    resolved = _resolve_family_path(
        style.get("fontFamily") or top.get("family"), weight, italic, _font_options(config))
    if resolved:
        top["path"] = resolved
        top.pop("local_family", None)
        top.pop("fit", None)        # the outvoted face's fit is not this face's
        top["family_resolved"] = True
    else:
        path_italic = path_is_italic(top.get("path"))
        if path_italic is not None and path_italic != italic:
            top.pop("path", None)
    top["style"] = _style_name(weight, italic=italic)
    top["weight"] = weight
    style["fontCandidates"] = [top] + cands[1:]


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
    cap_ratio = max(0.45, min(0.90, _num(config.get("cap_height_ratio"), 0.72)))
    # EVERY mid-line weight change must clear this ratio against the line's OWN words,
    # measured on composition-normalised density (see below). The absolute weight bucket
    # cannot carry a change by itself: a word's mask is gapless while its line's mask
    # includes the inter-word spaces, so words systematically read denser than the line
    # they sit in and the absolute buckets fire on that artefact alone (066's 'buildable'
    # is bucketed Bold-700 against a Regular-400 line while being exactly as dense as its
    # own line-mates).
    #
    # Swept over every bench fixture, the two populations are cleanly bimodal and NOTHING
    # lands between them:
    #     authored emphasis  >= 1.346  (104 '8K' 2.47, 009 'GC' 2.02 / '121K' 1.90,
    #                                   091 '120MG' 1.73 / 'CRANGESUNRISE' 1.56,
    #                                   067 'Sale' 1.56 / 'OFF' 1.50 / '40%' 1.346)
    #     composition jitter <= 1.174  (002 'pouroon' 1.17, 066 'buildable' 1.16 /
    #                                   'shades' 1.11, 091 'et' 1.10, 066 'on' 0.87)
    # 1.30 sits inside that gap: 10.7% clear of the worst jitter, 3.5% under the weakest
    # authored emphasis. Applied symmetrically (a genuinely LIGHTER word must be as far
    # below its peers as a bolder one is above) so real light runs still survive.
    relative_density_ratio = _num(config.get("word_style_relative_density_ratio"), 1.30)
    # A weight flip on a SHORT FUNCTION WORD wedged between two same-weight neighbours
    # is the residual jitter the gates above cannot see: 'we'/'to' carry so few glyphs
    # that their density is dominated by which letters they happen to contain, and a
    # single stray bold word mid-sentence is never authored emphasis (a designer bolds
    # a phrase, a number or a brand — not the connective tissue). Such a flip must be
    # corroborated by BOTH independent ink signals or it is clamped back to the line.
    jitter_density_ratio = _num(config.get("word_style_jitter_density_ratio"), 1.5)
    jitter_stroke_ratio = _num(config.get("word_style_jitter_stroke_ratio"), 1.25)

    def _styleable(candidate: Any) -> bool:
        # A lone punctuation mark or a 1-char sliver (",", "->", a stray digit) carries
        # no reliable per-word style and must never become its own Figma run.
        txt = str((candidate or {}).get("text") or "").strip()
        return len(txt) >= 2 and any(ch.isalnum() for ch in txt)

    valid_word_count = sum(
        1 for w in (line.get("words") or []) if isinstance(w, dict) and _styleable(w)
    )

    measured_words = []
    for raw_word in line.get("words") or []:
        if not isinstance(raw_word, dict) or not _styleable(raw_word):
            continue
        word = raw_word
        word["box"] = _clean_box(word.get("box"))
        # Word boxes are even tighter than line boxes. Sample a narrow exterior collar
        # for plate polarity/colour, while preserving the original OCR geometry on the
        # exported word. This prevents black all-caps words touching their crop border
        # from being misclassified as white runs (002: BUNDEL).
        probe = copy.deepcopy(word)
        probe["box"] = _collar_box(word["box"], image)
        painted, _baseline, colour, ink_conf, mask, paint = _painted_geometry(image, probe)
        if ink_conf < min_conf:
            continue
        # COMPOSITION-NORMALISED density. Raw ink density is ink/(w*h) and the ink box's
        # HEIGHT is set by the word's tallest and lowest glyphs — the very thing
        # _expected_ink_ratio already models for size. So density is inflated by ~1/ratio:
        # 066's 'remove' (x-height only, short box) measures 0.564 while its own line-mate
        # 'Easy' (cap + descender, tall box) measures 0.332 at the SAME authored weight —
        # a 1.70x split that lands them in different absolute weight buckets. Dividing the
        # composition factor out makes words with different glyph mixes directly
        # comparable: that same line collapses to a 1.06x spread, while authored emphasis
        # survives untouched (009 '121K' 1.90x, 067 'Sale' 1.56x / 'OFF' 1.50x).
        density = None
        try:
            if mask is not None and mask.size:
                ratio = _expected_ink_ratio(word.get("text"), cap_ratio)
                density = float(mask.mean()) * ratio / cap_ratio
        except Exception:
            density = None
        # Stroke width normalised by the word's own painted height: an ink signal that,
        # unlike density, does not move with glyph composition or letter-spacing. Gives
        # the jitter clamp a second, INDEPENDENT source of evidence for a real bold.
        stroke_ratio = None
        try:
            from . import handwriting

            width = handwriting.stroke_width_mean(mask)
            if width and painted.get("h", 0) > 0:
                stroke_ratio = float(width) / float(painted["h"])
        except Exception:
            stroke_ratio = None
        measured_words.append(
            {"word": word, "painted": painted, "colour": colour, "ink_conf": ink_conf,
             "mask": mask, "paint": paint, "density": density, "stroke_ratio": stroke_ratio}
        )

    densities = sorted(m["density"] for m in measured_words if m["density"] is not None)
    line_density = densities[len(densities) // 2] if densities else None
    strokes = sorted(m["stroke_ratio"] for m in measured_words if m["stroke_ratio"] is not None)
    line_stroke = strokes[len(strokes) // 2] if strokes else None

    # Measure every word's weight up front: the jitter clamp needs a word's NEIGHBOURS'
    # weights, which are not known while the words are still being styled one by one.
    for measured in measured_words:
        geo = _pre_font_signals(measured["word"], measured["painted"], measured["mask"], config)
        measured["geo"] = geo
        measured["weight"] = int(round(_num(geo.get("weight"), base_weight)))

    for position, measured in enumerate(measured_words):
        word = measured["word"]
        painted, colour = measured["painted"], measured["colour"]
        ink_conf, mask, paint = measured["ink_conf"], measured["mask"], measured["paint"]
        geo = measured["geo"]
        measured_size = max(1.0, _num(geo.get("font_size"), base_size))
        measured_weight = measured["weight"]
        ratio = max(measured_size, base_size) / max(1.0, min(measured_size, base_size))
        colour_changed = _colour_distance(colour, base_colour) >= colour_delta
        # A mid-line weight change needs the word to actually stand out from its OWN
        # line-mates, not merely to land in a different absolute bucket than the line's
        # (differently-supported) mask. A line with one styleable word has no peers to
        # stand out from, so the absolute bucket is all there is and remains sufficient.
        peer_density_ratio = (
            measured["density"] / line_density
            if line_density and measured["density"] else None
        )
        heavier = measured_weight > base_weight
        stands_out = (
            valid_word_count <= 1
            or (peer_density_ratio is not None
                and (peer_density_ratio >= relative_density_ratio if heavier
                     else peer_density_ratio <= 1.0 / relative_density_ratio))
        )
        weight_changed = (abs(measured_weight - base_weight) >= weight_delta
                          and ink_conf >= weight_min_conf
                          and stands_out)
        if (not weight_changed and peer_density_ratio is not None
                and abs(measured_weight - base_weight) >= weight_delta
                and ink_conf >= weight_min_conf):
            word.setdefault("style_debug", {})["weight_peer_clamped"] = {
                "measured": measured_weight, "clamped_to": base_weight,
                "peer_density_ratio": round(peer_density_ratio, 3),
                "required": relative_density_ratio,
            }
        # Relative-density bold: absolute density buckets cap thin display faces
        # well below Bold, so an authored bold word never clears the absolute
        # ±weight_delta gate (025: bold-italic "Hears" measured 600 vs base 400).
        # Ink density measured against the line's OWN words is face-independent —
        # a word ≥35% denser than its line's median word, whose absolute bucket
        # already reads ≥200 heavier, is authored emphasis, not jitter. This is the
        # path that keeps 067's 'Sale'/'40% OFF' bold once the absolute buckets stop
        # being trusted on their own.
        if (not weight_changed and peer_density_ratio is not None
                and len(densities) >= 3
                and measured_weight - base_weight >= 200
                and peer_density_ratio >= relative_density_ratio
                and ink_conf >= weight_min_conf):
            measured_weight = max(measured_weight, base_weight + 300)
            weight_changed = True
        # JITTER CLAMP (009: 'we'/'to' rendering Bold between Regular neighbours). A
        # short function word wedged between two neighbours that agree with each other
        # on weight is the one case where a measured flip is more likely noise than
        # design: too few glyphs for density to be stable, and no designer bolds a lone
        # connective mid-sentence. Demand BOTH ink signals — density ≥1.5x the line
        # median AND a corroborating stroke-width ratio — or clamp back to the line.
        # Genuine emphasis is untouched: it is either not a function word (009 '121K',
        # 067 'Sale'/'40% OFF'), or not sandwiched (a line-initial or standalone token).
        if weight_changed and _short_function_word(word.get("text")):
            neighbours = [measured_words[i]["weight"] for i in (position - 1, position + 1)
                          if 0 <= i < len(measured_words)]
            sandwiched = len(neighbours) == 2 and neighbours[0] == neighbours[1]
            if sandwiched:
                dense_enough = bool(
                    line_density and measured["density"]
                    and measured["density"] >= line_density * jitter_density_ratio
                )
                stroke_enough = bool(
                    line_stroke and measured["stroke_ratio"]
                    and measured["stroke_ratio"] >= line_stroke * jitter_stroke_ratio
                )
                if not (dense_enough and stroke_enough):
                    measured_weight = base_weight
                    weight_changed = False
                    word.setdefault("style_debug", {})["weight_jitter_clamped"] = {
                        "measured": measured["weight"], "clamped_to": base_weight,
                        "neighbour_weight": neighbours[0],
                        "density_ratio": (round(measured["density"] / line_density, 3)
                                          if line_density and measured["density"] else None),
                        "stroke_ratio": (round(measured["stroke_ratio"] / line_stroke, 3)
                                         if line_stroke and measured["stroke_ratio"] else None),
                    }
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
        base_italic = "italic" in str(base.get("fontStyle") or "").lower()
        word_italic = _word_italic_state(shear, base_italic, config)
        italic_changed = word_italic != base_italic
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
        if weight_changed or italic_changed:
            # One label for both signals: the slant is `word_italic` (== base_italic
            # whenever italic_changed is False), so a weight-only override still
            # reproduces the line's slant exactly as before.
            style["fontStyle"] = _style_name(
                int(style.get("fontWeight", base_weight)), italic=word_italic)
        if italic_changed:
            # The style is a deepcopy of the LINE's, so a flipped word would otherwise
            # keep the line's shear and contradict its own new label.
            style["italicShearDeg"] = shear if word_italic else None
            _match_candidate_slant(style, word_italic, config)
        word["style"] = style
        word["style_evidence"] = {
            "source": "word-pixels", "confidence": round(float(ink_conf), 4),
            "changed": [name for name, changed in (
                ("color", colour_changed), ("size", size_changed),
                ("weight", weight_changed), ("italic", italic_changed),
            ) if changed],
        }


# How much of a line box may sit ahead of its first glyph before the box is judged to
# have swallowed a neighbouring object. Measured on the bench: rows with an icon inside
# the box run 0.104-0.183, every clean row 0.000-0.010.
_NON_GLYPH_HEAD_FRACTION = 0.06


def _repair_non_glyph_line_paint(line: dict) -> Optional[dict]:
    """Drop a line's paint when the line's OWN GLYPHS unanimously contradict it.

    An OCR line box can swallow an adjacent ICON. 066's checklist rows are the case: the
    box for 'Smudges on upper lid' starts at x=825 while its first glyph starts at x=880,
    so 55px of red ✗ sit inside the box (OCR even reads the mark as a letter: 'X Up to 3
    shades'). _painted_geometry then measures the line's paint across icon+text and
    returns the ICON's red as the colour, a red->black LINEAR gradient as the fill, and a
    3px black OUTSIDE STROKE. render_preview hands that stroke to PIL's draw.text, which
    outlines every glyph with it — and a black outline around red text is exactly the
    "ghost double"/smeared bold seen on those two rows. Their clean neighbours ('Not
    disclosed', 'Tubing technology') have no icon inside the box and emit flat black with
    no stroke.

    The trigger is the FOREIGN REGION itself, not the disagreement: unanimous words are
    not enough on their own, because words share a failure mode and can be unanimously
    WRONG. 135's 'vezels suikers' is the counter-example — its line reads #1f1f1f flat
    with no stroke (correct, the label copy is dark) while both of its tight word boxes
    flip polarity and read #dedede, so trusting the words there would paint light grey
    text onto a light label. Requiring a word-free strip at the head of the box separates
    them by 10x: contaminated 0.104-0.183 (066's two rows, 067's bottle line) vs clean
    0.000-0.010 (066's own unaffected rows, 088, and 135).
    """
    style = line.get("style")
    words = [w for w in (line.get("words") or [])
             if isinstance(w, dict) and len(str(w.get("text") or "").strip()) >= 2
             and any(ch.isalnum() for ch in str(w.get("text") or ""))]
    if not style or len(words) < 2:
        return None
    box = _clean_box(line.get("box"))
    boxes = [_clean_box(w.get("box")) for w in words]
    if not box.get("w") or not all(b.get("w") for b in boxes):
        return None
    head = (min(b["x"] for b in boxes) - box["x"]) / max(1.0, box["w"])
    if head < _NON_GLYPH_HEAD_FRACTION:
        return None
    recoloured = [w for w in words
                  if "color" in ((w.get("style_evidence") or {}).get("changed") or [])]
    # Unanimity is the corroborating signal: one recoloured word is authored emphasis
    # ('ONLY $19'); every word disagreeing means the line's base is the thing that is off.
    if len(recoloured) != len(words):
        return None
    colours = [str((w.get("style") or {}).get("color") or "") for w in recoloured]
    if not all(colours):
        return None
    if any(_colour_distance(colours[0], c) > 40.0 for c in colours[1:]):
        return None  # the words do not agree either; no consensus to adopt
    consensus = colours[0]
    had_stroke = bool(style.get("stroke"))
    had_gradient = str((style.get("fill") or {}).get("kind") or "flat") != "flat"
    if not (had_stroke or had_gradient):
        return None
    # Drop ONLY the decoration the icon invented. The base COLOUR is deliberately left
    # alone even though the glyphs disagree with it: every word already carries its own
    # colour run, so the text still paints black, and the base is what keeps _can_join's
    # colour veto separating these icon-bearing rows from their neighbours. Recolouring
    # the base lets them join, and their boxes are still icon-inflated (825 against 880),
    # so the merged block's left edges go ragged, _infer_alignment reads CENTER, and every
    # row slides left under its own icon — 066's text recall fell 0.95 -> 0.85. The stroke
    # and the red->black gradient are what actually draw the ghost double, and neither
    # feeds grouping, so removing just those fixes the render and changes nothing else.
    style["fill"] = {"kind": "flat", "color": str(style.get("color") or consensus)}
    style["stroke"] = None
    # NOTE: the BOX is inflated by the icon too (825 against a first glyph at 880), and
    # trimming it here is NOT safe today. Measured on 066: the text stage's own blocks are
    # already correct either way (B7 = x872 w309 LEFT), but build_design_json emits that
    # block as a node at x=824 w=369.9 — expanded to the icon's own x once the restored
    # ✗/✓ icons exist. Against that polluted node box a ragged-left block infers CENTER,
    # which happens to pull the glyphs back to x=871 (near the true 881); trimming makes
    # the block infer LEFT, which honours the polluted box and renders at x=858, straight
    # under the icon (ssim 0.780 -> 0.731, recall 0.85 -> 0.80). The node-box expansion is
    # the real defect and it is downstream of this file; until it is fixed, leaving the box
    # alone is strictly better. See work/probe_066_doubles.py.
    # The per-word colour runs STAY: they are what paints the glyphs their real colour
    # over a base the icon poisoned.
    repair = {"text": line.get("text"), "glyph_color": consensus,
              "dropped_stroke": had_stroke, "dropped_gradient": had_gradient,
              "non_glyph_head_px": round(min(b["x"] for b in boxes) - box["x"], 1)}
    line.setdefault("meta", {})["non_glyph_paint_repaired"] = repair
    return repair


# ---------------------------------------------------------------------------
# Roles, paragraph grouping, alignment, hierarchy and repeated styles


_LEGAL_DISCLAIMER = re.compile(
    r"(?:fda|these statements|not intended to(?:\s+diagnose)?|consult your|"
    r"results may vary|supplement facts|proprietary blend|"
    r"\*\s*these|disclaimer)",
    re.I,
)


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
        elif y_ratio >= 0.82 and (
            _LEGAL_DISCLAIMER.search(text)
            or (size <= median_size * 0.85 and len(words) >= 6)
        ):
            role = "disclaimer"
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
            "disclaimer": 4,
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


def _infer_alignment(lines: list[dict], canvas_w: float,
                     siblings: Optional[list[dict]] = None) -> str:
    if len(lines) == 1:
        box = lines[0]["box"]
        center = _box_center(box)[0]
        left = float(box.get("x", 0) or 0)
        right = left + float(box.get("w", 0) or 0)
        # A lone line's own box cannot tell a floating callout from a row of a
        # left-aligned column: 014's floater (x=120 of 1080) and 101's checklist rows
        # (x=110 of 1000) are the SAME geometry. The distinguishing fact is elsewhere on
        # the canvas — a column row has other lines flush to its left edge, a floater does
        # not. Without this, 101's role-shattered singleton rows each fell into the
        # "floater" branch below and were emitted RIGHT-aligned, so every row's left edge
        # drifted with its own rendered width: the visible ragged indentation.
        if siblings:
            height = max(1.0, float(box.get("h", 0) or 0))
            flush = sum(
                1 for other in siblings
                if other is not lines[0]
                and abs(float((other.get("box") or {}).get("x", 0) or 0) - left) <= height * 0.75
            )
            if flush >= 2:
                return "LEFT"
        # True centered labels have *symmetric* side margins. Wide left-anchored
        # body lines (009 "Daarbovenop…") also have a geometric center near mid
        # canvas — requiring margin symmetry avoids flipping them to CENTER.
        left_margin = left
        right_margin = canvas_w - right
        if (
            box["w"] < canvas_w * 0.88
            and abs(center - canvas_w / 2.0) <= canvas_w * 0.055
            and abs(left_margin - right_margin) <= canvas_w * 0.06
            and min(left_margin, right_margin) >= canvas_w * 0.08
        ):
            return "CENTER"
        # Edge-flush columns keep outward alignment (007 left rail, edge callouts).
        if left <= canvas_w * 0.08:
            return "LEFT"
        if abs(right - canvas_w) <= canvas_w * 0.08:
            return "RIGHT"
        # Indented left-column UI (009 username / @handle under an avatar, ~0.14–0.35
        # of canvas width). These must stay LEFT — the older "center < mid → RIGHT"
        # floating-callout heuristic flipped them and made tracking look broken.
        if canvas_w * 0.14 <= left <= canvas_w * 0.38:
            return "LEFT"
        # Floating side callouts (014): align toward the product / canvas center.
        # True floaters sit just off the margin (left > 8%) with their center still
        # in the left half — not in the avatar-indented band above.
        if center < canvas_w * 0.45:
            return "RIGHT"
        if center > canvas_w * 0.55:
            return "LEFT"
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


def _interleaved(previous: dict, current: dict, others: list[dict]) -> bool:
    """True when a THIRD line sits vertically between ``previous`` and ``current``.

    _can_join tolerates a gap of ~1.25x line height, which lets a block chain skip
    straight over a foreign row (an OCR fontSize mis-measure ejects the middle line,
    e.g. 104 "Slower pace" fs 56 between fs 34 neighbours; 107 "Research says…" fs 43
    between fs 31 body lines). The block then emits a uniform lineHeight across the
    hole and every subsequent line renders one pitch off, colliding with the ejected
    line's own block. A paragraph never has ANOTHER text line living between two of
    its members, so any horizontally-overlapping interloper vetoes the join.
    """
    a, b = previous["box"], current["box"]
    gap_top = a["y"] + a["h"]
    gap_bottom = b["y"]
    if gap_bottom - gap_top < 4.0:
        return False  # adjacent rows: nothing can hide in the gap
    for other in others:
        if other is previous or other is current:
            continue
        obox = other["box"]
        ocy = obox["y"] + obox["h"] / 2.0
        if not (gap_top - 0.15 * obox["h"] < ocy < gap_bottom + 0.15 * obox["h"]):
            continue
        if (_horizontal_overlap(obox, a) >= 0.25
                or _horizontal_overlap(obox, b) >= 0.25):
            return True
    return False


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
            if _interleaved(previous, line, ordered):
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
        alignment = _infer_alignment(group, canvas_w, siblings=ordered)
        baselines = [line["baseline"]["y0"] for line in group]
        deltas = [baselines[i + 1] - baselines[i] for i in range(len(baselines) - 1)]
        median_size = _median((line["style"]["fontSize"] for line in group), 16.0)
        # Floor at 1.12× fontSize — never allow lh < fs. Dense display OCR baseline
        # gaps under-measure (ad 013: lh 195 < fs 230 clipped glyph tops on "We NEVER").
        line_height = max(median_size * 1.12, _median(deltas, median_size * 1.2))
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
        colored_price_rules = _native_colored_price_rules(image, line)
        if colored_price_rules:
            line.setdefault("meta", {})["native_decoration_shapes"] = colored_price_rules
        font_mask = mask
        if decoration_evidence and mask is not None:
            try:
                font_mask = mask.copy()
                row0, row1 = decoration_evidence["mask_rows"]
                font_mask[row0:row1 + 1, :] = False
            except Exception:
                font_mask = mask
        geo = _pre_font_signals(line, painted, font_mask, config)
        # Stash the line's COMPOSITION-NORMALISED ink density (see _enrich_word_styles):
        # the only weight signal that is comparable between two lines whose glyph mixes
        # differ. Peer unification needs it to tell a mis-measured weight from an
        # authored one — 025's 'Filters sound, not life' (no descender, so a short dense
        # box) reads heavier than its own list-mates at the same authored weight, while
        # 135's '50% KORTING OP' really is Bold among Regulars.
        try:
            if font_mask is not None and font_mask.size:
                cap = max(0.45, min(0.90, _num(config.get("cap_height_ratio"), 0.72)))
                line.setdefault("meta", {})["ink_density_norm"] = (
                    float(font_mask.mean()) * _expected_ink_ratio(line.get("text"), cap) / cap
                )
        except Exception:
            pass
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
        # Spend the bounded match budget on the text that actually SHOWS. Iterating in
        # document order spends it on whatever OCR happened to read first — on 091 that
        # is product-label microcopy ("FRODUCTIVITY", 1.2k px²), which exhausted all 16
        # matches and left the ad's biggest headline ("Foggy and Steady", 76k px² — a
        # serif) with no render match at all, falling back to a generic Inter that
        # renders visibly wrong. Rank clusters by their most prominent member's painted
        # ink area so the headline is matched first and the microcopy takes the leftovers.
        def _cluster_prominence(idxs: list[int]) -> float:
            best = 0.0
            for i in idxs:
                painted = prepared[i].get("painted") or {}
                best = max(best, _num(painted.get("w")) * _num(painted.get("h")))
            return best

        ranked_clusters = list(clusters.values())
        if font_options.get("prominence_budget", True):
            ranked_clusters = sorted(ranked_clusters, key=_cluster_prominence, reverse=True)
        for idxs in ranked_clusters:
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
        elif (line.get("meta") or {}).get("strikethrough") and not (line.get("meta") or {}).get("native_decoration_shapes"):
            # OCR's deterministic strike detector (_detect_strike) and the chroma-based
            # foreign-ink detector both flag hand-drawn scribble strikes that
            # _native_text_decoration (horizontal-rule only) cannot see (091: red
            # diagonal scribbles over "Foggy", "NOT BACKED"...). Author the decoration
            # here so preview and the exported Figma node both draw a strike; carry the
            # sampled strike colour and the struck x-span so the render matches (partial:
            # strike "Foggy", not "and Steady"). Coloured price rules already emit precise
            # vector shapes (native_decoration_shapes) — defer to them, don't double-draw.
            meta = line["meta"]
            swipe = _hand_swipe_rule(meta, line.get("painted_box") or line.get("box"))
            if swipe is not None:
                # A hand-drawn swipe is not a typographic rule: it runs at its own angle,
                # is several times thicker than a text-decoration line, and overshoots the
                # glyph run (091's strike starts LEFT of "Foggy"). Emitting a flat
                # box-width STRIKETHROUGH throws all three away. Re-emit the measured ink
                # as a real vector rule (the same shape contract the coloured price rules
                # already use) so the mark keeps its length, angle and weight — and stays
                # editable rather than becoming a raster chip.
                meta.setdefault("native_decoration_shapes", []).append(swipe)
                meta["strike_render"] = "vector-swipe"
            else:
                line["style"]["textDecoration"] = "STRIKETHROUGH"
                span = _strike_span_fraction(
                    meta.get("strikethrough_box"), line.get("painted_box") or line.get("box"))
                if span:
                    line["style"]["decorationSpan"] = span
                strike_col = meta.get("strike_ink_color")
                if strike_col:
                    line["style"]["decorationColor"] = strike_col
                meta["strike_render"] = "text-decoration"
        match_evidence = match_evidence_by_index.get(i)
        if match_evidence:
            line.setdefault("meta", {})["font_match"] = copy.deepcopy(match_evidence)
        item["line_fit"] = _apply_line_render_fit(line, item["font_mask"], item["painted"], render_fit_options)
        _enrich_word_styles(image, line, config)
        _repair_non_glyph_line_paint(line)
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

    # Pass 3.55: platform-UI Inter prior (CODIA-PARITY for social screenshots).
    # Chirp / system UI is visually Inter-like; per-line forensics scatter Carlito/
    # Arimo/Caladea and then the slice gate rasterizes correct-class fits.
    ui_prior = _apply_platform_ui_font_prior(prepared, cfg, render_fit_options)
    if ui_prior:
        result["platform_ui_font_prior"] = ui_prior

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

    # Pass 3.7: handwriting gate. Runs AFTER the fidelity gate because it consumes the
    # same render-back evidence and may override the verdict in one direction only —
    # confirmed hand-lettering is forced to a pixel-exact chip. It never rescues a line
    # the fidelity gate already sliced, and never rasterizes without a positive VLM
    # identification, so typeset copy cannot be demoted by this pass.
    handwriting_evidence = _apply_handwriting_gate(prepared, image, cfg, run_dir)
    if handwriting_evidence:
        result["handwriting"] = handwriting_evidence

    # Belt-and-suspenders Codia tracking policy: no stage may emit fitted letterSpacing.
    for line in enriched:
        style = line.get("style")
        if isinstance(style, dict):
            style["letterSpacing"] = 0.0
        for word in line.get("words") or []:
            if isinstance(word, dict) and isinstance(word.get("style"), dict):
                word["style"]["letterSpacing"] = 0.0

    _assign_roles(enriched, canvas)
    _prefer_plain_editable_text(enriched)
    blocks = _make_blocks(enriched, canvas, config)
    block_unify = _unify_block_families(enriched, prepared, render_fit_options, font_options)
    if block_unify:
        result["block_font_unification"] = block_unify
    row_label_unify = _unify_repeated_row_labels(enriched)
    if row_label_unify:
        result["row_label_unification"] = row_label_unify
    peer_scale_unify = _unify_peer_text_scale(enriched)
    # Blocks first (tightest evidence), then columns — which recover the peers the block
    # grouper's role veto stranded in singleton blocks (101's checklist).
    peer_scale_unify = (peer_scale_unify or []) + _unify_column_text_scale(enriched)
    if peer_scale_unify:
        result["peer_scale_unification"] = peer_scale_unify
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
