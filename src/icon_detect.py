"""icon_detect.py — post-fusion ✓ / ✗ / ? glyph icons and chart-region clusters.

Comparison ads live and die on their list glyphs: green checkmarks on the "us" column
and red X marks on the "them" column.  SAM's text prompts catch "verified checkmark"
but there is no cross prompt, and the residual-CC pass swallows small glyphs into the
full-canvas CC (nested consolidation) or masks them out under dilated OCR boxes — so
red ✗ marks routinely ship missing (benchmark 101) while detected glyphs ship as
stacked duplicate fragments (benchmark 066).

This module is a deterministic, CPU-only refinement pass that runs on the FUSED
element list (called from :func:`element_fusion.fuse` when a run_dir is available):

1. Row-anchored glyph detection — for every OCR body line, a small color-contrast
   blob in the strip left of the text is extracted and classified against synthesized
   check / cross / question templates (color-robust: red ✗ on white works exactly like
   green ✓; a filled chip container is peeled to its inner glyph first).  A column-
   consistency prior accepts borderline rows when two or more detections share a
   column x.
2. Standalone glyph scan — large ✓ / ✗ / ? marks not tied to a text row (the white
   "?" over the mystery product in 101) found via extreme-color CCs plus the same
   template gate.
3. Reconciliation — detections are matched onto fused icon elements: the primary
   element receives the glyph role (verified / cross / question-mark), row-attachment
   meta linking it to its text line, and a normalized vector template; overlapping
   duplicate fragments are absorbed into the primary (fixes 066's "X+check stacked");
   unmatched detections are appended as NEW icon elements with box-local masks.
4. Chart region — >=3 long, evenly spaced horizontal gridlines mark a chart; the
   fused raster element covering them is re-roled to "chart" (an intentional raster
   cluster per src/raster_clusters.py) so the whole region slices as ONE clean crop
   instead of half-remaining on the background plate (benchmark 107).  Axis labels
   below the plot area are intentionally left OUTSIDE the bbox so they stay native
   editable text.

Everything here is additive and advisory: any exception degrades to the unmodified
fused list.  Heavy deps (numpy, cv2, scipy) load lazily.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

DEFAULTS = {
    "enabled": True,
    # row-anchored search
    "row_min_line_h": 10,
    "row_max_line_h": 78,
    "row_window_left_x_factor": 2.8,   # search strip width, in line heights
    "row_window_y_pad": 0.45,          # vertical padding, in line heights
    "row_min_contrast": 40.0,          # min color distance for a glyph pixel
    "row_accept_score": 0.55,
    "row_column_accept_score": 0.42,   # relaxed floor once a column has >=2 members
    "column_x_tol_factor": 0.9,        # same-column tolerance, in glyph heights
    # inline platform-UI glyph scan (009's verified badge)
    "inline_enabled": True,
    "inline_min_h_ratio": 0.65,   # vs the adjacent line: a badge matches the cap
                              # height, a letter counter is a fraction of it
    "inline_max_h_ratio": 1.60,
    "inline_max_gap_ratio": 0.90,  # horizontal gap, in line heights
    "inline_min_center_overlap": 0.45,
    "inline_min_fill": 0.30,       # a UI glyph is a compact solid, not a stroke wisp
    "inline_max_aspect": 1.60,
    "inline_min_area": 80,
    "inline_row_min_candidates": 4,  # need enough glyphs to know the copy's ink
    "inline_max_tone_share": 0.25,   # a tone held by more of the row IS the copy
    # standalone scan
    "standalone_enabled": True,
    "standalone_min_h_frac": 0.035,
    "standalone_max_h_frac": 0.22,
    "standalone_accept_score": 0.60,
    "standalone_text_overlap_max": 0.35,
    # chip (filled container) handling
    "chip_fill_ratio": 0.55,
    "chip_inner_contrast": 60.0,
    # chart region
    "chart_enabled": True,
    "chart_min_gridlines": 3,
    "chart_min_len_frac": 0.30,
    "chart_grad_thresh": 14.0,
    "chart_spacing_max_ratio": 1.9,
    "chart_reroll_iou": 0.40,
    # reconciliation
    "match_iou": 0.20,
    "absorb_group_pad": 3,
    # row-icon / text-box overlap clip signal (066 L10/L15: OCR swallowed the leading
    # ✗ glyph into the text line box, so the box starts ON the icon column). We do not
    # own the text node; we publish the x the text should start at (meta.row.text_clip_x)
    # so build_design_json / merge_layers can clip the box's LEFT edge off the icon.
    "clip_overlap_tol": 2.0,
    # brand-lockup (wordmark) proposal — a small, isolated, TWO-TONE text stack (101
    # 'craFT'/'cadence': teal small-caps over black) is custom brand artwork, not two
    # editable font lines. Proposed here as a raster 'logo' element; downstream suppresses
    # the covered OCR lines and emits the pixel-exact chip (badge agent).
    "lockup_enabled": True,          # run detection + write evidence to icon_detect.json
    # Emitting the logo ELEMENT changes the render, and the badge agent must land the
    # OCR-line suppression + raster-chip emission in the SAME change or 101 ships a
    # duplicate (raster wordmark under editable text). Default-off until that lands
    # (mirrors peel's default-off wiring); flip with the badge diff.
    "lockup_emit": False,
    "lockup_sat_min": 25.0,          # min ink chroma (max-min RGB) to read as a brand hue
    "lockup_two_tone_dist": 60.0,    # min ink colour distance between two stack lines
    "lockup_stack_gap": 1.5,         # vertical gap between stacked lines, in line heights
    "lockup_align_tol": 0.35,        # left/center edge agreement, in line heights
    "lockup_iso_gap": 1.4,           # a foreign line this close (in heights) breaks isolation
    "lockup_max_area_frac": 0.035,   # union bbox vs canvas
    "lockup_max_chars": 16,
    "lockup_cover_frac": 0.70,       # skip if this deep inside an existing logo/badge/burst
}

_GLYPH_ROLE = {"check": "verified", "cross": "cross", "question": "question-mark"}
# Normalized (0..1 of glyph box) stroke templates handed downstream so a vector
# emitter can rebuild the mark natively instead of tracing raster noise.
_VECTOR_TEMPLATES = {
    "check": {"paths": [[[0.08, 0.55], [0.38, 0.88], [0.92, 0.12]]],
              "stroke_width_frac": 0.18},
    "cross": {"paths": [[[0.12, 0.12], [0.88, 0.88]], [[0.88, 0.12], [0.12, 0.88]]],
              "stroke_width_frac": 0.20},
    "question": {"paths": None, "stroke_width_frac": None},  # keep as chip
}


def _np():
    import numpy as np
    return np


def _cv2():
    import cv2
    return cv2


# ── template synthesis + classification ──────────────────────────────────────────────

_TEMPLATE_CACHE: Optional[dict] = None
_T = 48  # template edge


def _norm48(mask):
    """Crop a bool mask to its bbox, letterbox to square, resize to 48x48 bool."""
    np, cv2 = _np(), _cv2()
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = sub.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.uint8)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = sub
    out = cv2.resize(pad * 255, (_T, _T), interpolation=cv2.INTER_AREA)
    return out >= 96


def _templates() -> dict:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE
    np, cv2 = _np(), _cv2()
    out = {"check": [], "cross": [], "question": []}
    for t in (4, 6, 8, 11, 14):
        m = np.zeros((_T, _T), np.uint8)
        cv2.line(m, (5, 5), (42, 42), 255, t)
        cv2.line(m, (42, 5), (5, 42), 255, t)
        out["cross"].append(_norm48(m > 0))
        for pts in ([(3, 26), (17, 41), (44, 6)], [(4, 30), (16, 42), (43, 12)],
                    [(6, 24), (19, 37), (42, 9)]):
            m = np.zeros((_T, _T), np.uint8)
            cv2.polylines(m, [np.asarray(pts, np.int32)], False, 255, t)
            out["check"].append(_norm48(m > 0))
    for t in (2, 3, 5, 7):
        for font in (cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX):
            m = np.zeros((_T, _T), np.uint8)
            cv2.putText(m, "?", (8, 42), font, 1.5, 255, t)
            norm = _norm48(m > 0)
            if norm is not None:
                out["question"].append(norm)
    _TEMPLATE_CACHE = {k: [v for v in vs if v is not None] for k, vs in out.items()}
    return _TEMPLATE_CACHE


def _iou_masks(a, b) -> float:
    np = _np()
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def classify_glyph(mask, exclude=()) -> tuple[Optional[str], float]:
    """Return (glyph, score) for a box-local bool mask, or (None, 0.0).

    Template IoU over synthesized strokes, corrected by mirror symmetry: an ✗ is
    invariant under horizontal flip, a ✓ is decisively not.  ``exclude`` removes
    glyph classes from the running (row bullets are only ever ✓ or ✗).
    """
    np = _np()
    norm = _norm48(np.asarray(mask, bool))
    if norm is None or norm.sum() < 20:
        return None, 0.0
    scores = {}
    for glyph, temps in _templates().items():
        if glyph in exclude:
            continue
        scores[glyph] = max((_iou_masks(norm, t) for t in temps), default=0.0)
    sym = _iou_masks(norm, norm[:, ::-1])
    if sym < 0.45 and "cross" in scores:
        scores["cross"] *= 0.6
    if sym > 0.75 and "check" in scores:
        scores["check"] *= 0.6
    # A genuine stroke glyph never fills its bbox: a solid rounded plate (packshot
    # cap, sticker) can reach ~0.55 IoU against the fattest cross template.
    if float(norm.mean()) > 0.62:
        for key in ("cross", "check"):
            if key in scores:
                scores[key] *= 0.3
    if not scores:
        return None, 0.0
    glyph = max(scores, key=scores.get)
    return glyph, float(scores[glyph])


# ── shared small helpers ─────────────────────────────────────────────────────────────

def _lines_from_ocr(ocr) -> list[dict]:
    lines = []
    raw = ocr.get("lines", []) if isinstance(ocr, dict) else (ocr or [])
    for ln in raw:
        b = ln.get("box") if isinstance(ln, dict) else None
        if b and b.get("w", 0) > 0 and b.get("h", 0) > 0:
            lines.append({"id": ln.get("id"), "text": ln.get("text", ""),
                          "box": {k: float(b[k]) for k in ("x", "y", "w", "h")}})
    return lines


def _box_iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _overlap_frac(inner: dict, outer: dict) -> float:
    ix = max(0.0, min(inner["x"] + inner["w"], outer["x"] + outer["w"]) - max(inner["x"], outer["x"]))
    iy = max(0.0, min(inner["y"] + inner["h"], outer["y"] + outer["h"]) - max(inner["y"], outer["y"]))
    return (ix * iy) / max(1.0, inner["w"] * inner["h"])


def _dominant_color(crop):
    """Most common 4-bit quantized color of an RGB crop (the local plate fill)."""
    np = _np()
    q = (crop.astype(np.int64) >> 4)
    keys = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    vals, counts = np.unique(keys.ravel(), return_counts=True)
    k = int(vals[counts.argmax()])
    base = np.array([(k >> 8) & 15, (k >> 4) & 15, k & 15], np.float64) * 16 + 8
    sel = keys == k
    if sel.sum():
        base = crop[sel].astype(np.float64).mean(axis=0)
    return base


def _color_prior(mean_rgb) -> Optional[str]:
    r, g, b = [float(v) for v in mean_rgb]
    if r - max(g, b) >= 45:
        return "cross"
    if g - max(r, b) >= 30:
        return "check"
    return None


# ── 1. row-anchored glyph detection ──────────────────────────────────────────────────

def _extract_blobs(crop, min_contrast: float, max_blobs: int = 6):
    """Contrast blobs in a window crop, largest first → [(mask, bg_color), ...].

    Returns several candidate CCs: the LARGEST one is often window furniture (a
    card border crossing the strip), while the glyph is the compact runner-up.
    """
    np, cv2 = _np(), _cv2()
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return []
    bg = _dominant_color(crop)
    dist = np.sqrt(((crop.astype(np.float64) - bg) ** 2).sum(axis=2))
    if float(dist.max()) < min_contrast * 1.15:
        return []
    m = (dist > max(min_contrast, 0.40 * float(dist.max()))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    out = []
    for i in order[:max_blobs]:
        if int(stats[i, cv2.CC_STAT_AREA]) < 16:
            break
        out.append((labels == i, bg))
    return out


def _classify_blob(crop, blob, opts, bg=None, exclude=()) -> tuple[Optional[str], float, dict]:
    """Classify a blob, peeling a filled chip container to its inner glyph."""
    np, cv2 = _np(), _cv2()
    from scipy import ndimage as _ndi
    ys, xs = np.nonzero(blob)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    # Sealed hull: a check whose tip touches the chip rim leaks its "hole" to the
    # outside, so plain fill_holes misses it — close with a glyph-scale kernel first.
    k = max(3, int(round(0.22 * min(bw, bh))))
    hull = cv2.morphologyEx(blob.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8))
    hull = _ndi.binary_fill_holes(hull > 0)
    fill = float(hull.sum()) / max(1, bw * bh)
    info = {"fill_ratio": round(fill, 3), "chip": False}
    # Direct classification of the blob silhouette (bare ✓/✗ strokes).
    glyph, score = classify_glyph(blob, exclude=exclude)
    target = blob
    if fill >= float(opts["chip_fill_ratio"]):
        # Filled chip (green rounded square with white check, dark circle badge):
        # also try the INNER glyph instead of the container silhouette.  The glyph
        # is a HOLE in the blob (bg-colored check inside the chip) or a contrast
        # region against the chip fill — take the larger.  A thick rounded ✗ can
        # reach fill ~0.6 too, so the better of direct/inner wins.
        sel = crop[blob].astype(np.float64)
        glyph_color = np.median(sel, axis=0)
        dist = np.sqrt(((crop.astype(np.float64) - glyph_color) ** 2).sum(axis=2))
        contrast_inner = np.logical_and(blob, dist > float(opts["chip_inner_contrast"]))
        # Erode the sealed hull before extracting holes: the close above leaves a
        # dilated rim ring around the chip outline that would pollute the glyph.
        hull_core = cv2.erode(hull.astype(np.uint8),
                              np.ones((k // 2 + 2, k // 2 + 2), np.uint8)) > 0
        hole_inner = np.logical_and(hull_core, ~blob)
        # a chip border/gradient also lands in contrast_inner — drop the rim ring
        blob_core = cv2.erode(blob.astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
        candidates = [hole_inner, contrast_inner,
                      np.logical_and(contrast_inner, blob_core),
                      # chip-color contrast inside the eroded hull: catches an AA'd
                      # glyph whose pixels were thresholded INTO the blob, while the
                      # erosion drops the border ring entirely
                      np.logical_and(hull_core,
                                     dist > float(opts["chip_inner_contrast"]))]
        if hull_core.sum() > 30:
            # adaptive: bevel/gradient rims sit well below the glyph's contrast —
            # threshold at roughly half the strongest contrast inside the chip
            thr = max(float(opts["chip_inner_contrast"]),
                      0.45 * float(np.percentile(dist[hull_core], 99)))
            candidates.append(np.logical_and(hull_core, dist > thr))
        for inner in candidates:
            if inner.sum() < max(20, 0.03 * hull.sum()):
                continue
            inner_glyph, inner_score = classify_glyph(inner, exclude=exclude)
            if inner_score > score:
                glyph, score, target = inner_glyph, inner_score, inner
                info["chip"] = True
                info["chip_fill"] = [int(v) for v in glyph_color]
    prior = _color_prior(crop[target].astype(np.float64).mean(axis=0)
                         if target.sum() else crop[blob].astype(np.float64).mean(axis=0))
    if prior and glyph == prior:
        score = min(1.0, score + 0.06)
    info["color_prior"] = prior
    return glyph, score, info


_LEADING_GLYPH_TOKEN = ("x ", "x\t", "✗", "✕", "✖", "- ", "– ", "— ", "• ", "· ",
                        "v ", "√")


def _line_swallowed_glyph(text: str) -> bool:
    """OCR sometimes reads the row glyph as a leading 'X '/'- ' token; when it does,
    the line box starts AT the glyph, so the search window must extend into it."""
    return str(text or "").strip().lower().startswith(_LEADING_GLYPH_TOKEN)


def detect_row_icons(rgb, lines: list[dict], canvas: dict, opts: dict) -> list[dict]:
    np = _np()
    h_img, w_img = rgb.shape[:2]
    dets = []
    for line in lines:
        b = line["box"]
        h = b["h"]
        if not (float(opts["row_min_line_h"]) <= h <= float(opts["row_max_line_h"])):
            continue
        swallowed = _line_swallowed_glyph(line.get("text"))
        x1 = int(round(b["x"] + (1.4 * h if swallowed else -0.08 * h)))
        x0 = int(round(b["x"] - float(opts["row_window_left_x_factor"]) * h))
        pad = float(opts["row_window_y_pad"]) * h
        # multi-line rows center their glyph on the ROW, which extends downward
        # from this (first) line — give the window extra room below.
        y0 = int(round(b["y"] - pad))
        y1 = int(round(b["y"] + h + max(pad, 1.4 * h)))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w_img, x1), min(h_img, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = rgb[y0:y1, x0:x1]
        best = None
        for blob, bg in _extract_blobs(crop, float(opts["row_min_contrast"])):
            ys, xs = np.nonzero(blob)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            # size / aspect gates: a list glyph is about one line height and compact
            if not (0.40 * h <= max(bw, bh) <= 1.9 * h):
                continue
            if not (0.45 <= bw / max(1, bh) <= 2.2):
                continue
            # the glyph belongs to THIS row: its center may sit slightly below the
            # first line (multi-line rows) but never a full row further down
            cy = y0 + float(ys.min()) + bh / 2.0
            if not (b["y"] - 0.5 * h <= cy <= b["y"] + 1.25 * h):
                continue
            # a row bullet is ✓ or ✗; "?" only comes from the standalone scan
            glyph, score, info = _classify_blob(crop, blob, opts, bg=bg,
                                                exclude=("question",))
            if glyph is None:
                continue
            if best is None or score > best[1]:
                best = (glyph, score, info, blob, ys, xs, bw, bh)
        if best is None:
            continue
        glyph, score, info, blob, ys, xs, bw, bh = best
        box = {"x": x0 + int(xs.min()), "y": y0 + int(ys.min()), "w": bw, "h": bh}
        full = np.zeros((h_img, w_img), bool)
        full[y0 + ys.min():y0 + ys.max() + 1, x0 + xs.min():x0 + xs.max() + 1] = \
            blob[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if swallowed:
            info = dict(info)
            info["text_prefix_glyph"] = True
        dets.append({
            "glyph": glyph, "score": round(float(score), 4), "box": box, "mask": full,
            "row_text_id": line["id"], "row_box": dict(b), "info": info,
            "anchor": "row",
        })
    # de-dupe: the same blob reached from two wrapped lines keeps its best row anchor
    dets.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in dets:
        if any(_box_iou(d["box"], k["box"]) > 0.5 for k in kept):
            continue
        kept.append(d)
    # column-consistency prior: >=2 members sharing a column x accept a relaxed floor
    accept = float(opts["row_accept_score"])
    relaxed = float(opts["row_column_accept_score"])
    tol_factor = float(opts["column_x_tol_factor"])
    final = []
    for d in kept:
        cx = d["box"]["x"] + d["box"]["w"] / 2.0
        tol = tol_factor * max(8.0, d["box"]["h"])
        column = [o for o in kept
                  if abs((o["box"]["x"] + o["box"]["w"] / 2.0) - cx) <= tol]
        strong = [o for o in column if o["score"] >= accept]
        d["column_size"] = len(column)
        if d["score"] >= accept or (len(strong) >= 1 and len(column) >= 2
                                    and d["score"] >= relaxed):
            final.append(d)
    return final


# ── 2. standalone big-glyph scan (the 101 white "?") ─────────────────────────────────

def _extreme_masks(rgb):
    np = _np()
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return {
        "white": (r >= 232) & (g >= 232) & (b >= 232),
        "black": (r <= 38) & (g <= 38) & (b <= 38),
        "red": (r - np.maximum(g, b)) >= 60,
        "green": (g - np.maximum(r, b)) >= 45,
        # Platform UI chrome is overwhelmingly saturated blue: the X/Twitter and
        # Meta verified badges, inline app logos, link glyphs. Its absence here is
        # why 009's verified check was never a standalone candidate.
        "blue": (b - np.maximum(r, g)) >= 45,
    }


def detect_standalone_glyphs(rgb, lines: list[dict], canvas: dict, opts: dict,
                             taken: list[dict]) -> list[dict]:
    np, cv2 = _np(), _cv2()
    h_img, w_img = rgb.shape[:2]
    min_h = float(opts["standalone_min_h_frac"]) * h_img
    max_h = float(opts["standalone_max_h_frac"]) * h_img
    out = []
    for tone, m in _extreme_masks(rgb).items():
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), 8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if area < 60:
                continue
            comps.append({"i": i, "x": x, "y": y, "w": w, "h": h, "area": area})
        for c in comps:
            if not (min_h <= c["h"] <= max_h) and not (min_h <= c["w"] <= max_h):
                continue
            ids = [c["i"]]
            box = {k: float(c[k]) for k in ("x", "y", "w", "h")}
            # question-dot completion: a small aligned CC just below the hook
            for o in comps:
                if o["i"] == c["i"] or o["area"] > 0.35 * c["area"]:
                    continue
                cx, ox = c["x"] + c["w"] / 2, o["x"] + o["w"] / 2
                gap = o["y"] - (c["y"] + c["h"])
                if abs(cx - ox) <= 0.4 * c["w"] and 0 <= gap <= 1.0 * c["w"]:
                    ids.append(o["i"])
                    box["h"] = o["y"] + o["h"] - box["y"]
                    box["x"] = min(box["x"], float(o["x"]))
                    box["w"] = max(c["x"] + c["w"], o["x"] + o["w"]) - box["x"]
                    break
            if not (0.25 <= box["w"] / max(1.0, box["h"]) <= 1.6):
                continue
            # letters/digits guard: skip anything mostly inside an OCR line
            overlap = max((_overlap_frac(box, ln["box"]) for ln in lines), default=0.0)
            if overlap > float(opts["standalone_text_overlap_max"]):
                continue
            if any(_box_iou(box, t["box"]) > 0.3 for t in taken + out):
                continue
            x0, y0 = int(box["x"]), int(box["y"])
            x1, y1 = int(box["x"] + box["w"]), int(box["y"] + box["h"])
            blob = np.isin(labels[y0:y1, x0:x1], ids)
            glyph, score = classify_glyph(blob)
            if glyph is None or score < float(opts["standalone_accept_score"]):
                continue
            full = np.zeros((h_img, w_img), bool)
            full[y0:y1, x0:x1] = blob
            out.append({
                "glyph": glyph, "score": round(float(score), 4),
                "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                "mask": full, "row_text_id": None, "row_box": None,
                "info": {"tone": tone}, "anchor": "standalone",
            })
    out.sort(key=lambda d: -d["score"])
    kept = []
    for d in out:
        if any(_box_iou(d["box"], k["box"]) > 0.4 for k in kept):
            continue
        kept.append(d)
    return kept


# ── 2b. inline platform-UI glyph scan (the 009 verified badge) ───────────────────────

def detect_inline_glyphs(rgb, lines: list[dict], canvas: dict, opts: dict,
                         taken: list[dict]) -> list[dict]:
    """Compact saturated glyphs sitting inline with a text row.

    009's blue verified badge is missed by BOTH existing scans, on three
    independent counts: it is small (29px on a 1080 canvas, under
    ``standalone_min_h_frac``), it is ADJACENT to text so the letters/digits
    guard rejects it (the 'UPFRONT' line box actually ends at x=378, right across
    the badge at 351..380), and it is a filled disc rather than a stroke glyph, so
    ``classify_glyph`` scores it near zero.

    This scan deliberately does NOT classify the mark.  Per the raster-first icon
    policy a platform glyph ships as a pixel-exact IMAGE chip, so its identity is
    irrelevant — only that it is a compact, colour-pure blob riding a text
    baseline.  That generalises past the verified badge to inline app logos, link
    and lock glyphs, and engagement marks.  Each detection carries ``row_text_id``
    so the chip can be anchored after the display name rather than floated.
    """
    np, cv2 = _np(), _cv2()
    h_img, w_img = rgb.shape[:2]
    if not lines:
        return []
    candidates = []
    for tone, mask in _extreme_masks(rgb).items():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        for index in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[index]]
            if area < int(opts["inline_min_area"]) or w <= 0 or h <= 0:
                continue
            aspect = w / max(1.0, float(h))
            if not (1.0 / float(opts["inline_max_aspect"]) <= aspect
                    <= float(opts["inline_max_aspect"])):
                continue
            # A UI glyph is a compact solid; a stray stroke fragment is not.
            if area / float(w * h) < float(opts["inline_min_fill"]):
                continue
            box = {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
            row = _inline_row(box, lines, opts)
            if row is None:
                continue
            candidates.append({"tone": tone, "box": box, "row": row,
                               "label": index, "labels": labels,
                               "slice": (x, y, w, h)})
    # Letters guard.  Every glyph of the copy also rides its row and is also a
    # compact blob, so shape alone cannot separate them — scanning 009 without
    # this returns 137 detections, one per letter of the body text.  The
    # discriminator is COLOUR: a row's copy is set in one ink, so the majority
    # tone on a row IS the copy and only a minority tone can be chrome.  009's
    # 'UPFRONT' row is ten white letters plus one blue badge; the badge is the
    # only outlier.  This generalises to any inline logo/lock/link glyph, which
    # are likewise coloured differently from the text they sit beside.
    by_row: dict = {}
    for cand in candidates:
        by_row.setdefault(str(cand["row"].get("id")), []).append(cand)
    out = []
    min_peers = int(opts["inline_row_min_candidates"])
    max_share = float(opts["inline_max_tone_share"])
    for row_id, group in by_row.items():
        if len(group) < min_peers:
            continue  # too few glyphs on the row to know what the copy's ink is
        tones: dict = {}
        for cand in group:
            tones[cand["tone"]] = tones.get(cand["tone"], 0) + 1
        for cand in group:
            if tones[cand["tone"]] / float(len(group)) > max_share:
                continue  # this tone IS the row's copy
            box = cand["box"]
            if any(_box_iou(box, other["box"]) > 0.3 for other in taken + out):
                continue
            x, y, w, h = cand["slice"]
            blob = cand["labels"][y:y + h, x:x + w] == cand["label"]
            full = np.zeros((h_img, w_img), bool)
            full[y:y + h, x:x + w] = blob
            row = cand["row"]
            out.append({
                "glyph": None, "score": round(1.0 - tones[cand["tone"]] / float(len(group)), 4),
                "box": {"x": x, "y": y, "w": w, "h": h},
                "mask": full, "row_text_id": row.get("id"), "row_box": row.get("box"),
                "info": {"tone": cand["tone"], "inline": True,
                         "row_tone_share": round(tones[cand["tone"]] / float(len(group)), 4),
                         "row_text": str(row.get("text") or "")[:40]},
                "anchor": "inline", "role": "chrome",
            })
    kept = []
    for det in sorted(out, key=lambda d: -(d["box"]["w"] * d["box"]["h"])):
        if any(_box_iou(det["box"], other["box"]) > 0.4 for other in kept):
            continue
        kept.append(det)
    return kept


def _inline_row(box: dict, lines: list[dict], opts: dict) -> Optional[dict]:
    """The text row this glyph rides, or None.

    'Rides' means vertically centred on the row and horizontally touching its
    run — which is true whether the OCR box stops short of the glyph (the ⏳ after
    009's headline) or swallows it (the badge inside the 'UPFRONT' box).
    """
    best = None
    for line in lines:
        lbox = line.get("box") or {}
        lh = float(lbox.get("h") or 0.0)
        if lh <= 0:
            continue
        ratio = box["h"] / lh
        if not (float(opts["inline_min_h_ratio"]) <= ratio <= float(opts["inline_max_h_ratio"])):
            continue
        # Vertically centred on the row.
        gcy = box["y"] + box["h"] / 2.0
        lcy = float(lbox.get("y") or 0.0) + lh / 2.0
        if abs(gcy - lcy) > (1.0 - float(opts["inline_min_center_overlap"])) * lh:
            continue
        # Horizontally adjacent to (or inside) the row's run.
        lx0 = float(lbox.get("x") or 0.0)
        lx1 = lx0 + float(lbox.get("w") or 0.0)
        gap = max(lx0 - (box["x"] + box["w"]), box["x"] - lx1, 0.0)
        if gap > float(opts["inline_max_gap_ratio"]) * lh:
            continue
        if best is None or gap < best[0]:
            best = (gap, line)
    return best[1] if best else None


# ── 3. chart region detection ────────────────────────────────────────────────────────

def detect_chart_region(rgb, lines: list[dict], canvas: dict, opts: dict) -> Optional[dict]:
    """Return {"box", "gridlines": n} when >=N long, evenly spaced horizontal
    gridlines mark a plot area; labels beneath stay outside the bbox."""
    np, cv2 = _np(), _cv2()
    h_img, w_img = rgb.shape[:2]
    gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)
    gy = np.abs(np.pad(gray, ((1, 1), (0, 0)), mode="edge")[2:, :]
                - np.pad(gray, ((1, 1), (0, 0)), mode="edge")[:-2, :])
    edge = (gy > float(opts["chart_grad_thresh"])).astype(np.uint8)
    klen = max(31, w_img // 22)
    horiz = cv2.erode(edge, np.ones((1, klen), np.uint8))
    horiz = cv2.dilate(horiz, np.ones((1, klen), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    segs = []
    min_len = float(opts["chart_min_len_frac"]) * w_img
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if h <= 6 and w >= min_len:
            segs.append({"y": y + h / 2.0, "x0": x, "x1": x + w})
    if len(segs) < int(opts["chart_min_gridlines"]):
        return None
    # merge segments on the same y (a packshot may split one gridline in two)
    segs.sort(key=lambda s: s["y"])
    rows: list[dict] = []
    for s in segs:
        if rows and abs(s["y"] - rows[-1]["y"]) <= 5:
            rows[-1]["x0"] = min(rows[-1]["x0"], s["x0"])
            rows[-1]["x1"] = max(rows[-1]["x1"], s["x1"])
            rows[-1]["y"] = (rows[-1]["y"] + s["y"]) / 2.0
        else:
            rows.append(dict(s))
    if len(rows) < int(opts["chart_min_gridlines"]):
        return None
    # x-extent agreement: keep the largest subset with >=70% mutual overlap
    ref = max(rows, key=lambda r: r["x1"] - r["x0"])
    group = []
    for r in rows:
        inter = min(r["x1"], ref["x1"]) - max(r["x0"], ref["x0"])
        if inter >= 0.70 * (r["x1"] - r["x0"]):
            group.append(r)
    if len(group) < int(opts["chart_min_gridlines"]):
        return None
    gaps = [b["y"] - a["y"] for a, b in zip(group, group[1:])]
    if not gaps or min(gaps) <= 2:
        return None
    if max(gaps) / max(1e-6, min(gaps)) > float(opts["chart_spacing_max_ratio"]):
        return None
    sp = sum(gaps) / len(gaps)
    x0 = min(r["x0"] for r in group)
    x1 = max(r["x1"] for r in group)
    y0 = group[0]["y"] - 0.9 * sp   # headroom for a curve/arrow above the top line
    y1 = group[-1]["y"] + 0.5 * sp  # below the bottom gridline, above axis labels
    # never swallow the axis-label text row: clamp above the closest line below
    below = [ln["box"]["y"] for ln in lines
             if ln["box"]["y"] >= group[-1]["y"]
             and ln["box"]["x"] + ln["box"]["w"] > x0 and ln["box"]["x"] < x1]
    if below:
        y1 = min(y1, min(below) - 2)
    box = {"x": int(max(0, round(x0 - 0.015 * w_img))),
           "y": int(max(0, round(y0)))}
    box["w"] = int(min(w_img, round(x1 + 0.015 * w_img))) - box["x"]
    box["h"] = int(min(h_img, round(y1))) - box["y"]
    if box["w"] < 0.25 * w_img or box["h"] < 0.08 * h_img:
        return None
    return {"box": box, "gridlines": len(group), "spacing": round(sp, 1)}


# ── 3b. row-icon ↔ text-box overlap (066 L10/L15) ────────────────────────────────────

def _annotate_text_clip(dets: list[dict], opts: dict) -> None:
    """Flag row/inline glyphs whose linked OCR line box starts at/left of the icon's
    right edge and record where the text should begin.

    066's ``Smudges on upper lid`` / ``X Up to 3 shades`` rows have OCR boxes that start
    ON the red ✗ column (x≈824) instead of after it (x≈872) — OCR read the mark as a
    leading letter and swallowed it, so the text NODE inherits an icon-inflated left edge
    and the ✗ ink lands inside the text box (renders "Xmudges…"). We own the icon box, so
    we publish ``text_clip_x`` = icon-right + the list's own clean gap; build_design_json /
    merge_layers clip the text box's LEFT edge to it so icon ink never sits in a text node.
    Purely additive: clean rows (text starts after the icon) get nothing.
    """
    rows = [d for d in dets if d.get("row_box") and d.get("box")]
    tol = float(opts.get("clip_overlap_tol", 2.0))
    col_tol_factor = float(opts.get("column_x_tol_factor", 0.9))
    for d in rows:
        ib, tb = d["box"], d["row_box"]
        icon_right = ib["x"] + ib["w"]
        if tb["x"] >= icon_right - tol:
            continue  # text already starts after the icon — nothing to clip
        cx = ib["x"] + ib["w"] / 2.0
        col_tol = col_tol_factor * max(8.0, float(ib["h"]))
        clean_gaps = []
        for o in rows:
            ob, otb = o["box"], o["row_box"]
            if abs((ob["x"] + ob["w"] / 2.0) - cx) > col_tol:
                continue
            gap = otb["x"] - (ob["x"] + ob["w"])
            if gap >= tol:
                clean_gaps.append(gap)
        if clean_gaps:
            clean_gaps.sort()
            gap = clean_gaps[len(clean_gaps) // 2]
        else:
            gap = 0.5 * float(ib["h"])
        d["overlaps_text"] = True
        d["text_clip_x"] = int(round(icon_right + gap))


# ── 3c. brand-lockup (wordmark) proposal (101 'craft cadence') ────────────────────────

_LOCKUP_STOPCHARS = re.compile(r"[\d.!?,:;/@%$€£+*=]")
_LOCKUP_UI = re.compile(
    r"^(?:the|and|for|new|buy|get|off|free|sale|save|shop|now|our|your|with|vs\.?)$",
    re.I,
)


def _line_ink(rgb, box: dict):
    """Mean colour of the darkest ~30% (ink) pixels of an OCR line box, or None."""
    np = _np()
    x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
    x1 = min(rgb.shape[1], int(round(box["x"] + box["w"])))
    y1 = min(rgb.shape[0], int(round(box["y"] + box["h"])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    if crop.shape[0] < 4:
        return None
    lum = crop @ np.array([0.299, 0.587, 0.114])
    thr = float(np.percentile(lum, 30))
    ink = crop[lum <= thr]
    if ink.size == 0:
        ink = crop
    return ink.mean(axis=0)


def _brand_hue(mean_rgb, sat_min: float) -> bool:
    if mean_rgb is None:
        return False
    m = mean_rgb
    return (float(m.max() - m.min()) >= sat_min
            and 14.0 < float(m.mean()) < 232.0)


def _colour_dist(a, b) -> float:
    np = _np()
    if a is None or b is None:
        return 0.0
    return float(np.sqrt(((a - b) ** 2).sum()))


def _lockup_word_ok(text: str, max_chars: int) -> bool:
    t = str(text or "").strip()
    if not t or len(t) > max_chars:
        return False
    if _LOCKUP_STOPCHARS.search(t) or _LOCKUP_UI.match(t):
        return False
    words = t.split()
    if len(words) > 2:
        return False
    return sum(ch.isalpha() for ch in t) >= 3


def _union_box(boxes: list[dict]) -> dict:
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _x_overlap_frac(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    return ix / max(1.0, min(a["w"], b["w"]))


def detect_brand_lockups(rgb, lines: list[dict], canvas: dict, fused: list[dict],
                         opts: dict) -> list[dict]:
    """Propose small isolated TWO-TONE text stacks as raster 'logo' elements.

    A custom wordmark that font-matching cannot reproduce (101's teal 'craFT' small-caps
    over black 'cadence') ships today as two mismatched text lines because it sits
    mid-canvas beside the product — outside ``wordmark.is_wordmark_candidate``'s
    header/footer slot, and SAM's 'logo' prompt only caught the round BOGO badge. The
    decisive, low-false-positive signal is TWO-TONE INK in a tight, isolated, short-word
    stack: body columns and offer bursts are single-hue, and a genuine multi-colour
    lettermark is almost never ordinary copy. Emits a logo element carrying the covered
    OCR line ids; the badge agent suppresses those lines and rasterizes the chip.
    """
    if not opts.get("lockup_enabled", True) or not lines:
        return []
    np = _np()
    H, W = rgb.shape[:2]
    sat_min = float(opts.get("lockup_sat_min", 25.0))
    max_chars = int(opts.get("lockup_max_chars", 16))
    stack_gap = float(opts.get("lockup_stack_gap", 1.5))
    align_tol = float(opts.get("lockup_align_tol", 0.35))
    iso_gap = float(opts.get("lockup_iso_gap", 1.4))
    two_tone = float(opts.get("lockup_two_tone_dist", 60.0))

    feats = []
    for ln in lines:
        b = ln.get("box") or {}
        if float(b.get("w", 0)) < 3 or float(b.get("h", 0)) < 3:
            continue
        ok = _lockup_word_ok(ln.get("text"), max_chars)
        ink = _line_ink(rgb, b) if ok else None
        feats.append({"ln": ln, "box": {k: float(b[k]) for k in ("x", "y", "w", "h")},
                      "ok": ok, "ink": ink,
                      "hue": _brand_hue(ink, sat_min) if ok else False})
    candidates = [f for f in feats if f["ok"]]

    # existing elements that would make a coloured stack NOT a fresh wordmark: an offer
    # burst / badge / logo already owns the region (101's BOGO), or it is deep inside one.
    cover_frac = float(opts.get("lockup_cover_frac", 0.70))
    cover_roles = {"logo", "badge", "sale_burst", "price_burst", "starburst",
                   "sticker", "button", "seal", "verified"}
    covers = [el for el in (fused or [])
              if str(el.get("role") or "").lower() in cover_roles and el.get("box")]

    used: list[int] = []
    out = []
    for i, seed in enumerate(candidates):
        if i in used or not seed["hue"]:
            continue
        cluster = [seed]
        cidx = [i]
        sb, sh = seed["box"], seed["box"]["h"]
        scx = sb["x"] + sb["w"] / 2.0
        for j, other in enumerate(candidates):
            if j == i or j in used:
                continue
            ob = other["box"]
            cb = _union_box([c["box"] for c in cluster])
            vgap = max(0.0, max(ob["y"] - (cb["y"] + cb["h"]),
                                cb["y"] - (ob["y"] + ob["h"])))
            if vgap > stack_gap * max(sh, ob["h"]):
                continue
            if _x_overlap_frac(cb, ob) < 0.25:
                continue
            ocx = ob["x"] + ob["w"] / 2.0
            aligned = (abs(ob["x"] - cb["x"]) <= align_tol * sh
                       or abs(ocx - scx) <= align_tol * sh)
            if not aligned:
                continue
            if abs(ob["h"] - sh) > 0.8 * sh:
                continue
            cluster.append(other)
            cidx.append(j)
        if len(cluster) < 2:
            continue
        # TWO-TONE gate: at least two members whose ink colours are decisively apart.
        inks = [c["ink"] for c in cluster if c["ink"] is not None]
        if len(inks) < 2 or max(_colour_dist(a, b)
                                for a in inks for b in inks) < two_tone:
            continue
        box = _union_box([c["box"] for c in cluster])
        if box["w"] * box["h"] > float(opts.get("lockup_max_area_frac", 0.035)) * W * H:
            continue
        # ISOLATION: no FOREIGN OCR line is stacked right against the block.
        member_ids = {id(c["ln"]) for c in cluster}
        foreign = False
        for ln in lines:
            if id(ln) in member_ids:
                continue
            b = ln.get("box") or {}
            if float(b.get("w", 0)) < 3 or float(b.get("h", 0)) < 3:
                continue
            fb = {k: float(b[k]) for k in ("x", "y", "w", "h")}
            vgap = max(0.0, max(fb["y"] - (box["y"] + box["h"]),
                                box["y"] - (fb["y"] + fb["h"])))
            if vgap <= iso_gap * box["h"] and _x_overlap_frac(box, fb) >= 0.30:
                foreign = True
                break
        if foreign:
            continue
        # not already owned by a burst/badge/logo (BOGO), and not the product's own label.
        if any(_overlap_frac(box, c["box"]) >= cover_frac for c in covers):
            continue
        used.extend(cidx)
        cluster.sort(key=lambda c: c["box"]["y"])
        out.append({
            "box": {"x": int(round(box["x"])), "y": int(round(box["y"])),
                    "w": int(round(box["w"])), "h": int(round(box["h"]))},
            "text_ids": [c["ln"].get("id") for c in cluster],
            "text": " ".join(str(c["ln"].get("text") or "").strip() for c in cluster),
            "inks": [[int(v) for v in c["ink"]] for c in cluster if c["ink"] is not None],
        })
    return out


# ── 4. reconciliation with the fused element list ────────────────────────────────────

def _write_mask(mask, box: dict, path: str) -> None:
    np = _np()
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    local = mask[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
    Image.fromarray((local.astype(np.uint8)) * 255).save(path)


def _next_id(fused: list[dict]) -> int:
    best = -1
    for el in fused:
        raw = str(el.get("id") or "")
        if raw.startswith("E"):
            try:
                best = max(best, int(raw[1:]))
            except ValueError:
                continue
    return best + 1


def _glyph_meta(det: dict) -> dict:
    meta = {
        "glyph": det["glyph"],
        "icon_cv": {"score": det["score"], "anchor": det["anchor"],
                    **{k: v for k, v in (det.get("info") or {}).items()}},
    }
    if det.get("row_text_id") is not None:
        row = {
            "text_id": det["row_text_id"],
            "line_box": det.get("row_box"),
            "align": "left-of-text",
            "dy_center": round((det["box"]["y"] + det["box"]["h"] / 2.0)
                               - (det["row_box"]["y"] + det["row_box"]["h"] / 2.0), 1)
            if det.get("row_box") else None,
        }
        # OCR swallowed the leading glyph into the text box: publish where the text
        # should start so the box's LEFT edge can be clipped off the icon (066 L10/L15).
        if det.get("overlaps_text"):
            row["overlaps_text"] = True
            row["text_clip_x"] = det.get("text_clip_x")
        meta["row"] = row
    template = _VECTOR_TEMPLATES.get(det["glyph"]) or {}
    if template.get("paths"):
        meta["vector_template"] = dict(template)
    return meta


_ICONISH_KINDS = {"icon"}
_RASTERISH_CHART_ROLES = {"photo", "photo-fragment", "photo_fragment", "illustration",
                          "shape", "image", "graphic", "object", "diagram", "graph"}


def refine(fused: list[dict], canvas: dict, cfg: Optional[dict] = None,
           run_dir: Optional[str] = None) -> list[dict]:
    """Glyph + chart refinement over the fused element list (see module docstring)."""
    cfg = cfg or {}
    opts = dict(DEFAULTS)
    opts.update(cfg.get("icon_detect") or {})
    if not opts.get("enabled", True) or not run_dir:
        return fused
    np = _np()
    cv2 = _cv2()
    img_path = os.path.join(run_dir, "normalized.png")
    if not os.path.exists(img_path):
        return fused
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return fused
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ocr = {}
    ocr_path = os.path.join(run_dir, "ocr.json")
    if os.path.exists(ocr_path):
        with open(ocr_path, "r", encoding="utf-8") as fh:
            ocr = json.load(fh)
    lines = _lines_from_ocr(ocr)

    report = {"detections": [], "matched": [], "added": [], "absorbed": [],
              "chart": None}

    dets = detect_row_icons(rgb, lines, canvas, opts)
    if opts.get("standalone_enabled", True):
        dets += detect_standalone_glyphs(rgb, lines, canvas, opts, dets)
    if opts.get("inline_enabled", True):
        dets += detect_inline_glyphs(rgb, lines, canvas, opts, dets)
    _annotate_text_clip(dets, opts)

    dropped_ids: set[str] = set()
    small_limit = 0.02 * canvas["w"] * canvas["h"]
    for det in dets:
        # An inline platform glyph ships as a pixel-exact chip and carries its own
        # role: it is deliberately never template-classified.
        role = det.get("role") or _GLYPH_ROLE[det["glyph"]]
        group = []
        for el in fused:
            if el["id"] in dropped_ids:
                continue
            box = el.get("box") or {}
            if box.get("w", 0) * box.get("h", 0) > max(small_limit,
                                                       9.0 * det["box"]["w"] * det["box"]["h"]):
                continue
            if el.get("kind") not in _ICONISH_KINDS and \
                    _box_iou(det["box"], box) < 0.5:
                continue
            if _box_iou(det["box"], box) >= float(opts["match_iou"]) or \
                    _overlap_frac(box, det["box"]) >= 0.6 or \
                    _overlap_frac(det["box"], box) >= 0.6:
                group.append(el)
        entry = {"glyph": det["glyph"], "score": det["score"], "box": det["box"],
                 "row_text_id": det.get("row_text_id"), "anchor": det["anchor"],
                 "column_size": det.get("column_size")}
        if det.get("overlaps_text"):
            entry["overlaps_text"] = True
            entry["text_clip_x"] = det.get("text_clip_x")
        report["detections"].append(entry)
        meta_patch = _glyph_meta(det)
        if group:
            # primary: prefer an element already carrying a glyph-ish role, then max overlap
            def _rank(el):
                pref = 1 if str(el.get("role")) in ("verified", "cross",
                                                    "question-mark") else 0
                return (pref, _box_iou(det["box"], el["box"]),
                        el["box"]["w"] * el["box"]["h"])
            group.sort(key=_rank, reverse=True)
            primary = group[0]
            primary["role"] = role
            primary["kind"] = "icon"
            meta = dict(primary.get("meta") or {})
            meta.update(meta_patch)
            # a matched fragment narrower than the detection adopts the detection
            # geometry so the rendered icon is the whole glyph, not one stroke
            if primary["box"]["w"] * primary["box"]["h"] < \
                    0.80 * det["box"]["w"] * det["box"]["h"]:
                primary["box"] = dict(det["box"])
                primary["area"] = float(det["mask"].sum())
                primary["coverage"] = round(primary["area"] /
                                            (canvas["w"] * canvas["h"]), 6)
                if primary.get("mask_path"):
                    _write_mask(det["mask"], det["box"], primary["mask_path"])
            absorbed = [el["id"] for el in group[1:]]
            for el in group[1:]:
                dropped_ids.add(el["id"])
            if absorbed:
                meta["absorbed_ids"] = sorted(set(meta.get("absorbed_ids", [])) |
                                              set(absorbed))
                report["absorbed"].extend(absorbed)
            primary["meta"] = meta
            report["matched"].append({"id": primary["id"], **entry})
        else:
            cid = f"E{_next_id(fused):03d}"
            rel = os.path.join("fused_elements", f"{cid}.png")
            path = os.path.join(run_dir, rel)
            _write_mask(det["mask"], det["box"], path)
            area = float(det["mask"].sum())
            fused.append({
                "id": cid, "meta": meta_patch, "box": dict(det["box"]),
                "kind": "icon", "role": role, "score": det["score"],
                "area": area, "coverage": round(area / (canvas["w"] * canvas["h"]), 6),
                "source": "icon-cv",
                "mask": {"kind": "alpha", "src": rel}, "mask_src": rel,
                "mask_path": os.path.abspath(path),
                "asset_src": None, "asset_candidates": [],
                "parent_id": None, "relationships": [],
                "provenance": {"sources": ["icon-cv"], "observations": [],
                               "nms": {"observation_count": 1, "merged_count": 0,
                                       "merges": []}},
            })
            report["added"].append({"id": cid, **entry})

    if dropped_ids:
        fused = [el for el in fused if el["id"] not in dropped_ids]
        for el in fused:
            if el.get("parent_id") in dropped_ids:
                el["parent_id"] = None
            rels = el.get("relationships")
            if rels:
                el["relationships"] = [r for r in rels
                                       if r.get("target") not in dropped_ids]

    # parent-link new icons into their containing card/shape so layout keeps rows local
    shells = [el for el in fused
              if el.get("kind") == "shape"
              and el["box"]["w"] * el["box"]["h"] < 0.95 * canvas["w"] * canvas["h"]]
    for el in fused:
        if el.get("source") == "icon-cv" and not el.get("parent_id"):
            hosts = [s for s in shells if _overlap_frac(el["box"], s["box"]) >= 0.9]
            if hosts:
                host = min(hosts, key=lambda s: s["box"]["w"] * s["box"]["h"])
                el["parent_id"] = host["id"]
                el["relationships"].append({"type": "nested-in", "target": host["id"],
                                            "containment": 1.0, "area_ratio": round(
                        el["box"]["w"] * el["box"]["h"] /
                        max(1.0, host["box"]["w"] * host["box"]["h"]), 4)})

    # brand lockups: small isolated two-tone text stacks → raster 'logo' proposals.
    # Detection always records evidence; element emission is gated (see lockup_emit).
    if opts.get("lockup_enabled", True):
        report["lockups"] = []
        emit = bool(opts.get("lockup_emit", False))
        for lk in detect_brand_lockups(rgb, lines, canvas, fused, opts):
            if not emit:
                report["lockups"].append({"id": None, "emitted": False, **lk})
                continue
            box = lk["box"]
            cid = f"E{_next_id(fused):03d}"
            rel = os.path.join("fused_elements", f"{cid}.png")
            full = np.zeros((canvas["h"], canvas["w"]), bool)
            full[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]] = True
            path = os.path.join(run_dir, rel)
            _write_mask(full, box, path)
            area = float(box["w"] * box["h"])
            fused.append({
                "id": cid,
                "meta": {"wordmark": True, "brand_lockup": True,
                         "lockup_text_ids": lk["text_ids"], "lockup_text": lk["text"],
                         "lockup_inks": lk["inks"], "raster_first": True,
                         "source": "icon-cv"},
                "box": dict(box), "kind": "icon", "role": "logo", "score": 0.62,
                "area": area, "coverage": round(area / (canvas["w"] * canvas["h"]), 6),
                "source": "icon-cv",
                "mask": {"kind": "alpha", "src": rel}, "mask_src": rel,
                "mask_path": os.path.abspath(path),
                "asset_src": None, "asset_candidates": [],
                "parent_id": None, "relationships": [],
                "provenance": {"sources": ["icon-cv"], "observations": [],
                               "nms": {"observation_count": 1, "merged_count": 0,
                                       "merges": []}},
            })
            report["lockups"].append({"id": cid, **lk})

    if opts.get("chart_enabled", True):
        chart = detect_chart_region(rgb, lines, canvas, opts)
        if chart:
            best, best_iou = None, 0.0
            for el in fused:
                role = str(el.get("role") or "").lower()
                if role not in _RASTERISH_CHART_ROLES:
                    continue
                iou = _box_iou(chart["box"], el["box"])
                if iou > best_iou:
                    best, best_iou = el, iou
            record = {"box": chart["box"], "gridlines": chart["gridlines"],
                      "spacing": chart["spacing"]}
            if best is not None and best_iou >= float(opts["chart_reroll_iou"]):
                best["role"] = "chart"
                meta = dict(best.get("meta") or {})
                meta["intentional_raster_cluster"] = True
                meta["chart"] = record
                best["meta"] = meta
                record = {**record, "element_id": best["id"], "iou": round(best_iou, 3),
                          "mode": "re-role"}
            else:
                cid = f"E{_next_id(fused):03d}"
                rel = os.path.join("fused_elements", f"{cid}.png")
                box = chart["box"]
                full = np.zeros((canvas["h"], canvas["w"]), bool)
                full[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]] = True
                path = os.path.join(run_dir, rel)
                _write_mask(full, box, path)
                area = float(box["w"] * box["h"])
                fused.append({
                    "id": cid,
                    "meta": {"intentional_raster_cluster": True, "chart": record},
                    "box": dict(box), "kind": "photo-fragment", "role": "chart",
                    "score": 0.6, "area": area,
                    "coverage": round(area / (canvas["w"] * canvas["h"]), 6),
                    "source": "icon-cv",
                    "mask": {"kind": "alpha", "src": rel}, "mask_src": rel,
                    "mask_path": os.path.abspath(path),
                    "asset_src": None, "asset_candidates": [],
                    "parent_id": None, "relationships": [],
                    "provenance": {"sources": ["icon-cv"], "observations": [],
                                   "nms": {"observation_count": 1, "merged_count": 0,
                                           "merges": []}},
                })
                record = {**record, "element_id": cid, "mode": "new"}
            report["chart"] = record

    for entry in report["detections"]:
        entry.pop("mask", None)
    try:
        with open(os.path.join(run_dir, "icon_detect.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return fused


if __name__ == "__main__":  # CPU-safe smoke: synthetic comparison list
    np = _np()
    cv2 = _cv2()
    img = np.full((400, 800, 3), 255, np.uint8)
    # green check chips on the left column, red crosses on the right
    for i in range(3):
        y = 80 + i * 70
        cv2.rectangle(img, (40, y), (70, y + 30), (60, 180, 60), -1)
        cv2.polylines(img, [np.asarray([(46, y + 16), (54, y + 24), (65, y + 6)],
                                       np.int32)], False, (255, 255, 255), 4)
        cv2.line(img, (440, y + 3), (465, y + 28), (40, 40, 220), 5)
        cv2.line(img, (465, y + 3), (440, y + 28), (40, 40, 220), 5)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = []
    for i in range(3):
        y = 80 + i * 70
        lines.append({"id": f"L{i}", "text": "left row", "box": {"x": 84, "y": y, "w": 200, "h": 30}})
        lines.append({"id": f"R{i}", "text": "right row", "box": {"x": 480, "y": y, "w": 200, "h": 30}})
    dets = detect_row_icons(rgb, lines, {"w": 800, "h": 400}, DEFAULTS)
    for d in dets:
        print(d["glyph"], d["score"], d["box"], d["row_text_id"])
