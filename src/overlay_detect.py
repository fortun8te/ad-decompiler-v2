"""overlay_detect.py — the ROUNDED-PLATE OVERLAY family (classic CV).

Given a normalized RGB image plus the existing element/text detections, this module
proposes the "solid rounded plate over busy content" overlays that the pipeline could
not previously recognise on photographic backgrounds:

  * rounded-rect pills  (H1 — muted-green pills with white emoji+text)
  * stadium rows        (H5 — dark ✓/✗ list rows, radius == height/2)
  * banner strips       (H3 — full-width muted-green strip with white text)
  * stacked cards       (H4 — white IG rectangular text cards)

The detector is deliberately classic-CV only (flat-colour region + rounded-corner
fitting + text-box containment). No models. It proves out on synthetic composites with
known geometry (tests/test_overlay_detect.py) before ever reaching for a VLM.

CONTRACT (emission side, see docs/OVERLAY-ELEMENTS.md):
    overlay = native SOLID rect (shape_kind="rect" + cornerRadius + flat fill)
            + native TEXT child(ren)
            + emoji rendered as inline IMAGE chips (letterSpacing == 0)
    stadium row = rect(radius = height/2) + icon chip + TEXT
    banner      = full-width rect + TEXT

`detect_overlays(...)` returns plate proposals (pure geometry + colour + containment).
`emit_overlay_group(...)` turns one proposal + its resolved text/emoji candidates into a
`target="group"` candidate dict that build_design_json.build() compiles directly. All
coordinates are ABSOLUTE source pixels; layout._relativize handles parent-relative
rewriting downstream, exactly like every other grouped candidate.
"""
from __future__ import annotations

from typing import Any, Optional


# ── numpy / cv2 are optional at import time (CPU-only, skips cleanly in CI) ──────────
def _deps():
    import numpy as np
    try:
        import cv2
    except Exception:  # pragma: no cover - exercised only where cv2 is absent
        cv2 = None
    return np, cv2


# ── tunables ────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    # A flat plate colour must cover at least this fraction of the frame to be a plate.
    "min_area_frac": 0.004,
    # ...but never larger than this (that is a background, not an overlay plate).
    "max_area_frac": 0.75,
    # Per-channel tolerance when growing a flat-colour mask around a quantized colour.
    "color_tol": 20,
    # A solid plate fills most of its own bounding box (rounded corners nip the rest).
    "min_fill_ratio": 0.80,
    # Interior colour uniformity: fraction of interior pixels within color_tol of median.
    "min_interior_uniform": 0.82,
    # A banner spans (almost) the full canvas width.
    "banner_width_frac": 0.90,
    # A plate must contain at least this much text-box overlap to be a *text* plate.
    # Banners/pills carry copy; a bare decorative panel is left to element_detect.
    "require_text": True,
    # Corner-fit acceptance (fraction of quadrant pixels the radius model explains).
    "corner_fit_min": 0.93,
    # Quantization step for the palette pass.
    "quant_step": 24,
}

CORNER_NAMES = ("topLeft", "topRight", "bottomRight", "bottomLeft")


# ── geometry helpers ─────────────────────────────────────────────────────────────────
def _as_rgb_u8(rgb):
    np, _ = _deps()
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3]


def _box_of(mask):
    np, _ = _deps()
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def _hex(color):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(v)))) for v in color[:3])


def _fit_quarter_radius(quadrant):
    """Best-fit rounded-corner radius for a corner quadrant oriented corner-at-(0,0).

    One-parameter model fitted over the whole quadrant (robust to edge debris). Returns
    ``(radius, match_fraction)``. Mirrors reconstruct._fit_quarter_radius so the emitted
    radius agrees with the rest of the pipeline, but is self-contained here.
    """
    np, cv2 = _deps()
    actual = quadrant > 0
    size = actual.shape[0]
    if size == 0:
        return 0.0, 0.0
    scale = 1.0
    if size > 64 and cv2 is not None:
        scale = size / 64.0
        actual = cv2.resize(actual.astype(np.uint8), (64, 64),
                            interpolation=cv2.INTER_NEAREST) > 0
        size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    best_r, best_m = 0.0, -1.0
    for radius in range(size + 1):
        inside = (xx >= radius) | (yy >= radius) | (
            (xx - radius) ** 2 + (yy - radius) ** 2 <= radius * radius
        )
        match = float((inside == actual).mean())
        if match > best_m:
            best_r, best_m = float(radius), match
    return best_r * scale, best_m


def estimate_corner_radius(local_mask, opts=None):
    """Infer an axis-aligned rounded-rect radius (scalar), a per-corner dict, or None.

    A cap that rounds through the whole half-side is a stadium/pill end and snaps to
    the exact pill radius (min(h,w)/2). Returns None when the mask does not support a
    clean rounded-rect model (a noisy plate stays a plain rect, which is safer).
    """
    np, _ = _deps()
    o = dict(DEFAULTS)
    o.update(opts or {})
    if local_mask.size == 0 or min(local_mask.shape) < 8:
        return None
    h, w = local_mask.shape
    if float(local_mask.mean()) < 0.55:
        return None
    quad = min(h, w) // 2
    if quad < 4:
        return None
    quadrants = (
        local_mask[:quad, :quad],
        local_mask[:quad, ::-1][:, :quad],
        local_mask[::-1, ::-1][:quad, :quad],
        local_mask[::-1, :][:quad, :quad],
    )
    fits = [_fit_quarter_radius(p) for p in quadrants]
    if any(m < float(o["corner_fit_min"]) for _, m in fits):
        return None
    pill_gate = quad - max(2.0, quad * 0.12)
    radii = []
    for radius, _ in fits:
        if radius < 1.25:
            radii.append(0.0)
        elif radius >= pill_gate:
            radii.append(min(h, w) / 2.0)
        else:
            radii.append(radius)
    nonzero = [v for v in radii if v > 0]
    if len(nonzero) < 2:
        return 0.0 if not nonzero else None
    if max(nonzero) - min(nonzero) <= max(1.5, min(h, w) * 0.05):
        return round(float(np.median(nonzero)), 2)
    return {name: round(v, 2) for name, v in zip(CORNER_NAMES, radii)}


# ── flat-region proposal ───────────────────────────────────────────────────────────
def _line_boxes(text_lines):
    boxes = []
    for ln in text_lines or []:
        b = (ln or {}).get("box") or (ln or {}).get("ink_box")
        if not b:
            continue
        try:
            boxes.append((str(ln.get("id") or f"L{len(boxes)}"),
                          {k: float(b.get(k, 0)) for k in ("x", "y", "w", "h")}))
        except (TypeError, ValueError):
            continue
    return boxes


def _contained_text_ids(plate_box, line_boxes, min_cover=0.6):
    """Text ids whose box is >= min_cover contained inside the plate box."""
    out = []
    px, py = plate_box["x"], plate_box["y"]
    pw, ph = plate_box["w"], plate_box["h"]
    for lid, b in line_boxes:
        ix = max(0.0, min(px + pw, b["x"] + b["w"]) - max(px, b["x"]))
        iy = max(0.0, min(py + ph, b["y"] + b["h"]) - max(py, b["y"]))
        inter = ix * iy
        area = max(1.0, b["w"] * b["h"])
        if inter / area >= min_cover:
            out.append(lid)
    return out


def _classify(plate_box, radius, canvas_w, opts):
    """Geometric taxonomy of a rounded plate.

    banner  — spans (almost) the full canvas width (H3 muted-green strip).
    stadium — a capsule: radius == height/2 and clearly wider than tall (H5 list rows).
    pill    — a wide, short rounded-rect bar (H1 emoji+text pills; aspect >= ~2.2).
    card    — a blockier rounded-rect panel (H4 IG cards; aspect < ~2.2).

    The pill/card split is aspect-based: both are rounded rects, but H1 pills are wide
    short bars while IG cards are larger, squarer blocks. Semantic fill/context can
    refine this downstream; geometry alone gives a defensible default.
    """
    h, w = plate_box["h"], plate_box["w"]
    scalar_r = radius if isinstance(radius, (int, float)) else None
    if w >= float(opts["banner_width_frac"]) * canvas_w:
        return "banner"
    is_capsule = scalar_r is not None and abs(scalar_r - h / 2.0) <= max(2.0, h * 0.10)
    if is_capsule and w > h * 1.6:
        return "stadium"
    aspect = w / max(1.0, h)
    if aspect >= 2.2:
        return "pill"
    return "card"


def detect_overlays(rgb, elements=None, text_lines=None, canvas=None, cfg=None):
    """Propose rounded-plate overlays from a normalized image + existing detections.

    Parameters
    ----------
    rgb : HxWx3 array (uint8 or float; RGB)
    elements : list of element dicts (element_detect output) — advisory, may be None
    text_lines : list of OCR line dicts {id, box|ink_box, text} — used for containment
    canvas : {"w","h"} — defaults to image shape
    cfg : optional dict; cfg["overlay_detect"] overrides DEFAULTS

    Returns a list of plate proposals sorted top-to-bottom, each:
        {
          "id": "OV0",
          "bbox": {x,y,w,h},           # absolute source pixels
          "kind": "pill"|"stadium"|"banner"|"card",
          "corner_radius": float | {topLeft,...} | None,
          "fill": "#rrggbb",
          "text_ids": [...],           # contained OCR line ids
          "z_order": float,            # above background, below its own text
          "fill_ratio": float,
          "source": "overlay-cv",
        }
    """
    np, cv2 = _deps()
    if cv2 is None:
        return []
    opts = dict(DEFAULTS)
    opts.update((cfg or {}).get("overlay_detect") or {})
    arr = _as_rgb_u8(rgb)
    H, W = arr.shape[:2]
    if canvas is None:
        canvas = {"w": W, "h": H}
    n = H * W
    min_area = int(float(opts["min_area_frac"]) * n)
    max_area = int(float(opts["max_area_frac"]) * n)
    tol = float(opts["color_tol"])
    line_boxes = _line_boxes(text_lines)

    # Palette pass: quantize, take each populated colour, grow a tolerance mask, and
    # test each connected component for "solid rounded rect".
    step = int(opts["quant_step"])
    quant = (arr.astype(np.int16) // step) * step + step // 2
    flat = quant.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]

    proposals = []
    seen_masks = np.zeros((H, W), dtype=bool)
    for idx in order:
        color = colors[idx]
        if counts[idx] < min_area:
            break  # counts are sorted desc; nothing smaller qualifies
        mask = (np.abs(arr.astype(np.int16) - color.astype(np.int16)).max(axis=2) <= tol)
        if mask.sum() < min_area:
            continue
        mask_u8 = (mask.astype(np.uint8)) * 255
        # close small gaps (anti-aliased text sitting inside the plate punches holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for comp in range(1, num):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x = int(stats[comp, cv2.CC_STAT_LEFT])
            y = int(stats[comp, cv2.CC_STAT_TOP])
            w = int(stats[comp, cv2.CC_STAT_WIDTH])
            h = int(stats[comp, cv2.CC_STAT_HEIGHT])
            if w < 12 or h < 10:
                continue
            # A flat region touching all four image borders is the page background
            # (H5 sage bg), not an overlay plate — skip it. A full-width banner touches
            # only two borders and survives.
            if x <= 1 and y <= 1 and (x + w) >= (W - 1) and (y + h) >= (H - 1):
                continue
            # The closed mask bridged text gaps for grouping, but it dilated corners and
            # the bbox. Recover the TRUE silhouette from the raw colour mask, restricted
            # to this component, then re-tighten the box so radius/geometry are exact.
            comp_region = labels[y:y + h, x:x + w] == comp
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            def _clean(sil):
                # Shed 1-2px anti-aliased protrusions so corner quadrants fit cleanly.
                return cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, open_k) > 0

            raw_sil = _clean(mask[y:y + h, x:x + w] & comp_region)
            tight = _box_of(raw_sil)
            if tight is not None:
                x, y = x + tight["x"], y + tight["y"]
                w, h = tight["w"], tight["h"]
                comp_region = labels[y:y + h, x:x + w] == comp
                raw_sil = _clean(mask[y:y + h, x:x + w] & comp_region)
            # fill ratio measures how much of the box the plate fills, counting bridged
            # text holes as filled (they belong to the plate), so use the closed region.
            comp_mask = comp_region
            fill_ratio = float(comp_mask.mean())
            if fill_ratio < float(opts["min_fill_ratio"]):
                continue
            # avoid re-proposing an overlapping region from a near-duplicate colour bin
            if seen_masks[y:y + h, x:x + w][comp_mask].mean() > 0.5:
                continue
            plate_box = {"x": x, "y": y, "w": w, "h": h}
            radius = estimate_corner_radius(raw_sil, opts)
            # interior colour + uniform fill sample (erode to dodge the AA rim/text)
            er = cv2.erode(comp_mask.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            interior = arr[y:y + h, x:x + w][er > 0]
            if interior.size == 0:
                interior = arr[y:y + h, x:x + w][comp_mask]
            median = np.median(interior, axis=0)
            uniform = float(
                (np.abs(interior.astype(np.int16) - median.astype(np.int16)).max(axis=1)
                 <= tol).mean()
            )
            if uniform < float(opts["min_interior_uniform"]):
                continue
            text_ids = _contained_text_ids(plate_box, line_boxes)
            if opts["require_text"] and not text_ids:
                # keep a full-width banner even if OCR missed its copy, else drop
                if not (w >= float(opts["banner_width_frac"]) * canvas["w"]):
                    continue
            kind = _classify(plate_box, radius, canvas["w"], opts)
            proposals.append({
                "id": f"OV{len(proposals)}",
                "bbox": plate_box,
                "kind": kind,
                "corner_radius": radius,
                "fill": _hex(median),
                "text_ids": text_ids,
                "z_order": 20.0,
                "fill_ratio": round(fill_ratio, 3),
                "interior_uniform": round(uniform, 3),
                "source": "overlay-cv",
            })
            seen_masks[y:y + h, x:x + w] |= comp_mask

    proposals.sort(key=lambda p: (p["bbox"]["y"], p["bbox"]["x"]))
    for i, p in enumerate(proposals):
        p["id"] = f"OV{i}"
    return proposals


# ── emission (candidate dicts for build_design_json.build) ──────────────────────────
def _rect_child(plate, run_z):
    """The SOLID rounded-rect surface of the plate (native, editable)."""
    box = dict(plate["bbox"])
    fill = {"kind": "flat", "color": plate["fill"]}
    child = {
        "id": f"{plate['id']}__plate",
        "target": "shape",
        "shape_kind": "rect",
        "box": box,
        "fill": fill,
        "z_index": run_z,
        "meta": {"role": "overlay-plate", "overlay_kind": plate["kind"],
                 "z": run_z, "source": "overlay-cv"},
    }
    cr = plate.get("corner_radius")
    if cr is not None:
        child["radius"] = cr
    return child


def emit_overlay_group(plate, texts=None, emojis=None, icons=None):
    """Compile one plate proposal into a ``target="group"`` candidate.

    Parameters
    ----------
    plate : a proposal dict from detect_overlays
    texts : list of text candidate dicts to nest (already resolved by OCR/text stages).
            Each should be a normal text candidate: {id,target:"text",text,box,style,...}.
    emojis : list of emoji IMAGE-chip candidate dicts (contract: emoji = image chips,
             positioned inline, letterSpacing == 0 on the sibling TEXT). Each:
             {id,target:"image",src,box,meta:{role:"emoji"}}.
    icons : list of icon chip candidate dicts (stadium ✓/✗). {id,target:"icon"/"image",...}

    Returns a group candidate in ABSOLUTE coordinates. layout._relativize will convert
    children to parent-relative space at the normal stage boundary; build_design_json
    then compiles the group -> Figma FRAME with a native RECT + native TEXT + chips.
    """
    base_z = float(plate.get("z_order", 20.0))
    children = [_rect_child(plate, base_z)]
    for i, t in enumerate(texts or []):
        t = dict(t)
        t.setdefault("target", "text")
        t.setdefault("id", f"{plate['id']}__t{i}")
        t["z_index"] = base_z + 2.0
        st = dict(t.get("style") or {})
        st["letterSpacing"] = 0.0  # contract: chips carry spacing, text stays untracked
        t["style"] = st
        meta = dict(t.get("meta") or {})
        meta.setdefault("role", "overlay-text")
        meta["z"] = base_z + 2.0
        t["meta"] = meta
        children.append(t)
    for i, e in enumerate(emojis or []):
        e = dict(e)
        e.setdefault("target", "image")
        e.setdefault("id", f"{plate['id']}__emoji{i}")
        e["z_index"] = base_z + 3.0
        meta = dict(e.get("meta") or {})
        meta.setdefault("role", "emoji")
        meta["z"] = base_z + 3.0
        meta["emoji_chip"] = True
        e["meta"] = meta
        children.append(e)
    for i, ic in enumerate(icons or []):
        ic = dict(ic)
        ic.setdefault("target", ic.get("target") or "icon")
        ic.setdefault("id", f"{plate['id']}__icon{i}")
        ic["z_index"] = base_z + 3.0
        meta = dict(ic.get("meta") or {})
        meta.setdefault("role", "icon")
        meta["z"] = base_z + 3.0
        ic["meta"] = meta
        children.append(ic)

    role = {
        "pill": "overlay-pill", "stadium": "stadium-row",
        "banner": "banner", "card": "overlay-card",
    }.get(plate["kind"], "overlay")
    return {
        "id": plate["id"],
        "target": "group",
        "box": dict(plate["bbox"]),
        "z_index": base_z,
        "meta": {"role": role, "overlay_kind": plate["kind"], "z": base_z,
                 "source": "overlay-cv", "corner_radius": plate.get("corner_radius"),
                 "text_ids": list(plate.get("text_ids") or [])},
        "children": children,
    }


def emit_all(plates, texts_by_id=None, emojis_by_plate=None, icons_by_plate=None):
    """Convenience: emit a group candidate for every proposal.

    ``texts_by_id`` maps an OCR line id -> its resolved text candidate dict. Each plate's
    ``text_ids`` selects which texts it nests. ``emojis_by_plate``/``icons_by_plate`` map
    a plate id -> list of chip candidates. Any missing mapping is simply skipped.
    """
    texts_by_id = texts_by_id or {}
    emojis_by_plate = emojis_by_plate or {}
    icons_by_plate = icons_by_plate or {}
    out = []
    for plate in plates:
        texts = [texts_by_id[i] for i in plate.get("text_ids", []) if i in texts_by_id]
        out.append(emit_overlay_group(
            plate, texts=texts,
            emojis=emojis_by_plate.get(plate["id"]),
            icons=icons_by_plate.get(plate["id"]),
        ))
    return out
