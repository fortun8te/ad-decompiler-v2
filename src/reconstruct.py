"""Materialize canonical assets and a duplicate-free background plate.

This is the first stage that turns detections into pixels with ownership.  It resolves all
run-relative paths, removes duplicate observations, extracts alpha crops, routes simple
graphics through the vector fidelity gate, samples native shape fills, and sends one final
union mask to :mod:`src.inpaint`.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from . import inpaint, vectorize
from .schema import dump


def _deps():
    import cv2
    import numpy as np
    from PIL import Image
    return cv2, np, Image


def _iou(a, b):
    ix = max(0.0, min(a.get("x", 0) + a.get("w", 0), b.get("x", 0) + b.get("w", 0))
             - max(a.get("x", 0), b.get("x", 0)))
    iy = max(0.0, min(a.get("y", 0) + a.get("h", 0), b.get("y", 0) + b.get("h", 0))
             - max(a.get("y", 0), b.get("y", 0)))
    inter = ix * iy
    union = a.get("w", 0) * a.get("h", 0) + b.get("w", 0) * b.get("h", 0) - inter
    return inter / union if union > 0 else 0.0


def _confidence(candidate):
    return float((candidate.get("meta") or {}).get("confidence") or candidate.get("score") or 0)


def _source_priority(candidate):
    source = str((candidate.get("meta") or {}).get("source") or candidate.get("source") or "")
    if "sam3" in source:
        return 4
    if "element+qwen" in source:
        return 3
    if "element" in source:
        return 2
    if "qwen" in source:
        return 1
    return 0


def _is_background_plate(candidate, width, height):
    box = candidate.get("box") or {}
    area_frac = box.get("w", 0) * box.get("h", 0) / max(1, width * height)
    role = str((candidate.get("meta") or {}).get("role") or "")
    tolerance_x, tolerance_y = width * .025, height * .025
    touches = sum((
        box.get("x", 0) <= tolerance_x,
        box.get("y", 0) <= tolerance_y,
        box.get("x", 0) + box.get("w", 0) >= width - tolerance_x,
        box.get("y", 0) + box.get("h", 0) >= height - tolerance_y,
    ))
    return (role == "background" or area_frac > .92 or
            (role in ("photo", "illustration", "image") and area_frac > .58 and touches >= 3))


def deduplicate(candidates: list, threshold: float = 0.86):
    """Drop same-object observations while preserving nested, semantically different layers."""
    ordered = sorted((dict(c) for c in candidates),
                     key=lambda c: (_source_priority(c), _confidence(c)), reverse=True)
    kept = []
    for candidate in ordered:
        if candidate.get("target") == "drop":
            kept.append(candidate)
            continue
        role = (candidate.get("meta") or {}).get("role") or candidate.get("kind")
        duplicate = False
        for other in kept:
            if other.get("target") == "drop":
                continue
            other_role = (other.get("meta") or {}).get("role") or other.get("kind")
            # Text and its backing button/shape are intentionally nested, not duplicates.
            if {candidate.get("target"), other.get("target")} == {"text", "shape"}:
                continue
            if role != other_role and candidate.get("target") != other.get("target"):
                continue
            if _iou(candidate.get("box", {}), other.get("box", {})) >= threshold:
                other.setdefault("meta", {}).setdefault("merged_observations", []).append(
                    candidate.get("id")
                )
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    # Preserve the upstream paint order after selecting winners.
    order = {c.get("id"): i for i, c in enumerate(candidates)}
    return sorted(kept, key=lambda c: order.get(c.get("id"), 10**9))


def _mask_path(candidate):
    mask = candidate.get("mask")
    if isinstance(mask, dict):
        return mask.get("src")
    if isinstance(mask, str):
        return mask
    return candidate.get("mask_path")


def _candidate_mask(candidate, rgb, run_dir, ocr_lines=None):
    _, np, _ = _deps()
    h, w = rgb.shape[:2]
    meta = candidate.get("meta") or {}
    if candidate.get("target") == "text" or (candidate.get("text") and meta.get("wordmark")):
        members = [ocr_lines[line_id] for line_id in meta.get("line_ids", [])
                   if ocr_lines and line_id in ocr_lines]
        if members:
            combined = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for line in members:
                combined = np.maximum(combined, inpaint.text_ink_mask(
                    rgb, line.get("painted_box") or line.get("box", {}), line.get("quad")
                ))
            return combined
        return inpaint.text_ink_mask(
            rgb,
            candidate.get("visible_box") or candidate.get("ink_box") or candidate.get("box", {}),
            candidate.get("quad") or meta.get("quad"),
        )
    return inpaint.mask_on_canvas(_mask_path(candidate), candidate.get("box", {}), (w, h), run_dir)


def _crop_rgba(rgb, mask, box):
    _, np, Image = _deps()
    h, w = rgb.shape[:2]
    x0 = max(0, int(round(box.get("x", 0))))
    y0 = max(0, int(round(box.get("y", 0))))
    x1 = min(w, int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(h, int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    rgba = np.dstack([rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
    return Image.fromarray(rgba.astype(np.uint8))


def _source_rgba(candidate, rgb, mask, run_dir):
    """Prefer a model-provided clean RGBA layer, correctly cropped to its tight box."""
    _, np, Image = _deps()
    path = inpaint.resolve_path(candidate.get("src"), run_dir)
    box = candidate.get("box", {})
    if path:
        image = Image.open(path).convert("RGBA")
        canvas_h, canvas_w = rgb.shape[:2]
        if image.size == (canvas_w, canvas_h):
            x = max(0, int(round(box.get("x", 0))))
            y = max(0, int(round(box.get("y", 0))))
            w = max(1, int(round(box.get("w", 1))))
            h = max(1, int(round(box.get("h", 1))))
            return image.crop((x, y, min(canvas_w, x + w), min(canvas_h, y + h)))
        target = (max(1, int(round(box.get("w", image.width)))),
                  max(1, int(round(box.get("h", image.height)))))
        return image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
    return _crop_rgba(rgb, mask, box)


def _apply_owned_alpha(image, owned_mask, box):
    """Ensure a foreground pixel is present in at most one exported raster asset."""
    _, np, Image = _deps()
    x0 = max(0, int(round(box.get("x", 0))))
    y0 = max(0, int(round(box.get("y", 0))))
    x1 = min(owned_mask.shape[1], int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(owned_mask.shape[0], int(round(box.get("y", 0) + box.get("h", 0))))
    local = (owned_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    if local.size == 0:
        return image
    if (local.shape[1], local.shape[0]) != image.size:
        local = np.asarray(Image.fromarray(local).resize(image.size, Image.Resampling.NEAREST))
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], local)
    return Image.fromarray(rgba)


def _dominant_fill(rgb, mask, box):
    _, np, _ = _deps()
    x0, y0 = max(0, int(box.get("x", 0))), max(0, int(box.get("y", 0)))
    x1 = min(rgb.shape[1], int(box.get("x", 0) + box.get("w", 0)))
    y1 = min(rgb.shape[0], int(box.get("y", 0) + box.get("h", 0)))
    pixels = rgb[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0]
    if pixels.size == 0:
        return "#cccccc"
    quant = (pixels.astype(np.uint16) // 8) * 8
    colors, counts = np.unique(quant.reshape(-1, 3), axis=0, return_counts=True)
    color = colors[int(counts.argmax())]
    return "#%02x%02x%02x" % tuple(int(v) for v in color)


def _hex(color):
    """Turn a sampled RGB triplet into the paint spelling used by design.json."""
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(value)))) for value in color)


def _local_shape_pixels(rgb, mask, box):
    """Return the image and a binary, tight shape mask for native-paint analysis.

    Segmentation edges are normally anti-aliased.  Treating any non-zero value as shape
    keeps the analysis stable for both SAM masks and source-alpha masks, while the later
    erosion prevents those edge pixels from polluting the sampled fill.
    """
    _, np, _ = _deps()
    x0, y0 = max(0, int(round(box.get("x", 0)))), max(0, int(round(box.get("y", 0))))
    x1 = min(rgb.shape[1], int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(rgb.shape[0], int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return rgb[:0, :0], np.zeros((0, 0), dtype=bool)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 16


def _corner_radius(local_mask):
    """Infer an axis-aligned rounded-rectangle radius from its four clipped corners.

    This intentionally returns ``None`` for a noisy/partial mask.  A wrong native radius
    is worse than a rectangular fallback because it visibly bends otherwise straight art.
    """
    _, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 8:
        return None
    h, w = local_mask.shape
    if float(local_mask.mean()) < .62:
        return None

    def first_true(values):
        hit = np.flatnonzero(values)
        return int(hit[0]) if hit.size else None

    pairs = [
        (first_true(local_mask[0, :]), first_true(local_mask[:, 0])),
        (first_true(local_mask[0, ::-1]), first_true(local_mask[:, -1])),
        (first_true(local_mask[-1, ::-1]), first_true(local_mask[::-1, -1])),
        (first_true(local_mask[-1, :]), first_true(local_mask[::-1, 0])),
    ]
    radii = []
    max_radius = min(h, w) * .48
    for horizontal, vertical in pairs:
        if horizontal is None or vertical is None:
            return None
        # A real quarter-circle has the same first occupied distance on both edges.
        if abs(horizontal - vertical) > max(2, min(h, w) * .08):
            return None
        radius = (horizontal + vertical) / 2
        if radius < 1.25 or radius > max_radius:
            radii.append(0.0)
        else:
            radii.append(radius)
    if not any(radii):
        return 0
    nonzero = [value for value in radii if value > 0]
    if len(nonzero) < 2:
        return None
    # Equal corners are common and should compile to Figma's simple scalar radius.
    if max(nonzero) - min(nonzero) <= max(1.5, min(h, w) * .04):
        return round(float(np.median(nonzero)), 2)
    names = ("topLeft", "topRight", "bottomRight", "bottomLeft")
    return {name: round(value, 2) for name, value in zip(names, radii)}


def _simple_shape_geometry(local_mask):
    """Return rect/ellipse only where the segmentation really supports a primitive."""
    _, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 4:
        return None
    fill = float(local_mask.mean())
    h, w = local_mask.shape
    aspect = w / max(1, h)
    corners = sum(bool(value) for value in (
        local_mask[0, 0], local_mask[0, -1], local_mask[-1, 0], local_mask[-1, -1]
    ))
    # Keep the existing ellipse heuristic, but do not call arbitrary sparse SAM masks rects.
    if .75 <= aspect <= 1.33 and corners <= 1 and .55 <= fill <= .90:
        return "ellipse"
    if fill >= .70:
        return "rect"
    return None


def _robust_color(pixels, fallback=(204, 204, 204)):
    _, np, _ = _deps()
    if pixels is None or not len(pixels):
        return np.asarray(fallback, dtype=np.float32)
    # Median is much less likely than a mean to absorb antialiasing or a few specular pixels.
    return np.median(np.asarray(pixels, dtype=np.float32), axis=0)


def _gradient_fill(local_rgb, interior, min_range=18.0, min_r2=.86):
    """Fit a two-stop linear paint when the interior is genuinely explained by a plane.

    Decorative photos can be colourful too.  The R² gate, high quantile colour range and
    primitive-only caller together make this deliberately conservative.
    """
    _, np, _ = _deps()
    ys, xs = np.nonzero(interior)
    if len(xs) < 80:
        return None
    h, w = interior.shape
    x = (xs.astype(np.float32) - (w - 1) / 2) / max(1.0, (w - 1) / 2)
    y = (ys.astype(np.float32) - (h - 1) / 2) / max(1.0, (h - 1) / 2)
    colors = local_rgb[ys, xs].astype(np.float32)
    # Subsample huge surfaces deterministically; it avoids a 4K panel dominating runtime.
    if len(colors) > 12000:
        pick = np.linspace(0, len(colors) - 1, 12000).astype(int)
        x, y, colors = x[pick], y[pick], colors[pick]
    spread = np.percentile(colors, 95, axis=0) - np.percentile(colors, 5, axis=0)
    if float(np.linalg.norm(spread)) < min_range:
        return None
    design = np.column_stack((np.ones(len(x)), x, y))
    coefficients, _, _, _ = np.linalg.lstsq(design, colors, rcond=None)
    prediction = design @ coefficients
    total = float(np.square(colors - colors.mean(axis=0)).sum())
    if total <= 1e-6:
        return None
    r2 = 1 - float(np.square(colors - prediction).sum()) / total
    if r2 < min_r2:
        return None
    # PCA turns a three-channel plane into a deterministic visual direction.
    _, _, vh = np.linalg.svd(colors - colors.mean(axis=0), full_matrices=False)
    principal = vh[0]
    dx, dy = float(coefficients[1] @ principal), float(coefficients[2] @ principal)
    magnitude = (dx * dx + dy * dy) ** .5
    if magnitude < .5:
        return None
    dx, dy = dx / magnitude, dy / magnitude
    projection = x * dx + y * dy
    low, high = np.percentile(projection, (2, 98))
    if high - low < .25:
        return None
    endpoint = lambda value: coefficients[0] + coefficients[1] * (value * dx) + coefficients[2] * (value * dy)
    # The Figma/compiler convention is 0° left->right and positive angles go down.
    import math
    return {
        "kind": "linear",
        "angle": round(math.degrees(math.atan2(dy, dx)), 2),
        "stops": [
            {"position": 0, "color": _hex(endpoint(low))},
            {"position": 1, "color": _hex(endpoint(high))},
        ],
        "meta": {"r2": round(r2, 4), "range": round(float(np.linalg.norm(spread)), 2)},
    }


def _stroke_and_interior(local_rgb, local_mask, max_width=8):
    """Detect a coherent inset stroke and return (stroke, safe_fill_pixels_mask).

    A gradient has different colours at opposite edges; it therefore fails the coherent
    border gate instead of being mislabelled as a stroke.
    """
    cv2, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 10:
        return None, local_mask
    distance = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 3)
    min_side = min(local_mask.shape)
    band = max(1, min(max_width, int(round(min_side * .12))))
    # Learn the paint from the first 1-2 pixels only.  Sampling a possible 7px band
    # would include the fill itself and make a perfectly normal 3px border look incoherent.
    probe_width = min(2, band)
    ring = local_mask & (distance > .2) & (distance <= probe_width)
    core = local_mask & (distance >= max(3, band + 1))
    if ring.sum() < 24 or core.sum() < 20:
        return None, local_mask
    edge = _robust_color(local_rgb[ring])
    edge_dist = np.linalg.norm(local_rgb.astype(np.float32) - edge, axis=2)
    coherent = ring & (edge_dist <= 14)
    if coherent.sum() / max(1, ring.sum()) < .78:
        return None, local_mask
    interior_color = _robust_color(local_rgb[core])
    if float(np.linalg.norm(edge - interior_color)) < 20:
        return None, local_mask
    width = 0
    for candidate_width in range(1, band + 1):
        candidate_ring = local_mask & (distance > .2) & (distance <= candidate_width)
        if candidate_ring.sum() and (coherent & candidate_ring).sum() / candidate_ring.sum() >= .78:
            width = candidate_width
    if not width:
        return None, local_mask
    safe_interior = local_mask & (distance >= width + 1)
    return {"color": _hex(edge), "width": int(width), "align": "INSIDE"}, safe_interior


def _shadow_effect(rgb, mask, box, geometry):
    """Find a modest drop shadow only against an otherwise flat surrounding field.

    The flat-background gate is crucial: a neighbouring photo edge should stay in the clean
    plate, not turn into a made-up Figma shadow.
    """
    cv2, np, _ = _deps()
    if geometry not in ("rect", "ellipse"):
        return None
    x, y, w, h = (int(round(box.get(key, 0))) for key in ("x", "y", "w", "h"))
    if w < 12 or h < 12:
        return None
    pad = max(5, min(18, int(round(min(w, h) * .28))))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
    if x1 - x0 < w + 4 or y1 - y0 < h + 4:
        return None
    crop = rgb[y0:y1, x0:x1]
    shape = mask[y0:y1, x0:x1] > 16
    # The outermost two-pixel border supplies the local background estimate.
    outer = np.zeros(shape.shape, dtype=bool)
    edge = min(2, max(1, min(shape.shape) // 8))
    outer[:edge, :] = outer[-edge:, :] = True
    outer[:, :edge] = outer[:, -edge:] = True
    samples = crop[outer & ~shape]
    if len(samples) < 20:
        return None
    background = _robust_color(samples)
    if float(np.max(np.std(samples.astype(np.float32), axis=0))) > 7.5:
        return None
    difference = np.linalg.norm(crop.astype(np.float32) - background, axis=2)
    best = None
    for dy in range(-pad // 2, pad // 2 + 1):
        for dx in range(-pad // 2, pad // 2 + 1):
            if dx == dy == 0:
                continue
            translation = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(shape.astype(np.uint8), translation, (shape.shape[1], shape.shape[0]),
                                     flags=cv2.INTER_NEAREST) > 0
            halo = shifted & ~shape
            count = int(halo.sum())
            if count < max(12, (w + h) // 3):
                continue
            response = float(np.mean(difference[halo]))
            # Blurred shadow should be visible but softer than a separate hard object.
            score = response * min(1.0, count / max(1, w + h))
            if response >= 9 and (best is None or score > best[0]):
                best = (score, response, dx, dy, halo)
    if best is None:
        return None
    _, response, dx, dy, halo = best
    halo_color = _robust_color(crop[halo])
    # Only accept a shadow-like halo that moves toward neutral/darker colour from its field.
    if float(np.linalg.norm(halo_color - background)) < 9:
        return None
    opacity = max(.12, min(.72, response / 255 * 1.55))
    return {
        "type": "drop-shadow", "color": _hex(halo_color), "opacity": round(opacity, 3),
        "x": int(dx), "y": int(dy), "radius": max(2, int(round(min(pad, max(abs(dx), abs(dy)) * 1.8 + 2)))),
    }


def _extract_shape_style(rgb, mask, box, cfg):
    """Conservative native-style extraction for semantic primitive candidates."""
    _, np, _ = _deps()
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    geometry = _simple_shape_geometry(local_mask)
    if geometry is None:
        return None
    style_cfg = ((cfg.get("reconstruct") or {}).get("style_extraction") or {})
    stroke, interior = _stroke_and_interior(
        local_rgb, local_mask, int(style_cfg.get("max_stroke_width", 8))
    )
    gradient = _gradient_fill(
        local_rgb, interior,
        float(style_cfg.get("gradient_min_range", 18)),
        float(style_cfg.get("gradient_min_r2", .86)),
    )
    fill_color = _robust_color(local_rgb[interior])
    fill = gradient or {"kind": "flat", "color": _hex(fill_color)}
    radius = _corner_radius(local_mask) if geometry == "rect" else None
    effect = _shadow_effect(rgb, mask, box, geometry) if style_cfg.get("detect_shadows", True) else None
    return {
        "shape_kind": geometry,
        "fill": fill,
        "stroke": stroke,
        "radius": radius,
        "effects": [effect] if effect else [],
        "meta": {
            "geometry": geometry,
            "gradient": gradient.get("meta") if gradient else None,
            "stroke_detected": bool(stroke),
            "shadow_detected": bool(effect),
        },
    }


def _infer_shape(mask, box):
    _, np, _ = _deps()
    x0, y0 = max(0, int(box.get("x", 0))), max(0, int(box.get("y", 0)))
    x1 = min(mask.shape[1], int(box.get("x", 0) + box.get("w", 0)))
    y1 = min(mask.shape[0], int(box.get("y", 0) + box.get("h", 0)))
    local = mask[y0:y1, x0:x1] > 0
    if local.size == 0:
        return "rect", 0
    fill = float(local.mean())
    corners = [local[0, 0], local[0, -1], local[-1, 0], local[-1, -1]]
    aspect = local.shape[1] / max(1, local.shape[0])
    if 0.75 <= aspect <= 1.33 and sum(bool(x) for x in corners) <= 1 and 0.55 <= fill <= 0.88:
        return "ellipse", min(local.shape) / 2
    # Missing corner pixels on an otherwise solid region indicate a rounded rectangle.
    radius = min(local.shape) * 0.12 if fill > 0.75 and sum(bool(x) for x in corners) < 4 else 0
    return "rect", round(radius, 2)


def _paths_to_svg(paths, width, height):
    body = []
    for path in paths:
        fill = path.get("fill") or "#000000"
        winding = path.get("windingRule") or "nonzero"
        body.append(f'<path d="{path.get("d", "")}" fill="{fill}" fill-rule="{winding}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>")


def _write_asset(image, assets_dir, candidate_id):
    raw = image.tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:10]
    name = f"{candidate_id}_{digest}.png"
    path = os.path.join(assets_dir, name)
    image.save(path)
    return os.path.join("assets", name)


def reconstruct(image_path: str, ocr: dict, candidates: list, run_dir: str,
                cfg: Optional[dict] = None) -> dict:
    cv2, np, Image = _deps()
    cfg = cfg or {}
    rcfg = cfg.get("reconstruct") or {}
    os.makedirs(run_dir, exist_ok=True)
    assets_dir = os.path.join(run_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    canonical = deduplicate(candidates, float(rcfg.get("dedup_iou", 0.86)))
    ocr_lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    masks = {}
    for candidate in canonical:
        if candidate.get("target") != "drop":
            masks[candidate.get("id")] = _candidate_mask(candidate, rgb, run_dir, ocr_lines)

    # Front-to-back ownership is diagnostic and makes overlapping raster assets exclusive.
    # Text/icons are frontmost; smaller nested layers win over broad photo regions.
    # A background plate becomes no foreground layer at all.  Excluding it *before*
    # ownership matters: otherwise a large Qwen/background observation can claim every pixel
    # and leave the real product/icon cutout with an empty alpha channel.
    front = sorted(
        (c for c in canonical if c.get("target") != "drop" and not (
            c.get("target") == "image" and _is_background_plate(c, w, h)
        )),
        key=lambda c: (
            3 if c.get("target") == "text" else 2 if c.get("target") == "icon" else 1,
            float(c.get("z", 0)),
            -c.get("box", {}).get("w", 0) * c.get("box", {}).get("h", 0),
        ), reverse=True,
    )
    ownership = np.zeros((h, w), dtype=np.uint16)
    owner_index = {}
    owner_number = {}
    for index, candidate in enumerate(front, start=1):
        cid = candidate.get("id")
        owner_index[str(index)] = cid
        owner_number[cid] = index
        available = (masks[cid] > 0) & (ownership == 0)
        ownership[available] = index

    # Materialize native shapes, vectors, and isolated alpha rasters.
    updated = []
    vector_ok = vector_fallback = 0
    for candidate in canonical:
        c = dict(candidate)
        c["meta"] = dict(c.get("meta") or {})
        target = c.get("target")
        cid = c.get("id")
        is_plate = _is_background_plate(c, w, h)
        if is_plate and target == "image":
            c["target"] = "drop"
            c["meta"]["keep_in_background"] = True
            updated.append(c)
            continue
        if target in ("drop", "text"):
            updated.append(c)
            continue
        mask = masks.get(cid)
        if mask is None:
            updated.append(c)
            continue
        if target == "shape":
            # Do not overwrite upstream paint facts.  This fills only the gaps left by
            # segmentation/Qwen and tags every inference for later QA/debugging.
            extracted = _extract_shape_style(rgb, mask, c.get("box", {}), cfg)
            if extracted:
                c["shape_kind"] = c.get("shape_kind") or extracted["shape_kind"]
                c["fill"] = c.get("fill") or extracted["fill"]
                c["stroke"] = c.get("stroke") or extracted["stroke"]
                if not c.get("effects") and extracted["effects"]:
                    c["effects"] = extracted["effects"]
                if c.get("radius") is None and extracted["radius"] not in (None, 0):
                    c["radius"] = extracted["radius"]
                c["meta"]["style_extraction"] = extracted["meta"]
            else:
                kind, radius = _infer_shape(mask, c.get("box", {}))
                c["shape_kind"] = c.get("shape_kind") or kind
                c["fill"] = c.get("fill") or {
                    "kind": "flat", "color": _dominant_fill(rgb, mask, c.get("box", {}))
                }
                if radius and kind == "rect" and c.get("radius") is None:
                    c["radius"] = radius
            updated.append(c)
            continue

        image = _source_rgba(c, rgb, mask, run_dir)
        owned = (ownership == owner_number.get(cid, 0)).astype(np.uint8) * 255
        image = _apply_owned_alpha(image, owned, c.get("box", {}))
        if target == "icon":
            traced = vectorize.vectorize_crop(np.asarray(image), cfg)
            c["meta"]["vectorize"] = {
                k: traced.get(k) for k in ("ok", "engine", "score", "note")
            }
            if traced.get("ok"):
                c["paths"] = traced["paths"]
                c["svg"] = traced.get("svg") or _paths_to_svg(traced["paths"], image.width, image.height)
                c["fill"] = {"kind": "flat", "color": traced["paths"][0].get("fill", "#000000")}
                vector_ok += 1
                updated.append(c)
                continue
            c["target"] = "image"
            c["meta"]["vector_fallback"] = True
            vector_fallback += 1

        c["src"] = _write_asset(image, assets_dir, cid)
        c["mask"] = {"kind": "alpha"}
        updated.append(c)

    removal = []
    for c in updated:
        if c.get("target") == "drop":
            continue
        box = c.get("box", {})
        area_frac = box.get("w", 0) * box.get("h", 0) / max(1, w * h)
        # A full-canvas raster is the plate itself. Everything else is removed from the plate.
        is_background = bool(c.get("meta", {}).get("role") == "background" or area_frac > 0.92)
        removal.append({
            "box": box,
            "mask_array": masks.get(c.get("id")),
            "is_background": is_background,
            "dilate": 1 if c.get("target") == "text" else int(rcfg.get("mask_dilate", 2)),
        })
    union = inpaint.build_union_mask((w, h), removal, run_dir, default_dilate=2)
    mask_path = os.path.join(run_dir, "removal_mask.png")
    Image.fromarray(union).save(mask_path)
    background_path = os.path.join(run_dir, "background_clean.png")
    inpaint_result = inpaint.inpaint_once(image_path, union, background_path, cfg)

    # Visual ownership map plus a machine-readable legend.
    ownership_path = os.path.join(run_dir, "ownership.png")
    scale = max(1, 65535 // max(1, len(front)))
    Image.fromarray((ownership * scale).astype(np.uint16)).save(ownership_path)
    result = {
        "schema_version": 2,
        "background": "background_clean.png",
        "removal_mask": "removal_mask.png",
        "ownership": "ownership.png",
        "owner_index": owner_index,
        "candidates": updated,
        "stats": {
            "input_candidates": len(candidates),
            "canonical_entities": len(updated),
            "duplicates_removed": len(candidates) - len(updated),
            "vectorized": vector_ok,
            "vector_fallback": vector_fallback,
            "inpaint": inpaint_result,
        },
    }
    dump(result, os.path.join(run_dir, "reconstruction.json"))
    return result
