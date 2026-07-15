"""element_detect.py — stage 3: non-text, non-background elements via residual CCs.

Deterministic CPU port of the validated Node harness
`studio/lib/element-detect.mjs` (residual connected-component detector).

Algorithm (identical to the .mjs, plus optional adaptive threshold scaling):
  1. residual = |source - explained(background)| thresholded on luma AND chroma
     (thresholds and minArea adapt to residual noise / canvas area when
     opts["adaptive"] is on — see _adaptive_opts; defaults stay the anchors)
  2. mask OUT every OCR line box, dilated ~30% (15% each side)
  3. morphological close (radius 2 -> 5px window)
  4. 8-connected component labeling
  5. per-CC: drop area < 24 px^2; drop CC >=90% inside union of RAW OCR boxes
     (text remnants); drop CC >60% inside a declared photo region
  6. classify kind: 'shape' (<=3 dominant colors AND solidity>=0.45),
     else 'icon' (<4% canvas AND edge density>=0.12), else 'photo-fragment'
  7. nested consolidation: a CC whose bbox sits >=85% inside a larger NON-photo
     element folds into it (badge ring + inner mark -> one element)

`background` may be:
  - None                       -> flat median border color (honest fallback)
  - np.ndarray HxWx3 / HxWx4   -> a reconstructed background plate (resampled)
  - {'kind':'flat','color':[r,g,b]} -> flat fill (linear/radial best-effort flat)

Returns list[schema.Element] as dicts. When `run_dir` is provided, writes
`elements.json` and per-element box-local masks to `<run_dir>/elements/E<i>.png`
(255 = element) so merge/vectorize can pick them up by id convention.

Heavy deps (numpy, opencv-python, scipy) are imported lazily with a clear error.
"""
from __future__ import annotations
import importlib
import os
from typing import Optional


# ── sibling schema import (works as src.*, flat, or script) ─────────────────────────
def _load_schema():
    for name in ("src.schema", "schema"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("schema")


DEFAULTS = {
    "lumaThresh": 14.0,
    "chromaThresh": 20.0,
    "textDilate": 0.15,
    "closeRadius": 2,
    "minArea": 24,
    "photoInsideFrac": 0.6,
    "textOverlapFrac": 0.9,
    "shapeSolidity": 0.45,
    "shapeMaxColors": 3,
    "iconMaxCanvasFrac": 0.04,
    "iconEdgeDensity": 0.12,
    "edgeGradThresh": 50.0,
    "nestedInsideFrac": 0.85,
    "maxElements": 64,
    # ── adaptive scaling (see _adaptive_opts) ────────────────────────────────────────
    # The historical constants above remain the anchors; when `adaptive` is on they are
    # scaled by measured residual noise (contrast thresholds) and canvas area (minArea),
    # clamped so effective values never stray past the min/max scale bounds. At the
    # reference canvas (1080x1080) with reference noise the effective values equal the
    # defaults exactly.  Set "adaptive": False to restore fixed constants.
    "adaptive": True,
    "adaptiveRefSigma": 12.0,     # robust residual-luma sigma at which scale == 1.0
    "adaptiveScaleMin": 0.6,      # flat, clean backgrounds: lower bars, better small recall
    "adaptiveScaleMax": 1.6,      # noisy/photographic residuals: raise bars, less junk
    "adaptiveRefArea": 1166400,   # 1080 * 1080 normalized benchmark canvas
    "adaptiveMinAreaFloor": 8,    # px^2; never drop below this even on tiny canvases
    "adaptiveMinAreaCap": 4.0,    # x minArea; upper bound on canvas-area scaling
}


def _require_np():
    try:
        import numpy as np
        return np
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "element_detect requires numpy.  pip install numpy"
        ) from e


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "element_detect requires OpenCV.  pip install opencv-python"
        ) from e


def _require_ndimage():
    try:
        from scipy import ndimage
        return ndimage
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "element_detect requires SciPy.  pip install scipy"
        ) from e


def _inside_frac(a, b):
    """Fraction of box a's area that lies inside box b."""
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1, a["w"] * a["h"])


def estimate_border_color(rgb):
    """Per-channel median of a 2-px border ring (numpy HxWx3)."""
    np = _require_np()
    h, w = rgb.shape[:2]
    ring = min(2, h, w)
    top = rgb[:ring, :, :].reshape(-1, 3)
    bot = rgb[h - ring:, :, :].reshape(-1, 3)
    left = rgb[:, :ring, :].reshape(-1, 3)
    right = rgb[:, w - ring:, :].reshape(-1, 3)
    ring_px = np.concatenate([top, bot, left, right], axis=0)
    return np.median(ring_px, axis=0).astype(np.float64)


def _edge_gradient_magnitude(gray):
    """Edge-clamped gradient magnitude ``|dI/dx| + |dI/dy|`` for a 2-D array.

    Uses replicate-edge padding rather than circular wraparound (``np.roll``), so an
    outermost row/column pixel is differenced against its own edge neighbor instead of
    against the pixel at the OPPOSITE edge of the image. Circular wraparound skews
    edge_density -- and can misclassify -- border-touching elements.
    """
    np = _require_np()
    gray_x_pad = np.pad(gray, ((0, 0), (1, 1)), mode="edge")
    gray_y_pad = np.pad(gray, ((1, 1), (0, 0)), mode="edge")
    gx = np.abs(gray_x_pad[:, 2:] - gray_x_pad[:, :-2])
    gy = np.abs(gray_y_pad[2:, :] - gray_y_pad[:-2, :])
    return gx + gy


def _explained(rgb, background):
    """Return an HxWx3 float background estimate matching rgb's size."""
    np = _require_np()
    h, w = rgb.shape[:2]
    if background is not None and hasattr(background, "shape"):
        bg = background
        if bg.ndim == 3 and bg.shape[2] >= 3:
            bg = bg[:, :, :3]
        if bg.shape[0] == h and bg.shape[1] == w:
            return bg.astype(np.float64)
        # nearest resample to source dims
        cv2 = _require_cv2()
        return cv2.resize(
            bg.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float64)
    if isinstance(background, dict):
        kind = background.get("kind")
        if kind == "flat" and background.get("color") is not None:
            c = np.asarray(background["color"][:3], dtype=np.float64)
            return np.ones((h, w, 3), dtype=np.float64) * c
        # linear/radial best-effort: use mean stop color as a flat plate
        stops = background.get("stops")
        if stops:
            cols = np.asarray(
                [s.get("color", [0, 0, 0])[:3] for s in stops], dtype=np.float64
            )
            return np.ones((h, w, 3), dtype=np.float64) * cols.mean(axis=0)
    # None / 'photo' / unknown -> flat median border color
    c = estimate_border_color(rgb)
    return np.ones((h, w, 3), dtype=np.float64) * c


def _luma(arr):
    # arr: HxWx3 float -> HxW
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _adaptive_opts(opts, dY, n):
    """Return opts with contrast thresholds and minArea adapted to this image.

    * ``lumaThresh``/``chromaThresh`` scale with the robust (median/MAD) sigma of the
      residual luma difference.  A flat, clean background (sigma ~ 0, e.g. the dark X-post
      benchmark 009) lowers the bars toward ``adaptiveScaleMin`` so low-contrast small
      badges/checkmarks survive; a noisy or photographic residual raises them toward
      ``adaptiveScaleMax`` so junk CCs shrink.  MAD is used instead of std so real
      elements (a minority of pixels) do not inflate the noise estimate.
    * ``minArea`` scales linearly with canvas area relative to ``adaptiveRefArea``
      (24 px^2 means something very different at 1080p vs 4K), clamped to
      [adaptiveMinAreaFloor, minArea * adaptiveMinAreaCap].

    At the reference canvas and reference sigma the effective values equal the configured
    constants, so the historical defaults remain the calibration anchors.
    """
    np = _require_np()
    out = dict(opts)
    med = float(np.median(dY))
    sigma = 1.4826 * float(np.median(np.abs(dY - med)))
    ref_sigma = float(opts.get("adaptiveRefSigma", 12.0))
    lo = float(opts.get("adaptiveScaleMin", 0.6))
    hi = float(opts.get("adaptiveScaleMax", 1.6))
    scale = min(hi, max(lo, (sigma / ref_sigma) if ref_sigma > 0 else 1.0))
    out["lumaThresh"] = float(opts["lumaThresh"]) * scale
    out["chromaThresh"] = float(opts["chromaThresh"]) * scale
    area_scale = n / max(1.0, float(opts.get("adaptiveRefArea", 1166400)))
    min_area = int(round(float(opts["minArea"]) * area_scale))
    cap = int(round(float(opts["minArea"]) * float(opts.get("adaptiveMinAreaCap", 4.0))))
    out["minArea"] = max(int(opts.get("adaptiveMinAreaFloor", 8)), min(min_area, cap))
    out["_adaptive"] = {
        "noise_sigma": round(sigma, 3),
        "threshold_scale": round(scale, 3),
        "lumaThresh": round(out["lumaThresh"], 3),
        "chromaThresh": round(out["chromaThresh"], 3),
        "minArea": out["minArea"],
    }
    return out


def _line_boxes(ocr):
    """Extract axis boxes from a schema.OcrResult dict (or a list of boxes/lines)."""
    boxes = []
    lines = []
    if isinstance(ocr, dict):
        lines = ocr.get("lines", [])
    elif isinstance(ocr, list):
        lines = ocr
    for ln in lines or []:
        b = ln.get("box") if isinstance(ln, dict) else None
        if b is None and isinstance(ln, dict) and "x" in ln:
            b = ln
        if b and b.get("w", 0) > 0 and b.get("h", 0) > 0:
            boxes.append({"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]})
    return boxes


def detect(
    img_path,
    ocr,
    cfg: Optional[dict] = None,
    background=None,
    run_dir: Optional[str] = None,
):
    schema = _load_schema()
    np = _require_np()
    cv2 = _require_cv2()
    ndimage = _require_ndimage()

    cfg = cfg or {}
    opts = dict(DEFAULTS)
    opts.update(cfg.get("element_detect") or {})
    if run_dir is None:
        run_dir = cfg.get("run_dir")

    # decode source as RGB
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"element_detect: cannot read image {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    h, w = rgb.shape[:2]
    n = h * w
    if not n:
        return []

    explained = _explained(rgb, background)

    # 1. residual mask (luma + chroma)
    ys = _luma(rgb)
    ye = _luma(explained)
    dY = np.abs(ys - ye)
    dCr = np.abs((rgb[..., 0] - ys) - (explained[..., 0] - ye))
    dCb = np.abs((rgb[..., 2] - ys) - (explained[..., 2] - ye))
    if opts.get("adaptive", True):
        opts = _adaptive_opts(opts, dY, n)
    mask = ((dY > opts["lumaThresh"]) | (dCr + dCb > opts["chromaThresh"])).astype(
        np.uint8
    )
    gray = ys  # source luminance, reused for edge density

    # 2. mask OUT overlay text boxes, dilated ~30%
    line_boxes = _line_boxes(ocr)
    for b in line_boxes:
        dx = b["w"] * opts["textDilate"]
        dy = b["h"] * opts["textDilate"]
        x0 = max(0, int(np.floor(b["x"] - dx)))
        x1 = min(w, int(np.ceil(b["x"] + b["w"] + dx)))
        y0 = max(0, int(np.floor(b["y"] - dy)))
        y1 = min(h, int(np.ceil(b["y"] + b["h"] + dy)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0

    # 3. morphological close (radius r -> (2r+1) window)
    r = int(opts["closeRadius"])
    ksz = 2 * r + 1
    kernel = np.ones((ksz, ksz), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(closed, structure=structure)
    if not count:
        if run_dir:
            _write_artifacts(schema, [], run_dir)
        return []

    # raw (undilated) OCR + photo membership masks for the drop rules
    in_text = np.zeros((h, w), dtype=bool)
    for b in line_boxes:
        x0 = max(0, int(np.floor(b["x"])))
        x1 = min(w, int(np.ceil(b["x"] + b["w"])))
        y0 = max(0, int(np.floor(b["y"])))
        y1 = min(h, int(np.ceil(b["y"] + b["h"])))
        in_text[y0:y1, x0:x1] = True

    photo_regions = (cfg.get("element_detect") or {}).get("photoRegions") or cfg.get(
        "photoRegions"
    ) or []
    in_photo = np.zeros((h, w), dtype=bool)
    for b in photo_regions:
        if not (b.get("w", 0) > 0 and b.get("h", 0) > 0):
            continue
        x0 = max(0, int(np.floor(b["x"])))
        x1 = min(w, int(np.ceil(b["x"] + b["w"])))
        y0 = max(0, int(np.floor(b["y"])))
        y1 = min(h, int(np.ceil(b["y"] + b["h"])))
        in_photo[y0:y1, x0:x1] = True

    # 5. per-CC accumulation
    flat = labels.ravel()
    area = np.bincount(flat, minlength=count + 1)
    text_hit = np.bincount(flat, weights=in_text.ravel(), minlength=count + 1)
    photo_hit = np.bincount(flat, weights=in_photo.ravel(), minlength=count + 1)
    slices = ndimage.find_objects(labels)

    # precompute a padded gradient magnitude field for edge density (edge-clamped, see
    # _edge_gradient_magnitude docstring for why this must not use np.roll).
    grad = _edge_gradient_magnitude(gray)

    candidates = []
    for L in range(1, count + 1):
        a = int(area[L])
        if a < opts["minArea"]:
            continue
        if text_hit[L] / a >= opts["textOverlapFrac"]:
            continue  # OCR-CC rule: text remnant
        if photo_hit[L] / a > opts["photoInsideFrac"]:
            continue  # inside declared photo
        sl = slices[L - 1]
        if sl is None:
            continue
        ys_sl, xs_sl = sl
        bx, by = xs_sl.start, ys_sl.start
        bw = xs_sl.stop - xs_sl.start
        bh = ys_sl.stop - ys_sl.start
        sub = labels[ys_sl, xs_sl] == L  # bool mask, box-local
        solidity = a / float(bw * bh)

        # dominant colors: 3-bit/channel quantization, bins covering 85%
        sub_rgb = rgb[ys_sl, xs_sl][sub].astype(np.int64)
        q = (
            ((sub_rgb[:, 0] >> 5) << 6)
            | ((sub_rgb[:, 1] >> 5) << 3)
            | (sub_rgb[:, 2] >> 5)
        )
        counts = np.sort(np.bincount(q))[::-1]
        acc = 0
        dom_colors = 0
        thresh = 0.85 * a
        for c in counts:
            acc += int(c)
            dom_colors += 1
            if acc >= thresh:
                break

        # edge density over CC pixels
        sub_grad = grad[ys_sl, xs_sl][sub]
        edge_px = int(np.count_nonzero(sub_grad > opts["edgeGradThresh"]))
        edge_density = edge_px / a

        if dom_colors <= opts["shapeMaxColors"] and solidity >= opts["shapeSolidity"]:
            kind = "shape"
        elif a < opts["iconMaxCanvasFrac"] * n and edge_density >= opts["iconEdgeDensity"]:
            kind = "icon"
        else:
            kind = "photo-fragment"

        candidates.append(
            {
                "box": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
                "_mask": (sub.astype(np.uint8) * 255),
                "kind": kind,
                "area": a,
                "coverage": round(a / n, 4),
                "_stats": {
                    "solidity": round(solidity, 2),
                    "domColors": dom_colors,
                    "edgeDensity": round(edge_density, 2),
                },
            }
        )

    # 7. nested-CC consolidation (largest bbox first; photo hosts never absorb)
    candidates.sort(key=lambda c: c["box"]["w"] * c["box"]["h"], reverse=True)
    merged = []
    for c in candidates:
        host = next(
            (
                m
                for m in merged
                if m["kind"] != "photo-fragment"
                and _inside_frac(c["box"], m["box"]) >= opts["nestedInsideFrac"]
            ),
            None,
        )
        if host is None:
            merged.append(c)
            continue
        hb, cb = host["box"], c["box"]
        x0 = min(hb["x"], cb["x"])
        y0 = min(hb["y"], cb["y"])
        x1 = max(hb["x"] + hb["w"], cb["x"] + cb["w"])
        y1 = max(hb["y"] + hb["h"], cb["y"] + cb["h"])
        ub = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        um = np.zeros((ub["h"], ub["w"]), dtype=np.uint8)
        for part in (host, c):
            pb = part["box"]
            um[
                pb["y"] - y0 : pb["y"] - y0 + pb["h"],
                pb["x"] - x0 : pb["x"] - x0 + pb["w"],
            ] |= part["_mask"]
        host["box"] = ub
        host["_mask"] = um
        host["area"] += c["area"]
        host["coverage"] = round(host["area"] / n, 4)

    merged.sort(key=lambda c: c["area"], reverse=True)
    merged = merged[: opts["maxElements"]]

    elements = []
    masks = []
    for i, e in enumerate(merged):
        eid = f"E{i}"
        elements.append(
            {
                "id": eid,
                "box": e["box"],
                "kind": e["kind"],
                "area": float(e["area"]),
                "coverage": e["coverage"],
                "source": "residual-cc",
                "mask": os.path.join("elements", f"{eid}.png") if run_dir else None,
            }
        )
        masks.append((eid, e["_mask"]))

    if run_dir:
        _write_artifacts(schema, elements, run_dir, masks)
    return elements


def _write_artifacts(schema, elements, run_dir, masks=None):
    os.makedirs(run_dir, exist_ok=True)
    schema.dump(elements, os.path.join(run_dir, "elements.json"))
    if masks:
        try:
            from PIL import Image
            mdir = os.path.join(run_dir, "elements")
            os.makedirs(mdir, exist_ok=True)
            for eid, m in masks:
                Image.fromarray(m).save(os.path.join(mdir, f"{eid}.png"))
        except ImportError:
            pass  # masks are optional; elements.json still written


if __name__ == "__main__":  # CPU-safe smoke: synthetic ad
    import tempfile

    np = _require_np()
    cv2 = _require_cv2()
    img = np.full((400, 600, 3), 240, np.uint8)  # light gray bg
    cv2.rectangle(img, (40, 40), (160, 160), (30, 90, 200), -1)   # blue rect
    cv2.rectangle(img, (400, 220), (540, 340), (40, 180, 60), -1)  # green rect
    cv2.putText(img, "SALE", (250, 210), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (20, 20, 20), 6)
    tmp = tempfile.mkdtemp(prefix="eldetect_smoke_")
    p = os.path.join(tmp, "ad.png")
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ocr = {"lines": [{"box": {"x": 245, "y": 175, "w": 150, "h": 45}}]}
    els = detect(p, ocr, {}, run_dir=tmp)
    for e in els:
        print(e["id"], e["kind"], e["box"], "area", e["area"])
