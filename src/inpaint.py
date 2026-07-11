"""Build one final removal mask and inpaint the background exactly once.

The old pipeline kept the untouched input as its "background", guaranteeing that every
editable overlay appeared twice.  This module owns the opposite invariant:

    final entities -> one union mask -> one background plate

Heavy local backends are optional.  Big-LaMa (``simple-lama-inpainting``) is preferred on
the GPU workstation; OpenCV Telea is a deterministic fallback used by tests and on the Mac.
Generated pixels are composited only inside the requested mask so an inpainting model can
never alter the rest of the source image.
"""
from __future__ import annotations

import os
import shutil
from typing import Iterable, Optional


def _deps():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("inpaint requires numpy, pillow and opencv-python") from exc
    return cv2, np, Image


def resolve_path(path: Optional[str], run_dir: Optional[str] = None) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in (run_dir, os.getcwd()):
        if base:
            candidate = os.path.normpath(os.path.join(base, path))
            if os.path.exists(candidate):
                return candidate
    return None


def mask_on_canvas(mask_path: Optional[str], box: dict, canvas: tuple[int, int],
                   run_dir: Optional[str] = None):
    """Load a full-canvas or box-local mask into a uint8 canvas.

    Segmentation and Qwen cutouts are commonly stored as RGBA PNGs.  Their RGB pixels are
    deliberately undefined outside the cutout (and are often white), so using luminance there
    turns a transparent full-canvas cutout into a full-canvas removal mask.  Alpha is therefore
    authoritative whenever the source image has transparency; grayscale remains the fallback for
    normal mask files.
    """
    cv2, np, Image = _deps()
    width, height = canvas
    out = np.zeros((height, width), dtype=np.uint8)
    path = resolve_path(mask_path, run_dir)
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    w = max(1, int(round(box.get("w", 1))))
    h = max(1, int(round(box.get("h", 1))))

    if path:
        with Image.open(path) as image:
            has_alpha = "A" in image.getbands() or (
                image.mode == "P" and "transparency" in image.info
            )
            if has_alpha:
                arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3]
            else:
                arr = np.asarray(image.convert("L"), dtype=np.uint8)
        if arr.shape == (height, width):
            return arr
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        arr = np.full((h, w), 255, dtype=np.uint8)

    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x or y1 <= y:
        return out
    out[y:y1, x:x1] = arr[: y1 - y, : x1 - x]
    return out


def text_ink_mask(rgb, box: dict, quad: Optional[list] = None):
    """Estimate painted glyph pixels, avoiding a destructive whole OCR rectangle.

    Border pixels estimate the local plate.  Pixels whose RGB distance from that plate is
    above an Otsu threshold are kept, then constrained by the OCR polygon.  The fallback is
    the polygon itself when the crop has too little contrast.
    """
    cv2, np, _ = _deps()
    height, width = rgb.shape[:2]
    x0 = max(0, int(box.get("x", 0)))
    y0 = max(0, int(box.get("y", 0)))
    x1 = min(width, int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(height, int(round(box.get("y", 0) + box.get("h", 0))))
    out = np.zeros((height, width), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return out
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    border = np.concatenate(
        [crop[:1].reshape(-1, 3), crop[-1:].reshape(-1, 3),
         crop[:, :1].reshape(-1, 3), crop[:, -1:].reshape(-1, 3)], axis=0
    )
    plate = np.median(border, axis=0)
    distance = np.sqrt(((crop - plate) ** 2).sum(axis=2))
    dist_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    threshold, local = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can select noise on textured photos. Require a meaningful contrast floor.
    local = ((distance >= max(18.0, float(threshold))) * 255).astype(np.uint8)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    coverage = float(np.count_nonzero(local)) / max(1, local.size)
    if coverage < 0.01 or coverage > 0.72:
        local[:] = 255
    out[y0:y1, x0:x1] = local

    if quad and len(quad) >= 4:
        polygon = np.zeros_like(out)
        pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in quad], np.int32)
        cv2.fillPoly(polygon, [pts], 255)
        out = cv2.bitwise_and(out, polygon)
    return out


def build_union_mask(canvas: tuple[int, int], observations: Iterable[dict],
                     run_dir: Optional[str] = None, default_dilate: int = 2):
    """Return one uint8 union mask from canonical (already deduplicated) entities."""
    cv2, np, _ = _deps()
    width, height = canvas
    union = np.zeros((height, width), dtype=np.uint8)
    for item in observations:
        if item.get("keep_in_background") or item.get("is_background"):
            continue
        box = item.get("box") or {}
        path = item.get("mask_path") or item.get("mask_src")
        if not path and isinstance(item.get("mask"), dict):
            path = item["mask"].get("src")
        mask = item.get("mask_array")
        if mask is None:
            mask = mask_on_canvas(path, box, canvas, run_dir)
        radius = int(item.get("dilate", default_dilate))
        if radius > 0:
            k = 2 * radius + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        union = cv2.bitwise_or(union, mask.astype(np.uint8))
    return union


def _opencv_inpaint(rgb, mask, radius: int = 5, method=None):
    cv2, np, _ = _deps()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_NS if str(method or "telea").lower() in ("ns", "navier-stokes", "navier_stokes") else cv2.INPAINT_TELEA
    result = cv2.inpaint(bgr, mask.astype(np.uint8), max(1, radius), flag)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _seam_energy(candidate, mask):
    """Lower is better: penalize artificial high-frequency seams around a filled hole."""
    cv2, np, _ = _deps()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return 0.0
    # A narrow band straddling the old boundary is where Telea/NS failures are visible.
    kernel = np.ones((3, 3), np.uint8)
    outer = cv2.dilate(binary, kernel, iterations=1)
    inner = cv2.erode(binary, kernel, iterations=1)
    band = (outer > 0) & (inner == 0)
    if not band.any():
        band = binary > 0
    image = np.asarray(candidate, dtype=np.float32)
    gray = image[:, :, 0] * .2126 + image[:, :, 1] * .7152 + image[:, :, 2] * .0722
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    # Colour variance catches chroma blotches that a grayscale-only gradient misses.
    local = image[band]
    chroma = float(local.std(axis=0).mean()) if local.size else 0.0
    return float(gradient[band].mean()) + .20 * chroma


def _opencv_auto(rgb, mask, radius):
    """Pick the less seam-prone deterministic OpenCV fill for the fallback path."""
    telea = _opencv_inpaint(rgb, mask, radius, "telea")
    ns = _opencv_inpaint(rgb, mask, radius, "ns")
    telea_score = _seam_energy(telea, mask)
    ns_score = _seam_energy(ns, mask)
    if ns_score + 1e-6 < telea_score:
        return ns, "opencv-ns", {"telea_seam": round(telea_score, 4), "ns_seam": round(ns_score, 4)}
    return telea, "opencv-telea", {"telea_seam": round(telea_score, 4), "ns_seam": round(ns_score, 4)}


def _simple_lama(rgb, mask):
    _, np, Image = _deps()
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise ImportError(
            "Big-LaMa backend requires simple-lama-inpainting; install it in the GPU env"
        ) from exc
    model = _simple_lama.__dict__.get("_model")
    if model is None:
        model = SimpleLama()
        _simple_lama.__dict__["_model"] = model
    result = model(Image.fromarray(rgb), Image.fromarray(mask).convert("L"))
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


def inpaint_array(rgb, mask, cfg: Optional[dict] = None, return_diagnostics: bool = False):
    """Inpaint RGB pixels and guarantee that pixels outside ``mask`` stay byte-identical."""
    _, np, _ = _deps()
    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    mode = str(icfg.get("mode", "auto")).lower()
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not np.any(mask):
        result = (np.asarray(rgb, dtype=np.uint8).copy(), "none", {})
        return result if return_diagnostics else result[:2]

    generated = None
    used = mode
    diagnostics = {}
    if mode in ("auto", "lama", "big-lama", "simple-lama"):
        try:
            generated = _simple_lama(np.asarray(rgb, dtype=np.uint8), mask)
            used = "big-lama"
        except Exception as exc:
            if mode != "auto" and not icfg.get("allow_fallback", True):
                raise
            print(f"[inpaint] Big-LaMa unavailable ({exc}); using OpenCV fallback")
    if generated is None:
        radius = int(icfg.get("opencv_radius", 5))
        fallback_method = str(icfg.get("opencv_method", "auto" if mode == "auto" else "telea")).lower()
        if fallback_method in ("auto", "best"):
            generated, used, diagnostics = _opencv_auto(np.asarray(rgb, dtype=np.uint8), mask, radius)
        else:
            generated = _opencv_inpaint(np.asarray(rgb, dtype=np.uint8), mask, radius, fallback_method)
            used = "opencv-ns" if fallback_method in ("ns", "navier-stokes", "navier_stokes") else "opencv-telea"

    original = np.asarray(rgb, dtype=np.uint8)
    out = original.copy()
    selected = mask > 0
    out[selected] = np.asarray(generated, dtype=np.uint8)[selected]
    result = (out, used, diagnostics)
    return result if return_diagnostics else result[:2]


def inpaint_once(image_path: str, mask, output_path: str, cfg: Optional[dict] = None) -> dict:
    """Create the canonical clean background artifact."""
    _, np, Image = _deps()
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if isinstance(mask, str):
        mask_arr = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)
    else:
        mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.shape != rgb.shape[:2]:
        raise ValueError(f"inpaint mask {mask_arr.shape} does not match image {rgb.shape[:2]}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not np.any(mask_arr):
        shutil.copyfile(image_path, output_path)
        return {"ok": True, "path": output_path, "backend": "none", "masked_fraction": 0.0}
    out, backend, diagnostics = inpaint_array(rgb, mask_arr, cfg, return_diagnostics=True)
    Image.fromarray(out).save(output_path)
    return {
        "ok": True,
        "path": output_path,
        "backend": backend,
        "masked_fraction": round(float(np.count_nonzero(mask_arr)) / mask_arr.size, 6),
        "diagnostics": diagnostics,
    }
