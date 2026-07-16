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
import time
from typing import Iterable, Optional

try:  # Supports both ``src.inpaint`` and the legacy bare ``inpaint`` import.
    from .inpaint_quality import candidate_metrics
except ImportError:  # pragma: no cover - exercised only by direct module invocation
    from inpaint_quality import candidate_metrics


def _deps():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("inpaint requires numpy, pillow and opencv-python") from exc
    return cv2, np, Image


def _load_qwen_worker():
    """Import the ComfyUI client module regardless of whether we're imported as
    ``src.inpaint`` or a bare ``inpaint``.  Returns ``None`` if it cannot be found."""
    import importlib
    for name in ("src.qwen_worker", "qwen_worker"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _flux_comfy_inpaint(rgb, mask, cfg: Optional[dict] = None):
    """Run the Flux Fill ComfyUI backend, returning an HxWx3 uint8 plate or ``None``.

    Thin seam over ``qwen_worker.flux_inpaint`` so tests can monkeypatch the whole GPU
    backend and so an offline/absent ComfyUI degrades to Big-LaMa/OpenCV.
    """
    worker = _load_qwen_worker()
    if worker is None or not hasattr(worker, "flux_inpaint"):
        return None
    return worker.flux_inpaint(rgb, mask, cfg)


def _powerpaint_adapter_status(cfg: Optional[dict] = None) -> dict:
    """Report only whether an optional user-supplied PowerPaint adapter is importable.

    Importability is deliberately *not* reported as model readiness: the adapter may still
    need model weights, a CUDA runtime, or its own server.  The first actual inpaint call is
    the only honest runtime validation available without adding a GPU package to this project.
    """
    import importlib.util

    powerpaint = ((cfg or {}).get("inpaint") or {}).get("powerpaint") or {}
    module = str(powerpaint.get("adapter_module") or "").strip()
    callable_name = str(powerpaint.get("callable") or "inpaint").strip()
    enabled = bool(powerpaint.get("enabled", False))
    if not module:
        return {
            "configured": False, "importable": False, "runtime_validated": False,
            "detail": "adapter_module not configured", "adapter_module": "", "callable": callable_name,
        }
    try:
        importable = importlib.util.find_spec(module) is not None
        detail = ("adapter importable; model/device not validated" if importable
                  else "adapter module not installed")
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError) as exc:
        importable, detail = False, str(exc)
    return {
        "configured": enabled, "importable": importable, "runtime_validated": False,
        "detail": detail, "adapter_module": module, "callable": callable_name,
    }


def _powerpaint_inpaint(rgb, mask, cfg: Optional[dict] = None):
    """Call an optional local PowerPaint adapter without importing GPU dependencies here.

    The adapter contract is ``callable(rgb_uint8, mask_uint8, cfg) -> RGB image | None``.
    This intentionally provides a thin seam only; it neither vendors PowerPaint nor pretends
    that a Mac has validated a CUDA/model install.
    """
    import importlib

    status = _powerpaint_adapter_status(cfg)
    if not status["configured"]:
        return None
    if not status["importable"]:
        return None
    module = importlib.import_module(status["adapter_module"])
    adapter = getattr(module, status["callable"], None)
    if not callable(adapter):
        raise RuntimeError(
            f"PowerPaint adapter {status['adapter_module']!r} has no callable {status['callable']!r}"
        )
    return adapter(rgb, mask, cfg or {})


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


def box_fill_mask(canvas: tuple[int, int], box: dict, pad: int = 2):
    """Solid rectangle mask for a layer box, used to complete sparse text ink masks."""
    _, np, _ = _deps()
    height, width = canvas
    out = np.zeros((height, width), dtype=np.uint8)
    x0 = max(0, int(round(box.get("x", 0))) - pad)
    y0 = max(0, int(round(box.get("y", 0))) - pad)
    x1 = min(width, int(round(box.get("x", 0) + box.get("w", 0))) + pad)
    y1 = min(height, int(round(box.get("y", 0) + box.get("h", 0))) + pad)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def text_ink_mask(rgb, box: dict, quad: Optional[list] = None,
                  allow_box_fallback: bool = True):
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
    # OCR boxes are often glyph-tight: their first/last rows can be mostly black cap
    # strokes rather than plate. Estimate the plate from a small exterior collar while
    # still constraining the removal mask to the authored OCR box. Otherwise the white
    # plate is selected as "ink" and the ghost-text guard paints a solid black slab.
    collar = max(2, min(6, int(round(min(x1 - x0, y1 - y0) * 0.10))))
    sx0, sy0 = max(0, x0 - collar), max(0, y0 - collar)
    sx1, sy1 = min(width, x1 + collar), min(height, y1 + collar)
    sample = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    border = np.concatenate(
        [sample[:1].reshape(-1, 3), sample[-1:].reshape(-1, 3),
         sample[:, :1].reshape(-1, 3), sample[:, -1:].reshape(-1, 3)], axis=0
    )
    plate = np.median(border, axis=0)
    distance = np.sqrt(((crop - plate) ** 2).sum(axis=2))
    dist_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    threshold, local = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can select noise on textured photos. Require a meaningful contrast floor.
    local = ((distance >= max(18.0, float(threshold))) * 255).astype(np.uint8)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    coverage = float(np.count_nonzero(local)) / max(1, local.size)
    if coverage < 0.01:
        # Overlay text must fail closed.  A box-sized hole is materially worse than
        # leaving uncertain pixels in the background plate.
        local[:] = 255 if allow_box_fallback else 0
    elif coverage > 0.72:
        # A textured/gradient plate can make the first-pass contrast mask nearly full.
        # Replacing it with the OCR rectangle is worse: dilation then creates the solid
        # rectangular holes seen on body copy.  Tighten the contrast instead and retain
        # the measured geometry; only use the rectangle for the genuinely low-contrast
        # case where there is no useful ink signal at all.
        strict = ((distance >= max(32.0, float(threshold) * 1.5)) * 255).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if np.count_nonzero(strict):
            local = strict
        elif not allow_box_fallback:
            local[:] = 0

    if not allow_box_fallback and np.any(local):
        # Keep only compact ink components from the current candidate crop.  This
        # prevents a low-frequency plate gradient from becoming one connected block.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (local > 0).astype(np.uint8), connectivity=8
        )
        components = np.zeros_like(local)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= 2:
                components[labels == label] = 255
        local = components
    out[y0:y1, x0:x1] = local

    if quad and len(quad) >= 4:
        polygon = np.zeros_like(out)
        pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in quad], np.int32)
        cv2.fillPoly(polygon, [pts], 255)
        out = cv2.bitwise_and(out, polygon)
    return out


def solidify_mask(mask, threshold: int = 16):
    """Turn soft segmentation alpha into a binary removal footprint."""
    _, np, _ = _deps()
    return np.where(np.asarray(mask) > threshold, 255, 0).astype(np.uint8)


def fill_enclosed_mask_holes(mask, threshold: int = 16):
    """Fill only transparent islands enclosed by a segmentation matte.

    This keeps the exterior transparent while preventing a SAM void from
    punching a visible hole through a product/person raster layer.  It is
    intentionally opt-in at the reconstruction callsite; icons may have
    legitimate counters.
    """
    cv2, np, _ = _deps()
    binary = solidify_mask(mask, threshold)
    padded = np.pad(binary, 1, mode="constant")
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    enclosed = (padded == 0) & (flood == 0)
    padded[enclosed] = 255
    return padded[1:-1, 1:-1].astype(np.uint8)


def feather_mask_edges(mask, radius: int = 1):
    """Soften only the outer rim of a binary mask — helps LaMa blend without widening the hole."""
    cv2, np, _ = _deps()
    radius = max(0, int(radius))
    binary = solidify_mask(mask)
    if radius <= 0 or not np.any(binary):
        return binary
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.erode(binary, kernel, iterations=radius)
    boundary = cv2.subtract(binary, core)
    k = 2 * radius + 1
    soft_rim = cv2.GaussianBlur(boundary, (k, k), 0)
    return np.clip(np.maximum(core, soft_rim), 0, 255).astype(np.uint8)


def comfyui_healthy(cfg: Optional[dict] = None, probe=None) -> bool:
    """True when the configured ComfyUI backend answers /system_stats.

  On the RTX workstation ComfyUI liveness is a practical proxy for "GPU box is up".
  Mac/unit-test runs pass ``probe`` to avoid real HTTP.
    """
    cfg = cfg or {}
    qwen = cfg.get("qwen") or {}
    if not qwen.get("enabled", True):
        return True
    if str(qwen.get("mode", "comfyui")).lower() != "comfyui":
        return True
    base = str(cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
    if probe is not None:
        return bool(probe(f"{base}/system_stats"))
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/system_stats", timeout=0.5) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _big_lama_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("simple_lama_inpainting") is not None
    except Exception:
        return False


def check_backends(cfg: Optional[dict] = None) -> dict:
    """Return JSON-safe availability for the inpainting backends.

    This is intentionally a cheap import/API probe; model construction is deferred to
    the first real inpaint call.  ``ready`` means the configured active backend can run,
    while ``fallback_ready`` describes the deterministic OpenCV alternative.
    """
    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    mode = str(icfg.get("mode", "auto")).lower()
    strict_acceptance = bool(icfg.get("strict_acceptance", False))
    try:
        import cv2  # noqa: F401
        opencv_ok, opencv_detail = True, "opencv-python importable"
    except Exception as exc:
        opencv_ok, opencv_detail = False, str(exc)
    lama_ok = _big_lama_available()
    powerpaint = _powerpaint_adapter_status(cfg)
    if mode in ("auto", "lama", "big-lama", "simple-lama"):
        active = lama_ok
    elif mode in ("opencv", "cv2"):
        active = opencv_ok
    elif mode in ("powerpaint", "power-paint"):
        # An adapter can be importable while its GPU/weights are unavailable.  Do not lie.
        active = False
    else:
        active = False
    return {
        "mode": mode,
        "strict_acceptance": strict_acceptance,
        "big_lama": {"ok": lama_ok, "detail": "simple-lama-inpainting importable" if lama_ok else "not installed"},
        "opencv": {"ok": opencv_ok, "detail": opencv_detail},
        "powerpaint": powerpaint,
        "ready": active,
        "fallback_ready": opencv_ok,
        "fallback_permitted": opencv_ok and (not strict_acceptance or mode in ("opencv", "cv2")),
    }


def _mask_fraction(mask) -> float:
    _, np, _ = _deps()
    arr = np.asarray(mask)
    return float(np.count_nonzero(arr)) / max(1, arr.size)


def resolve_mask_dilate(candidate: dict, cfg: Optional[dict] = None) -> int:
    """Per-target inpaint dilation radius in pixels.

    ``inpaint.mask_dilate`` may be a scalar or a mapping with keys such as
    ``default``, ``button``, ``shape``, ``text``, ``photo``, and ``image``.
    Falls back to ``reconstruct.mask_dilate`` when unset.
    """
    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    rcfg = cfg.get("reconstruct") or {}
    mcfg = icfg.get("mask_dilate")
    legacy = int(rcfg.get("mask_dilate", 2))

    target = str(candidate.get("target") or "image")
    role = str((candidate.get("meta") or {}).get("role") or "")

    if isinstance(mcfg, dict):
        default = int(mcfg.get("default", legacy))
        if target == "text":
            # Overlay copy is re-drawn as editable text.  Give it an explicit
            # knob so the halo can be covered without widening every OCR mask.
            if (candidate.get("meta") or {}).get("overlay_text"):
                return int(mcfg.get("overlay_text", mcfg.get("text", default)))
            return int(mcfg.get("text", default))
        if target == "shape":
            if role in ("button", "badge", "chip", "card"):
                return int(mcfg.get("button", mcfg.get("shape", default)))
            return int(mcfg.get("shape", default))
        if target == "icon":
            return int(mcfg.get("icon", default))
        if target == "image":
            if role in ("product", "person", "photo"):
                return int(mcfg.get("photo", mcfg.get("image", default)))
            return int(mcfg.get("image", default))
        return default

    base = int(mcfg) if isinstance(mcfg, (int, float)) else legacy
    if target == "text":
        if mcfg is not None:
            return base
        # OCR boxes contain painted glyphs only approximately.  The editable text
        # is rendered over the plate, so a one-pixel halo is still a duplicate.  Use
        # a larger role-aware footprint for overlay text while keeping ordinary OCR
        # text conservative (it may intentionally remain part of a photo).
        if (candidate.get("meta") or {}).get("overlay_text"):
            return {"headline": 5, "offer": 5, "cta": 4}.get(role, 4)
        return max(2, base - 1)
    if target == "shape":
        return base + 1 if mcfg is None else base
    if target == "image" and role in ("product", "person", "photo") and mcfg is None:
        return max(1, base - 1)
    return base


def default_mask_dilate(cfg: Optional[dict] = None) -> int:
    """Fallback dilation when an observation omits an explicit ``dilate`` value."""
    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    mcfg = icfg.get("mask_dilate")
    rcfg = cfg.get("reconstruct") or {}
    if isinstance(mcfg, dict):
        return int(mcfg.get("default", rcfg.get("mask_dilate", 2)))
    if isinstance(mcfg, (int, float)):
        return int(mcfg)
    return int(rcfg.get("mask_dilate", 2))


def build_union_mask(canvas: tuple[int, int], observations: Iterable[dict],
                     run_dir: Optional[str] = None, default_dilate: int = 2,
                     cfg: Optional[dict] = None):
    """Return one uint8 union mask from canonical (already deduplicated) entities."""
    cv2, np, _ = _deps()
    icfg = (cfg or {}).get("inpaint") or {}
    feather = int(icfg.get("mask_feather", 0))
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
        mask = solidify_mask(mask)
        radius = int(item.get("dilate", default_dilate))
        if radius > 0:
            k = 2 * radius + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        union = cv2.bitwise_or(union, mask.astype(np.uint8))
    if feather > 0 and np.any(union):
        union = feather_mask_edges(union, feather)
    return union


def _mask_bbox(mask):
    """Return the tight exclusive ``(x0, y0, x1, y1)`` bounds of a mask."""
    _, np, _ = _deps()
    ys, xs = np.where(np.asarray(mask) > 0)
    if not xs.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _observation_mask(item: dict, canvas: tuple[int, int], run_dir=None):
    """Materialize one canonical observation's dilated binary removal mask."""
    cv2, np, _ = _deps()
    width, height = canvas
    if item.get("keep_in_background") or item.get("is_background"):
        return np.zeros((height, width), dtype=np.uint8)
    mask = item.get("mask_array")
    if mask is None:
        path = item.get("mask_path") or item.get("mask_src")
        if not path and isinstance(item.get("mask"), dict):
            path = item["mask"].get("src")
        mask = mask_on_canvas(path, item.get("box") or {}, canvas, run_dir)
    mask = solidify_mask(mask)
    radius = max(0, int(item.get("dilate", 0)))
    if radius:
        mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    return mask


def build_inpaint_regions(canvas: tuple[int, int], observations: Iterable[dict], union_mask,
                          cfg: Optional[dict] = None, run_dir: Optional[str] = None):
    """Group canonical removals without globally bridging unrelated objects.

    Regions begin at the semantic-candidate boundary.  Duplicate masks and text nested in a
    removable product/button/card are joined, but nearby headline/product masks are not.  The
    returned masks are disjoint and their OR is exactly ``union_mask``.
    """
    cv2, np, _ = _deps()
    width, height = canvas
    rcfg = ((cfg or {}).get("inpaint") or {}).get("regional") or {}
    containment = float(rcfg.get("nested_containment", 0.88))
    duplicate_overlap = float(rcfg.get("duplicate_overlap", 0.65))
    items = []
    for index, source in enumerate(observations):
        mask = _observation_mask(source, canvas, run_dir)
        area = int(np.count_nonzero(mask))
        bbox = _mask_bbox(mask)
        if not area or bbox is None:
            continue
        meta = source.get("meta") or {}
        items.append({
            "id": str(source.get("id") or f"region-{index}"),
            "target": str(source.get("target") or "image"),
            "role": str(source.get("role") or meta.get("role") or ""),
            "parent_id": source.get("parent_id") or meta.get("parent_id"),
            "mask": mask, "area": area, "bbox": bbox,
        })

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner_roles = {"product", "person", "photo", "button", "badge", "chip", "card", "shape"}
    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            right = items[j]
            intersection = int(np.count_nonzero((left["mask"] > 0) & (right["mask"] > 0)))
            overlap = intersection / max(1, min(left["area"], right["area"]))
            explicit_parent = (
                left["parent_id"] == right["id"] or right["parent_id"] == left["id"]
            )
            nested_text = False
            if {left["target"], right["target"]} & {"text"}:
                text = left if left["target"] == "text" else right
                owner = right if text is left else left
                x0, y0, x1, y1 = text["bbox"]
                ox0, oy0, ox1, oy1 = owner["bbox"]
                bbox_contained = x0 >= ox0 and y0 >= oy0 and x1 <= ox1 and y1 <= oy1
                owner_semantic = owner["target"] in ("image", "shape") and (
                    owner["role"].lower() in owner_roles or owner["area"] > text["area"] * 2
                )
                nested_text = owner_semantic and (overlap >= containment or bbox_contained)
            if explicit_parent or overlap >= duplicate_overlap or nested_text:
                join(i, j)

    grouped = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)

    regions = []
    for members in grouped.values():
        mask = np.zeros((height, width), dtype=np.uint8)
        for member in members:
            mask = cv2.bitwise_or(mask, member["mask"])
        regions.append({
            "ids": [member["id"] for member in members],
            "targets": sorted(set(member["target"] for member in members)),
            "roles": sorted(set(member["role"] for member in members if member["role"])),
            "mask": mask,
            "group_reason": "semantic-overlap" if len(members) > 1 else "candidate",
        })

    # Largest first gives later small crops already-clean context. Assign every union pixel
    # once so overlapping-but-unmerged artistic elements cannot be regenerated twice.
    regions.sort(key=lambda r: int(np.count_nonzero(r["mask"])), reverse=True)
    wanted = solidify_mask(union_mask)
    assigned = np.zeros_like(wanted)
    for region in regions:
        region["mask"] = cv2.bitwise_and(region["mask"], wanted)
        region["mask"] = cv2.bitwise_and(region["mask"], cv2.bitwise_not(assigned))
        assigned = cv2.bitwise_or(assigned, region["mask"])
    missing = cv2.bitwise_and(wanted, cv2.bitwise_not(assigned))
    if np.any(missing):
        regions.append({
            "ids": ["union-remainder"], "targets": [], "roles": [],
            "mask": missing, "group_reason": "union-coverage",
        })
    regions = [region for region in regions if np.any(region["mask"])]
    return regions


def _regional_crop(mask, cfg: Optional[dict] = None):
    """Mask-derived crop with configurable source context and model padding.

    ``context_mode`` may be ``local`` (default), ``expanded`` (a larger local window),
    or ``global`` (the complete source canvas).  The caller always supplies original
    source pixels, never earlier generated pixels, so larger context cannot propagate a
    previous region's hallucination into a later regional inpaint call.
    """
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    h, w = mask.shape[:2]
    regional = ((cfg or {}).get("inpaint") or {}).get("regional") or {}
    min_context = int(regional.get("min_context", 64))
    max_context = int(regional.get("max_context", 96))
    min_crop = int(regional.get("min_crop", 256))
    alignment = max(1, int(regional.get("alignment", 16)))
    context_mode = str(regional.get("context_mode", "local")).lower()
    if context_mode not in ("local", "expanded", "global"):
        context_mode = "local"
    if context_mode == "global":
        crop_w, crop_h = w, h
        pad_right = (-crop_w) % alignment
        pad_bottom = (-crop_h) % alignment
        return (0, 0, w, h), (0, 0, pad_right, pad_bottom), max(w, h)
    x0, y0, x1, y1 = bbox
    span = max(x1 - x0, y1 - y0)
    scale = float(regional.get("context_scale", 1.0))
    if context_mode == "expanded":
        scale *= float(regional.get("expanded_context_scale", 2.0))
    scale = max(0.1, scale)
    context = int(round(min(max_context * scale, max(min_context * scale, span * .18 * scale))))
    x0, y0 = max(0, x0 - context), max(0, y0 - context)
    x1, y1 = min(w, x1 + context), min(h, y1 + context)
    if x1 - x0 < min_crop:
        extra = min_crop - (x1 - x0)
        x0, x1 = max(0, x0 - extra // 2), min(w, x1 + extra - extra // 2)
        x0 = max(0, x1 - min_crop)
    if y1 - y0 < min_crop:
        extra = min_crop - (y1 - y0)
        y0, y1 = max(0, y0 - extra // 2), min(h, y1 + extra - extra // 2)
        y0 = max(0, y1 - min_crop)
    crop_w, crop_h = x1 - x0, y1 - y0
    pad_right = (-crop_w) % alignment
    pad_bottom = (-crop_h) % alignment
    return (x0, y0, x1, y1), (0, 0, pad_right, pad_bottom), context


def _analytic_fill_allowed(complexity, regional_cfg, *, has_model, flat_residual,
                           flat_gradient, flux_gradient, flat_ui_archetype,
                           ui_chrome_hole):
    """May this hole take a FLAT/affine analytic fill? -> (analytic, ui_chrome_analytic).

    Analytic fills are cheap and never hallucinate, but they paint ONE colour (or a plane)
    across the hole. That is right for a flat plate and wrong for a gradient: filling a ramp
    with its dominant colour lays down an off-palette RECTANGLE (013: the "61% OFF" hole on
    the green seal came back a dark-green block that also clipped the baked "+ FREE GIFTS";
    016: the flat teal patch between the bears).

    Both "earn it back" clauses below used to ignore ``gradient_p90`` entirely and so could
    overturn the base rule's correct refusal:

      * dominant-flat-rgb: a high dominant_fraction only says "one colour covers most of the
        ring" -- on a ramp that colour is just the middle of it.
      * flat-UI archetype: the ARCHETYPE being flat does not make every hole in it flat; a
        gradient seal sits on a flat plate all the time.

    Past ``flux_gradient`` the router already classifies the same region as
    ``complex_background``, so calling it dominant-flat is self-contradictory. Both clauses
    therefore share that ceiling and hand a real gradient to the active backend.
    """
    if not has_model:
        return False, False
    # NB: `or 1e9` is wrong here — a perfectly flat plate measures residual/gradient 0.0,
    # which is falsy and would read as "infinitely complex", refusing analytic on exactly
    # the holes it suits best.
    def _metric(key):
        value = complexity.get(key)
        return float(value) if isinstance(value, (int, float)) else 1e9

    residual = _metric("residual_p90")
    gradient = _metric("gradient_p90")
    analytic = bool(
        residual <= flat_residual
        # A panel boundary can raise ring gradient even when a constant/affine model
        # explains the actual colours almost perfectly (ad9's black/charcoal bars).
        and (gradient <= flat_gradient or residual <= flat_residual * .4)
    )
    dominant_flat_gradient = float(regional_cfg.get(
        "dominant_flat_gradient_p90", flux_gradient))
    if (complexity.get("model") == "dominant-flat-rgb"
            and residual <= flat_residual
            and gradient <= dominant_flat_gradient):
        analytic = True
    if (not analytic and flat_ui_archetype and ui_chrome_hole
            and float(complexity.get("dominant_fraction") or 0) >= float(
                regional_cfg.get("ui_analytic_dominant_fraction", 0.40))
            and residual <= float(
                regional_cfg.get("ui_analytic_residual_p90", flat_residual * 2.5))
            and gradient <= float(
                regional_cfg.get("ui_analytic_gradient_p90", flux_gradient))):
        return True, True
    return analytic, False


def _background_model(rgb, mask, global_union, cfg: Optional[dict] = None):
    """Fit a robust local affine RGB plate and report exterior complexity."""
    cv2, np, _ = _deps()
    regional = ((cfg or {}).get("inpaint") or {}).get("regional") or {}
    ring_radius = max(4, int(regional.get("ring_radius", 24)))
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ring = (cv2.dilate(binary, np.ones((2 * ring_radius + 1,) * 2, np.uint8)) > 0)
    ring &= binary == 0
    ring &= np.asarray(global_union) == 0
    ys, xs = np.where(ring)
    minimum = int(regional.get("min_ring_samples", 64))
    if xs.size < minimum:
        return None, {"ring_samples": int(xs.size), "model": "insufficient"}
    if xs.size > 20000:
        take = np.linspace(0, xs.size - 1, 20000).astype(int)
        xs, ys = xs[take], ys[take]
    height, width = binary.shape
    design = np.column_stack([
        np.ones(xs.size), xs.astype(np.float32) / max(1, width - 1),
        ys.astype(np.float32) / max(1, height - 1),
    ])
    values = np.asarray(rgb, dtype=np.float32)[ys, xs]
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)[ys, xs]
    # Large product masks are frequently incomplete, so their ring can contain a strip
    # of package/chocolate pixels. If a clear plate colour still dominates, fit that mode
    # directly instead of misclassifying the contamination as photographic context.
    plate = np.median(values, axis=0)
    tolerance = float(regional.get("dominant_plate_tolerance", 12.0))
    near_plate = np.max(np.abs(values - plate), axis=1) <= tolerance
    dominant_fraction = float(np.mean(near_plate))
    dominant_required = float(regional.get("dominant_plate_fraction", 0.55))
    if dominant_fraction >= dominant_required and int(np.count_nonzero(near_plate)) >= minimum:
        coeff = np.zeros((3, 3), dtype=np.float32)
        coeff[0] = np.median(values[near_plate], axis=0)
        error = np.max(np.abs(values[near_plate] - coeff[0]), axis=1)
        model_name = "dominant-flat-rgb"
    else:
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        error = np.max(np.abs(values - design @ coeff), axis=1)
        keep = error <= np.percentile(error, 80)
        if int(np.count_nonzero(keep)) >= minimum:
            coeff, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            error = np.max(np.abs(values - design @ coeff), axis=1)
        model_name = "affine-rgb"
    diagnostics = {
        "ring_samples": int(xs.size), "model": model_name,
        "dominant_fraction": round(dominant_fraction, 4),
        "residual_median": round(float(np.median(error)), 4),
        "residual_p90": round(float(np.percentile(error, 90)), 4),
        "gradient_p90": round(float(np.percentile(gradient, 90)), 4),
    }
    return coeff, diagnostics


def _render_background_model(shape, coeff):
    _, np, _ = _deps()
    height, width = shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    design = np.stack([
        np.ones_like(xx, dtype=np.float32), xx.astype(np.float32) / max(1, width - 1),
        yy.astype(np.float32) / max(1, height - 1),
    ], axis=-1)
    return np.clip(design @ coeff, 0, 255).astype(np.uint8)


def _opencv_inpaint(rgb, mask, radius: int = 5, method=None):
    cv2, np, _ = _deps()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    # Flat ad plates are common and OpenCV's diffusion otherwise drags the
    # button/product colour deep into a wide hole.  When the exterior ring is
    # genuinely uniform, its robust median is a more faithful deterministic fill.
    ring_radius = max(2, int(radius))
    kernel = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), np.uint8)
    ring = (cv2.dilate(binary, kernel, iterations=1) > 0) & (binary == 0)
    exterior = np.asarray(rgb, dtype=np.uint8)[ring]
    plate = np.median(exterior, axis=0) if exterior.size else np.zeros(3)
    near_plate = np.max(np.abs(exterior.astype(np.float32) - plate), axis=1) <= 8.0 if exterior.size else []
    # Use a robust majority test rather than raw variance: imperfect masks leave a
    # thin strip of the foreground in the ring, but it should not veto an otherwise
    # uniform plate.
    if exterior.shape[0] >= 16 and float(np.mean(near_plate)) >= 0.72:
        result = np.asarray(rgb, dtype=np.uint8).copy()
        result[binary > 0] = plate.astype(np.uint8)
        return result
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


def _boundary_color_match(source, generated, mask, radius: int = 5, max_shift: float = 24.0):
    """Align generated boundary colour to retained context without touching source pixels."""
    cv2, np, _ = _deps()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return np.asarray(generated, dtype=np.uint8), [0.0, 0.0, 0.0]
    kernel = np.ones((3, 3), np.uint8)
    ring = (cv2.dilate(binary, kernel, iterations=max(1, int(radius))) > 0) & (binary == 0)
    if int(np.count_nonzero(ring)) < 16:
        return np.asarray(generated, dtype=np.uint8), [0.0, 0.0, 0.0]
    src = np.asarray(source, dtype=np.float32)
    gen = np.asarray(generated, dtype=np.float32)
    delta = np.median(src[ring], axis=0) - np.median(gen[ring], axis=0)
    delta = np.clip(delta, -float(max_shift), float(max_shift))
    matched = np.clip(gen + delta.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return matched, [round(float(value), 3) for value in delta]


def _opencv_auto(rgb, mask, radius):
    """Pick the best deterministic OpenCV candidate using continuity and residue signals."""
    telea = _opencv_inpaint(rgb, mask, radius, "telea")
    ns = _opencv_inpaint(rgb, mask, radius, "ns")
    telea_quality = _candidate_quality(rgb, telea, mask)
    ns_quality = _candidate_quality(rgb, ns, mask)
    if ns_quality["total"] + 1e-6 < telea_quality["total"]:
        return ns, "opencv-ns", {
            "telea_seam": telea_quality["seam"], "ns_seam": ns_quality["seam"],
            "telea_quality": telea_quality, "ns_quality": ns_quality,
        }
    return telea, "opencv-telea", {
        "telea_seam": telea_quality["seam"], "ns_seam": ns_quality["seam"],
        "telea_quality": telea_quality, "ns_quality": ns_quality,
    }


def _candidate_quality(source, candidate, mask, cfg: Optional[dict] = None) -> dict:
    """Rank a filled candidate without requiring a learned evaluator.

    Seam energy remains available as the established signal.  Texture continuity,
    structural continuity, and compact high-frequency residue provide deterministic
    counterweights when two candidates have similarly clean seams.
    """
    cv2, np, _ = _deps()
    icfg = (cfg or {}).get("inpaint") or {}
    quality_cfg = icfg.get("quality") or {}
    # Preserve the direct call for compatibility with existing seam probes/tests.
    seam = float(_seam_energy(candidate, mask))
    composed = np.asarray(source, dtype=np.uint8).copy()
    selected = np.asarray(mask) > 0
    metric_candidate = np.asarray(candidate, dtype=np.uint8)
    # A defensive resize is needed for permissive third-party/test backends that return
    # a padded or full-resolution image during a coarse pass. The seam probe above still
    # receives the untouched candidate for legacy diagnostics.
    if metric_candidate.shape[:2] != composed.shape[:2]:
        metric_candidate = cv2.resize(
            metric_candidate, (composed.shape[1], composed.shape[0]), interpolation=cv2.INTER_LINEAR,
        )
    composed[selected] = metric_candidate[selected]
    metrics = candidate_metrics(source, composed, mask)
    weights = {
        "seam": float(quality_cfg.get("seam_weight", 1.0)),
        "texture": float(quality_cfg.get("texture_weight", 0.35)),
        "structure": float(quality_cfg.get("structure_weight", 0.20)),
        "residue": float(quality_cfg.get("residue_weight", 0.80)),
    }
    total = (
        seam * weights["seam"]
        + float(metrics["texture"]) * weights["texture"]
        + float(metrics["structure"]) * weights["structure"]
        + float(metrics["residue"]) * weights["residue"]
    )
    return {
        "seam": round(seam, 6),
        "texture": round(float(metrics["texture"]), 6),
        "structure": round(float(metrics["structure"]), 6),
        "residue": round(float(metrics["residue"]), 6),
        "total": round(float(total), 6),
        "weights": weights,
    }


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
    # simple-lama-inpainting expects a binary L-mode mask.  Passing a view with a
    # soft/odd-shaped dtype can make the package normalize the mask incorrectly,
    # which leaves the original glyphs in its result even though our union mask is
    # correct.  Normalize at the backend boundary and keep the source dimensions.
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")
    binary_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(np.ascontiguousarray(binary_mask), mode="L")
    result = model(image, mask_image)
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


# Archetypes whose plates are flat/banded UI chrome (Codia ships these as SOLID rects,
# never a generative inpaint). For these, uniform mask holes are filled with their local
# plate colour analytically and never routed through Flux/LaMa — this removes the messy
# inpainted chrome + residue that a generative backend leaves on 009's dark bands.
_FLAT_PLATE_ARCHETYPES = frozenset({
    "social_screenshot", "caption_over_photo", "comparison_grid", "product_on_flat",
})


def _solid_flat_enabled(cfg: Optional[dict]) -> bool:
    """Route flat/uniform holes to a solid plate-colour fill instead of a generative model."""
    icfg = (cfg or {}).get("inpaint") or {}
    if "solid_flat_regions" in icfg:
        return bool(icfg.get("solid_flat_regions"))
    from src import format_readiness

    # Prefer format capability (flat_plate / ui_chrome) so new formats need no preset.
    fmt = format_readiness.format_from_cfg(cfg)
    if fmt.get("capabilities"):
        return format_readiness.prefers_solid_flat(cfg)
    # Default ON for flat/UI archetypes only; genuine-photo archetypes keep pure inpaint.
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()
    extra = set(icfg.get("solid_flat_archetypes") or ())
    return archetype in (_FLAT_PLATE_ARCHETYPES | extra)


def _flat_hole_fill(rgb, mask, cfg: Optional[dict] = None):
    """Solid-fill every uniform mask component with its local plate colour.

    Returns ``(working, remaining_mask, info)`` where ``working`` has the flat components
    replaced by their surrounding plate colour and ``remaining_mask`` is the still-unfilled
    (genuinely textured) portion for the generative backend, or ``None`` when disabled.

    A component is "flat" when the ring of source pixels just outside it is dominated by a
    single colour (>= ``uniform_fraction`` within ``tolerance``). Textured holes (photo
    plates) stay in ``remaining_mask`` so real inpainting still runs for them.
    """
    if not _solid_flat_enabled(cfg):
        return None
    cv2, np, _ = _deps()
    icfg = (cfg or {}).get("inpaint") or {}
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return None
    ring_radius = max(2, int(icfg.get("solid_flat_ring", 6)))
    uniform_fraction = float(icfg.get("solid_flat_uniform_fraction", 0.85))
    tolerance = float(icfg.get("solid_flat_tolerance", 8.0))
    min_ring = int(icfg.get("solid_flat_min_ring", 24))
    source = np.asarray(rgb, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kernel = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), np.uint8)
    working = source.copy()
    remaining = np.zeros_like(binary)
    filled = filled_px = 0
    for lab in range(1, num):
        comp = labels == lab
        comp_u8 = comp.astype(np.uint8)
        ring = (cv2.dilate(comp_u8, kernel) > 0) & (~comp) & (binary == 0)
        exterior = source[ring]
        if exterior.shape[0] < min_ring:
            remaining[comp] = 1
            continue
        plate = np.median(exterior.astype(np.float32), axis=0)
        near = np.max(np.abs(exterior.astype(np.float32) - plate), axis=1) <= tolerance
        if float(np.mean(near)) >= uniform_fraction:
            working[comp] = plate.astype(np.uint8)
            filled += 1
            filled_px += int(comp.sum())
        else:
            remaining[comp] = 1
    info = {
        "components": int(num - 1),
        "flat_filled": filled,
        "flat_filled_px": filled_px,
        "remaining_px": int(remaining.sum()),
    }
    return working, (remaining * 255).astype(np.uint8), info


def _inpaint_single_pass(rgb, mask, cfg: Optional[dict] = None):
    """Run one inpaint backend selection without multi-pass orchestration."""
    _, np, _ = _deps()
    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    mode = str(icfg.get("mode", "auto")).lower()
    strict_acceptance = bool(icfg.get("strict_acceptance", False))
    comfy_ok = comfyui_healthy(cfg, probe=icfg.get("_comfyui_probe"))
    lama_ok = _big_lama_available()
    diagnostics: dict = {
        "comfyui_healthy": comfy_ok,
        "big_lama_installed": lama_ok,
        "strict_acceptance": strict_acceptance,
    }

    generated = None
    used = mode
    fallback_reason = None

    # Flux Fill via ComfyUI (real GPU inpaint, quantized GGUF, native Fill workflow).
    # Attempted first when explicitly selected (inpaint.mode: flux_comfy) or opted into
    # under auto mode (inpaint.comfy.enabled: true). When the GPU box is offline the seam
    # returns None and we degrade to Big-LaMa/OpenCV below, so the pipeline never crashes
    # merely because ComfyUI is down.
    comfy_inpaint = icfg.get("comfy") or {}
    flux_modes = ("flux_comfy", "flux-comfy", "flux")
    flux_requested = mode in flux_modes or (mode == "auto" and bool(comfy_inpaint.get("enabled")))
    if flux_requested:
        flux_out = None
        try:
            flux_out = _flux_comfy_inpaint(np.asarray(rgb, dtype=np.uint8), mask, cfg)
        except Exception as exc:  # defensive: a client bug must not crash the run
            diagnostics["flux_comfy_error"] = str(exc)
        if flux_out is not None:
            generated = np.asarray(flux_out, dtype=np.uint8)
            used = "flux-comfy"
            diagnostics["backend_choice"] = "flux-comfy"
            diagnostics["flux_comfy"] = "ok"
        else:
            diagnostics["flux_comfy"] = "unavailable"
            fallback_reason = "flux_comfy_unavailable"
            worker = _load_qwen_worker()
            last_error = ""
            try:
                last_error = str(getattr(getattr(worker, "flux_inpaint", None), "__dict__", {}).get("last_error", "") or "")
            except Exception:
                last_error = ""
            if last_error:
                diagnostics["flux_comfy_last_error"] = last_error
            flux_required = bool(comfy_inpaint.get("required"))
            require_active = bool((cfg.get("runtime") or {}).get("require_active_models"))
            if (mode in flux_modes and (flux_required or require_active)
                    and not icfg.get("allow_fallback", True)):
                diagnostics["active_model_required"] = True
                raise RuntimeError(
                    "flux_comfy inpaint required but ComfyUI/Flux Fill is unavailable"
                )

    # Optional PowerPaint seam. The adapter is user-provided and deliberately has no
    # bundled GPU dependency. An importable adapter is not treated as model validation.
    powerpaint_modes = ("powerpaint", "power-paint")
    if generated is None and mode in powerpaint_modes:
        try:
            powerpaint_out = _powerpaint_inpaint(np.asarray(rgb, dtype=np.uint8), mask, cfg)
        except Exception as exc:
            powerpaint_out = None
            diagnostics["powerpaint_error"] = str(exc)
        if powerpaint_out is not None:
            generated = np.asarray(powerpaint_out, dtype=np.uint8)
            used = "powerpaint"
            diagnostics["powerpaint"] = "ok"
            diagnostics["backend_choice"] = used
        else:
            diagnostics["powerpaint"] = "unavailable"
            fallback_reason = fallback_reason or "powerpaint_unavailable"
            powerpaint_cfg = icfg.get("powerpaint") or {}
            if bool(powerpaint_cfg.get("required", False)) and not icfg.get("allow_fallback", True):
                raise RuntimeError("PowerPaint inpaint required but its adapter is unavailable")

    try_lama = mode in ("lama", "big-lama", "simple-lama")
    if mode == "auto":
        # Big-LaMa is a local pip package; whether it can run has nothing to do with
        # ComfyUI (the Qwen layered-diffusion backend, advisory per run_report._required).
        # Gating on comfy_ok here silently downgraded every run on this RTX box to the
        # OpenCV fallback whenever ComfyUI was off — which then failed all 16 benchmark
        # images as "inpaint-unavailable" runtime violations under require_active_models.
        try_lama = lama_ok
        if not lama_ok:
            diagnostics["auto_skip_reason"] = "big_lama_missing"
            fallback_reason = fallback_reason or "big_lama_missing"
    # A downed Flux ComfyUI degrades to Big-LaMa (higher quality than OpenCV) before OpenCV.
    if generated is None and mode in flux_modes and lama_ok:
        try_lama = True
    if generated is None and mode in powerpaint_modes and lama_ok:
        try_lama = True

    if generated is None and try_lama:
        try:
            generated = _simple_lama(np.asarray(rgb, dtype=np.uint8), mask)
            used = "big-lama"
            diagnostics["backend_choice"] = "big-lama"
        except Exception as exc:
            diagnostics["big_lama_error"] = str(exc)
            fallback_reason = fallback_reason or "big_lama_error"
            require_active = bool((cfg.get("runtime") or {}).get("require_active_models"))
            if ((mode != "auto" and not icfg.get("allow_fallback", True))
                    or require_active or strict_acceptance):
                diagnostics["active_model_required"] = require_active
                raise
            print(f"[inpaint] Big-LaMa unavailable ({exc}); using OpenCV fallback")

    if generated is None:
        explicit_opencv = mode in ("opencv", "cv2")
        if strict_acceptance and not explicit_opencv:
            reason = fallback_reason or "no_active_inpaint_backend"
            diagnostics["opencv_fallback_blocked"] = reason
            raise RuntimeError(
                "strict inpaint acceptance blocks the OpenCV fallback "
                f"(requested={mode}, reason={reason})"
            )
        radius = int(icfg.get("opencv_radius", 5))
        auto_default = "auto" if mode in ("auto", *flux_modes) else "telea"
        fallback_method = str(icfg.get("opencv_method", auto_default)).lower()
        if fallback_method in ("auto", "best"):
            generated, used, auto_diag = _opencv_auto(np.asarray(rgb, dtype=np.uint8), mask, radius)
            diagnostics.update(auto_diag)
        else:
            generated = _opencv_inpaint(np.asarray(rgb, dtype=np.uint8), mask, radius, fallback_method)
            used = "opencv-ns" if fallback_method in ("ns", "navier-stokes", "navier_stokes") else "opencv-telea"
        diagnostics["backend_choice"] = used

    opencv_fallback = used.startswith("opencv") and mode not in ("opencv", "cv2")
    selected_class = "deterministic-fallback" if used.startswith("opencv") else "active-model"
    diagnostics["backend_route"] = {
        "requested": mode,
        "selected": used,
        "selected_class": selected_class,
        "strict_acceptance": strict_acceptance,
        "opencv_fallback_used": opencv_fallback,
        "fallback_reason": fallback_reason,
    }
    diagnostics["backend_class"] = "fallback" if used.startswith("opencv") else "active"

    return generated, used, diagnostics


def _multipass_inpaint(rgb, mask, cfg: Optional[dict] = None):
    """Coarse-to-fine inpaint for large removal regions."""
    cv2, np, _ = _deps()
    icfg = (cfg or {}).get("inpaint") or {}
    threshold = float(icfg.get("multipass_fraction", 0.12))
    fraction = _mask_fraction(mask)
    if fraction < threshold:
        generated, used, diagnostics = _inpaint_single_pass(rgb, mask, cfg)
        diagnostics["inpaint_passes"] = 1
        diagnostics["masked_fraction"] = round(fraction, 6)
        return generated, used, diagnostics

    h, w = rgb.shape[:2]
    half = (max(1, w // 2), max(1, h // 2))
    small_rgb = cv2.resize(np.asarray(rgb, dtype=np.uint8), half, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(np.asarray(mask, dtype=np.uint8), half, interpolation=cv2.INTER_NEAREST)
    coarse, coarse_backend, coarse_diag = _inpaint_single_pass(small_rgb, small_mask, cfg)
    coarse_up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    working = np.asarray(rgb, dtype=np.uint8).copy()
    selected = np.asarray(mask) > 0
    working[selected] = coarse_up[selected]
    generated, used, diagnostics = _inpaint_single_pass(working, mask, cfg)
    diagnostics.update({
        "inpaint_passes": 2,
        "masked_fraction": round(fraction, 6),
        "coarse_backend": coarse_backend,
        "coarse_fraction": threshold,
    })
    diagnostics.update({f"coarse_{key}": value for key, value in coarse_diag.items()})
    return generated, used, diagnostics


def _gradient_hole_fill(rgb, mask, cfg: Optional[dict] = None):
    """Fill mask holes analytically when the visible plate is a clean linear gradient.

    Fits colour(x,y) = a·x + b·y + c per channel (least squares) on visible pixels,
    validates the fit on a held-out visible subsample, and fills masked pixels by
    evaluating the plane. Returns ``(filled, remaining_mask, info)`` or ``None`` when
    the plate is not a clean gradient (validation residual too high). Deliberately
    linear-only: radial/blob washes fail validation and keep their generative route.
    """
    cv2, np, _ = _deps()
    opts = ((cfg or {}).get("inpaint") or {}).get("gradient_fill") or {}
    if opts.get("enabled", True) is False:
        return None
    img = np.asarray(rgb, dtype=np.uint8)
    hole = np.asarray(mask) > 0
    h, w = img.shape[:2]
    if float(hole.sum()) / hole.size < float(opts.get("min_hole_fraction", 0.02)):
        return None
    # Fit on a LOCAL ring around the hole, not the whole visible canvas: on a real ad
    # the global visible set includes products/badges/photos, so a whole-image plane
    # fit never validates and the fill never fired (013 run-3: headline smudge band
    # survived because the grüns bag dominated the fit). The ring is the plate the
    # hole actually continues into.
    cv2_mod, _, _ = _deps()
    ring_px = max(16, int(opts.get("ring_px", 48)))
    kernel = np.ones((2 * ring_px + 1,) * 2, np.uint8)
    ring = (cv2_mod.dilate(hole.astype(np.uint8), kernel) > 0) & ~hole
    visible = ring
    n_visible = int(visible.sum())
    if n_visible < 500:
        return None
    ys, xs = np.nonzero(visible)
    rng = np.random.default_rng(0)
    take = min(len(xs), int(opts.get("fit_samples", 40000)))
    idx = rng.choice(len(xs), size=take, replace=False)
    fit_n = take * 3 // 4
    fi, vi = idx[:fit_n], idx[fit_n:]

    # Quadratic basis: ad washes are commonly EASED, not linear (013's red channel
    # ramps ~273 levels with visible easing — a plane misfits it by ~15 levels).
    def _design(xa, ya):
        x = xa / max(1, w)
        y = ya / max(1, h)
        return np.stack([x, y, y * y, x * x, np.ones(len(xa))], axis=1).astype(np.float64)

    A = _design(xs[fi], ys[fi])
    Av = _design(xs[vi], ys[vi])
    coeffs = []
    # Validation is a TRIMMED quantile (q35): the ring legitimately clips stray element
    # edges (gummy bears, badge rims) whose pixels are not plate; the plate's own pixels
    # must fit tightly, contamination may not. A loose median guard rejects the
    # bimodal case where "q35 fits" only because half the ring is one flat color.
    max_err = float(opts.get("max_validation_error", 7.0))
    max_median = float(opts.get("max_validation_median", 18.0))
    slope = 0.0
    for ch in range(3):
        b = img[ys[fi], xs[fi], ch].astype(np.float64)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        # Iterated trimmed refit (IRLS-lite): one trim pass is not enough when a large
        # element (013's product bag) clips the ring — its pixels survive the first
        # quartile cut and skew the surface (measured 25/255 truth error on a clean
        # plate strip). Three fit→trim rounds converge on the dominant smooth plate.
        keep = np.ones(len(b), dtype=bool)
        for _ in range(3):
            resid = np.abs(A @ sol - b)
            thresh = np.quantile(resid[keep], 0.75)
            new_keep = resid <= thresh
            if int(new_keep.sum()) < 50:
                break
            keep = new_keep
            sol, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        errs = np.abs(Av @ sol - img[ys[vi], xs[vi], ch].astype(np.float64))
        if float(np.quantile(errs, 0.35)) > max_err or float(np.median(errs)) > max_median:
            return None
        slope = max(slope, abs(float(sol[0])), abs(float(sol[1])),
                    abs(float(sol[2])), abs(float(sol[3])))
        coeffs.append(sol)
    # A (near-)constant plate is not a gradient — that is solid-flat's territory, and if
    # solid-flat already declined it there was a reason (e.g. hole-interior evidence).
    # Require a real colour ramp across the canvas before claiming the analytic fill.
    if slope < float(opts.get("min_slope", 8.0)):
        return None
    hy, hx = np.nonzero(hole)
    filled = img.copy()
    Ah = _design(hx, hy)
    for ch in range(3):
        filled[hy, hx, ch] = np.clip(Ah @ coeffs[ch], 0, 255).astype(np.uint8)
    remaining = np.zeros_like(np.asarray(mask, dtype=np.uint8))
    info = {"kind": "linear", "validation_max_err": max_err,
            "hole_fraction": round(float(hole.sum()) / hole.size, 4)}
    return filled, remaining, info


def _gradient_components_fill(rgb, mask, cfg: Optional[dict] = None):
    """Per-connected-component gradient fill over a (possibly scattered) union mask.

    Reconstruct's regional pass hands inpaint_array ONE union of many disjoint holes
    (every removed text line + element at once). A ring around that union spans the
    whole canvas, so the single-fit gradient never validated and the union fell to the
    generative backend wholesale — 013's plate kept readable LaMa ghost-glyphs of the
    headline. Fitting per component keeps each ring local. Components that validate
    fill analytically; the remainder stays on the generative route.
    """
    cv2, np, _ = _deps()
    opts = ((cfg or {}).get("inpaint") or {}).get("gradient_fill") or {}
    if opts.get("enabled", True) is False:
        return None
    hole = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(hole, connectivity=8)
    if n <= 2:  # 0/1 components: single-hole path
        return _gradient_hole_fill(rgb, mask, cfg)
    max_components = int(opts.get("max_components", 48))
    if n - 1 > max_components:
        return _gradient_hole_fill(rgb, mask, cfg)
    filled = np.asarray(rgb, dtype=np.uint8).copy()
    remaining = hole.copy() * 255
    total = int(hole.sum())
    done_px = 0
    comp_opts = dict(cfg or {})
    comp_opts["inpaint"] = dict((cfg or {}).get("inpaint") or {})
    comp_opts["inpaint"]["gradient_fill"] = {**opts, "min_hole_fraction": 0.0}
    fills = 0
    for label in range(1, n):
        comp = (labels == label).astype(np.uint8) * 255
        comp_px = int(np.count_nonzero(comp))
        if comp_px < int(opts.get("min_component_px", 400)):
            continue
        one = _gradient_hole_fill(filled, comp, comp_opts)
        if one is None:
            continue
        filled, _, _ = one
        remaining[comp > 0] = 0
        done_px += comp_px
        fills += 1
    if fills == 0:
        return None
    info = {"kind": "linear-per-component", "components_filled": fills,
            "filled_fraction": round(done_px / max(1, total), 4)}
    return filled, remaining, info


def inpaint_array(rgb, mask, cfg: Optional[dict] = None, return_diagnostics: bool = False):
    """Inpaint RGB pixels and guarantee that pixels outside ``mask`` stay byte-identical."""
    cv2, np, Image = _deps()
    cfg = cfg or {}
    composite_mask = solidify_mask(mask)
    if not np.any(composite_mask):
        requested = str((cfg.get("inpaint") or {}).get("mode", "auto")).lower()
        result = (np.asarray(rgb, dtype=np.uint8).copy(), "none", {
            "backend_class": "none",
            "backend_route": {
                "requested": requested, "selected": "none", "selected_class": "none",
                "strict_acceptance": bool((cfg.get("inpaint") or {}).get("strict_acceptance", False)),
                "opencv_fallback_used": False, "fallback_reason": "empty_mask",
            },
        })
        return result if return_diagnostics else result[:2]

    original = np.asarray(rgb, dtype=np.uint8)
    # Flat/UI plates (Codia's solid-rect strategy): fill uniform holes with their local
    # plate colour and only route the genuinely-textured remainder through the generative
    # backend. This keeps 009's dark chrome clean instead of a messy inpaint + residue.
    flat = _flat_hole_fill(original, composite_mask, cfg)
    solid_flat_info = None
    generative_mask = composite_mask
    flat_source = original
    if flat is not None:
        flat_source, generative_mask, solid_flat_info = flat

    if solid_flat_info is not None and not np.any(generative_mask):
        # Whole plate was uniform → no generative backend needed at all.
        out = original.copy()
        selected = composite_mask > 0
        out[selected] = flat_source[selected]
        diagnostics = {
            "backend_class": "analytic", "solid_flat": solid_flat_info,
            "backend_route": {
                "requested": str((cfg.get("inpaint") or {}).get("mode", "auto")).lower(),
                "selected": "solid-flat", "selected_class": "analytic",
                "strict_acceptance": bool((cfg.get("inpaint") or {}).get("strict_acceptance", False)),
                "opencv_fallback_used": False, "fallback_reason": None,
            },
        }
        result = (out, "solid-flat", diagnostics)
        return result if return_diagnostics else result[:2]

    # Gradient plates (013 grüns: green→yellow wash): a smooth gradient background fails
    # the solid-median test above (no single flat colour) and previously fell straight
    # through to the generative backend — Big-LaMa then re-painted a near-full-canvas
    # hole per layer, tripping excessive-plate-destruction at 55%+. A linear per-channel
    # plane colour(x,y)=a·x+b·y+c fitted on the VISIBLE plate reconstructs such holes
    # exactly and analytically; a plate that is not a clean gradient fails the residual
    # check and routes to the generative backend as before.
    gradient_info = None
    if np.any(generative_mask):
        grad = _gradient_components_fill(flat_source, generative_mask, cfg)
        if grad is not None:
            flat_source, generative_mask, gradient_info = grad
            if not np.any(generative_mask):
                out = original.copy()
                selected = composite_mask > 0
                out[selected] = flat_source[selected]
                diagnostics = {
                    "backend_class": "analytic", "gradient_flat": gradient_info,
                    "backend_route": {
                        "requested": str((cfg.get("inpaint") or {}).get("mode", "auto")).lower(),
                        "selected": "gradient-flat", "selected_class": "analytic",
                        "strict_acceptance": bool((cfg.get("inpaint") or {}).get("strict_acceptance", False)),
                        "opencv_fallback_used": False, "fallback_reason": None,
                    },
                }
                if solid_flat_info is not None:
                    diagnostics["solid_flat"] = solid_flat_info
                result = (out, "gradient-flat", diagnostics)
                return result if return_diagnostics else result[:2]

    generated, used, diagnostics = _multipass_inpaint(
        flat_source, generative_mask, cfg,
    )
    if solid_flat_info is not None:
        diagnostics["solid_flat"] = solid_flat_info

    generated = np.asarray(generated, dtype=np.uint8)
    # Big-LaMa internally pads the image up to a multiple of 8 and returns at that padded
    # size, so its output can be a few pixels larger/smaller than the input (e.g. 344 vs
    # 338). Compositing that back against the original's boolean mask then raised
    # IndexError and crashed the whole run (crashed 3/16 real benchmark images the moment
    # Big-LaMa actually started running). Snap the generated plate back to the exact input
    # HxW before compositing so the mask always aligns.
    if generated.shape[:2] != original.shape[:2]:
        oh, ow = original.shape[:2]
        diagnostics["resized_from"] = f"{generated.shape[1]}x{generated.shape[0]}"
        generated = cv2.resize(generated, (ow, oh), interpolation=cv2.INTER_LINEAR)
    out = original.copy()
    selected = composite_mask > 0
    out[selected] = generated[selected]
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


def inpaint_regional(image_path: str, observations: Iterable[dict], union_mask,
                     output_path: str, cfg: Optional[dict] = None,
                     run_dir: Optional[str] = None) -> dict:
    """Build one clean plate through crop-local, per-canonical-region inpainting.

    The public artifact remains one canonical union mask and one background plate.  Internally,
    unrelated holes are never shown to a generative backend together, and every union pixel is
    assigned to exactly one region.
    """
    cv2, np, Image = _deps()
    cfg = cfg or {}
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    union = solidify_mask(union_mask)
    if union.shape != source.shape[:2]:
        raise ValueError(f"inpaint mask {union.shape} does not match image {source.shape[:2]}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not np.any(union):
        shutil.copyfile(image_path, output_path)
        return {"ok": True, "path": output_path, "backend": "none", "strategy": "regional",
                "masked_fraction": 0.0, "regions": [], "backend_counts": {}}

    regions = build_inpaint_regions(
        (source.shape[1], source.shape[0]), observations, union, cfg, run_dir,
    )
    # Each region sees immutable source pixels as context.  The composited working plate
    # is only the destination, never input to a later backend/model call; otherwise a
    # bad earlier completion can influence the next region's texture or structure.
    working = source.copy()
    regional_cfg = ((cfg.get("inpaint") or {}).get("regional") or {})
    flat_residual = float(regional_cfg.get("flat_residual_p90", 10.0))
    flat_gradient = float(regional_cfg.get("flat_gradient_p90", 10.0))
    flux_residual = float(regional_cfg.get("flux_residual_p90", 18.0))
    flux_gradient = float(regional_cfg.get("flux_gradient_p90", 18.0))
    flux_max_fraction = float(regional_cfg.get("flux_max_canvas_fraction", 0.025))
    analytic_max_fraction = float(regional_cfg.get("analytic_max_canvas_fraction", 0.12))
    force_active_for_large = float(regional_cfg.get("force_active_above_fraction", 1.0))
    base_mode = str((cfg.get("inpaint") or {}).get("mode", "auto")).lower()
    comfy = (cfg.get("inpaint") or {}).get("comfy") or {}
    flux_allowed = base_mode in ("flux", "flux-comfy", "flux_comfy") or (
        base_mode == "auto" and bool(comfy.get("enabled"))
    )
    records, backend_counts = [], {}
    processed = np.zeros_like(union)

    for number, region in enumerate(regions, start=1):
        mask = solidify_mask(region["mask"])
        region_fraction = float(np.count_nonzero(mask)) / max(1, mask.size)
        region_roles = {str(role).lower() for role in (region.get("roles") or []) if role}
        region_targets = {str(target).lower() for target in (region.get("targets") or []) if target}
        photo_hole = bool(
            region_targets & {"image"}
            or region_roles & {"photo", "product", "person", "cutout"}
        )
        ui_chrome_hole = bool(
            not photo_hole
            and (
                region_targets <= {"text", "shape", "icon"}
                or region_roles & {
                    "badge", "button", "chip", "pill", "cta", "icon", "chrome",
                    "divider", "bar", "ui-label", "label",
                    "banner", "ribbon", "brushstroke", "seal",
                    "starburst", "price_burst", "sale_burst", "burst",
                }
            )
        )
        archetype = str(((cfg.get("scene") or {}).get("archetype") or "")).lower()
        extra_flat = {
            str(a).lower()
            for a in ((cfg.get("inpaint") or {}).get("solid_flat_archetypes") or ())
        }
        from src import format_readiness
        fmt = format_readiness.format_from_cfg(cfg)
        if fmt.get("capabilities"):
            flat_ui_archetype = format_readiness.prefers_solid_flat(cfg)
        else:
            flat_ui_archetype = archetype in (_FLAT_PLATE_ARCHETYPES | extra_flat)
        crop_spec = _regional_crop(mask, cfg)
        if crop_spec is None:
            continue
        (x0, y0, x1, y1), padding, context = crop_spec
        _, _, pad_right, pad_bottom = padding
        crop_rgb = source[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].copy()
        crop_union = union[y0:y1, x0:x1]
        if pad_right or pad_bottom:
            crop_rgb = cv2.copyMakeBorder(crop_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
            crop_mask = cv2.copyMakeBorder(crop_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
            crop_union = cv2.copyMakeBorder(crop_union, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)

        coeff, complexity = _background_model(crop_rgb, crop_mask, crop_union, cfg)
        analytic, ui_chrome_analytic = _analytic_fill_allowed(
            complexity, regional_cfg,
            has_model=coeff is not None,
            flat_residual=flat_residual, flat_gradient=flat_gradient,
            flux_gradient=flux_gradient,
            flat_ui_archetype=flat_ui_archetype, ui_chrome_hole=ui_chrome_hole,
        )
        if ui_chrome_analytic:
            complexity = dict(complexity)
            complexity["ui_chrome_analytic"] = True
        # Analytic plates are safe (non-hallucinating) but can wipe subtle photographic
        # texture inside a removed product/person/photo region even when the exterior ring
        # looks flat. Prefer an active inpaint model for those holes unless the plate is
        # overwhelmingly uniform.
        if analytic and photo_hole and complexity.get("model") != "dominant-flat-rgb":
            analytic = False
        # Guardrail: analytic fills are great for small UI cutouts, but they destroy large
        # regions (e.g. a photo that was masked) by collapsing them to a single plate.
        # Flat/UI text+chrome holes get a higher ceiling — they are plates, not photos.
        analytic_cap = analytic_max_fraction
        if flat_ui_archetype and ui_chrome_hole:
            analytic_cap = float(regional_cfg.get(
                "ui_analytic_max_canvas_fraction",
                max(analytic_max_fraction, 0.35),
            ))
        if region_fraction >= analytic_cap:
            analytic = False
        complex_background = bool(
            coeff is not None and (
                complexity["residual_p90"] >= flux_residual
                or complexity["gradient_p90"] >= flux_gradient
            )
        )
        force_flux = bool(regional_cfg.get("force_flux", False))
        requested = "analytic-affine" if analytic and not force_flux else "big-lama"
        # Never spend Flux on text / badge / chip chrome — it regenerates glyph residue
        # and doubles wall time for a hole analytic/LaMa already owns better.
        flux_blocked = bool(
            "text" in region_targets
            or ui_chrome_hole
            or region_roles & {
                "badge", "button", "chip", "pill", "cta",
                "banner", "ribbon", "seal", "starburst", "burst",
            }
        )
        if not analytic or force_flux:
            if force_flux and flux_allowed and not flux_blocked:
                requested = "flux-comfy"
            elif base_mode == "opencv":
                requested = "opencv"
            elif base_mode in ("lama", "big-lama", "simple-lama"):
                requested = "big-lama"
            # A high residual around a large product/text mask often means the mask is
            # incomplete and the ring still sees foreground pixels. Flux interprets that
            # as photographic context and regenerates the product. Reserve it for genuinely
            # local complex holes; any region containing editable text uses conservative
            # LaMa. Flux can otherwise regenerate glyph-like residue around an adjacent
            # icon, creating a duplicate behind the new native Figma text.
            elif (flux_allowed and complex_background
                  and region_fraction <= flux_max_fraction
                  and not flux_blocked):
                requested = "flux-comfy"
            # If we removed a huge fraction of the canvas, avoid analytic and prefer an
            # active model when possible; otherwise the plate will flatten/hallucinate.
            if (region_fraction >= force_active_for_large
                    and requested == "big-lama"
                    and flux_allowed
                    and not flux_blocked):
                requested = "flux-comfy"

        started = time.monotonic()
        backend_diagnostics = {}
        if requested == "analytic-affine":
            generated = _render_background_model(crop_rgb.shape, coeff)
            used = "analytic-affine"
            backend_diagnostics["backend_choice"] = used
            backend_diagnostics["backend_class"] = "analytic"
            backend_diagnostics["backend_route"] = {
                "requested": requested, "selected": used, "selected_class": "analytic",
                "strict_acceptance": bool((cfg.get("inpaint") or {}).get("strict_acceptance", False)),
                "opencv_fallback_used": False, "fallback_reason": None,
            }
        elif requested == "big-lama" and (
                grad_region := _gradient_hole_fill(crop_rgb, crop_mask, cfg)) is not None:
            # Robust trimmed-gradient rescue: the affine gate above rejects a region
            # whose ring is contaminated by undetected foreground (013's headline ring
            # clips gummy bears), sending a smooth wash to Big-LaMa — which left
            # readable ghost glyphs of "We NEVER do this!" in the shipped plate. The
            # IRLS-trimmed quadratic fit validates on the plate's own pixels and fills
            # exactly; regions that genuinely aren't washes fail validation and keep LaMa.
            generated = grad_region[0]
            used = "gradient-flat"
            backend_diagnostics["backend_choice"] = used
            backend_diagnostics["backend_class"] = "analytic"
            backend_diagnostics["gradient_flat"] = grad_region[2]
            backend_diagnostics["backend_route"] = {
                "requested": "big-lama", "selected": used, "selected_class": "analytic",
                "strict_acceptance": bool((cfg.get("inpaint") or {}).get("strict_acceptance", False)),
                "opencv_fallback_used": False, "fallback_reason": None,
            }
        else:
            # Lazy Flux VRAM: unload VLM / pick GGUF only when a region actually needs Flux.
            # Analytic/LaMa-only runs (common under regional routing) skip the ~8–12 s churn.
            flux_vram_prep = None
            if requested == "flux-comfy":
                try:
                    from src import vram as _vram
                except ImportError:  # pragma: no cover
                    import vram as _vram  # type: ignore
                flux_vram_prep = _vram.ensure_flux_vram(cfg)
            region_cfg = dict(cfg)
            region_cfg["inpaint"] = dict(cfg.get("inpaint") or {})
            region_cfg["inpaint"]["mode"] = requested
            # A regional crop is already the coarse context window. Calling the global
            # coarse-to-fine wrapper would run Flux twice and invite a second hallucination.
            attempts = (max(1, int(comfy.get("attempts", 1)))
                        if requested == "flux-comfy" else 1)
            candidates = []
            base_seed = int(comfy.get("seed", 0))
            for attempt in range(attempts):
                attempt_cfg = dict(region_cfg)
                attempt_cfg["inpaint"] = dict(region_cfg["inpaint"])
                attempt_cfg["inpaint"]["comfy"] = dict(comfy)
                attempt_cfg["inpaint"]["comfy"]["seed"] = base_seed + attempt
                if requested == "flux-comfy":
                    if photo_hole:
                        attempt_cfg["inpaint"]["comfy"]["prompt"] = str(
                            regional_cfg.get("photo_prompt", "")
                        )
                    else:
                        attempt_cfg["inpaint"]["comfy"]["prompt"] = str(
                            regional_cfg.get(
                                "plate_prompt",
                                comfy.get("prompt", ""),
                            )
                        )
                candidate, candidate_backend, candidate_diag = _inpaint_single_pass(
                    crop_rgb, crop_mask, attempt_cfg,
                )
                candidate = np.asarray(candidate, dtype=np.uint8)
                if candidate.shape[:2] != crop_rgb.shape[:2]:
                    candidate = cv2.resize(candidate, (crop_rgb.shape[1], crop_rgb.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                if bool(regional_cfg.get("color_match_enabled", False)):
                    matched, shift = _boundary_color_match(
                        crop_rgb, candidate, crop_mask,
                        radius=int(regional_cfg.get("color_match_ring", 5)),
                        max_shift=float(regional_cfg.get("color_match_max_shift", 24)),
                    )
                else:
                    matched, shift = candidate, [0.0, 0.0, 0.0]
                preview_candidate = crop_rgb.copy()
                preview_candidate[crop_mask > 0] = matched[crop_mask > 0]
                quality = _candidate_quality(crop_rgb, preview_candidate, crop_mask, attempt_cfg)
                candidates.append((quality, matched, candidate_backend, candidate_diag,
                                   shift, base_seed + attempt))
            quality, generated, used, backend_diagnostics, shift, chosen_seed = min(
                candidates, key=lambda row: row[0]["total"]
            )
            backend_diagnostics = dict(backend_diagnostics)
            if flux_vram_prep and (
                    flux_vram_prep.get("vlm_evicted") or flux_vram_prep.get("flux_quant")
                    or not flux_vram_prep.get("already_prepared")):
                backend_diagnostics["flux_vram_prep"] = {
                    k: flux_vram_prep.get(k)
                    for k in ("vlm_evicted", "flux_quant", "already_prepared",
                              "free_mib_after")
                }
            backend_diagnostics.update({
                "attempts": attempts,
                "chosen_seed": chosen_seed,
                "seam_score": round(float(quality["seam"]), 4),
                "candidate_quality": quality,
                "candidate_quality_scores": [row[0] for row in candidates],
                "candidate_seam_scores": [round(float(row[0]["seam"]), 4) for row in candidates],
                "boundary_color_shift": shift,
            })
            if generated.shape[:2] != crop_rgb.shape[:2]:
                backend_diagnostics["resized_from"] = f"{generated.shape[1]}x{generated.shape[0]}"
                generated = cv2.resize(generated, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        original_h, original_w = y1 - y0, x1 - x0
        generated = np.asarray(generated, dtype=np.uint8)[:original_h, :original_w]
        local_mask = crop_mask[:original_h, :original_w] > 0
        target = working[y0:y1, x0:x1]
        target[local_mask] = generated[local_mask]
        processed = cv2.bitwise_or(processed, mask)
        backend_counts[used] = backend_counts.get(used, 0) + 1
        records.append({
            "index": number, "ids": region["ids"], "targets": region["targets"],
            "roles": region["roles"], "group_reason": region["group_reason"],
            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "model_size": {"w": crop_rgb.shape[1], "h": crop_rgb.shape[0]},
            "context": context,
            "context_mode": str(regional_cfg.get("context_mode", "local")).lower(),
            "context_source": "original-source-only",
            "masked_fraction_canvas": round(region_fraction, 6),
            "masked_fraction_crop": round(float(np.count_nonzero(crop_mask)) / crop_mask.size, 6),
            "complexity": complexity, "route": requested, "backend": used,
            "diagnostics": backend_diagnostics,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        })

    # Defense-in-depth: regional compositing must never change a byte outside the canonical union.
    result = source.copy()
    selected = union > 0
    result[selected] = working[selected]
    Image.fromarray(result).save(output_path)
    degraded = any(name.startswith("opencv") for name in backend_counts)
    return {
        "ok": True, "path": output_path, "backend": "regional", "strategy": "regional",
        "backend_class": "fallback" if degraded else "active",
        "backend_counts": backend_counts,
        "masked_fraction": round(float(np.count_nonzero(union)) / union.size, 6),
        "region_count": len(records), "regions": records,
        "coverage_fraction": round(
            float(np.count_nonzero((processed > 0) & (union > 0))) / max(1, np.count_nonzero(union)), 6
        ),
    }


def inpaint_role_aware(image_path: str, masks: dict, output_path: str,
                       cfg: Optional[dict] = None) -> dict:
    """Inpaint semantic removal masks with role-specific backends.

    ``text`` is deliberately processed with OpenCV on the original pixels; all
    remaining pixels use the configured backend (normally Big-LaMa).  Masks are
    made disjoint so the large-region pass cannot overwrite a text repair.
    """
    cv2, np, Image = _deps()
    cfg = cfg or {}
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    text_mask = solidify_mask(masks.get("text")) if masks.get("text") is not None else np.zeros(source.shape[:2], dtype=np.uint8)
    large_mask = solidify_mask(masks.get("large")) if masks.get("large") is not None else np.zeros(source.shape[:2], dtype=np.uint8)
    overlap = cv2.bitwise_and(text_mask, large_mask)
    # Ownership masks are intentionally disjoint, so a label carved out of its
    # button may have no literal overlap. Treat touching/near-touching text as part
    # of the removable large object; otherwise its cut-out island survives.
    if not np.any(overlap) and np.any(text_mask) and np.any(large_mask):
        near_large = cv2.dilate(large_mask, np.ones((5, 5), np.uint8), iterations=1)
        nearby = cv2.bitwise_and(text_mask, near_large)
        near_fraction = float(np.count_nonzero(nearby)) / max(1, np.count_nonzero(text_mask))
        ty, tx = np.where(text_mask > 0)
        ly, lx = np.where(large_mask > 0)
        contained = bool(
            tx.size and lx.size
            and tx.min() >= lx.min() and tx.max() <= lx.max()
            and ty.min() >= ly.min() and ty.max() <= ly.max()
        )
        if near_fraction >= 0.8 or contained:
            large_mask = cv2.bitwise_or(large_mask, text_mask)
            overlap = cv2.bitwise_and(text_mask, large_mask)
    working = source.copy()
    parts = []
    # When text sits on a removable button/card, the large object must own the
    # overlap. Filling the label first and then excluding it from the button pass
    # leaves a blue/white island in the clean plate. For disjoint masks retain the
    # legacy text-first ordering; for overlap, remove the complete large object first
    # and only repair text pixels that lie outside it.
    if np.any(overlap):
        working, backend, diagnostics = inpaint_array(working, large_mask, cfg, return_diagnostics=True)
        parts.append({"role": "large", "backend": backend, "masked_fraction": round(float(np.count_nonzero(large_mask)) / large_mask.size, 6), "diagnostics": diagnostics})
        text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(large_mask))
    if np.any(text_mask):
        text_cfg = dict(cfg)
        text_cfg["inpaint"] = dict(cfg.get("inpaint") or {})
        text_cfg["inpaint"].update({"mode": "opencv", "opencv_method": "telea"})
        working, backend, diagnostics = inpaint_array(working, text_mask, text_cfg, return_diagnostics=True)
        parts.append({"role": "text", "backend": backend, "masked_fraction": round(float(np.count_nonzero(text_mask)) / text_mask.size, 6), "diagnostics": diagnostics})
    if np.any(large_mask) and not np.any(overlap):
        working, backend, diagnostics = inpaint_array(working, large_mask, cfg, return_diagnostics=True)
        parts.append({"role": "large", "backend": backend, "masked_fraction": round(float(np.count_nonzero(large_mask)) / large_mask.size, 6), "diagnostics": diagnostics})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(working).save(output_path)
    return {
        "ok": True,
        "path": output_path,
        "backend": "role-aware",
        "masked_fraction": round(float(np.count_nonzero(text_mask | large_mask)) / text_mask.size, 6),
        "parts": parts,
    }
