"""normalize.py — stage 1: load + normalize an input ad into a clean sRGB PNG.

Opens the source, applies EXIF orientation, converts to sRGB RGB, caps the long
edge at 2048 px (aspect preserved), and writes `original.png` (verbatim copy of the
decoded source) + `normalized.png` under the run dir. Records the downscale factor so
downstream coordinates can be mapped back to the true source resolution if needed.

Real-world ad inputs are messy — recompressed platform re-exports, WebP, EXIF-rotated
phone screenshots, ICC-tagged exports, palette PNGs with alpha, animated GIF/WebP, and
both very large and very small canvases. This stage makes each of those classes safe and
records what it had to do, because every downstream stage trusts ``normalized.png``.

Pure CPU. Only heavy dep is Pillow. numpy is used *if present* for the JPEG-blockiness
estimate but is not required (the metric is simply omitted when numpy is unavailable).

Contract:
    load_normalize(input_path, run_dir, cfg=None, report=None)
        -> (normalized_path, {'w','h','scale','orig_w','orig_h', ...})

Artifacts written to run_dir:
    original.png     decoded source, re-encoded losslessly (EXIF transpose applied)
    normalized.png   sRGB RGB, min_edge <= long edge <= max_edge
    normalize.json   durable per-stage record: scale, color/alpha handling, blockiness,
                     frame count, and any degradations (QA/archetype may adapt thresholds)
"""
from __future__ import annotations
import io
import json
import os
from typing import Optional, Tuple

MAX_EDGE_DEFAULT = 2048
MIN_EDGE_DEFAULT = 512
MAX_UPSCALE_DEFAULT = 4.0
MAX_INPUT_MEGAPIXELS_DEFAULT = 200
_ALPHA_OPAQUE = 128


def _require_pillow():
    try:
        from PIL import Image, ImageCms, ImageOps  # noqa: F401
        return Image, ImageOps, ImageCms
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            "normalize.py requires Pillow. Install it with:  pip install pillow"
        ) from e


def _atomic_json(path: str, value: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _estimate_bg_color(rgba) -> Tuple[int, int, int]:
    """Estimate a neutral fill from the opaque border pixels of a small thumbnail.

    Ads are usually laid over a solid field, so the (mostly-opaque) frame around the
    canvas is the best guess for what transparent regions were meant to sit on. Falls
    back to white when the border is itself transparent. Pure PIL — no numpy needed.
    """
    thumb = rgba.copy()
    thumb.thumbnail((64, 64))
    px = thumb.load()
    tw, th = thumb.size
    coords = set()
    for x in range(tw):
        coords.add((x, 0))
        coords.add((x, th - 1))
    for y in range(th):
        coords.add((0, y))
        coords.add((tw - 1, y))
    r = g = b = n = 0
    for (x, y) in coords:
        pr, pg, pb, pa = px[x, y]
        if pa >= _ALPHA_OPAQUE:
            r += pr
            g += pg
            b += pb
            n += 1
    if n == 0:
        return (255, 255, 255)
    return (round(r / n), round(g / n), round(b / n))


def _flatten_alpha(rgba, bg, Image):
    base = Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.split()[-1])
    return base


def _estimate_blockiness(rgb_img) -> Optional[float]:
    """Cheap 8px-grid blockiness proxy for recompression/JPEG awareness.

    Returns the *relative* excess of luma gradient energy on the 8-pixel block lattice
    versus off it. ~0 means clean; larger values indicate visible JPEG blocking. Metadata
    only — never used to filter pixels. Computed on a centred, 8px-aligned crop so cost is
    bounded while the source JPEG lattice is preserved. Omitted (None) without numpy.
    """
    try:
        import numpy as np
    except Exception:
        return None
    w, h = rgb_img.size
    if w < 16 or h < 16:
        return None
    cw = min(w, 1536) & ~7
    ch = min(h, 1536) & ~7
    if cw < 16 or ch < 16:
        return None
    x0 = ((w - cw) // 2) & ~7
    y0 = ((h - ch) // 2) & ~7
    crop = rgb_img.crop((x0, y0, x0 + cw, y0 + ch)).convert("L")
    a = np.asarray(crop, dtype=np.float32)
    dh = np.abs(np.diff(a, axis=1))
    cols = np.arange(dh.shape[1]) % 8
    dv = np.abs(np.diff(a, axis=0))
    rows = np.arange(dv.shape[0]) % 8
    on = np.concatenate([dh[:, cols == 7].ravel(), dv[rows == 7, :].ravel()])
    off = np.concatenate([dh[:, cols != 7].ravel(), dv[rows != 7, :].ravel()])
    if on.size == 0 or off.size == 0:
        return None
    on_m = float(on.mean())
    off_m = float(off.mean())
    if off_m <= 1e-6:
        return 0.0
    return round(max(0.0, (on_m - off_m) / off_m), 4)


def _colour_manage(im, icc, Image, ImageCms):
    """Return an sRGB RGB image plus a record of what colour/alpha handling was applied.

    Order matters: palette -> flatten alpha over an estimated neutral field -> ICC->sRGB.
    A bad/unsupported profile or an odd source mode degrades to a plain RGB convert with a
    recorded note rather than crashing or silently shifting colour.
    """
    notes: list[str] = []
    icc_applied = False
    alpha_flattened = False
    bg_color: Optional[Tuple[int, int, int]] = None
    mode = im.mode

    if mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        mode = im.mode

    if mode in ("RGBA", "LA", "PA") or (mode == "RGBa"):
        rgba = im.convert("RGBA")
        bg_color = _estimate_bg_color(rgba)
        im = _flatten_alpha(rgba, bg_color, Image)
        alpha_flattened = True
        notes.append(f"flattened transparency onto estimated background rgb{bg_color}")
        mode = "RGB"

    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
            icc_applied = True
            mode = "RGB"
        except Exception as exc:
            notes.append(
                f"embedded ICC profile could not be applied ({exc}); colors are approximate"
            )
            if mode == "CMYK":
                notes.append("CMYK converted to sRGB without a usable profile; colors are approximate")
            if mode != "RGB":
                im = im.convert("RGB")
                mode = "RGB"
    elif mode != "RGB":
        if mode == "CMYK":
            notes.append("CMYK image has no embedded profile; converted to sRGB with approximate colors")
        im = im.convert("RGB")
        mode = "RGB"

    if im.mode != "RGB":
        im = im.convert("RGB")
    return im, {
        "icc_applied": icc_applied,
        "alpha_flattened": alpha_flattened,
        "background_color": list(bg_color) if bg_color is not None else None,
        "notes": notes,
    }


def load_normalize(
    input_path: str,
    run_dir: str,
    cfg: Optional[dict] = None,
    report=None,
) -> Tuple[str, dict]:
    Image, ImageOps, ImageCms = _require_pillow()
    from PIL import UnidentifiedImageError

    cfg = cfg or {}
    ncfg = cfg.get("normalize") or {}
    max_edge = int(ncfg.get("max_edge", MAX_EDGE_DEFAULT))
    min_edge = int(ncfg.get("min_edge", MIN_EDGE_DEFAULT))
    max_upscale = float(ncfg.get("max_upscale", MAX_UPSCALE_DEFAULT))
    max_input_mp = float(ncfg.get("max_input_megapixels", MAX_INPUT_MEGAPIXELS_DEFAULT))
    cap_px = int(max_input_mp * 1_000_000)

    os.makedirs(run_dir, exist_ok=True)

    if not os.path.isfile(input_path):
        raise ValueError(f"Input image not found: {input_path}")
    if os.path.getsize(input_path) <= 0:
        raise ValueError(f"Input image is empty (0 bytes): {input_path}. Re-download or re-export the source.")

    notes: list[dict] = []

    def _note(reason: str) -> None:
        item = {"component": "normalize", "reason": reason}
        if item not in notes:
            notes.append(item)
        if report is not None:
            try:
                report.degraded("normalize", reason)
            except Exception:
                pass

    # We enforce our own megapixel cap (below) from the header, so disable Pillow's global
    # decompression-bomb guard for the duration of this decode and restore it afterwards.
    prev_bomb_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        try:
            opened = Image.open(input_path)
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Unrecognized or unsupported image format: {input_path}. "
                "Supported inputs are PNG, JPEG, WebP, GIF, BMP and TIFF. "
                "AVIF/HEIC and other formats must be converted to PNG or JPEG first."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Could not open image {input_path}: {exc}. The file may be unreadable or corrupt."
            ) from exc

        with opened as im:
            hdr_w, hdr_h = im.size
            if hdr_w * hdr_h > cap_px:
                raise ValueError(
                    f"Input image is extremely large: {hdr_w}x{hdr_h} "
                    f"({hdr_w * hdr_h / 1e6:.0f} MP), above the {max_input_mp:.0f} MP safety limit. "
                    "Crop or downscale the source before running."
                )

            src_format = (im.format or "").upper() or "UNKNOWN"
            src_mode = im.mode
            icc = im.info.get("icc_profile")
            n_frames = int(getattr(im, "n_frames", 1) or 1)

            try:
                if n_frames > 1:
                    im.seek(0)
                    _note(
                        f"animated {src_format or 'image'} with {n_frames} frames; "
                        "used frame 0 only (motion is dropped)"
                    )
                im.load()
                # honour camera / screenshot orientation before anything measures geometry
                im = ImageOps.exif_transpose(im)
            except OSError as exc:
                raise ValueError(
                    f"Input image is truncated or corrupt and could not be fully decoded: "
                    f"{input_path} ({exc}). Re-download or re-export the original ad and try again."
                ) from exc

            orig_w, orig_h = im.size

            # verbatim (post-transpose) provenance copy — naive RGB, pre colour-management
            original_path = os.path.join(run_dir, "original.png")
            # Avoid an extra full-frame buffer when the decoded source is already RGB.
            (im if im.mode == "RGB" else im.convert("RGB")).save(original_path, format="PNG")

            # sRGB RGB, alpha flattened over an estimated neutral field
            im, cm = _colour_manage(im, icc, Image, ImageCms)
            for reason in cm["notes"]:
                _note(reason)

            # recompression / JPEG-blocking awareness (metadata only) on the source grid
            blockiness = _estimate_blockiness(im)

            # size policy: cap huge inputs; for tiny inputs warn (and optionally upscale).
            # Upscaling only interpolates — it can hurt as much as help — so it is opt-in via
            # ``normalize.upscale_tiny``. The warning is always recorded either way, and the
            # downscale contract (below) is unchanged so downstream coordinates still map back.
            long_edge = max(orig_w, orig_h)
            scale = 1.0
            resample = "none"
            if long_edge > max_edge:
                scale = max_edge / float(long_edge)
                resample = "lanczos-downscale"
            elif long_edge < min_edge:
                if bool(ncfg.get("upscale_tiny", False)):
                    target_scale = min(min_edge / float(long_edge), max_upscale)
                    if target_scale > 1.0:
                        scale = target_scale
                        resample = "lanczos-upscale"
                        reached = round(long_edge * scale)
                        if reached < min_edge:
                            _note(
                                f"input is very small ({orig_w}x{orig_h}); upscaled {scale:.2f}x to a "
                                f"{reached}px long edge (capped at {max_upscale:g}x) but remains below "
                                f"the {min_edge}px target — downstream OCR/detection accuracy may be reduced"
                            )
                        else:
                            _note(
                                f"input is small ({orig_w}x{orig_h}); upscaled {scale:.2f}x to a "
                                f"{min_edge}px long edge — pixels are interpolated, not original detail"
                            )
                else:
                    _note(
                        f"input is small ({orig_w}x{orig_h}); long edge {long_edge}px is below the "
                        f"{min_edge}px working target — downstream OCR/detection accuracy may be reduced "
                        "(set normalize.upscale_tiny to interpolate it up)"
                    )
            if scale != 1.0:
                new_w = max(1, round(orig_w * scale))
                new_h = max(1, round(orig_h * scale))
                im = im.resize((new_w, new_h), Image.LANCZOS)

            w, h = im.size
            normalized_path = os.path.join(run_dir, "normalized.png")
            im.save(normalized_path, format="PNG")
    finally:
        Image.MAX_IMAGE_PIXELS = prev_bomb_limit

    meta = {
        "w": w,
        "h": h,
        "scale": round(scale, 6),
        "orig_w": orig_w,
        "orig_h": orig_h,
        "source_format": src_format,
        "source_mode": src_mode,
        "frames": n_frames,
        "resample": resample,
        "icc_applied": cm["icc_applied"],
        "alpha_flattened": cm["alpha_flattened"],
        "background_color": cm["background_color"],
        "blockiness": blockiness,
        "degraded": list(notes),
    }

    record = {
        "artifact_version": 1,
        "input": os.path.abspath(input_path),
        "megapixels": round(orig_w * orig_h / 1e6, 3),
        **meta,
    }
    _atomic_json(os.path.join(run_dir, "normalize.json"), record)

    return normalized_path, meta


if __name__ == "__main__":  # smoke: CPU-safe, synthesizes an input if none given
    import sys
    import tempfile

    Image, _, _ = _require_pillow()
    run_dir = tempfile.mkdtemp(prefix="normalize_smoke_")
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = os.path.join(run_dir, "_synthetic.png")
        Image.new("RGB", (3000, 1500), (200, 60, 60)).save(src)
    out, meta = load_normalize(src, run_dir)
    print("normalized ->", out)
    print("meta        ->", meta)
