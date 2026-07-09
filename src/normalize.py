"""normalize.py — stage 1: load + normalize an input ad into a clean sRGB PNG.

Opens the source, applies EXIF orientation, converts to sRGB RGB, caps the long
edge at 2048 px (aspect preserved), and writes `original.png` (verbatim copy of the
decoded source) + `normalized.png` under the run dir. Records the downscale factor so
downstream coordinates can be mapped back to the true source resolution if needed.

Pure CPU. Only heavy dep is Pillow (+ numpy is optional, not required here).

Contract:
    load_normalize(input_path, run_dir, cfg=None) -> (normalized_path, {'w','h','scale','orig_w','orig_h'})

Artifacts written to run_dir:
    original.png     decoded source, re-encoded losslessly (EXIF transpose applied)
    normalized.png   sRGB RGB, long edge <= max_edge
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

MAX_EDGE_DEFAULT = 2048


def _require_pillow():
    try:
        from PIL import Image, ImageCms, ImageOps  # noqa: F401
        return Image, ImageOps, ImageCms
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            "normalize.py requires Pillow. Install it with:  pip install pillow"
        ) from e


def _to_srgb(img, Image, ImageCms):
    """Convert an image to sRGB RGB, honoring an embedded ICC profile when present."""
    icc = img.info.get("icc_profile")
    if icc:
        try:
            import io
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst_profile = ImageCms.createProfile("sRGB")
            # convert in whatever mode, output RGB
            img = ImageCms.profileToProfile(
                img, src_profile, dst_profile, outputMode="RGB"
            )
            return img
        except Exception:
            # bad/unsupported profile — fall through to a plain RGB convert
            pass
    if img.mode != "RGB":
        # flatten alpha onto white so a transparent source doesn't go black
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            return bg
        return img.convert("RGB")
    return img


def load_normalize(
    input_path: str, run_dir: str, cfg: Optional[dict] = None
) -> Tuple[str, dict]:
    Image, ImageOps, ImageCms = _require_pillow()
    cfg = cfg or {}
    max_edge = int((cfg.get("normalize") or {}).get("max_edge", MAX_EDGE_DEFAULT))

    os.makedirs(run_dir, exist_ok=True)

    with Image.open(input_path) as im:
        im.load()
        # 1. respect camera orientation
        im = ImageOps.exif_transpose(im)
        orig_w, orig_h = im.size

        # 2. verbatim (post-transpose) copy for provenance
        original_path = os.path.join(run_dir, "original.png")
        im.convert("RGB").save(original_path, format="PNG")

        # 3. sRGB RGB
        im = _to_srgb(im, Image, ImageCms)

        # 4. cap long edge, preserve aspect
        long_edge = max(orig_w, orig_h)
        scale = 1.0
        if long_edge > max_edge:
            scale = max_edge / float(long_edge)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

        w, h = im.size
        normalized_path = os.path.join(run_dir, "normalized.png")
        im.save(normalized_path, format="PNG")

    return normalized_path, {
        "w": w,
        "h": h,
        "scale": round(scale, 6),
        "orig_w": orig_w,
        "orig_h": orig_h,
    }


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
