"""CPU tests for stage-1 input normalization (src/normalize.py).

Every fixture is synthesized on the fly with Pillow — no binary assets on disk. Covers the
messy real-world input classes normalize must survive: EXIF-rotated screenshots, palette
PNGs with alpha, animated GIFs, ICC-tagged exports, CMYK, huge/tiny canvases, and genuinely
broken files that must fail loud with an actionable message.
"""
import io
import json
import os

import pytest
from PIL import Image, ImageCms

from src.normalize import load_normalize


# --------------------------------------------------------------------------- helpers

def _open_rgb(path):
    with Image.open(path) as im:
        assert im.mode == "RGB"
        return im.size


def _srgb_icc_bytes():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _read_sidecar(run_dir):
    with open(os.path.join(run_dir, "normalize.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- basics

def test_plain_rgb_passthrough_and_sidecar(tmp_path):
    src = tmp_path / "plain.png"
    Image.new("RGB", (800, 600), (10, 120, 200)).save(src)
    run = tmp_path / "run"

    out, meta = load_normalize(str(src), str(run))

    assert os.path.basename(out) == "normalized.png"
    assert meta["w"] == 800 and meta["h"] == 600
    assert meta["orig_w"] == 800 and meta["orig_h"] == 600
    assert meta["scale"] == 1.0
    assert meta["resample"] == "none"
    assert _open_rgb(out) == (800, 600)
    # provenance + durable per-stage record are both written
    assert (run / "original.png").exists()
    sidecar = _read_sidecar(str(run))
    assert sidecar["w"] == 800 and sidecar["scale"] == 1.0
    assert sidecar["input"] == os.path.abspath(str(src))


def test_blockiness_is_recorded_as_number(tmp_path):
    src = tmp_path / "photo.png"
    # gradient so off-grid gradient energy is non-zero and the metric is well defined
    img = Image.new("RGB", (256, 256))
    img.putdata([((x + y) % 256, x % 256, y % 256) for y in range(256) for x in range(256)])
    img.save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert isinstance(meta["blockiness"], float)
    assert meta["blockiness"] >= 0.0


# --------------------------------------------------------------------------- EXIF

def test_exif_orientation_is_applied(tmp_path):
    # 40x20 landscape stored with orientation=6 (rotate 90) must decode as 20x40 portrait.
    src = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = img.getexif()
    exif[274] = 6
    img.save(src, format="JPEG", exif=exif.tobytes())
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert (meta["orig_w"], meta["orig_h"]) == (20, 40)


# --------------------------------------------------------------------------- alpha / palette

def test_palette_png_with_alpha_is_flattened(tmp_path):
    base = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    # opaque red block on a transparent field
    for y in range(50, 150):
        for x in range(50, 250):
            base.putpixel((x, y), (220, 40, 40, 255))
    pal = base.convert("P", palette=Image.ADAPTIVE, colors=16)
    src = tmp_path / "pal.png"
    pal.save(src, transparency=0)
    run = tmp_path / "run"

    out, meta = load_normalize(str(src), str(run))
    assert meta["alpha_flattened"] is True
    assert meta["background_color"] is not None and len(meta["background_color"]) == 3
    assert _open_rgb(out)  # decodes as RGB
    assert any("transparency" in d["reason"] for d in meta["degraded"])


def test_rgba_alpha_flatten_uses_estimated_background(tmp_path):
    img = Image.new("RGBA", (200, 200), (12, 200, 90, 255))  # opaque green border
    for y in range(60, 140):
        for x in range(60, 140):
            img.putpixel((x, y), (0, 0, 0, 0))  # transparent hole in the middle
    src = tmp_path / "rgba.png"
    img.save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert meta["alpha_flattened"] is True
    r, g, b = meta["background_color"]
    assert g > r and g > b  # neutral estimate picked up the green border, not default white


# --------------------------------------------------------------------------- animated

def test_animated_gif_takes_frame_zero_with_note(tmp_path):
    f0 = Image.new("RGB", (320, 240), (255, 0, 0))
    f1 = Image.new("RGB", (320, 240), (0, 0, 255))
    src = tmp_path / "anim.gif"
    f0.save(src, save_all=True, append_images=[f1], duration=100, loop=0)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert meta["frames"] == 2
    assert any("frame 0" in d["reason"] for d in meta["degraded"])


# --------------------------------------------------------------------------- ICC / CMYK

def test_icc_profile_is_converted(tmp_path):
    src = tmp_path / "icc.jpg"
    Image.new("RGB", (600, 400), (90, 90, 90)).save(src, format="JPEG", icc_profile=_srgb_icc_bytes())
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert meta["icc_applied"] is True
    assert _open_rgb(str(run / "normalized.png"))


def test_cmyk_without_profile_is_converted_and_noted(tmp_path):
    src = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (600, 400), (0, 200, 200, 10)).save(src, format="JPEG")
    run = tmp_path / "run"

    out, meta = load_normalize(str(src), str(run))
    assert meta["source_mode"] == "CMYK"
    assert _open_rgb(out)
    assert any("CMYK" in d["reason"] for d in meta["degraded"])


# --------------------------------------------------------------------------- size policy

def test_huge_input_is_downscaled_and_scale_recorded(tmp_path):
    src = tmp_path / "huge.png"
    Image.new("RGB", (5000, 2500), (30, 30, 30)).save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert meta["w"] == 2048 and meta["h"] == 1024
    assert meta["orig_w"] == 5000 and meta["orig_h"] == 2500
    assert 0 < meta["scale"] < 1
    assert meta["resample"] == "lanczos-downscale"
    # scale contract: normalized = round(orig * scale) so downstream can map back
    assert round(meta["orig_w"] * meta["scale"]) == meta["w"]


def test_small_input_warns_without_resizing_by_default(tmp_path):
    # Upscaling only interpolates, so it is opt-in: by default a tiny input is passed
    # through at native size with a recorded warning (never silently resized).
    src = tmp_path / "small.png"
    Image.new("RGB", (200, 120), (200, 200, 200)).save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run))
    assert meta["scale"] == 1.0
    assert meta["resample"] == "none"
    assert (meta["w"], meta["h"]) == (200, 120)
    assert any("below the" in d["reason"] for d in meta["degraded"])


def test_tiny_input_is_upscaled_when_opted_in(tmp_path):
    src = tmp_path / "tiny.png"
    Image.new("RGB", (200, 120), (200, 200, 200)).save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run), cfg={"normalize": {"upscale_tiny": True}})
    assert meta["scale"] > 1.0
    assert meta["resample"] == "lanczos-upscale"
    assert max(meta["w"], meta["h"]) >= 512
    assert any("upscal" in d["reason"] for d in meta["degraded"])


def test_very_small_input_is_capped_and_flagged_below_target(tmp_path):
    src = tmp_path / "thumb.png"
    Image.new("RGB", (40, 30), (5, 5, 5)).save(src)
    run = tmp_path / "run"

    _, meta = load_normalize(str(src), str(run), cfg={"normalize": {"upscale_tiny": True}})
    # capped at 4x, so still below the 512 target -> explicit reduced-accuracy warning
    assert meta["scale"] == pytest.approx(4.0, rel=1e-6)
    assert any("below the" in d["reason"] for d in meta["degraded"])


# --------------------------------------------------------------------------- fail loud

def test_truncated_file_fails_loud(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (123, 222, 111)).save(buf, format="JPEG")
    data = buf.getvalue()
    src = tmp_path / "broken.jpg"
    src.write_bytes(data[: len(data) // 2])  # header intact, scan data cut off
    run = tmp_path / "run"

    with pytest.raises(ValueError) as exc:
        load_normalize(str(src), str(run))
    assert "truncated" in str(exc.value).lower()


def test_unrecognized_format_fails_loud(tmp_path):
    src = tmp_path / "notimage.bin"
    src.write_bytes(b"this is definitely not an image file" * 40)
    run = tmp_path / "run"

    with pytest.raises(ValueError) as exc:
        load_normalize(str(src), str(run))
    assert "unrecognized" in str(exc.value).lower() or "unsupported" in str(exc.value).lower()


def test_empty_file_fails_loud(tmp_path):
    src = tmp_path / "empty.png"
    src.write_bytes(b"")
    run = tmp_path / "run"

    with pytest.raises(ValueError) as exc:
        load_normalize(str(src), str(run))
    assert "empty" in str(exc.value).lower()


def test_missing_file_fails_loud(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_normalize(str(tmp_path / "nope.png"), str(tmp_path / "run"))
    assert "not found" in str(exc.value).lower()


def test_oversized_input_is_rejected(tmp_path):
    # tiny image but a deliberately tiny megapixel cap makes the header check fire
    src = tmp_path / "big.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(src)
    run = tmp_path / "run"

    with pytest.raises(ValueError) as exc:
        load_normalize(str(src), str(run), cfg={"normalize": {"max_input_megapixels": 0.01}})
    assert "large" in str(exc.value).lower()


# --------------------------------------------------------------------------- report wiring

def test_report_receives_normalize_degradations(tmp_path):
    from src.run_report import RunReport

    src = tmp_path / "rgba.png"
    Image.new("RGBA", (300, 300), (255, 255, 255, 0)).save(src)  # fully transparent
    run = tmp_path / "run"
    run.mkdir()  # RunReport writes into an existing run dir (run_one makes it first)
    report = RunReport(str(run), str(src), {}, "normalize")

    load_normalize(str(src), str(run), report=report)

    components = {d["component"] for d in report.data["degraded"]}
    assert "normalize" in components


# --------------------------------------------------------------------------- webp

def test_webp_input_is_handled(tmp_path):
    src = tmp_path / "ad.webp"
    Image.new("RGB", (900, 700), (44, 88, 132)).save(src, format="WEBP")
    run = tmp_path / "run"

    out, meta = load_normalize(str(src), str(run))
    assert meta["source_format"] == "WEBP"
    assert _open_rgb(out) == (900, 700)
