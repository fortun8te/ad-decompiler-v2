"""Tests for the shipped richer-reconstruction features (audit YELLOW -> GREEN):

  Feature 1  hand-drawn annotation -> editable vector stroke (not a raster slice)
  Feature 3  multi-stop linear + radial gradients on shapes and backgrounds

(Feature 2, glassmorphism / BACKGROUND_BLUR, was dropped from scope — detection was
unreliable — so it is not implemented or tested here.)
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import reconstruct, vectorize
from src.reconstruct import _extract_shape_style, _multistop_linear_gradient_fill


# ── Feature 3: multi-stop + radial gradients ───────────────────────────────────────
def _linear_ramp_crop(size=(160, 60), stops=((0.0, (220, 20, 20)),
                                              (0.5, (250, 250, 250)),
                                              (1.0, (20, 20, 220)))):
    """A piecewise-linear multi-hue horizontal ramp; a 2-stop plane cannot fit it."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.float32)
    positions = [p for p, _ in stops]
    colors = np.array([c for _, c in stops], dtype=np.float32)
    xs = np.linspace(0.0, 1.0, w)
    for ch in range(3):
        arr[:, :, ch] = np.interp(xs, positions, colors[:, ch])[None, :]
    return arr.astype(np.uint8)


def test_multistop_linear_gradient_fit_emits_three_stops():
    rgb = _linear_ramp_crop()
    interior = np.ones(rgb.shape[:2], dtype=bool)
    fill = _multistop_linear_gradient_fill(rgb, interior)
    assert fill is not None, "multi-hue ramp should fit a >2-stop linear gradient"
    assert fill["kind"] == "linear"
    assert len(fill["stops"]) >= 3
    assert fill["meta"]["multistop"] is True
    # The near-white waypoint between red and blue must survive as an interior stop; a plain
    # 2-stop red->blue ramp would never reproduce it.
    def _minch(hex_color):
        return min(int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))
    interior_stops = fill["stops"][1:-1]
    assert any(_minch(s["color"]) > 200 for s in interior_stops)


def test_multistop_declines_a_plain_two_stop_ramp():
    rgb = _linear_ramp_crop(stops=((0.0, (220, 20, 20)), (1.0, (20, 20, 220))))
    interior = np.ones(rgb.shape[:2], dtype=bool)
    assert _multistop_linear_gradient_fill(rgb, interior) is None


def test_extract_shape_style_upgrades_to_multistop_linear():
    rgb = _linear_ramp_crop(size=(200, 90))
    mask = np.full((90, 200), 255, dtype=np.uint8)
    box = {"x": 0, "y": 0, "w": 200, "h": 90}
    style = _extract_shape_style(rgb, mask, box, {}, role="panel")
    assert style is not None
    assert style["fill"]["kind"] == "linear"
    assert len(style["fill"]["stops"]) >= 3


def _radial_glow(size=256, inner=(255, 250, 235), outer=(20, 20, 60)):
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = np.clip(np.hypot(xx - cx, yy - cy) / float(np.hypot(cx, cy)), 0.0, 1.0)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(3):
        arr[:, :, ch] = inner[ch] * (1 - t) + outer[ch] * t
    return arr.astype(np.uint8)


def test_extract_background_gradient_detects_radial_glow(tmp_path):
    plate = tmp_path / "plate.png"
    Image.fromarray(_radial_glow()).save(plate)
    fill = reconstruct.extract_background_gradient(str(plate), {})
    assert fill is not None
    assert fill["kind"] == "radial"
    assert fill["meta"]["background"] is True
    assert fill["meta"]["reconstruction_mae"] <= 6.0


def test_extract_background_gradient_rejects_texture(tmp_path):
    plate = tmp_path / "noise.png"
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(plate)
    assert reconstruct.extract_background_gradient(str(plate), {}) is None


def test_build_emits_editable_background_gradient_layer(tmp_path):
    from src import build_design_json
    plate = tmp_path / "background_clean.png"
    Image.fromarray(_radial_glow()).save(plate)
    doc = build_design_json.build([], {"w": 256, "h": 256}, str(tmp_path), base_src=str(plate),
                                  doc_id="bg", name="bg")
    layers = {layer.id: layer for layer in doc.layers}
    assert "background" in layers and layers["background"].type == "image"  # raster floor kept
    grad = layers.get("background-gradient")
    assert grad is not None and grad.type == "shape"
    assert grad.fill["kind"] == "radial"
    # It sits just above the raster plate, and role=background keeps leaf accounting honest.
    assert grad.z_index > layers["background"].z_index
    assert grad.meta.get("role") == "background"


def test_build_leaves_flat_plate_as_raster_only(tmp_path):
    from src import build_design_json
    plate = tmp_path / "background_clean.png"
    Image.new("RGB", (300, 300), (200, 60, 40)).save(plate)
    doc = build_design_json.build([], {"w": 300, "h": 300}, str(tmp_path), base_src=str(plate),
                                  doc_id="flat", name="flat")
    assert not any(layer.id == "background-gradient" for layer in doc.layers)


def test_extract_background_gradient_skips_small_and_disabled(tmp_path):
    small = tmp_path / "tiny.png"
    Image.fromarray(_radial_glow(size=100)).save(small)
    assert reconstruct.extract_background_gradient(str(small), {}) is None  # below min dim
    big = tmp_path / "big.png"
    Image.fromarray(_radial_glow()).save(big)
    disabled = {"reconstruct": {"background_gradient": {"enabled": False}}}
    assert reconstruct.extract_background_gradient(str(big), disabled) is None


# ── Feature 1: annotation role -> editable vector stroke through reconstruct ────────
def _rasterizer_available():
    return vectorize._rasterize_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
        '<path d="M0 0 L8 8" stroke="#000" stroke-width="1"/></svg>', 8, 8) is not None


def test_annotation_role_becomes_editable_vector_not_raster(tmp_path):
    if not _rasterizer_available():
        pytest.skip("no SVG rasterizer (cairosvg/resvg) for the vector render-back gate")
    source = tmp_path / "underline.png"
    image = Image.new("RGB", (140, 70), (245, 245, 245))
    ImageDraw.Draw(image).rectangle((20, 40, 118, 45), fill=(210, 30, 30))  # marker underline
    image.save(source)
    Image.new("L", (100, 8), 255).save(tmp_path / "mark-mask.png")
    candidates = [{
        "id": "mark", "target": "icon",
        "box": {"x": 20, "y": 39, "w": 100, "h": 8},
        "mask": {"src": "mark-mask.png"},
        "meta": {"role": "underline"},
    }]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"inpaint": {"mode": "opencv", "mask_dilate": 0}},
    )
    mark = {c["id"]: c for c in result["candidates"]}["mark"]
    assert mark["target"] == "icon"                       # stayed a vector, not raster fallback
    assert mark["meta"]["vectorize"]["ok"] is True
    assert mark.get("svg") and "<svg" in mark["svg"]
    assert mark.get("stroke") and mark["stroke"].get("color")  # recolorable stroke emitted
    assert mark["meta"].get("annotation_stroke")
    assert result["stats"]["vectorized"] >= 1
