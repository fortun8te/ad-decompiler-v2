"""Preview parity for Codia-style text: weight-split runs, weight-aware font
selection, and emoji drawn as color glyphs instead of tofu boxes.

See docs/CODIA-PARITY-SPEC.md section 2/2a/2b.
"""
import os

import numpy as np
import pytest
from PIL import Image

from src import render_preview

WINDIR = os.environ.get("WINDIR", r"C:\Windows")
ARIAL = os.path.join(WINDIR, "Fonts", "arial.ttf")
ARIAL_BOLD = os.path.join(WINDIR, "Fonts", "arialbd.ttf")
EMOJI = os.path.join(WINDIR, "Fonts", "seguiemj.ttf")

needs_arial = pytest.mark.skipif(not (os.path.exists(ARIAL) and os.path.exists(ARIAL_BOLD)),
                                 reason="Arial regular/bold not installed")


def _render(tmp_path, layers, size=(320, 120)):
    result = render_preview.render(
        {"canvas": {"w": size[0], "h": size[1]}, "layers": layers}, str(tmp_path))
    return Image.open(result["preview"]).convert("RGB")


def _ink(image, box):
    region = np.asarray(image.crop(box).convert("L"), dtype=np.float32)
    return float((255.0 - region).sum())


@needs_arial
def test_text_font_prefers_weight_matching_candidate():
    candidates = [
        {"family": "Inter", "weight": 400, "path": ARIAL},
        {"family": "Inter", "weight": 700, "path": ARIAL_BOLD},
    ]
    bold = render_preview._text_font(
        {"fontFamily": "Inter", "fontWeight": 700, "fontCandidates": candidates}, 24)
    light = render_preview._text_font(
        {"fontFamily": "Inter", "fontWeight": 300, "fontCandidates": candidates}, 24)
    assert str(getattr(bold, "path", "")).lower().endswith("arialbd.ttf")
    assert str(getattr(light, "path", "")).lower().endswith("arial.ttf")


@needs_arial
def test_weight_split_runs_render_with_distinct_weights(tmp_path):
    """One line, two runs: '121K'(700) then ' weergaven'(300). The bold half must
    put down visibly more ink per glyph than the light half."""
    candidates = [
        {"family": "Inter", "weight": 400, "path": ARIAL},
        {"family": "Inter", "weight": 700, "path": ARIAL_BOLD},
    ]
    layer = {
        "id": "t", "type": "text", "box": {"x": 8, "y": 30, "w": 300, "h": 40},
        "text": "HHHH HHHH",
        "style": {"fontFamily": "Inter", "fontSize": 30, "fontWeight": 300,
                  "color": "#000000", "letterSpacing": 0,
                  "fontCandidates": candidates},
        "text_runs": [
            {"start": 0, "end": 4, "style": {"fontWeight": 700,
                                             "fontCandidates": candidates}},
            {"start": 4, "end": 9, "style": {"fontWeight": 300,
                                             "fontCandidates": candidates}},
        ],
    }
    tile, _ = render_preview._text_tile(layer, (300, 40))
    columns = np.asarray(tile.convert("RGBA"))[:, :, 3].sum(axis=0)
    filled = np.nonzero(columns)[0]
    assert filled.size > 0
    mid = (filled[0] + filled[-1]) // 2
    left_ink = float(columns[:mid].sum())
    right_ink = float(columns[mid:].sum())
    # Same glyphs each side; the bold run must be clearly heavier.
    assert left_ink > right_ink * 1.15


@needs_arial
def test_single_style_layers_render_unchanged_with_valid_runs(tmp_path):
    """Runs that carry no differing style must not change the rendering path."""
    base = {
        "id": "t", "type": "text", "box": {"x": 8, "y": 30, "w": 300, "h": 40},
        "text": "hello world",
        "style": {"fontFamily": "Inter", "fontSize": 28, "color": "#000000",
                  "fontCandidates": [{"family": "Inter", "weight": 400, "path": ARIAL}]},
    }
    with_runs = dict(base)
    with_runs["text_runs"] = [{"start": 0, "end": 11, "style": {}}]
    tile_a, off_a = render_preview._text_tile(base, (300, 40))
    tile_b, off_b = render_preview._text_tile(with_runs, (300, 40))
    assert off_a == off_b
    assert np.array_equal(np.asarray(tile_a), np.asarray(tile_b))


def test_malformed_runs_fall_back_to_base_style():
    layer = {"text_runs": [{"start": 5, "end": 2}]}
    assert render_preview._run_segments(layer, "abcdef") is None
    overlapping = {"text_runs": [{"start": 0, "end": 4}, {"start": 2, "end": 6}]}
    assert render_preview._run_segments(overlapping, "abcdef") is None
    gaps = {"text_runs": [{"start": 2, "end": 4}]}
    segments = render_preview._run_segments(gaps, "abcdef")
    assert [(s, e) for s, e, _ in segments] == [(0, 2), (2, 4), (4, 6)]


@pytest.mark.skipif(not os.path.exists(EMOJI), reason="Segoe UI Emoji not installed")
def test_emoji_renders_in_color_not_tofu(tmp_path):
    image = _render(tmp_path, [{
        "id": "t", "type": "text", "box": {"x": 10, "y": 30, "w": 200, "h": 48},
        "text": "zien ⏳ je",   # hourglass pictograph mid-line
        "style": {"fontSize": 36, "color": "#202020"},
    }], size=(240, 110))
    array = np.asarray(image, dtype=np.int16)
    saturation = array.max(axis=2) - array.min(axis=2)
    # The color-emoji glyph contributes chromatic pixels; plain dark-gray text and a
    # tofu rectangle both have (near-)zero saturation.
    assert int((saturation > 40).sum()) > 10


def test_zero_width_joiners_are_skipped(tmp_path):
    # Must not raise and must not draw tofu boxes for VS-16/ZWJ.
    layer = {
        "id": "t", "type": "text", "box": {"x": 4, "y": 4, "w": 120, "h": 30},
        "text": "a‍️b", "style": {"fontSize": 20, "color": "#000000"},
    }
    tile, _ = render_preview._text_tile(layer, (120, 30))
    assert tile.width > 0
