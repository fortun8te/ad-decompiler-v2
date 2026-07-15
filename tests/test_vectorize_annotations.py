"""Feature 1 vector-side tests: annotation stroke roles trace to editable stroke geometry."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import vectorize


def test_annotation_roles_have_gate_limits():
    # The stroke annotation roles must be tuned (not fall to the strict 0.85 default),
    # otherwise a couple of aliased edge pixels force a raster slice.
    for role in ("underline", "strikethrough", "connector", "callout_leader", "leader"):
        score, paths = vectorize._gate_limits(role, {})
        assert score <= 0.80
        assert paths >= 14
    # Config still wins over the defaults.
    cfg = {"vectorize": {"score_min": {"underline": 0.70}, "max_paths": {"underline": 99}}}
    score, paths = vectorize._gate_limits("underline", cfg)
    assert score == 0.70 and paths == 99


def test_analytic_straight_line_emits_stroke_for_underline(tmp_path):
    crop = tmp_path / "underline.png"
    img = Image.new("RGBA", (120, 16), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((4, 6, 115, 10), fill=(200, 30, 30, 255))
    img.save(crop)
    svg = vectorize._analytic_straight_line_svg(str(crop), "underline")
    assert svg is not None
    assert "stroke=" in svg and 'fill="none"' in svg
    # The stroke style parses out for downstream (color + width), not a filled blob.
    paths = vectorize._parse_svg_paths(svg)
    assert len(paths) == 1
    assert paths[0]["fill"] == "none"
    assert paths[0]["stroke"]["color"].startswith("#")
    assert paths[0]["stroke"]["width"] > 0


def test_vectorize_crop_underline_is_editable_stroke_not_raster(tmp_path):
    if vectorize._rasterize_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
        '<path d="M0 0 L8 8" stroke="#000" stroke-width="1"/></svg>', 8, 8) is None:
        pytest.skip("no SVG rasterizer (cairosvg/resvg) for the vector render-back gate")
    arr = np.zeros((16, 120, 4), dtype=np.uint8)
    arr[6:11, 4:116] = (200, 30, 30, 255)
    result = vectorize.vectorize_crop(arr, {}, role="underline")
    assert result["ok"] is True
    assert result["engine"] == "analytic-line"
    assert result["paths"][0]["stroke"]["color"].startswith("#")
