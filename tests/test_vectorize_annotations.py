"""Feature 1 vector-side tests: annotation stroke roles trace to editable stroke geometry."""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import vectorize


def _gate_available():
    return vectorize._rasterize_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
        '<path d="M0 0 L8 8" stroke="#000" stroke-width="1"/></svg>', 8, 8) is not None


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


def _star_crop(path, n=12, w=160, h=160, ro=70, ri=40, fill=(120, 130, 40, 255)):
    """H11 olive starburst seal (regular N-point star on a transparent bg)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2
    pts = []
    for i in range(n):
        ao = -math.pi / 2 + 2 * math.pi * i / n
        ai = -math.pi / 2 + 2 * math.pi * (i + 0.5) / n
        pts.append((cx + ro * math.cos(ao), cy + ro * math.sin(ao)))
        pts.append((cx + ri * math.cos(ai), cy + ri * math.sin(ai)))
    d.polygon(pts, fill=fill)
    img.save(path)


def test_starburst_seal_fits_native_star_polygon_not_wobbly_trace(tmp_path):
    # H11: a starburst/sunburst seal badge must become a clean regular-star primitive
    # (a Figma star: point count + two radii), never a dozen-Bezier VTracer path.
    if not _gate_available():
        pytest.skip("no SVG rasterizer (cairosvg/resvg) for the vector render-back gate")
    source = tmp_path / "starburst.png"
    _star_crop(source, n=12)

    result = vectorize.vectorize_crop(str(source), {}, role="starburst")

    assert result["ok"] is True
    assert result["engine"] == "analytic-star"
    prim = result["primitive"]
    assert prim["kind"] == "star"
    assert prim["points"] == 12
    assert prim["r_outer"] > prim["r_inner"] > 0
    assert prim["iou"] >= 0.90
    assert len(result["paths"]) == 1  # one native star, not stacked trace fragments


def test_scribble_strike_stays_editable_vector_not_raster(tmp_path):
    # H17: a hand-drawn scribble strikethrough crossing text must be emitted as an
    # editable vector (looser gate for irregular marks), not a raster fallback. The
    # decorated text stays native; only this mark is vectorized.
    if not _gate_available():
        pytest.skip("no SVG rasterizer (cairosvg/resvg) for the vector render-back gate")
    w, h = 220, 60
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for off in (26, 32):
        pts = [(x, off + 6 * math.sin(x / 12.0)) for x in range(6, w - 6, 3)]
        d.line(pts, fill=(210, 30, 30, 255), width=4, joint="curve")
    source = tmp_path / "scribble.png"
    img.save(source)

    result = vectorize.vectorize_crop(str(source), {}, role="scribble_strikethrough")

    assert result["ok"] is True
    assert result["engine"] != "none"  # a real vector engine, not a raster escape hatch
    assert result["paths"]  # editable path geometry emitted
