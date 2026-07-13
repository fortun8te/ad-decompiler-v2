"""CPU-only tests for text geometry, typography, grouping, and font retrieval."""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import text_analysis  # noqa: E402


def _font_path():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    candidates += glob.glob("/usr/share/fonts/**/*DejaVuSans.ttf", recursive=True)[:1]
    return next((path for path in candidates if os.path.isfile(path)), None)


def _font(size):
    path = _font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def _line(line_id, text, bbox, image_box=None, conf=0.98):
    x0, y0, x1, y1 = image_box or bbox
    box = {"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)}
    return {
        "id": line_id,
        "text": text,
        "conf": conf,
        "box": box,
        "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "words": [],
    }


def _draw_text(draw, xy, text, font, fill):
    bbox = draw.textbbox(xy, text, font=font)
    draw.text(xy, text, font=font, fill=fill)
    # OCR boxes usually include some line-box breathing room.
    return (max(0, bbox[0] - 7), max(0, bbox[1] - 6), bbox[2] + 7, bbox[3] + 6)


def test_fit_text_box_scales_multiline_line_height_with_font(monkeypatch):
    monkeypatch.setattr(text_analysis, "_fit_font", lambda style, size: _font(max(1, int(size))))
    _, resize, patch = text_analysis.fit_text_box(
        "First long line\nSecond long line\nThird long line",
        {"fontSize": 40, "lineHeight": 52, "letterSpacing": 0},
        {"x": 0, "y": 0, "w": 180, "h": 95},
    )
    assert resize == "HEIGHT"
    assert patch["fontSize"] < 40
    assert patch["lineHeight"] < 52


def test_enriches_painted_geometry_colour_baseline_and_style(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "ad.png"
    image.save(path)
    original = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 640, "h": 260},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }
    frozen = copy.deepcopy(original)

    result = text_analysis.analyze_text(str(path), original, {})
    line = result["lines"][0]

    assert original == frozen, "analysis must not mutate upstream OCR"
    assert line["painted_box"]["w"] < line["box"]["w"]
    assert line["painted_box"]["h"] < line["box"]["h"]
    r, g, b = line["style"]["colorRGB"]
    assert r > 160 and g < 90 and b < 90
    assert line["painted_box"]["y"] <= line["baseline"]["y0"] <= (
        line["painted_box"]["y"] + line["painted_box"]["h"] + 2
    )
    assert line["rotation_deg"] == pytest.approx(0.0, abs=0.01)
    assert line["style"]["fontSize"] > line["painted_box"]["h"]
    assert line["style"]["fontCandidates"]
    assert result["blocks"] and result["styles"] and result["sections"]


def test_groups_paragraph_lines_and_reuses_style_id(tmp_path):
    image = Image.new("RGB", (720, 420), "white")
    draw = ImageDraw.Draw(image)
    headline_font, body_font = _font(48), _font(25)
    head_box = _draw_text(draw, (55, 35), "A BETTER ROUTINE", headline_font, (18, 18, 18))
    body1_box = _draw_text(draw, (58, 165), "Made for everyday use and easy styling.", body_font, (35, 35, 35))
    body2_box = _draw_text(draw, (58, 204), "Clean ingredients with a natural finish.", body_font, (35, 35, 35))
    path = tmp_path / "paragraph.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 720, "h": 420},
        "lines": [
            _line("L0", "A BETTER ROUTINE", head_box),
            _line("L1", "Made for everyday use and easy styling.", body1_box),
            _line("L2", "Clean ingredients with a natural finish.", body2_box),
        ],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    paragraph = next(block for block in result["blocks"] if block["line_ids"] == ["L1", "L2"])
    by_id = {line["id"]: line for line in result["lines"]}

    assert paragraph["type"] == "paragraph"
    assert paragraph["role"] == "body"
    assert paragraph["alignment"] == "LEFT"
    assert by_id["L0"]["role"] == "headline"
    assert by_id["L1"]["hierarchy"]["parent_id"] == paragraph["id"]
    assert by_id["L1"]["style_id"] == by_id["L2"]["style_id"]
    shared = next(style for style in result["styles"] if style["id"] == by_id["L1"]["style_id"])
    assert shared["repeated"] is True
    assert set(shared["usage"]) == {"L1", "L2"}


def test_rotation_and_missing_image_fallback_are_safe(tmp_path):
    angle = 12.0
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 60.0, 180.0, 34.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = {
        "id": "L0",
        "text": "Rotated copy",
        "conf": 0.8,
        "box": {"x": x0, "y": y0, "w": width, "h": height},
        "quad": [[x0, y0], [x0 + dx, y0 + dy],
                 [x0 + dx, y0 + dy + height], [x0, y0 + height]],
        "words": [],
    }
    missing = tmp_path / "not-there.png"
    result = text_analysis.analyze_text(
        str(missing),
        {"engine": "synthetic", "source": {"w": 400, "h": 300}, "lines": [line]},
        {},
    )
    enriched = result["lines"][0]

    assert enriched["rotation_deg"] == pytest.approx(angle, abs=0.01)
    assert enriched["painted_box"] == line["box"]
    assert enriched["style"]["fontCandidates"][0]["source"] == "fallback"
    assert result["text_analysis"]["image_available"] is False


def test_quad_rotation_uses_long_text_edge_when_ocr_starts_with_short_edge():
    # This is the winding emitted by the benchmark's horizontal lines: the
    # first edge is vertical, while the long opposite edge is horizontal.
    quad = [[10, 40], [10, 10], [210, 10], [210, 40]]
    assert text_analysis._quad_rotation(quad) == pytest.approx(0.0)


def test_paragraph_rotation_aggregates_stacked_horizontal_lines(tmp_path):
    path = tmp_path / "stack.png"
    Image.new("RGB", (500, 180), "white").save(path)
    lines = []
    for index, y in enumerate((30, 70, 110)):
        line = _line(f"L{index}", "STACKED LINE", (30, y, 260, y + 20))
        line["quad"] = [[30, y + 20], [30, y], [260, y], [260, y + 20]]
        lines.append(line)
    result = text_analysis.analyze_text(str(path), {"source": {"w": 500, "h": 180}, "lines": lines}, {})
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == pytest.approx(0.0)
    assert all(line["rotation_deg"] == pytest.approx(0.0) for line in result["lines"])


def test_single_rotated_quad_keeps_supported_angle():
    angle = 32.0
    radians = math.radians(angle)
    width, height = 180.0, 30.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    quad = [[0, height], [0, 0], [dx, dy], [dx, dy + height]]
    assert text_analysis._quad_rotation(quad) == pytest.approx(angle, abs=0.01)


def test_shear_measurement_rejects_implausible_rotation_like_drift():
    mask = np.zeros((24, 40), dtype=bool)
    mask[2:10, 4:12] = True
    mask[14:22, 25:33] = True
    assert text_analysis._measure_shear_angle(mask) is None


def test_optional_local_font_matching_is_bounded(tmp_path):
    font = _font(38)
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")

    image = Image.new("RGB", (520, 180), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (35, 45), "Font Match", font, (0, 0, 0))
    path = tmp_path / "font.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 520, "h": 180},
        "lines": [_line("L0", "Font Match", ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": [],
                "max_fonts": 1,
                "max_lines": 1,
                "top_k": 1,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    candidates = result["lines"][0]["style"]["fontCandidates"]

    assert len(candidates) == 1
    assert candidates[0]["source"] == "local-render"
    assert candidates[0]["score"] > 0.2
    assert result["text_analysis"]["font_matches_attempted"] == 1


def test_empty_font_dirs_still_uses_platform_inventory(tmp_path, monkeypatch):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    staged = tmp_path / "Platform.ttf"
    staged.write_bytes(Path(font_path).read_bytes())
    monkeypatch.setattr(text_analysis, "_platform_font_dirs", lambda: [str(tmp_path)])
    text_analysis._FONT_DISCOVERY_CACHE.clear()

    discovered = text_analysis._discover_fonts({"font_dirs": [], "scan_limit": 4})

    assert any(Path(item["path"]).name == "Platform.ttf" for item in discovered)


def test_multicolumn_paragraphs_do_not_cross_merge_by_reading_order(tmp_path):
    path = tmp_path / "columns.png"
    Image.new("RGB", (400, 220), "white").save(path)
    raw = {
        "source": {"w": 400, "h": 220},
        "lines": [
            _line("L0", "Left first sentence stays", (20, 50, 170, 66)),
            _line("L1", "Right first sentence stays", (230, 50, 380, 66)),
            _line("L2", "Left second sentence follows", (20, 73, 170, 89)),
            _line("L3", "Right second sentence follows", (230, 73, 380, 89)),
        ],
    }

    result = text_analysis.analyze_text(
        str(path), raw, {"text_analysis": {"font_matching": {"enabled": False}}}
    )

    members = {tuple(block["line_ids"]) for block in result["blocks"]}
    assert members == {("L0", "L2"), ("L1", "L3")}


# ---------------------------------------------------------------------------
# Bug 1: confidence/fidelity gate — low-confidence ink -> masked-pixel fallback


def test_low_ink_confidence_flags_low_fidelity_and_saves_fallback_crop(tmp_path):
    # Very low text/background contrast produces a genuine (but low-confidence) ink
    # mask — this is the "can't be faithfully represented" case the fidelity gate
    # exists to catch, and it should still have real pixels to fall back to.
    font = _font(50)
    image = Image.new("RGB", (300, 140), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (30, 30), "SALE", font, (188, 188, 188))
    path = tmp_path / "faint.png"
    image.save(path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ocr = {
        "source": {"path": str(path), "w": 300, "h": 140},
        "lines": [_line("L0", "SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {"run_dir": str(run_dir)})
    line = result["lines"][0]

    assert line["meta"]["low_fidelity"] is True
    assert "fidelity_reason" in line["meta"]
    assert line["meta"]["substitution"]["from"] == "text"
    fallback_src = line["meta"].get("fallback_src")
    assert fallback_src
    assert (run_dir / fallback_src).exists()

    # Regression: _make_blocks used to drop the fidelity signal entirely — blocks had
    # no "meta" key at all, so merge_layers (which prefers ocr["blocks"] over
    # ocr["lines"] whenever blocks is non-empty, i.e. on every real run) would never
    # see low_fidelity/fallback_src and would silently emit guessed text instead of
    # the masked-pixel fallback. The block must carry the same fidelity signal.
    assert result["blocks"], "expected at least one block"
    block = result["blocks"][0]
    assert block["line_ids"] == [line["id"]]
    assert block["meta"]["low_fidelity"] is True
    assert block["meta"]["fallback_src"] == fallback_src
    assert block["meta"]["fidelity_reason"] == line["meta"]["fidelity_reason"]


def test_confident_text_is_not_flagged_low_fidelity(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "ad.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 640, "h": 260},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    line = result["lines"][0]

    assert line["meta"]["low_fidelity"] is False
    assert "fallback_src" not in line["meta"]
    assert "substitution" not in line["meta"]


# ---------------------------------------------------------------------------
# Bug 2: gradient-stop and stroke-colour extraction


def _gradient_text_image(text, font, top_rgb, bottom_rgb, size=(360, 160), pos=(40, 40)):
    probe = Image.new("L", size, 0)
    ImageDraw.Draw(probe).text(pos, text, font=font, fill=255)
    mask = np.asarray(probe) > 32
    h, w = size[1], size[0]
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    grad = (1 - ramp) * np.array(top_rgb, dtype=np.float32) + ramp * np.array(bottom_rgb, dtype=np.float32)
    base = np.full((h, w, 3), 255.0, dtype=np.float32)
    base[mask] = grad.repeat(w, axis=1)[mask]
    image = Image.fromarray(base.astype(np.uint8), "RGB")
    bbox = ImageDraw.Draw(Image.new("L", size, 0)).textbbox(pos, text, font=font)
    box = (max(0, bbox[0] - 7), max(0, bbox[1] - 6), bbox[2] + 7, bbox[3] + 6)
    return image, box


def test_gradient_fill_extracted_as_linear_stops(tmp_path):
    font = _font(72)
    top_rgb, bottom_rgb = (235, 60, 20), (20, 70, 235)
    image, ocr_box = _gradient_text_image("SALE", font, top_rgb, bottom_rgb)
    path = tmp_path / "gradient.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    fill = result["lines"][0]["style"]["fill"]

    assert fill["kind"] == "linear"
    assert len(fill["stops"]) == 2
    start = np.array(text_analysis._hex_rgb(fill["stops"][0]["color"]))
    end = np.array(text_analysis._hex_rgb(fill["stops"][-1]["color"]))
    assert np.linalg.norm(start - end) > 60


def test_stroked_text_extracts_distinct_stroke_and_fill_colour(tmp_path):
    font = _font(80)
    fill_rgb, stroke_rgb = (250, 250, 250), (15, 15, 15)
    image = Image.new("RGB", (360, 180), (120, 170, 230))
    draw = ImageDraw.Draw(image)
    pos = (40, 40)
    draw.text(pos, "OFF", font=font, fill=fill_rgb, stroke_width=6, stroke_fill=stroke_rgb)
    bbox = draw.textbbox(pos, "OFF", font=font, stroke_width=6)
    ocr_box = (max(0, bbox[0] - 8), max(0, bbox[1] - 8), bbox[2] + 8, bbox[3] + 8)
    path = tmp_path / "stroke.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "OFF", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    style = result["lines"][0]["style"]

    assert style["stroke"] is not None
    stroke_hex = style["stroke"]["color"]
    fill_hex = style["fill"]["color"]
    assert text_analysis._colour_distance(stroke_hex, fill_hex) > 60
    # stroke sample should land closer to the outline colour than to the fill colour
    stroke_rgb_hex = text_analysis._rgb_hex(stroke_rgb)
    fill_rgb_hex = text_analysis._rgb_hex(fill_rgb)
    assert text_analysis._colour_distance(stroke_hex, stroke_rgb_hex) < text_analysis._colour_distance(
        stroke_hex, fill_rgb_hex
    )


# ---------------------------------------------------------------------------
# Bug 3: glyph-shear (italic) measurement independent of font matching


def test_measures_shear_angle_on_italic_glyph_mask_without_font_matching(tmp_path):
    italic_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "C:/Windows/Fonts/ariali.ttf",
    ]
    italic_path = next((p for p in italic_candidates if os.path.isfile(p)), None)
    if not italic_path:
        pytest.skip("no system italic font available")
    font = ImageFont.truetype(italic_path, 64)

    image = Image.new("RGB", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (40, 40), "Slanted", font, (10, 10, 10))
    path = tmp_path / "italic.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "Slanted", ocr_box)],
    }

    # font_matching stays disabled: italic must be detected from the ink mask alone.
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]

    assert style["italicShearDeg"] is not None
    assert abs(style["italicShearDeg"]) >= 6.0
    assert "italic" in style["fontStyle"].lower()


def test_upright_glyph_mask_measures_near_zero_shear(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "upright.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    style = result["lines"][0]["style"]

    assert style["italicShearDeg"] is None or abs(style["italicShearDeg"]) < 6.0
    assert "italic" not in style["fontStyle"].lower()


# ---------------------------------------------------------------------------
# Bug 5: fontStyleCandidates must preserve weight, not hardcode "Italic"/"Regular"


def test_font_style_candidates_preserve_weight_when_alternating_italic(tmp_path):
    image = Image.new("RGB", (420, 160), "white")
    draw = ImageDraw.Draw(image)
    font = _font(60)
    # A heavy/dense stroke pushes the density-based weight estimate to Bold.
    ocr_box = _draw_text(draw, (30, 30), "BOLD", font, (0, 0, 0))
    path = tmp_path / "bold.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "BOLD", ocr_box)],
    }

    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]
    candidates = {c["value"] for c in style["fontStyleCandidates"]}

    assert "Italic" not in candidates  # bare "Italic" would drop the weight
    assert any(value.endswith("Italic") and value != "Italic" for value in candidates) or \
        style["fontWeight"] < 700  # only assert alternation when weight actually landed Bold


# ---------------------------------------------------------------------------
# Bug 4: style-cluster representative matching propagates beyond max_lines


def test_style_cluster_propagates_font_match_beyond_max_lines_budget(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    font = _font(30)

    n_lines = 14
    line_h = 40
    image = Image.new("RGB", (500, line_h * n_lines + 20), "white")
    draw = ImageDraw.Draw(image)
    lines = []
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
             "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November"]
    for i, word in enumerate(words[:n_lines]):
        y = 10 + i * line_h
        box = _draw_text(draw, (20, y), word, font, (20, 20, 20))
        lines.append(_line(f"L{i}", word, box))
    path = tmp_path / "many_lines.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": image.width, "h": image.height}, "lines": lines}

    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": [],
                "max_fonts": 1,
                "max_lines": 3,   # far fewer than n_lines — the old per-line budget
                "top_k": 1,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    sources = [line["style"]["fontCandidates"][0]["source"] for line in result["lines"]]

    # All lines share one style cluster, so the 3-slot budget should still cover
    # every line via propagation from a single representative match.
    assert sources.count("local-render") == n_lines
    assert result["text_analysis"]["font_matches_attempted"] <= 3


def test_meta_alignment_prefers_matching_weight():
    profile = text_analysis._typography_profile({"weight": 700, "shear_angle": None, "font_size": 24})
    bold_meta = {"family": "Inter", "style": "Bold", "weight": 700}
    light_meta = {"family": "Inter", "style": "Light", "weight": 300}
    assert text_analysis._meta_alignment_adjustment(bold_meta, profile) > \
        text_analysis._meta_alignment_adjustment(light_meta, profile)


def test_fallback_chain_uses_weight_and_italic(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (200, 80), "white").save(path)
    ocr = {
        "source": {"w": 200, "h": 80},
        "lines": [_line("L0", "SALE", (20, 20, 120, 50))],
    }
    result = text_analysis.analyze_text(
        str(path), ocr,
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    style = result["lines"][0]["style"]
    assert style["fontWeight"] in {300, 400, 500, 600, 700}
    assert style["fontCandidates"][0]["weight"] == style["fontWeight"]
    assert "Italic" not in style["fontCandidates"][0]["style"]


def test_google_fonts_cache_candidates_merge_into_chain(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    cache_dir = tmp_path / "google-fonts"
    cache_dir.mkdir()
    cached_font = cache_dir / "Inter-Regular.ttf"
    cached_font.write_bytes(open(font_path, "rb").read())

    image = Image.new("RGB", (520, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 38)
    ocr_box = _draw_text(draw, (35, 45), "Cache", font, (0, 0, 0))
    path = tmp_path / "font.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 520, "h": 180},
        "lines": [_line("L0", "Cache", ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [],
                "font_dirs": [],
                "google_fonts_cache": str(cache_dir),
                "max_fonts": 1,
                "max_lines": 1,
                "top_k": 3,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    sources = {item.get("source") for item in result["lines"][0]["style"]["fontCandidates"]}
    assert "google-cache" in sources
    assert "fallback" in sources


def test_needs_vlm_font_judge_when_local_score_is_weak():
    ocr = {
        "lines": [{
            "style": {
                "fontCandidates": [
                    {"family": "Inter", "source": "local-render", "score": 0.31, "path": "/tmp/a.ttf"},
                    {"family": "Arial", "source": "fallback", "score": 0.55},
                ]
            }
        }]
    }
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "local_score_threshold": 0.55}}}
    assert text_analysis.needs_vlm_font_judge(ocr, cfg) is True


def test_design_json_preserves_font_candidates(tmp_path):
    from src.build_design_json import build

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidates = [{
        "id": "T0",
        "target": "text",
        "text": "SALE",
        "box": {"x": 10, "y": 10, "w": 120, "h": 40},
        "visible_box": {"x": 10, "y": 10, "w": 120, "h": 40},
        "style": {
            "fontFamily": "Inter",
            "fontSize": 28,
            "fontWeight": 700,
            "fontStyle": "Bold",
            "color": "#111111",
            "fontCandidates": [
                {"family": "Inter", "style": "Bold", "weight": 700, "score": 0.82, "source": "local-render"},
                {"family": "Arial", "style": "Bold", "weight": 700, "score": 0.71, "source": "fallback"},
            ],
            "fontSizeCandidates": [{"value": 28, "score": 0.75}],
        },
    }]
    build(candidates, {"w": 200, "h": 120}, str(run_dir))
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    style = design["layers"][0]["style"]
    assert style["fontCandidates"][0]["family"] == "Inter"
    assert style["fontSizeCandidates"][0]["value"] == 28
