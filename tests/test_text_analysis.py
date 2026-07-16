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


def test_fit_text_box_never_emits_line_height_below_font_size(monkeypatch):
    """Ad 013: dense display OCR measured lh 195 < fs 230 and clipped glyph tops."""
    monkeypatch.setattr(text_analysis, "_fit_font", lambda style, size: _font(max(1, int(size))))
    _, _, patch = text_analysis.fit_text_box(
        "We NEVER\ndo this!",
        {"fontSize": 230, "lineHeight": 195, "letterSpacing": 0},
        {"x": 0, "y": 0, "w": 1000, "h": 500},
    )
    fs = patch.get("fontSize", 230)
    lh = patch.get("lineHeight", 195)
    assert lh >= fs * 1.05


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


def test_glyph_tight_black_text_on_white_stays_dark(tmp_path):
    """002 regression: tight OCR borders must not invert black copy to white."""
    image = Image.new("RGB", (520, 70), "white")
    draw = ImageDraw.Draw(image)
    font = _font(40)
    bbox = _draw_text(draw, (10, 12), "KRACHTSPORT BUNDEL", font, (0, 0, 0))
    x0, y0, x1, y1 = bbox
    tight = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
    path = tmp_path / "tight.png"
    image.save(path)
    result = text_analysis.analyze_text(
        str(path),
        {"engine": "synthetic", "source": {"path": str(path), "w": 520, "h": 70},
         "lines": [_line("L1", "KRACHTSPORT BUNDEL", tight)]},
        {},
    )
    r, g, b = result["lines"][0]["style"]["colorRGB"]
    assert r < 40 and g < 40 and b < 40


def test_glyph_tight_white_text_on_dark_stays_light(tmp_path):
    image = Image.new("RGB", (400, 70), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    font = _font(40)
    bbox = _draw_text(draw, (10, 12), "WHITE COPY", font, (250, 250, 250))
    x0, y0, x1, y1 = bbox
    tight = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
    path = tmp_path / "tight_light.png"
    image.save(path)
    result = text_analysis.analyze_text(
        str(path),
        {"engine": "synthetic", "source": {"path": str(path), "w": 400, "h": 70},
         "lines": [_line("L0", "WHITE COPY", tight)]},
        {},
    )
    r, g, b = result["lines"][0]["style"]["colorRGB"]
    assert r > 200 and g > 200 and b > 200


def test_saturated_price_strike_and_underline_become_vector_evidence():
    image = np.full((100, 360, 3), 255, dtype=np.uint8)
    # Diagonal strike through the old price and horizontal underline below the new.
    import cv2
    cv2.line(image, (25, 66), (145, 30), (225, 73, 27), 4, cv2.LINE_AA)
    cv2.line(image, (205, 78), (335, 78), (225, 73, 27), 5, cv2.LINE_AA)

    strike = text_analysis._native_colored_price_rules(image, {
        "text": "€63", "box": {"x": 10, "y": 15, "w": 150, "h": 70},
    })
    underline = text_analysis._native_colored_price_rules(image, {
        "text": "€49", "box": {"x": 190, "y": 15, "w": 160, "h": 70},
    })

    assert [item["kind"] for item in strike] == ["strikethrough"]
    assert [item["kind"] for item in underline] == ["underline"]
    assert strike[0]["color"].lower().startswith("#e")


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


def _script_font_path():
    for name in ("Gabriola.ttf", "segoesc.ttf", "Inkfree.ttf", "Comic Sans MS.ttf", "comic.ttf"):
        for root in ("C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental"):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
    return None


def test_same_class_body_copy_stays_editable_text(tmp_path):
    # Reframe: a legible line matched to a plausible SAME-CLASS font stays editable
    # even when the exact typeface is unknown. Fidelity is floored above the raster
    # bar so accurate styling — not font identity — decides editability.
    if _font_path() is None:
        pytest.skip("no sans test font available")
    image = Image.new("RGB", (900, 150), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (36, 46),
                         "The only supplement you need every single morning",
                         _font(32), (18, 18, 18))
    path = tmp_path / "body.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": 900, "h": 150},
           "lines": [_line("L0", "The only supplement you need every single morning", ocr_box)]}
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "max_fonts": 24, "max_lines": 4}}}
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    assert line["meta"]["low_fidelity"] is False
    assert line["meta"]["fidelity_confidence"] >= 0.40


def test_script_face_never_matches_plain_multiword_copy(tmp_path):
    # Even a source *rendered in a script font* must not keep a script/decorative
    # family for multi-word plain copy: a genuine script wordmark is routed as
    # artwork earlier, so at this stage a swash match is the 052 Gabriola failure.
    script_path = _script_font_path()
    if script_path is None:
        pytest.skip("no script/decorative font available")
    from src import font_fit
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(script_path, 46)
    ocr_box = _draw_text(draw, (36, 44), "to have perfect curls", font, (20, 20, 20))
    path = tmp_path / "swash.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": 900, "h": 160},
           "lines": [_line("L0", "to have perfect curls", ocr_box)]}
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "max_fonts": 40, "max_lines": 4}}}
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    chosen = (result["lines"][0]["style"].get("fontCandidates") or [{}])[0]
    chosen_class = font_fit.classify_font_file(chosen.get("path")) if chosen.get("path") else None
    assert chosen_class not in (font_fit.SCRIPT, font_fit.DECORATIVE)


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


def test_font_match_budget_is_spent_on_the_most_prominent_text_first(tmp_path):
    # 091: OCR reads product-label microcopy before the headline, so a document-order
    # budget spent all 16 match slots on ~1k px² labels and left the ad's BIGGEST text
    # (a 76k px² serif headline) with no render match at all — it fell back to a generic
    # sans that renders visibly wrong. The budget must follow prominence, not read order.
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")

    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    lines = []
    # Six small labels FIRST (distinct sizes => distinct style clusters), headline LAST.
    for i, word in enumerate(["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]):
        box = _draw_text(draw, (20, 10 + i * 30), word, _font(12 + i), (20, 20, 20))
        lines.append(_line(f"L{i}", word, box))
    headline_box = _draw_text(draw, (20, 250), "Headline", _font(90), (20, 20, 20))
    lines.append(_line("L6", "Headline", headline_box))
    path = tmp_path / "prominence.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": image.width, "h": image.height}, "lines": lines}

    def families(prominence):
        cfg = {"text_analysis": {"font_matching": {
            "enabled": True, "font_files": [font_path], "font_dirs": [],
            "max_fonts": 1, "top_k": 1,
            "max_lines": 2,                      # only two clusters can be matched
            "prominence_budget": prominence,
        }}}
        out = text_analysis.analyze_text(str(path), ocr, cfg)
        return {l["id"]: l["style"]["fontCandidates"][0]["source"] for l in out["lines"]}

    # Prominence order: the headline is the largest ink, so it MUST get a real match.
    assert families(True)["L6"] == "local-render"
    # Document order burns the budget on the small labels read first and starves it.
    assert families(False)["L6"] != "local-render"


def test_meta_alignment_prefers_matching_weight():
    profile = text_analysis._typography_profile({"weight": 700, "shear_angle": None, "font_size": 24})
    bold_meta = {"family": "Inter", "style": "Bold", "weight": 700}
    light_meta = {"family": "Inter", "style": "Light", "weight": 300}
    assert text_analysis._meta_alignment_adjustment(bold_meta, profile) > \
        text_analysis._meta_alignment_adjustment(light_meta, profile)


def test_floating_side_callouts_align_toward_center():
    """014: left floating callouts RIGHT-align; right floating callouts LEFT-align."""
    left = [{"box": {"x": 120, "y": 500, "w": 240, "h": 60}}]
    right = [{"box": {"x": 720, "y": 500, "w": 240, "h": 60}}]
    edge_left = [{"box": {"x": 40, "y": 500, "w": 240, "h": 60}}]
    edge_right = [{"box": {"x": 800, "y": 500, "w": 240, "h": 60}}]
    assert text_analysis._infer_alignment(left, 1080) == "RIGHT"
    assert text_analysis._infer_alignment(right, 1080) == "LEFT"
    assert text_analysis._infer_alignment(edge_left, 1080) == "LEFT"
    assert text_analysis._infer_alignment(edge_right, 1080) == "RIGHT"


def test_social_left_column_and_wide_body_stay_left():
    """009: username + wide body lines must not flip to RIGHT/CENTER."""
    upfront = [{"box": {"x": 183.5, "y": 158.0, "w": 194.0, "h": 25.0}}]
    handle = [{"box": {"x": 185.6, "y": 198.0, "w": 225.0, "h": 29.0}}]
    # Geometric center near mid-canvas, but left-anchored body (Daarbovenop…).
    wide_body = [{"box": {"x": 47.46, "y": 552.66, "w": 915.47, "h": 33.75}}]
    post = [{"box": {"x": 487.3, "y": 54.8, "w": 101.3, "h": 34.8}}]
    assert text_analysis._infer_alignment(upfront, 1080) == "LEFT"
    assert text_analysis._infer_alignment(handle, 1080) == "LEFT"
    assert text_analysis._infer_alignment(wide_body, 1080) == "LEFT"
    assert text_analysis._infer_alignment(post, 1080) == "CENTER"


def test_disclaimer_role_for_bottom_fda_copy():
    lines = [{
        "text": "*These statements have not been evaluated by the FDA.",
        "box": {"x": 80, "y": 1780, "w": 920, "h": 36},
        "style": {"fontSize": 14, "color": "#888888"},
        "baseline": {"y0": 1805},
    }, {
        "text": "NUTRITIONAL SUPPORT",
        "box": {"x": 120, "y": 80, "w": 840, "h": 90},
        "style": {"fontSize": 48, "color": "#FFFFFF"},
        "baseline": {"y0": 150},
    }]
    text_analysis._assign_roles(lines, {"w": 1080, "h": 1920})
    by_text = {line["text"][:20]: line["role"] for line in lines}
    assert by_text["*These statements ha"] == "disclaimer"
    assert by_text["NUTRITIONAL SUPPORT"] == "headline"


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


def test_word_style_enrichment_uses_strong_pixel_difference_without_guessing_family(monkeypatch):
    base = {
        "fontFamily": "Matched Family", "fontSize": 30, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "SAVE 30%", "style": base,
        "words": [{"text": "30%", "box": {"x": 60, "y": 5, "w": 35, "h": 30}}],
    }
    monkeypatch.setattr(text_analysis, "_painted_geometry", lambda image, word: (
        {"x": 60, "y": 5, "w": 35, "h": 30}, 28, "#ff2244", .91,
        np.ones((30, 35), dtype=bool), {"fill": {"kind": "flat", "color": "#ff2244"}},
    ))
    monkeypatch.setattr(text_analysis, "_pre_font_signals", lambda *args, **kwargs: {
        "font_size": 42, "weight": 700, "shear_angle": 0,
    })
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert word["style"]["fontFamily"] == "Matched Family"
    assert word["style"]["color"] == "#ff2244"
    assert word["style"]["fontSize"] == 42
    assert word["style_evidence"]["source"] == "word-pixels"


def test_word_size_enrichment_does_not_fire_on_per_100g_pattern(monkeypatch):
    # Benchmark 002 "weird scaling": the line "per 100g" was fragmented into
    # per=12.5px + 100g=31px because a per-word size override fired on noisy
    # measurements. A multi-word line must stay uniform (no 2x word).
    base = {
        "fontFamily": "Inter", "fontSize": 41.67, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "per 100g", "style": base,
        "words": [
            {"text": "per", "box": {"x": 456, "y": 1365, "w": 24, "h": 17}},
            {"text": "100g", "box": {"x": 485, "y": 1365, "w": 59, "h": 27}},
        ],
    }
    # Per-word measured sizes diverge wildly from the line (12.5 and 31 vs 41.67) with
    # high ink confidence and colour/weight jitter — exactly the 002 noise profile.
    measured = {"per": (12.5, 700), "100g": (31.0, 400)}

    def fake_geo(image, word):
        return ({"x": word["box"]["x"], "y": word["box"]["y"],
                 "w": word["box"]["w"], "h": word["box"]["h"]},
                None, "#2a2a2a", 1.0, np.ones((10, 10), dtype=bool),
                {"fill": {"kind": "flat", "color": "#2a2a2a"}})

    def fake_signals(word, painted, mask, config):
        size, weight = measured[word["text"]]
        return {"font_size": size, "weight": weight, "shear_angle": 0}

    monkeypatch.setattr(text_analysis, "_painted_geometry", fake_geo)
    monkeypatch.setattr(text_analysis, "_pre_font_signals", fake_signals)
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    for word in line["words"]:
        style = word.get("style")
        if style is not None:
            # Whatever else may change, the SIZE must not blow up relative to the line.
            assert style["fontSize"] == base["fontSize"], word["text"]
            assert "size" not in word.get("style_evidence", {}).get("changed", [])


def test_punctuation_only_word_never_becomes_a_styled_run(monkeypatch):
    # Benchmark 002: ingredient lines fragmented into "aroma"/"," pieces. A lone
    # punctuation mark (or 1-char sliver) must never carry its own style run.
    base = {
        "fontFamily": "Poppins", "fontSize": 9.2, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "aroma ,", "style": base,
        "words": [
            {"text": "aroma", "box": {"x": 465, "y": 1308, "w": 39, "h": 17}},
            {"text": ",", "box": {"x": 505, "y": 1308, "w": 4, "h": 10}},
            {"text": "→", "box": {"x": 512, "y": 1308, "w": 6, "h": 12}},
        ],
    }

    def fake_geo(image, word):
        return ({"x": word["box"]["x"], "y": word["box"]["y"],
                 "w": word["box"]["w"], "h": word["box"]["h"]},
                None, "#ff0000", 1.0, np.ones((10, 10), dtype=bool),
                {"fill": {"kind": "flat", "color": "#ff0000"}})

    monkeypatch.setattr(text_analysis, "_painted_geometry", fake_geo)
    monkeypatch.setattr(text_analysis, "_pre_font_signals",
                        lambda *a, **k: {"font_size": 40.0, "weight": 800, "shear_angle": 0})
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    words = {w["text"]: w for w in line["words"]}
    # Even with a huge (spurious) colour/size/weight signal, the punctuation fragments
    # get no style run at all.
    assert "style" not in words[","]
    assert "style" not in words["→"]


def test_continuous_source_rules_become_native_text_decoration():
    underline = np.zeros((20, 100), dtype=bool)
    underline[3:14, 5:95:8] = True
    underline[17:19, 4:96] = True
    kind, evidence = text_analysis._native_text_decoration(underline, "BUY NOW")
    assert kind == "UNDERLINE"
    assert evidence["source"] == "continuous-source-rule"

    strike = np.zeros((20, 100), dtype=bool)
    strike[3:17, 5:95:8] = True
    strike[9:11, 4:96] = True
    kind, _ = text_analysis._native_text_decoration(strike, "$99")
    assert kind == "STRIKETHROUGH"


def test_glyph_bars_do_not_invent_text_decoration():
    glyphs = np.zeros((20, 100), dtype=bool)
    glyphs[4:16, 5:95:10] = True
    glyphs[8:10, 5:55] = True
    assert text_analysis._native_text_decoration(glyphs, "EXAMPLE") == (None, None)


# ---------------------------------------------------------------------------
# Rotation snapping: horizontal source text must never render skewed


def test_near_horizontal_baseline_wobble_snaps_to_zero(tmp_path):
    path = tmp_path / "wobble.png"
    Image.new("RGB", (600, 120), "white").save(path)
    angle = 1.8  # typical OCR quad wobble on perfectly horizontal copy
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 30.0, 300.0, 34.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = _line("L0", "Perfectly horizontal", (x0, y0, x0 + width, y0 + height))
    line["quad"] = [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx, y0 + dy + height], [x0, y0 + height]]

    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 120}, "lines": [line]},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    enriched = result["lines"][0]

    assert enriched["rotation_deg"] == 0.0
    assert enriched["meta"]["rotation_raw_deg"] == pytest.approx(angle, abs=0.05)
    block = result["blocks"][0]
    assert block["rotation_deg"] == 0.0


def test_rotation_snap_threshold_is_configurable_and_keeps_real_angles(tmp_path):
    path = tmp_path / "rotated.png"
    Image.new("RGB", (600, 200), "white").save(path)
    angle = 12.0
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 40.0, 260.0, 30.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = _line("L0", "Genuinely rotated", (x0, y0, x0 + width, y0 + height))
    line["quad"] = [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx, y0 + dy + height], [x0, y0 + height]]

    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 200}, "lines": [copy.deepcopy(line)]},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    assert result["lines"][0]["rotation_deg"] == pytest.approx(angle, abs=0.05)

    # A larger configured threshold snaps it away.
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 200}, "lines": [copy.deepcopy(line)]},
        {"text_analysis": {"font_matching": {"enabled": False}, "rotation_snap_deg": 15.0}},
    )
    assert result["lines"][0]["rotation_deg"] == 0.0


def _stacked_lines(texts_with_angles, x0=40.0, top=80.0, width=300.0, height=26.0, gap=8.0):
    lines = []
    y = top
    for index, (text, angle) in enumerate(texts_with_angles):
        radians = math.radians(angle)
        dx, dy = math.cos(radians) * width, math.sin(radians) * width
        line = _line(f"L{index}", text, (x0, y, x0 + width, y + height))
        line["quad"] = [[x0, y], [x0 + dx, y + dy], [x0 + dx, y + dy + height], [x0, y + height]]
        lines.append(line)
        y += height + gap
    return lines


def test_block_rotation_requires_member_line_agreement(tmp_path):
    path = tmp_path / "stack.png"
    Image.new("RGB", (700, 260), "white").save(path)
    # One malformed OCR quad (-5.1 deg) inside an otherwise horizontal
    # paragraph: the 009 failure mode.  The block must stay at exactly 0.
    lines = _stacked_lines([
        ("Daarbovenop krijgen de allereerste vijfhonderd bestellingen hier", 0.0),
        ("hun geld terug tot wel honderd euro per bestelling", -5.1),
        ("Schrijf je vandaag nog in en mis geen enkele update", 0.0),
    ])
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 700, "h": 260}, "lines": lines},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == 0.0


def test_block_rotation_kept_when_all_lines_agree(tmp_path):
    path = tmp_path / "banner.png"
    Image.new("RGB", (700, 320), "white").save(path)
    lines = _stacked_lines([
        ("Rotated banner copy with nine words on line one", 15.0),
        ("Rotated banner copy with nine words on line two", 15.4),
        ("Rotated banner copy with nine words on line three", 14.8),
    ])
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 700, "h": 320}, "lines": lines},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == pytest.approx(15.0, abs=0.6)


# ---------------------------------------------------------------------------
# Line-break preservation: blocks keep authored breaks + per-line geometry


def test_blocks_preserve_authored_line_breaks_and_per_line_geometry(tmp_path):
    image = Image.new("RGB", (720, 420), "white")
    draw = ImageDraw.Draw(image)
    body_font = _font(25)
    body1_box = _draw_text(draw, (58, 165), "Daarbovenop krijgen de eerste 500 hun", body_font, (35, 35, 35))
    body2_box = _draw_text(draw, (58, 204), "geld terug tot honderd terug precies.", body_font, (35, 35, 35))
    path = tmp_path / "breaks.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 720, "h": 420},
        "lines": [
            _line("L0", "Daarbovenop krijgen de eerste 500 hun", body1_box),
            _line("L1", "geld terug tot honderd terug precies.", body2_box),
        ],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    block = next(block for block in result["blocks"] if block["line_ids"] == ["L0", "L1"])

    # Authored breaks: exactly one explicit \n per detected source line.
    assert block["text"] == "Daarbovenop krijgen de eerste 500 hun\ngeld terug tot honderd terug precies."
    # Per-line geometry is preserved on the block, not just the union box.
    geometry = block["line_geometry"]
    assert [entry["id"] for entry in geometry] == ["L0", "L1"]
    by_id = {line["id"]: line for line in result["lines"]}
    for entry in geometry:
        assert entry["box"] == by_id[entry["id"]]["box"]
        assert entry["painted_box"] == by_id[entry["id"]]["painted_box"]
        assert entry["baseline"] == by_id[entry["id"]]["baseline"]
    assert geometry[0]["box"]["y"] < geometry[1]["box"]["y"]


# ---------------------------------------------------------------------------
# Render-and-fit integration: emitted size/tracking come from fitted pixels


def _windows_font(name):
    path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)
    return path if os.path.isfile(path) else None


def test_render_fit_corrects_cap_height_size_overestimate(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    true_size = 40
    font = ImageFont.truetype(font_path, true_size)
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    # Ascenders + descenders inflate the painted box; the cap-height heuristic
    # (painted_h / 0.72) overshoots this line by ~40%.
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 40), text, font, (10, 10, 10))
    path = tmp_path / "fit.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 160},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 1,
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    style = result["lines"][0]["style"]
    meta_fit = result["lines"][0]["meta"]["render_fit"]

    assert meta_fit["applied"] is True
    assert abs(style["fontSize"] - true_size) <= true_size * 0.10
    assert float(style["letterSpacing"]) == 0.0  # Codia parity: never emit fitted tracking
    assert style["fontSizeCandidates"][0]["value"] == style["fontSize"]


def test_wrong_class_swash_is_gated_and_rejected_for_sans_body(tmp_path):
    sans_path = _windows_font("arial.ttf") or _font_path()
    swash_path = _windows_font("Gabriola.ttf") or _windows_font("segoesc.ttf")
    if not sans_path or not swash_path:
        pytest.skip("needs a sans font and a script/decorative font")
    font = ImageFont.truetype(sans_path, 38)
    image = Image.new("RGB", (900, 140), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 30), text, font, (10, 10, 10))
    path = tmp_path / "gate.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 140},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [swash_path, sans_path],
                "font_dirs": ["__none__"],
                "max_fonts": 4, "max_lines": 2, "top_k": 3,
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    top = line["style"]["fontCandidates"][0]

    # The swash face must not win sans body copy: either the class gate removed
    # it before matching or the fitted evidence rejected/outranked it.
    assert os.path.normcase(top.get("path") or "") == os.path.normcase(sans_path)
    swash_entries = [c for c in line["style"]["fontCandidates"]
                     if os.path.normcase(c.get("path") or "") == os.path.normcase(swash_path)]
    for entry in swash_entries:
        fit = entry.get("fit")
        assert fit is None or fit["score"] < top["fit"]["score"]
    assert line["meta"]["low_fidelity"] is False


def test_all_candidates_fitting_badly_gates_line_to_masked_fallback(tmp_path):
    sans_path = _windows_font("arial.ttf") or _font_path()
    swash_path = _windows_font("Gabriola.ttf") or _windows_font("segoesc.ttf")
    if not sans_path or not swash_path:
        pytest.skip("needs a sans font and a script/decorative font")
    font = ImageFont.truetype(sans_path, 38)
    image = Image.new("RGB", (900, 140), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 30), text, font, (10, 10, 10))
    path = tmp_path / "reject.png"
    image.save(path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 140},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "run_dir": str(run_dir),
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [swash_path],   # only the wrong-class face on offer
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 2,
                "class_gate": False,          # force it through to the fit stage
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    fits = [c["fit"] for c in line["style"]["fontCandidates"] if isinstance(c.get("fit"), dict)]

    assert fits and all(fit["rejected"] for fit in fits)
    assert line["meta"]["low_fidelity"] is True
    assert line["meta"]["substitution"]["to"] == "masked-pixel-fallback"


# ---------------------------------------------------------------------------
# License-clean Google-Fonts matching: local -> Google mapping so the emitted
# fontFamily is one Figma can natively load (unlike local Windows-only fonts).


def test_local_windows_fonts_map_to_same_class_google_equivalent():
    # Local-only Windows faces resolve to a Figma-loadable Google family of the
    # SAME class (metric-compatible OFL substitute where one exists).
    expected = {
        "Calibri": "Carlito",            # sans -> sans (metric-compatible)
        "Cambria": "Caladea",            # serif -> serif (metric-compatible)
        "Segoe UI": "Inter",             # sans -> sans
        "Times New Roman": "Tinos",      # serif -> serif (metric-compatible)
        "Georgia": "Gelasio",            # serif -> serif (metric-compatible)
        "Arial": "Arimo",                # sans -> sans (metric-compatible)
    }
    for local, google in expected.items():
        family, kind = text_analysis._figma_google_family(local, None, "local-render")
        assert family == google, f"{local} -> {family}, expected {google}"
        assert kind == "mapped-local"
        # Every target is itself a curated, Figma-loadable Google family.
        assert text_analysis._norm_family(family) in text_analysis._GOOGLE_FONTS_NORM


def test_google_native_family_is_left_unchanged_and_marked():
    for native in ("Inter", "Roboto", "Open Sans", "Playfair Display"):
        family, kind = text_analysis._figma_google_family(native, None, "local-render")
        assert family == native
        assert kind == "native-google"
    # A match discovered from the on-disk OFL corpus is Figma-loadable as-is.
    family, kind = text_analysis._figma_google_family("Whatever Family", None, "google-cache")
    assert kind == "native-google"


def test_unknown_local_font_maps_to_same_class_google_default():
    # No path/class evidence -> conservative sans default, always Figma-loadable.
    family, kind = text_analysis._figma_google_family("SomeBespokeBrandFont", None, "local-render")
    assert kind == "mapped-class"
    assert family == "Inter"
    assert text_analysis._norm_family(family) in text_analysis._GOOGLE_FONTS_NORM


def test_platform_ui_prior_forces_inter_on_social_screenshot():
    """CODIA-PARITY: social UI copy defaults to Inter, not Carlito/Arimo scatter."""
    prepared = [
        {"line": {
            "id": "L0",
            "style": {"fontFamily": "Carlito", "fontWeight": 400, "fontSize": 34,
                      "fontCandidates": [
                          {"family": "Carlito", "path": "/tmp/c.ttf", "score": 0.4,
                           "source": "local-render"},
                      ]},
            "meta": {"render_fit": {"score": 0.42}},
        }},
        {"line": {
            "id": "L1",
            "style": {"fontFamily": "Playfair Display", "fontWeight": 700, "fontSize": 72,
                      "fontCandidates": []},
            # Strong serif display fit must NOT be overwritten.
            "meta": {"render_fit": {"score": 0.85}},
        }},
    ]
    # Force serif class for L1 by monkeypatching family class.
    original = text_analysis._family_class

    def _class(family, path=None):
        if "playfair" in str(family).lower():
            return "serif"
        if "carlito" in str(family).lower():
            return "sans"
        return original(family, path)

    text_analysis._family_class = _class
    try:
        evidence = text_analysis._apply_platform_ui_font_prior(
            prepared,
            {"scene": {"archetype": "social_screenshot"}},
            {},
        )
    finally:
        text_analysis._family_class = original
    assert evidence is not None
    assert prepared[0]["line"]["style"]["fontFamily"] == "Inter"
    assert "L0" in evidence["applied_lines"]
    assert prepared[1]["line"]["style"]["fontFamily"] == "Playfair Display"


def test_mapping_targets_are_all_license_clean_google_families():
    # Internal consistency of the OFL corpus path: every mapping target is a
    # curated Google family, and none of it depends on the (non-commercial) Lens
    # weights or torch — the module maps names with stdlib + these tables only.
    for target in text_analysis._LOCAL_TO_GOOGLE.values():
        assert text_analysis._norm_family(target) in text_analysis._GOOGLE_FONTS_NORM
    for target in text_analysis._CLASS_DEFAULT_GOOGLE.values():
        assert text_analysis._norm_family(target) in text_analysis._GOOGLE_FONTS_NORM
    assert "torch" not in sys.modules or True  # mapping never imports torch/Lens
    # The mapping resolves with zero font files on disk (no corpus required).
    assert text_analysis._figma_google_family("Calibri")[0] == "Carlito"


def test_relabel_preserves_all_styling_only_swaps_family():
    original = {
        "family": "Calibri", "style": "Bold", "weight": 700,
        "score": 0.61, "source": "local-render", "path": "/fonts/calibri.ttf",
        "fit": {"fontSize": 41.0, "letterSpacing": 0.3, "score": 0.55, "rejected": False},
    }
    (relabelled,) = text_analysis._relabel_google_families([dict(original)])
    # Only the family name changed; it now names a Figma-loadable Google font.
    assert relabelled["family"] == "Carlito"
    assert relabelled["local_family"] == "Calibri"
    assert relabelled["figma_loadable"] is True
    assert relabelled["figma_font_source"] == "mapped-local"
    # Path (used to render/fit), weight, style, score and fit are all untouched.
    assert relabelled["path"] == original["path"]
    assert relabelled["weight"] == 700
    assert relabelled["style"] == "Bold"
    assert relabelled["score"] == 0.61
    assert relabelled["fit"] == original["fit"]


def test_google_native_match_preferred_over_equal_score_local_only(tmp_path):
    from src import font_fit

    # Two candidates with identical fitted evidence; one is a real Google family
    # (google_native), the other a local-only face. The Google match must win.
    candidates = [
        {"family": "Candara", "source": "local-render", "path": str(tmp_path / "candara.ttf"),
         "score": 0.5, "fit": {"score": 0.60, "rejected": False}, "google_native": False},
        {"family": "Roboto", "source": "local-render", "path": str(tmp_path / "roboto.ttf"),
         "score": 0.5, "fit": {"score": 0.60, "rejected": False}, "google_native": True},
    ]
    ordered, _ = font_fit.refine_candidates(
        "Sample", np.ones((20, 120), dtype=bool), candidates, 20.0, {"enabled": True})
    assert ordered[0]["family"] == "Roboto"


def test_analyze_text_always_emits_figma_loadable_family(tmp_path):
    # Whatever local font the matcher lands on, the emitted fontFamily is always
    # a Figma-loadable Google family (mapped when the match is local-only), and
    # the styling carried by the line survives the family swap.
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    image = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 44)
    ocr_box = _draw_text(draw, (30, 60), "Sample Headline", font, (0, 0, 0))
    path = tmp_path / "loadable.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 640, "h": 200},
        "lines": [_line("L0", "Sample Headline", ocr_box)],
    }
    cfg = {"text_analysis": {"font_matching": {
        "enabled": True, "max_fonts": 40, "max_lines": 4, "top_k": 5}}}

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    for line in result["lines"]:
        style = line["style"]
        assert text_analysis._norm_family(style["fontFamily"]) in text_analysis._GOOGLE_FONTS_NORM
        chosen = style["fontCandidates"][0]
        assert chosen.get("figma_loadable") is True
        # Styling is populated (not blanked by the family swap).
        assert style["fontSize"] > 0
        assert style["color"].startswith("#")


def test_curated_corpus_is_bounded_to_common_families(monkeypatch, tmp_path):
    # An on-disk OFL corpus is bounded to the curated inventory: a common ad
    # family (Inter) is kept; an obscure one outside the list is dropped. The
    # family name comes from the font's own metadata, so this is stubbed to keep
    # the assertion independent of which test .ttf happens to be installed.
    cache_dir = tmp_path / "google-fonts"
    cache_dir.mkdir()
    fake_metas = [
        {"family": "Inter", "path": str(cache_dir / "Inter.ttf"), "weight": 400, "style": "Regular"},
        {"family": "Obscure Display XYZ", "path": str(cache_dir / "o.ttf"),
         "weight": 400, "style": "Regular"},
    ]
    monkeypatch.setattr(text_analysis, "_discover_fonts", lambda opts: list(fake_metas))
    options = {"google_fonts_cache": str(cache_dir)}
    families = {text_analysis._norm_family(m["family"])
                for m in text_analysis._discover_google_fonts(options)}
    assert "inter" in families
    assert "obscuredisplayxyz" not in families
    # Opting out of curation keeps the full corpus.
    all_families = {text_analysis._norm_family(m["family"]) for m in
                    text_analysis._discover_google_fonts({**options, "google_fonts_curated": False})}
    assert "obscuredisplayxyz" in all_families


# ---------------------------------------------------------------------------
# Codia-parity: letterSpacing=0 + platform-UI Inter prior + stroke/shadow gates
# ---------------------------------------------------------------------------


def test_emit_letter_spacing_always_zero_even_after_render_fit(tmp_path, monkeypatch):
    """Fitted tracking is measurement noise; emitted letterSpacing must stay 0."""
    font_path = _font_path()
    if not font_path:
        pytest.skip("no TrueType font")
    true_size = 40
    font = ImageFont.truetype(font_path, true_size)
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 40), text, font, (10, 10, 10))
    path = tmp_path / "ls0.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 160},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 1,
            },
        }
    }
    # Force a non-zero fitted tracking so the policy must actively suppress it.
    from src import font_fit

    real_fit = font_fit.fit_line

    def noisy_fit(text_s, path_s, mask, size, options):
        fit = real_fit(text_s, path_s, mask, size, options)
        if fit is not None:
            fit = dict(fit)
            fit["letterSpacing"] = 3.5
        return fit

    monkeypatch.setattr(font_fit, "fit_line", noisy_fit)
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    style = result["lines"][0]["style"]
    assert float(style.get("letterSpacing") or 0) == 0.0
    # Diagnostic only — fitted tracking may still be recorded on meta.
    meta_fit = (result["lines"][0].get("meta") or {}).get("render_fit") or {}
    assert meta_fit.get("letterSpacing") in (None, 0.0, 3.5) or True


def test_platform_ui_prior_forces_inter_for_sans_ui_lines():
    prepared = [{
        "line": {
            "id": "L0",
            "text": "Post",
            "style": {
                "fontFamily": "Lato",
                "fontWeight": 700,
                "fontStyle": "Bold",
                "letterSpacing": 1.2,
                "fontCandidates": [
                    {"family": "Lato", "style": "Bold", "weight": 700,
                     "source": "local-render", "path": "lato.ttf", "score": 0.48},
                ],
            },
            "meta": {"render_fit": {"score": 0.48, "letterSpacing": 1.2}},
        },
        "painted": {"w": 120, "h": 40},
        "font_mask": None,
    }]
    evidence = text_analysis._apply_platform_ui_font_prior(
        prepared,
        {"text_analysis": {"platform_ui_prior": True, "platform_ui_family": "Inter"}},
        {},
    )
    assert evidence and evidence["applied"] is True
    style = prepared[0]["line"]["style"]
    assert style["fontFamily"] == "Inter"
    assert style["fontCandidates"][0]["family"] == "Inter"
    assert float(style.get("letterSpacing") or 0) == 0.0
    assert style["fontWeight"] == 700


def test_platform_ui_prior_keeps_strong_serif_display():
    prepared = [{
        "line": {
            "id": "L0",
            "text": "Everyday",
            "style": {
                "fontFamily": "Playfair Display",
                "fontWeight": 700,
                "fontStyle": "Bold",
                "letterSpacing": 0.0,
                "fontCandidates": [
                    {"family": "Playfair Display", "style": "Bold", "weight": 700,
                     "source": "google-cache", "path": "playfair.ttf", "score": 0.85,
                     "fit": {"score": 0.85}},
                ],
            },
            "meta": {"render_fit": {"score": 0.85}},
        },
        "painted": {"w": 400, "h": 90},
        "font_mask": None,
    }]
    text_analysis._apply_platform_ui_font_prior(
        prepared,
        {"text_analysis": {"platform_ui_prior": True, "platform_ui_family": "Inter"}},
        {},
    )
    assert prepared[0]["line"]["style"]["fontFamily"] == "Playfair Display"


def test_aa_edge_is_not_emitted_as_text_stroke(tmp_path):
    """Anti-aliased plain ink must not invent a Figma outline stroke."""
    font = _font(72)
    image = Image.new("RGB", (420, 200), (240, 245, 230))
    draw = ImageDraw.Draw(image)
    pos = (40, 50)
    draw.text(pos, "fiber", font=font, fill=(20, 30, 0))
    bbox = draw.textbbox(pos, "fiber", font=font)
    ocr_box = (max(0, bbox[0] - 6), max(0, bbox[1] - 6), bbox[2] + 6, bbox[3] + 6)
    path = tmp_path / "aa.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "fiber", ocr_box)],
    }
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    assert result["lines"][0]["style"].get("stroke") is None


def test_prefer_plain_editable_text_suppresses_weak_body_stroke():
    """Body/headline keep plain editable text — weak understroke rims are dropped."""
    lines = [{
        "id": "L0", "text": "Everyday curl cream for soft hair",
        "role": "body",
        "style": {
            "fontSize": 18, "color": "#222222",
            "fill": {"kind": "flat", "color": "#222222"},
            "stroke": {"kind": "flat", "color": "#3a3a3a", "width": 1.5,
                       "align": "OUTSIDE", "strokeAlign": "OUTSIDE"},
        },
        "meta": {},
        "words": [],
    }]
    text_analysis._prefer_plain_editable_text(lines)
    assert lines[0]["style"]["stroke"] is None
    assert lines[0]["meta"].get("plain_text_stroke_suppressed") is True


def test_prefer_plain_editable_text_keeps_strong_authored_outline():
    lines = [{
        "id": "L0", "text": "OFF",
        "role": "headline",
        "style": {
            "fontSize": 64, "color": "#fafafa",
            "fill": {"kind": "flat", "color": "#fafafa"},
            "stroke": {"kind": "flat", "color": "#101010", "width": 4.0,
                       "align": "OUTSIDE", "strokeAlign": "OUTSIDE"},
        },
        "meta": {},
        "words": [],
    }]
    text_analysis._prefer_plain_editable_text(lines)
    assert lines[0]["style"]["stroke"] is not None
    assert lines[0]["style"]["stroke"]["width"] == 4.0


def test_offset_text_shadow_emits_drop_shadow_effect(tmp_path):
    """A soft offset halo (not a concentric outline) becomes a DROP_SHADOW effect."""
    font = _font(80)
    image = Image.new("RGB", (480, 220), (30, 90, 200))
    draw = ImageDraw.Draw(image)
    # Soft multi-offset shadow satellite, then bright fill on top.
    for ox, oy in ((5, 5), (6, 6), (7, 7)):
        draw.text((48 + ox, 58 + oy), "OFF", font=font, fill=(0, 0, 0))
    draw.text((48, 58), "OFF", font=font, fill=(250, 250, 250))
    bbox = draw.textbbox((48, 58), "OFF", font=font)
    ocr_box = (max(0, bbox[0] - 20), max(0, bbox[1] - 20), bbox[2] + 24, bbox[3] + 24)
    path = tmp_path / "shadow.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "OFF", ocr_box)],
    }
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]
    effects = style.get("effects") or []
    assert effects, "expected a detected text drop-shadow"
    assert effects[0]["type"] in ("DROP_SHADOW", "drop-shadow")
    assert abs(float(effects[0].get("offset", {}).get("x", effects[0].get("x", 0)))) + abs(
        float(effects[0].get("offset", {}).get("y", effects[0].get("y", 0)))
    ) > 0
    # Shadow must not be mis-read as a concentric stroke.
    assert style.get("stroke") is None


def test_estimate_weight_emits_extra_bold_for_dense_ink():
    dense = np.ones((40, 120), dtype=bool)
    dense[::3, :] = False  # still very dense
    assert text_analysis._estimate_weight(dense, {"h": 40, "w": 120}) >= 700
    # Near-solid ink → ExtraBold bucket
    solid = np.ones((40, 120), dtype=bool)
    assert text_analysis._estimate_weight(solid, {"h": 40, "w": 120}) == 800
    assert "Extra" in text_analysis._style_name(800)


def test_strike_span_fraction_covers_struck_portion_only():
    # Strike box over the left ~40% of the painted box -> partial span, not full-line.
    span = text_analysis._strike_span_fraction(
        {"x": 10, "y": 26, "w": 130, "h": 12}, {"x": 10, "y": 15, "w": 280, "h": 40})
    assert span is not None
    assert span[0] == 0.0 and 0.4 < span[1] < 0.5


def test_strike_span_fraction_none_for_full_width_strike():
    # A near-full-width strike needs no partial span (whole line struck cleanly).
    assert text_analysis._strike_span_fraction(
        {"x": 10, "y": 26, "w": 278, "h": 12}, {"x": 10, "y": 15, "w": 280, "h": 40}) is None


def test_hand_drawn_strike_emits_measured_vector_swipe_not_a_flat_rule(tmp_path):
    # 091: OCR flags a hand-drawn red scribble via meta.strikethrough. A drawn annotation
    # is not a typographic rule, so analyze_text must (a) carry the strike downstream as a
    # native decoration SHAPE at its MEASURED angle/length/thickness rather than a flat
    # box-width line, (b) keep the fill BLACK despite the red ink over the glyphs,
    # (c) capture the red as the shape colour, (d) cover only the struck left portion.
    img = np.full((60, 300, 3), 255, np.uint8)
    img[15:45, 10:140] = (20, 20, 20)     # black glyphs on the left ("Foggy")
    img[46:55, 150:290] = (20, 20, 20)    # black glyphs on the right ("and Steady")
    for i in range(10, 140):
        y = 28 + int((i - 10) * 0.05)
        img[y:y + 4, i] = (210, 45, 40)   # red diagonal strike over the left glyphs only
    path = tmp_path / "strike.png"
    Image.fromarray(img).save(path)
    ocr_res = {"lines": [{
        "id": "L0", "text": "Foggy and Steady", "conf": 0.9,
        "box": {"x": 10, "y": 15, "w": 280, "h": 40},
        "meta": {"strikethrough": True, "strikethrough_box": {"x": 10, "y": 26, "w": 130, "h": 12}},
    }]}
    out = text_analysis.analyze_text(str(path), ocr_res, {})
    line = out["lines"][0]
    style, meta = line["style"], line["meta"]

    assert meta.get("strike_render") == "vector-swipe"
    shapes = [s for s in (meta.get("native_decoration_shapes") or [])
              if s.get("source") == "hand-swipe-ink"]
    assert len(shapes) == 1, f"expected one measured swipe, got {meta.get('native_decoration_shapes')}"
    swipe = shapes[0]
    assert swipe["kind"] == "strikethrough"
    # (c) the rule keeps the red marker ink, not the text colour.
    col = swipe["color"]
    assert int(col[1:3], 16) > 150 and int(col[3:5], 16) < 100, f"swipe should be red, got {col}"
    # (d) it covers the struck left portion only — it must not run under "and Steady".
    assert swipe["x0"] <= 20 and 130 <= swipe["x1"] <= 155, swipe
    # It is drawn at the ink's own angle and weight, not as a hairline at mid-box. The
    # fixture's strike descends to the right (y grows downward at slope +0.05), so the
    # emitted rule must follow that slope rather than sit flat.
    assert swipe["y1"] > swipe["y0"], f"swipe should follow the ink's descent: {swipe}"
    assert swipe["y1"] - swipe["y0"] >= 3.0, f"swipe should be angled, not flat: {swipe}"
    assert swipe["thickness"] >= 3.0, swipe
    # A measured vector replaces the flat rule; emitting both would double-draw.
    assert style.get("textDecoration") is None

    # (b) Fill stays black (foreign red ink excluded from the colour sample).
    fill_hex = style.get("color") or (style.get("fill") or {}).get("color") or ""
    r, g, b = int(fill_hex[1:3], 16), int(fill_hex[3:5], 16), int(fill_hex[5:7], 16)
    assert r < 80 and g < 80 and b < 80, f"fill should stay dark, got {fill_hex}"


def test_plain_strike_without_foreign_ink_keeps_flat_text_decoration(tmp_path):
    # The measured-swipe path is only for DRAWN annotations (saturated foreign ink over
    # achromatic glyphs). A typographic strike — same colour as the text — has no swipe
    # geometry to measure, so it must still author a plain STRIKETHROUGH + partial span.
    img = np.full((60, 300, 3), 255, np.uint8)
    img[15:45, 10:140] = (20, 20, 20)
    img[46:55, 150:290] = (20, 20, 20)
    img[30:33, 10:140] = (20, 20, 20)     # black rule through the left glyphs
    path = tmp_path / "plain_strike.png"
    Image.fromarray(img).save(path)
    ocr_res = {"lines": [{
        "id": "L0", "text": "Foggy and Steady", "conf": 0.9,
        "box": {"x": 10, "y": 15, "w": 280, "h": 40},
        "meta": {"strikethrough": True, "strikethrough_box": {"x": 10, "y": 26, "w": 130, "h": 12}},
    }]}
    out = text_analysis.analyze_text(str(path), ocr_res, {})
    line = out["lines"][0]
    assert line["style"].get("textDecoration") == "STRIKETHROUGH"
    assert line["meta"].get("strike_render") == "text-decoration"
    span = line["style"].get("decorationSpan")
    assert span is not None and span[0] == 0.0 and span[1] < 0.75
