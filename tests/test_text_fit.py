"""Text-box fitting: boxes must grow/shrink so a text run can never clip, and each
text layer must carry a Figma ``autoResize`` hint."""
import json

from src import build_design_json
from src.text_analysis import fit_text_box, _fit_font, _line_advance


def _max_line_advance(text, style):
    font = _fit_font(style, style["fontSize"])
    tracking = float(style.get("letterSpacing", 0) or 0)
    return max((_line_advance(font, line, tracking) for line in text.split("\n")), default=0.0)


def test_single_line_label_grows_width_from_left_anchor():
    text = "The quick brown fox jumps over"
    style = {"fontSize": 40, "align": "LEFT", "letterSpacing": 0}
    box = {"x": 100.0, "y": 50.0, "w": 40.0, "h": 44.0}
    fitted, auto, patch = fit_text_box(text, style, box)
    assert auto == "WIDTH"
    assert patch == {}
    assert fitted["x"] == 100.0                       # left anchor stays put
    assert fitted["w"] >= _max_line_advance(text, style)  # nothing clips
    assert fitted["w"] > box["w"]


def test_single_line_right_aligned_label_grows_leftward():
    text = "121K weergaven"
    style = {"fontSize": 36, "align": "RIGHT", "letterSpacing": 0}
    box = {"x": 900.0, "y": 50.0, "w": 40.0, "h": 40.0}
    fitted, auto, _ = fit_text_box(text, style, box)
    assert auto == "WIDTH"
    # The right edge is the anchor for right-aligned text, so it must not move.
    assert round(fitted["x"] + fitted["w"], 1) == round(box["x"] + box["w"], 1)
    assert fitted["x"] < box["x"]


def test_label_that_already_fits_is_left_unchanged():
    style = {"fontSize": 28, "align": "LEFT", "letterSpacing": 0}
    box = {"x": 10.0, "y": 10.0, "w": 400.0, "h": 40.0}
    fitted, auto, patch = fit_text_box("SALE", style, box)
    assert auto == "WIDTH"
    assert fitted["w"] == 400.0                        # not shrunk
    assert patch == {}


def test_paragraph_shrinks_to_fit_width_and_grows_height():
    text = "Daarbovenop krijgen de eerste 500 bestellingen hun\ngeld terug tot 100 euro."
    style = {"fontSize": 46, "align": "LEFT", "letterSpacing": 0, "lineHeight": 55}
    box = {"x": 0.0, "y": 0.0, "w": 300.0, "h": 60.0}
    fitted, auto, patch = fit_text_box(text, style, box)
    assert auto == "HEIGHT"
    assert fitted["w"] == 300.0                        # fixed-width paragraph
    assert patch.get("fontSize", style["fontSize"]) < style["fontSize"]
    assert fitted["h"] > box["h"]                      # grew to hold both lines
    # After applying the patch, every line fits inside the box width.
    shrunk = {**style, **patch}
    assert _max_line_advance(text, shrunk) <= box["w"] + 1.0


def test_build_design_json_emits_autoresize_hint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    build_design_json.build([{
        "id": "T0", "target": "text", "text": "Hello world",
        "box": {"x": 10, "y": 10, "w": 300, "h": 40},
        "style": {"fontFamily": "Inter", "fontSize": 28, "align": "LEFT"},
    }], {"w": 400, "h": 200}, str(run))
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    assert design["layers"][0]["style"]["autoResize"] in ("WIDTH", "HEIGHT", "NONE")


def test_build_design_json_preserves_unknown_reconstruct_fields(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    build_design_json.build([{
        "id": "P0", "target": "image", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
        "ref": "image-ref-42", "custom_hint": {"keep": True},
    }], {"w": 20, "h": 20}, str(run))
    design = json.loads((run / "design.json").read_text(encoding="utf-8"))
    passthrough = design["layers"][0]["meta"]["passthrough"]
    assert passthrough["ref"] == "image-ref-42"
    assert passthrough["custom_hint"] == {"keep": True}
