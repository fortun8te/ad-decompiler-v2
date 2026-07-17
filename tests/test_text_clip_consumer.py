"""ad 066: build_design_json consumes icon_detect's meta.row.text_clip_x.

OCR read the checklist ✗/✓ bullets as a leading letter and swallowed them into the
line box, so the emitted text node started ON the icon ("Xmudges…"). icon_detect
publishes row.text_clip_x (icon-right + the list's own clean gap); build() resolves
it onto the owning text candidate (OCR line id -> block via meta.line_ids) and the
text branch shifts the FINAL generous box's left edge onto it.
"""

import pytest
from PIL import Image

from src import build_design_json


def _plate(tmp_path, w=1000, h=400):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (w, h), "white").save(background)
    return str(background)


def _icon_candidate(clip_x, text_id="L10", icon_x=824.0, *, box_y=100.0):
    return {
        "id": f"c_E_{text_id}", "target": "icon", "z": 5,
        "box": {"x": icon_x, "y": box_y, "w": 40.0, "h": 40.0},
        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h40v40z"/></svg>',
        "meta": {
            "role": "icon",
            "row": {
                "text_id": text_id,
                "overlaps_text": True,
                "text_clip_x": clip_x,
                "line_box": {"x": icon_x, "y": box_y, "w": 200.0, "h": 40.0},
            },
        },
    }


def _text_candidate(text_id="L10", *, align="LEFT", x=824.0, w=300.0, y=100.0,
                    text="Smudges on upper lid"):
    return {
        "id": "c_B1", "target": "text", "z": 6, "text": text,
        "box": {"x": x, "y": y, "w": w, "h": 40.0},
        "style": {"fontSize": 20, "align": align},
        "meta": {"source": "ocr", "ocr_id": text_id, "line_ids": [text_id]},
    }


def _text_layer(doc, layer_id="c_B1"):
    return next(layer for layer in doc.layers if layer.id == layer_id)


def test_clip_shifts_box_left_edge_off_the_icon(tmp_path):
    background = _plate(tmp_path)
    doc = build_design_json.build(
        [_icon_candidate(877.0), _text_candidate()],
        {"w": 1000, "h": 400}, str(tmp_path), base_src=background,
    )
    layer = _text_layer(doc)
    assert layer.meta.get("text_clip_applied") == pytest.approx(877.0)
    # Left edge moved onto the clip x; the box still swallows its pre-fit ink.
    assert layer.box["x"] == pytest.approx(877.0)
    assert layer.box["w"] >= layer.meta["prefit_ink_box"]["w"] - 0.01


def test_clip_preserves_y_and_h(tmp_path):
    background = _plate(tmp_path)
    clipped = _text_layer(build_design_json.build(
        [_icon_candidate(877.0), _text_candidate()],
        {"w": 1000, "h": 400}, str(tmp_path / "a"), base_src=background,
    ))
    control = _text_layer(build_design_json.build(
        [_text_candidate()],
        {"w": 1000, "h": 400}, str(tmp_path / "b"), base_src=background,
    ))
    # leadingTrim=CAP_HEIGHT anchors the first-line cap-top at box.y: untouched.
    assert clipped.box["y"] == pytest.approx(control.box["y"])
    assert clipped.box["h"] == pytest.approx(control.box["h"])


def test_right_aligned_text_is_untouched(tmp_path):
    background = _plate(tmp_path)
    doc = build_design_json.build(
        [_icon_candidate(877.0), _text_candidate(align="RIGHT")],
        {"w": 1000, "h": 400}, str(tmp_path / "a"), base_src=background,
    )
    layer = _text_layer(doc)
    assert "text_clip_applied" not in layer.meta
    # RIGHT anchors on the right edge: the box must match the no-icon control exactly.
    control = _text_layer(build_design_json.build(
        [_text_candidate(align="RIGHT")],
        {"w": 1000, "h": 400}, str(tmp_path / "b"), base_src=background,
    ))
    assert layer.box["x"] == pytest.approx(control.box["x"])
    assert layer.box["w"] == pytest.approx(control.box["w"])


def test_clip_left_of_box_is_a_noop(tmp_path):
    background = _plate(tmp_path)
    doc = build_design_json.build(
        [_icon_candidate(800.0, icon_x=740.0), _text_candidate()],
        {"w": 1000, "h": 400}, str(tmp_path), base_src=background,
    )
    layer = _text_layer(doc)
    assert "text_clip_applied" not in layer.meta


def test_multiple_rows_take_the_max_clip(tmp_path):
    background = _plate(tmp_path)
    block = _text_candidate()
    block["meta"]["line_ids"] = ["L10", "L11"]
    block["text"] = "Smudges on upper lid\nUp to 3 shades"
    doc = build_design_json.build(
        [_icon_candidate(877.0, "L10", box_y=100.0),
         _icon_candidate(901.0, "L11", box_y=140.0),
         block],
        {"w": 1000, "h": 400}, str(tmp_path), base_src=background,
    )
    layer = _text_layer(doc)
    assert layer.meta.get("text_clip_applied") == pytest.approx(901.0)
    assert layer.box["x"] == pytest.approx(901.0)


def test_center_label_on_button_shell_is_not_recentred(tmp_path):
    background = _plate(tmp_path)
    group = {
        "id": "g_button", "target": "group", "z": 4,
        "box": {"x": 700.0, "y": 80.0, "w": 320.0, "h": 60.0},
        "meta": {"role": "button", "button_shell": True,
                 "absolute_box": {"x": 700.0, "y": 80.0, "w": 320.0, "h": 60.0}},
        "children": [
            {
                "id": "c_B1", "target": "text", "z": 6, "text": "SHOP NOW",
                # parent-relative (layout._relativize): absolute x = 700 + 124 = 824
                "box": {"x": 124.0, "y": 20.0, "w": 200.0, "h": 40.0},
                "style": {"fontSize": 20, "align": "CENTER"},
                "meta": {"source": "ocr", "ocr_id": "L10", "line_ids": ["L10"],
                         "absolute_box": {"x": 824.0, "y": 100.0, "w": 200.0, "h": 40.0}},
            },
        ],
    }
    doc = build_design_json.build(
        [_icon_candidate(877.0), group],
        {"w": 1000, "h": 400}, str(tmp_path), base_src=background,
    )
    label = next(
        child for layer in doc.layers if layer.id == "g_button"
        for child in (layer.children or []) if child.id == "c_B1"
    )
    assert "text_clip_applied" not in label.meta


def test_missing_prefit_ink_falls_back_to_visible_ink_width():
    # Direct helper probe: no pre-fit ink evidence -> floor is visible ink w + pad.
    generous = {"x": 824.0, "y": 100.0, "w": 200.0, "h": 64.0}
    candidate = {"visible_box": {"x": 824.0, "y": 100.0, "w": 250.0, "h": 40.0}}
    applied = build_design_json._apply_text_clip(
        generous, {"x": 824.0, "y": 100.0, "w": 0.0, "h": 40.0},
        candidate, {"align": "LEFT"}, {"text_clip_x": 877.0},
    )
    assert applied == pytest.approx(877.0)
    assert generous["x"] == pytest.approx(877.0)
    assert generous["w"] == pytest.approx(250.0 + 2.0)
    # y/h never move (leadingTrim=CAP_HEIGHT invariant).
    assert generous["y"] == pytest.approx(100.0)
    assert generous["h"] == pytest.approx(64.0)


def test_merge_row_bullet_clipped_signal_is_consumed(tmp_path):
    # Layout may absorb the icon into a baked checklist shell (066): then only
    # merge's resolved clip target survives, on the TEXT candidate itself.
    background = _plate(tmp_path)
    text = _text_candidate()
    text["meta"]["row_bullet_clipped"] = {"from_x": 824.0, "to_x": 877.0}
    doc = build_design_json.build(
        [text], {"w": 1000, "h": 400}, str(tmp_path), base_src=background,
    )
    layer = _text_layer(doc)
    assert layer.meta.get("text_clip_applied") == pytest.approx(877.0)
    assert layer.box["x"] == pytest.approx(877.0)
