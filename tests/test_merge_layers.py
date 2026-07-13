"""test_merge_layers.py — fixture-driven tests for the OCR+element+Qwen fusion.

CPU-only, no heavy deps (merge_layers is pure). Uses the inline fallback router when
src.routing is absent (another builder owns it), which is the case at test time.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import merge_layers  # noqa: E402


CANVAS = {"w": 600, "h": 600}


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _ocr():
    return {
        "lines": [
            {"id": "L0", "text": "BIG SALE", "conf": 0.98,
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}, "role": "headline"},
            {"id": "L1", "text": "engraved on the watch", "conf": 0.8,
             "box": {"x": 320, "y": 400, "w": 150, "h": 22}},
        ]
    }


def _elements():
    return [
        # a button/card behind the headline
        {"id": "E0", "box": {"x": 20, "y": 20, "w": 260, "h": 90}, "kind": "shape",
         "area": 20000, "coverage": 0.1, "source": "residual-cc"},
        # the product photo region (scene text lives inside it)
        {"id": "E1", "box": {"x": 300, "y": 300, "w": 240, "h": 260},
         "kind": "photo-fragment", "area": 40000, "coverage": 0.2,
         "source": "residual-cc", "role": "product"},
        # a small icon
        {"id": "E2", "box": {"x": 500, "y": 40, "w": 40, "h": 40}, "kind": "icon",
         "area": 1200, "coverage": 0.006, "source": "residual-cc"},
    ]


def _qwen():
    return [
        {"id": "Q0", "box": {"x": 300, "y": 300, "w": 240, "h": 260},
         "png": "qwen_layers/Q0.png", "kind_hint": "photo"},
    ]


def test_targets_assigned():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    assert m["c_L0"]["target"] == "text"
    assert m["c_E0"]["target"] == "shape"
    assert m["c_E2"]["target"] == "icon"
    # photo element gets Qwen alpha + image target + z from the qwen layer
    assert m["c_E1"]["target"] == "image"
    assert m["c_E1"]["src"] == "qwen_layers/Q0.png"
    assert m["c_E1"]["meta"]["qwen_id"] == "Q0"


def test_scene_text_dropped():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    # L1 sits inside the photo region -> kept in photo, routed to drop
    assert m["c_L1"]["meta"].get("kept_in_photo") is True
    assert m["c_L1"]["target"] == "drop"
    assert m["c_L1"]["meta"]["role"] == "scene-text"


def test_headline_text_survives_and_is_editable():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    assert "c_L0" in m
    assert m["c_L0"]["text"] == "BIG SALE"
    # text sits above shapes
    assert m["c_L0"]["z"] >= m["c_E0"]["z"]
    assert m["c_L0"]["meta"]["overlay_text"] is True
    assert m["c_L0"]["meta"]["removal_required"] is True


def test_platform_lockup_is_a_separate_cropped_logo_not_baked_into_card():
    ocr = {"lines": [{
        "id": "X", "text": "X.com", "conf": .99,
        "box": {"x": 450, "y": 42, "w": 90, "h": 24},
        "meta": {"ownership_decision": {"placement": "ui_metadata", "owner": "card",
                                         "action": "raster_keep", "confidence": .99}},
    }]}
    merged = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))
    x = merged["c_X"]
    assert x["target"] == "image"
    assert x["meta"]["platform_lockup"] is True
    assert x["meta"]["role"] == "platform-logo"
    assert not x["meta"].get("kept_in_photo")


def test_unlabelled_overlay_text_receives_a_semantic_figma_role():
    ocr = {"lines": [{"id": "L", "text": "SHOP NOW", "conf": .99,
                      "box": {"x": 40, "y": 400, "w": 180, "h": 35},
                      "meta": {"ownership_decision": {"placement": "overlay", "owner": "none",
                                                          "action": "recreate", "confidence": .99}}}]}
    layer = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))["c_L"]
    assert layer["target"] == "text"
    assert layer["meta"]["role"] == "cta"


def test_shape_that_is_pure_text_box_is_deduped():
    """A 'shape' element whose box is essentially an OCR text box is dropped in
    favor of the editable text candidate."""
    ocr = {"lines": [{"id": "L0", "text": "CLICK", "conf": 0.95,
                      "box": {"x": 100, "y": 100, "w": 120, "h": 40}}]}
    elements = [{"id": "E0", "box": {"x": 99, "y": 99, "w": 122, "h": 42},
                 "kind": "shape", "area": 5100, "coverage": 0.02,
                 "source": "residual-cc"}]
    cands = merge_layers.merge(ocr, elements, [], CANVAS, {})
    ids = {c["id"] for c in cands}
    assert "c_L0" in ids
    assert "c_E0" not in ids  # deduped away


def test_qwen_only_layer_becomes_image():
    qwen = [{"id": "Q0", "box": {"x": 10, "y": 10, "w": 100, "h": 100},
             "png": "qwen_layers/Q0.png", "kind_hint": "object"}]
    cands = merge_layers.merge({"lines": []}, [], qwen, CANVAS, {})
    m = _by_id(cands)
    assert "c_Q0" in m
    assert m["c_Q0"]["target"] == "image"
    assert m["c_Q0"]["src"] == "qwen_layers/Q0.png"


def test_empty_inputs():
    assert merge_layers.merge({"lines": []}, [], [], CANVAS, {}) == []


def test_paragraph_block_preserves_wrapping_alignment_line_height_and_baselines():
    lines = [
        {"id": "L0", "text": "First", "conf": .99,
         "box": {"x": 20, "y": 20, "w": 100, "h": 20},
         "baseline": {"x0": 20, "y0": 36, "x1": 120, "y1": 36},
         "style": {"fontFamily": "Inter", "fontSize": 16}},
        {"id": "L1", "text": "Second", "conf": .98,
         "box": {"x": 20, "y": 44, "w": 110, "h": 20},
         "baseline": {"x0": 20, "y0": 60, "x1": 130, "y1": 60},
         "style": {"fontFamily": "Inter", "fontSize": 16}},
    ]
    block = {
        "id": "B0", "line_ids": ["L0", "L1"], "text": "First\nSecond",
        "box": {"x": 20, "y": 20, "w": 110, "h": 44},
        "painted_box": {"x": 22, "y": 22, "w": 106, "h": 40},
        "alignment": "CENTER", "line_height": 24, "role": "body", "meta": {},
    }
    candidate = _by_id(merge_layers.merge(
        {"lines": lines, "blocks": [block], "styles": []}, [], [], CANVAS, {}
    ))["c_B0"]
    assert candidate["style"]["lineCount"] == 2
    assert candidate["style"]["lineHeight"] == 24
    assert candidate["style"]["align"] == "CENTER"
    assert candidate["meta"]["baseline_first"]["y0"] == 36
    assert candidate["meta"]["baseline_last"]["y0"] == 60


def test_partial_block_list_cannot_delete_orphan_ocr_lines():
    lines = [
        {"id": "L0", "text": "Headline", "conf": .99,
         "box": {"x": 20, "y": 20, "w": 120, "h": 30}},
        {"id": "L1", "text": "SHOP NOW", "conf": .98,
         "box": {"x": 20, "y": 100, "w": 110, "h": 24}, "role": "cta"},
    ]
    blocks = [{"id": "B0", "line_ids": ["L0"], "text": "Headline",
               "box": dict(lines[0]["box"]), "painted_box": dict(lines[0]["box"]),
               "role": "headline", "meta": {}}]
    merged = merge_layers.merge({"lines": lines, "blocks": blocks}, [], [], CANVAS, {})
    texts = {candidate.get("text") for candidate in merged}
    assert {"Headline", "SHOP NOW"} <= texts


def test_low_fidelity_block_meta_survives_the_block_path_into_the_candidate():
    """Regression for the block-path fidelity drop: production OCR always carries a
    non-empty "blocks" array (text_analysis._make_blocks emits >=1 block per line), so
    _text_sources always prefers blocks over lines. A block produced by a real
    low-fidelity line must still carry meta.low_fidelity/fallback_src through to the
    merged candidate so routing.py can gate it to a masked-pixel fallback instead of
    guessed text."""
    line = {
        "id": "L0", "text": "SALE", "conf": 0.9,
        "box": {"x": 40, "y": 30, "w": 220, "h": 60},
        "meta": {
            "fidelity_confidence": 0.1,
            "low_fidelity": True,
            "fidelity_reason": "ink_confidence:0.10<0.30",
            "fallback_src": "fallback_crops/L0.png",
            "substitution": {"from": "text", "to": "masked-pixel-fallback",
                              "reason": "ink_confidence:0.10<0.30", "confidence": 0.1},
        },
    }
    block = {
        "id": "B0", "type": "text", "line_ids": ["L0"], "text": "SALE",
        "box": dict(line["box"]), "painted_box": dict(line["box"]),
        "alignment": "left", "line_height": 20.0, "role": "text",
        "hierarchy": {"level": 0, "parent_id": None}, "style_id": None,
        "meta": {
            "fidelity_confidence": 0.1,
            "low_fidelity": True,
            "fidelity_reason": "ink_confidence:0.10<0.30",
            "fallback_src": "fallback_crops/L0.png",
            "substitution": {"from": "text", "to": "masked-pixel-fallback",
                              "reason": "ink_confidence:0.10<0.30", "confidence": 0.1},
        },
    }
    ocr = {"lines": [line], "blocks": [block], "styles": []}

    cands = merge_layers.merge(ocr, [], [], CANVAS, {})
    m = _by_id(cands)
    assert "c_B0" in m
    meta = m["c_B0"]["meta"]
    assert meta.get("low_fidelity") is True
    assert meta.get("fallback_src") == "fallback_crops/L0.png"
    # routing.py must gate this to the masked-pixel fallback, not guessed text, and it
    # must use the real saved crop (src) rather than falling into the genericmask-only
    # "else" branch that loses the actual pixels.
    assert m["c_B0"]["target"] == "image"
    assert m["c_B0"].get("src") == "fallback_crops/L0.png"


def test_vlm_font_winner_on_line_overrides_stale_block_font():
    line = {
        "id": "L0", "text": "Post body", "conf": .9,
        "box": {"x": 10, "y": 20, "w": 200, "h": 30},
        "vlm_font_judged": True,
        "style": {"fontFamily": "Arial", "fontStyle": "Regular", "fontWeight": 400,
                  "fontCandidates": [{"family": "Arial", "vlm_score": 9}]},
    }
    block = {
        "id": "B0", "text": "Post body", "line_ids": ["L0"],
        "box": dict(line["box"]), "style": {"fontFamily": "Comic Sans MS"},
    }
    merged = merge_layers.merge({"lines": [line], "blocks": [block]}, [], [], CANVAS, {})
    text = next(item for item in merged if item["id"] == "c_B0")
    assert text["style"]["fontFamily"] == "Arial"
