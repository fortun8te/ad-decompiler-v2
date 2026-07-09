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
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}},
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
         "source": "residual-cc"},
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
