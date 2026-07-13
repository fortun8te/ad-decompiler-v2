"""Spine tests — routing, wordmark, build_design_json, pixel_diff. Pure-CPU, no GPU deps."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import routing, build_design_json, pixel_diff
from src.wordmark import is_wordmark_candidate, is_platform_lockup, partition_wordmarks, semantic_text_role

CANVAS = {"w": 1080, "h": 1350}


def test_scene_text_dropped():
    c = {"id": "L1", "text": "ingredienten", "box": {"x": 200, "y": 900, "w": 180, "h": 30},
         "meta": {"origin": "scene"}}
    assert routing.route(c, CANVAS)["target"] == "drop"


def test_overlay_text_is_text():
    c = {"id": "L2", "text": "Save 30% today", "box": {"x": 120, "y": 500, "w": 400, "h": 60},
         "meta": {"origin": "overlay"}}
    assert routing.route(c, CANVAS)["target"] == "text"


def test_low_fidelity_text_falls_back_to_masked_pixel_layer_not_guessed_text():
    c = {
        "id": "L9", "text": "muddled copy", "box": {"x": 120, "y": 500, "w": 400, "h": 60},
        "style": {"confidence": 0.12},
        "meta": {"origin": "overlay", "low_fidelity": True, "fidelity_confidence": 0.12,
                 "fidelity_reason": "ink_confidence:0.12<0.30", "fallback_src": "text_fallback/L9.png"},
    }
    r = routing.route(c, CANVAS)
    assert r["target"] == "image"
    assert r["src"] == "text_fallback/L9.png"
    assert r["meta"]["substitution"]["from"] == "text"
    assert r["meta"]["substitution"]["to"] == "image"


def test_confident_text_style_confidence_alone_does_not_trigger_fallback():
    c = {"id": "L10", "text": "Save 30% today", "box": {"x": 120, "y": 500, "w": 400, "h": 60},
         "style": {"confidence": 0.91}, "meta": {"origin": "overlay"}}
    assert routing.route(c, CANVAS)["target"] == "text"


def test_ambiguous_font_match_uses_exact_raster_fallback_by_default():
    c = {
        "id": "L11", "text": "Display headline",
        "box": {"x": 120, "y": 120, "w": 620, "h": 100},
        "style": {"confidence": 0.96},
        "meta": {"origin": "overlay", "fidelity_confidence": 0.80},
    }
    result = routing.route(c, CANVAS)
    assert result["target"] == "image"
    assert result["meta"]["fallback"] is True
    assert result["meta"]["substitution"]["reason"] == "low-confidence font/effect match"


def test_wordmark_becomes_artwork_not_text():
    c = {"id": "L3", "text": "grüns", "box": {"x": 460, "y": 40, "w": 160, "h": 70}}
    r = routing.route(c, CANVAS)
    assert r["target"] in ("image", "icon") and r["meta"].get("wordmark") is True


def test_heart_wordmark_detected():
    line = {"id": "L4", "text": "♡ Hears Earplugs", "box": {"x": 380, "y": 30, "w": 320, "h": 48}}
    assert is_wordmark_candidate(line, CANVAS) is True


def test_social_ui_labels_and_handles_are_not_wordmarks():
    cases = [
        {"id": "post", "text": "Post", "box": {"x": 487, "y": 55, "w": 101, "h": 35}},
        {"id": "following", "text": "Volgend", "box": {"x": 867, "y": 154, "w": 133, "h": 31}},
        {"id": "handle", "text": "@UpfrontFood", "box": {"x": 185, "y": 198, "w": 226, "h": 30}},
    ]
    assert all(is_wordmark_candidate(line, {"w": 1080, "h": 1080}) is False for line in cases)
    assert all(routing.route(line, {"w": 1080, "h": 1080})["target"] == "text" for line in cases)


def test_x_dot_com_platform_lockup_is_wordmark_in_top_right():
    line = {"id": "x", "text": "X.com", "box": {"x": 900, "y": 100, "w": 140, "h": 35}}
    assert is_wordmark_candidate(line, {"w": 1080, "h": 1920}) is True
    assert is_platform_lockup(line, {"w": 1080, "h": 1920}) is True
    assert semantic_text_role(line, {"w": 1080, "h": 1920}) == "platform-logo"


def test_body_copy_is_not_wordmark():
    line = {"id": "L5", "text": "Clinically proven to reduce noise by 20 decibels overall.",
            "box": {"x": 80, "y": 300, "w": 900, "h": 40}}
    assert is_wordmark_candidate(line, CANVAS) is False


def test_photo_fragment_is_image_with_mask():
    c = {"id": "E1", "kind": "photo-fragment", "box": {"x": 40, "y": 200, "w": 500, "h": 500}}
    r = routing.route(c, CANVAS)
    assert r["target"] == "image" and r.get("mask")


def test_small_icon_vectorized_big_not():
    small = routing.route({"id": "E2", "kind": "icon", "box": {"x": 10, "y": 10, "w": 60, "h": 60}}, CANVAS)
    big = routing.route({"id": "E3", "kind": "icon", "box": {"x": 0, "y": 0, "w": 900, "h": 900}}, CANVAS)
    assert small["target"] == "icon" and big["target"] != "icon"


def test_build_design_json_orders_and_keeps_scene():
    cands = [
        {"id": "L2", "text": "BUY NOW", "target": "text", "box": {"x": 100, "y": 1200, "w": 300, "h": 60},
         "style": {"fontSize": 40}, "meta": {"role": "cta", "origin": "overlay"}},
        {"id": "L1", "text": "ingredienten", "target": "drop", "box": {"x": 200, "y": 900, "w": 180, "h": 30}},
        {"id": "E1", "kind": "photo-fragment", "target": "image", "box": {"x": 40, "y": 200, "w": 500, "h": 500}},
    ]
    with tempfile.TemporaryDirectory() as d:
        doc = build_design_json.build(cands, CANVAS, d)
        types = [L.type for L in doc.layers]
        assert "ingredienten" in doc.kept_in_photo          # scene text kept, not a layer
        assert doc.layers[-1].type == "text"                # text painted last (front)
        assert os.path.exists(os.path.join(d, "design.json"))
        assert all(t in ("text", "image", "shape") for t in types)


def test_build_design_json_wires_gradient_fill_and_stroke_for_text_layer():
    cands = [
        {"id": "L5", "text": "OFF", "target": "text", "box": {"x": 10, "y": 10, "w": 200, "h": 80},
         "style": {
             "fontSize": 60, "color": "#0f0f0f",
             "fill": {"kind": "linear", "angle": 90.0,
                      "stops": [{"offset": 0.0, "color": "#eb3c14"}, {"offset": 1.0, "color": "#1446eb"}]},
             "stroke": {"kind": "flat", "color": "#0f0f0f", "width": 2.0},
         },
         "meta": {"role": "headline", "origin": "overlay"}},
    ]
    with tempfile.TemporaryDirectory() as d:
        doc = build_design_json.build(cands, CANVAS, d)
        text_layer = next(L for L in doc.layers if L.type == "text")
        assert text_layer.fill["kind"] == "linear"
        assert text_layer.fill["stops"][0]["color"] == "#eb3c14"
        assert text_layer.stroke["color"] == "#0f0f0f"
        # the paint description is not duplicated inside the style dict
        assert "fill" not in text_layer.style
        assert "stroke" not in text_layer.style


def test_build_design_json_records_text_fidelity_substitution_warning():
    cands = [
        {"id": "L6", "text": "muddled copy", "target": "image", "box": {"x": 10, "y": 10, "w": 200, "h": 40},
         "meta": {"low_fidelity": True,
                  "substitution": {"from": "text", "to": "image",
                                    "reason": "ink_confidence:0.10<0.30", "confidence": 0.10}}},
    ]
    with tempfile.TemporaryDirectory() as d:
        doc = build_design_json.build(cands, CANVAS, d)
        image_layer = next(L for L in doc.layers if L.type == "image")
        assert "fallback" in image_layer.name.lower() or "Text (fallback)" in image_layer.name
        warning_codes = [w.get("code") for w in doc.meta["warnings"]]
        assert "text-fidelity-fallback" in warning_codes


def test_partition_wordmarks():
    lines = [
        {"id": "a", "text": "UPFRONT", "box": {"x": 440, "y": 30, "w": 200, "h": 60}},
        {"id": "b", "text": "The only supplement you need every morning", "box": {"x": 80, "y": 400, "w": 900, "h": 40}},
    ]
    p = partition_wordmarks(lines, CANVAS)
    assert [w["id"] for w in p["wordmarks"]] == ["a"] and [t["id"] for t in p["text"]] == ["b"]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print("ok  ", fn.__name__)
        except Exception:
            print("FAIL", fn.__name__); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
