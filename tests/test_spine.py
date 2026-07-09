"""Spine tests — routing, wordmark, build_design_json, pixel_diff. Pure-CPU, no GPU deps."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import routing, build_design_json, pixel_diff
from src.wordmark import is_wordmark_candidate, partition_wordmarks

CANVAS = {"w": 1080, "h": 1350}


def test_scene_text_dropped():
    c = {"id": "L1", "text": "ingredienten", "box": {"x": 200, "y": 900, "w": 180, "h": 30},
         "meta": {"origin": "scene"}}
    assert routing.route(c, CANVAS)["target"] == "drop"


def test_overlay_text_is_text():
    c = {"id": "L2", "text": "Save 30% today", "box": {"x": 120, "y": 500, "w": 400, "h": 60},
         "meta": {"origin": "overlay"}}
    assert routing.route(c, CANVAS)["target"] == "text"


def test_wordmark_becomes_artwork_not_text():
    c = {"id": "L3", "text": "grüns", "box": {"x": 460, "y": 40, "w": 160, "h": 70}}
    r = routing.route(c, CANVAS)
    assert r["target"] in ("image", "icon") and r["meta"].get("wordmark") is True


def test_heart_wordmark_detected():
    line = {"id": "L4", "text": "♡ Hears Earplugs", "box": {"x": 380, "y": 30, "w": 320, "h": 48}}
    assert is_wordmark_candidate(line, CANVAS) is True


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
