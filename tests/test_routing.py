from src import routing


CANVAS = {"w": 1000, "h": 1000}


def test_printed_on_product_text_is_scene_owned_and_dropped():
    out = routing.route(
        {"id": "L0", "text": "ORIGINAL", "kind": "text",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20},
         "meta": {"scene_text_role": "printed_on_product"}},
        CANVAS,
    )
    assert out["target"] == "drop"
    assert out["meta"]["kept_in_photo"] is True


def test_overlay_copy_remains_native_text():
    out = routing.route(
        {"id": "L1", "text": "BUY NOW", "kind": "text",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20},
         "meta": {"scene_text_role": "overlay_copy"}},
        CANVAS,
    )
    assert out["target"] == "text"


def test_scene_origin_still_wins_over_overlay_like_role():
    out = routing.route(
        {"id": "L2", "text": "LABEL", "kind": "text",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20},
         "meta": {"origin": "scene", "scene_text_role": "overlay_copy"}},
        CANVAS,
    )
    assert out["target"] == "drop"


def test_flat_button_stays_native_primitive_instead_of_being_traced():
    out = routing.route(
        {"id": "B1", "kind": "shape", "box": {"x": 10, "y": 10, "w": 120, "h": 40},
         "radius": 12, "meta": {"role": "button", "flat_fill": True, "simple_graphic": True}},
        CANVAS,
    )
    assert out["target"] == "shape"
    assert out["meta"]["button_shell"] is True


def test_explicit_small_nonprimitive_graphic_is_vectorized():
    out = routing.route(
        {"id": "G1", "kind": "shape", "box": {"x": 10, "y": 10, "w": 40, "h": 40},
         "meta": {"role": "ornament", "simple_graphic": True}},
        CANVAS,
    )
    assert out["target"] == "icon"
