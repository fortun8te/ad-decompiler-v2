from src import routing


CANVAS = {"w": 1000, "h": 1000}


def test_printed_on_product_text_is_scene_owned_and_dropped():
    out = routing.route(
        {"id": "L0", "text": "ORIGINAL", "kind": "text",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20},
         "meta": {"scene_text_role": "printed_on_product", "scene_text_corroborated": True}},
        CANVAS,
    )
    assert out["target"] == "drop"
    assert out["meta"]["kept_in_photo"] is True


def test_uncorroborated_vlm_scene_label_cannot_delete_ocr_text():
    out = routing.route(
        {"id": "L0", "text": "SHOP NOW", "kind": "text",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20},
         "meta": {"scene_text_role": "printed_on_product"}},
        CANVAS,
    )
    assert out["target"] == "text"


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


def test_avatar_photo_routes_to_image_with_ellipse_mask():
    out = routing.route(
        {"id": "AV", "kind": "photo-fragment", "box": {"x": 24, "y": 132, "w": 122, "h": 123},
         "meta": {"role": "avatar"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["mask"]["kind"] == "ellipse"


def test_photo_role_keeps_alpha_mask_and_preserves_src():
    out = routing.route(
        {"id": "P0", "kind": "photo-fragment", "box": {"x": 0, "y": 0, "w": 400, "h": 200},
         "mask": {"src": "fused_elements/E0.png"}, "meta": {"role": "photo"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["mask"]["kind"] == "alpha"
    # The matte path reconstruct needs to load the mask must survive routing.
    assert out["mask"]["src"] == "fused_elements/E0.png"


def test_card_role_routes_to_rounded_rect_mask_with_radius():
    out = routing.route(
        {"id": "C0", "kind": "photo-fragment", "box": {"x": 10, "y": 10, "w": 200, "h": 120},
         "radius": 18, "meta": {"role": "card"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["mask"]["kind"] == "rrect"
    assert out["mask"]["radius"] == 18


def test_wordmark_logo_routes_to_image_with_path_mask():
    out = routing.route(
        {"id": "W0", "text": "UpfrontFood", "kind": "text",
         "box": {"x": 185, "y": 198, "w": 226, "h": 30},
         "meta": {"scene_text_role": "wordmark"}},
        CANVAS, {"wordmark_as_raster": True},
    )
    assert out["target"] == "image"
    assert out["meta"]["wordmark"] is True
    assert out["mask"]["kind"] == "path"


def test_explicit_upstream_mask_shape_is_not_downgraded():
    out = routing.route(
        {"id": "M0", "kind": "photo-fragment", "box": {"x": 0, "y": 0, "w": 80, "h": 80},
         "mask": {"kind": "ellipse"}, "meta": {"role": "photo"}},
        CANVAS,
    )
    assert out["mask"]["kind"] == "ellipse"
