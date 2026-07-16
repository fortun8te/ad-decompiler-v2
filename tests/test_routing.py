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
    assert out["meta"]["overlay_text"] is True
    assert out["meta"]["removal_required"] is True


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


def test_text_bearing_logo_shell_routes_to_image_cutout_not_shape():
    """Sale seals mislabeled as logo → exact IMAGE cutout (OCR baked upstream)."""
    out = routing.route(
        {"id": "E014", "kind": "icon", "box": {"x": 774, "y": 540, "w": 256, "h": 254},
         "meta": {"role": "logo", "text_bearing_shell": True, "plate_shell": True}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["meta"]["role"] == "badge"
    assert out["meta"]["shell_raster_chip"] is True
    assert out["meta"]["baked_badge_text"] is True
    assert out["meta"].get("reclassified_from") == "logo"
    assert out["mask"]["kind"] == "alpha"


def test_text_bearing_wide_shape_routes_as_banner_plate():
    """Wide brushstroke banners stay editable SHAPE plates (not chrome-as-raster)."""
    out = routing.route(
        {"id": "E_ban", "kind": "shape", "box": {"x": 80, "y": 220, "w": 920, "h": 140},
         "meta": {"role": "shape", "text_bearing_shell": True, "plate_shell": True}},
        {"w": 1080, "h": 1350},
    )
    assert out["target"] == "shape"
    assert out["meta"]["role"] == "banner"
    assert out["meta"]["plate_shell"] is True
    assert out["meta"].get("shell_raster_chip") is not True


def test_badge_icon_wordmark_always_image_under_chrome_as_raster():
    for role, kind in (("badge", "icon"), ("icon", "icon"), ("wordmark", "icon"),
                       ("starburst", "icon"), ("price_burst", "icon")):
        out = routing.route(
            {"id": role, "kind": kind, "box": {"x": 20, "y": 20, "w": 120, "h": 120},
             "meta": {"role": role}},
            CANVAS,
        )
        assert out["target"] == "image", role
        assert out["meta"].get("shell_raster_chip") is True, role


def test_chrome_as_raster_can_be_disabled_for_legacy_shells():
    out = routing.route(
        {"id": "E014", "kind": "icon", "box": {"x": 774, "y": 540, "w": 256, "h": 254},
         "meta": {"role": "logo", "text_bearing_shell": True, "plate_shell": True}},
        CANVAS,
        {"routing": {"chrome_as_raster": False}},
    )
    assert out["target"] == "shape"
    assert out["meta"]["role"] == "badge"


def test_thin_divider_stays_a_native_bar_instead_of_vector_tracing():
    out = routing.route(
        {"id": "rule", "kind": "divider", "box": {"x": 10, "y": 10, "w": 180, "h": 2},
         "meta": {"role": "divider"}},
        CANVAS,
    )
    assert out["target"] == "shape"
    assert out["shape_kind"] == "rect"
    assert out["meta"]["native_divider"] is True


def test_diagonal_callout_line_keeps_gated_vector_route_not_divider_rectangle():
    out = routing.route(
        {"id": "leader", "kind": "line", "rotation": -18,
         "box": {"x": 30, "y": 42, "w": 92, "h": 3},
         "meta": {"role": "callout_leader"}},
        CANVAS,
    )
    assert out["target"] == "icon"
    assert "shape_kind" not in out
    assert out["meta"].get("native_divider") is not True


def test_intentional_raster_cluster_roles_stay_named_swappable_source_crops():
    for role in ("screenshot", "ui_panel", "receipt", "chart", "graph", "table",
                 "nutrition_panel", "diagram", "infographic", "product_cluster"):
        out = routing.route(
            {"id": role, "kind": "photo-fragment", "box": {"x": 10, "y": 10, "w": 220, "h": 140},
             "meta": {"role": role}},
            CANVAS,
        )
        assert out["target"] == "image"
        assert out["mask"]["kind"] == "rrect"
        assert out["meta"]["intentional_raster_cluster"] is True
        assert out["meta"]["swappable"] is True
        assert out["meta"]["semantic_name"]


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


def test_wordmark_defaults_to_exact_raster_when_config_is_empty():
    out = routing.route(
        {"id": "W-default", "text": "UpfrontFood", "kind": "text",
         "box": {"x": 185, "y": 198, "w": 226, "h": 30},
         "meta": {"scene_text_role": "wordmark"}},
        CANVAS,
        {},
    )
    assert out["target"] == "image"
    assert out["mask"]["kind"] == "path"


def test_explicit_upstream_mask_shape_is_not_downgraded():
    out = routing.route(
        {"id": "M0", "kind": "photo-fragment", "box": {"x": 0, "y": 0, "w": 80, "h": 80},
         "mask": {"kind": "ellipse"}, "meta": {"role": "photo"}},
        CANVAS,
    )
    assert out["mask"]["kind"] == "ellipse"


def test_foreground_raster_disposition_overrides_generic_detector_shape():
    out = routing.route(
        {"id": "header", "kind": "shape", "box": {"x": 10, "y": 10, "w": 220, "h": 80},
         "mask": {"src": "sam3_masks/header.png"},
         "meta": {"role": "shape", "layer_disposition": "foreground_raster",
                  "z_band": "chrome"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["mask"]["src"] == "sam3_masks/header.png"


def test_foreground_vector_disposition_reaches_vector_fidelity_gate():
    out = routing.route(
        {"id": "leader", "kind": "photo-fragment", "box": {"x": 10, "y": 10, "w": 90, "h": 12},
         "meta": {"role": "photo", "layer_disposition": "foreground_vector",
                  "z_band": "overlay"}},
        CANVAS,
    )
    assert out["target"] == "icon"


def test_plate_disposition_is_never_materialized_as_an_independent_layer():
    out = routing.route(
        {"id": "scene", "kind": "photo-fragment", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "meta": {"role": "photo", "layer_disposition": "plate"}},
        CANVAS,
    )
    assert out["target"] == "drop"
    assert out["meta"]["keep_in_background"] is True


def test_large_ad_arrow_and_callout_leader_reach_vector_gate_not_fake_rectangle():
    """Thin arrows / leaders keep the vector gate; bursts are always-raster chrome."""
    for role in ("arrow", "callout_leader"):
        out = routing.route(
            {"id": role, "kind": "icon", "box": {"x": 20, "y": 20, "w": 360, "h": 300},
             "meta": {"role": role, "flat_fill": True}},
            CANVAS,
        )
        assert out["target"] == "icon", role


def test_price_burst_is_exact_raster_under_chrome_as_raster():
    out = routing.route(
        {"id": "burst", "kind": "icon", "box": {"x": 0, "y": 0, "w": 700, "h": 700},
         "meta": {"role": "price_burst", "flat_fill": True}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["meta"]["shell_raster_chip"] is True


def test_monte_flat_brand_headline_stays_editable_text_not_wordmark_raster():
    """Short brand on cream/black plate with overlay flags → TEXT, not logo IMAGE."""
    out = routing.route(
        {"id": "brand", "text": "MONTE", "kind": "text",
         "box": {"x": 70, "y": 120, "w": 300, "h": 80},
         "meta": {"role": "headline", "overlay_text": True, "removal_required": True}},
        CANVAS,
    )
    assert out["target"] == "text"
    assert not out["meta"].get("wordmark")
    assert out["meta"].get("overlay_text") is True
