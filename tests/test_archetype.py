from src.archetype import apply_preset, classify, image_facts, scene_facts


def chosen(**facts):
    return classify(facts)["archetype"]


def test_social_screenshot_from_scene_chrome_not_filename():
    facts = scene_facts(
        {"w": 1080, "h": 1920},
        {"lines": [{"text": "Started using this and finally feel better"},
                   {"text": "14:07 · 6.9K Views"}]},
        {"photo_coverage": .58, "social_header": True, "avatar_present": True},
    )
    result = classify(facts)
    assert result["archetype"] == "social_screenshot"
    assert result["preset"]["social_header"]["unreadable_identity"] == "masked_raster_cluster"
    assert result["preset"]["photo_regions"]["suppress_descendants"] is True


def test_caption_over_photo_beats_generic_lifestyle_for_repeated_backplates():
    assert chosen(photo_coverage=.82, text_backplate_count=4, caption_language=True) == "caption_over_photo"


def test_comparison_grid_from_columns_divider_or_labels():
    assert chosen(photo_coverage=.45, column_count=2, center_divider=True) == "comparison_grid"
    assert chosen(photo_coverage=.50, before_after_labels=True) == "comparison_grid"


def test_lifestyle_overlay_for_photo_annotations():
    assert chosen(photo_coverage=.91, leader_lines=True, circular_inset=True) == "lifestyle_overlay"


def test_product_on_flat_for_packshot_field():
    assert chosen(photo_coverage=.28, flat_background_fraction=.86, product_count=1) == "product_on_flat"


def test_configured_override_is_deterministic():
    result = classify({"social_metadata": True}, configured="comparison_grid")
    assert result["archetype"] == "comparison_grid"
    assert result["reasons"] == ["configured override"]


def test_preset_is_wired_to_downstream_namespaces_without_overwriting_user_gate():
    result = classify({"social_header": True})
    cfg = apply_preset({"routing": {"min_text_fidelity": .51}}, result)
    assert cfg["scene"]["archetype"] == "social_screenshot"
    assert cfg["routing"]["min_text_fidelity"] == .51
    assert cfg["routing"]["photo_regions"]["default_mask"] == "rrect"
    assert cfg["layout"]["scene_grouping"]["header_cluster"] is True
    assert cfg["qa"]["archetype_thresholds"]["text_recall_min"] == .90


def test_image_facts_separates_flat_plate_from_photo(tmp_path):
    import numpy as np
    from PIL import Image

    flat = np.full((120, 120, 3), 245, dtype=np.uint8)
    photo = np.random.default_rng(4).integers(0, 256, (120, 120, 3), dtype=np.uint8)
    Image.fromarray(flat).save(tmp_path / "flat.png")
    Image.fromarray(photo).save(tmp_path / "photo.png")

    flat_facts = image_facts(str(tmp_path / "flat.png"))
    photo_facts = image_facts(str(tmp_path / "photo.png"))

    assert flat_facts["flat_background_fraction"] > .9
    assert photo_facts["photo_coverage"] > .9


def test_social_metadata_beats_generic_caption_language_on_photo():
    result = classify({"social_metadata": True, "caption_language": True,
                       "photo_coverage": .8})
    assert result["archetype"] == "social_screenshot"


def test_emoji_observation_is_derived_from_ocr_text():
    facts = scene_facts({"w": 100, "h": 100}, {"lines": [{"text": "Smells great 😅"}]})
    assert facts["emoji_present"] is True
