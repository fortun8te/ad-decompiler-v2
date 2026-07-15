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


def test_comparison_labels_beat_generic_caption_language_on_photo():
    result = classify({
        "photo_coverage": .59,
        "before_after_labels": True,
        "before_after_pair": True,
        "caption_language": True,
    })
    assert result["archetype"] == "comparison_grid"
    cfg = apply_preset({}, result)
    assert cfg["scene"]["preset"]["photo_regions"]["suppress_descendants"] is False


def test_vs_table_does_not_enable_before_after_photo_rebuild():
    result = classify({"before_after_labels": True, "before_after_pair": False})
    cfg = apply_preset({}, result)
    assert cfg["scene"]["preset"]["photo_regions"]["suppress_descendants"] is True


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


# ── Phase-2 reward contract (docs/HARNESS-PHASE2.md §1c) ─────────────────────────────

def test_every_preset_carries_reward_weights_and_gate_floors():
    from src.archetype import ARCHETYPES, PRESETS

    for name in ARCHETYPES:
        preset = PRESETS[name]
        weights = preset["reward_weights"]
        assert set(weights) == {"local_ssim", "lpips", "text"}
        assert abs(sum(weights.values()) - 1.0) < 1e-6, name
        thresholds = preset["thresholds"]
        assert 0.0 < thresholds["lpips_similarity_min"] <= 1.0, name
        assert 0.0 < thresholds["reward_local_ssim_min"] <= 1.0, name


def test_dark_text_heavy_social_weights_demote_lpips_and_favor_text():
    from src.archetype import PRESETS

    social = PRESETS["social_screenshot"]["reward_weights"]
    product = PRESETS["product_on_flat"]["reward_weights"]
    # A tweet must not need a high global perceptual score; it must need correct text
    # and per-element structure (spec: "must NOT need 0.90 visual SSIM").
    assert social["text"] > social["lpips"]
    assert social["local_ssim"] > social["lpips"]
    assert social["text"] > product["text"]
    assert social["lpips"] < product["lpips"]


def test_apply_preset_exposes_reward_weights_and_gate_floors_to_qa():
    result = classify({"social_header": True})
    cfg = apply_preset({}, result)
    assert cfg["qa"]["reward_weights"] == {"local_ssim": 0.40, "lpips": 0.15, "text": 0.45}
    # F12 recalibration: social floors lifted from 0.20/0.45 into the discriminating gap
    # (measured good social run 009 = LPIPS 0.981 / local 0.601, both clear these).
    assert cfg["qa"]["archetype_thresholds"]["lpips_similarity_min"] == 0.60
    assert cfg["qa"]["archetype_thresholds"]["reward_local_ssim_min"] == 0.50

    from src import qa_reward
    assert qa_reward.reward_weights(cfg) == cfg["qa"]["reward_weights"]
    floors = qa_reward.gate_thresholds(cfg)
    assert floors == {"lpips_similarity_min": 0.60, "local_ssim_min": 0.50}


def test_apply_preset_exposes_text_recall_min_flat_for_metrics_layer():
    # F8: the per-archetype text-recall contract must be readable at qa.text_recall_min,
    # not only buried in archetype_thresholds, so pixel_diff can enforce it.
    for facts, expected in (
        ({"social_header": True}, 0.90),          # social_screenshot
        ({"before_after_labels": True}, 0.93),    # comparison_grid
    ):
        cfg = apply_preset({}, classify(facts))
        assert cfg["qa"]["text_recall_min"] == expected
        # still mirrored inside archetype_thresholds for existing consumers
        assert cfg["qa"]["archetype_thresholds"]["text_recall_min"] == expected


def test_image_facts_flag_dark_backgrounds(tmp_path):
    import numpy as np
    from PIL import Image

    dark = np.full((120, 120, 3), 18, dtype=np.uint8)
    bright = np.full((120, 120, 3), 240, dtype=np.uint8)
    Image.fromarray(dark).save(tmp_path / "dark.png")
    Image.fromarray(bright).save(tmp_path / "bright.png")

    assert image_facts(str(tmp_path / "dark.png"))["dark_background"] is True
    assert image_facts(str(tmp_path / "bright.png"))["dark_background"] is False


def test_dark_chrome_only_boosts_social_alongside_social_evidence():
    with_social = classify({"dark_background": True, "social_metadata": True})
    assert with_social["archetype"] == "social_screenshot"
    assert "dark UI chrome" in with_social["reasons"]
    # A dark poster with no social evidence must not become a tweet.
    alone = classify({"dark_background": True, "flat_background_fraction": .8})
    assert alone["archetype"] != "social_screenshot"
