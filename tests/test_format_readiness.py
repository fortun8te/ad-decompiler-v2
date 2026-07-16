"""Format readiness: aspect classes + capabilities (not new named presets)."""
from __future__ import annotations

import json

from src import archetype
from src.format_readiness import (
    attach_to_decision,
    build_format_profile,
    classify_aspect,
    has_capability,
    infer_capabilities,
    load_format_index,
    planned_image_entry,
    prefers_icon_chips,
    prefers_solid_flat,
)


def test_aspect_classes_cover_common_ad_frames():
    assert classify_aspect(1080, 1920)["aspect_class"] == "story"
    assert classify_aspect(1080, 1350)["aspect_class"] == "portrait"
    assert classify_aspect(1080, 1080)["aspect_class"] == "square"
    assert classify_aspect(1920, 1080)["aspect_class"] == "landscape"
    assert classify_aspect(2400, 600)["aspect_class"] == "wide"


def test_simpletics_ig_caption_gets_story_caption_stack():
    """1080x1920 stacked caption pills → story + text_plates + caption_stack."""
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1920},
        {"lines": [
            {"text": "Didn't know I needed this"},
            {"text": "My gut finally feels calm"},
            {"text": "Wish I started sooner"},
        ]},
        {"photo_coverage": 0.72, "text_backplate_count": 3, "caption_language": True},
    )
    decision = archetype.classify(facts)
    decision["facts"] = facts
    decision = attach_to_decision(decision, {"w": 1080, "h": 1920})
    fmt = decision["format"]
    assert fmt["aspect_class"] == "story"
    assert fmt["capabilities"]["text_plates"] is True
    assert fmt["capabilities"]["caption_stack"] is True
    assert decision["archetype"] == "caption_over_photo"


def test_ui_screenshot_tag_enables_chrome_without_new_preset():
    profile = build_format_profile(
        {"w": 1080, "h": 1920},
        {"aspect_ratio": 1080 / 1920, "flat_background_fraction": 0.4},
        archetype="lifestyle_overlay",
        tags=["ui_screenshot"],
    )
    assert profile["capabilities"]["ui_chrome"] is True
    assert profile["capabilities"]["flat_plate"] is True
    assert profile["capabilities"]["icons_as_chips"] is True


def test_capability_override_can_disable_inference():
    profile = build_format_profile(
        {"w": 100, "h": 100},
        {"social_header": True, "dark_background": True},
        archetype="social_screenshot",
        capability_overrides={"flat_plate": False},
    )
    assert profile["capabilities"]["ui_chrome"] is True
    assert profile["capabilities"]["flat_plate"] is False
    assert "flat_plate" in profile["overrides_applied"]


def test_apply_preset_attaches_scene_format():
    result = archetype.classify({"social_header": True, "aspect_ratio": 0.56})
    result["facts"] = {"social_header": True, "aspect_ratio": 0.56, "dark_background": True}
    cfg = archetype.apply_preset({"format": {"tags": ["ui_screenshot"]}}, result)
    fmt = cfg["scene"]["format"]
    assert fmt["aspect_class"] == "story"
    assert has_capability(cfg, "ui_chrome")
    assert prefers_solid_flat(cfg) is True
    assert prefers_icon_chips(cfg) is True


def test_infer_comparison_columns_from_before_after_facts():
    caps = infer_capabilities(
        {"before_after_pair": True, "before_after_labels": True, "column_count": 2},
        archetype="comparison_grid",
    )
    assert caps["comparison_columns"] is True
    assert caps["flat_plate"] is True


def test_lifestyle_is_not_flat_plate_by_default():
    caps = infer_capabilities(
        {"photo_coverage": 0.9, "flat_background_fraction": 0.2, "leader_lines": True},
        archetype="lifestyle_overlay",
    )
    assert caps["gradients"] is True
    assert caps["diagrams"] is True
    assert caps["flat_plate"] is False


def test_planned_image_entry_reads_format_index(tmp_path):
    from PIL import Image

    img = tmp_path / "201_before_after.png"
    Image.new("RGB", (1080, 1080), (20, 20, 20)).save(img)
    index_path = tmp_path / "format_index.json"
    index_path.write_text(json.dumps({
        "201": {"tags": ["before_after"], "capabilities": {"comparison_columns": True}},
    }), encoding="utf-8")
    index = load_format_index(index_path)
    row = planned_image_entry(img, fixture_id="201", format_index=index)
    assert row["aspect_class"] == "square"
    assert row["format_tags"] == ["before_after"]
    assert row["format_capability_overrides"]["comparison_columns"] is True


def test_routing_and_inpaint_respect_format_capabilities():
    from src.routing import _icons_as_chips
    from src.inpaint import _solid_flat_enabled

    # Lifestyle archetype name alone would NOT chip icons / solid-fill — capability does.
    cfg = {
        "scene": {
            "archetype": "lifestyle_overlay",
            "format": {
                "capabilities": {
                    "icons_as_chips": True,
                    "flat_plate": True,
                    "ui_chrome": False,
                    "text_plates": False,
                    "cutouts": False,
                    "diagrams": False,
                    "gradients": True,
                    "comparison_columns": False,
                    "caption_stack": False,
                },
            },
        },
    }
    assert _icons_as_chips(cfg) is True
    assert _solid_flat_enabled(cfg) is True

    # Explicit opt-out requires chrome_as_raster off (it forces chips when on).
    cfg["routing"] = {"chrome_as_raster": False, "icons_as_chips": False}
    assert _icons_as_chips(cfg) is False


def test_format_index_normalizes_numeric_ids(tmp_path):
    path = tmp_path / "format_index.json"
    path.write_text(json.dumps({"16": {"tags": ["story"]}}), encoding="utf-8")
    loaded = load_format_index(path)
    assert "016" in loaded
    assert loaded["016"]["tags"] == ["story"]


def test_product_packshot_capabilities_cutouts_and_text_plates():
    facts = {
        "product_count": 2,
        "dark_background": True,
        "flat_background_fraction": 0.55,
        "photo_coverage": 0.40,
        "text_backplate_count": 3,
        "mean_luma": 40,
    }
    decision = archetype.classify(facts)
    decision["facts"] = facts
    decision = attach_to_decision(decision, {"w": 1080, "h": 1350})
    caps = decision["format"]["capabilities"]
    assert decision["archetype"] == "product_on_flat"
    assert caps["cutouts"] is True
    assert caps["text_plates"] is True
    assert caps["flat_plate"] is True


def test_dm_chat_ui_enables_ui_chrome_capability():
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1920},
        {"lines": [{"text": "New Messages"}, {"text": "Active now"}]},
        {"dark_background": True, "avatar_present": True},
    )
    decision = archetype.classify(facts)
    decision["facts"] = facts
    cfg = archetype.apply_preset({}, decision)
    assert cfg["scene"]["archetype"] == "social_screenshot"
    assert has_capability(cfg, "ui_chrome")
    assert prefers_solid_flat(cfg) is True
    assert prefers_icon_chips(cfg) is True
