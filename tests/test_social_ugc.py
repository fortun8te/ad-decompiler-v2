"""Lock social UGC / screenshot chrome patterns (no VLM, unit-only).

Covers X dark posts, IG AMA stickers, circular insets, quote frames, engagement
rows / story swipe-up chips, and meta interpunct restoration.
"""
from __future__ import annotations

from src import archetype, layout
from src.ocr import _restore_interpuncts
from src.routing import route


# ── Archetype / facts ──────────────────────────────────────────────────────────


def test_ama_sticker_ocr_selects_social_screenshot():
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1350},
        {"lines": [{"text": "Ask me anything"}, {"text": "What's your favorite shade?"}]},
    )
    assert facts["ama_sticker"] is True
    result = archetype.classify({**facts, "photo_coverage": 0.4, "flat_background_fraction": 0.5})
    assert result["archetype"] == "social_screenshot"
    assert result["preset"]["grouping"]["ama_sticker"] is True


def test_social_screenshot_preset_exposes_ugc_grouping_and_inter_prior():
    result = archetype.classify({"social_metadata": True, "avatar_present": True})
    cfg = archetype.apply_preset({}, result)
    grouping = cfg["layout"]["scene_grouping"]
    assert grouping["header_cluster"] is True
    assert grouping["message_bubbles"] is True
    assert grouping["engagement_row"] is True
    assert grouping["ama_sticker"] is True
    assert grouping["quote_frame"] is True
    assert grouping["circular_insets_use_ellipse_mask"] is True
    assert cfg["routing"]["avatar_mask"] == "ellipse"
    assert cfg["routing"]["circular_inset_ellipse"] is True
    assert cfg["text_analysis"]["platform_ui_prior"] is True
    assert cfg["text_analysis"]["platform_ui_family"] == "Inter"


def test_caption_over_photo_gets_quote_frame_and_circular_inset_flags():
    result = archetype.classify({
        "caption_language": True, "photo_coverage": 0.7, "text_backplate_count": 2,
    })
    assert result["archetype"] == "caption_over_photo"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"]["quote_frame"] is True
    assert cfg["routing"]["circular_inset_ellipse"] is True


# ── Meta interpuncts ───────────────────────────────────────────────────────────


def test_meta_interpuncts_restore_time_date_views_without_breaking_decimals():
    assert _restore_interpuncts("05:00 PM . 12-05-2026 .") == "05:00 PM · 12-05-2026 ·"
    assert _restore_interpuncts("9:41 AM • May 12 • 1.2M views") == "9:41 AM · May 12 · 1.2M views"
    # Decimal counts after a separator must not become "1 · 2M".
    assert _restore_interpuncts("May 12 . 1.2M views") == "May 12 · 1.2M views"
    assert _restore_interpuncts("121K weergaven") == "121K weergaven"


# ── Layout: X header / engagement / AMA / quote / circular inset ───────────────


def test_engagement_row_groups_icons_and_counts():
    cfg = {
        "scene": {"archetype": "social_screenshot"},
        "layout": {"scene_grouping": {"engagement_row": True}},
    }
    candidates = [
        {"id": "like", "target": "icon", "box": {"x": 40, "y": 900, "w": 36, "h": 36},
         "meta": {"role": "like"}},
        {"id": "lc", "target": "text", "box": {"x": 82, "y": 906, "w": 40, "h": 24},
         "text": "257", "meta": {"role": "meta"}},
        {"id": "rt", "target": "icon", "box": {"x": 160, "y": 900, "w": 36, "h": 36},
         "meta": {"role": "repost"}},
        {"id": "rc", "target": "text", "box": {"x": 202, "y": 906, "w": 40, "h": 24},
         "text": "21K", "meta": {"role": "meta"}},
        {"id": "views", "target": "icon", "box": {"x": 280, "y": 900, "w": 36, "h": 36},
         "meta": {"role": "views"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1080}, cfg)
    rows = [n for n in tree if (n.get("meta") or {}).get("role") == "engagement-row"]
    assert len(rows) == 1
    assert rows[0]["layout"]["mode"] == "HORIZONTAL"
    assert len(rows[0]["children"]) >= 4


def test_ama_sticker_pairs_dark_header_white_body_and_question():
    cfg = {
        "scene": {"archetype": "social_screenshot", "facts": {"ama_sticker": True}},
        "layout": {"scene_grouping": {"ama_sticker": True}},
    }
    candidates = [
        {"id": "hdr", "target": "shape", "box": {"x": 120, "y": 200, "w": 840, "h": 72},
         "fill": {"kind": "flat", "color": "#111111"},
         "meta": {"role": "ama_header"}},
        {"id": "body", "target": "shape", "box": {"x": 120, "y": 272, "w": 840, "h": 220},
         "fill": {"kind": "flat", "color": "#ffffff"},
         "meta": {"role": "ama_body"}},
        {"id": "q", "target": "text", "box": {"x": 160, "y": 320, "w": 760, "h": 80},
         "text": "What's your holy-grail serum?", "meta": {"role": "body"}},
        {"id": "reply", "target": "shape", "box": {"x": 160, "y": 560, "w": 600, "h": 100},
         "fill": {"kind": "flat", "color": "#ffffff"}, "radius": 16,
         "meta": {"role": "card"}},
        {"id": "reply_t", "target": "text", "box": {"x": 180, "y": 580, "w": 540, "h": 48},
         "text": "The vitamin C one, always.", "style": {"align": "LEFT"},
         "meta": {"role": "body"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    stickers = [n for n in tree if (n.get("meta") or {}).get("role") == "ama-sticker"]
    assert len(stickers) == 1
    child_ids = {c.get("id") for c in stickers[0]["children"]}
    assert {"hdr", "body"}.issubset(child_ids)
    # Question may stay nested under the body plate from containment.
    body = next(c for c in stickers[0]["children"] if c.get("id") == "body")
    nested = {c.get("id") for c in (body.get("children") or [])}
    assert "q" in child_ids or "q" in nested
    assert stickers[0]["layout"]["mode"] == "VERTICAL"
    # Reply box still forms a message-bubble / plate outside the sticker.
    assert "reply" not in child_ids


def test_circular_inset_pairs_product_with_white_ring_product_above_stroke():
    cfg = {
        "scene": {"archetype": "lifestyle_overlay"},
        "layout": {"scene_grouping": {"circular_insets_use_ellipse_mask": True}},
        "routing": {"circular_inset_ellipse": True},
    }
    candidates = [
        {"id": "ring", "target": "shape", "box": {"x": 100, "y": 100, "w": 220, "h": 220},
         "stroke": {"color": "#ffffff", "width": 6},
         "meta": {"role": "ring", "stroke_outline_shell": True, "white_ring": True}},
        {"id": "prod", "target": "image", "box": {"x": 118, "y": 118, "w": 184, "h": 184},
         "mask": {"kind": "alpha"},
         "meta": {"role": "product", "circular": True}, "z": 10},
    ]
    tree = layout.infer(candidates, {"w": 800, "h": 800}, cfg)
    groups = [n for n in tree if (n.get("meta") or {}).get("role") == "circular-inset"]
    assert len(groups) == 1
    kids = {c["id"]: c for c in groups[0]["children"]}
    assert set(kids) == {"ring", "prod"}
    assert kids["prod"]["mask"]["kind"] == "ellipse"
    assert kids["prod"].get("z", 0) > layout._node_z(kids["ring"])


def test_quote_frame_groups_stroke_stars_and_text_with_product_above_stroke():
    cfg = {
        "scene": {"archetype": "caption_over_photo"},
        "layout": {"scene_grouping": {"quote_frame": True}},
    }
    candidates = [
        {"id": "frame", "target": "shape", "box": {"x": 80, "y": 200, "w": 500, "h": 360},
         "stroke": {"color": "#ffffff", "width": 2}, "radius": 18,
         "meta": {"role": "quote_frame", "stroke_outline_shell": True}, "z": 20},
        {"id": "stars", "target": "text", "box": {"x": 120, "y": 230, "w": 160, "h": 28},
         "text": "★★★★★", "meta": {"role": "rating"}},
        {"id": "quote", "target": "text", "box": {"x": 120, "y": 280, "w": 400, "h": 120},
         "text": "This serum changed my skin.", "meta": {"role": "quote"}},
        {"id": "hand", "target": "image", "box": {"x": 420, "y": 480, "w": 200, "h": 160},
         "meta": {"role": "product"}, "z": 15},
    ]
    tree = layout.infer(candidates, {"w": 800, "h": 1000}, cfg)
    frames = [n for n in tree if (n.get("meta") or {}).get("role") == "quote-frame"]
    assert len(frames) == 1
    kids = {c["id"]: c for c in frames[0]["children"]}
    assert {"frame", "stars", "quote", "hand"}.issubset(kids)
    assert kids["hand"].get("z", 0) > layout._node_z(kids["frame"])
    assert kids["hand"]["meta"].get("quote_frame_break") is True


# ── Routing: circular inset ellipse + story swipe-up chip ──────────────────────


def test_circular_inset_routes_to_ellipse_mask():
    candidate = {
        "id": "ci", "kind": "photo", "box": {"x": 10, "y": 10, "w": 120, "h": 120},
        "meta": {"role": "circular_inset"},
    }
    routed = route(candidate, {"w": 1000, "h": 1000}, {"scene": {"archetype": "lifestyle_overlay"}})
    assert routed["target"] == "image"
    assert routed["mask"]["kind"] == "ellipse"


def test_story_swipe_up_chrome_chips_on_social_screenshot():
    candidate = {
        "id": "su", "kind": "icon", "box": {"x": 400, "y": 1700, "w": 80, "h": 40},
        "meta": {"role": "swipe_up"},
    }
    routed = route(dict(candidate), {"w": 1080, "h": 1920},
                   {"scene": {"archetype": "social_screenshot"}})
    assert routed["target"] == "image"
    assert routed["meta"].get("icon_chip") is True
    assert routed["meta"].get("story_chrome_chip") is True
