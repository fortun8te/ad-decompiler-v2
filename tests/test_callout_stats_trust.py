"""Unit locks for callouts, stats rows, trust strips, and sale/story CTAs.

Covers: leader endpoint dots, 3-col stats Auto Layout, benefit-pill stacks,
Trustpilot rating strips, AS SEEN IN logo strips, story swipe-up / Get Yours CTAs,
and sale-circle shells. Geometry-only — no VLM.
"""
from __future__ import annotations

from src import layout, merge_layers, raster_clusters, sam3_detect


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _text(node_id, text, box, role="body"):
    return {
        "id": node_id, "target": "text", "text": text, "box": box,
        "style": {"align": "LEFT", "fontSize": 22},
        "meta": {"role": role},
    }


# ── 1) Leader endpoint dots (Wavy beach / 014) ───────────────────────────────────

def test_leader_endpoint_dot_survives_guide_drop_and_pairs():
    """Small filled circle at a leader tip is kept + tagged, never guide-dropped."""
    canvas = {"w": 1080, "h": 1350}
    ocr = {"lines": [
        {"id": "c1", "text": "Soft sand underfoot", "conf": 0.93, "role": "body",
         "box": {"x": 40, "y": 420, "w": 220, "h": 40}},
    ]}
    elements = [
        {"id": "PROD", "box": {"x": 420, "y": 500, "w": 280, "h": 420},
         "kind": "photo-fragment", "role": "product", "area": 100000,
         "coverage": 0.08, "score": 0.95},
        {"id": "L1", "box": {"x": 250, "y": 430, "w": 120, "h": 22},
         "kind": "shape", "role": "shape", "area": 380, "coverage": 0.0002,
         "score": 0.7, "stroke_only": True},
        # Endpoint dot near the product tip of the leader.
        {"id": "DOT", "box": {"x": 360, "y": 428, "w": 18, "h": 18},
         "kind": "icon", "role": "shape", "area": 280, "coverage": 0.0002,
         "score": 0.8},
    ]
    cfg = {"scene": {"preset": {"grouping": {"preserve_callout_leaders": True}},
                    "facts": {"leader_lines": True}}}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_L1"]["target"] != "drop"
    assert m["c_L1"]["meta"].get("callout_leader") or m["c_L1"]["meta"].get("role") in {
        "arrow", "callout_leader",
    }
    assert m["c_DOT"]["target"] != "drop"
    assert m["c_DOT"]["meta"].get("leader_dot") is True
    assert m["c_DOT"]["meta"].get("role") == "leader_dot"
    assert m["c_DOT"]["meta"].get("pairs_with") == "c_L1"
    assert m["c_L1"]["meta"].get("endpoint_dot_id") == "c_DOT"
    assert m["c_c1"]["target"] == "text"


# ── 2) Three-column stats Auto Layout (MONTE) ────────────────────────────────────

def test_three_column_stats_form_horizontal_stat_row():
    """0% / 100% / 87%-style columns → HORIZONTAL Stats row when aligned.

    Feeds three already-formed vertical text-stacks (as OCR paragraph blocks or the
    stack pass would emit per column) — the row gate requires that column evidence.
    """
    stacks = []
    for i, (x, pct, label) in enumerate((
        (80, "0%", "Synthetic"),
        (380, "100%", "Natural"),
        (680, "87%", "Better sleep"),
    )):
        kids = [
            _text(f"pct{i}", pct, {"x": x, "y": 400, "w": 120, "h": 48}, "label"),
            _text(f"lab{i}", label, {"x": x, "y": 460, "w": 160, "h": 36}, "body"),
            _text(f"body{i}", f"Detail line {i}", {"x": x, "y": 510, "w": 200, "h": 40}, "body"),
        ]
        kids[0]["style"] = {"align": "CENTER", "fontSize": 42}
        stacks.append({
            "id": f"col{i}",
            "target": "group",
            "box": {"x": x, "y": 400, "w": 200, "h": 150},
            "children": kids,
            "layout": {
                "mode": "VERTICAL", "gap": 12, "align": "MIN", "counterAlign": "MIN",
                "primarySizing": "HUG", "counterSizing": "HUG", "confidence": 0.9,
            },
            "meta": {"role": "text-stack"},
        })
    tree = layout.infer(stacks, {"w": 1080, "h": 1350}, {})
    row = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "stat-row"),
        None,
    )
    assert row is not None, "expected HORIZONTAL stat-row"
    assert row["layout"]["mode"] == "HORIZONTAL"
    assert len(row["children"]) == 3
    assert (row.get("meta") or {}).get("column_count") == 3


# ── 3) Left-column benefit pill stack ────────────────────────────────────────────

def test_left_column_benefit_pills_stack():
    """Outlined/filled callout pills in the left column stack as Benefits."""
    candidates = [
        {"id": "p1", "target": "shape", "box": {"x": 36, "y": 280, "w": 260, "h": 56},
         "fill": {"kind": "flat", "color": "#ffffff00"},
         "stroke": {"color": "#111111", "weight": 2}, "radius": 28,
         "meta": {"role": "callout", "text_bearing_shell": True, "stroke_outline_shell": True}},
        _text("t1", "Daily digestive support", {"x": 52, "y": 292, "w": 220, "h": 32}, "callout"),
        {"id": "p2", "target": "shape", "box": {"x": 36, "y": 360, "w": 260, "h": 56},
         "fill": {"kind": "flat", "color": "#ffffff00"},
         "stroke": {"color": "#111111", "weight": 2}, "radius": 28,
         "meta": {"role": "callout", "text_bearing_shell": True, "stroke_outline_shell": True}},
        _text("t2", "Immune health", {"x": 52, "y": 372, "w": 180, "h": 32}, "callout"),
    ]
    cfg = {"layout": {"scene_grouping": {"pair_text_with_backplate": True}}}
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    stack = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "benefit-stack"),
        None,
    )
    assert stack is not None
    assert stack["layout"]["mode"] == "VERTICAL"
    assert len(stack["children"]) == 2


# ── 4) Trustpilot rating strip ───────────────────────────────────────────────────

def test_trustpilot_stars_and_rating_text_form_rating_strip():
    stars = [
        {"id": f"s{i}", "target": "icon",
         "box": {"x": 80 + i * 36, "y": 900, "w": 28, "h": 28},
         "meta": {"role": "rating"}}
        for i in range(5)
    ]
    rating = _text(
        "exc", "Excellent", {"x": 270, "y": 902, "w": 140, "h": 28}, "label",
    )
    cfg = {"layout": {"scene_grouping": {"rating_strip_atomic_fallback": True}}}
    tree = layout.infer(stars + [rating], {"w": 1080, "h": 1350}, cfg)
    strip = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "rating-strip"),
        None,
    )
    assert strip is not None
    assert strip["layout"]["mode"] == "HORIZONTAL"
    ids = {c["id"] for c in strip["children"]}
    assert "exc" in ids
    assert len([c for c in strip["children"] if c.get("target") == "icon"]) == 5


# ── 5) AS SEEN IN logo strip (honest raster) ─────────────────────────────────────

def test_as_seen_in_logos_tagged_intentional_raster_strip():
    canvas = {"w": 1080, "h": 1350}
    ocr = {"lines": [
        {"id": "seen", "text": "AS SEEN IN", "conf": 0.95, "role": "eyebrow",
         "box": {"x": 360, "y": 1100, "w": 360, "h": 28}},
    ]}
    logos = [
        {"id": f"L{i}", "box": {"x": 200 + i * 140, "y": 1140, "w": 100, "h": 40},
         "kind": "icon", "role": "logo", "area": 3500, "coverage": 0.002, "score": 0.85}
        for i in range(4)
    ]
    m = _by_id(merge_layers.merge(ocr, logos, [], canvas, {}))
    assert m["c_seen"]["target"] == "text"
    tagged = [m[f"c_L{i}"] for i in range(4)]
    for node in tagged:
        assert node["meta"].get("role") == "logo-strip"
        assert node["meta"].get("intentional_raster_cluster") is True
        assert node["meta"].get("logo_strip_group_id")
    assert raster_clusters.is_intentional_raster_cluster("logo-strip")
    assert raster_clusters.is_intentional_raster_cluster("as-seen-in")


def test_as_seen_in_layout_wraps_logo_strip():
    label = _text("seen", "AS SEEN IN", {"x": 400, "y": 1000, "w": 280, "h": 24}, "eyebrow")
    logos = [
        {"id": f"lg{i}", "target": "image",
         "box": {"x": 180 + i * 160, "y": 1040, "w": 120, "h": 36},
         "meta": {"role": "logo", "intentional_raster_cluster": True}}
        for i in range(3)
    ]
    tree = layout.infer([label] + logos, {"w": 1080, "h": 1350}, {})
    strip = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "logo-strip"),
        None,
    )
    assert strip is not None
    assert strip["layout"]["mode"] == "HORIZONTAL"
    assert len(strip["children"]) >= 3


# ── 6) Sale circle + green CTA + story swipe-up ──────────────────────────────────

def test_sale_circle_shell_plus_offer_text():
    canvas = {"w": 1080, "h": 1920}
    elements = [{
        "id": "SALE", "box": {"x": 820, "y": 160, "w": 180, "h": 180},
        "kind": "icon", "role": "sale_burst", "area": int(180 * 180 * 0.55),
        "coverage": 0.03, "score": 0.9,
    }]
    ocr = {"lines": [{
        "id": "off", "text": "50% OFF", "conf": 0.96, "role": "offer",
        "box": {"x": 850, "y": 220, "w": 120, "h": 50},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    # Raster-first chrome policy: the seal ships as a pixel-exact raster and its
    # offer text rides the raster instead of half-rebuilding an SVG shell.
    assert m["c_SALE"]["target"] == "image"
    assert m["c_off"]["target"] == "drop"
    assert m["c_off"]["meta"].get("kept_in_photo") is True
    assert m["c_SALE"]["meta"].get("role") in {
        "sale_burst", "seal", "badge", "starburst", "price_burst",
    }


def test_green_get_yours_cta_shell_and_story_swipe_up():
    canvas = {"w": 1080, "h": 1920}  # story aspect
    elements = [{
        "id": "BTN", "box": {"x": 280, "y": 1680, "w": 520, "h": 88},
        "kind": "shape", "role": "button", "area": 45000, "coverage": 0.02, "score": 0.9,
    }]
    ocr = {"lines": [
        {"id": "gy", "text": "Get Yours", "conf": 0.97, "role": "cta",
         "box": {"x": 400, "y": 1700, "w": 280, "h": 48}},
        {"id": "su", "text": "Swipe Up", "conf": 0.94, "role": "body",
         "box": {"x": 440, "y": 1800, "w": 200, "h": 36}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_gy"]["target"] == "text"
    assert m["c_gy"]["meta"].get("role") == "cta"
    assert m["c_BTN"]["target"] in {"shape", "icon"}
    assert m["c_su"]["target"] == "text"
    assert m["c_su"]["meta"].get("story_cta") is True
    assert m["c_su"]["meta"].get("role") == "cta"


# ── 7) HiStrips-style bullet row + pink seal ─────────────────────────────────────

def test_histrips_bullet_icon_pairs_with_body_as_row():
    """Bullet chip + body on one baseline → HORIZONTAL text-row (lines spaced apart)."""
    candidates = [
        {"id": "b1", "target": "icon", "box": {"x": 80, "y": 520, "w": 22, "h": 22},
         "meta": {"role": "icon"}},
        _text("t1", "Peel and stick under nose", {"x": 116, "y": 518, "w": 420, "h": 28}, "body"),
        # Far below so the two body lines do not form a vertical text-stack first.
        {"id": "b2", "target": "icon", "box": {"x": 80, "y": 720, "w": 22, "h": 22},
         "meta": {"role": "icon"}},
        _text("t2", "Breathe easier overnight", {"x": 116, "y": 718, "w": 400, "h": 28}, "body"),
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, {})
    rows = [
        n for n in tree
        if (n.get("meta") or {}).get("role") in {"text-row", "checklist"}
    ]
    assert len(rows) >= 1
    assert any(
        {c.get("id") for c in (r.get("children") or [])} >= {"b1", "t1"}
        or {c.get("id") for c in (r.get("children") or [])} >= {"b2", "t2"}
        for r in rows
    )


def test_pink_valentines_seal_promotes_like_badge_shell():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "SEAL", "box": {"x": 780, "y": 120, "w": 200, "h": 200},
        "kind": "icon", "role": "seal", "area": int(200 * 200 * 0.4),
        "coverage": 0.03, "score": 0.9,
    }]
    ocr = {"lines": [{
        "id": "vday", "text": "Valentine's Deal", "conf": 0.95, "role": "offer",
        "box": {"x": 810, "y": 190, "w": 140, "h": 60},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    # Raster-first chrome policy: seal rasters whole, its text rides the raster.
    assert m["c_SEAL"]["target"] == "image"
    assert m["c_vday"]["target"] == "drop"
    assert m["c_vday"]["meta"].get("kept_in_photo") is True


# ── SAM prompts + cluster contracts ──────────────────────────────────────────────

def test_sam_prompts_include_trust_and_leader_dot_roles():
    roles = {spec["role"] for spec in sam3_detect._prompt_specs(None)}
    assert "leader_dot" in roles
    assert "rating" in roles
    assert "logo-strip" in roles
    assert "sale_burst" in roles
