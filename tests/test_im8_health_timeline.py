"""Unit locks for IM8-style health ads: timeline + product + seals + review bar.

Geometry-only — no VLM. Reuses badge/banner/stat/rating paths; adds Timeline grouping.
"""
from __future__ import annotations

from src import layout, merge_layers, routing
from src.format_readiness import build_format_profile, prefers_icon_chips


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _text(node_id, text, box, role="body"):
    return {
        "id": node_id, "target": "text", "text": text, "box": box,
        "style": {"align": "LEFT", "fontSize": 22},
        "meta": {"role": role},
    }


# ── 1) Vertical Day 1/10/30/90 timeline ──────────────────────────────────────────

def test_im8_day_timeline_groups_chips_connector_and_editable_text():
    """Circular step icons + spine + Day/body TEXT → VERTICAL Timeline; icons stay chips."""
    candidates = []
    days = [(1, 200, "More energy in the morning"),
            (10, 360, "Digestion feels smoother"),
            (30, 520, "Skin looks clearer"),
            (90, 680, "This is my daily ritual")]
    for day, y, body in days:
        candidates.append({
            "id": f"ico{day}", "target": "icon",
            "box": {"x": 80, "y": y, "w": 56, "h": 56},
            "meta": {"role": "icon"},
        })
        candidates.append(_text(
            f"d{day}", f"Day {day}", {"x": 160, "y": y + 4, "w": 100, "h": 28}, "label",
        ))
        candidates.append(_text(
            f"b{day}", body, {"x": 160, "y": y + 34, "w": 420, "h": 32}, "body",
        ))
    candidates.append({
        "id": "spine", "target": "shape",
        "box": {"x": 104, "y": 250, "w": 8, "h": 470},
        "meta": {"role": "connector"},
    })
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, {})
    timeline = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "timeline"),
        None,
    )
    assert timeline is not None, "expected VERTICAL timeline group"
    assert timeline["layout"]["mode"] == "VERTICAL"
    assert (timeline.get("meta") or {}).get("step_count") == 4
    assert (timeline.get("meta") or {}).get("has_connector") is True
    steps = [
        c for c in (timeline.get("children") or [])
        if (c.get("meta") or {}).get("role") == "timeline-step"
    ]
    assert len(steps) == 4

    def _walk(nodes):
        for node in nodes:
            yield node
            yield from _walk(node.get("children") or [])

    # Icons remain chips; day/body stay native TEXT (may sit in a nested text-stack).
    for step in steps:
        kids = step.get("children") or []
        assert any(k.get("target") == "icon" and (k.get("meta") or {}).get("icon_chip") for k in kids)
        flat = list(_walk(kids))
        assert any(k.get("target") == "text" for k in flat)
        assert any(
            k.get("target") == "text" and str(k.get("text") or "").lower().startswith("day")
            for k in flat
        )
    connectors = [
        c for c in (timeline.get("children") or [])
        if (c.get("meta") or {}).get("timeline_connector")
    ]
    assert len(connectors) == 1
    assert (connectors[0].get("layout") or {}).get("layoutPositioning") == "ABSOLUTE"


def test_im8_timeline_tag_enables_icon_chips_without_new_preset():
    profile = build_format_profile(
        {"w": 1080, "h": 1350},
        {"aspect_ratio": 1080 / 1350, "product_count": 1, "text_backplate_count": 1},
        archetype="product_on_flat",
        tags=["timeline", "health_product"],
    )
    assert profile["capabilities"]["icons_as_chips"] is True
    assert profile["capabilities"]["diagrams"] is True
    assert profile["capabilities"]["cutouts"] is True
    cfg = {"scene": {"format": profile}, "routing": {}}
    assert prefers_icon_chips(cfg) is True
    assert routing._icons_as_chips(cfg) is True


# ── 2) Scalloped / circle sale seals (028 path) ──────────────────────────────────

def test_im8_get_up_to_off_seal_is_text_bearing_shell():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "SEAL", "box": {"x": 780, "y": 140, "w": 200, "h": 200},
        "kind": "icon", "role": "sale_burst", "area": int(200 * 200 * 0.45),
        "coverage": 0.03, "score": 0.9,
    }]
    ocr = {"lines": [{
        "id": "off", "text": "GET UP TO 30% OFF", "conf": 0.96, "role": "offer",
        "box": {"x": 810, "y": 200, "w": 140, "h": 70},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    # Raster-first chrome policy: compact seal rasters whole, offer text rides it.
    assert m["c_SEAL"]["target"] == "image"
    assert m["c_off"]["target"] == "drop"
    assert m["c_off"]["meta"].get("kept_in_photo") is True
    assert m["c_SEAL"]["meta"].get("role") in {
        "sale_burst", "seal", "badge", "starburst", "price_burst",
    }


def test_im8_subscribe_save_seal_promotes_like_badge():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "SAVE", "box": {"x": 40, "y": 160, "w": 180, "h": 180},
        "kind": "icon", "role": "shape", "area": int(180 * 180 * 0.30),
        "coverage": 0.025, "score": 0.88, "source": "sam3",
    }]
    ocr = {"lines": [{
        "id": "sub", "text": "Subscribe & Save 30%", "conf": 0.95, "role": "offer",
        "box": {"x": 60, "y": 210, "w": 140, "h": 70},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    # Raster-first chrome policy: seal rasters whole, its label rides the raster.
    assert m["c_SAVE"]["target"] == "image"
    assert m["c_sub"]["target"] == "drop"
    assert m["c_sub"]["meta"].get("kept_in_photo") is True
    assert m["c_SAVE"]["meta"].get("role") in {"seal", "badge", "starburst", "price_burst", "sale_burst"}


# ── 3) Product cutout; on-pack text baked ────────────────────────────────────────

def test_im8_sachet_on_pack_text_kept_in_photo():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "SACHET", "box": {"x": 620, "y": 420, "w": 320, "h": 520},
        "kind": "photo-fragment", "role": "sachet", "area": 160000,
        "coverage": 0.12, "score": 0.94,
    }]
    ocr = {"lines": [
        {"id": "brand", "text": "IM8", "conf": 0.97, "role": "label",
         "box": {"x": 700, "y": 560, "w": 120, "h": 40}},
        {"id": "flavor", "text": "MIXED BERRY", "conf": 0.93, "role": "body",
         "box": {"x": 680, "y": 620, "w": 180, "h": 32}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_SACHET"]["target"] == "image"
    for cid in ("c_brand", "c_flavor"):
        assert m[cid].get("kept_in_photo") is True, cid
        assert m[cid]["target"] == "drop"
        assert m[cid]["meta"].get("baked_owner_id") == "c_SACHET"


def test_im8_hex_jar_on_pack_text_kept_in_photo():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "JAR", "box": {"x": 400, "y": 380, "w": 280, "h": 420},
        "kind": "photo-fragment", "role": "jar", "area": 110000,
        "coverage": 0.09, "score": 0.93,
    }]
    ocr = {"lines": [{
        "id": "nutri", "text": "30 SERVINGS", "conf": 0.92, "role": "label",
        "box": {"x": 460, "y": 700, "w": 160, "h": 28},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_nutri"].get("kept_in_photo") is True
    assert m["c_JAR"]["meta"].get("text_bearing_shell") is not True


# ── 4) Cream/maroon wave plate stays honest background / plate ───────────────────

def test_im8_wave_split_plate_not_forced_product_cutout():
    """Full-bleed two-tone plate is background territory — not a product raster owner."""
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "WAVE", "box": {"x": 0, "y": 0, "w": 1080, "h": 1350},
        "kind": "photo-fragment", "role": "background", "area": 1080 * 1350,
        "coverage": 1.0, "score": 0.5,
    }]
    ocr = {"lines": [{
        "id": "headline", "text": "Feel the difference in 90 days", "conf": 0.95,
        "role": "headline", "box": {"x": 80, "y": 120, "w": 700, "h": 64},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    # Overlay headline on a full-bleed plate stays editable (not baked into bg).
    assert m["c_headline"]["target"] == "text"
    assert not m["c_headline"].get("kept_in_photo")


# ── 5) White CTA pill ───────────────────────────────────────────────────────────

def test_im8_get_im8_health_cta_button_shell_plus_text():
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "CTA", "box": {"x": 240, "y": 1100, "w": 600, "h": 88},
        "kind": "shape", "role": "button", "area": 52000, "coverage": 0.03, "score": 0.92,
    }]
    ocr = {"lines": [{
        "id": "gy", "text": "GET IM8 HEALTH", "conf": 0.98, "role": "cta",
        "box": {"x": 360, "y": 1124, "w": 360, "h": 40},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_gy"]["target"] == "text"
    assert m["c_gy"]["meta"].get("role") == "cta"
    assert not m["c_gy"].get("kept_in_photo")
    assert m["c_CTA"]["target"] in {"shape", "icon"}
    assert m["c_CTA"]["meta"].get("text_bearing_shell") is True or m["c_CTA"]["meta"].get("role") in {
        "button", "cta", "badge", "chip",
    }
    # Layout pairs shell + label as a button group when surface is present.
    tree = layout.infer([
        {**m["c_CTA"], "fill": {"kind": "flat", "color": "#ffffff"}, "radius": 44},
        m["c_gy"],
    ], canvas, {"layout": {"scene_grouping": {"pair_text_with_backplate": True}}})
    btn = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "button"),
        None,
    )
    assert btn is not None
    assert any(c.get("target") == "text" for c in (btn.get("children") or []))


# ── 6) Review footer bar ────────────────────────────────────────────────────────

def test_im8_review_footer_bar_plate_plus_stars_and_text():
    stars = [
        {"id": f"s{i}", "target": "icon",
         "box": {"x": 120 + i * 34, "y": 1248, "w": 28, "h": 28},
         "meta": {"role": "rating"}}
        for i in range(5)
    ]
    rating = _text(
        "rev", "4.8/5 REVIEWS | 24M+ SERVINGS",
        {"x": 300, "y": 1248, "w": 520, "h": 28}, "label",
    )
    bar = {
        "id": "bar", "target": "shape",
        "box": {"x": 0, "y": 1220, "w": 1080, "h": 90},
        "fill": {"kind": "flat", "color": "#5a1a2a"},
        "meta": {"role": "footer", "plate_shell": True},
    }
    cfg = {"layout": {"scene_grouping": {"rating_strip_atomic_fallback": True}}}
    tree = layout.infer(stars + [rating, bar], {"w": 1080, "h": 1350}, cfg)
    review = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "review-bar"),
        None,
    )
    strip = next(
        (n for n in tree if (n.get("meta") or {}).get("role") == "rating-strip"),
        None,
    )
    assert review is not None or strip is not None, "expected review-bar or rating-strip"

    def _collect_text(node):
        out = []
        if node.get("target") == "text":
            out.append(node)
        for child in node.get("children") or []:
            out.extend(_collect_text(child))
        return out

    host = review or strip
    texts = _collect_text(host)
    assert any("REVIEW" in str(t.get("text") or "").upper() for t in texts)
    if review is not None:
        # Bar may be retagged in place (id=bar) or wrap a separate plate child.
        assert (
            review.get("id") == "bar"
            or (review.get("meta") or {}).get("plate_shell")
            or any(
                (c.get("meta") or {}).get("plate_shell") or c.get("id") == "bar"
                for c in (review.get("children") or [])
            )
        )
        star_count = sum(
            1 for c in (review.get("children") or [])
            if str((c.get("meta") or {}).get("role") or "").lower() in {
                "rating", "star", "stars",
            }
        )
        assert star_count >= 3 or any(
            (c.get("meta") or {}).get("role") == "rating-strip"
            for c in (review.get("children") or [])
        )
