"""IM8 comparison / problem-solution creatives — geometry-only lock (no new VLM).

Patterns:
1. PATCHED TOGETHER vs DAILY IM8 — pill cloud raster + glass/hand + thin divider
2. STRUGGLE vs ANSWER — pinned tags + string leaders → product; on-pack bake
3. BEFORE / RITUAL / RESET stage strip (+ progress bar); body morph = intentional raster
4. Two canisters side-by-side + headline/CTA/review footer (multi-product anti-merge)
"""
from __future__ import annotations

from src import (
    archetype,
    build_design_json,
    element_fusion,
    format_readiness,
    layout,
    merge_layers,
    raster_clusters,
    routing,
)


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _all_groups(nodes):
    for node in nodes:
        if node.get("target") == "group":
            yield node
        yield from _all_groups(node.get("children") or [])


# ── archetype / format ───────────────────────────────────────────────────────


def test_im8_struggle_answer_classifies_comparison_grid():
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1350},
        {"lines": [
            {"text": "STRUGGLE"}, {"text": "ANSWER"}, {"text": "vs"},
            {"text": "Daily support"},
        ]},
        {"column_count": 2, "leader_lines": True},
    )
    assert facts["before_after_labels"] is True
    assert facts["before_after_pair"] is True
    result = archetype.classify(facts)
    assert result["archetype"] == "comparison_grid"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"]["preserve_columns"] is True
    assert cfg["layout"]["scene_grouping"]["preserve_callout_leaders"] is True


def test_im8_before_ritual_reset_sets_stage_progression():
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1350},
        {"lines": [{"text": "BEFORE"}, {"text": "RITUAL"}, {"text": "RESET"}]},
        {"column_count": 1},
    )
    assert facts["stage_progression"] is True
    assert facts["before_after_labels"] is True
    result = archetype.classify(facts)
    assert result["archetype"] == "comparison_grid"


def test_im8_patched_daily_pair_and_problem_solution_tag():
    facts = archetype.scene_facts(
        {"w": 1080, "h": 1080},
        {"lines": [{"text": "PATCHED TOGETHER"}, {"text": "DAILY IM8"}, {"text": "VS"}]},
        {"column_count": 2, "center_divider": True},
    )
    assert facts["before_after_pair"] is True
    caps = format_readiness.infer_capabilities(
        facts,
        archetype="comparison_grid",
        preset=archetype.PRESETS["comparison_grid"],
    )
    assert caps["comparison_columns"] is True
    profile = format_readiness.build_format_profile(
        {"w": 1080, "h": 1080},
        facts,
        archetype="comparison_grid",
        preset=archetype.PRESETS["comparison_grid"],
        tags=["problem_solution"],
    )
    assert profile["capabilities"]["comparison_columns"] is True
    assert profile["capabilities"]["diagrams"] is True


# ── pattern 1: pill cloud vs glass+hand ──────────────────────────────────────


def test_im8_pill_cloud_is_intentional_raster_and_stays_separate_from_hand(tmp_path):
    assert raster_clusters.is_intentional_raster_cluster("pill-cloud")
    assert raster_clusters.cluster_label("chaotic-pill-cluster") == "Pill cloud"
    left = element_fusion._rect_mask({"x": 40, "y": 200, "w": 420, "h": 700}, 1080, 1350)
    right = element_fusion._rect_mask({"x": 580, "y": 220, "w": 400, "h": 680}, 1080, 1350)
    assert element_fusion._mask_metrics(left, right)["iou"] < 0.2
    sam3 = {
        "elements": [
            {
                "id": "S-pills", "box": {"x": 40, "y": 200, "w": 420, "h": 700},
                "role": "pill-cloud", "kind": "photo-fragment", "score": 0.92,
                "_mask": left,
                "provenance": {"mode": "text-prompt", "prompt": "chaotic pile of pills"},
            },
            {
                "id": "S-glass", "box": {"x": 580, "y": 220, "w": 400, "h": 680},
                "role": "product", "kind": "photo-fragment", "score": 0.94,
                "_mask": right,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], {"w": 1080, "h": 1350}, run_dir=str(tmp_path))
    assert len(fused) == 2
    roles = {e["role"] for e in fused}
    assert "pill-cloud" in roles or "product-cluster" in roles or "product" in roles
    # Right glass/hand must remain a discrete cutout.
    assert any(e["role"] == "product" for e in fused)

    routed = routing.route(
        {"id": "pills", "kind": "photo-fragment",
         "box": {"x": 40, "y": 200, "w": 420, "h": 700},
         "meta": {"role": "pill-cloud"}},
        {"w": 1080, "h": 1350},
    )
    assert routed["target"] == "image"
    assert routed["meta"].get("intentional_raster_cluster") is True


def test_im8_vs_merge_tags_pill_cloud_and_divider():
    canvas = {"w": 1080, "h": 1350}
    elements = [
        {"id": "PILLS", "box": {"x": 40, "y": 220, "w": 420, "h": 700},
         "kind": "photo-fragment", "role": "product-cluster", "area": 200000,
         "coverage": 0.18, "score": 0.93},
        {"id": "GLASS", "box": {"x": 600, "y": 240, "w": 380, "h": 680},
         "kind": "photo-fragment", "role": "product", "area": 190000,
         "coverage": 0.16, "score": 0.94},
        {"id": "DIV", "box": {"x": 536, "y": 200, "w": 6, "h": 760},
         "kind": "shape", "role": "divider", "area": 4000, "coverage": 0.003, "score": 0.8},
        {"id": "VS", "box": {"x": 500, "y": 520, "w": 80, "h": 80},
         "kind": "shape", "role": "badge", "area": 5000, "coverage": 0.004, "score": 0.8},
    ]
    ocr = {"lines": [
        {"id": "vs", "text": "VS", "conf": 0.97, "role": "label",
         "box": {"x": 515, "y": 540, "w": 50, "h": 40}},
        {"id": "bl", "text": "PATCHED TOGETHER", "conf": 0.95, "role": "label",
         "box": {"x": 80, "y": 160, "w": 320, "h": 36}},
        {"id": "al", "text": "DAILY IM8", "conf": 0.95, "role": "label",
         "box": {"x": 680, "y": 160, "w": 220, "h": 36}},
    ]}
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "facts": {
                "before_after_pair": True, "before_after_labels": True,
                "center_divider": True, "column_count": 2,
            },
        },
        "layout": {"scene_grouping": {"preserve_columns": True}},
    }
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_PILLS"]["target"] == "image"
    assert m["c_GLASS"]["target"] == "image"
    assert m["c_PILLS"]["meta"].get("comparison_side") == "before"
    assert m["c_GLASS"]["meta"].get("comparison_side") == "after"
    assert m["c_DIV"]["target"] != "drop"
    assert m["c_bl"]["meta"].get("semantic_name") == "Patched"
    assert m["c_al"]["meta"].get("semantic_name") == "Daily"
    assert m["c_vs"]["meta"].get("role") == "vs" or m["c_vs"]["meta"].get("semantic_name") == "VS"


# ── pattern 2: STRUGGLE tags + string leaders + on-pack bake ─────────────────


def test_im8_string_leaders_not_eaten_as_guide_artifact():
    canvas = {"w": 1080, "h": 1350}
    ocr = {"lines": [
        {"id": "t1", "text": "Bloating", "conf": 0.93, "role": "body",
         "box": {"x": 40, "y": 360, "w": 160, "h": 36}},
        {"id": "t2", "text": "Brain fog", "conf": 0.92, "role": "body",
         "box": {"x": 40, "y": 520, "w": 180, "h": 36}},
        {"id": "pack", "text": "IM8", "conf": 0.90, "role": "label",
         "box": {"x": 720, "y": 620, "w": 80, "h": 40}},
    ]}
    elements = [
        {"id": "PROD", "box": {"x": 640, "y": 480, "w": 280, "h": 420},
         "kind": "photo-fragment", "role": "sachet", "area": 90000,
         "coverage": 0.08, "score": 0.95},
        {"id": "PL1", "box": {"x": 36, "y": 350, "w": 200, "h": 56},
         "kind": "shape", "role": "badge", "area": 9000, "coverage": 0.006, "score": 0.85},
        {"id": "IC1", "box": {"x": 48, "y": 362, "w": 28, "h": 28},
         "kind": "icon", "role": "icon", "area": 600, "coverage": 0.0004, "score": 0.8},
        # Sparse diagonal string toward the sachet — must NOT guide-drop.
        {"id": "STR1", "box": {"x": 220, "y": 380, "w": 360, "h": 120},
         "kind": "shape", "role": "string", "area": 900, "coverage": 0.0003,
         "score": 0.7, "stroke_only": True},
        {"id": "STR2", "box": {"x": 230, "y": 500, "w": 340, "h": 100},
         "kind": "shape", "role": "shape", "area": 700, "coverage": 0.0002,
         "score": 0.65},
    ]
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "preset": {"grouping": {"preserve_callout_leaders": True}},
            "facts": {
                "before_after_pair": True, "before_after_labels": True,
                "leader_lines": True,
            },
        },
    }
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_STR1"]["target"] != "drop"
    assert m["c_STR1"]["meta"].get("suppression_reason") != "guide_artifact"
    assert m["c_STR1"]["meta"].get("callout_leader") or m["c_STR1"]["meta"].get("role") in {
        "string", "callout_leader", "arrow", "leader",
    }
    assert m["c_STR2"]["target"] != "drop"
    assert m["c_STR2"]["meta"].get("suppression_reason") != "guide_artifact"
    # On-pack brand glyph stays baked in the sachet cutout.
    assert m["c_pack"]["target"] == "drop" or m["c_pack"].get("kept_in_photo") or (
        m["c_pack"].get("meta") or {}
    ).get("kept_in_photo")
    assert m["c_t1"]["target"] == "text"
    assert m["c_PROD"]["target"] == "image"


def test_im8_pinned_tag_icon_text_pairs_on_plate():
    cfg = {
        "scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True}},
        "layout": {"scene_grouping": {"pair_text_with_backplate": True}},
    }
    candidates = [
        {"id": "plate", "target": "shape",
         "box": {"x": 40, "y": 400, "w": 220, "h": 52},
         "fill": {"kind": "flat", "color": "#FFFFFF"},
         "meta": {"role": "badge", "text_bearing_shell": True, "plate_shell": True}},
        {"id": "ic", "target": "icon", "box": {"x": 52, "y": 412, "w": 28, "h": 28},
         "meta": {"role": "icon", "icon_chip": True}},
        {"id": "lbl", "target": "text", "text": "Gut stress",
         "box": {"x": 90, "y": 412, "w": 150, "h": 28},
         "meta": {"role": "callout"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    host = next(
        (n for n in _all_groups(tree) if (n.get("meta") or {}).get("pair_text_with_backplate")),
        tree[0] if tree else None,
    )
    assert host is not None
    assert host["target"] == "group"
    kids = {c.get("id") for c in host.get("children") or []}
    assert "lbl" in kids


# ── pattern 3: BEFORE / RITUAL / RESET strip ─────────────────────────────────


def test_im8_stage_progression_strip_groups_labels():
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "facts": {"stage_progression": True, "before_after_labels": True},
        },
    }
    candidates = [
        {"id": "b", "target": "text", "text": "BEFORE",
         "box": {"x": 120, "y": 900, "w": 140, "h": 36},
         "meta": {"role": "label", "before_after_side": "before"}},
        {"id": "r", "target": "text", "text": "RITUAL",
         "box": {"x": 420, "y": 900, "w": 140, "h": 36},
         "meta": {"role": "label", "before_after_side": "mid", "stage_index": 1}},
        {"id": "e", "target": "text", "text": "RESET",
         "box": {"x": 720, "y": 900, "w": 120, "h": 36},
         "meta": {"role": "label", "before_after_side": "after"}},
        {"id": "bar", "target": "shape",
         "box": {"x": 140, "y": 950, "w": 700, "h": 8},
         "meta": {"role": "progress-bar"}},
        {"id": "morph", "target": "image",
         "box": {"x": 80, "y": 200, "w": 920, "h": 620},
         "meta": {"role": "body-progression", "intentional_raster_cluster": True}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    strip = next(
        (n for n in _all_groups(tree)
         if (n.get("meta") or {}).get("role") == "stage-progression"),
        None,
    )
    assert strip is not None
    assert strip["layout"]["mode"] == "HORIZONTAL"
    assert (strip.get("meta") or {}).get("stage_count") == 3
    kid_ids = {c.get("id") for c in strip.get("children") or []}
    assert {"b", "r", "e"} <= kid_ids
    # Body morph stays a top-level intentional raster (honest YELLOW), not invented cutouts.
    morph = next(
        (n for n in tree if n.get("id") == "morph"
         or (n.get("meta") or {}).get("role") == "body-progression"),
        None,
    )
    assert morph is not None
    assert morph["target"] == "image"
    assert raster_clusters.is_intentional_raster_cluster("body-progression")


def test_im8_merge_tags_ritual_reset_labels():
    canvas = {"w": 1080, "h": 1350}
    ocr = {"lines": [
        {"id": "b", "text": "BEFORE", "conf": 0.96, "role": "label",
         "box": {"x": 120, "y": 900, "w": 140, "h": 36}},
        {"id": "r", "text": "RITUAL", "conf": 0.96, "role": "label",
         "box": {"x": 420, "y": 900, "w": 140, "h": 36}},
        {"id": "e", "text": "RESET", "conf": 0.96, "role": "label",
         "box": {"x": 720, "y": 900, "w": 120, "h": 36}},
    ]}
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "facts": {"stage_progression": True, "before_after_labels": True},
        },
    }
    m = _by_id(merge_layers.merge(ocr, [], [], canvas, cfg))
    assert m["c_b"]["meta"].get("before_after_side") == "before"
    assert m["c_r"]["meta"].get("before_after_side") == "mid"
    assert m["c_e"]["meta"].get("semantic_name") == "Reset"
    assert build_design_json._name(m["c_r"]) == "Ritual"
    assert build_design_json._name(m["c_e"]) == "Reset"


# ── pattern 4: two canisters + CTA + review footer ───────────────────────────


def test_im8_two_canisters_do_not_merge(tmp_path):
    a = element_fusion._rect_mask({"x": 180, "y": 320, "w": 260, "h": 480}, 1080, 1350)
    b = element_fusion._rect_mask({"x": 640, "y": 320, "w": 260, "h": 480}, 1080, 1350)
    assert element_fusion._mask_metrics(a, b)["iou"] < 0.15
    sam3 = {
        "elements": [
            {
                "id": "S-a", "box": {"x": 180, "y": 320, "w": 260, "h": 480},
                "role": "canister", "kind": "photo-fragment", "score": 0.94,
                "_mask": a,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
            {
                "id": "S-b", "box": {"x": 640, "y": 320, "w": 260, "h": 480},
                "role": "canister", "kind": "photo-fragment", "score": 0.93,
                "_mask": b,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], {"w": 1080, "h": 1350}, run_dir=str(tmp_path))
    assert len(fused) == 2


def test_im8_dual_canister_scene_classifies_product_on_flat():
    result = archetype.classify({
        "product_count": 2,
        "flat_background_fraction": 0.62,
        "photo_coverage": 0.28,
        "mean_luma": 210,
    })
    assert result["archetype"] == "product_on_flat"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"].get("rating_strip_atomic_fallback") is True or (
        cfg["layout"]["scene_grouping"].get("pair_text_with_backplate") is True
    )


def test_im8_cta_and_review_footer_survive():
    cfg = {
        "scene": {"archetype": "product_on_flat", "facts": {"product_count": 2}},
        "layout": {"scene_grouping": {
            "pair_text_with_backplate": True,
            "rating_strip_atomic_fallback": True,
        }},
    }
    candidates = [
        {"id": "c1", "target": "image",
         "box": {"x": 180, "y": 300, "w": 260, "h": 480},
         "meta": {"role": "canister"}},
        {"id": "c2", "target": "image",
         "box": {"x": 640, "y": 300, "w": 260, "h": 480},
         "meta": {"role": "product"}},
        {"id": "hl", "target": "text", "text": "Feel the difference",
         "box": {"x": 200, "y": 80, "w": 680, "h": 56},
         "meta": {"role": "headline"}},
        {"id": "btn", "target": "shape",
         "box": {"x": 300, "y": 1000, "w": 480, "h": 72},
         "radius": 36,
         "fill": {"kind": "flat", "color": "#FFFFFF"},
         "meta": {"role": "button", "text_bearing_shell": True, "button_shell": True}},
        {"id": "cta", "target": "text", "text": "GET IM8 HEALTH",
         "box": {"x": 360, "y": 1018, "w": 360, "h": 36},
         "meta": {"role": "cta"}},
        {"id": "s0", "target": "icon", "box": {"x": 200, "y": 1140, "w": 28, "h": 28},
         "meta": {"role": "rating"}},
        {"id": "s1", "target": "icon", "box": {"x": 236, "y": 1140, "w": 28, "h": 28},
         "meta": {"role": "rating"}},
        {"id": "rev", "target": "text", "text": "4.8/5 REVIEWS | 24M+ SERVINGS",
         "box": {"x": 280, "y": 1142, "w": 520, "h": 28},
         "meta": {"role": "footer"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    # CTA shell + text nest; review strip forms.
    assert any(
        (n.get("meta") or {}).get("role") in {"button", "cta"}
        or (n.get("meta") or {}).get("button_shell")
        for n in _all_groups(tree)
    ) or any(n.get("id") == "cta" for n in tree)
    strip = next(
        (n for n in _all_groups(tree)
         if (n.get("meta") or {}).get("role") in {"rating-strip", "review-bar"}),
        None,
    )
    # Rating strip OR review bar is acceptable; stars+copy must not be dropped.
    flat_ids = set()
    for n in tree:
        flat_ids.add(n.get("id"))
        for c in n.get("children") or []:
            flat_ids.add(c.get("id"))
    assert "rev" in flat_ids or strip is not None
    assert "hl" in flat_ids
