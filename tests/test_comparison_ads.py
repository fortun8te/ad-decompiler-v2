"""BEFORE/AFTER + VS comparison ads — geometry-only lock (no new presets / VLM).

Patterns: Huel (photos+VS+social), MONTE (WITHOUT/WITH), Wavy (checklist pills),
HiStrips (grayscale/color panels). Reuses comparison_grid / before_after /
pair_text_with_backplate.
"""
from __future__ import annotations

from src import archetype, build_design_json, element_fusion, format_readiness, layout, merge_layers


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _all_groups(nodes):
    for node in nodes:
        if node.get("target") == "group":
            yield node
        yield from _all_groups(node.get("children") or [])


# ── archetype / format ───────────────────────────────────────────────────────


def test_huel_social_chrome_enables_header_cluster_on_comparison():
    result = archetype.classify({
        "before_after_pair": True,
        "before_after_labels": True,
        "column_count": 2,
        "social_header": True,
        "avatar_present": True,
    })
    assert result["archetype"] == "comparison_grid"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"]["header_cluster"] is True
    assert cfg["layout"]["scene_grouping"]["pair_text_with_backplate"] is True


def test_wavy_circular_inset_flag_reused_on_comparison():
    result = archetype.classify({
        "before_after_pair": True,
        "before_after_labels": True,
        "circular_inset": True,
        "column_count": 2,
    })
    assert result["archetype"] == "comparison_grid"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"].get("circular_insets_use_ellipse_mask") is True

    facts = archetype.scene_facts(
        {"w": 1080, "h": 1350},
        {"lines": [{"text": "WITHOUT"}, {"text": "WITH"}, {"text": "vs"}]},
        {"column_count": 2, "photo_coverage": 0.42},
    )
    assert facts["before_after_labels"] is True
    assert facts["before_after_pair"] is True
    result = archetype.classify(facts)
    assert result["archetype"] == "comparison_grid"
    cfg = archetype.apply_preset({}, result)
    assert cfg["layout"]["scene_grouping"]["pair_text_with_backplate"] is True
    assert cfg["layout"]["scene_grouping"]["preserve_columns"] is True


def test_without_with_alone_sets_pair_without_body_with_false_positive():
    pair = archetype.scene_facts(
        {"w": 800, "h": 800},
        {"lines": [{"text": "WITHOUT"}, {"text": "WITH"}]},
        {"column_count": 2},
    )
    assert pair["before_after_pair"] is True
    body = archetype.scene_facts(
        {"w": 800, "h": 800},
        {"lines": [{"text": "Start with better sleep tonight"}]},
        {},
    )
    assert body["before_after_labels"] is False
    assert body["before_after_pair"] is False


def test_comparison_columns_capability_and_icon_chips():
    caps = format_readiness.infer_capabilities(
        {"before_after_pair": True, "column_count": 2},
        archetype="comparison_grid",
        preset=archetype.PRESETS["comparison_grid"],
    )
    assert caps["comparison_columns"] is True
    assert caps["icons_as_chips"] is True
    assert format_readiness.prefers_icon_chips(
        {"scene": {"archetype": "comparison_grid"}}
    ) is True


# ── element fusion: two photo panels stay separate ───────────────────────────


def test_histrips_two_runner_panels_do_not_merge(tmp_path):
    """HiStrips: grayscale vs color panels must remain two cutouts."""
    left = element_fusion._rect_mask({"x": 40, "y": 120, "w": 420, "h": 760}, 1080, 1080)
    right = element_fusion._rect_mask({"x": 620, "y": 120, "w": 420, "h": 760}, 1080, 1080)
    assert element_fusion._mask_metrics(left, right)["iou"] < 0.2
    sam3 = {
        "elements": [
            {
                "id": "S-left", "box": {"x": 40, "y": 120, "w": 420, "h": 760},
                "role": "comparison-panel", "kind": "photo-fragment", "score": 0.93,
                "_mask": left,
                "provenance": {"mode": "text-prompt", "prompt": "comparison panel"},
            },
            {
                "id": "S-right", "box": {"x": 620, "y": 120, "w": 420, "h": 760},
                "role": "person", "kind": "photo-fragment", "score": 0.91,
                "_mask": right,
                "provenance": {"mode": "text-prompt", "prompt": "person"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], {"w": 1080, "h": 1080}, run_dir=str(tmp_path))
    assert len(fused) == 2


# ── merge: VS chip + two photos tagged ───────────────────────────────────────


def test_huel_merge_tags_two_photos_and_vs_chip():
    canvas = {"w": 1080, "h": 1350}
    elements = [
        {"id": "P_L", "box": {"x": 60, "y": 320, "w": 420, "h": 520},
         "kind": "photo-fragment", "role": "photo", "area": 180000,
         "coverage": 0.15, "score": 0.94},
        {"id": "P_R", "box": {"x": 600, "y": 320, "w": 420, "h": 520},
         "kind": "photo-fragment", "role": "photo", "area": 180000,
         "coverage": 0.15, "score": 0.93},
        {"id": "VS", "box": {"x": 500, "y": 520, "w": 80, "h": 80},
         "kind": "shape", "role": "badge", "area": 5000, "coverage": 0.004, "score": 0.8},
    ]
    ocr = {"lines": [
        {"id": "vs", "text": "VS", "conf": 0.97, "role": "label",
         "box": {"x": 515, "y": 540, "w": 50, "h": 40}},
        {"id": "bl", "text": "Before", "conf": 0.96, "role": "label",
         "box": {"x": 180, "y": 280, "w": 120, "h": 32}},
        {"id": "al", "text": "After", "conf": 0.96, "role": "label",
         "box": {"x": 740, "y": 280, "w": 100, "h": 32}},
    ]}
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "facts": {"before_after_pair": True, "before_after_labels": True},
        },
        "layout": {"scene_grouping": {"pair_text_with_backplate": True, "preserve_columns": True}},
    }
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_P_L"]["target"] == "image"
    assert m["c_P_R"]["target"] == "image"
    assert m["c_P_L"]["meta"].get("comparison_side") == "before"
    assert m["c_P_R"]["meta"].get("comparison_side") == "after"
    assert m["c_vs"]["target"] == "text"
    assert m["c_vs"]["meta"].get("semantic_name") == "VS" or m["c_vs"]["meta"].get("role") == "vs"
    assert m["c_bl"]["meta"].get("semantic_name") == "Before"
    assert m["c_al"]["meta"].get("semantic_name") == "After"
    # Prefer native TEXT for labels; image fallback still keeps the Before name.
    assert m["c_al"]["target"] == "text"
    assert m["c_VS"]["meta"].get("text_bearing_shell") is True


# ── layout: comparison-set nest + checklist rows ─────────────────────────────


def test_huel_layout_nests_photos_with_vs_between():
    cfg = {
        "scene": {
            "archetype": "comparison_grid",
            "facts": {"before_after_pair": True, "before_after_labels": True},
        },
        "layout": {"scene_grouping": {
            "preserve_columns": True, "pair_text_with_backplate": True,
        }},
    }
    candidates = [
        {"id": "L", "target": "image", "box": {"x": 60, "y": 300, "w": 400, "h": 480},
         "meta": {"role": "photo", "comparison_side": "before"}},
        {"id": "vs", "target": "text", "text": "VS",
         "box": {"x": 500, "y": 500, "w": 60, "h": 40},
         "meta": {"role": "vs"}},
        {"id": "R", "target": "image", "box": {"x": 620, "y": 300, "w": 400, "h": 480},
         "meta": {"role": "photo", "comparison_side": "after"}},
        {"id": "h", "target": "text", "text": "Real results",
         "box": {"x": 200, "y": 80, "w": 600, "h": 48}, "meta": {"role": "headline"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, cfg)
    cmp_set = next(
        n for n in _all_groups(tree)
        if (n.get("meta") or {}).get("role") == "comparison-set"
    )
    assert cmp_set["layout"]["mode"] == "HORIZONTAL"
    assert cmp_set["meta"]["semantic_name"] == "Comparison"
    kids = {c["id"]: c for c in cmp_set["children"]}
    assert set(kids) == {"L", "vs", "R"}
    assert kids["L"]["meta"]["semantic_name"] == "Photo / Before"
    assert kids["R"]["meta"]["semantic_name"] == "Photo / After"
    assert kids["vs"]["meta"]["semantic_name"] == "VS"


def test_wavy_checklist_icon_text_row_named_checklist():
    """X/check icon chip + TEXT → Checklist (pair_text_with_backplate already on preset)."""
    cfg = {
        "scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True}},
        "layout": {"scene_grouping": {"pair_text_with_backplate": True}},
    }
    candidates = [
        {"id": "x1", "target": "icon", "box": {"x": 80, "y": 900, "w": 28, "h": 28},
         "meta": {"role": "close", "icon_chip": True}},
        {"id": "t1", "target": "text", "text": "Dry flaky skin",
         "box": {"x": 120, "y": 898, "w": 220, "h": 32}, "meta": {"role": "body"}},
        {"id": "ck", "target": "icon", "box": {"x": 560, "y": 900, "w": 28, "h": 28},
         "meta": {"role": "checkmark", "icon_chip": True}},
        {"id": "t2", "target": "text", "text": "Smooth glow",
         "box": {"x": 600, "y": 898, "w": 200, "h": 32}, "meta": {"role": "body"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, cfg)
    checklists = [
        n for n in _all_groups(tree)
        if (n.get("meta") or {}).get("semantic_name") == "Checklist"
        or (n.get("meta") or {}).get("checklist")
    ]
    assert len(checklists) >= 2
    assert all(n["layout"]["mode"] == "HORIZONTAL" for n in checklists)
    assert all((n.get("meta") or {}).get("role") == "text-row" for n in checklists)


def test_pair_text_with_backplate_nests_before_after_tag_on_plate():
    """Before/After green tags: TEXT on small plate via pair_text_with_backplate."""
    cfg = {
        "scene": {"archetype": "comparison_grid", "facts": {"before_after_labels": True}},
        "layout": {"scene_grouping": {"pair_text_with_backplate": True}},
    }
    candidates = [
        {"id": "plate", "target": "shape",
         "box": {"x": 100, "y": 200, "w": 140, "h": 44},
         "fill": {"kind": "flat", "color": "#22AA55"},
         "meta": {"role": "badge", "text_bearing_shell": True, "plate_shell": True}},
        {"id": "lbl", "target": "text", "text": "Before",
         "box": {"x": 118, "y": 208, "w": 100, "h": 28},
         "meta": {"role": "label", "before_after_side": "before"}},
    ]
    tree = layout.infer(candidates, {"w": 800, "h": 800}, cfg)
    assert len(tree) == 1
    host = tree[0]
    assert host["target"] == "group"
    assert host["meta"].get("pair_text_with_backplate") is True
    assert any(c.get("id") == "lbl" for c in host.get("children") or [])


# ── naming ───────────────────────────────────────────────────────────────────


def test_comparison_local_names():
    assert build_design_json._name({
        "id": "p", "target": "image",
        "meta": {"role": "photo", "comparison_side": "before"},
    }) == "Photo / Before"
    assert build_design_json._name({
        "id": "a", "target": "image",
        "meta": {"role": "photo", "comparison_side": "after"},
    }) == "Photo / After"
    assert build_design_json._name({
        "id": "v", "target": "text", "text": "VS", "meta": {"role": "label"},
    }) == "VS"
    assert build_design_json._name({
        "id": "vb", "target": "shape",
        "meta": {
            "role": "badge", "text_bearing_shell": True,
            "shell_text_snippet": "VS",
        },
    }) == "VS"
    assert build_design_json._name({
        "id": "b", "target": "text", "text": "Before", "meta": {"role": "label"},
    }) == "Before"
    assert build_design_json._name({
        "id": "w", "target": "text", "text": "WITHOUT",
        "meta": {"role": "label", "before_after_side": "before"},
    }) == "Before"
    assert build_design_json._name({
        "id": "cl", "target": "group", "meta": {"role": "checklist"}, "children": [],
    }) == "Checklist"
