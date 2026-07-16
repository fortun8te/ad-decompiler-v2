"""test_merge_layers.py — fixture-driven tests for the OCR+element+Qwen fusion.

CPU-only, no heavy deps (merge_layers is pure). Uses the inline fallback router when
src.routing is absent (another builder owns it), which is the case at test time.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import merge_layers  # noqa: E402


CANVAS = {"w": 600, "h": 600}


def _by_id(cands):
    return {c["id"]: c for c in cands}


def _ocr():
    return {
        "lines": [
            {"id": "L0", "text": "BIG SALE", "conf": 0.98,
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}, "role": "headline"},
            {"id": "L1", "text": "engraved on the watch", "conf": 0.8,
             "box": {"x": 320, "y": 400, "w": 150, "h": 22}},
        ]
    }


def _elements():
    return [
        # a button/card behind the headline
        {"id": "E0", "box": {"x": 20, "y": 20, "w": 260, "h": 90}, "kind": "shape",
         "area": 20000, "coverage": 0.1, "source": "residual-cc"},
        # the product photo region (scene text lives inside it)
        {"id": "E1", "box": {"x": 300, "y": 300, "w": 240, "h": 260},
         "kind": "photo-fragment", "area": 40000, "coverage": 0.2,
         "source": "residual-cc", "role": "product"},
        # a small icon
        {"id": "E2", "box": {"x": 500, "y": 40, "w": 40, "h": 40}, "kind": "icon",
         "area": 1200, "coverage": 0.006, "source": "residual-cc"},
    ]


def _qwen():
    return [
        {"id": "Q0", "box": {"x": 300, "y": 300, "w": 240, "h": 260},
         "png": "qwen_layers/Q0.png", "kind_hint": "photo"},
    ]


def test_targets_assigned():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    assert m["c_L0"]["target"] == "text"
    assert m["c_E0"]["target"] == "shape"
    assert m["c_E2"]["target"] == "icon"
    # photo element gets Qwen alpha + image target + z from the qwen layer
    assert m["c_E1"]["target"] == "image"
    assert m["c_E1"]["src"] == "qwen_layers/Q0.png"
    assert m["c_E1"]["meta"]["qwen_id"] == "Q0"


def test_structural_detector_ids_survive_merge_for_layout_planning():
    elements = [{
        "id": "P0", "box": {"x": 20, "y": 20, "w": 120, "h": 180},
        "kind": "photo-fragment", "role": "panel", "area": 21_600,
        "score": .91, "grid_group_id": "comparison", "row_index": 0,
        "column_index": 0,
    }]

    candidate = _by_id(merge_layers.merge({"lines": []}, elements, [], CANVAS, {}))["c_P0"]

    assert candidate["meta"]["grid_group_id"] == "comparison"
    assert candidate["meta"]["row_index"] == 0
    assert candidate["meta"]["column_index"] == 0


def test_scene_text_dropped():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    # L1 sits inside the photo region -> kept in photo, routed to drop
    assert m["c_L1"]["meta"].get("kept_in_photo") is True
    assert m["c_L1"]["target"] == "drop"
    assert m["c_L1"]["meta"]["role"] == "scene-text"


def test_vlm_overlay_copy_inside_a_product_is_rebuilt_while_printed_copy_stays_baked():
    elements = [{
        "id": "product", "box": {"x": 100, "y": 100, "w": 260, "h": 260},
        "kind": "photo-fragment", "area": 67_600, "coverage": .2, "role": "product",
    }]
    ocr = {"lines": [
        {"id": "overlay", "text": "Real overlay", "conf": .98, "role": "body",
         "box": {"x": 130, "y": 130, "w": 110, "h": 18},
         "meta": {"scene_text_role": "overlay_copy"}},
        {"id": "printed", "text": "PACKAGING", "conf": .98, "role": "body",
         "box": {"x": 150, "y": 190, "w": 100, "h": 18},
         "meta": {"scene_text_role": "printed_on_product"}},
    ]}

    merged = _by_id(merge_layers.merge(ocr, elements, [], CANVAS, {}))

    assert merged["c_overlay"]["target"] == "text"
    assert merged["c_overlay"]["meta"]["overlay_text"] is True
    assert merged["c_overlay"]["meta"]["removal_required"] is True
    assert merged["c_printed"]["target"] == "drop"
    assert merged["c_printed"]["meta"]["kept_in_photo"] is True


def test_headline_text_survives_and_is_editable():
    cands = merge_layers.merge(_ocr(), _elements(), _qwen(), CANVAS, {})
    m = _by_id(cands)
    assert "c_L0" in m
    assert m["c_L0"]["text"] == "BIG SALE"
    # text sits above shapes
    assert m["c_L0"]["z"] >= m["c_E0"]["z"]
    assert m["c_L0"]["meta"]["overlay_text"] is True
    assert m["c_L0"]["meta"]["removal_required"] is True


def test_platform_lockup_is_a_separate_cropped_logo_not_baked_into_card():
    ocr = {"lines": [{
        "id": "X", "text": "X.com", "conf": .99,
        "box": {"x": 450, "y": 42, "w": 90, "h": 24},
        "meta": {"ownership_decision": {"placement": "ui_metadata", "owner": "card",
                                         "action": "raster_keep", "confidence": .99}},
    }]}
    merged = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))
    x = merged["c_X"]
    assert x["target"] == "image"
    assert x["meta"]["platform_lockup"] is True
    assert x["meta"]["role"] == "platform-logo"
    assert not x["meta"].get("kept_in_photo")


def test_unlabelled_overlay_text_receives_a_semantic_figma_role():
    ocr = {"lines": [{"id": "L", "text": "SHOP NOW", "conf": .99,
                      "box": {"x": 40, "y": 400, "w": 180, "h": 35},
                      "meta": {"ownership_decision": {"placement": "overlay", "owner": "none",
                                                          "action": "recreate", "confidence": .99}}}]}
    layer = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))["c_L"]
    assert layer["target"] == "text"
    assert layer["meta"]["role"] == "cta"


def test_intentional_raster_cluster_bakes_internal_text_but_preserves_positive_overlay():
    panel = {
        "id": "panel", "box": {"x": 80, "y": 80, "w": 400, "h": 320},
        "kind": "photo-fragment", "area": 128_000, "coverage": .35, "role": "ui_panel",
    }
    ocr = {"lines": [
        {"id": "inside", "text": "likes 128", "conf": .99, "role": "headline",
         "box": {"x": 120, "y": 150, "w": 150, "h": 30}},
        {"id": "overlay", "text": "External offer", "conf": .99, "role": "headline",
         "box": {"x": 120, "y": 220, "w": 220, "h": 30},
         "meta": {"ownership_decision": {"placement": "overlay", "owner": "none",
                                             "action": "recreate", "confidence": .99}}},
    ]}
    merged = _by_id(merge_layers.merge(ocr, [panel], [], CANVAS, {}))

    assert merged["c_panel"]["target"] == "image"
    assert merged["c_inside"]["target"] == "drop"
    assert merged["c_inside"]["meta"]["baked_owner_id"] == "c_panel"
    assert merged["c_overlay"]["target"] == "text"
    assert merged["c_overlay"]["meta"]["parent_id"] == "c_panel"
    assert merged["c_overlay"]["meta"]["external_overlay"] is True


def test_word_level_style_evidence_becomes_exact_editable_text_run():
    base = {"fontFamily": "Inter", "fontSize": 36, "fontWeight": 700, "color": "#111111"}
    accent = {**base, "color": "#ff2244", "colorRGB": [255, 34, 68],
              "fill": {"kind": "flat", "color": "#ff2244"}}
    ocr = {"lines": [{
        "id": "mixed", "text": "SAVE 30%", "conf": .99,
        "box": {"x": 40, "y": 40, "w": 220, "h": 46}, "style": base,
        "words": [
            {"text": "SAVE", "box": {"x": 40, "y": 40, "w": 100, "h": 46}},
            {"text": "30%", "box": {"x": 150, "y": 40, "w": 80, "h": 46},
             "style": accent,
             "style_evidence": {"source": "word-pixels", "confidence": .93,
                                "changed": ["color"]}},
        ],
    }]}
    mixed = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))["c_mixed"]
    assert [(run["start"], run["end"]) for run in mixed["text_runs"]] == [(5, 8)]
    assert mixed["text_runs"][0]["style"]["color"] == "#ff2244"


def test_inline_ticker_is_marked_only_from_three_exact_observed_repeats():
    ocr = {"lines": [{
        "id": "ticker", "text": "SALE • SALE • SALE", "conf": .99,
        "box": {"x": 10, "y": 10, "w": 560, "h": 30},
        "style": {"fontFamily": "Inter", "fontSize": 20, "fontWeight": 700},
    }]}
    ticker = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))["c_ticker"]
    assert ticker["target"] == "text"
    assert ticker["meta"]["native_repeat"] == {
        "phrase": "SALE", "count": 3, "source": "exact-ocr-sequence",
    }


def test_internal_ui_chrome_is_not_rebuilt_out_of_screenshot_but_external_overlay_is():
    elements = [
        {"id": "shot", "box": {"x": 80, "y": 80, "w": 400, "h": 360},
         "kind": "photo-fragment", "area": 144000, "coverage": .4, "role": "screenshot"},
        {"id": "internal_button", "box": {"x": 130, "y": 330, "w": 120, "h": 40},
         "kind": "shape", "area": 4800, "coverage": .02, "role": "button"},
        {"id": "offer_badge", "box": {"x": 360, "y": 100, "w": 80, "h": 45},
         "kind": "shape", "area": 3600, "coverage": .01, "role": "badge",
         "meta": {"external_overlay": True}},
    ]
    merged = _by_id(merge_layers.merge({"lines": []}, elements, [], CANVAS, {}))
    assert merged["c_shot"]["target"] == "image"
    assert merged["c_shot"]["meta"]["decomposition_policy"]["internal_chrome"] == \
        "baked-in-raster-owner"
    assert merged["c_internal_button"]["target"] == "drop"
    assert merged["c_internal_button"]["meta"]["baked_owner_id"] == "c_shot"
    assert merged["c_offer_badge"]["target"] in {"shape", "icon"}
    assert merged["c_offer_badge"]["meta"]["parent_id"] == "c_shot"


def test_source_detected_outer_shell_around_screenshot_remains_native():
    elements = [
        {"id": "shell", "box": {"x": 65, "y": 65, "w": 430, "h": 390},
         "kind": "shape", "area": 167700, "coverage": .46, "role": "card",
         "meta": {"source_evidenced_shell": True}},
        {"id": "shot", "box": {"x": 80, "y": 80, "w": 400, "h": 360},
         "kind": "photo-fragment", "area": 144000, "coverage": .4, "role": "ui-panel"},
    ]
    merged = _by_id(merge_layers.merge({"lines": []}, elements, [], CANVAS, {}))
    assert merged["c_shell"]["target"] == "shape"
    assert not merged["c_shell"]["meta"].get("baked_owner_id")
    assert merged["c_shot"]["target"] == "image"


def test_vlm_failure_does_not_flatten_explicit_cta_or_headline():
    ocr = {"lines": [{"id": "L", "text": "SHOP NOW", "conf": .99,
                      "role": "cta", "box": {"x": 40, "y": 400, "w": 180, "h": 35},
                      "meta": {"ownership_decision": {"placement": "artifact", "owner": "none",
                                                          "action": "raster_keep", "confidence": 0,
                                                          "reason": "vlm_disagreement"}}}]}
    layer = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))["c_L"]
    assert layer["target"] == "text"
    assert layer["meta"]["ownership_recovery"] == "explicit-overlay-role-after-vlm-failure"


def test_shape_that_is_pure_text_box_is_deduped():
    """A 'shape' element whose box is essentially an OCR text box is dropped in
    favor of the editable text candidate."""
    ocr = {"lines": [{"id": "L0", "text": "CLICK", "conf": 0.95,
                      "box": {"x": 100, "y": 100, "w": 120, "h": 40}}]}
    elements = [{"id": "E0", "box": {"x": 99, "y": 99, "w": 122, "h": 42},
                 "kind": "shape", "area": 5100, "coverage": 0.02,
                 "source": "residual-cc"}]
    cands = merge_layers.merge(ocr, elements, [], CANVAS, {})
    ids = {c["id"] for c in cands}
    assert "c_L0" in ids
    assert "c_E0" not in ids  # deduped away


def test_qwen_only_layer_becomes_image():
    qwen = [{"id": "Q0", "box": {"x": 10, "y": 10, "w": 100, "h": 100},
             "png": "qwen_layers/Q0.png", "kind_hint": "object"}]
    cands = merge_layers.merge({"lines": []}, [], qwen, CANVAS, {})
    m = _by_id(cands)
    assert "c_Q0" in m
    assert m["c_Q0"]["target"] == "image"
    assert m["c_Q0"]["src"] == "qwen_layers/Q0.png"


def test_empty_inputs():
    assert merge_layers.merge({"lines": []}, [], [], CANVAS, {}) == []


def test_paragraph_block_preserves_wrapping_alignment_line_height_and_baselines():
    lines = [
        {"id": "L0", "text": "First", "conf": .99,
         "box": {"x": 20, "y": 20, "w": 100, "h": 20},
         "baseline": {"x0": 20, "y0": 36, "x1": 120, "y1": 36},
         "style": {"fontFamily": "Inter", "fontSize": 16}},
        {"id": "L1", "text": "Second", "conf": .98,
         "box": {"x": 20, "y": 44, "w": 110, "h": 20},
         "baseline": {"x0": 20, "y0": 60, "x1": 130, "y1": 60},
         "style": {"fontFamily": "Inter", "fontSize": 16}},
    ]
    block = {
        "id": "B0", "line_ids": ["L0", "L1"], "text": "First\nSecond",
        "box": {"x": 20, "y": 20, "w": 110, "h": 44},
        "painted_box": {"x": 22, "y": 22, "w": 106, "h": 40},
        "alignment": "CENTER", "line_height": 24, "role": "body", "meta": {},
    }
    candidate = _by_id(merge_layers.merge(
        {"lines": lines, "blocks": [block], "styles": []}, [], [], CANVAS, {}
    ))["c_B0"]
    assert candidate["style"]["lineCount"] == 2
    assert candidate["style"]["lineHeight"] == 24
    assert candidate["style"]["align"] == "CENTER"
    assert candidate["meta"]["baseline_first"]["y0"] == 36
    assert candidate["meta"]["baseline_last"]["y0"] == 60
    assert candidate["text_runs"] == []


def test_paragraph_block_preserves_distinct_line_styles_as_exact_text_runs():
    """A multi-line node must not flatten its second line to the first line's style."""
    lines = [
        {"id": "L0", "text": "Regular", "conf": .99,
         "box": {"x": 20, "y": 20, "w": 100, "h": 20}, "role": "body",
         "style": {"fontFamily": "Inter", "fontStyle": "Regular", "fontSize": 16,
                   "fontWeight": 400, "color": "#111111", "lineHeight": 24}},
        {"id": "L1", "text": "Bold", "conf": .98,
         "box": {"x": 20, "y": 44, "w": 100, "h": 20}, "role": "body",
         "style": {"fontFamily": "Inter", "fontStyle": "Bold", "fontSize": 18,
                   "fontWeight": 700, "color": "#dd2244", "lineHeight": 24}},
    ]
    block = {
        "id": "B0", "line_ids": ["L0", "L1"], "text": "Regular\nBold",
        "box": {"x": 20, "y": 20, "w": 100, "h": 44},
        "painted_box": {"x": 20, "y": 20, "w": 100, "h": 44},
        "alignment": "LEFT", "line_height": 24, "role": "body", "meta": {},
    }

    candidate = _by_id(merge_layers.merge(
        {"lines": lines, "blocks": [block], "styles": []}, [], [], CANVAS, {}
    ))["c_B0"]

    assert candidate["text"] == "Regular\nBold"
    assert [(run["start"], run["end"]) for run in candidate["text_runs"]] == [(0, 7), (8, 12)]
    assert candidate["text_runs"][0]["style"]["fontWeight"] == 400
    assert candidate["text_runs"][1]["style"]["fontWeight"] == 700
    assert candidate["text_runs"][1]["style"]["color"] == "#dd2244"


def test_partial_block_list_cannot_delete_orphan_ocr_lines():
    lines = [
        {"id": "L0", "text": "Headline", "conf": .99,
         "box": {"x": 20, "y": 20, "w": 120, "h": 30}},
        {"id": "L1", "text": "SHOP NOW", "conf": .98,
         "box": {"x": 20, "y": 100, "w": 110, "h": 24}, "role": "cta"},
    ]
    blocks = [{"id": "B0", "line_ids": ["L0"], "text": "Headline",
               "box": dict(lines[0]["box"]), "painted_box": dict(lines[0]["box"]),
               "role": "headline", "meta": {}}]
    merged = merge_layers.merge({"lines": lines, "blocks": blocks}, [], [], CANVAS, {})
    texts = {candidate.get("text") for candidate in merged}
    assert {"Headline", "SHOP NOW"} <= texts


def test_low_fidelity_block_meta_survives_the_block_path_into_the_candidate():
    """Regression for the block-path fidelity drop: production OCR always carries a
    non-empty "blocks" array (text_analysis._make_blocks emits >=1 block per line), so
    _text_sources always prefers blocks over lines. A block produced by a real
    low-fidelity line must still carry meta.low_fidelity/fallback_src through to the
    merged candidate so routing.py can gate it to a masked-pixel fallback instead of
    guessed text."""
    line = {
        "id": "L0", "text": "SALE", "conf": 0.9,
        "box": {"x": 40, "y": 30, "w": 220, "h": 60},
        "meta": {
            "fidelity_confidence": 0.1,
            "low_fidelity": True,
            "fidelity_reason": "ink_confidence:0.10<0.30",
            "fallback_src": "fallback_crops/L0.png",
            "substitution": {"from": "text", "to": "masked-pixel-fallback",
                              "reason": "ink_confidence:0.10<0.30", "confidence": 0.1},
        },
    }
    block = {
        "id": "B0", "type": "text", "line_ids": ["L0"], "text": "SALE",
        "box": dict(line["box"]), "painted_box": dict(line["box"]),
        "alignment": "left", "line_height": 20.0, "role": "text",
        "hierarchy": {"level": 0, "parent_id": None}, "style_id": None,
        "meta": {
            "fidelity_confidence": 0.1,
            "low_fidelity": True,
            "fidelity_reason": "ink_confidence:0.10<0.30",
            "fallback_src": "fallback_crops/L0.png",
            "substitution": {"from": "text", "to": "masked-pixel-fallback",
                              "reason": "ink_confidence:0.10<0.30", "confidence": 0.1},
        },
    }
    ocr = {"lines": [line], "blocks": [block], "styles": []}

    cands = merge_layers.merge(ocr, [], [], CANVAS, {})
    m = _by_id(cands)
    assert "c_B0" in m
    meta = m["c_B0"]["meta"]
    assert meta.get("low_fidelity") is True
    assert meta.get("fallback_src") == "fallback_crops/L0.png"
    # routing.py must gate this to the masked-pixel fallback, not guessed text, and it
    # must use the real saved crop (src) rather than falling into the genericmask-only
    # "else" branch that loses the actual pixels.
    assert m["c_B0"]["target"] == "image"
    assert m["c_B0"].get("src") == "fallback_crops/L0.png"


def test_vlm_font_winner_on_line_overrides_stale_block_font():
    line = {
        "id": "L0", "text": "Post body", "conf": .9,
        "box": {"x": 10, "y": 20, "w": 200, "h": 30},
        "vlm_font_judged": True,
        "style": {"fontFamily": "Arial", "fontStyle": "Regular", "fontWeight": 400,
                  "fontCandidates": [{"family": "Arial", "vlm_score": 9}]},
    }
    block = {
        "id": "B0", "text": "Post body", "line_ids": ["L0"],
        "box": dict(line["box"]), "style": {"fontFamily": "Comic Sans MS"},
    }
    merged = merge_layers.merge({"lines": [line], "blocks": [block]}, [], [], CANVAS, {})
    text = next(item for item in merged if item["id"] == "c_B0")
    assert text["style"]["fontFamily"] == "Arial"


def test_dedup_text_collapses_explicit_layer_ids():
    ocr = {
        "lines": [
            {"id": "B3", "text": "UPFRONT", "conf": 0.99,
             "box": {"x": 40, "y": 30, "w": 120, "h": 24}, "role": "headline"},
            {"id": "B19", "text": "UPFRONT", "conf": 0.62,
             "box": {"x": 42, "y": 32, "w": 116, "h": 22}, "role": "headline"},
        ]
    }
    cfg = {"merge": {
        "dedup_text": True,
        "duplicate_text": ["UPFRONT"],
        "layer_ids": ["c_B3", "c_B19"],
        "dedup_iou": 0.72,
    }}
    merged = merge_layers.merge(ocr, [], [], CANVAS, cfg)
    ids = {item["id"] for item in merged}
    assert "c_B3" in ids
    assert "c_B19" not in ids


def test_duplicate_timestamp_rows_from_different_sources_collapse_to_one():
    """Regression for run 009: one OCR engine read the whole timestamp row, another split
    it into a fragment. Near-identical geometry + one content subset of the other -> the
    complete row survives, the fragment is dropped, and provenance is recorded on the
    keeper. Without this, all copies re-render on top of each other (ghost timestamp)."""
    ocr = {"lines": [
        {"id": "full", "text": "05:00 PM . 12-05-2026 - 121K weergaven", "conf": 0.82,
         "box": {"x": 20, "y": 520, "w": 560, "h": 30}, "role": "label"},
        {"id": "frag", "text": "12-05-2026 121K weergaven", "conf": 0.90,
         "box": {"x": 180, "y": 521, "w": 400, "h": 32}, "role": "label"},
    ]}
    merged = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))
    assert "c_full" in merged
    assert "c_frag" not in merged  # the fragment collapsed into the complete row
    assert "c_frag" in merged["c_full"]["meta"]["deduped_text_ids"]


def test_text_dedup_keys_on_content_containment_not_iou_alone():
    """A short subset fragment ('05:00 PM') can share a low IoU with the full row yet be
    fully contained inside it. Geometry+content dedup must still collapse it — the point of
    not relying on an IoU threshold alone."""
    ocr = {"lines": [
        {"id": "row", "text": "05:00 PM . 12-05-2026 - 121K weergaven", "conf": 0.80,
         "box": {"x": 20, "y": 520, "w": 560, "h": 30}, "role": "label"},
        {"id": "left", "text": "05:00 PM", "conf": 0.85,
         "box": {"x": 16, "y": 521, "w": 140, "h": 28}, "role": "label"},
    ]}
    merged = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))
    assert "c_row" in merged
    assert "c_left" not in merged  # low IoU, but fully contained + content subset


def test_distinct_neighbouring_text_is_not_deduped():
    """Guard against over-dedup: two different footer counts near each other must both
    survive (no shared content tokens)."""
    ocr = {"lines": [
        {"id": "likes", "text": "257", "conf": 0.98,
         "box": {"x": 100, "y": 540, "w": 60, "h": 28}, "role": "footer"},
        {"id": "views", "text": "21K", "conf": 0.98,
         "box": {"x": 110, "y": 542, "w": 60, "h": 28}, "role": "footer"},
    ]}
    merged = _by_id(merge_layers.merge(ocr, [], [], CANVAS, {}))
    assert "c_likes" in merged
    assert "c_views" in merged


def test_fragment_absorbed_into_paragraph_block_is_not_rendered_twice():
    """Regression for run 009 'geld': a lone fragment line the paragraph block absorbed
    duplicates a richer standalone line, so the word rendered twice at different offsets.
    The fragment is stripped from the block; the richer standalone line survives."""
    lines = [
        {"id": "L0", "text": "bestellingen hun", "conf": .95,
         "box": {"x": 40, "y": 100, "w": 400, "h": 24}, "role": "body"},
        {"id": "L_frag", "text": "geld", "conf": .70,
         "box": {"x": 40, "y": 128, "w": 60, "h": 26}, "role": "body"},
        {"id": "L2", "text": "Schrijf je nu in", "conf": .95,
         "box": {"x": 40, "y": 220, "w": 380, "h": 24}, "role": "body"},
        {"id": "L_full", "text": "geld terug tot €100.", "conf": .92,
         "box": {"x": 44, "y": 130, "w": 250, "h": 24}, "role": "price"},
    ]
    blocks = [
        {"id": "B_body", "line_ids": ["L0", "L_frag", "L2"],
         "text": "bestellingen hun\ngeld\nSchrijf je nu in",
         "box": {"x": 40, "y": 100, "w": 400, "h": 144}, "role": "body", "meta": {}},
        {"id": "B_price", "line_ids": ["L_full"], "text": "geld terug tot €100.",
         "box": dict(lines[3]["box"]), "role": "price", "meta": {}},
    ]
    merged = _by_id(merge_layers.merge(
        {"lines": lines, "blocks": blocks, "styles": []}, [], [], CANVAS, {}))
    assert merged["c_B_body"]["text"] == "bestellingen hun\nSchrijf je nu in"
    assert "geld" not in merged["c_B_body"]["text"]
    assert merged["c_B_price"]["text"] == "geld terug tot €100."


def test_scene_text_is_never_also_an_editable_overlay():
    """Contract with reconstruct: text kept baked in a product photo must not also carry
    overlay_text/removal_required, or reconstruct._is_text_removal erases its pixels from
    the plate without re-emitting a layer. merge enforces mutual exclusivity here."""
    elements = [{"id": "product", "box": {"x": 100, "y": 100, "w": 300, "h": 300},
                 "kind": "photo-fragment", "area": 90000, "coverage": .25, "role": "product"}]
    ocr = {"lines": [{
        "id": "printed", "text": "500ML", "conf": .95, "role": "body",
        "box": {"x": 160, "y": 200, "w": 90, "h": 24},
        "meta": {"scene_text_role": "printed_on_product",
                 "overlay_text": True, "removal_required": True}}]}
    qwen = [{"id": "Q", "box": {"x": 100, "y": 100, "w": 300, "h": 300},
             "png": "q.png", "kind_hint": "photo"}]
    printed = _by_id(merge_layers.merge(ocr, elements, qwen, CANVAS, {}))["c_printed"]
    assert printed["target"] == "drop"
    assert printed["meta"]["kept_in_photo"] is True
    assert not printed["meta"].get("overlay_text")
    assert not printed["meta"].get("removal_required")
    assert "overlay_text" in printed["meta"]["scene_text_contract_enforced"]


def test_merge_layer_order_is_deterministic():
    """Stable sort keys (z, area, reading order, id) make the merged layer list reproducible
    for identical inputs, so downstream runs diff cleanly."""
    ocr = _ocr()
    elements = _elements()
    order_a = [c["id"] for c in merge_layers.merge(ocr, elements, _qwen(), CANVAS, {})]
    order_b = [c["id"] for c in merge_layers.merge(ocr, elements, _qwen(), CANVAS, {})]
    assert order_a == order_b


# ── product-label scene text (benchmark 002 real geometry) ───────────────────────────
# 002 is an UPFRONT supplement bundle: three product packages sit on a white plate under
# an "ALLE ESSENTIALS" headline / "€63 → €49" price / "KOOP NU" CTA. OCR reads every
# printed package label (product names, nutrition tables, ingredient lists) with a bold
# headline-ish style, so the semantic router labels them "subheadline"/"offer". These
# boxes are copied from the real run's canvas coordinates.
_C002 = {"w": 1080, "h": 1920}


def _products_002():
    # E005/E006/E007 are the segmented product cutouts (role=product); E003 is the big
    # low-confidence scene "shape" that is NOT a discrete cutout (must not own scene text).
    return [
        {"id": "E003", "box": {"x": 55, "y": 502, "w": 1025, "h": 1418}, "kind": "shape",
         "area": 1453450, "coverage": .70, "role": "shape"},
        {"id": "E005", "box": {"x": 673, "y": 910, "w": 343, "h": 382},
         "kind": "photo-fragment", "area": 128580, "coverage": .06, "role": "product"},
        {"id": "E006", "box": {"x": 61, "y": 933, "w": 609, "h": 740},
         "kind": "photo-fragment", "area": 443490, "coverage": .21, "role": "product"},
        {"id": "E007", "box": {"x": 675, "y": 1292, "w": 339, "h": 383},
         "kind": "photo-fragment", "area": 127582, "coverage": .06, "role": "product"},
    ]


def test_product_label_text_over_a_cutout_is_kept_in_photo_not_native():
    """The #1 002 defect: a product name printed on the package (read as a bold
    'subheadline') sits inside the product cutout, so it is scene text — baked into the
    raster, never a native layer, never removed from the plate."""
    ocr = {"lines": [
        # UPFRONT / CREATINE printed on the E005 cutout (0.6-1.0 inside it)
        {"id": "up", "text": "UPFRONT", "conf": .9, "role": "subheadline",
         "box": {"x": 668, "y": 863, "w": 260, "h": 125}},
        {"id": "cr", "text": "CREATINE", "conf": .9, "role": "subheadline",
         "box": {"x": 668, "y": 944, "w": 122, "h": 56}},
        # a nutrition/ingredient line printed on E006, read as 'offer'
        {"id": "ing", "text": "wei-eiwit concentraat (melk)", "conf": .8, "role": "offer",
         "box": {"x": 106, "y": 1100, "w": 318, "h": 49}},
    ]}
    m = _by_id(merge_layers.merge(ocr, _products_002(), [], _C002, {}))
    for cid in ("c_up", "c_cr", "c_ing"):
        node = m[cid]
        assert node.get("kept_in_photo") is True, cid
        assert node["target"] == "drop", cid
        assert node["meta"]["role"] == "scene-text", cid
        # single owner: baked, so its pixels are NOT scheduled for removal
        assert not node["meta"].get("overlay_text"), cid
        assert not node["meta"].get("removal_required"), cid
        assert node["meta"].get("baked_owner_id") in {"c_E005", "c_E006", "c_E007"}, cid


def test_overlay_headline_and_price_on_the_plate_stay_editable():
    """Guard against over-baking: the ALLE ESSENTIALS headline, €63→€49 price and KOOP NU
    CTA live on the white plate (above/outside every product cutout), so they remain
    native editable text with their glyphs removed from the plate."""
    ocr = {"lines": [
        {"id": "hl", "text": "ALLE ESSENTIALS", "conf": .95, "role": "headline",
         "box": {"x": 116, "y": 250, "w": 700, "h": 90}},
        {"id": "pr", "text": "€63 → €49", "conf": .9, "role": "price",
         "box": {"x": 285, "y": 360, "w": 390, "h": 120}},
        {"id": "cta", "text": "KOOP NU VIA UPFRONT.NL", "conf": .9, "role": "cta",
         "box": {"x": 226, "y": 430, "w": 518, "h": 60}},
    ]}
    m = _by_id(merge_layers.merge(ocr, _products_002(), [], _C002, {}))
    for cid in ("c_hl", "c_pr", "c_cta"):
        node = m[cid]
        assert node["target"] == "text", cid
        assert not node.get("kept_in_photo"), cid
        assert node["meta"].get("overlay_text") is True, cid
        assert node["meta"].get("removal_required") is True, cid


def test_full_bleed_background_photo_overlay_copy_stays_editable():
    """A full-bleed hero/UGC photo IS the canvas background, not a discrete object cutout,
    so deliberate overlay copy printed on top of it must stay editable — a full-bleed
    raster is excluded from the product-cutout owners."""
    elements = [{"id": "hero", "box": {"x": 0, "y": 0, "w": 1080, "h": 1920},
                 "kind": "photo-fragment", "area": 2073600, "coverage": 1.0, "role": "photo"}]
    ocr = {"lines": [{"id": "copy", "text": "SUMMER SALE", "conf": .95, "role": "headline",
                      "box": {"x": 300, "y": 900, "w": 480, "h": 120}}]}
    node = _by_id(merge_layers.merge(ocr, elements, [], _C002, {}))["c_copy"]
    assert node["target"] == "text"
    assert not node.get("kept_in_photo")
    assert node["meta"].get("overlay_text") is True


def test_product_label_bakes_even_below_the_raster_cluster_threshold():
    """A label only ~60% inside its package box (ascenders/kerning spill past the mask)
    still bakes: scene_text_inside_frac (0.55) is deliberately looser than the strict
    raster-cluster containment gate."""
    ocr = {"lines": [{"id": "sp", "text": "UPFRONT", "conf": .9, "role": "subheadline",
                      "box": {"x": 668, "y": 863, "w": 260, "h": 125}}]}  # ~0.61 inside E005
    node = _by_id(merge_layers.merge(ocr, _products_002(), [], _C002, {}))["c_sp"]
    assert node.get("kept_in_photo") is True
    assert node["meta"].get("baked_owner_id") == "c_E005"


def test_positive_overlay_evidence_beats_product_geometry():
    """If a VLM (or explicit promotion) positively says overlay copy, that wins even when
    the box sits inside a product cutout — geometry only decides the unlabeled case."""
    elements = [{"id": "E005", "box": {"x": 673, "y": 910, "w": 343, "h": 382},
                 "kind": "photo-fragment", "area": 128580, "coverage": .06, "role": "product"}]
    ocr = {"lines": [{"id": "ov", "text": "50% OFF", "conf": .9, "role": "subheadline",
                      "box": {"x": 700, "y": 1000, "w": 200, "h": 60},
                      "meta": {"external_overlay": True, "overlay_text": True}}]}
    node = _by_id(merge_layers.merge(ocr, elements, [], _C002, {}))["c_ov"]
    assert node["target"] == "text"
    assert not node.get("kept_in_photo")


def test_text_bearing_logo_badge_extracts_ocr_as_native_text():
    """Benchmark 016 green seal: a mislabeled 'logo' hosting 'Get up to' / '45%' / 'Off'
    must become TEXT + plate shell, never bake OCR into the badge raster."""
    canvas = {"w": 1080, "h": 1080}
    elements = [{"id": "E014", "box": {"x": 774, "y": 540, "w": 256, "h": 254},
                 "kind": "icon", "area": 65024, "coverage": .06, "role": "logo"}]
    ocr = {"lines": [
        {"id": "get", "text": "Get up to", "conf": .9,
         "box": {"x": 820, "y": 602, "w": 162, "h": 38}},
        {"id": "pct", "text": "45%", "conf": .95,
         "box": {"x": 792, "y": 646, "w": 187, "h": 60}},
        {"id": "off", "text": "Off", "conf": .9,
         "box": {"x": 875, "y": 716, "w": 54, "h": 30}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    for cid in ("c_get", "c_pct", "c_off"):
        node = m[cid]
        assert node["target"] == "text", cid
        assert not node.get("kept_in_photo"), cid
        assert node["meta"].get("overlay_text") is True, cid
        assert node["meta"].get("removal_required") is True, cid
        assert node["meta"].get("shell_text_host") == "c_E014", cid
        assert node["meta"].get("suppression_reason") != "text-inside-product-cutout", cid
    shell = m["c_E014"]
    assert shell["meta"].get("text_bearing_shell") is True
    assert shell["meta"].get("plate_shell") is True
    assert shell["meta"].get("role") == "badge"
    assert shell["target"] == "shape"


def test_ad013_circular_offer_badge_extracts_native_text():
    """Ad 013: ``61% OFF`` / ``+ FREE GIFTS`` on a circular logo seal must stay editable."""
    canvas = {"w": 1080, "h": 1920}
    elements = [{"id": "E007", "box": {"x": 97, "y": 730, "w": 303, "h": 304},
                 "kind": "icon", "area": 71757, "coverage": .03, "role": "logo"}]
    ocr = {"lines": [
        {"id": "off", "text": "61% OFF", "conf": .95,
         "box": {"x": 97, "y": 807, "w": 284, "h": 105}},
        {"id": "gifts", "text": "+ FREE GIFTS", "conf": .9,
         "box": {"x": 117, "y": 873, "w": 268, "h": 88}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    for cid in ("c_off", "c_gifts"):
        node = m[cid]
        assert node["target"] == "text", cid
        assert not node.get("kept_in_photo"), cid
        assert node["meta"].get("shell_text_host") == "c_E007", cid
    assert m["c_E007"]["meta"].get("role") == "badge"
    assert m["c_E007"]["meta"].get("text_bearing_shell") is True


def test_badge_shell_preferred_over_enclosing_product_for_ocr():
    """When offer text sits inside both a product pouch and a badge seal, the seal wins."""
    canvas = {"w": 1080, "h": 1080}
    elements = [
        {"id": "E013", "box": {"x": 40, "y": 420, "w": 1000, "h": 620},
         "kind": "photo-fragment", "area": 620000, "coverage": .5, "role": "product"},
        {"id": "E014", "box": {"x": 774, "y": 540, "w": 256, "h": 254},
         "kind": "icon", "area": 65024, "coverage": .06, "role": "logo"},
    ]
    ocr = {"lines": [
        {"id": "pct", "text": "45%", "conf": .95,
         "box": {"x": 792, "y": 646, "w": 187, "h": 60}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    node = m["c_pct"]
    assert node["target"] == "text"
    assert node["meta"].get("shell_text_host") == "c_E014"
    assert node["meta"].get("suppression_reason") != "text-inside-product-cutout"
    assert m["c_E014"]["meta"].get("text_bearing_shell") is True
    assert m["c_E014"]["target"] == "shape"


def test_brushstroke_banner_shape_pairs_editable_text():
    """028-style olive brushstroke: irregular shape + inset copy → shell + native TEXT.

    Geometry only (no VLM): high inside_frac of OCR in a wide colored shape marks
    text_bearing_shell / plate_shell and keeps the copy editable.
    """
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "E_banner", "box": {"x": 80, "y": 220, "w": 920, "h": 140},
        "kind": "shape", "area": 920 * 140, "coverage": 0.09, "role": "shape",
        "source": "sam3",
    }]
    ocr = {"lines": [{
        "id": "sold", "text": "ALMOST SOLD OUT...", "conf": 0.95, "role": "offer",
        "box": {"x": 160, "y": 255, "w": 760, "h": 70},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    text = m["c_sold"]
    shell = m["c_E_banner"]
    assert text["target"] == "text"
    assert not text.get("kept_in_photo")
    assert text["meta"].get("overlay_text") is True
    assert text["meta"].get("removal_required") is True
    assert text["meta"].get("shell_text_host") == "c_E_banner"
    assert shell["target"] == "shape"
    assert shell["meta"].get("text_bearing_shell") is True
    assert shell["meta"].get("plate_shell") is True
    assert shell["meta"].get("role") == "banner"
    assert shell["meta"].get("geometric_text_shell") is True
    assert "ALMOST SOLD OUT" in (shell["meta"].get("shell_text_snippet") or "")


def test_generic_broad_residual_is_not_promoted_to_text_shell():
    """002 regression: white residual negative-space slabs are not banners/badges."""
    canvas = {"w": 1080, "h": 1920}
    elements = [{
        "id": "E_plate", "box": {"x": 102, "y": 336, "w": 876, "h": 147},
        "kind": "shape", "area": 96_830, "coverage": 0.047, "role": "shape",
        "provenance": {"sources": ["residual", "sam3:box-refine"]},
    }]
    ocr = {"lines": [{
        "id": "headline", "text": "KRACHTSPORT BUNDEL", "conf": 0.98,
        "role": "headline", "box": {"x": 150, "y": 360, "w": 780, "h": 84},
    }]}
    merged = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    shell = merged["c_E_plate"]
    text = merged["c_headline"]

    assert shell["meta"].get("geometric_text_shell") is not True
    assert shell["meta"].get("role") == "shape"
    assert shell["target"] == "drop"
    assert shell["meta"]["suppression_reason"] == "residual-negative-space-around-text"
    assert text["meta"].get("shell_text_host") is None


def test_explicit_broad_banner_can_still_host_editable_text():
    """A detector-named banner is not rejected merely because it spans the canvas."""
    canvas = {"w": 1080, "h": 1920}
    elements = [{
        "id": "E_banner", "box": {"x": 102, "y": 336, "w": 876, "h": 147},
        "kind": "shape", "area": 96_830, "coverage": 0.047, "role": "banner",
        "provenance": {"sources": ["residual", "sam3:box-refine"]},
    }]
    ocr = {"lines": [{
        "id": "headline", "text": "ALMOST SOLD OUT", "conf": 0.98,
        "role": "headline", "box": {"x": 150, "y": 360, "w": 780, "h": 84},
    }]}
    merged = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))

    assert merged["c_E_banner"]["meta"].get("text_bearing_shell") is True
    assert merged["c_headline"]["meta"].get("shell_text_host") == "c_E_banner"


def test_starburst_seal_shape_pairs_editable_text():
    """028-style starburst seal: square irregular badge + inset offer copy."""
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "E_seal", "box": {"x": 780, "y": 520, "w": 240, "h": 240},
        "kind": "icon", "area": 57600, "coverage": 0.04, "role": "shape",
        "source": "sam3",
    }]
    ocr = {"lines": [{
        "id": "ltd", "text": "LIMITED TIME OFFER", "conf": 0.92, "role": "offer",
        "box": {"x": 810, "y": 590, "w": 180, "h": 100},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_ltd"]["target"] == "text"
    assert m["c_ltd"]["meta"].get("shell_text_host") == "c_E_seal"
    assert m["c_E_seal"]["meta"].get("text_bearing_shell") is True
    assert m["c_E_seal"]["meta"].get("role") == "badge"
    assert m["c_E_seal"]["target"] == "shape"


def test_product_box_inset_text_stays_kept_in_photo():
    """Embossed product/mask text must bake — geometric shell promotion must not steal it."""
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "E_mask", "box": {"x": 200, "y": 600, "w": 680, "h": 520},
        "kind": "photo-fragment", "area": 680 * 520, "coverage": 0.24, "role": "product",
        "source": "sam3",
    }]
    ocr = {"lines": [{
        "id": "brand", "text": "DORE & ROSE", "conf": 0.9, "role": "label",
        "box": {"x": 320, "y": 820, "w": 440, "h": 50},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    node = m["c_brand"]
    assert node.get("kept_in_photo") is True
    assert node["target"] == "drop"
    assert node["meta"].get("baked_owner_id") == "c_E_mask"
    assert m["c_E_mask"]["meta"].get("text_bearing_shell") is not True


def test_merge_report_records_dedup_reasons_when_run_dir_given(tmp_path):
    """Diagnostics counts/reasons are emitted as a sidecar report next to merged.json."""
    import json
    ocr = {"lines": [
        {"id": "full", "text": "05:00 PM . 12-05-2026 - 121K weergaven", "conf": 0.82,
         "box": {"x": 20, "y": 520, "w": 560, "h": 30}, "role": "label"},
        {"id": "frag", "text": "12-05-2026 121K weergaven", "conf": 0.90,
         "box": {"x": 180, "y": 521, "w": 400, "h": 32}, "role": "label"},
    ]}
    run_dir = str(tmp_path)
    merge_layers.merge(ocr, [], [], CANVAS, {}, run_dir=run_dir)
    merged = json.load(open(os.path.join(run_dir, "merged.json"), encoding="utf-8"))
    assert isinstance(merged, list)  # scene_intent requires a plain list
    report = json.load(open(os.path.join(run_dir, "merge_report.json"), encoding="utf-8"))
    assert report["counts"]["text_dedup"] == 1
    assert report["text_dedup"][0]["dropped"] == "c_frag"
    assert report["text_dedup"][0]["kept"] == "c_full"


def test_fusion_parent_id_rewritten_to_canonical_candidate_id():
    """Fusion emits parent_id=E010; merge prefixes ids as c_E010. Layout nesting
    keys on meta.parent_id ∈ candidate ids, so the link must be rewritten or the
    nested icon ships as a duplicate top-level layer (benchmark-final4/009)."""
    elements = [
        {"id": "E010", "box": {"x": 26, "y": 1010, "w": 54, "h": 50},
         "kind": "shape", "role": "button", "area": 1911, "coverage": 0.01,
         "score": 0.5},
        {"id": "E011", "box": {"x": 40, "y": 1020, "w": 24, "h": 24},
         "kind": "icon", "role": "icon", "area": 400, "coverage": 0.002,
         "score": 0.9, "parent_id": "E010"},
    ]
    merged = _by_id(merge_layers.merge({"lines": []}, elements, [], {"w": 1080, "h": 1080}, {}))
    assert "c_E010" in merged and "c_E011" in merged
    assert merged["c_E011"]["meta"]["parent_id"] == "c_E010"


def test_near_duplicate_button_icon_nest_collapses_to_one_owner():
    """Run 009 engagement chrome: button shell + icon share ~the same box (IoU≈0.98).
    Keep the more specific icon; drop the redundant button shell so ownership stays
    single and node count does not balloon."""
    elements = [
        {"id": "E010", "box": {"x": 23, "y": 1007, "w": 60, "h": 56},
         "kind": "shape", "role": "button", "area": 1911, "coverage": 0.01,
         "score": 0.5},
        {"id": "E011", "box": {"x": 23, "y": 1007, "w": 59, "h": 56},
         "kind": "icon", "role": "icon", "area": 1158, "coverage": 0.01,
         "score": 0.98, "parent_id": "E010"},
    ]
    merged = _by_id(merge_layers.merge({"lines": []}, elements, [], {"w": 1080, "h": 1080}, {}))
    assert merged["c_E010"]["target"] == "drop"
    assert merged["c_E010"]["meta"]["suppression_reason"] == "near-duplicate-nested-shell"
    assert merged["c_E011"]["target"] in {"icon", "image", "shape"}
    assert merged["c_E011"]["meta"].get("parent_id") in (None, "")
    assert merged["c_E011"]["meta"].get("absorbed_shell_id") == "c_E010"


def test_redundant_arrow_icon_dropped_when_price_text_already_has_arrow():
    """Bench 002: OCR reads ``€63 → €49`` and SAM also proposes an arrow icon over the
    glyph — keep native text, drop the overlapping arrow vector."""
    ocr = {"lines": [
        {"id": "pr", "text": "€63 → €49", "conf": 0.95, "role": "price",
         "box": {"x": 346, "y": 544, "w": 381, "h": 58}},
    ]}
    elements = [
        {"id": "E004", "box": {"x": 506, "y": 545, "w": 57, "h": 57},
         "kind": "icon", "role": "arrow", "area": 1133, "coverage": 0.001, "score": 0.9},
        # Distant arrow must survive (not overlapping the price line).
        {"id": "E099", "box": {"x": 40, "y": 40, "w": 40, "h": 40},
         "kind": "icon", "role": "arrow", "area": 800, "coverage": 0.001, "score": 0.9},
    ]
    merged = _by_id(merge_layers.merge(ocr, elements, [], _C002, {}))
    assert merged["c_pr"]["target"] == "text"
    assert "→" in merged["c_pr"]["text"]
    assert merged["c_E004"]["target"] == "drop"
    assert merged["c_E004"]["meta"]["suppression_reason"] == "redundant-arrow-in-text"
    assert merged["c_E099"]["target"] != "drop"


def test_dashed_guide_rect_hugging_text_is_dropped_underline_kept():
    """Sparse stroke-only rect matching a text box is layout-guide junk; a short
    annotation underline under the same line must survive."""
    ocr = {"lines": [
        {"id": "hl", "text": "SALE", "conf": 0.95, "role": "headline",
         "box": {"x": 100, "y": 200, "w": 220, "h": 48}},
    ]}
    elements = [
        # Dashed guide: text-sized box, low fill fraction (perimeter ink only).
        {"id": "G0", "box": {"x": 96, "y": 196, "w": 228, "h": 56},
         "kind": "shape", "role": "shape", "area": 1800, "coverage": 0.002, "score": 0.4},
        # Short underline bar under the text — keep.
        {"id": "U0", "box": {"x": 110, "y": 250, "w": 160, "h": 6},
         "kind": "shape", "role": "underline", "area": 900, "coverage": 0.0005, "score": 0.8},
    ]
    merged = _by_id(merge_layers.merge(ocr, elements, [], CANVAS, {}))
    assert merged["c_hl"]["target"] == "text"
    assert merged["c_G0"]["target"] == "drop"
    assert merged["c_G0"]["meta"]["suppression_reason"] == "guide_artifact"
    assert merged["c_G0"]["meta"].get("guide_artifact") is True
    assert merged["c_U0"]["target"] != "drop"


_C014 = {"w": 1080, "h": 1920}


def test_014_callout_leaders_survive_guide_drop_and_stay_off_product():
    """014-style explainer: 4 floating callouts + 4 thin leader strokes around a
    center product. Leaders must not be guide-dropped, text stays TEXT, product
    stays image, and preserve_callout_leaders tags grouping meta."""
    ocr = {"lines": [
        {"id": "hl", "text": "NUTRITIONAL SUPPORT FOR YOUR BODY", "conf": 0.96,
         "role": "headline", "box": {"x": 120, "y": 80, "w": 840, "h": 90}},
        {"id": "c1", "text": "Vitamin D3 for immune health", "conf": 0.92,
         "role": "body", "box": {"x": 60, "y": 520, "w": 260, "h": 70}},
        {"id": "c2", "text": "Zinc for cellular repair", "conf": 0.91,
         "role": "body", "box": {"x": 760, "y": 500, "w": 250, "h": 70}},
        {"id": "c3", "text": "B12 for energy metabolism", "conf": 0.90,
         "role": "body", "box": {"x": 70, "y": 1100, "w": 250, "h": 70}},
        {"id": "c4", "text": "Antioxidants for recovery", "conf": 0.90,
         "role": "body", "box": {"x": 760, "y": 1120, "w": 250, "h": 70}},
        {"id": "fda", "text": "*These statements have not been evaluated by the FDA.",
         "conf": 0.85, "role": "disclaimer",
         "box": {"x": 80, "y": 1780, "w": 920, "h": 36}},
    ]}
    # Center product (hand + gummy).
    product = {
        "id": "PROD", "box": {"x": 380, "y": 640, "w": 320, "h": 520},
        "kind": "photo-fragment", "role": "product", "area": 140000,
        "coverage": 0.07, "score": 0.95,
    }
    # Thin leader strokes from each callout toward the product. Two tagged as
    # callout_leader/arrow; two as generic sparse shapes (the regression case).
    leaders = [
        {"id": "A1", "box": {"x": 300, "y": 560, "w": 90, "h": 28},
         "kind": "icon", "role": "callout_leader", "area": 420, "coverage": 0.0002,
         "score": 0.8, "stroke_only": True},
        {"id": "A2", "box": {"x": 690, "y": 540, "w": 85, "h": 30},
         "kind": "icon", "role": "arrow", "area": 400, "coverage": 0.0002,
         "score": 0.8, "stroke_only": True},
        {"id": "A3", "box": {"x": 300, "y": 1080, "w": 95, "h": 26},
         "kind": "shape", "role": "shape", "area": 380, "coverage": 0.0002,
         "score": 0.7},  # low fill → stroke-like; must NOT guide-drop
        {"id": "A4", "box": {"x": 680, "y": 1100, "w": 90, "h": 28},
         "kind": "shape", "role": "shape", "area": 360, "coverage": 0.0002,
         "score": 0.7},
    ]
    # Poison: a true guide rect hugging the headline — must still drop.
    guide = {
        "id": "G1", "box": {"x": 110, "y": 70, "w": 860, "h": 110},
        "kind": "shape", "role": "shape", "area": 2200, "coverage": 0.001, "score": 0.3,
    }
    cfg = {
        "scene": {
            "preset": {"grouping": {"preserve_callout_leaders": True}},
            "facts": {"leader_lines": True, "photo_coverage": 0.85},
        },
    }
    merged = _by_id(merge_layers.merge(
        ocr, [product] + leaders + [guide], [], _C014, cfg,
    ))

    assert merged["c_hl"]["target"] == "text"
    assert merged["c_fda"]["target"] == "text"
    assert merged["c_PROD"]["target"] == "image"
    assert merged["c_G1"]["target"] == "drop"
    assert merged["c_G1"]["meta"].get("suppression_reason") == "guide_artifact"

    for aid in ("c_A1", "c_A2", "c_A3", "c_A4"):
        assert merged[aid]["target"] != "drop", aid
        assert merged[aid]["meta"].get("guide_artifact") is not True, aid
        assert merged[aid]["meta"].get("callout_leader") or merged[aid]["meta"].get("role") in {
            "arrow", "callout_leader",
        }, aid
        # Never nested under the product photo.
        assert merged[aid]["meta"].get("parent_id") != "c_PROD", aid
        assert merged[aid]["meta"].get("callout_group_id"), aid

    for cid in ("c_c1", "c_c2", "c_c3", "c_c4"):
        assert merged[cid]["target"] == "text", cid
        assert merged[cid]["meta"].get("role") == "callout", cid
        assert merged[cid]["meta"].get("callout_group_id"), cid
        assert merged[cid]["meta"].get("overlay_text") is True, cid


def test_thin_leader_near_text_not_guide_dropped_without_role():
    """Untagged thin stroke grazing callout text but pointing at product must survive."""
    ocr = {"lines": [
        {"id": "c1", "text": "Vitamin D3", "conf": 0.9, "role": "body",
         "box": {"x": 40, "y": 200, "w": 180, "h": 40}},
    ]}
    elements = [
        {"id": "P0", "box": {"x": 320, "y": 220, "w": 200, "h": 240},
         "kind": "photo-fragment", "role": "product", "area": 48000, "coverage": 0.13,
         "score": 0.9},
        # Sparse stroke between text and product — would look guide-like without
        # the callout-leader geometry gate.
        {"id": "L0", "box": {"x": 200, "y": 210, "w": 110, "h": 18},
         "kind": "shape", "role": "shape", "area": 280, "coverage": 0.0004, "score": 0.6},
    ]
    merged = _by_id(merge_layers.merge(ocr, elements, [], CANVAS, {
        "scene": {"preset": {"grouping": {"preserve_callout_leaders": True}}},
    }))
    assert merged["c_L0"]["target"] != "drop"
    assert merged["c_L0"]["meta"].get("suppression_reason") != "guide_artifact"
    assert merged["c_c1"]["target"] == "text"
    assert merged["c_P0"]["target"] == "image"


def test_dense_wide_plate_is_not_thin_stroke_geometry():
    """002 regression: a dense 5:1 slab must not become an arrow/callout leader."""
    box = {"x": 100, "y": 88, "w": 881, "h": 159}

    assert merge_layers._is_thin_stroke_geometry(box, fill_frac=0.76) is False
    assert merge_layers._is_thin_stroke_geometry(
        {"x": 200, "y": 210, "w": 110, "h": 18}, fill_frac=0.14,
    ) is True


def test_verified_arrow_replaces_price_placeholder_and_rules_survive_dedup():
    price = {
        "id": "c_B5", "target": "text", "text": "€63 J €49",
        "box": {"x": 340, "y": 540, "w": 390, "h": 70},
        "text_runs": [{"start": 6, "end": 9, "style": {"fontWeight": 700}}],
        "meta": {"source": "ocr", "role": "price", "pairs_with": "c_arrow"},
    }
    arrow = {
        "id": "c_arrow", "target": "icon", "kind": "icon",
        "box": {"x": 505, "y": 545, "w": 58, "h": 58},
        "meta": {"source": "element", "role": "arrow", "pairs_with": "c_B5"},
    }
    old = {
        "id": "c_old", "target": "text", "text": "€63",
        "box": {"x": 338, "y": 536, "w": 148, "h": 76},
        "meta": {"source": "ocr", "role": "price", "native_decoration_shapes": [{
            "kind": "strikethrough", "x0": 346, "y0": 603,
            "x1": 475, "y1": 548, "color": "#e1491b", "thickness": 4,
        }]},
    }
    new = {
        "id": "c_new", "target": "text", "text": "€49",
        "box": {"x": 586, "y": 536, "w": 154, "h": 76},
        "meta": {"source": "ocr", "role": "price", "native_decoration_shapes": [{
            "kind": "underline", "x0": 595, "y0": 606,
            "x1": 734, "y1": 606, "color": "#e1491b", "thickness": 5,
        }]},
    }

    assert merge_layers._normalize_price_placeholder_with_verified_arrow([price, arrow]) == 1
    assert price["text"] == "€63 €49"
    assert price["text_runs"][0]["start"] == 4
    deduped = merge_layers._dedup_overlapping_text(
        [price, old, new], {}, 0.5, [],
    )
    decorated = merge_layers._materialize_native_price_decorations(deduped)

    assert [item["id"] for item in deduped] == ["c_B5"]
    rules = [item for item in decorated if (item.get("meta") or {}).get("native_decoration")]
    assert {item["meta"]["role"] for item in rules} == {"strikethrough", "underline"}
    assert all(item.get("svg") for item in rules)


# ── 007-like product label vs left-column overlay ─────────────────────────────────────
_C007 = {"w": 1080, "h": 1920}


def _can_007():
    # Right-side silver can cutout (not full-bleed).
    return {"id": "E_can", "box": {"x": 420, "y": 180, "w": 620, "h": 1600},
            "kind": "photo-fragment", "area": 992000, "coverage": .48, "role": "product"}


def _overlay_007():
    return [
        {"id": "date", "text": "20 MEI 20:00", "conf": .95, "role": "headline",
         "box": {"x": 80, "y": 220, "w": 320, "h": 70}},
        {"id": "cta", "text": "Schrijf je nu in, mis geen enkele update.", "conf": .9, "role": "cta",
         "box": {"x": 90, "y": 330, "w": 300, "h": 50}},
        {"id": "sale", "text": "Allerlaatste site-wide sale van 2026.", "conf": .9, "role": "offer",
         "box": {"x": 70, "y": 520, "w": 340, "h": 80}},
    ]


def _on_can_007():
    return [
        {"id": "up", "text": "UPFRONT", "conf": .9, "role": "headline",
         "box": {"x": 560, "y": 420, "w": 280, "h": 90}},
        {"id": "caf", "text": "CAFFEINE 150mg", "conf": .85, "role": "offer",
         "box": {"x": 600, "y": 520, "w": 200, "h": 50}},
        {"id": "nut", "text": "ingredienten bruisend water 9 kJ", "conf": .7, "role": "body",
         "box": {"x": 500, "y": 1100, "w": 400, "h": 200}},
    ]


def test_007_overlay_outside_product_stays_native_text():
    """Left-column flat marketing copy is NOT inside the can → editable TEXT."""
    chrome = [
        {"id": "E_br", "box": {"x": 60, "y": 200, "w": 360, "h": 120},
         "kind": "shape", "role": "shape", "area": 40000, "coverage": .02},
        {"id": "E_cta", "box": {"x": 80, "y": 320, "w": 320, "h": 70},
         "kind": "shape", "role": "button", "area": 22000, "coverage": .01},
    ]
    m = _by_id(merge_layers.merge(
        {"lines": _overlay_007() + _on_can_007()},
        [_can_007()] + chrome, [], _C007,
        {"scene": {"facts": {"flat_background_fraction": 0.55, "photo_coverage": 0.45}}},
    ))
    for cid in ("c_date", "c_cta", "c_sale"):
        node = m[cid]
        assert node["target"] == "text", cid
        assert not node.get("kept_in_photo"), cid
        assert node["meta"].get("overlay_text") is True, cid


def test_007_on_can_label_text_kept_in_photo():
    """Warped on-can branding / nutrition stays baked into the product cutout."""
    m = _by_id(merge_layers.merge(
        {"lines": _overlay_007() + _on_can_007()},
        [_can_007()], [], _C007,
        {"scene": {"facts": {"flat_background_fraction": 0.55, "photo_coverage": 0.45}}},
    ))
    for cid in ("c_up", "c_caf", "c_nut"):
        node = m[cid]
        assert node.get("kept_in_photo") is True, cid
        assert node["target"] == "drop", cid
        assert node["meta"].get("baked_owner_id") == "c_E_can", cid
        assert not node["meta"].get("overlay_text"), cid
        assert not node["meta"].get("removal_required"), cid


def test_oversized_product_box_does_not_swallow_flat_overlay():
    """Loose SAM box covering plate+can must not bake left-column headline/CTA."""
    giant = {"id": "E_big", "box": {"x": 40, "y": 80, "w": 1000, "h": 1750},
             "kind": "photo-fragment", "area": 1_750_000, "coverage": .84, "role": "product"}
    m = _by_id(merge_layers.merge(
        {"lines": _overlay_007() + _on_can_007()},
        [giant], [], _C007,
        {"scene": {"facts": {"flat_background_fraction": 0.55, "photo_coverage": 0.45}}},
    ))
    assert m["c_date"]["target"] == "text"
    assert not m["c_date"].get("kept_in_photo")
    # Overlayish roles (incl. OCR that semantic_text_role promotes to headline) must
    # not bake into the oversized merge — left column stays editable.
    assert m["c_sale"]["target"] == "text"
    assert not m["c_sale"].get("kept_in_photo")


def test_021_handwriting_photo_facts_suppress_editable_ocr():
    """Explicit photo-of-handwriting / scene-text-only facts bake all OCR."""
    canvas = {"w": 338, "h": 600}
    elements = [{"id": "E000", "box": {"x": 0, "y": 0, "w": 338, "h": 600},
                 "kind": "photo-fragment", "area": 202800, "coverage": 1.0, "role": "photo"}]
    ocr = {"lines": [
        {"id": "buy", "text": "BUY TWO", "conf": .6, "role": "cta",
         "box": {"x": 58, "y": 292, "w": 66, "h": 67}},
        {"id": "free", "text": "FREE", "conf": .55, "role": "offer",
         "box": {"x": 204, "y": 222, "w": 58, "h": 23}},
    ]}
    cfg = {"scene": {"facts": {
        "photo_of_handwriting": True,
        "text_on_photographic_surfaces_only": True,
        "flat_background_fraction": 0.31,
        "photo_coverage": 0.69,
    }}}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    for cid in ("c_buy", "c_free"):
        assert m[cid].get("kept_in_photo") is True, cid
        assert m[cid]["target"] == "drop", cid
        assert m[cid]["meta"]["suppression_reason"] == "text-on-photographic-surface-only", cid


def test_021_geometric_photo_of_text_without_special_case_id():
    """Full-bleed photo + all OCR inside + no flat plate/backplates → bake (021-like).

    Tiny person/logo chips inside the photo must not disable photographic-scene mode
    (real 021 fused_elements has several person fragments + a 16px logo).
    """
    canvas = {"w": 338, "h": 600}
    elements = [
        {"id": "E000", "box": {"x": 0, "y": 0, "w": 338, "h": 600},
         "kind": "photo-fragment", "area": 202800, "coverage": 1.0, "role": "photo"},
        {"id": "E001", "box": {"x": 14, "y": 0, "w": 145, "h": 140},
         "kind": "photo-fragment", "area": 20300, "coverage": .1, "role": "person"},
        {"id": "E005", "box": {"x": 123, "y": 183, "w": 16, "h": 7},
         "kind": "icon", "area": 112, "coverage": .001, "role": "logo"},
    ]
    ocr = {"lines": [
        {"id": "buy", "text": "BUY TWO", "conf": .6, "role": "headline",
         "box": {"x": 58, "y": 292, "w": 66, "h": 67}},
        {"id": "dj", "text": "Pioneer DJ", "conf": .5, "role": "caption",
         "box": {"x": 198, "y": 137, "w": 37, "h": 7}},
    ]}
    cfg = {"scene": {"facts": {
        "flat_background_fraction": 0.31,
        "photo_coverage": 0.69,
        "text_backplate_count": 0,
    }}}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    for cid in ("c_buy", "c_dj"):
        assert m[cid].get("kept_in_photo") is True, cid
        assert m[cid]["meta"]["suppression_reason"] == "text-on-photographic-surface-only", cid


def test_vector_chrome_button_shell_stays_shape_not_fake_text_plate():
    """Outlined CTA rect routes as button/shape; label stays native TEXT."""
    elements = [
        _can_007(),
        {"id": "E_cta", "box": {"x": 80, "y": 320, "w": 320, "h": 70},
         "kind": "shape", "role": "button", "area": 22000, "coverage": .01},
    ]
    ocr = {"lines": [
        {"id": "cta", "text": "Schrijf je nu in, mis geen enkele update.", "conf": .9, "role": "cta",
         "box": {"x": 90, "y": 330, "w": 300, "h": 50}},
    ]}
    m = _by_id(merge_layers.merge(ocr, elements, [], _C007, {
        "scene": {"facts": {"flat_background_fraction": 0.55}},
    }))
    assert m["c_cta"]["target"] == "text"
    assert not m["c_cta"].get("kept_in_photo")
    assert m["c_E_cta"]["target"] in {"shape", "icon"}
    assert m["c_E_cta"]["meta"].get("role") in {"button", "shape", "badge"}

def test_biomel_stroke_outline_pill_shell_plus_text_not_guide_dropped():
    """Biomel outlined benefit pill: sparse perimeter ink + inset copy → callout shell + TEXT.

    Must not be mis-routed as a layout guide or left as empty chrome.
    """
    canvas = {"w": 1080, "h": 1350}
    # Hollow outline pill: area << box (fill_frac ~0.18).
    pill_box = {"x": 40, "y": 420, "w": 280, "h": 56}
    elements = [
        {"id": "E_pill", "box": pill_box, "kind": "shape", "role": "shape",
         "area": int(280 * 56 * 0.18), "coverage": 0.002, "score": 0.7,
         "source": "sam3"},
        {"id": "PROD", "box": {"x": 400, "y": 500, "w": 320, "h": 520},
         "kind": "photo-fragment", "role": "product", "area": 140000,
         "coverage": 0.07, "score": 0.95},
    ]
    ocr = {"lines": [{
        "id": "digest", "text": "Daily digestive support", "conf": 0.94, "role": "body",
        "box": {"x": 55, "y": 430, "w": 250, "h": 36},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    shell = m["c_E_pill"]
    text = m["c_digest"]
    assert shell["target"] == "shape"
    assert shell["meta"].get("suppression_reason") != "guide_artifact"
    assert shell["meta"].get("text_bearing_shell") is True
    assert shell["meta"].get("plate_shell") is True
    assert shell["meta"].get("stroke_outline_shell") is True
    assert shell["meta"].get("role") == "callout"
    assert text["target"] == "text"
    assert not text.get("kept_in_photo")
    assert text["meta"].get("shell_text_host") == "c_E_pill"
    assert text["meta"].get("role") == "callout"


def test_biomel_scalloped_save_badge_promotes_like_seal():
    """Scalloped SAVE / price badge: near-square sparse chrome + offer copy → seal/badge shell."""
    canvas = {"w": 1080, "h": 1350}
    elements = [{
        "id": "E_save", "box": {"x": 820, "y": 180, "w": 200, "h": 200},
        "kind": "icon", "role": "shape", "area": int(200 * 200 * 0.28),
        "coverage": 0.03, "score": 0.85, "source": "sam3",
    }]
    ocr = {"lines": [{
        "id": "save", "text": "SAVE 10%", "conf": 0.95, "role": "offer",
        "box": {"x": 850, "y": 250, "w": 140, "h": 60},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    assert m["c_save"]["target"] == "text"
    assert m["c_save"]["meta"].get("shell_text_host") == "c_E_save"
    assert m["c_E_save"]["target"] == "shape"
    assert m["c_E_save"]["meta"].get("text_bearing_shell") is True
    assert m["c_E_save"]["meta"].get("role") in {"seal", "badge", "starburst", "price_burst"}


def test_biomel_two_products_plus_vs_not_merged():
    """VS comparison: coffee + bag stay two product cutouts; VS chip stays editable."""
    canvas = {"w": 1080, "h": 1080}
    elements = [
        {"id": "P_coffee", "box": {"x": 80, "y": 280, "w": 360, "h": 520},
         "kind": "photo-fragment", "role": "product", "area": 160000,
         "coverage": 0.14, "score": 0.94},
        {"id": "P_bag", "box": {"x": 640, "y": 260, "w": 360, "h": 540},
         "kind": "photo-fragment", "role": "product", "area": 170000,
         "coverage": 0.15, "score": 0.93},
        {"id": "VS", "box": {"x": 500, "y": 480, "w": 80, "h": 80},
         "kind": "shape", "role": "badge", "area": 5000, "coverage": 0.004, "score": 0.8},
    ]
    ocr = {"lines": [
        {"id": "vs", "text": "VS", "conf": 0.97, "role": "label",
         "box": {"x": 515, "y": 500, "w": 50, "h": 40}},
        {"id": "left_price", "text": "£3.20", "conf": 0.94, "role": "offer",
         "box": {"x": 160, "y": 200, "w": 120, "h": 48}},
        {"id": "right_price", "text": "£1.00/day", "conf": 0.93, "role": "offer",
         "box": {"x": 720, "y": 200, "w": 160, "h": 48}},
    ]}
    cfg = {"scene": {"archetype": "comparison_grid", "facts": {"before_after_labels": True}}}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_P_coffee"]["target"] == "image"
    assert m["c_P_bag"]["target"] == "image"
    assert m["c_P_coffee"]["meta"].get("suppression_reason") != "near-duplicate-nested-shell"
    assert m["c_P_bag"]["meta"].get("suppression_reason") != "near-duplicate-nested-shell"
    assert m["c_vs"]["target"] == "text"
    assert m["c_VS"]["target"] == "shape"
    assert m["c_VS"]["meta"].get("text_bearing_shell") is True


def test_biomel_callout_leader_near_outline_pill_kept():
    """Leader stroke beside an outline pill must survive guide-drop (014 + Biomel)."""
    canvas = {"w": 1080, "h": 1350}
    elements = [
        {"id": "E_pill", "box": {"x": 40, "y": 420, "w": 280, "h": 56},
         "kind": "shape", "role": "shape", "area": int(280 * 56 * 0.18),
         "coverage": 0.002, "score": 0.7},
        {"id": "A1", "box": {"x": 310, "y": 430, "w": 90, "h": 28},
         "kind": "shape", "role": "shape", "area": 380, "coverage": 0.0002, "score": 0.7},
        {"id": "PROD", "box": {"x": 480, "y": 500, "w": 300, "h": 480},
         "kind": "photo-fragment", "role": "product", "area": 120000,
         "coverage": 0.08, "score": 0.95},
    ]
    ocr = {"lines": [{
        "id": "digest", "text": "Daily digestive support", "conf": 0.94, "role": "body",
        "box": {"x": 55, "y": 430, "w": 250, "h": 36},
    }]}
    cfg = {"scene": {"preset": {"grouping": {"preserve_callout_leaders": True}},
                    "facts": {"leader_lines": True}}}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, cfg))
    assert m["c_E_pill"]["target"] == "shape"
    assert m["c_E_pill"]["meta"].get("stroke_outline_shell") is True
    assert m["c_A1"]["target"] != "drop"
    assert m["c_A1"]["meta"].get("callout_leader") or m["c_A1"]["meta"].get("role") in {
        "arrow", "callout_leader",
    }


# ── MONTE / Wavy / Biomel product+copy DTC lock ───────────────────────────────────────

_C_DTC = {"w": 1080, "h": 1350}


def test_monte_black_plate_left_text_right_product_on_pack_baked():
    """MONTE black: left marketing stack editable; on-tube label kept_in_photo."""
    elements = [
        {"id": "TUBE", "box": {"x": 520, "y": 280, "w": 420, "h": 900},
         "kind": "photo-fragment", "role": "product", "area": 280000,
         "coverage": 0.19, "score": 0.96},
    ]
    ocr = {"lines": [
        {"id": "brand", "text": "MONTE", "conf": 0.97, "role": "headline",
         "box": {"x": 70, "y": 160, "w": 320, "h": 90}},
        {"id": "sub", "text": "Fuel your day", "conf": 0.92, "role": "subheadline",
         "box": {"x": 70, "y": 270, "w": 300, "h": 48}},
        {"id": "onpack", "text": "25g PROTEIN", "conf": 0.88, "role": "offer",
         "box": {"x": 600, "y": 520, "w": 220, "h": 60}},
    ]}
    cfg = {"scene": {
        "archetype": "product_on_flat",
        "facts": {"flat_background_fraction": 0.78, "photo_coverage": 0.22,
                  "dark_background": True},
    }}
    m = _by_id(merge_layers.merge(ocr, elements, [], _C_DTC, cfg))
    assert m["c_brand"]["target"] == "text"
    assert not m["c_brand"].get("kept_in_photo")
    assert not m["c_brand"]["meta"].get("wordmark")
    assert m["c_brand"]["meta"].get("overlay_text") is True
    assert m["c_sub"]["target"] == "text"
    assert not m["c_sub"].get("kept_in_photo")
    assert m["c_onpack"].get("kept_in_photo") is True
    assert m["c_onpack"]["target"] == "drop"
    assert m["c_onpack"]["meta"].get("baked_owner_id") == "c_TUBE"
    assert m["c_TUBE"]["target"] == "image"


def test_wavy_cream_tube_wordmark_baked_brand_overlay_editable():
    """Wavy cream: decorative on-tube script bakes; flat display brand stays TEXT."""
    elements = [
        {"id": "TUBE", "box": {"x": 480, "y": 360, "w": 300, "h": 760},
         "kind": "photo-fragment", "role": "product", "area": 180000,
         "coverage": 0.12, "score": 0.95},
    ]
    ocr = {"lines": [
        {"id": "brand", "text": "WAVY", "conf": 0.96, "role": "headline",
         "box": {"x": 80, "y": 100, "w": 280, "h": 80}},
        {"id": "script", "text": "wavy", "conf": 0.9, "role": "logo",
         "box": {"x": 520, "y": 620, "w": 200, "h": 90},
         "scene_text_role": "wordmark"},
    ]}
    cfg = {"scene": {
        "archetype": "product_on_flat",
        "facts": {"flat_background_fraction": 0.74, "photo_coverage": 0.26},
    }}
    m = _by_id(merge_layers.merge(ocr, elements, [], _C_DTC, cfg))
    assert m["c_brand"]["target"] == "text"
    assert not m["c_brand"].get("kept_in_photo")
    assert not m["c_brand"]["meta"].get("wordmark")
    assert m["c_script"].get("kept_in_photo") is True
    assert m["c_script"]["target"] == "drop"
    assert m["c_script"]["meta"].get("baked_owner_id") == "c_TUBE"


def test_biomel_outlined_pill_stroke_only_meta_without_area_still_shells():
    """Stroke-only hollow pill (explicit stroke, no fill) shells even without area frac."""
    canvas = {"w": 1080, "h": 1350}
    pill_box = {"x": 40, "y": 420, "w": 280, "h": 56}
    elements = [
        {"id": "E_pill", "box": pill_box, "kind": "shape", "role": "shape",
         "score": 0.7, "source": "sam3", "stroke": {"color": "#111111", "width": 2},
         "fill": None},
        {"id": "PROD", "box": {"x": 400, "y": 500, "w": 320, "h": 520},
         "kind": "photo-fragment", "role": "product", "area": 140000,
         "coverage": 0.07, "score": 0.95},
    ]
    ocr = {"lines": [{
        "id": "digest", "text": "Daily digestive support", "conf": 0.94, "role": "body",
        "box": {"x": 55, "y": 430, "w": 250, "h": 36},
    }]}
    m = _by_id(merge_layers.merge(ocr, elements, [], canvas, {}))
    shell = m["c_E_pill"]
    assert shell["target"] == "shape"
    assert shell["meta"].get("text_bearing_shell") is True
    assert shell["meta"].get("stroke_outline_shell") or shell["meta"].get("stroke_only")
    assert m["c_digest"]["target"] == "text"
    assert not m["c_digest"].get("kept_in_photo")
