"""CPU tests for the Codia-construction-CONTRACT QA/reward priority.

These cover the "score the contract first, SSIM second" rewrite:
  * scripts.codia_parity.score_construction — template-free rule-based construction score;
  * src.pixel_diff._contract_summary — qa.json contract block + pass/fail;
  * src.qa_reward — construction component dominates the metric ladder;
  * benchmark.contract_verdict — the --contract per-run verdict.

The calibration invariant (docs/CODIA-PARITY-SPEC.md): a Codia-shaped output (100% native
text, clean plate, decent placement) PASSES the contract even at a modest SSIM, while a
high-SSIM run with baked/sliced text FAILS on native_text_ratio.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import benchmark
from scripts import codia_parity
from src import pixel_diff, qa_reward


# ── synthetic designs ─────────────────────────────────────────────────────────────────

def _text(node_id, text, x, y, w, h, size=35, weight=400, family="Inter", spacing=0.0):
    return {"id": node_id, "type": "text", "box": {"x": x, "y": y, "w": w, "h": h},
            "text": text, "style": {"fontFamily": family, "fontWeight": weight,
                                     "fontSize": size, "letterSpacing": spacing}}


def codia_shaped_design():
    """Every string native Inter TEXT, emoji as an image cutout, flat solid plate."""
    return {"canvas": {"w": 1000, "h": 1000}, "layers": [
        {"id": "bg", "type": "shape", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "shape_kind": "rect", "fill": {"kind": "flat", "color": "#000000"}},
        _text("t1", "LAATSTE SALE VAN 2026", 48, 318, 651, 47, size=37),
        {"id": "emoji", "type": "image", "box": {"x": 711, "y": 322, "w": 26, "h": 38},
         "src": "assets/emoji.png", "meta": {"emoji": True}},
        _text("t2", "121K", 430, 921, 82, 44, weight=700),
        _text("t3", "weergaven", 520, 923, 182, 46, weight=300),
    ]}


def baked_text_design():
    """The 025/104 failure mode: headlines sliced to pixels, tracking noise, wrong family."""
    return {"canvas": {"w": 1000, "h": 1000}, "layers": [
        {"id": "plate", "type": "image", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "src": "assets/plate.png"},
        {"id": "s1", "type": "image", "box": {"x": 48, "y": 318, "w": 651, "h": 47},
         "name": "LAATSTE SALE VAN 2026 — raster slice", "meta": {"raster_slice": True}},
        {"id": "s2", "type": "image", "box": {"x": 20, "y": 920, "w": 682, "h": 46},
         "name": "121K weergaven — raster slice", "meta": {"raster_slice": True}},
        _text("t2", "Volgend", 860, 145, 143, 44, family="Caladea", spacing=-1.8),
    ]}


# ── scripts.codia_parity.score_construction (template-free) ────────────────────────────

def test_score_construction_needs_no_template_and_rewards_native_text():
    good = codia_parity.score_construction(codia_shaped_design(), archetype="social_screenshot")
    bad = codia_parity.score_construction(baked_text_design(), archetype="social_screenshot")
    assert good["score"] > bad["score"] + 30
    assert good["scores"]["native_text_ratio"] == 1.0
    assert good["scores"]["font_policy"] == 1.0
    assert good["scores"]["emoji_as_image"] == 1.0
    # Baked text: two of three readable lines are raster slices, wrong family, so native
    # text and font policy both crater.
    assert bad["scores"]["native_text_ratio"] < 0.5
    assert bad["scores"]["font_policy"] < 0.5


def test_score_construction_metrics_native_text_ratio_override_wins():
    # The OCR-accurate ratio from the metrics layer overrides the design-internal proxy.
    report = codia_parity.score_construction(codia_shaped_design(), native_text_ratio=0.25)
    assert report["scores"]["native_text_ratio"] == 0.25
    assert report["detail"]["native_text_ratio"]["source"] == "metrics"


def test_score_construction_penalizes_emoji_as_glyph_and_bloat():
    design = codia_shaped_design()
    # Emoji baked onto a text node instead of an image cutout.
    design["layers"][2] = _text("emoji", "party 🎉", 711, 322, 120, 38)
    report = codia_parity.score_construction(design, archetype="social_screenshot")
    assert report["scores"]["emoji_as_image"] == 0.0

    # Node bloat against a simple-scene budget.
    fat = {"canvas": {"w": 1000, "h": 1000},
           "layers": [_text(f"t{i}", f"line {i}", 0, i * 10, 100, 20) for i in range(60)]}
    assert codia_parity.score_construction(fat, complexity="simple")["scores"]["node_budget"] < 0.5


def test_score_construction_flags_mixed_weight_line():
    design = codia_shaped_design()
    design["layers"].append({
        "id": "mixed", "type": "text", "box": {"x": 0, "y": 800, "w": 300, "h": 40},
        "text": "121K weergaven",
        "text_runs": [{"style": {"fontWeight": 700}}, {"style": {"fontWeight": 300}}]})
    report = codia_parity.score_construction(design)
    assert report["scores"]["weight_split"] < 1.0


# ── src.pixel_diff contract summary (pure helpers, no GPU) ─────────────────────────────

def test_placement_ink_iou_averages_text_rows_only():
    rows = [{"type": "text", "ink_iou": 0.8}, {"type": "text", "ink_iou": 0.6},
            {"type": "image", "ink_iou": 0.1}, {"type": "text", "region_ssim": 0.9}]
    assert pixel_diff._placement_ink_iou(rows) == pytest.approx(0.7)
    assert pixel_diff._placement_ink_iou([{"type": "image"}]) is None


def _structure(native_text_ratio, hard_fails=()):
    return {"native_text_ratio": native_text_ratio, "editable_text_recall": native_text_ratio,
            "hard_fails": list(hard_fails)}


def test_contract_summary_passes_codia_shape_at_modest_ssim():
    # native 1.0, clean plate, decent placement, MODEST ssim 0.6 -> contract PASS.
    per_layer = [{"type": "text", "ink_iou": 0.7}]
    summary = pixel_diff._contract_summary(
        codia_shaped_design(), _structure(1.0), None, per_layer, 0.60,
        pixel_diff.DEFAULT_THRESHOLDS, archetype="social_screenshot")
    assert summary["pass"] is True
    assert summary["native_text_ok"] is True
    assert summary["glyph_residue_clean"] is True
    assert summary["ssim_floor_ok"] is True
    assert summary["construction"]["scores"]["native_text_ratio"] == 1.0


def test_contract_summary_fails_high_ssim_baked_text():
    # native 0.25 (025-class), HIGH ssim 0.98 -> contract FAILS on native text.
    summary = pixel_diff._contract_summary(
        baked_text_design(), _structure(0.25), None, [{"type": "text", "ink_iou": 0.7}],
        0.98, pixel_diff.DEFAULT_THRESHOLDS, archetype="comparison_grid")
    assert summary["pass"] is False
    assert summary["native_text_ok"] is False
    # SSIM is high, so it is NOT the reason for failure — the contract leads with native text.
    assert summary["ssim_floor_ok"] is True


def test_contract_summary_fails_on_glyph_residue_even_with_native_text():
    residue = _structure(1.0, hard_fails=[{"rule": "glyph-residue", "detail": "c_B1"}])
    summary = pixel_diff._contract_summary(
        codia_shaped_design(), residue, None, [{"type": "text", "ink_iou": 0.9}], 0.95,
        pixel_diff.DEFAULT_THRESHOLDS)
    assert summary["pass"] is False
    assert summary["glyph_residue_clean"] is False


# ── src.qa_reward: construction dominates the ladder ──────────────────────────────────

def test_construction_component_prefers_contract_score_then_native_ratio():
    assert qa_reward.construction_component(
        {"contract": {"contract_score": 0.9, "native_text_ratio": 1.0}})["score"] == 0.9
    fallback = qa_reward.construction_component({"native_text_ratio": 0.4})
    assert fallback["score"] == 0.4 and fallback["source"] == "native_text_ratio"
    assert qa_reward.construction_component({}) is None


def test_reward_scores_codia_shape_above_high_ssim_baked_text(tmp_path):
    baked = {"ssim": 0.98, "text_recall": 0.9,
             "contract": {"contract_score": 0.44, "native_text_ratio": 0.25, "pass": False}}
    codia = {"ssim": 0.60, "text_recall": 0.95,
             "contract": {"contract_score": 0.93, "native_text_ratio": 1.0, "pass": True}}
    rb = qa_reward.compute_reward(str(tmp_path), {}, qa=baked)
    rc = qa_reward.compute_reward(str(tmp_path), {}, qa=codia)
    # Despite far higher SSIM, the baked-text run scores LOWER: the contract leads.
    assert rc["score"] > rb["score"]
    assert rc["components"]["construction"]["score"] == 0.93
    assert qa_reward.reward_evidence(rc)["construction"] == 0.93


def test_reward_without_contract_is_unchanged(tmp_path):
    # Runs predating the contract fields use the legacy 3-key ladder untouched.
    reward = qa_reward.compute_reward(str(tmp_path), {}, qa={"ssim": 0.5})
    assert reward["score"] == pytest.approx(0.5, abs=1e-6)
    assert reward["components"]["construction"] is None


def test_critique_prompt_leads_with_editability():
    prompt = qa_reward._CRITIQUE_PROMPT.lower()
    # Editability / double-print / misplacement asked FIRST; structural words still present.
    assert "editable text" in prompt
    assert prompt.index("not editable") < prompt.index("missing / erased")
    for word in ("double-printed", "misplaced", "rasterized", "ghost", "severity"):
        assert word in prompt, word


# ── benchmark.contract_verdict + --contract summary ───────────────────────────────────

def test_contract_verdict_requires_native_text_clean_plate_placement():
    ok = benchmark.contract_verdict({"id": "x", "native_text_ratio": 0.95,
                                     "glyph_residue_clean": True, "placement_ok": True,
                                     "contract_pass": True})
    assert ok["pass"] is True and ok["reasons"] == []

    low = benchmark.contract_verdict({"id": "025", "native_text_ratio": 0.25,
                                      "glyph_residue_clean": True, "placement_ok": True})
    assert low["pass"] is False and any("native text" in r for r in low["reasons"])

    residue = benchmark.contract_verdict({"id": "101", "native_text_ratio": 1.0,
                                          "glyph_residue_clean": False, "placement_ok": True})
    assert residue["pass"] is False and "unresolved glyph residue" in residue["reasons"]
