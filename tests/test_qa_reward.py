"""Tests for the Phase-2 metric-ladder reward (src/qa_reward.py).

CPU-only. The VLM critique is always mocked (same approach as the other vlm_* stage
tests); LPIPS runs for real on tiny synthetic images and skips when the backbone
cannot be loaded (e.g. offline first run).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, qa_reward


def _write_png(path, color, size=(24, 24)):
    from PIL import Image

    Image.new("RGB", size, color).save(str(path))


# ── config accessors ─────────────────────────────────────────────────────────────────

def test_reward_mode_defaults_to_phase2_with_legacy_fallback():
    assert qa_reward.reward_mode({}) == "phase2"
    assert qa_reward.reward_mode(None) == "phase2"
    assert qa_reward.reward_mode({"runtime": {"harness": {"reward": "legacy"}}}) == "legacy"
    assert qa_reward.reward_mode({"runtime": {"harness": {"reward": "bogus"}}}) == "phase2"


def test_reward_weights_prefer_override_then_archetype_then_dark_default():
    override = {"qa": {"reward": {"weights": {"local_ssim": 1, "lpips": 1, "text": 2}},
                       "reward_weights": {"local_ssim": 9, "lpips": 0, "text": 0}}}
    assert qa_reward.reward_weights(override)["text"] == 2.0

    preset = {"qa": {"reward_weights": {"local_ssim": 0.4, "lpips": 0.15, "text": 0.45}}}
    assert qa_reward.reward_weights(preset)["lpips"] == 0.15

    dark = {"scene": {"facts": {"dark_background": True}}}
    assert qa_reward.reward_weights(dark) == qa_reward._DARK_TEXT_WEIGHTS
    assert qa_reward.reward_weights({}) == qa_reward._DEFAULT_WEIGHTS


def test_gate_thresholds_archetype_wins_over_reward_defaults():
    cfg = {"qa": {"archetype_thresholds": {"lpips_similarity_min": 0.5,
                                           "reward_local_ssim_min": 0.6},
                  "reward": {"lpips_similarity_min": 0.1, "local_ssim_min": 0.1}}}
    floors = qa_reward.gate_thresholds(cfg)
    assert floors == {"lpips_similarity_min": 0.5, "local_ssim_min": 0.6,
                      "worst_local_ssim_min": qa_reward._DEFAULT_WORST_LOCAL_SSIM_MIN}
    defaults = qa_reward.gate_thresholds({})
    assert defaults["lpips_similarity_min"] == qa_reward._DEFAULT_LPIPS_SIMILARITY_MIN
    assert defaults["local_ssim_min"] == qa_reward._DEFAULT_LOCAL_SSIM_MIN


# ── rung 1: per-element local scores (duck-typed pixel_diff rows) ─────────────────────

def test_local_component_consumes_region_rows_and_folds_ink_iou():
    qa = {"per_layer": [
        {"id": "c_B0", "type": "text", "region_ssim": 0.9, "ink_iou": 0.5, "region_px": 400},
        {"id": "c_B1", "type": "shape", "region_ssim": 0.8, "region_px": 400},
        {"id": "c_B2", "type": "image", "score": 0.7},          # legacy row shape
        {"id": "c_B4", "type": "shape", "region_ssim": 1.0, "region_color": 0.0},
        "not-a-dict",
        {"id": "c_B3", "type": "text"},                          # no metrics — skipped
    ]}
    local = qa_reward.local_component(qa)
    assert local["source"] == "per_layer"
    assert local["count"] == 4
    # text row is 0.7*0.9 + 0.3*0.5 = 0.78, so the worst row is the legacy 0.7 one.
    assert local["worst"][0]["id"] == "c_B2"
    # region_color folds into non-text rows (recolour drift): 0.85*1.0 + 0.15*0.0.
    by_id = {row["id"]: row["score"] for row in local["worst"]}
    assert by_id["c_B4"] == pytest.approx(0.85)
    assert 0.0 < local["score"] <= 1.0


def test_local_component_falls_back_to_multiscale_ssim_then_none():
    assert qa_reward.local_component({"ssim": 0.62})["source"] == "multiscale_ssim"
    assert qa_reward.local_component({"ssim": 0.62})["score"] == 0.62
    assert qa_reward.local_component({}) is None
    assert qa_reward.local_component(None) is None
    assert qa_reward.local_component({"per_layer": "garbage", "ssim": 0.5})["score"] == 0.5


def test_text_component_uses_strictest_recall():
    assert qa_reward.text_component({"text_recall": 0.9, "editable_text_recall": 0.4}) == 0.4
    assert qa_reward.text_component({"text_recall": 0.9}) == 0.9
    assert qa_reward.text_component({}) is None


# ── rung 2: LPIPS on tiny synthetic images (real model, CPU) ──────────────────────────

def test_lpips_identical_scores_better_than_different(tmp_path):
    pytest.importorskip("lpips")
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_png(a, (20, 20, 20), size=(32, 32))
    _write_png(b, (235, 235, 235), size=(32, 32))
    try:
        same = qa_reward.lpips_score(str(a), str(a), {})
    except Exception:  # pragma: no cover - defensive; lpips_score itself never raises
        same = None
    if same is None:
        pytest.skip("LPIPS backbone unavailable (offline?)")
    diff = qa_reward.lpips_score(str(a), str(b), {})
    assert same["distance"] == pytest.approx(0.0, abs=1e-4)
    assert diff["distance"] > same["distance"]
    assert diff["similarity"] < same["similarity"]
    assert same["net"] == "squeeze"


def test_lpips_score_graceful_on_missing_files_and_disabled(tmp_path):
    assert qa_reward.lpips_score(None, None, {}) is None
    assert qa_reward.lpips_score(str(tmp_path / "x.png"), str(tmp_path / "y.png"), {}) is None
    a = tmp_path / "a.png"
    _write_png(a, (0, 0, 0))
    disabled = {"qa": {"reward": {"lpips": {"enabled": False}}}}
    assert qa_reward.lpips_score(str(a), str(a), disabled) is None


# ── the ladder: compute_reward + acceptance gate ─────────────────────────────────────

def test_compute_reward_weights_components_and_penalizes_hard_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(qa_reward, "lpips_score",
                        lambda *a, **k: {"distance": 0.2, "similarity": 0.8, "net": "squeeze",
                                         "max_edge": 256})
    (tmp_path / "normalized.png").write_bytes(b"png")
    (tmp_path / "preview.png").write_bytes(b"png")
    qa = {"ssim": 0.6, "text_recall": 0.9, "hard_fails": []}
    cfg = {"qa": {"reward": {"weights": {"local_ssim": 0.4, "lpips": 0.3, "text": 0.3}}}}

    reward = qa_reward.compute_reward(str(tmp_path), cfg, qa=qa)
    expected = (0.4 * 0.6 + 0.3 * 0.8 + 0.3 * 0.9)
    assert reward["score"] == pytest.approx(expected, abs=1e-6)
    assert reward["components"]["lpips"]["similarity"] == 0.8
    assert reward["hard_fails"] == 0

    qa_failed = dict(qa, hard_fails=[{"rule": "missing-assets"}])
    penalized = qa_reward.compute_reward(str(tmp_path), cfg, qa=qa_failed)
    assert penalized["score"] == pytest.approx(expected - 0.12, abs=1e-6)
    assert penalized["hard_fails"] == 1


def test_compute_reward_renormalizes_over_available_components(tmp_path):
    # No files -> no LPIPS; no text fields -> only local_ssim remains.
    reward = qa_reward.compute_reward(str(tmp_path), {}, qa={"ssim": 0.5})
    assert reward["score"] == pytest.approx(0.5, abs=1e-6)
    empty = qa_reward.compute_reward(str(tmp_path), {}, qa={})
    assert empty["score"] is None


def test_compute_reward_never_raises_on_garbage():
    reward = qa_reward.compute_reward("", None, qa={"per_layer": [{"region_ssim": "NaNish"}],
                                                    "ssim": "bad", "hard_fails": "bad"})
    assert reward["score"] is None or isinstance(reward["score"], float)


# ── F12: recalibrated gate floors reject 002-class degradation ────────────────────────

def test_recalibrated_default_gate_rejects_002_class_but_accepts_good_runs():
    """Measured floors: known-BAD 002 = LPIPS-sim 0.732 / local 0.465; good runs
    (009/013/052) = LPIPS >= 0.976 / local >= 0.60. The default floors (0.80 / 0.50) must
    sit in that gap — reject the degraded numbers, accept the good ones."""
    floors = qa_reward.gate_thresholds({})
    assert floors == {"lpips_similarity_min": 0.80, "local_ssim_min": 0.50,
                      "worst_local_ssim_min": qa_reward._DEFAULT_WORST_LOCAL_SSIM_MIN}
    bad = {"components": {"lpips": {"similarity": 0.732}, "local_ssim": {"score": 0.465}}}
    good = {"components": {"lpips": {"similarity": 0.981}, "local_ssim": {"score": 0.601}}}
    assert qa_reward.acceptance_gate("", {}, reward=bad)["ok"] is False
    assert qa_reward.acceptance_gate("", {}, reward=good)["ok"] is True


def test_recalibrated_gate_each_floor_independently_rejects_002():
    lp_only = {"components": {"lpips": {"similarity": 0.73}, "local_ssim": {"score": 0.9}}}
    gate = qa_reward.acceptance_gate("", {}, reward=lp_only)
    assert gate["ok"] is False and gate["checks"]["lpips_similarity"]["ok"] is False
    ls_only = {"components": {"lpips": {"similarity": 0.99}, "local_ssim": {"score": 0.465}}}
    gate = qa_reward.acceptance_gate("", {}, reward=ls_only)
    assert gate["ok"] is False and gate["checks"]["local_ssim"]["ok"] is False


def test_lpips_floor_rejects_degraded_synthetic_accepts_near_match(tmp_path):
    """Real LPIPS on synthetic images: a near-match clears the 0.80 floor; a 002-style
    erasure (half the canvas wiped to flat gray) is rejected by it."""
    pytest.importorskip("lpips")
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    base = np.zeros((128, 128, 3), np.uint8)
    base[:64, :64] = (200, 60, 60); base[:64, 64:] = (60, 160, 90)
    base[64:, :64] = (70, 90, 200); base[64:, 64:] = (230, 200, 40)
    Image.fromarray(base).save(tmp_path / "src.png")
    near = np.clip(base.astype(int) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(near).save(tmp_path / "near.png")
    degraded = base.copy(); degraded[:, 64:] = 128            # right half erased to gray
    Image.fromarray(degraded).save(tmp_path / "deg.png")

    near_lp = qa_reward.lpips_score(str(tmp_path / "src.png"), str(tmp_path / "near.png"), {})
    deg_lp = qa_reward.lpips_score(str(tmp_path / "src.png"), str(tmp_path / "deg.png"), {})
    if near_lp is None or deg_lp is None:
        pytest.skip("LPIPS backbone unavailable (offline?)")
    assert near_lp["similarity"] > deg_lp["similarity"]
    assert qa_reward.acceptance_gate("", {}, reward={"components": {"lpips": near_lp}})["ok"] is True
    assert qa_reward.acceptance_gate("", {}, reward={"components": {"lpips": deg_lp}})["ok"] is False


# ── F12c: content penalty for rasterized text / low native-leaf ───────────────────────

def test_content_penalty_penalizes_rasterized_text_and_low_native_leaf():
    # pixel_diff writes these under qa.structural; read defensively from either place.
    qa = {"structural": {"rasterized_text_ratio": 0.9, "native_leaf_ratio": 0.1}}
    pen = qa_reward.content_penalty(qa)
    assert pen["total"] > 0
    assert "rasterized_text" in pen["terms"] and "low_native_leaf" in pen["terms"]
    # top-level fields work too
    assert qa_reward.content_penalty({"native_leaf_ratio": 0.1})["total"] > 0


def test_content_penalty_zero_when_absent_or_healthy_and_capped():
    assert qa_reward.content_penalty({})["total"] == 0.0
    assert qa_reward.content_penalty(None)["total"] == 0.0
    healthy = {"structural": {"native_leaf_ratio": 0.9, "rasterized_text_ratio": 0.1}}
    assert qa_reward.content_penalty(healthy)["total"] == 0.0
    worst = {"rasterized_text_ratio": 1.0, "native_leaf_ratio": 0.0}
    assert qa_reward.content_penalty(worst)["total"] == qa_reward._CONTENT_PENALTY_CAP


def test_compute_reward_scores_rasterized_run_below_editable_one(tmp_path):
    """F12c: a globally-plausible run whose text is all pixels / leaves non-native must
    score LOWER than an otherwise-identical editable run."""
    editable = {"ssim": 0.9, "structural": {"native_leaf_ratio": 0.9}}
    rasterized = {"ssim": 0.9, "structural": {"native_leaf_ratio": 0.1,
                                              "rasterized_text_ratio": 0.9}}
    good = qa_reward.compute_reward(str(tmp_path), {}, qa=editable)
    bad = qa_reward.compute_reward(str(tmp_path), {}, qa=rasterized)
    assert bad["score"] < good["score"]
    assert bad["content_penalty"]["total"] > 0
    assert good["content_penalty"]["total"] == 0.0
    # penalty surfaces in the compact evidence trail
    assert qa_reward.reward_evidence(bad)["content_penalty"] > 0


def test_acceptance_gate_blocks_low_lpips_and_skips_when_unavailable(tmp_path, monkeypatch):
    cfg = {"qa": {"reward": {"lpips_similarity_min": 0.2, "local_ssim_min": 0.3}}}
    reward = {"components": {"lpips": {"similarity": 0.05}, "local_ssim": {"score": 0.9}}}
    gate = qa_reward.acceptance_gate(str(tmp_path), cfg, reward=reward)
    assert gate["ok"] is False
    assert gate["checks"]["lpips_similarity"]["ok"] is False
    assert gate["checks"]["local_ssim"]["ok"] is True

    # Missing metrics -> the gate never blocks (behaviour unchanged where LPIPS absent).
    empty = qa_reward.acceptance_gate(str(tmp_path), cfg, reward={"components": {}})
    assert empty == {"ok": True, "skipped": "no_metrics"}
    legacy = qa_reward.acceptance_gate(str(tmp_path),
                                       {"runtime": {"harness": {"reward": "legacy"}}})
    assert legacy["ok"] is True and legacy["skipped"] == "legacy"


# ── rung 4: VLM critique (mocked) ─────────────────────────────────────────────────────

def _critique_cfg(**over):
    critique = {"enabled": True}
    critique.update(over)
    return {"qa": {"reward": {"critique": critique}},
            "vlm": {"base_url": "http://127.0.0.1:1234/v1", "model": "google/gemma-4-e4b"}}


def _seed_pair(tmp_path):
    (tmp_path / "normalized.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    (tmp_path / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"1" * 16)


def test_run_critique_parses_items_and_writes_artifact(tmp_path, monkeypatch):
    _seed_pair(tmp_path)
    payload = json.dumps({"critique": [
        {"element": "SHOP NOW", "issue": "text clipped at right edge",
         "suggested_fix": "widen the button text box"},
        {"element": "SHOP NOW", "issue": "text clipped at right edge",
         "suggested_fix": "duplicate entry"},                    # deduped
        {"element": "", "issue": "", "suggested_fix": "no issue"},  # dropped
    ]})
    seen = {}

    def fake_pair(source, render, prompt, **kwargs):
        seen["model"] = kwargs.get("model")
        seen["images"] = (len(source), len(render))
        return payload

    monkeypatch.setattr(qa_reward, "_ask_vlm_pair", fake_pair)
    result = qa_reward.run_critique(str(tmp_path), _critique_cfg())

    assert [item["issue"] for item in result["items"]] == ["text clipped at right edge"]
    assert seen["model"] == "google/gemma-4-e4b"
    assert seen["images"][0] > 0 and seen["images"][1] > 0
    saved = json.loads((tmp_path / "qa_critique.json").read_text(encoding="utf-8"))
    assert saved["items"][0]["element"] == "SHOP NOW"


def test_run_critique_disabled_makes_no_call(tmp_path, monkeypatch):
    _seed_pair(tmp_path)

    def boom(*a, **k):
        raise AssertionError("VLM must not be called when critique is disabled")

    monkeypatch.setattr(qa_reward, "_ask_vlm_pair", boom)
    result = qa_reward.run_critique(str(tmp_path), {})
    assert result["items"] == [] and result.get("skipped") == "disabled"


def test_run_critique_graceful_when_vlm_unavailable_or_artifacts_missing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LM Studio offline / model evicted")

    monkeypatch.setattr(qa_reward, "_ask_vlm_pair", boom)
    missing = qa_reward.run_critique(str(tmp_path), _critique_cfg())
    assert missing["items"] == [] and missing["error"] == "missing_artifacts"

    _seed_pair(tmp_path)
    offline = qa_reward.run_critique(str(tmp_path), _critique_cfg())
    assert offline["items"] == []
    assert "vlm_unavailable" in offline["error"]


def test_run_critique_attaches_layer_ids_from_design(tmp_path, monkeypatch):
    _seed_pair(tmp_path)
    (tmp_path / "design.json").write_text(json.dumps({"layers": [
        {"id": "c_B7", "type": "text", "text": "30% KORTING op alles"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(qa_reward, "_ask_vlm_pair", lambda *a, **k: json.dumps({"critique": [
        {"element": "30% KORTING", "issue": "text clipped", "suggested_fix": "refit"}]}))

    result = qa_reward.run_critique(str(tmp_path), _critique_cfg())
    assert result["items"][0]["layer_ids"] == ["c_B7"]


def test_critique_respects_max_items(tmp_path, monkeypatch):
    _seed_pair(tmp_path)
    many = json.dumps({"critique": [
        {"element": f"E{i}", "issue": f"issue {i} misaligned", "suggested_fix": "move"}
        for i in range(20)
    ]})
    monkeypatch.setattr(qa_reward, "_ask_vlm_pair", lambda *a, **k: many)
    result = qa_reward.run_critique(str(tmp_path), _critique_cfg(max_items=3))
    assert len(result["items"]) == 3


# ── critique → repair-action vocabulary ──────────────────────────────────────────────

def test_critique_to_repairs_maps_onto_existing_actionable_vocabulary():
    items = [
        {"element": "UPFRONT", "issue": "the same headline appears twice (ghosted)",
         "suggested_fix": "remove the duplicate", "layer_ids": ["c_B3", "c_B19"]},
        {"element": "30% KORTING", "issue": "text clipped at the right edge",
         "suggested_fix": "widen the text box", "layer_ids": ["c_B18"]},
        {"element": "ingredienten", "issue": "garbled glyphs, looks like the wrong font",
         "suggested_fix": "match the source typeface"},
        {"element": "CTA button", "issue": "wrong color, much darker than the original",
         "suggested_fix": "recolor to match"},
        {"element": "logo", "issue": "misaligned, shifted to the left",
         "suggested_fix": "reposition"},
        {"element": "star rating", "issue": "element is missing from the reconstruction",
         "suggested_fix": "detect and add it"},
        {"element": "background", "issue": "visible inpaint halo behind the product",
         "suggested_fix": "rebuild the clean plate"},
    ]
    repairs = qa_reward.critique_to_repairs(items)
    pairs = [(r["stage"], r["action"]) for r in repairs]
    assert pairs == [
        ("merge", "dedup"),
        ("text-analysis", "refit-text-box"),
        ("text-analysis", "resolve-fonts"),
        ("text-analysis", "refit-colors-effects"),
        ("layout", "refit-geometry"),
        ("sam3", "rerun-detection"),
        ("inpaint", "rebuild-clean-plate"),
    ]
    # Every mapped repair is actionable and resumes a real pipeline stage.
    for repair in repairs:
        assert harness.is_actionable(repair), repair
        assert harness.resume_stage_for(repair)
    dedup = repairs[0]
    assert dedup["target_id"] == "c_B3"
    assert dedup["params"]["duplicate_text"] == ["UPFRONT"]
    assert dedup["params"]["source"] == "vlm_critique"
    refit = repairs[1]
    assert refit["params"]["clipped_text"] == ["30% KORTING"]
    choice = harness.recommended_resume([refit])
    assert choice["resume"] == "text"
    assert choice["patches"]["text_analysis"]["fit"]["widen_clipped"] is True


def test_critique_to_repairs_skips_unmapped_and_dedupes_signatures():
    items = [
        {"element": "vibe", "issue": "general mood feels different", "suggested_fix": ""},
        {"element": "A", "issue": "text clipped", "suggested_fix": "", "layer_ids": ["x"]},
        {"element": "B", "issue": "clipped again", "suggested_fix": "", "layer_ids": ["x"]},
        "not-a-dict",
        {"issue": ""},
    ]
    repairs = qa_reward.critique_to_repairs(items)
    assert len(repairs) == 1
    assert (repairs[0]["stage"], repairs[0]["action"], repairs[0]["target_id"]) == (
        "text-analysis", "refit-text-box", "x")


def test_critique_missing_text_maps_to_restore_editable_text_not_sam():
    repairs = qa_reward.critique_to_repairs([
        {"element": "subtitle", "issue": "text missing below the headline",
         "suggested_fix": "restore it"}])
    assert (repairs[0]["stage"], repairs[0]["action"]) == (
        "text-analysis", "restore-editable-text")


def test_critique_erased_content_maps_to_rerun_detection_high(monkeypatch):
    """F12b: the 002 class — a sharpened critic reporting an erased product cluster must
    map to sam3 re-detection at high severity, not be dropped as unmapped."""
    for issue in ("the whole product cluster is erased and replaced by a gray smear",
                  "the three jars have vanished from the reconstruction",
                  "the bottle is gone / wiped out"):
        repairs = qa_reward.critique_to_repairs([
            {"element": "products", "issue": issue, "suggested_fix": "detect them again"}])
        assert repairs, issue
        assert (repairs[0]["stage"], repairs[0]["action"]) == ("sam3", "rerun-detection")
        assert repairs[0]["severity"] == "high"


def test_critique_rasterized_headline_maps_to_restore_editable_text():
    repairs = qa_reward.critique_to_repairs([
        {"element": "LAATSTE SALE", "issue": "the headline is rasterized / baked into an image",
         "suggested_fix": "make it editable text"}])
    assert (repairs[0]["stage"], repairs[0]["action"]) == ("text-analysis", "restore-editable-text")
    assert repairs[0]["severity"] == "high"


def test_critique_severity_escalates_but_never_downgrades():
    # a stray "low" from the VLM cannot soften a structural rule's high floor
    low_call = qa_reward.critique_to_repairs([
        {"element": "x", "issue": "duplicated text appears twice", "severity": "low",
         "suggested_fix": "dedup"}])
    assert low_call[0]["severity"] == "high"        # merge/dedup rule floor wins
    # a "high" from the VLM escalates a rule that defaults to medium (color)
    high_call = qa_reward.critique_to_repairs([
        {"element": "cta", "issue": "wrong color, too dark", "severity": "high",
         "suggested_fix": "recolor"}])
    assert high_call[0]["severity"] == "high"


def test_critique_prompt_and_schema_target_structural_failures():
    # F12b: the prompt must steer the model to report missing/erased/ghost/rasterized
    # first, and the schema must carry a severity field so those map to high-sev repairs.
    prompt = qa_reward._CRITIQUE_PROMPT.lower()
    for word in ("missing", "erased", "ghost", "rasterized", "severity"):
        assert word in prompt, word
    item_schema = qa_reward._CRITIQUE_SCHEMA["properties"]["critique"]["items"]
    assert "severity" in item_schema["properties"]
    assert "severity" in item_schema["required"]


def test_parse_critique_tolerates_missing_severity():
    # live payloads / older mocks omit severity; parsing must not drop those items.
    items = qa_reward._parse_critique(
        '{"critique":[{"element":"a","issue":"text clipped","suggested_fix":"widen"}]}', 8)
    assert items and items[0]["issue"] == "text clipped"
    assert "severity" not in items[0]


def test_reward_evidence_is_compact():
    reward = {"mode": "phase2", "score": 0.71, "hard_fails": 1, "components": {
        "local_ssim": {"score": 0.6, "mean": 0.7, "worst": []},
        "lpips": {"similarity": 0.8, "distance": 0.2},
        "text": 0.9,
    }}
    evidence = qa_reward.reward_evidence(reward)
    assert evidence == {"score": 0.71, "mode": "phase2", "local_ssim": 0.6,
                        "lpips_similarity": 0.8, "text": 0.9, "hard_fails": 1}
    assert qa_reward.reward_evidence(None) is None
