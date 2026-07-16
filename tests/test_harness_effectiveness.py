"""Replay of runs/postfix-benchmark-4 through the repaired harness effectiveness fixes.

postfix-benchmark-4 evidence: auto_fixed_runs was 0 across 16 fixtures while ~70
refit-text-box and ~36 inspect-worst-regions repairs ran — every attempt logged
metric_deltas of exactly 0.0. Root causes pinned by these tests:

  A  repairs whose config patch writes keys NO pipeline stage reads (text_analysis.fit,
     design.restore_native_nodes, design.rebuild_schema, reconstruct.restage_assets)
     are guaranteed byte-identical reruns -> now not actionable / rejected at admission
  B  box-only worst-window focus_regions entries (no layer_id) are ignored by
     reconstruct.apply_raster_slice_fallback -> repair.assess now resolves the window
     to the measured per-layer rows under it so the patch carries real layer ids
  C  raster-slice suggestions for layers fallback.json already dropped/refused
     ("plate-already-holds-source-pixels", the 013 loop) are terminal -> excluded
  D  GB6: the loop must never resume earlier than the first stage a patch can affect
  E  a rerun that leaves the watched stage outputs byte-identical short-circuits the
     rest of the round (no fixer pass, no follow-up pipeline rerun)
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, harness_loop, repair

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "runs", "postfix-benchmark-4")
RUN_013 = os.path.join(BENCH, "013_attached_a32b069cce97685c")

needs_bench = pytest.mark.skipif(
    not os.path.isdir(RUN_013), reason="postfix-benchmark-4 artifacts not present")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ── A: unreachable config patches are not actionable ────────────────────────────────

def test_unreachable_patch_actions_are_not_actionable():
    # Each of these writes ONLY config keys no pipeline stage reads (verified against
    # stage code); re-running the pipeline with them is a guaranteed no-op.
    for stage, action in [
        ("text-analysis", "refit-text-box"),   # text_analysis.fit — no consumer
        ("design", "restore-native-nodes"),    # design.restore_native_nodes — no consumer
        ("design", "rebuild-schema"),          # design.rebuild_schema — no consumer
        ("reconstruct", "restage-assets"),     # reconstruct.restage_assets — no consumer
    ]:
        repair_record = {"stage": stage, "action": action,
                         "params": {}, "severity": "high"}
        assert harness.is_actionable(repair_record) is False, (stage, action)


def test_reachable_patch_actions_stay_actionable():
    for stage, action, params in [
        ("text-analysis", "resolve-fonts", {}),
        ("merge", "dedup", {"raise_dedup_iou": True}),
        ("sam3", "rerun-detection", {"lower_confidence": True}),
        ("vectorize", "raster-fallback", {}),
        ("ocr", "rerun", {"upscale": True}),
        ("figma", "restage-inbox", {}),
        # untargeted whole-stage rerun with an empty patch stays admissible when it
        # resumes AFTER the peel stack (merge index > peel index)
        ("merge", "enforce-single-owner", {}),
    ]:
        repair_record = {"stage": stage, "action": action,
                         "params": params, "severity": "high"}
        assert harness.is_actionable(repair_record) is True, (stage, action)


def test_empty_patch_rerun_refused_when_it_would_replay_peel():
    # A config-identical rerun resuming at "text" replays residual→qwen→sam→elements→
    # peel (091: 12 Flux inpaints) for a near-certain no-op. Refused pre-peel; the
    # action stays a valid human-review suggestion.
    early = {"stage": "text-analysis", "action": "restore-editable-text", "params": {}}
    assert harness.resume_stage_for(early) == "text"
    assert harness.is_actionable(early) is False
    late = {"stage": "merge", "action": "enforce-single-owner", "params": {}}
    assert harness.is_actionable(late) is True


def test_box_only_focus_region_is_not_actionable():
    # reconstruct.apply_raster_slice_fallback only honors layer_id entries: a
    # box-only worst-window region (the 002/016 no-ops) cannot force anything.
    box_only = {"stage": "reconstruct", "action": "inspect-worst-regions",
                "params": {"regions": [{"box": {"x": 128, "y": 320, "w": 64, "h": 64}}]}}
    assert harness.is_actionable(box_only) is False
    targeted = {"stage": "reconstruct", "action": "inspect-worst-regions",
                "params": {"regions": [{"layer_id": "c_E1", "box": {}}]}}
    assert harness.is_actionable(targeted) is True


# ── B: worst-window regions resolve to measured layers ──────────────────────────────

def test_worst_window_repair_resolves_overlapping_layer_ids():
    qa = {
        "hard_fails": [{"rule": "local-ssim-worst-window",
                        "detail": "worst local window ssim 0.015 < 0.100 at x=128 y=320",
                        "bbox": {"x": 128, "y": 320, "w": 64, "h": 64}}],
        "local_ssim_worst_window": {"ssim": 0.015,
                                    "bbox": {"x": 128, "y": 320, "w": 64, "h": 64}},
        "per_layer": [
            {"id": "c_far", "region_ssim": 0.9,
             "abs_box": {"x": 600, "y": 600, "w": 50, "h": 50}},
            {"id": "c_bad", "region_ssim": 0.10,
             "abs_box": {"x": 120, "y": 300, "w": 100, "h": 100}},
            {"id": "c_ok", "region_ssim": 0.70,
             "abs_box": {"x": 140, "y": 330, "w": 30, "h": 30}},
        ],
    }
    repairs = repair.assess({}, qa, {"lines": []}, {})
    windows = [r for r in repairs
               if (r["stage"], r["action"]) == ("reconstruct", "inspect-worst-regions")
               and (r.get("params") or {}).get("worst_window")]
    assert windows, "worst-window hard fail must map to an inspect repair"
    record = windows[0]
    layer_ids = [entry.get("layer_id") for entry in record["params"]["regions"]
                 if entry.get("layer_id")]
    # worst overlapping layer first, non-overlapping layer excluded
    assert layer_ids[0] == "c_bad"
    assert "c_ok" in layer_ids and "c_far" not in layer_ids
    assert record.get("target_id") == "c_bad"
    # and the resulting plan is admissible: the patch reaches the pipeline
    assert harness.is_actionable(record) is True
    patches = harness.config_patches_for(record)
    assert harness.patch_reaches_pipeline(patches) is True


# ── C: terminal fallback dispositions are not re-suggested (013 replay) ─────────────

@needs_bench
def test_013_dropped_plate_layers_not_resuggested_for_slicing(tmp_path):
    qa = _load(os.path.join(RUN_013, "qa.json"))
    run_dir = tmp_path / "run013"
    run_dir.mkdir()
    fallback = _load(os.path.join(RUN_013, "fallback.json"))
    dropped_ids = {str(e["id"]) for e in fallback.get("dropped") or []}
    assert {"c_E006__hostbg", "c_E007__hostbg"} <= dropped_ids  # the observed loop
    (run_dir / "fallback.json").write_text(json.dumps(fallback), encoding="utf-8")

    repairs = repair.assess({}, qa, {"lines": []}, {"run_dir": str(run_dir)})
    for record in repairs:
        if (record.get("stage"), record.get("action")) != ("reconstruct",
                                                           "inspect-worst-regions"):
            continue
        if not (record.get("params") or {}).get("raster_slice"):
            continue
        suggested = {str(entry.get("layer_id"))
                     for entry in record["params"].get("regions") or []}
        assert not (suggested & dropped_ids), (
            "slicer already dropped these layers; re-forcing them is the 013 no-op loop")


# ── D: GB6 — never resume earlier than the patch can act ────────────────────────────

def test_earliest_patched_stage_orders_sections():
    assert harness.earliest_patched_stage({"design": {"x": 1}}) is None  # no lever
    assert harness.earliest_patched_stage({"figma": {"enabled": True}}) == "figma"
    assert harness.earliest_patched_stage({"vlm": {"font_judge": {"enabled": True}}}) == "text"
    assert harness.earliest_patched_stage(
        {"layout": {"min_container_frac": 0.001},
         "vectorize": {"force_raster_fallback": True}}) == "reconstruct"


def test_recommended_resume_never_precedes_patched_stage():
    repairs = [{"stage": "vectorize", "action": "raster-fallback",
                "reason": "x", "severity": "high"}]
    choice = harness.recommended_resume(repairs)
    assert choice is not None
    patched = harness.earliest_patched_stage(choice["patches"])
    order = harness.PIPELINE_STAGE_ORDER
    assert order.index(choice["resume"]) >= order.index(patched)


# ── E: identical-artifact rounds short-circuit ───────────────────────────────────────

def _seed_run(tmp_path, repairs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps(repairs), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": False, "ssim": 0.8, "hard_fails": [
            {"rule": "visual", "detail": "ssim low"}], "repairs": repairs}),
        encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps({"layers": []}), encoding="utf-8")
    (run_dir / "preview.png").write_bytes(b"prev")
    (run_dir / "merged.json").write_text("{}", encoding="utf-8")
    return run_dir, input_path


def test_execute_repairs_flags_identical_artifact_attempts(tmp_path):
    repairs = [{"stage": "merge", "action": "dedup", "reason": "overlap",
                "params": {"raise_dedup_iou": True}, "severity": "high"}]
    run_dir, _ = _seed_run(tmp_path, repairs)

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        # rewrites qa.json with DIFFERENT bytes but identical metrics; stage outputs
        # (merged.json) and the design/preview tail untouched — the observed
        # "metric_deltas all 0.0" class.
        qa_path = os.path.join(rd, "qa.json")
        qa = json.loads(open(qa_path, encoding="utf-8").read())
        qa["timestamp"] = qa.get("timestamp", 0) + 1
        with open(qa_path, "w", encoding="utf-8") as handle:
            json.dump(qa, handle)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=1,
                                      run_one=fake_run_one)
    executed = [a for a in summary["attempts"]
                if not a.get("admission_skipped") and not a.get("admission_rejected")]
    assert executed and executed[0]["artifacts_changed"] is False
    assert executed[0]["no_effect"] == "identical-artifacts"


def test_round_short_circuits_fixer_on_identical_artifacts(tmp_path):
    repairs = [{"stage": "merge", "action": "dedup", "reason": "overlap",
                "params": {"raise_dedup_iou": True}, "severity": "high"}]
    run_dir, input_path = _seed_run(tmp_path, repairs)
    fixer_calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        qa_path = os.path.join(rd, "qa.json")
        qa = json.loads(open(qa_path, encoding="utf-8").read())
        qa["timestamp"] = qa.get("timestamp", 0) + 1
        with open(qa_path, "w", encoding="utf-8") as handle:
            json.dump(qa, handle)
        return {"ok": True, "run_dir": rd}

    import src.harness_fixer as harness_fixer
    original = harness_fixer.apply_fixer_round

    def spy(rd, cfg, critic_output):
        fixer_calls.append(rd)
        return original(rd, cfg, critic_output)

    harness_fixer.apply_fixer_round = spy
    try:
        summary = harness_loop.run_until_acceptable(
            str(input_path), str(run_dir), {}, max_rounds=3,
            pipeline_already_ran=True, run_one=fake_run_one)
    finally:
        harness_fixer.apply_fixer_round = original

    round_records = summary["rounds"]
    identical = [r for r in round_records
                 if r.get("short_circuit") == "identical-artifacts"]
    assert identical, "an identical-artifact round must be short-circuited"
    assert identical[0]["fixer"]["skipped"] == "identical-artifacts"
    assert not fixer_calls, "fixer must not run for an identical-artifact round"
    # and the loop stopped instead of burning the remaining rounds on reruns
    assert summary["rounds_completed"] < 3
