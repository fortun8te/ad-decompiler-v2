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


# ── P1 raise-floors: gap 3 — speculative SAM element-growth admission ────────────────
#
# Backtest (run-4 admitted rounds, real qa numbers). Every run-4 fixture recorded a
# single no-op round; 021's was an outright regression (rolled back). At HEAD the
# metric-driven element-growth repairs (sam3 rerun-detection, "element recall 0.00")
# were still admitted on already-faithful renders; the element-growth floor refuses them.

def test_element_growth_refused_on_faithful_render_009_021_025():
    floors = harness.admission_floors({})
    # 009: ssim 1.0 / text_recall 1.0 — a perfect render; growing elements is pointless.
    assert harness.element_growth_refused({"ssim": 1.0, "text_recall": 1.0}, floors)
    # 021: ssim 1.0 / text_recall 1.0 — this round REGRESSED in run 4 (rolled back).
    assert harness.element_growth_refused({"ssim": 1.0, "text_recall": 1.0}, floors)
    # 025: ssim 0.989 / text_recall 0.933 — refused via the ssim floor.
    assert harness.element_growth_refused({"ssim": 0.989, "text_recall": 0.933}, floors)


def test_element_growth_admitted_when_render_is_genuinely_broken():
    # A real missing object leaves ssim LOW — element growth stays admissible there.
    floors = harness.admission_floors({})
    assert harness.element_growth_refused({"ssim": 0.60, "text_recall": 0.50}, floors) is None


def test_002_element_growth_refused_at_pre_round_qa_but_runs_on_degraded_qa():
    # The 002 sam3 element-growth is admitted against the PRE-round qa (ssim 0.8174 /
    # text_recall 0.963); the recall floor refuses it there, so the 15-minute round never
    # starts. The regression-shipping tests seed the POST-regression qa (0.7712 / 0.9259)
    # to exercise the cost-control machinery — that must still be admissible (floors sit
    # between the two so those tests stay green).
    floors = harness.admission_floors({})
    assert harness.element_growth_refused({"ssim": 0.8174, "text_recall": 0.963}, floors)
    assert harness.element_growth_refused({"ssim": 0.7712, "text_recall": 0.9259}, floors) is None


def _seed_qa_run(tmp_path, qa, repairs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps(repairs), encoding="utf-8")
    return run_dir


def test_admission_reject_reason_refuses_speculative_element_growth(tmp_path):
    qa = {"ok": False, "ssim": 1.0, "text_recall": 1.0,
          "hard_fails": [{"rule": "low-element-recall"}]}
    repair = {"stage": "sam3", "action": "rerun-detection", "severity": "high",
              "params": {"lower_confidence": True, "enable_element_propose": True}}
    _seed_qa_run(tmp_path, qa, [repair])
    reason = harness.admission_reject_reason(repair, run_dir=str(tmp_path / "run"))
    assert reason and reason.startswith("element-growth")
    # A layer-targeted sam3 rerun (acts on a known object) is NOT speculative growth.
    targeted = {"stage": "sam3", "action": "rerun-detection", "severity": "high",
                "params": {"layer_ids": ["c_E1"]}}
    assert harness._is_speculative_element_growth(targeted) is False


# ── P1 gap 1a — baked/scene-text recall shortfall is unfixable ──────────────────────

def test_baked_majority_via_text_recall_detail_refuses_ocr_rerun(tmp_path):
    # pixel_diff.text_recall_detail excludes verified scene-baked lines from the recall
    # denominator; when they dominate the detected lines an OCR rerun cannot lift recall.
    qa = {"ok": False, "text_recall": 0.42,
          "text_recall_detail": {"recall": 0.42, "found": 3, "lines_total": 12,
                                 "baked_excluded": 8, "baked_excluded_lines": ["x"] * 8}}
    repair = {"stage": "ocr", "action": "rerun", "params": {"upscale": True}}
    _seed_qa_run(tmp_path, qa, [repair])
    reason = harness.admission_reject_reason(repair, run_dir=str(tmp_path / "run"))
    assert reason == "kept-in-photo-text-deficit"


def test_non_baked_recall_miss_still_admits_ocr_rerun(tmp_path):
    # Genuine editable misses (baked lines a minority) keep the OCR rerun admissible.
    qa = {"ok": False, "text_recall": 0.42,
          "text_recall_detail": {"recall": 0.42, "found": 5, "lines_total": 12,
                                 "baked_excluded": 1, "baked_excluded_lines": ["x"]}}
    repair = {"stage": "ocr", "action": "rerun", "params": {"upscale": True}}
    _seed_qa_run(tmp_path, qa, [repair])
    assert harness.admission_reject_reason(repair, run_dir=str(tmp_path / "run")) is None


# ── P1 gap 1b — OCR-truth mismatch: design already carries the correct text ──────────

def test_ocr_truth_mismatch_refuses_recall_rerun_101(tmp_path):
    # 101 real numbers: editable_text_recall 1.0 (render text is correct vs the design)
    # while text_recall 0.13 (source-OCR ground-truth disagrees). A render/OCR rerun is a
    # guaranteed no-op — the defect is the source OCR truth, not the render.
    qa = {"ok": False, "editable_text_recall": 1.0, "text_recall": 0.1304}
    for action in ("rerun", "restore-editable-text", "boost-stack"):
        stage = "ocr" if action != "restore-editable-text" else "text-analysis"
        repair = {"stage": stage, "action": action, "params": {}}
        rd = tmp_path / action
        rd.mkdir()
        (rd / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
        reason = harness.admission_reject_reason(repair, run_dir=str(rd))
        assert reason and reason.startswith("ocr-truth-mismatch"), (action, reason)


def test_ocr_truth_not_triggered_when_editable_recall_also_low(tmp_path):
    # 067: editable 0.778 / text_recall 0.833 — both low, the text is genuinely missing,
    # so the OCR rerun is a legitimate (non-refused) repair.
    qa = {"ok": False, "editable_text_recall": 0.7778, "text_recall": 0.8333}
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    repair = {"stage": "ocr", "action": "rerun", "params": {"upscale": True}}
    assert harness.admission_reject_reason(repair, run_dir=str(rd)) is None


# ── P1 gap 2 — critic crop targets the genuinely-worst region (per-layer aware) ──────

def test_worst_crop_prefers_worst_per_layer_over_milder_window():
    from src import qa_reward
    # A per-layer region (ssim 0.01) is worse than the local worst-window (ssim 0.30):
    # the crop must target the per-layer region, not the milder window.
    qa = {
        "local_ssim_worst_window": {"ssim": 0.30, "bbox": {"x": 8, "y": 8, "w": 16, "h": 16}},
        "per_layer": [
            {"id": "c_ok", "region_ssim": 0.80, "abs_box": {"x": 500, "y": 500, "w": 40, "h": 40}},
            {"id": "c_bad", "region_ssim": 0.01, "abs_box": {"x": 700, "y": 300, "w": 60, "h": 60}},
        ],
    }
    box = qa_reward._worst_crop_box(qa)
    assert box == {"x": 700, "y": 300, "w": 60, "h": 60}
    # Window alone (no per-layer) is unchanged — the F9 behaviour is preserved.
    only_window = {"local_ssim_worst_window": {"ssim": 0.05, "bbox": {"x": 8, "y": 8, "w": 16, "h": 16}}}
    assert qa_reward._worst_crop_box(only_window) == {"x": 8, "y": 8, "w": 16, "h": 16}


# ── F: the second pass that changed nothing (postfix-benchmark-7 / 013) ──────────────
#
# Real record, runs/postfix-benchmark-7/013_attached_a32b069cce97685c:
#
#   [01:28:00] qa          -> ssim=0.7073 text_recall=0.8462 repairs=6
#   [01:29:53] reconstruct -> 25 entities, background=regional        <- pass 2
#   [01:30:09] qa          -> ssim=0.7073 text_recall=0.8462 repairs=6   IDENTICAL
#   [01:30:10] harness     -> plateau after 1/2 round(s)
#
#   repair: merge:dedup c_B2, "VLM critique: duplicated text '% OFF'"
#   resume: merge | artifacts_changed: TRUE | no_effect: null | elapsed_s: 122.5
#
# ``execute_repairs`` fingerprinted stage output BYTES, so re-serialized design.json
# diagnostics plus a re-encoded preview.png flipped artifacts_changed to True while the
# render and every QA metric were bit-identical. The round escaped the identical-artifact
# short-circuit and burned 122.5s to change nothing — the user's "im not seeing any
# fucking changes bruv". These tests seed 013's real numbers and layers.

# 013's real merged.json text layers. Nothing is duplicated: c_B2 is the only layer
# carrying "% OFF", and it reads "61% OFF" — the VLM's duplicate never existed.
_013_MERGED = [
    {"id": "c_B0", "target": "text", "text": "We NEVER\ndo this!",
     "box": {"x": 82.0, "y": 120.0, "w": 500.0, "h": 160.0}, "meta": {"confidence": 0.9}},
    {"id": "c_B2", "target": "text", "text": "61% OFF",
     "box": {"x": 97.03126609325409, "y": 807.3987579345703,
             "w": 283.7109214067459, "h": 104.68568801879883},
     "meta": {"confidence": 0.95}},
    {"id": "c_B13", "target": "text", "text": "ONLINE EXCLUSIVE OFFER ENDING SOON",
     "box": {"x": 82.0, "y": 1462.0, "w": 918.0, "h": 54.0}, "meta": {"confidence": 0.88}},
]

# The repair the VLM proposed, exactly as repair.py builds it for a duplicate_text anomaly.
_013_DEDUP_REPAIR = {
    "stage": "merge", "action": "dedup",
    "reason": "VLM critique: duplicated text - '% OFF'",
    "params": {"raise_dedup_iou": True, "duplicate_text": ["% OFF"],
               "layer_ids": ["c_B2"], "source": "vlm_critique"},
    "severity": "high", "target_id": "c_B2",
}

# 013's real QA numbers, before and after the 122.5s no-op round.
_013_QA = {"ok": False, "ssim": 0.7073, "text_recall": 0.8462,
           "hard_fails": [{"rule": "visual", "detail": "local ssim 0.459 < 0.5"}]}


def _write_preview(path, mark=None):
    """013-shaped 1080x1920 preview; ``mark`` paints a real, local render change."""
    from PIL import Image
    image = Image.new("RGB", (1080, 1920), (18, 32, 24))
    if mark:
        pixels = image.load()
        for y in range(mark):
            for x in range(mark):
                pixels[x, y] = (255, 0, 0)
    image.save(path)


def _013_design(layers=None):
    return {
        "id": "013_attached_a32b069cce97685c", "schema_version": 2,
        "canvas": {"w": 1080, "h": 1920},
        "layers": layers or [
            {"id": "c_B2", "type": "text", "text": "61% OFF",
             "box": {"x": 97.03126609325409, "y": 807.3987579345703,
                     "w": 283.7109214067459, "h": 104.68568801879883},
             "fill": "#FFFFFF", "meta": {"confidence": 0.95}},
        ],
        # Diagnostics: rewritten on every rerun, read by nobody, decide nothing.
        "meta": {"layer_count": 12, "editable_ratio": 0.5, "warnings": [],
                 "single_ownership": {"collapsed": 2, "noop": 1},
                 "compiler": "scene-graph-v2"},
    }


def _seed_013(tmp_path, repairs, merged=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps(repairs), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({**_013_QA, "repairs": repairs}), encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps(_013_design(), indent=2),
                                         encoding="utf-8")
    (run_dir / "merged.json").write_text(
        json.dumps(_013_MERGED if merged is None else merged), encoding="utf-8")
    _write_preview(str(run_dir / "preview.png"))
    return run_dir, input_path


def _noop_rerun(run_dir):
    """Reproduce 013's byte churn: diagnostics rewritten, preview re-encoded, QA re-emitted.

    Every metric holds at ssim=0.7073 / text_recall=0.8462 and every pixel holds.
    """
    design_path = os.path.join(run_dir, "design.json")
    design = _load(design_path)
    design["meta"]["single_ownership"]["noop"] = 99      # diagnostic counter moved
    design["meta"]["warnings"] = ["merge rerun: dedup applied to 0 layers"]
    design["meta"]["generated_at"] = "2026-07-16T01:29:53"   # timestamp
    design = dict(reversed(list(design.items())))            # key order churn
    with open(design_path, "w", encoding="utf-8") as handle:
        json.dump(design, handle, separators=(",", ":"))     # re-serialized compactly
    _write_preview(os.path.join(run_dir, "preview.png"))      # re-encoded, same pixels
    qa_path = os.path.join(run_dir, "qa.json")
    qa = _load(qa_path)
    qa["timestamp"] = qa.get("timestamp", 0) + 1
    with open(qa_path, "w", encoding="utf-8") as handle:
        json.dump(qa, handle)


def test_013_byte_delta_with_identical_render_is_a_no_op(tmp_path):
    """The bug: bytes moved, render did not. Must short-circuit, not spend a round."""
    # An untargeted probe, so the admission dedup screen is not what is under test here.
    repairs = [{"stage": "merge", "action": "dedup", "reason": "overlap",
                "params": {"raise_dedup_iou": True}, "severity": "high"}]
    run_dir, _ = _seed_013(tmp_path, repairs)
    before_bytes = harness._artifact_fingerprint(str(run_dir / "design.json"))

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        _noop_rerun(rd)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=1,
                                      run_one=fake_run_one)
    executed = [a for a in summary["attempts"]
                if not a.get("admission_skipped") and not a.get("admission_rejected")]
    assert executed, "the probe must actually run"
    attempt = executed[0]
    # The bytes really did change — this is the exact condition that fooled HEAD.
    assert harness._artifact_fingerprint(str(run_dir / "design.json")) != before_bytes
    # ...but the OUTCOME did not, so the round is a no-op and says why.
    assert attempt["artifacts_changed"] is False, (
        "a re-serialized diagnostic and a re-encoded PNG are not a change")
    assert attempt["no_effect"] == "identical-render"
    detail = attempt["no_effect_detail"]
    assert detail["bytes_changed"] is True          # names what actually happened
    assert "design.json" in detail["compared"] and "preview.png" in detail["compared"]
    assert "ssim" in detail["qa_metrics_identical"]


def test_013_identical_render_round_short_circuits_and_stops(tmp_path):
    """122.5s of merge->reconstruct->qa must not be followed by another round."""
    repairs = [{"stage": "merge", "action": "dedup", "reason": "overlap",
                "params": {"raise_dedup_iou": True}, "severity": "high"}]
    run_dir, input_path = _seed_013(tmp_path, repairs)
    runs = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        runs.append(start_from)
        _noop_rerun(rd)
        return {"ok": True, "run_dir": rd}

    import src.harness_fixer as harness_fixer
    original = harness_fixer.apply_fixer_round
    fixer_calls = []

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

    short = [r for r in summary["rounds"] if r.get("short_circuit")]
    assert short, "an identical-render round must be short-circuited"
    assert short[0]["short_circuit"] == "identical-render"
    assert short[0]["fixer"]["skipped"] == "identical-render"
    assert short[0]["short_circuit_detail"], "must record what was compared"
    assert not fixer_calls, "no fixer pass on an unchanged design"
    # The lever is not re-run on a second round — the 013 complaint.
    assert summary["rounds_completed"] < 3
    assert len(runs) == 1, f"the no-op lever must not be replayed, got {runs}"


def test_013_targeted_dedup_screened_before_spending_122_seconds(tmp_path):
    """The real 013 repair: screened at admission, so run_one is never called."""
    run_dir, _ = _seed_013(tmp_path, [_013_DEDUP_REPAIR])
    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=1,
                                      run_one=fake_run_one)
    assert not calls, "a provably inert dedup must never start a pipeline rerun"
    skipped = [a for a in summary["attempts"] if a.get("admission_skipped")]
    assert skipped and skipped[0]["reason"] == "dedup-target-not-duplicated"
    detail = skipped[0]["detail"]
    assert "structurally incapable" in detail
    assert "'% OFF'" in detail or "% OFF" in detail
    assert "drops 0 of 3" in detail          # merge's own dedup, asked directly
    # and the audit trail is persisted for the run
    admission = _load(os.path.join(str(run_dir), "harness_admission.json"))
    assert any(item["reason"] == "dedup-target-not-duplicated"
               for item in admission["skipped"])


def test_013_dedup_screen_does_not_block_capable_repairs(tmp_path):
    """Do not over-block: a real duplicate, or an untargeted IoU probe, still runs."""
    run_dir, _ = _seed_013(tmp_path, [])
    patches = harness.recommended_resume([_013_DEDUP_REPAIR])["patches"]
    # The real 013 patch against the real 013 layers: provably inert.
    assert harness.targeted_dedup_noop_reason(str(run_dir), patches)

    # A genuine duplicate ("61% OFF" twice, overlapping) is capable -> not screened.
    duplicated = _013_MERGED + [
        {"id": "c_B99", "target": "text", "text": "61% OFF",
         "box": {"x": 97.03126609325409, "y": 807.3987579345703,
                 "w": 283.7109214067459, "h": 104.68568801879883},
         "meta": {"confidence": 0.6}},
    ]
    dup_dir, _ = _seed_013(tmp_path / "dup", [], merged=duplicated)
    real = harness.recommended_resume([{
        **_013_DEDUP_REPAIR,
        "params": {**_013_DEDUP_REPAIR["params"], "duplicate_text": ["61% OFF"]},
    }])["patches"]
    assert harness.targeted_dedup_noop_reason(str(dup_dir), real) is None

    # Two named layers: the id path can drop layer_ids[1:] -> capable.
    two_ids = {"merge": {"dedup_iou": 0.72, "dedup_text": True,
                         "layer_ids": ["c_B2", "c_B13"]}}
    assert harness.targeted_dedup_noop_reason(str(run_dir), two_ids) is None
    # The untargeted raise_dedup_iou probe is not this screen's business.
    assert harness.targeted_dedup_noop_reason(
        str(run_dir), {"merge": {"dedup_iou": 0.72}}) is None
    # Missing evidence fails OPEN rather than silently blocking work.
    assert harness.targeted_dedup_noop_reason(
        str(tmp_path / "nonexistent"), patches) is None


def test_013_genuine_improvement_still_proceeds(tmp_path):
    """The over-block guard: a round that really moves the render is not short-circuited."""
    repairs = [{"stage": "merge", "action": "dedup", "reason": "overlap",
                "params": {"raise_dedup_iou": True}, "severity": "high"}]
    run_dir, _ = _seed_013(tmp_path, repairs)

    def improving_run_one(path, rd, cfg, start_from="normalize"):
        # A real repaint plus a real node-content change and a real QA gain.
        _write_preview(os.path.join(rd, "preview.png"), mark=60)
        design = _013_design()
        design["layers"][0]["text"] = "61% OFF"
        design["layers"][0]["fill"] = "#FF0000"
        with open(os.path.join(rd, "design.json"), "w", encoding="utf-8") as handle:
            json.dump(design, handle)
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as handle:
            json.dump({**_013_QA, "ssim": 0.7073 + 0.04, "repairs": repairs}, handle)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=1,
                                      run_one=improving_run_one)
    executed = [a for a in summary["attempts"]
                if not a.get("admission_skipped") and not a.get("admission_rejected")]
    attempt = executed[0]
    assert attempt["artifacts_changed"] is True
    assert attempt["qa_improved"] is True
    assert "no_effect" not in attempt


def test_013_local_render_change_survives_the_pixel_epsilon(tmp_path):
    """A LOCAL edit must count: repairs are local, so a whole-image mean would hide them.

    On 013's 1080x1920 preview a real 60x60 repaint averages to 0.243/255 — under any
    sane mean epsilon — while a lossless re-encode moves exactly 0 pixels.
    """
    base = str(tmp_path / "a.png")
    same = str(tmp_path / "b.png")
    local = str(tmp_path / "c.png")
    _write_preview(base)
    _write_preview(same)          # identical pixels, independently encoded
    _write_preview(local, mark=60)
    before = harness._outcome_fingerprint(base)
    assert harness._outcome_changed(before, harness._outcome_fingerprint(same)) is False
    assert harness._outcome_changed(before, harness._outcome_fingerprint(local)) is True


def test_semantic_json_ignores_diagnostics_but_not_content(tmp_path):
    """Node set / text / boxes / fills decide; diagnostics, timings, key order do not."""
    path = str(tmp_path / "design.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_013_design(), handle, indent=2)
    before = harness._outcome_fingerprint(path)

    def rewrite(mutate):
        design = _013_design()
        mutate(design)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(design, handle, separators=(",", ":"))
        return harness._outcome_changed(before, harness._outcome_fingerprint(path))

    # Ignored: diagnostics, timings, key order, float jitter below QA's resolution.
    assert rewrite(lambda d: d["meta"].update({"warnings": ["x"], "layer_count": 99})) is False
    assert rewrite(lambda d: d.update({"meta": {"generated_at": "2026-07-16"}})) is False
    assert rewrite(lambda d: d["layers"][0]["meta"].update({"confidence": 0.1})) is False
    assert rewrite(lambda d: d["layers"][0]["box"].__setitem__(
        "x", d["layers"][0]["box"]["x"] + 1e-9)) is False
    # Honoured: the content that decides the render.
    assert rewrite(lambda d: d["layers"][0].__setitem__("text", "62% OFF")) is True
    assert rewrite(lambda d: d["layers"][0].__setitem__("fill", "#000000")) is True
    assert rewrite(lambda d: d["layers"][0]["box"].__setitem__("x", 400.0)) is True
    assert rewrite(lambda d: d["layers"].append({"id": "c_new", "type": "text"})) is True
    assert rewrite(lambda d: d["layers"].clear()) is True


def test_qa_metric_movement_counts_as_a_change():
    """A QA move beyond noise is a change even if the render digest held."""
    assert harness._qa_metrics_moved({"ssim": 0.0, "text_recall": 0.0}) is False
    assert harness._qa_metrics_moved({"ssim": 0.001}) is False       # under tolerance
    assert harness._qa_metrics_moved({"ssim": 0.02}) is True
    assert harness._qa_metrics_moved({"ssim": -0.02}) is True        # regressions too
    assert harness._qa_metrics_moved({"text_recall": 0.05}) is True
    assert harness._qa_metrics_moved({"hard_fails": 1}) is True
    # 013's real deltas: every metric flat.
    assert harness._qa_metrics_moved({
        "ssim": 0.0, "visual_score": 0.0, "text_recall": 0.0,
        "editable_text_recall": 0.0, "edge_f1": 0.0, "color_similarity": 0.0,
        "hard_fails": 0}) is False
