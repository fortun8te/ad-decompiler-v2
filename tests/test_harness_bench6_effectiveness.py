"""Replay of runs/postfix-benchmark-6 through the bench-6 harness effectiveness fixes.

postfix-benchmark-6 evidence: every fixture stopped on "plateau" or "no_progress" and not
one round converted a failure into a pass. The honest per-fixture audit (harness*.json +
qa.json) showed the repair vocabulary — not the loop machinery — was the problem:

  A  inpaint:rebuild-clean-plate patched {"mode": "auto", "allow_fallback": False}.
     config.yaml ships inpaint.mode=flux_comfy and the regional router resolves "auto" to
     the SAME per-region engines, so the resumed rerun rebuilt a byte-identical plate.
     002/013/066/091 logged metric_deltas of exactly 0.0; 088 moved edge_f1 by -0.0001.
     Five rounds, zero pixels. The levers that physically move glyph residue are the
     removal-mask footprint and the scrub pass -> now an escalation ladder.

  B  _PIPELINE_LEVERS["inpaint"] declared only {mode, allow_fallback}, so the reachability
     screen would have rejected mask_dilate / multipass_fraction as "unreachable" even
     though inpaint.py demonstrably reads both. The screen was blocking the only levers
     that could fix the defect it was screening for.

  C  Nothing carried escalation state, so round 2 re-planned round 1's patch, hit the
     admission `seen` fingerprint and was skipped as "unchanged-repair-plan-and-inputs"
     -> a guaranteed plateau. The ladder rung now advances with attempt history.

  D  Blockers with no config lever (placement_ink_iou, native_leaf_ratio, native_text_ratio,
     and a worst-SSIM window that is NOT residue) were retried anyway. 021's round was not
     merely wasted but harmful: a sam3 rerun against a low-native-leaf-ratio hard-fail cost
     -0.40 text_recall. These are now REFUSED with a stated reason.

  E  The worst local-SSIM window is not automatically structural: in 002/013/066/091 it
     OVERLAPS a flagged glyph-residue box, so the clean-plate lever reaches it. Refusing it
     blindly would have been wrong; co-location decides.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "runs", "postfix-benchmark-6")

needs_bench = pytest.mark.skipif(
    not os.path.isdir(os.path.join(BENCH, "013_attached_a32b069cce97685c")),
    reason="postfix-benchmark-6 artifacts not present")


def _load(path, fallback=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return fallback


def _run(fixture):
    return os.path.join(BENCH, fixture)


# ── A: the clean-plate patch is no longer inert ─────────────────────────────────────

def test_rebuild_clean_plate_escalates_real_levers():
    patch = harness.config_patches_for({
        "stage": "inpaint", "action": "rebuild-clean-plate", "params": {},
    })["inpaint"]
    # The bench-6 patch: mode-only. Anything that only names an engine is inert here.
    assert set(patch) - {"mode", "allow_fallback"}, "patch must move more than the engine"
    assert patch["multipass_fraction"] < 0.12
    assert patch["mask_dilate"]["text"] > 2


def test_ladder_rungs_are_distinct_and_monotonic():
    rungs = [
        harness.config_patches_for({
            "stage": "inpaint", "action": "rebuild-clean-plate",
            "params": {"escalation_level": level},
        })["inpaint"]
        for level in range(len(harness._CLEAN_PLATE_LADDER))
    ]
    assert len(rungs) >= 2
    for lo, hi in zip(rungs, rungs[1:]):
        assert hi != lo, "each rung must be a different experiment"
        assert hi["mask_dilate"]["text"] > lo["mask_dilate"]["text"]
        assert hi["multipass_fraction"] <= lo["multipass_fraction"]


def test_escalation_level_is_clamped_to_the_ladder():
    for level in (-5, 0, 99):
        patch = harness.config_patches_for({
            "stage": "inpaint", "action": "rebuild-clean-plate",
            "params": {"escalation_level": level},
        })
        assert harness.patch_reaches_pipeline(patch)


# ── B: the levers the screen must admit ─────────────────────────────────────────────

def test_inpaint_levers_include_keys_inpaint_actually_reads():
    # Verified against src/inpaint.py: resolve_mask_dilate (375), multipass_fraction
    # (1099), mask_feather (448), strict_acceptance (340/955).
    for key in ("mask_dilate", "multipass_fraction", "mask_feather", "strict_acceptance"):
        assert key in harness._PIPELINE_LEVERS["inpaint"], key
    assert harness.patch_reaches_pipeline({"inpaint": {"mask_dilate": {"text": 6}}})
    assert harness.patch_reaches_pipeline({"inpaint": {"multipass_fraction": 0.06}})


def test_clean_plate_patch_resumes_at_reconstruct_not_earlier():
    # GB6: mask/scrub levers are read by reconstruct; resuming earlier would replay
    # SAM and the peel Flux stack for nothing (the 091 class).
    patch = harness.config_patches_for({
        "stage": "inpaint", "action": "rebuild-clean-plate", "params": {},
    })
    assert harness.earliest_patched_stage(patch) == "reconstruct"


# ── C: escalation advances with history ─────────────────────────────────────────────

def test_escalation_level_advances_with_admission_history(tmp_path):
    run_dir = str(tmp_path)
    assert harness.escalation_level_from_history(run_dir, "inpaint", "rebuild-clean-plate") == 0
    with open(os.path.join(run_dir, "harness_admission.json"), "w", encoding="utf-8") as fh:
        json.dump({"seen": {"fp1": {"plan": {"stage": "inpaint",
                                             "action": "rebuild-clean-plate"}}}}, fh)
    assert harness.escalation_level_from_history(run_dir, "inpaint", "rebuild-clean-plate") == 1
    # An unrelated action must not advance the ladder.
    assert harness.escalation_level_from_history(run_dir, "ocr", "rerun") == 0


def test_stamped_candidates_produce_a_different_patch_on_the_second_round(tmp_path):
    run_dir = str(tmp_path)
    repair = {"stage": "inpaint", "action": "rebuild-clean-plate", "reason": "residue",
              "severity": "high"}
    first = harness._stamp_escalation([dict(repair)], run_dir)[0]
    patch_1 = harness.config_patches_for(first)
    with open(os.path.join(run_dir, "harness_admission.json"), "w", encoding="utf-8") as fh:
        json.dump({"seen": {"fp1": {"plan": {"stage": "inpaint",
                                             "action": "rebuild-clean-plate"}}}}, fh)
    second = harness._stamp_escalation([dict(repair)], run_dir)[0]
    patch_2 = harness.config_patches_for(second)
    assert patch_1 != patch_2, "round 2 must not replay round 1's patch"


# ── D/E: refusal verdicts vs co-located residue ─────────────────────────────────────

@needs_bench
@pytest.mark.parametrize("fixture,expected", [
    # All blockers have levers -> the round is worth spending.
    ("013_attached_a32b069cce97685c", "repair"),
    ("066_attached_c683c2ec1a1648f4", "repair"),
    # No lever for the remaining blockers -> refuse instead of burning a round.
    # 021: bench-6 spent a sam3 rerun here and LOST 0.40 text_recall.
    ("021_attached_6c71fd37fa6c5be4", "refuse"),
    ("104_attached_527b1b7ddbbc5c97", "refuse"),
])
def test_bench6_verdicts(fixture, expected):
    run_dir = _run(fixture)
    qa = _load(os.path.join(run_dir, "qa.json"))
    reward = (_load(os.path.join(run_dir, "harness.json"), {}) or {}).get("reward") or {}
    diagnosis = harness.diagnose_blockers(qa, reward, run_dir=run_dir)
    assert diagnosis["verdict"] == expected, diagnosis


@needs_bench
def test_refusals_always_carry_a_reason():
    for fixture in sorted(os.listdir(BENCH)):
        run_dir = _run(fixture)
        qa = _load(os.path.join(run_dir, "qa.json"))
        if not isinstance(qa, dict) or not qa:
            continue
        reward = (_load(os.path.join(run_dir, "harness.json"), {}) or {}).get("reward") or {}
        diagnosis = harness.diagnose_blockers(qa, reward, run_dir=run_dir)
        for refused in diagnosis["refused"]:
            assert refused.get("reason"), refused
            assert refused.get("blocker"), refused


@needs_bench
def test_worst_window_over_residue_is_reachable_not_structural():
    # 013's worst 64x64 window (192,768) sits inside residue box c_B2 (97..380, 807..912).
    run_dir = _run("013_attached_a32b069cce97685c")
    qa = _load(os.path.join(run_dir, "qa.json"))
    assert harness.worst_window_is_residue(qa, run_dir) is True
    diagnosis = harness.diagnose_blockers(qa, {}, run_dir=run_dir)
    assert "worst_local_ssim" not in [r["blocker"] for r in diagnosis["refused"]]
    assert "worst_local_ssim" in [f["blocker"] for f in diagnosis["fixable"]]


def test_worst_window_without_residue_is_structural(tmp_path):
    # No residue on disk -> the window is a real localized defect with no config lever.
    qa = {"ok": False, "quality_flags": [
        {"rule": "local-ssim-worst-region", "bbox": {"x": 0, "y": 0, "w": 64, "h": 64}}]}
    assert harness.worst_window_is_residue(qa, str(tmp_path)) is False
    diagnosis = harness.diagnose_blockers(qa, {}, run_dir=str(tmp_path))
    assert diagnosis["verdict"] == "refuse"
    assert diagnosis["refused"][0]["reason"] == harness._STRUCTURAL_NEEDS_CODE


@needs_bench
def test_021_native_leaf_ratio_is_refused_not_retried():
    # bench-6 misread this hard-fail as element_recall and spent a sam3 rerun that cost
    # -0.40 text_recall. The blocker must be named correctly and refused.
    run_dir = _run("021_attached_6c71fd37fa6c5be4")
    qa = _load(os.path.join(run_dir, "qa.json"))
    diagnosis = harness.diagnose_blockers(qa, {}, run_dir=run_dir)
    assert diagnosis["verdict"] == "refuse"
    assert "native_leaf_ratio" in diagnosis["blockers"]


def test_clean_qa_yields_no_blockers():
    assert harness.diagnose_blockers({"ok": True}, {})["verdict"] == "clean"
