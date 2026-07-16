"""Replay of runs/postfix-benchmark-5-aborted/002 — regressions must never ship.

Hard evidence from that run (pipeline.log + harness_loop.json, real numbers used below):

  round 0 (first pass, 62s):   ssim 0.8174  text_recall 0.9630  reward 0.563795
  round 1 (repair, ~15 min):   ssim 0.7712  text_recall 0.9259  reward 0.560539

The sam3 rerun-detection repair lowered BOTH ssim (−0.0462) and text_recall (−0.0371),
yet the phase2 scalar ladder compressed that into a −0.003256 reward delta — INSIDE the
0.005 epsilon (local_ssim actually rose 0.4433→0.4501, LPIPS barely moved). So:
  * the in-round score-regression rollback did not fire (delta within epsilon, and it
    was additionally gated on ``not should_stop`` while the round stopped no_progress);
  * only the final safety net's strict ``live < best`` comparison saved the artifacts,
    and nothing would have if the noise had gone +0.004 the other way (the round would
    have been "kept" as the new best) or if the process had been killed mid-loop (the
    benchmark WAS aborted).

Invariant pinned here: a round that lowers ssim AND text_recall versus the best round
must never ship — vetoed and rolled back IN the round, regardless of the scalar ladder
and regardless of the round stopping the loop. Plus the cost-control ceiling: a repair
round may not grind for 15 minutes (918.1s vs run 4's 69s for this same fixture).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, harness_loop, qa_reward

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE = os.path.join(ROOT, "runs", "postfix-benchmark-5-aborted",
                       "002_attached_5885519ba4359843")

# Real numbers from the 002 artifacts (qa.json / harness_loop.json trail).
BEST_SSIM, BEST_RECALL = 0.8174, 0.963
BAD_SSIM, BAD_RECALL = 0.7712, 0.9259
BEST_REWARD, BAD_REWARD = 0.563795, 0.560539
ROUND_WALL_S = 856.0            # the observed repair-round wall time (918.1s − 62.3s)

SAM3_REPAIR = {
    "stage": "sam3", "action": "rerun-detection", "severity": "high",
    "reason": "VLM critique: missing element — 'ADVERTISEMENT CONTENT'",
    "params": {"lower_confidence": True, "enable_element_propose": True},
}


def _qa(ssim, recall, marker):
    return {"ok": False, "ssim": ssim, "text_recall": recall, "marker": marker,
            "hard_fails": [{"rule": "editable-text-recall"}, {"rule": "contract"}],
            "repairs": [dict(SAM3_REPAIR)]}


def _seed_002(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path), "round": 0}), encoding="utf-8")
    (run_dir / "design.json").write_text(
        json.dumps({"marker": "ROUND0", "layers": []}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps(_qa(BEST_SSIM, BEST_RECALL, "ROUND0")), encoding="utf-8")
    (run_dir / "repairs.json").write_text(
        json.dumps([dict(SAM3_REPAIR)]), encoding="utf-8")
    return str(input_path), str(run_dir)


def _fake_reward(monkeypatch):
    """Score rounds with the ladder values the real run recorded, keyed by qa ssim."""
    table = {BEST_SSIM: BEST_REWARD, BAD_SSIM: BAD_REWARD}

    def compute(run_dir, cfg=None, *, qa=None, **kwargs):
        qa = qa or {}
        score = table.get(qa.get("ssim"))
        return {"mode": "phase2", "score": score, "components": {}}

    monkeypatch.setattr(qa_reward, "compute_reward", compute)
    return table


def _regressing_exec(run_dir_holder=None, reward=None):
    """execute_repairs stand-in reproducing 002's recorded round-1 attempt."""
    def exec_repairs(rd, cfg, max_iterations=1, run_one=None, blocked_repairs=None):
        if ("sam3", "rerun-detection", None) in set(blocked_repairs or ()):
            return {"stopped": "no_actionable_repairs", "qa_ok": False,
                    "iterations": 0, "attempts": []}
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump(_qa(BAD_SSIM, BAD_RECALL, "ROUND1"), fh)
        with open(os.path.join(rd, "design.json"), "w", encoding="utf-8") as fh:
            json.dump({"marker": "ROUND1", "layers": []}, fh)
        with open(os.path.join(rd, "runtime_report.json"), "w", encoding="utf-8") as fh:
            json.dump({"round": 1}, fh)
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1,
                "attempts": [{
                    "iteration": 1, "resume": "sam",
                    "repair": {"stage": "sam3", "action": "rerun-detection",
                               "target_id": None},
                    "pipeline_ok": True, "qa_fresh": True, "qa_improved": False,
                    "artifacts_changed": True,
                    "metric_deltas": {"ssim": -0.0462, "text_recall": -0.0371},
                }]}
    return exec_repairs


def _stub_critic_fixer(monkeypatch):
    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": []})


CFG_002 = {"runtime": {"harness": {"reward": "phase2", "plateau_rounds": 1}}}


# ── the shipped-artifact invariant ───────────────────────────────────────────────────

def test_002_regression_inside_epsilon_is_vetoed_and_rolled_back_in_round(tmp_path, monkeypatch):
    """The exact 002 failure: −0.0462 ssim / −0.0371 text_recall but only −0.003256 on
    the reward ladder (inside epsilon 0.005), round stops no_progress. The metric veto
    must roll the round back IN-round — not leave damaged artifacts on disk until (or
    unless) the final safety net runs."""
    input_path, run_dir = _seed_002(tmp_path)
    _fake_reward(monkeypatch)
    _stub_critic_fixer(monkeypatch)

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, CFG_002, max_rounds=2,
        pipeline_already_ran=True,
        run_one=lambda *a, **k: {"ok": True},
        execute_repairs_fn=_regressing_exec())

    qa = json.loads((tmp_path / "run" / "qa.json").read_text(encoding="utf-8"))
    design = json.loads((tmp_path / "run" / "design.json").read_text(encoding="utf-8"))
    assert qa["ssim"] == BEST_SSIM, "shipped ssim must stay 0.8174, not 0.7712"
    assert qa["text_recall"] == BEST_RECALL
    assert design["marker"] == "ROUND0"
    assert summary["shipped_round"] == 0

    trail = summary["convergence"]["trail"]
    entry = trail[0]
    # The rollback happened IN the round (recorded on the round's own trail entry),
    # not merely via the end-of-loop safety net.
    assert entry["rolled_back"] is True
    assert entry["kept"] is False
    assert entry["metric_veto"]["best"]["ssim"] == BEST_SSIM
    assert entry["metric_veto"]["after"]["ssim"] == BAD_SSIM
    assert ["sam3", "rerun-detection", None] in entry["blocked"] or \
        ("sam3", "rerun-detection", None) in {tuple(sig) for sig in entry["blocked"]}
    assert summary["convergence"]["rolled_back_rounds"] >= 1
    assert summary["convergence"]["best_round"] == 0


def test_002_regressed_round_must_not_become_best_even_if_ladder_score_rises(tmp_path, monkeypatch):
    """Counterfactual 002: same double-metric regression but the scalar ladder NOISES
    UPWARD (+0.004). Without the veto the damaged round would be 'kept' as the new best
    and legitimately shipped. The veto must outrank the ladder."""
    input_path, run_dir = _seed_002(tmp_path)
    _stub_critic_fixer(monkeypatch)

    def compute(run_dir_, cfg=None, *, qa=None, **kwargs):
        qa = qa or {}
        score = {BEST_SSIM: BEST_REWARD, BAD_SSIM: BEST_REWARD + 0.004}.get(qa.get("ssim"))
        return {"mode": "phase2", "score": score, "components": {}}

    monkeypatch.setattr(qa_reward, "compute_reward", compute)

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, CFG_002, max_rounds=2,
        pipeline_already_ran=True,
        run_one=lambda *a, **k: {"ok": True},
        execute_repairs_fn=_regressing_exec())

    qa = json.loads((tmp_path / "run" / "qa.json").read_text(encoding="utf-8"))
    assert qa["ssim"] == BEST_SSIM
    entry = summary["convergence"]["trail"][0]
    assert entry["kept"] is False
    assert entry["rolled_back"] is True
    assert summary["convergence"]["best_round"] == 0
    assert summary["shipped_round"] == 0


def test_metric_veto_requires_both_metrics_to_drop():
    best = {"ssim": BEST_SSIM, "text_recall": BEST_RECALL}
    # 002's real regression: both dropped well past tolerance → veto.
    assert harness_loop._metrics_regressed(best, {"ssim": BAD_SSIM,
                                                  "text_recall": BAD_RECALL}) is True
    # The known 0.87/0.5 ↔ 0.37/0.79 oscillation trades one metric for the other —
    # NOT a veto (score/rollback logic owns that case).
    assert harness_loop._metrics_regressed({"ssim": 0.87, "text_recall": 0.5},
                                           {"ssim": 0.37, "text_recall": 0.79}) is False
    # Within-noise moves and missing metrics are never a veto.
    assert harness_loop._metrics_regressed(best, {"ssim": BEST_SSIM - 0.004,
                                                  "text_recall": BEST_RECALL - 0.004}) is False
    assert harness_loop._metrics_regressed(best, {"ssim": 0.5}) is False
    assert harness_loop._metrics_regressed(None, {"ssim": 0.5, "text_recall": 0.5}) is False


def test_accepted_round_is_exempt_from_the_veto(tmp_path, monkeypatch):
    """Acceptance is authoritative: a QA-accepted round ships even if its ssim dipped."""
    input_path, run_dir = _seed_002(tmp_path)
    _stub_critic_fixer(monkeypatch)
    monkeypatch.setattr(qa_reward, "acceptance_gate",
                        lambda *a, **k: {"ok": True, "checks": {}})

    def accepting_exec(rd, cfg, max_iterations=1, run_one=None, blocked_repairs=None):
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True, "ssim": 0.80, "text_recall": 0.95,
                       "hard_fails": [], "repairs": []}, fh)
        return {"stopped": "qa_ok", "qa_ok": True, "iterations": 1, "attempts": [{
            "repair": {"stage": "sam3", "action": "rerun-detection", "target_id": None},
            "pipeline_ok": True, "qa_fresh": True, "qa_improved": True,
            "artifacts_changed": True}]}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, CFG_002, max_rounds=2,
        pipeline_already_ran=True,
        run_one=lambda *a, **k: {"ok": True},
        execute_repairs_fn=accepting_exec)

    assert summary["qa_ok"] is True
    qa = json.loads((tmp_path / "run" / "qa.json").read_text(encoding="utf-8"))
    assert qa["ok"] is True and qa["ssim"] == 0.80


# ── cost control: the 15-minute round must abort and roll back ───────────────────────

class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_round_over_wall_clock_budget_aborts_loop_and_rolls_back(tmp_path, monkeypatch):
    """002's repair round cost 856s of wall time to produce a worse design. With the
    default 600s ceiling the loop must stop after that round (no round 2) AND the
    regressed artifacts must be rolled back."""
    input_path, run_dir = _seed_002(tmp_path)
    _fake_reward(monkeypatch)
    _stub_critic_fixer(monkeypatch)
    clock = _FakeClock()
    monkeypatch.setattr(harness_loop.time, "monotonic", clock)

    inner = _regressing_exec()

    def slow_exec(rd, cfg, **kwargs):
        clock.now += ROUND_WALL_S        # the real 002 round wall time
        return inner(rd, cfg, **kwargs)

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir,
        {"runtime": {"harness": {"reward": "phase2", "plateau_rounds": 5}}},
        max_rounds=4, pipeline_already_ran=True,
        run_one=lambda *a, **k: {"ok": True},
        execute_repairs_fn=slow_exec)

    assert summary["stopped"] == "round_budget_exceeded"
    assert summary["rounds_completed"] == 1
    assert summary["rounds"][0]["budget"]["exceeded"] is True
    assert summary["rounds"][0]["elapsed_s"] >= 600
    qa = json.loads((tmp_path / "run" / "qa.json").read_text(encoding="utf-8"))
    assert qa["ssim"] == BEST_SSIM, "over-budget regressed round must roll back"
    assert summary["shipped_round"] == 0
    assert summary["round_budget"]["wall_clock_s"] == 600.0


def test_execute_repairs_wall_clock_ceiling_stops_further_attempts(tmp_path, monkeypatch):
    """Within one round, once an attempt blew the ceiling no further rerun may start."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "input.png").write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(tmp_path / "input.png")}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps(_qa(BAD_SSIM, BAD_RECALL, "R")), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        dict(SAM3_REPAIR),
        {"stage": "ocr", "action": "rerun", "severity": "high",
         "params": {"upscale": True}},
    ]), encoding="utf-8")

    clock = _FakeClock()
    monkeypatch.setattr(harness.time, "monotonic", clock)
    calls = {"n": 0}

    def slow_run_one(path, rd, cfg, start_from="normalize"):
        calls["n"] += 1
        clock.now += 700.0
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump(_qa(BAD_SSIM, BAD_RECALL, f"r{calls['n']}"), fh)
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=3,
                                      run_one=slow_run_one)

    assert calls["n"] == 1, "second pipeline rerun must not start over budget"
    assert summary["stopped"] == "round_budget_exceeded"
    assert summary["budget"]["elapsed_s"] >= 600
    assert len([a for a in summary["attempts"] if a.get("pipeline_ok")]) == 1


def test_repair_rerun_at_or_before_peel_clamps_flux_budget(tmp_path):
    """Peel discipline for reruns: the 002 sam3 repair resumes at 'sam' (before peel) —
    its rerun must carry a clamped peel.flux_budget so the element-propose cascade can
    never fund a fresh Flux storm. Consumes peel_scene's own config; never raises it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "input.png").write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(tmp_path / "input.png")}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps(_qa(BAD_SSIM, BAD_RECALL, "R")), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([dict(SAM3_REPAIR)]),
                                          encoding="utf-8")
    seen_cfg = {}

    def capture_run_one(path, rd, cfg, start_from="normalize"):
        seen_cfg["cfg"] = cfg
        seen_cfg["start_from"] = start_from
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump(_qa(BAD_SSIM, BAD_RECALL, "again"), fh)
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {"peel": {"flux_budget": 4}},
                                      max_iterations=1, run_one=capture_run_one)

    assert seen_cfg["start_from"] == "sam"
    assert seen_cfg["cfg"]["peel"]["flux_budget"] == 2, \
        "default round budget must halve peel's flux_budget for reruns"
    attempt = summary["attempts"][-1]
    assert attempt["budget_clamp"]["flux_budget"] == {"from": 4, "to": 2}
    assert "elapsed_s" in attempt


def test_round_budget_clamp_semantics():
    # Clamp applies at/before peel and only ever lowers the pipeline's own values.
    budget = {"wall_clock_s": 600.0, "flux_calls": 2, "peel_iterations": None}
    cfg, rec = harness.apply_round_budget_clamp({"peel": {"flux_budget": 4}}, "sam", budget)
    assert cfg["peel"]["flux_budget"] == 2 and rec["flux_budget"] == {"from": 4, "to": 2}
    # Already tighter than the ceiling → untouched.
    cfg, rec = harness.apply_round_budget_clamp({"peel": {"flux_budget": 1}}, "sam", budget)
    assert cfg["peel"]["flux_budget"] == 1
    # Resume after peel → no clamp (the rerun cannot replay the peel stack).
    cfg, rec = harness.apply_round_budget_clamp({"peel": {"flux_budget": 4}}, "layout", budget)
    assert cfg["peel"]["flux_budget"] == 4 and rec is None
    # flux_calls None disables the flux clamp; peel_iterations clamps max_iterations.
    cfg, rec = harness.apply_round_budget_clamp(
        {"peel": {"flux_budget": 4, "max_iterations": 3}}, "sam",
        {"wall_clock_s": 600.0, "flux_calls": None, "peel_iterations": 2})
    assert cfg["peel"]["flux_budget"] == 4
    assert cfg["peel"]["max_iterations"] == 2


def test_round_budget_config_defaults_and_overrides():
    default = harness.round_budget({})
    assert default["wall_clock_s"] == 600.0
    assert default["flux_calls"] == 2
    assert default["peel_iterations"] is None
    custom = harness.round_budget({"runtime": {"harness": {"round_budget": {
        "wall_clock_s": 300, "flux_calls": 1, "peel_iterations": 2}}}})
    assert custom == {"wall_clock_s": 300.0, "flux_calls": 1, "peel_iterations": 2}
    disabled = harness.round_budget({"runtime": {"harness": {"round_budget": {
        "wall_clock_s": 0, "flux_calls": -1}}}})
    assert disabled["wall_clock_s"] == float("inf")
    assert disabled["flux_calls"] is None


# ── replay against the saved 002 artifacts (skipped when the fixture is absent) ──────

needs_fixture = pytest.mark.skipif(
    not os.path.isdir(FIXTURE), reason="postfix-benchmark-5-aborted 002 fixture not present")


@needs_fixture
def test_002_fixture_recorded_regression_triggers_the_veto():
    """The saved artifacts themselves: the recorded round-1 metrics versus the shipped
    qa.json must be exactly the double-regression the veto now catches."""
    with open(os.path.join(FIXTURE, "qa.json"), encoding="utf-8") as fh:
        shipped_qa = json.load(fh)
    with open(os.path.join(FIXTURE, "harness_loop.json"), encoding="utf-8") as fh:
        loop = json.load(fh)
    assert shipped_qa["ssim"] == BEST_SSIM and shipped_qa["text_recall"] == BEST_RECALL
    round1 = loop["rounds"][0]["qa_after_repairs"]
    assert round1["ssim"] == BAD_SSIM and round1["text_recall"] == BAD_RECALL
    assert harness_loop._metrics_regressed(
        harness_loop._veto_metrics(shipped_qa), round1) is True
    # The recorded ladder delta really was inside epsilon — the class the veto exists for.
    trail = loop["convergence"]["trail"][0]
    epsilon = loop["convergence"]["epsilon"]
    assert abs(trail["after_score"] - trail["before_score"]) < epsilon


@needs_fixture
def test_002_fixture_full_replay_ships_best_round(tmp_path, monkeypatch):
    """CPU-only replay: seed a run dir with the fixture's REAL shipped qa.json/design.json,
    re-apply the recorded round-1 regression through the loop, and prove the shipped
    artifacts stay at ssim 0.8174.

    The invariant under test is the SHIPPING one: a regressing sam3 round must never leave
    a worse design on disk. There are now two legitimate ways to honour it, and this test
    accepts either:

      rolled back — the round ran, regressed, and was undone (the original behaviour; the
                    mechanism itself is still covered directly by test_harness_loop.py's
                    test_regressing_round_rolls_back_and_best_design_is_returned)
      refused     — postfix-benchmark-6: this fixture's only blockers are native_text_ratio
                    and a worst-SSIM window that is not glyph residue. Neither has a config
                    lever, so the loop now refuses BEFORE spending the round rather than
                    regressing and undoing it. That is strictly better: the recorded
                    round-1 regression this test replays is exactly the harm being avoided
                    (021 lost 0.40 text_recall to the same sam3-on-a-leverless-blocker
                    move). Prevention and rollback protect the same invariant.
    """
    with open(os.path.join(FIXTURE, "qa.json"), encoding="utf-8") as fh:
        best_qa = json.load(fh)
    with open(os.path.join(FIXTURE, "harness_loop.json"), encoding="utf-8") as fh:
        loop = json.load(fh)
    round1_qa = dict(best_qa)
    for key, value in (loop["rounds"][0]["qa_after_repairs"]).items():
        if key in ("ssim", "text_recall", "visual_score", "edge_f1"):
            round1_qa[key] = value

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "input.png").write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(tmp_path / "input.png")}), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps(best_qa), encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps({"marker": "ROUND0"}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([dict(SAM3_REPAIR)]), encoding="utf-8")

    trail0 = loop["convergence"]["trail"][0]
    rewards = {best_qa["ssim"]: trail0["before_score"],
               round1_qa["ssim"]: trail0["after_score"]}
    monkeypatch.setattr(
        qa_reward, "compute_reward",
        lambda rd, cfg=None, *, qa=None, **kw: {
            "mode": "phase2", "score": rewards.get((qa or {}).get("ssim")),
            "components": {}})
    _stub_critic_fixer(monkeypatch)

    def exec_repairs(rd, cfg, max_iterations=1, run_one=None, blocked_repairs=None):
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump(round1_qa, fh)
        with open(os.path.join(rd, "design.json"), "w", encoding="utf-8") as fh:
            json.dump({"marker": "ROUND1"}, fh)
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1,
                "attempts": [{"repair": {"stage": "sam3", "action": "rerun-detection",
                                         "target_id": None},
                              "pipeline_ok": True, "qa_fresh": True,
                              "qa_improved": False, "artifacts_changed": True}]}

    summary = harness_loop.run_until_acceptable(
        str(tmp_path / "input.png"), str(run_dir), CFG_002, max_rounds=2,
        pipeline_already_ran=True, run_one=lambda *a, **k: {"ok": True},
        execute_repairs_fn=exec_repairs)

    shipped = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    assert shipped["ssim"] == best_qa["ssim"] == BEST_SSIM
    assert shipped["text_recall"] == best_qa["text_recall"]
    assert json.loads((run_dir / "design.json").read_text(encoding="utf-8"))["marker"] == "ROUND0"
    trail = summary["convergence"]["trail"]
    if trail:
        assert trail[0]["rolled_back"] is True          # ran, regressed, undone
    else:
        assert summary["stopped"] == "refused_no_lever"  # never regressed at all
        assert summary["refusal"]["verdict"] == "refuse"
        assert summary["refusal"]["refused"], "a refusal must name what blocked it"
    assert summary["shipped_round"] == 0
