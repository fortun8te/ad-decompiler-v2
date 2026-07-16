"""Tests for the failure-proof harness loop orchestrator."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness_loop


def _seed_run(tmp_path, qa_ok=False):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": qa_ok, "ssim": 0.7, "text_recall": 0.6, "hard_fails": [],
                    "repairs": [
                        {"stage": "ocr", "action": "rerun", "reason": "low recall",
                         "severity": "high", "params": {"upscale": True}},
                    ]}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(
        json.dumps([
            {"stage": "ocr", "action": "rerun", "reason": "low recall",
             "severity": "high", "params": {"upscale": True}},
        ]), encoding="utf-8")
    return str(input_path), str(run_dir)


def test_run_until_acceptable_stops_when_qa_ok(tmp_path):
    input_path, run_dir = _seed_run(tmp_path, qa_ok=True)
    calls = {"pipeline": 0, "repairs": 0}

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls["pipeline"] += 1
        (tmp_path / "run" / "qa.json").write_text(
            json.dumps({"ok": True, "repairs": [], "hard_fails": [], "ssim": 0.95}),
            encoding="utf-8")
        return {"ok": True, "run_dir": rd, "qa_ok": True, "runtime_ok": True}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None):
        calls["repairs"] += 1
        return {"stopped": "already_ok", "qa_ok": True, "iterations": 0, "attempts": []}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=3,
        run_one=fake_run_one, execute_repairs_fn=fake_repairs,
    )

    assert summary["qa_ok"] is True
    assert summary["stopped"] in {"qa_ok", "qa_ok_after_repairs"}
    assert summary["rounds_completed"] == 1
    assert calls["pipeline"] == 1
    assert calls["repairs"] == 0
    assert (tmp_path / "run" / "harness_loop.json").exists()


def test_run_until_acceptable_repairs_then_critic_fixer(tmp_path, monkeypatch):
    input_path, run_dir = _seed_run(tmp_path)
    calls = {"pipeline": 0, "repairs": 0, "critic": 0, "fixer": 0}

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls["pipeline"] += 1
        qa_ok = calls["pipeline"] >= 2
        (tmp_path / "run" / "qa.json").write_text(
            json.dumps({"ok": qa_ok, "repairs": []}), encoding="utf-8")
        return {"ok": True, "run_dir": rd}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None):
        calls["repairs"] += 1
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": []}

    def fake_critic(rd, cfg):
        calls["critic"] += 1
        return {"prioritized_issues": [{"category": "ocr"}], "suggested_fix_ids": ["fix_ocr_stack"],
                "blockers": [], "filtered_repairs": []}

    def fake_fixer(rd, cfg, critic_output):
        calls["fixer"] += 1
        patched = dict(cfg)
        patched.setdefault("ocr", {})["challengers"] = ["easyocr"]
        return {"cfg": patched, "fixes": ["fix_ocr_stack"]}

    monkeypatch.setattr(harness_loop, "_run_critic_pass", fake_critic)
    monkeypatch.setattr(harness_loop, "_run_fixer_pass", fake_fixer)

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {"runtime": {"harness": {"repair_iterations": 1}}},
        max_rounds=3, run_one=fake_run_one, execute_repairs_fn=fake_repairs,
    )

    assert summary["qa_ok"] is True
    assert summary["stopped"] == "qa_ok"
    assert calls["critic"] >= 1
    assert calls["fixer"] >= 1
    assert (tmp_path / "run" / "critic.json").exists()
    assert (tmp_path / "run" / "fixer.json").exists()
    loop = json.loads((tmp_path / "run" / "harness_loop.json").read_text(encoding="utf-8"))
    assert loop["rounds_completed"] >= 2
    assert loop["thresholds"]["visual_pass_ssim"] == 0.9


def test_run_until_acceptable_respects_max_rounds(tmp_path, monkeypatch):
    input_path, run_dir = _seed_run(tmp_path)

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        return {"ok": True, "run_dir": rd}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None):
        return {"stopped": "no_actionable_repairs", "qa_ok": False, "iterations": 0, "attempts": []}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": []})

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2,
        run_one=fake_run_one, execute_repairs_fn=fake_repairs,
    )

    assert summary["qa_ok"] is False
    assert summary["stopped"] == "no_progress"
    assert summary["rounds_completed"] == 1


def test_threshold_guard_rejects_lowered_ssim(tmp_path, monkeypatch):
    input_path, run_dir = _seed_run(tmp_path)
    cfg = {"qa": {"visual_pass_ssim": 0.9}}

    def fake_run_one(path, rd, c, start_from="normalize"):
        return {"ok": True, "run_dir": rd}

    def fake_repairs(rd, c, max_iterations=2, run_one=None):
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": []}

    def bad_fixer(rd, c, critic_output):
        patched = dict(c)
        patched.setdefault("qa", {})["visual_pass_ssim"] = 0.5
        return {"cfg": patched, "fixes": ["bad"]}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass", bad_fixer)

    with pytest.raises(ValueError, match="must not lower QA thresholds"):
        harness_loop.run_until_acceptable(
            input_path, run_dir, cfg, max_rounds=1,
            run_one=fake_run_one, execute_repairs_fn=fake_repairs,
        )


def test_harness_after_pipeline_skips_initial_run(tmp_path, monkeypatch):
    input_path, run_dir = _seed_run(tmp_path)
    calls = {"pipeline": 0}

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls["pipeline"] += 1
        (tmp_path / "run" / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True, "run_dir": rd}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None):
        (tmp_path / "run" / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"stopped": "qa_ok", "qa_ok": True, "iterations": 1, "attempts": []}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": []})

    summary = harness_loop.run_harness_after_pipeline(
        input_path, run_dir, {}, max_rounds=2,
        run_one=fake_run_one, execute_repairs_fn=fake_repairs,
    )

    assert summary["qa_ok"] is True
    assert calls["pipeline"] == 0


def test_in_harness_loop_guard():
    assert harness_loop.in_harness_loop({"runtime": {"harness": {"_in_loop": True}}})
    assert not harness_loop.in_harness_loop({"runtime": {"harness": {"enabled": True}}})


def test_pipeline_exception_is_reported_and_summary_is_written(tmp_path):
    input_path, run_dir = _seed_run(tmp_path)

    def broken_runner(*args, **kwargs):
        raise RuntimeError("GPU worker exited")

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2, run_one=broken_runner)

    assert summary["qa_ok"] is False
    assert summary["stopped"] == "pipeline_exception"
    assert summary["rounds"][0]["pipeline"]["error"] == "GPU worker exited"
    assert json.loads((tmp_path / "run" / "harness_loop.json").read_text())["qa_ok"] is False


def test_stale_passing_qa_still_runs_repairs(tmp_path, monkeypatch):
    """Production runners that skip rewriting qa.json must not short-circuit the harness."""
    input_path, run_dir = _seed_run(tmp_path, qa_ok=True)
    calls = {"repairs": 0}

    def stale_runner(path, rd, cfg, start_from="normalize"):
        return {"ok": True, "qa_ok": True, "runtime_ok": True}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        calls["repairs"] += 1
        return {"stopped": "no_actionable_repairs", "qa_ok": True, "iterations": 0, "attempts": []}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": []})

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2, run_one=stale_runner,
        execute_repairs_fn=fake_repairs,
    )
    assert summary["rounds"][0]["pipeline"]["qa_fresh"] is False
    assert summary["rounds"][0]["pipeline"].get("qa_stale") is True
    assert calls["repairs"] >= 1
    assert summary["stopped"] != "qa_not_refreshed"


def test_malformed_critic_falls_back_with_visible_error(tmp_path, monkeypatch):
    input_path, run_dir = _seed_run(tmp_path)
    monkeypatch.setattr("src.harness_critic.analyze", lambda *a, **k: None)

    output = harness_loop._run_critic_pass(run_dir, {})

    assert output["critic_error"] == "critic returned malformed output"
    assert isinstance(output["filtered_repairs"], list)


# ── convergence guarantees (best-kept / rollback / no-repeat / plateau) ─────────────────

def _seed_scored_run(tmp_path, *, ssim, design_marker="BASE"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "design.json").write_text(
        json.dumps({"marker": design_marker, "layers": []}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": False, "ssim": ssim, "hard_fails": [], "repairs": [
            {"stage": "ocr", "action": "rerun", "severity": "high"}]}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "severity": "high"},
    ]), encoding="utf-8")
    return str(input_path), str(run_dir)


def _write_state(run_dir, *, ssim, marker):
    with open(os.path.join(run_dir, "design.json"), "w", encoding="utf-8") as fh:
        json.dump({"marker": marker, "layers": []}, fh)
    with open(os.path.join(run_dir, "qa.json"), "w", encoding="utf-8") as fh:
        json.dump({"ok": False, "ssim": ssim, "marker": marker, "hard_fails": [],
                   "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high"}]}, fh)


def _read_marker(run_dir):
    with open(os.path.join(run_dir, "design.json"), encoding="utf-8") as fh:
        return json.load(fh)["marker"]


def test_regressing_round_rolls_back_and_best_design_is_returned(tmp_path, monkeypatch):
    input_path, run_dir = _seed_scored_run(tmp_path, ssim=0.70, design_marker="BEST")
    state = {"n": 0}

    def run_one(path, rd, cfg, start_from="normalize"):
        state["n"] += 1
        _write_state(rd, ssim=0.30 + 0.02 * state["n"], marker=f"BAD{state['n']}")
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": [
            {"repair": {"stage": "ocr", "action": "rerun", "target_id": None},
             "qa_improved": False, "pipeline_ok": True}]}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": ["keep-going"]})

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    # (c) best-kept: every round regressed, so the original BEST design is emitted.
    assert _read_marker(run_dir) == "BEST"
    # (a) at least one regressing round was rolled back.
    trail = summary["convergence"]["trail"]
    assert any(entry["rolled_back"] for entry in trail)
    assert summary["convergence"]["best_round"] == 0
    assert summary["convergence"]["rolled_back_rounds"] >= 1


def test_no_op_repair_is_not_retried_and_plateau_stops(tmp_path, monkeypatch):
    input_path, run_dir = _seed_scored_run(tmp_path, ssim=0.50)
    received_blocked = []

    def run_one(path, rd, cfg, start_from="normalize"):
        # Fresh qa each round but identical quality — a genuine no-op / plateau.
        _write_state(rd, ssim=0.50, marker=f"round-{len(received_blocked)}")
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        received_blocked.append(set(blocked_repairs or ()))
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": [
            {"repair": {"stage": "ocr", "action": "rerun", "target_id": None},
             "qa_improved": False, "pipeline_ok": True}]}

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": ["keep-going"]})

    # plateau_rounds=2 so the loop reaches a second round (needed to observe the
    # blocked-repair handoff); production default remains 1 (workstream E).
    cfg = {"runtime": {"harness": {"plateau_rounds": 2}}}
    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, cfg, max_rounds=5,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    # (b) the no-op ocr:rerun applied in round 1 is blocked for every later round.
    assert len(received_blocked) >= 2
    assert ("ocr", "rerun", None) in received_blocked[1]
    # PLATEAU stop: identical scores for plateau_rounds rounds ends the loop early.
    assert summary["stopped"] == "plateau"
    assert summary["rounds_completed"] < 5


def test_convergence_config_keys_are_reportable():
    cfg = {"runtime": {"harness": {"epsilon": 0.02, "plateau_rounds": 3}}}
    assert harness_loop.convergence_epsilon(cfg) == 0.02
    assert harness_loop.plateau_round_limit(cfg) == 3
    assert harness_loop.convergence_epsilon({}) == harness_loop._DEFAULT_EPSILON
    assert harness_loop.plateau_round_limit({}) == harness_loop._DEFAULT_PLATEAU_ROUNDS


def test_default_harness_budget_is_bounded_to_two_measured_repairs():
    assert harness_loop.max_harness_rounds({}) == 2
    assert harness_loop.repair_iterations({}) == 1
    # Workstream E: plateau after 1 no-gain round (matches config.example.yaml).
    assert harness_loop._DEFAULT_PLATEAU_ROUNDS == 1
    assert harness_loop.plateau_round_limit({}) == 1


def test_config_example_harness_budget_is_at_most_two_rounds():
    import os
    import yaml

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "config.example.yaml"), encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    harness_cfg = cfg["runtime"]["harness"]
    assert harness_cfg["max_rounds"] <= 2
    assert harness_cfg["plateau_rounds"] == 1
    reward = cfg["qa"]["reward"]
    assert reward["local_ssim_min"] >= 0.55
    assert reward["worst_local_ssim_min"] >= 0.15
    assert reward["critique"].get("crop_worst", True) is True


# ── regression: rollback must cover every artifact a report reads, not just qa.json ─────

def test_regression_across_all_artifacts_ships_best_round_and_records_it(tmp_path, monkeypatch):
    """Reproduces runs/benchmark-final/016_attached_ac1eeeabce759396: round 1 is the best
    round and every later round degrades. The final on-disk state (and qa.json) must equal
    round 1's snapshot for EVERY artifact a report treats as authoritative -- not just
    design.json/qa.json/preview.png/layout.json -- and the harness must record which round
    shipped (harness.json + harness_loop.json ``shipped_round``).

    Before the fix, only design.json/qa.json/preview.png/layout.json/figma_export.png were
    snapshotted and restored on rollback. reconstruction.json, fallback.json, and
    runtime_report.json (which embeds its own qa_evidence/qa_ok mirror of qa.json) were left
    holding whatever the last, regressed round's pipeline run had written -- a "mixed
    rounds" state where the shipped qa.json disagreed with the shipped runtime_report.json.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")

    # Seed a deliberately-worse "round 0" (pre-loop) state so round 1 becomes the new best.
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"round": 0, "input": str(input_path)}), encoding="utf-8")
    (run_dir / "design.json").write_text(
        json.dumps({"marker": "ROUND0", "layers": []}), encoding="utf-8")
    (run_dir / "reconstruction.json").write_text(json.dumps({"round": 0}), encoding="utf-8")
    (run_dir / "fallback.json").write_text(json.dumps({"round": 0}), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": False, "ssim": 0.10, "hard_fails": [], "repairs": [
            {"stage": "ocr", "action": "rerun", "severity": "high"}]}), encoding="utf-8")

    scores = {1: 0.90, 2: 0.50, 3: 0.30}  # round 1 is the best; every later round degrades
    state = {"n": 0}

    def run_one(path, rd, cfg, start_from="normalize"):
        state["n"] += 1
        round_num = state["n"]
        ssim = scores[round_num]
        with open(os.path.join(rd, "design.json"), "w", encoding="utf-8") as fh:
            json.dump({"marker": f"ROUND{round_num}", "layers": []}, fh)
        with open(os.path.join(rd, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": False, "ssim": ssim, "hard_fails": [], "repairs": [
                {"stage": "ocr", "action": "rerun", "severity": "high"}]}, fh)
        with open(os.path.join(rd, "reconstruction.json"), "w", encoding="utf-8") as fh:
            json.dump({"round": round_num}, fh)
        with open(os.path.join(rd, "fallback.json"), "w", encoding="utf-8") as fh:
            json.dump({"round": round_num}, fh)
        with open(os.path.join(rd, "runtime_report.json"), "w", encoding="utf-8") as fh:
            json.dump({"round": round_num, "input": str(input_path)}, fh)
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        summary = {"run_dir": rd, "iterations": 1, "qa_ok": False, "stopped": "max_iterations",
                   "attempts": [{"repair": {"stage": "ocr", "action": "rerun", "target_id": None},
                                 "qa_improved": False, "pipeline_ok": True}]}
        # execute_repairs (harness.py) writes harness.json on every call in production;
        # reproduce that here so _patch_harness_report has a file to annotate.
        with open(os.path.join(rd, "harness.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        return summary

    monkeypatch.setattr(harness_loop, "_run_critic_pass",
                        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                                         "blockers": [], "filtered_repairs": []})
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": ["keep-going"]})

    summary = harness_loop.run_until_acceptable(
        str(input_path), str(run_dir), {}, max_rounds=3,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    assert summary["convergence"]["best_round"] == 1
    assert summary["convergence"]["rolled_back_rounds"] >= 1
    assert summary["shipped_round"] == 1
    assert summary["convergence"]["shipped_round"] == 1

    design = json.loads((run_dir / "design.json").read_text())
    qa = json.loads((run_dir / "qa.json").read_text())
    reconstruction = json.loads((run_dir / "reconstruction.json").read_text())
    fallback = json.loads((run_dir / "fallback.json").read_text())
    runtime_report = json.loads((run_dir / "runtime_report.json").read_text())

    assert design["marker"] == "ROUND1"
    assert qa["ssim"] == 0.90
    assert reconstruction["round"] == 1, "reconstruction.json must roll back with qa.json"
    assert fallback["round"] == 1, "fallback.json must roll back with qa.json"
    assert runtime_report["round"] == 1, "runtime_report.json must roll back with qa.json"

    harness_report = json.loads((run_dir / "harness.json").read_text())
    assert harness_report["shipped_round"] == 1
