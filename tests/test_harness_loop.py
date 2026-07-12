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
        return {"ok": True, "run_dir": rd}

    def fake_repairs(rd, cfg, max_iterations=2, run_one=None):
        calls["repairs"] += 1
        return {"stopped": "already_ok", "qa_ok": True, "iterations": 0, "attempts": []}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=3,
        run_one=fake_run_one, execute_repairs_fn=fake_repairs,
    )

    assert summary["qa_ok"] is True
    assert summary["stopped"] == "qa_ok"
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


def test_production_runner_cannot_reuse_stale_passing_qa(tmp_path):
    input_path, run_dir = _seed_run(tmp_path, qa_ok=True)

    def stale_runner(path, rd, cfg, start_from="normalize"):
        return {"ok": True, "qa_ok": True, "runtime_ok": True}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2, run_one=stale_runner)
    assert summary["qa_ok"] is False
    assert summary["stopped"] == "qa_not_refreshed"
    assert summary["rounds"][0]["pipeline"]["qa_fresh"] is False


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

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=5,
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
