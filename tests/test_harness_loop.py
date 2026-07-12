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
