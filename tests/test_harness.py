"""Tests for the repair harness executor."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, repair


def test_recommended_resume_maps_top_actionable_repair():
    repairs = [
        {"stage": "pipeline", "action": "review", "reason": "low composite", "severity": "low"},
        {"stage": "ocr", "action": "rerun", "reason": "text_recall 0.60", "severity": "high",
         "params": {"upscale": True, "challengers": ["surya"]}},
    ]
    choice = harness.recommended_resume(repairs)
    assert choice is not None
    assert choice["resume"] == "ocr"
    assert choice["stage"] == "ocr"
    assert choice["action"] == "rerun"
    assert choice["patches"]["ocr"]["retry_2x"]["enabled"] is True
    assert choice["patches"]["ocr"]["challengers"] == ["surya"]


def test_resume_stage_aliases_text_analysis_and_inpaint():
    assert harness.resume_stage_for({
        "stage": "text-analysis", "action": "resolve-fonts",
    }) == "text"
    assert harness.resume_stage_for({
        "stage": "inpaint", "action": "rebuild-clean-plate",
    }) == "reconstruct"
    assert harness.resume_stage_for({
        "stage": "merge", "action": "dedup", "params": {"raise_dedup_iou": True},
    }) == "merge"


def test_merge_dedup_patch_raises_iou_thresholds():
    patches = harness.config_patches_for({
        "stage": "merge", "action": "dedup", "params": {"raise_dedup_iou": True},
    })
    assert patches["merge"]["dedup_iou"] == 0.72
    assert patches["reconstruct"]["dedup_iou"] == 0.90


def test_execute_repairs_stops_when_qa_already_ok(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    summary = harness.execute_repairs(str(run_dir), {})

    assert summary["stopped"] == "already_ok"
    assert summary["qa_ok"] is True
    assert summary["iterations"] == 0


def test_execute_repairs_reruns_mapped_stage_until_qa_ok(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "reason": "text_recall low",
         "params": {"upscale": True}, "severity": "high"},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False, "repairs": []}), encoding="utf-8")

    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append({"path": path, "run_dir": rd, "start_from": start_from, "cfg": cfg})
        qa_ok = len(calls) >= 2
        (run_dir / "qa.json").write_text(json.dumps({
            "ok": qa_ok,
            "repairs": [] if qa_ok else [
                {"stage": "ocr", "action": "rerun", "reason": "still low", "severity": "high"},
            ],
        }), encoding="utf-8")
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=3, run_one=fake_run_one)

    assert len(calls) == 2
    assert calls[0]["start_from"] == "ocr"
    assert calls[0]["cfg"]["ocr"]["retry_2x"]["enabled"] is True
    assert summary["stopped"] == "qa_ok"
    assert summary["qa_ok"] is True
    assert (run_dir / "harness.json").exists()


def test_execute_repairs_respects_max_iterations(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "qwen", "action": "retry", "reason": "alpha noisy", "severity": "medium"},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}), encoding="utf-8")

    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=2, run_one=fake_run_one)

    assert len(calls) == 2
    assert all(stage == "qwen" for stage in calls)
    assert summary["stopped"] == "max_iterations"
    assert summary["qa_ok"] is False


def test_execute_repairs_stops_without_actionable_repairs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "pipeline", "action": "review", "reason": "manual", "severity": "low"},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}), encoding="utf-8")

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=2, run_one=lambda *a, **k: {"ok": True})

    assert summary["stopped"] == "no_actionable_repairs"
    assert summary["iterations"] == 0


def test_harness_layout_and_inpaint_patches():
    inpaint = harness.config_patches_for({
        "stage": "inpaint", "action": "rebuild-clean-plate", "params": {},
    })
    assert inpaint["inpaint"]["mode"] == "auto"

    layout = harness.config_patches_for({
        "stage": "layout", "action": "refit-geometry",
        "params": {"tighten_containers": True},
    })
    assert layout["layout"]["min_container_frac"] == 0.001


def test_harness_should_repair_on_qa_or_staging_failure():
    ok_result = {"ok": True, "runtime_ok": True}
    assert harness.harness_should_repair(ok_result, qa={"ok": True}, staging={"staged": True}) == (False, "ok")
    assert harness.harness_should_repair(ok_result, qa={"ok": False}, staging={"staged": True}) == (True, "qa_failed")
    assert harness.harness_should_repair(ok_result, qa={"ok": True}, staging={"staged": False}) == (True, "staging_failed")
    assert harness.harness_should_repair(
        {"ok": True, "runtime_ok": False}, qa=None, staging={"staged": True},
    ) == (True, "runtime_degraded")
    assert harness.harness_should_repair({"ok": False}, qa={"ok": False}) == (False, "pipeline_failed")


def test_harness_loop_skips_when_qa_ok(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    pipeline = {"ok": True, "runtime_ok": True, "run_dir": str(run_dir)}

    loop = harness.harness_loop(str(run_dir), {}, pipeline_result=pipeline, staging={"staged": True})

    assert loop["repaired"] is False
    assert loop["reason"] == "ok"


def test_harness_loop_runs_repairs_on_failed_qa(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({
        "ok": False,
        "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high"}],
    }), encoding="utf-8")
    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        (run_dir / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True, "run_dir": rd}

    loop = harness.harness_loop(
        str(run_dir),
        {},
        pipeline_result={"ok": True, "runtime_ok": True},
        staging={"staged": True},
        run_one=fake_run_one,
    )

    assert loop["repaired"] is True
    assert loop["reason"] == "qa_failed"
    assert calls == ["ocr"]
    assert loop["pipeline_result"]["qa_ok"] is True


def test_repair_assess_pairs_with_recommended_resume(tmp_path):
    qa = {
        "text_recall": 0.6,
        "hard_fails": [],
        "per_layer": [],
    }
    repairs = repair.assess({}, qa, {"lines": []}, {"run_dir": str(tmp_path)})
    choice = harness.recommended_resume(repairs)
    assert choice is not None
    assert choice["resume"] == "ocr"
    assert (tmp_path / "repairs.json").exists()


def test_execute_repairs_reads_max_rounds_from_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "qwen", "action": "retry", "reason": "alpha noisy", "severity": "medium"},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}), encoding="utf-8")

    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        return {"ok": True, "run_dir": rd}

    cfg = {"runtime": {"harness": {"max_rounds": 2}}}
    summary = harness.execute_repairs(str(run_dir), cfg, run_one=fake_run_one)

    assert len(calls) == 2
    assert summary["stopped"] == "max_iterations"
