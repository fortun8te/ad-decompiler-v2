"""Proof for (b) summary-report-on-finish-or-abort and (c) the .activity_current pointer.

postfix-benchmark-6 never wrote benchmark.json/.md at all: the only write happened after
the full per-image loop completed, so a run that died mid-loop (as bench-6 did on fixture
094) left nothing behind but pipeline.log files. These tests drive benchmark.main() with
every real model/service call faked out (no GPU/network/VLM, no real pipeline stages) and
assert that a run which raises partway through still leaves a partial/aborted summary on
disk, and that .activity_current still correctly names the run dir afterwards -- both for
a mid-loop abort and for a second, resumed invocation against the same --output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

import benchmark


def _tiny_png(path: Path) -> None:
    Image.new("RGB", (8, 8), "white").save(path)


def _patch_common(monkeypatch):
    # No GPU/network involved: fake out the two service-readiness calls main() makes
    # before it ever touches an image. skip_doctor covers the third (doctor.inspect).
    monkeypatch.setattr("src.runtime_bootstrap.ensure_services",
                        lambda cfg: {"ok": True, "checks": []})
    monkeypatch.setattr(benchmark, "_emit_html_report", lambda output: None)


def _make_input_dir(tmp_path) -> Path:
    input_dir = tmp_path / "ads"
    input_dir.mkdir()
    _tiny_png(input_dir / "001_a.png")
    _tiny_png(input_dir / "002_b.png")
    return input_dir


def test_main_writes_a_partial_summary_and_holds_the_pointer_on_a_mid_loop_abort(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output = tmp_path / "runs" / "bench"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    _patch_common(monkeypatch)

    calls = {"n": 0}

    def fake_run_fixture(image, run_dir, cfg, resume, timeout_s):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated wedge the watchdog could not absorb")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "run_dir": str(run_dir), "runtime_ok": True, "duration_s": 0.01}

    monkeypatch.setattr(benchmark, "_run_fixture", fake_run_fixture)
    monkeypatch.setattr(sys, "argv", [
        "benchmark.py", "--input-dir", str(input_dir), "--output", str(output),
        "--config", str(cfg_path), "--skip-doctor",
    ])

    with pytest.raises(RuntimeError, match="simulated wedge"):
        benchmark.main()

    # (b) a summary exists even though the run never finished.
    report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert report["partial"] is True
    assert "simulated wedge" in report["aborted_reason"]
    assert report["fixtures_completed"] == 1
    assert report["fixtures_planned"] == 2
    assert report["wall_time_s"] >= 0
    assert (output / "benchmark.md").is_file()
    assert (output / "text_contract_report.json").is_file()

    # (c) the dashboard pointer still correctly names this (now-dead) run dir.
    pointer = output.resolve().parent / ".activity_current"
    assert pointer.is_file()
    assert Path(pointer.read_text(encoding="utf-8").strip()) == output.resolve()


def test_main_holds_the_pointer_and_refreshes_the_summary_on_a_resumed_invocation(tmp_path, monkeypatch):
    """(c): re-running benchmark.py against the SAME --output (a "resume") must still
    leave .activity_current naming that run dir, and the summary must reflect the
    resumed invocation's results, not go stale or point elsewhere."""
    input_dir = _make_input_dir(tmp_path)
    output = tmp_path / "runs" / "bench"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    _patch_common(monkeypatch)

    def make_fake_run_fixture(ok_ids):
        def fake_run_fixture(image, run_dir, cfg, resume, timeout_s):
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            ok = image.stem in ok_ids
            return {"ok": ok, "run_dir": str(run_dir), "runtime_ok": ok, "duration_s": 0.01,
                    **({} if ok else {"error": "boom", "failed_stage": "ocr"})}
        return fake_run_fixture

    argv = ["benchmark.py", "--input-dir", str(input_dir), "--output", str(output),
            "--config", str(cfg_path), "--skip-doctor"]
    monkeypatch.setattr(sys, "argv", argv)

    # First invocation: one fixture fails.
    monkeypatch.setattr(benchmark, "_run_fixture", make_fake_run_fixture({"001_a"}))
    with pytest.raises(SystemExit):
        benchmark.main()
    pointer = output.resolve().parent / ".activity_current"
    assert Path(pointer.read_text(encoding="utf-8").strip()) == output.resolve()
    first_report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert first_report["partial"] is False  # loop reached the end; not partial
    assert first_report["summary"]["complete_runs"] == 0  # no run wrote real artifacts

    # Second ("resumed") invocation against the same --output: both fixtures now succeed.
    monkeypatch.setattr(benchmark, "_run_fixture", make_fake_run_fixture({"001_a", "002_b"}))
    with pytest.raises(SystemExit):
        benchmark.main()

    # The pointer still names this same run dir -- it was not lost or redirected.
    assert Path(pointer.read_text(encoding="utf-8").strip()) == output.resolve()
    second_report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert second_report["fixtures_completed"] == 2
    assert second_report["aborted_reason"] is None


def test_run_text_contract_check_skips_a_fixture_that_never_reached_design(tmp_path):
    """A fixture the watchdog killed before the design stage has no design.json yet --
    there is nothing to contract-check, matching text_contract_check.py's own CLI
    behaviour of skipping run dirs without one."""
    run_dir = tmp_path / "094_stuck"
    run_dir.mkdir()
    assert benchmark._run_text_contract_check(run_dir) is None


def test_run_text_contract_check_returns_check_run_shape_for_a_reached_fixture(tmp_path):
    run_dir = tmp_path / "001_a"
    run_dir.mkdir()
    (run_dir / "design.json").write_text(json.dumps({
        "canvas": {"w": 100, "h": 100}, "layers": [], "kept_in_photo": [],
    }), encoding="utf-8")
    (run_dir / "ocr.json").write_text(json.dumps({"lines": []}), encoding="utf-8")

    rep = benchmark._run_text_contract_check(run_dir)

    assert rep is not None
    assert rep["fixture"] == "001_a"
    assert rep["violations"] == []
    assert rep["nodes"] == 0
    assert rep["source_lines"] == 0


def test_build_contract_report_aggregates_hard_and_warn_counts():
    reports = [
        {"fixture": "a", "violations": [{"severity": "HARD"}, {"severity": "WARN"}]},
        {"fixture": "b", "violations": [{"severity": "WARN"}]},
        {"fixture": "c", "violations": []},
    ]

    rep = benchmark._build_contract_report(reports)

    assert rep["checked"] == 3
    assert rep["hard_total"] == 1
    assert rep["warn_total"] == 2
    assert rep["runs"] == reports


def test_infer_wedged_stage_reports_qa_when_every_artifact_is_present(tmp_path):
    run_dir = tmp_path / "091_done"
    run_dir.mkdir()
    for name in benchmark.REQUIRED_ARTIFACTS:
        (run_dir / name).write_bytes(b"x")

    assert benchmark._infer_wedged_stage(run_dir) == "qa"
