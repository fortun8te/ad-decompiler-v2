"""Proof for the per-fixture watchdog (benchmark.run_bounded / benchmark._run_fixture).

postfix-benchmark-6 lost all 16 fixtures when fixture 094 wedged on a VLM call:
benchmark.py called run_pipeline.run_one() directly in-process with no timeout, so once
that call stalled there was no way to interrupt it and every later fixture (101, 104,
107, 131, 135) never ran. signal.alarm does not exist on win32, so the fix runs each
fixture in its own child process (multiprocessing, spawn context) that the parent can
actually kill on timeout.

These tests simulate a wedged stage with a target that blocks forever and assert:
  * the watchdog kills it within a bounded wall-clock budget (no multi-second sleeps
    beyond the short timeout itself -- no GPU/network/VLM involved anywhere),
  * the failure is recorded with a stage name + timeout reason,
  * a subsequent fixture still runs to completion (the benchmark "continues").
"""
from __future__ import annotations

import json
import time

import benchmark
from benchmark import _entry, run_bounded


def _wedge_forever(run_dir, output_queue):
    """Module-level (picklable) target standing in for a stuck pipeline stage."""
    time.sleep(9999)


def _quick_ok(run_dir, output_queue):
    """Module-level (picklable) target standing in for a fixture that finishes fine."""
    output_queue.put({"ok": True, "run_dir": run_dir, "runtime_ok": True, "duration_s": 0.01})


def test_watchdog_kills_a_wedged_stage_and_records_the_failure(tmp_path):
    run_dir = tmp_path / "094_attached"
    run_dir.mkdir()
    # Pretend normalize finished (both of its required artifacts exist) and it wedged
    # in ocr -- the next stage in REQUIRED_ARTIFACTS order.
    (run_dir / "input_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "normalized.png").write_bytes(b"fake")

    started = time.monotonic()
    result = run_bounded(_wedge_forever, (str(run_dir),), run_dir, timeout_s=0.5)
    elapsed = time.monotonic() - started

    # Killed promptly -- not left to run out any real multi-second budget.
    assert elapsed < 15.0
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert result["runtime_status"] == "timeout"
    assert "0s timeout" in result["error"] or "timeout" in result["error"]
    # normalize's artifact exists but ocr's does not -- the watchdog should name the
    # next stage (ocr), not silently say "normalize".
    assert result["failed_stage"] == "ocr"

    # Persisted to disk too, so it survives even if the whole process later dies.
    watchdog_path = run_dir / "watchdog_timeout.json"
    assert watchdog_path.is_file()
    on_disk = json.loads(watchdog_path.read_text(encoding="utf-8"))
    assert on_disk["timed_out"] is True
    assert on_disk["failed_stage"] == "ocr"


def test_benchmark_continues_to_the_next_fixture_after_a_kill(tmp_path):
    """The core proof of (a): a wedged fixture must not take the whole benchmark down --
    the very next fixture (simulated here as a second run_bounded call, exactly the shape
    of main()'s per-image loop) must still complete normally."""
    stuck_dir = tmp_path / "094_stuck"
    stuck_dir.mkdir()
    ok_dir = tmp_path / "095_next"
    ok_dir.mkdir()

    stuck_result = run_bounded(_wedge_forever, (str(stuck_dir),), stuck_dir, timeout_s=0.5)
    assert stuck_result["timed_out"] is True

    # The loop in main() does not stop here -- prove the next fixture still runs fine.
    next_result = run_bounded(_quick_ok, (str(ok_dir),), ok_dir, timeout_s=30)
    assert next_result["ok"] is True
    assert next_result.get("timed_out") is not True


def test_entry_surfaces_a_watchdog_timeout_as_a_hard_fail(tmp_path):
    """_entry() (used by benchmark.py's report) must not silently drop a killed fixture --
    the stage name + timeout reason must land in hard_fails, the same column every other
    QA failure already reports through."""
    run = tmp_path / "094_attached_stuck"
    run.mkdir()
    result = {
        "ok": False, "run_dir": str(run), "runtime_ok": False, "runtime_status": "timeout",
        "error": "watchdog: fixture exceeded 3600s timeout while in stage 'ocr'",
        "failed_stage": "ocr", "timed_out": True, "timeout_s": 3600.0,
    }

    row = _entry(run, result)

    assert row["timed_out"] is True
    assert row["failed_stage"] == "ocr"
    assert row["failure_reason"] == result["error"]
    assert row["pipeline_ok"] is False
    assert row["qa_ok"] is False
    assert any(f["rule"] == "fixture-timeout" for f in row["hard_fails"])
    fail = next(f for f in row["hard_fails"] if f["rule"] == "fixture-timeout")
    assert fail["stage"] == "ocr"
