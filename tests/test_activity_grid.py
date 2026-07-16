"""Activity grid: timers, planned latch, and watch-root resolution."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.activity_grid import (
    Tracker,
    estimate_started_at,
    load_planned,
    merge_progress,
    parse_done_seconds,
    parse_log,
    read_activity_pointer,
    read_bench_summary,
    read_contract_check,
    resolve_watch_root,
    write_activity_pointer,
)


def _write_log(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


COMPLETE_LOG = """\
[00:46:12] normalize → 1080x1920
[00:48:19] ocr[doctr] → 12 lines
[00:50:00] text analysis → 6 blocks
[00:50:01] residual proposals → 2
[00:50:02] qwen → 0 layers
[00:51:00] sam3[ok] → 4 observations
[00:51:10] element fusion → 4 canonical
[00:52:00] peel → 4 complete layers
[00:52:01] merge → 4 candidates
[00:52:02] structure → 1 root layers
[00:53:00] reconstruct → 4 entities
[00:54:00] layout → ok
[00:55:00] design.json → 4 layers
[00:56:00] preview → preview.png
[00:57:00] figma import: Import latest
[00:57:10] export: figma_export.png
[00:57:40] qa → ssim=0.9
[00:57:48] done in 212.9s
"""

PARTIAL_LOG = """\
[01:12:31] normalize → 338x600
[01:12:40] ocr[doctr] → 8 lines
[01:13:00] text analysis → 4 blocks
[01:13:01] residual proposals → 1
[01:13:02] qwen → 0 layers
[01:13:20] sam3[ok] → 3 observations
[01:13:30] element fusion → 3 canonical
[01:13:41] peel → 3 complete layers
[01:13:41] merge → 3 candidates
[01:13:41] structure → 1 root layers
[01:13:43] reconstruct → 3 entities, background=regional
"""


def test_parse_done_seconds_and_complete():
    done, active, complete, failed = parse_log(COMPLETE_LOG)
    assert complete is True
    assert active is None
    assert failed is False
    assert done[-1] == "qa"
    assert parse_done_seconds(COMPLETE_LOG) == 212.9


def test_harness_resume_reopens_after_done():
    text = COMPLETE_LOG + "[00:58:00] resuming from reconstruct\n[00:58:01] reconstruct → 4 entities\n"
    done, active, complete, failed = parse_log(text)
    assert complete is False
    assert "reconstruct" in done
    assert active == "layout"


def test_estimate_started_at_uses_log_span():
    now = 1_700_000_000.0
    log_mtime = now
    started = estimate_started_at(COMPLETE_LOG, log_mtime, now)
    # 00:46:12 → 00:57:48 = 696s
    assert started == pytest.approx(now - 696.0, abs=1.0)


def test_frozen_elapsed_uses_log_span_not_last_harness_done(tmp_path: Path):
    run = tmp_path / "bench" / "002_ad"
    # Two done-in lines: short harness re-run must not win over wall span.
    body = COMPLETE_LOG + "[00:58:00] done in 3.6s\n"
    _write_log(run / "pipeline.log", body)
    t = Tracker(tmp_path / "bench", auto_root=False)
    t.refresh()
    snap = t.snapshot()
    runs = snap["benchmark"]["runs"]
    assert len(runs) == 1
    assert runs[0]["complete"] is True
    # 00:46:12 → 00:58:00 = 708s
    assert runs[0]["elapsed_s"] == 708.0
    time.sleep(0.05)
    t.refresh()
    assert t.snapshot()["benchmark"]["runs"][0]["elapsed_s"] == 708.0


def test_frozen_elapsed_falls_back_to_done_in_without_span(tmp_path: Path):
    run = tmp_path / "bench" / "solo"
    _write_log(run / "pipeline.log", "[12:00:00] qa → ok\n[12:00:00] done in 42.0s\n")
    t = Tracker(tmp_path / "bench", auto_root=False)
    t.refresh()
    assert t.snapshot()["benchmark"]["runs"][0]["elapsed_s"] == 42.0


def test_running_elapsed_does_not_use_stale_mtime_alone(tmp_path: Path):
    run = tmp_path / "bench" / "021_ad"
    _write_log(run / "pipeline.log", PARTIAL_LOG)
    t = Tracker(tmp_path / "bench", auto_root=False)
    t.refresh()
    r0 = t.snapshot()["benchmark"]["runs"][0]
    assert r0["status"] in ("running", "stalled")
    assert r0["complete"] is False
    assert r0["started_at"] is not None
    # Elapsed should reflect log span (~72s), not "hours since file birth".
    assert 30.0 <= r0["elapsed_s"] <= 600.0


def test_resolve_watch_root_follows_pointer(tmp_path: Path):
    old = tmp_path / "benchmark-old"
    new = tmp_path / "benchmark-new"
    old.mkdir()
    new.mkdir()
    _write_log(old / "a" / "pipeline.log", PARTIAL_LOG)
    (new / "planned.json").write_text(
        json.dumps({"images": [{"id": "x"}, {"id": "y"}]}),
        encoding="utf-8",
    )
    (new / "x").mkdir()
    (new / "y").mkdir()
    write_activity_pointer(new)
    assert resolve_watch_root(old) == new.resolve()
    assert resolve_watch_root(tmp_path) == new.resolve()
    assert read_activity_pointer(old) == new.resolve()


def test_resolve_watch_root_planned_only_without_logs(tmp_path: Path):
    stale = tmp_path / "stale-bench"
    fresh = tmp_path / "fresh-bench"
    stale.mkdir()
    fresh.mkdir()
    _write_log(stale / "old" / "pipeline.log", COMPLETE_LOG)
    # Make stale log look older than planned.json
    old_mtime = time.time() - 3600
    log = stale / "old" / "pipeline.log"
    import os
    os.utime(log, (old_mtime, old_mtime))
    (fresh / "planned.json").write_text(
        json.dumps({"images": ["ad1", "ad2", "ad3"]}),
        encoding="utf-8",
    )
    time.sleep(0.02)
    assert resolve_watch_root(tmp_path) == fresh.resolve()


def test_tracker_merges_planned_stubs(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "planned.json").write_text(
        json.dumps({"images": [{"id": "001_a"}, {"id": "002_b"}]}),
        encoding="utf-8",
    )
    (bench / "001_a").mkdir()
    (bench / "002_b").mkdir()
    _write_log(bench / "001_a" / "pipeline.log", PARTIAL_LOG)
    t = Tracker(bench, auto_root=False)
    t.refresh()
    runs = t.snapshot()["benchmark"]["runs"]
    assert [r["id"] for r in runs] == ["001_a", "002_b"]
    assert runs[0]["status"] in ("running", "stalled")
    assert runs[1]["status"] == "pending"
    assert load_planned(bench) == ["001_a", "002_b"]


def test_merge_progress_ignores_stale_artifacts_far_ahead():
    done, active, complete = merge_progress(
        ["normalize", "ocr"], "text", False, art_idx=14
    )
    assert complete is False
    assert done == ["normalize", "ocr"]
    assert active == "text"


def test_read_bench_summary_absent_before_benchmark_json_exists(tmp_path: Path):
    assert read_bench_summary(tmp_path) is None


def test_read_bench_summary_surfaces_a_partial_aborted_run(tmp_path: Path):
    # Shape benchmark.py's _build_report writes on a mid-loop abort (see
    # tests/test_benchmark_reporting.py for the producer side).
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "partial": True, "aborted_reason": "RuntimeError: simulated wedge",
        "wall_time_s": 12.5, "fixtures_planned": 16, "fixtures_completed": 9,
        "summary": {"images": 9, "qa_passing": 5, "mean_ssim": 0.83},
    }), encoding="utf-8")

    summary = read_bench_summary(tmp_path)

    assert summary["partial"] is True
    assert summary["aborted_reason"] == "RuntimeError: simulated wedge"
    assert summary["fixtures_planned"] == 16
    assert summary["fixtures_completed"] == 9
    assert summary["summary"]["mean_ssim"] == 0.83


def test_read_contract_check_absent_and_present(tmp_path: Path):
    assert read_contract_check(tmp_path) is None
    (tmp_path / "text_contract_report.json").write_text(json.dumps({
        "checked": 9, "hard_total": 2, "warn_total": 5, "runs": [],
    }), encoding="utf-8")

    check = read_contract_check(tmp_path)

    assert check["checked"] == 9
    assert check["hard_total"] == 2
    assert check["warn_total"] == 5


def test_tracker_snapshot_surfaces_summary_and_contract_check(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    _write_log(bench / "001_a" / "pipeline.log", PARTIAL_LOG)
    (bench / "benchmark.json").write_text(json.dumps({
        "partial": True, "aborted_reason": None, "wall_time_s": 1.0,
        "fixtures_planned": 1, "fixtures_completed": 1,
        "summary": {"images": 1},
    }), encoding="utf-8")
    (bench / "text_contract_report.json").write_text(json.dumps({
        "checked": 1, "hard_total": 0, "warn_total": 1, "runs": [],
    }), encoding="utf-8")

    t = Tracker(bench, auto_root=False)
    t.refresh()
    snap = t.snapshot()

    assert snap["summary"]["fixtures_completed"] == 1
    assert snap["contract_check"]["warn_total"] == 1


def test_tracker_snapshot_summary_is_none_before_benchmark_json_exists(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    _write_log(bench / "001_a" / "pipeline.log", PARTIAL_LOG)
    t = Tracker(bench, auto_root=False)
    t.refresh()
    snap = t.snapshot()
    assert snap["summary"] is None
    assert snap["contract_check"] is None
