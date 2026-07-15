"""CPU tests for scripts/report_html.py -- builds reports from synthetic run dirs."""
import json
import os
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import report_html  # noqa: E402

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class _BalanceChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.unmatched_close = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass
        else:
            self.unmatched_close.append(tag)


def assert_well_formed(html_text: str):
    assert html_text.lower().startswith("<!doctype html")
    assert "<html" in html_text and "</html>" in html_text
    checker = _BalanceChecker()
    checker.feed(html_text)
    assert checker.unmatched_close == [], f"unmatched close tags: {checker.unmatched_close}"
    assert checker.stack == [], f"unclosed tags: {checker.stack}"


def assert_self_contained(html_text: str):
    lowered = html_text.lower()
    assert "http://" not in lowered and "https://" not in lowered
    assert "<script" not in lowered
    assert "<link" not in lowered
    assert 'src="http' not in lowered


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _complete_run(run_dir: Path, run_id: str, *, qa_ok: bool, per_layer=None,
                  images=("original.png", "preview.png", "diff.png")):
    sub = run_dir / run_id
    sub.mkdir(parents=True, exist_ok=True)
    for name in images:
        (sub / name).write_bytes(b"\x89PNG\r\n")  # existence-only stub
    _write(sub / "qa.json", {
        "ok": qa_ok,
        "visual_score": 0.83,
        "ssim": 0.80,
        "text_recall": 0.72,
        "editable_text_recall": 1.0,
        "edge_f1": 0.66,
        "hard_fails": [] if qa_ok else [{"rule": "edge-fidelity", "detail": "0.37 < 0.55"}],
        "structural": {"element_recall": 1.0},
        "per_layer": per_layer or [],
    })


def _benchmark(run_dir: Path, runs, summary=None):
    _write(run_dir / "benchmark.json", {
        "version": 1,
        "input_dir": "/fixtures/ads",
        "runs": runs,
        "summary": summary or {
            "images": len(runs),
            "qa_passing": sum(1 for r in runs if r.get("qa_ok")),
            "complete_runs": sum(1 for r in runs if r.get("complete")),
            "runtime_accepted": sum(1 for r in runs if r.get("runtime_ok")),
            "mean_visual_score": 0.83,
            "mean_ssim": 0.80,
            "mean_text_recall": 0.72,
            "mean_editable_text_recall": 1.0,
            "mean_edge_f1": 0.66,
            "mean_element_recall": 1.0,
        },
    })


def _entry(run_id, *, qa_ok, complete=True, missing=None, **extra):
    row = {
        "id": run_id,
        "qa_ok": qa_ok,
        "complete": complete,
        "missing_artifacts": missing or [],
        "runtime_ok": qa_ok,
        "runtime_status": "ok",
        "duration_s": 42.0,
        "visual_score": 0.83,
        "ssim": 0.80,
        "text_recall": 0.72,
        "editable_text_recall": 1.0,
        "edge_f1": 0.66,
        "element_recall": 1.0,
        "regional_inpaint_routes": {"analytic-affine": 3},
        "hard_fails": [] if qa_ok else [{"rule": "edge-fidelity", "detail": "0.37 < 0.55"}],
    }
    row.update(extra)
    return row


def _mini_run(base: Path) -> Path:
    """A run dir: one passing run (with per-layer scores) and one partial run."""
    run_dir = base / "run-after"
    _complete_run(run_dir, "001_pass", qa_ok=True, per_layer=[
        {"id": "headline", "type": "text", "role": "headline", "ssim": 0.91, "recall": 0.88, "score": 0.88},
        {"id": "cta", "type": "text", "role": "button", "ssim": 0.72, "recall": 0.95, "score": 0.72},
    ])
    # Partial run: missing preview/diff images + a missing artifact recorded.
    _complete_run(run_dir, "002_partial", qa_ok=False, images=("original.png",))
    _benchmark(run_dir, [
        _entry("001_pass", qa_ok=True),
        _entry("002_partial", qa_ok=False, complete=False,
               missing=["preview.png", "diff.png"]),
    ])
    return run_dir


def test_generate_report_writes_file_and_returns_path(tmp_path):
    run_dir = _mini_run(tmp_path)
    out = report_html.generate_report(run_dir)
    assert out == run_dir / "report.html"
    assert out.is_file()
    assert out.stat().st_size > 2000


def test_report_is_well_formed_and_self_contained(tmp_path):
    run_dir = _mini_run(tmp_path)
    text = report_html.build_html(run_dir)
    assert_well_formed(text)
    assert_self_contained(text)
    assert "Benchmark report" in text


def test_report_shows_pass_fail_and_warning_chips(tmp_path):
    run_dir = _mini_run(tmp_path)
    text = report_html.build_html(run_dir)
    assert "chip-pass" in text and ">PASS<" in text
    assert "chip-fail" in text and ">FAIL<" in text
    # Partial run -> warning chip listing the missing artifacts.
    assert "chip-warn" in text
    assert "incomplete" in text
    assert "preview.png" in text and "diff.png" in text


def test_report_embeds_triptych_and_flags_missing_images(tmp_path):
    run_dir = _mini_run(tmp_path)
    text = report_html.build_html(run_dir)
    # Passing run references all three images by relative path.
    assert 'src="001_pass/original.png"' in text
    assert 'src="001_pass/preview.png"' in text
    assert 'src="001_pass/diff.png"' in text
    # Partial run keeps the original but marks the missing slots.
    assert 'src="002_partial/original.png"' in text
    assert "img-missing" in text
    assert "missing" in text


def test_report_renders_metric_row_and_hard_fails(tmp_path):
    run_dir = _mini_run(tmp_path)
    text = report_html.build_html(run_dir)
    for label in ("Visual", "Text", "Editable", "Edge", "Element recall", "Regional routes", "Hard fails"):
        assert label in text
    assert "analytic-affine" in text
    assert "edge-fidelity" in text  # hard-fail pill on the failing run


def test_per_layer_details_render_when_present_and_absent(tmp_path):
    run_dir = _mini_run(tmp_path)
    text = report_html.build_html(run_dir)
    # Present on the passing run.
    assert "Per-layer detail" in text
    assert "headline" in text and "button" in text
    assert "<table" in text
    # Absent on the partial run -> graceful message, no crash.
    assert "No per-layer scores recorded" in text


def test_baseline_deltas_render(tmp_path):
    after = _mini_run(tmp_path)
    before = tmp_path / "run-before"
    _complete_run(before, "001_pass", qa_ok=True)
    _benchmark(before, [_entry("001_pass", qa_ok=True, visual_score=0.60)],
               summary={"images": 1, "qa_passing": 1, "complete_runs": 1, "runtime_accepted": 1,
                        "mean_visual_score": 0.60, "mean_edge_f1": 0.40})
    text = report_html.build_html(after, baseline_dir=before)
    assert "Deltas shown vs baseline" in text
    assert "run-before" in text
    assert "delta-up" in text  # 0.83 > 0.60 => improvement arrow


def test_orphan_subdir_is_surfaced_with_warning(tmp_path):
    run_dir = tmp_path / "run-orphan"
    _complete_run(run_dir, "001_pass", qa_ok=True)
    _benchmark(run_dir, [_entry("001_pass", qa_ok=True)])
    # A stray subdir with a qa.json but no benchmark.json entry (aborted/partial run).
    _complete_run(run_dir, "099_stray", qa_ok=False, images=("original.png",))
    text = report_html.build_html(run_dir)
    assert "099_stray" in text
    assert "orphan run" in text
    assert_well_formed(text)


def test_missing_benchmark_json_still_builds_from_qa(tmp_path):
    run_dir = tmp_path / "run-no-manifest"
    _complete_run(run_dir, "001_pass", qa_ok=True)
    _complete_run(run_dir, "002_fail", qa_ok=False)
    text = report_html.build_html(run_dir)  # no benchmark.json present
    assert_well_formed(text)
    assert "001_pass" in text and "002_fail" in text
    assert "QA passing" in text


def test_generate_report_rejects_non_directory(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(NotADirectoryError):
        report_html.generate_report(missing)


def test_cli_main_writes_report(tmp_path, capsys):
    run_dir = _mini_run(tmp_path)
    rc = report_html.main([str(run_dir)])
    assert rc == 0
    assert (run_dir / "report.html").is_file()
    out = capsys.readouterr().out
    assert "report.html" in out


def test_benchmark_hook_emits_report(tmp_path):
    """benchmark._emit_html_report wires the generator without importing heavy deps."""
    import benchmark
    run_dir = _mini_run(tmp_path)
    benchmark._emit_html_report(run_dir)
    assert (run_dir / "report.html").is_file()
