import json

from benchmark import _entry, _harness_telemetry, _markdown
from src import harness
from src.harness import harness_enabled


def _fixture_run(tmp_path, *, qa=None, harness_loop=None, harness_legacy=None, critic=None, fixer=None):
    run = tmp_path / "fixture"
    run.mkdir()
    (run / "qa.json").write_text(encoding="utf-8", data=json.dumps(qa or {
        "ok": False,
        "visual_score": 0.72,
        "ssim": 0.70,
        "text_recall": 0.80,
        "edge_f1": 0.60,
        "color_similarity": 0.90,
        "hard_fails": [{"rule": "missing-assets", "detail": "product crop absent"}],
        "structural": {"missing_assets": ["assets/product.png"]},
    }))
    (run / "reconstruction.json").write_text(encoding="utf-8", data=json.dumps({
        "stats": {"duplicates_removed": 2, "vectorized": 1, "vector_fallback": 3},
    }))
    (run / "design.json").write_text(encoding="utf-8", data=json.dumps({"meta": {"editable_ratio": 0.75}}))
    (run / "runtime_report.json").write_text(encoding="utf-8", data=json.dumps({
        "status": "degraded", "acceptable": True,
        "degraded": [{"component": "qwen", "reason": "offline", "required": False}],
        "violations": [],
    }))
    if harness_loop is not None:
        (run / "harness_loop.json").write_text(encoding="utf-8", data=json.dumps(harness_loop))
    if harness_legacy is not None:
        (run / "harness.json").write_text(encoding="utf-8", data=json.dumps(harness_legacy))
    if critic is not None:
        (run / "critic.json").write_text(encoding="utf-8", data=json.dumps(critic))
    if fixer is not None:
        (run / "fixer.json").write_text(encoding="utf-8", data=json.dumps(fixer))
    return run


def test_benchmark_entry_and_report_capture_hard_fail_evidence(tmp_path):
    run = _fixture_run(tmp_path)

    row = _entry(run, {"ok": True})

    assert row["duplicate_observations_removed"] == 2
    assert row["runtime_status"] == "degraded"
    assert row["runtime_ok"] is True
    assert row["hard_fails"][0]["rule"] == "missing-assets"
    assert row["harness_rounds"] == 0
    assert row["final_qa_ok"] is False
    assert row["auto_fixed"] is False
    report = {"summary": {"images": 1, "qa_passing": 0}, "runs": [row]}
    assert "missing-assets" in _markdown(report)


def test_harness_telemetry_from_harness_loop(tmp_path):
    run = _fixture_run(tmp_path, harness_loop={
        "stopped": "qa_ok",
        "round_count": 2,
        "initial_qa_ok": False,
        "final_qa_ok": True,
        "auto_fixed": True,
        "rounds": [
            {"round": 1, "qa_ok_before": False, "qa_ok_after": False},
            {"round": 2, "qa_ok_before": False, "qa_ok_after": True},
        ],
    }, critic={"issues": []}, fixer={"applied": ["merge/dedup"]})

    telemetry = _harness_telemetry(run)
    assert telemetry["harness_rounds"] == 2
    assert telemetry["final_qa_ok"] is True
    assert telemetry["auto_fixed"] is True
    assert telemetry["harness_loop_path"] == str(run / "harness_loop.json")
    assert telemetry["critic_path"] == str(run / "critic.json")
    assert telemetry["fixer_path"] == str(run / "fixer.json")
    assert telemetry["harness"]["round_count"] == 2

    row = _entry(run, {"ok": True})
    assert row["harness_rounds"] == 2
    assert row["auto_fixed"] is True
    assert row["final_qa_ok"] is True
    assert row["harness"]["stopped"] == "qa_ok"


def test_harness_telemetry_falls_back_to_legacy_harness_json(tmp_path):
    run = _fixture_run(tmp_path, harness_legacy={
        "stopped": "qa_ok",
        "iterations": 1,
        "qa_ok": True,
        "attempts": [{"iteration": 1, "qa_ok": True}],
    })

    telemetry = _harness_telemetry(run)
    assert telemetry["harness_rounds"] == 1
    assert telemetry["final_qa_ok"] is True
    assert telemetry["auto_fixed"] is True


def test_harness_enabled_defaults_from_config():
    assert harness_enabled({"runtime": {"auto_repair": True}}) is True
    assert harness_enabled({"runtime": {"harness": {"enabled": True}}}) is True
    assert harness_enabled({"runtime": {"harness": {"enabled": False}, "auto_repair": True}}) is False
    assert harness_enabled({}) is False


def test_harness_max_rounds_defaults_to_three():
    assert harness.harness_max_rounds({}) == 3
    assert harness.harness_max_rounds({"runtime": {"harness": {"max_rounds": 5}}}) == 5
