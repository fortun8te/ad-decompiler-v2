import json

from benchmark import _entry, _markdown


def test_benchmark_entry_and_report_capture_hard_fail_evidence(tmp_path):
    run = tmp_path / "fixture"
    run.mkdir()
    (run / "qa.json").write_text(json.dumps({
        "ok": False,
        "visual_score": 0.72,
        "ssim": 0.70,
        "text_recall": 0.80,
        "edge_f1": 0.60,
        "color_similarity": 0.90,
        "hard_fails": [{"rule": "missing-assets", "detail": "product crop absent"}],
        "structural": {"missing_assets": ["assets/product.png"]},
    }))
    (run / "reconstruction.json").write_text(json.dumps({
        "stats": {"duplicates_removed": 2, "vectorized": 1, "vector_fallback": 3},
    }))
    (run / "design.json").write_text(json.dumps({"meta": {"editable_ratio": 0.75}}))
    (run / "runtime_report.json").write_text(json.dumps({
        "status": "degraded", "acceptable": True,
        "degraded": [{"component": "qwen", "reason": "offline", "required": False}],
        "violations": [],
    }))

    row = _entry(run, {"ok": True})

    assert row["duplicate_observations_removed"] == 2
    assert row["runtime_status"] == "degraded"
    assert row["runtime_ok"] is True
    assert row["hard_fails"][0]["rule"] == "missing-assets"
    report = {"summary": {"images": 1, "qa_passing": 0}, "runs": [row]}
    assert "missing-assets" in _markdown(report)
