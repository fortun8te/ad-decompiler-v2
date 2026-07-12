import json

from benchmark import _entry, _harness_telemetry, _markdown, select_images
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
        "structural": {
            "missing_assets": ["assets/product.png"],
            "background": {"outside_changed_ratio": 0.0},
            "layer_alpha": [],
            "element_recall": 1.0,
            "hard_fails": [{"rule": "missing-assets", "detail": "product crop absent"}],
        },
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
    for name in (
        "input_manifest.json", "normalized.png", "ocr_raw.json", "ocr.json",
        "residual.json", "qwen.json", "sam3.json", "fused_elements.json",
        "elements.json", "merged.json", "layout.json", "design_preflight.json",
        "background_clean.png", "removal_mask.png", "ownership.png", "layers_contact.png",
        "preview.png", "diff.png",
    ):
        (run / name).write_text("{}", encoding="utf-8")
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


def test_benchmark_selection_is_stable_and_supports_five_image_cap(tmp_path):
    for name in ("03.webp", "01.png", "02.jpg", "05.jpeg", "04.png", "ignored.txt"):
        (tmp_path / name).write_bytes(b"fixture")

    assert [p.name for p in select_images(tmp_path, 5)] == [
        "01.png", "02.jpg", "03.webp", "04.png", "05.jpeg"
    ]


def test_benchmark_entry_rejects_partial_run_even_if_result_claims_success(tmp_path):
    run = _fixture_run(tmp_path)
    (run / "qa.json").unlink()

    row = _entry(run, {"ok": True, "runtime_ok": True})

    assert row["complete"] is False
    assert row["qa_ok"] is False
    assert row["runtime_ok"] is False
    assert "qa.json" in row["missing_artifacts"]


def test_benchmark_fault_injection_cannot_omit_visual_gate_evidence(tmp_path):
    run = _fixture_run(tmp_path, qa={
        "ok": True, "visual_score": 0.99, "ssim": 0.99, "hard_fails": [],
        "structural": {},  # simulates an old/broken GPU smoke omitting mask evidence
    })
    row = _entry(run, {"ok": True})
    assert row["qa_evidence_complete"] is False
    assert row["qa_ok"] is False


def test_benchmark_scorecard_surfaces_each_visual_failure_rule(tmp_path):
    rules = ["inpaint-outside-mask", "layer-alpha-holes", "empty-layer-alpha", "low-element-recall"]
    run = _fixture_run(tmp_path, qa={
        "ok": False, "visual_score": 0.99, "ssim": 0.99,
        "hard_fails": [{"rule": rule, "detail": "injected"} for rule in rules],
        "structural": {"background": {}, "layer_alpha": [], "element_recall": 0.25,
                       "hard_fails": [{"rule": rule, "detail": "injected"} for rule in rules]},
    })
    row = _entry(run, {"ok": True})
    assert set(row["visual_failure_rules"]) == set(rules)
    assert row["inpaint_outside_mask"] is True
    assert row["layer_alpha_holes"] is True
    assert row["empty_layer_alpha"] is True
    assert row["low_element_recall"] is True
    assert "inpaint-outside-mask" in _markdown({"summary": {"images": 1, "qa_passing": 0}, "runs": [row]})
