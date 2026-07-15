import json
import pytest

from benchmark import (
    _entry, _harness_telemetry, _markdown, _source_manifest,
    configure_auto_repair, configure_figma_acceptance, parse_fixture_ids,
    requires_runtime_smoke, select_images,
)
from src import harness
from src.harness import harness_enabled


def test_real_model_configs_cannot_skip_benchmark_runtime_smoke():
    assert requires_runtime_smoke({"runtime": {"require_active_models": True}}) is True
    assert requires_runtime_smoke({"inpaint": {"strict_acceptance": True}}) is True
    assert requires_runtime_smoke({"runtime": {"require_active_models": False},
                                   "inpaint": {"strict_acceptance": False}}) is False


def test_figma_acceptance_forces_fresh_plugin_evidence_and_a_nonzero_wait():
    cfg = {"figma": {"enabled": False, "require_export": False}, "export_wait_s": 0}

    configure_figma_acceptance(cfg, 0)

    assert cfg["figma"] == {"enabled": True, "require_export": True}
    assert cfg["export_wait_s"] == 1


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


def test_no_auto_repair_forces_legacy_and_harness_switches_off():
    cfg = {"runtime": {"auto_repair": True, "harness": {"enabled": True}}}
    configure_auto_repair(cfg, False)
    assert cfg["runtime"]["auto_repair"] is False
    assert cfg["runtime"]["harness"]["enabled"] is False
    assert harness_enabled(cfg) is False


def test_harness_max_rounds_defaults_to_two():
    assert harness.harness_max_rounds({}) == 2
    assert harness.harness_max_rounds({"runtime": {"harness": {"max_rounds": 5}}}) == 5


def test_benchmark_selection_is_stable_and_supports_five_image_cap(tmp_path):
    for name in ("03.webp", "01.png", "02.jpg", "05.jpeg", "04.png", "ignored.txt"):
        (tmp_path / name).write_bytes(b"fixture")

    assert [p.name for p in select_images(tmp_path, 5)] == [
        "01.png", "02.jpg", "03.webp", "04.png", "05.jpeg"
    ]


def test_benchmark_selects_requested_ids_in_request_order_and_normalizes_padding(tmp_path):
    for name in ("026_first.webp", "034_second.png", "103_third.jpg", "ignored.txt"):
        (tmp_path / name).write_bytes(name.encode())

    ids = parse_fixture_ids(["103,26", "034"])
    assert ids == ["103", "026", "034"]
    assert [p.name for p in select_images(tmp_path, fixture_ids=ids)] == [
        "103_third.jpg", "026_first.webp", "034_second.png",
    ]


def test_benchmark_named_selection_fails_on_missing_or_ambiguous_ids(tmp_path):
    (tmp_path / "026_a.webp").write_bytes(b"a")
    with pytest.raises(ValueError, match="missing fixture IDs: 034"):
        select_images(tmp_path, fixture_ids=["034"])

    (tmp_path / "026_b.png").write_bytes(b"b")
    with pytest.raises(ValueError, match="duplicate files for fixture IDs: 026"):
        select_images(tmp_path, fixture_ids=["26"])
    with pytest.raises(ValueError, match="duplicate fixture IDs requested: 026"):
        parse_fixture_ids(["26", "026"])
    with pytest.raises(ValueError, match="cannot truncate named fixtures"):
        select_images(tmp_path, max_images=1, fixture_ids=["026", "034"])


def test_source_manifest_records_requested_resolution_and_sha256(tmp_path):
    image = tmp_path / "026_ad.webp"
    image.write_bytes(b"fixture")
    manifest = _source_manifest([image], ["026"])
    assert manifest["requested_ids"] == ["026"]
    assert manifest["resolved"][0]["id"] == "026"
    assert manifest["resolved"][0]["filename"] == image.name
    assert len(manifest["resolved"][0]["sha256"]) == 64


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


def test_entry_surfaces_existing_archetype_preset_and_regional_dispositions(tmp_path):
    run = _fixture_run(tmp_path)
    design = json.loads((run / "design.json").read_text())
    design["meta"].update({
        "archetype": "social_screenshot", "preset": "instagram_caption",
        "native_leaf_ratio": 0.625,
        "leaf_accounting": {
            "intentional_raster_cluster_count": 2,
            "unexplained_raster_count": 0,
        },
    })
    (run / "design.json").write_text(json.dumps(design), encoding="utf-8")
    reconstruction = json.loads((run / "reconstruction.json").read_text())
    reconstruction["stats"]["inpaint"] = {
        "backend": "regional",
        "regions": [
            {"ids": ["a"], "route": "flux-comfy"},
            {"ids": ["b"], "route": "big-lama", "fallback": True,
             "fallback_reason": "flat-large-hole"},
        ],
    }
    (run / "reconstruction.json").write_text(json.dumps(reconstruction), encoding="utf-8")
    qa = json.loads((run / "qa.json").read_text())
    qa["editable_text_recall"] = 0.875
    (run / "qa.json").write_text(json.dumps(qa), encoding="utf-8")

    row = _entry(run, {"ok": True})
    assert row["archetype"] == "social_screenshot"
    assert row["preset"] == "instagram_caption"
    assert row["editable_text_recall"] == 0.875
    assert row["native_leaf_ratio"] == 0.625
    assert row["intentional_raster_clusters"] == 2
    assert row["unexplained_raster_fallbacks"] == 0
    assert row["regional_inpaint_routes"] == {"flux-comfy": 1, "big-lama": 1}
    assert row["fallback_dispositions"][0]["fallback_reason"] == "flat-large-hole"


def test_element_recall_prefers_the_top_level_qa_json_mirror(tmp_path):
    """pixel_diff now hoists element_recall/element_survival to qa.json's top level (the
    same way editable_text_recall already was); _entry must read that mirror first and
    only fall back to the nested structural.* copy for older run artifacts that predate it.
    """
    run = _fixture_run(tmp_path, qa={
        "ok": True, "visual_score": 0.99, "ssim": 0.99, "hard_fails": [],
        "element_recall": 0.42,
        "element_survival": {"proposed": 12, "kept": 5, "recall": 0.42, "missing_ids": ["E1"]},
        "structural": {
            "background": {}, "layer_alpha": [],
            # A stale/older nested value must lose to the top-level mirror above.
            "element_recall": 1.0,
            "hard_fails": [],
        },
    })
    row = _entry(run, {"ok": True})
    assert row["element_recall"] == 0.42
    assert row["element_survival"]["missing_ids"] == ["E1"]

    # Older run artifacts (no top-level mirror at all) still fall back to structural.*.
    legacy_base = tmp_path / "legacy"
    legacy_base.mkdir()
    legacy_run = _fixture_run(legacy_base, qa={
        "ok": True, "visual_score": 0.99, "ssim": 0.99, "hard_fails": [],
        "structural": {"background": {}, "layer_alpha": [], "element_recall": 0.9, "hard_fails": []},
    })
    legacy_row = _entry(legacy_run, {"ok": True})
    assert legacy_row["element_recall"] == 0.9


def test_true_text_coverage_is_surfaced_in_entry_and_markdown_column(tmp_path):
    """021-style denominator lie: editable_text_recall reads a perfect 1.0 but text_recall
    is low (OCR only found a sliver of the ad's copy). true_text_coverage must be surfaced
    in the per-run entry and rendered as its own benchmark.md column so it can't be missed
    the way a nested-only field could be.
    """
    run = _fixture_run(tmp_path, qa={
        "ok": False, "visual_score": 0.99, "ssim": 0.99,
        "text_recall": 0.17, "editable_text_recall": 1.0, "true_text_coverage": 0.17,
        "hard_fails": [{"rule": "missing-editable-text", "detail": "true text coverage 0.17 < 0.20"}],
        "structural": {
            "background": {}, "layer_alpha": [], "element_recall": 1.0,
            "editable_text_recall": 1.0, "true_text_coverage": 0.17,
            "hard_fails": [{"rule": "missing-editable-text", "detail": "true text coverage 0.17 < 0.20"}],
        },
    })
    row = _entry(run, {"ok": True})
    assert row["true_text_coverage"] == 0.17

    report = {"summary": {"images": 1, "qa_passing": 0}, "runs": [row]}
    md = _markdown(report)
    assert "true text coverage" in md
    assert "0.170" in md
