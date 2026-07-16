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


def test_resolve_fonts_patch_enables_aggressive_rematch():
    patches = harness.config_patches_for({
        "stage": "text-analysis", "action": "resolve-fonts",
    })
    fm = patches["text_analysis"]["font_matching"]
    assert fm["enabled"] is True
    assert fm["repair_pass"] is True
    assert fm["max_fonts"] == 96
    assert fm["max_lines"] == 24
    assert patches["vlm"]["font_judge"]["enabled"] is True


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

    assert len(calls) == 1
    assert calls[0]["start_from"] == "ocr"
    assert calls[0]["cfg"]["ocr"]["retry_2x"]["enabled"] is True
    assert summary["stopped"] == "all_repairs_failed"
    assert summary["qa_ok"] is False
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

    assert len(calls) == 1
    assert all(stage == "qwen" for stage in calls)
    assert summary["stopped"] == "all_repairs_failed"
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
    # postfix-benchmark-6: this used to assert mode == "auto". That patch was the bug —
    # config.yaml ships mode=flux_comfy and the regional router resolves "auto" to the
    # same per-region engines, so the rerun replayed a byte-identical plate (002/013/066/
    # 091 all logged metric_deltas of exactly 0.0). rebuild-clean-plate now escalates the
    # levers that physically move residue: the removal-mask footprint and the scrub pass.
    inpaint = harness.config_patches_for({
        "stage": "inpaint", "action": "rebuild-clean-plate", "params": {},
    })["inpaint"]
    assert inpaint["mask_dilate"]["text"] > 2        # widen past the default halo
    assert inpaint["multipass_fraction"] < 0.12      # scrub harder than the default
    assert harness.patch_reaches_pipeline({"inpaint": inpaint})

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


def test_harness_gating_rejects_stale_or_string_false_qa_summaries():
    """A completed pipeline must not look ready when its QA summary says false."""
    assert harness.harness_should_repair(
        {"ok": True, "runtime_ok": True, "qa_ok": False}, qa=None,
        staging={"staged": True},
    ) == (True, "qa_failed")
    assert harness.harness_should_repair(
        {"ok": True, "runtime_ok": True}, qa={"ok": "false"},
        staging={"staged": True},
    ) == (True, "qa_failed")


def test_execute_repairs_does_not_treat_string_false_as_already_ok(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "qa.json").write_text(json.dumps({"ok": "false"}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([]), encoding="utf-8")
    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=1)
    # The fixture intentionally has no input image, so execution stops before
    # loading repair actions; the important invariant is that string "false"
    # is not treated as an already-OK QA result.
    assert summary["stopped"] == "missing_input"
    assert summary["qa_ok"] is False


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
        "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}}],
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

    assert len(calls) == 1
    assert summary["stopped"] == "all_repairs_failed"


def test_execute_repairs_tries_alternative_after_pipeline_exception(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}},
        {"stage": "qwen", "action": "retry", "severity": "medium"},
    ]), encoding="utf-8")
    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        if start_from == "ocr":
            raise RuntimeError("ocr backend crashed")
        (run_dir / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    summary = harness.execute_repairs(
        str(run_dir), {}, max_iterations=3, run_one=fake_run_one)

    assert calls == ["ocr", "qwen"]
    assert summary["qa_ok"] is True
    assert summary["attempts"][0]["pipeline_error"] == "ocr backend crashed"
    assert json.loads((run_dir / "harness.json").read_text())["qa_ok"] is True


def test_execute_repairs_tries_alternative_when_runner_leaves_stale_qa(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}))
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}))
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}},
        {"stage": "qwen", "action": "retry", "severity": "medium"},
    ]))
    calls = []

    def runner(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        if start_from == "qwen":
            (run_dir / "qa.json").write_text(json.dumps({"ok": True}))
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=3, run_one=runner)
    assert calls == ["ocr", "qwen"]
    assert summary["qa_ok"] is True
    assert summary["attempts"][0]["qa_fresh"] is False


def test_execute_repairs_switches_tactic_when_metrics_stagnate(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}))
    base_qa = {"ok": False, "ssim": 0.48, "visual_score": 0.50, "text_recall": 0.0,
               "hard_fails": [{"rule": "local-ssim", "detail": "low"}]}
    (run_dir / "qa.json").write_text(json.dumps(base_qa))
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}},
        {"stage": "vlm", "action": "boost-stack", "severity": "medium", "params": {"focus": "text"}},
    ]))
    calls = []

    def runner(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        rewritten = dict(base_qa)
        rewritten["attempt_marker"] = len(calls)  # fresh file, identical quality
        (run_dir / "qa.json").write_text(json.dumps(rewritten))
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=3, run_one=runner)
    assert calls == ["ocr", "text"]
    assert summary["qa_ok"] is False
    assert summary["stopped"] == "all_repairs_failed"
    assert all(attempt["qa_improved"] is False for attempt in summary["attempts"])


def test_execute_repairs_supports_three_argument_runner(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}},
    ]), encoding="utf-8")
    calls = []

    def three_arg_runner(path, rd, cfg):
        calls.append((path, rd, cfg))
        (run_dir / "qa.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    summary = harness.execute_repairs(
        str(run_dir), {}, max_iterations=1, run_one=three_arg_runner)

    assert len(calls) == 1
    assert summary["qa_ok"] is True


def test_qa_progress_ignores_missing_baseline_metrics():
    before = {"ok": False, "ssim": 0.5}
    after = {"ok": False, "ssim": 0.5, "text_recall": 0.4}
    improved, deltas = harness._qa_progress(before, after)
    assert improved is False
    assert deltas.get("text_recall") is None


def test_harness_should_repair_on_actionable_repairs_when_qa_ok():
    ok_result = {"ok": True, "runtime_ok": True}
    qa = {
        "ok": True,
        "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}}],
    }
    assert harness.harness_should_repair(ok_result, qa=qa, staging={"staged": True}) == (
        True, "actionable_repairs",
    )


def test_execute_repairs_continues_when_qa_ok_but_repairs_remain(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(json.dumps({"input": str(input_path)}))
    (run_dir / "qa.json").write_text(json.dumps({
        "ok": True, "text_recall": 0.5,
        "repairs": [
            {"stage": "ocr", "action": "rerun", "severity": "high", "params": {"upscale": True}},
            {"stage": "text-analysis", "action": "resolve-fonts", "severity": "medium"},
        ],
    }), encoding="utf-8")
    calls = []

    def runner(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        (run_dir / "qa.json").write_text(json.dumps({
            "ok": True, "text_recall": 0.95, "repairs": [],
        }), encoding="utf-8")
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=2, run_one=runner)
    assert calls == ["ocr"]
    assert summary["stopped"] == "qa_ok"


# ── Workstream E: admission for baked chrome / kept_in_photo / glyph residue ─────────

def test_admission_rejects_baked_chrome_text_promote(tmp_path):
    design = {"layers": [{
        "id": "c_seal", "type": "image",
        "meta": {"role": "badge", "shell_raster_chip": True, "baked_badge_text": True},
    }]}
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    repair = {
        "stage": "text-analysis", "action": "restore-editable-text",
        "target_id": "c_seal", "severity": "high",
    }
    assert harness.admission_reject_reason(repair, run_dir=str(tmp_path), design=design)
    assert "baked-chrome" in harness.admission_reject_reason(
        repair, run_dir=str(tmp_path), design=design)


def test_admission_rejects_already_sliced_and_kept_in_photo(tmp_path):
    design = {"layers": [
        {"id": "c_slice", "type": "image",
         "meta": {"fallback": "raster-slice", "fallback_scores": {"region_ssim": 0.4}}},
        {"id": "c_pack", "type": "text", "kept_in_photo": True,
         "meta": {"kept_in_photo": True, "baked_owner_id": "c_product"}},
    ]}
    sliced = {"stage": "reconstruct", "action": "inspect-worst-regions",
              "target_id": "c_slice",
              "params": {"regions": [{"layer_id": "c_slice"}]}}
    kip = {"stage": "ocr", "action": "rerun", "target_id": "c_pack",
           "params": {"upscale": True}}
    assert harness.admission_reject_reason(sliced, design=design).startswith("already-sliced")
    assert harness.admission_reject_reason(kip, design=design).startswith("kept-in-photo")


def test_execute_repairs_skips_baked_chrome_without_pipeline_rerun(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    repairs = [{
        "stage": "reconstruct", "action": "inspect-worst-regions",
        "target_id": "c_chip", "severity": "high",
        "params": {"regions": [{"layer_id": "c_chip",
                                "box": {"x": 0, "y": 0, "w": 10, "h": 10}}]},
    }]
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps({
        "layers": [{"id": "c_chip", "type": "image",
                    "meta": {"shell_raster_chip": True, "baked_badge_text": True}}],
    }), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps(repairs), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps({
        "ok": False, "hard_fails": [], "ssim": 0.7, "repairs": repairs,
    }), encoding="utf-8")
    calls = []

    def runner(*_a, **_k):
        calls.append(1)
        return {"ok": True}

    summary = harness.execute_repairs(str(run_dir), {}, max_iterations=2, run_one=runner)
    assert calls == []
    assert any(a.get("admission_skipped") for a in summary["attempts"])
    assert any(a.get("reason") == "baked-or-sliced-deficit" for a in summary["attempts"])


def test_qa_accepts_never_true_over_glyph_residue():
    qa = {
        "ok": True, "ssim": 0.99, "hard_fails": [],
        "contract": {"glyph_residue_clean": False, "pass": False},
        "structural": {"glyph_residue_unresolved": 1},
    }
    assert harness._qa_accepts(qa) is False
    assert harness._qa_accepts({
        "ok": True, "ssim": 0.99,
        "hard_fails": [{"rule": "glyph-residue", "detail": "c_ESSENTIALS"}],
    }) is False


def test_structure_plan_fingerprint_includes_scene_intent(tmp_path):
    (tmp_path / "merged.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scene_intent.json").write_text(
        json.dumps({"planning_fingerprint": "abc123"}), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text("{}", encoding="utf-8")
    choice = {"stage": "layout", "action": "refit-geometry", "resume": "layout",
              "params": {"tighten_containers": True},
              "patches": {"layout": {"min_container_frac": 0.001}}}
    fp1, payload = harness._repair_plan_fingerprint(str(tmp_path), choice)
    assert payload.get("planning_fingerprint") == "abc123"
    assert "scene_intent.json" in payload["inputs"]
    # Unchanged inputs → identical fingerprint (skip structure/VLM resume).
    fp2, _ = harness._repair_plan_fingerprint(str(tmp_path), choice)
    assert fp1 == fp2
