import json

import numpy as np
import pytest

from src.background_benchmark import (
    AcceptanceThresholds,
    evaluate_background,
    generate_synthetic_cases,
    load_bakeoff_config,
    run_inpaint_bakeoff,
    run_synthetic_benchmark,
)


def test_synthetic_pairs_have_known_foreground_only_difference():
    cases = generate_synthetic_cases(cases_per_family=1, size=(160, 128), seed=7)

    assert len(cases) == 4
    for case in cases:
        changed = np.any(case.clean_background != case.composite_input, axis=2)
        assert case.removal_mask.any()
        assert np.all(changed <= case.removal_mask)
        assert np.array_equal(case.clean_background[~case.removal_mask], case.composite_input[~case.removal_mask])


def test_perfect_generated_background_passes_inside_and_byte_exact_outside_gate():
    case = generate_synthetic_cases(cases_per_family=1, size=(160, 128), seed=11)[0]

    result = evaluate_background(
        case.clean_background,
        case.composite_input,
        case.removal_mask,
        case.clean_background,
    )

    assert result["accepted"] is True
    assert result["inside"]["mae"] == 0.0
    assert result["inside"]["psnr_db"] == 99.0
    assert result["outside"]["changed_pixels"] == 0


def test_inside_metrics_ignore_exterior_but_exterior_gate_rejects_damage():
    case = generate_synthetic_cases(cases_per_family=1, size=(160, 128), seed=13)[0]
    candidate = case.clean_background.copy()
    candidate[~case.removal_mask] = 0

    result = evaluate_background(
        case.clean_background,
        case.composite_input,
        case.removal_mask,
        candidate,
    )

    assert result["inside"]["mae"] == 0.0
    assert result["inside_quality_ok"] is True
    assert result["outside_mask_ok"] is False
    assert result["accepted"] is False
    assert result["failure_reasons"] == ["outside-mask"]


def test_wrong_inside_pixels_fail_without_conflating_outside_gate():
    case = generate_synthetic_cases(cases_per_family=1, size=(160, 128), seed=17)[0]
    candidate = case.composite_input.copy()

    result = evaluate_background(
        case.clean_background,
        case.composite_input,
        case.removal_mask,
        candidate,
        AcceptanceThresholds(max_inside_mae=1.0, min_inside_psnr_db=45.0, min_inside_ssim=0.99),
    )

    assert result["inside"]["mae"] > 1.0
    assert result["outside_mask_ok"] is True
    assert result["inside_quality_ok"] is False
    assert result["accepted"] is False


def test_runner_writes_review_artifacts_and_acceptance_summary(tmp_path):
    report = run_synthetic_benchmark(
        tmp_path,
        cases_per_family=1,
        size=(160, 128),
        seed=23,
        method="oracle",
    )

    assert report["summary"]["accepted"] is True
    assert report["summary"]["accepted_cases"] == 4
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["thresholds"]["max_outside_changed_pixels"] == 0
    case_dir = tmp_path / "cases" / "flat-01"
    for name in (
        "clean_background.png", "composite_input.png", "removal_mask.png",
        "generated_background.png", "inside_diff.png", "outside_diff.png", "metrics.json",
    ):
        assert (case_dir / name).is_file()


def test_external_candidates_are_evaluated_untouched_and_missing_is_reported(tmp_path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    cases = generate_synthetic_cases(cases_per_family=1, size=(160, 128), seed=29)
    for case in cases:
        candidate = case.clean_background.copy()
        candidate[0, 0] = (0, 0, 0)  # exterior mutation must survive into the report
        from PIL import Image
        Image.fromarray(candidate).save(candidate_dir / f"{case.case_id}.png")

    report = run_synthetic_benchmark(
        tmp_path / "report",
        cases_per_family=1,
        size=(160, 128),
        seed=29,
        candidate_dir=candidate_dir,
    )
    assert report["summary"]["accepted"] is False
    assert report["summary"]["outside_mask_passing_cases"] < 4
    assert all(item["candidate"]["kind"] == "external" for item in report["cases"])

    with pytest.raises(FileNotFoundError, match="missing generated background"):
        run_synthetic_benchmark(
            tmp_path / "missing",
            cases_per_family=1,
            size=(160, 128),
            seed=29,
            candidate_dir=tmp_path / "missing-candidate",
        )


def test_bakeoff_loads_json_and_yaml_config(tmp_path):
    json_config = tmp_path / "config.json"
    json_config.write_text('{"inpaint": {"allow_fallback": true}}', encoding="utf-8")
    assert load_bakeoff_config(json_config)["inpaint"]["allow_fallback"] is True

    yaml_config = tmp_path / "config.yaml"
    yaml_config.write_text("inpaint:\n  allow_fallback: false\n", encoding="utf-8")
    assert load_bakeoff_config(yaml_config)["inpaint"]["allow_fallback"] is False


def test_bakeoff_runs_actual_inpaint_seam_with_explicit_requested_mode(tmp_path, monkeypatch):
    from src import inpaint

    calls = []

    def fake_inpaint_array(rgb, mask, cfg, return_diagnostics=False):
        calls.append((mask.copy(), cfg))
        assert return_diagnostics is True
        assert cfg["inpaint"]["mode"] == "big-lama"
        # This oracle-like fake is only used to prove the bakeoff wiring and record format.
        clean = rgb.copy()
        clean[mask > 0] = rgb[0, 0]
        return clean, "big-lama", {
            "backend_class": "active",
            "backend_route": {"requested": "big-lama", "selected": "big-lama", "selected_class": "active"},
        }

    # Use an all-flat case so replacing the masked area with its known plate value is exact.
    monkeypatch.setattr(inpaint, "inpaint_array", fake_inpaint_array)
    report = run_inpaint_bakeoff(
        tmp_path,
        config={"inpaint": {"allow_fallback": True}},
        requested_backend="big-lama",
        cases_per_family=1,
        size=(160, 128),
        seed=31,
        families=("flat",),
    )

    assert len(calls) == 1
    assert report["summary"]["accepted"] is True
    case = report["cases"][0]
    assert case["backend"]["requested"] == "big-lama"
    assert case["backend"]["selected"] == "big-lama"
    assert case["backend"]["matches_requested"] is True
    assert "requested backend" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_bakeoff_rejects_silent_backend_substitution_unless_allowed(tmp_path, monkeypatch):
    from src import inpaint

    def fallback_inpaint(rgb, mask, cfg, return_diagnostics=False):
        assert cfg["inpaint"]["mode"] == "flux_comfy"
        clean = rgb.copy()
        clean[mask > 0] = rgb[0, 0]
        return clean, "opencv-telea", {
            "backend_class": "fallback",
            "backend_route": {"requested": "flux_comfy", "selected": "opencv-telea", "selected_class": "fallback"},
        }

    monkeypatch.setattr(inpaint, "inpaint_array", fallback_inpaint)
    strict = run_inpaint_bakeoff(
        tmp_path / "strict",
        config={},
        requested_backend="flux_comfy",
        cases_per_family=1,
        size=(160, 128),
        seed=37,
        families=("flat",),
    )
    assert strict["summary"]["accepted"] is False
    assert strict["summary"]["backend_substitution_cases"] == 1
    assert strict["cases"][0]["metrics"]["failure_reasons"] == ["backend-substitution"]

    allowed = run_inpaint_bakeoff(
        tmp_path / "allowed",
        config={},
        requested_backend="flux_comfy",
        cases_per_family=1,
        size=(160, 128),
        seed=37,
        families=("flat",),
        allow_backend_substitution=True,
    )
    assert allowed["summary"]["accepted"] is True
    assert allowed["cases"][0]["backend"]["substituted"] is True
    assert allowed["cases"][0]["backend"]["substitution_allowed"] is True
