import json

from src.run_report import RunReport


def test_required_sam_fallback_is_visible_and_invalidates_acceptance(tmp_path):
    report = RunReport(
        str(tmp_path), "input.png",
        {"runtime": {"require_active_models": True}, "sam3": {"enabled": True}},
        "normalize",
    )
    report.degraded("sam3", "checkpoint missing")
    report.finish(qa_ok=False)

    saved = json.loads((tmp_path / "runtime_report.json").read_text())
    assert saved["status"] == "degraded"
    assert saved["acceptable"] is False
    assert saved["violations"][0]["rule"] == "sam3-unavailable"


def test_advisory_qwen_fallback_is_reported_without_invalidating_sam_ocr_run(tmp_path):
    report = RunReport(
        str(tmp_path), "input.png",
        {"runtime": {"require_active_models": True}, "qwen": {"enabled": True}},
        "normalize",
    )
    report.degraded("qwen", "ComfyUI offline")
    report.finish(qa_ok=True)

    assert report.data["status"] == "degraded"
    assert report.acceptable is True
    assert report.data["violations"] == []


def test_inpaint_fallback_is_required_and_invalidates_acceptance_under_require_active_models(tmp_path):
    """Regression: a silent Big-LaMa -> OpenCV fallback used to be invisible to
    acceptance (only 'stats.inpaint.backend' in reconstruction.json, never checked).
    Under runtime.require_active_models it must behave like sam3/ocr: reported and a
    hard violation."""
    report = RunReport(
        str(tmp_path), "input.png",
        {"runtime": {"require_active_models": True}},
        "normalize",
    )
    assert "inpaint" in report.data["policy"]["required_components"]
    report.degraded("inpaint", "Big-LaMa unavailable; used opencv-telea fallback for background plate")
    report.finish(qa_ok=False)

    saved = json.loads((tmp_path / "runtime_report.json").read_text())
    assert saved["status"] == "degraded"
    assert saved["acceptable"] is False
    assert any(v["rule"] == "inpaint-unavailable" for v in saved["violations"])


def test_inpaint_fallback_is_not_required_without_require_active_models(tmp_path):
    report = RunReport(str(tmp_path), "input.png", {}, "normalize")
    report.degraded("inpaint", "Big-LaMa unavailable; used opencv-telea fallback for background plate")
    report.finish(qa_ok=True)

    assert report.acceptable is True
    assert report.data["violations"] == []


def test_inpaint_explicit_opencv_mode_is_not_required_even_under_require_active_models(tmp_path):
    """Regression: doctor.py treats an explicit ``inpaint.mode: opencv`` as a legitimate,
    READY configuration under require_active_models (it isn't a silent fallback -- the
    operator asked for it). ``_required``/RunReport must agree, or a doctor-approved config
    fails every run's acceptance policy (see doctor.py's Big-LaMa check + CLAUDE_FINAL_TWEAKS.md)."""
    report = RunReport(
        str(tmp_path), "input.png",
        {"runtime": {"require_active_models": True}, "inpaint": {"mode": "opencv"}},
        "normalize",
    )
    assert "inpaint" not in report.data["policy"]["required_components"]
    report.degraded("inpaint", "opencv-telea backend used for background plate")
    report.finish(qa_ok=True)

    saved = json.loads((tmp_path / "runtime_report.json").read_text())
    assert saved["acceptable"] is True
    assert saved["violations"] == []


def test_inpaint_default_auto_mode_still_required_under_require_active_models(tmp_path):
    """An unset/'auto' inpaint.mode must not accidentally gain the opencv exemption."""
    report = RunReport(
        str(tmp_path), "input.png",
        {"runtime": {"require_active_models": True}, "inpaint": {"mode": "auto"}},
        "normalize",
    )
    assert "inpaint" in report.data["policy"]["required_components"]
    report.degraded("inpaint", "Big-LaMa unavailable; used opencv-telea fallback for background plate")
    report.finish(qa_ok=False)

    saved = json.loads((tmp_path / "runtime_report.json").read_text())
    assert saved["acceptable"] is False
    assert any(v["rule"] == "inpaint-unavailable" for v in saved["violations"])
