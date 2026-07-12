import json
import os

from src.error_messages import classify_processing_error, detect_failed_stage, tail_running_stage


def test_tail_running_stage_reads_last_pipeline_marker(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pipeline.log").write_text(
        "[12:00:00] normalize → 1080x1080\n"
        "[12:00:05] ocr[doctr] → 2 lines\n",
        encoding="utf-8",
    )
    assert tail_running_stage(str(run_dir)) == "ocr"


def test_detect_failed_stage_ocr_after_normalize_without_ocr_success(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pipeline.log").write_text(
        "[12:00:00] normalize → 1080x1080\n[12:00:01] ERROR: no configured OCR backend completed (ppocr-v6: cudnn error)\n",
        encoding="utf-8",
    )
    stage = detect_failed_stage(
        str(run_dir),
        error_text="no configured OCR backend completed (ppocr-v6: cudnn error)",
    )
    assert stage == "ocr"


def test_detect_failed_stage_prefers_explicit_stage(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pipeline.log").write_text("[12:00:00] normalize → 1080x1080\n", encoding="utf-8")
    stage = detect_failed_stage(str(run_dir), explicit_stage="sam")
    assert stage == "sam"


def test_detect_failed_stage_from_agent_debug(tmp_path):
    debug = [
        {"location": "ocr.py:run_ocr", "message": "primary backend failed", "data": {"error": "cudnn"}},
    ]
    stage = detect_failed_stage(str(tmp_path), agent_debug=debug)
    assert stage == "ocr"


def test_classify_processing_error_ocr_cudnn():
    out = classify_processing_error(
        error="no configured OCR backend completed (ppocr-v6: cudnn not found)",
        failed_stage="normalize",
    )
    assert out["error_code"] == "cudnn_unavailable"
    assert out["failed_stage"] == "ocr"
    assert "cuDNN" in out["error_hint"]
    assert out["user_title"] == "GPU library (cuDNN) issue"


def test_classify_processing_error_windows_charmap():
    out = classify_processing_error(
        error="'charmap' codec can't encode character '\\u2192' in position 12",
    )
    assert out["error_code"] == "windows_encoding"
    assert "UTF-8" in out["error_hint"]


def test_classify_processing_error_docker():
    out = classify_processing_error(error="Cannot connect to the Docker daemon")
    assert out["error_code"] == "docker_not_supported"
    assert "Start Bridge.bat" in out["error_hint"]


def test_layout_container_repair_is_not_misclassified_as_docker():
    out = classify_processing_error(
        error="QA did not pass after layout tighten-containers repair",
        failed_stage="acceptance",
        agent_debug=[{
            "location": "harness_fixer.py:fix_layout",
            "message": "tighten container inference",
            "data": {"engine": "tighten-containers"},
        }],
    )

    assert out["error_code"] == "pipeline_failed"
    assert out["user_title"] == "Stopped during acceptance"
    assert "tighten-containers" in out["user_detail"]


def test_classify_processing_error_dependency_missing():
    out = classify_processing_error(error="ModuleNotFoundError: No module named 'paddleocr'")
    assert out["error_code"] == "dependency_missing"
    assert "setup_rtx.ps1" in out["error_hint"]


def test_classify_processing_error_sanitizes_traceback():
    tb = '\n  File "run_pipeline.py", line 10, in run_one\n    raise RuntimeError("boom")\nRuntimeError: boom\n'
    out = classify_processing_error(error="boom", traceback_text=tb)
    assert "File " not in out["user_detail"]
    assert "boom" in out["user_detail"]
