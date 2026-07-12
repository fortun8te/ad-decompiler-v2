from pathlib import Path

from doctor import inspect, ocr_ready_summary


def test_doctor_marks_active_missing_sam_and_primary_ocr_as_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: False)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "cuda", "ok": True, "required": True, "detail": "fake"})
    cfg = {
        "device": "cuda",
        "ocr": {"primary": "ppocr-v6", "challengers": ["surya"]},
        "sam3": {"enabled": True, "checkpoint": str(tmp_path / "missing.pt")},
        "qwen": {"enabled": False},
    }

    report = inspect(cfg, Path(tmp_path))

    assert not report["ok"]
    assert {item["name"] for item in report["blockers"]} >= {"ocr:ppocr-v6", "sam3 package", "sam3 checkpoint"}
    assert any(item["name"] == "ocr challenger:surya" for item in report["warnings"])


def test_doctor_accepts_a_minimal_cpu_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor.shutil.which", lambda name: "/bin/tool")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor.sys.version_info", (3, 12, 0))

    report = inspect({"device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False}}, Path(tmp_path))

    assert report["ok"]
    assert report["blockers"] == []


def test_doctor_makes_qwen_backend_a_blocker_only_when_explicitly_required(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._http", lambda url: False)

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"},
        "qwen": {"enabled": True, "required": True, "mode": "comfyui", "workflow": str(workflow)},
    }, Path(tmp_path))

    assert not report["ok"]
    assert any(item["name"] == "ComfyUI" for item in report["blockers"])
    assert report["policy"]["qwen_required"] is True


def test_doctor_promotes_big_lama_to_a_blocker_under_require_active_models(tmp_path, monkeypatch):
    """Regression: Big-LaMa was unconditionally required=False, so a missing/broken
    install never blocked doctor.py even with runtime.require_active_models=true, even
    though a silent OpenCV fallback degrades the background plate."""
    monkeypatch.setattr("doctor._module", lambda name: name != "simple_lama_inpainting")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False},
        "runtime": {"require_active_models": True},
    }, Path(tmp_path))

    assert not report["ok"]
    assert any(item["name"] == "Big-LaMa" for item in report["blockers"])


def test_doctor_leaves_big_lama_optional_without_require_active_models(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name != "simple_lama_inpainting")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})

    report = inspect({"device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False}}, Path(tmp_path))

    assert any(item["name"] == "Big-LaMa" for item in report["warnings"])
    assert not any(item["name"] == "Big-LaMa" for item in report["blockers"])


def test_ocr_ready_summary_flags_primary_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name != "paddleocr")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "cuda", "ok": True, "required": True, "detail": "fake"})
    monkeypatch.setattr("doctor._cudnn", lambda device: {"name": "cudnn", "ok": False, "required": False, "detail": "missing"})

    summary = ocr_ready_summary(
        {"device": "cuda", "ocr": {"primary": "ppocr-v6"}, "qwen": {"enabled": False}},
        Path(tmp_path),
    )

    assert summary["ok"] is False
    assert summary["primary"] == "ppocr-v6"
    assert any(item["name"] == "ocr:ppocr-v6" for item in summary["blockers"])
    assert any(item["name"] == "cudnn" for item in summary["warnings"])


def test_doctor_reports_tesseract_fallback_when_binary_present(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name == "pytesseract")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor.shutil.which", lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)

    report = inspect({
        "device": "cuda",
        "ocr": {"primary": "ppocr-v6", "fallback_engines": ["tesseract"]},
        "qwen": {"enabled": False},
    }, Path(tmp_path))

    assert report["ocr_fallback"]["ready"] is True
    assert any(item["engine"] == "tesseract" for item in report["ocr_fallback"]["available"])
    summary = ocr_ready_summary({
        "device": "cuda",
        "ocr": {"primary": "ppocr-v6"},
        "qwen": {"enabled": False},
    }, Path(tmp_path))
    if not report["blockers"]:
        assert summary["ok"] is True


def test_doctor_reports_doctr_gpu_check_when_primary_is_doctr_on_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name == "doctr")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "cuda", "ok": True, "required": True, "detail": "fake"})
    monkeypatch.setattr("doctor._cudnn", lambda device: {"name": "cudnn", "ok": True, "required": False, "detail": "ok"})
    monkeypatch.setattr("doctor._doctr_gpu", lambda device, primary: {
        "name": "doctr gpu", "ok": True, "required": True, "detail": "doctr will run on RTX 5080",
    })

    report = inspect({
        "device": "cuda",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": False},
    }, Path(tmp_path))

    doctr_gpu = next(item for item in report["checks"] if item["name"] == "doctr gpu")
    assert doctr_gpu["ok"] is True
    assert doctr_gpu["required"] is True


def test_doctor_blocks_when_doctr_primary_but_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name == "doctr")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "cuda", "ok": False, "required": True, "detail": "missing"})
    monkeypatch.setattr("doctor._cudnn", lambda device: {"name": "cudnn", "ok": True, "required": False, "detail": "ok"})
    monkeypatch.setattr("doctor._doctr_gpu", lambda device, primary: {
        "name": "doctr gpu", "ok": False, "required": True,
        "detail": "torch cannot see a CUDA device for doctr primary",
    })

    report = inspect({
        "device": "cuda",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": False},
    }, Path(tmp_path))

    assert any(item["name"] == "doctr gpu" and not item["ok"] for item in report["blockers"])
