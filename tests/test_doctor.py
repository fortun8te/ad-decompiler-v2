from pathlib import Path

from doctor import _powerpaint_adapter_importable, inspect, ocr_ready_summary


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
    sam = next(item for item in report["blockers"] if item["name"] == "sam3 checkpoint")
    assert "config.yaml" in sam["fix"]


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


def test_doctor_requires_real_vectorization_stack_for_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {
        "name": "torch", "ok": True, "required": False, "detail": "cpu"
    })
    monkeypatch.setattr("src.vectorize.check_binaries", lambda cfg: {
        "vtracer": {"ok": False, "path": "missing"},
        "potrace": {"ok": False, "path": "missing"},
        "cairosvg": {"ok": True, "path": "python:cairosvg"},
        "resvg": {"ok": False, "path": "missing"},
    })

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False},
        "runtime": {"require_active_models": True}, "inpaint": {"mode": "opencv"},
    }, Path(tmp_path))

    assert any(item["name"] == "vectorization stack" for item in report["blockers"])
    assert report["policy"]["vectorization_required"] is True


def test_doctor_reports_inpaint_stack_ok_when_lama_installed_even_if_comfyui_down(tmp_path, monkeypatch):
    # Qwen/ComfyUI is an advisory, separate capability from background inpainting (see
    # run_report._required's comment: it "must not make the main SAM/OCR scene graph look
    # unavailable merely because a separate ComfyUI process is offline"). Big-LaMa alone
    # must be sufficient for the inpaint stack to read READY.
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("doctor._module", lambda name: name == "simple_lama_inpainting")
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._http", lambda url: False)

    report = inspect({
        "device": "cpu",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": True, "mode": "comfyui", "workflow": str(workflow)},
        "inpaint": {"mode": "auto"},
        "runtime": {"require_active_models": True},
    }, Path(tmp_path))

    stack = next(item for item in report["checks"] if item["name"] == "inpaint stack (Big-LaMa)")
    assert stack["ok"] is True
    assert not any(item["name"] == "inpaint stack (Big-LaMa)" for item in report["blockers"])


def test_doctor_reports_inpaint_stack_blocked_when_lama_missing(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("doctor._module", lambda name: False)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._http", lambda url: False)

    report = inspect({
        "device": "cpu",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": True, "mode": "comfyui", "workflow": str(workflow)},
        "inpaint": {"mode": "auto"},
        "runtime": {"require_active_models": True},
    }, Path(tmp_path))

    stack = next(item for item in report["checks"] if item["name"] == "inpaint stack (Big-LaMa)")
    assert stack["ok"] is False
    assert any(item["name"] == "inpaint stack (Big-LaMa)" for item in report["blockers"])


def test_strict_flux_selection_blocks_unverifiable_local_weights(tmp_path, monkeypatch):
    workflow = tmp_path / "flux.json"
    workflow.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("doctor._module", lambda _name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {
        "name": "torch", "ok": True, "required": False, "detail": "cpu",
    })
    monkeypatch.setattr("doctor._http", lambda _url: True)

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False},
        "inpaint": {"mode": "flux_comfy", "strict_acceptance": True,
                    "comfy": {"enabled": True, "workflow": str(workflow)}},
    }, tmp_path)

    flux_models = next(item for item in report["checks"] if item["name"] == "flux inpaint models")
    assert flux_models["ok"] is False
    assert flux_models["required"] is True
    assert any(item["name"] == "flux inpaint models" for item in report["blockers"])
    assert report["policy"]["inpaint_selected"] == "flux_comfy"
    assert report["policy"]["inpaint_strict_acceptance"] is True


def test_strict_powerpaint_selection_blocks_missing_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda _name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {
        "name": "torch", "ok": True, "required": False, "detail": "cpu",
    })

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False},
        "inpaint": {"mode": "powerpaint", "strict_acceptance": True, "powerpaint": {}},
    }, tmp_path)

    adapter = next(item for item in report["checks"] if item["name"] == "PowerPaint adapter configuration")
    assert adapter["ok"] is False
    assert adapter["required"] is True
    assert any(item["name"] == "PowerPaint adapter configuration" for item in report["blockers"])
    assert report["policy"]["inpaint_selected"] == "powerpaint"


def test_strict_powerpaint_accepts_only_an_enabled_importable_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: name != "missing_powerpaint_adapter")
    monkeypatch.setattr("doctor._torch", lambda device: {
        "name": "torch", "ok": True, "required": False, "detail": "cpu",
    })

    report = inspect({
        "device": "cpu", "ocr": {"primary": "doctr"}, "qwen": {"enabled": False},
        "inpaint": {"mode": "powerpaint", "strict_acceptance": True, "powerpaint": {
            "enabled": True, "adapter_module": "missing_powerpaint_adapter", "callable": "inpaint",
        }},
    }, tmp_path)

    imported = next(item for item in report["checks"] if item["name"] == "PowerPaint adapter import")
    assert imported["ok"] is False
    assert imported["required"] is True
    assert "not importable" in imported["detail"]


def test_powerpaint_adapter_probe_checks_callable_but_does_not_run_a_model():
    ok, detail = _powerpaint_adapter_importable("math", "sqrt")

    assert ok is True
    assert "callable" in detail
    assert "not executed" in detail


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


def test_doctor_warns_when_scene_text_enabled_but_server_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._http", lambda url: False)

    report = inspect({
        "device": "cpu",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": False},
        "vlm": {"scene_text": {"enabled": True}, "base_url": "http://127.0.0.1:1234/v1"},
    }, Path(tmp_path))

    assert any(item["name"] == "VLM server" and not item["ok"] for item in report["warnings"])


def test_doctor_warns_when_vlm_enabled_but_server_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._http", lambda url: False)

    report = inspect({
        "device": "cpu",
        "ocr": {"primary": "doctr"},
        "qwen": {"enabled": False},
        "vlm": {"enabled": True, "base_url": "http://127.0.0.1:1234/v1"},
    }, Path(tmp_path))

    assert any(item["name"] == "VLM server" and not item["ok"] for item in report["warnings"])


def test_doctor_requires_exact_gemma_identity_for_active_model_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("doctor._module", lambda name: True)
    monkeypatch.setattr("doctor._torch", lambda device: {"name": "torch", "ok": True, "required": False, "detail": "cpu"})
    monkeypatch.setattr("doctor._vlm_model_loaded", lambda base, model: (True, f"{model} loaded"))
    cfg = {
        "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
        "inpaint": {"mode": "opencv"}, "runtime": {"require_active_models": True},
        "vlm": {"enabled": True, "model": "some-other-model"},
        "ocr": {"primary": "easyocr", "auto_fallback_tesseract": False},
    }
    report = inspect(cfg, tmp_path)
    assert any(item["name"] == "VLM model identity" for item in report["blockers"])
