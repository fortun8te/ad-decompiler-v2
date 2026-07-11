from pathlib import Path

from doctor import inspect


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
