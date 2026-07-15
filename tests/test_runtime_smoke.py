import json
import queue

from PIL import Image

import runtime_smoke


def test_fixture_is_small_and_contains_real_mask(tmp_path):
    image, mask = runtime_smoke._fixture(tmp_path)
    assert Image.open(image).size == (256, 192)
    assert Image.open(mask).getbbox() is not None


def test_figma_staging_probe_verifies_manifest_asset_hash(tmp_path):
    result = runtime_smoke._probe_figma_staging({}, tmp_path)
    assert result["ok"] is True
    assert result["evidence"]["doc_id"] == "gpu-smoke"
    manifest = json.loads((tmp_path / "figma-inbox" / "inbox.json").read_text())
    dot = next(item for item in manifest["files"] if item["path"] == "assets/dot.png")
    assert len(dot["sha256"]) == 64


def test_vlm_probe_requires_schema_answer(monkeypatch, tmp_path):
    monkeypatch.setattr("src.vlm_client.ask_vlm", lambda *args, **kwargs: '{"label":"gpu-smoke"}')
    assert runtime_smoke._probe_vlm({"vlm": {"model": "google/gemma-4-e4b"}}, tmp_path)["ok"]
    monkeypatch.setattr("src.vlm_client.ask_vlm", lambda *args, **kwargs: '{"label":"wrong"}')
    assert not runtime_smoke._probe_vlm({}, tmp_path / "wrong")["ok"]


def test_vlm_probe_defaults_to_current_gemma_identity(monkeypatch, tmp_path):
    seen = {}

    def capture(image_bytes, prompt, **kwargs):
        seen.update(kwargs)
        return '{"label":"gpu-smoke"}'

    monkeypatch.setattr("src.vlm_client.ask_vlm", capture)
    result = runtime_smoke._probe_vlm({}, tmp_path)
    assert result["ok"] is True
    assert seen["model"] == "google/gemma-4-e4b"
    assert "google/gemma-4-e4b" in result["detail"]
    assert runtime_smoke.DEFAULT_VLM_MODEL == "google/gemma-4-e4b"


def test_vlm_probe_gives_windows_first_action_when_server_down(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("src.vlm_client.ask_vlm", boom)
    result = runtime_smoke._probe_vlm({}, tmp_path)
    assert result["ok"] is False
    assert "lms load google/gemma-4-e4b" in result["detail"]
    assert "LM Studio" in result["detail"]


def test_lms_cli_required_only_when_vram_eviction_is_configured(monkeypatch):
    monkeypatch.setattr(runtime_smoke, "_which_lms", lambda: None)
    disabled = runtime_smoke.check_lms_cli({})
    assert disabled["ok"] is True
    assert disabled["evidence"]["required"] is False

    enabled = runtime_smoke.check_lms_cli(
        {"runtime": {"vram": {"evict_vlm_for_inpaint": True}}})
    assert enabled["ok"] is False
    assert "lms bootstrap" in enabled["detail"]

    monkeypatch.setattr(runtime_smoke, "_which_lms", lambda: r"C:\Users\me\.lmstudio\bin\lms.exe")
    present = runtime_smoke.check_lms_cli(
        {"runtime": {"vram": {"reload_vlm_after_inpaint": True}}})
    assert present["ok"] is True
    assert present["evidence"]["path"].endswith("lms.exe")


def test_tesseract_discovery_is_advisory_unless_primary(monkeypatch):
    monkeypatch.setattr(runtime_smoke, "_which_tesseract", lambda: None)
    monkeypatch.setattr(runtime_smoke.shutil, "which", lambda name: None)
    advisory = runtime_smoke.check_tesseract({"ocr": {"primary": "doctr"}})
    assert advisory["ok"] is True
    assert "winget install UB-Mannheim.TesseractOCR" in advisory["detail"]

    required = runtime_smoke.check_tesseract({"ocr": {"primary": "tesseract"}})
    assert required["ok"] is False
    assert "winget install UB-Mannheim.TesseractOCR" in required["detail"]

    known = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    monkeypatch.setattr(runtime_smoke, "_which_tesseract", lambda: known)
    off_path = runtime_smoke.check_tesseract({"ocr": {"primary": "doctr"}})
    assert off_path["ok"] is True
    assert "not on PATH" in off_path["detail"]
    assert off_path["evidence"]["path"] == known


def test_environment_checks_returns_lms_and_tesseract_rows():
    names = [row["name"] for row in runtime_smoke.environment_checks({})]
    assert names == ["lms_cli", "tesseract"]


def test_ocr_probe_rejects_silent_fallback_engine(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ocr.run_ocr", lambda *args, **kwargs: {
        "status": "ok", "engine": "tesseract", "lines": [{"text": "GPU SMOKE"}],
    })
    result = runtime_smoke._probe_ocr({"ocr": {"primary": "doctr"}}, tmp_path)
    assert result["ok"] is False
    assert "engine=tesseract" in result["detail"]


def test_flux_probe_requires_real_backend_and_outside_identity(monkeypatch, tmp_path):
    import numpy as np

    def fake_once(image_path, mask_path, output_path, cfg):
        before = np.asarray(Image.open(image_path).convert("RGB")).copy()
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        before[mask] = (12, 34, 56)
        Image.fromarray(before).save(output_path)
        return {"backend": "flux-comfy"}

    monkeypatch.setattr("src.inpaint.inpaint_once", fake_once)
    result = runtime_smoke._probe_flux_comfy({}, tmp_path)
    assert result["ok"] is True
    assert result["evidence"]["outside_identical"] is True


def test_powerpaint_probe_requires_exact_backend_and_disables_fallback(monkeypatch, tmp_path):
    import numpy as np
    seen = {}

    def fake_once(image_path, mask_path, output_path, cfg):
        seen["cfg"] = cfg
        before = np.asarray(Image.open(image_path).convert("RGB")).copy()
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        before[mask] = (12, 34, 56)
        Image.fromarray(before).save(output_path)
        return {"backend": "powerpaint"}

    monkeypatch.setattr("src.inpaint.inpaint_once", fake_once)
    result = runtime_smoke._probe_powerpaint(
        {"inpaint": {"powerpaint": {"adapter_module": "local_powerpaint_adapter"}}}, tmp_path,
    )

    assert result["ok"] is True
    assert result["evidence"]["backend"] == "powerpaint"
    assert seen["cfg"]["inpaint"]["allow_fallback"] is False
    assert seen["cfg"]["inpaint"]["powerpaint"]["required"] is True


def test_default_probe_selection_follows_requested_inpaint_backend():
    assert "flux_comfy" in runtime_smoke.selected_probes({"inpaint": {"mode": "flux_comfy"}})
    assert "big_lama" not in runtime_smoke.selected_probes({"inpaint": {"mode": "flux_comfy"}})
    assert "powerpaint" in runtime_smoke.selected_probes({"inpaint": {"mode": "powerpaint"}})
    assert "big_lama" in runtime_smoke.selected_probes({"inpaint": {"mode": "auto"}})
    assert not ({"big_lama", "flux_comfy", "powerpaint"} & set(
        runtime_smoke.selected_probes({"inpaint": {"mode": "opencv"}})
    ))


def test_run_all_uses_selected_inpaint_probe_when_unspecified(monkeypatch, tmp_path):
    calls = []

    def fake(name, cfg, work, timeout):
        calls.append(name)
        return {"name": name, "ok": True, "detail": "fixture"}

    monkeypatch.setattr(runtime_smoke, "_run_bounded", fake)
    report = runtime_smoke.run_all({"inpaint": {"mode": "powerpaint"}}, tmp_path, timeout_s=3)

    assert "powerpaint" in calls
    assert "big_lama" not in calls
    assert "flux_comfy" not in calls
    assert report["probes"] == calls


def test_worker_turns_probe_exception_into_evidence(monkeypatch, tmp_path):
    monkeypatch.setitem(runtime_smoke._IMPLEMENTATIONS, "ocr",
                        lambda cfg, work: (_ for _ in ()).throw(RuntimeError("boom")))
    output = queue.Queue()
    runtime_smoke._worker("ocr", {}, str(tmp_path), output)
    result = output.get_nowait()
    assert result["ok"] is False
    assert "RuntimeError: boom" in result["detail"]


def test_run_all_persists_fail_closed_summary(monkeypatch, tmp_path):
    def fake(name, cfg, work, timeout):
        return {"name": name, "ok": name != "sam3", "detail": "fixture"}
    monkeypatch.setattr(runtime_smoke, "_run_bounded", fake)
    report = runtime_smoke.run_all({}, tmp_path, probes=("ocr", "sam3"), timeout_s=3)
    assert report["ok"] is False
    assert [item["name"] for item in report["checks"]] == ["ocr", "sam3"]
    assert json.loads((tmp_path / "runtime_smoke.json").read_text())["ok"] is False
