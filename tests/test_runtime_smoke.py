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


def test_ocr_probe_rejects_silent_fallback_engine(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ocr.run_ocr", lambda *args, **kwargs: {
        "status": "ok", "engine": "tesseract", "lines": [{"text": "GPU SMOKE"}],
    })
    result = runtime_smoke._probe_ocr({"ocr": {"primary": "doctr"}}, tmp_path)
    assert result["ok"] is False
    assert "engine=tesseract" in result["detail"]


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
