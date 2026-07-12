import json

import rtx_self_test


def test_cache_is_valid_only_for_matching_config_and_existing_evidence(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("device: cuda\n", encoding="utf-8")
    output = tmp_path / "out"
    evidence = output / "proof" / "self_test.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    payload = {
        "ok": True,
        "fingerprint": rtx_self_test.fingerprint(config),
        "finished_at": 1000,
        "evidence_path": str(evidence),
    }
    (output / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    assert rtx_self_test.cache_status(output, config, now=1100)["valid"] is True
    config.write_text("device: cpu\n", encoding="utf-8")
    status = rtx_self_test.cache_status(output, config, now=1100)
    assert status["valid"] is False
    assert status["reason"] == "config_or_code_changed"


def test_cache_expires_and_never_loads_models(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("device: cuda\n", encoding="utf-8")
    output = tmp_path / "out"
    evidence = output / "self_test.json"
    output.mkdir()
    evidence.write_text("{}", encoding="utf-8")
    payload = {"ok": True, "fingerprint": rtx_self_test.fingerprint(config),
               "finished_at": 1, "evidence_path": str(evidence)}
    (output / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    status = rtx_self_test.cache_status(
        output, config, now=rtx_self_test.CACHE_MAX_AGE_S + 2,
    )
    assert status["valid"] is False
    assert status["reason"] == "expired"


def test_pipeline_evidence_requires_real_primary_ocr_and_sam_masks(tmp_path):
    (tmp_path / "ocr_raw.json").write_text(json.dumps({
        "lines": [{"text": "GPU SMOKE"}],
        "metrics": {"cross_check": {"successful_engines": ["doctr"]}},
    }), encoding="utf-8")
    (tmp_path / "sam3.json").write_text(json.dumps({
        "engine": "sam3", "diagnostics": {"model_elements": 2, "text_prompts_succeeded": 1},
    }), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "stats": {"inpaint": {"backend": "big-lama"}},
    }), encoding="utf-8")
    (tmp_path / "runtime_report.json").write_text(json.dumps({"status": "ok", "stages": []}), encoding="utf-8")

    checks = rtx_self_test.evaluate_pipeline(
        tmp_path, {"ok": True}, {"ocr": {"primary": "doctr"}},
    )

    assert all(item["ok"] for item in checks)
    sam = json.loads((tmp_path / "sam3.json").read_text())
    sam["diagnostics"]["model_elements"] = 0
    (tmp_path / "sam3.json").write_text(json.dumps(sam), encoding="utf-8")
    checks = rtx_self_test.evaluate_pipeline(
        tmp_path, {"ok": True}, {"ocr": {"primary": "doctr"}},
    )
    assert next(item for item in checks if item["name"] == "sam3_runtime")["ok"] is False
