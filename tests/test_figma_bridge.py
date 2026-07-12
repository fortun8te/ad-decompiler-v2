import json
import os
import sys
import threading
import time
import types
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.figma_bridge import (
    _estimate_eta,
    _parse_stage_fractions,
    _read_history,
    _record_history,
    make_handler,
)


def _write_passing_config(tmp_path):
    """Minimal config that passes doctor when modules are monkeypatched."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "device: cpu\nocr:\n  primary: doctr\nqwen:\n  enabled: false\nsam3:\n  enabled: false\n",
        encoding="utf-8",
    )
    return str(path)


def _allow_machine_ready(monkeypatch):
    monkeypatch.setattr(
        "doctor.inspect",
        lambda cfg, root: {"ok": True, "blockers": [], "warnings": [], "checks": []},
    )
    monkeypatch.setattr(
        "doctor.ocr_ready_summary",
        lambda cfg, root: {"ok": True, "primary": "doctr", "blockers": [], "warnings": []},
    )


def test_bridge_serves_staged_run_and_persists_plugin_report(tmp_path):
    inbox = tmp_path / "inbox"
    staged = inbox / "runs" / "demo"
    staged.mkdir(parents=True)
    (staged / "design.json").write_text(encoding="utf-8", data=json.dumps({"id": "demo", "canvas": {"w": 1, "h": 1}, "layers": []}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "schema_version": 2, "doc_id": "demo", "design": "design.json",
        "staged_dir": "runs/demo", "run_dir": str(run_dir), "summary": {"layers": 0},
    }
    (inbox / "inbox.json").write_text(encoding="utf-8", data=json.dumps(manifest))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urlopen(base + "/health", timeout=2).read())
        assert health["ok"] is True and health["has_run"] is True
        assert health.get("supports_process") is True
        design = json.loads(urlopen(base + "/design.json", timeout=2).read())
        assert design["id"] == "demo"
        report = {"version": 1, "doc_id": "demo", "report": {"ok": True, "assets": {"missing": 0}}}
        request = Request(base + "/report", data=json.dumps(report).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        assert json.loads(urlopen(request, timeout=2).read())["ok"] is True
        assert json.loads((run_dir / "figma_report.json").read_text(encoding="utf-8"))["doc_id"] == "demo"
        try:
            urlopen(base + "/asset?path=../../etc/passwd", timeout=2)
            assert False, "path traversal should be rejected"
        except HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()
        server.server_close()


def _start_server(inbox, config_path=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_export_with_garbage_content_length_returns_400_not_connection_reset(tmp_path):
    """Regression: a malformed/non-numeric Content-Length used to raise ValueError inside
    do_POST, which socketserver's handle_error() swallows into a dropped connection -- the
    plugin sees a reset with no diagnosable HTTP error and nothing is written to disk."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/export", data=b"not-a-real-png", method="POST",
                          headers={"Content-Length": "not-a-number"})
        try:
            urlopen(request, timeout=2)
            assert False, "garbage Content-Length should be rejected with a clear 400"
        except HTTPError as error:
            assert error.code == 400
        assert not (inbox / "figma_export.png").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_export_with_oversized_content_length_returns_400(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/export", data=b"x", method="POST",
                          headers={"Content-Length": str(64 * 1024 * 1024)})
        try:
            urlopen(request, timeout=2)
            assert False, "oversized Content-Length should be rejected"
        except HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_report_with_garbage_content_length_returns_400(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/report", data=b"{}", method="POST",
                          headers={"Content-Length": "garbage"})
        try:
            urlopen(request, timeout=2)
            assert False, "garbage Content-Length should be rejected with a clear 400"
        except HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()


def _install_fake_run_pipeline(monkeypatch, *, ok=True, sleep_s=0.0):
    """POST /process lazy-imports run_pipeline (heavy: torch/paddleocr/sam3/...) so it never
    loads at bridge-startup time. Stub it out in sys.modules so this runs without those deps,
    exactly like the real thing would from the caller's point of view (a plain function call
    that returns a run_one()-shaped dict and, on success, would have staged the inbox itself
    -- staging is exercised for real by test_process_stages_inbox_json below)."""
    calls = []
    fake = types.ModuleType("run_pipeline")

    def run_one(image_path, run_dir, cfg):
        calls.append({"image_path": image_path, "run_dir": run_dir, "cfg": cfg})
        if sleep_s:
            time.sleep(sleep_s)
        if not ok:
            return {"ok": False, "run_dir": run_dir, "error": "boom"}
        import os
        os.makedirs(run_dir, exist_ok=True)
        design = {"id": "upload", "canvas": {"w": 10, "h": 10}, "layers": []}
        design_path = os.path.join(run_dir, "design.json")
        with open(design_path, "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        # A real run_one() with cfg.figma.enabled stages the inbox itself via figma_import;
        # do the same here so /inbox.json + /design.json are populated end-to-end.
        if (cfg.get("figma") or {}).get("enabled"):
            from src import figma_import
            figma_import.import_design(design_path, run_dir, cfg)
        return {"ok": True, "run_dir": run_dir, "duration_s": 0.01}

    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)
    return calls


def _poll_job(base, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = json.loads(urlopen(f"{base}/process?job_id={job_id}", timeout=2).read())
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def test_plugin_log_endpoint_appends_text_and_json(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"run_dir": str(run_dir), "doc_id": "demo"}
    (inbox / "inbox.json").write_text(encoding="utf-8", data=json.dumps(manifest))
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        events = [
            {"at": "2026-07-12T00:00:00Z", "level": "info", "title": "Upload started", "detail": "ad.png"},
            {"at": "2026-07-12T00:00:05Z", "level": "warn", "title": "Font substituted", "detail": "A → B"},
        ]
        request = Request(base + "/log", data=json.dumps({"events": events}).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        assert json.loads(urlopen(request, timeout=2).read())["ok"] is True
        inbox_log = (inbox / "plugin.log").read_text(encoding="utf-8")
        assert "Upload started" in inbox_log and "Font substituted" in inbox_log
        assert json.loads((inbox / "plugin_events.json").read_text(encoding="utf-8"))[0]["title"] == "Upload started"
        assert (run_dir / "plugin.log").exists()
        assert json.loads((run_dir / "plugin_events.json").read_text(encoding="utf-8"))[1]["level"] == "warn"
    finally:
        server.shutdown()
        server.server_close()


def test_health_includes_ocr_ready_summary(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    monkeypatch.setattr("src.figma_bridge._runtime_self_test_status", lambda config_path=None: {
        "valid": True, "reason": "passed", "evidence_path": "self_test.json",
    })
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = _start_server(inbox, config_path=config)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urlopen(base + "/health", timeout=2).read())
        assert health["ok"] is True
        assert health["ocr_ready"]["ok"] is True
        assert health["ocr_ready"]["primary"] == "doctr"
        assert health["machine_ready"] is True
        assert health["runtime_self_test"]["valid"] is True
        assert health["active_job"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_health_exposes_active_job_for_plugin_reconnect(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.5)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(
            base + "/process?filename=reconnect.png", data=b"x", method="POST",
        ), timeout=2).read())
        health = json.loads(urlopen(base + "/health", timeout=4).read())
        assert health["active_job"]["job_id"] == queued["job_id"]
        assert health["active_job"]["filename"] == "reconnect.png"
        assert health["active_job"]["status"] in ("queued", "running")
        _poll_job(base, queued["job_id"])
    finally:
        server.shutdown()
        server.server_close()


def test_health_includes_bridge_and_plugin_client_build(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "plugin_client.json").write_text(
        json.dumps({"label": "v2.0.0+b5.abc", "build": 5, "seen_at": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urlopen(base + "/health", timeout=2).read())
        assert health["plugin_client"]["build"] == 5
    finally:
        server.shutdown()
        server.server_close()


def test_log_endpoint_records_plugin_build_client(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = {
            "events": [{"at": "2026-07-12T00:00:00Z", "level": "info", "title": "Plugin started", "detail": "v2+b1"}],
            "plugin_build": {"version": "2.0.0", "build": 7, "commit": "abc", "label": "v2.0.0+b7.abc"},
        }
        request = Request(base + "/log", data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        assert json.loads(urlopen(request, timeout=2).read())["ok"] is True
        client = json.loads((inbox / "plugin_client.json").read_text(encoding="utf-8"))
        assert client["build"] == 7
        assert client["seen_at"]
    finally:
        server.shutdown()
        server.server_close()


def test_process_uploads_runs_pipeline_and_stages_inbox(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    calls = _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=my%20ad.png", data=b"\x89PNG-fake-bytes",
                          method="POST", headers={"Content-Type": "application/octet-stream"})
        response = urlopen(request, timeout=2)
        assert response.status == 202
        queued = json.loads(response.read())
        assert queued["status"] == "queued" and queued["job_id"]

        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done", status
        assert status["result"]["ok"] is True

        # cfg.figma was forced on and pointed at THIS bridge's own inbox, not whatever the
        # (possibly nonexistent) config.yaml said -- otherwise a run would stage into the
        # wrong folder and the plugin would never see it.
        assert len(calls) == 1
        assert calls[0]["cfg"]["figma"]["enabled"] is True
        assert os.path.abspath(calls[0]["cfg"]["figma"]["inbox"]) == os.path.abspath(str(inbox))

        # the uploaded filename was sanitized into the job dir, not written wherever the
        # caller-supplied filename would otherwise point (e.g. path traversal).
        assert "my-ad.png" in calls[0]["image_path"] or "my ad.png" not in calls[0]["image_path"]

        manifest = json.loads(urlopen(base + "/inbox.json", timeout=2).read())
        assert manifest["doc_id"] == "upload"
        design = json.loads(urlopen(base + "/design.json", timeout=2).read())
        assert design["id"] == "upload"

        assert status.get("staged") is True
        assert status.get("doc_id") == "upload"
        assert status.get("design_url") == "/design.json"

        summary = json.loads(urlopen(f"{base}/run-summary?job_id={queued['job_id']}", timeout=2).read())
        assert summary["ok"] is True
        assert summary["staged"] is True
        assert summary["manifest"]["doc_id"] == "upload"
    finally:
        server.shutdown()
        server.server_close()


def test_process_stages_when_figma_pipeline_stage_skipped(tmp_path, monkeypatch):
    """Bridge must re-stage even when run_one skips figma_import (resume / wrong inbox)."""
    _allow_machine_ready(monkeypatch)
    calls = []

    def run_one(image_path, run_dir, cfg):
        calls.append(cfg)
        os.makedirs(run_dir, exist_ok=True)
        design = {"id": "skipped-figma", "canvas": {"w": 10, "h": 10}, "layers": []}
        with open(os.path.join(run_dir, "design.json"), "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        return {"ok": True, "run_dir": run_dir, "duration_s": 0.01}

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    config_data = config
    # config points at a different inbox — bridge must override to its own inbox.
    import yaml
    cfg = yaml.safe_load(open(config_data, encoding="utf-8"))
    cfg["figma"] = {"enabled": True, "mode": "plugin", "inbox": str(tmp_path / "other-inbox")}
    with open(config_data, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(base + "/process?filename=ad.png", data=b"x", method="POST"), timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done"
        assert status["staged"] is True
        assert status["doc_id"] == "skipped-figma"
        assert not (tmp_path / "other-inbox" / "inbox.json").exists()
        manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
        assert manifest["doc_id"] == "skipped-figma"
        assert os.path.abspath(calls[0]["figma"]["inbox"]) == os.path.abspath(str(inbox))
    finally:
        server.shutdown()
        server.server_close()


def test_run_summary_returns_manifest_for_finished_job(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(base + "/process?filename=a.png", data=b"x", method="POST"), timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done"
        summary = json.loads(urlopen(f"{base}/run-summary?job_id={queued['job_id']}", timeout=2).read())
        assert summary["ok"] is True
        assert summary["staged"] is True
        assert summary["manifest"]["doc_id"] == "upload"
        assert summary["design_url"] == "/design.json"
        assert summary["run_dir"]
    finally:
        server.shutdown()
        server.server_close()


def test_process_rejects_concurrent_uploads(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.3)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        first = Request(base + "/process?filename=a.png", data=b"x", method="POST")
        queued = json.loads(urlopen(first, timeout=2).read())
        assert queued["status"] == "queued"

        second = Request(base + "/process?filename=b.png", data=b"y", method="POST")
        try:
            urlopen(second, timeout=2)
            assert False, "a second concurrent upload should be rejected"
        except HTTPError as error:
            assert error.code == 409

        _poll_job(base, queued["job_id"])
    finally:
        server.shutdown()
        server.server_close()


def test_process_cancel_endpoint_marks_job_cancelled(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.5)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(base + "/process?filename=a.png", data=b"x", method="POST"), timeout=2).read())
        cancel = Request(base + "/process/cancel?job_id=" + queued["job_id"], method="POST")
        payload = json.loads(urlopen(cancel, timeout=2).read())
        assert payload["status"] == "cancelled"
        status = json.loads(urlopen(base + "/process?job_id=" + queued["job_id"], timeout=2).read())
        assert status["status"] == "cancelled"
        time.sleep(0.65)
    finally:
        server.shutdown()
        server.server_close()


def test_process_reports_pipeline_failure_without_crashing_the_bridge(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, ok=False)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=broken.jpg", data=b"z", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "failed"
        assert status["error"]
        # the bridge itself must still be serving other routes after a failed job.
        health = json.loads(urlopen(base + "/health", timeout=2).read())
        assert health["ok"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_process_failed_job_includes_error_detail(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    calls = []

    def run_one(image_path, run_dir, cfg):
        calls.append(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "pipeline.log"), "w", encoding="utf-8") as fh:
            fh.write("ocr[1] starting\n")
        raise RuntimeError("CUDA out of memory")

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=broken.jpg", data=b"z", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "failed"
        assert "CUDA out of memory" in status["error"]
        assert status.get("error_detail")
        assert status.get("failed_stage") == "ocr"
        assert status.get("error_code") == "cuda_unavailable"
        assert status.get("error_hint")
        assert status.get("user_title")
        assert "traceback" not in status
    finally:
        server.shutdown()
        server.server_close()


def test_process_failed_job_reports_ocr_stage_after_normalize(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)

    def run_one(image_path, run_dir, cfg):
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "pipeline.log"), "w", encoding="utf-8") as fh:
            fh.write("[12:00:00] normalize → 1080x1080\n")
        return {
            "ok": False,
            "run_dir": run_dir,
            "error": "no configured OCR backend completed (ppocr-v6: cudnn not found)",
            "failed_stage": "ocr",
        }

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=broken.jpg", data=b"z", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "failed"
        assert status.get("failed_stage") == "ocr"
        assert status.get("error_code") == "cudnn_unavailable"
        assert "cuDNN" in status.get("error_hint", "")
        assert status.get("user_title") == "GPU library (cuDNN) issue"
    finally:
        server.shutdown()
        server.server_close()


def test_process_rejects_upload_when_machine_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.figma_bridge._preflight_blockers",
        lambda cfg: [{"name": "ocr:ppocr-v6", "detail": "python module paddleocr"}],
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=a.png", data=b"x", method="POST")
        try:
            urlopen(request, timeout=2)
            assert False, "upload should be rejected when doctor reports blockers"
        except HTTPError as error:
            assert error.code == 503
            payload = json.loads(error.read())
            assert "not ready" in payload["error"]
            assert payload["blockers"][0]["name"] == "ocr:ppocr-v6"
    finally:
        server.shutdown()
        server.server_close()


def test_process_allows_ocr_blockers_when_tesseract_fallback_ready(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    monkeypatch.setattr("src.figma_bridge._preflight_blockers", lambda cfg: None)
    calls = _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=a.png", data=b"x", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        assert queued["status"] == "queued"
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done"
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_process_missing_content_length_returns_400(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=a.png", data=b"x", method="POST",
                          headers={"Content-Length": "not-a-number"})
        try:
            urlopen(request, timeout=2)
            assert False, "garbage Content-Length should be rejected with a clear 400"
        except HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_process_reports_eta_from_history_after_a_prior_run(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    """First upload has no history -> no eta_s. Second upload sees the first run's recorded
    duration and gets a stage-weighted eta_s + progress_pct while running."""
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.15)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        first = json.loads(urlopen(Request(base + "/process?filename=a.png", data=b"x", method="POST"), timeout=2).read())
        during_first = json.loads(urlopen(f"{base}/process?job_id={first['job_id']}", timeout=2).read())
        assert "eta_s" not in during_first, "no history yet -> no fabricated ETA"
        assert "progress_pct" not in during_first
        _poll_job(base, first["job_id"])
        history = json.loads((inbox / ".process_history.json").read_text(encoding="utf-8"))
        assert history["durations_s"]
        assert history["runs"][0]["duration_s"] > 0

        second = json.loads(urlopen(Request(base + "/process?filename=b.png", data=b"y", method="POST"), timeout=2).read())
        time.sleep(0.05)
        during_second = json.loads(urlopen(f"{base}/process?job_id={second['job_id']}", timeout=2).read())
        assert during_second["status"] == "running"
        assert "eta_s" in during_second and during_second["eta_s"] >= 0
        assert during_second["eta_sample_size"] == 1
        assert during_second.get("elapsed_s") is not None
        assert "progress_pct" in during_second
        assert 0 < during_second["progress_pct"] < 100
        _poll_job(base, second["job_id"])
    finally:
        server.shutdown()
        server.server_close()


def test_process_poll_returns_stage_from_pipeline_log(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)

    def run_one(image_path, run_dir, cfg):
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "pipeline.log"), "w", encoding="utf-8") as fh:
            fh.write("[12:00:00] normalize → 1080x1080\n")
            fh.write("[12:00:02] ocr[doctr] → 3 lines\n")
            fh.write("[12:00:10] text analysis → 2 blocks, 1 styles\n")
        time.sleep(0.2)
        design = {"id": "upload", "canvas": {"w": 10, "h": 10}, "layers": []}
        design_path = os.path.join(run_dir, "design.json")
        with open(design_path, "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        if (cfg.get("figma") or {}).get("enabled"):
            from src import figma_import
            figma_import.import_design(design_path, run_dir, cfg)
        return {"ok": True, "run_dir": run_dir}

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(base + "/process?filename=a.png", data=b"x", method="POST"), timeout=2).read())
        saw_stage = False
        deadline = time.time() + 3
        while time.time() < deadline:
            status = json.loads(urlopen(f"{base}/process?job_id={queued['job_id']}", timeout=2).read())
            if status.get("stage") == "text":
                saw_stage = True
                break
            time.sleep(0.05)
        assert saw_stage, status
        _poll_job(base, queued["job_id"])
    finally:
        server.shutdown()
        server.server_close()


def test_process_queued_job_has_no_elapsed_or_eta(tmp_path, monkeypatch):
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.4)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        queued = json.loads(urlopen(Request(base + "/process?filename=a.png", data=b"x", method="POST"), timeout=2).read())
        status = json.loads(urlopen(f"{base}/process?job_id={queued['job_id']}", timeout=2).read())
        if status["status"] == "queued":
            assert "elapsed_s" not in status
            assert "eta_s" not in status
        _poll_job(base, queued["job_id"])
    finally:
        server.shutdown()
        server.server_close()


def test_history_records_stage_fractions_from_pipeline_log(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "pipeline.log"
    log_path.write_text(
        "[12:00:00] normalize → 1080x1080\n"
        "[12:00:10] ocr[doctr] → 3 lines\n"
        "[12:01:00] text analysis → 2 blocks, 1 styles\n"
        "[12:01:30] done in 90.0s\n",
        encoding="utf-8",
    )
    fracs = _parse_stage_fractions(str(run_dir), 90.0)
    assert fracs is not None
    assert fracs["normalize"] > 0
    assert fracs["ocr"] > fracs["normalize"]
    _record_history(str(inbox), 90.0, str(run_dir))
    history = _read_history(str(inbox))
    assert history["runs"][0]["stage_fractions"]["ocr"] > 0


def test_estimate_eta_uses_stage_weighted_progress(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    history_path = inbox / ".process_history.json"
    history_path.write_text(
        json.dumps({
            "durations_s": [100.0, 110.0],
            "runs": [
                {
                    "duration_s": 100.0,
                    "stage_fractions": {
                        "normalize": 0.05, "ocr": 0.35, "text": 0.1,
                        "residual": 0.05, "qwen": 0.05, "sam": 0.1,
                        "elements": 0.05, "merge": 0.05, "reconstruct": 0.1,
                        "layout": 0.05, "design": 0.05, "preview": 0.02,
                        "figma": 0.01, "export": 0.01, "qa": 0.01,
                    },
                },
                {"duration_s": 110.0},
            ],
        }),
        encoding="utf-8",
    )
    early_eta, sample_size, early_pct = _estimate_eta(str(inbox), 5.0, "ocr")
    late_eta, _, late_pct = _estimate_eta(str(inbox), 70.0, "layout")
    assert sample_size == 2
    assert early_eta is not None and late_eta is not None
    assert late_eta < early_eta
    assert late_pct > early_pct


def test_process_triggers_auto_repair_on_failed_qa(tmp_path, monkeypatch):
    """Bridge upload jobs must run harness_loop when QA fails before reporting done."""
    _allow_machine_ready(monkeypatch)
    repair_calls = []

    def fake_execute_repairs(run_dir, cfg, max_iterations=2, run_one=None):
        repair_calls.append(run_dir)
        qa_path = os.path.join(run_dir, "qa.json")
        if os.path.exists(qa_path):
            qa = json.loads(open(qa_path, encoding="utf-8").read())
            qa["ok"] = True
            with open(qa_path, "w", encoding="utf-8") as fh:
                json.dump(qa, fh)
        return {"stopped": "qa_ok", "qa_ok": True, "iterations": 1, "attempts": []}

    monkeypatch.setattr("src.harness_loop.execute_repairs", fake_execute_repairs)
    monkeypatch.setattr(
        "src.harness_loop._run_critic_pass",
        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                         "blockers": [], "filtered_repairs": []},
    )
    monkeypatch.setattr(
        "src.harness_loop._run_fixer_pass",
        lambda rd, cfg, c: {"cfg": cfg, "fixes": []},
    )

    def run_one(image_path, run_dir, cfg, start_from="normalize"):
        os.makedirs(run_dir, exist_ok=True)
        design = {"id": "repair-test", "canvas": {"w": 10, "h": 10}, "layers": []}
        with open(os.path.join(run_dir, "design.json"), "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        with open(os.path.join(run_dir, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "ok": False,
                "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high"}],
            }, fh)
        with open(os.path.join(run_dir, "runtime_report.json"), "w", encoding="utf-8") as fh:
            json.dump({"input": image_path}, fh)
        return {"ok": True, "run_dir": run_dir, "runtime_ok": True, "duration_s": 0.01}

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=qa-fail.png", data=b"\x89PNG", method="POST",
                          headers={"Content-Type": "application/octet-stream"})
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done", status
        assert len(repair_calls) == 1
        assert status["result"].get("repair", {}).get("qa_ok") is True
        assert status["result"].get("qa_ok") is True
        assert status.get("harness_rounds") == 1
        assert status.get("harness_stopped") == "qa_ok_after_repairs"
        assert status.get("final_qa_ok") is True
        assert status.get("staged") is True
        assert status.get("doc_id") == "repair-test"
        assert (tmp_path / "inbox" / "uploads" / queued["job_id"] / "run" / "harness_loop.json").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_process_triggers_auto_repair_when_staging_fails(tmp_path, monkeypatch):
    """Failed figma staging should trigger repair before the plugin sees done."""
    _allow_machine_ready(monkeypatch)
    repair_calls = []

    def fake_execute_repairs(run_dir, cfg, max_iterations=2, run_one=None):
        repair_calls.append(run_dir)
        return {"stopped": "already_ok", "qa_ok": True, "iterations": 0, "attempts": []}

    monkeypatch.setattr("src.harness_loop.execute_repairs", fake_execute_repairs)
    monkeypatch.setattr(
        "src.harness_loop._run_critic_pass",
        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                         "blockers": [], "filtered_repairs": []},
    )
    monkeypatch.setattr(
        "src.harness_loop._run_fixer_pass",
        lambda rd, cfg, c: {"cfg": cfg, "fixes": []},
    )

    stage_calls = {"n": 0}
    original_stage = __import__("src.figma_bridge", fromlist=["_stage_job_output"])._stage_job_output

    def flaky_stage(inbox, run_dir, cfg):
        stage_calls["n"] += 1
        if stage_calls["n"] == 1:
            return {"staged": False, "doc_id": None, "layer_count": None,
                    "staging_error": "inbox.json missing", "design_url": "/design.json"}
        return original_stage(inbox, run_dir, cfg)

    monkeypatch.setattr("src.figma_bridge._stage_job_output", flaky_stage)

    def run_one(image_path, run_dir, cfg, start_from="normalize"):
        os.makedirs(run_dir, exist_ok=True)
        design = {"id": "stage-repair", "canvas": {"w": 10, "h": 10}, "layers": []}
        with open(os.path.join(run_dir, "design.json"), "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        with open(os.path.join(run_dir, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True}, fh)
        return {"ok": True, "run_dir": run_dir, "runtime_ok": True, "duration_s": 0.01}

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=stage.png", data=b"x", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done", status
        assert len(repair_calls) == 1
        assert stage_calls["n"] >= 2
        assert status.get("harness_rounds") == 1
        assert status.get("harness_stopped") == "qa_ok_after_repairs"
        assert status.get("final_qa_ok") is True
        assert status.get("staged") is True
    finally:
        server.shutdown()
        server.server_close()


def test_process_skips_harness_when_run_one_already_healed(tmp_path, monkeypatch):
    """Bridge must not double-run harness when run_one already fixed QA."""
    _allow_machine_ready(monkeypatch)
    repair_calls = []

    def fake_execute_repairs(run_dir, cfg, max_iterations=2, run_one=None):
        repair_calls.append(run_dir)
        return {"stopped": "qa_ok", "qa_ok": True, "iterations": 1, "attempts": []}

    monkeypatch.setattr("src.harness_loop.execute_repairs", fake_execute_repairs)

    def run_one(image_path, run_dir, cfg, start_from="normalize"):
        os.makedirs(run_dir, exist_ok=True)
        design = {"id": "pre-healed", "canvas": {"w": 10, "h": 10}, "layers": []}
        with open(os.path.join(run_dir, "design.json"), "w", encoding="utf-8") as fh:
            json.dump(design, fh)
        with open(os.path.join(run_dir, "qa.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True}, fh)
        harness_summary = {
            "rounds_completed": 2,
            "stopped": "qa_ok_after_repairs",
            "qa_ok": True,
            "rounds": [],
        }
        with open(os.path.join(run_dir, "harness_loop.json"), "w", encoding="utf-8") as fh:
            json.dump(harness_summary, fh)
        return {
            "ok": True,
            "run_dir": run_dir,
            "runtime_ok": True,
            "duration_s": 0.01,
            "qa_ok": True,
            "repair": harness_summary,
            "harness_rounds": 2,
            "harness_stopped": "qa_ok_after_repairs",
        }

    fake = types.ModuleType("run_pipeline")
    fake.run_one = run_one
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/process?filename=healed.png", data=b"x", method="POST")
        queued = json.loads(urlopen(request, timeout=2).read())
        status = _poll_job(base, queued["job_id"])
        assert status["status"] == "done", status
        assert len(repair_calls) == 0
        assert status.get("harness_rounds") == 2
        assert status.get("harness_stopped") == "qa_ok_after_repairs"
        assert status.get("final_qa_ok") is True
    finally:
        server.shutdown()
        server.server_close()


def test_estimate_eta_returns_none_without_history(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    eta, sample_size, progress_pct = _estimate_eta(str(inbox), 12.0, "ocr")
    assert eta is None and sample_size == 0 and progress_pct is None


def test_repo_update_runs_git_pull(tmp_path, monkeypatch):
    import subprocess

    from src import figma_bridge

    calls = []

    class FakeResult:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if list(args[:2]) == ["git", "fetch"]:
            return FakeResult()
        if list(args[:2]) == ["git", "pull"]:
            return FakeResult(stdout="Already up to date.\n")
        if list(args[:2]) == ["git", "rev-parse"]:
            return FakeResult(stdout="cd807f2\n")
        return FakeResult()

    monkeypatch.setattr(figma_bridge, "_repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / ".git").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "stamp_plugin_build.py").write_text("print('ok')", encoding="utf-8")

    result = figma_bridge._run_repo_update()
    assert result["ok"] is True
    assert result["commit"] == "cd807f2"
    assert any(call[:3] == ["git", "pull", "newrepo"] for call in calls)


def test_repo_update_http_endpoint(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    _allow_machine_ready(monkeypatch)

    from src import figma_bridge

    monkeypatch.setattr(
        figma_bridge, "_run_repo_update",
        lambda remote="newrepo", branch="main": {"ok": True, "commit": "abc1234", "restart_required": True},
    )

    handler = make_handler(str(inbox), config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{port}/repo/update", data=b"", method="POST")
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["commit"] == "abc1234"
    finally:
        server.shutdown()
        server.server_close()
def test_stage_job_output_rejects_stale_manifest(tmp_path, monkeypatch):
    from src import figma_bridge

    inbox = tmp_path / "inbox"
    run_dir = tmp_path / "run"
    inbox.mkdir()
    run_dir.mkdir()
    (run_dir / "design.json").write_text(json.dumps({"layers": []}), encoding="utf-8")
    (inbox / "inbox.json").write_text(json.dumps({
        "doc_id": "old", "run_dir": str(tmp_path / "old-run"),
        "summary": {"layers": 99},
    }), encoding="utf-8")
    monkeypatch.setattr("src.figma_import.import_design", lambda *a, **k: {
        "ok": False, "error": "disk full",
    })

    result = figma_bridge._stage_job_output(
        str(inbox), str(run_dir), {"figma": {"inbox": str(inbox)}})

    assert result["staged"] is False
    assert result["staging_error"] == "disk full"
