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
    _atomic_write,
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


def test_roundtrip_callbacks_cannot_attach_to_a_newer_staged_run(tmp_path):
    """A Figma import can finish after the user has uploaded another image.

    The late PNG/report must fail clearly instead of silently writing into the newer run
    selected by the mutable top-level inbox manifest.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    old_run = tmp_path / "old-run"
    new_run = tmp_path / "new-run"
    old_run.mkdir()
    new_run.mkdir()
    staged = inbox / "runs" / "new-doc"
    staged.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "doc_id": "new-doc",
        "roundtrip_token": "new-token",
        "design": "design.json",
        "staged_dir": "runs/new-doc",
        "run_dir": str(new_run),
        "export_to": str(new_run / "figma_export.png"),
    }
    (inbox / "inbox.json").write_text(json.dumps(manifest), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        old_report = {
            "doc_id": "old-doc", "roundtrip_token": "old-token",
            "report": {"ok": True},
        }
        try:
            urlopen(Request(base + "/report", data=json.dumps(old_report).encode(), method="POST",
                            headers={"Content-Type": "application/json"}), timeout=2)
            assert False, "stale compiler report should be rejected"
        except HTTPError as error:
            assert error.code == 409
            assert "stale Figma callback" in json.loads(error.read())["error"]
        try:
            urlopen(Request(base + "/export?doc_id=old-doc&roundtrip_token=old-token",
                            data=b"old-png", method="POST"), timeout=2)
            assert False, "stale Figma PNG should be rejected"
        except HTTPError as error:
            assert error.code == 409
        assert not (new_run / "figma_export.png").exists()
        assert not (new_run / "figma_report.json").exists()

        current_report = {
            "doc_id": "new-doc", "roundtrip_token": "new-token",
            "report": {"ok": True},
        }
        report_response = json.loads(urlopen(Request(
            base + "/report", data=json.dumps(current_report).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        ), timeout=2).read())
        assert report_response["ok"] is True
        export_response = json.loads(urlopen(Request(
            base + "/export?doc_id=new-doc&roundtrip_token=new-token",
            data=b"new-png", method="POST",
        ), timeout=2).read())
        assert export_response["ok"] is True
        assert (new_run / "figma_report.json").exists()
        assert (new_run / "figma_export.png").read_bytes() == b"new-png"
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


def test_process_preview_endpoints_serve_upload_and_snapshot(tmp_path, monkeypatch):
    """Plugin live preview can fetch the uploaded ad while the pipeline runs."""
    _allow_machine_ready(monkeypatch)
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.4)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    config = _write_passing_config(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), config))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    png = b"\x89PNG\r\n\x1a\n" + b"preview-test-bytes"
    try:
        queued = json.loads(urlopen(
            Request(base + "/process?filename=ad.png", data=png, method="POST"), timeout=2,
        ).read())
        assert queued["input_url"] == f"/process/input?job_id={queued['job_id']}"
        assert queued["snapshot_url"] == f"/process/snapshot?job_id={queued['job_id']}"
        input_bytes = urlopen(base + queued["input_url"], timeout=2).read()
        assert input_bytes == png
        snapshot_bytes = urlopen(base + queued["snapshot_url"], timeout=2).read()
        assert snapshot_bytes == png
        status = json.loads(urlopen(base + "/process?job_id=" + queued["job_id"], timeout=2).read())
        assert status["input_url"] == queued["input_url"]
        assert status["snapshot_url"] == queued["snapshot_url"]
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
    _record_history(
        str(inbox), 90.0, str(run_dir),
        job_id="job-123", filename="/tmp/offer.png", doc_id="offer-doc",
        finished_at=1234.5, qa_ok=True, layer_count=18,
    )
    history = _read_history(str(inbox))
    assert history["runs"][0]["stage_fractions"]["ocr"] > 0
    assert history["runs"][0]["filename"] == "offer.png"
    assert history["runs"][0]["doc_id"] == "offer-doc"
    assert history["runs"][0]["qa_ok"] is True


def test_history_endpoint_exposes_recent_conversion_identity_without_run_paths(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _record_history(
        str(inbox), 24.0,
        job_id="job-new", filename="new-ad.png", doc_id="new-doc",
        finished_at=200.0, qa_ok=True, layer_count=12,
    )
    _record_history(
        str(inbox), 18.0,
        job_id="job-old", filename="old-ad.png", doc_id="old-doc",
        finished_at=100.0, qa_ok=True, layer_count=9,
    )
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = json.loads(urlopen(base + "/history", timeout=2).read())
        assert payload["ok"] is True
        assert payload["max"] >= 40
        assert [row["job_id"] for row in payload["runs"]] == ["job-old", "job-new"]
        assert payload["runs"][0]["filename"] == "old-ad.png"
        assert "run_dir" not in payload["runs"][0]
    finally:
        server.shutdown()
        server.server_close()


def test_history_keeps_forty_runs(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(45):
        _record_history(
            str(inbox), 10.0 + i,
            job_id=f"job-{i}", filename=f"ad-{i}.png", doc_id=f"doc-{i}",
            finished_at=1000.0 + i, qa_ok=True, layer_count=i,
        )
    history = _read_history(str(inbox))
    assert len(history["runs"]) == 40
    assert history["runs"][0]["job_id"] == "job-5"
    assert history["runs"][-1]["job_id"] == "job-44"


def test_history_restage_promotes_past_run_into_inbox(tmp_path, monkeypatch):
    from pathlib import Path
    from src import figma_bridge as fb

    inbox = tmp_path / "inbox"
    upload = inbox / "uploads" / "job-past" / "run"
    upload.mkdir(parents=True)
    (upload / "design.json").write_text('{"schema_version":2,"id":"past-doc","layers":[]}', encoding="utf-8")
    _record_history(
        str(inbox), 12.0,
        job_id="job-past", filename="past.png", doc_id="past-doc",
        finished_at=50.0, qa_ok=True, layer_count=3,
    )

    def fake_stage(inbox_path, run_dir, cfg):
        assert Path(run_dir) == upload
        manifest = {
            "doc_id": "past-doc", "run_dir": str(upload), "design": "design.json",
            "summary": {"layers": 3},
        }
        (Path(inbox_path) / "inbox.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"staged": True, "doc_id": "past-doc", "layer_count": 3,
                "staging_error": None, "design_url": "/design.json", "manifest": manifest}

    monkeypatch.setattr(fb, "_stage_job_output", fake_stage)
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        req = Request(
            base + "/history/restage",
            data=json.dumps({"job_id": "job-past"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.loads(urlopen(req, timeout=2).read())
        assert payload["ok"] is True
        assert payload["doc_id"] == "past-doc"
        assert payload["job_id"] == "job-past"
        staged = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
        assert staged["doc_id"] == "past-doc"
    finally:
        server.shutdown()
        server.server_close()


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


# ---------------------------------------------------------------------------
# Deferred UI item 1: /history carries visual_score + qa_ok so the plugin can
# render a QA badge for a staged run that has no live-session job.
# ---------------------------------------------------------------------------
def test_history_round_trips_visual_score_for_qa_badge(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _record_history(
        str(inbox), 30.0, job_id="j1", filename="ad.png", doc_id="doc-1",
        finished_at=100.0, qa_ok=True, visual_score=0.9123, layer_count=7,
    )
    # A stray bool must never masquerade as a score (would render "QA 1.00").
    _record_history(
        str(inbox), 12.0, job_id="j2", filename="b.png", doc_id="doc-2",
        finished_at=110.0, qa_ok=True, visual_score=True,
    )
    runs = {row["doc_id"]: row for row in _read_history(str(inbox))["runs"]}
    assert runs["doc-1"]["visual_score"] == 0.9123
    assert runs["doc-1"]["qa_ok"] is True
    assert "visual_score" not in runs["doc-2"]

    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        served = {row["doc_id"]: row for row in json.loads(urlopen(base + "/history", timeout=2).read())["runs"]}
        assert served["doc-1"]["visual_score"] == 0.9123
        assert served["doc-1"]["qa_ok"] is True
        assert "run_dir" not in served["doc-1"]  # still path-free
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Deferred UI item 2: a staged run's QA evidence is retrievable by doc_id, not
# only by a live job_id -- via the enriched /run.json and /run-summary?doc_id=.
# ---------------------------------------------------------------------------
def _write_qa(run_dir, *, ok=True, visual_score=0.88, complete=True):
    structural = (
        {"background": {"ok": True}, "layer_alpha": [], "element_recall": 1.0, "hard_fails": []}
        if complete else {}
    )
    (run_dir / "qa.json").write_text(json.dumps({
        "ok": ok, "visual_score": visual_score, "ssim": 0.95,
        "hard_fails": [], "structural": structural,
    }), encoding="utf-8")


def test_run_json_exposes_staged_qa_evidence(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    staged = inbox / "runs" / "demo"
    staged.mkdir(parents=True)
    (staged / "design.json").write_text(json.dumps({"id": "demo", "canvas": {"w": 1, "h": 1}, "layers": []}), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_qa(run_dir, visual_score=0.88)
    manifest = {
        "schema_version": 2, "doc_id": "demo", "design": "design.json",
        "staged_dir": "runs/demo", "run_dir": str(run_dir), "staged_at": 123,
        "summary": {"layers": 0}, "preview": "preview.png",
    }
    (inbox / "inbox.json").write_text(json.dumps(manifest), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        run = json.loads(urlopen(base + "/run.json", timeout=2).read())
        assert run["ok"] is True
        assert run["doc_id"] == "demo"
        assert run["qa_ok"] is True
        assert run["visual_score"] == 0.88
        assert run["qa_evidence_complete"] is True
        assert run["hard_fails"] == []
        # legacy identity fields remain (backward compatible)
        assert run["staged_at"] == 123 and run["preview"] == "preview.png"
    finally:
        server.shutdown()
        server.server_close()


def test_run_json_returns_json_404_when_nothing_staged(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        urlopen(base + "/run.json", timeout=2)
        assert False, "empty inbox should 404"
    except HTTPError as error:
        assert error.code == 404
        assert json.loads(error.read())["ok"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_run_summary_by_doc_id_from_currently_staged_manifest(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_qa(run_dir, visual_score=0.81)
    manifest = {"schema_version": 2, "doc_id": "staged", "design": "design.json",
                "staged_dir": ".", "run_dir": str(run_dir)}
    (inbox / "inbox.json").write_text(json.dumps(manifest), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        summary = json.loads(urlopen(base + "/run-summary?doc_id=staged", timeout=2).read())
        assert summary["ok"] is True
        assert summary["doc_id"] == "staged"
        assert summary["staged"] is True
        assert summary["visual_score"] == 0.81
        assert summary["final_qa_ok"] is True
        assert summary["manifest"]["doc_id"] == "staged"
    finally:
        server.shutdown()
        server.server_close()


def test_run_summary_by_doc_id_reconstructs_from_history_after_restart(tmp_path):
    """No live job_id (bridge restarted) -> resolve run_dir from the history job_id."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    job_id = "abc123def456"
    run_dir = inbox / "uploads" / job_id / "run"
    run_dir.mkdir(parents=True)
    _write_qa(run_dir, visual_score=0.77, complete=False)
    _record_history(
        str(inbox), 20.0, str(run_dir), job_id=job_id, filename="old.png",
        doc_id="old-doc", finished_at=100.0, qa_ok=True, visual_score=0.77,
    )
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        summary = json.loads(urlopen(base + "/run-summary?doc_id=old-doc", timeout=2).read())
        assert summary["ok"] is True
        assert summary["doc_id"] == "old-doc"
        assert summary["job_id"] == job_id
        assert summary["staged"] is False
        assert summary["visual_score"] == 0.77
        assert summary["final_qa_ok"] is True
        try:
            urlopen(base + "/run-summary?doc_id=does-not-exist", timeout=2)
            assert False, "unknown doc_id should 404"
        except HTTPError as error:
            assert error.code == 404
            assert json.loads(error.read())["ok"] is False
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Task 3: /health answers staged doc_id + plugin client build + job state in one call.
# ---------------------------------------------------------------------------
def test_health_reports_staged_doc_and_client_and_job_state_in_one_call(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "plugin_client.json").write_text(json.dumps({"build": 9, "label": "v2+b9"}), encoding="utf-8")
    (inbox / "inbox.json").write_text(json.dumps({"schema_version": 2, "doc_id": "staged-doc", "staged_at": 555}), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urlopen(base + "/health", timeout=4).read())
        assert health["ok"] is True
        assert health["has_run"] is True
        assert health["staged_doc_id"] == "staged-doc"
        assert health["staged_at"] == 555
        assert health["plugin_client"]["build"] == 9
        assert "active_job" in health and health["active_job"] is None
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Task 2: path-traversal safety (incl. absolute + cross-drive), caching headers,
# graceful JSON errors, and torn-read safety under concurrent restaging.
# ---------------------------------------------------------------------------
def test_asset_rejects_absolute_and_cross_drive_paths(tmp_path):
    from urllib.parse import quote

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    staged = inbox / "runs" / "demo"
    staged.mkdir(parents=True)
    (staged / "keep.png").write_bytes(b"ok")
    (inbox / "inbox.json").write_text(json.dumps(
        {"doc_id": "demo", "design": "design.json", "staged_dir": "runs/demo"}), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        # A crafted cross-drive path makes os.path.commonpath raise ValueError; the handler
        # must reject it as 404, not 500 (the ValueError guard) -- and never leak a file.
        for bad in ["../../secret.txt", "..\\..\\secret", "/etc/passwd",
                    "C:/Windows/win.ini", "Z:/nope/passwd"]:
            try:
                urlopen(base + "/asset?path=" + quote(bad, safe=""), timeout=2)
                assert False, f"expected 404 for {bad}"
            except HTTPError as error:
                assert error.code == 404, f"{bad} -> {error.code}"
        # the legitimate staged asset still serves
        assert urlopen(base + "/asset?path=keep.png", timeout=2).read() == b"ok"
    finally:
        server.shutdown()
        server.server_close()


def test_preview_and_asset_use_revalidate_cache_json_stays_no_store(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    staged = inbox / "runs" / "demo"
    staged.mkdir(parents=True)
    (staged / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (inbox / "inbox.json").write_text(json.dumps({
        "doc_id": "demo", "design": "design.json", "staged_dir": "runs/demo",
        "preview": "preview.png",
    }), encoding="utf-8")
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        # The plugin busts the cache with ?t=<staged_at>; the image is cacheable but must
        # revalidate so two stages in the same wall-clock second never show a stale render.
        resp = urlopen(base + "/preview.png?t=123", timeout=2)
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        assert resp.headers.get("Cache-Control") == "no-cache"
        assert resp.read() == b"\x89PNG\r\n\x1a\nfake"
        json_resp = urlopen(base + "/inbox.json", timeout=2)
        assert json_resp.headers.get("Content-Type") == "application/json"
        assert json_resp.headers.get("Cache-Control") == "no-store"
    finally:
        server.shutdown()
        server.server_close()


def test_handler_returns_json_500_instead_of_dropping_connection(tmp_path, monkeypatch):
    import src.figma_bridge as fb

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fb, "_read_history", boom)
    try:
        urlopen(base + "/history", timeout=2)
        assert False, "an unhandled handler error should surface as HTTP 500, not a reset"
    except HTTPError as error:
        assert error.code == 500
        body = json.loads(error.read())
        assert body["ok"] is False
        assert "kaboom" in body.get("detail", "")
    finally:
        server.shutdown()
        server.server_close()


def test_inbox_polling_never_sees_a_torn_read_during_restage(tmp_path):
    """The plugin polls /inbox.json + /design.json while a run re-stages. Atomic replace
    plus a read-retry must guarantee every 200 body is a whole JSON doc, never a torn one."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    staged = inbox / "runs" / "demo"
    staged.mkdir(parents=True)
    (staged / "design.json").write_text(
        json.dumps({"id": "demo", "canvas": {"w": 1, "h": 1}, "layers": []}), encoding="utf-8")
    base_manifest = {"schema_version": 2, "design": "design.json", "staged_dir": "runs/demo"}
    inbox_json = str(inbox / "inbox.json")
    _atomic_write(inbox_json, json.dumps({**base_manifest, "doc_id": "demo-0"}).encode())

    server = _start_server(inbox)
    base = f"http://127.0.0.1:{server.server_port}"
    stop = threading.Event()
    errors = []

    def restage():
        i = 0
        while not stop.is_set():
            i += 1
            payload = {**base_manifest, "doc_id": f"demo-{i}", "pad": "x" * (i % 400)}
            try:
                _atomic_write(inbox_json, json.dumps(payload).encode())
            except Exception as exc:  # pragma: no cover - writer must not die
                errors.append(("write", str(exc)))
            time.sleep(0.001)

    writer = threading.Thread(target=restage, daemon=True)
    writer.start()
    try:
        for _ in range(150):
            for path in ("/inbox.json", "/design.json"):
                try:
                    resp = urlopen(base + path, timeout=3)
                    assert resp.status == 200
                    parsed = json.loads(resp.read())  # a torn read raises here
                    assert "doc_id" in parsed or "canvas" in parsed
                except HTTPError as error:
                    # a transient manifest-read miss may 404 /design.json; never a 500
                    assert error.code == 404, f"{path} -> {error.code}"
                except Exception as exc:
                    errors.append((path, str(exc)))
    finally:
        stop.set()
        writer.join(timeout=2)
        server.shutdown()
        server.server_close()
    assert not errors, errors[:5]
