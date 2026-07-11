import json
import sys
import threading
import time
import types
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.figma_bridge import make_handler


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


def _start_server(inbox):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox)))
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


def test_process_uploads_runs_pipeline_and_stages_inbox(tmp_path, monkeypatch):
    calls = _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), None))
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
        assert calls[0]["cfg"]["figma"]["inbox"] == str(inbox)

        # the uploaded filename was sanitized into the job dir, not written wherever the
        # caller-supplied filename would otherwise point (e.g. path traversal).
        assert "my-ad.png" in calls[0]["image_path"] or "my ad.png" not in calls[0]["image_path"]

        manifest = json.loads(urlopen(base + "/inbox.json", timeout=2).read())
        assert manifest["doc_id"] == "upload"
        design = json.loads(urlopen(base + "/design.json", timeout=2).read())
        assert design["id"] == "upload"
    finally:
        server.shutdown()
        server.server_close()


def test_process_rejects_a_second_upload_while_one_is_running(tmp_path, monkeypatch):
    _install_fake_run_pipeline(monkeypatch, sleep_s=0.3)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), None))
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


def test_process_reports_pipeline_failure_without_crashing_the_bridge(tmp_path, monkeypatch):
    _install_fake_run_pipeline(monkeypatch, ok=False)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), None))
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


def test_process_missing_content_length_returns_400(tmp_path, monkeypatch):
    _install_fake_run_pipeline(monkeypatch)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(inbox), None))
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
