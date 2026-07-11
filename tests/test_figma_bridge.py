import json
import threading
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
