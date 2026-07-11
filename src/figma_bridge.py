"""figma_bridge.py — tiny local HTTP bridge so the Figma plugin can close the loop headlessly.

Serves the staged inbox (design.json + assets) to the plugin and receives the exported PNG
back into the run dir. Zero deps (http.server). Start it before clicking "Import latest":

    python -m src.figma_bridge --inbox ~/figma-inbox --port 8790

Endpoints:
    GET  /inbox.json           -> manifest (design path, assets dir, export_to)
    GET  /design.json          -> the staged design.json
    GET  /asset?path=<rel>     -> an asset PNG (resolved under inbox/assets or run dir)
    POST /export               -> body = PNG bytes; written to manifest.export_to
"""
from __future__ import annotations
import argparse, json, os, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def make_handler(inbox):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body=b"", ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _manifest(self):
            path = os.path.join(inbox, "inbox.json")
            return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None

        def _staged_root(self, manifest):
            rel = (manifest or {}).get("staged_dir") or "."
            root = os.path.realpath(os.path.join(inbox, rel))
            allowed = os.path.realpath(inbox)
            return root if os.path.commonpath([root, allowed]) == allowed else allowed

        def _safe_file(self, root, rel):
            rel = str(rel or "").lstrip("/\\")
            path = os.path.realpath(os.path.join(root, rel))
            return path if os.path.commonpath([path, root]) == root else None

        def _content_length(self, max_bytes):
            """Parse Content-Length safely; returns None if missing/non-numeric/<=0/oversized
            so callers can respond with a clear 400 instead of letting ValueError propagate
            up to socketserver's handle_error() (which drops the connection silently)."""
            raw = self.headers.get("Content-Length")
            try:
                n = int(raw)
            except (TypeError, ValueError):
                return None
            if n <= 0 or n > max_bytes:
                return None
            return n

        def do_OPTIONS(self):
            return self._send(204)

        def do_GET(self):
            u = urlparse(self.path)
            manifest = self._manifest()
            if u.path == "/health":
                payload = {"ok": True, "service": "ad-decompiler-bridge",
                           "has_run": bool(manifest), "schema_version": (manifest or {}).get("schema_version")}
                return self._send(200, json.dumps(payload).encode(), "application/json")
            if u.path == "/inbox.json":
                p = os.path.join(inbox, "inbox.json")
                return self._send(200, open(p, "rb").read(), "application/json") if os.path.exists(p) else self._send(404)
            if u.path == "/design.json":
                root = self._staged_root(manifest)
                p = self._safe_file(root, (manifest or {}).get("design", "design.json"))
                return self._send(200, open(p, "rb").read(), "application/json") if p and os.path.exists(p) else self._send(404)
            if u.path == "/run.json":
                if not manifest:
                    return self._send(404)
                payload = {"doc_id": manifest.get("doc_id"), "staged_at": manifest.get("staged_at"),
                           "summary": manifest.get("summary") or {}, "preview": manifest.get("preview")}
                return self._send(200, json.dumps(payload).encode(), "application/json")
            if u.path == "/preview.png":
                root = self._staged_root(manifest)
                p = self._safe_file(root, (manifest or {}).get("preview"))
                return self._send(200, open(p, "rb").read(), "image/png") if p and os.path.exists(p) else self._send(404)
            if u.path == "/asset":
                rel = parse_qs(u.query).get("path", [""])[0]
                root = self._staged_root(manifest)
                p = self._safe_file(root, rel)
                if p and os.path.isfile(p):
                    ext = os.path.splitext(p)[1].lower()
                    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                             ".webp": "image/webp", ".svg": "image/svg+xml"}.get(ext, "application/octet-stream")
                    return self._send(200, open(p, "rb").read(), ctype)
                return self._send(404)
            return self._send(404)

        def do_POST(self):
            route = urlparse(self.path).path
            if route == "/export":
                n = self._content_length(max_bytes=32 * 1024 * 1024)
                if n is None:
                    return self._send(400, b'{"ok":false,"error":"missing, invalid, or oversized Content-Length"}',
                                       "application/json")
                data = self.rfile.read(n)
                man = self._manifest() or {}
                out = man.get("export_to") or os.path.join(inbox, "figma_export.png")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                fd, temp_path = tempfile.mkstemp(prefix=".figma-export-", suffix=".png",
                                                 dir=os.path.dirname(out))
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(temp_path, out)
                return self._send(200, b'{"ok":true}', "application/json")
            if route == "/report":
                n = self._content_length(max_bytes=2 * 1024 * 1024)
                if n is None:
                    return self._send(400, b'{"ok":false,"error":"missing, invalid, or oversized Content-Length"}',
                                       "application/json")
                try:
                    report = json.loads(self.rfile.read(n).decode("utf-8"))
                except Exception:
                    return self._send(400, b'{"ok":false,"error":"invalid json"}', "application/json")
                man = self._manifest() or {}
                outputs = []
                run_dir = man.get("run_dir")
                if run_dir:
                    outputs.append(os.path.join(run_dir, "figma_report.json"))
                staged = self._staged_root(man)
                if staged:
                    outputs.append(os.path.join(staged, "figma_report.json"))
                for out in outputs:
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    fd, temp_path = tempfile.mkstemp(prefix=".figma-report-", suffix=".json",
                                                     dir=os.path.dirname(out))
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(report, fh, ensure_ascii=False, indent=2)
                    os.replace(temp_path, out)
                return self._send(200, b'{"ok":true}', "application/json")
            return self._send(404)

        def log_message(self, *a):  # quiet
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=os.path.expanduser("~/figma-inbox"))
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1",
                     help="bind address; use 0.0.0.0 to accept connections from other machines (e.g. over Tailscale)")
    a = ap.parse_args()
    os.makedirs(a.inbox, exist_ok=True)
    print(f"ad-decompiler bridge on http://{a.host}:{a.port} serving {a.inbox}")
    ThreadingHTTPServer((a.host, a.port), make_handler(a.inbox)).serve_forever()


if __name__ == "__main__":
    main()
