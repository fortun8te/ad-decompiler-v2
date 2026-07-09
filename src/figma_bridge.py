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
import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def make_handler(inbox):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body=b"", ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/inbox.json":
                p = os.path.join(inbox, "inbox.json")
                return self._send(200, open(p, "rb").read(), "application/json") if os.path.exists(p) else self._send(404)
            if u.path == "/design.json":
                p = os.path.join(inbox, "design.json")
                return self._send(200, open(p, "rb").read(), "application/json") if os.path.exists(p) else self._send(404)
            if u.path == "/asset":
                rel = parse_qs(u.query).get("path", [""])[0]
                for base in (inbox, os.path.join(inbox, "assets")):
                    p = os.path.normpath(os.path.join(base, os.path.basename(rel)))
                    if os.path.exists(p):
                        return self._send(200, open(p, "rb").read(), "image/png")
                return self._send(404)
            return self._send(404)

        def do_POST(self):
            if urlparse(self.path).path == "/export":
                n = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(n)
                man = json.load(open(os.path.join(inbox, "inbox.json")))
                out = man.get("export_to") or os.path.join(inbox, "figma_export.png")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(data)
                return self._send(200, b'{"ok":true}', "application/json")
            return self._send(404)

        def log_message(self, *a):  # quiet
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=os.path.expanduser("~/figma-inbox"))
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    os.makedirs(a.inbox, exist_ok=True)
    print(f"ad-decompiler bridge on http://127.0.0.1:{a.port} serving {a.inbox}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.inbox)).serve_forever()


if __name__ == "__main__":
    main()
