#!/usr/bin/env python3
"""Live activity grid for ad-decompiler benchmarks.

Background watcher tails pipeline.log files; UI gets push updates via SSE
(/events) with a /status.json poll fallback.

  python scripts/activity_grid.py --root runs/benchmark
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

STAGES = [
    "normalize", "ocr", "text", "residual", "qwen", "sam", "elements",
    "peel", "merge", "structure", "reconstruct", "layout", "design",
    "preview", "figma", "export", "diff", "qa",
]

_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("normalize", ("normalize →",)),
    ("ocr", ("ocr[",)),
    ("text", ("text analysis →",)),
    ("residual", ("residual proposals →",)),
    ("qwen", ("qwen →",)),
    ("sam", ("sam3[",)),
    ("elements", ("element fusion →",)),
    ("peel", ("peel →", "peel fallback →")),
    ("merge", ("merge →",)),
    ("structure", ("structure →", "structure fallback →")),
    ("reconstruct", ("reconstruct →",)),
    ("layout", ("layout →", "layout legacy", "layout.json →")),
    ("design", ("design.json →",)),
    ("preview", ("preview →",)),
    ("figma", ("figma import:",)),
    ("export", ("export:",)),
    ("qa", ("qa →", "diff/qa skipped")),
]

_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*)$")
_DONE_RE = re.compile(r"\bdone in\s+\d+(?:\.\d+)?s\b")
_STALL_S = 90.0
_POLL_S = 0.1  # watcher cadence — snappy for live benches
_ROOT_RECHECK_S = 1.0
_SKIP_DIRS = {
    "assets", "layers", "elements", "fused_elements", "sam3_masks",
    "text_fallback", "peel", "__pycache__", "runtime-smoke",
}
_SKIP_ROOT_NAMES = {
    ".cache", "runtime-smoke", "_activity_demo",
}
_ASSET_NAMES = ("original.png", "preview.png", "normalized.png", "diff.png")

_HERE = Path(__file__).resolve().parent
_HTML_PATH = _HERE / "activity_grid.html"
_REPO = _HERE.parent


def resolve_watch_root(root: Path) -> Path:
    """Latch onto the hottest active run folder under an umbrella like runs/.

    Prefer whichever container has the newest pipeline.log write — including a
    brand-new benchmark with only one ad so far. Never treat the umbrella itself
    as a mega-batch of unrelated single-run folders.
    """
    root = root.resolve()
    if not root.is_dir():
        return root

    candidates: list[tuple[Path, float, int]] = []

    def consider(folder: Path) -> None:
        if folder.name in _SKIP_ROOT_NAMES or folder.name in _SKIP_DIRS:
            return
        try:
            logs = [
                p for p in folder.glob("*/pipeline.log")
                if p.parent.name not in _SKIP_ROOT_NAMES and p.parent.name not in _SKIP_DIRS
            ]
        except OSError:
            return
        if not logs:
            return
        try:
            newest = max(p.stat().st_mtime for p in logs)
        except OSError:
            return
        candidates.append((folder, newest, len(logs)))

    # Only consider children of an umbrella — not the umbrella's mixed leaf runs.
    nested = False
    try:
        for child in root.iterdir():
            if child.is_dir():
                before = len(candidates)
                consider(child)
                if len(candidates) > before:
                    nested = True
    except OSError:
        pass

    if not nested:
        # root itself is the benchmark folder (e.g. --root runs/benchmark-contract).
        consider(root)

    if not candidates:
        return root

    # Hottest log wins. Tie-break: more ads (real batch) over a lone stale sibling.
    candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
    return candidates[0][0]


def parse_log(text: str) -> tuple[list[str], str | None, bool, bool]:
    """Return (done_stages, active, complete, failed)."""
    last_idx = -1
    complete = False
    failed = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        body = m.group(2) if m else line
        if "ERROR:" in body:
            failed = True
        if _DONE_RE.search(body):
            complete = True
            last_idx = len(STAGES) - 1
            continue
        for name, markers in _MARKERS:
            if any(marker in body for marker in markers):
                idx = STAGES.index(name)
                if idx > last_idx:
                    last_idx = idx
                break

    if complete or last_idx >= STAGES.index("qa"):
        return STAGES[:], None, True, failed

    if last_idx < 0:
        # Empty / non-stage log — pending, not yet active.
        if not text.strip() or not any(m in text for _, ms in _MARKERS for m in ms):
            return [], None, False, failed
        return [], "normalize", False, failed

    done = STAGES[: last_idx + 1]
    nxt = last_idx + 1
    active = STAGES[nxt] if nxt < len(STAGES) else None
    return done, active, False, failed


def _run_fingerprint(run: dict[str, Any]) -> tuple:
    return (
        run["id"],
        run["status"],
        run["active"],
        tuple(run["done"]),
        run["percent"],
        run["complete"],
        run["failed"],
        run["stalled"],
        run["has_original"],
        run["has_preview"],
        round(run["mtime"], 3),
        (run.get("last_line") or "")[:80],
    )


def _last_log_line(text: str) -> str:
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line:
            # Strip leading [HH:MM:SS] for the ticker.
            if len(line) > 10 and line[0] == "[" and "]" in line[:12]:
                return line.split("]", 1)[1].strip()
            return line
    return ""


class Tracker:
    def __init__(self, root: Path, *, auto_root: bool = True) -> None:
        self._requested_root = root
        self.root = resolve_watch_root(root) if auto_root else root
        self._auto_root = auto_root
        self.generation = 0
        self.rev = 0
        self._live_id: str | None = None
        self._started: dict[str, float] = {}
        self._cache: dict[str, dict[str, Any]] = {}  # id -> disk cache
        self._lock = threading.Lock()
        self._status: dict[str, Any] = self._empty()
        self._stop = threading.Event()
        self._changed = threading.Condition(self._lock)
        self._root_check_at = 0.0

    def _empty(self) -> dict[str, Any]:
        return {
            "stages": STAGES,
            "updated_at": time.time(),
            "generation": self.generation,
            "rev": self.rev,
            "benchmark": {
                "root": str(self.root),
                "name": self.root.name,
                "total": 0,
                "done_count": 0,
                "running_count": 0,
                "percent": 0.0,
                "runs": [],
            },
            "live": None,
        }

    def start(self) -> None:
        t = threading.Thread(target=self._watch_loop, name="activity-watch", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()
        with self._changed:
            self._changed.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    def wait_change(self, since_rev: int, timeout: float = 25.0) -> dict[str, Any]:
        """Block until rev advances past since_rev, or timeout (long-poll / SSE)."""
        deadline = time.time() + timeout
        with self._changed:
            while self.rev <= since_rev and not self._stop.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._changed.wait(timeout=min(remaining, 0.5))
            return json.loads(json.dumps(self._status))

    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:
                pass
            self._stop.wait(_POLL_S)

    def _discover_runs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        found: set[Path] = set()
        # Prefer flat benchmark layout: root/<image_id>/pipeline.log
        try:
            for child in self.root.iterdir():
                if not child.is_dir():
                    continue
                if child.name in _SKIP_DIRS or child.name in _SKIP_ROOT_NAMES:
                    continue
                if (child / "pipeline.log").is_file():
                    found.add(child)
        except OSError:
            pass
        if found:
            return sorted(found, key=lambda p: p.name.lower())

        # Fallback: shallow walk for nested demos.
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and d not in _SKIP_ROOT_NAMES
            ]
            if "pipeline.log" in filenames:
                found.add(Path(dirpath))
        return sorted(found, key=lambda p: p.name.lower())

    def _read_text_cached(self, image_id: str, log_path: Path, size: int, mtime: float) -> str:
        prev = self._cache.get(image_id)
        if prev and prev.get("size") == size and prev.get("mtime") == mtime and "text" in prev:
            return prev["text"]

        # Append-only fast path: grow from previous offset.
        if prev and prev.get("size", 0) < size and prev.get("mtime", 0) <= mtime and "text" in prev:
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(prev["size"])
                    chunk = fh.read()
                text = prev["text"] + chunk.decode("utf-8", errors="replace")
                return text
            except OSError:
                pass

        try:
            return log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _read_run(self, run_dir: Path, now: float) -> dict[str, Any]:
        image_id = run_dir.name
        log_path = run_dir / "pipeline.log"
        try:
            st = log_path.stat()
            size = st.st_size
            mtime = st.st_mtime
        except OSError:
            return {
                "id": image_id,
                "run_dir": str(run_dir),
                "done": [],
                "active": None,
                "percent": 0.0,
                "complete": False,
                "failed": False,
                "stalled": False,
                "status": "pending",
                "level": 0,
                "elapsed_s": 0.0,
                "started_at": None,
                "mtime": 0.0,
                "has_original": False,
                "has_preview": False,
                "last_line": "",
            }

        prev = self._cache.get(image_id)
        truncated = bool(prev and size < prev.get("size", 0))
        if truncated:
            self._started[image_id] = now
            self.generation += 1

        unchanged = (
            prev
            and not truncated
            and prev.get("size") == size
            and prev.get("mtime") == mtime
            and "parsed" in prev
        )

        if unchanged:
            done, active, complete, failed = prev["parsed"]
            text = prev.get("text", "")
        else:
            text = self._read_text_cached(image_id, log_path, size, mtime)
            done, active, complete, failed = parse_log(text)

        has_original = (run_dir / "original.png").is_file()
        has_preview = (run_dir / "preview.png").is_file()

        # Start clock on first real stage line (not stub "waiting").
        if image_id not in self._started and (done or active or complete):
            self._started[image_id] = now if truncated else min(mtime, now)

        started_at = self._started.get(image_id)
        elapsed = max(0.0, now - started_at) if started_at else 0.0
        stalled = (
            not complete
            and not failed
            and active is not None
            and (now - mtime) > _STALL_S
        )
        percent = 100.0 if complete else round(100.0 * len(done) / len(STAGES), 1)

        if failed and not complete:
            status = "failed"
        elif complete:
            status = "done"
        elif stalled:
            status = "stalled"
        elif active or done:
            status = "running"
        else:
            status = "pending"

        if status == "pending":
            level = 0
        elif complete:
            level = 5
        else:
            level = max(1, min(5, int(percent // 20) + 1))

        self._cache[image_id] = {
            "size": size,
            "mtime": mtime,
            "text": text,
            "parsed": (done, active, complete, failed),
        }

        return {
            "id": image_id,
            "run_dir": str(run_dir),
            "done": done,
            "active": active,
            "percent": percent,
            "complete": complete,
            "failed": failed,
            "stalled": stalled,
            "status": status,
            "level": level,
            "elapsed_s": round(elapsed, 1),
            "started_at": started_at,
            "mtime": mtime,
            "has_original": has_original,
            "has_preview": has_preview,
            "last_line": _last_log_line(text),
        }

    def refresh(self) -> None:
        now = time.time()
        # Periodically re-latch onto the hottest benchmark under an umbrella root.
        if self._auto_root and (now - self._root_check_at) > _ROOT_RECHECK_S:
            self._root_check_at = now
            nxt = resolve_watch_root(self._requested_root)
            if nxt != self.root:
                self.root = nxt
                self._cache.clear()
                self._started.clear()
                self._live_id = None
                self.generation += 1

        runs = [self._read_run(p, now) for p in self._discover_runs()]

        # Drop cache for vanished runs.
        alive = {r["id"] for r in runs}
        for key in list(self._cache):
            if key not in alive:
                self._cache.pop(key, None)
                self._started.pop(key, None)

        # Benchmarks run ONE ad at a time. The live ad is the incomplete run with
        # the newest pipeline.log mtime (or the newest overall if all are done).
        incomplete = [r for r in runs if not r["complete"] and r["status"] != "pending"]
        pending_only = [r for r in runs if r["status"] == "pending"]
        live = None
        if incomplete:
            live = max(incomplete, key=lambda r: r["mtime"])
        elif pending_only:
            # About to start / empty stub — no live stage yet.
            live = None
        elif runs:
            live = max(runs, key=lambda r: r["mtime"])

        # Reclassify: only the live ad may be running/stalled. Everything else that
        # stopped mid-pipeline is just partial (sequential batch moved on).
        for r in runs:
            if live and r["id"] == live["id"]:
                if r["failed"] and not r["complete"]:
                    r["status"] = "failed"
                elif r["complete"]:
                    r["status"] = "done"
                elif r["stalled"]:
                    r["status"] = "stalled"
                elif r["active"] or r["done"]:
                    r["status"] = "running"
                    r["stalled"] = False
                else:
                    r["status"] = "pending"
                    r["stalled"] = False
            else:
                r["stalled"] = False
                if r["complete"]:
                    r["status"] = "done"
                elif r["failed"]:
                    r["status"] = "failed"
                elif r["done"] or r["active"]:
                    # Finished earlier stages, batch moved on — not "running".
                    r["status"] = "partial"
                    r["active"] = None  # don't pulse a stage on a non-live ad
                else:
                    r["status"] = "pending"

        if live and live["id"] != self._live_id:
            self._live_id = live["id"]
            self.generation += 1

        # Refresh live pointer after reclassify.
        if live:
            live = next(r for r in runs if r["id"] == live["id"])

        done_count = sum(1 for r in runs if r["complete"])
        running_count = sum(1 for r in runs if r["status"] in ("running", "stalled"))
        bench_pct = round(sum(r["percent"] for r in runs) / len(runs), 1) if runs else 0.0

        status = {
            "stages": STAGES,
            "updated_at": now,
            "generation": self.generation,
            "rev": self.rev,
            "benchmark": {
                "root": str(self.root),
                "name": self.root.name,
                "total": len(runs),
                "done_count": done_count,
                "running_count": running_count,
                "percent": bench_pct,
                "runs": runs,
            },
            "live": None if not live else {
                "image_id": live["id"],
                "active": live["active"],
                "done": live["done"],
                "percent": live["percent"],
                "elapsed_s": live["elapsed_s"],
                "started_at": live["started_at"],
                "stalled": live["stalled"],
                "complete": live["complete"],
                "failed": live["failed"],
                "status": live["status"],
                "has_original": live["has_original"],
                "has_preview": live["has_preview"],
                "run_dir": live["run_dir"],
                "mtime": live["mtime"],
                "last_line": live.get("last_line") or "",
            },
        }

        fp = (
            self.generation,
            done_count,
            running_count,
            bench_pct,
            None if not live else (
                live["id"], live["active"], live["percent"], live["status"],
                tuple(live["done"]), live["stalled"], live["complete"],
            ),
            tuple(_run_fingerprint(r) for r in runs),
        )

        with self._lock:
            prev_fp = self._status.get("_fp")
            status["rev"] = self.rev
            status["_fp"] = fp
            if fp != prev_fp:
                self.rev += 1
                status["rev"] = self.rev
                self._status = status
                self._changed.notify_all()
            else:
                status["rev"] = self.rev
                status["_fp"] = fp
                self._status = status

    def resolve_asset(self, image_id: str, name: str) -> Path | None:
        if name not in _ASSET_NAMES:
            return None
        if "/" in image_id or "\\" in image_id or ".." in image_id:
            return None
        path = (self.root / image_id / name).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None


def make_handler(tracker: Tracker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path in ("/", "/index.html", "/activity_grid.html"):
                self._send(200, _HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return

            if path == "/status.json":
                body = json.dumps(
                    {k: v for k, v in tracker.snapshot().items() if k != "_fp"},
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return

            if path == "/events":
                # Server-Sent Events — push on every fingerprint change.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                last_rev = -1
                try:
                    while not tracker._stop.is_set():
                        status = tracker.wait_change(last_rev, timeout=8.0)
                        payload = {k: v for k, v in status.items() if k != "_fp"}
                        rev = int(payload.get("rev") or 0)
                        # Heartbeat keep-alive even if unchanged.
                        if rev == last_rev:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        last_rev = rev
                        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        self.wfile.write(b"data: " + data + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return
                return

            if path.startswith("/asset/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    asset = tracker.resolve_asset(parts[1], parts[2])
                    if asset is not None:
                        data = asset.read_bytes()
                        ctype = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                        self._send(200, data, ctype)
                        return
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return

            self._send(404, b"not found\n", "text/plain; charset=utf-8")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Live activity grid for pipeline.log")
    ap.add_argument("--root", default=str(_REPO / "runs"),
                    help="benchmark/run root containing <image_id>/pipeline.log")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    root = (_REPO / root).resolve() if not root.is_absolute() else root.resolve()

    if not _HTML_PATH.is_file():
        raise SystemExit(f"missing {_HTML_PATH}")

    tracker = Tracker(root, auto_root=True)
    tracker.refresh()
    tracker.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tracker))
    print(f"activity grid -> http://{args.host}:{args.port}/", flush=True)
    print(f"watching {tracker.root} (from {root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
