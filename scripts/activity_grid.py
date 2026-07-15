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
_POLL_S = 0.08
_ROOT_RECHECK_S = 1.0
_FEED_MAX = 14
# Artifact presence advances the stage cursor when pipeline.log lags (long peel/inpaint).
_ARTIFACT_STAGE: list[tuple[str, tuple[str, ...]]] = [
    ("normalize", ("normalized.png", "original.png")),
    ("ocr", ("ocr.json",)),
    ("residual", ("residual.json",)),
    ("qwen", ("qwen.json",)),
    ("sam", ("sam3.json",)),
    ("elements", ("fused_elements.json",)),
    ("peel", ("peel.json",)),
    ("merge", ("merged.json",)),
    ("reconstruct", ("reconstruction.json", "background_clean.png")),
    ("layout", ("layout.json",)),
    ("design", ("design.json",)),
    ("preview", ("preview.png",)),
    ("figma", ("figma_import.json",)),
    ("export", ("figma_export.png",)),
    ("diff", ("diff.png",)),
    ("qa", ("qa.json",)),
]
_ACTIVITY_DIRS = (
    "peel", "peel_layers", "fused_elements", "elements", "assets",
    "layers", "sam3_masks", "text_fallback",
)
_SMOKE_PROBES = (
    "ocr", "sam3", "vlm", "big_lama", "flux_comfy", "powerpaint",
    "vectorization", "figma_staging",
)
_SKIP_DIRS = {
    "assets", "layers", "elements", "fused_elements", "sam3_masks",
    "text_fallback", "peel", "__pycache__", "runtime-smoke",
}
_SKIP_ROOT_NAMES = {
    ".cache", "runtime-smoke", "_activity_demo",
}
_ASSET_NAMES = ("original.png", "preview.png", "normalized.png", "diff.png")
# Noisy log bodies we never put in the live feed.
_FEED_NOISE = re.compile(
    r"(?i)^(vram\[|vram:|lms unload|lms load|torch\.|cuda |allocated=|evict|restore_vlm)"
)

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
    """Return (done_stages, active, complete, failed).

    A mid-log ``done in Xs`` (harness / resume) is not final — later stage lines
    reopen the run and reset the stage cursor.
    """
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
        # Harness / resume after a prior done — drop the complete seal.
        if complete and (
            body.startswith("config changed")
            or "resuming from" in body
            or body.startswith("archetype ")
        ):
            complete = False
            last_idx = -1
        stage_hit = False
        for name, markers in _MARKERS:
            if any(marker in body for marker in markers):
                idx = STAGES.index(name)
                if complete:
                    # Stage work after done-in without an explicit resume line.
                    complete = False
                    last_idx = idx
                elif idx > last_idx:
                    last_idx = idx
                stage_hit = True
                break
        if stage_hit:
            continue

    if complete:
        return STAGES[:], None, True, failed

    if last_idx < 0:
        if not text.strip() or not any(m in text for _, ms in _MARKERS for m in ms):
            return [], None, False, failed
        return [], "normalize", False, failed

    done = STAGES[: last_idx + 1]
    # qa marker without done-in: still treat as complete (terminal stage).
    if last_idx >= STAGES.index("qa"):
        return STAGES[:], None, True, failed
    nxt = last_idx + 1
    active = STAGES[nxt] if nxt < len(STAGES) else None
    return done, active, False, failed


def artifact_progress(run_dir: Path) -> tuple[int, float]:
    """Return (highest_stage_index_done, newest_activity_mtime).

    Uses on-disk artifacts so long GPU stages stay in sync even when pipeline.log
    is quiet between markers.
    """
    newest = 0.0
    highest = -1
    try:
        for p in run_dir.iterdir():
            if p.is_file():
                try:
                    newest = max(newest, p.stat().st_mtime)
                except OSError:
                    pass
    except OSError:
        return -1, 0.0

    for name in _ACTIVITY_DIRS:
        d = run_dir / name
        if not d.is_dir():
            continue
        try:
            for p in d.iterdir():
                if p.is_file():
                    try:
                        newest = max(newest, p.stat().st_mtime)
                    except OSError:
                        pass
        except OSError:
            pass

    for stage_name, files in _ARTIFACT_STAGE:
        if any((run_dir / f).is_file() for f in files):
            highest = max(highest, STAGES.index(stage_name))

    return highest, newest


def merge_progress(
    log_done: list[str],
    log_active: str | None,
    log_complete: bool,
    art_idx: int,
) -> tuple[list[str], str | None, bool]:
    """Advance done/active to the farther of log markers vs fresh artifacts.

    Never mark complete from artifacts alone (harness resume leaves qa.json around).
    Ignore artifact cursor when it is far ahead of the log — that's a prior round.
    """
    if log_complete:
        return STAGES[:], None, True
    log_idx = STAGES.index(log_done[-1]) if log_done else -1
    if art_idx > log_idx + 2:
        best = log_idx
    else:
        best = max(log_idx, art_idx)
    if best < 0:
        return [], log_active or "normalize", False
    done = STAGES[: best + 1]
    nxt = best + 1
    active = STAGES[nxt] if nxt < len(STAGES) else None
    if log_active and log_idx >= art_idx:
        active = log_active
    return done, active, False


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
        round(run.get("activity_mtime") or 0.0, 3),
        (run.get("last_line") or "")[:80],
    )


def _strip_ts(line: str) -> tuple[str | None, str]:
    """Return (HH:MM:SS | None, body)."""
    m = _LINE_RE.match(line.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return None, line.strip()


def clean_line(body: str) -> str:
    """Human-readable one-liner for the live feed (no VRAM soup, no mojibake arrows)."""
    if not body:
        return ""
    s = body.replace("\u2192", "->").replace("→", "->").replace("\ufffd", "")
    s = re.sub(r"\s+", " ", s).strip()
    # Drop trailing path spam: keep the verb + short count.
    if _FEED_NOISE.search(s):
        return ""
    # Compress absolute Windows paths to basename.
    s = re.sub(r"[A-Za-z]:\\[^\s]+\\([^\\\s]+)", r"\1", s)
    s = re.sub(r"/(?:[^/\s]+/)+([^/\s]+)", r"\1", s)
    if len(s) > 96:
        s = s[:93] + "..."
    return s


def _last_log_line(text: str) -> str:
    for raw in reversed(text.splitlines()):
        _ts, body = _strip_ts(raw)
        cleaned = clean_line(body)
        if cleaned:
            return cleaned
    # Fallback: last non-empty even if noisy.
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        _ts, body = _strip_ts(line)
        return clean_line(body) or body[:96]
    return ""


def extract_feed(text: str, *, image_id: str = "", limit: int = _FEED_MAX) -> list[dict[str, str]]:
    """Recent meaningful stage / milestone lines from a pipeline.log."""
    events: list[dict[str, str]] = []
    for raw in text.splitlines():
        ts, body = _strip_ts(raw)
        if not body:
            continue
        if "ERROR:" in body:
            events.append({"t": ts or "", "kind": "error", "text": clean_line(body) or body[:96],
                           "ad": image_id})
            continue
        if _FEED_NOISE.search(body):
            continue
        is_stage = any(any(m in body for m in markers) for _, markers in _MARKERS)
        is_done = bool(_DONE_RE.search(body))
        if not (is_stage or is_done or "peel" in body.lower() or "harness" in body.lower()):
            continue
        cleaned = clean_line(body)
        if not cleaned:
            continue
        kind = "done" if is_done else ("stage" if is_stage else "note")
        events.append({"t": ts or "", "kind": kind, "text": cleaned, "ad": image_id})
    return events[-limit:]


def read_smoke(root: Path) -> dict[str, Any] | None:
    """Surface runtime-smoke progress so the UI isn't blank before ads start."""
    smoke_dir = root / "runtime-smoke"
    if not smoke_dir.is_dir():
        return None
    report_path = smoke_dir / "runtime_smoke.json"
    now = time.time()
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        checks = report.get("checks") or []
        done = [c.get("name") for c in checks if isinstance(c, dict) and c.get("name")]
        failed = [c.get("name") for c in checks
                  if isinstance(c, dict) and c.get("name") and not c.get("ok")]
        try:
            mtime = report_path.stat().st_mtime
        except OSError:
            mtime = now
        return {
            "phase": "smoke",
            "status": "done" if report.get("ok") else ("failed" if failed else "done"),
            "active": None,
            "done": done,
            "failed": failed,
            "probes": list(report.get("probes") or done),
            "ok": bool(report.get("ok")),
            "mtime": mtime,
            "stalled": False,
            "last_line": (
                f"smoke {'ok' if report.get('ok') else 'failed'}: "
                + ", ".join(done or ["(empty)"])
            ),
        }

    # In-flight: probe subdirs appear as each probe starts.
    present: list[tuple[str, float]] = []
    for name in _SMOKE_PROBES:
        p = smoke_dir / name
        if not p.is_dir():
            continue
        try:
            newest = max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=p.stat().st_mtime)
        except OSError:
            newest = now
        present.append((name, newest))

    try:
        empty = not any(smoke_dir.iterdir())
    except OSError:
        empty = True
    if not present and empty:
        # Empty smoke folder just created.
        try:
            mtime = smoke_dir.stat().st_mtime
        except OSError:
            mtime = now
        return {
            "phase": "smoke",
            "status": "running",
            "active": "starting",
            "done": [],
            "failed": [],
            "probes": list(_SMOKE_PROBES),
            "ok": None,
            "mtime": mtime,
            "stalled": (now - mtime) > _STALL_S,
            "last_line": "runtime smoke starting…",
        }

    if not present:
        try:
            mtime = smoke_dir.stat().st_mtime
        except OSError:
            mtime = now
        return {
            "phase": "smoke",
            "status": "running",
            "active": "starting",
            "done": [],
            "failed": [],
            "probes": list(_SMOKE_PROBES),
            "ok": None,
            "mtime": mtime,
            "stalled": (now - mtime) > _STALL_S,
            "last_line": "runtime smoke starting…",
        }

    present.sort(key=lambda x: x[1])
    # Last dir with recent writes is active; older ones are done.
    active_name, active_mtime = present[-1]
    stalled = (now - active_mtime) > _STALL_S
    done_names = [n for n, _ in present if n != active_name]
    return {
        "phase": "smoke",
        "status": "stalled" if stalled else "running",
        "active": active_name,
        "done": done_names,
        "failed": [],
        "probes": list(_SMOKE_PROBES),
        "ok": None,
        "mtime": active_mtime,
        "stalled": stalled,
        "last_line": f"smoke · {active_name}" + (" (stalled)" if stalled else ""),
    }


def smoke_feed(smoke: dict[str, Any] | None) -> list[dict[str, str]]:
    if not smoke:
        return []
    events: list[dict[str, str]] = []
    for name in smoke.get("done") or []:
        if name in (smoke.get("failed") or []):
            continue
        events.append({"t": "", "kind": "stage", "text": f"{name} ok", "ad": "smoke"})
    active = smoke.get("active")
    if active:
        suffix = " stalled" if smoke.get("stalled") else "…"
        events.append({"t": "", "kind": "stage", "text": f"{active}{suffix}", "ad": "smoke"})
    for name in smoke.get("failed") or []:
        events.append({"t": "", "kind": "error", "text": f"{name} failed", "ad": "smoke"})
    if smoke.get("status") == "done" and smoke.get("ok"):
        events.append({"t": "", "kind": "done", "text": "smoke passed — starting ads", "ad": "smoke"})
    return events[-_FEED_MAX:]


def load_planned(root: Path) -> list[str]:
    """Return planned image stems from planned.json (written by benchmark.py pre-smoke)."""
    path = root / "planned.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list):
        return []
    out: list[str] = []
    for item in images:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _pending_stub(image_id: str, run_dir: Path) -> dict[str, Any]:
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
        "has_original": (run_dir / "original.png").is_file() if run_dir.is_dir() else False,
        "has_preview": False,
        "last_line": "",
        "feed": [],
    }


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
            "phase": "waiting",
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
            "smoke": None,
            "feed": [],
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

    def _merge_planned(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure every planned ad appears as a pending slot before it starts."""
        planned = load_planned(self.root)
        if not planned:
            return runs
        by_id = {r["id"]: r for r in runs}
        merged: list[dict[str, Any]] = []
        for image_id in planned:
            if image_id in by_id:
                merged.append(by_id.pop(image_id))
            else:
                merged.append(_pending_stub(image_id, self.root / image_id))
        # Any unexpected live runs (not in planned) append at end.
        for image_id, run in sorted(by_id.items()):
            merged.append(run)
        return merged

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
                "feed": [],
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

        art_idx, activity_mtime = artifact_progress(run_dir)
        activity_mtime = max(activity_mtime, mtime)
        done, active, complete = merge_progress(done, active, complete, art_idx)

        has_original = (run_dir / "original.png").is_file()
        has_preview = (run_dir / "preview.png").is_file()

        # Start clock on first real stage line (not stub "waiting").
        if image_id not in self._started and (done or active or complete):
            self._started[image_id] = now if truncated else min(mtime, now)

        started_at = self._started.get(image_id)
        elapsed = max(0.0, now - started_at) if started_at else 0.0
        # Stall off artifact activity, not just pipeline.log — peel/inpaint are quiet.
        stalled = (
            not complete
            and not failed
            and active is not None
            and (now - activity_mtime) > _STALL_S
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
            "activity_mtime": activity_mtime,
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
            "activity_mtime": activity_mtime,
            "has_original": has_original,
            "has_preview": has_preview,
            "last_line": _last_log_line(text),
            "feed": extract_feed(text, image_id=image_id),
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
        runs = self._merge_planned(runs)

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

        smoke = read_smoke(self.root)
        feed: list[dict[str, str]] = []
        if live:
            feed.extend(live.get("feed") or [])
        # Also pull a bit of history from recently finished ads.
        for r in sorted(runs, key=lambda x: x["mtime"], reverse=True)[:3]:
            if live and r["id"] == live["id"]:
                continue
            feed.extend((r.get("feed") or [])[-4:])
        if not runs:
            feed = smoke_feed(smoke)
        else:
            # Keep smoke trail at the top of an otherwise-empty early feed.
            if smoke and smoke.get("status") != "done":
                feed = smoke_feed(smoke) + feed
        feed = feed[-_FEED_MAX:]

        phase = "ads"
        live_payload: dict[str, Any] | None = None
        if live:
            live_payload = {
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
                "feed": live.get("feed") or [],
            }
        elif smoke and smoke.get("status") in ("running", "stalled", "failed"):
            phase = "smoke"
            probes = smoke.get("probes") or list(_SMOKE_PROBES)
            n_done = len(smoke.get("done") or [])
            pct = round(100.0 * n_done / max(1, len(probes)), 1)
            live_payload = {
                "image_id": "runtime-smoke",
                "active": smoke.get("active") or "smoke",
                "done": list(smoke.get("done") or []),
                "percent": pct,
                "elapsed_s": round(max(0.0, now - float(smoke.get("mtime") or now)), 1),
                "started_at": None,
                "stalled": bool(smoke.get("stalled")),
                "complete": False,
                "failed": smoke.get("status") == "failed",
                "status": smoke.get("status") or "running",
                "has_original": False,
                "has_preview": False,
                "run_dir": str(self.root / "runtime-smoke"),
                "mtime": smoke.get("mtime") or now,
                "last_line": smoke.get("last_line") or "",
                "feed": smoke_feed(smoke),
            }
            if live_payload["image_id"] != self._live_id:
                self._live_id = live_payload["image_id"]
                self.generation += 1
        elif smoke and smoke.get("status") == "done" and not runs:
            phase = "smoke"
        else:
            phase = "waiting"

        status = {
            "stages": STAGES,
            "updated_at": now,
            "generation": self.generation,
            "rev": self.rev,
            "phase": phase,
            "benchmark": {
                "root": str(self.root),
                "name": self.root.name,
                "total": len(runs),
                "done_count": done_count,
                "running_count": running_count,
                "percent": bench_pct,
                "runs": runs,
            },
            "live": live_payload,
            "smoke": smoke,
            "feed": feed,
        }

        fp = (
            self.generation,
            done_count,
            running_count,
            bench_pct,
            phase,
            None if not live_payload else (
                live_payload["image_id"], live_payload.get("active"),
                live_payload.get("percent"), live_payload.get("status"),
                tuple(live_payload.get("done") or ()),
                live_payload.get("stalled"), live_payload.get("complete"),
                (live_payload.get("last_line") or "")[:80],
            ),
            tuple(
                (e.get("kind"), e.get("text"), e.get("ad"))
                for e in feed[-8:]
            ),
            None if not smoke else (
                smoke.get("status"), smoke.get("active"),
                tuple(smoke.get("done") or ()), smoke.get("stalled"),
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
