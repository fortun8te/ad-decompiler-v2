"""Folded agent debug logging for debug sessions (NDJSON)."""
from __future__ import annotations

import json
import os
import time
from typing import Any

SESSION_ID = "0bad44"
DEFAULT_LOG = os.environ.get(
    "AD_AGENT_DEBUG_LOG",
    os.path.expanduser("~/.cursor/debug-0bad44.log"),
)


def _targets(run_dir: str | None) -> list[str]:
    paths: list[str] = []
    if run_dir:
        paths.append(os.path.join(run_dir, f"debug-{SESSION_ID}.jsonl"))
    if DEFAULT_LOG:
        paths.append(DEFAULT_LOG)
    return paths


def log(
    location: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
    hypothesis_id: str = "",
    run_dir: str | None = None,
) -> None:
    entry = {
        "sessionId": SESSION_ID,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "hypothesisId": hypothesis_id,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    for path in _targets(run_dir):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass


def tail(run_dir: str | None, limit: int = 30) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    path = os.path.join(run_dir, f"debug-{SESSION_ID}.jsonl")
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError):
            continue
    return out
