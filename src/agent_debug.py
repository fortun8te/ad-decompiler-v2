"""Folded agent debug logging for debug sessions (NDJSON)."""
from __future__ import annotations

import json
import os
import time
from typing import Any


def session_id(*, cfg: dict[str, Any] | None = None, run_dir: str | None = None) -> str | None:
    env_sid = os.environ.get("AD_DEBUG_SESSION", "").strip()
    if env_sid:
        return env_sid
    if cfg:
        sid = str((cfg.get("runtime") or {}).get("debug_session") or "").strip()
        if sid:
            return sid
    return None


def enabled(*, cfg: dict[str, Any] | None = None, run_dir: str | None = None) -> bool:
    return bool(session_id(cfg=cfg, run_dir=run_dir))


def _default_log_path(sid: str) -> str:
    override = os.environ.get("AD_AGENT_DEBUG_LOG", "").strip()
    if override:
        return override
    return os.path.expanduser(f"~/.cursor/debug-{sid}.log")


def _targets(sid: str, run_dir: str | None) -> list[str]:
    paths: list[str] = []
    if run_dir:
        paths.append(os.path.join(run_dir, f"debug-{sid}.jsonl"))
    paths.append(_default_log_path(sid))
    return paths


def log(
    location: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
    hypothesis_id: str = "",
    run_dir: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> None:
    sid = session_id(cfg=cfg, run_dir=run_dir)
    if not sid:
        return
    entry = {
        "sessionId": sid,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "hypothesisId": hypothesis_id,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    for path in _targets(sid, run_dir):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass


def tail(
    run_dir: str | None,
    limit: int = 30,
    *,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sid = session_id(cfg=cfg, run_dir=run_dir)
    if not sid or not run_dir:
        return []
    path = os.path.join(run_dir, f"debug-{sid}.jsonl")
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
