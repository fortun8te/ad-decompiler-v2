"""Durable, machine-readable evidence for a decompiler run.

Model workers intentionally have a graceful local fallback so the deterministic pipeline can
still be developed on a laptop.  A production benchmark must *see* that fallback, however.
This small module records that distinction independently of console logs: a run can complete,
be diagnostically useful, and still be marked degraded or unacceptable for the configured
acceptance policy.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any


def _atomic_json(path: str, value: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _required(cfg: dict, component: str) -> bool:
    """Whether a degraded component invalidates this run.

    ``runtime.require_active_models`` is intentionally opt-in.  It lets CPU/unit-test runs
    exercise disabled model paths, while the supplied RTX config makes SAM/OCR health a real
    acceptance condition.  A component can also explicitly opt in with ``required: true``.
    """
    runtime = cfg.get("runtime") or {}
    section = cfg.get(component) or {}
    if isinstance(section, dict) and section.get("required") is True:
        return True
    if not runtime.get("require_active_models", False):
        return False
    if component == "inpaint":
        # Inpaint quality directly determines the clean background plate; a silent
        # Big-LaMa -> OpenCV fallback must be as much of an acceptance condition as
        # OCR/SAM under require_active_models (see CLAUDE_FINAL_TWEAKS.md "leaked
        # foreground pixels in the clean plate" reject condition) -- unless the operator
        # explicitly opted into OpenCV mode, mirroring doctor.py's READY exemption for
        # ``inpaint.mode: opencv`` (see doctor.py's Big-LaMa check).
        inpaint_cfg = cfg.get("inpaint") or {}
        explicit_opencv = isinstance(inpaint_cfg, dict) and str(inpaint_cfg.get("mode", "auto")).lower() == "opencv"
        return not explicit_opencv
    if component == "ocr":
        return True
    # Qwen is an advisory alpha/z observation.  It may be required for a deliberately
    # Qwen-dependent experiment, but it must not make the main SAM/OCR scene graph look
    # unavailable merely because a separate ComfyUI process is offline.
    if component == "qwen":
        return False
    return bool(isinstance(section, dict) and section.get("enabled", False))


class RunReport:
    """Incremental report writer; safe to call even while the run later crashes."""

    def __init__(self, run_dir: str, input_path: str, cfg: dict, start_from: str):
        self.path = os.path.join(run_dir, "runtime_report.json")
        self.cfg = copy.deepcopy(cfg or {})
        self.data: dict[str, Any] = {
            "version": 1,
            "started_at": int(time.time()),
            "input": os.path.abspath(input_path),
            "resume_from": start_from,
            "policy": {
                "require_active_models": bool((self.cfg.get("runtime") or {}).get("require_active_models", False)),
                "required_components": [name for name in ("ocr", "sam3", "qwen", "inpaint") if _required(self.cfg, name)],
            },
            "stages": [],
            "degraded": [],
            "violations": [],
            "retries": [],
            "status": "running",
            "acceptable": False,
        }
        self.write()

    def write(self) -> None:
        self.data["updated_at"] = int(time.time())
        _atomic_json(self.path, self.data)

    def stage(self, name: str, status: str = "ok", *, detail: str | None = None,
              artifacts: list[str] | None = None, duration_s: float | None = None) -> None:
        entry: dict[str, Any] = {"name": name, "status": status}
        if detail:
            entry["detail"] = str(detail)
        if artifacts:
            entry["artifacts"] = list(artifacts)
        if duration_s is not None:
            entry["duration_s"] = round(float(duration_s), 3)
        self.data["stages"].append(entry)
        self.write()

    def retry(self, component: str, reason: str, outcome: str) -> None:
        self.data["retries"].append({
            "component": component, "reason": str(reason), "outcome": str(outcome),
        })
        self.write()

    def degraded(self, component: str, reason: str, *, required: bool | None = None) -> None:
        required = _required(self.cfg, component) if required is None else bool(required)
        item = {"component": component, "reason": str(reason), "required": required}
        if item not in self.data["degraded"]:
            self.data["degraded"].append(item)
        if required:
            violation = {
                "rule": f"{component}-unavailable",
                "detail": f"required {component} did not complete: {reason}",
                "hard": True,
            }
            if violation not in self.data["violations"]:
                self.data["violations"].append(violation)
        self.write()

    @property
    def violations(self) -> list[dict]:
        return list(self.data.get("violations") or [])

    @property
    def acceptable(self) -> bool:
        return not self.data.get("violations") and self.data.get("status") != "failed"

    def finish(self, *, error: str | None = None, qa_ok: bool | None = None) -> None:
        if error:
            self.data["status"] = "failed"
            self.data["error"] = str(error)
        elif self.data.get("degraded"):
            self.data["status"] = "degraded"
        else:
            self.data["status"] = "ok"
        self.data["qa_ok"] = qa_ok
        self.data["acceptable"] = self.acceptable
        self.data["finished_at"] = int(time.time())
        self.write()


def qwen_degradation(run_dir: str, enabled: bool) -> str | None:
    """Read the advisory worker's explicit note without treating empty scenes as failures."""
    if not enabled:
        return None
    note_path = os.path.join(run_dir, "qwen.note.txt")
    if not os.path.exists(note_path):
        return None
    try:
        with open(note_path, encoding="utf-8") as handle:
            return handle.read().strip() or "Qwen produced no layer proposal"
    except OSError as exc:
        return f"could not read qwen note: {exc}"
