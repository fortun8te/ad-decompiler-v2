"""figma_bridge.py — tiny local HTTP bridge so the Figma plugin can close the loop headlessly.

Serves the staged inbox (design.json + assets) to the plugin and receives the exported PNG
back into the run dir. Zero deps at rest (http.server) — start it before clicking "Import":

    python -m src.figma_bridge --inbox ~/figma-inbox --port 8790 --config config.yaml

Endpoints:
    GET  /inbox.json           -> manifest (design path, assets dir, export_to)
    GET  /design.json          -> the staged design.json
    GET  /asset?path=<rel>     -> an asset PNG (resolved under inbox/assets or run dir)
    POST /export               -> body = PNG bytes; written to manifest.export_to
    POST /report               -> compiler report JSON (figma_report.json)
    POST /log                  -> append plugin UI/compiler events to plugin.log
    POST /process?filename=x   -> body = raw image bytes (png/jpg/webp/whatever PIL reads);
                                   runs the full pipeline on it in a background thread and
                                   re-stages /inbox.json + /design.json when done. Returns
                                   {job_id, status:"queued"} immediately — poll with:
    GET  /process?job_id=x     -> {status:"running"|"done"|"failed", ...}
    GET  /run-summary?job_id=x -> manifest + staging fields for a finished job (no inbox race)
                                   Requires the pipeline's own deps (torch, paddleocr, sam3,
                                   ...) to be importable in this interpreter; the /process
                                   routes lazy-import run_pipeline so the bridge itself still
                                   starts with zero deps when those aren't installed — every
                                   other endpoint keeps working either way.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, sys, tempfile, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from src.console_io import configure_stdio, safe_print
from src.agent_debug import log as _agent_log, session_id as _debug_session_id, tail as _agent_debug_tail
from src.error_messages import (
    STAGE_ORDER,
    classify_processing_error,
    detect_failed_stage,
    tail_running_stage,
)


def _safe_name(name, fallback="upload"):
    """Basename-only, alnum/dot/dash/underscore, non-empty — never a path component."""
    base = os.path.basename(str(name or "").strip().replace("\\", "/"))
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", base).strip(".-") or fallback
    return cleaned[:120]


def _load_cfg(path):
    """Mirrors run_pipeline.load_cfg without importing run_pipeline (and its heavy deps)
    just to read a config file — this keeps the bridge itself zero-dep at rest."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)


def _history_path(inbox):
    return os.path.join(inbox, ".process_history.json")


def _tail_stage(run_dir):
    """Best-effort read of pipeline.log's last matched stage — purely informational."""
    return tail_running_stage(run_dir)


_HISTORY_MAX = 10
_STAGE_PROGRESS_IN_STAGE = 0.35  # assume ~35% through the active stage for ETA/progress


def _empty_history():
    return {"durations_s": [], "runs": []}


def _read_history(inbox):
    try:
        with open(_history_path(inbox), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return _empty_history()
    runs = data.get("runs") or []
    durations = [
        float(d) for d in (data.get("durations_s") or [])
        if isinstance(d, (int, float))
    ]
    if not runs and durations:
        runs = [{"duration_s": d} for d in durations]
    cleaned = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        duration = entry.get("duration_s")
        if not isinstance(duration, (int, float)):
            continue
        row = {"duration_s": float(duration)}
        fracs = entry.get("stage_fractions")
        if isinstance(fracs, dict):
            row["stage_fractions"] = {
                str(k): float(v) for k, v in fracs.items()
                if k in STAGE_ORDER and isinstance(v, (int, float)) and v >= 0
            }
        cleaned.append(row)
    cleaned = cleaned[-_HISTORY_MAX:]
    return {
        "durations_s": [row["duration_s"] for row in cleaned],
        "runs": cleaned,
    }


def _parse_stage_fractions(run_dir, total_s):
    """Derive per-stage time fractions from pipeline.log timestamps."""
    if not run_dir or not total_s or total_s <= 0:
        return None
    log_path = os.path.join(run_dir, "pipeline.log")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    stamps = []
    for line in lines:
        m = re.search(r"\[(\d{2}):(\d{2}):(\d{2})\]", line)
        if not m:
            continue
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        stage = None
        for name, marker in (
            ("normalize", "normalize →"),
            ("ocr", "ocr["),
            ("text", "text analysis →"),
            ("residual", "residual proposals →"),
            ("qwen", "qwen →"),
            ("sam", "sam3["),
            ("elements", "element fusion →"),
            ("merge", "merge →"),
            ("reconstruct", "reconstruct →"),
            ("layout", "layout →"),
            ("design", "design.json →"),
            ("preview", "preview →"),
            ("figma", "figma import:"),
            ("export", "export:"),
            ("qa", "qa →"),
        ):
            if marker in line:
                stage = name
                break
        if stage:
            stamps.append((secs, stage))
    if len(stamps) < 2:
        return None
    durations = {}
    for i in range(len(stamps) - 1):
        delta = stamps[i + 1][0] - stamps[i][0]
        if delta < 0:
            delta += 24 * 3600
        if delta > 0:
            durations[stamps[i][1]] = durations.get(stamps[i][1], 0.0) + delta
    tail = stamps[-1][1]
    tail_delta = max(0.0, float(total_s) - sum(durations.values()))
    if tail_delta > 0:
        durations[tail] = durations.get(tail, 0.0) + tail_delta
    accounted = sum(durations.values()) or float(total_s)
    return {stage: round(max(0.0, secs) / accounted, 4) for stage, secs in durations.items()}


def _median_stage_weights(history):
    runs = [row for row in history.get("runs") or [] if row.get("stage_fractions")]
    if not runs:
        return None
    weights = {stage: 0.0 for stage in STAGE_ORDER}
    for row in runs:
        fracs = row.get("stage_fractions") or {}
        for stage in STAGE_ORDER:
            weights[stage] += float(fracs.get(stage, 0.0))
    n = len(runs)
    return {stage: weights[stage] / n for stage in STAGE_ORDER}


def _stage_progress_fraction(stage, weights=None):
    weights = weights or {s: 1.0 / max(1, len(STAGE_ORDER)) for s in STAGE_ORDER}
    if stage not in STAGE_ORDER:
        return 0.0
    idx = STAGE_ORDER.index(stage)
    completed = sum(weights.get(STAGE_ORDER[i], 0.0) for i in range(idx))
    current = weights.get(stage, 0.0) * _STAGE_PROGRESS_IN_STAGE
    return min(0.98, max(0.0, completed + current))


def _estimate_eta(inbox, elapsed_s, stage=None):
    history = _read_history(inbox)
    durations = history.get("durations_s") or []
    if not durations:
        return None, 0, None
    median_total = statistics.median(durations)
    weights = _median_stage_weights(history)
    if stage and stage in STAGE_ORDER:
        progress = _stage_progress_fraction(stage, weights)
    else:
        progress = min(0.95, max(0.05, elapsed_s / median_total)) if median_total > 0 else 0.05
    progress = max(0.05, min(0.98, progress))
    remaining = max(0.0, median_total * (1.0 - progress) - elapsed_s)
    cap = max(30.0, median_total * 1.25)
    eta_s = min(remaining, cap)
    progress_pct = round(progress * 100)
    return round(eta_s, 1), len(durations), progress_pct


def _record_history(inbox, duration_s, run_dir=None):
    history = _read_history(inbox)
    entry = {"duration_s": float(duration_s)}
    fracs = _parse_stage_fractions(run_dir, duration_s)
    if fracs:
        entry["stage_fractions"] = fracs
    runs = (history.get("runs") or []) + [entry]
    runs = runs[-_HISTORY_MAX:]
    payload = {
        "durations_s": [row["duration_s"] for row in runs],
        "runs": runs,
    }
    try:
        _atomic_write(_history_path(inbox), json.dumps(payload).encode())
    except OSError:
        pass


def _atomic_write(path, data: bytes):
    """Atomic replace with retries on Windows transient file locks.

    os.replace() can raise PermissionError (WinError 5) on Windows when the
    destination is momentarily open by another thread/process (e.g. a
    background job thread racing the request thread in _append_plugin_logs).
    Retry a few times with a short backoff before giving up.
    """
    attempts = 3
    for attempt in range(attempts):
        fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(temp_path, path)
            return
        except PermissionError:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            if attempt < attempts - 1:
                time.sleep(0.05)
                continue
            raise
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


_plugin_log_lock = threading.Lock()


def _atomic_append_text(path, text: str):
    """Append UTF-8 text via read-modify-write atomic replace with retry."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    block = text.encode("utf-8")
    for attempt in range(2):
        try:
            existing = b""
            try:
                with open(path, "rb") as fh:
                    existing = fh.read()
            except OSError:
                pass
            _atomic_write(path, existing + block)
            return
        except PermissionError:
            if attempt == 0:
                time.sleep(0.05)
                continue
            raise


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_repo_update(remote: str = "newrepo", branch: str = "main") -> dict:
    """git pull on the bridge host so Mac can sync the RTX box over Tailscale."""
    import subprocess

    root = _repo_root()
    git_dir = os.path.join(root, ".git")
    if not os.path.isdir(git_dir):
        return {"ok": False, "error": "not a git checkout", "root": root}

    def run(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(args),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    try:
        fetch = run("git", "fetch", remote, branch)
        if fetch.returncode != 0:
            return {
                "ok": False,
                "error": (fetch.stderr or fetch.stdout or "git fetch failed").strip(),
                "root": root,
            }
        pull = run("git", "pull", remote, branch)
        if pull.returncode != 0:
            return {
                "ok": False,
                "error": (pull.stderr or pull.stdout or "git pull failed").strip(),
                "root": root,
            }
        head = run("git", "rev-parse", "--short", "HEAD")
        commit = head.stdout.strip() if head.returncode == 0 else None
        python = sys.executable
        stamp_note = None
        stamp_script = os.path.join(root, "scripts", "stamp_plugin_build.py")
        if os.path.isfile(stamp_script):
            stamp = run(python, stamp_script, "--quiet", timeout=60)
            if stamp.returncode != 0:
                stamp_note = (stamp.stderr or stamp.stdout or "stamp failed").strip()
        return {
            "ok": True,
            "root": root,
            "remote": remote,
            "branch": branch,
            "commit": commit,
            "pull_output": (pull.stdout or "").strip(),
            "stamp_note": stamp_note,
            "restart_required": True,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "git command timed out", "root": root}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "root": root}

def _resolve_config_path(path: str) -> str:
    if not path:
        return os.path.join(_repo_root(), "config.yaml")
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _ocr_preflight(cfg, repo_root):
    """Doctor OCR readiness without importing run_pipeline."""
    from pathlib import Path
    from doctor import inspect as doctor_inspect
    doctor = doctor_inspect(cfg or {}, Path(repo_root))
    ocr_blockers = [
        item for item in doctor.get("blockers", [])
        if str(item.get("name", "")).startswith("ocr")
    ]
    fallback_ready = bool((doctor.get("ocr_fallback") or {}).get("ready"))
    return doctor, ocr_blockers, fallback_ready


def _append_plugin_logs(inbox, events, manifest=None):
    if not events:
        return
    paths = [os.path.join(inbox, "plugin.log")]
    run_dir = (manifest or {}).get("run_dir")
    if run_dir:
        paths.append(os.path.join(run_dir, "plugin.log"))
    lines = []
    for event in events:
        at = event.get("at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        level = str(event.get("level") or "info")
        title = str(event.get("title") or "").strip()
        detail = str(event.get("detail") or "").strip()
        line = f"{at} [{level}] {title}"
        if detail:
            line += f" — {detail}"
        lines.append(line + "\n")
    text = "".join(lines)
    with _plugin_log_lock:
        for path in paths:
            _atomic_append_text(path, text)
        json_path = os.path.join(inbox, "plugin_events.json")
        existing = []
        try:
            with open(json_path, encoding="utf-8") as fh:
                existing = json.load(fh)
            if not isinstance(existing, list):
                existing = []
        except (OSError, ValueError, TypeError):
            existing = []
        existing.extend(events)
        existing = existing[-5000:]
        payload = json.dumps(existing, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write(json_path, payload)
        if run_dir:
            _atomic_write(os.path.join(run_dir, "plugin_events.json"), payload)


def _read_json_file(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


def _read_bridge_build():
    path = os.path.join(_repo_root(), "figma-plugin", "build-info.json")
    data = _read_json_file(path)
    return data if isinstance(data, dict) else None


def _read_plugin_client(inbox):
    data = _read_json_file(os.path.join(inbox, "plugin_client.json"))
    return data if isinstance(data, dict) else None


def _health_payload(inbox, base_cfg):
    payload = {
        "ok": True,
        "service": "ad-decompiler-bridge",
        "has_run": bool(_read_json_file(os.path.join(inbox, "inbox.json"))),
        "supports_process": True,
        "bridge_build": _read_bridge_build(),
        "plugin_client": _read_plugin_client(inbox),
    }
    manifest = _read_json_file(os.path.join(inbox, "inbox.json"))
    if manifest:
        payload["schema_version"] = manifest.get("schema_version")
    debug_sid = _debug_session_id(cfg=base_cfg)
    if debug_sid:
        payload["debug_session"] = debug_sid
    try:
        # doctor._http's short timeout keeps the ComfyUI liveness probe from stalling
        # /health, so this stays cheap enough to compute per request.
        from doctor import inspect as _doctor_inspect, ocr_ready_summary as _ocr_ready_summary
        from pathlib import Path
        repo_root = Path(_repo_root())
        cfg = base_cfg or {}
        doctor = _doctor_inspect(cfg, repo_root)
        payload["machine_ready"] = doctor.get("ok")
        payload["machine_blockers"] = doctor.get("blockers") or []
        # Reuse the already-computed inspect() report instead of letting
        # ocr_ready_summary re-run inspect() (which re-probes ComfyUI/VLM and
        # nearly doubled /health latency under a cold, firewall-dropped port).
        # Fall back to the two-arg call if a monkeypatched ocr_ready_summary
        # (e.g. in tests) doesn't accept the report= kwarg.
        try:
            payload["ocr_ready"] = _ocr_ready_summary(cfg, repo_root, report=doctor)
        except TypeError:
            payload["ocr_ready"] = _ocr_ready_summary(cfg, repo_root)
    except Exception as exc:
        payload["machine_ready"] = False
        payload["machine_blockers"] = [{"name": "doctor", "detail": str(exc)}]
        payload["ocr_ready"] = {"ok": False, "error": str(exc)}
    return payload


def _preflight_blockers(base_cfg):
    """Return doctor blockers for early pipeline rejection, or None if ready.

    OCR module blockers are waived when a configured fallback engine is runnable
    (e.g. tesseract on PATH) so a cuDNN-broken Paddle GPU install can still process.
    """
    try:
        doctor, ocr_blockers, fallback_ready = _ocr_preflight(base_cfg, _repo_root())
        blockers = list(doctor.get("blockers") or [])
        if ocr_blockers and fallback_ready:
            ocr_names = {item.get("name") for item in ocr_blockers}
            blockers = [item for item in blockers if item.get("name") not in ocr_names]
        return blockers if blockers else None
    except Exception as exc:
        return [{"name": "doctor", "detail": str(exc)}]


def _format_blockers(blockers):
    return "; ".join(f"{item.get('name', 'check')}: {item.get('detail', '')}" for item in blockers[:4])


def _save_plugin_client(inbox, build_info):
    if not isinstance(build_info, dict):
        return
    payload = dict(build_info)
    payload["seen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write(
        os.path.join(inbox, "plugin_client.json"),
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _bridge_inbox_path(inbox):
    return os.path.abspath(os.path.expanduser(inbox))


def _upload_cfg(base_cfg, inbox):
    """Force upload jobs to stage into this bridge's inbox, not config.yaml's path."""
    import copy
    bridge_inbox = _bridge_inbox_path(inbox)
    cfg = copy.deepcopy(base_cfg or {})
    cfg["figma"] = {**cfg.get("figma", {}), "enabled": True, "mode": "plugin", "inbox": bridge_inbox}
    return cfg


def _load_harness_loop_summary(run_dir):
    """Read harness_loop.json written by src.harness_loop."""
    return _read_json_file(os.path.join(run_dir, "harness_loop.json"))


def _run_one_already_ran_harness(result, run_dir):
    """True when run_pipeline.run_one already executed the full harness loop."""
    result = result or {}
    if result.get("repair") is not None or result.get("harness") is not None:
        return True
    if result.get("harness_rounds") is not None:
        return True
    return os.path.exists(os.path.join(run_dir, "harness_loop.json"))


def _harness_summary_fields(harness_summary=None, *, run_dir=None, pipeline_result=None):
    """Normalize harness_loop.json / run_harness_after_pipeline output for poll responses."""
    summary = harness_summary
    if summary is None and run_dir:
        summary = _load_harness_loop_summary(run_dir)
    summary = summary or {}
    pipeline_result = pipeline_result or {}
    if "rounds_completed" in summary or "stopped" in summary:
        qa_ok = summary.get("qa_ok")
        if qa_ok is None:
            qa_ok = pipeline_result.get("qa_ok")
        return {
            "harness_rounds": summary.get("rounds_completed", 0),
            "harness_stopped": summary.get("stopped"),
            "final_qa_ok": qa_ok,
        }
    return {
        "harness_rounds": summary.get("harness_rounds", 0),
        "harness_stopped": summary.get("harness_stopped") or summary.get("reason", "ok"),
        "final_qa_ok": summary.get("qa_ok", pipeline_result.get("qa_ok")),
    }


def _stage_job_output(inbox, run_dir, cfg):
    """Always re-stage design.json into the bridge inbox before reporting job done."""
    design_path = os.path.join(run_dir, "design.json")
    staging_error = None
    if not os.path.exists(design_path):
        return {"staged": False, "doc_id": None, "layer_count": None, "staging_error": None,
                "design_url": "/design.json"}
    try:
        from src import figma_import
        figma_import.import_design(design_path, run_dir, cfg)
        # #region agent log
        _agent_log(
            "figma_bridge.py:_stage_job_output", "figma_import after pipeline",
            data={"inbox": _bridge_inbox_path(inbox), "run_dir": run_dir, "design_path": design_path},
            hypothesis_id="H1", run_dir=run_dir, cfg=cfg,
        )
        # #endregion
    except Exception as exc:
        staging_error = str(exc)
        # #region agent log
        _agent_log(
            "figma_bridge.py:_stage_job_output", "figma_import failed",
            data={"error": staging_error},
            hypothesis_id="H1", run_dir=run_dir, cfg=cfg,
        )
        # #endregion
    manifest = _read_json_file(os.path.join(inbox, "inbox.json"))
    staged = bool(manifest)
    if not staged and staging_error is None:
        staging_error = "design.json exists but inbox.json was not written"
    return {
        "staged": staged,
        "doc_id": (manifest or {}).get("doc_id"),
        "layer_count": ((manifest or {}).get("summary") or {}).get("layers"),
        "staging_error": staging_error,
        "design_url": "/design.json",
        "manifest": manifest,
    }


def make_handler(inbox, config_path=None):
    jobs = {}
    jobs_lock = threading.RLock()
    base_cfg = _load_cfg(config_path)
    active_job = {"id": None}

    def _job_cancelled(job_id):
        with jobs_lock:
            return jobs.get(job_id, {}).get("status") == "cancelled"

    def _finish_job_thread(job_id):
        with jobs_lock:
            if active_job["id"] == job_id:
                active_job["id"] = None
            job = jobs.get(job_id)
            if job:
                event = job.get("thread_done")
                if event is not None:
                    event.set()

    def run_job(job_id, image_path, run_dir):
        try:
            started = time.time()
            manifest = None
            try:
                with open(os.path.join(inbox, "inbox.json"), encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, ValueError, TypeError):
                pass
            if _job_cancelled(job_id):
                return
            _append_plugin_logs(inbox, [{
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": "info",
                "title": "Pipeline started",
                "detail": f"{job_id} — {os.path.basename(image_path)}",
                "extra": {"job_id": job_id, "run_dir": run_dir},
            }], manifest)
            with jobs_lock:
                if _job_cancelled(job_id):
                    return
                jobs[job_id]["status"] = "running"
                jobs[job_id]["started_at"] = started
                jobs[job_id]["run_dir"] = run_dir
            try:
                repo_root = _repo_root()
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                # #region agent log
                try:
                    doctor, ocr_blockers, fallback_ready = _ocr_preflight(base_cfg, repo_root)
                    _agent_log(
                        "figma_bridge.py:run_job", "doctor preflight",
                        data={
                            "ok": doctor.get("ok"),
                            "ocr_blockers": ocr_blockers,
                            "blockers": doctor.get("blockers"),
                            "ocr_fallback_ready": fallback_ready,
                        },
                        hypothesis_id="H4", run_dir=run_dir, cfg=base_cfg,
                    )
                    blockers = list(doctor.get("blockers") or [])
                    if ocr_blockers and fallback_ready:
                        ocr_names = {item.get("name") for item in ocr_blockers}
                        blockers = [item for item in blockers if item.get("name") not in ocr_names]
                    if blockers:
                        raise RuntimeError(
                            "Machine not ready — run doctor.py on the bridge host. "
                            + _format_blockers(blockers)
                        )
                except RuntimeError:
                    raise
                except Exception as doctor_error:
                    _agent_log(
                        "figma_bridge.py:run_job", "doctor preflight failed",
                        data={"error": str(doctor_error)},
                        hypothesis_id="H4", run_dir=run_dir, cfg=base_cfg,
                    )
                # #endregion
                import run_pipeline  # heavy: torch/paddleocr/sam3/... — only imported on first use
                from src.harness import harness_should_repair, load_qa
                from src.harness_loop import max_harness_rounds, run_harness_after_pipeline
                cfg = _upload_cfg(base_cfg, inbox)
                result = run_pipeline.run_one(image_path, run_dir, cfg)
                with jobs_lock:
                    if _job_cancelled(job_id):
                        return
                staging = None
                harness_summary = None
                if result.get("ok"):
                    staging = _stage_job_output(inbox, run_dir, cfg)
                    qa_path = os.path.join(run_dir, "qa.json")
                    qa = load_qa(run_dir) if os.path.exists(qa_path) else None
                    should_repair, repair_reason = harness_should_repair(
                        result, qa=qa, staging=staging,
                    )
                    already_ran_harness = _run_one_already_ran_harness(result, run_dir)
                    skip_harness = already_ran_harness and repair_reason != "staging_failed"
                    if should_repair and not skip_harness:
                        with jobs_lock:
                            if _job_cancelled(job_id):
                                return
                            jobs[job_id]["harness_running"] = True
                        _append_plugin_logs(inbox, [{
                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "level": "info",
                            "title": "Auto-fix started",
                            "detail": job_id,
                            "extra": {
                                "job_id": job_id,
                                "run_dir": run_dir,
                                "reason": repair_reason,
                            },
                        }], manifest)
                        try:
                            harness_summary = run_harness_after_pipeline(
                                image_path,
                                run_dir,
                                cfg,
                                max_rounds=max_harness_rounds(cfg),
                                run_one=run_pipeline.run_one,
                            )
                            result["repair"] = harness_summary
                            result["qa_ok"] = harness_summary.get("qa_ok")
                            result["harness_rounds"] = harness_summary.get("rounds_completed")
                            result["harness_stopped"] = harness_summary.get("stopped")
                            if harness_summary.get("qa_ok") or repair_reason == "staging_failed":
                                staging = _stage_job_output(inbox, run_dir, cfg)
                            _agent_log(
                                "figma_bridge.py:run_job", "harness repair loop",
                                data={
                                    "reason": repair_reason,
                                    "stopped": harness_summary.get("stopped"),
                                    "rounds": harness_summary.get("rounds_completed"),
                                    "qa_ok": result.get("qa_ok"),
                                    "staged": (staging or {}).get("staged"),
                                },
                                hypothesis_id="H5", run_dir=run_dir, cfg=cfg,
                            )
                        finally:
                            with jobs_lock:
                                jobs[job_id]["harness_running"] = False
                        _append_plugin_logs(inbox, [{
                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "level": "info",
                            "title": "Auto-fix finished",
                            "detail": job_id,
                            "extra": {
                                "job_id": job_id,
                                "run_dir": run_dir,
                                "harness": _harness_summary_fields(
                                    harness_summary, pipeline_result=result,
                                ),
                            },
                        }], manifest)
                    else:
                        harness_summary = (
                            result.get("repair")
                            or result.get("harness")
                            or _load_harness_loop_summary(run_dir)
                        )
                        if harness_summary is None and already_ran_harness:
                            harness_summary = {
                                "rounds_completed": result.get("harness_rounds", 0),
                                "stopped": result.get("harness_stopped"),
                                "qa_ok": result.get("qa_ok"),
                            }
                        if harness_summary is None:
                            harness_summary = {
                                "rounds_completed": 0,
                                "stopped": repair_reason,
                                "qa_ok": (qa or {}).get("ok"),
                            }
                with jobs_lock:
                    if _job_cancelled(job_id):
                        return
                    jobs[job_id].update(
                        status="done" if result.get("ok") else "failed",
                        result=result, run_dir=run_dir,
                        error=None if result.get("ok") else (result.get("error") or "pipeline reported failure"),
                        failed_stage=None if result.get("ok") else result.get("failed_stage"),
                    )
                    if staging:
                        jobs[job_id].update(
                            staged=staging["staged"],
                            doc_id=staging.get("doc_id"),
                            layer_count=staging.get("layer_count"),
                            staging_error=staging.get("staging_error"),
                            design_url=staging.get("design_url"),
                        )
                    if harness_summary is not None:
                        jobs[job_id].update(_harness_summary_fields(
                            harness_summary, run_dir=run_dir, pipeline_result=result,
                        ))
                if result.get("ok"):
                    _record_history(inbox, time.time() - started, run_dir)
                    _append_plugin_logs(inbox, [{
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "level": "info",
                        "title": "Pipeline complete",
                        "detail": job_id,
                        "extra": {"job_id": job_id, "run_dir": run_dir, "duration_s": round(time.time() - started, 2)},
                    }], manifest)
                else:
                    _append_plugin_logs(inbox, [{
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "level": "error",
                        "title": "Pipeline failed",
                        "detail": result.get("error") or "pipeline reported failure",
                        "extra": {"job_id": job_id, "run_dir": run_dir, "result": result},
                    }], manifest)
            except Exception as exc:
                with jobs_lock:
                    if not _job_cancelled(job_id):
                        jobs[job_id].update(
                            status="failed",
                            error=str(exc),
                            traceback=traceback.format_exc(),
                            run_dir=run_dir,
                            failed_stage=getattr(exc, "failed_stage", None),
                        )
                if _job_cancelled(job_id):
                    return
                _append_plugin_logs(inbox, [{
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "level": "error",
                    "title": "Pipeline exception",
                    "detail": str(exc),
                    "extra": {"job_id": job_id, "run_dir": run_dir, "traceback": traceback.format_exc()},
                }], manifest)
        finally:
            _finish_job_thread(job_id)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body=b"", ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
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
                manifest = self._manifest()
                payload = _health_payload(inbox, base_cfg)
                payload["has_run"] = bool(manifest)
                if manifest:
                    payload["schema_version"] = manifest.get("schema_version")
                return self._send(200, json.dumps(payload, default=str).encode(), "application/json")
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
            if u.path == "/run-summary":
                job_id = parse_qs(u.query).get("job_id", [""])[0]
                with jobs_lock:
                    job = dict(jobs.get(job_id) or {})
                if not job:
                    return self._send(404, b'{"ok":false,"error":"unknown job_id"}', "application/json")
                manifest = _read_json_file(os.path.join(inbox, "inbox.json"))
                run_dir = job.get("run_dir")
                if manifest and run_dir and manifest.get("run_dir"):
                    manifest_run = os.path.abspath(str(manifest.get("run_dir")))
                    if manifest_run != os.path.abspath(str(run_dir)):
                        manifest = None
                payload = {
                    "ok": True,
                    "job_id": job_id,
                    "status": job.get("status"),
                    "staged": job.get("staged") if job.get("staged") is not None else bool(manifest),
                    "doc_id": job.get("doc_id") or (manifest or {}).get("doc_id"),
                    "design_url": job.get("design_url") or "/design.json",
                    "staging_error": job.get("staging_error"),
                    "run_dir": run_dir,
                    "manifest": manifest,
                    "harness_rounds": job.get("harness_rounds"),
                    "harness_stopped": job.get("harness_stopped"),
                    "final_qa_ok": job.get("final_qa_ok"),
                }
                return self._send(200, json.dumps(payload, default=str).encode(), "application/json")
            if u.path == "/process":
                job_id = parse_qs(u.query).get("job_id", [""])[0]
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        return self._send(404, b'{"ok":false,"error":"unknown job_id"}', "application/json")
                    snapshot = dict(job)
                    debug_sid = _debug_session_id(cfg=base_cfg)
                    if debug_sid:
                        snapshot["debug_session"] = debug_sid
                    status = snapshot.get("status")
                    if status == "running" and snapshot.get("run_dir"):
                        snapshot["stage"] = tail_running_stage(snapshot["run_dir"])
                        snapshot["agent_debug"] = _agent_debug_tail(snapshot.get("run_dir"), cfg=base_cfg)
                    if status == "failed":
                        tb = snapshot.get("traceback") or ""
                        if tb:
                            lines = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
                            snapshot["error_detail"] = "\n".join(lines[-5:])
                        agent_debug = _agent_debug_tail(snapshot.get("run_dir"), cfg=base_cfg)
                        snapshot["agent_debug"] = agent_debug
                        if snapshot.get("run_dir"):
                            snapshot["failed_stage"] = detect_failed_stage(
                                snapshot["run_dir"],
                                error_text=str(snapshot.get("error") or ""),
                                explicit_stage=snapshot.get("failed_stage"),
                                agent_debug=agent_debug,
                            )
                        classified = classify_processing_error(
                            error=str(snapshot.get("error") or ""),
                            traceback_text=tb,
                            failed_stage=snapshot.get("failed_stage"),
                            agent_debug=agent_debug,
                        )
                        snapshot["failed_stage"] = classified.get("failed_stage")
                        snapshot["error_code"] = classified.get("error_code")
                        snapshot["error_hint"] = classified.get("error_hint")
                        snapshot["user_title"] = classified.get("user_title")
                        snapshot["user_detail"] = classified.get("user_detail")
                    snapshot.pop("traceback", None)
                    if status == "running" and snapshot.get("started_at"):
                        elapsed = time.time() - snapshot["started_at"]
                        snapshot["elapsed_s"] = round(elapsed, 1)
                        eta_s, sample_size, progress_pct = _estimate_eta(
                            inbox, elapsed, snapshot.get("stage"),
                        )
                        if eta_s is not None:
                            snapshot["eta_s"] = eta_s
                            snapshot["eta_sample_size"] = sample_size
                        if progress_pct is not None:
                            snapshot["progress_pct"] = progress_pct
                    if status == "done":
                        snapshot.setdefault("design_url", "/design.json")
                        if snapshot.get("staged") is None:
                            manifest = _read_json_file(os.path.join(inbox, "inbox.json"))
                            snapshot["staged"] = bool(manifest)
                            if manifest and not snapshot.get("doc_id"):
                                snapshot["doc_id"] = manifest.get("doc_id")
                payload = {"ok": True, "job_id": job_id, **snapshot}
                return self._send(200, json.dumps(payload, default=str).encode(), "application/json")
            return self._send(404)

        def do_POST(self):
            route = urlparse(self.path).path
            if route == "/repo/update":
                query = parse_qs(urlparse(self.path).query)
                remote = (query.get("remote") or ["newrepo"])[0] or "newrepo"
                branch = (query.get("branch") or ["main"])[0] or "main"
                result = _run_repo_update(remote=remote, branch=branch)
                status = 200 if result.get("ok") else 500
                return self._send(status, json.dumps(result).encode(), "application/json")
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
                plugin_build = report.get("plugin_build")
                if isinstance(plugin_build, dict):
                    _save_plugin_client(inbox, plugin_build)
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
            if route == "/log":
                n = self._content_length(max_bytes=2 * 1024 * 1024)
                if n is None:
                    return self._send(400, b'{"ok":false,"error":"missing, invalid, or oversized Content-Length"}',
                                       "application/json")
                try:
                    payload = json.loads(self.rfile.read(n).decode("utf-8"))
                except Exception:
                    return self._send(400, b'{"ok":false,"error":"invalid json"}', "application/json")
                events = payload.get("events") or []
                if not isinstance(events, list):
                    return self._send(400, b'{"ok":false,"error":"events must be a list"}', "application/json")
                plugin_build = payload.get("plugin_build")
                if isinstance(plugin_build, dict):
                    _save_plugin_client(inbox, plugin_build)
                _append_plugin_logs(inbox, events, self._manifest())
                return self._send(200, b'{"ok":true}', "application/json")
            if route == "/process/cancel":
                job_id = parse_qs(urlparse(self.path).query).get("job_id", [""])[0]
                thread_done = None
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        return self._send(404, b'{"ok":false,"error":"unknown job_id"}', "application/json")
                    thread_done = job.get("thread_done")
                    if job.get("status") not in ("done", "failed", "cancelled"):
                        job["status"] = "cancelled"
                        job["error"] = "cancelled by user"
                    if active_job["id"] == job_id:
                        active_job["id"] = None
                if thread_done is not None:
                    thread_done.wait(timeout=60.0)
                _append_plugin_logs(inbox, [{
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "level": "info",
                    "title": "Processing cancelled",
                    "detail": job_id,
                }], self._manifest())
                return self._send(200, json.dumps({"ok": True, "job_id": job_id, "status": "cancelled"}).encode(),
                                   "application/json")
            if route == "/process":
                blockers = _preflight_blockers(base_cfg)
                if blockers:
                    detail = _format_blockers(blockers)
                    return self._send(
                        503,
                        json.dumps({
                            "ok": False,
                            "error": "bridge machine not ready — fix doctor.py blockers before uploading",
                            "detail": detail,
                            "blockers": blockers,
                        }).encode(),
                        "application/json",
                    )
                n = self._content_length(max_bytes=64 * 1024 * 1024)
                if n is None:
                    return self._send(400, b'{"ok":false,"error":"missing, invalid, or oversized Content-Length (64MB max)"}',
                                       "application/json")
                filename = _safe_name(parse_qs(urlparse(self.path).query).get("filename", [""])[0])
                data = self.rfile.read(n)
                job_id = uuid.uuid4().hex[:12]
                job_dir = os.path.join(inbox, "uploads", job_id)
                run_dir = os.path.join(job_dir, "run")
                queued_at = time.time()
                with jobs_lock:
                    if active_job["id"] is not None:
                        return self._send(409, b'{"ok":false,"error":"another image is already processing; wait for it to finish"}',
                                           "application/json")
                    os.makedirs(job_dir, exist_ok=True)
                    image_path = os.path.join(job_dir, filename)
                    with open(image_path, "wb") as fh:
                        fh.write(data)
                    jobs[job_id] = {
                        "status": "queued", "filename": filename, "image_path": image_path,
                        "queued_at": queued_at,
                        "thread_done": threading.Event(),
                    }
                    active_job["id"] = job_id
                _append_plugin_logs(inbox, [{
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "level": "info",
                    "title": "Upload received",
                    "detail": f"{filename} ({n} bytes)",
                    "extra": {"job_id": job_id, "filename": filename},
                }], self._manifest())
                threading.Thread(target=run_job, args=(job_id, image_path, run_dir), daemon=True).start()
                payload = {"ok": True, "job_id": job_id, "status": "queued", "filename": filename}
                return self._send(202, json.dumps(payload).encode(), "application/json")
            return self._send(404)

        def log_message(self, *a):  # quiet
            pass
    return H


def main():
    configure_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=os.path.expanduser("~/figma-inbox"))
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1",
                     help="bind address; use 0.0.0.0 to accept connections from other machines (e.g. over Tailscale)")
    ap.add_argument("--config", default="config.yaml",
                     help="pipeline config for POST /process (only read if that endpoint is used)")
    ap.add_argument("--no-bootstrap", action="store_true",
                     help="skip auto-creating config.yaml and inbox")
    a = ap.parse_args()
    inbox = os.path.expanduser(a.inbox)
    config_path = _resolve_config_path(a.config)
    if not a.no_bootstrap:
        from src.bridge_bootstrap import prepare
        status = prepare(config_path=config_path, inbox=inbox)
        config_path = status["config_path"]
        inbox = status["inbox"]
        for warning in status.get("gpu_warnings") or []:
            safe_print(f"WARNING: {warning}")
    else:
        os.makedirs(inbox, exist_ok=True)
    host_label = "localhost" if a.host in ("127.0.0.1", "localhost") else a.host
    safe_print()
    safe_print("=" * 52)
    safe_print("  Ad Decompiler bridge")
    safe_print(f"  http://{host_label}:{a.port}/health")
    safe_print(f"  inbox:  {inbox}")
    safe_print(f"  config: {config_path if os.path.exists(config_path) else '(missing - uploads need config.yaml)'}")
    safe_print("=" * 52)
    safe_print()
    ThreadingHTTPServer((a.host, a.port), make_handler(inbox, config_path)).serve_forever()


if __name__ == "__main__":
    main()
