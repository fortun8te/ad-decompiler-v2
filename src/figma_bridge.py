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
from src.agent_debug import log as _agent_log, tail as _agent_debug_tail
from src.error_messages import classify_processing_error, detect_failed_stage


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


def _tail_stage(run_dir):
    """Best-effort read of pipeline.log's last matched stage — purely informational, never
    raises (a half-written line or a run_dir that doesn't exist yet just means 'no stage yet')."""
    return detect_failed_stage(run_dir)


def _history_path(inbox):
    return os.path.join(inbox, ".process_history.json")


def _read_history(inbox):
    try:
        with open(_history_path(inbox), encoding="utf-8") as fh:
            data = json.load(fh)
        return [float(d) for d in data.get("durations_s") or [] if isinstance(d, (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def _record_history(inbox, duration_s):
    durations = (_read_history(inbox) + [duration_s])[-10:]  # a short rolling window adapts to a changed machine/config
    try:
        _atomic_write(_history_path(inbox), json.dumps({"durations_s": durations}).encode())
    except OSError:
        pass


def _atomic_write(path, data: bytes):
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(temp_path, path)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    for path in paths:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.writelines(lines)
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
    try:
        from doctor import inspect as _doctor_inspect, ocr_ready_summary as _ocr_ready_summary
        from pathlib import Path
        repo_root = Path(_repo_root())
        cfg = base_cfg or {}
        doctor = _doctor_inspect(cfg, repo_root)
        payload["machine_ready"] = doctor.get("ok")
        payload["machine_blockers"] = doctor.get("blockers") or []
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


def make_handler(inbox, config_path=None):
    jobs = {}
    jobs_lock = threading.RLock()
    base_cfg = _load_cfg(config_path)
    active_job = {"id": None}

    def _job_cancelled(job_id):
        with jobs_lock:
            return jobs.get(job_id, {}).get("status") == "cancelled"

    def run_job(job_id, image_path, run_dir):
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
            import copy
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
                    hypothesis_id="H4", run_dir=run_dir,
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
                    hypothesis_id="H4", run_dir=run_dir,
                )
            # #endregion
            import run_pipeline  # heavy: torch/paddleocr/sam3/... — only imported on first use
            cfg = copy.deepcopy(base_cfg or {})
            cfg["figma"] = {**cfg.get("figma", {}), "enabled": True, "mode": "plugin", "inbox": inbox}
            result = run_pipeline.run_one(image_path, run_dir, cfg)
            with jobs_lock:
                if _job_cancelled(job_id):
                    return
                jobs[job_id].update(
                    status="done" if result.get("ok") else "failed",
                    result=result, run_dir=run_dir,
                    error=None if result.get("ok") else (result.get("error") or "pipeline reported failure"),
                    failed_stage=None if result.get("ok") else result.get("failed_stage"),
                )
            if result.get("ok"):
                _record_history(inbox, time.time() - started)
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
            with jobs_lock:
                if active_job["id"] == job_id:
                    active_job["id"] = None

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
            if u.path == "/process":
                job_id = parse_qs(u.query).get("job_id", [""])[0]
                with jobs_lock:
                    job = dict(jobs.get(job_id) or {})
                if not job:
                    return self._send(404, b'{"ok":false,"error":"unknown job_id"}', "application/json")
                if job.get("status") == "failed":
                    tb = job.get("traceback") or ""
                    if tb:
                        lines = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
                        job["error_detail"] = "\n".join(lines[-5:])
                    agent_debug = _agent_debug_tail(job.get("run_dir"))
                    job["agent_debug"] = agent_debug
                    if job.get("run_dir"):
                        job["failed_stage"] = detect_failed_stage(
                            job["run_dir"],
                            error_text=str(job.get("error") or ""),
                            explicit_stage=job.get("failed_stage"),
                            agent_debug=agent_debug,
                        )
                    classified = classify_processing_error(
                        error=str(job.get("error") or ""),
                        traceback_text=tb,
                        failed_stage=job.get("failed_stage"),
                        agent_debug=agent_debug,
                    )
                    job["failed_stage"] = classified.get("failed_stage")
                    job["error_code"] = classified.get("error_code")
                    job["error_hint"] = classified.get("error_hint")
                    job["user_title"] = classified.get("user_title")
                    job["user_detail"] = classified.get("user_detail")
                job.pop("traceback", None)
                if job.get("status") == "running" and job.get("run_dir"):
                    job["stage"] = _tail_stage(job["run_dir"])
                    job["agent_debug"] = _agent_debug_tail(job.get("run_dir"))
                if job.get("started_at") and job.get("status") in ("running", "queued"):
                    elapsed = time.time() - job["started_at"]
                    job["elapsed_s"] = round(elapsed, 1)
                    history = _read_history(inbox)
                    if history:
                        # Median, not mean: one very slow/cold-start run shouldn't blow out
                        # every ETA after it, and a short rolling window (see
                        # _record_history) already adapts if the machine/config changes.
                        job["eta_s"] = max(0, round(statistics.median(history) - elapsed, 1))
                        job["eta_sample_size"] = len(history)
                payload = {"ok": True, "job_id": job_id, **job}
                return self._send(200, json.dumps(payload, default=str).encode(), "application/json")
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
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        return self._send(404, b'{"ok":false,"error":"unknown job_id"}', "application/json")
                    if job.get("status") not in ("done", "failed", "cancelled"):
                        job["status"] = "cancelled"
                        job["error"] = "cancelled by user"
                    if active_job["id"] == job_id:
                        active_job["id"] = None
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
                        "started_at": queued_at,
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
