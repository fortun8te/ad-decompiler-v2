"""Start required local quality services before doctor/benchmark acceptance checks."""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from urllib.parse import urlparse


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def _wait(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if _http_ok(url):
            return True
        time.sleep(1.0)
    return _http_ok(url)


def _start_comfy(cfg: dict, timeout_s: float) -> dict:
    comfy = ((cfg.get("inpaint") or {}).get("comfy") or {})
    base = str(comfy.get("base_url") or cfg.get("backend_url") or "http://127.0.0.1:8188").rstrip("/")
    health = base + "/system_stats"
    if _http_ok(health):
        return {"name": "comfyui", "ok": True, "action": "already-running", "detail": base}
    comfy_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(str(comfy.get("comfy_dir") or ""))))
    python = os.path.join(comfy_dir, ".venv", "Scripts", "python.exe")
    main = os.path.join(comfy_dir, "main.py")
    if not os.path.isfile(python) or not os.path.isfile(main):
        return {"name": "comfyui", "ok": False, "action": "not-started",
                "detail": f"missing ComfyUI runtime under {comfy_dir}"}
    parsed = urlparse(base)
    port = str(parsed.port or 8188)
    host = parsed.hostname or "127.0.0.1"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(
            [python, "main.py", "--port", port, "--listen", host], cwd=comfy_dir,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except Exception as exc:
        return {"name": "comfyui", "ok": False, "action": "start-failed", "detail": str(exc)}
    ok = _wait(health, timeout_s)
    return {"name": "comfyui", "ok": ok, "action": "started" if ok else "start-timeout",
            "detail": base}


def _vlm_loaded(base: str, model: str) -> bool:
    try:
        import json
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return any(str(item.get("id", "")).casefold() == model.casefold()
                   for item in (payload.get("data") or []))
    except Exception:
        return False


def _start_vlm(cfg: dict, timeout_s: float) -> dict:
    vlm = cfg.get("vlm") or {}
    base = str(vlm.get("base_url") or "http://127.0.0.1:1234/v1").rstrip("/")
    model = str(vlm.get("model") or "google/gemma-4-e4b")
    if _vlm_loaded(base, model):
        return {"name": "vlm", "ok": True, "action": "already-loaded", "detail": model}
    try:
        completed = subprocess.run(
            ["lms", "load", model], check=False, capture_output=True, text=True,
            timeout=max(10.0, timeout_s), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return {"name": "vlm", "ok": False, "action": "load-failed", "detail": str(exc)}
    ok = completed.returncode == 0 and _vlm_loaded(base, model)
    detail = model if ok else (completed.stderr or completed.stdout or "model did not load")[-500:]
    return {"name": "vlm", "ok": ok, "action": "loaded" if ok else "load-failed", "detail": detail}


def ensure_services(cfg: dict) -> dict:
    startup = ((cfg.get("runtime") or {}).get("autostart") or {})
    if not startup.get("enabled", False):
        return {"ok": True, "enabled": False, "checks": []}
    timeout_s = float(startup.get("timeout_s", 90))
    checks = []
    if startup.get("comfyui", True) and ((cfg.get("inpaint") or {}).get("comfy") or {}).get("enabled"):
        checks.append(_start_comfy(cfg, timeout_s))
    if startup.get("vlm", True) and (cfg.get("vlm") or {}).get("enabled"):
        checks.append(_start_vlm(cfg, timeout_s))
    return {"ok": all(item.get("ok") for item in checks), "enabled": True, "checks": checks}


__all__ = ["ensure_services"]
