#!/usr/bin/env python3
"""Fail fast on a GPU worker before spending time on a broken image run.

The decompiler intentionally degrades when an optional model is unavailable.  That is useful
on a Mac, but dangerous on the RTX box: a benchmark can otherwise finish with fallback boxes
and look like a model run.  This command makes every active dependency explicit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_pipeline import load_cfg


def _check(name, ok, detail, required=False):
    return {"name": name, "ok": bool(ok), "required": bool(required), "detail": str(detail)}


def _module(name):
    return importlib.util.find_spec(name) is not None


def _torch(device):
    try:
        import torch
        available = bool(torch.cuda.is_available())
        if device != "cuda":
            return _check("torch", True, f"device configured as {device}")
        if not available:
            return _check("cuda", False, "torch cannot see a CUDA device", required=True)
        return _check(
            "cuda", True,
            f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}",
            required=True,
        )
    except Exception as exc:
        return _check("torch", device != "cuda", f"unavailable: {exc}", required=device == "cuda")


def _http(url):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _required_qwen(cfg: dict) -> bool:
    """Qwen is advisory unless a particular benchmark explicitly makes it required."""
    qwen = cfg.get("qwen") or {}
    return bool(qwen.get("enabled", True) and qwen.get("required", False))


def inspect(cfg, root: Path) -> dict:
    """Return JSON-friendly readiness evidence without importing heavyweight models."""
    checks = []
    device = str(cfg.get("device", "cpu")).lower()
    checks.append(_check("python", sys.version_info >= (3, 10), sys.version.split()[0], required=True))
    checks.append(_torch(device))

    ocr_cfg = cfg.get("ocr") or {}
    primary = str(ocr_cfg.get("primary", "ppocr-v6")).lower()
    module_for_ocr = {"ppocr-v6": "paddleocr", "ppocr": "paddleocr", "surya": "surya",
                      "doctr": "doctr", "tesseract": "pytesseract"}.get(primary, primary)
    checks.append(_check(f"ocr:{primary}", _module(module_for_ocr),
                         f"python module {module_for_ocr}", required=True))
    for challenger in ocr_cfg.get("challengers") or []:
        name = str(challenger).lower()
        module = {"ppocr-v6": "paddleocr", "ppocr": "paddleocr", "surya": "surya",
                  "doctr": "doctr", "tesseract": "pytesseract"}.get(name, name)
        checks.append(_check(f"ocr challenger:{name}", _module(module), f"python module {module}"))

    runtime = cfg.get("runtime") or {}
    sam = cfg.get("sam3") or {}
    if sam.get("enabled", False):
        checkpoint = os.path.expandvars(os.path.expanduser(str(sam.get("checkpoint", ""))))
        checks.append(_check("sam3 package", _module("sam3"), "official sam3 package", required=True))
        checkpoint_ok = bool(checkpoint and os.path.isfile(checkpoint)
                             and os.path.getsize(checkpoint) > 0)
        checks.append(_check("sam3 checkpoint", checkpoint_ok, checkpoint or "not set", required=True))
        bpe = sam.get("bpe_path")
        if bpe:
            bpe = os.path.expandvars(os.path.expanduser(str(bpe)))
            checks.append(_check("sam3 BPE", os.path.isfile(bpe), bpe, required=True))

    qwen = cfg.get("qwen") or {}
    if qwen.get("enabled", True):
        mode = str(qwen.get("mode", "comfyui"))
        if mode == "comfyui":
            workflow = Path(str(qwen.get("workflow", "")))
            if not workflow.is_absolute():
                workflow = root / workflow
            checks.append(_check("qwen workflow", workflow.is_file(), str(workflow), required=_required_qwen(cfg)))
            base = str(cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
            checks.append(_check("ComfyUI", _http(f"{base}/system_stats"), base, required=_required_qwen(cfg)))
        else:
            checks.append(_check("Qwen layered pipeline", _module("diffusers"), "git diffusers build",
                                 required=_required_qwen(cfg)))

    for binary in ("vtracer", "potrace"):
        checks.append(_check(binary, bool(shutil.which(binary)), "on PATH"))
    # Big-LaMa quality directly determines the clean background plate. Under
    # require_active_models it must be a real acceptance condition like SAM/OCR,
    # not silently optional — an OpenCV fallback degrades plate quality (see
    # src/run_report.py::_required, run_pipeline.py inpaint stage).
    inpaint_required = bool(runtime.get("require_active_models", False) and
                             str((cfg.get("inpaint") or {}).get("mode", "auto")).lower() != "opencv")
    checks.append(_check("Big-LaMa", _module("simple_lama_inpainting"),
                          "optional; OpenCV fallback exists" if not inpaint_required
                          else "required by runtime.require_active_models; OpenCV fallback degrades background plate",
                          required=inpaint_required))
    checks.append(_check("Figma bridge", _module("requests"), "required only for plugin staging"))

    blockers = [item for item in checks if item["required"] and not item["ok"]]
    warnings = [item for item in checks if not item["required"] and not item["ok"]]
    return {
        "ok": not blockers,
        "device": device,
        "policy": {
            "require_active_models": bool(runtime.get("require_active_models", False)),
            "qwen_required": _required_qwen(cfg),
        },
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Check whether this machine can run the configured decompiler")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true", help="print only JSON")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    result = inspect(cfg, Path(__file__).resolve().parent)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("READY" if result["ok"] else "NOT READY")
        for item in result["checks"]:
            state = "OK" if item["ok"] else ("BLOCK" if item["required"] else "WARN")
            print(f"{state:5} {item['name']}: {item['detail']}")
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
