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
import sys
from pathlib import Path

def load_cfg(path):
    """Lightweight config loader — avoids importing run_pipeline (heavy torch/paddle deps)."""
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


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


def _doctr_gpu(device, primary: str):
    """docTR primary on CUDA needs torch GPU — the working RTX 50-series OCR path."""
    if device != "cuda" or primary != "doctr":
        return _check("doctr gpu", True, "not required", required=False)
    try:
        import torch
        if not _module("doctr"):
            return _check("doctr gpu", False, "python-doctr not installed", required=True)
        if not torch.cuda.is_available():
            return _check("doctr gpu", False, "torch cannot see a CUDA device for doctr primary", required=True)
        return _check(
            "doctr gpu", True,
            f"doctr will run on {torch.cuda.get_device_name(0)} via torch {torch.__version__}",
            required=True,
        )
    except Exception as exc:
        return _check("doctr gpu", False, f"probe failed: {exc}", required=True)


def _cudnn(device):
    """Lightweight cuDNN probe — PaddleOCR GPU on Windows often fails without it."""
    if device != "cuda":
        return _check("cudnn", True, "not required (cpu)", required=False)
    try:
        import torch
        if not torch.cuda.is_available():
            return _check("cudnn", False, "CUDA unavailable — cuDNN cannot be used", required=False)
        available = bool(torch.backends.cudnn.is_available())
        detail = (
            f"cuDNN available (version {torch.backends.cudnn.version()})"
            if available
            else "cuDNN missing — PaddleOCR GPU on Windows often fails; reinstall paddlepaddle-gpu + matching cuDNN"
        )
        return _check("cudnn", available, detail, required=False)
    except Exception as exc:
        return _check("cudnn", False, f"probe failed: {exc}", required=False)


def _http(url, timeout=0.3):
    # Short timeout on purpose: this is a liveness probe (e.g. ComfyUI on :8188), and
    # /health calls it. A refused port fails instantly; a firewall-*dropped* port would
    # otherwise hang for the full timeout, so keep it small — a real local backend
    # answers in well under 0.3s. Cold /health also pays a one-time torch import cost
    # (~1s), so probes must stay well under the bridge tests' 2s client timeout.
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _vlm_model_loaded(base: str, model: str, timeout=0.5) -> tuple[bool, str]:
    """True only when the OpenAI-compatible server is up AND lists the configured model.

    Liveness goes through _http first so tests (and any future probe policy) keep a single
    seam to stub; only a live server gets the follow-up model-list read."""
    if not _http(f"{base}/models"):
        return False, f"{base} unreachable"
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/models", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return False, f"{base} model list unreadable ({exc})"
    ids = [str(item.get("id", "")) for item in (data.get("data") or [])]
    if not ids:
        return False, f"{base} up but NO model loaded (e.g. `lms load {model or '<model>'}`)"
    if model and model not in ids:
        return False, f"{base} up but '{model}' not loaded (loaded: {', '.join(ids[:4])})"
    return True, f"{base} ({model or ids[0]} loaded)"


def _required_qwen(cfg: dict) -> bool:
    """Qwen is advisory unless a particular benchmark explicitly makes it required."""
    qwen = cfg.get("qwen") or {}
    return bool(qwen.get("enabled", True) and qwen.get("required", False))


def _vlm_feature_enabled(cfg: dict) -> bool:
    vlm = cfg.get("vlm") or {}
    if vlm.get("enabled"):
        return True
    if (vlm.get("segment_filter") or {}).get("enabled"):
        return True
    if (vlm.get("font_judge") or {}).get("enabled"):
        return True
    if (vlm.get("scene_text") or {}).get("enabled"):
        return True
    if (vlm.get("ocr_judge") or {}).get("enabled"):
        return True
    if (vlm.get("element_propose") or {}).get("enabled"):
        return True
    return False


def _tesseract_binary() -> str | None:
    return shutil.which("tesseract")


def _ocr_engine_module(name: str) -> str:
    return {"ppocr-v6": "paddleocr", "ppocr": "paddleocr", "surya": "surya",
            "doctr": "doctr", "easyocr": "easyocr", "tesseract": "pytesseract"}.get(
                str(name).lower(), str(name).lower())


def _ocr_fallback_ready(engine: str) -> tuple[bool, str]:
    """Return whether a configured OCR fallback can run on this machine."""
    name = str(engine).lower()
    if name == "tesseract":
        binary = _tesseract_binary()
        if not binary:
            return False, "tesseract binary not on PATH"
        if not _module("pytesseract"):
            return False, "pip install pytesseract"
        return True, binary
    module = _ocr_engine_module(name)
    if not _module(module):
        return False, f"python module {module}"
    return True, f"python module {module}"


def ocr_fallback_status(cfg: dict) -> dict:
    """Summarize configured OCR fallbacks that are actually runnable."""
    ocr_cfg = cfg.get("ocr") or {}
    configured = [str(name).lower() for name in (ocr_cfg.get("fallback_engines") or [])]
    auto = ocr_cfg.get("auto_fallback_tesseract", True)
    if auto and "tesseract" not in configured:
        configured.append("tesseract")
    available, unavailable = [], []
    for name in configured:
        ok, detail = _ocr_fallback_ready(name)
        item = {"engine": name, "detail": detail}
        if ok:
            available.append(item)
        else:
            unavailable.append(item)
    return {
        "configured": configured,
        "available": available,
        "unavailable": unavailable,
        "ready": bool(available),
    }


def inspect(cfg, root: Path) -> dict:
    """Return JSON-friendly readiness evidence without importing heavyweight models."""
    checks = []
    device = str(cfg.get("device", "cpu")).lower()
    checks.append(_check("python", sys.version_info >= (3, 10), sys.version.split()[0], required=True))
    checks.append(_torch(device))
    cudnn = _cudnn(device)
    checks.append(cudnn)

    ocr_cfg = cfg.get("ocr") or {}
    primary = str(ocr_cfg.get("primary", "ppocr-v6")).lower()
    module_for_ocr = _ocr_engine_module(primary)
    checks.append(_check(f"ocr:{primary}", _module(module_for_ocr),
                         f"python module {module_for_ocr}", required=True))
    if primary == "doctr":
        checks.append(_doctr_gpu(device, primary))
    if primary == "tesseract":
        checks.append(_check("tesseract binary", bool(_tesseract_binary()),
                             _tesseract_binary() or "not on PATH", required=True))
    for challenger in ocr_cfg.get("challengers") or []:
        name = str(challenger).lower()
        module = _ocr_engine_module(name)
        checks.append(_check(f"ocr challenger:{name}", _module(module), f"python module {module}"))
        if name == "tesseract":
            checks.append(_check("tesseract binary", bool(_tesseract_binary()),
                                 _tesseract_binary() or "not on PATH"))
    fallback = ocr_fallback_status(cfg)
    for item in fallback["configured"]:
        ok, detail = _ocr_fallback_ready(item)
        checks.append(_check(f"ocr fallback:{item}", ok, detail))

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

    # ComfyUI liveness is probed at most once per inspect() call and the result reused
    # below (Big-LaMa/inpaint stack check) -- probing the same base URL twice doubled
    # /health latency under a cold firewall-dropped port (see doctor.py timing notes).
    comfy_probe_cache: dict[str, bool] = {}

    def _comfy_probe(base: str) -> bool:
        if base not in comfy_probe_cache:
            comfy_probe_cache[base] = _http(f"{base}/system_stats")
        return comfy_probe_cache[base]

    qwen = cfg.get("qwen") or {}
    if qwen.get("enabled", True):
        mode = str(qwen.get("mode", "comfyui"))
        if mode == "comfyui":
            workflow = Path(str(qwen.get("workflow", "")))
            if not workflow.is_absolute():
                workflow = root / workflow
            checks.append(_check("qwen workflow", workflow.is_file(), str(workflow), required=_required_qwen(cfg)))
            base = str(cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
            checks.append(_check("ComfyUI", _comfy_probe(base), base, required=_required_qwen(cfg)))
        else:
            checks.append(_check("Qwen layered pipeline", _module("diffusers"), "git diffusers build",
                                 required=_required_qwen(cfg)))

    vlm = cfg.get("vlm") or {}
    if _vlm_feature_enabled(cfg):
        base = str(vlm.get("base_url", "http://127.0.0.1:1234/v1")).rstrip("/")
        model = str(vlm.get("model", ""))
        # /models answering 200 does NOT mean the configured model is loaded — LM Studio
        # returns an empty list after it idles a model out, and then every VLM call 400s
        # ("No models loaded"). That exact failure silently zeroed all VLM corrections in
        # a 16-image benchmark, so check for the model by id, not just server liveness.
        loaded, detail = _vlm_model_loaded(base, model)
        checks.append(_check("VLM server", loaded, detail, required=False))

    try:
        from src.vectorize import check_binaries as _vectorize_binaries
    except Exception:
        _vectorize_binaries = None
    if _vectorize_binaries:
        # None of these are required checks -- a broken/missing native dependency here
        # must never crash the whole readiness report (see the cairosvg/libcairo OSError
        # this guards against; check_binaries() also degrades that specific case, but this
        # is the last line of defense for any other native-lib surprise).
        try:
            vz = _vectorize_binaries(cfg)
        except Exception as exc:
            vz = {}
            checks.append(_check("vectorize:binaries", False, f"probe failed: {exc}", required=False))
        for name in ("vtracer", "potrace", "cairosvg"):
            info = vz.get(name) or {}
            detail = info.get("path", "unknown")
            if name == "cairosvg":
                checks.append(_check(
                    "cairosvg (vectorize gate)", info.get("ok"), detail, required=False,
                ))
            else:
                checks.append(_check(f"vectorize:{name}", info.get("ok"), detail, required=False))
    else:
        for binary in ("vtracer", "potrace"):
            checks.append(_check(f"vectorize:{binary}", bool(shutil.which(binary)), "on PATH"))
    # Big-LaMa quality directly determines the clean background plate. Under
    # require_active_models it must be a real acceptance condition like SAM/OCR,
    # not silently optional — an OpenCV fallback degrades plate quality (see
    # src/run_report.py::_required, run_pipeline.py inpaint stage).
    inpaint_cfg = cfg.get("inpaint") or {}
    inpaint_mode = str(inpaint_cfg.get("mode", "auto")).lower()
    inpaint_required = bool(runtime.get("require_active_models", False) and inpaint_mode != "opencv")
    lama_ok = _module("simple_lama_inpainting")
    comfy_ok = True
    comfy_detail = "not required (qwen disabled or non-comfyui mode)"
    qwen_enabled = bool((cfg.get("qwen") or {}).get("enabled", True))
    if qwen_enabled and str((cfg.get("qwen") or {}).get("mode", "comfyui")).lower() == "comfyui":
        base = str(cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
        comfy_ok = _comfy_probe(base)
        comfy_detail = base if comfy_ok else f"{base} unreachable"
    checks.append(_check("Big-LaMa", lama_ok,
                          "optional; OpenCV fallback exists" if not inpaint_required
                          else "required by runtime.require_active_models; OpenCV fallback degrades background plate",
                          required=inpaint_required))
    if inpaint_mode in ("auto", "big-lama", "lama", "simple-lama"):
        # Background inpainting (Big-LaMa) and Qwen's layered decomposition (ComfyUI) are
        # unrelated capabilities -- Qwen is advisory (see run_report._required's comment:
        # "it must not make the main SAM/OCR scene graph look unavailable merely because a
        # separate ComfyUI process is offline"). Gating inpaint readiness on ComfyUI made
        # this BLOCK even when Big-LaMa alone is installed and the real pipeline already
        # completes fine with ComfyUI down (qwen degrades to its own fallback separately).
        stack_ok = lama_ok
        stack_detail = (
            f"Big-LaMa={'installed' if lama_ok else 'missing'}; ComfyUI={comfy_detail} (qwen is advisory, not required here)"
        )
        checks.append(_check(
            "inpaint stack (Big-LaMa)",
            stack_ok,
            stack_detail,
            required=inpaint_required and inpaint_mode == "auto",
        ))
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
        "ocr_fallback": fallback,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
    }


def ocr_ready_summary(cfg, root: Path, report=None) -> dict:
    """Compact OCR readiness for bridge /health — module/CUDA checks only, no model loads.

    Pass an already-computed ``inspect()`` result as ``report`` to avoid a second
    inspection (which re-imports torch and re-probes ComfyUI)."""
    if report is None:
        report = inspect(cfg, root)
    primary = str((cfg.get("ocr") or {}).get("primary", "ppocr-v6")).lower()
    ocr_blockers = [item for item in report["blockers"] if item["name"].startswith("ocr")]
    if str(cfg.get("device", "cpu")).lower() == "cuda":
        cuda_check = next((item for item in report["checks"] if item["name"] == "cuda"), None)
        if cuda_check and not cuda_check["ok"]:
            ocr_blockers.append(cuda_check)
    ocr_warnings = [item for item in report["warnings"] if item["name"].startswith("ocr")]
    cudnn_check = next((item for item in report["checks"] if item["name"] == "cudnn"), None)
    if cudnn_check and not cudnn_check["ok"]:
        ocr_warnings.append(cudnn_check)
    fallback = report.get("ocr_fallback") or ocr_fallback_status(cfg)
    ocr_ok = not ocr_blockers or bool(fallback.get("ready"))
    return {
        "ok": ocr_ok,
        "primary": primary,
        "device": report.get("device"),
        "blockers": ocr_blockers,
        "warnings": ocr_warnings,
        "fallback": fallback,
        "machine_ok": report.get("ok"),
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
