#!/usr/bin/env python3
"""Fail fast on a GPU worker before spending time on a broken image run.

The decompiler intentionally degrades when an optional model is unavailable.  That is useful
on a Mac, but dangerous on the RTX box: a benchmark can otherwise finish with fallback boxes
and look like a model run.  This command makes every active dependency explicit.
"""
from __future__ import annotations

import argparse
import importlib
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


def _fix_for(check: dict) -> str | None:
    """Plain, copyable next action for failed checks shown by launchers and /health."""
    if check.get("ok"):
        return None
    name = str(check.get("name", "")).lower()
    if name == "python":
        return "Install Python 3.12, then run setup_rtx.ps1 again."
    if name in ("cuda", "torch"):
        return "Update the NVIDIA driver, then rerun setup_rtx.ps1."
    if name == "doctr gpu":
        return "Rerun setup_rtx.ps1; if CUDA still fails, set device: cpu temporarily."
    if name == "cudnn":
        return "Rerun setup_rtx.ps1 after updating the NVIDIA driver."
    if name == "sam3 package":
        return "Rerun setup_rtx.ps1 to install the official SAM 3 code."
    if name in ("sam3 checkpoint", "sam3 bpe"):
        return "Download the official SAM 3 image checkpoint, then set its path under sam3 in config.yaml."
    if name.startswith("ocr:") or name.startswith("ocr fallback:"):
        return "Rerun setup_rtx.ps1. For Tesseract, run: winget install UB-Mannheim.TesseractOCR"
    if name == "tesseract binary":
        return "Run: winget install UB-Mannheim.TesseractOCR, then restart the bridge."
    if name == "vlm server":
        return f"Start LM Studio's local server and load {_DEFAULT_VLM_MODEL}."
    if name == "vlm model identity":
        return f"Set vlm.model to {_DEFAULT_VLM_MODEL} in config.yaml and load that model in LM Studio."
    if name == "comfyui":
        return "Start ComfyUI on port 8188, or set qwen.required: false."
    if name == "qwen workflow":
        return "Put the Qwen workflow JSON at the config path, or set qwen.required: false."
    if name == "flux comfyui":
        return "Start ComfyUI on port 8188 (or set inpaint.comfy.base_url), or set inpaint.comfy.required: false."
    if name == "flux inpaint workflow":
        return "Ensure workflows/flux_fill_inpaint_api.json exists (path from inpaint.comfy.workflow)."
    if name == "flux inpaint models":
        return "Run scripts/setup_flux_inpaint.ps1 -ComfyDir <ComfyUI> to fetch the Flux Fill GGUF + encoders + VAE."
    if name == "powerpaint adapter configuration":
        return "On the RTX worker, set inpaint.powerpaint.enabled: true plus adapter_module and callable in config.yaml."
    if name == "powerpaint adapter import":
        return "Install the configured PowerPaint adapter in this RTX Python environment, then run doctor.py --deep."
    if name.startswith("big-lama") or name.startswith("inpaint stack"):
        return "Run: .venv\\Scripts\\python.exe -m pip install simple-lama-inpainting"
    if name == "vectorize:vtracer":
        return "Rerun setup_rtx.ps1 to install the VTracer Python backend."
    if name == "vectorize:potrace":
        return "Run: choco install potrace, then restart the bridge."
    if name == "vectorization stack":
        return "Rerun setup_rtx.ps1 to install VTracer and the SVG render-back checker."
    if "vectorize gate" in name:
        return "Rerun setup_rtx.ps1 to install the SVG render-back checker."
    return None


def _module(name):
    return importlib.util.find_spec(name) is not None


def _cuda_total_mib() -> int | None:
    """Total VRAM (MiB) on device 0, or None when CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception:
        return None


# Rough resident footprints (MiB) for the concurrent-model VRAM sanity check.  gemma-4-e4b
# (~6.3 GB) lives persistently in LM Studio; Flux Fill (unet + t5xxl fp8 + clip_l + vae)
# lives in ComfyUI.  On a 16 GB card these two cannot coexist, which is why the inpaint
# boundary can evict the VLM (runtime.vram.evict_vlm_for_inpaint).
_VRAM_FOOTPRINT_MIB = {"vlm": 6300, "flux": 14000, "sam3": 3000}


def _vram_footprint_check(cfg: dict, flux_enabled: bool) -> dict | None:
    """Warn when the persistent VLM and Flux Fill would exceed VRAM without eviction."""
    if str(cfg.get("device", "cpu")).lower() != "cuda":
        return None
    vlm_on = _vlm_feature_enabled(cfg)
    if not (vlm_on and flux_enabled):
        return None
    evict = bool(((cfg.get("runtime") or {}).get("vram") or {}).get("evict_vlm_for_inpaint", False))
    total = _cuda_total_mib()
    concurrent = _VRAM_FOOTPRINT_MIB["vlm"] + _VRAM_FOOTPRINT_MIB["flux"]
    over = total is not None and concurrent > total
    if evict:
        return _check(
            "vram headroom", True,
            f"VLM (~{_VRAM_FOOTPRINT_MIB['vlm']}MiB) + Flux Fill (~{_VRAM_FOOTPRINT_MIB['flux']}MiB) "
            f"= ~{concurrent}MiB > {total or '?'}MiB GPU, but runtime.vram.evict_vlm_for_inpaint "
            "unloads the VLM during inpaint",
            required=False,
        )
    return _check(
        "vram headroom", not over,
        (f"VLM + Flux Fill ~{concurrent}MiB exceeds {total}MiB GPU with no eviction; set "
         "runtime.vram.evict_vlm_for_inpaint: true to unload the VLM during Flux inpaint"
         if over else f"~{concurrent}MiB estimated concurrent footprint fits {total or '?'}MiB GPU"),
        required=False,
    )


_FLUX_INPAINT_MODES = {"flux_comfy", "flux-comfy", "flux"}
_POWERPAINT_MODES = {"powerpaint", "power-paint"}


def _inpaint_policy(cfg: dict) -> dict:
    """Describe the configured inpaint route without importing or loading a model.

    Strict acceptance deliberately means *the requested route* has to be ready on the RTX
    worker.  It is stronger than the production fallback policy in ``src.inpaint`` and avoids
    presenting a benchmark as Flux/PowerPaint evidence after a quiet route downgrade.
    """
    inpaint = cfg.get("inpaint") or {}
    runtime = cfg.get("runtime") or {}
    mode = str(inpaint.get("mode", "auto")).lower()
    strict = bool(inpaint.get("strict_acceptance", False))
    active_models = bool(runtime.get("require_active_models", False))
    if mode in _FLUX_INPAINT_MODES:
        selected = "flux_comfy"
    elif mode in _POWERPAINT_MODES:
        selected = "powerpaint"
    elif mode in ("opencv", "cv2"):
        selected = "opencv"
    else:
        selected = "big_lama"
    requested_required = bool(selected != "opencv" and (strict or active_models))
    return {
        "mode": mode,
        "selected": selected,
        "strict_acceptance": strict,
        "require_active_models": active_models,
        "requested_model_required": requested_required,
    }


def _powerpaint_adapter_importable(module_name: str, callable_name: str) -> tuple[bool, str]:
    """Check the adapter boundary only; never load or infer with its model here."""
    if not module_name:
        return False, "adapter_module is not set"
    if not _module(module_name):
        return False, f"{module_name} is not importable"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{module_name} import failed ({type(exc).__name__}: {exc})"
    adapter = getattr(module, callable_name, None)
    if not callable(adapter):
        return False, f"{module_name}.{callable_name} is not callable"
    return True, f"{module_name}.{callable_name} is callable; model/weights not executed"


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


_DEFAULT_VLM_MODEL = "google/gemma-4-e4b"


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
    inpaint_policy = _inpaint_policy(cfg)
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

    # Flux Fill inpaint backend (ComfyUI GGUF). It is advisory for ordinary auto mode,
    # but a specifically selected Flux route is a blocker for strict/active acceptance.
    # Reports ComfyUI reachability, the workflow file, and (when comfy_dir is known) the
    # presence of the GGUF/CLIP/VAE model files used by the checked-in workflow.
    # A generic Flux-dev turbo LoRA is intentionally not required: Flux Fill has a
    # different mask-concatenated input shape and that LoRA makes the workflow fail.
    inpaint_cfg_flux = cfg.get("inpaint") or {}
    comfy_inpaint = inpaint_cfg_flux.get("comfy") or {}
    inpaint_mode_flux = inpaint_policy["mode"]
    flux_selected = inpaint_policy["selected"] == "flux_comfy"
    flux_enabled = bool(comfy_inpaint.get("enabled")) or flux_selected
    if flux_enabled:
        flux_required = bool(
            comfy_inpaint.get("required")
            or (flux_selected and inpaint_policy["requested_model_required"])
        )
        flux_wf = Path(str(comfy_inpaint.get("workflow", "workflows/flux_fill_inpaint_api.json")))
        if not flux_wf.is_absolute():
            flux_wf = root / flux_wf
        checks.append(_check("flux inpaint workflow", flux_wf.is_file(), str(flux_wf), required=flux_required))
        flux_base = str(comfy_inpaint.get("base_url") or cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
        checks.append(_check("flux ComfyUI", _comfy_probe(flux_base), flux_base, required=flux_required))
        comfy_dir = comfy_inpaint.get("comfy_dir")
        model_names = comfy_inpaint.get("models") or {}
        flux_defaults = {
            "unet_gguf": "flux1-fill-dev-Q6_K.gguf",
            "t5xxl": "t5xxl_fp8_e4m3fn.safetensors",
            "clip_l": "clip_l.safetensors",
            "vae": "ae.safetensors",
        }
        flux_subdirs = {
            "unet_gguf": ["unet", "diffusion_models"],
            "t5xxl": ["clip", "text_encoders"],
            "clip_l": ["clip", "text_encoders"],
            "vae": ["vae"],
        }
        if comfy_dir:
            comfy_dir = os.path.expandvars(os.path.expanduser(str(comfy_dir)))
            missing = []
            for key, default_name in flux_defaults.items():
                fname = str(model_names.get(key, default_name))
                found = any(
                    os.path.isfile(os.path.join(comfy_dir, "models", sub, fname))
                    for sub in flux_subdirs[key]
                )
                if not found:
                    missing.append(f"{fname} (models/{'|'.join(flux_subdirs[key])})")
            checks.append(_check(
                "flux inpaint models",
                not missing,
                "all Flux Fill model files present" if not missing else "missing: " + "; ".join(missing),
                required=flux_required,
            ))
        else:
            checks.append(_check(
                "flux inpaint models", not flux_required,
                ("set inpaint.comfy.comfy_dir to verify the GGUF/CLIP/VAE files"
                 if not flux_required else
                 "cannot verify Flux Fill weights without inpaint.comfy.comfy_dir; "
                 "a strict Flux acceptance run must expose the local ComfyUI model directory"),
                required=flux_required,
            ))

    # PowerPaint stays an explicitly user-supplied adapter: this project neither installs
    # nor guesses a model package. Doctor can honestly verify configuration/importability;
    # runtime_smoke is the separate proof that the adapter and its model actually execute.
    powerpaint = inpaint_cfg_flux.get("powerpaint") or {}
    powerpaint_selected = inpaint_policy["selected"] == "powerpaint"
    powerpaint_enabled = bool(powerpaint.get("enabled", False))
    adapter_module = str(powerpaint.get("adapter_module") or "").strip()
    powerpaint_required = bool(
        powerpaint.get("required")
        or (powerpaint_selected and inpaint_policy["requested_model_required"])
    )
    if powerpaint_selected or powerpaint_enabled or adapter_module:
        configured = powerpaint_enabled and bool(adapter_module)
        checks.append(_check(
            "PowerPaint adapter configuration", configured,
            (f"enabled={powerpaint_enabled}; adapter_module={adapter_module or '<unset>'}; "
             "runtime not yet proven"),
            required=powerpaint_required,
        ))
        importable, adapter_detail = _powerpaint_adapter_importable(
            adapter_module, str(powerpaint.get("callable") or "inpaint").strip(),
        )
        checks.append(_check("PowerPaint adapter import", importable, adapter_detail,
                             required=powerpaint_required))

    vlm = cfg.get("vlm") or {}
    if _vlm_feature_enabled(cfg):
        base = str(vlm.get("base_url", "http://127.0.0.1:1234/v1")).rstrip("/")
        model = str(vlm.get("model") or _DEFAULT_VLM_MODEL)
        # /models answering 200 does NOT mean the configured model is loaded — LM Studio
        # returns an empty list after it idles a model out, and then every VLM call 400s
        # ("No models loaded"). That exact failure silently zeroed all VLM corrections in
        # a 16-image benchmark, so check for the model by id, not just server liveness.
        loaded, detail = _vlm_model_loaded(base, model)
        vlm_required = bool(runtime.get("require_active_models", False))
        checks.append(_check("VLM server", loaded, detail, required=vlm_required))
        # Keep the requested identity explicit in doctor.json. This prevents a run being
        # presented as Gemma 4 e4b evidence when the config quietly points at another VLM.
        gemma_identity = model.casefold() == _DEFAULT_VLM_MODEL.casefold()
        checks.append(_check(
            "VLM model identity",
            gemma_identity,
            f"configured={model}; expected={_DEFAULT_VLM_MODEL}",
            required=vlm_required,
        ))

    try:
        from src.vectorize import check_binaries as _vectorize_binaries
    except Exception:
        _vectorize_binaries = None
    vector_required = bool(
        runtime.get("require_active_models", False)
        and (cfg.get("vectorize") or {}).get("enabled", True)
    )
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
        for name in ("vtracer", "potrace", "cairosvg", "resvg"):
            info = vz.get(name) or {}
            detail = info.get("path", "unknown")
            if name in ("cairosvg", "resvg"):
                checks.append(_check(
                    f"{name} (vectorize gate)", info.get("ok"), detail, required=False,
                ))
            else:
                checks.append(_check(f"vectorize:{name}", info.get("ok"), detail, required=False))
        tracer_ok = bool((vz.get("vtracer") or {}).get("ok"))
        gate_ok = bool(
            (vz.get("cairosvg") or {}).get("ok")
            or (vz.get("resvg") or {}).get("ok")
        )
        checks.append(_check(
            "vectorization stack",
            tracer_ok and gate_ok,
            "needs color-capable VTracer plus CairoSVG or resvg render-back validation; Potrace remains a monochrome fallback",
            required=vector_required,
        ))
    else:
        tracer_paths = {binary: shutil.which(binary) for binary in ("vtracer", "potrace")}
        for binary, binary_path in tracer_paths.items():
            checks.append(_check(f"vectorize:{binary}", bool(binary_path), binary_path or "not on PATH"))
        checks.append(_check(
            "vectorization stack", False,
            "vectorization probe unavailable; install VTracer or Potrace and CairoSVG",
            required=vector_required,
        ))
    # Big-LaMa is the selected route for auto/LaMa configurations. It is not a blocker
    # for a strict Flux or PowerPaint selection: those runs must prove their requested
    # backend rather than silently requiring an unrelated fallback model as well.
    inpaint_mode = inpaint_policy["mode"]
    lama_selected = inpaint_policy["selected"] == "big_lama"
    lama_required = bool(lama_selected and inpaint_policy["requested_model_required"])
    lama_ok = _module("simple_lama_inpainting")
    comfy_ok = True
    comfy_detail = "not required (qwen disabled or non-comfyui mode)"
    qwen_enabled = bool((cfg.get("qwen") or {}).get("enabled", True))
    if qwen_enabled and str((cfg.get("qwen") or {}).get("mode", "comfyui")).lower() == "comfyui":
        base = str(cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
        comfy_ok = _comfy_probe(base)
        comfy_detail = base if comfy_ok else f"{base} unreachable"
    checks.append(_check("Big-LaMa", lama_ok,
                          "optional; OpenCV fallback exists" if not lama_required
                          else "required by strict/active acceptance; OpenCV fallback degrades background plate",
                          required=lama_required))
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
            required=lama_required,
        ))
    footprint = _vram_footprint_check(cfg, flux_enabled)
    if footprint is not None:
        checks.append(footprint)
    checks.append(_check("Figma bridge", _module("requests"), "required only for plugin staging"))

    for item in checks:
        fix = _fix_for(item)
        if fix:
            item["fix"] = fix
    blockers = [item for item in checks if item["required"] and not item["ok"]]
    warnings = [item for item in checks if not item["required"] and not item["ok"]]
    return {
        "ok": not blockers,
        "device": device,
        "policy": {
            "require_active_models": bool(runtime.get("require_active_models", False)),
            "qwen_required": _required_qwen(cfg),
            "vectorization_required": vector_required,
            "inpaint_selected": inpaint_policy["selected"],
            "inpaint_strict_acceptance": inpaint_policy["strict_acceptance"],
            "inpaint_requested_model_required": inpaint_policy["requested_model_required"],
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
    parser.add_argument("--deep", action="store_true",
                        help="execute bounded real OCR/SAM/VLM/inpaint/vector/Figma staging probes")
    parser.add_argument("--deep-output", default="runs/runtime-smoke")
    parser.add_argument("--probe-timeout", type=float, default=120)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    result = inspect(cfg, Path(__file__).resolve().parent)
    if args.deep and result.get("ok"):
        from runtime_smoke import run_all
        result["runtime_smoke"] = run_all(cfg, args.deep_output, timeout_s=args.probe_timeout)
        result["ok"] = bool(result["runtime_smoke"].get("ok"))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("READY" if result["ok"] else "NOT READY")
        for item in result["checks"]:
            state = "OK" if item["ok"] else ("BLOCK" if item["required"] else "WARN")
            print(f"{state:5} {item['name']}: {item['detail']}")
            if not item["ok"] and item.get("fix"):
                print(f"      FIX: {item['fix']}")
        if args.deep:
            for item in (result.get("runtime_smoke") or {}).get("checks", []):
                print(f"{'OK' if item.get('ok') else 'BLOCK':5} runtime:{item['name']}: {item.get('detail')}")
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
