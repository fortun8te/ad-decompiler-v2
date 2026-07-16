"""VRAM management hooks for CUDA pipelines (e.g. RTX 5080 / 16 GB).

Heavy stages (OCR, SAM, inpaint) each cache GPU models.  ``stage_boundary`` unloads
the prior stage's caches between transitions so the next stage has headroom.  It also
records free/used telemetry per boundary and, before a *real* ComfyUI Flux inpaint, can
(config-gated) evict the persistent LM Studio VLM and pick a Flux GGUF quant that fits
the free VRAM measured after eviction.

With regional routing (``inpaint.regional.enabled`` + ``force_flux: false``), most holes
go analytic / Big-LaMa and never touch Flux.  Eager VLM unload+reload then costs ~8–12 s
for nothing.  ``lazy_flux_prep: true`` (default) defers eviction/quant until the first
region actually routes to ``flux-comfy`` via ``ensure_flux_vram``.

16 GB cannot hold gemma-4-e4b (~6.3 GB, resident in LM Studio) + SAM3 (~3 GB) + Flux Fill
Q6 GGUF (9.2 GB) + t5xxl fp8 (4.5 GB) at once.  The knobs live under ``runtime.vram``:

    runtime:
      vram:
        empty_cache_between_stages: true   # gc + torch.cuda.empty_cache() at boundaries
        evict_vlm_for_inpaint: true        # `lms unload` the VLM before Flux inpaint
        reload_vlm_after_inpaint: true      # `lms load` it back once inpaint is done
        lazy_flux_prep: true               # defer unload/quant until a Flux region runs

Every hook is best-effort and never raises: a missing ``lms`` CLI or ``nvidia-smi`` only
means the corresponding optimisation is skipped, recorded honestly in telemetry.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
from typing import Callable, Optional

_MiB = 1024 * 1024

# Per-run telemetry accumulator.  Reset at the start of each pipeline run so a long-lived
# bridge process does not concatenate boundaries across images.
_TELEMETRY: list[dict] = []

# Boundaries that immediately precede the heavy ComfyUI Flux inpaint.
# ``peel`` is included so SAM is unloaded and Flux prep can run for large photo holes
# (peel previously banned Flux because SAM was still resident → ~25 min/call).
_INPAINT_BOUNDARIES = {"reconstruct", "inpaint", "peel"}

# Idempotency for lazy Flux prep within a single pipeline run.
_FLUX_PREP_DONE: bool = False


def reset_telemetry() -> None:
    """Clear accumulated per-boundary VRAM telemetry (call once per run)."""
    global _FLUX_PREP_DONE
    _TELEMETRY.clear()
    _FLUX_PREP_DONE = False


def telemetry() -> list[dict]:
    """Return a copy of the per-boundary VRAM telemetry recorded so far."""
    return list(_TELEMETRY)


def _emit(log_fn: Optional[Callable[[str], None]], message: str) -> None:
    if log_fn is not None:
        try:
            log_fn(message)
        except Exception:
            pass


def optional_torch_cuda_empty_cache() -> None:
    """Collect Python garbage, then release cached CUDA blocks when available.

    ``gc.collect()`` is load-bearing: the OCR/SAM caches hold torch modules with
    reference cycles, so clearing the cache dict alone does not drop their tensors until
    the collector runs.  Without it ``empty_cache`` reclaims nothing and the next stage
    inherits the previous stage's resident weights.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_memory_bytes() -> Optional[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except Exception:
        pass
    return None


def cuda_mem_info() -> Optional[dict]:
    """Torch-visible CUDA memory in MiB (this process only)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "free_mib": round(free / _MiB, 1),
            "total_mib": round(total / _MiB, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / _MiB, 1),
            "allocated_mib": round(torch.cuda.memory_allocated() / _MiB, 1),
        }
    except Exception:
        return None


def nvidia_smi_mem() -> Optional[dict]:
    """Whole-GPU memory in MiB via nvidia-smi.

    Unlike torch this sees *all* processes (LM Studio, ComfyUI), which is what actually
    determines whether the next model fits.  Returns ``None`` when nvidia-smi is absent.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        total, used, free = (int(part.strip()) for part in first.split(","))
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    except Exception:
        return None


def free_vram_mib() -> Optional[float]:
    """Best whole-GPU free estimate: prefer nvidia-smi (all processes), else torch."""
    smi = nvidia_smi_mem()
    if smi is not None:
        return float(smi["free_mib"])
    cuda = cuda_mem_info()
    if cuda is not None:
        return float(cuda["free_mib"])
    return None


def _snapshot() -> dict:
    """Best-effort free/used snapshot for telemetry."""
    snap: dict = {}
    smi = nvidia_smi_mem()
    if smi is not None:
        snap["gpu"] = smi
    cuda = cuda_mem_info()
    if cuda is not None:
        snap["torch"] = cuda
    return snap


def log_vram(label: str, log_fn: Optional[Callable[[str], None]] = None) -> Optional[int]:
    """Log allocated CUDA memory (MiB) via ``log_fn``; return bytes when available."""
    allocated = _cuda_memory_bytes()
    if allocated is None or log_fn is None:
        return allocated
    mib = allocated / (1024 * 1024)
    log_fn(f"vram[{label}] allocated={mib:.1f}MiB")
    return allocated


def unload_ocr_engines() -> None:
    """Drop cached OCR backends so their GPU weights can be reclaimed."""
    from src import ocr

    ocr.clear_engine_caches()


def unload_sam_backend() -> None:
    """Drop cached SAM3 backends so their GPU weights can be reclaimed."""
    from src import sam3_detect

    sam3_detect.unload_backend()


# ── LM Studio VLM eviction ──────────────────────────────────────────────────────────
_COMFY_FREED_AT: float = 0.0


def free_comfy_vram(cfg: Optional[dict] = None, *,
                    log_fn: Optional[Callable[[str], None]] = None,
                    min_used_mib: float = 11000.0,
                    throttle_s: float = 30.0) -> bool:
    """Ask ComfyUI to unload its resident models (Flux stays loaded after an inpaint).

    A resident Flux Fill (~6.4 GB + T5) alongside the LM Studio VLM (~6.3 GB) exhausts a
    16 GB card, and a starved llama-server then times out EVERY VLM call at the full
    timeout ceiling — measured live 2026-07-16: fixture 021 spent 8+ minutes in
    ocr-judge/font-judge purely because Flux from the previous fixture's inpaint was
    still resident.  Called when entering VLM-heavy boundaries; only fires when the
    ComfyUI queue is idle (never interrupts a running job), GPU usage is actually high,
    and not more than once per ``throttle_s``.  Best-effort, never raises.
    """
    global _COMFY_FREED_AT
    import json as _json
    import time as _time
    import urllib.request as _rq
    try:
        if _time.monotonic() - _COMFY_FREED_AT < throttle_s:
            return False
        mem = nvidia_smi_mem()
        if mem and float(mem.get("used_mib") or 0) < min_used_mib:
            return False
        icfg = ((cfg or {}).get("inpaint") or {}).get("comfy") or {}
        base = str(icfg.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
        with _rq.urlopen(_rq.Request(base + "/queue"), timeout=2) as resp:
            queue = _json.loads(resp.read().decode("utf-8", "replace"))
        if (queue.get("queue_running") or []) or (queue.get("queue_pending") or []):
            return False
        body = _json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = _rq.Request(base + "/free", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        with _rq.urlopen(req, timeout=5):
            pass
        _COMFY_FREED_AT = _time.monotonic()
        _emit(log_fn, "vram: freed resident ComfyUI models (Flux) before VLM stage")
        return True
    except Exception:
        return False


def _lms_path(cfg: Optional[dict] = None) -> Optional[str]:
    """Locate the LM Studio CLI (`lms`) on PATH or at its default install location."""
    override = str(((cfg or {}).get("runtime") or {}).get("vram", {}).get("lms_path") or "").strip()
    if override and os.path.isfile(override):
        return override
    found = shutil.which("lms")
    if found:
        return found
    default = os.path.expanduser(os.path.join("~", ".lmstudio", "bin", "lms"))
    for candidate in (default, default + ".exe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _loaded_vlm_instances(cfg: Optional[dict], model: str) -> Optional[list]:
    """Instance identifiers currently loaded for ``model`` (exact or ``model:N`` suffix),
    via the LM Studio OpenAI endpoint. None when the server can't be queried."""
    base_url = str(((cfg or {}).get("vlm") or {}).get("base_url") or "").strip().rstrip("/")
    if not base_url or not model:
        return None
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=3) as resp:
            data = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    ids = [str(entry.get("id") or "") for entry in (data.get("data") or [])]
    return [i for i in ids if i == model or i.startswith(model + ":")]


def evict_vlm(cfg: Optional[dict] = None, *, log_fn: Optional[Callable[[str], None]] = None) -> bool:
    """`lms unload` the configured VLM so the GPU has room for Flux Fill.  Best-effort."""
    vlm = (cfg or {}).get("vlm") or {}
    model = str(vlm.get("model") or "").strip()
    tool = _lms_path(cfg)
    if not tool:
        _emit(log_fn, "vram: lms CLI not found; cannot evict VLM")
        return False
    # Unload by actual instance identifier: a reload cycle can leave the instance
    # registered as "model:2", which `lms unload <model>` does not match.
    instances = _loaded_vlm_instances(cfg, model) if model else None
    if instances:
        ok_all = True
        for ident in instances:
            try:
                result = subprocess.run([tool, "unload", ident], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=30)
                ok_all = ok_all and result.returncode == 0
            except Exception as exc:  # pragma: no cover - subprocess/env specific
                _emit(log_fn, f"vram: lms unload error ({exc})")
                return False
        _emit(log_fn, f"vram: lms unload {','.join(instances)} {'ok' if ok_all else 'noop/failed'}")
        return ok_all
    args = [tool, "unload", model] if model else [tool, "unload", "--all"]
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
    except Exception as exc:  # pragma: no cover - subprocess/env specific
        _emit(log_fn, f"vram: lms unload error ({exc})")
        return False
    ok = result.returncode == 0
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    _emit(log_fn, f"vram: lms unload {model or '--all'} {'ok' if ok else 'noop/failed'}"
                  + (f" ({detail[-1][:100]})" if detail else ""))
    return ok


def restore_vlm(cfg: Optional[dict] = None, run_dir: Optional[str] = None, *,
                log_fn: Optional[Callable[[str], None]] = None) -> bool:
    """`lms load` the VLM back after heavy inpaint, when ``reload_vlm_after_inpaint``."""
    del run_dir
    opts = _vram_cfg(cfg)
    if not opts["reload_vlm_after_inpaint"]:
        return False
    vlm = (cfg or {}).get("vlm") or {}
    model = str(vlm.get("model") or "").strip()
    if not model or not _vlm_feature_enabled(cfg):
        return False
    tool = _lms_path(cfg)
    if not tool:
        _emit(log_fn, "vram: lms CLI not found; cannot reload VLM")
        return False
    # Idempotency guard: `lms load` on an already-loaded model registers a duplicate
    # "model:2" instance, which then fails the doctor/runtime identity check.
    already = _loaded_vlm_instances(cfg, model)
    if already:
        _emit(log_fn, f"vram: VLM already loaded ({already[0]}); skipping lms load")
        return True
    ttl = int(((cfg.get("runtime") or {}).get("vram") or {}).get("vlm_reload_ttl_s", 0) or 0)
    args = [tool, "load", model, "-y"]
    if ttl > 0:
        args += ["--ttl", str(ttl)]
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
    except Exception as exc:  # pragma: no cover - subprocess/env specific
        _emit(log_fn, f"vram: lms load error ({exc})")
        return False
    ok = result.returncode == 0
    _emit(log_fn, f"vram: lms load {model} {'ok' if ok else 'failed'}")
    return ok


# ── Flux GGUF quant selection by available VRAM ─────────────────────────────────────
_DEFAULT_QUANT_LADDER = {
    "high": "flux1-fill-dev-Q6_K.gguf",
    "mid": "flux1-fill-dev-Q5_K_S.gguf",
    "low": "flux1-fill-dev-Q4_K_S.gguf",
}


def select_flux_quant(cfg: Optional[dict] = None, *, free_mib: Optional[float] = None) -> Optional[str]:
    """Pick the Flux Fill GGUF that fits ``free_mib`` of whole-GPU VRAM.

    Returns ``None`` when ``inpaint.comfy.vram_adaptive_quant`` is off — the explicit
    ``inpaint.comfy.models.unet_gguf`` is then left untouched (explicit override wins).
    Q6_K needs the most headroom; Q5_K_S/Q4_K_S step down for a tighter budget.
    """
    comfy = ((cfg or {}).get("inpaint") or {}).get("comfy") or {}
    if not comfy.get("vram_adaptive_quant"):
        return None
    ladder = {**_DEFAULT_QUANT_LADDER, **(comfy.get("quant_ladder") or {})}
    thresholds = comfy.get("quant_vram_thresholds") or {}
    high_min = float(thresholds.get("high_min_free_mib", 10240))
    mid_min = float(thresholds.get("mid_min_free_mib", 7680))
    if free_mib is None:
        free_mib = free_vram_mib()
    if free_mib is None:
        # Unknown free VRAM: after a successful eviction we expect ample room, so assume
        # the high quant rather than needlessly degrading quality.
        return ladder.get("high")
    if free_mib >= high_min:
        return ladder.get("high")
    if free_mib >= mid_min:
        return ladder.get("mid")
    return ladder.get("low")


def _flux_inpaint_active(cfg: Optional[dict]) -> bool:
    """True when the run's inpaint backend is ComfyUI Flux Fill (mode flux, or auto+comfy)."""
    icfg = (cfg or {}).get("inpaint") or {}
    mode = str(icfg.get("mode", "auto")).lower()
    comfy = icfg.get("comfy") or {}
    return mode in ("flux", "flux-comfy", "flux_comfy") or (
        mode == "auto" and bool(comfy.get("enabled"))
    )


def _should_eager_flux_prep(cfg: Optional[dict], opts: Optional[dict] = None) -> bool:
    """Whether merge→reconstruct should unload the VLM / pick a Flux quant immediately.

    When regional routing can still send every hole to analytic/LaMa, eager prep wastes
    ~8–12 s of unload+reload.  Eager only when Flux is certain to run (force_flux, or
    Flux mode with regional disabled) or when ``lazy_flux_prep`` is off.
    """
    if not _flux_inpaint_active(cfg):
        return False
    opts = opts or _vram_cfg(cfg)
    if not opts.get("lazy_flux_prep", True):
        return True
    regional = ((cfg or {}).get("inpaint") or {}).get("regional") or {}
    if bool(regional.get("force_flux")):
        return True
    mode = str(((cfg or {}).get("inpaint") or {}).get("mode", "auto")).lower()
    if mode in ("flux", "flux-comfy", "flux_comfy") and not bool(regional.get("enabled", True)):
        return True
    return False


def prepare_inpaint_vram(cfg: Optional[dict], opts: Optional[dict] = None, *,
                         log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """Evict the VLM and select a fitting Flux quant just before the heavy inpaint.

    Mutates ``cfg['inpaint']['comfy']['models']['unet_gguf']`` in place when adaptive
    quant selection is enabled, using the free VRAM measured *after* eviction so the
    decision reflects the real budget the ComfyUI worker will see.  Returns a record for
    telemetry.  No-op (record with all-false) when Flux is not the active backend.
    """
    opts = opts or _vram_cfg(cfg)
    record: dict = {
        "vlm_evicted": False, "flux_quant": None, "flux_quant_prev": None,
        "free_mib_before": None, "free_mib_after": None,
    }
    if not _flux_inpaint_active(cfg) or opts["device"] != "cuda":
        return record
    record["free_mib_before"] = free_vram_mib()
    if opts["evict_vlm_for_inpaint"] and _vlm_feature_enabled(cfg):
        record["vlm_evicted"] = evict_vlm(cfg, log_fn=log_fn)
        optional_torch_cuda_empty_cache()
    free_after = free_vram_mib()
    record["free_mib_after"] = free_after
    quant = select_flux_quant(cfg, free_mib=free_after)
    if quant:
        comfy = cfg.setdefault("inpaint", {}).setdefault("comfy", {})
        models = comfy.get("models")
        if not isinstance(models, dict):
            models = {}
            comfy["models"] = models
        record["flux_quant_prev"] = models.get("unet_gguf")
        models["unet_gguf"] = quant
        record["flux_quant"] = quant
        if record["flux_quant_prev"] != quant:
            _emit(log_fn, f"vram: flux quant -> {quant} (free~{free_after}MiB)")
    return record


def ensure_flux_vram(cfg: Optional[dict], opts: Optional[dict] = None, *,
                     log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """Run Flux VRAM prep at most once per pipeline run (lazy regional path).

    Call immediately before the first region that routes to ``flux-comfy``.  Subsequent
    calls are no-ops so multi-region Flux fills do not re-evict the VLM.
    """
    global _FLUX_PREP_DONE
    if _FLUX_PREP_DONE:
        return {
            "vlm_evicted": False, "flux_quant": None, "flux_quant_prev": None,
            "free_mib_before": None, "free_mib_after": None,
            "already_prepared": True,
        }
    record = prepare_inpaint_vram(cfg, opts, log_fn=log_fn)
    _FLUX_PREP_DONE = True
    record["already_prepared"] = False
    return record


def _vlm_feature_enabled(cfg: Optional[dict]) -> bool:
    """Mirror doctor's VLM-feature gate: any enabled VLM sub-feature keeps gemma resident."""
    vlm = (cfg or {}).get("vlm") or {}
    if vlm.get("enabled"):
        return True
    for key in ("segment_filter", "font_judge", "scene_text", "ocr_judge", "element_propose", "anomaly"):
        if (vlm.get(key) or {}).get("enabled"):
            return True
    return False


def _vram_cfg(cfg: Optional[dict]) -> dict:
    runtime = (cfg or {}).get("runtime") or {}
    vram = runtime.get("vram") or {}
    device = str((cfg or {}).get("device", "cpu")).lower()
    empty_cache = vram.get("empty_cache_between_stages")
    if empty_cache is None:
        empty_cache = device == "cuda"
    evict = bool(vram.get("evict_vlm_for_inpaint", False))
    lazy = vram.get("lazy_flux_prep")
    if lazy is None:
        lazy = True
    return {
        "unload_ocr_before_sam": bool(vram.get("unload_ocr_before_sam", True)),
        "unload_ocr_before_vlm": bool(vram.get("unload_ocr_before_vlm", vram.get("unload_ocr_before_sam", True))),
        "empty_cache_between_stages": bool(empty_cache),
        "evict_vlm_for_inpaint": evict,
        "reload_vlm_after_inpaint": bool(vram.get("reload_vlm_after_inpaint", evict)),
        "lazy_flux_prep": bool(lazy),
        "device": device,
    }


def stage_boundary(
    from_stage: str,
    to_stage: str,
    cfg: Optional[dict],
    run_dir: Optional[str] = None,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Free GPU memory between heavy pipeline stages and record VRAM telemetry."""
    global _FLUX_PREP_DONE
    del run_dir  # reserved for future per-run diagnostics
    opts = _vram_cfg(cfg)
    label = f"{from_stage}->{to_stage}"
    before = _snapshot()
    log_vram(f"before-{label}", log_fn)

    if to_stage == "sam" and opts["unload_ocr_before_sam"]:
        unload_ocr_engines()
    if to_stage in {"vlm-ocr-judge", "vlm-proofread", "vlm-font-judge", "vlm-scene-text"} and opts["unload_ocr_before_vlm"]:
        unload_ocr_engines()
    if to_stage in {"vlm-ocr-judge", "vlm-proofread", "vlm-font-judge", "vlm-scene-text",
                    "vlm-element-propose", "vlm-grouping"}:
        # A Flux model left resident by the PREVIOUS fixture's inpaint starves the
        # llama-server the VLM stages depend on (every call times out at the ceiling).
        free_comfy_vram(cfg, log_fn=log_fn)
    if to_stage in {"reconstruct", "inpaint", "vlm-segment-filter", "peel"}:
        unload_sam_backend()

    inpaint_prep: dict = {}
    if to_stage in _INPAINT_BOUNDARIES:
        if _should_eager_flux_prep(cfg, opts):
            inpaint_prep = prepare_inpaint_vram(cfg, opts, log_fn=log_fn)
            _FLUX_PREP_DONE = True
        elif _flux_inpaint_active(cfg) and opts.get("lazy_flux_prep", True):
            _emit(log_fn, "vram: deferring Flux prep until a region routes to flux-comfy")
            inpaint_prep = {"deferred": True, "vlm_evicted": False, "flux_quant": None}

    if opts["empty_cache_between_stages"]:
        optional_torch_cuda_empty_cache()

    after = _snapshot()
    log_vram(f"after-{label}", log_fn)

    if before or after or inpaint_prep.get("vlm_evicted") or inpaint_prep.get("flux_quant") or inpaint_prep.get("deferred"):
        entry = {"boundary": label, "before": before, "after": after}
        if inpaint_prep:
            entry["inpaint_prep"] = inpaint_prep
        _TELEMETRY.append(entry)
