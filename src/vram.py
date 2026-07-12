"""VRAM management hooks for CUDA pipelines (e.g. RTX 5080 / 16 GB).

Heavy stages (OCR, SAM, inpaint) each cache GPU models.  ``stage_boundary`` unloads
the prior stage's caches between transitions so the next stage has headroom.
"""
from __future__ import annotations

from typing import Callable, Optional


def optional_torch_cuda_empty_cache() -> None:
    """Call ``torch.cuda.empty_cache()`` when CUDA is available."""
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


def _vram_cfg(cfg: Optional[dict]) -> dict:
    runtime = (cfg or {}).get("runtime") or {}
    vram = runtime.get("vram") or {}
    device = str((cfg or {}).get("device", "cpu")).lower()
    empty_cache = vram.get("empty_cache_between_stages")
    if empty_cache is None:
        empty_cache = device == "cuda"
    return {
        "unload_ocr_before_sam": bool(vram.get("unload_ocr_before_sam", True)),
        "empty_cache_between_stages": bool(empty_cache),
    }


def stage_boundary(
    from_stage: str,
    to_stage: str,
    cfg: Optional[dict],
    run_dir: Optional[str] = None,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Free GPU memory between heavy pipeline stages."""
    del run_dir  # reserved for future per-run diagnostics
    opts = _vram_cfg(cfg)
    label = f"{from_stage}->{to_stage}"
    log_vram(f"before-{label}", log_fn)

    if to_stage == "sam" and opts["unload_ocr_before_sam"]:
        unload_ocr_engines()
    if to_stage in {"reconstruct", "inpaint"}:
        unload_sam_backend()

    if opts["empty_cache_between_stages"]:
        optional_torch_cuda_empty_cache()

    log_vram(f"after-{label}", log_fn)
