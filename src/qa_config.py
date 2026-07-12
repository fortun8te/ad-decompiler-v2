"""Shared QA acceptance thresholds for pixel_diff, repair, and run_pipeline."""
from __future__ import annotations

from typing import Optional

DEFAULT_VISUAL_PASS_SSIM = 0.9


def visual_pass_ssim(cfg: Optional[dict] = None) -> float:
    """Return ``cfg.qa.visual_pass_ssim`` (default 0.9)."""
    cfg = cfg or {}
    qa = cfg.get("qa") or {}
    value = qa.get("visual_pass_ssim", DEFAULT_VISUAL_PASS_SSIM)
    return float(value)


def pixel_diff_thresholds(cfg: Optional[dict] = None) -> dict:
    """Threshold overrides for :func:`src.pixel_diff.compare`."""
    return {"local_ssim_min": visual_pass_ssim(cfg)}
