"""Shared QA acceptance thresholds for pixel_diff, repair, and run_pipeline."""
from __future__ import annotations

from typing import Optional

DEFAULT_VISUAL_PASS_SSIM = 0.9


def _archetype_thresholds(cfg: Optional[dict]) -> dict:
    return dict((cfg or {}).get("qa", {}).get("archetype_thresholds") or {})


def visual_pass_ssim(cfg: Optional[dict] = None) -> float:
    """Return QA SSIM gate from archetype preset, then cfg.qa, then default."""
    archetype = _archetype_thresholds(cfg)
    if archetype.get("visual_pass_ssim_min") is not None:
        return float(archetype["visual_pass_ssim_min"])
    qa = (cfg or {}).get("qa") or {}
    if qa.get("visual_pass_ssim") is not None:
        return float(qa["visual_pass_ssim"])
    return DEFAULT_VISUAL_PASS_SSIM


def pixel_diff_thresholds(cfg: Optional[dict] = None) -> dict:
    """Threshold overrides for :func:`src.pixel_diff.compare`."""
    opts = {"local_ssim_min": visual_pass_ssim(cfg)}
    archetype = _archetype_thresholds(cfg)
    if archetype.get("edge_f1_min") is not None:
        opts["edge_f1_min"] = float(archetype["edge_f1_min"])
    if archetype.get("editable_text_recall_min") is not None:
        opts["editable_text_recall_min"] = float(archetype["editable_text_recall_min"])
    return opts
