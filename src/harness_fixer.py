"""harness_fixer.py — expanded automatic config fixes driven by harness_critic output.

Applies deterministic config patches (and optional figma re-staging) beyond the repair
records that ``harness.execute_repairs`` already understands.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

from src.harness import deep_merge

DEFAULT_INBOX = os.environ.get("FIGMA_INBOX", os.path.expanduser("~/figma-inbox"))

FIX_DISPATCH = {
    "staging": "fix_staging",
    "fix_staging": "fix_staging",
    "restage-inbox": "fix_staging",
    "ocr_stack": "fix_ocr_stack",
    "fix_ocr_stack": "fix_ocr_stack",
    "boost-ocr-stack": "fix_ocr_stack",
    "vlm_stack": "fix_vlm_stack",
    "fix_vlm_stack": "fix_vlm_stack",
    "boost-vlm-stack": "fix_vlm_stack",
    "inpaint": "fix_inpaint",
    "fix_inpaint": "fix_inpaint",
    "force-lama-inpaint": "fix_inpaint",
    "layout": "fix_layout",
    "fix_layout": "fix_layout",
    "tighten-containers": "fix_layout",
}

ELEMENT_CATEGORIES = frozenset({"sam", "element", "elements"})
TEXT_CATEGORIES = frozenset({"text", "ocr"})


def _inbox_path(cfg: dict | None) -> str:
    figma = (cfg or {}).get("figma") or {}
    return os.path.abspath(os.path.expanduser(figma.get("inbox") or DEFAULT_INBOX))


def staging_needs_fix(run_dir: str, cfg: dict | None = None) -> bool:
    """Return True when the Figma plugin inbox is missing or does not match *run_dir*."""
    run_dir = os.path.abspath(run_dir)
    inbox = _inbox_path(cfg)
    manifest_path = os.path.join(inbox, "inbox.json")
    if not os.path.exists(manifest_path):
        return True
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return True

    manifest_run = manifest.get("run_dir")
    if manifest_run and os.path.abspath(str(manifest_run)) != run_dir:
        return True

    staged_dir = manifest.get("staged_dir")
    if not staged_dir:
        return True
    staged_root = os.path.join(inbox, staged_dir)
    if not os.path.isdir(staged_root):
        return True
    if not os.path.exists(os.path.join(staged_root, "design.json")):
        return True
    return False


def fix_staging(run_dir: str, cfg: dict | None = None) -> tuple[dict, list[str]]:
    """Re-run figma_import when the plugin inbox is missing or stale."""
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    if not staging_needs_fix(run_dir, cfg):
        return cfg, []

    design_path = os.path.join(run_dir, "design.json")
    if not os.path.exists(design_path):
        return cfg, []

    from src.figma_import import import_design

    figma_cfg = cfg.setdefault("figma", {})
    figma_cfg.setdefault("enabled", True)
    figma_cfg.setdefault("mode", "plugin")
    result = import_design(design_path, run_dir, cfg)
    if not result.get("ok"):
        return cfg, []

    return cfg, ["restage-inbox"]


def fix_ocr_stack(cfg: dict | None = None, issue: dict | None = None) -> tuple[dict, list[str]]:
    """Enable VLM OCR judge and add easyocr as a challenger engine."""
    del issue  # reserved for future per-issue tuning
    cfg = copy.deepcopy(cfg or {})
    applied: list[str] = []

    vlm = cfg.setdefault("vlm", {})
    if not vlm.get("enabled"):
        vlm["enabled"] = True
        applied.append("vlm.enabled")

    judge = vlm.setdefault("ocr_judge", {})
    if not judge.get("enabled"):
        judge["enabled"] = True
        applied.append("vlm.ocr_judge")

    ocr = cfg.setdefault("ocr", {})
    challengers = list(ocr.get("challengers") or [])
    if "easyocr" not in challengers:
        challengers.append("easyocr")
        ocr["challengers"] = challengers
        applied.append("ocr.easyocr_challenger")

    if applied:
        applied.insert(0, "boost-ocr-stack")
    return cfg, applied


def fix_vlm_stack(cfg: dict | None = None, issue: dict | None = None) -> tuple[dict, list[str]]:
    """Enable segment_filter for element issues and scene_text for text issues."""
    issue = issue or {}
    cfg = copy.deepcopy(cfg or {})
    applied: list[str] = []

    category = str(issue.get("category") or "").lower()
    enable_segment = category in ELEMENT_CATEGORIES or issue.get("kind") == "element"
    enable_scene = category in TEXT_CATEGORIES or issue.get("kind") == "text"
    if not enable_segment and not enable_scene:
        enable_segment = enable_scene = True

    vlm = cfg.setdefault("vlm", {})
    if not vlm.get("enabled"):
        vlm["enabled"] = True
        applied.append("vlm.enabled")

    if enable_segment:
        segment = vlm.setdefault("segment_filter", {})
        if not segment.get("enabled"):
            segment["enabled"] = True
            applied.append("vlm.segment_filter")

    if enable_scene:
        scene = vlm.setdefault("scene_text", {})
        if not scene.get("enabled"):
            scene["enabled"] = True
            applied.append("vlm.scene_text")

    if applied:
        applied.insert(0, "boost-vlm-stack")
    return cfg, applied


def fix_inpaint(cfg: dict | None = None) -> tuple[dict, list[str]]:
    """Force Big-LaMa and widen button removal masks."""
    cfg = copy.deepcopy(cfg or {})
    applied: list[str] = []
    inpaint = cfg.setdefault("inpaint", {})

    mode = str(inpaint.get("mode", "auto")).lower()
    if mode not in ("big-lama", "lama", "simple-lama"):
        inpaint["mode"] = "big-lama"
        inpaint["allow_fallback"] = False
        applied.append("inpaint.big-lama")

    mask_dilate = inpaint.setdefault("mask_dilate", {})
    button = int(mask_dilate.get("button", 4))
    target = max(6, button + 2)
    if button < target:
        mask_dilate["button"] = target
        applied.append("inpaint.button_mask_dilate")

    if applied:
        applied.insert(0, "force-lama-inpaint")
    return cfg, applied


def fix_layout(cfg: dict | None = None) -> tuple[dict, list[str]]:
    """Tighten container inference thresholds."""
    cfg = copy.deepcopy(cfg or {})
    applied: list[str] = []
    layout = cfg.setdefault("layout", {})
    current = float(layout.get("min_container_frac", 0.002))
    if current > 0.001:
        layout["min_container_frac"] = 0.001
        applied.append("layout.min_container_frac")

    if applied:
        applied.insert(0, "tighten-containers")
    return cfg, applied


def _issue_for_category(critic_output: dict, category: str) -> dict | None:
    for issue in critic_output.get("issues") or []:
        if str(issue.get("category") or "").lower() == category:
            return issue
    return None


def _collect_fix_ids(critic_output: dict) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(fix_id: str) -> None:
        key = str(fix_id or "").strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)

    for issue in critic_output.get("issues") or []:
        for fix_id in issue.get("suggested_fix_ids") or []:
            add(fix_id)
    for fix_id in critic_output.get("suggested_fix_ids") or []:
        add(fix_id)
    return ordered


def apply_fixer_round(
    run_dir: str,
    cfg: dict | None,
    critic_output: dict | None,
) -> tuple[dict, list[str]]:
    """Apply critic-suggested fixes and return patched config + fix ids applied."""
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    critic_output = critic_output or {}
    fixes_applied: list[str] = []

    for fix_id in _collect_fix_ids(critic_output):
        handler = FIX_DISPATCH.get(fix_id)
        if handler == "fix_staging":
            cfg, applied = fix_staging(run_dir, cfg)
        elif handler == "fix_ocr_stack":
            cfg, applied = fix_ocr_stack(cfg, _issue_for_category(critic_output, "ocr"))
        elif handler == "fix_vlm_stack":
            issue = (
                _issue_for_category(critic_output, "sam")
                or _issue_for_category(critic_output, "text")
                or {}
            )
            cfg, applied = fix_vlm_stack(cfg, issue)
        elif handler == "fix_inpaint":
            cfg, applied = fix_inpaint(cfg)
        elif handler == "fix_layout":
            cfg, applied = fix_layout(cfg)
        else:
            continue

        for item in applied:
            if item not in fixes_applied:
                fixes_applied.append(item)

    return cfg, fixes_applied


def repair_for_fix(fix_id: str, issue: dict | None = None) -> dict | None:
    """Map a fixer id to a harness-actionable repair record."""
    issue = issue or {}
    handler = FIX_DISPATCH.get(fix_id)
    if handler == "fix_staging":
        return {"stage": "figma", "action": "restage-inbox", "reason": "figma inbox missing or stale"}
    if handler == "fix_ocr_stack":
        return {
            "stage": "ocr",
            "action": "boost-stack",
            "reason": issue.get("detail") or "boost OCR stack",
            "severity": issue.get("severity") or "high",
        }
    if handler == "fix_vlm_stack":
        category = str(issue.get("category") or "").lower()
        focus = "text" if category in TEXT_CATEGORIES else "elements"
        return {
            "stage": "vlm",
            "action": "boost-stack",
            "reason": issue.get("detail") or "boost VLM stack",
            "params": {"focus": focus},
            "severity": issue.get("severity") or "medium",
        }
    if handler == "fix_inpaint":
        return {
            "stage": "inpaint",
            "action": "force-lama",
            "reason": issue.get("detail") or "force Big-LaMa inpaint",
            "severity": issue.get("severity") or "high",
        }
    if handler == "fix_layout":
        return {
            "stage": "layout",
            "action": "tighten-containers",
            "reason": issue.get("detail") or "tighten container inference",
            "severity": issue.get("severity") or "medium",
        }
    return None


def config_patches_for_fixer(repair: dict) -> dict:
    """Translate fixer repair records into config overrides."""
    stage = repair.get("stage")
    action = repair.get("action")
    params = dict(repair.get("params") or {})
    patches: dict[str, Any] = {}

    if stage == "figma" and action == "restage-inbox":
        patches["figma"] = {"enabled": True, "mode": "plugin"}

    elif stage == "ocr" and action == "boost-stack":
        _, applied = fix_ocr_stack({})
        if applied:
            patches = deep_merge(patches, {
                "vlm": {"enabled": True, "ocr_judge": {"enabled": True}},
                "ocr": {"challengers": ["easyocr"]},
            })

    elif stage == "vlm" and action == "boost-stack":
        focus = params.get("focus", "elements")
        issue = {"category": "text" if focus == "text" else "sam"}
        patched, _ = fix_vlm_stack({}, issue)
        if patched.get("vlm"):
            patches["vlm"] = patched["vlm"]

    elif stage == "inpaint" and action == "force-lama":
        patched, _ = fix_inpaint({})
        if patched.get("inpaint"):
            patches["inpaint"] = patched["inpaint"]

    elif stage == "layout" and action == "tighten-containers":
        patched, _ = fix_layout({})
        if patched.get("layout"):
            patches["layout"] = patched["layout"]

    return patches


__all__ = [
    "apply_fixer_round",
    "config_patches_for_fixer",
    "fix_inpaint",
    "fix_layout",
    "fix_ocr_stack",
    "fix_staging",
    "fix_vlm_stack",
    "repair_for_fix",
    "staging_needs_fix",
]
