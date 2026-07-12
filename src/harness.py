"""harness.py — execute repair.assess suggestions by resuming the pipeline.

Maps each (stage, action) repair to a ``--resume`` stage plus optional config patches,
then re-runs ``run_pipeline.run_one`` until QA passes or ``max_iterations`` is reached.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Optional

# Repair modules use logical stage names; the orchestrator uses STAGES in run_pipeline.
STAGE_ALIASES = {
    "text-analysis": "text",
    "inpaint": "reconstruct",
    "vectorize": "reconstruct",
    "build": "design",
    "sam3": "sam",
}

# Actions the harness can drive without human review.
ACTIONABLE = {
    ("ocr", "rerun"),
    ("text-analysis", "restore-editable-text"),
    ("text-analysis", "refit-colors-effects"),
    ("text-analysis", "resolve-fonts"),
    ("qwen", "retry"),
    ("inpaint", "rebuild-clean-plate"),
    ("reconstruct", "inspect-worst-regions"),
    ("reconstruct", "restage-assets"),
    ("layout", "refit-geometry"),
    ("design", "restore-native-nodes"),
    ("figma", "fix-compiler-report"),
    ("merge", "dedup"),
    ("merge", "enforce-single-owner"),
    ("vectorize", "raster-fallback"),
}


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge *patch* into a copy of *base*."""
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _pipeline_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def config_patches_for(repair: dict) -> dict:
    """Translate repair params into config overrides the pipeline already understands."""
    stage = repair.get("stage")
    action = repair.get("action")
    params = dict(repair.get("params") or {})
    patches: dict[str, Any] = {}

    if stage == "ocr" and action == "rerun":
        ocr_patch: dict[str, Any] = {}
        if params.get("upscale"):
            ocr_patch["retry_2x"] = {"enabled": True}
        challengers = params.get("challengers")
        if challengers:
            ocr_patch["challengers"] = list(challengers)
        if ocr_patch:
            patches["ocr"] = ocr_patch

    elif stage == "qwen" and action == "retry":
        qwen_patch: dict[str, Any] = {"enabled": True}
        if "layers" in params:
            qwen_patch["layers"] = params["layers"]
        patches["qwen"] = qwen_patch

    elif stage == "merge" and action == "dedup":
        if params.get("raise_dedup_iou"):
            patches["merge"] = {"dedup_iou": 0.72}
            patches["reconstruct"] = {"dedup_iou": 0.90}

    elif stage == "text-analysis" and action == "resolve-fonts":
        patches["text_analysis"] = {"font_matching": {"enabled": True}}

    elif stage == "reconstruct" and action == "inspect-worst-regions":
        regions = params.get("regions") or []
        if regions:
            patches["reconstruct"] = {"focus_regions": regions[:4]}

    target_id = repair.get("target_id")
    if target_id:
        patches.setdefault("harness", {})["target_id"] = target_id

    return patches


def resume_stage_for(repair: dict) -> Optional[str]:
    """Map a repair record to a pipeline resume stage."""
    stage = repair.get("stage")
    action = repair.get("action")
    if not stage or not action:
        return None
    if (stage, action) not in ACTIONABLE:
        return None

    if stage == "ocr":
        return "ocr"
    if stage == "text-analysis":
        return "text"
    if stage == "qwen":
        return "qwen"
    if stage in ("inpaint", "vectorize", "reconstruct"):
        return "reconstruct"
    if stage == "layout":
        return "layout"
    if stage in ("design", "build"):
        return "design"
    if stage == "figma":
        return "figma"
    if stage == "merge":
        return "merge"
    return _pipeline_stage(stage)


def is_actionable(repair: dict) -> bool:
    stage = repair.get("stage")
    action = repair.get("action")
    return bool(stage and action and (stage, action) in ACTIONABLE and resume_stage_for(repair))


def recommended_resume(repairs: list) -> Optional[dict]:
    """Pick the highest-priority actionable repair and expose resume metadata."""
    for repair in repairs or []:
        if not is_actionable(repair):
            continue
        resume = resume_stage_for(repair)
        if not resume:
            continue
        return {
            "stage": repair.get("stage"),
            "action": repair.get("action"),
            "resume": resume,
            "target_id": repair.get("target_id"),
            "reason": repair.get("reason"),
            "severity": repair.get("severity"),
            "patches": config_patches_for(repair),
        }
    return None


def _load_json(path: str, fallback):
    if not path or not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return fallback


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)


def load_repairs(run_dir: str, cfg: Optional[dict] = None) -> list:
    """Read repairs from repairs.json, qa.json, or re-run repair.assess."""
    run_dir = os.path.abspath(run_dir)
    repairs_path = os.path.join(run_dir, "repairs.json")
    repairs = _load_json(repairs_path, None)
    if isinstance(repairs, list) and repairs:
        return repairs

    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    if isinstance(qa.get("repairs"), list) and qa["repairs"]:
        return qa["repairs"]

    if cfg is None:
        return []

    try:
        from src import repair as repair_mod
        from src.schema import load as schema_load
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src import repair as repair_mod
        from src.schema import load as schema_load

    design = schema_load(os.path.join(run_dir, "design.json")) if os.path.exists(os.path.join(run_dir, "design.json")) else {}
    ocr = schema_load(os.path.join(run_dir, "ocr.json")) if os.path.exists(os.path.join(run_dir, "ocr.json")) else {}
    assess_cfg = copy.deepcopy(cfg)
    assess_cfg["run_dir"] = run_dir
    return repair_mod.assess(design, qa, ocr, assess_cfg)


def _input_path(run_dir: str) -> Optional[str]:
    report = _load_json(os.path.join(run_dir, "runtime_report.json"), {})
    if report.get("input"):
        return str(report["input"])
    manifest = _load_json(os.path.join(run_dir, "input_manifest.json"), {})
    if manifest.get("source_path"):
        return str(manifest["source_path"])
    normalized = os.path.join(run_dir, "normalized.png")
    return normalized if os.path.exists(normalized) else None


def execute_repairs(
    run_dir: str,
    cfg: Optional[dict] = None,
    max_iterations: int = 2,
    *,
    run_one: Optional[Callable[..., dict]] = None,
) -> dict:
    """Apply actionable repairs by resuming the pipeline from mapped stages."""
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    if run_one is None:
        import run_pipeline
        run_one = run_pipeline.run_one

    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    if qa.get("ok"):
        return {
            "run_dir": run_dir,
            "iterations": 0,
            "qa_ok": True,
            "stopped": "already_ok",
            "attempts": [],
        }

    input_path = _input_path(run_dir)
    if not input_path:
        return {
            "run_dir": run_dir,
            "iterations": 0,
            "qa_ok": False,
            "stopped": "missing_input",
            "error": "could not resolve input image for repair rerun",
            "attempts": [],
        }

    attempts = []
    working_cfg = copy.deepcopy(cfg)
    working_cfg.setdefault("runtime", {})["auto_repair"] = False

    for iteration in range(1, max(0, int(max_iterations)) + 1):
        repairs = load_repairs(run_dir, working_cfg)
        choice = recommended_resume(repairs)
        if not choice:
            return {
                "run_dir": run_dir,
                "iterations": iteration - 1,
                "qa_ok": bool(_load_json(os.path.join(run_dir, "qa.json"), {}).get("ok")),
                "stopped": "no_actionable_repairs",
                "attempts": attempts,
            }

        resume = choice["resume"]
        patches = choice.get("patches") or {}
        iter_cfg = deep_merge(working_cfg, patches)
        iter_cfg["run_dir"] = run_dir
        iter_cfg.setdefault("runtime", {})["auto_repair"] = False

        result = run_one(input_path, run_dir, iter_cfg, resume)
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        attempt = {
            "iteration": iteration,
            "resume": resume,
            "repair": {
                "stage": choice.get("stage"),
                "action": choice.get("action"),
                "reason": choice.get("reason"),
                "target_id": choice.get("target_id"),
            },
            "patches": patches,
            "pipeline_ok": bool(result.get("ok")),
            "qa_ok": bool(qa.get("ok")),
        }
        attempts.append(attempt)
        working_cfg = iter_cfg

        if qa.get("ok"):
            summary = {
                "run_dir": run_dir,
                "iterations": iteration,
                "qa_ok": True,
                "stopped": "qa_ok",
                "attempts": attempts,
            }
            _write_json(os.path.join(run_dir, "harness.json"), summary)
            return summary

    final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    summary = {
        "run_dir": run_dir,
        "iterations": len(attempts),
        "qa_ok": bool(final_qa.get("ok")),
        "stopped": "max_iterations",
        "attempts": attempts,
    }
    _write_json(os.path.join(run_dir, "harness.json"), summary)
    return summary
