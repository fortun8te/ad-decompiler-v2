"""harness.py — execute repair.assess suggestions by resuming the pipeline.

Maps each (stage, action) repair to a ``--resume`` stage plus optional config patches,
then re-runs ``run_pipeline.run_one`` until QA passes or ``max_iterations`` is reached.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
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
    ("ocr", "boost-stack"),
    ("text-analysis", "restore-editable-text"),
    ("text-analysis", "refit-colors-effects"),
    ("text-analysis", "resolve-fonts"),
    ("qwen", "retry"),
    ("vlm", "boost-stack"),
    ("inpaint", "rebuild-clean-plate"),
    ("inpaint", "force-lama"),
    ("reconstruct", "inspect-worst-regions"),
    ("reconstruct", "restage-assets"),
    ("layout", "refit-geometry"),
    ("layout", "tighten-containers"),
    ("design", "restore-native-nodes"),
    ("design", "rebuild-schema"),
    ("figma", "fix-compiler-report"),
    ("figma", "restage-inbox"),
    ("merge", "dedup"),
    ("merge", "enforce-single-owner"),
    ("vectorize", "raster-fallback"),
    ("sam3", "rerun-detection"),
    ("sam3", "revalidate-rejected"),
}


def _flag(value: Any, default: bool = False) -> bool:
    """Interpret persisted boolean flags without treating ``"false"`` as true."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on", "ready", "ok"}:
            return True
        if normalized in {"false", "no", "0", "off", "failed", "not_ready"}:
            return False
    if value is None:
        return default
    return value is True


def _qa_accepts(qa: Any, *, allow_summary: bool = False) -> bool:
    """Fail closed on missing, malformed, or contradictory QA summaries."""
    if not isinstance(qa, dict) or not _flag(qa.get("ok")):
        return False
    # Lightweight bridge/test summaries intentionally carry only the boolean
    # result.  They are not artifact QA and must be handled before requiring
    # artifact fields below; real qa.json still takes the strict path.
    if allow_summary and "hard_fails" not in qa and "structural" not in qa and not any(
        key in qa for key in ("ssim", "visual_score", "composite")
    ):
        return True
    hard = qa.get("hard_fails")
    structural = qa.get("structural") or {}
    if not isinstance(hard, list) or hard:
        return False
    structural_hard = structural.get("hard_fails") if isinstance(structural, dict) else None
    if structural_hard is not None and (not isinstance(structural_hard, list) or structural_hard):
        return False
    if any(key in qa for key in ("ssim", "visual_score", "composite")):
        return True
    return False


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
        patches["text_analysis"] = {
            "font_matching": {
                "enabled": True,
                "repair_pass": True,
                "max_fonts": int(params.get("max_fonts", 96)),
                "max_lines": int(params.get("max_lines", 24)),
                "top_k": int(params.get("top_k", 5)),
            }
        }
        if params.get("enable_vlm_font_judge") is not False:
            patches["vlm"] = {"font_judge": {"enabled": True}}

    elif stage == "inpaint" and action == "rebuild-clean-plate":
        patches["inpaint"] = {
            "mode": params.get("mode", "auto"),
            "allow_fallback": False,
        }

    elif stage == "layout" and action == "refit-geometry":
        layout_patch: dict[str, Any] = {}
        if params.get("tighten_containers"):
            layout_patch["min_container_frac"] = 0.001
        if "min_container_frac" in params:
            layout_patch["min_container_frac"] = params["min_container_frac"]
        if "max_container_frac" in params:
            layout_patch["max_container_frac"] = params["max_container_frac"]
        if layout_patch:
            patches["layout"] = layout_patch

    elif (stage, action) in {
        ("figma", "restage-inbox"),
        ("ocr", "boost-stack"),
        ("vlm", "boost-stack"),
        ("inpaint", "force-lama"),
        ("layout", "tighten-containers"),
    }:
        from src.harness_fixer import config_patches_for_fixer

        patches = deep_merge(patches, config_patches_for_fixer(repair))

    elif stage == "reconstruct" and action == "inspect-worst-regions":
        regions = params.get("regions") or []
        if regions:
            patches["reconstruct"] = {"focus_regions": regions[:4]}

    elif stage == "reconstruct" and action == "restage-assets":
        patches["reconstruct"] = {"restage_assets": True}

    elif stage == "design" and action == "restore-native-nodes":
        patches["design"] = {"restore_native_nodes": True}

    elif stage == "design" and action == "rebuild-schema":
        patches["design"] = {"rebuild_schema": True}

    elif stage == "figma" and action == "fix-compiler-report":
        patches["figma"] = {"enabled": True, "reimport": True}

    elif stage == "sam3" and action == "rerun-detection":
        sam3_patch: dict[str, Any] = {"enabled": True}
        if params.get("lower_confidence"):
            sam3_patch["confidence"] = float(params.get("confidence", 0.38))
            sam3_patch["box_refine_confidence"] = float(
                params.get("box_refine_confidence", 0.30)
            )
        patches["sam3"] = sam3_patch
        if params.get("enable_element_propose"):
            patches.setdefault("vlm", {})["element_propose"] = {
                "enabled": True,
                "lightweight_grid": bool(params.get("lightweight_grid", True)),
            }
        if params.get("disable_segment_filter"):
            patches.setdefault("vlm", {})["segment_filter"] = {"enabled": False}
        if params.get("reject_internal_holes"):
            # Recorded for SAM/mask judges that support alternate-matte selection. It also
            # forces a fresh detection pass instead of silently reusing a corrupt asset.
            sam3_patch["reject_internal_holes"] = True

    elif stage == "sam3" and action == "revalidate-rejected":
        patches["sam3"] = {
            "enabled": True,
            "confidence": float(params.get("confidence", 0.40)),
        }
        patches["vlm"] = {
            "segment_filter": {
                "enabled": not params.get("disable_segment_filter"),
                "reject_mode": params.get("reject_mode", "remove"),
            }
        }

    elif stage == "vectorize" and action == "raster-fallback":
        patches["vectorize"] = {"force_raster_fallback": True}

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
    if stage == "sam3":
        return "sam"
    if stage == "text-analysis":
        return "text"
    if stage == "qwen":
        return "qwen"
    if stage == "vlm" and action == "boost-stack":
        focus = (repair.get("params") or {}).get("focus", "elements")
        return "text" if focus == "text" else "elements"
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


def _artifact_fingerprint(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _invoke_run_one(run_one: Callable[..., dict], input_path: str, run_dir: str,
                    cfg: dict, resume: str) -> dict:
    """Support production and lightweight runners while preserving the resume stage."""
    try:
        parameters = inspect.signature(run_one).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "start_from" in parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        return run_one(input_path, run_dir, cfg, start_from=resume)
    return run_one(input_path, run_dir, cfg)


def _repair_id(repair: dict) -> tuple:
    return (repair.get("stage"), repair.get("action"), repair.get("target_id"))


def _save_harness_summary(run_dir: str, summary: dict) -> dict:
    _write_json(os.path.join(run_dir, "harness.json"), summary)
    return summary


def load_repairs(run_dir: str, cfg: Optional[dict] = None) -> list:
    """Read repairs from repairs.json, qa.json, or re-run repair.assess."""
    run_dir = os.path.abspath(run_dir)
    repairs_path = os.path.join(run_dir, "repairs.json")
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    qa_path = os.path.join(run_dir, "qa.json")
    repairs = _load_json(repairs_path, None)
    # A resumed pipeline can rewrite qa.json without rewriting repairs.json.  In that
    # case the QA-owned list is authoritative; otherwise the loop can repeat a stale
    # repair forever and never reach an available alternative.
    qa_is_newer = (
        os.path.exists(qa_path)
        and os.path.exists(repairs_path)
        and os.stat(qa_path).st_mtime_ns > os.stat(repairs_path).st_mtime_ns
    )
    if (
        qa_is_newer
        and isinstance(qa.get("repairs"), list)
        and (qa.get("repairs") or _qa_accepts(qa, allow_summary=True))
    ):
        return qa["repairs"]
    if isinstance(repairs, list) and repairs:
        return repairs
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


def load_qa(run_dir: str) -> dict:
    """Read qa.json when present."""
    return _load_json(os.path.join(os.path.abspath(run_dir), "qa.json"), {})


def harness_enabled(cfg: Optional[dict]) -> bool:
    """True when the failure-proof harness should run after QA/staging issues."""
    runtime = (cfg or {}).get("runtime") or {}
    harness = runtime.get("harness") or {}
    if "enabled" in harness:
        return _flag(harness["enabled"])
    return _flag(runtime.get("auto_repair"))


def harness_max_rounds(cfg: Optional[dict]) -> int:
    """Max repair rounds from config (default 3)."""
    runtime = (cfg or {}).get("runtime") or {}
    harness = runtime.get("harness") or {}
    if harness.get("max_rounds") is not None:
        return max(0, int(harness["max_rounds"]))
    if runtime.get("harness_max_iterations") is not None:
        return max(0, int(runtime["harness_max_iterations"]))
    if runtime.get("auto_repair_max_iterations") is not None:
        return max(0, int(runtime["auto_repair_max_iterations"]))
    return 3


def harness_should_repair(
    pipeline_result: Optional[dict],
    *,
    qa: Optional[dict] = None,
    staging: Optional[dict] = None,
) -> tuple[bool, str]:
    """Decide whether to run repairs before reporting a job done to the plugin."""
    result = pipeline_result or {}
    if not result.get("ok"):
        return False, "pipeline_failed"

    if staging is not None and not _flag(staging.get("staged")):
        return True, "staging_failed"

    if qa is not None and not _qa_accepts(qa, allow_summary=True):
        return True, "qa_failed"

    if "runtime_ok" in result and not _flag(result.get("runtime_ok")):
        return True, "runtime_degraded"

    # Bridge/pipeline callers may already have summarized QA and omit qa.json.
    # Do not report a run as ready when that summary explicitly failed QA.
    for key in ("final_qa_ok", "qa_ok"):
        if key in result and not _flag(result.get(key)):
            return True, "qa_failed"

    if qa is None:
        return True, "qa_missing"
    return False, "ok"


def harness_loop(
    run_dir: str,
    cfg: Optional[dict] = None,
    *,
    pipeline_result: Optional[dict] = None,
    staging: Optional[dict] = None,
    max_iterations: Optional[int] = None,
    run_one: Optional[Callable[..., dict]] = None,
) -> dict:
    """After a completed pipeline run, repair QA/staging failures before returning done."""
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    pipeline_result = dict(pipeline_result or {})

    qa_path = os.path.join(run_dir, "qa.json")
    qa = load_qa(run_dir) if os.path.exists(qa_path) else None
    should, reason = harness_should_repair(pipeline_result, qa=qa, staging=staging)
    if not should:
        if qa is not None:
            pipeline_result.setdefault("qa_ok", _qa_accepts(qa, allow_summary=True))
        return {
            "repaired": False,
            "reason": reason,
            "pipeline_result": pipeline_result,
        }

    iterations = max_iterations
    if iterations is None:
        iterations = harness_max_rounds(cfg)

    repair_summary = execute_repairs(
        run_dir,
        cfg,
        max_iterations=iterations,
        run_one=run_one,
    )
    pipeline_result["repair"] = repair_summary
    pipeline_result["qa_ok"] = _flag(repair_summary.get("qa_ok"))

    final_qa = load_qa(run_dir) if os.path.exists(qa_path) else {}
    if final_qa:
        pipeline_result["qa_ok"] = _qa_accepts(final_qa, allow_summary=True)

    return {
        "repaired": True,
        "reason": reason,
        "repair": repair_summary,
        "pipeline_result": pipeline_result,
    }


def execute_repairs(
    run_dir: str,
    cfg: Optional[dict] = None,
    max_iterations: Optional[int] = None,
    *,
    run_one: Optional[Callable[..., dict]] = None,
) -> dict:
    """Apply actionable repairs by resuming the pipeline from mapped stages."""
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    if max_iterations is None:
        max_iterations = harness_max_rounds(cfg)
    if run_one is None:
        import run_pipeline
        run_one = run_pipeline.run_one

    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    if _qa_accepts(qa, allow_summary=True):
        return _save_harness_summary(run_dir, {
            "run_dir": run_dir,
            "iterations": 0,
            "qa_ok": True,
            "stopped": "already_ok",
            "attempts": [],
        })

    input_path = _input_path(run_dir)
    if not input_path:
        return _save_harness_summary(run_dir, {
            "run_dir": run_dir,
            "iterations": 0,
            "qa_ok": False,
            "stopped": "missing_input",
            "error": "could not resolve input image for repair rerun",
            "attempts": [],
        })

    attempts = []
    exhausted: set[tuple] = set()
    working_cfg = copy.deepcopy(cfg)
    working_cfg.setdefault("runtime", {})["auto_repair"] = False

    for iteration in range(1, max(0, int(max_iterations)) + 1):
        repairs = load_repairs(run_dir, working_cfg)
        candidates = [repair for repair in repairs if _repair_id(repair) not in exhausted]
        choice = recommended_resume(candidates)
        if not choice:
            stopped = "all_repairs_failed" if repairs and exhausted else "no_actionable_repairs"
            return _save_harness_summary(run_dir, {
                "run_dir": run_dir,
                "iterations": iteration - 1,
                "qa_ok": _qa_accepts(_load_json(os.path.join(run_dir, "qa.json"), {})),
                "stopped": stopped,
                "attempts": attempts,
            })

        resume = choice["resume"]
        patches = choice.get("patches") or {}
        iter_cfg = deep_merge(working_cfg, patches)
        iter_cfg["run_dir"] = run_dir
        iter_cfg.setdefault("runtime", {})["auto_repair"] = False

        qa_path = os.path.join(run_dir, "qa.json")
        before_qa_fingerprint = _artifact_fingerprint(qa_path)
        try:
            result = _invoke_run_one(run_one, input_path, run_dir, iter_cfg, resume)
            if not isinstance(result, dict):
                raise TypeError(f"pipeline runner returned {type(result).__name__}, expected dict")
            pipeline_error = result.get("error")
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "exception": type(exc).__name__}
            pipeline_error = str(exc)
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        after_qa_fingerprint = _artifact_fingerprint(qa_path)
        qa_fresh = (
            after_qa_fingerprint is not None
            and after_qa_fingerprint != before_qa_fingerprint
        )
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
            "pipeline_error": pipeline_error,
            "qa_fresh": qa_fresh,
            "qa_ok": qa_fresh and _qa_accepts(qa, allow_summary=True),
        }
        attempts.append(attempt)
        working_cfg = iter_cfg

        # A failed stage or a run that produced no new QA cannot prove progress.
        # Exhaust that exact tactic and let the next repair act as the fallback.
        has_alternative = any(_repair_id(item) != _repair_id(choice) for item in repairs)
        if not result.get("ok") or (not qa_fresh and has_alternative):
            exhausted.add((choice.get("stage"), choice.get("action"), choice.get("target_id")))

        if qa_fresh and _qa_accepts(qa, allow_summary=True):
            summary = {
                "run_dir": run_dir,
                "iterations": iteration,
                "qa_ok": True,
                "stopped": "qa_ok",
                "attempts": attempts,
            }
            return _save_harness_summary(run_dir, summary)

    final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    summary = {
        "run_dir": run_dir,
        "iterations": len(attempts),
        "qa_ok": _qa_accepts(final_qa),
        "stopped": "max_iterations",
        "attempts": attempts,
    }
    return _save_harness_summary(run_dir, summary)
