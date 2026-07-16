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
import time
from typing import Any, Callable, Optional

# Repair modules use logical stage names; the orchestrator uses STAGES in run_pipeline.
STAGE_ALIASES = {
    "text-analysis": "text",
    "inpaint": "reconstruct",
    "vectorize": "reconstruct",
    "build": "design",
    "sam3": "sam",
}

# Canonical pipeline stage order (mirrors run_pipeline.STAGES). Resuming at stage i
# re-runs every stage >= i via the orchestrator's dirty cascade — including the peel
# stack's Flux inpaints — so the harness must never resume earlier than the first
# stage a repair's config patch can actually affect (GB6).
PIPELINE_STAGE_ORDER = [
    "normalize", "ocr", "text", "residual", "qwen", "sam", "elements",
    "peel", "merge", "structure", "reconstruct", "layout", "design", "preview",
    "figma", "export", "diff", "qa",
]

# Config levers the pipeline stages actually read, verified against stage code. A repair
# whose patch only writes keys outside this registry cannot change any stage's behaviour;
# the resumed rerun is guaranteed byte-identical. postfix-benchmark-4 evidence: ~70
# refit-text-box repairs wrote ``text_analysis.fit`` and 009's restore-native-nodes wrote
# ``design.restore_native_nodes`` — no stage reads either key, so every one of those
# reruns replayed the full downstream stack (091: 12 Flux peel inpaints) for a no-op.
_PIPELINE_LEVERS: dict[str, set] = {
    "ocr": {"challengers", "retry_2x"},
    "text_analysis": {"font_matching"},
    "vlm": {"enabled", "ocr_judge", "font_judge", "scene_text",
            "segment_filter", "element_propose"},
    "qwen": {"enabled", "layers"},
    "merge": {"dedup_iou", "dedup_text", "duplicate_text", "layer_ids"},
    "reconstruct": {"focus_regions", "dedup_iou"},
    "inpaint": {"mode", "allow_fallback"},
    # measured_dx/measured_dy are the admission contract for measured VLM geometry
    # evidence (CRITIC-REVIEW 002 replay test) even though layout currently derives its
    # own offsets; min/max_container_frac are read directly by layout.py.
    "layout": {"min_container_frac", "max_container_frac", "measured_dx", "measured_dy"},
    "sam3": {"enabled", "confidence", "box_refine_confidence", "reject_internal_holes"},
    "vectorize": {"force_raster_fallback"},
    "figma": {"enabled", "reimport"},
    "fallback": {"force_slice_ids"},
}

# Earliest pipeline stage that reads each config section (vlm resolved per-subkey).
_SECTION_EARLIEST_STAGE = {
    "ocr": "ocr", "text_analysis": "text", "qwen": "qwen", "sam3": "sam",
    "merge": "merge", "inpaint": "reconstruct", "reconstruct": "reconstruct",
    "vectorize": "reconstruct", "fallback": "reconstruct", "layout": "layout",
    "design": "design", "figma": "figma",
}
_VLM_SUBKEY_STAGE = {
    "ocr_judge": "ocr", "scene_text": "text", "font_judge": "text",
    "segment_filter": "sam", "element_propose": "sam",
}

# Primary artifacts each resume stage writes. If none of these change after a repair
# rerun, the round produced byte-identical outputs (the plateau-with-identical-artifacts
# class) and the rest of the round can be short-circuited.
_STAGE_OUTPUTS: dict[str, tuple] = {
    "ocr": ("ocr_raw.json",),
    "text": ("ocr.json",),
    "qwen": ("qwen.json",),
    "sam": ("elements.json",),
    "elements": ("fused_elements.json",),
    "merge": ("merged.json",),
    "reconstruct": ("reconstruction.json", "fallback.json"),
    "layout": ("layout.json",),
    "design": (),
    "figma": ("figma_import.json",),
}
# Tail sentinels watched for every resume stage: any real repair must eventually move
# the compiled design or its render.
_TAIL_OUTPUTS = ("design.json", "preview.png")


def _stage_order_index(name: Optional[str]) -> int:
    try:
        return PIPELINE_STAGE_ORDER.index(str(name))
    except ValueError:
        return -1


def _reachable_levers(patches: dict) -> list[tuple[str, str]]:
    """(section, key) pairs of a patch that a pipeline stage actually reads."""
    out: list[tuple[str, str]] = []
    for section, body in (patches or {}).items():
        if section == "harness" or not isinstance(body, dict):
            continue
        levers = _PIPELINE_LEVERS.get(section)
        if not levers:
            continue
        for key, value in body.items():
            if key not in levers:
                continue
            if section == "reconstruct" and key == "focus_regions":
                # apply_raster_slice_fallback only honors entries carrying a layer_id;
                # box-only "worst window" regions are dead weight (the 002/016 no-ops).
                if not any(isinstance(entry, dict) and entry.get("layer_id")
                           for entry in (value or [])):
                    continue
            out.append((section, key))
    return out


def patch_reaches_pipeline(patches: dict) -> bool:
    """True when at least one patched key is a lever some pipeline stage reads."""
    return bool(_reachable_levers(patches))


def earliest_patched_stage(patches: dict) -> Optional[str]:
    """Earliest pipeline stage that reads any lever this patch writes (GB6)."""
    best: Optional[int] = None
    for section, key in _reachable_levers(patches):
        if section == "vlm":
            stage = _VLM_SUBKEY_STAGE.get(key, "ocr")
        else:
            stage = _SECTION_EARLIEST_STAGE.get(section)
        index = _stage_order_index(stage)
        if index >= 0 and (best is None or index < best):
            best = index
    return PIPELINE_STAGE_ORDER[best] if best is not None else None

# Actions the harness can drive without human review.
ACTIONABLE = {
    ("ocr", "rerun"),
    ("ocr", "boost-stack"),
    ("text-analysis", "restore-editable-text"),
    ("text-analysis", "refit-colors-effects"),
    ("text-analysis", "resolve-fonts"),
    ("text-analysis", "refit-text-box"),
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
            # retry_2x is enabled by default (ocr.py: scale 2.0, low_confidence 0.72,
            # max_regions 6), so {"enabled": True} was a placebo — the rerun was
            # config-identical. Escalate for real: higher upscale, retry more lines.
            ocr_patch["retry_2x"] = {
                "enabled": True, "scale": 3.0,
                "low_confidence": 0.85, "max_regions": 12,
            }
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
        if params.get("raise_dedup_iou") or params.get("duplicate_text") or params.get("layer_ids"):
            merge_patch: dict[str, Any] = {"dedup_iou": 0.72}
            # Duplicate/ghosted text (from the anomaly pass): tell merge to dedupe text
            # layers and which offending text/layers to collapse to a single owner.
            if params.get("duplicate_text"):
                merge_patch["dedup_text"] = True
                merge_patch["duplicate_text"] = list(params["duplicate_text"])
            if params.get("layer_ids"):
                merge_patch["dedup_text"] = True
                merge_patch["layer_ids"] = list(params["layer_ids"])
            patches["merge"] = merge_patch
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

    elif stage == "text-analysis" and action == "refit-text-box":
        # Clipped/cut-off text: let the text stage widen the box and/or shrink-to-fit so
        # the glyphs stop being cropped at a container or image edge.
        fit_patch: dict[str, Any] = {"refit": True}
        if params.get("widen", True):
            fit_patch["widen_clipped"] = True
        if params.get("shrink_to_fit", True):
            fit_patch["shrink_to_fit"] = True
        clipped = params.get("clipped_text") or params.get("clipped")
        if clipped:
            fit_patch["clipped_text"] = list(clipped)
        layer_ids = params.get("layer_ids")
        if layer_ids:
            fit_patch["layer_ids"] = list(layer_ids)
        patches["text_analysis"] = {"fit": fit_patch}

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
    if not (stage and action and (stage, action) in ACTIONABLE and resume_stage_for(repair)):
        return False
    if (stage, action) in _SELF_CONCRETE_ACTIONS:
        return True
    # Reachability screen: a repair whose entire patch writes config keys no pipeline
    # stage reads can never change the output — it is not actionable, so it must not
    # trigger harness rounds or burn a full-pipeline rerun (postfix-benchmark-4:
    # refit-text-box/restore-native-nodes/rebuild-schema were all in this class).
    patches = {key: value for key, value in config_patches_for(repair).items()
               if key != "harness"}
    if any(value for value in patches.values()):
        return patch_reaches_pipeline(patches)
    # Empty-patch untargeted rerun: config-identical, so it can only differ through
    # stage nondeterminism. That lottery ticket must never replay the expensive peel
    # stack (Flux inpaints) — admit it only when the resume point is after peel.
    return _stage_order_index(resume_stage_for(repair)) > _stage_order_index("peel")


# Actions whose resume performs real, state-changing I/O even with an identical config
# (re-staging the inbox, re-importing after a compiler report). Everything else needs a
# concrete config delta to be worth a rerun. NOTE: ("reconstruct", "restage-assets") was
# removed — its ``reconstruct.restage_assets`` patch has no consumer anywhere in the
# pipeline, so the rerun it bought was always byte-identical.
_SELF_CONCRETE_ACTIONS = {
    ("figma", "restage-inbox"),
    ("figma", "fix-compiler-report"),
}


def plan_is_concrete(choice: dict) -> bool:
    """Actionability gate (admission): a plan must specify a concrete, checkable change.

    The 002 failure class: a VLM opinion became ``layout/refit-geometry`` whose ONLY
    config change was ``harness.target_id`` — no dx/dy, no box, no font, no measurable
    delta — a guaranteed no-op rerun that then poisoned the plateau logic. A targeted
    plan (target_id set, or a VLM-sourced opinion) must patch something real beyond the
    target id; untargeted whole-stage reruns stay admissible (the plan-fingerprint
    memory already blocks their replays on unchanged inputs)."""
    if (choice.get("stage"), choice.get("action")) in _SELF_CONCRETE_ACTIONS:
        return True
    patches = {key: value for key, value in (choice.get("patches") or {}).items()
               if key != "harness"}
    if any(value for value in patches.values()):
        # A non-empty patch is only concrete when at least one written key is a lever a
        # pipeline stage actually reads; a patch made entirely of unread keys is a
        # guaranteed byte-identical rerun (the "repairs never convert" class).
        return patch_reaches_pipeline(patches)
    params = choice.get("params") or {}
    if params.get("source") == "vlm_critique":
        return False
    if choice.get("target_id"):
        return False
    # Empty-patch untargeted rerun: only concrete when its resume cannot replay the
    # expensive peel stack (same guard as is_actionable).
    return _stage_order_index(choice.get("resume")) > _stage_order_index("peel")


def _layer_local_scores(qa: Optional[dict]) -> dict:
    """Blended per-layer local scores from qa.per_layer (region_ssim + ink/colour)."""
    out: dict[str, float] = {}
    for row in (qa or {}).get("per_layer") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        value = row.get("region_ssim")
        if not isinstance(value, (int, float)):
            value = row.get("score")
        if not isinstance(value, (int, float)):
            continue
        value = max(0.0, min(1.0, float(value)))
        ink_iou = row.get("ink_iou")
        region_color = row.get("region_color")
        if isinstance(ink_iou, (int, float)):
            value = 0.7 * value + 0.3 * max(0.0, min(1.0, float(ink_iou)))
        elif isinstance(region_color, (int, float)):
            value = 0.85 * value + 0.15 * max(0.0, min(1.0, float(region_color)))
        out[str(row["id"])] = value
    return out


def _repair_measured_badness(repair: dict, layer_scores: dict) -> float:
    """Measured badness of what a repair targets (0 when it carries no local evidence)."""
    worst = None
    params = repair.get("params") or {}
    regions = list(params.get("regions") or [])
    region = params.get("region")
    if isinstance(region, dict):
        regions.append(region)
    for entry in regions:
        if not isinstance(entry, dict):
            continue
        value = entry.get("local_score")
        if not isinstance(value, (int, float)):
            value = entry.get("region_ssim")
        if isinstance(value, (int, float)):
            worst = value if worst is None else min(worst, value)
    target = repair.get("target_id")
    if target is not None and str(target) in layer_scores:
        value = layer_scores[str(target)]
        worst = value if worst is None else min(worst, value)
    if worst is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(worst)))


_RANK_SEVERITY = {"high": 3, "medium": 2, "low": 1}


def rank_repairs(repairs: list, qa: Optional[dict] = None) -> list:
    """Deterministic, measured evidence outranks VLM opinions; worst measured first.

    Order: severity desc → deterministic (metric/tool) before VLM-critique-sourced at
    the same severity (VLM opinions are tiebreakers, never the primary driver) →
    measured local badness desc → original order (stable)."""
    layer_scores = _layer_local_scores(qa)

    def key(indexed):
        index, repair = indexed
        params = repair.get("params") or {}
        vlm = 1 if params.get("source") == "vlm_critique" else 0
        severity = _RANK_SEVERITY.get(str(repair.get("severity") or "").lower(), 0)
        badness = _repair_measured_badness(repair, layer_scores)
        return (-severity, vlm, -badness, index)

    return [repair for _, repair in sorted(enumerate(repairs or []), key=key)]


def has_actionable_repairs(repairs: list) -> bool:
    """True when QA still has medium/high severity actionable repairs."""
    for repair in repairs or []:
        if not is_actionable(repair):
            continue
        if str(repair.get("severity") or "").lower() in {"high", "medium"}:
            return True
    return False


def load_repair_candidates(run_dir: str, cfg: Optional[dict] = None) -> list:
    """Prefer critic-filtered repairs; fall back to QA/repair assess list.

    Either list is re-ranked against the measured per-layer QA evidence so deterministic
    measured failures outrank VLM opinions and the worst measured layer is first."""
    run_dir = os.path.abspath(run_dir)
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    critic = _load_json(os.path.join(run_dir, "critic.json"), {})
    filtered = critic.get("filtered_repairs")
    if isinstance(filtered, list) and filtered:
        return rank_repairs(filtered, qa)
    return rank_repairs(load_repairs(run_dir, cfg), qa)


def recommended_resume(repairs: list) -> Optional[dict]:
    """Pick the highest-priority actionable repair and expose resume metadata."""
    for repair in repairs or []:
        if not is_actionable(repair):
            continue
        resume = resume_stage_for(repair)
        if not resume:
            continue
        patches = config_patches_for(repair)
        # GB6: never resume earlier than the first stage the patch can actually affect.
        # Resuming earlier replays expensive stages (peel's Flux inpaints, SAM) whose
        # inputs and config the patch does not touch, producing identical outputs at
        # full cost (the 091 full peel-stack replay class).
        if (repair.get("stage"), repair.get("action")) not in _SELF_CONCRETE_ACTIONS:
            earliest = earliest_patched_stage(patches)
            if earliest and _stage_order_index(resume) >= 0 \
                    and _stage_order_index(earliest) > _stage_order_index(resume):
                resume = earliest
        return {
            "stage": repair.get("stage"),
            "action": repair.get("action"),
            "resume": resume,
            "target_id": repair.get("target_id"),
            "reason": repair.get("reason"),
            "severity": repair.get("severity"),
            "params": dict(repair.get("params") or {}),
            "patches": patches,
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


def _qa_progress(before: dict, after: dict) -> tuple[bool, dict]:
    """Return meaningful metric progress, not merely a rewritten qa.json."""
    before, after = before or {}, after or {}
    deltas = {}
    tolerances = {
        "ssim": 0.005, "visual_score": 0.005, "text_recall": 0.01,
        "editable_text_recall": 0.01, "edge_f1": 0.005,
        "color_similarity": 0.005,
    }
    improved = False
    if _qa_accepts(after, allow_summary=True) and not _qa_accepts(before, allow_summary=True):
        improved = True
    for key, minimum in tolerances.items():
        old, new = before.get(key), after.get(key)
        if isinstance(new, (int, float)):
            delta = None if not isinstance(old, (int, float)) else float(new) - float(old)
            deltas[key] = None if delta is None else round(delta, 6)
            if isinstance(old, (int, float)) and delta >= minimum:
                improved = True
    before_fails = before.get("hard_fails") or []
    after_fails = after.get("hard_fails") or []
    deltas["hard_fails"] = len(after_fails) - len(before_fails)
    if len(after_fails) < len(before_fails):
        improved = True
    return bool(improved), deltas


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


def _repair_plan_fingerprint(run_dir: str, choice: dict) -> tuple[str, dict]:
    """Fingerprint a repair plus the artifacts it would consume.

    If neither the plan nor its stage inputs changed, repeating the repair cannot add
    information. Persisting this across harness invocations prevents identical OCR/SAM/
    reconstruction reruns from burning another full pass after a plateau or interruption.
    """
    resume = str(choice.get("resume") or "")
    inputs = {
        "ocr": ("normalized.png",),
        "text": ("ocr_raw.json", "normalized.png"),
        "sam": ("residual.json", "qwen.json", "ocr.json"),
        "elements": ("residual.json", "qwen.json", "ocr.json"),
        "reconstruct": ("merged.json", "ocr.json"),
        "layout": ("reconstruction.json",),
        "design": ("layout.json", "reconstruction.json"),
        "figma": ("design.json",),
    }.get(resume, ("qa.json",))
    evidence = {name: _artifact_fingerprint(os.path.join(run_dir, name)) for name in inputs}
    payload = {
        "stage": choice.get("stage"), "action": choice.get("action"),
        "target_id": choice.get("target_id"), "resume": resume,
        "patches": choice.get("patches") or {}, "inputs": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload


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


# ── round budget (cost control) ─────────────────────────────────────────────────────────
# postfix-benchmark-5 002: ONE repair round ran 15 minutes (a sam3 rerun-detection repair
# cascaded vlm element propose → sam3 → fusion → a ~40-inpaint peel storm → Flux calls at
# 178-266s each in reconstruct) and produced a WORSE design. A repair round must never be
# allowed to grind unbounded: it gets a wall-clock ceiling and, for reruns that replay the
# peel stack, a clamped peel Flux budget (consuming peel_scene's own flux_budget /
# max_iterations config — the harness only tightens those knobs, never loosens them).
_DEFAULT_ROUND_WALL_CLOCK_S = 600.0   # a repair round may not cost 10+ minutes
_DEFAULT_ROUND_FLUX_CALLS = 2         # half of peel_scene DEFAULTS["flux_budget"] (4)


def round_budget(cfg: Optional[dict]) -> dict:
    """Per-harness-round cost ceiling from ``runtime.harness.round_budget``.

    Keys:
      wall_clock_s     max seconds a single repair round may spend (<=0 disables)
      flux_calls       peel.flux_budget clamp for reruns that resume at/before peel
                       (None/negative disables the clamp)
      peel_iterations  optional peel.max_iterations clamp for the same reruns
    """
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    raw = harness.get("round_budget") or {}
    if not isinstance(raw, dict):
        raw = {}
    try:
        wall = float(raw.get("wall_clock_s", _DEFAULT_ROUND_WALL_CLOCK_S))
    except (TypeError, ValueError):
        wall = _DEFAULT_ROUND_WALL_CLOCK_S
    if wall <= 0:
        wall = float("inf")
    flux: Optional[int]
    try:
        flux_raw = raw.get("flux_calls", _DEFAULT_ROUND_FLUX_CALLS)
        flux = None if flux_raw is None else int(flux_raw)
    except (TypeError, ValueError):
        flux = _DEFAULT_ROUND_FLUX_CALLS
    if flux is not None and flux < 0:
        flux = None
    peel_iters: Optional[int]
    try:
        peel_raw = raw.get("peel_iterations")
        peel_iters = None if peel_raw is None else max(1, int(peel_raw))
    except (TypeError, ValueError):
        peel_iters = None
    return {"wall_clock_s": wall, "flux_calls": flux, "peel_iterations": peel_iters}


def apply_round_budget_clamp(cfg: dict, resume: Optional[str],
                             budget: Optional[dict] = None) -> tuple[dict, Optional[dict]]:
    """Clamp the peel stack's own budget knobs for a rerun that replays peel.

    Reuses peel_scene's config semantics (``peel.flux_budget``, ``peel.max_iterations``)
    rather than touching peel code: a harness rerun resuming at or before the peel stage
    gets AT MOST ``round_budget.flux_calls`` Flux peel inpaints. The clamp only ever
    lowers the pipeline's existing values. Returns (cfg, clamp_record|None)."""
    if budget is None:
        budget = round_budget(cfg)
    resume_index = _stage_order_index(resume)
    if resume_index < 0 or resume_index > _stage_order_index("peel"):
        return cfg, None
    record: dict[str, Any] = {}
    existing = cfg.get("peel")
    if existing is not None and not isinstance(existing, dict):
        return cfg, None  # cfg["peel"] exists but is not a dict — leave it alone
    peel_cfg = cfg.setdefault("peel", {})
    flux_ceiling = budget.get("flux_calls")
    if flux_ceiling is not None:
        try:
            current = int(peel_cfg.get("flux_budget", 4))
        except (TypeError, ValueError):
            current = 4
        clamped = max(0, min(current, int(flux_ceiling)))
        if clamped != current or "flux_budget" not in peel_cfg:
            record["flux_budget"] = {"from": current, "to": clamped}
        peel_cfg["flux_budget"] = clamped
    iter_ceiling = budget.get("peel_iterations")
    if iter_ceiling is not None:
        try:
            current_iters = int(peel_cfg.get("max_iterations", 3))
        except (TypeError, ValueError):
            current_iters = 3
        clamped_iters = max(1, min(current_iters, int(iter_ceiling)))
        if clamped_iters != current_iters:
            record["max_iterations"] = {"from": current_iters, "to": clamped_iters}
        peel_cfg["max_iterations"] = clamped_iters
    return cfg, (record or None)


def harness_max_rounds(cfg: Optional[dict]) -> int:
    """Max repair rounds from config (default 2)."""
    runtime = (cfg or {}).get("runtime") or {}
    harness = runtime.get("harness") or {}
    if harness.get("max_rounds") is not None:
        return max(0, int(harness["max_rounds"]))
    if runtime.get("harness_max_iterations") is not None:
        return max(0, int(runtime["harness_max_iterations"]))
    if runtime.get("auto_repair_max_iterations") is not None:
        return max(0, int(runtime["auto_repair_max_iterations"]))
    return 2


def harness_should_repair(
    pipeline_result: Optional[dict],
    *,
    qa: Optional[dict] = None,
    staging: Optional[dict] = None,
    run_dir: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> tuple[bool, str]:
    """Decide whether to run repairs before reporting a job done to the plugin."""
    result = pipeline_result or {}
    if not result.get("ok"):
        return False, "pipeline_failed"

    if staging is not None and not _flag(staging.get("staged")):
        return True, "staging_failed"

    repairs = list((qa or {}).get("repairs") or [])
    if run_dir and not repairs:
        repairs = load_repairs(run_dir, cfg)

    if qa is not None and not _qa_accepts(qa, allow_summary=True):
        return True, "qa_failed"

    if qa is not None and has_actionable_repairs(repairs):
        return True, "actionable_repairs"

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
    blocked_repairs: Optional[set] = None,
) -> dict:
    """Apply actionable repairs by resuming the pipeline from mapped stages.

    ``blocked_repairs`` seeds the exhausted set with repair signatures (stage, action,
    target_id) the caller already knows produced no improvement — the harness loop passes
    these across rounds so a non-improving repair is never re-applied (no oscillation).
    """
    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    if max_iterations is None:
        max_iterations = harness_max_rounds(cfg)
    if run_one is None:
        import run_pipeline
        run_one = run_pipeline.run_one

    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    repairs = load_repair_candidates(run_dir, cfg)
    if _qa_accepts(qa, allow_summary=True) and not has_actionable_repairs(repairs):
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
    exhausted: set[tuple] = set(blocked_repairs or ())
    working_cfg = copy.deepcopy(cfg)
    working_cfg.setdefault("runtime", {})["auto_repair"] = False
    budget = round_budget(cfg)
    round_started = time.monotonic()
    admission_path = os.path.join(run_dir, "harness_admission.json")
    admission = _load_json(admission_path, {"seen": {}, "skipped": []})
    seen_plans = admission.get("seen") if isinstance(admission.get("seen"), dict) else {}
    seen_classes = (admission.get("classes")
                    if isinstance(admission.get("classes"), dict) else {})

    for iteration in range(1, max(0, int(max_iterations)) + 1):
        # Cost control: a repair round has a wall-clock ceiling. Once an attempt has
        # blown it, no further pipeline reruns may start this round — the loop above
        # decides whether the expensive attempt's output is kept or rolled back.
        elapsed = time.monotonic() - round_started
        if elapsed > budget["wall_clock_s"]:
            return _save_harness_summary(run_dir, {
                "run_dir": run_dir,
                "iterations": iteration - 1,
                "qa_ok": _qa_accepts(_load_json(os.path.join(run_dir, "qa.json"), {})),
                "stopped": "round_budget_exceeded",
                "budget": {"wall_clock_s": budget["wall_clock_s"],
                           "elapsed_s": round(elapsed, 1)},
                "attempts": attempts,
            })
        # Admission screen (GB3): rejecting a non-concrete / already-failed / unchanged
        # plan must move to the NEXT candidate WITHIN this iteration, not burn the whole
        # iteration budget. With repair_iterations=1 the old `continue` let a single
        # rejected top candidate exhaust the round before any concrete repair ran. Only a
        # real pipeline rerun (below the while) counts against max_iterations.
        while True:
            repairs = load_repair_candidates(run_dir, working_cfg)
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

            # Actionability gate BEFORE spending a rerun: a plan whose only config change is
            # harness.target_id (no dx/dy, no box, no font, no measurable delta) is a
            # guaranteed no-op and is rejected at admission.
            if not plan_is_concrete(choice):
                exhausted.add((choice.get("stage"), choice.get("action"), choice.get("target_id")))
                rejected = {
                    "iteration": iteration, "resume": resume,
                    "repair": {key: choice.get(key) for key in ("stage", "action", "target_id")},
                    "reason": "not-actionable-no-concrete-change",
                    "detail": ("plan specifies no concrete, checkable change beyond a target "
                               "id — rejected before spending a pipeline rerun"),
                }
                attempts.append({**rejected, "admission_rejected": True})
                admission.setdefault("rejected", []).append(rejected)
                _write_json(admission_path, admission)
                continue

            # Semantic no-repeat for VLM opinions: two equivalent whole-image opinions (same
            # operator class, different target — the €63 vs €49 case) must not each burn a
            # pass once the class is proven non-improving.
            choice_params = choice.get("params") or {}
            vlm_class = None
            if choice_params.get("source") == "vlm_critique":
                vlm_class = f"vlm:{resume}:{choice.get('action')}"
                prior = seen_classes.get(vlm_class)
                if isinstance(prior, dict) and not prior.get("qa_improved"):
                    exhausted.add((choice.get("stage"), choice.get("action"),
                                   choice.get("target_id")))
                    skipped = {
                        "iteration": iteration, "resume": resume,
                        "repair": {key: choice.get(key)
                                   for key in ("stage", "action", "target_id")},
                        "reason": "equivalent-vlm-opinion-already-failed",
                        "class": vlm_class,
                    }
                    attempts.append({**skipped, "admission_skipped": True})
                    admission.setdefault("skipped", []).append(skipped)
                    _write_json(admission_path, admission)
                    continue

            plan_fingerprint, plan_payload = _repair_plan_fingerprint(run_dir, choice)
            if plan_fingerprint in seen_plans:
                exhausted.add((choice.get("stage"), choice.get("action"), choice.get("target_id")))
                skipped = {
                    "iteration": iteration, "resume": resume,
                    "repair": {key: choice.get(key) for key in ("stage", "action", "target_id")},
                    "reason": "unchanged-repair-plan-and-inputs",
                    "plan_fingerprint": plan_fingerprint,
                }
                attempts.append({**skipped, "admission_skipped": True})
                admission.setdefault("skipped", []).append(skipped)
                _write_json(admission_path, admission)
                continue

            # Admissible, concrete plan — leave the screen and spend one iteration on it.
            break

        iter_cfg = deep_merge(working_cfg, patches)
        iter_cfg["run_dir"] = run_dir
        iter_cfg.setdefault("runtime", {})["auto_repair"] = False
        # Peel discipline for reruns: a repair that resumes at/before peel replays the
        # peel stack — clamp its Flux budget so an element-propose repair can never
        # trigger an unbounded big-lama/Flux storm (the 002 15-minute round).
        iter_cfg, budget_clamp = apply_round_budget_clamp(iter_cfg, resume, budget)

        qa_path = os.path.join(run_dir, "qa.json")
        qa_before = copy.deepcopy(qa)
        before_qa_fingerprint = _artifact_fingerprint(qa_path)
        # Watch the resumed stage's primary outputs plus the compiled-design tail. If
        # none change, the repair provably had no effect (qa.json alone can be rewritten
        # with identical metrics), and the loop can short-circuit the rest of the round.
        watched = tuple(dict.fromkeys(_STAGE_OUTPUTS.get(resume, ()) + _TAIL_OUTPUTS))
        before_watched = {
            name: _artifact_fingerprint(os.path.join(run_dir, name)) for name in watched
        }
        attempt_started = time.monotonic()
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
        qa_improved, metric_deltas = _qa_progress(qa_before, qa) if qa_fresh else (False, {})
        artifacts_changed = any(
            _artifact_fingerprint(os.path.join(run_dir, name)) != before_watched[name]
            for name in watched
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
            "qa_improved": qa_improved,
            "artifacts_changed": artifacts_changed,
            "metric_deltas": metric_deltas,
            "qa_ok": qa_fresh and _qa_accepts(qa, allow_summary=True),
            "elapsed_s": round(time.monotonic() - attempt_started, 1),
        }
        if budget_clamp:
            attempt["budget_clamp"] = budget_clamp
        if result.get("ok") and not artifacts_changed and not qa_improved:
            attempt["no_effect"] = "identical-artifacts"
        attempts.append(attempt)
        seen_plans[plan_fingerprint] = {
            "plan": plan_payload, "qa_fresh": qa_fresh, "qa_improved": qa_improved,
        }
        admission["seen"] = seen_plans
        if vlm_class:
            seen_classes[vlm_class] = {
                "qa_improved": qa_improved, "target_id": choice.get("target_id"),
            }
            admission["classes"] = seen_classes
        _write_json(admission_path, admission)
        if qa_improved or (qa_fresh and _qa_accepts(qa, allow_summary=True)):
            working_cfg = iter_cfg

        # A failed stage or a run that produced no new QA cannot prove progress.
        # Exhaust that exact tactic and let the next repair act as the fallback.
        if not result.get("ok") or not qa_fresh or not qa_improved:
            exhausted.add((choice.get("stage"), choice.get("action"), choice.get("target_id")))

        if qa_fresh and _qa_accepts(qa, allow_summary=True):
            explicit_repairs = qa.get("repairs")
            if explicit_repairs is None:
                remaining: list = []
            else:
                remaining = explicit_repairs
            if not has_actionable_repairs(remaining):
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
