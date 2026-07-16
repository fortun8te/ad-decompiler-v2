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
    # postfix-benchmark-6: {"mode", "allow_fallback"} alone under-declared this section.
    # inpaint.py genuinely reads all of the following, verified against stage code:
    #   mask_dilate        -> resolve_mask_dilate / default_mask_dilate (inpaint.py:375,429)
    #   multipass_fraction -> inpaint.py:1099
    #   mask_feather       -> inpaint.py:448
    #   strict_acceptance  -> inpaint.py:340,955
    # Omitting them made the ONLY levers that can actually scrub glyph residue look
    # "unreachable" to the screen, so the planner was left with the inert mode flip.
    "inpaint": {"mode", "allow_fallback", "mask_dilate", "multipass_fraction",
                "mask_feather", "strict_acceptance"},
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

# Escalation ladder for inpaint:rebuild-clean-plate (glyph residue under removed text).
# Each rung is a strictly stronger, *reachable* experiment than the one before it, so a
# second round tests something new instead of replaying round 1 (the bench-6 "no_progress"
# class). Rung values are deliberately above config.yaml's effective defaults
# (mask_dilate unset -> text halo of 2-5px via resolve_mask_dilate's role table,
# multipass_fraction 0.12) so every rung is a real config delta.
_CLEAN_PLATE_LADDER: tuple[dict, ...] = (
    # Rung 0 — widen the removal footprint and scrub harder on the current engine.
    {"mask_dilate": {"default": 3, "text": 6, "overlay_text": 7, "shape": 6, "button": 8},
     "multipass_fraction": 0.06,
     "allow_fallback": False},
    # Rung 1 — rung 0, wider, on a deterministic CPU engine. Big-LaMa removes the
    # Flux budget/nondeterminism from the equation and is the fixer's known-good
    # escalation (harness_fixer.fix_inpaint).
    {"mode": "big-lama",
     "mask_dilate": {"default": 4, "text": 8, "overlay_text": 9, "shape": 7, "button": 10},
     "multipass_fraction": 0.04,
     "allow_fallback": False},
)


# Repairs whose patch depends on how many times they already ran.
_LADDERED_ACTIONS = {("inpaint", "rebuild-clean-plate")}

# ── blocker → lever map (postfix-benchmark-6 honesty pass) ──────────────────────────
# For every QA failure that blocks acceptance, either name the config lever that can
# move it, or state plainly that none exists. A blocker with no lever must be REFUSED
# with a reason, not retried: rounds spent on it are pure cost. Verified against stage
# code — each "structural" entry below has no config key any stage reads that would
# change the measured quantity.
_STRUCTURAL_NEEDS_CODE = "structural defect: needs code fix, not config"

_BLOCKER_LEVERS: dict[str, dict] = {
    "glyph_residue": {
        "fixable": True,
        "lever": "inpaint:rebuild-clean-plate",
        "detail": "removal-mask footprint / scrub pass (inpaint.mask_dilate, multipass_fraction)",
    },
    "low_text_recall": {
        "fixable": True,
        "lever": "ocr:rerun",
        "detail": "ocr.retry_2x upscale + challengers",
    },
    "element_recall": {
        "fixable": True,
        "lever": "sam3:rerun-detection",
        "detail": "sam3.confidence / vlm.element_propose",
    },
    # ── no lever exists ────────────────────────────────────────────────────────────
    "placement_ink_iou": {
        "fixable": False,
        "reason": _STRUCTURAL_NEEDS_CODE,
        "detail": ("layer geometry is computed by layout/reconstruct from structure "
                   "evidence; no config key any stage reads sets a layer's box, so no "
                   "patch can move placement_ink_iou"),
    },
    "worst_local_ssim": {
        "fixable": False,
        "reason": _STRUCTURAL_NEEDS_CODE,
        "detail": ("a single worst 64x64 window is a localized reconstruction defect "
                   "(missing icon / dropped element / badge double); no config lever "
                   "targets one window"),
    },
    "native_text_ratio": {
        "fixable": False,
        "reason": _STRUCTURAL_NEEDS_CODE,
        "detail": ("native-vs-baked text is decided by text/design promotion logic from "
                   "OCR + kept_in_photo evidence; no config key overrides it per layer"),
    },
    "native_leaf_ratio": {
        "fixable": False,
        "reason": _STRUCTURAL_NEEDS_CODE,
        "detail": ("everything was rasterized: promotion to native leaves is a "
                   "reconstruct/vectorize decision. The only vectorize lever "
                   "(force_raster_fallback) pushes the WRONG way — it forces more "
                   "raster, not less. 021 proved acting here is harmful: the planner "
                   "misread this as element_recall and the sam3 rerun cost -0.40 "
                   "text_recall"),
    },
}

# qa.json hard_fail rule -> blocker name in _BLOCKER_LEVERS.
_HARD_FAIL_RULES = {
    "glyph-residue": "glyph_residue",
    "low-text-recall": "low_text_recall",
    "low-native-leaf-ratio": "native_leaf_ratio",
    "low-native-text-ratio": "native_text_ratio",
    "element-recall": "element_recall",
    "low-element-recall": "element_recall",
}


def _boxes_overlap(a: Any, b: Any) -> bool:
    """True when two {x,y,w,h} boxes share any area."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    try:
        ax, ay, aw, ah = float(a["x"]), float(a["y"]), float(a["w"]), float(a["h"])
        bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
    except (KeyError, TypeError, ValueError):
        return False
    return (ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah)


def _residue_boxes(run_dir: Optional[str]) -> list:
    """Bounding boxes of unresolved glyph-residue regions (reconstruction audit)."""
    if not run_dir:
        return []
    stats = (_load_json(os.path.join(run_dir, "reconstruction.json"), {}) or {}).get("stats") or {}
    residual = stats.get("text_residual") or {}
    out = []
    for entry in residual.get("flagged") or []:
        if isinstance(entry, dict) and not entry.get("resolved"):
            box = entry.get("box") or entry.get("bbox")
            if isinstance(box, dict):
                out.append(box)
    return out


def worst_window_is_residue(qa: Any, run_dir: Optional[str]) -> bool:
    """True when the worst local-SSIM window sits on top of unresolved glyph residue.

    postfix-benchmark-6: in 002/013/066/091 the worst 64x64 window overlaps a flagged
    residue box — the window IS the residue, not a separate structural defect. That makes
    it reachable by the clean-plate lever, so refusing it as "structural" would be wrong.
    When it does NOT overlap residue it is a genuine localized defect (missing icon /
    dropped element / badge double) with no config lever.
    """
    if not isinstance(qa, dict):
        return False
    boxes = _residue_boxes(run_dir)
    if not boxes:
        return False
    for flag in qa.get("quality_flags") or []:
        if str((flag or {}).get("rule") or "") != "local-ssim-worst-region":
            continue
        window = (flag or {}).get("bbox")
        if any(_boxes_overlap(window, box) for box in boxes):
            return True
    return False


def _blocker_names(qa: Any, reward: Any = None) -> list[str]:
    """Names of the QA/gate failures currently blocking acceptance."""
    names: list[str] = []
    if not isinstance(qa, dict):
        return names
    if _glyph_residue_unresolved(qa):
        names.append("glyph_residue")
    # Read declared hard_fails generically (qa.json and the structural sub-report both
    # carry them). 021's real blocker was "low-native-leaf-ratio", which the planner never
    # saw — it acted on an artifact-derived element_recall instead and regressed the run.
    structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
    for fail in list(qa.get("hard_fails") or []) + list(structural.get("hard_fails") or []):
        if not isinstance(fail, dict):
            continue
        mapped = _HARD_FAIL_RULES.get(str(fail.get("rule") or ""))
        if mapped:
            names.append(mapped)
    contract = qa.get("contract") if isinstance(qa.get("contract"), dict) else {}
    if contract.get("placement_ok") is False:
        names.append("placement_ink_iou")
    if contract.get("native_text_ok") is False:
        names.append("native_text_ratio")
    for flag in qa.get("quality_flags") or []:
        rule = str((flag or {}).get("rule") or "")
        if rule == "low-text-recall":
            names.append("low_text_recall")
        elif rule == "local-ssim-worst-region":
            names.append("worst_local_ssim")
    gate = ((reward or {}).get("gate") or {}) if isinstance(reward, dict) else {}
    for key, check in (gate.get("checks") or {}).items():
        if isinstance(check, dict) and check.get("ok") is False and key == "worst_local_ssim":
            names.append("worst_local_ssim")
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def diagnose_blockers(qa: Any, reward: Any = None, run_dir: Optional[str] = None) -> dict:
    """Explain every acceptance blocker as fixable-by-lever or refused-with-reason.

    Returns ``{"blockers": [...], "fixable": [...], "refused": [...], "verdict": str}``.
    ``verdict`` is ``"refuse"`` when NOTHING blocking has a lever — the honest outcome
    the bench-6 plateau rounds were hiding (002/013/066/088/091 all burned a round on a
    residue patch while placement/native-ratio blockers had no lever at all).
    """
    names = _blocker_names(qa, reward)
    residue_window = worst_window_is_residue(qa, run_dir)
    fixable: list[dict] = []
    refused: list[dict] = []
    for name in names:
        spec = _BLOCKER_LEVERS.get(name)
        if not spec:
            refused.append({"blocker": name, "reason": "unclassified blocker: no mapped lever"})
            continue
        if name == "worst_local_ssim" and residue_window:
            # The worst window sits on unresolved residue: the clean-plate lever reaches it.
            fixable.append({
                "blocker": name, "lever": "inpaint:rebuild-clean-plate",
                "detail": "worst local-SSIM window overlaps a flagged glyph-residue box",
            })
            continue
        if spec.get("fixable"):
            fixable.append({"blocker": name, "lever": spec.get("lever"),
                            "detail": spec.get("detail")})
        else:
            refused.append({"blocker": name, "reason": spec.get("reason"),
                            "detail": spec.get("detail")})
    if not names:
        verdict = "clean"
    elif fixable:
        verdict = "repair"
    else:
        verdict = "refuse"
    return {"blockers": names, "fixable": fixable, "refused": refused, "verdict": verdict}


def _escalation_level(params: Optional[dict], ladder_len: int = len(_CLEAN_PLATE_LADDER)) -> int:
    """Clamp a repair's requested escalation rung into the ladder's range."""
    try:
        level = int((params or {}).get("escalation_level") or 0)
    except (TypeError, ValueError):
        level = 0
    return max(0, min(ladder_len - 1, level))


def escalation_level_from_history(run_dir: Optional[str], stage: str, action: str) -> int:
    """How many times this (stage, action) already ran — its next ladder rung.

    Reads the admission ledger, which persists every admitted plan across rounds. Without
    this, round 2 re-planned the identical patch, hit the ``seen`` fingerprint and was
    skipped as "unchanged-repair-plan-and-inputs" -> guaranteed plateau.
    """
    if not run_dir:
        return 0
    admission = _load_json(os.path.join(run_dir, "harness_admission.json"), {})
    seen = admission.get("seen") if isinstance(admission.get("seen"), dict) else {}
    count = 0
    for entry in seen.values():
        plan = (entry or {}).get("plan") if isinstance(entry, dict) else None
        if isinstance(plan, dict) and plan.get("stage") == stage and plan.get("action") == action:
            count += 1
    return count


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


def _glyph_residue_unresolved(qa: Any) -> bool:
    """True when Agent-B glyph-residue hard-fail / contract still blocks acceptance."""
    if not isinstance(qa, dict):
        return False
    for fail in list(qa.get("hard_fails") or []):
        if isinstance(fail, dict) and fail.get("rule") == "glyph-residue":
            return True
    structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
    for fail in list(structural.get("hard_fails") or []):
        if isinstance(fail, dict) and fail.get("rule") == "glyph-residue":
            return True
    if structural.get("glyph_residue_unresolved"):
        try:
            if int(structural["glyph_residue_unresolved"]) > 0:
                return True
        except (TypeError, ValueError):
            pass
    contract = qa.get("contract") if isinstance(qa.get("contract"), dict) else {}
    if contract.get("glyph_residue_clean") is False:
        return True
    return False


def _qa_accepts(qa: Any, *, allow_summary: bool = False) -> bool:
    """Fail closed on missing, malformed, or contradictory QA summaries."""
    if not isinstance(qa, dict) or not _flag(qa.get("ok")):
        return False
    # Unresolved glyph residue is a hard structural fail (Agent B). Never declare
    # harness success over it — even when a lightweight summary omits hard_fails.
    if _glyph_residue_unresolved(qa):
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


def _flatten_design_layers(layers) -> list:
    out = []
    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        out.append(layer)
        out.extend(_flatten_design_layers(layer.get("children")))
    return out


def _layer_lookup(run_dir: Optional[str] = None, design: Optional[dict] = None) -> dict:
    """id → layer dict from design.json (or an in-memory design)."""
    if design is None and run_dir:
        design = _load_json(os.path.join(run_dir, "design.json"), {})
    if not isinstance(design, dict):
        return {}
    return {
        str(layer["id"]): layer
        for layer in _flatten_design_layers(design.get("layers") or [])
        if layer.get("id") is not None
    }


def _meta_flags(layer: Optional[dict]) -> dict:
    if not isinstance(layer, dict):
        return {}
    meta = layer.get("meta") if isinstance(layer.get("meta"), dict) else {}
    return meta


def is_baked_chrome_layer(layer: Optional[dict]) -> bool:
    """Badge/seal/chip cutouts whose OCR is intentionally baked (Agent A meta)."""
    if not isinstance(layer, dict):
        return False
    meta = _meta_flags(layer)
    if layer.get("kept_in_photo") or meta.get("kept_in_photo"):
        return True
    return bool(
        meta.get("shell_raster_chip")
        or meta.get("baked_badge_text")
        or meta.get("chrome_as_raster")
        or meta.get("suppression_reason") == "baked-chrome-text"
    )


def is_already_sliced_layer(layer: Optional[dict]) -> bool:
    """True when the layer already shipped as a confidence-gated raster slice."""
    if not isinstance(layer, dict):
        return False
    meta = _meta_flags(layer)
    try:
        from src.schema import is_raster_slice
        return is_raster_slice(meta) or is_raster_slice({"fallback": layer.get("fallback")})
    except Exception:
        fallback = meta.get("fallback") or layer.get("fallback")
        return fallback in {"raster-slice", "raster_slice", "slice"}


def _repair_target_ids(repair: dict) -> list[str]:
    ids: list[str] = []
    target = repair.get("target_id")
    if target is not None:
        ids.append(str(target))
    params = repair.get("params") or {}
    for key in ("layer_ids",):
        for value in params.get(key) or []:
            if value is not None:
                ids.append(str(value))
    for entry in params.get("regions") or []:
        if isinstance(entry, dict) and entry.get("layer_id") is not None:
            ids.append(str(entry["layer_id"]))
    # Preserve order, drop dupes.
    return list(dict.fromkeys(ids))


_BAKED_TEXT_PROMOTE_ACTIONS = frozenset({
    "restore-editable-text", "refit-text-box", "refit-colors-effects", "resolve-fonts",
})

# ── raise-floors admission (P1 gap 1 / gap 3) ────────────────────────────────────────
# A repair round should only run when a *reachable, plausibly-converting* repair exists.
# These floors refuse three classes the harness provably cannot convert:
#   element_growth_*   speculative SAM element-growth on an already-faithful render
#                      (run-4 021 regressed; run-5 002 burned 15 min for a noise delta)
#   ocr_truth_*        a low text_recall whose editable text is already correct in the
#                      design — the miss is source-OCR ground-truth, not the render, so a
#                      render/OCR rerun is a guaranteed no-op (run-4 101: edit 1.0 / tr 0.13)
#   baked_majority_frac fraction of detected lines that are scene-baked-by-design; an OCR
#                      rerun cannot lift recall when the denominator is baked-dominated
#                      (run-4 135's 32 nutrition-label lines; consumed via text_recall_detail)
_DEFAULT_ADMISSION_FLOORS = {
    "element_growth_ssim": 0.90,
    "element_growth_recall": 0.95,
    "ocr_truth_editable_recall": 0.85,
    "ocr_truth_recall_margin": 0.05,
    "baked_majority_frac": 0.5,
}


def admission_floors(cfg: Optional[dict] = None) -> dict:
    """Admission floors from ``runtime.harness.admission.floors`` (config-only, tunable)."""
    base = dict(_DEFAULT_ADMISSION_FLOORS)
    admission = (((cfg or {}).get("runtime") or {}).get("harness") or {}).get("admission") or {}
    override = admission.get("floors") if isinstance(admission.get("floors"), dict) else {}
    for key, value in override.items():
        if key in base:
            try:
                base[key] = float(value)
            except (TypeError, ValueError):
                pass
    return base


def _is_speculative_element_growth(repair: dict) -> bool:
    """A sam3 rerun that GROWS elements (element_propose / lowered confidence / no target).

    revalidate-rejected and layer-targeted reruns are excluded — those act on a specific
    known object, not a speculative sweep for a new one."""
    if (repair.get("stage"), repair.get("action")) != ("sam3", "rerun-detection"):
        return False
    params = repair.get("params") or {}
    return bool(
        params.get("enable_element_propose")
        or params.get("lower_confidence")
        or params.get("source") == "vlm_critique"
        or not params.get("layer_ids")
    )


def element_growth_refused(qa: Any, floors: Optional[dict] = None) -> Optional[str]:
    """Reason a speculative element-growth cannot convert an already-faithful render.

    A missing SAM element that leaves global SSIM high (or the text essentially complete)
    is not visible in the diff — growing it cannot raise the metric and empirically
    regresses (021). ``qa`` supplies the render's current global evidence."""
    if not isinstance(qa, dict):
        return None
    floors = floors or _DEFAULT_ADMISSION_FLOORS
    ssim = qa.get("ssim")
    if isinstance(ssim, (int, float)) and float(ssim) >= floors["element_growth_ssim"]:
        return f"element-growth-on-faithful-render:ssim={float(ssim):.3f}"
    recall = qa.get("text_recall")
    if isinstance(recall, (int, float)) and float(recall) >= floors["element_growth_recall"]:
        return f"element-growth-with-complete-text:text_recall={float(recall):.3f}"
    return None


def ocr_truth_mismatch_refused(qa: Any, floors: Optional[dict] = None) -> Optional[str]:
    """Reason a text_recall rerun is a no-op: the design already carries the correct text.

    editable_text_recall measures render-vs-DESIGN text; text_recall measures render-vs-
    SOURCE OCR. When editable recall is high but text_recall is far lower, the shortfall is
    a source-OCR ground-truth artifact (nondeterministic OCR / a source misread) that no
    render or text-stage rerun can move — 101: editable 1.0, text_recall 0.13."""
    if not isinstance(qa, dict):
        return None
    floors = floors or _DEFAULT_ADMISSION_FLOORS
    edit = qa.get("editable_text_recall")
    if not isinstance(edit, (int, float)):
        structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
        edit = structural.get("editable_text_recall")
    recall = qa.get("text_recall")
    if not isinstance(edit, (int, float)) or not isinstance(recall, (int, float)):
        return None
    if (float(edit) >= floors["ocr_truth_editable_recall"]
            and float(recall) < float(edit) - floors["ocr_truth_recall_margin"]):
        return (f"ocr-truth-mismatch:editable_recall={float(edit):.2f}"
                f">text_recall={float(recall):.2f}")
    return None


def targeted_dedup_noop_reason(run_dir: str, patches: dict) -> Optional[str]:
    """Reason a TARGETED merge:dedup provably cannot drop a layer, else None.

    postfix-benchmark-7 013 (the run-4 class): the VLM reported "duplicated text
    '% OFF'" and repair.py aimed merge:dedup at c_B2 with
    ``duplicate_text=['% OFF'], layer_ids=['c_B2']``. Nothing was duplicated — c_B2 is
    the sole ``'61% OFF'`` layer — so ``_dedup_text_candidates`` was structurally
    incapable of dropping anything, three ways over:

      * ``layer_ids[1:]`` is empty for a single target (it keeps the first, drops the
        rest), so the id path can never drop with one id;
      * ``'% OFF'`` does not normalize-equal ``'61% OFF'``, so the text path never
        keys a group;
      * a group needs ``len(group) >= 2`` — one layer can never be its own duplicate.

    The rerun still replayed merge→…→reconstruct→qa for 122.5s to reach a
    bit-identical render. Screen it at admission instead, by asking merge's REAL dedup
    (ground truth, never a reimplementation that can drift) whether it drops anything.

    Deliberately narrow, so a genuine experiment is never blocked:
      * only fires for a TARGETED dedup (``duplicate_text``/``layer_ids`` present).
        The untargeted ``raise_dedup_iou`` probe, whose whole point IS the IoU sweep,
        is left alone;
      * fails OPEN on any missing/unreadable evidence or import problem.
    """
    merge_patch = (patches or {}).get("merge")
    if not isinstance(merge_patch, dict) or not merge_patch.get("dedup_text"):
        return None
    duplicate_text = [str(text) for text in (merge_patch.get("duplicate_text") or [])
                      if str(text).strip()]
    layer_ids = [str(layer_id) for layer_id in (merge_patch.get("layer_ids") or [])
                 if layer_id]
    if not duplicate_text and not layer_ids:
        return None
    # Two or more ids: the id path CAN drop layer_ids[1:] — a real, capable repair.
    if len(layer_ids) >= 2:
        return None
    candidates = _load_json(os.path.join(run_dir, "merged.json"), None)
    if not isinstance(candidates, list) or not candidates:
        return None
    try:
        from src.merge_layers import _dedup_text_candidates, _normalize_text_key
    except Exception:
        return None
    dedup_iou = merge_patch.get("dedup_iou")
    try:
        dedup_iou = float(dedup_iou) if dedup_iou is not None else 0.6
    except (TypeError, ValueError):
        return None
    probe_cfg = {
        "dedup_text": True,
        "duplicate_text": list(duplicate_text),
        "layer_ids": list(layer_ids),
    }
    try:
        kept = _dedup_text_candidates([dict(item) for item in candidates],
                                      probe_cfg, dedup_iou)
    except Exception:
        return None
    if not isinstance(kept, list) or len(kept) < len(candidates):
        return None  # it drops something — capable, let it run
    # Provably inert. Report WHY against the real merged layers, so the audit trail
    # shows the premise was false rather than merely "it didn't help".
    wanted = {_normalize_text_key(text) for text in duplicate_text}
    wanted.discard("")
    matches = sorted(
        str(item.get("id"))
        for item in candidates
        if _normalize_text_key(item.get("text")) in wanted
    ) if wanted else []
    details = []
    if duplicate_text:
        details.append(
            f"no two layers share text {duplicate_text!r} "
            f"(normalized matches in merged.json: {matches or 'none'})"
        )
    if len(layer_ids) == 1:
        details.append(
            f"a single target {layer_ids[0]!r} cannot dedup against itself "
            "(merge keeps layer_ids[0] and drops only layer_ids[1:])"
        )
    return (
        "merge:dedup is structurally incapable of dropping a layer here — "
        + "; ".join(details)
        + "; merge's own dedup drops 0 of "
        + f"{len(candidates)} merged candidates, so the rerun would replay "
        "merge -> reconstruct -> qa for a bit-identical render"
    )


def admission_reject_reason(
    repair: dict,
    *,
    run_dir: Optional[str] = None,
    design: Optional[dict] = None,
    layers: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> Optional[str]:
    """Return a skip reason when a repair targets kept_in_photo / baked chrome / sliced.

    Workstream E: these deficits are by-design (exact cutout / baked pack copy). Re-running
    OCR/text/slice on them thrashs the harness and can re-promote baked OCR to TEXT.
    """
    if not isinstance(repair, dict):
        return "malformed-repair"
    stage = repair.get("stage")
    action = repair.get("action")
    floors = admission_floors(cfg)
    lookup = layers if layers is not None else _layer_lookup(run_dir, design)
    target_ids = _repair_target_ids(repair)

    # Gap 3 — raise floors: a speculative SAM element-growth cannot convert an already-
    # faithful render (021 regressed; 002 burned a 15-minute round for a noise delta).
    # Refuse before it starts, using the render's current global QA evidence. Targeted or
    # untargeted: the check reads global ssim/text_recall, not the (usually absent) target.
    if _is_speculative_element_growth(repair) and run_dir:
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        reason = element_growth_refused(qa, floors)
        if reason:
            return reason

    # Global OCR / editable-text restores with no target: drop when QA already attributes
    # the shortfall to kept_in_photo / baked chrome (unfixable by rerun).
    if not target_ids and stage in {"ocr", "text-analysis", "vlm"} and action in {
        "rerun", "restore-editable-text", "boost-stack", "review",
    }:
        qa = _load_json(os.path.join(run_dir, "qa.json"), {}) if run_dir else {}
        structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
        # Gap 1b — OCR-truth mismatch: the design already renders the correct editable text,
        # so a low text_recall is a source-OCR ground-truth artifact no rerun can move.
        ocr_truth = ocr_truth_mismatch_refused(qa, floors)
        if ocr_truth and action in {"rerun", "restore-editable-text", "boost-stack"}:
            return ocr_truth
        kept = qa.get("kept_in_photo_lines")
        if kept is None:
            kept = structural.get("kept_in_photo_lines")
        total = qa.get("text_lines_total")
        if total is None:
            total = structural.get("text_lines_total")
        # Gap 1a — consume pixel_diff's text_recall_detail: it already excludes verified
        # scene-baked lines from the recall denominator, listing how many. When baked lines
        # dominate the detected text, an OCR/text rerun cannot lift recall (135's 32
        # nutrition-label lines). This is the live signal on new qa.json; the legacy
        # kept_in_photo_lines/text_lines_total counts are the fallback for older artifacts.
        detail = qa.get("text_recall_detail")
        if isinstance(detail, dict):
            if kept is None:
                kept = detail.get("baked_excluded")
            if total is None:
                total = detail.get("lines_total")
        try:
            kept_n = int(kept) if kept is not None else 0
            total_n = int(total) if total is not None else 0
        except (TypeError, ValueError):
            kept_n, total_n = 0, 0
        if total_n > 0 and kept_n > 0 and kept_n >= max(1, int(floors["baked_majority_frac"] * total_n)):
            # Majority of detected lines are intentionally baked — OCR rerun cannot help.
            if action in {"rerun", "restore-editable-text", "boost-stack", "review"}:
                return "kept-in-photo-text-deficit"
        # design.kept_in_photo non-empty + restore-editable with no target: refuse promotion.
        if design is None and run_dir:
            design = _load_json(os.path.join(run_dir, "design.json"), {})
        kept_texts = list((design or {}).get("kept_in_photo") or []) if isinstance(design, dict) else []
        if action == "restore-editable-text" and kept_texts and not target_ids:
            return "baked-ocr-must-not-promote-to-text"

    if not target_ids:
        return None

    for lid in target_ids:
        layer = lookup.get(lid)
        if layer is None:
            continue
        if is_already_sliced_layer(layer):
            return f"already-sliced:{lid}"
        if is_baked_chrome_layer(layer):
            # Never re-promote baked badge/pack OCR to editable TEXT; also drop
            # reconstruct/slice thrash on an exact chrome cutout.
            if stage == "text-analysis" and action in _BAKED_TEXT_PROMOTE_ACTIONS:
                return f"baked-chrome-text:{lid}"
            if stage == "ocr":
                return f"kept-in-photo:{lid}"
            if stage in {"reconstruct", "vectorize", "qwen"} and action in {
                "inspect-worst-regions", "raster-fallback", "retry", "restage-assets",
            }:
                meta = _meta_flags(layer)
                if meta.get("shell_raster_chip") or meta.get("baked_badge_text") or meta.get("chrome_as_raster"):
                    return f"shell-raster-chip:{lid}"
                if layer.get("kept_in_photo") or meta.get("kept_in_photo"):
                    return f"kept-in-photo:{lid}"
    return None


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
        # postfix-benchmark-6: the old patch was {"mode": "auto", "allow_fallback": False}.
        # config.yaml ships inpaint.mode=flux_comfy, and for the regional router "auto" and
        # "flux_comfy" resolve to the SAME per-region engine choice, so the rerun replayed
        # a byte-identical plate: 002/013/066/091 logged metric_deltas of exactly 0.0 and
        # 088 moved edge_f1 by -0.0001. Five rounds, zero pixels changed.
        #
        # Glyph residue is a *physical* defect: the removal mask did not cover the glyph's
        # anti-aliased halo, so ink survives under re-drawn editable text. The levers that
        # actually move it are the mask footprint and the scrub pass — not the engine name.
        # Escalate along a ladder so a second round is a genuinely different experiment.
        level = _escalation_level(params)
        patches["inpaint"] = copy.deepcopy(_CLEAN_PLATE_LADDER[level])
        if params.get("mode"):
            patches["inpaint"]["mode"] = params["mode"]

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
        ranked = rank_repairs(filtered, qa)
    else:
        ranked = rank_repairs(load_repairs(run_dir, cfg), qa)
    return _stamp_escalation(ranked, run_dir)


def _stamp_escalation(repairs: list, run_dir: str) -> list:
    """Advance laddered repairs to the rung their history has earned.

    Pure planners (repair.assess) are stateless, so without this every round emits rung 0
    forever. Stamping here — the one place that has both the candidate list and run_dir —
    keeps config_patches_for a pure function of the repair record.
    """
    for repair in repairs or []:
        if not isinstance(repair, dict):
            continue
        key = (repair.get("stage"), repair.get("action"))
        if key not in _LADDERED_ACTIONS:
            continue
        params = repair.setdefault("params", {})
        if not isinstance(params, dict):
            continue
        if "escalation_level" not in params:
            params["escalation_level"] = escalation_level_from_history(
                run_dir, repair.get("stage"), repair.get("action"))
    return repairs


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


# ── render-semantic artifact comparison ──────────────────────────────────────────────
#
# postfix-benchmark-7 013 evidence: a merge:dedup rerun re-serialized design.json (its
# ``meta`` block: layer_count, single_ownership counters, warnings) and re-encoded
# preview.png. Every byte-level fingerprint flipped, so ``artifacts_changed`` said True
# — while the render was pixel-identical and every QA metric held at ssim=0.7073 /
# text_recall=0.8462. The round escaped the identical-artifact short-circuit and burned
# 122.5s to change nothing. "Did anything change" must therefore be RENDER-SEMANTIC:
# only the outcome counts — pixels, the compiled node content, and the QA metrics.

# Keys carrying diagnostics, timings, or provenance rather than rendered outcome. A
# rerun that moves only these moved nothing a viewer or a QA metric can resolve.
_DIAGNOSTIC_KEYS = frozenset({
    "meta", "diagnostics", "diagnostic", "timings", "timing", "timestamp",
    "elapsed", "elapsed_s", "duration", "duration_s", "generated_at", "created_at",
    "runtime", "telemetry", "debug", "_debug", "warnings", "log", "logs",
    "provenance",
})

# JSON round-trips jitter floats far below anything QA can resolve (its tightest
# tolerance is 0.005); quantize before comparing so re-serialization is not "a change".
_SEMANTIC_FLOAT_NDIGITS = 4

# Preview comparison is PER-PIXEL, never a whole-image mean. Repairs are local: a
# deduped layer or a refit text box touches a small patch. On 013's 1080x1920 preview a
# real 60x60 edit averages to 0.243/255 — under any sane mean epsilon — so a mean test
# would silently swallow genuine improvements (over-blocking, the opposite failure).
# Counting pixels that actually moved separates the cases cleanly: that same edit moves
# 3600 pixels, while a lossless PNG re-encode moves exactly 0.
#
# Channel delta tolerated per pixel (0-255): absorbs any dither/AA jitter without
# hiding a real repaint. Lossless re-encode measures 0 here.
_PREVIEW_CHANNEL_EPSILON = 2.0
# Moved pixels tolerated before a render counts as changed: ignores isolated speckle
# while staying far below the footprint of any repair worth keeping.
_PREVIEW_MIN_CHANGED_PIXELS = 16

_QA_TOLERANCES = {
    "ssim": 0.005, "visual_score": 0.005, "text_recall": 0.01,
    "editable_text_recall": 0.01, "edge_f1": 0.005, "color_similarity": 0.005,
}


def _semantic_json(value):
    """Outcome-bearing projection of a JSON artifact.

    Strips diagnostics/timings and normalizes key order plus float jitter. What
    survives is the content that decides the render: ids, boxes, text, fills, styles,
    and the shape of the node tree.
    """
    if isinstance(value, dict):
        return {
            key: _semantic_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [_semantic_json(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, _SEMANTIC_FLOAT_NDIGITS)
    return value


def _json_semantic_digest(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    canonical = json.dumps(_semantic_json(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _image_signature(path: str):
    """Pixel array for render comparison, or None when unreadable."""
    try:
        import numpy
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as handle:
            return numpy.asarray(handle.convert("RGB"), dtype=numpy.float32)
    except Exception:
        return None


def _outcome_fingerprint(path: str) -> tuple:
    """Render-semantic fingerprint: what this artifact contributes to the OUTCOME.

    PNGs compare as pixels (epsilon-tolerant), JSON as a diagnostics-stripped digest.
    Either falls back to raw bytes when unreadable, so an unparseable artifact is
    never silently treated as unchanged.
    """
    lowered = path.lower()
    if lowered.endswith(".png"):
        signature = _image_signature(path)
        if signature is not None:
            return ("pixels", signature)
    elif lowered.endswith(".json"):
        digest = _json_semantic_digest(path)
        if digest is not None:
            return ("json", digest)
    return ("bytes", _artifact_fingerprint(path))


def _outcome_changed(before: tuple, after: tuple) -> bool:
    """True when two outcome fingerprints differ beyond serialization/encode noise."""
    kind_before, value_before = before
    kind_after, value_after = after
    if kind_before != kind_after:
        return True
    if kind_before == "pixels":
        if value_before.shape != value_after.shape:
            return True
        try:
            import numpy
        except Exception:  # pragma: no cover - numpy present wherever pixels were read
            return True
        per_pixel = numpy.abs(value_before - value_after).max(axis=2)
        moved = int((per_pixel > _PREVIEW_CHANNEL_EPSILON).sum())
        return moved > _PREVIEW_MIN_CHANGED_PIXELS
    return value_before != value_after


def _qa_metrics_moved(deltas: dict) -> bool:
    """True when any QA metric moved beyond its noise floor (either direction)."""
    for key, minimum in _QA_TOLERANCES.items():
        delta = (deltas or {}).get(key)
        if isinstance(delta, (int, float)) and abs(float(delta)) >= minimum:
            return True
    return bool((deltas or {}).get("hard_fails"))


def _qa_progress(before: dict, after: dict) -> tuple[bool, dict]:
    """Return meaningful metric progress, not merely a rewritten qa.json."""
    before, after = before or {}, after or {}
    deltas = {}
    tolerances = _QA_TOLERANCES
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

    Structure/VLM resumes (merge → structure → layout) include ``scene_intent.json`` and
    ``merged.json`` so an unchanged planning fingerprint skips the expensive VLM grouping
    replay (CRITIC F13 / workstream E).
    """
    resume = str(choice.get("resume") or "")
    inputs = {
        "ocr": ("normalized.png",),
        "text": ("ocr_raw.json", "normalized.png"),
        "sam": ("residual.json", "qwen.json", "ocr.json"),
        "elements": ("residual.json", "qwen.json", "ocr.json"),
        "merge": ("fused_elements.json", "ocr.json", "scene_intent.json"),
        "structure": ("merged.json", "scene_intent.json"),
        "reconstruct": ("merged.json", "ocr.json", "scene_intent.json"),
        "layout": ("reconstruction.json", "merged.json", "scene_intent.json"),
        "design": ("layout.json", "reconstruction.json"),
        "figma": ("design.json",),
    }.get(resume, ("qa.json",))
    evidence = {name: _artifact_fingerprint(os.path.join(run_dir, name)) for name in inputs}
    # Prefer the planner's own fingerprint when present — byte-identical scene_intent with
    # the same planning_fingerprint is conclusive "structure/VLM inputs unchanged".
    intent = _load_json(os.path.join(run_dir, "scene_intent.json"), {})
    planning_fp = intent.get("planning_fingerprint") if isinstance(intent, dict) else None
    payload = {
        "stage": choice.get("stage"), "action": choice.get("action"),
        "target_id": choice.get("target_id"), "resume": resume,
        "patches": choice.get("patches") or {}, "inputs": evidence,
        "planning_fingerprint": planning_fp,
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

            # Workstream E: drop kept_in_photo / baked chrome / already-sliced deficits —
            # they cannot improve via OCR promote / re-slice thrash.
            baked_reason = admission_reject_reason(choice, run_dir=run_dir, cfg=cfg)
            if baked_reason:
                exhausted.add((choice.get("stage"), choice.get("action"), choice.get("target_id")))
                skipped = {
                    "iteration": iteration, "resume": resume,
                    "repair": {key: choice.get(key) for key in ("stage", "action", "target_id")},
                    "reason": "baked-or-sliced-deficit",
                    "detail": baked_reason,
                }
                attempts.append({**skipped, "admission_skipped": True})
                admission.setdefault("skipped", []).append(skipped)
                _write_json(admission_path, admission)
                continue

            # Structurally-incapable targeted dedup (013): the named duplicate does not
            # exist, so merge's dedup provably drops nothing. Ask the real dedup before
            # spending 122s proving it the expensive way.
            dedup_reason = targeted_dedup_noop_reason(run_dir, patches)
            if dedup_reason:
                exhausted.add((choice.get("stage"), choice.get("action"),
                               choice.get("target_id")))
                skipped = {
                    "iteration": iteration, "resume": resume,
                    "repair": {key: choice.get(key) for key in ("stage", "action", "target_id")},
                    "reason": "dedup-target-not-duplicated",
                    "detail": dedup_reason,
                }
                attempts.append({**skipped, "admission_skipped": True})
                admission.setdefault("skipped", []).append(skipped)
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
                # Structure/VLM resumes share the planning fingerprint — surface that reason
                # so the audit trail shows we skipped an unchanged structure plan, not a
                # generic "same repair".
                skip_reason = "unchanged-repair-plan-and-inputs"
                if resume in {"merge", "structure", "layout", "reconstruct"} and (
                        plan_payload.get("planning_fingerprint")
                        or "scene_intent.json" in (plan_payload.get("inputs") or {})):
                    skip_reason = "unchanged-structure-vlm-plan"
                skipped = {
                    "iteration": iteration, "resume": resume,
                    "repair": {key: choice.get(key) for key in ("stage", "action", "target_id")},
                    "reason": skip_reason,
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
        # Watch the resumed stage's primary outputs plus the compiled-design tail, both
        # by raw bytes and by render-semantic outcome. Only the OUTCOME decides whether
        # the repair did anything: bytes flip on any re-serialized diagnostic (013).
        stage_outputs = tuple(_STAGE_OUTPUTS.get(resume, ()))
        watched = tuple(dict.fromkeys(stage_outputs + _TAIL_OUTPUTS))
        before_watched = {
            name: _artifact_fingerprint(os.path.join(run_dir, name)) for name in watched
        }
        before_outcome = {
            name: _outcome_fingerprint(os.path.join(run_dir, name)) for name in watched
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
        bytes_changed = any(
            _artifact_fingerprint(os.path.join(run_dir, name)) != before_watched[name]
            for name in watched
        )
        after_outcome = {
            name: _outcome_fingerprint(os.path.join(run_dir, name)) for name in watched
        }
        changed_artifacts = [
            name for name in watched
            if _outcome_changed(before_outcome[name], after_outcome[name])
        ]
        render_changed = [name for name in changed_artifacts if name in _TAIL_OUTPUTS]
        stage_changed = [name for name in changed_artifacts if name in stage_outputs]
        qa_moved = _qa_metrics_moved(metric_deltas) if qa_fresh else False
        # THE outcome test. A repair changed something only if the render moved
        # (design.json node content / preview.png pixels) or a QA metric moved beyond
        # noise. A re-serialized diagnostic or a re-encoded PNG is not a change.
        artifacts_changed = bool(render_changed) or qa_moved
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
        if stage_changed:
            # Audit trail for the 013 class: the resumed stage genuinely re-emitted
            # different content, yet nothing survived to the render. Downstream stages
            # absorbed it — the repair still bought nothing.
            attempt["stage_output_changed"] = stage_changed
        if result.get("ok") and not artifacts_changed and not qa_improved:
            # Byte-identical is the narrow case; render-identical-but-bytes-differ is
            # the 013 case the byte fingerprint used to miss. Name them apart so the
            # audit trail says which one actually happened.
            attempt["no_effect"] = (
                "identical-artifacts" if not bytes_changed else "identical-render"
            )
            attempt["no_effect_detail"] = {
                "compared": list(watched),
                "render_identical": sorted(set(_TAIL_OUTPUTS) & set(watched)),
                "qa_metrics_identical": sorted(_QA_TOLERANCES),
                "bytes_changed": bytes_changed,
                "note": ("outcome unchanged: design.json node content and preview.png "
                         "pixels held, and no QA metric moved beyond its noise floor"),
            }
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
