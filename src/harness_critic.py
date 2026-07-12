"""harness_critic.py — deterministic post-QA failure analysis for the repair harness.

Reads run artifacts after QA fails, scores failure categories, writes critic.json,
and filters contradictory or low-confidence repairs via critic_review().
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from src.agent_debug import tail as agent_debug_tail
from src.error_messages import detect_failed_stage
from src.harness import is_actionable
from src.qa_config import visual_pass_ssim

CATEGORIES = ("ocr", "text", "sam", "inpaint", "layout", "staging")

STAGE_TO_CATEGORY = {
    "ocr": "ocr",
    "text-analysis": "text",
    "text": "text",
    "qwen": "sam",
    "sam": "sam",
    "sam3": "sam",
    "inpaint": "inpaint",
    "layout": "layout",
    "reconstruct": "staging",
    "design": "staging",
    "build": "staging",
    "figma": "staging",
    "merge": "staging",
    "vectorize": "staging",
    "pipeline": "staging",
}

RULE_TO_CATEGORY = {
    "background-leakage": "inpaint",
    "unclean-background": "inpaint",
    "inpaint-outside-mask": "inpaint",
    "layer-alpha-holes": "sam",
    "empty-layer-alpha": "sam",
    "low-element-recall": "sam",
    "missing-assets": "staging",
    "missing-fonts": "text",
    "figma-compiler-errors": "staging",
    "low-editable-ratio": "staging",
    "no-editable-content": "staging",
    "missing-editable-text": "text",
    "duplicate-ownership": "staging",
    "local-ssim": "layout",
    "edge-fidelity": "layout",
    "color-fidelity": "text",
    "duplicate-text": "staging",
    "clipped-text": "text",
    "wrong-glyphs": "text",
}

# Rendered-output anomaly types (src.vlm_anomaly) → failure category + score weight.
ANOMALY_CATEGORY = {
    "duplicate_text": ("staging", 0.5),
    "clipped_text": ("text", 0.5),
    "wrong_glyphs": ("text", 0.45),
}

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
REPAIR_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}

THRESHOLDS = {
    "text_recall_min": 0.85,
    "editable_text_recall_min": 0.80,
    "edge_f1_min": 0.68,
    "color_similarity_min": 0.82,
    "layer_score_min": 0.80,
    "composite_min": 85.0,
}

LOG_TAIL_LINES = 80
LOW_CONFIDENCE_CUTOFF = 0.45

_INFRA_BLOCKER = re.compile(
    r"cudnn|cuda|docker|ModuleNotFoundError|no configured OCR backend",
    re.I,
)


def _load_json(path: str, fallback: Any) -> Any:
    if not path or not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return fallback


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)


def _tail_log(run_dir: str, limit: int = LOG_TAIL_LINES) -> list[str]:
    path = os.path.join(run_dir, "pipeline.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[-limit:]]
    except OSError:
        return []


def _category_for_stage(stage: str | None) -> str:
    if not stage:
        return "staging"
    return STAGE_TO_CATEGORY.get(stage, "staging")


def fix_id(repair: dict) -> str:
    stage = repair.get("stage") or "unknown"
    action = repair.get("action") or "unknown"
    target = repair.get("target_id")
    if target:
        return f"{stage}:{action}:{target}"
    return f"{stage}:{action}"


def _severity_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "medium"
    if score >= 0.15:
        return "low"
    return "none"


def _bump(bucket: dict[str, Any], amount: float, evidence: str) -> None:
    bucket["score"] = min(1.0, bucket["score"] + amount)
    if evidence and evidence not in bucket["evidence"]:
        bucket["evidence"].append(evidence)


def _empty_scores() -> dict[str, dict[str, Any]]:
    return {
        cat: {"score": 0.0, "severity": "none", "evidence": []}
        for cat in CATEGORIES
    }


def _score_categories(
    qa: dict,
    runtime: dict,
    log_tail: list[str],
    agent_debug: list[dict],
) -> dict[str, dict[str, Any]]:
    scores = _empty_scores()
    structural = qa.get("structural") or {}
    hard_fails = list(qa.get("hard_fails") or [])
    seen = {(h.get("rule"), h.get("detail")) for h in hard_fails if isinstance(h, dict)}
    for failure in structural.get("hard_fails") or []:
        key = (failure.get("rule"), failure.get("detail")) if isinstance(failure, dict) else None
        if key and key not in seen:
            hard_fails.append(failure)
            seen.add(key)

    pass_ssim = visual_pass_ssim({})
    t = dict(THRESHOLDS)
    t["ssim_min"] = pass_ssim

    text_recall = qa.get("text_recall")
    if text_recall is not None and text_recall < t["text_recall_min"]:
        gap = (t["text_recall_min"] - text_recall) / max(t["text_recall_min"], 0.01)
        _bump(scores["ocr"], min(1.0, 0.35 + gap), f"text_recall {text_recall:.2f}")

    editable_text_recall = qa.get("editable_text_recall", structural.get("editable_text_recall"))
    if editable_text_recall is not None and editable_text_recall < t["editable_text_recall_min"]:
        gap = (t["editable_text_recall_min"] - editable_text_recall) / max(t["editable_text_recall_min"], 0.01)
        _bump(scores["text"], min(1.0, 0.4 + gap), f"editable_text_recall {editable_text_recall:.2f}")

    color_similarity = qa.get("color_similarity")
    if color_similarity is not None and color_similarity < t["color_similarity_min"]:
        gap = (t["color_similarity_min"] - color_similarity) / max(t["color_similarity_min"], 0.01)
        _bump(scores["text"], min(1.0, 0.3 + gap), f"color_similarity {color_similarity:.2f}")

    edge_f1 = qa.get("edge_f1")
    if edge_f1 is not None and edge_f1 < t["edge_f1_min"]:
        gap = (t["edge_f1_min"] - edge_f1) / max(t["edge_f1_min"], 0.01)
        _bump(scores["layout"], min(1.0, 0.35 + gap), f"edge_f1 {edge_f1:.2f}")

    ssim = qa.get("ssim")
    if ssim is not None and ssim < t["ssim_min"]:
        gap = (t["ssim_min"] - ssim) / max(t["ssim_min"], 0.01)
        _bump(scores["sam"], min(1.0, 0.25 + gap), f"ssim {ssim:.2f} (layering)")

    visual_score = qa.get("visual_score")
    if visual_score is not None and visual_score < t["ssim_min"]:
        gap = (t["ssim_min"] - visual_score) / max(t["ssim_min"], 0.01)
        _bump(scores["staging"], min(1.0, 0.2 + gap), f"visual_score {visual_score:.2f}")

    composite = qa.get("composite")
    if composite is not None and composite < t["composite_min"]:
        gap = (t["composite_min"] - composite) / max(t["composite_min"], 1.0)
        _bump(scores["staging"], min(1.0, 0.15 + gap), f"composite {composite:.1f}")

    for layer in qa.get("per_layer") or []:
        score = layer.get("score")
        if score is None or score >= t["layer_score_min"]:
            continue
        lid = layer.get("id", "?")
        role = layer.get("role") or layer.get("type") or "layer"
        if layer.get("alpha_noise") or layer.get("ghost"):
            _bump(scores["sam"], 0.35, f"layer {lid} alpha/ghost matte")
        elif role in ("icon", "shape") and layer.get("vectorized"):
            _bump(scores["staging"], 0.25, f"layer {lid} vector trace score {score:.2f}")
        elif role in ("image", "photo"):
            _bump(scores["sam"], 0.3, f"layer {lid} noisy alpha score {score:.2f}")
        else:
            _bump(scores["staging"], 0.2, f"layer {lid} score {score:.2f}")

    for hf in hard_fails:
        if not isinstance(hf, dict):
            continue
        rule = str(hf.get("rule") or "")
        detail = str(hf.get("detail") or rule)
        cat = RULE_TO_CATEGORY.get(rule)
        if not cat:
            if "overlap" in rule:
                cat = "staging"
            elif "text" in rule:
                cat = "ocr"
            elif "alpha" in rule or "matte" in rule:
                cat = "sam"
            else:
                cat = "staging"
        weight = 0.65 if rule in (
            "background-leakage", "unclean-background", "inpaint-outside-mask",
            "layer-alpha-holes", "empty-layer-alpha",
        ) else 0.45
        _bump(scores[cat], weight, f"{rule}: {detail}")

    for item in runtime.get("degraded") or []:
        component = str(item.get("component") or "")
        reason = str(item.get("reason") or component)
        cat = _category_for_stage(component.replace("sam3", "sam"))
        weight = 0.55 if item.get("required") else 0.25
        _bump(scores[cat], weight, f"degraded {component}: {reason}")

    for violation in runtime.get("violations") or []:
        rule = str(violation.get("rule") or "")
        detail = str(violation.get("detail") or rule)
        component = rule.split("-")[0] if "-" in rule else "staging"
        cat = _category_for_stage(component.replace("sam3", "sam"))
        _bump(scores[cat], 0.7, f"violation {rule}: {detail}")

    error_blob = " ".join(
        part for part in (
            runtime.get("error"),
            *(line for line in log_tail if "ERROR:" in line),
        )
        if part
    )
    failed_stage = detect_failed_stage(
        None,
        error_text=error_blob,
        agent_debug=agent_debug,
    )
    if failed_stage:
        cat = _category_for_stage(failed_stage)
        _bump(scores[cat], 0.5, f"pipeline failed during {failed_stage}")

    for line in log_tail:
        if "sam3[" in line and ("failed" in line.lower() or "degraded" in line.lower()):
            _bump(scores["sam"], 0.4, line.strip()[:160])
        if "inpaint" in line.lower() and ("fallback" in line.lower() or "opencv" in line.lower()):
            _bump(scores["inpaint"], 0.35, line.strip()[:160])
        if "ocr[" in line and ("failed" in line.lower() or "0 lines" in line.lower()):
            _bump(scores["ocr"], 0.4, line.strip()[:160])

    for entry in agent_debug:
        location = str(entry.get("location") or "")
        message = str(entry.get("message") or "")
        if "failed" not in message.lower() and "error" not in message.lower():
            continue
        for needle, stage in (
            ("ocr.py", "ocr"),
            ("text_analysis", "text"),
            ("sam3_detect", "sam"),
            ("qwen_worker", "sam"),
            ("reconstruct", "inpaint"),
            ("layout", "layout"),
            ("figma_import", "staging"),
            ("build_design_json", "staging"),
        ):
            if needle in location:
                _bump(scores[_category_for_stage(stage)], 0.35, f"{location}: {message}")
                break

    for cat, bucket in scores.items():
        bucket["score"] = round(min(1.0, bucket["score"]), 3)
        bucket["severity"] = _severity_from_score(bucket["score"])
    return scores


def _collect_blockers(
    qa: dict,
    runtime: dict,
    repairs: list[dict],
    log_tail: list[str],
    agent_debug: list[dict],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    error_blob = " ".join(
        part for part in (
            runtime.get("error"),
            *(line for line in log_tail if "ERROR:" in line),
        )
        if part
    )
    if _INFRA_BLOCKER.search(error_blob):
        failed = detect_failed_stage(None, error_text=error_blob, agent_debug=agent_debug) or "ocr"
        blockers.append({
            "category": _category_for_stage(failed),
            "reason": "infrastructure failure prevents automatic repair",
            "detail": error_blob[:240],
            "auto_fixable": False,
        })

    for violation in runtime.get("violations") or []:
        if not violation.get("hard"):
            continue
        rule = str(violation.get("rule") or "")
        component = rule.split("-")[0] if "-" in rule else "staging"
        blockers.append({
            "category": _category_for_stage(component.replace("sam3", "sam")),
            "reason": rule,
            "detail": str(violation.get("detail") or rule),
            "auto_fixable": False,
        })

    for repair in repairs:
        if repair.get("action") == "review" or not is_actionable(repair):
            blockers.append({
                "category": _category_for_stage(repair.get("stage")),
                "reason": "manual review required",
                "detail": str(repair.get("reason") or ""),
                "fix_id": fix_id(repair),
                "auto_fixable": False,
            })

    for hf in qa.get("hard_fails") or []:
        if not isinstance(hf, dict):
            continue
        rule = str(hf.get("rule") or "")
        if rule == "figma-compiler-errors":
            blockers.append({
                "category": "staging",
                "reason": rule,
                "detail": str(hf.get("detail") or rule),
                "auto_fixable": False,
            })

    if runtime.get("status") == "failed" and runtime.get("error"):
        failed = detect_failed_stage(None, error_text=str(runtime["error"]), agent_debug=agent_debug)
        if failed and _INFRA_BLOCKER.search(str(runtime["error"])):
            blockers.append({
                "category": _category_for_stage(failed),
                "reason": "runtime failure",
                "detail": str(runtime["error"])[:240],
                "auto_fixable": False,
            })

    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for item in blockers:
        key = (item.get("category"), item.get("reason"), item.get("fix_id"), item.get("detail"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _repair_confidence(repair: dict, scores: dict[str, dict[str, Any]]) -> float:
    base = {3: 0.9, 2: 0.7, 1: 0.5}.get(REPAIR_SEVERITY_RANK.get(repair.get("severity"), 0), 0.35)
    cat = _category_for_stage(repair.get("stage"))
    cat_score = float((scores.get(cat) or {}).get("score") or 0.0)
    if cat_score > 0.2:
        base = min(1.0, base + cat_score * 0.15)
    if repair.get("action") == "review":
        base *= 0.25
    if not is_actionable(repair):
        base *= 0.15
    return round(base, 3)


def _prioritized_issues(scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    for category in CATEGORIES:
        bucket = scores[category]
        if bucket["score"] <= 0:
            continue
        issues.append({
            "category": category,
            "severity": bucket["severity"],
            "score": bucket["score"],
            "summary": bucket["evidence"][0] if bucket["evidence"] else f"{category} failure",
            "evidence": list(bucket["evidence"]),
        })
    issues.sort(
        key=lambda item: (SEVERITY_RANK.get(item["severity"], 0), item["score"]),
        reverse=True,
    )
    return issues


def _suggested_fix_ids(repairs: list[dict], scores: dict[str, dict[str, Any]]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for repair in repairs:
        if not is_actionable(repair):
            continue
        confidence = _repair_confidence(repair, scores)
        if confidence < LOW_CONFIDENCE_CUTOFF:
            continue
        ranked.append((confidence, fix_id(repair)))
    ranked.sort(reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _, fid in ranked:
        if fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
    return out


def _detect_anomalies(run_dir: str, qa: dict, cfg: Optional[dict]) -> list[dict]:
    """Run the optional VLM anomaly pass (safe, capped). Reuse qa.anomalies if present."""
    existing = qa.get("anomalies") if isinstance(qa, dict) else None
    if isinstance(existing, list) and existing:
        return [a for a in existing if isinstance(a, dict)]
    try:
        from src import vlm_anomaly
        if not vlm_anomaly.enabled(cfg):
            return []
        return vlm_anomaly.detect_anomalies(run_dir, cfg)
    except Exception:
        return []


def _score_anomalies(scores: dict[str, dict[str, Any]], anomalies: list[dict]) -> None:
    for anomaly in anomalies or []:
        if not isinstance(anomaly, dict):
            continue
        kind = str(anomaly.get("type") or "").strip().lower()
        mapping = ANOMALY_CATEGORY.get(kind)
        if not mapping:
            continue
        category, weight = mapping
        label = str(anomaly.get("text") or anomaly.get("detail") or kind)[:120]
        _bump(scores[category], weight, f"anomaly {kind}: {label}")
    for bucket in scores.values():
        bucket["score"] = round(min(1.0, bucket["score"]), 3)
        bucket["severity"] = _severity_from_score(bucket["score"])


def analyze(run_dir: str, *, write: bool = True, cfg: Optional[dict] = None) -> dict:
    """Analyze a run directory after QA failure and optionally write critic.json."""
    run_dir = os.path.abspath(run_dir)
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    repairs = _load_json(os.path.join(run_dir, "repairs.json"), None)
    if not isinstance(repairs, list):
        repairs = list(qa.get("repairs") or [])
    runtime = _load_json(os.path.join(run_dir, "runtime_report.json"), {})
    log_tail = _tail_log(run_dir)
    agent_debug = agent_debug_tail(run_dir, cfg=cfg)

    scores = _score_categories(qa, runtime, log_tail, agent_debug)
    anomalies = _detect_anomalies(run_dir, qa, cfg)
    if anomalies:
        _score_anomalies(scores, anomalies)
    blockers = _collect_blockers(qa, runtime, repairs, log_tail, agent_debug)
    issues = _prioritized_issues(scores)
    suggested = _suggested_fix_ids(repairs, scores)

    output = {
        "run_dir": run_dir,
        "qa_ok": bool(qa.get("ok")),
        "scores": scores,
        "prioritized_issues": issues,
        "suggested_fix_ids": suggested,
        "blockers": blockers,
        "anomalies": anomalies,
        "sources": {
            "qa": os.path.exists(os.path.join(run_dir, "qa.json")),
            "repairs": os.path.exists(os.path.join(run_dir, "repairs.json")),
            "runtime_report": os.path.exists(os.path.join(run_dir, "runtime_report.json")),
            "pipeline_log": os.path.exists(os.path.join(run_dir, "pipeline.log")),
            "agent_debug": bool(agent_debug),
            "anomalies": bool(anomalies),
        },
    }
    if write:
        _write_json(os.path.join(run_dir, "critic.json"), output)
    return output


def _blocked_categories(blockers: list[dict]) -> set[str]:
    blocked = set()
    for item in blockers:
        if item.get("auto_fixable") is False and not item.get("fix_id"):
            blocked.add(str(item.get("category") or "staging"))
    return blocked


def _blocked_fix_ids(blockers: list[dict]) -> set[str]:
    return {str(item["fix_id"]) for item in blockers if item.get("fix_id")}


def _same_fix(a: dict, b: dict) -> bool:
    return (
        a.get("stage") == b.get("stage")
        and a.get("action") == b.get("action")
        and a.get("target_id") == b.get("target_id")
    )


def _contradicts(existing: dict, candidate: dict, scores: dict[str, dict[str, Any]]) -> bool:
    stage_a = existing.get("stage")
    stage_b = candidate.get("stage")
    action_a = existing.get("action")
    action_b = candidate.get("action")

    pair = {(stage_a, action_a), (stage_b, action_b)}
    if pair == {("merge", "dedup"), ("merge", "enforce-single-owner")}:
        return scores.get("staging", {}).get("score", 0) >= scores.get("sam", {}).get("score", 0)

    visual_pair = {("inpaint", "rebuild-clean-plate"), ("layout", "refit-geometry")}
    if pair == visual_pair:
        inpaint_score = float(scores.get("inpaint", {}).get("score") or 0)
        layout_score = float(scores.get("layout", {}).get("score") or 0)
        if inpaint_score >= layout_score + 0.1:
            return (stage_b, action_b) == ("layout", "refit-geometry")
        if layout_score >= inpaint_score + 0.1:
            return (stage_b, action_b) == ("inpaint", "rebuild-clean-plate")

    sam_blocked = float(scores.get("sam", {}).get("score") or 0) < 0.2
    if sam_blocked and stage_b == "qwen" and action_b == "retry":
        return True
    return False


def critic_review(repairs: list[dict], critic_output: dict) -> list[dict]:
    """Filter repairs using critic output — drop blockers and contradictions."""
    repairs = list(repairs or [])
    if not repairs:
        return []

    scores = critic_output.get("scores") or {}
    blockers = critic_output.get("blockers") or []
    blocked_categories = _blocked_categories(blockers)
    blocked_ids = _blocked_fix_ids(blockers)

    sorted_repairs = sorted(
        repairs,
        key=lambda item: (
            _repair_confidence(item, scores),
            REPAIR_SEVERITY_RANK.get(item.get("severity"), 0),
        ),
        reverse=True,
    )

    kept: list[dict] = []
    for repair in sorted_repairs:
        fid = fix_id(repair)
        if fid in blocked_ids:
            continue
        cat = _category_for_stage(repair.get("stage"))
        if cat in blocked_categories:
            continue
        confidence = _repair_confidence(repair, scores)
        if confidence < LOW_CONFIDENCE_CUTOFF:
            continue
        if any(_same_fix(existing, repair) for existing in kept):
            continue
        if any(_contradicts(existing, repair, scores) for existing in kept):
            continue
        kept.append(repair)

    return kept
