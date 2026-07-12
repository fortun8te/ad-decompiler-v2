"""repair.py — stage 12: rule-based repair suggestions the orchestrating agent acts on.

assess(design, qa, ocr, cfg) reads qa.hard_fails + qa.per_layer + coarse OCR signals
and emits a list of actionable suggestions, e.g.:

  {stage:'ocr',       action:'rerun',           reason:'text_recall 0.60', params:{upscale:True}}
  {stage:'qwen',      action:'retry',           reason:'layer alpha noisy'}
  {stage:'vectorize', action:'raster-fallback', target_id:'E3', reason:'trace score 0.71'}

Pure and deterministic — no model, no I/O beyond the optional artifact write. Safe to
import anywhere (stdlib only). Suggestions are ordered by descending severity so the
agent can act on the highest-impact repair first.
"""
from __future__ import annotations
import importlib
import os
from typing import Optional

from src.qa_config import visual_pass_ssim

# thresholds (overridable via cfg.repair)
DEFAULTS = {
    "text_recall_min": 0.85,
    "editable_text_recall_min": 0.80,
    "edge_f1_min": 0.68,
    "color_similarity_min": 0.82,
    "editable_ratio_min": 0.15,
    "composite_min": 85.0,
    "layer_score_min": 0.80,
    "vectorize_score_min": 0.90,
    "low_conf_ocr": 0.55,
}


def _sev(x):
    return {"high": 3, "medium": 2, "low": 1}.get(x, 0)


def assess(design, qa, ocr, cfg: Optional[dict] = None):
    cfg = cfg or {}
    pass_ssim = visual_pass_ssim(cfg)
    t = dict(DEFAULTS)
    t["ssim_min"] = pass_ssim
    t["visual_score_min"] = pass_ssim
    t.update(cfg.get("repair") or {})
    out = []

    qa = qa or {}
    structural = qa.get("structural", {}) or {}
    hard_fails = list(qa.get("hard_fails", []) or [])
    # Some callers append/replace top-level hard failures after pixel_diff.compare(). Keep
    # the structural copy authoritative too, so missing assets/leakage cannot disappear.
    seen_hard = {(h.get("rule"), h.get("detail")) for h in hard_fails if isinstance(h, dict)}
    for failure in structural.get("hard_fails", []) or []:
        key = (failure.get("rule"), failure.get("detail")) if isinstance(failure, dict) else None
        if key and key not in seen_hard:
            hard_fails.append(failure)
            seen_hard.add(key)
    per_layer = qa.get("per_layer", []) or []

    # ── global text recall ────────────────────────────────────────────────────────────
    text_recall = qa.get("text_recall")
    if text_recall is not None and text_recall < t["text_recall_min"]:
        out.append(
            {
                "stage": "ocr",
                "action": "rerun",
                "reason": f"text_recall {text_recall:.2f} < {t['text_recall_min']}",
                "params": {"upscale": True, "challengers": ["surya"]},
                "severity": "high" if text_recall < 0.6 else "medium",
            }
        )

    editable_text_recall = qa.get("editable_text_recall")
    if editable_text_recall is None:
        editable_text_recall = structural.get("editable_text_recall")
    if editable_text_recall is not None and editable_text_recall < t["editable_text_recall_min"]:
        out.append(
            {
                "stage": "text-analysis",
                "action": "restore-editable-text",
                "reason": f"editable text recall {editable_text_recall:.2f} < "
                          f"{t['editable_text_recall_min']:.2f}",
                "severity": "high",
            }
        )

    # ── structural similarity / composite ─────────────────────────────────────────────
    ssim = qa.get("ssim")
    if ssim is not None and ssim < t["ssim_min"]:
        out.append(
            {
                "stage": "qwen",
                "action": "retry",
                "reason": f"ssim {ssim:.2f} < {t['ssim_min']} (layering likely off)",
                "params": {"layers": (cfg.get("qwen") or {}).get("layers", 8)},
                "severity": "medium",
            }
        )
    visual_score = qa.get("visual_score")
    if visual_score is not None and visual_score < t["visual_score_min"]:
        out.append(
            {
                "stage": "reconstruct",
                "action": "inspect-worst-regions",
                "reason": f"visual score {visual_score:.2f} < {t['visual_score_min']:.2f}",
                "params": {"regions": ((qa.get("per_region") or {}).get("worst") or [])[:4]},
                "severity": "medium",
            }
        )
    edge_f1 = qa.get("edge_f1")
    if edge_f1 is not None and edge_f1 < t["edge_f1_min"]:
        out.append(
            {
                "stage": "layout",
                "action": "refit-geometry",
                "reason": f"edge fidelity {edge_f1:.2f} < {t['edge_f1_min']:.2f}",
                "severity": "medium",
            }
        )
    color_similarity = qa.get("color_similarity")
    if color_similarity is not None and color_similarity < t["color_similarity_min"]:
        out.append(
            {
                "stage": "text-analysis",
                "action": "refit-colors-effects",
                "reason": f"color fidelity {color_similarity:.2f} < "
                          f"{t['color_similarity_min']:.2f}",
                "severity": "medium",
            }
        )
    composite = qa.get("composite")
    if composite is not None and composite < t["composite_min"]:
        out.append(
            {
                "stage": "pipeline",
                "action": "review",
                "reason": f"composite {composite:.1f} < {t['composite_min']}",
                "severity": "low",
            }
        )

    # ── hard fails carry explicit rules ───────────────────────────────────────────────
    for hf in hard_fails:
        rule = hf.get("rule", "")
        detail = hf.get("detail", "")
        if rule in ("background-leakage", "unclean-background"):
            out.append({"stage": "inpaint", "action": "rebuild-clean-plate", "reason": detail,
                        "severity": "high"})
        elif rule == "missing-assets":
            out.append({"stage": "reconstruct", "action": "restage-assets", "reason": detail,
                        "severity": "high"})
        elif rule == "missing-fonts":
            out.append({"stage": "text-analysis", "action": "resolve-fonts", "reason": detail,
                        "severity": "high"})
        elif rule == "figma-compiler-errors":
            out.append({"stage": "figma", "action": "fix-compiler-report", "reason": detail,
                        "severity": "high"})
        elif rule in ("low-editable-ratio", "no-editable-content"):
            out.append({"stage": "design", "action": "restore-native-nodes", "reason": detail,
                        "severity": "high"})
        elif rule == "missing-editable-text":
            out.append({"stage": "text-analysis", "action": "restore-editable-text", "reason": detail,
                        "severity": "high"})
        elif rule == "duplicate-ownership":
            out.append({"stage": "merge", "action": "enforce-single-owner", "reason": detail,
                        "severity": "high"})
        elif "overlap" in rule:
            out.append({"stage": "merge", "action": "dedup", "reason": detail,
                        "params": {"raise_dedup_iou": True}, "severity": "high"})
        elif "text" in rule:
            out.append({"stage": "ocr", "action": "rerun", "reason": detail,
                        "params": {"upscale": True}, "severity": "high"})
        elif "alpha" in rule or "matte" in rule:
            out.append({"stage": "qwen", "action": "retry", "reason": detail,
                        "severity": "medium"})
        else:
            out.append({"stage": "pipeline", "action": "review",
                        "reason": f"{rule}: {detail}", "severity": "medium"})

    # Consume structural scalars even when a caller did not ask pixel_diff to convert them
    # into hard-fail records.
    editable_ratio = structural.get("editable_ratio")
    if editable_ratio is not None and editable_ratio < t["editable_ratio_min"]:
        out.append(
            {
                "stage": "design",
                "action": "restore-native-nodes",
                "reason": f"editable ratio {editable_ratio:.2f} < {t['editable_ratio_min']:.2f}",
                "severity": "high",
            }
        )
    if structural.get("duplicate_ownership") and not any(
        r.get("action") == "enforce-single-owner" for r in out
    ):
        out.append(
            {
                "stage": "merge",
                "action": "enforce-single-owner",
                "reason": f"{len(structural['duplicate_ownership'])} duplicate ownership conflict(s)",
                "severity": "high",
            }
        )

    # ── per-layer diagnostics ─────────────────────────────────────────────────────────
    for pl in per_layer:
        lid = pl.get("id")
        score = pl.get("score")
        role = pl.get("role") or pl.get("type")
        if score is not None and score < t["layer_score_min"]:
            if role in ("icon", "shape") and pl.get("vectorized"):
                out.append(
                    {
                        "stage": "vectorize",
                        "action": "raster-fallback",
                        "target_id": lid,
                        "reason": f"trace score {score:.2f} < {t['vectorize_score_min']}",
                        "severity": "medium",
                    }
                )
            elif role in ("image", "photo"):
                out.append(
                    {
                        "stage": "qwen",
                        "action": "retry",
                        "target_id": lid,
                        "reason": f"layer alpha noisy (score {score:.2f})",
                        "severity": "medium",
                    }
                )
            else:
                out.append(
                    {
                        "stage": "build",
                        "action": "review",
                        "target_id": lid,
                        "reason": f"layer {lid} score {score:.2f}",
                        "severity": "low",
                    }
                )
        if pl.get("alpha_noise") or pl.get("ghost"):
            out.append(
                {
                    "stage": "qwen",
                    "action": "retry",
                    "target_id": lid,
                    "reason": "translucent/ghost matte -> rect fallback candidate",
                    "severity": "medium",
                }
            )

    # ── low-confidence OCR lines (from the OCR artifact directly) ──────────────────────
    lines = ocr.get("lines", []) if isinstance(ocr, dict) else (ocr or [])
    low = [l for l in lines if float(l.get("conf", 1.0)) < t["low_conf_ocr"]]
    if low:
        out.append(
            {
                "stage": "ocr",
                "action": "rerun",
                "reason": f"{len(low)} low-confidence line(s): "
                + ", ".join(repr(l.get("text", ""))[:20] for l in low[:3]),
                "params": {"upscale": True},
                "severity": "low",
            }
        )
    # disagreement flags from challenger reconciliation
    disagree = [l for l in lines if (l.get("meta") or {}).get("disagreement")]
    if disagree:
        out.append(
            {
                "stage": "ocr",
                "action": "review",
                "reason": f"{len(disagree)} line(s) with backend disagreement",
                "severity": "low",
            }
        )

    # One underlying failure can arrive both as a scalar and a hard-fail record. Remove exact
    # duplicate repair actions while retaining distinct evidence/reasons.
    unique = []
    seen = set()
    for item in out:
        key = (item.get("stage"), item.get("action"), item.get("target_id"), item.get("reason"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    out = unique
    out.sort(key=lambda r: _sev(r.get("severity")), reverse=True)

    run_dir = cfg.get("run_dir")
    if run_dir:
        try:
            schema = importlib.import_module("src.schema")
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            schema = importlib.import_module("schema")
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(out, os.path.join(run_dir, "repairs.json"))
    return out


if __name__ == "__main__":  # CPU-safe smoke
    design = {"layers": []}
    qa = {
        "ok": False,
        "composite": 78.0,
        "ssim": 0.72,
        "text_recall": 0.6,
        "hard_fails": [{"rule": "overlap", "detail": "E2 overlaps E5"}],
        "per_layer": [
            {"id": "E3", "role": "icon", "vectorized": True, "score": 0.71},
            {"id": "E7", "role": "photo", "score": 0.6, "ghost": True},
        ],
    }
    ocr = {"lines": [{"id": "L0", "text": "blurry", "conf": 0.4}]}
    for r in assess(design, qa, ocr, {}):
        print(f"[{r['severity']:>6}] {r['stage']:<9} {r['action']:<16} {r['reason']}")
