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

# thresholds (overridable via cfg.repair)
DEFAULTS = {
    "text_recall_min": 0.85,
    "ssim_min": 0.80,
    "composite_min": 85.0,
    "layer_score_min": 0.80,
    "vectorize_score_min": 0.90,
    "low_conf_ocr": 0.55,
}


def _sev(x):
    return {"high": 3, "medium": 2, "low": 1}.get(x, 0)


def assess(design, qa, ocr, cfg: Optional[dict] = None):
    cfg = cfg or {}
    t = dict(DEFAULTS)
    t.update(cfg.get("repair") or {})
    out = []

    qa = qa or {}
    hard_fails = qa.get("hard_fails", []) or []
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
        if "overlap" in rule:
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
