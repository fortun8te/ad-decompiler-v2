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
import json
import os
from typing import Optional

from src.qa_config import visual_pass_ssim
from src.schema import raster_slice_failures, raster_slice_thresholds

# Canonical fallback-contract helper (F11): read meta.fallback the same way every stage
# does instead of an ad-hoc truthiness check. Guarded so repair.py stays import-safe on a
# checkout that predates the schema helper landing (a thin adapter, not a redefinition).
try:
    from src.schema import is_raster_slice as _is_raster_slice
except Exception:
    def _is_raster_slice(meta) -> bool:
        return bool(isinstance(meta, dict) and meta.get("fallback") == "raster-slice")

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
    "element_recall_min": 0.75,
    "low_conf_ocr": 0.55,
    # Per-layer blended local score (region_ssim + ink IoU / colour) below which a layer
    # is measurably broken and must steer a targeted repair (worst first).
    "local_layer_min": 0.35,
}

_TEXT_ROLES = {"text", "headline", "subheadline", "body", "cta", "price", "text-block"}


def _sev(x):
    return {"high": 3, "medium": 2, "low": 1}.get(x, 0)


def _local_layer_score(pl):
    """Blend pixel_diff per-layer evidence into one local score (same recipe as
    qa_reward.local_component): region SSIM, then rendered-ink IoU for text rows or
    local colour similarity for shape/image rows. None when the row carries no metric."""
    if not isinstance(pl, dict):
        return None
    value = pl.get("region_ssim")
    if not isinstance(value, (int, float)):
        value = pl.get("score")
    if not isinstance(value, (int, float)):
        return None
    value = max(0.0, min(1.0, float(value)))
    ink_iou = pl.get("ink_iou")
    region_color = pl.get("region_color")
    if isinstance(ink_iou, (int, float)):
        value = 0.7 * value + 0.3 * max(0.0, min(1.0, float(ink_iou)))
    elif isinstance(region_color, (int, float)):
        value = 0.85 * value + 0.15 * max(0.0, min(1.0, float(region_color)))
    return value


def _repair_badness(item):
    """Measured badness of the region/layer a repair targets (0 = no local evidence).

    Repairs that carry measured per-layer/window evidence sort ahead of vague ones at
    equal severity, so the harness attacks the worst measured failure first."""
    params = item.get("params") or {}
    worst = None
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
    if worst is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(worst)))


def repairs_from_anomalies(anomalies, design=None):
    """Translate VLM anomaly records (see src.vlm_anomaly) into actionable repairs.

    duplicate/ghosted/overlapping text -> merge dedup (drop the redundant layer);
    clipped/cut-off text               -> text-analysis refit-text-box (widen/shrink-to-fit);
    clearly wrong glyphs               -> text-analysis resolve-fonts (font substitution).

    Pure and deterministic. Layer ids, when the anomaly carries them, become the repair
    target so the resumed stage acts on the offending layer instead of the whole design.
    """
    out = []
    for anomaly in anomalies or []:
        if not isinstance(anomaly, dict):
            continue
        kind = str(anomaly.get("type") or "").strip().lower()
        text = str(anomaly.get("text") or "").strip()
        detail = str(anomaly.get("detail") or "").strip()
        layer_ids = [str(i) for i in (anomaly.get("layer_ids") or []) if i]
        label = text or detail or kind
        if kind == "duplicate_text":
            reason = f"duplicate/ghosted text {label!r}" + (
                f" across layers {', '.join(layer_ids[:4])}" if layer_ids else "")
            item = {
                "stage": "merge",
                "action": "dedup",
                "reason": reason,
                "params": {"raise_dedup_iou": True, "duplicate_text": [text] if text else [],
                           "layer_ids": layer_ids},
                "severity": "high",
            }
            if layer_ids:
                item["target_id"] = layer_ids[0]
            out.append(item)
        elif kind == "clipped_text":
            reason = f"clipped/cut-off text {label!r}"
            item = {
                "stage": "text-analysis",
                "action": "refit-text-box",
                "reason": reason,
                "params": {"widen": True, "shrink_to_fit": True,
                           "clipped_text": [text] if text else [],
                           "layer_ids": layer_ids},
                "severity": "high",
            }
            if layer_ids:
                item["target_id"] = layer_ids[0]
            out.append(item)
        elif kind == "wrong_glyphs":
            item = {
                "stage": "text-analysis",
                "action": "resolve-fonts",
                "reason": f"wrong glyphs {label!r} (likely font substitution)",
                "params": {"wrong_glyphs": [text] if text else [], "layer_ids": layer_ids},
                "severity": "medium",
            }
            if layer_ids:
                item["target_id"] = layer_ids[0]
            out.append(item)
        elif kind in {"inpaint_halo", "inpaint_patch"}:
            out.append({
                "stage": "inpaint",
                "action": "rebuild-clean-plate",
                "reason": f"visible {kind.replace('_', ' ')}: {label}",
                "params": {"score_candidates": True, "color_match": True,
                           "halo_review": kind == "inpaint_halo"},
                "severity": "high",
            })
    return out


def _load_json(path: str, fallback):
    if not path or not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return fallback


def _flatten_layers(layers):
    out = []
    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        out.append(layer)
        out.extend(_flatten_layers(layer.get("children")))
    return out


def _element_recall_from_artifacts(run_dir: str):
    """Approximate detection recall: fused proposals that survived reconstruction."""
    elements = _load_json(os.path.join(run_dir, "elements.json"), [])
    if not elements:
        sam = _load_json(os.path.join(run_dir, "sam3.json"), {})
        elements = sam.get("elements") or []
    recon = _load_json(os.path.join(run_dir, "reconstruction.json"), {})
    candidates = [
        candidate for candidate in recon.get("candidates", []) or []
        if isinstance(candidate, dict) and candidate.get("target") != "drop"
    ]
    proposed = [
        element for element in elements or []
        if isinstance(element, dict) and (element.get("meta") or {}).get("role") != "background"
    ]
    if not proposed:
        return None
    proposed_ids = {str(element.get("id")) for element in proposed if element.get("id")}
    kept_ids = {str(candidate.get("id")) for candidate in candidates if candidate.get("id")}
    if not proposed_ids:
        return None
    return len(proposed_ids & kept_ids) / len(proposed_ids)


def _resolve_element_recall(qa, structural, run_dir):
    recall = qa.get("element_recall")
    if recall is None:
        recall = structural.get("element_recall")
    if recall is None and run_dir:
        recall = _element_recall_from_artifacts(run_dir)
    return None if recall is None else float(recall)


def _vlm_rejected_segments(design, run_dir):
    rejected = []
    for layer in _flatten_layers((design or {}).get("layers") or []):
        meta = layer.get("meta") or {}
        if meta.get("vlm_rejected"):
            rejected.append(str(layer.get("id") or "unknown"))
    if run_dir:
        recon = _load_json(os.path.join(run_dir, "reconstruction.json"), {})
        for candidate in recon.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            if (candidate.get("meta") or {}).get("vlm_rejected"):
                rejected.append(str(candidate.get("id") or "unknown"))
    return sorted(set(rejected))


def _staging_failure(qa, structural, run_dir):
    staging_error = qa.get("staging_error")
    staged = qa.get("staged")
    if staging_error is None:
        staging = qa.get("staging") or structural.get("staging") or {}
        if isinstance(staging, dict):
            staging_error = staging.get("staging_error")
            if staged is None:
                staged = staging.get("staged")
    if staging_error is None and run_dir:
        report = _load_json(os.path.join(run_dir, "runtime_report.json"), {})
        staging_error = report.get("staging_error")
        if staged is None:
            staged = report.get("staged")
    if staging_error:
        return staging_error
    if staged is False:
        return "Figma inbox staging did not complete"
    return None


def _unclean_background_signal(structural):
    background = structural.get("background") or {}
    if not background:
        return None
    if background.get("exact_match_ratio", 0) > 0.995 and background.get("changed_ratio", 1) < 0.01:
        return "background plate still matches source inside removal region"
    for failure in structural.get("hard_fails", []) or []:
        if isinstance(failure, dict) and failure.get("rule") in (
            "background-leakage", "unclean-background",
        ):
            return None
    layers = structural.get("layers") or []
    for layer in layers:
        meta = (layer or {}).get("meta") or {}
        if meta.get("role") == "background" and meta.get("source") != "inpaint":
            return "background layer is not sourced from the clean inpaint plate"
    return None


def assess(design, qa, ocr, cfg: Optional[dict] = None):
    cfg = cfg or {}
    # Qwen is an optional, separately hosted layer proposal service.  A failed visual
    # score must never turn it on just because it happens to be mentioned in an old
    # repair recipe: that produces a predictable offline request and wastes a repair
    # iteration.  Explicitly enabled installations still retain Qwen retries.
    qwen_enabled = bool((cfg.get("qwen") or {}).get("enabled", False))
    run_dir = cfg.get("run_dir")
    if qwen_enabled and run_dir:
        try:
            note = open(os.path.join(run_dir, "qwen.note.txt"), encoding="utf-8").read().lower()
            if any(marker in note for marker in (
                "backend offline", "backend likely down", "connection refused",
                # Validation failures are deterministic until the workflow/models change.
                # Repeating the same upload and rejected prompt cannot improve the scene.
                "/prompt failed", "prompt_outputs_failed_validation", "validation=",
            )):
                qwen_enabled = False
        except OSError:
            pass
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
                # Keep the repair in-process. Surya is intentionally not part of the RTX
                # install and an external agent previously interpreted it as a Docker job.
                "params": {"upscale": True, "use_configured_ocr": True,
                           "vlm_ocr_judge": True},
                "severity": "high" if text_recall < 0.6 else "medium",
            }
        )
        if text_recall < 0.25:
            out.append(
                {
                    "stage": "vlm",
                    "action": "boost-stack",
                    "reason": f"near-zero text recall {text_recall:.2f}; use scene-text alternative",
                    "params": {"focus": "text"},
                    "severity": "medium",
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

    run_dir = cfg.get("run_dir")

    # ── element recall (detection → reconstruction survival) ────────────────────────────
    element_recall = _resolve_element_recall(qa, structural, run_dir)
    if element_recall is not None and element_recall < t["element_recall_min"]:
        out.append(
            {
                "stage": "sam3",
                "action": "rerun-detection",
                "reason": f"element recall {element_recall:.2f} < {t['element_recall_min']:.2f}",
                "params": {"lower_confidence": True, "enable_element_propose": True},
                "severity": "high" if element_recall < 0.5 else "medium",
            }
        )

    # ── VLM-rejected segments still present in the design graph ───────────────────────
    rejected_segments = _vlm_rejected_segments(design, run_dir)
    if rejected_segments and not any(
        item.get("action") == "revalidate-rejected" for item in out
    ):
        out.append(
            {
                "stage": "sam3",
                "action": "revalidate-rejected",
                "reason": f"{len(rejected_segments)} VLM-rejected segment(s): "
                            + ", ".join(rejected_segments[:4]),
                "params": {"disable_segment_filter": True, "lower_confidence": True},
                "severity": "high",
            }
        )

    # ── Figma bridge staging failures (QA can pass while inbox write fails) ─────────────
    staging_detail = _staging_failure(qa, structural, run_dir)
    if staging_detail and not any(item.get("action") == "restage-inbox" for item in out):
        out.append(
            {
                "stage": "figma",
                "action": "restage-inbox",
                "reason": staging_detail,
                "severity": "high",
            }
        )

    # ── unclean background without an explicit hard-fail record ───────────────────────
    roots = (design or {}).get("layers") or []
    if roots and (roots[0].get("meta") or {}).get("source") != "inpaint":
        if not any(
            item.get("action") == "rebuild-clean-plate" for item in out
        ):
            out.append(
                {
                    "stage": "inpaint",
                    "action": "rebuild-clean-plate",
                    "reason": "background is not the reconstructed inpaint plate",
                    "severity": "high",
                }
            )
    elif run_dir:
        unclean = _unclean_background_signal(structural)
        if unclean and not any(item.get("action") == "rebuild-clean-plate" for item in out):
            out.append(
                {
                    "stage": "inpaint",
                    "action": "rebuild-clean-plate",
                    "reason": unclean,
                    "severity": "high",
                }
            )

    # ── hard fails carry explicit rules ───────────────────────────────────────────────
    for hf in hard_fails:
        rule = hf.get("rule", "")
        detail = hf.get("detail", "")
        if rule in ("background-leakage", "unclean-background"):
            out.append({"stage": "inpaint", "action": "rebuild-clean-plate", "reason": detail,
                        "severity": "high"})
        elif rule == "inpaint-outside-mask":
            out.append({"stage": "inpaint", "action": "rebuild-clean-plate", "reason": detail,
                        "params": {"strict_mask_composite": True}, "severity": "high"})
        elif rule in ("layer-alpha-holes", "empty-layer-alpha"):
            target = None
            if str(detail).startswith("layer "):
                target = str(detail).split()[1]
            item = {"stage": "sam3", "action": "rerun-detection", "reason": detail,
                    "params": {"lower_confidence": False, "enable_element_propose": True,
                               "reject_internal_holes": True},
                    "severity": "high"}
            if target:
                item["target_id"] = target
            out.append(item)
        elif rule == "missing-assets":
            out.append({"stage": "reconstruct", "action": "restage-assets", "reason": detail,
                        "severity": "high"})
        elif rule in ("staging-failed", "figma-staging-failed", "staging-failure"):
            out.append({"stage": "figma", "action": "restage-inbox", "reason": detail,
                        "severity": "high"})
        elif rule in ("low-element-recall", "element-recall"):
            out.append({"stage": "sam3", "action": "rerun-detection", "reason": detail,
                        "params": {"lower_confidence": True, "enable_element_propose": True},
                        "severity": "high"})
        elif rule in ("vlm-rejected-segments", "vlm-rejected"):
            out.append({"stage": "sam3", "action": "revalidate-rejected", "reason": detail,
                        "params": {"disable_segment_filter": True, "lower_confidence": True},
                        "severity": "high"})
        elif rule == "invalid-schema":
            out.append({"stage": "design", "action": "rebuild-schema", "reason": detail,
                        "severity": "high"})
        elif rule == "native-accounting-missing":
            out.append({"stage": "design", "action": "rebuild-schema", "reason": detail,
                        "severity": "high"})
        elif rule == "unexplained-raster-fallback":
            out.append({
                "stage": "sam3", "action": "rerun-detection", "reason": detail,
                "params": {"lower_confidence": False, "enable_element_propose": True},
                "severity": "high",
            })
        elif rule.endswith("-unavailable"):
            component = rule[: -len("-unavailable")]
            if component == "sam3":
                out.append({"stage": "sam3", "action": "rerun-detection", "reason": detail,
                            "params": {"lower_confidence": True}, "severity": "high"})
            elif component == "ocr":
                out.append({"stage": "ocr", "action": "rerun", "reason": detail,
                            "params": {"upscale": True}, "severity": "high"})
            elif component == "inpaint":
                out.append({"stage": "inpaint", "action": "rebuild-clean-plate", "reason": detail,
                            "severity": "high"})
            else:
                out.append({"stage": "pipeline", "action": "review",
                            "reason": f"{rule}: {detail}", "severity": "medium"})
        elif rule in ("local-ssim-worst-region", "local-ssim-worst-window"):
            # The worst measured local window is a concrete, boxed failure — it must map
            # to an actionable, targeted reconstruction pass, not a manual "pipeline
            # review" (002: worst window ssim 0.009 at x=704,y=512 got filed as a
            # medium review while the harness chased a vague VLM opinion instead).
            region = {"box": hf.get("bbox") or hf.get("box"), "reasons": [detail]}
            worst_window = qa.get("local_ssim_worst_window") or {}
            if isinstance(worst_window.get("ssim"), (int, float)):
                region["region_ssim"] = worst_window["ssim"]
                region["local_score"] = worst_window["ssim"]
                if not region.get("box"):
                    region["box"] = worst_window.get("bbox")
            out.append({
                "stage": "reconstruct", "action": "inspect-worst-regions",
                "reason": detail,
                "params": {"regions": [region], "worst_window": True},
                "severity": "high",
            })
        elif rule in ("local-ssim", "edge-fidelity", "color-fidelity"):
            if rule == "local-ssim":
                out.append({"stage": "qwen", "action": "retry", "reason": detail,
                            "severity": "medium"})
            elif rule == "edge-fidelity":
                out.append({"stage": "layout", "action": "refit-geometry", "reason": detail,
                            "severity": "medium"})
            else:
                out.append({"stage": "text-analysis", "action": "refit-colors-effects",
                            "reason": detail, "severity": "medium"})
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

    # ── confidence-gated raster-slice fallback (Codia-style fidelity floor) ────────────
    # A region the pipeline cannot reconstruct faithfully should ship as a pixel-exact
    # source slice rather than as visible garbage: 'looks right + partially editable'
    # beats 'looks wrong + fully editable'. The repair maps to the already-actionable
    # (reconstruct, inspect-worst-regions) resume; reconstruct.apply_raster_slice_fallback
    # honors the layer ids via the reconstruct.focus_regions config patch.
    fallback_thresholds = raster_slice_thresholds(cfg)
    slice_ids: set = set()
    if fallback_thresholds.get("enabled", True):
        slice_rows = []
        for pl in per_layer:
            # Skip layers ALREADY shipped as a pixel-exact raster slice (re-gating them just
            # wastes a harness round). Other fallback kinds are still evaluated — an ad-hoc
            # "any truthy fallback" skip hid text→image substitutions from re-gating (F11).
            if not isinstance(pl, dict) or _is_raster_slice({"fallback": pl.get("fallback")}):
                continue
            reasons = raster_slice_failures(pl, fallback_thresholds)
            if reasons:
                slice_rows.append((pl, reasons))
        # Worst first: the repair target and the region order must follow the measured
        # per-layer scores, not the incidental per_layer row order.
        slice_rows.sort(key=lambda row: _local_layer_score(row[0])
                        if _local_layer_score(row[0]) is not None else 1.0)
        slice_ids = {str(pl.get("id")) for pl, _ in slice_rows}
        if slice_rows:
            regions = [{
                "layer_id": pl.get("id"),
                "region_ssim": pl.get("region_ssim"),
                "region_color": pl.get("region_color"),
                "ink_iou": pl.get("ink_iou"),
                "box": pl.get("abs_box") or pl.get("box"),
                "reasons": reasons,
            } for pl, reasons in slice_rows[:8]]
            out.append({
                "stage": "reconstruct",
                "action": "inspect-worst-regions",
                "target_id": str(slice_rows[0][0].get("id")),
                "reason": (
                    f"raster-slice fallback: {len(slice_rows)} region(s) below the "
                    "confidence gate — replace with pixel-exact source slices: "
                    + ", ".join(sorted(slice_ids))
                ),
                "params": {"raster_slice": True, "regions": regions},
                "severity": "high",
            })

    # ── unresolved glyph residue under removed text (reconstruction audit) ────────────
    if run_dir:
        recon_stats = (_load_json(os.path.join(run_dir, "reconstruction.json"), {})
                       .get("stats") or {})
        residual = recon_stats.get("text_residual") or {}
        unresolved = [entry for entry in residual.get("flagged") or []
                      if isinstance(entry, dict) and not entry.get("resolved")]
        if unresolved:
            out.append({
                "stage": "inpaint",
                "action": "rebuild-clean-plate",
                "reason": (f"glyph residue remains under {len(unresolved)} removed text "
                           "region(s): "
                           + ", ".join(str(entry.get("id")) for entry in unresolved[:4])),
                "params": {"strict_mask_composite": True},
                "severity": "high",
            })

    # ── worst measured layers steer targeted repairs (worst first) ─────────────────────
    # pixel_diff writes region_ssim/ink_iou/region_color per layer; a layer whose blended
    # local score collapses is a measured, boxed failure and outranks whole-image
    # opinions. 002: CTA 0.112, €49 0.185, strike 0.191, €63 0.310 were all ignored while
    # the harness acted on a vague VLM "misaligned" note.
    measured_rows = []
    for pl in per_layer:
        if not isinstance(pl, dict):
            continue
        lid = str(pl.get("id") or "")
        role = str(pl.get("role") or pl.get("type") or "")
        if not lid or lid in slice_ids or role == "background":
            continue
        local = _local_layer_score(pl)
        if local is None or local >= t["local_layer_min"]:
            continue
        measured_rows.append((local, pl))
    measured_rows.sort(key=lambda row: row[0])
    for local, pl in measured_rows:
        lid = str(pl.get("id"))
        role = str(pl.get("role") or pl.get("type") or "")
        box = pl.get("abs_box") or pl.get("box")
        region_evidence = {
            "layer_id": lid,
            "region_ssim": pl.get("region_ssim"),
            "region_color": pl.get("region_color"),
            "ink_iou": pl.get("ink_iou"),
            "box": box,
            "local_score": round(local, 4),
        }
        if pl.get("type") == "text" or role in _TEXT_ROLES:
            out.append({
                "stage": "text-analysis",
                "action": "refit-text-box",
                "target_id": lid,
                "reason": (f"text layer {lid} local score {local:.3f} < "
                           f"{t['local_layer_min']:.2f} (worst measured)"),
                "params": {"widen": True, "shrink_to_fit": True, "layer_ids": [lid],
                           "box": box, "region": region_evidence},
                "severity": "high",
            })
        else:
            out.append({
                "stage": "reconstruct",
                "action": "inspect-worst-regions",
                "target_id": lid,
                "reason": (f"layer {lid} local score {local:.3f} < "
                           f"{t['local_layer_min']:.2f} (worst measured)"),
                "params": {"regions": [region_evidence]},
                "severity": "high",
            })

    # ── per-layer diagnostics ─────────────────────────────────────────────────────────
    for pl in per_layer:
        lid = pl.get("id")
        if str(lid) in slice_ids:
            # The raster-slice repair supersedes generic per-layer suggestions for this
            # region; re-running qwen/build on a region already gated to a slice wastes
            # a harness round.
            continue
        score = pl.get("score")
        role = pl.get("role") or pl.get("type")
        recall = pl.get("recall")
        if recall is not None and role not in (None, "text", "headline", "body", "cta", "text-block"):
            if recall < t["element_recall_min"]:
                out.append(
                    {
                        "stage": "sam3",
                        "action": "rerun-detection",
                        "target_id": lid,
                        "reason": f"element {lid} recall {recall:.2f} < {t['element_recall_min']:.2f}",
                        "params": {"lower_confidence": True, "enable_element_propose": True},
                        "severity": "medium",
                    }
                )
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

    # ── low-confidence + disagreeing OCR lines (from the OCR artifact directly) ─────────
    # Text baked into element-owned product/raster regions (packaging labels, nutrition
    # panels) is not part of the editable-ad contract: swapping the product image swaps
    # that text.  Its low confidence and cross-backend disagreement are expected noise and
    # must not trigger an OCR rerun/review.  Exclude those lines up front using the same
    # ownership signal the pipeline already produces (elements.json), so repair, QA and the
    # cross-check metric all count one canonical editable-disagreement set.
    lines = ocr.get("lines", []) if isinstance(ocr, dict) else (ocr or [])
    from src import ocr as _ocr_mod
    _product_boxes = []
    if run_dir:
        _elements = _load_json(os.path.join(run_dir, "elements.json"), [])
        _product_boxes = _ocr_mod.product_region_boxes(_elements)

    def _editable(seq):
        if not _product_boxes:
            return list(seq)
        return [l for l in seq if not _ocr_mod.line_in_product_region(l, _product_boxes)]

    low = _editable(l for l in lines if float(l.get("conf", 1.0)) < t["low_conf_ocr"])
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
    # disagreement flags from challenger reconciliation (editable text only)
    disagree = _ocr_mod.disagreement_lines(lines, product_boxes=_product_boxes)
    if disagree:
        out.append(
            {
                "stage": "ocr",
                "action": "review",
                "reason": f"{len(disagree)} line(s) with backend disagreement",
                "severity": "low",
            }
        )

    # ── rendered-output anomalies (VLM pass) ──────────────────────────────────────────
    # The metric QA never reads the compiled ad, so duplicate/ghosted text, clipped text,
    # and mojibake glyphs only surface via src.vlm_anomaly. Its findings (from qa.anomalies
    # or anomalies.json) become merge-dedup / text-refit / resolve-fonts repairs here so the
    # harness resumes the right stage.
    anomalies = qa.get("anomalies") if isinstance(qa, dict) else None
    if not anomalies and run_dir:
        anomalies = _load_json(os.path.join(run_dir, "anomalies.json"), None)
        if isinstance(anomalies, dict):
            anomalies = anomalies.get("anomalies")
    if isinstance(anomalies, list) and anomalies:
        out.extend(repairs_from_anomalies(anomalies, design))

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
    # Severity first, then measured local badness: at equal severity the repair that
    # targets the worst measured region/layer is acted on first (worst first).
    out.sort(key=lambda r: (_sev(r.get("severity")), _repair_badness(r)), reverse=True)

    if not qwen_enabled:
        # Keep the repair actionable when Qwen is unavailable.  Reconstruction is the
        # native SAM/residual route and has no external ComfyUI dependency.
        for item in out:
            if item.get("stage") == "qwen" and item.get("action") == "retry":
                item["stage"] = "reconstruct"
                item["action"] = "inspect-worst-regions"
                item["reason"] = f"Qwen disabled; {item.get('reason', 'inspect visual layer mismatch')}"
                item["params"] = {"qwen_disabled": True}

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
