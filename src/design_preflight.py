"""Design preflight: predict QA hard-fails BEFORE the expensive render/QA stage.

build_design_json writes a structural design_preflight.json (compile warnings only).
That preflight was green on runs QA then hard-failed, so it was useless as a gate.
This module enriches design_preflight.json IN PLACE with evidence-based predictions
computed from artifacts that already exist at design time (design.json, ocr.json,
background_clean.png, removal_mask.png, removal_ownership.png, reconstruction.json)
— no render needed:

  * empty-group           — reuses pixel_diff._has_empty_group (021 junk-group shape)
  * excessive-plate-destruction — the SAME fair destruction accounting QA gates on
    (pixel_diff._background_audit: destroyed = altered plate pixels with no real
    re-rendered removal-ledger owner)
  * unexplained-raster-fallback — leaf accounting already in design.meta
  * missing-editable-text — pixel_diff._text_editability is render-free; the QA gate
    is fully computable at design time (incl. the scene-baked exemption mirror)
  * low-text-recall       — an UPPER BOUND on render text recall: a source line that is
    neither native text, nor carried by a raster layer, nor verbatim in the plate or in
    an emitted asset cannot appear in any render
  * placement-drift       — mean box-IoU between native text leaves and their matching
    source OCR lines; badly displaced text predicts contract placement / worst-window
    failures

Every prediction names the QA rule it predicts, so a preflight-red run can be stopped
(or repaired) before render/QA spends its budget. Predictions are conservative: each
one only fires on evidence that the corresponding QA gate will also see.
"""
from __future__ import annotations

import json
import os

from src import pixel_diff
from src.pixel_diff import DEFAULT_THRESHOLDS

PREFLIGHT_PLACEMENT_IOU_MIN = 0.25


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _thresholds(run_dir, cfg=None):
    opts = dict(DEFAULT_THRESHOLDS)
    arch = _load_json(os.path.join(run_dir, "archetype.json"), {}) or {}
    preset = (arch.get("preset") or {}).get("thresholds") or {}
    for key in ("text_recall_min", "editable_text_recall_min",
                "background_changed_ratio_max"):
        if preset.get(key) is not None:
            try:
                opts[key] = float(preset[key])
            except (TypeError, ValueError):
                pass
    qa_cfg = (cfg or {}).get("qa") or {}
    for key in ("background_changed_ratio_max", "editable_text_recall_min",
                "text_recall_min"):
        if qa_cfg.get(key) is not None:
            opts[key] = qa_cfg[key]
    pf_cfg = (cfg or {}).get("preflight") or {}
    opts["preflight_placement_iou_min"] = float(
        pf_cfg.get("placement_iou_min", PREFLIGHT_PLACEMENT_IOU_MIN))
    return opts


def _box_iou(a, b):
    ax0, ay0 = float(a.get("x", 0)), float(a.get("y", 0))
    ax1, ay1 = ax0 + float(a.get("w", 0)), ay0 + float(a.get("h", 0))
    bx0, by0 = float(b.get("x", 0)), float(b.get("y", 0))
    bx1, by1 = bx0 + float(b.get("w", 0)), by0 + float(b.get("h", 0))
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _scene_baked_exemption(run_dir, design, source_lines):
    """Mirror of the QA scene-baked exemption (pixel_diff._structural_audit)."""
    merge_report = _load_json(os.path.join(run_dir, "merge_report.json"), {}) or {}
    if not merge_report.get("photographic_scene_text"):
        return False
    norm = pixel_diff._norm
    kept_norm = [norm(t) for t in ((design or {}).get("kept_in_photo") or [])]
    src_norm = [norm(l.get("text", "")) for l in source_lines
                if l.get("conf", 1) >= 0.5 and len(norm(l.get("text", ""))) >= 3]
    if not (src_norm and all(
            any(s == k or s in k or k in s for k in kept_norm) for s in src_norm)):
        return False
    return not pixel_diff._scene_baked_exemption_block_reason(run_dir, design)


def _placement_check(design, source_lines):
    """Mean box-IoU between native text leaves and matching source OCR lines."""
    norm = pixel_diff._norm
    lines = [(norm(l.get("text", "")), l.get("box") or {}) for l in source_lines
             if l.get("conf", 1) >= 0.5 and len(norm(l.get("text", ""))) >= 3]
    ious = []
    for leaf, box in pixel_diff._iter_leaf_layers_abs((design or {}).get("layers") or []):
        if leaf.get("type") != "text":
            continue
        text = norm(leaf.get("text"))
        if len(text) < 3:
            continue
        candidates = [b for t, b in lines if t and (t in text or text in t)]
        if not candidates:
            continue
        ious.append(max(_box_iou(box, b) for b in candidates))
    if not ious:
        return None, []
    mean_iou = sum(ious) / len(ious)
    return round(mean_iou, 4), ious


def _text_presence_bound(run_dir, design, source_lines, source_gray):
    """Estimate render text recall without a render, from ownership evidence.

    A line is renderable only when it ships as native text, is carried by a raster
    text layer, or its pixels are verbatim in the plate or in an emitted asset. A line
    with none of those owners cannot appear in any render of this design (that part IS
    a hard bound). Kept-in-photo lines verbatim in their asset leave the denominator
    (same fairness as QA's _text_recall_detail), which makes the ratio an estimate —
    QA's render-OCR measurement may exclude a different subset."""
    import numpy as np

    norm = pixel_diff._norm
    layers = pixel_diff._flatten_layers((design or {}).get("layers") or [])
    editable_blob = " ".join(
        norm(l.get("text")) for l in layers if l.get("type") == "text")
    raster_line_ids = set()
    raster_blob_parts = []
    for layer in layers:
        meta = layer.get("meta") or {}
        if layer.get("type") != "image":
            continue
        for line_id in meta.get("line_ids") or []:
            raster_line_ids.add(str(line_id))
        carried = norm(layer.get("text") or meta.get("source_text"))
        if carried:
            raster_blob_parts.append(carried)
    raster_blob = " ".join(raster_blob_parts)
    kept_blob = " ".join(norm(t) for t in ((design or {}).get("kept_in_photo") or []))
    baked_leaves = pixel_diff._baked_line_leaves(design, source_gray)
    plate_gray = None
    plate_path = os.path.join(run_dir, "background_clean.png")
    if source_gray is not None and os.path.exists(plate_path):
        h, w = source_gray.shape[:2]
        plate_rgb = pixel_diff._load_rgb(plate_path, size=(w, h))
        plate_gray = (plate_rgb[..., 0] * 0.299 + plate_rgb[..., 1] * 0.587
                      + plate_rgb[..., 2] * 0.114)
    asset_cache = {}
    total = present = excluded = 0
    missing = []
    for line in source_lines:
        text = norm(line.get("text", ""))
        if line.get("conf", 1) < 0.5 or len(text) < 3:
            continue
        total += 1
        if text in editable_blob:
            present += 1
            continue
        if str(line.get("id") or "") in raster_line_ids or (
                raster_blob and text in raster_blob):
            present += 1
            continue
        box = line.get("box") or {}
        # Verbatim in the plate: the base layer will show these pixels in every render.
        if plate_gray is not None and source_gray is not None:
            clipped = pixel_diff._clip_box(box, *source_gray.shape[1::-1])
            if clipped is not None:
                x0, y0, x1, y1 = clipped
                delta = np.abs(source_gray[y0:y1, x0:x1].astype(np.float32)
                               - plate_gray[y0:y1, x0:x1].astype(np.float32))
                if float(delta.mean()) <= 2.0:
                    present += 1
                    continue
        # Verbatim in an emitted product/photo asset.
        baked = False
        try:
            baked = pixel_diff._line_baked_in_asset(
                box, source_gray, baked_leaves, run_dir, asset_cache)
        except Exception:
            baked = False
        if baked:
            if kept_blob and text in kept_blob:
                excluded += 1  # by-design baked scene text: out of the denominator
            else:
                present += 1  # pixels will render, even if OCR may fumble them
            continue
        missing.append(str(line.get("text"))[:60])
    denominator = total - excluded
    bound = 1.0 if denominator <= 0 else (present / denominator)
    return {"lines_total": total, "present": present,
            "baked_excluded": excluded, "unowned_lines": missing,
            "recall_estimate": round(bound, 4)}


def run(run_dir, cfg=None):
    """Compute QA predictions and merge them into run_dir/design_preflight.json."""
    thresholds = _thresholds(run_dir, cfg)
    design = _load_json(os.path.join(run_dir, "design.json"), {}) or {}
    source_ocr = _load_json(os.path.join(run_dir, "ocr.json"), {}) or {}
    source_lines = source_ocr.get("lines") or []
    source_gray = None
    norm_path = os.path.join(run_dir, "normalized.png")
    if os.path.exists(norm_path):
        rgb = pixel_diff._load_rgb(norm_path)
        source_gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    else:
        rgb = None

    predicted = []
    checks = {}

    # 1) Empty structural junk groups (021 false-pass shape). QA has no direct
    # empty-group hard-fail — its effect is blocking the scene-baked exemption, which the
    # editable-text prediction below already accounts for (_scene_baked_exemption reuses
    # pixel_diff's block-reason check). So this is a WARNING, never a gate stricter than
    # the QA it predicts.
    has_empty = pixel_diff._has_empty_group(design.get("layers") or [])
    checks["empty_group"] = bool(has_empty)
    empty_group_warning = None
    if has_empty:
        empty_group_warning = {
            "code": "empty-group",
            "detail": "design contains group/frame node(s) with zero leaf descendants "
                      "— structural junk; blocks the scene-baked editability exemption",
        }

    # 2) Fair plate-destruction fraction — the exact accounting QA gates on.
    background = None
    plate_path = os.path.join(run_dir, "background_clean.png")
    mask_path = os.path.join(run_dir, "removal_mask.png")
    if rgb is not None and os.path.exists(plate_path):
        background = pixel_diff._background_audit(
            rgb, plate_path, mask_path if os.path.exists(mask_path) else None,
            run_dir=run_dir, design=design)
    if background:
        checks["plate"] = {
            "changed_canvas_ratio": background.get("changed_canvas_ratio"),
            "destroyed_canvas_ratio": background.get("destroyed_canvas_ratio"),
            "claimed_canvas_ratio": background.get("claimed_canvas_ratio"),
            "owners_unclaimed": (background.get("removal_claims") or {}).get(
                "owners_unclaimed"),
        }
        ceiling = thresholds.get("background_changed_ratio_max")
        destroyed = background.get("destroyed_canvas_ratio")
        if (background.get("mask_supplied") and destroyed is not None
                and ceiling is not None and destroyed > float(ceiling)):
            predicted.append({
                "rule": "excessive-plate-destruction",
                "detail": f"{destroyed:.1%} of the canvas altered with no re-rendered "
                          f"owner (> {float(ceiling):.0%})",
            })

    # 3) Unexplained raster fallbacks (accounting already in design.meta).
    leaf_accounting = ((design.get("meta") or {}).get("leaf_accounting")) or {}
    unexplained = int(leaf_accounting.get("unexplained_raster_count", 0) or 0)
    checks["unexplained_raster_count"] = unexplained
    if unexplained and thresholds.get("enforce_native_leaf_accounting", True):
        ids = ", ".join(str(v) for v in
                        (leaf_accounting.get("unexplained_raster_ids") or [])[:4])
        predicted.append({
            "rule": "unexplained-raster-fallback",
            "detail": f"{unexplained} raster fallback(s) without a recorded reason: {ids}",
        })

    # 4) Editable-text recall — QA's gate is render-free, so preflight can run it as-is.
    text_editability = pixel_diff._text_editability(
        source_ocr, design, pixel_diff._flatten_layers(design.get("layers") or []),
        source_gray=source_gray)
    exempt = _scene_baked_exemption(run_dir, design, source_lines)
    checks["scene_baked_exemption"] = bool(exempt)
    if text_editability is not None:
        etr = text_editability["editable_text_recall"]
        checks["editable_text_recall"] = etr
        checks["editable_text_fraction"] = text_editability.get("editable_text_fraction")
        checks["all_source_text_baked"] = text_editability.get("all_source_text_baked")
        floor = thresholds.get("editable_text_recall_min")
        # etr is None when editable recall is undefined (every line baked-by-design, denom==0).
        # That is the no-editable-content case, not a low-recall miss; predict it only when the
        # bake is NOT an allowed photographic scene, mirroring the QA gate in pixel_diff.
        if etr is None:
            if text_editability.get("all_source_text_baked") and not exempt:
                predicted.append({
                    "rule": "no-editable-content",
                    "detail": "every readable source line was baked into a photo layer "
                              "(0 editable text nodes) with no photographic-scene verdict",
                })
        elif floor is not None and not exempt and etr < float(floor):
            predicted.append({
                "rule": "missing-editable-text",
                "detail": f"editable text recall {etr:.2f} < {float(floor):.2f}",
            })

    # 5) Render text-recall upper bound: unowned lines cannot appear in any render.
    presence = _text_presence_bound(run_dir, design, source_lines, source_gray)
    checks["text_presence"] = presence
    recall_min = thresholds.get("text_recall_min")
    if (recall_min is not None and presence["lines_total"]
            and presence["recall_estimate"] < float(recall_min)):
        sample = ", ".join(presence["unowned_lines"][:4])
        predicted.append({
            "rule": "low-text-recall",
            "detail": f"estimated render recall {presence['recall_estimate']:.2f} "
                      f"< {float(recall_min):.2f} — {len(presence['unowned_lines'])} "
                      f"source line(s) have no owner anywhere in the design: {sample}",
        })

    # 6) Placement drift of native text vs its source OCR lines.
    placement_iou, _ = _placement_check(design, source_lines)
    checks["placement_box_iou"] = placement_iou
    if (placement_iou is not None
            and placement_iou < thresholds["preflight_placement_iou_min"]):
        predicted.append({
            "rule": "placement-drift",
            "detail": f"mean native-text box IoU vs source OCR {placement_iou:.2f} < "
                      f"{thresholds['preflight_placement_iou_min']:.2f} — predicts "
                      "contract placement / worst-window failures",
        })

    # Merge into the structural preflight build_design_json wrote.
    pf_path = os.path.join(run_dir, "design_preflight.json")
    preflight = _load_json(pf_path, None)
    if not isinstance(preflight, dict):
        preflight = {"ok": True, "warnings": []}
    structural_warnings = [
        w for w in (preflight.get("warnings") or [])
        if not str((w or {}).get("code", "")).startswith("qa-predict:")
        and str((w or {}).get("code", "")) != "empty-group"]
    # Idempotent: structural_ok mirrors build_design_json's own ``ok = not warnings``
    # over the structural warnings only, so re-running the prediction pass can never
    # feed its previous verdict back into itself.
    structural_ok = not structural_warnings
    warnings = list(structural_warnings)
    if empty_group_warning:
        warnings.append(empty_group_warning)
    for item in predicted:
        warnings.append({"code": f"qa-predict:{item['rule']}", "detail": item["detail"]})
    preflight["warnings"] = warnings
    preflight["qa_predictions"] = {
        "predicted_hard_fails": predicted,
        "checks": checks,
    }
    preflight["structural_ok"] = structural_ok
    preflight["ok"] = structural_ok and not predicted
    try:
        with open(pf_path, "w", encoding="utf-8") as fh:
            json.dump(preflight, fh, indent=2)
    except Exception:
        pass
    return preflight
