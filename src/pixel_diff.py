"""Visual and structural QA for source-image reconstructions.

The legacy gate used one whole-image SSIM number.  A copied source image therefore scored
perfectly even when it contained no editable reconstruction.  :func:`compare` now keeps the
backward-compatible ``ssim`` field but defines it as a local, multi-scale score and adds edge,
colour, editable-text, asset, font, ownership, and clean-background checks.

All dependencies are CPU-side and imported lazily.  The original five positional arguments
remain valid; the additional QA inputs are optional keyword arguments.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.qa_config import DEFAULT_VISUAL_PASS_SSIM


DEFAULT_THRESHOLDS = {
    "local_ssim_min": DEFAULT_VISUAL_PASS_SSIM,
    "edge_f1_min": 0.68,
    "color_similarity_min": 0.82,
    "editable_ratio_min": 0.15,
    "editable_text_recall_min": 0.80,
    "background_exact_match_max": 0.995,
    "background_changed_min": 0.01,
    "background_edge_retention_max": 0.90,
}


def _load_rgb(path, size=None):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def _load_gray(path, size=None):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("L")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def _ssim(a, b):
    """Legacy whole-array SSIM helper, retained for callers/tests that import it."""
    mu_a, mu_b = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / den) if den else 1.0


def _resize_gray(arr, scale: float):
    import numpy as np
    from PIL import Image

    if scale == 1.0:
        return arr
    h, w = arr.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
            size, Image.Resampling.LANCZOS
        ),
        dtype=np.float64,
    )


def _local_ssim_values(a, b, target_windows=10):
    """Non-overlapping local SSIM cells with partial edge cells retained."""
    h, w = a.shape
    window = max(8, min(64, int(round(min(h, w) / max(1, target_windows)))))
    values = []
    for y in range(0, h, window):
        for x in range(0, w, window):
            pa = a[y : min(h, y + window), x : min(w, x + window)]
            pb = b[y : min(h, y + window), x : min(w, x + window)]
            if pa.size:
                values.append(_ssim(pa, pb))
    return values or [_ssim(a, b)]


def _multiscale_ssim(a, b):
    import numpy as np

    scales = ((1.0, 0.50), (0.5, 0.30), (0.25, 0.20))
    per_scale = []
    combined = 0.0
    for scale, weight in scales:
        aa, bb = _resize_gray(a, scale), _resize_gray(b, scale)
        values = np.clip(np.asarray(_local_ssim_values(aa, bb), dtype=np.float64), 0, 1)
        mean = float(values.mean())
        p10 = float(np.percentile(values, 10))
        minimum = float(values.min())
        # The lower tail prevents a small broken region from disappearing in a canvas mean.
        robust = 0.68 * mean + 0.24 * p10 + 0.08 * minimum
        combined += weight * robust
        per_scale.append(
            {
                "scale": scale,
                "mean": round(mean, 5),
                "p10": round(p10, 5),
                "min": round(minimum, 5),
                "robust": round(robust, 5),
                "windows": int(values.size),
            }
        )
    first = per_scale[0]
    return max(0.0, min(1.0, combined)), per_scale, {
        "mean": first["mean"],
        "p10": first["p10"],
        "min": first["min"],
    }


def _gradient(gray):
    import numpy as np

    gx = np.zeros_like(gray, dtype=np.float64)
    gy = np.zeros_like(gray, dtype=np.float64)
    if gray.shape[1] > 1:
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gx[:, 0] = gray[:, 1] - gray[:, 0]
        gx[:, -1] = gray[:, -1] - gray[:, -2]
    if gray.shape[0] > 1:
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        gy[0, :] = gray[1, :] - gray[0, :]
        gy[-1, :] = gray[-1, :] - gray[-2, :]
    return np.hypot(gx, gy)


def _dilate(binary):
    import numpy as np

    padded = np.pad(binary, 1, mode="constant")
    out = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def _edge_metrics(source, render):
    import numpy as np

    src_mag = _gradient(source)
    ren_mag = _gradient(render)
    positive = src_mag[src_mag > 2]
    threshold = max(8.0, float(np.percentile(positive, 65)) if positive.size else 8.0)
    src_edges, ren_edges = src_mag >= threshold, ren_mag >= threshold
    ns, nr = int(src_edges.sum()), int(ren_edges.sum())
    if not ns and not nr:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "threshold": threshold}
    if not ns or not nr:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": threshold}
    precision = float((ren_edges & _dilate(src_edges)).sum()) / nr
    recall = float((src_edges & _dilate(ren_edges)).sum()) / ns
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "threshold": threshold}


def _metric_rgb(rgb, max_edge=384):
    import numpy as np
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(1.0, max_edge / max(1, h, w))
    if scale == 1.0:
        return rgb
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize(
            size, Image.Resampling.LANCZOS
        ),
        dtype=np.float64,
    )


def _rgb_to_lab(rgb):
    """Vectorized sRGB -> CIE Lab (D65), adequate for deterministic QA."""
    import numpy as np

    x = np.clip(rgb / 255.0, 0, 1)
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = x @ np.asarray(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float64,
    ).T
    xyz /= np.asarray([0.95047, 1.0, 1.08883])
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def _color_metrics(source_rgb, render_rgb):
    import numpy as np

    src, ren = _metric_rgb(source_rgb), _metric_rgb(render_rgb)
    delta = np.linalg.norm(_rgb_to_lab(src) - _rgb_to_lab(ren), axis=2)
    mean = float(delta.mean())
    p95 = float(np.percentile(delta, 95))
    mae = float(np.abs(src - ren).mean())
    return {
        "similarity": max(0.0, 1.0 - mean / 50.0),
        "delta_e_mean": mean,
        "delta_e_p95": p95,
        "rgb_mae": mae,
    }


def _block_mean(a, gy, gx):
    import numpy as np

    h, w = a.shape
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    out = np.zeros((gy, gx), dtype=np.float64)
    for i in range(gy):
        for j in range(gx):
            block = a[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def _norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _text_recall(source_ocr, render_ocr):
    src_lines = [_norm(l["text"]) for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5]
    src_lines = [s for s in src_lines if len(s) >= 3]
    ren_blob = " ".join(_norm(l["text"]) for l in render_ocr.get("lines", []))
    if not src_lines:
        return 1.0
    return sum(1 for s in src_lines if s in ren_blob) / len(src_lines)


def _load_design(design, run_dir):
    if isinstance(design, dict):
        return design
    path = design if isinstance(design, str) else os.path.join(run_dir, "design.json")
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _flatten_layers(layers):
    out = []
    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        out.append(layer)
        out.extend(_flatten_layers(layer.get("children") or []))
    return out


def _reported_items(value, label):
    if value is None or value is False:
        return []
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if isinstance(value, bool):
        return [label] if value else []
    if isinstance(value, (int, float)):
        return [f"{label}: {value}"] if value else []
    return [str(value)]


def _observation_key(observation):
    if isinstance(observation, str):
        return observation
    if not isinstance(observation, dict):
        return None
    if observation.get("key"):
        return str(observation["key"])
    if observation.get("id") is not None:
        return f"{observation.get('source', 'unknown')}:{observation['id']}"
    return None


def _load_reconstruction(run_dir):
    path = os.path.join(run_dir, "reconstruction.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _layer_qa_metrics(source):
    """Pull optional ssim/recall fields from a stats row or candidate meta blob."""
    if not isinstance(source, dict):
        return None, None
    qa = source.get("qa") if isinstance(source.get("qa"), dict) else source
    ssim = qa.get("ssim")
    recall = qa.get("recall", qa.get("text_recall"))
    return ssim, recall


def _text_per_layer_entry(layer_id, layer, ssim, recall):
    if ssim is None and recall is None:
        return None
    meta = (layer or {}).get("meta") or {}
    item = {
        "id": str(layer_id),
        "type": "text",
        "role": meta.get("role") or "text",
    }
    if ssim is not None:
        item["ssim"] = round(float(ssim), 4)
    if recall is not None:
        item["recall"] = round(float(recall), 4)
    scores = [item[key] for key in ("ssim", "recall") if key in item]
    if scores:
        item["score"] = round(min(scores), 4)
    return item


def _build_per_layer(reconstruction, design):
    """Populate text-layer QA rows from reconstruction stats/candidates when available."""
    reconstruction = reconstruction or {}
    stats = reconstruction.get("stats") or {}
    out = []
    seen = set()

    for entry in stats.get("per_layer") or []:
        if not isinstance(entry, dict):
            continue
        lid = str(entry.get("id") or "")
        if not lid or lid in seen:
            continue
        kind = entry.get("type") or entry.get("role")
        ssim, recall = _layer_qa_metrics(entry)
        if kind not in (None, "text") and ssim is None and recall is None:
            continue
        item = _text_per_layer_entry(lid, entry, ssim, recall)
        if item is None:
            continue
        if entry.get("role"):
            item["role"] = entry["role"]
        out.append(item)
        seen.add(lid)

    text_layers = [
        layer for layer in _flatten_layers((design or {}).get("layers") or [])
        if layer.get("type") == "text"
    ]
    candidates = {
        str(candidate.get("id")): candidate
        for candidate in (reconstruction.get("candidates") or [])
        if isinstance(candidate, dict) and candidate.get("id") is not None
    }

    for layer in text_layers:
        lid = str(layer.get("id") or "")
        if not lid or lid in seen:
            continue
        meta = layer.get("meta") or {}
        candidate = candidates.get(lid) or candidates.get(str(meta.get("source_id") or ""))
        cand_meta = (candidate or {}).get("meta") or {}
        ssim, recall = _layer_qa_metrics(cand_meta)
        if ssim is None and recall is None and candidate:
            ssim, recall = _layer_qa_metrics(candidate)
        item = _text_per_layer_entry(lid, layer, ssim, recall)
        if item is None:
            continue
        out.append(item)
        seen.add(lid)

    return out


def _duplicate_ownership(layers):
    owners, duplicates = {}, []
    for layer in layers:
        lid = str(layer.get("id") or layer.get("name") or "unnamed")
        meta = layer.get("meta") or {}
        provenance = meta.get("provenance") or {}
        observations = provenance.get("observations") or meta.get("observations") or []
        for observation in observations:
            key = _observation_key(observation)
            if not key:
                continue
            previous = owners.get(key)
            if previous and previous != lid:
                duplicates.append(f"{key} owned by {previous} and {lid}")
            else:
                owners[key] = lid
    return sorted(set(duplicates))


def _editable_text_recall(source_ocr, design, layers):
    if not source_ocr or not design:
        return None
    source = [_norm(line.get("text")) for line in source_ocr.get("lines", [])
              if line.get("conf", 1) >= 0.5 and len(_norm(line.get("text"))) >= 3]
    if not source:
        return 1.0
    editable = " ".join(_norm(layer.get("text")) for layer in layers if layer.get("type") == "text")
    kept = " ".join(_norm(text) for text in design.get("kept_in_photo", []))
    return sum(1 for text in source if text in editable or text in kept) / len(source)


def _resolve_path(path, run_dir):
    if not path:
        return None
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.join(run_dir, str(path))


def _load_mask(mask, size):
    import numpy as np
    from PIL import Image

    if mask is None:
        return None
    if isinstance(mask, str):
        if not os.path.exists(mask):
            return None
        im = Image.open(mask).convert("L")
        if im.size != tuple(size):
            im = im.resize(tuple(size), Image.Resampling.NEAREST)
        return np.asarray(im) > 0
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.shape != (size[1], size[0]):
        im = Image.fromarray((arr > 0).astype(np.uint8) * 255).resize(
            tuple(size), Image.Resampling.NEAREST
        )
        arr = np.asarray(im)
    return arr > 0


def _background_audit(source_rgb, background_path, removal_mask):
    import numpy as np

    if not background_path or not os.path.exists(background_path):
        return None
    height, width = source_rgb.shape[:2]
    background = _load_rgb(background_path, size=(width, height))
    mask = _load_mask(removal_mask, (width, height))
    mask_supplied = mask is not None
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    if not np.any(mask):
        return {"mask_supplied": mask_supplied, "masked_pixels": 0,
                "exact_match_ratio": 0.0, "changed_ratio": 1.0,
                "edge_retention": None, "mean_change": 0.0}
    delta = np.abs(source_rgb - background).mean(axis=2)
    exact = float((delta[mask] < 0.5).mean())
    changed = float((delta[mask] > 8.0).mean())
    mean_change = float(delta[mask].mean())
    source_edges = _gradient(
        source_rgb[..., 0] * 0.299 + source_rgb[..., 1] * 0.587 + source_rgb[..., 2] * 0.114
    ) >= 12
    background_edges = _gradient(
        background[..., 0] * 0.299 + background[..., 1] * 0.587 + background[..., 2] * 0.114
    ) >= 12
    source_edge_mask = source_edges & mask
    retained = None
    if int(source_edge_mask.sum()):
        retained = float((source_edge_mask & _dilate(background_edges)).sum()) / int(source_edge_mask.sum())
    return {
        "mask_supplied": mask_supplied,
        "masked_pixels": int(mask.sum()),
        "exact_match_ratio": round(exact, 5),
        "changed_ratio": round(changed, 5),
        "edge_retention": None if retained is None else round(retained, 5),
        "mean_change": round(mean_change, 4),
    }


def _add_fail(fails, rule, detail):
    if not any(item.get("rule") == rule and item.get("detail") == detail for item in fails):
        fails.append({"rule": rule, "detail": detail, "hard": True})


def _structural_audit(
    source_rgb,
    run_dir,
    design,
    source_ocr,
    supplied,
    background_path,
    removal_mask,
    thresholds,
):
    supplied = dict(supplied or {})
    layers = _flatten_layers((design or {}).get("layers") or [])
    missing_assets = []
    missing_fonts = []
    for layer in layers:
        lid = str(layer.get("id") or layer.get("name") or "unnamed")
        if layer.get("type") == "image":
            src = layer.get("src")
            path = _resolve_path(src, run_dir)
            if not src or (not str(src).startswith("data:") and (not path or not os.path.exists(path))):
                missing_assets.append(lid)
        mask = layer.get("mask") or {}
        if isinstance(mask, dict) and mask.get("src"):
            path = _resolve_path(mask["src"], run_dir)
            if not path or not os.path.exists(path):
                missing_assets.append(f"{lid}:mask")
        if layer.get("type") == "text" and not (layer.get("meta") or {}).get("emoji"):
            style = layer.get("style") or {}
            if not style.get("fontFamily") or style.get("fontResolved") is False:
                missing_fonts.append(lid)
    schema_errors = []
    for warning in ((design or {}).get("meta") or {}).get("warnings") or []:
        code = str(warning.get("code", "")) if isinstance(warning, dict) else ""
        if code == "missing-asset":
            missing_assets.append(str(warning.get("layer_id") or warning.get("path") or warning))
        if code in ("missing-font", "font-load-failed"):
            missing_fonts.append(str(warning.get("layer_id") or warning.get("font") or warning))
        if code == "invalid-schema":
            schema_errors.append(str(warning.get("detail") or warning))

    figma_report = {}
    report_path = os.path.join(run_dir, "figma_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            candidate = payload.get("report") if isinstance(payload, dict) else None
            reported_doc = str(payload.get("doc_id") or (candidate or {}).get("docId") or "")
            design_id = str((design or {}).get("id") or "")
            if isinstance(candidate, dict) and (not reported_doc or not design_id or reported_doc == design_id):
                figma_report = candidate
        except Exception:
            figma_report = {}
    if int((figma_report.get("assets") or {}).get("missing", 0) or 0) > 0:
        missing_assets.append("Figma compiler report")
    font_substitutions = list((figma_report.get("fonts") or {}).get("selections") or [])
    missing_assets.extend(_reported_items(supplied.get("missing_assets"), "missing assets"))
    missing_fonts.extend(_reported_items(supplied.get("missing_fonts"), "missing fonts"))
    missing_assets, missing_fonts = sorted(set(missing_assets)), sorted(set(missing_fonts))

    editable_ratio = supplied.get("editable_ratio")
    if editable_ratio is None and design:
        editable_ratio = ((design.get("meta") or {}).get("editable_ratio"))
    if editable_ratio is None and layers:
        editable = sum(1 for layer in layers if layer.get("type") in ("text", "shape", "group"))
        editable_ratio = editable / len(layers)
    editable_ratio = None if editable_ratio is None else float(editable_ratio)
    editable_text_recall = _editable_text_recall(source_ocr, design, layers)

    duplicate_ownership = _duplicate_ownership(layers)
    duplicate_ownership.extend(
        _reported_items(supplied.get("duplicate_ownership"), "duplicate ownership")
    )
    duplicate_ownership = sorted(set(duplicate_ownership))

    reconstruction = _load_reconstruction(run_dir)

    if background_path is None:
        candidate = os.path.join(run_dir, "background_clean.png")
        background_path = candidate if os.path.exists(candidate) else None
    else:
        background_path = _resolve_path(background_path, run_dir)
    if removal_mask is None:
        candidate = os.path.join(run_dir, "removal_mask.png")
        removal_mask = candidate if os.path.exists(candidate) else None
    elif isinstance(removal_mask, str):
        removal_mask = _resolve_path(removal_mask, run_dir)
    background = _background_audit(source_rgb, background_path, removal_mask)
    explicit_leakage = supplied.get("background_leakage")

    fails = []
    compiler_errors = list(figma_report.get("errors") or [])
    if figma_report and (figma_report.get("ok") is False or compiler_errors):
        detail = compiler_errors[0].get("detail") if compiler_errors and isinstance(compiler_errors[0], dict) else "Figma import did not complete"
        _add_fail(fails, "figma-compiler-errors", str(detail))
    if missing_assets:
        _add_fail(fails, "missing-assets", f"{len(missing_assets)} unresolved asset(s): " + ", ".join(missing_assets[:4]))
    if missing_fonts:
        _add_fail(fails, "missing-fonts", f"{len(missing_fonts)} unresolved text font(s): " + ", ".join(missing_fonts[:4]))
    if schema_errors:
        _add_fail(fails, "invalid-schema", f"{len(schema_errors)} design.json shape error(s): " + "; ".join(schema_errors[:3]))
    source_lines = (source_ocr or {}).get("lines", []) if isinstance(source_ocr, dict) else []
    if source_lines and editable_ratio is not None and editable_ratio < thresholds["editable_ratio_min"]:
        _add_fail(fails, "low-editable-ratio",
                  f"editable ratio {editable_ratio:.2f} < {thresholds['editable_ratio_min']:.2f}")
    if editable_text_recall is not None and editable_text_recall < thresholds["editable_text_recall_min"]:
        _add_fail(fails, "missing-editable-text",
                  f"editable text recall {editable_text_recall:.2f} < {thresholds['editable_text_recall_min']:.2f}")
    if duplicate_ownership:
        _add_fail(fails, "duplicate-ownership",
                  f"{len(duplicate_ownership)} observation ownership conflict(s): " + duplicate_ownership[0])
    if design and layers:
        background_layers = [layer for layer in layers if (layer.get("meta") or {}).get("role") == "background"]
        if background_layers and any((layer.get("meta") or {}).get("source") != "inpaint" for layer in background_layers):
            _add_fail(fails, "unclean-background", "background layer is not sourced from the clean inpaint plate")
    if background:
        retained = background.get("edge_retention")
        leaked = (
            background["exact_match_ratio"] > thresholds["background_exact_match_max"]
            or background["changed_ratio"] < thresholds["background_changed_min"]
            or (retained is not None and retained > thresholds["background_edge_retention_max"]
                and background["changed_ratio"] < 0.05)
        )
        if leaked and (source_lines or len(layers) > 1 or background.get("mask_supplied")):
            _add_fail(
                fails,
                "background-leakage",
                "clean plate still matches extracted foreground inside the removal region",
            )
    if explicit_leakage:
        _add_fail(fails, "background-leakage", f"reported background leakage: {explicit_leakage}")

    # Preserve externally supplied structural failures without trusting only one caller field.
    for item in supplied.get("hard_fails", []) or []:
        if isinstance(item, dict) and item.get("rule"):
            _add_fail(fails, str(item["rule"]), str(item.get("detail", item["rule"])))

    stats = reconstruction.get("stats") or {}
    return {
        "missing_assets": missing_assets,
        "missing_fonts": missing_fonts,
        "font_substitutions": font_substitutions,
        "figma_report": {
            "ok": figma_report.get("ok") if figma_report else None,
            "created": figma_report.get("created") if figma_report else None,
            "skipped": figma_report.get("skipped") if figma_report else None,
        },
        "editable_ratio": None if editable_ratio is None else round(editable_ratio, 4),
        "editable_text_recall": None if editable_text_recall is None else round(editable_text_recall, 4),
        "duplicate_ownership": duplicate_ownership,
        "duplicates_removed": int(stats.get("duplicates_removed", supplied.get("duplicates_removed", 0)) or 0),
        "background": background,
        "hard_fails": fails,
    }


def compare(
    source_path,
    render_path,
    run_dir,
    source_ocr=None,
    render_ocr=None,
    *,
    design=None,
    structural=None,
    background_path=None,
    removal_mask=None,
    thresholds: Optional[dict] = None,
):
    """Compare a source and reconstruction.

    Backward-compatible positional API::

        compare(source, render, run_dir, source_ocr=None, render_ocr=None)

    New optional keyword inputs are ``design`` (dict/path), ``structural`` (reported
    missing-assets/fonts, editable ratio, duplicate ownership, or leakage),
    ``background_path``, ``removal_mask``, and threshold overrides.
    """
    import numpy as np
    from PIL import Image

    os.makedirs(run_dir, exist_ok=True)
    source_rgb = _load_rgb(source_path)
    height, width = source_rgb.shape[:2]
    render_rgb = _load_rgb(render_path, size=(width, height))
    source_gray = source_rgb[..., 0] * 0.299 + source_rgb[..., 1] * 0.587 + source_rgb[..., 2] * 0.114
    render_gray = render_rgb[..., 0] * 0.299 + render_rgb[..., 1] * 0.587 + render_rgb[..., 2] * 0.114

    global_ssim = max(0.0, min(1.0, _ssim(source_gray, render_gray)))
    multiscale, per_scale, local = _multiscale_ssim(source_gray, render_gray)
    edge = _edge_metrics(source_gray, render_gray)
    color = _color_metrics(source_rgb, render_rgb)
    visual_score = max(0.0, min(1.0,
        0.65 * multiscale + 0.20 * edge["f1"] + 0.15 * color["similarity"]
    ))

    diff = np.abs(source_gray - render_gray)
    gy, gx = 16, 16
    cells = _block_mean(diff, gy, gx)
    heat = (cells / max(1e-6, float(cells.max())) * 255).astype(np.uint8)
    diff_png = os.path.join(run_dir, "diff.png")
    Image.fromarray(heat).resize((width, height), Image.Resampling.NEAREST).save(diff_png)
    ranked = sorted(
        ({"row": int(i), "col": int(j), "mean_delta": round(float(cells[i, j]), 3)}
         for i in range(gy) for j in range(gx)),
        key=lambda item: item["mean_delta"],
        reverse=True,
    )

    text_recall = None
    if source_ocr and render_ocr:
        text_recall = _text_recall(source_ocr, render_ocr)

    opts = dict(DEFAULT_THRESHOLDS)
    opts.update(thresholds or {})
    design_data = _load_design(design, run_dir)
    structure = _structural_audit(
        source_rgb,
        run_dir,
        design_data,
        source_ocr,
        structural,
        background_path,
        removal_mask,
        opts,
    )
    quality_flags = []
    if multiscale < opts["local_ssim_min"]:
        quality_flags.append({"rule": "local-ssim", "detail": f"{multiscale:.3f} < {opts['local_ssim_min']:.3f}"})
    if edge["f1"] < opts["edge_f1_min"]:
        quality_flags.append({"rule": "edge-fidelity", "detail": f"{edge['f1']:.3f} < {opts['edge_f1_min']:.3f}"})
    if color["similarity"] < opts["color_similarity_min"]:
        quality_flags.append({"rule": "color-fidelity", "detail": f"{color['similarity']:.3f} < {opts['color_similarity_min']:.3f}"})

    # quality_flags must actually gate acceptance — merge them into hard_fails rather than
    # leaving them as inert diagnostics that _structural_audit knows nothing about.
    hard_fails = list(structure["hard_fails"])
    seen_fails = {(item.get("rule"), item.get("detail")) for item in hard_fails}
    for flag in quality_flags:
        key = (flag.get("rule"), flag.get("detail"))
        if key not in seen_fails:
            hard_fails.append({**flag, "hard": True})
            seen_fails.add(key)
    structure["hard_fails"] = hard_fails
    per_layer = _build_per_layer(_load_reconstruction(run_dir), design_data)

    return {
        # Compatibility: callers still read `ssim`, now the harder local/multiscale metric.
        "ssim": round(multiscale, 4),
        "global_ssim": round(global_ssim, 4),
        "multiscale_ssim": round(multiscale, 4),
        "local_ssim": local,
        "ssim_scales": per_scale,
        "edge_f1": round(edge["f1"], 4),
        "edge_precision": round(edge["precision"], 4),
        "edge_recall": round(edge["recall"], 4),
        "color_similarity": round(color["similarity"], 4),
        "delta_e_mean": round(color["delta_e_mean"], 4),
        "delta_e_p95": round(color["delta_e_p95"], 4),
        "rgb_mae": round(color["rgb_mae"], 4),
        "visual_score": round(visual_score, 4),
        "quality_flags": quality_flags,
        "text_recall": None if text_recall is None else round(text_recall, 4),
        "editable_text_recall": structure["editable_text_recall"],
        "per_layer": per_layer,
        "per_region_max_delta": float(cells.max()),
        "per_region": {"rows": gy, "cols": gx, "mean_delta": cells.round(3).tolist(), "worst": ranked[:8]},
        "diff_png": diff_png,
        "structural": structure,
        "hard_fails": hard_fails,
    }
