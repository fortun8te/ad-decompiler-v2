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

# Canonical fallback-contract helpers live in src.schema (F11). Import them so every stage
# reads meta.fallback the SAME way instead of ad-hoc truthiness/equality checks. They are
# added by the reconstruct/schema agent; guard the import so this module stays usable if a
# checkout predates them (the guard is a thin adapter, NOT a redefinition of the contract).
try:
    from src.schema import is_raster_slice as _is_raster_slice, fallback_kind as _fallback_kind
except Exception:  # pragma: no cover - only when a checkout predates the schema helpers
    def _is_raster_slice(meta) -> bool:
        return bool(isinstance(meta, dict) and meta.get("fallback") == "raster-slice")

    def _fallback_kind(meta):
        if not isinstance(meta, dict) or meta.get("fallback") in (None, False, "", 0):
            return None
        return "raster-slice" if meta.get("fallback") == "raster-slice" else "fidelity-image"

# Template-free Codia construction scoring (scripts/codia_parity.score_construction). Guarded
# so pixel_diff stays importable if the scripts package is unavailable; the construction block
# then degrades to None rather than breaking QA.
try:
    import sys as _sys
    _scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    from codia_parity import score_construction as _score_construction
except Exception:  # pragma: no cover - only when scripts/codia_parity is unavailable
    _score_construction = None


DEFAULT_THRESHOLDS = {
    "local_ssim_min": DEFAULT_VISUAL_PASS_SSIM,
    "edge_f1_min": 0.68,
    "color_similarity_min": 0.82,
    "editable_ratio_min": 0.15,
    # native_leaf_ratio is native_leaf_count / foreground_leaf_count (text/shape leaves only,
    # background excluded). 0.30 is conservative on purpose: it only needs to catch the
    # "almost everything got rasterized" failure mode (a wrapper frame around one raster
    # image, or a page that gave up and rasterized nearly all foreground). Legitimate ads
    # with a couple of photos plus real text/shape layers clear this easily; it is not meant
    # to police photo-heavy-but-honest layouts, only near-total rasterization.
    "native_leaf_ratio_min": 0.30,
    "editable_text_recall_min": 0.80,
    # Structural-honesty gates that are NOT keyed to a Figma acceptance run (F2): they
    # evaluate whenever leaf accounting exists (it always does now) and fire the
    # unexplained-raster / near-total-rasterization hard-fails. Config can turn them off,
    # but the sane default is ON so a good screenshot cannot buy off a rasterized page.
    "enforce_native_leaf_accounting": True,
    # Upper bound on how much of the canvas the removal/inpaint pass may destroy (F3). 002
    # rebuilt 85% of the plate (products erased) and tripped nothing; >0.55 of canvas being
    # altered inside the removal mask is a red flag that real content was erased.
    "background_changed_ratio_max": 0.55,
    # Unresolved glyph residue under a removed text region is a structural failure, not a
    # mere repair suggestion (F15): QA must not report ok while it stands.
    "glyph_residue_gate": True,
    "background_exact_match_max": 0.995,
    "background_changed_min": 0.01,
    "background_edge_retention_max": 0.90,
    "background_outside_damage_max": 0.01,
    "layer_internal_hole_fraction_max": 0.025,
    "element_survival_min": 0.75,
    # Per-archetype text strictness (F8) is threaded in by the caller (archetype preset's
    # text_recall_min). Left None here so a bare compare() keeps its old behaviour and only
    # archetype-configured runs gate on global text recall.
    "text_recall_min": None,
    # F-honesty: editable_text_recall alone can lie about its own denominator. If OCR only
    # found 17% of the source text and every found line happens to be editable, recall reads
    # 1.0 while 83% of the ad's copy silently doesn't exist anywhere in the reconstruction.
    # true_text_coverage = text_recall * editable_text_recall folds both fractions into one
    # honest share-of-ALL-source-text-that-became-correct-editable-text number. 0.20 is a
    # deliberately low floor: it only rejects the severe "OCR missed most of the ad" case
    # (measured on the 021 fixture: text_recall 0.17 x editable_text_recall 1.0 = 0.17) and
    # does not double-penalize runs that already clear editable_text_recall_min on its own.
    "true_text_coverage_min": 0.20,
    # F-worst-region: the multiscale/local-ssim aggregate is deliberately mean-dominated
    # (0.72 mean + 0.26 p10 + 0.02 min per scale — see _multiscale_ssim) so one isolated
    # near-zero cell from a legitimate editable overlap does not sink an otherwise-good
    # score. That same de-emphasis lets a genuinely catastrophic region (009/016 measured
    # worst-window SSIM ~0.03-0.04) hide under a good aggregate. This floor is a SEPARATE,
    # independent gate on the single worst window regardless of the aggregate score.
    # Configurable — callers/archetypes may raise or lower it.
    "local_ssim_worst_window_min": 0.10,
    # ── CODIA CONSTRUCTION CONTRACT (docs/CODIA-PARITY-SPEC.md) ───────────────────────
    # The QA objective is Codia's construction, not screenshot SSIM: every string is native
    # editable TEXT, everything hard is an image cutout, flat chrome is a solid plate,
    # placed absolutely. These floors score that contract and demote global SSIM to a floor
    # gate. They are REPORTED on every run; the contract pass/fail summary leads with them.
    #
    # native_text_ratio (= native editable TEXT lines / all readable OCR lines) must be high
    # for every non-handwriting archetype — there is no handwriting archetype, so 0.90 is the
    # universal contract bar. It is enforced by the existing missing-editable-text /
    # true_text_coverage hard-fails (which fire well below this); this floor drives the
    # contract PASS summary and the harness reward, not a new duplicate hard-fail.
    "native_text_ratio_min": 0.90,
    # Global SSIM is a FLOOR gate for the contract summary, not the objective. A Codia-shaped
    # output (100% native text, clean plate, decent placement) must PASS the contract even at
    # a modest SSIM, so the contract's own SSIM floor is deliberately low and separate from
    # the archetype visual_pass_ssim gate. Anti-degenerate only: a near-empty/garbage render
    # still fails it.
    "contract_ssim_floor": 0.45,
    # Placement: mean translation-aligned text ink-IoU across native text rows. "decent
    # placement" for the contract; Figma re-fits glyph placement so this is lenient.
    "contract_placement_ink_iou_min": 0.35,
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


def _local_ssim_values_preserved(a, b, preserve_mask, target_windows=10):
    """Local SSIM on windows that mostly lie outside the removal union mask."""
    import numpy as np

    h, w = a.shape
    window = max(8, min(64, int(round(min(h, w) / max(1, target_windows)))))
    values = []
    relaxed = []
    for y in range(0, h, window):
        for x in range(0, w, window):
            km = preserve_mask[y : min(h, y + window), x : min(w, x + window)]
            if km.size == 0 or float(km.mean()) < 0.70:
                if km.size and float(km.mean()) >= 0.30:
                    relaxed.append((y, x))
                continue
            pa = a[y : min(h, y + window), x : min(w, x + window)]
            pb = b[y : min(h, y + window), x : min(w, x + window)]
            if pa.size:
                values.append(_ssim(pa, pb))
    if values:
        return values
    # Heavy-removal creatives can have very little preserved area; do not fall back to
    # whole-image SSIM (that would penalize deliberate inpaint). Use a relaxed mask.
    for y, x in relaxed:
        pa = a[y : min(h, y + window), x : min(w, x + window)]
        pb = b[y : min(h, y + window), x : min(w, x + window)]
        if pa.size:
            values.append(_ssim(pa, pb))
    return values or _local_ssim_values(a, b, target_windows)


def _local_ssim_worst_window(a, b, preserve_mask=None, target_windows=10):
    """Locate the single worst-scoring local SSIM window, with its pixel bbox.

    Uses the exact same non-overlapping window grid as :func:`_local_ssim_values`
    (and, at scale 1.0, the same grid `_multiscale_ssim` scores) so its "ssim" value
    matches ``local_ssim.min`` for the un-resized image — this is deliberately the
    same signal, just kept alongside a locatable bbox instead of being diluted into
    a 2%-weighted term of the aggregate (F-worst-region: a catastrophic region must
    be nameable/pinpointable evidence, not just a number lost inside a mean).
    """
    h, w = a.shape
    window = max(8, min(64, int(round(min(h, w) / max(1, target_windows)))))
    cells = []
    relaxed = []
    for y in range(0, h, window):
        for x in range(0, w, window):
            pa = a[y : min(h, y + window), x : min(w, x + window)]
            pb = b[y : min(h, y + window), x : min(w, x + window)]
            if not pa.size:
                continue
            bbox = {"x": int(x), "y": int(y), "w": int(pa.shape[1]), "h": int(pa.shape[0])}
            if preserve_mask is not None:
                km = preserve_mask[y : min(h, y + window), x : min(w, x + window)]
                if km.size == 0 or float(km.mean()) < 0.70:
                    if km.size and float(km.mean()) >= 0.30:
                        relaxed.append((bbox, pa, pb))
                    continue
            cells.append((bbox, pa, pb))
    source_cells = cells or relaxed
    if not source_cells:
        return {"ssim": round(float(_ssim(a, b)), 5), "bbox": {"x": 0, "y": 0, "w": int(w), "h": int(h)}}
    worst_bbox, worst_pa, worst_pb = min(source_cells, key=lambda item: _ssim(item[1], item[2]))
    return {"ssim": round(float(_ssim(worst_pa, worst_pb)), 5), "bbox": worst_bbox}


def _multiscale_ssim(a, b, preserve_mask=None):
    import numpy as np

    scales = ((1.0, 0.50), (0.5, 0.30), (0.25, 0.20))
    per_scale = []
    combined = 0.0
    local_fn = _local_ssim_values_preserved if preserve_mask is not None else _local_ssim_values
    for scale, weight in scales:
        aa, bb = _resize_gray(a, scale), _resize_gray(b, scale)
        mask = None
        if preserve_mask is not None:
            import numpy as np
            from PIL import Image
            mask_img = Image.fromarray(preserve_mask.astype(np.uint8) * 255).resize(
                (aa.shape[1], aa.shape[0]), Image.Resampling.NEAREST
            )
            mask = np.asarray(mask_img) > 0
        if mask is not None:
            values = np.clip(np.asarray(local_fn(aa, bb, mask), dtype=np.float64), 0, 1)
        else:
            values = np.clip(np.asarray(local_fn(aa, bb), dtype=np.float64), 0, 1)
        mean = float(values.mean())
        p10 = float(np.percentile(values, 10))
        minimum = float(values.min())
        # The lower tail remains diagnostic, but an isolated zero-valued cell is
        # often a deliberate editable overlap (for example a button label over
        # its shell).  Broad defects lower both mean and p10 and are still gated
        # hard; structural QA independently rejects empty layers, matte holes,
        # ownership loss, and out-of-mask inpainting.
        robust = 0.72 * mean + 0.26 * p10 + 0.02 * minimum
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


def _edge_metrics(source, render, preserve_mask=None):
    import numpy as np

    src_mag = _gradient(source)
    ren_mag = _gradient(render)
    positive = src_mag[src_mag > 2]
    threshold = max(8.0, float(np.percentile(positive, 65)) if positive.size else 8.0)
    src_edges, ren_edges = src_mag >= threshold, ren_mag >= threshold
    if preserve_mask is not None:
        # If almost everything was removed/inpainted, edge scoring becomes unstable and
        # should not hard-fail the run.
        if float(np.mean(preserve_mask)) < 0.08:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "threshold": threshold}
        src_edges = src_edges & preserve_mask
        ren_edges = ren_edges & preserve_mask
    ns, nr = int(src_edges.sum()), int(ren_edges.sum())
    # If the preserved region contains almost no edges (common for big photo removals
    # leaving mostly flat UI chrome), edge F1 becomes meaningless; don't hard-fail.
    if preserve_mask is not None and ns < 1200:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "threshold": threshold}
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


# Archetypes whose "text" is genuinely interface copy (posts, tweets, DMs, chat), never
# printed-into-a-photograph scene text.  A screenshot that baked all its copy is a FAILURE,
# not the contract-correct single-photo answer, so it can never earn the scene-baked
# exemption regardless of what merge's photographic_scene_text flag says.
_NON_PHOTOGRAPHIC_ARCHETYPES = {"social_screenshot"}


def _read_archetype(run_dir) -> str:
    try:
        with open(os.path.join(run_dir, "archetype.json"), encoding="utf-8") as fh:
            return str(json.load(fh).get("archetype") or "")
    except Exception:
        return ""


def _has_empty_group(layers) -> bool:
    """True if any group/frame node has zero leaf descendants (empty structural junk).

    Empty wrapper groups are the 021 false-pass shape: a design that emitted a handful of
    groups with NOTHING inside them is not a legitimate photographic output, so it must not
    slip past the editability floors through the scene-baked exemption.
    """
    for node in layers or []:
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list) and node.get("type") in ("group", "frame"):
            if not any(_is_leaf_present(child) for child in children):
                return True
            if _has_empty_group(children):
                return True
    return False


def _is_leaf_present(node) -> bool:
    if not isinstance(node, dict):
        return False
    children = node.get("children")
    if isinstance(children, list) and children:
        return any(_is_leaf_present(child) for child in children)
    return True


def _scene_baked_exemption_block_reason(run_dir, design) -> str:
    """Why a photographic-scene exemption must NOT apply here (empty string = allowed).

    The exemption (a 1-photo output is contract-correct when all source text is printed in
    the scene) is only legitimate for genuinely photographic archetypes with a real, non-empty
    layer tree.  Screenshots and empty-junk-group outputs never qualify.
    """
    archetype = _read_archetype(run_dir)
    if archetype in _NON_PHOTOGRAPHIC_ARCHETYPES:
        return f"archetype {archetype} is a screenshot, not a photographic scene"
    if _has_empty_group((design or {}).get("layers") or []):
        return "design contains empty group(s) — not a legitimate photographic output"
    return ""


def _text_recall(source_ocr, render_ocr, source_gray=None, render_gray=None):
    kept = [l for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5
            and len(_norm(l.get("text", ""))) >= 3]
    if not kept:
        return 1.0
    ren_blob = " ".join(_norm(l["text"]) for l in render_ocr.get("lines", []))
    found = 0
    for line in kept:
        if _norm(line["text"]) in ren_blob:
            found += 1
            continue
        # Baked-verbatim fallback: OCR is not deterministic even on IDENTICAL pixels
        # (021 measured recall 0.6 on a render with ssim 1.0 vs source). If the pixel
        # region under this line's box survives essentially unchanged in the render,
        # the text is literally present — count it, no re-OCR roulette. Un-gameable:
        # pixel identity cannot be faked by a wrong reconstruction.
        if source_gray is None or render_gray is None:
            continue
        box = line.get("box") or {}
        try:
            h, w = source_gray.shape[:2]
            x0 = max(0, int(box.get("x", 0)));  y0 = max(0, int(box.get("y", 0)))
            x1 = min(w, int(box.get("x", 0) + box.get("w", 0)))
            y1 = min(h, int(box.get("y", 0) + box.get("h", 0)))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            import numpy as _np
            delta = _np.abs(source_gray[y0:y1, x0:x1].astype(_np.float32)
                            - render_gray[y0:y1, x0:x1].astype(_np.float32))
            if float(delta.mean()) <= 2.0:
                found += 1
        except Exception:
            continue
    return found / len(kept)


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


def _normalized_element_id(value):
    """Normalize canonical element ids without conflating other detector namespaces."""
    import re

    match = re.fullmatch(r"(?:c_)?e0*(\d+)", str(value or "").strip(), re.IGNORECASE)
    return f"E{int(match.group(1))}" if match else None


def _candidate_element_lineage(candidate):
    """Return canonical element ids owned by a surviving reconstruction candidate."""
    if not isinstance(candidate, dict):
        return set()
    values = [candidate.get("id")]
    meta = candidate.get("meta") or {}
    values.extend((meta.get("source_id"), meta.get("canonical_id"), meta.get("element_id")))
    # NMS/dedup retains the winning candidate but records canonical ids it
    # absorbed.  They are survived observations, not dropped elements.
    values.extend(meta.get("merged_observations") or [])

    provenance = meta.get("provenance") or {}
    if isinstance(provenance, list):
        observations = provenance
    elif isinstance(provenance, dict):
        observations = provenance.get("observations") or []
    else:
        observations = []
    observations = list(observations) + list(meta.get("observations") or [])
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        # Residual E ids are a separate detector namespace and cannot be assumed to
        # identify fused/canonical E ids.  Only explicitly canonical lineage is safe.
        source = str(observation.get("source") or "").strip().lower()
        if source in {"element", "elements", "fused", "fused-element", "fused_elements", "canonical"}:
            values.extend((observation.get("id"), observation.get("element_id")))

    return {normalized for value in values if (normalized := _normalized_element_id(value))}


def _element_survival_audit(run_dir, reconstruction):
    """Prove detected non-background elements survived into reconstruction.

    This is intentionally an artifact lineage check, not another vision guess. A visually
    similar background can conceal a dropped element, but its canonical id cannot disappear
    from the reconstruction without being reported.
    """
    elements = []
    for name in ("elements.json", "fused_elements.json", "sam3.json"):
        payload = _load_design(os.path.join(run_dir, name), run_dir)
        if isinstance(payload, list):
            elements = payload
        elif isinstance(payload, dict):
            elements = payload.get("elements") or payload.get("candidates") or []
        if elements:
            break
    proposed_by_id = {
        normalized: str(item.get("id")) for item in elements
        if isinstance(item, dict) and item.get("id") is not None
        and str((item.get("meta") or {}).get("role") or item.get("role") or "").lower()
        != "background"
        if (normalized := _normalized_element_id(item.get("id")))
    }
    if not proposed_by_id:
        return None
    kept = set()
    canonical_elements = []
    protected = []
    for item in reconstruction.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        lineage = _candidate_element_lineage(item)
        if not lineage:
            continue
        canonical_elements.append(item)
        meta = item.get("meta") or {}
        is_protected = bool(
            meta.get("keep_in_background") or meta.get("kept_in_photo")
            or meta.get("raster_fallback") or meta.get("suppression_reason")
            or meta.get("baked_owner_id") or meta.get("flattened_scene_artwork")
        )
        if item.get("target") != "drop" or is_protected:
            kept.update(lineage)
        if is_protected:
            protected.append(item)

    # Flattened/photo-scene presets intentionally account for SAM proposals in the
    # protected raster plate instead of emitting one Figma layer per segmentation.
    # In that mode standalone element recall is not applicable; reporting 0/N causes
    # the repair loop to rerun SAM even though no canonical element was lost.
    if canonical_elements and len(protected) == len(canonical_elements):
        return {
            "proposed": len(proposed_by_id), "kept": len(set(proposed_by_id) & kept),
            "recall": None, "missing_ids": [], "protected": len(protected),
            "expected_standalone": 0, "not_applicable": True,
        }
    survived = set(proposed_by_id) & kept
    missing = sorted(proposed_by_id[item] for item in set(proposed_by_id) - kept)
    return {
        "proposed": len(proposed_by_id), "kept": len(survived),
        "recall": round(len(survived) / len(proposed_by_id), 5),
        "missing_ids": missing,
    }
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


def _iter_leaf_layers_abs(layers, offset_x=0.0, offset_y=0.0):
    """Yield (leaf_layer, absolute_box) pairs.

    design.json children carry PARENT-RELATIVE coordinates (coordinate_space="local"),
    so group offsets accumulate on the way down. Only leaves are yielded; groups are
    containers, not paint.
    """
    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        box = layer.get("box") or {}
        try:
            ax = offset_x + float(box.get("x", 0) or 0)
            ay = offset_y + float(box.get("y", 0) or 0)
            w = float(box.get("w", 0) or 0)
            h = float(box.get("h", 0) or 0)
        except (TypeError, ValueError):
            continue
        if layer.get("type") == "group":
            yield from _iter_leaf_layers_abs(layer.get("children") or [], ax, ay)
            continue
        yield layer, {"x": ax, "y": ay, "w": w, "h": h}


def _region_ink_mask(crop_rgb):
    """Painted-ink estimate for a crop: pixels contrasting with the local plate.

    Border pixels estimate the plate colour; an Otsu split on the colour distance
    separates glyph/graphic ink from background. Deterministic and CPU-only.
    """
    import cv2
    import numpy as np

    if crop_rgb.size == 0 or crop_rgb.shape[0] < 2 or crop_rgb.shape[1] < 2:
        return None
    crop = crop_rgb.astype(np.float32)
    border = np.concatenate([
        crop[:1].reshape(-1, 3), crop[-1:].reshape(-1, 3),
        crop[:, :1].reshape(-1, 3), crop[:, -1:].reshape(-1, 3),
    ], axis=0)
    plate = np.median(border, axis=0)
    distance = np.linalg.norm(crop - plate, axis=2)
    peak = float(distance.max())
    if peak < 18.0:
        return np.zeros(distance.shape, dtype=bool)
    scaled = np.clip(distance / peak * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return distance >= max(24.0, float(threshold) / 255.0 * peak)


def _layer_region_rows(source_rgb, render_rgb, design):
    """PER-LAYER region scores: crop SSIM for every foreground leaf, ink-IoU for text.

    This is the measurement half of the Codia-style confidence gate: each emitted
    layer's bbox is compared render-vs-source locally, so one wrong region cannot
    hide inside a good global score. Rows carry raw metrics only; thresholds are
    applied by schema.raster_slice_failures (repair.assess and the reconstruct
    fallback share that gate).
    """
    import numpy as np

    layers = (design or {}).get("layers") or []
    if not layers:
        return []
    height, width = source_rgb.shape[:2]
    source_gray = source_rgb[..., 0] * 0.299 + source_rgb[..., 1] * 0.587 + source_rgb[..., 2] * 0.114
    render_gray = render_rgb[..., 0] * 0.299 + render_rgb[..., 1] * 0.587 + render_rgb[..., 2] * 0.114
    rows = []
    for layer, abs_box in _iter_leaf_layers_abs(layers):
        if layer.get("type") not in ("text", "shape", "image"):
            continue
        meta = layer.get("meta") or {}
        lid = str(layer.get("id") or "")
        if not lid or lid == "background":
            continue
        if str(meta.get("role") or "").lower() == "background" or meta.get("source") == "inpaint":
            continue
        pad = 2
        if layer.get("type") == "text":
            # Text boxes are emitted GENEROUSLY (>=1.6x lineHeight, Codia-style), so
            # scoring on the emitted box would pull neighbouring lines' ink into the
            # region. The compiler preserves the pre-growth ink box in meta — score
            # against that tight evidence box instead (same parent offset as `box`).
            prefit = meta.get("prefit_ink_box")
            own_box = layer.get("box") or {}
            if isinstance(prefit, dict) and prefit.get("w") and prefit.get("h"):
                abs_box = {
                    "x": abs_box["x"] + float(prefit.get("x", 0) or 0) - float(own_box.get("x", 0) or 0),
                    "y": abs_box["y"] + float(prefit.get("y", 0) or 0) - float(own_box.get("y", 0) or 0),
                    "w": float(prefit.get("w", 0) or 0),
                    "h": float(prefit.get("h", 0) or 0),
                }
            # Preview text may legitimately spill a little outside its fitted box;
            # a proportional margin also catches ghost ink right next to the box.
            pad = max(3, int(round(abs_box["h"] * 0.18)))
        x0 = max(0, int(np.floor(abs_box["x"])) - pad)
        y0 = max(0, int(np.floor(abs_box["y"])) - pad)
        x1 = min(width, int(np.ceil(abs_box["x"] + abs_box["w"])) + pad)
        y1 = min(height, int(np.ceil(abs_box["y"] + abs_box["h"])) + pad)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crop_source = source_gray[y0:y1, x0:x1]
        crop_render = render_gray[y0:y1, x0:x1]
        values = np.clip(
            np.asarray(_local_ssim_values(crop_source, crop_render, target_windows=4),
                       dtype=np.float64), 0, 1,
        )
        region_ssim = float(0.7 * values.mean() + 0.3 * np.percentile(values, 10))
        # Grayscale SSIM is blind to pure hue swaps on flat regions (a red button
        # rendered blue can score ~1.0); a local Lab delta catches exactly that.
        delta_e = float(np.linalg.norm(
            _rgb_to_lab(source_rgb[y0:y1, x0:x1]) - _rgb_to_lab(render_rgb[y0:y1, x0:x1]),
            axis=2,
        ).mean())
        row = {
            "id": lid,
            "type": layer.get("type"),
            "role": meta.get("role") or layer.get("type"),
            "abs_box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "region_px": int((x1 - x0) * (y1 - y0)),
            "region_ssim": round(region_ssim, 4),
            "region_color": round(max(0.0, 1.0 - delta_e / 50.0), 4),
        }
        if meta.get("fallback"):
            row["fallback"] = meta.get("fallback")
        if layer.get("type") == "text":
            source_ink = _region_ink_mask(source_rgb[y0:y1, x0:x1])
            render_ink = _region_ink_mask(render_rgb[y0:y1, x0:x1])
            if source_ink is not None and render_ink is not None:
                src_px = int(source_ink.sum())
                if src_px >= 8:
                    row["ink_iou"] = _aligned_ink_iou(source_ink, render_ink)
                    # Extra rendered ink relative to the source: double/ghosted text
                    # produces near-duplicate ink mass that IoU alone can miss.
                    row["ink_excess"] = round(int((render_ink & ~source_ink).sum()) / src_px, 4)
        rows.append(row)
    return rows


def _aligned_ink_iou(source_ink, render_ink, max_shift=None):
    """Best ink IoU over small translations of the rendered ink.

    The local preview is a placement PROXY: Figma performs its own render-and-fit,
    so a text block drawn a few pixels off its source baseline is repairable and
    must not be judged as a wrong reconstruction. Wrong glyph shapes, rotated
    baselines, ghost doubles, and missing text cannot be fixed by translation and
    stay low under the best alignment.
    """
    import numpy as np

    src_px = int(source_ink.sum())
    ren_px = int(render_ink.sum())
    if not src_px and not ren_px:
        return None
    if not src_px or not ren_px:
        return 0.0
    if max_shift is None:
        max_shift = max(4, int(round(source_ink.shape[0] * 0.25)))
        max_shift = min(max_shift, 12)
    best = 0.0
    h, w = source_ink.shape
    for dy in range(-max_shift, max_shift + 1):
        sy0, sy1 = max(0, dy), min(h, h + dy)
        ry0, ry1 = max(0, -dy), min(h, h - dy)
        if sy1 <= sy0:
            continue
        src_rows = source_ink[sy0:sy1]
        ren_rows = render_ink[ry0:ry1]
        for dx in range(-max_shift, max_shift + 1):
            sx0, sx1 = max(0, dx), min(w, w + dx)
            rx0, rx1 = max(0, -dx), min(w, w - dx)
            if sx1 <= sx0:
                continue
            inter = int(np.count_nonzero(src_rows[:, sx0:sx1] & ren_rows[:, rx0:rx1]))
            union = src_px + ren_px - inter
            if union and inter / union > best:
                best = inter / union
    return round(best, 4)


def score_layer_regions(source_path, render_path, design, run_dir=None):
    """Public loader wrapper around :func:`_layer_region_rows` (used by reconstruct)."""
    source_rgb = _load_rgb(source_path)
    height, width = source_rgb.shape[:2]
    render_rgb = _load_rgb(render_path, size=(width, height))
    design_data = _load_design(design, run_dir or "")
    return _layer_region_rows(source_rgb, render_rgb, design_data)


def _merge_region_rows(per_layer, region_rows):
    """Attach region metrics to existing per-layer rows; add rows for new layers."""
    by_id = {str(row.get("id")): row for row in per_layer if isinstance(row, dict)}
    for region in region_rows or []:
        rid = str(region.get("id"))
        existing = by_id.get(rid)
        if existing is None:
            per_layer.append(region)
            by_id[rid] = region
        else:
            for key, value in region.items():
                existing.setdefault(key, value)
    return per_layer


def _duplicate_ownership(layers):
    owners, duplicates = {}, []
    for layer in layers:
        lid = str(layer.get("id") or layer.get("name") or "unnamed")
        meta = layer.get("meta") or {}
        provenance = meta.get("provenance") or {}
        # Older fusion records stored provenance directly as a list; repair
        # rounds can reintroduce that shape. Normalize both forms so QA never
        # crashes while checking duplicate ownership.
        if isinstance(provenance, list):
            observations = provenance
        elif isinstance(provenance, dict):
            observations = provenance.get("observations") or meta.get("observations") or []
        else:
            observations = meta.get("observations") or []
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


def _text_editability(source_ocr, design, layers):
    """Honest text-editability accounting (F4).

    ``editable_text_recall`` = detected source text lines that ship as a CORRECT editable
    TEXT node / all detected source text lines that are NOT baked scene-text-by-design.

    Critically, a raster slice, a wordmark/lockup image, or a ``foreground_raster`` image
    that carries a text line counts as **non-editable** and LOWERS the recall — rasterizing
    failed overlay text is a quality loss, never a way to remove that text from the metric's
    denominator. The only text excluded from the denominator is ``kept_in_photo`` scene text
    (legitimately not-editable-by-design), which is tracked separately.

    Returns a dict of counts/ratios (or ``None`` when there is nothing to score):
      editable_text_recall, text_lines_total, kept_in_photo_lines,
      rasterized_text_count, rasterized_text_ratio, editable_text_correct.
    """
    if not source_ocr or not design:
        return None
    editable_texts = []
    raster_line_ids = set()
    raster_texts = []
    for layer in _flatten_layers(layers):
        meta = layer.get("meta") or {}
        ltype = layer.get("type")
        if ltype == "text":
            # A TEXT node is editable in Figma regardless of wordmark/lockup styling.
            norm = _norm(layer.get("text"))
            if norm:
                editable_texts.append(norm)
            continue
        # Image layers that carry text pixels are non-editable text: raster slices of failed
        # overlay copy, fidelity-image substitutions (legacy fallback=True headline images),
        # wordmark/lockup artwork, or whole-region foreground_raster fallbacks.
        is_raster_text = ltype == "image" and (
            _fallback_kind(meta) is not None
            or meta.get("wordmark") or meta.get("platform_lockup")
            or meta.get("layer_disposition") == "foreground_raster"
        )
        if not is_raster_text:
            continue
        for line_id in meta.get("line_ids") or []:
            raster_line_ids.add(str(line_id))
        norm = _norm(layer.get("text") or meta.get("source_text"))
        if norm:
            raster_texts.append(norm)

    editable_blob = " ".join(editable_texts)
    raster_blob = " ".join(raster_texts)
    kept_blob = " ".join(_norm(text) for text in (design.get("kept_in_photo") or []))

    total = kept = correct = rasterized = 0
    for line in source_ocr.get("lines", []):
        if line.get("conf", 1) < 0.5:
            continue
        norm = _norm(line.get("text"))
        if len(norm) < 3:
            continue
        line_id = str(line.get("id") or "")
        total += 1
        # By-design baked scene text is not something the pipeline promises to make editable.
        if kept_blob and norm and norm in kept_blob:
            kept += 1
            continue
        if norm and norm in editable_blob:
            correct += 1
        elif line_id in raster_line_ids or (raster_blob and norm and norm in raster_blob):
            rasterized += 1
        # else: the line is simply missing — not editable and not even rasterized. It counts
        # against recall (stays in the denominator, never in the numerator).
    denom = total - kept
    recall = 1.0 if denom <= 0 else correct / denom
    rasterized_ratio = (rasterized / denom) if denom > 0 else 0.0
    return {
        "editable_text_recall": round(recall, 4),
        # native_text_ratio (Codia contract, docs/CODIA-PARITY-SPEC.md §2): native editable
        # TEXT lines / all readable OCR lines that are NOT by-design baked scene text. Slices,
        # wordmark/lockup rasters, foreground_raster bakes, AND simply-missing lines all count
        # against it (they stay in the denominator, never in the numerator). This is the QA
        # objective — "every string is native TEXT" — surfaced under its contract name. It is
        # numerically the same fraction as editable_text_recall; the alias makes the contract
        # dimension nameable by codia_parity/qa_reward/benchmark without re-deriving it.
        "native_text_ratio": round(recall, 4),
        "text_lines_total": total,
        "kept_in_photo_lines": kept,
        "rasterized_text_count": rasterized,
        "rasterized_text_ratio": round(rasterized_ratio, 4),
        "editable_text_correct": correct,
    }


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
                "changed_canvas_ratio": 0.0,
                "edge_retention": None, "mean_change": 0.0,
                "outside_changed_ratio": 0.0, "outside_mean_change": 0.0}
    total_px = int(height) * int(width)
    delta = np.abs(source_rgb - background).mean(axis=2)
    exact = float((delta[mask] < 0.5).mean())
    changed = float((delta[mask] > 8.0).mean())
    # Fraction of the WHOLE canvas that was actually altered inside the removal region.
    # changed_ratio is per-mask; this is per-canvas, so it measures how much of the plate
    # the removal/inpaint pass destroyed regardless of how large the mask was (F3).
    changed_canvas = float(int((delta[mask] > 8.0).sum()) / max(1, total_px))
    mean_change = float(delta[mask].mean())
    outside = ~mask
    outside_changed = float((delta[outside] > 8.0).mean()) if np.any(outside) else 0.0
    outside_mean = float(delta[outside].mean()) if np.any(outside) else 0.0
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
        "changed_canvas_ratio": round(changed_canvas, 5),
        "edge_retention": None if retained is None else round(retained, 5),
        "mean_change": round(mean_change, 4),
        "outside_changed_ratio": round(outside_changed, 5),
        "outside_mean_change": round(outside_mean, 4),
    }


def _enclosed_alpha_holes(alpha):
    """Return enclosed transparent pixels and component count for an RGBA alpha matte.

    Transparency connected to an asset edge is ordinary cutout background. Transparent
    islands completely enclosed by opaque pixels are materially different: for photos,
    products, and people they are usually SAM/matting holes. The operation is deterministic
    and deliberately does not judge icons, where counters and rings are legitimate.
    """
    import cv2
    import numpy as np

    opaque = (np.asarray(alpha) > 16).astype(np.uint8)
    transparent = (opaque == 0).astype(np.uint8)
    if not transparent.any():
        return 0, 0
    exterior = np.zeros_like(transparent)
    count, labels = cv2.connectedComponents(transparent, connectivity=8)
    exterior_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    for label in exterior_labels:
        exterior[labels == label] = 1
    holes = transparent & (exterior == 0)
    if not holes.any():
        return 0, 0
    hole_count = len(set(labels[holes > 0]))
    return int(holes.sum()), int(hole_count)


def _layer_alpha_audit(layers, run_dir, thresholds):
    """Inspect actual raster mattes so corrupt/empty cutouts cannot pass screenshot QA."""
    import numpy as np
    from PIL import Image

    rows = []
    failures = []
    hole_roles = {"photo", "product", "person", "people", "portrait", "cutout", "image"}
    for layer in layers:
        if layer.get("type") != "image":
            continue
        meta = layer.get("meta") or {}
        role = str(meta.get("role") or "image").lower()
        if role == "background":
            continue
        src = layer.get("src")
        if not src or str(src).startswith("data:"):
            continue
        path = _resolve_path(src, run_dir)
        if not path or not os.path.exists(path):
            continue  # missing-assets owns this failure
        try:
            image = Image.open(path)
            if "A" not in image.getbands():
                continue
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        except Exception:
            continue  # corrupt assets are reported at the design/import boundary
        lid = str(layer.get("id") or layer.get("name") or "unnamed")
        opaque = int((alpha > 16).sum())
        total = int(alpha.size)
        coverage = opaque / max(1, total)
        hole_pixels, hole_count = _enclosed_alpha_holes(alpha)
        hole_fraction = hole_pixels / max(1, opaque + hole_pixels)
        row = {
            "id": lid, "role": role, "alpha_coverage": round(coverage, 5),
            "internal_hole_pixels": hole_pixels, "internal_hole_count": hole_count,
            "internal_hole_fraction": round(hole_fraction, 5),
        }
        rows.append(row)
        if opaque == 0:
            failures.append(("empty-layer-alpha", f"layer {lid} has no visible pixels"))
        elif (role in hole_roles and not meta.get("ownership_cutout") and hole_pixels >= 16
              and hole_fraction > thresholds["layer_internal_hole_fraction_max"]):
            failures.append((
                "layer-alpha-holes",
                f"layer {lid} has {hole_count} enclosed alpha hole(s) covering {hole_fraction:.1%}",
            ))
    return rows, failures


def _add_fail(fails, rule, detail):
    if not any(item.get("rule") == rule and item.get("detail") == detail for item in fails):
        fails.append({"rule": rule, "detail": detail, "hard": True})


def _archetype_threshold_overrides(run_dir):
    """Read the archetype preset's own edge/color floors straight from archetype.json.

    F-per-archetype-floor: only 2 of the 5 presets in src.archetype.PRESETS define
    edge_f1_min (social_screenshot 0.35, comparison_grid 0.45) and NONE define
    color_similarity_min, so most runs silently fell back to a single global default
    regardless of archetype — the same gap text_recall_min had before F8 threaded it
    through explicitly. archetype.json is written by every run (it is the persisted
    decision from src.archetype.decision), so pixel_diff can read the preset's own
    floors itself instead of depending on every caller to forward each key by hand.
    A DEFAULT_THRESHOLDS floor still always applies when the archetype doesn't define
    one, so a floor is guaranteed to exist per archetype either way — never silently
    absent. Explicit caller-supplied ``thresholds=`` still wins (merged in after this).
    """
    payload = _load_design(os.path.join(run_dir, "archetype.json"), run_dir)
    if not isinstance(payload, dict):
        return {}
    preset_thresholds = (payload.get("preset") or {}).get("thresholds") or {}
    if not isinstance(preset_thresholds, dict):
        return {}
    overrides = {}
    # native_text_ratio_min lets a future handwriting/scene-text archetype relax the contract
    # bar; contract_* floors let an archetype tune the contract summary's SSIM/placement floor.
    for key in ("edge_f1_min", "color_similarity_min", "native_text_ratio_min",
                "contract_ssim_floor", "contract_placement_ink_iou_min"):
        value = preset_thresholds.get(key)
        if value is not None:
            try:
                overrides[key] = float(value)
            except (TypeError, ValueError):
                continue
    return overrides


def _structural_audit(
    source_rgb,
    run_dir,
    design,
    source_ocr,
    supplied,
    background_path,
    removal_mask,
    thresholds,
    text_recall=None,
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
    compiler_errors_from_design = []
    for warning in ((design or {}).get("meta") or {}).get("warnings") or []:
        code = str(warning.get("code", "")) if isinstance(warning, dict) else ""
        if code in ("missing-asset", "corrupt-asset"):
            missing_assets.append(str(warning.get("layer_id") or warning.get("path") or warning))
        if code in ("missing-font", "font-load-failed"):
            missing_fonts.append(str(warning.get("layer_id") or warning.get("font") or warning))
        if code == "invalid-schema":
            schema_errors.append(str(warning.get("detail") or warning))
        if code == "layer-compile-error":
            compiler_errors_from_design.append(str(warning.get("detail") or warning))

    figma_report = {}
    report_path = os.path.join(run_dir, "figma_report.json")
    design_path = os.path.join(run_dir, "design.json")
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
    require_figma = bool(supplied.get("require_figma_report"))
    figma_warnings = list(figma_report.get("warnings") or [])
    fidelity = figma_report.get("fidelity") or {}
    figma_unsupported = sum(int(fidelity.get(key, 0) or 0) for key in (
        "unsupported_paint", "unsupported_stroke", "unsupported_effect",
        "unsupported_paints", "unsupported_strokes", "unsupported_effects",
    ))
    if int((figma_report.get("assets") or {}).get("missing", 0) or 0) > 0:
        missing_assets.append("Figma compiler report")
    font_substitutions = list((figma_report.get("fonts") or {}).get("selections") or [])
    # If the plugin substituted a requested font, that is a real fidelity/structure issue
    # even when Figma did not raise a load error (e.g. "closest installed style" fallback).
    # Treat it as "missing" for QA gating so it cannot silently pass.
    for entry in font_substitutions:
        if not isinstance(entry, dict):
            continue
        requested = str(entry.get("requested") or "").strip()
        selected = str(entry.get("selected") or "").strip()
        if not requested or not selected:
            continue
        if requested.casefold() == selected.casefold():
            continue
        label = str(entry.get("label") or "text").strip()
        missing_fonts.append(f"{label}: {requested} → {selected}")
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
    text_editability = _text_editability(source_ocr, design, layers)
    editable_text_recall = None if text_editability is None else text_editability["editable_text_recall"]
    native_text_ratio_metric = None if text_editability is None else text_editability["native_text_ratio"]
    rasterized_text_count = None if text_editability is None else text_editability["rasterized_text_count"]
    rasterized_text_ratio = None if text_editability is None else text_editability["rasterized_text_ratio"]
    # F-honesty: editable_text_recall's own denominator is "detected source text lines"
    # (from OCR), never the FULL source text. When OCR itself only finds a sliver of the
    # ad's copy, a 100%-editable sliver still reads as a perfect 1.0 and hides the loss.
    # true_text_coverage multiplies in text_recall (OCR-detected / all-source-text) so the
    # combined number is honest about the share of ALL source text that ended up correct
    # AND editable — it can only be high when both stages are.
    true_text_coverage = (
        None if text_recall is None or editable_text_recall is None
        else round(float(text_recall) * float(editable_text_recall), 4)
    )
    design_meta = (design or {}).get("meta") or {}
    leaf_accounting = design_meta.get("leaf_accounting")
    if not isinstance(leaf_accounting, dict):
        leaf_accounting = None
    native_leaf_ratio = design_meta.get("native_leaf_ratio")
    if native_leaf_ratio is None and leaf_accounting:
        native_leaf_ratio = leaf_accounting.get("native_leaf_ratio")
    native_leaf_ratio = None if native_leaf_ratio is None else float(native_leaf_ratio)
    require_native_accounting = bool(supplied.get("require_native_accounting"))

    duplicate_ownership = _duplicate_ownership(layers)
    duplicate_ownership.extend(
        _reported_items(supplied.get("duplicate_ownership"), "duplicate ownership")
    )
    duplicate_ownership = sorted(set(duplicate_ownership))

    reconstruction = _load_reconstruction(run_dir)
    element_survival = _element_survival_audit(run_dir, reconstruction)
    reconstruction_stats = reconstruction.get("stats") or {}
    degradations = design_meta.get("degradations")
    degradations = degradations if isinstance(degradations, list) else []

    # Confidence-gated raster slices are honest, reported degradations of editability,
    # never hidden: QA surfaces the count/ids so 'looks right + partially editable'
    # is a visible, auditable tradeoff instead of a silent one.
    raster_slice_ids = sorted(
        str(layer.get("id") or "unnamed") for layer in layers
        if _is_raster_slice(layer.get("meta") or {})
    )

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
    alpha_layers, alpha_failures = _layer_alpha_audit(layers, run_dir, thresholds)
    explicit_leakage = supplied.get("background_leakage")

    fails = []
    if require_figma and not figma_report:
        _add_fail(fails, "figma-report-missing", "Figma plugin report not received — import+export in Figma desktop")
    elif require_figma and figma_report and os.path.exists(design_path):
        try:
            if os.path.getmtime(report_path) < os.path.getmtime(design_path):
                _add_fail(fails, "figma-report-stale", "figma_report.json is older than design.json")
        except OSError:
            pass
    if figma_warnings and require_figma:
        _add_fail(fails, "figma-compiler-warnings",
                  f"{len(figma_warnings)} plugin warning(s): " + str(figma_warnings[0].get("detail") or figma_warnings[0])[:120])
    if figma_unsupported and require_figma:
        _add_fail(fails, "figma-fidelity-fallback",
                  f"{figma_unsupported} unsupported paint/stroke/effect fallback(s) in plugin")
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
    if compiler_errors_from_design:
        _add_fail(fails, "layer-compile-errors",
                  f"{len(compiler_errors_from_design)} layer(s) could not compile: " +
                  "; ".join(compiler_errors_from_design[:3]))
    source_lines = (source_ocr or {}).get("lines", []) if isinstance(source_ocr, dict) else []
    # Photographic-scene exemption: when merge's evidence-based verdict says ALL source
    # text is in-scene photo content (021: "BUY TWO … FOR FREE" printed inside the photo)
    # and every confident OCR line is accounted for in design.kept_in_photo, a 1-layer
    # photo output is the CONTRACT-CORRECT answer (Codia ships the same). Editability
    # floors must not punish it. The verdict comes from source-image facts
    # (photo_coverage etc.), never from the output, so a bad reconstruction cannot fake
    # its way into this exemption — and any text NOT accounted for keeps full strictness.
    scene_baked_all_text = False
    try:
        with open(os.path.join(run_dir, "merge_report.json"), encoding="utf-8") as fh:
            _photo_scene = bool(json.load(fh).get("photographic_scene_text"))
    except Exception:
        _photo_scene = False
    if _photo_scene:
        kept_norm = [_norm(t) for t in ((design or {}).get("kept_in_photo") or [])]
        src_norm = [_norm(l.get("text", "")) for l in source_lines
                    if l.get("conf", 1) >= 0.5 and len(_norm(l.get("text", ""))) >= 3]
        scene_baked_all_text = bool(src_norm) and all(
            any(s == k or s in k or k in s for k in kept_norm) for s in src_norm)
    # Reconciliation (audit: 009/021): the exemption must ONLY hold for a genuinely
    # photographic archetype whose layer tree is real. Screenshots (social/tweet/DM) and
    # outputs full of empty junk groups must keep full editability strictness even if merge
    # tagged the source text photographic.
    if scene_baked_all_text:
        _block_reason = _scene_baked_exemption_block_reason(run_dir, design)
        if _block_reason:
            scene_baked_all_text = False
            supplied.setdefault("scene_baked_exemption_denied", _block_reason)
    if source_lines and editable_ratio is not None and editable_ratio < thresholds["editable_ratio_min"]:
        if scene_baked_all_text:
            supplied.setdefault("scene_baked_photo", True)
        else:
            _add_fail(fails, "low-editable-ratio",
                      f"editable ratio {editable_ratio:.2f} < {thresholds['editable_ratio_min']:.2f}")
    if require_native_accounting and leaf_accounting is None:
        _add_fail(
            fails, "native-accounting-missing",
            "acceptance requires foreground leaf accounting; rebuild design.json with the current compiler",
        )
    # F2: the anti-rasterization honesty gates are NOT keyed to a Figma acceptance run. They
    # fire whenever leaf accounting exists (it always does now) so a page that quietly
    # rasterized nearly everything, or emitted an unexplained raster fallback, hard-fails in
    # ordinary QA — not only when someone opts into figma.require_export. Config-gated ON.
    if leaf_accounting and thresholds.get("enforce_native_leaf_accounting", True):
        unexplained = int(leaf_accounting.get("unexplained_raster_count", 0) or 0)
        if unexplained:
            ids = ", ".join(str(value) for value in
                            (leaf_accounting.get("unexplained_raster_ids") or [])[:4])
            _add_fail(
                fails, "unexplained-raster-fallback",
                f"{unexplained} raster fallback(s) have no semantic or fidelity reason"
                + (f": {ids}" if ids else ""),
            )
        foreground_leaf_count = int(leaf_accounting.get("foreground_leaf_count", 0) or 0)
        ratio = leaf_accounting.get("native_leaf_ratio")
        ratio = native_leaf_ratio if ratio is None else float(ratio)
        if (foreground_leaf_count > 1 and ratio is not None
                and ratio < thresholds["native_leaf_ratio_min"]
                and not scene_baked_all_text):
            _add_fail(
                fails, "low-native-leaf-ratio",
                f"native leaf ratio {ratio:.2f} < {thresholds['native_leaf_ratio_min']:.2f} "
                f"over {foreground_leaf_count} foreground leaf(ves) — almost everything was rasterized",
            )
    low_editable_text_recall = (
        editable_text_recall is not None
        and editable_text_recall < thresholds["editable_text_recall_min"]
    )
    true_text_coverage_min = thresholds.get("true_text_coverage_min")
    low_true_text_coverage = (
        true_text_coverage is not None and true_text_coverage_min is not None
        and true_text_coverage < float(true_text_coverage_min)
    )
    # F-honesty: an ad where OCR missed most of the text must not pass the text gate just
    # because the sliver it did find happens to be 100% editable (021: text_recall 0.17,
    # editable_text_recall 1.0). true_text_coverage catches that denominator trick even
    # when editable_text_recall alone clears its own bar.
    if (low_editable_text_recall or low_true_text_coverage) and scene_baked_all_text:
        low_editable_text_recall = low_true_text_coverage = False
        supplied.setdefault("scene_baked_photo", True)
    if low_editable_text_recall or low_true_text_coverage:
        details = []
        if low_editable_text_recall:
            details.append(
                f"editable text recall {editable_text_recall:.2f} < {thresholds['editable_text_recall_min']:.2f}"
            )
        if low_true_text_coverage:
            details.append(
                f"true text coverage {true_text_coverage:.2f} < {float(true_text_coverage_min):.2f} "
                f"(text_recall {text_recall:.2f} x editable_text_recall {editable_text_recall:.2f} "
                "— OCR missed most of the source text)"
            )
        _add_fail(fails, "missing-editable-text", "; ".join(details))
    if duplicate_ownership:
        _add_fail(fails, "duplicate-ownership",
                  f"{len(duplicate_ownership)} observation ownership conflict(s): " + duplicate_ownership[0])
    if (element_survival and element_survival.get("recall") is not None
            and element_survival["recall"] < thresholds["element_survival_min"]):
        _add_fail(
            fails,
            "low-element-recall",
            f"only {element_survival['kept']}/{element_survival['proposed']} detected elements "
            "survived reconstruction; missing " + ", ".join(element_survival["missing_ids"][:4]),
        )
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
        if (background.get("mask_supplied")
                and background.get("outside_changed_ratio", 0.0)
                > thresholds["background_outside_damage_max"]):
            _add_fail(
                fails,
                "inpaint-outside-mask",
                "clean plate changed "
                f"{background['outside_changed_ratio']:.1%} of pixels outside the removal mask",
            )
        # F3: cap plate destruction. There are gates for an untouched plate and a no-op
        # removal, but none for the opposite failure — a removal/inpaint that rebuilds most
        # of the canvas (002 erased the whole product cluster, changed_canvas_ratio 0.69).
        # When more than the configured fraction of the WHOLE canvas is altered inside the
        # removal region, real content was almost certainly erased.
        canvas_ratio = background.get("changed_canvas_ratio")
        ceiling = thresholds.get("background_changed_ratio_max")
        if (background.get("mask_supplied") and canvas_ratio is not None
                and ceiling is not None and canvas_ratio > float(ceiling)):
            _add_fail(
                fails,
                "excessive-plate-destruction",
                f"removal/inpaint altered {canvas_ratio:.1%} of the canvas "
                f"(> {float(ceiling):.0%}) — likely erased real content, not just a removed object",
            )
    # F15: unresolved glyph residue under a removed text region is a structural failure, not
    # a bare repair suggestion. QA must not report ok while it stands (009 shipped ok with a
    # high-severity glyph-residue repair still outstanding after no-op harness rounds).
    if thresholds.get("glyph_residue_gate", True):
        text_residual = reconstruction_stats.get("text_residual") or {}
        unresolved_residue = [
            entry for entry in (text_residual.get("flagged") or [])
            if isinstance(entry, dict) and not entry.get("resolved")
        ]
        if unresolved_residue:
            ids = ", ".join(str(entry.get("id")) for entry in unresolved_residue[:4])
            _add_fail(
                fails, "glyph-residue",
                f"{len(unresolved_residue)} removed text region(s) still show glyph residue: {ids}",
            )
    for rule, detail in alpha_failures:
        _add_fail(fails, rule, detail)
    if explicit_leakage:
        _add_fail(fails, "background-leakage", f"reported background leakage: {explicit_leakage}")

    # Preserve externally supplied structural failures without trusting only one caller field.
    for item in supplied.get("hard_fails", []) or []:
        if isinstance(item, dict) and item.get("rule"):
            _add_fail(fails, str(item["rule"]), str(item.get("detail", item["rule"])))

    # Degradation propagation from upstream stages must only gate acceptance runs, never
    # ordinary diagnostic compares -- both are gated behind require_native_accounting.
    if require_native_accounting and reconstruction_stats.get("opencv_fallback_used"):
        _add_fail(
            fails, "inpaint-degraded-opencv",
            "background plate used low-quality OpenCV fallback",
        )
    if require_native_accounting and degradations:
        reasons = ", ".join(
            str((item or {}).get("reason") or item) if isinstance(item, dict) else str(item)
            for item in degradations[:4]
        )
        _add_fail(
            fails, "pipeline-degraded",
            f"{len(degradations)} pipeline degradation(s): {reasons}",
        )

    stats = reconstruction_stats
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
        "native_leaf_ratio": None if native_leaf_ratio is None else round(native_leaf_ratio, 4),
        "leaf_accounting": leaf_accounting,
        "editable_text_recall": None if editable_text_recall is None else round(editable_text_recall, 4),
        # native_text_ratio (Codia contract): native editable TEXT / all readable lines.
        "native_text_ratio": None if native_text_ratio_metric is None else round(native_text_ratio_metric, 4),
        "true_text_coverage": true_text_coverage,
        "rasterized_text_count": rasterized_text_count,
        "rasterized_text_ratio": rasterized_text_ratio,
        "raster_slices": {"count": len(raster_slice_ids), "ids": raster_slice_ids},
        "duplicate_ownership": duplicate_ownership,
        "duplicates_removed": int(stats.get("duplicates_removed", supplied.get("duplicates_removed", 0)) or 0),
        "element_recall": None if element_survival is None else element_survival["recall"],
        "element_survival": element_survival,
        "background": background,
        "layer_alpha": alpha_layers,
        "hard_fails": fails,
    }


def _placement_ink_iou(per_layer):
    """Mean translation-aligned ink-IoU over native text rows (contract placement).

    Figma re-fits glyph placement, so this is the lenient "decent placement" signal, not a
    pixel-exact one. Returns None when no text row carries ink evidence.
    """
    values = [row.get("ink_iou") for row in per_layer or []
              if isinstance(row, dict) and row.get("type") == "text"
              and isinstance(row.get("ink_iou"), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _contract_summary(design_data, structure, source_ocr, per_layer, ssim, thresholds,
                      archetype=None):
    """Lead-with-the-contract QA summary (docs/CODIA-PARITY-SPEC.md).

    Scores Codia's construction CONTRACT — native editable text, Inter/display font policy,
    single-weight nodes, emoji-as-image, node budget, flatness — and folds in the two gates
    the contract also demands: zero unresolved glyph residue and decent placement. Global
    SSIM is a low floor here, never the objective. ``contract_score`` is the QA metric of
    record; ``pass`` is the per-run contract verdict the benchmark's --contract line uses.
    """
    native_text_ratio = structure.get("native_text_ratio")
    if native_text_ratio is None:
        native_text_ratio = structure.get("editable_text_recall")

    construction = None
    if _score_construction is not None and design_data:
        try:
            construction = _score_construction(
                design_data, native_text_ratio=native_text_ratio,
                archetype=archetype, ocr=source_ocr)
        except Exception:
            construction = None

    # Glyph-residue cleanliness: a removed text region still showing glyph ghosts is a
    # contract failure (the plate is not clean). Mirror the glyph-residue hard-fail gate.
    residue_unresolved = 0
    for fail in structure.get("hard_fails") or []:
        if isinstance(fail, dict) and fail.get("rule") == "glyph-residue":
            residue_unresolved += 1
    glyph_residue_clean = residue_unresolved == 0

    placement_iou = _placement_ink_iou(per_layer)
    placement_min = thresholds.get("contract_placement_ink_iou_min")
    placement_ok = (placement_iou is None or placement_min is None
                    or placement_iou >= float(placement_min))

    ntr_min = thresholds.get("native_text_ratio_min")
    native_ok = (native_text_ratio is None or ntr_min is None
                 or float(native_text_ratio) >= float(ntr_min))
    ssim_floor = thresholds.get("contract_ssim_floor")
    ssim_floor_ok = ssim is None or ssim_floor is None or float(ssim) >= float(ssim_floor)

    # contract_score (0..1) LEADS with native text, then construction quality, with global
    # SSIM contributing only a small floor-shaped term. This is the QA composite of record.
    construction_norm = None if construction is None else construction["score"] / 100.0
    terms = []
    if native_text_ratio is not None:
        terms.append((0.50, float(native_text_ratio)))
    if construction_norm is not None:
        terms.append((0.35, construction_norm))
    if ssim is not None:
        terms.append((0.15, float(ssim)))
    if terms:
        wsum = sum(w for w, _ in terms)
        contract_score = round(sum(w * v for w, v in terms) / wsum, 4)
    else:
        contract_score = None

    contract_pass = bool(native_ok and glyph_residue_clean and placement_ok and ssim_floor_ok)
    return {
        "contract_score": contract_score,
        "pass": contract_pass,
        "native_text_ratio": None if native_text_ratio is None else round(float(native_text_ratio), 4),
        "native_text_ratio_min": ntr_min,
        "native_text_ok": bool(native_ok),
        "glyph_residue_clean": glyph_residue_clean,
        "glyph_residue_unresolved": residue_unresolved,
        "placement_ink_iou": placement_iou,
        "placement_ok": bool(placement_ok),
        "ssim": None if ssim is None else round(float(ssim), 4),
        "ssim_floor": ssim_floor,
        "ssim_floor_ok": bool(ssim_floor_ok),
        "construction": construction,
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
    resolved_removal = removal_mask
    if resolved_removal is None:
        candidate = os.path.join(run_dir, "removal_mask.png")
        resolved_removal = candidate if os.path.exists(candidate) else None
    elif isinstance(resolved_removal, str):
        if not os.path.isabs(resolved_removal) and not os.path.exists(resolved_removal):
            resolved_removal = _resolve_path(resolved_removal, run_dir)
    preserve_mask = None
    if resolved_removal:
        mask_arr = _load_mask(resolved_removal, (width, height))
        if mask_arr is not None:
            # Convert the uint8 removal mask (0/255) to a boolean "preserve" mask.
            # Using bitwise inversion on uint8 produces another uint8 array which can
            # interact poorly with later boolean ops, occasionally zeroing edge metrics.
            preserve_mask = (mask_arr == 0)
    reconstruction_ssim, recon_scales, recon_local = _multiscale_ssim(source_gray, render_gray)
    if preserve_mask is not None:
        multiscale, per_scale, local = _multiscale_ssim(source_gray, render_gray, preserve_mask)
    else:
        multiscale, per_scale, local = reconstruction_ssim, recon_scales, recon_local
    edge = _edge_metrics(source_gray, render_gray, preserve_mask)
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
        text_recall = _text_recall(source_ocr, render_ocr, source_gray, render_gray)

    opts = dict(DEFAULT_THRESHOLDS)
    # F-per-archetype-floor: apply the archetype preset's own edge/color floors (if any)
    # before the caller's explicit thresholds, so an explicit override always still wins.
    opts.update(_archetype_threshold_overrides(run_dir))
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
        text_recall=text_recall,
    )
    quality_flags = []
    if multiscale < opts["local_ssim_min"]:
        quality_flags.append({"rule": "local-ssim", "detail": f"{multiscale:.3f} < {opts['local_ssim_min']:.3f}"})
    if edge["f1"] < opts["edge_f1_min"]:
        quality_flags.append({"rule": "edge-fidelity", "detail": f"{edge['f1']:.3f} < {opts['edge_f1_min']:.3f}"})
    if color["similarity"] < opts["color_similarity_min"]:
        quality_flags.append({"rule": "color-fidelity", "detail": f"{color['similarity']:.3f} < {opts['color_similarity_min']:.3f}"})
    # F8: per-archetype text strictness. The archetype preset's text_recall_min (0.90 for
    # social) is threaded into thresholds by the caller; enforce it here so the strict text
    # bar the preset promises actually gates instead of being wired nowhere. Only fires when
    # a render-OCR text_recall exists AND a threshold was supplied.
    text_recall_min = opts.get("text_recall_min")
    if text_recall is not None and text_recall_min is not None and text_recall < float(text_recall_min):
        quality_flags.append({"rule": "low-text-recall",
                              "detail": f"text recall {text_recall:.3f} < {float(text_recall_min):.3f}"})

    # F-worst-region: gate the single worst local SSIM window independently of the
    # mean-dominated aggregate, so a catastrophic region (009/016: worst window ~0.03-0.04)
    # cannot hide under a good global/aggregate score. Evidence carries the pixel bbox.
    worst_window = _local_ssim_worst_window(source_gray, render_gray, preserve_mask)
    worst_window_min = opts.get("local_ssim_worst_window_min")
    if (worst_window is not None and worst_window_min is not None
            and worst_window["ssim"] < float(worst_window_min)):
        bbox = worst_window["bbox"]
        quality_flags.append({
            "rule": "local-ssim-worst-region",
            "detail": (
                f"worst local window ssim {worst_window['ssim']:.3f} < {float(worst_window_min):.3f} "
                f"at x={bbox['x']} y={bbox['y']} w={bbox['w']} h={bbox['h']}"
            ),
            "bbox": bbox,
        })

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
    # PER-LAYER region scores (crop SSIM + text ink-IoU) make one wrong region visible
    # even under a good global score, and drive the raster-slice confidence fallback.
    per_layer = _merge_region_rows(per_layer, _layer_region_rows(source_rgb, render_rgb, design_data))

    # ── CODIA CONSTRUCTION CONTRACT: the QA objective, scored ahead of SSIM. ──────────
    # SSIM stays a REPORT + a low floor gate; the contract (native text, font policy,
    # emoji-as-image, clean plate, placement) is what pass/fail and the reward now lead with.
    archetype = None
    arch_decision = _load_design(os.path.join(run_dir, "archetype.json"), run_dir)
    if isinstance(arch_decision, dict):
        archetype = arch_decision.get("archetype")
    if archetype is None:
        archetype = ((design_data or {}).get("meta") or {}).get("archetype")
    contract = _contract_summary(design_data, structure, source_ocr, per_layer,
                                 round(multiscale, 4), opts, archetype=archetype)

    return {
        # Compatibility: callers still read `ssim`, now the harder local/multiscale metric.
        "ssim": round(multiscale, 4),
        "global_ssim": round(global_ssim, 4),
        "multiscale_ssim": round(multiscale, 4),
        "reconstruction_ssim": round(reconstruction_ssim, 4),
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
        # ── Codia construction contract (the QA objective — see _contract_summary) ────
        # contract_score is the composite of record (native text first, then construction
        # quality, then a small SSIM floor term). contract.pass is the per-run verdict.
        # native_text_ratio / construction are surfaced for benchmark columns + the reward.
        "native_text_ratio": structure.get("native_text_ratio"),
        "contract_score": contract.get("contract_score"),
        "contract_pass": contract.get("pass"),
        "contract": contract,
        "construction": contract.get("construction"),
        # F-honesty: mirror element_recall/element_survival and true_text_coverage to the
        # top level the same way editable_text_recall/rasterized_text_* already are. Nested
        # structural.element_recall was always computed correctly (see _element_survival_audit)
        # but never hoisted here when editable_text_recall's top-level mirror was added, so any
        # caller reading qa.get("element_recall") directly (rather than
        # qa["structural"]["element_recall"]) — including a bare ``jq .element_recall qa.json``
        # look — always saw null even on runs where the nested value was 1.0.
        "element_recall": structure.get("element_recall"),
        "element_survival": structure.get("element_survival"),
        "true_text_coverage": structure.get("true_text_coverage"),
        "rasterized_text_count": structure.get("rasterized_text_count"),
        "rasterized_text_ratio": structure.get("rasterized_text_ratio"),
        "local_ssim_worst_window": worst_window,
        "per_layer": per_layer,
        "per_region_max_delta": float(cells.max()),
        "per_region": {"rows": gy, "cols": gx, "mean_delta": cells.round(3).tolist(), "worst": ranked[:8]},
        "diff_png": diff_png,
        "structural": structure,
        "hard_fails": hard_fails,
    }
