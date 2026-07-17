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
    # F-colour-honesty: corroboration floor for a color-fidelity HARD fail. Scored on
    # `local_similarity` (colour measured where colour is comparable), NOT on the whole-image
    # number above — the two are different scales on purpose and 0.98 is calibrated from
    # measured data, not carried over. See the `_color_metrics` docstring for the full table:
    # every real bench-10 render lands at 0.989-1.000 while injected colour defects land at
    # 0.970 and below, so 1.0 dE sits in the empty gap between them.
    "color_local_similarity_min": 0.98,
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
    # F-worst-region-honesty: the floor above fired on 13 of 16 bench-10 fixtures, which made
    # it useless as a signal — it no longer discriminated good from bad. Montaging every
    # bench-9 worst window (work/worst_windows.png) showed EVERY one lands on a text glyph,
    # and most are the same benign phenomenon: we substitute fonts (a font agent measured 41
    # of 99 delivered text nodes drawing a different face in preview), so glyph advance
    # widths differ slightly and positions DRIFT along a text run — 0px at the run's anchor,
    # 20-30px by its far end. A 64px window sitting at the END of a run then compares "tail
    # of a glyph" against "blank" and collapses to ~0.02 SSIM for text a designer cannot tell
    # apart (measured: 009 "We zien je", 104 "ration", 101 "ARE BU", 107 "EEK 4").
    #
    # The fix is NOT a looser floor (that would trade one lie for another) — it is asking
    # whether a pure TRANSLATION explains the window. The test is deliberately SYMMETRIC:
    #   forward  = source window vs the render searched over +/-radius   (catches deletions)
    #   backward = render window vs the source searched over +/-radius   (catches ghosts)
    # and a window is only excused when min(forward, backward) clears the bar. One-sided
    # search would excuse an ADDED ghost double (source blank -> some nearby render patch is
    # also blank -> "explained"); requiring the reverse match too closes that hole, because a
    # fabricated ghost has no counterpart anywhere in the source. Measured on the real
    # artifacts (see the table in the F-worst-region-honesty notes on _translation_explains).
    #
    # Radius is HALF a window and no more. Beyond that the search stops measuring
    # displacement and starts finding coincidental look-alikes: 025's genuinely missing emoji
    # (bench-9, x=64 y=896) stays at 0.096 out to radius 32 but spuriously matches a
    # look-alike wood-grain patch at 0.639 once radius reaches 48. Half a window is also the
    # honest semantic boundary — content displaced by more than half a window is misplaced,
    # not drifted.
    "local_ssim_shift_radius_ratio": 0.5,
    # The bar for "a translation EXPLAINS this window" is deliberately much higher than the
    # 0.10 fail floor: being merely no-longer-failing after shifting is not an explanation. A
    # real drift snaps to a GOOD match (measured 0.59-0.98); genuine damage does not come
    # close (025 missing emoji 0.096, 016 60px-misplaced headline 0.012, 067 0.155, 135
    # wrong-weight glyphs 0.422). 0.50 sits in the empty gap between those two populations.
    "local_ssim_shift_explained_min": 0.50,
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


def _window_grid(a, preserve_mask=None, target_windows=10):
    """The exact non-overlapping window grid `_local_ssim_values` scores, as bboxes.

    Single source of truth for "which windows are scored", so the worst-window gate and the
    aggregate can never drift apart on geometry.
    """
    h, w = a.shape
    window = max(8, min(64, int(round(min(h, w) / max(1, target_windows)))))
    out = []
    for y in range(0, h, window):
        for x in range(0, w, window):
            hh = min(h, y + window) - y
            ww = min(w, x + window) - x
            if hh <= 0 or ww <= 0:
                continue
            if preserve_mask is not None:
                km = preserve_mask[y : y + hh, x : x + ww]
                if km.size == 0 or float(km.mean()) < 0.70:
                    continue
            out.append({"x": int(x), "y": int(y), "w": int(ww), "h": int(hh)})
    return out, window


def _window_at(image, x, y, w, h):
    if x < 0 or y < 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return None
    return image[y : y + h, x : x + w]


def _translation_explains(source_gray, render_gray, box, radius, bar):
    """Is this window's low SSIM explained by a pure local translation of intact content?

    Symmetric on purpose — see the ``local_ssim_shift_radius_ratio`` threshold notes.
    ``forward`` alone would excuse a fabricated ghost; ``backward`` alone would excuse a
    deletion. Only content that is present on BOTH sides, merely displaced, clears both.

    Measured on the real bench-9/bench-10 artifacts (radius 32, bar 0.50):

        fixture  raw     fwd    bwd    min     verdict   what it actually is
        009      0.011   0.982  1.000  0.982   drift     "We zien je" tail, identical face
        131     -0.091   0.985  0.934  0.934   drift
        091      0.046   0.902  0.918  0.902   drift
        066      0.024   0.959  1.000  0.959   drift
        101      0.016   0.964  1.000  0.964   drift     window blank in both to the eye
        104      0.018   0.867  1.000  0.867   drift     "ration", tail clips the corner
        094     -0.031   0.975  0.841  0.841   drift
        002     -0.099   0.829  0.841  0.829   drift
        013      0.070   0.853  0.977  0.853   drift*    *italic-vs-upright; see below
        107      0.024   0.593  0.679  0.593   drift     "EEK 4", visually identical
        135     -0.023   0.422  0.947  0.422   DAMAGE    renders "OP" bold, source is light
        067      0.001   0.155  0.864  0.155   DAMAGE
        025 b9   0.074   0.096  0.982  0.096   DAMAGE    emoji missing entirely
        016     -0.017   0.012  0.982  0.012   DAMAGE    headline misplaced by 60px

    The two populations do not overlap: worst drift 0.593, best damage 0.422.

    *013 is the one deliberate soft edge. Its window catches our render drawing ITALIC where
    the source is upright (the declared-Bold/italic-file bug), and translation "explains" it
    at 0.853 because the same glyph exists a few px away. It is not lost: the
    deliverable-consistency check (`_font_consistency_audit`) names it exactly — "declares
    Bold, preview drew Italic" — which is a far better report than an anonymous 0.07 window.

    Both directions are scored at ONE SHARED displacement (forward at ``+shift``, backward at
    ``-shift``) rather than each picking its own best. A real translation is a single vector:
    if the content moved by (dx,dy) then source@(x,y) matches render@(x+dx,y+dy) AND
    render@(x,y) matches source@(x-dx,y-dy). Letting the two directions choose independent
    shifts would let a window be "explained" by two unrelated coincidental look-alikes.
    """
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
    src_win = source_gray[y : y + bh, x : x + bw]
    ren_win = render_gray[y : y + bh, x : x + bw]
    best, best_shift, best_pair = -2.0, (0, 0), (0.0, 0.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ahead = _window_at(render_gray, x + dx, y + dy, bw, bh)
            behind = _window_at(source_gray, x - dx, y - dy, bw, bh)
            if ahead is None or behind is None:
                continue
            forward = float(_ssim(src_win, ahead))
            if forward <= best:
                continue  # min(forward, backward) cannot beat `best` — skip the 2nd SSIM
            backward = float(_ssim(ren_win, behind))
            agreed = min(forward, backward)
            if agreed > best:
                best, best_shift, best_pair = agreed, (int(dx), int(dy)), (forward, backward)
    if best < -1.0:  # degenerate: every shift left the image
        value = float(_ssim(src_win, ren_win))
        best, best_pair = value, (value, value)
    return {
        "forward": round(float(best_pair[0]), 5),
        "backward": round(float(best_pair[1]), 5),
        "shift_tolerant_ssim": round(float(best), 5),
        "shift": {"dx": best_shift[0], "dy": best_shift[1]},
        "explained": bool(best >= float(bar)),
    }


def _classify_subfloor_windows(source_gray, render_gray, preserve_mask, floor, radius_ratio,
                               bar, max_windows=48):
    """Split every below-floor window into genuine DAMAGE vs benign translation DRIFT.

    Checking every sub-floor window (not just the single worst) matters: otherwise a drift
    artifact that happens to score worst would mask a real defect sitting one window over.
    Cheap in practice — measured 0-26 sub-floor windows per bench-10 fixture (median 2).
    """
    boxes, window = _window_grid(source_gray, preserve_mask)
    radius = max(1, int(round(window * float(radius_ratio))))
    scored = []
    for box in boxes:
        pa = source_gray[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
        pb = render_gray[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
        value = float(_ssim(pa, pb))
        if value < float(floor):
            scored.append((value, box))
    scored.sort(key=lambda item: item[0])
    damage, drift = [], []
    # Bound the work on pathological runs (067: 26 sub-floor windows). The cap is far above
    # the observed maximum and windows are examined worst-first, so a real defect cannot be
    # cropped out by it in practice.
    for value, box in scored[:max_windows]:
        verdict = _translation_explains(source_gray, render_gray, box, radius, bar)
        entry = {"ssim": round(value, 5), "bbox": box, **verdict}
        (drift if verdict["explained"] else damage).append(entry)
    return {"damage": damage, "drift": drift, "radius": radius,
            "subfloor_total": len(scored), "examined": len(scored[:max_windows])}


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


def _local_color_error(source_lab, render_lab, radius=16, step=2):
    """Per-pixel: how far is the colour we PAINTED from the nearest colour the source has
    within +/-radius px of that spot.

    Colour fidelity, isolated from coverage. It answers "is every colour we emit a colour
    that belongs there", which is what `color-fidelity` claims to measure, and it is blind
    by construction to WHERE a faithful colour landed:
      * a glyph rendered 3px over paints black onto a pixel whose source neighbourhood
        contains that same black -> ~0 error (correct: placement is not colour);
      * a red headline painted blue paints a colour the source has nowhere nearby -> large;
      * any tint shifts every painted colour off its local source palette -> large;
      * MISSING content paints the background colour, which IS in the source neighbourhood
        -> ~0 (correct: absence is a coverage defect, and other gates own it).
    Deliberately one-directional (painted -> source). The reverse (source colours absent from
    our render) is exactly the coverage question this metric must NOT answer.
    """
    import numpy as np

    best = None
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            shifted = np.roll(np.roll(source_lab, dy, axis=0), dx, axis=1)
            distance = np.linalg.norm(render_lab - shifted, axis=2)
            best = distance if best is None else np.minimum(best, distance)
    return best


def _color_metrics(source_rgb, render_rgb):
    """Colour fidelity — reported two ways, because the classic one is not about colour.

    F-colour-honesty. `similarity` is ``1 - mean(deltaE)/50`` over EVERY pixel. On text-heavy
    art that is not a colour metric at all: it degenerates into a GLYPH-COVERAGE proxy,
    because "black glyph here / white paper there" is the maximally opposite colour pair and
    swamps the mean. Measured on 067 (the only fixture that carries this fail): every hue the
    pipeline emits is essentially exact — red headline source (252,2,2) vs render (252,1,0);
    black body (0,0,0) vs (0,0,0); background (246,246,246) vs (246,246,246); jars dE 0.59;
    signature dE 1.15; median text-region dE 1.04; ZERO tint anywhere. Yet mean dE was 11.84
    (similarity 0.763 < the 0.820 gate) because the dE distribution is bimodal: 55% of text
    pixels at dE<2 but 20.2% at dE>50. By contribution 99% of the "colour" error was text
    placement (headline 21.5%, para 1 47.8%, para 2 26.1%, jars 0.6%) — a font-substitution
    artefact ALREADY counted by ssim/worst-region, re-reported under a colour name. No colour
    fix could ever clear it: fixing 067's paragraph break moved it 0.739 -> 0.763.

    `similarity` keeps its exact old definition and scale — harness_critic, repair and the
    archetype `color_similarity_min` presets are all calibrated to it, and silently rescaling
    a number they threshold would disable their gates without saying so. What changes is that
    it may no longer HARD-fail on its own: a `color-fidelity` hard fail must now be
    corroborated by `local_similarity`, which measures colour where colour is comparable (see
    `_local_color_error`). Uncorroborated = the coverage artefact 067 proves exists, and it
    is reported as a warning instead.

    Calibration of `local_similarity` is measured, not guessed — mean local dE at radius 16
    over real ad renders vs injected colour defects:
        all 16 bench-10 real renders   0.00 - 0.39   (-> local_similarity 0.992 - 1.000)
        region erased to background    0.14          coverage defect, correctly NOT colour
        067 HEAD (colour proven exact) 0.16
        20px pure translation, sparse  0.00          drift is not colour
        red headline -> pink           ~0.5          MISSED (see limitation below)
        top band red +90               1.24          fires
        red headline -> blue           2.55          fires
        blue tint +25 / +60            4.72 / 7.66   fires
        magenta block over a region    6.80          fires
    A floor of 1.0 dE (local_similarity 0.98) sits in the gap between the worst real render
    (0.39) and the weakest defect it catches (1.24).

    The radius is 16, also measured rather than assumed. It must exceed the glyph drift a
    substituted font produces (20-32px measured on the worst-window study) or a pure
    TRANSLATION starts reading as a colour error when content is sparse enough that a
    displaced glyph finds no same-coloured neighbour: on an adversarial isolated-glyph page a
    20px translation scores 1.83 at radius 8 (false fire) but 0.00 at radius 16. Real ad text
    is dense enough that radius 8 also passed all 16 fixtures; 16 is simply the setting with
    margin on both sides. Larger is not free — at radius 24 the weakest real defect drops to
    1.14 and starts crowding the floor.

    Known limitation, stated rather than hidden: this is a MEAN, so a subtle hue shift over a
    small area (the pink-headline injection: ~1% of canvas, dE ~20 locally) dilutes below the
    floor and is missed. A large or strongly off-palette colour error is caught; a small
    tasteful one is not. Catching that needs per-element colour comparison against
    design.json fills, which is a bigger build than this pass.
    """
    import numpy as np

    src, ren = _metric_rgb(source_rgb), _metric_rgb(render_rgb)
    src_lab, ren_lab = _rgb_to_lab(src), _rgb_to_lab(ren)
    delta = np.linalg.norm(src_lab - ren_lab, axis=2)
    mean = float(delta.mean())
    p95 = float(np.percentile(delta, 95))
    mae = float(np.abs(src - ren).mean())
    local_mean = float(_local_color_error(src_lab, ren_lab).mean())
    return {
        # Unchanged definition/scale — every existing consumer and threshold keeps working.
        "similarity": max(0.0, 1.0 - mean / 50.0),
        "delta_e_mean": mean,
        # Coverage-blind colour signal; the corroboration a color-fidelity HARD fail needs.
        "local_similarity": max(0.0, 1.0 - local_mean / 50.0),
        "delta_e_local_mean": local_mean,
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


def _text_block_key(layer):
    """Source-block identity of a text node, so word-splits of one line regroup.

    Emitters split a single OCR block into per-word/per-run nodes with a ``__`` suffix
    (``c_B5`` -> ``c_B5__w0``/``c_B5__w1``). The stem before ``__`` is the block the
    fragments came from. Falls back to the node's own id (its own block) so an unsplit
    node is simply a group of one, and to an object-identity key when a node carries no
    id at all — never to a shared constant, which would fuse unrelated nodes into one
    phantom string.
    """
    meta = layer.get("meta") or {}
    for key in ("source_block_id", "block_id", "split_of"):
        value = meta.get(key)
        if value:
            return str(value)
    lid = str(layer.get("id") or "")
    if not lid:
        return f"__anon_{id(layer)}"
    return lid.split("__", 1)[0] or lid


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


def _text_recall(source_ocr, render_ocr, source_gray=None, render_gray=None,
                 design=None, run_dir=None):
    """Share of source lines an OCR of the RENDERED IMAGE reads back. Kept deliberately.

    This is the render-OCR round-trip, and it stays exactly as it was because it has one
    legitimate caller: `figma_verify`, which OCRs the REAL FIGMA EXPORT. There, re-reading
    the rendered pixels is the whole point — the export IS the deliverable, so text that
    cannot be read back really did go missing in Figma.

    `compare()` no longer uses it. Against our own PREVIEW the same round-trip is a
    consistency metric, not a correctness one, and it punished correct work (131) — see
    `_text_recall_detail`, which is what compare() scores instead. Same question, two
    different right answers, depending on whether you are looking at the real export or at
    our own proxy of it.
    """
    kept = [l for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5
            and len(_norm(l.get("text", ""))) >= 3]
    if not kept:
        return 1.0
    ren_blob = " ".join(_norm(l["text"]) for l in (render_ocr or {}).get("lines", []))
    kept_blob = " ".join(_norm(t) for t in ((design or {}).get("kept_in_photo") or []))
    baked_leaves = _baked_line_leaves(design, source_gray) if kept_blob else []
    asset_cache = {}
    found = 0
    excluded = 0
    for line in kept:
        norm = _norm(line["text"])
        if norm in ren_blob:
            found += 1
            continue
        # Same pixel-verbatim fallback as before: OCR is not deterministic even on IDENTICAL
        # pixels (021 measured recall 0.6 on a render with ssim 1.0 vs source).
        if source_gray is not None and render_gray is not None:
            clipped = _clip_box(line.get("box") or {}, *source_gray.shape[1::-1])
            if clipped is not None:
                try:
                    import numpy as _np
                    x0, y0, x1, y1 = clipped
                    delta = _np.abs(source_gray[y0:y1, x0:x1].astype(_np.float32)
                                    - render_gray[y0:y1, x0:x1].astype(_np.float32))
                    if float(delta.mean()) <= 2.0:
                        found += 1
                        continue
                except Exception:
                    pass
        if kept_blob and norm and norm in kept_blob:
            try:
                if _line_baked_in_asset(line.get("box") or {}, source_gray,
                                        baked_leaves, run_dir, asset_cache):
                    excluded += 1
            except Exception:
                pass
    denominator = len(kept) - excluded
    return 1.0 if denominator <= 0 else found / denominator


def _clip_box(box, width, height):
    try:
        x0 = max(0, int(box.get("x", 0)));  y0 = max(0, int(box.get("y", 0)))
        x1 = min(width, int(box.get("x", 0) + box.get("w", 0)))
        y1 = min(height, int(box.get("y", 0) + box.get("h", 0)))
    except (TypeError, ValueError):
        return None
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1, y1


def _baked_line_leaves(design, source_gray):
    """Raster leaves eligible to legitimately carry baked (kept-in-photo) text.

    Product/photo cutouts and slices only — never the background plate and never a
    near-full-canvas raster (a lazy whole-page rasterization must not launder its text
    through this exemption; the native-leaf/unexplained-raster gates own that failure,
    and this list refuses to help it)."""
    if design is None or source_gray is None:
        return []
    height, width = source_gray.shape[:2]
    canvas_area = float(width * height)
    leaves = []
    for leaf, box in _iter_leaf_layers_abs((design or {}).get("layers") or []):
        if leaf.get("type") != "image" or not leaf.get("src"):
            continue
        if (leaf.get("meta") or {}).get("role") == "background":
            continue
        area = max(0.0, float(box.get("w", 0))) * max(0.0, float(box.get("h", 0)))
        if canvas_area and area / canvas_area >= 0.90:
            continue
        leaves.append((leaf, box))
    return leaves


def _line_baked_in_asset(line_box, source_gray, leaves, run_dir, asset_cache):
    """True when a source text line's pixels are verbatim present in an emitted asset.

    The pixel-identity idea from the render fallback, applied to the OWNING asset: crop
    the layer's real asset file at the line's coordinates and require the source pixels
    to be there (>=50% opaque coverage, small mean delta — resampling tolerance only).
    Un-gameable: it reads the actual pixels the export will show, so a reconstruction
    that dropped or repainted the label text cannot pass it."""
    import numpy as np
    from PIL import Image

    height, width = source_gray.shape[:2]
    clipped = _clip_box(line_box, width, height)
    if clipped is None:
        return False
    x0, y0, x1, y1 = clipped
    line_area = float((x1 - x0) * (y1 - y0))
    for leaf, box in leaves:
        bx, by = float(box.get("x", 0)), float(box.get("y", 0))
        bw, bh = float(box.get("w", 0)), float(box.get("h", 0))
        if bw < 1 or bh < 1:
            continue
        ix0, iy0 = max(x0, bx), max(y0, by)
        ix1, iy1 = min(x1, bx + bw), min(y1, by + bh)
        if ix1 - ix0 <= 0 or iy1 - iy0 <= 0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) < 0.6 * line_area:
            continue
        src = str(leaf.get("src"))
        cached = asset_cache.get(src)
        if cached is None:
            path = _resolve_path(src, run_dir)
            if not path or not os.path.exists(path):
                asset_cache[src] = False
                continue
            try:
                image = Image.open(path).convert("RGBA")
                target = (max(1, int(round(bw))), max(1, int(round(bh))))
                if image.size != target:
                    image = image.resize(target, Image.Resampling.LANCZOS)
                arr = np.asarray(image, dtype=np.float32)
                gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
                cached = (gray, arr[..., 3])
            except Exception:
                cached = False
            asset_cache[src] = cached
        if cached is False:
            continue
        gray, alpha = cached
        ah, aw = gray.shape[:2]
        rx0 = max(0, int(round(ix0 - bx))); ry0 = max(0, int(round(iy0 - by)))
        rx1 = min(aw, int(round(ix1 - bx))); ry1 = min(ah, int(round(iy1 - by)))
        if rx1 - rx0 < 3 or ry1 - ry0 < 3:
            continue
        sx0 = int(round(ix0)); sy0 = int(round(iy0))
        crop_gray = gray[ry0:ry1, rx0:rx1]
        crop_alpha = alpha[ry0:ry1, rx0:rx1]
        src_crop = source_gray[sy0:sy0 + (ry1 - ry0), sx0:sx0 + (rx1 - rx0)]
        if src_crop.shape != crop_gray.shape:
            continue
        opaque = crop_alpha >= 128
        if float(opaque.mean()) < 0.5:
            continue
        delta = np.abs(src_crop.astype(np.float32) - crop_gray)[opaque]
        if delta.size and float(delta.mean()) <= 8.0:
            return True
    return False


def _confusable_fold(text):
    """`_confusable_key` from src.ocr — canonical form under known OCR glyph confusion.

    Imported lazily (and reused, never re-implemented) so the SAME folding the OCR
    arbitration uses also governs how QA compares strings. Falls back to plain `_norm` if
    src.ocr cannot be imported, which only makes the comparison stricter, never looser.
    """
    global _CONFUSABLE_FOLD_FN
    if _CONFUSABLE_FOLD_FN is None:
        fn = None
        for module in ("src.ocr", "ocr"):
            try:
                fn = getattr(__import__(module, fromlist=["_confusable_key"]),
                             "_confusable_key")
                break
            except Exception:
                continue
        _CONFUSABLE_FOLD_FN = fn or (lambda value: _norm(value))
    return _CONFUSABLE_FOLD_FN(text)


_CONFUSABLE_FOLD_FN = None


def _delivered_text_blob(design, fold=True):
    """The copy we actually DELIVER, in any form, block-joined.

    Counts TEXT nodes AND text carried by raster layers (fallback slices, wordmark/lockup
    art, foreground_raster bakes — the same layers `_text_editability` accounts for). A
    rasterized line IS delivered: the copy is present in the output, just not editably.
    Keeping it here is what keeps `text_recall` (is the copy present at all) a different
    question from `editable_text_recall` (is it editable) — and therefore what keeps their
    product, `true_text_coverage`, meaningful rather than a squared restatement of one
    number. Rasterizing text still costs the run: it lowers editable_text_recall /
    native_text_ratio, which is where that loss belongs.

    Same block-regrouping as `_text_editability` (one source line is often emitted as
    several sibling word nodes): fragments of one block concatenate, separate blocks stay
    space-separated so unrelated nodes can never fuse into a phantom match.
    """
    key = _confusable_fold if fold else _norm
    blocks = {}
    raster = []
    for layer in _flatten_layers((design or {}).get("layers") or []):
        if layer.get("type") == "text":
            value = key(layer.get("text"))
            if value:
                blocks.setdefault(_text_block_key(layer), []).append(value)
            continue
        meta = layer.get("meta") or {}
        is_raster_text = layer.get("type") == "image" and (
            _fallback_kind(meta) is not None
            or meta.get("wordmark") or meta.get("platform_lockup")
            or meta.get("layer_disposition") == "foreground_raster"
        )
        if not is_raster_text:
            continue
        value = key(layer.get("text") or meta.get("source_text"))
        if value:
            raster.append(value)
    return " ".join(["".join(parts) for parts in blocks.values()] + raster)


def _ink_ownership_ledger(source_ocr, design, run_dir, source_gray):
    """Per-line ink ownership: every source text line must end in EXACTLY ONE state.

    The invariant behind our most persistent defect family (ghost doubles, baked strikes
    under native text, dropped lines): a line's ink is owned by the plate OR a native
    node — never both, never neither. We fixed 13+ INSTANCES of violations across
    benches 5-13 (025 'Blocks everything' ghost, 002's baked euro-strike under native
    text, 013's badge double, 066's outlined doubles...) without ever enforcing the
    invariant, so each new code path could violate it again. This ledger enforces it.

    States per line:
      native-clean   node exists, plate under the box no longer holds the source ink  OK
      baked          no node; plate (or an owning asset) still holds the ink          OK
      DOUBLE         node exists AND the plate still holds the source ink verbatim    HARD
      dropped        no node and no surface holds the ink                             report
    The DOUBLE test is deliberately conservative: the plate crop must match the source
    crop nearly verbatim (mean |diff| <= 4.0 over the box) — partially-cleaned smears are
    reconstruct's residue detector's job, not this gate's, and a loose threshold here
    would hard-fail every textured carrier.
    """
    import numpy as np
    from PIL import Image

    out = {"lines": [], "doubles": 0, "dropped": 0, "native_clean": 0, "baked": 0,
           "basis": "plate-vs-source verbatim ink check"}
    if design is None or source_gray is None or not run_dir:
        out["basis"] = "unavailable (missing design/plate/source)"
        return out
    plate_path = os.path.join(run_dir, "background_clean.png")
    if not os.path.isfile(plate_path):
        out["basis"] = "unavailable (no background_clean.png)"
        return out
    try:
        plate = np.asarray(Image.open(plate_path).convert("L"), dtype=np.float32)
    except Exception:
        out["basis"] = "unavailable (plate unreadable)"
        return out
    if plate.shape != source_gray.shape:
        try:
            plate = np.asarray(
                Image.open(plate_path).convert("L").resize(source_gray.shape[1::-1]),
                dtype=np.float32)
        except Exception:
            out["basis"] = "unavailable (plate size mismatch)"
            return out

    baked_leaves = _baked_line_leaves(design, source_gray)
    asset_cache = {}
    # Text-only matching aliases: a ribbon fragment reading 'BLACK' would match the
    # HEADLINE node's 'Black Friday' and read as a double (measured on 088). A line is
    # only "native" if a text node both CONTAINS its folded text and OVERLAPS its box.
    text_nodes = []
    for leaf, abs_box in _iter_leaf_layers_abs((design or {}).get("layers") or []):
        if leaf.get("type") == "text" and (leaf.get("text") or "").strip():
            text_nodes.append((_confusable_fold(leaf.get("text", "")), abs_box))

    def _overlaps(a, b):
        ax0, ay0 = float(a.get("x", 0)), float(a.get("y", 0))
        ax1, ay1 = ax0 + float(a.get("w", 0)), ay0 + float(a.get("h", 0))
        bx0, by0 = float(b.get("x", 0)), float(b.get("y", 0))
        bx1, by1 = bx0 + float(b.get("w", 0)), by0 + float(b.get("h", 0))
        return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1

    kept = [l for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5
            and len(_norm(l.get("text", ""))) >= 3]
    for line in kept:
        folded = _confusable_fold(line.get("text", ""))
        lbox = line.get("box") or {}
        native = bool(folded) and any(
            folded in node_text and _overlaps(lbox, node_box)
            for node_text, node_box in text_nodes if node_text)
        clipped = _clip_box(line.get("box") or {}, *source_gray.shape[1::-1])
        plate_holds = None
        if clipped is not None:
            x0, y0, x1, y1 = clipped
            if x1 > x0 and y1 > y0:
                delta = float(np.abs(source_gray[y0:y1, x0:x1]
                                     - plate[y0:y1, x0:x1]).mean())
                plate_holds = delta <= 4.0
        asset_holds = False
        if not plate_holds:
            try:
                asset_holds = _line_baked_in_asset(line.get("box") or {}, source_gray,
                                                   baked_leaves, run_dir, asset_cache)
            except Exception:
                asset_holds = False
        if native and plate_holds:
            state = "DOUBLE"
            out["doubles"] += 1
        elif native:
            state = "native-clean"
            out["native_clean"] += 1
        elif plate_holds or asset_holds:
            state = "baked"
            out["baked"] += 1
        else:
            state = "dropped"
            out["dropped"] += 1
        out["lines"].append({"text": str(line.get("text"))[:48], "state": state,
                             "plate_holds": plate_holds, "asset_holds": asset_holds})
    return out


def _text_recall_detail(source_ocr, render_ocr, source_gray=None, render_gray=None,
                        design=None, run_dir=None):
    """Share of source text lines we actually DELIVER, scored against source truth.

    F-recall-honesty. This metric used to re-OCR our own render and compare that reading to
    the source's reading, which made it a CONSISTENCY metric wearing a correctness metric's
    name: it rewarded being wrong the same way twice and PUNISHED being right. Measured on
    ad 131 at bench-10 — the deliverable is character-perfect:

        source truth (ocr.json)   'BUY 2, GET 1 FREE + FREE SHIPPING $100+'
        delivered  (design.json)  'BUY 2, GET 1 FREE + FREE SHIPPING $100+'   <- correct
        render re-OCR             'BUY2. GETIFRE: + FREE SHPPINC $100+'       <- engine junk
        source truth              'BLACK FRIDAY SALE'
        delivered                 'BLACK FRIDAY SALE'                         <- correct
        render re-OCR             'BLACKERIDAYSALE'

    ...and the old metric scored that 0.556, the WORST in the run, purely because doctr
    re-reads our own correct glyphs as 'SHPPINC'/'TODAV'. Every text fix the pipeline landed
    (the deterministic OCR fix here, the VLM's TODAV->TODAY earlier) showed up as a
    regression. A metric that moves DOWN when the work gets RIGHT is worse than no metric.

    So recall now asks the only question that matters for a deliverable: **is each source
    line present in what we ship?** A line counts as delivered when either

      1. its text is in `design.json`'s TEXT nodes — the thing we actually hand to Figma,
         compared under `_confusable_key` folding so a G/C engine quirk in the SOURCE
         reading (131's raw 'SHIPPINC') cancels against our corrected string instead of
         scoring as a miss; or
      2. its pixels survive verbatim in the render — text legitimately baked into a kept
         photo IS present, and pixel identity cannot be faked by a wrong reconstruction.

    Render-OCR is NOT part of this number any more. It moved to `render_text_legibility`
    (see `_render_text_legibility`), which is where a "does the preview draw it legibly"
    question belongs — reported, never a hard gate, because the preview is our own proxy and
    Figma, not the preview, draws the shipped file.
    """
    kept = [l for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5
            and len(_norm(l.get("text", ""))) >= 3]
    if not kept:
        return {"recall": 1.0, "found": 0, "lines_total": 0,
                "baked_excluded": 0, "baked_excluded_lines": [],
                "delivered": 0, "baked_verbatim": 0, "missing_lines": [],
                "basis": "no-source-text"}
    # No design means no deliverable to read, so "did we deliver it" is unanswerable and
    # must not be answered with a confident 0.0. figma_verify is exactly this caller: it
    # compares a REAL FIGMA EXPORT with no design.json in hand, and there the rendered
    # pixels genuinely ARE the deliverable, so OCRing them is the right measure. Score what
    # we were actually given, and record which question got answered.
    if design is None:
        return {"recall": _text_recall(source_ocr, render_ocr, source_gray, render_gray,
                                       design=design, run_dir=run_dir),
                "found": None, "lines_total": len(kept), "baked_excluded": 0,
                "baked_excluded_lines": [], "delivered": None, "baked_verbatim": None,
                "missing_lines": [], "basis": "render-ocr (no design supplied)"}
    delivered_blob = _delivered_text_blob(design)
    kept_blob = " ".join(_norm(t) for t in ((design or {}).get("kept_in_photo") or []))
    baked_leaves = _baked_line_leaves(design, source_gray) if kept_blob else []
    asset_cache = {}
    found = delivered = baked_verbatim = 0
    baked_excluded = []
    missing = []
    for line in kept:
        norm = _norm(line["text"])
        # 1. Present in the DELIVERABLE (design.json text nodes), confusable-folded.
        folded = _confusable_fold(line["text"])
        if folded and delivered_blob and folded in delivered_blob:
            found += 1
            delivered += 1
            continue
        # 2. Baked-verbatim fallback: the pixel region under this line's box survives
        # essentially unchanged in the render, so the text is literally present. (Also the
        # old guard against OCR non-determinism on IDENTICAL pixels — 021 measured recall
        # 0.6 on a render with ssim 1.0 vs source.) Un-gameable: pixel identity cannot be
        # faked by a wrong reconstruction.
        if source_gray is not None and render_gray is not None:
            clipped = _clip_box(line.get("box") or {}, *source_gray.shape[1::-1])
            if clipped is not None:
                try:
                    import numpy as _np
                    x0, y0, x1, y1 = clipped
                    delta = _np.abs(source_gray[y0:y1, x0:x1].astype(_np.float32)
                                    - render_gray[y0:y1, x0:x1].astype(_np.float32))
                    if float(delta.mean()) <= 2.0:
                        found += 1
                        baked_verbatim += 1
                        continue
                except Exception:
                    pass
        # Product-printed-text fairness (135): a line that merge deliberately kept in the
        # photo (design.kept_in_photo) AND whose pixels are verbatim present in the owning
        # product/photo ASSET is correctly baked-by-design — the pipeline never promised to
        # make it editable. Exclude it from the recall DENOMINATOR so the metric reflects
        # editable-ad text only. Both conditions are evidence-based: the merge verdict comes
        # from the source image, and the pixels are read from the real asset.
        if kept_blob and norm and norm in kept_blob:
            try:
                if _line_baked_in_asset(line.get("box") or {}, source_gray,
                                        baked_leaves, run_dir, asset_cache):
                    baked_excluded.append(str(line.get("text"))[:60])
                    continue
            except Exception:
                pass
        missing.append(str(line.get("text"))[:60])
    denominator = len(kept) - len(baked_excluded)
    recall = 1.0 if denominator <= 0 else found / denominator
    return {"recall": recall, "found": found, "lines_total": len(kept),
            "baked_excluded": len(baked_excluded),
            "baked_excluded_lines": baked_excluded,
            # Auditable split of the numerator: how many lines we ship as real text vs how
            # many are only "present" as preserved pixels. A run whose recall leans on
            # baked_verbatim is NOT delivering editable text, and native_text_ratio /
            # editable_text_recall are the metrics that say so.
            "delivered": delivered, "baked_verbatim": baked_verbatim,
            # Never silent about what we dropped.
            "missing_lines": missing,
            # Which question this number answers — never leave that to inference.
            "basis": "delivered-vs-source-truth"}


def _render_text_legibility(render_ocr, design):
    """Does our PREVIEW visibly draw the text we delivered? Reported, never a hard gate.

    This is what the old `text_recall` was actually measuring, restored to an honest scope
    and an honest name. It compares render-OCR against the DELIVERED strings (not against
    source truth) under `_confusable_key` folding, so the engine's own glyph confusions
    (131: doctr reads our correct 'FREE SHIPPING' as 'FREE SHPPINC') cancel on both sides
    instead of manufacturing a phantom regression.

    It is deliberately NOT gated:
      * the preview is OUR proxy renderer, not the deliverable — Figma draws the shipped
        file, and a font agent measured 41 of 99 delivered text nodes drawing a DIFFERENT
        face in preview than Figma will resolve;
      * OCR on stylised display type is noisy enough that a low score is a hint to look,
        not evidence of a defect.
    A low value with `text_recall` at 1.0 means "we shipped the right strings; the preview
    may be drawing them unreadably" — worth a look, not a rejection.
    """
    if not render_ocr or not design:
        return None
    delivered_lines = [layer.get("text") for layer in
                       _flatten_layers((design or {}).get("layers") or [])
                       if layer.get("type") == "text" and _norm(layer.get("text"))]
    if not delivered_lines:
        return None
    render_blob = " ".join(_confusable_fold(l.get("text"))
                           for l in render_ocr.get("lines", []) if l.get("conf", 1) >= 0.5)
    legible = 0
    illegible = []
    for text in delivered_lines:
        folded = _confusable_fold(text)
        if folded and render_blob and folded in render_blob:
            legible += 1
        else:
            illegible.append(str(text)[:60])
    return {"legible": legible, "delivered_lines": len(delivered_lines),
            "ratio": round(legible / len(delivered_lines), 4),
            "not_read_back": illegible}


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
    # 066: comparison column chips / checklist rasters record the photo+card+icon
    # ids they folded in. Those detections survived inside the chip, not as drops.
    values.extend(meta.get("merged_from") or [])
    values.extend(meta.get("absorbed_list_icons") or [])
    if meta.get("absorbed_into"):
        values.append(meta.get("absorbed_into"))
    if meta.get("baked_owner_id"):
        values.append(meta.get("baked_owner_id"))

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


_FONT_WEIGHT_WORDS = (
    ("thin", 100), ("extralight", 200), ("ultralight", 200), ("demilight", 350),
    ("semilight", 350), ("light", 300), ("semibold", 600), ("demibold", 600),
    ("extrabold", 800), ("ultrabold", 800), ("black", 900), ("heavy", 900),
    ("medium", 500), ("bold", 700), ("regular", 400), ("normal", 400), ("book", 400),
)


def _subfamily_weight(subfamily):
    """Weight class a font FILE advertises in its own subfamily name."""
    name = str(subfamily or "").casefold().replace(" ", "").replace("-", "")
    for word, value in _FONT_WEIGHT_WORDS:
        if word in name:
            return value
    return 400


def _is_italic_name(name):
    lowered = str(name or "").casefold()
    return "italic" in lowered or "oblique" in lowered


def _font_consistency_audit(design):
    """Does each text node's DECLARED font match the file the preview actually drew?

    F-deliverable-consistency. Two different things resolve a font, and QA never compared
    them:
      * FIGMA (our deliverable) resolves the declared ``fontFamily`` + ``fontStyle`` NAME;
      * our preview resolves a FILE from ``fontCandidates`` and draws that.
    When they disagree, the preview's pixels stop being evidence about the deliverable — and
    a whole defect class becomes invisible to SSIM BY CONSTRUCTION, because SSIM only ever
    sees the preview. Ad 013 is the proof: it declared ``fontStyle`` Bold while carrying an
    ITALIC file, so the preview resolved the FILE and drew italic (matching the source, SSIM
    happy) while Figma resolves the STYLE NAME and would ship it UPRIGHT. No pixel metric
    could ever have caught that. This check is cheap, needs no render, and reads both sides.

    Measured at bench-10 (103 text nodes): 33 draw a different FAMILY than declared —
    `_relabel_google_families` rewrites a matched local face to a Figma-loadable Google name
    (declared "Open Sans" -> draws Candara on 088/101/107; declared "Barlow Condensed" ->
    draws Bahnschrift on 091; declared "Inter" -> draws Arial on 094). Those nodes' preview
    pixels are NOT what Figma will draw, which is exactly why preview SSIM overstates
    deliverable fidelity.

    Severity is evidence-led, not uniform:
      * family substitution -> WARNING. It is endemic (33/103) and deliberate; hard-failing
        it would fire on 10 of 16 fixtures and be exactly the useless always-on gate this
        pass exists to remove. It is reported per node so it can never be silent again.
      * drawn-italic-while-declared-upright -> HARD. This is 013's class and it ships wrong.
        A missing font can never CAUSE it (every fallback face is upright), so it cannot be
        an environment artefact, and it fires 0/103 at bench-10 — a gate that is silent today
        and speaks only when the defect returns.
      * declared-italic-but-drawn-upright -> WARNING, not hard: that IS what a missing italic
        file on this machine looks like, and Figma would still ship italic correctly.
      * weight -> WARNING, and only when the axis was NOT driven. A variable face
        (``Inter[opsz,wght].ttf``) advertises subfamily "Regular" while `_text_font` dials
        its wght axis to the declared weight, so reading the file's name alone reports a
        phantom mismatch (measured: it falsely flagged 009 w700 and 013 w800). Mirrors
        `_text_font`'s own ``family_resolved`` rule. What survives is real: 135's
        ``c_B1__w1`` declares Poppins/Regular/w400 but its Regular candidate has no file, so
        the preview draws Poppins **ExtraBold** — independently corroborated by 135's worst
        window, where the render draws "OP" bold over a light source.
    """
    try:
        from src import render_preview
    except Exception:
        try:
            import render_preview  # type: ignore
        except Exception:
            return None
    nodes = []
    for layer in _flatten_layers((design or {}).get("layers") or []):
        if layer.get("type") != "text":
            continue
        style = layer.get("style") or {}
        try:
            font = render_preview._text_font(style, style.get("fontSize") or 12)
            path = getattr(font, "path", None)
            file_family, file_style = font.font.family, font.font.style
        except Exception:
            continue
        if not file_family:
            continue
        declared_family = str(style.get("fontFamily") or "")
        declared_style = str(style.get("fontStyle") or "")
        try:
            declared_weight = float(style.get("fontWeight") or 400)
        except (TypeError, ValueError):
            declared_weight = 400.0
        # Was the wght axis driven for this node? Same rule _text_font itself uses.
        axis_driven = False
        for candidate in (style.get("fontCandidates") or []):
            if (isinstance(candidate, dict) and candidate.get("path") and path
                    and os.path.normcase(str(candidate["path"])) == os.path.normcase(str(path))):
                axis_driven = bool(candidate.get("family_resolved"))
                break
        issues = []
        if declared_family.strip().casefold() != str(file_family).strip().casefold():
            issues.append("family")
        declared_italic, file_italic = _is_italic_name(declared_style), _is_italic_name(file_style)
        if file_italic and not declared_italic:
            issues.append("italic-drawn-not-declared")   # 013's class — hard
        elif declared_italic and not file_italic:
            issues.append("italic-declared-not-drawn")
        if not axis_driven and abs(_subfamily_weight(file_style) - declared_weight) >= 200:
            issues.append("weight")
        if issues:
            nodes.append({
                "id": layer.get("id"), "issues": issues,
                "declared": {"family": declared_family, "style": declared_style,
                             "weight": declared_weight},
                "drawn": {"family": str(file_family), "style": str(file_style),
                          "file": os.path.basename(str(path or "")),
                          "weight_axis_driven": axis_driven},
            })
    total = sum(1 for layer in _flatten_layers((design or {}).get("layers") or [])
                if layer.get("type") == "text")
    if not total:
        return None
    consistent = total - len(nodes)
    return {
        "text_nodes": total,
        "consistent_nodes": consistent,
        # How much of the preview is actually evidence about the deliverable. Read this
        # BEFORE trusting any preview-derived pixel number.
        "preview_matches_declaration_ratio": round(consistent / total, 4),
        "mismatched_nodes": nodes,
        "family_substituted": sum(1 for n in nodes if "family" in n["issues"]),
        "italic_drawn_not_declared": sum(
            1 for n in nodes if "italic-drawn-not-declared" in n["issues"]),
        "italic_declared_not_drawn": sum(
            1 for n in nodes if "italic-declared-not-drawn" in n["issues"]),
        "weight_mismatched": sum(1 for n in nodes if "weight" in n["issues"]),
    }


def _text_editability(source_ocr, design, layers, source_gray=None):
    """Honest text-editability accounting (F4).

    ``editable_text_recall`` = detected source text lines that ship as a CORRECT editable
    TEXT node / all detected source text lines that are NOT baked scene-text-by-design.

    Critically, a raster slice, a wordmark/lockup image, or a ``foreground_raster`` image
    that carries a text line counts as **non-editable** and LOWERS the recall — rasterizing
    failed overlay text is a quality loss, never a way to remove that text from the metric's
    denominator. The only text excluded from the denominator is ``kept_in_photo`` scene text
    (legitimately not-editable-by-design), which is tracked separately.

    ``editable_text_fraction`` is the user-facing COVERAGE number — "how much of THIS ad can
    I actually edit?" — over ALL readable source text (kept_in_photo INCLUDED in the
    denominator). When ``source_gray`` is supplied it is weighted by measured text INK per
    line, not by line count, because a giant editable headline plus a dozen tiny baked
    package lines is highly batch-editable even though only a fraction of the *lines* are
    editable (013: 3/15 lines = 0.20 by count, but 0.63 by ink — the headline dominates).
    Ink weighting matches the user's own framing ("editable text ink / total source text
    ink") and the batch use case ("we can simply modify the text"). Falls back to line-count
    weighting when the source image is unavailable.

    Returns a dict of counts/ratios (or ``None`` when there is nothing to score):
      editable_text_recall, editable_text_fraction, all_source_text_baked, text_lines_total,
      kept_in_photo_lines, rasterized_text_count, rasterized_text_ratio, editable_text_correct.
    """
    if not source_ocr or not design:
        return None
    editable_blocks: "dict[str, list]" = {}
    raster_line_ids = set()
    raster_texts = []
    for layer in _flatten_layers(layers):
        meta = layer.get("meta") or {}
        ltype = layer.get("type")
        if ltype == "text":
            # A TEXT node is editable in Figma regardless of wordmark/lockup styling.
            norm = _norm(layer.get("text"))
            if norm:
                # Group by SOURCE BLOCK, not per node: one OCR line is frequently emitted
                # as several sibling word nodes (002 "€63 → €49" -> c_B5__w0 "€63 →" +
                # c_B5__w1 "€49"; 066 "Smudges on upper lid" -> c_B5__w0/w1/w2). Both nodes
                # ARE editable, but _norm strips the separators, so the source line
                # ("6349") could never be a substring of a space-joined blob ("63 49") and
                # the line was scored as MISSING — a false `missing-editable-text` /
                # `low-text-recall` on work we actually shipped correctly. Re-joining the
                # fragments of the same block (and only that block) restores the honest
                # numerator without letting unrelated nodes concatenate into phantom
                # matches.
                editable_blocks.setdefault(_text_block_key(layer), []).append(norm)
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

    # Per-block fragments concatenate (a split line reads as one string); separate blocks
    # stay space-separated so they can never fuse into a match that was never rendered.
    editable_blob = " ".join("".join(parts) for parts in editable_blocks.values())
    raster_blob = " ".join(raster_texts)
    kept_blob = " ".join(_norm(text) for text in (design.get("kept_in_photo") or []))

    # Ink weight per line for editable_text_fraction. Measured text-stroke pixels (robust to
    # fg/bg polarity via distance from the box's local median) when the source image is
    # available; a value of 1.0 per line (line-count) otherwise. This is prominence, not
    # a count, so an editable headline outweighs a dozen tiny baked package lines.
    def _line_ink(line):
        if source_gray is None:
            return 1.0
        box = line.get("box") or {}
        try:
            x, y = int(box.get("x", 0)), int(box.get("y", 0))
            w, h = int(box.get("w", 0)), int(box.get("h", 0))
        except (TypeError, ValueError):
            return 1.0
        H, W = source_gray.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = source_gray[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        import numpy as _np
        med = float(_np.median(patch))
        # Text ink = pixels well away from the local plate colour; area floor of 1px so a
        # detected-but-faint line never contributes literally zero weight to the denominator.
        return float(max(1.0, _np.count_nonzero(_np.abs(patch - med) > 40)))

    total = kept = correct = rasterized = 0
    total_ink = correct_ink = 0.0
    for line in source_ocr.get("lines", []):
        if line.get("conf", 1) < 0.5:
            continue
        norm = _norm(line.get("text"))
        if len(norm) < 3:
            continue
        line_id = str(line.get("id") or "")
        total += 1
        ink = _line_ink(line)
        total_ink += ink
        # By-design baked scene text is not something the pipeline promises to make editable.
        if kept_blob and norm and norm in kept_blob:
            kept += 1
            continue
        if norm and norm in editable_blob:
            correct += 1
            correct_ink += ink
        elif line_id in raster_line_ids or (raster_blob and norm and norm in raster_blob):
            rasterized += 1
        # else: the line is simply missing — not editable and not even rasterized. It counts
        # against recall (stays in the denominator, never in the numerator).
    denom = total - kept
    # F-vacuous (021): when EVERY readable source line is by-design baked scene text
    # (denom == 0), there is nothing the pipeline was SUPPOSED to make editable. The old
    # convention returned 1.0 here — a 0/0 pass — so "we baked the whole ad into one photo"
    # scored identically to "we perfectly decompiled everything". 021 (a UGC photo of a
    # laptop with physical BUY-TWO / FOR-FREE sticky notes) read editable_text_recall /
    # native_text_ratio 1.0 / CLEAN while delivering ZERO editable text nodes. The honest
    # value is *undefined*: recall is None (not applicable), reported distinctly and gated
    # separately (no-editable-content in the structural audit), never a success score. The
    # precedent for closing a vacuous 0/0 pass is benchmark.py:contract_verdict (it vetoed
    # the same vacuous native_text_ratio on bake hard-fails); this closes it at the source.
    all_source_text_baked = denom <= 0
    recall = None if all_source_text_baked else correct / denom
    rasterized_ratio = (rasterized / denom) if denom > 0 else 0.0
    # editable_text_fraction answers the user's real question — "how much of THIS ad can I
    # actually edit?" — over ALL readable source text, kept_in_photo INCLUDED in the
    # denominator (unlike recall, which excludes by-design-baked lines). Weighted by measured
    # text ink (see _line_ink) so an editable headline outweighs many tiny baked package
    # lines: a fully-baked photo (021) reads 0.0 even though recall is undefined; a real
    # decompile whose dominant copy is editable (013) reads ~0.63. It can never go vacuous:
    # when there is genuinely no source text at all (total == 0) it is None, not a false 1.0.
    editable_fraction = None if total_ink <= 0 else correct_ink / total_ink
    recall_rounded = None if recall is None else round(recall, 4)
    return {
        "editable_text_recall": recall_rounded,
        # native_text_ratio (Codia contract, docs/CODIA-PARITY-SPEC.md §2): native editable
        # TEXT lines / all readable OCR lines that are NOT by-design baked scene text. Slices,
        # wordmark/lockup rasters, foreground_raster bakes, AND simply-missing lines all count
        # against it (they stay in the denominator, never in the numerator). This is the QA
        # objective — "every string is native TEXT" — surfaced under its contract name. It is
        # numerically the same fraction as editable_text_recall (None when 0/0-vacuous); the
        # alias makes the contract dimension nameable by codia_parity/qa_reward/benchmark.
        "native_text_ratio": recall_rounded,
        # Coverage of ALL source text (kept included in denominator). See note above.
        "editable_text_fraction": None if editable_fraction is None else round(editable_fraction, 4),
        # True iff every readable source line was baked-by-design (denom == 0): the signal
        # that separates "nothing to decompile" from "decompiled perfectly".
        "all_source_text_baked": bool(all_source_text_baked and total > 0),
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


def _asset_paints(src, run_dir, min_alpha_px=24):
    """True when a layer's raster asset actually paints visible pixels.

    QA-side mirror of reconstruct._asset_has_content (the removal-ledger accounting this
    module consumes): a blank/near-empty PNG re-renders nothing, so an owner shipping one
    does NOT count as re-rendering its removal region. Evidence-based — reads the real
    file on disk, so it cannot be satisfied by metadata alone."""
    import numpy as np
    from PIL import Image

    if not src or str(src).startswith("data:"):
        return False
    path = _resolve_path(src, run_dir)
    if not path or not os.path.exists(path):
        return False
    try:
        image = Image.open(path)
    except Exception:
        return False
    if "A" in image.getbands():
        alpha = np.asarray(image.split()[-1])
        return int(np.count_nonzero(alpha > 16)) >= int(min_alpha_px)
    return image.width * image.height >= int(min_alpha_px)


def _leaf_claim_keys(leaf):
    """Identity keys under which an emitted leaf can claim a removal-ledger owner.

    build_design_json splits candidates into derived leaves (``c_B10`` → ``c_B10__w0``,
    ``c_E003`` → ``c_E003__hostbg``), so the base id before ``__`` is the lineage key.
    Element ids are additionally normalized (c_E007/E007/e7 → E7)."""
    keys = set()
    for value in (leaf.get("id"), (leaf.get("meta") or {}).get("source_id")):
        if not value:
            continue
        base = str(value).split("__")[0]
        keys.add(base)
        normalized = _normalized_element_id(base)
        if normalized:
            keys.add(normalized)
    return keys


def _leaf_paints(leaf, run_dir):
    """True when an emitted design leaf visibly re-renders REAL pixels."""
    ltype = leaf.get("type")
    if ltype == "text":
        return bool(_norm(leaf.get("text")))
    if ltype == "shape":
        if leaf.get("fill") or leaf.get("stroke") or leaf.get("path") \
                or leaf.get("paths") or leaf.get("svg"):
            return True
        return _asset_paints(leaf.get("src"), run_dir)
    if ltype in ("image", "icon"):
        return _asset_paints(leaf.get("src"), run_dir)
    return False


def _removal_claim_mask(run_dir, design, size):
    """Boolean mask of removal pixels whose ledger owner re-renders with real pixels.

    Consumes reconstruct's removal-ledger accounting (removal_ownership.png +
    reconstruction.json's removal_owner_index) and independently verifies each owner
    against the EMITTED design: the owner must ship a leaf whose lineage matches the
    candidate id AND that leaf must actually paint (native text, a real shape, or a
    non-empty raster asset on disk). Altered plate pixels under such an owner are
    re-rendered content (e.g. 066's two eye photos shipped as swappable cutouts), not
    destruction. Returns (mask_or_None, accounting_dict)."""
    import numpy as np
    from PIL import Image

    accounting = {"available": False, "owners_total": 0,
                  "owners_claimed": [], "owners_unclaimed": []}
    ownership_path = os.path.join(run_dir or "", "removal_ownership.png")
    reconstruction = _load_reconstruction(run_dir) if run_dir else {}
    owner_index = reconstruction.get("removal_owner_index") or {}
    if not os.path.exists(ownership_path) or not owner_index:
        return None, accounting
    # Index emitted leaves by lineage key → does any matching leaf paint real pixels?
    paints_by_key = {}
    for leaf, _box in _iter_leaf_layers_abs((design or {}).get("layers") or []):
        keys = _leaf_claim_keys(leaf)
        if not keys:
            continue
        painted = _leaf_paints(leaf, run_dir)
        for key in keys:
            paints_by_key[key] = paints_by_key.get(key, False) or painted
    claimed_numbers = set()
    for number, cid in owner_index.items():
        cid = str(cid)
        keys = {cid}
        normalized = _normalized_element_id(cid)
        if normalized:
            keys.add(normalized)
        claimed = any(paints_by_key.get(key) for key in keys)
        try:
            number_int = int(number)
        except (TypeError, ValueError):
            continue
        accounting["owners_total"] += 1
        if claimed:
            claimed_numbers.add(number_int)
            accounting["owners_claimed"].append(cid)
        else:
            accounting["owners_unclaimed"].append(cid)
    try:
        ownership = np.asarray(Image.open(ownership_path))
    except Exception:
        return None, accounting
    if ownership.ndim > 2:
        ownership = ownership[..., 0]
    width, height = size
    if ownership.shape != (height, width):
        ownership = np.asarray(
            Image.fromarray(ownership).resize((width, height), Image.Resampling.NEAREST))
    scale = max(1, 65535 // max(1, len(owner_index)))
    numbers = np.rint(ownership.astype(np.float64) / scale).astype(np.int64)
    claimed_mask = np.isin(numbers, sorted(claimed_numbers)) & (ownership > 0)
    accounting["available"] = True
    return claimed_mask, accounting


def _background_audit(source_rgb, background_path, removal_mask, run_dir=None, design=None):
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
                "changed_canvas_ratio": 0.0, "destroyed_canvas_ratio": 0.0,
                "claimed_canvas_ratio": 0.0, "removal_claims": None,
                "edge_retention": None, "mean_change": 0.0,
                "outside_changed_ratio": 0.0, "outside_mean_change": 0.0}
    total_px = int(height) * int(width)
    delta = np.abs(source_rgb - background).mean(axis=2)
    exact = float((delta[mask] < 0.5).mean())
    changed = float((delta[mask] > 8.0).mean())
    # Fraction of the WHOLE canvas that was actually altered inside the removal region.
    # changed_ratio is per-mask; this is per-canvas, so it measures how much of the plate
    # the removal/inpaint pass altered regardless of how large the mask was (F3).
    changed_px = (delta > 8.0) & mask
    changed_canvas = float(int(changed_px.sum()) / max(1, total_px))
    # Destruction-accounting fairness: an altered plate pixel whose removal-ledger owner
    # re-renders REAL pixels on top (e.g. 066's eye photos inpainted out and shipped back
    # as swappable cutouts) is preserved-and-editable content, not destruction. Destroyed
    # = altered pixels with NO real re-rendered owner. Falls back to the raw changed ratio
    # when the ledger artifacts are unavailable (never more lenient without evidence).
    destroyed_canvas = changed_canvas
    claimed_canvas = 0.0
    claim_accounting = None
    if run_dir:
        claimed_mask, claim_accounting = _removal_claim_mask(run_dir, design, (width, height))
        if claimed_mask is not None:
            claimed_canvas = float(int((changed_px & claimed_mask).sum()) / max(1, total_px))
            destroyed_canvas = float(
                int((changed_px & ~claimed_mask).sum()) / max(1, total_px))
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
        "destroyed_canvas_ratio": round(destroyed_canvas, 5),
        "claimed_canvas_ratio": round(claimed_canvas, 5),
        "removal_claims": claim_accounting,
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
    # Source grayscale (luma) drives the ink-weighted editable_text_fraction. Derived from the
    # RGB source the audit already holds; None-safe so a caller without an image still scores.
    _src_gray = None
    if source_rgb is not None:
        try:
            _arr = source_rgb.astype("float32") if hasattr(source_rgb, "astype") else None
            if _arr is not None and _arr.ndim == 3:
                _src_gray = _arr[..., 0] * 0.299 + _arr[..., 1] * 0.587 + _arr[..., 2] * 0.114
            elif _arr is not None:
                _src_gray = _arr
        except Exception:
            _src_gray = None
    text_editability = _text_editability(source_ocr, design, layers, source_gray=_src_gray)
    editable_text_recall = None if text_editability is None else text_editability["editable_text_recall"]
    native_text_ratio_metric = None if text_editability is None else text_editability["native_text_ratio"]
    editable_text_fraction = None if text_editability is None else text_editability.get("editable_text_fraction")
    all_source_text_baked = bool(text_editability and text_editability.get("all_source_text_baked"))
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
    background = _background_audit(source_rgb, background_path, removal_mask,
                                   run_dir=run_dir, design=design)
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
    # Scene-baked verdict: merge's explicit ``photographic_scene_text`` flag OR a
    # ``caption_over_photo`` archetype whose merge routing baked EVERY confident source
    # line into the photo (021: in-scene sticky-note / printed copy). merge does not
    # always raise the top-level flag even when it correctly bakes every line via the
    # geometric cutout / scene-text path, so a caption_over_photo whose kept_in_photo
    # already accounts for all source text is just as legitimately scene-baked. The
    # archetype allowlist mirrors _photographic_scene_text_mode (merge): comparison_grid
    # (025), social_screenshot (009), lifestyle overlays keep full editability strictness
    # — a design that rasterized their genuinely-editable copy still hard-fails. The
    # per-fixture proof is condition-2 below (all source lines baked) plus the block-reason
    # gate (real, non-empty photographic tree).
    _archetype = _read_archetype(run_dir)
    _scene_baked_verdict = _photo_scene or _archetype == "caption_over_photo"
    if _scene_baked_verdict:
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
    # F-vacuous gate (021 class): every readable source line was baked-by-design, so
    # editable_text_recall is undefined (denom == 0). editable_text_recall / true_text_coverage
    # can no longer read a false 1.0, so the two gates above skip this run entirely — but a run
    # that rasterized EVERY line of copy must never come back CLEAN. If merge's evidence says
    # this is a genuine photographic scene (021: printed sticky-note copy IN the photo), it is
    # exempt and reported distinctly as "nothing to decompile"; if it is NOT scene-baked, it is
    # a photocopy that silently delivered zero editable text and hard-fails. This is the
    # explicit gate the None value is paired with — it strengthens QA, it does not relax it.
    if all_source_text_baked and source_lines:
        if scene_baked_all_text:
            supplied.setdefault("scene_baked_photo", True)
            supplied.setdefault("nothing_to_decompile", True)
        else:
            _add_fail(
                fails, "no-editable-content",
                "every readable source line was baked into a photo layer (0 editable text "
                "nodes) with no photographic-scene verdict — the ad's copy is not editable",
            )
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
        # Fairness: the gate counts DESTROYED pixels — altered plate pixels with NO real
        # re-rendered owner (removal-ledger accounting, see _removal_claim_mask). 066's
        # changed_canvas ~0.69 is legitimate: the eye photos were inpainted out and shipped
        # back as swappable cutouts, so their pixels are preserved content, not destruction.
        # When the ledger artifacts are missing, destroyed == changed (no unearned leniency).
        canvas_ratio = background.get("changed_canvas_ratio")
        destroyed_ratio = background.get("destroyed_canvas_ratio", canvas_ratio)
        ceiling = thresholds.get("background_changed_ratio_max")
        if (background.get("mask_supplied") and destroyed_ratio is not None
                and ceiling is not None and destroyed_ratio > float(ceiling)):
            unclaimed = ((background.get("removal_claims") or {}).get("owners_unclaimed")
                         or [])[:4]
            _add_fail(
                fails,
                "excessive-plate-destruction",
                f"removal/inpaint destroyed {destroyed_ratio:.1%} of the canvas with no "
                f"re-rendered owner (> {float(ceiling):.0%}; altered total {canvas_ratio:.1%}"
                + (f"; unclaimed owners: {', '.join(unclaimed)}" if unclaimed else "")
                + ") — real content was erased, not re-rendered",
            )
    # F15: unresolved glyph residue under a removed text region is a structural failure, not
    # a bare repair suggestion. QA must not report ok while it stands (009 shipped ok with a
    # high-severity glyph-residue repair still outstanding after no-op harness rounds).
    # Honesty: treat ``hard_fail: True`` OR ``resolved`` falsy as unresolved — a closer that
    # greenwashes by flipping resolved while leaving hard_fail set still blocks acceptance.
    glyph_residue_unresolved = 0
    if thresholds.get("glyph_residue_gate", True):
        text_residual = reconstruction_stats.get("text_residual") or {}
        unresolved_residue = []
        for entry in (text_residual.get("flagged") or []):
            if not isinstance(entry, dict):
                continue
            if entry.get("hard_fail") or not entry.get("resolved"):
                unresolved_residue.append(entry)
        glyph_residue_unresolved = len(unresolved_residue)
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
        # None (not 1.0) when every line was baked-by-design — see _text_editability F-vacuous.
        "native_text_ratio": None if native_text_ratio_metric is None else round(native_text_ratio_metric, 4),
        # Coverage of ALL source text that is editable (kept_in_photo in the denominator).
        # Reads ~0.0 for a fully-baked photo (021), high for a real decompile (013).
        "editable_text_fraction": None if editable_text_fraction is None else round(editable_text_fraction, 4),
        # True iff editable recall is undefined because 100% of the copy is baked-by-design.
        # Paired with the scene_baked_photo / nothing_to_decompile flags below so a reader can
        # tell "nothing to decompile" (021) apart from "decompiled perfectly".
        "all_source_text_baked": all_source_text_baked,
        "scene_baked_photo": bool(supplied.get("scene_baked_photo")),
        "nothing_to_decompile": bool(supplied.get("nothing_to_decompile")),
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
        "glyph_residue_unresolved": glyph_residue_unresolved,
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
    # contract failure (the plate is not clean). Prefer the auditor count; also fail if
    # any glyph-residue hard-fail row is present (harness cannot greenwash via count=0).
    residue_unresolved = structure.get("glyph_residue_unresolved")
    if not isinstance(residue_unresolved, int):
        residue_unresolved = 0
    has_residue_fail = False
    for fail in structure.get("hard_fails") or []:
        if isinstance(fail, dict) and fail.get("rule") == "glyph-residue":
            has_residue_fail = True
            if residue_unresolved <= 0:
                detail = str(fail.get("detail") or "")
                n = next((int(t) for t in detail.split() if t.isdigit()), 1)
                residue_unresolved += n if n > 0 else 1
    if has_residue_fail and residue_unresolved <= 0:
        residue_unresolved = 1
    glyph_residue_clean = (residue_unresolved == 0) and not has_residue_fail

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


def _write_worst_window_crop(source_rgb, render_rgb, bbox, run_dir, pad=24):
    """Write qa_worst_window.png: source|render side-by-side crop of the worst window.

    Forensic artifact for the recurring local-ssim-worst-region fails (002/016/088/094/
    104/131): the exact region that sank the score, source on the left, render on the
    right, red frame marking the un-padded window. Small crops are upscaled so glyph-level
    damage is visible without a zoom tool."""
    import numpy as np
    from PIL import Image

    height, width = source_rgb.shape[:2]
    x0 = max(0, int(bbox.get("x", 0)) - pad)
    y0 = max(0, int(bbox.get("y", 0)) - pad)
    x1 = min(width, int(bbox.get("x", 0) + bbox.get("w", 0)) + pad)
    y1 = min(height, int(bbox.get("y", 0) + bbox.get("h", 0)) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    def _crop(arr):
        crop = np.clip(arr[y0:y1, x0:x1], 0, 255).astype(np.uint8)
        # Red frame on the un-padded window bounds so the scored region is unambiguous.
        fx0 = max(0, int(bbox.get("x", 0)) - x0); fy0 = max(0, int(bbox.get("y", 0)) - y0)
        fx1 = min(crop.shape[1] - 1, fx0 + max(1, int(bbox.get("w", 0))))
        fy1 = min(crop.shape[0] - 1, fy0 + max(1, int(bbox.get("h", 0))))
        crop = crop.copy()
        crop[fy0:fy0 + 2, fx0:fx1] = (255, 0, 0)
        crop[max(0, fy1 - 2):fy1, fx0:fx1] = (255, 0, 0)
        crop[fy0:fy1, fx0:fx0 + 2] = (255, 0, 0)
        crop[fy0:fy1, max(0, fx1 - 2):fx1] = (255, 0, 0)
        return crop

    left, right = _crop(source_rgb), _crop(render_rgb)
    scale = 1
    if max(left.shape[0], left.shape[1]) < 200:
        scale = max(1, int(round(200 / max(1, max(left.shape[0], left.shape[1])))))
    divider = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    panel = np.concatenate([left, divider, right], axis=1)
    image = Image.fromarray(panel)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale),
                             Image.Resampling.NEAREST)
    out_path = os.path.join(run_dir, "qa_worst_window.png")
    image.save(out_path)
    return out_path


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

    design_data = _load_design(design, run_dir)
    text_recall = None
    text_recall_detail = None
    # F-recall-honesty: recall is now delivered-vs-source-truth and no longer needs
    # render_ocr, so it is computed whenever source text exists. Previously a run with no
    # render-OCR silently reported text_recall=None (no gate at all).
    if source_ocr:
        text_recall_detail = _text_recall_detail(
            source_ocr, render_ocr or {}, source_gray, render_gray,
            design=design_data, run_dir=run_dir)
        text_recall = text_recall_detail["recall"]
    render_legibility = _render_text_legibility(render_ocr, design_data)
    ink_ownership = _ink_ownership_ledger(source_ocr or {}, design_data, run_dir,
                                          source_gray) if source_ocr else None

    opts = dict(DEFAULT_THRESHOLDS)
    # F-per-archetype-floor: apply the archetype preset's own edge/color floors (if any)
    # before the caller's explicit thresholds, so an explicit override always still wins.
    opts.update(_archetype_threshold_overrides(run_dir))
    opts.update(thresholds or {})
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
    # quality_warnings are REPORTED but never merged into hard_fails: they are the channel
    # for "we measured something real, and it is not a defect a human would see". Keeping
    # them out of quality_flags is deliberate — harness.py/_blocker_names treats every
    # quality_flag as an acceptance blocker.
    quality_warnings = []
    if ink_ownership and ink_ownership.get("doubles"):
        _doubled = [l["text"] for l in ink_ownership["lines"] if l["state"] == "DOUBLE"]
        quality_flags.append({
            "rule": "ink-double-render",
            "detail": ("%d line(s) ship BOTH a native text node AND verbatim source ink "
                       "still in the plate — a visible ghost double: %s"
                       % (len(_doubled), ", ".join(repr(t) for t in _doubled[:4]))),
        })
    if multiscale < opts["local_ssim_min"]:
        quality_flags.append({"rule": "local-ssim", "detail": f"{multiscale:.3f} < {opts['local_ssim_min']:.3f}"})
    if edge["f1"] < opts["edge_f1_min"]:
        quality_flags.append({"rule": "edge-fidelity", "detail": f"{edge['f1']:.3f} < {opts['edge_f1_min']:.3f}"})
    # F-colour-honesty: a whole-image colour miss only HARD-fails when a measure that can
    # actually see colour agrees. Uncorroborated means the dE is glyph coverage wearing a
    # colour name (067: every emitted hue exact, 99% of the "colour" error is text
    # placement) — reported, not blocking. See _color_metrics.
    if color["similarity"] < opts["color_similarity_min"]:
        local_min = opts.get("color_local_similarity_min", 0.98)
        if color["local_similarity"] < float(local_min):
            quality_flags.append({
                "rule": "color-fidelity",
                "detail": (f"{color['similarity']:.3f} < {opts['color_similarity_min']:.3f} "
                           f"and colour-comparable local similarity "
                           f"{color['local_similarity']:.3f} < {float(local_min):.3f} "
                           f"(mean local dE {color['delta_e_local_mean']:.2f}) — real colour error")})
        else:
            quality_warnings.append({
                "rule": "color-fidelity-coverage-artifact",
                "detail": (f"whole-image color_similarity {color['similarity']:.3f} < "
                           f"{opts['color_similarity_min']:.3f}, but every colour we paint is "
                           f"within mean dE {color['delta_e_local_mean']:.2f} of the source's "
                           f"own local colours (local similarity {color['local_similarity']:.3f}) "
                           f"— the dE is glyph COVERAGE, not colour; see local-ssim-worst-region "
                           f"and the font checks for the placement/face defects that cause it")})
    # F8: per-archetype text strictness. The archetype preset's text_recall_min (0.90 for
    # social) is threaded into thresholds by the caller; enforce it here so the strict text
    # bar the preset promises actually gates instead of being wired nowhere. Only fires when
    # a render-OCR text_recall exists AND a threshold was supplied.
    text_recall_min = opts.get("text_recall_min")
    if text_recall is not None and text_recall_min is not None and text_recall < float(text_recall_min):
        quality_flags.append({"rule": "low-text-recall",
                              "detail": f"text recall {text_recall:.3f} < {float(text_recall_min):.3f}"})

    # F-deliverable-consistency: compare the DECLARED font against the file the preview drew.
    # Catches the SSIM-invisible class (013: declared Bold, italic file, preview drew italic,
    # Figma ships upright) and quantifies how much of the preview is evidence about the
    # deliverable at all. See _font_consistency_audit.
    font_consistency = _font_consistency_audit(design_data)
    if font_consistency:
        for node in font_consistency["mismatched_nodes"]:
            declared, drawn = node["declared"], node["drawn"]
            where = (f"node {node['id']} declares {declared['family']!r}/{declared['style']!r}"
                     f"/w{declared['weight']:.0f} but the preview drew "
                     f"{drawn['family']!r}/{drawn['style']!r} ({drawn['file']})")
            if "italic-drawn-not-declared" in node["issues"]:
                # Ships wrong: Figma resolves the STYLE NAME (upright), the preview resolved
                # the FILE (italic). No fallback face is italic, so this cannot be a missing
                # font on this machine — it is a real declaration/file disagreement.
                quality_flags.append({"rule": "font-style-mismatch", "detail":
                    where + " — Figma resolves the declared style NAME and would ship this "
                            "UPRIGHT while the preview shows italic; pixel metrics cannot "
                            "see this"})
            else:
                quality_warnings.append({"rule": "font-declaration-mismatch",
                                         "detail": where + f" [{', '.join(node['issues'])}]"})

    # F-worst-region: gate the single worst local SSIM window independently of the
    # mean-dominated aggregate, so a catastrophic region (009/016: worst window ~0.03-0.04)
    # cannot hide under a good global/aggregate score. Evidence carries the pixel bbox.
    worst_window = _local_ssim_worst_window(source_gray, render_gray, preserve_mask)
    # Worst-window forensics: every run gets a side-by-side source|render crop of the
    # single worst window (qa_worst_window.png) so a local-ssim-worst-region fail is
    # diagnosable at a glance without re-deriving the bbox by hand.
    worst_window_png = None
    if worst_window and worst_window.get("bbox"):
        try:
            worst_window_png = _write_worst_window_crop(
                source_rgb, render_rgb, worst_window["bbox"], run_dir)
        except Exception:
            worst_window_png = None
    worst_window_min = opts.get("local_ssim_worst_window_min")
    # F-worst-region-honesty: a below-floor window only HARD-fails when no local translation
    # explains it. Windows that are merely DRIFTED (our substituted font's advance widths
    # walking the glyphs a few px along the run) are reported as a non-blocking warning with
    # their measured shift, NOT as a hard fail — see `_translation_explains` for the
    # per-fixture evidence and for why the test is symmetric.
    window_report = None
    if worst_window is not None and worst_window_min is not None:
        window_report = _classify_subfloor_windows(
            source_gray, render_gray, preserve_mask, worst_window_min,
            opts.get("local_ssim_shift_radius_ratio", 0.5),
            opts.get("local_ssim_shift_explained_min", 0.50))
        if window_report["damage"]:
            worst_damage = window_report["damage"][0]
            bbox = worst_damage["bbox"]
            quality_flags.append({
                "rule": "local-ssim-worst-region",
                "detail": (
                    f"worst unexplained local window ssim {worst_damage['ssim']:.3f} < "
                    f"{float(worst_window_min):.3f} at x={bbox['x']} y={bbox['y']} "
                    f"w={bbox['w']} h={bbox['h']} — not explained by translation "
                    f"(best match within +/-{window_report['radius']}px: "
                    f"{worst_damage['shift_tolerant_ssim']:.3f})"
                ),
                "bbox": bbox,
            })
        if window_report["drift"]:
            # One aggregated warning, not one per window: 067 has 20 drifted windows and a
            # wall of near-identical entries is its own kind of noise. Every window is still
            # listed in full under qa["local_ssim_window_report"]["drift"].
            worst_drift = min(window_report["drift"], key=lambda item: item["ssim"])
            bbox = worst_drift["bbox"]
            quality_warnings.append({
                "rule": "local-ssim-worst-region-shifted",
                "detail": (
                    f"{len(window_report['drift'])} local window(s) below "
                    f"{float(worst_window_min):.3f} are explained by translation, not damage; "
                    f"worst is ssim {worst_drift['ssim']:.3f} at x={bbox['x']} y={bbox['y']} "
                    f"w={bbox['w']} h={bbox['h']}, which a dx={worst_drift['shift']['dx']} "
                    f"dy={worst_drift['shift']['dy']} shift restores to "
                    f"{worst_drift['shift_tolerant_ssim']:.3f} — glyph drift from font "
                    f"substitution (see deliverable_font_consistency), not local damage"
                ),
                "bbox": bbox,
                "count": len(window_report["drift"]),
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
        # F-colour-honesty: colour measured where colour is comparable. This is the reading
        # that says whether the HUES are right; color_similarity above says whether the ink
        # landed on the same pixels, which is a coverage question. See _color_metrics.
        "color_local_similarity": round(color["local_similarity"], 4),
        "delta_e_local_mean": round(color["delta_e_local_mean"], 4),
        "delta_e_mean": round(color["delta_e_mean"], 4),
        "delta_e_p95": round(color["delta_e_p95"], 4),
        "rgb_mae": round(color["rgb_mae"], 4),
        "visual_score": round(visual_score, 4),
        "quality_flags": quality_flags,
        # Non-blocking, still reported: see the quality_warnings note above.
        "quality_warnings": quality_warnings,
        "text_recall": None if text_recall is None else round(text_recall, 4),
        # Denominator honesty: how many source lines were excluded as correctly-baked
        # product text (kept_in_photo + pixels verbatim in the owning asset), with the
        # excluded line texts listed so the exclusion is auditable, never silent.
        "text_recall_detail": text_recall_detail,
        "ink_ownership": ink_ownership,
        # "Does our preview draw what we delivered" — the old render-OCR round-trip, kept as
        # an honest, separately-named, NON-gating signal. See _render_text_legibility.
        "render_text_legibility": render_legibility,
        # F-deliverable-consistency: declared font vs the face the preview actually drew.
        # `preview_matches_declaration_ratio` is the honest caveat on every preview-derived
        # pixel number in this file — see _font_consistency_audit.
        "deliverable_font_consistency": font_consistency,
        "editable_text_recall": structure["editable_text_recall"],
        # Coverage of ALL source text that ships editable (kept_in_photo in the denominator).
        # Top-level mirror so a bare ``jq .editable_text_fraction qa.json`` — the user's real
        # "how much of this ad can I edit?" question — reads 0.0 on a fully-baked photo (021)
        # instead of the vacuous 1.0 editable_text_recall used to show.
        "editable_text_fraction": structure.get("editable_text_fraction"),
        # "Nothing to decompile" (021) vs "decompiled perfectly": both used to read as CLEAN
        # with recall 1.0; these flags make the distinction first-class and machine-readable.
        "all_source_text_baked": structure.get("all_source_text_baked"),
        "nothing_to_decompile": structure.get("nothing_to_decompile"),
        "scene_baked_photo": structure.get("scene_baked_photo"),
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
        # Unchanged raw signal (compat: qa_reward/repair/harness read it): the single worst
        # window's RAW SSIM, still honestly reported even when it is only glyph drift.
        "local_ssim_worst_window": worst_window,
        # ...and the honest reading of the same grid: which below-floor windows are real
        # damage vs translation drift, each with its measured shift and both-way match.
        "local_ssim_window_report": window_report,
        "worst_window_png": worst_window_png,
        "per_layer": per_layer,
        "per_region_max_delta": float(cells.max()),
        "per_region": {"rows": gy, "cols": gx, "mean_delta": cells.round(3).tolist(), "worst": ranked[:8]},
        "diff_png": diff_png,
        "structural": structure,
        "hard_fails": hard_fails,
    }
