"""Materialize canonical assets and a duplicate-free background plate.

This is the first stage that turns detections into pixels with ownership.  It resolves all
run-relative paths, removes duplicate observations, extracts alpha crops, routes simple
graphics through the vector fidelity gate, samples native shape fills, and sends one final
union mask to :mod:`src.inpaint`.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Optional

from . import inpaint, vectorize
from .schema import (
    dump, load, raster_slice_failures, raster_slice_thresholds, is_raster_slice,
)
from .raster_clusters import INTENTIONAL_RASTER_CLUSTER_ROLES, is_intentional_raster_cluster
from .diagram_editability import prefer_decomposed_charts


def _deps():
    import cv2
    import numpy as np
    from PIL import Image
    return cv2, np, Image


def _inpaint_used_opencv(inpaint_result) -> bool:
    """True if any inpaint pass fell back to the low-quality OpenCV backend.

    Defensive across the three inpaint result shapes: regional (``backend_counts``),
    single-pass (``diagnostics.backend_route.opencv_fallback_used`` / ``backend``), and the
    degenerate empty-mask ``backend: "none"`` case (which is not a fallback).
    """
    if not isinstance(inpaint_result, dict):
        return False
    # Regional: any region that used an opencv-* backend counts as a fallback.
    counts = inpaint_result.get("backend_counts")
    if isinstance(counts, dict) and any(str(k).startswith("opencv") for k in counts):
        return True
    # Single-pass diagnostics carry the explicit flag on the backend route.
    diagnostics = inpaint_result.get("diagnostics")
    if isinstance(diagnostics, dict):
        route = diagnostics.get("backend_route")
        if isinstance(route, dict) and route.get("opencv_fallback_used"):
            return True
    # Fall back to the resolved backend name on either shape.
    if str(inpaint_result.get("backend") or "").startswith("opencv"):
        return True
    return False


# A "shape" region whose interior colour dispersion (max per-channel std) exceeds this is
# photographic (a real photo/avatar), not a flat/gradient design fill, so it must stay a
# swappable IMAGE clipped by its detected primitive instead of being flattened to a colour.
PHOTO_SHAPE_MIN_STD = 28.0


def _iou(a, b):
    ix = max(0.0, min(a.get("x", 0) + a.get("w", 0), b.get("x", 0) + b.get("w", 0))
             - max(a.get("x", 0), b.get("x", 0)))
    iy = max(0.0, min(a.get("y", 0) + a.get("h", 0), b.get("y", 0) + b.get("h", 0))
             - max(a.get("y", 0), b.get("y", 0)))
    inter = ix * iy
    union = a.get("w", 0) * a.get("h", 0) + b.get("w", 0) * b.get("h", 0) - inter
    return inter / union if union > 0 else 0.0


def _inside_frac(inner, outer):
    """Fraction of ``inner`` box area that lies inside ``outer``."""
    ix0 = max(float(inner.get("x", 0) or 0), float(outer.get("x", 0) or 0))
    iy0 = max(float(inner.get("y", 0) or 0), float(outer.get("y", 0) or 0))
    ix1 = min(float(inner.get("x", 0) or 0) + float(inner.get("w", 0) or 0),
              float(outer.get("x", 0) or 0) + float(outer.get("w", 0) or 0))
    iy1 = min(float(inner.get("y", 0) or 0) + float(inner.get("h", 0) or 0),
              float(outer.get("y", 0) or 0) + float(outer.get("h", 0) or 0))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, float(inner.get("w", 0) or 0) * float(inner.get("h", 0) or 0))
    return inter / area


_SHELL_VECTORIZE_ROLES = frozenset({
    "badge", "button", "chip", "pill", "cta", "callout", "logo", "sticker",
    "price_burst", "sale_burst", "starburst", "burst",
    "banner", "ribbon", "brushstroke", "seal", "shape",
})


def _classify_shell_role_from_box(shell_box: dict, current_role: str) -> str:
    """Wide plates → banner; square seals → badge. Preserves button/starburst labels."""
    role = str(current_role or "").lower().replace("-", "_")
    if role in {"button", "cta", "chip", "pill", "callout"}:
        return "button" if role == "cta" else role
    if role in {
        "starburst", "price_burst", "sale_burst", "burst", "splat", "sticker_burst", "seal",
    }:
        return role
    if role in {"banner", "ribbon", "brushstroke", "stroke_banner"}:
        return "banner"
    if role == "badge":
        return "badge"
    w = float((shell_box or {}).get("w", 0) or 0)
    h = float((shell_box or {}).get("h", 0) or 0)
    if h > 0 and w / h >= 2.2:
        return "banner"
    return "badge"


def _promote_ocr_overlapping_shells(
    candidates: list,
    cfg: Optional[dict] = None,
    canvas: Optional[dict] = None,
) -> int:
    """Skip vectorize/raster-chip paths for shells that clearly host OCR text.

    Merge normally flags ``text_bearing_shell``; this is a reconstruct-time safety net so
    a missed flag cannot spend VTracer time then bake the seal (016 green badge).
    Contained OCR stays TEXT with removal; the shell becomes a SHAPE plate.
    """
    rcfg = (cfg or {}).get("reconstruct") or {}
    if not bool(rcfg.get("promote_text_shells", True)):
        return 0
    threshold = float(rcfg.get("text_shell_inside", 0.55))
    texts = [
        c for c in candidates
        if c.get("target") == "text" and (c.get("text") or (c.get("meta") or {}).get("text"))
    ]
    if not texts:
        return 0
    promoted = 0
    for shell in candidates:
        meta = shell.setdefault("meta", {})
        if meta.get("text_bearing_shell") or meta.get("plate_shell"):
            continue
        if shell.get("target") not in {"icon", "shape", "image"}:
            continue
        role = str(meta.get("role") or "").lower()
        # Packaging / photo cutouts keep baked scene text; only chrome hosts promote.
        if role in {
            "product", "person", "photo", "photo-fragment", "photo_fragment",
            "cutout", "package", "packaging", "pouch", "bottle", "jar",
        }:
            continue
        if role not in _SHELL_VECTORIZE_ROLES and shell.get("target") not in {"icon", "shape"}:
            continue
        hosts = [
            t for t in texts
            if _inside_frac(t.get("box") or {}, shell.get("box") or {}) >= threshold
        ]
        if not hosts:
            continue
        shell_box = shell.get("box") or {}
        # Near-coincident OCR boxes are not chrome plates.
        if any(
            _inside_frac(shell_box, t.get("box") or {}) >= 0.85
            for t in hosts
        ):
            continue
        # Mirror merge's residual-shell admission gate.  This safety-net runs after
        # merge, so it must not undo merge's deliberate rejection and turn a huge
        # residual negative-space component into a badge/banner again (002).  Once
        # promoted, photo-shape detection is skipped and the region becomes a giant
        # inpaint hole; keeping it neutral lets the photo override + removal cap retain
        # the exact source panel instead.
        canvas_area = (
            float((canvas or {}).get("w", 0) or 0)
            * float((canvas or {}).get("h", 0) or 0)
        )
        shell_area = (
            float(shell_box.get("w", 0) or 0)
            * float(shell_box.get("h", 0) or 0)
        )
        area_frac = shell_area / canvas_area if canvas_area > 0 else 0.0
        width_frac = (
            float(shell_box.get("w", 0) or 0)
            / float((canvas or {}).get("w", 0) or 1)
            if float((canvas or {}).get("w", 0) or 0) > 0 else 0.0
        )
        provenance_sources = {
            str(source).lower()
            for source in ((meta.get("provenance") or {}).get("sources") or [])
        }
        generic_residual = role in {"", "shape", "photo"} and any(
            "residual" in source for source in provenance_sources
        )
        if canvas_area > 0 and (
            area_frac >= 0.35
            or (generic_residual and area_frac >= 0.05 and width_frac >= 0.70)
        ):
            meta["text_shell_rejected"] = "oversized-residual-shell"
            continue
        new_role = _classify_shell_role_from_box(shell_box, role)
        if role and role != new_role:
            meta["reclassified_from"] = meta.get("reclassified_from") or role
        meta["role"] = new_role
        meta["text_bearing_shell"] = True
        meta["plate_shell"] = True
        meta["shell_text_promoted"] = True
        snippet = " ".join(
            str(t.get("text") or (t.get("meta") or {}).get("text") or "").strip()
            for t in hosts
        ).strip()
        if snippet:
            meta["shell_text_snippet"] = snippet[:48]
        shell["target"] = "shape"
        for t in hosts:
            tm = t.setdefault("meta", {})
            tm["overlay_text"] = True
            tm["removal_required"] = True
            tm.setdefault("shell_text_host", shell.get("id"))
            tm.setdefault("parent_id", shell.get("id"))
            t["target"] = "text"
            t.pop("kept_in_photo", None)
        promoted += 1
    return promoted


def _confidence(candidate):
    return float((candidate.get("meta") or {}).get("confidence") or candidate.get("score") or 0)


def _source_priority(candidate):
    source = str((candidate.get("meta") or {}).get("source") or candidate.get("source") or "")
    if "sam3" in source:
        return 4
    if "element+qwen" in source:
        return 3
    if "element" in source:
        return 2
    if "qwen" in source:
        return 1
    return 0


_LIST_GLYPH_ROLES = frozenset({
    "verified", "checkmark", "check", "check-mark", "check_mark", "tick",
    "cross", "question-mark", "question_mark",
})


def _is_cv_list_glyph(meta: dict) -> bool:
    """A ✓/✗/? glyph located by src/icon_detect.py's template match.

    The match itself is the independent evidence (and the chip box is the glyph's own
    matte), so these do not need ``_verified_semantic_mask``: 101's four crosses carry
    icon-cv provenance only, at scores .94-.95, with no SAM observation at all.
    """
    meta = meta or {}
    role = str(meta.get("role") or "").lower().replace("_", "-")
    return bool(meta.get("icon_cv")) or role in _LIST_GLYPH_ROLES


def _verified_semantic_mask(meta: dict) -> bool:
    """Whether an alpha matte has independent high-confidence SAM evidence.

    Residual connected components often cut holes through water, paper texture, or white
    lettering.  Conversely a product matte confirmed by SAM can legitimately contain the
    counters of a logo/wordmark.  Preserve that distinction for both asset selection and
    structural QA; an arbitrary alpha PNG must *not* gain this exemption.
    """
    provenance = (meta or {}).get("provenance") or {}
    # Some fused/Qwen observations carry a provenance list rather than the
    # element-fusion mapping. It is valid diagnostic data, but not independent
    # SAM-mask evidence.
    observations = provenance.get("observations") if isinstance(provenance, dict) else []
    observations = observations if isinstance(observations, list) else []
    for observation in observations:
        if (str(observation.get("source") or "").lower() == "sam3"
                and str(observation.get("mask_quality") or "").lower() == "mask"
                and float(observation.get("score") or 0) >= .70):
            return True
    return False


def _is_background_plate(candidate, width, height):
    box = candidate.get("box") or {}
    area_frac = box.get("w", 0) * box.get("h", 0) / max(1, width * height)
    role = str((candidate.get("meta") or {}).get("role") or "")
    tolerance_x, tolerance_y = width * .025, height * .025
    touches = sum((
        box.get("x", 0) <= tolerance_x,
        box.get("y", 0) <= tolerance_y,
        box.get("x", 0) + box.get("w", 0) >= width - tolerance_x,
        box.get("y", 0) + box.get("h", 0) >= height - tolerance_y,
    ))
    # A large edge-touching product/photo is the scene's hero photograph, not a small
    # editable cutout.  Removing it from the plate asks the inpainter to hallucinate
    # detailed packaging across a broad region; the same raster would then be painted
    # back on top, producing the characteristic smeared/ghosted preview.  Keep this
    # lower bound deliberately scoped to edge-touching photographic candidates so a
    # genuinely isolated photo remains editable.
    return (role == "background" or area_frac > .92 or
            # A package/product can legitimately be a large, edge-touching
            # foreground cutout (016 is exactly this case).  Treating it as a
            # plate makes it target=drop, skips removal, and leaves the final
            # reconstruction with only the broad photo observation.  Keep the
            # heuristic for scene photographs, but never for semantic product
            # candidates.
            (role in ("photo", "illustration", "image") and area_frac > .40 and touches >= 3))


def deduplicate(candidates: list, threshold: float = 0.86):
    """Drop same-object observations while preserving nested, semantically different layers."""
    ordered = sorted((dict(c) for c in candidates),
                     key=lambda c: (_source_priority(c), _confidence(c)), reverse=True)
    kept = []
    for candidate in ordered:
        if candidate.get("target") == "drop":
            kept.append(candidate)
            continue
        role = (candidate.get("meta") or {}).get("role") or candidate.get("kind")
        duplicate = False
        for other in kept:
            if other.get("target") == "drop":
                continue
            other_role = (other.get("meta") or {}).get("role") or other.get("kind")
            # Text and its backing button/shape are intentionally nested, not duplicates.
            if {candidate.get("target"), other.get("target")} == {"text", "shape"}:
                continue
            # Strong overlap is not sufficient when semantics differ: a product
            # commonly sits inside a broad photo observation and both are needed.
            # Only generic/unknown labels may collapse into a semantic winner.
            generic_roles = {None, "", "object", "image", "photo-fragment"}
            if role != other_role and role not in generic_roles and other_role not in generic_roles:
                continue
            if role != other_role and candidate.get("target") != other.get("target"):
                continue
            if _iou(candidate.get("box", {}), other.get("box", {})) >= threshold:
                other.setdefault("meta", {}).setdefault("merged_observations", []).append(
                    candidate.get("id")
                )
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    # Preserve the upstream paint order after selecting winners.
    order = {c.get("id"): i for i, c in enumerate(candidates)}
    return sorted(kept, key=lambda c: order.get(c.get("id"), 10**9))


def _box_containment(inner, outer):
    """Fraction of ``inner`` covered by ``outer`` (boxes are canvas coordinates)."""
    ix = max(0.0, min(inner.get("x", 0) + inner.get("w", 0),
                      outer.get("x", 0) + outer.get("w", 0))
             - max(inner.get("x", 0), outer.get("x", 0)))
    iy = max(0.0, min(inner.get("y", 0) + inner.get("h", 0),
                      outer.get("y", 0) + outer.get("h", 0))
             - max(inner.get("y", 0), outer.get("y", 0)))
    area = max(0.0, inner.get("w", 0) * inner.get("h", 0))
    return (ix * iy) / area if area else 0.0


def _suppress_baked_raster_text(candidates: list, threshold: float = .90) -> list:
    """Keep package/scene OCR baked into its canonical raster owner.

    OCR eagerly reads labels on cans, tubes, screenshots and photo cards.  Exporting those
    observations as editable text both duplicates the pixels in the raster and makes the
    removal mask damage the asset.  A caller can explicitly promote genuine overlay copy with
    ``overlay_text``/``promote_text``/``editable_text``.
    """
    raster_roles = {
        "photo", "product", "person", "foreground", "cutout", "avatar",
        "profile", "profile_photo", "thumbnail", "photo_card", "image",
    } | set(INTENTIONAL_RASTER_CLUSTER_ROLES)
    rasters = []
    for candidate in candidates:
        meta = candidate.get("meta") or {}
        role = str(meta.get("role") or "").lower()
        if candidate.get("target") == "image" and role in raster_roles:
            rasters.append(candidate)

    out = []
    for candidate in candidates:
        c = dict(candidate)
        c["meta"] = dict(candidate.get("meta") or {})
        meta = c["meta"]
        promoted = bool(meta.get("overlay_text") or meta.get("promote_text")
                        or meta.get("editable_text") or meta.get("text_promoted"))
        if c.get("target") == "text" and not promoted:
            parent_id = meta.get("parent_id") or c.get("parent_id")
            owner = next((r for r in rasters if r.get("id") == parent_id), None)
            if owner is None:
                owner = next((r for r in rasters
                              if _box_containment(c.get("box") or {}, r.get("box") or {})
                              >= threshold), None)
            if owner is not None:
                c["target"] = "drop"
                meta["kept_in_photo"] = True
                meta["baked_owner_id"] = owner.get("id")
                meta["suppression_reason"] = "text-contained-in-raster-owner"
        out.append(c)
    return out


def _suppress_comparison_column_labels(candidates: list, cfg: dict) -> list:
    """Drop standalone Before/After OCR when labels are baked into column photos."""
    scene = cfg.get("scene") or {}
    if scene.get("archetype") != "comparison_grid":
        return candidates
    facts = scene.get("facts") or {}
    photo_columns = [
        item for item in candidates
        if item.get("target") == "image"
        and ((item.get("meta") or {}).get("comparison_side")
             or str((item.get("meta") or {}).get("semantic_name") or "").lower().endswith("image"))
    ]
    if not facts.get("before_after_pair") and len(photo_columns) < 2:
        return candidates
    label_re = re.compile(r"^\s*(before|after)\s*$", re.I)
    out = []
    for candidate in candidates:
        c = dict(candidate)
        c["meta"] = dict(candidate.get("meta") or {})
        meta = c["meta"]
        if c.get("target") == "text" and label_re.match(str(c.get("text") or "")):
            box = c.get("visible_box") or c.get("box") or {}
            owner = next(
                (col for col in photo_columns
                 if _box_containment(box, col.get("visible_box") or col.get("box") or {}) >= 0.35
                 or _iou(box, col.get("visible_box") or col.get("box") or {}) >= 0.08),
                None,
            )
            # F7: only bake a Before/After label into a column photo when it is actually
            # CONTAINED in that photo (its pixels are part of the raster). The mere
            # existence of a before_after_pair used to force every such label uneditable
            # even when it sits in the gutter/below the photos, contradicting the
            # comparison-grid archetype's "rebuild column copy as editable" contract
            # (benchmark 052 pills baked with no overlap). A non-overlapping label stays a
            # real, swappable TEXT layer.
            if owner is not None:
                c["target"] = "drop"
                meta["kept_in_photo"] = True
                meta["baked_owner_id"] = owner.get("id")
                meta["suppression_reason"] = "comparison-column-label-baked"
        out.append(c)
    return out


# Semantic foreground/asset roles that legitimately get a clean plate behind them even
# when large: removing them keeps a genuinely-swappable Figma asset. The per-candidate
# removal cap (which stops a spurious backdrop blob from inpainting most of the canvas)
# never fires on these — it targets big, LOW-confidence, generic-role rasters only.
_CAP_EXEMPT_ROLES = frozenset({
    "product", "person", "people", "portrait", "foreground", "cutout", "subject",
    "hero", "avatar", "profile", "profile_photo", "logo", "brand", "wordmark",
    "platform-logo", "badge", "icon", "card", "thumbnail", "photo_card", "photo-card",
    "panel", "image-panel", "photo-panel", "comparison-column", "comparison-panel",
})

# Thin-stroke marks whose SAM mattes chronically miss rim detail; their removal masks
# get unioned with a measured ink-contrast mask (see the cutout_ink_union block).
_LINE_ART_ROLES = frozenset({
    "logo", "brand", "wordmark", "platform-logo", "icon", "badge", "seal", "stamp",
})


def _keeps_underlay(candidate: dict) -> bool:
    """True for overlay layers whose already-valid underlying plate must not be erased."""
    meta = candidate.get("meta") or {}
    return bool(meta.get("keep_underlay") or meta.get("preserve_underlay")
                or meta.get("overlay_without_removal"))


def _residual_baked_raster(meta: dict) -> bool:
    """A pixel-exact baked slice sourced from a residual connected component.

    Display glyphs a display face defeats OCR on (088 "SALE"/"21%") never become TEXT;
    they survive as residual-CC observations and ship as baked raster slices
    (``fallback: raster-slice`` / ``ownership_cutout`` with a ``residual`` provenance
    source).  Those slices are literally original pixels at the original position.
    """
    meta = meta or {}
    if str(meta.get("fallback") or "") == "raster-slice":
        return True
    provenance = meta.get("provenance") or {}
    sources = set()
    if isinstance(provenance, dict):
        sources = {str(s).lower() for s in (provenance.get("sources") or [])}
    src = str(meta.get("source") or "").lower()
    return bool(meta.get("ownership_cutout")) and (
        "residual" in src or any("residual" in s for s in sources)
    )


def _plate_ring_is_flat(rgb, box, ink_mask, rcfg) -> bool:
    """Is the plate around this ink fragment a flat, uniform colour?

    Used to decide whether a residual-CC baked glyph may keep its ink in the plate: on a
    flat plate the kept ink is exactly the scene colour the slice re-paints, so leaving it
    fills the ownership gaps between fragmented slices without any photographic double.  A
    fragment sitting on a gradient/photographic carrier (a ribbon, a product) fails this and
    is left to the normal removal path.
    """
    _, np, _ = _deps()
    h, w = rgb.shape[:2]
    bx = int(round(float(box.get("x", 0) or 0)))
    by = int(round(float(box.get("y", 0) or 0)))
    bw = max(1, int(round(float(box.get("w", 0) or 0))))
    bh = max(1, int(round(float(box.get("h", 0) or 0))))
    pad = int(rcfg.get("residual_glyph_ring_pad", 8))
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    region = rgb[y0:y1, x0:x1].astype(np.float32)
    bg = region.reshape(-1, 3)
    if ink_mask is not None:
        sub = np.asarray(ink_mask)[y0:y1, x0:x1]
        if sub.shape == region.shape[:2]:
            keep = sub == 0
            if int(np.count_nonzero(keep)) >= 30:
                bg = region[keep]
    if bg.shape[0] < 30:
        return False
    return float(np.max(np.std(bg, axis=0))) <= float(rcfg.get("residual_glyph_flat_std", 14.0))


# Fragment roles eligible for the flat-plate glyph keep: generic un-roled fragments plus
# the buckets residual-CC slices actually land in.  088's four residual raster-slice
# fragments (c_E000 / c_E002 / c_E006 / c_E007, the "SALE" letterforms) are ALL role
# "shape"; its remaining residual-provenance ownership cutouts are "icon" (c_E003 /
# c_E005 / c_E008) — inside the list — and "product" (c_E009) — deliberately OUT, since
# a photographic cutout must vacate the plate to swap cleanly.
_RESIDUAL_GLYPH_KEEP_ROLES = frozenset({"", "shape", "icon"})


def _skip_removal_for_flat_residual_glyph(c, candidate_mask, ownership, owner_number,
                                          rgb, rcfg):
    """Ownership fraction when a residual-CC baked glyph may keep its ink in the plate.

    Returns the fraction of the fragment's own mask that front-to-back ownership actually
    handed it when plate removal must be WITHHELD (under-covered ownership on a flat
    plate ring), ``None`` when the normal removal path applies.  A pure predicate — the
    caller stamps the meta and skips; the full reasoning comment lives at that call site
    in ``reconstruct``.  Role-scoped by ``_RESIDUAL_GLYPH_KEEP_ROLES`` and gated on
    ``reconstruct.keep_flat_residual_glyphs`` so photographic cutouts stay untouched.
    """
    _, np, _ = _deps()
    role = str((c.get("meta") or {}).get("role") or "").lower()
    if (c.get("target") != "image"
            or candidate_mask is None
            or role not in _RESIDUAL_GLYPH_KEEP_ROLES
            or not _residual_baked_raster(c.get("meta"))
            or not bool(rcfg.get("keep_flat_residual_glyphs", True))):
        return None
    owned_here = int(np.count_nonzero(
        (ownership == owner_number.get(c.get("id"), 0)) & (candidate_mask > 0)))
    mask_here = int(np.count_nonzero(candidate_mask))
    frac = owned_here / max(1, mask_here)
    if (frac < float(rcfg.get("residual_glyph_ownership_full", 0.995))
            and _plate_ring_is_flat(rgb, c.get("box") or {}, candidate_mask, rcfg)):
        return frac
    return None


def _stamp_residual_glyph_overlay(c: dict, ownership_fraction: float) -> None:
    """Mark a kept-in-plate residual glyph: original ink stays, the slice ships on top.

    ``overlay_without_removal`` is the vocabulary ``_keeps_underlay`` (and routing's
    preserve-underlay read) honour when later passes consider erasing an underlay;
    ``removal_skipped`` records WHY the removal ledger holds no pixels for this id — so a
    later raster-slice fallback drops to plate-passthrough instead of slicing on top.
    """
    meta = c.setdefault("meta", {})
    meta["removal_skipped"] = "flat-residual-glyph-kept-in-plate"
    meta["overlay_without_removal"] = True
    meta["residual_glyph_ownership_fraction"] = round(ownership_fraction, 4)


def _flatten_photo_scene(candidates: list, cfg: dict) -> tuple[list, int]:
    """Select independent foreground owners while retaining only scene fragments in the plate.

    This intentionally keeps the historical function name for callers, but no longer
    implements a flatten-first output.  A product, person, avatar, card, logo, badge,
    icon, or substantial photo frame is a separate Figma asset; only a full/edge-touching
    scenic photo and tiny unclassified detector debris remain in the background plate.
    """
    scene = cfg.get("scene") or {}
    photo_policy = (scene.get("preset") or {}).get("photo_regions") or {}
    out, flattened = [], 0
    separate_roles = {
        "product", "person", "foreground", "cutout", "avatar", "profile",
        "profile_photo", "card", "thumbnail", "photo_card", "logo", "brand",
        "wordmark", "platform-logo", "icon", "badge", "button", "chip", "callout_leader",
        "arrow", "leader", "leader_line", "connector",
        "text_backplate", "panel", "image-panel", "photo-panel", "triptych-panel",
        "comparison-panel", "comparison-column",
        # List-row chrome chips (066 ✓/✗). chrome_as_raster routes them to target=image,
        # so they must stay semantic here — otherwise _flatten drops them as
        # background-root-or-fragment and the checklist loses every icon.
        "verified", "checkmark", "check", "check-mark", "check_mark", "tick",
        "cross", "question-mark", "question_mark", "emoji", "symbol", "pictogram",
    } | set(INTENTIONAL_RASTER_CLUSTER_ROLES)
    min_photo_fraction = float(photo_policy.get("min_separate_photo_fraction", 0.012))
    max_photo_fraction = float(photo_policy.get("max_separate_photo_fraction", 0.62))
    min_shape_fraction = float(photo_policy.get("min_separate_shape_fraction", 0.001))
    canvas = cfg.get("canvas") or {}
    canvas_w = float(canvas.get("w") or 0)
    canvas_h = float(canvas.get("h") or 0)
    if canvas_w <= 1 or canvas_h <= 1:
        # ``reconstruct`` owns the true canvas but this compatibility helper is
        # also unit-tested directly. Infer a conservative canvas from candidates
        # when a caller did not supply it.
        boxes = [item.get("visible_box") or item.get("box") or {} for item in candidates]
        canvas_w = max([1.0] + [float(box.get("x") or 0) + float(box.get("w") or 0) for box in boxes])
        canvas_h = max([1.0] + [float(box.get("y") or 0) + float(box.get("h") or 0) for box in boxes])
    canvas_area = max(1.0, canvas_w * canvas_h)
    # Large photographic frames are the independent replacement unit for a
    # screenshot/card.  Their internal product pixels are intentionally part of
    # that one swappable image, not a collection of low-quality SAM cutouts.
    photo_roots = []
    for item in candidates:
        meta = item.get("meta") or {}
        box = item.get("visible_box") or item.get("box") or {}
        fraction = (float(box.get("w") or 0) * float(box.get("h") or 0)) / canvas_area
        role = str(meta.get("role") or "").lower()
        if (item.get("target") == "image" and (role in {"photo", "image"}
                                                  or is_intentional_raster_cluster(role))
                and fraction >= 0.15 and min(float(box.get("w") or 0), float(box.get("h") or 0)) >= 250):
            photo_roots.append(item)
    for candidate in candidates:
        c = dict(candidate)
        c["meta"] = dict(candidate.get("meta") or {})
        meta = c["meta"]
        text_value = str(c.get("text") or "")
        box = c.get("visible_box") or c.get("box") or {}
        confidence = float(meta.get("confidence") or c.get("confidence") or 0.0)
        compact_text = "".join(ch for ch in text_value if ch.isalnum())
        invalid_text = c.get("target") == "text" and (
            "\ufffd" in text_value or not any(ch.isalnum() for ch in text_value)
            or (len(compact_text) <= 2 and confidence < 0.75)
            or (float(box.get("h") or 0) > float(box.get("w") or 0) * 1.15)
        ) and not meta.get("emoji")
        if invalid_text:
            c["target"] = "drop"
            meta["keep_in_background"] = True
            meta["suppression_reason"] = "invalid-photo-scene-ocr"
            out.append(c)
            continue
        substitution = meta.get("substitution") if isinstance(meta.get("substitution"), dict) else {}
        exact_text_fallback = bool(
            meta.get("fallback") and substitution.get("from") == "text"
        )
        role = str(meta.get("role") or "").lower()
        target = str(c.get("target") or "")
        box = c.get("visible_box") or c.get("box") or {}
        area_fraction = (float(box.get("w") or 0) * float(box.get("h") or 0)) / canvas_area
        aspect = float(box.get("w") or 0) / max(1.0, float(box.get("h") or 0))
        has_explicit_mask = bool(_mask_path(c))
        x0, y0 = float(box.get("x") or 0), float(box.get("y") or 0)
        x1, y1 = x0 + float(box.get("w") or 0), y0 + float(box.get("h") or 0)
        edge_margin = max(3.0, min(canvas_w, canvas_h) * .012)
        edge_count = sum((x0 <= edge_margin, y0 <= edge_margin,
                          x1 >= canvas_w - edge_margin, y1 >= canvas_h - edge_margin))
        # Price strikes/underlines are already evidence-backed editable SVG paths.
        # They are deliberately thin and often contained by a broad residual host, so
        # the generic fragment/photo flattening rules below must never absorb them.
        if meta.get("native_decoration") and target == "shape":
            meta["layer_disposition"] = "foreground_vector"
            meta["z_band"] = "overlay"
            out.append(c)
            continue
        # A low-confidence residual-only photo fragment is texture/debris, not an
        # independently editable image.  Exporting it creates tiny alpha islands (and
        # sometimes holes) over a still-present scene plate.  Keep it in the plate until
        # SAM supplies a real semantic matte.  This is intentionally limited to full-size
        # ads; small unit fixtures and explicitly verified/promoted assets remain intact.
        residual_only = not _verified_semantic_mask(meta)
        if (min(canvas_w, canvas_h) >= 400 and target == "image"
                and role in {"photo", "image", "photo-fragment"}
                and confidence < .45 and residual_only
                and not (meta.get("promote_element") or meta.get("editable_element"))):
            c["target"] = "drop"
            meta["keep_in_background"] = True
            meta["suppression_reason"] = "low-confidence-residual-photo-fragment"
            out.append(c)
            continue
        # A huge person/photo joined to multiple canvas edges is part of the
        # continuous lifestyle scene.  Cutting it out creates an enormous Flux
        # hole and a less-editable result than retaining the scene root.
        if (min(canvas_w, canvas_h) >= 400 and role in {"photo", "image", "person"}
                and area_fraction >= .40 and edge_count >= 2):
            c["target"] = "drop"
            meta["keep_in_background"] = True
            meta["background_root"] = True
            meta["suppression_reason"] = "edge-touching-continuous-scene"
            out.append(c)
            continue
        containing_frame = next((root for root in photo_roots
                                 if root.get("id") != c.get("id")
                                 and _box_containment(box, root.get("visible_box") or root.get("box") or {}) >= .96), None)
        # A continuous edge-to-edge scenic plate is deliberately *not* a
        # swappable frame.  Its contained product/logo detections must remain
        # independent layers; otherwise the scene root is dropped and it takes
        # every meaningful foreground object with it (notably tall product ads).
        root_is_background_scene = False
        if containing_frame:
            root_box = containing_frame.get("visible_box") or containing_frame.get("box") or {}
            rx0, ry0 = float(root_box.get("x") or 0), float(root_box.get("y") or 0)
            rx1 = rx0 + float(root_box.get("w") or 0)
            ry1 = ry0 + float(root_box.get("h") or 0)
            root_fraction = (float(root_box.get("w") or 0) * float(root_box.get("h") or 0)) / canvas_area
            root_edges = sum((rx0 <= edge_margin, ry0 <= edge_margin,
                              rx1 >= canvas_w - edge_margin, ry1 >= canvas_h - edge_margin))
            root_role = str((containing_frame.get("meta") or {}).get("role") or "").lower()
            root_is_background_scene = (min(canvas_w, canvas_h) >= 400
                                        and root_role in {"photo", "image", "person"}
                                        and root_fraction >= .40 and root_edges >= 2)
        # Chrome that merge explicitly marked extractable is NOT "internal detail of a
        # swappable photo": merge_layers (see its comment at the raster-cluster owner loop,
        # "009 verified check, engagement glyphs") sets chrome_as_raster/extract_from_cluster
        # precisely to say "keep this as its own IMAGE cutout". reconstruct read neither flag
        # and dropped every such icon here because role "icon" is not in the avatar/badge/
        # button allowlist — baking them and dropping the layer punches a hole the editable
        # text then paints over (009 lost its verified badge, timer, eyes, back arrow, avatar
        # and all five action icons). Honour the upstream contract; raster-first icons depend
        # on it.
        if (containing_frame and not root_is_background_scene
                and target not in {"text", "drop"} and role not in {"avatar", "badge", "button"}
                and not (meta.get("chrome_as_raster") or meta.get("extract_from_cluster"))):
            c["target"] = "drop"
            meta["kept_in_owner"] = containing_frame.get("id")
            meta["suppression_reason"] = "contained-in-swappable-photo-frame"
            out.append(c)
            continue
        promoted = bool(meta.get("promote_element") or meta.get("editable_element")
                        or meta.get("verified_mask") or exact_text_fallback)
        is_semantic = (
            role in separate_roles
            or is_intentional_raster_cluster(role)
            or target == "icon"
            # chrome_as_raster marks chips with icon_chip even when target is image
            or bool(meta.get("icon_chip"))
        )
        is_photo_frame = role in {"photo", "image"} and (
            not box or min_photo_fraction <= area_fraction <= max_photo_fraction
        )
        # Generic SAM ``shape`` detections are frequently edge specks (041 produced
        # dozens).  Preserve only semantically named UI/chrome shapes; meaningful
        # unlabelled marks should arrive as an icon/vector candidate instead.
        is_meaningful_shape = target == "shape" and (
            role in separate_roles or (has_explicit_mask and area_fraction >= .005)
            or (bool(meta.get("simple_graphic")) and area_fraction >= min_shape_fraction)
        )
        # A very thin unparented SAM shape is frequently only the rim/highlight
        # of a badge or button.  Rebuilding that fragment creates visible clipped
        # arcs; retain it in the plate unless it is part of an explicitly-owned
        # control group.
        if (target == "shape" and not meta.get("parent_id") and role in {"", "shape"}
                and aspect >= 6 and area_fraction < .01):
            c["target"] = "drop"
            meta["keep_in_background"] = True
            meta["suppression_reason"] = "thin-unparented-ui-fragment"
            out.append(c)
            continue
        if target not in {"text", "drop"} and (promoted or is_semantic or is_photo_frame or is_meaningful_shape):
            # A substantial rectangular photo is an image frame/card, not an
            # irregular cutout.  Use a native rounded-rectangle mask and a full
            # crop so small SAM holes cannot create black seams around a swappable
            # image.  Small/tall photo detections remain alpha cutouts.
            if (target == "image" and role in {"photo", "image"} and area_fraction >= 0.15
                    and min(float(box.get("w") or 0), float(box.get("h") or 0)) >= 250):
                c["mask"] = {"kind": "rrect"}
                meta["photo_frame"] = True
            meta["layer_disposition"] = (
                "foreground_vector" if target == "icon" else
                "native_shape" if target == "shape" else "foreground_raster"
            )
            # A VLM segment review may have supplied a stronger semantic z-band
            # (for example, a rasterized UI header is still chrome even though its
            # routed target is ``image``).  Never overwrite that contract while
            # assigning the deterministic fallback for unclassified detections.
            meta["z_band"] = meta.get("z_band") or (
                "chrome" if target in {"icon", "shape"} else "content"
            )
            out.append(c)
            continue
        if target not in {"text", "drop"}:
            c["target"] = "drop"
            meta["keep_in_background"] = True
            meta["raster_fallback"] = "background-root-or-fragment"
            flattened += 1
        out.append(c)
    return out, flattened


def _mask_path(candidate):
    mask = candidate.get("mask")
    if isinstance(mask, dict):
        return mask.get("src")
    if isinstance(mask, str):
        return mask
    return candidate.get("mask_path")


def _candidate_mask(candidate, rgb, run_dir, ocr_lines=None, cfg: Optional[dict] = None):
    cv2, np, _ = _deps()
    h, w = rgb.shape[:2]
    meta = candidate.get("meta") or {}
    rcfg = (cfg or {}).get("reconstruct") or {}
    if meta.get("native_decoration") and isinstance(meta.get("line"), dict):
        line = meta["line"]
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            p0 = (int(round(float(line["x0"]))), int(round(float(line["y0"]))))
            p1 = (int(round(float(line["x1"]))), int(round(float(line["y1"]))))
            thickness = max(1, int(math.ceil(float(line.get("thickness", 2.0)))))
            cv2.line(mask, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_AA)
            return mask
        except (KeyError, TypeError, ValueError):
            return mask
    substitution = meta.get("substitution") if isinstance(meta.get("substitution"), dict) else {}
    exact_text_fallback = bool(
        candidate.get("text") and meta.get("fallback") and substitution.get("from") == "text"
    )
    removal_text = bool(candidate.get("text") and meta.get("removal_required"))
    if (candidate.get("target") == "text" or exact_text_fallback
            or removal_text or (candidate.get("text") and meta.get("wordmark"))):
        # line_ids are provenance only: merge/reordering can leave them stale or point
        # at an unrelated OCR line. The merged candidate geometry is canonical.
        box = candidate.get("visible_box") or candidate.get("ink_box") or candidate.get("box", {})
        allow_box_fallback = bool(
            rcfg.get("allow_text_box_fallback", False)
            or meta.get("force_box_removal")
        ) and not bool(meta.get("overlay_text"))
        mask = inpaint.text_ink_mask(
            rgb,
            box,
            candidate.get("quad") or meta.get("quad"),
            allow_box_fallback=allow_box_fallback,
        )
        # A crop that is itself a flat, opaque overlay (rather than sparse glyphs on the
        # plate) legitimately owns its complete rectangle. Detect that from source pixels
        # and the immediate outside ring instead of using an unconditional box fallback.
        if (not np.any(mask)
                and bool(rcfg.get("allow_opaque_text_region_fallback", True))):
            exact = inpaint.box_fill_mask((h, w), box, pad=0)
            ring = (inpaint.box_fill_mask((h, w), box, pad=3) > 0) & (exact == 0)
            inside_pixels = rgb[exact > 0].astype(np.float32)
            outside_pixels = rgb[ring].astype(np.float32)
            if inside_pixels.size and outside_pixels.size:
                flatness = float(np.max(np.std(inside_pixels, axis=0)))
                contrast = float(np.linalg.norm(
                    np.median(inside_pixels, axis=0) - np.median(outside_pixels, axis=0)
                ))
                if flatness <= 8.0 and contrast >= 24.0:
                    mask = exact
        if candidate.get("target") == "text":
            solid = inpaint.box_fill_mask((h, w), box, pad=5)
            coverage = float(np.count_nonzero(mask & solid)) / max(1, np.count_nonzero(solid))
            # Most normal text is sparse ink, so "coverage" is usually low. Unioning the
            # full box for large paragraphs creates huge removal rectangles and damages
            # plates (notably photo ads like 041). Only promote to the full box when the
            # box is small enough that this can't become a destructive slab.
            solid_fraction = float(np.count_nonzero(solid)) / max(1, h * w)
            promote_limit = float(rcfg.get("text_box_promote_max_fraction", 0.0))
            if (promote_limit > 0 and coverage < 0.92
                    and solid_fraction <= promote_limit):
                mask = np.maximum(mask, solid)
        return mask
    # A recognised screenshot/receipt/chart/table/diagram/product cluster is defined by
    # its complete rectangular crop, not a sparse SAM matte. This keeps every internal
    # pixel (including UI chrome and overlaps) inside the swappable source asset.
    if is_intentional_raster_cluster(meta.get("role")) or meta.get("intentional_raster_cluster"):
        return inpaint.box_fill_mask((h, w), candidate.get("box") or {}, pad=0)
    mask = inpaint.mask_on_canvas(_mask_path(candidate), candidate.get("box", {}), (w, h), run_dir)
    if candidate.get("target") in ("shape", "icon", "image"):
        mask = inpaint.solidify_mask(mask)
    # For tangible foreground subjects an enclosed transparent island is almost
    # always an accidental segmentation void, not a design feature.  Fill it
    # before ownership/inpainting so the product stays a clean swappable asset.
    role = str(meta.get("role") or "").lower()
    if candidate.get("target") == "image" and role in {"product", "person", "people", "portrait", "cutout"}:
        mask = inpaint.fill_enclosed_mask_holes(mask)
    return mask


def _is_text_removal(item):
    """Text removals: emitted editable text and drop-observations that erase text."""
    return item.get("target") == "text" or (
        item.get("target") == "drop" and (item.get("meta") or {}).get("removal_required")
    )


def _strike_ink_regions(ocr):
    """Strikethrough decoration boxes OCR flagged (091: red strike scribble on a headline).

    The scribble is FOREIGN ink — a different colour than the glyph and extending past the
    glyph box (a hand-drawn swipe). The per-line glyph-ink residue check keys on the glyph
    colour, so leftover strike ink is invisible to it and ships as a squiggle fragment.
    When the struck line is promoted to native TEXT the whole scribble must leave the plate;
    the residual audit consumes these regions to hunt that leftover decoration ink."""
    regions = []
    for line in (ocr or {}).get("lines") or []:
        meta = line.get("meta") or {}
        sbox = meta.get("strikethrough_box")
        if not (meta.get("strikethrough") and isinstance(sbox, dict)):
            continue
        if not (float(sbox.get("w", 0) or 0) > 0 and float(sbox.get("h", 0) or 0) > 0):
            continue
        regions.append({
            "box": {k: float(sbox.get(k, 0) or 0) for k in ("x", "y", "w", "h")},
            "line_box": line.get("box") or {},
            "line_id": line.get("id"),
            "kind": "strikethrough",
        })
    return regions


def _ensure_text_removal_coverage(candidate, mask, rgb, cfg):
    """Every EMITTED text layer must have its source pixels in the removal mask.

    Ghost/duplicate text (009 timestamp row) happens exactly when the ink mask misses
    low-contrast or tiny glyphs: the original stays in the plate AND the editable text
    is re-rendered on top.  When the estimated ink under-covers the candidate box,
    force the OCR polygon (or, failing that, the box) so removal is guaranteed.
    Display / edge-touching copy also gets a small coverage dilate so antialias fringe
    and cramped headline stems (002 ESSENTIALS, 131 ENDS TODAY) cannot stay in the plate
    while the sparse ink count still clears the area floor. Config-gated, default ON
    (``reconstruct.force_text_removal_coverage``).
    """
    cv2, np, _ = _deps()
    rcfg = (cfg or {}).get("reconstruct") or {}
    if not bool(rcfg.get("force_text_removal_coverage", True)):
        return mask
    if _keeps_underlay(candidate):
        return mask
    box = candidate.get("visible_box") or candidate.get("ink_box") or candidate.get("box", {})
    height, width = rgb.shape[:2]
    area = max(1.0, float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0))
    required = max(12, int(area * float(rcfg.get("min_text_mask_area_fraction", 0.015))))
    covered = int(np.count_nonzero(mask))
    meta = candidate.setdefault("meta", {})
    bx = int(round(float(box.get("x", 0) or 0)))
    by = int(round(float(box.get("y", 0) or 0)))
    bw = max(1, int(round(float(box.get("w", 0) or 0))))
    bh = max(1, int(round(float(box.get("h", 0) or 0))))
    edge_pad = int(rcfg.get("text_coverage_edge_pad", 4))
    touches_edge = (
        bx <= edge_pad or by <= edge_pad
        or (bx + bw) >= (width - edge_pad) or (by + bh) >= (height - edge_pad)
    )
    display = bh > float(rcfg.get("display_text_coverage_min_h", 48))
    under = covered < required
    if under:
        forced = inpaint.text_ink_mask(
            rgb, box, candidate.get("quad") or meta.get("quad"), allow_box_fallback=True,
        )
        if int(np.count_nonzero(forced)) < required:
            # Edge / cramped headlines: pad the box so fringe outside the OCR box is
            # claimed (131 top banner, 025 overlay gutters).
            pad = 2 if (display or touches_edge) else 1
            forced = np.maximum(forced, inpaint.box_fill_mask((height, width), box, pad=pad))
        merged = np.maximum(np.asarray(mask, dtype=np.uint8), np.asarray(forced, dtype=np.uint8))
        meta["removal_coverage_forced"] = {
            "reason": "text-ink-mask-under-coverage",
            "mask_px_before": covered,
            "mask_px_after": int(np.count_nonzero(merged)),
            "required_px": required,
        }
    else:
        merged = np.asarray(mask, dtype=np.uint8)
    # Sparse ink can clear ``required`` yet miss stem/antialias fringe. Dilate coverage
    # for display + edge-touching copy so removal owns the readable halo.
    cov_dilate = int(rcfg.get("text_coverage_dilate", 0))
    if cov_dilate <= 0:
        if display:
            cov_dilate = int(rcfg.get("display_text_coverage_dilate", 3))
        elif touches_edge:
            cov_dilate = int(rcfg.get("edge_text_coverage_dilate", 2))
        else:
            cov_dilate = 0
    if cov_dilate > 0 and np.any(merged):
        kernel = np.ones((2 * cov_dilate + 1, 2 * cov_dilate + 1), np.uint8)
        dilated = cv2.dilate(merged, kernel)
        if int(np.count_nonzero(dilated)) > int(np.count_nonzero(merged)):
            meta["removal_coverage_dilated"] = {
                "px": cov_dilate, "display": bool(display), "edge": bool(touches_edge),
            }
            merged = dilated
    merged = _claim_clipped_glyph_edges(candidate, merged, rgb, cfg)
    return merged


def _claim_clipped_glyph_edges(candidate, mask, rgb, cfg):
    """Claim glyph ink that CONTINUES just outside the OCR box.

    ``inpaint.text_ink_mask`` constrains ink to the authored OCR box, and OCR boxes
    routinely clip a glyph's last few pixels: 107's '58%' box ends at x=737 while the '%'
    bowl runs to x=751, and 'Of Athletes Are Chronically' ends at x=939 with the 'y' tail
    at x=943.  Those slivers are never claimed, so they stay baked in the plate and render
    as stray dots beside the re-rendered native text.  The defect reads as a speck; the
    cause is a mask that stops at an arbitrary rectangle instead of at the end of the glyph.

    Grow into source ink that is 8-CONNECTED to already-claimed ink and lies within
    ``pad`` px of the box.  Connectivity is the safety property: a neighbouring icon or
    word that merely sits near the box is untouched unless its pixels physically continue
    the glyph already being removed.  A bounded extra-area guard fails closed if the local
    contrast estimate ever selects a plate region rather than a glyph edge.  Config:
    ``reconstruct.text_clipped_edge_pad`` (0 disables).
    """
    cv2, np, _ = _deps()
    rcfg = (cfg or {}).get("reconstruct") or {}
    base_pad = int(rcfg.get("text_clipped_edge_pad", 10))
    base = np.asarray(mask, dtype=np.uint8)
    if base_pad <= 0 or not base.any():
        return mask
    box = candidate.get("visible_box") or candidate.get("ink_box") or candidate.get("box", {})
    height, width = rgb.shape[:2]
    bx = float(box.get("x", 0) or 0)
    by = float(box.get("y", 0) or 0)
    bw = float(box.get("w", 0) or 0)
    bh = float(box.get("h", 0) or 0)
    if bw <= 0 or bh <= 0:
        return mask
    # How far OCR clips a glyph scales with the glyph: 107's 139px-tall '58%' loses 14px
    # of the '%' bowl, which a fixed small pad cannot reach. Keep the pad proportional so
    # display copy is covered without over-reaching around body copy.
    pad = max(base_pad, min(int(rcfg.get("text_clipped_edge_max_pad", 24)),
                            int(round(float(rcfg.get("text_clipped_edge_h_frac", 0.12)) * bh))))
    x0 = max(0, int(round(bx)) - pad)
    y0 = max(0, int(round(by)) - pad)
    x1 = min(width, int(round(bx + bw)) + pad)
    y1 = min(height, int(round(by + bh)) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return mask
    try:
        ink = np.asarray(inpaint.text_ink_mask(
            rgb, {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            allow_box_fallback=False,
        ), dtype=np.uint8)
    except Exception:
        return mask
    if not ink.any():
        return mask
    seed = base > 0
    # Connectivity alone is NOT enough: a strike-through scribble physically touches the
    # glyphs it crosses (091), and swallowing it into this text's removal would erase a
    # decoration that is its own element. A clipped edge is literally the same glyph, so
    # also require the SAME ink colour — that keeps the '%' bowl and rejects the red swipe.
    claimed_ink = seed & (ink > 0)
    if int(np.count_nonzero(claimed_ink)) < 8:
        return mask
    # Prefer the node's DECLARED colour. Sampling the mask is unreliable exactly where it
    # matters: a strike crossing the glyphs contaminates the in-box sample with its own
    # colour, and a median over that mix would then admit the scribble it must reject.
    style_color = ((candidate.get("style") or {}).get("color"))
    if style_color:
        ink_color = np.asarray(_hex_to_rgb(style_color), dtype=np.float32)
    else:
        ink_color = np.median(rgb[claimed_ink].astype(np.float32), axis=0)
    colour_tol = float(rcfg.get("text_clipped_edge_colour_tol", 40.0))
    colour_ok = np.abs(rgb.astype(np.float32) - ink_color).mean(axis=2) <= colour_tol
    combined = (((ink > 0) & colour_ok) | seed).astype(np.uint8)
    count, labels = cv2.connectedComponents(combined, connectivity=8)
    if count <= 1:
        return mask
    keep = np.unique(labels[seed])
    keep = keep[keep > 0]
    if keep.size == 0:
        return mask
    grown = np.isin(labels, keep) & (ink > 0) & colour_ok & (~seed)
    extra = int(np.count_nonzero(grown))
    if extra <= 0:
        return mask
    # Fail closed on a runaway claim: a real clipped edge is a sliver, not a region.
    max_extra = max(96, int(float(rcfg.get("text_clipped_edge_max_frac", 0.35))
                            * int(np.count_nonzero(seed))))
    if extra > max_extra:
        return mask
    # The Otsu ink estimate stops at the glyph core; its antialias fringe is what would
    # still read as a grey speck. Grow the claimed sliver (only) by a hair to take it.
    fringe = int(rcfg.get("text_clipped_edge_fringe", 2))
    claimed = (grown * 255).astype(np.uint8)
    if fringe > 0:
        claimed = cv2.dilate(claimed, np.ones((2 * fringe + 1, 2 * fringe + 1), np.uint8))
    merged = np.maximum(base, claimed)
    meta = candidate.setdefault("meta", {})
    meta["removal_clipped_edges_claimed"] = {"px": extra, "pad": pad}
    return merged


def _post_inpaint_text_residual(rgb, background_path, records, ink_masks, union,
                                ledger, cfg, decoration_regions=None):
    """Ghost-text guard: detect glyph residue left on the clean plate under removed text.

    A glyph-ink plate pixel is leftover ghost when it either (a) still matches the
    SOURCE (|plate - source| <= tolerance — the inpaint skipped it), OR (b) still sits
    near the glyph's OWN colour (|plate - ink_colour| <= ink_tolerance — a *smeared*
    inpaint that blurred the value out of source-match range but is visibly still inky).
    The old audit used only (a); a smear (067 "WE'RE SAYING GOODBYE" stayed red, 021
    sticky-note copy) dropped the source-match count and falsely read as "resolved".

    Flagged regions expand the removal mask/ledger in place (single-union contract) and
    get up to ``reinpaint_max_passes`` targeted text-backend re-inpaint passes with a
    growing dilation, so thin residue that only needs cleaner context resolves.
    "Resolved" is an ABSOLUTE bar (remaining px / ratio), not a fraction of the original
    residue — a catastrophic ghost can no longer pass just because it shrank 30%.  Any
    EMITTED native-text layer still ghosting after the passes is cleaned with a
    deterministic local plate-colour fill (Codia: keep native TEXT; never bake OCR into
    a raster slice for SSIM). ``force_raster_ids`` is opt-in only
    (``reconstruct.text_residual.force_raster``, default OFF). This module only *calls*
    inpaint; it does not modify the backends. Config-gated, default ON
    (``reconstruct.text_residual``).
    """
    cv2, np, Image = _deps()
    rcfg = (cfg or {}).get("reconstruct") or {}
    audit_cfg = rcfg.get("text_residual") if isinstance(rcfg.get("text_residual"), dict) else {}
    if not bool(audit_cfg.get("enabled", True)):
        return {"enabled": False, "checked": 0, "flagged": []}
    text_records = [item for item in records if _is_text_removal(item)]
    if not text_records or not os.path.exists(background_path):
        return {"enabled": True, "checked": 0, "flagged": []}
    plate = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.int16)
    if plate.shape != rgb.shape:
        return {"enabled": True, "checked": 0, "flagged": [], "note": "plate-shape-mismatch"}
    source = rgb.astype(np.int16)
    # Reused across residue checks / solid-fill (invalidated after reinpaint rewrites disk).
    plate_f32 = None
    plate_u8 = None
    tolerance = float(audit_cfg.get("tolerance", 12.0))
    ink_tolerance = float(audit_cfg.get("ink_tolerance", 40.0))
    min_px = int(audit_cfg.get("min_px", 24))
    min_ratio = float(audit_cfg.get("min_ratio", 0.30))
    grow = max(1, int(audit_cfg.get("reinpaint_dilate", 3)))
    resolved_ratio = float(audit_cfg.get("resolved_ratio", 0.15))
    # Residue below the raster-slice minimum region size is considered resolved: it is
    # both visually negligible and too small for the looks-right floor to even act on
    # (schema.RASTER_SLICE_FALLBACK_DEFAULTS["min_region_px"] == 120). A genuine ghost
    # (009 c_B1 ~1729px, 067, 101 c_B0 ~5146px) sits far above this and still fails.
    resolved_abs_px = int(audit_cfg.get("resolved_abs_px", 120))
    # Default OFF — Codia: never bake readable OCR into a raster slice for fidelity.
    force_raster = bool(audit_cfg.get("force_raster", False))
    solid_fill = bool(audit_cfg.get("solid_fill_residue", True))
    ring_radius = max(4, int(audit_cfg.get("solid_fill_ring", 10)))
    # Flat/UI plates: skip wasted generative reinpaint and go straight to local plate
    # fill (009 dark chrome, caption plates). Photo archetypes keep a short reinpaint
    # ladder first. When solid fill is on, default to 1 reinpaint pass (not 3) — the
    # fill is the real closer and extra passes mostly burn time on smears.
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()
    flat_residual_archetypes = frozenset({
        "social_screenshot", "caption_over_photo", "comparison_grid", "product_on_flat",
    }) | {
        str(a).lower() for a in (audit_cfg.get("solid_fill_first_archetypes") or [])
    }
    from src import format_readiness
    fmt = format_readiness.format_from_cfg(cfg)
    capability_flat = bool(fmt.get("capabilities")) and format_readiness.prefers_solid_flat(cfg)
    default_solid_first = capability_flat or archetype in flat_residual_archetypes
    solid_first = bool(audit_cfg.get("solid_fill_first", default_solid_first))
    if "solid_fill_first" in audit_cfg and audit_cfg.get("solid_fill_first") is None:
        solid_first = default_solid_first
    default_passes = 0 if solid_first else (1 if solid_fill else 3)
    max_passes = max(0, int(audit_cfg.get("reinpaint_max_passes", default_passes)))
    # Crowded / edge glyphs (002 ESSENTIALS, 025, 066, 131): wipe a halo around the
    # glyph footprint so LaMa/Telea smear bleed outside residual ink clears too.
    fill_dilate = max(0, int(audit_cfg.get(
        "solid_fill_dilate", 5 if solid_first else 3,
    )))
    min_ring_samples = max(4, int(audit_cfg.get("solid_fill_min_ring", 8)))
    ink_away = max(ink_tolerance, float(audit_cfg.get("solid_fill_ink_away", 32.0)))
    # Foreign decoration ink (strike scribble) is a strong mark against the plate colour.
    deco_mark_tol = float(audit_cfg.get("decoration_mark_tolerance", 48.0))
    deco_min_px = int(audit_cfg.get("decoration_min_px", 32))
    # Third ghost class: a *partial* inpaint smear (see _smear_residue).
    smear_enabled = bool(audit_cfg.get("smear_enabled", True))
    smear_max_plate_hf = float(audit_cfg.get("smear_max_plate_hf", 3.0))
    smear_min_contrast = float(audit_cfg.get("smear_min_contrast", 25.0))
    smear_retain_frac = float(audit_cfg.get("smear_retain_frac", 0.30))
    smear_min_ratio = float(audit_cfg.get("smear_min_ratio", 0.18))
    smear_ring = max(3, int(audit_cfg.get("smear_ring", 13)) | 1)
    smear_window = max(5, int(audit_cfg.get("smear_window", 21)) | 1)
    # Gradient-tolerant smear path (025 'Blocks everything' row): when the ring is NOT
    # smooth — a soft card gradient, vignette or panel wash — the flat local-mean model
    # is invalid and the detector used to fail closed, shipping baked ink UNDER the
    # emitted native TEXT (ghost double).  Fit the plate per-pixel instead and score
    # surviving glyph contrast against that model.  Floors sit slightly looser than the
    # flat bars (a gradient fit is noisier than a local mean) but stay conservative:
    # genuine photographic texture fails the ring-fit gate and is left to the absolute
    # tests — a false fill over a real photo is worse than a miss.
    smear_grad_min_contrast = float(audit_cfg.get("smear_grad_min_contrast", 18.0))
    smear_grad_min_ratio = float(audit_cfg.get("smear_grad_min_ratio", 0.14))
    smear_grad_max_ring_residual = float(
        audit_cfg.get("smear_grad_max_ring_residual", 8.0))
    smear_gradient_fill = bool(audit_cfg.get("smear_gradient_fill", True))
    smear_fill_window = max(5, int(audit_cfg.get("smear_fill_window", 41)) | 1)
    smear_fill_min_samples = float(audit_cfg.get("smear_fill_min_samples", 8))

    def _residue(plate_arr, ink, ink_color):
        nonlocal plate_f32
        d_src = np.abs(plate_arr - source).mean(axis=2)
        res = ink & (d_src <= tolerance)
        if ink_color is not None:
            if plate_f32 is None or plate_f32.shape != plate_arr.shape:
                plate_f32 = plate_arr.astype(np.float32)
            d_ink = np.abs(plate_f32 - ink_color).mean(axis=2)
            res = res | (ink & (d_ink <= ink_tolerance))
        return res

    def _luma(arr):
        a = np.asarray(arr, dtype=np.float32)
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114

    def _smear_residue(plate_arr, ink):
        """Detect a *partial* inpaint smear: the ghost class (a)/(b) structurally cannot see.

        (a) and (b) are ABSOLUTE colour tests — "plate still equals source" and "plate
        still equals saturated ink".  A generative fill over a translucent/glass card
        leaves a blurred, dimmed glyph that matches NEITHER yet is plainly readable to
        the user (025 'Cuts you off' measured |plate-source| ~93 and |plate-ink| ~111
        against bars of 12 / 40, so it scored 0.09 residue and shipped doubled under the
        native TEXT node).  Detect it structurally instead: how much of each glyph
        pixel's ORIGINAL local contrast still survives on the plate.

        The local background is estimated from NON-ink pixels only, so a backdrop shared
        by source and plate cancels and only glyph-shaped energy is measured.  Two plate
        regimes, one verdict shape — ``(mask, ratio, gate, variant)`` where ``gate`` is
        the flag bar the caller must apply for that variant:

          * FLAT (ring is locally smooth on BOTH plate and source): the classic
            local-mean background below — unchanged fast path, same result as before.
          * GRADIENT (ring is not smooth): a soft gradient/vignette/panel wash used to
            fail closed here and ship 025's 'Blocks everything' row doubled; hand off to
            ``_gradient_smear_residue``, which fits the plate per-pixel.  GENUINE fine
            texture (025's hair headline, whose plate is genuinely clean) still fails
            closed inside that variant: there the smooth model cannot even explain the
            ring, the estimate is noise, and the plate-colour fill would be wrong anyway,
            so a photographic backdrop is left to the absolute tests.

        Only ever called for text-removal rows post-inpaint, so kept photo fragments
        (a different ownership path) never reach it.  Abstentions return ``None``.
        """
        ring_k = np.ones((smear_ring, smear_ring), np.uint8)
        ring = (cv2.dilate(ink.astype(np.uint8), ring_k) > 0) & (~ink)
        if int(np.count_nonzero(ring)) < 20:
            return None, 0.0, smear_min_ratio, "flat"
        plate_l = _luma(plate_arr)
        src_l = _luma(source)
        hf = np.abs(plate_l - cv2.GaussianBlur(plate_l, (5, 5), 0))
        hf_src = np.abs(src_l - cv2.GaussianBlur(src_l, (5, 5), 0))
        # The smoothness precondition must hold on BOTH sides of the comparison: a
        # generous inpaint can flatten the plate halo to a single colour (smooth plate
        # ring) while the SOURCE backdrop is genuine fine texture (photo grain, hair,
        # fabric).  There the flat local-mean "background" of the source is just noise
        # and any retained-contrast verdict — including a true positive — is untrusted:
        # fail over to the per-pixel variant, whose source-ring fit gate abstains.
        if (float(np.median(hf[ring])) > smear_max_plate_hf
                or float(np.median(hf_src[ring])) > smear_max_plate_hf):
            # Not a smooth plate — that is no longer an automatic fail-closed.  A soft
            # gradient is exactly what the per-pixel fill estimator models; let the
            # gradient-tolerant variant decide (it keeps real photos out itself).
            return _gradient_smear_residue(plate_arr, ink, ring, plate_l)
        keep = (~ink).astype(np.float32)
        win = (smear_window, smear_window)
        den = cv2.boxFilter(keep, -1, win, normalize=False)
        safe_den = np.maximum(den, 1e-6)

        def _bg(channel):
            return cv2.boxFilter(channel * keep, -1, win, normalize=False) / safe_den

        src_l = _luma(source)
        c_src = src_l - _bg(src_l)
        c_plate = plate_l - _bg(plate_l)
        strong = ink & (den > 4) & (np.abs(c_src) > smear_min_contrast)
        strong_px = int(np.count_nonzero(strong))
        if strong_px < max(8, min_px):
            return None, 0.0, smear_min_ratio, "flat"
        retained = np.zeros_like(c_src)
        np.divide(c_plate, c_src, out=retained, where=np.abs(c_src) > 1e-6)
        hit = strong & (retained > smear_retain_frac)
        return hit, int(np.count_nonzero(hit)) / max(1, strong_px), smear_min_ratio, "flat"

    def _gradient_plate_fill(plate_u8_arr, fill_area):
        """Per-pixel plate estimate for a smear fill, sampled from OUTSIDE the fill.

        The scalar ring median is right for flat UI chrome, but 025's cards are a soft
        gradient behind a vignette: one median colour over a whole row paints a visible
        tinted patch (it reads as a highlighter bar behind the copy). Estimate the plate
        per pixel from surrounding non-fill pixels so the fill follows the gradient.

        This is only offered to the smear class, whose detector already REQUIRES a locally
        smooth plate — the precondition that makes the detection valid is exactly the one
        that makes this interpolation valid. Returns None when the neighbourhood is too
        crowded to sample honestly, so the caller falls back to the scalar median.
        """
        keep = (~fill_area).astype(np.float32)
        win = (smear_fill_window, smear_fill_window)
        den = cv2.boxFilter(keep, -1, win, normalize=False)
        if float(np.mean(den[fill_area] > smear_fill_min_samples)) < 0.98:
            return None  # not enough clean plate around the glyphs to interpolate
        safe = np.maximum(den, 1e-6)
        out = np.empty(plate_u8_arr.shape, dtype=np.float32)
        src = plate_u8_arr.astype(np.float32)
        for ch in range(3):
            out[..., ch] = cv2.boxFilter(src[..., ch] * keep, -1, win, normalize=False) / safe
        return np.clip(out, 0, 255).astype(np.uint8)

    def _gradient_smear_residue(plate_arr, ink, ring, plate_l):
        """Smear verdict for a NON-smooth plate (gradient / vignette / panel wash).

        025's 'Blocks everything' row sits on a soft card gradient over a vignette: the
        ring's high-freq energy tops ``smear_max_plate_hf``, so the flat path used to
        fail closed and the dimmed smear shipped UNDER the emitted native TEXT (ghost
        double).  The same per-pixel estimator the smear FILL trusts
        (``_gradient_plate_fill``) is a valid background model here — the precondition
        that makes a gradient fill honest is exactly the one that makes scoring against
        it honest — so fit the plate (and the source) per-pixel and measure how much of
        each strong glyph pixel's original local contrast survives against the model.

        Conservatisms keep genuine photos out (a false fill over real texture is worse
        than the miss, and this audit only ever sees text-removal rows post-inpaint):
          * the RING must be explained by the smooth model ON BOTH SIDES — if the
            plate's own ring pixels (or the source's: a generous inpaint can flatten
            the plate halo while the source backdrop stays photographic) deviate from
            their fits by more than ``smear_grad_max_ring_residual`` the backdrop is
            real fine texture, not a gradient, and we abstain;
          * the contrast/ratio floors stay close to the flat path's bars
            (``smear_grad_min_contrast`` / ``smear_grad_min_ratio``), only slightly
            looser because the fit is noisier than a local mean.

        Returns the same ``(mask, ratio, gate, variant)`` shape as the flat path.
        """
        plate_u8_local = np.clip(plate_arr, 0, 255).astype(np.uint8)
        model_plate = _gradient_plate_fill(plate_u8_local, ink)
        if model_plate is None:
            # Too crowded to sample honestly — fail closed exactly as the flat gate did.
            return None, 0.0, smear_grad_min_ratio, "gradient"
        fit = np.abs(plate_l - _luma(model_plate))
        if float(np.median(fit[ring])) > smear_grad_max_ring_residual:
            # The smooth model cannot even explain the plate AROUND the glyphs: genuine
            # photographic texture (hair, fabric, grain), not a gradient.  Leave it to
            # the absolute tests — a plate-colour fill would paint a slab over a photo.
            return None, 0.0, smear_grad_min_ratio, "gradient"
        src_u8 = np.clip(source, 0, 255).astype(np.uint8)
        model_src = _gradient_plate_fill(src_u8, ink)
        if model_src is None:
            return None, 0.0, smear_grad_min_ratio, "gradient"
        src_l = _luma(source)
        fit_src = np.abs(src_l - _luma(model_src))
        if float(np.median(fit_src[ring])) > smear_grad_max_ring_residual:
            # The SOURCE ring is genuine fine texture, not a gradient: the contrast
            # baseline itself is noise (a generous inpaint can leave the plate ring
            # smooth — halo flattened to one colour — while the source backdrop is a
            # photo).  A retained-contrast verdict there is untrusted in BOTH
            # directions, so abstain: a plate-colour fill over real texture is worse
            # than the miss (025 hair headline; ghost photographic-plate guard).
            return None, 0.0, smear_grad_min_ratio, "gradient"
        c_src = src_l - _luma(model_src)
        c_plate = plate_l - _luma(model_plate)
        # Mirror the flat path's density term: a strong pixel must sit in a real glyph
        # neighbourhood (den > 4), or a 4x4 sliver's isolated pixels clear the count
        # floor and the variant guesses instead of abstaining (few_strong_pixels test).
        # Computed locally — the flat path's own ``den`` is a sibling scope's local.
        den = cv2.boxFilter((~ink).astype(np.float32), -1,
                            (smear_window, smear_window), normalize=False)
        strong = ink & (den > 4) & (np.abs(c_src) > smear_grad_min_contrast)
        strong_px = int(np.count_nonzero(strong))
        if strong_px < max(8, min_px):
            return None, 0.0, smear_grad_min_ratio, "gradient"
        retained = np.zeros_like(c_src)
        np.divide(c_plate, c_src, out=retained, where=np.abs(c_src) > 1e-6)
        hit = strong & (retained > smear_retain_frac)
        ratio = int(np.count_nonzero(hit)) / max(1, strong_px)
        return hit, ratio, smear_grad_min_ratio, "gradient"

    def _ink_safe_exterior(samples, ink_color):
        """Keep ring samples that are clearly plate-coloured, not leftover glyph ink."""
        if samples is None or getattr(samples, "shape", (0,))[0] == 0:
            return None
        if ink_color is None:
            return samples if samples.shape[0] >= min_ring_samples else None
        ink_rgb = np.asarray(ink_color, dtype=np.float32).reshape(3)
        away = np.abs(samples.astype(np.float32) - ink_rgb).mean(axis=1) > ink_away
        filtered = samples[away]
        return filtered if filtered.shape[0] >= min_ring_samples else None

    def _solid_fill_unresolved(tracked_rows, unresolved_rows, report_dict):
        """Paint local plate colour under leftover glyph ink; keep native TEXT."""
        nonlocal plate, plate_u8, plate_f32
        if not unresolved_rows or not solid_fill:
            return unresolved_rows
        try:
            if plate_u8 is None:
                # Same pixels as the already-loaded int16 plate — avoid a second PNG decode.
                plate_u8 = np.clip(plate, 0, 255).astype(np.uint8)
        except Exception:
            return unresolved_rows
        filled_ids = []
        skipped_ids = []
        ring_k = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), np.uint8)
        # Crowded dark UI (009 "krijgen") + edge banners need a wider search.
        max_ring = max(ring_radius * 5, ring_radius + 24)
        for t in unresolved_rows:
            residue = t["residue"]
            ink = t["ink"]
            if not np.any(residue) and not np.any(ink):
                continue
            # Prefer full glyph footprint (+ halo): residue-only leaves readable fringe
            # on crowded display copy. solid_fill_first always uses ink; after reinpaint
            # still union ink∪residue so smear outside the residual mask clears.
            if solid_first and np.any(ink):
                seed = ink.astype(np.uint8)
            else:
                seed = np.maximum(ink.astype(np.uint8), residue.astype(np.uint8))
            if fill_dilate:
                fill_k = np.ones((2 * fill_dilate + 1, 2 * fill_dilate + 1), np.uint8)
                fill_area = cv2.dilate(seed * 255, fill_k) > 0
            else:
                fill_area = seed > 0
            binary = fill_area.astype(np.uint8)
            ink_color = t.get("ink_color")

            def _sample_ring(radius):
                k = (ring_k if radius == ring_radius
                     else np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
                ring = (cv2.dilate(binary * 255, k) > 0) & (~fill_area)
                return plate_u8[ring]

            # Expand until we have ink-safe plate samples (never accept an inky median —
            # that greenwashes residue as "resolved" while repainting the ghost).
            exterior = _ink_safe_exterior(_sample_ring(ring_radius), ink_color)
            expand_r = ring_radius
            while exterior is None and expand_r < max_ring:
                expand_r = min(max_ring, expand_r + max(2, ring_radius // 2))
                exterior = _ink_safe_exterior(_sample_ring(expand_r), ink_color)
            # Last resort: sample the box neighbourhood outside fill (edge banners where
            # the ring hugs the canvas). Still ink-safe — else leave unresolved.
            if exterior is None:
                box = t["item"].get("box") or {}
                x0 = max(0, int(box.get("x", 0) or 0) - max_ring)
                y0 = max(0, int(box.get("y", 0) or 0) - max_ring)
                x1 = min(plate_u8.shape[1], int((box.get("x", 0) or 0)
                         + (box.get("w", 0) or 0) + max_ring))
                y1 = min(plate_u8.shape[0], int((box.get("y", 0) or 0)
                         + (box.get("h", 0) or 0) + max_ring))
                if x1 > x0 and y1 > y0:
                    neigh = np.ones(plate_u8.shape[:2], dtype=bool)
                    neigh[:] = False
                    neigh[y0:y1, x0:x1] = True
                    neigh &= ~fill_area
                    exterior = _ink_safe_exterior(plate_u8[neigh], ink_color)
            if exterior is None or exterior.shape[0] < min_ring_samples:
                skipped_ids.append(str(t["item"].get("id")))
                continue
            fill = np.median(exterior.astype(np.float32), axis=0).astype(np.uint8)
            # Refuse an ink-like fill colour — that is the crowded-glyph greenwash.
            if ink_color is not None:
                fill_dist = float(np.abs(
                    fill.astype(np.float32) - np.asarray(ink_color, dtype=np.float32).reshape(3)
                ).mean())
                if fill_dist <= ink_away:
                    skipped_ids.append(str(t["item"].get("id")))
                    continue
            # Trial paint + residue check before committing union/ledger.
            before = plate_u8[fill_area].copy()
            gradient = (_gradient_plate_fill(plate_u8, fill_area)
                        if (smear_gradient_fill and t.get("kind") == "smear") else None)
            if gradient is not None:
                plate_u8[fill_area] = gradient[fill_area]
                used_gradient = True
            else:
                plate_u8[fill_area] = fill
                used_gradient = False
            plate_f32 = None  # trial pixels must be scored, not a stale cache
            trial = plate_u8.astype(np.int16)
            remaining = int(_residue(trial, ink, ink_color).sum())
            rem_ratio = remaining / max(1, t["total"])
            reverted = remaining >= resolved_abs_px and rem_ratio >= resolved_ratio
            if not reverted and t.get("kind") == "smear":
                # A smear was never visible to the absolute tests, so their "resolved"
                # verdict cannot clear it either — re-run the detector that flagged it.
                # The returned gate tracks the variant (flat vs gradient) that re-fires.
                _smear_after, smear_after_ratio, after_gate, _ = _smear_residue(trial, ink)
                reverted = (_smear_after is not None
                            and smear_after_ratio >= after_gate)
            if reverted:
                plate_u8[fill_area] = before
                plate_f32 = None
                skipped_ids.append(str(t["item"].get("id")))
                continue
            fill_u8 = fill_area.astype(np.uint8) * 255
            owner = int(t["item"].get("removal_owner") or 0)
            if owner:
                ledger[(fill_area) & (ledger == 0)] = owner
            np.maximum(union, fill_u8, out=union)
            np.maximum(expand_total, fill_u8, out=expand_total)
            t["resolved"] = True
            t["flag"]["resolved"] = True
            t["flag"]["resolved_by"] = ("gradient-plate-fill" if used_gradient
                                        else "solid-plate-fill")
            t["flag"]["hard_fail"] = False
            t["flag"]["residual_px_after"] = remaining
            filled_ids.append(str(t["item"].get("id")))
        if filled_ids:
            Image.fromarray(plate_u8).save(background_path)
            plate = plate_u8.astype(np.int16)
            plate_f32 = None
            report_dict["solid_filled_ids"] = list(dict.fromkeys(
                list(report_dict.get("solid_filled_ids") or []) + filled_ids
            ))
            report_dict["solid_fill_first"] = bool(solid_first and max_passes == 0)
        if skipped_ids:
            report_dict["solid_fill_skipped_ids"] = list(dict.fromkeys(
                list(report_dict.get("solid_fill_skipped_ids") or []) + skipped_ids
            ))
        return [t for t in tracked_rows if not t["resolved"]]

    checked = 0
    tracked = []
    source_u8 = None
    for item in text_records:
        # Use the pre-dilation glyph-ink mask: the ledger's dilated halo is plate-
        # coloured on both sides and would count as false "residue".
        ink_mask = ink_masks.get(item.get("id"))
        if ink_mask is None:
            continue
        ink = np.asarray(ink_mask) > 0
        # Fat candidate masks (box-promoted / AA fringe) include plate-coloured holes
        # that still match the source plate after a clean inpaint — 066's cream
        # headline flagged 42% "residue" while background_clean had zero dark glyphs.
        # Intersect with a fresh tight ink estimate so only real glyph pixels gate.
        box = item.get("box") or {}
        if box and (box.get("w") or 0) > 0 and (box.get("h") or 0) > 0:
            try:
                if source_u8 is None:
                    source_u8 = np.clip(source, 0, 255).astype(np.uint8)
                tight = inpaint.text_ink_mask(
                    source_u8, box, allow_box_fallback=False,
                ) > 0
                if np.count_nonzero(tight) >= 8:
                    ink = ink & tight if np.any(ink & tight) else tight
            except Exception:
                pass
        total = int(ink.sum())
        if total < 8:
            continue
        checked += 1
        ink_color = np.median(source[ink].astype(np.float32), axis=0).reshape(1, 1, 3)
        residue = _residue(plate, ink, ink_color)
        # Plate-coloured fringe inside the mask is not glyph ghosting.
        source_inkish = np.abs(
            source.astype(np.float32) - ink_color
        ).mean(axis=2) <= ink_tolerance
        residue = residue & source_inkish
        count = int(residue.sum())
        ratio = count / total
        kind = None
        smear_ratio = None
        smear_variant = None
        if count >= min_px and ratio >= min_ratio:
            kind = "ink-match"
        elif smear_enabled:
            # The absolute tests cleared it; a partial smear still reads as doubled copy.
            # The detector returns the gate for whichever variant (flat / gradient) fired.
            smear, s_ratio, s_gate, s_variant = _smear_residue(plate, ink)
            if (smear is not None and s_ratio >= s_gate
                    and int(smear.sum()) >= min_px):
                residue = residue | smear
                count = int(residue.sum())
                ratio = count / total
                kind = "smear"
                smear_ratio = round(float(s_ratio), 4)
                smear_variant = s_variant
        if kind is None:
            continue
        flag = {
            "id": item.get("id"), "box": item.get("box"),
            "residual_px": count, "residual_ratio": round(ratio, 4),
            "kind": kind, "resolved": False,
        }
        if smear_ratio is not None:
            flag["smear_ratio"] = smear_ratio
        if smear_variant is not None:
            flag["smear_variant"] = smear_variant
        tracked.append({
            "item": item, "ink": ink, "ink_color": ink_color, "total": total,
            "residue": residue, "kind": kind,
            "flag": flag,
            "resolved": False,
        })

    # Foreign decoration ink (091 red strike scribble): a different colour than the glyph
    # and reaching past the glyph box, so the per-line loop above cannot see it. Only clean
    # it when the struck LINE was actually promoted to native text (a text removal record
    # overlaps its box) — otherwise the scribble is legitimately part of a raster slice.
    for region in (decoration_regions or []):
        sbox = region.get("box") or {}
        sw = int(round(float(sbox.get("w", 0) or 0)))
        sh = int(round(float(sbox.get("h", 0) or 0)))
        if sw <= 0 or sh <= 0:
            continue
        line_box = region.get("line_box") or sbox
        owner_item = None
        for item in text_records:
            ibox = item.get("box") or {}
            if not ibox:
                continue
            if _inside_frac(line_box, ibox) > 0.35 or _inside_frac(ibox, line_box) > 0.35:
                owner_item = item
                break
        if owner_item is None:
            continue
        x0 = max(0, int(round(float(sbox.get("x", 0) or 0))))
        y0 = max(0, int(round(float(sbox.get("y", 0) or 0))))
        x1 = min(source.shape[1], x0 + sw)
        y1 = min(source.shape[0], y0 + sh)
        if x1 <= x0 or y1 <= y0:
            continue
        region_mask = np.zeros(source.shape[:2], dtype=bool)
        region_mask[y0:y1, x0:x1] = True
        plate_bg = np.median(
            plate[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32), axis=0)
        mark = region_mask & (
            np.abs(source.astype(np.float32) - plate_bg).mean(axis=2) > deco_mark_tol)
        # Still uncleaned == plate pixel still equals source (inpaint skipped it).
        still = np.abs(plate - source).mean(axis=2) <= tolerance
        deco_residue = mark & still
        resid_px = int(deco_residue.sum())
        if resid_px < deco_min_px:
            continue
        ink_color = np.median(
            source[deco_residue].astype(np.float32), axis=0).reshape(1, 1, 3)
        total = int(mark.sum()) or resid_px
        deco_id = f"{owner_item.get('id')}__strike"
        deco_flag = {
            "id": deco_id, "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "residual_px": resid_px, "residual_ratio": round(resid_px / max(1, total), 4),
            "kind": "decoration-strikethrough", "resolved": False,
        }
        tracked.append({
            "item": {
                "id": deco_id,
                "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                "removal_owner": owner_item.get("removal_owner"),
                "target": "text",
            },
            "ink": deco_residue, "ink_color": ink_color, "total": total,
            "residue": deco_residue, "flag": deco_flag, "resolved": False,
        })
        checked += 1

    expand_total = np.zeros(union.shape, dtype=np.uint8)
    report = {
        "enabled": True, "checked": checked,
        "flagged": [t["flag"] for t in tracked],
        "expanded_px": 0, "reinpainted": False, "passes": 0,
        "solid_filled_ids": [],
    }
    if not tracked:
        report["expanded_px"] = int(np.count_nonzero(expand_total))
        return report

    # Flat/UI: solid-fill FIRST so we never spend 1–3 role-aware reinpaint calls on a
    # plate that analytic fill already owns. Photo scenes keep a short reinpaint ladder.
    unresolved = [t for t in tracked if not t["resolved"]]
    if solid_first and solid_fill and unresolved:
        unresolved = _solid_fill_unresolved(tracked, unresolved, report)

    passes_done = 0
    backend = None
    if unresolved and bool(audit_cfg.get("reinpaint", True)) and max_passes > 0:
        for pass_index in range(max_passes):
            pending = [t for t in tracked if not t["resolved"]]
            if not pending:
                break
            dilate_px = grow * (pass_index + 1)
            kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
            expand = np.zeros(union.shape, dtype=np.uint8)
            for t in pending:
                grown = cv2.dilate(t["residue"].astype(np.uint8) * 255, kernel)
                owner = int(t["item"].get("removal_owner") or 0)
                if owner:
                    ledger[(grown > 0) & (ledger == 0)] = owner
                np.maximum(union, grown, out=union)
                np.maximum(expand, grown, out=expand)
                np.maximum(expand_total, grown, out=expand_total)
            if not np.any(expand):
                break
            try:
                second = inpaint.inpaint_role_aware(
                    background_path, {"text": expand}, background_path, cfg,
                )
                report["reinpainted"] = bool(second.get("ok", True))
                backend = second.get("backend")
            except Exception as exc:  # a failed repair pass is evidence, never a crash
                report["reinpaint_error"] = str(exc)
                break
            passes_done = pass_index + 1
            plate = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.int16)
            plate_u8 = None
            plate_f32 = None
            for t in pending:
                residue = _residue(plate, t["ink"], t["ink_color"])
                remaining = int(residue.sum())
                t["residue"] = residue
                t["flag"]["residual_px_after"] = remaining
                rem_ratio = remaining / max(1, t["total"])
                if remaining < resolved_abs_px or rem_ratio < resolved_ratio:
                    t["resolved"] = True
                    t["flag"]["resolved"] = True
    report["passes"] = passes_done
    report["reinpaint_backend"] = backend
    report["expanded_px"] = int(np.count_nonzero(expand_total))
    # Genuine residue under an EMITTED native-text layer would double-render. Prefer a
    # deterministic local plate fill (keep editable TEXT) over baking OCR into a slice.
    unresolved = [t for t in tracked if not t["resolved"]]
    unresolved = _solid_fill_unresolved(tracked, unresolved, report)
    for t in unresolved:
        t["flag"]["hard_fail"] = True
    # Explicit top-level surfacing: residue that could NOT be re-inpainted or solid-filled
    # is a reconstruction failure, not a silent ship. pixel_diff already reads per-flag
    # hard_fail, but a top-level roster lets QA/harness/repair see the honest count directly
    # (Mandate 2: never silently ship unresolved glyph/decoration ghosts).
    if unresolved:
        report["hard_fail"] = True
        report["hard_fail_ids"] = sorted(str(t["item"].get("id")) for t in unresolved)
        report["unresolved_px"] = int(sum(
            int(t["flag"].get("residual_px_after", t["flag"].get("residual_px", 0)) or 0)
            for t in unresolved
        ))
    if force_raster:
        report["force_raster_ids"] = sorted({
            str(t["item"].get("id")) for t in unresolved
            if t["item"].get("target") == "text"
        })
    return report


def _build_removal_ledger(observations: list, canvas: tuple[int, int]):
    """Assign every final removal pixel to exactly one accepted observation.

    Asset alpha ownership is solved earlier from canonical masks.  This second, final ledger
    starts only from observations that survived routing and mask approval, expands their halo
    once, and makes overlaps exclusive before any inpaint backend sees them.  Consequently a
    dropped/retained observation cannot silently punch a hole and regional inpainting cannot
    process the same pixel twice.
    """
    cv2, np, _ = _deps()
    width, height = canvas

    def priority(item):
        target = str(item.get("target") or "")
        role = str(item.get("role") or "").lower()
        meta = item.get("meta") or {}
        target_rank = {"text": 40, "icon": 30, "shape": 20, "image": 10}.get(target, 0)
        # A verified arrow or authored price rule overlaps OCR by design. It must own
        # those source pixels before the broad price text mask, otherwise only a tiny
        # sliver reaches its asset/fallback and the mark disappears from the export.
        if role in {"arrow", "callout_leader", "leader", "leader_line", "connector",
                    "underline", "strikethrough", "annotation"} or meta.get("native_decoration"):
            target_rank = 60
        if role in {"product", "person", "foreground", "cutout"}:
            target_rank += 5
        box = item.get("box") or {}
        area = float(box.get("w", 0)) * float(box.get("h", 0))
        return target_rank, float(item.get("z", 0)), -area, str(item.get("id") or "")

    ledger = np.zeros((height, width), dtype=np.uint16)
    records = []
    owner_index = {}
    for number, item in enumerate(sorted(observations, key=priority, reverse=True), start=1):
        mask = item.get("mask_array")
        if mask is None:
            continue
        mask = inpaint.solidify_mask(mask)
        radius = max(0, int(item.get("dilate", 0)))
        if radius:
            mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        available = (mask > 0) & (ledger == 0)
        if not np.any(available):
            continue
        ledger[available] = number
        owned = dict(item)
        owned["mask_array"] = available.astype(np.uint8) * 255
        owned["dilate"] = 0
        owned["removal_owner"] = number
        records.append(owned)
        owner_index[str(number)] = item.get("id")
    union = (ledger > 0).astype(np.uint8) * 255
    return records, union, ledger, owner_index


def _cover_kept_raster_footprints(background_path, candidates, masks, cfg, union=None):
    """Fill kept-in-plate opaque-raster footprints with plate colour (no ghost silhouette).

    Targets ONLY opaque rasters that genuinely still re-render on top of the plate
    (``meta.keep_in_background`` with an emitted ``target == "image"`` layer). Those pixels
    are hidden by the re-rendered asset, so replacing them with the surrounding plate
    colour is loss-free for the composite and removes the silhouette QA sees in
    background_clean.

    Plate-owned candidates (``meta.plate_passthrough`` / removal-capped drops) are NEVER
    covered: they are not re-rendered, so the original source pixels ARE the plate.
    Covering them repaints huge out-of-mask regions with the ring median — on 002 this
    painted the entire 46%-of-canvas white product panel orange, with a hard seam at the
    footprint's dilated boundary (y≈499) and edge (x≈53).

    Safety gates (both configurable under ``reconstruct.*``):
      * ``cover_footprint_max_fraction`` (default 0.10) — a footprint bigger than this
        cannot plausibly be "one flat plate colour"; skip it.
      * the exterior ring must be near-uniform (``cover_footprint_uniform_fraction``,
        default 0.70 within ``cover_footprint_tolerance``, default 12) — otherwise the
        median fill invents a colour that matches nothing.

    Every covered pixel is OR'ed into ``union`` (when given) so the plate-integrity
    invariant — plate differs from source only inside removal_mask.png — stays true.
    Config gate ``reconstruct.cover_kept_footprints`` (default ON)."""
    cv2, np, Image = _deps()
    rcfg = (cfg or {}).get("reconstruct") or {}
    if not bool(rcfg.get("cover_kept_footprints", True)):
        return 0
    if not os.path.exists(background_path):
        return 0
    targets = []
    for c in candidates or []:
        meta = c.get("meta") or {}
        if meta.get("plate_passthrough"):
            continue
        # Only opaque rasters that still ship as layers re-render over their own
        # footprint; those are the only ones whose plate pixels are provably hidden.
        if not (meta.get("keep_in_background") and c.get("target") == "image"):
            continue
        mask = masks.get(c.get("id"))
        if mask is not None and np.count_nonzero(mask):
            targets.append(mask)
    if not targets:
        return 0
    try:
        plate = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8).copy()
    except Exception:
        return 0
    ph, pw = plate.shape[:2]
    dilate = max(1, int(rcfg.get("cover_footprint_dilate", 3)))
    ring_radius = max(4, int(rcfg.get("cover_footprint_ring", 12)))
    max_fraction = float(rcfg.get("cover_footprint_max_fraction", 0.10))
    uniform_fraction = float(rcfg.get("cover_footprint_uniform_fraction", 0.70))
    tolerance = float(rcfg.get("cover_footprint_tolerance", 12.0))
    covered = 0
    for mask in targets:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.shape != (ph, pw):
            binary = cv2.resize(binary, (pw, ph), interpolation=cv2.INTER_NEAREST)
        binary = cv2.dilate(binary, np.ones((2 * dilate + 1,) * 2, np.uint8))
        if float(np.count_nonzero(binary)) / max(1, binary.size) > max_fraction:
            continue
        ring = (cv2.dilate(binary, np.ones((2 * ring_radius + 1,) * 2, np.uint8)) > 0) & (binary == 0)
        exterior = plate[ring]
        if exterior.shape[0] < 16:
            continue
        fill = np.median(exterior.astype(np.float32), axis=0)
        near = np.max(np.abs(exterior.astype(np.float32) - fill), axis=1) <= tolerance
        if float(np.mean(near)) < uniform_fraction:
            continue
        plate[binary > 0] = fill.astype(np.uint8)
        if union is not None:
            np.maximum(union, binary * 255, out=union)
        covered += 1
    if covered:
        Image.fromarray(plate).save(background_path)
    return covered


def _asset_has_content(candidate, run_dir, min_alpha_px=24):
    """True when a candidate's emitted raster/vector actually paints visible pixels.

    An "emitted" image layer whose asset is a blank/near-empty PNG re-renders nothing, so
    the plate underneath it is the only surface a viewer sees. Such an owner must be
    treated as absent for the unclaimed-removal restore (104/107 shipped 8KB blank product
    PNGs while the product was inpainted out of the plate).
    """
    _, np, Image = _deps()
    if candidate.get("paths") or candidate.get("svg"):
        return True
    src = candidate.get("src")
    if not src or str(src).startswith("data:"):
        return False
    path = inpaint.resolve_path(src, run_dir)
    if not path or not os.path.exists(path):
        return False
    try:
        image = Image.open(path)
    except Exception:
        return False
    if "A" in image.getbands():
        alpha = np.asarray(image.split()[-1])
        return int(np.count_nonzero(alpha > 16)) >= int(min_alpha_px)
    # No alpha channel: an opaque raster always paints something.
    return image.width * image.height >= int(min_alpha_px)


def _removal_owner_rerenders(candidate, run_dir):
    """Whether the candidate that owns a removal region actually re-renders over it.

    A removal owner that ships a visible layer (native text, a flat shape/plate, or a
    non-empty raster/vector asset) legitimately keeps its inpainted footprint. Everything
    else — a drop that is not an explicit removal, a plate passthrough, a blank asset —
    leaves the plate as the only visible surface, so its removal must be restored.
    """
    meta = candidate.get("meta") or {}
    target = candidate.get("target")
    if meta.get("plate_passthrough"):
        return False
    if meta.get("removal_required"):
        # Explicit removal: overlay text repainted natively, or a source owner kept only
        # to rebuild the plate for its split children (they re-render the pixels).
        return True
    if target == "text":
        return True
    if target == "shape":
        return True
    if target in ("image", "icon"):
        return _asset_has_content(candidate, run_dir)
    return False


def _restore_unclaimed_removals(rgb, background_path, removal, candidates, run_dir,
                                union, removal_ownership, cfg):
    """Restore source pixels for removal regions no emitted layer will re-render.

    ``removal`` is the final exclusive ledger (each record owns a disjoint pixel set).
    For every record whose owning candidate does not re-render, copy the original pixels
    back into the plate and clear that region from ``union`` + ``removal_ownership`` (both
    mutated in place). The plate-integrity invariant is preserved: restored pixels equal
    the source and leave the union simultaneously, so they read as out-of-mask AND
    unchanged. Config gate ``reconstruct.restore_unclaimed_removals`` (default ON)."""
    _, np, Image = _deps()
    rcfg = (cfg or {}).get("reconstruct") or {}
    if not bool(rcfg.get("restore_unclaimed_removals", True)):
        return {"regions": 0, "restored_px": 0, "ids": []}
    if not os.path.exists(background_path):
        return {"regions": 0, "restored_px": 0, "ids": []}
    by_id = {str(c.get("id")): c for c in (candidates or [])}
    try:
        plate = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8).copy()
    except Exception:
        return {"regions": 0, "restored_px": 0, "ids": []}
    ph, pw = plate.shape[:2]
    if rgb.shape[:2] != (ph, pw):
        return {"regions": 0, "restored_px": 0, "ids": []}
    restored_ids = []
    restored_px = 0
    for record in removal or []:
        cand = by_id.get(str(record.get("id")))
        if cand is None:
            continue
        if _removal_owner_rerenders(cand, run_dir):
            continue
        mask = record.get("mask_array")
        if mask is None:
            continue
        region = np.asarray(mask) > 0
        if region.shape != (ph, pw) or not region.any():
            continue
        plate[region] = rgb[region]
        union[region] = 0
        if removal_ownership is not None:
            removal_ownership[region] = 0
        restored_px += int(region.sum())
        restored_ids.append(str(record.get("id")))
    if restored_px:
        Image.fromarray(plate).save(background_path)
    return {"regions": len(restored_ids), "restored_px": restored_px, "ids": restored_ids}


def _crop_rgba(rgb, mask, box, element_role=None):
    _, np, Image = _deps()
    h, w = rgb.shape[:2]
    x0 = max(0, int(round(box.get("x", 0))))
    y0 = max(0, int(round(box.get("y", 0))))
    x1 = min(w, int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(h, int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    # Refine the raw binary SAM mask into a production alpha (feather + colour
    # decontamination + background-fringe suppression) instead of pasting the hard
    # mask straight in — a binary alpha paste is the exact source of the doubled-
    # contour/white-halo defect (002 audit finding #5, edge precision 0.5526).
    try:
        from src import matting
        out = matting.refine(rgb, mask, box={"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                             element_role=element_role)
        return Image.fromarray(out.rgba)
    except Exception:
        rgba = np.dstack([rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
        return Image.fromarray(rgba.astype(np.uint8))


def _source_rgba(candidate, rgb, mask, run_dir):
    """Prefer a model-provided clean RGBA layer, correctly cropped to its tight box."""
    _, np, Image = _deps()
    cluster_meta = candidate.get("meta") or {}
    role = str(cluster_meta.get("role") or "").lower()
    if role in {"arrow", "callout_leader", "leader", "leader_line", "connector"}:
        # Peeling can estimate a white/grey foreground for a black mark when the mark
        # overlaps OCR. SAM already supplies the verified matte; retain the original
        # authored pixels inside it instead of trusting the colour-hallucinated peel.
        return _crop_rgba(rgb, mask, candidate.get("box", {}), element_role=role)
    # Chrome-baked checklist cards (066): peel may have already punched ✓/✗ holes into
    # the under-layer before merge absorbed those chips. Always take the authored
    # source crop so icons + copy stay intact in the shell raster.
    if (
        cluster_meta.get("checklist_raster_chip")
        or cluster_meta.get("baked_badge_text")
        or cluster_meta.get("shell_raster_chip")
    ):
        return _crop_rgba(rgb, mask, candidate.get("box", {}), element_role=role)
    if (is_intentional_raster_cluster(cluster_meta.get("role"))
            or cluster_meta.get("intentional_raster_cluster")):
        # Do not use a transparent Qwen/SAM crop here: the original full crop is the
        # fidelity contract for an inseparable cluster.
        return _crop_rgba(rgb, mask, candidate.get("box", {}), element_role=role)
    path = inpaint.resolve_path(candidate.get("src"), run_dir)
    box = candidate.get("box", {})
    if path:
        image = Image.open(path).convert("RGBA")
        canvas_h, canvas_w = rgb.shape[:2]
        if image.size == (canvas_w, canvas_h):
            x = max(0, int(round(box.get("x", 0))))
            y = max(0, int(round(box.get("y", 0))))
            w = max(1, int(round(box.get("w", 1))))
            h = max(1, int(round(box.get("h", 1))))
            return image.crop((x, y, min(canvas_w, x + w), min(canvas_h, y + h)))
        target = (max(1, int(round(box.get("w", image.width)))),
                  max(1, int(round(box.get("h", image.height)))))
        return image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
    return _crop_rgba(rgb, mask, box, element_role=role)


def _apply_owned_alpha(image, owned_mask, box):
    """Ensure a foreground pixel is present in at most one exported raster asset."""
    _, np, Image = _deps()
    x0 = max(0, int(round(box.get("x", 0))))
    y0 = max(0, int(round(box.get("y", 0))))
    x1 = min(owned_mask.shape[1], int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(owned_mask.shape[0], int(round(box.get("y", 0) + box.get("h", 0))))
    local = (owned_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    if local.size == 0:
        return image
    if (local.shape[1], local.shape[0]) != image.size:
        local = np.asarray(Image.fromarray(local).resize(image.size, Image.Resampling.NEAREST))
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], local)
    return Image.fromarray(rgba)


def _split_comparison_frame(candidate, image, assets_dir, cfg):
    """Turn a verified wide before/after raster into independently swappable columns.

    The comparison preset is explicit evidence that the two halves have semantic value.
    Splitting only its broad photographic frame avoids guessing at arbitrary two-column
    layouts while making the common before/after creative genuinely editable in Figma.
    The original owner is retained as a removal-only observation by the caller, so the
    clean plate is still rebuilt exactly once.
    """
    scene = cfg.get("scene") or {}
    if scene.get("archetype") != "comparison_grid":
        return []
    if not (scene.get("facts") or {}).get("before_after_pair"):
        return []
    if candidate.get("target") != "image":
        return []
    meta = candidate.get("meta") or {}
    if str(meta.get("role") or "").lower() not in {"photo", "image", "photo-card"}:
        return []
    box = candidate.get("box") or {}
    width, height = image.size
    if width < 80 or height < 60 or width / max(1, height) < 1.25:
        return []
    # A near 50/50 split is deliberately used only after archetype evidence (the OCR
    # has already supplied BEFORE/AFTER labels).  Odd pixels stay with the right side.
    cut = width // 2
    if cut < 32 or width - cut < 32:
        return []
    out = []
    for index, (label, left, right) in enumerate((("Before", 0, cut), ("After", cut, width))):
        part = image.crop((left, 0, right, height))
        child = dict(candidate)
        child["id"] = f"{candidate.get('id')}-{'before' if index == 0 else 'after'}"
        child["name"] = f"{label} image — swappable crop"
        child["box"] = {
            **box,
            "x": float(box.get("x") or 0) + float(left),
            "w": float(right - left),
        }
        child["meta"] = dict(meta)
        side = "before" if index == 0 else "after"
        child["meta"].update({
            "role": "comparison-column",
            "comparison_side": label.lower(),
            "semantic_name": f"{label} image",
            "parent_id": candidate.get("id"),
        })
        prov = child["meta"].get("provenance") or {}
        if isinstance(prov, dict):
            observations = list(prov.get("observations") or child["meta"].get("observations") or [])
            remapped = []
            for observation in observations:
                if not isinstance(observation, dict):
                    remapped.append(observation)
                    continue
                entry = dict(observation)
                if entry.get("key"):
                    entry["key"] = f"{entry['key']}-{side}"
                elif entry.get("id") is not None:
                    entry["id"] = f"{entry['id']}-{side}"
                remapped.append(entry)
            if remapped:
                prov = dict(prov)
                prov["observations"] = remapped
                child["meta"]["provenance"] = prov
                child["meta"]["observations"] = remapped
        child["mask"] = {"kind": "rrect", "radius": 0.0}
        child["src"] = _write_asset(part, assets_dir, child["id"])
        out.append(child)
    return out


def _comparison_plate_columns(background_path, assets_dir, width, height, cfg, candidates):
    """Expose a full-bleed comparison plate as two swappable base-image layers."""
    if (cfg.get("scene") or {}).get("archetype") != "comparison_grid":
        return []
    if not ((cfg.get("scene") or {}).get("facts") or {}).get("before_after_pair"):
        return []
    if any((item.get("meta") or {}).get("comparison_side") for item in candidates):
        return []
    _, _, Image = _deps()
    plate = Image.open(background_path).convert("RGBA")
    if plate.size != (width, height) or width < 80:
        return []
    cut = width // 2
    out = []
    for side, left, right in (("before", 0, cut), ("after", cut, width)):
        label = side.title()
        candidate = {
            "id": f"comparison-plate-{side}",
            "target": "image",
            "name": f"{label} image — swappable base",
            "box": {"x": left, "y": 0, "w": right - left, "h": height},
            "z_index": -999_999,
            "mask": {"kind": "rect"},
            "meta": {
                "role": "comparison-column",
                "comparison_side": side,
                "semantic_name": f"{label} image",
                "swappable": True,
                "source": "clean-plate-column",
            },
        }
        candidate["src"] = _write_asset(
            plate.crop((left, 0, right, height)), assets_dir, candidate["id"]
        )
        out.append(candidate)
    return out


_PLATE_SHELL_ROLES = {"button", "badge", "chip", "pill", "message-bubble", "message", "bubble"}
_ENGAGEMENT_ICON_ROLES = {
    "icon", "engagement", "like", "reply", "repost", "share", "comment", "views",
}


def _fill_luma(fill) -> Optional[float]:
    """Approximate luma of a flat fill; None when not a flat hex colour."""
    if not isinstance(fill, dict):
        return None
    if str(fill.get("kind") or "flat").lower() not in {"flat", "solid", ""}:
        return None
    color = str(fill.get("color") or "")
    if not color.startswith("#") or len(color) < 7:
        return None
    try:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
    except ValueError:
        return None
    return 0.299 * r + 0.587 * g + 0.114 * b


def _suppress_engagement_underlay_shells(candidates, cfg=None):
    """Drop tiny dark button/ellipse plates that sit under engagement icons.

    CODIA-PARITY (009): a bogus near-black ellipse tagged ``Button`` under the
    comment icon. Real engagement chrome is the icon cutout; the underlay is a
    plate fragment, not an editable control. Gated to social_screenshot by
    default (``reconstruct.suppress_engagement_underlays``).
    """
    rcfg = (cfg or {}).get("reconstruct") or {}
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    enabled = rcfg.get("suppress_engagement_underlays")
    if enabled is None:
        enabled = archetype == "social_screenshot"
    if not enabled:
        return candidates, 0
    icons = [
        item for item in candidates
        if item.get("target") in {"icon", "image"}
        and str((item.get("meta") or {}).get("role") or "").lower() in _ENGAGEMENT_ICON_ROLES
    ]
    if not icons:
        return candidates, 0
    suppressed = 0
    for item in candidates:
        if item.get("target") != "shape" or item.get("text"):
            continue
        meta = item.get("meta") or {}
        role = str(meta.get("role") or "").lower()
        if role not in {"button", "badge", "chip", "shape", ""} and not meta.get("button_shell"):
            continue
        box = item.get("box") or {}
        fw, fh = float(box.get("w") or 0), float(box.get("h") or 0)
        if fw <= 0 or fh <= 0 or max(fw, fh) > 64 or min(fw, fh) < 6:
            continue
        # Near-square / circular underlays only (the bogus 009 ellipse).
        aspect = max(fw, fh) / max(1.0, min(fw, fh))
        if aspect > 1.45:
            continue
        luma = _fill_luma(item.get("fill") or (item.get("style") or {}).get("fill"))
        if luma is None or luma > 40.0:
            continue
        for icon in icons:
            if _overlap_frac(box, icon.get("box") or {}) < 0.45:
                continue
            # Underlay should be comparable to or smaller than the icon.
            ib = icon.get("box") or {}
            if fw * fh > 1.35 * max(1.0, float(ib.get("w") or 0) * float(ib.get("h") or 0)):
                continue
            meta = item.setdefault("meta", {})
            item["target"] = "drop"
            meta["keep_in_background"] = True
            meta["removal_required"] = True
            meta["suppression_reason"] = "engagement-icon-underlay"
            meta["underlay_of"] = icon.get("id")
            suppressed += 1
            break
    return candidates, suppressed


def _overlap_frac(a, b) -> float:
    """Intersection over the smaller box area (0..1)."""
    ax0, ay0 = float(a.get("x") or 0), float(a.get("y") or 0)
    ax1, ay1 = ax0 + float(a.get("w") or 0), ay0 + float(a.get("h") or 0)
    bx0, by0 = float(b.get("x") or 0), float(b.get("y") or 0)
    bx1, by1 = bx0 + float(b.get("w") or 0), by0 + float(b.get("h") or 0)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    smaller = min(max(1.0, (ax1 - ax0) * (ay1 - ay0)), max(1.0, (bx1 - bx0) * (by1 - by0)))
    return inter / smaller


def _suppress_plate_boundary_fragments(candidates, cfg=None):
    """Fold sliver debris on a button/badge/chip rim back into its plate.

    Segmentation frequently peels a rounded plate's anti-aliased end caps into
    separate icon/shape fragments (009 "Volgend": both pill caps became tiny
    "icons").  Rendered on top of the restored native plate they re-draw the
    original's background ring — the "bitten edge" artifact.  A real glyph or
    icon inside a button sits clear of the rim, so only small fragments that
    cross the plate's boundary ring are suppressed; their source pixels remain
    covered by the plate's own removal footprint.  Config-gated, default ON
    (``reconstruct.suppress_plate_fragments``).
    """
    rcfg = (cfg or {}).get("reconstruct") or {}
    if rcfg.get("suppress_plate_fragments", True) is False:
        return candidates, 0
    plates = []
    for item in candidates:
        meta = item.get("meta") or {}
        role = str(meta.get("role") or "").lower()
        box = item.get("box") or {}
        if (item.get("target") == "shape"
                and (role in _PLATE_SHELL_ROLES or meta.get("button_shell"))
                and float(box.get("w") or 0) > 0 and float(box.get("h") or 0) > 0):
            plates.append(item)
    if not plates:
        return candidates, 0
    suppressed = 0
    for item in candidates:
        if item.get("target") not in {"icon", "shape", "image"} or item.get("text"):
            continue
        if any(plate is item for plate in plates):
            continue
        box = item.get("box") or {}
        fw, fh = float(box.get("w") or 0), float(box.get("h") or 0)
        if fw <= 0 or fh <= 0:
            continue
        x0, y0 = float(box.get("x") or 0), float(box.get("y") or 0)
        x1, y1 = x0 + fw, y0 + fh
        for plate in plates:
            pbox = plate.get("box") or {}
            pw, ph = float(pbox.get("w") or 0), float(pbox.get("h") or 0)
            if fw * fh > 0.08 * pw * ph:
                continue
            px0, py0 = float(pbox.get("x") or 0), float(pbox.get("y") or 0)
            px1, py1 = px0 + pw, py0 + ph
            # The fragment must live on the plate (a small anti-alias overhang
            # past the true edge is exactly the failure signature, so allow it).
            reach = max(4.0, 0.06 * min(pw, ph))
            if x0 < px0 - reach or y0 < py0 - reach or x1 > px1 + reach or y1 > py1 + reach:
                continue
            # Content that sits clear of the rim (a chevron/glyph icon) is real.
            ring = max(4.0, 0.08 * min(pw, ph))
            if (x0 >= px0 + ring and y0 >= py0 + ring
                    and x1 <= px1 - ring and y1 <= py1 - ring):
                continue
            meta = item.setdefault("meta", {})
            item["target"] = "drop"
            meta["keep_in_background"] = True
            meta["removal_required"] = True
            meta["suppression_reason"] = "plate-boundary-fragment"
            meta["plate_id"] = plate.get("id")
            suppressed += 1
            break
    return candidates, suppressed


def _dominant_fill(rgb, mask, box):
    _, np, _ = _deps()
    x0, y0 = max(0, int(box.get("x", 0))), max(0, int(box.get("y", 0)))
    x1 = min(rgb.shape[1], int(box.get("x", 0) + box.get("w", 0)))
    y1 = min(rgb.shape[0], int(box.get("y", 0) + box.get("h", 0)))
    pixels = rgb[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0]
    if pixels.size == 0:
        return "#cccccc"
    quant = (pixels.astype(np.uint16) // 8) * 8
    colors, counts = np.unique(quant.reshape(-1, 3), axis=0, return_counts=True)
    color = colors[int(counts.argmax())]
    return "#%02x%02x%02x" % tuple(int(v) for v in color)


def _hex(color):
    """Turn a sampled RGB triplet into the paint spelling used by design.json."""
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(value)))) for value in color)


def _local_shape_pixels(rgb, mask, box):
    """Return the image and a binary, tight shape mask for native-paint analysis.

    Segmentation edges are normally anti-aliased.  Treating any non-zero value as shape
    keeps the analysis stable for both SAM masks and source-alpha masks, while the later
    erosion prevents those edge pixels from polluting the sampled fill.
    """
    _, np, _ = _deps()
    x0, y0 = max(0, int(round(box.get("x", 0)))), max(0, int(round(box.get("y", 0))))
    x1 = min(rgb.shape[1], int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(rgb.shape[0], int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return rgb[:0, :0], np.zeros((0, 0), dtype=bool)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 16


CORNER_FIT_MIN_MATCH = 0.94


def _fit_quarter_radius(quadrant):
    """Best-fit rounded-corner radius for a square quadrant oriented corner-at-(0, 0).

    A one-parameter model fit over the whole quadrant instead of a first-occupied
    edge scan: debris elsewhere on the silhouette (009's "Volgend" pill carried a
    one-row residual ledge welded mid-bottom, which made the old row/column scan
    bail) cannot corrupt the estimate, and a full pill cap measures its true
    radius instead of the anti-aliased chord shortfall.  Returns
    ``(radius, match_fraction)`` where match is the fraction of quadrant pixels
    the fitted model explains.
    """
    cv2, np, _ = _deps()
    actual = quadrant > 0
    size = actual.shape[0]
    scale = 1.0
    # Large plates carry no extra corner information; cap the search resolution.
    if size > 64:
        scale = size / 64.0
        actual = cv2.resize(actual.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST) > 0
        size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    best_radius, best_match = 0.0, -1.0
    for radius in range(size + 1):
        inside = (xx >= radius) | (yy >= radius) | (
            (xx - radius) ** 2 + (yy - radius) ** 2 <= radius * radius
        )
        match = float((inside == actual).mean())
        if match > best_match:
            best_radius, best_match = float(radius), match
    return best_radius * scale, best_match


def _corner_radius(local_mask):
    """Infer an axis-aligned rounded-rectangle radius from its four clipped corners.

    Each corner quadrant is fitted against the one-parameter rounded-corner model
    and must explain its pixels almost perfectly; a noisy/partial mask still
    returns ``None`` because a wrong native radius is worse than a rectangular
    fallback.  A cap that rounds through the whole half-height is a pill end
    (radius == min(h, w) / 2 — the single most common ad button) and snaps to the
    exact pill radius instead of being clamped to a smaller value.
    """
    _, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 8:
        return None
    h, w = local_mask.shape
    if float(local_mask.mean()) < .62:
        return None
    quadrant = min(h, w) // 2
    if quadrant < 4:
        return None
    quadrants = (
        local_mask[:quadrant, :quadrant],
        local_mask[:quadrant, ::-1][:, :quadrant],
        local_mask[::-1, ::-1][:quadrant, :quadrant],
        local_mask[::-1, :][:quadrant, :quadrant],
    )
    fits = [_fit_quarter_radius(piece) for piece in quadrants]
    if any(match < CORNER_FIT_MIN_MATCH for _, match in fits):
        return None
    pill_gate = quadrant - max(2.0, quadrant * .12)
    radii = []
    for radius, _ in fits:
        if radius < 1.25:
            radii.append(0.0)
        elif radius >= pill_gate:
            radii.append(min(h, w) / 2)
        else:
            radii.append(radius)
    if not any(radii):
        return 0
    nonzero = [value for value in radii if value > 0]
    if len(nonzero) < 2:
        return None
    # Equal corners are common and should compile to Figma's simple scalar radius.
    if max(nonzero) - min(nonzero) <= max(1.5, min(h, w) * .04):
        return round(float(np.median(nonzero)), 2)
    names = ("topLeft", "topRight", "bottomRight", "bottomLeft")
    return {name: round(value, 2) for name, value in zip(names, radii)}


# An "ellipse" verdict becomes a CLIP: whatever the alpha has outside the inscribed
# ellipse is deleted from the render. So the verdict has to be earned against the pixels,
# not guessed from aspect/fill — those three cheap features are satisfied by any tilted,
# roughly-square blob. 013's grüns pouch scored aspect .80 / fill .76 / corners 0 and was
# called an ellipse at IoU .787, which clipped 10.6% of the bag away (the "weird cropping"
# of the pouch). 0.94 matches vectorize._fit_primitive's gate, the house convention for
# "this silhouette really is that primitive".
SIMPLE_SHAPE_MIN_IOU = 0.94


def _inscribed_ellipse_iou(local_mask):
    """IoU of a silhouette against the ellipse inscribed in its own bounds.

    Mirrors the ellipse render_preview actually draws for ``mask={"kind": "ellipse"}``
    (and vectorize._fit_primitive's fit), so the gate measures the real clip loss.
    """
    _, np, _ = _deps()
    h, w = local_mask.shape
    if h < 2 or w < 2:
        return 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    ellipse = (((xx - (w - 1) / 2.0) / (w / 2.0)) ** 2
               + ((yy - (h - 1) / 2.0) / (h / 2.0)) ** 2) <= 1.0
    union = float(np.logical_or(ellipse, local_mask).sum())
    if not union:
        return 0.0
    return float(np.logical_and(ellipse, local_mask).sum()) / union


def _simple_shape_geometry(local_mask):
    """Return rect/ellipse only where the segmentation really supports a primitive."""
    _, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 4:
        return None
    fill = float(local_mask.mean())
    h, w = local_mask.shape
    aspect = w / max(1, h)
    corners = sum(bool(value) for value in (
        local_mask[0, 0], local_mask[0, -1], local_mask[-1, 0], local_mask[-1, -1]
    ))
    # Keep the existing ellipse heuristic as a cheap pre-filter, but only call it an
    # ellipse once the alpha itself agrees; otherwise the clip eats real content.
    if (.75 <= aspect <= 1.33 and corners <= 1 and .55 <= fill <= .90
            and _inscribed_ellipse_iou(local_mask) >= SIMPLE_SHAPE_MIN_IOU):
        return "ellipse"
    if fill >= .70:
        return "rect"
    return None


def _robust_color(pixels, fallback=(204, 204, 204)):
    _, np, _ = _deps()
    if pixels is None or not len(pixels):
        return np.asarray(fallback, dtype=np.float32)
    # Median is much less likely than a mean to absorb antialiasing or a few specular pixels.
    return np.median(np.asarray(pixels, dtype=np.float32), axis=0)


def _gradient_fill(local_rgb, interior, min_range=18.0, min_r2=.86):
    """Fit a two-stop linear paint when the interior is genuinely explained by a plane.

    Decorative photos can be colourful too.  The R² gate, high quantile colour range and
    primitive-only caller together make this deliberately conservative.
    """
    _, np, _ = _deps()
    ys, xs = np.nonzero(interior)
    if len(xs) < 80:
        return None
    h, w = interior.shape
    x = (xs.astype(np.float32) - (w - 1) / 2) / max(1.0, (w - 1) / 2)
    y = (ys.astype(np.float32) - (h - 1) / 2) / max(1.0, (h - 1) / 2)
    colors = local_rgb[ys, xs].astype(np.float32)
    # Subsample huge surfaces deterministically; it avoids a 4K panel dominating runtime.
    if len(colors) > 12000:
        pick = np.linspace(0, len(colors) - 1, 12000).astype(int)
        x, y, colors = x[pick], y[pick], colors[pick]
    spread = np.percentile(colors, 95, axis=0) - np.percentile(colors, 5, axis=0)
    if float(np.linalg.norm(spread)) < min_range:
        return None
    design = np.column_stack((np.ones(len(x)), x, y))
    coefficients, _, _, _ = np.linalg.lstsq(design, colors, rcond=None)
    prediction = design @ coefficients
    total = float(np.square(colors - colors.mean(axis=0)).sum())
    if total <= 1e-6:
        return None
    r2 = 1 - float(np.square(colors - prediction).sum()) / total
    if r2 < min_r2:
        return None
    # PCA turns a three-channel plane into a deterministic visual direction.
    _, _, vh = np.linalg.svd(colors - colors.mean(axis=0), full_matrices=False)
    principal = vh[0]
    dx, dy = float(coefficients[1] @ principal), float(coefficients[2] @ principal)
    magnitude = (dx * dx + dy * dy) ** .5
    if magnitude < .5:
        return None
    dx, dy = dx / magnitude, dy / magnitude
    projection = x * dx + y * dy
    low, high = np.percentile(projection, (2, 98))
    if high - low < .25:
        return None
    endpoint = lambda value: coefficients[0] + coefficients[1] * (value * dx) + coefficients[2] * (value * dy)
    # The Figma/compiler convention is 0° left->right and positive angles go down.
    import math
    return {
        "kind": "linear",
        "angle": round(math.degrees(math.atan2(dy, dx)), 2),
        "stops": [
            {"position": 0, "color": _hex(endpoint(low))},
            {"position": 1, "color": _hex(endpoint(high))},
        ],
        "meta": {"r2": round(r2, 4), "range": round(float(np.linalg.norm(spread)), 2)},
    }


def _radial_gradient_fill(local_rgb, interior, min_range=18.0, min_r2=.91):
    """Fit the centered circular radial paint supported identically by preview/Figma.

    Off-centre, elliptical, noisy, or multi-lobed fields intentionally fail this model and
    remain raster. A high R² alone is not enough: the colour range must be meaningful and
    the fitted colour slope must change strongly from centre to edge.
    """
    _, np, _ = _deps()
    ys, xs = np.nonzero(interior)
    if len(xs) < 120:
        return None
    h, w = interior.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    normalizer = max(1.0, float(np.hypot(cx, cy)))
    radius = np.hypot(xs.astype(np.float32) - cx, ys.astype(np.float32) - cy) / normalizer
    colors = local_rgb[ys, xs].astype(np.float32)
    if len(colors) > 12000:
        pick = np.linspace(0, len(colors) - 1, 12000).astype(int)
        radius, colors = radius[pick], colors[pick]
    spread = np.percentile(colors, 95, axis=0) - np.percentile(colors, 5, axis=0)
    if float(np.linalg.norm(spread)) < min_range:
        return None
    design = np.column_stack((np.ones(len(radius)), radius))
    coefficients, _, _, _ = np.linalg.lstsq(design, colors, rcond=None)
    prediction = design @ coefficients
    total = float(np.square(colors - colors.mean(axis=0)).sum())
    if total <= 1e-6:
        return None
    r2 = 1 - float(np.square(colors - prediction).sum()) / total
    if r2 < min_r2 or float(np.linalg.norm(coefficients[1])) < min_range * .55:
        return None
    low, high = np.percentile(radius, (2, 98))
    if high - low < .35:
        return None
    endpoint = lambda value: coefficients[0] + coefficients[1] * value
    return {
        "kind": "radial",
        "stops": [
            {"position": 0, "color": _hex(endpoint(0))},
            {"position": 1, "color": _hex(endpoint(1))},
        ],
        "meta": {"r2": round(r2, 4), "range": round(float(np.linalg.norm(spread)), 2),
                 "center": [0.5, 0.5]},
    }


def _hex_to_rgb(value):
    """Parse a ``#rrggbb`` design colour into an (r, g, b) float triplet."""
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return tuple(float(int(s[i:i + 2], 16)) for i in (0, 2, 4))
    except ValueError:
        return (0.0, 0.0, 0.0)


def _multistop_linear_gradient_fill(local_rgb, interior, min_range=18.0, min_r2=.90,
                                    margin=.03, max_stops=5):
    """Fit a >2-stop linear paint when a 2-stop ramp leaves clear residual.

    A 2-stop linear gradient is a straight colour ramp; metallic / multi-hue ad plates bend
    along the axis (blue→white→orange glows, brushed-metal sweeps), and a bent ramp is
    non-monotonic per channel, so a single colour-plane regression cannot recover the axis.
    Instead this searches candidate axis angles, and for each samples the median colour at
    equally spaced positions along that axis, prunes any collinear middle stop, and scores a
    piecewise-linear fit. It only returns when the best axis's piecewise fit is BOTH good on
    its own AND materially better than the 2-stop endpoint fit — otherwise None, so the
    established 2-stop/flat path keeps handling it.
    """
    import math
    _, np, _ = _deps()
    ys, xs = np.nonzero(interior)
    if len(xs) < 200:
        return None
    h, w = interior.shape
    x = (xs.astype(np.float32) - (w - 1) / 2) / max(1.0, (w - 1) / 2)
    y = (ys.astype(np.float32) - (h - 1) / 2) / max(1.0, (h - 1) / 2)
    colors = local_rgb[ys, xs].astype(np.float32)
    if len(colors) > 12000:
        pick = np.linspace(0, len(colors) - 1, 12000).astype(int)
        x, y, colors = x[pick], y[pick], colors[pick]
    spread = np.percentile(colors, 95, axis=0) - np.percentile(colors, 5, axis=0)
    range_norm = float(np.linalg.norm(spread))
    if range_norm < min_range:
        return None
    total = float(np.square(colors - colors.mean(axis=0)).sum())
    if total <= 1e-6:
        return None

    def fit_axis(angle_deg):
        dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        t = x * dx + y * dy
        lo, hi = np.percentile(t, (2, 98))
        if hi - lo < .25:
            return None
        tn = np.clip((t - lo) / (hi - lo), 0.0, 1.0)

        def sample(pos, halfwin):
            sel = np.abs(tn - pos) <= halfwin
            if int(sel.sum()) < 12:
                sel = np.abs(tn - pos) <= halfwin * 2
            if int(sel.sum()) < 6:
                return None
            return np.median(colors[sel], axis=0)

        stops = []
        for i in range(max_stops):
            pos = i / float(max_stops - 1)
            colour = sample(pos, 0.2 if pos in (0.0, 1.0) else 0.12)
            if colour is not None:
                stops.append((pos, colour))
        if len(stops) < 3 or stops[0][0] > 0.001 or stops[-1][0] < 0.999:
            return None
        pruned = [stops[0]]
        for i in range(1, len(stops) - 1):
            p0, c0 = pruned[-1]
            p1, c1 = stops[i]
            p2, c2 = stops[i + 1]
            span = max(1e-6, p2 - p0)
            interp = c0 + (c2 - c0) * ((p1 - p0) / span)
            if float(np.linalg.norm(c1 - interp)) > 10.0:
                pruned.append(stops[i])
        pruned.append(stops[-1])
        if len(pruned) < 3:
            return None

        def piecewise(model):
            ps = np.array([p for p, _ in model], dtype=np.float32)
            cs = np.array([c for _, c in model], dtype=np.float32)
            out = np.empty((len(tn), 3), dtype=np.float32)
            for ch in range(3):
                out[:, ch] = np.interp(tn, ps, cs[:, ch])
            return out

        r2_multi = 1 - float(np.square(colors - piecewise(pruned)).sum()) / total
        r2_two = 1 - float(np.square(colors - piecewise([pruned[0], pruned[-1]])).sum()) / total
        return {"angle": float(angle_deg), "stops": pruned, "r2": r2_multi, "r2_two": r2_two}

    best = None
    for angle_deg in range(0, 180, 15):
        fit = fit_axis(angle_deg)
        if fit and (best is None or fit["r2"] > best["r2"]):
            best = fit
    if best is None or best["r2"] < min_r2 or best["r2"] < best["r2_two"] + margin:
        return None
    return {
        "kind": "linear",
        "angle": round(best["angle"], 2),
        "stops": [{"position": round(float(p), 4), "color": _hex(c)} for p, c in best["stops"]],
        "meta": {"r2": round(best["r2"], 4), "r2_two_stop": round(best["r2_two"], 4),
                 "stops": len(best["stops"]), "multistop": True,
                 "range": round(range_norm, 2)},
    }


def _gradient_reconstruction_error(rgb, gradient):
    """Mean absolute per-pixel error of an analytic linear/radial gradient vs an image.

    Used to VERIFY a fitted background gradient really explains the plate before it replaces
    a raster: a textured or photographic plate that a plane numerically half-explains still
    fails this render-back check, so the honest fidelity gate holds.
    """
    _, np, _ = _deps()
    h, w = rgb.shape[:2]
    stops = gradient.get("stops") or []
    if len(stops) < 2:
        return None
    ps = np.array([float(s.get("position", i)) for i, s in enumerate(stops)], dtype=np.float32)
    cols = np.array([_hex_to_rgb(s.get("color")) for s in stops], dtype=np.float32)
    order = np.argsort(ps)
    ps, cols = ps[order], cols[order]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if gradient.get("kind") == "radial":
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        t = np.hypot(xx - cx, yy - cy) / max(1.0, float(np.hypot(cx, cy)))
    else:
        import math
        angle = math.radians(float(gradient.get("angle", 0)))
        proj = (xx / max(1, w - 1)) * math.cos(angle) + (yy / max(1, h - 1)) * math.sin(angle)
        low, high = float(proj.min()), float(proj.max())
        t = (proj - low) / max(1e-6, high - low)
    t = np.clip(t, 0.0, 1.0)
    pred = np.empty((h, w, 3), dtype=np.float32)
    flat_t = t.ravel()
    for ch in range(3):
        pred[:, :, ch] = np.interp(flat_t, ps, cols[:, ch]).reshape(h, w)
    return float(np.abs(rgb.astype(np.float32) - pred).mean())


def extract_background_gradient(image_path, cfg=None, rgb=None):
    """Fit an editable native radial / multi-stop-linear gradient over a full clean plate.

    Radial-glow and smooth-gradient ad backgrounds are flattened to a raster today (item 2
    in the recreatability audit). When the WHOLE clean plate is explained by a single
    analytic gradient to very high fidelity, this returns a design.json ``fill`` dict so the
    background can be emitted as an editable gradient instead. Two gates keep it honest: a
    high R² analytic fit AND a render-back mean-absolute-error check on the reconstructed
    gradient. Anything textured/photographic fails and stays raster. Returns None otherwise.
    Config: ``reconstruct.background_gradient`` (defaults ON).

    ``rgb`` may be a preloaded HxWx3 uint8 array (same pixels as ``image_path``) so callers
    that already decoded the plate avoid a duplicate PNG read.
    """
    cv2, np, Image = _deps()
    bg = ((cfg or {}).get("reconstruct") or {}).get("background_gradient")
    bg = bg if isinstance(bg, dict) else {}
    if bg.get("enabled", True) is False:
        return None
    try:
        if rgb is not None:
            rgb_full = np.asarray(rgb, dtype=np.uint8)
            if rgb_full.ndim != 3 or rgb_full.shape[2] < 3:
                return None
            if rgb_full.shape[2] > 3:
                rgb_full = rgb_full[:, :, :3]
        else:
            rgb_full = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    height, width = rgb_full.shape[:2]
    if min(height, width) < int(bg.get("min_canvas_dim", 256)):
        return None
    scale = 256.0 / float(max(height, width))
    if scale < 1.0:
        small = cv2.resize(rgb_full, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb_full
    interior = np.ones(small.shape[:2], dtype=bool)
    min_range = float(bg.get("min_range", 24.0))
    radial = _radial_gradient_fill(small, interior, min_range, float(bg.get("radial_min_r2", .95)))
    linear = _gradient_fill(small, interior, min_range, float(bg.get("linear_min_r2", .95)))
    if linear:
        multi = _multistop_linear_gradient_fill(
            small, interior, min_range, float(bg.get("multistop_min_r2", .95)),
        )
        if multi:
            linear = multi
    r_radial = float(((radial or {}).get("meta") or {}).get("r2", -1))
    r_linear = float(((linear or {}).get("meta") or {}).get("r2", -1))
    gradient = radial if radial and r_radial >= r_linear + float(bg.get("radial_margin", 0.0)) else linear
    if not gradient:
        return None
    error = _gradient_reconstruction_error(small, gradient)
    if error is None or error > float(bg.get("max_mean_abs_error", 6.0)):
        return None
    gradient = dict(gradient)
    gradient["meta"] = dict(gradient.get("meta") or {}, background=True,
                            reconstruction_mae=round(error, 3))
    return gradient


def _stroke_and_interior(local_rgb, local_mask, max_width=8):
    """Detect a coherent inset stroke and return (stroke, safe_fill_pixels_mask).

    A gradient has different colours at opposite edges; it therefore fails the coherent
    border gate instead of being mislabelled as a stroke.
    """
    cv2, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 10:
        return None, local_mask
    distance = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 3)
    min_side = min(local_mask.shape)
    band = max(1, min(max_width, int(round(min_side * .12))))
    # Learn the paint from the first 1-2 pixels only.  Sampling a possible 7px band
    # would include the fill itself and make a perfectly normal 3px border look incoherent.
    probe_width = min(2, band)
    ring = local_mask & (distance > .2) & (distance <= probe_width)
    core = local_mask & (distance >= max(3, band + 1))
    if ring.sum() < 24 or core.sum() < 20:
        return None, local_mask
    edge = _robust_color(local_rgb[ring])
    edge_dist = np.linalg.norm(local_rgb.astype(np.float32) - edge, axis=2)
    coherent = ring & (edge_dist <= 14)
    if coherent.sum() / max(1, ring.sum()) < .78:
        return None, local_mask
    interior_color = _robust_color(local_rgb[core])
    if float(np.linalg.norm(edge - interior_color)) < 20:
        return None, local_mask
    width = 0
    for candidate_width in range(1, band + 1):
        candidate_ring = local_mask & (distance > .2) & (distance <= candidate_width)
        if candidate_ring.sum() and (coherent & candidate_ring).sum() / candidate_ring.sum() >= .78:
            width = candidate_width
    if not width:
        return None, local_mask
    safe_interior = local_mask & (distance >= width + 1)
    return {"color": _hex(edge), "width": int(width), "align": "INSIDE"}, safe_interior


def _shadow_effect(rgb, mask, box, geometry):
    """Find a modest drop shadow only against an otherwise flat surrounding field.

    The flat-background gate is crucial: a neighbouring photo edge should stay in the clean
    plate, not turn into a made-up Figma shadow.
    """
    cv2, np, _ = _deps()
    if geometry not in ("rect", "ellipse"):
        return None
    x, y, w, h = (int(round(box.get(key, 0))) for key in ("x", "y", "w", "h"))
    if w < 12 or h < 12:
        return None
    pad = max(5, min(18, int(round(min(w, h) * .28))))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
    if x1 - x0 < w + 4 or y1 - y0 < h + 4:
        return None
    crop = rgb[y0:y1, x0:x1]
    shape = mask[y0:y1, x0:x1] > 16
    # The outermost two-pixel border supplies the local background estimate.
    outer = np.zeros(shape.shape, dtype=bool)
    edge = min(2, max(1, min(shape.shape) // 8))
    outer[:edge, :] = outer[-edge:, :] = True
    outer[:, :edge] = outer[:, -edge:] = True
    samples = crop[outer & ~shape]
    if len(samples) < 20:
        return None
    background = _robust_color(samples)
    if float(np.max(np.std(samples.astype(np.float32), axis=0))) > 7.5:
        return None
    difference = np.linalg.norm(crop.astype(np.float32) - background, axis=2)
    best = None
    for dy in range(-pad // 2, pad // 2 + 1):
        for dx in range(-pad // 2, pad // 2 + 1):
            if dx == dy == 0:
                continue
            translation = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(shape.astype(np.uint8), translation, (shape.shape[1], shape.shape[0]),
                                     flags=cv2.INTER_NEAREST) > 0
            halo = shifted & ~shape
            count = int(halo.sum())
            if count < max(12, (w + h) // 3):
                continue
            response = float(np.mean(difference[halo]))
            # Blurred shadow should be visible but softer than a separate hard object.
            score = response * min(1.0, count / max(1, w + h))
            if response >= 9 and (best is None or score > best[0]):
                best = (score, response, dx, dy, halo)
    if best is None:
        return None
    _, response, dx, dy, halo = best
    halo_color = _robust_color(crop[halo])
    # Only accept a shadow-like halo that moves toward neutral/darker colour from its field.
    if float(np.linalg.norm(halo_color - background)) < 9:
        return None
    opacity = max(.12, min(.72, response / 255 * 1.55))
    return {
        "type": "drop-shadow", "color": _hex(halo_color), "opacity": round(opacity, 3),
        "x": int(dx), "y": int(dy), "radius": max(2, int(round(min(pad, max(abs(dx), abs(dy)) * 1.8 + 2)))),
    }


_PLATE_RESTORE_ROLES = {
    "button", "badge", "chip", "pill", "cta", "callout", "logo",
    "message-bubble", "message", "bubble",
    "banner", "ribbon", "brushstroke", "seal",
    "starburst", "price_burst", "sale_burst", "burst", "sticker",
}


def _mask_fill_fraction(local_mask) -> float:
    _, np, _ = _deps()
    if local_mask is None or local_mask.size == 0:
        return 0.0
    return float(np.mean(local_mask > 0))


def _is_hollow_stroke_ring(local_mask, max_fill: float = 0.34) -> bool:
    """True when the matte is mostly perimeter ink (outline pill / ghost button)."""
    fill = _mask_fill_fraction(local_mask)
    return 0.02 <= fill <= max_fill


def _stroke_from_ring_mask(local_rgb, local_mask, max_width: int = 8):
    """Sample stroke paint from a hollow ring matte (no solid interior required)."""
    cv2, np, _ = _deps()
    if local_mask is None or local_mask.size == 0 or not np.any(local_mask):
        return None
    ring = local_mask > 0
    if int(ring.sum()) < 24:
        return None
    distance = cv2.distanceTransform(ring.astype(np.uint8), cv2.DIST_L2, 3)
    # Width ≈ max distance from exterior through the ring band.
    width = int(max(1, min(max_width, round(float(distance.max()) * 2.0))))
    color = _robust_color(local_rgb[ring])
    return {"color": _hex(color), "width": width, "align": "CENTER"}


def _outline_shell_style(rgb, mask, box, cfg, role=None):
    """Native stroke-only plate for Biomel-style outline pills (no invented opaque fill)."""
    _, np, _ = _deps()
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    if local_mask.size == 0 or not np.any(local_mask):
        return None
    style_cfg = ((cfg.get("reconstruct") or {}).get("style_extraction") or {})
    stroke = _stroke_from_ring_mask(
        local_rgb, local_mask, int(style_cfg.get("max_stroke_width", 8)),
    )
    if stroke is None:
        return None
    w = max(1, int(round(float((box or {}).get("w", 0) or 0))))
    h = max(1, int(round(float((box or {}).get("h", 0) or 0))))
    aspect = w / max(1, h)
    role_l = str(role or "").lower().replace("-", "_")
    if 0.75 <= aspect <= 1.35 or role_l in {"badge", "seal", "starburst", "price_burst"}:
        geometry = "ellipse"
        radius = None
    else:
        geometry = "rect"
        radius = round(min(w, h) * 0.5, 2)
    return {
        "shape_kind": geometry,
        "fill": None,
        "stroke": stroke,
        "radius": radius,
        "effects": [],
        "meta": {
            "geometry": geometry,
            "stroke_outline_shell": True,
            "stroke_detected": True,
            "fill_transparent": True,
        },
    }


def _fill_plate_holes(local_mask):
    """Restore a shape-role plate whose label was carved out of its mask.

    Ownership gives glyph pixels to the editable text layer, which can leave
    text-shaped holes inside a button/badge/chip mask.  The label renders on top
    of the native plate, so the plate must be fitted (geometry, radius, fill) as
    the full primitive — a pill with a text hole is still a pill, never a donut.
    Only enclosed islands are filled; the outer silhouette is untouched, and a
    mostly-hollow ring (an outline-only ghost button) is left alone because
    filling it would invent a solid fill that is not in the source.
    Returns ``(mask, filled_pixel_count)``.
    """
    _, np, _ = _deps()
    if local_mask.size == 0 or not local_mask.any():
        return local_mask, 0
    original = int(np.count_nonzero(local_mask))
    filled = inpaint.fill_enclosed_mask_holes(local_mask.astype(np.uint8) * 255) > 0
    restored = int(np.count_nonzero(filled))
    if restored <= original or restored > original * 1.6:
        return local_mask, 0
    return filled, restored - original


def _fill_shell_text_holes(image, rgb, box):
    """Paint enclosed transparent holes in a shell crop with the plate fill colour.

    Used when a text-bearing badge falls back to IMAGE (complex scalloped seal) so the
    exported chrome is a solid plate under native TEXT, not a matte with glyph cutouts.
    Returns ``(image, filled_pixel_count)``.
    """
    _, np, Image = _deps()
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = arr[:, :, 3] > 0
    if not alpha.any():
        return image, 0
    filled_mask, n = _fill_plate_holes(alpha)
    if n <= 0:
        return image, 0
    holes = filled_mask & ~alpha
    if not holes.any():
        return image, 0
    # Sample plate colour from remaining opaque shell pixels (prefer interior).
    opaque = arr[alpha]
    fill = np.median(opaque[:, :3].astype(np.float32), axis=0).astype(np.uint8)
    arr[holes, 0:3] = fill
    arr[holes, 3] = 255
    return Image.fromarray(arr, mode="RGBA"), int(np.count_nonzero(holes))


def _extract_shape_style(rgb, mask, box, cfg, role=None, restore_plate=False,
                         stroke_outline=False):
    """Conservative native-style extraction for semantic primitive candidates."""
    _, np, _ = _deps()
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    if stroke_outline or _is_hollow_stroke_ring(local_mask):
        outline = _outline_shell_style(rgb, mask, box, cfg, role=role)
        if outline is not None:
            return outline
    style_cfg = ((cfg.get("reconstruct") or {}).get("style_extraction") or {})
    plate_holes = 0
    if ((restore_plate or str(role or "").lower() in _PLATE_RESTORE_ROLES)
            and style_cfg.get("restore_plate_mask", True) is not False):
        local_mask, plate_holes = _fill_plate_holes(local_mask)
    geometry = _simple_shape_geometry(local_mask)
    if geometry is None:
        return None
    stroke, interior = _stroke_and_interior(
        local_rgb, local_mask, int(style_cfg.get("max_stroke_width", 8))
    )
    linear = _gradient_fill(
        local_rgb, interior,
        float(style_cfg.get("gradient_min_range", 18)),
        float(style_cfg.get("gradient_min_r2", .86)),
    )
    radial = _radial_gradient_fill(
        local_rgb, interior,
        float(style_cfg.get("gradient_min_range", 18)),
        float(style_cfg.get("radial_gradient_min_r2", .91)),
    )
    # Prefer radial only when it explains materially more variance. Near-ties use the
    # simpler established linear path, preventing soft photographic lighting from being
    # mislabeled as an editable radial gradient.
    radial_margin = float(style_cfg.get("radial_gradient_r2_margin", .035))
    linear_r2 = float(((linear or {}).get("meta") or {}).get("r2", -1))
    radial_r2 = float(((radial or {}).get("meta") or {}).get("r2", -1))
    gradient = radial if radial and radial_r2 >= linear_r2 + radial_margin else linear
    # A multi-hue / metallic sweep bends along its axis, so a 2-stop plane fit either leaves
    # clear residual OR fails outright (non-monotonic per channel). Attempt the >2-stop fit
    # whenever the winner is not radial — including when the 2-stop linear was rejected — and
    # keep it only when its piecewise fit is clean and materially better than 2 stops.
    if (style_cfg.get("multistop_gradients", True) is not False
            and (gradient is None or gradient.get("kind") == "linear")):
        multi = _multistop_linear_gradient_fill(
            local_rgb, interior,
            float(style_cfg.get("gradient_min_range", 18)),
            float(style_cfg.get("multistop_gradient_min_r2", .90)),
            float(style_cfg.get("multistop_gradient_margin", .03)),
        )
        if multi:
            gradient = multi
    fill_color = _robust_color(local_rgb[interior])
    fill = gradient or {"kind": "flat", "color": _hex(fill_color)}
    radius = _corner_radius(local_mask) if geometry == "rect" else None
    effect = _shadow_effect(rgb, mask, box, geometry) if style_cfg.get("detect_shadows", True) else None
    return {
        "shape_kind": geometry,
        "fill": fill,
        "stroke": stroke,
        "radius": radius,
        "effects": [effect] if effect else [],
        "meta": {
            "geometry": geometry,
            "gradient": gradient.get("meta") if gradient else None,
            "stroke_detected": bool(stroke),
            "shadow_detected": bool(effect),
            **({"plate_holes_filled_px": plate_holes} if plate_holes else {}),
        },
    }


def _image_frame_stroke(rgb, mask, box, mask_spec, cfg):
    """Return a proven inside border for a rounded/elliptical raster frame.

    Image layers intentionally keep their source pixels, so this never edits their
    alpha or crop.  It merely promotes a uniform source-evidenced rim into the
    native Figma stroke that sits over those same edge pixels.  Plain photo edges
    fail the coherence gate in ``_stroke_and_interior`` and remain raster-only.
    """
    style_cfg = ((cfg.get("reconstruct") or {}).get("style_extraction") or {})
    if style_cfg.get("detect_image_frame_strokes", True) is False:
        return None
    kind = str((mask_spec or {}).get("kind") or "").lower()
    if kind not in {"ellipse", "circle", "rrect", "rounded_rect"}:
        return None
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    stroke, _ = _stroke_and_interior(
        local_rgb, local_mask, int(style_cfg.get("max_stroke_width", 8))
    )
    return stroke


def _infer_shape(mask, box):
    _, np, _ = _deps()
    x0, y0 = max(0, int(box.get("x", 0))), max(0, int(box.get("y", 0)))
    x1 = min(mask.shape[1], int(box.get("x", 0) + box.get("w", 0)))
    y1 = min(mask.shape[0], int(box.get("y", 0) + box.get("h", 0)))
    local = mask[y0:y1, x0:x1] > 0
    if local.size == 0:
        return "rect", 0
    fill = float(local.mean())
    corners = [local[0, 0], local[0, -1], local[-1, 0], local[-1, -1]]
    aspect = local.shape[1] / max(1, local.shape[0])
    if 0.75 <= aspect <= 1.33 and sum(bool(x) for x in corners) <= 1 and 0.55 <= fill <= 0.88:
        return "ellipse", min(local.shape) / 2
    # Missing corner pixels on an otherwise solid region indicate a rounded rectangle.
    radius = min(local.shape) * 0.12 if fill > 0.75 and sum(bool(x) for x in corners) < 4 else 0
    return "rect", round(radius, 2)


def _local_alpha(mask, box):
    """Binary (bool) crop of the canvas-space alpha for a candidate's box."""
    _, np, _ = _deps()
    x0, y0 = max(0, int(round(box.get("x", 0)))), max(0, int(round(box.get("y", 0))))
    x1 = min(mask.shape[1], int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(mask.shape[0], int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=bool)
    return mask[y0:y1, x0:x1] > 16


def _alpha_silhouette_path(mask, box):
    """Trace a single clean alpha silhouette as an SVG ``d`` string in local box pixels.

    Used for logo/brand cutouts: the mask becomes the logo's own outline so the raster fill
    can be swapped while the shape holds.  Only a single dominant contour qualifies as a
    "clean silhouette"; multi-blob artwork (e.g. multi-word lettering) returns ``None`` so
    the caller falls back to the image's own alpha rather than emitting messy geometry.
    """
    cv2, np, _ = _deps()
    local = _local_alpha(mask, box)
    if local.size == 0 or not local.any():
        return None
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = [float(cv2.contourArea(c)) for c in contours]
    total = sum(areas)
    if total <= 0:
        return None
    largest = max(range(len(contours)), key=lambda i: areas[i])
    if areas[largest] < 0.90 * total:
        return None
    contour = contours[largest]
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(approx) < 3 or len(approx) > 200:
        return None
    return "M " + " L ".join("%.1f %.1f" % (float(x), float(y)) for x, y in approx) + " Z"


def _image_mask_spec(candidate, mask, box):
    """Finalize the swappable mask spec for an image cutout.

    Honors a routing/role hint (ellipse/rrect/path) and completes its geometry from the
    alpha; when no shape was requested, infers a primitive from the alpha coverage so a
    near-square round cutout (a circular avatar) becomes an ellipse and a genuinely rounded
    cutout becomes a rounded rect.  Icons keep their own alpha silhouette.
    """
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    existing = candidate.get("mask") if isinstance(candidate.get("mask"), dict) else {}
    kind = str(existing.get("kind") or "").lower()

    # An icon's shape IS its art; keep the raster's own alpha rather than a primitive clip.
    if meta.get("vector_fallback") or role == "icon":
        return {"kind": "alpha"}

    if kind in ("ellipse", "circle"):
        return {"kind": "ellipse"}
    if kind in ("rrect", "rounded_rect"):
        radius = existing.get("radius")
        if radius is None:
            _, radius = _infer_shape(mask, box)
        return {"kind": "rrect", "radius": round(float(radius or 0), 2)}
    if kind == "path":
        path = existing.get("path") or _alpha_silhouette_path(mask, box)
        return {"kind": "path", "path": path} if path else {"kind": "alpha"}

    # No shape requested: infer a swappable primitive from the actual alpha coverage.
    # Circular product insets (white-ring crops) honor explicit circular roles/meta.
    if (
        meta.get("circular_inset")
        or role in {"circular_inset", "inset", "product_inset", "round_inset", "circle_crop"}
        or meta.get("circular")
    ):
        return {"kind": "ellipse"}
    local = _local_alpha(mask, box)
    if local.size and min(local.shape) >= 8:
        if _simple_shape_geometry(local) == "ellipse":
            return {"kind": "ellipse"}
        radius = _corner_radius(local)
        if isinstance(radius, (int, float)) and radius >= 2:
            return {"kind": "rrect", "radius": round(float(radius), 2)}
    return {"kind": "alpha"}


def _photo_shape_override(rgb, mask, box, extracted, candidate):
    """Return a mask spec when a ``shape`` region is really a photo that must stay a swappable
    image, or ``None`` to keep it a flat native primitive.

    A flat button, gradient panel or bordered card is faithfully a primitive and must NOT be
    rasterized.  Only a genuinely photographic interior — high colour dispersion that no
    flat/gradient paint explains — is reclassified, e.g. the circular Twitter avatar on ad9
    that would otherwise flatten to a solid ``#fcfcfc`` ellipse.
    """
    _, np, _ = _deps()
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    # Interactive / line chrome is always a primitive, regardless of any texture.
    if role in ("button", "cta", "chip", "divider", "bar", "chart-bar", "axis", "axis-line",
                "gridline"):
        return None
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    if local_mask.size == 0 or min(local_mask.shape) < 8:
        return None
    geometry = (extracted or {}).get("shape_kind") or _simple_shape_geometry(local_mask)
    if geometry not in ("ellipse", "rect"):
        return None
    # A clean gradient/solid surface is a design fill, not a photo.
    if extracted and (extracted.get("fill") or {}).get("kind") in ("linear", "radial"):
        return None
    pixels = local_rgb[local_mask]
    if pixels.shape[0] < 60:
        return None
    dispersion = float(np.max(np.std(pixels.astype(np.float32), axis=0)))
    if dispersion < PHOTO_SHAPE_MIN_STD:
        return None
    if geometry == "ellipse":
        return {"kind": "ellipse"}
    radius = (extracted or {}).get("radius")
    if not isinstance(radius, (int, float)) or radius <= 0:
        radius = _corner_radius(local_mask)
    spec = {"kind": "rrect", "radius": 0.0}
    if isinstance(radius, (int, float)) and radius > 0:
        spec["radius"] = round(float(radius), 2)
    return spec


def _paths_to_svg(paths, width, height):
    body = []
    for path in paths:
        fill = path.get("fill") or "#000000"
        winding = path.get("windingRule") or "nonzero"
        body.append(f'<path d="{path.get("d", "")}" fill="{fill}" fill-rule="{winding}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>")


def _write_asset(image, assets_dir, candidate_id):
    raw = image.tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:10]
    name = f"{candidate_id}_{digest}.png"
    path = os.path.join(assets_dir, name)
    image.save(path)
    return f"assets/{name}"


def _badge_chip_clean(rgb, candidates: list, cfg: dict):
    """Rebuild every promoted offer badge's chrome with its OWN ink lifted off, chip-locally.

    Returns ``(chip_plate, chip_mask, plate_chip_regions, records)``: a canvas-sized copy of
    the source whose badge chips are clean, the mask of exactly the pixels rebuilt, the
    footprints of cleaned PLATE-HOSTED chips only (the caller withholds those from the
    generative union; an element-hosted chip's plate hole is legitimate and must inpaint),
    and a per-chip ledger.

    THE ARCHITECTURE THIS FIXES (both halves measured on 013, not assumed):

    1. FUSION. The canvas removal union merges every hole before a backend is chosen, so the
       61%-OFF disc's ink and the pouch cutout -- which touch at x=348-385 -- become ONE
       component, box=(0,676,1080,769), area=684,685px, spanning the full canvas width.
       ``_flat_hole_fill`` measures THAT component's ring, finds it 18% uniform against the
       85% it needs, correctly refuses, and Flux smears a slab across the disc.
    2. ANTI-ALIASING. Cropping to the chip is necessary but NOT sufficient. Even unfused and
       chip-local, ``text_ink_mask`` is glyph-tight, so 2-3px of cream-into-green halo sits
       just OUTSIDE the hole and poisons the ring the fill measures: 0/15 glyph components
       route flat at zero dilation, 5/15 at the pipeline's default 2px. The rim has to be
       swallowed into the hole (``inpaint.chip_local_ink_removal``) before any of this works.

    With both: 4 of 6 components route to a flat #057C3E fill and the rest to chip-local
    Big-LaMa -- a solid disc instead of a slab, and never a generative pass over a canvas band.

    The result feeds BOTH consumers, which is the point of doing it here, once, on the source:
      * ELEMENT-HOSTED chips (101/131) -- ``_source_rgba`` slices cutouts straight from these
        pixels, so a badge cutout drawn over the plate carries clean chrome. Slicing the raw
        original instead is what put the ink back on top of its own removal (double render).
      * PLATE-HOSTED chips (013) -- SAM never emitted an element for the disc, so the disc IS
        the plate; the caller hands these pixels to the inpaint as its source and holds the
        rebuilt mask out of the generative union.
    """
    _, np, _ = _deps()
    chips: dict = {}
    for c in candidates:
        meta = c.get("meta") or {}
        chip_id = meta.get("offer_badge_id")
        if not (chip_id and meta.get("badge_offer_lockup") and meta.get("badge_chip_box")):
            continue
        entry = chips.setdefault(chip_id, {
            "chip_box": meta.get("badge_chip_box"),
            "host_id": meta.get("badge_chip_host"),
            "ids": [], "boxes": [],
        })
        entry["ids"].append(c.get("id"))
        entry["boxes"].append(c.get("box") or {})

    plate = np.asarray(rgb, dtype=np.uint8).copy()
    chip_mask = np.zeros(plate.shape[:2], dtype=np.uint8)
    chip_regions = np.zeros(plate.shape[:2], dtype=np.uint8)
    records: list = []
    for chip_id, entry in chips.items():
        ink = np.zeros(plate.shape[:2], dtype=np.uint8)
        for box in entry["boxes"]:
            ink = np.maximum(ink, np.asarray(
                inpaint.text_ink_mask(rgb, box, allow_box_fallback=False), dtype=np.uint8))
        cleaned, filled, info = inpaint.chip_local_ink_removal(
            rgb, entry["chip_box"], ink, cfg,
        )
        record = {"chip_id": chip_id, "ids": list(entry["ids"]),
                  "host_id": entry.get("host_id"), **{
                      k: v for k, v in (info or {}).items() if k != "chip_backend_route"}}
        if cleaned is None:
            # Honest fallback: the caller leaves this chip's text baked and says why.
            record["cleaned"] = False
            records.append(record)
            continue
        cx, cy, cw, ch = info["chip_box"]
        plate[cy:cy + ch, cx:cx + cw] = cleaned
        chip_mask[cy:cy + ch, cx:cx + cw] = np.maximum(
            chip_mask[cy:cy + ch, cx:cx + cw], (filled > 0).astype(np.uint8) * 255)
        if not entry.get("host_id"):
            chip_regions[cy:cy + ch, cx:cx + cw] = 255
        record["cleaned"] = True
        # Publish the proof: this line's ink is gone from every surface downstream reads, so
        # the single-ownership audit must not "clean" it a second time. That audit erases
        # anything inside the text's BOX that differs from the box's ring median, which is
        # only safe when the box holds nothing but plate and the text's own ink. 013's
        # '+ FREE GIFTS' box runs to x=385 and the pouch starts at x=348, so with the ink
        # already gone the only thing left for it to find IS the pouch — which it painted
        # over with disc green, a hard rectangle across the bag (badge ssim stuck at 0.55).
        _members = set(entry["ids"])
        for c in candidates:
            if c.get("id") in _members:
                c.setdefault("meta", {})["badge_chip_cleaned"] = True
            elif entry.get("host_id") and c.get("id") == entry["host_id"]:
                # The chrome host now carries NO ink, so its raster may own its whole matte
                # again (see the ownership rescue in reconstruct()).
                c.setdefault("meta", {})["badge_chip_host_cleaned"] = True
        records.append(record)
    return plate, chip_mask, chip_regions, records


def reconstruct(image_path: str, ocr: dict, candidates: list, run_dir: str,
                cfg: Optional[dict] = None) -> dict:
    cv2, np, Image = _deps()
    cfg = cfg or {}
    rcfg = cfg.get("reconstruct") or {}
    os.makedirs(run_dir, exist_ok=True)
    assets_dir = os.path.join(run_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    # Safety net: if merge skipped demotion, drop whole-plot rasters that the same
    # chart_group already covers with native primitives (see docs/DIAGRAM-EDITABILITY.md).
    candidates = prefer_decomposed_charts(candidates)

    canonical = deduplicate(candidates, float(rcfg.get("dedup_iou", 0.86)))
    text_shells_promoted = _promote_ocr_overlapping_shells(
        canonical, cfg, canvas={"w": w, "h": h},
    )
    photo_policy = (((cfg.get("scene") or {}).get("preset") or {}).get("photo_regions") or {})
    if photo_policy.get("suppress_descendants", True):
        canonical = _suppress_baked_raster_text(
            canonical, float(rcfg.get("raster_text_containment", .90)),
        )
    canonical = _suppress_comparison_column_labels(canonical, cfg)
    canonical, flattened_scene_artwork = _flatten_photo_scene(canonical, cfg)
    canonical, plate_fragments_suppressed = _suppress_plate_boundary_fragments(canonical, cfg)
    canonical, engagement_underlays_suppressed = _suppress_engagement_underlay_shells(canonical, cfg)
    ocr_lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    masks = {}
    for candidate in canonical:
        if (candidate.get("target") != "drop"
                or (candidate.get("meta") or {}).get("removal_required")):
            mask = _candidate_mask(candidate, rgb, run_dir, ocr_lines, cfg)
            if candidate.get("target") == "text":
                # Ghost-text invariant: an emitted editable text layer whose source
                # pixels stay in the plate produces double text in the render.
                mask = _ensure_text_removal_coverage(candidate, mask, rgb, cfg)
            masks[candidate.get("id")] = mask

    # Chip-local badge ink removal, BEFORE anything reads source pixels. ``chip_rgb`` is the
    # source with every promoted badge's chrome rebuilt clean; it is what the raster cutouts
    # and the plate are both built from. ``rgb`` stays the pristine original and remains the
    # reference for the plate-integrity gate, so the two never get confused.
    chip_rgb, chip_clean_mask, chip_regions, chip_records = _badge_chip_clean(rgb, canonical, cfg)
    if chip_records:
        print(f"[chip] badge chips: {len(chip_records)} "
              f"({sum(1 for r in chip_records if r.get('cleaned'))} cleaned)")

    # Front-to-back ownership is diagnostic and makes overlapping raster assets exclusive.
    # Text/icons are frontmost; smaller nested layers win over broad photo regions.
    # A background plate becomes no foreground layer at all.  Excluding it *before*
    # ownership matters: otherwise a large Qwen/background observation can claim every pixel
    # and leave the real product/icon cutout with an empty alpha channel.
    def _ownership_priority(candidate):
        target = candidate.get("target")
        meta = candidate.get("meta") or {}
        role = str(meta.get("role") or "").lower()
        # VLM/SAM may have classified an element as UI chrome, an overlay, or scene
        # content.  Keep that explicit top-down contract ahead of the generic target
        # heuristic so a verified badge/callout leader cannot be swallowed by the
        # photo/card it sits on.  Unknown candidates retain the old conservative order.
        band = str(meta.get("z_band") or "").lower()
        band_priority = {
            "chrome": 40, "ui": 40,
            "overlay": 30, "foreground": 30,
            "content": 20, "scene": 20,
            "background": 0, "plate": 0,
        }.get(band)
        if band_priority is not None:
            return band_priority
        # Semantic foreground cutouts must claim their pixels before a broad
        # scene/photo region.  z is often unavailable for SAM/residual-only
        # runs, so relying on z alone silently reverses ownership.
        if target == "text":
            return 4
        if target == "icon" or role in {
            "arrow", "callout_leader", "leader", "leader_line", "connector",
        }:
            return 3
        if target in ("shape", "image") and role in ("product", "person", "foreground", "cutout"):
            return 2
        return 1

    front = sorted(
        (c for c in canonical if c.get("target") != "drop" and not (
            c.get("target") == "image" and _is_background_plate(c, w, h)
        )),
        key=lambda c: (
            _ownership_priority(c),
            float(c.get("z", 0)),
            -c.get("box", {}).get("w", 0) * c.get("box", {}).get("h", 0),
        ), reverse=True,
    )
    ownership = np.zeros((h, w), dtype=np.uint16)
    owner_index = {}
    owner_number = {}
    for index, candidate in enumerate(front, start=1):
        cid = candidate.get("id")
        owner_index[str(index)] = cid
        owner_number[cid] = index
        available = (masks[cid] > 0) & (ownership == 0)
        ownership[available] = index

    # Materialize native shapes, vectors, and isolated alpha rasters.
    updated = []
    vector_ok = vector_fallback = 0
    for candidate in canonical:
        c = dict(candidate)
        c["meta"] = dict(c.get("meta") or {})
        target = c.get("target")
        cid = c.get("id")
        is_plate = _is_background_plate(c, w, h)
        if is_plate and target == "image":
            c["target"] = "drop"
            c["meta"]["keep_in_background"] = True
            updated.append(c)
            continue
        if target in ("drop", "text"):
            updated.append(c)
            continue
        if (c.get("meta") or {}).get("native_decoration"):
            # Already an evidence-backed SVG/path. Preserve it verbatim; style
            # extraction from its thin removal mask would flatten or reclassify it.
            updated.append(c)
            continue
        mask = masks.get(cid)
        if mask is None:
            updated.append(c)
            continue
        if target == "shape":
            # Do not overwrite upstream paint facts.  This fills only the gaps left by
            # segmentation/Qwen and tags every inference for later QA/debugging.
            is_text_shell = bool(
                (c.get("meta") or {}).get("text_bearing_shell")
                or (c.get("meta") or {}).get("plate_shell")
            )
            stroke_outline = bool(
                (c.get("meta") or {}).get("stroke_outline_shell")
                or (c.get("meta") or {}).get("stroke_only")
            )
            extracted = _extract_shape_style(
                rgb, mask, c.get("box", {}), cfg,
                role=(c.get("meta") or {}).get("role"),
                restore_plate=is_text_shell and not stroke_outline,
                stroke_outline=stroke_outline or (
                    is_text_shell and _is_hollow_stroke_ring(
                        _local_shape_pixels(rgb, mask, c.get("box", {}))[1]
                    )
                ),
            )
            # Text-bearing chrome is never a photo avatar — skip photo reclass so a
            # textured olive brushstroke cannot flatten to a wrong ellipse/rect.
            photo_mask = None if is_text_shell else _photo_shape_override(
                rgb, mask, c.get("box", {}), extracted, c
            )
            if photo_mask is None:
                if extracted:
                    c["shape_kind"] = c.get("shape_kind") or extracted["shape_kind"]
                    if extracted.get("fill") is not None:
                        c["fill"] = c.get("fill") or extracted["fill"]
                    elif extracted.get("meta", {}).get("fill_transparent"):
                        c["fill"] = None
                        c["meta"]["stroke_outline_shell"] = True
                    c["stroke"] = c.get("stroke") or extracted["stroke"]
                    if not c.get("effects") and extracted["effects"]:
                        c["effects"] = extracted["effects"]
                    if c.get("radius") is None and extracted["radius"] not in (None, 0):
                        c["radius"] = extracted["radius"]
                    c["meta"]["style_extraction"] = extracted["meta"]
                    updated.append(c)
                    continue
                if is_text_shell and not stroke_outline:
                    # Irregular brushstroke / starburst: prefer a solid-fill path silhouette;
                    # if contour fidelity fails, honest alpha CHIP of the stroke only
                    # (never a full-ad slice, never a bounding rect that invents area).
                    path_d = _alpha_silhouette_path(mask, c.get("box", {}))
                    fill_color = _dominant_fill(rgb, mask, c.get("box", {}))
                    if path_d:
                        c["shape_kind"] = "path"
                        c["path"] = path_d
                        c["fill"] = c.get("fill") or {"kind": "flat", "color": fill_color}
                        c["meta"]["irregular_shell_path"] = True
                        updated.append(c)
                        continue
                    c["meta"]["shell_raster_chip"] = True
                    c["meta"]["vector_fallback"] = True
                    c["meta"]["reclassified"] = "shape->shell-chip"
                    c["mask"] = {"kind": "alpha"}
                    target = c["target"] = "image"
                    # fall through to image materialization (holes filled below)
                elif is_text_shell and stroke_outline:
                    # Hollow outline plate: never invent an opaque fill over the photo.
                    outline = _outline_shell_style(
                        rgb, mask, c.get("box", {}), cfg,
                        role=(c.get("meta") or {}).get("role"),
                    )
                    if outline:
                        c["shape_kind"] = outline["shape_kind"]
                        c["fill"] = None
                        c["stroke"] = c.get("stroke") or outline["stroke"]
                        if c.get("radius") is None and outline.get("radius") not in (None, 0):
                            c["radius"] = outline["radius"]
                        c["meta"]["stroke_outline_shell"] = True
                        c["meta"]["style_extraction"] = outline["meta"]
                        updated.append(c)
                        continue
                    # Last resort: alpha chip of the stroke ring only (no plate fill).
                    c["meta"]["shell_raster_chip"] = True
                    c["meta"]["stroke_outline_shell"] = True
                    c["meta"]["vector_fallback"] = True
                    c["meta"]["reclassified"] = "shape->outline-chip"
                    c["mask"] = {"kind": "alpha"}
                    target = c["target"] = "image"
                else:
                    kind, radius = _infer_shape(mask, c.get("box", {}))
                    c["shape_kind"] = c.get("shape_kind") or kind
                    c["fill"] = c.get("fill") or {
                        "kind": "flat", "color": _dominant_fill(rgb, mask, c.get("box", {}))
                    }
                    if radius and kind == "rect" and c.get("radius") is None:
                        c["radius"] = radius
                    updated.append(c)
                    continue
            else:
                # Photographic region (e.g. the ad9 circular avatar): deliver the real pixels as
                # a swappable IMAGE clipped by the detected primitive, not a flattened solid fill.
                if extracted and extracted.get("effects") and not c.get("effects"):
                    c["effects"] = extracted["effects"]
                if extracted and extracted.get("stroke") and not c.get("stroke"):
                    c["stroke"] = extracted["stroke"]
                if extracted:
                    c["meta"]["style_extraction"] = extracted["meta"]
                c["meta"]["reclassified"] = "shape->image"
                c["meta"]["photo_shape"] = True
                c["mask"] = photo_mask
                target = c["target"] = "image"
                # fall through to the image materialization below

        owned = (ownership == owner_number.get(cid, 0)).astype(np.uint8) * 255
        # Holes introduced by front-to-back ownership are intentional: a child
        # logo/badge sits in that exact region and is rendered independently.
        # Preserve this provenance so QA can distinguish it from a broken SAM
        # matte hole.
        if np.any((mask > 0) & (owned == 0)):
            c["meta"]["ownership_cutout"] = True
        # A SAM-confirmed semantic matte may have intentional enclosed negative space
        # (the counter of a product/logo mark).  It is a valid Figma alpha-masked crop,
        # unlike residual-only fragments, which are filtered above.  QA still rejects
        # generic product mattes with holes, so this cannot hide an unverified mask bug.
        if _verified_semantic_mask(c["meta"]):
            c["meta"]["ownership_cutout"] = True
            c["meta"]["mask_provenance"] = "sam3-verified"
        # A lower-priority fallback that owns no pixels would export an empty
        # Figma image layer.  It is already faithfully present in its retained
        # owner, so drop it instead of producing a misleading layer.
        substitution = c["meta"].get("substitution") if isinstance(c["meta"].get("substitution"), dict) else {}
        is_text_fallback = bool(c["meta"].get("fallback") and substitution.get("from") == "text")
        role = str(c["meta"].get("role") or "").lower()
        # F1: a DISTINCT, independently-verified semantic cutout (a product/person/logo
        # with its own high-confidence SAM matte) must not be dropped merely because a
        # DIFFERENT entity's owner box was ranked ahead of it and claimed all its pixels
        # first. A product sitting on a panel is not "part of" the panel: when the panel
        # carries e.g. z_band="chrome" it out-ranks the product in ownership and steals
        # its pixels, and the product then vanishes (benchmark 002: whey bag + jars
        # erased). Re-claim the cutout's own matte so it survives as an editable/swappable
        # image; the front owner keeps its raster too (identical pixels, same position ->
        # no visible double), so this only ever adds a layer, never a hole.
        _DISTINCT_CUTOUT_ROLES = {
            "product", "person", "foreground", "cutout", "avatar", "profile",
            "profile_photo", "logo", "brand", "wordmark",
            "arrow", "callout_leader", "leader", "leader_line", "connector",
            "underline", "strikethrough", "annotation",
        }
        mask_px = int(np.count_nonzero(mask))
        owned_px = int(np.count_nonzero(owned))
        ownership_fraction = owned_px / max(1, mask_px)
        if (not is_text_fallback
                and role in _DISTINCT_CUTOUT_ROLES
                and _verified_semantic_mask(c["meta"])
                and ownership_fraction < .55):
            owned = (mask > 0).astype(np.uint8) * 255
            c["meta"]["ownership_rescued"] = "distinct-verified-cutout"
            c["meta"]["ownership_fraction_before_rescue"] = round(ownership_fraction, 4)
            c["meta"]["ownership_cutout"] = True
        # Same rescue, for list-row ✓/✗/? glyphs. A checklist card carries z_band=chrome,
        # so it out-ranks its own row glyphs in ownership and claims every glyph pixel;
        # the glyph then drops as "fully-contained-in-foreground-owner" on the premise
        # that the owner's raster still carries it. For these cards that premise is false
        # — they ship as a flat fill / native `__shell`, or fold into the plate that the
        # card paints over — so the drop deleted all nine of 101's marks and fifteen of
        # 066's. Glyphs always ship as pixel-exact raster chips: re-claim the chip's own
        # matte. Identical pixels at the same position, so this adds a layer, never a hole.
        elif (not is_text_fallback
                and _is_cv_list_glyph(c["meta"])
                and ownership_fraction < .55):
            owned = (mask > 0).astype(np.uint8) * 255
            c["meta"]["ownership_rescued"] = "cv-list-glyph"
            c["meta"]["ownership_fraction_before_rescue"] = round(ownership_fraction, 4)
            c["meta"]["ownership_cutout"] = True
        # Same rescue, for a badge chip whose ink was lifted chip-locally. Its editable text
        # sits ON this chrome and is frontmost, so ownership hands the text the glyph pixels
        # and the chrome keeps a matte full of holes — 101's 87x87 teal disc came out of the
        # ownership pass with 33 opaque pixels of 7569, blank enough that materialization
        # swapped it for a slice of the plate, which by then had the disc punched out of it.
        # A white hole with the offer missing.
        #
        # The premise of that punch — "these pixels are the text, and the text re-renders
        # natively, so the raster must not also carry them" — is FALSE here, and provably so:
        # _badge_chip_clean already removed this chip's ink and stamps the host only when it
        # succeeded. What sits under the glyphs now is the disc's own chrome. Handing it back
        # adds no ink and duplicates nothing; it just stops the chrome being deleted.
        elif not is_text_fallback and c["meta"].get("badge_chip_host_cleaned"):
            owned = (mask > 0).astype(np.uint8) * 255
            c["meta"]["ownership_rescued"] = "chip-cleaned-badge-chrome"
            c["meta"]["ownership_fraction_before_rescue"] = round(ownership_fraction, 4)
        # Do retain broad photo frames for the layout regression contract, but
        # never export an empty semantic cutout (product/person/icon) merely
        # because a frontmost owner already claims all of its pixels.
        if not np.any(owned) and (is_text_fallback or role not in {"photo", "image"}):
            c["target"] = "drop"
            c["meta"]["keep_in_background"] = True
            c["meta"]["suppression_reason"] = "fully-contained-in-foreground-owner"
            updated.append(c)
            continue
        # Cutouts are sliced from the CHIP-CLEANED source, not the raw original. This is the
        # whole fix for element-hosted badges (101's c_E005, 131's c_E003): _source_rgba
        # slices a badge cutout straight out of the source and draws it OVER the plate, so
        # slicing the original hands back the very ink the plate just removed -- the double
        # render. Outside badge chips chip_rgb is byte-identical to rgb, and a neighbour that
        # merely overlaps a chip (013's pouch) only ever takes pixels its own matte covers.
        image = _source_rgba(c, chip_rgb, mask, run_dir)
        image = _apply_owned_alpha(image, owned, c.get("box", {}))
        # Text-bearing badge/button shells: ownership punches glyph holes into the
        # chrome matte. Fill those enclosed holes with the shell's plate colour so
        # native TEXT paints over a solid plate (not a donut that leaks the plate).
        # Only shells flagged by merge (never ordinary logos — counters must survive).
        if (
            (c["meta"].get("text_bearing_shell") or c["meta"].get("plate_shell"))
            and not c["meta"].get("stroke_outline_shell")
        ):
            image, filled_px = _fill_shell_text_holes(image, rgb, c.get("box") or {})
            if filled_px:
                c["meta"]["shell_text_holes_filled_px"] = filled_px
        comparison_columns = _split_comparison_frame(c, image, assets_dir, cfg)
        if comparison_columns:
            # Keep the broad source owner only as the inpaint/removal observation.  The
            # two children above are the actual Figma layers, each with its own crop.
            c["target"] = "drop"
            c["meta"]["removal_required"] = True
            c["meta"]["suppression_reason"] = "split-into-before-after-columns"
            updated.append(c)
            updated.extend(comparison_columns)
            continue
        # Keep the exact reconstructed crop even when the editable vector passes
        # the fidelity gate.  It is the deterministic preview fallback for SVGs
        # that CairoSVG cannot paint (or paints fully transparent).
        raster_src = _write_asset(image, assets_dir, cid)
        if target == "icon":
            role = (c.get("meta") or {}).get("role")
            # Harness repairs are target-scoped.  A bad trace on one icon must not flatten
            # every otherwise-good vector in the run.
            vector_cfg = cfg
            repair_target = ((cfg.get("harness") or {}).get("target_id"))
            if repair_target and repair_target != cid and (cfg.get("vectorize") or {}).get("force_raster_fallback"):
                vector_cfg = dict(cfg)
                vector_cfg["vectorize"] = dict(cfg.get("vectorize") or {})
                vector_cfg["vectorize"].pop("force_raster_fallback", None)
            traced = vectorize.vectorize_crop(np.asarray(image), vector_cfg, role=role)
            c["meta"]["vectorize"] = {
                k: traced.get(k) for k in ("ok", "engine", "score", "note")
            }
            if traced.get("ok"):
                c["paths"] = traced["paths"]
                c["svg"] = traced.get("svg") or _paths_to_svg(traced["paths"], image.width, image.height)
                c["src"] = raster_src
                first_path = traced["paths"][0] if traced["paths"] else {}
                fill_value = first_path.get("fill", "#000000")
                # A hand-drawn annotation (marker underline / strikethrough / connector /
                # arrow shaft) traces to a stroke-only path (fill="none"). Keep it as an
                # editable, recolorable Figma stroke rather than stamping a bogus flat fill;
                # propagate its stroke style (colour + approx width) so the vector is
                # movable AND recolorable in Figma. Filled marks (badge/arrow head) behave
                # exactly as before.
                if isinstance(fill_value, str) and fill_value.strip().lower() == "none":
                    c.pop("fill", None)
                else:
                    c["fill"] = {"kind": "flat", "color": fill_value}
                stroke_spec = first_path.get("stroke") if isinstance(first_path, dict) else None
                if stroke_spec and not c.get("stroke"):
                    c["stroke"] = dict(stroke_spec)
                    c["meta"]["annotation_stroke"] = {
                        "color": stroke_spec.get("color"),
                        "width": stroke_spec.get("width"),
                    }
                vector_ok += 1
                updated.append(c)
                continue
            # Active Big-LaMa/inpainting is independent from the optional icon
            # vector fidelity gate.  A complex icon may legitimately fail the
            # path-count/colour gate; retain it as an explicit raster fallback
            # so the batch can finish and QA can report the degradation.
            if bool(((cfg.get("vectorize") or {}).get("require_active", False))):
                raise RuntimeError(
                    f"vectorization required for icon {cid}, but no gated trace was available: "
                    f"{traced.get('note', 'unknown vectorization failure')}"
                )
            c["target"] = "image"
            c["meta"]["vector_fallback"] = True
            vector_fallback += 1

        c["src"] = raster_src
        # Swappable mask shape: ellipse for round avatars, rounded-rect for cards, path for
        # a clean logo silhouette; irregular cutouts keep their own alpha.
        final_mask = _image_mask_spec(c, mask, c.get("box", {}))
        if target == "image" and not c.get("stroke"):
            frame_stroke = _image_frame_stroke(rgb, mask, c.get("box", {}), final_mask, cfg)
            if frame_stroke:
                c["stroke"] = frame_stroke
                c["meta"]["image_frame_stroke"] = {
                    "source": "uniform-border-ring", "width": frame_stroke["width"],
                }
        c["mask"] = final_mask
        updated.append(c)

    removal = []
    mask_rejected = 0
    removal_capped = 0
    scene_vlm_required = bool((((cfg.get("vlm") or {}).get("scene_text") or {}).get("enabled")))
    max_text_mask_fraction = float(rcfg.get("max_text_mask_canvas_fraction", 0.035))
    # Bound how much of the canvas any single OPAQUE raster may claim as a removal hole.
    # A photo/image layer re-renders over its own footprint, so inpainting the plate
    # underneath it is invisible work; admitting a quarter-plus of the canvas as one
    # removal element instead nukes the plate (002 c_E003 — a low-confidence residual-CC
    # "shape->image" — claimed 34.5% and destroyed the background). Keep those original
    # pixels; the opaque raster still ships on top and shows the same pixels.
    max_candidate_removal_fraction = float(rcfg.get("max_candidate_removal_fraction", 0.25))
    max_candidate_removal_confidence = float(rcfg.get("max_candidate_removal_confidence", 0.55))
    for c in updated:
        if c.get("target") == "drop" and not (c.get("meta") or {}).get("removal_required"):
            continue
        # Oversized residual shells (merge rejected them as geometric text-shells) are
        # NOT independent objects: they are the negative space around copy or a whole
        # card/panel that re-renders as its own flat plate slice. Every real element on
        # top (text, product, icon) is a separate candidate that removes its OWN ink, so
        # inpainting the entire shell footprint is redundant — and it is exactly what
        # destroys the plate: the card interiors turn to Big-LaMa mush and any kept
        # content inside them is erased (101: E001/E002 half-cards claimed ~75% of the
        # removal union; the whole plate was inpainted and the product tubes were wiped).
        # Keep the source pixels: skip the removal observation entirely. A full-canvas
        # shell that would ALSO ship as a top image is a plate duplicate that paints over
        # the real cutouts underneath it — drop it to a plate passthrough so it neither
        # inpaints nor re-renders (101: E000 shipped as a "Photo" over the tube slices).
        if (c.get("meta") or {}).get("text_shell_rejected") == "oversized-residual-shell":
            c["meta"]["removal_skipped"] = "oversized-residual-shell"
            if c.get("target") == "image":
                c["target"] = "drop"
                c.pop("src", None)
                c.pop("mask", None)
                c["meta"]["plate_passthrough"] = True
                c["meta"]["raster_fallback"] = "oversized-residual-shell-plate-passthrough"
                # Declare the reason in the vocabulary scene_intent reconciliation reads
                # (`_is_explicit_suppression`: keep_in_background / suppression_reason /
                # removal_required). This drop is DELIBERATE and reasoned — the plate owns
                # these pixels — but it only recorded `removal_skipped`/`plate_passthrough`,
                # which the reconciler cannot see. An undeclared drop of a PLANNED id read
                # as "reconstruction made a structural decision the frozen intent cannot
                # explain", raising SceneIntentError -> the whole structure-first tree was
                # discarded for the legacy layout and the run took a hard
                # `structure-unavailable` (002 c_E003, 101 E000/E001/E002). The pixels are
                # kept in the background either way; saying so keeps the structure.
                c["meta"]["keep_in_background"] = True
                c["meta"]["suppression_reason"] = "oversized-residual-shell"
            continue
        if _keeps_underlay(c):
            # Insets, avatars, callout chrome and other true overlays sit above a valid
            # retained photo/plate. They remain editable owners, but must not punch a
            # generative-inpaint hole into that underlay.
            continue
        box = c.get("box", {})
        area_frac = box.get("w", 0) * box.get("h", 0) / max(1, w * h)
        candidate_mask = masks.get(c.get("id"))
        mask_fraction = (float(np.count_nonzero(candidate_mask)) / max(1, w * h)
                         if candidate_mask is not None else 0.0)
        # Residual-CC display glyphs (088 "SALE"/"21%") that OCR could not read ship as
        # pixel-exact baked raster slices.  Front-to-back ownership then hands each slice
        # only PART of its own ink (a boxy sub-region), so removing the full glyph ink from
        # the plate and re-pasting the fragmented slices leaves grey plate showing THROUGH
        # the letterforms — a broken "S" missing its top, an "E" cut by a seam.  The slice
        # pixels ARE the original pixels at the original position, so on a flat plate keeping
        # the ink is lossless where a slice covers and fills the ownership gap where it does
        # not.  Withhold removal for these; the plate keeps the exact glyphs and the slices
        # still ship on top for editability.  Scoped to flat plates + under-covered ownership
        # so photographic cutouts (which must vacate the plate to swap cleanly) are untouched.
        _glyph_frac = _skip_removal_for_flat_residual_glyph(
            c, candidate_mask, ownership, owner_number, rgb, rcfg)
        if _glyph_frac is not None:
            _stamp_residual_glyph_overlay(c, _glyph_frac)
            continue
        if c.get("target") == "text":
            ownership_decision = (c.get("meta") or {}).get("ownership_decision")
            text_meta = c.get("meta") or {}
            semantic_role = str(text_meta.get("semantic_role") or text_meta.get("role") or "").lower()
            # The VLM is a conservative ownership judge, not a prerequisite for
            # an explicitly detected marketing overlay/CTA.  A timeout or
            # disagreement must not erase the editable CTA inside a native
            # button while leaving an otherwise correct ad as a raster plate.
            explicit_overlay = bool(text_meta.get("overlay_text") or text_meta.get("removal_required"))
            safe_marketing_copy = semantic_role in {"cta", "headline", "eyebrow", "offer"}
            rejection_reason = None
            if scene_vlm_required and not ownership_decision and not (explicit_overlay or safe_marketing_copy):
                rejection_reason = "missing-ownership-decision"
            elif ownership_decision and ownership_decision.get("action") != "recreate":
                rejection_reason = "ownership-does-not-allow-recreate"
            elif scene_vlm_required and mask_fraction > max_text_mask_fraction:
                rejection_reason = "text-mask-too-broad"
            if rejection_reason:
                c["target"] = "drop"
                c.setdefault("meta", {})["keep_in_background"] = True
                c["meta"]["raster_fallback"] = "mask-approval-rejected"
                c["meta"]["mask_approval"] = {
                    "accepted": False, "reason": rejection_reason,
                    "mask_fraction_canvas": round(mask_fraction, 6),
                }
                mask_rejected += 1
                continue
            c.setdefault("meta", {})["mask_approval"] = {
                "accepted": True, "reason": "ownership-and-geometry-approved",
                "mask_fraction_canvas": round(mask_fraction, 6),
            }
        # A full-canvas raster is the plate itself. Everything else is removed from the plate.
        is_background = bool(c.get("meta", {}).get("role") == "background" or area_frac > 0.92)
        # Per-candidate removal cap: one bad element must not claim most of the canvas.
        # Only opaque rasters qualify — a photo/image layer re-renders over its own
        # footprint, so the plate underneath it never shows and need not be inpainted.
        # A genuine large foreground (product/person/photo) still gets a clean plate so
        # it stays a swappable asset; the cap targets the spurious-blob signature that
        # actually destroys the plate: a big generic-role raster with LOW detector
        # confidence (002 c_E003 — a residual-CC "shape->image" at conf 0.405 claimed
        # 34.5%). Explicit removal_required overlays and background plates are exempt.
        _cap_meta = c.get("meta") or {}
        _cap_role = str(_cap_meta.get("role") or "").lower()
        _cap_conf = _cap_meta.get("confidence")
        if (c.get("target") == "image" and not is_background
                and mask_fraction > max_candidate_removal_fraction
                and not _cap_meta.get("removal_required")
                and _cap_role not in _CAP_EXEMPT_ROLES
                and isinstance(_cap_conf, (int, float))
                and float(_cap_conf) < max_candidate_removal_confidence):
            c.setdefault("meta", {})["keep_in_background"] = True
            c["meta"]["removal_capped"] = {
                "mask_fraction_canvas": round(mask_fraction, 6),
                "cap": max_candidate_removal_fraction,
                "confidence": float(_cap_conf),
                "reason": "low-confidence-opaque-raster-exceeds-removal-cap",
            }
            # A capped raster is PLATE-OWNED, full stop. It must not also ship as a
            # layer: re-emitting it (directly, or as a group-host "__hostbg" slice cut
            # by build_design_json from candidate src) bakes the plate + every extracted
            # element back into one screenshot fragment. That shipped 002's 1025x1418
            # conf-0.405 host raster with pale product silhouettes doubled under the
            # real cutouts. Drop the target and strip src so no downstream stage can
            # re-emit these pixels; the clean plate (which keeps the original pixels
            # here, minus the extracted elements' own removals) is the single owner.
            c["target"] = "drop"
            c.pop("src", None)
            c.pop("mask", None)
            c["meta"]["plate_passthrough"] = True
            c["meta"]["raster_fallback"] = "removal-capped-plate-passthrough"
            removal_capped += 1
            continue
        dilate = inpaint.resolve_mask_dilate(c, cfg)
        # Anti-aliased glyph fringes extend past the Otsu ink mask; the global 2px
        # default (reconstruct.mask_dilate) demonstrably leaves readable halos (009
        # timestamp, 052 "125ML"). Text removals get their own wider floor.
        if c.get("target") == "text" or (
                c.get("target") == "drop" and c.get("text")
                and (c.get("meta") or {}).get("removal_required")):
            dilate = max(dilate, int(rcfg.get("text_removal_dilate", 4)))
            # Fixed floors under-cover DISPLAY text: a 120px headline has 12-18px
            # stroke stems, and a 4px dilation of a slightly-tight ink mask leaves
            # whole stems standing in the plate (094: leftover "B" stem rendered as
            # a stray vertical bar beside the re-placed node; 002: orphan stroke
            # after "BUNDEL"). Scale the floor with glyph height, capped so body
            # text is untouched and a banner headline can't nuke its surroundings.
            _tb = c.get("ink_box") or c.get("visible_box") or c.get("box") or {}
            _th = float(_tb.get("h", 0) or 0)
            if _th > 48:
                dilate = max(dilate, min(int(rcfg.get("display_text_dilate_max", 14)),
                                         int(round(_th * float(rcfg.get("display_text_dilate_frac", 0.08))))))
        # Comparison cards place bright editable copy over textured/translucent photo
        # surfaces. OCR ink masks are slightly tighter than the antialiased glyph fringe;
        # a two-pixel default leaves readable duplicate halos after reconstruction.
        if ((cfg.get("scene") or {}).get("archetype") == "comparison_grid"
                and ((cfg.get("scene") or {}).get("facts") or {}).get("before_after_pair")
                and c.get("target") == "text"):
            dilate = max(dilate, int(rcfg.get("comparison_text_dilate", 5)))
        if ((cfg.get("scene") or {}).get("archetype") == "social_screenshot"
                and c.get("target") == "text"):
            dilate = max(dilate, int(rcfg.get("social_text_dilate", 6)))
        # An opaque cutout (product/person/photo) is re-rendered on top of the plate, so
        # the removal mask MUST cover its full footprint plus the anti-aliased rim — a
        # tighter removal leaves a halo of the ORIGINAL object peeking around the placed
        # cutout (002 product silhouettes). Give image cutouts a rim-dilation floor so
        # removal ⊇ cutout footprint.
        _role = str((c.get("meta") or {}).get("role") or "").lower()
        if c.get("target") == "image" and _role in _CAP_EXEMPT_ROLES:
            dilate = max(dilate, int(rcfg.get("cutout_rim_dilate", 3)))
        # Line-art marks (logo/icon/badge/seal/wordmark) have thin strokes and scalloped
        # rims that SAM mattes routinely under-cover — 002's logo left a ring-shaped ink
        # ghost in the clean plate up to ~17px outside the matte (hidden only because the
        # replacement logo rendered on top). Dilation alone cannot reach that far without
        # damaging neighbours, so union the matte with the measured high-contrast ink
        # inside the candidate box. Config gate ``reconstruct.cutout_ink_union`` (ON).
        if (c.get("target") in ("image", "icon")
                and _role in _LINE_ART_ROLES
                and candidate_mask is not None
                and bool(rcfg.get("cutout_ink_union", True))):
            ink = inpaint.text_ink_mask(rgb, box, allow_box_fallback=False)
            ink_px = int(np.count_nonzero(ink))
            matte_px = int(np.count_nonzero(candidate_mask))
            # Guard against a textured plate turning "ink" into most of the box.
            box_px = max(1, int(round(box.get("w", 1))) * int(round(box.get("h", 1))))
            if ink_px and ink_px <= box_px * float(rcfg.get(
                    "cutout_ink_union_max_box_fraction", 0.85)):
                candidate_mask = np.maximum(
                    np.asarray(candidate_mask, dtype=np.uint8), ink.astype(np.uint8),
                )
                added = int(np.count_nonzero(candidate_mask)) - matte_px
                if added > 0:
                    c.setdefault("meta", {})["cutout_ink_union"] = {
                        "added_px": added, "matte_px": matte_px,
                    }
        observation = {
            "id": c.get("id"),
            "target": c.get("target"),
            "role": (c.get("meta") or {}).get("role"),
            "parent_id": (c.get("meta") or {}).get("parent_id"),
            "z": c.get("z", 0),
            "meta": c.get("meta") or {},
            "box": box,
            "mask_array": candidate_mask,
            "is_background": is_background,
            "dilate": dilate,
        }
        removal.append(observation)
    removal, union, removal_ownership, removal_owner_index = _build_removal_ledger(
        removal, (w, h),
    )
    text_removal = [item for item in removal if _is_text_removal(item)]
    large_removal = [item for item in removal if not _is_text_removal(item)]
    background_path = os.path.join(run_dir, "background_clean.png")
    regional_enabled = bool(((cfg.get("inpaint") or {}).get("regional") or {}).get("enabled", False))

    # A badge chip's chrome has already been rebuilt from its own surface, so the plate is
    # inpainted FROM those pixels and the rebuilt mask is withheld from the generative union.
    # Both halves are load-bearing:
    #   * source  — the chip pixels the backend never touches must already be clean, since
    #     for a plate-hosted badge (013) the plate IS the disc.
    #   * mask    — leaving the badge ink in the union is what fuses it to the pouch cutout
    #     into one canvas-wide component and re-summons the slab. Withheld, the pouch's hole
    #     is judged alone, on its own merits.
    # ``union`` itself is deliberately NOT narrowed: it stays the honest public ledger of
    # every pixel this run removed (removal_mask.png, ownership, the plate-integrity gate).
    # The gate asks that the plate differ from source only INSIDE union, and the chip's
    # rebuilt pixels are a subset of the ink it replaced, so it holds.
    plate_source_path = image_path
    gen_union = union
    chip_withheld = np.zeros((h, w), dtype=np.uint8)
    if np.any(chip_regions):
        # Withhold the badge lines' OWN removal, across the whole chip -- not merely the
        # pixels the fill rewrote. The removal hole for a badge line is its OCR QUAD, not its
        # glyphs (measured on 013: 35,888 quad px inside the chip vs 24,247 of dilated ink),
        # so withholding only the ink would leave the quad's chrome-coloured remainder in the
        # union for Flux to smear -- the slab again, just thinner. Inside a chip we take the
        # whole quad: its ink is rebuilt and its untouched remainder IS the disc's own chrome,
        # already correct in chip_rgb and needing no backend at all.
        #
        # Scoped by per-pixel OWNERSHIP and by the chip footprint, so this can only ever
        # withhold the badge's own hole. 013's pouch cutout overlaps the disc and keeps its
        # removal in full -- the point is to stop the two FUSING, not to skip the pouch.
        # removal_owner_index maps NUMBER -> id (and its keys are strings), so it has to be
        # inverted to look a candidate up. Reading it the intuitive way round silently
        # yields no matches, the quad remainder stays in the union, and the backend repaints
        # a rim around every glyph — a ghost outline of the text it was asked to remove.
        _number_by_id = {v: int(k) for k, v in (removal_owner_index or {}).items()}
        # ONLY plate-hosted chips (host_id None). For those the disc IS the plate — 013's
        # seal was never an element — so the plate must carry the cleaned chrome. An
        # ELEMENT-hosted chip is the opposite: its disc ships as its own raster and its
        # footprint is removed from the plate on purpose, the raster covering the hole.
        # Withholding there strands an island of un-inpainted chip pixels in the middle of
        # that hole (101: a teal rounded-square marooned in the white gap where the disc was).
        chip_ids = {
            i for r in chip_records
            if r.get("cleaned") and not r.get("host_id") for i in (r.get("ids") or [])
        }
        chip_numbers = [n for n in (_number_by_id.get(i) for i in chip_ids) if n]
        if chip_numbers:
            chip_withheld = (
                np.isin(removal_ownership, chip_numbers) & (chip_regions > 0)
            ).astype(np.uint8) * 255
        chip_withheld = np.maximum(
            chip_withheld, cv2.bitwise_and(chip_clean_mask, chip_regions))
        plate_source_path = os.path.join(run_dir, "chip_source.png")
        Image.fromarray(chip_rgb).save(plate_source_path)
        gen_union = cv2.bitwise_and(union, cv2.bitwise_not(chip_withheld))
        print(f"[chip] withheld {int((chip_withheld>0).sum())}px from the generative union "
              f"({int((union>0).sum())} -> {int((gen_union>0).sum())})")

    if regional_enabled:
        inpaint_result = inpaint.inpaint_regional(
            plate_source_path, removal, gen_union, background_path, cfg, run_dir,
        )
    else:
        text_union = inpaint.build_union_mask(
            (w, h), text_removal, run_dir, default_dilate=inpaint.default_mask_dilate(cfg), cfg=cfg,
        )
        large_union = inpaint.build_union_mask(
            (w, h), large_removal, run_dir, default_dilate=inpaint.default_mask_dilate(cfg), cfg=cfg,
        )
        if np.any(text_union) and np.any(large_union):
            large_union = cv2.bitwise_and(large_union, cv2.bitwise_not(text_union))
        if np.any(chip_withheld):
            keep = cv2.bitwise_not(chip_withheld)
            text_union = cv2.bitwise_and(text_union, keep)
            large_union = cv2.bitwise_and(large_union, keep)
        if text_removal and large_removal:
            inpaint_result = inpaint.inpaint_role_aware(
                plate_source_path, {"text": text_union, "large": large_union}, background_path, cfg,
            )
        else:
            inpaint_result = inpaint.inpaint_once(
                plate_source_path, gen_union, background_path, cfg)
    if np.any(chip_clean_mask):
        inpaint_result = dict(inpaint_result or {}, chip_local_badges=chip_records)

    # Post-inpaint ghost-text audit: residue under removed text expands the masks
    # in-place (union + ledger) and triggers one targeted text-backend repair pass,
    # so the artifacts written below always describe the final plate.
    text_residual = _post_inpaint_text_residual(
        rgb, background_path, removal, masks, union, removal_ownership, cfg,
        decoration_regions=_strike_ink_regions(ocr),
    )
    # Clean-plate cover pass: a low-confidence opaque raster kept in the plate (removal
    # cap) leaves a ghost SILHOUETTE in background_clean (002 product panel). Because that
    # raster is re-rendered opaquely on top, its plate pixels are never seen — cover its
    # footprint with the surrounding plate colour so the clean plate has no silhouette.
    footprints_covered = _cover_kept_raster_footprints(
        background_path, updated, masks, cfg, union=union,
    )
    # Unclaimed-removal restore: a removal-ledger region whose owning candidate does NOT
    # re-render into the emitted design is being erased for nothing — the plate is the
    # ONLY surface that will ever show there, so it must hold the ORIGINAL pixels, not an
    # inpaint hole. This covers a dropped/kept-in-background owner, a peel hole punched for
    # an element that later stayed in the plate, and an "emitted" image layer whose asset
    # is actually blank (104/107: products burned into the plate while their asset groups
    # shipped empty). Restore those source pixels and shrink the union + ledger to match.
    # Conservative: an owner that ships a NON-EMPTY raster/text/shape layer keeps its clean
    # inpainted footprint, so products/photos stay swappable.
    restored = _restore_unclaimed_removals(
        rgb, background_path, removal, updated, run_dir, union, removal_ownership, cfg,
    )
    mask_path = os.path.join(run_dir, "removal_mask.png")
    Image.fromarray(union).save(mask_path)
    # Deterministic plate-integrity gate: the plate may differ from the source ONLY
    # inside the just-saved removal union. Every writer composites through its mask and
    # every deliberate post-pass expands the union it mutates under, so any out-of-mask
    # change is a compositing/ownership bug — fail LOUDLY instead of shipping a broken
    # plate hidden under a raster fallback (002's orange-panel plate shipped silently).
    from .inpaint_quality import plate_integrity as _plate_integrity_check
    integrity_cfg = rcfg.get("plate_integrity") if isinstance(
        rcfg.get("plate_integrity"), dict) else {}
    plate_final = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    plate_integrity = _plate_integrity_check(
        rgb, plate_final, union,
        changed_tolerance=int(integrity_cfg.get("changed_tolerance", 0)),
    )
    max_changed = float(integrity_cfg.get("max_out_of_mask_change", 0.0005))
    if (bool(integrity_cfg.get("enforce", True))
            and float(plate_integrity.get("out_of_mask_changed_ratio", 1.0)) > max_changed):
        raise RuntimeError(
            "plate integrity violated: "
            f"{plate_integrity.get('out_of_mask_changed_ratio'):.4%} of pixels outside "
            f"removal_mask.png changed in background_clean.png (limit {max_changed:.4%}; "
            f"edge_retention={plate_integrity.get('edge_retention')}). A plate writer "
            "modified pixels it does not own — see reconstruct/inpaint compositing."
        )
    removal_ownership_path = os.path.join(run_dir, "removal_ownership.png")
    removal_scale = max(1, 65535 // max(1, len(removal_owner_index)))
    Image.fromarray((removal_ownership * removal_scale).astype(np.uint16)).save(
        removal_ownership_path
    )

    updated.extend(_comparison_plate_columns(
        background_path, assets_dir, w, h, cfg, updated,
    ))

    # Visual ownership map plus a machine-readable legend.
    ownership_path = os.path.join(run_dir, "ownership.png")
    scale = max(1, 65535 // max(1, len(front)))
    Image.fromarray((ownership * scale).astype(np.uint16)).save(ownership_path)
    result = {
        "schema_version": 2,
        "background": "background_clean.png",
        "removal_mask": "removal_mask.png",
        "ownership": "ownership.png",
        "owner_index": owner_index,
        "removal_ownership": "removal_ownership.png",
        "removal_owner_index": removal_owner_index,
        "candidates": updated,
        "stats": {
            "input_candidates": len(candidates),
            "canonical_entities": len(updated),
            "duplicates_removed": len(candidates) - len(updated),
            "vectorized": vector_ok,
            "vector_fallback": vector_fallback,
            "flattened_scene_artwork": flattened_scene_artwork,
            "plate_fragments_suppressed": plate_fragments_suppressed,
            "engagement_underlays_suppressed": engagement_underlays_suppressed,
            "text_shells_promoted": text_shells_promoted,
            "mask_rejected": mask_rejected,
            "removal_capped": removal_capped,
            "kept_footprints_covered": footprints_covered,
            "unclaimed_removals_restored": restored,
            "plate_integrity": plate_integrity,
            # Ghost-text audit evidence: repair.assess turns unresolved residue into a
            # rebuild-clean-plate repair instead of letting duplicate text ship.
            "text_residual": text_residual,
            # Surface the low-quality OpenCV fallback at the top level so acceptance QA
            # (pixel_diff._structural_audit) can hard-fail on it. The producer emits it
            # nested (per-region backend_counts, or single-pass diagnostics.backend_route),
            # so hoist it here rather than leaving the QA gate reading a key that never exists.
            "opencv_fallback_used": _inpaint_used_opencv(inpaint_result),
            "inpaint": inpaint_result,
        },
    }
    dump(result, os.path.join(run_dir, "reconstruction.json"))
    return result


# ── Codia-style confidence-gated raster-slice fallback ─────────────────────────────
#
# Any emitted layer whose preview region no longer matches the source (per-layer crop
# SSIM, plus ink-IoU/ghost gates for text — see pixel_diff._layer_region_rows and
# schema.raster_slice_failures) is replaced by a pixel-exact slice of the ORIGINAL
# source pixels.  The slice's alpha is exactly the set of pixels the removal ledger
# inpainted out on behalf of that layer, so:
#   * the slice covers the inpainted hole and nothing else (no background leakage),
#   * a pixel never renders twice (ledger pixels are exclusive by construction),
#   * remaining editable layers keep their own ledger pixels untouched.
# A failing layer whose pixels were never removed (keep_underlay overlays) is simply
# dropped — the plate already shows the original pixels.  The failed editable attempt
# is preserved in meta["fallback_editable"] so a later repair can restore it.


def _find_layer_node(nodes, layer_id, offset=(0.0, 0.0)):
    """Locate a node by id in a (possibly nested) candidate/layer tree.

    Returns (container_list, index, node, parent_offset).  Children of groups carry
    parent-relative coordinates; ``parent_offset`` converts them to canvas space.
    """
    for index, node in enumerate(nodes or []):
        if not isinstance(node, dict):
            continue
        if str(node.get("id")) == str(layer_id):
            return nodes, index, node, offset
        children = node.get("children") or []
        if children:
            box = node.get("box") or {}
            child_offset = (
                offset[0] + float(box.get("x") or 0),
                offset[1] + float(box.get("y") or 0),
            )
            found = _find_layer_node(children, layer_id, child_offset)
            if found:
                return found
    return None


_SLICE_EDITABLE_KEYS = (
    "text", "style", "text_runs", "fill", "stroke", "shape_kind", "radius",
    "path", "svg", "paths", "src", "mask",
)


def _apply_slice_mutation(node, local_box, src_rel, scores, reasons):
    """Turn a failed editable node into a positioned raster slice, keeping provenance."""
    meta = dict(node.get("meta") or {})
    editable = {key: node.get(key) for key in _SLICE_EDITABLE_KEYS
                if node.get(key) not in (None, [], {})}
    editable["kind"] = node.get("target") or node.get("type")
    if node.get("text"):
        meta["source_text"] = str(node.get("text"))
    label = str(node.get("name") or meta.get("semantic_name")
                or node.get("text") or node.get("id") or "layer").strip()
    label = " ".join(label.split())
    if len(label) > 40:
        label = label[:39] + "…"
    meta.update({
        "fallback": "raster-slice",
        "fallback_reasons": list(reasons),
        "fallback_scores": scores,
        "fallback_editable": editable,
        "layer_disposition": "foreground_raster",
        # The alpha is a ledger cutout by construction, not a broken matte.
        "ownership_cutout": True,
    })
    if "target" in node:
        node["target"] = "image"
    if "type" in node:
        node["type"] = "image"
    node["name"] = f"{label} — raster slice (low confidence)"
    node["box"] = dict(local_box)
    node["src"] = src_rel
    node["rotation"] = 0.0
    node["opacity"] = 1.0
    node["blend_mode"] = "NORMAL"
    node["effects"] = []
    node["style"] = {}
    for key in ("fill", "stroke", "shape_kind", "radius", "path", "svg", "paths",
                "text_runs", "visible_box", "ink_box", "mask", "text"):
        node.pop(key, None)
    node["meta"] = meta
    return node


def _apply_drop_mutation(node, scores, reasons):
    """Retire a failed layer whose source pixels still live in the plate."""
    meta = dict(node.get("meta") or {})
    meta.update({
        "fallback": "plate-passthrough",
        "fallback_reasons": list(reasons),
        "fallback_scores": scores,
        "keep_in_background": True,
        "suppression_reason": "confidence-fallback-plate-already-correct",
    })
    if "target" in node:
        node["target"] = "drop"
    node["meta"] = meta
    return node


def _boxes_intersect(a, b) -> bool:
    """True when two canvas-space boxes share any area."""
    ax0 = float(a.get("x", 0) or 0)
    ay0 = float(a.get("y", 0) or 0)
    ax1 = ax0 + float(a.get("w", 0) or 0)
    ay1 = ay0 + float(a.get("h", 0) or 0)
    bx0 = float(b.get("x", 0) or 0)
    by0 = float(b.get("y", 0) or 0)
    bx1 = bx0 + float(b.get("w", 0) or 0)
    by1 = by0 + float(b.get("h", 0) or 0)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


_LIVE_OVERLAY_KINDS = frozenset({"text", "shape", "icon", "vector"})


def _collect_live_overlays(nodes, exclude_ids, offset=(0.0, 0.0), out=None):
    """Absolute ``(box, z, id)`` of every content-bearing leaf that re-renders natively.

    Used to keep a hoisted raster slice BELOW the live native overlays it intersects.
    A slice lifted to a blind foreground z (a failed badge card sliced up to z=60)
    otherwise buries the live chip/badge text painted on top of it — 013's snacks chip
    went blank exactly this way. Groups are descended (child boxes are parent-relative);
    only real editable leaves count as overlays, never backgrounds or the slices we are
    currently producing (``exclude_ids``)."""
    out = [] if out is None else out
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        box = node.get("box") or {}
        ax = offset[0] + float(box.get("x") or 0)
        ay = offset[1] + float(box.get("y") or 0)
        children = node.get("children") or []
        if children:
            _collect_live_overlays(children, exclude_ids, (ax, ay), out)
            continue
        nid = str(node.get("id"))
        if nid in exclude_ids:
            continue
        meta = node.get("meta") or {}
        if meta.get("is_background") or str(meta.get("role") or "") == "background":
            continue
        if meta.get("fallback") in ("raster-slice", "plate-passthrough"):
            continue
        kind = str(node.get("target") or node.get("type") or "")
        if kind not in _LIVE_OVERLAY_KINDS:
            continue
        z_raw = node.get("z_index")
        if z_raw is None:
            z_raw = node.get("z")
        try:
            z = float(z_raw) if z_raw is not None else 0.0
        except (TypeError, ValueError):
            z = 0.0
        out.append(({
            "x": ax, "y": ay,
            "w": float(box.get("w") or 0), "h": float(box.get("h") or 0),
        }, z, nid))
    return out


def _group_paints(node):
    """True when a group actually puts pixels down that could bury a hoisted slice.

    A hoisted slice only needs to clear the group it left if that group PAINTS over it:
    101's white panel carries a ``#ffffff`` fill and wiped the tube, but 107's
    ``root__band1`` is a bare container (``fill=None``, no backdrop child), so re-basing
    above it is unnecessary AND harmful — it lifted 107's opaque callout-leader slice
    (z 5 -> 21) over the button group and pasted original glyph tops across the native
    "DAILY HYDRATION" (1040px of damage; z=5 renders pixel-identical to the baseline).
    """
    if not isinstance(node, dict):
        return False
    if node.get("fill") or node.get("fills"):
        return True
    if node.get("src") or node.get("type") in ("image", "text", "shape"):
        return True
    # A backdrop/plate child paints on the group's behalf.
    for child in node.get("children") or []:
        if not isinstance(child, dict):
            continue
        cid = str(child.get("id") or "")
        role = str((child.get("meta") or {}).get("role") or "")
        if cid.endswith("__groupbg") or role in ("background", "backdrop", "plate"):
            return True
        if child.get("fill") or child.get("fills"):
            return True
    return False


def _root_ancestor_z(root_nodes, layer_id):
    """z of the TOP-LEVEL node whose subtree holds ``layer_id``.

    ``None`` when the layer already sits at root (its z is root-scoped already), when no
    ancestor carries a usable z, or when the ancestor does not PAINT — all three mean
    "nothing to clear".
    """

    def _holds(nodes):
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            if str(node.get("id")) == str(layer_id):
                return True
            if _holds(node.get("children") or []):
                return True
        return False

    for node in root_nodes or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("id")) == str(layer_id):
            return None
        if _holds(node.get("children") or []):
            if not _group_paints(node):
                return None
            z_raw = node.get("z_index")
            if z_raw is None:
                z_raw = node.get("z")
            try:
                return float(z_raw) if z_raw is not None else None
            except (TypeError, ValueError):
                return None
    return None


def _hoist_floor(targets, layer_id):
    """Highest root z the slice must clear across every tree it is hoisted in.

    One ``hoist_z`` is written to layout.json AND design.json, so it has to clear the
    layer's group in whichever tree nests it deepest; a layer already at root in every
    tree yields ``None`` (nothing to clear, historical behaviour preserved).
    """
    floors = [
        z for z in (_root_ancestor_z(root, layer_id) for _target, root in targets or [])
        if z is not None
    ]
    return max(floors) if floors else None


def _safe_hoist_z(node, abs_box, overlays, default=60.0, floor=None):
    """Root z for a hoisted slice that never rises above an intersecting live overlay.

    Preserves a legitimately lower original z; otherwise sits just under the lowest
    live overlay the slice intersects. Falls back to ``default`` foreground z only when
    nothing live overlaps (the historical behaviour, safe when there is nothing to bury).

    ``floor`` is the root z of the group the slice is being hoisted OUT of.  z ranks a
    node against its SIBLINGS, so a child's z is only meaningful inside its own group:
    the hoist keeps the number but changes the scope.  101's TPU tube sat at in-group
    z 2 inside the white panel (root z 50) and rendered fine; hoisted to root it kept
    z 2, the panel's own #ffffff fill painted over it, and the entire product plus its
    'BUY 3, GET 1 FREE' badge vanished into a white void.  Re-basing above ``floor``
    keeps the hoist rendering what it rendered before it moved.
    """
    z_raw = node.get("z_index")
    if z_raw is None:
        z_raw = node.get("z")
    try:
        base = float(z_raw) if z_raw not in (None, "", "0", "0.0") else 0.0
    except (TypeError, ValueError):
        base = 0.0
    if floor is not None:
        base = max(base, float(floor) + 1.0)
    ceil = None
    for obox, oz, _oid in overlays:
        if _boxes_intersect(obox, abs_box):
            ceil = oz if ceil is None else min(ceil, oz)
    if ceil is not None:
        if base and base < ceil:
            return base
        # Ducking under the overlay must never sink the slice back beneath the group
        # fill we just cleared: invisible beats mildly mis-ordered.
        lowered = ceil - 1.0
        if floor is None or lowered > float(floor):
            return lowered
        return base
    return base if base > 0 else default


def _removal_ledger_numbers(run_dir, reconstruction, shape):
    """Load the removal-ownership ledger as owner numbers plus id → numbers map."""
    _, np, Image = _deps()
    index = reconstruction.get("removal_owner_index") or {}
    rel = reconstruction.get("removal_ownership") or "removal_ownership.png"
    path = os.path.join(run_dir, rel)
    if not index or not os.path.exists(path):
        return None, {}
    arr = np.asarray(Image.open(path), dtype=np.uint32)
    if arr.shape != shape[:2]:
        return None, {}
    scale = max(1, 65535 // max(1, len(index)))
    numbers = arr // scale
    id_to_numbers: dict = {}
    for number, owner in index.items():
        try:
            id_to_numbers.setdefault(str(owner), []).append(int(number))
        except (TypeError, ValueError):
            continue
    return numbers, id_to_numbers


def apply_raster_slice_fallback(run_dir: str, source_path: str, cfg: Optional[dict] = None) -> dict:
    """Replace low-confidence regions of design.json with pixel-exact source slices.

    Runs after the local preview render (and again on every harness round that reaches
    the preview stage, so a repair can force specific layers via
    ``reconstruct.focus_regions``/``fallback.force_slice_ids``).  Mutates
    reconstruction.json + layout.json, rebuilds design.json through the normal
    compiler, re-renders the preview, and writes an auditable ``fallback.json``.
    Config-gated with defaults ON (``fallback.enabled``).
    """
    cfg = cfg or {}
    thresholds = raster_slice_thresholds(cfg)
    report: dict = {
        "enabled": bool(thresholds.get("enabled", True)),
        "thresholds": {k: v for k, v in thresholds.items() if k != "enabled"},
        "scored": 0, "slices": [], "dropped": [], "skipped": [],
    }
    out_path = os.path.join(run_dir, "fallback.json")
    if not report["enabled"]:
        dump(report, out_path)
        return report
    design_path = os.path.join(run_dir, "design.json")
    preview_path = os.path.join(run_dir, "preview.png")
    recon_path = os.path.join(run_dir, "reconstruction.json")
    for required in (design_path, preview_path, recon_path, source_path):
        if not required or not os.path.exists(required):
            report["note"] = f"missing artifact: {os.path.basename(str(required))}"
            dump(report, out_path)
            return report
    from . import pixel_diff  # lazy: keeps reconstruct importable without QA deps

    _, np, Image = _deps()
    design = load(design_path)
    rows = pixel_diff.score_layer_regions(source_path, preview_path, design, run_dir)
    report["scored"] = len(rows)

    forced = {str(value) for value in (thresholds.get("force_slice_ids") or [])}
    for entry in ((cfg.get("reconstruct") or {}).get("focus_regions") or []):
        if isinstance(entry, dict) and entry.get("layer_id"):
            forced.add(str(entry["layer_id"]))
    # Legacy / forensic: residual audit may still emit force_raster_ids when
    # reconstruct.text_residual.force_raster is explicitly enabled. Readable TEXT is
    # still refused below unless text_slice_gate_enabled is on.
    audit_forced = set()
    try:
        _audit = ((load(recon_path).get("stats") or {}).get("text_residual") or {})
        audit_forced = {str(i) for i in (_audit.get("force_raster_ids") or [])}
    except Exception:
        audit_forced = set()
    forced |= audit_forced
    text_slice_ok = bool(thresholds.get("text_slice_gate_enabled", False))

    canvas = design.get("canvas") or {}
    canvas_area = max(1.0, float(canvas.get("w") or 1) * float(canvas.get("h") or 1))
    failing = []
    for row in rows:
        rid = str(row.get("id"))
        if is_raster_slice({"fallback": row.get("fallback")}):
            continue  # our own slice output — never re-gate it (F11 canonical read)
        # Upstream fidelity fallbacks (meta.fallback == True, e.g. the masked-pixel
        # text path) stay gated on purpose: a fallback whose ink mask is broken
        # renders garbage and must still be replaceable by a source slice
        # (benchmark 009 c_B6: masked-pixel render shipped at region_ssim 0.33).
        reasons = raster_slice_failures(row, thresholds)
        if rid in forced and not reasons:
            reasons = ["forced by repair (reconstruct.focus_regions)"]
        if not reasons:
            continue
        found = _find_layer_node(design.get("layers") or [], rid)
        node = found[2] if found else {}
        if (node.get("meta") or {}).get("native_decoration"):
            # This path is already a direct fit to saturated source pixels. Region SSIM
            # is invalid for a thin overlay because the crop is dominated by unrelated
            # price glyphs; slicing would destroy the very editability this fallback is
            # meant to protect.
            report["skipped"].append({
                "id": rid,
                "reason": "evidence-backed-native-decoration",
                "failing_reasons": list(reasons),
            })
            continue
        # Codia: never raster-slice readable OCR/native TEXT to boost fidelity.
        is_text_layer = str(row.get("type") or "") == "text"
        if is_text_layer and not text_slice_ok:
            report["skipped"].append({
                "id": rid,
                "reason": "codia-never-slice-readable-text",
                "failing_reasons": list(reasons),
            })
            continue
        if (node.get("meta") or {}).get("badge_chip_host_cleaned"):
            # A badge chip whose offer is now NATIVE TEXT scores low on region SSIM by
            # construction — the region deliberately no longer matches the source, because
            # the baked offer was replaced by a re-rendered one (101: region_ssim 0.470
            # against a 0.58 floor). That is the feature, not a defect, and the same logic
            # the text exemption above already applies.
            #
            # Slicing it is strictly destructive here. The slice's alpha is only the pixels
            # the ledger inpainted FOR THIS LAYER, which excludes the quad its own text
            # owns, so it re-pastes the chrome ring and leaves a transparent rectangle where
            # the offer used to be — 101 shipped a teal disc with a white hole punched in
            # it, 2230 alpha px of 5980. And with the text quad included it would simply
            # paste the original ink back under the native copy: the double render.
            report["skipped"].append({
                "id": rid,
                "reason": "chip-cleaned-badge-chrome-native-offer",
                "failing_reasons": list(reasons),
            })
            continue
        if int(row.get("region_px") or 0) < int(thresholds["min_region_px"]):
            report["skipped"].append({"id": rid, "reason": "region-too-small"})
            continue
        if float(row.get("region_px") or 0) / canvas_area > float(thresholds["max_layer_canvas_fraction"]):
            # A slice this big would approach an untouched source copy — the exact
            # failure mode the architecture forbids. Leave it to stage-level repairs.
            report["skipped"].append({"id": rid, "reason": "region-too-large-for-slice"})
            continue
        failing.append((row, reasons))
    # Worst (lowest region_ssim) first, so a slice budget spends on the most-broken regions.
    failing.sort(key=lambda item: float(item[0].get("region_ssim") or 0.0))
    slice_budget = int(thresholds.get("max_slices", 8))
    if len(failing) > slice_budget:
        # F10: the excess used to be dropped silently, which made "every sub-threshold
        # region resolved" unauditable. Record the un-sliced failing regions honestly so
        # QA/Gate 3 can see the budget was exhausted rather than the regions being clean.
        truncated = failing[slice_budget:]
        report["truncated"] = {
            "reason": "slice-budget-exhausted",
            "max_slices": slice_budget,
            "un_sliced_count": len(truncated),
            "un_sliced_ids": [str(item[0].get("id")) for item in truncated],
        }
        for item in truncated:
            report["skipped"].append({
                "id": str(item[0].get("id")),
                "reason": "slice-budget-exhausted",
                "failing_reasons": list(item[1]),
            })
        failing = failing[:slice_budget]
    if not failing:
        dump(report, out_path)
        return report

    reconstruction = load(recon_path)
    rgb = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
    removal_mask_path = os.path.join(
        run_dir, reconstruction.get("removal_mask") or "removal_mask.png")
    removal_union = None
    if os.path.exists(removal_mask_path):
        removal_union = np.asarray(Image.open(removal_mask_path).convert("L")) > 0
    numbers, id_to_numbers = _removal_ledger_numbers(run_dir, reconstruction, rgb.shape)
    assets_dir = os.path.join(run_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    layout_path = os.path.join(run_dir, "layout.json")
    tree = load(layout_path) if os.path.exists(layout_path) else None
    candidates = [c for c in reconstruction.get("candidates") or [] if isinstance(c, dict)]
    candidates_by_id = {str(c.get("id")): c for c in candidates}
    # Live native overlays (chips/badges/copy that re-render on top) so a hoisted slice
    # is never lifted above the very overlay it sits under (013 snacks chip). The regions
    # we are about to slice/drop are NOT overlays — exclude them.
    failing_ids = {str(r.get("id")) for r, _ in failing}
    live_overlays = _collect_live_overlays(tree, failing_ids) if tree is not None else []
    plate_passthrough_max_inpaint = float(
        thresholds.get("plate_passthrough_max_inpaint_frac", 0.12))

    changed = False
    for row, reasons in failing:
        rid = str(row.get("id"))
        scores = {key: row.get(key)
                  for key in ("region_ssim", "region_color", "ink_iou", "ink_excess")
                  if row.get(key) is not None}
        alpha = None
        wanted = id_to_numbers.get(rid)
        if numbers is not None and wanted:
            alpha = np.isin(numbers, np.asarray(wanted, dtype=numbers.dtype))
            if not alpha.any():
                alpha = None
        targets = []
        if tree is not None:
            found = _find_layer_node(tree, rid)
            if found:
                targets.append((found, tree))
        design_root = design.get("layers") or []
        design_hit = _find_layer_node(design_root, rid)
        if design_hit:
            targets.append((design_hit, design_root))
        candidate = candidates_by_id.get(rid)
        if not targets and candidate is None:
            report["skipped"].append({"id": rid, "reason": "layer-not-found"})
            continue

        if alpha is None:
            # No ledger cutout for this layer. Two very different situations hide here:
            #   (a) the layer's pixels were never removed (keep_underlay overlay, plate
            #       cap): the plate genuinely still shows the ORIGINAL, so dropping the
            #       broken recreation is the pixel-exact fallback; OR
            #   (b) the region WAS inpainted (its content was removed for a recreation
            #       that then failed its gate): the plate there is a hole/smear, so a
            #       silent drop ships a ghost. This is the badge/seal shell case
            #       (016 c_E013__shell white patch). Chrome must never drop to a ghost —
            #       box-slice the ORIGINAL source for the region instead.
            abs_box = None
            for (container, index, node, off), _root in targets:
                b = node.get("box") or {}
                abs_box = {
                    "x": float(b.get("x") or 0) + off[0],
                    "y": float(b.get("y") or 0) + off[1],
                    "w": float(b.get("w") or 0), "h": float(b.get("h") or 0),
                }
                break
            inpainted_frac = 0.0
            if (removal_union is not None and abs_box
                    and abs_box["w"] > 0 and abs_box["h"] > 0):
                bx0 = max(0, int(round(abs_box["x"])))
                by0 = max(0, int(round(abs_box["y"])))
                bx1 = min(removal_union.shape[1], int(round(abs_box["x"] + abs_box["w"])))
                by1 = min(removal_union.shape[0], int(round(abs_box["y"] + abs_box["h"])))
                if bx1 > bx0 and by1 > by0:
                    sub = removal_union[by0:by1, bx0:bx1]
                    inpainted_frac = float(np.count_nonzero(sub)) / max(1, sub.size)
            if inpainted_frac >= plate_passthrough_max_inpaint and abs_box:
                bx0 = max(0, int(round(abs_box["x"])))
                by0 = max(0, int(round(abs_box["y"])))
                bx1 = min(rgb.shape[1], int(round(abs_box["x"] + abs_box["w"])))
                by1 = min(rgb.shape[0], int(round(abs_box["y"] + abs_box["h"])))
                if bx1 > bx0 and by1 > by0:
                    tile = np.dstack([
                        rgb[by0:by1, bx0:bx1],
                        np.full((by1 - by0, bx1 - bx0), 255, dtype=np.uint8),
                    ])
                    src_rel = _write_asset(
                        Image.fromarray(tile), assets_dir, f"{rid}_boxslice")
                    box_slice = {"x": bx0, "y": by0, "w": bx1 - bx0, "h": by1 - by0}
                    hoist_z = _safe_hoist_z(targets[0][0][2], box_slice, live_overlays,
                                            floor=_hoist_floor(targets, rid))
                    for (container, index, node, _offset), root in targets:
                        _apply_slice_mutation(node, box_slice, src_rel, scores, reasons)
                        node["meta"]["fallback_slice_kind"] = "box-no-ledger-alpha"
                        if root is not None and container is not root:
                            container.pop(index)
                            node["z_index"] = hoist_z
                            node["z"] = hoist_z
                            root.append(node)
                    if candidate is not None and not any(
                            node is candidate for (_c, _i, node, _o), _r in targets):
                        _apply_slice_mutation(candidate, box_slice, src_rel, scores, reasons)
                    report["slices"].append({
                        "id": rid, "reasons": reasons, "scores": scores, "box": box_slice,
                        "src": src_rel, "alpha_px": int((bx1 - bx0) * (by1 - by0)),
                        "covered_by_removal_mask": True, "hoist_z": hoist_z,
                        "note": "box-slice (no ledger alpha; plate was inpainted)",
                    })
                    changed = True
                    continue
            # (a) plate genuinely holds the original — drop.
            for (container, index, node, _offset), _root in targets:
                if "target" in node:
                    _apply_drop_mutation(node, scores, reasons)
                else:
                    container.pop(index)
            if candidate is not None:
                _apply_drop_mutation(candidate, scores, reasons)
            report["dropped"].append({
                "id": rid, "reasons": reasons, "scores": scores,
                "note": "plate-already-holds-source-pixels",
            })
            changed = True
            continue

        ys, xs = np.nonzero(alpha)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        # Minimum-alpha gate: the box above is alpha's OWN tight bbox, so a low coverage
        # inside it means the ledger handed us scattered specks, not a cutout. Shipping
        # that "slice" REPLACES a good chip with a near-empty one and the element visually
        # disappears (107: a 10-px alpha over an 88x89 bbox = 0.13% coverage overwrote a
        # 755-dark-px down-arrow icon). A slice may only replace an existing render when it
        # actually carries content; otherwise keep what we have — an imperfect chip beats a
        # blank one, and this fails toward NOT destroying material we cannot improve.
        alpha_px = int(alpha.sum())
        box_area = max(1, (x1 - x0) * (y1 - y0))
        coverage = alpha_px / float(box_area)
        min_alpha_px = int(thresholds.get("min_slice_alpha_px", 24))
        min_alpha_frac = float(thresholds.get("min_slice_alpha_frac", 0.02))
        if alpha_px < min_alpha_px or coverage < min_alpha_frac:
            report["skipped"].append({
                "id": rid,
                "reason": "slice-alpha-too-sparse",
                "alpha_px": alpha_px,
                "alpha_coverage": round(coverage, 5),
                "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                "failing_reasons": list(reasons),
            })
            continue
        tile = np.dstack([
            rgb[y0:y1, x0:x1],
            (alpha[y0:y1, x0:x1].astype(np.uint8) * 255),
        ])
        src_rel = _write_asset(Image.fromarray(tile), assets_dir, f"{rid}_slice")
        abs_box = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        # Slices are POSITIONED rasters measured in canvas space. Leaving them nested
        # under their original group re-derives their position through parent offsets —
        # and design-compile can later EXPAND a group box to its child union, shifting
        # every parent-relative child (013: c_E007__hostbg_P12 rendered as a displaced
        # plain-green square over the bag's printed bear logo). Hoist each slice to its
        # tree ROOT with its absolute box: root placement cannot drift.
        hoist_z = (_safe_hoist_z(targets[0][0][2], abs_box, live_overlays,
                                 floor=_hoist_floor(targets, rid))
                   if targets else 60.0)
        for (container, index, node, _offset), root in targets:
            _apply_slice_mutation(node, abs_box, src_rel, scores, reasons)
            if root is not None and container is not root:
                container.pop(index)
                # Keep the slice below any live native overlay it intersects — hoisting
                # blindly to z=60 buried the 013 snacks chip painted on top of it.
                node["z_index"] = hoist_z
                node["z"] = hoist_z
                root.append(node)
        if candidate is not None and not any(node is candidate for (_c, _i, node, _o), _r in targets):
            _apply_slice_mutation(candidate, abs_box, src_rel, scores, reasons)
        covered = True
        if removal_union is not None:
            outside = int(np.count_nonzero(alpha & ~removal_union))
            covered = outside == 0
        report["slices"].append({
            "id": rid, "reasons": reasons, "scores": scores, "box": abs_box,
            "src": src_rel, "alpha_px": int(alpha.sum()),
            "covered_by_removal_mask": covered, "hoist_z": hoist_z,
        })
        changed = True

    if changed:
        # A ghost-text layer the residual audit flagged and we just replaced with a
        # pixel-exact slice (or dropped to plate-passthrough) no longer double-renders:
        # the slice IS the resolution. Mark those audit flags resolved so QA's
        # glyph-residue gate and repair's rebuild-clean-plate don't re-fire on a region
        # the looks-right floor already handled.
        handled_ids = {str(s.get("id")) for s in report["slices"]} | {
            str(d.get("id")) for d in report["dropped"]}
        resolved_by_slice = audit_forced & handled_ids
        if resolved_by_slice:
            audit_stats = (reconstruction.get("stats") or {}).get("text_residual") or {}
            for entry in audit_stats.get("flagged") or []:
                if isinstance(entry, dict) and str(entry.get("id")) in resolved_by_slice:
                    entry["resolved"] = True
                    entry["resolved_by"] = "raster-slice"
            report["residue_resolved_by_slice"] = sorted(resolved_by_slice)
        dump(reconstruction, recon_path)
        if tree is not None:
            dump(tree, layout_path)
            from . import build_design_json  # lazy import; heavy transitive deps
            base_rel = reconstruction.get("background") or "background_clean.png"
            build_design_json.build(
                tree,
                {"w": canvas.get("w"), "h": canvas.get("h")},
                run_dir,
                base_src=os.path.join(run_dir, base_rel),
                doc_id=str(design.get("id") or os.path.basename(run_dir)),
                name=str(design.get("name") or os.path.basename(run_dir)),
                kept_in_photo=list(design.get("kept_in_photo") or []),
            )
        else:
            # Minimal path for fixtures without layout.json: patch design.json in
            # place and keep its honest editable accounting roughly current.
            layers = design.get("layers") or []
            flat = []
            def _visit(items):
                for item in items:
                    flat.append(item)
                    _visit(item.get("children") or [])
            _visit(layers)
            editable = sum(1 for item in flat if item.get("type") in ("text", "shape", "group"))
            meta = design.setdefault("meta", {})
            meta["editable_ratio"] = round(editable / max(1, len(flat)), 4)
            dump(design, design_path)
        try:
            from . import render_preview  # lazy: PIL-only, but keep import cost off the hot path
            preview_result = render_preview.render(design_path, run_dir)
            report["preview"] = {"rerendered": True, "errors": preview_result.get("errors") or []}
        except Exception as exc:
            report["preview"] = {"rerendered": False, "error": str(exc)}
    dump(report, out_path)
    return report
