"""Materialize canonical assets and a duplicate-free background plate.

This is the first stage that turns detections into pixels with ownership.  It resolves all
run-relative paths, removes duplicate observations, extracts alpha crops, routes simple
graphics through the vector fidelity gate, samples native shape fills, and sends one final
union mask to :mod:`src.inpaint`.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

from . import inpaint, vectorize
from .schema import dump
from .raster_clusters import INTENTIONAL_RASTER_CLUSTER_ROLES, is_intentional_raster_cluster


def _deps():
    import cv2
    import numpy as np
    from PIL import Image
    return cv2, np, Image


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
            if owner is not None or facts.get("before_after_pair"):
                c["target"] = "drop"
                meta["kept_in_photo"] = True
                meta["baked_owner_id"] = (owner or {}).get("id")
                meta["suppression_reason"] = "comparison-column-label-baked"
        out.append(c)
    return out


def _keeps_underlay(candidate: dict) -> bool:
    """True for overlay layers whose already-valid underlying plate must not be erased."""
    meta = candidate.get("meta") or {}
    return bool(meta.get("keep_underlay") or meta.get("preserve_underlay")
                or meta.get("overlay_without_removal"))


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
        "text_backplate", "panel", "image-panel", "photo-panel", "triptych-panel",
        "comparison-panel", "comparison-column",
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
        if (containing_frame and not root_is_background_scene
                and target not in {"text", "drop"} and role not in {"avatar", "badge", "button"}):
            c["target"] = "drop"
            meta["kept_in_owner"] = containing_frame.get("id")
            meta["suppression_reason"] = "contained-in-swappable-photo-frame"
            out.append(c)
            continue
        promoted = bool(meta.get("promote_element") or meta.get("editable_element")
                        or meta.get("verified_mask") or exact_text_fallback)
        is_semantic = (role in separate_roles or is_intentional_raster_cluster(role)
                       or target == "icon")
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
    _, np, _ = _deps()
    h, w = rgb.shape[:2]
    meta = candidate.get("meta") or {}
    rcfg = (cfg or {}).get("reconstruct") or {}
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
        target_rank = {"text": 40, "icon": 30, "shape": 20, "image": 10}.get(target, 0)
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


def _crop_rgba(rgb, mask, box):
    _, np, Image = _deps()
    h, w = rgb.shape[:2]
    x0 = max(0, int(round(box.get("x", 0))))
    y0 = max(0, int(round(box.get("y", 0))))
    x1 = min(w, int(round(box.get("x", 0) + box.get("w", 0))))
    y1 = min(h, int(round(box.get("y", 0) + box.get("h", 0))))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    rgba = np.dstack([rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
    return Image.fromarray(rgba.astype(np.uint8))


def _source_rgba(candidate, rgb, mask, run_dir):
    """Prefer a model-provided clean RGBA layer, correctly cropped to its tight box."""
    _, np, Image = _deps()
    cluster_meta = candidate.get("meta") or {}
    if (is_intentional_raster_cluster(cluster_meta.get("role"))
            or cluster_meta.get("intentional_raster_cluster")):
        # Do not use a transparent Qwen/SAM crop here: the original full crop is the
        # fidelity contract for an inseparable cluster.
        return _crop_rgba(rgb, mask, candidate.get("box", {}))
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
    return _crop_rgba(rgb, mask, box)


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


def _corner_radius(local_mask):
    """Infer an axis-aligned rounded-rectangle radius from its four clipped corners.

    This intentionally returns ``None`` for a noisy/partial mask.  A wrong native radius
    is worse than a rectangular fallback because it visibly bends otherwise straight art.
    """
    _, np, _ = _deps()
    if local_mask.size == 0 or min(local_mask.shape) < 8:
        return None
    h, w = local_mask.shape
    if float(local_mask.mean()) < .62:
        return None

    def first_true(values):
        hit = np.flatnonzero(values)
        return int(hit[0]) if hit.size else None

    pairs = [
        (first_true(local_mask[0, :]), first_true(local_mask[:, 0])),
        (first_true(local_mask[0, ::-1]), first_true(local_mask[:, -1])),
        (first_true(local_mask[-1, ::-1]), first_true(local_mask[::-1, -1])),
        (first_true(local_mask[-1, :]), first_true(local_mask[::-1, 0])),
    ]
    radii = []
    max_radius = min(h, w) * .48
    for horizontal, vertical in pairs:
        if horizontal is None or vertical is None:
            return None
        # A real quarter-circle has the same first occupied distance on both edges.
        if abs(horizontal - vertical) > max(2, min(h, w) * .08):
            return None
        radius = (horizontal + vertical) / 2
        if radius < 1.25 or radius > max_radius:
            radii.append(0.0)
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
    # Keep the existing ellipse heuristic, but do not call arbitrary sparse SAM masks rects.
    if .75 <= aspect <= 1.33 and corners <= 1 and .55 <= fill <= .90:
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


def _extract_shape_style(rgb, mask, box, cfg):
    """Conservative native-style extraction for semantic primitive candidates."""
    _, np, _ = _deps()
    local_rgb, local_mask = _local_shape_pixels(rgb, mask, box)
    geometry = _simple_shape_geometry(local_mask)
    if geometry is None:
        return None
    style_cfg = ((cfg.get("reconstruct") or {}).get("style_extraction") or {})
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
    if role in ("button", "cta", "chip", "divider", "bar"):
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

    canonical = deduplicate(candidates, float(rcfg.get("dedup_iou", 0.86)))
    photo_policy = (((cfg.get("scene") or {}).get("preset") or {}).get("photo_regions") or {})
    if photo_policy.get("suppress_descendants", True):
        canonical = _suppress_baked_raster_text(
            canonical, float(rcfg.get("raster_text_containment", .90)),
        )
    canonical = _suppress_comparison_column_labels(canonical, cfg)
    canonical, flattened_scene_artwork = _flatten_photo_scene(canonical, cfg)
    ocr_lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    masks = {}
    for candidate in canonical:
        if (candidate.get("target") != "drop"
                or (candidate.get("meta") or {}).get("removal_required")):
            masks[candidate.get("id")] = _candidate_mask(
                candidate, rgb, run_dir, ocr_lines, cfg,
            )

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
        if target == "icon":
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
        mask = masks.get(cid)
        if mask is None:
            updated.append(c)
            continue
        if target == "shape":
            # Do not overwrite upstream paint facts.  This fills only the gaps left by
            # segmentation/Qwen and tags every inference for later QA/debugging.
            extracted = _extract_shape_style(rgb, mask, c.get("box", {}), cfg)
            photo_mask = _photo_shape_override(rgb, mask, c.get("box", {}), extracted, c)
            if photo_mask is None:
                if extracted:
                    c["shape_kind"] = c.get("shape_kind") or extracted["shape_kind"]
                    c["fill"] = c.get("fill") or extracted["fill"]
                    c["stroke"] = c.get("stroke") or extracted["stroke"]
                    if not c.get("effects") and extracted["effects"]:
                        c["effects"] = extracted["effects"]
                    if c.get("radius") is None and extracted["radius"] not in (None, 0):
                        c["radius"] = extracted["radius"]
                    c["meta"]["style_extraction"] = extracted["meta"]
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
        # Do retain broad photo frames for the layout regression contract, but
        # never export an empty semantic cutout (product/person/icon) merely
        # because a frontmost owner already claims all of its pixels.
        if not np.any(owned) and (is_text_fallback or role not in {"photo", "image"}):
            c["target"] = "drop"
            c["meta"]["keep_in_background"] = True
            c["meta"]["suppression_reason"] = "fully-contained-in-foreground-owner"
            updated.append(c)
            continue
        image = _source_rgba(c, rgb, mask, run_dir)
        image = _apply_owned_alpha(image, owned, c.get("box", {}))
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
                c["fill"] = {"kind": "flat", "color": traced["paths"][0].get("fill", "#000000")}
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
    scene_vlm_required = bool((((cfg.get("vlm") or {}).get("scene_text") or {}).get("enabled")))
    max_text_mask_fraction = float(rcfg.get("max_text_mask_canvas_fraction", 0.035))
    for c in updated:
        if c.get("target") == "drop" and not (c.get("meta") or {}).get("removal_required"):
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
        dilate = inpaint.resolve_mask_dilate(c, cfg)
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
    def _is_text_removal(item):
        return item.get("target") == "text" or (
            item.get("target") == "drop" and (item.get("meta") or {}).get("removal_required")
        )
    text_removal = [item for item in removal if _is_text_removal(item)]
    large_removal = [item for item in removal if not _is_text_removal(item)]
    mask_path = os.path.join(run_dir, "removal_mask.png")
    Image.fromarray(union).save(mask_path)
    removal_ownership_path = os.path.join(run_dir, "removal_ownership.png")
    removal_scale = max(1, 65535 // max(1, len(removal_owner_index)))
    Image.fromarray((removal_ownership * removal_scale).astype(np.uint16)).save(
        removal_ownership_path
    )
    background_path = os.path.join(run_dir, "background_clean.png")
    regional_enabled = bool(((cfg.get("inpaint") or {}).get("regional") or {}).get("enabled", False))
    if regional_enabled:
        inpaint_result = inpaint.inpaint_regional(
            image_path, removal, union, background_path, cfg, run_dir,
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
        if text_removal and large_removal:
            inpaint_result = inpaint.inpaint_role_aware(
                image_path, {"text": text_union, "large": large_union}, background_path, cfg,
            )
        else:
            inpaint_result = inpaint.inpaint_once(image_path, union, background_path, cfg)

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
            "mask_rejected": mask_rejected,
            "inpaint": inpaint_result,
        },
    }
    dump(result, os.path.join(run_dir, "reconstruction.json"))
    return result
