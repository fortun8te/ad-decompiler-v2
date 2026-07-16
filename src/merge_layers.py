"""merge_layers.py — stage 6: fuse OCR + element_detect + Qwen into routed candidates.

merge(ocr, elements, qwen, canvas, cfg) turns the three evidence streams into ONE
candidate list that build_design_json can consume:

  * every OCR line          -> a text candidate (crisp text + box)
  * every detected element  -> a shape / icon / photo candidate (crisp box + mask)
  * every Qwen layer        -> supplies back-to-front z-order + clean RGBA alpha,
                               matched to overlapping OCR/element candidates by IoU

Each candidate is passed through routing.route(candidate, canvas, cfg), which sets
`target` in {text, shape, image, icon, drop}. routing.py is owned by another builder;
if it is not importable yet we fall back to a conservative inline router (used ONLY at
runtime, never written to disk).

Post-routing cleanup:
  * dedup: a shape/icon candidate that is essentially an OCR text box is dropped
    (prefer the editable text candidate)
  * scene text: an OCR line sitting inside a photo region is flagged
    meta.kept_in_photo=True and routed to `drop` (it stays baked into the base)

Candidates carry everything downstream needs: box, target, z, and either
text/style, or src(alpha png)+mask, or source_crop, plus meta{source,role,confidence}.
"""
from __future__ import annotations
import importlib
import os
import copy
import difflib
import json
import re
from collections import Counter
from typing import Optional
from .wordmark import is_platform_lockup, semantic_text_role
from .raster_clusters import is_intentional_raster_cluster
from .raster_clusters import normalized_role as cluster_normalized_role
from .diagram_editability import (
    is_chart_label_role,
    is_chart_primitive_role,
    prefer_decomposed_charts,
)


def _load_routing():
    """Return the real routing.route, or a conservative inline fallback."""
    for name in ("src.routing", "routing"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "route"):
                return mod.route, True
        except ImportError:
            continue
    return _fallback_route, False


def _iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def _inside_frac(a, b):
    """Fraction of a inside b."""
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1.0, a["w"] * a["h"])


# ── discrete product/photo cutouts own their printed text ─────────────────────────────
# Text physically printed on a product package/photo cutout (a label like "WHEY
# MILKSHAKE", a nutrition table, an ingredient list) is NOT overlay copy: the pixels
# already live inside the product raster. Emitting it as a native text layer double-
# prints it (native glyphs stacked on the cutout that still shows them) and, worse, the
# removal mask then erases the label from the product. These roles name a *bounded
# object* cutout, never the flat plate or a full-bleed hero photo.
_PRODUCT_CUTOUT_ROLES = frozenset({
    "product", "package", "packaging", "bottle", "jar", "tube", "can", "canister",
    "pouch", "sachet", "shaker", "box", "carton", "device", "sign", "label",
    "photo", "photo-fragment", "photo_fragment", "person", "people", "portrait",
    "hand", "product-cluster",
    "logo",  # still a cutout owner for geometry; text-bearing shells extract OCR below
})

# Flat UI chrome that hosts overlay copy: native TEXT + solid/plate fill, never a
# baked OCR raster. Includes "logo" because detectors routinely tag price seals /
# offer bursts as logos (benchmark 016 green seal). Irregular brushstroke banners
# and starburst seals arrive as banner/starburst/shape and are promoted the same way.
_TEXT_BEARING_SHELL_ROLES = frozenset({
    "badge", "button", "chip", "pill", "cta", "callout", "sticker",
    "price_burst", "sale_burst", "starburst", "burst", "splat", "sticker_burst",
    "logo",
    "banner", "ribbon", "brushstroke", "stroke_banner", "seal",
})

# Roles that may *become* text-bearing shells via geometry (high OCR inside_frac).
# Broader than _TEXT_BEARING_SHELL_ROLES: SAM often labels a brushstroke "shape".
_SHELL_HOST_CANDIDATE_ROLES = _TEXT_BEARING_SHELL_ROLES | frozenset({
    "shape", "card", "panel", "frame", "container", "plate", "",
})

# Never treat these as dashed layout-guide junk (Biomel outline pills / SAVE seals).
_SHELL_GUIDE_EXEMPT_ROLES = frozenset({
    "badge", "button", "chip", "pill", "cta", "callout", "seal", "sticker",
    "starburst", "price_burst", "sale_burst", "burst", "splat", "sticker_burst",
    "banner", "ribbon", "brushstroke", "stroke_banner",
})

# Offer / SAVE seal copy — near-square hollow or filled chrome promotes like a starburst.
_SAVE_BADGE_RE = re.compile(
    r"(?i)\b(?:save\s*\d|subscribe\s*&\s*save|get\s+up\s+to|"
    r"%\s*off|\d+\s*%|£\s*\d|\$\s*\d|€\s*\d|\d+[.,]\d+\s*/\s*day)\b",
)

# Product/person cutouts that must NEVER be promoted to editable-shell chrome.
_BAKE_CUTOUT_ROLES = _PRODUCT_CUTOUT_ROLES - {"logo"}

# Photographic PANEL owners (comparison sides, lifestyle crops). A person/photo panel
# rarely carries printed copy of its own: checklist/card/label text sitting on it is
# normally composited by the designer, not photographed ink. Product/pack faces are
# deliberately NOT here — printed-on-product stays baked (091/094/002 invariant).
_PHOTOGRAPHIC_PANEL_ROLES = frozenset({
    "photo", "photo-fragment", "photo_fragment", "person", "people", "portrait",
})


def _positive_scene_ink_evidence(meta: dict, ownership: dict | None) -> bool:
    """True only with a REAL photographic-scene verdict for this OCR line.

    Fail-open contract (009 tweet body / 025 checklist cards): the failed-closed VLM
    sentinels (vlm_error / vlm_parse_error / vlm_disagreement, confidence 0.0) are NOT
    scene evidence. Text the archetype preset promises editable must never bake on
    *missing* data — only a positive "this ink is photographed/printed in the scene"
    verdict may keep it in the raster.
    """
    ownership = ownership or {}
    reason = str(ownership.get("reason") or "")
    real_printed_verdict = (
        ownership.get("action") == "raster_keep"
        and str(ownership.get("placement") or "") == "printed"
        and float(ownership.get("confidence") or 0) > 0.01
        and reason not in {"vlm_disagreement", "vlm_error", "vlm_parse_error"}
    )
    return bool(
        real_printed_verdict
        or str(meta.get("scene_text_role") or "") == "printed_on_product"
    )


def _is_full_bleed(box: dict, canvas: dict, frac: float = 0.92) -> bool:
    """True when ``box`` is the canvas plate/hero, not an inset product cutout.

    A full-bleed background photo is the canvas, so deliberate overlay copy on top of
    it must stay editable — unless photographic-scene mode decides the whole ad is
    in-image text. Coverage alone is not enough: a large inset can/package that fills
    most of a story frame must remain a discrete cutout that owns its printed labels.
    """
    cw = float((canvas or {}).get("w", 0) or 0)
    ch = float((canvas or {}).get("h", 0) or 0)
    if cw <= 0 or ch <= 0:
        return False
    x = float((box or {}).get("x", 0) or 0)
    y = float((box or {}).get("y", 0) or 0)
    w = float((box or {}).get("w", 0) or 0)
    h = float((box or {}).get("h", 0) or 0)
    if not ((w >= frac * cw) and (h >= frac * ch)):
        return False
    # Inset products often cover >90% of one axis; require near-origin anchoring so a
    # right-side can is never mistaken for the plate.
    return (x <= 0.03 * cw) and (y <= 0.03 * ch)


_PHOTOGRAPHIC_SURFACE_ROLES = frozenset({
    "photo", "photo-fragment", "photo_fragment", "product", "package", "packaging",
    "bottle", "jar", "tube", "can", "canister", "pouch", "sachet", "shaker",
    "box", "carton", "device", "sign", "label", "person", "people", "portrait",
    "image", "foreground", "cutout",
})

# Roles that prove a product-vs-overlay split is possible (007 can on a flat plate).
# person/photo chips inside a full-bleed photo (021 sticky faces) must NOT disable
# photographic-scene mode.
_PRODUCT_VS_OVERLAY_ROLES = frozenset({
    "product", "package", "packaging", "bottle", "jar", "tube", "can", "canister",
    "pouch", "sachet", "shaker", "box", "carton", "device", "sign", "label",
})

# Semantic roles that usually mark intentional marketing overlay. Package labels that
# OCR mis-tags as these still bake when they sit in a *bounded* cutout; an oversized
# SAM merge of plate+product must not swallow left-column overlay via these roles.
_OVERLAYISH_TEXT_ROLES = frozenset({
    "headline", "title", "subtitle", "subheadline", "eyebrow",
    "cta", "button", "price", "offer",
})


def _canvas_area(canvas: dict) -> float:
    return max(1.0, float((canvas or {}).get("w", 0) or 0)
               * float((canvas or {}).get("h", 0) or 0))


def _box_area_frac(box: dict, canvas: dict) -> float:
    w = float((box or {}).get("w", 0) or 0)
    h = float((box or {}).get("h", 0) or 0)
    return (w * h) / _canvas_area(canvas)


def _scene_cutout_owner(box: dict, regions: list[dict], threshold: float):
    """Smallest discrete cutout region that ``box`` sits at least ``threshold`` inside.

    Prefer text-bearing shell chrome (badge/button/logo seal) over a larger product
    pouch that also contains the text — otherwise 016-style offer seals bake OCR into
    the product raster. Among equal preference, smallest area wins.
    """
    matches = [r for r in regions if _inside_frac(box, r["box"]) >= threshold]
    if not matches:
        return None

    def _rank(region):
        role = str(region.get("role") or "").lower()
        shell = 0 if role in _TEXT_BEARING_SHELL_ROLES else 1
        area = float(region["box"].get("w", 0) or 0) * float(region["box"].get("h", 0) or 0)
        return (shell, area, str(region.get("id") or ""))

    return min(matches, key=_rank)


def _classify_shell_role(
    shell_box: dict,
    current_role: str,
    *,
    stroke_outline: bool = False,
    snippet: str = "",
) -> str:
    """Pick a designer-facing shell role from geometry + detector label.

    Wide irregular plates (brushstroke banners) → ``banner``; square seals → ``badge``.
    Stroke-outline elongated pills → ``callout`` (Biomel benefit chips). Explicit
    button/starburst labels are preserved. SAVE/£ seal copy on near-square chrome → seal.
    """
    role = str(current_role or "").lower().replace("-", "_")
    if role in {"button", "cta"}:
        return "button" if role == "cta" else role
    if role == "callout":
        return "callout"
    if role in {"chip", "pill"} and not stroke_outline:
        return role
    if role in {
        "starburst", "price_burst", "sale_burst", "burst", "splat", "sticker_burst", "seal",
    }:
        return role
    if role in {"banner", "ribbon", "brushstroke", "stroke_banner"}:
        return "banner"
    if role == "badge" and not stroke_outline:
        return "badge"
    w = float((shell_box or {}).get("w", 0) or 0)
    h = float((shell_box or {}).get("h", 0) or 0)
    aspect = (w / h) if h > 0 else 0.0
    text = str(snippet or "")
    if _SAVE_BADGE_RE.search(text) and 0.75 <= aspect <= 1.45:
        return "seal" if role in {"", "shape", "badge", "logo", "icon"} else (
            role if role in {"starburst", "price_burst", "sale_burst", "seal"} else "badge"
        )
    if stroke_outline and aspect >= 1.55:
        return "callout"
    if stroke_outline and 0.75 <= aspect <= 1.45:
        return "badge"
    if h > 0 and aspect >= 2.2:
        return "banner"
    return "badge"


def _promote_geometric_text_shells(candidates: list, canvas: dict, cfg: dict, diagnostics: dict) -> int:
    """Flag non-photo shells that geometrically host OCR as TEXT + plate_shell.

    Fast geometry only (no VLM): when editable OCR sits mostly inside a bounded
    shape/icon (brushstroke banner, starburst seal), mark the host ``text_bearing_shell``
    / ``plate_shell``, keep the text as overlay TEXT, and never bake glyphs into the
    shell raster. Product/person cutouts are excluded so packaging text stays baked.
    """
    mcfg = (cfg or {}).get("merge") or {}
    if mcfg.get("geometric_text_shells", True) is False:
        return 0
    threshold = float(mcfg.get("text_shell_inside", 0.55))
    texts = [
        c for c in candidates
        if c.get("target") == "text"
        and not c.get("kept_in_photo")
        and (c.get("text") or (c.get("meta") or {}).get("text"))
    ]
    if not texts:
        return 0
    promoted = 0
    for shell in candidates:
        meta = shell.setdefault("meta", {})
        if meta.get("text_bearing_shell") or meta.get("plate_shell"):
            continue
        if shell.get("target") not in {"shape", "icon", "image"}:
            continue
        role = str(meta.get("role") or "").lower()
        if role in _BAKE_CUTOUT_ROLES:
            continue
        if role not in _SHELL_HOST_CANDIDATE_ROLES and shell.get("target") not in {"shape", "icon"}:
            continue
        if _is_full_bleed(shell.get("box") or {}, canvas):
            continue
        hosts = [
            t for t in texts
            if _inside_frac(t.get("box") or {}, shell.get("box") or {}) >= threshold
        ]
        if not hosts:
            continue
        # Near-coincident OCR boxes are text layers, not chrome plates. Leave those to
        # the element-is-OCR-box dedup (CLICK-sized residual CCs hugging a glyph box).
        shell_box = shell.get("box") or {}
        if any(
            _iou(t.get("box") or {}, shell_box) >= 0.72
            or _inside_frac(shell_box, t.get("box") or {}) >= 0.85
            for t in hosts
        ):
            continue
        shell_area = float(shell_box.get("w", 0) or 0) * float(shell_box.get("h", 0) or 0)
        host_area = sum(
            float((t.get("box") or {}).get("w", 0) or 0)
            * float((t.get("box") or {}).get("h", 0) or 0)
            for t in hosts
        )
        if shell_area <= 0 or host_area / shell_area > 0.92:
            continue
        snippet = " ".join(
            str(t.get("text") or (t.get("meta") or {}).get("text") or "").strip()
            for t in hosts
        ).strip()
        stroke_outline = _is_stroke_outline_plate(shell)
        # Residual segmentation often returns the *negative space around copy* as one
        # large, dense component.  Treating that component as a banner/badge changes a
        # harmless source slice into a semantic parent and causes the reconstruction
        # stage to erase/repaint most of the ad (002's white headline/product slabs).
        # Real wide banners are still eligible when a detector names their role; this
        # guard only rejects generic residual shells with implausibly broad geometry.
        canvas_area = (
            float((canvas or {}).get("w", 0) or 0)
            * float((canvas or {}).get("h", 0) or 0)
        )
        shell_w = float(shell_box.get("w", 0) or 0)
        shell_h = float(shell_box.get("h", 0) or 0)
        area_frac = shell_area / canvas_area if canvas_area > 0 else 0.0
        width_frac = (
            shell_w / float((canvas or {}).get("w", 0) or 1)
            if float((canvas or {}).get("w", 0) or 0) > 0 else 0.0
        )
        provenance_sources = {
            str(source).lower()
            for source in ((meta.get("provenance") or {}).get("sources") or [])
        }
        generic_residual = role in {"", "shape", "photo"} and any(
            "residual" in source for source in provenance_sources
        )
        if not stroke_outline and (
            area_frac >= 0.35
            or (generic_residual and area_frac >= 0.05 and width_frac >= 0.70)
        ):
            diagnostics.setdefault("scene_text_contract", []).append({
                "id": shell.get("id"),
                "action": "reject-geometric-text-shell",
                "reason": "oversized-residual-shell",
                "area_frac": round(area_frac, 4),
                "width_frac": round(width_frac, 4),
                "text_ids": [t.get("id") for t in hosts],
            })
            # A broad but non-giant residual strip around copy is negative space,
            # not an independent visual layer. Shipping it as a confidence raster
            # slice duplicates the original glyphs underneath the native headline.
            # Keep giant regions neutral (they may host product cutouts), but suppress
            # this narrow text-plate signature and let the clean plate + TEXT own it.
            if generic_residual and area_frac < 0.35 and width_frac >= 0.70:
                meta["residual_text_plate"] = True
                meta["keep_in_background"] = True
                meta["suppression_reason"] = "residual-negative-space-around-text"
                shell["target"] = "drop"
            continue
        new_role = _classify_shell_role(
            shell_box, role, stroke_outline=stroke_outline, snippet=snippet,
        )
        if role and role != new_role:
            meta.setdefault("reclassified_from", role)
        meta["role"] = new_role
        meta["text_bearing_shell"] = True
        meta["plate_shell"] = True
        meta["geometric_text_shell"] = True
        if stroke_outline:
            meta["stroke_outline_shell"] = True
            meta["stroke_only"] = True
        if snippet:
            meta["shell_text_snippet"] = snippet[:48]
        shell["target"] = "shape"
        if new_role == "button":
            meta["button_shell"] = True
        for t in hosts:
            tm = t.setdefault("meta", {})
            tm["overlay_text"] = True
            tm["removal_required"] = True
            tm["parent_id"] = shell.get("id")
            tm["shell_text_host"] = shell.get("id")
            tm["ownership_enforced"] = True
            # Benefit copy on outline pills → callout for Figma naming.
            if new_role == "callout" and str(tm.get("role") or "").lower() in {
                "", "body", "body-copy", "body_copy", "copy", "label", "text",
            }:
                tm["role"] = "callout"
            t["target"] = "text"
            t.pop("kept_in_photo", None)
            for key in ("suppression_reason", "baked_owner_id", "scene_text_geometric"):
                tm.pop(key, None)
        diagnostics.setdefault("scene_text_contract", []).append({
            "id": shell.get("id"),
            "action": "geometric-text-bearing-shell",
            "host_role": new_role,
            "stroke_outline": bool(stroke_outline),
            "text_ids": [t.get("id") for t in hosts],
        })
        promoted += 1
    return promoted


def _is_stroke_outline_plate(candidate: dict) -> bool:
    """True for hollow/perimeter-ink plates (Biomel outline pills), not solid chrome."""
    meta = candidate.get("meta") or {}
    if meta.get("stroke_outline_shell"):
        return True
    if meta.get("stroke_only") or meta.get("dashed") or meta.get("dash"):
        return True
    # Explicit stroke + transparent/missing fill (detector or reconstruct hint).
    stroke = candidate.get("stroke") if "stroke" in candidate else meta.get("stroke")
    fill = candidate.get("fill") if "fill" in candidate else meta.get("fill")
    if stroke and not fill:
        return True
    fill_frac = _box_fill_fraction(candidate)
    # Perimeter ring / scalloped outline: high box area, sparse mask ink.
    if fill_frac is not None and fill_frac <= 0.34:
        return True
    return False


def _has_product_vs_overlay_split(product_regions: list[dict], canvas: dict) -> bool:
    """True when a substantial package cutout or overlay chrome shell is present."""
    for region in product_regions:
        role = str(region.get("role") or "").lower()
        area = _box_area_frac(region.get("box") or {}, canvas)
        if role in _TEXT_BEARING_SHELL_ROLES and area >= 0.008:
            return True
        if role in _PRODUCT_VS_OVERLAY_ROLES and area >= 0.08:
            return True
    return False


_COMPARISON_SIDE_BEFORE_RE = re.compile(
    r"^\s*(before|without|struggle|problem|patched(?:\s+together)?)\s*$",
    re.I,
)
_COMPARISON_SIDE_MID_RE = re.compile(r"^\s*(ritual|middle|during)\s*$", re.I)
_COMPARISON_SIDE_AFTER_RE = re.compile(
    r"^\s*(after|with|answer|solution|reset|daily(?:\s+im8)?)\s*$",
    re.I,
)
_COMPARISON_VS_RE = re.compile(r"^\s*(vs\.?|versus)\s*$", re.I)
_COMPARISON_PHOTO_ROLES = frozenset({
    "photo", "image", "photo-card", "product", "package", "person", "people",
    "portrait", "comparison-column", "comparison-panel", "photo-panel", "image-panel",
    "panel", "photo-fragment", "photo_fragment", "product-cluster", "hand", "canister",
    "sachet", "pouch", "packshot",
})


def _tag_comparison_columns(candidates: list, canvas: dict, cfg: dict | None) -> None:
    """Geometry-only before/after + VS tagging for comparison_grid scenes.

    Left/right photo cutouts get ``comparison_side``; VS OCR stays editable TEXT on a
    badge shell; Before/After (or WITHOUT/WITH) labels stay editable TEXT. No VLM.
    """
    scene = (cfg or {}).get("scene") or {}
    facts = scene.get("facts") or {}
    archetype = str(scene.get("archetype") or "")
    allow = (
        archetype == "comparison_grid"
        or facts.get("before_after_labels")
        or facts.get("before_after_pair")
        or facts.get("stage_progression")
        or int(facts.get("column_count") or 0) >= 2
    )
    if not allow:
        return

    photos = []
    for c in candidates:
        if c.get("target") != "image":
            continue
        role = str((c.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        if role not in _COMPARISON_PHOTO_ROLES and not (c.get("meta") or {}).get("comparison_side"):
            continue
        box = c.get("box") or {}
        if float(box.get("w") or 0) <= 0 or float(box.get("h") or 0) <= 0:
            continue
        # Ignore full-bleed plates — comparison frames are discrete cutouts.
        if _box_area_frac(box, canvas) >= 0.72:
            continue
        photos.append(c)

    photos = sorted(
        photos,
        key=lambda c: ((c.get("box") or {}).get("x", 0), c.get("id", "")),
    )
    if len(photos) >= 2:
        left, right = photos[0], photos[-1]
        if left is not right and left.get("id") != right.get("id"):
            lb, rb = left.get("box") or {}, right.get("box") or {}
            lh = max(1.0, float(lb.get("h") or 1))
            rh = max(1.0, float(rb.get("h") or 1))
            if abs(lh - rh) / max(lh, rh) <= 0.40:
                lcy = float(lb.get("y", 0)) + lh / 2
                rcy = float(rb.get("y", 0)) + rh / 2
                if abs(lcy - rcy) <= max(lh, rh) * 0.40:
                    group = f"cmp-{left.get('id')}-{right.get('id')}"
                    for side, node in (("before", left), ("after", right)):
                        meta = node.setdefault("meta", {})
                        meta.setdefault("comparison_side", side)
                        meta.setdefault("before_after_side", side)
                        meta.setdefault("comparison_group_id", group)
                        meta.setdefault(
                            "semantic_name",
                            "Photo / Before" if side == "before" else "Photo / After",
                        )

    for c in candidates:
        if c.get("target") == "drop":
            continue
        text = str(c.get("text") or "").strip()
        meta = c.setdefault("meta", {})
        # Editable overlay labels may briefly route as image on low fidelity; still tag
        # by OCR string so Before/After/VS names survive.
        if text:
            if _COMPARISON_SIDE_BEFORE_RE.match(text):
                token = text.strip()
                if re.fullmatch(r"struggle\.?", token, re.I):
                    side_name = "Struggle"
                elif re.fullmatch(r"problem\.?", token, re.I):
                    side_name = "Problem"
                elif re.fullmatch(r"patched(?:\s+together)?\.?", token, re.I):
                    side_name = "Patched"
                else:
                    side_name = "Before"
                meta["before_after_side"] = "before"
                meta["comparison_side"] = meta.get("comparison_side") or "before"
                meta.setdefault("semantic_name", side_name)
                if not meta.get("role") or meta.get("role") in {"text", "body", "label"}:
                    meta["role"] = "label"
                continue
            if _COMPARISON_SIDE_MID_RE.match(text):
                meta["before_after_side"] = "mid"
                meta["comparison_side"] = meta.get("comparison_side") or "mid"
                meta["stage_index"] = 1
                meta.setdefault("semantic_name", "Ritual")
                if not meta.get("role") or meta.get("role") in {"text", "body", "label"}:
                    meta["role"] = "label"
                continue
            if _COMPARISON_SIDE_AFTER_RE.match(text):
                # Prefer the literal token for IM8 stage labels (Reset / Answer / Daily).
                token = text.strip()
                if re.fullmatch(r"reset\.?", token, re.I):
                    side_name = "Reset"
                elif re.fullmatch(r"answer\.?", token, re.I):
                    side_name = "Answer"
                elif re.fullmatch(r"daily(?:\s+im8)?\.?", token, re.I):
                    side_name = "Daily"
                elif re.fullmatch(r"solution\.?", token, re.I):
                    side_name = "Solution"
                else:
                    side_name = "After"
                meta["before_after_side"] = "after"
                meta["comparison_side"] = meta.get("comparison_side") or "after"
                meta.setdefault("semantic_name", side_name)
                if not meta.get("role") or meta.get("role") in {"text", "body", "label"}:
                    meta["role"] = "label"
                continue
            if _COMPARISON_VS_RE.match(text):
                meta["role"] = "vs"
                meta.setdefault("semantic_name", "VS")
                host_id = meta.get("shell_text_host")
                if host_id:
                    for host in candidates:
                        if host.get("id") == host_id:
                            hm = host.setdefault("meta", {})
                            hm["text_bearing_shell"] = True
                            hm.setdefault("role", "badge")
                            hm.setdefault("semantic_name", "VS")
                            hm["shell_text_snippet"] = text
                            break
                continue
        # Shape/badge whose only purpose is hosting VS copy.
        snippet = str(meta.get("shell_text_snippet") or meta.get("text") or "").strip()
        role = str(meta.get("role") or "").lower()
        if _COMPARISON_VS_RE.match(snippet) or (
            role in {"badge", "chip", "seal", "shape", "vs", ""}
            and _COMPARISON_VS_RE.match(text)
        ):
            if _COMPARISON_VS_RE.match(snippet) or _COMPARISON_VS_RE.match(text):
                meta.setdefault("semantic_name", "VS")
                if role in {"", "shape"}:
                    meta["role"] = "badge"
                meta["text_bearing_shell"] = True



def _photographic_scene_text_mode(
    text_cands: list,
    elem_cands: list,
    canvas: dict,
    cfg: dict,
    product_regions: list[dict],
) -> bool:
    """True when OCR is in-image scene text (photo of handwriting / sticky notes / laptop).

    Prefer explicit scene facts / text policy. Geometric inference only fires when there
    is a dominant photographic surface, every OCR line lives inside it, no substantial
    product cutout explains a product-vs-overlay split, and no overlay structure
    (backplates / VLM overlay_copy) is present — the 021 sticky-note case.
    """
    scene = (cfg or {}).get("scene") or {}
    facts = scene.get("facts") or {}
    if facts.get("photo_of_handwriting") or facts.get("text_on_photographic_surfaces_only"):
        return True
    text_policy = ((cfg or {}).get("routing") or {}).get("text_policy") or {}
    if text_policy.get("scene_text_only") or text_policy.get("suppress_editable_ocr"):
        return True
    if not text_cands:
        return False
    # The geometric inference below exists for ONE case: a photograph whose text is
    # physically part of the scene (021 sticky-notes / printed packaging). Every other
    # archetype that can trip its conditions has text the contract demands editable:
    # social_screenshot (tweet body baked → 009 shipped 1 raster layer, native 0.00),
    # comparison_grid (before/after plates baked → 025 native 0.00 over 15 leaves),
    # lifestyle_overlay (H7: overlay copy on photos must peel, never bake). Allowlist
    # the single archetype the heuristic was designed for instead of chasing exclusions.
    if str(scene.get("archetype") or "") not in ("caption_over_photo", ""):
        return False
    # 007-style can + left column: product geometry decides. Tiny person/logo chips
    # inside a full-bleed photo must not block 021 photographic-scene mode.
    if _has_product_vs_overlay_split(product_regions, canvas):
        return False
    if int(facts.get("text_backplate_count") or 0) > 0:
        return False
    # Geometric inference needs real plate/photo evidence from image_facts. A lone
    # full-bleed photo element plus overlay headline (lifestyle) must stay editable
    # when no photo_coverage was measured.
    photo_cov = float(facts.get("photo_coverage") or 0)
    flat = float(facts.get("flat_background_fraction") or 0)
    if photo_cov < 0.55:
        return False
    if flat >= 0.45:
        return False

    photo_surfs = []
    for c in elem_cands:
        role = str((c.get("meta") or {}).get("role") or "").lower()
        box = c.get("box") or {}
        if role not in _PHOTOGRAPHIC_SURFACE_ROLES:
            continue
        if not (float(box.get("w", 0) or 0) > 0 and float(box.get("h", 0) or 0) > 0):
            continue
        photo_surfs.append(c)
    if not photo_surfs:
        return False
    dominant = [
        c for c in photo_surfs
        if _is_full_bleed(c.get("box") or {}, canvas) or _box_area_frac(c.get("box") or {}, canvas) >= 0.70
    ]
    if not dominant:
        return False

    for t in text_cands:
        meta = t.get("meta") or {}
        if (
            meta.get("scene_text_role") == "overlay_copy"
            or meta.get("external_overlay")
            or meta.get("promote_text")
            or meta.get("editable_text")
            or meta.get("text_promoted")
            or (meta.get("ownership_decision") or {}).get("action") == "recreate"
        ):
            return False
        box = t.get("box") or {}
        if not any(_inside_frac(box, (p.get("box") or {})) >= 0.70 for p in photo_surfs):
            return False
    return True


def _should_bake_in_cutout(
    candidate: dict,
    cutout_owner: dict,
    canvas: dict,
    scene_text_inside: float,
    facts: dict | None = None,
) -> bool:
    """Decide whether geometric cutout containment should bake this OCR line.

    Marketing overlay roles require a *bounded* cutout so an oversized SAM merge of
    plate+product cannot swallow left-column copy. Body/caption/label text and
    VLM-printed roles bake at the normal scene_text_inside threshold.
    """
    box = candidate.get("box") or {}
    owner_box = cutout_owner.get("box") or {}
    inside = _inside_frac(box, owner_box)
    if inside < scene_text_inside:
        return False
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    scene_role = str(meta.get("scene_text_role") or "").lower()
    if scene_role == "printed_on_product" or role in {"body", "caption", "label", "ingredients"}:
        return True
    owner_area = _box_area_frac(owner_box, canvas)
    if role in _OVERLAYISH_TEXT_ROLES:
        # Bounded package cutout: bold product names often read as headline/subheadline.
        # Keep the normal scene_text_inside threshold (ascenders spill past the mask).
        if owner_area < 0.60:
            return inside >= scene_text_inside
        # Oversized "product" on a flat-plate ad → treat overlayish roles as plate copy.
        flat = float((facts or {}).get("flat_background_fraction") or 0)
        if flat >= 0.30:
            return False
        return inside >= 0.85
    return True


def _raster_cluster_owner(box: dict, owners: list[dict], threshold: float):
    """Return the smallest positive intentional-raster owner around ``box``."""
    area = float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
    matches = [
        owner for owner in owners
        if (float((owner.get("box") or {}).get("w", 0) or 0)
            * float((owner.get("box") or {}).get("h", 0) or 0)) >= area
        and _inside_frac(box, owner.get("box") or {}) >= threshold
    ]
    if not matches:
        return None
    return min(matches, key=lambda owner: (
        owner["box"]["w"] * owner["box"]["h"], owner["id"],
    ))


def _normalize_text_key(value):
    return " ".join(str(value or "").strip().upper().split())


# ── content-similarity helpers ───────────────────────────────────────────────────────
# Dedup keys on geometry overlap AND content, never IoU alone: two OCR engines (or the
# block-vs-orphan-line paths) that describe the same physical text usually disagree only
# on separators/whitespace, so we compare normalized alphanumeric/currency tokens.
_CONTENT_TOKEN_RE = re.compile(r"[0-9A-Z€£$%]+")


def _content_tokens(value):
    """Uppercased alphanumeric/currency runs; OCR-noisy punctuation/whitespace dropped."""
    return _CONTENT_TOKEN_RE.findall(_normalize_text_key(value))


def _token_containment(inner, outer):
    """Multiset fraction of ``inner`` tokens also present in ``outer`` (0..1)."""
    inner_tokens = _content_tokens(inner)
    if not inner_tokens:
        return 0.0
    available = Counter(_content_tokens(outer))
    hit = 0
    for token in inner_tokens:
        if available.get(token, 0) > 0:
            available[token] -= 1
            hit += 1
    return hit / len(inner_tokens)


def _text_similarity(a, b):
    """Character-level ratio on normalized text (0..1); 1.0 on an exact normalized match."""
    na, nb = _normalize_text_key(a), _normalize_text_key(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _geometry_overlap(a, b):
    """Return (iou, fraction_of_smaller_box_inside_the_larger) — orientation-independent."""
    iou = _iou(a, b)
    area_a = float(a.get("w", 0) or 0) * float(a.get("h", 0) or 0)
    area_b = float(b.get("w", 0) or 0) * float(b.get("h", 0) or 0)
    small, big = (a, b) if area_a <= area_b else (b, a)
    return iou, _inside_frac(small, big)


def _dedup_overlapping_text(candidates, merge_cfg: dict, dedup_iou: float, diagnostics=None):
    """Collapse text candidates that describe the SAME physical text but arrived from
    different sources (OCR ensemble members, block-vs-orphan line, fragment recombination).

    A pair is a duplicate when it overlaps geometrically AND one's content is essentially
    contained in the other (content-similarity, not IoU alone). The most complete
    observation survives; the redundant fragment is dropped with provenance recorded on the
    keeper (``meta.deduped_text_ids``) and in the merge diagnostics.

    Real failure this fixes: run 009 emitted the timestamp row three times
    ('05:00 PM . 12-05-2026 - 121K weergaven', '05:00 PM', '12-05-2026 121K weergaven')
    because one engine read the whole row and another split it into two fragments; all three
    survived as separate sole-member blocks and re-rendered on top of each other.
    """
    overlap_thresh = float(merge_cfg.get("dedup_text_overlap", 0.6))
    content_thresh = float(merge_cfg.get("dedup_text_content", 0.8))
    sim_thresh = float(merge_cfg.get("dedup_text_similarity", 0.85))

    texts = [
        c for c in candidates
        if (c.get("meta") or {}).get("source") == "ocr"
        and c.get("target") != "drop"
        and str(c.get("text") or "").strip()
    ]

    def _order_key(c):
        # Best (most complete) observation first: more content tokens, longer text, higher
        # confidence, larger visible coverage. Ascending id is the final deterministic
        # tie-break so the surviving layer is reproducible across runs.
        box = c.get("visible_box") or c.get("box") or {}
        return (
            -len(_content_tokens(c.get("text"))),
            -len(_normalize_text_key(c.get("text"))),
            -float((c.get("meta") or {}).get("confidence", 0) or 0),
            -(float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)),
            str(c.get("id") or ""),
        )

    ordered = sorted(texts, key=_order_key)
    dropped = {}
    for index, keeper in enumerate(ordered):
        if keeper.get("id") in dropped:
            continue
        for other in ordered[index + 1:]:
            oid = other.get("id")
            if oid in dropped:
                continue
            iou, inside = _geometry_overlap(keeper.get("box", {}), other.get("box", {}))
            # Same normalized string overlapping in place is a near-certain duplicate, so
            # accept a lower overlap (IoU/inside >= 0.5) for an exact match than the general
            # content-similarity gate requires — this collapses same-string promo/ribbon text
            # (131) that lands just under the 0.6 geometry thresholds. Different strings still
            # need the stricter overlap.
            exact_key = _normalize_text_key(keeper.get("text"))
            exact_match = bool(exact_key) and _normalize_text_key(other.get("text")) == exact_key
            geom_ok = (
                inside >= overlap_thresh or iou >= dedup_iou
                or (exact_match and (iou >= 0.5 or inside >= 0.5))
            )
            if not geom_ok:
                continue
            containment = _token_containment(other.get("text"), keeper.get("text"))
            similarity = _text_similarity(other.get("text"), keeper.get("text"))
            if containment < content_thresh and similarity < sim_thresh:
                continue
            dropped[oid] = {
                "dropped": oid,
                "kept": keeper.get("id"),
                "reason": "duplicate-text-overlapping-geometry",
                "content_containment": round(containment, 3),
                "text_similarity": round(similarity, 3),
                "iou": round(iou, 3),
                "inside_frac": round(inside, 3),
            }
            keeper_meta = keeper.setdefault("meta", {})
            keeper_meta.setdefault("deduped_text_ids", []).append(oid)
            decorations = (other.get("meta") or {}).get("native_decoration_shapes") or []
            if decorations:
                keeper_meta.setdefault("native_decoration_shapes", []).extend(
                    copy.deepcopy(decorations)
                )
    if diagnostics is not None:
        diagnostics.extend(dropped[oid] for oid in dropped)
    if not dropped:
        return candidates
    return [c for c in candidates if c.get("id") not in dropped]


def _suppressed_fragment_line_ids(lines_by_id: dict, blocks: list, diagnostics=None):
    """OCR line ids to drop from *multi-line* blocks because they are near-duplicate
    fragments of a richer line owned by a DIFFERENT block.

    This is the fragment-recombination failure (run 009 'geld'): OCR read the wrapped line
    'geld terug tot €100.' as its own line/block AND a lone 'geld' fragment that
    text_analysis absorbed into the body paragraph, so 'geld' rendered twice at different
    offsets. Only members of blocks that retain another line are eligible here, so a whole
    duplicate block or orphan line is left for candidate-level dedup rather than emptied.
    """
    owner = {}
    member_count = {}
    for block in blocks:
        bid = block.get("id")
        ids = [lid for lid in (block.get("line_ids") or []) if lid in lines_by_id]
        member_count[bid] = len(ids)
        for lid in ids:
            owner[lid] = bid
    suppressed = {}
    for lid, line in lines_by_id.items():
        bid = owner.get(lid)
        if bid is None or member_count.get(bid, 0) < 2:
            continue  # orphan or sole-member block -> candidate dedup owns this case
        frag_tokens = _content_tokens(line.get("text"))
        if not frag_tokens:
            continue
        for other_id, other in lines_by_id.items():
            if other_id == lid or owner.get(other_id) == bid:
                continue
            if len(_content_tokens(other.get("text"))) <= len(frag_tokens):
                continue  # only a strictly richer line may subsume the fragment
            if _token_containment(line.get("text"), other.get("text")) < 0.9:
                continue
            _iou_v, inside = _geometry_overlap(line.get("box", {}), other.get("box", {}))
            if inside < 0.5:
                continue
            suppressed[lid] = {
                "line_id": lid,
                "block_id": bid,
                "superset_line_id": other_id,
                "reason": "fragment-duplicate-of-richer-line-in-another-block",
            }
            break
    if diagnostics is not None:
        diagnostics.extend(suppressed[lid] for lid in suppressed)
    return suppressed


# ── scene-text contract enforcement ──────────────────────────────────────────────────
# kept_in_photo scene text stays baked in the photo; reconstruct._is_text_removal erases
# pixels for any target=='drop' item that still carries removal_required, so a scene-text
# candidate holding an editable/overlay flag would be deleted from the plate without a
# replacement layer. These flags are mutually exclusive with baked scene text.
_SCENE_TEXT_EDITABLE_FLAGS = (
    "overlay_text", "removal_required", "promote_text", "editable_text",
    "text_promoted", "external_overlay",
)


def _enforce_scene_text_contract(candidate: dict, diagnostics=None):
    """Guarantee a kept_in_photo scene-text candidate is never ALSO an editable overlay."""
    if candidate.get("text") is None:
        return
    meta = candidate.setdefault("meta", {})
    is_scene = bool(
        candidate.get("kept_in_photo") or meta.get("kept_in_photo")
        or meta.get("origin") == "scene" or meta.get("role") == "scene-text"
    )
    if not is_scene:
        return
    candidate["target"] = "drop"
    candidate["kept_in_photo"] = True
    meta["kept_in_photo"] = True
    cleared = [flag for flag in _SCENE_TEXT_EDITABLE_FLAGS if meta.get(flag)]
    for flag in cleared:
        meta.pop(flag, None)
    if cleared:
        meta["scene_text_contract_enforced"] = cleared
        if diagnostics is not None:
            diagnostics.append({
                "id": candidate.get("id"),
                "cleared_flags": cleared,
                "reason": "scene-text-not-also-editable-overlay",
            })


def _dedup_text_candidates(candidates, merge_cfg: dict, dedup_iou: float):
    """Collapse ghosted/duplicate OCR text layers flagged by the harness or VLM critic."""
    if not merge_cfg.get("dedup_text"):
        return candidates
    drop_ids = set()
    layer_ids = [str(layer_id) for layer_id in (merge_cfg.get("layer_ids") or []) if layer_id]
    if layer_ids:
        drop_ids.update(layer_ids[1:])
    duplicate_texts = {
        _normalize_text_key(text)
        for text in (merge_cfg.get("duplicate_text") or [])
        if _normalize_text_key(text)
    }
    if duplicate_texts:
        texts = [
            c for c in candidates
            if c.get("target") == "text" and c.get("id") not in drop_ids
        ]
        by_text = {}
        for candidate in texts:
            key = _normalize_text_key(candidate.get("text"))
            if key not in duplicate_texts:
                continue
            by_text.setdefault(key, []).append(candidate)
        for group in by_text.values():
            if len(group) < 2:
                continue
            group.sort(
                key=lambda c: (
                    float((c.get("meta") or {}).get("confidence", 0) or 0),
                    -_inside_frac(c.get("box", {}), c.get("box", {})),
                    str(c.get("id", "")),
                ),
                reverse=True,
            )
            keeper = group[0]
            for duplicate in group[1:]:
                if _iou(keeper.get("box", {}), duplicate.get("box", {})) >= dedup_iou:
                    drop_ids.add(duplicate.get("id"))
                    keeper.setdefault("meta", {}).setdefault("deduped_text_ids", []).append(
                        duplicate.get("id")
                    )
    if not drop_ids:
        return candidates
    return [c for c in candidates if c.get("id") not in drop_ids]


# ── conservative inline router (fallback only) ───────────────────────────────────────
def _fallback_route(candidate: dict, canvas: dict, cfg: dict) -> dict:
    """Minimal port of routing intent, used ONLY if src.routing is unimportable.
    Reads the same top-level fields the real router does. Returns a copy."""
    c = dict(candidate)
    meta = c.setdefault("meta", {})
    kind = c.get("kind")
    if c.get("text") is not None and kind in (None, "text"):
        if c.get("kept_in_photo") or meta.get("origin") == "scene":
            c["target"] = "drop"
            meta["kept_in_photo"] = True
        else:
            c["target"] = "text"
    elif kind == "photo-fragment" or meta.get("role") in ("photo", "product", "person"):
        c["target"] = "image"
    elif kind == "icon" or meta.get("role") in ("icon", "badge", "logo", "arrow"):
        c["target"] = "icon"
    elif kind == "shape":
        c["target"] = "shape"
    else:
        c["target"] = "image"
        meta["fallback"] = True
    return c


# ── candidate builders ───────────────────────────────────────────────────────────────
def _word_aligned_text_runs(line: dict) -> list[dict]:
    """Map positively evidenced OCR-word styles to exact character ranges.

    OCR engines disagree about punctuation and whitespace, so matching is sequential
    and fail-closed. A word without ``style_evidence`` is deliberately left in the
    node's base style.
    """
    text = str(line.get("text") or "")
    base_signature = _run_style_signature(dict(line.get("style") or {}))
    cursor = 0
    runs = []
    for word in line.get("words") or []:
        if not isinstance(word, dict) or not word.get("style_evidence"):
            continue
        token = str(word.get("text") or "").strip()
        style = dict(word.get("style") or {})
        if not token or not style or _run_style_signature(style) == base_signature:
            continue
        # Exact search first. Whitespace-tolerant matching supports OCR tokens such
        # as ``30 %`` without allowing punctuation/content to drift.
        start = text.find(token, cursor)
        end = start + len(token) if start >= 0 else -1
        if start < 0 and any(ch.isspace() for ch in token):
            pattern = r"\s+".join(re.escape(part) for part in token.split())
            match = re.search(pattern, text[cursor:])
            if match:
                start, end = cursor + match.start(), cursor + match.end()
        if start < 0:
            continue
        runs.append({"start": start, "end": end, "style": copy.deepcopy(style)})
        cursor = end
    return runs


def _word_geometry_fractions(line) -> list[dict]:
    """Per-word geometry as FRACTIONS of the line box (coordinate-space independent).

    Fractions survive every later rebasing (layout parents relativize x/y, never scale),
    so the design compiler can recover each word's measured span inside whatever local
    box the candidate ends up with. Fail-closed: any missing/degenerate geometry
    returns [] and downstream keeps the old proportional-advance behavior.
    """
    if "\n" in str(line.get("text") or ""):
        return []
    words = line.get("words") or []
    if len(words) < 2:
        return []
    box = line.get("box") or {}
    try:
        lx, ly = float(box.get("x")), float(box.get("y"))
        lw, lh = float(box.get("w")), float(box.get("h"))
    except (TypeError, ValueError):
        return []
    if lw <= 1.0 or lh <= 1.0:
        return []
    out = []
    for word in words:
        if not isinstance(word, dict):
            return []
        token = str(word.get("text") or "").strip()
        wbox = word.get("box") or {}
        try:
            wx, wy = float(wbox.get("x")), float(wbox.get("y"))
            ww, wh = float(wbox.get("w")), float(wbox.get("h"))
        except (TypeError, ValueError):
            return []
        if not token or ww <= 0 or wh <= 0:
            return []
        out.append({
            "text": token,
            "fx": round((wx - lx) / lw, 6), "fw": round(ww / lw, 6),
            "fy": round((wy - ly) / lh, 6), "fh": round(wh / lh, 6),
        })
    return out


def _text_candidate(line):
    meta = dict(line.get("meta") or {})
    if line.get("baseline"):
        meta["baseline"] = dict(line["baseline"])
    # Top-level OCR ownership hints (VLM scene_text, tests) must land in meta —
    # otherwise printed/wordmark labels never reach the bake path.
    for key in ("scene_text_role", "ownership_decision", "kept_in_photo"):
        if line.get(key) is not None and meta.get(key) is None:
            meta[key] = line[key]
    declared_role = line.get("role") or meta.get("role")
    # OCR normally labels every observation merely "text". Replace only that
    # placeholder with a semantic role so Figma layers are easy to work with.
    if not declared_role or str(declared_role).strip().lower() in {"text", "body"}:
        declared_role = meta.get("semantic_role") or "text"
    meta.update({
        "source": "ocr",
        "role": declared_role,
        "confidence": round(float(line.get("conf", meta.get("confidence", 1.0))), 4),
        "ocr_id": line["id"],
        "line_ids": line.get("line_ids") or [line["id"]],
        "quad": line.get("quad"),
        "hierarchy": line.get("hierarchy"),
        "style_id": line.get("repeated_style_id") or line.get("style_id"),
    })
    word_geometry = _word_geometry_fractions(line)
    if word_geometry:
        meta["word_geometry"] = word_geometry
    style = dict(line.get("style") or {})
    candidate = {
        "id": f"c_{line['id']}",
        "box": dict(line["box"]),
        "z": 0,
        "text": line.get("text", ""),
        "style": style,
        # Paragraph assembly can preserve distinct styles for each source line.  Keep
        # those exact character ranges all the way to design.json; otherwise Figma sees
        # only the block's representative style and silently flattens bold/colour/font
        # changes across its lines.
        "text_runs": copy.deepcopy(line.get("text_runs") or _word_aligned_text_runs(line)),
        "visible_box": dict(line.get("ink_box") or line.get("painted_box") or line["box"]),
        "rotation": float(line.get("rotation", 0.0) or 0.0),
        "quad": line.get("quad"),
        "meta": meta,
    }
    # Promote paint effects so design/reconstruct see the same stroke/shadow as style.
    if style.get("fill") is not None:
        candidate["fill"] = copy.deepcopy(style.get("fill"))
    if style.get("stroke") is not None:
        candidate["stroke"] = copy.deepcopy(style.get("stroke"))
    if style.get("effects"):
        candidate["effects"] = copy.deepcopy(style.get("effects"))
    return candidate


_RUN_VISUAL_STYLE_KEYS = (
    "fontFamily", "font_family", "family", "fontStyle", "font_style",
    "fontWeight", "font_weight", "weight", "fontSize", "font_size", "size",
    "italic", "color", "colorRGB", "fill", "fills", "stroke", "strokes",
    "letterSpacing", "letter_spacing", "tracking", "lineHeight", "line_height",
    "leading", "textDecoration", "text_decoration", "decoration", "textCase",
    "text_case", "opacity",
)


def _run_style_signature(style: dict) -> str:
    """Return a stable visual-style key without letting evidence metadata create runs."""
    visible = {key: style[key] for key in _RUN_VISUAL_STYLE_KEYS if key in style}
    return json.dumps(visible, sort_keys=True, separators=(",", ":"), default=str)


def _line_aligned_text_runs(block_text: str, members: list[dict]) -> list[dict]:
    """Build per-line Figma ranges only when a paragraph really changes visual style.

    ``text_analysis`` builds a paragraph by joining OCR lines with literal newlines.  Check
    that invariant before emitting ranges so a malformed or post-corrected block cannot style
    the wrong characters.  Newline separators intentionally receive the node's base style.
    """
    texts = [str(member.get("text") or "") for member in members]
    if block_text != "\n".join(texts):
        return []
    styles = [dict(member.get("style") or {}) for member in members]
    line_styles_differ = len({_run_style_signature(style) for style in styles}) > 1
    runs = []
    cursor = 0
    for index, (text, style) in enumerate(zip(texts, styles)):
        end = cursor + len(text)
        word_runs = _word_aligned_text_runs(members[index])
        if end > cursor and line_styles_differ:
            runs.append({"start": cursor, "end": end, "style": copy.deepcopy(style)})
        if word_runs:
            # Preserve only the exceptional word ranges; the paragraph node's base
            # style already covers the rest of a same-style line.
            for run in word_runs:
                shifted = copy.deepcopy(run)
                shifted["start"] += cursor
                shifted["end"] += cursor
                runs.append(shifted)
        cursor = end + (1 if index < len(texts) - 1 else 0)
    return runs


def _inline_repeat_pattern(text: str) -> Optional[dict]:
    """Recognise an explicit ticker/marquee phrase repeated three or more times."""
    parts = [part.strip() for part in re.split(r"\s*(?:[•·|]|—|–)\s*", text) if part.strip()]
    if len(parts) < 3:
        return None
    normalized = [_normalize_text_key(part) for part in parts]
    if not normalized[0] or any(part != normalized[0] for part in normalized[1:]):
        return None
    return {"phrase": parts[0], "count": len(parts), "source": "exact-ocr-sequence"}


def _annotate_native_text_repetitions(candidates: list[dict]) -> None:
    """Label only exact, style-consistent repetition; never synthesize missing copies."""
    groups: dict[tuple, list[dict]] = {}
    for candidate in candidates:
        inline = _inline_repeat_pattern(str(candidate.get("text") or ""))
        if inline:
            candidate.setdefault("meta", {})["native_repeat"] = inline
            candidate["meta"]["repeat_signature"] = (
                "ticker:" + _normalize_text_key(inline["phrase"])
            )
        key = (
            _normalize_text_key(candidate.get("text")),
            _run_style_signature(dict(candidate.get("style") or {})),
        )
        if key[0]:
            groups.setdefault(key, []).append(candidate)
    for (text_key, _style), matches in groups.items():
        if len(matches) < 2:
            continue
        xs = [float(item["box"].get("x", 0)) for item in matches]
        ys = [float(item["box"].get("y", 0)) for item in matches]
        hs = [max(1.0, float(item["box"].get("h", 1))) for item in matches]
        row = max(ys) - min(ys) <= max(hs) * .65
        column = max(xs) - min(xs) <= max(hs) * .9
        if not (row or column):
            continue
        signature = "text-repeat:" + text_key
        ordered = sorted(matches, key=lambda item: (
            float(item["box"].get("x", 0)) if row else float(item["box"].get("y", 0)),
            str(item.get("id") or ""),
        ))
        for index, item in enumerate(ordered):
            item.setdefault("meta", {}).update({
                "repeat_signature": signature,
                "native_repeat": {
                    "count": len(ordered), "index": index,
                    "axis": "horizontal" if row else "vertical",
                    "source": "exact-ocr-observations",
                },
            })


def _text_sources(ocr, diagnostics=None):
    if not isinstance(ocr, dict):
        return ocr or []
    blocks = ocr.get("blocks") or []
    if not blocks:
        return ocr.get("lines", [])
    styles = {style.get("id"): style for style in (ocr.get("styles") or [])}
    lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    # Drop fragment lines that duplicate a richer line owned by a different block before
    # a block's paragraph text is assembled (run 009 'geld' render-twice bug).
    suppressed_fragments = _suppressed_fragment_line_ids(lines, blocks, diagnostics)
    out = []
    represented_line_ids = set()
    for raw in blocks:
        block = dict(raw)
        member_ids = [line_id for line_id in block.get("line_ids", []) if line_id in lines]
        kept_ids = [line_id for line_id in member_ids if line_id not in suppressed_fragments]
        members = [lines[line_id] for line_id in kept_ids]
        if len(kept_ids) != len(member_ids) and members:
            # Only rebuild the paragraph when text_analysis's newline-join invariant holds,
            # so a suppressed fragment is removed without mangling a post-corrected block.
            original_join = "\n".join(str(lines[lid].get("text") or "") for lid in member_ids)
            if str(block.get("text") or "") == original_join:
                block["text"] = "\n".join(str(m.get("text") or "") for m in members)
            else:
                members = [lines[line_id] for line_id in member_ids]  # fail-closed: keep all
        # Represent every ORIGINAL member (incl. suppressed) so a dropped fragment is not
        # resurrected below as an orphan candidate.
        represented_line_ids.update(line_id for line_id in member_ids if line_id)
        style_id = block.get("style_id")
        block_style = dict(block.get("style") or styles.get(style_id) or
                           (members[0].get("style") if members else {}) or {})
        if members:
            representative = members[0].get("style") or {}
            for key in ("fontCandidates", "fontSizeCandidates", "fontWeightCandidates",
                        "fontStyleCandidates", "confidence"):
                if key in representative:
                    block_style.setdefault(key, representative[key])
            # Font judging runs after block construction and updates OCR lines. Carry
            # its promoted winner into the actual downstream block instead of leaving
            # the stale pre-judge family on the editable Figma node.
            judged_member = next((line for line in members if line.get("vlm_font_judged")), None)
            if judged_member:
                judged_style = judged_member.get("style") or {}
                for key in ("fontFamily", "fontStyle", "fontWeight", "fontCandidates"):
                    if key in judged_style:
                        block_style[key] = copy.deepcopy(judged_style[key])
        # Blocks are the actual downstream text nodes. Preserve paragraph facts on the
        # block itself rather than relying on the first OCR line's one-line style.
        block_style.setdefault("align", block.get("alignment") or
                               (members[0].get("style") or {}).get("align") if members else "LEFT")
        if block.get("line_height") is not None:
            block_style["lineHeight"] = block["line_height"]
        block_style["lineCount"] = max(1, len(members), str(block.get("text") or "").count("\n") + 1)
        # Single-line blocks inherit their line's word boxes so downstream word-level
        # geometry (weight-run sibling split, decoration anchoring) can place each
        # word at its MEASURED position instead of a proportional font-advance guess
        # (benchmark 002: "€63 €49" has an arrow-sized gap the advance model cannot see).
        if len(members) == 1 and members[0].get("words") and not block.get("words"):
            block["words"] = copy.deepcopy(members[0]["words"])
        if members and members[0].get("baseline"):
            block.setdefault("meta", {})["baseline_first"] = dict(members[0]["baseline"])
        if members and members[-1].get("baseline"):
            block.setdefault("meta", {})["baseline_last"] = dict(members[-1]["baseline"])
        decoration_shapes = []
        for member in members:
            decoration_shapes.extend(copy.deepcopy(
                (member.get("meta") or {}).get("native_decoration_shapes") or []
            ))
        if decoration_shapes:
            block.setdefault("meta", {})["native_decoration_shapes"] = decoration_shapes
        block["style"] = block_style
        text_runs = _line_aligned_text_runs(str(block.get("text") or ""), members)
        if text_runs:
            block["text_runs"] = text_runs
        else:
            # A raw/resumed block can carry stale ranges from a prior version.  Never let
            # those ranges address a newly assembled paragraph.
            block.pop("text_runs", None)
        block["ink_box"] = block.get("painted_box") or block.get("box")
        block["conf"] = (sum(float(line.get("conf", 1)) for line in members) / len(members)
                         if members else float(block.get("conf", 1)))
        block["rotation"] = (sum(float(line.get("rotation", 0)) for line in members) / len(members)
                             if members else float(block.get("rotation", 0)))
        block["repeated_style_id"] = style_id
        out.append(block)
    # A partial/malformed block list must never delete otherwise valid OCR observations.
    # This happened in a live run where OCR saw text but only the surviving blocks reached
    # merge/design, producing text_recall=0. Preserve every orphan as its own text node.
    out.extend(copy.deepcopy(line) for line_id, line in lines.items()
               if line_id and line_id not in represented_line_ids)
    return out


def _canonical_element_id(raw_id) -> Optional[str]:
    """Map a fused element id (``E010``) onto the merge candidate id (``c_E010``)."""
    if raw_id in (None, ""):
        return None
    text = str(raw_id)
    return text if text.startswith("c_") else f"c_{text}"


def _element_candidate(el):
    kind = el.get("kind", "shape")  # top-level kind -> routing.route reads this
    role = el.get("role") or {"shape": "shape", "icon": "icon", "photo-fragment": "photo"}.get(kind, "shape")
    raw_mask = el.get("mask")
    mask_src = (raw_mask.get("src") if isinstance(raw_mask, dict) else raw_mask)
    mask_src = mask_src or el.get("mask_src") or el.get("mask_path")
    structural_meta = {
        key: el.get(key)
        for key in (
            "structure_group_id", "repeat_group_id", "panel_set_id", "grid_group_id",
            "comparison_group_id", "chart_group_id", "row_index", "column_index",
        )
        if el.get(key) not in (None, "")
    }
    metadata = copy.deepcopy(el.get("meta") or {})
    raw_parent = el.get("parent_id") or metadata.get("parent_id")
    metadata.update({
        "source": "element", "role": role, "kind": kind,
        "confidence": round(float(el.get("score", el.get("coverage", 0.0))), 4),
        "element_id": el["id"], "area": el.get("area"), "prompt": el.get("prompt"),
        # Fusion records parent_id against the raw element id; candidates are c_<id>.
        "parent_id": _canonical_element_id(raw_parent),
        "observations": el.get("observation_ids") or metadata.get("observations") or [],
        "provenance": el.get("provenance") or metadata.get("provenance") or {},
        **structural_meta,
    })
    # Preserve detector stroke/fill hints so hollow outline pills (Biomel) promote as
    # stroke_outline shells even when mask area is missing from the element record.
    for key in ("stroke_only", "stroke_outline_shell", "dashed", "dash"):
        if el.get(key) is not None and metadata.get(key) is None:
            metadata[key] = el[key]
    candidate = {
        "id": f"c_{el['id']}",
        "box": dict(el["box"]),
        "z": 0,
        "kind": kind,
        # box-local mask written by element_detect at elements/<id>.png (by convention)
        "mask": {
            "kind": "alpha",
            "src": mask_src or os.path.join("elements", f"{el['id']}.png"),
        },
        "src": el.get("asset_src"),
        "source_crop": {"element_id": el["id"]},
        "meta": metadata,
    }
    if el.get("stroke") is not None:
        candidate["stroke"] = copy.deepcopy(el.get("stroke"))
    if "fill" in el:
        candidate["fill"] = copy.deepcopy(el.get("fill"))
    return candidate


_SHELL_ROLES = {"button", "shape", "chip", "badge", "container", "card"}
_SPECIFIC_CHILD_ROLES = {"icon", "badge", "logo", "verified", "emoji", "arrow"}

# Arrow glyphs that already appear in OCR/price copy — a separate SAM arrow icon is redundant.
_ARROW_GLYPHS = ("→", "⇒", "⟹", "➜", "➝", "➞", "➔", "►", "▶", "▸", "▹", "▻", "⇨", "⇛")
_PRICE_RANGE_RE = re.compile(
    r"[€$£]?\s*\d+(?:[.,]\d+)?\s*(?:→|⇒|►|▶|->|=>|[–—\-])\s*[€$£]?\s*\d+",
)
_GUIDE_ROLES = frozenset({
    "shape", "icon", "divider", "chrome", "card", "container", "frame", "rect", "box",
})
_ANNOTATION_STROKE_ROLES = frozenset({
    "underline", "strikethrough", "strike_through", "annotation", "callout_leader",
    "leader", "leader_line", "connector", "arrow",
    # IM8 STRUGGLE→product string/thread leaders (same preserve path as callout strokes).
    "string", "thread", "string_leader", "leader_string", "callout_string",
})
_LEADER_PROMOTE_ROLES = frozenset({
    "shape", "icon", "divider", "chrome", "line", "rect", "box", "",
    "string", "thread", "string_leader", "leader_string", "callout_string",
})


def _text_has_arrow_glyph(text) -> bool:
    raw = str(text or "")
    if not raw:
        return False
    if any(glyph in raw for glyph in _ARROW_GLYPHS):
        return True
    if "->" in raw or "=>" in raw:
        return True
    return bool(_PRICE_RANGE_RE.search(raw))


def _is_arrow_icon_candidate(candidate: dict) -> bool:
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    kind = str(candidate.get("kind") or "").lower()
    if candidate.get("target") == "drop":
        return False
    return role == "arrow" or kind == "arrow" or (
        candidate.get("target") == "icon" and "arrow" in role
    )


def _drop_redundant_arrow_icons(candidates: list, diagnostics=None) -> list:
    """Drop arrow icons that sit on TEXT already carrying → / price-range arrows.

    Price ads like ``€63 → €49`` often detect a separate SAM arrow over the glyph,
    producing a double arrow (text + vector). Keep the native text; drop the icon.
    """
    texts = [
        c for c in candidates
        if c.get("target") == "text" and _text_has_arrow_glyph(c.get("text"))
    ]
    if not texts:
        return candidates
    for cand in candidates:
        if not _is_arrow_icon_candidate(cand):
            continue
        box = cand.get("box") or {}
        for text in texts:
            tbox = text.get("box") or {}
            iou, inside = _geometry_overlap(box, tbox)
            # Arrow head sits inside / on the price line (bench 002: 57px icon in €63→€49).
            if inside < 0.35 and iou < 0.12 and _inside_frac(box, tbox) < 0.45:
                continue
            meta = cand.setdefault("meta", {})
            cand["target"] = "drop"
            meta["suppression_reason"] = "redundant-arrow-in-text"
            meta["guide_artifact"] = False
            meta["absorbed_into"] = text.get("id")
            if diagnostics is not None:
                diagnostics.append({
                    "dropped": cand.get("id"),
                    "kept": text.get("id"),
                    "reason": "redundant-arrow-in-arrow-bearing-text",
                    "iou": round(iou, 3),
                    "inside_frac": round(inside, 3),
                })
            break
    return candidates


_PRICE_PLACEHOLDER_RE = re.compile(
    r"([€$£]\s*\d+(?:[.,]\d+)?)\s+([A-Za-z|])\s+([€$£]\s*\d+(?:[.,]\d+)?)",
)


def _normalize_price_placeholder_with_verified_arrow(candidates: list) -> int:
    """Remove an OCR placeholder glyph only when a verified arrow owns that slot."""
    arrows = [c for c in candidates if _is_arrow_icon_candidate(c)]
    changed = 0
    for text in candidates:
        if text.get("target") != "text":
            continue
        meta = text.get("meta") or {}
        if str(meta.get("role") or "").lower() != "price":
            continue
        raw = str(text.get("text") or "")
        match = _PRICE_PLACEHOLDER_RE.search(raw)
        if not match:
            continue
        arrow = next((candidate for candidate in arrows if (
            (candidate.get("meta") or {}).get("pairs_with") == text.get("id")
            or meta.get("pairs_with") == candidate.get("id")
            or _inside_frac(candidate.get("box") or {}, text.get("box") or {}) >= 0.35
        )), None)
        if arrow is None:
            continue
        replacement = f"{match.group(1)} {match.group(3)}"
        normalized = raw[:match.start()] + replacement + raw[match.end():]
        right_start = normalized.find(match.group(3), match.start())
        right_end = right_start + len(match.group(3))
        adjusted_runs = []
        for run in text.get("text_runs") or []:
            if not isinstance(run, dict):
                continue
            try:
                start, end = int(run.get("start", 0)), int(run.get("end", 0))
            except (TypeError, ValueError):
                continue
            if end <= match.start(3) or start >= match.end(3):
                continue
            adjusted = copy.deepcopy(run)
            adjusted["start"], adjusted["end"] = right_start, right_end
            adjusted_runs.append(adjusted)
        text["text"] = normalized
        text["text_runs"] = adjusted_runs
        meta["price_separator_recovered"] = {
            "removed_ocr_glyph": match.group(2),
            "arrow_id": arrow.get("id"),
            "source": "verified-overlapping-arrow",
        }
        changed += 1
    return changed


def _decoration_anchor(owner: dict, x0: float, y0: float, x1: float, y1: float) -> Optional[dict]:
    """Bind a text decoration to the word (or whole node) it decorates.

    Returns endpoint FRACTIONS relative to the anchor box, so the design compiler can
    re-project the decoration onto the emitted text node's FINAL geometry instead of
    trusting absolute source coordinates (benchmark 002: the €63 strike stayed at the
    source x while the split word node moved, so the diagonal crossed the wrong glyphs).
    """
    obox = owner.get("box") or {}
    try:
        ox, oy = float(obox.get("x")), float(obox.get("y"))
        ow, oh = float(obox.get("w")), float(obox.get("h"))
    except (TypeError, ValueError):
        return None
    if ow <= 1.0 or oh <= 1.0:
        return None
    anchor_box = (ox, oy, ow, oh)
    word_text = None
    span0, span1 = min(x0, x1), max(x0, x1)
    best_overlap = 0.0
    for word in (owner.get("meta") or {}).get("word_geometry") or []:
        try:
            wx = ox + float(word["fx"]) * ow
            ww = float(word["fw"]) * ow
            wy = oy + float(word["fy"]) * oh
            wh = float(word["fh"]) * oh
        except (KeyError, TypeError, ValueError):
            continue
        overlap = max(0.0, min(span1, wx + ww) - max(span0, wx))
        if overlap > best_overlap:
            best_overlap = overlap
            anchor_box = (wx, wy, ww, wh)
            word_text = str(word.get("text") or "")
    if word_text is not None and best_overlap < 0.5 * max(1.0, span1 - span0):
        # The decoration does not clearly belong to a single word — anchor to the node.
        anchor_box = (ox, oy, ow, oh)
        word_text = None
    ax, ay, aw, ah = anchor_box
    if aw <= 1.0 or ah <= 1.0:
        return None
    return {
        "owner_id": str(owner.get("id") or ""),
        "word_text": word_text,
        "fx0": round((x0 - ax) / aw, 6), "fy0": round((y0 - ay) / ah, 6),
        "fx1": round((x1 - ax) / aw, 6), "fy1": round((y1 - ay) / ah, 6),
    }


def _decoration_shape_candidate(owner: dict, rule: dict, index: int) -> Optional[dict]:
    try:
        x0, y0 = float(rule["x0"]), float(rule["y0"])
        x1, y1 = float(rule["x1"]), float(rule["y1"])
        thickness = max(1.0, float(rule.get("thickness", 2.0)))
    except (KeyError, TypeError, ValueError):
        return None
    pad = max(1.0, thickness)
    bx, by = min(x0, x1) - pad, min(y0, y1) - pad
    bw, bh = max(1.0, abs(x1 - x0) + pad * 2), max(1.0, abs(y1 - y0) + pad * 2)
    lx0, ly0, lx1, ly1 = x0 - bx, y0 - by, x1 - bx, y1 - by
    colour = str(rule.get("color") or "#e1491b")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{bw:.2f}" height="{bh:.2f}" '
        f'viewBox="0 0 {bw:.2f} {bh:.2f}"><path d="M {lx0:.2f} {ly0:.2f} '
        f'L {lx1:.2f} {ly1:.2f}" fill="none" stroke="{colour}" '
        f'stroke-width="{thickness:.2f}" stroke-linecap="round"/></svg>'
    )
    return {
        "id": f"{owner.get('id')}__decoration_{index}",
        "target": "shape",
        "kind": "shape",
        "shape_kind": "path",
        "svg": svg,
        "box": {"x": bx, "y": by, "w": bw, "h": bh},
        "z": 100.0 + index * 0.01,
        # Preview uses the SVG alpha as a mask and the node fill as paint; Figma uses
        # the SVG stroke directly. Keep both representations source-colour accurate.
        "fill": {"kind": "flat", "color": colour},
        "stroke": {"color": colour, "width": thickness, "cap": "ROUND"},
        "meta": {
            "source": "native-price-decoration",
            "role": str(rule.get("kind") or "annotation"),
            "native_decoration": True,
            "decoration_owner_id": owner.get("id"),
            "line": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                     "thickness": thickness},
            "confidence": float(rule.get("confidence", 1.0) or 1.0),
            "z_band": "overlay",
            "external_overlay": True,
            "separate_layer": True,
            "removal_required": True,
            "anchor": _decoration_anchor(owner, x0, y0, x1, y1),
        },
    }


def _materialize_native_price_decorations(candidates: list) -> list:
    additions = []
    accepted_rules: list[tuple[dict, int]] = []

    def same_rule(a: dict, b: dict) -> bool:
        if str(a.get("kind") or "") != str(b.get("kind") or ""):
            return False
        direct = max(abs(float(a.get(k, 0) or 0) - float(b.get(k, 0) or 0))
                     for k in ("x0", "y0", "x1", "y1"))
        reverse = max(
            abs(float(a.get(ak, 0) or 0) - float(b.get(bk, 0) or 0))
            for ak, bk in (("x0", "x1"), ("y0", "y1"),
                           ("x1", "x0"), ("y1", "y0"))
        )
        return min(direct, reverse) <= 2.0

    for owner in candidates:
        if owner.get("target") != "text":
            continue
        for rule in (owner.get("meta") or {}).get("native_decoration_shapes") or []:
            candidate = _decoration_shape_candidate(owner, rule, len(additions))
            if candidate is None:
                continue
            duplicate = next(((prior, index) for prior, index in accepted_rules
                              if same_rule(prior, rule)), None)
            if duplicate is not None:
                prior, index = duplicate
                if float(rule.get("confidence", 0) or 0) > float(prior.get("confidence", 0) or 0):
                    candidate["id"] = additions[index]["id"]
                    additions[index] = candidate
                    accepted_rules.remove(duplicate)
                    accepted_rules.append((rule, index))
                continue
            accepted_rules.append((rule, len(additions)))
            additions.append(candidate)
    return candidates + additions


def _box_fill_fraction(candidate: dict) -> float | None:
    """Mask fill density inside the candidate box (None when area is unknown)."""
    box = candidate.get("box") or {}
    w = float(box.get("w", 0) or 0)
    h = float(box.get("h", 0) or 0)
    if w <= 0 or h <= 0:
        return None
    meta = candidate.get("meta") or {}
    area = meta.get("area")
    if area is None:
        area = candidate.get("area")
    try:
        area_f = float(area)
    except (TypeError, ValueError):
        return None
    if area_f <= 0:
        return None
    return area_f / (w * h)


def _is_short_annotation_underline(box: dict) -> bool:
    """True for short, flat strokes under text — keep these (not layout guides)."""
    w = float((box or {}).get("w", 0) or 0)
    h = float((box or {}).get("h", 0) or 0)
    if w <= 0 or h <= 0:
        return False
    return h <= 14.0 and (w / h) >= 3.5


def _box_center_xy(box: dict) -> tuple[float, float]:
    return (
        float((box or {}).get("x", 0) or 0) + float((box or {}).get("w", 0) or 0) * 0.5,
        float((box or {}).get("y", 0) or 0) + float((box or {}).get("h", 0) or 0) * 0.5,
    )


def _box_area(box: dict) -> float:
    return max(0.0, float((box or {}).get("w", 0) or 0) * float((box or {}).get("h", 0) or 0))


def _is_thin_stroke_geometry(box: dict, fill_frac: float | None = None) -> bool:
    """True for elongated or sparse strokes (leaders/arrows), not filled chrome."""
    w = float((box or {}).get("w", 0) or 0)
    h = float((box or {}).get("h", 0) or 0)
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / max(1.0, min(w, h))
    # Aspect ratio alone is not stroke evidence: a dense headline/product plate can
    # easily be 5:1.  Permit the aspect shortcut only for genuinely thin geometry or
    # when the mask itself is sparse.
    minor = min(w, h)
    if aspect >= 2.2 and (minor <= 32.0 or (fill_frac is not None and fill_frac <= 0.32)):
        return True
    if fill_frac is not None and fill_frac <= 0.32 and max(w, h) >= 18.0:
        return True
    return False


def _product_boxes(candidates: list) -> list[dict]:
    boxes = []
    for cand in candidates or []:
        if cand.get("target") == "drop":
            continue
        role = str((cand.get("meta") or {}).get("role") or cand.get("kind") or "").lower()
        if role in _PRODUCT_CUTOUT_ROLES or role == "product":
            box = cand.get("box") or {}
            if _box_area(box) > 0:
                boxes.append(box)
    return boxes


def _is_leader_role(role: str) -> bool:
    return str(role or "").lower().replace("-", "_") in _ANNOTATION_STROKE_ROLES


def _looks_like_callout_leader(
    candidate: dict,
    text_cands: list,
    product_boxes: list | None = None,
) -> bool:
    """Short curved/line stroke near overlay text pointing at a product — keep it.

    Layout-guide junk hugs a text bbox. Callout leaders sit *beside* text and reach
    toward a product cutout (014-style explainer arrows).
    """
    if candidate.get("target") == "drop":
        return False
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or candidate.get("kind") or "").lower()
    if _is_leader_role(role) or meta.get("annotation_stroke") or meta.get("callout_leader"):
        return True
    # Press logos / wordmarks are never annotation leaders (AS SEEN IN strips).
    if role.replace("-", "_") in {
        "logo", "platform_logo", "wordmark", "brand", "logo_strip", "as_seen_in",
        "press_logos", "product", "photo", "person",
    }:
        return False
    if role and role not in _LEADER_PROMOTE_ROLES and candidate.get("target") not in (
        "shape", "icon",
    ):
        return False
    box = candidate.get("box") or {}
    fill_frac = _box_fill_fraction(candidate)
    if not (
        _is_thin_stroke_geometry(box, fill_frac)
        or _is_short_annotation_underline(box)
        or meta.get("stroke_only")
        or meta.get("dashed")
        or meta.get("dash")
        or (candidate.get("stroke") and not candidate.get("fill"))
    ):
        return False
    cand_area = _box_area(box)
    if cand_area <= 0:
        return False
    cx, cy = _box_center_xy(box)
    near_text = None
    for text in text_cands or []:
        if text.get("target") == "drop":
            continue
        tbox = text.get("box") or {}
        t_area = _box_area(tbox)
        if t_area <= 0:
            continue
        iou = _iou(box, tbox)
        # Guides nearly coincide with text; leaders are smaller and only graze it.
        if iou >= 0.35 and cand_area >= t_area * 0.45:
            continue
        tx, ty = _box_center_xy(tbox)
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        reach = max(float(tbox.get("w", 0) or 0), float(tbox.get("h", 0) or 0)) * 1.8
        grazes = _inside_frac(box, tbox) > 0.02 or _inside_frac(tbox, box) > 0.02
        if dist <= reach or grazes:
            near_text = tbox
            break
    if near_text is None:
        return False
    products = product_boxes if product_boxes is not None else []
    if not products:
        # No product yet — still treat thin near-text strokes as leaders when they
        # clearly do not hug the text frame (014 arrows before product is tagged).
        return cand_area < _box_area(near_text) * 0.55
    px, py = _box_center_xy(products[0])
    best = None
    for pbox in products:
        pcx, pcy = _box_center_xy(pbox)
        d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, pcx, pcy, pbox)
    _, px, py, pbox = best
    tx, ty = _box_center_xy(near_text)
    # Leader center should sit between text and product (or on the product fringe).
    toward_product = ((cx - tx) * (px - tx) + (cy - ty) * (py - ty)) >= 0
    on_product_fringe = _inside_frac(box, pbox) >= 0.08 or _inside_frac(pbox, box) >= 0.02
    return toward_product or on_product_fringe


def _is_guide_artifact(
    candidate: dict,
    text_cands: list,
    product_boxes: list | None = None,
) -> bool:
    """Dashed/stroke-only rect that hugs a text box (layout-guide junk), not an underline."""
    if candidate.get("target") == "drop":
        return False
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or candidate.get("kind") or "").lower().replace("-", "_")
    # Editable chrome hosting OCR (outline pills, SAVE seals, buttons) is never a guide.
    if (
        meta.get("text_bearing_shell")
        or meta.get("plate_shell")
        or meta.get("stroke_outline_shell")
        or meta.get("button_shell")
        or role in _SHELL_GUIDE_EXEMPT_ROLES
    ):
        return False
    if _is_leader_role(role) or meta.get("annotation_stroke") or meta.get("callout_leader"):
        return False
    if meta.get("leader_dot") or role in {"leader_dot", "endpoint_dot", "dot"}:
        return False
    if _looks_like_callout_leader(candidate, text_cands, product_boxes):
        return False
    if _looks_like_leader_endpoint_dot(candidate):
        return False
    source = str(meta.get("source") or "")
    if source not in ("element", "element+qwen") and candidate.get("text") is not None:
        return False
    box = candidate.get("box") or {}
    if _is_short_annotation_underline(box):
        return False
    fill_frac = _box_fill_fraction(candidate)
    stroke_only = bool(
        meta.get("stroke_only")
        or meta.get("dashed")
        or meta.get("dash")
        or (candidate.get("stroke") and not candidate.get("fill"))
        or (fill_frac is not None and fill_frac <= 0.28)
    )
    if not stroke_only:
        return False
    # Prefer known chrome roles; still allow sparse shape/icon with no role.
    if role and role not in _GUIDE_ROLES and candidate.get("target") not in ("shape", "icon"):
        return False
    cand_area = _box_area(box)
    for text in text_cands:
        if text.get("target") == "drop":
            continue
        tbox = text.get("box") or {}
        t_area = _box_area(tbox)
        iou = _iou(box, tbox)
        mutual = min(_inside_frac(box, tbox), _inside_frac(tbox, box))
        # Inset text inside a larger hollow plate is a callout shell, not a guide frame.
        if (
            t_area > 0
            and cand_area > t_area * 1.15
            and _inside_frac(tbox, box) >= 0.55
            and iou < 0.72
        ):
            continue
        # Guide rects nearly match the text bbox (dashed selection / layout frame).
        # A much smaller thin stroke next to text is a leader/underline, not a guide.
        if iou >= 0.35 or mutual >= 0.72:
            if (
                t_area > 0
                and cand_area < t_area * 0.45
                and _is_thin_stroke_geometry(box, fill_frac)
            ):
                continue
            return True
    return False


def _drop_guide_artifacts(candidates: list, diagnostics=None) -> list:
    """Drop dashed/stroke-only guide rects that hug text boxes (keep short underlines)."""
    text_cands = [c for c in candidates if c.get("target") == "text"]
    if not text_cands:
        return candidates
    product_boxes = _product_boxes(candidates)
    for cand in candidates:
        if not _is_guide_artifact(cand, text_cands, product_boxes):
            continue
        meta = cand.setdefault("meta", {})
        cand["target"] = "drop"
        meta["suppression_reason"] = "guide_artifact"
        meta["guide_artifact"] = True
        if diagnostics is not None:
            diagnostics.append({
                "dropped": cand.get("id"),
                "reason": "guide_artifact",
                "fill_frac": (
                    round(_box_fill_fraction(cand), 3)
                    if _box_fill_fraction(cand) is not None else None
                ),
            })
    return candidates


def _should_preserve_callout_leaders(cfg: dict | None) -> bool:
    cfg = cfg or {}
    scene = cfg.get("scene") or {}
    grouping = (scene.get("preset") or {}).get("grouping") or {}
    if grouping.get("preserve_callout_leaders"):
        return True
    facts = scene.get("facts") or {}
    if facts.get("leader_lines"):
        return True
    try:
        from .format_readiness import has_capability
        if has_capability(cfg, "diagrams"):
            return True
    except Exception:
        pass
    return False


def _preserve_callout_leaders(candidates: list, canvas: dict, cfg: dict | None,
                              diagnostics=None) -> list:
    """Keep callout leaders as separate annotation layers; never bake into the product.

    When ``preserve_callout_leaders`` (lifestyle_overlay / diagrams capability) is on, or
    when thin near-text→product strokes are already present, promote/tag them so routing
    treats them as vectors/chips and layout does not nest them under the product photo.
    """
    cfg = cfg or {}
    text_cands = [c for c in candidates if c.get("target") == "text"]
    product_boxes = _product_boxes(candidates)
    force = _should_preserve_callout_leaders(cfg)
    leader_ids = []
    for cand in candidates:
        if cand.get("target") == "drop":
            continue
        meta = cand.setdefault("meta", {})
        # Text-bearing outline pills / seals are chrome plates, not annotation leaders.
        if (
            meta.get("text_bearing_shell")
            or meta.get("plate_shell")
            or meta.get("stroke_outline_shell")
        ):
            continue
        role = str(meta.get("role") or cand.get("kind") or "").lower()
        is_leader = (
            _is_leader_role(role)
            or meta.get("annotation_stroke")
            or meta.get("callout_leader")
            or _looks_like_callout_leader(cand, text_cands, product_boxes)
        )
        if not is_leader:
            continue
        if not force and not _is_leader_role(role) and not meta.get("annotation_stroke"):
            # Geometry-only promotion still runs when leaders are obvious, so 014-like
            # ads work even before archetype facts land.
            if not _looks_like_callout_leader(cand, text_cands, product_boxes):
                continue
        if not _is_leader_role(role):
            meta["role"] = "callout_leader"
            meta["role_promoted_from"] = role or cand.get("kind") or "shape"
        if cand.get("target") in ("shape", "image", None, ""):
            cand["target"] = "icon"
        meta["callout_leader"] = True
        meta["external_overlay"] = True
        meta["separate_layer"] = True
        meta["extract_from_cluster"] = True
        meta.setdefault("z_band", "overlay")
        if meta.get("layer_disposition") in {"plate", "background", "keep_in_background"}:
            meta["layer_disposition"] = "foreground_vector"
        meta.pop("keep_in_background", None)
        meta.pop("baked_owner_id", None)
        # Detach from product parents — leaders annotate the product, they are not part of it.
        parent_id = meta.get("parent_id")
        if parent_id:
            parent = next((c for c in candidates if c.get("id") == parent_id), None)
            parent_role = str(((parent or {}).get("meta") or {}).get("role") or "").lower()
            if parent_role in _PRODUCT_CUTOUT_ROLES or parent_role == "product":
                meta["parent_id"] = None
                meta["detached_from_product"] = True
        leader_ids.append(cand.get("id"))

    if not leader_ids:
        return candidates

    # Pair each leader with the nearest overlay text and tag that copy as a callout.
    by_id = {c.get("id"): c for c in candidates if c.get("id")}
    groups = []
    for lid in leader_ids:
        leader = by_id.get(lid)
        if not leader:
            continue
        lbox = leader.get("box") or {}
        lcx, lcy = _box_center_xy(lbox)
        best_text = None
        best_dist = None
        for text in text_cands:
            if text.get("target") == "drop":
                continue
            t_role = str((text.get("meta") or {}).get("role") or "").lower()
            if t_role in {"cta", "button", "disclaimer", "legal", "footer"}:
                continue
            # Skip the primary top-band display headline; side callouts may still
            # arrive mis-tagged as headline and should pair with leaders.
            tbox = text.get("box") or {}
            if t_role in {"headline", "title"} and float(tbox.get("y", 0) or 0) <= float(
                (canvas or {}).get("h") or 1
            ) * 0.28 and float(tbox.get("w", 0) or 0) >= float(
                (canvas or {}).get("w") or 1
            ) * 0.45:
                continue
            tcx, tcy = _box_center_xy(tbox)
            dist = ((lcx - tcx) ** 2 + (lcy - tcy) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_text = text
        group_id = f"callout-{len(groups)}"
        leader_meta = leader.setdefault("meta", {})
        leader_meta["callout_group_id"] = group_id
        if best_text is not None:
            leader_meta["pairs_with"] = best_text.get("id")
            t_meta = best_text.setdefault("meta", {})
            t_meta["callout_group_id"] = group_id
            t_meta["pairs_with"] = lid
            t_meta["overlay_text"] = True
            t_meta["removal_required"] = True
            t_role = str(t_meta.get("role") or "").lower()
            if t_role in {
                "", "text", "body", "body-copy", "body_copy", "copy", "label",
                "caption", "headline", "subheadline", "title",
            }:
                # Don't demote a true top display headline.
                tbox = best_text.get("box") or {}
                if not (
                    t_role in {"headline", "title"}
                    and float(tbox.get("y", 0) or 0) <= float((canvas or {}).get("h") or 1) * 0.28
                    and float(tbox.get("w", 0) or 0) >= float((canvas or {}).get("w") or 1) * 0.45
                ):
                    t_meta["role"] = "callout"
                    t_meta.setdefault("semantic_role", "callout")
            groups.append({"id": group_id, "leader": lid, "text": best_text.get("id")})
        else:
            groups.append({"id": group_id, "leader": lid, "text": None})

    if diagnostics is not None:
        diagnostics.extend(groups)
    return candidates


_LEADER_DOT_ROLES = frozenset({
    "leader_dot", "endpoint_dot", "dot", "bullet", "decoration", "shape", "icon", "",
})
_AS_SEEN_IN_RE = re.compile(r"\bas\s+seen\s+in\b", re.I)
_SWIPE_UP_CTA_RE = re.compile(
    r"^\s*(swipe\s*up|tap\s*(here|to\s*(shop|buy|learn|get))|"
    r"shop\s*now|get\s*yours?|buy\s*now)\s*!?\s*$",
    re.I,
)


def _looks_like_leader_endpoint_dot(candidate: dict) -> bool:
    """Small near-square filled chip at a leader tip (Wavy beach / 041 endpoint dots)."""
    if candidate.get("target") == "drop":
        return False
    meta = candidate.get("meta") or {}
    if meta.get("leader_dot") or meta.get("callout_leader") or meta.get("text_bearing_shell"):
        return bool(meta.get("leader_dot"))
    role = str(meta.get("role") or candidate.get("kind") or "").lower().replace("-", "_")
    if role not in _LEADER_DOT_ROLES and role not in {"badge"}:
        return False
    box = candidate.get("box") or {}
    w = float(box.get("w") or 0)
    h = float(box.get("h") or 0)
    if w <= 0 or h <= 0:
        return False
    if max(w, h) > 52.0 or min(w, h) < 4.0:
        return False
    if max(w, h) / max(1.0, min(w, h)) > 1.55:
        return False
    # Elongated leaders are strokes, not dots.
    if _is_thin_stroke_geometry(box, _box_fill_fraction(candidate)):
        return False
    fill_frac = _box_fill_fraction(candidate)
    if fill_frac is not None and fill_frac < 0.22:
        return False
    return True


def _preserve_leader_endpoint_dots(candidates: list, diagnostics=None) -> list:
    """Keep circular leader terminators as separate chips; pair them with nearest leader."""
    leaders = [
        c for c in candidates
        if c.get("target") != "drop"
        and (
            (c.get("meta") or {}).get("callout_leader")
            or _is_leader_role(str((c.get("meta") or {}).get("role") or ""))
        )
    ]
    if not leaders:
        return candidates
    pairs = []
    for cand in candidates:
        if cand.get("target") == "drop":
            continue
        meta = cand.setdefault("meta", {})
        if meta.get("callout_leader") or _is_leader_role(str(meta.get("role") or "")):
            continue
        if meta.get("text_bearing_shell") or meta.get("plate_shell"):
            continue
        if not _looks_like_leader_endpoint_dot(cand):
            continue
        dbox = cand.get("box") or {}
        dcx, dcy = _box_center_xy(dbox)
        best, best_dist = None, None
        for leader in leaders:
            lbox = leader.get("box") or {}
            lcx, lcy = _box_center_xy(lbox)
            # Dot sits near either tip of the leader bbox (or grazes it).
            reach = max(float(lbox.get("w") or 0), float(lbox.get("h") or 0)) * 0.85 + 28.0
            dist = ((dcx - lcx) ** 2 + (dcy - lcy) ** 2) ** 0.5
            grazes = _inside_frac(dbox, lbox) > 0.02 or _iou(dbox, lbox) > 0.02
            # Prefer dots near the leader's extremities rather than its center.
            lx0, ly0 = float(lbox.get("x") or 0), float(lbox.get("y") or 0)
            lx1, ly1 = lx0 + float(lbox.get("w") or 0), ly0 + float(lbox.get("h") or 0)
            tip_dist = min(
                ((dcx - lx0) ** 2 + (dcy - ly0) ** 2) ** 0.5,
                ((dcx - lx1) ** 2 + (dcy - ly0) ** 2) ** 0.5,
                ((dcx - lx0) ** 2 + (dcy - ly1) ** 2) ** 0.5,
                ((dcx - lx1) ** 2 + (dcy - ly1) ** 2) ** 0.5,
            )
            score = tip_dist if grazes or tip_dist <= reach else dist
            if tip_dist > reach and not grazes:
                continue
            if best is None or score < best_dist:
                best, best_dist = leader, score
        if best is None:
            continue
        meta["role"] = "leader_dot"
        meta["leader_dot"] = True
        meta.pop("callout_leader", None)
        meta["external_overlay"] = True
        meta["separate_layer"] = True
        group_id = (best.get("meta") or {}).get("callout_group_id")
        if group_id:
            meta["callout_group_id"] = group_id
        meta["pairs_with"] = best.get("id")
        best.setdefault("meta", {})["endpoint_dot_id"] = cand.get("id")
        if cand.get("target") in ("shape", "image", None, ""):
            cand["target"] = "icon"
        pairs.append({"dot": cand.get("id"), "leader": best.get("id")})
    if diagnostics is not None and pairs:
        diagnostics.extend(pairs)
    return candidates


def _tag_as_seen_in_logo_strip(candidates: list, canvas: dict | None = None,
                              diagnostics=None) -> list:
    """Mark AS SEEN IN press logos as an intentional raster strip (honest chips)."""
    texts = [
        c for c in candidates
        if c.get("target") == "text"
        and _AS_SEEN_IN_RE.search(str(c.get("text") or ""))
    ]
    if not texts:
        return candidates
    canvas = canvas or {}
    cw = float(canvas.get("w") or 0)
    logos = []
    for cand in candidates:
        if cand.get("target") not in {"image", "icon", "shape"}:
            continue
        meta = cand.get("meta") or {}
        role = str(meta.get("role") or cand.get("kind") or "").lower().replace("_", "-")
        if role not in {
            "logo", "platform-logo", "icon", "badge", "wordmark", "brand",
            "logo-strip", "as-seen-in", "press-logos", "shape", "",
        }:
            continue
        box = cand.get("box") or {}
        w = float(box.get("w") or 0)
        h = float(box.get("h") or 0)
        if w <= 0 or h <= 0:
            continue
        # Press marks are compact; skip large product cutouts.
        if cw > 0 and w * h > cw * float(canvas.get("h") or cw) * 0.08:
            continue
        if max(w, h) > max(220.0, (cw or 1080) * 0.28):
            continue
        logos.append(cand)
    if len(logos) < 2:
        return candidates
    tagged = []
    for text in texts:
        tbox = text.get("box") or {}
        ty1 = float(tbox.get("y") or 0) + float(tbox.get("h") or 0)
        band = []
        for logo in logos:
            lbox = logo.get("box") or {}
            ly = float(lbox.get("y") or 0)
            # Logos sit on the same band or just below the AS SEEN IN label.
            if ly + float(lbox.get("h") or 0) < float(tbox.get("y") or 0) - 12:
                continue
            if ly > ty1 + max(160.0, float(tbox.get("h") or 0) * 6.0):
                continue
            band.append(logo)
        if len(band) < 2:
            continue
        group_id = f"logo-strip-{text.get('id') or 'as-seen'}"
        text.setdefault("meta", {})["logo_strip_group_id"] = group_id
        text.setdefault("meta", {})["role"] = text.get("meta", {}).get("role") or "eyebrow"
        for logo in band:
            meta = logo.setdefault("meta", {})
            meta["role"] = "logo-strip"
            meta["intentional_raster_cluster"] = True
            meta["logo_strip_group_id"] = group_id
            meta["external_overlay"] = True
            tagged.append(logo.get("id"))
        if diagnostics is not None:
            diagnostics.append({
                "group_id": group_id,
                "label": text.get("id"),
                "logos": [l.get("id") for l in band],
            })
    return candidates


def _promote_story_cta(candidates: list, canvas: dict | None = None) -> list:
    """Tag bottom-band swipe-up / Get Yours copy as story CTA (editable TEXT)."""
    canvas = canvas or {}
    h = float(canvas.get("h") or 0) or 1.0
    w = float(canvas.get("w") or 0) or 1.0
    storyish = (h / w) >= 1.55
    for cand in candidates:
        if cand.get("target") != "text":
            continue
        if cand.get("kept_in_photo"):
            continue
        text = str(cand.get("text") or "").strip()
        if not text:
            continue
        box = cand.get("box") or {}
        y_ratio = float(box.get("y") or 0) / h
        meta = cand.setdefault("meta", {})
        role = str(meta.get("role") or "").lower()
        is_swipe = bool(_SWIPE_UP_CTA_RE.match(text))
        if not is_swipe and not (
            storyish and y_ratio >= 0.78 and role in {"cta", "button", "offer", ""}
            and len(text.split()) <= 5
        ):
            continue
        if y_ratio < 0.70 and not is_swipe:
            continue
        meta["role"] = "cta"
        meta["story_cta"] = True
        meta["overlay_text"] = True
        meta["removal_required"] = True
        meta.setdefault("semantic_role", "cta")
    return candidates


def _collapse_near_duplicate_nests(candidates: list, iou_thresh: float = 0.85) -> list:
    """Drop a parent shell that occupies nearly the same box as its nested child.

    Dense UI scenes often detect engagement chrome twice — once as a button/shape
    shell and once as the icon itself (benchmark-final4/009 comment bubble). When
    the boxes nearly coincide, shipping both produces duplicate raster ownership.
    Prefer the more specific child and clear its parent link.
    """
    by_id = {c.get("id"): c for c in candidates if c.get("id")}
    drop_ids = set()
    for child in candidates:
        meta = child.get("meta") or {}
        parent_id = meta.get("parent_id")
        if not parent_id or parent_id not in by_id or parent_id == child.get("id"):
            continue
        parent = by_id[parent_id]
        if parent.get("id") in drop_ids:
            continue
        parent_role = str((parent.get("meta") or {}).get("role") or parent.get("kind") or "").lower()
        child_role = str(meta.get("role") or child.get("kind") or "").lower()
        if parent_role not in _SHELL_ROLES:
            continue
        if child_role not in _SPECIFIC_CHILD_ROLES and child.get("kind") not in {"icon"}:
            continue
        if _iou(child.get("box") or {}, parent.get("box") or {}) < iou_thresh:
            continue
        parent_meta = parent.setdefault("meta", {})
        parent["target"] = "drop"
        parent_meta["suppression_reason"] = "near-duplicate-nested-shell"
        parent_meta["absorbed_into"] = child.get("id")
        drop_ids.add(parent.get("id"))
        child_meta = child.setdefault("meta", {})
        child_meta["absorbed_shell_id"] = parent.get("id")
        child_meta["parent_id"] = None
        # Point any other siblings at the surviving child so later layout nesting
        # still has a coherent owner when a shell was only a duplicate cutout.
        for other in candidates:
            if other is child or other is parent:
                continue
            other_meta = other.get("meta") or {}
            if other_meta.get("parent_id") == parent.get("id"):
                other_meta["parent_id"] = child.get("id")
    return candidates


# ── public API ───────────────────────────────────────────────────────────────────────
def merge(ocr, elements, qwen, canvas, cfg: Optional[dict] = None, run_dir=None):
    cfg = cfg or {}
    # Ownership aggregation is cheap and deterministic. Recompute it at the merge
    # boundary so a routing-policy refinement can resume from merge without paying for
    # every VLM call again.
    scene_text_propagation_failed = False
    if isinstance(ocr, dict) and ocr.get("lines") and ocr.get("blocks"):
        # Propagate / text-source assembly may touch line+block meta. Clone those lists
        # deeply but keep the OCR shell (styles, VLM blobs, etc.) shared — identical
        # merge outputs, far less deepcopy work on large ocr.json payloads.
        ocr = dict(ocr)
        ocr["lines"] = copy.deepcopy(ocr["lines"])
        ocr["blocks"] = copy.deepcopy(ocr["blocks"])
        try:
            from src.vlm_scene_text import _propagate_to_blocks
            _propagate_to_blocks(ocr["lines"], ocr["blocks"])
        except Exception as exc:  # never let scene-text ownership hints sink merge()
            scene_text_propagation_failed = True
            print(f"[merge] scene_text_propagation_failed: {exc}; continuing without VLM ownership hints")
    route, real = _load_routing()
    if run_dir is None:
        run_dir = cfg.get("run_dir")
    dedup_iou = float((cfg.get("merge") or {}).get("dedup_iou", 0.6))
    match_iou = float((cfg.get("merge") or {}).get("qwen_match_iou", 0.3))
    photo_inside = float((cfg.get("merge") or {}).get("photo_inside_frac", 0.82))
    # Geometric threshold for "printed on a discrete product cutout" scene text. Lower and
    # separate from ``photo_inside`` (which gates the stricter raster-cluster ownership):
    # a product name/label frequently sits only ~55-70% inside the segmented package box
    # because ascenders/kerning spill past the mask, yet it is unambiguously baked-in.
    scene_text_inside = float((cfg.get("merge") or {}).get("scene_text_inside_frac", 0.55))
    scene_roles = set((cfg.get("merge") or {}).get(
        "scene_text_roles",
        ["product", "package", "bottle", "jar", "tube", "sachet", "shaker",
         "pouch", "device", "sign"],
    ))
    overlay_text_roles = {"headline", "title", "subtitle", "subheadline", "eyebrow",
                          "cta", "button", "price", "offer", "callout", "disclaimer",
                          "legal"}

    # Merge diagnostics: counts + reasons for every deduped/suppressed/enforced item, so a
    # ghost/duplicate regression is auditable from the run dir (see merge_report.json).
    diagnostics = {
        "kind": "merge-diagnostics",
        "fragment_suppressed": [],
        "text_dedup": [],
        "element_dedup": [],
        "arrow_dedup": [],
        "guide_artifacts": [],
        "callout_groups": [],
        "leader_dots": [],
        "logo_strips": [],
        "scene_text_contract": [],
    }

    # text_analysis emits paragraph/headline blocks. Prefer those over one Figma node per
    # OCR line so wrapping, hierarchy, and repeated text styles survive downstream.
    ocr_lines = _text_sources(ocr, diagnostics["fragment_suppressed"])
    elements = elements or []
    qwen = qwen or []

    text_cands = [_text_candidate(l) for l in ocr_lines]
    for source, candidate in zip(ocr_lines, text_cands):
        meta = candidate["meta"]
        # OCR commonly reads a single typographic em dash as two separated ASCII
        # hyphens.  Normalise that unambiguous separator before font fitting so a
        # CTA remains both readable and editable instead of inheriting a visible
        # double-hyphen artefact.
        text_value = str(candidate.get("text") or "")
        if "- -" in text_value:
            candidate["text"] = text_value.replace("- -", "—")
            meta["ocr_normalized"] = "double-hyphen-to-emdash"
        if str(meta.get("role") or "").lower() in {"", "text", "body"}:
            meta["role"] = semantic_text_role(source, canvas)
        meta.setdefault("semantic_role", meta["role"])
    _annotate_native_text_repetitions(text_cands)
    elem_cands = [_element_candidate(e) for e in elements]

    # ── qwen z-order + alpha: match each qwen layer to overlapping candidates ──────────
    # qwen list is back-to-front; index = z (lower index = further back).
    scene_regions = []  # product/package regions where printed text should stay baked in
    for zi, ql in enumerate(qwen):
        qbox = ql.get("box", {})
        if not (qbox.get("w", 0) > 0 and qbox.get("h", 0) > 0):
            continue
        # best-matching element candidate gets this layer's clean alpha + z
        best = None
        best_iou = match_iou
        for c in elem_cands:
            iou = _iou(c["box"], qbox)
            if iou > best_iou:
                best_iou, best = iou, c
        if best is not None:
            best["z"] = zi
            best["src"] = ql.get("png")  # clean RGBA alpha from qwen
            best["meta"]["qwen_id"] = ql.get("id")
            best["meta"]["source"] = "element+qwen"
            if best["meta"].get("role") in scene_roles:
                scene_regions.append(qbox)
        else:
            # qwen-only layer with no element match -> raster image candidate.
            # meta.role='photo' makes routing.route pick target='image' + alpha mask.
            elem_cands.append(
                {
                    "id": f"c_{ql['id']}",
                    "box": dict(qbox),
                    "z": zi,
                    "src": ql.get("png"),
                    "meta": {
                        "source": "qwen",
                        "role": "photo",
                        "confidence": 0.5,
                        "qwen_id": ql.get("id"),
                    },
                }
            )
    # Only product/package-like rasters own printed scene text. Generic photos/backgrounds
    # often have intentional editable overlay copy, so containment alone is not enough.
    for c in elem_cands:
        if c["meta"].get("role") in scene_roles or c["meta"].get("contains_scene_text"):
            scene_regions.append(c["box"])
    raster_cluster_owners = [
        c for c in elem_cands
        if is_intentional_raster_cluster(c.get("meta", {}).get("role"))
    ]
    # Do not rebuild buttons, icons and other internal chrome detected *inside* a
    # screenshot/UI crop. The screenshot is the exact, swappable visual owner. Only a
    # separately corroborated external overlay may escape; a surrounding source-evidenced
    # card/shell remains eligible because it contains (rather than sits inside) the crop.
    for child in elem_cands:
        if child in raster_cluster_owners:
            child["meta"].setdefault("decomposition_policy", {
                "outer_shell": "source-evidenced-only",
                "internal_chrome": "baked-in-raster-owner",
            })
            continue
        owner = _raster_cluster_owner(child.get("box") or {}, raster_cluster_owners, photo_inside)
        if owner is None:
            continue
        meta = child.setdefault("meta", {})
        ownership = meta.get("ownership_decision") or {}
        positive_external = bool(
            meta.get("external_overlay") or meta.get("extract_from_cluster")
            or meta.get("separate_layer") or ownership.get("action") == "recreate"
            # Decomposed chart/diagram marks tagged with chart_group_id must stay
            # selectable layers, not baked internal chrome of a whole-plot crop.
            or (meta.get("chart_group_id") and is_chart_primitive_role(meta.get("role")))
        )
        if positive_external:
            meta["parent_id"] = owner["id"]
            meta["raster_cluster_owner"] = owner["id"]
            if meta.get("chart_group_id") and is_chart_primitive_role(meta.get("role")):
                meta["extract_from_cluster"] = True
                meta["diagram_member"] = True
            continue
        meta.update({
            "layer_disposition": "plate",
            "keep_in_background": True,
            "baked_owner_id": owner["id"],
            "suppression_reason": "internal-chrome-contained-in-raster-cluster",
            "raster_cluster_owner": owner["id"],
        })

    # Discrete product/photo cutouts (bounded objects, never the full-bleed plate/hero)
    # own any text printed on them. Built from the FINAL elem_cands so qwen-corrected
    # roles are reflected. A full-bleed raster is excluded so overlay copy on a background
    # photo/plate stays editable — unless photographic-scene mode (below) says otherwise.
    product_regions = []
    for c in elem_cands:
        box = c.get("box") or {}
        if not (float(box.get("w", 0) or 0) > 0 and float(box.get("h", 0) or 0) > 0):
            continue
        if _is_full_bleed(box, canvas):
            continue
        role = str(c["meta"].get("role") or "").lower()
        # An oversized, low-confidence photo/residual mask is a loose plate+product
        # merge, NOT a bounded product cutout. Letting it own scene text bakes real
        # white-card checklist / caption / comparison copy into a raster, and the plate
        # inpaint then erases it (066: a conf-0.405 photo-fragment spanning 75% of the
        # canvas swallowed every comparison bullet). Product FACES are bounded and
        # confident; card interiors are not. Require one or the other before a
        # photo/photo-fragment mask may own printed text.
        _conf = float(c["meta"].get("confidence") or c.get("score") or 0.0)
        if (role in {"photo", "photo-fragment", "photo_fragment"}
                and _box_area_frac(box, canvas) >= 0.5
                and _conf < 0.5):
            c["meta"]["oversized_loose_residual"] = True
            continue
        if (role in _PRODUCT_CUTOUT_ROLES or role in _TEXT_BEARING_SHELL_ROLES
                or c["meta"].get("contains_scene_text")):
            product_regions.append({
                "id": c.get("id"),
                "box": dict(box),
                "role": role,
            })

    scene_cfg = cfg.get("scene") or {}
    scene_facts = (scene_cfg.get("facts") or {})
    scene_archetype = str(scene_cfg.get("archetype") or "")
    scene_text_preset = ((scene_cfg.get("preset") or {}).get("text") or {})
    photo_scene_only = _photographic_scene_text_mode(
        text_cands, elem_cands, canvas, cfg, product_regions,
    )
    if photo_scene_only:
        diagnostics.setdefault("photographic_scene_text", True)

    # ── assemble + route ──────────────────────────────────────────────────────────────
    candidates = text_cands + elem_cands

    # scene text: OCR line inside a photo region -> keep baked in the base.
    # VLM scene_text_role overrides geometry when confident; geometry remains fallback.
    for c in text_cands:
        ownership = c["meta"].get("ownership_decision") or {}
        ownership_action = ownership.get("action")
        # Platform lockups are individual artwork layers even when they sit on a
        # social-card raster.  The VLM correctly says "raster_keep" (do not
        # recreate glyphs), but that must mean a cropped/logo asset, not baking
        # the X.com lockup into the whole screenshot.
        if is_platform_lockup(c, canvas):
            c["meta"].update({
                "wordmark": True,
                "platform_lockup": True,
                "role": "platform-logo",
                "semantic_role": "platform-logo",
                "ownership_enforced": True,
                "preserve_underlay": True,
            })
            continue
        # Pure photograph of text (sticky notes / handwriting / laptop UI in-frame):
        # every OCR line lives on a photographic surface with no overlay structure.
        # Bake unless positive external-overlay evidence already marked this line.
        if photo_scene_only:
            meta = c["meta"]
            positive_overlay = bool(
                ownership_action == "recreate"
                or meta.get("scene_text_role") == "overlay_copy"
                or meta.get("overlay_text") or meta.get("promote_text")
                or meta.get("editable_text") or meta.get("text_promoted")
                or meta.get("external_overlay")
            )
            if not positive_overlay:
                c["kept_in_photo"] = True
                meta["origin"] = "scene"
                meta["role"] = "scene-text"
                meta["suppression_reason"] = "text-on-photographic-surface-only"
                meta["ownership_enforced"] = True
                continue
        # A recognised UI/receipt/chart/table/diagram/product cluster owns its internal
        # pixels as one exact source crop. A generic text role is not enough to extract
        # it: only positive external-overlay evidence can make a contained text layer
        # editable. The parent link keeps such an overlay selectable with the raster.
        owner = _raster_cluster_owner(c["box"], raster_cluster_owners, photo_inside)
        if owner is not None:
            owner_id = owner["id"]
            meta = c["meta"]
            positive_overlay = bool(
                ownership_action == "recreate"
                or meta.get("scene_text_role") == "overlay_copy"
                or meta.get("overlay_text") or meta.get("promote_text")
                or meta.get("editable_text") or meta.get("text_promoted")
                # Semi-editable diagram labels: keep native TEXT when tagged as a
                # chart_group member (axis/data labels), not baked scene text.
                or (meta.get("chart_group_id") and is_chart_label_role(meta.get("role")))
            )
            meta["raster_cluster_owner"] = owner_id
            # SCREENSHOT-FAMILY FAIL-OPEN (009): on a social_screenshot the archetype
            # preset promises platform UI copy editable (text.editable_ui_copy) — post
            # body, header, handle, timestamps, engagement counts. Missing VLM ownership
            # on such a line must fail OPEN → native TEXT parked over the screenshot
            # crop, never bake. Only positive photographic-scene evidence (a real VLM
            # "printed" verdict or printed_on_product) keeps the ink in the raster.
            # Non-UI cluster owners (chart/receipt/nutrition/product-cluster…) keep the
            # conservative fail-closed contract.
            ui_copy_fail_open = (
                not positive_overlay
                and scene_archetype == "social_screenshot"
                and bool(scene_text_preset.get("editable_ui_copy"))
                and cluster_normalized_role(
                    (owner.get("meta") or {}).get("role")) in ("screenshot", "ui-panel")
                # A one-glyph OCR blip must not authorize destructive inpaint on the
                # screenshot crop; real UI copy (counts, handles, body) is >= 2 glyphs.
                and len(re.sub(r"[^0-9A-Z€£$%]", "",
                               _normalize_text_key(c.get("text")))) >= 2
                and not _positive_scene_ink_evidence(meta, ownership)
            )
            if ui_copy_fail_open:
                meta["ui_copy_fail_open"] = "social-screenshot-ui-copy"
            if positive_overlay or ui_copy_fail_open:
                meta["overlay_text"] = True
                meta["removal_required"] = True
                meta["parent_id"] = owner_id
                meta["external_overlay"] = True
                meta["ownership_enforced"] = True
                if meta.get("chart_group_id") and is_chart_label_role(meta.get("role")):
                    meta["diagram_label"] = True
                continue
            c["kept_in_photo"] = True
            meta["origin"] = "scene"
            meta["role"] = "scene-text"
            meta["baked_owner_id"] = owner_id
            meta["suppression_reason"] = "text-contained-in-intentional-raster-cluster"
            meta["ownership_enforced"] = True
            continue
        # A failed/disagreeing VLM response deliberately becomes raster_keep so it
        # can never cause destructive inpainting by itself.  But that uncertainty
        # must not flatten an otherwise clear high-confidence CTA/headline: OCR
        # hierarchy already proves it is intentional overlay copy.  Promote only
        # explicit overlay roles, never generic body/product text.
        ownership_failed_closed = (
            ownership_action == "raster_keep"
            and str(ownership.get("reason") or "") in {
                "vlm_disagreement", "vlm_error", "vlm_parse_error",
            }
            and float(ownership.get("confidence") or 0) <= 0.01
            and str(c["meta"].get("role") or "").lower() in overlay_text_roles
        )
        if ownership_failed_closed:
            c["meta"].update({
                "overlay_text": True,
                "removal_required": True,
                "ownership_enforced": True,
                "ownership_recovery": "explicit-overlay-role-after-vlm-failure",
            })
            continue
        if ownership_action != "recreate" and ownership:
            c["kept_in_photo"] = True
            c["meta"]["origin"] = "scene"
            c["meta"]["role"] = "scene-text"
            c["meta"]["ownership_enforced"] = True
            continue
        if ownership_action == "recreate":
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
            c["meta"]["ownership_enforced"] = True
            continue
        scene_text_role = c["meta"].get("scene_text_role")
        if scene_text_role == "printed_on_product":
            # A VLM crop classification is advisory. It can easily see a product near
            # overlay copy and label every word "printed". Only bake/drop text when a
            # separately detected product/package region geometrically corroborates it.
            corroborated = any(_inside_frac(c["box"], region) >= photo_inside
                               for region in scene_regions)
            if corroborated:
                c["kept_in_photo"] = True
                c["meta"]["origin"] = "scene"
                c["meta"]["role"] = "scene-text"
                c["meta"]["scene_text_corroborated"] = True
                continue
            c["meta"]["scene_text_uncorroborated"] = True
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
        if scene_text_role == "wordmark":
            # Decorative packaging wordmark (Wavy script on the tube) stays baked in
            # the product cutout. Flat-plate display brand is handled as overlay TEXT
            # later / in routing — only bake when geometry corroborates a cutout owner.
            cutout_owner = _scene_cutout_owner(c["box"], product_regions, scene_text_inside)
            if cutout_owner is not None:
                owner_elem = next(
                    (e for e in elem_cands if e.get("id") == cutout_owner["id"]),
                    None,
                )
                owner_role = str(
                    ((owner_elem or {}).get("meta") or {}).get("role")
                    or cutout_owner.get("role")
                    or ""
                ).lower()
                if owner_role in _BAKE_CUTOUT_ROLES:
                    c["kept_in_photo"] = True
                    c["meta"]["origin"] = "scene"
                    c["meta"]["role"] = "scene-text"
                    c["meta"]["baked_owner_id"] = cutout_owner["id"]
                    c["meta"]["suppression_reason"] = "wordmark-inside-product-cutout"
                    c["meta"]["ownership_enforced"] = True
                    continue
            c["meta"]["wordmark"] = True
            c["meta"]["role"] = "logo"
            continue
        if scene_text_role == "overlay_copy":
            continue
        # Geometric scene-text on a discrete product/photo cutout. This runs BEFORE the
        # role-based overlay promotion below because a large product name reads as a
        # "subheadline" and a package's ingredient list reads as "offer"/"body": the
        # semantic role cannot distinguish printed-on-product text from real overlay copy,
        # but geometry can. If the line sits substantially inside a bounded product/photo
        # cutout (not the plate/hero, which is excluded from product_regions), it is baked
        # into that raster — no native layer, no pixel removal, single owner. Positive
        # external-overlay/recreate evidence (a VLM verdict or an explicit promotion flag)
        # still wins; a merely *tentative* overlay flag from an uncorroborated VLM guess
        # loses to corroborating geometry and is cleared to keep the scene-text contract.
        positive_overlay_evidence = bool(
            c["meta"].get("external_overlay") or c["meta"].get("promote_text")
            or c["meta"].get("editable_text") or c["meta"].get("text_promoted")
            or scene_text_role == "overlay_copy" or ownership_action == "recreate"
        )
        if not positive_overlay_evidence:
            cutout_owner = _scene_cutout_owner(c["box"], product_regions, scene_text_inside)
            if cutout_owner is not None:
                owner_elem = next(
                    (e for e in elem_cands if e.get("id") == cutout_owner["id"]),
                    None,
                )
                owner_role = str(
                    ((owner_elem or {}).get("meta") or {}).get("role") or ""
                ).lower()
                # Text-bearing badge/button/logo chrome: recreate as editable TEXT and
                # rebuild the plate underneath. Never bake readable OCR into the shell
                # raster for fidelity (Codia contract / benchmark 016 green seal).
                if owner_role in _TEXT_BEARING_SHELL_ROLES:
                    c["meta"]["overlay_text"] = True
                    c["meta"]["removal_required"] = True
                    c["meta"]["parent_id"] = cutout_owner["id"]
                    c["meta"]["shell_text_host"] = cutout_owner["id"]
                    c["meta"]["ownership_enforced"] = True
                    if owner_elem is not None:
                        om = owner_elem.setdefault("meta", {})
                        om["text_bearing_shell"] = True
                        om["plate_shell"] = True
                        if owner_role == "logo":
                            om["reclassified_from"] = "logo"
                            om["role"] = "badge"
                        diagnostics["scene_text_contract"].append({
                            "id": c.get("id"),
                            "host": cutout_owner["id"],
                            "action": "text-bearing-shell-extract",
                            "host_role": owner_role,
                        })
                    continue
                # BEFORE/AFTER PANEL FAIL-OPEN (025): a literal before/after pair
                # authorizes rebuilding all contained column copy — the archetype
                # preset already sets suppress_descendants=False for exactly this
                # scene. White-card checklist lines and the BEFORE/AFTER labels
                # composited over the two photographic panels are designer overlay,
                # not scene ink, so missing VLM ownership fails OPEN → editable
                # TEXT + removal. A real photographic "printed" verdict still bakes
                # (a caption photographed inside the panel). Product/pack cutout
                # owners are untouched: printed-on-product keeps its bake.
                # Guard 1: standalone BEFORE/AFTER column labels are governed by
                # reconstruct._suppress_comparison_column_labels — when contained in a
                # column photo it re-bakes them regardless of promotion flags. Promoting
                # them here with removal_required would erase their ink from the panel
                # AND drop the native node (mangled label, no replacement). Leave them
                # to the established bake path. Guard 2: a one/two-glyph OCR blip must
                # never authorize destructive inpaint on a photographic panel.
                _line_tokens = _normalize_text_key(c.get("text"))
                _is_column_label = bool(re.match(r"^\s*(BEFORE|AFTER)\s*$", _line_tokens))
                if (owner_role in _PHOTOGRAPHIC_PANEL_ROLES
                        and not _is_column_label
                        and len(re.sub(r"[^0-9A-Z€£$%]", "", _line_tokens)) >= 3
                        and (scene_facts.get("before_after_pair")
                             or scene_facts.get("before_after_labels"))
                        and not _positive_scene_ink_evidence(c["meta"], ownership)):
                    c["meta"]["overlay_text"] = True
                    c["meta"]["removal_required"] = True
                    c["meta"]["ownership_enforced"] = True
                    c["meta"]["ui_copy_fail_open"] = "before-after-panel-copy"
                    diagnostics["scene_text_contract"].append({
                        "id": c.get("id"),
                        "host": cutout_owner["id"],
                        "action": "before-after-panel-copy-promote",
                        "host_role": owner_role,
                    })
                    continue
                if not _should_bake_in_cutout(
                    c, cutout_owner, canvas, scene_text_inside, scene_facts,
                ):
                    # Oversized cutout + overlayish role on a flat plate: leave as
                    # candidate overlay copy (007 left column under a loose SAM box).
                    pass
                else:
                    c["kept_in_photo"] = True
                    c["meta"]["origin"] = "scene"
                    c["meta"]["role"] = "scene-text"
                    c["meta"]["baked_owner_id"] = cutout_owner["id"]
                    c["meta"]["scene_text_geometric"] = True
                    c["meta"]["suppression_reason"] = "text-inside-product-cutout"
                    for tentative in ("overlay_text", "removal_required"):
                        c["meta"].pop(tentative, None)
                    continue
        if c["meta"].get("role") in overlay_text_roles:
            # Overlay copy must be painted back as editable text, so its original
            # glyphs have to be removed from the Big-LaMa plate first.  Keep this
            # explicit because a downstream router may conservatively return drop.
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
            continue
        for pr in scene_regions:
            if _inside_frac(c["box"], pr) >= photo_inside:
                c["kept_in_photo"] = True
                c["meta"]["origin"] = "scene"
                c["meta"]["role"] = "scene-text"
                break

    # routing.route returns a NEW dict (shallow copy) — capture it.
    routed = []
    for c in candidates:
        try:
            rc = route(c, canvas, cfg)
        except Exception as e:  # a bad router must not sink the whole stage
            print(f"[merge] routing error on {c.get('id')}: {e}; defaulting")
            rc = _fallback_route(c, canvas, cfg)
        # enforce scene-text -> drop regardless of router (contract hard rule)
        if rc.get("kept_in_photo") or rc.get("meta", {}).get("kept_in_photo"):
            rc["target"] = "drop"
        routed.append(rc)
    candidates = prefer_decomposed_charts(routed)
    # Geometry-only: irregular colored plates (brushstroke banner, starburst seal)
    # hosting OCR → TEXT + plate_shell. No VLM roundtrip.
    _promote_geometric_text_shells(candidates, canvas, cfg, diagnostics)
    # Collapse button/shape shells that are near-identical to a nested icon so dense
    # UI chrome does not ship duplicate owners (run 009 engagement icons).
    candidates = _collapse_near_duplicate_nests(candidates)
    # Price lines with → already encode the arrow — drop overlapping SAM arrow icons.
    candidates = _drop_redundant_arrow_icons(candidates, diagnostics["arrow_dedup"])
    # Dashed / stroke-only layout-guide rects that hug text boxes (not short underlines
    # or 014-style callout leaders pointing at a product).
    candidates = _drop_guide_artifacts(candidates, diagnostics["guide_artifacts"])
    # AS SEEN IN press logos → intentional raster strip BEFORE leader promotion
    # (wide logo chips must not be mistaken for callout strokes).
    candidates = _tag_as_seen_in_logo_strip(
        candidates, canvas, diagnostics.setdefault("logo_strips", []),
    )
    # Preserve callout leaders as separate annotation layers (lifestyle_overlay / diagrams).
    candidates = _preserve_callout_leaders(
        candidates, canvas, cfg, diagnostics["callout_groups"],
    )
    _normalize_price_placeholder_with_verified_arrow(candidates)
    # Circular endpoint dots on leaders (Wavy beach) — never guide-drop; pair with leader.
    candidates = _preserve_leader_endpoint_dots(
        candidates, diagnostics.setdefault("leader_dots", []),
    )
    # Story swipe-up / Get Yours bottom CTA stays editable TEXT.
    candidates = _promote_story_cta(candidates, canvas)
    text_cands = [c for c in candidates if c["meta"].get("source") == "ocr"]

    # ── dedup: shape/icon that is really an OCR text box -> drop (prefer text) ─────────
    kept = []
    for c in candidates:
        if c["meta"].get("source") in ("element", "element+qwen") and c.get("target") in (
            "shape",
            "icon",
        ):
            role = c["meta"].get("role")
            if (role in ("button", "badge", "chip", "banner", "starburst", "seal", "callout", "pill",
                         "leader_dot", "sale_burst", "price_burst")
                    or c["meta"].get("text_bearing_shell") or c["meta"].get("plate_shell")
                    or c["meta"].get("stroke_outline_shell")
                    or c["meta"].get("leader_dot") or c["meta"].get("callout_leader")):
                kept.append(c)
                continue
            # A shape/icon is really just an OCR text box when the text sits almost
            # entirely inside it AND the two boxes are near-coincident — measured by IoU
            # OR mutual containment (the element is also mostly inside the text box), so a
            # slightly different aspect ratio no longer keeps a redundant plate fragment.
            cover_text = next(
                (t for t in text_cands
                 if t.get("target") != "drop"
                 and _inside_frac(t["box"], c["box"]) >= 0.9
                 and (_iou(t["box"], c["box"]) >= dedup_iou
                      or _inside_frac(c["box"], t["box"]) >= 0.85)),
                None,
            )
            if cover_text is not None:
                diagnostics["element_dedup"].append({
                    "dropped": c.get("id"),
                    "covered_by": cover_text.get("id"),
                    "reason": "element-box-is-ocr-text-box",
                    "iou": round(_iou(cover_text["box"], c["box"]), 3),
                })
                continue  # the "shape" is just the text's bounding box
            # Button shells are larger than the CTA label — keep the painted backdrop.
            if role in ("shape", "card", "container", None, "") and any(
                t.get("target") != "drop"
                and (t.get("meta") or {}).get("role") in ("cta", "button", "offer", "price")
                and 0.55 <= _inside_frac(t["box"], c["box"]) < 0.98
                for t in text_cands
            ):
                c.setdefault("meta", {})["role"] = "button"
                kept.append(c)
                continue
        kept.append(c)

    # text-vs-text dedup: collapse the same physical text arriving from different sources
    # (OCR ensemble split disagreement, block-vs-orphan line) before the explicit
    # harness/VLM-critic list is applied.
    merge_cfg = cfg.get("merge") or {}
    kept = _dedup_overlapping_text(kept, merge_cfg, dedup_iou, diagnostics["text_dedup"])
    kept = _materialize_native_price_decorations(kept)
    ids_before_cfg = {c.get("id") for c in kept}
    kept = _dedup_text_candidates(kept, merge_cfg, dedup_iou)
    already_recorded = {entry["dropped"] for entry in diagnostics["text_dedup"]}
    for rid in sorted(x for x in (ids_before_cfg - {c.get("id") for c in kept})
                      if x and x not in already_recorded):
        diagnostics["text_dedup"].append({
            "dropped": rid, "reason": "explicit-duplicate-text-config",
        })

    # Contract: scene text that stays baked in the photo is never ALSO an editable overlay
    # (reconstruct would erase its pixels without re-emitting the layer).
    for c in kept:
        _enforce_scene_text_contract(c, diagnostics["scene_text_contract"])

    # comparison_grid: tag left/right photo cutouts + Before/After/VS labels (geometry only).
    _tag_comparison_columns(kept, canvas, cfg)

    # stable z: keep qwen-derived z, then order remaining by area (large=back)
    def _area(c):
        return c["box"]["w"] * c["box"]["h"]

    max_z = max((c["z"] for c in kept), default=0)
    for c in kept:
        if c["z"] == 0 and c["meta"].get("source") == "ocr":
            c["z"] = max_z + 1  # text sits above shapes by default

    # Deterministic reading/z order: z, then back-to-front by area, then reading order
    # (top-to-bottom, left-to-right), with a stable id tie-break so identical scenes
    # reproduce byte-for-byte across runs.
    def _sort_key(c):
        box = c.get("box") or {}
        return (
            c["z"], -_area(c),
            round(float(box.get("y", 0) or 0), 3),
            round(float(box.get("x", 0) or 0), 3),
            str(c.get("id") or ""),
        )

    kept.sort(key=_sort_key)

    diagnostics["counts"] = {
        "candidates": len(kept),
        "fragment_suppressed": len(diagnostics["fragment_suppressed"]),
        "text_dedup": len(diagnostics["text_dedup"]),
        "element_dedup": len(diagnostics["element_dedup"]),
        "arrow_dedup": len(diagnostics["arrow_dedup"]),
        "guide_artifacts": len(diagnostics["guide_artifacts"]),
        "callout_groups": len(diagnostics["callout_groups"]),
        "scene_text_contract": len(diagnostics["scene_text_contract"]),
    }

    if run_dir:
        try:
            schema = importlib.import_module("src.schema")
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            schema = importlib.import_module("schema")
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(kept, os.path.join(run_dir, "merged.json"))
        # merged.json is a strict list (scene_intent contract); diagnostics ride alongside
        # as a sidecar report, mirroring reconstruction.json / qa.json.
        schema.dump(diagnostics, os.path.join(run_dir, "merge_report.json"))

    return kept


if __name__ == "__main__":  # CPU-safe smoke with fixtures
    ocr = {
        "lines": [
            {"id": "L0", "text": "BIG SALE", "conf": 0.98,
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}},
            {"id": "L1", "text": "Model wears watch", "conf": 0.8,
             "box": {"x": 300, "y": 400, "w": 180, "h": 24}},
        ]
    }
    elements = [
        {"id": "E0", "box": {"x": 20, "y": 20, "w": 260, "h": 90}, "kind": "shape",
         "area": 20000, "coverage": 0.1, "source": "residual-cc"},
        {"id": "E1", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "kind": "photo-fragment", "area": 40000, "coverage": 0.2,
         "source": "residual-cc"},
    ]
    qwen = [
        {"id": "Q0", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "png": "qwen_layers/Q0.png", "kind_hint": "photo"},
    ]
    cands = merge(ocr, elements, qwen, {"w": 600, "h": 600}, {})
    for c in cands:
        print(c["id"], "target=", c.get("target"), "z=", c["z"],
              "role=", c["meta"].get("role"), "kip=", c["meta"].get("kept_in_photo"))
