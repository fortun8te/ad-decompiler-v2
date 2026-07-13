"""routing.py — decide what each merged candidate BECOMES in Figma.

This is the validated decision layer, ported from the Mac harness. The rules here are the
reason the output is editable instead of a flattened trace. The agent never overrides these;
it only chooses tool settings and triggers retries.

route(candidate, canvas, cfg) -> candidate with `target` set to one of:
  'text'  editable Figma TEXT node
  'shape' Figma primitive (rect/ellipse + fitted fill) or VECTOR path
  'image' raster fill + mask (photos/products/people)
  'icon'  vectorized simple graphic (VTracer/Potrace), gated by fidelity
  'drop'  scene text / redundant — not a layer (text goes to kept_in_photo)
"""
from __future__ import annotations
from .wordmark import is_wordmark_candidate

# Simple-graphic size ceiling: only small cropped elements are eligible for vectorization.
ICON_MAX_AREA_FRAC = 0.06
EMOJI_RE_HINT = ("emoji", "pictograph")

# Roles that should attempt vector tracing when small enough.
VECTORIZE_ROLES = (
    "icon", "badge", "logo", "arrow", "symbol", "pictogram", "chip", "divider", "chrome",
)
# Flat UI chrome shapes that are often simple enough to trace instead of primitive-fit.
PRIMITIVE_SHAPE_ROLES = ("badge", "chip", "button", "divider", "card")

# Below this combined ink/font-match confidence, a text candidate cannot be faithfully
# reproduced as editable text (glyph too hard to isolate, or the closest font/effect
# match is a poor fit) — it is routed to a masked-pixel fallback layer instead of
# emitting a guessed rendering. Overridable via cfg["routing"]["min_text_fidelity"].
# Scores in the low 0.70s are generally still a solid local glyph match; below
# that we preserve exact pixels.  A higher cutoff made normal, proofread overlay
# copy disappear into raster fallbacks, defeating the editable Figma contract.
MIN_TEXT_FIDELITY = 0.75

# Roles whose rasterized cutout should be delivered as an IMAGE clipped by a swappable
# shape mask (see _image_mask). The raster is the swappable fill; the mask is the shape.
AVATAR_ROLES = ("avatar", "profile", "profile_picture", "profile_photo", "pfp",
                "headshot", "user_photo")
CARD_ROLES = ("card", "badge", "thumbnail", "tile")
LOGO_ROLES = ("logo", "wordmark", "brand", "logotype")


def _image_mask(candidate: dict, canvas: dict) -> dict:
    """Choose the swappable mask SHAPE for a rasterized image element.

    The raster cutout stays the (swappable) fill; the mask spec defines the clip so a
    logo/photo/avatar can be replaced in Figma without re-flattening it into the plate:
      avatar/profile  -> {"kind": "ellipse"}          (circular profile picture)
      card/badge      -> {"kind": "rrect", radius?}   (rounded card/thumbnail)
      logo/wordmark   -> {"kind": "path"}             (silhouette, traced in reconstruct)
      everything else -> {"kind": "alpha"}            (irregular cutout, own transparency)

    Only the coarse, role-driven hint is set here. Geometry that needs pixel evidence
    (round alpha coverage, corner radius, silhouette path) is finalized in reconstruct.py.
    Any pre-existing mask keys (notably ``src``, used by reconstruct to load the matte)
    are preserved.
    """
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    mask = dict(candidate.get("mask")) if isinstance(candidate.get("mask"), dict) else {}
    # A shape already decided upstream (or by an earlier pass) is authoritative.
    if mask.get("kind") and str(mask.get("kind")).lower() != "alpha":
        return mask
    if role in AVATAR_ROLES or meta.get("avatar") or meta.get("circular"):
        mask["kind"] = "ellipse"
    elif role in LOGO_ROLES or meta.get("wordmark"):
        mask["kind"] = "path"
    elif role in CARD_ROLES:
        mask["kind"] = "rrect"
        radius = candidate.get("radius")
        if radius is None:
            radius = (candidate.get("style") or {}).get("radius")
        if radius is not None:
            mask.setdefault("radius", radius)
    else:
        mask["kind"] = "alpha"
    return mask


def _area_frac(box, canvas):
    W = max(1, canvas.get("w", 1)); H = max(1, canvas.get("h", 1))
    return (box.get("w", 0) * box.get("h", 0)) / (W * H)


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text_fidelity_fallback(c: dict, meta: dict, cfg: dict | None) -> dict | None:
    """If this text candidate's ink/font-match confidence is below the fidelity gate,
    return a routed masked-pixel-fallback candidate; otherwise None (route as text)."""
    threshold = _num((cfg or {}).get("routing", {}).get("min_text_fidelity"), MIN_TEXT_FIDELITY)
    # Long social/body copy amplifies a small font mismatch across a large area.
    # Keep the ordinary overlay threshold permissive for CTA/headline editing,
    # but require a stronger render match before rebuilding a paragraph.
    semantic_role = str(meta.get("semantic_role") or meta.get("role") or "").lower()
    if semantic_role in {"body-copy", "body", "caption"}:
        threshold = max(threshold, 0.85)
    style = c.get("style") or {}
    fidelity_conf = meta.get("fidelity_confidence")
    if fidelity_conf is None:
        fidelity_conf = style.get("confidence")
    low_conf = bool(meta.get("low_fidelity")) or (fidelity_conf is not None and fidelity_conf < threshold)
    if not low_conf:
        return None
    c["target"] = "image"
    fallback_src = meta.get("fallback_src")
    if fallback_src:
        c["src"] = fallback_src
    else:
        mask = c.get("mask") if isinstance(c.get("mask"), dict) else {}
        mask.setdefault("kind", "alpha")
        c["mask"] = mask
    meta.setdefault("substitution", {
        "from": "text", "to": "image",
        "reason": meta.get("fidelity_reason") or "low-confidence font/effect match",
        "confidence": fidelity_conf,
    })
    meta["fallback"] = True
    return c


def route(candidate: dict, canvas: dict, cfg: dict | None = None) -> dict:
    c = dict(candidate)
    kind = c.get("kind")            # from element_detect / merge
    meta = c.setdefault("meta", {})

    # Normalize the public overlay vocabulary once. Reconstruction consumes
    # ``keep_underlay`` when building the canonical removal mask.
    if meta.get("preserve_underlay") or meta.get("overlay_without_removal"):
        meta["keep_underlay"] = True

    # 1. TEXT candidates (from OCR) --------------------------------------------------
    if c.get("text") is not None and kind in (None, "text"):
        # scene text printed on a product/photo → never a layer
        if (
            meta.get("origin") == "scene"
            or (meta.get("scene_text_role") == "printed_on_product"
                and meta.get("scene_text_corroborated") is True)
            or c.get("kept_in_photo")
        ):
            c["target"] = "drop"; meta["kept_in_photo"] = True
            return c
        # wordmark / brand lettering → artwork, not editable text, never font-matched
        if meta.get("scene_text_role") == "wordmark" or meta.get("wordmark"):
            # A platform lockup such as X.com is normally a logo glyph plus domain
            # lettering. Keep an exact cropped asset rather than risking a lossy
            # trace or treating the lettering as UI text.
            force_raster = bool(meta.get("platform_lockup"))
            c["target"] = "image" if force_raster or (cfg and cfg.get("wordmark_as_raster", True)) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            if c["target"] == "image":
                c["mask"] = _image_mask(c, canvas)
            return c
        if is_wordmark_candidate({"text": c.get("text"), "box": c.get("box"), "id": c.get("id")}, canvas):
            c["target"] = "image" if cfg and cfg.get("wordmark_as_raster", True) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            if c["target"] == "image":
                c["mask"] = _image_mask(c, canvas)
            return c
        # confidence/fidelity gate: a font/effect we cannot faithfully reproduce as
        # editable text falls back to the original painted pixels instead of a guess.
        fallback = _text_fidelity_fallback(c, meta, cfg)
        if fallback is not None:
            return fallback
        # emoji → keep as character in the text run, never vectorize (handled in build step)
        c["target"] = "text"
        return c

    # A VLM/SAM ownership pass may already have made the materialization decision.
    # Honor that contract before applying the broad role heuristics below: otherwise a
    # confirmed card/avatar/logo can be silently re-routed as a plate fragment simply
    # because a detector called it a generic ``shape``.  Text is deliberately handled
    # above -- its ownership contract is the stricter scene-text path.
    disposition = str(meta.get("layer_disposition") or meta.get("disposition") or "").lower()
    if disposition in {"plate", "background", "keep_in_background"}:
        c["target"] = "drop"
        meta["keep_in_background"] = True
        return c
    if disposition in {"foreground_raster", "raster", "image"}:
        c["target"] = "image"
        c["mask"] = _image_mask(c, canvas)
        return c
    if disposition in {"foreground_vector", "vector", "icon"}:
        c["target"] = "icon"
        return c
    if disposition in {"native_shape", "shape", "primitive"}:
        c["target"] = "shape"
        return c

    # 2. Explicit emoji candidate ----------------------------------------------------
    if meta.get("emoji") or c.get("codepoint"):
        c["target"] = "text"; meta["emoji"] = True
        return c

    # 3. Photos / products / people → raster + mask ---------------------------------
    if kind == "photo-fragment" or meta.get("role") in ("product", "person", "photo"):
        c["target"] = "image"
        # Swappable raster-in-shape (alpha for irregular cutouts; an avatar/card/logo role
        # upgrades to ellipse/rrect/path). reconstruct refines the geometry from pixels.
        c["mask"] = _image_mask(c, canvas)
        return c

    # 4. Icons / badges / simple graphics → vectorize (small only) ------------------
    if kind == "icon" or meta.get("role") in VECTORIZE_ROLES:
        if _area_frac(c.get("box", {}), canvas) <= ICON_MAX_AREA_FRAC:
            c["target"] = "icon"
        else:
            # too big to be a simple graphic → treat as shape or raster
            c["target"] = "shape" if meta.get("flat_fill") else "image"
        return c

    # 5. Shapes / cards / buttons → primitive when fill is solid/gradient -----------
    if kind == "shape":
        role = meta.get("role")
        small = _area_frac(c.get("box", {}), canvas) <= ICON_MAX_AREA_FRAC
        # A solid button/card/chip is already an editable native primitive. Tracing it
        # creates needless paths and worse corner geometry. Only explicitly non-primitive
        # artwork should enter the vector tracer from the generic shape branch.
        if small and meta.get("simple_graphic") and role not in PRIMITIVE_SHAPE_ROLES:
            c["target"] = "icon"
            return c
        c["target"] = "shape"
        if role in ("button", "badge", "chip", "card"):
            radius = c.get("radius")
            if radius is None:
                radius = (c.get("style") or {}).get("radius")
            if radius is not None:
                meta.setdefault("cornerRadius", radius)
                if c.get("radius") is None and isinstance(radius, (int, float)):
                    c["radius"] = radius
            if meta.get("role") == "button":
                meta["button_shell"] = True
        return c

    # 6. Fallback: unknown residual → raster crop (never a placeholder, never a trace)
    c["target"] = "image"
    meta["fallback"] = True
    c["mask"] = _image_mask(c, canvas)
    return c


def summarize(candidates: list) -> dict:
    out = {"text": 0, "shape": 0, "image": 0, "icon": 0, "drop": 0}
    for c in candidates:
        out[c.get("target", "image")] = out.get(c.get("target", "image"), 0) + 1
    return out
