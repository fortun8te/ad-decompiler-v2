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
from .raster_clusters import cluster_label, is_intentional_raster_cluster
from .diagram_editability import (
    is_chart_shape_role,
    is_chart_vector_role,
    route_target_for_role,
    should_route_as_chart_primitive,
)

# Simple-graphic size ceiling: only small cropped elements are eligible for vectorization.
ICON_MAX_AREA_FRAC = 0.06
EMOJI_RE_HINT = ("emoji", "pictograph")

# Roles that should attempt vector tracing when small enough.
VECTORIZE_ROLES = (
    "icon", "badge", "logo", "arrow", "symbol", "pictogram", "chip", "divider", "chrome",
    # Callout leaders are thin but frequently diagonal.  They need the normal
    # vector-render gate (with an exact raster fallback), never a rectangular
    # divider shortcut that would lose their direction/endpoint.
    "callout_leader", "leader", "leader_line", "connector",
    "string", "thread", "string_leader", "leader_string", "callout_string",
    "leader_dot", "endpoint_dot",
    "starburst", "price_burst", "sale_burst", "burst", "splat", "sticker_burst",
    "underline", "strikethrough", "strike_through",
    # IG story swipe-up / engagement chrome — chip on flat social plates.
    "swipe_up", "swipe-up", "story_cta", "story_chrome", "story_chip",
)
# Flat UI chrome shapes that are often simple enough to trace instead of primitive-fit.
PRIMITIVE_SHAPE_ROLES = (
    "badge", "chip", "button", "divider", "card", "banner", "seal", "callout", "pill",
)

# Below this combined ink/font-match confidence, a text candidate cannot be faithfully
# reproduced as editable text (glyph too hard to isolate, or the closest font/effect
# match is a poor fit) — it is routed to a masked-pixel fallback layer instead of
# emitting a guessed rendering. Overridable via cfg["routing"]["min_text_fidelity"].
# Anchored to the render-fit score scale (font_fit.py): exact font 0.72-0.88,
# correct-class substitute 0.33-0.58, wrong class 0.14-0.42 (wrong-class candidates
# are already filtered by the serif/sans class gate before this confidence is
# published). A correct-class fit must route as editable TEXT — the per-layer
# ink-IoU raster-slice gate (schema.raster_slice_thresholds) is the downstream
# arbiter that catches a fit that still renders badly. Pre-emptively rasterizing
# here at the old shape-match scale (0.75) silently converted whole ads to images
# (benchmark 009: 12/14 text blocks became rasters).
MIN_TEXT_FIDELITY = 0.40

# Roles whose rasterized cutout should be delivered as an IMAGE clipped by a swappable
# shape mask (see _image_mask). The raster is the swappable fill; the mask is the shape.
AVATAR_ROLES = ("avatar", "profile", "profile_picture", "profile_photo", "pfp",
                "headshot", "user_photo")
CIRCULAR_INSET_ROLES = (
    "circular_inset", "inset", "product_inset", "round_inset", "circle_crop",
)
CARD_ROLES = ("card", "badge", "thumbnail", "tile")
LOGO_ROLES = ("logo", "wordmark", "brand", "logotype")
# Story swipe-up / engagement chrome that should chip on flat social plates.
STORY_CHROME_ROLES = (
    "swipe_up", "swipe-up", "story_cta", "story_chrome", "story_chip",
)
EXTENDED_VECTOR_ROLES = {
    # These are commonly much larger than a small UI icon in advertising.  They still
    # enter the render-back gate; the larger ceiling only prevents premature flattening.
    "arrow", "callout_leader", "leader", "leader_line", "connector",
    "string", "thread", "string_leader", "leader_string", "callout_string",
    "leader_dot", "endpoint_dot",
    "starburst", "price_burst", "sale_burst", "burst", "splat", "sticker_burst",
    # Chart/diagram strokes and markers: often larger than ICON_MAX_AREA_FRAC but still
    # eligible for the same render-back gate (fail → exact raster, never a fake rect).
    "plot_line", "data_line", "data_point", "marker",
}


def _image_mask(candidate: dict, canvas: dict) -> dict:
    """Choose the swappable mask SHAPE for a rasterized image element.

    The raster cutout stays the (swappable) fill; the mask spec defines the clip so a
    logo/photo/avatar can be replaced in Figma without re-flattening it into the plate:
      avatar/profile  -> {"kind": "ellipse"}          (circular profile picture)
      circular inset  -> {"kind": "ellipse"}          (product in white ring)
      card/badge      -> {"kind": "rrect", radius?}   (rounded card/thumbnail)
      logo/wordmark   -> {"kind": "path"}             (silhouette, traced in reconstruct)
      everything else -> {"kind": "alpha"}            (irregular cutout, own transparency)

    Only the coarse, role-driven hint is set here. Geometry that needs pixel evidence
    (round alpha coverage, corner radius, silhouette path) is finalized in reconstruct.py.
    Any pre-existing mask keys (notably ``src``, used by reconstruct to load the matte)
    are preserved.
    """
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").lower().replace("-", "_")
    mask = dict(candidate.get("mask")) if isinstance(candidate.get("mask"), dict) else {}
    # A recognised UI/receipt/diagram/product cluster keeps its full source crop. A loose
    # SAM alpha can find the region, but it is not the intended visual boundary.
    if is_intentional_raster_cluster(role):
        radius = candidate.get("radius")
        if radius is None:
            radius = (candidate.get("style") or {}).get("radius", 0)
        mask["kind"] = "rrect"
        mask["radius"] = radius if isinstance(radius, (int, float)) else 0
        return mask
    # A shape already decided upstream (or by an earlier pass) is authoritative.
    if mask.get("kind") and str(mask.get("kind")).lower() != "alpha":
        return mask
    if (
        role in AVATAR_ROLES
        or role in CIRCULAR_INSET_ROLES
        or meta.get("avatar")
        or meta.get("circular")
        or meta.get("circular_inset")
    ):
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


# Archetypes whose plate is flat enough that an icon chip's baked-in surround is
# invisible (Codia's cutout trick). Photographic archetypes keep the vector/matte path.
# Prefer ``scene.format.capabilities.icons_as_chips`` when the format profile is present.
_CHIP_ARCHETYPES = {"social_screenshot", "product_on_flat", "comparison_grid"}


def _icons_as_chips(cfg) -> bool:
    """Config gate for icon→image-chip routing (default: ON for flat-plate archetypes)."""
    from src import format_readiness

    routing_cfg = (cfg or {}).get("routing") or {}
    if routing_cfg.get("icons_as_chips") is not None:
        return bool(routing_cfg.get("icons_as_chips"))
    fmt = format_readiness.format_from_cfg(cfg)
    if fmt.get("capabilities"):
        return format_readiness.prefers_icon_chips(cfg)
    return str(((cfg or {}).get("scene") or {}).get("archetype") or "") in _CHIP_ARCHETYPES


def _wordmark_as_raster(cfg: dict | None) -> bool:
    """Raster is the conservative default when brand artwork has no explicit override."""
    return bool((cfg or {}).get("wordmark_as_raster", True))


_FLAT_BRAND_OVERLAY_ROLES = frozenset({
    "headline", "title", "subheadline", "subtitle", "eyebrow", "brand", "logo",
})


def _editable_flat_brand_overlay(candidate: dict, meta: dict) -> bool:
    """True for MONTE/WAVY-style display brand on a cream/black plate.

    Short all-caps brand tokens sit in the header wordmark slot, but when merge has
    already marked them as flat-plate overlay copy (not baked into a product cutout),
    they must stay editable TEXT — not a separate logo raster.
    """
    if candidate.get("kept_in_photo") or meta.get("kept_in_photo"):
        return False
    if meta.get("baked_owner_id") or meta.get("origin") == "scene":
        return False
    if meta.get("platform_lockup"):
        return False
    # Never override the fidelity gate — low-confidence ink still becomes a pixel crop.
    if meta.get("low_fidelity") or meta.get("fallback"):
        return False
    if not (
        meta.get("overlay_text")
        or meta.get("removal_required")
        or meta.get("external_overlay")
        or meta.get("scene_text_role") == "overlay_copy"
    ):
        return False
    role = str(meta.get("role") or "").lower().replace("-", "_")
    return role in _FLAT_BRAND_OVERLAY_ROLES or meta.get("overlay_text") is True


def _text_fidelity_fallback(c: dict, meta: dict, cfg: dict | None) -> dict | None:
    """If this text candidate's ink/font-match confidence is below the fidelity gate,
    return a routed masked-pixel-fallback candidate; otherwise None (route as text)."""
    threshold = _num((cfg or {}).get("routing", {}).get("min_text_fidelity"), MIN_TEXT_FIDELITY)
    # One bar for every text role. The user's explicit preference is to keep text
    # EDITABLE in a plausible same-class font rather than slice it to pixels: body
    # copy in a clean substitute sans reads fine and is worth editing. The upstream
    # fidelity_confidence already encodes correct-class + legibility (a wrong-CLASS
    # render is capped below this bar in text_analysis), so a paragraph no longer
    # needs a stricter gate than a headline — the old body-copy bump to 0.50 only
    # rasterized long copy that a substitute font renders perfectly well
    # (benchmark 009: all 5 body blocks became pixels).
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
        # ``overlay_copy`` is a positive ownership decision, not merely a hint that
        # the text should remain a TEXT node.  It must also be removed from the clean
        # plate before that editable node is painted back on top.  Without this flag,
        # a body-style overlay inside a photo can be mistaken for printed packaging
        # text by reconstruction and quietly remain baked into the image.
        if meta.get("scene_text_role") == "overlay_copy":
            meta["overlay_text"] = True
            meta["removal_required"] = True
        # Flat-plate display brand (MONTE / WAVY headlines on cream/black) stays
        # editable TEXT. Packaging wordmarks that belong on a cutout are already
        # kept_in_photo above; platform lockups still force a raster asset.
        if _editable_flat_brand_overlay(c, meta):
            meta["overlay_text"] = True
            meta["removal_required"] = True
            meta.pop("wordmark", None)
            c["target"] = "text"
            return c
        # wordmark / brand lettering → artwork, not editable text, never font-matched
        if meta.get("scene_text_role") == "wordmark" or meta.get("wordmark"):
            # A platform lockup such as X.com is normally a logo glyph plus domain
            # lettering. Keep an exact cropped asset rather than risking a lossy
            # trace or treating the lettering as UI text.
            force_raster = bool(meta.get("platform_lockup"))
            c["target"] = "image" if force_raster or _wordmark_as_raster(cfg) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            if c["target"] == "image":
                c["mask"] = _image_mask(c, canvas)
            return c
        if is_wordmark_candidate({"text": c.get("text"), "box": c.get("box"), "id": c.get("id")}, canvas):
            c["target"] = "image" if _wordmark_as_raster(cfg) else "icon"
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

    # Structured but inseparable artwork is one exact source-backed image, not many
    # guessed layers. merge_layers permits contained text back out only with positive
    # external-overlay evidence and layout keeps that overlay grouped with this owner.
    if is_intentional_raster_cluster(meta.get("role")):
        c["target"] = "image"
        c["mask"] = _image_mask(c, canvas)
        meta["intentional_raster_cluster"] = True
        meta["swappable"] = True
        meta["contains_scene_text"] = True
        meta.setdefault("semantic_name", cluster_label(meta.get("role")))
        meta.setdefault("layer_disposition", "foreground_raster")
        meta.setdefault("z_band", "chrome" if str(meta.get("role") or "").lower() in {
            "screenshot", "ui-panel"
        } else "content")
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
    # Codia parity: emoji are PIXELS, never glyphs or traces (spec §2b/§7.4).
    # A color emoji vectorized to a flat single-color path was the single worst
    # visual artifact in benchmark 009; a text glyph depends on the platform's
    # emoji font and never matches the painted pixels. Ship the exact source
    # cutout at its tight pixel box (the baked-in local plate surround makes a
    # matte unnecessary on flat plates, exactly like Codia's cutouts).
    if meta.get("emoji") or c.get("codepoint"):
        c["target"] = "image"
        meta["emoji"] = True
        meta["role"] = meta.get("role") or "emoji"
        meta.setdefault("layer_disposition", "foreground_raster")
        meta.setdefault("intentional_raster_cluster", True)
        mask = dict(c.get("mask")) if isinstance(c.get("mask"), dict) else {}
        mask["kind"] = "alpha"
        c["mask"] = mask
        return c

    # 3. Chart / diagram primitives (semi-editable) ---------------------------------
    # Positively tagged marks must not fall through the photo-fragment catch-all.
    # Whole chart/diagram clusters stay intentional rasters above; only decomposed
    # members with chart primitive roles reach this branch.
    if should_route_as_chart_primitive(c):
        chart_target = route_target_for_role(meta.get("role"))
        if chart_target == "shape" and is_chart_shape_role(meta.get("role")):
            c["target"] = "shape"
            c["shape_kind"] = "rect"
            meta["diagram_mark"] = True
            meta["native_chart_primitive"] = True
            return c
        if chart_target == "icon" and is_chart_vector_role(meta.get("role")):
            role_key = str(meta.get("role") or "").lower().replace("-", "_")
            max_fraction = 0.20 if role_key in EXTENDED_VECTOR_ROLES else ICON_MAX_AREA_FRAC
            if _area_frac(c.get("box", {}), canvas) <= max_fraction:
                c["target"] = "icon"
                meta["diagram_mark"] = True
                meta["native_chart_primitive"] = True
            else:
                c["target"] = "image"
                meta["vector_fallback"] = True
                meta["diagram_mark"] = True
                c["mask"] = _image_mask(c, canvas)
            return c

    # 4. Photos / products / people → raster + mask ---------------------------------
    # Never vectorize photographic material, even when it sits near a diagram.
    if kind == "photo-fragment" or meta.get("role") in ("product", "person", "photo"):
        c["target"] = "image"
        # Swappable raster-in-shape (alpha for irregular cutouts; an avatar/card/logo role
        # upgrades to ellipse/rrect/path). reconstruct refines the geometry from pixels.
        c["mask"] = _image_mask(c, canvas)
        return c

    # A true divider is already a native Figma primitive.  Do *not* include every
    # ``kind: line`` here: annotation leaders are often diagonal (e.g. product
    # explainer ads) and a rectangle would erase their direction and endpoint.
    # Those remain on the gated vector/raster route below.
    if kind == "divider" or str(meta.get("role") or "").lower() in {"divider", "rule", "separator"}:
        c["target"] = "shape"
        c["shape_kind"] = "rect"  # preserves the observed horizontal/vertical bar thickness.
        meta["native_divider"] = True
        return c

    # 4. Text-bearing badge/button/logo/banner chrome → native SHAPE plate (not a
    #    traced/rasterized seal with OCR baked in). Contained OCR stays TEXT with removal.
    #    Irregular brushstrokes/starbursts may later fall back to an alpha CHIP in
    #    reconstruct when primitive/path fidelity fails — never a full-image slice.
    if meta.get("text_bearing_shell") or meta.get("plate_shell"):
        role = str(meta.get("role") or "").lower().replace("-", "_")
        if role in {"logo", "wordmark", "brand"}:
            meta["reclassified_from"] = meta.get("reclassified_from") or role
            meta["role"] = "badge"
            role = "badge"
        elif role in {"shape", "card", "panel", "frame", "container", "plate", ""}:
            # Geometry classify: wide → banner, else badge (matches merge/reconstruct).
            box = c.get("box") or {}
            w = float(box.get("w", 0) or 0)
            h = float(box.get("h", 0) or 0)
            new_role = "banner" if h > 0 and w / h >= 2.2 else "badge"
            if role and role != new_role:
                meta.setdefault("reclassified_from", role or "shape")
            meta["role"] = new_role
            role = new_role
        c["target"] = "shape"
        meta["plate_shell"] = True
        if role == "button":
            meta["button_shell"] = True
        radius = c.get("radius")
        if radius is None:
            radius = (c.get("style") or {}).get("radius")
        if radius is not None:
            meta.setdefault("cornerRadius", radius)
            if c.get("radius") is None and isinstance(radius, (int, float)):
                c["radius"] = radius
        return c

    # 5. Icons / badges / simple graphics → vectorize (small only) ------------------
    if kind in {"icon", "line"} or meta.get("role") in VECTORIZE_ROLES:
        role = str(meta.get("role") or "").lower().replace("-", "_")
        # Codia confidence ladder: on flat-plate archetypes an icon ships as an exact
        # IMAGE chip with its local plate surround baked in — vector tracing is a
        # declared non-goal (chips are pixel-exact and trivially swappable; Codia's
        # engagement icons/badges are all cutouts). Leaders/bursts keep the vector
        # path (they are often diagonal linework a chip box would mangle).
        # Codia confidence ladder: on flat-plate archetypes an icon ships as an exact
        # IMAGE chip with its local plate surround baked in — vector tracing is a
        # declared non-goal (chips are pixel-exact and trivially swappable; Codia's
        # engagement icons/badges are all cutouts). Leaders/bursts keep the vector
        # path (they are often diagonal linework a chip box would mangle).
        story_chrome = role in STORY_CHROME_ROLES or bool(meta.get("story_chrome"))
        if (_icons_as_chips(cfg) and role not in EXTENDED_VECTOR_ROLES
                and _area_frac(c.get("box", {}), canvas) <= 0.20):
            c["target"] = "image"
            meta["icon_chip"] = True
            if story_chrome:
                meta["story_chrome_chip"] = True
            meta.setdefault("intentional_raster_cluster", True)
            mask = dict(c.get("mask")) if isinstance(c.get("mask"), dict) else {}
            mask["kind"] = "alpha"
            c["mask"] = mask
            return c
        max_fraction = 0.20 if role in EXTENDED_VECTOR_ROLES else ICON_MAX_AREA_FRAC
        if _area_frac(c.get("box", {}), canvas) <= max_fraction:
            c["target"] = "icon"
        else:
            # Never approximate a detailed burst/arrow as a rectangle. Oversized
            # decorative graphics keep their exact source crop.
            c["target"] = "image"
            meta["vector_fallback"] = True
            c["mask"] = _image_mask(c, canvas)
        return c

    # 6. Shapes / cards / buttons → primitive when fill is solid/gradient -----------
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

    # 7. Fallback: unknown residual → raster crop (never a placeholder, never a trace)
    c["target"] = "image"
    meta["fallback"] = True
    c["mask"] = _image_mask(c, canvas)
    return c


def summarize(candidates: list) -> dict:
    out = {"text": 0, "shape": 0, "image": 0, "icon": 0, "drop": 0}
    for c in candidates:
        out[c.get("target", "image")] = out.get(c.get("target", "image"), 0) + 1
    return out
