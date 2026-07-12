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
VECTORIZE_SHAPE_ROLES = ("badge", "chip", "button", "divider")

# Below this combined ink/font-match confidence, a text candidate cannot be faithfully
# reproduced as editable text (glyph too hard to isolate, or the closest font/effect
# match is a poor fit) — it is routed to a masked-pixel fallback layer instead of
# emitting a guessed rendering. Overridable via cfg["routing"]["min_text_fidelity"].
MIN_TEXT_FIDELITY = 0.30


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

    # 1. TEXT candidates (from OCR) --------------------------------------------------
    if c.get("text") is not None and kind in (None, "text"):
        # scene text printed on a product/photo → never a layer
        if (
            meta.get("origin") == "scene"
            or meta.get("scene_text_role") == "printed_on_product"
            or c.get("kept_in_photo")
        ):
            c["target"] = "drop"; meta["kept_in_photo"] = True
            return c
        # wordmark / brand lettering → artwork, not editable text, never font-matched
        if meta.get("scene_text_role") == "wordmark" or meta.get("wordmark"):
            c["target"] = "image" if cfg and cfg.get("wordmark_as_raster", True) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            return c
        if is_wordmark_candidate({"text": c.get("text"), "box": c.get("box"), "id": c.get("id")}, canvas):
            c["target"] = "image" if cfg and cfg.get("wordmark_as_raster", True) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            return c
        # confidence/fidelity gate: a font/effect we cannot faithfully reproduce as
        # editable text falls back to the original painted pixels instead of a guess.
        fallback = _text_fidelity_fallback(c, meta, cfg)
        if fallback is not None:
            return fallback
        # emoji → keep as character in the text run, never vectorize (handled in build step)
        c["target"] = "text"
        return c

    # 2. Explicit emoji candidate ----------------------------------------------------
    if meta.get("emoji") or c.get("codepoint"):
        c["target"] = "text"; meta["emoji"] = True
        return c

    # 3. Photos / products / people → raster + mask ---------------------------------
    if kind == "photo-fragment" or meta.get("role") in ("product", "person", "photo"):
        c["target"] = "image"
        c.setdefault("mask", {"kind": "alpha"})   # alpha from qwen layer or matte
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
        if small and (
            meta.get("flat_fill")
            or meta.get("simple_graphic")
            or role in VECTORIZE_SHAPE_ROLES
        ):
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
    return c


def summarize(candidates: list) -> dict:
    out = {"text": 0, "shape": 0, "image": 0, "icon": 0, "drop": 0}
    for c in candidates:
        out[c.get("target", "image")] = out.get(c.get("target", "image"), 0) + 1
    return out
