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


def _area_frac(box, canvas):
    W = max(1, canvas.get("w", 1)); H = max(1, canvas.get("h", 1))
    return (box.get("w", 0) * box.get("h", 0)) / (W * H)


def route(candidate: dict, canvas: dict, cfg: dict | None = None) -> dict:
    c = dict(candidate)
    src = c.get("source", "")
    kind = c.get("kind")            # from element_detect / merge
    meta = c.setdefault("meta", {})

    # 1. TEXT candidates (from OCR) --------------------------------------------------
    if c.get("text") is not None and kind in (None, "text"):
        # scene text printed on a product/photo → never a layer
        if meta.get("origin") == "scene" or c.get("kept_in_photo"):
            c["target"] = "drop"; meta["kept_in_photo"] = True
            return c
        # wordmark / brand lettering → artwork, not editable text, never font-matched
        if is_wordmark_candidate({"text": c.get("text"), "box": c.get("box"), "id": c.get("id")}, canvas):
            c["target"] = "image" if cfg and cfg.get("wordmark_as_raster", True) else "icon"
            meta["wordmark"] = True
            meta["role"] = meta.get("role") or "logo"
            return c
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
    if kind == "icon" or meta.get("role") in ("icon", "badge", "logo", "arrow"):
        if _area_frac(c.get("box", {}), canvas) <= ICON_MAX_AREA_FRAC:
            c["target"] = "icon"
        else:
            # too big to be a simple graphic → treat as shape or raster
            c["target"] = "shape" if meta.get("flat_fill") else "image"
        return c

    # 5. Shapes / cards / buttons → primitive when fill is solid/gradient -----------
    if kind == "shape":
        c["target"] = "shape"
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
