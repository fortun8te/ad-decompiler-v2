"""peel_scene.py — element-guided, occlusion-attributed peel decomposition.

``peel_decompose.py`` is the *blind* LayerD loop: it re-discovers layers with a matting
model because it knows nothing about the scene.  This module is the pipeline-native
sibling for the case where detection has ALREADY happened: the fused elements
(fused_elements.json) and a z-order tell us exactly which pixels belong to which layer,
so peeling becomes a bookkeeping problem — *whose* pixels does the peeled footprint
reveal? — instead of a re-segmentation problem.

The core correctness contract (the reason this module exists):

    When a top element sits over MULTIPLE distinct underlying layers, peeling it must
    inpaint its footprint ONLY into the specific underlying layer(s) directly beneath
    it, and underlying layers that were NOT covered stay byte-identical.

Concretely (the before/after ad): a circular product shot straddles the seam between a
left portrait and a right portrait.  The circle's footprint splits — the left part of
the hole is inpainted into the LEFT portrait using ONLY the left portrait's own pixels
as context, the right part into the RIGHT portrait likewise.  Neither portrait gets a
hole where the circle never covered it, and the scene is never treated as one background
with one hole.

Algorithm (equivalent to iterating LayerD peels topmost-first, computed closed-form):

  1. Order elements by z (topmost first).  Peel order is reverse z-order; each
     element's footprint is its detection mask — no matting model in the loop.
  2. For every layer L, its occluded region is  mask(L) ∩ ⋃ mask(higher).  Attribute
     each occluded pixel to its DIRECT occluder (the lowest element above L covering
     it) — that is exactly the layer-owner split an iterative peel produces: peeling
     top layer T attributes T's footprint pixels to the next-lower owner at each pixel.
  3. Complete L by inpainting its occluded region into L's OWN RGBA with
     **context isolation**: the inpaint call sees only L's visible pixels as known
     context (every non-L pixel in the crop is masked as unknown), so a hole at a seam
     can never bleed the neighbouring layer's colors into L.
  4. Pixels of L that were visible in the flat image are copied byte-identical; only
     the occluded region (plus an optional anti-alias fringe ring) is synthesized.
     A layer with nothing on top of it comes out untouched.
  5. The background plate is completed the same way (context = pixels no element
     covers), or reused verbatim when the caller passes the pipeline's clean plate.

Re-compositing background + layers back-to-front reproduces the input exactly (with
hard masks): at every pixel the topmost owner painted its original flattened pixel,
and inpainted pixels are always covered by an occluder above.

Selective use: peel is only worth running when elements genuinely overlap.
``overlap_report`` gates it — no qualifying element-over-element overlap means
``peel_scene`` returns a skipped result and the pipeline keeps the existing
single-plate path (see docs/PEEL-DECOMPOSITION.md §"When peel runs").

Zero import coupling to the heavy pipeline modules: inpainting is an injected
callable (default: deterministic OpenCV Telea from peel_decompose).  The pipeline
injects its Big-LaMa/Flux router at the call site; per-call ``meta`` lets the router
keep text-shaped holes away from Flux (glyph residue).

Injected inpaint contract (either signature works; ``meta`` is detected once):

    inpaint(rgb: HxWx3 uint8, mask: HxW bool) -> HxWx3 uint8
    inpaint(rgb: HxWx3 uint8, mask: HxW bool, meta: dict) -> HxWx3 uint8

    * only ``mask`` pixels may be treated as unknown/rewritten; the caller copies back
      an even smaller region, so extra conservatism is safe but never required
    * ``meta`` = {"under_id", "under_kind", "occluder_ids", "text_occluder": bool,
                  "isolated_context": bool}
"""
from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from src import peel_decompose
from src.peel_decompose import _deps  # numpy/cv2/PIL ladder with the shared error text


# ── configuration ──────────────────────────────────────────────────────────────────

#: Scene-peel additions on top of peel_decompose.DEFAULTS (same cfg["peel"] block).
SCENE_DEFAULTS = {
    "min_overlap_area": 64,      # px² — smaller intersections don't justify a peel run
    "min_overlap_frac": 0.02,    # fraction of the SMALLER element's area
    "hole_dilate_px": 2,         # widen holes past anti-aliased fringes (0 = exact masks)
    "context_pad_px": 24,        # crop padding around a layer for inpaint context
    "min_context_frac": 0.05,    # below this visible/total ratio, isolation is hopeless
    "text_occluders": "box",     # box | off — see docs (ghost-text vs box overfill)
    "refine_alpha": False,       # matting-refine cutout EDGES (needs a matting callable)
    "refine_band_px": 3,         # width of the edge band the matting may adjust
}


def _options(cfg: Optional[dict]) -> dict:
    opts = {**SCENE_DEFAULTS, **peel_decompose._options(cfg)}
    # peel_decompose._options already merged cfg["peel"] over its own DEFAULTS but not
    # over SCENE_DEFAULTS; re-apply the user block so scene keys win too.
    for key, value in ((cfg or {}).get("peel") or {}).items():
        if key in SCENE_DEFAULTS:
            opts[key] = value
    return opts


# ── inputs / results ───────────────────────────────────────────────────────────────

@dataclass
class SceneElement:
    """One detected layer participating in the peel. ``z``: higher = closer to viewer."""
    id: str
    mask: object                  # HxW bool full-canvas footprint (np.ndarray)
    z: float
    kind: str = "element"         # semantic hint (icon/photo/shape/…); advisory
    is_text: bool = False         # text acts as an occluder but is never emitted
    alpha: object = None          # optional HxW float refined alpha for the cutout
    meta: dict = field(default_factory=dict)


@dataclass
class OcclusionFill:
    """One attributed sub-hole: ``occluder_id``'s footprint filled into an under-layer."""
    occluder_id: str
    area: int                     # px of the attributed sub-hole (pre-dilation)
    bbox: dict                    # tight bbox of the sub-hole
    text_occluder: bool = False


@dataclass
class ScenePeelLayer:
    """One COMPLETE output layer: original visible pixels + inpainted occluded region."""
    id: str
    rgba: object                  # HxWx4 uint8 full canvas
    bbox: dict                    # tight bbox of the mask
    z_index: int                  # 0 = furthest back foreground (background plate is below all)
    kind: str
    occluded_by: list             # ids of higher layers whose masks intersect this one
    occludes: list                # ids of lower non-text layers this one covers
    fills: list = field(default_factory=list)   # list[OcclusionFill]
    meta: dict = field(default_factory=dict)

    @property
    def filled_area(self) -> int:
        return sum(f.area for f in self.fills)


@dataclass
class ScenePeelResult:
    layers: list                  # list[ScenePeelLayer] back-to-front (ascending z_index)
    background: object            # HxWx3 uint8 complete plate (np.ndarray) or None if skipped
    canvas: dict                  # {"w", "h"}
    skipped: bool = False
    skip_reason: Optional[str] = None
    overlap: dict = field(default_factory=dict)   # overlap_report() output
    background_fills: list = field(default_factory=list)   # list[OcclusionFill]
    meta: dict = field(default_factory=dict)      # recomposite check, notes

    def layer(self, layer_id: str) -> Optional[ScenePeelLayer]:
        return next((l for l in self.layers if l.id == layer_id), None)


# ── z-order derivation (mirrors reconstruct._ownership_priority, not imported) ──────

_KIND_BAND = {
    # text is frontmost; icons/badges above cutouts; cutouts above generic shapes;
    # broad photos/backgrounds furthest back.  Intentionally the same bands as
    # reconstruct._ownership_priority so peel and ownership agree on who is on top.
    "text": 40,
    "icon": 30, "badge": 30, "logo": 30, "arrow": 30, "button": 25,
    "product": 20, "person": 20, "cutout": 20, "foreground": 20,
    "shape": 10, "card": 10, "panel": 10,
    "photo": 5, "photo-fragment": 5, "image": 5, "background": 0,
}


def derive_z_order(elements: list) -> list:
    """Assign unique integer ``z`` (higher = front) to SceneElements missing one.

    Heuristic mirror of reconstruct's ownership priority: semantic band first, then a
    strict-containment boost (a mask sitting mostly inside another is on top of it),
    then smaller-area-in-front.  Elements that already carry distinct ``z`` values are
    left untouched.
    """
    _, np, _ = _deps()
    zs = [e.z for e in elements]
    if len(set(zs)) == len(zs) and any(z != 0 for z in zs):
        return elements
    areas = {e.id: max(1, int(np.count_nonzero(e.mask))) for e in elements}
    contained_in = {e.id: 0 for e in elements}
    for a in elements:
        for b in elements:
            if a.id == b.id:
                continue
            inter = int(np.count_nonzero(np.logical_and(a.mask, b.mask)))
            if inter / areas[a.id] >= 0.85 and areas[a.id] < areas[b.id]:
                contained_in[a.id] += 1     # a rides on top of b
    ranked = sorted(elements, key=lambda e: (
        _KIND_BAND.get(str(e.kind).lower(), 10 if not e.is_text else 40),
        contained_in[e.id],
        -areas[e.id],
        e.id,
    ))
    for z, element in enumerate(ranked):
        element.z = float(z)
    return elements


# ── run-artifact loader ─────────────────────────────────────────────────────────────

def elements_from_run(run_dir: str, fused_elements: list, canvas: dict,
                      cfg: Optional[dict] = None, ocr: Optional[dict] = None) -> list:
    """Build SceneElements from fused_elements.json entries (+ optional OCR occluders).

    Element masks are the pipeline's own bbox-cropped alpha PNGs (``mask_src``) pasted
    into a full-canvas bool; an element without a readable mask falls back to its box.
    Text lines from ``ocr`` become box-footprint occluders when
    ``peel.text_occluders == "box"`` (they are never emitted as peel layers — text
    stays native OCR/font layers).  z is derived via ``derive_z_order``.
    """
    _, np, Image = _deps()
    opts = _options(cfg)
    w, h = int(canvas.get("w", 0)), int(canvas.get("h", 0))
    out: list = []
    for element in fused_elements or []:
        box = element.get("box") or {}
        mask = np.zeros((h, w), bool)
        x = max(0, int(round(box.get("x", 0))))
        y = max(0, int(round(box.get("y", 0))))
        bw = max(0, int(round(box.get("w", 0))))
        bh = max(0, int(round(box.get("h", 0))))
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x or y1 <= y:
            continue
        src = element.get("mask_src") or (element.get("mask") or {}).get("src")
        painted = False
        if src:
            path = src if os.path.isabs(src) else os.path.join(run_dir, src)
            if os.path.exists(path):
                crop = np.asarray(Image.open(path).convert("L")) > 127
                ch, cw = crop.shape
                mask[y:min(h, y + ch), x:min(w, x + cw)] = \
                    crop[:min(h, y + ch) - y, :min(w, x + cw) - x]
                painted = True
        if not painted:
            mask[y:y1, x:x1] = True
        if not mask.any():
            continue
        kind = str(element.get("kind") or element.get("role") or "element")
        out.append(SceneElement(id=str(element.get("id")), mask=mask, z=0.0, kind=kind,
                                meta={"role": element.get("role"),
                                      "box_only_mask": not painted}))
    derive_z_order(out)
    top_z = max([e.z for e in out], default=-1.0) + 1.0
    if ocr and str(opts.get("text_occluders", "box")).lower() == "box":
        for line in ocr.get("lines") or []:
            box = line.get("box") or {}
            x = max(0, int(round(box.get("x", 0))))
            y = max(0, int(round(box.get("y", 0))))
            x1 = min(w, x + max(0, int(round(box.get("w", 0)))))
            y1 = min(h, y + max(0, int(round(box.get("h", 0)))))
            if x1 <= x or y1 <= y:
                continue
            mask = np.zeros((h, w), bool)
            mask[y:y1, x:x1] = True
            out.append(SceneElement(id=f"text_{line.get('id')}", mask=mask, z=top_z,
                                    kind="text", is_text=True,
                                    meta={"footprint": "ocr-box",
                                          "note": "box overfill accepted to avoid ghost text"}))
            top_z += 1.0
    return out


# ── overlap gate ───────────────────────────────────────────────────────────────────

def _tight_bbox(mask) -> dict:
    return peel_decompose._tight_bbox(mask)


def overlap_report(elements: list, cfg: Optional[dict] = None) -> dict:
    """Pairwise occlusion census + the go/no-go gate for running peel at all.

    A pair qualifies when the intersection is at least ``peel.min_overlap_area`` px AND
    at least ``peel.min_overlap_frac`` of the smaller element's area.  Peel is *needed*
    only when some qualifying pair covers a NON-TEXT under-layer — an element sitting
    on nothing but background is already handled by the single-plate path, and text
    under-layers stay native.
    """
    _, np, _ = _deps()
    opts = _options(cfg)
    min_area = int(opts["min_overlap_area"])
    min_frac = float(opts["min_overlap_frac"])
    boxes = {e.id: _tight_bbox(e.mask) for e in elements}
    areas = {e.id: int(np.count_nonzero(e.mask)) for e in elements}
    ordered = sorted(elements, key=lambda e: -e.z)
    pairs = []
    for i, top in enumerate(ordered):
        for under in ordered[i + 1:]:
            if top.z <= under.z:
                continue
            bt, bu = boxes[top.id], boxes[under.id]
            if (bt["x"] + bt["w"] <= bu["x"] or bu["x"] + bu["w"] <= bt["x"]
                    or bt["y"] + bt["h"] <= bu["y"] or bu["y"] + bu["h"] <= bt["y"]):
                continue
            inter = int(np.count_nonzero(np.logical_and(top.mask, under.mask)))
            if inter <= 0:
                continue
            frac = inter / max(1, min(areas[top.id], areas[under.id]))
            qualifies = inter >= min_area and frac >= min_frac
            pairs.append({"top": top.id, "under": under.id, "area": inter,
                          "frac": round(frac, 5), "under_is_text": under.is_text,
                          "qualifies": qualifies})
    needed = any(p["qualifies"] and not p["under_is_text"] for p in pairs)
    return {"pairs": pairs, "needed": needed,
            "thresholds": {"min_overlap_area": min_area, "min_overlap_frac": min_frac}}


# ── the layer-owner hole split ─────────────────────────────────────────────────────

def attribute_footprint(footprint, lower_elements: list) -> dict:
    """Split a peeled footprint by which underlying layer owns each pixel.

    ``lower_elements`` are the layers strictly below the peeled element, ANY z order.
    Each footprint pixel is attributed to the topmost lower element covering it; pixels
    no lower element covers go to ``"background"``.  Returns ``{owner_id: bool submask}``
    with empty owners omitted.  This is the per-peel view; ``peel_scene`` computes the
    accumulated equivalent per under-layer via direct-occluder maps.
    """
    _, np, _ = _deps()
    footprint = np.asarray(footprint, bool)
    owner = np.zeros(footprint.shape, np.int32)      # 0 = background
    order = sorted(lower_elements, key=lambda e: e.z)  # bottom→top; later paint wins
    ids = {}
    for number, element in enumerate(order, start=1):
        ids[number] = element.id
        owner[np.asarray(element.mask, bool)] = number
    split = {}
    for number, element_id in list(ids.items()) + [(0, "background")]:
        sub = footprint & (owner == number)
        if sub.any():
            split[element_id] = sub
    return split


def _direct_occluder_map(under: SceneElement, higher: list):
    """int map over ``under.mask``: per occluded pixel, the index+1 into ``higher`` of
    the DIRECT occluder (the lowest element above ``under`` covering the pixel)."""
    _, np, _ = _deps()
    occluder = np.zeros(under.mask.shape, np.int32)
    # Paint from topmost down; the last painter (lowest above `under`) wins the pixel.
    for number, element in enumerate(sorted(higher, key=lambda e: -e.z), start=1):
        occluder[np.asarray(element.mask, bool)] = number
    occluder[~np.asarray(under.mask, bool)] = 0
    return occluder, sorted(higher, key=lambda e: -e.z)


# ── optional matting-based EDGE refinement (never re-detection) ────────────────────

def refine_element_alpha(image, element: SceneElement, matting: Callable,
                         cfg: Optional[dict] = None) -> SceneElement:
    """Refine the anti-aliased EDGE of one element's alpha with a matting model.

    The matting (e.g. BiRefNet, as in the blind LayerD loop) is consulted ONLY inside
    a ±``peel.refine_band_px`` ring around the detection mask's boundary: the interior
    stays fully opaque and the exterior fully transparent, so the model can soften a
    cutout edge but can never grow, shrink, or re-detect the layer.  Sets
    ``element.alpha`` in place and returns the element.  Hole geometry still uses the
    hard detection mask — refinement only affects the peeled cutout's own edge.
    """
    cv2, np, _ = _deps()
    opts = _options(cfg)
    band_px = max(1, int(opts["refine_band_px"]))
    flat = peel_decompose._to_rgb(image)
    h, w = flat.shape[:2]
    mask = np.asarray(element.mask, bool)
    if not mask.any():
        return element
    box = _tight_bbox(mask)
    x0, y0, x1, y1 = _pad_bbox(box, band_px + int(opts["context_pad_px"]), w, h)
    crop = np.ascontiguousarray(flat[y0:y1, x0:x1])
    matte = np.clip(np.asarray(matting(crop), np.float64), 0.0, 1.0)
    if matte.shape != crop.shape[:2]:
        raise ValueError(f"matting returned {matte.shape}, expected {crop.shape[:2]}")

    kernel = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
    hard = mask.astype(np.uint8)
    inner = cv2.erode(hard, kernel) > 0
    outer = cv2.dilate(hard, kernel) > 0
    band = outer & ~inner

    alpha = mask.astype(np.float64)
    full_matte = np.zeros((h, w), np.float64)
    full_matte[y0:y1, x0:x1] = matte
    alpha[band] = full_matte[band]
    alpha[inner] = 1.0
    alpha[~outer] = 0.0
    element.alpha = alpha
    element.meta["alpha_refined"] = {"band_px": band_px}
    return element


# ── inpaint invocation ─────────────────────────────────────────────────────────────

def _accepts_meta(fn: Callable) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "meta" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _pad_bbox(box: dict, pad: int, w: int, h: int) -> tuple:
    x0 = max(0, box["x"] - pad)
    y0 = max(0, box["y"] - pad)
    x1 = min(w, box["x"] + box["w"] + pad)
    y1 = min(h, box["y"] + box["h"] + pad)
    return x0, y0, x1, y1


def _dilate_px(mask, px: int):
    if px <= 0:
        return mask
    return peel_decompose._expand_mask(mask, (2 * px + 1, 2 * px + 1))


def _fill_region(flat, layer_rgb, element_mask, visible, write, inpaint,
                 accepts_meta, meta, opts):
    """Inpaint ``write`` pixels into ``layer_rgb`` using ONLY ``visible`` as context.

    Context isolation: within the padded crop, everything that is not a visible pixel
    of this layer is part of the inpaint mask, so the filler cannot read neighbouring
    layers (the seam-bleed failure).  Only ``write`` pixels are copied back.  When the
    layer is almost fully covered there is no usable context; degrade honestly to an
    unisolated fill and record it in the returned note.
    """
    _, np, _ = _deps()
    h, w = flat.shape[:2]
    box = _tight_bbox(np.logical_or(element_mask, write))
    x0, y0, x1, y1 = _pad_bbox(box, int(opts["context_pad_px"]), w, h)
    crop = flat[y0:y1, x0:x1]
    visible_crop = visible[y0:y1, x0:x1]
    write_crop = write[y0:y1, x0:x1]

    mask_area = max(1, int(np.count_nonzero(element_mask)))
    isolated = (int(np.count_nonzero(visible)) / mask_area) >= float(opts["min_context_frac"])
    if isolated:
        call_mask = np.logical_or(~visible_crop, write_crop)
    else:
        call_mask = _dilate_px(write_crop, int(opts["hole_dilate_px"]))
    meta = dict(meta, isolated_context=bool(isolated))
    if accepts_meta:
        filled = inpaint(np.ascontiguousarray(crop), call_mask, meta=meta)
    else:
        filled = inpaint(np.ascontiguousarray(crop), call_mask)
    filled = np.asarray(filled, dtype=np.uint8)
    if filled.shape != crop.shape:
        raise ValueError(f"inpaint returned {filled.shape}, expected {crop.shape}")
    region = layer_rgb[y0:y1, x0:x1]
    region[write_crop] = filled[write_crop]
    return isolated


# ── the peel ───────────────────────────────────────────────────────────────────────

def peel_scene(image, elements: list, inpaint: Optional[Callable] = None,
               cfg: Optional[dict] = None, background=None,
               matting: Optional[Callable] = None,
               force: bool = False) -> ScenePeelResult:
    """Occlusion-attributed peel over detected elements (see module docstring).

    Args:
        image: path / PIL / HxWx3(4) uint8 flattened design.
        elements: list[SceneElement] with full-canvas bool masks and z (higher = front).
            Text elements participate as occluders only.
        inpaint: hole filler callable (see module docstring); default OpenCV Telea.
        cfg: pipeline config; reads the ``peel`` block.
        background: optional pre-built clean plate (path or HxWx3 array) — e.g. the
            pipeline's background_clean.png — reused verbatim instead of re-inpainting.
        matting: optional ``rgb -> float alpha`` callable (BiRefNet); used ONLY to
            refine cutout edges when ``peel.refine_alpha`` is on — never to detect.
        force: bypass the overlap gate (demo/tests).

    Returns:
        ScenePeelResult. ``skipped=True`` (with ``skip_reason``) when the overlap gate
        says the single-plate path already covers this scene — the caller keeps the
        existing pipeline behavior in that case.
    """
    _, np, _ = _deps()
    opts = _options(cfg)
    flat = peel_decompose._to_rgb(image)
    h, w = flat.shape[:2]
    canvas = {"w": w, "h": h}
    for element in elements:
        if np.asarray(element.mask).shape != (h, w):
            raise ValueError(f"element {element.id}: mask shape "
                             f"{np.asarray(element.mask).shape} != canvas {(h, w)}")

    report = overlap_report(elements, cfg)
    if not report["needed"] and not force:
        return ScenePeelResult(layers=[], background=None, canvas=canvas, skipped=True,
                               skip_reason="no-overlap", overlap=report)

    if inpaint is None:
        inpaint = peel_decompose.opencv_inpaint
    accepts_meta = _accepts_meta(inpaint)
    dilate = int(opts["hole_dilate_px"])

    ordered = sorted([e for e in elements], key=lambda e: e.z)   # back-to-front
    non_text = [e for e in ordered if not e.is_text]
    if matting is not None and opts.get("refine_alpha"):
        for element in non_text:
            if element.alpha is None:
                refine_element_alpha(flat, element, matting, cfg)
    layers: list = []
    for z_index, element in enumerate(non_text):
        mask = np.asarray(element.mask, bool)
        higher = [o for o in elements if o.z > element.z
                  and bool(np.logical_and(o.mask, mask).any())]
        lower = [u for u in non_text if u.z < element.z
                 and bool(np.logical_and(u.mask, mask).any())]

        rgb = flat.copy()
        alpha = (np.round(np.clip(np.asarray(element.alpha, np.float64), 0, 1) * 255)
                 .astype(np.uint8) if element.alpha is not None
                 else mask.astype(np.uint8) * 255)
        fills: list = []
        meta: dict = {}

        if higher:
            occluder_map, occ_order = _direct_occluder_map(element, higher)
            occluded = occluder_map > 0
            visible = mask & ~occluded
            # Split by routing class so the injected router can keep text-shaped
            # holes away from generative backends (glyph residue).
            for is_text_class in (False, True):
                class_mask = np.zeros((h, w), bool)
                class_occluders = []
                for number, occluder in enumerate(occ_order, start=1):
                    if occluder.is_text != is_text_class:
                        continue
                    sub = occluder_map == number
                    if not sub.any():
                        continue
                    class_mask |= sub
                    class_occluders.append(occluder.id)
                    fills.append(OcclusionFill(occluder_id=occluder.id,
                                               area=int(np.count_nonzero(sub)),
                                               bbox=_tight_bbox(sub),
                                               text_occluder=occluder.is_text))
                if not class_mask.any():
                    continue
                # Fringe ring: widen the write region past anti-aliased edges, but
                # ONLY inside this layer's own mask — a pixel outside mask(L) was
                # never L's to begin with, so it is never synthesized into L.
                write = _dilate_px(class_mask, dilate) & mask if dilate else class_mask
                isolated = _fill_region(
                    flat, rgb, mask, visible, write, inpaint, accepts_meta,
                    {"under_id": element.id, "under_kind": element.kind,
                     "occluder_ids": class_occluders, "text_occluder": is_text_class},
                    opts)
                if not isolated:
                    meta["low_context_fill"] = True
        # Visible pixels are flat-image originals by construction (rgb started as a
        # copy of flat and only `write ⊆ mask` pixels were rewritten).
        rgba = np.dstack([rgb, alpha])
        rgba[:, :, :3][alpha == 0] = 0   # deterministic padding outside the footprint
        layers.append(ScenePeelLayer(
            id=element.id, rgba=rgba, bbox=_tight_bbox(mask), z_index=z_index,
            kind=element.kind,
            occluded_by=[o.id for o in sorted(higher, key=lambda e: -e.z)],
            occludes=[u.id for u in sorted(lower, key=lambda e: -e.z)],
            fills=fills, meta=meta))

    # Background plate: occluded by definition under every element footprint.
    background_fills: list = []
    bg_meta = {}
    if background is not None:
        plate = peel_decompose._to_rgb(background)
        if plate.shape != flat.shape:
            raise ValueError(f"background plate shape {plate.shape} != {flat.shape}")
        bg_meta["background"] = "provided"
    else:
        union = np.zeros((h, w), bool)
        for element in elements:
            union |= np.asarray(element.mask, bool)
        plate = flat.copy()
        if union.any():
            pseudo = SceneElement(id="background", mask=np.ones((h, w), bool), z=-1.0)
            occluder_map, occ_order = _direct_occluder_map(pseudo, list(elements))
            for is_text_class in (False, True):
                class_mask = np.zeros((h, w), bool)
                class_occluders = []
                for number, occluder in enumerate(occ_order, start=1):
                    sub = occluder_map == number
                    if occluder.is_text != is_text_class or not sub.any():
                        continue
                    class_mask |= sub
                    class_occluders.append(occluder.id)
                    background_fills.append(OcclusionFill(
                        occluder_id=occluder.id, area=int(np.count_nonzero(sub)),
                        bbox=_tight_bbox(sub), text_occluder=occluder.is_text))
                if not class_mask.any():
                    continue
                write = _dilate_px(class_mask, dilate)
                _fill_region(flat, plate, np.ones((h, w), bool), ~_dilate_px(union, dilate),
                             write, inpaint, accepts_meta,
                             {"under_id": "background", "under_kind": "background",
                              "occluder_ids": class_occluders,
                              "text_occluder": is_text_class}, opts)
        bg_meta["background"] = "inpainted"

    result = ScenePeelResult(layers=layers, background=plate, canvas=canvas,
                             overlap=report, background_fills=background_fills,
                             meta=bg_meta)
    text_union = np.zeros((h, w), bool)
    for element in elements:
        if element.is_text:
            text_union |= np.asarray(element.mask, bool)
    result.meta["recomposite"] = _recomposite_check(result, flat, text_union)
    return result


def _recomposite_check(result: ScenePeelResult, flat, text_union=None) -> dict:
    """Composite background + layers back-to-front and diff against the input.

    Text occluders are never emitted as peel layers (the pipeline renders native text
    on top), so pixels under text footprints are excluded from the diff and reported
    separately as ``text_excluded_px``.
    """
    _, np, _ = _deps()
    plate = np.asarray(result.background, np.float64)
    for layer in sorted(result.layers, key=lambda l: l.z_index):
        rgba = np.asarray(layer.rgba, np.float64)
        a = rgba[:, :, 3:4] / 255.0
        plate = rgba[:, :, :3] * a + plate * (1.0 - a)
    diff = np.abs(plate.round() - np.asarray(flat, np.float64))
    excluded = 0
    if text_union is not None and text_union.any():
        excluded = int(np.count_nonzero(text_union))
        diff[text_union] = 0.0
    return {"max_abs_diff": int(diff.max()), "mean_abs_diff": round(float(diff.mean()), 4),
            "exact": bool(diff.max() == 0), "text_excluded_px": excluded}


# ── artifacts ──────────────────────────────────────────────────────────────────────

def _fill_entry(fill: OcclusionFill) -> dict:
    return {"occluder": fill.occluder_id, "area": fill.area, "bbox": fill.bbox,
            "text_occluder": fill.text_occluder}


def write_outputs(result: ScenePeelResult, out_dir: str) -> dict:
    """Write per-layer RGBA PNGs, background.png and peel_scene_manifest.json."""
    _, np, Image = _deps()
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for layer in sorted(result.layers, key=lambda l: l.z_index):
        name = f"layer_{layer.z_index:02d}_{layer.id}.png"
        Image.fromarray(np.asarray(layer.rgba, np.uint8)).save(os.path.join(out_dir, name))
        entries.append({
            "file": name, "id": layer.id, "z": layer.z_index + 1,   # background is z=0
            "kind": layer.kind, "bbox": layer.bbox,
            "occluded_by": layer.occluded_by, "occludes": layer.occludes,
            "fills": [_fill_entry(f) for f in layer.fills],
            "filled_area": layer.filled_area, "meta": layer.meta,
        })
    if result.background is not None:
        Image.fromarray(np.asarray(result.background, np.uint8)).save(
            os.path.join(out_dir, "background.png"))
    manifest = {
        "version": 1, "mode": "scene", "canvas": result.canvas,
        "skipped": result.skipped, "skip_reason": result.skip_reason,
        "overlap": result.overlap,
        "background": ({"file": "background.png", "z": 0,
                        "fills": [_fill_entry(f) for f in result.background_fills]}
                       if result.background is not None else None),
        "layers": entries,
        "meta": result.meta,
    }
    with open(os.path.join(out_dir, "peel_scene_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_pipeline_layers(result: ScenePeelResult, run_dir: str,
                          subdir: str = "peel_layers") -> list:
    """Publish complete layers in the decomposed-layer shape merge_layers consumes
    (schema.QwenLayer: {"id","png","box","kind_hint"}, back-to-front).  The extra
    ``fused_id`` key records the exact fused element each layer completes — merge's
    IoU match will re-find it (masks are identical), the key is for audit.
    """
    _, np, Image = _deps()
    out_dir = os.path.join(run_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    published = []
    for layer in sorted(result.layers, key=lambda l: l.z_index):
        if layer.bbox["w"] <= 0 or layer.bbox["h"] <= 0:
            continue
        rel = os.path.join(subdir, f"P{len(published)}.png")
        Image.fromarray(np.asarray(layer.rgba, np.uint8)).save(os.path.join(run_dir, rel))
        published.append({"id": f"P{len(published)}", "png": rel,
                          "box": dict(layer.bbox), "kind_hint": layer.kind,
                          "fused_id": layer.id})
    return published
