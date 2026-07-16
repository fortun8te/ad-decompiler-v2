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
    "text_hole_dilate_px": 4,    # extra bleed for text-shaped holes (glyph AA / residue)
    "context_pad_px": 24,        # crop padding around a layer for inpaint context
    "min_context_frac": 0.05,    # below this visible/total ratio, isolation is hopeless
    "text_occluders": "box",     # box | off — see docs (ghost-text vs box overfill)
    "refine_alpha": False,       # matting-refine cutout EDGES (needs a matting callable)
    "refine_band_px": 3,         # width of the edge band the matting may adjust
    # ── detection-granularity guard (element eligibility for the gate) ──
    "require_eligible": True,    # gate needs BOTH pair members to be solid elements
    "min_cc_frac": 0.85,         # largest connected component ≥ this fraction of mask
    "max_hole_frac": 0.25,       # interior holes ≤ this fraction of (mask + holes)
    "max_components": 24,        # > this many CCs → fragmented residual (swiss-cheese)
    # ── fill-quality knobs ──
    "context_shadow_px": 12,     # blind this band around the hole from the inpaint
                                 # context — occluder drop shadows / AA halos live just
                                 # outside the detection mask and smear the fill
    # flat-fill fast path (solid-color holes, e.g. cards/plates under products):
    # sample a ring BEYOND the shadow band; when ≥ flat_fill_inlier_frac of ring pixels
    # sit within ±flat_fill_tol of the ring median, the surface is flat — fill with the
    # inlier median (crisper than any inpainter, robust to shadow/edge contamination).
    "flat_fill_tol": 0.0,        # per-pixel max-channel deviation; 0 = off (module
                                 # default; config.yaml enables it for pipeline runs)
    "flat_fill_inlier_frac": 0.60,
    "flat_fill_ring_px": 16,     # ring width sampled beyond the shadow band
    "flat_fill_min_px": 40,      # minimum inlier samples required to trust the ring
    # Thin-rim guard (016): refuse flat-fill when almost none of the under-layer is
    # still visible — a thin flat margin falsely looks solid and paints the whole
    # footprint one colour.  Area/frac caps are secondary; 0 disables each.
    # Solid cards under large products (002) keep visible_frac high → still flat-fill.
    "flat_fill_min_visible_frac": 0.12,
    "flat_fill_max_area": 0,     # 0 = no area cap; set >0 to refuse oversized writes
    "flat_fill_max_frac": 0.0,   # 0 = no frac cap; e.g. 0.95 refuses near-total coverage
    # Background plate: LOCAL solid fill per CC when the ring agrees — only on
    # flat-plate archetypes by default (002 orange chrome). Lifestyle keeps LaMa.
    "flat_fill_allow_background": False,
    "flat_fill_bg_max_area": 0,      # 0 = no cap when background flat is allowed
    "flat_fill_bg_max_frac": 0.0,
    # Text holes on flat cards: solid-fill when the ring agrees (LaMa left glyph haze).
    "flat_fill_text": True,
    # Split large write masks into connected components before fill.
    "fill_cc_split": True,
    "fill_cc_min_area": 64,
    "inpaint_feather_px": 0,     # soft edge on generative write-back (config enables)
    # Large occluders batched into ONE LaMa call with context isolation invent haze
    # (benchmark 016 E000←E013). Split element-class holes at this area threshold.
    "per_occluder_area": 6000,   # element-class hole ≥ this → own inpaint call
    # Photo under-layers with a hole bigger than this fraction of their mask cannot
    # be recovered by LaMa under isolation.  See large_photo_hole for what we do
    # instead of generative fill (bake keeps original pixels; abandon punches alpha).
    "abandon_hole_frac": 0.15,
    # Absolute px gate for photo holes — used when Flux is off or the hole is past
    # flux_max_hole_px (bake/abandon rather than LaMa haze).
    "abandon_photo_min_area": 12000,
    # Soft LaMa ceiling when Flux is unavailable. With allow_flux, Flux covers the
    # mid/large band and this only kicks in above flux_max_hole_px.
    "max_generative_photo_hole_px": 8000,
    # Prefer Flux Fill for photo under-layer holes (VRAM-cleared at peel boundary).
    # Flat plates still solid-fill upstream; text stays Telea/LaMa.
    "allow_flux": True,
    "flux_min_hole_px": 4000,     # below → Telea/LaMa (Flux overhead not worth it)
    "flux_max_hole_px": 220000,   # above → bake (Flux hallucinates on giant masks)
    # bake = leave flat pixels; abandon = transparent. Only for holes Flux won't touch.
    "large_photo_hole": "bake",
    # Extra context for large photo fills (gives Flux/LaMa more of the under-layer).
    "context_pad_large_px": 64,
    "photo_context_shadow_px": 4,  # less blinding than context_shadow_px on photos
    # Regional peel ladder: after LaMa, if a solid ring candidate scores cleaner,
    # fail closed to that flat fill instead of keeping generative smear.
    "fail_closed_to_flat": True,
    "fail_closed_residue": 8.0,
    # Peel objects only: OCR text / logos punch plates, never photo cutouts
    # (printed ink / wordmarks stay on the product raster).
    "punch_text_into_photos": False,
    "punch_artwork_into_photos": False,
}

#: Archetypes whose plates are Codia-style solid/banded chrome — peel holes prefer
#: analytic flat fill; generative backends are last resort and never Flux at peel.
_FLAT_PLATE_ARCHETYPES = frozenset({
    "product_on_flat", "social_screenshot", "comparison_grid",
})

#: Under-kinds that are plate chrome (safe for large solid fills when the ring agrees).
_PLATE_KINDS = frozenset({
    "shape", "card", "panel", "button", "background", "plate", "element",
})

#: Brand lettering / logos — artwork, not peelable objects that activate the gate.
_ARTWORK_ROLES = frozenset({"logo", "wordmark", "brand", "logotype"})
_ARTWORK_KINDS = frozenset({"logo", "wordmark", "brand"})


def _options(cfg: Optional[dict]) -> dict:
    opts = {**SCENE_DEFAULTS, **peel_decompose._options(cfg)}
    # peel_decompose._options already merged cfg["peel"] over its own DEFAULTS but not
    # over SCENE_DEFAULTS; re-apply the user block so scene keys win too.
    for key, value in ((cfg or {}).get("peel") or {}).items():
        if key in SCENE_DEFAULTS:
            opts[key] = value
    return opts


def _archetype(cfg: Optional[dict]) -> str:
    return str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()


def _is_flat_plate_archetype(cfg: Optional[dict] = None, archetype: str = "") -> bool:
    # Format capability wins when present — batch multi-format runs should not need a
    # new named archetype just to prefer analytic plate fill.
    from src import format_readiness
    fmt = format_readiness.format_from_cfg(cfg)
    if fmt.get("capabilities"):
        return format_readiness.prefers_solid_flat(cfg)
    name = (archetype or _archetype(cfg)).lower()
    extra = set((((cfg or {}).get("peel") or {}).get("flat_plate_archetypes")) or ())
    return name in (_FLAT_PLATE_ARCHETYPES | {str(x).lower() for x in extra})


def _element_role(element: SceneElement) -> str:
    return str((element.meta or {}).get("role") or "").lower()


def is_artwork_element(element: SceneElement) -> bool:
    """Logo / wordmark / brand lettering — artwork, not a peelable object."""
    if element.is_text:
        return False
    kind = str(element.kind or "").lower()
    role = _element_role(element)
    return kind in _ARTWORK_KINDS or role in _ARTWORK_ROLES


def is_peel_object(element: SceneElement) -> bool:
    """True when ``element`` is an object that may activate peel (not text/artwork)."""
    if element.is_text or is_artwork_element(element):
        return False
    kind = str(element.kind or "").lower()
    role = _element_role(element)
    if kind in ("background", "plate"):
        return False
    if role in _ARTWORK_ROLES:
        return False
    return True


def resolve_peel_fill_policy(cfg: Optional[dict] = None, *, under_kind: str = "",
                             text_occluder: bool = False) -> dict:
    """Archetype-aware peel hole policy: when to prefer flat vs LaMa (never Flux).

    Returns ``{"prefer_flat": bool, "allow_background_flat": bool,
    "backend": "flat"|"lama"|"abandon_photo", "archetype": str}``.
    """
    opts = _options(cfg)
    archetype = _archetype(cfg)
    flat_scene = _is_flat_plate_archetype(cfg, archetype)
    kind = str(under_kind or "").lower()
    photo = kind in _PHOTO_KINDS
    plate = kind in _PLATE_KINDS
    allow_bg = bool(opts.get("flat_fill_allow_background")) or flat_scene
    # Text holes on flat chrome: prefer solid (LaMa left glyph haze on 002/016).
    # On textured/photo unders, keep LaMa (or skip via punch_text_into_photos).
    if text_occluder:
        prefer = bool(opts.get("flat_fill_text", True)) and (flat_scene or plate) and not photo
        return {"prefer_flat": prefer, "allow_background_flat": allow_bg,
                "backend": "flat" if prefer else "lama",
                "archetype": archetype, "flat_scene": flat_scene}
    if photo:
        # Lifestyle / photo plates: never solid-paint. Large holes bake/abandon
        # upstream; small holes may still hit LaMa/Telea.
        return {"prefer_flat": False, "allow_background_flat": False,
                "backend": "lama", "archetype": archetype, "flat_scene": flat_scene}
    # Shape/card/panel/background on flat archetypes → Codia solid plates.
    if flat_scene and plate:
        return {"prefer_flat": True, "allow_background_flat": True,
                "backend": "flat", "archetype": archetype, "flat_scene": flat_scene}
    if plate:
        return {"prefer_flat": True, "allow_background_flat": allow_bg,
                "backend": "flat", "archetype": archetype, "flat_scene": flat_scene}
    return {"prefer_flat": flat_scene, "allow_background_flat": allow_bg,
            "backend": "lama" if not flat_scene else "flat",
            "archetype": archetype, "flat_scene": flat_scene}


def peel_inpaint_mode(cfg: Optional[dict] = None, meta: Optional[dict] = None) -> str:
    """Backend mode for the pipeline peel adapter.

    Flat/analytic fills happen upstream in ``_fill_region``. Text/tiny holes stay
    Telea in the adapter. Photo holes in the Flux band use Flux Fill after SAM is
    unloaded at the peel VRAM boundary; everything else stays LaMa.
    """
    meta = meta or {}
    opts = _options(cfg)
    if meta.get("text_occluder"):
        return "lama"
    under = str(meta.get("under_kind") or "")
    hole_px = int(meta.get("hole_px") or 0)
    if bool(opts.get("allow_flux")) and _is_photo_kind(under) and hole_px > 0:
        lo = int(opts.get("flux_min_hole_px") or 0)
        hi = int(opts.get("flux_max_hole_px") or 0) or 10**9
        if lo <= hole_px <= hi:
            return "flux_comfy"
    return "lama"


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


def _band_of(element: "SceneElement") -> int:
    """Semantic z band for one element: the MAX over its ``kind`` and its detected
    ``role`` (meta) — fusion often reports kind="photo-fragment" (band 5) for what the
    detector role-tags as a product/person cutout (band 20); the role is the stronger
    stacking signal (a product cutout rides ON TOP of the card it sits on, a
    photo-fragment panel sits under it)."""
    fallback = 40 if element.is_text else 10
    bands = [b for b in (
        _KIND_BAND.get(str(element.kind).lower()),
        _KIND_BAND.get(str((element.meta or {}).get("role") or "").lower()),
    ) if b is not None]
    return max(bands) if bands else fallback


def derive_z_order(elements: list) -> list:
    """Assign unique integer ``z`` (higher = front) to SceneElements missing one.

    Heuristic mirror of reconstruct's ownership priority: semantic band first (max of
    kind band and role band, see ``_band_of``), then a strict-containment boost (a mask
    sitting mostly inside another is on top of it), then smaller-area-in-front.
    Elements that already carry distinct ``z`` values are left untouched.
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
        _band_of(e),
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


# ── overlap gate + detection-granularity guard ──────────────────────────────────────

def _tight_bbox(mask) -> dict:
    return peel_decompose._tight_bbox(mask)


def mask_integrity(mask) -> dict:
    """Fragmentation metrics for one full-canvas bool mask (computed on its tight crop).

    * ``cc_frac`` — largest connected component / total mask area.  A residual
      "photo-fragment" mask (the plate minus everything else, swiss-cheese) scores
      ~0.5 with hundreds of components; a genuine cutout scores ~1.0 with one.
    * ``hole_frac`` — interior holes (background regions NOT reachable from the crop
      border) / (mask + holes).  Perforated masks make hopeless inpaint context.
    """
    cv2, np, _ = _deps()
    m = np.asarray(mask, bool)
    area = int(np.count_nonzero(m))
    if area == 0:
        return {"area": 0, "components": 0, "cc_frac": 0.0, "hole_frac": 1.0}
    box = _tight_bbox(m)
    crop = m[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]].astype(np.uint8)
    ncc, labels = cv2.connectedComponents(crop)
    counts = np.bincount(labels.ravel(), minlength=ncc)
    largest = int(counts[1:].max()) if ncc > 1 else 0
    ninv, inv_labels = cv2.connectedComponents((crop == 0).astype(np.uint8))
    border = np.unique(np.concatenate([inv_labels[0], inv_labels[-1],
                                       inv_labels[:, 0], inv_labels[:, -1]]))
    inv_counts = np.bincount(inv_labels.ravel(), minlength=ninv)
    holes = int(inv_counts[1:].sum() - sum(inv_counts[b] for b in border if b != 0))
    return {"area": area, "components": int(ncc - 1),
            "cc_frac": round(largest / max(1, area), 4),
            "hole_frac": round(holes / max(1, area + holes), 4)}


#: Kinds whose visible pixels are photographic / textured — never flat-fill, and
#: oversized holes under them are abandoned (transparent) rather than LaMa-hazed.
_PHOTO_KINDS = frozenset({
    "photo", "photo-fragment", "person", "product", "image", "cutout",
})

#: Product cutouts that carry printed label ink — OCR/wordmarks must not punch these
#: (ghost-text still punches plates and overlay photo panels).
_PRODUCT_INK_KINDS = frozenset({"product", "photo-fragment", "cutout"})


def element_eligibility(element: SceneElement, cfg: Optional[dict] = None) -> dict:
    """Detection-granularity guard: is this element solid enough for peel to trust?

    Peel is only as good as the elements it is fed.  A residual/fragmented mask (e.g.
    the "photo panel minus persons minus product" leftovers fusion sometimes emits) can
    neither be a trustworthy occluder footprint nor provide clean inpaint context — a
    pair involving one must not switch peel on.  Text and background/plate kinds are
    never eligible pair members (text stays native; the plate is the single-plate
    path's job).  Logos / wordmarks may punch holes (``as_top``) but never activate
    peel by themselves and are not completed as under-layers.

    Roles differ: a perforated TOP is still a usable hole-punch footprint
    (``as_top=True``), but a perforated UNDER cannot provide clean fill context
    (``as_under=False``).  Fragmented masks fail both roles.  ``eligible`` remains
    ``as_top and as_under`` for backward-compatible summaries.
    """
    if element.is_text:
        return {"eligible": False, "as_top": False, "as_under": False, "reason": "text"}
    if str(element.kind).lower() in ("background", "plate"):
        return {"eligible": False, "as_top": False, "as_under": False,
                "reason": "background-plate"}
    if is_artwork_element(element):
        # Artwork punches plates when peel already runs, but never activates the gate
        # and never hosts a fill (brand lettering stays a cutout, not a completed plate).
        return {"eligible": False, "as_top": True, "as_under": False,
                "reason": "artwork-wordmark"}
    opts = _options(cfg)
    info = mask_integrity(element.mask)
    max_cc = int(opts.get("max_components") or 0)
    if max_cc > 0 and int(info["components"]) > max_cc:
        return {"eligible": False, "as_top": False, "as_under": False, "integrity": info,
                "reason": (f"fragmented-mask ({info['components']} components > "
                           f"{max_cc} max)")}
    if info["cc_frac"] < float(opts["min_cc_frac"]):
        return {"eligible": False, "as_top": False, "as_under": False, "integrity": info,
                "reason": (f"fragmented-mask (largest component {info['cc_frac']:.0%} "
                           f"of {info['components']} pieces < "
                           f"{float(opts['min_cc_frac']):.0%})")}
    if info["hole_frac"] > float(opts["max_hole_frac"]):
        # Hollow / ring-like occluders still punch a useful hole; they just cannot
        # host a fill.  Keep as_top so solid unders under perforated tops still peel.
        return {"eligible": False, "as_top": True, "as_under": False, "integrity": info,
                "reason": (f"perforated-mask (interior holes {info['hole_frac']:.0%} > "
                           f"{float(opts['max_hole_frac']):.0%})")}
    return {"eligible": True, "as_top": True, "as_under": True,
            "reason": "ok", "integrity": info}


def overlap_report(elements: list, cfg: Optional[dict] = None) -> dict:
    """Pairwise occlusion census + the go/no-go gate for running peel at all.

    A pair qualifies when the intersection is at least ``peel.min_overlap_area`` px AND
    at least ``peel.min_overlap_frac`` of the smaller element's area.  Peel is *needed*
    only when some qualifying pair covers a NON-TEXT under-layer with a peel *object*
    on top (product / person / icon / … — not a logo/wordmark).  Text stays native;
    artwork may punch holes once peel runs for a real object pair, but never activates
    the stage alone.  With ``peel.require_eligible`` (default) the under must pass
    ``as_under`` and the top ``as_top``.  Qualifying pairs blocked only by eligibility
    are counted in ``blocked_qualifying``.
    """
    _, np, _ = _deps()
    opts = _options(cfg)
    min_area = int(opts["min_overlap_area"])
    min_frac = float(opts["min_overlap_frac"])
    require_eligible = bool(opts.get("require_eligible", True))
    boxes = {e.id: _tight_bbox(e.mask) for e in elements}
    areas = {e.id: int(np.count_nonzero(e.mask)) for e in elements}
    by_id = {e.id: e for e in elements}
    eligibility = {e.id: element_eligibility(e, cfg) for e in elements if not e.is_text}
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
            top_ok = bool(eligibility.get(top.id, {}).get("as_top", False))
            under_ok = bool(eligibility.get(under.id, {}).get("as_under", False))
            object_top = is_peel_object(top)
            eligible = (not top.is_text and not under.is_text and top_ok and under_ok
                        and object_top)
            pairs.append({"top": top.id, "under": under.id, "area": inter,
                          "frac": round(frac, 5), "top_is_text": top.is_text,
                          "under_is_text": under.is_text,
                          "top_is_artwork": is_artwork_element(top),
                          "top_is_object": object_top,
                          "qualifies": qualifies, "eligible": eligible})
    element_pairs = [p for p in pairs if p["qualifies"]
                     and not p["top_is_text"] and not p["under_is_text"]]
    activating = [p for p in element_pairs if p["eligible"] or not require_eligible]
    if not require_eligible:
        activating = [p for p in activating if is_peel_object(by_id[p["top"]])]
    blocked = [p for p in element_pairs if not p["eligible"]]
    return {"pairs": pairs, "needed": bool(activating),
            "blocked_qualifying": len(blocked),
            "eligibility": {eid: {k: v for k, v in e.items() if k != "integrity"}
                            for eid, e in eligibility.items()},
            "thresholds": {"min_overlap_area": min_area, "min_overlap_frac": min_frac,
                           "min_cc_frac": float(opts["min_cc_frac"]),
                           "max_hole_frac": float(opts["max_hole_frac"]),
                           "max_components": int(opts.get("max_components") or 0),
                           "require_eligible": require_eligible}}


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


def _is_photo_kind(kind) -> bool:
    return str(kind or "").lower() in _PHOTO_KINDS


def _is_plate_kind(kind) -> bool:
    return str(kind or "").lower() in _PLATE_KINDS


def _may_punch_into(occluder: SceneElement, under_kind: str, opts: dict) -> bool:
    """Whether ``occluder`` should punch a hole into an under-layer of ``under_kind``.

    Peel objects only: OCR text and logo/wordmark artwork still punch plates and
    overlay photo panels (ghost-text invariant), but never product cutouts —
    printed label ink stays on the product raster.
    """
    kind = str(under_kind or "").lower()
    if occluder.is_text:
        if kind in _PRODUCT_INK_KINDS and not opts.get("punch_text_into_photos"):
            return False
        return True
    if is_artwork_element(occluder):
        if kind in _PRODUCT_INK_KINDS and not opts.get("punch_artwork_into_photos"):
            return False
        return True
    return True


def _flat_fill_allowed(write, element_mask, visible, meta, opts,
                       cfg=None) -> bool:
    """Whether the solid-median fast path may run for this hole.

    Photo kinds are always denied.  Background / text holes are allowed when policy
    + caps say so (flat chrome ads).  Thin-rim guard (`flat_fill_min_visible_frac`)
    blocks the 016 failure mode where a tiny flat margin painted a whole plate.
    """
    kind = str(meta.get("under_kind") or "").lower()
    if _is_photo_kind(kind):
        return False
    write_area = int(write.sum()) if hasattr(write, "sum") else 0
    if write_area <= 0:
        return False
    policy = resolve_peel_fill_policy(
        cfg, under_kind=kind, text_occluder=bool(meta.get("text_occluder")))
    if meta.get("text_occluder"):
        if not bool(opts.get("flat_fill_text", True)):
            return False
        # Text solid-fill only on plate chrome (card/shape/bg), never photos.
        if kind not in _PLATE_KINDS and kind not in ("background", "plate"):
            return False
    if kind in ("background", "plate"):
        if not (policy.get("allow_background_flat")
                or bool(opts.get("flat_fill_allow_background"))):
            return False
        max_area = int(opts.get("flat_fill_bg_max_area")
                       or opts.get("flat_fill_max_area") or 0)
        max_frac = float(opts.get("flat_fill_bg_max_frac")
                         or opts.get("flat_fill_max_frac") or 0.0)
    else:
        max_area = int(opts.get("flat_fill_max_area") or 0)
        max_frac = float(opts.get("flat_fill_max_frac") or 0.0)
    min_vis = float(opts.get("flat_fill_min_visible_frac") or 0.0)
    # Background plate: visible ≈ everything outside the union — skip thin-rim.
    if min_vis > 0 and kind not in ("background", "plate"):
        mask_area = max(1, int(element_mask.sum()))
        vis_frac = float(int(visible.sum()) / mask_area) if hasattr(visible, "sum") else 0.0
        if vis_frac < min_vis:
            return False
    if max_area > 0 and write_area > max_area:
        return False
    if max_frac > 0:
        mask_area = max(1, int(element_mask.sum()))
        if (write_area / mask_area) > max_frac:
            return False
    return True


def _photo_hole_decision(write, element_mask, meta, opts) -> Optional[str]:
    """Decide how to handle a photo under-layer hole that generative fill can't recover.

    Returns:
        ``None`` — try Telea/LaMa/Flux (size is in a recoverable band).
        ``"bake"`` — leave original flat pixels (no fill write).
        ``"abandon"`` — zero alpha in the hole.
    """
    if meta.get("text_occluder"):
        return None
    if not _is_photo_kind(meta.get("under_kind")):
        return None
    write_area = int(write.sum()) if hasattr(write, "sum") else 0
    if write_area <= 0:
        return None
    mask_area = max(1, int(element_mask.sum()))
    frac = float(opts.get("abandon_hole_frac") or 0.0)
    min_area = int(opts.get("abandon_photo_min_area") or 0)
    allow_flux = bool(opts.get("allow_flux"))
    flux_max = int(opts.get("flux_max_hole_px") or 0)
    # Flux band: let the adapter try real generative fill (VRAM-cleared).
    if allow_flux and flux_max > 0 and write_area <= flux_max:
        return None
    max_gen = int(opts.get("max_generative_photo_hole_px") or 0)
    oversized = (
        (frac > 0 and (write_area / mask_area) >= frac)
        or (min_area > 0 and write_area >= min_area)
        or (max_gen > 0 and write_area >= max_gen)
        or (allow_flux and flux_max > 0 and write_area > flux_max)
    )
    if not oversized:
        return None
    mode = str(opts.get("large_photo_hole") or "bake").strip().lower()
    return "abandon" if mode == "abandon" else "bake"


def _should_abandon_hole(write, element_mask, meta, opts) -> bool:
    """Back-compat: True only when the photo-hole policy chooses transparent abandon."""
    return _photo_hole_decision(write, element_mask, meta, opts) == "abandon"


def _sample_flat_color(crop, write_crop, visible_crop, opts, flat_tol: float):
    """Return uint8 RGB median when the ring beyond the hole is flat, else None."""
    _, np, _ = _deps()
    shadow_px = max(0, int(opts.get("context_shadow_px") or 0))
    inner = _dilate_px(write_crop, shadow_px) if shadow_px else write_crop
    ring = _dilate_px(write_crop, shadow_px + int(opts["flat_fill_ring_px"])) \
           & ~inner & visible_crop
    samples = crop[ring].astype(np.float64)
    if samples.shape[0] < int(opts["flat_fill_min_px"]):
        return None
    med = np.median(samples, axis=0)
    inlier = np.abs(samples - med).max(axis=1) <= flat_tol
    if (float(inlier.mean()) < float(opts["flat_fill_inlier_frac"])
            or int(inlier.sum()) < int(opts["flat_fill_min_px"])):
        return None
    return np.round(np.median(samples[inlier], axis=0)).astype(np.uint8)


def _feather_copy(region, write_crop, filled, feather_px: int):
    """Copy `filled` into `region` on `write_crop`, with a soft edge ring."""
    _, np, _ = _deps()
    if feather_px <= 0 or not write_crop.any() or write_crop.all():
        region[write_crop] = filled[write_crop]
        return
    core = write_crop & ~_dilate_px(~write_crop, feather_px)
    region[core] = filled[core]
    edge = write_crop & ~core
    if not edge.any():
        return
    weight = np.zeros(write_crop.shape, np.float64)
    grown = ~write_crop
    for step in range(1, feather_px + 1):
        nxt = _dilate_px(grown, 1)
        band = nxt & ~grown & write_crop
        weight[band] = step / float(feather_px + 1)
        grown = nxt
    weight[core] = 1.0
    w = weight[edge][:, None]
    src = region[edge].astype(np.float64)
    dst = filled[edge].astype(np.float64)
    region[edge] = np.clip(np.round(src * (1.0 - w) + dst * w), 0, 255).astype(np.uint8)


def _iter_write_regions(write, opts):
    """Yield connected components of `write` when `fill_cc_split` is on."""
    cv2, np, _ = _deps()
    write = np.asarray(write, bool)
    if not write.any():
        return
    if not bool(opts.get("fill_cc_split", True)):
        yield write
        return
    min_area = int(opts.get("fill_cc_min_area") or 0)
    ncc, labels = cv2.connectedComponents(write.astype(np.uint8), connectivity=8)
    if ncc <= 2:
        yield write
        return
    emitted = False
    for lab in range(1, ncc):
        cc = labels == lab
        if min_area > 0 and int(cc.sum()) < min_area:
            continue
        emitted = True
        yield cc
    if not emitted:
        yield write


def _fill_region(flat, layer_rgb, element_mask, visible, write, inpaint,
                 accepts_meta, meta, opts, cfg=None):
    """Inpaint `write` pixels into `layer_rgb` using ONLY `visible` as context.

    Context isolation + flat-fill / fail-closed-to-flat / feathered generative copy.
    Returns `{"isolated": bool, "backend": "solid" | "inpaint" | "abandoned" | "baked"}`.
    """
    _, np, _ = _deps()
    h, w = flat.shape[:2]
    write_area = int(np.count_nonzero(write))
    is_photo = _is_photo_kind(meta.get("under_kind"))
    pad = int(opts["context_pad_px"])
    large_pad = int(opts.get("context_pad_large_px") or 0)
    if is_photo and large_pad > pad and write_area >= int(opts.get("flux_min_hole_px") or 4000):
        pad = large_pad
    box = _tight_bbox(np.logical_or(element_mask, write))
    x0, y0, x1, y1 = _pad_bbox(box, pad, w, h)
    crop = flat[y0:y1, x0:x1]
    visible_crop = visible[y0:y1, x0:x1]
    write_crop = write[y0:y1, x0:x1]
    region = layer_rgb[y0:y1, x0:x1]

    mask_area = max(1, int(np.count_nonzero(element_mask)))
    isolated = (int(np.count_nonzero(visible)) / mask_area) >= float(opts["min_context_frac"])
    shadow_px = max(0, int(opts.get("context_shadow_px") or 0))
    if is_photo and opts.get("photo_context_shadow_px") is not None:
        shadow_px = max(0, int(opts.get("photo_context_shadow_px") or 0))

    photo_decision = _photo_hole_decision(write, element_mask, meta, opts)
    if photo_decision == "abandon":
        return {"isolated": isolated, "backend": "abandoned"}
    if photo_decision == "bake":
        # Leave ``layer_rgb`` untouched — only for holes past flux_max (or Flux off).
        return {"isolated": isolated, "backend": "baked"}

    flat_tol = float(opts.get("flat_fill_tol") or 0.0)
    solid_color = None
    if (flat_tol > 0 and isolated
            and _flat_fill_allowed(write, element_mask, visible, meta, opts, cfg=cfg)):
        solid_color = _sample_flat_color(
            crop, write_crop, visible_crop, opts, flat_tol)
        if solid_color is not None:
            region[write_crop] = solid_color
            return {"isolated": True, "backend": "solid"}

    if isolated:
        call_mask = np.logical_or(~visible_crop, write_crop)
        if shadow_px:
            blinded = call_mask | _dilate_px(write_crop, shadow_px)
            if not blinded.all():
                call_mask = blinded
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

    # Fail-closed: if the ring is geometrically flat and generative smear is
    # worse than solid, keep solid — even when area caps blocked the *primary*
    # flat path (those caps exist to avoid painting busy plates without evidence;
    # after a dirty LaMa pass the solid ring is the safer plate).
    if (bool(opts.get("fail_closed_to_flat", True)) and flat_tol > 0 and isolated
            and not _is_photo_kind(meta.get("under_kind"))):
        if solid_color is None:
            solid_color = _sample_flat_color(
                crop, write_crop, visible_crop, opts, flat_tol)
            if solid_color is None and int(opts.get("context_shadow_px") or 0) > 0:
                bare = dict(opts, context_shadow_px=0)
                solid_color = _sample_flat_color(
                    crop, write_crop, visible_crop, bare, flat_tol)
        if solid_color is not None and write_crop.any():
            gen = filled[write_crop].astype(np.float64)
            solid = solid_color.astype(np.float64)
            residue = float(np.mean(np.abs(gen - solid).max(axis=1)))
            if residue >= float(opts.get("fail_closed_residue") or 8.0):
                region[write_crop] = solid_color
                return {"isolated": True, "backend": "solid"}

    _feather_copy(region, write_crop, filled, int(opts.get("inpaint_feather_px") or 0))
    return {"isolated": isolated, "backend": "inpaint"}


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
        # Distinguish "nothing overlaps" from "overlaps exist but detection did not
        # surface solid distinct elements for them" (the granularity guard, §"When
        # peel runs" in docs/PEEL-DECOMPOSITION.md).
        if report.get("blocked_qualifying"):
            blocked_ids = {mid for p in report["pairs"]
                           if p["qualifies"] and not p["under_is_text"]
                           and not p.get("top_is_text") and not p["eligible"]
                           for mid in (p["top"], p["under"])}
            reasons = {eid: e["reason"]
                       for eid, e in (report.get("eligibility") or {}).items()
                       if not e["eligible"] and eid in blocked_ids}
            reason = ("no-eligible-overlap: " + "; ".join(
                f"{eid}: {why}" for eid, why in sorted(reasons.items()))
                if reasons else "no-eligible-overlap")
        else:
            reason = "no-overlap"
        return ScenePeelResult(layers=[], background=None, canvas=canvas, skipped=True,
                               skip_reason=reason, overlap=report)

    if inpaint is None:
        inpaint = peel_decompose.opencv_inpaint
    accepts_meta = _accepts_meta(inpaint)
    dilate = int(opts["hole_dilate_px"])
    text_dilate = max(dilate, int(opts.get("text_hole_dilate_px") or dilate))

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
            # holes away from generative backends (glyph residue).  Text stays
            # batched (many small boxes).  Element-class holes at/above
            # per_occluder_area get their own fill — batching a 500k photo hole
            # with icon holes produced LaMa haze + rectangular smears (016).
            per_area = int(opts.get("per_occluder_area") or 0)
            for is_text_class in (False, True):
                jobs = []  # list of (write_mask, [occluder_ids])
                small_mask = np.zeros((h, w), bool)
                small_ids = []
                hole_px = text_dilate if is_text_class else dilate
                for number, occluder in enumerate(occ_order, start=1):
                    if occluder.is_text != is_text_class:
                        continue
                    if not _may_punch_into(occluder, element.kind, opts):
                        continue
                    sub = occluder_map == number
                    if not sub.any():
                        continue
                    fills.append(OcclusionFill(occluder_id=occluder.id,
                                               area=int(np.count_nonzero(sub)),
                                               bbox=_tight_bbox(sub),
                                               text_occluder=occluder.is_text))
                    write_sub = (_dilate_px(sub, hole_px) & mask) if hole_px else sub
                    if (not is_text_class and per_area > 0
                            and int(np.count_nonzero(write_sub)) >= per_area):
                        jobs.append((write_sub, [occluder.id]))
                    else:
                        small_mask |= write_sub
                        small_ids.append(occluder.id)
                if small_mask.any():
                    jobs.append((small_mask, small_ids))
                for write, occ_ids in jobs:
                    for write_cc in _iter_write_regions(write, opts):
                        if not write_cc.any():
                            continue
                        fill_info = _fill_region(
                            flat, rgb, mask, visible, write_cc, inpaint, accepts_meta,
                            {"under_id": element.id, "under_kind": element.kind,
                             "occluder_ids": occ_ids, "text_occluder": is_text_class,
                             "hole_px": int(np.count_nonzero(write_cc))},
                            opts, cfg=cfg)
                        if fill_info["backend"] == "abandoned":
                            alpha[write_cc] = 0
                            meta["abandoned_fill"] = True
                        elif fill_info["backend"] == "baked":
                            meta["baked_large_photo_hole"] = True
                        if not fill_info["isolated"]:
                            meta["low_context_fill"] = True
                        meta.setdefault("fill_backends", []).append(
                            {"text_occluder": is_text_class,
                             "backend": fill_info["backend"],
                             "area": int(np.count_nonzero(write_cc)),
                             "occluder_ids": list(occ_ids)})
        if meta.get("baked_large_photo_hole") or meta.get("abandoned_fill"):
            meta["peel_quality"] = "incomplete-photo"
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
            bg_visible = ~_dilate_px(union, dilate)
            for is_text_class in (False, True):
                class_mask = np.zeros((h, w), bool)
                class_occluders = []
                hole_px = text_dilate if is_text_class else dilate
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
                write = _dilate_px(class_mask, hole_px)
                for write_cc in _iter_write_regions(write, opts):
                    fill_info = _fill_region(
                        flat, plate, np.ones((h, w), bool), bg_visible,
                        write_cc, inpaint, accepts_meta,
                        {"under_id": "background", "under_kind": "background",
                         "occluder_ids": class_occluders,
                         "text_occluder": is_text_class,
                         "hole_px": int(np.count_nonzero(write_cc))}, opts, cfg=cfg)
                    bg_meta.setdefault("fill_backends", []).append(
                        {"text_occluder": is_text_class,
                         "backend": fill_info["backend"],
                         "area": int(np.count_nonzero(write_cc))})
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
