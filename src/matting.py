"""matting.py — production-grade cutout alpha refinement.

Turns SAM3 *binary* masks into production alpha for element cutouts:
edge feathering matched to image resolution, color decontamination (kills the
white/background fringe that caused the 002 "doubled contour"), hole/notch
cleanup, and guided-filter alpha matting on the boundary band.

Design constraints (RTX 5080, 16GB VRAM shared with a 6.3GB VLM):
  * 100% CPU + classic CV (cv2 / numpy / scipy). No model weights, no GPU lock.
  * A self-contained guided filter (He et al. 2010) replaces cv2.ximgproc,
    which is not built into this opencv wheel.

Public API
----------
    refine(image, mask, box=None, *, bg_color=None, config=None) -> RefinedCutout
    separability_score(image, mask, box=None) -> Separability
    save_cutout(refined, out_dir, elem_id) -> dict   # first-class alpha artifact

Shadow policy
-------------
Default policy (a): cut the product TIGHT and leave its soft shadow in the photo
plate. `refine()` detects a soft-shadow region attached to the object and, by
default, EXCLUDES it from the cutout alpha (so the shadow is never half-cut —
it stays whole, in the plate). When `config.shadow_mode == "separate"` (or the
caller passes `emit_shadow=True`), it additionally returns a separate
soft-shadow element (grayscale alpha ramp) meant to be placed UNDER the cutout.
`RefinedCutout.shadow` is None when no shadow is emitted.

The module never produces a half-cut shadow: a shadow is either fully in the
plate (default) or fully emitted as its own element.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import cv2
except Exception as _e:  # pragma: no cover
    cv2 = None
    _CV2_ERR = _e

from PIL import Image


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class MattingConfig:
    # Trimap band half-width as a fraction of the object's min bbox dimension.
    # Clamped to [band_px_min, band_px_max]. This is what "matches feather to
    # resolution" — a 4000px product gets a wider refine band than a 120px icon.
    band_frac: float = 0.010
    band_px_min: int = 2
    band_px_max: int = 40

    # Guided filter params operating on the boundary band.
    guided_radius_frac: float = 0.008   # of min dim
    guided_radius_min: int = 2
    guided_radius_max: int = 32
    guided_eps: float = 1e-4            # edge-preservation (smaller = sharper)

    # Final feather (Gaussian sigma) as fraction of min dim; the guided filter
    # already does most of the anti-aliasing, this is a tiny polish.
    feather_frac: float = 0.0015
    feather_min: float = 0.5
    feather_max: float = 3.0

    # Hole / island cleanup.
    fill_holes: bool = True
    min_island_frac: float = 0.002      # islands smaller than this * area dropped
    close_notch_px: int = 3             # morphological close to seal thin notches

    # Color decontamination.
    decontaminate: bool = True
    decon_clip: bool = True             # clip unmixed FG to [0,255]

    # Background-fringe suppression. The erode->dilate trimap grows the unknown
    # band OUTWARD past the original silhouette; on a bright/plain background the
    # guided filter then hands partial alpha to background pixels, re-creating the
    # white halo we are trying to kill (measured worst on H15/H16-class products
    # sitting on white). This pulls alpha back to 0 for band pixels whose colour
    # matches the local background, killing the fringe regardless of band width.
    suppress_bg_fringe: bool = True
    fringe_bg_dist_lo: float = 10.0     # <=: pixel IS background -> alpha forced 0
    fringe_bg_dist_hi: float = 42.0     # >=: pixel is clearly FG -> alpha kept
    fringe_only_outside: bool = True    # only suppress in the OUTWARD grown ring

    # Shadow policy.
    shadow_mode: str = "plate"          # "plate" (default a) | "separate" (b)
    shadow_max_sat: int = 45            # shadow = low saturation (0-255 HSV S)
    shadow_darker_than_bg: int = 8      # shadow V at least this much below bg V
    shadow_min_frac: float = 0.004      # ignore tiny shadow blobs

    # Separability thresholds (advisory; downstream decides).
    sep_cutout_threshold: float = 0.55  # >= keep as cutout; < stays in plate

    # Card snapping: photo cards / screenshot cards (straight edges + rounded
    # corners) get a GEOMETRIC mask (fitted rect + radius) instead of trusting
    # the segmentation boundary. Kills black/white fringe on card edges (H6, H16).
    card_snap: str = "auto"             # "auto" | "off" | "force"
    card_min_rectangularity: float = 0.90  # contour_area / minAreaRect_area
    card_max_angle: float = 6.0         # deg from axis-aligned to treat as a card
    card_max_radius_frac: float = 0.28  # corner radius cap as frac of min side


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class ShadowElement:
    """Separate soft-shadow element placed UNDER the cutout (policy b)."""
    alpha: np.ndarray                    # float 0..1, full crop size
    box: Tuple[int, int, int, int]       # x,y,w,h in source coords
    color: Tuple[int, int, int] = (0, 0, 0)
    opacity: float = 0.35


@dataclass
class RefinedCutout:
    rgba: np.ndarray                     # HxWx4 uint8 — the production cutout
    alpha: np.ndarray                    # HxW float 0..1 — refined alpha
    box: Tuple[int, int, int, int]       # x,y,w,h in source coords
    metrics: Dict[str, Any] = field(default_factory=dict)
    shadow: Optional[ShadowElement] = None

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.alpha.shape[:2]
        return (w, h)


@dataclass
class Separability:
    score: float                         # 0..1, higher = safer to cut out
    boundary_contrast: float
    bg_complexity: float
    hand_occlusion: float                # 0..1, fraction of boundary near skin
    recommend_cutout: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Roles whose silhouette must never be squared off into a rectangle, even when
# the (often bbox-fallback) mask looks rectangular. Products/people/logos here.
CARD_BLOCK_ROLES = frozenset({
    "product", "product-cluster", "person", "people", "face", "hand",
    "logo", "wordmark", "brand", "logotype", "icon", "avatar", "sticker",
    "subject", "foreground", "profile", "profile-photo",
})
# Roles that ARE rectangular photo/screenshot cards -> geometric snap wanted.
CARD_ROLES = frozenset({
    "image", "photo", "card", "screenshot", "frame", "media", "inset",
    "thumbnail", "photo-card", "image-card",
})


def _card_snap_allowed(mode: str, role: Optional[str], kind: Optional[str]) -> bool:
    """Gate the geometric card path by element role/kind.

    "force" always snaps; "off" never. In "auto": block products/people/logos
    (a rectangular-looking product mask is a segmentation artifact, not a card),
    allow known card roles, and allow unknown roles so pure-geometry detection
    can still catch cards the router didn't label.
    """
    if mode == "off":
        return False
    if mode == "force":
        return True
    r = (role or "").strip().lower()
    k = (kind or "").strip().lower()
    if r in CARD_BLOCK_ROLES:
        return False
    if "card" in k or "screenshot" in k:
        return True
    if r in CARD_ROLES or r in ("", "shape", "image", "background-card"):
        return True
    return False


def _require_cv2():
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"matting requires opencv (cv2); import failed: {_CV2_ERR}")


def _as_float_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] == 4 else m[..., 0]
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return np.clip(m, 0.0, 1.0)


def _crop_to_box(image: np.ndarray, mask: np.ndarray, box):
    """Return (img_crop RGB uint8, mask_crop float, box tuple).

    If box is None, use the full image (mask must already match). If box is
    given and mask matches the *full* image, both get cropped; if mask already
    matches the box, only the image is cropped.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = np.ascontiguousarray(img).astype(np.uint8)
    H, W = img.shape[:2]

    if box is None:
        return img, _as_float_mask(mask), (0, 0, W, H)

    if isinstance(box, dict):
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    else:
        x, y, w, h = (int(v) for v in box)
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)

    img_crop = img[y:y + h, x:x + w]
    m = np.asarray(mask)
    mh, mw = m.shape[:2]
    if (mh, mw) == (H, W):
        m = m[y:y + h, x:x + w]
    # else assume mask already box-sized
    mask_crop = _as_float_mask(m)
    if mask_crop.shape[:2] != img_crop.shape[:2]:
        mask_crop = cv2.resize(mask_crop, (img_crop.shape[1], img_crop.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    return img_crop, mask_crop, (x, y, w, h)


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter. guide/src are float32 in [0,1]. Grayscale guide."""
    radius = max(1, int(radius))
    d = 2 * radius + 1
    ksize = (d, d)
    I = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * I + mean_b


def _min_dim(mask_bool: np.ndarray) -> int:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return max(mask_bool.shape)
    return max(1, min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def _scaled(frac: float, ref: int, lo, hi) -> int:
    return int(np.clip(round(frac * ref), lo, hi))


# --------------------------------------------------------------------------- #
# Cleanup: holes, islands, notches
# --------------------------------------------------------------------------- #

def _cleanup_binary(m: np.ndarray, cfg: MattingConfig) -> np.ndarray:
    """m: bool. Fill enclosed holes, drop tiny islands, seal thin notches."""
    mu = (m > 0).astype(np.uint8)
    if mu.sum() == 0:
        return mu.astype(bool)

    if cfg.close_notch_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (cfg.close_notch_px, cfg.close_notch_px))
        mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, k)

    # Drop small disconnected islands.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mu, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(areas.max())
        keep = np.zeros_like(mu)
        thr = max(1, int(cfg.min_island_frac * biggest))
        for i, ar in enumerate(areas, start=1):
            if ar >= thr:
                keep[labels == i] = 1
        mu = keep

    # Fill fully-enclosed holes via flood fill from the border.
    if cfg.fill_holes:
        ff = mu.copy()
        h, w = ff.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask, (0, 0), 1)
        holes = (ff == 0)          # background not reachable from corner = hole
        mu[holes] = 1

    return mu.astype(bool)


# --------------------------------------------------------------------------- #
# Card detection + geometric rounded-rect snap (H6, H9, H12, H16)
# --------------------------------------------------------------------------- #

def detect_card(mask_bool: np.ndarray, cfg: Optional[MattingConfig] = None) -> Optional[Dict[str, Any]]:
    """Decide whether an element is a rectangular photo/screenshot card.

    A card fills almost its whole min-area rect (rectangularity high) and is
    near axis-aligned. Product tubes/bottles/sachets fill far less of their
    bounding rect, so they are correctly rejected.

    Returns a geometry dict {x0,y0,x1,y1,radius,angle,rectangularity} in *crop*
    coords, or None if not a card.
    """
    cfg = cfg or MattingConfig()
    mu = (mask_bool > 0).astype(np.uint8) * 255
    # De-jitter the boundary so the area-deficit radius estimate isn't inflated
    # by SAM's ragged 1px edge noise (median removes salt/pepper, keeps corners).
    if min(mu.shape) >= 7:
        mu = cv2.medianBlur(mu, 5)
    mu = (mu > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 64:
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
    if rw < 2 or rh < 2:
        return None
    rectangularity = area / (rw * rh)
    # Normalize angle to [0,90) then measure deviation from axis-aligned.
    a = abs(angle) % 90.0
    dev = min(a, 90.0 - a)
    if rectangularity < cfg.card_min_rectangularity or dev > cfg.card_max_angle:
        return None

    # Axis-aligned bbox of the mask = the card rectangle (near-axis-aligned).
    ys, xs = np.where(mu)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    W, H = x1 - x0, y1 - y0
    # Corner radius by direct inset measurement: for an axis-aligned rounded
    # rect the top edge's flat span starts r pixels in from the corner, so the
    # offset from the side where the row/column first reaches ~full extent = r.
    sub = mu[y0:y1, x0:x1].astype(bool)
    row_w = sub.sum(axis=1)
    col_h = sub.sum(axis=0)
    full_w = int(np.percentile(row_w, 90))
    full_h = int(np.percentile(col_h, 90))
    tol = 2
    def _first_reach(arr, full):
        idx = np.where(arr >= full - tol)[0]
        return int(idx[0]) if len(idx) else 0
    r_top = _first_reach(row_w, full_w)
    r_bot = _first_reach(row_w[::-1], full_w)
    r_left = _first_reach(col_h, full_h)
    r_right = _first_reach(col_h[::-1], full_h)
    ests = [e for e in (r_top, r_bot, r_left, r_right) if e > 0]
    r = float(np.median(ests)) if ests else 0.0
    r = float(np.clip(r, 0.0, cfg.card_max_radius_frac * min(W, H)))
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "radius": round(r, 1),
            "angle": round(float(angle), 2),
            "rectangularity": round(float(rectangularity), 3)}


def _rounded_rect_alpha(shape_hw, geom, supersample: int = 4) -> np.ndarray:
    """Anti-aliased axis-aligned rounded-rect alpha (float 0..1) for a card.

    Rendered at `supersample`x then area-averaged down, giving clean straight
    edges and smooth rounded corners with no segmentation jitter or fringe.
    """
    H, W = shape_hw
    s = supersample
    big = np.zeros((H * s, W * s), np.uint8)
    x0, y0, x1, y1 = geom["x0"] * s, geom["y0"] * s, geom["x1"] * s, geom["y1"] * s
    r = int(round(geom["radius"] * s))
    w = x1 - x0; h = y1 - y0
    r = max(0, min(r, min(w, h) // 2))
    if r <= 0:
        cv2.rectangle(big, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    else:
        cv2.rectangle(big, (x0 + r, y0), (x1 - 1 - r, y1 - 1), 255, -1)
        cv2.rectangle(big, (x0, y0 + r), (x1 - 1, y1 - 1 - r), 255, -1)
        for (ccx, ccy) in [(x0 + r, y0 + r), (x1 - 1 - r, y0 + r),
                           (x0 + r, y1 - 1 - r), (x1 - 1 - r, y1 - 1 - r)]:
            cv2.circle(big, (ccx, ccy), r, 255, -1)
    small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------- #
# Shadow detection
# --------------------------------------------------------------------------- #

def _detect_shadow(img_rgb: np.ndarray, fg: np.ndarray, cfg: MattingConfig):
    """Detect a soft-shadow region attached to the object.

    Returns (shadow_bool, bg_value_estimate). Shadow = low-saturation, darker
    than the surrounding background, spatially attached to the FG's dilated
    ring (typically below it). Returns an empty mask when none found.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)

    fg_u = fg.astype(np.uint8)
    # Ring band just outside the object where a contact shadow would live.
    ring_out = cv2.dilate(fg_u, np.ones((3, 3), np.uint8),
                          iterations=max(4, int(0.03 * _min_dim(fg)))).astype(bool)
    ring_out &= ~fg
    if ring_out.sum() == 0:
        return np.zeros_like(fg), None

    # Background value = median V of the outer frame (far from object).
    far = cv2.dilate(fg_u, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    outside = ~cv2.dilate(fg_u, np.ones((3, 3), np.uint8),
                          iterations=max(6, int(0.06 * _min_dim(fg)))).astype(bool)
    bg_pixels_V = V[outside] if outside.sum() > 50 else V[~far]
    bg_V = int(np.median(bg_pixels_V)) if bg_pixels_V.size else int(np.median(V))

    shadow = ring_out & (S < cfg.shadow_max_sat) & (V < bg_V - cfg.shadow_darker_than_bg)
    # Grow the seed to capture the full soft falloff, then keep only the part
    # that is still darker than bg (so we don't eat the whole plate).
    if shadow.sum() > 0:
        grown = cv2.dilate(shadow.astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=6).astype(bool)
        shadow = grown & (S < cfg.shadow_max_sat + 15) & (V < bg_V - 2) & (~fg)
        if shadow.sum() < cfg.shadow_min_frac * fg.sum():
            shadow = np.zeros_like(fg)
    return shadow, bg_V


# --------------------------------------------------------------------------- #
# Core refinement
# --------------------------------------------------------------------------- #

def refine(image, mask, box=None, *, bg_color=None, emit_shadow: Optional[bool] = None,
           element_role: Optional[str] = None, element_kind: Optional[str] = None,
           config: Optional[MattingConfig] = None) -> RefinedCutout:
    """Refine a binary/soft mask into a production alpha cutout.

    Parameters
    ----------
    image : HxWx3 (or 4/gray) uint8 RGB source. Either the full ad or the crop.
    mask  : binary or soft mask. May be full-image sized or box-sized.
    box   : (x,y,w,h) tuple or {'x','y','w','h'} dict in *source* coords, or None.
    bg_color : optional (r,g,b) known background color for decontamination; if
               None, background is estimated locally from the outside band.
    emit_shadow : override config.shadow_mode; True -> emit separate shadow.
    element_role : optional element role (e.g. "product", "image", "card").
        Gates the geometric card path: products/people/logos are never snapped
        to a rectangle; card/image/unknown roles may be. See CARD_BLOCK_ROLES.
    element_kind : optional element kind hint ("...card...", "screenshot").
    config : MattingConfig.

    Returns
    -------
    RefinedCutout with .rgba (uint8 HxWx4), .alpha (float), .box, .metrics,
    and optional .shadow.
    """
    _require_cv2()
    cfg = config or MattingConfig()
    img, m0, box_out = _crop_to_box(image, mask, box)
    H, W = img.shape[:2]

    raw_bool = m0 > 0.5
    if raw_bool.sum() == 0:
        empty = np.zeros((H, W, 4), np.uint8)
        return RefinedCutout(empty, np.zeros((H, W), np.float32), box_out,
                             metrics={"empty": True})

    ref = _min_dim(raw_bool)

    # 1) Cleanup on the binary silhouette.
    clean = _cleanup_binary(raw_bool, cfg)

    # 1b) Card path: if the element is a rectangular photo/screenshot card,
    #     snap to a geometric rounded-rect mask (exact straight edges + corners,
    #     no black/white fringe). Skips segmentation matting entirely.
    card_geom = None
    if _card_snap_allowed(cfg.card_snap, element_role, element_kind):
        card_geom = detect_card(clean, cfg)
    if card_geom is not None or (cfg.card_snap == "force"):
        if card_geom is None:  # forced: fit bbox of the cleaned mask
            card_geom = detect_card(clean, MattingConfig(card_min_rectangularity=0.0,
                                                         card_max_angle=90.0)) or {}
        alpha = _rounded_rect_alpha((H, W), card_geom)
        inner = alpha > 0.98
        outer = alpha > 0.02
        unknown = outer & (~inner)
        rgb = img.astype(np.float32)
        if cfg.decontaminate:
            rgb = _decontaminate(rgb, alpha, unknown, bg_color, inner, outer)
        rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                          (alpha * 255).astype(np.uint8)])
        metrics = _quality_metrics(rgba, alpha, raw_bool, clean)
        metrics.update({"ref_dim": int(ref), "path": "card_snap",
                        "card": card_geom, "shadow_policy": "plate"})
        return RefinedCutout(rgba=rgba, alpha=alpha.astype(np.float32),
                             box=box_out, metrics=metrics)

    # 2) Shadow policy — decide before building the FG trimap so a detected
    #    shadow is EXCLUDED from FG (never half-cut).
    want_shadow = (emit_shadow if emit_shadow is not None
                   else cfg.shadow_mode == "separate")
    shadow_mask, bg_V = _detect_shadow(img, clean, cfg)
    shadow_area = int(shadow_mask.sum())
    # Never let shadow pixels leak into the product FG.
    fg_core = clean & (~shadow_mask)

    # 3) Trimap: erode->FG, dilate->BG, band=unknown. Band width scales w/ res.
    band = _scaled(cfg.band_frac, ref, cfg.band_px_min, cfg.band_px_max)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(fg_core.astype(np.uint8), k, iterations=band).astype(bool)
    outer = cv2.dilate(fg_core.astype(np.uint8), k, iterations=band).astype(bool)
    unknown = outer & (~inner)

    # 4) Guided-filter alpha matting on the boundary band.
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    seed = fg_core.astype(np.float32)
    gr = _scaled(cfg.guided_radius_frac, ref, cfg.guided_radius_min, cfg.guided_radius_max)
    alpha = _guided_filter(guide, seed, gr, cfg.guided_eps)
    alpha = np.clip(alpha, 0.0, 1.0)
    # Hard-constrain definite regions; only trust the filter inside the band.
    alpha = np.where(inner, 1.0, alpha)
    alpha = np.where(~outer, 0.0, alpha)

    # 5) Resolution-matched feather polish.
    sigma = float(np.clip(cfg.feather_frac * ref, cfg.feather_min, cfg.feather_max))
    if sigma > 0.05:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha = np.where(inner, np.maximum(alpha, 0.995), alpha)
        alpha = np.where(~outer, 0.0, alpha)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # 5b) Background-fringe suppression — pull alpha back to 0 for band pixels
    #     whose colour matches the local background. This kills the white/bg halo
    #     the outward-grown trimap otherwise re-introduces (E006-class regression).
    if cfg.suppress_bg_fringe:
        alpha = _suppress_bg_fringe(alpha, img.astype(np.float32), inner, outer,
                                    bg_color, cfg, orig_fg=fg_core)
        alpha = np.where(inner, 1.0, alpha)
        alpha = np.where(~outer, 0.0, alpha)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # 6) Color decontamination — unmix FG color on partial-alpha pixels to kill
    #    the white/background fringe (the doubled-contour root cause).
    rgb = img.astype(np.float32)
    if cfg.decontaminate:
        rgb = _decontaminate(rgb, alpha, unknown, bg_color, inner, outer)

    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                      (alpha * 255).astype(np.uint8)])

    # 7) Optional separate shadow element (policy b).
    shadow_el = None
    if want_shadow and shadow_area > 0:
        s_alpha = shadow_mask.astype(np.float32)
        s_sigma = max(2.0, 0.02 * ref)
        s_alpha = cv2.GaussianBlur(s_alpha, (0, 0), s_sigma)
        s_alpha = np.clip(s_alpha, 0, 1).astype(np.float32)
        shadow_el = ShadowElement(alpha=s_alpha, box=box_out)

    metrics = _quality_metrics(rgba, alpha, raw_bool, fg_core)
    metrics.update({
        "ref_dim": int(ref), "band_px": int(band), "guided_radius": int(gr),
        "feather_sigma": round(sigma, 2),
        "shadow_detected": bool(shadow_area > 0),
        "shadow_area_px": shadow_area,
        "shadow_policy": ("separate" if (want_shadow and shadow_area > 0) else "plate"),
        "holes_filled": bool(fg_core.sum() != raw_bool.sum()),
        "bg_color_used": (list(bg_color) if bg_color is not None else "estimated"),
    })
    return RefinedCutout(rgba=rgba, alpha=alpha, box=box_out, metrics=metrics,
                         shadow=shadow_el)


def _local_bg_field(rgb, outer, bg_color):
    """Per-pixel local background colour field B (HxWx3 float).

    If a global bg_color is known, that; otherwise the colours just OUTSIDE the
    object spread inward with a large blur (an inpaint-like bg estimate).
    """
    H, W = outer.shape
    if bg_color is not None:
        B = np.empty((H, W, 3), np.float32)
        B[:] = np.array(bg_color, np.float32)
        return B
    outside_band = (~outer).astype(np.float32)
    ks = max(9, int(0.05 * max(H, W)) | 1)
    num = cv2.GaussianBlur(rgb * outside_band[..., None], (ks, ks), 0)
    den = cv2.GaussianBlur(outside_band, (ks, ks), 0)[..., None] + 1e-6
    return num / den


def _suppress_bg_fringe(alpha, rgb, inner, outer, bg_color, cfg, orig_fg=None):
    """Kill background-coloured halo in the unknown band.

    The guided-filter/dilate step can assign partial alpha to pixels that are
    really background (this is the white-fringe/doubled-contour defect). Where a
    band pixel's colour is within ``fringe_bg_dist_lo`` of the local background
    colour it is forced transparent; between lo and hi it is ramped down.

    Safety: when ``fringe_only_outside`` (default) and ``orig_fg`` is given, the
    suppression is restricted to pixels the dilate *added* OUTSIDE the original
    silhouette (``outer & ~orig_fg``). Originally-foreground pixels are never
    touched, so a genuinely white product edge on a white background is preserved
    (only the outward-grown halo is removed).
    """
    B = _local_bg_field(rgb, outer, bg_color)
    dist = np.sqrt(((rgb - B) ** 2).sum(axis=2))  # per-pixel colour distance to bg
    lo, hi = cfg.fringe_bg_dist_lo, max(cfg.fringe_bg_dist_hi, cfg.fringe_bg_dist_lo + 1.0)
    keep = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)  # 0 = bg (drop), 1 = fg (keep)
    if cfg.fringe_only_outside and orig_fg is not None:
        band = outer & (~orig_fg)          # only the outward-grown ring
    else:
        band = (~inner) & outer            # whole unknown band (aggressive)
    factor = np.where(band, keep, 1.0).astype(np.float32)
    return (alpha * factor).astype(np.float32)


def _decontaminate(rgb, alpha, unknown, bg_color, inner, outer):
    """Estimate true FG color on partial-alpha pixels: F = (C - (1-a)B) / a.

    B (local background) is either the provided bg_color or estimated from the
    just-outside band per-pixel (blurred). Only touches the unknown band.
    """
    H, W = alpha.shape
    if bg_color is not None:
        B = np.empty((H, W, 3), np.float32)
        B[:] = np.array(bg_color, np.float32)
    else:
        # Local background: take colors just OUTSIDE the object, spread inward
        # with a large blur so each edge pixel sees its nearby bg color.
        outside_band = (~outer).astype(np.float32)
        Bsrc = rgb.copy()
        # Blur bg colors and weights, normalize -> inpaint-like bg field.
        ks = max(9, int(0.05 * max(H, W)) | 1)
        num = cv2.GaussianBlur(Bsrc * outside_band[..., None], (ks, ks), 0)
        den = cv2.GaussianBlur(outside_band, (ks, ks), 0)[..., None] + 1e-6
        B = num / den

    a = alpha[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (rgb - (1.0 - a) * B) / np.clip(a, 1e-3, 1.0)
    F = np.clip(F, 0, 255)
    # The unmix is unstable at low alpha (a small error sends F to 0 or 255 --
    # the "over-dark edge" failure). Blend toward the unmixed color with a
    # confidence weight that ramps in only for reasonably opaque edge pixels,
    # so faint edge pixels keep their observed color instead of overshooting.
    conf = np.clip((alpha - 0.25) / 0.45, 0.0, 1.0)[..., None]
    # Extra guard: never let a single pixel move more than this far toward F.
    max_shift = 96.0
    delta = np.clip(F - rgb, -max_shift, max_shift)
    out = rgb + conf * delta
    band3 = (unknown & (alpha > 0.02) & (alpha < 0.98))[..., None]
    return np.where(band3, np.clip(out, 0, 255), rgb)


def _quality_metrics(rgba, alpha, raw_bool, fg_core):
    """Before/after quality numbers used by tests + reports."""
    a = alpha
    total = a.size
    partial = float(((a > 0.02) & (a < 0.98)).sum()) / total
    # Halo: brightness of inside-edge ring vs interior (the 002 defect).
    m = (a > 0.5).astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    inside_edge = (m > 0) & (er == 0)
    rgb = rgba[..., :3].astype(np.float32)
    if inside_edge.sum() and (er > 0).sum():
        edge_lum = float(rgb[inside_edge].mean())
        int_lum = float(rgb[er > 0].mean())
        white_frac = float((rgb[inside_edge].min(axis=1) > 235).mean())
    else:
        edge_lum = int_lum = white_frac = float("nan")
    return {
        "alpha_partial_frac": round(partial, 4),
        "edge_luma": round(edge_lum, 1) if edge_lum == edge_lum else None,
        "interior_luma": round(int_lum, 1) if int_lum == int_lum else None,
        "halo_delta": round(edge_lum - int_lum, 1) if edge_lum == edge_lum else None,
        "edge_white_frac": round(white_frac, 3) if white_frac == white_frac else None,
    }


# --------------------------------------------------------------------------- #
# Separability score (spec item 7): in-scene vs cutout judgment
# --------------------------------------------------------------------------- #

# Approximate skin-tone gate in YCrCb (hands / held products).
def _skin_mask(img_rgb: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Cr = ycc[..., 1]; Cb = ycc[..., 2]
    return ((Cr > 135) & (Cr < 180) & (Cb > 85) & (Cb < 135)).astype(np.uint8)


def separability_score(image, mask, box=None, *,
                       config: Optional[MattingConfig] = None) -> Separability:
    """Score how cleanly an element can be cut out (0..1).

    Combines:
      * boundary_contrast — mean gradient magnitude of the image along the mask
        boundary (a crisp studio product on a plain bg = high).
      * bg_complexity — texture/variance of the region just OUTSIDE the object
        (busy photo bg = high -> harder, lowers score).
      * hand_occlusion — fraction of the object boundary adjacent to skin tone
        (hand-held product -> should stay in the photo plate).

    recommend_cutout is True when score >= config.sep_cutout_threshold. Hand-held
    / in-scene objects fall below and should stay in the plate (spec item 7).
    """
    _require_cv2()
    cfg = config or MattingConfig()
    img, m, _ = _crop_to_box(image, mask, box)
    fg = m > 0.5
    if fg.sum() < 16:
        return Separability(0.0, 0.0, 1.0, 0.0, False, "mask too small")

    k = np.ones((3, 3), np.uint8)
    fgu = fg.astype(np.uint8)
    dil = cv2.dilate(fgu, k, iterations=2).astype(bool)
    ero = cv2.erode(fgu, k, iterations=2).astype(bool)
    boundary = dil & (~ero)
    outside = cv2.dilate(fgu, k, iterations=max(4, int(0.05 * _min_dim(fg)))).astype(bool) & (~dil)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    # Normalize contrast against the image's own strong edges (95th pct).
    ref_edge = np.percentile(gmag, 95) + 1e-6
    boundary_contrast = float(np.clip(gmag[boundary].mean() / ref_edge, 0, 1)) if boundary.sum() else 0.0

    # Background complexity: normalized std of luma just outside the object.
    if outside.sum() > 20:
        bg_complexity = float(np.clip(gray[outside].std() / 64.0, 0, 1))
    else:
        bg_complexity = 1.0

    # Hand occlusion: skin adjacency along the boundary.
    skin = _skin_mask(img)
    near = cv2.dilate(boundary.astype(np.uint8), k, iterations=3).astype(bool)
    denom = max(1, int(boundary.sum()))
    hand_occlusion = float((skin.astype(bool) & near).sum()) / (near.sum() + 1e-6)
    hand_occlusion = float(np.clip(hand_occlusion, 0, 1))

    # Combine. Contrast helps; complexity + hands hurt.
    score = (0.55 * boundary_contrast
             + 0.25 * (1.0 - bg_complexity)
             + 0.20 * (1.0 - hand_occlusion))
    # Hard penalty: substantial hand contact => almost certainly in-scene.
    if hand_occlusion > 0.20:
        score = min(score, 0.45)
    score = float(np.clip(score, 0, 1))

    reason = []
    if boundary_contrast < 0.3: reason.append("soft/low-contrast boundary")
    if bg_complexity > 0.6: reason.append("busy background")
    if hand_occlusion > 0.2: reason.append("hand-held / skin contact")
    rec = score >= cfg.sep_cutout_threshold
    return Separability(round(score, 3), round(boundary_contrast, 3),
                        round(bg_complexity, 3), round(hand_occlusion, 3),
                        rec, "; ".join(reason) or "clean, separable")


# --------------------------------------------------------------------------- #
# First-class artifact saving (spec item 5): mask PNG + cutout, predictable name
# --------------------------------------------------------------------------- #

def save_cutout(refined: RefinedCutout, out_dir: str, elem_id: str) -> Dict[str, str]:
    """Write the cutout RGBA + its alpha mask (and shadow) as first-class assets.

    Predictable names the Figma staging / preflight step can rely on:
        {elem_id}_cutout.png   RGBA cutout
        {elem_id}_alpha.png     8-bit grayscale alpha mask  <-- was missing at preflight
        {elem_id}_shadow.png    grayscale shadow alpha (only if a shadow element)

    Returns a dict of {kind: absolute_path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}

    cutout_path = os.path.join(out_dir, f"{elem_id}_cutout.png")
    Image.fromarray(refined.rgba, "RGBA").save(cutout_path)
    out["cutout"] = os.path.abspath(cutout_path)

    alpha_path = os.path.join(out_dir, f"{elem_id}_alpha.png")
    Image.fromarray((refined.alpha * 255).astype(np.uint8), "L").save(alpha_path)
    out["alpha"] = os.path.abspath(alpha_path)

    if refined.shadow is not None:
        shadow_path = os.path.join(out_dir, f"{elem_id}_shadow.png")
        Image.fromarray((refined.shadow.alpha * 255).astype(np.uint8), "L").save(shadow_path)
        out["shadow"] = os.path.abspath(shadow_path)

    return out
