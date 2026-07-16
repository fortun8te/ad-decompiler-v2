"""Frosted-glass / translucent-panel detection and parameter estimation from pixels.

Given the original ad crop plus an inpainted *clean background plate* (no overlay,
NOT blurred — the LaMa output the pipeline already produces for occlusion recovery),
estimate for a candidate rounded-rect chip region:

  - ``alpha``       fill opacity 0..1 (Figma per-fill opacity, NOT layer opacity)
  - ``fill_color``  the constant fill RGB (white-ish for typical glass)
  - ``sigma``       the backdrop Gaussian blur standard deviation, in PIL px

and decide whether the region is really glass, a flat solid, or too ambiguous to
trust (low confidence → caller ships a plain solid rect at ``fill_color``).

Forward (compositing) model, per pixel, per channel, region-constant alpha/fill/sigma::

    observed = alpha * fill_color + (1 - alpha) * GaussianBlur(bg, sigma)

Solved WITHOUT blind deconvolution because ``bg`` (what is underneath) is already
known from inpainting: the blur estimate becomes a 1-D search over candidate sigma.
For each fixed sigma the model is linear once reparametrized ``beta_c = alpha*fill_c``::

    observed_c - bg_blur_c = beta_c - alpha * bg_blur_c        (linear in beta_r,beta_g,beta_b,alpha)

solved by ``numpy.linalg.lstsq``; pick the sigma with minimum residual sum of squares.

Detection trigger (is it glass at all?) combines:
  * glass-model RSS vs a flat-solid-color model RSS (real translucency leaks
    background structure through, a solid chip does not), and
  * the alpha->1 degeneracy: as alpha saturates, ``(1-alpha)->0`` makes sigma
    unidentifiable (a flat RSS-vs-sigma curve) — the mathematical signature of
    "not glass". Bias hard against false positives per FEATURE-PLAN §W2.

Figma blur-radius conversion (see docs/GLASS-RESEARCH-DETECTION.md §2.4):
  figma_radius = 2.272728 * sigma  (CSS spec sigma=radius/2  x  Bjango's measured
  Figma 1.136364x factor). Estimation->emission multiplies; the QA renderer's
  backdrop blur divides Figma radius back to a PIL sigma before GaussianBlur.

CPU / numpy / Pillow only. No GPU, no ML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

# ── Verified Figma blur-radius <-> Gaussian-sigma conversion (§2.4) ────────────────
# CSS spec: sigma = css_radius / 2.  Bjango measured: figma_radius = 1.136364 * css_radius.
# Chained: figma_radius = 1.136364 * (2 * sigma) = 2.272728 * sigma.
FIGMA_RADIUS_PER_SIGMA = 2.272728


def sigma_to_figma_radius(sigma: float) -> float:
    """PIL GaussianBlur sigma -> Figma BackgroundBlurEffect.radius (for emission)."""
    return float(sigma) * FIGMA_RADIUS_PER_SIGMA


def figma_radius_to_sigma(figma_radius: float) -> float:
    """Figma BackgroundBlurEffect.radius -> PIL GaussianBlur sigma (for the QA renderer).

    Feeding a raw Figma radius straight into PIL.GaussianBlur over-blurs by ~2.27x —
    always convert first (docs/GLASS-RESEARCH-DETECTION.md §2.4 caught bug).
    """
    return float(figma_radius) / FIGMA_RADIUS_PER_SIGMA


# ── Tunables (validated in tests/test_glass_detect.py against §4 synthetic cases) ──
DEFAULT_SIGMA_GRID = tuple(round(float(s), 1) for s in np.arange(0.5, 20.5, 0.5))

# alpha at/above this reads as visually solid — emit a solid rect, never blur.
ALPHA_SOLID = 0.97
# alpha in [ALPHA_GRAY, ALPHA_SOLID) is the conservative gray zone -> low confidence.
ALPHA_GRAY = 0.85
# glass model must explain far more variance than a flat solid (real bg leakage).
# Real glass separates by 3-4 orders of magnitude (§4); a generous ceiling still
# rejects "flat light chip" impostors whose glass_rss ~ solid_rss.
RATIO_GLASS_MAX = 0.30
# RSS-vs-sigma curve flatness: min_rss / median_rss near 1.0 means sigma is
# unidentifiable (the alpha->1 degeneracy) -> treat as solid, not glass.
FLATNESS_SOLID = 0.90
# A region whose flat-solid RSS is already ~0 is a genuine single-color chip.
SOLID_RSS_FLOOR = 1e-6
# Text-box dilation (px) to swallow anti-aliased glyph edges before excluding them.
TEXT_DILATION = 3


@dataclass
class GlassFit:
    """Raw least-squares fit result for a single best sigma."""
    alpha: float
    fill_color: tuple  # (r, g, b), 0..255 float
    sigma: float       # PIL GaussianBlur sigma
    rss: float         # glass-model residual sum of squares at the best sigma
    solid_rss: float   # flat-single-color model RSS over the same pixels
    rss_ratio: float   # rss / max(solid_rss, eps)
    flatness: float    # min_rss / median_rss across the sigma grid (~1 == degenerate)
    n_pixels: int      # pixels actually used in the fit (after text exclusion)


@dataclass
class GlassResult:
    """Detection verdict plus emission-ready fields."""
    classification: str            # "glass" | "solid" | "low-confidence"
    is_glass: bool                 # True only when classification == "glass"
    fill_color: tuple              # (r, g, b) int 0..255 — always usable as a solid fill
    fill_opacity: Optional[float]  # alpha for the glass fill; None unless is_glass
    background_blur_radius: Optional[float]  # FIGMA-space radius; None unless is_glass
    sigma: Optional[float]         # PIL sigma (pre-conversion); None unless is_glass
    reason: str                    # human-readable classification rationale
    fit: GlassFit = field(repr=False, default=None)


# ── core estimator ────────────────────────────────────────────────────────────────

def _as_rgb_array(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.asarray(image.convert("RGB"))
    return arr.astype(np.float64)


def _region_pixel_mask(region, shape_hw, exclude_boxes, dilation):
    """Boolean mask (H, W) True for region pixels kept in the fit (text excluded)."""
    x0, y0, x1, y1 = region
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    for box in exclude_boxes or []:
        bx0, by0, bx1, by1 = box
        bx0 = max(0, int(bx0) - dilation)
        by0 = max(0, int(by0) - dilation)
        bx1 = min(w, int(bx1) + dilation)
        by1 = min(h, int(by1) + dilation)
        if bx1 > bx0 and by1 > by0:
            mask[by0:by1, bx0:bx1] = False
    return mask


def _solve_linear(obs: np.ndarray, bgp: np.ndarray):
    """Least-squares solve of the reparametrized glass model for a fixed sigma.

    obs, bgp: (n, 3) float arrays of observed and blurred-bg pixels.
    Returns (alpha, beta[3], rss). beta_c = alpha * fill_c.
    """
    n = obs.shape[0]
    A = np.zeros((n * 3, 4))
    b = np.zeros(n * 3)
    for c in range(3):
        rows = slice(c * n, (c + 1) * n)
        A[rows, c] = 1.0
        A[rows, 3] = -bgp[:, c]
        b[rows] = obs[:, c] - bgp[:, c]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    beta = sol[:3]
    alpha = float(sol[3])
    pred = beta[None, :] + (1.0 - alpha) * bgp
    rss = float(np.sum((pred - obs) ** 2))
    return alpha, beta, rss


def estimate_glass(
    orig,
    bg,
    region,
    sigma_grid: Sequence[float] = DEFAULT_SIGMA_GRID,
    exclude_boxes: Optional[Sequence] = None,
    text_dilation: int = TEXT_DILATION,
) -> GlassFit:
    """Fit (alpha, fill_color, sigma) for one candidate region.

    orig, bg: PIL images or HxWx3 arrays, pixel-aligned, same size.
    region:   (x0, y0, x1, y1) in image px.
    exclude_boxes: OCR/text bounding boxes (image px) to drop from the fit (§2.5).

    Returns a GlassFit; it always returns *some* fit (the trigger in
    ``detect_glass`` decides whether to trust it).
    """
    orig_arr = _as_rgb_array(orig)
    bg_img = bg if isinstance(bg, Image.Image) else Image.fromarray(
        np.clip(np.asarray(bg), 0, 255).astype(np.uint8)
    )
    bg_arr = _as_rgb_array(bg_img)
    h, w = orig_arr.shape[:2]

    x0, y0, x1, y1 = (int(round(v)) for v in region)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty region {region} for image {w}x{h}")

    mask = _region_pixel_mask((x0, y0, x1, y1), (h, w), exclude_boxes, text_dilation)
    if mask.sum() < 16:  # fell back too aggressively; use the raw region
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True

    obs = orig_arr[mask].reshape(-1, 3)

    # Flat-solid competing hypothesis: RSS around the region mean color (§3.1).
    mean_c = obs.mean(axis=0)
    solid_rss = float(np.sum((obs - mean_c) ** 2))

    best = None
    rss_values = []
    for sigma in sigma_grid:
        blurred = _as_rgb_array(bg_img.filter(ImageFilter.GaussianBlur(float(sigma))))
        bgp = blurred[mask].reshape(-1, 3)
        alpha, beta, rss = _solve_linear(obs, bgp)
        rss_values.append(rss)
        if best is None or rss < best[0]:
            fill = beta / alpha if abs(alpha) > 1e-6 else beta * 0.0
            best = (rss, float(sigma), alpha, tuple(float(v) for v in fill))

    rss_arr = np.asarray(rss_values, dtype=np.float64)
    median_rss = float(np.median(rss_arr))
    min_rss = float(rss_arr.min())
    flatness = min_rss / median_rss if median_rss > SOLID_RSS_FLOOR else 1.0

    rss, sigma, alpha, fill = best
    return GlassFit(
        alpha=alpha,
        fill_color=fill,
        sigma=sigma,
        rss=rss,
        solid_rss=solid_rss,
        rss_ratio=rss / max(solid_rss, SOLID_RSS_FLOOR),
        flatness=flatness,
        n_pixels=int(obs.shape[0]),
    )


def _clamp_rgb(color) -> tuple:
    return tuple(int(round(min(255.0, max(0.0, c)))) for c in color[:3])


def detect_glass(
    orig,
    bg,
    region,
    sigma_grid: Sequence[float] = DEFAULT_SIGMA_GRID,
    exclude_boxes: Optional[Sequence] = None,
    text_dilation: int = TEXT_DILATION,
) -> GlassResult:
    """Full detect + classify. Returns a GlassResult with emission-ready fields.

    Classification (§3):
      * alpha >= ALPHA_SOLID, or a flat RSS-vs-sigma curve, or a region that is
        already a single flat color   -> "solid"  (emit a plain rect at fill_color)
      * alpha in the gray zone, or the glass model fails to beat flat-solid
        clearly                       -> "low-confidence" (also degrades to solid)
      * alpha well below 1, glass RSS << solid RSS, and a well-defined sigma
        minimum                       -> "glass" (emit fill.opacity + background-blur)
    """
    fit = estimate_glass(orig, bg, region, sigma_grid, exclude_boxes, text_dilation)
    color = _clamp_rgb(fit.fill_color)

    def solid(reason):
        return GlassResult("solid", False, color, None, None, None, reason, fit)

    def low_conf(reason):
        return GlassResult("low-confidence", False, color, None, None, None, reason, fit)

    # Genuine single-color chip: nothing for the bg model to explain.
    if fit.solid_rss < SOLID_RSS_FLOOR:
        return solid("region is a flat single color (solid_rss~0)")

    # alpha->1 degeneracy: fully/near opaque, sigma unidentifiable (§3.2).
    if fit.alpha >= ALPHA_SOLID:
        return solid(f"alpha={fit.alpha:.3f} >= {ALPHA_SOLID} (opaque, sigma unidentifiable)")

    # Flat RSS-vs-sigma curve == sigma not identified == effectively solid.
    if fit.flatness >= FLATNESS_SOLID:
        return solid(f"flat RSS-vs-sigma curve (flatness={fit.flatness:.3f}); sigma unidentified")

    # Conservative gray zone: visually indistinguishable from solid, don't risk a blur.
    if fit.alpha >= ALPHA_GRAY:
        return low_conf(f"alpha={fit.alpha:.3f} in gray zone [{ALPHA_GRAY},{ALPHA_SOLID})")

    # Glass model must clearly beat the flat-solid hypothesis.
    if fit.rss_ratio > RATIO_GLASS_MAX:
        return low_conf(
            f"glass model does not clearly beat flat-solid "
            f"(ratio={fit.rss_ratio:.4f} > {RATIO_GLASS_MAX})"
        )

    # Passed every gate -> real glass.
    return GlassResult(
        classification="glass",
        is_glass=True,
        fill_color=color,
        fill_opacity=round(float(fit.alpha), 4),
        background_blur_radius=round(sigma_to_figma_radius(fit.sigma), 3),
        sigma=fit.sigma,
        reason=(
            f"alpha={fit.alpha:.3f}, ratio={fit.rss_ratio:.4g}, "
            f"sigma={fit.sigma:.1f} (figma_radius={sigma_to_figma_radius(fit.sigma):.2f})"
        ),
        fit=fit,
    )
