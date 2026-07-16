"""Glass detection/estimation unit tests.

Reproduces the 5 synthetic-composite validation cases (A-E) from
docs/GLASS-RESEARCH-DETECTION.md §4, plus the detection-trigger separation table,
plus the Figma blur-radius conversion, text exclusion, and the solid fallback.

Synthetic "photo" = smooth low-freq color field + Gaussian pixel noise + sharp blobs
(so a real background-blur has visible structure to remove). Fixed RNG seed -> the
recovery numbers match the doc's table.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.glass_detect import (
    FIGMA_RADIUS_PER_SIGMA,
    detect_glass,
    estimate_glass,
    figma_radius_to_sigma,
    sigma_to_figma_radius,
)

SIGMA_GRID = tuple(round(float(s), 1) for s in np.arange(0.5, 20.5, 0.5))


def _make_photo(w, h, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        120 + 60 * np.sin(xx / 40) + 30 * np.cos(yy / 33),
        90 + 50 * np.cos(xx / 29) + 40 * np.sin(yy / 51),
        140 + 40 * np.sin((xx + yy) / 37),
    ], axis=-1)
    img = np.clip(base + rng.normal(0, 18, size=(h, w, 3)), 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(25):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 14))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)
    return pil


def _composite_glass(bg, region, alpha, fill_rgb, sigma):
    x0, y0, x1, y1 = region
    bg_arr = np.asarray(bg).astype(np.float32)
    out = bg_arr.copy()
    if sigma > 0:
        blurred = np.asarray(bg.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32)
    else:
        blurred = bg_arr
    fill = np.array(fill_rgb, dtype=np.float32)
    patch = blurred[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = alpha * fill + (1 - alpha) * patch
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _run(region, alpha, fill, sigma, w=300, h=200):
    bg = _make_photo(w, h)
    observed = _composite_glass(bg, region, alpha, fill, sigma)
    return estimate_glass(observed, bg, region, SIGMA_GRID)


# ── Recovery accuracy (§4 table, cases A-E) ───────────────────────────────────────

def test_case_a_typical_h18_chip():
    fit = _run((150, 60, 280, 140), 0.18, (255, 255, 255), 8.0)
    assert abs(fit.alpha - 0.18) < 0.01
    assert abs(fit.sigma - 8.0) <= 0.5           # exact to grid resolution
    assert np.linalg.norm(np.array(fit.fill_color) - 255) < 12


def test_case_b_opaque_frosted_panel():
    fit = _run((150, 60, 280, 140), 0.28, (255, 255, 255), 14.0)
    assert abs(fit.alpha - 0.28) < 0.01
    assert abs(fit.sigma - 14.0) <= 0.5
    assert np.linalg.norm(np.array(fit.fill_color) - 255) < 12


def test_case_c_low_opacity_low_blur():
    fit = _run((150, 60, 280, 140), 0.12, (255, 255, 255), 3.0)
    assert abs(fit.alpha - 0.12) < 0.01
    assert abs(fit.sigma - 3.0) <= 0.5
    assert np.linalg.norm(np.array(fit.fill_color) - 255) < 14


def test_case_d_solid_opaque_control_degeneracy():
    # alpha must saturate to ~1; sigma is meaningless (unidentifiable) but the
    # DETECTOR must classify this as solid, never glass.
    bg = _make_photo(300, 200)
    observed = _composite_glass(bg, (150, 60, 280, 140), 1.0, (245, 245, 240), 0.0)
    fit = estimate_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
    assert abs(fit.alpha - 1.0) < 0.01
    result = detect_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
    assert result.classification == "solid"
    assert not result.is_glass
    assert result.background_blur_radius is None
    # solid fill color still recovered close to ground truth
    assert np.linalg.norm(np.array(result.fill_color) - np.array([245, 245, 240])) < 6


def test_case_e_small_tinted_chip():
    fit = _run((170, 100, 260, 140), 0.20, (250, 248, 235), 10.0)
    assert abs(fit.alpha - 0.20) < 0.01
    assert abs(fit.sigma - 10.0) <= 0.5
    assert np.linalg.norm(np.array(fit.fill_color) - np.array([250, 248, 235])) < 10


# ── Detection-trigger separation (§4 second table) ────────────────────────────────

def test_detection_classifies_real_glass_as_glass():
    bg = _make_photo(300, 200)
    for alpha, sigma in [(0.18, 8.0), (0.28, 14.0)]:
        observed = _composite_glass(bg, (150, 60, 280, 140), alpha, (255, 255, 255), sigma)
        result = detect_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
        assert result.is_glass, (alpha, sigma, result.reason)
        assert result.fill_opacity is not None
        assert abs(result.fill_opacity - alpha) < 0.01
        # emitted radius is Figma-space (sigma * ~2.273), not raw sigma
        assert abs(result.background_blur_radius - sigma_to_figma_radius(sigma)) < 1.5
        assert result.fit.rss_ratio < 0.01     # 3-4 orders below solid


def test_near_opaque_95_falls_back_not_glass():
    # 95% opacity: technically translucent but visually solid -> must NOT emit blur.
    bg = _make_photo(300, 200)
    observed = _composite_glass(bg, (150, 60, 280, 140), 0.95, (255, 255, 255), 5.0)
    result = detect_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
    assert not result.is_glass
    assert result.classification in ("solid", "low-confidence")
    assert result.background_blur_radius is None


def test_near_invisible_5pct_glass_still_detected():
    bg = _make_photo(300, 200)
    observed = _composite_glass(bg, (150, 60, 280, 140), 0.05, (255, 255, 255), 8.0)
    result = detect_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
    assert result.is_glass, result.reason
    assert abs(result.fill_opacity - 0.05) < 0.02


# ── Fallback: a flat solid chip over the photo must degrade to solid ───────────────

def test_flat_solid_chip_degrades_to_solid():
    bg = _make_photo(300, 200)
    observed = np.asarray(bg).copy()
    observed[60:140, 150:280] = (210, 205, 190)   # opaque flat chip, no bg leakage
    observed = Image.fromarray(observed)
    result = detect_glass(observed, bg, (150, 60, 280, 140), SIGMA_GRID)
    assert result.classification == "solid"
    assert not result.is_glass
    assert np.linalg.norm(np.array(result.fill_color) - np.array([210, 205, 190])) < 4


# ── Text-inside-glass exclusion (§2.5) ────────────────────────────────────────────

def test_text_pixels_corrupt_fit_and_exclusion_recovers_it():
    bg = _make_photo(300, 200)
    region = (150, 60, 280, 140)
    observed = _composite_glass(bg, region, 0.18, (255, 255, 255), 8.0)
    # Paint dark "text" strokes over the glass fill (violates region-constant color).
    arr = np.asarray(observed).copy()
    text_box = (170, 90, 250, 110)
    arr[90:110, 170:250] = (20, 20, 20)
    corrupted = Image.fromarray(arr)

    polluted = estimate_glass(corrupted, bg, region, SIGMA_GRID)
    cleaned = estimate_glass(corrupted, bg, region, SIGMA_GRID, exclude_boxes=[text_box])
    # Excluding the text box brings alpha back near the true 0.18; leaving it in does not.
    assert abs(cleaned.alpha - 0.18) < abs(polluted.alpha - 0.18)
    assert abs(cleaned.alpha - 0.18) < 0.02


# ── Figma blur-radius conversion (§2.4) ───────────────────────────────────────────

def test_figma_radius_conversion_roundtrip():
    assert abs(FIGMA_RADIUS_PER_SIGMA - 2.272728) < 1e-6
    assert abs(sigma_to_figma_radius(8.0) - 18.18) < 0.01
    assert abs(figma_radius_to_sigma(sigma_to_figma_radius(11.5)) - 11.5) < 1e-9
    # A raw Figma radius fed to PIL unconverted would over-blur by ~2.27x.
    assert abs(sigma_to_figma_radius(1.0) / 1.0 - 2.272728) < 1e-6
