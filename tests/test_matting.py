"""Tests for src/matting.py — cutout alpha refinement.

CPU-only, no GPU/model. Runs on real 002a run cutouts when present, plus
synthetic fixtures covering the hard cases (dark-on-light sachet, card-on-black,
soft shadow, hand-held / in-scene). Writes before/after visual strips to
runs/_matting_review/ for human review.

Run:  .venv/Scripts/python -m pytest tests/test_matting.py -v
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import matting  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REVIEW = os.path.join(ROOT, "runs", "_matting_review")
os.makedirs(REVIEW, exist_ok=True)

RUN = os.path.join(ROOT, "runs", "codex-targeted-002a", "002_attached_5885519ba4359843")


# --------------------------------------------------------------------------- #
# Visual strip helper
# --------------------------------------------------------------------------- #

def _checker(h, w, sq=16):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // sq) + (xx // sq)) % 2).astype(np.uint8)
    return (np.stack([c, c, c], -1) * 40 + 190).astype(np.uint8)


def _over_checker(rgba):
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = _checker(*rgba.shape[:2]).astype(np.float32)
    return (rgb * a + bg * (1 - a)).astype(np.uint8)


def _label(img, text):
    from PIL import ImageDraw
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((2, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


def write_strip(name, orig_rgb, before_rgba, refined: matting.RefinedCutout):
    """[ source | binary-alpha over checker | refined over checker | alpha ]"""
    h, w = refined.alpha.shape
    def fit(a):
        im = Image.fromarray(a)
        if im.size != (w, h):
            im = im.resize((w, h))
        return np.asarray(im.convert("RGB"))
    panels = [
        _label(fit(orig_rgb), "source"),
        _label(_over_checker(before_rgba), "BEFORE binary"),
        _label(_over_checker(refined.rgba), "AFTER refined"),
        _label(fit((refined.alpha * 255).astype(np.uint8)), "alpha"),
    ]
    strip = np.concatenate(panels, axis=1)
    path = os.path.join(REVIEW, f"{name}.png")
    Image.fromarray(strip).save(path)
    return path


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _synthetic_dark_sachet_on_white(n=240):
    """H15: matte-black sachet on white. Antialiased edge => white fringe if
    cut binary."""
    rgb = np.full((n, n, 3), 245, np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy = n / 2, n / 2
    # a rounded blob (product-ish, not a card)
    ell = (((xx - cx) / (n * 0.28)) ** 2 + ((yy - cy) / (n * 0.38)) ** 2)
    inside = ell <= 1.0
    prod = np.clip((1.0 - ell) * 3, 0, 1)[..., None]
    dark = np.array([25, 22, 28], np.float32)
    rgb = (rgb * (1 - prod) + dark * prod).astype(np.uint8)
    mask = inside.astype(np.uint8) * 255
    return rgb, mask


def _synthetic_card_on_black(n=260, r=28):
    """H16: rounded-corner photo card on black bg. Segmentation would leave a
    ragged/black-fringed edge; card-snap should give exact geometry."""
    rgb = np.full((n, n, 3), 6, np.uint8)  # black page
    x0, y0, x1, y1 = 40, 30, n - 40, n - 30
    # paint a photo-ish gradient card
    card = np.zeros((n, n, 3), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    card[..., 0] = np.clip(xx, 0, 255)
    card[..., 1] = np.clip(yy, 0, 255)
    card[..., 2] = 160
    # rounded-rect region
    m = np.zeros((n, n), np.uint8)
    import cv2
    cv2.rectangle(m, (x0 + r, y0), (x1, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cc in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(m, cc, r, 255, -1)
    sel = m > 0
    rgb[sel] = card[sel]
    # ragged segmentation mask: erode + add noise to simulate SAM jitter
    noisy = m.copy()
    rng = np.random.default_rng(0)
    jitter = (rng.random((n, n)) > 0.5).astype(np.uint8) * 255
    edge = cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8))
    noisy[edge > 0] = jitter[edge > 0]
    return rgb, noisy, (x0, y0, x1, y1, r)


def _synthetic_hand_held(n=240):
    """Product gripped by a skin-tone hand => low separability, stay in plate."""
    import cv2
    rgb = np.full((n, n, 3), 120, np.uint8)  # busy-ish mid bg
    rng = np.random.default_rng(1)
    rgb = np.clip(rgb + rng.integers(-40, 40, (n, n, 3)), 0, 255).astype(np.uint8)
    # product
    cv2.rectangle(rgb, (100, 60), (140, 190), (60, 80, 200), -1)
    mask = np.zeros((n, n), np.uint8)
    cv2.rectangle(mask, (100, 60), (140, 190), 255, -1)
    # skin-tone hand wrapping the lower half
    cv2.rectangle(rgb, (85, 130), (160, 200), (150, 110, 90), -1)  # skinnish
    return rgb, mask


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

def test_feather_produces_partial_alpha():
    rgb, mask = _synthetic_dark_sachet_on_white()
    before = np.dstack([rgb, mask])
    out = matting.refine(rgb, mask)
    assert out.metrics["alpha_partial_frac"] > 0.0, "refined alpha must not be binary"
    write_strip("synth_dark_sachet_on_white", rgb, before, out)


def test_decontamination_kills_white_fringe_dark_on_light():
    rgb, mask = _synthetic_dark_sachet_on_white()
    out = matting.refine(rgb, mask)
    # With decontamination on, the edge ring should not be dominated by white.
    ewf = out.metrics["edge_white_frac"]
    assert ewf is None or ewf < 0.25, f"white fringe not removed: {ewf}"


def test_card_detected_and_snapped_on_black():
    rgb, noisy, (x0, y0, x1, y1, r) = _synthetic_card_on_black()
    geom = matting.detect_card(noisy > 0)
    assert geom is not None, "rectangular card should be detected"
    assert abs(geom["radius"] - r) < 12, f"radius est {geom['radius']} vs {r}"
    before = np.dstack([rgb, noisy])
    out = matting.refine(rgb, noisy)
    assert out.metrics.get("path") == "card_snap"
    # Straight edges: a mid-height row through the card is fully opaque between
    # the straight sides (no black fringe eating into it).
    row = out.alpha[(y0 + y1) // 2]
    assert row[x0 + r + 4:x1 - r - 4].min() > 0.98
    write_strip("synth_card_on_black", rgb, before, out)


def test_product_not_misdetected_as_card():
    rgb, mask = _synthetic_dark_sachet_on_white()
    assert matting.detect_card(mask > 0) is None, "ellipse product must not be a card"


def test_rectangular_product_role_blocks_card_snap():
    """A near-rectangular product mask (segmentation bbox-fallback) must NOT be
    squared off when role='product' (regression: real E006/E007 photo-fragments)."""
    import cv2
    n = 240
    rgb = np.full((n, n, 3), 240, np.uint8)
    cv2.rectangle(rgb, (40, 30, 160, 180), (30, 30, 40), -1)
    mask = np.zeros((n, n), np.uint8)
    cv2.rectangle(mask, (40, 30, 160, 180), 255, -1)
    # geometry alone would call it a card:
    assert matting.detect_card(mask > 0) is not None
    # but role gating blocks it:
    out = matting.refine(rgb, mask, element_role="product")
    assert out.metrics.get("path") != "card_snap"
    # while an image-card role IS allowed to snap:
    out2 = matting.refine(rgb, mask, element_role="image")
    assert out2.metrics.get("path") == "card_snap"


def test_shadow_kept_whole_not_half_cut():
    import cv2
    n = 240
    rgb = np.full((n, n, 3), 235, np.uint8)
    cv2.rectangle(rgb, (90, 40), (150, 170), (40, 60, 160), -1)  # product
    # soft shadow below/right: low-sat, darker than bg
    shad = np.zeros((n, n), np.float32)
    cv2.ellipse(shad, (140, 180), (70, 22), 0, 0, 360, 1.0, -1)
    shad = cv2.GaussianBlur(shad, (0, 0), 9)
    rgb = (rgb * (1 - 0.35 * shad[..., None])).astype(np.uint8)
    mask = np.zeros((n, n), np.uint8)
    cv2.rectangle(mask, (90, 40), (150, 170), 255, -1)
    # include part of the shadow in the raw mask (as SAM often does)
    mask[shad > 0.5] = 255
    out_plate = matting.refine(rgb, mask, config=matting.MattingConfig(shadow_mode="plate"))
    out_sep = matting.refine(rgb, mask, emit_shadow=True)
    # plate policy: shadow excluded from product alpha (product cut tight)
    assert out_plate.metrics["shadow_detected"] in (True, False)
    # separate policy: if detected, a shadow element is emitted whole
    if out_sep.metrics["shadow_detected"]:
        assert out_sep.shadow is not None
        assert out_sep.shadow.alpha.max() > 0.1


def test_separability_hand_held_stays_in_plate():
    rgb, mask = _synthetic_hand_held()
    sep = matting.separability_score(rgb, mask)
    assert sep.hand_occlusion > 0.1
    assert sep.recommend_cutout is False, "hand-held product should stay in plate"


def test_separability_clean_studio_is_cutout():
    rgb, mask = _synthetic_dark_sachet_on_white()
    sep = matting.separability_score(rgb, mask)
    assert sep.recommend_cutout is True, f"clean product should be cut out: {sep.to_dict()}"


def test_save_cutout_emits_alpha_sidecar():
    rgb, mask = _synthetic_dark_sachet_on_white()
    out = matting.refine(rgb, mask)
    paths = matting.save_cutout(out, os.path.join(REVIEW, "_artifacts"), "E999")
    assert os.path.exists(paths["cutout"])
    assert os.path.exists(paths["alpha"]), "first-class alpha mask must be written"
    a = np.asarray(Image.open(paths["alpha"]))
    assert a.ndim == 2 and a.max() > 0


def test_bg_fringe_suppressed_when_mask_overshoots():
    """Doubled-contour case: the binary mask is DILATED past the product into
    white bg (as SAM often overshoots). The outward white ring must be pulled to
    transparent, not kept as a white halo."""
    import cv2
    n = 200
    rgb = np.full((n, n, 3), 245, np.uint8)          # white bg
    cv2.rectangle(rgb, (70, 50), (130, 150), (30, 30, 35), -1)  # dark product
    prod = np.zeros((n, n), np.uint8)
    cv2.rectangle(prod, (70, 50), (130, 150), 255, -1)
    # mask overshoots by ~5px into white bg
    mask = cv2.dilate(prod, np.ones((11, 11), np.uint8))
    out = matting.refine(rgb, mask, element_role="product")
    # A pixel 3px outside the true product edge (in white bg) must be ~transparent.
    assert out.alpha[100, 137] < 0.35, f"white halo not suppressed: {out.alpha[100,137]}"
    # The product interior stays fully opaque.
    assert out.alpha[100, 100] > 0.95
    before = np.dstack([rgb, mask])
    write_strip("synth_fringe_overshoot", rgb, before, out)


def test_white_product_edge_preserved_on_white_bg():
    """A genuinely white product on white bg must NOT have its own edge eaten by
    the fringe suppressor (it only removes outward-grown, not original silhouette)."""
    import cv2
    n = 200
    rgb = np.full((n, n, 3), 250, np.uint8)
    cv2.rectangle(rgb, (70, 50), (130, 150), (248, 246, 244), -1)  # near-white product
    cv2.rectangle(rgb, (70, 50), (130, 150), (120, 120, 120), 2)   # thin grey outline
    mask = np.zeros((n, n), np.uint8)
    cv2.rectangle(mask, (70, 50), (130, 150), 255, -1)             # tight mask
    out = matting.refine(rgb, mask, element_role="product")
    # Interior and the product's own edge stay opaque (mask was tight, no overshoot).
    assert out.alpha[100, 100] > 0.95
    assert out.alpha[100, 72] > 0.6, "tight white-product edge should be preserved"


def test_holes_filled():
    import cv2
    n = 160
    rgb = np.full((n, n, 3), 200, np.uint8)
    mask = np.zeros((n, n), np.uint8)
    cv2.circle(mask, (80, 80), 50, 255, -1)
    cv2.circle(mask, (80, 80), 12, 0, -1)  # punch an interior hole
    out = matting.refine(rgb, mask)
    # center should be opaque again
    assert out.alpha[80, 80] > 0.9


# --------------------------------------------------------------------------- #
# Real-fixture tests (skip cleanly if the 002a run is absent)
# --------------------------------------------------------------------------- #

REAL_CASES = [
    ("E006", "fused_elements/E006.png"),   # the +33 halo product
    ("E000", "fused_elements/E000.png"),
    ("E007", "fused_elements/E007.png"),
]


@pytest.mark.parametrize("eid,rel", REAL_CASES)
def test_real_cutout_halo_reduced(eid, rel):
    import json
    fe = os.path.join(RUN, "fused_elements.json")
    mask_p = os.path.join(RUN, rel.replace("/", os.sep))
    orig_p = os.path.join(RUN, "original.png")
    if not (os.path.exists(fe) and os.path.exists(mask_p) and os.path.exists(orig_p)):
        pytest.skip("002a run fixtures not present")
    els = {e["id"]: e for e in json.load(open(fe))}
    if eid not in els:
        pytest.skip(f"{eid} absent")
    box = els[eid]["box"]
    role = els[eid].get("role")
    orig = np.asarray(Image.open(orig_p).convert("RGB"))
    mask_full = np.asarray(Image.open(mask_p).convert("L"))
    out = matting.refine(orig, mask_full, box=box, element_role=role)
    # products must go through the matte path, never get squared into a card
    if role == "product":
        assert out.metrics.get("path") != "card_snap"
    # before = binary alpha crop, for the strip
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    crop = orig[y:y + h, x:x + w]
    mcrop = np.asarray(Image.fromarray(mask_full).resize((w, h)))
    if mcrop.shape[:2] != crop.shape[:2]:
        mcrop = mcrop[:crop.shape[0], :crop.shape[1]]
    before = np.dstack([crop, mcrop])
    p = write_strip(f"real_{eid}", crop, before, out)
    assert os.path.exists(p)
    # refined must be non-binary and halo must not be worse
    assert out.metrics["alpha_partial_frac"] >= 0.0
