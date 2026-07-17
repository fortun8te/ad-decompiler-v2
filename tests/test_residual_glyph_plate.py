"""CPU tests for the 088 "SALE broken E" slice-seam groundwork (commit 20d5373).

Display glyphs a display face defeats OCR on (088 "SALE"/"21%") never become TEXT; they
survive as residual-CC observations and ship as pixel-exact baked raster slices.
Front-to-back ownership hands each slice only PART of its own ink (a boxy sub-region), so
removing the full glyph ink from the plate and re-pasting the fragmented slices leaves
grey plate showing THROUGH the letterforms.  The call site in ``reconstruct`` therefore
withholds plate removal for residual-CC baked glyph slices when ownership under-covers
the glyph (``residual_glyph_ownership_full``) and the plate ring is flat
(``_plate_ring_is_flat``): the plate keeps the original ink and the slices ship on top.

  * ``_residual_baked_raster`` — the provenance sniff: ``fallback: raster-slice``, or an
    ``ownership_cutout`` whose source / provenance sources mention "residual".
  * ``_plate_ring_is_flat`` — the plate-flatness gate: a uniform ring may keep ink; a
    gradient/photographic carrier (ribbon, product) fails closed to the normal removal
    path, as does an ink-dominated ring with too few trustworthy background samples.
  * ``_skip_removal_for_flat_residual_glyph`` + ``_stamp_residual_glyph_overlay`` — the
    call-site decision and stamp, factored out of the ``reconstruct`` loop body so the
    predicate is unit-testable with synthetic candidates/ownership/masks.  Behaviour is
    identical to the inline block they replace.

DOUBLE-DRAW VERDICT (verified against ``apply_raster_slice_fallback`` L4377+ and its
alpha read ``alpha = np.isin(numbers, wanted)`` L4549-4551): the slice alpha THERE is
built from the REMOVAL LEDGER — pixels actually inpainted for this layer id.  A
removal-skipped glyph never enters the ledger (the call site ``continue``s before the
observation append), so ``wanted`` is empty, alpha is None, and the layer falls into the
drop branch ("plate-already-holds-source-pixels").  Within that fallback path kept-ink +
slice-on-top CANNOT double-draw: the ledger alpha excludes plate-kept px by construction.
The slices 088 actually ships (c_E000/c_E002/c_E006/c_E007) are baked ``ownership_cutout``
assets from the main materialization loop, whose alpha is the front-to-back OWNED mask —
that alpha does NOT exclude plate-kept px, so kept ink + slice-on-top DO overlap.  That
double-draw is lossless by construction: both copies are the same original pixels at the
same position under a binary alpha (X over X = X), the same invariant the ownership-rescue
path already relies on ("identical pixels, same position -> no visible double").  Noted
prominently; NO fix attempted here.

All CPU-only; no GPU backends are exercised.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import reconstruct  # noqa: E402


# ── _residual_baked_raster: the provenance sniff ─────────────────────────────────────

def test_residual_baked_raster_accepts_raster_slice_fallback():
    # The canonical 088 shape: a baked slice emitted by the raster-slice fallback.
    meta = {"fallback": "raster-slice", "role": "shape"}
    assert reconstruct._residual_baked_raster(meta) is True


def test_residual_baked_raster_accepts_residual_provenance_cutout():
    # An ownership cutout whose provenance sources carry a residual observation.
    meta = {
        "ownership_cutout": True,
        "source": "element+qwen",
        "provenance": {"sources": ["residual", "sam3:box-refine"]},
    }
    assert reconstruct._residual_baked_raster(meta) is True


def test_residual_baked_raster_rejects_non_residual_cutout():
    # ownership_cutout alone is not enough — a SAM-only provenance is an ordinary
    # semantic cutout and must stay on the normal removal path.
    meta = {
        "ownership_cutout": True,
        "source": "sam3:text-prompt",
        "provenance": {"sources": ["sam3:box-refine"]},
    }
    assert reconstruct._residual_baked_raster(meta) is False


def test_residual_baked_raster_rejects_plain_photo_cutout():
    meta = {"ownership_cutout": True, "source": "element"}
    assert reconstruct._residual_baked_raster(meta) is False


def test_residual_baked_raster_rejects_missing_meta():
    assert reconstruct._residual_baked_raster(None) is False
    assert reconstruct._residual_baked_raster({}) is False


# ── _plate_ring_is_flat: the plate-flatness gate ─────────────────────────────────────

def _flat_plate(h=60, w=60, base=250, noise=2.0, seed=7):
    """Near-uniform plate: per-channel std ~2, well under residual_glyph_flat_std=14."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(base, noise, (h, w, 3)), 0, 255).astype(np.uint8)


def test_flat_plate_ring_is_flat():
    rgb = _flat_plate()
    box = {"x": 20, "y": 20, "w": 16, "h": 16}
    assert reconstruct._plate_ring_is_flat(rgb, box, None, {}) is True


def test_smooth_gradient_ring_is_not_flat():
    # A stepped/two-tone carrier (a ribbon edge crossing the ring): an equal mix of
    # 180/220 gives per-channel std exactly 20 — over the 14.0 flat gate.
    rgb = np.full((60, 60, 3), 180, dtype=np.uint8)
    rgb[:, 30:] = 220
    box = {"x": 22, "y": 22, "w": 16, "h": 16}  # padded ring spans the x=30 seam
    assert reconstruct._plate_ring_is_flat(rgb, box, None, {}) is False


def test_photographic_noise_ring_is_not_flat():
    rng = np.random.default_rng(11)
    rgb = np.clip(rng.normal(128, 40, (60, 60, 3)), 0, 255).astype(np.uint8)
    box = {"x": 20, "y": 20, "w": 16, "h": 16}
    assert reconstruct._plate_ring_is_flat(rgb, box, None, {}) is False


def test_ink_dominated_ring_fails_closed():
    # Ink covers the ring except 20 px: under 30 trustworthy background samples the
    # estimate cannot be trusted (and the contaminating dark ink inflates the std either
    # way) — fail closed to the normal removal path.
    rgb = _flat_plate()
    rgb[12:44, 12:44] = (10, 10, 10)    # the dark glyph ink sitting on the flat plate
    box = {"x": 20, "y": 20, "w": 16, "h": 16}
    ink_mask = np.zeros((60, 60), dtype=np.uint8)
    ink_mask[12:44, 12:44] = 255        # ink fills the whole padded ring (32x32) ...
    ink_mask[12:32, 12] = 0             # ... except a single 20-px sliver of background
    rgb[12:32, 12] = (250, 250, 250)    # that sliver still shows the flat plate
    bg_left = int(np.count_nonzero(ink_mask[12:44, 12:44] == 0))
    assert bg_left < 30
    assert reconstruct._plate_ring_is_flat(rgb, box, ink_mask, {}) is False


def test_ring_with_too_few_pixels_fails_closed():
    # The other fail-closed rail: a tiny padded ring never reaches 30 samples at all.
    rgb = _flat_plate()
    box = {"x": 30, "y": 30, "w": 2, "h": 2}
    assert reconstruct._plate_ring_is_flat(rgb, box, None,
                                           {"residual_glyph_ring_pad": 1}) is False


def test_ring_pad_respects_image_borders():
    # A corner box: the pad clamps to the image edge instead of reading past it.  A
    # bright bar just OUTSIDE the clamped ring must not pollute the estimate...
    rgb = _flat_plate()
    rgb[:, 30:40] = (10, 10, 10)
    box = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert reconstruct._plate_ring_is_flat(rgb, box, None, {}) is True
    # ...while the same bar INSIDE the padded ring (x=12..17 < 10+8) flips the verdict.
    rgb2 = _flat_plate()
    rgb2[:, 12:18] = (10, 10, 10)
    assert reconstruct._plate_ring_is_flat(rgb2, box, None, {}) is False


# ── the call-site decision (factored predicate + stamp) ──────────────────────────────

def _glyph_candidate(role="shape"):
    """A residual-CC baked glyph slice, modelled on 088's c_E000 (role "shape",
    fallback raster-slice, residual provenance, ownership cutout)."""
    return {
        "id": "c_G0", "target": "image",
        "box": {"x": 20, "y": 20, "w": 20, "h": 20},
        "meta": {
            "role": role,
            "fallback": "raster-slice",
            "ownership_cutout": True,
            "source": "element+qwen",
            "provenance": {"sources": ["residual", "sam3:box-refine"]},
        },
    }


def _glyph_scene(owned_rows):
    """Flat plate + a 20x20 glyph mask at (20,20); ``owned_rows`` of its rows are owned
    by the glyph itself (owner number 5), the rest was stolen by another owner (9)."""
    rgb = _flat_plate()
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    ownership = np.full((60, 60), 9, dtype=np.int64)
    ownership[20:20 + owned_rows, 20:40] = 5
    owner_number = {"c_G0": 5}
    return rgb, mask, ownership, owner_number


def test_under_covered_residual_glyph_on_flat_plate_skips_removal():
    # 60% ownership coverage (< 0.995) on a flat ring: withhold removal, stamp the keep.
    rgb, mask, ownership, owner_number = _glyph_scene(owned_rows=12)  # 240/400 px
    cand = _glyph_candidate()
    frac = reconstruct._skip_removal_for_flat_residual_glyph(
        cand, mask, ownership, owner_number, rgb, {})
    assert frac is not None and abs(frac - 0.6) < 1e-6
    reconstruct._stamp_residual_glyph_overlay(cand, frac)
    meta = cand["meta"]
    assert meta["removal_skipped"] == "flat-residual-glyph-kept-in-plate"
    assert meta["overlay_without_removal"] is True
    assert meta["residual_glyph_ownership_fraction"] == 0.6


def test_fully_owned_residual_glyph_removes_normally():
    # Coverage >= residual_glyph_ownership_full: the slice set spans the whole glyph, so
    # there is no ownership gap to fill — normal removal, no skip.
    rgb, mask, ownership, owner_number = _glyph_scene(owned_rows=20)  # 400/400 px
    cand = _glyph_candidate()
    assert reconstruct._skip_removal_for_flat_residual_glyph(
        cand, mask, ownership, owner_number, rgb, {}) is None
    assert "removal_skipped" not in cand["meta"]


def test_photo_fragment_role_never_skips_removal():
    # A photographic fragment on a flat ring still vacates the plate (swap-clean).
    rgb, mask, ownership, owner_number = _glyph_scene(owned_rows=12)
    cand = _glyph_candidate(role="photo-fragment")
    assert reconstruct._skip_removal_for_flat_residual_glyph(
        cand, mask, ownership, owner_number, rgb, {}) is None


def test_banner_and_logo_roles_never_skip_removal():
    # The allowlist is {"", "shape", "icon"} and needs NO extension: in 088's actual
    # reconstruction.json every residual raster-slice fragment is role "shape"
    # (c_E000/c_E002/c_E006/c_E007) and the remaining residual-provenance cutouts are
    # "icon" (c_E003/c_E005/c_E008) — both covered — or "product" (c_E009), which must
    # stay excluded.  "banner"/"logo" simply never occur on 088's residual fragments.
    rgb, mask, ownership, owner_number = _glyph_scene(owned_rows=12)
    for role in ("banner", "logo"):
        cand = _glyph_candidate(role=role)
        assert reconstruct._skip_removal_for_flat_residual_glyph(
            cand, mask, ownership, owner_number, rgb, {}) is None


def test_keep_gate_off_removes_normally():
    # reconstruct.keep_flat_residual_glyphs: false disables the whole keep.
    rgb, mask, ownership, owner_number = _glyph_scene(owned_rows=12)
    cand = _glyph_candidate()
    assert reconstruct._skip_removal_for_flat_residual_glyph(
        cand, mask, ownership, owner_number, rgb,
        {"keep_flat_residual_glyphs": False}) is None


def test_non_flat_plate_removes_normally():
    # Under-covered ownership but a photographic carrier: fail closed to removal.
    rng = np.random.default_rng(11)
    rgb = np.clip(rng.normal(128, 40, (60, 60, 3)), 0, 255).astype(np.uint8)
    _, mask, ownership, owner_number = _glyph_scene(owned_rows=12)
    cand = _glyph_candidate()
    assert reconstruct._skip_removal_for_flat_residual_glyph(
        cand, mask, ownership, owner_number, rgb, {}) is None
