"""CPU tests for the plate/ownership fixes (one owner per pixel; clean plates; solid rects).

Three defects from the benchmark forensics (002 product bundle, 009 social screenshot):

  * DOUBLE-RENDER — a native TEXT layer AND a baked raster carrier under it print the same
    content, doubling it (002 "WHEYMILKSHAKE", 009 "geld geld"). The single-ownership audit
    gives the native text sole ownership by erasing its baked duplicate from the carrier.
  * GHOST SILHOUETTES — an opaque raster that still ships as a re-rendered layer
    (``keep_in_background`` + ``target == "image"``) leaves a silhouette in background_clean
    even though it's hidden underneath that layer. The cover pass fills its footprint with the
    plate colour. NOTE (2026-07-16, post-002-forensics correction): a removal-capped /
    ``plate_passthrough`` raster does NOT re-render on top — it is dropped entirely (no src,
    no mask, target="drop"), so its plate pixels ARE the only representation. Covering those
    is what caused the 002 catastrophic seam (an entire 46%-of-canvas panel painted with a
    ring-median colour). Plate-passthrough footprints are therefore NEVER covered — see
    ``test_cover_pass_skips_plate_passthrough`` in test_reconstruct.py.
  * FLAT UI PLATES — Codia ships flat/banded UI as SOLID rects, never a generative inpaint.
    Flat mask holes are solid-filled analytically; only genuine photo holes inpaint.

All CPU-only; no GPU backends are exercised.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import build_design_json, inpaint, reconstruct  # noqa: E402


# ── Fix 1: single-ownership audit (no double render) ─────────────────────────────────

def _baked_plate(path, size=(200, 160), text_box=(40, 60, 150, 80)):
    """White plate with a black 'baked text' ink bar (a carrier that double-prints)."""
    image = Image.new("RGB", size, (255, 255, 255))
    ImageDraw.Draw(image).rectangle(text_box, fill=(10, 10, 10))
    image.save(path)
    return image


def test_native_text_over_baked_carrier_collapses_to_one_owner(tmp_path):
    plate = tmp_path / "background_clean.png"
    _baked_plate(plate)  # baked ink at x40..150 y60..80 on a white page
    # A native text layer covering the same region — without the audit both render.
    candidates = [{
        "id": "c_T0", "target": "text", "text": "HELLO", "z": 5,
        "box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "visible_box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "style": {"fontSize": 18, "fontFamily": "Arial", "color": "#0a0a0a"},
        "meta": {"source": "ocr", "role": "headline"},
    }]
    doc = build_design_json.build(candidates, {"w": 200, "h": 160}, str(tmp_path),
                                  base_src=str(plate))
    audit = doc.meta["single_ownership"]
    assert audit["enabled"] and audit["collapsed"] >= 1

    # The staged background carrier no longer carries the baked ink under the text.
    bg = next(l for l in doc.layers if l.id == "background")
    staged = np.asarray(Image.open(tmp_path / bg.src).convert("RGB"))
    region = staged[60:80, 40:150]
    assert region.min() > 235  # baked ink erased → uniform white plate


def test_single_ownership_leaves_photo_carrier_untouched(tmp_path):
    """A native label over a genuine PHOTO must NOT smear the photo — only uniform
    plate regions are cleaned, textured carriers are left to the raster."""
    plate = tmp_path / "background_clean.png"
    rng = np.random.default_rng(3)
    photo = rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)  # textured everywhere
    Image.fromarray(photo).save(plate)
    before = photo.copy()
    candidates = [{
        "id": "c_T0", "target": "text", "text": "LABEL", "z": 5,
        "box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "visible_box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "style": {"fontSize": 18, "fontFamily": "Arial", "color": "#ffffff"},
        "meta": {"source": "ocr", "role": "headline"},
    }]
    doc = build_design_json.build(candidates, {"w": 200, "h": 160}, str(tmp_path),
                                  base_src=str(plate))
    audit = doc.meta["single_ownership"]
    assert audit["collapsed"] == 0 and audit["textured_skipped"] >= 1
    bg = next(l for l in doc.layers if l.id == "background")
    staged = np.asarray(Image.open(tmp_path / bg.src).convert("RGB"))
    assert np.array_equal(staged, before)  # photo carrier untouched


def test_single_ownership_config_gate(tmp_path):
    plate = tmp_path / "background_clean.png"
    _baked_plate(plate)
    candidates = [{
        "id": "c_T0", "target": "text", "text": "HELLO", "z": 5,
        "box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "visible_box": {"x": 40, "y": 60, "w": 110, "h": 20},
        "style": {"fontSize": 18, "fontFamily": "Arial", "color": "#0a0a0a"},
        "meta": {"source": "ocr", "role": "headline"},
    }]
    doc = build_design_json.build(
        candidates, {"w": 200, "h": 160}, str(tmp_path), base_src=str(plate),
        cfg={"design": {"single_ownership": {"enabled": False}}})
    assert doc.meta["single_ownership"]["enabled"] is False
    bg = next(l for l in doc.layers if l.id == "background")
    staged = np.asarray(Image.open(tmp_path / bg.src).convert("RGB"))
    assert staged[60:80, 40:150].min() < 50  # baked ink preserved when disabled


# ── Fix 3: flat regions → solid fill (not generative inpaint) ────────────────────────

def test_flat_region_is_solid_filled_not_inpainted():
    rgb = np.full((80, 120, 3), (18, 18, 18), dtype=np.uint8)  # flat dark UI plate
    rgb[30:50, 40:80] = (240, 240, 240)  # baked text ink to remove
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 40:80] = 255
    cfg = {"scene": {"archetype": "social_screenshot"}, "inpaint": {"mode": "flux"}}
    out, backend, diag = inpaint.inpaint_array(rgb, mask, cfg, return_diagnostics=True)
    assert backend == "solid-flat"  # never routed to Flux/LaMa
    assert diag["solid_flat"]["flat_filled"] >= 1
    # The hole is filled with the surrounding dark plate colour, not white ink.
    assert np.abs(out[30:50, 40:80].astype(int) - 18).max() <= 6


def test_photo_region_still_inpaints():
    rng = np.random.default_rng(11)
    rgb = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)  # textured photo
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 40:80] = 255
    cfg = {"scene": {"archetype": "social_screenshot"},
           "inpaint": {"mode": "opencv"}}  # opencv so no GPU needed
    out, backend, diag = inpaint.inpaint_array(rgb, mask, cfg, return_diagnostics=True)
    # A textured hole is NOT solid-filled; it goes to the real inpaint backend.
    assert backend != "solid-flat"
    assert diag["solid_flat"]["flat_filled"] == 0
    assert diag["solid_flat"]["remaining_px"] > 0


def test_solid_flat_disabled_for_photo_archetype():
    rgb = np.full((80, 120, 3), (18, 18, 18), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 40:80] = 255
    cfg = {"scene": {"archetype": "lifestyle_overlay"}, "inpaint": {"mode": "opencv"}}
    out, backend, diag = inpaint.inpaint_array(rgb, mask, cfg, return_diagnostics=True)
    assert "solid_flat" not in diag  # flat-fill off for genuine-photo archetypes


def test_solid_flat_explicit_override_forces_on():
    rgb = np.full((80, 120, 3), (18, 18, 18), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 40:80] = 255
    cfg = {"scene": {"archetype": "lifestyle_overlay"},
           "inpaint": {"mode": "opencv", "solid_flat_regions": True}}
    out, backend, diag = inpaint.inpaint_array(rgb, mask, cfg, return_diagnostics=True)
    assert backend == "solid-flat"


# ── Fix 2: clean plates — removal mask ⊇ cutout footprint; ghost silhouette covered ──

def _cutout_source(path, size=(160, 160)):
    image = Image.new("RGB", size, (245, 245, 245))
    ImageDraw.Draw(image).ellipse((40, 40, 120, 120), fill=(180, 60, 40))  # product blob
    image.save(path)


def test_removal_mask_covers_cutout_footprint_with_rim(tmp_path):
    """An opaque product cutout is re-rendered on top, so removal must cover its full
    footprint + anti-aliased rim (otherwise a halo of the original shows)."""
    source = tmp_path / "source.png"
    _cutout_source(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    m = np.zeros((160, 160), np.uint8)
    m[45:115, 45:115] = 255  # tight cutout mask (inside the blob's AA edge)
    Image.fromarray(m).save(masks / "prod.png")
    candidates = [{
        "id": "prod", "target": "image", "z": 1,
        "box": {"x": 40, "y": 40, "w": 80, "h": 80},
        "mask": {"src": "masks/prod.png"},
        "meta": {"role": "product", "confidence": 0.9},
    }]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"},
                                                     "reconstruct": {"cutout_rim_dilate": 4}})
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L")) > 0
    tight = np.zeros((160, 160), bool)
    tight[45:115, 45:115] = True
    # Removal strictly SUPERSETS the tight cutout mask (rim dilation added around it).
    assert removal[tight].all()
    assert int(removal.sum()) > int(tight.sum())


def _capped_scene(tmp_path):
    """Source + candidates where the low-conf blob is genuinely removal-capped (a second
    element keeps _flatten_photo_scene from turning the blob into the scene background)."""
    source = tmp_path / "source.png"
    img = Image.new("RGB", (120, 100), (238, 232, 220))
    ImageDraw.Draw(img).rectangle((5, 5, 104, 44), fill=(30, 90, 160))  # blue ghost blob
    img.save(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (100, 40), 255).save(masks / "blob.png")
    Image.new("L", (100, 40), 255).save(masks / "product.png")
    candidates = [
        {"id": "blob", "target": "image", "z": 0,
         "box": {"x": 5, "y": 5, "w": 100, "h": 40},
         "mask": {"src": "masks/blob.png"},
         "meta": {"role": "photo", "confidence": 0.40}},   # capped, kept in plate
        {"id": "product", "target": "image", "z": 0,
         "box": {"x": 5, "y": 55, "w": 100, "h": 40},
         "mask": {"src": "masks/product.png"},
         "meta": {"role": "product", "confidence": 0.40}},  # exempt, removed
    ]
    return source, candidates


def test_capped_raster_footprint_is_never_covered_plate_owned(tmp_path):
    """A removal-capped raster is dropped (target="drop", no src/mask) and never
    re-rendered, so its plate pixels are the sole representation — the cover pass must
    NEVER touch a plate_passthrough footprint (covering it repaints real plate content
    with a guessed colour; this is the exact 002 seam-catastrophe mechanism). See the
    module docstring's GHOST SILHOUETTES note and test_reconstruct.py's
    test_cover_pass_skips_plate_passthrough for the unit-level invariant."""
    source, candidates = _capped_scene(tmp_path)
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    assert result["stats"]["removal_capped"] == 1
    assert result["stats"]["kept_footprints_covered"] == 0
    plate = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    footprint = plate[8:42, 8:100]
    # The blue blob is plate-owned and untouched by the cover pass — it remains
    # (blob was (30,90,160); a covered plate would read ~(238,232,220) instead).
    assert footprint[:, :, 2].mean() > 120  # blue channel still shows the blob


def test_cover_footprints_config_gate(tmp_path):
    source, candidates = _capped_scene(tmp_path)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"inpaint": {"mode": "opencv"}, "reconstruct": {"cover_kept_footprints": False}})
    assert result["stats"]["removal_capped"] == 1
    assert result["stats"]["kept_footprints_covered"] == 0
    plate = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    assert plate[8:42, 8:100][:, :, 2].mean() > 120  # blue silhouette still present
