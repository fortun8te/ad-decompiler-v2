import json
import os

import numpy as np

from src.reconstruct import (
    _is_background_plate, _inpaint_used_opencv, _promote_ocr_overlapping_shells,
)


def test_promote_ocr_overlapping_logo_shell_skips_vectorize_path():
    """Reconstruct safety net: logo hosting OCR → shape plate, not icon vectorize."""
    candidates = [
        {
            "id": "c_E014", "target": "icon",
            "box": {"x": 774, "y": 540, "w": 256, "h": 254},
            "meta": {"role": "logo"},
        },
        {
            "id": "c_pct", "target": "text", "text": "45%",
            "box": {"x": 792, "y": 646, "w": 187, "h": 60},
            "meta": {"role": "offer"},
        },
    ]
    n = _promote_ocr_overlapping_shells(candidates, {})
    assert n == 1
    assert candidates[0]["target"] == "shape"
    assert candidates[0]["meta"]["text_bearing_shell"] is True
    assert candidates[0]["meta"]["role"] == "badge"
    assert candidates[1]["meta"]["removal_required"] is True
    assert candidates[1]["meta"]["shell_text_host"] == "c_E014"


def test_promote_does_not_steal_product_packaging_text():
    candidates = [
        {
            "id": "c_E013", "target": "image",
            "box": {"x": 40, "y": 420, "w": 1000, "h": 620},
            "meta": {"role": "product"},
        },
        {
            "id": "c_label", "target": "text", "text": "VANILLE",
            "box": {"x": 200, "y": 700, "w": 200, "h": 40},
            "meta": {"role": "label"},
        },
    ]
    assert _promote_ocr_overlapping_shells(candidates, {}) == 0
    assert candidates[0]["target"] == "image"
    assert candidates[1].get("meta", {}).get("removal_required") is not True


def test_promote_wide_shape_hosting_ocr_becomes_banner_shell():
    """Reconstruct safety net: brushstroke-like shape + inset OCR → banner plate."""
    candidates = [
        {
            "id": "c_E_banner", "target": "shape",
            "box": {"x": 80, "y": 220, "w": 920, "h": 140},
            "meta": {"role": "shape"},
        },
        {
            "id": "c_sold", "target": "text", "text": "ALMOST SOLD OUT...",
            "box": {"x": 160, "y": 255, "w": 760, "h": 70},
            "meta": {"role": "offer"},
        },
    ]
    assert _promote_ocr_overlapping_shells(candidates, {}) == 1
    assert candidates[0]["meta"]["text_bearing_shell"] is True
    assert candidates[0]["meta"]["role"] == "banner"
    assert candidates[1]["meta"]["removal_required"] is True
    assert candidates[1]["meta"]["shell_text_host"] == "c_E_banner"


def test_inpaint_used_opencv_detects_regional_fallback():
    # Regional inpaint reports per-region backends; any opencv-* region is a fallback.
    assert _inpaint_used_opencv({"backend_counts": {"flux-comfy": 2, "opencv-telea": 1}}) is True
    assert _inpaint_used_opencv({"backend_counts": {"flux-comfy": 3, "big-lama": 1}}) is False


def test_inpaint_used_opencv_detects_single_pass_fallback():
    # Single-pass inpaint carries the explicit flag on diagnostics.backend_route.
    assert _inpaint_used_opencv(
        {"backend": "opencv-ns", "diagnostics": {"backend_route": {"opencv_fallback_used": True}}}
    ) is True
    assert _inpaint_used_opencv(
        {"backend": "big-lama", "diagnostics": {"backend_route": {"opencv_fallback_used": False}}}
    ) is False


def test_inpaint_used_opencv_ignores_empty_and_malformed():
    assert _inpaint_used_opencv({"backend": "none", "masked_fraction": 0.0}) is False
    assert _inpaint_used_opencv(None) is False
    assert _inpaint_used_opencv({}) is False


def test_large_edge_touching_product_is_foreground_not_plate():
    # Hero packaging is a foreground product cutout, not a clean plate.  It must remain
    # available for reconstruction and claim its pixels before the broad photo.
    product = {"box": {"x": 10, "y": 536, "w": 1057, "h": 544},
               "meta": {"role": "product"}}
    isolated = {"box": {"x": 100, "y": 100, "w": 700, "h": 700},
                "meta": {"role": "product"}}

    assert not _is_background_plate(product, 1080, 1080)
    assert not _is_background_plate(isolated, 1080, 1080)
from PIL import Image, ImageDraw, ImageFilter

from src import reconstruct


def _source(path, size=(180, 120)):
    image = Image.new("RGB", size, (238, 232, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 48, 135, 65), fill=(20, 20, 20))
    image.save(path)
    return image


def test_text_is_removed_from_background_once(tmp_path):
    source = tmp_path / "source.png"
    original = _source(source)
    candidates = [{
        "id": "c_B0", "target": "text", "text": "SALE", "z": 4,
        "box": {"x": 45, "y": 48, "w": 90, "h": 18},
        "visible_box": {"x": 45, "y": 48, "w": 90, "h": 18},
        "meta": {"source": "ocr", "role": "headline", "line_ids": []},
    }]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"inpaint": {"mode": "opencv", "opencv_radius": 4}},
    )
    clean = Image.open(tmp_path / result["background"]).convert("RGB")
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[55, 80] > 0
    assert clean.getpixel((80, 55))[0] > 100  # dark overlay is gone
    assert clean.getpixel((5, 5)) == original.getpixel((5, 5))  # untouched outside mask
    assert result["stats"]["inpaint"]["backend"] == "opencv-telea"


def test_text_and_large_removals_use_role_aware_backends(tmp_path, monkeypatch):
    source = tmp_path / "role-aware.png"
    _source(source, (180, 120))
    photo_mask = Image.new("L", (60, 40), 255)
    photo_mask.save(tmp_path / "photo-mask.png")
    candidates = [
        {"id": "text", "target": "text", "text": "SALE", "box": {"x": 45, "y": 48, "w": 90, "h": 18},
         "visible_box": {"x": 45, "y": 48, "w": 90, "h": 18}, "meta": {"role": "body"}},
        {"id": "photo", "target": "image", "box": {"x": 110, "y": 70, "w": 60, "h": 40},
         "mask": {"src": "photo-mask.png"}, "meta": {"role": "photo"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv", "opencv_method": "telea"}})
    assert result["stats"]["inpaint"]["backend"] == "role-aware"
    assert [part["role"] for part in result["stats"]["inpaint"]["parts"]] == ["text", "large"]
    assert result["stats"]["inpaint"]["parts"][0]["backend"] == "opencv-telea"


def test_reconstruct_wires_enriched_canonical_observations_to_regional_inpaint(tmp_path, monkeypatch):
    source = tmp_path / "regional.png"
    _source(source, (120, 90))
    product_mask = Image.new("L", (50, 50), 255)
    product_mask.save(tmp_path / "product-mask.png")
    candidates = [
        {"id": "product", "target": "image", "box": {"x": 35, "y": 25, "w": 50, "h": 50},
         "mask": {"src": "product-mask.png"}, "meta": {"role": "product"}},
        {"id": "label", "target": "text", "text": "SALE",
         "box": {"x": 45, "y": 40, "w": 30, "h": 12},
         "visible_box": {"x": 45, "y": 40, "w": 30, "h": 12},
         # Explicit promotion distinguishes editable ad copy from packaging OCR.
         "meta": {"role": "offer", "parent_id": "product", "overlay_text": True}},
    ]
    captured = {}

    def fake_regional(image_path, observations, union, output_path, cfg, run_dir=None):
        captured["observations"] = observations
        captured["union"] = union.copy()
        Image.open(image_path).save(output_path)
        return {"ok": True, "path": output_path, "backend": "regional",
                "strategy": "regional", "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_regional", fake_regional)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"inpaint": {"mode": "opencv", "regional": {"enabled": True}}},
    )

    assert result["stats"]["inpaint"]["backend"] == "regional"
    by_id = {item["id"]: item for item in captured["observations"]}
    assert by_id["product"]["target"] == "image"
    assert by_id["product"]["role"] == "product"
    assert by_id["label"]["parent_id"] == "product"
    assert np.any(captured["union"])


def test_inset_overlay_keeps_valid_photo_underlay_out_of_removal_mask(tmp_path):
    source = tmp_path / "inset.png"
    Image.new("RGB", (120, 90), (40, 110, 150)).save(source)
    Image.new("L", (30, 30), 255).save(tmp_path / "inset-mask.png")
    candidates = [{
        "id": "inset", "target": "image", "box": {"x": 78, "y": 8, "w": 30, "h": 30},
        "mask": {"src": "inset-mask.png"},
        "meta": {"role": "photo", "keep_underlay": True, "circular": True},
    }]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert not removal.any()
    assert result["candidates"][0]["target"] == "image"


def test_explicit_z_band_keeps_chrome_above_smaller_content_overlap(tmp_path):
    """A large UI/header cluster must own its overlap before a smaller photo fragment.

    Area-only ownership used to let the small content crop punch a transparent hole through
    a bigger verified chrome/header asset.  The VLM/SAM z contract is authoritative.
    """
    source = tmp_path / "z-band.png"
    Image.new("RGB", (100, 100), (230, 230, 230)).save(source)
    Image.new("L", (60, 60), 255).save(tmp_path / "chrome-mask.png")
    Image.new("L", (20, 20), 255).save(tmp_path / "content-mask.png")
    candidates = [
        {"id": "chrome", "target": "image", "box": {"x": 20, "y": 20, "w": 60, "h": 60},
         "mask": {"src": "chrome-mask.png"},
         "meta": {"role": "icon", "layer_disposition": "foreground_raster", "z_band": "chrome"}},
        {"id": "content", "target": "image", "box": {"x": 40, "y": 40, "w": 20, "h": 20},
         "mask": {"src": "content-mask.png"},
         "meta": {"role": "photo", "layer_disposition": "foreground_raster", "z_band": "content"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv", "mask_dilate": 0}})
    by_id = {item["id"]: item for item in result["candidates"]}
    chrome = Image.open(tmp_path / by_id["chrome"]["src"]).convert("RGBA")
    content = Image.open(tmp_path / by_id["content"]["src"]).convert("RGBA")
    # Global (50,50) corresponds to local (30,30) / (10,10) respectively.
    assert chrome.getpixel((30, 30))[3] == 255
    assert content.getpixel((10, 10))[3] == 0


def test_photo_card_suppresses_contained_scene_ocr_but_preserves_overlay_copy(tmp_path):
    source = tmp_path / "card.png"
    Image.new("RGB", (140, 100), (80, 90, 100)).save(source)
    Image.new("L", (100, 70), 255).save(tmp_path / "card-mask.png")
    candidates = [
        {"id": "card", "target": "image", "box": {"x": 20, "y": 20, "w": 100, "h": 70},
         "mask": {"src": "card-mask.png"}, "meta": {"role": "photo_card"}},
        {"id": "label", "target": "text", "text": "CANDLE",
         "box": {"x": 50, "y": 55, "w": 35, "h": 10}, "meta": {"source": "ocr"}},
        {"id": "caption", "target": "text", "text": "Real overlay",
         "box": {"x": 30, "y": 28, "w": 60, "h": 12},
         "meta": {"source": "ocr", "overlay_text": True}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv", "mask_dilate": 0}})
    by_id = {c["id"]: c for c in result["candidates"]}
    assert by_id["label"]["target"] == "drop"
    assert by_id["label"]["meta"]["baked_owner_id"] == "card"
    assert by_id["caption"]["target"] == "text"


def test_intentional_raster_cluster_uses_full_source_crop_and_keeps_only_positive_overlay(tmp_path):
    source = tmp_path / "receipt-source.png"
    image = Image.new("RGB", (140, 100), (230, 230, 230))
    ImageDraw.Draw(image).rectangle((20, 20, 119, 79), fill=(40, 120, 190))
    image.save(source)
    # A loose circular SAM matte must not erase source-crop corners for a receipt/UI/table.
    matte = Image.new("L", (100, 60), 0)
    ImageDraw.Draw(matte).ellipse((12, 2, 87, 57), fill=255)
    matte.save(tmp_path / "loose-mask.png")
    candidates = [
        {"id": "receipt", "target": "image", "box": {"x": 20, "y": 20, "w": 100, "h": 60},
         "mask": {"kind": "rrect", "radius": 0, "src": "loose-mask.png"},
         "meta": {"role": "receipt", "intentional_raster_cluster": True}},
        {"id": "printed", "target": "text", "text": "subtotal",
         "box": {"x": 35, "y": 44, "w": 45, "h": 10}, "meta": {"source": "ocr"}},
        {"id": "offer", "target": "text", "text": "External sale",
         "box": {"x": 30, "y": 26, "w": 70, "h": 10},
         "meta": {"source": "ocr", "overlay_text": True, "parent_id": "receipt"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv", "mask_dilate": 0}})
    by_id = {item["id"]: item for item in result["candidates"]}
    receipt = Image.open(tmp_path / by_id["receipt"]["src"]).convert("RGBA")
    assert receipt.getpixel((0, 0))[3] == 255
    assert receipt.getpixel((99, 59))[3] == 255
    assert by_id["receipt"]["mask"]["kind"] == "rrect"
    assert by_id["printed"]["target"] == "drop"
    assert by_id["printed"]["meta"]["baked_owner_id"] == "receipt"
    assert by_id["offer"]["target"] == "text"


def test_comparison_policy_keeps_contained_column_copy_editable(tmp_path):
    source = tmp_path / "comparison-text.png"
    Image.new("RGB", (200, 120), (80, 90, 100)).save(source)
    Image.new("L", (100, 120), 255).save(tmp_path / "right-mask.png")
    candidates = [
        {"id": "right-photo", "target": "image",
         "box": {"x": 100, "y": 0, "w": 100, "h": 120},
         "mask": {"src": "right-mask.png"}, "meta": {"role": "person"}},
        {"id": "after-copy", "target": "text", "text": "AFTER\nClear conversations",
         "box": {"x": 115, "y": 55, "w": 75, "h": 35},
         "meta": {"role": "body-copy"}},
    ]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True},
                   "preset": {"photo_regions": {
            "suppress_descendants": False,
        }}}, "inpaint": {"mode": "opencv"}},
    )
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["after-copy"]["target"] == "text"
    assert by_id["after-copy"]["meta"].get("suppression_reason") is None


def test_product_label_ocr_is_baked_unless_explicitly_promoted(tmp_path):
    source = tmp_path / "product.png"
    Image.new("RGB", (100, 100), "white").save(source)
    Image.new("L", (50, 70), 255).save(tmp_path / "product-mask.png")
    candidates = [
        {"id": "can", "target": "image", "box": {"x": 25, "y": 15, "w": 50, "h": 70},
         "mask": {"src": "product-mask.png"}, "meta": {"role": "product"}},
        {"id": "brand", "target": "text", "text": "BRAND",
         "box": {"x": 35, "y": 40, "w": 30, "h": 10}, "meta": {"source": "ocr"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    brand = next(c for c in result["candidates"] if c["id"] == "brand")
    assert brand["target"] == "drop"
    assert brand["meta"]["kept_in_photo"] is True


def test_comparison_grid_drops_standalone_before_after_labels(tmp_path):
    source = tmp_path / "comparison-labels.png"
    Image.new("RGB", (240, 160), (98, 127, 102)).save(source)
    Image.new("L", (90, 120), 255).save(tmp_path / "before-mask.png")
    Image.new("L", (90, 120), 255).save(tmp_path / "after-mask.png")
    candidates = [
        {"id": "before-photo", "target": "image",
         "box": {"x": 20, "y": 30, "w": 90, "h": 120},
         "mask": {"src": "before-mask.png"},
         "meta": {"role": "photo", "semantic_name": "Before image", "comparison_side": "before"}},
        {"id": "after-photo", "target": "image",
         "box": {"x": 130, "y": 30, "w": 90, "h": 120},
         "mask": {"src": "after-mask.png"},
         "meta": {"role": "photo", "semantic_name": "After image", "comparison_side": "after"}},
        {"id": "before-label", "target": "text", "text": "Before",
         "box": {"x": 40, "y": 36, "w": 70, "h": 24}, "meta": {"role": "headline"}},
        {"id": "after-label", "target": "text", "text": "After",
         "box": {"x": 150, "y": 36, "w": 60, "h": 24}, "meta": {"role": "headline"}},
        {"id": "headline", "target": "text", "text": "Perfect curls",
         "box": {"x": 30, "y": 8, "w": 180, "h": 18}, "meta": {"role": "headline"}},
    ]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True}},
         "inpaint": {"mode": "opencv"}},
    )
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["before-label"]["target"] == "drop"
    assert by_id["after-label"]["target"] == "drop"
    assert by_id["headline"]["target"] == "text"


def test_photo_heavy_preset_flattens_unverified_fragments_but_keeps_text():
    candidates = [
        {"id": "person", "target": "image", "meta": {"role": "person"}},
        {"id": "inset", "target": "image", "meta": {"role": "photo"}},
        {"id": "headline", "target": "text", "text": "Beach ready", "meta": {}},
        {"id": "noise", "target": "text", "text": "\ufffd ", "meta": {}},
        {"id": "tiny-noise", "target": "text", "text": "cm", "box": {"w": 40, "h": 12},
         "meta": {"confidence": 0.45}},
        {"id": "vertical-package", "target": "text", "text": "Cadence",
         "box": {"w": 30, "h": 90}, "meta": {"confidence": 0.95}},
        {"id": "verified", "target": "image", "meta": {"verified_mask": True}},
        {"id": "exact-text", "target": "image", "text": "Display headline", "meta": {
            "fallback": True,
            "substitution": {"from": "text", "to": "image"},
        }},
    ]
    result, count = reconstruct._flatten_photo_scene(
        candidates, {"scene": {"archetype": "lifestyle_overlay", "preset": {
            "photo_regions": {"flatten_scene_artwork": True},
        }}},
    )
    by_id = {item["id"]: item for item in result}

    assert count == 0
    assert by_id["person"]["target"] == "image"
    assert by_id["inset"]["target"] == "image"
    assert by_id["headline"]["target"] == "text"
    assert by_id["noise"]["target"] == "drop"
    assert by_id["noise"]["meta"]["suppression_reason"] == "invalid-photo-scene-ocr"
    assert by_id["tiny-noise"]["target"] == "drop"
    assert by_id["vertical-package"]["target"] == "drop"
    assert by_id["verified"]["target"] == "image"
    assert by_id["exact-text"]["target"] == "image"


def test_low_confidence_residual_photo_fragment_stays_in_large_scene_plate():
    candidates = [{
        "id": "water-speck", "target": "image", "box": {"x": 80, "y": 50, "w": 120, "h": 60},
        "meta": {"role": "photo", "confidence": .38,
                 "provenance": {"observations": [{"source": "residual", "score": .38}]}},
    }]
    result, _ = reconstruct._flatten_photo_scene(candidates, {"canvas": {"w": 1080, "h": 1080}})
    assert result[0]["target"] == "drop"
    assert result[0]["meta"]["suppression_reason"] == "low-confidence-residual-photo-fragment"


def test_sam_verified_product_matte_is_retained_even_when_small():
    candidates = [{
        "id": "tube", "target": "image", "box": {"x": 80, "y": 50, "w": 120, "h": 300},
        "meta": {"role": "photo", "confidence": .38, "provenance": {"observations": [
            {"source": "sam3", "mask_quality": "mask", "score": .91, "role": "product"},
        ]}},
    }]
    result, _ = reconstruct._flatten_photo_scene(candidates, {"canvas": {"w": 1080, "h": 1920}})
    assert result[0]["target"] == "image"


def test_comparison_grid_photo_exports_before_and_after_as_separate_crops(tmp_path):
    source = tmp_path / "comparison.png"
    image = Image.new("RGB", (200, 160), "white")
    ImageDraw.Draw(image).rectangle((20, 30, 99, 129), fill=(180, 50, 40))
    ImageDraw.Draw(image).rectangle((100, 30, 179, 129), fill=(40, 160, 90))
    image.save(source)
    masks = tmp_path / "masks"; masks.mkdir()
    Image.new("L", (160, 100), 255).save(masks / "comparison.png")
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [{
            "id": "comparison", "target": "image", "box": {"x": 20, "y": 30, "w": 160, "h": 100},
            "mask": {"kind": "alpha", "src": "masks/comparison.png"},
            "meta": {"role": "photo", "confidence": .95},
        }], str(tmp_path), {
            "canvas": {"w": 200, "h": 160},
            "scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True}},
            "inpaint": {"mode": "opencv"},
        })
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["comparison"]["target"] == "drop"
    assert by_id["comparison"]["meta"]["removal_required"] is True
    assert by_id["comparison-before"]["meta"]["comparison_side"] == "before"
    assert by_id["comparison-after"]["meta"]["comparison_side"] == "after"
    assert by_id["comparison-before"]["box"]["w"] == 80
    assert Image.open(tmp_path / by_id["comparison-before"]["src"]).size == (80, 100)


def test_full_bleed_comparison_plate_exports_two_swappable_clean_bases(tmp_path):
    source = tmp_path / "full-comparison.png"
    image = Image.new("RGB", (200, 120), (30, 40, 50))
    ImageDraw.Draw(image).rectangle((100, 0, 199, 119), fill=(70, 80, 90))
    image.save(source)
    Image.new("L", (200, 120), 255).save(tmp_path / "full-mask.png")
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [{
            "id": "plate", "target": "image", "box": {"x": 0, "y": 0, "w": 200, "h": 120},
            "mask": {"src": "full-mask.png"}, "meta": {"role": "photo"},
        }], str(tmp_path), {
            "scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True},
                      "preset": {"photo_regions": {
                "suppress_descendants": False,
            }}}, "inpaint": {"mode": "opencv"},
        },
    )
    by_id = {item["id"]: item for item in result["candidates"]}
    before = by_id["comparison-plate-before"]
    after = by_id["comparison-plate-after"]
    assert before["meta"]["swappable"] is True
    assert after["box"] == {"x": 100, "y": 0, "w": 100, "h": 120}
    assert Image.open(tmp_path / before["src"]).size == (100, 120)


def test_platform_lockup_is_kept_as_a_separate_raster_asset():
    candidates = [{
        "id": "x-lockup", "target": "image", "box": {"x": 820, "y": 40, "w": 130, "h": 32},
        "meta": {"role": "platform-logo", "wordmark": True, "confidence": .9},
    }]
    result, _ = reconstruct._flatten_photo_scene(candidates, {"canvas": {"w": 1080, "h": 1920}})
    assert result[0]["target"] == "image"


def test_scene_vlm_mode_never_inpaints_text_without_ownership(tmp_path):
    source = tmp_path / "source.png"
    _source(source)
    candidates = [{
        "id": "headline", "target": "text", "text": "SALE",
        "box": {"x": 10, "y": 10, "w": 45, "h": 18}, "meta": {},
    }]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"vlm": {"scene_text": {"enabled": True}}, "inpaint": {"mode": "opencv"}},
    )
    headline = next(c for c in result["candidates"] if c["id"] == "headline")
    assert headline["target"] == "drop"
    assert headline["meta"]["mask_approval"]["reason"] == "missing-ownership-decision"
    assert result["stats"]["mask_rejected"] == 1


def test_duplicate_observations_collapse_before_asset_work(tmp_path):
    source = tmp_path / "source.png"
    _source(source)
    mask_dir = tmp_path / "fused_elements"
    mask_dir.mkdir()
    Image.new("L", (40, 30), 255).save(mask_dir / "E0.png")
    base = {
        "target": "shape", "box": {"x": 10, "y": 10, "w": 40, "h": 30},
        "kind": "shape", "mask": {"kind": "alpha", "src": "fused_elements/E0.png"},
        "meta": {"role": "button", "source": "sam3", "confidence": .9},
    }
    candidates = [{"id": "a", **base}, {"id": "b", **base}]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    assert result["stats"]["canonical_entities"] == 1
    assert result["stats"]["duplicates_removed"] == 1


def test_run_relative_qwen_asset_is_cropped_and_staged(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 100))
    qwen = tmp_path / "qwen_layers"
    qwen.mkdir()
    layer = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle((20, 30, 59, 69), fill=(220, 30, 20, 255))
    layer.save(qwen / "Q0.png")
    candidate = {
        "id": "product", "target": "image", "src": "qwen_layers/Q0.png",
        "box": {"x": 20, "y": 30, "w": 40, "h": 40},
        "mask": {"kind": "alpha", "src": "qwen_layers/Q0.png"},
        "meta": {"role": "product", "source": "qwen", "confidence": .8},
    }
    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    compiled = result["candidates"][0]
    assert compiled["src"].startswith("assets/")
    assert Image.open(tmp_path / compiled["src"]).size == (40, 40)


def test_transparent_qwen_canvas_uses_alpha_not_hidden_rgb_for_removal(tmp_path):
    """Transparent white pixels must never cause a destructive full-canvas inpaint."""
    source = tmp_path / "source.png"
    _source(source, (100, 100))
    qwen = tmp_path / "qwen_layers"
    qwen.mkdir()
    # RGB is white even where alpha is zero, which is common in generated cutouts.
    layer = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    ImageDraw.Draw(layer).rectangle((20, 30, 59, 69), fill=(255, 255, 255, 255))
    layer.save(qwen / "Q-white-transparent.png")
    candidate = {
        "id": "product", "target": "image", "src": "qwen_layers/Q-white-transparent.png",
        "box": {"x": 20, "y": 30, "w": 40, "h": 40},
        "mask": {"kind": "alpha", "src": "qwen_layers/Q-white-transparent.png"},
        "meta": {"role": "product", "source": "qwen", "confidence": .8},
    }
    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[0, 0] == 0
    assert removal[45, 40] > 0


def test_overlapping_raster_assets_are_exclusive_after_ownership_assignment(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 80))
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (30, 30), 255).save(masks / "back.png")
    Image.new("L", (30, 30), 255).save(masks / "front.png")
    candidates = [
        {"id": "back", "target": "image", "z": 0,
         "box": {"x": 10, "y": 10, "w": 30, "h": 30},
         "mask": {"src": "masks/back.png"}, "meta": {"role": "product"}},
        {"id": "front", "target": "image", "z": 2,
         "box": {"x": 20, "y": 20, "w": 30, "h": 30},
         "mask": {"src": "masks/front.png"}, "meta": {"role": "product"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    by_id = {item["id"]: item for item in result["candidates"]}
    back = np.asarray(Image.open(tmp_path / by_id["back"]["src"]).convert("RGBA"))
    front = np.asarray(Image.open(tmp_path / by_id["front"]["src"]).convert("RGBA"))
    # Canvas (25, 25) is represented by (15, 15) in the back crop and (5, 5) in front.
    assert back[15, 15, 3] == 0
    assert front[5, 5, 3] > 0


def test_dropped_background_plate_cannot_claim_product_ownership(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 80))
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (100, 80), 255).save(masks / "plate.png")
    Image.new("L", (20, 20), 255).save(masks / "product.png")
    candidates = [
        {"id": "plate", "target": "image", "z": 99,
         "box": {"x": 0, "y": 0, "w": 100, "h": 80},
         "mask": {"src": "masks/plate.png"}, "meta": {"role": "background"}},
        {"id": "product", "target": "image", "z": 0,
         "box": {"x": 30, "y": 20, "w": 20, "h": 20},
         "mask": {"src": "masks/product.png"}, "meta": {"role": "product"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["plate"]["target"] == "drop"
    product = np.asarray(Image.open(tmp_path / by_id["product"]["src"]).convert("RGBA"))
    assert product[:, :, 3].max() == 255


def test_edge_touching_product_is_foreground_not_background_plate(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 80))
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (90, 40), 255).save(masks / "photo.png")
    Image.new("L", (100, 40), 255).save(masks / "product.png")
    candidates = [
        {"id": "photo", "target": "image", "z": 0,
         "box": {"x": 5, "y": 40, "w": 90, "h": 40},
         "mask": {"src": "masks/photo.png"}, "meta": {"role": "photo"}},
        {"id": "product", "target": "image", "z": 0,
         "box": {"x": 0, "y": 40, "w": 100, "h": 40},
         "mask": {"src": "masks/product.png"}, "meta": {"role": "product"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["product"]["target"] == "image"
    assert by_id["product"]["meta"].get("keep_in_background") is not True
    assert by_id["photo"]["meta"].get("keep_in_background") is not True


def test_product_owns_overlap_before_broad_photo_without_z(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 80))
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (80, 40), 255).save(masks / "photo.png")
    Image.new("L", (40, 40), 255).save(masks / "product.png")
    candidates = [
        {"id": "photo", "target": "image", "z": 0,
         "box": {"x": 10, "y": 20, "w": 80, "h": 40},
         "mask": {"src": "masks/photo.png"}, "meta": {"role": "photo"}},
        {"id": "product", "target": "image", "z": 0,
         "box": {"x": 30, "y": 20, "w": 40, "h": 40},
         "mask": {"src": "masks/product.png"}, "meta": {"role": "product"}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    by_id = {item["id"]: item for item in result["candidates"]}
    photo = np.asarray(Image.open(tmp_path / by_id["photo"]["src"]).convert("RGBA"))
    product = np.asarray(Image.open(tmp_path / by_id["product"]["src"]).convert("RGBA"))
    assert photo[20, 40, 3] == 0
    assert product[20, 10, 3] > 0


def _shape_candidate(tmp_path, name, box, mask):
    masks = tmp_path / "style_masks"
    masks.mkdir(exist_ok=True)
    path = masks / f"{name}.png"
    mask.save(path)
    return {
        "id": name, "target": "shape", "box": box,
        "mask": {"src": f"style_masks/{name}.png"},
        "meta": {"role": "button", "source": "sam3", "confidence": .95},
    }


def test_shape_style_extracts_gradient_and_keeps_native_paint_fields(tmp_path):
    source = tmp_path / "gradient.png"
    image = Image.new("RGB", (140, 100), "white")
    pixels = np.asarray(image).copy()
    # 0 degrees is left -> right in the local preview and Figma compiler contract.
    for x in range(20, 120):
        mix = (x - 20) / 99
        pixels[25:75, x] = (round(245 * (1 - mix) + 35 * mix), round(70 * (1 - mix) + 110 * mix),
                            round(85 * (1 - mix) + 220 * mix))
    Image.fromarray(pixels).save(source)
    mask = Image.new("L", (100, 50), 255)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_shape_candidate(tmp_path, "gradient", {"x": 20, "y": 25, "w": 100, "h": 50}, mask)], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},
    )
    shape = result["candidates"][0]
    assert shape["shape_kind"] == "rect"
    assert shape["fill"]["kind"] == "linear"
    assert len(shape["fill"]["stops"]) == 2
    assert abs(shape["fill"]["angle"]) < 5
    assert shape["meta"]["style_extraction"]["gradient"]["r2"] > .95


def test_shape_style_extracts_only_strong_centered_radial_gradient(tmp_path):
    source = tmp_path / "radial-gradient.png"
    h, w = 140, 180
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    radius = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
    start = np.array([250, 220, 80], dtype=float)
    end = np.array([35, 55, 180], dtype=float)
    pixels = np.clip(start[None, None, :] + radius[:, :, None] * (end - start), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(source)
    mask = Image.new("L", (w, h), 255)

    result = reconstruct.reconstruct(
        str(source), {"lines": []},
        [_shape_candidate(tmp_path, "radial", {"x": 0, "y": 0, "w": w, "h": h}, mask)],
        str(tmp_path), {"inpaint": {"mode": "opencv"}},
    )

    shape = result["candidates"][0]
    assert shape["fill"]["kind"] == "radial"
    assert shape["meta"]["style_extraction"]["gradient"]["r2"] > .98
    assert shape["meta"]["style_extraction"]["gradient"]["center"] == [0.5, 0.5]


def test_off_center_fuzzy_field_is_not_claimed_as_native_radial():
    h, w = 100, 140
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(xx - 15, yy - 20) / np.hypot(w, h)
    # Add an angular component: this is a soft lighting/texture field, not the exact
    # centered circular paint supported by the compiler.
    values = 220 - 100 * radius + 28 * np.sin(xx / 9) * np.cos(yy / 11)
    rgb = np.stack((values, values * .8, 255 - values * .45), axis=2).clip(0, 255).astype(np.uint8)
    radial = reconstruct._radial_gradient_fill(rgb, np.ones((h, w), dtype=bool))

    assert radial is None


def test_shape_style_extracts_stroke_and_real_corner_radius(tmp_path):
    source = tmp_path / "rounded-stroke.png"
    image = Image.new("RGB", (140, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((25, 25, 114, 84), radius=12, fill="#101820", outline="#050505", width=3)
    image.save(source)
    mask = Image.new("L", (90, 60), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 89, 59), radius=12, fill=255)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_shape_candidate(tmp_path, "rounded", {"x": 25, "y": 25, "w": 90, "h": 60}, mask)], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},
    )
    shape = result["candidates"][0]
    assert shape["stroke"]["color"] == "#050505"
    assert 1 <= shape["stroke"]["width"] <= 4
    assert 8 <= shape["radius"] <= 14
    assert shape["fill"]["kind"] == "flat"
    assert shape["meta"]["style_extraction"]["stroke_detected"] is True


def test_shape_style_extracts_shadow_only_on_flat_backdrop(tmp_path):
    source = tmp_path / "shadow.png"
    image = Image.new("RGBA", (150, 110), "white")
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((31, 34, 120, 83), radius=10, fill=(0, 0, 0, 105))
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(4)))
    ImageDraw.Draw(image).rounded_rectangle((25, 27, 114, 76), radius=10, fill="#ec5a3c")
    image.convert("RGB").save(source)
    mask = Image.new("L", (90, 50), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 89, 49), radius=10, fill=255)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_shape_candidate(tmp_path, "shadow", {"x": 25, "y": 27, "w": 90, "h": 50}, mask)], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},
    )
    effects = result["candidates"][0].get("effects") or []
    assert effects and effects[0]["type"] == "drop-shadow"
    assert effects[0]["y"] > 0


def test_style_extraction_refuses_shadow_on_non_uniform_scene(tmp_path):
    """A nearby image boundary must not turn into a made-up Figma shadow."""
    source = tmp_path / "busy.png"
    data = np.zeros((100, 140, 3), dtype=np.uint8)
    data[:, :, 0] = np.arange(140, dtype=np.uint8)[None, :]
    data[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]
    data[:, :, 2] = 80
    data[25:75, 20:120] = (230, 70, 60)
    Image.fromarray(data).save(source)
    mask = Image.new("L", (100, 50), 255)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_shape_candidate(tmp_path, "no-shadow", {"x": 20, "y": 25, "w": 100, "h": 50}, mask)], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},
    )
    assert not result["candidates"][0].get("effects")


def test_button_and_text_both_removed_while_text_stays_editable(tmp_path):
    """Button shell + label must leave a clean plate; text remains an editable layer."""
    source = tmp_path / "cta.png"
    image = Image.new("RGB", (200, 100), (245, 240, 230))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 30, 159, 69), radius=10, fill=(32, 96, 220))
    draw.rectangle((70, 42, 130, 58), fill=(255, 255, 255))
    image.save(source)

    masks = tmp_path / "cta_masks"
    masks.mkdir()
    button_mask = Image.new("L", (120, 40), 0)
    ImageDraw.Draw(button_mask).rounded_rectangle((0, 0, 119, 39), radius=10, fill=200)
    button_mask.save(masks / "button.png")

    candidates = [
        {
            "id": "btn", "target": "shape", "z": 1,
            "box": {"x": 40, "y": 30, "w": 120, "h": 40},
            "mask": {"src": "cta_masks/button.png"},
            "meta": {"role": "button", "source": "sam3", "confidence": .95},
        },
        {
            "id": "label", "target": "text", "text": "SHOP", "z": 3,
            "box": {"x": 70, "y": 42, "w": 60, "h": 16},
            "visible_box": {"x": 70, "y": 42, "w": 60, "h": 16},
            "meta": {"role": "cta", "source": "ocr", "line_ids": []},
        },
    ]
    cfg = {"inpaint": {"mode": "opencv", "opencv_radius": 8,
                       "mask_dilate": {"button": 4, "text": 2}}}
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path), cfg)
    by_id = {item["id"]: item for item in result["candidates"]}

    assert by_id["label"]["target"] == "text"
    assert by_id["btn"]["target"] == "shape"

    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    clean = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    source_rgb = np.asarray(Image.open(source).convert("RGB"))

    assert removal[50, 100] > 0  # button + label overlap in removal union
    plate = source_rgb[5, 5]
    button_px = source_rgb[50, 100]
    clean_px = clean[50, 100]
    assert np.linalg.norm(clean_px.astype(float) - plate.astype(float)) < (
        np.linalg.norm(button_px.astype(float) - plate.astype(float)) * 0.55
    )
    assert np.all(clean[removal == 0] == source_rgb[removal == 0])


def test_drop_overlay_text_is_removed_before_editable_redraw(tmp_path):
    source = tmp_path / "overlay.png"
    image = Image.new("RGB", (80, 40), (220, 220, 220))
    ImageDraw.Draw(image).rectangle((20, 12, 59, 27), fill=(20, 20, 20))
    image.save(source)
    candidate = {"id": "overlay", "target": "drop", "text": "SALE", "z": 2,
                 "box": {"x": 20, "y": 12, "w": 40, "h": 16},
                 "meta": {"role": "headline", "overlay_text": True,
                          "removal_required": True}}
    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path),
                                     {"inpaint": {"mode": "opencv", "mask_dilate": 0}})
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[20, 40] > 0


def test_overlay_text_removal_uses_candidate_box_not_stale_ocr_line(tmp_path):
    source = tmp_path / "overlay-misaligned.png"
    image = Image.new("RGB", (100, 60), (220, 220, 220))
    ImageDraw.Draw(image).text((60, 20), "SALE", fill=(20, 20, 20))
    image.save(source)
    candidate = {
        "id": "overlay", "target": "text", "text": "SALE",
        "box": {"x": 60, "y": 20, "w": 30, "h": 15},
        "meta": {"role": "headline", "overlay_text": True,
                 "removal_required": True, "line_ids": ["stale"]},
    }
    result = reconstruct.reconstruct(
        str(source), {"lines": [{"id": "stale", "box": {"x": 5, "y": 5, "w": 20, "h": 10}}]},
        [candidate], str(tmp_path), {"inpaint": {"mode": "opencv", "mask_dilate": 0}},
    )
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[20:35, 60:90].any()
    assert removal[10, 15] == 0


def test_body_text_removal_uses_candidate_box_not_stale_ocr_lines(tmp_path):
    source = tmp_path / "body-stale.png"
    image = Image.new("RGB", (120, 70), (220, 220, 220))
    ImageDraw.Draw(image).text((70, 20), "BODY", fill=(20, 20, 20))
    image.save(source)
    candidate = {"id": "body", "target": "text", "text": "BODY",
                 "box": {"x": 68, "y": 18, "w": 45, "h": 18},
                 "meta": {"role": "body", "line_ids": ["stale"]}}
    result = reconstruct.reconstruct(
        str(source), {"lines": [{"id": "stale", "box": {"x": 2, "y": 2, "w": 30, "h": 15}}]},
        [candidate], str(tmp_path), {"inpaint": {"mode": "opencv", "mask_dilate": 0}},
    )
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[18:36, 68:113].any()
    assert removal[8, 15] == 0


def test_photo_mask_uses_zero_dilate_to_protect_surroundings(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (100, 100))
    qwen = tmp_path / "qwen_layers"
    qwen.mkdir()
    layer = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse((30, 30, 69, 69), fill=(200, 40, 40, 255))
    layer.save(qwen / "product.png")
    candidate = {
        "id": "product", "target": "image", "src": "qwen_layers/product.png",
        "box": {"x": 30, "y": 30, "w": 40, "h": 40},
        "mask": {"kind": "alpha", "src": "qwen_layers/product.png"},
        "meta": {"role": "product", "source": "qwen", "confidence": .8},
    }
    cfg = {"inpaint": {"mode": "opencv", "mask_dilate": {"photo": 0, "image": 2}, "mask_feather": 0}}
    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path), cfg)
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))

    assert removal[50, 50] > 0
    assert removal[25, 50] == 0
    assert removal[75, 50] == 0


def test_soft_product_alpha_is_solidified_before_inpaint(tmp_path):
    source = tmp_path / "source.png"
    _source(source, (80, 80))
    qwen = tmp_path / "qwen_layers"
    qwen.mkdir()
    layer = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse((20, 20, 59, 59), fill=(180, 30, 30, 80))
    layer.save(qwen / "soft.png")
    candidate = {
        "id": "product", "target": "image", "src": "qwen_layers/soft.png",
        "box": {"x": 20, "y": 20, "w": 40, "h": 40},
        "mask": {"kind": "alpha", "src": "qwen_layers/soft.png"},
        "meta": {"role": "product", "source": "qwen", "confidence": .8},
    }
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [candidate], str(tmp_path),
        {"inpaint": {"mode": "opencv", "mask_dilate": {"photo": 0}, "mask_feather": 0}},
    )
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))

    assert set(np.unique(removal)).issubset({0, 255})
    assert removal[40, 40] == 255


def test_list_provenance_is_not_treated_as_verified_sam_evidence():
    assert reconstruct._verified_semantic_mask({"provenance": []}) is False


def test_text_mask_stays_ink_shaped_by_default():
    image = np.full((200, 300, 3), 230, dtype=np.uint8)
    image[80:96, 120:125] = 20
    candidate = {
        "id": "thin-copy", "target": "text", "text": "I",
        "box": {"x": 110, "y": 75, "w": 50, "h": 25},
        "meta": {"role": "body"},
    }

    mask = reconstruct._candidate_mask(candidate, image, None, cfg={})
    box_area = candidate["box"]["w"] * candidate["box"]["h"]

    assert 0 < np.count_nonzero(mask) < box_area * 0.35


def test_text_box_promotion_is_explicit_opt_in():
    image = np.full((200, 300, 3), 230, dtype=np.uint8)
    image[80:96, 120:125] = 20
    candidate = {
        "id": "thin-copy", "target": "text", "text": "I",
        "box": {"x": 110, "y": 75, "w": 50, "h": 25},
        "meta": {"role": "body"},
    }

    mask = reconstruct._candidate_mask(
        candidate, image, None,
        cfg={"reconstruct": {"text_box_promote_max_fraction": 0.06}},
    )

    assert np.count_nonzero(mask) > candidate["box"]["w"] * candidate["box"]["h"]


def test_removal_ledger_makes_overlapping_regions_exclusive():
    left = np.zeros((50, 80), dtype=np.uint8)
    right = np.zeros_like(left)
    left[10:35, 10:45] = 255
    right[20:45, 30:70] = 255
    observations = [
        {"id": "back", "target": "image", "role": "product", "z": 1,
         "box": {"x": 10, "y": 10, "w": 35, "h": 25},
         "mask_array": left, "dilate": 0},
        {"id": "front", "target": "text", "role": "headline", "z": 2,
         "box": {"x": 30, "y": 20, "w": 40, "h": 25},
         "mask_array": right, "dilate": 0},
    ]

    records, union, ledger, owner_index = reconstruct._build_removal_ledger(
        observations, (80, 50),
    )

    assert len(records) == 2
    assert set(owner_index.values()) == {"front", "back"}
    assert np.array_equal(union > 0, (left > 0) | (right > 0))
    assert set(np.unique(ledger)) == {0, 1, 2}
    assert not np.any((records[0]["mask_array"] > 0) & (records[1]["mask_array"] > 0))


# ── Button/pill plate fidelity (009 "Volgend" failure class) ───────────────────────


def _pill_mask(w=202, h=67, holes=()):
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=(h - 1) // 2, fill=255)
    for x0, y0, x1, y1 in holes:
        draw.rectangle((x0, y0, x1, y1), fill=0)
    return np.asarray(image) > 16


def test_corner_radius_full_pill_snaps_to_half_height():
    # A fully-rounded pill (radius == h/2) is the most common ad button. The old
    # first-occupied-edge scan clamped it below min(h, w) * .48 and shipped a
    # sharp rectangle instead.
    assert reconstruct._corner_radius(_pill_mask()) == 33.5


def test_corner_radius_survives_debris_outside_the_corners():
    # 009's "Volgend" pill mask carried a one-row residual ledge welded to the
    # mid-bottom of its silhouette; the old whole-row scan returned None for it.
    mask = _pill_mask().copy()
    mask[-1, :] = False
    mask[-1, 72:114] = True
    assert reconstruct._corner_radius(mask) == 33.5


def test_corner_radius_moderate_rounding_is_not_snapped_to_pill():
    image = Image.new("L", (160, 80), 0)
    ImageDraw.Draw(image).rounded_rectangle((0, 0, 159, 79), radius=12, fill=255)
    radius = reconstruct._corner_radius(np.asarray(image) > 16)
    assert isinstance(radius, float)
    assert 9 <= radius <= 14


def test_corner_radius_sharp_rect_and_noise_stay_conservative():
    assert reconstruct._corner_radius(np.ones((50, 100), dtype=bool)) == 0
    rng = np.random.default_rng(7)
    assert reconstruct._corner_radius(rng.random((60, 120)) > 0.3) is None


def test_plate_hole_restoration_keeps_carved_button_a_full_pill():
    # Ownership carves the label's glyph pixels out of the plate mask. The text
    # renders on top as its own editable layer, so the plate must still fit as
    # the full primitive — a pill with a text hole is a pill, never a donut.
    holes = ((30, 14, 78, 52), (88, 14, 136, 52))
    mask = _pill_mask(holes=holes)
    assert float(mask.mean()) < .70  # carved enough that geometry would fail
    rgb = np.full((120, 260, 3), 24, dtype=np.uint8)
    canvas_mask = np.zeros((120, 260), dtype=np.uint8)
    canvas_mask[30:97, 20:222] = mask.astype(np.uint8) * 255
    rgb[30:97, 20:222][mask] = (239, 243, 244)
    box = {"x": 20, "y": 30, "w": 202, "h": 67}

    extracted = reconstruct._extract_shape_style(rgb, canvas_mask, box, {}, role="button")

    assert extracted is not None
    assert extracted["shape_kind"] == "rect"
    assert extracted["radius"] == 33.5
    assert extracted["fill"] == {"kind": "flat", "color": "#eff3f4"}
    assert extracted["meta"]["plate_holes_filled_px"] > 0
    # Without the plate role the carved mask stays conservative (no invented shape).
    assert reconstruct._extract_shape_style(rgb, canvas_mask, box, {}, role="photo") is None
    # Config gate: restoration off reproduces the conservative behaviour.
    off = {"reconstruct": {"style_extraction": {"restore_plate_mask": False}}}
    assert reconstruct._extract_shape_style(rgb, canvas_mask, box, off, role="button") is None


def test_outline_only_ghost_button_is_not_solidified():
    # A hollow ring (outline/ghost button) must not be "restored" into a solid
    # plate: filling its interior would invent a fill that is not in the source.
    image = Image.new("L", (160, 60), 0)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 159, 59), radius=29, fill=255)
    draw.rounded_rectangle((4, 4, 155, 55), radius=25, fill=0)
    ring = np.asarray(image) > 16

    restored, filled = reconstruct._fill_plate_holes(ring)

    assert filled == 0
    assert np.array_equal(restored, ring)


def test_biomel_stroke_outline_pill_extracts_stroke_without_opaque_fill():
    """Hollow outline pill → native stroke + transparent fill (no photo blot)."""
    image = Image.new("L", (220, 60), 0)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 219, 59), radius=29, fill=255)
    draw.rounded_rectangle((3, 3, 216, 56), radius=26, fill=0)
    ring = np.asarray(image) > 16
    rgb = np.full((60, 220, 3), 40, dtype=np.uint8)  # photo-like dark field
    rgb[ring] = (10, 10, 10)  # black stroke
    canvas_mask = (ring.astype(np.uint8) * 255)
    box = {"x": 0, "y": 0, "w": 220, "h": 60}

    extracted = reconstruct._extract_shape_style(
        rgb, canvas_mask, box, {}, role="callout", stroke_outline=True,
    )
    assert extracted is not None
    assert extracted["fill"] is None
    assert extracted["stroke"] is not None
    assert extracted["stroke"]["width"] >= 1
    assert extracted["meta"].get("stroke_outline_shell") is True
    assert extracted["meta"].get("fill_transparent") is True
    # Auto-detect hollow ring even without the flag.
    auto = reconstruct._extract_shape_style(rgb, canvas_mask, box, {}, role="callout")
    assert auto is not None
    assert auto["fill"] is None
    assert auto["stroke"] is not None


def test_plate_boundary_fragments_fold_back_into_the_button():
    # SAM peeled both anti-aliased end caps of 009's pill into separate "icons";
    # rendered above the native plate they re-drew the source's dark background
    # ring (the bitten edge). A real icon that sits clear of the rim is kept.
    def candidates():
        return [
            {"id": "plate", "target": "shape", "box": {"x": 833, "y": 134, "w": 202, "h": 67},
             "meta": {"role": "button", "button_shell": True}},
            {"id": "left-cap", "target": "icon", "box": {"x": 832, "y": 147, "w": 12, "h": 39},
             "meta": {"role": "icon"}},
            {"id": "right-cap", "target": "icon", "box": {"x": 1032, "y": 153, "w": 7, "h": 28},
             "meta": {"role": "icon"}},
            {"id": "chevron", "target": "icon", "box": {"x": 1000, "y": 155, "w": 24, "h": 24},
             "meta": {"role": "icon"}},
            {"id": "far-icon", "target": "icon", "box": {"x": 351, "y": 157, "w": 29, "h": 29},
             "meta": {"role": "icon"}},
        ]

    out, suppressed = reconstruct._suppress_plate_boundary_fragments(candidates(), {})

    assert suppressed == 2
    by_id = {c["id"]: c for c in out}
    for cap in ("left-cap", "right-cap"):
        assert by_id[cap]["target"] == "drop"
        assert by_id[cap]["meta"]["suppression_reason"] == "plate-boundary-fragment"
        assert by_id[cap]["meta"]["plate_id"] == "plate"
        assert by_id[cap]["meta"]["removal_required"] is True
    assert by_id["chevron"]["target"] == "icon"
    assert by_id["far-icon"]["target"] == "icon"
    assert by_id["plate"]["target"] == "shape"

    # Config gate keeps the old behaviour available.
    off, count = reconstruct._suppress_plate_boundary_fragments(
        candidates(), {"reconstruct": {"suppress_plate_fragments": False}})
    assert count == 0
    assert all(c["target"] != "drop" for c in off)


def test_engagement_underlay_shell_is_suppressed_on_social():
    """CODIA 009: bogus dark ellipse 'Button' under a comment icon must drop."""
    candidates = [
        {"id": "comment", "target": "icon",
         "box": {"x": 100, "y": 100, "w": 36, "h": 36},
         "meta": {"role": "comment"}},
        {"id": "bogus", "target": "shape",
         "box": {"x": 102, "y": 102, "w": 32, "h": 32},
         "fill": {"kind": "flat", "color": "#030506"},
         "meta": {"role": "button", "button_shell": True}},
        {"id": "real-pill", "target": "shape",
         "box": {"x": 200, "y": 100, "w": 160, "h": 48},
         "fill": {"kind": "flat", "color": "#eff3f4"},
         "meta": {"role": "button"}},
    ]
    cfg = {"scene": {"archetype": "social_screenshot"}}
    out, suppressed = reconstruct._suppress_engagement_underlay_shells(candidates, cfg)
    by_id = {c["id"]: c for c in out}
    assert suppressed == 1
    assert by_id["bogus"]["target"] == "drop"
    assert by_id["bogus"]["meta"]["suppression_reason"] == "engagement-icon-underlay"
    assert by_id["comment"]["target"] == "icon"
    assert by_id["real-pill"]["target"] == "shape"


def test_f7_before_after_label_stays_editable_when_not_over_a_column():
    # F7: a literal before_after_pair no longer forces every Before/After label to be
    # baked into a column photo. A label sitting in the gutter (overlapping no column)
    # must remain a real, swappable TEXT layer; only a label physically inside a column
    # photo is baked (its pixels are part of that raster).
    cfg = {"scene": {"archetype": "comparison_grid", "facts": {"before_after_pair": True}}}
    candidates = [
        {"id": "col-before", "target": "image",
         "box": {"x": 0, "y": 0, "w": 100, "h": 200},
         "meta": {"comparison_side": "before"}},
        {"id": "col-after", "target": "image",
         "box": {"x": 120, "y": 0, "w": 100, "h": 200},
         "meta": {"comparison_side": "after"}},
        # Label baked into the left column photo (contained) -> dropped.
        {"id": "lbl-inside", "target": "text", "text": "Before",
         "box": {"x": 20, "y": 10, "w": 40, "h": 16}},
        # Label in the gutter/below, overlapping no column -> stays editable.
        {"id": "lbl-gutter", "target": "text", "text": "After",
         "box": {"x": 105, "y": 210, "w": 40, "h": 16}},
    ]
    out = {c["id"]: c for c in reconstruct._suppress_comparison_column_labels(candidates, cfg)}
    assert out["lbl-inside"]["target"] == "drop"
    assert out["lbl-inside"]["meta"]["suppression_reason"] == "comparison-column-label-baked"
    assert out["lbl-gutter"]["target"] == "text", "gutter label must stay editable (F7)"
    assert "suppression_reason" not in out["lbl-gutter"].get("meta", {})


def test_f10_slice_budget_truncation_is_recorded(tmp_path, monkeypatch):
    # F10: when more regions fail than the slice budget, the excess used to be dropped
    # silently. They must now be recorded honestly in fallback.json.
    from PIL import Image
    from src import pixel_diff

    run_dir = tmp_path
    canvas = {"w": 1000, "h": 1000}
    Image.new("RGB", (1000, 1000), "white").save(run_dir / "preview.png")
    source = run_dir / "source.png"
    Image.new("RGB", (1000, 1000), "white").save(source)
    (run_dir / "design.json").write_text(json.dumps({
        "id": "d", "canvas": canvas, "layers": []}), encoding="utf-8")
    (run_dir / "reconstruction.json").write_text(json.dumps({"candidates": []}),
                                                 encoding="utf-8")

    failing_rows = [
        {"id": f"c_{i}", "type": "shape", "region_ssim": 0.10 + 0.01 * i,
         "region_px": 5000}
        for i in range(5)
    ]
    monkeypatch.setattr(pixel_diff, "score_layer_regions",
                        lambda *a, **k: failing_rows)

    # Budget 0 records every failing region as truncated and returns before slicing.
    report = reconstruct.apply_raster_slice_fallback(
        str(run_dir), str(source), {"fallback": {"max_slices": 0}})

    assert report["truncated"]["reason"] == "slice-budget-exhausted"
    assert report["truncated"]["un_sliced_count"] == 5
    assert set(report["truncated"]["un_sliced_ids"]) == {f"c_{i}" for i in range(5)}
    budget_skips = [s for s in report["skipped"]
                    if s.get("reason") == "slice-budget-exhausted"]
    assert len(budget_skips) == 5
    assert all("failing_reasons" in s for s in budget_skips)
    # Auditable on disk too.
    on_disk = json.loads((run_dir / "fallback.json").read_text(encoding="utf-8"))
    assert on_disk["truncated"]["un_sliced_count"] == 5
