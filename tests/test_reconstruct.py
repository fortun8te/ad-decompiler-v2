import json
import os

import numpy as np

from src.reconstruct import _is_background_plate


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
         "meta": {"role": "offer", "parent_id": "product"}},
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
