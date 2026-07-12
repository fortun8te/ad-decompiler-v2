import json
import os

import numpy as np
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
