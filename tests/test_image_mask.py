"""Image-as-masked-element: a rasterized logo/photo/avatar is delivered as a swappable
image FILL clipped by a shape mask (ellipse / rounded-rect / path), never flattened into
the background plate.  These exercise the routing + reconstruct decision layers; the Figma
plugin (figma-plugin/code.js) turns the mask spec into the actual clipped node.
"""
import numpy as np
from PIL import Image, ImageDraw

from src import reconstruct


def _noise_source(path, size, box, seed=7):
    """A flat plate with a high-variance (photographic) patch filling ``box``."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size[1], size[0], 3), 240, dtype=np.uint8)
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    canvas[y:y + h, x:x + w] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(canvas).save(path)


def _circle_mask(path, box):
    mask = Image.new("L", (box["w"], box["h"]), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, box["w"] - 1, box["h"] - 1), fill=255)
    mask.save(path)


def _run(source, candidates, tmp_path):
    return reconstruct.reconstruct(str(source), {"lines": []}, candidates, str(tmp_path),
                                   {"inpaint": {"mode": "opencv"}})


def test_photographic_circle_shape_becomes_swappable_image_with_ellipse_mask(tmp_path):
    """The ad9 Twitter avatar case: a near-square, round, photographic region detected as a
    ``shape`` must be reclassified to an IMAGE clipped by an ellipse — not a flat fill."""
    box = {"x": 40, "y": 40, "w": 120, "h": 120}
    source = tmp_path / "avatar.png"
    _noise_source(source, (200, 200), box)
    masks = tmp_path / "m"; masks.mkdir()
    _circle_mask(masks / "av.png", box)
    candidate = {
        "id": "av", "target": "shape", "box": box,
        "mask": {"kind": "alpha", "src": "m/av.png"},
        "meta": {"role": "shape", "source": "element", "confidence": 1.0},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["target"] == "image"                         # not flattened to a solid shape
    assert out["mask"]["kind"] == "ellipse"                 # swappable circular clip
    assert out["src"].startswith("assets/")                 # real pixels kept as the fill
    assert out["meta"]["reclassified"] == "shape->image"


def test_flat_circle_shape_stays_a_native_shape(tmp_path):
    """A flat-coloured circle is faithfully a primitive and must NOT be rasterized."""
    box = {"x": 40, "y": 40, "w": 120, "h": 120}
    source = tmp_path / "dot.png"
    image = Image.new("RGB", (200, 200), (240, 240, 240))
    ImageDraw.Draw(image).ellipse((40, 40, 159, 159), fill=(32, 120, 220))
    image.save(source)
    masks = tmp_path / "m"; masks.mkdir()
    _circle_mask(masks / "dot.png", box)
    candidate = {
        "id": "dot", "target": "shape", "box": box,
        "mask": {"kind": "alpha", "src": "m/dot.png"},
        "meta": {"role": "shape", "source": "element", "confidence": 1.0},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["target"] == "shape"
    assert out["shape_kind"] == "ellipse"
    assert out.get("meta", {}).get("reclassified") is None


def test_round_image_cutout_infers_ellipse_mask_without_role_hint(tmp_path):
    """An image cutout with round alpha coverage becomes an ellipse even when the role is a
    generic 'photo' (routing left the mask as alpha)."""
    box = {"x": 30, "y": 30, "w": 100, "h": 100}
    source = tmp_path / "photo.png"
    _noise_source(source, (160, 160), box)
    masks = tmp_path / "m"; masks.mkdir()
    _circle_mask(masks / "p.png", box)
    candidate = {
        "id": "p", "target": "image", "box": box,
        "mask": {"kind": "alpha", "src": "m/p.png"},
        "meta": {"role": "photo", "source": "element", "confidence": 0.9},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["target"] == "image"
    assert out["mask"]["kind"] == "ellipse"


def test_icon_image_cutout_keeps_its_own_alpha(tmp_path):
    """An icon's shape IS its art; it must keep its raster alpha, not a primitive clip."""
    box = {"x": 30, "y": 30, "w": 100, "h": 100}
    source = tmp_path / "icon.png"
    _noise_source(source, (160, 160), box)
    masks = tmp_path / "m"; masks.mkdir()
    _circle_mask(masks / "i.png", box)
    candidate = {
        "id": "i", "target": "image", "box": box,
        "mask": {"kind": "alpha", "src": "m/i.png"},
        "meta": {"role": "icon", "source": "element", "confidence": 0.9,
                 "vector_fallback": True},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["mask"]["kind"] == "alpha"


def test_logo_cutout_emits_clean_silhouette_path_mask(tmp_path):
    """A logo/brand cutout with one clean silhouette contour becomes a path mask so the
    raster fill can be swapped while the outline holds."""
    box = {"x": 40, "y": 40, "w": 100, "h": 80}
    source = tmp_path / "logo.png"
    _noise_source(source, (200, 160), box)
    masks = tmp_path / "m"; masks.mkdir()
    blob = Image.new("L", (box["w"], box["h"]), 0)
    ImageDraw.Draw(blob).rounded_rectangle((6, 6, box["w"] - 7, box["h"] - 7), radius=16, fill=255)
    blob.save(masks / "logo.png")
    candidate = {
        "id": "logo", "target": "image", "box": box,
        "mask": {"kind": "path", "src": "m/logo.png"},
        "meta": {"role": "logo", "source": "element", "confidence": 0.9},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["mask"]["kind"] == "path"
    assert isinstance(out["mask"]["path"], str) and out["mask"]["path"].startswith("M")


def test_rounded_card_cutout_gets_rounded_rect_mask(tmp_path):
    box = {"x": 20, "y": 20, "w": 160, "h": 100}
    source = tmp_path / "card.png"
    _noise_source(source, (220, 160), box)
    masks = tmp_path / "m"; masks.mkdir()
    card = Image.new("L", (box["w"], box["h"]), 0)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, box["w"] - 1, box["h"] - 1), radius=20, fill=255)
    card.save(masks / "card.png")
    candidate = {
        "id": "card", "target": "image", "box": box,
        "mask": {"kind": "rrect", "src": "m/card.png"},
        "meta": {"role": "card", "source": "element", "confidence": 0.9},
    }
    out = _run(source, [candidate], tmp_path)["candidates"][0]
    assert out["mask"]["kind"] == "rrect"
    assert isinstance(out["mask"]["radius"], (int, float))
