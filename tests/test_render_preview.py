import numpy as np
from PIL import Image, ImageDraw

from src import render_preview


def _render(tmp_path, layers, size=(80, 60)):
    result = render_preview.render({"canvas": {"w": size[0], "h": size[1]}, "layers": layers}, str(tmp_path))
    return Image.open(result["preview"]).convert("RGB")


def test_preview_honors_multistop_angle_gradient_and_stroke(tmp_path):
    image = _render(tmp_path, [{
        "id": "surface", "type": "shape", "box": {"x": 5, "y": 5, "w": 60, "h": 30},
        "shape_kind": "rect", "radius": 6,
        "fill": {"kind": "linear", "angle": 0, "stops": [
            {"position": 0, "color": "#ff0000"},
            {"position": .5, "color": "#00ff00"},
            {"position": 1, "color": "#0000ff"},
        ]},
        "stroke": {"color": "#000000", "width": 2},
    }])
    assert image.getpixel((7, 20)) == (0, 0, 0)
    left, middle, right = image.getpixel((10, 20)), image.getpixel((35, 20)), image.getpixel((60, 20))
    assert left[0] > left[2] and middle[1] > middle[0] and right[2] > right[0]


def test_preview_keeps_source_transparency_when_an_image_has_a_geometric_mask(tmp_path):
    asset = tmp_path / "cutout.png"
    image = Image.new("RGBA", (30, 30), (255, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 29, 29), fill=(255, 0, 0, 255))
    draw.rectangle((12, 12, 17, 17), fill=(0, 0, 0, 0))
    image.save(asset)
    preview = _render(tmp_path, [{
        "id": "photo", "type": "image", "box": {"x": 10, "y": 10, "w": 30, "h": 30},
        "src": "cutout.png", "mask": {"kind": "ellipse"},
    }])
    assert preview.getpixel((25, 25)) == (255, 255, 255)  # transparent source hole stayed transparent
    assert preview.getpixel((25, 12))[0] > 200            # still has actual painted pixels
    assert preview.getpixel((10, 10)) == (255, 255, 255)  # outside ellipse is clipped


def test_preview_applies_nested_group_opacity_rotation_and_shadow(tmp_path):
    preview = _render(tmp_path, [{
        "id": "card", "type": "group", "box": {"x": 24, "y": 14, "w": 24, "h": 24}, "opacity": .5,
        "rotation": 25, "effects": [{"type": "drop-shadow", "color": "#000000", "opacity": .7,
                                         "x": 4, "y": 4, "blur": 2}],
        "children": [{
            "id": "inside", "type": "shape", "box": {"x": 2, "y": 2, "w": 20, "h": 20},
            "shape_kind": "rect", "fill": {"color": "#ff0000"},
        }],
    }])
    # A 50% red child over white is pink, not opaque red; the rotated corners and shadow
    # prove this is drawn as one transformed/effected group rather than flat child rectangles.
    pixel = preview.getpixel((36, 26))
    assert pixel[0] > 230 and 90 < pixel[1] < 180 and 90 < pixel[2] < 180
    # The lower-right area contains the expanded shadow; it is not clipped to the 24px group box.
    assert np.asarray(preview)[34:52, 42:62].min() < 245


def test_preview_uses_non_normal_blend_mode(tmp_path):
    preview = _render(tmp_path, [
        {"id": "back", "type": "shape", "box": {"x": 5, "y": 5, "w": 30, "h": 30},
         "fill": {"color": "#808080"}},
        {"id": "top", "type": "shape", "box": {"x": 10, "y": 10, "w": 30, "h": 30},
         "fill": {"color": "#808080"}, "blend_mode": "MULTIPLY"},
    ])
    assert preview.getpixel((20, 20))[0] < 100
