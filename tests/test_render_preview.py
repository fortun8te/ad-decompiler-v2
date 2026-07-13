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


def test_preview_does_not_replace_failed_vector_path_with_rectangle(tmp_path, monkeypatch):
    monkeypatch.setattr(render_preview, "_svg_or_path_mask", lambda layer, size: None)
    preview = _render(tmp_path, [{
        "id": "arrow", "type": "shape", "box": {"x": 10, "y": 10, "w": 20, "h": 30},
        "shape_kind": "path", "path": "malformed", "fill": {"color": "#203010"},
    }])
    assert preview.getpixel((20, 20)) == (255, 255, 255)


def test_preview_uses_raster_fallback_when_vector_is_empty(tmp_path, monkeypatch):
    fallback = tmp_path / "icon.png"
    Image.new("RGBA", (10, 10), (12, 180, 40, 255)).save(fallback)
    monkeypatch.setattr(render_preview, "_svg_or_path_mask", lambda layer, size: None)
    preview = _render(tmp_path, [{
        "id": "icon", "type": "shape", "box": {"x": 10, "y": 10, "w": 20, "h": 20},
        "shape_kind": "path", "svg": "<svg></svg>", "src": "icon.png",
    }])
    assert preview.getpixel((20, 20)) == (12, 180, 40)


def test_preview_text_uses_layer_fill_when_style_color_is_absent(tmp_path):
    preview = _render(tmp_path, [{
        "id": "copy", "type": "text", "box": {"x": 5, "y": 5, "w": 60, "h": 20},
        "text": "A", "style": {"fontSize": 18}, "fill": {"color": "#ff0000"},
    }])
    assert any(pixel[0] > pixel[1] * 2 for pixel in preview.getdata())


def test_preview_prefers_selected_font_family_over_stale_candidate_order():
    font = render_preview._text_font({
        "fontFamily": "Arial",
        "fontCandidates": [
            {"family": "Comic Sans MS", "path": "C:\\Windows\\Fonts\\comic.ttf"},
            {"family": "Arial", "path": "C:\\Windows\\Fonts\\arial.ttf"},
        ],
    }, 20)
    assert "arial" in " ".join(font.getname()).lower()


def test_preview_honors_text_horizontal_and_vertical_alignment(tmp_path):
    left_top = np.asarray(_render(tmp_path / "a", [{
        "id": "copy", "type": "text", "box": {"x": 0, "y": 0, "w": 70, "h": 40},
        "text": "Hi", "style": {"fontSize": 16, "align": "left", "verticalAlign": "top"},
    }], size=(70, 40)))
    right_bottom = np.asarray(_render(tmp_path / "b", [{
        "id": "copy", "type": "text", "box": {"x": 0, "y": 0, "w": 70, "h": 40},
        "text": "Hi", "style": {"fontSize": 16, "align": "right", "verticalAlign": "bottom"},
    }], size=(70, 40)))
    first_ink = np.argwhere(np.any(left_top < 220, axis=2))
    second_ink = np.argwhere(np.any(right_bottom < 220, axis=2))
    assert second_ink[:, 1].mean() > first_ink[:, 1].mean() + 20
    assert second_ink[:, 0].mean() > first_ink[:, 0].mean() + 10


def test_preview_never_clips_text_ascenders_at_top(tmp_path):
    preview = np.asarray(_render(tmp_path, [{
        "id": "headline", "type": "text", "box": {"x": 10, "y": 0, "w": 200, "h": 28},
        "text": "LAATSTE SALE", "style": {"fontSize": 24, "align": "left", "verticalAlign": "top"},
    }], size=(240, 50)))
    ink_rows = np.where(np.any(preview < 200, axis=(1, 2)))[0]
    assert ink_rows.size
    assert ink_rows.min() <= 8


def test_preview_never_clips_text_wider_than_its_box(tmp_path):
    # The box is deliberately far too narrow for the run; the renderer must grow the
    # drawn region instead of cutting the last words off at the right edge (ad9 defect).
    preview = np.asarray(_render(tmp_path, [{
        "id": "copy", "type": "text", "box": {"x": 5, "y": 15, "w": 30, "h": 22},
        "text": "waarbij je 20%", "style": {"fontSize": 20, "align": "left"},
    }], size=(400, 60)))
    ink_cols = np.where(np.any(preview < 200, axis=(0, 2)))[0]
    assert ink_cols.size, "text should have been drawn"
    # Ink must extend well past the 35px box right edge — nothing is clipped away.
    assert ink_cols.max() > 120
    assert ink_cols.min() <= 12


def test_preview_applies_letter_spacing_tracking(tmp_path):
    def right_edge(tracking):
        preview = np.asarray(_render(tmp_path / f"t{tracking}", [{
            "id": "copy", "type": "text", "box": {"x": 5, "y": 15, "w": 380, "h": 22},
            "text": "MMMMMM", "style": {"fontSize": 20, "align": "left", "letterSpacing": tracking},
        }], size=(400, 60)))
        cols = np.where(np.any(preview < 200, axis=(0, 2)))[0]
        return int(cols.max())
    # Positive tracking spreads the glyphs, so the same string reaches further right;
    # a renderer that ignored letterSpacing (as PIL's multiline_text does) would tie.
    assert right_edge(12) > right_edge(0) + 30


def test_preview_does_not_draw_fake_gray_for_missing_image(tmp_path):
    preview = _render(tmp_path, [{
        "id": "missing", "type": "image", "box": {"x": 10, "y": 10, "w": 20, "h": 20},
        "src": "does-not-exist.png",
    }])
    assert preview.getpixel((15, 15)) == (255, 255, 255)
