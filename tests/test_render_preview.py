import os

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


def test_preview_paints_underline_text_decoration(tmp_path):
    preview = np.asarray(_render(tmp_path, [{
        "id": "link", "type": "text", "box": {"x": 8, "y": 10, "w": 120, "h": 36},
        "text": "SALE", "style": {
            "fontSize": 28, "align": "left", "verticalAlign": "top",
            "color": "#111111", "textDecoration": "UNDERLINE",
        },
    }], size=(160, 60)))
    # Underline sits below the glyph baseline — ink must appear in the lower band.
    lower = preview[32:55, 8:130]
    assert np.any(lower < 180), "underline decoration missing from preview"


def test_preview_paints_text_stroke_and_drop_shadow(tmp_path):
    plain = np.asarray(_render(tmp_path / "plain", [{
        "id": "t", "type": "text", "box": {"x": 20, "y": 16, "w": 100, "h": 40},
        "text": "OFF", "style": {"fontSize": 32, "color": "#ffffff", "align": "left"},
    }], size=(160, 80)))
    stroked = np.asarray(_render(tmp_path / "stroke", [{
        "id": "t", "type": "text", "box": {"x": 20, "y": 16, "w": 100, "h": 40},
        "text": "OFF",
        "style": {"fontSize": 32, "color": "#ffffff", "align": "left"},
        "stroke": {"color": "#000000", "width": 3, "align": "OUTSIDE"},
        "effects": [{"type": "DROP_SHADOW", "color": "#00000099",
                     "offset": {"x": 3, "y": 3}, "radius": 2, "visible": True}],
    }], size=(160, 80)))
    # Stroke + shadow add dark ink beyond the plain white fill footprint.
    assert int((stroked.mean(2) < 80).sum()) > int((plain.mean(2) < 80).sum()) + 40


def test_preview_text_clip_allows_top_overflow_inside_group(tmp_path):
    """067/131 class: CENTER text with pad must not be clipped by a tight host frame."""
    preview = np.asarray(_render(tmp_path, [{
        "id": "host", "type": "group",
        "box": {"x": 0, "y": 20, "w": 200, "h": 40},
        "children": [{
            "id": "headline", "type": "text",
            "box": {"x": 4, "y": 0, "w": 190, "h": 40},
            "text": "TOP LINE",
            "style": {
                "fontSize": 30, "align": "left", "verticalAlign": "center",
                "color": "#111111", "lineHeight": 36,
            },
        }],
    }], size=(220, 90)))
    ink_rows = np.where(np.any(preview < 200, axis=(1, 2)))[0]
    assert ink_rows.size, "headline should paint"
    # Ascenders near the group top must still appear in the upper half of the host
    # (not sheared away by clipsContent). Group y=20; ink by ~y=32 is on-frame.
    assert ink_rows.min() <= 32


def test_mixed_italic_runs_render_upright_run_with_its_own_font():
    # 013 headline: run 1 "We NEVER" italic, run 2 "do this!" upright. Same weight
    # and colour, so the weight/colour styled-trigger stays quiet; the italic-aware
    # trigger must still activate the per-run path so the upright run keeps its own
    # upright font instead of inheriting the node's italic slant. Differential proof:
    # rendering run 2 as upright must differ from rendering it italic.
    windir = os.environ.get("WINDIR", r"C:\Windows")
    italic_path = os.path.join(windir, "Fonts", "arialbi.ttf")   # bold italic
    upright_path = os.path.join(windir, "Fonts", "arialbd.ttf")  # bold upright
    if not (os.path.exists(italic_path) and os.path.exists(upright_path)):
        import pytest
        pytest.skip("system Arial bold/italic faces unavailable")

    def build(run2_upright):
        r2_path = upright_path if run2_upright else italic_path
        r2_style = {"fontCandidates": [{"family": "Poppins", "weight": 700, "path": r2_path}]}
        r2_style["fontStyle"] = "Bold" if run2_upright else "Bold Italic"
        if not run2_upright:
            r2_style["italicShearDeg"] = -12
        return {
            "type": "text", "text": "NNNN\nHHHH", "box": {"x": 0, "y": 0, "w": 400, "h": 200},
            # Non-"Arial" family so _text_font's Arial hardcode does not override the
            # per-run candidate paths (the real 013 node is Poppins).
            "style": {"fontFamily": "Poppins", "fontSize": 90, "fontWeight": 700,
                      "fontStyle": "Bold Italic", "color": "#000000", "align": "left",
                      "fontCandidates": [{"family": "Poppins", "weight": 700, "path": italic_path}]},
            "text_runs": [
                {"start": 0, "end": 4, "style": {"fontStyle": "Bold Italic", "italicShearDeg": -12,
                 "fontCandidates": [{"family": "Poppins", "weight": 700, "path": italic_path}]}},
                {"start": 5, "end": 9, "style": r2_style},
            ],
        }

    mixed, _ = render_preview._text_tile(build(run2_upright=True), (400, 200))
    all_italic, _ = render_preview._text_tile(build(run2_upright=False), (400, 200))

    def bottom_line_ink(tile):
        ink = np.asarray(tile)[:, :, 3] > 128
        ys = np.where(np.any(ink, axis=1))[0]
        mid = (int(ys.min()) + int(ys.max())) // 2
        return ink[mid:]

    a, b = bottom_line_ink(mixed), bottom_line_ink(all_italic)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    diff = int(np.logical_xor(a[:h, :w], b[:h, :w]).sum())
    # The upright run and the italic run of the same letters produce visibly different
    # ink; a nonzero (substantial) XOR proves run 2 honoured its own upright font.
    assert diff > 200, f"upright run should differ from italic render (xor={diff})"


def test_same_colour_stroke_does_not_fatten_glyphs_into_a_blob():
    # 009 "UPFRONT": a white #fefefe stroke sampled from the adjacent blue badge over
    # white glyphs only bloats the ink until letters merge. A same-colour stroke must
    # be dropped so glyph mass matches the plain fitted weight.
    base = {"type": "text", "text": "UPFRONT", "box": {"x": 0, "y": 0, "w": 220, "h": 50},
            "style": {"fontFamily": "Arial", "fontSize": 34, "fontWeight": 700, "color": "#ffffff",
                      "align": "left"}}
    no_stroke, _ = render_preview._text_tile(base, (220, 50))
    with_stroke = dict(base)
    with_stroke["stroke"] = {"kind": "flat", "color": "#fefefe", "width": 3.0, "strokeAlign": "OUTSIDE"}
    stroked, _ = render_preview._text_tile(with_stroke, (220, 50))
    ink_plain = int((np.asarray(no_stroke)[:, :, 3] > 128).sum())
    ink_stroked = int((np.asarray(stroked)[:, :, 3] > 128).sum())
    # The near-white stroke is suppressed, so ink mass stays within a few % of plain.
    assert abs(ink_stroked - ink_plain) <= ink_plain * 0.08, (ink_plain, ink_stroked)


def test_contrasting_stroke_is_still_painted():
    # Guard: only SAME-colour strokes are dropped. A black outline on white text must
    # still render (its dark ink appears where the plain fill has none).
    layer = {"type": "text", "text": "HI", "box": {"x": 0, "y": 0, "w": 120, "h": 60},
             "style": {"fontFamily": "Arial", "fontSize": 40, "fontWeight": 700, "color": "#ffffff",
                       "align": "left"},
             "stroke": {"kind": "flat", "color": "#000000", "width": 3.0, "strokeAlign": "OUTSIDE"}}
    tile, _ = render_preview._text_tile(layer, (120, 60))
    arr = np.asarray(tile)
    dark = (arr[:, :, 3] > 128) & (arr[:, :, 0] < 60) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
    assert int(dark.sum()) > 30, "contrasting outline should paint dark ink"


def test_strikethrough_draws_partial_coloured_strike():
    # 091: a hand-drawn strike is authored as textDecoration STRIKETHROUGH with a
    # sampled decorationColor and a decorationSpan covering only the struck words.
    layer = {"type": "text", "text": "Foggy and Steady", "box": {"x": 0, "y": 0, "w": 780, "h": 135},
             "style": {"fontFamily": "Arial", "fontSize": 90, "fontWeight": 700, "color": "#000000",
                       "align": "left", "textDecoration": "STRIKETHROUGH",
                       "decorationColor": "#d23b2f", "decorationSpan": [0.0, 0.42]}}
    tile, _ = render_preview._text_tile(layer, (780, 135))
    arr = np.asarray(tile)
    red = (arr[:, :, 3] > 100) & (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    xs = np.nonzero(red)[1]
    assert xs.size > 100, "coloured strike should be drawn"
    # Strike stays within the struck span (well left of the full width) — "and Steady"
    # is not struck.
    assert xs.max() < tile.width * 0.5, (int(xs.max()), tile.width)
    black = (arr[:, :, 3] > 100) & (arr[:, :, 0] < 60) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
    assert int(black.sum()) > 500, "glyph ink still black under the strike"
