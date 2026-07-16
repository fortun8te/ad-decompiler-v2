"""Inline platform-UI glyphs — 009's blue verified badge.

The badge is missed by both pre-existing scans on three independent counts: it is
smaller than standalone_min_h_frac, it is adjacent to text so the letters guard
rejects it, and it is a filled disc that classify_glyph scores near zero. Blue was
also absent from the extreme-colour set entirely.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src import icon_detect

OPTS = dict(icon_detect.DEFAULTS)


def _canvas(w=600, h=200, bg=(0, 0, 0)):
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _line(id_, x, y, w, h, text=""):
    return {"id": id_, "text": text, "box": {"x": float(x), "y": float(y),
                                             "w": float(w), "h": float(h)}}


def _letters(img, x0, y, count, h=25, colour=(255, 255, 255), step=34):
    """A row of copy set in one ink (step > h so glyphs stay separate blobs)."""
    for i in range(count):
        x = x0 + i * step
        img[y:y + h, x:x + h] = colour


def test_blue_is_in_the_extreme_colour_set():
    """Its absence is why the verified badge was never even a candidate."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :] = (29, 155, 240)  # X/Twitter blue
    masks = icon_detect._extreme_masks(rgb)
    assert "blue" in masks
    assert masks["blue"].all()


def test_blue_mask_does_not_fire_on_grey_or_white():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 255, 255)
    rgb[0, 1] = (128, 128, 128)
    rgb[0, 2] = (10, 10, 10)
    assert not icon_detect._extreme_masks(rgb)["blue"][0].any()


def test_009_blue_badge_beside_white_copy_is_detected():
    """The real geometry: a 28px blue disc at the end of a 25px white name row."""
    img = _canvas()
    _letters(img, 40, 158, 8)                      # 'UPFRONT' in white
    cv2.circle(img, (366, 172), 14, (29, 155, 240), -1)  # the badge
    lines = [_line("L8", 40, 158, 340, 25, "UPFRONT")]
    dets = icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, [])
    assert len(dets) == 1, dets
    det = dets[0]
    assert det["info"]["tone"] == "blue"
    assert det["row_text_id"] == "L8", "chip must anchor to the display name row"
    assert det["anchor"] == "inline" and det["role"] == "chrome"
    # Raster-first: the mark is shipped as pixels, never template-classified.
    assert det["glyph"] is None
    assert abs(det["box"]["x"] - 352) <= 3 and abs(det["box"]["y"] - 158) <= 3


def test_letters_of_the_copy_are_not_detected():
    """Without the majority-tone guard this returns one detection per letter
    (137 across real 009)."""
    img = _canvas()
    _letters(img, 40, 158, 10)
    lines = [_line("L8", 40, 158, 300, 25, "UPFRONT")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, []) == []


def test_a_row_of_black_copy_on_a_light_pill_yields_nothing():
    """009's 'Volgend' button: dark copy on a light plate is still just copy."""
    img = _canvas(bg=(255, 255, 255))
    _letters(img, 40, 158, 8, colour=(0, 0, 0))
    lines = [_line("L1", 40, 158, 300, 25, "Volgend")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, []) == []


def test_a_row_with_too_few_glyphs_is_skipped():
    """With no majority we cannot know the copy's ink, so we decline to guess."""
    img = _canvas()
    cv2.circle(img, (366, 172), 14, (29, 155, 240), -1)
    lines = [_line("L8", 40, 158, 340, 25, "A")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, []) == []


def test_a_letter_counter_is_too_short_to_be_a_badge():
    """A badge matches the row's cap height; a counter is a fraction of it."""
    img = _canvas()
    _letters(img, 40, 158, 8)
    img[164:178, 300:314] = (29, 155, 240)  # 14px blob on a 25px row -> ratio 0.56
    lines = [_line("L8", 40, 158, 340, 25, "UPFRONT")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, []) == []


def test_a_glyph_far_from_any_row_is_not_inline():
    img = _canvas(h=400)
    _letters(img, 40, 40, 8)
    cv2.circle(img, (366, 340), 14, (29, 155, 240), -1)
    lines = [_line("L8", 40, 40, 340, 25, "UPFRONT")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 400}, OPTS, []) == []


def test_an_elongated_blob_is_not_a_badge():
    img = _canvas()
    _letters(img, 40, 158, 8)
    img[158:186, 300:460] = (29, 155, 240)  # a blue rule, aspect ~5.7
    lines = [_line("L8", 40, 158, 340, 25, "UPFRONT")]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, []) == []


def test_inline_scan_respects_already_taken_boxes():
    img = _canvas()
    _letters(img, 40, 158, 8)
    cv2.circle(img, (366, 172), 14, (29, 155, 240), -1)
    lines = [_line("L8", 40, 158, 340, 25, "UPFRONT")]
    taken = [{"box": {"x": 352.0, "y": 158.0, "w": 28.0, "h": 28.0}}]
    assert icon_detect.detect_inline_glyphs(img, lines, {"w": 600, "h": 200}, OPTS, taken) == []


def test_inline_scan_can_be_disabled():
    opts = dict(OPTS, inline_enabled=False)
    assert opts["inline_enabled"] is False


def test_inline_row_matches_a_glyph_the_ocr_box_overhangs():
    """009's 'UPFRONT' box ends at x=378, right across the badge at 351..380 —
    the glyph is INSIDE its row's box, not merely beside it."""
    lines = [_line("L8", 184, 158, 194, 25, "UPFRONT")]
    row = icon_detect._inline_row({"x": 351.0, "y": 157.0, "w": 29.0, "h": 29.0},
                                  lines, OPTS)
    assert row is not None and row["id"] == "L8"


def test_inline_row_matches_a_glyph_past_the_box_edge():
    """009's ⏳ sits at x=712 while the headline box stops at 699."""
    lines = [_line("L2", 43, 321, 656, 44, "LAATSTE SITE WIDE SALE VAN 2026")]
    row = icon_detect._inline_row({"x": 712.0, "y": 322.0, "w": 25.0, "h": 38.0},
                                  lines, OPTS)
    assert row is not None and row["id"] == "L2"


def test_inline_row_rejects_a_vertically_offset_glyph():
    lines = [_line("L2", 43, 321, 656, 44, "headline")]
    assert icon_detect._inline_row({"x": 712.0, "y": 900.0, "w": 25.0, "h": 38.0},
                                   lines, OPTS) is None
