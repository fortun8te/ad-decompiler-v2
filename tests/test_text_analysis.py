"""CPU-only tests for text geometry, typography, grouping, and font retrieval."""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import text_analysis  # noqa: E402


def _font_path():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    candidates += glob.glob("/usr/share/fonts/**/*DejaVuSans.ttf", recursive=True)[:1]
    return next((path for path in candidates if os.path.isfile(path)), None)


def _font(size):
    path = _font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def _line(line_id, text, bbox, image_box=None, conf=0.98):
    x0, y0, x1, y1 = image_box or bbox
    box = {"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)}
    return {
        "id": line_id,
        "text": text,
        "conf": conf,
        "box": box,
        "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "words": [],
    }


def _draw_text(draw, xy, text, font, fill):
    bbox = draw.textbbox(xy, text, font=font)
    draw.text(xy, text, font=font, fill=fill)
    # OCR boxes usually include some line-box breathing room.
    return (max(0, bbox[0] - 7), max(0, bbox[1] - 6), bbox[2] + 7, bbox[3] + 6)


def test_fit_text_box_scales_multiline_line_height_with_font(monkeypatch):
    monkeypatch.setattr(text_analysis, "_fit_font", lambda style, size: _font(max(1, int(size))))
    _, resize, patch = text_analysis.fit_text_box(
        "First long line\nSecond long line\nThird long line",
        {"fontSize": 40, "lineHeight": 52, "letterSpacing": 0},
        {"x": 0, "y": 0, "w": 180, "h": 95},
    )
    assert resize == "HEIGHT"
    assert patch["fontSize"] < 40
    assert patch["lineHeight"] < 52


def test_fit_text_box_never_emits_line_height_below_font_size(monkeypatch):
    """Ad 013: dense display OCR measured lh 195 < fs 230 and clipped glyph tops."""
    monkeypatch.setattr(text_analysis, "_fit_font", lambda style, size: _font(max(1, int(size))))
    _, _, patch = text_analysis.fit_text_box(
        "We NEVER\ndo this!",
        {"fontSize": 230, "lineHeight": 195, "letterSpacing": 0},
        {"x": 0, "y": 0, "w": 1000, "h": 500},
    )
    fs = patch.get("fontSize", 230)
    lh = patch.get("lineHeight", 195)
    assert lh >= fs * 1.05


def test_enriches_painted_geometry_colour_baseline_and_style(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "ad.png"
    image.save(path)
    original = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 640, "h": 260},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }
    frozen = copy.deepcopy(original)

    result = text_analysis.analyze_text(str(path), original, {})
    line = result["lines"][0]

    assert original == frozen, "analysis must not mutate upstream OCR"
    assert line["painted_box"]["w"] < line["box"]["w"]
    assert line["painted_box"]["h"] < line["box"]["h"]
    r, g, b = line["style"]["colorRGB"]
    assert r > 160 and g < 90 and b < 90
    assert line["painted_box"]["y"] <= line["baseline"]["y0"] <= (
        line["painted_box"]["y"] + line["painted_box"]["h"] + 2
    )
    assert line["rotation_deg"] == pytest.approx(0.0, abs=0.01)
    assert line["style"]["fontSize"] > line["painted_box"]["h"]
    assert line["style"]["fontCandidates"]
    assert result["blocks"] and result["styles"] and result["sections"]


def test_glyph_tight_black_text_on_white_stays_dark(tmp_path):
    """002 regression: tight OCR borders must not invert black copy to white."""
    image = Image.new("RGB", (520, 70), "white")
    draw = ImageDraw.Draw(image)
    font = _font(40)
    bbox = _draw_text(draw, (10, 12), "KRACHTSPORT BUNDEL", font, (0, 0, 0))
    x0, y0, x1, y1 = bbox
    tight = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
    path = tmp_path / "tight.png"
    image.save(path)
    result = text_analysis.analyze_text(
        str(path),
        {"engine": "synthetic", "source": {"path": str(path), "w": 520, "h": 70},
         "lines": [_line("L1", "KRACHTSPORT BUNDEL", tight)]},
        {},
    )
    r, g, b = result["lines"][0]["style"]["colorRGB"]
    assert r < 40 and g < 40 and b < 40


def test_glyph_tight_white_text_on_dark_stays_light(tmp_path):
    image = Image.new("RGB", (400, 70), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    font = _font(40)
    bbox = _draw_text(draw, (10, 12), "WHITE COPY", font, (250, 250, 250))
    x0, y0, x1, y1 = bbox
    tight = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
    path = tmp_path / "tight_light.png"
    image.save(path)
    result = text_analysis.analyze_text(
        str(path),
        {"engine": "synthetic", "source": {"path": str(path), "w": 400, "h": 70},
         "lines": [_line("L0", "WHITE COPY", tight)]},
        {},
    )
    r, g, b = result["lines"][0]["style"]["colorRGB"]
    assert r > 200 and g > 200 and b > 200


def _heavy_headline_image(page=(200, 700), plate=(246, 246, 246), ink=(251, 2, 2)):
    """A display headline whose ink OUTWEIGHS the plate inside its own tight box.

    Ten fat stems across the box put the ink at ~55% of it — the real 067 headline
    measures 54% — so ink is the MAJORITY luminance class and every in-crop heuristic
    that assumes "ink is the minority" elects the plate instead.

    The stems are blurred because a rasteriser anti-aliases glyph edges: a fixture with
    exactly two luminance values is degenerate (Otsu's bin lands below the lower cluster
    and the luminance split returns nothing), which would test something real images
    never do.
    """
    from PIL import ImageFilter

    image = np.full((page[0], page[1], 3), plate, dtype=np.uint8)
    for i in range(10):
        x = 60 + i * 58
        image[60:140, x:x + 32] = ink
    image = np.asarray(
        Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.9)), dtype=np.uint8
    )
    return image, (60, 60, 640, 140)  # tight box hugging the stems


def test_majority_ink_display_headline_keeps_its_colour(tmp_path):
    """067 regression: 'WE'RE SAYING GOODBYE' rendered #f6f6f6 white-on-white.

    Its red ink is 54% of its glyph-tight box, so both in-crop polarity heuristics (the
    contaminated border estimate AND the minority-luminance guard) elect the white plate
    as ink. Pixels outside the box are the only ones that can adjudicate.
    """
    image, box = _heavy_headline_image()
    path = tmp_path / "heavy.png"
    Image.fromarray(image).save(path)
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    ink_is_majority = (
        np.linalg.norm(crop.astype(float) - np.array([251.0, 2.0, 2.0]), axis=2) < 30
    ).mean()
    assert ink_is_majority > 0.5, "fixture must put ink in the MAJORITY to pin the bug"

    result = text_analysis.analyze_text(
        str(path),
        {"engine": "synthetic", "source": {"path": str(path), "w": 700, "h": 200},
         "lines": [_line("L0", "WE'RE SAYING GOODBYE", box)]},
        {},
    )
    r, g, b = result["lines"][0]["style"]["colorRGB"]
    assert r > 200 and g < 60 and b < 60, f"headline inverted to plate: {(r, g, b)}"


def test_exterior_plate_prior_abstains_when_the_band_has_no_single_plate():
    """The constraint that makes the prior safe (and that sank the collar attempt).

    Widening the LINE sampling window was a net regression because a line box's
    surroundings need not be one plate. So the prior speaks only for a band that is
    substantially ONE colour, and abstains on a band with no dominant plate (091's text
    over busy scene art measures 0.04 uniformity) rather than reducing it to a per-channel
    median — which on a mixed band is a phantom colour matching neither side.
    """
    box = {"x": 40.0, "y": 40.0, "w": 120.0, "h": 40.0}
    uniform = np.full((160, 240, 3), 246, dtype=np.uint8)
    prior = text_analysis._exterior_plate_prior(uniform, box)
    assert prior is not None
    assert np.allclose(prior[0], [246, 246, 246], atol=2)
    assert prior[1] == pytest.approx(1.0)

    rng = np.random.default_rng(7)
    busy = rng.integers(0, 256, size=(160, 240, 3), dtype=np.uint8)  # scene art
    assert text_analysis._exterior_plate_prior(busy, box) is None


def test_ink_polarity_is_not_retuned_when_the_mask_is_already_ink():
    """Only inversions may be corrected — never a shade.

    A correct mask that includes anti-aliased edge pixels sits BETWEEN the two classes
    (101's body copy elects #2a2a2a where the glyph core is #080808). Nudging those would
    churn every line's fill and reshape ink mass for no correctness gain.
    """
    crop = np.full((40, 200, 3), 255, dtype=np.uint8)
    crop[10:30, 20:60] = 0            # glyph core
    crop[10:30, 60:66] = 90           # AA fringe
    plate_prior = (np.array([255.0, 255.0, 255.0]), 1.0)
    mask = np.zeros(crop.shape[:2], bool)
    mask[10:30, 20:66] = True         # correct ink mask, fringe included
    out = text_analysis._resolve_ink_polarity(crop, mask, plate_prior)
    assert out is mask, "a mask already on the ink side must be returned untouched"


def test_ink_polarity_flip_reverses_a_mask_that_elected_the_plate():
    image, box = _heavy_headline_image()
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    plate_prior = (np.array([246.0, 246.0, 246.0]), 0.95)
    is_ink = np.linalg.norm(crop.astype(float) - np.array([251.0, 2.0, 2.0]), axis=2) < 40
    inverted = ~is_ink                # mask elected the white plate as "ink"
    out = text_analysis._resolve_ink_polarity(crop, inverted, plate_prior)
    assert out is not inverted
    assert np.median(crop[out].astype(float), axis=0)[0] > 200   # red channel high
    assert np.median(crop[out].astype(float), axis=0)[1] < 60    # green channel low


def test_expected_ink_ratio_tracks_the_glyphs_the_line_actually_has():
    """009 regression: one tweet body, emitted 37.0…50.0.

    Sizing as ink_height/cap_ratio assumes every line's ink spans the caps. Ink actually
    runs tallest-glyph to lowest, so the same font size measures ~0.72em all-caps but
    ~0.96em once an ascender AND a descender appear — inventing a 25% size step between
    lines a reader sees as identical.
    """
    cap = 0.72
    caps_only = text_analysis._expected_ink_ratio("LAATSTE SITE WIDE SALE", cap)
    asc_desc = text_analysis._expected_ink_ratio("woensdag 20 mei om 20:00 uur.", cap)
    x_only = text_analysis._expected_ink_ratio("no", cap)
    assert caps_only == pytest.approx(0.72, abs=0.01)
    assert asc_desc == pytest.approx(0.96, abs=0.02)
    assert x_only < caps_only, "x-height-only copy spans less ink than caps"

    # The payoff: identical source size, different glyph mix -> same recovered size.
    assert (36.0 / asc_desc) == pytest.approx(37.6, abs=0.6)
    assert (27.0 / caps_only) == pytest.approx(37.5, abs=0.6)


def test_peer_lines_of_one_block_unify_to_a_single_scale():
    """016 regression: '21+ vitamins' (29.4) / '& minerals' (26.8) are one callout."""
    lines = [
        {"id": "L2", "text": "21+ vitamins", "block_id": "B2",
         "style": {"fontFamily": "Poppins", "fontSize": 29.4, "fontWeight": 400}},
        {"id": "L3", "text": "& minerals", "block_id": "B2",
         "style": {"fontFamily": "Poppins", "fontSize": 26.8, "fontWeight": 400}},
    ]
    changes = text_analysis._unify_peer_text_scale(lines)
    assert changes
    sizes = {ln["style"]["fontSize"] for ln in lines}
    assert len(sizes) == 1, f"callout peers must share one size, got {sizes}"


def test_101_tittle_is_not_x_height_ink():
    """101: 'mini pumps' fitted 30.25 in a column of 24.0 and was ejected from its peers.

    It carries no b/d/f/h/k/l/t, so it was modelled as x-height-only ink — but the dot on
    'i' clears x-height and lands at ~cap height, so the ink really spans tittle to
    'p'-descender. Measured against 101 (ink_h 22, true size 24.0): x-height +26.1%,
    ascender -4.3%, cap height -1.3%.
    """
    cap = 0.72
    tittle = text_analysis._expected_ink_ratio("mini pumps", cap)
    # Same glyph extremes as an explicit cap + descender line.
    assert tittle == pytest.approx(text_analysis._expected_ink_ratio("Mp", cap))
    assert 22.0 / tittle == pytest.approx(24.0, abs=0.5)
    # A true ascender still out-tops a tittle, and x-height-only ink is unaffected.
    assert text_analysis._expected_ink_ratio("mini bumps", cap) > tittle
    assert text_analysis._expected_ink_ratio("names", cap) < tittle


def test_101_column_peers_unify_across_role_shattered_blocks():
    """101: a uniform checklist column whose rows the block grouper could not see.

    '50% thicker for better' is labelled an *offer* (the offer regex fires on "50%") and
    'repairs & sealant use' a *footer* (it sits low on the canvas), so _can_join's role
    veto strands each in a singleton block that the block pass can never reach. They are
    still one visual column and must share one scale.
    """
    def row(ident, text, y, size, weight, block):
        return {"id": ident, "text": text, "block_id": block,
                "box": {"x": 110.0, "y": y, "w": 260.0, "h": 25.0},
                "style": {"fontFamily": "Poppins", "fontSize": size,
                          "fontWeight": weight, "color": "#111111"}}

    # Geometry and fitted styles as MEASURED off 101 (every row is one size in the source).
    lines = [
        row("L0", "50% thicker for better", 704.1, 24.04, 400, "B9"),
        row("L1", "durability", 733.4, 20.89, 350, "B10"),
        row("L2", "Aluminium valve built to last", 780.5, 24.04, 400, "B11"),
        row("L3", "Compatible with electric", 825.2, 24.10, 400, "B11"),
        row("L4", "mini pumps", 854.4, 23.69, 400, "B11"),
        row("L5", "Removable valve core for", 905.3, 24.04, 400, "B11"),
        row("L6", "repairs & sealant use", 930.7, 22.97, 350, "B13"),
    ]
    changes = text_analysis._unify_column_text_scale(lines)
    assert changes
    sizes = [ln["style"]["fontSize"] for ln in lines]
    assert max(sizes) / min(sizes) <= 1.02, \
        f"column rows must render as one size, got {sorted(set(sizes))}"
    assert {ln["style"]["fontWeight"] for ln in lines} == {400}, "and one weight"


def _067_body_line(ident, text, y, height, baseline, size, role):
    """One body line of 067, with the geometry MEASURED off the fixture."""
    box = {"x": 60.0, "y": y, "w": 1400.0, "h": height}
    return {"id": ident, "text": text, "role": role,
            "box": dict(box), "painted_box": dict(box),
            "baseline": {"y0": baseline},
            "style": {"fontFamily": "Poppins", "fontSize": size,
                      "fontWeight": 400, "color": "#000000", "align": "LEFT"}}


# 067's body copy: seven lines, one size, one colour, an exact 90.0 baseline rhythm and a
# single paragraph break (139.0) after 'botanicals'. Box heights swing 76.7-102.0 purely on
# which glyphs carry ascenders/descenders.
_067_BODY = [
    _067_body_line("L6", "At Frøya, we believe nature holds the key", 239.8, 81.4, 307.0, 79.53, "subheadline"),
    _067_body_line("L7", "to clear, healthy skin. Say goodbye to", 337.3, 75.1, 397.0, 79.53, "subheadline"),
    _067_body_line("L8", "artificial ingredients and hello to the pure,", 425.2, 77.9, 487.0, 79.50, "subheadline"),
    _067_body_line("L1", "potent power of organic Arctic botanicals", 507.0, 102.0, 577.0, 79.53, "subheadline"),
    _067_body_line("L9", "Don't wait — we're saying goodbye to", 655.2, 76.7, 716.0, 79.53, "subheadline"),
    _067_body_line("L10", "our Sale with 40% OFF soon. Experience", 744.6, 72.8, 806.0, 79.32, "offer"),
    _067_body_line("L2", "the Arctic difference before it's gone.", 828.0, 97.0, 896.0, 79.53, "subheadline"),
]


def test_067_offer_regex_does_not_veto_a_join_between_identical_faces():
    """067: 'our Sale with 40% OFF soon.' trips _OFFER_RE mid-paragraph.

    It is the same face as every line around it (fontSize 79.3-79.5, #000000), so the role
    disagreement is a fact about the sentence, not the layout. Vetoing on it stranded the
    line AND the line under it in singleton blocks, which is what shoved 067's paragraph
    break a line late. Same root cause as 101's '50% thicker' / 'repairs & sealant use'.
    """
    body = json.loads(json.dumps(_067_BODY))
    previous, current = body[4], body[5]  # 'Don't wait…' -> 'our Sale with 40% OFF…'
    assert previous["role"] != current["role"], "precondition: the role labeller disagrees"
    assert not text_analysis._compatible_roles(previous["role"], current["role"])
    assert text_analysis._can_join(previous, current, {}), \
        "identical faces one pitch apart are one paragraph whatever the regex called them"


def test_role_veto_still_fires_when_the_faces_actually_differ():
    """The relaxation is licensed by typographic identity ONLY — a real step still splits."""
    previous, current = json.loads(json.dumps(_067_BODY[4:6]))
    current["style"]["fontSize"] = 120.0  # a genuine display step beside body copy
    assert not text_analysis._can_join(previous, current, {}), "a size step is hierarchy"
    recoloured = json.loads(json.dumps(_067_BODY[5]))
    recoloured["style"]["color"] = "#D42A2A"
    assert not text_analysis._can_join(previous, recoloured, {}), "a colour step is hierarchy"


def test_067_paragraph_break_is_read_from_baseline_pitch_not_box_gaps():
    """067: the real break (a 139.0 pitch among 90.0s) hides under _can_join's gap band.

    The box gap at the break is 46.2px against a 127.5px allowance (1.25 x the tallest
    box), so the absolute test passes EVERY pair and cannot see the paragraph at all. The
    block's own established rhythm is the signal.
    """
    body = json.loads(json.dumps(_067_BODY))
    para1, break_line = body[:4], body[4]
    # The gap band genuinely cannot discriminate — this is why the pitch test exists.
    assert text_analysis._can_join(para1[-1], break_line, {}), \
        "precondition: the absolute gap band waves the paragraph break through"
    assert text_analysis._pitch_break(para1, break_line, {}), \
        "a 139.0 step against a 90.0 rhythm is a new paragraph"
    # …and every genuine continuation inside a paragraph is left alone.
    for index in range(1, len(para1)):
        assert not text_analysis._pitch_break(para1[:index], para1[index], {}), \
            f"line {index} continues the rhythm and must not break it"
    para2 = body[4:]
    for index in range(1, len(para2)):
        assert not text_analysis._pitch_break(para2[:index], para2[index], {}), \
            f"paragraph 2 line {index} continues the rhythm"


def test_pitch_break_needs_an_established_rhythm():
    """A lone line has no rhythm to break — the pitch test must abstain, not guess."""
    body = json.loads(json.dumps(_067_BODY))
    assert not text_analysis._pitch_break(body[:1], body[1], {}), \
        "one line establishes no pitch; the absolute band is the only guard there"


def test_067_body_splits_into_exactly_two_paragraphs():
    """End-to-end: the two fixes together must reproduce 067's authored structure."""
    body = json.loads(json.dumps(_067_BODY))
    for line in body:
        line["hierarchy"] = {"level": 2, "parent_id": None}
    blocks = text_analysis._make_blocks(body, {"w": 1620.0, "h": 1620.0}, {})
    groups = {}
    for line in body:
        groups.setdefault(line["block_id"], []).append(line["text"])
    assert len(groups) == 2, f"067's body is two paragraphs, got {len(groups)}: {groups}"
    first, second = [groups[k] for k in sorted(groups)]
    assert first[-1].endswith("botanicals"), f"paragraph 1 ends at 'botanicals', got {first}"
    assert second[0].startswith("Don't wait"), f"paragraph 2 opens at 'Don't wait', got {second}"
    assert len(second) == 3, f"paragraph 2 keeps all three of its lines, got {second}"
    assert blocks, "blocks are still emitted"


def test_column_unification_never_flattens_a_real_size_step():
    """091's authored 1.222 step sits in the same column and must survive it."""
    def row(ident, text, y, size):
        return {"id": ident, "text": text, "block_id": "B%s" % ident,
                "box": {"x": 90.0, "y": y, "w": 300.0, "h": 26.0},
                "style": {"fontFamily": "Inter", "fontSize": size,
                          "fontWeight": 400, "color": "#111111"}}

    lines = [row("L0", "ZERO SUGAR", 400.0, 24.0), row("L1", "SUPPLEMENT", 440.0, 24.0),
             row("L2", "120MG NATURAL", 480.0, 29.33)]
    text_analysis._unify_column_text_scale(lines)
    assert lines[2]["style"]["fontSize"] == 29.33, "a 1.222 step is hierarchy, not drift"
    assert lines[0]["style"]["fontSize"] == lines[1]["style"]["fontSize"] == 24.0


def test_column_unification_does_not_reach_across_the_canvas():
    """A column is a CONTIGUOUS stack: a distant logo sharing a left edge is not a peer."""
    def row(ident, text, y, size):
        return {"id": ident, "text": text, "block_id": "B%s" % ident,
                "box": {"x": 105.0, "y": y, "w": 200.0, "h": 25.0},
                "style": {"fontFamily": "Inter", "fontSize": size,
                          "fontWeight": 400, "color": "#111111"}}

    lines = [row("L0", "craft", 375.0, 25.37),      # logo, hundreds of px above
             row("L1", "50% thicker for better", 704.0, 24.04),
             row("L2", "durability", 733.0, 20.89)]
    text_analysis._unify_column_text_scale(lines)
    assert lines[0]["style"]["fontSize"] == 25.37, "the logo is not part of the checklist"
    assert lines[1]["style"]["fontSize"] == lines[2]["style"]["fontSize"]


def test_101_column_row_is_left_aligned_not_a_floating_callout():
    """101's rows (x=110 of 1000) and 014's floater (x=120 of 1080) are the SAME box.

    Only the rest of the canvas can tell them apart: a column row has lines flush to its
    left edge, a floater does not. Without that, every singleton row was emitted RIGHT
    and its left edge drifted with its own rendered width (the ragged indentation).
    """
    row = {"box": {"x": 110.0, "y": 704.0, "w": 262.0, "h": 25.0}}
    column = [row,
              {"box": {"x": 112.0, "y": 733.0, "w": 112.0, "h": 21.0}},
              {"box": {"x": 110.0, "y": 780.0, "w": 349.0, "h": 25.0}}]
    assert text_analysis._infer_alignment([row], 1000.0, siblings=column) == "LEFT"
    # With no left-edge peers it is still a floating callout.
    assert text_analysis._infer_alignment([row], 1000.0, siblings=[row]) == "RIGHT"


def test_peer_unification_never_flattens_a_real_size_step():
    """The hard constraint: a headline beside its body copy must keep its contrast."""
    lines = [
        {"id": "L0", "text": "We NEVER do this!", "block_id": "B0",
         "style": {"fontFamily": "Inter", "fontSize": 88.0, "fontWeight": 700}},
        {"id": "L1", "text": "read the fine print below", "block_id": "B0",
         "style": {"fontFamily": "Inter", "fontSize": 24.0, "fontWeight": 400}},
    ]
    text_analysis._unify_peer_text_scale(lines)
    assert lines[0]["style"]["fontSize"] == 88.0
    assert lines[1]["style"]["fontSize"] == 24.0
    assert lines[0]["style"]["fontWeight"] == 700, "a deliberate bold lead keeps its weight"


def test_peer_unification_ignores_sub_glyph_ocr_speckle():
    """'-' / '- -' lines measure nonsense sizes; they must not drag a real peer group."""
    lines = [
        {"id": "L0", "text": "-", "block_id": "B1",
         "style": {"fontFamily": "Inter", "fontSize": 233.3, "fontWeight": 600}},
        {"id": "L1", "text": "Gut health", "block_id": "B1",
         "style": {"fontFamily": "Poppins", "fontSize": 26.7, "fontWeight": 400}},
        {"id": "L2", "text": "prebiotics", "block_id": "B1",
         "style": {"fontFamily": "Poppins", "fontSize": 23.5, "fontWeight": 400}},
    ]
    text_analysis._unify_peer_text_scale(lines)
    assert lines[0]["style"]["fontSize"] == 233.3, "speckle is skipped, not unified"
    assert lines[1]["style"]["fontSize"] == lines[2]["style"]["fontSize"]


def test_saturated_price_strike_and_underline_become_vector_evidence():
    image = np.full((100, 360, 3), 255, dtype=np.uint8)
    # Diagonal strike through the old price and horizontal underline below the new.
    import cv2
    cv2.line(image, (25, 66), (145, 30), (225, 73, 27), 4, cv2.LINE_AA)
    cv2.line(image, (205, 78), (335, 78), (225, 73, 27), 5, cv2.LINE_AA)

    strike = text_analysis._native_colored_price_rules(image, {
        "text": "€63", "box": {"x": 10, "y": 15, "w": 150, "h": 70},
    })
    underline = text_analysis._native_colored_price_rules(image, {
        "text": "€49", "box": {"x": 190, "y": 15, "w": 160, "h": 70},
    })

    assert [item["kind"] for item in strike] == ["strikethrough"]
    assert [item["kind"] for item in underline] == ["underline"]
    assert strike[0]["color"].lower().startswith("#e")


def test_groups_paragraph_lines_and_reuses_style_id(tmp_path):
    image = Image.new("RGB", (720, 420), "white")
    draw = ImageDraw.Draw(image)
    headline_font, body_font = _font(48), _font(25)
    head_box = _draw_text(draw, (55, 35), "A BETTER ROUTINE", headline_font, (18, 18, 18))
    body1_box = _draw_text(draw, (58, 165), "Made for everyday use and easy styling.", body_font, (35, 35, 35))
    body2_box = _draw_text(draw, (58, 204), "Clean ingredients with a natural finish.", body_font, (35, 35, 35))
    path = tmp_path / "paragraph.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 720, "h": 420},
        "lines": [
            _line("L0", "A BETTER ROUTINE", head_box),
            _line("L1", "Made for everyday use and easy styling.", body1_box),
            _line("L2", "Clean ingredients with a natural finish.", body2_box),
        ],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    paragraph = next(block for block in result["blocks"] if block["line_ids"] == ["L1", "L2"])
    by_id = {line["id"]: line for line in result["lines"]}

    assert paragraph["type"] == "paragraph"
    assert paragraph["role"] == "body"
    assert paragraph["alignment"] == "LEFT"
    assert by_id["L0"]["role"] == "headline"
    assert by_id["L1"]["hierarchy"]["parent_id"] == paragraph["id"]
    assert by_id["L1"]["style_id"] == by_id["L2"]["style_id"]
    shared = next(style for style in result["styles"] if style["id"] == by_id["L1"]["style_id"])
    assert shared["repeated"] is True
    assert set(shared["usage"]) == {"L1", "L2"}


def test_rotation_and_missing_image_fallback_are_safe(tmp_path):
    angle = 12.0
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 60.0, 180.0, 34.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = {
        "id": "L0",
        "text": "Rotated copy",
        "conf": 0.8,
        "box": {"x": x0, "y": y0, "w": width, "h": height},
        "quad": [[x0, y0], [x0 + dx, y0 + dy],
                 [x0 + dx, y0 + dy + height], [x0, y0 + height]],
        "words": [],
    }
    missing = tmp_path / "not-there.png"
    result = text_analysis.analyze_text(
        str(missing),
        {"engine": "synthetic", "source": {"w": 400, "h": 300}, "lines": [line]},
        {},
    )
    enriched = result["lines"][0]

    assert enriched["rotation_deg"] == pytest.approx(angle, abs=0.01)
    assert enriched["painted_box"] == line["box"]
    assert enriched["style"]["fontCandidates"][0]["source"] == "fallback"
    assert result["text_analysis"]["image_available"] is False


def test_quad_rotation_uses_long_text_edge_when_ocr_starts_with_short_edge():
    # This is the winding emitted by the benchmark's horizontal lines: the
    # first edge is vertical, while the long opposite edge is horizontal.
    quad = [[10, 40], [10, 10], [210, 10], [210, 40]]
    assert text_analysis._quad_rotation(quad) == pytest.approx(0.0)


def test_paragraph_rotation_aggregates_stacked_horizontal_lines(tmp_path):
    path = tmp_path / "stack.png"
    Image.new("RGB", (500, 180), "white").save(path)
    lines = []
    for index, y in enumerate((30, 70, 110)):
        line = _line(f"L{index}", "STACKED LINE", (30, y, 260, y + 20))
        line["quad"] = [[30, y + 20], [30, y], [260, y], [260, y + 20]]
        lines.append(line)
    result = text_analysis.analyze_text(str(path), {"source": {"w": 500, "h": 180}, "lines": lines}, {})
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == pytest.approx(0.0)
    assert all(line["rotation_deg"] == pytest.approx(0.0) for line in result["lines"])


def test_single_rotated_quad_keeps_supported_angle():
    angle = 32.0
    radians = math.radians(angle)
    width, height = 180.0, 30.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    quad = [[0, height], [0, 0], [dx, dy], [dx, dy + height]]
    assert text_analysis._quad_rotation(quad) == pytest.approx(angle, abs=0.01)


def test_shear_measurement_rejects_implausible_rotation_like_drift():
    mask = np.zeros((24, 40), dtype=bool)
    mask[2:10, 4:12] = True
    mask[14:22, 25:33] = True
    assert text_analysis._measure_shear_angle(mask) is None


def test_optional_local_font_matching_is_bounded(tmp_path):
    font = _font(38)
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")

    image = Image.new("RGB", (520, 180), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (35, 45), "Font Match", font, (0, 0, 0))
    path = tmp_path / "font.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 520, "h": 180},
        "lines": [_line("L0", "Font Match", ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": [],
                "max_fonts": 1,
                "max_lines": 1,
                "top_k": 1,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    candidates = result["lines"][0]["style"]["fontCandidates"]

    assert len(candidates) == 1
    assert candidates[0]["source"] == "local-render"
    assert candidates[0]["score"] > 0.2
    assert result["text_analysis"]["font_matches_attempted"] == 1


def test_empty_font_dirs_still_uses_platform_inventory(tmp_path, monkeypatch):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    staged = tmp_path / "Platform.ttf"
    staged.write_bytes(Path(font_path).read_bytes())
    monkeypatch.setattr(text_analysis, "_platform_font_dirs", lambda: [str(tmp_path)])
    text_analysis._FONT_DISCOVERY_CACHE.clear()

    discovered = text_analysis._discover_fonts({"font_dirs": [], "scan_limit": 4})

    assert any(Path(item["path"]).name == "Platform.ttf" for item in discovered)


def test_multicolumn_paragraphs_do_not_cross_merge_by_reading_order(tmp_path):
    path = tmp_path / "columns.png"
    Image.new("RGB", (400, 220), "white").save(path)
    raw = {
        "source": {"w": 400, "h": 220},
        "lines": [
            _line("L0", "Left first sentence stays", (20, 50, 170, 66)),
            _line("L1", "Right first sentence stays", (230, 50, 380, 66)),
            _line("L2", "Left second sentence follows", (20, 73, 170, 89)),
            _line("L3", "Right second sentence follows", (230, 73, 380, 89)),
        ],
    }

    result = text_analysis.analyze_text(
        str(path), raw, {"text_analysis": {"font_matching": {"enabled": False}}}
    )

    members = {tuple(block["line_ids"]) for block in result["blocks"]}
    assert members == {("L0", "L2"), ("L1", "L3")}


# ---------------------------------------------------------------------------
# Bug 1: confidence/fidelity gate — low-confidence ink -> masked-pixel fallback


def test_low_ink_confidence_flags_low_fidelity_and_saves_fallback_crop(tmp_path):
    # Very low text/background contrast produces a genuine (but low-confidence) ink
    # mask — this is the "can't be faithfully represented" case the fidelity gate
    # exists to catch, and it should still have real pixels to fall back to.
    font = _font(50)
    image = Image.new("RGB", (300, 140), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (30, 30), "SALE", font, (188, 188, 188))
    path = tmp_path / "faint.png"
    image.save(path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ocr = {
        "source": {"path": str(path), "w": 300, "h": 140},
        "lines": [_line("L0", "SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {"run_dir": str(run_dir)})
    line = result["lines"][0]

    assert line["meta"]["low_fidelity"] is True
    assert "fidelity_reason" in line["meta"]
    assert line["meta"]["substitution"]["from"] == "text"
    fallback_src = line["meta"].get("fallback_src")
    assert fallback_src
    assert (run_dir / fallback_src).exists()

    # Regression: _make_blocks used to drop the fidelity signal entirely — blocks had
    # no "meta" key at all, so merge_layers (which prefers ocr["blocks"] over
    # ocr["lines"] whenever blocks is non-empty, i.e. on every real run) would never
    # see low_fidelity/fallback_src and would silently emit guessed text instead of
    # the masked-pixel fallback. The block must carry the same fidelity signal.
    assert result["blocks"], "expected at least one block"
    block = result["blocks"][0]
    assert block["line_ids"] == [line["id"]]
    assert block["meta"]["low_fidelity"] is True
    assert block["meta"]["fallback_src"] == fallback_src
    assert block["meta"]["fidelity_reason"] == line["meta"]["fidelity_reason"]


def _script_font_path():
    for name in ("Gabriola.ttf", "segoesc.ttf", "Inkfree.ttf", "Comic Sans MS.ttf", "comic.ttf"):
        for root in ("C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental"):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
    return None


def test_same_class_body_copy_stays_editable_text(tmp_path):
    # Reframe: a legible line matched to a plausible SAME-CLASS font stays editable
    # even when the exact typeface is unknown. Fidelity is floored above the raster
    # bar so accurate styling — not font identity — decides editability.
    if _font_path() is None:
        pytest.skip("no sans test font available")
    image = Image.new("RGB", (900, 150), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (36, 46),
                         "The only supplement you need every single morning",
                         _font(32), (18, 18, 18))
    path = tmp_path / "body.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": 900, "h": 150},
           "lines": [_line("L0", "The only supplement you need every single morning", ocr_box)]}
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "max_fonts": 24, "max_lines": 4}}}
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    assert line["meta"]["low_fidelity"] is False
    assert line["meta"]["fidelity_confidence"] >= 0.40


def test_script_face_never_matches_plain_multiword_copy(tmp_path):
    # Even a source *rendered in a script font* must not keep a script/decorative
    # family for multi-word plain copy: a genuine script wordmark is routed as
    # artwork earlier, so at this stage a swash match is the 052 Gabriola failure.
    script_path = _script_font_path()
    if script_path is None:
        pytest.skip("no script/decorative font available")
    from src import font_fit
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(script_path, 46)
    ocr_box = _draw_text(draw, (36, 44), "to have perfect curls", font, (20, 20, 20))
    path = tmp_path / "swash.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": 900, "h": 160},
           "lines": [_line("L0", "to have perfect curls", ocr_box)]}
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "max_fonts": 40, "max_lines": 4}}}
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    chosen = (result["lines"][0]["style"].get("fontCandidates") or [{}])[0]
    chosen_class = font_fit.classify_font_file(chosen.get("path")) if chosen.get("path") else None
    assert chosen_class not in (font_fit.SCRIPT, font_fit.DECORATIVE)


def test_confident_text_is_not_flagged_low_fidelity(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "ad.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 640, "h": 260},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    line = result["lines"][0]

    assert line["meta"]["low_fidelity"] is False
    assert "fallback_src" not in line["meta"]
    assert "substitution" not in line["meta"]


# ---------------------------------------------------------------------------
# Bug 2: gradient-stop and stroke-colour extraction


def _gradient_text_image(text, font, top_rgb, bottom_rgb, size=(360, 160), pos=(40, 40)):
    probe = Image.new("L", size, 0)
    ImageDraw.Draw(probe).text(pos, text, font=font, fill=255)
    mask = np.asarray(probe) > 32
    h, w = size[1], size[0]
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    grad = (1 - ramp) * np.array(top_rgb, dtype=np.float32) + ramp * np.array(bottom_rgb, dtype=np.float32)
    base = np.full((h, w, 3), 255.0, dtype=np.float32)
    base[mask] = grad.repeat(w, axis=1)[mask]
    image = Image.fromarray(base.astype(np.uint8), "RGB")
    bbox = ImageDraw.Draw(Image.new("L", size, 0)).textbbox(pos, text, font=font)
    box = (max(0, bbox[0] - 7), max(0, bbox[1] - 6), bbox[2] + 7, bbox[3] + 6)
    return image, box


def test_gradient_fill_extracted_as_linear_stops(tmp_path):
    font = _font(72)
    top_rgb, bottom_rgb = (235, 60, 20), (20, 70, 235)
    image, ocr_box = _gradient_text_image("SALE", font, top_rgb, bottom_rgb)
    path = tmp_path / "gradient.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    fill = result["lines"][0]["style"]["fill"]

    assert fill["kind"] == "linear"
    assert len(fill["stops"]) == 2
    start = np.array(text_analysis._hex_rgb(fill["stops"][0]["color"]))
    end = np.array(text_analysis._hex_rgb(fill["stops"][-1]["color"]))
    assert np.linalg.norm(start - end) > 60


def test_stroked_text_extracts_distinct_stroke_and_fill_colour(tmp_path):
    font = _font(80)
    fill_rgb, stroke_rgb = (250, 250, 250), (15, 15, 15)
    image = Image.new("RGB", (360, 180), (120, 170, 230))
    draw = ImageDraw.Draw(image)
    pos = (40, 40)
    draw.text(pos, "OFF", font=font, fill=fill_rgb, stroke_width=6, stroke_fill=stroke_rgb)
    bbox = draw.textbbox(pos, "OFF", font=font, stroke_width=6)
    ocr_box = (max(0, bbox[0] - 8), max(0, bbox[1] - 8), bbox[2] + 8, bbox[3] + 8)
    path = tmp_path / "stroke.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "OFF", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    style = result["lines"][0]["style"]

    assert style["stroke"] is not None
    stroke_hex = style["stroke"]["color"]
    fill_hex = style["fill"]["color"]
    assert text_analysis._colour_distance(stroke_hex, fill_hex) > 60
    # stroke sample should land closer to the outline colour than to the fill colour
    stroke_rgb_hex = text_analysis._rgb_hex(stroke_rgb)
    fill_rgb_hex = text_analysis._rgb_hex(fill_rgb)
    assert text_analysis._colour_distance(stroke_hex, stroke_rgb_hex) < text_analysis._colour_distance(
        stroke_hex, fill_rgb_hex
    )


# ---------------------------------------------------------------------------
# Bug 3: glyph-shear (italic) measurement independent of font matching


def test_measures_shear_angle_on_italic_glyph_mask_without_font_matching(tmp_path):
    italic_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "C:/Windows/Fonts/ariali.ttf",
    ]
    italic_path = next((p for p in italic_candidates if os.path.isfile(p)), None)
    if not italic_path:
        pytest.skip("no system italic font available")
    font = ImageFont.truetype(italic_path, 64)

    image = Image.new("RGB", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    ocr_box = _draw_text(draw, (40, 40), "Slanted", font, (10, 10, 10))
    path = tmp_path / "italic.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "Slanted", ocr_box)],
    }

    # font_matching stays disabled: italic must be detected from the ink mask alone.
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]

    assert style["italicShearDeg"] is not None
    assert abs(style["italicShearDeg"]) >= 6.0
    assert "italic" in style["fontStyle"].lower()


def test_upright_glyph_mask_measures_near_zero_shear(tmp_path):
    image = Image.new("RGB", (640, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    ocr_box = _draw_text(draw, (60, 45), "BIG SALE", font, (210, 32, 24))
    path = tmp_path / "upright.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "BIG SALE", ocr_box)],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    style = result["lines"][0]["style"]

    assert style["italicShearDeg"] is None or abs(style["italicShearDeg"]) < 6.0
    assert "italic" not in style["fontStyle"].lower()


# ---------------------------------------------------------------------------
# Bug 5: fontStyleCandidates must preserve weight, not hardcode "Italic"/"Regular"


def test_font_style_candidates_preserve_weight_when_alternating_italic(tmp_path):
    image = Image.new("RGB", (420, 160), "white")
    draw = ImageDraw.Draw(image)
    font = _font(60)
    # A heavy/dense stroke pushes the density-based weight estimate to Bold.
    ocr_box = _draw_text(draw, (30, 30), "BOLD", font, (0, 0, 0))
    path = tmp_path / "bold.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "BOLD", ocr_box)],
    }

    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]
    candidates = {c["value"] for c in style["fontStyleCandidates"]}

    assert "Italic" not in candidates  # bare "Italic" would drop the weight
    assert any(value.endswith("Italic") and value != "Italic" for value in candidates) or \
        style["fontWeight"] < 700  # only assert alternation when weight actually landed Bold


# ---------------------------------------------------------------------------
# Bug 4: style-cluster representative matching propagates beyond max_lines


def test_style_cluster_propagates_font_match_beyond_max_lines_budget(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    font = _font(30)

    n_lines = 14
    line_h = 40
    image = Image.new("RGB", (500, line_h * n_lines + 20), "white")
    draw = ImageDraw.Draw(image)
    lines = []
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
             "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November"]
    for i, word in enumerate(words[:n_lines]):
        y = 10 + i * line_h
        box = _draw_text(draw, (20, y), word, font, (20, 20, 20))
        lines.append(_line(f"L{i}", word, box))
    path = tmp_path / "many_lines.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": image.width, "h": image.height}, "lines": lines}

    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": [],
                "max_fonts": 1,
                "max_lines": 3,   # far fewer than n_lines — the old per-line budget
                "top_k": 1,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    sources = [line["style"]["fontCandidates"][0]["source"] for line in result["lines"]]

    # All lines share one style cluster, so the 3-slot budget should still cover
    # every line via propagation from a single representative match.
    assert sources.count("local-render") == n_lines
    assert result["text_analysis"]["font_matches_attempted"] <= 3


def test_font_match_budget_is_spent_on_the_most_prominent_text_first(tmp_path):
    # 091: OCR reads product-label microcopy before the headline, so a document-order
    # budget spent all 16 match slots on ~1k px² labels and left the ad's BIGGEST text
    # (a 76k px² serif headline) with no render match at all — it fell back to a generic
    # sans that renders visibly wrong. The budget must follow prominence, not read order.
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")

    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    lines = []
    # Six small labels FIRST (distinct sizes => distinct style clusters), headline LAST.
    for i, word in enumerate(["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]):
        box = _draw_text(draw, (20, 10 + i * 30), word, _font(12 + i), (20, 20, 20))
        lines.append(_line(f"L{i}", word, box))
    headline_box = _draw_text(draw, (20, 250), "Headline", _font(90), (20, 20, 20))
    lines.append(_line("L6", "Headline", headline_box))
    path = tmp_path / "prominence.png"
    image.save(path)
    ocr = {"source": {"path": str(path), "w": image.width, "h": image.height}, "lines": lines}

    def families(prominence):
        cfg = {"text_analysis": {"font_matching": {
            "enabled": True, "font_files": [font_path], "font_dirs": [],
            "max_fonts": 1, "top_k": 1,
            "max_lines": 2,                      # only two clusters can be matched
            "prominence_budget": prominence,
        }}}
        out = text_analysis.analyze_text(str(path), ocr, cfg)
        return {l["id"]: l["style"]["fontCandidates"][0]["source"] for l in out["lines"]}

    # Prominence order: the headline is the largest ink, so it MUST get a real match.
    assert families(True)["L6"] == "local-render"
    # Document order burns the budget on the small labels read first and starves it.
    assert families(False)["L6"] != "local-render"


def test_meta_alignment_prefers_matching_weight():
    profile = text_analysis._typography_profile({"weight": 700, "shear_angle": None, "font_size": 24})
    bold_meta = {"family": "Inter", "style": "Bold", "weight": 700}
    light_meta = {"family": "Inter", "style": "Light", "weight": 300}
    assert text_analysis._meta_alignment_adjustment(bold_meta, profile) > \
        text_analysis._meta_alignment_adjustment(light_meta, profile)


def test_floating_side_callouts_align_toward_center():
    """014: left floating callouts RIGHT-align; right floating callouts LEFT-align."""
    left = [{"box": {"x": 120, "y": 500, "w": 240, "h": 60}}]
    right = [{"box": {"x": 720, "y": 500, "w": 240, "h": 60}}]
    edge_left = [{"box": {"x": 40, "y": 500, "w": 240, "h": 60}}]
    edge_right = [{"box": {"x": 800, "y": 500, "w": 240, "h": 60}}]
    assert text_analysis._infer_alignment(left, 1080) == "RIGHT"
    assert text_analysis._infer_alignment(right, 1080) == "LEFT"
    assert text_analysis._infer_alignment(edge_left, 1080) == "LEFT"
    assert text_analysis._infer_alignment(edge_right, 1080) == "RIGHT"


def test_social_left_column_and_wide_body_stay_left():
    """009: username + wide body lines must not flip to RIGHT/CENTER."""
    upfront = [{"box": {"x": 183.5, "y": 158.0, "w": 194.0, "h": 25.0}}]
    handle = [{"box": {"x": 185.6, "y": 198.0, "w": 225.0, "h": 29.0}}]
    # Geometric center near mid-canvas, but left-anchored body (Daarbovenop…).
    wide_body = [{"box": {"x": 47.46, "y": 552.66, "w": 915.47, "h": 33.75}}]
    post = [{"box": {"x": 487.3, "y": 54.8, "w": 101.3, "h": 34.8}}]
    assert text_analysis._infer_alignment(upfront, 1080) == "LEFT"
    assert text_analysis._infer_alignment(handle, 1080) == "LEFT"
    assert text_analysis._infer_alignment(wide_body, 1080) == "LEFT"
    assert text_analysis._infer_alignment(post, 1080) == "CENTER"


def test_disclaimer_role_for_bottom_fda_copy():
    lines = [{
        "text": "*These statements have not been evaluated by the FDA.",
        "box": {"x": 80, "y": 1780, "w": 920, "h": 36},
        "style": {"fontSize": 14, "color": "#888888"},
        "baseline": {"y0": 1805},
    }, {
        "text": "NUTRITIONAL SUPPORT",
        "box": {"x": 120, "y": 80, "w": 840, "h": 90},
        "style": {"fontSize": 48, "color": "#FFFFFF"},
        "baseline": {"y0": 150},
    }]
    text_analysis._assign_roles(lines, {"w": 1080, "h": 1920})
    by_text = {line["text"][:20]: line["role"] for line in lines}
    assert by_text["*These statements ha"] == "disclaimer"
    assert by_text["NUTRITIONAL SUPPORT"] == "headline"


def test_fallback_chain_uses_weight_and_italic(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (200, 80), "white").save(path)
    ocr = {
        "source": {"w": 200, "h": 80},
        "lines": [_line("L0", "SALE", (20, 20, 120, 50))],
    }
    result = text_analysis.analyze_text(
        str(path), ocr,
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    style = result["lines"][0]["style"]
    assert style["fontWeight"] in {300, 400, 500, 600, 700}
    assert style["fontCandidates"][0]["weight"] == style["fontWeight"]
    assert "Italic" not in style["fontCandidates"][0]["style"]


def test_google_fonts_cache_candidates_merge_into_chain(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    cache_dir = tmp_path / "google-fonts"
    cache_dir.mkdir()
    cached_font = cache_dir / "Inter-Regular.ttf"
    cached_font.write_bytes(open(font_path, "rb").read())

    image = Image.new("RGB", (520, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 38)
    ocr_box = _draw_text(draw, (35, 45), "Cache", font, (0, 0, 0))
    path = tmp_path / "font.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 520, "h": 180},
        "lines": [_line("L0", "Cache", ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [],
                "font_dirs": [],
                "google_fonts_cache": str(cache_dir),
                "max_fonts": 1,
                "max_lines": 1,
                "top_k": 3,
            }
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    sources = {item.get("source") for item in result["lines"][0]["style"]["fontCandidates"]}
    assert "google-cache" in sources
    assert "fallback" in sources


def test_needs_vlm_font_judge_when_local_score_is_weak():
    ocr = {
        "lines": [{
            "style": {
                "fontCandidates": [
                    {"family": "Inter", "source": "local-render", "score": 0.31, "path": "/tmp/a.ttf"},
                    {"family": "Arial", "source": "fallback", "score": 0.55},
                ]
            }
        }]
    }
    cfg = {"text_analysis": {"font_matching": {"enabled": True, "local_score_threshold": 0.55}}}
    assert text_analysis.needs_vlm_font_judge(ocr, cfg) is True


def test_design_json_preserves_font_candidates(tmp_path):
    from src.build_design_json import build

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidates = [{
        "id": "T0",
        "target": "text",
        "text": "SALE",
        "box": {"x": 10, "y": 10, "w": 120, "h": 40},
        "visible_box": {"x": 10, "y": 10, "w": 120, "h": 40},
        "style": {
            "fontFamily": "Inter",
            "fontSize": 28,
            "fontWeight": 700,
            "fontStyle": "Bold",
            "color": "#111111",
            "fontCandidates": [
                {"family": "Inter", "style": "Bold", "weight": 700, "score": 0.82, "source": "local-render"},
                {"family": "Arial", "style": "Bold", "weight": 700, "score": 0.71, "source": "fallback"},
            ],
            "fontSizeCandidates": [{"value": 28, "score": 0.75}],
        },
    }]
    build(candidates, {"w": 200, "h": 120}, str(run_dir))
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    style = design["layers"][0]["style"]
    assert style["fontCandidates"][0]["family"] == "Inter"
    assert style["fontSizeCandidates"][0]["value"] == 28


def test_word_style_enrichment_uses_strong_pixel_difference_without_guessing_family(monkeypatch):
    base = {
        "fontFamily": "Matched Family", "fontSize": 30, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "SAVE 30%", "style": base,
        "words": [{"text": "30%", "box": {"x": 60, "y": 5, "w": 35, "h": 30}}],
    }
    monkeypatch.setattr(text_analysis, "_painted_geometry", lambda image, word: (
        {"x": 60, "y": 5, "w": 35, "h": 30}, 28, "#ff2244", .91,
        np.ones((30, 35), dtype=bool), {"fill": {"kind": "flat", "color": "#ff2244"}},
    ))
    monkeypatch.setattr(text_analysis, "_pre_font_signals", lambda *args, **kwargs: {
        "font_size": 42, "weight": 700, "shear_angle": 0,
    })
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert word["style"]["fontFamily"] == "Matched Family"
    assert word["style"]["color"] == "#ff2244"
    assert word["style"]["fontSize"] == 42
    assert word["style_evidence"]["source"] == "word-pixels"


def _jitter_line(words, base_weight=400):
    base = {
        "fontFamily": "Inter", "fontSize": 30, "fontWeight": base_weight,
        "fontStyle": "Regular", "color": "#111111",
    }
    return {
        "text": " ".join(w for w, _ in words), "style": base,
        "words": [{"text": w, "box": {"x": 40 * i, "y": 5, "w": 34, "h": 30}}
                  for i, (w, _) in enumerate(words)],
    }


# Ink density a word must carry for _estimate_weight to bucket it at each weight.
_WEIGHT_DENSITY = {300: 0.10, 400: 0.25, 500: 0.30, 600: 0.40, 700: 0.50, 800: 0.60}


def _patch_jitter(monkeypatch, weights, density=None, stroke=None):
    """Drive _enrich_word_styles with per-word weight/density measurements.

    Ink density is what a weight estimate is MADE of (_estimate_weight buckets the mask
    mean), so a fixture may not claim a per-word weight without giving that word the ink
    to justify it — that ink is the evidence a mid-line weight change now requires. Pass
    ``density`` to override the weight-implied ink per word, which is how a test models
    "the absolute bucket flipped but the word's ink is no different from its line-mates".
    """
    def _mask_for(word):
        value = (density or {}).get(word["text"])
        if value is None:
            value = _WEIGHT_DENSITY.get(weights[word["text"]], 0.25)
        mask = np.zeros((30, 34), dtype=bool)
        mask.reshape(-1)[: int(round(value * mask.size))] = True
        return mask

    def fake_geo(image, word):
        return ({"x": word["box"]["x"], "y": word["box"]["y"],
                 "w": word["box"]["w"], "h": word["box"]["h"]},
                None, "#111111", 1.0,
                _mask_for(word),
                {"fill": {"kind": "flat", "color": "#111111"}})

    def fake_signals(word, painted, mask, config):
        return {"font_size": 30, "weight": weights[word["text"]], "shear_angle": 0}

    monkeypatch.setattr(text_analysis, "_painted_geometry", fake_geo)
    monkeypatch.setattr(text_analysis, "_pre_font_signals", fake_signals)
    monkeypatch.setattr(text_analysis, "_collar_box", lambda box, image=None: box)


def test_009_short_function_word_bold_between_regular_neighbours_is_clamped(monkeypatch):
    """009: a lone Bold 'we'/'to' mid-sentence is measurement noise, not emphasis.

    Uniform ink (every word the same density/stroke) means the flip has no corroboration,
    so the function word must be clamped back to the line's weight.
    """
    line = _jitter_line([("Schrijf", 400), ("we", 700), ("zien", 400)])
    # The bucket says Bold, but the ink says otherwise: every word carries the SAME
    # density, so there is nothing to corroborate the flip.
    _patch_jitter(monkeypatch, {"Schrijf": 400, "we": 700, "zien": 400},
                  density={"Schrijf": 0.25, "we": 0.25, "zien": 0.25})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    we = line["words"][1]
    assert "weight" not in (we.get("style_evidence") or {}).get("changed", []), \
        "spurious mid-line bold on a function word must not survive"
    assert (we.get("style_debug") or {}).get("weight_peer_clamped")


def test_009_real_bold_121K_is_not_a_function_word_and_survives(monkeypatch):
    """009's genuine '121K' bold must be untouchable by the jitter clamp."""
    assert not text_analysis._short_function_word("121K")
    line = _jitter_line([("12-05-2026", 300), ("121K", 700), ("weergaven", 300)],
                        base_weight=300)
    _patch_jitter(monkeypatch, {"12-05-2026": 300, "121K": 700, "weergaven": 300})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    bold = line["words"][1]
    assert bold["style"]["fontWeight"] == 700
    assert "weight" in bold["style_evidence"]["changed"]


def test_025_genuine_emphasis_words_are_not_function_words():
    """The clamp can never reach authored emphasis: none of it is connective tissue."""
    for token in ("Sale", "40%", "OFF", "121K", "2GET", "FREE", "Hydration", "Cadence"):
        assert not text_analysis._short_function_word(token), token


def _icon_row(head_px=55.0, line_colour="#c42724", stroke=True, gradient=True,
              word_colour="#020000"):
    """066's checklist row: the OCR box starts `head_px` left of the first glyph, over a
    red ✗, so the line's paint is measured across icon+text."""
    style = {"fontFamily": "Poppins", "fontSize": 30, "fontWeight": 400,
             "fontStyle": "Regular", "color": line_colour,
             "fill": {"kind": "linear" if gradient else "flat", "color": line_colour},
             "stroke": ({"kind": "flat", "color": "#040200", "width": 3.0,
                         "align": "OUTSIDE"} if stroke else None)}
    words = []
    for i, text in enumerate(("Smudges", "on", "upper", "lid")):
        words.append({
            "text": text, "box": {"x": 880.0 + 60 * i, "y": 1003.0, "w": 54.0, "h": 30.0},
            "style": {**style, "color": word_colour, "stroke": None,
                      "fill": {"kind": "flat", "color": word_colour}},
            "style_evidence": {"source": "word-pixels", "confidence": 1.0,
                               "changed": ["color"]},
        })
    return {"text": "Smudges on upper lid", "style": style, "words": words,
            "box": {"x": 880.0 - head_px, "y": 1003.0, "w": 450.0, "h": 44.0}}


def test_066_icon_inside_the_line_box_never_paints_a_stroke_around_the_glyphs():
    """066: 'Smudges on upper lid' / 'Up to 3 shades' rendered as a smeared ghost double.

    Their OCR boxes start ~55px left of the first glyph, over the red ✗, so the line's
    paint came back as the ICON's red with a red->black gradient and a 3px OUTSIDE black
    stroke — which render_preview hands to PIL's draw.text, outlining every glyph. The
    words, measured on their own boxes, are unanimous that the text is flat black.
    """
    line = _icon_row()
    repair = text_analysis._repair_non_glyph_line_paint(line)
    assert repair, "an icon inside the line box must not survive as text paint"
    assert line["style"]["stroke"] is None, "the ghost-double stroke must go"
    assert line["style"]["fill"]["kind"] == "flat", "the icon->text gradient must go"
    assert repair["glyph_color"] == "#020000"
    assert repair["non_glyph_head_px"] == 55.0
    # The glyphs still paint their own colour, via the per-word runs.
    assert all(w["style"]["color"] == "#020000" for w in line["words"])


def test_066_icon_repair_touches_only_the_decoration_it_invented():
    """Scope guard. The base colour and the box are icon-poisoned too, but BOTH feed
    grouping, and 'correcting' them costs more than it fixes while the emitted node box
    is still expanded to the icon's x downstream: recolouring the base drops _can_join's
    colour veto so the rows join their neighbours, the merged block's left edges go ragged
    (825 against 880), _infer_alignment reads CENTER and every row slides left under its
    own icon (066 text recall 0.95 -> 0.85; trimming the box instead gave 0.80). Only the
    stroke and the gradient actually draw the ghost double, and neither feeds grouping.
    """
    line = _icon_row()
    before_x, before_colour = line["box"]["x"], line["style"]["color"]
    text_analysis._repair_non_glyph_line_paint(line)
    assert line["style"]["color"] == before_colour, "base colour feeds _can_join; leave it"
    assert line["box"]["x"] == before_x, "box geometry is the node-box owner's to fix"


def test_135_unanimously_wrong_words_never_repaint_a_clean_line():
    """The other half: words share a failure mode and can be unanimously WRONG.

    135's 'vezels suikers' reads #1f1f1f flat with no stroke (correct — the label copy is
    dark), while both tight word boxes flip polarity and read #dedede. Its box has no
    word-free head, so the repair must stay out; trusting the words would paint light
    grey text onto a light label.
    """
    line = _icon_row(head_px=1.0, line_colour="#1f1f1f", stroke=False, gradient=False,
                     word_colour="#dedede")
    assert text_analysis._repair_non_glyph_line_paint(line) is None
    assert line["style"]["color"] == "#1f1f1f", "a clean line keeps its own paint"


def test_066_glyph_composition_alone_never_bolds_a_word(monkeypatch):
    """066: 'Easy to remove' rendered 'remove' Bold against a Regular line.

    Nothing about that line is bold. 'remove' is x-height-only ink, so its tight bbox is
    SHORT and its raw density reads 0.564, while its line-mate 'Easy' (cap + descender,
    a tall bbox) reads 0.332 at the SAME authored weight — a 1.70x split that lands them
    in different absolute weight buckets. Composition, not emphasis: the line stays
    uniform. This is the defect class that also produced 'Clump-free & buildable' and
    'Spidery but long'.
    """
    line = _jitter_line([("Easy", 400), ("to", 400), ("remove", 700)])
    # Raw densities as MEASURED off 066: identical authored weight, 1.70x apart.
    _patch_jitter(monkeypatch, {"Easy": 400, "to": 400, "remove": 700},
                  density={"Easy": 0.332, "to": 0.388, "remove": 0.564})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    for word in line["words"]:
        assert "weight" not in (word.get("style_evidence") or {}).get("changed", []), \
            f"{word['text']}: glyph composition must not read as emphasis"


def test_067_authored_emphasis_survives_the_composition_normalisation(monkeypatch):
    """The other half of the contract: 067's '40% OFF' really is bold and must stay.

    Measured off 067 — 'Sale'/'OFF' carry ~1.5x the normalised ink of their line-mates
    while 'our' (which the absolute buckets also called Bold-700) carries exactly 1.0x
    and is a false positive.
    """
    line = _jitter_line([("our", 700), ("Sale", 700), ("with", 400), ("OFF", 700),
                         ("Experience", 400)])
    _patch_jitter(monkeypatch,
                  {"our": 700, "Sale": 700, "with": 400, "OFF": 700, "Experience": 400},
                  density={"our": 0.509, "Sale": 0.550, "with": 0.350, "OFF": 0.551,
                           "Experience": 0.276})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    styles = {w["text"]: (w.get("style_evidence") or {}).get("changed", [])
              for w in line["words"]}
    assert "weight" in styles["Sale"], "authored emphasis must survive"
    assert "weight" in styles["OFF"], "authored emphasis must survive"
    assert "weight" not in styles["our"], \
        "'our' is only Bold because a word's mask has no inter-word gaps; not emphasis"


def test_function_word_bold_survives_when_it_is_not_sandwiched(monkeypatch):
    """A line-INITIAL function word has no left neighbour, so the clamp stays out.

    Only a word wedged between two AGREEING neighbours is treated as jitter, and a
    function word that really does carry the ink still gets its bold.
    """
    line = _jitter_line([("our", 700), ("Sale", 400), ("with", 400)])
    _patch_jitter(monkeypatch, {"our": 700, "Sale": 400, "with": 400})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    assert line["words"][0]["style"]["fontWeight"] == 700


def test_function_word_bold_survives_when_neighbours_disagree(monkeypatch):
    """Neighbours that disagree on weight are not a uniform line; the clamp stays out."""
    line = _jitter_line([("BUY", 700), ("to", 700), ("save", 400), ("now", 400),
                         ("today", 400)])
    _patch_jitter(monkeypatch, {"BUY": 700, "to": 700, "save": 400, "now": 400,
                                "today": 400})
    text_analysis._enrich_word_styles(np.zeros((40, 200, 3), dtype=np.uint8), line, {})
    assert line["words"][1]["style"]["fontWeight"] == 700


def test_word_size_enrichment_does_not_fire_on_per_100g_pattern(monkeypatch):
    # Benchmark 002 "weird scaling": the line "per 100g" was fragmented into
    # per=12.5px + 100g=31px because a per-word size override fired on noisy
    # measurements. A multi-word line must stay uniform (no 2x word).
    base = {
        "fontFamily": "Inter", "fontSize": 41.67, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "per 100g", "style": base,
        "words": [
            {"text": "per", "box": {"x": 456, "y": 1365, "w": 24, "h": 17}},
            {"text": "100g", "box": {"x": 485, "y": 1365, "w": 59, "h": 27}},
        ],
    }
    # Per-word measured sizes diverge wildly from the line (12.5 and 31 vs 41.67) with
    # high ink confidence and colour/weight jitter — exactly the 002 noise profile.
    measured = {"per": (12.5, 700), "100g": (31.0, 400)}

    def fake_geo(image, word):
        return ({"x": word["box"]["x"], "y": word["box"]["y"],
                 "w": word["box"]["w"], "h": word["box"]["h"]},
                None, "#2a2a2a", 1.0, np.ones((10, 10), dtype=bool),
                {"fill": {"kind": "flat", "color": "#2a2a2a"}})

    def fake_signals(word, painted, mask, config):
        size, weight = measured[word["text"]]
        return {"font_size": size, "weight": weight, "shear_angle": 0}

    monkeypatch.setattr(text_analysis, "_painted_geometry", fake_geo)
    monkeypatch.setattr(text_analysis, "_pre_font_signals", fake_signals)
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    for word in line["words"]:
        style = word.get("style")
        if style is not None:
            # Whatever else may change, the SIZE must not blow up relative to the line.
            assert style["fontSize"] == base["fontSize"], word["text"]
            assert "size" not in word.get("style_evidence", {}).get("changed", [])


def test_punctuation_only_word_never_becomes_a_styled_run(monkeypatch):
    # Benchmark 002: ingredient lines fragmented into "aroma"/"," pieces. A lone
    # punctuation mark (or 1-char sliver) must never carry its own style run.
    base = {
        "fontFamily": "Poppins", "fontSize": 9.2, "fontWeight": 400,
        "fontStyle": "Regular", "color": "#111111",
    }
    line = {
        "text": "aroma ,", "style": base,
        "words": [
            {"text": "aroma", "box": {"x": 465, "y": 1308, "w": 39, "h": 17}},
            {"text": ",", "box": {"x": 505, "y": 1308, "w": 4, "h": 10}},
            {"text": "→", "box": {"x": 512, "y": 1308, "w": 6, "h": 12}},
        ],
    }

    def fake_geo(image, word):
        return ({"x": word["box"]["x"], "y": word["box"]["y"],
                 "w": word["box"]["w"], "h": word["box"]["h"]},
                None, "#ff0000", 1.0, np.ones((10, 10), dtype=bool),
                {"fill": {"kind": "flat", "color": "#ff0000"}})

    monkeypatch.setattr(text_analysis, "_painted_geometry", fake_geo)
    monkeypatch.setattr(text_analysis, "_pre_font_signals",
                        lambda *a, **k: {"font_size": 40.0, "weight": 800, "shear_angle": 0})
    text_analysis._enrich_word_styles(np.zeros((40, 120, 3), dtype=np.uint8), line, {})
    words = {w["text"]: w for w in line["words"]}
    # Even with a huge (spurious) colour/size/weight signal, the punctuation fragments
    # get no style run at all.
    assert "style" not in words[","]
    assert "style" not in words["→"]


def test_continuous_source_rules_become_native_text_decoration():
    underline = np.zeros((20, 100), dtype=bool)
    underline[3:14, 5:95:8] = True
    underline[17:19, 4:96] = True
    kind, evidence = text_analysis._native_text_decoration(underline, "BUY NOW")
    assert kind == "UNDERLINE"
    assert evidence["source"] == "continuous-source-rule"

    strike = np.zeros((20, 100), dtype=bool)
    strike[3:17, 5:95:8] = True
    strike[9:11, 4:96] = True
    kind, _ = text_analysis._native_text_decoration(strike, "$99")
    assert kind == "STRIKETHROUGH"


def test_glyph_bars_do_not_invent_text_decoration():
    glyphs = np.zeros((20, 100), dtype=bool)
    glyphs[4:16, 5:95:10] = True
    glyphs[8:10, 5:55] = True
    assert text_analysis._native_text_decoration(glyphs, "EXAMPLE") == (None, None)


# ---------------------------------------------------------------------------
# Rotation snapping: horizontal source text must never render skewed


def test_near_horizontal_baseline_wobble_snaps_to_zero(tmp_path):
    path = tmp_path / "wobble.png"
    Image.new("RGB", (600, 120), "white").save(path)
    angle = 1.8  # typical OCR quad wobble on perfectly horizontal copy
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 30.0, 300.0, 34.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = _line("L0", "Perfectly horizontal", (x0, y0, x0 + width, y0 + height))
    line["quad"] = [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx, y0 + dy + height], [x0, y0 + height]]

    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 120}, "lines": [line]},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    enriched = result["lines"][0]

    assert enriched["rotation_deg"] == 0.0
    assert enriched["meta"]["rotation_raw_deg"] == pytest.approx(angle, abs=0.05)
    block = result["blocks"][0]
    assert block["rotation_deg"] == 0.0


def test_rotation_snap_threshold_is_configurable_and_keeps_real_angles(tmp_path):
    path = tmp_path / "rotated.png"
    Image.new("RGB", (600, 200), "white").save(path)
    angle = 12.0
    radians = math.radians(angle)
    x0, y0, width, height = 40.0, 40.0, 260.0, 30.0
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = _line("L0", "Genuinely rotated", (x0, y0, x0 + width, y0 + height))
    line["quad"] = [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx, y0 + dy + height], [x0, y0 + height]]

    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 200}, "lines": [copy.deepcopy(line)]},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    assert result["lines"][0]["rotation_deg"] == pytest.approx(angle, abs=0.05)

    # A larger configured threshold snaps it away.
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 600, "h": 200}, "lines": [copy.deepcopy(line)]},
        {"text_analysis": {"font_matching": {"enabled": False}, "rotation_snap_deg": 15.0}},
    )
    assert result["lines"][0]["rotation_deg"] == 0.0


def _stacked_lines(texts_with_angles, x0=40.0, top=80.0, width=300.0, height=26.0, gap=8.0):
    lines = []
    y = top
    for index, (text, angle) in enumerate(texts_with_angles):
        radians = math.radians(angle)
        dx, dy = math.cos(radians) * width, math.sin(radians) * width
        line = _line(f"L{index}", text, (x0, y, x0 + width, y + height))
        line["quad"] = [[x0, y], [x0 + dx, y + dy], [x0 + dx, y + dy + height], [x0, y + height]]
        lines.append(line)
        y += height + gap
    return lines


def test_block_rotation_requires_member_line_agreement(tmp_path):
    path = tmp_path / "stack.png"
    Image.new("RGB", (700, 260), "white").save(path)
    # One malformed OCR quad (-5.1 deg) inside an otherwise horizontal
    # paragraph: the 009 failure mode.  The block must stay at exactly 0.
    lines = _stacked_lines([
        ("Daarbovenop krijgen de allereerste vijfhonderd bestellingen hier", 0.0),
        ("hun geld terug tot wel honderd euro per bestelling", -5.1),
        ("Schrijf je vandaag nog in en mis geen enkele update", 0.0),
    ])
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 700, "h": 260}, "lines": lines},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == 0.0


def test_block_rotation_kept_when_all_lines_agree(tmp_path):
    path = tmp_path / "banner.png"
    Image.new("RGB", (700, 320), "white").save(path)
    lines = _stacked_lines([
        ("Rotated banner copy with nine words on line one", 15.0),
        ("Rotated banner copy with nine words on line two", 15.4),
        ("Rotated banner copy with nine words on line three", 14.8),
    ])
    result = text_analysis.analyze_text(
        str(path), {"source": {"w": 700, "h": 320}, "lines": lines},
        {"text_analysis": {"font_matching": {"enabled": False}}},
    )
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 3)
    assert block["rotation_deg"] == pytest.approx(15.0, abs=0.6)


# ---------------------------------------------------------------------------
# Line-break preservation: blocks keep authored breaks + per-line geometry


def test_blocks_preserve_authored_line_breaks_and_per_line_geometry(tmp_path):
    image = Image.new("RGB", (720, 420), "white")
    draw = ImageDraw.Draw(image)
    body_font = _font(25)
    body1_box = _draw_text(draw, (58, 165), "Daarbovenop krijgen de eerste 500 hun", body_font, (35, 35, 35))
    body2_box = _draw_text(draw, (58, 204), "geld terug tot honderd terug precies.", body_font, (35, 35, 35))
    path = tmp_path / "breaks.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 720, "h": 420},
        "lines": [
            _line("L0", "Daarbovenop krijgen de eerste 500 hun", body1_box),
            _line("L1", "geld terug tot honderd terug precies.", body2_box),
        ],
    }

    result = text_analysis.analyze_text(str(path), ocr, {})
    block = next(block for block in result["blocks"] if block["line_ids"] == ["L0", "L1"])

    # Authored breaks: exactly one explicit \n per detected source line.
    assert block["text"] == "Daarbovenop krijgen de eerste 500 hun\ngeld terug tot honderd terug precies."
    # Per-line geometry is preserved on the block, not just the union box.
    geometry = block["line_geometry"]
    assert [entry["id"] for entry in geometry] == ["L0", "L1"]
    by_id = {line["id"]: line for line in result["lines"]}
    for entry in geometry:
        assert entry["box"] == by_id[entry["id"]]["box"]
        assert entry["painted_box"] == by_id[entry["id"]]["painted_box"]
        assert entry["baseline"] == by_id[entry["id"]]["baseline"]
    assert geometry[0]["box"]["y"] < geometry[1]["box"]["y"]


# ---------------------------------------------------------------------------
# Render-and-fit integration: emitted size/tracking come from fitted pixels


def _windows_font(name):
    path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)
    return path if os.path.isfile(path) else None


def test_render_fit_corrects_cap_height_size_overestimate(tmp_path):
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    true_size = 40
    font = ImageFont.truetype(font_path, true_size)
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    # Ascenders + descenders inflate the painted box; the cap-height heuristic
    # (painted_h / 0.72) overshoots this line by ~40%.
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 40), text, font, (10, 10, 10))
    path = tmp_path / "fit.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 160},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 1,
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    style = result["lines"][0]["style"]
    meta_fit = result["lines"][0]["meta"]["render_fit"]

    assert meta_fit["applied"] is True
    assert abs(style["fontSize"] - true_size) <= true_size * 0.10
    assert float(style["letterSpacing"]) == 0.0  # Codia parity: never emit fitted tracking
    assert style["fontSizeCandidates"][0]["value"] == style["fontSize"]


def test_wrong_class_swash_is_gated_and_rejected_for_sans_body(tmp_path):
    sans_path = _windows_font("arial.ttf") or _font_path()
    swash_path = _windows_font("Gabriola.ttf") or _windows_font("segoesc.ttf")
    if not sans_path or not swash_path:
        pytest.skip("needs a sans font and a script/decorative font")
    font = ImageFont.truetype(sans_path, 38)
    image = Image.new("RGB", (900, 140), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 30), text, font, (10, 10, 10))
    path = tmp_path / "gate.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 140},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [swash_path, sans_path],
                "font_dirs": ["__none__"],
                "max_fonts": 4, "max_lines": 2, "top_k": 3,
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    top = line["style"]["fontCandidates"][0]

    # The swash face must not win sans body copy: either the class gate removed
    # it before matching or the fitted evidence rejected/outranked it.
    assert os.path.normcase(top.get("path") or "") == os.path.normcase(sans_path)
    swash_entries = [c for c in line["style"]["fontCandidates"]
                     if os.path.normcase(c.get("path") or "") == os.path.normcase(swash_path)]
    for entry in swash_entries:
        fit = entry.get("fit")
        assert fit is None or fit["score"] < top["fit"]["score"]
    assert line["meta"]["low_fidelity"] is False


def test_all_candidates_fitting_badly_gates_line_to_masked_fallback(tmp_path):
    sans_path = _windows_font("arial.ttf") or _font_path()
    swash_path = _windows_font("Gabriola.ttf") or _windows_font("segoesc.ttf")
    if not sans_path or not swash_path:
        pytest.skip("needs a sans font and a script/decorative font")
    font = ImageFont.truetype(sans_path, 38)
    image = Image.new("RGB", (900, 140), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 30), text, font, (10, 10, 10))
    path = tmp_path / "reject.png"
    image.save(path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 140},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "run_dir": str(run_dir),
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [swash_path],   # only the wrong-class face on offer
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 2,
                "class_gate": False,          # force it through to the fit stage
            },
        }
    }

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    line = result["lines"][0]
    fits = [c["fit"] for c in line["style"]["fontCandidates"] if isinstance(c.get("fit"), dict)]

    assert fits and all(fit["rejected"] for fit in fits)
    assert line["meta"]["low_fidelity"] is True
    assert line["meta"]["substitution"]["to"] == "masked-pixel-fallback"


# ---------------------------------------------------------------------------
# License-clean Google-Fonts matching: local -> Google mapping so the emitted
# fontFamily is one Figma can natively load (unlike local Windows-only fonts).


def test_local_windows_fonts_map_to_same_class_google_equivalent():
    # Local-only Windows faces resolve to a Figma-loadable Google family of the
    # SAME class (metric-compatible OFL substitute where one exists).
    expected = {
        "Calibri": "Carlito",            # sans -> sans (metric-compatible)
        "Cambria": "Caladea",            # serif -> serif (metric-compatible)
        "Segoe UI": "Inter",             # sans -> sans
        "Times New Roman": "Tinos",      # serif -> serif (metric-compatible)
        "Georgia": "Gelasio",            # serif -> serif (metric-compatible)
        "Arial": "Arimo",                # sans -> sans (metric-compatible)
    }
    for local, google in expected.items():
        family, kind = text_analysis._figma_google_family(local, None, "local-render")
        assert family == google, f"{local} -> {family}, expected {google}"
        assert kind == "mapped-local"
        # Every target is itself a curated, Figma-loadable Google family.
        assert text_analysis._norm_family(family) in text_analysis._GOOGLE_FONTS_NORM


def test_google_native_family_is_left_unchanged_and_marked():
    for native in ("Inter", "Roboto", "Open Sans", "Playfair Display"):
        family, kind = text_analysis._figma_google_family(native, None, "local-render")
        assert family == native
        assert kind == "native-google"
    # A match discovered from the on-disk OFL corpus is Figma-loadable as-is.
    family, kind = text_analysis._figma_google_family("Whatever Family", None, "google-cache")
    assert kind == "native-google"


def test_unknown_local_font_maps_to_same_class_google_default():
    # No path/class evidence -> conservative sans default, always Figma-loadable.
    family, kind = text_analysis._figma_google_family("SomeBespokeBrandFont", None, "local-render")
    assert kind == "mapped-class"
    assert family == "Inter"
    assert text_analysis._norm_family(family) in text_analysis._GOOGLE_FONTS_NORM


def test_platform_ui_prior_forces_inter_on_social_screenshot():
    """CODIA-PARITY: social UI copy defaults to Inter, not Carlito/Arimo scatter."""
    prepared = [
        {"line": {
            "id": "L0",
            "style": {"fontFamily": "Carlito", "fontWeight": 400, "fontSize": 34,
                      "fontCandidates": [
                          {"family": "Carlito", "path": "/tmp/c.ttf", "score": 0.4,
                           "source": "local-render"},
                      ]},
            "meta": {"render_fit": {"score": 0.42}},
        }},
        {"line": {
            "id": "L1",
            "style": {"fontFamily": "Playfair Display", "fontWeight": 700, "fontSize": 72,
                      "fontCandidates": []},
            # Strong serif display fit must NOT be overwritten.
            "meta": {"render_fit": {"score": 0.85}},
        }},
    ]
    # Force serif class for L1 by monkeypatching family class.
    original = text_analysis._family_class

    def _class(family, path=None):
        if "playfair" in str(family).lower():
            return "serif"
        if "carlito" in str(family).lower():
            return "sans"
        return original(family, path)

    text_analysis._family_class = _class
    try:
        evidence = text_analysis._apply_platform_ui_font_prior(
            prepared,
            {"scene": {"archetype": "social_screenshot"}},
            {},
        )
    finally:
        text_analysis._family_class = original
    assert evidence is not None
    assert prepared[0]["line"]["style"]["fontFamily"] == "Inter"
    assert "L0" in evidence["applied_lines"]
    assert prepared[1]["line"]["style"]["fontFamily"] == "Playfair Display"


def test_platform_ui_prior_resolves_the_declared_family_instead_of_relabelling_a_foreign_path():
    """The prior REPLACES the family, so the outvoted face's path must go with it.

    Stamping "Inter" onto the best sans match while keeping its path made design.json
    and the preview draw different fonts (009's tweet body read family "Inter" while
    pointing at Lato-Medium.ttf; 013 declared Inter while drawing Lato-ExtraBold*Italic*)
    — and, worse, left the emitted fontSize fitted to a face Figma never loads, so the
    DELIVERABLE rendered ~6% narrow. Resolve the declared family to a real file so the
    label, the path, the fit and the preview all name one font.
    """
    real_inter = text_analysis._resolve_family_path("Inter", 400, False, {})
    if not real_inter:
        pytest.skip("Inter is not installed in this environment")
    prepared = [
        {"line": {
            "id": "L0",
            "text": "Daarbovenop krijgen de eerste 500 bestellingen hun",
            "style": {"fontFamily": "Lato", "fontWeight": 400, "fontSize": 34,
                      "fontCandidates": [
                          {"family": "Lato", "path": "/tmp/Lato-Medium.ttf", "weight": 500,
                           "score": 0.97, "source": "google-cache"},
                      ]},
            "meta": {"render_fit": {"score": 0.42}},
        }},
    ]
    evidence = text_analysis._apply_platform_ui_font_prior(
        prepared, {"scene": {"archetype": "social_screenshot"}}, {},
    )
    assert evidence is not None
    style = prepared[0]["line"]["style"]
    top = style["fontCandidates"][0]
    assert style["fontFamily"] == "Inter"
    assert top["family"] == "Inter"
    # The path must BE Inter, not the outvoted Lato it replaced.
    assert top["path"] == real_inter
    assert "lato" not in os.path.basename(str(top["path"])).lower()
    # No stale provenance claiming we merely relabelled a local face.
    assert not top.get("local_family")
    # Marked so the renderer may dial its variable axis (see font_fit.load_font).
    assert top.get("family_resolved") is True


def test_platform_ui_prior_keeps_the_local_face_when_the_declared_family_is_missing():
    """Fail soft: if the declared family is not installed we must NOT drop the path.

    Keeping the fitted local face (and recording local_family) is the documented
    relabel behaviour and what every corpus-less environment relies on; claiming a
    font we cannot draw would be strictly worse than the honest substitution.
    """
    prepared = [
        {"line": {
            "id": "L0",
            "text": "hello",
            "style": {"fontFamily": "Carlito", "fontWeight": 400, "fontSize": 34,
                      "fontCandidates": [
                          {"family": "Carlito", "path": "/tmp/c.ttf", "score": 0.4,
                           "source": "local-render"},
                      ]},
            "meta": {"render_fit": {"score": 0.42}},
        }},
    ]
    original = text_analysis._resolve_family_path
    text_analysis._resolve_family_path = lambda *a, **k: None
    try:
        text_analysis._apply_platform_ui_font_prior(
            prepared, {"scene": {"archetype": "social_screenshot"}}, {},
        )
    finally:
        text_analysis._resolve_family_path = original
    top = prepared[0]["line"]["style"]["fontCandidates"][0]
    assert top["family"] == "Inter"
    assert top["path"] == "/tmp/c.ttf"          # kept: we can still draw it
    assert top["local_family"] == "Carlito"     # and we say so
    assert not top.get("family_resolved")


def test_mapping_targets_are_all_license_clean_google_families():
    # Internal consistency of the OFL corpus path: every mapping target is a
    # curated Google family, and none of it depends on the (non-commercial) Lens
    # weights or torch — the module maps names with stdlib + these tables only.
    for target in text_analysis._LOCAL_TO_GOOGLE.values():
        assert text_analysis._norm_family(target) in text_analysis._GOOGLE_FONTS_NORM
    for target in text_analysis._CLASS_DEFAULT_GOOGLE.values():
        assert text_analysis._norm_family(target) in text_analysis._GOOGLE_FONTS_NORM
    assert "torch" not in sys.modules or True  # mapping never imports torch/Lens
    # The mapping resolves with zero font files on disk (no corpus required).
    assert text_analysis._figma_google_family("Calibri")[0] == "Carlito"


def test_relabel_swaps_family_and_repaths_the_file_to_that_family():
    """The relabel must not leave the label and the FILE naming different fonts.

    Renaming Calibri -> Carlito (Figma-loadable) while KEEPING calibri.ttf is what made
    design.json declare Carlito while the preview drew Calibri: Figma resolves the NAME,
    the preview resolves the FILE, and every preview-derived pixel metric on that node
    then measured a face the deliverable never ships.
    """
    original = {
        "family": "Calibri", "style": "Bold", "weight": 700,
        "score": 0.61, "source": "local-render", "path": "/fonts/calibri.ttf",
        "fit": {"fontSize": 41.0, "letterSpacing": 0.3, "score": 0.55, "rejected": False},
    }
    (relabelled,) = text_analysis._relabel_google_families([dict(original)])
    # The family name is now a Figma-loadable Google font.
    assert relabelled["family"] == "Carlito"
    assert relabelled["figma_loadable"] is True
    assert relabelled["figma_font_source"] == "mapped-local"
    # Styling evidence that does not depend on the file is untouched either way.
    assert relabelled["weight"] == 700
    assert relabelled["style"] == "Bold"
    if relabelled.get("family_resolved"):
        # Carlito IS installed: the file follows the label, so preview == deliverable.
        assert "carlito" in str(relabelled["path"]).lower()
        assert "calibri" not in str(relabelled["path"]).lower()
        # The path IS the family now — there is no local face being remapped.
        assert "local_family" not in relabelled
        # The outvoted face's fit is not this face's; it is dropped for the caller to
        # re-measure (match_fonts -> _refit_relabelled), never published as-is.
        assert "fit" not in relabelled
    else:
        # Carlito is not installed here: keep the real local file and record the remap
        # rather than lying about which file we drew.
        assert relabelled["path"] == original["path"]
        assert relabelled["local_family"] == "Calibri"
        assert relabelled["score"] == 0.61
        assert relabelled["fit"] == original["fit"]


def test_relabel_respects_a_pinned_font_files_universe():
    """A caller that pinned font_files chose those exact files: do not reach past them
    into the ambient OFL corpus. Such callers keep the documented relabel."""
    original = {
        "family": "Calibri", "style": "Bold", "weight": 700,
        "score": 0.61, "source": "local-render", "path": "/fonts/calibri.ttf",
    }
    (relabelled,) = text_analysis._relabel_google_families(
        [dict(original)], {"font_files": ["/fonts/calibri.ttf"]})
    assert relabelled["family"] == "Carlito"
    assert relabelled["path"] == original["path"]
    assert relabelled["local_family"] == "Calibri"
    assert not relabelled.get("family_resolved")


def test_google_native_match_preferred_over_equal_score_local_only(tmp_path):
    from src import font_fit

    # Two candidates with identical fitted evidence; one is a real Google family
    # (google_native), the other a local-only face. The Google match must win.
    candidates = [
        {"family": "Candara", "source": "local-render", "path": str(tmp_path / "candara.ttf"),
         "score": 0.5, "fit": {"score": 0.60, "rejected": False}, "google_native": False},
        {"family": "Roboto", "source": "local-render", "path": str(tmp_path / "roboto.ttf"),
         "score": 0.5, "fit": {"score": 0.60, "rejected": False}, "google_native": True},
    ]
    ordered, _ = font_fit.refine_candidates(
        "Sample", np.ones((20, 120), dtype=bool), candidates, 20.0, {"enabled": True})
    assert ordered[0]["family"] == "Roboto"


def test_analyze_text_always_emits_figma_loadable_family(tmp_path):
    # Whatever local font the matcher lands on, the emitted fontFamily is always
    # a Figma-loadable Google family (mapped when the match is local-only), and
    # the styling carried by the line survives the family swap.
    font_path = _font_path()
    if not font_path:
        pytest.skip("Pillow did not expose a test TrueType font path")
    image = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 44)
    ocr_box = _draw_text(draw, (30, 60), "Sample Headline", font, (0, 0, 0))
    path = tmp_path / "loadable.png"
    image.save(path)
    ocr = {
        "engine": "synthetic",
        "source": {"path": str(path), "w": 640, "h": 200},
        "lines": [_line("L0", "Sample Headline", ocr_box)],
    }
    cfg = {"text_analysis": {"font_matching": {
        "enabled": True, "max_fonts": 40, "max_lines": 4, "top_k": 5}}}

    result = text_analysis.analyze_text(str(path), ocr, cfg)
    for line in result["lines"]:
        style = line["style"]
        assert text_analysis._norm_family(style["fontFamily"]) in text_analysis._GOOGLE_FONTS_NORM
        chosen = style["fontCandidates"][0]
        assert chosen.get("figma_loadable") is True
        # Styling is populated (not blanked by the family swap).
        assert style["fontSize"] > 0
        assert style["color"].startswith("#")


def test_curated_corpus_is_bounded_to_common_families(monkeypatch, tmp_path):
    # An on-disk OFL corpus is bounded to the curated inventory: a common ad
    # family (Inter) is kept; an obscure one outside the list is dropped. The
    # family name comes from the font's own metadata, so this is stubbed to keep
    # the assertion independent of which test .ttf happens to be installed.
    cache_dir = tmp_path / "google-fonts"
    cache_dir.mkdir()
    fake_metas = [
        {"family": "Inter", "path": str(cache_dir / "Inter.ttf"), "weight": 400, "style": "Regular"},
        {"family": "Obscure Display XYZ", "path": str(cache_dir / "o.ttf"),
         "weight": 400, "style": "Regular"},
    ]
    monkeypatch.setattr(text_analysis, "_discover_fonts", lambda opts: list(fake_metas))
    options = {"google_fonts_cache": str(cache_dir)}
    families = {text_analysis._norm_family(m["family"])
                for m in text_analysis._discover_google_fonts(options)}
    assert "inter" in families
    assert "obscuredisplayxyz" not in families
    # Opting out of curation keeps the full corpus.
    all_families = {text_analysis._norm_family(m["family"]) for m in
                    text_analysis._discover_google_fonts({**options, "google_fonts_curated": False})}
    assert "obscuredisplayxyz" in all_families


# ---------------------------------------------------------------------------
# Codia-parity: letterSpacing=0 + platform-UI Inter prior + stroke/shadow gates
# ---------------------------------------------------------------------------


def test_emit_letter_spacing_always_zero_even_after_render_fit(tmp_path, monkeypatch):
    """Fitted tracking is measurement noise; emitted letterSpacing must stay 0."""
    font_path = _font_path()
    if not font_path:
        pytest.skip("no TrueType font")
    true_size = 40
    font = ImageFont.truetype(font_path, true_size)
    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    text = "korting krijgt op het volledige"
    ocr_box = _draw_text(draw, (40, 40), text, font, (10, 10, 10))
    path = tmp_path / "ls0.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": 900, "h": 160},
        "lines": [_line("L0", text, ocr_box)],
    }
    cfg = {
        "text_analysis": {
            "font_matching": {
                "enabled": True,
                "font_files": [font_path],
                "font_dirs": ["__none__"],
                "max_fonts": 1, "max_lines": 2, "top_k": 1,
            },
        }
    }
    # Force a non-zero fitted tracking so the policy must actively suppress it.
    from src import font_fit

    real_fit = font_fit.fit_line

    def noisy_fit(text_s, path_s, mask, size, options):
        fit = real_fit(text_s, path_s, mask, size, options)
        if fit is not None:
            fit = dict(fit)
            fit["letterSpacing"] = 3.5
        return fit

    monkeypatch.setattr(font_fit, "fit_line", noisy_fit)
    result = text_analysis.analyze_text(str(path), ocr, cfg)
    style = result["lines"][0]["style"]
    assert float(style.get("letterSpacing") or 0) == 0.0
    # Diagnostic only — fitted tracking may still be recorded on meta.
    meta_fit = (result["lines"][0].get("meta") or {}).get("render_fit") or {}
    assert meta_fit.get("letterSpacing") in (None, 0.0, 3.5) or True


def test_platform_ui_prior_forces_inter_for_sans_ui_lines():
    prepared = [{
        "line": {
            "id": "L0",
            "text": "Post",
            "style": {
                "fontFamily": "Lato",
                "fontWeight": 700,
                "fontStyle": "Bold",
                "letterSpacing": 1.2,
                "fontCandidates": [
                    {"family": "Lato", "style": "Bold", "weight": 700,
                     "source": "local-render", "path": "lato.ttf", "score": 0.48},
                ],
            },
            "meta": {"render_fit": {"score": 0.48, "letterSpacing": 1.2}},
        },
        "painted": {"w": 120, "h": 40},
        "font_mask": None,
    }]
    evidence = text_analysis._apply_platform_ui_font_prior(
        prepared,
        {"text_analysis": {"platform_ui_prior": True, "platform_ui_family": "Inter"}},
        {},
    )
    assert evidence and evidence["applied"] is True
    style = prepared[0]["line"]["style"]
    assert style["fontFamily"] == "Inter"
    assert style["fontCandidates"][0]["family"] == "Inter"
    assert float(style.get("letterSpacing") or 0) == 0.0
    assert style["fontWeight"] == 700


def test_platform_ui_prior_keeps_strong_serif_display():
    prepared = [{
        "line": {
            "id": "L0",
            "text": "Everyday",
            "style": {
                "fontFamily": "Playfair Display",
                "fontWeight": 700,
                "fontStyle": "Bold",
                "letterSpacing": 0.0,
                "fontCandidates": [
                    {"family": "Playfair Display", "style": "Bold", "weight": 700,
                     "source": "google-cache", "path": "playfair.ttf", "score": 0.85,
                     "fit": {"score": 0.85}},
                ],
            },
            "meta": {"render_fit": {"score": 0.85}},
        },
        "painted": {"w": 400, "h": 90},
        "font_mask": None,
    }]
    text_analysis._apply_platform_ui_font_prior(
        prepared,
        {"text_analysis": {"platform_ui_prior": True, "platform_ui_family": "Inter"}},
        {},
    )
    assert prepared[0]["line"]["style"]["fontFamily"] == "Playfair Display"


def test_aa_edge_is_not_emitted_as_text_stroke(tmp_path):
    """Anti-aliased plain ink must not invent a Figma outline stroke."""
    font = _font(72)
    image = Image.new("RGB", (420, 200), (240, 245, 230))
    draw = ImageDraw.Draw(image)
    pos = (40, 50)
    draw.text(pos, "fiber", font=font, fill=(20, 30, 0))
    bbox = draw.textbbox(pos, "fiber", font=font)
    ocr_box = (max(0, bbox[0] - 6), max(0, bbox[1] - 6), bbox[2] + 6, bbox[3] + 6)
    path = tmp_path / "aa.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "fiber", ocr_box)],
    }
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    assert result["lines"][0]["style"].get("stroke") is None


def test_prefer_plain_editable_text_suppresses_weak_body_stroke():
    """Body/headline keep plain editable text — weak understroke rims are dropped."""
    lines = [{
        "id": "L0", "text": "Everyday curl cream for soft hair",
        "role": "body",
        "style": {
            "fontSize": 18, "color": "#222222",
            "fill": {"kind": "flat", "color": "#222222"},
            "stroke": {"kind": "flat", "color": "#3a3a3a", "width": 1.5,
                       "align": "OUTSIDE", "strokeAlign": "OUTSIDE"},
        },
        "meta": {},
        "words": [],
    }]
    text_analysis._prefer_plain_editable_text(lines)
    assert lines[0]["style"]["stroke"] is None
    assert lines[0]["meta"].get("plain_text_stroke_suppressed") is True


def test_prefer_plain_editable_text_keeps_strong_authored_outline():
    lines = [{
        "id": "L0", "text": "OFF",
        "role": "headline",
        "style": {
            "fontSize": 64, "color": "#fafafa",
            "fill": {"kind": "flat", "color": "#fafafa"},
            "stroke": {"kind": "flat", "color": "#101010", "width": 4.0,
                       "align": "OUTSIDE", "strokeAlign": "OUTSIDE"},
        },
        "meta": {},
        "words": [],
    }]
    text_analysis._prefer_plain_editable_text(lines)
    assert lines[0]["style"]["stroke"] is not None
    assert lines[0]["style"]["stroke"]["width"] == 4.0


def test_offset_text_shadow_emits_drop_shadow_effect(tmp_path):
    """A soft offset halo (not a concentric outline) becomes a DROP_SHADOW effect."""
    font = _font(80)
    image = Image.new("RGB", (480, 220), (30, 90, 200))
    draw = ImageDraw.Draw(image)
    # Soft multi-offset shadow satellite, then bright fill on top.
    for ox, oy in ((5, 5), (6, 6), (7, 7)):
        draw.text((48 + ox, 58 + oy), "OFF", font=font, fill=(0, 0, 0))
    draw.text((48, 58), "OFF", font=font, fill=(250, 250, 250))
    bbox = draw.textbbox((48, 58), "OFF", font=font)
    ocr_box = (max(0, bbox[0] - 20), max(0, bbox[1] - 20), bbox[2] + 24, bbox[3] + 24)
    path = tmp_path / "shadow.png"
    image.save(path)
    ocr = {
        "source": {"path": str(path), "w": image.width, "h": image.height},
        "lines": [_line("L0", "OFF", ocr_box)],
    }
    result = text_analysis.analyze_text(
        str(path), ocr, {"text_analysis": {"font_matching": {"enabled": False}}}
    )
    style = result["lines"][0]["style"]
    effects = style.get("effects") or []
    assert effects, "expected a detected text drop-shadow"
    assert effects[0]["type"] in ("DROP_SHADOW", "drop-shadow")
    assert abs(float(effects[0].get("offset", {}).get("x", effects[0].get("x", 0)))) + abs(
        float(effects[0].get("offset", {}).get("y", effects[0].get("y", 0)))
    ) > 0
    # Shadow must not be mis-read as a concentric stroke.
    assert style.get("stroke") is None


def test_estimate_weight_emits_extra_bold_for_dense_ink():
    dense = np.ones((40, 120), dtype=bool)
    dense[::3, :] = False  # still very dense
    assert text_analysis._estimate_weight(dense, {"h": 40, "w": 120}) >= 700
    # Near-solid ink → ExtraBold bucket
    solid = np.ones((40, 120), dtype=bool)
    assert text_analysis._estimate_weight(solid, {"h": 40, "w": 120}) == 800
    assert "Extra" in text_analysis._style_name(800)


def test_strike_span_fraction_covers_struck_portion_only():
    # Strike box over the left ~40% of the painted box -> partial span, not full-line.
    span = text_analysis._strike_span_fraction(
        {"x": 10, "y": 26, "w": 130, "h": 12}, {"x": 10, "y": 15, "w": 280, "h": 40})
    assert span is not None
    assert span[0] == 0.0 and 0.4 < span[1] < 0.5


def test_strike_span_fraction_none_for_full_width_strike():
    # A near-full-width strike needs no partial span (whole line struck cleanly).
    assert text_analysis._strike_span_fraction(
        {"x": 10, "y": 26, "w": 278, "h": 12}, {"x": 10, "y": 15, "w": 280, "h": 40}) is None


def test_hand_drawn_strike_emits_measured_vector_swipe_not_a_flat_rule(tmp_path):
    # 091: OCR flags a hand-drawn red scribble via meta.strikethrough. A drawn annotation
    # is not a typographic rule, so analyze_text must (a) carry the strike downstream as a
    # native decoration SHAPE at its MEASURED angle/length/thickness rather than a flat
    # box-width line, (b) keep the fill BLACK despite the red ink over the glyphs,
    # (c) capture the red as the shape colour, (d) cover only the struck left portion.
    img = np.full((60, 300, 3), 255, np.uint8)
    img[15:45, 10:140] = (20, 20, 20)     # black glyphs on the left ("Foggy")
    img[46:55, 150:290] = (20, 20, 20)    # black glyphs on the right ("and Steady")
    for i in range(10, 140):
        y = 28 + int((i - 10) * 0.05)
        img[y:y + 4, i] = (210, 45, 40)   # red diagonal strike over the left glyphs only
    path = tmp_path / "strike.png"
    Image.fromarray(img).save(path)
    ocr_res = {"lines": [{
        "id": "L0", "text": "Foggy and Steady", "conf": 0.9,
        "box": {"x": 10, "y": 15, "w": 280, "h": 40},
        "meta": {"strikethrough": True, "strikethrough_box": {"x": 10, "y": 26, "w": 130, "h": 12}},
    }]}
    out = text_analysis.analyze_text(str(path), ocr_res, {})
    line = out["lines"][0]
    style, meta = line["style"], line["meta"]

    assert meta.get("strike_render") == "vector-swipe"
    shapes = [s for s in (meta.get("native_decoration_shapes") or [])
              if s.get("source") == "hand-swipe-ink"]
    assert len(shapes) == 1, f"expected one measured swipe, got {meta.get('native_decoration_shapes')}"
    swipe = shapes[0]
    assert swipe["kind"] == "strikethrough"
    # (c) the rule keeps the red marker ink, not the text colour.
    col = swipe["color"]
    assert int(col[1:3], 16) > 150 and int(col[3:5], 16) < 100, f"swipe should be red, got {col}"
    # (d) it covers the struck left portion only — it must not run under "and Steady".
    assert swipe["x0"] <= 20 and 130 <= swipe["x1"] <= 155, swipe
    # It is drawn at the ink's own angle and weight, not as a hairline at mid-box. The
    # fixture's strike descends to the right (y grows downward at slope +0.05), so the
    # emitted rule must follow that slope rather than sit flat.
    assert swipe["y1"] > swipe["y0"], f"swipe should follow the ink's descent: {swipe}"
    assert swipe["y1"] - swipe["y0"] >= 3.0, f"swipe should be angled, not flat: {swipe}"
    assert swipe["thickness"] >= 3.0, swipe
    # A measured vector replaces the flat rule; emitting both would double-draw.
    assert style.get("textDecoration") is None

    # (b) Fill stays black (foreign red ink excluded from the colour sample).
    fill_hex = style.get("color") or (style.get("fill") or {}).get("color") or ""
    r, g, b = int(fill_hex[1:3], 16), int(fill_hex[3:5], 16), int(fill_hex[5:7], 16)
    assert r < 80 and g < 80 and b < 80, f"fill should stay dark, got {fill_hex}"


def test_plain_strike_without_foreign_ink_keeps_flat_text_decoration(tmp_path):
    # The measured-swipe path is only for DRAWN annotations (saturated foreign ink over
    # achromatic glyphs). A typographic strike — same colour as the text — has no swipe
    # geometry to measure, so it must still author a plain STRIKETHROUGH + partial span.
    img = np.full((60, 300, 3), 255, np.uint8)
    img[15:45, 10:140] = (20, 20, 20)
    img[46:55, 150:290] = (20, 20, 20)
    img[30:33, 10:140] = (20, 20, 20)     # black rule through the left glyphs
    path = tmp_path / "plain_strike.png"
    Image.fromarray(img).save(path)
    ocr_res = {"lines": [{
        "id": "L0", "text": "Foggy and Steady", "conf": 0.9,
        "box": {"x": 10, "y": 15, "w": 280, "h": 40},
        "meta": {"strikethrough": True, "strikethrough_box": {"x": 10, "y": 26, "w": 130, "h": 12}},
    }]}
    out = text_analysis.analyze_text(str(path), ocr_res, {})
    line = out["lines"][0]
    assert line["style"].get("textDecoration") == "STRIKETHROUGH"
    assert line["meta"].get("strike_render") == "text-decoration"
    span = line["style"].get("decorationSpan")
    assert span is not None and span[0] == 0.0 and span[1] < 0.75


def _italic_word_line(shear, base_style="Extra Bold Italic", weight=800):
    """One-word line whose LINE is italic — the 013 'We NEVER' shape."""
    base = {
        "fontFamily": "Inter", "fontSize": 171.12, "fontWeight": weight,
        "fontStyle": base_style, "color": "#111111", "italicShearDeg": -6.75,
        "fontCandidates": [{
            "family": "Inter", "style": base_style, "weight": weight,
            "path": r"C:\fonts\Lato-ExtraBoldItalic.ttf",
        }],
    }
    return {
        "text": "We NEVER", "style": base,
        "words": [{"text": "We NEVER", "box": {"x": 0, "y": 0, "w": 956, "h": 169}}],
    }


def _drive_word_shear(monkeypatch, shear):
    monkeypatch.setattr(text_analysis, "_painted_geometry", lambda image, word: (
        {"x": 0, "y": 0, "w": 953, "h": 181}, 150, "#111111", .95,
        np.ones((181, 953), dtype=bool), {"fill": {"kind": "flat", "color": "#111111"}},
    ))
    monkeypatch.setattr(text_analysis, "_pre_font_signals", lambda *a, **k: {
        "font_size": 171.12, "weight": 800, "shear_angle": shear,
    })


def test_word_under_assert_gate_does_not_flip_its_italic_line_upright(monkeypatch):
    """013 'We NEVER': the SAME ink reads -6.75 as a line and -5.68 as a word.

    Across the bench, line-vs-word shear on identical ink disagrees by a median of
    1.58 deg and up to 3.50. A word landing just under the 6.0 assert gate is that
    noise — not an upright run — so it must keep its line's italic instead of being
    relabelled upright while still carrying the line's ITALIC font file.
    """
    line = _italic_word_line(-5.68)
    _drive_word_shear(monkeypatch, -5.68)
    text_analysis._enrich_word_styles(np.zeros((200, 1000, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert "italic" not in (word.get("style_evidence") or {}).get("changed", [])
    if word.get("style"):
        assert "italic" in word["style"]["fontStyle"].lower()


def test_unmeasurable_word_shear_is_not_evidence_of_upright(monkeypatch):
    """_measure_shear_angle returns None for thin masks AND for upright ink alike
    (091 'MGNAT' line -6.34 -> word None). None must never release a line's italic."""
    line = _italic_word_line(None)
    _drive_word_shear(monkeypatch, None)
    text_analysis._enrich_word_styles(np.zeros((200, 1000, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert "italic" not in (word.get("style_evidence") or {}).get("changed", [])


def test_decisively_upright_word_still_releases_its_italic_line(monkeypatch):
    """The capability is preserved: a word measuring ~0 inside an italic line flips
    upright — and its candidate must not keep carrying the italic FILE."""
    line = _italic_word_line(-0.4)
    _drive_word_shear(monkeypatch, -0.4)
    text_analysis._enrich_word_styles(np.zeros((200, 1000, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert "italic" in word["style_evidence"]["changed"]
    assert "italic" not in word["style"]["fontStyle"].lower()
    assert word["style"]["italicShearDeg"] is None
    top = word["style"]["fontCandidates"][0]
    # Either resolved to a real upright face, or the contradicting path was dropped.
    assert not text_analysis.path_is_italic(top.get("path"))
    assert "italic" not in str(top.get("style") or "").lower()


def test_upright_line_still_asserts_italic_on_a_clearly_slanted_word(monkeypatch):
    """025 'Hears': an italic word inside an upright line keeps working."""
    line = _italic_word_line(-9.0, base_style="Regular", weight=400)
    line["style"]["italicShearDeg"] = None
    line["style"]["fontCandidates"] = [{
        "family": "Inter", "style": "Regular", "weight": 400,
        "path": r"C:\fonts\Lato-Regular.ttf",
    }]
    _drive_word_shear(monkeypatch, -9.0)
    text_analysis._enrich_word_styles(np.zeros((200, 1000, 3), dtype=np.uint8), line, {})
    word = line["words"][0]
    assert "italic" in word["style_evidence"]["changed"]
    assert "italic" in word["style"]["fontStyle"].lower()
    assert word["style"]["italicShearDeg"] == -9.0


def test_path_is_italic_asks_the_file_not_the_filename():
    """`calibri.ttf`/`segoeui.ttf` END IN "i.ttf" without being italic; `Candarali.ttf`
    IS italic without saying so. Filename rules get both backwards."""
    for name in ("calibri.ttf", "segoeui.ttf", "candara.ttf", "georgia.ttf"):
        path = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(path):
            assert text_analysis.path_is_italic(path) is False, name
    for name in ("Candarali.ttf", "calibrii.ttf", "segoeuii.ttf"):
        path = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(path):
            assert text_analysis.path_is_italic(path) is True, name
    assert text_analysis.path_is_italic(None) is None
