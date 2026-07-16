"""In-image vs overlay text — the correctness contract.

"For image ads with overlaid text, it must accurately detect whether the text is
part of the image or separate."  The two poles are pinned by fixture:
021's handwritten sticky notes stay BAKED, 009's tweet copy stays EDITABLE.
Measured per-line accuracy over 021/009/002/107 rose 29.8% -> 93.4% when these
weights landed; these tests guard the reasoning that got there.
"""

import numpy as np
import pytest

from src import scene_intent

CANVAS = {"w": 1080, "h": 1080}


def _line(id_, x, y, w, h, text="", quad=None):
    return {"id": id_, "text": text, "box": {"x": x, "y": y, "w": w, "h": h},
            "quad": quad or [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            "meta": {}}


def _rotated_quad(x, y, w, h, deg):
    import math
    r = math.radians(deg)
    cx, cy = x + w / 2, y + h / 2
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [[cx + px * math.cos(r) - py * math.sin(r),
             cy + px * math.sin(r) + py * math.cos(r)] for px, py in pts]


def _element(id_, role, x, y, w, h):
    return {"id": id_, "role": role, "box": {"x": x, "y": y, "w": w, "h": h}, "meta": {}}


# ── rotation ────────────────────────────────────────────────────────────────────

def test_rotation_is_measured_against_the_canvas_axis():
    assert abs(scene_intent._quad_rotation_deg([[0, 0], [100, 0], [100, 10], [0, 10]])) < 0.01
    rot = scene_intent._quad_rotation_deg(_rotated_quad(0, 0, 100, 20, -25))
    assert rot is not None and abs(rot - (-25)) < 1.0


def test_rotation_folds_a_right_to_left_run_onto_the_same_baseline():
    """A quad wound the other way is the same baseline, not a 180° rotation."""
    rot = scene_intent._quad_rotation_deg([[100, 0], [0, 0], [0, 10], [100, 10]])
    assert rot is not None and abs(rot) < 0.01


def test_rotation_of_a_degenerate_quad_is_none():
    assert scene_intent._quad_rotation_deg(None) is None
    assert scene_intent._quad_rotation_deg([[0, 0]]) is None
    assert scene_intent._quad_rotation_deg("nonsense") is None


def test_021_rotated_sticky_note_reads_baked():
    """021: 'BUY TWO' on a physical note sits -24° off axis inside a photo."""
    line = _line("l1", 60, 300, 120, 40, "BUY TWO", quad=_rotated_quad(60, 300, 120, 40, -24))
    photo = _element("E000", "photo", 0, 0, 338, 600)
    out = scene_intent.classify_text_placement(line, CANVAS, owner=photo)
    assert out["placement"] == "baked"
    assert any("rotated" in r for r in out["reasons"])


# ── the containment prior must not be a verdict ─────────────────────────────────

def test_009_tweet_copy_inside_a_screenshot_stays_editable():
    """The whole point: 009's copy IS inside a raster, and IS still editable.
    A screenshot raster carries role 'screenshot', never 'photo'."""
    line = _line("l1", 71, 315, 683, 46, "LAATSTE SITE WIDE SALE VAN 2026")
    shot = _element("E000", "screenshot", 3, 5, 1077, 1075)
    peers = [_line("l2", 71, 409, 883, 46), _line("l3", 71, 454, 772, 47)]
    out = scene_intent.classify_text_placement(line, CANVAS, owner=shot, peers=peers)
    assert out["placement"] == "overlay", out["reasons"]


def test_text_inside_a_product_raster_reads_baked():
    """002: 'VANILLE SMAAK' is printed on the pouch, not set on the card."""
    line = _line("l1", 150, 1240, 160, 24, "VANILLE SMAAK")
    pouch = _element("E005", "product", 61, 910, 955, 765)
    out = scene_intent.classify_text_placement(line, CANVAS, owner=pouch)
    assert out["placement"] == "baked"
    assert any("product" in r for r in out["reasons"])


def test_containment_alone_never_outvotes_nothing():
    """A line with no owner and no image evidence must not be guessed."""
    out = scene_intent.classify_text_placement(_line("l1", 0, 0, 10, 10), CANVAS)
    assert out["placement"] in {"overlay", "unknown"}


# ── grid alignment is a designer signal only OUT on the canvas ──────────────────

def test_grid_alignment_is_ignored_inside_a_product_raster():
    """002's nutrition table shares a left edge down every row; that is the
    product's own artwork, not the designer's copy grid."""
    rows = [_line(f"r{i}", 150, 1300 + i * 30, 200, 20, f"row {i}") for i in range(4)]
    pouch = _element("E005", "product", 61, 910, 955, 765)
    out = scene_intent.classify_text_placement(rows[0], CANVAS, owner=pouch, peers=rows[1:])
    assert out["placement"] == "baked"
    assert not any("copy grid" in r for r in out["reasons"])


def test_grid_alignment_counts_on_the_open_canvas():
    rows = [_line(f"r{i}", 100, 100 + i * 60, 400, 40, f"row {i}") for i in range(4)]
    out = scene_intent.classify_text_placement(rows[0], CANVAS, peers=rows[1:])
    assert any("copy grid" in r for r in out["reasons"])


def test_grid_alignment_needs_enough_peers():
    box = {"x": 100, "y": 100, "w": 200, "h": 20}
    opts = scene_intent.placement_options({})
    assert not scene_intent._grid_aligned(box, [{"box": {"x": 500, "y": 0, "w": 10, "h": 10}}], opts)
    peers = [{"box": {"x": 100, "y": 200, "w": 200, "h": 20}},
             {"box": {"x": 100, "y": 300, "w": 200, "h": 20}}]
    assert scene_intent._grid_aligned(box, peers, opts)


# ── ink statistics ──────────────────────────────────────────────────────────────

def _crisp_text_crop():
    """Two-level ink: background or glyph, ~no intermediate pixels."""
    arr = np.full((40, 200, 3), 250, dtype=np.uint8)
    arr[10:30, 20:180] = 12
    return arr


def _blurred_text_crop():
    """A photographed glyph ramps through every intermediate value."""
    arr = _crisp_text_crop().astype("float64")
    for _ in range(6):  # cheap repeated box blur
        arr = (arr + np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
               + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)) / 5.0
    return arr.astype(np.uint8)


def test_sharpness_separates_rendered_ink_from_photographed_ink():
    box = {"x": 0, "y": 0, "w": 200, "h": 40}
    crisp = scene_intent.ink_statistics(_crisp_text_crop(), box)
    blurred = scene_intent.ink_statistics(_blurred_text_crop(), box)
    assert crisp and blurred
    assert crisp["sharpness"] > blurred["sharpness"], (crisp, blurred)
    opts = scene_intent.placement_options({})
    assert crisp["sharpness"] >= opts["sharpness_overlay_min"]


def test_ink_statistics_returns_none_on_a_blank_crop():
    flat = np.full((40, 200, 3), 250, dtype=np.uint8)
    assert scene_intent.ink_statistics(flat, {"x": 0, "y": 0, "w": 200, "h": 40}) is None


def test_ink_statistics_returns_none_on_a_tiny_crop():
    assert scene_intent.ink_statistics(_crisp_text_crop(), {"x": 0, "y": 0, "w": 2, "h": 2}) is None


def test_ink_statistics_never_raises_on_junk():
    assert scene_intent.ink_statistics(None, {"x": 0, "y": 0, "w": 10, "h": 10}) is None
    assert scene_intent.ink_statistics("nope", {"x": 0, "y": 0, "w": 10, "h": 10}) is None


def test_crisp_ink_alone_does_not_prove_overlay():
    """A label printed onto a mockup pouch is crisp too. Weighting crisp ink as
    strong overlay evidence scored 002 at 6.9%."""
    line = _line("l1", 0, 0, 200, 40, "ingredienten")
    pouch = _element("E005", "product", 0, 0, 400, 400)
    out = scene_intent.classify_text_placement(line, CANVAS, owner=pouch,
                                               image=_crisp_text_crop())
    assert out["placement"] == "baked", out["reasons"]


# ── the VLM corroborates, never vetoes ──────────────────────────────────────────

def test_vlm_agreeing_raises_confidence():
    line = _line("l1", 60, 300, 120, 40, "BUY TWO", quad=_rotated_quad(60, 300, 120, 40, -24))
    photo = _element("E000", "photo", 0, 0, 338, 600)
    without = scene_intent.classify_text_placement(line, CANVAS, owner=photo)
    with_vlm = scene_intent.classify_text_placement(line, CANVAS, owner=photo,
                                                    vlm_placement="printed")
    assert with_vlm["placement"] == without["placement"] == "baked"
    assert with_vlm["confidence"] >= without["confidence"]


def test_a_dissenting_vlm_cannot_flip_strong_physical_evidence():
    """A slow local VLM must never be load-bearing for a correctness contract."""
    line = _line("l1", 60, 300, 120, 40, "BUY TWO", quad=_rotated_quad(60, 300, 120, 40, -24))
    photo = _element("E000", "photo", 0, 0, 338, 600)
    out = scene_intent.classify_text_placement(line, CANVAS, owner=photo,
                                               vlm_placement="overlay_copy")
    assert out["placement"] == "baked", out["reasons"]


def test_unknown_vlm_verdict_is_ignored():
    line = _line("l1", 0, 0, 100, 20, "hi")
    out = scene_intent.classify_text_placement(line, CANVAS, vlm_placement="banana")
    assert not any("vlm" in r for r in out["reasons"])


# ── report shape / config ───────────────────────────────────────────────────────

def test_classify_lines_reports_every_line_and_counts():
    lines = [_line("a", 0, 0, 100, 20, "one"), _line("b", 0, 40, 100, 20, "two")]
    report = scene_intent.classify_lines(lines, CANVAS, elements=[])
    assert set(report["lines"]) == {"a", "b"}
    assert sum(report["counts"].values()) == 2


def test_classify_lines_can_be_disabled():
    report = scene_intent.classify_lines([_line("a", 0, 0, 10, 10)], CANVAS,
                                         cfg={"scene_intent": {"placement": {"enabled": False}}})
    assert report["enabled"] is False and report["lines"] == {}


def test_placement_options_are_overridable():
    opts = scene_intent.placement_options({"scene_intent": {"placement": {"rotation_baked_deg": 15.0}}})
    assert opts["rotation_baked_deg"] == 15.0
    assert opts["axis_tolerance_deg"] == scene_intent.PLACEMENT_DEFAULTS["axis_tolerance_deg"]


def test_placement_owner_picks_the_smallest_container():
    line = _line("l1", 100, 100, 50, 20)
    big = _element("big", "photo", 0, 0, 1000, 1000)
    small = _element("small", "product", 90, 90, 200, 200)
    opts = scene_intent.placement_options({})
    assert scene_intent._placement_owner(line, [big, small], opts)["id"] == "small"


def test_placement_owner_ignores_a_raster_that_barely_overlaps():
    line = _line("l1", 0, 0, 100, 20)
    grazing = _element("g", "photo", 90, 0, 100, 20)
    opts = scene_intent.placement_options({})
    assert scene_intent._placement_owner(line, [grazing], opts) is None
