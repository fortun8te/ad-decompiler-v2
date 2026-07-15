"""CPU-only tests for render-and-fit font refinement and glyph-class gating."""
from __future__ import annotations

import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import font_fit  # noqa: E402


def _find_font(*names):
    roots = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
    ]
    for name in names:
        if os.path.isabs(name) and os.path.isfile(name):
            return name
        for root in roots:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
    return None


SANS_PATH = _find_font("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
SANS_PATH_2 = _find_font("segoeui.ttf", "Helvetica.ttc", "DejaVuSans.ttf")
SERIF_PATH = _find_font("times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf",
                        "LiberationSerif-Regular.ttf")
SCRIPT_PATH = _find_font("Gabriola.ttf", "segoesc.ttf", "Inkfree.ttf", "Comic Sans MS.ttf")


def _render_line_mask(text, path, size, tracking=0.0):
    """Source-mask fixture rendered the same way ads paint text."""
    font = ImageFont.truetype(path, size)
    probe = Image.new("L", (8, 8), 0)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    advances = [font.getlength(ch) for ch in text]
    width = int(sum(advances) + tracking * max(0, len(text) - 1)) + 16
    canvas = Image.new("L", (width, bbox[3] - bbox[1] + 16), 0)
    draw = ImageDraw.Draw(canvas)
    x = 8.0
    for ch, adv in zip(text, advances):
        draw.text((x, 8 - bbox[1]), ch, fill=255, font=font)
        x += adv + tracking
    mask = np.asarray(canvas) > 32
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ---------------------------------------------------------------------------
# Render-and-fit refinement


@pytest.mark.skipif(SANS_PATH is None, reason="no sans test font available")
def test_fit_line_recovers_size_and_tracking_from_bad_initial_estimate():
    text = "korting krijgt op het volledige"
    source = _render_line_mask(text, SANS_PATH, 42, tracking=0.0)
    # Deliberately wrong initial size — the cap-height heuristic overshoots
    # lines containing both ascenders and descenders by ~40%.
    fit = font_fit.fit_line(text, SANS_PATH, source, initial_size=42 * 1.6)
    assert fit is not None
    assert abs(fit["fontSize"] - 42) <= 42 * 0.08
    assert abs(fit["letterSpacing"]) <= 1.0
    assert fit["score"] >= 0.55


@pytest.mark.skipif(SANS_PATH is None, reason="no sans test font available")
def test_render_and_fit_improves_ink_overlap_over_initial_estimate():
    text = "UPFRONT"
    source = _render_line_mask(text, SANS_PATH, 36, tracking=0.0)
    # The pre-fit pipeline estimate: cap-height-derived size and heuristic
    # tracking (this exact combination produced the tracked-out headline).
    bad_size, bad_tracking = 36 * 1.35, 4.9
    before = font_fit._render_tracked_mask(text, SANS_PATH, bad_size, bad_tracking)
    score_before = font_fit._aligned_iou(source, before)
    fit = font_fit.fit_line(text, SANS_PATH, source, initial_size=bad_size)
    after = font_fit._render_tracked_mask(text, SANS_PATH, fit["fontSize"], fit["letterSpacing"])
    score_after = font_fit._aligned_iou(source, after)
    assert score_after > score_before
    assert score_after >= 0.6


@pytest.mark.skipif(SANS_PATH is None, reason="no sans test font available")
def test_fit_line_fails_soft_on_missing_inputs():
    source = _render_line_mask("Hi", SANS_PATH, 30)
    assert font_fit.fit_line("", SANS_PATH, source, 30) is None
    assert font_fit.fit_line("Hi", "/no/such/font.ttf", source, 30) is None
    assert font_fit.fit_line("Hi", SANS_PATH, None, 30) is None
    assert font_fit.fit_line("Hi", SANS_PATH, np.zeros((10, 10), dtype=bool), 30) is None


# ---------------------------------------------------------------------------
# Candidate classification (PANOSE + name heuristics)


@pytest.mark.skipif(SANS_PATH is None or SERIF_PATH is None,
                    reason="need one sans and one serif font")
def test_classify_font_file_separates_sans_and_serif():
    assert font_fit.classify_font_file(SANS_PATH) == font_fit.SANS
    assert font_fit.classify_font_file(SERIF_PATH) == font_fit.SERIF


@pytest.mark.skipif(SCRIPT_PATH is None, reason="no script/decorative font available")
def test_classify_font_file_flags_script_or_decorative():
    assert font_fit.classify_font_file(SCRIPT_PATH) in (font_fit.SCRIPT, font_fit.DECORATIVE)


def test_classify_font_file_name_heuristics_without_panose(tmp_path):
    missing = tmp_path / "Fancy-Handwriting-Script.ttf"  # unreadable -> name fallback
    missing.write_bytes(b"not a font")
    assert font_fit.classify_font_file(str(missing)) == font_fit.SCRIPT


def test_compatibility_fails_open_on_unknown_classes():
    assert font_fit.compatible(None, font_fit.SERIF) is True
    assert font_fit.compatible(font_fit.SANS, None) is True
    assert font_fit.compatible(font_fit.SANS, font_fit.SERIF) is False
    assert font_fit.compatible(font_fit.SANS, font_fit.DECORATIVE) is False
    assert font_fit.compatible(font_fit.SCRIPT, font_fit.DECORATIVE) is True


@pytest.mark.skipif(SANS_PATH is None or SERIF_PATH is None,
                    reason="need one sans and one serif font")
def test_filter_fonts_by_class_hard_filters_but_never_empties():
    fonts = [{"path": SANS_PATH, "family": "Sans"}, {"path": SERIF_PATH, "family": "Serif"}]
    kept = font_fit.filter_fonts_by_class(fonts, font_fit.SANS)
    assert [meta["family"] for meta in kept] == ["Sans"]
    # Filtering everything falls open instead of leaving nothing to match.
    only_serif = [{"path": SERIF_PATH, "family": "Serif"}]
    assert font_fit.filter_fonts_by_class(only_serif, font_fit.SANS) == only_serif


# ---------------------------------------------------------------------------
# Comparative source classification


@pytest.mark.skipif(SANS_PATH is None or SERIF_PATH is None,
                    reason="need one sans and one serif font")
def test_classify_source_votes_by_reference_fit():
    text = "Daarbovenop krijgen de eerste"
    refs = {"sans": [SANS_PATH], "serif": [SERIF_PATH]}
    if SCRIPT_PATH:
        refs["script"] = [SCRIPT_PATH]
    options = {"reference_fonts": refs}

    sans_mask = _render_line_mask(text, SANS_PATH_2 or SANS_PATH, 40)
    sans_info = font_fit.classify_source(text, sans_mask, 40, options)
    assert sans_info["class"] == font_fit.SANS

    serif_mask = _render_line_mask(text, SERIF_PATH, 40)
    serif_info = font_fit.classify_source(text, serif_mask, 40, options)
    assert serif_info["class"] == font_fit.SERIF


def test_classify_source_returns_none_without_references():
    mask = np.zeros((20, 80), dtype=bool)
    mask[4:16, 4:70] = True
    info = font_fit.classify_source("Hello", mask, 16, {"reference_fonts": {}})
    assert info["class"] is None


# ---------------------------------------------------------------------------
# Candidate refinement / rejection ranking


@pytest.mark.skipif(SANS_PATH is None or SCRIPT_PATH is None,
                    reason="need a sans and a script font")
def test_refine_candidates_rejects_swash_face_for_sans_body_text():
    text = "korting krijgt op het volledige"
    source = _render_line_mask(text, SANS_PATH, 42)
    candidates = [
        # The aspect-blind shape matcher scored the swash *higher* — the exact
        # 009 failure. The fitted evidence must invert that ranking.
        {"family": "Swash", "style": "Regular", "weight": 400, "score": 0.86,
         "source": "local-render", "path": SCRIPT_PATH},
        {"family": "PlainSans", "style": "Regular", "weight": 400, "score": 0.80,
         "source": "local-render", "path": SANS_PATH},
        {"family": "Inter", "style": "Regular", "weight": 400, "score": 0.62,
         "source": "fallback"},
    ]
    ranked, evidence = font_fit.refine_candidates(text, source, candidates, 42, {"enabled": True})
    assert evidence["fitted"] == 2
    assert ranked[0]["family"] == "PlainSans"
    assert ranked[0]["fit"]["rejected"] is False
    swash = next(item for item in ranked if item["family"] == "Swash")
    if swash["fit"]["rejected"]:
        # A rejected fit is evidence against the font: below even the fallback.
        assert ranked[-1]["family"] == "Swash"
    else:
        assert swash["fit"]["score"] < ranked[0]["fit"]["score"]


@pytest.mark.skipif(SANS_PATH is None, reason="no sans test font available")
def test_refine_candidates_prefers_google_cache_on_tied_fits(tmp_path):
    text = "Cache"
    source = _render_line_mask(text, SANS_PATH, 36)
    google_copy = tmp_path / "SameSans-Regular.ttf"
    google_copy.write_bytes(open(SANS_PATH, "rb").read())
    candidates = [
        {"family": "SameSans", "style": "Regular", "weight": 400, "score": 0.9,
         "source": "local-render", "path": SANS_PATH},
        {"family": "SameSans", "style": "Regular", "weight": 400, "score": 0.9,
         "source": "google-cache", "path": str(google_copy)},
    ]
    ranked, _ = font_fit.refine_candidates(text, source, candidates, 36, {"enabled": True})
    # Identical font bytes fit identically; the Figma-insertable Google-cache
    # entry must win the tie so the plugin can actually load the family.
    assert ranked[0]["source"] == "google-cache"


def test_refine_candidates_disabled_passthrough():
    candidates = [{"family": "A", "source": "local-render", "path": "/x.ttf", "score": 0.5}]
    ranked, evidence = font_fit.refine_candidates("Hi", None, candidates, 20, {"enabled": False})
    assert ranked == candidates
    assert evidence["enabled"] is False


# ---------------------------------------------------------------------------
# Confidence calibration: short-string reliability + correct-vs-wrong separation


def test_length_reliability_ramps_with_glyph_count():
    # A 1-3 glyph run is discounted; a run of _RELIABLE_GLYPHS+ is full confidence.
    assert font_fit._length_reliability("6") < font_fit._length_reliability("66")
    assert font_fit._length_reliability("66") < font_fit._length_reliability("257")
    assert font_fit._length_reliability("257") < 1.0
    assert font_fit._length_reliability("assortiment") == 1.0
    # Whitespace does not count toward the glyph budget.
    assert font_fit._length_reliability("a b") == font_fit._length_reliability("ab")


@pytest.mark.skipif(SANS_PATH is None, reason="no sans test font available")
def test_short_digit_string_fit_is_not_spuriously_certain():
    # The exact 009 failure: 2-3 digit runs ("66", "257") reported ~0.94 and the
    # wrong family passed as confident text. Even the *exact* source font must not
    # publish near-certain confidence on so few glyphs, and the published score
    # must sit below the raw self-overlap IoU (the reliability discount).
    for text in ("66", "257"):
        source = _render_line_mask(text, SANS_PATH, 40)
        fit = font_fit.fit_line(text, SANS_PATH, source, initial_size=40)
        assert fit is not None
        assert fit["score"] < 0.90
        raw = font_fit._aligned_iou(source, source)  # 1.0 self-overlap upper bound
        assert fit["score"] < raw
    # A long same-class run is undiscounted and still lands in exact-font territory.
    long_text = "Daarbovenop krijgen de eerste"
    src_long = _render_line_mask(long_text, SANS_PATH, 40)
    long_fit = font_fit.fit_line(long_text, SANS_PATH, src_long, initial_size=40)
    assert long_fit is not None and long_fit["score"] >= 0.80


@pytest.mark.skipif(SANS_PATH is None or SCRIPT_PATH is None,
                    reason="need a sans and a script/decorative font")
def test_correct_family_scores_clearly_above_wrong_family_on_same_ink():
    # The calibrated fit must be a trustworthy separator: on identical ink the
    # correct-class render beats a wrong-class (script/decorative) render by a
    # clear margin, across sizes.
    text = "Summer Sale Today"
    for size in (32, 56):
        source = _render_line_mask(text, SANS_PATH, size)
        correct = font_fit.fit_line(text, SANS_PATH, source, initial_size=size)
        wrong = font_fit.fit_line(text, SCRIPT_PATH, source, initial_size=size)
        assert correct is not None and wrong is not None
        assert correct["score"] >= wrong["score"] + 0.15


def test_compatible_text_class_admits_sans_serif_rejects_script():
    assert font_fit.compatible(font_fit.TEXT, font_fit.SANS) is True
    assert font_fit.compatible(font_fit.TEXT, font_fit.SERIF) is True
    assert font_fit.compatible(font_fit.TEXT, font_fit.SCRIPT) is False
    assert font_fit.compatible(font_fit.TEXT, font_fit.DECORATIVE) is False
    # Only positively-classed script/decorative faces are excluded; an unknown
    # candidate class still fails open.
    assert font_fit.compatible(font_fit.TEXT, None) is True


@pytest.mark.skipif(SANS_PATH is None or SERIF_PATH is None or SCRIPT_PATH is None,
                    reason="need sans, serif and script fonts")
def test_text_source_class_gate_excludes_script_candidate():
    # A serif headline (or any plain text) must never keep a script/decorative
    # candidate: the TEXT gate drops it while retaining both text classes.
    fonts = [
        {"path": SANS_PATH, "family": "Sans"},
        {"path": SERIF_PATH, "family": "Serif"},
        {"path": SCRIPT_PATH, "family": "Script"},
    ]
    kept = {meta["family"] for meta in font_fit.filter_fonts_by_class(fonts, font_fit.TEXT)}
    assert "Script" not in kept
    assert {"Sans", "Serif"} <= kept


@pytest.mark.skipif(SANS_PATH is None or SERIF_PATH is None or SCRIPT_PATH is None,
                    reason="need sans, serif and script fonts")
def test_classify_source_reports_high_text_confidence_for_plain_text():
    refs = {"sans": [SANS_PATH], "serif": [SERIF_PATH], "script": [SCRIPT_PATH]}
    options = {"reference_fonts": refs}
    # Plain sans caps/digits: sans-vs-serif may be a toss-up, but the source is
    # unmistakably TEXT (fits both text references far better than the script one).
    for text, font in (("257", SANS_PATH_2 or SANS_PATH), ("Perfect curls", SERIF_PATH)):
        info = font_fit.classify_source(text, _render_line_mask(text, font, 44), 44, options)
        assert info["text_confidence"] >= 0.5
    # A source that matches the script reference is detected as script, not text.
    script_info = font_fit.classify_source(
        "Sale", _render_line_mask("Sale", SCRIPT_PATH, 44), 44, options)
    assert script_info["class"] == font_fit.SCRIPT
