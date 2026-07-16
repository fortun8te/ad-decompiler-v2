"""Handwriting / marker-face detection: the cheap stages, the fail-safe, and the eval.

The unit tests here are offline and deterministic.  The Stage-B classifier eval at the
bottom needs a live VLM and is therefore opt-in (``AD_DECOMP_VLM_EVAL=1``): a test that
silently dials an endpoint would trip vlm_client's process-global circuit breaker and
poison every unrelated VLM test in the session.
"""
import json
import os

import numpy as np
import pytest

from src import handwriting


def _typeset_mask(width=200, height=40):
    """A blocky, uniform-stroke run: stands in for clean typeset ink."""
    m = np.zeros((height, width), bool)
    for x in range(10, width - 10, 24):
        m[8:height - 8, x:x + 6] = True
    return m


# --- cheap stats -----------------------------------------------------------------


def test_ink_stats_are_none_for_degenerate_masks():
    assert handwriting.ink_stats(np.zeros((3, 3), bool)) == {
        "stroke_width_cv": None, "baseline_wobble": None}


def test_stroke_width_cv_is_low_for_a_uniform_stroke():
    cv = handwriting.stroke_width_cv(_typeset_mask())
    assert cv is not None and cv < 0.35, cv


def test_stroke_width_mean_tracks_the_authored_stroke_width():
    """A 6px-wide bar measures ~4.7px: the percentile ridge includes some off-centre
    pixels, so the estimate reads slightly UNDER the true width. That bias is constant
    and cancels in the word/line-median RATIO the jitter clamp actually consumes, so the
    contract is proportionality, not calibration."""
    width = handwriting.stroke_width_mean(_typeset_mask())
    assert width is not None
    assert 4.0 <= width <= 7.0, width


def test_stroke_width_mean_separates_a_bold_run_from_a_regular_one():
    """The signal the word-weight jitter clamp leans on: heavier ink, wider strokes.

    It must move with the STROKE, not with how much of the box is inked, which is what
    makes it independent corroboration for a density-driven weight flip.
    """
    def bars(stroke_px):
        m = np.zeros((40, 200), bool)
        for x in range(10, 190, 24):
            m[8:32, x:x + stroke_px] = True
        return m

    regular = handwriting.stroke_width_mean(bars(4))
    bold = handwriting.stroke_width_mean(bars(10))
    assert regular is not None and bold is not None
    assert bold > regular * 1.5, (regular, bold)


def test_stroke_width_mean_is_none_for_a_degenerate_mask():
    assert handwriting.stroke_width_mean(np.zeros((3, 3), bool)) is None


def test_baseline_wobble_ignores_a_slanted_baseline():
    # An italic/oblique run sits on a straight but SLANTED baseline. Wobble must measure
    # roughness, not slant, or every oblique typeset line reads as hand-lettered.
    m = np.zeros((60, 200), bool)
    for x in range(10, 190):
        top = 10 + int(x * 0.08)
        m[top:top + 20, x] = True
    wobble = handwriting.baseline_wobble(m)
    assert wobble is not None and wobble < 0.1, wobble


# --- Stage A: recall, and the guards that keep it off short/garbled lines --------


def test_stage_a_skips_fragments_too_short_to_classify():
    ok, sig = handwriting.stage_a_candidate({"text": "a", "conf": 0.9}, {}, {})
    assert not ok and sig["skip"] == "too-short"


def test_stage_a_skips_low_confidence_ocr():
    # Garbled/occluded ink already routes to the ink fallback on its own merits; spending
    # a raster decision on it would be a guess about a line we cannot even read.
    ok, sig = handwriting.stage_a_candidate({"text": "Sharp", "conf": 0.2}, {}, {})
    assert not ok and sig["skip"].startswith("low-conf")


def test_stage_a_flags_a_plausible_font_that_renders_back_wrong():
    # The 091 "Sharp" signal: a strong shape match that nonetheless cannot reproduce the
    # ink. Stage A is deliberately loose here — Stage B is what decides.
    line = {
        "text": "Sharp", "conf": 0.9,
        "meta": {"render_fit": {"score": 0.20}},
        "style": {"fontCandidates": [{"family": "Barlow Condensed", "score": 0.89,
                                      "source": "local-render"}]},
    }
    ok, sig = handwriting.stage_a_candidate(line, {}, {})
    assert ok and "renderback_mismatch" in sig["flags"]


def test_stage_a_keeps_a_line_whose_font_reproduces_it():
    # 013 "do this!" renders back at 0.52 — a library font DOES reproduce it, so it must
    # never reach the gate no matter how display-y it looks.
    line = {
        "text": "do this!", "conf": 0.9,
        "meta": {"render_fit": {"score": 0.52}},
        "style": {"fontCandidates": [{"family": "Inter", "score": 1.0,
                                      "source": "local-render"}]},
    }
    ok, _ = handwriting.stage_a_candidate(line, {}, {})
    assert not ok


def test_select_candidates_spends_the_budget_on_the_largest_ink():
    entries = [(f"L{i}", {"w": 10 * i, "h": 10}) for i in range(1, 10)]
    chosen = handwriting.select_candidates(entries, {"handwriting": {"vlm": {"max_lines": 3}}})
    assert chosen == ["L9", "L8", "L7"]


# --- the fail-safe: no VLM, no rasterization ------------------------------------


def _candidate_line():
    return {
        "text": "Sharp", "conf": 0.9,
        "meta": {"render_fit": {"score": 0.20}},
        "style": {"fontCandidates": [{"family": "Barlow Condensed", "score": 0.89,
                                      "source": "local-render"}]},
    }


def test_decide_never_rasterizes_without_the_vlm():
    # Stats CANNOT separate handwriting from display type (091 "Sharp" scores a LOWER
    # stroke-width CV than the typeset lines it must be told apart from), so a missing
    # VLM must mean "leave it editable", never "guess from stats".
    got = handwriting.decide(_candidate_line(), {"stroke_width_cv": 0.9}, None, {})
    assert got["rasterize"] is False
    assert "keeping-native" in got["reason"]


def test_decide_rasterizes_only_on_a_confirmed_non_substitutable_face(monkeypatch):
    cfg = {"vlm": {"enabled": True},
           "text_analysis": {"handwriting": {"vlm": {"enabled": True}}}}

    monkeypatch.setattr(handwriting, "vlm_classify", lambda *a, **k: {
        "available": True, "handwritten": True, "style": "marker",
        "confidence": 0.9, "note": "ok"})
    got = handwriting.decide(_candidate_line(), {}, b"png", cfg)
    assert got["rasterize"] is True and got["handwriting"] is True

    # Same weak render-back, but the face IS an ordinary typeface -> stays editable.
    monkeypatch.setattr(handwriting, "vlm_classify", lambda *a, **k: {
        "available": True, "handwritten": False, "style": "plain_serif",
        "confidence": 0.95, "note": "ok"})
    got = handwriting.decide(_candidate_line(), {}, b"png", cfg)
    assert got["rasterize"] is False and got["reason"] == "vlm-typeset"


def test_a_line_its_font_reproduces_never_reaches_the_gate():
    # A strong render-back means a library face DOES reproduce the ink. Such a line is
    # not even a Stage-A candidate, so it can never be rasterized however hand-drawn it
    # looks — the cheapest possible protection for editable copy.
    line = _candidate_line()
    line["meta"]["render_fit"]["score"] = 0.61
    got = handwriting.decide(line, {}, b"png", {})
    assert got["rasterize"] is False and got["reason"] == "not-a-candidate"


def test_confirmed_marker_face_stays_native_when_a_font_reproduces_it(monkeypatch):
    # Reached via the stroke-CV flag rather than the render-back flag: the VLM confirms a
    # script face, but a library font renders it back well. A chip is a last resort, not
    # a reward for looking hand-drawn — so keep it editable and record the evidence.
    cfg = {"vlm": {"enabled": True},
           "text_analysis": {"handwriting": {"vlm": {"enabled": True}}}}
    monkeypatch.setattr(handwriting, "vlm_classify", lambda *a, **k: {
        "available": True, "handwritten": True, "style": "script",
        "confidence": 0.9, "note": "ok"})
    line = _candidate_line()
    line["meta"]["render_fit"]["score"] = 0.61          # a real match
    got = handwriting.decide(line, {"stroke_width_cv": 0.55}, b"png", cfg)
    assert got["rasterize"] is False
    assert got["handwriting"] is True                    # still recorded as evidence
    assert got["reason"] == "vlm-handwritten-but-font-reproduces"


def test_vlm_classify_is_off_unless_config_opts_in():
    # Default OFF, and gated on the pipeline's root vlm.enabled: analyze_text must stay a
    # pure local computation for unit tests and --no-vlm runs.
    assert handwriting.vlm_classify(b"png", {})["note"] == "vlm-disabled"
    assert handwriting.vlm_classify(b"png", {"vlm": {"enabled": True}})["note"] == "vlm-disabled"
    cfg = {"text_analysis": {"handwriting": {"vlm": {"enabled": True}}}}   # root off
    assert handwriting.vlm_classify(b"png", cfg)["note"] == "vlm-disabled"


# --- Stage B classifier eval (opt-in; needs a live gemma-4-e4b) ------------------

# Ground truth by inspection of the benchmark-6 sources. "hand" = a marker/script face
# with no ordinary-typeface substitute (must be reproduced faithfully, i.e. chipped);
# "plain" = an ordinary typeface that MUST stay native.
VLM_EVAL_CASES = [
    ("091", "L0", "hand"),     # "Sharp" — rounded marker face; corpus best fit is 0.27
    ("091", "L25", "plain"),   # "Foggy and Steady" — serif, with a red swipe drawn over it
    ("013", "L7", "plain"),    # "do this!" — typeset display
    ("013", "L0", "plain"),    # "We NEVER"
    ("013", "L2", "plain"),    # "+ FREE GIFTS"
    ("025", "L0", "plain"),    # "Why Everyone's"
    ("025", "L1", "plain"),    # "Switching to Hears"
    ("067", "L3", "plain"),    # "frøya" wordmark
    ("016", "L1", "plain"),    # "nutrients on Ozempic:"
    ("088", "L0", "plain"),    # "Black Friday" — serif italic
    ("091", "L31", "plain"),   # "SUPPORTS FOCUS"
    ("104", "L0", "plain"),    # "Cadence" — display serif
]

_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "runs", "postfix-benchmark-6")


@pytest.mark.skipif(os.environ.get("AD_DECOMP_VLM_EVAL") != "1",
                    reason="needs a live VLM; opt in with AD_DECOMP_VLM_EVAL=1")
def test_vlm_style_classifier_eval():
    """Measure Stage B against the labelled set before trusting it to rasterize.

    The HARD contract is zero false rasterization of typeset copy, so this asserts on
    false POSITIVES first: every 'plain' line must be classified plain.  Run with
    LM Studio serving google/gemma-4-e4b.
    """
    from PIL import Image

    from src import vlm_client

    cfg = {"vlm": {"enabled": True},
           "text_analysis": {"handwriting": {"vlm": {"enabled": True}}}}
    results, missing = [], []
    for fx, lid, truth in VLM_EVAL_CASES:
        run = next((os.path.join(_BENCH, d) for d in sorted(os.listdir(_BENCH))
                    if d.startswith(fx)), None)
        if not run or not os.path.isfile(os.path.join(run, "ocr.json")):
            missing.append(fx)
            continue
        ocr = json.load(open(os.path.join(run, "ocr.json"), encoding="utf-8"))
        line = next((l for l in ocr["lines"] if l["id"] == lid), None)
        if line is None:
            missing.append(f"{fx}/{lid}")
            continue
        img = Image.open(os.path.join(run, "normalized.png")).convert("RGB")
        crop = vlm_client.crop_box_bytes(img, line.get("painted_box") or line["box"], 6)
        got = handwriting.vlm_classify(crop, cfg)
        assert got.get("available"), f"{fx}/{lid}: VLM unavailable ({got.get('note')})"
        bucket = "hand" if got["handwritten"] else "plain"
        results.append((fx, lid, line["text"][:24], truth, got["style"], bucket))
    if missing:
        pytest.skip(f"benchmark-6 artifacts missing for {missing}")

    false_pos = [r for r in results if r[3] == "plain" and r[5] == "hand"]
    false_neg = [r for r in results if r[3] == "hand" and r[5] == "plain"]
    assert not false_pos, (
        "typeset copy classified as a non-substitutable face — these would be wrongly "
        f"rasterized: {false_pos}")
    assert not false_neg, f"marker/script face missed: {false_neg}"
