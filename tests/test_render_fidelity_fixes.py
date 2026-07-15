"""CPU regression tests for the text-render fidelity fixes.

Covers the three defects behind the benchmark's editable-text collapse and
text-recall loss:

  1. font-consensus flattened per-line WEIGHT (a family's bold headline face was
     promoted onto regular body copy), so style.fontWeight desynced from the
     rendered top candidate and rendered ink roughly doubled -> raster slices;
  2. numeric/stat strings ("666", "257", "21K") matched a script/serif display
     face by luck when the class vote was too weak to fire the gate;
  3. the raster-slice gate sliced a right-colour, not-ink-heavy render that was
     only a repairable positional offset.
"""
from __future__ import annotations

import numpy as np

from src import text_analysis, font_fit, schema


# ── 1. weight reconciliation + consensus weight preservation ────────────────────

def test_reconcile_style_weight_mirrors_top_candidate():
    # candidates[0] is what render_preview draws; fontWeight/fontStyle must match it.
    style = {
        "fontFamily": "Arimo", "fontWeight": 400, "fontStyle": "Regular",
        "fontCandidates": [
            {"family": "Arimo", "style": "Bold", "weight": 700,
             "source": "local-render", "path": "arialbd.ttf"},
        ],
    }
    text_analysis._reconcile_style_weight(style)
    assert style["fontWeight"] == 700
    assert style["fontStyle"] == "Bold"
    # the invariant the pipeline relies on
    assert style["fontCandidates"][0]["weight"] == style["fontWeight"]


def test_reconcile_preserves_italic_token():
    style = {
        "fontFamily": "Arimo", "fontWeight": 400, "fontStyle": "Italic",
        "fontCandidates": [
            {"family": "Arimo", "style": "Bold", "weight": 700,
             "source": "local-render", "path": "x.ttf"},
        ],
    }
    text_analysis._reconcile_style_weight(style)
    assert style["fontWeight"] == 700
    assert "italic" in style["fontStyle"].lower()


def _consensus_item(line_id, family, fit_score, *, weight, cand_weight, cand_style,
                    w=600.0, h=40.0, text="Sample text", path="font.ttf"):
    return {
        "line": {
            "id": line_id, "text": text,
            "style": {
                "fontFamily": family, "fontSize": 32.0, "fontWeight": weight,
                "fontStyle": cand_style, "letterSpacing": 0.0, "lineHeight": 40.0,
                "fontCandidates": [
                    {"family": family, "path": path, "source": "local-render",
                     "score": 0.5, "weight": cand_weight, "style": cand_style},
                ],
            },
            "meta": {"render_fit": {"family": family, "score": fit_score,
                                    "fontSize": 32.0, "letterSpacing": 0.0,
                                    "applied": True}},
        },
        "painted": {"w": w, "h": h},
        "font_mask": np.ones((16, 64), dtype=bool),
    }


def test_consensus_does_not_flatten_regular_body_to_bold(monkeypatch, tmp_path):
    # A bold headline family (Inter/Bold, w700) must NOT be promoted onto a
    # regular-weight (w400) body line: that renders the wrong stroke weight and
    # doubles ink. The line keeps its own correct-weight match.
    font_path = tmp_path / "f.ttf"
    font_path.write_bytes(b"stub")

    def fake_fit_line(text, path, mask, size, options):
        return {"score": 0.60, "fontSize": 30.0, "letterSpacing": 0.4}
    monkeypatch.setattr(font_fit, "fit_line", fake_fit_line)

    dominant = [
        _consensus_item(f"L{i}", "Inter", 0.60, weight=700, cand_weight=700,
                        cand_style="Bold", path=str(font_path)) for i in range(3)
    ]
    body = _consensus_item("L9", "Courier New", 0.42, weight=400, cand_weight=400,
                           cand_style="Regular", text="body copy line")
    prepared = dominant + [body]
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    # NOT flattened to the bold consensus family.
    assert prepared[-1]["line"]["style"]["fontFamily"] == "Courier New"
    assert prepared[-1]["line"]["style"]["fontWeight"] == 400


def test_consensus_still_unifies_same_weight_lines(monkeypatch, tmp_path):
    # Same-weight consistency still works: a w400 outlier adopts a w400 consensus
    # family, and weight/style stay consistent with the promoted candidate.
    font_path = tmp_path / "f.ttf"
    font_path.write_bytes(b"stub")

    def fake_fit_line(text, path, mask, size, options):
        return {"score": 0.58, "fontSize": 30.0, "letterSpacing": 0.4}
    monkeypatch.setattr(font_fit, "fit_line", fake_fit_line)

    dominant = [
        _consensus_item(f"L{i}", "Inter", 0.58, weight=400, cand_weight=400,
                        cand_style="Regular", path=str(font_path)) for i in range(3)
    ]
    outlier = _consensus_item("L9", "Courier New", 0.42, weight=400, cand_weight=400,
                              cand_style="Regular", text="UPFRONT")
    prepared = dominant + [outlier]
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    style = prepared[-1]["line"]["style"]
    assert style["fontFamily"] == "Inter"
    # promoted candidate == declared weight (render matches export)
    assert style["fontCandidates"][0]["weight"] == style["fontWeight"] == 400


# ── 2. numeric strings never match a script/decorative face ─────────────────────

def test_numeric_string_excludes_script_when_class_undecided(monkeypatch):
    # When the sans/serif call is undecided (weak vote on few simple glyphs), a
    # numeric/stat string still excludes script/decorative so a swash face can't
    # win by luck (benchmark 009: '666'/'89' -> Dancing Script).
    monkeypatch.setattr(font_fit, "classify_source",
                        lambda *a, **k: {"class": None, "confidence": 0.0,
                                         "text_confidence": 0.0, "scores": {}})
    captured = {}
    real_filter = font_fit.filter_fonts_by_class

    def spy_filter(fonts, source_class):
        captured["source_class"] = source_class
        return real_filter(fonts, source_class)
    monkeypatch.setattr(font_fit, "filter_fonts_by_class", spy_filter)

    mask = np.ones((20, 60), dtype=bool)
    geo = {"weight": 400, "italic": False, "font_size": 40.0, "shear_angle": None}
    text_analysis._resolve_font_candidates(
        "257", mask, geo, {"enabled": True, "class_gate": True, "top_k": 3, "max_fonts": 8},
        render_fit={"enabled": False})
    # TEXT admits sans+serif but excludes script/decorative.
    assert captured.get("source_class") == font_fit.TEXT
    assert font_fit.compatible(font_fit.TEXT, font_fit.SCRIPT) is False
    assert font_fit.compatible(font_fit.TEXT, font_fit.SANS) is True


def test_non_numeric_short_word_unaffected(monkeypatch):
    # A short non-numeric word is NOT forced to TEXT by the numeric rule (it may be
    # a genuine short wordmark handled elsewhere); the gate stays open.
    monkeypatch.setattr(font_fit, "classify_source",
                        lambda *a, **k: {"class": None, "confidence": 0.0,
                                         "text_confidence": 0.0, "scores": {}})
    captured = {}

    def spy_filter(fonts, source_class):
        captured["source_class"] = source_class
        return fonts
    monkeypatch.setattr(font_fit, "filter_fonts_by_class", spy_filter)
    mask = np.ones((20, 60), dtype=bool)
    geo = {"weight": 400, "italic": False, "font_size": 40.0, "shear_angle": None}
    text_analysis._resolve_font_candidates(
        "Go", mask, geo, {"enabled": True, "class_gate": True, "top_k": 3, "max_fonts": 8},
        render_fit={"enabled": False})
    assert captured.get("source_class") is None


# ── 3. raster-slice gate keeps a plausible positional-offset render editable ─────

def test_gate_keeps_right_colour_light_offset_editable():
    t = schema.raster_slice_thresholds({})
    # Low ink-IoU but RIGHT colour and NOT ink-heavy = repairable offset -> keep.
    keep = schema.raster_slice_failures(
        {"type": "text", "ink_iou": 0.24, "region_ssim": 0.5,
         "region_color": 0.92, "ink_excess": 0.30}, t)
    assert keep == []


def test_gate_still_slices_offcolour_low_iou_render():
    # Text bypasses the slice gate by default (Codia-parity policy); the legacy
    # ink gates remain testable behind the forensic flag.
    t = schema.raster_slice_thresholds({"fallback": {"text_slice_gate_enabled": True}})
    assert schema.raster_slice_failures(
        {"type": "text", "ink_iou": 0.24, "region_ssim": 0.5,
         "region_color": 0.55, "ink_excess": 0.30},
        schema.raster_slice_thresholds({})) == []
    # Low ink-IoU AND off-colour render is genuinely wrong -> forensic gate reports it.
    reasons = schema.raster_slice_failures(
        {"type": "text", "ink_iou": 0.24, "region_ssim": 0.5,
         "region_color": 0.55, "ink_excess": 0.30}, t)
    assert reasons and any("ink_iou" in r for r in reasons)


def test_gate_still_slices_low_iou_ink_heavy_render():
    t = schema.raster_slice_thresholds({"fallback": {"text_slice_gate_enabled": True}})
    # Low ink-IoU with heavy excess ink -> forensic gate (flag-only) reports it.
    reasons = schema.raster_slice_failures(
        {"type": "text", "ink_iou": 0.24, "region_ssim": 0.5,
         "region_color": 0.92, "ink_excess": 0.80}, t)
    assert reasons and any("ink_iou" in r for r in reasons)
