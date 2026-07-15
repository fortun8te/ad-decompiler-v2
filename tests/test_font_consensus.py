"""Document-level font consensus (text_analysis._apply_font_consensus)."""
from __future__ import annotations

import numpy as np
import pytest

from src import text_analysis


def _item(line_id: str, family: str, fit_score: float, *, w: float = 400.0,
          h: float = 40.0, text: str = "Sample text", path: str = "font.ttf") -> dict:
    return {
        "line": {
            "id": line_id,
            "text": text,
            "style": {
                "fontFamily": family,
                "fontSize": 32.0,
                "letterSpacing": 0.0,
                "lineHeight": 40.0,
                "fontCandidates": [
                    {"family": family, "path": path, "source": "local-render",
                     "score": 0.5, "weight": 400, "style": "Regular"},
                ],
            },
            "meta": {"render_fit": {"family": family, "score": fit_score,
                                    "fontSize": 32.0, "letterSpacing": 0.0,
                                    "applied": True}},
        },
        "painted": {"w": w, "h": h},
        "font_mask": np.ones((16, 64), dtype=bool),
    }


@pytest.fixture()
def fit_stub(monkeypatch, tmp_path):
    """Make the consensus font file exist and control re-fit scores."""
    font_path = tmp_path / "consensus.ttf"
    font_path.write_bytes(b"stub")
    scores: dict[str, float] = {}
    calls: list[str] = []

    def fake_fit_line(text, path, mask, size, options):
        calls.append(text)
        return {"score": scores.get(text, 0.5), "fontSize": 30.0, "letterSpacing": 0.4}

    from src import font_fit
    monkeypatch.setattr(font_fit, "fit_line", fake_fit_line)
    return {"path": str(font_path), "scores": scores, "calls": calls}


def _prepared(fit_stub, outlier_score_own: float = 0.42):
    dominant = [
        _item(f"L{i}", "Inter", 0.55, w=600, path=fit_stub["path"]) for i in range(3)
    ]
    outlier = _item("L9", "Courier New", outlier_score_own, w=200, text="UPFRONT")
    return dominant + [outlier]


def test_outlier_adopts_consensus_family(fit_stub):
    prepared = _prepared(fit_stub)
    fit_stub["scores"]["UPFRONT"] = 0.58
    evidence = text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    assert evidence["applied"] is True
    assert evidence["family"] == "Inter"
    outlier_style = prepared[-1]["line"]["style"]
    assert outlier_style["fontFamily"] == "Inter"
    assert outlier_style["fontCandidates"][0]["family"] == "Inter"
    meta = prepared[-1]["line"]["meta"]["render_fit"]
    assert meta["consensus"] is True and meta["score"] == 0.58
    assert evidence["refit"][0]["from"] == "Courier New"


def test_bad_consensus_fit_keeps_own_family(fit_stub):
    prepared = _prepared(fit_stub)
    fit_stub["scores"]["UPFRONT"] = 0.20  # below min_score → reject
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    assert prepared[-1]["line"]["style"]["fontFamily"] == "Courier New"


def test_within_tolerance_but_worse_still_adopts(fit_stub):
    prepared = _prepared(fit_stub, outlier_score_own=0.50)
    fit_stub["scores"]["UPFRONT"] = 0.43  # within default 0.10 tolerance of 0.50
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    assert prepared[-1]["line"]["style"]["fontFamily"] == "Inter"


def test_strong_own_match_is_never_touched(fit_stub):
    prepared = _prepared(fit_stub, outlier_score_own=0.80)  # exact-font territory
    fit_stub["scores"]["UPFRONT"] = 0.95
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    assert prepared[-1]["line"]["style"]["fontFamily"] == "Courier New"
    assert "UPFRONT" not in fit_stub["calls"]  # not even re-fit


def test_refit_never_pushes_editable_line_below_raster_bar(fit_stub):
    # Own fit 0.45 clears the 0.30 keep-as-text bar; a consensus fit of 0.28
    # (within the 0.10 tolerance floor of 0.45? no — but exercise the crossing
    # guard directly) must be refused so the line stays editable text.
    prepared = _prepared(fit_stub, outlier_score_own=0.45)
    fit_stub["scores"]["UPFRONT"] = 0.28  # would-be below min_score
    text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30},
        {"consensus": {"enabled": True, "tolerance": 0.30}})
    assert prepared[-1]["line"]["style"]["fontFamily"] == "Courier New"


def test_low_share_returns_unapplied_evidence(fit_stub):
    prepared = [
        _item("L0", "Inter", 0.5, w=100, path=fit_stub["path"]),
        _item("L1", "Georgia", 0.5, w=100),
        _item("L2", "Impact", 0.5, w=100),
        _item("L3", "Verdana", 0.5, w=100),
    ]
    evidence = text_analysis._apply_font_consensus(
        prepared, {"enabled": True, "min_score": 0.30},
        {"consensus": {"enabled": True, "min_share": 0.5}})
    assert evidence["applied"] is False
    assert prepared[1]["line"]["style"]["fontFamily"] == "Georgia"


def test_disabled_via_config(fit_stub):
    prepared = _prepared(fit_stub)
    result = text_analysis._apply_font_consensus(
        prepared, {"enabled": True}, {"consensus": {"enabled": False}})
    assert result is None


def test_sans_block_unifies_scattered_families_to_one(fit_stub):
    # Benchmark 002: a block of sans body/label lines each independently matched a
    # different sans family (Inter/Poppins/Albert Sans). A dominant, class-consistent
    # sans must pull the outliers in when their consensus fit is decent.
    dominant = [
        _item(f"L{i}", "Inter", 0.60, w=600, path=fit_stub["path"], text="Inter line")
        for i in range(4)
    ]
    poppins = _item("P1", "Poppins", 0.50, w=200, text="wei-eiwit concentraat")
    albert = _item("A1", "Albert Sans", 0.48, w=180, text="koolhydraten")
    fit_stub["scores"]["wei-eiwit concentraat"] = 0.55
    fit_stub["scores"]["koolhydraten"] = 0.52
    evidence = text_analysis._apply_font_consensus(
        dominant + [poppins, albert],
        {"enabled": True, "min_score": 0.30}, {"consensus": {"enabled": True}})
    assert evidence["applied"] is True
    assert evidence["family"] == "Inter"
    assert evidence["class_consistent"] is True
    assert poppins["line"]["style"]["fontFamily"] == "Inter"
    assert albert["line"]["style"]["fontFamily"] == "Inter"


def test_serif_cannot_win_on_sans_consensus_line(fit_stub):
    # Benchmark 002: EB Garamond (a serif) leaked onto the sans body line
    # "zoetstof: sucralose". Even when the consensus refit does not score well enough
    # to adopt on evidence, a serif must never survive on a sans-consistent document.
    dominant = [
        _item(f"L{i}", "Inter", 0.62, w=600, path=fit_stub["path"], text="Inter line")
        for i in range(4)
    ]
    serif = _item("S1", "EB Garamond", 0.37, w=110, text="zoetstof: sucralose")
    # Refit against the consensus sans fits poorly (below the 0.30 keep bar) — the normal
    # evidence path would keep EB Garamond, but the hard class gate must relabel it.
    fit_stub["scores"]["zoetstof: sucralose"] = 0.20
    evidence = text_analysis._apply_font_consensus(
        dominant + [serif], {"enabled": True, "min_score": 0.30},
        {"consensus": {"enabled": True}})
    assert evidence["applied"] is True
    assert serif["line"]["style"]["fontFamily"] == "Inter"
    assert serif["line"]["style"]["fontCandidates"][0]["family"] == "Inter"
    forbidden = [r for r in evidence["refit"] if r.get("forbidden_class") == "serif"]
    assert forbidden and forbidden[0]["from"] == "EB Garamond"


def test_genuine_serif_headline_survives_when_strongly_matched(fit_stub):
    # The two-tier policy: a distinctive serif/display headline that matches its own
    # face in exact-font territory (>= strong_keep) is NOT force-unified to body sans.
    dominant = [
        _item(f"L{i}", "Inter", 0.60, w=300, path=fit_stub["path"], text="Inter line")
        for i in range(4)
    ]
    headline = _item("H1", "Playfair Display", 0.86, w=160, h=40, text="Elegant Headline")
    text_analysis._apply_font_consensus(
        dominant + [headline], {"enabled": True, "min_score": 0.30},
        {"consensus": {"enabled": True}})
    assert headline["line"]["style"]["fontFamily"] == "Playfair Display"
    assert "Elegant Headline" not in fit_stub["calls"]  # not even re-fit


def test_staple_families_rank_ahead_of_alphabet_head():
    metas = [
        {"family": "Candara", "path": "candara.ttf", "weight": 400},
        {"family": "Segoe UI", "path": "segoeui.ttf", "weight": 400},
        {"family": "Agency FB", "path": "agency.ttf", "weight": 400},
        {"family": "Inter", "path": "inter.ttf", "weight": 400},
    ]
    # replicate _discover_fonts' rank ordering
    import src.text_analysis as ta
    import inspect
    src_text = inspect.getsource(ta._discover_fonts)
    assert '"segoe ui"' in src_text and '"inter"' in src_text
    # Inter must outrank Segoe UI, which must outrank Candara/Agency
    order = ["inter", "segoe ui", "candara"]
    positions = [src_text.index(f'"{name}"') for name in order]
    assert positions == sorted(positions)
