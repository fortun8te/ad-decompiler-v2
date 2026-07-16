"""Deterministic backstops for high-visibility OCR misreads (ads 013 / 066 / 091).

The VLM judge is nondeterministic (it fixed 013's ``do1 this`` in one run and
missed it in the next when every call errored); these paths must hold without it.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import ocr  # noqa: E402


def _box(x, y, w, h):
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def _quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _line(text, conf, box, engine, words=None):
    return {
        "text": text,
        "conf": conf,
        "box": box,
        "quad": ocr._rect_quad(box),
        "words": words or [],
        "meta": {"engine": engine, "source_kind": "line"},
    }


# ---------------------------------------------------------------------------
# Ad 013: trailing '!' misread as '1'/'ı' and relocated


def test_exclamation_confusion_relocated_glyph_restores_bang():
    assert ocr._fix_exclamation_confusion("do1 this", ["do thisı"]) == "do this!"
    assert ocr._fix_exclamation_confusion("do thisı", ["do1 this"]) == "do this!"
    # Peer literally saw the '!' — prefer the punctuation reading.
    assert ocr._fix_exclamation_confusion("do this1", ["do this!"]) == "do this!"


def test_exclamation_confusion_conservative_negatives():
    # Same-position l/1 substitution: ambiguous letter vs punctuation — no fix.
    assert ocr._fix_exclamation_confusion("special deal", ["special dea1"]) is None
    # Identical readings never fire.
    assert ocr._fix_exclamation_confusion("buy it all", ["buy it all"]) is None
    # Non-alpha lines (prices, offers) are out of scope.
    assert ocr._fix_exclamation_confusion("SAVE 30%", ["SAVF 30%"]) is None
    # No peer evidence -> no fix.
    assert ocr._fix_exclamation_confusion("do1 this", []) is None


def test_reconcile_fixes_exclamation_confusion_deterministically():
    primary = _line("do1 this", 0.86, _box(216, 442, 654, 165), "doctr")
    challenger = _line("do thisı", 0.41, _box(184, 434, 716, 199), "easyocr")
    fused = ocr._reconcile([primary], [[challenger]], cfg={})
    assert len(fused) == 1
    assert fused[0]["text"] == "do this!"
    assert fused[0]["meta"]["exclamation_fix"]["from"] == "do1 this"
    # Judge candidates must include the corrected reading.
    assert "do this!" in fused[0]["meta"]["disagreement"]


# ---------------------------------------------------------------------------
# Ad 066: hallucinated isolated dash mid-headline


def test_reconcile_drops_stray_dash_token_with_weak_word_evidence():
    words = [
        {"text": tok, "conf": conf, "box": _box(10 + i * 40, 10, 35, 20),
         "quad": _quad(10 + i * 40, 10, 35, 20), "meta": {"engine": "doctr"}}
        for i, (tok, conf) in enumerate([
            ("MASCARAS", 0.95), ("SO", 0.99), ("YOU", 0.98), ("-", 0.4954),
            ("DON'T", 0.71), ("HAVE", 0.96), ("TO", 0.81),
        ])
    ]
    primary = _line("MASCARAS SO YOU - DON'T HAVE TO", 0.80,
                    _box(251, 163, 982, 42), "doctr", words)
    challenger = _line("MASCARAS SO YOU DON'T HAVE TO", 0.66,
                       _box(245, 153, 1000, 68), "easyocr")
    fused = ocr._reconcile([primary], [[challenger]], cfg={})
    assert fused[0]["text"] == "MASCARAS SO YOU DON'T HAVE TO"
    assert fused[0]["meta"]["stray_punct_dropped"]["token"] == "-"
    assert all(w["text"] != "-" for w in fused[0]["words"])


def test_reconcile_keeps_confident_dash_with_single_dissenting_peer():
    words = [
        {"text": tok, "conf": conf, "box": _box(10 + i * 40, 10, 35, 20),
         "quad": _quad(10 + i * 40, 10, 35, 20), "meta": {"engine": "doctr"}}
        for i, (tok, conf) in enumerate([("SALE", 0.95), ("-", 0.9), ("50%", 0.95)])
    ]
    primary = _line("SALE - 50%", 0.9, _box(10, 10, 200, 30), "doctr", words)
    challenger = _line("SALE 50%", 0.7, _box(10, 10, 200, 30), "easyocr")
    fused = ocr._reconcile([primary], [[challenger]], cfg={})
    assert fused[0]["text"] == "SALE - 50%"


# ---------------------------------------------------------------------------
# Ad 091: strike ink pollutes recognition ('A900A')


def _strike_line_image(tmp_path, strike=True, strike_color=(220, 30, 30)):
    """White canvas, blocky black 'glyphs', optional strike stroke across them."""
    path = tmp_path / ("strike.png" if strike else "plain.png")
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    for x in range(30, 330, 30):
        draw.rectangle([x, 20, x + 18, 60], fill=(10, 10, 10))
    if strike:
        draw.line([(20, 40), (340, 44)], fill=strike_color, width=5)
    img.save(path)
    return str(path)


def test_detect_strike_foreign_color_over_text(tmp_path):
    image = Image.open(_strike_line_image(tmp_path)).convert("RGB")
    detection = ocr._detect_strike(image)
    assert detection is not None
    mask, bbox = detection
    assert bbox["w"] >= 0.6 * image.width
    # Mask covers the strike band, not the whole glyph area.
    assert 0.0 < float(mask.mean()) < 0.3


def test_detect_strike_ignores_plain_text(tmp_path):
    image = Image.open(_strike_line_image(tmp_path, strike=False)).convert("RGB")
    assert ocr._detect_strike(image) is None


def test_fix_strikethrough_lines_prefers_peer_agreeing_with_reocr(tmp_path):
    image_path = _strike_line_image(tmp_path)
    line = {
        "id": "L0",
        "text": "A900A and Steady",
        "conf": 0.77,
        "box": _box(0, 0, 400, 80),
        "quad": _quad(0, 0, 400, 80),
        "words": [{"text": "A900A", "conf": 0.8, "box": _box(20, 20, 100, 40),
                   "quad": _quad(20, 20, 100, 40), "meta": {}}],
        "meta": {"provenance": [
            {"engine": "doctr", "text": "A900A and Steady",
             "calibrated_confidence": 0.7728, "selected": True},
            {"engine": "easyocr", "text": "Foggy and Steady",
             "calibrated_confidence": 0.7399, "selected": False},
        ]},
    }

    def _fake_runner(engine, path, cfg, use_cache=True):
        return {"engine": engine, "lines": [
            {"text": "Foggy and Steady", "conf": 0.9, "box": _box(0, 0, 380, 70)},
        ]}

    out = ocr._fix_strikethrough_lines(image_path, [line], {}, runner=_fake_runner)
    assert out[0]["text"] == "Foggy and Steady"
    assert out[0]["ocr_text"] == "A900A and Steady"
    assert out[0]["meta"]["strikethrough"] is True
    assert out[0]["meta"]["strikethrough_fix"]["method"] == "reocr-matches-peer"
    assert out[0]["meta"]["strikethrough_box"]["w"] > 0


def test_fix_strikethrough_falls_back_to_clean_peer_when_reocr_fails(tmp_path):
    image_path = _strike_line_image(tmp_path)
    line = {
        "id": "L0",
        "text": "A900A and Steady",
        "conf": 0.77,
        "box": _box(0, 0, 400, 80),
        "quad": _quad(0, 0, 400, 80),
        "words": [],
        "meta": {"provenance": [
            {"engine": "doctr", "text": "A900A and Steady",
             "calibrated_confidence": 0.7728},
            {"engine": "easyocr", "text": "Foggy and Steady",
             "calibrated_confidence": 0.7399},
        ]},
    }

    def _broken_runner(engine, path, cfg, use_cache=True):
        raise RuntimeError("engine unavailable")

    out = ocr._fix_strikethrough_lines(image_path, [line], {}, runner=_broken_runner)
    assert out[0]["text"] == "Foggy and Steady"
    assert out[0]["meta"]["strikethrough_fix"]["method"] == "clean-peer-preferred"


def test_fix_strikethrough_skips_undisputed_lines(tmp_path):
    image_path = _strike_line_image(tmp_path)
    line = {
        "id": "L0", "text": "Foggy and Steady", "conf": 0.9,
        "box": _box(0, 0, 400, 80), "quad": _quad(0, 0, 400, 80), "words": [],
        "meta": {"provenance": [
            {"engine": "doctr", "text": "Foggy and Steady"},
            {"engine": "easyocr", "text": "Foggy and Steady"},
        ]},
    }
    calls = []

    def _runner(engine, path, cfg, use_cache=True):
        calls.append(engine)
        return {"lines": []}

    out = ocr._fix_strikethrough_lines(image_path, [line], {}, runner=_runner)
    assert calls == []
    assert out[0]["text"] == "Foggy and Steady"
    assert "strikethrough" not in out[0]["meta"]
