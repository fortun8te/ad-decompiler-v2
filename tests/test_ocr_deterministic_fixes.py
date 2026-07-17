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


# ---------------------------------------------------------------------------
# Ad 131 (postfix-benchmark-7, text_recall 0.636 — worst of the run).
#
# Every corruption below shipped from ONE cluster where both readings were present
# and the wrong one won on calibrated confidence alone:
#
#   doctr   0.765  "BUY2. GETIFREE + FREE SHIPPINC +OOLS"    <- selected
#   easyocr 0.519  "BUY 2, GET 1 FREE + FREE SHIPPING $1OO+"  <- correct, discarded
#
# meta.disagreement was set, so the VLM judge should have arbitrated — but all 11
# judge calls errored that run (model evicted -> breaker opened), exactly as they
# did for 013's "do1 this". These paths must hold with the VLM absent.

_D131_DOCTR = "BUY2. GETIFREE + FREE SHIPPINC +OOLS"
_D131_EASY = "BUY 2, GET 1 FREE + FREE SHIPPING $1OO+"
_D131_TRUTH = "BUY 2, GET 1 FREE + FREE SHIPPING $100+"


def test_confusable_key_normalizes_display_glyph_confusion():
    # The two readings are the same ink: near-identical once G/C, 1/I and 0/O fold.
    assert ocr._confusable_similarity(_D131_DOCTR, _D131_EASY) >= 0.9
    # ...but very different as plain text, which is why plain similarity cannot gate.
    assert ocr._text_similarity(_D131_DOCTR, _D131_EASY) < 0.9
    assert ocr._confusable_key("SHIPPINC") == ocr._confusable_key("SHIPPING")
    # Letters outside a confusable class never collapse — a peer must not overwrite
    # a correct brand wordmark (067 'PROYK').
    assert ocr._confusable_key("PROYK") != ocr._confusable_key("PROVE")


def test_lexical_plausibility_separates_corrupt_from_clean_reading():
    assert ocr._lexical_plausibility(_D131_DOCTR) < 0.3
    assert ocr._lexical_plausibility(_D131_EASY) > 0.8


def test_corruption_signals_identify_131_damage():
    signals = ocr._corruption_signals(_D131_DOCTR, _D131_EASY)
    assert any(s.startswith("mixed-alnum:BUY2.") for s in signals)
    assert any(s.startswith("glued-run:GETIFREE") for s in signals)
    assert any(s.startswith("confusable-of-word:SHIPPINC") for s in signals)
    # A correct out-of-lexicon wordmark is 'unknown', never a corruption signal.
    assert ocr._corruption_signals("PROYK", "PROVE") == []


def test_reconcile_promotes_lexically_plausible_peer_over_confident_garbage():
    """131 L0 end-to-end: real boxes and confidences from bench-7 ocr.json."""
    primary = _line(_D131_DOCTR, 0.805, _box(122, 136.5, 1391.9, 67.4), "doctr")
    challenger = _line(_D131_EASY, 0.59, _box(113, 126, 1414, 90), "easyocr")
    fused = ocr._reconcile([primary], [[challenger]], cfg={"ocr": {}})
    assert len(fused) == 1
    assert fused[0]["text"] == _D131_EASY
    arb = fused[0]["meta"]["lexical_arbitration"]
    assert arb["from_engine"] == "doctr" and arb["to_engine"] == "easyocr"
    # 'from' is the winner as it stood at arbitration — the token-level lexicon
    # backstop has already repaired SHIPPINC by then; the glued/mixed damage that
    # confidence alone could not see is what the promotion is deciding on.
    assert "GETIFREE" in arb["from"] and "+OOLS" in arb["from"]
    assert arb["from_plausibility"] < arb["to_plausibility"]
    # Full stage output must equal the ad's actual copy, character for character.
    assert ocr.cleanup_line_text(fused[0]["text"]) == _D131_TRUTH
    # The raw engine readings stay on the record for the judge and for diagnostics.
    assert _D131_DOCTR in fused[0]["meta"]["disagreement"]


def test_lexical_arbitration_refuses_short_and_unrelated_readings():
    # Different lines in one cluster: not arbitration's call, whatever they score.
    members = [_line("SHOP THE SALE NOW", 0.9, _box(0, 0, 10, 10), "doctr"),
               _line("Free returns within 30 days", 0.5, _box(0, 0, 10, 10), "easyocr")]
    assert ocr._pick_lexical_peer(members, 0) is None
    # Winner is an out-of-lexicon wordmark, peer is 'more lexical' but wrong: no
    # corruption signal, so confidence keeps its win.
    members = [_line("froya Arctic Skincare", 0.9, _box(0, 0, 10, 10), "doctr"),
               _line("freya Arctic Skincare", 0.5, _box(0, 0, 10, 10), "easyocr")]
    assert ocr._pick_lexical_peer(members, 0) is None
    # Too short to gamble on.
    members = [_line("GETI FREE", 0.9, _box(0, 0, 10, 10), "doctr"),
               _line("GET 1 FREE", 0.5, _box(0, 0, 10, 10), "easyocr")]
    assert ocr._pick_lexical_peer(members, 0) is None


# ---------------------------------------------------------------------------
# G->C on display caps: the benchmark's most-repeated misread.
# 067 'SAYING'->'SAYINC' / 'GOODBYE'->'COODBYE', 131 'SHIPPING'->'SHIPPINC'.


def test_confusable_against_peers_fixes_g_to_c_on_display_caps():
    assert ocr._fix_confusable_against_peers(
        "FREE SHIPPINC", ["FREE SHIPPING"]) == "FREE SHIPPING"
    assert ocr._fix_confusable_against_peers(
        "WE'RE SAYINC COODBYE", ["WE'RE SAYING GOODBYE"]) == "WE'RE SAYING GOODBYE"


def test_confusable_against_peers_is_conservative():
    # No peer evidence -> no fix (the peer supplies the glyph choice, not a guess).
    assert ocr._fix_confusable_against_peers("FREE SHIPPINC", []) is None
    # Peer is not a real word either -> nothing authoritative to adopt.
    assert ocr._fix_confusable_against_peers("FREE SHIPPINC", ["FREE SHIPPINK"]) is None
    # Correct text is left exactly alone.
    assert ocr._fix_confusable_against_peers("FREE SHIPPING", ["FREE SHIPPING"]) is None
    # Non-confusable letter differences are real differences: never adopt the peer.
    assert ocr._fix_confusable_against_peers("PROYK", ["PROVE"]) is None


def test_reconcile_applies_confusable_fix_without_whole_line_promotion():
    """Token-local G->C rescue on a line the whole-reading promotion declines."""
    primary = _line("WE'RE SAYINC COODBYE", 0.9, _box(0, 0, 900, 60), "doctr")
    challenger = _line("WE'RE SAYING GOODBYE", 0.5, _box(0, 0, 900, 60), "easyocr")
    fused = ocr._reconcile([primary], [[challenger]], cfg={"ocr": {}})
    assert fused[0]["text"] == "WE'RE SAYING GOODBYE"


# ---------------------------------------------------------------------------
# G->C where BOTH engines agree on the misread, so there is no peer to learn from:
# 067 'COODBYE', 131's bottom marquee reading 'SHIPPINC' twice. The VLM judge was
# the only thing catching these, and on 131 it errored on all 11 lines.


def test_confusable_against_lexicon_needs_no_peer():
    assert ocr.cleanup_line_text("WERE SAYINC COODBYE") == "WERE SAYING GOODBYE"
    # 131's marquee, verbatim from bench-7 ocr.json.
    assert ocr.cleanup_line_text(
        "SHIPPINC $100+ BUY 2 GET FREE · FREE SHIPPINC $100+"
    ) == "SHIPPING $100+ BUY 2 GET FREE · FREE SHIPPING $100+"


def test_lexicon_rescue_never_rewrites_wordmarks_or_punctuated_tokens():
    # Brand wordmarks (067) have no lexicon skeleton and must survive untouched.
    for text in ["PROYK", "FROYK", "AS", "III", "$100+", "B2B",
                 "COMPREHENSIVE NUTRITION", "ONLINE EXCLUSIVE OFFER ENDING SOON"]:
        assert ocr.cleanup_line_text(text) == text, text
    # Purely-alphabetic gate: an apostrophe must never be dropped chasing a skeleton.
    assert ocr._fix_confusable_against_lexicon("DON'T") is None
    # Lowercase body copy is out of scope (the confusion is a display-caps artifact).
    assert ocr._fix_confusable_against_lexicon("free shippinc") is None
    # A real misread with no unique lexicon skeleton is left for a human.
    assert ocr._fix_confusable_against_lexicon("ENDS TODAV") is None


def test_word_tokens_are_cleaned_in_step_with_their_line():
    """131 shipped a line reading 'FREE SHIPPING' whose word_geometry said 'SHIPPINC'.

    build_design_json paints from word_geometry, so a stale token puts the misread
    straight back into the Figma output the line text had just been cleared of.
    """
    line = {
        "text": "SHIPPINC $1OO+",
        "conf": 0.9,
        "box": _box(0, 0, 400, 40),
        "words": [
            {"text": "SHIPPINC", "conf": 0.8, "box": _box(0, 0, 200, 40)},
            {"text": "$1OO+", "conf": 0.8, "box": _box(200, 0, 200, 40)},
        ],
    }
    out = ocr._apply_line_text_cleanup([line])[0]
    assert out["text"] == "SHIPPING $100+"
    assert [w["text"] for w in out["words"]] == ["SHIPPING", "$100+"]


def test_cleanup_word_text_never_splits_a_token():
    # A word token owns exactly one box, so a space-inserting fix must not run here
    # even though the same string would gain a space as a line.
    assert ocr.cleanup_line_text("BLACKFRIDAY") == "BLACK FRIDAY"
    assert ocr.cleanup_word_text("BLACKFRIDAY") == "BLACKFRIDAY"
    assert " " not in ocr.cleanup_word_text("WeNEVER")


def test_lexicon_skeletons_are_unambiguous():
    """Every lexicon word must own a unique confusable skeleton.

    A collision would make the 'exactly one match' rule silently coin-flip between
    two real words; assert the invariant instead of trusting it.
    """
    index = ocr._lexicon_by_confusable_key()
    collisions = {k: sorted(v) for k, v in index.items() if len(v) > 1}
    assert collisions == {}, collisions
    assert len(index) == len(ocr._LEXICON)


# ---------------------------------------------------------------------------
# Ad 131: '$100+' read as '$1OO+' by BOTH engines — no peer to arbitrate against.


def test_digit_run_letters_restore_price():
    assert ocr.cleanup_line_text("FREE SHIPPING $1OO+") == "FREE SHIPPING $100+"
    assert ocr.cleanup_line_text("Save $5O today") == "Save $50 today"
    assert ocr._fix_digit_run_letters("1OO%") == "100%"


def test_digit_run_letters_never_touch_real_alphanumerics():
    # Every one of these must survive: the token must become ALL digits to convert.
    for text in ["6g fiber", "5G ready", "B2B pricing", "3D render", "1080p video",
                 "100ri0", "10ar/30", "SIZE10", "COVID19", "45%", "$100+"]:
        assert ocr.cleanup_line_text(text) == text, text


# ---------------------------------------------------------------------------
# Ad 131: 'BLACK FRIDAY' glued by both engines (doctr 'BLACKFRIDAYSALE',
# easyocr 'BLACKFRIDAY SALE') — no peer evidence, so a curated phrase map.


def test_glued_display_phrase_regains_space():
    assert ocr.cleanup_line_text("BLACKFRIDAY SALE") == "BLACK FRIDAY SALE"
    assert ocr.cleanup_line_text("Blackfriday Sale") == "Black Friday Sale"
    assert ocr.cleanup_line_text("BLACK FRIDAY SALE") == "BLACK FRIDAY SALE"


def test_glued_phrase_map_does_not_split_authored_compounds():
    # A generic 'split when both halves are words' rule would wreck these.
    for text in ["Superfoods", "COMPREHENSIVE NUTRITION", "Greens"]:
        assert ocr.cleanup_line_text(text) == text, text


# ---------------------------------------------------------------------------
# Ad 016: sub-glyph detector noise shipped as real text lines and polluted recall.


def test_subglyph_noise_lines_are_suppressed():
    for text in ["-", "- -", "- 6", ".", "  ", "|", "- - -"]:
        assert ocr._is_glyphless_noise(text) is True, text


def test_subglyph_suppression_keeps_real_short_lines():
    # Real 016/131 lines that must survive: any letter, any multi-digit number, or
    # a lone digit with no stray marks.
    for text in ["Off", "45%", "8", "$100+", "III", "6g", "a", "50"]:
        assert ocr._is_glyphless_noise(text) is False, text


def test_drop_glyphless_lines_reports_what_it_removed():
    lines = [{"text": "Get up to"}, {"text": "- -"}, {"text": "45%"}, {"text": "."}]
    kept, dropped = ocr._drop_glyphless_lines(lines)
    assert [line["text"] for line in kept] == ["Get up to", "45%"]
    assert dropped == ["- -", "."]
