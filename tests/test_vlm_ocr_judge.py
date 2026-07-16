"""CPU-only tests for optional VLM OCR disagreement judging."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src import vlm_client, vlm_ocr_judge  # noqa: E402


def _font_path():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if os.path.isfile(path):
            return path
    return path


def _image(tmp_path, size=(200, 100), texts=None):
    path = tmp_path / "ad.png"
    img = Image.new("RGB", size, "white")
    if texts:
        draw = ImageDraw.Draw(img)
        font_path = _font_path()
        font = ImageFont.truetype(font_path, 28) if os.path.isfile(font_path) else ImageFont.load_default()
        for text, xy in texts:
            draw.text(xy, text, fill=(20, 20, 20), font=font)
    img.save(path)
    return str(path)


def _line(text, conf=0.9, box=None, meta=None):
    return {
        "id": "L0",
        "text": text,
        "conf": conf,
        "box": box or {"x": 10, "y": 10, "w": 120, "h": 30},
        "meta": meta or {},
    }


def _cfg(enabled=True, ocr_read=False):
    judge = {"enabled": enabled, "passes": 2}
    if ocr_read:
        judge["ocr_read"] = {"enabled": True, "grid_cols": 1, "grid_rows": 1, "max_regions": 1}
    return {"vlm": {"ocr_judge": judge}}


def _mock_multi(monkeypatch, answers):
    it = iter(answers)

    def _multi(crop, prompt, **kwargs):
        return next(it), None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _multi)

def test_disabled_returns_input_unchanged(tmp_path):
    ocr_result = {"lines": [_line("SAVE 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]})]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, {})
    assert out is ocr_result


def test_no_disagreement_skips_vlm_call(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        vlm_client, "multi_pass_answer",
        lambda *a, **k: called.append(1) or ("x", None),
    )
    ocr_result = {"lines": [_line("SAVE 30%")]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert called == []
    assert "vlm_ocr_judge" in out
    assert out["vlm_ocr_judge"]["lines_checked"] == 0


def test_disagreement_corrected_when_passes_agree(tmp_path, monkeypatch):
    _mock_multi(monkeypatch, ["SAVE 30%"])
    ocr_result = {
        "lines": [_line(
            "SAVF 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]},
        )],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    line = out["lines"][0]
    assert line["text"] == "SAVE 30%"
    assert line["ocr_text"] == "SAVF 30%"
    assert line["vlm_ocr_judged"] is True
    assert "disagreement" not in line.get("meta", {})
    assert out["vlm_ocr_judge"]["lines_corrected"] == 1


def test_provenance_disagreement_triggers_judge(tmp_path, monkeypatch):
    _mock_multi(monkeypatch, ["€49"])
    ocr_result = {
        "lines": [_line("E49", meta={
            "provenance": [
                {"engine": "doctr", "text": "€49"},
                {"engine": "surya", "text": "E49"},
            ],
        })],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert out["lines"][0]["text"] == "€49"


def test_disagreement_prompt_lists_engine_readings(tmp_path, monkeypatch):
    prompts = []

    def _capture(crop, prompt, **kwargs):
        prompts.append(prompt)
        return "SAVE 30%", None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _capture)
    ocr_result = {
        "lines": [_line(
            "SAVF 30%",
            meta={"disagreement": ["SAVE 30%", "SAVF 30%", "SAVE 3O%"]},
        )],
    }
    vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert 'A: "SAVE 30%"' in prompts[0]
    assert 'B: "SAVF 30%"' in prompts[0]
    assert 'C: "SAVE 3O%"' in prompts[0]


def test_passes_disagree_leaves_original(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: (None, "vlm_disagreement"))
    ocr_result = {
        "lines": [_line("SAVF 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]})],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert out["lines"][0]["text"] == "SAVF 30%"
    assert out["vlm_ocr_judge"]["lines_disagreed"] == 1


def test_passes_disagree_prefers_spaced_case_split(tmp_path, monkeypatch):
    """Ad 013: VLM split on WeNEVER vs We NEVER — fall back to the spaced reading."""
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: (None, "vlm_disagreement"))
    ocr_result = {
        "lines": [_line("WeNEVER", meta={"disagreement": ["We NEVER", "WeNEVER"]})],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert out["lines"][0]["text"] == "We NEVER"
    assert out["lines"][0]["meta"]["vlm_ocr_fallback"]["to"] == "We NEVER"


def test_vlm_answer_collapses_repeated_tokens(tmp_path, monkeypatch):
    """Ad 013: VLM invented ``do do this!`` from ``do1 this`` — hygiene collapses it."""
    _mock_multi(monkeypatch, ["do do this!"])
    ocr_result = {
        "lines": [_line("do1 this", meta={"disagreement": ["do1 this", "do thisı"]})],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert out["lines"][0]["text"] == "do this!"


def test_vlm_error_degrades_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: (None, "vlm_error"))
    ocr_result = {
        "lines": [_line("SAVF 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]})],
    }
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert out["lines"][0]["text"] == "SAVF 30%"
    assert out["vlm_ocr_judge"]["lines_errored"] == 1
    assert out["vlm_ocr_judge"]["notes"][0]["note"] == "vlm_error"


def test_judge_resolves_disagreement_even_when_original_was_correct(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: ("SAVE 30%", None))
    out = vlm_ocr_judge.judge_ocr_lines(
        _image(tmp_path),
        {"lines": [_line("SAVE 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]})]},
        _cfg(),
    )
    line = out["lines"][0]
    assert "disagreement" not in line["meta"]
    assert line["vlm_ocr_judged"] is True
    assert line["meta"]["vlm_ocr_consensus"]["passes"] == 2


def test_judge_rejects_unrelated_third_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: ("BUY SHOES TODAY", None))
    out = vlm_ocr_judge.judge_ocr_lines(
        _image(tmp_path),
        {"lines": [_line("SAVE 30%", meta={"disagreement": ["SAVE 30%", "SAVF 30%"]})]},
        _cfg(),
    )
    assert out["lines"][0]["text"] == "SAVE 30%"
    assert out["vlm_ocr_judge"]["notes"][0]["note"] == "vlm_novel_reading"


def _cfg_proofread(max_conf=0.80, brand_tokens=True, min_similarity=0.6):
    return {"vlm": {"ocr_judge": {
        "enabled": True, "passes": 2,
        "proofread": {"enabled": True, "max_conf": max_conf,
                      "brand_tokens": brand_tokens, "min_similarity": min_similarity},
    }}}


def test_brand_token_single_char_misread_is_proofread(tmp_path, monkeypatch):
    # High-confidence, single-engine (no disagreement) brand-token misread: P->H.
    _mock_multi(monkeypatch, ["PINDAKAAS"])
    ocr_result = {"lines": [_line("HINDAKAAS", conf=0.95)]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg_proofread())
    line = out["lines"][0]
    assert line["text"] == "PINDAKAAS"
    assert line["ocr_text"] == "HINDAKAAS"
    assert line["vlm_ocr_judged"] is True
    assert line["meta"]["vlm_ocr_proofread"]["reason"] == "brand-token"
    assert out["vlm_ocr_judge"]["proofread_corrected"] == 1


def test_proofread_rejects_wholesale_rewrite(tmp_path, monkeypatch):
    # A dissimilar answer (hallucinated real word) must not replace the OCR reading.
    _mock_multi(monkeypatch, ["CHOCOLADE"])
    ocr_result = {"lines": [_line("HINDAKAAS", conf=0.95)]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg_proofread())
    assert out["lines"][0]["text"] == "HINDAKAAS"
    assert out["vlm_ocr_judge"]["proofread_corrected"] == 0
    assert out["vlm_ocr_judge"]["notes"][0]["note"] == "vlm_low_similarity"


def test_proofread_routes_low_confidence_line(tmp_path, monkeypatch):
    _mock_multi(monkeypatch, ["energie"])
    ocr_result = {"lines": [_line("energi", conf=0.55)]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg_proofread())
    line = out["lines"][0]
    assert line["text"] == "energie"
    assert line["meta"]["vlm_ocr_proofread"]["reason"] == "low-confidence"


def test_proofread_disabled_by_default(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(vlm_client, "multi_pass_answer",
                        lambda *a, **k: called.append(1) or ("PINDAKAAS", None))
    ocr_result = {"lines": [_line("HINDAKAAS", conf=0.95)]}
    out = vlm_ocr_judge.judge_ocr_lines(_image(tmp_path), ocr_result, _cfg())
    assert called == []  # no disagreement + proofread off => no VLM call
    assert out["lines"][0]["text"] == "HINDAKAAS"


def test_brand_token_heuristic():
    assert vlm_ocr_judge._looks_like_brand_token("PINDAKAAS")
    assert vlm_ocr_judge._looks_like_brand_token("UPFRONT")
    assert not vlm_ocr_judge._looks_like_brand_token("save today")  # lowercase body copy
    assert not vlm_ocr_judge._looks_like_brand_token("De Vakantiegeldsale komt eraan")
    assert not vlm_ocr_judge._looks_like_brand_token("AB")  # too short


def test_ocr_read_adds_missed_line(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vlm_client, "multi_pass_answer",
        lambda *a, **k: ("SALE", None),
    )
    img = _image(tmp_path, size=(200, 120), texts=[("SALE", (120, 70))])
    ocr_result = {"lines": []}
    out = vlm_ocr_judge.judge_ocr_lines(img, ocr_result, _cfg(ocr_read=True))
    assert out["vlm_ocr_judge"]["ocr_read_added"] == 1
    assert len(out["lines"]) == 1
    assert out["lines"][0]["text"] == "SALE"
    assert out["lines"][0].get("vlm_ocr_read") is True


def test_ocr_read_skipped_when_image_too_big(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        vlm_client, "multi_pass_answer",
        lambda *a, **k: called.append(1) or ("x", None),
    )
    img = _image(tmp_path, size=(3000, 3000))
    out = vlm_ocr_judge.judge_ocr_lines(
        img,
        {"lines": []},
        {
            "vlm": {
                "ocr_judge": {
                    "enabled": True,
                    "ocr_read": {
                        "enabled": True,
                        "max_image_pixels": 1000,
                    },
                },
            },
        },
    )
    assert called == []
    assert out["vlm_ocr_judge"]["ocr_read_added"] == 0


def test_ocr_read_skips_regions_overlapping_existing_lines(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _count(crop, prompt, **kwargs):
        calls["n"] += 1
        return "", None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _count)
    img = _image(tmp_path, size=(200, 200))
    ocr_result = {
        "lines": [_line("BIG", box={"x": 0, "y": 0, "w": 200, "h": 200})],
    }
    vlm_ocr_judge.judge_ocr_lines(img, ocr_result, _cfg(ocr_read=True))
    assert calls["n"] == 0


def test_disagreement_max_lines_caps_vlm_calls(tmp_path, monkeypatch):
    calls = []

    def _capture(crop, prompt, **kwargs):
        calls.append(prompt)
        return "OK", None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _capture)
    lines = [
        _line(
            f"body {i}",
            conf=0.9,
            box={"x": i * 5, "y": 50, "w": 20, "h": 10},
            meta={"disagreement": [f"body {i}", f"b0dy {i}"]},
        )
        for i in range(5)
    ]
    # Distinct ids so notes/assertions stay unambiguous.
    for i, line in enumerate(lines):
        line["id"] = f"B{i}"
    out = vlm_ocr_judge.judge_ocr_lines(
        _image(tmp_path),
        {"lines": lines},
        {"vlm": {"ocr_judge": {"enabled": True, "passes": 1, "max_lines": 2}}},
    )
    assert len(calls) == 2
    assert out["vlm_ocr_judge"]["lines_checked"] == 2
    assert out["vlm_ocr_judge"]["max_lines"] == 2


def test_disagreement_priority_prefers_brand_price_cta(tmp_path, monkeypatch):
    prompts = []

    def _capture(crop, prompt, **kwargs):
        prompts.append(prompt)
        # Echo a reading that stays close enough to pass the novel-reading gate.
        if "UPFRONT" in prompt:
            return "UPFRONT", None
        if "SAVE 30%" in prompt:
            return "SAVE 30%", None
        if "Shop now" in prompt:
            return "Shop now", None
        return "ingredients listed here", None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _capture)
    body = _line(
        "ingredients listed here",
        conf=0.5,
        box={"x": 0, "y": 80, "w": 400, "h": 40},  # large but low-impact
        meta={"disagreement": ["ingredients listed here", "ingredientz listed here"]},
    )
    body["id"] = "BODY"
    brand = _line(
        "UPFRONT",
        conf=0.9,
        box={"x": 10, "y": 10, "w": 80, "h": 24},
        meta={"disagreement": ["UPFRONT", "UPFR0NT"]},
    )
    brand["id"] = "BRAND"
    price = _line(
        "SAVE 30%",
        conf=0.9,
        box={"x": 10, "y": 40, "w": 90, "h": 20},
        meta={"disagreement": ["SAVE 30%", "SAVF 30%"]},
    )
    price["id"] = "PRICE"
    cta = _line(
        "Shop now",
        conf=0.9,
        box={"x": 10, "y": 60, "w": 70, "h": 18},
        meta={"disagreement": ["Shop now", "Shop naw"]},
    )
    cta["id"] = "CTA"
    # Body first in list order — without priority sort it would win the budget.
    out = vlm_ocr_judge.judge_ocr_lines(
        _image(tmp_path),
        {"lines": [body, brand, price, cta]},
        {"vlm": {"ocr_judge": {"enabled": True, "passes": 1, "max_lines": 3}}},
    )
    assert out["vlm_ocr_judge"]["lines_checked"] == 3
    joined = "\n".join(prompts)
    assert "UPFRONT" in joined
    assert "SAVE 30%" in joined
    assert "Shop now" in joined
    assert "ingredients listed here" not in joined
    by_id = {line["id"]: line for line in out["lines"]}
    assert by_id["BODY"].get("vlm_ocr_judged") is not True
    assert by_id["BRAND"].get("vlm_ocr_judged") is True
    assert by_id["PRICE"].get("vlm_ocr_judged") is True
    assert by_id["CTA"].get("vlm_ocr_judged") is True


def test_disagreement_skips_already_numeric_verified(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        vlm_client, "multi_pass_answer",
        lambda *a, **k: called.append(1) or ("99", None),
    )
    line = _line(
        "99",
        meta={"disagreement": ["66", "99"]},
    )
    line["vlm_ocr_judged"] = True  # numeric_verify already settled this
    out = vlm_ocr_judge.judge_ocr_lines(
        _image(tmp_path),
        {"lines": [line]},
        _cfg(),
    )
    assert called == []
    assert out["vlm_ocr_judge"]["lines_checked"] == 0


def test_high_impact_heuristics():
    assert vlm_ocr_judge._looks_like_price_or_offer("SAVE 30%")
    assert vlm_ocr_judge._looks_like_price_or_offer("€49")
    assert vlm_ocr_judge._looks_like_cta("Shop now")
    assert vlm_ocr_judge._looks_like_cta("Get up to")
    assert not vlm_ocr_judge._is_high_impact_text("ingredients listed here")
    # Brand still wins via existing heuristic.
    assert vlm_ocr_judge._is_high_impact_text("UPFRONT")


def test_resolve_options_defaults_max_lines():
    opts = vlm_ocr_judge._resolve_options({"vlm": {"ocr_judge": {"enabled": True}}})
    assert opts["max_lines"] == vlm_ocr_judge._DEFAULT_MAX_DISAGREE_LINES
    opts_null = vlm_ocr_judge._resolve_options(
        {"vlm": {"ocr_judge": {"enabled": True, "max_lines": None}}},
    )
    assert opts_null["max_lines"] == vlm_ocr_judge._DEFAULT_MAX_DISAGREE_LINES
    opts_zero = vlm_ocr_judge._resolve_options(
        {"vlm": {"ocr_judge": {"enabled": True, "max_lines": 0}}},
    )
    assert opts_zero["max_lines"] == 0
