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
