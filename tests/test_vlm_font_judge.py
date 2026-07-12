"""CPU-only tests for optional VLM visual font judging."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src import vlm_client, vlm_font_judge  # noqa: E402


def _font_path():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if os.path.isfile(path):
            return path
    return None


def _image_with_text(tmp_path, text="SALE"):
    path = tmp_path / "ad.png"
    img = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(img)
    font_path = _font_path()
    font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.load_default()
    bbox = draw.textbbox((20, 16), text, font=font)
    draw.text((20, 16), text, fill=(20, 20, 20), font=font)
    img.save(path)
    box = {
        "x": float(max(0, bbox[0] - 4)),
        "y": float(max(0, bbox[1] - 4)),
        "w": float(bbox[2] - bbox[0] + 8),
        "h": float(bbox[3] - bbox[1] + 8),
    }
    return str(path), box


def _ocr_line(line_id, text, box, candidates):
    return {
        "id": line_id,
        "text": text,
        "conf": 0.95,
        "ink_confidence": 0.9,
        "style_id": "ST0",
        "box": box,
        "painted_box": dict(box),
        "style": {
            "fontFamily": candidates[0]["family"],
            "fontSize": 36,
            "fontWeight": 400,
            "fontStyle": "Regular",
            "color": "#141414",
            "colorRGB": [20, 20, 20],
            "fontCandidates": candidates,
        },
    }


def _candidates():
    path = _font_path() or "arial.ttf"
    return [
        {"family": "Arial", "style": "Regular", "weight": 400, "score": 0.82,
         "source": "local-render", "path": path},
        {"family": "ArialAlt", "style": "Regular", "weight": 400, "score": 0.71,
         "source": "local-render", "path": path},
        {"family": "Inter", "style": "Regular", "weight": 400, "score": 0.55,
         "source": "fallback"},
    ]


def _cfg(enabled=True):
    return {
        "vlm": {
            "font_judge": {
                "enabled": enabled,
                "score_threshold": 7,
                "max_candidates": 3,
                "passes": 2,
            },
        },
        "text_analysis": {"font_matching": {"enabled": True}},
    }


def _mock_vlm(monkeypatch, answers):
    it = iter(answers)

    def _multi(crop, prompt, **kwargs):
        return next(it), None

    monkeypatch.setattr(vlm_client, "multi_pass_answer", _multi)


def test_disabled_returns_input_unchanged(tmp_path):
    img, box = _image_with_text(tmp_path)
    ocr = {"lines": [_ocr_line("L0", "SALE", box, _candidates())]}
    out = vlm_font_judge.judge_fonts(img, ocr, {})
    assert out is ocr
    assert "vlm_font_judge" not in out


def test_font_matching_disabled_skips(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: called.append(1) or (None, None))
    img, box = _image_with_text(tmp_path)
    ocr = {"lines": [_ocr_line("L0", "SALE", box, _candidates())]}
    cfg = {"vlm": {"font_judge": {"enabled": True}}, "text_analysis": {"font_matching": {"enabled": False}}}
    out = vlm_font_judge.judge_fonts(img, ocr, cfg)
    assert called == []
    assert out is ocr


def test_single_candidate_promotes_when_passes_agree(tmp_path, monkeypatch):
    _mock_vlm(monkeypatch, [json.dumps({"score": 8, "reject": False})])
    img, box = _image_with_text(tmp_path)
    cands = _candidates()[:1]
    ocr = {"lines": [_ocr_line("L0", "SALE", box, cands)]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    line = out["lines"][0]
    assert line["vlm_font_judged"] is True
    assert line["style"]["fontCandidates"][0]["vlm_score"] == 8.0
    assert out["vlm_font_judge"]["clusters_updated"] == 1


def test_compare_mode_picks_candidate_b(tmp_path, monkeypatch):
    _mock_vlm(monkeypatch, [json.dumps({"choice": "B", "score": 9, "reject": False})])
    img, box = _image_with_text(tmp_path)
    cands = _candidates()[:2]
    ocr = {"lines": [_ocr_line("L0", "SALE", box, cands)]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    winner = out["lines"][0]["style"]["fontCandidates"][0]
    assert winner["family"] == "ArialAlt"
    assert out["vlm_font_judge"]["notes"][0]["method"] == "compare"


def test_below_threshold_tries_next_candidate(tmp_path, monkeypatch):
    _mock_vlm(monkeypatch, [
        json.dumps({"choice": "A", "score": 4, "reject": False}),
        json.dumps({"score": 4, "reject": False}),
        json.dumps({"score": 8, "reject": False}),
    ])
    img, box = _image_with_text(tmp_path)
    cands = _candidates()[:2]
    ocr = {"lines": [_ocr_line("L0", "SALE", box, cands)]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    assert out["lines"][0]["style"]["fontCandidates"][0]["family"] == "ArialAlt"
    assert out["vlm_font_judge"]["notes"][0]["attempts"] == 2


def test_disagreement_leaves_ranking_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: (None, "vlm_disagreement"))
    img, box = _image_with_text(tmp_path)
    cands = _candidates()[:1]
    ocr = {"lines": [_ocr_line("L0", "SALE", box, cands)]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    line = out["lines"][0]
    assert "vlm_font_judged" not in line
    assert line["style"]["fontCandidates"][0]["family"] == "Arial"
    assert out["vlm_font_judge"]["clusters_disagreed"] == 1


def test_illegible_reject_skips_update(tmp_path, monkeypatch):
    _mock_vlm(monkeypatch, [json.dumps({"reject": True})])
    img, box = _image_with_text(tmp_path)
    ocr = {"lines": [_ocr_line("L0", "SALE", box, _candidates()[:1])]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    assert "vlm_font_judged" not in out["lines"][0]
    assert out["vlm_font_judge"]["clusters_rejected"] == 1


def test_vlm_error_degrades_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_client, "multi_pass_answer", lambda *a, **k: (None, "vlm_error"))
    img, box = _image_with_text(tmp_path)
    ocr = {"lines": [_ocr_line("L0", "SALE", box, _candidates()[:1])]}
    out = vlm_font_judge.judge_fonts(img, ocr, _cfg())
    assert out["vlm_font_judge"]["clusters_errored"] == 1
    assert "vlm_font_judged" not in out["lines"][0]
