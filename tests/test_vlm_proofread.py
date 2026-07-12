"""CPU-only tests for the optional VLM OCR-proofreading pass."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image  # noqa: E402

from src import vlm_proofread  # noqa: E402


def _image(tmp_path):
    path = tmp_path / "ad.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return str(path)


def _line(text, conf, box=None):
    return {"id": "L0", "text": text, "conf": conf,
            "box": box or {"x": 10, "y": 10, "w": 80, "h": 20}}


def test_disabled_by_default_returns_input_unchanged(tmp_path):
    ocr_result = {"lines": [_line("UPERONT", 0.3)]}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, {})
    assert out is ocr_result


def test_no_low_confidence_lines_skips_vlm_call(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: called.append(1) or "x")
    ocr_result = {"lines": [_line("UPFRONT", 0.98)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert called == []
    assert out["lines"][0]["text"] == "UPFRONT"


def test_low_confidence_line_gets_corrected(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "UPFRONT")
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    line = out["lines"][0]
    assert line["text"] == "UPFRONT"
    assert line["ocr_text"] == "UPERONT"
    assert line["vlm_corrected"] is True
    assert out["vlm_proofread"]["lines_corrected"] == 1


def test_vlm_agreeing_with_ocr_leaves_line_unmarked(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "UPERONT")
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    line = out["lines"][0]
    assert line["text"] == "UPERONT"
    assert "vlm_corrected" not in line


def test_implausible_answer_is_rejected(tmp_path, monkeypatch):
    # A VLM answer many times longer than the source line reads as a truncated
    # reasoning trace, not a real transcription -- must not overwrite the OCR text.
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "x" * 300)
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert out["lines"][0]["text"] == "UPERONT"


def test_multiline_answer_is_rejected(tmp_path, monkeypatch):
    # A newline means the model read past this line's crop into a neighboring line --
    # never a legitimate transcription of a single OCR line.
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "vezels\neiwitten")
    ocr_result = {"lines": [_line("eiwitten", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert out["lines"][0]["text"] == "eiwitten"


def test_empty_answer_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "")
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert out["lines"][0]["text"] == "UPERONT"


def test_vlm_error_degrades_silently(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("LM Studio not running")
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", _boom)
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert out["lines"][0]["text"] == "UPERONT"
    assert out["vlm_proofread"]["lines_corrected"] == 0


def test_missing_image_degrades_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: "UPFRONT")
    ocr_result = {"lines": [_line("UPERONT", 0.5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85}}
    out = vlm_proofread.proofread_lines(str(tmp_path / "missing.png"), ocr_result, cfg)
    assert out is ocr_result


def test_max_lines_caps_vlm_calls(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vlm_proofread, "_ask_vlm", lambda *a, **k: calls.append(1) or "fixed")
    ocr_result = {"lines": [_line(f"bad{i}", 0.1, {"x": i * 10, "y": 0, "w": 5, "h": 5})
                             for i in range(5)]}
    cfg = {"vlm": {"enabled": True, "confidence_threshold": 0.85, "max_lines": 2}}
    vlm_proofread.proofread_lines(_image(tmp_path), ocr_result, cfg)
    assert len(calls) == 2
