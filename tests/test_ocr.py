"""Fixture-driven CPU tests for OCR API parsing, retry, and reconciliation."""
from __future__ import annotations

from types import SimpleNamespace
import os
import sys

import pytest

pytest.importorskip("PIL")
np = pytest.importorskip("numpy")
from PIL import Image

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


def test_parse_paddle_v3_result_object_and_aligned_arrays():
    fixture = SimpleNamespace(json={
        "res": {
            "rec_texts": ["SAVE 30%", "Today only"],
            "rec_scores": [0.93, 0.81],
            "rec_polys": [
                [[10, 20], [150, 20], [150, 50], [10, 50]],
                [[12, 62], [132, 60], [133, 84], [13, 86]],
            ],
            "rec_boxes": [[10, 20, 150, 50], [12, 60, 133, 86]],
            "dt_scores": [0.96, 0.89],
            "textline_orientation_angles": [0, 0],
        }
    })

    lines = ocr._parse_paddle_result([fixture])

    assert [line["text"] for line in lines] == ["SAVE 30%", "Today only"]
    assert lines[0]["box"] == _box(10, 20, 140, 30)
    assert lines[1]["quad"][1] == [132.0, 60.0]
    assert lines[0]["meta"]["backend_api"] == "paddle-v3-predict"
    assert lines[0]["meta"]["detection_confidence"] == pytest.approx(0.96)
    assert lines[0]["words"] == []


def test_parse_paddle_v3_accepts_numpy_geometry_without_truthiness_errors():
    fixture = SimpleNamespace(json={
        "res": {
            "rec_texts": ["Array geometry"],
            "rec_scores": np.asarray([0.9], dtype=np.float32),
            "rec_polys": [np.asarray(_quad(4, 6, 80, 20), dtype=np.int16)],
            "rec_boxes": np.asarray([[4, 6, 84, 26]], dtype=np.int16),
        }
    })

    lines = ocr._parse_paddle_result([fixture])

    assert len(lines) == 1
    assert lines[0]["box"] == _box(4, 6, 80, 20)


def test_parse_paddle_legacy_without_promoting_nested_items_twice():
    legacy = [[
        [_quad(5, 8, 90, 22), ("Legacy one", 0.88)],
        [_quad(6, 40, 110, 24), ("Legacy two", 0.91)],
    ]]

    lines = ocr._parse_paddle_result(legacy)

    assert [line["text"] for line in lines] == ["Legacy one", "Legacy two"]
    assert all(line["meta"]["backend_api"] == "paddle-legacy-ocr" for line in lines)


def test_surya_v2_blocks_and_v1_text_lines_are_tolerated():
    v2 = SimpleNamespace(blocks=[
        SimpleNamespace(
            html="<p>Hello <strong>world</strong></p>", confidence=0.86,
            polygon=_quad(10, 20, 140, 35), bbox=[10, 20, 150, 55],
            label="Text", raw_label="text", reading_order=0, skipped=False, error=False,
        ),
        SimpleNamespace(html="", confidence=0.1, polygon=_quad(0, 0, 10, 10),
                        skipped=True, error=False),
    ])
    v1 = SimpleNamespace(text_lines=[
        SimpleNamespace(text="Old API", confidence=0.77, bbox=[20, 80, 120, 104])
    ])

    lines = ocr._parse_surya_predictions([v2, v1])

    assert [line["text"] for line in lines] == ["Hello world", "Old API"]
    assert lines[0]["meta"]["source_kind"] == "block"
    assert lines[1]["meta"]["backend_api"] == "surya-v1-text-lines"


def test_doctr_preserves_word_and_rotated_quad_geometry():
    word1 = SimpleNamespace(
        value="Hello", confidence=0.91,
        geometry=[[0.10, 0.20], [0.30, 0.19], [0.31, 0.27], [0.11, 0.28]],
        objectness_score=0.95, crop_orientation={"value": 2.0, "confidence": 0.8},
    )
    word2 = SimpleNamespace(
        value="there", confidence=0.87,
        geometry=((0.32, 0.20), (0.50, 0.28)),
        objectness_score=0.93, crop_orientation=None,
    )
    source_line = SimpleNamespace(words=[word1, word2], geometry=None, objectness_score=0.9)
    document = SimpleNamespace(pages=[
        SimpleNamespace(dimensions=(500, 1000), blocks=[SimpleNamespace(lines=[source_line])])
    ])

    lines = ocr._parse_doctr_document(document)

    assert len(lines) == 1
    assert lines[0]["text"] == "Hello there"
    assert len(lines[0]["words"]) == 2
    assert lines[0]["words"][0]["quad"][1] == [300.0, 95.0]
    assert lines[0]["words"][1]["box"] == _box(320, 100, 180, 40)
    assert lines[0]["meta"]["objectness_score"] == pytest.approx(0.9)


def test_reconcile_uses_engine_calibration_and_exact_text_support():
    word = {"text": "SAVE", "conf": 0.79, "box": _box(10, 10, 55, 20),
            "quad": _quad(10, 10, 55, 20), "meta": {"engine": "ppocr-v6"}}
    paddle = _line("SAVE 30%", 0.78, _box(10, 10, 140, 28), "ppocr-v6", [word])
    surya = _line("SAVE 30%", 0.79, _box(9, 9, 142, 30), "surya")
    tesseract = _line("SAVF 30%", 0.93, _box(11, 10, 139, 28), "tesseract")

    fused = ocr._reconcile([paddle], [[surya], [tesseract]], cfg={})

    assert len(fused) == 1
    assert fused[0]["text"] == "SAVE 30%"
    assert fused[0]["words"][0]["text"] == "SAVE"
    assert fused[0]["meta"]["support_engines"] == ["ppocr-v6", "surya"]
    assert len(fused[0]["meta"]["provenance"]) == 3
    assert "disagreement" in fused[0]["meta"]
    consensus = fused[0]["meta"]["consensus"]
    assert consensus["engine_count"] == 3
    assert consensus["support_count"] == 2
    assert consensus["dissent_count"] == 1
    assert 0 < consensus["confidence"] < fused[0]["conf"]


def test_targeted_retry_collapses_word_fragments_into_one_line(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (400, 180), "white").save(image_path)
    original = _line("BUV NOW", 0.42, _box(100, 60, 120, 18), "ppocr-v6")
    calls = []

    def runner(engine, crop_path, cfg, use_cache=False):
        calls.append((engine, use_cache, Image.open(crop_path).size))
        return {
            "engine": engine,
            "lines": [
                _line("BUY", 0.96, _box(14, 14, 62, 34), engine),
                _line("NOW", 0.95, _box(82, 14, 72, 34), engine),
            ],
        }

    result = ocr._targeted_retry(
        str(image_path), [original], "ppocr-v6",
        {"ocr": {"retry_2x": {"enabled": True, "scale": 2, "max_regions": 1}}},
        runner=runner,
    )

    assert len(result) == 1
    assert result[0]["text"] == "BUY NOW"
    assert result[0]["box"] == original["box"]  # full-image detector keeps placement
    assert len(result[0]["words"]) == 2
    assert result[0]["words"][0]["box"]["x"] > 90
    assert result[0]["meta"]["retry_2x"]["selected"] is True
    assert calls[0][0:2] == ("ppocr-v6", False)


def test_targeted_retry_is_bounded_and_skips_large_confident_lines(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (500, 260), "white").save(image_path)
    lines = [
        _line("Good large text", 0.98, _box(20, 20, 300, 60), "ppocr-v6"),
        _line("small one", 0.60, _box(20, 110, 100, 16), "ppocr-v6"),
        _line("small two", 0.55, _box(150, 110, 100, 16), "ppocr-v6"),
    ]
    calls = []

    def runner(engine, crop_path, cfg, use_cache=False):
        calls.append(crop_path)
        return {"engine": engine, "lines": []}

    result = ocr._targeted_retry(
        str(image_path), lines, "ppocr-v6",
        {"ocr": {"retry_2x": {"enabled": True, "max_regions": 1}}},
        runner=runner,
    )

    assert len(result) == 3
    assert len(calls) == 1
    assert "retry_2x" not in result[0]["meta"]


def test_ordering_keeps_words_inside_parent_line():
    word = {"text": "inside", "conf": 0.9, "box": _box(30, 20, 45, 12),
            "quad": _quad(30, 20, 45, 12), "meta": {"engine": "fixture"}}
    first = _line("Parent line", 0.9, _box(20, 18, 120, 24), "fixture", [word])
    second = _line("Second line", 0.9, _box(20, 60, 120, 24), "fixture")

    ordered = ocr._order_lines([second, first])

    assert len(ordered) == 2
    assert ordered[0]["words"] == [word]
    assert [line["id"] for line in ordered] == ["L0", "L1"]


def test_ordering_reads_stable_two_column_copy_column_major():
    # A regular y/x pass interleaves the two paragraphs.  The column-aware pass
    # must keep each multi-line copy column intact while retaining the spanning title.
    title = _line("THE ROUTINE", 0.98, _box(20, 8, 280, 30), "fixture")
    left1 = _line("Left line one", 0.96, _box(20, 60, 130, 18), "fixture")
    right1 = _line("Right line one", 0.96, _box(260, 60, 130, 18), "fixture")
    left2 = _line("Left line two", 0.96, _box(20, 86, 130, 18), "fixture")
    right2 = _line("Right line two", 0.96, _box(260, 86, 130, 18), "fixture")

    ordered = ocr._order_lines([right2, left2, title, right1, left1])

    assert [line["text"] for line in ordered] == [
        "THE ROUTINE", "Left line one", "Left line two", "Right line one", "Right line two",
    ]


def test_recombine_fragments_repairs_tight_same_row_split_but_not_adjacent_labels():
    split = [
        _line("LIMITED", 0.91, _box(20, 40, 70, 20), "ppocr-v6"),
        _line("EDITION", 0.94, _box(96, 40, 68, 20), "ppocr-v6"),
    ]
    result = ocr._recombine_fragments(split, {"ocr": {"recombine_fragments": True}})

    assert len(result) == 1
    assert result[0]["text"] == "LIMITED EDITION"
    assert result[0]["meta"]["source_kind"] == "recombined-fragments"
    assert len(result[0]["meta"]["fragments"]) == 2

    distinct = [
        _line("SAVE 20%.", 0.98, _box(20, 40, 90, 20), "ppocr-v6"),
        _line("SHOP", 0.98, _box(116, 40, 52, 20), "ppocr-v6"),
    ]
    untouched = ocr._recombine_fragments(distinct, {"ocr": {"recombine_fragments": True}})
    assert [line["text"] for line in untouched] == ["SAVE 20%.", "SHOP"]


def test_primary_failure_uses_challenger_but_exposes_partial_status(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 80), "white").save(image_path)

    def fake_backend(name, path, cfg, use_cache=True):
        if name == "ppocr-v6":
            raise ImportError("Paddle GPU wheel unavailable")
        return {"engine": "surya", "lines": [
            _line("SALE", 0.99, _box(12, 15, 60, 20), "surya"),
        ]}

    monkeypatch.setattr(ocr, "_run_backend", fake_backend)
    result = ocr.run_ocr(
        str(image_path),
        {"ocr": {"primary": "ppocr-v6", "challengers": ["surya"], "retry_2x": False}},
    )

    assert result["status"] == "partial"
    assert result["engine"] == "surya"
    assert result["errors"][0]["engine"] == "ppocr-v6"
    assert result["lines"][0]["text"] == "SALE"


def test_empty_configured_challenger_is_fail_closed(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 80), "white").save(image_path)

    def fake_backend(name, path, cfg, use_cache=True):
        if name == "surya":
            return {"engine": name, "lines": []}
        return {"engine": name, "lines": [_line("SALE", 0.98, _box(10, 10, 50, 20), name)]}

    monkeypatch.setattr(ocr, "_run_backend", fake_backend)
    result = ocr.run_ocr(str(image_path), {
        "ocr": {"primary": "ppocr-v6", "challengers": ["surya"], "retry_2x": False}
    })

    assert result["status"] == "partial"
    assert result["metrics"]["cross_check"]["fail_closed"] is True
    assert result["metrics"]["cross_check"]["missing_engines"] == ["surya"]


def test_ocr_metrics_report_geometry_and_cross_check_health():
    lines = [_line("OK", 0.9, _box(1, 2, 30, 10), "primary")]
    assert ocr._geometry_metrics(lines) == {
        "lines": 1, "valid_lines": 1, "invalid_lines": 0,
        "missing_quad": 0, "valid": True,
    }
    metrics = ocr._cross_check_metrics(lines, ["primary", "challenger"], ["primary"])
    assert metrics["complete"] is False
    assert metrics["fail_closed"] is True


def test_paddle_gpu_failure_retries_cpu_once(monkeypatch):
    calls = []

    class _Engine:
        def predict(self, path):
            return []

    def fake_engine(cfg, *, device_override=None):
        device = device_override or str(cfg.get("device", "cpu"))
        calls.append(device)
        if device.startswith("cuda") or device == "gpu":
            raise OSError("[WinError 127] cudnn_cnn64_9.dll not found")
        return _Engine(), "v3"

    monkeypatch.setattr(ocr, "_paddle_engine", fake_engine)
    monkeypatch.setattr(ocr, "_parse_paddle_result", lambda result: [
        _line("CPU OK", 0.9, _box(1, 2, 30, 10), "ppocr-v6"),
    ])

    lines, engine = ocr._paddle("fake.png", {"device": "cuda"})

    assert engine == "ppocr-v6"
    assert lines[0]["text"] == "CPU OK"
    assert calls == ["cuda", "cpu"]


def test_run_ocr_uses_tesseract_fallback_when_all_configured_fail(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 80), "white").save(image_path)
    calls = []

    def fake_backend(name, path, cfg, use_cache=True):
        calls.append(name)
        if name in ("ppocr-v6", "surya"):
            raise RuntimeError(f"{name} unavailable")
        return {"engine": "tesseract", "lines": [
            _line("FALLBACK", 0.7, _box(5, 5, 40, 12), "tesseract"),
        ]}

    monkeypatch.setattr(ocr, "_run_backend", fake_backend)
    monkeypatch.setattr(ocr, "_fallback_engine_names", lambda cfg: ["tesseract"])
    result = ocr.run_ocr(
        str(image_path),
        {"ocr": {"primary": "ppocr-v6", "challengers": ["surya"], "retry_2x": False}},
    )

    assert "tesseract" in calls
    assert result["engine"] == "tesseract"
    assert result["lines"][0]["text"] == "FALLBACK"
    assert any(item.get("role") == "fallback" for item in result["errors"])


def test_run_ocr_uses_fallback_when_configured_engines_silently_return_empty(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 80), "white").save(image_path)
    calls = []

    def fake_backend(name, path, cfg, use_cache=True):
        calls.append(name)
        if name == "tesseract":
            return {"engine": name, "lines": [
                _line("RECOVERED", 0.75, _box(5, 5, 50, 12), name),
            ]}
        return {"engine": name, "lines": []}

    monkeypatch.setattr(ocr, "_run_backend", fake_backend)
    monkeypatch.setattr(ocr, "_fallback_engine_names", lambda cfg: ["tesseract"])
    result = ocr.run_ocr(
        str(image_path),
        {"ocr": {"primary": "ppocr-v6", "challengers": ["surya"], "retry_2x": False}},
    )

    assert calls == ["ppocr-v6", "surya", "tesseract"]
    assert result["engine"] == "tesseract"
    assert result["lines"][0]["text"] == "RECOVERED"
    assert result["status"] == "partial"


def test_run_ocr_empty_evidence_is_not_reported_healthy(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 80), "white").save(image_path)
    monkeypatch.setattr(
        ocr, "_run_backend", lambda name, path, cfg, use_cache=True: {"engine": name, "lines": []}
    )
    monkeypatch.setattr(ocr, "_fallback_engine_names", lambda cfg: [])

    result = ocr.run_ocr(str(image_path), {"ocr": {"primary": "ppocr-v6"}})

    assert result["lines"] == []
    assert result["status"] == "partial"
    assert any(item.get("role") == "empty-evidence" for item in result["errors"])


def test_run_ocr_failure_message_includes_cudnn_guidance(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    def fake_backend(name, path, cfg, use_cache=True):
        raise OSError("Could not load cudnn_cnn64_9.dll")

    monkeypatch.setattr(ocr, "_run_backend", fake_backend)
    monkeypatch.setattr(ocr, "_fallback_engine_names", lambda cfg: [])
    monkeypatch.setattr(ocr, "_tesseract_available", lambda: False)

    with pytest.raises(RuntimeError, match="cuDNN"):
        ocr.run_ocr(str(image_path), {"ocr": {"primary": "ppocr-v6", "challengers": []}})


def test_doctr_moves_predictor_to_configured_device(monkeypatch):
    moved = []

    class _Predictor:
        def to(self, device):
            moved.append(str(device))
            return self

        def __call__(self, document):
            return SimpleNamespace(pages=[])

    monkeypatch.setitem(sys.modules, "doctr", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "doctr.io", SimpleNamespace(
        DocumentFile=SimpleNamespace(from_images=lambda path: "doc"),
    ))
    monkeypatch.setitem(sys.modules, "doctr.models", SimpleNamespace(
        ocr_predictor=lambda **kwargs: _Predictor(),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
    ))

    ocr._DOCTR_ENGINES.clear()
    ocr._doctr("fake.png", {"device": "cuda"})

    assert moved == ["cuda"]


def test_doctr_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    moved = []

    class _Predictor:
        def to(self, device):
            moved.append(str(device))
            return self

        def __call__(self, document):
            return SimpleNamespace(pages=[])

    monkeypatch.setitem(sys.modules, "doctr", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "doctr.io", SimpleNamespace(
        DocumentFile=SimpleNamespace(from_images=lambda path: "doc"),
    ))
    monkeypatch.setitem(sys.modules, "doctr.models", SimpleNamespace(
        ocr_predictor=lambda **kwargs: _Predictor(),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
    ))

    ocr._DOCTR_ENGINES.clear()
    ocr._doctr("fake.png", {"device": "cuda"})

    assert moved == ["cpu"]


def test_ensemble_disagreement_lines_filters_confident_disputes():
    lines = [
        {
            "text": "SAVE 30%", "conf": 0.93, "box": _box(1, 2, 30, 10),
            "meta": {"disagreement": ["SAVE 30%", "SAVF 30%"]},
        },
        {
            "text": "LOW", "conf": 0.55, "box": _box(1, 20, 30, 10),
            "meta": {"disagreement": ["LOW", "L0W"]},
        },
        {"text": "OK", "conf": 0.99, "box": _box(1, 40, 30, 10), "meta": {}},
    ]

    picked = ocr.ensemble_disagreement_lines(
        lines,
        {"ocr": {"ensemble_disagreement": {"enabled": True, "min_confidence": 0.85}}},
    )

    assert [line["text"] for line in picked] == ["SAVE 30%"]


def test_parse_easyocr_readtext_results():
    results = [
        ([[10, 20], [150, 20], [150, 50], [10, 50]], "SAVE 30%", 0.93),
        ([[12, 62], [132, 60], [133, 84], [13, 86]], "Today only", 0.81),
    ]

    lines = ocr._parse_easyocr_results(results)

    assert [line["text"] for line in lines] == ["SAVE 30%", "Today only"]
    assert lines[0]["box"] == _box(10, 20, 140, 30)
    assert lines[1]["quad"][1] == [132.0, 60.0]
    assert lines[0]["meta"]["backend_api"] == "easyocr-readtext"
    assert lines[0]["words"] == []


def test_easyocr_backend_uses_cached_reader(monkeypatch):
    created = []

    class _Reader:
        def __init__(self, langs, gpu=False):
            created.append((langs, gpu))

        def readtext(self, path):
            return [([[1, 2], [31, 2], [31, 12], [1, 12]], "Hello", 0.9)]

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=_Reader))
    ocr._EASYOCR_ENGINES.clear()

    lines, engine = ocr._easyocr("fake.png", {"device": "cuda", "ocr": {"lang": "en"}})

    assert engine == "easyocr"
    assert lines[0]["text"] == "Hello"
    assert created == [(["en"], True)]

    ocr._easyocr("fake2.png", {"device": "cuda", "ocr": {"lang": "en"}})
    assert len(created) == 1


def test_clear_engine_caches_clears_easyocr():
    ocr._EASYOCR_ENGINES[("en", False)] = object()
    ocr.clear_engine_caches()
    assert ocr._EASYOCR_ENGINES == {}
