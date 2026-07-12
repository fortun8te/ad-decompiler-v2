"""Unit tests for CUDA VRAM management hooks (mocked — no GPU required)."""
from __future__ import annotations

import pytest

from src import ocr, sam3_detect, vram


@pytest.fixture(autouse=True)
def _reset_engine_caches():
    ocr.clear_engine_caches()
    sam3_detect.unload_backend()
    yield
    ocr.clear_engine_caches()
    sam3_detect.unload_backend()


def test_optional_torch_cuda_empty_cache_no_torch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    vram.optional_torch_cuda_empty_cache()  # must not raise


def test_optional_torch_cuda_empty_cache_calls_cuda(monkeypatch):
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls.append("empty_cache")

    fake_torch = type("torch", (), {"cuda": FakeCuda})()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    vram.optional_torch_cuda_empty_cache()
    assert calls == ["empty_cache"]


def test_log_vram_reports_allocated_mib(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def memory_allocated():
            return 256 * 1024 * 1024

    fake_torch = type("torch", (), {"cuda": FakeCuda})()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    lines = []
    allocated = vram.log_vram("probe", log_fn=lines.append)
    assert allocated == 256 * 1024 * 1024
    assert lines == ["vram[probe] allocated=256.0MiB"]


def test_unload_ocr_engines_clears_caches(monkeypatch):
    ocr._PADDLE_ENGINES[("k",)] = ("engine", "api")
    ocr._DOCTR_ENGINES[("k",)] = "predictor"
    ocr._SURYA_ENGINES[("k",)] = ("pred", "api", "det")
    vram.unload_ocr_engines()
    assert not ocr._PADDLE_ENGINES
    assert not ocr._DOCTR_ENGINES
    assert not ocr._SURYA_ENGINES


def test_unload_sam_backend_clears_cache():
    sam3_detect._BACKEND_CACHE["key"] = object()
    vram.unload_sam_backend()
    assert not sam3_detect._BACKEND_CACHE


def test_stage_boundary_pre_sam_unloads_ocr_and_empty_cache(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: events.append("sam"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: events.append(f"log:{label}"))

    cfg = {"device": "cuda", "runtime": {"vram": {"unload_ocr_before_sam": True, "empty_cache_between_stages": True}}}
    vram.stage_boundary("qwen", "sam", cfg, "/tmp/run")

    assert events == [
        "log:before-qwen->sam",
        "ocr",
        "cache",
        "log:after-qwen->sam",
    ]


def test_stage_boundary_pre_reconstruct_unloads_sam(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: events.append("sam"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"empty_cache_between_stages": False}}}
    vram.stage_boundary("merge", "reconstruct", cfg, "/tmp/run")

    assert events == ["sam"]


def test_empty_cache_defaults_true_on_cuda():
    assert vram._vram_cfg({"device": "cuda"})["empty_cache_between_stages"] is True


def test_empty_cache_defaults_false_on_cpu():
    assert vram._vram_cfg({"device": "cpu"})["empty_cache_between_stages"] is False


def test_unload_ocr_before_sam_can_be_disabled(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"unload_ocr_before_sam": False}}}
    vram.stage_boundary("qwen", "sam", cfg, "/tmp/run")

    assert "ocr" not in events
    assert "cache" in events
