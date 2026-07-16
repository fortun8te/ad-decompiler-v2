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


def test_stage_boundary_pre_vlm_segment_filter_unloads_sam(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: events.append("sam"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"empty_cache_between_stages": False}}}
    vram.stage_boundary("fusion", "vlm-segment-filter", cfg, "/tmp/run")

    assert events == ["sam"]


def test_stage_boundary_pre_vlm_scene_text_unloads_ocr(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"empty_cache_between_stages": False}}}
    vram.stage_boundary("text", "vlm-scene-text", cfg, "/tmp/run")

    assert events == ["ocr"]


def test_unload_ocr_before_vlm_can_be_disabled(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"unload_ocr_before_vlm": False}}}
    vram.stage_boundary("ocr", "vlm-proofread", cfg, "/tmp/run")

    assert "ocr" not in events


def test_unload_ocr_before_sam_can_be_disabled(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: events.append("ocr"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)

    cfg = {"device": "cuda", "runtime": {"vram": {"unload_ocr_before_sam": False}}}
    vram.stage_boundary("qwen", "sam", cfg, "/tmp/run")

    assert "ocr" not in events
    assert "cache" in events


def test_empty_cache_collects_gc_before_cuda(monkeypatch):
    order = []
    monkeypatch.setattr(vram.gc, "collect", lambda: order.append("gc"))

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            order.append("empty_cache")

    fake_torch = type("torch", (), {"cuda": FakeCuda})()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    vram.optional_torch_cuda_empty_cache()
    assert order == ["gc", "empty_cache"]


# ── Flux GGUF quant selection by free VRAM ──────────────────────────────────────────
def _flux_cfg(**comfy):
    base = {"vram_adaptive_quant": True}
    base.update(comfy)
    return {"device": "cuda", "inpaint": {"mode": "flux_comfy", "comfy": {"enabled": True, **base}}}


def test_select_flux_quant_disabled_returns_none():
    cfg = {"inpaint": {"comfy": {"vram_adaptive_quant": False}}}
    assert vram.select_flux_quant(cfg, free_mib=15000) is None


def test_select_flux_quant_high_mid_low_by_free_vram():
    cfg = _flux_cfg()
    assert vram.select_flux_quant(cfg, free_mib=15000) == "flux1-fill-dev-Q6_K.gguf"
    assert vram.select_flux_quant(cfg, free_mib=9000) == "flux1-fill-dev-Q5_K_S.gguf"
    assert vram.select_flux_quant(cfg, free_mib=4000) == "flux1-fill-dev-Q4_K_S.gguf"


def test_select_flux_quant_unknown_free_prefers_high(monkeypatch):
    monkeypatch.setattr(vram, "free_vram_mib", lambda: None)
    cfg = _flux_cfg()
    assert vram.select_flux_quant(cfg, free_mib=None) == "flux1-fill-dev-Q6_K.gguf"


def test_select_flux_quant_custom_thresholds_and_ladder():
    cfg = _flux_cfg(
        quant_vram_thresholds={"high_min_free_mib": 12000, "mid_min_free_mib": 9000},
        quant_ladder={"high": "H.gguf", "mid": "M.gguf", "low": "L.gguf"},
    )
    assert vram.select_flux_quant(cfg, free_mib=13000) == "H.gguf"
    assert vram.select_flux_quant(cfg, free_mib=9500) == "M.gguf"
    assert vram.select_flux_quant(cfg, free_mib=8000) == "L.gguf"


def test_prepare_inpaint_vram_mutates_cfg_quant(monkeypatch):
    monkeypatch.setattr(vram, "free_vram_mib", lambda: 15000.0)
    cfg = _flux_cfg()
    cfg["inpaint"]["comfy"]["models"] = {"unet_gguf": "flux1-fill-dev-Q4_K_S.gguf"}
    record = vram.prepare_inpaint_vram(cfg)
    assert cfg["inpaint"]["comfy"]["models"]["unet_gguf"] == "flux1-fill-dev-Q6_K.gguf"
    assert record["flux_quant"] == "flux1-fill-dev-Q6_K.gguf"
    assert record["flux_quant_prev"] == "flux1-fill-dev-Q4_K_S.gguf"


def test_prepare_inpaint_vram_noop_when_flux_inactive(monkeypatch):
    monkeypatch.setattr(vram, "free_vram_mib", lambda: 15000.0)
    cfg = {"device": "cuda", "inpaint": {"mode": "lama", "comfy": {"vram_adaptive_quant": True}}}
    record = vram.prepare_inpaint_vram(cfg)
    assert record["flux_quant"] is None
    assert record["vlm_evicted"] is False


def test_prepare_inpaint_vram_evicts_vlm_when_enabled(monkeypatch):
    monkeypatch.setattr(vram, "free_vram_mib", lambda: 15000.0)
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: None)
    calls = []
    monkeypatch.setattr(vram, "evict_vlm", lambda cfg, log_fn=None: calls.append("evict") or True)
    cfg = _flux_cfg()
    cfg["vlm"] = {"enabled": True, "model": "google/gemma-4-e4b"}
    cfg["runtime"] = {"vram": {"evict_vlm_for_inpaint": True}}
    record = vram.prepare_inpaint_vram(cfg)
    assert calls == ["evict"]
    assert record["vlm_evicted"] is True


def test_lazy_flux_prep_defers_at_reconstruct_boundary(monkeypatch):
    """Regional + force_flux=false must not unload the VLM before reconstruct."""
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: None)
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: events.append("sam"))
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: None)
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)
    monkeypatch.setattr(
        vram, "prepare_inpaint_vram",
        lambda *a, **k: events.append("prep") or {"vlm_evicted": True})
    logs = []
    vram.reset_telemetry()
    cfg = {
        "device": "cuda",
        "inpaint": {
            "mode": "flux_comfy",
            "comfy": {"enabled": True},
            "regional": {"enabled": True, "force_flux": False},
        },
        "runtime": {"vram": {"evict_vlm_for_inpaint": True, "lazy_flux_prep": True,
                             "empty_cache_between_stages": False}},
    }
    vram.stage_boundary("merge", "reconstruct", cfg, "/tmp/run", log_fn=logs.append)
    assert "prep" not in events
    assert any("deferring Flux prep" in line for line in logs)
    tel = vram.telemetry()
    assert tel and tel[-1]["inpaint_prep"].get("deferred") is True


def test_ensure_flux_vram_runs_once(monkeypatch):
    monkeypatch.setattr(vram, "free_vram_mib", lambda: 15000.0)
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: None)
    calls = []
    monkeypatch.setattr(vram, "evict_vlm", lambda cfg, log_fn=None: calls.append("evict") or True)
    cfg = _flux_cfg()
    cfg["vlm"] = {"enabled": True, "model": "m"}
    cfg["runtime"] = {"vram": {"evict_vlm_for_inpaint": True}}
    vram.reset_telemetry()
    first = vram.ensure_flux_vram(cfg)
    second = vram.ensure_flux_vram(cfg)
    assert calls == ["evict"]
    assert first["vlm_evicted"] is True
    assert first["already_prepared"] is False
    assert second["already_prepared"] is True
    assert second["vlm_evicted"] is False


def test_eager_flux_prep_when_force_flux(monkeypatch):
    events = []
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: None)
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: None)
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: None)
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)
    monkeypatch.setattr(
        vram, "prepare_inpaint_vram",
        lambda *a, **k: events.append("prep") or {"vlm_evicted": True, "flux_quant": "Q.gguf"})
    vram.reset_telemetry()
    cfg = {
        "device": "cuda",
        "inpaint": {
            "mode": "flux_comfy",
            "comfy": {"enabled": True},
            "regional": {"enabled": True, "force_flux": True},
        },
        "runtime": {"vram": {"lazy_flux_prep": True, "empty_cache_between_stages": False}},
    }
    vram.stage_boundary("merge", "reconstruct", cfg, "/tmp/run")
    assert events == ["prep"]


def test_evict_vlm_shells_out_to_lms(monkeypatch):
    monkeypatch.setattr(vram, "_lms_path", lambda cfg=None: "/fake/lms")
    seen = {}

    class FakeResult:
        returncode = 0
        stdout = "Unloaded"
        stderr = ""

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeResult()

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    cfg = {"vlm": {"model": "google/gemma-4-e4b"}}
    assert vram.evict_vlm(cfg) is True
    assert seen["args"] == ["/fake/lms", "unload", "google/gemma-4-e4b"]


def test_evict_vlm_missing_cli_is_false(monkeypatch):
    monkeypatch.setattr(vram, "_lms_path", lambda cfg=None: None)
    assert vram.evict_vlm({"vlm": {"model": "m"}}) is False


def test_restore_vlm_gated_and_loads(monkeypatch):
    monkeypatch.setattr(vram, "_lms_path", lambda cfg=None: "/fake/lms")
    seen = {}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kw):
        seen["args"] = args
        return FakeResult()

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    # Disabled by default -> no load.
    off = {"vlm": {"enabled": True, "model": "m"}, "runtime": {"vram": {}}}
    assert vram.restore_vlm(off) is False
    assert "args" not in seen
    # Enabled -> lms load.
    on = {"vlm": {"enabled": True, "model": "google/gemma-4-e4b"},
          "runtime": {"vram": {"evict_vlm_for_inpaint": True, "reload_vlm_after_inpaint": True}}}
    assert vram.restore_vlm(on) is True
    assert seen["args"][:3] == ["/fake/lms", "load", "google/gemma-4-e4b"]


def test_telemetry_reset_and_accumulate(monkeypatch):
    monkeypatch.setattr(vram, "_snapshot", lambda: {"gpu": {"free_mib": 4000, "used_mib": 12000, "total_mib": 16000}})
    monkeypatch.setattr(vram, "unload_ocr_engines", lambda: None)
    monkeypatch.setattr(vram, "unload_sam_backend", lambda: None)
    monkeypatch.setattr(vram, "optional_torch_cuda_empty_cache", lambda: None)
    monkeypatch.setattr(vram, "log_vram", lambda label, log_fn=None: None)
    vram.reset_telemetry()
    cfg = {"device": "cuda", "runtime": {"vram": {}}}
    vram.stage_boundary("qwen", "sam", cfg, "/tmp/run")
    tel = vram.telemetry()
    assert len(tel) == 1
    assert tel[0]["boundary"] == "qwen->sam"
    assert tel[0]["before"]["gpu"]["free_mib"] == 4000
    vram.reset_telemetry()
    assert vram.telemetry() == []


def test_nvidia_smi_mem_parses_csv(monkeypatch):
    monkeypatch.setattr(vram.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class FakeResult:
        returncode = 0
        stdout = "16303, 12273, 3705\n"

    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: FakeResult())
    info = vram.nvidia_smi_mem()
    assert info == {"total_mib": 16303, "used_mib": 12273, "free_mib": 3705}
