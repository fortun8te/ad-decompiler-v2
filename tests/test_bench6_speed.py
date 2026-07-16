"""postfix-benchmark-6 speed fixes: VLM circuit breaker + the Flux->Big-LaMa collision.

Both fixes are wall-clock only. Neither may change a single output pixel or answer:

  1  VLM circuit breaker. LM Studio evicts gemma when CUDA engines are resident; the
     endpoint then returns HTTP 400 {"error":"terminated"} or stops responding, and every
     later call paid its FULL timeout before failing (bench-6: 9x60s + 6x24s = 684s of
     dead wait; 094 wedged the run). Those calls were already failing — the breaker only
     stops waiting for a known-dead endpoint, so callers see the same "no answer" they
     already saw, just sooner.

  2  Flux -> Big-LaMa VRAM collision. Measured over every big-lama peel call in bench-6:
     0s / 2s on a free card (013, 107) vs 715s / 163s directly after a flux-comfy fill
     (091) — 878s, 18% of the bench. Releasing Flux first cannot change the fill: it
     evicts another process's weights, not any input.
"""
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import peel_scene, vlm_client, vram


# ── 1: VLM circuit breaker ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_breaker():
    vlm_client.reset_breaker()
    vlm_client.reset_cache()
    yield
    vlm_client.reset_breaker()
    vlm_client.reset_cache()


def _timeout(_req, timeout=None):
    raise TimeoutError("timed out")


def test_breaker_stops_calling_a_dead_endpoint():
    calls = {"n": 0}

    def counted(req, timeout=None):
        calls["n"] += 1
        return _timeout(req, timeout)

    with mock.patch("urllib.request.urlopen", counted):
        notes = []
        for _ in range(10):
            try:
                vlm_client.ask_vlm(b"img", "prompt", timeout_s=60)
            except Exception as exc:
                notes.append(vlm_client.classify_vlm_exception(exc)[0])
    # Only the threshold's worth of real attempts are paid; the rest are refused instantly.
    assert calls["n"] == vlm_client._BREAKER_THRESHOLD
    assert notes[-1] == vlm_client.VLM_CIRCUIT_NOTE


def test_circuit_open_stays_inside_the_no_answer_contract():
    # Consumers branch on membership in VLM_ERROR_NOTES; a refused call must look like
    # every other "no usable answer" so existing fallbacks fire unchanged.
    assert vlm_client.VLM_CIRCUIT_NOTE in vlm_client.VLM_ERROR_NOTES
    note, detail = vlm_client.classify_vlm_exception(
        vlm_client.VLMCircuitOpen("endpoint down"))
    assert note == vlm_client.VLM_CIRCUIT_NOTE
    assert detail


def test_response_level_errors_never_trip_the_breaker():
    # Bad JSON is about THIS response, not endpoint health. Tripping on it would suppress
    # calls that would have succeeded — a quality regression.
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"not json"

    with mock.patch("urllib.request.urlopen", lambda *a, **k: _Resp()):
        for i in range(vlm_client._BREAKER_THRESHOLD + 2):
            with pytest.raises(vlm_client.VLMError):
                vlm_client.ask_vlm(b"img%d" % i, "prompt", timeout_s=5)
    assert not vlm_client.breaker_state()


def test_breaker_half_opens_and_a_success_closes_it(monkeypatch):
    with mock.patch("urllib.request.urlopen", _timeout):
        for _ in range(vlm_client._BREAKER_THRESHOLD):
            with pytest.raises(Exception):
                vlm_client.ask_vlm(b"img", "prompt", timeout_s=1)
    with mock.patch("urllib.request.urlopen", _timeout):
        with pytest.raises(vlm_client.VLMCircuitOpen):
            vlm_client.ask_vlm(b"img", "prompt", timeout_s=1)

    monkeypatch.setattr(vlm_client, "_BREAKER_COOLDOWN_S", 0.0)

    class _Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"hi"}}]}'

    with mock.patch("urllib.request.urlopen", lambda *a, **k: _Ok()):
        assert vlm_client.ask_vlm(b"img2", "prompt2", timeout_s=1) == "hi"
    state = next(iter(vlm_client.breaker_state().values()))
    assert state["consecutive"] == 0, "a success must close the breaker"


def test_breaker_can_be_disabled(monkeypatch):
    monkeypatch.setattr(vlm_client, "_BREAKER_ENABLED", False)
    calls = {"n": 0}

    def counted(req, timeout=None):
        calls["n"] += 1
        return _timeout(req, timeout)

    with mock.patch("urllib.request.urlopen", counted):
        for _ in range(5):
            with pytest.raises(Exception):
                vlm_client.ask_vlm(b"img", "prompt", timeout_s=1)
    assert calls["n"] == 5   # every call still attempted


# ── 2: Flux -> Big-LaMa VRAM collision ──────────────────────────────────────────────

_CFG = {"inpaint": {"mode": "flux_comfy", "comfy": {"enabled": True}}}


def _mode(meta):
    with mock.patch.object(vram, "free_comfy_vram", return_value=True) as freed:
        mode = peel_scene.peel_inpaint_mode(_CFG, meta)
    return mode, freed.called


def test_flux_is_released_before_a_big_lama_fill():
    mode, freed = _mode({"under_kind": "photo-fragment", "hole_px": 500,
                         "flux_state": {"used": 1, "budget": 4}})
    assert mode == "lama"
    assert freed is True


def test_no_release_when_flux_never_ran_this_peel():
    # Nothing of ours is resident; a /free round-trip would be pure overhead.
    mode, freed = _mode({"under_kind": "photo-fragment", "hole_px": 500,
                         "flux_state": {"used": 0, "budget": 4}})
    assert mode == "lama"
    assert freed is False


def test_no_release_on_paths_that_do_not_load_lama():
    for meta in ({"under_kind": "shape", "hole_px": 9000, "flux_state": {"used": 1}},
                 {"under_kind": "photo-fragment", "hole_px": 14618,
                  "flux_state": {"used": 0, "budget": 4, "canvas_px": 1080 * 1920}}):
        mode, freed = _mode(meta)
        assert mode in ("opencv", "flux_comfy")
        assert freed is False


def test_routing_decisions_are_unchanged_by_the_vram_hook():
    # The hook must be invisible to routing: same mode in, same mode out. This is the
    # no-quality-regression guarantee for the speed fix.
    cases = [
        ({"text_occluder": True}, "lama"),
        ({"under_kind": "shape", "hole_px": 9000}, "opencv"),
        ({"under_kind": "photo-fragment", "hole_px": 500}, "lama"),
        ({"under_kind": "photo-fragment", "hole_px": 14618,
          "flux_state": {"used": 0, "budget": 4, "canvas_px": 1080 * 1920}}, "flux_comfy"),
    ]
    for meta, expected in cases:
        with mock.patch.object(vram, "free_comfy_vram", return_value=True):
            assert peel_scene.peel_inpaint_mode(_CFG, meta) == expected, meta


def test_release_helper_only_fires_for_torch_engines():
    with mock.patch.object(vram, "free_comfy_vram", return_value=True) as freed:
        assert vram.release_flux_for_torch_inpaint(_CFG, "flux_comfy", flux_used=3) is False
        assert vram.release_flux_for_torch_inpaint(_CFG, "opencv", flux_used=3) is False
        assert freed.called is False
        assert vram.release_flux_for_torch_inpaint(_CFG, "big-lama", flux_used=3) is True


def test_release_never_raises_even_if_comfy_is_unreachable():
    with mock.patch.object(vram, "free_comfy_vram", side_effect=OSError("no comfy")):
        # peel must still route; VRAM policy is an optimization, never a failure mode.
        assert peel_scene.peel_inpaint_mode(
            _CFG, {"under_kind": "photo-fragment", "hole_px": 500,
                   "flux_state": {"used": 1}}) == "lama"
