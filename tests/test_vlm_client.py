"""CPU-only tests for the shared VLM HTTP client (src/vlm_client.py).

These mock urllib so no live server is required; they assert the exact
request payload shape and the empty-content/reasoning_content failure mode.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from src import vlm_client  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_vlm_cache():
    """Isolate the process-global result cache between tests.

    ask_vlm caches deterministic (image, prompt, params) -> answer. Tests reuse the
    same trivial inputs with different mocked responses, so without a reset the second
    test would receive the first test's cached answer. Production wants exactly that
    cross-call reuse; tests need each case to hit the mock fresh.
    """
    vlm_client.reset_cache()
    yield
    vlm_client.reset_cache()


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_request(monkeypatch, response_payload):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(response_payload)

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_ask_vlm_caches_identical_requests(monkeypatch):
    """A second identical (image, prompt, params) call is served from cache: no HTTP."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse({"choices": [{"message": {"content": "ANSWER"},
                                           "finish_reason": "stop"}]})

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)

    a = vlm_client.ask_vlm(b"img-bytes", "prompt-A", max_tokens=500)
    b = vlm_client.ask_vlm(b"img-bytes", "prompt-A", max_tokens=500)
    assert a == b == "ANSWER"
    assert calls["n"] == 1  # second call served from cache
    stats = vlm_client.cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_ask_vlm_cache_distinguishes_inputs(monkeypatch):
    """Different image OR prompt OR token budget must miss the cache (own answer)."""
    seq = iter(["R1", "R2", "R3", "R4"])

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": next(seq)},
                                           "finish_reason": "stop"}]})

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)

    assert vlm_client.ask_vlm(b"img1", "prompt", max_tokens=500) == "R1"
    assert vlm_client.ask_vlm(b"img2", "prompt", max_tokens=500) == "R2"      # diff image
    assert vlm_client.ask_vlm(b"img1", "prompt2", max_tokens=500) == "R3"     # diff prompt
    assert vlm_client.ask_vlm(b"img1", "prompt", max_tokens=900) == "R4"      # diff budget
    assert vlm_client.cache_stats()["hits"] == 0


def test_ask_vlm_cache_can_be_disabled(monkeypatch):
    """AD_VLM_CACHE=0 (module flag off) forces every call to hit the server."""
    monkeypatch.setattr(vlm_client, "_CACHE_ENABLED", False)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse({"choices": [{"message": {"content": "X"},
                                           "finish_reason": "stop"}]})

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)
    vlm_client.ask_vlm(b"img", "p", max_tokens=500)
    vlm_client.ask_vlm(b"img", "p", max_tokens=500)
    assert calls["n"] == 2


def test_ask_vlm_does_not_cache_failures(monkeypatch):
    """A failed call is never cached; a subsequent good call still reaches the server."""
    responses = [
        {"choices": [{"message": {"reasoning_content": "thinking"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "GOOD"}, "finish_reason": "stop"}]},
    ]
    idx = {"i": 0}

    def fake_urlopen(req, timeout=None):
        payload = responses[idx["i"]]
        idx["i"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(vlm_client.VLMError):
        vlm_client.ask_vlm(b"img", "p", max_tokens=500)
    # identical request retried: must NOT return a cached failure, must hit server again
    assert vlm_client.ask_vlm(b"img", "p", max_tokens=500) == "GOOD"


def test_ask_vlm_sends_only_known_fields(monkeypatch):
    captured = _capture_request(monkeypatch, {
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    })

    result = vlm_client.ask_vlm(
        b"fake-png-bytes",
        "Say OK",
        base_url="http://127.0.0.1:1234/v1",
        model="google/gemma-4-e4b",
        timeout_s=30,
        max_tokens=500,
    )

    assert result == "OK"
    body = captured["body"]
    assert set(body.keys()) == {"model", "messages", "max_tokens", "temperature", "reasoning_effort"}
    assert body["reasoning_effort"] == "none"
    assert body["model"] == "google/gemma-4-e4b"
    assert body["temperature"] == 0.0
    assert len(body["messages"]) == 1
    msg = body["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    parts = {part["type"] for part in msg["content"]}
    assert parts == {"text", "image_url"}
    text_part = next(p for p in msg["content"] if p["type"] == "text")
    assert text_part["text"] == "Say OK"
    image_part = next(p for p in msg["content"] if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert captured["headers"]["Content-type"] == "application/json"


def test_ask_vlm_enforces_min_max_tokens(monkeypatch):
    captured = _capture_request(monkeypatch, {
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    })
    vlm_client.ask_vlm(b"x", "prompt", max_tokens=20)
    assert captured["body"]["max_tokens"] >= vlm_client._MIN_MAX_TOKENS


def test_ask_vlm_strips_whitespace(monkeypatch):
    _capture_request(monkeypatch, {
        "choices": [{"message": {"content": "  UPFRONT  \n"}, "finish_reason": "stop"}],
    })
    result = vlm_client.ask_vlm(b"x", "prompt")
    assert result == "UPFRONT"


def test_ask_vlm_raises_when_only_reasoning_content_present(monkeypatch):
    """A reasoning model can burn its whole token budget on hidden reasoning
    and finish with an empty `content` but populated `reasoning_content`.
    That must not be silently treated as a valid (empty) answer."""
    _capture_request(monkeypatch, {
        "choices": [{
            "message": {"content": "", "reasoning_content": "Thinking..."},
            "finish_reason": "length",
        }],
    })
    with pytest.raises(RuntimeError):
        vlm_client.ask_vlm(b"x", "prompt", max_tokens=500)


def test_ask_vlm_empty_content_without_reasoning_returns_empty_string(monkeypatch):
    """A genuinely empty answer (no reasoning_content at all) should still
    surface as an empty string rather than raising, so callers that treat
    empty-as-no-answer keep working."""
    _capture_request(monkeypatch, {
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
    })
    result = vlm_client.ask_vlm(b"x", "prompt")
    assert result == ""


def test_ask_vlm_accepts_typed_text_parts(monkeypatch):
    _capture_request(monkeypatch, {
        "choices": [{"message": {"content": [
            {"type": "output_text", "text": "{\"role\":"},
            {"type": "text", "text": "\"wordmark\"}"},
        ]}}],
    })
    assert vlm_client.ask_vlm(b"x", "prompt") == '{"role":"wordmark"}'


def test_ask_vlm_sends_strict_json_schema_when_requested(monkeypatch):
    captured = _capture_request(monkeypatch, {
        "choices": [{"message": {"content": '{"role":"wordmark"}'}}],
    })
    schema = {"type": "object", "required": ["role"]}
    vlm_client.ask_vlm(b"x", "prompt", response_schema=schema)
    fmt = captured["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == schema


def test_ask_vlm_omits_reasoning_effort_when_none(monkeypatch):
    captured = _capture_request(monkeypatch, {
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    })
    vlm_client.ask_vlm(b"x", "prompt", reasoning_effort=None)
    assert "reasoning_effort" not in captured["body"]


def test_ask_vlm_reasoning_effort_override(monkeypatch):
    captured = _capture_request(monkeypatch, {
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    })
    vlm_client.ask_vlm(b"x", "prompt", reasoning_effort="low")
    assert captured["body"]["reasoning_effort"] == "low"


def test_multi_pass_answer_defaults_reasoning_effort_none(monkeypatch):
    seen = {}

    def fake_ask_vlm(crop, prompt, *, base_url, model, timeout_s, max_tokens,
                      response_schema=None, reasoning_effort="none"):
        seen["reasoning_effort"] = reasoning_effort
        return "OK"

    monkeypatch.setattr(vlm_client, "ask_vlm", fake_ask_vlm)
    answer, note = vlm_client.multi_pass_answer(
        b"crop", "prompt", base_url="x", model="m", timeout_s=1, max_tokens=500, passes=1,
    )
    assert note is None
    assert answer == "OK"
    assert seen["reasoning_effort"] == "none"


def test_multi_pass_consensus_canonicalizes_json_and_code_fences(monkeypatch):
    answers = iter(['{"label":"keep","score":1}', '```json\n{ "score": 1, "label": "keep" }\n```'])
    crops = []
    monkeypatch.setattr(vlm_client, "ask_vlm",
                        lambda crop, *args, **kwargs: crops.append(crop) or next(answers))
    answer, note = vlm_client.multi_pass_answer(
        b"crop", "prompt", base_url="x", model="m", timeout_s=1, max_tokens=500,
        passes=2, response_schema={"type": "object"}, crop_variants=[b"tight", b"wide"],
    )
    assert note is None
    assert json.loads(answer)["label"] == "keep"
    assert crops == [b"tight", b"wide"]


def test_parallelism_from_cfg_defaults_and_clamps():
    assert vlm_client.parallelism_from_cfg(None) == 4
    assert vlm_client.parallelism_from_cfg({}) == 4
    assert vlm_client.parallelism_from_cfg({"vlm": {"parallelism": 8}}) == 8
    assert vlm_client.parallelism_from_cfg({"vlm": {"parallel": 3}}) == 3
    assert vlm_client.parallelism_from_cfg({"vlm": {"parallelism": 0}}) == 1
    assert vlm_client.parallelism_from_cfg({"vlm": {"parallelism": "nope"}}) == 4


def test_map_parallel_preserves_order_and_uses_workers():
    seen = []

    def work(n):
        seen.append(n)
        return n * 10

    assert vlm_client.map_parallel(work, [1, 2, 3], workers=1) == [10, 20, 30]
    assert seen == [1, 2, 3]
    seen.clear()
    assert vlm_client.map_parallel(work, [1, 2, 3, 4], workers=4) == [10, 20, 30, 40]
    assert sorted(seen) == [1, 2, 3, 4]


def test_map_parallel_empty():
    assert vlm_client.map_parallel(lambda x: x, [], workers=4) == []


def test_parallel_crops_serialize_shared_lazy_decoder():
    class GuardedImage:
        width = 64
        height = 64

        def __init__(self):
            self._state = threading.Lock()
            self._active = False

        def crop(self, _box):
            with self._state:
                if self._active:
                    raise RuntimeError("concurrent image decode")
                self._active = True
            try:
                time.sleep(0.005)
                return Image.new("RGB", (16, 16), "white")
            finally:
                with self._state:
                    self._active = False

    shared = GuardedImage()
    box = {"x": 0, "y": 0, "w": 16, "h": 16}
    rows = vlm_client.map_parallel(
        lambda _n: vlm_client.crop_box_bytes(shared, box, 0),
        list(range(12)), workers=4,
    )
    assert all(value and value.startswith(b"\x89PNG") for value in rows)
