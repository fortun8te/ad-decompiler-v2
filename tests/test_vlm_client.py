"""CPU-only tests for the shared VLM HTTP client (src/vlm_client.py).

These mock urllib so no live server is required; they assert the exact
request payload shape and the empty-content/reasoning_content failure mode.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from src import vlm_client  # noqa: E402


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
