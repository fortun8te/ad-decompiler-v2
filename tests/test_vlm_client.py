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
    assert set(body.keys()) == {"model", "messages", "max_tokens", "temperature"}
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
