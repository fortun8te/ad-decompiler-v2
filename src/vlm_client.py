"""Shared HTTP client helpers for local vision-language models (LM Studio, etc.)."""
from __future__ import annotations

import base64
import io
import json
import urllib.request
import urllib.error

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_MODEL = "google/gemma-4-e4b"
_DEFAULT_TIMEOUT_S = 20
_DEFAULT_MAX_TOKENS = 500
# Reasoning models (e.g. gemma-4-e4b in LM Studio) burn part of the token
# budget on hidden "reasoning_content" before emitting the final answer in
# "content". If max_tokens is too small, generation can hit the length limit
# while still inside the reasoning block, leaving content empty. Enforce a
# floor so callers that pass a small max_tokens don't silently get truncated
# before any real answer is produced.
_MIN_MAX_TOKENS = 500


class VLMError(RuntimeError):
    """A useful, non-secret error from the local OpenAI-compatible endpoint."""


def consensus_key(answer) -> str:
    """Canonical comparison key without changing the answer returned to callers."""
    value = str(answer or "").strip().replace("\r\n", "\n")
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(value)
        return "json:" + json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (json.JSONDecodeError, TypeError):
        return "text:" + value


def _message_text(content) -> str:
    """Accept both OpenAI's string content and LM Studio's typed content parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
    return ""


def model_evidence(base_url: str, requested_model: str, response: dict | None = None) -> dict:
    """Machine-readable identity evidence stored beside every VLM stage result."""
    response = response or {}
    return {
        "provider": "openai-compatible",
        "base_url": base_url.rstrip("/"),
        "requested_model": requested_model,
        "response_model": str(response.get("model") or ""),
        "response_id": str(response.get("id") or ""),
    }


def crop_box_bytes(image, box: dict, padding: int):
    # Model-adjacent stages consume observations from several detectors. One malformed
    # box must skip only that observation, not crash every later VLM stage.
    try:
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["w"])
        h = float(box["h"])
        pad = int(padding)
        import math
        if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(image.width, int(x + w) + pad)
        y1 = min(image.height, int(y + h) + pad)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def ask_vlm(
    image_bytes: bytes,
    prompt: str,
    *,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    response_schema: dict | None = None,
    reasoning_effort: str | None = "none",
) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }],
        "max_tokens": max(max_tokens, _MIN_MAX_TOKENS),
        "temperature": 0.0,
    }
    if reasoning_effort is not None:
        # gemma-4-e4b (and other reasoning-capable models served via LM Studio) emit a
        # hidden reasoning_content block before the real answer. On structured/complex
        # prompts that reasoning can consume the whole token budget, leaving content
        # empty with finish_reason='length'. "none" disables reasoning entirely: no
        # reasoning tokens, direct answer, much faster, finish_reason='stop'.
        payload["reasoning_effort"] = reasoning_effort
    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "vlm_result",
                "strict": True,
                "schema": response_schema,
            },
        }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise VLMError(f"VLM HTTP {exc.code}: {detail}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VLMError(f"VLM returned an invalid JSON response: {exc}") from exc
    if data.get("error"):
        raise VLMError(f"VLM error response: {data['error']}")
    message = (data.get("choices") or [{}])[0].get("message", {})
    content = _message_text(message.get("content"))
    if not content and message.get("reasoning_content"):
        # The model spent its whole budget on hidden reasoning and never
        # emitted a final answer. Returning the reasoning text as if it were
        # the answer would silently corrupt downstream comparisons, so treat
        # this as a hard failure instead.
        raise VLMError(
            "VLM returned only reasoning_content with no final content "
            "(finish_reason=%r); increase max_tokens or inspect the prompt."
            % (data.get("choices") or [{}])[0].get("finish_reason")
        )
    return content


def multi_pass_answer(
    crop: bytes,
    prompt: str,
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    passes: int,
    response_schema: dict | None = None,
    crop_variants: list[bytes] | None = None,
    reasoning_effort: str | None = "none",
) -> tuple[str | None, str | None]:
    """Run the VLM up to `passes` times. Returns (accepted_answer, note).

    accepted_answer is set only when every pass succeeded and all answers match.
    note is vlm_disagreement when passes succeeded but disagreed, or vlm_error when
    any pass raised."""
    answers: list[str | None] = []
    variants = [value for value in (crop_variants or [crop]) if value] or [crop]
    for pass_index in range(max(1, passes)):
        try:
            answers.append(
                ask_vlm(
                    variants[pass_index % len(variants)],
                    prompt,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                    reasoning_effort=reasoning_effort,
                )
            )
        except Exception:
            return None, "vlm_error"
    # Structured responses can be semantically identical despite harmless whitespace or
    # JSON key ordering differences. Compare canonical forms while returning the first
    # original answer so transcription punctuation/case remains untouched.
    if len({consensus_key(answer) for answer in answers}) != 1:
        return None, "vlm_disagreement"
    return answers[0], None
