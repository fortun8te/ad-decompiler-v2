"""Shared HTTP client helpers for local vision-language models (LM Studio, etc.)."""
from __future__ import annotations

import base64
import io
import json
import urllib.request

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_MODEL = "google/gemma-4-e4b"
_DEFAULT_TIMEOUT_S = 20
_DEFAULT_MAX_TOKENS = 500


def crop_box_bytes(image, box: dict, padding: int):
    x0 = max(0, int(box["x"]) - padding)
    y0 = max(0, int(box["y"]) - padding)
    x1 = min(image.width, int(box["x"] + box["w"]) + padding)
    y1 = min(image.height, int(box["y"] + box["h"]) + padding)
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
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return content.strip()


def multi_pass_answer(
    crop: bytes,
    prompt: str,
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    passes: int,
) -> tuple[str | None, str | None]:
    """Run the VLM up to `passes` times. Returns (accepted_answer, note).

    accepted_answer is set only when every pass succeeded and all answers match.
    note is vlm_disagreement when passes succeeded but disagreed, or vlm_error when
    any pass raised."""
    answers: list[str | None] = []
    for _ in range(max(1, passes)):
        try:
            answers.append(
                ask_vlm(
                    crop,
                    prompt,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    max_tokens=max_tokens,
                )
            )
        except Exception:
            return None, "vlm_error"
    if len(set(answers)) != 1:
        return None, "vlm_disagreement"
    return answers[0], None
