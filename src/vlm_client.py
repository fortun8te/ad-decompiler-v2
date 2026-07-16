"""Shared HTTP client helpers for local vision-language models (LM Studio, etc.)."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

# PIL's lazy ImageFile decoder is not safe when several VLM workers crop the
# same open image concurrently.  Serialize only decode+crop; HTTP inference
# remains parallel.
_CROP_LOCK = threading.Lock()

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_MODEL = "google/gemma-4-e4b"
# Under GPU contention (dense ads fan out 30-70 crop calls at parallelism=4),
# LM Studio queues requests; a 20s ceiling caused ~54/56 OCR-judge calls on
# fixture 002 to time out and DISCARD queued-but-in-progress work, which the
# harness then re-issued (double work) — and flip-flopped plateau/rollback
# decisions between runs. A 60s ceiling lets genuinely-queued calls complete
# once; it costs nothing on the normal fast path (calls return in <3s) and only
# extends the tail when the server is truly saturated. Override per call site
# via cfg (e.g. layout.vlm_grouping.timeout_s pins grouping to a shorter bound).
_DEFAULT_TIMEOUT_S = 60
_DEFAULT_MAX_TOKENS = 500
# Reasoning models (e.g. gemma-4-e4b in LM Studio) burn part of the token
# budget on hidden "reasoning_content" before emitting the final answer in
# "content". If max_tokens is too small, generation can hit the length limit
# while still inside the reasoning block, leaving content empty. Enforce a
# floor so callers that pass a small max_tokens don't silently get truncated
# before any real answer is produced.
_MIN_MAX_TOKENS = 500
# LM Studio continuous batching is typically configured with parallel slots
# (lms ps shows "parallel: 4"). Sequential for-loops leave those slots idle.
_DEFAULT_PARALLELISM = 4

T = TypeVar("T")
R = TypeVar("R")

# Optional per-call tracing: set AD_VLM_TRACE=<path.jsonl> to append one JSON line per
# ask_vlm call (caller module, wall time, image size, token usage). Zero cost when unset;
# never raises; never changes request/response behaviour.
_TRACE_PATH = os.environ.get("AD_VLM_TRACE")
_TRACE_LOCK = threading.Lock()


def _trace_caller() -> str:
    """Nearest calling module outside vlm_client (best-effort, cheap)."""
    try:
        frame = sys._getframe(2)
        while frame is not None:
            name = frame.f_globals.get("__name__", "")
            if name and "vlm_client" not in name and not name.startswith(("concurrent", "threading")):
                return name
            frame = frame.f_back
    except Exception:
        pass
    return "unknown"


def _trace(record: dict) -> None:
    if not _TRACE_PATH:
        return
    try:
        with _TRACE_LOCK:
            with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# Content-addressed result cache. Requests are deterministic (temperature=0.0), so an
# identical (model, prompt, image, token budget, schema, reasoning) request always yields
# the same answer. The harness re-runs the pipeline for several repair rounds in the SAME
# process; without a cache the exact same font/OCR/grouping crops are re-inferred each round
# (measured: 35 of 86 calls on fixture 002 were byte-identical repeats). Caching returns the
# stored answer instantly with zero change to output — pure wall-clock savings.
# Disable with AD_VLM_CACHE=0. Never caches failures (only fully successful answers).
_CACHE_ENABLED = os.environ.get("AD_VLM_CACHE", "1") not in ("0", "false", "no", "")
_CACHE_LOCK = threading.Lock()
_RESULT_CACHE: dict[str, str] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cache_key(image_bytes: bytes, prompt: str, model: str, max_tokens: int,
               response_schema: dict | None, reasoning_effort: str | None) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(max_tokens).encode("utf-8"))
    h.update(b"\x00")
    h.update((reasoning_effort or "").encode("utf-8"))
    h.update(b"\x00")
    if response_schema:
        h.update(json.dumps(response_schema, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(hashlib.sha256(image_bytes).digest())
    return h.hexdigest()


def cache_stats() -> dict:
    """Snapshot of cache hit/miss counters (for run reporting)."""
    with _CACHE_LOCK:
        return dict(_CACHE_STATS)


def reset_cache() -> None:
    """Clear the result cache and counters (used between independent fixtures/tests)."""
    with _CACHE_LOCK:
        _RESULT_CACHE.clear()
        _CACHE_STATS["hits"] = 0
        _CACHE_STATS["misses"] = 0


class VLMError(RuntimeError):
    """A useful, non-secret error from the local OpenAI-compatible endpoint."""


def parallelism_from_cfg(cfg: dict | None = None, default: int = _DEFAULT_PARALLELISM) -> int:
    """``vlm.parallelism`` — concurrent independent VLM requests (LM Studio slots)."""
    vcfg = (cfg or {}).get("vlm") if isinstance(cfg, dict) else None
    if not isinstance(vcfg, dict):
        return max(1, int(default))
    raw = vcfg.get("parallelism", vcfg.get("parallel", default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def map_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int = _DEFAULT_PARALLELISM,
) -> list[R]:
    """Run ``fn`` over ``items`` with a thread pool; preserve input order.

    Independent VLM crop calls are I/O-bound on the LM Studio HTTP server, which
    already continuous-batches concurrent requests (typically 4 slots). Workers=1
    keeps the old sequential behaviour for tests / debugging.
    """
    sequence = list(items)
    if not sequence:
        return []
    worker_count = max(1, int(workers))
    if worker_count == 1 or len(sequence) == 1:
        return [fn(item) for item in sequence]

    results: list[R | None] = [None] * len(sequence)
    with ThreadPoolExecutor(max_workers=min(worker_count, len(sequence))) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(sequence)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results  # type: ignore[return-value]


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
    with _CROP_LOCK:
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
    cache_key: str | None = None
    if _CACHE_ENABLED:
        cache_key = _cache_key(image_bytes, prompt, model,
                               max(max_tokens, _MIN_MAX_TOKENS), response_schema, reasoning_effort)
        with _CACHE_LOCK:
            hit = _RESULT_CACHE.get(cache_key)
            if hit is not None:
                _CACHE_STATS["hits"] += 1
        if hit is not None:
            if _TRACE_PATH:
                _trace({
                    "ts": round(time.time(), 3),
                    "site": _trace_caller(),
                    "image_kb": round(len(image_bytes) / 1024, 1),
                    "prompt_chars": len(prompt),
                    "max_tokens": max(max_tokens, _MIN_MAX_TOKENS),
                    "timeout_s": timeout_s,
                    "schema": bool(response_schema),
                    "s": 0.0,
                    "cached": True,
                })
            return hit
        with _CACHE_LOCK:
            _CACHE_STATS["misses"] += 1
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
    trace: dict | None = None
    if _TRACE_PATH:
        trace = {
            "ts": round(time.time(), 3),
            "site": _trace_caller(),
            "image_kb": round(len(image_bytes) / 1024, 1),
            "prompt_chars": len(prompt),
            "max_tokens": max(max_tokens, _MIN_MAX_TOKENS),
            "timeout_s": timeout_s,
            "schema": bool(response_schema),
        }
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        if trace is not None:
            trace.update({"s": round(time.perf_counter() - started, 3), "error": f"http-{exc.code}"})
            _trace(trace)
        raise VLMError(f"VLM HTTP {exc.code}: {detail}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if trace is not None:
            trace.update({"s": round(time.perf_counter() - started, 3), "error": "bad-json"})
            _trace(trace)
        raise VLMError(f"VLM returned an invalid JSON response: {exc}") from exc
    except Exception as exc:
        if trace is not None:
            trace.update({"s": round(time.perf_counter() - started, 3),
                          "error": type(exc).__name__})
            _trace(trace)
        raise
    if trace is not None:
        usage = data.get("usage") or {}
        trace.update({
            "s": round(time.perf_counter() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
        _trace(trace)
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
    if cache_key is not None and content:
        with _CACHE_LOCK:
            _RESULT_CACHE[cache_key] = content
    return content


# Failure sentinels returned by multi_pass_answer in the `note` slot. All three
# denote "no usable answer"; consumers that only care about that should test
# membership in VLM_ERROR_NOTES rather than == "vlm_error". They are kept
# distinct so harness_loop.py (and other consumers) CAN branch on cause:
#   vlm_timeout      -> the endpoint did not respond within timeout_s (GPU
#                       contention / saturated queue). NON-deterministic under
#                       load: the same input may succeed on a quieter run, so
#                       treat as transient, not as evidence the VLM found nothing.
#   vlm_error        -> the endpoint responded with an error, or the request
#                       otherwise failed deterministically (HTTP 4xx/5xx, bad
#                       JSON, empty content after a full generation). Re-running
#                       identically will usually fail identically.
#   vlm_empty        -> every pass succeeded but the accepted answer was blank.
#   vlm_disagreement -> passes succeeded but the reads did not reach consensus
#                       (this is not a failure; it feeds deterministic fallback).
VLM_TIMEOUT_NOTE = "vlm_timeout"
VLM_ERROR_NOTE = "vlm_error"
VLM_EMPTY_NOTE = "vlm_empty"
VLM_DISAGREEMENT_NOTE = "vlm_disagreement"
# The set of notes that mean "no answer was accepted" (an error, not a disagreement).
VLM_ERROR_NOTES = frozenset({VLM_TIMEOUT_NOTE, VLM_ERROR_NOTE, VLM_EMPTY_NOTE})

# Best-effort visibility for discarded VLM failures. Previously every exception
# collapsed to a bare "vlm_error" with the cause thrown away, so a timeout under
# load was indistinguishable from a genuine endpoint error in the logs. Set
# AD_VLM_QUIET=1 to silence. Never raises.
_VLM_QUIET = os.environ.get("AD_VLM_QUIET", "0") in ("1", "true", "yes")


def classify_vlm_exception(exc: BaseException) -> tuple[str, str]:
    """Map a raised VLM exception to (note, human_detail).

    note is one of VLM_TIMEOUT_NOTE / VLM_ERROR_NOTE. detail is a short,
    non-secret description safe to log. A socket/connection timeout (including
    one wrapped in urllib.error.URLError) is reported as a timeout; everything
    else — HTTP status errors, bad JSON, empty content (VLMError) — is an error."""
    reason: BaseException | object = exc
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        reason = exc.reason
    is_timeout = (
        isinstance(exc, (socket.timeout, TimeoutError))
        or isinstance(reason, (socket.timeout, TimeoutError))
        or (isinstance(reason, OSError) and "timed out" in str(reason).lower())
        or "timed out" in str(exc).lower()
    )
    detail = f"{type(exc).__name__}: {exc}"[:400]
    return (VLM_TIMEOUT_NOTE if is_timeout else VLM_ERROR_NOTE), detail


def _log_vlm_failure(site: str, note: str, detail: str, timeout_s: float) -> None:
    if _VLM_QUIET:
        return
    try:
        sys.stderr.write(
            f"[vlm] {note} in {site} (timeout_s={timeout_s}): {detail}\n"
        )
    except Exception:
        pass


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
    On failure, note distinguishes the cause: vlm_timeout (transient, endpoint
    did not respond in time), vlm_error (deterministic endpoint/parse failure),
    vlm_empty (blank consensus answer), or vlm_disagreement (passes disagreed).
    See VLM_ERROR_NOTES for the "no usable answer" set."""
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
        except Exception as exc:
            note, detail = classify_vlm_exception(exc)
            _log_vlm_failure(_trace_caller(), note, detail, timeout_s)
            return None, note
    # Structured responses can be semantically identical despite harmless whitespace or
    # JSON key ordering differences. Compare canonical forms while returning the first
    # original answer so transcription punctuation/case remains untouched.
    if len({consensus_key(answer) for answer in answers}) != 1:
        return None, VLM_DISAGREEMENT_NOTE
    if not (answers[0] or "").strip():
        return None, VLM_EMPTY_NOTE
    return answers[0], None
