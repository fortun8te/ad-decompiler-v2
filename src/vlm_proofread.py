"""vlm_proofread.py — optional VLM double-check for low-confidence OCR lines.

OCR (doctr) can misread stylized/condensed ad typography outright, not just with low
confidence noise: "UPFRONT" -> "UPERONT", "€63 → €49" -> "E63 tf40 L". A geometry-only
OCR engine has no way to catch that a *reading* is wrong, only how confident it was.

This module crops each OCR line below a confidence threshold, sends the crop to a local
vision-language model (LM Studio's OpenAI-compatible /v1/chat/completions), and asks it to
transcribe the crop verbatim. If the VLM's answer differs from OCR's, the VLM reading wins.

Disabled by default (vlm.enabled: false) and fails silently: any network/timeout/parse
error just returns the original OCR lines unchanged, so a missing/stopped LM Studio never
breaks a pipeline run.  Not a source of new text — it only ever proofreads text OCR already
found at that location; it does not invent lines.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_MODEL = "google/gemma-4-e4b"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.85
_DEFAULT_TIMEOUT_S = 20
_DEFAULT_MAX_TOKENS = 500
_DEFAULT_PADDING = 3

_PROMPT = (
    "This crop contains exactly ONE line of text from an ad. Transcribe only that one line, "
    "character for character, preserving currency symbols (e.g. €), punctuation, and arrows "
    "(e.g. →) exactly as shown. Do not describe or transcribe any text outside this crop, and "
    "do not guess at text that is cut off at the edges. Output only the transcribed line, "
    "nothing else, no explanation, no newlines. If no legible text is visible, output an "
    "empty string."
)


def _crop_bytes(image, box: dict, padding: int):
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


def _ask_vlm(image_bytes: bytes, base_url: str, model: str, timeout_s: float, max_tokens: int):
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
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


def _looks_plausible(original: str, candidate: str) -> bool:
    """Reject obviously broken VLM answers: empty, a multi-line answer (a single OCR line's
    crop should never legitimately need one -- a newline means the model read past the crop
    into a neighboring line), or wildly longer than the source line (truncated reasoning, or
    the model padding out real content with guessed neighboring text)."""
    if not candidate:
        return False
    if "\n" in candidate:
        return False
    if len(candidate) > max(40, len(original) * 3):
        return False
    return True


def proofread_lines(image_path: str, ocr_result: dict, cfg: dict) -> dict:
    """Return a copy of ocr_result with low-confidence lines' text replaced by a VLM
    reading of the same crop, when the VLM produces a plausible, different transcription.
    Never raises -- any failure (LM Studio not running, bad response, PIL error) leaves the
    affected line's OCR text untouched."""
    vcfg = (cfg or {}).get("vlm") or {}
    if not vcfg.get("enabled", False):
        return ocr_result
    lines = ocr_result.get("lines") or []
    if not lines:
        return ocr_result

    threshold = float(vcfg.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD))
    base_url = str(vcfg.get("base_url", _DEFAULT_BASE_URL))
    model = str(vcfg.get("model", _DEFAULT_MODEL))
    timeout_s = float(vcfg.get("timeout_s", _DEFAULT_TIMEOUT_S))
    max_tokens = int(vcfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
    padding = int(vcfg.get("padding", _DEFAULT_PADDING))
    max_lines = vcfg.get("max_lines")

    candidates = [ln for ln in lines if float(ln.get("conf", 1.0)) < threshold and ln.get("box")]
    if max_lines is not None:
        candidates = candidates[: int(max_lines)]
    if not candidates:
        return ocr_result

    try:
        from PIL import Image
        image = Image.open(image_path)
    except Exception:
        return ocr_result

    checked = 0
    corrected = 0
    for line in candidates:
        crop = _crop_bytes(image, line["box"], padding)
        if crop is None:
            continue
        checked += 1
        try:
            answer = _ask_vlm(crop, base_url, model, timeout_s, max_tokens)
        except Exception:
            continue
        original = str(line.get("text", ""))
        if _looks_plausible(original, answer) and answer != original:
            line["ocr_text"] = original
            line["text"] = answer
            line["vlm_corrected"] = True
            corrected += 1

    result = dict(ocr_result)
    result["vlm_proofread"] = {
        "model": model, "threshold": threshold,
        "lines_checked": checked, "lines_corrected": corrected,
    }
    return result
