"""vlm_proofread.py — optional VLM double-check for low-confidence OCR lines.

OCR (doctr) can misread stylized/condensed ad typography outright, not just with low
confidence noise: "UPFRONT" -> "UPERONT", "€63 → €49" -> "E63 tf40 L". A geometry-only
OCR engine has no way to catch that a *reading* is wrong, only how confident it was.

This module crops each OCR line below a confidence threshold, sends the crop to a local
vision-language model (LM Studio's OpenAI-compatible /v1/chat/completions), and asks it to
transcribe the crop verbatim. Corrections require two independent VLM passes that agree
with each other before OCR text is replaced.

Disabled by default (vlm.enabled: false) and fails silently: any network/timeout/parse
error just returns the original OCR lines unchanged, so a missing/stopped LM Studio never
breaks a pipeline run.  Not a source of new text — it only ever proofreads text OCR already
found at that location; it does not invent lines.
"""
from __future__ import annotations

from src import vlm_client

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_MODEL = "google/gemma-4-e4b"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.85
_DEFAULT_TIMEOUT_S = 20
_DEFAULT_MAX_TOKENS = 500
_DEFAULT_PADDING = 3
_DEFAULT_PASSES = 2

_PROMPT = (
    "This crop contains exactly ONE line of text from an ad. Transcribe only that one line, "
    "character for character, preserving currency symbols (e.g. €), punctuation, and arrows "
    "(e.g. →) exactly as shown. Do not describe or transcribe any text outside this crop, and "
    "do not guess at text that is cut off at the edges. Output only the transcribed line, "
    "nothing else, no explanation, no newlines. If no legible text is visible, output an "
    "empty string."
)


def _ask_vlm(image_bytes: bytes, base_url: str, model: str, timeout_s: float, max_tokens: int):
    return vlm_client.ask_vlm(
        image_bytes,
        _PROMPT,
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )


def _multi_pass_answer(
    crop: bytes,
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    passes: int,
) -> tuple[str | None, str | None]:
    answers: list[str | None] = []
    for _ in range(max(1, passes)):
        try:
            answers.append(_ask_vlm(crop, base_url, model, timeout_s, max_tokens))
        except Exception:
            return None, "vlm_error"
    if len({vlm_client.consensus_key(answer) for answer in answers}) != 1:
        return None, "vlm_disagreement"
    return answers[0], None


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
    reading of the same crop when two independent passes agree. Never raises -- any failure
    (LM Studio not running, bad response, PIL error) leaves the affected line's OCR text
    untouched."""
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
    passes = int(vcfg.get("passes", _DEFAULT_PASSES))
    ocr_cfg = (cfg or {}).get("ocr") or {}
    ensemble_cfg = ocr_cfg.get("ensemble_disagreement")
    ensemble_enabled = bool(ensemble_cfg)
    ensemble_min_conf = 0.85
    if isinstance(ensemble_cfg, dict):
        ensemble_enabled = bool(ensemble_cfg.get("enabled", True))
        ensemble_min_conf = float(ensemble_cfg.get("min_confidence", ensemble_min_conf))

    def _proofread_candidate(line: dict) -> bool:
        if not line.get("box"):
            return False
        if float(line.get("conf", 1.0)) < threshold:
            return True
        if ensemble_enabled and (line.get("meta") or {}).get("disagreement"):
            return float(line.get("conf", 0.0)) >= ensemble_min_conf
        return False

    candidates = [ln for ln in lines if _proofread_candidate(ln)]
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
    disagreements = 0
    errors = 0
    ensemble_checked = 0
    notes: list[dict] = []
    workers = vlm_client.parallelism_from_cfg(cfg)

    def _one(line: dict):
        is_ensemble = (
            ensemble_enabled
            and float(line.get("conf", 0.0)) >= threshold
            and bool((line.get("meta") or {}).get("disagreement"))
        )
        crop = vlm_client.crop_box_bytes(image, line["box"], padding)
        if crop is None:
            return None
        answer, note = _multi_pass_answer(
            crop,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            passes=passes,
        )
        return {
            "line": line,
            "is_ensemble": is_ensemble,
            "answer": answer,
            "note": note,
            "original": str(line.get("text", "")),
        }

    for result in vlm_client.map_parallel(_one, candidates, workers=workers):
        if result is None:
            continue
        line = result["line"]
        checked += 1
        if result["is_ensemble"]:
            ensemble_checked += 1
        answer = result["answer"]
        note = result["note"]
        original = result["original"]
        if note == "vlm_disagreement":
            disagreements += 1
            notes.append({"line_id": line.get("id"), "note": "vlm_disagreement", "ocr_text": original})
            continue
        if note == "vlm_error":
            errors += 1
            continue
        if answer is not None and _looks_plausible(original, answer) and answer != original:
            line["ocr_text"] = original
            line["text"] = answer
            line["vlm_corrected"] = True
            corrected += 1

    result = dict(ocr_result)
    result["vlm_proofread"] = {
        "model": model,
        "threshold": threshold,
        "passes": passes,
        "parallelism": workers,
        "lines_checked": checked,
        "lines_corrected": corrected,
        "lines_disagreed": disagreements,
        "lines_errored": errors,
        "ensemble_disagreement_checked": ensemble_checked,
        "notes": notes,
    }
    return result
