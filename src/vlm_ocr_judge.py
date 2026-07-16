"""vlm_ocr_judge.py — optional VLM arbitration when OCR engines disagree.

When ensemble reconciliation flags ``meta.disagreement`` or challenger provenance
differs on a line, crops the region and asks a local vision model (Gemma via LM Studio)
to transcribe verbatim, showing engine readings A/B/C. Two agreeing passes replace the
fused text before low-confidence proofreading runs.

Optional ``ocr_read`` mode scans a coarse grid for text all engines missed; skipped when
the canvas exceeds ``max_image_pixels``. Disabled by default and never raises.
"""
from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher

from src import vlm_client

_DEFAULT_PASSES = 2
_DEFAULT_PADDING = 3
_DEFAULT_MAX_TOKENS = 500
_DEFAULT_GRID_COLS = 4
_DEFAULT_GRID_ROWS = 4
_DEFAULT_MAX_IMAGE_PIXELS = 4_000_000
_DEFAULT_MAX_OCR_READ_REGIONS = 8
# Disagreement arbitration budget — mirrors proofread.max_regions. Dense ads
# (session 002: 72 OCR lines, ~223s VLM) waste most of an uncapped pass on
# body/microcopy; high-impact lines are sorted first, then this cap applies.
_DEFAULT_MAX_DISAGREE_LINES = 12

# High-impact copy heuristics (aligned with text_analysis CTA/price/offer cues).
_CTA_RE = re.compile(
    r"\b(shop|buy|order|get|try|learn|discover|download|book|join|start|sign up|"
    r"subscribe|claim|apply|contact|swipe|tap|click)(\s+now|\s+today)?\b",
    re.IGNORECASE,
)
_PRICE_OR_OFFER_RE = re.compile(
    r"(?:[$€£¥]\s?\d|\d(?:[.,]\d{1,2})?\s?(?:usd|eur|gbp|dollars?|euros?)"
    r"|\b\d{1,3}\s?%|\bsave\b|\boff\b|\bfree\b)",
    re.IGNORECASE,
)

_DISAGREE_PROMPT_HEAD = (
    "This crop contains exactly ONE line of text from an ad. OCR engines read it "
    "differently:\n"
)
_DISAGREE_PROMPT_TAIL = (
    "\n\nTranscribe only that one line, character for character, preserving currency "
    "symbols (e.g. €), punctuation, and arrows (e.g. →) exactly as shown. Do not "
    "describe the image. Output only the transcribed line, nothing else, no explanation, "
    "no newlines. If no legible text is visible, output an empty string."
)

_OCR_READ_PROMPT = (
    "This crop is from a digital advertisement. If it contains legible text (even a "
    "single word or price), transcribe it character for character exactly as shown. "
    "Output only the transcribed text on one line, nothing else, no explanation. "
    "If there is no legible text, output an empty string."
)

_PROOFREAD_PROMPT = (
    "This crop contains exactly ONE short line of text from an ad (often a brand name, "
    "a headline word, or a price). Transcribe it character for character, preserving "
    "capitalization, currency symbols (e.g. €), punctuation, and arrows (e.g. →) exactly "
    "as shown. Read the letters that are actually there — do not substitute a more common "
    "word if the letters differ. Output only the transcribed line, nothing else, no "
    "explanation, no newlines. If no legible text is visible, output an empty string."
)


def _ocr_judge_cfg(cfg: dict) -> dict:
    root = (cfg or {}).get("vlm") or {}
    judge = root.get("ocr_judge") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
        "passes": root.get("passes"),
    }
    merged.update({k: v for k, v in judge.items()
                   if k not in {"enabled", "ocr_read", "proofread"}})
    return merged


def _ocr_read_cfg(cfg: dict) -> dict:
    judge = ((cfg or {}).get("vlm") or {}).get("ocr_judge") or {}
    read = judge.get("ocr_read") or {}
    if isinstance(read, bool):
        return {"enabled": read}
    return dict(read)


def _proofread_cfg(cfg: dict) -> dict:
    judge = ((cfg or {}).get("vlm") or {}).get("ocr_judge") or {}
    proof = judge.get("proofread") or {}
    if isinstance(proof, bool):
        return {"enabled": proof}
    return dict(proof)


def _looks_like_brand_token(text: str) -> bool:
    """Whether a line is a wordmark/brand-style token worth VLM proofing.

    Brand names and headline wordmarks are the tokens where a single-character OCR slip
    (P/H, O/0, I/l) is both most likely and most damaging, and — unlike body copy — they
    carry no dictionary context for the engine to self-correct.  They are short, (almost)
    all uppercase, and mostly letters.
    """
    stripped = str(text or "").strip()
    if not stripped or len(stripped.split()) > 2:
        return False
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 4:
        return False
    if len(letters) / len(re.sub(r"\s", "", stripped)) < 0.6:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) >= 0.8


def _looks_like_price_or_offer(text: str) -> bool:
    """Currency, percent-off, or SAVE/OFF/FREE phrasing — high-impact marketing copy."""
    return bool(_PRICE_OR_OFFER_RE.search(str(text or "")))


def _looks_like_cta(text: str) -> bool:
    """CTA verb phrases (Shop now / Get / Claim / …) worth prioritizing for arbitration."""
    return bool(_CTA_RE.search(str(text or "")))


def _is_high_impact_text(text: str) -> bool:
    return (
        _looks_like_brand_token(text)
        or _looks_like_price_or_offer(text)
        or _looks_like_cta(text)
    )


def _disagreement_priority(line: dict) -> tuple:
    """Sort key for capped disagreement arbitration (lower = sooner).

    Mirrors proofread's brand-then-area ordering, plus price/offer/CTA elevation.
    Fused text *and* engine readings are scanned so a garbage primary reading
    still ranks high when a challenger saw brand/price/CTA copy.
    """
    text = str(line.get("text") or "")
    readings = _engine_readings(line)
    impact = 0 if (
        _is_high_impact_text(text)
        or any(_is_high_impact_text(reading) for reading in readings)
    ) else 1
    box = line.get("box") or {}
    visual_area = float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
    # Prefer larger (more prominent) lines and wider engine splits next.
    return (impact, -visual_area, -len(readings), float(line.get("conf", 1.0) or 0.0))


def _cleanup_text(text: str) -> str:
    """Lazy OCR hygiene (case-smash + repeated tokens) without importing heavy OCR deps at module load."""
    from src.ocr import cleanup_line_text

    return cleanup_line_text(text)


def _deterministic_reading_fallback(original: str, readings: list[str]) -> str | None:
    """When VLM passes disagree, prefer a spaced/cleaned engine reading over a smashed primary.

    Ad 013: doctr ``WeNEVER`` vs easyocr ``We NEVER`` — VLM split on the two forms and left
    the smashed winner. Cleanup alone after a later stage is enough for smash, but preferring
    the already-spaced reading keeps provenance honest when arbitration aborts.
    """
    original = str(original or "")
    cleaned_original = _cleanup_text(original)
    spaced = [
        str(reading)
        for reading in readings
        if str(reading) and (" " in str(reading)) and (" " not in original)
        and _cleanup_text(str(reading)) == cleaned_original
    ]
    if spaced:
        # Prefer the spaced form that hygiene would emit.
        return max(spaced, key=lambda value: (value == cleaned_original, len(value)))
    if cleaned_original and cleaned_original != original:
        return cleaned_original
    return None


def _engine_readings(line: dict) -> list[str]:
    meta = line.get("meta") or {}
    disagreement = meta.get("disagreement")
    if isinstance(disagreement, list) and len(disagreement) > 1:
        return [str(value) for value in disagreement if str(value).strip()]

    provenance = meta.get("provenance") or []
    texts: list[str] = []
    seen: set[str] = set()
    for entry in provenance:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts if len(texts) > 1 else []


def _has_disagreement(line: dict) -> bool:
    return bool(_engine_readings(line))


def _disagreement_prompt(readings: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for index, text in enumerate(readings[: len(labels)]):
        lines.append(f'{labels[index]}: "{text}"')
    return _DISAGREE_PROMPT_HEAD + "\n".join(lines) + _DISAGREE_PROMPT_TAIL


def _looks_plausible(original: str, candidate: str, *, max_len_factor: float = 3.0) -> bool:
    if not candidate:
        return False
    if "\n" in candidate:
        return False
    if len(candidate) > max(40, len(original) * max_len_factor):
        return False
    return True


def _box_iou(a: dict, b: dict) -> float:
    ax0, ay0 = float(a.get("x", 0)), float(a.get("y", 0))
    ax1, ay1 = ax0 + float(a.get("w", 0)), ay0 + float(a.get("h", 0))
    bx0, by0 = float(b.get("x", 0)), float(b.get("y", 0))
    bx1, by1 = bx0 + float(b.get("w", 0)), by0 + float(b.get("h", 0))
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def _overlaps_existing(box: dict, lines: list[dict], min_iou: float = 0.08) -> bool:
    for line in lines:
        other = line.get("box") or {}
        if other and _box_iou(box, other) >= min_iou:
            return True
    return False


def _grid_boxes(width: int, height: int, cols: int, rows: int) -> list[dict]:
    boxes = []
    for row in range(max(1, rows)):
        for col in range(max(1, cols)):
            x0 = col * width // cols
            y0 = row * height // rows
            x1 = (col + 1) * width // cols
            y1 = (row + 1) * height // rows
            if x1 > x0 and y1 > y0:
                boxes.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return boxes


def _resolve_options(cfg: dict) -> dict:
    merged = _ocr_judge_cfg(cfg)
    read = _ocr_read_cfg(cfg)
    proof = _proofread_cfg(cfg)
    passes = int(merged.get("passes") or _DEFAULT_PASSES)
    # Cap disagreement judgements like proofread.max_regions. Explicit 0 disables
    # the path; omit/null falls back to the session-tuned default (not unlimited).
    raw_max_lines = merged.get("max_lines", _DEFAULT_MAX_DISAGREE_LINES)
    max_lines = _DEFAULT_MAX_DISAGREE_LINES if raw_max_lines is None else int(raw_max_lines)
    return {
        "base_url": str(merged.get("base_url") or vlm_client._DEFAULT_BASE_URL),
        "model": str(merged.get("model") or vlm_client._DEFAULT_MODEL),
        "timeout_s": float(merged.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S),
        "max_tokens": int(merged.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "padding": int(merged.get("padding") if merged.get("padding") is not None else _DEFAULT_PADDING),
        "passes": passes,
        "max_lines": max_lines,
        "parallelism": vlm_client.parallelism_from_cfg(cfg),
        "allow_novel_reading": bool(merged.get("allow_novel_reading", False)),
        "min_reading_similarity": float(merged.get("min_reading_similarity", .55)),
        "ocr_read_enabled": bool(read.get("enabled", False)),
        "ocr_read_cols": int(read.get("grid_cols") or _DEFAULT_GRID_COLS),
        "ocr_read_rows": int(read.get("grid_rows") or _DEFAULT_GRID_ROWS),
        "ocr_read_max_pixels": int(read.get("max_image_pixels") or _DEFAULT_MAX_IMAGE_PIXELS),
        "ocr_read_max_regions": int(read.get("max_regions") or _DEFAULT_MAX_OCR_READ_REGIONS),
        "ocr_read_passes": int(read.get("passes") or passes),
        "proofread_enabled": bool(proof.get("enabled", False)),
        # Single-engine OCR can be confidently wrong on high-impact ad copy.  A
        # 0.90 ceiling spends the bounded VLM budget on those near-misses (for
        # example one substituted word in a headline) before destructive text
        # removal is approved.
        "proofread_max_conf": float(proof.get("max_conf", 0.90)),
        "proofread_brand_tokens": bool(proof.get("brand_tokens", True)),
        "proofread_max_regions": int(proof.get("max_regions") or 8),
        "proofread_passes": int(proof.get("passes") or passes),
        "proofread_min_similarity": float(proof.get("min_similarity", 0.6)),
    }


def _judge_disagreements(
    image,
    lines: list[dict],
    options: dict,
) -> tuple[int, int, int, int, list[dict]]:
    # Skip lines already settled by numeric_verify (or a prior judge pass).
    candidates = [
        ln for ln in lines
        if ln.get("box") and _has_disagreement(ln) and not ln.get("vlm_ocr_judged")
    ]
    # Spend the bounded budget on brand / price / CTA / large lines first —
    # same lesson as proofread.max_regions priority sorting.
    candidates.sort(key=_disagreement_priority)
    max_lines = options["max_lines"]
    if max_lines is not None and max_lines >= 0:
        candidates = candidates[: int(max_lines)]

    checked = corrected = disagreements = errors = 0
    notes: list[dict] = []

    def _one(line: dict):
        readings = _engine_readings(line)
        crop = vlm_client.crop_box_bytes(image, line["box"], options["padding"])
        if crop is None:
            return None
        wider_crop = vlm_client.crop_box_bytes(image, line["box"], options["padding"] + 2)
        crop_variants = [crop] + ([wider_crop] if wider_crop and wider_crop != crop else [])
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _disagreement_prompt(readings),
            base_url=options["base_url"],
            model=options["model"],
            timeout_s=options["timeout_s"],
            max_tokens=options["max_tokens"],
            passes=options["passes"],
            crop_variants=crop_variants,
        )
        return {
            "line": line,
            "readings": readings,
            "crop_variants": len(crop_variants),
            "answer": answer,
            "note": note,
            "original": str(line.get("text", "")),
        }

    for result in vlm_client.map_parallel(
        _one, candidates, workers=options.get("parallelism", 1)
    ):
        if result is None:
            continue
        line = result["line"]
        readings = result["readings"]
        answer = result["answer"]
        note = result["note"]
        original = result["original"]
        checked += 1
        if note == "vlm_disagreement":
            fallback = _deterministic_reading_fallback(original, readings)
            if fallback and fallback != original:
                line["ocr_text"] = original
                line["text"] = fallback
                corrected += 1
                meta = copy.deepcopy(line.get("meta") or {})
                meta.pop("disagreement", None)
                meta["vlm_ocr_fallback"] = {
                    "reason": "vlm_disagreement_deterministic",
                    "from": original,
                    "to": fallback,
                    "readings": readings,
                }
                line["meta"] = meta
                line["vlm_ocr_judged"] = True
                notes.append({
                    "line_id": line.get("id"),
                    "note": "vlm_disagreement_fallback",
                    "readings": readings,
                    "ocr_text": original,
                    "fallback": fallback,
                })
                continue
            disagreements += 1
            notes.append({
                "line_id": line.get("id"),
                "note": "vlm_disagreement",
                "readings": readings,
                "ocr_text": original,
            })
            continue
        if note in vlm_client.VLM_ERROR_NOTES:
            errors += 1
            notes.append({"line_id": line.get("id"), "note": note, "ocr_text": original})
            continue
        if answer is not None and _looks_plausible(original, answer):
            # Hygiene after VLM: collapse ``do do this!`` and un-smash ``WeNEVER``.
            answer = _cleanup_text(answer)
            normalized_answer = answer.casefold().strip()
            similarity = max((SequenceMatcher(None, normalized_answer, reading.casefold().strip()).ratio()
                              for reading in readings), default=0.0)
            if not options["allow_novel_reading"] and similarity < options["min_reading_similarity"]:
                disagreements += 1
                notes.append({"line_id": line.get("id"), "note": "vlm_novel_reading",
                              "answer": answer, "readings": readings,
                              "similarity": round(similarity, 4)})
                continue
            if answer != original:
                line["ocr_text"] = original
                line["text"] = answer
                corrected += 1
            line["vlm_ocr_judged"] = True
            meta = copy.deepcopy(line.get("meta") or {})
            meta.pop("disagreement", None)
            meta["vlm_ocr_consensus"] = {"answer": answer, "passes": options["passes"],
                                         "crop_variants": result["crop_variants"],
                                         "reading_similarity": round(similarity, 4)}
            line["meta"] = meta
    return checked, corrected, disagreements, errors, notes


def _judge_uncertain(
    image,
    lines: list[dict],
    options: dict,
) -> tuple[int, int, int, list[dict]]:
    """Proofread low-confidence and brand-token lines that no engine disputed.

    A single-engine primary (docTR) produces no ``meta.disagreement`` to trigger the
    arbitration path, so a confident single-character misread on a wordmark (PINDAKAAS →
    HINDAKAAS) slips through.  This routes those lines through the same VLM, but — since
    there is only one reading to defend — accepts the answer only when it stays close to
    the OCR text (a character-level fix), never a wholesale rewrite."""
    if not options["proofread_enabled"]:
        return 0, 0, 0, []

    max_conf = options["proofread_max_conf"]
    want_brand = options["proofread_brand_tokens"]
    candidates: list[tuple[dict, bool]] = []
    for line in lines:
        if not line.get("box") or line.get("vlm_ocr_judged") or _has_disagreement(line):
            continue
        text = str(line.get("text", ""))
        low_conf = float(line.get("conf", 1.0) or 0.0) <= max_conf
        brandish = want_brand and _looks_like_brand_token(text)
        if low_conf or brandish:
            candidates.append((line, brandish))
    # Spend the bounded budget on brand/wordmark tokens first, then prominent
    # marketing copy.  A large headline with one bad leading glyph is much more
    # damaging than a tiny low-confidence ingredient line, especially when the
    # result will be rebuilt as editable text.
    def _priority(pair):
        line, brandish = pair
        box = line.get("box") or {}
        visual_area = float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
        return (0 if brandish else 1, -visual_area, float(line.get("conf", 1.0) or 0.0))
    candidates.sort(key=_priority)
    candidates = candidates[: options["proofread_max_regions"]]

    checked = corrected = errors = 0
    notes: list[dict] = []

    def _one(pair):
        line, brandish = pair
        crop = vlm_client.crop_box_bytes(image, line["box"], options["padding"])
        if crop is None:
            return None
        wider_crop = vlm_client.crop_box_bytes(image, line["box"], options["padding"] + 2)
        crop_variants = [crop] + ([wider_crop] if wider_crop and wider_crop != crop else [])
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _PROOFREAD_PROMPT,
            base_url=options["base_url"],
            model=options["model"],
            timeout_s=options["timeout_s"],
            max_tokens=options["max_tokens"],
            passes=options["proofread_passes"],
            crop_variants=crop_variants,
        )
        return {
            "line": line,
            "brandish": brandish,
            "answer": answer,
            "note": note,
            "original": str(line.get("text", "")),
        }

    for result in vlm_client.map_parallel(
        _one, candidates, workers=options.get("parallelism", 1)
    ):
        if result is None:
            continue
        line = result["line"]
        brandish = result["brandish"]
        answer = result["answer"]
        note = result["note"]
        original = result["original"]
        checked += 1
        if note in vlm_client.VLM_ERROR_NOTES:
            errors += 1
            notes.append({"line_id": line.get("id"), "note": note, "ocr_text": original})
            continue
        if note:
            notes.append({"line_id": line.get("id"), "note": note, "ocr_text": original})
            continue
        if answer is None or not _looks_plausible(original, answer):
            continue
        answer = _cleanup_text(answer)
        similarity = SequenceMatcher(
            None, answer.casefold().strip(), original.casefold().strip()
        ).ratio()
        if similarity < options["proofread_min_similarity"]:
            notes.append({"line_id": line.get("id"), "note": "vlm_low_similarity",
                          "answer": answer, "ocr_text": original,
                          "similarity": round(similarity, 4)})
            continue
        if answer != original:
            line["ocr_text"] = original
            line["text"] = answer
            corrected += 1
        line["vlm_ocr_judged"] = True
        meta = copy.deepcopy(line.get("meta") or {})
        meta["vlm_ocr_proofread"] = {
            "answer": answer,
            "passes": options["proofread_passes"],
            "reason": "brand-token" if brandish else "low-confidence",
            "similarity": round(similarity, 4),
        }
        line["meta"] = meta
    return checked, corrected, errors, notes


def _ocr_read_missed(
    image,
    lines: list[dict],
    options: dict,
) -> tuple[int, int, int, list[dict]]:
    width, height = image.size
    if width * height > options["ocr_read_max_pixels"]:
        return 0, 0, 0, []

    boxes = _grid_boxes(width, height, options["ocr_read_cols"], options["ocr_read_rows"])
    missed = [box for box in boxes if not _overlaps_existing(box, lines)]
    missed = missed[: options["ocr_read_max_regions"]]

    added = checked = errors = 0
    new_lines: list[dict] = []

    def _one(pair):
        index, box = pair
        crop = vlm_client.crop_box_bytes(image, box, 0)
        if crop is None:
            return None
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _OCR_READ_PROMPT,
            base_url=options["base_url"],
            model=options["model"],
            timeout_s=options["timeout_s"],
            max_tokens=options["max_tokens"],
            passes=options["ocr_read_passes"],
        )
        return {"index": index, "box": box, "answer": answer, "note": note}

    for result in vlm_client.map_parallel(
        _one, list(enumerate(missed)), workers=options.get("parallelism", 1)
    ):
        if result is None:
            continue
        checked += 1
        note = result["note"]
        answer = result["answer"]
        if note:
            if note in vlm_client.VLM_ERROR_NOTES:
                errors += 1
            continue
        if not answer or not _looks_plausible("", answer, max_len_factor=6.0):
            continue
        line_id = f"vlm-read-{result['index']}"
        new_lines.append({
            "id": line_id,
            "text": answer,
            "conf": 0.55,
            "box": result["box"],
            "vlm_ocr_read": True,
            "meta": {"source": "vlm-ocr-read"},
        })
        added += 1
    return checked, added, errors, new_lines


def judge_ocr_lines(image_path: str, ocr_result: dict, cfg: dict) -> dict:
    """Arbitrate OCR engine disagreements (and optionally recover missed grid text).

    Returns a copy of ``ocr_result`` with updated lines when the VLM agrees across passes.
    Never raises."""
    judge = ((cfg or {}).get("vlm") or {}).get("ocr_judge") or {}
    if not judge.get("enabled", False):
        return ocr_result

    lines = list(ocr_result.get("lines") or [])
    if not lines and not _ocr_read_cfg(cfg).get("enabled", False):
        return ocr_result

    try:
        from PIL import Image

        # Decode once on the caller thread before parallel crop workers share
        # the image.  This prevents intermittent broken-PNG/data-stream errors.
        image = Image.open(image_path).convert("RGB")
        image.load()
    except Exception:
        return ocr_result

    options = _resolve_options(cfg)
    checked, corrected, disagreements, errors, notes = _judge_disagreements(image, lines, options)

    proofread_checked = proofread_corrected = proofread_errors = 0
    if options["proofread_enabled"]:
        proofread_checked, proofread_corrected, proofread_errors, proofread_notes = _judge_uncertain(
            image, lines, options,
        )
        notes = notes + proofread_notes

    ocr_read_checked = ocr_read_added = ocr_read_errors = 0
    new_lines: list[dict] = []
    if options["ocr_read_enabled"]:
        ocr_read_checked, ocr_read_added, ocr_read_errors, new_lines = _ocr_read_missed(
            image, lines + new_lines, options,
        )
        lines.extend(new_lines)

    # Final hygiene pass: VLM answers and untouched lines both get case-smash /
    # repeated-token cleanup so ad-013 style ``WeNEVER`` / ``do do this!`` cannot ship.
    from src.ocr import _apply_line_text_cleanup

    lines = _apply_line_text_cleanup(lines)

    result = dict(ocr_result)
    result["lines"] = lines
    result["vlm_ocr_judge"] = {
        "model": options["model"],
        "passes": options["passes"],
        "max_lines": options["max_lines"],
        "lines_checked": checked,
        "lines_corrected": corrected,
        "lines_disagreed": disagreements,
        "lines_errored": errors,
        "proofread_enabled": options["proofread_enabled"],
        "proofread_checked": proofread_checked,
        "proofread_corrected": proofread_corrected,
        "proofread_errored": proofread_errors,
        "ocr_read_enabled": options["ocr_read_enabled"],
        "ocr_read_checked": ocr_read_checked,
        "ocr_read_added": ocr_read_added,
        "ocr_read_errored": ocr_read_errors,
        "notes": notes,
    }
    return result
