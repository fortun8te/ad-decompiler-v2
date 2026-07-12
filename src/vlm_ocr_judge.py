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

from src import vlm_client

_DEFAULT_PASSES = 2
_DEFAULT_PADDING = 3
_DEFAULT_MAX_TOKENS = 500
_DEFAULT_GRID_COLS = 4
_DEFAULT_GRID_ROWS = 4
_DEFAULT_MAX_IMAGE_PIXELS = 4_000_000
_DEFAULT_MAX_OCR_READ_REGIONS = 8

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
    merged.update({k: v for k, v in judge.items() if k not in {"enabled", "ocr_read"}})
    return merged


def _ocr_read_cfg(cfg: dict) -> dict:
    judge = ((cfg or {}).get("vlm") or {}).get("ocr_judge") or {}
    read = judge.get("ocr_read") or {}
    if isinstance(read, bool):
        return {"enabled": read}
    return dict(read)


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
    passes = int(merged.get("passes") or _DEFAULT_PASSES)
    return {
        "base_url": str(merged.get("base_url") or vlm_client._DEFAULT_BASE_URL),
        "model": str(merged.get("model") or vlm_client._DEFAULT_MODEL),
        "timeout_s": float(merged.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S),
        "max_tokens": int(merged.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "padding": int(merged.get("padding") if merged.get("padding") is not None else _DEFAULT_PADDING),
        "passes": passes,
        "max_lines": merged.get("max_lines"),
        "ocr_read_enabled": bool(read.get("enabled", False)),
        "ocr_read_cols": int(read.get("grid_cols") or _DEFAULT_GRID_COLS),
        "ocr_read_rows": int(read.get("grid_rows") or _DEFAULT_GRID_ROWS),
        "ocr_read_max_pixels": int(read.get("max_image_pixels") or _DEFAULT_MAX_IMAGE_PIXELS),
        "ocr_read_max_regions": int(read.get("max_regions") or _DEFAULT_MAX_OCR_READ_REGIONS),
        "ocr_read_passes": int(read.get("passes") or passes),
    }


def _judge_disagreements(
    image,
    lines: list[dict],
    options: dict,
) -> tuple[int, int, int, int, list[dict]]:
    candidates = [ln for ln in lines if ln.get("box") and _has_disagreement(ln)]
    max_lines = options["max_lines"]
    if max_lines is not None:
        candidates = candidates[: int(max_lines)]

    checked = corrected = disagreements = errors = 0
    notes: list[dict] = []
    for line in candidates:
        readings = _engine_readings(line)
        crop = vlm_client.crop_box_bytes(image, line["box"], options["padding"])
        if crop is None:
            continue
        checked += 1
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _disagreement_prompt(readings),
            base_url=options["base_url"],
            model=options["model"],
            timeout_s=options["timeout_s"],
            max_tokens=options["max_tokens"],
            passes=options["passes"],
        )
        original = str(line.get("text", ""))
        if note == "vlm_disagreement":
            disagreements += 1
            notes.append({
                "line_id": line.get("id"),
                "note": "vlm_disagreement",
                "readings": readings,
                "ocr_text": original,
            })
            continue
        if note == "vlm_error":
            errors += 1
            continue
        if answer is not None and _looks_plausible(original, answer) and answer != original:
            line["ocr_text"] = original
            line["text"] = answer
            line["vlm_ocr_judged"] = True
            meta = copy.deepcopy(line.get("meta") or {})
            meta.pop("disagreement", None)
            line["meta"] = meta
            corrected += 1
    return checked, corrected, disagreements, errors, notes


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
    for index, box in enumerate(missed):
        crop = vlm_client.crop_box_bytes(image, box, 0)
        if crop is None:
            continue
        checked += 1
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _OCR_READ_PROMPT,
            base_url=options["base_url"],
            model=options["model"],
            timeout_s=options["timeout_s"],
            max_tokens=options["max_tokens"],
            passes=options["ocr_read_passes"],
        )
        if note:
            if note == "vlm_error":
                errors += 1
            continue
        if not answer or not _looks_plausible("", answer, max_len_factor=6.0):
            continue
        line_id = f"vlm-read-{index}"
        new_lines.append({
            "id": line_id,
            "text": answer,
            "conf": 0.55,
            "box": copy.deepcopy(box),
            "meta": {"source": "vlm_ocr_read", "vlm_ocr_read": True},
            "vlm_ocr_read": True,
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

        image = Image.open(image_path)
    except Exception:
        return ocr_result

    options = _resolve_options(cfg)
    checked, corrected, disagreements, errors, notes = _judge_disagreements(image, lines, options)

    ocr_read_checked = ocr_read_added = ocr_read_errors = 0
    new_lines: list[dict] = []
    if options["ocr_read_enabled"]:
        ocr_read_checked, ocr_read_added, ocr_read_errors, new_lines = _ocr_read_missed(
            image, lines + new_lines, options,
        )
        lines.extend(new_lines)

    result = dict(ocr_result)
    result["lines"] = lines
    result["vlm_ocr_judge"] = {
        "model": options["model"],
        "passes": options["passes"],
        "lines_checked": checked,
        "lines_corrected": corrected,
        "lines_disagreed": disagreements,
        "lines_errored": errors,
        "ocr_read_enabled": options["ocr_read_enabled"],
        "ocr_read_checked": ocr_read_checked,
        "ocr_read_added": ocr_read_added,
        "ocr_read_errored": ocr_read_errors,
        "notes": notes,
    }
    return result
