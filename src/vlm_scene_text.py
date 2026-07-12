"""vlm_scene_text.py — optional VLM classification for OCR text placement.

After text_analysis, crops each OCR line with surrounding context and asks a local
vision model whether the text is overlay copy, printed on a product, or a wordmark.
Sets ``meta.scene_text_role`` on lines (and blocks) for merge_layers routing. Disabled
by default and never raises.
"""
from __future__ import annotations

import copy
import json
import re

from src import vlm_client

_DEFAULT_PASSES = 2
_DEFAULT_MAX_TOKENS = 80
_DEFAULT_CONTEXT_PADDING = 24
_DEFAULT_MAX_LINES = 32

_ROLES = frozenset({"overlay_copy", "printed_on_product", "wordmark"})

_PROMPT = (
    "This crop shows text from a digital advertisement plus a little surrounding context. "
    "Classify how this text relates to the ad.\n\n"
    "Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
    '{"role": "overlay_copy"|"printed_on_product"|"wordmark"}\n\n'
    "- overlay_copy: editable ad copy overlaid on the layout (headlines, CTAs, prices, offers)\n"
    "- printed_on_product: text physically printed on a product, package, label, or device in a photo\n"
    "- wordmark: brand logo lettering or stylized mark, not regular body copy"
)

_ROLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role"],
    "properties": {"role": {"type": "string", "enum": sorted(_ROLES)}},
}


def _scene_cfg(cfg: dict) -> dict:
    root = (cfg or {}).get("vlm") or {}
    scene = root.get("scene_text") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
        "passes": root.get("passes"),
    }
    merged.update({k: v for k, v in scene.items() if k != "enabled"})
    return merged


def _parse_role(raw: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    role = str(data.get("role", "")).strip().lower()
    return role if role in _ROLES else None


def _propagate_to_blocks(lines: list[dict], blocks: list[dict]) -> None:
    by_id = {ln.get("id"): ln for ln in lines if ln.get("id")}
    for block in blocks:
        member_ids = block.get("line_ids") or []
        roles = []
        for line_id in member_ids:
            line = by_id.get(line_id)
            if not line:
                continue
            role = (line.get("meta") or {}).get("scene_text_role")
            if role:
                roles.append(role)
        if not roles:
            continue
        if any(role == "printed_on_product" for role in roles):
            chosen = "printed_on_product"
        elif all(role == "wordmark" for role in roles):
            chosen = "wordmark"
        elif all(role == "overlay_copy" for role in roles):
            chosen = "overlay_copy"
        else:
            chosen = roles[0]
        block.setdefault("meta", {})["scene_text_role"] = chosen


def classify_scene_text(image_path: str, ocr_result: dict, cfg: dict) -> dict:
    """Classify OCR line placement via optional VLM. Never raises."""
    scene = ((cfg or {}).get("vlm") or {}).get("scene_text") or {}
    if not scene.get("enabled", False):
        return ocr_result

    lines = list(ocr_result.get("lines") or [])
    if not lines:
        return ocr_result

    vcfg = _scene_cfg(cfg)
    base_url = str(vcfg.get("base_url") or vlm_client._DEFAULT_BASE_URL)
    model = str(vcfg.get("model") or vlm_client._DEFAULT_MODEL)
    timeout_s = float(vcfg.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S)
    max_tokens = int(vcfg.get("max_tokens") or _DEFAULT_MAX_TOKENS)
    padding = int(vcfg.get("context_padding", vcfg.get("padding", _DEFAULT_CONTEXT_PADDING)))
    passes = int(vcfg.get("passes") or _DEFAULT_PASSES)
    max_lines = vcfg.get("max_lines", _DEFAULT_MAX_LINES)

    candidates = [ln for ln in lines if ln.get("box")]
    if max_lines is not None:
        candidates = candidates[: int(max_lines)]

    try:
        from PIL import Image

        image = Image.open(image_path)
    except Exception:
        return ocr_result

    checked = 0
    classified = 0
    disagreements = 0
    errors = 0
    notes: list[dict] = []
    role_counts = {role: 0 for role in _ROLES}

    for line in candidates:
        crop = vlm_client.crop_box_bytes(image, line["box"], padding)
        if crop is None:
            continue
        checked += 1
        answer, note = vlm_client.multi_pass_answer(
            crop,
            _PROMPT,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            passes=passes,
            response_schema=_ROLE_SCHEMA,
        )
        if note == "vlm_disagreement":
            disagreements += 1
            notes.append({"line_id": line.get("id"), "note": "vlm_disagreement"})
            continue
        if note == "vlm_error":
            errors += 1
            continue
        role = _parse_role(answer or "")
        if role is None:
            continue
        line.setdefault("meta", {})["scene_text_role"] = role
        classified += 1
        role_counts[role] += 1
        notes.append({"line_id": line.get("id"), "role": role})

    result = copy.deepcopy(ocr_result)
    result["lines"] = lines
    blocks = list(result.get("blocks") or [])
    if blocks:
        _propagate_to_blocks(lines, blocks)
        result["blocks"] = blocks
    result["vlm_scene_text"] = {
        "enabled": True,
        "model": model,
        "context_padding": padding,
        "lines_checked": checked,
        "lines_classified": classified,
        "lines_disagreed": disagreements,
        "lines_errored": errors,
        "role_counts": role_counts,
        "notes": notes,
    }
    return result


__all__ = ["classify_scene_text"]
