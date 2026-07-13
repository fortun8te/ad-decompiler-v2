"""vlm_scene_text.py — optional VLM classification for OCR text placement.

After text_analysis, crops each OCR line with surrounding context and asks a local
vision model whether the text is overlay copy, printed on a product, or a wordmark.
Sets ``meta.scene_text_role`` on lines (and blocks) for merge_layers routing. Disabled
by default and never raises.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

from src import vlm_client

_DEFAULT_PASSES = 2
_DEFAULT_MAX_TOKENS = 80
_DEFAULT_CONTEXT_PADDING = 24
_DEFAULT_MAX_LINES = 32
_DEFAULT_MAX_TOTAL_S = 90

_ROLES = frozenset({"overlay_copy", "printed_on_product", "wordmark"})
_PLACEMENTS = frozenset({"overlay", "printed", "ui_metadata", "artifact"})
_OWNERS = frozenset({"background", "photo", "product", "card", "none"})
_ACTIONS = frozenset({"recreate", "raster_keep", "remove"})

_PROMPT = (
    "You are deciding whether detected text in an advertisement may be safely removed and "
    "rebuilt as editable text. The image is either the full ad or a padded close crop. "
    "Use surrounding photo, product, card, and UI context. Be conservative.\n\n"
    "Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
    '{"placement":"overlay"|"printed"|"ui_metadata"|"artifact",'
    '"owner":"background"|"photo"|"product"|"card"|"none",'
    '"action":"recreate"|"raster_keep"|"remove","confidence":0.0}\n\n'
    "- recreate only for deliberate overlay copy whose glyphs should become editable.\n"
    "- raster_keep for text printed on or contained by a photo, product, screenshot card, "
    "logo/wordmark, or when ownership is uncertain.\n"
    "- remove only for a definite OCR artifact that is not visible text.\n"
    "If unsure, choose raster_keep."
)

_ROLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["placement", "owner", "action", "confidence"],
    "properties": {
        "placement": {"type": "string", "enum": sorted(_PLACEMENTS)},
        "owner": {"type": "string", "enum": sorted(_OWNERS)},
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
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


def _parse_ownership(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    placement = str(data.get("placement", "")).strip().lower()
    owner = str(data.get("owner", "")).strip().lower()
    action = str(data.get("action", "")).strip().lower()
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return None
    if placement not in _PLACEMENTS or owner not in _OWNERS or action not in _ACTIONS:
        return None
    # A model may contradict itself. Protected ownership always wins over a claimed
    # recreate action; uncertainty must never become a destructive mask.
    if owner in {"photo", "product", "card"} and action == "recreate":
        action = "raster_keep"
    return {"placement": placement, "owner": owner, "action": action,
            "confidence": round(confidence, 4)}


def _scene_context_bytes(image, box: dict, max_side: int = 768) -> bytes:
    import io
    from PIL import ImageDraw
    thumb = image.convert("RGB").copy()
    scale = min(1.0, float(max_side) / max(1, max(thumb.size)))
    thumb.thumbnail((max_side, max_side))
    try:
        x0 = float(box.get("x", 0)) * scale
        y0 = float(box.get("y", 0)) * scale
        x1 = x0 + float(box.get("w", 0)) * scale
        y1 = y0 + float(box.get("h", 0)) * scale
        ImageDraw.Draw(thumb).rectangle((x0, y0, x1, y1), outline=(255, 32, 32),
                                        width=max(2, int(round(4 * scale))))
    except (TypeError, ValueError):
        pass
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return buf.getvalue()


def _propagate_to_blocks(lines: list[dict], blocks: list[dict]) -> None:
    by_id = {ln.get("id"): ln for ln in lines if ln.get("id")}
    for block in blocks:
        member_ids = block.get("line_ids") or []
        roles = []
        ownership = []
        for line_id in member_ids:
            line = by_id.get(line_id)
            if not line:
                continue
            role = (line.get("meta") or {}).get("scene_text_role")
            if role:
                roles.append(role)
            decision = (line.get("meta") or {}).get("ownership_decision")
            if isinstance(decision, dict):
                ownership.append(decision)
        if ownership:
            # Blocks are only recreated when every constituent line is independently
            # safe, or a strong paragraph-level majority agrees and no line has a real
            # protected owner. This lets a 3-line post survive one ambiguous crop while
            # still protecting package/card/photo text absolutely.
            recreates = [item for item in ownership if item.get("action") == "recreate"]
            has_protected_owner = any(item.get("owner") in {"photo", "product", "card"}
                                      for item in ownership)
            strong_recreate_majority = len(recreates) >= max(1, (2 * len(ownership) + 2) // 3)
            if recreates and not has_protected_owner and strong_recreate_majority:
                chosen_decision = dict(recreates[0])
                chosen_decision["confidence"] = min(float(x.get("confidence", 0)) for x in recreates)
                chosen_decision["block_consensus"] = f"{len(recreates)}/{len(ownership)}"
            else:
                protected = next((x for x in ownership if x.get("action") == "raster_keep"), ownership[0])
                chosen_decision = dict(protected)
                chosen_decision["action"] = "raster_keep"
            block.setdefault("meta", {})["ownership_decision"] = chosen_decision
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
    # Scene ownership is advisory.  A slow local VLM must never prevent the
    # full decomposition from completing: unreviewed lines safely stay raster
    # owned, rather than being guessed as editable copy.
    max_total_s = float(vcfg.get("max_total_s", _DEFAULT_MAX_TOTAL_S))
    deadline = time.monotonic() + max(1.0, max_total_s)

    candidates = [ln for ln in lines if ln.get("box")]
    if max_lines is not None:
        candidates = candidates[: int(max_lines)]

    try:
        from PIL import Image

        image = Image.open(image_path)
    except Exception:
        return ocr_result
    cache_path = None
    cache: dict[str, list[str]] = {}
    prior_decisions: dict[tuple, dict] = {}
    run_dir = str((cfg or {}).get("run_dir") or "")
    if run_dir:
        cache_path = os.path.join(run_dir, "vlm_scene_text_cache.json")
        try:
            with open(cache_path, encoding="utf-8") as handle:
                loaded_cache = json.load(handle)
            if isinstance(loaded_cache, dict):
                cache = loaded_cache
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        try:
            with open(os.path.join(run_dir, "ocr.json"), encoding="utf-8") as handle:
                prior_ocr = json.load(handle)
            for prior in (prior_ocr.get("lines") or []):
                prior_box = prior.get("box") or {}
                key = (str(prior.get("text") or ""),) + tuple(
                    round(float(prior_box.get(name, 0)), 1) for name in ("x", "y", "w", "h")
                )
                decision = (prior.get("meta") or {}).get("ownership_decision")
                if isinstance(decision, dict) and decision.get("action") in _ACTIONS:
                    prior_decisions[key] = decision
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    checked = 0
    classified = 0
    disagreements = 0
    errors = 0
    notes: list[dict] = []
    role_counts = {role: 0 for role in _ROLES}

    for line in candidates:
        if time.monotonic() >= deadline:
            notes.append({"note": "vlm_budget_exhausted", "lines_remaining": len(candidates) - checked})
            break
        crop = vlm_client.crop_box_bytes(image, line["box"], padding)
        if crop is None:
            continue
        checked += 1
        scene_context = _scene_context_bytes(
            image, line["box"], int(vcfg.get("scene_max_side", 768)),
        )
        prompt = (_PROMPT + "\nThe text being classified is " + repr(str(line.get("text") or "")) +
                  ". In the full-ad view it is enclosed by a red rectangle.")
        cache_key = json.dumps({
            "version": 2, "model": model, "text": str(line.get("text") or ""),
            "box": {key: round(float(line["box"].get(key, 0)), 2)
                    for key in ("x", "y", "w", "h")}, "passes": passes,
        }, sort_keys=True, separators=(",", ":"))
        observation_key = (str(line.get("text") or ""),) + tuple(
            round(float(line["box"].get(name, 0)), 1) for name in ("x", "y", "w", "h")
        )
        prior_decision = prior_decisions.get(observation_key)
        raw_answers = ([json.dumps(prior_decision)] if prior_decision
                       else list(cache.get(cache_key) or []))
        note = None
        if not raw_answers:
            try:
                variants = [scene_context, crop] if passes >= 2 else [crop]
                for variant in variants:
                    raw_answers.append(vlm_client.ask_vlm(
                        variant, prompt, base_url=base_url, model=model,
                        timeout_s=timeout_s, max_tokens=max_tokens,
                        response_schema=_ROLE_SCHEMA,
                    ))
                cache[cache_key] = list(raw_answers)
                if cache_path:
                    temporary = cache_path + ".tmp"
                    with open(temporary, "w", encoding="utf-8") as handle:
                        json.dump(cache, handle, indent=2)
                    os.replace(temporary, cache_path)
            except Exception:
                note = "vlm_error"
        parsed_answers = [_parse_ownership(value) for value in raw_answers]
        if note is None and (not parsed_answers or any(value is None for value in parsed_answers)):
            note = "vlm_error"
        if note is None:
            actions = {value["action"] for value in parsed_answers}
            protected_owners = {value["owner"] for value in parsed_answers
                                if value["owner"] in {"photo", "product", "card"}}
            # Placement wording is advisory: a screenshot body can reasonably be called
            # both printed and overlay. Destructive safety depends on the agreed action;
            # any protected owner still vetoes recreation.
            if len(actions) != 1 or ("recreate" in actions and protected_owners):
                note = "vlm_disagreement"
        answer = None
        decision = None
        if note is None:
            decision = dict(parsed_answers[0])
            if protected_owners:
                decision["owner"] = sorted(protected_owners)[0]
                decision["action"] = "raster_keep"
            placements = {value["placement"] for value in parsed_answers}
            for conservative in ("artifact", "printed", "ui_metadata", "overlay"):
                if conservative in placements:
                    decision["placement"] = conservative
                    break
            decision["confidence"] = min(float(value["confidence"]) for value in parsed_answers)
        if note == "vlm_disagreement":
            disagreements += 1
            notes.append({"line_id": line.get("id"), "note": "vlm_disagreement"})
            line.setdefault("meta", {})["ownership_decision"] = {
                "placement": "artifact", "owner": "none", "action": "raster_keep",
                "confidence": 0.0, "reason": "vlm_disagreement",
            }
            continue
        if note == "vlm_error":
            errors += 1
            line.setdefault("meta", {})["ownership_decision"] = {
                "placement": "artifact", "owner": "none", "action": "raster_keep",
                "confidence": 0.0, "reason": "vlm_error",
            }
            continue
        if decision is None:
            decision = _parse_ownership(answer or "")
        if decision is None:
            legacy_role = _parse_role(answer or "")
            if legacy_role:
                decision = {
                    "placement": "overlay" if legacy_role == "overlay_copy" else "printed",
                    "owner": "none" if legacy_role == "overlay_copy" else "product",
                    "action": "recreate" if legacy_role == "overlay_copy" else "raster_keep",
                    "confidence": 1.0,
                }
        if decision is None:
            line.setdefault("meta", {})["ownership_decision"] = {
                "placement": "artifact", "owner": "none", "action": "raster_keep",
                "confidence": 0.0, "reason": "vlm_parse_error",
            }
            continue
        line.setdefault("meta", {})["ownership_decision"] = decision
        role = ("overlay_copy" if decision["action"] == "recreate" else
                "printed_on_product" if decision["owner"] in {"photo", "product", "card"} else
                "wordmark" if decision["placement"] == "printed" else None)
        if role is None:
            notes.append({"line_id": line.get("id"), "ownership": decision})
            classified += 1
            continue
        line.setdefault("meta", {})["scene_text_role"] = role
        classified += 1
        role_counts[role] += 1
        notes.append({"line_id": line.get("id"), "role": role, "ownership": decision})

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
        "budget_s": max_total_s,
        "role_counts": role_counts,
        "notes": notes,
    }
    return result


__all__ = ["classify_scene_text"]
