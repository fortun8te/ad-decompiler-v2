"""vlm_segment_filter.py — optional VLM crop review for fused SAM3 elements.

After mask-aware fusion, each canonical element can be cropped from the normalized ad
and sent to a local vision-language model for a keep/drop/review decision. Two independent
passes must agree before a decision is applied. Disabled by default and never raises:
LM Studio outages or parse failures leave elements unchanged.
"""
from __future__ import annotations

import copy
import json
import re

from src import vlm_client

_DEFAULT_PADDING = 8
_DEFAULT_PASSES = 2
_DEFAULT_MAX_TOKENS = 120
_DEFAULT_REFINE_MAX_TOKENS = 80

_LABELS = frozenset({
    "product",
    "person",
    "text_artifact",
    "background_bleed",
    "icon",
    "button",
    "badge",
    "junk",
})
_DECISIONS = frozenset({"keep", "drop", "review"})
_REFINE_ROLES = frozenset({"button", "icon", "product"})
_DROP_LABELS = frozenset({"text_artifact", "background_bleed", "junk"})
_KEEP_LABELS = frozenset({"product", "person", "icon", "button", "badge"})

_PROMPT = (
    "This crop shows one segmented element from a digital advertisement. "
    "Classify whether it is a real ad element worth preserving.\n\n"
    "Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
    '{"decision": "keep"|"drop"|"review", "label": "<label>"}\n\n'
    "Labels: product, person, text_artifact, background_bleed, icon, button, badge, junk\n\n"
    "- keep: real semantic element (product, person, icon, button, badge)\n"
    "- drop: segmentation noise (junk, background bleed, stray text artifact)\n"
    "- review: uncertain or borderline"
)

_REFINE_PROMPT = (
    "This crop shows one segmented element from a digital advertisement that was classified "
    "as worth preserving.\n\n"
    "Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
    '{"role": "button"|"icon"|"product"}\n\n'
    "- button: clickable CTA shell or pill-shaped control\n"
    "- icon: small graphic, badge, logo mark, or pictogram\n"
    "- product: physical product, package, bottle, jar, tube, or device"
)

_CLASSIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["decision", "label"],
    "properties": {
        "decision": {"type": "string", "enum": sorted(_DECISIONS)},
        "label": {"type": "string", "enum": sorted(_LABELS)},
    },
}
_REFINE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["role"],
    "properties": {"role": {"type": "string", "enum": sorted(_REFINE_ROLES)}},
}


def _vlm_cfg(cfg: dict) -> dict:
    root = (cfg or {}).get("vlm") or {}
    seg = root.get("segment_filter") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
    }
    merged.update({k: v for k, v in seg.items() if k != "enabled"})
    return merged


def _parse_classification(raw: str) -> dict | None:
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
    decision = str(data.get("decision", "")).strip().lower()
    label = str(data.get("label", "")).strip().lower()
    if decision not in _DECISIONS or label not in _LABELS:
        return None
    # Judge semantic consistency in addition to JSON shape. A model response such as
    # {"decision":"drop","label":"product"} must not delete a real asset merely
    # because repeated deterministic calls reproduced the same contradiction.
    if (decision == "drop" and label not in _DROP_LABELS) or (
        decision == "keep" and label not in _KEEP_LABELS
    ):
        return {"decision": "review", "label": label, "judge": "decision-label-conflict"}
    return {"decision": decision, "label": label}


def _parse_refine_role(raw: str) -> str | None:
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
    return role if role in _REFINE_ROLES else None


def _annotate(element: dict, *, decision: str, label: str, note: str | None = None) -> dict:
    out = copy.deepcopy(element)
    meta = dict(out.get("meta") or {})
    meta["vlm_segment"] = {"decision": decision, "label": label}
    if note:
        meta["vlm_segment"]["note"] = note
    out["meta"] = meta
    if decision == "review":
        meta["vlm_uncertain"] = True
    if decision == "drop":
        meta["vlm_rejected"] = True
    return out


def _refine_role(
    crop: bytes,
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    passes: int,
) -> tuple[str | None, str | None]:
    answer, note = vlm_client.multi_pass_answer(
        crop,
        _REFINE_PROMPT,
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        passes=passes,
        response_schema=_REFINE_SCHEMA,
    )
    if note:
        return None, note
    return _parse_refine_role(answer or ""), None


def filter_elements(image_path: str, elements: list[dict], cfg: dict) -> list[dict]:
    """Return fused elements after optional VLM crop filtering. Never raises."""
    seg = ((cfg or {}).get("vlm") or {}).get("segment_filter") or {}
    if not seg.get("enabled", False):
        return elements
    if not elements:
        return elements

    vcfg = _vlm_cfg(cfg)
    base_url = str(vcfg.get("base_url") or vlm_client._DEFAULT_BASE_URL)
    model = str(vcfg.get("model") or vlm_client._DEFAULT_MODEL)
    timeout_s = float(vcfg.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S)
    max_tokens = int(vcfg.get("max_tokens") or _DEFAULT_MAX_TOKENS)
    padding = int(vcfg.get("padding", _DEFAULT_PADDING))
    passes = int(vcfg.get("passes", _DEFAULT_PASSES))
    max_elements = vcfg.get("max_elements")
    reject_mode = str(vcfg.get("reject_mode", "remove")).strip().lower()
    refine_cfg = seg.get("refine_role") or {}
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_passes = int(refine_cfg.get("passes", passes))
    refine_max_tokens = int(refine_cfg.get("max_tokens", _DEFAULT_REFINE_MAX_TOKENS))

    candidates = [el for el in elements if el.get("box")]
    if max_elements is not None:
        candidates = candidates[: int(max_elements)]

    try:
        from PIL import Image

        image = Image.open(image_path)
    except Exception:
        return elements

    candidate_ids = {id(el) for el in candidates}
    workers = vlm_client.parallelism_from_cfg(cfg)

    def _classify_one(element: dict) -> dict | None:
        """Return annotated element, or None to drop (reject_mode=remove)."""
        try:
            crop = vlm_client.crop_box_bytes(image, element["box"], padding)
        except (KeyError, TypeError, ValueError, OverflowError):
            return _annotate(
                element, decision="review", label="unknown", note="invalid_crop_geometry"
            )
        if crop is None:
            return _annotate(
                element, decision="review", label="unknown", note="invalid_crop_geometry"
            )

        answer, note = vlm_client.multi_pass_answer(
            crop,
            _PROMPT,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            passes=passes,
            response_schema=_CLASSIFY_SCHEMA,
        )

        if note == "vlm_error":
            return _annotate(element, decision="review", label="unknown", note="vlm_error")
        if note == "vlm_disagreement":
            return _annotate(element, decision="review", label="junk", note="vlm_disagreement")

        parsed = _parse_classification(answer or "")
        if parsed is None:
            return _annotate(element, decision="review", label="unknown", note="vlm_parse_error")

        decision = parsed["decision"]
        label = parsed["label"]
        if decision == "drop":
            if reject_mode == "mark":
                return _annotate(element, decision=decision, label=label)
            return None
        if decision == "review":
            return _annotate(element, decision=decision, label=label)

        kept = _annotate(element, decision=decision, label=label)
        if refine_enabled:
            refined, refine_note = _refine_role(
                crop,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_tokens=refine_max_tokens,
                passes=refine_passes,
            )
            if refined:
                kept.setdefault("meta", {})["role"] = refined
                kept["meta"].setdefault("vlm_segment", {})["refined_role"] = refined
            elif refine_note == "vlm_disagreement":
                kept["meta"].setdefault("vlm_segment", {})["refine_note"] = "vlm_disagreement"
        return kept

    # Preserve original order: non-candidates pass through; candidates classified
    # in parallel then re-merged by index.
    classify_jobs = [el for el in elements if id(el) in candidate_ids]
    classified = {
        id(el): result
        for el, result in zip(
            classify_jobs,
            vlm_client.map_parallel(_classify_one, classify_jobs, workers=workers),
        )
    }
    results: list[dict] = []
    for element in elements:
        if id(element) not in candidate_ids:
            results.append(element)
            continue
        kept = classified.get(id(element))
        if kept is not None:
            results.append(kept)
    return results
