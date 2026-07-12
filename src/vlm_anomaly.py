"""vlm_anomaly.py — optional VLM pass that flags rendered-output text defects.

Metric QA (SSIM / edge / recall) never *reads* the compiled ad, so it cannot see
that the same headline was rendered twice, that a CTA is clipped at the frame edge,
or that a wordmark came out as mojibake. This pass sends the rendered ``preview.png``
to a local vision-language model (LM Studio's OpenAI-compatible endpoint, same as the
other ``vlm_*`` stages) and asks for a STRUCTURED list of visible text anomalies:

  - duplicate_text  : the same words appear twice / ghosted / overlapping another layer
  - clipped_text    : text is cut off at a container or image edge (missing letters)
  - wrong_glyphs    : garbled / mojibake / clearly incorrect letters

Disabled by default (``vlm.anomaly.enabled: false``) and fails silent: any missing
file, stopped LM Studio, bad response, or parse error returns ``[]`` so the harness
never breaks. Calls are hard-capped (one full-preview pass by default).

The returned anomalies are turned into repairs by ``repair.repairs_from_anomalies`` and
persisted to ``anomalies.json`` so the next ``repair.assess`` picks them up and the
harness resumes the right stage (merge dedup for duplicates, text refit for clipped).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from src import vlm_client

_DEFAULT_MAX_TOKENS = 700
_DEFAULT_PASSES = 1
_DEFAULT_MAX_CALLS = 1

_TYPES = ("duplicate_text", "clipped_text", "wrong_glyphs")

_PROMPT = (
    "You are inspecting a rendered advertisement that was reconstructed from detected "
    "layers. Look ONLY for text rendering defects. Report each defect you actually see:\n"
    "- duplicate_text: the same words appear twice, ghosted, or one text overlaps another\n"
    "- clipped_text: text is cut off at a container or image edge (letters are missing)\n"
    "- wrong_glyphs: garbled/mojibake/incorrect letters (e.g. 'UPERONT' instead of 'UPFRONT')\n\n"
    "Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
    '{"anomalies":[{"type":"duplicate_text|clipped_text|wrong_glyphs",'
    '"text":"<the exact affected words>","detail":"<short note>"}]}\n'
    "Copy the affected words verbatim into 'text'. If there are no defects, reply "
    '{"anomalies":[]}.'
)

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["anomalies"],
    "properties": {
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "text", "detail"],
                "properties": {
                    "type": {"type": "string", "enum": list(_TYPES)},
                    "text": {"type": "string"},
                    "detail": {"type": "string"},
                },
            },
        }
    },
}


def _anomaly_cfg(cfg: dict) -> dict:
    """Merge the shared vlm settings with the vlm.anomaly overrides."""
    root = (cfg or {}).get("vlm") or {}
    anomaly = root.get("anomaly") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
    }
    merged.update({key: value for key, value in anomaly.items() if key != "enabled"})
    return merged


def enabled(cfg: Optional[dict]) -> bool:
    root = (cfg or {}).get("vlm") or {}
    anomaly = root.get("anomaly") or {}
    return bool(anomaly.get("enabled", False))


def _load_json(path: str, fallback: Any) -> Any:
    if not path or not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return fallback


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)


def _parse_anomalies(raw: str) -> list[dict]:
    """Parse the VLM answer into a validated anomaly list. Never raises."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data: Any = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(data, dict):
        items = data.get("anomalies")
    elif isinstance(data, list):
        items = data
    else:
        items = None
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    seen: set[tuple] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").strip().lower()
        if kind not in _TYPES:
            continue
        anomaly_text = str(item.get("text") or "").strip()
        if not anomaly_text:
            continue
        key = (kind, anomaly_text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "type": kind,
            "text": anomaly_text,
            "detail": str(item.get("detail") or "").strip(),
        })
    return out


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _flatten_layers(layers):
    out = []
    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        out.append(layer)
        out.extend(_flatten_layers(layer.get("children")))
    return out


def _attach_layer_ids(anomalies: list[dict], design: Optional[dict]) -> None:
    """Best-effort: match each anomaly's text to design text layers by their words."""
    if not design:
        return
    text_layers = [
        layer for layer in _flatten_layers(design.get("layers") or [])
        if layer.get("type") == "text" and layer.get("id")
    ]
    if not text_layers:
        return
    for anomaly in anomalies:
        needle = _norm_text(anomaly.get("text", ""))
        if not needle:
            continue
        matched = []
        for layer in text_layers:
            haystack = _norm_text(str(layer.get("text") or ""))
            if not haystack:
                continue
            if needle == haystack or needle in haystack or haystack in needle:
                matched.append(str(layer.get("id")))
        if matched:
            anomaly["layer_ids"] = matched


def detect_anomalies(
    run_dir: str,
    cfg: Optional[dict] = None,
    *,
    preview_path: Optional[str] = None,
    design: Optional[dict] = None,
    write: bool = True,
) -> list[dict]:
    """Inspect the rendered preview for text anomalies. Never raises; ``[]`` on failure."""
    try:
        return _detect(run_dir, cfg or {}, preview_path, design, write)
    except Exception:
        return []


def _detect(
    run_dir: str,
    cfg: dict,
    preview_path: Optional[str],
    design: Optional[dict],
    write: bool,
) -> list[dict]:
    if not enabled(cfg):
        return []
    run_dir = os.path.abspath(run_dir) if run_dir else run_dir
    acfg = _anomaly_cfg(cfg)

    if preview_path is None and run_dir:
        for candidate in ("preview.png", "figma_export.png"):
            path = os.path.join(run_dir, candidate)
            if os.path.exists(path):
                preview_path = path
                break
    if not preview_path or not os.path.exists(preview_path):
        return []

    try:
        with open(preview_path, "rb") as handle:
            image_bytes = handle.read()
    except OSError:
        return []
    if not image_bytes:
        return []

    base_url = str(acfg.get("base_url") or vlm_client._DEFAULT_BASE_URL)
    model = str(acfg.get("model") or vlm_client._DEFAULT_MODEL)
    timeout_s = float(acfg.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S)
    max_tokens = int(acfg.get("max_tokens") or _DEFAULT_MAX_TOKENS)
    passes = max(1, int(acfg.get("passes", _DEFAULT_PASSES)))
    max_calls = max(1, int(acfg.get("max_calls", _DEFAULT_MAX_CALLS)))
    passes = min(passes, max_calls)

    try:
        answer, note = vlm_client.multi_pass_answer(
            image_bytes,
            _PROMPT,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            passes=passes,
            response_schema=_SCHEMA,
        )
    except Exception:
        return []
    if note or answer is None:
        return []

    anomalies = _parse_anomalies(answer)

    if design is None and run_dir:
        design = _load_json(os.path.join(run_dir, "design.json"), None)
    _attach_layer_ids(anomalies, design if isinstance(design, dict) else None)

    if write and run_dir:
        try:
            _write_json(os.path.join(run_dir, "anomalies.json"), {
                "model": model,
                "preview": os.path.basename(preview_path),
                "passes": passes,
                "anomalies": anomalies,
            })
        except OSError:
            pass
    return anomalies


def load_anomalies(run_dir: str) -> list[dict]:
    """Read a previously-detected anomalies.json (list form). ``[]`` when absent."""
    data = _load_json(os.path.join(os.path.abspath(run_dir), "anomalies.json"), None)
    if isinstance(data, dict):
        items = data.get("anomalies")
    else:
        items = data
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []
