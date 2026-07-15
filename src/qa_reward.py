"""qa_reward.py — Phase-2 metric-ladder reward for the harness loop.

The legacy harness reward was ``composite = 0.75·SSIM + 0.25·text_recall`` — the repo's
own docs show it misleading on dark/text-heavy ads (ad9 oscillated 0.87/0.5 ↔ 0.37/0.79,
docs/HANDOFF-2026-07-13.md §4). Following docs/HARNESS-PHASE2.md and gap P1-6
(docs/RESEARCH-CODIA-GAP-ANALYSIS.md), the reward is now a ladder:

  rung 0  structural hard fails            gate — cannot be bought (semantics unchanged)
  rung 1  per-element local scores         consumes pixel_diff ``per_layer`` rows
                                           (region_ssim / ink_iou), duck-typed
  rung 2  LPIPS global perceptual score    robust where global SSIM misleads (dark UIs)
  rung 3  text recall                      existing QA fields (strictest of the two)
  rung 4  VLM critique                     structured [{element, issue, suggested_fix}]
                                           — the primary repair driver, never a score

:func:`compute_reward` returns the scalar the loop uses for best-kept/rollback/plateau;
:func:`acceptance_gate` is a strictly-additional check (it can only make acceptance
stricter, never looser — the never-lower-thresholds invariant). All VLM/LPIPS paths fail
silent: any missing file, missing dependency, stopped LM Studio, or parse error degrades
to the deterministic metrics so the harness never breaks.

Config (all NEW keys; existing keys' semantics untouched)::

  runtime:
    harness:
      reward: phase2            # phase2 (default) | legacy composite scoring
  qa:
    reward:
      weights: {local_ssim: .4, lpips: .3, text: .3}   # optional explicit override
      lpips: {enabled: true, net: squeeze, max_edge: 256}
      lpips_similarity_min: 0.20    # acceptance-gate floor (archetype preset may raise)
      local_ssim_min: 0.30          # acceptance-gate floor (archetype preset may raise)
      critique: {enabled: false, max_items: 8, timeout_s: 45, max_tokens: 900}

``qa.reward_weights`` and ``qa.archetype_thresholds.{lpips_similarity_min,
reward_local_ssim_min}`` are populated per archetype by ``archetype.apply_preset``.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.request
from typing import Any, Optional

from src import vlm_client

# Weight defaults. Dark/text-heavy creatives (tweets, UI screenshots) are exactly where
# global perceptual scores mislead, so text correctness + per-element local SSIM dominate
# and LPIPS is demoted to a sanity term (HARNESS-PHASE2 §1c: "a tweet must NOT need 0.90
# visual SSIM; it must need high text_score + correct structure").
_DEFAULT_WEIGHTS = {"local_ssim": 0.40, "lpips": 0.30, "text": 0.30}
_DARK_TEXT_WEIGHTS = {"local_ssim": 0.40, "lpips": 0.15, "text": 0.45}
_WEIGHT_KEYS = ("local_ssim", "lpips", "text")

# Gate floors are anti-degenerate guards, not quality bars: quality acceptance stays with
# qa.ok / visual_pass_ssim. They block runs whose perceptual or per-element evidence is
# catastrophically worse than the global score claims (reward hacking / degenerate output).
#
# F12 recalibration — the old floors (0.20 / 0.30) were toothless. Measured on the review's
# fixtures, the known-BAD 002 (whole product cluster erased by inpaint, then QA-failed only
# by luck of the SSIM threshold) reads LPIPS-similarity 0.732 and reward local-SSIM 0.465,
# while every known-OK run (009/013/052) clusters at LPIPS 0.976-0.995 / local-SSIM
# 0.60-0.74. The old floors cleared 002 by 2-3x, so the gate rejected nothing. These
# defaults sit inside that gap so a degraded reconstruction is refused on BOTH gates while
# every good run passes with margin; per-archetype floors (archetype.PRESETS) override them
# and are the values actually used when a preset is applied.
_DEFAULT_LPIPS_SIMILARITY_MIN = 0.80
_DEFAULT_LOCAL_SSIM_MIN = 0.50

# F12 content penalty (docs/CRITIC-REVIEW-2026-07-15.md F12c). The scalar reward must not
# read "globally plausible" when content is erased or headlines are rasterized. The metrics
# agent (pixel_diff) writes honest ``rasterized_text_ratio`` and ``native_leaf_ratio`` into
# qa.json (top level or under ``structural``); we read them defensively and subtract a
# bounded penalty so a screenshot-plausible-but-non-editable run scores LOW, not high.
# Both terms are zero when the field is absent, so runs predating the metrics change (and
# every existing unit fixture) are unaffected.
_RASTERIZED_TEXT_FLOOR = 0.50      # above this fraction of text-as-pixels starts costing
_RASTERIZED_TEXT_WEIGHT = 0.30
_NATIVE_LEAF_FLOOR = 0.30          # mirrors pixel_diff's low-native-leaf gate
_NATIVE_LEAF_WEIGHT = 0.50
_CONTENT_PENALTY_CAP = 0.15        # never dominates the ladder; ~one hard-fail's weight

_DEFAULT_LPIPS_NET = "squeeze"     # ~5 MB backbone; CPU-capable. "alex" opt-in via config.
_DEFAULT_LPIPS_MAX_EDGE = 256

# CODIA CONSTRUCTION CONTRACT (docs/CODIA-PARITY-SPEC.md). Repairs must chase editability,
# not pixels: the contract score (native text ratio + font policy + clean plate + placement,
# computed by pixel_diff / scripts.codia_parity) DOMINATES the metric ladder. When a contract
# score is present it is blended in at this weight ABOVE the LPIPS/SSIM ladder, so a
# high-SSIM run with baked text scores LOW and a native-text Codia-shaped run scores HIGH.
# The 3-key ladder (local_ssim/lpips/text) is unchanged and still used for the remainder;
# runs predating the contract fields (every existing fixture) carry no contract score and are
# scored exactly as before.
_CONSTRUCTION_WEIGHT = 0.55

_HARD_FAIL_PENALTY = 0.12          # mirrors harness_loop._qa_score

_DEFAULT_CRITIQUE_MAX_ITEMS = 8
_DEFAULT_CRITIQUE_TIMEOUT_S = 45
_DEFAULT_CRITIQUE_MAX_TOKENS = 900

_LPIPS_CACHE: dict[str, Any] = {}

_CRITIQUE_PROMPT = (
    "You are auditing a RECONSTRUCTION (second image) against the ORIGINAL advertisement "
    "(first image). The goal is a Codia-style rebuild where EVERY string is crisp EDITABLE "
    "TEXT, everything else is a clean image cutout, and each element sits in the right place. "
    "So the failures that matter most are EDITABILITY and PLACEMENT — check these FIRST and "
    "report every one you find:\n"
    "- NOT EDITABLE TEXT / RASTERIZED TEXT: a headline or label that is baked into an image "
    "or sliced as pixels instead of being crisp editable text (this is the #1 defect).\n"
    "- DOUBLE-PRINTED / GHOST / DUPLICATE text: the same string appears twice, doubled, or as "
    "a faint echo (often a text node printed on top of a plate that still shows the old pixels).\n"
    "- MISPLACED / MISALIGNED: an element shifted, offset, or in the wrong position vs the original.\n"
    "- WRONG FONT: letters in a clearly different typeface (Codia uses Inter for UI/body text).\n"
    "- MISSING / ERASED element: a product, photo, logo, or text block present in the original "
    "but absent, blank, or replaced by a gray/blurred smear in the reconstruction.\n"
    "Then report finer defects: text clipped/cut off, wrong color, background artifact.\n\n"
    "For each difference report:\n"
    "- element: the affected element; copy its visible text verbatim when it has text\n"
    "- issue: what is wrong; name the failure type (missing element, erased content, ghost text, "
    "duplicated text, wrong font, rasterized text, clipped, wrong color, misaligned, artifact)\n"
    "- severity: 'high' for missing/erased/ghost/wrong-font/rasterized structural failures, "
    "'medium' for clipped/color/alignment, 'low' for cosmetic\n"
    "- suggested_fix: one short imperative fix\n\n"
    "Do NOT report 'none' or downplay a large missing region — an erased product cluster is the "
    "single most important thing to flag. Reply with ONLY valid JSON on one line, no markdown:\n"
    '{"critique":[{"element":"...","issue":"...","severity":"high","suggested_fix":"..."}]}\n'
    'If the reconstruction truly matches the original, reply {"critique":[]}.'
)

_CRITIQUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["critique"],
    "properties": {
        "critique": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["element", "issue", "severity", "suggested_fix"],
                "properties": {
                    "element": {"type": "string"},
                    "issue": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "suggested_fix": {"type": "string"},
                },
            },
        }
    },
}

# Critique → existing repair-action vocabulary (every pair is in harness.ACTIONABLE, so
# recommended_resume / config_patches_for drive the same resume stages as metric repairs).
# Order matters: specific text defects match before the generic "missing" → sam3 rule.
_CRITIQUE_RULES: tuple = (
    (("duplicat", "ghost", "twice", "double", "repeated"),
     "merge", "dedup", "high"),
    (("clip", "cut off", "cut-off", "cutoff", "truncat", "cropped"),
     "text-analysis", "refit-text-box", "high"),
    # Rasterized / baked-in headlines are present but non-editable — restore them as text
    # before the generic font rule claims them (F12b: sliced headlines are a top defect).
    (("rasteriz", "rasterised", "baked into", "baked in", "not editable", "pixels instead",
      "flattened text", "image instead of text", "text is an image"),
     "text-analysis", "restore-editable-text", "high"),
    (("mojibake", "garbl", "glyph", "font", "typeface", "letterform", "misspel"),
     "text-analysis", "resolve-fonts", "medium"),
    (("missing text", "text is missing", "text missing", "no text", "absent text",
      "text not present"),
     "text-analysis", "restore-editable-text", "high"),
    (("color", "colour", "hue", "tint", "shade", "darker", "lighter", "contrast"),
     "text-analysis", "refit-colors-effects", "medium"),
    (("misalign", "align", "position", "offset", "shifted", "spacing", "overlap"),
     "layout", "refit-geometry", "medium"),
    (("halo", "seam", "smudge", "blurry patch", "artifact", "background", "inpaint"),
     "inpaint", "rebuild-clean-plate", "high"),
    # Erased/vanished structural content (the 002 class): re-detect it. "erased", "gone",
    # "vanish", "smear" are the words a sharpened critic uses for an inpainted-away product.
    (("missing", "absent", "not present", "disappear", "lost", "omitted", "erased",
      "erase", "gone", "vanish", "smear", "wiped out", "removed entirely"),
     "sam3", "rerun-detection", "high"),
)


# ── config accessors ─────────────────────────────────────────────────────────────────


def reward_mode(cfg: Optional[dict] = None) -> str:
    """``runtime.harness.reward``: ``phase2`` (default) or ``legacy``."""
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    mode = str(harness.get("reward") or "phase2").strip().lower()
    return mode if mode in ("phase2", "legacy") else "phase2"


def _reward_cfg(cfg: Optional[dict]) -> dict:
    qa = (cfg or {}).get("qa") or {}
    reward = qa.get("reward")
    return reward if isinstance(reward, dict) else {}


def _critique_cfg(cfg: Optional[dict]) -> dict:
    critique = _reward_cfg(cfg).get("critique")
    return critique if isinstance(critique, dict) else {}


def critique_enabled(cfg: Optional[dict]) -> bool:
    """VLM critique is opt-in (same posture as vlm.anomaly): config must enable it."""
    return bool(_critique_cfg(cfg).get("enabled", False))


def reward_weights(cfg: Optional[dict] = None) -> dict:
    """Resolve component weights: explicit override > archetype preset > facts default."""
    qa = (cfg or {}).get("qa") or {}
    for candidate in (_reward_cfg(cfg).get("weights"), qa.get("reward_weights")):
        normalized = _normalized_weights(candidate)
        if normalized:
            return normalized
    facts = ((cfg or {}).get("scene") or {}).get("facts") or {}
    try:
        text_heavy = int(facts.get("text_line_count") or 0) >= 8
    except (TypeError, ValueError):
        text_heavy = False
    if facts.get("dark_background") or text_heavy:
        return dict(_DARK_TEXT_WEIGHTS)
    return dict(_DEFAULT_WEIGHTS)


def _normalized_weights(candidate: Any) -> Optional[dict]:
    if not isinstance(candidate, dict):
        return None
    out = {}
    for key in _WEIGHT_KEYS:
        try:
            value = float(candidate.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0.0
        out[key] = max(0.0, value)
    return out if sum(out.values()) > 0 else None


def gate_thresholds(cfg: Optional[dict] = None) -> dict:
    """Acceptance-gate floors; archetype thresholds win over qa.reward defaults."""
    qa = (cfg or {}).get("qa") or {}
    archetype = qa.get("archetype_thresholds") or {}
    reward = _reward_cfg(cfg)

    def _floor(archetype_key: str, reward_key: str, default: float) -> float:
        for value in (archetype.get(archetype_key), reward.get(reward_key)):
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return default

    return {
        "lpips_similarity_min": _floor(
            "lpips_similarity_min", "lpips_similarity_min", _DEFAULT_LPIPS_SIMILARITY_MIN),
        "local_ssim_min": _floor(
            "reward_local_ssim_min", "local_ssim_min", _DEFAULT_LOCAL_SSIM_MIN),
    }


# ── shared io helpers (module-local by repo convention) ──────────────────────────────


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


def _resolve_source(run_dir: str) -> Optional[str]:
    candidate = os.path.join(run_dir, "normalized.png")
    if os.path.exists(candidate):
        return candidate
    report = _load_json(os.path.join(run_dir, "runtime_report.json"), {})
    source = report.get("input") if isinstance(report, dict) else None
    return str(source) if source and os.path.exists(str(source)) else None


def _resolve_render(run_dir: str) -> Optional[str]:
    for name in ("preview.png", "figma_export.png"):
        candidate = os.path.join(run_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None


# ── rung 1: per-element local scores (consumes pixel_diff per_layer rows) ────────────


def local_component(qa: Optional[dict]) -> Optional[dict]:
    """Per-element local score from qa.per_layer rows (duck-typed, missing-key safe).

    Rows are produced by pixel_diff (region_ssim / ink_iou / region_px and, for text,
    reconstruction ssim/recall). Text rows fold rendered-ink IoU in so ghost/duplicate
    ink lowers the element even when crop SSIM stays high. Falls back to the multiscale
    local ``ssim`` field when no usable row exists.
    """
    qa = qa or {}
    rows = qa.get("per_layer")
    scored: list[tuple[str, float, float]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        value = row.get("region_ssim")
        if not isinstance(value, (int, float)):
            value = row.get("ssim")
        if not isinstance(value, (int, float)):
            value = row.get("score")
        if not isinstance(value, (int, float)):
            continue
        value = max(0.0, min(1.0, float(value)))
        ink_iou = row.get("ink_iou")
        region_color = row.get("region_color")
        if isinstance(ink_iou, (int, float)):
            # Text rows: shift-aligned rendered-ink IoU exposes ghost/duplicate glyphs
            # that crop SSIM alone can miss.
            value = 0.7 * value + 0.3 * max(0.0, min(1.0, float(ink_iou)))
        elif isinstance(region_color, (int, float)):
            # Shape/image rows: local colour similarity catches recolour drift.
            value = 0.85 * value + 0.15 * max(0.0, min(1.0, float(region_color)))
        pixels = row.get("region_px")
        weight = math.sqrt(float(pixels)) if isinstance(pixels, (int, float)) and pixels > 0 else 1.0
        scored.append((str(row.get("id") or "unnamed"), round(value, 4), weight))

    if scored:
        total = sum(weight for _, _, weight in scored)
        mean = sum(value * weight for _, value, weight in scored) / max(total, 1e-9)
        ordered = sorted(value for _, value, _ in scored)
        p10 = ordered[min(len(ordered) - 1, int(0.1 * (len(ordered) - 1)))]
        # Same robust blend pixel_diff uses per scale: the lower tail stays diagnostic.
        score = 0.7 * mean + 0.3 * p10
        worst = sorted(scored, key=lambda item: item[1])[:4]
        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "mean": round(mean, 4),
            "p10": round(p10, 4),
            "count": len(scored),
            "worst": [{"id": lid, "score": value} for lid, value, _ in worst],
            "source": "per_layer",
        }

    ssim = qa.get("ssim")
    if isinstance(ssim, (int, float)):
        return {"score": round(max(0.0, min(1.0, float(ssim))), 4),
                "count": 0, "source": "multiscale_ssim"}
    return None


# ── rung 2: LPIPS global perceptual score ─────────────────────────────────────────────


def lpips_enabled(cfg: Optional[dict]) -> bool:
    lpips_cfg = _reward_cfg(cfg).get("lpips")
    if not isinstance(lpips_cfg, dict):
        return True
    return bool(lpips_cfg.get("enabled", True))


def lpips_score(source_path: Optional[str], render_path: Optional[str],
                cfg: Optional[dict] = None) -> Optional[dict]:
    """LPIPS perceptual distance between source and render. ``None`` on any failure.

    CPU-capable: images are downscaled to ``qa.reward.lpips.max_edge`` and the small
    ``squeeze`` backbone is the default. The network is cached per process so repeated
    harness rounds pay the load cost once.
    """
    try:
        if not source_path or not render_path:
            return None
        if not os.path.exists(source_path) or not os.path.exists(render_path):
            return None
        if not lpips_enabled(cfg):
            return None
        lpips_cfg = _reward_cfg(cfg).get("lpips") or {}
        net_name = str(lpips_cfg.get("net") or _DEFAULT_LPIPS_NET)
        max_edge = int(lpips_cfg.get("max_edge") or _DEFAULT_LPIPS_MAX_EDGE)

        import numpy as np
        import torch
        from PIL import Image

        model = _LPIPS_CACHE.get(net_name)
        if model is None:
            import lpips as lpips_mod
            model = lpips_mod.LPIPS(net=net_name, verbose=False)
            model.eval()
            _LPIPS_CACHE[net_name] = model

        with Image.open(source_path) as image:
            source = image.convert("RGB")
            scale = min(1.0, max_edge / max(1, *source.size))
            if scale < 1.0:
                source = source.resize(
                    (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            with Image.open(render_path) as render_image:
                render = render_image.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
            tensors = []
            for img in (source, render):
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0))
        with torch.no_grad():
            distance = float(model(tensors[0], tensors[1]).item())
        return {
            "distance": round(distance, 6),
            "similarity": round(max(0.0, min(1.0, 1.0 - distance)), 6),
            "net": net_name,
            "max_edge": max_edge,
        }
    except Exception:
        return None


# ── rung 0.5: construction contract (the objective — native text > pixels) ────────────


def construction_component(qa: Optional[dict]) -> Optional[dict]:
    """Codia construction-contract score for the ladder (0..1), or None.

    Prefers pixel_diff's ``contract.contract_score`` (native text first, then construction
    quality, then a small SSIM floor term). Falls back to the raw construction score, then
    to native_text_ratio alone. Duck-typed and missing-key safe: absent → None, so runs
    predating the contract fields are scored by the legacy ladder untouched.
    """
    qa = qa or {}
    contract = qa.get("contract") if isinstance(qa.get("contract"), dict) else {}
    structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
    score = contract.get("contract_score")
    source = "contract_score"
    if not isinstance(score, (int, float)):
        construction = (qa.get("construction") or contract.get("construction")
                        or structural.get("construction"))
        if isinstance(construction, dict) and isinstance(construction.get("score"), (int, float)):
            score = float(construction["score"]) / 100.0
            source = "construction"
    if not isinstance(score, (int, float)):
        for src in (qa, contract, structural):
            value = src.get("native_text_ratio")
            if isinstance(value, (int, float)):
                score = float(value)
                source = "native_text_ratio"
                break
    if not isinstance(score, (int, float)):
        return None
    detail = {"score": round(max(0.0, min(1.0, float(score))), 4), "source": source}
    ntr = contract.get("native_text_ratio")
    if not isinstance(ntr, (int, float)):
        ntr = qa.get("native_text_ratio")
    if isinstance(ntr, (int, float)):
        detail["native_text_ratio"] = round(float(ntr), 4)
    if "pass" in contract:
        detail["contract_pass"] = bool(contract.get("pass"))
    return detail


# ── rung 3: text correctness (existing QA fields) ─────────────────────────────────────


def text_component(qa: Optional[dict]) -> Optional[float]:
    """Strictest available recall — editable-text loss must not hide behind OCR recall."""
    qa = qa or {}
    values = [
        float(qa[key]) for key in ("text_recall", "editable_text_recall")
        if isinstance(qa.get(key), (int, float))
    ]
    return round(min(values), 4) if values else None


# ── the ladder: scalar reward + acceptance gate ──────────────────────────────────────


def _structural(qa: Optional[dict]) -> dict:
    structural = (qa or {}).get("structural")
    return structural if isinstance(structural, dict) else {}


def _first_number(qa: dict, *keys: str) -> Optional[float]:
    """First numeric value for any of *keys* at qa top level or under qa.structural."""
    structural = _structural(qa)
    for source in (qa, structural):
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def content_penalty(qa: Optional[dict]) -> dict:
    """Bounded reward penalty for rasterized text / low native-leaf ratio (F12c).

    Reads the metrics agent's honest fields defensively: absent field → zero term, so
    a globally-plausible reconstruction that erased content or rasterized its headlines
    scores LOW once those fields exist, while runs without them are unchanged.
    """
    qa = qa or {}
    terms: dict[str, float] = {}
    rasterized = _first_number(qa, "rasterized_text_ratio")
    if rasterized is not None and rasterized > _RASTERIZED_TEXT_FLOOR:
        terms["rasterized_text"] = round(
            _RASTERIZED_TEXT_WEIGHT * (min(1.0, rasterized) - _RASTERIZED_TEXT_FLOOR), 6)
    native_leaf = _first_number(qa, "native_leaf_ratio")
    if native_leaf is not None and native_leaf < _NATIVE_LEAF_FLOOR:
        terms["low_native_leaf"] = round(
            _NATIVE_LEAF_WEIGHT * (_NATIVE_LEAF_FLOOR - max(0.0, native_leaf)), 6)
    total = min(_CONTENT_PENALTY_CAP, sum(terms.values()))
    return {"total": round(total, 6), "terms": terms,
            "rasterized_text_ratio": rasterized, "native_leaf_ratio": native_leaf}


def _hard_fail_count(qa: dict) -> int:
    fails = qa.get("hard_fails")
    count = len(fails) if isinstance(fails, list) else 0
    structural = qa.get("structural")
    if isinstance(structural, dict):
        seen = {
            (item.get("rule"), item.get("detail"))
            for item in (fails if isinstance(fails, list) else [])
            if isinstance(item, dict)
        }
        for item in structural.get("hard_fails") or []:
            if isinstance(item, dict) and (item.get("rule"), item.get("detail")) not in seen:
                count += 1
    return count


def compute_reward(run_dir: str, cfg: Optional[dict] = None, *,
                   qa: Optional[dict] = None,
                   source_path: Optional[str] = None,
                   render_path: Optional[str] = None) -> dict:
    """Metric-ladder reward for one round. Never raises; ``score`` is None when unknown."""
    try:
        return _compute_reward(run_dir, cfg or {}, qa, source_path, render_path)
    except Exception as exc:
        return {"mode": reward_mode(cfg), "score": None, "components": {},
                "error": f"{type(exc).__name__}: {exc}"}


def _compute_reward(run_dir, cfg, qa, source_path, render_path) -> dict:
    run_dir = os.path.abspath(run_dir) if run_dir else run_dir
    if qa is None:
        qa = _load_json(os.path.join(run_dir, "qa.json"), {}) if run_dir else {}
    qa = qa if isinstance(qa, dict) else {}

    local = local_component(qa)
    lpips = None
    if run_dir or (source_path and render_path):
        source = source_path or (_resolve_source(run_dir) if run_dir else None)
        render = render_path or (_resolve_render(run_dir) if run_dir else None)
        lpips = lpips_score(source, render, cfg)
    text = text_component(qa)

    weights = reward_weights(cfg)
    available: dict[str, float] = {}
    if local is not None and isinstance(local.get("score"), (int, float)):
        available["local_ssim"] = float(local["score"])
    if lpips is not None and isinstance(lpips.get("similarity"), (int, float)):
        available["lpips"] = float(lpips["similarity"])
    if text is not None:
        available["text"] = float(text)

    construction = construction_component(qa)
    hard_fails = _hard_fail_count(qa)
    penalty = content_penalty(qa)
    score = None
    ladder = None
    if available:
        weight_total = sum(weights.get(name, 0.0) for name in available)
        if weight_total > 0:
            ladder = sum(weights.get(name, 0.0) * value for name, value in available.items())
            ladder /= weight_total
        else:
            ladder = sum(available.values()) / len(available)
    # Contract-first: when a construction-contract score exists it DOMINATES the ladder, so
    # repairs chase editability (native text, clean plate, placement) over LPIPS/SSIM pixels.
    if construction is not None:
        contract_score = float(construction["score"])
        score = (contract_score if ladder is None
                 else _CONSTRUCTION_WEIGHT * contract_score + (1.0 - _CONSTRUCTION_WEIGHT) * ladder)
    else:
        score = ladder
    if score is not None:
        score -= _HARD_FAIL_PENALTY * hard_fails
        score -= penalty["total"]
        score = round(max(0.0, min(1.0, score)), 6)

    return {
        "mode": reward_mode(cfg),
        "score": score,
        "components": {"construction": construction, "local_ssim": local,
                       "lpips": lpips, "text": text},
        "weights": {**weights, "construction": _CONSTRUCTION_WEIGHT},
        "archetype": ((cfg or {}).get("scene") or {}).get("archetype"),
        "hard_fails": hard_fails,
        "content_penalty": penalty,
    }


def acceptance_gate(run_dir: str, cfg: Optional[dict] = None, *,
                    qa: Optional[dict] = None, reward: Optional[dict] = None) -> dict:
    """Strictly-additional acceptance check on LPIPS + per-element local SSIM.

    It can only make acceptance stricter (an anti-degenerate guard on top of qa.ok /
    hard-fail gating), never looser; when a metric is unavailable its check is skipped,
    so behaviour is unchanged where LPIPS or per-layer rows do not exist.
    """
    if reward_mode(cfg) != "phase2":
        return {"ok": True, "skipped": "legacy"}
    try:
        if not isinstance(reward, dict):
            reward = compute_reward(run_dir, cfg, qa=qa)
        floors = gate_thresholds(cfg)
        components = reward.get("components") or {}
        checks: dict[str, dict] = {}
        ok = True

        local = components.get("local_ssim") or {}
        if isinstance(local, dict) and isinstance(local.get("score"), (int, float)):
            passed = float(local["score"]) >= floors["local_ssim_min"]
            checks["local_ssim"] = {"value": local["score"],
                                    "min": floors["local_ssim_min"], "ok": passed}
            ok = ok and passed

        lpips = components.get("lpips") or {}
        if isinstance(lpips, dict) and isinstance(lpips.get("similarity"), (int, float)):
            passed = float(lpips["similarity"]) >= floors["lpips_similarity_min"]
            checks["lpips_similarity"] = {"value": lpips["similarity"],
                                          "min": floors["lpips_similarity_min"], "ok": passed}
            ok = ok and passed

        if not checks:
            return {"ok": True, "skipped": "no_metrics"}
        return {"ok": ok, "checks": checks}
    except Exception as exc:
        return {"ok": True, "skipped": f"gate_error:{type(exc).__name__}"}


def reward_evidence(reward: Optional[dict]) -> Optional[dict]:
    """Compact per-round record for harness_loop.json / runtime_report.json trails."""
    if not isinstance(reward, dict):
        return None
    components = reward.get("components") or {}
    out: dict[str, Any] = {"score": reward.get("score"), "mode": reward.get("mode")}
    construction = components.get("construction")
    if isinstance(construction, dict) and isinstance(construction.get("score"), (int, float)):
        out["construction"] = construction["score"]
        if "native_text_ratio" in construction:
            out["native_text_ratio"] = construction["native_text_ratio"]
    local = components.get("local_ssim")
    if isinstance(local, dict) and isinstance(local.get("score"), (int, float)):
        out["local_ssim"] = local["score"]
    lpips = components.get("lpips")
    if isinstance(lpips, dict) and isinstance(lpips.get("similarity"), (int, float)):
        out["lpips_similarity"] = lpips["similarity"]
    text = components.get("text")
    if isinstance(text, (int, float)):
        out["text"] = text
    if reward.get("hard_fails"):
        out["hard_fails"] = reward["hard_fails"]
    penalty = reward.get("content_penalty")
    if isinstance(penalty, dict) and penalty.get("total"):
        out["content_penalty"] = penalty["total"]
        if penalty.get("terms"):
            out["content_penalty_terms"] = penalty["terms"]
    return out


# ── rung 4: structured VLM critique (original + preview → repair driver) ─────────────


def _critique_vlm_settings(cfg: Optional[dict]) -> dict:
    """Merge shared vlm settings with qa.reward.critique overrides (vlm_anomaly pattern)."""
    root = (cfg or {}).get("vlm") or {}
    critique = _critique_cfg(cfg)
    return {
        "base_url": str(critique.get("base_url") or root.get("base_url")
                        or vlm_client._DEFAULT_BASE_URL),
        "model": str(critique.get("model") or root.get("model") or vlm_client._DEFAULT_MODEL),
        "timeout_s": float(critique.get("timeout_s") or root.get("timeout_s")
                           or _DEFAULT_CRITIQUE_TIMEOUT_S),
        "max_tokens": int(critique.get("max_tokens") or _DEFAULT_CRITIQUE_MAX_TOKENS),
        "max_items": max(1, int(critique.get("max_items", _DEFAULT_CRITIQUE_MAX_ITEMS))),
    }


def _ask_vlm_pair(source_bytes: bytes, render_bytes: bytes, prompt: str, *,
                  base_url: str, model: str, timeout_s: float, max_tokens: int,
                  response_schema: Optional[dict] = None) -> str:
    """Two-image chat completion against the local OpenAI-compatible endpoint.

    Mirrors :func:`vlm_client.ask_vlm` (single hard-capped call, reasoning disabled,
    strict json_schema) but attaches original + reconstruction in one user turn so the
    model can actually diff them (ReLook/UI2Code^N: the judge must SEE both).
    """
    content = [{"type": "text", "text": prompt}]
    for blob in (source_bytes, render_bytes):
        encoded = base64.b64encode(blob).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded}})
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max(max_tokens, vlm_client._MIN_MAX_TOKENS),
        "temperature": 0.0,
        "reasoning_effort": "none",
    }
    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "vlm_result", "strict": True, "schema": response_schema},
        }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    if data.get("error"):
        raise vlm_client.VLMError(f"VLM error response: {data['error']}")
    message = (data.get("choices") or [{}])[0].get("message", {})
    answer = vlm_client._message_text(message.get("content"))
    if not answer:
        raise vlm_client.VLMError("VLM critique returned no content")
    return answer


def _parse_critique(raw: str, max_items: int) -> list[dict]:
    """Parse the VLM answer into validated critique items. Never raises."""
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
    items = data.get("critique") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    seen: set[tuple] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        element = str(item.get("element") or "").strip()
        issue = str(item.get("issue") or "").strip()
        fix = str(item.get("suggested_fix") or "").strip()
        if not issue:
            continue
        key = (element.lower(), issue.lower())
        if key in seen:
            continue
        seen.add(key)
        parsed = {"element": element, "issue": issue, "suggested_fix": fix}
        severity = str(item.get("severity") or "").strip().lower()
        if severity in ("high", "medium", "low"):
            parsed["severity"] = severity
        out.append(parsed)
        if len(out) >= max_items:
            break
    return out


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _strictest_severity(*values: str) -> str:
    """Highest of the given severities (unknown/empty ignored); default 'medium'."""
    ranked = [v for v in values if v in _SEVERITY_RANK]
    return max(ranked, key=_SEVERITY_RANK.__getitem__) if ranked else "medium"


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


def _attach_layer_ids(items: list[dict], design: Optional[dict]) -> None:
    """Best-effort: match each item's element string to design layer ids or text."""
    if not isinstance(design, dict):
        return
    layers = [layer for layer in _flatten_layers(design.get("layers") or []) if layer.get("id")]
    if not layers:
        return
    by_id = {str(layer["id"]).lower(): str(layer["id"]) for layer in layers}
    for item in items:
        needle = _norm_text(item.get("element", ""))
        if not needle:
            continue
        direct = by_id.get(needle)
        if direct:
            item["layer_ids"] = [direct]
            continue
        matched = []
        for layer in layers:
            if layer.get("type") != "text":
                continue
            haystack = _norm_text(str(layer.get("text") or ""))
            if not haystack:
                continue
            if needle == haystack or needle in haystack or haystack in needle:
                matched.append(str(layer["id"]))
        if matched:
            item["layer_ids"] = matched


def run_critique(run_dir: str, cfg: Optional[dict] = None, *,
                 source_path: Optional[str] = None,
                 preview_path: Optional[str] = None,
                 design: Optional[dict] = None,
                 write: bool = True) -> dict:
    """One capped VLM critique call (original + preview). Never raises.

    Returns ``{"items": [...], "model": ...}`` — ``items`` empty (with ``error`` set)
    on any failure so the harness degrades to tool-only repairs when LM Studio is down
    or the model was evicted for the Flux inpaint stage.
    """
    try:
        return _run_critique(run_dir, cfg or {}, source_path, preview_path, design, write)
    except Exception as exc:
        return {"items": [], "error": f"{type(exc).__name__}: {exc}"}


def _run_critique(run_dir, cfg, source_path, preview_path, design, write) -> dict:
    if not critique_enabled(cfg):
        return {"items": [], "skipped": "disabled"}
    run_dir = os.path.abspath(run_dir) if run_dir else run_dir
    source = source_path or (_resolve_source(run_dir) if run_dir else None)
    preview = preview_path or (_resolve_render(run_dir) if run_dir else None)
    if not source or not preview:
        return {"items": [], "error": "missing_artifacts"}
    try:
        with open(source, "rb") as handle:
            source_bytes = handle.read()
        with open(preview, "rb") as handle:
            preview_bytes = handle.read()
    except OSError:
        return {"items": [], "error": "unreadable_artifacts"}
    if not source_bytes or not preview_bytes:
        return {"items": [], "error": "empty_artifacts"}

    settings = _critique_vlm_settings(cfg)
    try:
        answer = _ask_vlm_pair(
            source_bytes, preview_bytes, _CRITIQUE_PROMPT,
            base_url=settings["base_url"], model=settings["model"],
            timeout_s=settings["timeout_s"], max_tokens=settings["max_tokens"],
            response_schema=_CRITIQUE_SCHEMA,
        )
    except Exception as exc:
        return {"items": [], "error": f"vlm_unavailable: {type(exc).__name__}"}

    items = _parse_critique(answer, settings["max_items"])
    if design is None and run_dir:
        design = _load_json(os.path.join(run_dir, "design.json"), None)
    _attach_layer_ids(items, design if isinstance(design, dict) else None)

    result = {
        "items": items,
        "model": settings["model"],
        "source": os.path.basename(source),
        "preview": os.path.basename(preview),
    }
    if write and run_dir:
        try:
            _write_json(os.path.join(run_dir, "qa_critique.json"), result)
        except OSError:
            pass
    return result


def critique_to_repairs(items: Optional[list], design: Optional[dict] = None) -> list[dict]:
    """Map critique items onto the existing repair-action vocabulary.

    Every emitted (stage, action) is in ``harness.ACTIONABLE``, so the loop resumes the
    same pipeline stages metric-driven repairs use; unmapped issues are skipped rather
    than invented. Pure and deterministic (the VLM call already happened).
    """
    if items and design is not None:
        _attach_layer_ids([i for i in items if isinstance(i, dict)], design)
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        element = str(item.get("element") or "").strip()
        issue = str(item.get("issue") or "").strip()
        fix = str(item.get("suggested_fix") or "").strip()
        haystack = f"{issue} {fix}".lower()
        layer_ids = [str(i) for i in (item.get("layer_ids") or []) if i]
        # The sharpened critic reports its own severity; a VLM-flagged structural failure
        # can only ESCALATE the rule default, never downgrade it (so an erased-content "high"
        # is honored while a stray "medium" cannot soften a structural rule's high floor).
        item_severity = str(item.get("severity") or "").strip().lower()
        for needles, stage, action, rule_severity in _CRITIQUE_RULES:
            if not any(needle in haystack for needle in needles):
                continue
            severity = _strictest_severity(item_severity, rule_severity)
            params: dict[str, Any] = {"source": "vlm_critique", "layer_ids": layer_ids}
            if fix:
                params["suggested_fix"] = fix
            if action == "dedup":
                params.update({"raise_dedup_iou": True,
                               "duplicate_text": [element] if element else []})
            elif action == "refit-text-box":
                params.update({"widen": True, "shrink_to_fit": True,
                               "clipped_text": [element] if element else []})
            elif action == "resolve-fonts":
                params["wrong_glyphs"] = [element] if element else []
            elif action == "rebuild-clean-plate":
                params.update({"score_candidates": True, "color_match": True})
            elif action == "rerun-detection":
                params.update({"lower_confidence": True, "enable_element_propose": True})
            repair = {
                "stage": stage,
                "action": action,
                "reason": f"VLM critique: {issue}" + (f" — {element!r}" if element else ""),
                "params": params,
                "severity": severity,
            }
            if layer_ids:
                repair["target_id"] = layer_ids[0]
            out.append(repair)
            break
    # One defect can be reported twice with different wording; keep one repair per
    # (stage, action, target) so the no-repeat memory sees stable signatures.
    unique: list[dict] = []
    seen: set[tuple] = set()
    for repair in out:
        key = (repair.get("stage"), repair.get("action"), repair.get("target_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(repair)
    return unique
