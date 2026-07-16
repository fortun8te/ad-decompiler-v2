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
_DEFAULT_LOCAL_SSIM_MIN = 0.55
# F6 / workstream E: worst-local floor. Aggregate local SSIM 0.6064 passed while the worst
# 64px window sat at 0.0091 (002). Raised from 0.10→0.15 so erased/seamed windows fail the
# gate without always-failing good runs (009/013/052 locals ≥ 0.60, worst windows ≫ 0.15).
# Local aggregate floor raised 0.50→0.55 into the gap below good runs (0.60–0.74) and above
# known-bad 002 (0.465).
_DEFAULT_WORST_LOCAL_SSIM_MIN = 0.15
# One layer may contribute at most this share of the local-component weight, so a single
# huge raster cannot dominate the weighted mean (the reward-buying vector on 002).
_LOCAL_WEIGHT_SHARE_CAP = 0.25

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
# GA3: ceiling on the construction component when the contract EXPLICITLY failed and the
# score came from a fallback (native_text_ratio / bare construction) rather than the real
# contract_score. Keeps a failed-contract round from out-scoring a passing one.
_CONTRACT_FAIL_CONSTRUCTION_CEIL = 0.35

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
        "worst_local_ssim_min": _floor(
            "reward_worst_local_ssim_min", "worst_local_ssim_min",
            _DEFAULT_WORST_LOCAL_SSIM_MIN),
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
        # F6: cap any single layer's weight share. Pixel weighting let one giant host
        # raster (~70% of the 002 canvas, region_ssim 0.91) buy the aggregate while
        # every text/price layer under it had collapsed.
        raw_total = sum(weight for _, _, weight in scored)
        weight_cap = _LOCAL_WEIGHT_SHARE_CAP * raw_total
        scored = [(lid, value, min(weight, weight_cap)) for lid, value, weight in scored]
        total = sum(weight for _, _, weight in scored)
        mean = sum(value * weight for _, value, weight in scored) / max(total, 1e-9)
        ordered = sorted(value for _, value, _ in scored)
        p10 = ordered[min(len(ordered) - 1, int(0.1 * (len(ordered) - 1)))]
        worst_layer = ordered[0]
        # Robust blend with the lower tail AND the floor: the worst layer stays
        # diagnostic instead of vanishing into the weighted mean.
        score = 0.6 * mean + 0.25 * p10 + 0.15 * worst_layer
        worst_local = worst_layer
        worst_window = _worst_window_ssim(qa)
        if worst_window is not None:
            # The worst measured 64px window (pixel_diff) is the sharpest degenerate
            # signal available — fold it in so an erased/seamed region depresses the
            # component even when every per-layer crop still averages out.
            score = 0.85 * score + 0.15 * worst_window
            worst_local = min(worst_local, worst_window)
        worst = sorted(scored, key=lambda item: item[1])[:4]
        result = {
            "score": round(max(0.0, min(1.0, score)), 4),
            "mean": round(mean, 4),
            "p10": round(p10, 4),
            "min": round(worst_layer, 4),
            "worst_local": round(worst_local, 4),
            "count": len(scored),
            "worst": [{"id": lid, "score": value} for lid, value, _ in worst],
            "source": "per_layer",
        }
        if worst_window is not None:
            result["worst_window"] = round(worst_window, 4)
        return result

    ssim = qa.get("ssim")
    if isinstance(ssim, (int, float)):
        return {"score": round(max(0.0, min(1.0, float(ssim))), 4),
                "count": 0, "source": "multiscale_ssim"}
    return None


def _worst_window_ssim(qa: Optional[dict]) -> Optional[float]:
    """Worst measured local window SSIM from pixel_diff evidence, when present."""
    qa = qa or {}
    window = qa.get("local_ssim_worst_window")
    if isinstance(window, dict) and isinstance(window.get("ssim"), (int, float)):
        return max(0.0, min(1.0, float(window["ssim"])))
    local = qa.get("local_ssim")
    if isinstance(local, dict) and isinstance(local.get("min"), (int, float)):
        return max(0.0, min(1.0, float(local["min"])))
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
    score = max(0.0, min(1.0, float(score)))
    # GA3: when the construction contract EXPLICITLY failed (editability/placement/erasure),
    # the construction component must not be allowed to dominate the reward off a high raw
    # native_text_ratio (or bare construction) fallback — an erased-imagery round scores a
    # great "native text ratio" and would be picked as best. Clamp it below the fail ceiling
    # so a failed contract can never buy a top-of-ladder construction score.
    contract_failed = contract.get("pass") is False
    if contract_failed and source != "contract_score":
        score = min(score, _CONTRACT_FAIL_CONSTRUCTION_CEIL)
    detail = {"score": round(score, 4), "source": source}
    ntr = contract.get("native_text_ratio")
    if not isinstance(ntr, (int, float)):
        ntr = qa.get("native_text_ratio")
    if isinstance(ntr, (int, float)):
        detail["native_text_ratio"] = round(float(ntr), 4)
    if "pass" in contract:
        detail["contract_pass"] = bool(contract.get("pass"))
        if contract_failed and source != "contract_score":
            detail["clamped_contract_fail"] = True
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
        # GA2: LPIPS was attempted this round (paths available). If it is enabled but came
        # back None (torch/lpips import failed, model download failed, CPU OOM), mark it
        # ``unavailable`` so the acceptance gate can flag the missing perceptual floor RED
        # instead of silently skipping it. Absent-because-disabled leaves lpips None.
        if lpips is None and lpips_enabled(cfg) and (source and render):
            lpips = {"similarity": None, "unavailable": True}
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
        if qa is None and run_dir:
            qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        qa = qa if isinstance(qa, dict) else {}
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

        # F6: worst-local floor. An aggregate that passes while the worst window sits at
        # ~0 is a bought score (002: local 0.6064 vs worst window 0.0091) — the gate must
        # go RED on the worst measured local evidence, not just the mean.
        worst_local = None
        if isinstance(local, dict):
            for key in ("worst_local", "min"):
                value = local.get(key)
                if isinstance(value, (int, float)):
                    worst_local = float(value) if worst_local is None else min(
                        worst_local, float(value))
        window = _worst_window_ssim(qa)
        if window is not None:
            worst_local = window if worst_local is None else min(worst_local, window)
        if worst_local is not None:
            passed = worst_local >= floors["worst_local_ssim_min"]
            checks["worst_local_ssim"] = {"value": round(worst_local, 4),
                                          "min": floors["worst_local_ssim_min"],
                                          "ok": passed}
            ok = ok and passed

        lpips = components.get("lpips") or {}
        if isinstance(lpips, dict) and isinstance(lpips.get("similarity"), (int, float)):
            passed = float(lpips["similarity"]) >= floors["lpips_similarity_min"]
            checks["lpips_similarity"] = {"value": lpips["similarity"],
                                          "min": floors["lpips_similarity_min"], "ok": passed}
            ok = ok and passed
        elif isinstance(lpips, dict) and lpips.get("unavailable") and lpips_enabled(cfg):
            # GA2: LPIPS is one of the three advertised anti-degenerate floors. When it was
            # ATTEMPTED but failed to load (torch/lpips import failed, model download failed,
            # CPU OOM) compute_reward marks the component ``unavailable`` — the check used to
            # silently vanish, letting a perceptually catastrophic render clear the gate.
            # Record it RED so the missing floor is visible and non-passing. A reward that
            # simply never attempted LPIPS (no run_dir/paths, or explicitly disabled) carries
            # no sentinel and is still legitimately skipped.
            checks["lpips_similarity"] = {"value": None, "ok": False,
                                          "min": floors["lpips_similarity_min"],
                                          "reason": "unavailable"}
            ok = False

        # Standing hard failures and a failed construction contract can never be green:
        # the gate record itself must read RED, not just the separate qa.ok flag (002's
        # gate said ok:true beside 4 hard fails and contract_pass=false).
        hard_fails = reward.get("hard_fails")
        if not isinstance(hard_fails, int):
            hard_fails = _hard_fail_count(qa)
        if hard_fails:
            checks["hard_fails"] = {"value": hard_fails, "max": 0, "ok": False}
            ok = False
        # Agent B: unresolved glyph residue is a hard structural fail — the harness must
        # never declare success over it even if a summary omitted the hard_fails list.
        residue = False
        for fail in list(qa.get("hard_fails") or []):
            if isinstance(fail, dict) and fail.get("rule") == "glyph-residue":
                residue = True
                break
        structural = qa.get("structural") if isinstance(qa.get("structural"), dict) else {}
        try:
            if int(structural.get("glyph_residue_unresolved") or 0) > 0:
                residue = True
        except (TypeError, ValueError):
            pass
        contract = qa.get("contract") if isinstance(qa.get("contract"), dict) else {}
        if contract.get("glyph_residue_clean") is False:
            residue = True
        if residue:
            checks["glyph_residue"] = {"value": True, "ok": False,
                                       "detail": "unresolved glyph residue under removed text"}
            ok = False
        if contract.get("pass") is False:
            checks["contract"] = {
                "value": False, "ok": False,
                "detail": "construction contract failed (editability/placement)",
            }
            ok = False

        if not checks:
            return {"ok": True, "skipped": "no_metrics"}
        return {"ok": ok, "checks": checks}
    except Exception as exc:
        # Fail CLOSED. The gate is the anti-degenerate safety net; a gate that cannot
        # evaluate must never convert a would-be RED into GREEN (critic A GA1). A
        # malformed qa row / non-dict contract / NaN throwing inside the gate path used
        # to return ok:True, silently scoring a degenerate render as acceptable.
        return {"ok": False, "skipped": f"gate_error:{type(exc).__name__}", "error": True}


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


_CROP_PAD_PX = 64
_CROP_NOTE = (
    "\n\nNOTE: both images are the SAME cropped region of a larger advertisement — the "
    "region QA measured as the worst match. Report only defects visible inside the crop."
)


def _valid_box(box: Any) -> bool:
    return isinstance(box, dict) and all(
        isinstance(box.get(k), (int, float)) for k in ("x", "y", "w", "h"))


def _worst_crop_box(qa: Optional[dict]) -> Optional[dict]:
    """Box of the worst measured evidence across the worst local window AND per-layer rows.

    F9: the whole-image critique missed local defects (seam, broken plate, wrong fonts)
    and produced vague whole-image opinions. Feeding the VLM a crop of the worst measured
    region makes its report local and checkable — and cheap (fewer pixels).

    P1 gap 2: the crop now targets the genuinely-worst evidence. The single worst window
    (``local_ssim_worst_window``, produced alongside qa_worst_window.png) and every scored
    per-layer region are ranked together by measured score; the lowest-scoring region wins,
    so a per-layer region that is worse than the window drives the crop instead of being
    ignored. The window wins ties (unchanged behaviour when it is the sole/worst evidence)."""
    qa = qa or {}
    candidates: list[tuple[float, int, dict]] = []
    window = qa.get("local_ssim_worst_window")
    if isinstance(window, dict):
        box = window.get("bbox") or window.get("box")
        if _valid_box(box):
            score = window.get("ssim")
            candidates.append(
                (float(score) if isinstance(score, (int, float)) else 0.0, 0, dict(box)))
    for row in qa.get("per_layer") or []:
        if not isinstance(row, dict):
            continue
        value = row.get("region_ssim")
        if not isinstance(value, (int, float)):
            continue
        box = row.get("abs_box") or row.get("box")
        if not _valid_box(box):
            continue
        candidates.append((float(value), 1, dict(box)))
    if not candidates:
        return None
    # Worst (lowest) score first; the window (rank 0) wins ties for back-compat.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _crop_image_bytes(image_bytes: bytes, box: dict, pad: int = _CROP_PAD_PX) -> bytes:
    """Crop PNG bytes to *box* (padded, clamped). Raises on any failure — callers catch."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        left = max(0, int(box["x"]) - pad)
        top = max(0, int(box["y"]) - pad)
        right = min(image.width, int(box["x"] + box["w"]) + pad)
        bottom = min(image.height, int(box["y"] + box["h"]) + pad)
        if right - left < 8 or bottom - top < 8:
            raise ValueError("crop box degenerate")
        cropped = image.crop((left, top, right, bottom))
        out = io.BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()


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

    # F9: where per-layer/window QA evidence exists, critique CROPS of the worst
    # measured region instead of the whole image (opt-out via critique.crop_worst).
    crop_box = None
    prompt = _CRITIQUE_PROMPT
    if _critique_cfg(cfg).get("crop_worst", True) and run_dir:
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        box = _worst_crop_box(qa if isinstance(qa, dict) else {})
        if box:
            try:
                source_bytes = _crop_image_bytes(source_bytes, box)
                preview_bytes = _crop_image_bytes(preview_bytes, box)
                crop_box = box
                prompt = _CRITIQUE_PROMPT + _CROP_NOTE
            except Exception:
                crop_box = None  # fall back to the whole image, failure-proof

    settings = _critique_vlm_settings(cfg)
    try:
        answer = _ask_vlm_pair(
            source_bytes, preview_bytes, prompt,
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
    if crop_box:
        result["crop"] = crop_box
    # Evidence fingerprint lets the harness skip a repeat VLM call when the crop critic's
    # inputs (source/preview/qa worst-window) are unchanged across rounds.
    if run_dir:
        try:
            import hashlib as _hashlib
            parts = []
            for name in ("normalized.png", "preview.png", "qa.json", "design.json"):
                path = os.path.join(run_dir, name)
                try:
                    with open(path, "rb") as handle:
                        parts.append(_hashlib.sha256(handle.read()).hexdigest())
                except OSError:
                    parts.append("")
            qa_for_fp = _load_json(os.path.join(run_dir, "qa.json"), {})
            window = (qa_for_fp.get("local_ssim_worst_window")
                      if isinstance(qa_for_fp, dict) else None)
            parts.append(json.dumps(window, sort_keys=True, default=str) if window else "")
            result["evidence_fingerprint"] = _hashlib.sha256(
                "|".join(parts).encode("utf-8")).hexdigest()
        except Exception:
            pass
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

    Workstream E: never re-promote baked chrome / kept_in_photo OCR to editable TEXT —
    those cutouts are the intended representation (Agent A).
    """
    if items and design is not None:
        _attach_layer_ids([i for i in items if isinstance(i, dict)], design)
    baked_ids: set[str] = set()
    if isinstance(design, dict):
        try:
            from src.harness import is_baked_chrome_layer, is_already_sliced_layer
            for layer in _flatten_layers(design.get("layers") or []):
                if not isinstance(layer, dict) or not layer.get("id"):
                    continue
                if is_baked_chrome_layer(layer) or is_already_sliced_layer(layer):
                    baked_ids.add(str(layer["id"]))
        except Exception:
            baked_ids = set()
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
            # Baked chrome / already-sliced: skip TEXT promote and re-slice thrash.
            if layer_ids and baked_ids.intersection(layer_ids):
                if stage == "text-analysis" or action in {
                    "restore-editable-text", "refit-text-box", "resolve-fonts",
                }:
                    break
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
