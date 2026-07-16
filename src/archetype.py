"""Deterministic ad archetype classification and decomposition presets.

The classifier intentionally consumes scene facts rather than filenames.  Facts may be
derived cheaply from OCR/image geometry or supplied by a VLM scene pass.  Presets are
advisory contracts persisted with every run and exposed to all later stages through cfg.
"""
from __future__ import annotations

import copy
import re
from typing import Any

ARCHETYPES = (
    "social_screenshot", "caption_over_photo", "comparison_grid",
    "lifestyle_overlay", "product_on_flat",
)

# Phase-2 reward contract (docs/HARNESS-PHASE2.md §1c): acceptance thresholds and reward
# weights are per-archetype. ``reward_weights`` steer src.qa_reward's metric ladder
# (local per-element SSIM / LPIPS perceptual / text recall); the two ``*_min`` keys in
# ``thresholds`` are the anti-degenerate acceptance-gate floors. Dark/text-heavy
# creatives (social screenshots) weight text correctness + local SSIM heavily and demote
# the global perceptual term — a tweet must not need 0.90 visual SSIM, it must need high
# text fidelity and correct structure.
#
# F12 calibration (measured on runs/integration-full-16 + runs/integration-smoke-6):
# the known-BAD 002 (product cluster erased by inpaint) scores LPIPS-similarity 0.732
# and reward local-SSIM 0.465, while every known-OK run clusters at LPIPS 0.976-0.995 /
# local-SSIM 0.60-0.74. The old floors (lpips 0.20-0.35, local 0.35-0.45) let 002 clear
# both gates by 2-3x, so the gate rejected nothing. Floors are lifted to sit in the gap:
# photo archetypes use lpips>=0.80 / local>=0.55 (rejects 002 on BOTH gates, accepts the
# good runs by >=0.13 margin); social_screenshot keeps the most lenient floors
# (lpips>=0.60 / local>=0.50) because global perceptual scores are unreliable on dark UI
# chrome — there text recall (weight 0.45) is the honest signal.
PRESETS: dict[str, dict[str, Any]] = {
    "social_screenshot": {
        "photo_regions": {"retain_as_single_image": True, "suppress_descendants": True,
                          "default_mask": "rrect", "flatten_scene_artwork": False},
        "social_header": {"unreadable_identity": "masked_raster_cluster",
                          "avatar_mask": "ellipse"},
        # CODIA-PARITY: platform UI copy is Inter at letterSpacing 0; never rasterize
        # correct-class fits. Chat/header grouping stays conservative (strong evidence only).
        "text": {"editable_ui_copy": True, "emoji": "native_text_or_platform_raster",
                 "single_line_auto_resize": "WIDTH", "preserve_timestamp_group": True,
                 "default_family": "Inter", "platform_ui_prior": True},
        "grouping": {"photo_frame": True, "header_cluster": True,
                     "message_bubbles": True, "engagement_row": True,
                     "ama_sticker": True, "quote_frame": True,
                     "circular_insets_use_ellipse_mask": True},
        "thresholds": {"text_recall_min": 0.90, "editable_text_recall_min": 0.86,
                       "min_text_fidelity": 0.40, "visual_pass_ssim_min": 0.55,
                       "edge_f1_min": 0.35, "native_text_ratio_min": 0.90,
                       "lpips_similarity_min": 0.60, "reward_local_ssim_min": 0.50},
        "reward_weights": {"local_ssim": 0.40, "lpips": 0.15, "text": 0.45},
    },
    "caption_over_photo": {
        "photo_regions": {"retain_as_single_image": True, "suppress_descendants": True,
                          "default_mask": "rrect", "flatten_scene_artwork": False},
        "text": {"editable_ui_copy": True, "emoji": "native_text_or_platform_raster",
                 "single_line_auto_resize": "WIDTH"},
        "grouping": {"pair_text_with_backplate": True, "preserve_line_backplates": True,
                     "quote_frame": True, "circular_insets_use_ellipse_mask": True},
        "thresholds": {"text_recall_min": 0.92, "editable_text_recall_min": 0.88,
                       "min_text_fidelity": 0.40, "native_text_ratio_min": 0.90,
                       # SSIM is a FLOOR gate, not the objective (docs/CODIA-PARITY-SPEC.md):
                       # a photo-caption ad whose text is all native + plate is clean must
                       # pass at a modest global SSIM, so this floor is deliberately low.
                       "visual_pass_ssim_min": 0.60,
                       "lpips_similarity_min": 0.80, "reward_local_ssim_min": 0.50},
        "reward_weights": {"local_ssim": 0.35, "lpips": 0.25, "text": 0.40},
    },
    "comparison_grid": {
        "photo_regions": {"retain_as_single_image": True, "suppress_descendants": True,
                          "default_mask": "rect", "flatten_scene_artwork": False},
        "text": {"editable_ui_copy": True, "preserve_inline_styles": True},
        "grouping": {"preserve_columns": True, "prevent_cross_column_blocks": True,
                     "preserve_aligned_rows": True, "pair_text_with_backplate": True},
        "thresholds": {"text_recall_min": 0.93, "editable_text_recall_min": 0.88,
                       "min_text_fidelity": 0.40, "visual_pass_ssim_min": 0.65,
                       "edge_f1_min": 0.45, "native_text_ratio_min": 0.90,
                       "lpips_similarity_min": 0.80, "reward_local_ssim_min": 0.55},
        "reward_weights": {"local_ssim": 0.40, "lpips": 0.20, "text": 0.40},
    },
    "lifestyle_overlay": {
        "photo_regions": {"retain_as_single_image": True, "suppress_descendants": True,
                          "full_bleed_is_background": True, "flatten_scene_artwork": False},
        "text": {"editable_ui_copy": True, "single_line_auto_resize": "WIDTH"},
        "grouping": {"preserve_callout_leaders": True, "pair_text_with_backplate": True,
                     "circular_insets_use_ellipse_mask": True, "quote_frame": True},
        "thresholds": {"text_recall_min": 0.90, "editable_text_recall_min": 0.85,
                       "min_text_fidelity": 0.40, "native_text_ratio_min": 0.90,
                       "visual_pass_ssim_min": 0.60,
                       "lpips_similarity_min": 0.80, "reward_local_ssim_min": 0.55},
        "reward_weights": {"local_ssim": 0.35, "lpips": 0.35, "text": 0.30},
    },
    "product_on_flat": {
        "photo_regions": {"retain_product_cluster": True, "suppress_product_microtext": True,
                          "background_is_flat_plate": True, "flatten_scene_artwork": False},
        "text": {"editable_ui_copy": True, "wordmarks": "artwork"},
        "grouping": {"pair_text_with_backplate": True, "rating_strip_atomic_fallback": True},
        "thresholds": {"text_recall_min": 0.90, "editable_text_recall_min": 0.84,
                       "min_text_fidelity": 0.40, "native_text_ratio_min": 0.90,
                       "visual_pass_ssim_min": 0.60,
                       "lpips_similarity_min": 0.80, "reward_local_ssim_min": 0.55},
        "reward_weights": {"local_ssim": 0.40, "lpips": 0.35, "text": 0.25},
    },
}


def image_facts(image_path: str) -> dict:
    """Return cheap filename-independent plate/photo evidence from the normalized image."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {}
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.uint8)
    # Top quantized colours approximate how much of the canvas is a rebuildable flat
    # field. Using three bins handles near-white compression noise and split black/white
    # layouts without mistaking a diverse photograph for a flat plate.
    quantized = (rgb // 24).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    counts = np.sort(counts)[::-1]
    flat = float(counts[:3].sum()) / max(1, rgb.shape[0] * rgb.shape[1])
    # Mean luminance flags dark UI chrome (tweets, app screenshots) — the class where
    # global SSIM/LPIPS mislead and the Phase-2 reward shifts weight onto text + local
    # per-element scores.
    luma = float((rgb.astype(np.float64) @ (0.299, 0.587, 0.114)).mean())
    return {
        "flat_background_fraction": round(flat, 4),
        "photo_coverage": round(max(0.0, min(1.0, 1.0 - flat)), 4),
        "mean_luma": round(luma, 2),
        "dark_background": bool(luma < 68.0),
    }


def _line_box(line: dict) -> tuple[float, float, float, float] | None:
    box = line.get("bbox") or line.get("box") or {}
    try:
        return float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
    except (KeyError, TypeError, ValueError):
        return None


def _largest_xstart_cluster(members: list, tol: float) -> list:
    """Chain-cluster column candidates by x-start; return the largest cluster."""
    if not members:
        return []
    rows = sorted(members, key=lambda r: r[0])
    clusters, current = [], [rows[0]]
    for row in rows[1:]:
        if row[0] - current[-1][0] <= tol:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    clusters.append(current)
    return max(clusters, key=len)


def _mirrored_column_facts(canvas: dict, lines: list) -> dict:
    """Detect two mirrored checklist columns (rows of icon+text) from OCR geometry.

    Comparison creatives (BEFORE/AFTER cards, us-vs-competitor checklists, split-panel
    product tables) share a structure that survives OCR even when no BEFORE/AFTER token
    is printed: a left-aligned column of row texts in each canvas half, the two column
    x-starts separated by roughly half the canvas, with rows y-aligned pairwise. The
    detector is deliberately strict — packshot label microtext (nutrition tables,
    repeated can wordmarks) also produces y-coincident text, so membership requires
    OCR confidence >= 0.55, per-column x-start coherence, real text (median length >= 4,
    widest member >= 15% canvas width), wide column separation (>= 35% canvas width),
    and most rows of the smaller column mirrored (pair ratio >= 0.6). Calibrated on
    runs/postfix-benchmark-4: fires on 025/066/101 only; 002/091/135 packaging tables
    and 094 chart-axis labels all fail at least two of the quality gates.
    """
    w = max(1, int(canvas.get("w", 1)))
    boxed = []
    for line in lines:
        box = _line_box(line)
        text = str(line.get("text") or "").strip()
        conf = line.get("conf", line.get("confidence"))
        if conf is not None:
            try:
                if float(conf) < 0.55:
                    continue
            except (TypeError, ValueError):
                pass
        if box and text:
            boxed.append((*box, text))
    mid = w / 2.0
    # A column member sits inside one canvas half; canvas-spanning headlines excluded.
    left = [r for r in boxed if r[2] <= 0.55 * w and r[0] + r[2] <= 1.05 * mid and r[0] < mid]
    right = [r for r in boxed if r[2] <= 0.55 * w and r[0] >= 0.95 * mid]
    tol = 0.025 * w
    lcol = _largest_xstart_cluster(left, tol)
    rcol = _largest_xstart_cluster(right, tol)

    def column_ok(col: list) -> bool:
        if len(col) < 3:
            return False
        lens = sorted(len(r[4]) for r in col)
        return lens[len(lens) // 2] >= 4 and max(r[2] for r in col) >= 0.15 * w

    pairs = 0
    separation = (rcol[0][0] - lcol[0][0]) / w if (lcol and rcol) else 0.0
    if column_ok(lcol) and column_ok(rcol) and separation >= 0.35:
        used: set[int] = set()
        for lx, ly, lw, lh, _lt in sorted(lcol, key=lambda r: r[1]):
            lyc = ly + lh / 2
            best = None
            for j, (rx, ry, rw, rh, _rt) in enumerate(rcol):
                if j in used:
                    continue
                dist = abs(lyc - (ry + rh / 2))
                if dist <= 0.75 * max(lh, rh) and (best is None or dist < best[0]):
                    best = (dist, j)
            if best:
                used.add(best[1])
                pairs += 1
    ratio = pairs / max(1, min(len(lcol), len(rcol)))
    return {
        "mirrored_column_rows": pairs,
        "mirrored_columns": bool(pairs >= 4 and ratio >= 0.6),
    }


def scene_facts(canvas: dict, ocr: dict | None = None, observations: dict | None = None) -> dict:
    """Normalize cheap scene observations into stable, testable classifier facts."""
    observations = copy.deepcopy(observations or {})
    lines = list((ocr or {}).get("lines") or [])
    texts = [str(x.get("text") or "").strip() for x in lines]
    joined = " ".join(texts).lower()
    w, h = max(1, int(canvas.get("w", 1))), max(1, int(canvas.get("h", 1)))
    has_before = bool(re.search(r"\bbefore\b", joined))
    has_after = bool(re.search(r"\bafter\b", joined))
    # MONTE-style WITHOUT / WITH column tags. Bare "with" is too common in body copy,
    # so only count a WITH label when the OCR line is essentially that single token.
    has_without = bool(re.search(r"\bwithout\b", joined))
    has_with_label = any(bool(re.fullmatch(r"with\.?", t, re.I)) for t in texts if t)
    # IM8 problem-solution / VS column tags (STRUGGLE↔ANSWER, PATCHED↔DAILY).
    has_struggle = any(bool(re.fullmatch(r"struggle\.?", t, re.I)) for t in texts if t)
    has_answer = any(bool(re.fullmatch(r"answer\.?", t, re.I)) for t in texts if t)
    has_problem = any(bool(re.fullmatch(r"problem\.?", t, re.I)) for t in texts if t)
    has_solution = any(bool(re.fullmatch(r"solution\.?", t, re.I)) for t in texts if t)
    has_patched = any(
        bool(re.fullmatch(r"patched(?:\s+together)?\.?", t, re.I)) for t in texts if t
    )
    has_daily = any(
        bool(re.fullmatch(r"daily(?:\s+im8)?\.?", t, re.I)) for t in texts if t
    )
    # IM8 three-stage body strip: BEFORE / RITUAL / RESET (not a classic 2-side pair alone).
    has_ritual = any(bool(re.fullmatch(r"ritual\.?", t, re.I)) for t in texts if t)
    has_reset = any(bool(re.fullmatch(r"reset\.?", t, re.I)) for t in texts if t)
    has_before = (
        has_before or has_without or has_struggle or has_problem or has_patched
    )
    has_after = (
        has_after or has_with_label or has_answer or has_solution or has_daily or has_reset
    )
    stage_progression = bool(
        (has_before and has_ritual and has_reset)
        or (has_ritual and has_reset and (has_before or has_after))
    )
    # Photo-of-handwriting / in-image-only text: honor explicit observations. Geometric
    # inference lives in merge_layers (no VLM); do not invent the flag from a lone
    # script wordmark on an otherwise overlay-heavy ad.
    photo_of_handwriting = bool(
        observations.get("photo_of_handwriting")
        or observations.get("handwriting_photo")
    )
    text_on_photo_only = bool(
        observations.get("text_on_photographic_surfaces_only")
        or observations.get("scene_text_only")
        or photo_of_handwriting
    )
    facts = {
        "aspect_ratio": w / h,
        "text_line_count": len([x for x in texts if x]),
        "before_after_labels": bool(
            re.search(
                r"\b(before|after|without|others|versus|vs|struggle|answer|"
                r"problem|solution|patched|daily|ritual|reset)\b",
                joined,
            )
            or has_with_label
            or has_patched
            or has_daily
        ),
        "before_after_pair": has_before and has_after,
        "stage_progression": stage_progression,
        "social_metadata": bool(re.search(
            r"\b(views?|reposts?|likes?|reply|am|pm|weergaven)\b", joined
        )),
        # IG/DM chrome without tweet-style view counts.
        "chat_ui": bool(re.search(
            r"\b(new messages?|active now|delivered|seen\b|typing|direct message|"
            r"message requests?|online|yesterday|just now)\b",
            joined,
        )),
        # IG Ask-me-anything / question sticker chrome (035-style UGC).
        "ama_sticker": bool(re.search(
            r"\b(ask\s*me\s*anything|ask\s*me\b|\bama\b|questions?\s*for\s*me|"
            r"ask\s*me\s*anything\??)\b",
            joined,
        )),
        "caption_language": bool(re.search(r"\b(my|i\s|finally|why|started|wish)\b", joined)),
        "emoji_present": any(ord(ch) > 0xFFFF for text in texts for ch in text),
        "photo_of_handwriting": photo_of_handwriting,
        "text_on_photographic_surfaces_only": text_on_photo_only,
    }
    facts.update(_mirrored_column_facts(canvas, lines))
    # Column-header comparison cues (OUR COMPETITOR / TYPICAL X / VS) only count when a
    # genuinely mirrored row structure backs them — "typical" in body copy of a
    # lifestyle ad must not manufacture comparison evidence on its own.
    facts["column_vs_cues"] = bool(
        facts["mirrored_column_rows"] >= 3
        and re.search(r"\b(vs\.?|versus|competitors?|typical)\b", joined)
    )
    facts.update(observations)
    if photo_of_handwriting:
        facts["photo_of_handwriting"] = True
        facts["text_on_photographic_surfaces_only"] = True
    elif text_on_photo_only:
        facts["text_on_photographic_surfaces_only"] = True
    return facts


def classify(facts: dict, configured: str = "auto") -> dict:
    """Return a scored archetype decision. Ties use stable, explicit precedence."""
    if configured and configured != "auto":
        if configured not in ARCHETYPES:
            raise ValueError(f"unknown archetype: {configured}")
        return decision(configured, {configured: 1.0}, ["configured override"])

    f = facts or {}
    score = {name: 0.0 for name in ARCHETYPES}
    why: dict[str, list[str]] = {name: [] for name in ARCHETYPES}
    def add(name: str, amount: float, reason: str):
        score[name] += amount; why[name].append(reason)

    photo = float(f.get("photo_coverage", 0) or 0)
    flat = float(f.get("flat_background_fraction", 0) or 0)
    backplates = int(f.get("text_backplate_count", 0) or 0)
    columns = int(f.get("column_count", 0) or 0)
    if f.get("social_metadata"): add("social_screenshot", 8, "social metadata")
    if f.get("social_header"): add("social_screenshot", 4, "social header")
    if f.get("avatar_present"): add("social_screenshot", 1, "avatar")
    if f.get("chat_ui"): add("social_screenshot", 7, "chat/DM chrome")
    if f.get("ama_sticker"): add("social_screenshot", 7, "AMA/question sticker")
    # Dark UI chrome only counts alongside social evidence; a dark poster is not a tweet.
    if f.get("dark_background") and (
        f.get("social_metadata") or f.get("social_header") or f.get("chat_ui")
        or f.get("ama_sticker")
    ):
        add("social_screenshot", 1, "dark UI chrome")
    if photo >= .55 and backplates >= 2: add("caption_over_photo", 5, "photo with repeated text backplates")
    if f.get("caption_language") and photo >= .5: add("caption_over_photo", 5, "testimonial/caption language")
    if columns >= 2: add("comparison_grid", 4, "multiple columns")
    # Explicit comparison labels are a stronger structural signal than generic caption
    # words such as "why".  The latter often appears in the headline of a before/after
    # creative and previously won 5:4, which flattened one comparison column into the
    # photo plate and defeated the independently-swappable column contract.
    if f.get("before_after_labels"): add("comparison_grid", 7, "comparison labels")
    # Mirrored checklist columns fire on label-free comparisons (066 two-card mascara
    # checklists, 101 split-panel TPU table) that carry no BEFORE/AFTER token at all.
    # The detector is geometry-based and strict (see _mirrored_column_facts), so it is
    # worth almost as much as an explicit label pair.
    if f.get("mirrored_columns"): add("comparison_grid", 6, "mirrored column checklist rows")
    if f.get("column_vs_cues"): add("comparison_grid", 3, "VS/competitor column headers")
    if f.get("stage_progression"): add("comparison_grid", 5, "stage progression strip")
    if f.get("center_divider"): add("comparison_grid", 2, "center divider")
    if photo >= .65: add("lifestyle_overlay", 3, "dominant lifestyle photo")
    if f.get("leader_lines") or f.get("circular_inset"): add("lifestyle_overlay", 3, "annotation overlay")
    products = int(f.get("product_count", 0) or 0)
    if products and flat >= .45: add("product_on_flat", 5, "product on flat field")
    if flat >= .75: add("product_on_flat", 2, "dominant flat background")
    # Dark gradient packshots (Hears-style): multi-product cutouts on a dark plate even
    # when quantization under-counts flatness on a smooth gradient.
    mean_luma = float(f.get("mean_luma", 255) or 255)
    if products >= 2 and (f.get("dark_background") or mean_luma < 90):
        add("product_on_flat", 4, "multi-product on dark field")
    elif products >= 1 and f.get("dark_background") and flat >= .30:
        add("product_on_flat", 3, "product on dark plate")
    # Social chrome is more specific than generic caption-over-photo.
    precedence = {name: i for i, name in enumerate(ARCHETYPES[::-1])}
    chosen = max(ARCHETYPES, key=lambda n: (score[n], precedence[n]))
    if not score[chosen]:
        chosen = "product_on_flat" if flat >= .5 else "lifestyle_overlay"
        why[chosen].append("conservative visual default")
    result = decision(chosen, score, why[chosen])
    result["facts"] = copy.deepcopy(f)
    return result


def decision(name: str, scores: dict, reasons: list[str]) -> dict:
    return {"version": 1, "archetype": name, "scores": scores,
            "reasons": reasons, "preset": copy.deepcopy(PRESETS[name])}


def apply_preset(cfg: dict, result: dict) -> dict:
    """Expose the contract through real downstream config namespaces."""
    from src import format_readiness

    out = copy.deepcopy(cfg or {})
    preset = copy.deepcopy(result["preset"])
    facts = copy.deepcopy(result.get("facts") or {})
    # Only a literal BEFORE+AFTER pair authorizes rebuilding all contained column copy
    # and exposing two swappable photo bases. Generic "VS" comparison tables keep their
    # existing raster ownership policy.
    if result["archetype"] == "comparison_grid" and facts.get("before_after_pair"):
        preset["photo_regions"]["suppress_descendants"] = False
    # Huel-style social chrome on a comparison: reuse social header clustering without
    # inventing a new preset. Circular product insets on Wavy stories reuse lifestyle flag.
    if result["archetype"] == "comparison_grid":
        grouping = dict(preset.get("grouping") or {})
        if facts.get("social_metadata") or facts.get("social_header") or facts.get("avatar_present"):
            grouping["header_cluster"] = True
        if facts.get("circular_inset"):
            grouping["circular_insets_use_ellipse_mask"] = True
        # IM8 STRUGGLE→product string leaders: reuse lifestyle callout preserve flag.
        if facts.get("leader_lines") or facts.get("stage_progression"):
            grouping["preserve_callout_leaders"] = True
        preset["grouping"] = grouping
    out["scene"] = {"archetype": result["archetype"], "preset": preset, "facts": facts}
    routing = out.setdefault("routing", {})
    routing.setdefault("min_text_fidelity", preset["thresholds"]["min_text_fidelity"])
    routing["photo_regions"] = preset["photo_regions"]
    routing["text_policy"] = preset["text"]
    # Photo-of-handwriting / in-image-only OCR: suppress editable text emission downstream.
    if facts.get("photo_of_handwriting") or facts.get("text_on_photographic_surfaces_only"):
        text_policy = dict(routing.get("text_policy") or {})
        text_policy["scene_text_only"] = True
        text_policy["suppress_editable_ocr"] = True
        routing["text_policy"] = text_policy
    out.setdefault("layout", {})["scene_grouping"] = preset["grouping"]
    # Platform-UI font prior (Inter on social screenshots) is consumed by text_analysis.
    text_policy = preset.get("text") or {}
    if text_policy.get("platform_ui_prior") or text_policy.get("default_family"):
        ta = out.setdefault("text_analysis", {})
        if text_policy.get("platform_ui_prior") is not None:
            ta.setdefault("platform_ui_prior", bool(text_policy.get("platform_ui_prior")))
        if text_policy.get("default_family"):
            ta.setdefault("platform_ui_family", text_policy["default_family"])
    # Avatar ellipse mask hint for routing/reconstruct (009 circular profile crop).
    social_header = preset.get("social_header") or {}
    if social_header.get("avatar_mask"):
        routing.setdefault("avatar_mask", social_header["avatar_mask"])
    grouping = preset.get("grouping") or {}
    # Circular product insets (white ring over photo) → ellipse clip, not alpha matte.
    if grouping.get("circular_insets_use_ellipse_mask") or facts.get("circular_inset"):
        routing.setdefault("circular_inset_ellipse", True)
    out.setdefault("qa", {})["archetype_thresholds"] = preset["thresholds"]
    # F8: the preset's text-recall contract (0.90-0.93) was carried only inside
    # archetype_thresholds and enforced nowhere. Expose it at the flat ``qa.text_recall_min``
    # key so the metrics layer (pixel_diff, owned by the metrics agent) can thread it into
    # QA the same way editable_text_recall_min is threaded — otherwise the strict
    # per-archetype text bar stays decorative while only the lenient visual bar is wired.
    text_recall_min = preset["thresholds"].get("text_recall_min")
    if text_recall_min is not None:
        out["qa"]["text_recall_min"] = text_recall_min
    # Phase-2 reward weights (consumed by src.qa_reward); explicit qa.reward.weights
    # overrides still win inside qa_reward.reward_weights.
    if preset.get("reward_weights"):
        out["qa"]["reward_weights"] = copy.deepcopy(preset["reward_weights"])
    # Format readiness: aspect class + capabilities (not a new named preset). Prefer an
    # already-attached decision.format (pipeline may pass tags/overrides); else build one.
    format_cfg = (cfg or {}).get("format") or {}
    profile = result.get("format")
    if not isinstance(profile, dict):
        canvas = {
            "w": int(facts.get("width") or 0),
            "h": int(facts.get("height") or 0),
        }
        if not canvas["w"] or not canvas["h"]:
            # scene_facts stores aspect_ratio; synthesize when canvas dims absent.
            canvas = None
        profile = format_readiness.build_format_profile(
            canvas,
            facts,
            archetype=result["archetype"],
            preset=preset,
            capability_overrides=format_cfg.get("capabilities"),
            tags=format_cfg.get("tags") or result.get("format_tags"),
        )
    out = format_readiness.apply_format(out, profile)
    out["scene"]["archetype"] = result["archetype"]
    out["scene"]["preset"] = preset
    out["scene"]["facts"] = facts
    return out
