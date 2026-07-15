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
        "text": {"editable_ui_copy": True, "emoji": "native_text_or_platform_raster",
                 "single_line_auto_resize": "WIDTH", "preserve_timestamp_group": True},
        "grouping": {"photo_frame": True, "header_cluster": True},
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
        "grouping": {"pair_text_with_backplate": True, "preserve_line_backplates": True},
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
                     "circular_insets_use_ellipse_mask": True},
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


def scene_facts(canvas: dict, ocr: dict | None = None, observations: dict | None = None) -> dict:
    """Normalize cheap scene observations into stable, testable classifier facts."""
    observations = copy.deepcopy(observations or {})
    lines = list((ocr or {}).get("lines") or [])
    texts = [str(x.get("text") or "").strip() for x in lines]
    joined = " ".join(texts).lower()
    w, h = max(1, int(canvas.get("w", 1))), max(1, int(canvas.get("h", 1)))
    has_before = bool(re.search(r"\bbefore\b", joined))
    has_after = bool(re.search(r"\bafter\b", joined))
    facts = {
        "aspect_ratio": w / h,
        "text_line_count": len([x for x in texts if x]),
        "before_after_labels": bool(re.search(r"\b(before|after|others|versus|vs)\b", joined)),
        "before_after_pair": has_before and has_after,
        "social_metadata": bool(re.search(r"\b(views?|reposts?|likes?|reply|am|pm)\b", joined)),
        "caption_language": bool(re.search(r"\b(my|i\s|finally|why|started|wish)\b", joined)),
        "emoji_present": any(ord(ch) > 0xFFFF for text in texts for ch in text),
    }
    facts.update(observations)
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
    # Dark UI chrome only counts alongside social evidence; a dark poster is not a tweet.
    if f.get("dark_background") and (f.get("social_metadata") or f.get("social_header")):
        add("social_screenshot", 1, "dark UI chrome")
    if photo >= .55 and backplates >= 2: add("caption_over_photo", 5, "photo with repeated text backplates")
    if f.get("caption_language") and photo >= .5: add("caption_over_photo", 5, "testimonial/caption language")
    if columns >= 2: add("comparison_grid", 4, "multiple columns")
    # Explicit comparison labels are a stronger structural signal than generic caption
    # words such as "why".  The latter often appears in the headline of a before/after
    # creative and previously won 5:4, which flattened one comparison column into the
    # photo plate and defeated the independently-swappable column contract.
    if f.get("before_after_labels"): add("comparison_grid", 7, "comparison labels")
    if f.get("center_divider"): add("comparison_grid", 2, "center divider")
    if photo >= .65: add("lifestyle_overlay", 3, "dominant lifestyle photo")
    if f.get("leader_lines") or f.get("circular_inset"): add("lifestyle_overlay", 3, "annotation overlay")
    if f.get("product_count", 0) and flat >= .45: add("product_on_flat", 5, "product on flat field")
    if flat >= .75: add("product_on_flat", 2, "dominant flat background")
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
    out = copy.deepcopy(cfg or {})
    preset = copy.deepcopy(result["preset"])
    facts = copy.deepcopy(result.get("facts") or {})
    # Only a literal BEFORE+AFTER pair authorizes rebuilding all contained column copy
    # and exposing two swappable photo bases. Generic "VS" comparison tables keep their
    # existing raster ownership policy.
    if result["archetype"] == "comparison_grid" and facts.get("before_after_pair"):
        preset["photo_regions"]["suppress_descendants"] = False
    out["scene"] = {"archetype": result["archetype"], "preset": preset, "facts": facts}
    routing = out.setdefault("routing", {})
    routing.setdefault("min_text_fidelity", preset["thresholds"]["min_text_fidelity"])
    routing["photo_regions"] = preset["photo_regions"]
    routing["text_policy"] = preset["text"]
    out.setdefault("layout", {})["scene_grouping"] = preset["grouping"]
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
    return out
