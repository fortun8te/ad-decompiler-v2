"""Format readiness: aspect classes + scene capabilities (not named presets).

Archetypes stay the coarse decomposition contracts (5 presets). New creative formats
(stories, feed squares, carousels, UGC, testimonials, UI screenshots, caption stacks,
diagrams-in-ads, …) should not explode ``ARCHETYPES``. Instead:

1. Classify the *frame* into a small aspect class (story / square / landscape / …).
2. Infer boolean *capabilities* from scene facts + the chosen archetype preset.
3. Let stages gate on capabilities (``has_capability(cfg, "text_plates")``) rather than
   hardcoding dozens of format names.

Capability overrides may be supplied in config (``format.capabilities``) or per-image
benchmark tags (``format_index.json``) without inventing a new preset.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

# Stable aspect buckets. Ratios are width/height. Bounds are inclusive on the lower edge
# and exclusive on the upper edge except ``wide`` which is open-ended.
ASPECT_CLASSES = (
    "story",       # ~9:16 stories / reels / TikTok
    "portrait",    # taller than square, shorter than story
    "square",      # feed 1:1 (±15%)
    "landscape",   # ~4:5 to ~16:9 display
    "wide",        # ultrawide / banner
)

# Boolean scene capabilities. Stages should prefer these over archetype-name checks
# when the behavior is about *what the scene needs*, not which preset won.
CAPABILITIES = (
    "text_plates",          # caption pills / solid text backplates
    "ui_chrome",            # social headers, dark UI bands, timestamp chrome
    "cutouts",              # product/people alpha cutouts over a plate
    "diagrams",             # arrows, callouts, leader lines, chart-like art
    "gradients",            # non-flat photographic / gradient fields
    "comparison_columns",   # before/after or multi-column comparison
    "caption_stack",        # stacked IG-style caption frames (story + plates)
    "flat_plate",           # solid-fill-safe plate (analytic chrome fill OK)
    "icons_as_chips",       # route small icons as exact image chips
)

# Construction hints stages may read; advisory, not a second preset system.
CONSTRUCTION_HINTS = {
    "text_plates": {"pair_text_with_backplate": True, "preserve_line_backplates": True},
    "ui_chrome": {"prefer_solid_flat_fill": True, "platform_ui_font_prior": True},
    "cutouts": {"retain_alpha_raster": True, "suppress_microtext_on_product": True},
    "diagrams": {"preserve_callout_leaders": True, "vector_gate_required": True},
    "gradients": {"avoid_solid_fill_on_photo": True},
    "comparison_columns": {"preserve_columns": True, "prevent_cross_column_blocks": True},
    "caption_stack": {"preserve_line_backplates": True, "stack_caption_frames": True},
    "flat_plate": {"prefer_solid_flat_fill": True, "solid_fill_first_residue": True},
    "icons_as_chips": {"icons_as_chips": True},
}


def aspect_ratio(width: float | int, height: float | int) -> float:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return w / h


def classify_aspect(width: float | int, height: float | int) -> dict[str, Any]:
    """Return a stable aspect-class decision from canvas geometry alone."""
    ratio = aspect_ratio(width, height)
    if ratio < 0.62:
        name = "story"
    elif ratio < 0.85:
        name = "portrait"
    elif ratio <= 1.15:
        name = "square"
    elif ratio <= 1.90:
        name = "landscape"
    else:
        name = "wide"
    return {
        "aspect_class": name,
        "aspect_ratio": round(ratio, 4),
        "width": int(width),
        "height": int(height),
    }


def peek_image_aspect(path: str | Path) -> dict[str, Any]:
    """Cheap PIL size probe for benchmark planning (no decode of pixels)."""
    try:
        from PIL import Image
    except ImportError:
        return {}
    with Image.open(path) as image:
        w, h = image.size
    return classify_aspect(w, h)


def infer_capabilities(
    facts: dict | None,
    *,
    archetype: str = "",
    preset: dict | None = None,
) -> dict[str, bool]:
    """Derive capabilities from scene facts + the winning archetype preset.

    This is deliberately rule-based and cheap — not a second ML classifier.
    """
    f = facts or {}
    preset = preset or {}
    grouping = preset.get("grouping") or {}
    photo = preset.get("photo_regions") or {}
    text = preset.get("text") or {}
    name = str(archetype or "").lower()

    caps = {c: False for c in CAPABILITIES}

    backplates = int(f.get("text_backplate_count", 0) or 0)
    caps["text_plates"] = bool(
        backplates >= 1
        or grouping.get("pair_text_with_backplate")
        or grouping.get("preserve_line_backplates")
    )
    caps["ui_chrome"] = bool(
        f.get("social_metadata")
        or f.get("social_header")
        or f.get("chat_ui")
        or name == "social_screenshot"
        or (f.get("dark_background") and (
            f.get("social_metadata") or f.get("social_header") or f.get("chat_ui")
        ))
    )
    caps["cutouts"] = bool(
        int(f.get("product_count", 0) or 0) >= 1
        or photo.get("retain_product_cluster")
        or float(f.get("photo_coverage", 0) or 0) >= 0.35
    )
    caps["diagrams"] = bool(
        f.get("leader_lines")
        or f.get("circular_inset")
        or grouping.get("preserve_callout_leaders")
        or name == "lifestyle_overlay"
    )
    # Gradients / photo fields: inverse of flat-plate dominance.
    flat = float(f.get("flat_background_fraction", 0) or 0)
    photo_cov = float(f.get("photo_coverage", 0) or 0)
    caps["gradients"] = bool(
        photo.get("full_bleed_is_background")
        or (photo_cov >= 0.55 and flat < 0.55)
        or name == "lifestyle_overlay"
    )
    caps["comparison_columns"] = bool(
        name == "comparison_grid"
        or int(f.get("column_count", 0) or 0) >= 2
        or f.get("before_after_labels")
        or f.get("before_after_pair")
        or f.get("stage_progression")
        or f.get("center_divider")
    )
    # Stacked caption pills (Simpletics IG caption reference: 1080x1920 + backplates).
    ratio = float(f.get("aspect_ratio", 0) or 0)
    caps["caption_stack"] = bool(
        caps["text_plates"]
        and ((0 < ratio < 0.70) or name == "caption_over_photo")
        and (backplates >= 2 or grouping.get("preserve_line_backplates")
             or f.get("caption_language"))
    )
    caps["flat_plate"] = bool(
        name in {"social_screenshot", "caption_over_photo", "comparison_grid", "product_on_flat"}
        or photo.get("background_is_flat_plate")
        or flat >= 0.70
        or caps["ui_chrome"]
    )
    # Lifestyle photo fields are not flat plates even if an archetype list would say so.
    if name == "lifestyle_overlay" and not caps["ui_chrome"]:
        caps["flat_plate"] = False
    caps["icons_as_chips"] = bool(
        name in {"social_screenshot", "product_on_flat", "comparison_grid"}
        or caps["comparison_columns"]
        or (caps["flat_plate"] and caps["ui_chrome"])
        or text.get("editable_ui_copy") and name == "product_on_flat"
    )
    return caps


def construction_hints(capabilities: dict[str, bool] | None) -> dict[str, Any]:
    """Flatten capability → construction hint contracts for advisory consumers."""
    hints: dict[str, Any] = {}
    for name, enabled in (capabilities or {}).items():
        if not enabled:
            continue
        for key, value in (CONSTRUCTION_HINTS.get(name) or {}).items():
            hints[key] = value
    return hints


def build_format_profile(
    canvas: dict | None = None,
    facts: dict | None = None,
    *,
    archetype: str = "",
    preset: dict | None = None,
    capability_overrides: dict | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Assemble the format block persisted on ``cfg['scene']['format']`` / artifacts."""
    canvas = canvas or {}
    facts = dict(facts or {})
    w = int(canvas.get("w") or facts.get("width") or 0)
    h = int(canvas.get("h") or facts.get("height") or 0)
    if (not w or not h) and facts.get("aspect_ratio"):
        # Aspect-only path (tests / planned.json without canvas): synthesize a unit canvas.
        ratio = float(facts["aspect_ratio"])
        h = 1000
        w = max(1, int(round(ratio * h)))
    aspect = classify_aspect(w or 1, h or 1)
    if "aspect_ratio" not in facts:
        facts["aspect_ratio"] = aspect["aspect_ratio"]

    caps = infer_capabilities(facts, archetype=archetype, preset=preset)
    # Only keys explicitly present in overrides win (False can disable an inference).
    if isinstance(capability_overrides, dict):
        for name in CAPABILITIES:
            if name in capability_overrides:
                caps[name] = bool(capability_overrides[name])

    tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
    # Soft tag → capability boosts (batch metadata, not new presets).
    tag_set = {t.lower() for t in tag_list}
    tag_boosts = {
        "ugc": ("cutouts", "gradients"),
        "testimonial": ("text_plates", "caption_stack"),
        "caption_pill": ("text_plates", "caption_stack"),
        "caption_stack": ("text_plates", "caption_stack"),
        "before_after": ("comparison_columns",),
        "problem_solution": ("comparison_columns", "diagrams"),
        "stage_progression": ("comparison_columns",),
        "carousel": ("comparison_columns",),
        "ui_screenshot": ("ui_chrome", "flat_plate", "icons_as_chips"),
        "diagram": ("diagrams",),
        "product_copy": ("cutouts", "text_plates", "flat_plate"),
        # IM8-style health: Day 1/10/30/90 chips + product cutout + sale seals.
        "timeline": ("icons_as_chips", "diagrams", "text_plates"),
        "health_product": ("cutouts", "text_plates", "flat_plate", "icons_as_chips"),
        "story": (),  # aspect-only hint; classify_aspect already handles geometry
    }
    for tag, boost in tag_boosts.items():
        if tag in tag_set:
            for cap in boost:
                caps[cap] = True

    enabled = sorted(name for name, on in caps.items() if on)
    return {
        "version": 1,
        "aspect_class": aspect["aspect_class"],
        "aspect_ratio": aspect["aspect_ratio"],
        "width": aspect["width"],
        "height": aspect["height"],
        "capabilities": caps,
        "enabled_capabilities": enabled,
        "construction_hints": construction_hints(caps),
        "tags": tag_list,
        "archetype": str(archetype or "") or None,
        "overrides_applied": sorted(
            name for name in CAPABILITIES
            if isinstance(capability_overrides, dict) and name in capability_overrides
        ),
        "capability_overrides": {
            k: bool(v) for k, v in (capability_overrides or {}).items()
            if k in CAPABILITIES
        } if capability_overrides else {},
    }


def apply_format(cfg: dict | None, profile: dict) -> dict:
    """Expose format profile under ``scene.format`` without clobbering archetype preset."""
    out = copy.deepcopy(cfg or {})
    scene = out.setdefault("scene", {})
    clean = copy.deepcopy(profile)
    scene["format"] = clean
    # Convenience mirrors for stages that already read flat routing/qa keys.
    routing = out.setdefault("routing", {})
    if clean.get("capabilities", {}).get("icons_as_chips") and routing.get("icons_as_chips") is None:
        routing["icons_as_chips_inferred"] = True
    return out


def attach_to_decision(
    decision: dict,
    canvas: dict | None = None,
    *,
    capability_overrides: dict | None = None,
    tags: Iterable[str] | None = None,
) -> dict:
    """Add a ``format`` block onto an archetype decision (returns a copy)."""
    out = copy.deepcopy(decision or {})
    facts = out.get("facts") or {}
    profile = build_format_profile(
        canvas,
        facts,
        archetype=str(out.get("archetype") or ""),
        preset=out.get("preset") or {},
        capability_overrides=capability_overrides,
        tags=tags,
    )
    out["format"] = profile
    return out


def format_from_cfg(cfg: dict | None) -> dict:
    scene = (cfg or {}).get("scene") or {}
    fmt = scene.get("format")
    return fmt if isinstance(fmt, dict) else {}


def has_capability(cfg: dict | None, name: str) -> bool:
    """True when the run's format profile enables ``name``."""
    caps = format_from_cfg(cfg).get("capabilities") or {}
    return bool(caps.get(name))


def prefers_solid_flat(cfg: dict | None) -> bool:
    """Solid analytic plate fill is preferred (UI chrome / flat fields)."""
    if has_capability(cfg, "flat_plate") or has_capability(cfg, "ui_chrome"):
        return True
    # Fallback when format profile was not attached (legacy runs / unit tests).
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()
    return archetype in {
        "social_screenshot", "caption_over_photo", "comparison_grid", "product_on_flat",
    }


def prefers_icon_chips(cfg: dict | None) -> bool:
    if has_capability(cfg, "icons_as_chips"):
        return True
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "").lower()
    # Comparison checklists (Wavy X/✓ pills) need exact icon chips like flat UI chrome.
    return archetype in {"social_screenshot", "product_on_flat", "comparison_grid"}


def load_format_index(path: str | Path | None) -> dict[str, dict]:
    """Load optional ``format_index.json`` mapping fixture_id → tags/capabilities.

    Schema::

        {
          "016": {"tags": ["product_copy", "story"], "capabilities": {"cutouts": true}},
          "201": {"tags": ["before_after"]}
        }
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        fid = str(key).strip()
        if fid.isdigit():
            fid = fid.zfill(3)
        out[fid.lower()] = {
            "tags": list(value.get("tags") or []),
            "capabilities": dict(value.get("capabilities") or {}),
            "notes": value.get("notes"),
        }
    return out


def resolve_format_index(input_dir: str | Path) -> dict[str, dict]:
    """Prefer ``<input_dir>/format_index.json``, else repo ``benchmark_set/format_index.json``."""
    root = Path(input_dir)
    for candidate in (root / "format_index.json", Path("benchmark_set") / "format_index.json"):
        loaded = load_format_index(candidate)
        if loaded:
            return loaded
    return {}


def planned_image_entry(
    path: Path,
    *,
    fixture_id: str,
    format_index: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Build one ``planned.json`` image row with aspect + optional format tags."""
    aspect = peek_image_aspect(path)
    meta = (format_index or {}).get(str(fixture_id).lower()) or {}
    entry = {
        "id": path.stem,
        "fixture_id": fixture_id,
        "filename": path.name,
        "path": str(path.resolve()),
    }
    if aspect:
        entry["aspect_class"] = aspect["aspect_class"]
        entry["aspect_ratio"] = aspect["aspect_ratio"]
        entry["width"] = aspect["width"]
        entry["height"] = aspect["height"]
    tags = list(meta.get("tags") or [])
    if tags:
        entry["format_tags"] = tags
    caps = meta.get("capabilities") or {}
    if caps:
        entry["format_capability_overrides"] = caps
    if meta.get("notes"):
        entry["format_notes"] = meta["notes"]
    return entry
