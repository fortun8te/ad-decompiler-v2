"""Compile the canonical scene graph into the Figma-facing design schema v2."""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional

from .schema import (
    DesignDoc, Layer, SCHEMA_VERSION, dump, validate_design, fallback_kind,
)
from .text_analysis import fit_text_box, _fit_font, _line_advance, _style_name, _weight_candidates
from .raster_clusters import is_intentional_raster_cluster

# Candidate keys that ``_compile`` already routes to a concrete Layer field.  Anything
# else on a reconstruct entity (e.g. an image ``ref`` or a future mask spec another
# stage attaches) is unknown to the dataclass, so it is preserved under
# ``meta['passthrough']`` rather than silently dropped on the way to design.json.
_CONSUMED_CANDIDATE_KEYS = frozenset({
    "id", "target", "box", "meta", "z_index", "z", "visible_box", "ink_box",
    "rotation", "opacity", "blend_mode", "effects", "constraints", "component",
    "layout", "children", "text", "style", "text_runs", "fill", "stroke",
    "shape_kind", "path", "svg", "src", "radius", "paths", "mask", "name", "role",
    "sizing",
})

# Designer-facing role → layer label. Prefer meta.semantic_role / meta.role / style.role.
_ROLE_LABELS = {
    "headline": "Headline", "title": "Headline",
    "subheadline": "Subheadline", "subtitle": "Subheadline",
    "body": "Body", "body-copy": "Body", "body_copy": "Body", "copy": "Body",
    "callout": "Callout", "callout-text": "Callout", "benefit": "Callout",
    "caption": "Caption", "eyebrow": "Eyebrow", "offer": "Offer",
    "cta": "CTA", "button": "Button",
    "avatar": "Avatar", "profile": "Avatar", "profile_picture": "Avatar",
    "profile_photo": "Avatar", "pfp": "Avatar",
    "product": "Product", "photo": "Photo", "person": "Person", "people": "Person",
    "icon": "Icon", "arrow": "Arrow",
    "callout-leader": "Arrow", "callout_leader": "Arrow",
    "leader": "Arrow", "leader-line": "Arrow", "leader_line": "Arrow",
    "connector": "Arrow",
    "logo": "Logo", "wordmark": "Logo",
    "background": "Background", "plate": "Background", "clean-plate": "Background",
    "badge": "Badge", "chip": "Chip", "emoji": "Emoji", "decoration": "Decoration",
    "banner": "Banner", "ribbon": "Banner", "brushstroke": "Banner",
    "stroke-banner": "Banner", "seal": "Badge",
    "starburst": "Badge", "price-burst": "Badge", "sale-burst": "Badge", "burst": "Badge",
    "shape": "Shape", "illustration": "Illustration", "image": "Photo",
    "screenshot": "Screenshot", "ui-panel": "UI panel", "receipt": "Receipt",
    "chart": "Chart", "graph": "Graph", "table": "Table",
    "nutrition-panel": "Nutrition panel", "diagram": "Diagram",
    "infographic": "Infographic", "product-cluster": "Product",
    "text-stack": "Text Stack", "caption-stack": "Caption",
    "caption-plate": "Caption", "card-grid": "Card Grid", "panel-set": "Panel Set",
    "structural-grid": "Grid", "native-chart": "Chart", "card": "Card",
    "header": "Header", "footer": "Footer",
    "disclaimer": "Disclaimer", "legal": "Disclaimer", "fine-print": "Disclaimer",
    "hero": "Hero", "band": "Group",
    "asset-group": "Group", "text": "Text",
    "message-bubble": "Message", "message": "Message", "bubble": "Message",
    "message-row": "Message row", "reply-quote": "Reply", "quote": "Reply",
    "header-cluster": "Header", "stat-pill": "Stat", "stat-stack": "Stats",
    "stat-row": "Stats", "benefit-stack": "Benefits", "pill": "Stat",
    "rating-strip": "Rating", "rating": "Rating", "logo-strip": "Logo strip",
    "as-seen-in": "As seen in", "leader-dot": "Dot", "leader_dot": "Dot",
    "story-cta": "CTA", "sale-burst": "Badge", "sale_burst": "Badge",
    "comparison-set": "Comparison", "comparison-column": "Photo",
    "comparison-panel": "Photo", "photo-panel": "Photo",
    "vs": "VS", "versus": "VS", "vs-chip": "VS", "vs-badge": "VS",
    "checklist": "Checklist", "text-row": "Row", "label": "Label",
    "ama-sticker": "AMA sticker", "quote-frame": "Quote",
    "circular-inset": "Circular inset", "engagement-row": "Engagement",
    "timeline": "Timeline", "timeline-step": "Step", "review-bar": "Reviews",
}

_TARGET_FALLBACK = {
    "text": "Text", "image": "Photo", "icon": "Icon", "group": "Group", "shape": "Shape",
}

# Machine / pipeline names that must never ship to Figma as layer labels.
_MACHINE_NAME_RE = re.compile(
    r"(?i)"
    r"(?:^c_[a-z]?\d+\b)"
    r"|(?:\braster\s*slice\b)"
    r"|(?:\bswappable\b)"
    r"|(?:[—\-]+\s*vector\b)"
    r"|(?:\bclean\s*plate\b)"
    r"|(?:\blow\s*confidence\b)"
    r"|(?:\bpanel\s*raster\b)"
    r"|(?:\basset\s*group\b)"
    r"|(?:\braster\s*crop\b)"
    r"|(?:^band-[0-9a-f]+$)"
    r"|(?:^text-stack-)"
)
_QUOTE_STYLE_RE = re.compile(r'[—\-]\s*"')


def _strip_edge_emoji(text: str) -> str:
    """Strip leading/trailing emoji/pictographs (plus adjoining spaces) from a line.

    Codia removes emoji from TEXT characters and ships them as exact pixel cutouts
    (spec §2b); rendering them as glyphs produces platform-font tofu that never
    matches the painted pixels. Only the edges are stripped so text_runs offsets
    for the surviving characters stay valid (both 009 emoji are line-final).
    """
    import unicodedata

    def _is_emoji(ch: str) -> bool:
        code = ord(ch)
        if code in (0x200D, 0xFE0E, 0xFE0F, 0x20E3):
            return True
        if code >= 0x1F000:
            return True
        return unicodedata.category(ch) == "So"

    current = text
    while True:
        trimmed = current.strip()
        while trimmed and _is_emoji(trimmed[0]):
            trimmed = trimmed[1:].lstrip()
        while trimmed and _is_emoji(trimmed[-1]):
            trimmed = trimmed[:-1].rstrip()
        if trimmed == current:
            break
        current = trimmed
    # Never blank a line that was pure emoji; interior spacing is untouched, and a
    # fully-unchanged line returns the original string (runs offsets stay exact).
    return current if current else text


def _truncate(value, length=28):
    value = " ".join(str(value or "").split())
    return value if len(value) <= length else value[: length - 1] + "…"


def _clean_snippet(value, length=28) -> str:
    """Collapse whitespace/newlines and truncate for designer-facing layer suffixes."""
    return _truncate(value, length)


def _normalize_role_token(value) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    return "-".join(token.split())


def _role_token(candidate) -> str:
    meta = candidate.get("meta") or {}
    style = candidate.get("style") or {}
    for raw in (
        meta.get("semantic_role"), meta.get("role"), candidate.get("role"),
        style.get("role"),
    ):
        token = _normalize_role_token(raw)
        if token:
            return token
    return ""


def _role_label(role: str, target: Optional[str] = None) -> str:
    if role in _ROLE_LABELS:
        return _ROLE_LABELS[role]
    if role:
        return role.replace("-", " ").strip().title() or _TARGET_FALLBACK.get(target or "", "Layer")
    return _TARGET_FALLBACK.get(target or "", "Layer")


def _is_machine_name(value, candidate_id: Optional[str] = None) -> bool:
    name = " ".join(str(value or "").split()).strip()
    if not name:
        return True
    if candidate_id and name == str(candidate_id).strip():
        return True
    if candidate_id and name.lower().startswith(str(candidate_id).strip().lower() + " "):
        return True
    if _MACHINE_NAME_RE.search(name):
        return True
    if _QUOTE_STYLE_RE.search(name):
        return True
    return False


def _explicit_designer_name(candidate) -> Optional[str]:
    """Return a pre-existing designer name, ignoring ids / VLM / pipeline leftovers.

    ``vlm_name`` is intentionally omitted: naming is local and sync; VLM labels are
    not consulted on the design-stage hot path.
    """
    meta = candidate.get("meta") or {}
    cid = candidate.get("id")
    for raw in (
        candidate.get("name"), meta.get("semantic_name"),
        meta.get("layer_name"), meta.get("label"),
    ):
        if raw is None:
            continue
        text = " ".join(str(raw).split()).strip()
        if text and not _is_machine_name(text, cid):
            return _truncate(text, 56)
    return None


def _with_snippet(label: str, text) -> str:
    snippet = _clean_snippet(text, 28)
    if not snippet:
        return label
    # Avoid "Headline / Headline" when copy equals the role word.
    if snippet.casefold() == label.casefold():
        return label
    return f"{label} / {snippet}"


def _name(candidate):
    """Fast deterministic Figma layer name (no VLM / network)."""
    explicit = _explicit_designer_name(candidate)
    if explicit:
        return explicit

    meta = candidate.get("meta") or {}
    target = candidate.get("target")
    role = _role_token(candidate)
    text = candidate.get("text") or meta.get("source_text")

    # Comparison ads: Before / After / VS / Photo / Before — prefer short local names.
    side = str(meta.get("comparison_side") or meta.get("before_after_side") or "").lower()
    if side in {"before", "after", "without", "with", "mid"}:
        side_label = {
            "before": "Before", "without": "Before",
            "after": "After", "with": "After",
            "mid": "Ritual",
        }[side]
        # Prefer merge-assigned semantic names for IM8 Struggle/Answer/Daily/Reset.
        if meta.get("semantic_name") and target == "text":
            return str(meta["semantic_name"])
        if target == "image":
            return f"Photo / {side_label}"
        if target == "text" or role in {"label", "eyebrow", "caption", "tag"}:
            return side_label
    vs_blob = str(text or meta.get("shell_text_snippet") or "").strip()
    if role in {"vs", "versus", "vs-chip", "vs-badge"} or _normalize_role_token(role) in {
        "vs", "versus", "vs-chip", "vs-badge",
    }:
        return "VS"
    if re.fullmatch(r"vs\.?|versus", vs_blob, re.I):
        return "VS"
    if role == "stage-progression" or meta.get("stage_count"):
        return "Progression"
    if role == "checklist" or meta.get("checklist") or (
        role == "text-row" and any(
            _normalize_role_token((c.get("meta") or {}).get("role")) in {
                "verified", "checkmark", "check", "check-mark", "tick",
                "x", "close", "cross", "cancel",
            }
            for c in (candidate.get("children") or [])
        )
    ):
        return "Checklist"

    label = _role_label(role, target)

    if target == "text" or (text and role in {
        "headline", "title", "subheadline", "subtitle", "body", "body-copy",
        "body_copy", "copy", "caption", "eyebrow", "offer", "cta", "button", "text",
        "callout", "callout-text", "benefit", "disclaimer", "legal", "fine-print",
        "footer", "label",
    }):
        # Bare Before/After/VS / IM8 stage labels: no "Label / Before" padding.
        if re.fullmatch(
            r"before|after|without|with|vs\.?|versus|struggle|answer|problem|solution|"
            r"ritual|reset|patched(?:\s+together)?|daily(?:\s+im8)?",
            str(text or "").strip(),
            re.I,
        ):
            token = str(text).strip()
            if re.fullmatch(r"vs\.?|versus", token, re.I):
                return "VS"
            if re.fullmatch(r"before|without", token, re.I):
                return "Before"
            if re.fullmatch(r"after|with", token, re.I):
                return "After"
            return token.title()
        return _with_snippet(label if label != "Layer" else "Text", text)

    if target == "group":
        if role in {"button", "cta"}:
            return _with_snippet(label, text)
        return label if label != "Layer" else "Group"

    if meta.get("wordmark"):
        return _with_snippet("Logo", text or candidate.get("text"))

    if meta.get("substitution") or meta.get("low_fidelity"):
        return _with_snippet("Text", text)

    # Text-bearing brushstroke / seal / outline-pill plates.
    if meta.get("text_bearing_shell") or meta.get("plate_shell"):
        if role in {"banner", "ribbon", "brushstroke", "stroke-banner", "stroke_banner"}:
            shell_label = "Banner"
        elif role in {"button"}:
            shell_label = "Button"
        elif role in {"cta"}:
            shell_label = "CTA"
        elif (
            role in {"callout", "pill", "benefit"}
            or meta.get("stroke_outline_shell")
        ):
            shell_label = "Callout"
        else:
            shell_label = "Badge"
        snippet = meta.get("shell_text_snippet") or text
        return _with_snippet(shell_label, snippet)

    if target == "image":
        if meta.get("shell_raster_chip"):
            if meta.get("stroke_outline_shell") or role in {"callout", "pill"}:
                chip_label = "Callout"
            elif role in {"banner", "ribbon", "brushstroke"}:
                chip_label = "Banner"
            elif role:
                chip_label = "Badge"
            else:
                chip_label = "Shape"
            return _with_snippet(chip_label, meta.get("shell_text_snippet") or text)
        return label if label not in {"Layer", "Shape"} else "Photo"

    if target == "icon":
        return label if label != "Layer" else "Icon"

    return label if label != "Layer" else "Shape"


def _dedupe_sibling_names(layers: list) -> None:
    """Append ' / 2', ' / 3', … only when siblings share an identical name."""
    seen: dict[str, int] = {}
    for layer in layers or []:
        children = getattr(layer, "children", None) or []
        if children:
            _dedupe_sibling_names(children)
        base = str(getattr(layer, "name", "") or "Layer")
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count > 1:
            layer.name = f"{base} / {count}"


def _resolve(path: Optional[str], run_dir: str) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.normpath(os.path.join(run_dir, path))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return None


def _stage_asset(src: Optional[str], layer_id: str, run_dir: str, warnings: list) -> Optional[str]:
    resolved = _resolve(src, run_dir)
    if not resolved:
        warnings.append({"code": "missing-asset", "layer_id": layer_id, "path": src})
        return None
    # Existing-but-truncated assets are just as unusable as missing files.  Detect them
    # before copying so preview/Figma never receive a poisoned checkpoint.
    try:
        if os.path.getsize(resolved) <= 0:
            raise ValueError("empty file")
        if os.path.splitext(resolved)[1].lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"
        }:
            from PIL import Image
            with Image.open(resolved) as image:
                image.verify()
    except (OSError, ValueError, SyntaxError) as exc:
        warnings.append({
            "code": "corrupt-asset", "layer_id": layer_id, "path": src,
            "detail": str(exc),
        })
        return None
    assets = os.path.join(run_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    # Assets already in the run are canonical; do not duplicate them on every rebuild.
    if os.path.commonpath([os.path.abspath(resolved), os.path.abspath(assets)]) == os.path.abspath(assets):
        return os.path.relpath(resolved, run_dir)
    base = os.path.basename(resolved)
    destination = os.path.join(assets, f"{layer_id}_{base}")
    if os.path.abspath(resolved) != os.path.abspath(destination):
        shutil.copyfile(resolved, destination)
    return os.path.relpath(destination, run_dir)


def _surface_fill(candidate):
    fill = candidate.get("fill")
    if fill is not None:
        return fill
    style = candidate.get("style") or {}
    fills = style.get("fills")
    if isinstance(fills, list) and fills:
        return fills[0]
    if style.get("fill") is not None:
        return style.get("fill")
    if style.get("color"):
        return {"kind": "flat", "color": style["color"]}
    return None


# ── Mixed-weight lines → sibling TEXT nodes (Codia §2a) ───────────────────────────
# Codia never uses styled ranges: a weight change always produces a NEW sibling node
# ("05:00 PM · 12-05-2026 ·"(300) / "121K"(700) / "weergaven"(300)). Splitting is
# trivially safe in every Figma plugin and lets each run carry its own sampled color.
# Only CONTRAST-VERIFIED weight runs split (upstream _enrich_word_styles gates on a
# >=180 weight delta measured from word pixels); everything else stays one node.
_WEIGHT_SPLIT_MIN_DELTA = 250
# Candidates whose file weight is farther than this from the declared node weight are
# not trusted as the renderable face (Regular.ttf must not stand in for Bold).
_WEIGHT_CANDIDATE_MATCH_TOL = 150


def _promote_weight_candidate(style: dict) -> None:
    """Keep ``fontCandidates[0]`` consistent with the declared ``fontWeight``.

    Weight-split siblings update ``fontWeight`` (e.g. 700 for "121K") while inheriting
    the parent line's Regular candidates. Preview/`fit_text_box` then measure/draw the
    wrong face, and Figma's candidate-path retries can pick Regular over the primary
    Bold request. Promote the closest matching-weight candidate, or rewrite the top
    entry's weight/style and drop a mismatched file path so resolvers fall through to
    family+weight (Figma) / system bold (preview).
    """
    if not isinstance(style, dict):
        return
    try:
        weight = int(round(float(style.get("fontWeight") or 400)))
    except (TypeError, ValueError):
        return
    italic = "italic" in str(style.get("fontStyle") or "").lower()
    style_label = _style_name(weight, italic=italic)
    style["fontWeight"] = weight
    style["fontStyle"] = style_label
    style["fontWeightCandidates"] = _weight_candidates(weight)
    cands = [dict(c) for c in (style.get("fontCandidates") or []) if isinstance(c, dict)]
    if not cands:
        return

    def _cand_weight(candidate: dict) -> int:
        try:
            return int(round(float(candidate.get("weight") or 400)))
        except (TypeError, ValueError):
            return 400

    ranked = sorted(
        cands,
        key=lambda c: (abs(_cand_weight(c) - weight), -float(c.get("score") or 0.0)),
    )
    top = dict(ranked[0])
    if abs(_cand_weight(top) - weight) > _WEIGHT_CANDIDATE_MATCH_TOL:
        top["weight"] = weight
        top["style"] = style_label
        top.pop("path", None)
    else:
        top["weight"] = _cand_weight(top)
        top_style = str(top.get("style") or "")
        if top_style and (("italic" in top_style.lower()) == italic):
            style["fontStyle"] = top_style
        else:
            top["style"] = style_label
    rest = [c for c in ranked[1:]]
    style["fontCandidates"] = [top] + rest
    if top.get("family"):
        style["fontFamily"] = top["family"]


def _normalize_text_stroke(stroke, style: dict, effects: list) -> tuple:
    """Keep glyph fills readable: OUTSIDE strokes, capped width, fat outlines as effects.

    Figma's default text stroke is CENTER/INSIDE, which paints opaque outline ink over
    the fill and covers the letters. Authored marketing outlines sit outside the glyph.
    Very thick detected bands are converted to a soft drop-shadow ring instead.
    """
    if not stroke:
        return None, list(effects or [])
    out = dict(stroke) if isinstance(stroke, dict) else {
        "kind": "flat", "color": str(stroke), "width": 1.0,
    }
    try:
        font_size = float((style or {}).get("fontSize") or 0) or 16.0
    except (TypeError, ValueError):
        font_size = 16.0
    try:
        width = float(out.get("width", out.get("weight", 1.0)) or 1.0)
    except (TypeError, ValueError):
        width = 1.0
    width = max(0.5, width)
    if width > max(3.0, font_size * 0.14):
        color = out.get("color") or out.get("paint") or "#000000"
        if isinstance(color, dict):
            color = color.get("color") or "#000000"
        ring = {
            "type": "DROP_SHADOW",
            "color": color,
            "offset": {"x": 0, "y": 0},
            "radius": round(min(width, font_size * 0.2), 2),
            "spread": round(max(0.0, min(width * 0.35, font_size * 0.08)), 2),
            "visible": True,
        }
        merged = list(effects or [])
        merged.append(ring)
        return None, merged
    out["width"] = round(min(width, max(1.0, font_size * 0.08)), 2)
    align = str(out.get("align") or out.get("alignment") or out.get("strokeAlign") or "").upper()
    if align in ("", "CENTER", "INSIDE", "CENTRE"):
        out["align"] = "OUTSIDE"
    else:
        out["align"] = align
    out["strokeAlign"] = out["align"]
    return out, list(effects or [])


def _split_weight_run_siblings(candidate: dict) -> list[dict]:
    """Split a single-line text candidate at weight-run boundaries into siblings.

    Fail-closed: any measurement/shape problem returns the original candidate.
    """
    try:
        return _split_weight_run_siblings_unsafe(candidate)
    except Exception:
        return [candidate]


def _split_weight_run_siblings_unsafe(candidate: dict) -> list[dict]:
    if candidate.get("target") != "text":
        return [candidate]
    text = str(candidate.get("text") or "")
    if not text.strip() or "\n" in text:
        return [candidate]
    base_style = dict(candidate.get("style") or {})
    try:
        base_weight = int(round(float(base_style.get("fontWeight") or 400)))
    except (TypeError, ValueError):
        base_weight = 400
    runs = []
    for run in candidate.get("text_runs") or []:
        if not isinstance(run, dict):
            return [candidate]
        style = run.get("style") or {}
        try:
            start, end = int(run.get("start")), int(run.get("end"))
            weight = int(round(float(style.get("fontWeight") or base_weight)))
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text) and abs(weight - base_weight) >= _WEIGHT_SPLIT_MIN_DELTA:
            runs.append((start, end, style))
    if not runs:
        return [candidate]
    runs.sort(key=lambda item: item[0])
    # Overlapping runs: fail closed, keep the single node with runs.
    for (s0, e0, _), (s1, _e1, _s) in zip(runs, runs[1:]):
        if s1 < e0:
            return [candidate]
    font_size = float(base_style.get("fontSize") or 16.0)
    font = _fit_font(base_style, font_size)
    if font is None:
        return [candidate]
    total_adv = _line_advance(font, text, 0.0)
    if total_adv <= 0:
        return [candidate]
    box = dict(candidate.get("box") or {})
    vis = candidate.get("visible_box") or candidate.get("ink_box") or box

    segments = []
    cursor = 0
    for start, end, style in runs:
        if start > cursor:
            segments.append((cursor, start, None))
        segments.append((start, end, style))
        cursor = end
    if cursor < len(text):
        segments.append((cursor, len(text), None))

    out = []
    for index, (start, end, run_style) in enumerate(segments):
        segment_text = text[start:end].strip()
        if not segment_text:
            continue
        # Proportional advance mapping absorbs the proxy font's width error.
        lead = text[start:end].index(segment_text[0])
        frac_x = _line_advance(font, text[:start + lead], 0.0) / total_adv
        frac_w = _line_advance(font, text[start:end].strip(), 0.0) / total_adv
        piece = dict(candidate)
        piece.pop("text_runs", None)
        piece["id"] = f"{candidate.get('id') or 'text'}__w{index}"
        piece["text"] = segment_text
        style = dict(base_style)
        if run_style:
            style.update({k: v for k, v in run_style.items() if v is not None})
        _promote_weight_candidate(style)
        piece["style"] = style
        piece["box"] = {"x": float(box.get("x", 0) or 0) + float(box.get("w", 0) or 0) * frac_x,
                        "y": float(box.get("y", 0) or 0),
                        "w": max(1.0, float(box.get("w", 0) or 0) * frac_w),
                        "h": float(box.get("h", 0) or 0)}
        piece["visible_box"] = {
            "x": float(vis.get("x", 0) or 0) + float(vis.get("w", 0) or 0) * frac_x,
            "y": float(vis.get("y", 0) or 0),
            "w": max(1.0, float(vis.get("w", 0) or 0) * frac_w),
            "h": float(vis.get("h", 0) or 0)}
        meta = dict(candidate.get("meta") or {})
        meta["weight_split"] = {"of": str(candidate.get("id") or ""), "segment": index,
                                "segments": len(segments)}
        piece["meta"] = meta
        out.append(piece)
    return out if len(out) > 1 else [candidate]


# Codia's text boxes are LOOSE: a 56px font sits in a ~129px box, 72px in ~165px
# (~2x lineHeight), vertically centered. Tight OCR ink boxes are why our renders
# clipped at box edges. 1.6x lineHeight per line is the floor the parity spec sets.
_TEXT_BOX_HEIGHT_FACTOR = 1.6
_TEXT_BOX_WIDTH_SLACK = 0.06


def _generous_text_box(box: dict, style: dict, text: str, stroke=None) -> dict:
    """Grow a fitted text box Codia-style without moving the visible text.

    Height grows symmetrically around the ink center to >= 1.6x lineHeight per line
    (vertical CENTER alignment absorbs the slop, so the baseline never moves).
    Width gains ~6% slack away from the horizontal anchor (LEFT keeps the left edge,
    RIGHT the right edge, CENTER splits), so anchor-based placement is unchanged.
    Outside strokes get extra padding so outline ink is not clipped by the text frame.
    """
    out = dict(box or {})
    try:
        w = float(out.get("w", 0) or 0)
        h = float(out.get("h", 0) or 0)
        lines = max(1, str(text or "").count("\n") + 1)
        font_size = float(style.get("fontSize") or 0) or max(1.0, h / lines)
        line_height = float(style.get("lineHeight") or 0) or font_size * 1.2
        # Cap-height-derived fontSize exceeds painted ink height; keep >=1.25x fontSize
        # of vertical room so CAP_HEIGHT trim / descenders never clip.
        min_h = max(
            _TEXT_BOX_HEIGHT_FACTOR * line_height * lines,
            font_size * 1.25 * lines,
        )
        stroke_pad = 0.0
        if isinstance(stroke, dict):
            try:
                stroke_pad = max(0.0, float(stroke.get("width", stroke.get("weight", 0)) or 0))
            except (TypeError, ValueError):
                stroke_pad = 0.0
        min_h += 2.0 * stroke_pad
        if h < min_h:
            out["y"] = float(out.get("y", 0) or 0) - (min_h - h) / 2.0
            out["h"] = min_h
        if w > 0:
            extra = w * _TEXT_BOX_WIDTH_SLACK + 2.0 * stroke_pad
            align = str(style.get("align", "LEFT")).upper()
            if align == "RIGHT":
                out["x"] = float(out.get("x", 0) or 0) - extra
            elif align in ("CENTER", "JUSTIFIED"):
                out["x"] = float(out.get("x", 0) or 0) - extra / 2.0
            elif stroke_pad:
                out["x"] = float(out.get("x", 0) or 0) - stroke_pad
                extra = w * _TEXT_BOX_WIDTH_SLACK + stroke_pad
            out["w"] = w + extra
        out["x"] = round(float(out.get("x", 0) or 0), 2)
        out["y"] = round(float(out.get("y", 0) or 0), 2)
        out["w"] = round(float(out.get("w", 0) or 0), 2)
        out["h"] = round(float(out.get("h", 0) or 0), 2)
    except (TypeError, ValueError):
        return dict(box or {})
    return out


def _semantic_z(candidate, target):
    """Return a stable fallback when upstream stages emit the placeholder z=0."""
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or candidate.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"} or meta.get("source") == "inpaint":
        return -1_000_000
    band = str(meta.get("z_band") or "").lower()
    band_z = {
        "background": -1_000_000, "plate": -1_000_000,
        "content": 20, "scene": 20, "foreground": 30,
        "overlay": 40, "chrome": 50, "ui": 50,
    }.get(band)
    if band_z is not None:
        return band_z
    if target == "text":
        return 40
    if target == "icon":
        return 35
    if target == "image":
        return 30 if role not in {"background", "photo-fragment"} else 25
    if target in {"shape", "group"}:
        return 20
    return 10


def _compile(candidate: dict, run_dir: str, warnings: list) -> Layer:
    target = candidate.get("target")
    layer_id = str(candidate.get("id") or "layer")
    box = dict(candidate.get("box") or {"x": 0, "y": 0, "w": 1, "h": 1})
    meta = dict(candidate.get("meta") or {})
    # The Figma importer understands the complete style object (multiple paints,
    # strokes and effects).  Keep it intact for every editable layer instead of
    # reducing non-text layers to their first fill before export.
    source_style = dict(candidate.get("style") or {})
    source_effects = candidate.get("effects")
    if source_effects is None:
        source_effects = source_style.get("effects")
    if candidate.get("z_index") is not None:
        z_raw = candidate.get("z_index")
    elif candidate.get("z") is not None:
        z_raw = candidate.get("z")
    elif meta.get("z") is not None:
        z_raw = meta.get("z")
    else:
        # Missing z is not an explicit paint-order instruction.  Route it through the
        # semantic stack below so a gradient/background shape stays behind its image,
        # and icons/text stay above that image.  The old image default (10) silently
        # put unannotated photos behind native gradient surfaces.
        z_raw = None
    # Fusion assigns OCR a small ``z=1`` merely to distinguish it from its
    # detected shell.  It is not a final paint order: native button/card shapes
    # receive semantic z=20 and would otherwise cover their own CTA. Preserve
    # genuinely explicit text z-orders (>1), but promote the fusion placeholder
    # to the normal front text band.
    text_placeholder_z = target == "text" and z_raw in (None, 0, 1, "0", "0.0", "1", "1.0")
    z_index = float(_semantic_z(candidate, target) if text_placeholder_z or z_raw in (None, 0, "0", "0.0") else z_raw)
    if meta.get("substitution"):
        warnings.append({"code": "text-fidelity-fallback", "layer_id": layer_id, **meta["substitution"]})
    common = {
        "id": layer_id,
        "name": _name(candidate),
        "box": box,
        "z_index": z_index,
        "visible_box": candidate.get("visible_box") or candidate.get("ink_box"),
        "rotation": float(candidate.get("rotation", 0) or 0),
        "opacity": float(candidate.get("opacity", 1) if candidate.get("opacity") is not None else 1),
        "blend_mode": str(candidate.get("blend_mode") or "NORMAL"),
        "effects": list(source_effects) if isinstance(source_effects, list) else [],
        "meta": {**meta, "z": z_index, "source_id": layer_id},
        "constraints": dict(candidate.get("constraints") or {}),
        "component": dict(candidate.get("component") or {}),
        "layout": dict(candidate.get("layout") or {}),
        # Preserve any sizing an upstream stage already supplied; the geometry-evidence
        # inference below (_infer_sizing) fills the rest, first-writer-wins.
        "sizing": dict(candidate.get("sizing") or {}),
    }
    passthrough = {key: value for key, value in candidate.items()
                   if key not in _CONSUMED_CANDIDATE_KEYS}
    if passthrough:
        common["meta"]["passthrough"] = passthrough

    if target == "group":
        children = []
        for child in candidate.get("children") or []:
            for piece in _split_weight_run_siblings(child):
                try:
                    children.append(_compile(piece, run_dir, warnings))
                except Exception as exc:
                    warnings.append({
                        "code": "layer-compile-error", "layer_id": piece.get("id"),
                        "detail": str(exc),
                    })
        # F1: a container that carries its OWN raster material (an image/photo/product
        # host that was promoted to a group) must not lose those pixels. A Figma FRAME
        # has no image fill, so re-emit the host raster as a background image child that
        # fills the frame behind every other child. Without this a full-bleed product
        # panel promoted to a container compiles to an empty group and the removal mask +
        # inpaint erase the real product (benchmark 002: 1025x1418 panel emptied).
        raster_src = candidate.get("src")
        if raster_src:
            staged = _stage_asset(raster_src, f"{layer_id}__hostbg", run_dir, warnings)
            if staged:
                host_mask = (dict(candidate.get("mask"))
                             if isinstance(candidate.get("mask"), dict) else None)
                if host_mask and host_mask.get("src"):
                    host_mask["src"] = _stage_asset(
                        host_mask.get("src"), f"{layer_id}__hostbg_mask", run_dir, warnings)
                bg_z = min((child.z_index for child in children), default=z_index) - 1.0
                children.append(Layer(
                    id=f"{layer_id}__hostbg",
                    type="image",
                    name=_name(candidate),
                    box={"x": 0.0, "y": 0.0,
                         "w": box.get("w", 1), "h": box.get("h", 1)},
                    z_index=bg_z,
                    src=staged,
                    mask=host_mask,
                    constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
                    meta={"role": str((candidate.get("meta") or {}).get("role") or "image"),
                          "source": "group-host-raster", "z": bg_z,
                          "source_id": f"{layer_id}__hostbg",
                          "preserved_host_raster": True},
                ))
        children.sort(key=lambda child: child.z_index)
        return Layer(
            type="group",
            children=children,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            radius=candidate.get("radius") or source_style.get("radius"),
            style=source_style,
            shape_kind="frame",
            **common,
        )

    if target == "text":
        style = source_style
        fill = candidate.get("fill") or style.pop("fill", None)
        stroke = candidate.get("stroke") or style.pop("stroke", None)
        raw_text = str(candidate.get("text") or "")
        text_value = _strip_edge_emoji(raw_text)
        emoji_shift = raw_text.find(text_value) if text_value != raw_text else 0
        if text_value != raw_text:
            common["meta"]["emoji_stripped"] = raw_text
        _promote_weight_candidate(style)
        stroke, common["effects"] = _normalize_text_stroke(
            stroke, style, list(common.get("effects") or []))
        # Fit against ink/painted bounds when available so Python preview and the Figma
        # plugin agree on the same target box (plugin uses visible_box in fitTextToVisibleBox).
        fit_box = dict(
            candidate.get("visible_box") or candidate.get("ink_box") or common["box"]
        )
        fitted_box, auto_resize, style_patch = fit_text_box(text_value, style, fit_box)
        style.update(style_patch)
        # Tracking policy (Codia parity): emitted letterSpacing is always 0 —
        # fit_text_box measures untracked, and runs must agree with the base style.
        style["letterSpacing"] = 0.0
        # Anti-clip: never ship lh < fs (display headlines / CTA labels).
        try:
            _fs = float(style.get("fontSize") or 0)
            _lh = float(style.get("lineHeight") or 0)
            if _fs > 0 and (_lh <= 0 or _lh < _fs * 1.05):
                style["lineHeight"] = round(_fs * 1.12, 2)
        except (TypeError, ValueError):
            pass
        text_runs = []
        for run in list(candidate.get("text_runs") or []):
            if not isinstance(run, dict):
                continue
            run = dict(run)
            if emoji_shift or text_value != raw_text:
                try:
                    start = int(run.get("start", 0)) - emoji_shift
                    end = int(run.get("end", 0)) - emoji_shift
                except (TypeError, ValueError):
                    continue
                start, end = max(0, start), min(len(text_value), end)
                if end <= start:
                    continue
                run["start"], run["end"] = start, end
                if run.get("text") is not None:
                    run["text"] = text_value[start:end]
            run_style = run.get("style")
            if isinstance(run_style, dict):
                run_style = dict(run_style)
                if "letterSpacing" in run_style:
                    run_style["letterSpacing"] = 0.0
                _promote_weight_candidate(run_style)
                run["style"] = run_style
            text_runs.append(run)
        # Codia-style GENEROUS text boxes (anti-clipping): the tight ink box becomes a
        # loose box >= 1.6x lineHeight per line, grown symmetrically around the ink
        # center (vertical CENTER alignment keeps the visual baseline put), with ~6%
        # width slack away from the horizontal anchor. Preview, plugin and QA all read
        # the same grown box; the pre-fit ink evidence survives in meta.
        generous = _generous_text_box(fitted_box, style, text_value, stroke=stroke)
        common["box"] = generous
        common["visible_box"] = dict(generous)
        common["meta"]["prefit_ink_box"] = dict(fitted_box)
        style.setdefault("verticalAlign", "CENTER")
        style.setdefault("autoResize", auto_resize)
        style["preFitted"] = True
        style["fit"] = False
        return Layer(
            type="text",
            text=text_value,
            style=style,
            text_runs=text_runs,
            fill=fill,
            stroke=stroke,
            **common,
        )

    if target == "shape":
        return Layer(
            type="shape",
            shape_kind=candidate.get("shape_kind") or "rect",
            path=candidate.get("path"),
            svg=candidate.get("svg"),
            src=_stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
                if candidate.get("src") else None,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            radius=candidate.get("radius") or source_style.get("radius"),
            style=source_style,
            **common,
        )

    if target == "icon":
        paths = list(candidate.get("paths") or [])
        svg = candidate.get("svg")
        path = candidate.get("path") or (paths[0].get("d") if len(paths) == 1 else None)
        return Layer(
            type="shape",
            shape_kind="path",
            path=path,
            svg=svg,
            src=_stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
                if candidate.get("src") else None,
            fill=candidate.get("fill"),
            stroke=candidate.get("stroke"),
            style=source_style,
            meta={**common.pop("meta"), "vector_paths": paths},
            **common,
        )

    # Unknown candidates route conservatively to an alpha raster, never a fake gray box.
    src = _stage_asset(candidate.get("src"), layer_id, run_dir, warnings)
    layer_meta = common.pop("meta")
    if not src:
        layer_meta["compiler_error"] = "missing image asset"
    mask = dict(candidate.get("mask") or {}) if isinstance(candidate.get("mask"), dict) else None
    if mask and mask.get("src"):
        mask["src"] = _stage_asset(mask.get("src"), f"{layer_id}_mask", run_dir, warnings)
    return Layer(type="image", src=src, mask=mask, style=source_style, meta=layer_meta, **common)


# ── Flat/banded plate → solid RECTs (Codia plate strategy, spec §4/§7.5) ──────────
# When the clean plate is a stack of near-uniform horizontal bands (009: #060606 nav
# strip over #000000 body), Codia ships SOLID rectangles, not an inpainted PNG: cleaner,
# lighter and trivially editable. Reserve the raster plate for photographic backgrounds.
# Config gate: background.solid_plate (default ON). Deliberately strict thresholds —
# any texture/vignette keeps the raster plate.
_SOLID_PLATE_ROW_STD_MAX = 4.0     # per-row pixel std (mean over channels)
_SOLID_PLATE_BAND_DELTA = 4.0      # max channel delta of a row vs its band's color
                                   # (009's real bands differ by exactly 6 — keep below)
_SOLID_PLATE_MAX_BANDS = 4
_SOLID_PLATE_MIN_BAND_ROWS = 8


def _solid_plate_bands(plate_path, canvas, plate_rgb=None) -> Optional[list]:
    """Return [{y, h, color}] in canvas coords when the plate is flat/banded, else None.

    ``plate_rgb`` may be a preloaded HxWx3 array (uint8 or float) to skip a duplicate decode.
    """
    if plate_rgb is None and not plate_path:
        return None
    try:
        import numpy as np
        if plate_rgb is not None:
            plate = np.asarray(plate_rgb, dtype=np.float32)
            if plate.ndim != 3 or plate.shape[2] < 3:
                return None
            if plate.shape[2] > 3:
                plate = plate[:, :, :3]
        else:
            from PIL import Image
            with Image.open(plate_path) as image:
                plate = np.asarray(image.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    h, w = plate.shape[:2]
    if h < 16 or w < 16:
        return None
    row_std = plate.std(axis=1).mean(axis=1)
    if float(row_std.max()) > _SOLID_PLATE_ROW_STD_MAX:
        return None  # texture somewhere: not a flat/banded plate
    row_color = plate.mean(axis=1)  # (h, 3)
    bands = []  # [start, end, sum_color]
    for y in range(h):
        color = row_color[y]
        if bands:
            start, end, total = bands[-1]
            mean = total / (end - start)
            if float(np.abs(color - mean).max()) <= _SOLID_PLATE_BAND_DELTA:
                bands[-1] = [start, end + 1, total + color]
                continue
        bands.append([y, y + 1, row_color[y].copy()])
    # Only a genuinely BANDED plate (>=2 flat bands, e.g. 009's #060606 nav strip over
    # #000000 body) earns editable rects; a single uniform plate stays raster-only so
    # trivial flat fixtures/scenes don't grow an extra layer.
    if not (2 <= len(bands) <= _SOLID_PLATE_MAX_BANDS):
        return None
    if any((end - start) < _SOLID_PLATE_MIN_BAND_ROWS for start, end, _ in bands):
        return None
    scale = float(canvas.get("h", h)) / h
    out = []
    for start, end, total in bands:
        mean = total / (end - start)
        color = "#%02x%02x%02x" % tuple(int(round(min(255.0, max(0.0, c)))) for c in mean)
        out.append({"y": round(start * scale, 2), "h": round((end - start) * scale, 2),
                    "color": color})
    return out


# ── Background-per-group (Codia region/card construction) ─────────────────────────
# Every Codia region group that sits on a distinct plate region (card/band/callout)
# contains its OWN "Background" rect with an IMAGE fill = that region's slice of the
# clean plate (076 testimonial cards, 052 per-band rects). Groups become self-contained
# and movable. The slice comes from background_clean (honoring all removal work), so
# nothing double-prints. Config gate: background.per_group (default ON).
_GROUP_BG_MIN_AREA_PX = 2000
_GROUP_BG_MAX_CANVAS_FRAC = 0.72
_GROUP_BG_COLOR_DELTA = 8.0
_GROUP_BG_TEXTURE_STD = 10.0


def _add_group_backgrounds(layers, canvas, run_dir, base_src, warnings, plate_rgb=None) -> int:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return 0
    plate_path = _resolve(base_src, run_dir)
    if not plate_path and plate_rgb is None:
        return 0
    try:
        if plate_rgb is not None:
            plate = np.asarray(plate_rgb, dtype=np.float32)
            if plate.ndim != 3 or plate.shape[2] < 3:
                return 0
            if plate.shape[2] > 3:
                plate = plate[:, :, :3]
        else:
            with Image.open(plate_path) as image:
                plate = np.asarray(image.convert("RGB"), dtype=np.float32)
    except Exception as exc:
        warnings.append({"code": "group-background-error", "detail": str(exc)})
        return 0
    ph, pw = plate.shape[:2]
    canvas_area = max(1.0, float(canvas.get("w", pw)) * float(canvas.get("h", ph)))
    scale_x = pw / max(1.0, float(canvas.get("w", pw)))
    scale_y = ph / max(1.0, float(canvas.get("h", ph)))
    added = 0

    def visit(layer, off_x: float, off_y: float):
        nonlocal added
        box = layer.box or {}
        abs_x = off_x + float(box.get("x", 0) or 0)
        abs_y = off_y + float(box.get("y", 0) or 0)
        for child in layer.children or []:
            visit(child, abs_x, abs_y)
        if layer.type != "group" or not layer.children:
            return
        gid = str(layer.id or "group")
        if any(str(child.id or "").endswith(("__hostbg", "__groupbg"))
               for child in layer.children):
            return  # already self-contained (host raster / earlier pass)
        if layer.fill:
            return  # group already paints its own surface
        w = float(box.get("w", 0) or 0)
        h = float(box.get("h", 0) or 0)
        if w * h < _GROUP_BG_MIN_AREA_PX or w * h > _GROUP_BG_MAX_CANVAS_FRAC * canvas_area:
            return
        x0 = int(round(abs_x * scale_x)); y0 = int(round(abs_y * scale_y))
        x1 = int(round((abs_x + w) * scale_x)); y1 = int(round((abs_y + h) * scale_y))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(pw, x1), min(ph, y1)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return
        if (cx1 - cx0) * (cy1 - cy0) < 0.6 * max(1.0, (x1 - x0) * (y1 - y0)):
            return  # mostly off-plate
        tile = plate[cy0:cy1, cx0:cx1]
        # Distinctness gate: only groups on their OWN plate region (card/band/callout)
        # get a backdrop — a group floating on the shared page plate (009 engagement
        # row on flat black) must not duplicate it.
        ring = 8
        rx0, ry0 = max(0, cx0 - ring), max(0, cy0 - ring)
        rx1, ry1 = min(pw, cx1 + ring), min(ph, cy1 + ring)
        ring_px = []
        if ry0 < cy0:
            ring_px.append(plate[ry0:cy0, rx0:rx1].reshape(-1, 3))
        if cy1 < ry1:
            ring_px.append(plate[cy1:ry1, rx0:rx1].reshape(-1, 3))
        if rx0 < cx0:
            ring_px.append(plate[cy0:cy1, rx0:cx0].reshape(-1, 3))
        if cx1 < rx1:
            ring_px.append(plate[cy0:cy1, cx1:rx1].reshape(-1, 3))
        if not ring_px:
            return  # full-canvas group: the root plate already is its backdrop
        ring_all = np.concatenate(ring_px, axis=0)
        color_delta = float(np.abs(tile.mean(axis=(0, 1)) - np.median(ring_all, axis=0)).max())
        texture = float(tile.std(axis=(0, 1)).mean())
        if color_delta < _GROUP_BG_COLOR_DELTA and texture < _GROUP_BG_TEXTURE_STD:
            return
        assets = os.path.join(run_dir, "assets")
        os.makedirs(assets, exist_ok=True)
        rel = os.path.join("assets", f"{gid}__groupbg_plate.png")
        try:
            Image.fromarray(tile.astype("uint8")).save(os.path.join(run_dir, rel))
        except Exception as exc:
            warnings.append({"code": "group-background-error", "layer_id": gid,
                             "detail": str(exc)})
            return
        bg_z = min((child.z_index for child in layer.children), default=0.0) - 1.0
        layer.children.insert(0, Layer(
            id=f"{gid}__groupbg", type="image", name="Background",
            box={"x": 0.0, "y": 0.0, "w": w, "h": h},
            z_index=bg_z, src=rel.replace("\\", "/"),
            constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
            # The slice is cut from background_clean (the clean inpaint plate) — see module
            # docstring — so it is honestly sourced from the inpaint plate. Present
            # source="inpaint" for the ownership gate; keep provenance in plate_source.
            meta={"role": "background", "source": "inpaint",
                  "plate_source": "group-plate-slice", "z": bg_z,
                  "source_id": f"{gid}__groupbg", "per_group_background": True,
                  "plate_delta": round(color_delta, 2), "plate_texture": round(texture, 2)},
        ))
        added += 1

    for root in layers:
        visit(root, 0.0, 0.0)
    return added


# ── Single-ownership audit (one owner per pixel) ──────────────────────────────────
# Codia's construction contract: every pixel is owned exactly once — a region is EXACTLY
# ONE of {native text, raster slice, kept-in-photo baked}. When a native TEXT layer is
# emitted over a raster carrier (the clean plate, a host-raster product panel, a group
# plate slice, or a raster slice) that ALSO carries the same content baked in, both render
# and the result double-prints (002 "WHEYMILKSHAKE", 009 "geld geld"). This audit gives the
# native text sole ownership by erasing its baked ink from the carrier beneath it. Erasure
# is conservative: only uniform (non-photographic) plate regions under the text are cleaned,
# so a genuine textured photo is never smeared. Config gate: design.single_ownership.
_OWNERSHIP_CARRIER_ROLES = frozenset({
    "background", "photo", "image", "product", "product-cluster", "plate",
    "photo-fragment", "panel",
})


def _flatten_abs(layers):
    """Yield ``(layer, parent_off_x, parent_off_y)`` for every layer in the tree.

    ``parent_off`` is the accumulated origin of the layer's ancestors; the layer's own
    ``box`` x/y is added by the caller (Codia layers are parent-local)."""
    out = []

    def rec(layer, ox, oy):
        out.append((layer, ox, oy))
        bx = float((layer.box or {}).get("x", 0) or 0)
        by = float((layer.box or {}).get("y", 0) or 0)
        for child in layer.children or []:
            rec(child, ox + bx, oy + by)

    for root in layers:
        rec(root, 0.0, 0.0)
    return out


def _erase_baked_ink(asset_arr, region, np, tolerance=26.0, uniform_fraction=0.62,
                     ring=4, dilate=2):
    """Erase baked text ink inside ``region`` of a carrier asset, plate-uniform regions only.

    Returns ``"cleaned"``/``"noop"``/``"textured"``. ``region`` is (x0,y0,x1,y1) in asset
    pixels. The plate colour is the median of a ring just outside the ink box; ink is any
    pixel that differs from it beyond ``tolerance``. Only fires when the ring is genuinely
    uniform (so a photo under the text is left untouched)."""
    h, w = asset_arr.shape[:2]
    x0, y0, x1, y1 = region
    x0 = max(0, min(w, int(round(x0)))); x1 = max(0, min(w, int(round(x1))))
    y0 = max(0, min(h, int(round(y0)))); y1 = max(0, min(h, int(round(y1))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return "noop"
    rx0 = max(0, x0 - ring); ry0 = max(0, y0 - ring)
    rx1 = min(w, x1 + ring); ry1 = min(h, y1 + ring)
    window = asset_arr[ry0:ry1, rx0:rx1, :3].astype(np.float32)
    ring_mask = np.ones(window.shape[:2], dtype=bool)
    ring_mask[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False
    ring_px = window[ring_mask]
    if ring_px.shape[0] < 12:
        return "noop"
    plate = np.median(ring_px, axis=0)
    near_ring = np.mean(np.max(np.abs(ring_px - plate), axis=1) <= tolerance)
    if float(near_ring) < uniform_fraction:
        return "textured"  # ring is not a uniform plate — a photo, leave it to the raster
    interior = asset_arr[y0:y1, x0:x1, :3].astype(np.float32)
    ink = np.max(np.abs(interior - plate), axis=2) > tolerance
    if not ink.any():
        return "noop"
    if dilate > 0:
        try:
            cv2, _np, _ = _lazy_cv2()
            kernel = _np.ones((2 * dilate + 1, 2 * dilate + 1), _np.uint8)
            ink = cv2.dilate(ink.astype(_np.uint8), kernel) > 0
        except Exception:
            pass
    patch = asset_arr[y0:y1, x0:x1]
    patch[..., :3][ink] = plate.astype(asset_arr.dtype)
    asset_arr[y0:y1, x0:x1] = patch
    return "cleaned"


def _lazy_cv2():
    import cv2
    import numpy as np
    return cv2, np, None


def _audit_single_ownership(layers, run_dir, canvas, warnings, cfg):
    """Give native text sole ownership by erasing its baked duplicate from carriers beneath."""
    scfg = ((cfg or {}).get("design") or {}).get("single_ownership") or {}
    if not bool(scfg.get("enabled", True)):
        return {"enabled": False, "collapsed": 0}
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return {"enabled": False, "collapsed": 0, "reason": "numpy/PIL unavailable"}
    overlap_frac = float(scfg.get("min_overlap_fraction", 0.55))
    tolerance = float(scfg.get("ink_tolerance", 26.0))
    uniform_fraction = float(scfg.get("uniform_fraction", 0.62))
    flat = _flatten_abs(layers)

    texts = []
    for layer, ox, oy in flat:
        if layer.type != "text" or not str(layer.text or "").strip():
            continue
        meta = layer.meta or {}
        if fallback_kind(meta):  # raster slice / masked-pixel fallback: not native ink
            continue
        box = layer.box or {}
        ink = meta.get("prefit_ink_box") or box
        ax = ox + float(ink.get("x", box.get("x", 0)) or 0)
        ay = oy + float(ink.get("y", box.get("y", 0)) or 0)
        aw = float(ink.get("w", box.get("w", 0)) or 0)
        ah = float(ink.get("h", box.get("h", 0)) or 0)
        if aw <= 0 or ah <= 0:
            continue
        texts.append((layer, ax, ay, aw, ah))

    carriers = []
    for layer, ox, oy in flat:
        if not layer.src:
            continue
        meta = layer.meta or {}
        role = str(meta.get("role") or "").lower()
        is_carrier = (
            role in _OWNERSHIP_CARRIER_ROLES
            or str(meta.get("source") or "").lower() == "inpaint"
            or meta.get("plate_source") or meta.get("preserved_host_raster")
            or meta.get("per_group_background") or fallback_kind(meta) == "raster-slice"
        )
        if not is_carrier:
            continue
        box = layer.box or {}
        cw = float(box.get("w", 0) or 0)
        ch = float(box.get("h", 0) or 0)
        if cw <= 0 or ch <= 0:
            continue
        carriers.append({
            "layer": layer, "x": ox + float(box.get("x", 0) or 0),
            "y": oy + float(box.get("y", 0) or 0), "w": cw, "h": ch,
        })

    # Group erasures per physical asset so each PNG is opened/written once.
    per_asset: dict = {}
    plan = []
    for tlayer, ax, ay, aw, ah in texts:
        tz = float(tlayer.z_index or 0)
        for carrier in carriers:
            clayer = carrier["layer"]
            if clayer is tlayer or float(clayer.z_index or 0) >= tz:
                continue
            ix0 = max(ax, carrier["x"]); iy0 = max(ay, carrier["y"])
            ix1 = min(ax + aw, carrier["x"] + carrier["w"])
            iy1 = min(ay + ah, carrier["y"] + carrier["h"])
            if ix1 - ix0 <= 1 or iy1 - iy0 <= 1:
                continue
            if (ix1 - ix0) * (iy1 - iy0) < overlap_frac * (aw * ah):
                continue
            plan.append((carrier, tlayer, (ix0, iy0, ix1, iy1)))
            per_asset.setdefault(str(clayer.src), carrier)

    if not plan:
        return {"enabled": True, "collapsed": 0, "carriers_cleaned": 0}

    loaded: dict = {}
    result = {"cleaned": 0, "textured": 0, "noop": 0, "assets": []}
    for carrier, tlayer, (ix0, iy0, ix1, iy1) in plan:
        clayer = carrier["layer"]
        key = str(clayer.src)
        if key not in loaded:
            resolved = _resolve(clayer.src, run_dir)
            if not resolved:
                continue
            try:
                with Image.open(resolved) as image:
                    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
            except Exception as exc:
                warnings.append({"code": "single-ownership-error", "layer_id": clayer.id,
                                 "detail": str(exc)})
                loaded[key] = None
                continue
            loaded[key] = {"arr": arr, "path": resolved, "dirty": False}
        entry = loaded[key]
        if entry is None:
            continue
        arr = entry["arr"]
        ah_px, aw_px = arr.shape[:2]
        sx = aw_px / max(1e-6, carrier["w"]); sy = ah_px / max(1e-6, carrier["h"])
        region = ((ix0 - carrier["x"]) * sx, (iy0 - carrier["y"]) * sy,
                  (ix1 - carrier["x"]) * sx, (iy1 - carrier["y"]) * sy)
        status = _erase_baked_ink(arr, region, np, tolerance=tolerance,
                                  uniform_fraction=uniform_fraction)
        result[status] = result.get(status, 0) + 1
        if status == "cleaned":
            entry["dirty"] = True

    for key, entry in loaded.items():
        if entry and entry.get("dirty"):
            try:
                Image.fromarray(entry["arr"]).save(entry["path"])
                result["assets"].append(os.path.relpath(entry["path"], run_dir))
            except Exception as exc:
                warnings.append({"code": "single-ownership-error", "detail": str(exc)})
    return {
        "enabled": True,
        "collapsed": result["cleaned"],
        "carriers_cleaned": len(result["assets"]),
        "textured_skipped": result["textured"],
        "noop": result["noop"],
    }


def _count_layers(layers):
    return sum(1 + _count_layers(layer.children) for layer in layers)


def _count_editable(layers):
    return sum((1 if layer.type in ("text", "shape", "group") else 0) +
               _count_editable(layer.children) for layer in layers)


_LEGITIMATE_RASTER_ROLES = frozenset({
    "background", "photo", "image", "product", "product-cluster", "person",
    "people", "face", "hand", "avatar", "profile", "profile-photo", "thumbnail",
    "illustration", "package", "logo", "wordmark", "brand", "logotype",
})


def _leaf_accounting(layers):
    """Describe real foreground material without letting wrapper groups inflate editability.

    The historical editable ratio counted every FRAME/GROUP as editable, even when that frame
    contained only one raster screenshot. Keep the old metric for compatibility, but publish a
    leaf-only accounting contract that acceptance QA can audit honestly.
    """
    out = {
        "foreground_leaf_count": 0,
        "native_leaf_count": 0,
        "raster_leaf_count": 0,
        "intentional_raster_cluster_count": 0,
        "fallback_raster_count": 0,
        "raster_slice_count": 0,
        "raster_slice_ids": [],
        "unexplained_raster_count": 0,
        "unexplained_raster_ids": [],
    }

    def visit(layer):
        children = list(layer.children or [])
        if children:
            for child in children:
                visit(child)
            return
        meta = layer.meta or {}
        role = str(meta.get("role") or "").strip().lower().replace("_", "-")
        if role == "background":
            return
        out["foreground_leaf_count"] += 1
        if layer.type in ("text", "shape"):
            out["native_leaf_count"] += 1
            return
        if layer.type != "image":
            return
        out["raster_leaf_count"] += 1
        intentional = bool(meta.get("intentional_raster_cluster")) or is_intentional_raster_cluster(role)
        if intentional:
            out["intentional_raster_cluster_count"] += 1
        # "fallback" flags (fallback/raster_fallback/vector_fallback/substitution/low_fidelity)
        # are set by routing/vectorize exactly when they GIVE UP on producing a native layer.
        # A bare give-up is NOT self-justifying: a raster leaf only counts as explained when
        # it has an INDEPENDENT legitimate reason. The canonical fallback disposition is read
        # through schema.fallback_kind so this classification can never diverge from the other
        # readers (reconstruct/repair/pixel_diff) again (F11).
        kind = fallback_kind(meta)
        fallback = bool(
            meta.get("fallback") or meta.get("raster_fallback") or meta.get("vector_fallback")
            or meta.get("substitution") or meta.get("low_fidelity")
        )
        if fallback:
            out["fallback_raster_count"] += 1
        # A confidence-gated raster slice is a DOCUMENTED give-up: it carries the QA
        # evidence that rejected the editable attempt (fallback_scores) and preserves
        # that attempt for repair (fallback_editable). The editability cost still shows
        # up honestly in native_leaf_ratio; QA reports slice ids separately.
        raster_slice = bool(kind == "raster-slice" and meta.get("fallback_scores"))
        if raster_slice:
            out["raster_slice_count"] += 1
            out["raster_slice_ids"].append(str(layer.id))
        # A fidelity-image substitution (text->image, masked-pixel wordmark, vector/raster
        # fallback) is explained-but-non-native: it records WHY it gave up, so it is not a
        # quiet, unaccountable raster. It is only "explained" when that justification is
        # actually present (substitution details / fallback_scores / low_fidelity) -- a bare
        # marker with no evidence is still an unexplained give-up (F4 anti-laundering).
        fidelity_image = bool(kind == "fidelity-image" and (
            meta.get("substitution") or meta.get("fallback_scores")
            or meta.get("low_fidelity")))
        legitimate = bool(intentional or role in _LEGITIMATE_RASTER_ROLES or meta.get("wordmark"))
        explained = legitimate or raster_slice or fidelity_image
        if fallback and not explained:
            out["unexplained_raster_count"] += 1
            out["unexplained_raster_ids"].append(str(layer.id))

    for root in layers:
        visit(root)
    denominator = max(1, out["foreground_leaf_count"])
    out["native_leaf_ratio"] = round(out["native_leaf_count"] / denominator, 4)
    out["unexplained_raster_ids"] = sorted(out["unexplained_raster_ids"])
    out["raster_slice_ids"] = sorted(out["raster_slice_ids"])
    return out


# ── Per-dimension sizing inference (Codia DimensionSpec parity) ────────────────────
# Geometry-evidence based and deliberately CONSERVATIVE. Sizing is assigned only to
# layers that sit inside a real auto-layout stack (layout.mode HORIZONTAL/VERTICAL) and
# to the stack container itself. Outside auto layout we leave sizing empty, so absolute
# layers keep their pixel box and existing `constraints` untouched (no regression).
#
# It consumes ONLY fields layout.py already emits today — layout["mode"], layout["padding"]
# and the text style["autoResize"] fit_text_box stamps — and tolerates their absence, so
# a parallel agent adding/renaming layout fields cannot break it. All axis writes go
# through _set_sizing_axis (first-writer-wins) so a PARENT's fill decision on an axis is
# never overwritten by a nested container's own hug on that same axis (parent is visited
# first, top-down).
_SIZING_SPAN_FRACTION = 0.90   # child extent / container inner extent read as "spans full"
_SIZING_HUG_TOLERANCE = 0.06   # container box within 6% of children+padding => content-sized
_BUTTON_SIZING_ROLES = frozenset({"button", "cta", "badge", "chip", "pill", "tag"})


def _sizing_layout_mode(layer) -> Optional[str]:
    mode = str((getattr(layer, "layout", None) or {}).get("mode") or "").strip().upper()
    if mode == "ROW":
        return "HORIZONTAL"
    if mode == "COLUMN":
        return "VERTICAL"
    return mode if mode in ("HORIZONTAL", "VERTICAL") else None


def _sizing_padding(layout) -> dict:
    """Normalize layout['padding'] (dict|number|[t,r,b,l]|[v,h]) to l/r/t/b (mirrors code.js)."""
    raw = (layout or {}).get("padding")
    if isinstance(raw, (int, float)):
        v = float(raw)
        return {"left": v, "right": v, "top": v, "bottom": v}
    if isinstance(raw, (list, tuple)):
        if len(raw) == 2:
            return {"top": float(raw[0]), "right": float(raw[1]),
                    "bottom": float(raw[0]), "left": float(raw[1])}
        if len(raw) == 4:
            return {"top": float(raw[0]), "right": float(raw[1]),
                    "bottom": float(raw[2]), "left": float(raw[3])}
    src = raw if isinstance(raw, dict) else (layout or {})

    def _g(*keys):
        for k in keys:
            value = src.get(k)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    return {
        "left": _g("left", "paddingLeft", "padding_left"),
        "right": _g("right", "paddingRight", "padding_right"),
        "top": _g("top", "paddingTop", "padding_top"),
        "bottom": _g("bottom", "paddingBottom", "padding_bottom"),
    }


def _set_sizing_axis(layer, axis: str, value: str) -> None:
    """Write one sizing axis only if still unset (first-writer-wins)."""
    sizing = getattr(layer, "sizing", None)
    if not isinstance(sizing, dict):
        sizing = {}
        layer.sizing = sizing
    if not sizing.get(axis):
        sizing[axis] = value


def _sizing_is_button_like(layer) -> bool:
    role = str((getattr(layer, "meta", None) or {}).get("role") or "").strip().lower().replace("_", "-")
    return role in _BUTTON_SIZING_ROLES


def _sizing_is_absolute_child(child) -> bool:
    child_layout = getattr(child, "layout", None) or {}
    pos = str(child_layout.get("positioning") or child_layout.get("layoutPositioning") or "").strip().upper()
    return pos == "ABSOLUTE" or child_layout.get("absolute") is True


def _infer_child_sizing(child, mode: str, inner_w: float, inner_h: float) -> None:
    """Assign child.sizing from geometry evidence inside an auto-layout container.

    Axis vocabulary: a VERTICAL stack's CROSS axis is horizontal (width); a HORIZONTAL
    stack's cross axis is vertical (height). Children fill the cross axis when they span
    it (full-width divider / full-height rail), text hugs to its glyph run, buttons hug
    both, everything else stays fixed on the main axis.
    """
    cbox = getattr(child, "box", None) or {}
    cw = float(cbox.get("w", 0) or 0)
    ch = float(cbox.get("h", 0) or 0)
    cross_is_width = mode == "VERTICAL"
    spans_w = inner_w > 0 and cw >= inner_w * _SIZING_SPAN_FRACTION
    spans_h = inner_h > 0 and ch >= inner_h * _SIZING_SPAN_FRACTION

    # Button / pill frames hug both axes (their padding is already on layout).
    if child.type == "group" and _sizing_is_button_like(child):
        _set_sizing_axis(child, "w", "hug")
        _set_sizing_axis(child, "h", "hug")
        return

    if child.type == "text":
        auto = str((getattr(child, "style", None) or {}).get("autoResize") or "").strip().upper()
        if auto in ("WIDTH", "WIDTH_AND_HEIGHT"):
            # single-line label: shrink to the glyph run on both axes
            _set_sizing_axis(child, "w", "hug")
            _set_sizing_axis(child, "h", "hug")
        elif auto == "HEIGHT":
            # wrapping paragraph: height hugs the wrapped lines; width fills the container
            # when it spans the cross axis, else keeps its measured width.
            _set_sizing_axis(child, "w", "fill" if (cross_is_width and spans_w) else "fixed")
            _set_sizing_axis(child, "h", "hug")
        else:
            # no usable font metrics (autoResize NONE / unset): keep the painted box.
            _set_sizing_axis(child, "w", "fixed")
            _set_sizing_axis(child, "h", "fixed")
        return

    # Non-text leaf/sub-frame (divider, image band, nested stack): fill the cross axis
    # when it spans it, fixed on the main axis.
    if cross_is_width:
        _set_sizing_axis(child, "w", "fill" if spans_w else "fixed")
        _set_sizing_axis(child, "h", "fixed")
    else:
        _set_sizing_axis(child, "h", "fill" if spans_h else "fixed")
        _set_sizing_axis(child, "w", "fixed")


def _infer_container_sizing(container, mode: str, padding: dict) -> None:
    """Hug the container's STACKING axis when its box tightly wraps children+padding."""
    children = getattr(container, "children", None) or []
    if not children:
        return
    cbox = getattr(container, "box", None) or {}
    cw = float(cbox.get("w", 0) or 0)
    ch = float(cbox.get("h", 0) or 0)
    xs0, ys0, xs1, ys1 = [], [], [], []
    for c in children:
        b = getattr(c, "box", None) or {}
        x = float(b.get("x", 0) or 0)
        y = float(b.get("y", 0) or 0)
        xs0.append(x)
        ys0.append(y)
        xs1.append(x + float(b.get("w", 0) or 0))
        ys1.append(y + float(b.get("h", 0) or 0))
    # Children carry parent-local coords, so their union span already includes the gaps.
    content_w = (max(xs1) - min(xs0)) + padding["left"] + padding["right"]
    content_h = (max(ys1) - min(ys0)) + padding["top"] + padding["bottom"]
    if mode == "VERTICAL":
        if ch > 0 and abs(ch - content_h) <= ch * _SIZING_HUG_TOLERANCE:
            _set_sizing_axis(container, "h", "hug")
    else:  # HORIZONTAL
        if cw > 0 and abs(cw - content_w) <= cw * _SIZING_HUG_TOLERANCE:
            _set_sizing_axis(container, "w", "hug")


def _infer_sizing(layer) -> None:
    """Top-down sizing inference over one compiled Layer subtree.

    Parent-first ordering matters: a container assigns its children's sizing BEFORE the
    recursion descends into each child, so a parent's cross-axis FILL claim is written
    first and a nested container's own-axis HUG (first-writer-wins) will not clobber it.
    """
    mode = _sizing_layout_mode(layer)
    if mode:
        padding = _sizing_padding(getattr(layer, "layout", None))
        cbox = getattr(layer, "box", None) or {}
        inner_w = max(0.0, float(cbox.get("w", 0) or 0) - padding["left"] - padding["right"])
        inner_h = max(0.0, float(cbox.get("h", 0) or 0) - padding["top"] - padding["bottom"])
        _infer_container_sizing(layer, mode, padding)
        for child in getattr(layer, "children", None) or []:
            if _sizing_is_absolute_child(child):
                continue
            _infer_child_sizing(child, mode, inner_w, inner_h)
    for child in getattr(layer, "children", None) or []:
        _infer_sizing(child)


def build(candidates: list, canvas: dict, run_dir: str, base_src: str | None = None,
          doc_id: str = "doc", name: str = "design", kept_in_photo: Optional[list] = None,
          cfg: Optional[dict] = None) -> DesignDoc:
    """Build schema v2.

    ``base_src`` must be a reconstructed clean plate. Refusing the normalized/original source
    here prevents the old duplicate-elements architecture from silently returning.

    ``cfg`` is optional (callers that predate the background-gradient feature omit it, which
    keeps the defaults ON); it only gates the editable-gradient-background detection.
    """
    os.makedirs(run_dir, exist_ok=True)
    warnings = []
    if base_src and candidates and os.path.basename(base_src).lower() in ("normalized.png", "original.png"):
        raise ValueError("refusing untouched source as rebuilt background; run reconstruct/inpaint first")

    layers = []
    plate_u8 = None
    if base_src:
        base_rel = _stage_asset(base_src, "background", run_dir, warnings)
        layers.append(Layer(
            id="background", type="image", name="Background",
            box={"x": 0, "y": 0, "w": canvas["w"], "h": canvas["h"]},
            z_index=-1_000_000, src=base_rel,
            constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
            meta={"source": "inpaint", "role": "background", "z": -1_000_000},
        ))
        # Flat/banded plate → editable SOLID rect(s) just above the raster plate (Codia
        # ships solid rects for UI screenshots; the raster floor beneath keeps fidelity
        # guaranteed while the rects make the plate natively editable).
        resolved_base = _resolve(base_src, run_dir)
        # Decode the clean plate once for solid-band / gradient / per-group background work.
        plate_u8 = None
        if resolved_base:
            try:
                import numpy as np
                from PIL import Image
                with Image.open(resolved_base) as image:
                    plate_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception:
                plate_u8 = None
        solid_bands = None
        if resolved_base and bool(((cfg or {}).get("background") or {}).get("solid_plate", True)):
            try:
                solid_bands = _solid_plate_bands(resolved_base, canvas, plate_rgb=plate_u8)
            except Exception as exc:
                solid_bands = None
                warnings.append({"code": "solid-plate-error", "detail": str(exc)})
        if solid_bands:
            for index, band in enumerate(solid_bands):
                layers.append(Layer(
                    id=f"background-band-{index}", type="shape", shape_kind="rect",
                    name="Background",
                    box={"x": 0, "y": band["y"], "w": canvas["w"], "h": band["h"]},
                    z_index=-999_999 + index,
                    fill={"kind": "flat", "color": band["color"]},
                    constraints={"horizontal": "STRETCH",
                                 "vertical": "STRETCH" if len(solid_bands) == 1 else "TOP"},
                    # These editable rects ARE the clean inpaint plate, re-expressed as a
                    # native solid fill (Codia's flat-UI strategy). Present source="inpaint"
                    # so the clean-plate ownership gate treats them as the plate they are;
                    # keep the synthesis provenance in plate_source.
                    meta={"source": "inpaint", "plate_source": "solid-plate-band",
                          "role": "background",
                          "z": -999_999 + index, "band_color": band["color"]},
                ))
        # A radial-glow / smooth-gradient plate is emitted as an editable native gradient
        # sitting just ABOVE the raster plate. The raster stays the guaranteed fidelity
        # floor beneath; the gradient is only added when it explains the whole plate to high
        # fidelity (analytic R² + render-back error gate in reconstruct.extract_background_gradient).
        if resolved_base and not solid_bands:
            try:
                from . import reconstruct as _reconstruct  # lazy: heavy transitive deps
                bg_gradient = _reconstruct.extract_background_gradient(
                    resolved_base, cfg, rgb=plate_u8,
                )
            except Exception as exc:  # detection must never break the compile
                bg_gradient = None
                warnings.append({"code": "background-gradient-error", "detail": str(exc)})
            if bg_gradient:
                layers.append(Layer(
                    id="background-gradient", type="shape", shape_kind="rect",
                    name="Background",
                    box={"x": 0, "y": 0, "w": canvas["w"], "h": canvas["h"]},
                    z_index=-999_999, fill=bg_gradient,
                    constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
                    meta={"source": "inpaint", "plate_source": "background-gradient",
                          "role": "background",
                          "z": -999_999, "gradient": bg_gradient.get("meta")},
                ))

    kept = list(kept_in_photo or [])
    for candidate in candidates:
        if candidate.get("target") == "drop":
            if candidate.get("text"):
                kept.append(str(candidate["text"]).strip())
            continue
        for piece in _split_weight_run_siblings(candidate):
            try:
                layers.append(_compile(piece, run_dir, warnings))
            except Exception as exc:
                # One malformed entity must not hide all other editable layers. Omit only
                # the broken entity and make the partial compilation a hard structural QA
                # failure.
                warnings.append({
                    "code": "layer-compile-error", "layer_id": piece.get("id"),
                    "detail": str(exc),
                })
    layers.sort(key=lambda layer: layer.z_index)
    # Background-per-group (Codia region construction; config background.per_group).
    if base_src and bool(((cfg or {}).get("background") or {}).get("per_group", True)):
        try:
            _add_group_backgrounds(
                layers, canvas, run_dir, base_src, warnings, plate_rgb=plate_u8,
            )
        except Exception as exc:
            warnings.append({"code": "group-background-error", "detail": str(exc)})
    # Single-ownership enforcement (one owner per pixel): erase every native text layer's
    # baked duplicate from the raster carrier beneath it so nothing double-renders.
    single_ownership = {"enabled": False}
    if base_src:
        try:
            single_ownership = _audit_single_ownership(layers, run_dir, canvas, warnings, cfg)
        except Exception as exc:
            warnings.append({"code": "single-ownership-error", "detail": str(exc)})
    # Per-dimension sizing inference (Codia DimensionSpec parity). Wrapped so a geometry
    # edge case can never break the compile — worst case sizing stays empty (= fixed).
    try:
        for layer in layers:
            _infer_sizing(layer)
    except Exception as exc:
        warnings.append({"code": "sizing-inference-error", "detail": str(exc)})
    # Designer-facing uniqueness among siblings (local, O(n); no VLM).
    _dedupe_sibling_names(layers)
    total = _count_layers(layers)
    editable = _count_editable(layers)
    leaf_accounting = _leaf_accounting(layers)
    doc = DesignDoc(
        id=doc_id,
        name=name,
        canvas={"w": canvas["w"], "h": canvas["h"]},
        schema_version=SCHEMA_VERSION,
        layers=layers,
        kept_in_photo=sorted(set(x for x in kept if x)),
        meta={
            "layer_count": total,
            "root_layer_count": len(layers),
            # Legacy metric, kept only for back-compat: it counts every FRAME/GROUP as
            # "editable", so a wrapper frame around a single raster image inflates this
            # number even though nothing inside is actually editable. `native_leaf_ratio`
            # below (leaf-only, no wrapper credit) is the honest metric acceptance gates on.
            "editable_ratio": round(editable / max(1, total), 4),
            "native_leaf_ratio": leaf_accounting["native_leaf_ratio"],
            "leaf_accounting": leaf_accounting,
            "single_ownership": single_ownership,
            "warnings": warnings,
            "compiler": "scene-graph-v2",
            "coordinate_space": "local",
        },
    )
    schema_errors = validate_design(doc)
    if schema_errors:
        # Same list object backs doc.meta["warnings"], so this mutation is visible in
        # the document we're about to dump without reconstructing it.
        warnings.extend({"code": "invalid-schema", "detail": msg} for msg in schema_errors)

    dump(doc, os.path.join(run_dir, "design.json"))
    dump({"ok": not warnings, "warnings": warnings, "layer_count": total},
         os.path.join(run_dir, "design_preflight.json"))
    return doc
