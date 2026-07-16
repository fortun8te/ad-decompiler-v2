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
    "ui-label": "Label", "ui_label": "Label", "ui-text": "Label",
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


_MAX_LAYER_NAME_LEN = 40


def _with_snippet(label: str, text) -> str:
    # Keep the whole "Label / snippet" name within the ~40-char designer budget: long
    # role labels (e.g. "Disclaimer") get a shorter snippet than short ones ("Body"),
    # but never grow past the 28-char snippet cap already proven safe for short labels.
    budget = min(28, max(8, _MAX_LAYER_NAME_LEN - len(label) - 3))
    snippet = _clean_snippet(text, budget)
    if not snippet:
        return label
    # Avoid "Headline / Headline" when copy equals the role word.
    if snippet.casefold() == label.casefold():
        return label
    return f"{label} / {snippet}"


_GENERIC_GROUP_ROLES = frozenset({"", "shape", "group", "asset-group", "band", "residual"})
_GROUP_CONTENT_DECOR_LABELS = frozenset({
    "Shape", "Decoration", "Arrow", "Underline", "Strikethrough", "Dot",
})


def _group_content_name(candidate) -> Optional[str]:
    """Derive a group name from what its children actually are.

    Element-fusion sometimes wraps a mixed bag of children (product photos, a
    price, a CTA line) in a container whose own role is a low-confidence
    catch-all like "shape" or "group" (see benchmark 002: an E-series residual
    group of 3 products + 2 prices + a subheadline was literally named "Shape",
    which collided with a sibling image also named "Shape" and produced the
    designer-facing "Shape / 2"). Rank the children's role labels by frequency,
    preferring content-bearing roles (Product, Price, Headline, ...) over purely
    decorative ones (Shape, Arrow, Underline, ...), and join the top two so the
    name reads like what the group is FOR, e.g. "Product + Price".
    """
    children = candidate.get("children") or []
    if not children:
        return None
    counts: dict[str, int] = {}
    order: list[str] = []
    for child in children:
        target = child.get("target")
        role = _role_token(child)
        label = _role_label(role, target) if role else _TARGET_FALLBACK.get(target or "", None)
        if not label or label == "Layer":
            continue
        if label not in counts:
            order.append(label)
        counts[label] = counts.get(label, 0) + 1
    content = {label: n for label, n in counts.items() if label not in _GROUP_CONTENT_DECOR_LABELS}
    pool = content or counts
    if not pool:
        return None
    ranked = sorted(pool.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    top_labels = [label for label, _ in ranked[:2]]
    return top_labels[0] if len(top_labels) == 1 else " + ".join(top_labels)


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
        if role in _GENERIC_GROUP_ROLES:
            content_name = _group_content_name(candidate)
            if content_name:
                return content_name
        return label if label != "Layer" else "Group"

    if meta.get("wordmark"):
        return _with_snippet("Logo", text or candidate.get("text"))

    if meta.get("substitution") or meta.get("low_fidelity"):
        # This candidate was demoted from editable text to a raster image because OCR/
        # fitting confidence was too low to trust (see the "text-fidelity-fallback"
        # warning below). Flag it in the name so a designer opening the file knows this
        # layer is a pixel fallback, not a mistakenly-non-editable headline.
        return _with_snippet("Text (fallback)", text)

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


def _crop_staged_to_box(staged_rel, run_dir, box, layer_id, warnings):
    """Crop a canvas-sized staged asset down to its layer box.

    qwen-layered decomposition emits FULL-CANVAS RGBA layers; staging one verbatim
    under an element whose box is a fraction of the canvas makes every renderer
    (preview + Figma) squash the whole canvas into the box — 013's headline band
    rendered the entire 1080x1920 bear-fragment layer inside a 1066x269 rect as a
    dark smear. When the staged image is markedly larger than the box and the box
    fits inside it in canvas coordinates, crop to the box region.
    """
    if not staged_rel:
        return staged_rel
    try:
        from PIL import Image
        path = os.path.join(run_dir, staged_rel)
        with Image.open(path) as img:
            bw = max(1, int(round(float(box.get("w", 1) or 1))))
            bh = max(1, int(round(float(box.get("h", 1) or 1))))
            bx = int(round(float(box.get("x", 0) or 0)))
            by = int(round(float(box.get("y", 0) or 0)))
            if (img.width >= bw * 1.5 and img.height >= bh * 1.5
                    and bx + bw <= img.width and by + bh <= img.height
                    and bx >= 0 and by >= 0):
                img.crop((bx, by, bx + bw, by + bh)).save(path)
    except Exception as exc:
        warnings.append({"code": "asset-crop-failed", "layer_id": layer_id, "detail": str(exc)})
    return staged_rel


# ── Native SOLID shell primitives for flat badges / chips / pills (Codia contract) ──
# North star: "flat chrome = SOLID". A text-bearing shell (BOGO circle, 61%-OFF seal,
# a "snacks" pill, a % badge) whose upstream geometry is a flat ellipse / rect / star
# must be REBUILT as a native primitive, not shipped as a raster ``__hostbg`` image.
# The matte for these small saturated plates routinely comes back near-empty
# (alpha ~= 0), which lands the badge as an invisible ghost OR — because a FRAME paints
# its flat fill as a rectangle — as a plain coloured SQUARE that loses the real silhouette
# (101 BOGO circle, 013 61%-OFF ellipse, 016 45% starburst seal, 104/107 rect badges).
# Preferring the analytic primitive is also what kills the "hostbg alpha~=0 ghost" class.
_SHELL_SHAPE_ROLES = frozenset({
    "badge", "chip", "pill", "sticker", "tag", "button", "cta", "banner",
    "vs", "vs-badge", "vs-chip", "seal", "starburst", "star_badge", "label",
})


def _flat_fill_color(fill):
    """Hex colour of a flat fill spec, or None (gradients/images are not flat chrome)."""
    if isinstance(fill, dict) and str(fill.get("kind") or "flat") == "flat":
        color = fill.get("color")
        if isinstance(color, str) and color.strip():
            return color.strip()
    return None


def _starburst_path_for_shell(run_dir, abs_box, color):
    """Fit a regular-star polygon to the flat-colour silhouette inside ``abs_box``.

    Returns (svg_d, primitive) when the shell is genuinely scalloped/spiky (a seal),
    else None so the caller keeps its ellipse. Uses vectorize's VERIFIED analytic-star
    fitter (role="starburst") on a colour-segmented, hole-filled mask cut from the
    ORIGINAL source pixels — deterministic, no model. ``abs_box`` MUST be canvas-absolute;
    the returned path is in the shell's LOCAL box space (0..w, 0..h).
    """
    try:
        import numpy as np
        from PIL import Image
        from . import vectorize
    except Exception:
        return None
    source = None
    for name in ("normalized.png", "original.png"):
        cand = os.path.join(run_dir, name)
        if os.path.exists(cand):
            source = cand
            break
    if not source:
        return None
    try:
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16)
        x = int(round(float(abs_box.get("x", 0) or 0)))
        y = int(round(float(abs_box.get("y", 0) or 0)))
        w = int(round(float(abs_box.get("w", 1) or 1)))
        h = int(round(float(abs_box.get("h", 1) or 1)))
        if w < 24 or h < 24 or x < 0 or y < 0:
            return None
        crop = rgb[y:y + h, x:x + w]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        target = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.int16)
        mask = np.abs(crop - target).sum(axis=2) < 110
        if not (0.30 <= float(mask.mean()) <= 0.85):
            return None
        try:  # fill the interior (text/holes) so only the outer silhouette drives the fit
            from scipy import ndimage
            mask = ndimage.binary_fill_holes(mask)
        except Exception:
            pass
        prim = vectorize._fit_star_polygon(mask, 0.90, max_points=36)
        if not prim or int(prim.get("points") or 0) < 5:
            return None
        # A near-perfect disc (valley within a few % of the tip) is an ellipse, not a
        # star: keep the star only when the scallops are real (>3% radial modulation).
        ro = float(prim.get("r_outer") or 0.0)
        ri = float(prim.get("r_inner") or 0.0)
        if ro <= 0 or ri / ro > 0.985:
            return None
        return vectorize._star_d(
            prim["cx"], prim["cy"], prim["r_outer"], prim["r_inner"],
            prim["points"], prim.get("rotation", 0.0)), prim
    except Exception:
        return None


def _native_shell_shape(candidate, box, z_index):
    """Native SOLID shape Layer for a flat badge/chip/pill/seal shell, or None.

    Emits an ellipse or a (rounded) rect from the candidate's own upstream geometry +
    flat fill, so the shell is native+editable instead of a blank raster hostbg. Returns
    None when the shell is not flat chrome (no flat fill, or an irregular/photographic
    shape) — the caller then keeps the raster hostbg path unchanged.

    A scalloped seal is upgraded from the ellipse to a real star polygon LATER, by
    ``_upgrade_starburst_shells``: that needs to sample the source, and ``box`` here is
    PARENT-RELATIVE (016's badge is rel {769,344} but abs {776,542}), so sampling from
    this frame would read the wrong pixels.
    """
    meta = candidate.get("meta") or {}
    role = str(meta.get("role") or "").strip().lower().replace("_", "-")
    is_shell = bool(meta.get("plate_shell") or meta.get("text_bearing_shell")
                    or role in _SHELL_SHAPE_ROLES)
    if not is_shell:
        return None
    shape_kind = str(candidate.get("shape_kind") or "").strip().lower()
    if shape_kind not in ("ellipse", "rect"):
        return None
    fill = candidate.get("fill") or _surface_fill(candidate)
    color = _flat_fill_color(fill)
    if not color or len(color) != 7 or not color.startswith("#"):
        return None
    w = float(box.get("w", 1) or 1)
    h = float(box.get("h", 1) or 1)
    local_box = {"x": 0.0, "y": 0.0, "w": w, "h": h}
    fill_spec = dict(fill) if isinstance(fill, dict) else {"kind": "flat", "color": color}
    stroke = candidate.get("stroke")
    stroke_spec = dict(stroke) if isinstance(stroke, dict) else None
    common_meta = {"role": role or "badge", "z": z_index,
                   "source_id": f"{candidate.get('id')}__shell",
                   "source": "native-shell-primitive"}
    shell_id = f"{candidate.get('id')}__shell"
    label = _name(candidate)
    aspect = w / max(1.0, h)
    # Near-circular flat shell → native ellipse (101 BOGO, 013 61%-OFF). May be upgraded
    # to a starburst star downstream once absolute coordinates are known.
    if shape_kind == "ellipse":
        return Layer(
            id=shell_id, type="shape", shape_kind="ellipse",
            name=label, box=local_box, z_index=z_index, fill=fill_spec,
            stroke=stroke_spec,
            constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
            meta={**common_meta, "rebuilt_from": "flat-ellipse-shell"},
        )
    # 3) Flat rect / pill → native (rounded) rect. A wide short rect with no explicit
    #    radius reads as a stadium pill; snap it so it is not a hard-cornered box
    #    (013 "snacks" chip).
    radius = candidate.get("radius")
    if radius in (None, 0, 0.0) and aspect >= 2.0:
        radius = round(min(w, h) / 2.0, 2)
    return Layer(
        id=shell_id, type="shape", shape_kind="rect",
        name=label, box=local_box, z_index=z_index, fill=fill_spec,
        stroke=stroke_spec, radius=radius,
        constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
        meta={**common_meta, "rebuilt_from": "flat-rect-shell"},
    )


def _apply_glass_fill(fill, fill_opacity):
    """Fold a glass fill-opacity into the fill dict (fill-only alpha, not layer opacity)."""
    if fill_opacity is None or not isinstance(fill, dict):
        return fill
    fill = dict(fill)
    fill["opacity"] = float(fill_opacity)
    return fill


# ── Mixed-weight lines → sibling TEXT nodes (Codia §2a) ───────────────────────────
# Codia never uses styled ranges: a weight change always produces a NEW sibling node
# ("05:00 PM · 12-05-2026 ·"(300) / "121K"(700) / "weergaven"(300)). Splitting is
# trivially safe in every Figma plugin and lets each run carry its own sampled color.
# Only CONTRAST-VERIFIED weight runs split (upstream _enrich_word_styles gates on a
# >=180 weight delta measured from word pixels); everything else stays one node.
_WEIGHT_SPLIT_MIN_DELTA = 250
# Word-bomb guard: a genuine display emphasis splits a line into a HANDFUL of siblings
# ("Save " / "50%" / " Now" = 3). When per-word weight estimates wobble across a body-copy
# line, every emphasized word plus every base-weight gap between them becomes its own
# sibling and the paragraph detonates into per-word nodes (067: "our Sale with 40% OFF
# soon. Experience" -> 6 fragments, wrecking placement). Above this many non-empty
# segments the split is treated as measurement noise: the line stays one node with its
# text_runs intact, so mixed weight still renders inline without shredding the copy.
_WEIGHT_SPLIT_MAX_SEGMENTS = 3
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

    # Word-bomb guard: bail out (keep the single node + inline text_runs) when the split
    # would shred the line into more siblings than a genuine emphasis ever produces. This
    # is what stops body copy and offer/ribbon lines with wobbly per-word weight estimates
    # from detonating into per-word nodes (067).
    if sum(1 for s, e, _ in segments if text[s:e].strip()) > _WEIGHT_SPLIT_MAX_SEGMENTS:
        return [candidate]

    # MEASURED word geometry beats the proportional advance model: OCR word boxes see
    # real gaps (benchmark 002: an arrow sits between "€63" and "€49", so equal-advance
    # placement squeezed both prices toward the center). Fractions were recorded against
    # the source line box and survive rebasing, so they apply to the current local box.
    measured = _measured_segment_fractions(
        (candidate.get("meta") or {}).get("word_geometry") or [], segments, text)

    out = []
    for index, (start, end, run_style) in enumerate(segments):
        segment_text = text[start:end].strip()
        if not segment_text:
            continue
        if measured is not None:
            frac_x, frac_w = measured[index]
        else:
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
                                "segments": len(segments),
                                "placement": "word-geometry" if measured is not None
                                else "advance-proportional"}
        piece["meta"] = meta
        out.append(piece)
    return out if len(out) > 1 else [candidate]


def _measured_segment_fractions(word_geometry: list, segments: list, text: str):
    """Map each split segment onto measured OCR word spans; None when unmatchable.

    Words are matched sequentially by token so an OCR-only glyph (the recovered price
    separator "J" that a verified arrow replaced) is skipped without disturbing the
    match. Fail-closed: every non-empty segment must resolve to a contiguous word run,
    otherwise the caller keeps the proportional-advance fallback for ALL segments.
    """
    if not word_geometry:
        return None
    try:
        words = [(str(w["text"]).strip(), float(w["fx"]), float(w["fw"]))
                 for w in word_geometry]
    except (KeyError, TypeError, ValueError):
        return None
    if any((not t) or fw <= 0 for t, _fx, fw in words):
        return None
    fractions = []
    cursor = 0
    for start, end, _style in segments:
        tokens = text[start:end].split()
        if not tokens:
            fractions.append((0.0, 0.0))  # dropped by the caller (empty segment)
            continue
        # Skip source-only words (e.g. the removed separator glyph) before the segment.
        first = cursor
        while first < len(words) and words[first][0] != tokens[0]:
            first += 1
        if first + len(tokens) > len(words):
            return None
        if any(words[first + i][0] != token for i, token in enumerate(tokens)):
            return None
        last = first + len(tokens) - 1
        fx = words[first][1]
        fw = (words[last][1] + words[last][2]) - fx
        if fw <= 0:
            return None
        fractions.append((fx, fw))
        cursor = last + 1
    return fractions


def _flatten_with_offsets(children: list, offx: float = 0.0, offy: float = 0.0):
    """Yield (layer, abs_offset_x, abs_offset_y) for every node in the tree.

    Child boxes are stored RELATIVE to their parent group's box origin, so a node's
    absolute canvas position is the running sum of its ancestors' box origins. The
    offset yielded for a node is the frame its OWN box lives in (i.e. the parent's
    accumulated origin); add the node's box.x/y to get the node's absolute position.
    """
    for layer in children:
        yield layer, offx, offy
        kids = getattr(layer, "children", None)
        if kids:
            box = getattr(layer, "box", None) or {}
            try:
                cx = offx + float(box.get("x", 0) or 0)
                cy = offy + float(box.get("y", 0) or 0)
            except (TypeError, ValueError):
                cx, cy = offx, offy
            yield from _flatten_with_offsets(kids, cx, cy)


def _reanchor_decorations(children: list) -> int:
    """Re-project native text decorations onto their owner's EMITTED geometry.

    Contract (audit 002 finding): a decoration attached to a text node is positioned
    RELATIVE to that node's final geometry, never at absolute source coordinates. The
    merge stage records endpoint fractions against the source word/node box
    (meta.anchor); here they are re-applied to the owner's compiled ink box
    (meta.prefit_ink_box — the box preview and the Figma plugin fit glyphs into).

    A decoration and its owner do NOT necessarily share a parent: layout routinely
    lifts price strikes/underlines to the root while the owning word nodes sink into a
    text-stack group (benchmark 002). So the owner is resolved across the WHOLE tree,
    and every box is projected into absolute canvas space (accumulating parent-group
    origins) before the endpoint fractions are applied — otherwise the decoration
    lands a group-offset away from the glyphs it should cross. Returns count moved.
    """
    nodes = list(_flatten_with_offsets(children))
    texts = [(layer, ox, oy) for layer, ox, oy in nodes
             if getattr(layer, "type", None) == "text"]
    moved = 0
    for layer, dox, doy in nodes:
        meta = layer.meta or {}
        anchor = meta.get("anchor")
        if not meta.get("native_decoration") or not isinstance(anchor, dict):
            continue
        owner_id = str(anchor.get("owner_id") or meta.get("decoration_owner_id") or "")
        if not owner_id:
            continue
        candidates = [(t, ox, oy) for t, ox, oy in texts
                      if t.id == owner_id or str(t.id).startswith(f"{owner_id}__w")]
        if not candidates:
            continue
        word_text = anchor.get("word_text")
        owner = None
        if word_text:
            owner = next((c for c in candidates
                          if str(c[0].text or "").strip() == str(word_text).strip()), None)
        if owner is None and len(candidates) == 1:
            owner = candidates[0]
        if owner is None:
            continue
        owner_layer, oox, ooy = owner
        # prefit_ink_box is the tight glyph ink, but recorded in the PRE-move frame
        # (== visible_box origin). Layout translates the emitted node from visible_box
        # to box without resizing (benchmark 002: €63 moved +25,+45). The rendered
        # geometry lives at `box`, so shift the ink by (box - visible_box) to land on
        # the FINAL glyphs. Without this, the decoration tracks the stale source ink.
        ometa = owner_layer.meta or {}
        fbox = dict(owner_layer.box or {})
        vbox = dict(owner_layer.visible_box or {})
        prefit = dict(ometa.get("prefit_ink_box") or {})
        if prefit and fbox and vbox:
            try:
                dx = float(fbox.get("x", 0) or 0) - float(vbox.get("x", 0) or 0)
                dy = float(fbox.get("y", 0) or 0) - float(vbox.get("y", 0) or 0)
            except (TypeError, ValueError):
                dx = dy = 0.0
            ink = {"x": float(prefit.get("x", 0) or 0) + dx,
                   "y": float(prefit.get("y", 0) or 0) + dy,
                   "w": prefit.get("w"), "h": prefit.get("h")}
        else:
            # No reliable pre-move frame: anchor to the final emitted box directly.
            ink = fbox or vbox or prefit
        try:
            # Owner ink -> absolute canvas coordinates (add owner's ancestor offset).
            ax, ay = oox + float(ink["x"]), ooy + float(ink["y"])
            aw, ah = float(ink["w"]), float(ink["h"])
            fx0, fy0 = float(anchor["fx0"]), float(anchor["fy0"])
            fx1, fy1 = float(anchor["fx1"]), float(anchor["fy1"])
            thickness = max(1.0, float((meta.get("line") or {}).get("thickness", 2.0)))
        except (KeyError, TypeError, ValueError):
            continue
        if aw <= 0 or ah <= 0:
            continue
        # Absolute endpoints of the decoration across the owner's final ink box.
        x0, y0 = ax + fx0 * aw, ay + fy0 * ah
        x1, y1 = ax + fx1 * aw, ay + fy1 * ah
        old_line = dict(meta.get("line") or {})
        pad = max(1.0, thickness)
        bx, by = min(x0, x1) - pad, min(y0, y1) - pad
        bw = max(1.0, abs(x1 - x0) + pad * 2)
        bh = max(1.0, abs(y1 - y0) + pad * 2)
        colour = str((layer.stroke or {}).get("color") or "#e1491b")
        # Store the box in the decoration's OWN frame (absolute minus its parent offset).
        layer.box = {"x": round(bx - dox, 2), "y": round(by - doy, 2),
                     "w": round(bw, 2), "h": round(bh, 2)}
        layer.svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{bw:.2f}" height="{bh:.2f}" '
            f'viewBox="0 0 {bw:.2f} {bh:.2f}"><path d="M {x0 - bx:.2f} {y0 - by:.2f} '
            f'L {x1 - bx:.2f} {y1 - by:.2f}" fill="none" stroke="{colour}" '
            f'stroke-width="{thickness:.2f}" stroke-linecap="round"/></svg>'
        )
        meta["line"] = {"x0": round(x0, 2), "y0": round(y0, 2),
                        "x1": round(x1, 2), "y1": round(y1, 2), "thickness": thickness}
        meta["reanchored_to"] = owner_layer.id
        if old_line:
            meta["source_line"] = old_line
        moved += 1
    return moved


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
        # Editable marketing copy must paint above chrome/product hosts. Group z
        # scopes child z, so a text child at 8 cannot escape a sibling chrome group
        # at 50 unless the text stack itself is lifted as well below.
        if target == "text":
            return max(band_z, 60)
        return band_z
    if target == "text":
        return 60
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
    # Glass/translucent chips: glass_detect.detect_glass may attach fill_opacity (0..1)
    # and background_blur_radius (FIGMA-space px) onto the candidate's meta. Fold into
    # the fill's own opacity (NOT layer opacity) + an appended background-blur effect.
    # Absence of these == solid fallback (no separate code path).
    glass_fill_opacity = meta.get("fill_opacity", candidate.get("fill_opacity"))
    glass_blur_radius = meta.get("background_blur_radius", candidate.get("background_blur_radius"))
    if glass_blur_radius is not None:
        source_effects = list(source_effects or [])
        source_effects.append({"type": "background-blur",
                               "radius": float(glass_blur_radius), "visible": True})
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
    if target == "text" and not text_placeholder_z and z_raw is not None:
        try:
            text_placeholder_z = float(z_raw) <= 15.0
        except (TypeError, ValueError):
            pass
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
            # A child the confidence fallback retired to a plate-passthrough (target=="drop")
            # must be dropped here too — the top-level loop already skips drops, but nested
            # drops used to fall through _compile's unknown-target tail and re-emit as a
            # blank image leaf (088 c_E011: an "unexplained-raster" hard-fail on a badge the
            # clean plate already holds). Honour the drop; keep any baked text for QA recall.
            if isinstance(child, dict) and child.get("target") == "drop":
                continue
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
        #
        # BUT flat chrome shells (badge / chip / pill / seal) are rebuilt as NATIVE solid
        # primitives, never a raster hostbg: their matte comes back near-empty and the
        # FRAME paints its flat fill as a coloured SQUARE, so the real circle/seal is lost
        # (101 BOGO, 013 61%-OFF + "snacks" pill, 016 45% starburst, 104/107 rect badges).
        raster_src = candidate.get("src")
        bg_z_shell = min((child.z_index for child in children), default=z_index) - 1.0
        native_shell = _native_shell_shape(candidate, box, bg_z_shell)
        if native_shell is not None:
            children.append(native_shell)
        elif raster_src:
            staged = _crop_staged_to_box(
                _stage_asset(raster_src, f"{layer_id}__hostbg", run_dir, warnings),
                run_dir, box, layer_id, warnings)
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
        role = str((candidate.get("meta") or {}).get("role") or "").lower()
        if children and role in {
            "text-stack", "text-row", "stat-stack", "stat-column", "copy-stack",
        }:
            z_index = max(float(z_index), max(float(child.z_index) for child in children))
            common["z_index"] = z_index
            common["meta"]["z"] = z_index
            # Text boxes are deliberately generous to survive font substitution, but
            # layout groups were sized from the original tight ink boxes. A FRAME-sized
            # group therefore clips its own enlarged children (002 lost half of BUNDEL).
            # Expand the transparent text container to the compiled child union while
            # shifting children when the union extends left/up, preserving absolute
            # positions and keeping the group non-painting/editable.
            group_box = common["box"]
            min_x = min([0.0] + [float((child.box or {}).get("x", 0) or 0) for child in children])
            min_y = min([0.0] + [float((child.box or {}).get("y", 0) or 0) for child in children])
            max_x = max(
                [float(group_box.get("w", 0) or 0)]
                + [float((child.box or {}).get("x", 0) or 0)
                   + float((child.box or {}).get("w", 0) or 0) for child in children]
            )
            max_y = max(
                [float(group_box.get("h", 0) or 0)]
                + [float((child.box or {}).get("y", 0) or 0)
                   + float((child.box or {}).get("h", 0) or 0) for child in children]
            )
            if min_x < 0 or min_y < 0 or max_x > float(group_box.get("w", 0) or 0) or max_y > float(group_box.get("h", 0) or 0):
                group_box["x"] = float(group_box.get("x", 0) or 0) + min_x
                group_box["y"] = float(group_box.get("y", 0) or 0) + min_y
                group_box["w"] = max_x - min_x
                group_box["h"] = max_y - min_y
                for child in children:
                    child.box["x"] = float((child.box or {}).get("x", 0) or 0) - min_x
                    child.box["y"] = float((child.box or {}).get("y", 0) or 0) - min_y
                common["meta"]["expanded_to_child_union"] = True
        # When a native shell primitive now carries the shell paint, the FRAME must NOT
        # also paint its flat fill/stroke/radius — a frame paints its fill as a rectangle
        # and would re-introduce the coloured square behind the ellipse/star.
        if native_shell is not None:
            group_fill = group_stroke = group_radius = None
        else:
            group_fill = _apply_glass_fill(candidate.get("fill"), glass_fill_opacity)
            group_stroke = candidate.get("stroke")
            group_radius = candidate.get("radius") or source_style.get("radius")
        return Layer(
            type="group",
            children=children,
            fill=group_fill,
            stroke=group_stroke,
            radius=group_radius,
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
        shape_src = None
        if candidate.get("src"):
            shape_src = _crop_staged_to_box(
                _stage_asset(candidate.get("src"), layer_id, run_dir, warnings),
                run_dir, box, layer_id, warnings)
        shape_fill = _apply_glass_fill(candidate.get("fill"), glass_fill_opacity)
        if shape_src and shape_fill is not None:
            # A raster src defines the appearance; painting a detector-estimated flat
            # fill UNDER it double-renders (013: dark #0a723d fill showed through the
            # asset's transparent majority as a solid smudge band). Keep the estimate
            # as provenance only.
            common["meta"]["detected_fill"] = shape_fill
            shape_fill = None
        return Layer(
            type="shape",
            shape_kind=candidate.get("shape_kind") or "rect",
            path=candidate.get("path"),
            svg=candidate.get("svg"),
            src=shape_src,
            fill=shape_fill,
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
        role = str((layer.meta or {}).get("role") or "").lower()
        if role in {"text-stack", "text-row", "copy-stack", "stat-stack", "stat-column"}:
            return
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


# ── Empty-asset materialization ban (no silent blank groups / ghost rasters) ────────
# A group with zero children, or an image whose staged asset is blank (alpha ~= 0 or a
# trivial byte-size), ships NOTHING while SSIM stays high because the plate still holds
# the burned-in subject (104: Product/Photo-After groups are 8KB blank PNGs, the phones
# live only in Background; 107: pack -> black blob; 021: 3 empty groups). Photo/product
# material must be REAL: such a layer either materializes pixel-exact source pixels cropped
# from its own box (like a confidence slice) or is dropped with a recorded reason.
# Deliberately photo/product-shaped only. "logo"/"icon" are EXCLUDED: a thin wordmark or
# line icon is legitimately sparse, and stamping an opaque box crop over one would paste
# the background back in — a worse bug than the ghost it fixes.
_PIXEL_REQUIRED_ROLES = frozenset({
    "photo", "image", "product", "product-cluster", "person", "people", "face",
    "hand", "avatar", "profile", "profile-photo", "thumbnail", "illustration",
    "package", "pack", "hero", "asset-group", "picture", "screenshot",
})
_BLANK_ALPHA_COV = 0.06     # < 6 % opaque pixels == a ghost matte
_BLANK_BYTES = 1200         # a real photo crop is never this small


def _norm_role(meta):
    return str((meta or {}).get("role") or "").strip().lower().replace("_", "-")


def _asset_alpha_coverage(run_dir, src):
    """Fraction of opaque pixels in a staged asset, plus its byte size (or (None, 0))."""
    if not src or not isinstance(src, str):
        return None, 0
    path = os.path.join(run_dir, src)
    if not os.path.exists(path):
        return None, 0
    try:
        import numpy as np
        from PIL import Image
        size = os.path.getsize(path)
        arr = np.asarray(Image.open(path).convert("RGBA"))
        if arr.size == 0:
            return 0.0, size
        return float((arr[..., 3] > 8).mean()), size
    except Exception:
        return None, 0


def _materialize_source_crop(run_dir, abs_box, layer_id, warnings, min_std=6.0):
    """Crop opaque, pixel-exact pixels from the ORIGINAL source at ``abs_box``.

    Returns ``(status, rel_src)`` where status is one of:
      "ok"          — real source pixels were written to ``rel_src``
      "no-subject"  — the source region is a near-uniform patch (nothing to rebuild),
                      so the caller may DROP the layer with a recorded reason
      "no-source"   — the source could not be read at all. The caller must KEEP the
                      layer untouched: deleting content we simply failed to VERIFY is
                      content erasure (F1), a worse bug than the ghost it would fix.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return "no-source", None
    source = None
    for name in ("normalized.png", "original.png"):
        cand = os.path.join(run_dir, name)
        if os.path.exists(cand):
            source = cand
            break
    if not source:
        return "no-source", None
    try:
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        H, W = rgb.shape[:2]
        x = max(0, int(round(float(abs_box.get("x", 0) or 0))))
        y = max(0, int(round(float(abs_box.get("y", 0) or 0))))
        w = int(round(float(abs_box.get("w", 1) or 1)))
        h = int(round(float(abs_box.get("h", 1) or 1)))
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        if x1 - x < 2 or y1 - y < 2:
            return "no-source", None
        crop = rgb[y:y1, x:x1]
        if float(crop.reshape(-1, 3).std(axis=0).mean()) < min_std:
            return "no-subject", None
        tile = np.dstack([crop, np.full(crop.shape[:2], 255, dtype=np.uint8)])
        assets = os.path.join(run_dir, "assets")
        os.makedirs(assets, exist_ok=True)
        rel = os.path.join("assets", f"{layer_id}_materialized.png")
        Image.fromarray(tile, "RGBA").save(os.path.join(run_dir, rel))
        return "ok", rel
    except Exception as exc:
        warnings.append({"code": "materialize-failed", "layer_id": layer_id, "detail": str(exc)})
        return "no-source", None


def _upgrade_starburst_shells(layers, run_dir, warnings):
    """Promote near-circular native shells that are really SCALLOPED SEALS to star paths.

    Runs post-compile because it samples the ORIGINAL source and only here are absolute
    coordinates known (a compiled group box is parent-relative). 016's "Get up to 45% Off"
    seal is a 26-scallop starburst that shipped as a plain teal square; an ellipse is
    closer but still wrong. vectorize's analytic-star fitter reconstructs it natively
    (IoU ~0.97). A genuine disc/pill never fits, so it keeps its ellipse.
    """
    upgraded = []

    def visit(nodes, offset):
        for node in nodes or []:
            if not isinstance(node, Layer):
                continue
            box = node.box or {}
            abs_box = {
                "x": float(offset[0]) + float(box.get("x", 0) or 0),
                "y": float(offset[1]) + float(box.get("y", 0) or 0),
                "w": float(box.get("w", 1) or 1),
                "h": float(box.get("h", 1) or 1),
            }
            if node.children:
                visit(node.children, (abs_box["x"], abs_box["y"]))
            if (node.meta or {}).get("rebuilt_from") != "flat-ellipse-shell":
                continue
            aspect = abs_box["w"] / max(1.0, abs_box["h"])
            if not (0.75 <= aspect <= 1.34):
                continue
            color = _flat_fill_color(node.fill)
            if not color:
                continue
            star = _starburst_path_for_shell(run_dir, abs_box, color)
            if star is None:
                continue
            star_d, prim = star
            node.shape_kind = "path"
            node.path = star_d
            node.meta = {**(node.meta or {}), "rebuilt_from": "starburst-seal",
                         "star_primitive": prim}
            upgraded.append({"id": node.id, "points": prim["points"], "iou": prim["iou"]})

    visit(layers, (0.0, 0.0))
    return upgraded


def _enforce_asset_materialization(layers, run_dir, canvas, warnings):
    """Ban empty asset groups and blank/ghost photo-product rasters (recorded, not muted).

    Walks the compiled tree with absolute offsets. Any empty group or blank image in a
    pixel-required role is rebuilt from source pixels cropped to its box, or dropped with
    a recorded reason. Returns an audit report for design_preflight.json.
    """
    report = {"materialized": [], "dropped": [], "checked": 0}

    def visit(nodes, offset):
        kept = []
        for node in nodes or []:
            if not isinstance(node, Layer):
                kept.append(node)
                continue
            box = node.box or {}
            abs_box = {
                "x": float(offset[0]) + float(box.get("x", 0) or 0),
                "y": float(offset[1]) + float(box.get("y", 0) or 0),
                "w": float(box.get("w", 1) or 1),
                "h": float(box.get("h", 1) or 1),
            }
            role = _norm_role(node.meta)
            child_offset = (abs_box["x"], abs_box["y"])
            if node.children:
                node.children = visit(node.children, child_offset)
            report["checked"] += 1
            # 1) Empty group in a pixel-required role → materialize a source-pixel child.
            if node.type == "group" and not node.children:
                if role in _PIXEL_REQUIRED_ROLES or str(node.id or "").startswith("asset-group"):
                    status, rel = _materialize_source_crop(
                        run_dir, abs_box, str(node.id), warnings)
                    if status == "ok":
                        node.children = [Layer(
                            id=f"{node.id}__materialized", type="image", name=node.name,
                            box={"x": 0.0, "y": 0.0, "w": abs_box["w"], "h": abs_box["h"]},
                            z_index=float(node.z_index),
                            src=rel,
                            constraints={"horizontal": "STRETCH", "vertical": "STRETCH"},
                            meta={"role": role or "image", "source": "materialized-source-crop",
                                  "materialized_reason": "empty-asset-group",
                                  "source_id": f"{node.id}__materialized"},
                        )]
                        report["materialized"].append(
                            {"id": node.id, "reason": "empty-asset-group", "box": abs_box})
                    elif status == "no-subject":
                        report["dropped"].append(
                            {"id": node.id, "reason": "empty-asset-group-no-subject"})
                        continue  # drop the empty group entirely
                    else:
                        # Unverifiable (no readable source): keep it and say so.
                        warnings.append({"code": "empty-asset-group-unverified",
                                         "layer_id": node.id})
                else:
                    # A non-photo empty group (pure layout wrapper) is legitimate; keep it.
                    kept.append(node)
                    continue
            # 2) Blank / ghost image in a pixel-required role → opaque source slice or drop.
            if node.type == "image" and role in _PIXEL_REQUIRED_ROLES:
                # Confidence slices are already pixel-exact source cutouts — never re-gate.
                if fallback_kind(node.meta or {}) != "raster-slice":
                    cov, size = _asset_alpha_coverage(run_dir, node.src)
                    blank = (node.src is None) or (cov is not None and cov < _BLANK_ALPHA_COV) \
                        or (size and size < _BLANK_BYTES and (cov or 0) < 0.25)
                    if blank:
                        status, rel = _materialize_source_crop(
                            run_dir, abs_box, str(node.id), warnings)
                        if status == "ok":
                            node.src = rel
                            node.mask = None
                            node.meta = {**(node.meta or {}),
                                         "source": "materialized-source-crop",
                                         "materialized_reason": "blank-ghost-raster",
                                         "prev_alpha_cov": round(cov, 4) if cov is not None else None}
                            report["materialized"].append(
                                {"id": node.id, "reason": "blank-ghost-raster",
                                 "alpha_cov": cov, "box": abs_box})
                        elif status == "no-subject":
                            report["dropped"].append(
                                {"id": node.id, "reason": "blank-ghost-raster-no-subject",
                                 "alpha_cov": cov})
                            continue  # drop the ghost
                        else:
                            warnings.append({"code": "blank-raster-unverified",
                                             "layer_id": node.id})
            kept.append(node)
        return kept

    return visit(layers, (0.0, 0.0)), report


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
    # Decoration-follows-text contract: strikes/underlines ride their owner text node's
    # FINAL geometry, never absolute source coordinates (wrapped: geometry repair must
    # never break the compile — worst case a decoration stays at source coordinates).
    try:
        _reanchor_decorations(layers)
    except Exception as exc:
        warnings.append({"code": "decoration-reanchor-error", "detail": str(exc)})
    # A near-circular native shell that is really a scalloped SEAL becomes a star path
    # (016). Needs absolute coordinates, hence post-compile. Never breaks the compile:
    # worst case the shell keeps its ellipse.
    starbursts = []
    try:
        starbursts = _upgrade_starburst_shells(layers, run_dir, warnings)
    except Exception as exc:
        warnings.append({"code": "starburst-upgrade-error", "detail": str(exc)})
    # Ban empty asset groups and blank/ghost photo-product rasters — materialize real
    # source pixels or drop with a recorded reason (never ship a silent 8KB blank PNG).
    materialization = {"materialized": [], "dropped": [], "checked": 0}
    try:
        layers, materialization = _enforce_asset_materialization(
            layers, run_dir, canvas, warnings)
    except Exception as exc:
        warnings.append({"code": "asset-materialization-error", "detail": str(exc)})
    materialization["starburst_seals"] = starbursts
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
            "asset_materialization": materialization,
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
    dump({"ok": not warnings, "warnings": warnings, "layer_count": total,
          "asset_materialization": materialization},
         os.path.join(run_dir, "design_preflight.json"))
    return doc
