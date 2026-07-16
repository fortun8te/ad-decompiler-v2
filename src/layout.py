"""Infer a conservative native frame tree from canonical visual entities.

The goal is not to force every artistic composition into Auto Layout.  We create frames only
when a real container shape owns contained children, and enable Auto Layout only when the
row/column evidence is strong.  Everything else stays accurately absolutely positioned with
Figma constraints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from statistics import median
from typing import Optional

from . import vlm_layout_group
from .diagram_editability import members_support_native_chart


def _area(box):
    return max(0.0, box.get("w", 0)) * max(0.0, box.get("h", 0))


def _inside(inner, outer):
    ix = max(0.0, min(inner.get("x", 0) + inner.get("w", 0), outer.get("x", 0) + outer.get("w", 0))
             - max(inner.get("x", 0), outer.get("x", 0)))
    iy = max(0.0, min(inner.get("y", 0) + inner.get("h", 0), outer.get("y", 0) + outer.get("h", 0))
             - max(inner.get("y", 0), outer.get("y", 0)))
    return (ix * iy) / max(1.0, _area(inner))


def _overlap(a, b):
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1.0, min(_area(a), _area(b)))


def _constraints(child, parent):
    left = child["x"] - parent["x"]
    top = child["y"] - parent["y"]
    right = parent["x"] + parent["w"] - (child["x"] + child["w"])
    bottom = parent["y"] + parent["h"] - (child["y"] + child["h"])
    tol_x = max(3.0, parent["w"] * 0.035)
    tol_y = max(3.0, parent["h"] * 0.035)
    if abs(left - right) <= tol_x:
        horizontal = "CENTER"
    elif left <= tol_x and right <= tol_x:
        horizontal = "STRETCH"
    elif right < left:
        horizontal = "RIGHT"
    else:
        horizontal = "LEFT"
    if abs(top - bottom) <= tol_y:
        vertical = "CENTER"
    elif top <= tol_y and bottom <= tol_y:
        vertical = "STRETCH"
    elif bottom < top:
        vertical = "BOTTOM"
    else:
        vertical = "TOP"
    return {"horizontal": horizontal, "vertical": vertical}


def _consistent(values, max_cv=0.28):
    values = [float(v) for v in values if v >= 0]
    if len(values) <= 1:
        return True
    mean = sum(values) / len(values)
    if mean <= 1:
        return max(values, default=0) <= 2
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance) / mean <= max_cv


def _paint_box(node):
    """Prefer ink/painted bounds over loose OCR boxes for padding + centering."""
    return (node.get("visible_box") or node.get("ink_box") or node.get("box") or {})


def _is_centered(child_box, parent_box, tol_x=None, tol_y=None):
    pb = parent_box or {}
    cb = child_box or {}
    tol_x = tol_x if tol_x is not None else max(3.0, pb.get("w", 0) * 0.04)
    tol_y = tol_y if tol_y is not None else max(3.0, pb.get("h", 0) * 0.08)
    cx = cb.get("x", 0) + cb.get("w", 0) / 2
    cy = cb.get("y", 0) + cb.get("h", 0) / 2
    pcx = pb.get("x", 0) + pb.get("w", 0) / 2
    pcy = pb.get("y", 0) + pb.get("h", 0) / 2
    return abs(cx - pcx) <= tol_x and abs(cy - pcy) <= tol_y


def _is_padded_child(child_box, parent_box):
    """Whether a single child genuinely fills a plate with real inset padding.

    Guards the padded-card HUG path: the child must sit inside the plate, be a
    substantial fraction of it (not a speck on a backdrop), and not be full-bleed
    (which would leave no padding to hug).  Because it is fully inside, the
    four-side measured padding reconstructs the plate's box exactly.
    """
    cb, pb = child_box or {}, parent_box or {}
    if _inside(cb, pb) < 0.95:
        return False
    ca, pa = _area(cb), _area(pb)
    if pa <= 0 or ca < pa * 0.25 or ca > pa * 0.98:
        return False
    return (cb.get("w", 0) >= pb.get("w", 0) * 0.5) or (cb.get("h", 0) >= pb.get("h", 0) * 0.5)


_BUTTON_TEXT_ROLES = {"cta", "button", "offer", "price"}
_BUTTON_CONTAINER_ROLES = {
    "button", "badge", "chip", "card", "banner", "seal",
    "starburst", "price_burst", "sale_burst", "burst",
}
_MESSAGE_BUBBLE_ROLES = {
    "message", "bubble", "message_bubble", "chat", "chat_bubble",
    "sms", "imessage", "comment",
}
_MESSAGE_BUBBLE_SHELL_ROLES = _MESSAGE_BUBBLE_ROLES | {
    None, "", "card", "container", "plate", "panel", "shape",
}
_AVATAR_ROLES = {
    "avatar", "profile", "profile_picture", "profile_photo", "pfp", "headshot",
    "user_photo",
}
_HEADER_IDENTITY_ROLES = {
    "handle", "username", "name", "display_name", "displayname", "author",
    "timestamp", "meta", "label", "subhead", "body", "verified",
}

# Brand marks stay independent so a wordmark never fuses into a paragraph flow.
# Every other text role is an ordinary copy line and may join a stack/row when the
# geometry gate below agrees.
_NON_FLOW_TEXT_ROLES = {"logo", "wordmark", "watermark", "brand"}

_ENGAGEMENT_ICON_ROLES = {
    "engagement", "like", "reply", "repost", "share", "comment", "views",
    "bookmark", "save", "heart", "retweet", "favourite", "favorite",
}
_VS_TEXT_RE = re.compile(r"^\s*(vs\.?|versus)\s*$", re.I)
_CHECKLIST_ICON_ROLES = frozenset({
    "verified", "checkmark", "check", "check-mark", "check_mark", "tick",
    "x", "close", "cross", "cancel", "reject", "deny",
})


def _looks_like_checklist_row(children) -> bool:
    """Icon chip (✓/X) + label text — Wavy before/after benefit pills."""
    if not children or len(children) < 2:
        return False
    has_text = any(n.get("target") == "text" for n in children)
    has_check = False
    for n in children:
        role = str((n.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        if n.get("target") in {"icon", "image", "shape"} and role in _CHECKLIST_ICON_ROLES:
            has_check = True
            break
        if (n.get("meta") or {}).get("icon_chip") and role in _CHECKLIST_ICON_ROLES | {""}:
            has_check = True
            break
    return has_text and has_check


def _scene_grouping(cfg) -> dict:
    return ((cfg or {}).get("layout") or {}).get("scene_grouping") or {}


def _is_button_pattern(container, children):
    """Shape/card shell with a single centered CTA-style label from text_analysis."""
    if len(children) != 1:
        return False
    child = children[0]
    if child.get("target") != "text":
        return False
    child_role = (child.get("meta") or {}).get("role", "text")
    host_meta = container.get("meta") or {}
    host_role = host_meta.get("role")
    # Wide brushstroke banners pair as caption/backplate groups, not CTA buttons.
    if host_role in {"banner", "ribbon", "brushstroke"} or (
            host_meta.get("text_bearing_shell") and host_role == "banner"):
        return False
    if child_role not in _BUTTON_TEXT_ROLES and host_role not in _BUTTON_CONTAINER_ROLES:
        return False
    if host_role == "card" and child_role not in _BUTTON_TEXT_ROLES:
        return False
    if not _is_centered(_paint_box(child), container.get("box") or {}):
        return False
    if host_role in _BUTTON_CONTAINER_ROLES or child_role in _BUTTON_TEXT_ROLES:
        return (
            _has_surface(container)
            or host_role in {"button", "badge", "chip"}
            or host_meta.get("text_bearing_shell")
            or host_meta.get("plate_shell")
        )
    return False


_CAPTION_SHELL_ROLES = {None, "card", "container", "plate", "panel", "shape"}


def _text_align_token(node):
    style = node.get("style") or {}
    return str(style.get("align") or style.get("textAlignHorizontal") or "").upper()


def _pair_text_with_backplate_enabled(lcfg):
    """Consume archetype ``scene_grouping.pair_text_with_backplate`` (was previously dead)."""
    grouping = (lcfg or {}).get("scene_grouping") or {}
    return bool(grouping.get("pair_text_with_backplate"))


def _is_caption_plate(container, children):
    """Surfaced pill/plate with one centered copy line — IG caption-paragraph geometry.

    Distinct from CTA buttons: roles are ordinary headline/body/caption copy on a
    painted shell (card/shape/plate). Requires real inset padding so a speck on a
    backdrop never becomes a HUG caption frame.

    Explicit LEFT/RIGHT alignment never counts as a caption plate (chat bubbles and
    left-column stats share the padded-shell geometry but are not IG caption pills).
    """
    if len(children) != 1:
        return False
    child = children[0]
    if child.get("target") != "text":
        return False
    if not _has_surface(container):
        return False
    host_role = (container.get("meta") or {}).get("role")
    if host_role not in _CAPTION_SHELL_ROLES:
        return False
    child_role = str((child.get("meta") or {}).get("role", "text")).lower()
    if child_role in _NON_FLOW_TEXT_ROLES:
        return False
    paint = _paint_box(child)
    pb = container.get("box") or {}
    if not _is_padded_child(paint, pb):
        return False
    align = _text_align_token(child)
    if align in {"LEFT", "RIGHT", "MIN", "MAX"}:
        return False
    if align == "CENTER":
        return True
    return _is_centered(paint, pb)


def _is_stat_pill(container, children, canvas=None):
    """Left-column text + semi-transparent plate (Hears-style stats), not centered captions.

    Requires a painted shell, padded copy, left/MIN alignment (or left-biased paint), and
    when canvas is known the plate must sit in the left ~42% of the frame.
    """
    if len(children) != 1:
        return False
    child = children[0]
    if child.get("target") != "text":
        return False
    if not _has_surface(container):
        return False
    host_role = (container.get("meta") or {}).get("role")
    if host_role not in (_CAPTION_SHELL_ROLES | {"pill", "chip", "badge", "stat-pill"}):
        return False
    child_role = str((child.get("meta") or {}).get("role", "text")).lower()
    if child_role in _NON_FLOW_TEXT_ROLES | _BUTTON_TEXT_ROLES:
        return False
    paint = _paint_box(child)
    pb = container.get("box") or {}
    if not _is_padded_child(paint, pb):
        return False
    align = _text_align_token(child)
    # Explicit LEFT wins over geometric-center (wide copy on a narrow pill looks centered).
    if align == "CENTER":
        return False
    if align not in {"LEFT", "MIN"} and _is_centered(paint, pb):
        return False
    left_pad = float(paint.get("x", 0) - pb.get("x", 0))
    right_pad = float(
        pb.get("x", 0) + pb.get("w", 0) - (paint.get("x", 0) + paint.get("w", 0))
    )
    left_biased = align in {"LEFT", "MIN"} or left_pad + 1.0 < right_pad
    if not left_biased:
        return False
    if canvas:
        cw = float((canvas or {}).get("w") or 0)
        if cw > 0:
            cx = float(pb.get("x", 0)) + float(pb.get("w", 0)) / 2
            if cx > cw * 0.42:
                return False
    return True


def _shell_corner_radius(container) -> float:
    radius = container.get("radius")
    if radius is None:
        radius = (container.get("style") or {}).get("radius")
    try:
        return float(radius or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_reply_quote_node(node, host=None):
    """Nested reply-quote plate/group inside a chat bubble (IG DM style)."""
    if not node:
        return False
    role = str((node.get("meta") or {}).get("role") or "").lower()
    if role in {"reply-quote", "quote", "nested-quote"}:
        return True
    if node.get("target") not in {"group", "shape"}:
        return False
    box = node.get("box") or {}
    host_box = (host or {}).get("box") or {}
    if host_box and _area(host_box) > 0:
        if _area(box) > _area(host_box) * 0.78:
            return False
        if _inside(box, host_box) < 0.90:
            return False
        # Quotes sit inset from the left rail of the parent bubble.
        left_inset = float(box.get("x", 0) - host_box.get("x", 0))
        if left_inset < 4.0:
            return False
    kids = list(node.get("children") or [])
    if kids:
        if not any(c.get("target") == "text" for c in kids):
            return False
        if any(c.get("target") not in {"text", "shape", "icon"} for c in kids):
            return False
        return _has_surface(node) or any(_has_surface(c) for c in kids)
    return bool(_has_surface(node) and _shell_corner_radius(node) >= 4.0)


def _bubble_content_children(children):
    """Split bubble kids into body texts vs optional nested reply-quote plates."""
    quotes, texts, others = [], [], []
    for child in children or []:
        if child.get("target") == "text":
            texts.append(child)
        elif _is_reply_quote_node(child):
            quotes.append(child)
        else:
            others.append(child)
    return texts, quotes, others


def _is_message_bubble_pattern(container, children):
    """Rounded plate with inset body copy — chat/SMS bubbles, not centered CTAs.

    Strong-evidence only: surfaced shell, rounded corners (or explicit bubble role),
    text (+ optional nested reply-quote) children that are padded in, not a button.
    """
    if not children:
        return False
    texts, quotes, others = _bubble_content_children(children)
    if others or not texts:
        return False
    host_role = str((container.get("meta") or {}).get("role") or "").lower() or None
    if host_role in {"button", "badge", "chip", "cta"}:
        return False
    if host_role not in _MESSAGE_BUBBLE_SHELL_ROLES and host_role not in _MESSAGE_BUBBLE_ROLES:
        return False
    if not _has_surface(container):
        return False
    box = container.get("box") or {}
    h = float(box.get("h") or 0)
    radius = _shell_corner_radius(container)
    explicit = host_role in _MESSAGE_BUBBLE_ROLES or bool(
        (container.get("meta") or {}).get("message_bubble"))
    if not explicit and not (h > 0 and radius >= max(8.0, 0.12 * h)):
        return False
    if len(texts) == 1 and not quotes and _is_centered(_paint_box(texts[0]), box):
        child_role = str((texts[0].get("meta") or {}).get("role") or "text").lower()
        align = _text_align_token(texts[0])
        # Explicit left/right chat copy is not a CTA even on a card shell.
        if align not in {"LEFT", "RIGHT", "MIN", "MAX"}:
            if child_role in _BUTTON_TEXT_ROLES or host_role in _BUTTON_CONTAINER_ROLES:
                return False
            # A single centered label on a pill is a button, not a chat bubble.
            if h > 0 and radius >= 0.40 * h and child_role in {"label", "cta", "button", "text"}:
                paint = _paint_box(texts[0])
                if paint.get("w", 0) < box.get("w", 0) * 0.85:
                    return False
    padding = _layout_padding(box, texts + quotes)
    # Bubbles always inset the copy; flush full-bleed text is a plate label, not a bubble.
    if max(padding.get("left", 0), padding.get("right", 0)) < 6.0:
        return False
    if max(padding.get("top", 0), padding.get("bottom", 0)) < 4.0:
        return False
    return True


def _message_bubble_layout(container, children):
    """Auto Layout for a chat bubble: vertical stack of lines, HUG, left/top padding."""
    pb = container.get("box") or {}
    padding = _layout_padding(pb, children)
    if len(children) == 1:
        mode = "VERTICAL" if pb.get("h", 0) > pb.get("w", 0) * 1.25 else "HORIZONTAL"
        return _emit_figma_layout_aliases({
            "mode": mode, "confidence": 0.88, "gap": 0, "itemSpacing": 0,
            "padding": padding, "align": "MIN", "counterAlign": "MIN",
            "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN",
            "primarySizing": "HUG", "counterSizing": "HUG",
        })
    boxes = [_paint_box(c) for c in children]
    ordered = sorted(boxes, key=lambda b: b.get("y", 0))
    gaps = [ordered[i + 1]["y"] - (ordered[i]["y"] + ordered[i]["h"])
            for i in range(len(ordered) - 1)]
    return _emit_figma_layout_aliases({
        "mode": "VERTICAL", "confidence": 0.90,
        "gap": _item_spacing([g for g in gaps if g >= 0]),
        "padding": padding, "align": "MIN", "counterAlign": "MIN",
        "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN",
        "primarySizing": "HUG", "counterSizing": "HUG",
    })


def _layout_padding(container_box, children):
    pb = container_box or {}
    boxes = [_paint_box(child) for child in children]
    return {
        "left": round(max(0.0, min(b.get("x", 0) for b in boxes) - pb.get("x", 0)), 2),
        "right": round(max(0.0, pb.get("x", 0) + pb.get("w", 0) - max(b.get("x", 0) + b.get("w", 0) for b in boxes)), 2),
        "top": round(max(0.0, min(b.get("y", 0) for b in boxes) - pb.get("y", 0)), 2),
        "bottom": round(max(0.0, pb.get("y", 0) + pb.get("h", 0) - max(b.get("y", 0) + b.get("h", 0) for b in boxes)), 2),
    }


def _item_spacing(gaps):
    """Median gap, snapped to the nearest integer when the samples justify it."""
    if not gaps:
        return 0
    value = median(gaps)
    if abs(value - round(value)) <= 0.75:
        return int(round(value))
    return round(value, 2)


def _counter_alignment(boxes, mode):
    """Measure the counter-axis edge children actually share instead of assuming one.

    Returns the Figma alignment token with the tightest measured spread (MIN/CENTER/MAX)
    when that spread is within tolerance, otherwise the historical default for the axis.
    """
    if mode == "HORIZONTAL":
        starts = [b.get("y", 0) for b in boxes]
        ends = [b.get("y", 0) + b.get("h", 0) for b in boxes]
        centers = [b.get("y", 0) + b.get("h", 0) / 2 for b in boxes]
        tol = max(2.0, median([max(1.0, b.get("h", 1)) for b in boxes]) * 0.08)
        default = "CENTER"
        candidates = ("CENTER", "MIN", "MAX")
    else:
        starts = [b.get("x", 0) for b in boxes]
        ends = [b.get("x", 0) + b.get("w", 0) for b in boxes]
        centers = [b.get("x", 0) + b.get("w", 0) / 2 for b in boxes]
        tol = max(2.0, median([max(1.0, b.get("w", 1)) for b in boxes]) * 0.08)
        default = "MIN"
        candidates = ("MIN", "CENTER", "MAX")
    spreads = {
        "MIN": max(starts) - min(starts),
        "CENTER": max(centers) - min(centers),
        "MAX": max(ends) - min(ends),
    }
    best = min(candidates, key=lambda name: spreads[name])
    return best if spreads[best] <= tol else default


def _emit_figma_layout_aliases(layout):
    if layout.get("mode") not in ("HORIZONTAL", "VERTICAL"):
        return layout
    layout.setdefault("itemSpacing", layout.get("gap", 0))
    if layout.get("align") is not None:
        layout.setdefault("primaryAxisAlignItems", layout["align"])
    if layout.get("counterAlign") is not None:
        layout.setdefault("counterAxisAlignItems", layout["counterAlign"])
    return layout


def _passthrough_corner_radius(node):
    radius = node.get("radius")
    if radius is None:
        radius = (node.get("style") or {}).get("radius")
    if radius is None:
        return
    node.setdefault("meta", {})["cornerRadius"] = radius
    if node.get("radius") is None and isinstance(radius, (int, float)):
        node["radius"] = radius


def infer_auto_layout(container, children):
    """Return Figma layout intent or NONE when geometry should remain absolute."""
    pb = container["box"]
    if not children:
        return {"mode": "NONE", "confidence": 0.0}
    role = (container.get("meta") or {}).get("role")
    if role in _MESSAGE_BUBBLE_ROLES or role == "message-bubble":
        return _message_bubble_layout(container, children)
    if role == "stat-pill" and len(children) == 1:
        paint = _paint_box(children[0])
        padding = _layout_padding(pb, children)
        mode = "VERTICAL" if pb.get("h", 0) >= pb.get("w", 0) else "HORIZONTAL"
        return _emit_figma_layout_aliases({
            "mode": mode, "confidence": 0.90, "gap": 0, "itemSpacing": 0,
            "padding": padding, "align": "MIN", "counterAlign": "MIN",
            "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN",
            "primarySizing": "HUG", "counterSizing": "HUG",
        })
    if role == "message-row":
        boxes = [_paint_box(c) for c in children]
        ordered = sorted(boxes, key=lambda b: b.get("x", 0))
        gaps = [ordered[i + 1]["x"] - (ordered[i]["x"] + ordered[i]["w"])
                for i in range(len(ordered) - 1)]
        return _emit_figma_layout_aliases({
            "mode": "HORIZONTAL", "confidence": 0.88,
            "gap": _item_spacing([g for g in gaps if g >= 0]),
            "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "align": "MIN",
            "counterAlign": _counter_alignment(boxes, "HORIZONTAL"),
            "primarySizing": "HUG", "counterSizing": "HUG",
        })
    boxes = [_paint_box(c) for c in children]
    padding = _layout_padding(pb, children)
    if len(children) == 1:
        paint = _paint_box(children[0])
        is_button = _is_button_pattern(container, children)
        is_caption = (not is_button) and _is_caption_plate(container, children)
        is_stat = (not is_button) and (not is_caption) and _is_stat_pill(container, children)
        if is_button or is_caption or (
            _is_centered(paint, pb)
            and role in ("button", "badge", "chip", "caption-plate")
        ):
            mode = "VERTICAL" if pb.get("h", 0) > pb.get("w", 0) * 1.35 else "HORIZONTAL"
            # Floor padding so CTA labels never sit flush against chrome (ad 013 badge /
            # bottom bar). Measured insets stay when already generous.
            child_style = children[0].get("style") or {}
            try:
                fs = float(child_style.get("fontSize") or 0)
            except (TypeError, ValueError):
                fs = 0.0
            min_pad = max(4.0, round(fs * 0.12, 2)) if fs > 0 else 4.0
            padding = {
                side: round(max(float(padding.get(side, 0) or 0), min_pad), 2)
                for side in ("left", "right", "top", "bottom")
            }
            return _emit_figma_layout_aliases({
                "mode": mode,
                "confidence": 0.92 if (is_button or is_caption) else 0.9,
                "gap": 0, "itemSpacing": 0,
                "padding": padding, "align": "CENTER", "counterAlign": "CENTER",
                "primaryAxisAlignItems": "CENTER", "counterAxisAlignItems": "CENTER",
                "primarySizing": "HUG", "counterSizing": "HUG",
            })
        if is_stat or role == "stat-pill":
            mode = "VERTICAL" if pb.get("h", 0) >= pb.get("w", 0) else "HORIZONTAL"
            return _emit_figma_layout_aliases({
                "mode": mode, "confidence": 0.90, "gap": 0, "itemSpacing": 0,
                "padding": padding, "align": "MIN", "counterAlign": "MIN",
                "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN",
                "primarySizing": "HUG", "counterSizing": "HUG",
            })
        # Padded card: a surfaced plate/card wrapping one substantial, fully-inset
        # child becomes a HUG frame so the plate resizes with its content instead of
        # freezing at pixel size.  The measured four-side padding reproduces the
        # original box exactly (see _is_padded_child), so this never moves geometry.
        # Centered copy prefers CENTER; left-biased copy keeps MIN.
        if (_has_surface(container)
                and role in (None, "card", "container", "plate", "panel", "shape")
                and children[0].get("target") in ("text", "image", "icon")
                and _is_padded_child(paint, pb)):
            mode = "VERTICAL" if pb.get("h", 0) >= pb.get("w", 0) else "HORIZONTAL"
            centered = (
                _text_align_token(children[0]) == "CENTER" or _is_centered(paint, pb)
            )
            align = "CENTER" if centered else "MIN"
            return _emit_figma_layout_aliases({
                "mode": mode, "confidence": 0.85, "gap": 0, "itemSpacing": 0,
                "padding": padding, "align": align, "counterAlign": align,
                "primaryAxisAlignItems": align, "counterAxisAlignItems": align,
                "primarySizing": "HUG", "counterSizing": "HUG",
            })
        return {"mode": "NONE", "confidence": 0.3}

    if any(_overlap(a, b) > 0.06 for i, a in enumerate(boxes) for b in boxes[i + 1:]):
        return {"mode": "NONE", "confidence": 0.2}
    mh = max(1.0, median(b["h"] for b in boxes))
    mw = max(1.0, median(b["w"] for b in boxes))
    cy = [b["y"] + b["h"] / 2 for b in boxes]
    cx = [b["x"] + b["w"] / 2 for b in boxes]
    row_spread = (max(cy) - min(cy)) / mh
    col_spread = (max(cx) - min(cx)) / mw

    if row_spread <= 0.35:
        ordered = sorted(boxes, key=lambda b: b["x"])
        gaps = [ordered[i + 1]["x"] - (ordered[i]["x"] + ordered[i]["w"])
                for i in range(len(ordered) - 1)]
        if _consistent(gaps):
            return _emit_figma_layout_aliases({
                "mode": "HORIZONTAL", "confidence": round(0.95 - min(.2, row_spread * .2), 3),
                "gap": _item_spacing(gaps), "padding": padding,
                "align": "MIN", "counterAlign": _counter_alignment(boxes, "HORIZONTAL"),
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            })
    if col_spread <= 0.35:
        ordered = sorted(boxes, key=lambda b: b["y"])
        gaps = [ordered[i + 1]["y"] - (ordered[i]["y"] + ordered[i]["h"])
                for i in range(len(ordered) - 1)]
        if _consistent(gaps):
            return _emit_figma_layout_aliases({
                "mode": "VERTICAL", "confidence": round(0.95 - min(.2, col_spread * .2), 3),
                "gap": _item_spacing(gaps), "padding": padding,
                "align": "MIN", "counterAlign": _counter_alignment(boxes, "VERTICAL"),
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            })
    return {"mode": "NONE", "confidence": 0.25}


def _has_surface(node):
    if node.get("fill") or node.get("stroke"):
        return True
    style = node.get("style") or {}
    return bool(style.get("fills") or style.get("fill") or style.get("color")
                or node.get("radius") or style.get("radius"))


def _surface_from(node):
    if node.get("fill"):
        return node.get("fill")
    style = node.get("style") or {}
    fills = style.get("fills")
    if isinstance(fills, list) and fills:
        return fills[0]
    if style.get("fill") is not None:
        return style.get("fill")
    if style.get("color"):
        return {"kind": "flat", "color": style["color"]}
    return None


def _hoist_surface_material(host, shell):
    """Move a folded full-bleed shell's complete paint contract onto its frame.

    A card/button shell is intentionally removed once its parent becomes the native
    Figma frame.  Copying only the first fill made multi-paint cards and shadows
    disappear at that structural boundary.
    """
    shell_style = shell.get("style") or {}
    host_style = dict(host.get("style") or {})

    # Preserve all style-provided paints instead of reducing them to _surface_from's
    # first fill.  A top-level fill remains authoritative when upstream supplied one.
    if host.get("fill") is None and shell.get("fill") is not None:
        host["fill"] = deepcopy(shell["fill"])
    for key in ("fills", "paints", "fill", "background", "color"):
        if key in shell_style and key not in host_style:
            host_style[key] = deepcopy(shell_style[key])

    if host.get("stroke") is None and shell.get("stroke") is not None:
        host["stroke"] = deepcopy(shell["stroke"])
    for key in ("strokes", "stroke"):
        if key in shell_style and key not in host_style:
            host_style[key] = deepcopy(shell_style[key])

    if host_style:
        host["style"] = host_style
    if not host.get("effects"):
        effects = shell.get("effects")
        if not isinstance(effects, list):
            effects = shell_style.get("effects")
        if isinstance(effects, list) and effects:
            host["effects"] = deepcopy(effects)


def _normalize_group_surface(node):
    """Promote style-only fills onto groups so the Figma compiler can frame-promote cards."""
    if node.get("target") != "group" or _has_surface(node):
        return
    fill = _surface_from(node)
    if fill is not None:
        node["fill"] = fill
    style = node.get("style") or {}
    if node.get("radius") is None and style.get("radius") is not None:
        node["radius"] = style.get("radius")


def _hoist_background_surface(group):
    """Card panels often keep the painted background on an inner shape — hoist it to the group."""
    if group.get("target") != "group" or _has_surface(group):
        return
    children = group.get("children") or []
    parent_box = group.get("box") or {}
    best = None
    best_area = 0.0
    for child in children:
        if child.get("target") != "shape" or not _has_surface(child):
            continue
        meta = child.get("meta") or {}
        role = str(meta.get("role") or "").lower().replace("-", "_")
        # Outline rings / quote strokes are intentional siblings, not plate fills.
        if (
            meta.get("stroke_outline_shell")
            or meta.get("white_ring")
            or meta.get("quote_frame")
            or role in {
                "ring", "inset_ring", "circular_ring", "quote_frame", "quote",
                "testimonial_frame",
            }
        ):
            continue
        child_box = child.get("box") or {}
        if _inside(child_box, parent_box) < 0.88 or _area(child_box) < _area(parent_box) * 0.72:
            continue
        area = _area(child_box)
        if area > best_area:
            best_area = area
            best = child
    if not best:
        return
    _hoist_surface_material(group, best)
    if group.get("radius") is None:
        group["radius"] = best.get("radius") or (best.get("style") or {}).get("radius")
    shell_id = best.get("id")
    if shell_id:
        group["children"] = [child for child in children if child.get("id") != shell_id]


def _annotate_stack_children(parent, children):
    """Emit child layout hints consumed by the Figma plugin's applyChildLayout()."""
    layout = parent.get("layout") or {}
    mode = layout.get("mode")
    if mode not in ("HORIZONTAL", "VERTICAL") or not children:
        return
    role = (parent.get("meta") or {}).get("role")
    if (role in ("button", "badge", "chip", "caption-plate", "message-bubble", "stat-pill")
            or role in _MESSAGE_BUBBLE_ROLES
            or _is_button_pattern(parent, children)
            or _is_caption_plate(parent, children)
            or _is_stat_pill(parent, children)
            or _is_message_bubble_pattern(parent, children)):
        align = "CENTER" if (
            role in ("button", "badge", "chip", "caption-plate")
            or _is_button_pattern(parent, children)
            or _is_caption_plate(parent, children)
        ) else "MIN"
        for child in children:
            hints = dict(child.get("layout") or {})
            hints["layoutAlign"] = align
            hints["layoutSizingHorizontal"] = "HUG"
            hints["layoutSizingVertical"] = "HUG"
            hints.pop("layoutPositioning", None)
            child["layout"] = hints
        return
    if role == "caption-stack":
        for child in children:
            hints = dict(child.get("layout") or {})
            hints["layoutAlign"] = "CENTER"
            hints["layoutSizingHorizontal"] = "HUG"
            hints["layoutSizingVertical"] = "HUG"
            hints.pop("layoutPositioning", None)
            child["layout"] = hints
        return
    if role in {"stat-stack", "stat-row", "benefit-stack", "rating-strip", "logo-strip",
                "message-row", "engagement-row", "comparison-set",
                "ama-sticker", "circular-inset", "timeline", "timeline-step",
                "review-bar"}:
        for child in children:
            hints = dict(child.get("layout") or {})
            # Timeline spines stay absolutely positioned inside the VERTICAL frame.
            if (
                role == "timeline"
                and (
                    (child.get("meta") or {}).get("timeline_connector")
                    or hints.get("layoutPositioning") == "ABSOLUTE"
                )
            ):
                hints["layoutPositioning"] = "ABSOLUTE"
                child["layout"] = hints
                continue
            hints["layoutAlign"] = "MIN" if role != "comparison-set" else "CENTER"
            hints["layoutSizingHorizontal"] = "HUG"
            hints["layoutSizingVertical"] = "HUG"
            hints.pop("layoutPositioning", None)
            child["layout"] = hints
        return
    parent_box = parent.get("box") or {}
    boxes = [child.get("box") or {} for child in children]
    if mode == "VERTICAL":
        axis_centers = [box.get("x", 0) + box.get("w", 0) / 2 for box in boxes]
        spread = max(1.0, median([max(1.0, box.get("w", 1)) for box in boxes]))
    else:
        axis_centers = [box.get("y", 0) + box.get("h", 0) / 2 for box in boxes]
        spread = max(1.0, median([max(1.0, box.get("h", 1)) for box in boxes]))
    axis_center = median(axis_centers)

    for index, child in enumerate(children):
        child_box = child.get("box") or {}
        constraints = child.get("constraints") or _constraints(child_box, parent_box)
        hints = dict(child.get("layout") or {})
        overlaps = any(
            index != other and _overlap(child_box, boxes[other]) > 0.12
            for other in range(len(children))
        )
        if mode == "VERTICAL":
            child_center = child_box.get("x", 0) + child_box.get("w", 0) / 2
        else:
            child_center = child_box.get("y", 0) + child_box.get("h", 0) / 2
        if overlaps or abs(child_center - axis_center) > max(6.0, spread * 0.45):
            hints["layoutPositioning"] = "ABSOLUTE"
        elif mode == "VERTICAL":
            width_frac = child_box.get("w", 0) / max(1.0, parent_box.get("w", 1))
            if constraints.get("horizontal") == "STRETCH" or width_frac >= 0.92:
                hints["layoutGrow"] = 1
                hints["layoutSizingHorizontal"] = "FILL"
            elif constraints.get("horizontal") == "CENTER":
                hints["layoutAlign"] = "CENTER"
        else:
            height_frac = child_box.get("h", 0) / max(1.0, parent_box.get("h", 1))
            if constraints.get("vertical") == "STRETCH" or height_frac >= 0.92:
                hints["layoutGrow"] = 1
                hints["layoutSizingVertical"] = "FILL"
            elif constraints.get("vertical") == "CENTER":
                hints["layoutAlign"] = "CENTER"
        if hints:
            child["layout"] = hints


def _wrap_repeated_card_grids(roots):
    """Wrap benchmark-style repeated cards sharing a signature into one auto-layout row/column."""
    by_signature = {}
    for node in roots:
        if node.get("target") != "group":
            continue
        signature = (node.get("meta") or {}).get("repeat_signature")
        if signature:
            by_signature.setdefault(signature, []).append(node)
    wrappers = []
    consumed = set()
    for signature, members in by_signature.items():
        if len(members) < 2:
            continue
        box = _union([member.get("box") or {} for member in members])
        layout = infer_auto_layout({"box": box, "meta": {"role": "card-grid"}}, members)
        if layout.get("mode") not in ("HORIZONTAL", "VERTICAL") or layout.get("confidence", 0) < 0.5:
            continue
        if layout["mode"] == "HORIZONTAL":
            members = sorted(members, key=lambda node: (node.get("box", {}).get("x", 0), node.get("id", "")))
        else:
            members = sorted(members, key=lambda node: (node.get("box", {}).get("y", 0), node.get("id", "")))
        wrapper = {
            "id": f"repeat-grid-{signature}",
            "target": "group",
            "box": box,
            "z": min(_node_z(member) for member in members),
            "children": members,
            "layout": layout,
            "meta": {
                "role": "card-grid",
                "repeat_signature": signature,
                "layout_confidence": layout.get("confidence"),
            },
        }
        _annotate_stack_children(wrapper, members)
        wrappers.append(wrapper)
        consumed.update(member.get("id") for member in members)
    if not wrappers:
        return roots
    out = [node for node in roots if node.get("id") not in consumed]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _backgroundish(node, canvas):
    meta = node.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"}:
        return True
    canvas_area = max(1.0, float(canvas.get("w", 1) or 1) * float(canvas.get("h", 1) or 1))
    return _area(node.get("box") or {}) >= canvas_area * 0.88


def _merged_spans(boxes, axis):
    """Merge box projections on one axis into disjoint occupied spans."""
    key, size = ("y", "h") if axis == "y" else ("x", "w")
    intervals = sorted(
        (float(b.get(key, 0) or 0), float(b.get(key, 0) or 0) + float(b.get(size, 0) or 0))
        for b in boxes
    )
    spans = []
    for start, end in intervals:
        if spans and start <= spans[-1][1] + 2.0:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return spans


def _cut_bands(members, canvas, ncfg):
    """One XY-cut level: split members along genuinely empty whitespace, or None."""
    if len(members) < 4:
        return None
    boxes = [node.get("box") or {} for node in members]
    for axis in ("y", "x"):
        dim = float(canvas.get("h" if axis == "y" else "w", 1) or 1)
        min_gap = max(float(ncfg.get("min_gap_px", 18.0)),
                      float(ncfg.get("min_gap_frac", 0.05)) * dim)
        spans = _merged_spans(boxes, axis)
        if len(spans) < 2:
            continue
        cuts = [index for index in range(len(spans) - 1)
                if spans[index + 1][0] - spans[index][1] >= min_gap]
        if not cuts:
            continue
        # Band ranges between cuts; assign members by projected center.
        limits = [spans[index][1] for index in cuts]
        key, size = ("y", "h") if axis == "y" else ("x", "w")
        bands = [[] for _ in range(len(limits) + 1)]
        for node in members:
            box = node.get("box") or {}
            center = float(box.get(key, 0) or 0) + float(box.get(size, 0) or 0) / 2
            slot = sum(1 for limit in limits if center > limit)
            bands[slot].append(node)
        bands = [band for band in bands if band]
        if len(bands) >= 2:
            return axis, bands
    return None


def _band_name(members, box, canvas):
    roles = {str((node.get("meta") or {}).get("role") or "").lower() for node in members}
    targets = {node.get("target") for node in members}
    ch = max(1.0, float(canvas.get("h", 1) or 1))
    cy = (box.get("y", 0) + box.get("h", 0) / 2) / ch
    if roles & {"cta", "button"} and len(members) <= 4:
        return "CTA"
    if "logo" in roles and cy <= 0.30:
        return "Header"
    if cy <= 0.16:
        return "Header"
    if cy >= 0.84:
        return "Footer"
    if roles & {"product", "person", "product_cluster", "illustration", "avatar"}:
        return "Hero"
    if targets <= {"text"}:
        return "Text Stack"
    return "Group"


def _band_wrap(members, canvas, ncfg, depth):
    """Recursively wrap whitespace-separated bands, or None when no confident cut exists."""
    if depth > int(ncfg.get("max_depth", 2)):
        return None
    cut = _cut_bands(members, canvas, ncfg)
    if not cut:
        return None
    axis, bands = cut
    if not any(len(band) >= 2 for band in bands):
        return None
    out = []
    for band in bands:
        if len(band) < 2:
            out.extend(band)
            continue
        inner = _band_wrap(band, canvas, ncfg, depth + 1)
        children = inner if inner else sorted(
            band, key=lambda node: (_node_z(node), node.get("id", "")))
        box = _union([node.get("box") or {} for node in band])
        layout = infer_auto_layout({"box": box, "meta": {"role": "band"}}, children)
        wrapper = {
            "id": "band-" + hashlib.sha1(
                "|".join(sorted(str(node.get("id")) for node in band)).encode("utf-8")
            ).hexdigest()[:10],
            "target": "group",
            "box": box,
            "z": min(_node_z(node) for node in band),
            "children": children,
            "layout": layout,
            "meta": {
                "role": "band",
                "band_axis": axis,
                "semantic_name": _band_name(band, box, canvas),
                "layout_confidence": layout.get("confidence"),
                "deterministic_geometry": True,
                "source": "xycut",
            },
        }
        out.append(wrapper)
    return out


def _band_groups(roots, canvas, lcfg):
    """Conservative XY-cut: only whitespace that no element crosses can split bands.

    Ads are simpler than app UIs — a clear horizontal/vertical whitespace corridor is
    almost always a real design seam (header / hero / footer).  Groups are created only
    when a cut produces at least two bands and a band has two or more members, so a
    layout without strong separation stays exactly as flat as before.
    """
    ncfg = (lcfg or {}).get("nesting") or {}
    if not ncfg.get("enabled", True):
        return roots
    movable = [node for node in roots if not _backgroundish(node, canvas)]
    if len(movable) < int(ncfg.get("min_nodes", 6)):
        return roots
    wrapped = _band_wrap(movable, canvas, ncfg, depth=1)
    if wrapped is None:
        return roots
    out = [node for node in roots if _backgroundish(node, canvas)]
    out.extend(wrapped)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _relaxed_group_signature(group):
    """Structure-only signature: ignores text content and fine size differences."""
    box = group.get("box") or {}
    payload = [
        (
            child.get("target"),
            (child.get("meta") or {}).get("role"),
            round(float((child.get("box") or {}).get("w", 0) or 0) / max(1.0, box.get("w", 1)), 1),
            round(float((child.get("box") or {}).get("h", 0) or 0) / max(1.0, box.get("h", 1)), 1),
        )
        for child in group.get("children") or []
    ]
    payload.append(round(float(box.get("w", 1) or 1) / max(1.0, float(box.get("h", 1) or 1)), 1))
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _annotate_component_candidates(roots, rcfg):
    """Mark repeated structures/leaves as component candidates (metadata only).

    Exact repeats are already instantiated via ``component``; this pass adds the
    additive ``meta.component_candidate`` marker for near-repeats (same structure,
    different copy) and repeated identical leaves (rating stars, feature icons) so
    the plugin/compiler can turn them into components later without any geometry
    or material change here.
    """
    if not (rcfg or {}).get("enabled", True):
        return
    size_tol = float((rcfg or {}).get("size_tolerance", 0.12))
    min_leaf = int((rcfg or {}).get("min_leaf_instances", 3))

    groups, leaves = [], []

    def _walk(node):
        children = node.get("children") or []
        if node.get("target") == "group" and children:
            groups.append(node)
        elif not children and node.get("target") in {"icon", "image", "shape"}:
            leaves.append(node)
        for child in children:
            _walk(child)

    for root in roots:
        _walk(root)

    by_relaxed = {}
    for group in groups:
        by_relaxed.setdefault(_relaxed_group_signature(group), []).append(group)
    for signature, members in by_relaxed.items():
        if len(members) < 2 or all(member.get("component") for member in members):
            continue
        ids = sorted(str(member.get("id")) for member in members)
        for member in members:
            member.setdefault("meta", {})["component_candidate"] = {
                "key": f"repeat~{signature}", "confidence": 0.75,
                "count": len(members), "members": ids,
            }

    by_kind = {}
    for leaf in leaves:
        role = str((leaf.get("meta") or {}).get("role") or "").lower()
        if role in {"background", "plate", "clean plate"}:
            continue
        by_kind.setdefault((leaf.get("target"), role), []).append(leaf)
    for (target, role), members in by_kind.items():
        if len(members) < min_leaf:
            continue
        med_w = median([max(1.0, (leaf.get("box") or {}).get("w", 1)) for leaf in members])
        med_h = median([max(1.0, (leaf.get("box") or {}).get("h", 1)) for leaf in members])
        similar = [
            leaf for leaf in members
            if abs((leaf.get("box") or {}).get("w", 0) - med_w) <= med_w * size_tol
            and abs((leaf.get("box") or {}).get("h", 0) - med_h) <= med_h * size_tol
        ]
        if len(similar) < min_leaf:
            continue
        signature = hashlib.sha1(
            f"{target}:{role}:{round(med_w, 1)}x{round(med_h, 1)}".encode()
        ).hexdigest()[:10]
        ids = sorted(str(leaf.get("id")) for leaf in similar)
        for leaf in similar:
            leaf.setdefault("meta", {}).setdefault("component_candidate", {
                "key": f"leafrep~{signature}", "confidence": 0.6,
                "count": len(similar), "members": ids,
            })


def _first_text_content(node):
    if node.get("target") == "text" and node.get("text"):
        return str(node["text"])
    best = None
    for child in sorted(
        node.get("children") or [],
        key=lambda item: ((item.get("box") or {}).get("y", 0), (item.get("box") or {}).get("x", 0)),
    ):
        best = _first_text_content(child)
        if best:
            return best
    return best


def _short(value, length=24):
    value = " ".join(str(value or "").split())
    return value if len(value) <= length else value[: length - 1] + "…"


def _apply_semantic_names(nodes):
    """Give structural frames designer-facing names; explicit clean names always win."""
    for node in nodes:
        children = node.get("children") or []
        if children:
            _apply_semantic_names(children)
        if node.get("target") != "group":
            continue
        meta = node.setdefault("meta", {})
        if node.get("name") or meta.get("semantic_name"):
            continue
        role = str(meta.get("role") or "")
        label = None
        if role == "button":
            text = _first_text_content(node)
            label = f"Button / {_short(text)}" if text else "Button"
        elif role == "caption-plate":
            text = _first_text_content(node)
            label = f"Caption / {_short(text)}" if text else "Caption"
        elif role == "stat-pill":
            text = _first_text_content(node)
            label = f"Stat / {_short(text)}" if text else "Stat"
        elif role == "stat-stack":
            label = "Stats"
        elif role == "stat-row":
            label = "Stats"
        elif role == "comparison-set":
            label = "Comparison"
        elif role == "stage-progression":
            label = "Progression"
        elif role == "checklist":
            label = "Checklist"
        elif role == "text-row":
            label = "Checklist" if (
                meta.get("checklist") or _looks_like_checklist_row(children)
            ) else "Row"
        elif role == "engagement-row":
            label = "Engagement"
        elif role == "ama-sticker":
            text = _first_text_content(node)
            label = f"AMA / {_short(text)}" if text else "AMA sticker"
        elif role == "quote-frame":
            text = _first_text_content(node)
            label = f"Quote / {_short(text)}" if text else "Quote"
        elif role == "circular-inset":
            label = "Circular inset"
        elif role == "timeline":
            label = "Timeline"
        elif role == "timeline-step":
            text = _first_text_content(node)
            label = f"Step / {_short(text)}" if text else "Step"
        elif role == "review-bar":
            label = "Reviews"
        elif role == "message-bubble":
            text = _first_text_content(node)
            label = f"Message / {_short(text)}" if text else "Message"
        elif role == "message-row":
            label = "Message row"
        elif role == "reply-quote":
            text = _first_text_content(node)
            label = f"Reply / {_short(text)}" if text else "Reply"
        elif role == "header-cluster":
            label = "Header"
        elif role == "caption-stack":
            label = "Caption"
        elif role == "text-stack":
            label = "Text Stack"
        elif role == "card-grid":
            label = f"Card Grid ({len(children)})"
        elif role == "panel-set":
            label = f"Panel Set ({len(children)})"
        elif role == "structural-grid":
            label = f"Grid ({len(children)})"
        elif role == "native-chart":
            label = "Chart"
        elif role == "card":
            label = "Card"
        if label:
            meta["semantic_name"] = label


def _finalize_vlm_group_layouts(nodes):
    """Evidence-gated Auto Layout for VLM wrappers: the hint never overrides geometry."""
    for node in nodes:
        children = node.get("children") or []
        if children:
            _finalize_vlm_group_layouts(children)
        meta = node.get("meta") or {}
        if meta.get("source") != "vlm-grouping" or node.get("layout") is not None:
            continue
        layout = infer_auto_layout(node, children)
        hint = str(meta.get("vlm_direction_hint") or "none")
        if layout.get("mode") in ("HORIZONTAL", "VERTICAL"):
            agrees = (layout["mode"] == "HORIZONTAL") == (hint == "row") if hint != "none" else None
            if agrees is not None:
                meta["vlm_direction_agrees"] = agrees
        node["layout"] = layout
        meta["layout_confidence"] = layout.get("confidence")


class _TreeWithNotice(list):
    """Root list that carries the optional VLM-grouping outcome for the caller.

    Subclassing list keeps every existing consumer working unchanged (iteration,
    JSON serialization, equality), while scene_intent.plan can surface the advisory
    grouping status instead of it being silently dropped."""

    vlm_grouping: Optional[dict] = None


_STRUCTURE_GROUP_KEYS = (
    "structure_group_id", "repeat_group_id", "panel_set_id", "grid_group_id",
    "comparison_group_id", "chart_group_id",
)
_IMPLICIT_STRUCTURE_ROLES = {
    "panel", "image-panel", "photo-panel", "comparison-panel", "comparison-column", "triptych-panel",
    "repeated-row", "stat-row", "table-row", "data-row",
}


def _chart_is_deterministic(members):
    """Native-chart only when every member is a routed chart primitive (no raster plot)."""
    if any(member.get("target") not in {"shape", "text", "icon"} for member in members):
        return False
    return members_support_native_chart(members)


def _structure_key(node):
    meta = node.get("meta") or {}
    for field in _STRUCTURE_GROUP_KEYS:
        value = meta.get(field)
        if value not in (None, ""):
            return field, str(value)
    role = str(meta.get("role") or "").strip().lower().replace("_", "-")
    if role in _IMPLICIT_STRUCTURE_ROLES:
        # Implicit grouping is deliberately role-scoped. It still has to pass the
        # strict deterministic geometry gate below, so two unrelated panels do not
        # become a made-up responsive layout.
        return "role", role
    return None


def _axis_layout(members):
    box = _union([member.get("box") or {} for member in members])
    layout = infer_auto_layout({"box": box, "meta": {"role": "structural-set"}}, members)
    if layout.get("mode") not in ("HORIZONTAL", "VERTICAL"):
        return None
    if float(layout.get("confidence", 0) or 0) < .82:
        return None
    widths = [max(1.0, (member.get("box") or {}).get("w", 1)) for member in members]
    heights = [max(1.0, (member.get("box") or {}).get("h", 1)) for member in members]
    cross_sizes = heights if layout["mode"] == "HORIZONTAL" else widths
    if not _consistent(cross_sizes, max_cv=.16):
        return None
    return box, layout


def _grid_rows(members):
    """Return deterministic equal-column rows, or None for artistic/uneven geometry."""
    if len(members) < 4:
        return None
    ordered = sorted(members, key=lambda node: (
        (node.get("box") or {}).get("y", 0) + (node.get("box") or {}).get("h", 0) / 2,
        (node.get("box") or {}).get("x", 0), node.get("id", ""),
    ))
    typical_h = median([max(1.0, (node.get("box") or {}).get("h", 1)) for node in ordered])
    rows = []
    for node in ordered:
        cy = (node.get("box") or {}).get("y", 0) + (node.get("box") or {}).get("h", 0) / 2
        if not rows:
            rows.append([node])
            continue
        prior_centers = [
            (item.get("box") or {}).get("y", 0) + (item.get("box") or {}).get("h", 0) / 2
            for item in rows[-1]
        ]
        if abs(cy - median(prior_centers)) <= typical_h * .22:
            rows[-1].append(node)
        else:
            rows.append([node])
    if len(rows) < 2 or min(len(row) for row in rows) < 2:
        return None
    if len({len(row) for row in rows}) != 1:
        return None
    normalized = []
    reference_centers = None
    for row in rows:
        row = sorted(row, key=lambda node: ((node.get("box") or {}).get("x", 0), node.get("id", "")))
        axis = _axis_layout(row)
        if not axis or axis[1]["mode"] != "HORIZONTAL":
            return None
        centers = [
            (node.get("box") or {}).get("x", 0) + (node.get("box") or {}).get("w", 0) / 2
            for node in row
        ]
        if reference_centers is None:
            reference_centers = centers
        elif any(abs(a - b) > max(4.0, typical_h * .12)
                 for a, b in zip(reference_centers, centers)):
            return None
        normalized.append((row, axis))
    return normalized


def _wrap_structural_sets(roots):
    """Preserve proven panels, comparisons, repeated rows, grids, and simple charts.

    This pass never performs visual guessing. Explicit detector/VLM group IDs are accepted;
    implicit panel roles additionally require strict equal-size/alignment evidence. Complex
    charts remain intentional raster clusters upstream, while a chart made entirely from
    positively identified native primitives may be grouped without changing its geometry.
    """
    sets = {}
    for node in roots:
        key = _structure_key(node)
        if key:
            sets.setdefault(key, []).append(node)
    wrappers, consumed = [], set()
    for (field, value), members in sorted(sets.items(), key=lambda item: item[0]):
        if len(members) < 2:
            continue
        if field == "role" and value in {"panel", "image-panel", "photo-panel", "triptych-panel"} \
                and len(members) < 3:
            continue
        is_chart = field == "chart_group_id"
        if is_chart:
            if not _chart_is_deterministic(members):
                continue
            box = _union([member.get("box") or {} for member in members])
            layout = {"mode": "NONE", "confidence": 1.0}
            children = sorted(members, key=lambda node: (_node_z(node), node.get("id", "")))
            role = "native-chart"
        else:
            axis = _axis_layout(members)
            grid = None if axis else _grid_rows(members)
            if not axis and not grid:
                continue
            if axis:
                box, layout = axis
                reverse = layout["mode"] == "VERTICAL"
                key_name = "y" if reverse else "x"
                children = sorted(members, key=lambda node: (
                    (node.get("box") or {}).get(key_name, 0), node.get("id", "")))
                role = "panel-set" if "panel" in value or "comparison" in value else "repeated-set"
            else:
                row_nodes = []
                for index, (row, (row_box, row_layout)) in enumerate(grid):
                    row_nodes.append({
                        "id": f"struct-row-{value}-{index}", "target": "group",
                        "box": row_box, "z": min(_node_z(node) for node in row),
                        "children": row, "layout": row_layout,
                        "meta": {"role": "grid-row", "layout_confidence": row_layout["confidence"]},
                    })
                    _annotate_stack_children(row_nodes[-1], row)
                box = _union([node["box"] for node in row_nodes])
                layout = infer_auto_layout({"box": box, "meta": {"role": "structural-grid"}}, row_nodes)
                if layout.get("mode") != "VERTICAL" or layout.get("confidence", 0) < .82:
                    continue
                children, role = row_nodes, "structural-grid"
        stable = hashlib.sha1(f"{field}:{value}".encode()).hexdigest()[:10]
        wrapper = {
            "id": f"struct-{role}-{stable}", "target": "group", "box": box,
            "z": min(_node_z(node) for node in members), "children": children,
            "layout": layout,
            "meta": {
                "role": role, "structure_source": field, "structure_value": value,
                "layout_confidence": layout.get("confidence"),
                "deterministic_geometry": True,
            },
        }
        _annotate_stack_children(wrapper, children)
        wrappers.append(wrapper)
        consumed.update(node.get("id") for node in members)
    if not wrappers:
        return roots
    out = [node for node in roots if node.get("id") not in consumed]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _order_button_children(children):
    """Keep editable labels above painted shells in the frame tree."""
    def _rank(child):
        target = child.get("target")
        role = (child.get("meta") or {}).get("role", "")
        if target == "text" or role in _BUTTON_TEXT_ROLES:
            return 1
        if target in ("shape", "image", "icon"):
            return 0
        return 0
    return sorted(children, key=lambda child: (_rank(child), float(child.get("z", 0)), child.get("id", "")))


def _finalize_layout(nodes):
    for node in nodes:
        children = node.get("children") or []
        if children:
            _finalize_layout(children)
        _normalize_group_surface(node)
        children_before = list(node.get("children") or [])
        _hoist_background_surface(node)
        children = node.get("children") or []
        _passthrough_corner_radius(node)
        if node.get("target") == "group" and children and len(children) != len(children_before):
            node["layout"] = infer_auto_layout(node, children)
            node.setdefault("meta", {})["layout_confidence"] = node["layout"].get("confidence")
        layout = node.get("layout") or {}
        if layout.get("mode") in ("HORIZONTAL", "VERTICAL"):
            role = (node.get("meta") or {}).get("role")
            if (role in ("button", "caption-plate")
                    or _is_button_pattern(node, children)
                    or _is_caption_plate(node, children)):
                ordered = _order_button_children(children)
                if ordered != children:
                    node["children"] = ordered
                    children = ordered
            _annotate_stack_children(node, children)


def _component_signature(node):
    children = node.get("children") or []
    payload = {
        "type": node.get("target"),
        "fill": node.get("fill"),
        "radius": (node.get("style") or {}).get("radius"),
        "children": [
            {
                "type": c.get("target"),
                "role": (c.get("meta") or {}).get("role"),
                "text": c.get("text"),
                "style": c.get("meta", {}).get("style_id"),
                "ratio": [round(c.get("box", {}).get("w", 0) / max(1, node["box"]["w"]), 2),
                          round(c.get("box", {}).get("h", 0) / max(1, node["box"]["h"]), 2)],
            }
            for c in children
        ],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _relativize(node, parent_abs=None):
    logical = dict(node.get("box") or {})
    painted = _paint_box(node)
    node.setdefault("meta", {})["absolute_box"] = logical
    if parent_abs:
        node["box"] = {
            **logical,
            "x": painted.get("x", logical.get("x", 0)) - parent_abs.get("x", 0),
            "y": painted.get("y", logical.get("y", 0)) - parent_abs.get("y", 0),
        }
        visible = node.get("visible_box")
        if visible:
            node["visible_box"] = {
                **visible,
                "x": visible.get("x", 0) - parent_abs.get("x", 0),
                "y": visible.get("y", 0) - parent_abs.get("y", 0),
            }
    for child in node.get("children") or []:
        _relativize(child, logical)


def _union(boxes):
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _text_alignment(a, b):
    """Strong alignment test for a real text stack, not loose nearby copy.

    A shared left edge or shared centre is the primary evidence.  The positional
    overlap fallback is measured against the *wider* line so a narrow element that
    merely sits within a much wider headline's horizontal span (e.g. a mid-canvas
    CTA under a full-bleed title) is not mistaken for the same column.
    """
    ax, aw = a["x"], max(1.0, a["w"])
    bx, bw = b["x"], max(1.0, b["w"])
    left = abs(ax - bx) <= max(4.0, min(aw, bw) * 0.12)
    center = abs((ax + aw / 2) - (bx + bw / 2)) <= max(5.0, min(aw, bw) * 0.10)
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) / max(aw, bw)
    return left or center or overlap >= 0.72


def _node_z(node):
    raw = node.get("z_index", node.get("z"))
    target = node.get("target")
    meta = node.get("meta") or {}
    # Match reconstruction's ownership contract: a VLM/SAM layer band is more
    # trustworthy than the common placeholder z=0, but never replaces a real
    # upstream paint order.
    if raw in (None, 0, "0", "0.0"):
        band = str(meta.get("z_band") or "").lower()
        band_z = {
            "background": -1_000_000.0, "plate": -1_000_000.0,
            "content": 20.0, "scene": 20.0, "foreground": 30.0,
            "overlay": 40.0, "chrome": 50.0, "ui": 50.0,
        }.get(band)
        if band_z is not None:
            return band_z
    # Fusion assigns OCR z=1 to distinguish shell vs label — not final paint order.
    if target == "text" and raw in (None, 0, 1, "0", "0.0", "1", "1.0"):
        return 40.0
    if raw not in (None, 0, "0", "0.0"):
        return float(raw)
    role = str(meta.get("role") or node.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"}:
        return -1_000_000.0
    return {"text": 40.0, "icon": 35.0, "image": 25.0}.get(target, 20.0)


def _semantic_text_stacks(roots):
    """Group only clearly contiguous text hierarchy into a vertical Figma frame.

    OCR already emits paragraph blocks. This handles the common separate headline/subhead/body
    stack without inventing a group for every unrelated sentence on the canvas.
    """
    # Any real copy line can participate in a vertical paragraph flow; the strict
    # alignment + gap gate below (not the semantic role) is what decides membership.
    # Only brand marks are held out so a wordmark never merges into body copy.
    texts = [node for node in roots if node.get("target") == "text"
             and str((node.get("meta") or {}).get("role", "text")).lower()
             not in _NON_FLOW_TEXT_ROLES]
    texts.sort(key=lambda node: (node.get("box", {}).get("y", 0), node.get("id", "")))
    groups, current = [], []
    for node in texts:
        box = node.get("box") or {}
        if not current:
            current = [node]
            continue
        previous = current[-1]
        prior_box = previous.get("box") or {}
        gap = box.get("y", 0) - (prior_box.get("y", 0) + prior_box.get("h", 0))
        median_h = median([max(1.0, item.get("box", {}).get("h", 1)) for item in current + [node]])
        pmeta = previous.get("meta") or {}
        nmeta = node.get("meta") or {}
        same_paragraph = any(pmeta.get(key) is not None and pmeta.get(key) == nmeta.get(key)
                             for key in ("paragraph_id", "block_id", "text_block_id"))
        if (same_paragraph or (0 <= gap <= max(14.0, median_h * 1.75)
                               and _text_alignment(prior_box, box))):
            current.append(node)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [node]
    if len(current) >= 2:
        groups.append(current)

    if not groups:
        return roots
    members = {node.get("id") for group in groups for node in group}
    out = [node for node in roots if node.get("id") not in members]
    for index, group in enumerate(groups):
        box = _union([node["box"] for node in group])
        group_id = "text-stack-" + hashlib.sha1(
            "|".join(str(node.get("id")) for node in group).encode()
        ).hexdigest()[:10]
        role_names = [str((node.get("meta") or {}).get("role") or "text") for node in group]
        out.append({
            "id": group_id,
            "target": "group",
            "box": box,
            "z": max(_node_z(node) for node in group),
            "children": group,
            "layout": {
                "mode": "VERTICAL", "confidence": 0.9,
                "gap": round(median([
                    max(0.0, group[i + 1]["box"]["y"] -
                        (group[i]["box"]["y"] + group[i]["box"]["h"]))
                    for i in range(len(group) - 1)
                ]), 2),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "MIN",
                "counterAlign": _counter_alignment([node["box"] for node in group], "VERTICAL"),
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            },
            "meta": {"role": "text-stack", "semantic_roles": role_names,
                     "layout_confidence": 0.9},
        })
        _annotate_stack_children(out[-1], group)
    return out


def _semantic_text_rows(roots):
    """Group evenly-spaced peer text/icon leaves on one baseline into a HORIZONTAL frame.

    Handles real horizontal bars the vertical stack pass leaves flat — stat rows,
    inline label runs, social action counts.  The gate is deliberately strict
    (shared baseline band, similar-height peers, left-to-right non-overlapping
    columns, evenly spaced) so items that merely share a ``y`` are never fused: a
    wrong row is worse than an absolute layer.  Runs after the stack pass, so any
    text already claimed by a vertical column is untouched here.
    """
    leaves = [node for node in roots
              if node.get("target") in ("text", "icon")
              and not node.get("children")
              and str((node.get("meta") or {}).get("role", "")).lower() not in _NON_FLOW_TEXT_ROLES
              # Rating stars belong to rating-strip, not a generic icon+label row.
              and str((node.get("meta") or {}).get("role", "")).lower().replace("-", "_")
              not in _RATING_STAR_ROLES]
    leaves.sort(key=lambda node: (node.get("box", {}).get("x", 0), node.get("id", "")))
    used, groups = set(), []
    for seed in leaves:
        if seed.get("id") in used:
            continue
        row = [seed]
        for node in leaves:
            if node is seed or node.get("id") in used or node in row:
                continue
            box = node.get("box") or {}
            prev = row[-1].get("box") or {}
            heights = [max(1.0, item.get("box", {}).get("h", 1)) for item in row + [node]]
            mh = median(heights)
            cy_row = median([item["box"].get("y", 0) + item["box"].get("h", 0) / 2 for item in row])
            cy = box.get("y", 0) + box.get("h", 0) / 2
            if abs(cy - cy_row) > max(4.0, mh * 0.30):       # off the shared baseline band
                continue
            if not _consistent(heights, max_cv=0.35):        # not a peer (very different size)
                continue
            mw = median([max(1.0, item.get("box", {}).get("w", 1)) for item in row + [node]])
            gap = box.get("x", 0) - (prev.get("x", 0) + prev.get("w", 0))
            # Inline row items are separated by roughly a line-height, not by their
            # own width — scaling tolerance to width would fuse far-apart display
            # fragments and side-by-side comparison columns into bogus rows.
            if gap < -0.15 * mw or gap > max(1.2 * mh, 0.5 * mw):
                continue
            row.append(node)
        if len(row) < 2:
            continue
        ordered = sorted(row, key=lambda n: (n["box"].get("x", 0), n.get("id", "")))
        gaps = [ordered[i + 1]["box"].get("x", 0)
                - (ordered[i]["box"].get("x", 0) + ordered[i]["box"].get("w", 0))
                for i in range(len(ordered) - 1)]
        mw = median([max(1.0, n["box"].get("w", 1)) for n in ordered])
        mh = median([max(1.0, n["box"].get("h", 1)) for n in ordered])
        positive = [g for g in gaps if g >= 0]
        if not _consistent(positive):                        # unevenly spaced -> not a real bar
            continue
        if positive and max(positive) > max(1.2 * mh, 0.5 * mw):  # a lone wide void -> not a row
            continue
        if not any(n.get("target") == "text" for n in ordered):  # need a label, not loose icons
            continue
        # A bare two-item text+text pair is weak evidence (adjacent display fragments
        # read as a row). Require either an icon (a labelled stat/action) or a genuine
        # three-plus-item bar before committing to a horizontal frame.
        if len(ordered) == 2 and not any(n.get("target") == "icon" for n in ordered):
            continue
        groups.append(ordered)
        used.update(n.get("id") for n in ordered)

    if not groups:
        return roots
    members = {node.get("id") for group in groups for node in group}
    out = [node for node in roots if node.get("id") not in members]
    for group in groups:
        boxes = [node["box"] for node in group]
        box = _union(boxes)
        gaps = [group[i + 1]["box"]["x"] - (group[i]["box"]["x"] + group[i]["box"]["w"])
                for i in range(len(group) - 1)]
        row_boxes = [_paint_box(node) for node in group]
        layout = _emit_figma_layout_aliases({
            "mode": "HORIZONTAL", "confidence": 0.88,
            "gap": _item_spacing([g for g in gaps if g >= 0]),
            "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "align": "MIN", "counterAlign": _counter_alignment(row_boxes, "HORIZONTAL"),
            "primarySizing": "FIXED", "counterSizing": "FIXED",
        })
        row_id = "text-row-" + hashlib.sha1(
            "|".join(str(node.get("id")) for node in group).encode()
        ).hexdigest()[:10]
        is_checklist = _looks_like_checklist_row(group)
        out.append({
            "id": row_id,
            "target": "group",
            "box": box,
            "z": max(_node_z(node) for node in group),
            "children": group,
            "layout": layout,
            "meta": {
                # Keep role=text-row for geometry consumers; Checklist is the local name.
                "role": "text-row",
                "layout_confidence": 0.88,
                **({"semantic_name": "Checklist", "checklist": True} if is_checklist else {}),
            },
        })
        _annotate_stack_children(out[-1], group)
    return out


def _semantic_header_clusters(roots, canvas, cfg=None):
    """Group avatar + identity copy + follow control into one social header frame.

    Consumes ``layout.scene_grouping.header_cluster`` (set by the social_screenshot
    archetype preset).  Requires an avatar-role image in the top band plus at least
    one nearby identity text or follow button on a shared baseline — weak proximity
    alone never creates a cluster.
    """
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    if not grouping.get("header_cluster") and archetype != "social_screenshot":
        return roots
    canvas_h = max(1.0, float((canvas or {}).get("h") or 1))
    avatars = []
    for node in roots:
        role = str((node.get("meta") or {}).get("role") or "").lower()
        if node.get("target") not in {"image", "icon"}:
            continue
        if role not in _AVATAR_ROLES and not (node.get("meta") or {}).get("avatar"):
            continue
        box = node.get("box") or {}
        cy = float(box.get("y") or 0) + float(box.get("h") or 0) / 2
        if cy > canvas_h * 0.28:
            continue
        avatars.append(node)
    if not avatars:
        return roots

    used = set()
    wrappers = []
    for avatar in avatars:
        if avatar.get("id") in used:
            continue
        abox = avatar.get("box") or {}
        aw = max(1.0, float(abox.get("w") or 1))
        ah = max(1.0, float(abox.get("h") or 1))
        acy = float(abox.get("y") or 0) + ah / 2
        ax1 = float(abox.get("x") or 0) + aw
        members = [avatar]
        for node in roots:
            if node is avatar or node.get("id") in used:
                continue
            box = node.get("box") or {}
            role = str((node.get("meta") or {}).get("role") or "").lower()
            target = node.get("target")
            cy = float(box.get("y") or 0) + float(box.get("h") or 0) / 2
            if abs(cy - acy) > max(ah * 0.85, 28.0):
                continue
            # Identity sits to the right of the avatar (LTR social chrome); follow
            # pills may sit further right. Reject anything left of the avatar.
            if float(box.get("x") or 0) + float(box.get("w") or 0) < float(abox.get("x") or 0):
                continue
            if float(box.get("x") or 0) > ax1 + max(4.5 * aw, 220.0):
                continue
            is_identity_text = (target == "text" and (
                role in _HEADER_IDENTITY_ROLES or role in {"", "text"}
                or (node.get("meta") or {}).get("social_identity")))
            is_follow = (
                (target == "group" and role in {"button", "badge", "chip"})
                or (target == "shape" and role in {"button", "badge", "chip"})
                or (target == "text" and role in {"cta", "button"})
            )
            is_badge = target in {"icon", "image"} and role in {
                "verified", "badge", "icon",
            }
            if not (is_identity_text or is_follow or is_badge):
                continue
            # Avoid pulling a large photo/card into the header.
            if _area(box) > _area(abox) * 6.0 and target == "image":
                continue
            members.append(node)
        texts = [m for m in members if m.get("target") == "text"]
        controls = [m for m in members if (m.get("meta") or {}).get("role") in {
            "button", "badge", "chip", "cta",
        } or m.get("target") == "group"]
        if len(members) < 2 or (not texts and not controls):
            continue
        # Nest stacked identity copy (name over handle) into a vertical sub-frame so
        # the outer header can be a clean HORIZONTAL [avatar | identity | follow].
        identity_texts = [
            m for m in members
            if m.get("target") == "text"
            and str((m.get("meta") or {}).get("role") or "").lower()
            in (_HEADER_IDENTITY_ROLES | {"", "text"})
        ]
        if len(identity_texts) >= 2:
            id_boxes = [n.get("box") or {} for n in identity_texts]
            cx = [b.get("x", 0) + b.get("w", 0) / 2 for b in id_boxes]
            mw = max(1.0, median(b.get("w", 1) for b in id_boxes))
            if (max(cx) - min(cx)) / mw <= 0.55:
                ordered_id = sorted(
                    identity_texts,
                    key=lambda n: ((n.get("box") or {}).get("y", 0), n.get("id", "")),
                )
                id_box = _union([n.get("box") or {} for n in ordered_id])
                id_gaps = [
                    ordered_id[i + 1]["box"]["y"]
                    - (ordered_id[i]["box"]["y"] + ordered_id[i]["box"]["h"])
                    for i in range(len(ordered_id) - 1)
                ]
                id_layout = _emit_figma_layout_aliases({
                    "mode": "VERTICAL", "confidence": 0.9,
                    "gap": _item_spacing([g for g in id_gaps if g >= 0]),
                    "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "align": "MIN",
                    "counterAlign": _counter_alignment(
                        [n.get("box") or {} for n in ordered_id], "VERTICAL"),
                    "primarySizing": "HUG", "counterSizing": "HUG",
                })
                identity_group = {
                    "id": "header-identity-" + hashlib.sha1(
                        "|".join(str(n.get("id")) for n in ordered_id).encode()
                    ).hexdigest()[:10],
                    "target": "group",
                    "name": "Identity",
                    "box": id_box,
                    "z": max(_node_z(n) for n in ordered_id),
                    "children": ordered_id,
                    "layout": id_layout,
                    "meta": {"role": "header-identity", "layout_confidence": 0.9},
                }
                _annotate_stack_children(identity_group, ordered_id)
                id_ids = {n.get("id") for n in ordered_id}
                members = [m for m in members if m.get("id") not in id_ids] + [identity_group]
        ordered = sorted(
            members,
            key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in ordered]
        box = _union(boxes)
        layout = infer_auto_layout(
            {"box": box, "meta": {"role": "header-cluster"}}, ordered,
        )
        # Header chrome is almost always a horizontal bar; force HORIZONTAL when
        # row evidence is present even if infer stayed NONE on mild overlaps.
        if layout.get("mode") == "NONE":
            cy_vals = [b.get("y", 0) + b.get("h", 0) / 2 for b in boxes]
            mh = max(1.0, median(max(1.0, b.get("h", 1)) for b in boxes))
            if (max(cy_vals) - min(cy_vals)) / mh <= 0.70:
                gaps = [
                    boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
                    for i in range(len(boxes) - 1)
                ]
                layout = _emit_figma_layout_aliases({
                    "mode": "HORIZONTAL",
                    "confidence": 0.86,
                    "gap": _item_spacing([g for g in gaps if g >= 0]),
                    "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "align": "MIN",
                    "counterAlign": _counter_alignment(boxes, "HORIZONTAL"),
                    "primarySizing": "FIXED",
                    "counterSizing": "FIXED",
                })
        cluster_id = "header-cluster-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in ordered).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": cluster_id,
            "target": "group",
            "name": "Social header",
            "box": box,
            "z": max(_node_z(n) for n in ordered),
            "children": ordered,
            "layout": layout,
            "meta": {
                "role": "header-cluster",
                "semantic_name": "Social header",
                "layout_confidence": layout.get("confidence"),
                "avatar_id": avatar.get("id"),
            },
        })
        _annotate_stack_children(wrappers[-1], ordered)
        for n in ordered:
            if (n.get("meta") or {}).get("role") == "header-identity":
                used.update(c.get("id") for c in (n.get("children") or []))
                used.add(n.get("id"))
            else:
                used.add(n.get("id"))

    if not wrappers:
        return roots
    out = [node for node in roots if node.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _resolve_parent_id(parent_id, by_id: dict):
    """Resolve fusion/merge parent aliases onto a live candidate id.

    Merge prefixes element ids as ``c_<id>`` while fusion may still record the raw
    ``E010`` form. Accept either spelling so nested chrome stays under its owner.
    """
    if not parent_id:
        return None
    if parent_id in by_id:
        return parent_id
    text = str(parent_id)
    alt = text if text.startswith("c_") else f"c_{text}"
    if alt in by_id:
        return alt
    if text.startswith("c_") and text[2:] in by_id:
        return text[2:]
    return None


def _semantic_asset_groups(roots):
    """Keep an explicit asset owner and its overlays together in the Figma tree.

    Element fusion already records ``parent_id`` when, for example, an avatar owns an
    online badge or a screenshot/card owns its UI chrome.  Geometry-only grouping used
    to discard that evidence unless the owner happened to be a flat shape.  That produced
    a flat layer list where a designer could not select a whole swappable photo/card.

    Only image/vector-like owners are wrapped here: native shape containers are handled
    by the card/button pass below.  We also require meaningful spatial overlap so a stale
    parent hint cannot accidentally pull a distant caption into an asset group.
    """
    by_id = {node.get("id"): node for node in roots if node.get("id")}
    children_by_parent = {}
    for node in roots:
        parent_id = _resolve_parent_id((node.get("meta") or {}).get("parent_id"), by_id)
        if parent_id and parent_id != node.get("id"):
            # Normalize so downstream relativization sees a live id.
            node.setdefault("meta", {})["parent_id"] = parent_id
            children_by_parent.setdefault(parent_id, []).append(node)

    consumed = set()
    wrappers = []
    for parent_id, children in children_by_parent.items():
        owner = by_id[parent_id]
        if owner.get("target") not in {"image", "icon"}:
            continue
        owner_box = owner.get("box") or {}
        accepted = []
        for child in children:
            child_box = child.get("box") or {}
            confidence = float((child.get("meta") or {}).get("parent_confidence", 0) or 0)
            if _inside(child_box, owner_box) >= .55 or confidence >= .85:
                accepted.append(child)
        if not accepted:
            continue
        # A semantic owner may itself already be nested in a native card frame.  Do
        # not manufacture a second root around it; the existing frame is the correct
        # designer-facing group in that case.
        if owner.get("id") in consumed or any(child.get("id") in consumed for child in accepted):
            continue
        role = str((owner.get("meta") or {}).get("role") or owner.get("target") or "asset")
        label = ((owner.get("meta") or {}).get("semantic_name") or
                 (owner.get("meta") or {}).get("label") or role.replace("-", " ").title())
        wrappers.append({
            "id": f"asset-group-{owner.get('id')}",
            "target": "group",
            "name": label,
            "box": dict(owner_box),
            "z": min([_node_z(owner)] + [_node_z(child) for child in accepted]),
            "children": [owner] + sorted(accepted, key=lambda node: (_node_z(node), node.get("id", ""))),
            "layout": {"mode": "NONE", "confidence": 1.0},
            "meta": {
                "role": "asset-group",
                "semantic_name": label,
                "semantic_owner": owner.get("id"),
                "semantic_label": label,
                "layout_confidence": 1.0,
            },
        })
        consumed.add(owner.get("id"))
        consumed.update(child.get("id") for child in accepted)

    if not wrappers:
        return roots
    out = [node for node in roots if node.get("id") not in consumed]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _subtree_ids(node) -> set:
    out = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.get("id"):
            out.add(current.get("id"))
        stack.extend(current.get("children") or [])
    return out


def dissolve_orphaned_asset_shells(roots: list) -> list:
    """Dissolve asset-group shells whose semantic owner escaped to the root level.

    The raster-slice fallback (and harness repair rounds) hoist a failed owner image
    out of its asset-group as a root-level absolute slice. The leftover shell then
    keeps only the OVERLAYS (badges, chips) at the group's low z while the hoisted
    owner and any later band crop paint over them: 013's product shell held the
    "snacks" badge at group z=5 under the root product slice (z=5, later id) and the
    photo band crop (z=10), blanking the badge's native text in preview AND Figma
    (placement ink IoU 0.0). A shell without its asset is not a semantic group — and
    the per-group background pass would cut a clean-plate slice for it that erases
    the very product it pretends to host. Promote the overlays back to root with
    absolute coordinates and their own (higher) z instead.

    Runs on the FINAL tree (both layout.infer and the structure-first hydrate path),
    so children arrive parent-relative and are re-absolutized here.
    """
    if not isinstance(roots, list):
        return roots
    all_ids = set()
    for root in roots:
        all_ids |= _subtree_ids(root)
    out = []
    changed = False
    for root in roots:
        meta = root.get("meta") or {}
        owner_id = meta.get("semantic_owner")
        if (
            root.get("target") != "group"
            or str(meta.get("role") or "") != "asset-group"
            or not owner_id
            or root.get("fill")
            or root.get("src")
        ):
            out.append(root)
            continue
        subtree = _subtree_ids(root)
        if owner_id in subtree or owner_id not in all_ids:
            out.append(root)  # intact group, or owner truly gone (materialize path)
            continue
        gbox = root.get("box") or {}
        gx, gy = float(gbox.get("x", 0) or 0), float(gbox.get("y", 0) or 0)
        for child in root.get("children") or []:
            cbox = dict(child.get("box") or {})
            cbox["x"] = float(cbox.get("x", 0) or 0) + gx
            cbox["y"] = float(cbox.get("y", 0) or 0) + gy
            child["box"] = cbox
            visible = child.get("visible_box")
            if isinstance(visible, dict):
                visible = dict(visible)
                visible["x"] = float(visible.get("x", 0) or 0) + gx
                visible["y"] = float(visible.get("y", 0) or 0) + gy
                child["visible_box"] = visible
            cmeta = child.setdefault("meta", {})
            cmeta["absolute_box"] = dict(cbox)
            cmeta["promoted_from_orphan_shell"] = root.get("id")
            out.append(child)
        changed = True
    if changed:
        out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _unwrap_passthrough_bands(nodes):
    """Drop single-child NONE bands that only inflate node count.

    XY-cut banding is valuable when it holds a real cluster. A band whose only child
    is already a frame/leaf and that carries no Auto Layout of its own is pure
    nesting overhead — unwrap it (Codia-like minimalism for dense scenes).
    """
    out = []
    for node in nodes:
        children = list(node.get("children") or [])
        if children:
            node = dict(node)
            node["children"] = _unwrap_passthrough_bands(children)
            children = node["children"]
        meta = node.get("meta") or {}
        layout = node.get("layout") or {}
        if (
            node.get("target") == "group"
            and meta.get("role") == "band"
            and layout.get("mode", "NONE") in (None, "NONE")
            and len(children) == 1
        ):
            out.append(children[0])
            continue
        out.append(node)
    return out


def _looks_like_caption_plate_group(node):
    """Root group that already hugs a single centered text line on a painted shell."""
    if node.get("target") != "group":
        return False
    children = node.get("children") or []
    if len(children) != 1 or children[0].get("target") != "text":
        return False
    if not _has_surface(node):
        return False
    role = str((node.get("meta") or {}).get("role") or "")
    if role not in (
        "caption-plate", "button", "card", "chip", "badge",
        "shape", "container", "plate", "panel", "",
    ):
        return False
    lay = node.get("layout") or {}
    if lay.get("mode") not in ("HORIZONTAL", "VERTICAL"):
        return False
    if lay.get("primarySizing") != "HUG" or lay.get("counterSizing") != "HUG":
        return False
    if lay.get("counterAlign") == "CENTER" or lay.get("align") == "CENTER":
        return True
    return _text_align_token(children[0]) == "CENTER"


def _stack_caption_plates(roots, lcfg):
    """Wrap sibling caption-paragraph plates into an IG-Caption VERTICAL HUG frame.

    Driven by archetype ``pair_text_with_backplate``. Gap is the measured median
    between plates; counterAlign CENTER keeps a narrower second pill centered.
    """
    if not _pair_text_with_backplate_enabled(lcfg):
        return roots
    plates = [node for node in roots if _looks_like_caption_plate_group(node)]
    if len(plates) < 2:
        return roots
    plates.sort(key=lambda node: ((node.get("box") or {}).get("y", 0), node.get("id", "")))

    clusters, current = [], [plates[0]]
    for node in plates[1:]:
        prev = current[-1]
        pb = prev.get("box") or {}
        nb = node.get("box") or {}
        gap = nb.get("y", 0) - (pb.get("y", 0) + pb.get("h", 0))
        heights = [max(1.0, (item.get("box") or {}).get("h", 1)) for item in current + [node]]
        mh = median(heights)
        if gap < 0 or gap > max(80.0, mh * 2.5):
            if len(current) >= 2:
                clusters.append(current)
            current = [node]
            continue
        pcx = pb.get("x", 0) + pb.get("w", 0) / 2
        ncx = nb.get("x", 0) + nb.get("w", 0) / 2
        if abs(pcx - ncx) > max(8.0, min(pb.get("w", 1), nb.get("w", 1)) * 0.12):
            if len(current) >= 2:
                clusters.append(current)
            current = [node]
            continue
        if abs(pb.get("h", 0) - nb.get("h", 0)) > max(12.0, mh * 0.35):
            if len(current) >= 2:
                clusters.append(current)
            current = [node]
            continue
        current.append(node)
    if len(current) >= 2:
        clusters.append(current)
    if not clusters:
        return roots

    consumed = {node.get("id") for cluster in clusters for node in cluster}
    out = [node for node in roots if node.get("id") not in consumed]
    for cluster in clusters:
        boxes = [node["box"] for node in cluster]
        box = _union(boxes)
        gaps = [
            cluster[i + 1]["box"]["y"] - (cluster[i]["box"]["y"] + cluster[i]["box"]["h"])
            for i in range(len(cluster) - 1)
        ]
        stack_id = "caption-stack-" + hashlib.sha1(
            "|".join(str(node.get("id")) for node in cluster).encode()
        ).hexdigest()[:10]
        wrapper = {
            "id": stack_id,
            "target": "group",
            "name": "IG Caption",
            "box": box,
            "z": max(_node_z(node) for node in cluster),
            "children": list(cluster),
            "layout": _emit_figma_layout_aliases({
                "mode": "VERTICAL",
                "confidence": 0.93,
                "gap": _item_spacing(gaps),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "MIN",
                "counterAlign": "CENTER",
                "primarySizing": "HUG",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": "caption-stack",
                "layout_confidence": 0.93,
                "semantic_name": "IG Caption",
                "pair_text_with_backplate": True,
            },
        }
        out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _looks_like_stat_pill_group(node):
    """Root group that hugs a left-biased text line on a painted left-column plate."""
    if node.get("target") != "group":
        return False
    children = node.get("children") or []
    if len(children) != 1 or children[0].get("target") != "text":
        return False
    if not _has_surface(node):
        return False
    role = str((node.get("meta") or {}).get("role") or "")
    if role in {"stat-pill", "callout", "benefit", "pill"}:
        return True
    if role not in (
        "caption-plate", "button", "card", "chip", "badge", "pill",
        "shape", "container", "plate", "panel", "",
    ):
        return False
    lay = node.get("layout") or {}
    if lay.get("mode") not in ("HORIZONTAL", "VERTICAL"):
        return False
    if lay.get("primarySizing") != "HUG" or lay.get("counterSizing") != "HUG":
        return False
    if lay.get("align") == "CENTER" or lay.get("counterAlign") == "CENTER":
        return False
    return _text_align_token(children[0]) in {"LEFT", "MIN", ""}


def _stack_stat_pills(roots, lcfg, canvas=None):
    """Wrap sibling left-column stat/benefit pills into a VERTICAL HUG Stats/Benefits frame."""
    if not _pair_text_with_backplate_enabled(lcfg):
        return roots
    canvas_w = float((canvas or {}).get("w") or 0)
    plates = []
    for node in roots:
        if not _looks_like_stat_pill_group(node):
            continue
        box = node.get("box") or {}
        if canvas_w > 0:
            cx = float(box.get("x", 0)) + float(box.get("w", 0)) / 2
            if cx > canvas_w * 0.42:
                continue
        plates.append(node)
    if len(plates) < 2:
        return roots
    plates.sort(key=lambda node: ((node.get("box") or {}).get("y", 0), node.get("id", "")))

    clusters, current = [], [plates[0]]
    for node in plates[1:]:
        prev = current[-1]
        pb = prev.get("box") or {}
        nb = node.get("box") or {}
        gap = nb.get("y", 0) - (pb.get("y", 0) + pb.get("h", 0))
        heights = [max(1.0, (item.get("box") or {}).get("h", 1)) for item in current + [node]]
        mh = median(heights)
        if gap < 0 or gap > max(64.0, mh * 2.2):
            if len(current) >= 2:
                clusters.append(current)
            current = [node]
            continue
        # Shared left edge for a column of pills.
        if abs(pb.get("x", 0) - nb.get("x", 0)) > max(10.0, min(pb.get("w", 1), nb.get("w", 1)) * 0.18):
            if len(current) >= 2:
                clusters.append(current)
            current = [node]
            continue
        current.append(node)
    if len(current) >= 2:
        clusters.append(current)
    if not clusters:
        return roots

    consumed = {node.get("id") for cluster in clusters for node in cluster}
    out = [node for node in roots if node.get("id") not in consumed]
    for cluster in clusters:
        boxes = [node["box"] for node in cluster]
        box = _union(boxes)
        gaps = [
            cluster[i + 1]["box"]["y"] - (cluster[i]["box"]["y"] + cluster[i]["box"]["h"])
            for i in range(len(cluster) - 1)
        ]
        is_benefit = any(
            str((node.get("meta") or {}).get("role") or "") in {"callout", "benefit"}
            for node in cluster
        )
        stack_role = "benefit-stack" if is_benefit else "stat-stack"
        stack_name = "Benefits" if is_benefit else "Stats"
        stack_id = f"{stack_role}-" + hashlib.sha1(
            "|".join(str(node.get("id")) for node in cluster).encode()
        ).hexdigest()[:10]
        wrapper = {
            "id": stack_id,
            "target": "group",
            "name": stack_name,
            "box": box,
            "z": max(_node_z(node) for node in cluster),
            "children": list(cluster),
            "layout": _emit_figma_layout_aliases({
                "mode": "VERTICAL",
                "confidence": 0.91,
                "gap": _item_spacing(gaps),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "MIN",
                "counterAlign": "MIN",
                "primarySizing": "HUG",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": stack_role,
                "layout_confidence": 0.91,
                "semantic_name": stack_name,
                "pair_text_with_backplate": True,
            },
        }
        _annotate_stack_children(wrapper, cluster)
        out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _is_vs_chip(node) -> bool:
    """True for a small VS / versus badge or label between comparison columns."""
    meta = node.get("meta") or {}
    role = str(meta.get("role") or "").lower().replace("-", "_")
    text = str(node.get("text") or meta.get("text") or meta.get("shell_text_snippet") or "").strip()
    if _VS_TEXT_RE.match(text):
        return True
    if role in {"vs", "versus", "vs_chip", "vs_badge"}:
        return True
    if node.get("target") == "group":
        kids = node.get("children") or []
        if any(_is_vs_chip(k) for k in kids):
            return True
    return False


def _comparison_photo_nodes(roots):
    """Side photos eligible for before/after nesting (explicit columns or pair meta)."""
    photos = []
    for node in roots:
        meta = node.get("meta") or {}
        role = str(meta.get("role") or "").lower().replace("_", "-")
        if node.get("target") not in {"image", "group"}:
            continue
        if role in {"comparison-column", "comparison-panel", "photo-panel", "image-panel"}:
            photos.append(node)
            continue
        if meta.get("comparison_side") or meta.get("before_after_side"):
            photos.append(node)
            continue
        if role in {"photo", "image", "photo-card", "product", "product-cluster",
                    "pill-cloud", "hand", "canister", "sachet", "person"} and meta.get("comparison_group_id"):
            photos.append(node)
    return photos


def _nest_comparison_with_vs(roots, cfg=None):
    """Nest two comparison photos + a middle VS chip into one HORIZONTAL frame.

    Fires for ``comparison_grid`` (or explicit comparison-column roles) when geometry
    shows left/right photos with a compact VS badge between them — MONTE/Huel/Biomel.
    """
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    facts = ((cfg or {}).get("scene") or {}).get("facts") or {}
    grouping = _scene_grouping(cfg)
    allow = (
        archetype == "comparison_grid"
        or grouping.get("preserve_columns")
        or facts.get("before_after_pair")
        or facts.get("before_after_labels")
        or facts.get("stage_progression")
    )
    photos = _comparison_photo_nodes(roots)
    if not allow and len(photos) < 2:
        return roots
    if len(photos) < 2:
        # Fall back: any two similarly-sized side-by-side photo images under comparison.
        if archetype != "comparison_grid" and not facts.get("before_after_pair"):
            return roots
        photos = [
            n for n in roots
            if n.get("target") == "image"
            and str((n.get("meta") or {}).get("role") or "").lower()
            in {"photo", "image", "photo-card", "product", "comparison-column", ""}
        ]
    if len(photos) < 2:
        return roots

    photos = sorted(
        photos,
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    # Prefer the leftmost + rightmost of a 2-photo pair; ignore extras.
    left, right = photos[0], photos[-1]
    if left is right or left.get("id") == right.get("id"):
        return roots
    lb, rb = left.get("box") or {}, right.get("box") or {}
    # Shared baseline / similar height.
    lh, rh = max(1.0, float(lb.get("h") or 1)), max(1.0, float(rb.get("h") or 1))
    if abs(lh - rh) / max(lh, rh) > 0.35:
        return roots
    lcy = float(lb.get("y", 0)) + lh / 2
    rcy = float(rb.get("y", 0)) + rh / 2
    if abs(lcy - rcy) > max(lh, rh) * 0.35:
        return roots
    gap_left = float(rb.get("x", 0)) - (float(lb.get("x", 0)) + float(lb.get("w", 0)))
    if gap_left < -min(lb.get("w", 1), rb.get("w", 1)) * 0.15:
        return roots

    vs_nodes = []
    for node in roots:
        if node is left or node is right or node.get("id") in {left.get("id"), right.get("id")}:
            continue
        if not _is_vs_chip(node):
            # Also accept a small badge/shape hosting VS text as shell.
            meta = node.get("meta") or {}
            role = str(meta.get("role") or "").lower()
            if node.get("target") not in {"shape", "badge", "icon", "group", "text"}:
                continue
            if role not in {"badge", "chip", "button", "seal", "shape", "vs", ""}:
                if not meta.get("text_bearing_shell"):
                    continue
            snippet = str(
                node.get("text") or meta.get("shell_text_snippet") or meta.get("text") or ""
            ).strip()
            if not _VS_TEXT_RE.match(snippet):
                continue
        box = node.get("box") or {}
        cx = float(box.get("x", 0)) + float(box.get("w", 0)) / 2
        left_edge = float(lb.get("x", 0)) + float(lb.get("w", 0)) * 0.35
        right_edge = float(rb.get("x", 0)) + float(rb.get("w", 0)) * 0.65
        if not (left_edge <= cx <= right_edge):
            continue
        # VS sits in the horizontal gap (or slightly overlapping the seam).
        cy = float(box.get("y", 0)) + float(box.get("h", 0)) / 2
        if abs(cy - (lcy + rcy) / 2) > max(lh, rh) * 0.55:
            continue
        vs_nodes.append(node)
    if not vs_nodes:
        return roots
    vs = min(
        vs_nodes,
        key=lambda n: abs(
            (float((n.get("box") or {}).get("x", 0)) + float((n.get("box") or {}).get("w", 0)) / 2)
            - (float(lb.get("x", 0)) + float(lb.get("w", 0)) + float(rb.get("x", 0))) / 2
        ),
    )

    members = sorted(
        [left, vs, right],
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    boxes = [n.get("box") or {} for n in members]
    box = _union(boxes)
    gaps = [
        boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
        for i in range(len(boxes) - 1)
    ]
    layout = _emit_figma_layout_aliases({
        "mode": "HORIZONTAL",
        "confidence": 0.9,
        "gap": _item_spacing([g for g in gaps if g >= 0]),
        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "align": "CENTER",
        "counterAlign": "CENTER",
        "primarySizing": "HUG",
        "counterSizing": "HUG",
    })
    wrap_id = "comparison-set-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in members).encode()
    ).hexdigest()[:10]
    # Designer-facing local names: Before / After photos + VS chip.
    left_meta = left.setdefault("meta", {})
    right_meta = right.setdefault("meta", {})
    vs_meta = vs.setdefault("meta", {})
    left_side = str(left_meta.get("comparison_side") or left_meta.get("before_after_side") or "before")
    right_side = str(right_meta.get("comparison_side") or right_meta.get("before_after_side") or "after")
    left_meta.setdefault(
        "semantic_name",
        "Photo / Before" if left_side.lower() in {"before", "without"} else f"Photo / {left_side.title()}",
    )
    right_meta.setdefault(
        "semantic_name",
        "Photo / After" if right_side.lower() in {"after", "with"} else f"Photo / {right_side.title()}",
    )
    left_meta.setdefault("comparison_side", "before" if left_side.lower() in {"before", "without"} else left_side)
    right_meta.setdefault("comparison_side", "after" if right_side.lower() in {"after", "with"} else right_side)
    vs_meta.setdefault("semantic_name", "VS")
    if vs.get("target") == "text":
        vs_meta.setdefault("role", "vs")
    wrapper = {
        "id": wrap_id,
        "target": "group",
        "name": "Comparison",
        "box": box,
        "z": max(_node_z(n) for n in members),
        "children": members,
        "layout": layout,
        "meta": {
            "role": "comparison-set",
            "semantic_name": "Comparison",
            "layout_confidence": 0.9,
            "vs_id": vs.get("id"),
        },
    }
    _annotate_stack_children(wrapper, members)
    consumed = {n.get("id") for n in members}
    out = [n for n in roots if n.get("id") not in consumed]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


_STAGE_LABEL_RE = re.compile(
    r"^\s*(before|ritual|reset|after|struggle|answer|problem|solution)\s*$",
    re.I,
)


def _is_stage_progression_label(node) -> bool:
    if node.get("target") != "text":
        return False
    meta = node.get("meta") or {}
    side = str(meta.get("before_after_side") or meta.get("comparison_side") or "").lower()
    if side in {"before", "mid", "after"} or meta.get("stage_index") is not None:
        return True
    text = str(node.get("text") or "").strip()
    return bool(_STAGE_LABEL_RE.match(text))


def _nest_stage_progression(roots, cfg=None):
    """IM8 BEFORE / RITUAL / RESET labels (+ optional progress marks) → HORIZONTAL strip.

    Body-morph photography stays an intentional raster elsewhere; this only groups the
    editable stage labels / progress-bar copy so they remain a coherent strip.
    """
    facts = ((cfg or {}).get("scene") or {}).get("facts") or {}
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    labels = [n for n in roots if _is_stage_progression_label(n)]
    if len(labels) < 3:
        return roots
    allow = (
        facts.get("stage_progression")
        or facts.get("before_after_labels")
        or archetype == "comparison_grid"
    )
    if not allow:
        # Still fire when three canonical stage tokens are present (geometry-only).
        tokens = {
            re.sub(r"[^a-z]", "", str(n.get("text") or "").lower())
            for n in labels
        }
        if not ({"before", "ritual", "reset"} <= tokens or len(labels) >= 3):
            return roots

    labels = sorted(
        labels,
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    # Prefer a contiguous left→right run of 3 labels on one baseline.
    best = None
    for start in range(len(labels) - 2):
        run = [labels[start]]
        for node in labels[start + 1:]:
            prev = run[-1]
            pb, nb = prev.get("box") or {}, node.get("box") or {}
            ph = max(1.0, float(pb.get("h") or 1))
            nh = max(1.0, float(nb.get("h") or 1))
            pcy = float(pb.get("y", 0)) + ph / 2
            ncy = float(nb.get("y", 0)) + nh / 2
            if abs(pcy - ncy) > max(ph, nh) * 0.55:
                break
            gap = float(nb.get("x", 0)) - (float(pb.get("x", 0)) + float(pb.get("w", 0)))
            if gap < -8 or gap > max(ph, nh) * 4.5:
                break
            run.append(node)
            if len(run) >= 3:
                break
        if len(run) >= 3 and (best is None or len(run) > len(best)):
            best = run[:3]
    if not best:
        return roots

    # Optional thin progress-bar chrome sitting under/between the labels.
    extras = []
    strip_box = _union([n.get("box") or {} for n in best])
    for node in roots:
        if node in best or node.get("target") == "drop":
            continue
        role = str((node.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        if role not in {"progress", "progress-bar", "divider", "bar", "rule", "shape", ""}:
            continue
        if node.get("target") not in {"shape", "icon"}:
            continue
        box = node.get("box") or {}
        if float(box.get("w") or 0) <= 0:
            continue
        # Thin horizontal bar overlapping the label band.
        if float(box.get("h") or 0) > max(28.0, float(strip_box.get("h") or 1) * 0.85):
            continue
        cy = float(box.get("y", 0)) + float(box.get("h") or 0) / 2
        scy = float(strip_box.get("y", 0)) + float(strip_box.get("h") or 0) / 2
        if abs(cy - scy) > max(float(strip_box.get("h") or 1) * 1.4, 40.0):
            continue
        # Prefer bars that share x-span with the label run.
        overlap_x = max(
            0.0,
            min(
                float(box.get("x", 0)) + float(box.get("w", 0)),
                float(strip_box.get("x", 0)) + float(strip_box.get("w", 0)),
            )
            - max(float(box.get("x", 0)), float(strip_box.get("x", 0))),
        )
        if overlap_x < float(strip_box.get("w") or 1) * 0.35:
            continue
        extras.append(node)

    members = list(best) + extras
    boxes = [n.get("box") or {} for n in members]
    box = _union(boxes)
    gaps = [
        (best[i + 1].get("box") or {}).get("x", 0)
        - (
            (best[i].get("box") or {}).get("x", 0)
            + (best[i].get("box") or {}).get("w", 0)
        )
        for i in range(len(best) - 1)
    ]
    positive = [g for g in gaps if g >= 0]
    layout = _emit_figma_layout_aliases({
        "mode": "HORIZONTAL",
        "confidence": 0.88,
        "gap": _item_spacing(positive) if positive else 24,
        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "align": "CENTER",
        "counterAlign": "CENTER",
        "primarySizing": "HUG",
        "counterSizing": "HUG",
    })
    row_id = "stage-strip-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in best).encode()
    ).hexdigest()[:10]
    # Name children from stage semantics.
    for node in best:
        meta = node.setdefault("meta", {})
        text = str(node.get("text") or "").strip()
        if text:
            meta.setdefault("semantic_name", text.title())
        meta.setdefault("role", "label")
    wrapper = {
        "id": row_id,
        "target": "group",
        "name": "Progression",
        "box": box,
        "z": max(_node_z(n) for n in members),
        "children": members,
        "layout": layout,
        "meta": {
            "role": "stage-progression",
            "semantic_name": "Progression",
            "layout_confidence": 0.88,
            "stage_count": len(best),
        },
    }
    _annotate_stack_children(wrapper, members)
    consumed = {n.get("id") for n in members}
    out = [n for n in roots if n.get("id") not in consumed]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _row_stat_columns(roots, canvas=None):
    """Wrap 3+ aligned vertical text-stacks into a HORIZONTAL Stats row (MONTE 3-col).

    Requires equal column count evidence: sibling text-stack groups of similar height
    sharing a baseline band with consistent horizontal gaps. Two columns alone stay
    absolute (before/after copy pairs are not stats).
    """
    stacks = []
    for node in roots:
        meta = node.get("meta") or {}
        role = str(meta.get("role") or "")
        if node.get("target") != "group":
            continue
        if role not in {"text-stack", "stat-stack", "stat-column"}:
            continue
        kids = node.get("children") or []
        if len(kids) < 2:
            continue
        if not any(k.get("target") == "text" for k in kids):
            continue
        stacks.append(node)
    if len(stacks) < 3:
        return roots

    stacks = sorted(
        stacks,
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    # Greedy: take a contiguous run of 3+ peers on one baseline.
    best = None
    for start in range(len(stacks) - 2):
        run = [stacks[start]]
        for node in stacks[start + 1:]:
            prev = run[-1]
            pb, nb = prev.get("box") or {}, node.get("box") or {}
            ph = max(1.0, float(pb.get("h") or 1))
            nh = max(1.0, float(nb.get("h") or 1))
            if abs(ph - nh) / max(ph, nh) > 0.40:
                break
            pcy = float(pb.get("y", 0)) + ph / 2
            ncy = float(nb.get("y", 0)) + nh / 2
            if abs(pcy - ncy) > max(ph, nh) * 0.35:
                break
            gap = float(nb.get("x", 0)) - (float(pb.get("x", 0)) + float(pb.get("w", 0)))
            mw = median([max(1.0, float((s.get("box") or {}).get("w") or 1)) for s in run + [node]])
            if gap < -0.1 * mw or gap > max(1.8 * max(ph, nh), 1.2 * mw):
                break
            run.append(node)
        if len(run) >= 3 and (best is None or len(run) > len(best)):
            best = run
    if not best:
        return roots

    boxes = [n.get("box") or {} for n in best]
    gaps = [
        boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
        for i in range(len(boxes) - 1)
    ]
    positive = [g for g in gaps if g >= 0]
    if positive and not _consistent(positive, max_cv=0.45):
        return roots
    box = _union(boxes)
    layout = _emit_figma_layout_aliases({
        "mode": "HORIZONTAL",
        "confidence": 0.9,
        "gap": _item_spacing(positive),
        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "align": "MIN",
        "counterAlign": "MIN",
        "primarySizing": "HUG",
        "counterSizing": "HUG",
    })
    row_id = "stat-row-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in best).encode()
    ).hexdigest()[:10]
    wrapper = {
        "id": row_id,
        "target": "group",
        "name": "Stats",
        "box": box,
        "z": max(_node_z(n) for n in best),
        "children": list(best),
        "layout": layout,
        "meta": {
            "role": "stat-row",
            "semantic_name": "Stats",
            "layout_confidence": 0.9,
            "column_count": len(best),
        },
    }
    _annotate_stack_children(wrapper, best)
    consumed = {n.get("id") for n in best}
    out = [n for n in roots if n.get("id") not in consumed]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _fill_luma(node) -> float | None:
    """Approximate fill luminance in 0..255, or None when paint is missing/non-flat."""
    fill = node.get("fill")
    if not isinstance(fill, dict):
        style = node.get("style") or {}
        fills = style.get("fills")
        if isinstance(fills, list) and fills:
            fill = fills[0] if isinstance(fills[0], dict) else None
        elif style.get("color"):
            fill = {"kind": "flat", "color": style["color"]}
        else:
            fill = None
    if not isinstance(fill, dict):
        return None
    color = fill.get("color") or fill.get("hex")
    if not isinstance(color, str) or not color.startswith("#"):
        return None
    hex6 = color[1:7]
    if len(hex6) != 6:
        return None
    try:
        r = int(hex6[0:2], 16)
        g = int(hex6[2:4], 16)
        b = int(hex6[4:6], 16)
    except ValueError:
        return None
    return 0.299 * r + 0.587 * g + 0.114 * b


def _box_center(box) -> tuple[float, float]:
    b = box or {}
    return (
        float(b.get("x", 0)) + float(b.get("w", 0)) / 2.0,
        float(b.get("y", 0)) + float(b.get("h", 0)) / 2.0,
    )


def _ama_sticker_frames(roots, canvas, cfg=None):
    """Pair IG AMA sticker chrome: dark header bar + light body plate + question TEXT.

    Consumes ``layout.scene_grouping.ama_sticker`` (social_screenshot). Reply boxes stay
    ordinary message-bubble / caption plates; this only builds the question sticker stack.
    """
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    facts = ((cfg or {}).get("scene") or {}).get("facts") or {}
    if not grouping.get("ama_sticker") and archetype != "social_screenshot" and not facts.get("ama_sticker"):
        return roots

    headers = []
    bodies = []
    for node in roots:
        if node.get("target") not in {"shape", "group"}:
            continue
        box = node.get("box") or {}
        w = max(1.0, float(box.get("w", 1)))
        h = max(1.0, float(box.get("h", 1)))
        aspect = w / h
        role = str((node.get("meta") or {}).get("role") or "").lower().replace("-", "_")
        luma = _fill_luma(node)
        meta = node.get("meta") or {}
        if meta.get("ama_header") or role in {"ama_header", "sticker_header"}:
            headers.append(node)
            continue
        if meta.get("ama_body") or role in {"ama_body", "ama_sticker", "question_sticker"}:
            bodies.append(node)
            continue
        # Dark short bar sitting above a plate (IG AMA header).
        if luma is not None and luma <= 55.0 and aspect >= 2.4 and h <= 120:
            headers.append(node)
        elif luma is not None and luma >= 200.0 and aspect >= 1.15 and h >= 60:
            bodies.append(node)

    if not headers or not bodies:
        return roots

    used = set()
    wrappers = []
    for header in headers:
        if header.get("id") in used:
            continue
        hbox = header.get("box") or {}
        hx = float(hbox.get("x", 0))
        hy = float(hbox.get("y", 0))
        hw = max(1.0, float(hbox.get("w", 1)))
        hh = max(1.0, float(hbox.get("h", 1)))
        best = None
        best_score = None
        for body in bodies:
            if body.get("id") in used or body is header:
                continue
            bbox = body.get("box") or {}
            bx = float(bbox.get("x", 0))
            by = float(bbox.get("y", 0))
            bw = max(1.0, float(bbox.get("w", 1)))
            # Header sits on / just above the body plate with shared x span.
            if by + 4.0 < hy:
                continue
            if abs(bx - hx) > max(24.0, hw * 0.20) and abs((bx + bw / 2) - (hx + hw / 2)) > max(40.0, hw * 0.35):
                continue
            gap = by - (hy + hh)
            if gap < -hh * 0.85 or gap > max(hh * 1.5, 48.0):
                continue
            if abs(bw - hw) / max(hw, bw) > 0.35:
                continue
            score = abs(gap) + abs(bx - hx)
            if best is None or score < best_score:
                best, best_score = body, score
        if best is None:
            continue
        # Pull question TEXT that lives inside the body plate (may already be nested).
        members = [header, best]
        nested_text = [
            c for c in (best.get("children") or [])
            if c.get("target") == "text"
        ]
        for node in roots:
            if node.get("id") in used or node in members:
                continue
            if node.get("target") != "text":
                continue
            if _inside(node.get("box") or {}, best.get("box") or {}) < 0.70:
                continue
            members.append(node)
        if not nested_text and not any(m.get("target") == "text" for m in members):
            if best.get("target") != "group":
                continue
        ordered = sorted(
            members,
            key=lambda n: ((n.get("box") or {}).get("y", 0), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in ordered]
        box = _union(boxes)
        layout = _emit_figma_layout_aliases({
            "mode": "VERTICAL",
            "confidence": 0.88,
            "gap": 0,
            "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "align": "MIN",
            "counterAlign": "CENTER",
            "primarySizing": "HUG",
            "counterSizing": "HUG",
        })
        sticker_id = "ama-sticker-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in ordered).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": sticker_id,
            "target": "group",
            "name": "AMA sticker",
            "box": box,
            "z": max(_node_z(n) for n in ordered),
            "children": ordered,
            "layout": layout,
            "meta": {
                "role": "ama-sticker",
                "semantic_name": "AMA sticker",
                "layout_confidence": 0.88,
                "header_id": header.get("id"),
                "body_id": best.get("id"),
            },
        })
        _annotate_stack_children(wrappers[-1], ordered)
        used.update(n.get("id") for n in ordered)

    if not wrappers:
        return roots
    out = [n for n in roots if n.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _quote_frames(roots, canvas, cfg=None):
    """Pair thin rounded quote borders with quote TEXT (+ optional stars).

    When a product/hand cutout overlaps the stroke, bump its z above the frame so the
    break reads correctly (product above stroke).
    """
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    if not grouping.get("quote_frame") and archetype not in {
        "social_screenshot", "caption_over_photo", "lifestyle_overlay",
    }:
        return roots

    frames = []
    for node in roots:
        meta = node.get("meta") or {}
        role = str(meta.get("role") or "").lower().replace("-", "_")
        if meta.get("quote_frame") or role in {"quote_frame", "quote", "testimonial_frame"}:
            frames.append(node)
            continue
        if node.get("target") not in {"shape", "group"}:
            continue
        if not (meta.get("stroke_outline_shell") or node.get("stroke") or meta.get("stroke_outline")):
            # Thin rounded border: stroke present, fill none/transparent.
            fill = node.get("fill")
            if fill not in (None, {}, {"kind": "none"}):
                if isinstance(fill, dict) and fill.get("kind") not in {"none", "transparent"}:
                    if fill.get("color") and _fill_luma(node) is not None:
                        continue
            if not node.get("stroke"):
                continue
        box = node.get("box") or {}
        w = max(1.0, float(box.get("w", 1)))
        h = max(1.0, float(box.get("h", 1)))
        if w < 80 or h < 60:
            continue
        frames.append(node)

    if not frames:
        return roots

    used = set()
    wrappers = []
    for frame in frames:
        if frame.get("id") in used:
            continue
        fbox = frame.get("box") or {}
        members = [frame]
        stars = []
        quotes = []
        products = []
        for node in roots:
            if node is frame or node.get("id") in used:
                continue
            box = node.get("box") or {}
            role = str((node.get("meta") or {}).get("role") or "").lower()
            text = str(node.get("text") or "")
            inside = _inside(box, fbox)
            overlap = _overlap(box, fbox)
            if node.get("target") == "text":
                if inside >= 0.55 or (overlap >= 0.35 and role in {
                    "quote", "body", "caption", "testimonial", "headline", "", "text",
                }):
                    quotes.append(node)
                elif re.search(r"[★☆✦✧⭐]", text) or role in {"rating", "stars", "star"}:
                    if overlap >= 0.20 or inside >= 0.40:
                        stars.append(node)
            elif node.get("target") in {"icon", "image"} and (
                role in {"rating", "stars", "star"} or (node.get("meta") or {}).get("stars")
            ):
                if overlap >= 0.20 or inside >= 0.40:
                    stars.append(node)
            elif node.get("target") in {"image", "icon"} and role in {
                "product", "person", "hand", "photo", "cutout",
            }:
                if overlap >= 0.15:
                    products.append(node)
        if not quotes and not stars:
            continue
        # Product / hand that breaks the frame sits above the stroke.
        for prod in products:
            pz = _node_z(prod)
            fz = _node_z(frame)
            if pz <= fz:
                prod["z"] = fz + 5.0
                prod.setdefault("meta", {})["quote_frame_break"] = True
            members.append(prod)
        members.extend(stars)
        members.extend(quotes)
        ordered = sorted(
            members,
            key=lambda n: (_node_z(n), (n.get("box") or {}).get("y", 0), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in ordered]
        box = _union(boxes)
        qid = "quote-frame-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in ordered).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": qid,
            "target": "group",
            "name": "Quote",
            "box": box,
            "z": max(_node_z(n) for n in ordered),
            "children": ordered,
            "layout": {"mode": "NONE", "confidence": 0.72},
            "meta": {
                "role": "quote-frame",
                "semantic_name": "Quote",
                "layout_confidence": 0.72,
                "frame_id": frame.get("id"),
                "star_count": len(stars),
            },
        })
        used.update(n.get("id") for n in ordered)

    if not wrappers:
        return roots
    out = [n for n in roots if n.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _circular_inset_groups(roots, canvas, cfg=None):
    """Pair a circular product photo with its white ring stroke into one inset group."""
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    routing = (cfg or {}).get("routing") or {}
    if not (
        grouping.get("circular_insets_use_ellipse_mask")
        or routing.get("circular_inset_ellipse")
        or archetype in {"lifestyle_overlay", "social_screenshot", "caption_over_photo"}
    ):
        return roots

    insets = []
    rings = []
    for node in roots:
        meta = node.get("meta") or {}
        role = str(meta.get("role") or "").lower().replace("-", "_")
        box = node.get("box") or {}
        w = max(1.0, float(box.get("w", 1)))
        h = max(1.0, float(box.get("h", 1)))
        aspect = w / max(1.0, h)
        if node.get("target") in {"image", "icon"} and (
            role in {"circular_inset", "inset", "product_inset", "product", "photo"}
            or meta.get("circular") or meta.get("circular_inset")
            or (0.85 <= aspect <= 1.18 and (meta.get("mask_kind") == "ellipse"
                or str((node.get("mask") or {}).get("kind") or "").lower() == "ellipse"))
        ):
            if min(w, h) >= 40:
                insets.append(node)
        if node.get("target") in {"shape", "group"} and (
            meta.get("stroke_outline_shell")
            or meta.get("white_ring")
            or role in {"ring", "inset_ring", "circular_ring"}
            or (node.get("stroke") and not node.get("fill") and 0.85 <= aspect <= 1.18)
        ):
            if min(w, h) >= 48:
                rings.append(node)

    if not insets or not rings:
        return roots

    used = set()
    wrappers = []
    for inset in insets:
        if inset.get("id") in used:
            continue
        icx, icy = _box_center(inset.get("box"))
        iw = max(1.0, float((inset.get("box") or {}).get("w", 1)))
        best = None
        best_dist = None
        for ring in rings:
            if ring.get("id") in used:
                continue
            rcx, rcy = _box_center(ring.get("box"))
            rw = max(1.0, float((ring.get("box") or {}).get("w", 1)))
            # Ring must enclose or tightly match the inset.
            if rw < iw * 0.92 or rw > iw * 1.55:
                continue
            dist = math.hypot(icx - rcx, icy - rcy)
            if dist > max(iw * 0.18, 24.0):
                continue
            if best is None or dist < best_dist:
                best, best_dist = ring, dist
        if best is None:
            continue
        # Product/photo above the white ring stroke.
        if _node_z(inset) <= _node_z(best):
            inset["z"] = _node_z(best) + 5.0
        inset.setdefault("meta", {})["circular_inset"] = True
        inset["meta"].setdefault("role", "circular_inset")
        if isinstance(inset.get("mask"), dict):
            inset["mask"]["kind"] = "ellipse"
        else:
            inset["mask"] = {"kind": "ellipse"}
        members = sorted(
            [best, inset],
            key=lambda n: (_node_z(n), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in members]
        box = _union(boxes)
        cid = "circular-inset-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in members).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": cid,
            "target": "group",
            "name": "Circular inset",
            "box": box,
            "z": max(_node_z(n) for n in members),
            "children": members,
            "layout": {"mode": "NONE", "confidence": 0.8},
            "meta": {
                "role": "circular-inset",
                "semantic_name": "Circular inset",
                "layout_confidence": 0.8,
                "inset_id": inset.get("id"),
                "ring_id": best.get("id"),
            },
        })
        used.update(n.get("id") for n in members)

    if not wrappers:
        return roots
    out = [n for n in roots if n.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _engagement_rows(roots, canvas, cfg=None):
    """Group engagement icons + adjacent count texts into a HORIZONTAL social row.

    Consumes ``layout.scene_grouping.engagement_row`` (social_screenshot). Requires
    at least two engagement-role icons; count labels may sit beside each icon.
    """
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    if not grouping.get("engagement_row") and archetype != "social_screenshot":
        return roots

    icons = []
    for node in roots:
        role = str((node.get("meta") or {}).get("role") or "").lower().replace("-", "_")
        if node.get("target") not in {"icon", "image", "shape"}:
            continue
        if role not in _ENGAGEMENT_ICON_ROLES and not (node.get("meta") or {}).get("engagement"):
            continue
        icons.append(node)
    if len(icons) < 2:
        return roots

    icons = sorted(icons, key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")))
    # Keep icons on a shared baseline band.
    heights = [max(1.0, float((n.get("box") or {}).get("h") or 1)) for n in icons]
    mh = median(heights)
    centers = [
        float((n.get("box") or {}).get("y", 0)) + float((n.get("box") or {}).get("h", 0)) / 2
        for n in icons
    ]
    band = median(centers)
    icons = [
        n for n, cy in zip(icons, centers)
        if abs(cy - band) <= max(mh * 0.55, 24.0)
    ]
    if len(icons) < 2:
        return roots

    used = {n.get("id") for n in icons}
    members = list(icons)
    for node in roots:
        if node.get("id") in used:
            continue
        if node.get("target") != "text":
            continue
        role = str((node.get("meta") or {}).get("role") or "").lower()
        text = str(node.get("text") or "").strip()
        # Counts / meta near engagement icons (257, 21K, likes).
        is_count = role in {"meta", "label", "count", "engagement", "views", ""} or bool(
            re.match(r"^[\d.,]+[kKmMbB]?$", text)
        )
        if not is_count:
            continue
        box = node.get("box") or {}
        cy = float(box.get("y", 0)) + float(box.get("h", 0)) / 2
        if abs(cy - band) > max(mh * 0.70, 28.0):
            continue
        # Must sit near at least one engagement icon.
        near = False
        for icon in icons:
            ib = icon.get("box") or {}
            gap = float(box.get("x", 0)) - (float(ib.get("x", 0)) + float(ib.get("w", 0)))
            if -8.0 <= gap <= max(4.0 * float(ib.get("w", 1)), 80.0):
                near = True
                break
            # Count may also sit under the icon.
            if (
                abs(
                    (float(box.get("x", 0)) + float(box.get("w", 0)) / 2)
                    - (float(ib.get("x", 0)) + float(ib.get("w", 0)) / 2)
                )
                <= max(float(ib.get("w", 1)), 36.0)
                and float(box.get("y", 0)) >= float(ib.get("y", 0))
            ):
                near = True
                break
        if not near:
            continue
        members.append(node)
        used.add(node.get("id"))

    members = sorted(
        members,
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    boxes = [n.get("box") or {} for n in members]
    box = _union(boxes)
    gaps = [
        boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
        for i in range(len(boxes) - 1)
    ]
    layout = _emit_figma_layout_aliases({
        "mode": "HORIZONTAL",
        "confidence": 0.88,
        "gap": _item_spacing([g for g in gaps if g >= 0]),
        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "align": "MIN",
        "counterAlign": "CENTER",
        "primarySizing": "HUG",
        "counterSizing": "HUG",
    })
    row_id = "engagement-row-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in members).encode()
    ).hexdigest()[:10]
    wrapper = {
        "id": row_id,
        "target": "group",
        "name": "Engagement",
        "box": box,
        "z": max(_node_z(n) for n in members),
        "children": members,
        "layout": layout,
        "meta": {
            "role": "engagement-row",
            "semantic_name": "Engagement",
            "layout_confidence": 0.88,
        },
    }
    _annotate_stack_children(wrapper, members)
    out = [n for n in roots if n.get("id") not in used]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


_RATING_STAR_ROLES = frozenset({
    "star", "stars", "rating", "rating_star", "rating-star", "trustpilot",
})
_RATING_TEXT_RE = re.compile(
    r"(excellent|great|good|trustpilot|reviews?|servings?|"
    r"\d(?:\.\d)?\s*/\s*5|\b\d\.\d\b|\b\d{2,}\s*reviews?\b|"
    r"\b\d+(?:\.\d+)?\s*[mk]\+?\s*(?:reviews?|servings?)?\b)",
    re.I,
)
_AS_SEEN_LAYOUT_RE = re.compile(r"as\s+seen\s+in", re.I)
_DAY_LABEL_RE = re.compile(r"^\s*day\s*\d{1,3}\s*$", re.I)
_TIMELINE_ICON_ROLES = frozenset({
    "icon", "chip", "timeline", "step", "day", "badge", "decoration", "shape", "",
})
_TIMELINE_CONNECTOR_ROLES = frozenset({
    "connector", "leader", "callout_leader", "leader_line", "line", "divider",
    "timeline_connector", "spine",
})


def _is_rating_star_node(node) -> bool:
    meta = node.get("meta") or {}
    role = str(meta.get("role") or "").lower().replace("-", "_")
    if role in _RATING_STAR_ROLES or meta.get("rating_star"):
        return True
    if node.get("target") not in {"icon", "shape", "image"}:
        return False
    # Small near-square chips in a star row (when role is generic).
    box = node.get("box") or {}
    w = float(box.get("w") or 0)
    h = float(box.get("h") or 0)
    if w <= 0 or h <= 0 or max(w, h) > 48 or min(w, h) < 6:
        return False
    if max(w, h) / max(1.0, min(w, h)) > 1.45:
        return False
    return role in {"icon", "badge", "decoration", "shape", ""}


def _is_timeline_chip(node) -> bool:
    """Circular Day-1/10/30/90 step icon — chip-sized, not a rating star or CTA."""
    if node.get("target") not in {"icon", "image", "shape"}:
        return False
    meta = node.get("meta") or {}
    if meta.get("rating_star") or meta.get("leader_dot"):
        return False
    role = str(meta.get("role") or "").lower().replace("-", "_").replace(" ", "_")
    if role in _RATING_STAR_ROLES:
        return False
    if role not in _TIMELINE_ICON_ROLES and not meta.get("icon_chip") and not meta.get("timeline_step"):
        return False
    box = node.get("box") or {}
    w = float(box.get("w") or 0)
    h = float(box.get("h") or 0)
    if w <= 0 or h <= 0:
        return False
    # Timeline day discs are larger than star chips and smaller than product art.
    if max(w, h) > 140 or min(w, h) < 22:
        return False
    if max(w, h) / max(1.0, min(w, h)) > 1.35:
        return False
    return True


def _is_timeline_connector(node) -> bool:
    """Thin vertical spine between stacked timeline discs."""
    meta = node.get("meta") or {}
    role = str(meta.get("role") or "").lower().replace("-", "_")
    if role in _TIMELINE_CONNECTOR_ROLES or meta.get("timeline_connector"):
        return True
    if node.get("target") not in {"shape", "icon", "vector"}:
        return False
    box = node.get("box") or {}
    w = float(box.get("w") or 0)
    h = float(box.get("h") or 0)
    if w <= 0 or h <= 0 or w > 28 or h < 48:
        return False
    return h / max(1.0, w) >= 3.0


def _day_label_text(node) -> bool:
    if node.get("target") != "text":
        return False
    return bool(_DAY_LABEL_RE.match(str(node.get("text") or "").strip()))


def _timeline_copy_for_chip(chip, candidates):
    """Pick Day-N label + body TEXT sitting to the right of a timeline disc.

    Prefers the nearest same-row Day label / text-stack so a Day-1 disc cannot
    swallow Day-10 copy when vertical spacing is compact.
    """
    cb = chip.get("box") or {}
    cy = float(cb.get("y", 0)) + float(cb.get("h", 0)) / 2
    chip_top = float(cb.get("y", 0))
    chip_bot = chip_top + float(cb.get("h", 0))
    chip_r = float(cb.get("x", 0)) + float(cb.get("w", 0))
    # Tight primary band (same row). Secondary allows a short body under Day-N.
    primary = max(float(cb.get("h", 0)) * 0.85, 36.0)
    secondary = max(float(cb.get("h", 0)) * 1.65, 56.0)

    scored = []
    for node in candidates:
        if node is chip:
            continue
        nb = node.get("box") or {}
        ny0 = float(nb.get("y", 0))
        ny1 = ny0 + float(nb.get("h", 0))
        ncy = (ny0 + ny1) / 2
        # Prefer copy to the right of the disc; allow slight overlap for tight layouts.
        if float(nb.get("x", 0)) + float(nb.get("w", 0)) * 0.25 < chip_r - 12.0:
            continue
        if float(nb.get("x", 0)) > chip_r + 320.0:
            continue
        dy = abs(ncy - cy)
        role = str((node.get("meta") or {}).get("role") or "")
        is_day = node.get("target") == "text" and _day_label_text(node)
        is_stack = node.get("target") == "group" and role in {
            "text-stack", "text-row", "timeline-step",
        }
        is_body = node.get("target") == "text" and not is_day
        if not (is_day or is_stack or is_body):
            continue
        # Day labels / stacks must sit on the disc row; body may sit just below.
        limit = secondary if is_body and ny0 >= chip_top - 4.0 else primary
        if is_stack:
            # Stacks often span Day + body; allow a slightly taller band but still
            # require vertical overlap with the disc row.
            limit = secondary
            if ny1 < chip_top - 4.0 or ny0 > chip_bot + secondary:
                continue
        if dy > limit and not (is_body and chip_top - 4.0 <= ny0 <= chip_bot + secondary):
            continue
        scored.append((dy, 0 if is_day else 1 if is_stack else 2, node, is_day, is_stack, is_body))

    scored.sort(key=lambda item: (item[0], item[1]))
    day = None
    bodies = []
    for _dy, _prio, node, is_day, is_stack, is_body in scored:
        if is_day and day is None:
            day = node
            continue
        if is_stack and not bodies:
            bodies.append(node)
            continue
        if is_body and not bodies and day is not None:
            # Only attach a loose body when we already claimed this chip's Day label.
            bodies.append(node)
            continue
        if is_body and day is None and not bodies:
            # Chip + body without an explicit Day label (OCR missed "Day N").
            bodies.append(node)
    return day, bodies


def _timeline_groups(roots, canvas=None, cfg=None):
    """Group Day 1/10/30/90 discs + connector + body TEXT into a VERTICAL Timeline.

    Icons stay chips (``icon_chip``); day/body copy stays editable TEXT. Connector
    strokes ride along as siblings — not fused into the discs.
    """
    del cfg  # geometry-only; format tags boost chips upstream via routing
    chips = [n for n in roots if _is_timeline_chip(n)]
    day_labels = [n for n in roots if _day_label_text(n)]
    if len(chips) < 3 and len(day_labels) < 3:
        return roots

    # Column of discs sharing a left edge / center-x.
    chips = sorted(chips, key=lambda n: ((n.get("box") or {}).get("y", 0), n.get("id", "")))
    columns = []
    for chip in chips:
        cb = chip.get("box") or {}
        cx = float(cb.get("x", 0)) + float(cb.get("w", 0)) / 2
        placed = False
        for col in columns:
            ref = col[0].get("box") or {}
            rcx = float(ref.get("x", 0)) + float(ref.get("w", 0)) / 2
            if abs(cx - rcx) <= max(36.0, float(ref.get("w", 0)) * 0.55):
                col.append(chip)
                placed = True
                break
        if not placed:
            columns.append([chip])
    columns = [col for col in columns if len(col) >= 3]
    if not columns and len(day_labels) >= 3:
        # Disc-less fallback: stacked Day labels with body to the right still form a timeline.
        day_labels = sorted(
            day_labels,
            key=lambda n: ((n.get("box") or {}).get("y", 0), n.get("id", "")),
        )
        # Require shared left edge.
        xs = [float((n.get("box") or {}).get("x", 0)) for n in day_labels]
        if max(xs) - min(xs) > 48.0:
            return roots
        steps = []
        used = set()
        for day in day_labels:
            used.add(day.get("id"))
            db = day.get("box") or {}
            dy = float(db.get("y", 0)) + float(db.get("h", 0)) / 2
            body = None
            for node in roots:
                if node.get("id") in used or node.get("target") != "text":
                    continue
                if _day_label_text(node):
                    continue
                nb = node.get("box") or {}
                ncy = float(nb.get("y", 0)) + float(nb.get("h", 0)) / 2
                if abs(ncy - dy) > 40.0:
                    continue
                if float(nb.get("x", 0)) < float(db.get("x", 0)) + float(db.get("w", 0)) - 8.0:
                    continue
                body = node
                break
            kids = [day] + ([body] if body else [])
            if body:
                used.add(body.get("id"))
            step_box = _union([n["box"] for n in kids])
            step_id = "timeline-step-" + hashlib.sha1(
                "|".join(str(n.get("id")) for n in kids).encode()
            ).hexdigest()[:10]
            steps.append({
                "id": step_id,
                "target": "group",
                "name": "Step",
                "box": step_box,
                "z": max(_node_z(n) for n in kids),
                "children": kids,
                "layout": _emit_figma_layout_aliases({
                    "mode": "HORIZONTAL",
                    "confidence": 0.88,
                    "gap": 12,
                    "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "align": "CENTER",
                    "counterAlign": "MIN",
                    "primarySizing": "HUG",
                    "counterSizing": "HUG",
                }),
                "meta": {
                    "role": "timeline-step",
                    "semantic_name": "Step",
                    "layout_confidence": 0.88,
                },
            })
            _annotate_stack_children(steps[-1], kids)
        if len(steps) < 3:
            return roots
        boxes = [s["box"] for s in steps]
        wrap_id = "timeline-" + hashlib.sha1(
            "|".join(s["id"] for s in steps).encode()
        ).hexdigest()[:10]
        wrapper = {
            "id": wrap_id,
            "target": "group",
            "name": "Timeline",
            "box": _union(boxes),
            "z": max(_node_z(s) for s in steps),
            "children": steps,
            "layout": _emit_figma_layout_aliases({
                "mode": "VERTICAL",
                "confidence": 0.9,
                "gap": _item_spacing([
                    steps[i + 1]["box"]["y"] - (steps[i]["box"]["y"] + steps[i]["box"]["h"])
                    for i in range(len(steps) - 1)
                ]),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "MIN",
                "counterAlign": "MIN",
                "primarySizing": "HUG",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": "timeline",
                "semantic_name": "Timeline",
                "layout_confidence": 0.9,
                "step_count": len(steps),
            },
        }
        _annotate_stack_children(wrapper, steps)
        out = [n for n in roots if n.get("id") not in used]
        out.append(wrapper)
        out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
        return out

    if not columns:
        return roots

    # Prefer the tallest column (most steps).
    col = max(columns, key=len)
    col = sorted(col, key=lambda n: ((n.get("box") or {}).get("y", 0), n.get("id", "")))
    # Even vertical rhythm — reject random icon piles.
    gaps = []
    for i in range(len(col) - 1):
        a, b = col[i].get("box") or {}, col[i + 1].get("box") or {}
        gaps.append(float(b.get("y", 0)) - (float(a.get("y", 0)) + float(a.get("h", 0))))
    if gaps and (min(gaps) < -8.0 or max(gaps) > max(220.0, median([
        float((n.get("box") or {}).get("h", 1)) for n in col
    ]) * 4.5)):
        return roots

    used = set()
    steps = []
    pool = [n for n in roots if n.get("id") not in {c.get("id") for c in col}]
    for chip in col:
        chip.setdefault("meta", {})["icon_chip"] = True
        chip["meta"].setdefault("role", "icon")
        chip["meta"]["timeline_step"] = True
        used.add(chip.get("id"))
        day, bodies = _timeline_copy_for_chip(
            chip, [n for n in pool if n.get("id") not in used],
        )
        kids = [chip]
        if day is not None:
            kids.append(day)
            used.add(day.get("id"))
        for body in bodies:
            if body.get("id") in used:
                continue
            kids.append(body)
            used.add(body.get("id"))
        # Require at least one editable text peer per step when day labels exist
        # elsewhere; otherwise allow chip-only if ≥1 step in the column has copy.
        step_box = _union([n["box"] for n in kids])
        step_id = "timeline-step-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in kids).encode()
        ).hexdigest()[:10]
        step = {
            "id": step_id,
            "target": "group",
            "name": "Step",
            "box": step_box,
            "z": max(_node_z(n) for n in kids),
            "children": kids,
            "layout": _emit_figma_layout_aliases({
                "mode": "HORIZONTAL",
                "confidence": 0.9,
                "gap": _item_spacing([
                    kids[i + 1]["box"]["x"] - (kids[i]["box"]["x"] + kids[i]["box"]["w"])
                    for i in range(len(kids) - 1)
                    if kids[i + 1]["box"]["x"] >= kids[i]["box"]["x"]
                ] or [12]),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "CENTER",
                "counterAlign": "MIN",
                "primarySizing": "HUG",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": "timeline-step",
                "semantic_name": "Step",
                "layout_confidence": 0.9,
            },
        }
        _annotate_stack_children(step, kids)
        steps.append(step)

    textful = sum(
        1 for s in steps
        if any(c.get("target") == "text" or (
            c.get("target") == "group" and (c.get("meta") or {}).get("role") in {
                "text-stack", "text-row",
            }
        ) for c in (s.get("children") or []))
    )
    if textful < 2:
        return roots

    # Optional vertical connector in the disc column band.
    first = col[0].get("box") or {}
    last = col[-1].get("box") or {}
    col_x0 = min(float((c.get("box") or {}).get("x", 0)) for c in col) - 8.0
    col_x1 = max(
        float((c.get("box") or {}).get("x", 0)) + float((c.get("box") or {}).get("w", 0))
        for c in col
    ) + 8.0
    y0 = float(first.get("y", 0))
    y1 = float(last.get("y", 0)) + float(last.get("h", 0))
    connectors = []
    for node in roots:
        if node.get("id") in used or not _is_timeline_connector(node):
            continue
        nb = node.get("box") or {}
        ncx = float(nb.get("x", 0)) + float(nb.get("w", 0)) / 2
        if ncx < col_x0 or ncx > col_x1:
            continue
        ny0 = float(nb.get("y", 0))
        ny1 = ny0 + float(nb.get("h", 0))
        if ny1 < y0 - 20.0 or ny0 > y1 + 20.0:
            continue
        node.setdefault("meta", {})["timeline_connector"] = True
        if not node["meta"].get("role"):
            node["meta"]["role"] = "connector"
        connectors.append(node)
        used.add(node.get("id"))

    # Steps own the VERTICAL Auto Layout; connector is an absolute sibling so the
    # tall spine does not participate in the step stack rhythm.
    for conn in connectors:
        hints = dict(conn.get("layout") or {})
        hints["layoutPositioning"] = "ABSOLUTE"
        conn["layout"] = hints
    children = list(connectors) + list(steps)
    boxes = [n["box"] for n in children]
    wrap_id = "timeline-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in children).encode()
    ).hexdigest()[:10]
    wrapper = {
        "id": wrap_id,
        "target": "group",
        "name": "Timeline",
        "box": _union(boxes),
        "z": max(_node_z(n) for n in children),
        "children": children,
        "layout": _emit_figma_layout_aliases({
            "mode": "VERTICAL",
            "confidence": 0.91,
            "gap": _item_spacing([
                steps[i + 1]["box"]["y"] - (steps[i]["box"]["y"] + steps[i]["box"]["h"])
                for i in range(len(steps) - 1)
            ]),
            "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "align": "MIN",
            "counterAlign": "MIN",
            "primarySizing": "HUG",
            "counterSizing": "HUG",
        }),
        "meta": {
            "role": "timeline",
            "semantic_name": "Timeline",
            "layout_confidence": 0.91,
            "step_count": len(steps),
            "has_connector": bool(connectors),
        },
    }
    _annotate_stack_children(wrapper, steps)
    out = [n for n in roots if n.get("id") not in used]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _review_footer_bars(roots, canvas=None, cfg=None):
    """Nest a rating-strip (or stars+copy) into a wide footer bar plate when present.

    IM8-style: cream/maroon footer band with stars + ``4.8/5 REVIEWS | 24M+ SERVINGS``.
    Plate stays a shell; rating TEXT stays editable; stars remain chips or a raster strip.
    """
    del cfg
    canvas_w = float((canvas or {}).get("w") or 0)
    canvas_h = float((canvas or {}).get("h") or 0)
    if canvas_w <= 0 or canvas_h <= 0:
        return roots

    bars = []
    for node in roots:
        role = str((node.get("meta") or {}).get("role") or "").lower().replace("-", "_")
        meta = node.get("meta") or {}
        if node.get("target") not in {"shape", "group", "image"}:
            continue
        if role not in {
            "footer", "footer_bar", "review_bar", "bar", "plate", "banner",
            "shape", "container", "panel", "",
        } and not meta.get("plate_shell") and not meta.get("text_bearing_shell"):
            continue
        box = node.get("box") or {}
        w = float(box.get("w") or 0)
        h = float(box.get("h") or 0)
        y = float(box.get("y") or 0)
        if w < canvas_w * 0.55 or h > canvas_h * 0.18 or h < 28:
            continue
        # Footer band: lower third of the frame.
        if y + h < canvas_h * 0.62:
            continue
        bars.append(node)
    if not bars:
        return roots

    rating_nodes = [
        n for n in roots
        if str((n.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        in {"rating-strip", "rating"}
        or (n.get("meta") or {}).get("intentional_raster_cluster")
        and str((n.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        in {"rating-strip", "star-rating"}
    ]
    # Also accept loose stars + review TEXT not yet wrapped.
    loose_stars = [n for n in roots if _is_rating_star_node(n)]
    review_text = [
        n for n in roots
        if n.get("target") == "text"
        and (
            _RATING_TEXT_RE.search(str(n.get("text") or ""))
            or str((n.get("meta") or {}).get("role") or "").lower() in {
                "rating", "footer", "meta", "label",
            }
        )
    ]

    def _node_has_rating_evidence(node) -> bool:
        role = str((node.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        if role in {"rating-strip", "rating", "review-bar"}:
            return True
        if _is_rating_star_node(node):
            return True
        if node.get("target") == "text" and _RATING_TEXT_RE.search(str(node.get("text") or "")):
            return True
        return any(_node_has_rating_evidence(c) for c in (node.get("children") or []))

    used = set()
    out = list(roots)
    for bar in bars:
        if bar.get("id") in used:
            continue
        bb = bar.get("box") or {}
        # Containment pass may already own stars+copy under the footer plate.
        if bar.get("target") == "group" and _node_has_rating_evidence(bar):
            bar.setdefault("meta", {})["role"] = "review-bar"
            bar["meta"]["semantic_name"] = "Reviews"
            bar["meta"]["plate_shell"] = True
            bar["name"] = "Reviews"
            continue
        members = [bar]
        # Prefer an existing rating-strip fully/mostly inside the bar.
        strip = None
        for node in rating_nodes:
            if node.get("id") in used or node is bar:
                continue
            if _inside(node.get("box") or {}, bb) >= 0.55 or _overlap(node.get("box") or {}, bb) >= 0.45:
                strip = node
                break
        if strip is not None:
            members.append(strip)
        else:
            for node in loose_stars + review_text:
                if node.get("id") in used or node is bar:
                    continue
                nb = node.get("box") or {}
                if _inside(nb, bb) >= 0.5 or _overlap(nb, bb) >= 0.4:
                    members.append(node)
        # Need the bar plus rating evidence (strip, ≥3 stars, or review copy).
        extras = [m for m in members if m is not bar]
        if not extras:
            continue
        has_rating = any(_node_has_rating_evidence(m) for m in extras)
        if not has_rating:
            continue
        # If bar is already a group owning these children, just retag.
        if bar.get("target") == "group" and bar.get("children"):
            child_ids = {c.get("id") for c in (bar.get("children") or [])}
            if all(m.get("id") in child_ids or m is bar for m in members):
                bar.setdefault("meta", {})["role"] = "review-bar"
                bar["meta"]["semantic_name"] = "Reviews"
                bar["name"] = "Reviews"
                continue
        member_ids = {m.get("id") for m in members}
        used |= member_ids
        wrap_id = "review-bar-" + hashlib.sha1(
            "|".join(sorted(member_ids)).encode()
        ).hexdigest()[:10]
        # Promote bar shell: keep surface on the wrapper when bar is a shape.
        wrapper = {
            "id": wrap_id,
            "target": "group",
            "name": "Reviews",
            "box": _union([m.get("box") or {} for m in members]),
            "z": max(_node_z(m) for m in members),
            "children": members,
            "layout": _emit_figma_layout_aliases({
                "mode": "HORIZONTAL",
                "confidence": 0.88,
                "gap": 12,
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "CENTER",
                "counterAlign": "CENTER",
                "primarySizing": "FIXED",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": "review-bar",
                "semantic_name": "Reviews",
                "layout_confidence": 0.88,
                "plate_shell": True,
            },
        }
        if _has_surface(bar):
            _hoist_surface_material(wrapper, bar)
        _annotate_stack_children(wrapper, members)
        out = [n for n in out if n.get("id") not in member_ids]
        out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _rating_strips(roots, canvas=None, cfg=None):
    """Group Trustpilot-style stars + adjacent rating TEXT into a HORIZONTAL strip.

    Fires when ``rating_strip_atomic_fallback`` is on (product_on_flat) or when
    geometry shows 3+ aligned star chips next to rating copy. Individual stars stay
    icons; rating copy stays editable TEXT. A single inseparable star blob may arrive
    tagged ``rating-strip`` / intentional raster — leave it alone.
    """
    grouping = _scene_grouping(cfg)
    allow_atomic = bool(grouping.get("rating_strip_atomic_fallback"))
    stars = [n for n in roots if _is_rating_star_node(n)]
    # Prefer explicitly tagged stars; fall back to compact chips only with the flag.
    tagged = [
        n for n in stars
        if str((n.get("meta") or {}).get("role") or "").lower().replace("-", "_")
        in _RATING_STAR_ROLES
        or (n.get("meta") or {}).get("rating_star")
    ]
    if len(tagged) >= 3:
        stars = tagged
    elif allow_atomic and len(stars) >= 3:
        pass
    elif len(tagged) >= 1 and allow_atomic:
        # One atomic star-row raster + rating text.
        stars = tagged
    else:
        return roots
    if not stars:
        return roots

    stars = sorted(stars, key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")))
    heights = [max(1.0, float((n.get("box") or {}).get("h") or 1)) for n in stars]
    mh = median(heights)
    centers = [
        float((n.get("box") or {}).get("y", 0)) + float((n.get("box") or {}).get("h", 0)) / 2
        for n in stars
    ]
    band = median(centers)
    stars = [
        n for n, cy in zip(stars, centers)
        if abs(cy - band) <= max(mh * 0.65, 20.0)
    ]
    if len(stars) < 1:
        return roots
    if len(stars) == 1 and not (
        (stars[0].get("meta") or {}).get("intentional_raster_cluster")
        or str((stars[0].get("meta") or {}).get("role") or "").lower() in {
            "rating", "rating-strip", "rating_strip",
        }
    ):
        return roots
    if len(stars) >= 2:
        # Shared baseline of discrete stars.
        pass
    elif len(stars) < 1:
        return roots

    used = {n.get("id") for n in stars}
    members = list(stars)
    for node in roots:
        if node.get("id") in used:
            continue
        if node.get("target") != "text":
            continue
        text = str(node.get("text") or "").strip()
        role = str((node.get("meta") or {}).get("role") or "").lower()
        if not (
            _RATING_TEXT_RE.search(text)
            or role in {"rating", "meta", "label", "caption", "body"}
        ):
            continue
        box = node.get("box") or {}
        cy = float(box.get("y", 0)) + float(box.get("h", 0)) / 2
        if abs(cy - band) > max(mh * 1.2, 36.0):
            continue
        # Must sit near the star band (right of last star or overlapping).
        last = stars[-1].get("box") or {}
        first = stars[0].get("box") or {}
        near_right = float(box.get("x", 0)) >= float(first.get("x", 0)) - 8.0
        gap = float(box.get("x", 0)) - (float(last.get("x", 0)) + float(last.get("w", 0)))
        if not near_right:
            continue
        if gap > max(180.0, mh * 6.0):
            continue
        members.append(node)
        used.add(node.get("id"))

    if len(members) < 2 and not (
        len(stars) >= 3 and allow_atomic
    ):
        # Need rating copy OR a clear multi-star row under the atomic flag.
        if len(stars) < 3:
            return roots

    members = sorted(
        members,
        key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
    )
    boxes = [n.get("box") or {} for n in members]
    box = _union(boxes)
    gaps = [
        boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
        for i in range(len(boxes) - 1)
    ]
    layout = _emit_figma_layout_aliases({
        "mode": "HORIZONTAL",
        "confidence": 0.9,
        "gap": _item_spacing([g for g in gaps if g >= 0]),
        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "align": "CENTER",
        "counterAlign": "CENTER",
        "primarySizing": "HUG",
        "counterSizing": "HUG",
    })
    row_id = "rating-strip-" + hashlib.sha1(
        "|".join(str(n.get("id")) for n in members).encode()
    ).hexdigest()[:10]
    wrapper = {
        "id": row_id,
        "target": "group",
        "name": "Rating",
        "box": box,
        "z": max(_node_z(n) for n in members),
        "children": members,
        "layout": layout,
        "meta": {
            "role": "rating-strip",
            "semantic_name": "Rating",
            "layout_confidence": 0.9,
        },
    }
    _annotate_stack_children(wrapper, members)
    out = [n for n in roots if n.get("id") not in used]
    out.append(wrapper)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _logo_strips(roots, canvas=None, cfg=None):
    """Wrap AS SEEN IN press logos into a HORIZONTAL intentional-raster strip."""
    labels = [
        n for n in roots
        if n.get("target") == "text"
        and _AS_SEEN_LAYOUT_RE.search(str(n.get("text") or ""))
    ]
    # Also accept merge-tagged logo-strip members without the label present.
    tagged = [
        n for n in roots
        if str((n.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        in {"logo-strip", "as-seen-in", "press-logos"}
        or (n.get("meta") or {}).get("logo_strip_group_id")
    ]
    if not labels and len(tagged) < 2:
        return roots

    def _collect_band(anchor_y0, anchor_y1, pool):
        band = []
        for node in pool:
            if node.get("target") not in {"image", "icon", "shape"}:
                continue
            meta = node.get("meta") or {}
            role = str(meta.get("role") or "").lower().replace("_", "-")
            if role not in {
                "logo", "platform-logo", "icon", "badge", "wordmark",
                "logo-strip", "as-seen-in", "press-logos", "brand", "",
            } and not meta.get("logo_strip_group_id"):
                continue
            box = node.get("box") or {}
            ly = float(box.get("y") or 0)
            lh = float(box.get("h") or 0)
            if ly + lh < anchor_y0 - 12:
                continue
            if ly > anchor_y1 + 160:
                continue
            w = float(box.get("w") or 0)
            if w <= 0 or lh <= 0 or max(w, lh) > 240:
                continue
            band.append(node)
        return band

    wrappers = []
    used = set()
    if labels:
        for label in labels:
            if label.get("id") in used:
                continue
            tbox = label.get("box") or {}
            y0 = float(tbox.get("y") or 0)
            y1 = y0 + float(tbox.get("h") or 0)
            pool = [n for n in roots if n.get("id") not in used and n is not label]
            band = _collect_band(y0, y1, pool)
            if len(band) < 2:
                continue
            members = sorted(
                [label] + band,
                key=lambda n: (
                    0 if n is label else 1,
                    (n.get("box") or {}).get("x", 0),
                    n.get("id", ""),
                ),
            )
            # Re-sort horizontally for layout; keep label first if it sits above.
            logos_sorted = sorted(
                band,
                key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
            )
            # Label above logos → VERTICAL outer with HORIZONTAL logo row would be nicer,
            # but keep one HORIZONTAL when label shares the band; else nest label + row.
            label_cy = y0 + float(tbox.get("h") or 0) / 2
            logo_band = median([
                float((n.get("box") or {}).get("y", 0))
                + float((n.get("box") or {}).get("h", 0)) / 2
                for n in logos_sorted
            ])
            if abs(label_cy - logo_band) <= max(float(tbox.get("h") or 1) * 1.2, 28.0):
                row_members = sorted(
                    [label] + logos_sorted,
                    key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
                )
            else:
                row_members = [label] + logos_sorted
            boxes = [n.get("box") or {} for n in row_members]
            box = _union(boxes)
            gaps = [
                boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
                for i in range(len(boxes) - 1)
            ]
            wrap_id = "logo-strip-" + hashlib.sha1(
                "|".join(str(n.get("id")) for n in row_members).encode()
            ).hexdigest()[:10]
            for logo in logos_sorted:
                logo.setdefault("meta", {})["intentional_raster_cluster"] = True
                logo.setdefault("meta", {})["role"] = (
                    logo.get("meta", {}).get("role") or "logo-strip"
                )
            wrappers.append({
                "id": wrap_id,
                "target": "group",
                "name": "Logo strip",
                "box": box,
                "z": max(_node_z(n) for n in row_members),
                "children": row_members,
                "layout": _emit_figma_layout_aliases({
                    "mode": "HORIZONTAL",
                    "confidence": 0.88,
                    "gap": _item_spacing([g for g in gaps if g >= 0]),
                    "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "align": "CENTER",
                    "counterAlign": "CENTER",
                    "primarySizing": "HUG",
                    "counterSizing": "HUG",
                }),
                "meta": {
                    "role": "logo-strip",
                    "semantic_name": "Logo strip",
                    "layout_confidence": 0.88,
                    "intentional_raster_cluster": True,
                },
            })
            _annotate_stack_children(wrappers[-1], row_members)
            used.update(n.get("id") for n in row_members)
    elif len(tagged) >= 2:
        band = sorted(
            tagged,
            key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in band]
        gaps = [
            boxes[i + 1].get("x", 0) - (boxes[i].get("x", 0) + boxes[i].get("w", 0))
            for i in range(len(boxes) - 1)
        ]
        wrap_id = "logo-strip-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in band).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": wrap_id,
            "target": "group",
            "name": "Logo strip",
            "box": _union(boxes),
            "z": max(_node_z(n) for n in band),
            "children": band,
            "layout": _emit_figma_layout_aliases({
                "mode": "HORIZONTAL",
                "confidence": 0.86,
                "gap": _item_spacing([g for g in gaps if g >= 0]),
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "align": "CENTER",
                "counterAlign": "CENTER",
                "primarySizing": "HUG",
                "counterSizing": "HUG",
            }),
            "meta": {
                "role": "logo-strip",
                "semantic_name": "Logo strip",
                "layout_confidence": 0.86,
                "intentional_raster_cluster": True,
            },
        })
        _annotate_stack_children(wrappers[-1], band)
        used.update(n.get("id") for n in band)

    if not wrappers:
        return roots
    out = [n for n in roots if n.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _dm_message_rows(roots, canvas, cfg=None):
    """Pair avatar ellipses with adjacent message bubbles into DM thread rows.

    Consumes ``message_bubbles`` grouping (social_screenshot). Unlike header clusters,
    thread avatars may sit at any y — not only the top band.
    """
    grouping = _scene_grouping(cfg)
    archetype = str(((cfg or {}).get("scene") or {}).get("archetype") or "")
    if not grouping.get("message_bubbles") and archetype != "social_screenshot":
        return roots

    bubbles = [
        node for node in roots
        if (node.get("meta") or {}).get("role") == "message-bubble"
        or (node.get("meta") or {}).get("message_bubble")
    ]
    if not bubbles:
        return roots

    avatars = []
    for node in roots:
        role = str((node.get("meta") or {}).get("role") or "").lower()
        if node.get("target") not in {"image", "icon"}:
            continue
        if role not in _AVATAR_ROLES and not (node.get("meta") or {}).get("avatar"):
            continue
        avatars.append(node)
    if not avatars:
        return roots

    used = set()
    wrappers = []
    for bubble in bubbles:
        if bubble.get("id") in used:
            continue
        bbox = bubble.get("box") or {}
        bcy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2
        bh = max(1.0, float(bbox.get("h", 1)))
        best = None
        best_dist = None
        for avatar in avatars:
            if avatar.get("id") in used:
                continue
            abox = avatar.get("box") or {}
            aw = max(1.0, float(abox.get("w", 1)))
            ah = max(1.0, float(abox.get("h", 1)))
            acy = float(abox.get("y", 0)) + ah / 2
            if abs(acy - bcy) > max(bh * 0.70, ah * 0.85, 36.0):
                continue
            # Avatar sits to the left of the bubble with a modest gap.
            ax1 = float(abox.get("x", 0)) + aw
            gap = float(bbox.get("x", 0)) - ax1
            if gap < -aw * 0.35 or gap > max(4.5 * aw, 120.0):
                continue
            dist = abs(acy - bcy) + max(0.0, gap)
            if best is None or dist < best_dist:
                best, best_dist = avatar, dist
        if best is None:
            continue
        members = sorted(
            [best, bubble],
            key=lambda n: ((n.get("box") or {}).get("x", 0), n.get("id", "")),
        )
        boxes = [n.get("box") or {} for n in members]
        box = _union(boxes)
        layout = infer_auto_layout({"box": box, "meta": {"role": "message-row"}}, members)
        row_id = "message-row-" + hashlib.sha1(
            "|".join(str(n.get("id")) for n in members).encode()
        ).hexdigest()[:10]
        wrappers.append({
            "id": row_id,
            "target": "group",
            "name": "Message row",
            "box": box,
            "z": max(_node_z(n) for n in members),
            "children": members,
            "layout": layout,
            "meta": {
                "role": "message-row",
                "semantic_name": "Message row",
                "layout_confidence": layout.get("confidence"),
                "avatar_id": best.get("id"),
                "bubble_id": bubble.get("id"),
            },
        })
        _annotate_stack_children(wrappers[-1], members)
        used.add(best.get("id"))
        used.add(bubble.get("id"))

    if not wrappers:
        return roots
    out = [node for node in roots if node.get("id") not in used]
    out.extend(wrappers)
    out.sort(key=lambda node: (_node_z(node), node.get("id", "")))
    return out


def _merge_card_shells(nodes, containers):
    """Fold a full-bleed painted backdrop into an otherwise empty card shell."""
    dropped = set()
    container_set = set(id(node) for node in containers)
    for host in list(containers):
        if _has_surface(host):
            continue
        role = (host.get("meta") or {}).get("role")
        if role not in (None, "card", "container", "button", "badge", "chip"):
            continue
        host_box = host.get("box") or {}
        backdrops = [node for node in nodes if node is not host and node.get("target") == "shape"
                     and id(node) not in dropped and _has_surface(node)
                     and _inside(node.get("box", {}), host_box) >= 0.94
                     and _area(node.get("box", {})) >= _area(host_box) * 0.88]
        if len(backdrops) != 1:
            continue
        backdrop = backdrops[0]
        _hoist_surface_material(host, backdrop)
        if host.get("radius") is None:
            host["radius"] = backdrop.get("radius") or (backdrop.get("style") or {}).get("radius")
        dropped.add(backdrop["id"])
        if id(backdrop) in container_set:
            containers.remove(backdrop)
            container_set.remove(id(backdrop))
    return dropped


def infer(candidates: list, canvas: dict, cfg: Optional[dict] = None) -> list:
    """Return a nested candidate tree with conservative frames and constraints."""
    cfg = cfg or {}
    lcfg = cfg.get("layout") or {}
    nodes = [deepcopy(c) for c in candidates if c.get("target") != "drop"]
    by_id = {n.get("id"): n for n in nodes}
    total_area = max(1, canvas.get("w", 1) * canvas.get("h", 1))
    # Boxes are stable through the O(N²) containment passes below — cache areas once.
    areas = {id(n): _area(n.get("box", {})) for n in nodes}

    # A shape is a container only when it visibly contains useful siblings. This avoids
    # generating arbitrary groups from loose geometric proximity.
    containers = []
    for node in nodes:
        if node.get("target") != "shape":
            continue
        node_area = areas[id(node)]
        frac = node_area / total_area
        if not (float(lcfg.get("min_container_frac", .002)) <= frac <= float(lcfg.get("max_container_frac", .82))):
            continue
        node_box = node.get("box", {})
        inside = [other for other in nodes if other is not node
                  and areas[id(other)] < node_area * .92
                  and _inside(other.get("box", {}), node_box) >= .92]
        role = (node.get("meta") or {}).get("role")
        meta = node.get("meta") or {}
        role_l = str(role or "").lower().replace("-", "_")
        # Outline rings / quote strokes are overlays, not parents of the photo/product
        # they surround (circular inset white ring, testimonial frame).
        # Exception: text-bearing outline pills (Biomel benefit chips) still own inset copy.
        if (
            meta.get("stroke_outline_shell")
            or meta.get("white_ring")
            or meta.get("quote_frame")
            or role_l in {
                "ring", "inset_ring", "circular_ring", "quote_frame", "quote",
                "testimonial_frame",
            }
        ):
            text_only_shell = (
                meta.get("text_bearing_shell")
                and len(inside) == 1
                and inside[0].get("target") == "text"
            )
            if not text_only_shell:
                continue
        # A giant painted panel that swallows most of the scene AND owns many
        # heterogeneous children is a BACKDROP, not a card: nesting products, prices,
        # decorations AND the CTA under it produces the "everything-under-host" monster
        # group the construction contract forbids (benchmark 002: c_E003 at 70% of
        # canvas owned 8 heterogeneous children). Size alone isn't a reliable signal —
        # an ordinary photo card with 1-2 overlays (a badge, a caption) is legitimately
        # large relative to its canvas and should still group normally; require BOTH the
        # size and the child-count symptom together so a simple photo+badge composition
        # isn't mistaken for a monster host.
        # Semantic shells (button/badge/card/...) keep their children regardless.
        semantic_shell = (
            role in ("button", "badge", "card", "chip", "banner", "seal", "starburst",
                     "callout", "pill", "ama_body", "ama_header", "ama_sticker")
            or meta.get("text_bearing_shell") or meta.get("plate_shell")
            or meta.get("ama_body") or meta.get("ama_header")
        )
        backdrop_host_children = int(lcfg.get("backdrop_host_min_children", 3))
        if (frac > float(lcfg.get("backdrop_host_frac", 0.55)) and not semantic_shell
                and len(inside) >= backdrop_host_children):
            continue
        if len(inside) >= 2 or (len(inside) == 1 and (
                inside[0].get("target") == "text"
                or role in ("button", "badge", "card", "chip", "banner", "seal", "starburst",
                            "callout", "pill", "ama_body", "ama_header", "ama_sticker")
                or meta.get("text_bearing_shell") or meta.get("plate_shell")
                or meta.get("ama_body") or meta.get("ama_header"))):
            containers.append(node)

    # Keep full-bleed painted backdrops as shape children when a larger semantic card owns them.
    pruned = []
    for host in containers:
        host_role = (host.get("meta") or {}).get("role")
        if _has_surface(host) and host_role not in (
            "button", "badge", "card", "chip", "banner", "seal", "starburst",
        ) and not (host.get("meta") or {}).get("text_bearing_shell"):
            host_area = areas[id(host)]
            host_box = host.get("box", {})
            owned_by = [other for other in containers if other is not host
                        and areas[id(other)] >= host_area * 0.98
                        and _inside(host_box, other.get("box", {})) >= 0.94]
            if owned_by:
                continue
        pruned.append(host)
    containers = pruned
    dropped = _merge_card_shells(nodes, containers)

    # Assign every node to its smallest containing frame. Containers can nest.
    parent = {}
    for node in nodes:
        if node.get("id") in dropped:
            continue
        node_area = areas[id(node)]
        node_box = node.get("box", {})
        eligible = [host for host in containers if host is not node
                    and areas[id(host)] > node_area * 1.08
                    and _inside(node_box, host["box"]) >= .92]
        if eligible:
            parent[node["id"]] = min(eligible, key=lambda x: areas[id(x)])["id"]

    for host in containers:
        host["target"] = "group"
        host["children"] = []
    for node in nodes:
        if node.get("id") in dropped:
            continue
        pid = parent.get(node["id"])
        if pid and pid in by_id:
            node["constraints"] = _constraints(node["box"], by_id[pid]["box"])
            by_id[pid].setdefault("children", []).append(node)

    for host in containers:
        direct = host.get("children") or []
        host_meta = host.setdefault("meta", {})
        prior_role = str(host_meta.get("role") or "").lower().replace("-", "_")
        # Preserve social UGC roles set upstream (AMA sticker / quote frame).
        preserve_social = prior_role in {
            "ama_header", "ama_body", "ama_sticker", "question_sticker",
            "quote_frame", "quote", "testimonial_frame",
        } or host_meta.get("ama_body") or host_meta.get("ama_header") or host_meta.get("quote_frame")
        if host_meta.get("text_bearing_shell") or host_meta.get("plate_shell"):
            # Preserve banner/badge shell roles; still mark backplate pairing for layout.
            host_meta["pair_text_with_backplate"] = _pair_text_with_backplate_enabled(lcfg)
            if host_meta.get("role") in (None, "", "shape", "container", "plate"):
                host_meta["role"] = "badge"
        elif preserve_social:
            pass
        elif _is_button_pattern(host, direct):
            host_meta["role"] = "button"
        elif _is_caption_plate(host, direct):
            host_meta["role"] = "caption-plate"
            host_meta["pair_text_with_backplate"] = _pair_text_with_backplate_enabled(lcfg)
        elif _is_stat_pill(host, direct, canvas):
            host_meta["role"] = "stat-pill"
            host_meta["pair_text_with_backplate"] = _pair_text_with_backplate_enabled(lcfg)
        elif (_scene_grouping(cfg).get("message_bubbles", True)
              and _is_message_bubble_pattern(host, direct)):
            host_meta["role"] = "message-bubble"
            host_meta["message_bubble"] = True
            for child in direct:
                if _is_reply_quote_node(child, host):
                    child.setdefault("meta", {})["role"] = "reply-quote"
        host["layout"] = infer_auto_layout(host, direct)
        host_meta["layout_confidence"] = host["layout"].get("confidence")
        host_meta["role"] = host_meta.get("role") or "container"
        if host_meta["role"] in (
            "button", "caption-plate", "message-bubble", "stat-pill", "banner", "badge",
            "callout", "pill",
        ):
            _passthrough_corner_radius(host)

    roots = [n for n in nodes if n.get("id") not in parent and n.get("id") not in dropped]
    # Preserve fusion's semantic image ownership before heuristic text-stack grouping.
    # This stops a UI label over a screenshot/avatar from being split back into a
    # distant top-level layer merely because its owner is an IMAGE rather than a RECT.
    roots = _semantic_asset_groups(roots)
    # Social header (avatar + identity + follow) before text stacks absorb the labels.
    roots = _semantic_header_clusters(roots, canvas, cfg)
    # DM thread rows: avatar ellipse + adjacent message bubble (any y).
    roots = _dm_message_rows(roots, canvas, cfg)
    # Social UGC chrome before text-stacks absorb quote/AMA labels.
    roots = _ama_sticker_frames(roots, canvas, cfg)
    roots = _quote_frames(roots, canvas, cfg)
    roots = _circular_inset_groups(roots, canvas, cfg)
    # Before/after photos + VS chip → one comparison frame (before structural panel wrap).
    roots = _nest_comparison_with_vs(roots, cfg)
    # IM8 BEFORE/RITUAL/RESET stage labels → HORIZONTAL progression strip.
    roots = _nest_stage_progression(roots, cfg)
    # Detector/VLM group IDs plus strict geometry preserve real panel and data
    # structures before the generic text-stack pass can absorb their labels.
    roots = _wrap_structural_sets(roots)
    roots = _semantic_text_stacks(roots)
    # IM8 Day 1/10/30/90 discs + connector + body TEXT → Timeline (before text-rows
    # fuse chip+label into a generic row).
    roots = _timeline_groups(roots, canvas, cfg)
    roots = _semantic_text_rows(roots)
    # IG caption pills: stack sibling caption-plate groups when archetype asks.
    roots = _stack_caption_plates(roots, lcfg)
    # Hears-style left-column stats: stack sibling left-biased plate+text frames.
    roots = _stack_stat_pills(roots, lcfg, canvas)
    # MONTE-style 3-col stats: row aligned vertical text-stacks.
    roots = _row_stat_columns(roots, canvas)
    # Trustpilot stars + rating TEXT; AS SEEN IN press logos (honest raster strip).
    roots = _rating_strips(roots, canvas, cfg)
    roots = _logo_strips(roots, canvas, cfg)
    # Footer review band: bar plate + stars/rating TEXT.
    roots = _review_footer_bars(roots, canvas, cfg)
    # IG/X engagement icons + counts when social chrome roles exist.
    roots = _engagement_rows(roots, canvas, cfg)
    for node in nodes:
        if node.get("children"):
            node["children"].sort(key=lambda c: (_node_z(c), c.get("id", "")))
    roots.sort(key=lambda c: (_node_z(c), c.get("id", "")))

    # Mark exact repeated groups as safe component candidates. Structural-but-different
    # repeats are still discoverable from the signature in metadata, but not instantiated.
    groups = [n for n in nodes if n.get("target") == "group"]
    signatures = {}
    for group in groups:
        sig = _component_signature(group)
        group.setdefault("meta", {})["repeat_signature"] = sig
        signatures.setdefault(sig, []).append(group)
    for sig, matches in signatures.items():
        if len(matches) >= 2:
            for index, match in enumerate(matches):
                match["component"] = {
                    "key": f"repeat-{sig}", "role": "master" if index == 0 else "instance",
                    "confidence": 1.0,
                }

    roots = _wrap_repeated_card_grids(roots)
    # Deterministic deeper nesting: whitespace bands (header/hero/footer) on top of the
    # proven containment groups above, then near-repeat component candidates (metadata).
    roots = _band_groups(roots, canvas, lcfg)
    roots = _unwrap_passthrough_bands(roots)
    _annotate_component_candidates(roots, lcfg.get("repeats") or {})

    # Advisory VLM semantic grouping/naming.  It can only ADD wrapper groups and
    # names on top of the deterministic tree; every invalid proposal is rejected
    # whole and recorded for the caller (scene_intent persists the outcome).
    vlm_notice = None
    if vlm_layout_group.enabled(cfg):
        roots, vlm_notice = vlm_layout_group.regroup(roots, canvas, cfg, z_key=_node_z)
        _finalize_vlm_group_layouts(roots)

    _apply_semantic_names(roots)
    _finalize_layout(roots)

    for root in roots:
        _relativize(root)
    out = _TreeWithNotice(roots)
    out.vlm_grouping = vlm_notice
    return out
