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
from statistics import median
from typing import Optional

from . import vlm_layout_group


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
_BUTTON_CONTAINER_ROLES = {"button", "badge", "chip", "card"}

# Brand marks stay independent so a wordmark never fuses into a paragraph flow.
# Every other text role is an ordinary copy line and may join a stack/row when the
# geometry gate below agrees.
_NON_FLOW_TEXT_ROLES = {"logo", "wordmark", "watermark", "brand"}


def _is_button_pattern(container, children):
    """Shape/card shell with a single centered CTA-style label from text_analysis."""
    if len(children) != 1:
        return False
    child = children[0]
    if child.get("target") != "text":
        return False
    child_role = (child.get("meta") or {}).get("role", "text")
    host_role = (container.get("meta") or {}).get("role")
    if child_role not in _BUTTON_TEXT_ROLES and host_role not in _BUTTON_CONTAINER_ROLES:
        return False
    if host_role == "card" and child_role not in _BUTTON_TEXT_ROLES:
        return False
    if not _is_centered(_paint_box(child), container.get("box") or {}):
        return False
    if host_role in _BUTTON_CONTAINER_ROLES or child_role in _BUTTON_TEXT_ROLES:
        return _has_surface(container) or host_role in {"button", "badge", "chip"}
    return False


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
    boxes = [_paint_box(c) for c in children]
    padding = _layout_padding(pb, children)
    if len(children) == 1:
        paint = _paint_box(children[0])
        role = (container.get("meta") or {}).get("role")
        is_button = _is_button_pattern(container, children)
        if is_button or (
            _is_centered(paint, pb)
            and role in ("button", "badge", "chip")
        ):
            mode = "VERTICAL" if pb.get("h", 0) > pb.get("w", 0) * 1.35 else "HORIZONTAL"
            return _emit_figma_layout_aliases({
                "mode": mode, "confidence": 0.92, "gap": 0, "itemSpacing": 0,
                "padding": padding, "align": "CENTER", "counterAlign": "CENTER",
                "primaryAxisAlignItems": "CENTER", "counterAxisAlignItems": "CENTER",
                "primarySizing": "HUG", "counterSizing": "HUG",
            })
        # Padded card: a surfaced plate/card wrapping one substantial, fully-inset
        # child becomes a HUG frame so the plate resizes with its content instead of
        # freezing at pixel size.  The measured four-side padding reproduces the
        # original box exactly (see _is_padded_child), so this never moves geometry.
        if (_has_surface(container)
                and role in (None, "card", "container", "plate", "panel")
                and children[0].get("target") in ("text", "image", "icon")
                and _is_padded_child(paint, pb)):
            mode = "VERTICAL" if pb.get("h", 0) >= pb.get("w", 0) else "HORIZONTAL"
            return _emit_figma_layout_aliases({
                "mode": mode, "confidence": 0.85, "gap": 0, "itemSpacing": 0,
                "padding": padding, "align": "MIN", "counterAlign": "MIN",
                "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN",
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
    if role in ("button", "badge", "chip") or _is_button_pattern(parent, children):
        for child in children:
            hints = dict(child.get("layout") or {})
            hints["layoutAlign"] = "CENTER"
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
        return "CTA cluster"
    if "logo" in roles and cy <= 0.30:
        return "Header"
    if cy <= 0.16:
        return "Header"
    if cy >= 0.84:
        return "Footer"
    if roles & {"product", "person", "product_cluster", "illustration", "avatar"}:
        return "Hero"
    if targets <= {"text"}:
        return "Copy block"
    return "Content group"


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
    """Give structural frames designer-facing names; explicit/VLM names always win."""
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
            label = f'CTA Button — "{_short(text)}"' if text else "CTA Button"
        elif role == "text-stack":
            text = _first_text_content(node)
            label = f'Copy — "{_short(text)}"' if text else "Copy block"
        elif role == "card-grid":
            label = f"Card grid ({len(children)})"
        elif role == "panel-set":
            label = f"Panel set ({len(children)})"
        elif role == "structural-grid":
            label = f"Grid ({len(children)} rows)"
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
_CHART_PRIMITIVE_ROLES = {
    "axis", "axis-line", "gridline", "divider", "bar", "chart-bar",
    "plot-line", "data-line", "data-point", "marker", "data-label", "axis-label",
}


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


def _chart_is_deterministic(members):
    roles = {
        str((member.get("meta") or {}).get("role") or "").lower().replace("_", "-")
        for member in members
    }
    if any(member.get("target") not in {"shape", "text", "icon"} for member in members):
        return False
    if any(role not in _CHART_PRIMITIVE_ROLES for role in roles):
        return False
    axes = roles & {"axis", "axis-line"}
    marks = sum(role in {"bar", "chart-bar", "plot-line", "data-line", "data-point", "marker"}
                for role in [str((member.get("meta") or {}).get("role") or "").lower().replace("_", "-")
                             for member in members])
    return bool(axes) and marks >= 2


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
            if role == "button" or _is_button_pattern(node, children):
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
              and str((node.get("meta") or {}).get("role", "")).lower() not in _NON_FLOW_TEXT_ROLES]
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
        out.append({
            "id": row_id,
            "target": "group",
            "box": box,
            "z": max(_node_z(node) for node in group),
            "children": group,
            "layout": layout,
            "meta": {"role": "text-row", "layout_confidence": 0.88},
        })
        _annotate_stack_children(out[-1], group)
    return out


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
        parent_id = (node.get("meta") or {}).get("parent_id")
        if parent_id and parent_id in by_id and parent_id != node.get("id"):
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
            "name": f"{label} — asset group",
            "box": dict(owner_box),
            "z": min([_node_z(owner)] + [_node_z(child) for child in accepted]),
            "children": [owner] + sorted(accepted, key=lambda node: (_node_z(node), node.get("id", ""))),
            "layout": {"mode": "NONE", "confidence": 1.0},
            "meta": {
                "role": "asset-group",
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

    # A shape is a container only when it visibly contains useful siblings. This avoids
    # generating arbitrary groups from loose geometric proximity.
    containers = []
    for node in nodes:
        if node.get("target") != "shape":
            continue
        frac = _area(node.get("box", {})) / total_area
        if not (float(lcfg.get("min_container_frac", .002)) <= frac <= float(lcfg.get("max_container_frac", .82))):
            continue
        inside = [other for other in nodes if other is not node
                  and _area(other.get("box", {})) < _area(node.get("box", {})) * .92
                  and _inside(other.get("box", {}), node.get("box", {})) >= .92]
        role = (node.get("meta") or {}).get("role")
        if len(inside) >= 2 or (len(inside) == 1 and
                                (inside[0].get("target") == "text" or role in ("button", "badge", "card", "chip"))):
            containers.append(node)

    # Keep full-bleed painted backdrops as shape children when a larger semantic card owns them.
    pruned = []
    for host in containers:
        if _has_surface(host) and (host.get("meta") or {}).get("role") not in ("button", "badge", "card", "chip"):
            owned_by = [other for other in containers if other is not host
                        and _area(other.get("box", {})) >= _area(host.get("box", {})) * 0.98
                        and _inside(host.get("box", {}), other.get("box", {})) >= 0.94]
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
        eligible = [host for host in containers if host is not node
                    and _area(host["box"]) > _area(node["box"]) * 1.08
                    and _inside(node["box"], host["box"]) >= .92]
        if eligible:
            parent[node["id"]] = min(eligible, key=lambda x: _area(x["box"]))["id"]

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
        if _is_button_pattern(host, direct):
            host.setdefault("meta", {})["role"] = "button"
        host["layout"] = infer_auto_layout(host, direct)
        host.setdefault("meta", {})["layout_confidence"] = host["layout"].get("confidence")
        host["meta"]["role"] = host["meta"].get("role") or "container"
        if host["meta"]["role"] == "button":
            _passthrough_corner_radius(host)

    roots = [n for n in nodes if n.get("id") not in parent and n.get("id") not in dropped]
    # Preserve fusion's semantic image ownership before heuristic text-stack grouping.
    # This stops a UI label over a screenshot/avatar from being split back into a
    # distant top-level layer merely because its owner is an IMAGE rather than a RECT.
    roots = _semantic_asset_groups(roots)
    # Detector/VLM group IDs plus strict geometry preserve real panel and data
    # structures before the generic text-stack pass can absorb their labels.
    roots = _wrap_structural_sets(roots)
    roots = _semantic_text_stacks(roots)
    roots = _semantic_text_rows(roots)
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
