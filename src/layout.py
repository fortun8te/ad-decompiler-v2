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


_BUTTON_TEXT_ROLES = {"cta", "button", "offer", "price"}
_BUTTON_CONTAINER_ROLES = {"button", "badge", "chip", "card"}


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
        "left": max(0.0, min(b.get("x", 0) for b in boxes) - pb.get("x", 0)),
        "right": max(0.0, pb.get("x", 0) + pb.get("w", 0) - max(b.get("x", 0) + b.get("w", 0) for b in boxes)),
        "top": max(0.0, min(b.get("y", 0) for b in boxes) - pb.get("y", 0)),
        "bottom": max(0.0, pb.get("y", 0) + pb.get("h", 0) - max(b.get("y", 0) + b.get("h", 0) for b in boxes)),
    }


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
                "gap": round(median(gaps), 2) if gaps else 0, "padding": padding,
                "align": "MIN", "counterAlign": "CENTER",
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            })
    if col_spread <= 0.35:
        ordered = sorted(boxes, key=lambda b: b["y"])
        gaps = [ordered[i + 1]["y"] - (ordered[i]["y"] + ordered[i]["h"])
                for i in range(len(ordered) - 1)]
        if _consistent(gaps):
            return _emit_figma_layout_aliases({
                "mode": "VERTICAL", "confidence": round(0.95 - min(.2, col_spread * .2), 3),
                "gap": round(median(gaps), 2) if gaps else 0, "padding": padding,
                "align": "MIN", "counterAlign": "MIN",
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
    group["fill"] = _surface_from(best)
    if group.get("radius") is None:
        group["radius"] = best.get("radius") or (best.get("style") or {}).get("radius")
    if group.get("stroke") is None and best.get("stroke") is not None:
        group["stroke"] = best.get("stroke")
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
    """Strong alignment test for a real text stack, not loose nearby copy."""
    ax, aw = a["x"], max(1.0, a["w"])
    bx, bw = b["x"], max(1.0, b["w"])
    left = abs(ax - bx) <= max(4.0, min(aw, bw) * 0.12)
    center = abs((ax + aw / 2) - (bx + bw / 2)) <= max(5.0, min(aw, bw) * 0.10)
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) / min(aw, bw)
    return left or center or overlap >= 0.72


def _node_z(node):
    raw = node.get("z_index", node.get("z"))
    target = node.get("target")
    # Fusion assigns OCR z=1 to distinguish shell vs label — not final paint order.
    if target == "text" and raw in (None, 0, 1, "0", "0.0", "1", "1.0"):
        return 40.0
    if raw not in (None, 0, "0", "0.0"):
        return float(raw)
    role = str((node.get("meta") or {}).get("role") or node.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"}:
        return -1_000_000.0
    return {"text": 40.0, "icon": 35.0, "image": 25.0}.get(target, 20.0)


def _semantic_text_stacks(roots):
    """Group only clearly contiguous text hierarchy into a vertical Figma frame.

    OCR already emits paragraph blocks. This handles the common separate headline/subhead/body
    stack without inventing a group for every unrelated sentence on the canvas.
    """
    eligible_roles = {"eyebrow", "headline", "title", "subtitle", "subheadline", "body", "caption"}
    texts = [node for node in roots if node.get("target") == "text"
             and (node.get("meta") or {}).get("role", "text") in eligible_roles]
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
                "align": "MIN", "counterAlign": "MIN",
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            },
            "meta": {"role": "text-stack", "semantic_roles": role_names,
                     "layout_confidence": 0.9},
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
        host["fill"] = _surface_from(backdrop)
        if host.get("radius") is None:
            host["radius"] = backdrop.get("radius") or (backdrop.get("style") or {}).get("radius")
        if host.get("stroke") is None and backdrop.get("stroke") is not None:
            host["stroke"] = backdrop.get("stroke")
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
    roots = _semantic_text_stacks(roots)
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
    _finalize_layout(roots)

    for root in roots:
        _relativize(root)
    return roots
