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


def infer_auto_layout(container, children):
    """Return Figma layout intent or NONE when geometry should remain absolute."""
    pb = container["box"]
    if not children:
        return {"mode": "NONE", "confidence": 0.0}
    boxes = [c["box"] for c in children]
    padding = {
        "left": max(0.0, min(b["x"] for b in boxes) - pb["x"]),
        "right": max(0.0, pb["x"] + pb["w"] - max(b["x"] + b["w"] for b in boxes)),
        "top": max(0.0, min(b["y"] for b in boxes) - pb["y"]),
        "bottom": max(0.0, pb["y"] + pb["h"] - max(b["y"] + b["h"] for b in boxes)),
    }
    if len(children) == 1:
        child = boxes[0]
        centered_x = abs((child["x"] + child["w"] / 2) - (pb["x"] + pb["w"] / 2)) <= max(3, pb["w"] * .04)
        centered_y = abs((child["y"] + child["h"] / 2) - (pb["y"] + pb["h"] / 2)) <= max(3, pb["h"] * .08)
        role = (container.get("meta") or {}).get("role")
        if centered_x and centered_y and (role in ("button", "badge", "chip") or container.get("target") == "shape"):
            return {
                "mode": "HORIZONTAL", "confidence": 0.92, "gap": 0,
                "padding": padding, "align": "CENTER", "counterAlign": "CENTER",
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            }
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
            return {
                "mode": "HORIZONTAL", "confidence": round(0.95 - min(.2, row_spread * .2), 3),
                "gap": round(median(gaps), 2) if gaps else 0, "padding": padding,
                "align": "MIN", "counterAlign": "CENTER",
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            }
    if col_spread <= 0.35:
        ordered = sorted(boxes, key=lambda b: b["y"])
        gaps = [ordered[i + 1]["y"] - (ordered[i]["y"] + ordered[i]["h"])
                for i in range(len(ordered) - 1)]
        if _consistent(gaps):
            return {
                "mode": "VERTICAL", "confidence": round(0.95 - min(.2, col_spread * .2), 3),
                "gap": round(median(gaps), 2) if gaps else 0, "padding": padding,
                "align": "MIN", "counterAlign": "MIN",
                "primarySizing": "FIXED", "counterSizing": "FIXED",
            }
    return {"mode": "NONE", "confidence": 0.25}


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
    absolute = dict(node.get("box") or {})
    node.setdefault("meta", {})["absolute_box"] = absolute
    if parent_abs:
        node["box"] = {
            **absolute,
            "x": absolute.get("x", 0) - parent_abs.get("x", 0),
            "y": absolute.get("y", 0) - parent_abs.get("y", 0),
        }
        visible = node.get("visible_box")
        if visible:
            node["visible_box"] = {
                **visible,
                "x": visible.get("x", 0) - parent_abs.get("x", 0),
                "y": visible.get("y", 0) - parent_abs.get("y", 0),
            }
    for child in node.get("children") or []:
        _relativize(child, absolute)


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
        if (0 <= gap <= max(14.0, median_h * 1.75)
                and _text_alignment(prior_box, box)):
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
            "z": min(float(node.get("z", 0)) for node in group),
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
    return out


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

    # Assign every node to its smallest containing frame. Containers can nest.
    parent = {}
    for node in nodes:
        eligible = [host for host in containers if host is not node
                    and _area(host["box"]) > _area(node["box"]) * 1.08
                    and _inside(node["box"], host["box"]) >= .92]
        if eligible:
            parent[node["id"]] = min(eligible, key=lambda x: _area(x["box"]))["id"]

    for host in containers:
        host["target"] = "group"
        host["children"] = []
    for node in nodes:
        pid = parent.get(node["id"])
        if pid and pid in by_id:
            node["constraints"] = _constraints(node["box"], by_id[pid]["box"])
            by_id[pid].setdefault("children", []).append(node)

    for host in containers:
        direct = host.get("children") or []
        host["layout"] = infer_auto_layout(host, direct)
        host.setdefault("meta", {})["layout_confidence"] = host["layout"].get("confidence")
        host["meta"]["role"] = host["meta"].get("role") or "container"

    roots = [n for n in nodes if n.get("id") not in parent]
    roots = _semantic_text_stacks(roots)
    for node in nodes:
        if node.get("children"):
            node["children"].sort(key=lambda c: (float(c.get("z", 0)), c.get("id", "")))
    roots.sort(key=lambda c: (float(c.get("z", 0)), c.get("id", "")))

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

    for root in roots:
        _relativize(root)
    return roots
