"""vlm_layout_group.py — optional VLM semantic grouping over the deterministic layout tree.

After deterministic grouping, the normalized ad plus the flat list of root layers
(id, kind, role, box, text) is shown to a local vision-language model, which proposes
a nested grouping with short semantic names ("header", "product hero", "cta cluster")
and a per-group flex direction hint.  The proposal is strictly advisory: it may only
ADD wrapper groups and names on top of the deterministic tree.  It can never move,
resize, or drop an element — any structurally invalid answer (an element claimed
twice, unknown ids, cyclic nesting, spatially incoherent groups) is rejected as a
whole and the deterministic tree is kept, with the degradation recorded for the
caller (the vlm_element_propose "loud but non-raising" pattern).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re

from src import vlm_client

_DEFAULT_MIN_ELEMENTS = 4
_DEFAULT_MAX_ELEMENTS = 40
_DEFAULT_MAX_GROUPS = 12
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_OVERLAP_TOLERANCE = 0.10
_DEFAULT_CAPTURE_TOLERANCE = 0.70
_DEFAULT_MAX_IMAGE_SIDE = 896
_DEFAULT_TIMEOUT_S = 60
_DEFAULT_MAX_TOKENS = 1200
_MAX_NAME_LEN = 48
_TEXT_SNIPPET_LEN = 40

_DIRECTIONS = frozenset({"row", "column", "none"})

_PROMPT_HEADER = (
    "This image is a flat advertisement that has been decomposed into the layer "
    "elements listed below (id, kind, role, bounding box x/y/w/h in pixels, optional "
    "text). Propose how a designer would organize these layers into nested semantic "
    "groups (for example: header, product hero, cta cluster, stats row, footer).\n\n"
    "Rules:\n"
    "- Use ONLY ids from the element list; never invent, move, resize, or drop elements.\n"
    "- Each id may appear in AT MOST ONE group's member_ids.\n"
    "- Groups may nest: a group's member_ids may include the id of another group you define.\n"
    "- Only group elements that clearly belong together visually; leave unrelated elements ungrouped.\n"
    "- Groups must not spatially overlap each other.\n"
    "- name: a short lowercase semantic name (e.g. 'header', 'product hero', 'cta cluster', 'stats row').\n"
    "- direction: 'row' when members flow left-to-right, 'column' when top-to-bottom, else 'none'.\n"
    "- element_names: optionally give individual elements a clearer short name.\n\n"
    'Reply with ONLY valid JSON: {"groups": [{"id": "g1", "name": "header", '
    '"direction": "row", "member_ids": ["E1", "T0"]}], '
    '"element_names": [{"id": "E2", "name": "brand logo"}]}\n\n'
    "Elements:\n"
)

_SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["groups", "element_names"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "name", "direction", "member_ids"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": sorted(_DIRECTIONS)},
                    "member_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "element_names": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        },
    },
}


def enabled(cfg: dict | None) -> bool:
    """Root VLM switch AND the layout-specific gate (default on when VLM is on)."""
    root = (cfg or {}).get("vlm") or {}
    vg = ((cfg or {}).get("layout") or {}).get("vlm_grouping") or {}
    return bool(root.get("enabled", False)) and bool(vg.get("enabled", True))


def _options(cfg: dict | None) -> dict:
    root = (cfg or {}).get("vlm") or {}
    vg = ((cfg or {}).get("layout") or {}).get("vlm_grouping") or {}
    return {
        "base_url": str(vg.get("base_url") or root.get("base_url") or vlm_client._DEFAULT_BASE_URL),
        "model": str(vg.get("model") or root.get("model") or vlm_client._DEFAULT_MODEL),
        "timeout_s": float(vg.get("timeout_s") or root.get("timeout_s") or _DEFAULT_TIMEOUT_S),
        "max_tokens": int(vg.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "min_elements": int(vg.get("min_elements", _DEFAULT_MIN_ELEMENTS)),
        "max_elements": int(vg.get("max_elements", _DEFAULT_MAX_ELEMENTS)),
        "max_groups": int(vg.get("max_groups", _DEFAULT_MAX_GROUPS)),
        "max_depth": int(vg.get("max_depth", _DEFAULT_MAX_DEPTH)),
        "overlap_tolerance": float(vg.get("overlap_tolerance", _DEFAULT_OVERLAP_TOLERANCE)),
        "capture_tolerance": float(vg.get("capture_tolerance", _DEFAULT_CAPTURE_TOLERANCE)),
        "max_image_side": int(vg.get("max_image_side", _DEFAULT_MAX_IMAGE_SIDE)),
    }


def _area(box: dict) -> float:
    return max(0.0, float(box.get("w", 0) or 0)) * max(0.0, float(box.get("h", 0) or 0))


def _inside(inner: dict, outer: dict) -> float:
    ix = max(0.0, min(inner["x"] + inner["w"], outer["x"] + outer["w"]) - max(inner["x"], outer["x"]))
    iy = max(0.0, min(inner["y"] + inner["h"], outer["y"] + outer["h"]) - max(inner["y"], outer["y"]))
    return (ix * iy) / max(1.0, _area(inner))

def _overlap(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1.0, min(_area(a), _area(b)))


def _union(boxes: list[dict]) -> dict:
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _node_box(node: dict) -> dict:
    box = node.get("box") or {}
    return {
        "x": float(box.get("x", 0) or 0), "y": float(box.get("y", 0) or 0),
        "w": max(0.0, float(box.get("w", 0) or 0)), "h": max(0.0, float(box.get("h", 0) or 0)),
    }


def _backgroundish(node: dict, canvas: dict) -> bool:
    meta = node.get("meta") or {}
    role = str(meta.get("role") or "").lower()
    if role in {"background", "plate", "clean plate"}:
        return True
    canvas_area = max(1.0, float(canvas.get("w", 1) or 1) * float(canvas.get("h", 1) or 1))
    return _area(_node_box(node)) >= canvas_area * 0.88


def _clean_name(raw, fallback: str = "") -> str:
    name = " ".join(str(raw or "").split()).strip()
    if not name:
        return fallback
    if len(name) > _MAX_NAME_LEN:
        name = name[: _MAX_NAME_LEN - 1].rstrip() + "…"
    return name[:1].upper() + name[1:]


def _element_summary(roots: list[dict]) -> list[dict]:
    summary = []
    for node in roots:
        box = _node_box(node)
        entry = {
            "id": str(node.get("id")),
            "kind": str(node.get("target") or "unknown"),
            "role": str((node.get("meta") or {}).get("role") or "unknown"),
            "box": {key: int(round(box[key])) for key in ("x", "y", "w", "h")},
        }
        text = node.get("text")
        if text:
            text = " ".join(str(text).split())
            entry["text"] = text[:_TEXT_SNIPPET_LEN]
        children = node.get("children") or []
        if children:
            entry["children"] = len(children)
        summary.append(entry)
    return summary


def _parse_spec(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        return None
    return data


def validate_spec(spec: dict, roots: list[dict], canvas: dict, *,
                  max_groups: int = _DEFAULT_MAX_GROUPS,
                  max_depth: int = _DEFAULT_MAX_DEPTH,
                  overlap_tolerance: float = _DEFAULT_OVERLAP_TOLERANCE,
                  capture_tolerance: float = _DEFAULT_CAPTURE_TOLERANCE) -> tuple[dict | None, str | None]:
    """Structurally validate a VLM grouping proposal against the real tree.

    Returns (plan, None) on success or (None, reason) when the whole answer must be
    rejected.  ``plan`` contains normalized groups plus advisory element names.  A
    single-member group is demoted to a name-only suggestion rather than a wrapper.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("groups"), list):
        return None, "malformed-spec"
    element_ids = {str(node.get("id")) for node in roots}
    boxes = {str(node.get("id")): _node_box(node) for node in roots}

    groups: dict[str, dict] = {}
    for raw in spec["groups"]:
        if not isinstance(raw, dict):
            return None, "malformed-group"
        gid = str(raw.get("id") or "").strip()
        if not gid:
            return None, "group-without-id"
        if gid in groups or gid in element_ids:
            return None, f"duplicate-group-id:{gid}"
        members = raw.get("member_ids")
        if not isinstance(members, list):
            return None, f"group-without-members:{gid}"
        direction = str(raw.get("direction") or "none").strip().lower()
        if direction not in _DIRECTIONS:
            direction = "none"
        groups[gid] = {
            "id": gid,
            "name": _clean_name(raw.get("name"), fallback="Group"),
            "direction": direction,
            "member_ids": [str(member) for member in members],
        }
    if len(groups) > max_groups:
        return None, f"too-many-groups:{len(groups)}"

    # Every member id must exist and may be claimed exactly once across all groups.
    claimed: set[str] = set()
    for group in groups.values():
        for member in group["member_ids"]:
            if member not in element_ids and member not in groups:
                return None, f"unknown-member:{member}"
            if member in claimed:
                return None, f"duplicate-member:{member}"
            claimed.add(member)
    if any(gid in group["member_ids"] for gid, group in groups.items()):
        return None, "self-membership"

    # Nesting must be an acyclic forest within the depth budget.
    parent_of = {}
    for gid, group in groups.items():
        for member in group["member_ids"]:
            if member in groups:
                parent_of[member] = gid

    def _depth(gid: str, seen: tuple = ()) -> int | None:
        if gid in seen:
            return None
        deepest = 1
        for member in groups[gid]["member_ids"]:
            if member in groups:
                below = _depth(member, seen + (gid,))
                if below is None:
                    return None
                deepest = max(deepest, below + 1)
        return deepest

    def _collect(gid: str, acc: set) -> None:
        acc.add(gid)
        for member in groups[gid]["member_ids"]:
            if member in groups and member not in acc:
                _collect(member, acc)

    reachable: set[str] = set()
    for gid in groups:
        if gid in parent_of:
            continue
        depth = _depth(gid)
        if depth is None:
            return None, "cyclic-groups"
        if depth > max_depth:
            return None, f"too-deep:{depth}"
        _collect(gid, reachable)
    # A cycle among parented groups never appears under any root; catch it explicitly.
    if reachable != set(groups):
        return None, "cyclic-groups"

    # Single-member groups become advisory names on the member, not wrappers.
    name_only: list[dict] = []
    for gid in list(groups):
        group = groups[gid]
        if len(group["member_ids"]) >= 2:
            continue
        members = group["member_ids"]
        if members and members[0] in element_ids:
            name_only.append({"id": members[0], "name": group["name"]})
        # Splice the demoted group out of any parent's member list.
        for other in groups.values():
            if gid in other["member_ids"]:
                index = other["member_ids"].index(gid)
                other["member_ids"][index:index + 1] = members
        del groups[gid]

    def _leaves(gid: str) -> list[str]:
        out = []
        for member in groups[gid]["member_ids"]:
            if member in groups:
                out.extend(_leaves(member))
            else:
                out.append(member)
        return out

    group_boxes = {}
    descendants = {}
    for gid in groups:
        leaves = _leaves(gid)
        if not leaves:
            return None, f"empty-group:{gid}"
        group_boxes[gid] = _union([boxes[leaf] for leaf in leaves])
        descendants[gid] = set(leaves) | {
            member for member in groups[gid]["member_ids"] if member in groups
        }

    def _related(a: str, b: str) -> bool:
        return b in descendants[a] or a in descendants[b]

    ordered = sorted(groups)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if _related(a, b):
                continue
            if _overlap(group_boxes[a], group_boxes[b]) > overlap_tolerance:
                return None, f"groups-overlap:{a}~{b}"

    # A group's bbox must not swallow a foreground element the VLM left outside it.
    for gid in groups:
        gbox = group_boxes[gid]
        for node in roots:
            node_id = str(node.get("id"))
            if node_id in descendants[gid]:
                continue
            if _backgroundish(node, canvas):
                continue
            nbox = boxes[node_id]
            if _area(nbox) >= _area(gbox):
                continue
            if _inside(nbox, gbox) >= capture_tolerance:
                return None, f"captures-nonmember:{gid}~{node_id}"

    element_names = []
    seen_names = set()
    for raw in (spec.get("element_names") or []):
        if not isinstance(raw, dict):
            continue
        target = str(raw.get("id") or "")
        name = _clean_name(raw.get("name"))
        if target in element_ids and name and target not in seen_names:
            element_names.append({"id": target, "name": name})
            seen_names.add(target)
    for item in name_only:
        if item["id"] not in seen_names:
            element_names.append(item)
            seen_names.add(item["id"])

    return {
        "groups": [groups[gid] for gid in ordered if gid in groups],
        "element_names": element_names,
        "group_boxes": group_boxes,
    }, None


def _default_z(node: dict) -> float:
    try:
        return float(node.get("z", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def apply_spec(plan: dict, roots: list[dict], z_key=None) -> tuple[list[dict], int, int]:
    """Wrap validated groups around the existing root nodes (never mutating leaves).

    Returns (new_roots, groups_added, names_applied)."""
    z_key = z_key or _default_z
    by_id = {str(node.get("id")): node for node in roots}
    groups = {group["id"]: group for group in plan["groups"]}

    names_applied = 0
    for item in plan.get("element_names") or []:
        node = by_id.get(item["id"])
        if node is None:
            continue
        meta = node.setdefault("meta", {})
        if not node.get("name") and not meta.get("semantic_name"):
            meta["semantic_name"] = item["name"]
            meta["vlm_named"] = True
            names_applied += 1

    # Build wrappers bottom-up so nested groups exist before their parents.
    built: dict[str, dict] = {}

    def _build(gid: str) -> dict:
        if gid in built:
            return built[gid]
        group = groups[gid]
        children = []
        for member in group["member_ids"]:
            if member in groups:
                children.append(_build(member))
            else:
                children.append(by_id[member])
        children = sorted(children, key=lambda node: (z_key(node), str(node.get("id"))))
        stable = hashlib.sha1(
            ("|".join(sorted(str(child.get("id")) for child in children)) + ":" + group["name"])
            .encode("utf-8")
        ).hexdigest()[:10]
        wrapper = {
            "id": f"vlm-group-{stable}",
            "target": "group",
            "name": group["name"],
            "box": dict(plan["group_boxes"][gid]),
            "z": min(z_key(child) for child in children),
            "children": children,
            "meta": {
                "role": "semantic-group",
                "semantic_name": group["name"],
                "source": "vlm-grouping",
                "vlm_direction_hint": group["direction"],
                "vlm_named": True,
            },
        }
        built[gid] = wrapper
        return wrapper

    consumed: set[str] = set()
    for group in groups.values():
        for member in group["member_ids"]:
            consumed.add(member)
    top_level = [gid for gid in groups if gid not in consumed]
    for gid in top_level:
        _build(gid)

    out = [node for node in roots if str(node.get("id")) not in consumed]
    out.extend(built[gid] for gid in top_level)
    out.sort(key=lambda node: (z_key(node), str(node.get("id"))))
    return out, len(built), names_applied


def _image_bytes(image_path: str, max_side: int) -> bytes | None:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def regroup(roots: list[dict], canvas: dict, cfg: dict | None, z_key=None) -> tuple[list[dict], dict]:
    """Advisory VLM regroup of the root layer list. Never raises.

    Returns (roots, info).  ``roots`` is unchanged whenever the VLM is unavailable
    or its proposal fails structural validation; ``info`` records why so the caller
    can persist the degradation instead of hiding it."""
    info = {"applied": False, "reason": None, "groups_added": 0, "names_applied": 0}
    if not enabled(cfg):
        info["reason"] = "disabled"
        return roots, info
    opts = _options(cfg)
    movable = [node for node in roots if not _backgroundish(node, canvas)]
    if len(movable) < opts["min_elements"]:
        info["reason"] = "too-few-elements"
        return roots, info
    if len(roots) > opts["max_elements"]:
        info["reason"] = "too-many-elements"
        return roots, info
    run_dir = (cfg or {}).get("run_dir")
    image_path = os.path.join(str(run_dir), "normalized.png") if run_dir else None
    if not image_path or not os.path.isfile(image_path):
        info["reason"] = "no-image"
        return roots, info
    image = _image_bytes(image_path, opts["max_image_side"])
    if image is None:
        info["reason"] = "image-error"
        return roots, info

    prompt = _PROMPT_HEADER + json.dumps(_element_summary(roots), ensure_ascii=False)
    try:
        raw = vlm_client.ask_vlm(
            image,
            prompt,
            base_url=opts["base_url"],
            model=opts["model"],
            timeout_s=opts["timeout_s"],
            max_tokens=opts["max_tokens"],
            response_schema=_SPEC_SCHEMA,
        )
    except Exception:
        info["reason"] = "vlm-error"
        return roots, info
    spec = _parse_spec(raw)
    if spec is None:
        info["reason"] = "vlm-parse-error"
        return roots, info
    plan, reason = validate_spec(
        spec, roots, canvas,
        max_groups=opts["max_groups"],
        max_depth=opts["max_depth"],
        overlap_tolerance=opts["overlap_tolerance"],
        capture_tolerance=opts["capture_tolerance"],
    )
    if plan is None:
        info["reason"] = f"vlm-invalid:{reason}"
        return roots, info
    if not plan["groups"] and not plan["element_names"]:
        info["reason"] = "no-groups-proposed"
        return roots, info
    out, added, named = apply_spec(plan, roots, z_key=z_key)
    info.update({"applied": True, "groups_added": added, "names_applied": named})
    return out, info
