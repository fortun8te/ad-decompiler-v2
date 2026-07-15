"""Freeze semantic structure before reconstruction changes asset material.

``merged.json`` is the first canonical, duplicate-free description of the scene.  The
ownership/inpainting stage that follows it is deliberately allowed to change *how* a
leaf is rendered (a shape can become a masked image, an icon can gain SVG paths), but
it must not be allowed to decide a different parent/child tree.  This module records
that boundary and later reattaches the reconstructed material to the frozen tree.

The planner intentionally reuses :mod:`src.layout` for now.  That keeps the existing
conservative grouping and Auto Layout evidence rules, while moving the decision to the
correct side of reconstruction.  If the two representations cannot be reconciled,
callers must use the legacy post-reconstruction layout path and record a degradation;
silently inventing a new tree would hide a fidelity regression.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Iterable

from . import layout


SCHEMA_VERSION = 1

# These fields describe the rendered material of a canonical entity.  In contrast,
# children, boxes, z-order, constraints, component/layout intent, and structural meta
# remain exactly as the pre-reconstruction plan decided them.
_MATERIAL_FIELDS = (
    "name", "text", "style", "text_runs", "fill", "stroke", "radius", "effects",
    "shape_kind", "path", "svg", "src", "mask", "paths", "rotation", "opacity",
    "blend_mode",
)


class SceneIntentError(ValueError):
    """A scene intent is malformed or cannot be reconciled safely."""


def _planning_fingerprint(candidates: list[dict], canvas: dict, cfg: dict | None = None) -> str:
    """Bind a reusable intent to the exact merge + layout inputs that created it."""
    payload = {
        "candidates": candidates,
        "canvas": {"w": canvas.get("w"), "h": canvas.get("h")},
        "layout": (cfg or {}).get("layout") or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk(nodes: Iterable[dict]):
    """Yield every tree node without depending on a particular layout depth."""
    for node in nodes or []:
        if not isinstance(node, dict):
            raise SceneIntentError("scene intent contains a non-object node")
        yield node
        yield from _walk(node.get("children") or [])


def _ids(nodes: Iterable[dict], *, label: str) -> set[str]:
    values: set[str] = set()
    duplicates: set[str] = set()
    for node in _walk(nodes):
        value = node.get("id")
        if value in (None, ""):
            raise SceneIntentError(f"{label} contains a node without an id")
        value = str(value)
        if value in values:
            duplicates.add(value)
        values.add(value)
    if duplicates:
        raise SceneIntentError(f"{label} contains duplicate id(s): {', '.join(sorted(duplicates))}")
    return values


def _source_ids(candidates: list[dict]) -> list[str]:
    values: list[str] = []
    duplicates: set[str] = set()
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SceneIntentError("merged candidates contains a non-object entry")
        if candidate.get("target") == "drop":
            continue
        value = candidate.get("id")
        if value in (None, ""):
            raise SceneIntentError("canonical candidate without an id")
        value = str(value)
        if value in seen:
            duplicates.add(value)
        seen.add(value)
        values.append(value)
    if duplicates:
        raise SceneIntentError(f"merged candidates contains duplicate id(s): {', '.join(sorted(duplicates))}")
    return values


def plan(candidates: list[dict], canvas: dict, cfg: dict | None = None) -> dict:
    """Create the immutable hierarchy/layout plan from canonical merged entities.

    The returned tree uses local child coordinates because that is the compiler-facing
    contract of ``layout.infer``.  Every node also retains ``meta.absolute_box`` from
    the planner, which remains the audit coordinate space across reconstruction.
    """
    if not isinstance(candidates, list):
        raise SceneIntentError("scene intent requires merged candidates as a list")
    if not isinstance(canvas, dict) or not all(isinstance(canvas.get(key), (int, float))
                                               for key in ("w", "h")):
        raise SceneIntentError("scene intent requires numeric canvas width and height")

    source_ids = _source_ids(candidates)
    tree = layout.infer(candidates, canvas, cfg or {})
    tree_ids = _ids(tree, label="planned tree")
    source_set = set(source_ids)
    planned_source_ids = [value for value in source_ids if value in tree_ids]

    # ``layout.infer`` may intentionally absorb a redundant painted shell into a frame.
    # Record that fact rather than pretending the source entity was never observed.
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "scene-intent",
        "planner": "layout.infer",
        "planning_fingerprint": _planning_fingerprint(candidates, canvas, cfg),
        "canvas": {"w": canvas["w"], "h": canvas["h"]},
        "source_ids": source_ids,
        "planned_source_ids": planned_source_ids,
        "suppressed_source_ids": [value for value in source_ids if value not in tree_ids],
        "synthetic_ids": sorted(tree_ids - source_set),
        "tree": tree,
    }
    # The advisory VLM grouping outcome (applied or the rejection reason) is part of
    # the deliverable's provenance: a degraded/skipped pass must be observable in
    # scene_intent.json, never silently identical to "the VLM agreed with us".
    vlm_grouping = getattr(tree, "vlm_grouping", None)
    if vlm_grouping is not None:
        intent["vlm_grouping"] = vlm_grouping
    return intent


def is_current(intent: dict, candidates: list[dict], canvas: dict, cfg: dict | None = None) -> bool:
    """Whether a checkpointed plan still describes the current merge/layout inputs."""
    try:
        return (isinstance(intent, dict)
                and intent.get("schema_version") == SCHEMA_VERSION
                and intent.get("planning_fingerprint") == _planning_fingerprint(candidates, canvas, cfg))
    except (AttributeError, TypeError, ValueError):
        return False


def _validate_intent(intent: dict) -> tuple[list[dict], set[str], set[str]]:
    if not isinstance(intent, dict):
        raise SceneIntentError("scene intent is not an object")
    if intent.get("schema_version") != SCHEMA_VERSION:
        raise SceneIntentError("unsupported or missing scene intent schema version")
    tree = intent.get("tree")
    if not isinstance(tree, list):
        raise SceneIntentError("scene intent tree is not a list")
    source_ids = intent.get("source_ids")
    planned_source_ids = intent.get("planned_source_ids")
    if not isinstance(source_ids, list) or not isinstance(planned_source_ids, list):
        raise SceneIntentError("scene intent is missing canonical id provenance")
    _ids(tree, label="planned tree")
    source_set = {str(value) for value in source_ids}
    planned_set = {str(value) for value in planned_source_ids}
    if not planned_set.issubset(source_set):
        raise SceneIntentError("scene intent planned ids are not canonical source ids")
    return tree, source_set, planned_set


def _reconstructed_by_id(reconstruction: dict) -> dict[str, dict]:
    if not isinstance(reconstruction, dict) or not isinstance(reconstruction.get("candidates"), list):
        raise SceneIntentError("reconstruction is missing its candidate list")
    by_id: dict[str, dict] = {}
    duplicates: set[str] = set()
    for candidate in reconstruction["candidates"]:
        if not isinstance(candidate, dict):
            raise SceneIntentError("reconstruction contains a non-object candidate")
        value = candidate.get("id")
        if value in (None, ""):
            raise SceneIntentError("reconstruction candidate without an id")
        value = str(value)
        if value in by_id:
            duplicates.add(value)
        by_id[value] = candidate
    if duplicates:
        raise SceneIntentError(
            f"reconstruction contains duplicate id(s): {', '.join(sorted(duplicates))}"
        )
    return by_id


def _hydrate_node(node: dict, material: dict | None, source_ids: set[str]) -> None:
    node_id = str(node["id"])
    planned_meta = deepcopy(node.get("meta") or {})
    if material is not None:
        reconstructed_meta = deepcopy(material.get("meta") or {})
        # Structural planner facts win on collision; reconstruction-only provenance
        # (style extraction, vectorization, ownership diagnostics) still survives.
        reconstructed_meta.update(planned_meta)
        planned_meta = reconstructed_meta
        for field in _MATERIAL_FIELDS:
            if field in material:
                node[field] = deepcopy(material[field])
        # A frame remains a frame: the source shell may have begun life as a shape.
        # Leaf material is allowed to change (for example shape -> image) after asset
        # extraction, but it cannot gain/lose a parent or change its coordinates.
        if node.get("target") != "group":
            node["target"] = material.get("target", node.get("target"))
    planned_meta["scene_intent_id"] = node_id
    if node_id not in source_ids:
        planned_meta["scene_intent_synthetic"] = True
    node["meta"] = planned_meta


def _hydrate_tree(nodes: list[dict], reconstructed: dict[str, dict], source_ids: set[str]) -> None:
    for node in nodes:
        node_id = str(node["id"])
        _hydrate_node(node, reconstructed.get(node_id), source_ids)
        # This explicit recursion makes the one-node/one-material boundary clear.
        _hydrate_tree(node.get("children") or [], reconstructed, source_ids)


def _absolute_box(node: dict) -> dict:
    return dict((node.get("meta") or {}).get("absolute_box") or node.get("box") or {})


def _inside(inner: dict, outer: dict) -> float:
    iw = max(0.0, float(inner.get("w", 0) or 0))
    ih = max(0.0, float(inner.get("h", 0) or 0))
    if iw <= 0 or ih <= 0:
        return 0.0
    x0 = max(float(inner.get("x", 0) or 0), float(outer.get("x", 0) or 0))
    y0 = max(float(inner.get("y", 0) or 0), float(outer.get("y", 0) or 0))
    x1 = min(float(inner.get("x", 0) or 0) + iw,
             float(outer.get("x", 0) or 0) + float(outer.get("w", 0) or 0))
    y1 = min(float(inner.get("y", 0) or 0) + ih,
             float(outer.get("y", 0) or 0) + float(outer.get("h", 0) or 0))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(1.0, iw * ih)


def _declared_derived_parent(candidate: dict, planned_ids: set[str]) -> str | None:
    """Return a narrowly declared source owner for a reconstruction-created child."""
    meta = candidate.get("meta") or {}
    parent = meta.get("derived_from")
    if parent not in (None, ""):
        return str(parent) if str(parent) in planned_ids else None
    # Current comparison splitting predates the explicit ``derived_from`` field.  Its
    # parent link plus a before/after side is still strong enough to be deterministic.
    if (str(meta.get("role") or "").lower() == "comparison-column"
            and str(meta.get("comparison_side") or "").lower() in {"before", "after"}
            and meta.get("parent_id") not in (None, "")):
        parent = str(meta["parent_id"])
        return parent if parent in planned_ids else None
    return None


def _is_comparison_plate_column(candidate: dict) -> bool:
    meta = candidate.get("meta") or {}
    return (str(meta.get("role") or "").lower() == "comparison-column"
            and str(meta.get("comparison_side") or "").lower() in {"before", "after"}
            and meta.get("source") == "clean-plate-column")


def _is_explicit_suppression(candidate: dict) -> bool:
    meta = candidate.get("meta") or {}
    return bool(meta.get("keep_in_background") or meta.get("suppression_reason")
                or meta.get("removal_required"))


def _clear_material(node: dict) -> None:
    """Ensure a retained structural wrapper cannot redraw a suppressed source shell."""
    for field in _MATERIAL_FIELDS:
        node.pop(field, None)
    style = node.get("style")
    if isinstance(style, dict):
        style = dict(style)
        for field in ("fill", "fills", "color", "stroke", "strokes"):
            style.pop(field, None)
        if style:
            node["style"] = style
        else:
            node.pop("style", None)


def _relative_box(box: dict, parent_absolute: dict) -> dict:
    relative = dict(box or {})
    relative["x"] = float(relative.get("x", 0) or 0) - float(parent_absolute.get("x", 0) or 0)
    relative["y"] = float(relative.get("y", 0) or 0) - float(parent_absolute.get("y", 0) or 0)
    return relative


def _derived_node(candidate: dict, parent_id: str | None, parent_absolute: dict | None) -> dict:
    node = deepcopy(candidate)
    absolute = dict(candidate.get("box") or {})
    if parent_absolute is not None:
        if _inside(absolute, parent_absolute) < 0.92:
            raise SceneIntentError(
                f"derived layer {candidate.get('id')} lies outside planned parent {parent_id}"
            )
        node["box"] = _relative_box(absolute, parent_absolute)
        for field in ("visible_box", "ink_box"):
            if isinstance(candidate.get(field), dict):
                node[field] = _relative_box(candidate[field], parent_absolute)
        node["constraints"] = {"horizontal": "LEFT", "vertical": "TOP"}
    meta = dict(node.get("meta") or {})
    meta["absolute_box"] = absolute
    meta["scene_intent_id"] = str(node.get("id"))
    if parent_id is None:
        meta["scene_intent_derived_root"] = True
    else:
        meta["scene_intent_derived_from"] = parent_id
    node["meta"] = meta
    return node


def _replace_with_derived_children(node: dict, children: list[dict]) -> None:
    """Turn a planned leaf into a transparent frame around explicit asset derivatives."""
    if node.get("children"):
        raise SceneIntentError(
            f"derived children cannot replace non-leaf planned node {node.get('id')}"
        )
    parent_id = str(node["id"])
    parent_absolute = _absolute_box(node)
    _clear_material(node)
    node["target"] = "group"
    node["layout"] = {"mode": "NONE", "confidence": 1.0}
    node["children"] = [
        _derived_node(child, parent_id, parent_absolute)
        for child in sorted(
            children,
            key=lambda item: (
                float(item.get("z", 0) or 0),
                float((item.get("box") or {}).get("x", 0) or 0),
                float((item.get("box") or {}).get("y", 0) or 0),
                str(item.get("id")),
            ),
        )
    ]
    node.setdefault("meta", {})["scene_intent_derived_group"] = True


def _apply_reconstruction_exceptions(nodes: list[dict], *, suppressed: set[str],
                                     derived_by_parent: dict[str, list[dict]]) -> list[dict]:
    """Apply only explicit reconstruction exceptions without re-inferring the tree."""
    kept = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id in derived_by_parent:
            _replace_with_derived_children(node, derived_by_parent[node_id])
            kept.append(node)
            continue
        node["children"] = _apply_reconstruction_exceptions(
            node.get("children") or [], suppressed=suppressed, derived_by_parent=derived_by_parent
        )
        if node_id not in suppressed:
            kept.append(node)
            continue
        if node.get("children"):
            _clear_material(node)
            node.setdefault("meta", {})["scene_intent_material_suppressed"] = True
            kept.append(node)
        # A leaf explicitly retained in the clean plate must not become an empty Figma
        # image node.  Removing only that leaf keeps all remaining planned geometry intact.
    return kept


def hydrate(intent: dict, reconstruction: dict) -> list[dict]:
    """Attach reconstructed paint/assets while preserving the planned hierarchy exactly.

    A non-bijective reconstruction is intentionally rejected.  The caller can then fall
    back to the old post-reconstruction layout path with a visible degradation instead
    of quietly shipping a different tree.
    """
    tree, source_ids, planned_source_ids = _validate_intent(intent)
    reconstructed = _reconstructed_by_id(reconstruction)
    reconstructed_ids = set(reconstructed)
    planned_tree_ids = _ids(tree, label="planned tree")

    derived_by_parent: dict[str, list[dict]] = {}
    derived_roots: list[dict] = []
    unexpected = []
    for value, candidate in reconstructed.items():
        if value in source_ids or candidate.get("target") == "drop":
            continue
        parent_id = _declared_derived_parent(candidate, planned_tree_ids)
        if parent_id is not None:
            derived_by_parent.setdefault(parent_id, []).append(candidate)
        elif _is_comparison_plate_column(candidate):
            derived_roots.append(candidate)
        else:
            unexpected.append(value)

    missing = sorted(planned_source_ids - reconstructed_ids)
    dropped = []
    suppressed = set()
    for value in planned_source_ids:
        candidate = reconstructed.get(value)
        if not candidate or candidate.get("target") != "drop":
            continue
        if value in derived_by_parent:
            continue
        if _is_explicit_suppression(candidate):
            suppressed.add(value)
        else:
            dropped.append(value)
    if missing or dropped or unexpected:
        pieces = []
        if missing:
            pieces.append(f"missing planned ids: {', '.join(missing)}")
        if dropped:
            pieces.append(f"planned ids became drop: {', '.join(dropped)}")
        if unexpected:
            pieces.append(f"unplanned reconstructed ids: {', '.join(sorted(unexpected))}")
        raise SceneIntentError("scene intent reconciliation failed; " + "; ".join(pieces))

    hydrated = deepcopy(tree)
    hydrated = _apply_reconstruction_exceptions(
        hydrated, suppressed=suppressed, derived_by_parent=derived_by_parent
    )
    _hydrate_tree(hydrated, reconstructed, source_ids)
    hydrated.extend(_derived_node(candidate, None, None) for candidate in derived_roots)
    _ids(hydrated, label="hydrated tree")
    return hydrated
