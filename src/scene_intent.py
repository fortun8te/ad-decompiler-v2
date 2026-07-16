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


def _merged_alias(reconstruction: dict) -> dict[str, str]:
    """Map an observation folded away during reconstruction to its surviving owner.

    Reconstruction collapses strongly-overlapping duplicate observations into a single
    winning candidate and records the losers in the winner's ``meta.merged_observations``
    (see reconstruct._dedupe).  A planned id that vanished this way is not truly missing:
    it reconciles, by mapping, to the owner that absorbed its geometry.
    """
    alias: dict[str, str] = {}
    for candidate in reconstruction.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        owner = candidate.get("id")
        if owner in (None, ""):
            continue
        for absorbed in (candidate.get("meta") or {}).get("merged_observations") or []:
            key = str(absorbed)
            # First writer wins; a well-formed reconstruction folds each id at most once.
            alias.setdefault(key, str(owner))
    return alias


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
                                     derived_by_parent: dict[str, list[dict]],
                                     absorbed: dict[str, str] | None = None) -> list[dict]:
    """Apply only explicit reconstruction exceptions without re-inferring the tree."""
    absorbed = absorbed or {}
    kept = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id in derived_by_parent:
            _replace_with_derived_children(node, derived_by_parent[node_id])
            kept.append(node)
            continue
        node["children"] = _apply_reconstruction_exceptions(
            node.get("children") or [], suppressed=suppressed,
            derived_by_parent=derived_by_parent, absorbed=absorbed,
        )
        if node_id in absorbed:
            # Reconstruction folded this planned observation into an overlapping owner
            # candidate (recorded via ``merged_observations``).  A retained wrapper keeps
            # planned children anchored; a bare leaf is dropped because the owner node
            # already paints the merged region — re-drawing it would double the asset.
            if node.get("children"):
                _clear_material(node)
                meta = node.setdefault("meta", {})
                meta["scene_intent_merged_into"] = absorbed[node_id]
                kept.append(node)
            continue
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

    # Reconcile planned ids that reconstruction folded into an overlapping owner.  A
    # merged-away observation is not a lost element; it maps to the candidate that
    # absorbed it.  When that owner is itself a planned node, the absorbed leaf is
    # redundant (the owner paints the union); when it is not, the absorbed node inherits
    # the owner's reconstructed material so its planned geometry still renders.
    merged_alias = _merged_alias(reconstruction)
    absorbed_into_owner: dict[str, str] = {}
    aliased_material: dict[str, str] = {}
    still_missing: list[str] = []
    for value in sorted(planned_source_ids - reconstructed_ids):
        owner = merged_alias.get(value)
        if owner is None or owner not in reconstructed_ids:
            still_missing.append(value)
        elif owner in planned_tree_ids:
            absorbed_into_owner[value] = owner
        else:
            aliased_material[value] = owner
    missing = still_missing

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

    # An absorbed node whose owner is not itself planned still needs paint: point it at
    # the owner's reconstructed material so its planned box renders instead of going blank.
    for absorbed_id, owner_id in aliased_material.items():
        reconstructed.setdefault(absorbed_id, reconstructed[owner_id])

    hydrated = deepcopy(tree)
    hydrated = _apply_reconstruction_exceptions(
        hydrated, suppressed=suppressed, derived_by_parent=derived_by_parent,
        absorbed=absorbed_into_owner,
    )
    _hydrate_tree(hydrated, reconstructed, source_ids)
    hydrated.extend(_derived_node(candidate, None, None) for candidate in derived_roots)
    _ids(hydrated, label="hydrated tree")
    return hydrated


# ---------------------------------------------------------------------------
# Text placement: printed IN the photographed scene vs composited ON TOP of it
#
# The decision the user cares about: "for image ads with overlaid text, it must
# accurately detect whether the text is part of the image or separate."  Both
# error directions are expensive — a baked line recreated as editable text
# double-strikes the photo (the original ink is still there), and an overlay line
# left baked is uneditable copy in a batch job.
#
# Benchmark-6 shows the two poles, and why geometry alone cannot separate them:
#   * 021 — handwritten sticky notes physically placed in a photo of a laptop.
#     The ink is rotated off-axis, camera-blurred and impure.  Must stay BAKED.
#   * 009 — an X/Twitter post screenshot.  Also "text inside a raster", but the
#     glyphs were rendered by a compositor: axis-aligned, hairline-sharp, one
#     pure colour.  Must stay EDITABLE.
# Containment is true for both, which is exactly why the merge-side
# `scene_text_inside` threshold leaks (002 promotes 'VANILLE SMAAK'/'1kg' —
# printed on a pouch — to editable text).  The signals below are the cheap
# physical ones that DO separate them.  Everything here is advisory evidence: a
# VLM verdict, when present, is one more voter and never a veto, because a slow
# local VLM must not be load-bearing for a correctness contract.

_PLACEMENT_BAKED = "baked"
_PLACEMENT_OVERLAY = "overlay"
_PLACEMENT_UNKNOWN = "unknown"

PLACEMENT_DEFAULTS = {
    "enabled": True,
    # A compositor never rotates body copy by accident; a camera always does.
    "axis_tolerance_deg": 2.5,
    "rotation_baked_deg": 4.0,
    # Digitally rendered ink is near-binary at its edges; photographed ink ramps.
    "sharpness_overlay_min": 0.42,
    "sharpness_baked_max": 0.22,
    # Flat design colour vs photographed ink under scene light.
    "purity_overlay_min": 0.60,
    "purity_baked_max": 0.34,
    "grid_tolerance_px": 6.0,
    "grid_min_peers": 2,
    "inside_photo_min": 0.55,
    "vlm_weight": 1.0,
    "decisive_margin": 0.75,
}

_PRODUCT_ROLES = {"product", "package", "packshot", "bottle", "pouch", "can", "tub"}
_SCENE_ROLES = {"photo", "photo-fragment", "photo_fragment", "person", "scene"}


def placement_options(cfg: dict | None = None) -> dict:
    merged = dict(PLACEMENT_DEFAULTS)
    for key, value in (((cfg or {}).get("scene_intent") or {}).get("placement") or {}).items():
        if key in merged and value is not None:
            merged[key] = value
    return merged


def _quad_rotation_deg(quad) -> float | None:
    """Signed rotation of a line's baseline against the canvas x-axis."""
    import math

    try:
        points = [(float(p[0]), float(p[1])) for p in (quad or [])]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) < 2:
        return None
    best, best_len = None, 0.0
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > best_len:
            best, best_len = (x0, y0, x1, y1), length
    if not best or best_len <= 0:
        return None
    x0, y0, x1, y1 = best
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # Fold into (-90, 90]: a run read right-to-left is the same baseline.
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _grid_aligned(box: dict, peers: list[dict], opts: dict) -> bool:
    """True when this line shares a left edge or a centre with enough peers.

    A designer sets copy on a grid: left edges stack, or the block is centred.
    Scene text lands wherever the photographed object happened to be.
    """
    tol = float(opts["grid_tolerance_px"])
    x0 = float(box.get("x") or 0.0)
    cx = x0 + float(box.get("w") or 0.0) / 2.0
    hits = 0
    for peer in peers:
        pbox = peer.get("box") or {}
        px0 = float(pbox.get("x") or 0.0)
        pcx = px0 + float(pbox.get("w") or 0.0) / 2.0
        if abs(px0 - x0) <= tol or abs(pcx - cx) <= tol:
            hits += 1
    return hits >= int(opts["grid_min_peers"])


def ink_statistics(image, box: dict) -> dict | None:
    """Edge sharpness and colour purity of the ink inside ``box``.

    ``sharpness`` — the steepest single-pixel luminance step at an ink boundary,
    normalised by the crop's own range.  Compositor-rendered glyphs step from
    background to ink across ~1px of anti-aliasing and score high; a
    photographed glyph ramps over several pixels and scores low.

    ``purity`` — 1 - (chroma spread of the ink pixels).  Flat design colour is
    one value; photographed ink varies with scene light, paper and JPEG noise.

    Returns None when the crop is unusable so callers degrade to geometry-only
    rather than guessing.  Never raises.
    """
    try:
        import numpy as np
    except Exception:
        return None
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    height, width = arr.shape[:2]
    x0 = max(0, int(float(box.get("x") or 0.0)))
    y0 = max(0, int(float(box.get("y") or 0.0)))
    x1 = min(width, int(x0 + float(box.get("w") or 0.0)))
    y1 = min(height, int(y0 + float(box.get("h") or 0.0)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = arr[y0:y1, x0:x1, :3].astype("float64")
    lum = crop.mean(axis=2)
    lo, hi = float(np.percentile(lum, 2)), float(np.percentile(lum, 98))
    spread = hi - lo
    if spread < 12.0:
        return None  # no ink here worth judging
    # Sharpness = bimodality, NOT contrast.  A compositor's glyph is background
    # or ink with ~1px of anti-aliasing between, so the histogram is two spikes;
    # a photographed glyph ramps through every intermediate value.  (Measuring
    # the steepest step relative to the crop's range instead scores camera-blurred
    # ink 0.5-0.9 — it reads contrast, which photos have plenty of.)
    mid = np.logical_and(lum > lo + 0.30 * spread, lum < lo + 0.70 * spread)
    mid_fraction = float(mid.mean())
    sharpness = max(0.0, min(1.0, 1.0 - mid_fraction / 0.40))
    ink = crop[lum <= float(lum.mean())]
    if ink.size == 0:
        return None
    chroma = ink.max(axis=1) - ink.min(axis=1)
    purity = max(0.0, min(1.0, 1.0 - float(chroma.std()) / 64.0))
    return {"sharpness": round(sharpness, 4), "purity": round(purity, 4),
            "ink_spread": round(spread, 2)}


def _placement_owner(line: dict, elements: list[dict], opts: dict) -> dict | None:
    """The smallest raster that meaningfully contains this line."""
    box = line.get("box") or {}
    best = None
    for element in elements:
        ebox = element.get("box") or {}
        if not ebox:
            continue
        if _inside(box, ebox) < float(opts["inside_photo_min"]):
            continue
        area = float(ebox.get("w") or 0.0) * float(ebox.get("h") or 0.0)
        if area <= 0:
            continue
        if best is None or area < best[0]:
            best = (area, element)
    return best[1] if best else None


def placement_evidence(line: dict, canvas: dict, *, peers: list[dict] | None = None,
                       owner: dict | None = None, image=None,
                       cfg: dict | None = None) -> dict:
    """Collect the cheap physical signals for one OCR line. Never raises."""
    opts = placement_options(cfg)
    box = line.get("box") or {}
    signals: dict = {}
    rotation = _quad_rotation_deg(line.get("quad"))
    if rotation is not None:
        signals["rotation_deg"] = round(rotation, 3)
        signals["axis_aligned"] = abs(rotation) <= float(opts["axis_tolerance_deg"])
    if peers:
        signals["grid_aligned"] = _grid_aligned(box, peers, opts)
    if owner is not None:
        signals["owner_id"] = owner.get("id")
        signals["owner_role"] = str((owner.get("meta") or {}).get("role")
                                    or owner.get("role") or "")
        signals["inside_owner"] = round(_inside(box, owner.get("box") or {}), 4)
    if image is not None:
        stats = ink_statistics(image, box)
        if stats:
            signals.update(stats)
    return {"line_id": line.get("id"), "signals": signals}


def classify_text_placement(line: dict, canvas: dict, *, peers: list[dict] | None = None,
                            owner: dict | None = None, image=None,
                            vlm_placement: str | None = None,
                            cfg: dict | None = None) -> dict:
    """Decide whether a line is printed in the scene or composited on top.

    Returns ``{"placement", "confidence", "reasons", "signals", "votes"}``.  The
    verdict is a weighted vote, so no single cheap signal can flip a line alone.
    With no usable evidence the verdict is ``unknown`` and the caller keeps its
    existing behaviour — this can be introduced without changing undecided lines.
    """
    opts = placement_options(cfg)
    evidence = placement_evidence(line, canvas, peers=peers, owner=owner,
                                  image=image, cfg=cfg)
    signals = evidence["signals"]
    baked, overlay, reasons = 0.0, 0.0, []

    rotation = signals.get("rotation_deg")
    if rotation is not None:
        if abs(rotation) >= float(opts["rotation_baked_deg"]):
            baked += 1.5
            reasons.append("rotated %+.1f deg off canvas axis" % rotation)
        elif signals.get("axis_aligned"):
            # Almost everything is axis-aligned, including every label printed
            # square-on to a pouch. Only the NEGATIVE (rotation) is informative.
            overlay += 0.25
            reasons.append("axis-aligned")

    # Sharpness and purity are ASYMMETRIC evidence.  Soft, scene-lit ink can only
    # come from a camera, so it is strong proof of baked.  Crisp, flat ink merely
    # proves the line was *rendered* — which is equally true of the label printed
    # onto 002's pouch in a product mockup.  Weighting crisp ink as strong overlay
    # evidence scored 002 at 6.9%: every gram of nutrition microcopy is crisp.
    sharpness = signals.get("sharpness")
    if sharpness is not None:
        if sharpness <= float(opts["sharpness_baked_max"]):
            baked += 1.5
            reasons.append("soft photographed ink edges (%.2f)" % sharpness)
        elif sharpness >= float(opts["sharpness_overlay_min"]):
            overlay += 0.5
            reasons.append("rendered ink edges (%.2f)" % sharpness)

    purity = signals.get("purity")
    if purity is not None:
        if purity <= float(opts["purity_baked_max"]):
            baked += 1.0
            reasons.append("scene-lit ink colour (%.2f)" % purity)
        elif purity >= float(opts["purity_overlay_min"]):
            overlay += 0.25
            reasons.append("flat ink colour (%.2f)" % purity)

    inside = signals.get("inside_owner")
    role = str(signals.get("owner_role") or "").lower()
    inside_raster = (inside is not None and inside >= float(opts["inside_photo_min"])
                     and role in (_PRODUCT_ROLES | _SCENE_ROLES))
    # A grid is evidence of a DESIGNER's layout only out on the canvas. Inside a
    # package the product's own artwork supplies the alignment: 002's nutrition
    # table shares a left edge down every row, which scored it "aligned to the
    # copy grid" and tied the vote to unknown for 51 of 72 lines.
    if signals.get("grid_aligned") and not inside_raster:
        overlay += 0.5
        reasons.append("aligned to the copy grid")
    if inside is not None and inside >= float(opts["inside_photo_min"]):
        # Containment is a weak PRIOR, never a verdict: 009's editable tweet copy
        # is fully inside a screenshot raster.  It only tips lines the physical
        # signals left undecided.
        if role in _PRODUCT_ROLES:
            # Ink on a package is printed on the package: the strongest single
            # signal for a mockup render, where no physical blur exists to find.
            baked += 2.0
            reasons.append("inside %s raster (%.2f)" % (role, inside))
        elif role in _SCENE_ROLES:
            # Text inside a photographed scene is part of that scene. Note a
            # screenshot raster carries role 'screenshot', NOT 'photo', so 009's
            # tweet copy is deliberately untouched by this.
            baked += 2.0
            reasons.append("inside %s raster (%.2f)" % (role, inside))

    if vlm_placement:
        verdict = str(vlm_placement).strip().lower()
        weight = float(opts["vlm_weight"])
        if verdict in {"printed", "printed_on_product", "baked"}:
            baked += weight
            reasons.append("vlm says %s" % verdict)
        elif verdict in {"overlay", "overlay_copy", "ui_metadata"}:
            overlay += weight
            reasons.append("vlm says %s" % verdict)

    total = baked + overlay
    if total <= 0:
        placement, confidence = _PLACEMENT_UNKNOWN, 0.0
    elif abs(baked - overlay) < float(opts["decisive_margin"]):
        placement, confidence = _PLACEMENT_UNKNOWN, round(abs(baked - overlay) / total, 4)
    elif baked > overlay:
        placement, confidence = _PLACEMENT_BAKED, round(baked / total, 4)
    else:
        placement, confidence = _PLACEMENT_OVERLAY, round(overlay / total, 4)
    return {"placement": placement, "confidence": confidence, "reasons": reasons,
            "signals": signals, "votes": {"baked": baked, "overlay": overlay}}


def classify_lines(lines: list[dict], canvas: dict, *, elements: list[dict] | None = None,
                   image=None, cfg: dict | None = None) -> dict:
    """Classify every OCR line.  Returns a report keyed by line id."""
    opts = placement_options(cfg)
    if not opts["enabled"] or not lines:
        return {"enabled": bool(opts["enabled"]), "lines": {}, "counts": {}}
    out: dict = {}
    for line in lines:
        peers = [other for other in lines if other is not line]
        owner = _placement_owner(line, elements or [], opts)
        vlm = ((line.get("meta") or {}).get("scene_text") or {}).get("placement")
        out[str(line.get("id"))] = classify_text_placement(
            line, canvas, peers=peers, owner=owner, image=image,
            vlm_placement=vlm, cfg=cfg)
    counts: dict = {}
    for verdict in out.values():
        counts[verdict["placement"]] = counts.get(verdict["placement"], 0) + 1
    return {"enabled": True, "lines": out, "counts": counts}
