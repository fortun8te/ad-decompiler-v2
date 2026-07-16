"""Semi-editable chart/diagram contract.

Default: a detector ``chart`` / ``diagram`` / ``infographic`` / ``graph`` region stays one
intentional raster cluster (exact swappable crop). Semi-editable upgrade is allowed only
when upstream positively decomposes the plot into chart primitive roles tagged with the
same ``chart_group_id``.

Editable vs raster (editability contract)
-----------------------------------------
Keep editable (when decomposed and gated):
  * ``data-label`` / ``axis-label`` / ``tick-label`` / ``legend-label`` → native TEXT
  * ``axis`` / ``axis-line`` / ``gridline`` / ``divider`` / ``bar`` / ``chart-bar`` → SHAPE
  * ``plot-line`` / ``data-line`` / ``connector`` / ``data-point`` / ``marker`` → VECTOR
    when the render-back gate passes; otherwise exact alpha raster

Stay raster (never invent native geometry):
  * Whole ``chart`` / ``diagram`` / ``infographic`` / ``graph`` / ``table`` clusters
  * Photos, products, people, screenshots, receipts inside or beside a diagram
  * Any mark that fails the vectorize render-back gate
  * Mixed groups that still contain an unexplained raster plot crop

Layout may wrap a fully primitive group as ``native-chart`` with absolute geometry
(``layout.mode = NONE``). It must not invent Auto Layout or fabricate missing marks.
"""
from __future__ import annotations

from .raster_clusters import is_intentional_raster_cluster, normalized_role


CHART_LABEL_ROLES = frozenset({
    "data-label", "axis-label", "tick-label", "legend-label",
})
CHART_SHAPE_ROLES = frozenset({
    "axis", "axis-line", "gridline", "divider", "bar", "chart-bar",
})
CHART_VECTOR_ROLES = frozenset({
    "plot-line", "data-line", "data-point", "marker", "connector",
})
CHART_PRIMITIVE_ROLES = CHART_LABEL_ROLES | CHART_SHAPE_ROLES | CHART_VECTOR_ROLES

# Roles that mean "keep the whole crop as one image" — opposite of decomposition.
CHART_CLUSTER_ROLES = frozenset({
    "chart", "graph", "diagram", "infographic", "table",
})

_MARK_ROLES = frozenset({
    "bar", "chart-bar", "plot-line", "data-line", "data-point", "marker",
})
_AXIS_ROLES = frozenset({"axis", "axis-line"})


def chart_role(value) -> str:
    """Normalize detector/VLM chart role spelling."""
    return normalized_role(value)


def is_chart_primitive_role(value) -> bool:
    return chart_role(value) in CHART_PRIMITIVE_ROLES


def is_chart_label_role(value) -> bool:
    return chart_role(value) in CHART_LABEL_ROLES


def is_chart_shape_role(value) -> bool:
    return chart_role(value) in CHART_SHAPE_ROLES


def is_chart_vector_role(value) -> bool:
    return chart_role(value) in CHART_VECTOR_ROLES


def is_chart_cluster_role(value) -> bool:
    return chart_role(value) in CHART_CLUSTER_ROLES


def chart_group_id(candidate: dict | None) -> str | None:
    if not isinstance(candidate, dict):
        return None
    meta = candidate.get("meta") or {}
    value = meta.get("chart_group_id")
    if value in (None, ""):
        value = candidate.get("chart_group_id")
    if value in (None, ""):
        return None
    return str(value)


_AMBIGUOUS_CHART_ROLES = frozenset({"divider", "connector"})


def should_route_as_chart_primitive(candidate: dict | None) -> bool:
    """True when this candidate is a positively identified chart/diagram mark.

    Unambiguous roles (bars, axes, plot lines, labels) always qualify. Ambiguous
    shared roles such as ``divider`` / ``connector`` require ``chart_group_id`` so
    ordinary UI rules and leader lines keep their existing routes.
    """
    if not isinstance(candidate, dict):
        return False
    meta = candidate.get("meta") or {}
    role = chart_role(meta.get("role") or candidate.get("role"))
    if role not in CHART_PRIMITIVE_ROLES:
        return False
    if role in _AMBIGUOUS_CHART_ROLES and not chart_group_id(candidate):
        return False
    return True


def route_target_for_role(value) -> str | None:
    """Map a chart primitive role to a routing target, or None if not a chart mark."""
    role = chart_role(value)
    if role in CHART_LABEL_ROLES:
        return "text"
    if role in CHART_SHAPE_ROLES:
        return "shape"
    if role in CHART_VECTOR_ROLES:
        return "icon"
    return None


def members_support_native_chart(members) -> bool:
    """True when a chart_group is entirely proven primitives (axis + ≥2 marks).

    Mirrors layout._chart_is_deterministic without requiring routed targets yet, so merge
    can demote a whole-chart raster once decomposition evidence exists.
    """
    roles = []
    for member in members or []:
        role = chart_role((member.get("meta") or {}).get("role") or member.get("role"))
        if is_intentional_raster_cluster(role) or is_chart_cluster_role(role):
            # Whole-plot rasters are demoted separately; they do not count as primitives.
            continue
        if role not in CHART_PRIMITIVE_ROLES:
            return False
        roles.append(role)
    if not roles:
        return False
    if any(role not in CHART_PRIMITIVE_ROLES for role in roles):
        return False
    axes = set(roles) & _AXIS_ROLES
    marks = sum(1 for role in roles if role in _MARK_ROLES)
    return bool(axes) and marks >= 2


def prefer_decomposed_charts(candidates: list) -> list:
    """Demote intentional chart/diagram rasters when the same group has native primitives.

    Prevents double ownership: one flat chart IMAGE plus bars/labels that already cover
    the same ink. Photos and non-chart clusters are untouched.
    """
    by_group: dict[str, list] = {}
    for candidate in candidates or []:
        gid = chart_group_id(candidate)
        if gid:
            by_group.setdefault(gid, []).append(candidate)

    demote_ids = set()
    for gid, members in by_group.items():
        primitives = [
            m for m in members
            if is_chart_primitive_role((m.get("meta") or {}).get("role") or m.get("role"))
        ]
        if not members_support_native_chart(primitives):
            continue
        for member in members:
            role = chart_role((member.get("meta") or {}).get("role") or member.get("role"))
            if is_intentional_raster_cluster(role) or is_chart_cluster_role(role):
                demote_ids.add(member.get("id"))

    if not demote_ids:
        return list(candidates or [])

    out = []
    for candidate in candidates:
        if candidate.get("id") not in demote_ids:
            out.append(candidate)
            continue
        demoted = dict(candidate)
        meta = dict(candidate.get("meta") or {})
        meta["layer_disposition"] = "plate"
        meta["keep_in_background"] = True
        meta["suppression_reason"] = "chart-decomposed-to-primitives"
        meta["diagram_editability"] = "raster-cluster-demoted"
        demoted["meta"] = meta
        demoted["target"] = "drop"
        out.append(demoted)
    return out
