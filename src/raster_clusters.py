"""Contracts for visual regions that should remain one exact swappable raster.

Some ad elements look structured but cannot be safely rebuilt from a screenshot: a social
post, receipt, chart, nutrition table, infographic, or a tightly overlapping product scene.
When a detector positively assigns one of these roles, retain its complete source crop as
one named Figma image rather than inventing many unreliable layers inside it.
"""
from __future__ import annotations


INTENTIONAL_RASTER_CLUSTER_ROLES = frozenset({
    "screenshot", "ui-panel", "receipt", "chart", "graph", "table",
    "nutrition-panel", "diagram", "infographic", "product-cluster",
})

_ALIASES = {
    "ui": "ui-panel",
    "ui-panel": "ui-panel",
    "ui-screenshot": "screenshot",
    "app-screenshot": "screenshot",
    "social-screenshot": "screenshot",
    "chart-graph": "chart",
    "nutrition": "nutrition-panel",
    "nutrition-facts": "nutrition-panel",
    "nutrition-table": "nutrition-panel",
    "info-graphic": "infographic",
    "inseparable-product-cluster": "product-cluster",
    "product-scene": "product-cluster",
}

_LABELS = {
    "screenshot": "Screenshot",
    "ui-panel": "UI panel",
    "receipt": "Receipt",
    "chart": "Chart",
    "graph": "Graph",
    "table": "Table",
    "nutrition-panel": "Nutrition panel",
    "diagram": "Diagram",
    "infographic": "Infographic",
    "product-cluster": "Product cluster",
}


def normalized_role(value) -> str:
    """Normalize detector/VLM spelling without widening the cluster policy."""
    token = str(value or "").strip().lower().replace("_", " ").replace("/", " ")
    token = "-".join(token.split())
    return _ALIASES.get(token, token)


def is_intentional_raster_cluster(value) -> bool:
    return normalized_role(value) in INTENTIONAL_RASTER_CLUSTER_ROLES


def cluster_label(value) -> str:
    """A stable, designer-facing layer name for an intentional raster owner."""
    role = normalized_role(value)
    return _LABELS.get(role, role.replace("-", " ").title() or "Raster cluster")
