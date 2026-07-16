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
    # Press / trust chrome: brand marks in an AS SEEN IN band are not worth
    # inventing as native vectors — keep one honest swappable strip (or chips).
    "logo-strip", "as-seen-in", "press-logos",
    # Atomic rating fallback when individual stars cannot be separated cleanly.
    "rating-strip",
    # IM8: chaotic pill clouds and photographic body-morph strips stay one crop.
    "pill-cloud", "body-progression", "body-morph",
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
    "pill-cloud": "pill-cloud",
    "pill_cloud": "pill-cloud",
    "chaotic-pill-cluster": "pill-cloud",
    "pill-cluster": "pill-cloud",
    "body-progression": "body-progression",
    "body_progression": "body-progression",
    "body-morph": "body-morph",
    "body_morph": "body-morph",
    "body-stage-strip": "body-progression",
    "logo-strip": "logo-strip",
    "logo_strip": "logo-strip",
    "as-seen-in": "as-seen-in",
    "as_seen_in": "as-seen-in",
    "asseenin": "as-seen-in",
    "press-logos": "press-logos",
    "press_logos": "press-logos",
    "press-strip": "logo-strip",
    "rating-strip": "rating-strip",
    "rating_strip": "rating-strip",
    "star-rating": "rating-strip",
    "trustpilot": "rating-strip",
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
    "pill-cloud": "Pill cloud",
    "body-progression": "Body progression",
    "body-morph": "Body morph",
    "logo-strip": "Logo strip",
    "as-seen-in": "As seen in",
    "press-logos": "Press logos",
    "rating-strip": "Rating",
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
