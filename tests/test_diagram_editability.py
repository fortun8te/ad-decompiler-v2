"""Semi-editable chart/diagram readiness: contract helpers, routing, merge demotion."""
from src import diagram_editability as de
from src import layout, routing, vectorize


CANVAS = {"w": 1000, "h": 1000}


def test_chart_primitive_roles_cover_labels_shapes_and_vectors():
    assert "data-label" in de.CHART_LABEL_ROLES
    assert "chart-bar" in de.CHART_SHAPE_ROLES
    assert "plot-line" in de.CHART_VECTOR_ROLES
    assert de.CHART_PRIMITIVE_ROLES == (
        de.CHART_LABEL_ROLES | de.CHART_SHAPE_ROLES | de.CHART_VECTOR_ROLES
    )


def test_ambiguous_divider_needs_chart_group_id():
    assert de.should_route_as_chart_primitive(
        {"meta": {"role": "chart-bar"}}
    )
    assert not de.should_route_as_chart_primitive(
        {"meta": {"role": "divider"}}
    )
    assert de.should_route_as_chart_primitive(
        {"meta": {"role": "divider", "chart_group_id": "sales"}}
    )


def test_chart_bar_photo_fragment_routes_to_native_shape_not_image():
    out = routing.route(
        {"id": "bar", "kind": "photo-fragment",
         "box": {"x": 40, "y": 80, "w": 36, "h": 120},
         "meta": {"role": "chart-bar", "chart_group_id": "sales"}},
        CANVAS,
    )
    assert out["target"] == "shape"
    assert out["shape_kind"] == "rect"
    assert out["meta"]["diagram_mark"] is True
    assert out["meta"]["native_chart_primitive"] is True


def test_plot_line_routes_to_vector_gate():
    out = routing.route(
        {"id": "line", "kind": "photo-fragment",
         "box": {"x": 20, "y": 40, "w": 180, "h": 8},
         "meta": {"role": "plot-line", "chart_group_id": "sales"}},
        CANVAS,
    )
    assert out["target"] == "icon"
    assert out["meta"]["diagram_mark"] is True


def test_whole_chart_cluster_still_stays_intentional_raster():
    out = routing.route(
        {"id": "plot", "kind": "photo-fragment",
         "box": {"x": 10, "y": 10, "w": 400, "h": 240},
         "meta": {"role": "chart"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["meta"]["intentional_raster_cluster"] is True


def test_photo_beside_diagram_is_never_vectorized():
    out = routing.route(
        {"id": "product", "kind": "photo-fragment",
         "box": {"x": 500, "y": 40, "w": 200, "h": 280},
         "meta": {"role": "product"}},
        CANVAS,
    )
    assert out["target"] == "image"
    assert out["meta"].get("diagram_mark") is not True


def test_axis_label_text_stays_native_text():
    out = routing.route(
        {"id": "lbl", "text": "Q1", "kind": "text",
         "box": {"x": 40, "y": 220, "w": 40, "h": 16},
         "meta": {"role": "axis-label", "chart_group_id": "sales"}},
        CANVAS,
    )
    assert out["target"] == "text"


def test_prefer_decomposed_charts_demotes_whole_plot_raster():
    candidates = [
        {"id": "cluster", "target": "image",
         "box": {"x": 0, "y": 0, "w": 300, "h": 200},
         "meta": {"role": "chart", "chart_group_id": "sales",
                  "intentional_raster_cluster": True}},
        {"id": "axis", "target": "shape",
         "box": {"x": 20, "y": 180, "w": 260, "h": 2},
         "meta": {"role": "axis", "chart_group_id": "sales"}},
        {"id": "bar-a", "target": "shape",
         "box": {"x": 40, "y": 100, "w": 36, "h": 80},
         "meta": {"role": "chart-bar", "chart_group_id": "sales"}},
        {"id": "bar-b", "target": "shape",
         "box": {"x": 100, "y": 60, "w": 36, "h": 120},
         "meta": {"role": "chart-bar", "chart_group_id": "sales"}},
    ]
    out = de.prefer_decomposed_charts(candidates)
    by_id = {c["id"]: c for c in out}
    assert by_id["cluster"]["target"] == "drop"
    assert by_id["cluster"]["meta"]["suppression_reason"] == "chart-decomposed-to-primitives"
    assert by_id["bar-a"]["target"] == "shape"


def test_partial_decomposition_does_not_demote_cluster():
    candidates = [
        {"id": "cluster", "target": "image",
         "box": {"x": 0, "y": 0, "w": 300, "h": 200},
         "meta": {"role": "diagram", "chart_group_id": "weak"}},
        {"id": "axis", "target": "shape",
         "box": {"x": 20, "y": 180, "w": 260, "h": 2},
         "meta": {"role": "axis", "chart_group_id": "weak"}},
        {"id": "bar-only", "target": "shape",
         "box": {"x": 40, "y": 100, "w": 36, "h": 80},
         "meta": {"role": "chart-bar", "chart_group_id": "weak"}},
    ]
    out = de.prefer_decomposed_charts(candidates)
    assert next(c for c in out if c["id"] == "cluster")["target"] == "image"


def test_layout_native_chart_still_groups_deterministic_primitives():
    candidates = [
        {"id": "axis", "target": "shape", "box": {"x": 20, "y": 190, "w": 260, "h": 2},
         "meta": {"role": "axis", "chart_group_id": "sales"}},
        {"id": "bar-a", "target": "shape", "box": {"x": 50, "y": 110, "w": 36, "h": 80},
         "meta": {"role": "chart-bar", "chart_group_id": "sales"}},
        {"id": "bar-b", "target": "shape", "box": {"x": 120, "y": 70, "w": 36, "h": 120},
         "meta": {"role": "chart-bar", "chart_group_id": "sales"}},
        {"id": "label", "target": "text", "text": "Sales",
         "box": {"x": 20, "y": 20, "w": 80, "h": 20},
         "meta": {"role": "data-label", "chart_group_id": "sales"}},
    ]
    tree = layout.infer(candidates, {"w": 320, "h": 240})
    assert tree[0]["meta"]["role"] == "native-chart"
    assert tree[0]["layout"]["mode"] == "NONE"


def test_vectorize_gate_limits_for_chart_mark_roles():
    for role in ("axis", "gridline", "plot-line", "data-point", "chart-bar"):
        score, paths = vectorize._gate_limits(role, {})
        assert score <= 0.82
        assert paths <= 16
