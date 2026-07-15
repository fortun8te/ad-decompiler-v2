from src import layout


def test_button_becomes_native_frame_with_relative_child():
    candidates = [
        {"id": "button", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
         "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 12},
         "meta": {"role": "button"}},
        {"id": "label", "target": "text", "box": {"x": 60, "y": 44, "w": 80, "h": 24},
         "visible_box": {"x": 62, "y": 47, "w": 76, "h": 18},
         "text": "Buy now", "meta": {"role": "cta"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    assert len(tree) == 1
    frame = tree[0]
    assert frame["target"] == "group"
    assert frame["meta"]["role"] == "button"
    assert frame["meta"]["cornerRadius"] == 12
    assert frame["layout"]["mode"] == "HORIZONTAL"
    assert frame["layout"]["padding"] == {"left": 42, "right": 42, "top": 17, "bottom": 17}
    assert frame["layout"]["primaryAxisAlignItems"] == "CENTER"
    assert frame["layout"]["counterAxisAlignItems"] == "CENTER"
    assert frame["layout"]["itemSpacing"] == 0
    assert frame["children"][0]["id"] == "label"
    assert frame["children"][0]["box"]["x"] == 42
    assert frame["children"][0]["box"]["y"] == 17
    assert frame["children"][0]["layout"]["layoutAlign"] == "CENTER"


def test_inferred_cta_shell_becomes_button_frame():
    candidates = [
        {"id": "shell", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
         "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 24}},
        {"id": "label", "target": "text", "box": {"x": 60, "y": 44, "w": 80, "h": 24},
         "text": "Buy now", "meta": {"role": "cta"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    frame = tree[0]
    assert frame["meta"]["role"] == "button"
    assert frame["meta"]["cornerRadius"] == 24
    assert frame["layout"]["mode"] == "HORIZONTAL"
    assert frame["layout"]["counterAxisAlignItems"] == "CENTER"
    assert frame["children"][0]["id"] == "label"


def test_vertical_pill_button_uses_vertical_centered_layout():
    candidates = [
        {"id": "pill", "target": "shape", "box": {"x": 40, "y": 20, "w": 48, "h": 120},
         "fill": {"kind": "flat", "color": "#222222"}, "style": {"radius": 24},
         "meta": {"role": "button"}},
        {"id": "label", "target": "text", "box": {"x": 44, "y": 68, "w": 40, "h": 24},
         "text": "Go", "meta": {"role": "cta"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    frame = tree[0]
    assert frame["layout"]["mode"] == "VERTICAL"
    assert frame["layout"]["primaryAxisAlignItems"] == "CENTER"


def test_overlapping_artistic_children_do_not_force_auto_layout():
    candidates = [
        {"id": "card", "target": "shape", "box": {"x": 0, "y": 0, "w": 300, "h": 240}},
        {"id": "photo", "target": "image", "box": {"x": 20, "y": 20, "w": 180, "h": 180}},
        {"id": "badge", "target": "icon", "box": {"x": 150, "y": 120, "w": 90, "h": 90}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    assert tree[0]["target"] == "group"
    assert tree[0]["layout"]["mode"] == "NONE"


def test_contiguous_headline_and_subhead_become_a_native_text_stack():
    candidates = [
        {"id": "eyebrow", "target": "text", "box": {"x": 30, "y": 20, "w": 100, "h": 14},
         "text": "NEW", "meta": {"role": "eyebrow"}},
        {"id": "headline", "target": "text", "box": {"x": 30, "y": 40, "w": 260, "h": 42},
         "text": "Fresh arrivals", "meta": {"role": "headline"}},
        {"id": "body", "target": "text", "box": {"x": 30, "y": 91, "w": 250, "h": 24},
         "text": "Made for every day.", "meta": {"role": "body"}},
        {"id": "cta", "target": "text", "box": {"x": 300, "y": 210, "w": 70, "h": 20},
         "text": "Shop", "meta": {"role": "cta"}},
    ]

    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})

    stack = next(node for node in tree if node.get("meta", {}).get("role") == "text-stack")
    assert stack["layout"]["mode"] == "VERTICAL"
    assert [node["id"] for node in stack["children"]] == ["eyebrow", "headline", "body"]
    assert stack["children"][0]["box"]["x"] == 0
    assert stack["children"][1]["box"]["y"] == 20
    assert any(node["id"] == "cta" for node in tree)


def test_vertical_stack_emits_stretch_hints_for_full_width_children():
    candidates = [
        {"id": "panel", "target": "shape", "box": {"x": 0, "y": 0, "w": 300, "h": 180},
         "fill": {"kind": "flat", "color": "#ffffff"}, "meta": {"role": "card"}},
        {"id": "line1", "target": "text", "box": {"x": 10, "y": 20, "w": 280, "h": 24},
         "text": "Line one", "meta": {"role": "body"}},
        {"id": "line2", "target": "text", "box": {"x": 10, "y": 60, "w": 280, "h": 24},
         "text": "Line two", "meta": {"role": "body"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    frame = tree[0]
    assert frame["layout"]["mode"] == "VERTICAL"
    assert frame["children"][0]["layout"]["layoutGrow"] == 1
    assert frame["children"][0]["layout"]["layoutSizingHorizontal"] == "FILL"


def _benchmark_card(card_id, x):
    return [
        {"id": card_id, "target": "shape", "box": {"x": x, "y": 40, "w": 100, "h": 120},
         "fill": {"kind": "flat", "color": "#ffffff"}, "style": {"radius": 8}, "meta": {"role": "card"}, "z": 1},
        {"id": f"{card_id}-img", "target": "image", "box": {"x": x + 10, "y": 50, "w": 80, "h": 60}, "z": 2},
        {"id": f"{card_id}-title", "target": "text", "box": {"x": x + 10, "y": 118, "w": 80, "h": 20},
         "text": "Item", "meta": {"role": "title"}, "z": 3},
    ]


def test_repeated_cards_wrap_into_horizontal_grid():
    candidates = []
    for index, x in enumerate([10, 130, 250]):
        candidates.extend(_benchmark_card(f"card-{index}", x))
    tree = layout.infer(candidates, {"w": 400, "h": 220}, {})
    grid = next(node for node in tree if node.get("meta", {}).get("role") == "card-grid")
    assert grid["layout"]["mode"] == "HORIZONTAL"
    assert [child["id"] for child in grid["children"]] == ["card-0", "card-1", "card-2"]
    assert all(child.get("component") for child in grid["children"])


def test_implicit_triptych_panels_form_one_proven_panel_set():
    tree = layout.infer([
        {"id": f"panel-{index}", "target": "image",
         "box": {"x": 20 + index * 110, "y": 30, "w": 100, "h": 180},
         "src": f"panel-{index}.png", "meta": {"role": "triptych-panel"}}
        for index in range(3)
    ], {"w": 360, "h": 240})

    assert len(tree) == 1
    panel_set = tree[0]
    assert panel_set["meta"]["role"] == "panel-set"
    assert panel_set["meta"]["deterministic_geometry"] is True
    assert panel_set["layout"]["mode"] == "HORIZONTAL"
    assert [child["id"] for child in panel_set["children"]] == ["panel-0", "panel-1", "panel-2"]


def test_explicit_two_dimensional_card_grid_builds_native_rows():
    candidates = []
    for row in range(2):
        for col in range(3):
            candidates.append({
                "id": f"cell-{row}-{col}", "target": "image",
                "box": {"x": 20 + col * 96, "y": 30 + row * 86, "w": 84, "h": 72},
                "src": f"cell-{row}-{col}.png",
                "meta": {"role": "panel", "grid_group_id": "benefits"},
            })
    tree = layout.infer(candidates, {"w": 340, "h": 230})

    assert len(tree) == 1
    grid = tree[0]
    assert grid["meta"]["role"] == "structural-grid"
    assert grid["layout"]["mode"] == "VERTICAL"
    assert len(grid["children"]) == 2
    assert all(row["meta"]["role"] == "grid-row" for row in grid["children"])
    assert all(row["layout"]["mode"] == "HORIZONTAL" for row in grid["children"])
    assert [[cell["id"] for cell in row["children"]] for row in grid["children"]] == [
        ["cell-0-0", "cell-0-1", "cell-0-2"],
        ["cell-1-0", "cell-1-1", "cell-1-2"],
    ]


def test_simple_chart_primitives_group_without_changing_absolute_geometry():
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

    assert len(tree) == 1
    chart = tree[0]
    assert chart["meta"]["role"] == "native-chart"
    assert chart["layout"]["mode"] == "NONE"
    assert chart["meta"]["deterministic_geometry"] is True
    by_id = {child["id"]: child for child in chart["children"]}
    assert by_id["bar-a"]["box"] == {"x": 30, "y": 90, "w": 36, "h": 80}
    assert by_id["bar-b"]["box"] == {"x": 100, "y": 50, "w": 36, "h": 120}


def test_chart_with_raster_or_unknown_parts_is_not_invented_as_native_chart():
    candidates = [
        {"id": "axis", "target": "shape", "box": {"x": 10, "y": 100, "w": 180, "h": 2},
         "meta": {"role": "axis", "chart_group_id": "ambiguous"}},
        {"id": "plot", "target": "image", "box": {"x": 10, "y": 10, "w": 180, "h": 90},
         "src": "plot.png", "meta": {"role": "chart", "chart_group_id": "ambiguous",
                                         "intentional_raster_cluster": True}},
    ]
    tree = layout.infer(candidates, {"w": 220, "h": 130})

    assert not any(node.get("meta", {}).get("role") == "native-chart" for node in tree)
    assert any(node["id"] == "plot" for node in tree)


def test_uneven_panel_geometry_stays_absolute_instead_of_fake_auto_layout():
    tree = layout.infer([
        {"id": "one", "target": "image", "box": {"x": 10, "y": 20, "w": 90, "h": 180},
         "src": "one.png", "meta": {"role": "panel"}},
        {"id": "two", "target": "image", "box": {"x": 130, "y": 45, "w": 150, "h": 90},
         "src": "two.png", "meta": {"role": "panel"}},
    ], {"w": 320, "h": 240})

    assert len(tree) == 2
    assert not any(node.get("meta", {}).get("role") == "panel-set" for node in tree)


def test_card_panel_hoists_inner_background_fill():
    candidates = [
        {"id": "card", "target": "shape", "box": {"x": 0, "y": 0, "w": 200, "h": 160}, "meta": {"role": "card"}},
        {"id": "panel", "target": "shape", "box": {"x": 0, "y": 0, "w": 200, "h": 160},
         "fill": {"kind": "flat", "color": "#eeeeee"}, "style": {"radius": 12}, "z": 0},
        {"id": "label", "target": "text", "box": {"x": 20, "y": 60, "w": 160, "h": 24},
         "text": "Card", "meta": {"role": "title"}, "z": 1},
    ]
    tree = layout.infer(candidates, {"w": 300, "h": 220}, {})
    frame = tree[0]
    assert frame["fill"]["color"] == "#eeeeee"
    assert frame["radius"] == 12


def test_card_panel_hoist_preserves_multi_paint_and_shadow():
    fills = [
        {"kind": "linear", "angle": 90, "stops": [{"color": "#ff2200", "offset": 0}, {"color": "#0044ff", "offset": 1}]},
        {"kind": "flat", "color": "#ffffff", "opacity": 0.12},
    ]
    effects = [{"type": "drop-shadow", "color": "#00000066", "x": 0, "y": 3, "blur": 8}]
    tree = layout.infer([
        {"id": "card", "target": "shape", "box": {"x": 0, "y": 0, "w": 200, "h": 160}, "meta": {"role": "card"}},
        {"id": "panel", "target": "shape", "box": {"x": 0, "y": 0, "w": 200, "h": 160},
         "style": {"fills": fills, "effects": effects, "radius": 12}, "z": 0},
        {"id": "label", "target": "text", "box": {"x": 20, "y": 60, "w": 160, "h": 24},
         "text": "Card", "meta": {"role": "title"}, "z": 1},
    ], {"w": 300, "h": 220}, {})

    frame = tree[0]
    assert frame.get("fill") is None
    assert frame["style"]["fills"] == fills
    assert frame["effects"] == effects
    assert all(child["id"] != "panel" for child in frame["children"])


def test_group_hoist_drops_inner_button_shell():
  candidates = [
      {"id": "cta", "target": "group", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
       "meta": {"role": "button"}, "children": []},
      {"id": "pill", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
       "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 24}, "z": 0},
      {"id": "label", "target": "text", "box": {"x": 60, "y": 44, "w": 80, "h": 24},
       "text": "Buy now", "meta": {"role": "cta"}, "z": 1},
  ]
  # Re-run infer on flat list — group must be built by container detection or passed in.
  tree = layout.infer([
      {"id": "shell", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
       "meta": {"role": "button"}},
      {"id": "pill", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
       "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 24}, "z": 0},
      {"id": "label", "target": "text", "box": {"x": 60, "y": 44, "w": 80, "h": 24},
       "text": "Buy now", "meta": {"role": "cta"}, "z": 1},
  ], {"w": 400, "h": 300}, {})
  frame = tree[0]
  assert frame["fill"]["color"] == "#111111"
  assert all(child["id"] != "pill" for child in frame.get("children") or [])
  assert frame["children"][0]["id"] == "label"


def test_zero_placeholder_z_keeps_text_above_plate_and_cutout():
    tree = layout.infer([
        {"id": "plate", "target": "shape", "box": {"x": 0, "y": 0, "w": 100, "h": 100}, "z": 0},
        {"id": "cutout", "target": "image", "box": {"x": 10, "y": 10, "w": 50, "h": 50}, "z": 0},
        {"id": "copy", "target": "text", "box": {"x": 10, "y": 10, "w": 50, "h": 20}, "text": "Hi", "z": 0},
    ], {"w": 100, "h": 100})
    assert [node["id"] for node in tree] == ["plate", "cutout", "copy"]


def test_semantic_z_band_orders_placeholder_layers_before_geometry_fallbacks():
    tree = layout.infer([
        {"id": "content", "target": "image", "box": {"x": 0, "y": 0, "w": 80, "h": 80}, "z": 0,
         "meta": {"z_band": "content"}},
        {"id": "overlay", "target": "image", "box": {"x": 10, "y": 10, "w": 40, "h": 40}, "z": 0,
         "meta": {"z_band": "overlay"}},
        {"id": "chrome", "target": "image", "box": {"x": 20, "y": 20, "w": 20, "h": 20}, "z": 0,
         "meta": {"z_band": "chrome"}},
    ], {"w": 100, "h": 100})

    assert [node["id"] for node in tree] == ["content", "overlay", "chrome"]


def test_paragraph_lines_with_shared_block_id_form_one_editable_group():
    tree = layout.infer([
        {"id": "line1", "target": "text", "box": {"x": 20, "y": 20, "w": 100, "h": 12}, "text": "First", "meta": {"role": "body", "block_id": "p1"}},
        {"id": "line2", "target": "text", "box": {"x": 20, "y": 80, "w": 90, "h": 12}, "text": "Second", "meta": {"role": "body", "block_id": "p1"}},
    ], {"w": 200, "h": 150})
    assert len(tree) == 1
    assert tree[0]["target"] == "group"
    assert [child["id"] for child in tree[0]["children"]] == ["line1", "line2"]


def test_semantic_image_owner_keeps_overlay_in_named_asset_group():
    """Fusion's parent link keeps an avatar/screenshot and its overlay selectable together."""
    tree = layout.infer([
        {"id": "avatar", "target": "image", "box": {"x": 40, "y": 50, "w": 120, "h": 120},
         "meta": {"role": "avatar", "semantic_name": "Creator avatar"}},
        {"id": "online", "target": "icon", "box": {"x": 132, "y": 142, "w": 22, "h": 22},
         "meta": {"role": "icon", "parent_id": "avatar"}},
    ], {"w": 300, "h": 250})

    assert len(tree) == 1
    group = tree[0]
    assert group["id"] == "asset-group-avatar"
    assert group["name"] == "Creator avatar — asset group"
    assert [child["id"] for child in group["children"]] == ["avatar", "online"]
    assert group["children"][0]["box"]["x"] == 0
    assert group["children"][1]["box"]["x"] == 92


def test_intentional_raster_cluster_keeps_positive_overlay_in_named_asset_group():
    tree = layout.infer([
        {"id": "receipt", "target": "image", "box": {"x": 40, "y": 30, "w": 220, "h": 300},
         "meta": {"role": "receipt", "semantic_name": "Receipt", "intentional_raster_cluster": True}},
        {"id": "offer", "target": "text", "text": "Save 20%",
         "box": {"x": 70, "y": 50, "w": 120, "h": 24},
         "meta": {"role": "headline", "overlay_text": True, "external_overlay": True,
                  "parent_id": "receipt"}},
    ], {"w": 300, "h": 380})
    assert len(tree) == 1
    group = tree[0]
    assert group["id"] == "asset-group-receipt"
    assert [child["id"] for child in group["children"]] == ["receipt", "offer"]


def test_ui_label_in_pill_shell_groups_as_button_with_pill_radius():
    # Exact 009 "Volgend" geometry: a full pill shell (radius == h/2 from
    # reconstruct) with a centered UI label whose role is plain "label" (not
    # cta). The pair must still group as one button frame that carries the
    # pill radius and centers its text.
    candidates = [
        {"id": "c_E004", "target": "shape", "box": {"x": 833, "y": 134, "w": 202, "h": 67},
         "fill": {"kind": "flat", "color": "#eff3f4"}, "radius": 33.5,
         "meta": {"role": "button", "button_shell": True}},
        {"id": "c_B1", "target": "text", "box": {"x": 860, "y": 146, "w": 150, "h": 48},
         "text": "Volgend", "meta": {"role": "label"}},
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1080}, {})
    assert len(tree) == 1
    frame = tree[0]
    assert frame["target"] == "group"
    assert frame["meta"]["role"] == "button"
    assert frame["radius"] == 33.5
    assert frame["meta"]["cornerRadius"] == 33.5
    assert frame["layout"]["mode"] == "HORIZONTAL"
    assert frame["layout"]["primaryAxisAlignItems"] == "CENTER"
    assert frame["layout"]["counterAxisAlignItems"] == "CENTER"
    assert frame["children"][0]["id"] == "c_B1"
