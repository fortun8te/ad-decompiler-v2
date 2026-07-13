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


def test_zero_placeholder_z_keeps_text_above_plate_and_cutout():
    tree = layout.infer([
        {"id": "plate", "target": "shape", "box": {"x": 0, "y": 0, "w": 100, "h": 100}, "z": 0},
        {"id": "cutout", "target": "image", "box": {"x": 10, "y": 10, "w": 50, "h": 50}, "z": 0},
        {"id": "copy", "target": "text", "box": {"x": 10, "y": 10, "w": 50, "h": 20}, "text": "Hi", "z": 0},
    ], {"w": 100, "h": 100})
    assert [node["id"] for node in tree] == ["plate", "cutout", "copy"]


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
