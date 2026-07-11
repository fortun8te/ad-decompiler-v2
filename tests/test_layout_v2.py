from src import layout


def test_button_becomes_native_frame_with_relative_child():
    candidates = [
        {"id": "button", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
         "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 12},
         "meta": {"role": "button"}},
        {"id": "label", "target": "text", "box": {"x": 60, "y": 44, "w": 80, "h": 24},
         "text": "Buy now", "meta": {"role": "cta"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    assert len(tree) == 1
    frame = tree[0]
    assert frame["target"] == "group"
    assert frame["layout"]["mode"] == "HORIZONTAL"
    assert frame["children"][0]["box"]["x"] == 40
    assert frame["children"][0]["box"]["y"] == 14


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
