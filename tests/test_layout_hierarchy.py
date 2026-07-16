"""CPU tests for deterministic deeper nesting, auto-layout precision, repeats, naming."""
import json

from PIL import Image

from src import build_design_json, layout, scene_intent, vlm_layout_group
from src.schema import validate_design

CANVAS = {"w": 400, "h": 600}


def _banded_candidates():
    """Header band, hero image, and a CTA row separated by real whitespace."""
    return [
        {"id": "logo", "target": "icon", "box": {"x": 20, "y": 20, "w": 60, "h": 30},
         "meta": {"role": "logo"}, "z": 3},
        {"id": "badge", "target": "icon", "box": {"x": 320, "y": 22, "w": 50, "h": 26},
         "meta": {"role": "badge"}, "z": 3},
        {"id": "product", "target": "image", "box": {"x": 100, "y": 150, "w": 200, "h": 260},
         "meta": {"role": "product"}, "z": 2},
        {"id": "btn-a", "target": "shape", "box": {"x": 40, "y": 520, "w": 80, "h": 40},
         "meta": {"role": "button"}, "z": 3},
        {"id": "btn-b", "target": "shape", "box": {"x": 160, "y": 520, "w": 80, "h": 40},
         "meta": {"role": "button"}, "z": 3},
        {"id": "btn-c", "target": "shape", "box": {"x": 280, "y": 520, "w": 80, "h": 40},
         "meta": {"role": "button"}, "z": 3},
    ]


def _flat_ids(nodes):
    out = []
    for node in nodes:
        children = node.get("children") or []
        if children:
            out.extend(_flat_ids(children))
        else:
            out.append(node["id"])
    return sorted(out)


def test_xycut_wraps_whitespace_separated_bands():
    tree = layout.infer(_banded_candidates(), CANVAS, {})

    assert _flat_ids(tree) == sorted(node["id"] for node in _banded_candidates())
    bands = [node for node in tree if (node.get("meta") or {}).get("role") == "band"]
    assert len(bands) == 2
    by_name = {node["meta"]["semantic_name"]: node for node in bands}
    assert set(by_name) == {"Header", "CTA"}
    assert [child["id"] for child in by_name["Header"]["children"]] == ["badge", "logo"]
    cta = by_name["CTA"]
    assert [child["id"] for child in cta["children"]] == ["btn-a", "btn-b", "btn-c"]
    assert cta["meta"]["deterministic_geometry"] is True
    # The lone hero image between the bands stays a bare root layer.
    assert any(node["id"] == "product" for node in tree)


def test_xycut_band_gets_precise_auto_layout():
    tree = layout.infer(_banded_candidates(), CANVAS, {})
    cta = next(node for node in tree
               if (node.get("meta") or {}).get("semantic_name") == "CTA")
    assert cta["layout"]["mode"] == "HORIZONTAL"
    assert cta["layout"]["itemSpacing"] == 40
    assert cta["layout"]["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
    # Children were relativized against the band frame.
    assert cta["children"][0]["box"]["x"] == 0
    assert cta["children"][0]["box"]["y"] == 0
    assert cta["children"][0]["meta"]["absolute_box"]["x"] == 40


def test_xycut_stays_flat_below_min_nodes():
    tree = layout.infer(_banded_candidates()[:4], CANVAS, {})
    assert not any(str(node.get("id", "")).startswith("band-") for node in tree)


def test_xycut_stays_flat_without_a_clean_gap():
    # Nodes tiled without any whitespace corridor >= 5% of the canvas.
    candidates = [
        {"id": f"n{i}", "target": "icon",
         "box": {"x": 20 + (i % 3) * 125, "y": 20 + (i // 3) * 95, "w": 120, "h": 90},
         "meta": {"role": "icon"}, "z": 2}
        for i in range(6)
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 220}, {})
    assert not any(str(node.get("id", "")).startswith("band-") for node in tree)


def test_xycut_respects_config_gate():
    cfg = {"layout": {"nesting": {"enabled": False}}}
    tree = layout.infer(_banded_candidates(), CANVAS, cfg)
    assert not any(str(node.get("id", "")).startswith("band-") for node in tree)


def test_background_plate_is_never_pulled_into_a_band():
    candidates = _banded_candidates() + [{
        "id": "plate", "target": "shape",
        "box": {"x": 0, "y": 0, "w": 400, "h": 600},
        "meta": {"role": "background"}, "z": -1000000,
    }]
    tree = layout.infer(candidates, CANVAS, {})
    assert any(node["id"] == "plate" and not node.get("children") for node in tree)


def test_counter_alignment_detects_right_aligned_column():
    container = {"box": {"x": 0, "y": 0, "w": 200, "h": 300}, "meta": {}}
    children = [
        {"id": "a", "target": "shape", "box": {"x": 80, "y": 20, "w": 100, "h": 18}},
        {"id": "b", "target": "shape", "box": {"x": 84, "y": 58, "w": 96, "h": 18}},
        {"id": "c", "target": "shape", "box": {"x": 90, "y": 96, "w": 90, "h": 18}},
    ]
    result = layout.infer_auto_layout(container, children)
    assert result["mode"] == "VERTICAL"
    assert result["counterAlign"] == "MAX"
    assert result["counterAxisAlignItems"] == "MAX"
    assert result["gap"] == 20
    assert result["padding"] == {"left": 80, "right": 20, "top": 20, "bottom": 186}


def test_counter_alignment_detects_top_aligned_row():
    container = {"box": {"x": 0, "y": 40, "w": 160, "h": 60}, "meta": {}}
    children = [
        {"id": "a", "target": "shape", "box": {"x": 10, "y": 50, "w": 40, "h": 30}},
        {"id": "b", "target": "shape", "box": {"x": 60, "y": 50, "w": 40, "h": 38}},
        {"id": "c", "target": "shape", "box": {"x": 110, "y": 50, "w": 40, "h": 26}},
    ]
    result = layout.infer_auto_layout(container, children)
    assert result["mode"] == "HORIZONTAL"
    assert result["counterAlign"] == "MIN"
    assert result["gap"] == 10


def test_item_spacing_snaps_subpixel_noise_to_integers():
    assert layout._item_spacing([11.9, 12.1]) == 12
    assert layout._item_spacing([12.4]) == 12
    assert layout._item_spacing([]) == 0
    assert layout._item_spacing([10.0, 14.0, 12.0]) == 12


def test_repeated_leaf_elements_get_component_candidate_metadata():
    candidates = [
        {"id": f"star-{i}", "target": "icon",
         "box": {"x": 40 + i * 30, "y": 100, "w": 24, "h": 24},
         "meta": {"role": "icon"}, "z": 3}
        for i in range(4)
    ]
    tree = layout.infer(candidates, {"w": 300, "h": 200}, {})
    stars = [node for node in tree if node["id"].startswith("star-")]
    assert len(stars) == 4
    for star in stars:
        candidate = star["meta"]["component_candidate"]
        assert candidate["key"].startswith("leafrep~")
        assert candidate["count"] == 4
        assert candidate["members"] == sorted(f"star-{i}" for i in range(4))


def test_structurally_repeated_cards_with_different_copy_become_candidates():
    def _card(card_id, x, text):
        return [
            {"id": card_id, "target": "shape", "box": {"x": x, "y": 10, "w": 100, "h": 120},
             "fill": {"kind": "flat", "color": "#ffffff"}, "meta": {"role": "card"}, "z": 1},
            {"id": f"{card_id}-title", "target": "text", "text": text,
             "box": {"x": x + 10, "y": 90, "w": 80, "h": 20}, "meta": {"role": "title"}, "z": 2},
        ]

    candidates = _card("card-a", 10, "Alpha") + _card("card-b", 150, "Totally different words")
    tree = layout.infer(candidates, {"w": 300, "h": 150}, {})
    cards = [node for node in tree if node.get("target") == "group"]
    assert len(cards) == 2
    for card in cards:
        assert not card.get("component")       # exact repeat instantiation untouched
        candidate = card["meta"]["component_candidate"]
        assert candidate["key"].startswith("repeat~")
        assert candidate["count"] == 2
        assert candidate["confidence"] == 0.75


def test_repeat_candidates_respect_config_gate():
    candidates = [
        {"id": f"star-{i}", "target": "icon",
         "box": {"x": 40 + i * 30, "y": 100, "w": 24, "h": 24},
         "meta": {"role": "icon"}, "z": 3}
        for i in range(4)
    ]
    cfg = {"layout": {"repeats": {"enabled": False}}}
    tree = layout.infer(candidates, {"w": 300, "h": 200}, cfg)
    assert all("component_candidate" not in node["meta"] for node in tree)


def test_button_and_text_stack_frames_get_semantic_names():
    candidates = [
        {"id": "shell", "target": "shape", "box": {"x": 20, "y": 230, "w": 160, "h": 52},
         "fill": {"kind": "flat", "color": "#111111"}, "style": {"radius": 12},
         "meta": {"role": "button"}},
        {"id": "label", "target": "text", "box": {"x": 60, "y": 244, "w": 80, "h": 24},
         "text": "Buy now", "meta": {"role": "cta"}},
        {"id": "eyebrow", "target": "text", "box": {"x": 30, "y": 20, "w": 100, "h": 14},
         "text": "NEW", "meta": {"role": "eyebrow"}},
        {"id": "headline", "target": "text", "box": {"x": 30, "y": 40, "w": 260, "h": 42},
         "text": "Fresh arrivals", "meta": {"role": "headline"}},
    ]
    tree = layout.infer(candidates, {"w": 400, "h": 300}, {})
    button = next(node for node in tree if (node.get("meta") or {}).get("role") == "button")
    assert button["meta"]["semantic_name"] == "Button / Buy now"
    stack = next(node for node in tree if (node.get("meta") or {}).get("role") == "text-stack")
    assert stack["meta"]["semantic_name"] == "Text Stack"


def test_explicit_names_survive_the_naming_pass():
    candidates = [
        {"id": "avatar", "target": "image", "box": {"x": 40, "y": 50, "w": 120, "h": 120},
         "meta": {"role": "avatar", "semantic_name": "Creator avatar"}},
        {"id": "online", "target": "icon", "box": {"x": 132, "y": 142, "w": 22, "h": 22},
         "meta": {"role": "icon", "parent_id": "avatar"}},
    ]
    tree = layout.infer(candidates, {"w": 300, "h": 250}, {})
    assert tree[0]["name"] == "Creator avatar"


def test_band_tree_compiles_through_the_design_schema(tmp_path):
    # Icon/shape-only fixture: no image assets required, so compilation must be warning-free.
    candidates = _banded_candidates()
    candidates[2] = {"id": "hero-icon", "target": "icon",
                     "box": {"x": 100, "y": 150, "w": 200, "h": 260},
                     "meta": {"role": "illustration"}, "z": 2}
    tree = layout.infer(candidates, CANVAS, {})
    doc = build_design_json.build(tree, CANVAS, str(tmp_path), base_src=None)
    assert validate_design(doc) == []
    assert doc.meta["warnings"] == []
    names = {layer.name for layer in doc.layers}
    assert "Header" in names
    assert "CTA" in names
    groups = [layer for layer in doc.layers if layer.type == "group"]
    assert all(layer.children for layer in groups)


def test_scene_intent_persists_the_vlm_grouping_outcome(tmp_path, monkeypatch):
    candidates = _banded_candidates()
    Image.new("RGB", (CANVAS["w"], CANVAS["h"]), "white").save(tmp_path / "normalized.png")
    cfg = {
        "vlm": {"enabled": True},
        "layout": {"vlm_grouping": {"enabled": True, "min_elements": 3}},
        "run_dir": str(tmp_path),
    }

    def _raise(*args, **kwargs):
        raise RuntimeError("model evicted for VRAM")

    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", _raise)
    intent = scene_intent.plan(candidates, CANVAS, cfg)
    assert intent["vlm_grouping"]["applied"] is False
    assert intent["vlm_grouping"]["reason"] == "vlm-error"
    # The persisted JSON keeps the notice and the intent stays reusable.
    payload = json.loads(json.dumps(intent))
    assert payload["vlm_grouping"]["reason"] == "vlm-error"
    assert scene_intent.is_current(payload, candidates, CANVAS, cfg)

    disabled = scene_intent.plan(candidates, CANVAS, {})
    assert "vlm_grouping" not in disabled


def test_vlm_grouping_flows_through_layout_and_hydration(tmp_path, monkeypatch):
    candidates = [
        {"id": "logo", "target": "icon", "box": {"x": 20, "y": 20, "w": 60, "h": 30},
         "meta": {"role": "logo"}, "z": 3},
        {"id": "menu", "target": "text", "text": "Menu",
         "box": {"x": 320, "y": 25, "w": 50, "h": 20}, "meta": {"role": "text"}, "z": 3},
        {"id": "product", "target": "image", "box": {"x": 100, "y": 150, "w": 200, "h": 200},
         "meta": {"role": "product"}, "z": 2},
        {"id": "headline", "target": "text", "text": "Big Sale",
         "box": {"x": 100, "y": 380, "w": 200, "h": 40}, "meta": {"role": "headline"}, "z": 3},
        {"id": "cta", "target": "shape", "box": {"x": 140, "y": 520, "w": 120, "h": 44},
         "meta": {"role": "button"}, "z": 3},
    ]
    Image.new("RGB", (CANVAS["w"], CANVAS["h"]), "white").save(tmp_path / "normalized.png")
    cfg = {
        "vlm": {"enabled": True},
        "layout": {"vlm_grouping": {"enabled": True, "min_elements": 3}},
        "run_dir": str(tmp_path),
    }
    payload = json.dumps({
        "groups": [
            {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "menu"]},
            {"id": "g2", "name": "product hero", "direction": "column",
             "member_ids": ["product", "headline"]},
        ],
        "element_names": [],
    })
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)

    intent = scene_intent.plan(candidates, CANVAS, cfg)

    assert intent["vlm_grouping"]["applied"] is True
    assert intent["vlm_grouping"]["groups_added"] == 2
    synthetic = set(intent["synthetic_ids"])
    assert len(synthetic) == 2 and all(value.startswith("vlm-group-") for value in synthetic)
    hero = next(node for node in intent["tree"] if node.get("name") == "Product hero")
    # Direction hint agrees with measured geometry -> evidence-gated Auto Layout fires.
    assert hero["layout"]["mode"] == "VERTICAL"
    assert hero["meta"]["vlm_direction_agrees"] is True
    assert hero["meta"]["layout_confidence"] == hero["layout"]["confidence"]
    # Children are relativized against the wrapper exactly like deterministic frames.
    assert hero["children"][0]["id"] == "product"
    assert hero["children"][0]["box"]["x"] == 0
    assert hero["children"][0]["meta"]["absolute_box"] == {"x": 100, "y": 150, "w": 200, "h": 200}

    hydrated = scene_intent.hydrate(intent, {"candidates": candidates})
    hero = next(node for node in hydrated if node.get("name") == "Product hero")
    assert [child["id"] for child in hero["children"]] == ["product", "headline"]
    assert hero["meta"]["scene_intent_synthetic"] is True
