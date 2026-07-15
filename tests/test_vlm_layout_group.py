"""CPU-only tests for the advisory VLM semantic grouping pass (mocked — no LM Studio)."""
import json

from PIL import Image

from src import vlm_layout_group

CANVAS = {"w": 400, "h": 600}


def _roots():
    return [
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


def _cfg(tmp_path, **vg):
    Image.new("RGB", (CANVAS["w"], CANVAS["h"]), (240, 235, 220)).save(tmp_path / "normalized.png")
    return {
        "vlm": {"enabled": True, "base_url": "http://127.0.0.1:1234/v1", "model": "test-model"},
        "layout": {"vlm_grouping": {"enabled": True, "min_elements": 3, **vg}},
        "run_dir": str(tmp_path),
    }


def _spec(groups, element_names=None):
    return json.dumps({"groups": groups, "element_names": element_names or []})


def _flat_ids(nodes):
    out = []
    for node in nodes:
        children = node.get("children") or []
        if children:
            out.extend(_flat_ids(children))
        else:
            out.append(node["id"])
    return sorted(out)


def test_disabled_makes_no_call(monkeypatch):
    calls = []
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm",
                        lambda *a, **k: calls.append(1) or "[]")
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, {})
    assert out == roots
    assert info == {"applied": False, "reason": "disabled", "groups_added": 0, "names_applied": 0}
    assert calls == []


def test_too_few_elements_skips_without_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm",
                        lambda *a, **k: calls.append(1) or "{}")
    roots = _roots()[:2]
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"] == "too-few-elements"
    assert calls == []


def test_missing_image_degrades_without_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm",
                        lambda *a, **k: calls.append(1) or "{}")
    cfg = _cfg(tmp_path)
    (tmp_path / "normalized.png").unlink()
    out, info = vlm_layout_group.regroup(_roots(), CANVAS, cfg)
    assert info["reason"] == "no-image"
    assert calls == []


def test_valid_grouping_adds_named_wrappers_without_moving_leaves(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "menu"]},
        {"id": "g2", "name": "product hero", "direction": "column",
         "member_ids": ["product", "headline"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    original_boxes = {node["id"]: dict(node["box"]) for node in roots}

    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))

    assert info["applied"] is True
    assert info["groups_added"] == 2
    assert _flat_ids(out) == sorted(original_boxes)          # nothing lost, nothing duplicated
    by_name = {node.get("name"): node for node in out if node.get("target") == "group"}
    assert set(by_name) == {"Header", "Product hero"}
    header = by_name["Header"]
    assert [child["id"] for child in header["children"]] == ["logo", "menu"]
    assert header["meta"]["source"] == "vlm-grouping"
    assert header["meta"]["semantic_name"] == "Header"
    assert header["meta"]["vlm_direction_hint"] == "row"
    assert header["box"] == {"x": 20.0, "y": 20.0, "w": 350.0, "h": 30.0}
    for node in out:
        for child in node.get("children") or []:
            assert child["box"] == original_boxes[child["id"]]  # leaves never move/resize
    assert any(node["id"] == "cta" for node in out)          # ungrouped element stays at root


def test_nested_groups_build_nested_wrappers(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "menu"]},
        {"id": "g2", "name": "top", "direction": "column", "member_ids": ["g1", "product"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)

    out, info = vlm_layout_group.regroup(_roots(), CANVAS, _cfg(tmp_path))

    assert info["applied"] is True
    top = next(node for node in out if node.get("name") == "Top")
    inner_names = {child.get("name") for child in top["children"]}
    assert "Header" in inner_names
    assert {child["id"] for child in top["children"] if child.get("target") != "group"} == {"product"}
    assert _flat_ids(out) == sorted(node["id"] for node in _roots())


def test_duplicate_membership_is_rejected_whole(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "menu"]},
        {"id": "g2", "name": "hero", "direction": "column", "member_ids": ["logo", "product"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["applied"] is False
    assert info["reason"] == "vlm-invalid:duplicate-member:logo"


def test_unknown_member_is_rejected(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "ghost"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"] == "vlm-invalid:unknown-member:ghost"


def test_overlapping_groups_are_rejected(tmp_path, monkeypatch):
    # Interleaved groups: both bbox unions cover the middle of the canvas.
    payload = _spec([
        {"id": "g1", "name": "a", "direction": "none", "member_ids": ["logo", "product"]},
        {"id": "g2", "name": "b", "direction": "none", "member_ids": ["menu", "headline"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"].startswith("vlm-invalid:groups-overlap")


def test_group_bbox_capturing_a_nonmember_is_rejected(tmp_path, monkeypatch):
    # logo+headline union spans y=20..420 and swallows the product entirely.
    payload = _spec([
        {"id": "g1", "name": "copy", "direction": "column", "member_ids": ["logo", "headline"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"] == "vlm-invalid:captures-nonmember:g1~product"


def test_cyclic_groups_are_rejected(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "a", "direction": "none", "member_ids": ["g2"]},
        {"id": "g2", "name": "b", "direction": "none", "member_ids": ["g1"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"] == "vlm-invalid:cyclic-groups"


def test_too_many_groups_are_rejected(tmp_path, monkeypatch):
    payload = _spec([
        {"id": f"g{i}", "name": "x", "direction": "none", "member_ids": []}
        for i in range(3)
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path, max_groups=2))
    assert out == roots
    assert info["reason"] == "vlm-invalid:too-many-groups:3"


def test_single_member_group_becomes_a_name_not_a_wrapper(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "product hero", "direction": "none", "member_ids": ["product"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    out, info = vlm_layout_group.regroup(_roots(), CANVAS, _cfg(tmp_path))
    assert info["applied"] is True
    assert info["groups_added"] == 0
    assert info["names_applied"] == 1
    product = next(node for node in out if node["id"] == "product")
    assert product["meta"]["semantic_name"] == "Product hero"
    assert product["meta"]["vlm_named"] is True


def test_element_names_apply_only_when_absent(tmp_path, monkeypatch):
    payload = _spec([], element_names=[
        {"id": "logo", "name": "brand logo"},
        {"id": "product", "name": "hero shot"},
        {"id": "ghost", "name": "ignored"},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots()
    roots[2]["meta"]["semantic_name"] = "Product cutout"   # pre-existing name must win

    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))

    assert info["applied"] is True
    assert info["names_applied"] == 1
    by_id = {node["id"]: node for node in out}
    assert by_id["logo"]["meta"]["semantic_name"] == "Brand logo"
    assert by_id["product"]["meta"]["semantic_name"] == "Product cutout"


def test_vlm_error_degrades_to_deterministic_tree(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("model evicted")

    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", _raise)
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info == {"applied": False, "reason": "vlm-error", "groups_added": 0, "names_applied": 0}


def test_unparseable_answer_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm",
                        lambda *a, **k: "I would group the header items together.")
    roots = _roots()
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert out == roots
    assert info["reason"] == "vlm-parse-error"


def test_background_plate_does_not_trigger_capture_rejection(tmp_path, monkeypatch):
    payload = _spec([
        {"id": "g1", "name": "header", "direction": "row", "member_ids": ["logo", "menu"]},
    ])
    monkeypatch.setattr(vlm_layout_group.vlm_client, "ask_vlm", lambda *a, **k: payload)
    roots = _roots() + [{
        "id": "plate", "target": "shape",
        "box": {"x": 0, "y": 0, "w": 400, "h": 600},
        "meta": {"role": "background"}, "z": -1000000,
    }]
    out, info = vlm_layout_group.regroup(roots, CANVAS, _cfg(tmp_path))
    assert info["applied"] is True
    assert any(node["id"] == "plate" for node in out)
