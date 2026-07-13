"""CPU-only tests for optional VLM scene-text classification."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image  # noqa: E402

from src import merge_layers, vlm_client, vlm_scene_text  # noqa: E402


def _image(tmp_path):
    path = tmp_path / "ad.png"
    Image.new("RGB", (200, 120), (245, 240, 230)).save(path)
    return str(path)


def _ocr(lines):
    return {"lines": lines, "blocks": [], "styles": []}


def test_disabled_returns_ocr_unchanged(tmp_path):
    ocr = _ocr([{"id": "L0", "text": "SALE", "box": {"x": 10, "y": 10, "w": 80, "h": 30}}])
    out = vlm_scene_text.classify_scene_text(_image(tmp_path), ocr, {})
    assert out == ocr
    assert "vlm_scene_text" not in out


def test_classifies_line_and_sets_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vlm_client, "ask_vlm",
        lambda *a, **k: '{"placement":"printed","owner":"product","action":"raster_keep","confidence":0.9}',
    )
    ocr = _ocr([{"id": "L0", "text": "50ml", "box": {"x": 10, "y": 10, "w": 80, "h": 30}}])
    cfg = {"vlm": {"scene_text": {"enabled": True}}}
    out = vlm_scene_text.classify_scene_text(_image(tmp_path), ocr, cfg)
    assert out["lines"][0]["meta"]["scene_text_role"] == "printed_on_product"
    assert out["vlm_scene_text"]["lines_classified"] == 1


def test_vlm_disagreement_leaves_line_untagged(tmp_path, monkeypatch):
    answers = iter([
        '{"placement":"overlay","owner":"background","action":"recreate","confidence":0.9}',
        '{"placement":"printed","owner":"product","action":"raster_keep","confidence":0.9}',
    ])
    monkeypatch.setattr(
        vlm_client, "ask_vlm",
        lambda *a, **k: next(answers),
    )
    ocr = _ocr([{"id": "L0", "text": "SALE", "box": {"x": 10, "y": 10, "w": 80, "h": 30}}])
    cfg = {"vlm": {"scene_text": {"enabled": True}}}
    out = vlm_scene_text.classify_scene_text(_image(tmp_path), ocr, cfg)
    assert "scene_text_role" not in (out["lines"][0].get("meta") or {})
    assert out["lines"][0]["meta"]["ownership_decision"]["action"] == "raster_keep"
    assert out["vlm_scene_text"]["lines_disagreed"] == 1


def test_propagates_printed_role_to_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vlm_client, "ask_vlm",
        lambda *a, **k: '{"placement":"printed","owner":"product","action":"raster_keep","confidence":0.9}',
    )
    ocr = {
        "lines": [{"id": "L0", "text": "50ml", "box": {"x": 10, "y": 10, "w": 80, "h": 30}}],
        "blocks": [{
            "id": "B0", "type": "text", "line_ids": ["L0"], "text": "50ml",
            "box": {"x": 10, "y": 10, "w": 80, "h": 30},
        }],
        "styles": [],
    }
    out = vlm_scene_text.classify_scene_text(
        _image(tmp_path), ocr, {"vlm": {"scene_text": {"enabled": True}}},
    )
    assert out["blocks"][0]["meta"]["scene_text_role"] == "printed_on_product"


def test_merge_keeps_uncorroborated_vlm_printed_label_editable(tmp_path):
    ocr = {
        "lines": [{
            "id": "L0", "text": "50ml", "conf": 0.9,
            "box": {"x": 40, "y": 30, "w": 80, "h": 24},
            "meta": {"scene_text_role": "printed_on_product"},
        }],
    }
    cands = merge_layers.merge(ocr, [], [], {"w": 600, "h": 600}, {})
    text = next(c for c in cands if c["id"] == "c_L0")
    assert text["target"] == "text"
    assert text["meta"].get("scene_text_uncorroborated") is True


def test_merge_drops_printed_label_when_product_geometry_corroborates_it(tmp_path):
    ocr = {"lines": [{"id": "L0", "text": "50ml", "conf": .9,
            "box": {"x": 40, "y": 30, "w": 80, "h": 24},
            "meta": {"scene_text_role": "printed_on_product"}}]}
    elements = [{"id": "E0", "box": {"x": 20, "y": 10, "w": 150, "h": 100},
                 "kind": "photo-fragment", "role": "product"}]
    text = next(c for c in merge_layers.merge(ocr, elements, [], {"w": 600, "h": 600}, {})
                if c["id"] == "c_L0")
    assert text["target"] == "drop"
    assert text["meta"]["scene_text_corroborated"] is True


def test_merge_respects_overlay_copy_over_geometry(tmp_path):
    ocr = {
        "lines": [{
            "id": "L0", "text": "engraved on the watch", "conf": 0.8,
            "box": {"x": 320, "y": 400, "w": 150, "h": 22},
            "meta": {"scene_text_role": "overlay_copy"},
        }],
    }
    elements = [{
        "id": "E1", "box": {"x": 300, "y": 300, "w": 240, "h": 260},
        "kind": "photo-fragment", "role": "product",
    }]
    cands = merge_layers.merge(ocr, elements, [], {"w": 600, "h": 600}, {})
    text = next(c for c in cands if c["id"] == "c_L0")
    assert text["target"] == "text"


def test_parse_role_accepts_json_blob():
    assert vlm_scene_text._parse_role('{"role": "wordmark"}') == "wordmark"
    assert vlm_scene_text._parse_role("not json") is None


def test_ownership_protects_raster_owner_even_if_model_claims_recreate():
    decision = vlm_scene_text._parse_ownership(
        '{"placement":"overlay","owner":"card","action":"recreate","confidence":0.9}'
    )
    assert decision["action"] == "raster_keep"


def test_merge_enforces_ownership_before_geometry():
    ocr = {"lines": [{
        "id": "L0", "text": "Cadence", "conf": .95,
        "box": {"x": 100, "y": 100, "w": 80, "h": 30},
        "meta": {"ownership_decision": {
            "placement": "printed", "owner": "product",
            "action": "raster_keep", "confidence": .95,
        }},
    }]}
    text = next(c for c in merge_layers.merge(ocr, [], [], {"w": 600, "h": 600}, {})
                if c["id"] == "c_L0")
    assert text["target"] == "drop"
    assert text["meta"]["ownership_enforced"] is True


def test_two_views_may_disagree_on_placement_when_safe_action_agrees(tmp_path, monkeypatch):
    answers = iter([
        '{"placement":"printed","owner":"none","action":"recreate","confidence":0.9}',
        '{"placement":"overlay","owner":"background","action":"recreate","confidence":1.0}',
    ])
    monkeypatch.setattr(vlm_client, "ask_vlm", lambda *a, **k: next(answers))
    ocr = _ocr([{"id": "L0", "text": "Post body",
                 "box": {"x": 10, "y": 10, "w": 80, "h": 30}}])
    out = vlm_scene_text.classify_scene_text(
        _image(tmp_path), ocr, {"vlm": {"scene_text": {"enabled": True, "passes": 2}}},
    )
    decision = out["lines"][0]["meta"]["ownership_decision"]
    assert decision["action"] == "recreate"
    assert decision["confidence"] == 0.9


def test_block_two_thirds_consensus_recreates_without_protected_owner():
    lines = []
    for index, action in enumerate(("recreate", "recreate", "raster_keep")):
        lines.append({"id": f"L{index}", "meta": {"ownership_decision": {
            "placement": "overlay", "owner": "none", "action": action,
            "confidence": .8 if action == "recreate" else 0,
        }}})
    blocks = [{"id": "B0", "line_ids": ["L0", "L1", "L2"]}]
    vlm_scene_text._propagate_to_blocks(lines, blocks)
    assert blocks[0]["meta"]["ownership_decision"]["action"] == "recreate"
    assert blocks[0]["meta"]["ownership_decision"]["block_consensus"] == "2/3"
