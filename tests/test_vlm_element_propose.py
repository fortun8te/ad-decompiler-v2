"""CPU-only tests for optional VLM element proposal before SAM."""
import json

from PIL import Image

from src import vlm_element_propose


def _image(tmp_path, size=(200, 160)):
    path = tmp_path / "ad.png"
    Image.new("RGB", size, (240, 235, 220)).save(path)
    return str(path)


def test_disabled_returns_residual_unchanged(tmp_path):
    residual = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}, "kind": "shape"}]
    out = vlm_element_propose.enrich_residual(_image(tmp_path), residual, {})
    assert out == residual


def test_adds_role_tagged_proposal_when_vlm_agrees(tmp_path, monkeypatch):
    payload = json.dumps([{
        "label": "button",
        "approx_box_fraction": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.25},
    }])

    def fake_ask(*args, **kwargs):
        return payload

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    residual = [{"id": "E0", "box": {"x": 150, "y": 120, "w": 30, "h": 20}, "kind": "shape"}]
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 1}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), residual, cfg)
    assert len(out) == 2
    added = [item for item in out if item.get("source") == "vlm-propose"]
    assert len(added) == 1
    assert added[0]["role"] == "button"
    assert added[0]["kind"] == "shape"
    assert added[0]["id"].startswith("VP")


def test_vlm_error_degrades_without_adding(tmp_path, monkeypatch):
    def fake_ask(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    residual = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}}]
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 2}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), residual, cfg)
    assert out == residual


def test_disagreement_on_count_skips_tile(tmp_path, monkeypatch):
    answers = [
        json.dumps([{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]),
        json.dumps([]),
    ]
    state = {"i": 0}

    def fake_ask(*args, **kwargs):
        raw = answers[state["i"] % len(answers)]
        state["i"] += 1
        return raw

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    residual = []
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 1, "passes": 2}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), residual, cfg)
    assert out == []


def test_dedupes_overlapping_vlm_boxes(tmp_path, monkeypatch):
    payload = json.dumps([
        {"label": "product", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4}},
        {"label": "product", "approx_box_fraction": {"x": 0.12, "y": 0.12, "w": 0.38, "h": 0.38}},
    ])

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", lambda *a, **k: payload)
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 1}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), [], cfg)
    assert len([item for item in out if item.get("source") == "vlm-propose"]) == 1


def test_two_pass_accepts_matching_boxes_with_different_labels():
    a = [{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]
    b = [{"label": "button", "approx_box_fraction": {"x": 0.11, "y": 0.11, "w": 0.2, "h": 0.2}}]
    assert vlm_element_propose._proposals_agree(a, b) is True


def test_two_pass_rejects_when_counts_differ():
    a = [{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]
    b = []
    assert vlm_element_propose._proposals_agree(a, b) is False


def test_two_pass_rejects_when_boxes_dont_match():
    a = [{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]
    b = [{"label": "icon", "approx_box_fraction": {"x": 0.7, "y": 0.7, "w": 0.2, "h": 0.2}}]
    assert vlm_element_propose._proposals_agree(a, b) is False


def test_tolerant_consensus_accepts_boxes_within_tolerance_but_different_labels(tmp_path, monkeypatch):
    answers = [
        json.dumps([{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}}]),
        json.dumps([{"label": "button", "approx_box_fraction": {"x": 0.11, "y": 0.11, "w": 0.3, "h": 0.3}}]),
    ]
    state = {"i": 0}

    def fake_ask(*args, **kwargs):
        raw = answers[state["i"] % len(answers)]
        state["i"] += 1
        return raw

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 1, "passes": 2}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), [], cfg)
    added = [item for item in out if item.get("source") == "vlm-propose"]
    assert len(added) == 1
    # First pass's original label/proposals are what get kept.
    assert added[0]["role"] == "icon"


def test_degradation_signal_present_on_vlm_error(tmp_path, monkeypatch):
    def fake_ask(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    residual = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}}]
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 2}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), residual, cfg)
    assert out == residual
    assert getattr(out, "vlm_degraded", None) is not None
    assert out.vlm_degraded["reason"] == "vlm_error"
    assert out.vlm_degraded["tile_count"] > 0


def test_degradation_signal_present_on_disagreement(tmp_path, monkeypatch):
    answers = [
        json.dumps([{"label": "icon", "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]),
        json.dumps([]),
    ]
    state = {"i": 0}

    def fake_ask(*args, **kwargs):
        raw = answers[state["i"] % len(answers)]
        state["i"] += 1
        return raw

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    cfg = {"vlm": {"element_propose": {"enabled": True, "grid": 2, "max_tiles": 1, "passes": 2}}}
    out = vlm_element_propose.enrich_residual(_image(tmp_path), [], cfg)
    assert out == []
    assert getattr(out, "vlm_degraded", None) is not None
    assert out.vlm_degraded["reason"] == "vlm_disagreement"


def test_fraction_to_pixel_helpers():
    tile = {"x": 100, "y": 50, "w": 200, "h": 100}
    frac = {"x": 0.25, "y": 0.5, "w": 0.5, "h": 0.25}
    box = vlm_element_propose._fraction_to_pixel(frac, tile, 400, 300)
    assert box == {"x": 150, "y": 100, "w": 100, "h": 25}


def test_intentional_raster_cluster_labels_are_accepted_as_photo_proposals():
    parsed = vlm_element_propose._parse_proposals(json.dumps([{
        "label": "nutrition_panel",
        "approx_box_fraction": {"x": .1, "y": .2, "w": .5, "h": .4},
    }]))
    assert parsed == [{"label": "nutrition_panel",
                       "approx_box_fraction": {"x": .1, "y": .2, "w": .5, "h": .4}}]
    assert vlm_element_propose._LABEL_TO_KIND[parsed[0]["label"]] == "photo-fragment"


def test_multi_panel_labels_are_accepted_as_photo_proposals():
    parsed = vlm_element_propose._parse_proposals(json.dumps([
        {"label": "panel", "approx_box_fraction": {"x": .0, "y": .1, "w": .3, "h": .8}},
        {"label": "comparison_panel", "approx_box_fraction": {"x": .5, "y": .1, "w": .5, "h": .8}},
    ]))
    assert [item["label"] for item in parsed] == ["panel", "comparison_panel"]
    assert all(vlm_element_propose._LABEL_TO_KIND[item["label"]] == "photo-fragment"
               for item in parsed)


def test_lightweight_grid_when_sam_count_below_threshold(tmp_path, monkeypatch):
    payload = json.dumps([{
        "label": "icon",
        "approx_box_fraction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
    }])
    seen = {"grid": None, "max_tiles": None, "tile_mode": None}

    def fake_ask(*args, **kwargs):
        return payload

    def fake_grid(width, height, grid, overlap):
        seen["grid"] = grid
        seen["tile_mode"] = "grid"
        return [{"x": 0, "y": 0, "w": width, "h": height}]

    monkeypatch.setattr(vlm_element_propose.vlm_client, "ask_vlm", fake_ask)
    monkeypatch.setattr(vlm_element_propose, "_grid_tiles", fake_grid)
    cfg = {
        "vlm": {
            "element_propose": {
                "enabled": True,
                "grid": 4,
                "max_tiles": 20,
                "lightweight_grid_below_sam_count": 3,
            }
        }
    }
    out = vlm_element_propose.enrich_residual(_image(tmp_path), [], cfg, sam_element_count=1)
    assert len(out) == 1
    assert seen["grid"] == 2
    assert seen["tile_mode"] == "grid"
