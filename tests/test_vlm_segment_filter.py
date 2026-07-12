"""CPU-only tests for optional VLM segment filtering."""
import json

from PIL import Image

from src import vlm_segment_filter


def _image(tmp_path):
    path = tmp_path / "ad.png"
    Image.new("RGB", (120, 80), (240, 235, 220)).save(path)
    return str(path)


def test_disabled_returns_elements_unchanged(tmp_path):
    elements = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}}]
    out = vlm_segment_filter.filter_elements(_image(tmp_path), elements, {})
    assert out == elements


def test_drop_removes_element_when_vlm_agrees(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vlm_segment_filter.vlm_client, "multi_pass_answer",
        lambda *a, **k: ('{"decision": "drop", "label": "junk"}', None),
    )
    elements = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}}]
    cfg = {"vlm": {"segment_filter": {"enabled": True}}}
    out = vlm_segment_filter.filter_elements(_image(tmp_path), elements, cfg)
    assert out == []


def test_vlm_error_degrades_without_dropping(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vlm_segment_filter.vlm_client, "multi_pass_answer",
        lambda *a, **k: (None, "vlm_error"),
    )
    elements = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}}]
    cfg = {"vlm": {"segment_filter": {"enabled": True}}}
    out = vlm_segment_filter.filter_elements(_image(tmp_path), elements, cfg)
    assert len(out) == 1
    assert out[0]["id"] == "E0"


def test_refine_role_sets_meta_role_after_keep(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_multi(crop, prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ('{"decision": "keep", "label": "icon"}', None)
        return ('{"role": "button"}', None)

    monkeypatch.setattr(vlm_segment_filter.vlm_client, "multi_pass_answer", fake_multi)
    elements = [{"id": "E0", "box": {"x": 10, "y": 10, "w": 40, "h": 30}, "meta": {"role": "shape"}}]
    cfg = {"vlm": {"segment_filter": {"enabled": True, "refine_role": {"enabled": True}}}}
    out = vlm_segment_filter.filter_elements(_image(tmp_path), elements, cfg)
    assert len(out) == 1
    assert out[0]["meta"]["role"] == "button"
    assert out[0]["meta"]["vlm_segment"]["refined_role"] == "button"
    assert calls["n"] == 2
