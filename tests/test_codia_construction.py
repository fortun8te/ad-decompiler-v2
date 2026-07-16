"""Codia-parity construction rules (2026-07 brief).

Covers the anti-clipping generous-text-box chain, the letterSpacing=0 tracking
policy, the inverted text slice gate (text is never raster-sliced), emoji edge
stripping, mixed-weight sibling splits, flat-plate solid bands, per-group plate
backgrounds and OCR interpunct restoration.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from src import build_design_json, render_preview, schema
from src.build_design_json import (
    _generous_text_box, _split_weight_run_siblings, _strip_edge_emoji,
    _solid_plate_bands, _promote_weight_candidate, _normalize_text_stroke,
)
from src.ocr import _restore_interpuncts
from src.text_analysis import fit_text_box


# ── inverted text gate ───────────────────────────────────────────────────────────


def test_text_rows_never_fail_raster_slice_gate():
    row = {"type": "text", "ink_iou": 0.05, "ink_excess": 3.0,
           "region_ssim": 0.05, "region_color": 0.1}
    assert schema.raster_slice_failures(row) == []


def test_text_gate_can_be_reenabled_for_forensics():
    row = {"type": "text", "ink_iou": 0.05, "ink_excess": 3.0,
           "region_ssim": 0.05, "region_color": 0.1}
    reasons = schema.raster_slice_failures(row, {"text_slice_gate_enabled": True})
    assert reasons  # legacy gates still available behind the flag


def test_non_text_rows_still_gated():
    row = {"type": "shape", "region_ssim": 0.10, "region_color": 0.2}
    assert schema.raster_slice_failures(row)


# ── tracking policy ──────────────────────────────────────────────────────────────


def test_fit_text_box_never_emits_tracking():
    style = {"fontSize": 34.0, "letterSpacing": 3.6, "lineHeight": 40.0}
    box = {"x": 10, "y": 10, "w": 90, "h": 40}
    _fitted, _auto, patch = fit_text_box("SALE", style, box)
    assert patch.get("letterSpacing") == 0.0


# ── generous text boxes (anti-clipping) ──────────────────────────────────────────


def test_generous_box_height_floor_and_symmetry():
    style = {"fontSize": 30.0, "lineHeight": 36.0, "align": "LEFT"}
    box = {"x": 100.0, "y": 200.0, "w": 300.0, "h": 34.0}
    out = _generous_text_box(box, style, "Hello world")
    assert out["h"] >= 1.6 * 36.0 - 0.01
    # symmetric growth: ink center unchanged
    assert abs((out["y"] + out["h"] / 2) - (box["y"] + box["h"] / 2)) < 0.51
    # width slack away from the LEFT anchor
    assert out["x"] == pytest.approx(box["x"], abs=0.01)
    assert out["w"] > box["w"]


def test_generous_box_right_anchor_keeps_right_edge():
    style = {"fontSize": 30.0, "lineHeight": 36.0, "align": "RIGHT"}
    box = {"x": 100.0, "y": 200.0, "w": 300.0, "h": 60.0}
    out = _generous_text_box(box, style, "Hello")
    assert out["x"] + out["w"] == pytest.approx(box["x"] + box["w"], abs=0.02)


def test_fitted_line_renders_with_zero_ink_on_box_edges(tmp_path):
    """The parity-spec regression: a fitted line rendered into its EMITTED box must
    leave no ink pixel touching the box edge (the clipping defect class)."""
    run_dir = tmp_path / "run"
    candidates = [{
        "id": "t0", "target": "text", "text": "Everyday Curl Cream",
        "box": {"x": 60, "y": 90, "w": 360, "h": 34},
        "visible_box": {"x": 60, "y": 90, "w": 360, "h": 34},
        "style": {"fontSize": 30.0, "fontWeight": 400, "color": "#111111",
                  "lineHeight": 36.0, "align": "LEFT"},
        "meta": {"role": "headline"},
    }]
    doc = build_design_json.build(candidates, {"w": 520, "h": 240}, str(run_dir))
    layer = next(l for l in doc.layers if l.type == "text")
    box = layer.box
    assert layer.style.get("verticalAlign") == "CENTER"
    assert float(layer.style.get("letterSpacing") or 0) == 0.0
    render_preview.render(str(run_dir / "design.json"), str(run_dir))
    img = np.asarray(Image.open(run_dir / "preview.png").convert("L"), dtype=np.uint8)
    ink = img < 128
    x0, y0 = int(round(box["x"])), int(round(box["y"]))
    x1, y1 = int(round(box["x"] + box["w"])), int(round(box["y"] + box["h"]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1] - 1, x1), min(img.shape[0] - 1, y1)
    edges = np.concatenate([
        ink[y0, x0:x1], ink[y1, x0:x1], ink[y0:y1, x0], ink[y0:y1, x1],
    ])
    assert not edges.any(), "ink touches the emitted text box edge (clipping)"


# ── emoji stripping ─────────────────────────────────────────────────────────────


def test_strip_edge_emoji():
    assert _strip_edge_emoji("LAATSTE SITE WIDE SALE VAN 2026 ⌛") == \
        "LAATSTE SITE WIDE SALE VAN 2026"
    assert _strip_edge_emoji("We zien je woensdag. \U0001F440") == "We zien je woensdag."
    assert _strip_edge_emoji("\U0001F525 HOT DEAL") == "HOT DEAL"
    assert _strip_edge_emoji("no emoji here.") == "no emoji here."
    # interior emoji untouched (runs offsets must stay valid)
    assert _strip_edge_emoji("A \U0001F440 B") == "A \U0001F440 B"
    # pure emoji line is not blanked
    assert _strip_edge_emoji("\U0001F440") == "\U0001F440"


# ── mixed-weight sibling split ──────────────────────────────────────────────────


def test_weight_run_split_produces_siblings():
    candidate = {
        "id": "c_footer", "target": "text",
        "text": "05:00 PM · 12-05-2026 · 121K weergaven",
        "box": {"x": 20, "y": 920, "w": 680, "h": 45},
        "style": {
            "fontSize": 34.0, "fontWeight": 300, "color": "#626465",
            "fontFamily": "Inter", "fontStyle": "Light",
            "fontCandidates": [
                {"family": "Inter", "style": "Light", "weight": 300,
                 "source": "local-render", "path": "inter-light.ttf", "score": 0.9},
            ],
        },
        "text_runs": [{"start": 24, "end": 28,
                       "style": {"fontWeight": 700, "color": "#CCCCCC"}}],
        "meta": {},
    }
    pieces = _split_weight_run_siblings(candidate)
    assert len(pieces) == 3
    texts = [p["text"] for p in pieces]
    assert texts == ["05:00 PM · 12-05-2026 ·", "121K", "weergaven"]
    weights = [p["style"].get("fontWeight") for p in pieces]
    assert weights == [300, 700, 300]
    bold = pieces[1]["style"]
    # Bold sibling must not keep the Light file as candidates[0].path
    assert bold["fontWeight"] == 700
    assert bold["fontCandidates"][0]["weight"] == 700
    assert "path" not in bold["fontCandidates"][0] or not bold["fontCandidates"][0].get("path")
    xs = [p["box"]["x"] for p in pieces]
    assert xs == sorted(xs)
    assert all(not p.get("text_runs") for p in pieces)


def test_promote_weight_candidate_rewrites_mismatched_regular_file():
    style = {
        "fontFamily": "Inter", "fontWeight": 700, "fontStyle": "Bold",
        "fontCandidates": [
            {"family": "Inter", "style": "Regular", "weight": 400,
             "source": "local-render", "path": "inter-regular.ttf", "score": 0.8},
        ],
    }
    _promote_weight_candidate(style)
    assert style["fontWeight"] == 700
    assert style["fontCandidates"][0]["weight"] == 700
    assert "path" not in style["fontCandidates"][0]


def test_normalize_text_stroke_outside_and_fat_to_effect():
    thin, effects = _normalize_text_stroke(
        {"kind": "flat", "color": "#0f0f0f", "width": 2.0, "align": "CENTER"},
        {"fontSize": 48.0}, [])
    assert thin is not None
    assert thin["align"] == "OUTSIDE"
    assert thin["strokeAlign"] == "OUTSIDE"
    assert thin["width"] <= 48.0 * 0.08 + 0.01
    assert effects == []

    fat, effects = _normalize_text_stroke(
        {"kind": "flat", "color": "#ffffff", "width": 12.0},
        {"fontSize": 40.0}, [])
    assert fat is None
    assert effects and effects[0]["type"] == "DROP_SHADOW"
    assert effects[0]["offset"] == {"x": 0, "y": 0}


def test_build_preserves_detected_text_shadow_effects(tmp_path):
    run_dir = tmp_path / "run"
    candidates = [{
        "id": "t_shadow", "target": "text", "text": "SALE",
        "box": {"x": 40, "y": 40, "w": 200, "h": 48},
        "visible_box": {"x": 40, "y": 40, "w": 200, "h": 48},
        "style": {
            "fontSize": 40.0, "fontWeight": 700, "color": "#ffffff",
            "lineHeight": 48.0, "align": "LEFT", "letterSpacing": 2.5,
            "effects": [{
                "type": "DROP_SHADOW", "color": "#00000099",
                "offset": {"x": 2, "y": 3}, "radius": 4, "spread": 0,
            }],
        },
        "meta": {"role": "headline"},
    }]
    doc = build_design_json.build(candidates, {"w": 400, "h": 200}, str(run_dir))
    layer = next(l for l in doc.layers if l.type == "text")
    assert float(layer.style.get("letterSpacing") or 0) == 0.0
    assert layer.effects and layer.effects[0]["type"] == "DROP_SHADOW"
    assert layer.effects[0]["offset"] == {"x": 2, "y": 3}


def test_generous_box_pads_for_outside_stroke():
    style = {"fontSize": 40.0, "lineHeight": 48.0, "align": "LEFT"}
    box = {"x": 10.0, "y": 20.0, "w": 200.0, "h": 40.0}
    stroke = {"width": 4.0, "align": "OUTSIDE", "color": "#000"}
    out = _generous_text_box(box, style, "SALE", stroke=stroke)
    assert out["h"] >= style["fontSize"] * 1.25 - 0.01
    assert out["w"] > box["w"] + 3.0
    assert out["x"] <= box["x"]


def test_weight_run_split_ignores_small_delta_and_multiline():
    candidate = {
        "id": "c1", "target": "text", "text": "hello bold world",
        "box": {"x": 0, "y": 0, "w": 200, "h": 20},
        "style": {"fontSize": 16.0, "fontWeight": 400},
        "text_runs": [{"start": 6, "end": 10, "style": {"fontWeight": 500}}],
        "meta": {},
    }
    assert len(_split_weight_run_siblings(candidate)) == 1
    candidate["text"] = "hello bold\nworld"
    candidate["text_runs"] = [{"start": 6, "end": 10, "style": {"fontWeight": 700}}]
    assert len(_split_weight_run_siblings(candidate)) == 1


# ── flat/banded plate → solid rects ─────────────────────────────────────────────


def test_solid_plate_bands_flat_dark_ui(tmp_path):
    plate = np.zeros((200, 100, 3), dtype=np.uint8)
    plate[:24] = (6, 6, 6)      # nav strip
    plate[24:] = (0, 0, 0)      # body
    path = tmp_path / "plate.png"
    Image.fromarray(plate).save(path)
    bands = _solid_plate_bands(str(path), {"w": 100, "h": 200})
    assert bands and len(bands) == 2
    assert bands[0]["color"] == "#060606"
    assert bands[1]["color"] == "#000000"


def test_solid_plate_rejects_photo(tmp_path):
    rng = np.random.default_rng(7)
    plate = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    path = tmp_path / "plate.png"
    Image.fromarray(plate.astype(np.uint8)).save(path)
    assert _solid_plate_bands(str(path), {"w": 120, "h": 120}) is None


def test_build_emits_solid_band_rects_over_plate(tmp_path):
    plate = np.zeros((200, 100, 3), dtype=np.uint8)
    plate[:24] = (6, 6, 6)
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(plate).save(plate_path)
    doc = build_design_json.build([], {"w": 100, "h": 200}, str(tmp_path),
                                  base_src=str(plate_path))
    kinds = [(l.id, l.type) for l in doc.layers]
    assert kinds[0] == ("background", "image")
    bands = [l for l in doc.layers if str(l.id).startswith("background-band-")]
    assert len(bands) == 2
    assert bands[0].fill == {"kind": "flat", "color": "#060606"}


# ── background-per-group ────────────────────────────────────────────────────────


def test_group_on_distinct_plate_region_gets_background_child(tmp_path):
    plate = np.full((400, 400, 3), 240, dtype=np.uint8)
    plate[100:280, 40:360] = (30, 60, 120)  # distinct card region
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(plate).save(plate_path)
    candidates = [{
        "id": "g0", "target": "group",
        "box": {"x": 40, "y": 100, "w": 320, "h": 180},
        "meta": {"role": "card"},
        "children": [{
            "id": "t0", "target": "text", "text": "Review",
            "box": {"x": 20, "y": 20, "w": 120, "h": 24},
            "style": {"fontSize": 20.0, "color": "#ffffff"},
            "meta": {},
        }],
    }]
    doc = build_design_json.build(candidates, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate_path))
    group = next(l for l in doc.layers if l.type == "group")
    bg = group.children[0]
    assert bg.id.endswith("__groupbg")
    assert bg.type == "image"
    assert bg.name == "Background"
    assert bg.box == {"x": 0.0, "y": 0.0, "w": 320.0, "h": 180.0}
    assert bg.z_index < min(c.z_index for c in group.children[1:])
    asset = tmp_path / bg.src
    assert asset.exists()
    tile = np.asarray(Image.open(asset).convert("RGB"))
    assert tile.shape[:2] == (180, 320)
    assert abs(int(tile[90, 160, 2]) - 120) <= 2  # slice of the CLEAN plate


def test_group_on_shared_plate_gets_no_background_child(tmp_path):
    plate = np.zeros((400, 400, 3), dtype=np.uint8)  # uniform page plate
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(plate).save(plate_path)
    candidates = [{
        "id": "g0", "target": "group",
        "box": {"x": 40, "y": 100, "w": 320, "h": 180},
        "meta": {"role": "engagement-row"},
        "children": [{
            "id": "t0", "target": "text", "text": "121K",
            "box": {"x": 20, "y": 20, "w": 80, "h": 24},
            "style": {"fontSize": 20.0, "color": "#cccccc"},
            "meta": {},
        }],
    }]
    doc = build_design_json.build(candidates, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate_path))
    group = next(l for l in doc.layers if l.type == "group")
    assert not any(str(c.id).endswith("__groupbg") for c in group.children)


def test_per_group_background_config_gate(tmp_path):
    plate = np.full((400, 400, 3), 240, dtype=np.uint8)
    plate[100:280, 40:360] = (30, 60, 120)
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(plate).save(plate_path)
    candidates = [{
        "id": "g0", "target": "group",
        "box": {"x": 40, "y": 100, "w": 320, "h": 180},
        "meta": {"role": "card"},
        "children": [{
            "id": "t0", "target": "text", "text": "Review",
            "box": {"x": 20, "y": 20, "w": 120, "h": 24},
            "style": {"fontSize": 20.0, "color": "#ffffff"}, "meta": {},
        }],
    }]
    doc = build_design_json.build(candidates, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate_path),
                                  cfg={"background": {"per_group": False}})
    group = next(l for l in doc.layers if l.type == "group")
    assert not any(str(c.id).endswith("__groupbg") for c in group.children)


# ── OCR interpunct restoration ──────────────────────────────────────────────────


def test_interpunct_restoration():
    assert _restore_interpuncts("05:00 PM . 12-05-2026 .") == "05:00 PM · 12-05-2026 ·"
    assert _restore_interpuncts("05:00 PM - 12-05-2026 -") == "05:00 PM · 12-05-2026 ·"
    # untouched: prose without digits, real hyphens, decimals
    assert _restore_interpuncts("well - known phrase") == "well - known phrase"
    assert _restore_interpuncts("Price 4.22 USD") == "Price 4.22 USD"
    assert _restore_interpuncts("12-05-2026") == "12-05-2026"
    # Bullet glyphs + decimal view counts after a date separator.
    assert _restore_interpuncts("9:41 AM • May 12 • 1.2M views") == "9:41 AM · May 12 · 1.2M views"
    assert _restore_interpuncts("May 12 . 1.2M views") == "May 12 · 1.2M views"


# ── routing: emoji + icon chips ─────────────────────────────────────────────────


def test_routing_emoji_is_image_cutout():
    from src.routing import route
    candidate = {"id": "e0", "kind": "icon", "box": {"x": 10, "y": 10, "w": 30, "h": 30},
                 "meta": {"emoji": True}}
    routed = route(candidate, {"w": 1000, "h": 1000}, {})
    assert routed["target"] == "image"
    assert routed["meta"]["emoji"] is True


def test_routing_icon_chip_on_flat_plate_archetype():
    from src.routing import route
    candidate = {"id": "i0", "kind": "icon", "box": {"x": 10, "y": 10, "w": 40, "h": 40},
                 "meta": {"role": "icon"}}
    cfg = {"scene": {"archetype": "social_screenshot"}}
    routed = route(dict(candidate), {"w": 1000, "h": 1000}, cfg)
    assert routed["target"] == "image"
    assert routed["meta"].get("icon_chip") is True
    # photographic archetype keeps the vector path
    routed2 = route(dict(candidate), {"w": 1000, "h": 1000},
                    {"scene": {"archetype": "caption_over_photo"}})
    assert routed2["target"] == "icon"
    # explicit config override wins
    routed3 = route(dict(candidate), {"w": 1000, "h": 1000},
                    {"scene": {"archetype": "social_screenshot"},
                     "routing": {"icons_as_chips": False}})
    assert routed3["target"] == "icon"
