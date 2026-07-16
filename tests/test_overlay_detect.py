"""test_overlay_detect.py — rounded-plate overlay detection + emission (CPU-only).

Synthesizes pill / stadium / banner / card plates over a BUSY (noise) background with
known geometry, and asserts:
  * the detector finds exactly those four plates,
  * each is classified correctly (pill/stadium/banner/card),
  * corner radius / fill colour are recovered within tolerance,
  * contained OCR text ids are attributed to the right plate,
  * emitted group candidates compile through build_design_json.build() as a native
    SOLID rounded RECT + native TEXT (no raster slice).

Skips cleanly where numpy/opencv/PIL are unavailable.
"""
import os
import sys
import tempfile
from dataclasses import asdict

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import overlay_detect, build_design_json, schema  # noqa: E402


W, H = 800, 560

# (kind label, box, radius, RGB fill)
PLATES = {
    "pill":    ({"x": 60,  "y": 80,  "w": 300, "h": 90},  14, (60, 140, 90)),
    "banner":  ({"x": 0,   "y": 300, "w": 800, "h": 70},  0,  (40, 150, 140)),
    "stadium": ({"x": 80,  "y": 420, "w": 360, "h": 70},  35, (40, 40, 45)),
    "card":    ({"x": 500, "y": 80,  "w": 260, "h": 200}, 18, (245, 245, 245)),
}
TEXT_LINES = [
    {"id": "L_pill",    "box": {"x": 90,  "y": 110, "w": 240, "h": 30}},
    {"id": "L_banner",  "box": {"x": 300, "y": 320, "w": 200, "h": 30}},
    {"id": "L_stadium", "box": {"x": 140, "y": 440, "w": 260, "h": 30}},
    {"id": "L_card",    "box": {"x": 520, "y": 120, "w": 220, "h": 40}},
]


def _compose():
    rng = np.random.default_rng(7)
    bg = rng.integers(70, 190, (H, W, 3), dtype=np.uint8)  # busy, non-flat photo stand-in
    img = Image.fromarray(bg, "RGB")
    draw = ImageDraw.Draw(img)
    for box, radius, fill in PLATES.values():
        xy = (box["x"], box["y"], box["x"] + box["w"] - 1, box["y"] + box["h"] - 1)
        if radius and radius > 0:
            draw.rounded_rectangle(xy, radius=radius, fill=fill)
        else:
            draw.rectangle(xy, fill=fill)
    return np.asarray(img, dtype=np.uint8)


def _by_kind(props):
    out = {}
    for p in props:
        out.setdefault(p["kind"], []).append(p)
    return out


def test_detects_four_plate_kinds():
    arr = _compose()
    props = overlay_detect.detect_overlays(arr, elements=None, text_lines=TEXT_LINES,
                                           canvas={"w": W, "h": H})
    kinds = _by_kind(props)
    assert set(kinds) == {"pill", "banner", "stadium", "card"}, [p["kind"] for p in props]
    assert len(props) == 4, [(p["kind"], p["bbox"]) for p in props]


def _match(props, kind):
    cands = [p for p in props if p["kind"] == kind]
    assert cands, f"no {kind} detected"
    return cands[0]


def test_geometry_radius_and_fill():
    arr = _compose()
    props = overlay_detect.detect_overlays(arr, text_lines=TEXT_LINES,
                                           canvas={"w": W, "h": H})

    pill = _match(props, "pill")
    # bbox within a few px of truth
    for k in ("x", "y", "w", "h"):
        assert abs(pill["bbox"][k] - PLATES["pill"][0][k]) <= 4, (k, pill["bbox"])
    r = pill["corner_radius"]
    r = r if isinstance(r, (int, float)) else float(np.mean(list(r.values())))
    assert 9 <= r <= 20, r  # drawn 14
    # fill close to the muted green
    fr, fg, fb = int(pill["fill"][1:3], 16), int(pill["fill"][3:5], 16), int(pill["fill"][5:7], 16)
    assert abs(fr - 60) < 22 and abs(fg - 140) < 22 and abs(fb - 90) < 22, pill["fill"]

    stadium = _match(props, "stadium")
    sr = stadium["corner_radius"]
    sr = sr if isinstance(sr, (int, float)) else float(np.mean(list(sr.values())))
    assert abs(sr - 35) <= 6, sr  # pill end == height/2


def test_text_containment():
    arr = _compose()
    props = overlay_detect.detect_overlays(arr, text_lines=TEXT_LINES,
                                           canvas={"w": W, "h": H})
    assert "L_pill" in _match(props, "pill")["text_ids"]
    assert "L_card" in _match(props, "card")["text_ids"]
    assert "L_stadium" in _match(props, "stadium")["text_ids"]
    # a plate must not claim another plate's text
    assert "L_card" not in _match(props, "pill")["text_ids"]


def test_emission_compiles_native_rect_and_text():
    arr = _compose()
    props = overlay_detect.detect_overlays(arr, text_lines=TEXT_LINES,
                                           canvas={"w": W, "h": H})
    pill = _match(props, "pill")
    text_cand = {
        "id": "L_pill", "target": "text", "text": "All-Day Weather Hold",
        "box": TEXT_LINES[0]["box"], "style": {"fontSize": 22, "color": "#ffffff"},
    }
    emoji_cand = {
        "id": "emoji0", "target": "image", "src": None,
        "box": {"x": 95, "y": 110, "w": 28, "h": 28}, "meta": {"role": "emoji"},
    }
    group = overlay_detect.emit_overlay_group(pill, texts=[text_cand], emojis=[emoji_cand])
    assert group["target"] == "group"

    run_dir = tempfile.mkdtemp(prefix="ovtest_")
    doc = build_design_json.build([group], {"w": W, "h": H}, run_dir, base_src=None)
    d = asdict(doc)
    assert schema.validate_design(d) == []

    grp = next(l for l in d["layers"] if l["id"] == pill["id"])
    assert grp["type"] == "group"
    kids = {c["type"] for c in grp["children"]}
    assert "shape" in kids and "text" in kids
    rect = next(c for c in grp["children"] if c["type"] == "shape")
    assert rect["shape_kind"] == "rect"
    assert rect["radius"] is not None
    assert rect["fill"]["kind"] == "flat"
    # the emoji chip is an image child; letterSpacing on the text is 0 (contract)
    txt = next(c for c in grp["children"] if c["type"] == "text")
    assert txt["style"]["letterSpacing"] == 0.0
    assert any(c["type"] == "image" for c in grp["children"])
    # no layer fell back to a raster slice
    for c in grp["children"]:
        assert schema.fallback_kind(c.get("meta")) is None


def test_corner_radius_helper_scalar_and_none():
    # perfect solid pill mask -> radius ~ h/2
    mask = np.zeros((80, 240), dtype=bool)
    m2 = (np.asarray(Image.new("L", (240, 80), 0)))
    img = Image.new("L", (240, 80), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, 239, 79), radius=40, fill=255)
    mask = np.asarray(img) > 0
    r = overlay_detect.estimate_corner_radius(mask)
    r = r if isinstance(r, (int, float)) else float(np.mean(list(r.values())))
    assert abs(r - 40) <= 6, r
    # pure noise -> None (never invent a radius)
    noise = np.random.default_rng(1).integers(0, 2, (60, 120)).astype(bool)
    assert overlay_detect.estimate_corner_radius(noise) is None
