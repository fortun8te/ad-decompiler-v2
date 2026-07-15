"""test_peel_decompose.py — loop-logic tests for LayerD-style iterative peeling.

CPU-only, no model downloads: matting is always injected (color-keyed or scripted
callables) and inpainting is either the deterministic OpenCV Telea default or an oracle
that returns the true underlying composite.  Builds a synthetic 3-layer composite
(background + two overlapping rectangles) and asserts the peel loop recovers the stack —
including the key selling point: the occluded part of the lower rectangle comes back
complete.
"""
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import peel_decompose  # noqa: E402

BG = (235, 235, 235)
COLOR_A = (30, 90, 200)    # bottom rectangle (partially occluded by B)
COLOR_B = (200, 30, 40)    # top rectangle
RECT_A = (40, 40, 200, 160)    # x0, y0, x1, y1 (exclusive)
RECT_B = (120, 100, 280, 220)  # overlaps A's bottom-right quadrant


def _rect_mask(shape, rect):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = rect
    m[y0:y1, x0:x1] = True
    return m


def _make_composite():
    """bg + A + B flattened, plus the ground-truth intermediate composites."""
    bg = np.full((300, 400, 3), BG, np.uint8)
    with_a = bg.copy()
    with_a[_rect_mask(bg.shape[:2], RECT_A)] = COLOR_A
    flat = with_a.copy()
    flat[_rect_mask(bg.shape[:2], RECT_B)] = COLOR_B
    return flat, with_a, bg


def _color_matting(*colors, tol=12):
    """Matte the first listed color still visible: exact-ish match → alpha 1.0.

    Mimics a top-layer matting model on flat-color art: the topmost color is listed
    first; once peeled+inpainted away, the next call falls through to the color below.
    """
    def matte(rgb):
        for color in colors:
            hit = (np.abs(rgb.astype(int) - np.array(color)).max(axis=2) <= tol)
            if hit.any():
                return hit.astype(np.float64)
        return np.zeros(rgb.shape[:2], np.float64)
    return matte


def _oracle_inpaint(truths):
    """Perfect inpainter: returns the next ground-truth composite regardless of mask,
    but only inside the mask (outside pixels must stay untouched, like the real one)."""
    queue = list(truths)

    def inpaint(rgb, mask):
        truth = queue.pop(0)
        out = rgb.copy()
        out[mask] = truth[mask]
        return out
    return inpaint


def _cfg(**peel):
    return {"peel": peel} if peel else {}


# ── the 3-layer composite peels into the correct stack ─────────────────────────────

def test_peels_three_layer_composite_with_oracle_inpaint():
    flat, with_a, bg = _make_composite()
    result = peel_decompose.peel(
        flat, max_layers=4, cfg=_cfg(),
        matting=_color_matting(COLOR_B, COLOR_A),
        inpaint=_oracle_inpaint([with_a, bg]),
    )
    assert result.stop_reason == "empty-matte"
    assert len(result.layers) == 2

    top, under = result.layers
    assert top.peel_order == 0 and under.peel_order == 1
    bx0, by0, bx1, by1 = RECT_B
    assert top.bbox == {"x": bx0, "y": by0, "w": bx1 - bx0, "h": by1 - by0}
    assert top.area == (bx1 - bx0) * (by1 - by0)

    # THE point of peeling: the lower rectangle comes back COMPLETE, including the
    # region that was occluded by B in the flattened input.
    ax0, ay0, ax1, ay1 = RECT_A
    assert under.bbox == {"x": ax0, "y": ay0, "w": ax1 - ax0, "h": ay1 - ay0}
    assert under.area == (ax1 - ax0) * (ay1 - ay0)
    occluded_y, occluded_x = 150, 180   # inside A ∩ B — invisible in the flat input
    assert tuple(under.rgba[occluded_y, occluded_x, :3]) == COLOR_A
    assert under.rgba[occluded_y, occluded_x, 3] == 255

    # Residual background carries neither rectangle color.
    assert tuple(result.background[150, 180]) == BG
    assert tuple(result.background[60, 60]) == BG

    # stack() = LayerD order: background first, then back-to-front foregrounds.
    stack = result.stack()
    assert len(stack) == 3
    assert stack[0] is result.background
    assert stack[1] is under.rgba and stack[2] is top.rgba


def test_peels_with_default_opencv_inpaint():
    """Telea on a flat background is good enough to reveal and peel the lower layer."""
    flat, _, _ = _make_composite()
    result = peel_decompose.peel(
        flat, max_layers=4, cfg=_cfg(),
        matting=_color_matting(COLOR_B, COLOR_A),
    )
    assert len(result.layers) >= 2
    assert result.layers[0].bbox["x"] == RECT_B[0]
    assert result.layers[0].bbox["y"] == RECT_B[1]
    # Lower layer must extend into the formerly-occluded overlap region.
    a_box = result.layers[1].bbox
    assert a_box["x"] == RECT_A[0] and a_box["y"] == RECT_A[1]
    assert a_box["x"] + a_box["w"] >= RECT_B[0] + 1  # reaches past B's left edge


def test_unblend_keeps_solid_colors_and_recovers_soft_edges():
    flat, with_a, bg = _make_composite()
    result = peel_decompose.peel(
        flat, max_layers=2, cfg=_cfg(),
        matting=_color_matting(COLOR_B, COLOR_A),
        inpaint=_oracle_inpaint([with_a, bg]),
    )
    cy = (RECT_B[1] + RECT_B[3]) // 2
    cx = (RECT_B[0] + RECT_B[2]) // 2
    assert tuple(result.layers[0].rgba[cy, cx, :3]) == COLOR_B

    # Direct unblending math: a 50% blend over a known background recovers the true fg.
    image = np.full((4, 4, 3), 0, np.uint8)
    image[:] = (118, 118, 118)                    # 0.5*200 + 0.5*36 ≈ 118
    background = np.full((4, 4, 3), 36, np.uint8)
    alpha = np.full((4, 4), 0.5, np.float64)
    fg = peel_decompose.estimate_fg_color(image, background, alpha)
    assert np.all(np.abs(fg.astype(int) - 200) <= 1)


# ── stop conditions (LayerD criteria + our guards) ─────────────────────────────────

def test_stop_empty_matte_returns_input_as_background():
    flat, _, _ = _make_composite()
    result = peel_decompose.peel(
        flat, cfg=_cfg(),
        matting=lambda rgb: np.zeros(rgb.shape[:2], np.float64),
        inpaint=peel_decompose.opencv_inpaint,
    )
    assert result.layers == []
    assert result.stop_reason == "empty-matte"
    assert np.array_equal(result.background, flat)


def test_stop_full_coverage_matte():
    """LayerD: mean(hard_mask) > 0.99 means no separable top layer — stop, don't peel
    the entire canvas as one fake layer."""
    flat, _, _ = _make_composite()
    result = peel_decompose.peel(
        flat, cfg=_cfg(),
        matting=lambda rgb: np.ones(rgb.shape[:2], np.float64),
        inpaint=peel_decompose.opencv_inpaint,
    )
    assert result.layers == []
    assert result.stop_reason == "full-coverage-matte"


def test_alpha_threshold_binarization():
    """Alpha ≤ 0.005 everywhere (LayerD _th_alpha) is an empty matte, not a layer."""
    flat, _, _ = _make_composite()
    result = peel_decompose.peel(
        flat, cfg=_cfg(),
        matting=lambda rgb: np.full(rgb.shape[:2], 0.005, np.float64),
        inpaint=peel_decompose.opencv_inpaint,
    )
    assert result.layers == []
    assert result.stop_reason == "empty-matte"


def test_stop_residual_below_threshold():
    flat, _, _ = _make_composite()
    speck = np.zeros(flat.shape[:2], np.float64)
    speck[0, 0] = 1.0
    result = peel_decompose.peel(
        flat, cfg=_cfg(min_coverage_stop=0.0005),
        matting=lambda rgb: speck,
        inpaint=peel_decompose.opencv_inpaint,
    )
    assert result.layers == []
    assert result.stop_reason == "residual-below-threshold"


def test_stop_max_layers():
    flat, with_a, bg = _make_composite()
    result = peel_decompose.peel(
        flat, max_layers=1, cfg=_cfg(),
        matting=_color_matting(COLOR_B, COLOR_A),
        inpaint=_oracle_inpaint([with_a]),
    )
    assert len(result.layers) == 1
    assert result.stop_reason == "max-layers"


def test_repeat_matte_guard_stops_a_stuck_loop():
    """If the inpainter fails to remove the layer, the matting proposes the same region
    forever; the guard must emit it once, then abort instead of duplicating."""
    flat, _, _ = _make_composite()
    constant = _rect_mask(flat.shape[:2], RECT_B).astype(np.float64)
    result = peel_decompose.peel(
        flat, max_layers=5, cfg=_cfg(),
        matting=lambda rgb: constant,
        inpaint=lambda rgb, mask: rgb.copy(),   # identity "inpaint": removes nothing
    )
    assert len(result.layers) == 1
    assert result.stop_reason == "repeat-matte"


def test_max_layers_must_be_positive():
    flat, _, _ = _make_composite()
    with pytest.raises(ValueError):
        peel_decompose.peel(flat, max_layers=0, cfg=_cfg(),
                            matting=lambda rgb: np.zeros(rgb.shape[:2], np.float64),
                            inpaint=peel_decompose.opencv_inpaint)


# ── artifacts: demo outputs + manifest integrity, pipeline adapter ─────────────────

def _peel_composite(tmp_path=None):
    flat, with_a, bg = _make_composite()
    return peel_decompose.peel(
        flat, max_layers=4, cfg=_cfg(),
        matting=_color_matting(COLOR_B, COLOR_A),
        inpaint=_oracle_inpaint([with_a, bg]),
    )


def test_write_outputs_manifest_integrity(tmp_path):
    result = _peel_composite()
    manifest = peel_decompose.write_outputs(result, str(tmp_path))

    for name in ("layer_00.png", "layer_01.png", "background.png", "manifest.json"):
        assert os.path.exists(os.path.join(str(tmp_path), name)), name
    with open(os.path.join(str(tmp_path), "manifest.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == manifest

    assert manifest["version"] == 1
    assert manifest["stop_reason"] == "empty-matte"
    assert manifest["canvas"] == {"w": 400, "h": 300}
    assert manifest["background"] == {"file": "background.png", "z": 0}
    assert len(manifest["layers"]) == 2
    zs = set()
    for entry in manifest["layers"]:
        box = entry["bbox"]
        assert 0 <= box["x"] and box["x"] + box["w"] <= 400
        assert 0 <= box["y"] and box["y"] + box["h"] <= 300
        assert entry["area"] > 0
        assert 0.0 < entry["coverage"] < 1.0
        assert entry["file"] == f"layer_{entry['peel_order']:02d}.png"
        zs.add(entry["z"])
    # z is bottom-to-top over the background (z=0): topmost peel gets the highest z.
    assert zs == {1, 2}
    top = next(e for e in manifest["layers"] if e["peel_order"] == 0)
    assert top["z"] == 2

    # Saved PNGs are full-canvas RGBA and re-compositing by ascending z reproduces
    # the flattened input (oracle inpaint ⇒ exact).
    Image = pytest.importorskip("PIL.Image")
    plate = np.asarray(Image.open(os.path.join(str(tmp_path), "background.png")).convert("RGB")).copy()
    for entry in sorted(manifest["layers"], key=lambda e: e["z"]):
        rgba = np.asarray(Image.open(os.path.join(str(tmp_path), entry["file"])))
        assert rgba.shape == (300, 400, 4)
        a = rgba[:, :, 3:4].astype(np.float64) / 255.0
        plate = (rgba[:, :, :3] * a + plate * (1 - a)).astype(np.uint8)
    flat, _, _ = _make_composite()
    assert np.array_equal(plate, flat)


def test_write_pipeline_layers_back_to_front(tmp_path):
    result = _peel_composite()
    layers = peel_decompose.write_pipeline_layers(result, str(tmp_path))
    assert [l["id"] for l in layers] == ["P0", "P1"]
    # Back-to-front: P0 is the LOWER rectangle (peeled last), P1 the top one.
    assert layers[0]["box"]["x"] == RECT_A[0]
    assert layers[1]["box"]["x"] == RECT_B[0]
    for layer in layers:
        path = os.path.join(str(tmp_path), layer["png"])
        assert os.path.exists(path)
        assert layer["kind_hint"] == "unknown"
        Image = pytest.importorskip("PIL.Image")
        assert Image.open(path).size == (400, 300)   # full canvas, box is the tight crop


def test_accepts_path_and_pil_inputs(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    flat, with_a, bg = _make_composite()
    path = os.path.join(str(tmp_path), "flat.png")
    Image.fromarray(flat).save(path)
    for source in (path, Image.open(path)):
        result = peel_decompose.peel(
            source, max_layers=4, cfg=_cfg(),
            matting=_color_matting(COLOR_B, COLOR_A),
            inpaint=_oracle_inpaint([with_a, bg]),
        )
        assert len(result.layers) == 2
