"""test_peel_scene.py — occlusion-attributed peel: the layer-owner hole split.

CPU-only, no models.  The centrepiece is a synthetic before/after-style composite —
two side-by-side full-frame "portraits" (left blue, right green) meeting at a seam,
a circular "product" element straddling the seam ON TOP of both, and a text block on
top of the left portrait.  The tests PROVE the core peel contract:

  * peeling the circle inpaints its footprint ONLY into the two portraits, split at
    the seam by which portrait owns each pixel (sentinel fills make this exact);
  * a portrait is byte-identical to its original everywhere it was NOT covered
    (no spurious holes), and untouched layers come out byte-identical entirely;
  * the inpaint call for each portrait sees ONLY that portrait's own pixels as
    context (context isolation — no seam bleeding is even possible);
  * re-compositing background + peeled layers (+ native text on top) reproduces the
    flattened input exactly;
  * z-order / occluded_by / occludes / fills metadata is correct;
  * a non-overlapping scene is left to the single-plate path (gate skips peel).
"""
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import peel_scene  # noqa: E402
from src.peel_scene import SceneElement  # noqa: E402

W, H = 440, 340
BG = (240, 240, 240)
BLUE = (40, 80, 200)      # left portrait
GREEN = (30, 160, 90)     # right portrait
RED = (210, 40, 50)       # circle straddling the seam
INK = (20, 20, 20)        # text block
SEAM_X = 220

LEFT_BOX = (20, 20, 220, 320)     # x0, y0, x1, y1 (exclusive)
RIGHT_BOX = (220, 20, 420, 320)
CIRCLE = (220, 170, 60)           # cx, cy, r — spans both portraits
TEXT_BOX = (40, 50, 180, 80)      # over the LEFT portrait only, clear of the circle

SENTINELS = {
    "left": (255, 0, 255),
    "right": (255, 128, 0),
    "background": (0, 255, 255),
}


def _rect_mask(box):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def _circle_mask():
    cx, cy, r = CIRCLE
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _scene():
    """Flattened composite + SceneElements (portraits z=0/1, circle z=2, text z=3)."""
    left, right, circle, text = (_rect_mask(LEFT_BOX), _rect_mask(RIGHT_BOX),
                                 _circle_mask(), _rect_mask(TEXT_BOX))
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[left] = BLUE
    flat[right] = GREEN
    flat[circle] = RED
    flat[text] = INK
    elements = [
        SceneElement(id="left", mask=left, z=0.0, kind="photo"),
        SceneElement(id="right", mask=right, z=1.0, kind="photo"),
        SceneElement(id="circle", mask=circle, z=2.0, kind="product"),
        SceneElement(id="text", mask=text, z=3.0, kind="text", is_text=True),
    ]
    return flat, elements, {"left": left, "right": right, "circle": circle, "text": text}


class SpyInpaint:
    """Sentinel filler that records, per call, the meta and the exact set of colors
    visible as context (pixels OUTSIDE the mask) — the context-isolation proof."""

    def __init__(self):
        self.calls = []

    def __call__(self, rgb, mask, meta=None):
        meta = dict(meta or {})
        context = rgb[~mask]
        colors = {tuple(c) for c in np.unique(context.reshape(-1, 3), axis=0)} \
            if context.size else set()
        self.calls.append({"meta": meta, "context_colors": colors,
                           "mask_px": int(mask.sum())})
        out = rgb.copy()
        out[mask] = SENTINELS.get(meta.get("under_id"), (1, 2, 3))
        return out


def _cfg(**peel):
    peel.setdefault("hole_dilate_px", 0)   # synthetic masks are exact — no AA fringe
    peel.setdefault("text_hole_dilate_px", 0)
    peel.setdefault("inpaint_feather_px", 0)  # unit tests assert exact fill pixels
    peel.setdefault("fail_closed_to_flat", False)
    peel.setdefault("flat_fill_allow_background", False)  # keep recomposite exact
    # Most of this file exercises the legacy text-punch attribution contract; the
    # §10 default (text_parallel_track=True — OCR ink never punches) has its own
    # tests below.
    peel.setdefault("text_parallel_track", False)
    return {"peel": peel}


# ── THE core requirement: seam-straddling occluder, layer-attributed holes ─────────

def test_circle_footprint_splits_between_left_and_right_portraits():
    flat, elements, masks = _scene()
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg())
    assert not result.skipped
    assert [l.id for l in result.layers] == ["left", "right", "circle"]  # back-to-front

    left = result.layer("left")
    right = result.layer("right")
    circle = result.layer("circle")

    circle_left = masks["circle"] & masks["left"]
    circle_right = masks["circle"] & masks["right"]
    text_left = masks["text"] & masks["left"]
    assert circle_left.any() and circle_right.any()   # the circle truly straddles

    # LEFT portrait: sentinel exactly where the circle/text covered it, original blue
    # pixels EVERYWHERE else in its mask — no spurious holes.
    lrgb = left.rgba[:, :, :3]
    hole = circle_left | text_left
    assert np.all(lrgb[circle_left] == SENTINELS["left"])
    assert np.all(lrgb[text_left] == SENTINELS["left"])
    assert np.all(lrgb[masks["left"] & ~hole] == BLUE)
    assert np.all(left.rgba[:, :, 3] == masks["left"] * 255)

    # RIGHT portrait: hole is ONLY the circle's right part.  The text (which never
    # covered the right portrait) must not have punched anything.
    rrgb = right.rgba[:, :, :3]
    assert np.all(rrgb[circle_right] == SENTINELS["right"])
    assert np.all(rrgb[masks["right"] & ~circle_right] == GREEN)

    # Circle: NOTHING on top of it — comes out untouched, byte-identical pixels.
    assert circle.occluded_by == []
    assert circle.fills == []
    assert np.all(circle.rgba[:, :, :3][masks["circle"]] == RED)
    assert np.all(circle.rgba[:, :, 3] == masks["circle"] * 255)


def test_inpaint_context_is_isolated_per_underlying_layer():
    """The seam-bleed proof: the fill for the left portrait can only ever see blue."""
    flat, elements, _ = _scene()
    spy = SpyInpaint()
    peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg())

    by_under = {}
    for call in spy.calls:
        by_under.setdefault(call["meta"]["under_id"], []).append(call)

    for call in by_under["left"]:
        assert call["meta"]["isolated_context"] is True
        assert call["context_colors"] <= {BLUE}, call["context_colors"]
    for call in by_under["right"]:
        assert call["context_colors"] <= {GREEN}
    for call in by_under["background"]:
        assert call["context_colors"] <= {BG}

    # Routing split: the left portrait gets one element-class call (circle) and one
    # text-class call (the text block) so a router can keep text holes off Flux.
    left_flags = sorted(c["meta"]["text_occluder"] for c in by_under["left"])
    assert left_flags == [False, True]
    text_call = next(c for c in by_under["left"] if c["meta"]["text_occluder"])
    assert text_call["meta"]["occluder_ids"] == ["text"]
    elem_call = next(c for c in by_under["left"] if not c["meta"]["text_occluder"])
    assert elem_call["meta"]["occluder_ids"] == ["circle"]
    assert all(not c["meta"]["text_occluder"] for c in by_under["right"])


def test_occlusion_metadata_and_fill_attribution():
    flat, elements, masks = _scene()
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())

    left, right, circle = (result.layer("left"), result.layer("right"),
                           result.layer("circle"))
    assert left.occluded_by == ["text", "circle"]       # front-to-back
    assert right.occluded_by == ["circle"]
    assert circle.occluded_by == []
    assert circle.occludes == ["right", "left"]
    assert left.occludes == [] and right.occludes == []
    assert (left.z_index, right.z_index, circle.z_index) == (0, 1, 2)

    fills = {f.occluder_id: f for f in left.fills}
    assert set(fills) == {"circle", "text"}
    assert fills["circle"].area == int((masks["circle"] & masks["left"]).sum())
    assert fills["text"].area == int((masks["text"] & masks["left"]).sum())
    assert fills["text"].text_occluder and not fills["circle"].text_occluder
    assert [f.occluder_id for f in right.fills] == ["circle"]
    assert right.fills[0].area == int((masks["circle"] & masks["right"]).sum())

    # Background fills: every pixel's DIRECT occluder is one of the portraits
    # (circle/text sit on the portraits, never directly on the background).
    assert {f.occluder_id for f in result.background_fills} == {"left", "right"}


def test_recomposite_reproduces_the_input_exactly():
    flat, elements, masks = _scene()
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())

    # Composite: background, then layers back-to-front, then native text on top
    # (text is never a peel layer — the pipeline renders it as an editable node).
    plate = result.background.astype(np.float64)
    for layer in sorted(result.layers, key=lambda l: l.z_index):
        a = layer.rgba[:, :, 3:4].astype(np.float64) / 255.0
        plate = layer.rgba[:, :, :3] * a + plate * (1.0 - a)
    plate = plate.round().astype(np.uint8)
    plate[masks["text"]] = flat[masks["text"]]
    assert np.array_equal(plate, flat)

    # The built-in check reports the same: exact outside the text footprint.
    check = result.meta["recomposite"]
    assert check["exact"] is True
    assert check["text_excluded_px"] == int(masks["text"].sum())


def test_attribute_footprint_owner_split():
    """Direct unit test of the layer-owner hole split for a peeled footprint."""
    _, elements, masks = _scene()
    lower = [e for e in elements if e.id in ("left", "right")]
    split = peel_scene.attribute_footprint(masks["circle"], lower)
    assert set(split) == {"left", "right"}          # portraits fully cover the circle
    assert np.array_equal(split["left"], masks["circle"] & masks["left"])
    assert np.array_equal(split["right"], masks["circle"] & masks["right"])
    # A footprint over nothing attributes to the background.
    lonely = np.zeros((H, W), bool)
    lonely[0:10, 0:10] = True
    assert set(peel_scene.attribute_footprint(lonely, lower)) == {"background"}
    # Three-deep stack: the TOPMOST lower element owns the pixel.
    stacked = peel_scene.attribute_footprint(
        masks["circle"], lower + [SceneElement(id="mid", mask=masks["circle"], z=1.5)])
    assert set(stacked) == {"mid"}


# ── selective use: the overlap gate ────────────────────────────────────────────────

def test_non_overlapping_scene_is_skipped_for_the_single_plate_path():
    flat = np.full((H, W, 3), BG, np.uint8)
    a, b = np.zeros((H, W), bool), np.zeros((H, W), bool)
    a[40:140, 40:140] = True
    b[40:140, 240:340] = True
    flat[a] = BLUE
    flat[b] = GREEN
    elements = [SceneElement(id="a", mask=a, z=0.0), SceneElement(id="b", mask=b, z=1.0)]
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    assert result.skipped and result.skip_reason == "no-overlap"
    assert result.layers == [] and result.background is None
    assert result.overlap["needed"] is False and result.overlap["pairs"] == []


def test_gate_thresholds_ignore_token_overlaps():
    flat, elements, _ = _scene()
    tiny_cfg = _cfg(min_overlap_area=10 ** 9)      # nothing qualifies
    report = peel_scene.overlap_report(elements, tiny_cfg)
    assert report["needed"] is False
    assert all(not p["qualifies"] for p in report["pairs"])
    # An element over ONLY text never triggers peel (text under-layers stay native).
    text = SceneElement(id="t", mask=_rect_mask((40, 40, 200, 80)), z=0.0,
                        kind="text", is_text=True)
    badge = SceneElement(id="badge", mask=_rect_mask((60, 50, 120, 70)), z=1.0)
    assert peel_scene.overlap_report([text, badge], _cfg())["needed"] is False
    # force=True bypasses the gate for demos.
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(),
                                   cfg=_cfg(min_overlap_area=10 ** 9), force=True)
    assert not result.skipped


# ── robustness / interface details ─────────────────────────────────────────────────

def test_meta_less_inpaint_callable_is_supported():
    flat, elements, masks = _scene()
    result = peel_scene.peel_scene(
        flat, elements, cfg=_cfg(),
        inpaint=lambda rgb, mask: np.where(mask[..., None], 7, rgb).astype(np.uint8))
    hole = masks["circle"] & masks["left"]
    assert np.all(result.layer("left").rgba[:, :, :3][hole] == 7)


def test_default_telea_fills_flat_holes_with_the_layer_color():
    flat, elements, masks = _scene()
    result = peel_scene.peel_scene(flat, elements, cfg=_cfg())   # default OpenCV Telea
    hole = masks["circle"] & masks["left"]
    filled = result.layer("left").rgba[:, :, :3][hole].astype(int)
    assert np.abs(filled - np.array(BLUE)).max() <= 12
    hole_r = masks["circle"] & masks["right"]
    filled_r = result.layer("right").rgba[:, :, :3][hole_r].astype(int)
    assert np.abs(filled_r - np.array(GREEN)).max() <= 12


def test_hole_dilation_stays_inside_the_under_layers_own_mask():
    flat, elements, masks = _scene()
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy,
                                   cfg=_cfg(hole_dilate_px=2))
    lrgb = result.layer("left").rgba[:, :, :3]
    changed = np.any(lrgb != np.where(masks["left"][..., None], BLUE, 0), axis=2)
    # Everything rewritten lies within the left mask and within 2 px of a true hole.
    assert not np.any(changed & ~masks["left"])
    import cv2
    hole = ((masks["circle"] | masks["text"]) & masks["left"]).astype(np.uint8)
    ring = cv2.dilate(hole, np.ones((5, 5), np.uint8)) > 0
    assert not np.any(changed & ~ring)


def test_fully_covered_layer_degrades_honestly():
    """A layer with (almost) no visible pixels has no context to isolate — the fill
    still happens (unisolated) and the layer is flagged, never crashed on."""
    flat = np.full((100, 100, 3), BG, np.uint8)
    under = np.zeros((100, 100), bool)
    under[20:60, 20:60] = True
    top = np.zeros((100, 100), bool)
    top[18:62, 18:62] = True          # completely swallows `under`
    flat[top] = RED
    elements = [SceneElement(id="under", mask=under, z=0.0),
                SceneElement(id="top", mask=top, z=1.0)]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg())
    layer = result.layer("under")
    assert layer.meta.get("low_context_fill") is True
    call = next(c for c in spy.calls if c["meta"]["under_id"] == "under")
    assert call["meta"]["isolated_context"] is False


def test_mask_shape_mismatch_is_rejected():
    flat, _, _ = _scene()
    bad = SceneElement(id="bad", mask=np.zeros((10, 10), bool), z=0.0)
    with pytest.raises(ValueError):
        peel_scene.peel_scene(flat, [bad], cfg=_cfg(), force=True)


def test_provided_background_plate_is_reused_verbatim():
    flat, elements, _ = _scene()
    plate = np.full((H, W, 3), (9, 9, 9), np.uint8)
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(),
                                   cfg=_cfg(), background=plate)
    assert np.array_equal(result.background, plate)
    assert result.meta["background"] == "provided"
    assert result.background_fills == []


def test_refine_alpha_touches_only_the_edge_band():
    """Matting may soften the cutout EDGE; it can never re-detect / grow the layer."""
    import cv2
    flat, elements, masks = _scene()
    circle = next(e for e in elements if e.id == "circle")
    # Adversarial matting: claims EVERYTHING is foreground at alpha 0.5.
    matting = lambda rgb: np.full(rgb.shape[:2], 0.5, np.float64)  # noqa: E731
    peel_scene.refine_element_alpha(flat, circle, matting, cfg=_cfg(refine_band_px=3))
    alpha = circle.alpha
    kernel = np.ones((7, 7), np.uint8)
    inner = cv2.erode(masks["circle"].astype(np.uint8), kernel) > 0
    outer = cv2.dilate(masks["circle"].astype(np.uint8), kernel) > 0
    band = outer & ~inner
    assert np.all(alpha[inner] == 1.0)          # interior untouched
    assert np.all(alpha[~outer] == 0.0)         # matting cannot grow the layer
    assert np.all(alpha[band] == 0.5)           # only the edge band was consulted
    assert circle.meta["alpha_refined"] == {"band_px": 3}

    # Wired through peel_scene: refined alpha lands in the layer RGBA.
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(),
                                   cfg=_cfg(refine_alpha=True, refine_band_px=3),
                                   matting=matting)
    layer_alpha = result.layer("circle").rgba[:, :, 3]
    assert np.all(layer_alpha[inner] == 255)
    assert np.all(layer_alpha[band] == 128)     # round(0.5 * 255)
    assert np.all(layer_alpha[~outer] == 0)


# ── z-order derivation + run-artifact loader ───────────────────────────────────────

def test_derive_z_order_bands_and_containment():
    photo = SceneElement(id="photo", mask=_rect_mask((0, 0, 400, 300)), z=0.0, kind="photo")
    button = SceneElement(id="button", mask=_rect_mask((50, 50, 150, 90)), z=0.0, kind="button")
    icon = SceneElement(id="icon", mask=_rect_mask((60, 55, 90, 85)), z=0.0, kind="icon")
    text = SceneElement(id="t", mask=_rect_mask((200, 40, 320, 60)), z=0.0,
                        kind="text", is_text=True)
    ordered = peel_scene.derive_z_order([photo, button, icon, text])
    z = {e.id: e.z for e in ordered}
    assert z["photo"] < z["button"] < z["icon"] < z["t"]
    assert len({e.z for e in ordered}) == 4      # unique z values
    # Pre-assigned distinct z values are left alone.
    a = SceneElement(id="a", mask=_rect_mask((0, 0, 10, 10)), z=5.0)
    b = SceneElement(id="b", mask=_rect_mask((0, 0, 10, 10)), z=2.0)
    peel_scene.derive_z_order([a, b])
    assert (a.z, b.z) == (5.0, 2.0)


def test_elements_from_run_loads_masks_and_text_occluders(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    run = str(tmp_path)
    os.makedirs(os.path.join(run, "fused_elements"), exist_ok=True)
    mask_crop = np.zeros((20, 30), np.uint8)
    mask_crop[5:15, 5:25] = 255
    Image.fromarray(mask_crop, mode="L").save(
        os.path.join(run, "fused_elements", "E000.png"))
    fused = [
        {"id": "E000", "box": {"x": 10, "y": 12, "w": 30, "h": 20}, "kind": "icon",
         "mask_src": os.path.join("fused_elements", "E000.png")},
        {"id": "E001", "box": {"x": 50, "y": 60, "w": 8, "h": 6}, "kind": "photo"},
    ]
    ocr = {"lines": [{"id": "L0", "box": {"x": 5, "y": 5, "w": 40, "h": 10}}]}
    elements = peel_scene.elements_from_run(run, fused, {"w": 100, "h": 90},
                                            cfg=_cfg(), ocr=ocr)
    by_id = {e.id: e for e in elements}
    assert set(by_id) == {"E000", "E001", "text_L0"}
    e0 = by_id["E000"]
    assert int(e0.mask.sum()) == int((mask_crop > 127).sum())
    assert e0.mask[12 + 5, 10 + 5] and not e0.mask[12, 10]   # crop pasted at the box
    assert by_id["E001"].meta["box_only_mask"] is True       # box fallback
    assert int(by_id["E001"].mask.sum()) == 8 * 6
    text = by_id["text_L0"]
    assert text.is_text and text.z > e0.z and text.z > by_id["E001"].z
    # text occluders can be disabled
    no_text = peel_scene.elements_from_run(run, fused, {"w": 100, "h": 90},
                                           cfg=_cfg(text_occluders="off"), ocr=ocr)
    assert {e.id for e in no_text} == {"E000", "E001"}


# ── detection-granularity guard ────────────────────────────────────────────────────

def _fragmented_mask(box, step=8):
    """A swiss-cheese residual mask (checkerboard of small tiles) — the shape fusion
    emits for 'photo panel minus persons minus product' leftovers."""
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            if ((x // step) + (y // step)) % 2 == 0:
                m[y:min(y + step // 2, y1), x:min(x + step // 2, x1)] = True
    return m


def test_mask_integrity_metrics():
    solid = peel_scene.mask_integrity(_rect_mask((20, 20, 120, 120)))
    assert solid["components"] == 1 and solid["cc_frac"] == 1.0 and solid["hole_frac"] == 0.0
    frag = peel_scene.mask_integrity(_fragmented_mask((20, 20, 220, 220)))
    assert frag["components"] > 50 and frag["cc_frac"] < 0.5
    ring = _rect_mask((20, 20, 120, 120)) & ~_rect_mask((45, 45, 95, 95))
    donut = peel_scene.mask_integrity(ring)
    assert donut["components"] == 1 and donut["hole_frac"] > 0.2
    empty = peel_scene.mask_integrity(np.zeros((H, W), bool))
    assert empty["area"] == 0 and not empty["cc_frac"]


def test_granularity_guard_blocks_fragmented_under_layer():
    """Peel is only as good as the elements it is fed: an occluder over a residual
    swiss-cheese 'panel' must NOT switch peel on — the guard skips with a reason."""
    frag = _fragmented_mask((20, 20, 220, 320))
    circle = _circle_mask()
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[frag] = BLUE
    flat[circle] = RED
    elements = [SceneElement(id="panel", mask=frag, z=0.0, kind="photo-fragment"),
                SceneElement(id="circle", mask=circle, z=1.0, kind="product")]
    report = peel_scene.overlap_report(elements, _cfg())
    assert report["needed"] is False and report["blocked_qualifying"] >= 1
    assert report["eligibility"]["panel"]["eligible"] is False
    assert "fragmented" in report["eligibility"]["panel"]["reason"]
    assert report["eligibility"]["circle"]["eligible"] is True

    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    assert result.skipped
    assert result.skip_reason.startswith("no-eligible-overlap")
    assert "panel" in result.skip_reason and "fragmented" in result.skip_reason

    # The guard is opt-out for research runs.
    loose = peel_scene.overlap_report(elements, _cfg(require_eligible=False))
    assert loose["needed"] is True


def test_granularity_guard_passes_solid_pairs_and_ignores_text_only_activation():
    flat, elements, _ = _scene()
    report = peel_scene.overlap_report(elements, _cfg())
    assert report["needed"] is True and report["blocked_qualifying"] == 0
    assert all(e["eligible"] for e in report["eligibility"].values())
    # Text-over-element pairs alone must not activate peel (text stays native;
    # activation needs a genuine element-over-element pair).
    text = SceneElement(id="t", mask=_rect_mask((40, 40, 200, 80)), z=1.0,
                        kind="text", is_text=True)
    under = SceneElement(id="photo", mask=_rect_mask((20, 20, 220, 320)), z=0.0,
                         kind="photo")
    assert peel_scene.overlap_report([under, text], _cfg())["needed"] is False


# ── role-aware z bands ─────────────────────────────────────────────────────────────

def test_role_band_puts_product_cutout_above_the_card_it_sits_on():
    """Fusion tags product cutouts kind='photo-fragment' (band 5) but role='product'
    (band 20); the role must win or the card is treated as the occluder (inverted z)."""
    card = SceneElement(id="card", mask=_rect_mask((20, 20, 420, 320)), z=0.0,
                        kind="shape", meta={"role": "shape"})
    product = SceneElement(id="product", mask=_rect_mask((100, 100, 220, 280)), z=0.0,
                           kind="photo-fragment", meta={"role": "product"})
    peel_scene.derive_z_order([card, product])
    assert product.z > card.z
    report = peel_scene.overlap_report([card, product], _cfg())
    assert report["needed"] is True
    pair = next(p for p in report["pairs"] if p["qualifies"])
    assert (pair["top"], pair["under"]) == ("product", "card")


# ── flat-fill fast path ────────────────────────────────────────────────────────────

def test_flat_fill_fills_solid_holes_exactly_without_calling_the_inpainter():
    # Flat-fill is for card/shape surfaces — photo kinds are denied (real photos
    # have textured margins that used to false-trigger a solid beige plate).
    # Text on non-photo under-layers is also eligible (native text over a card).
    flat, elements, masks = _scene()
    for el in elements:
        if el.id in ("left", "right"):
            el.kind = "shape"
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy,
                                   cfg=_cfg(flat_fill_tol=6.0))
    # Element + text holes on the solid shapes are flat — only the background
    # plate (which refuses flat-fill) reaches the inpainter.
    assert not any(c["meta"]["under_id"] in ("left", "right") for c in spy.calls)
    left = result.layer("left")
    hole = masks["circle"] & masks["left"]
    assert np.all(left.rgba[:, :, :3][hole] == BLUE)       # exact, no Telea blur
    text_left = masks["text"] & masks["left"]
    assert np.all(left.rgba[:, :, :3][text_left] == BLUE)
    right = result.layer("right")
    assert np.all(right.rgba[:, :, :3][masks["circle"] & masks["right"]] == GREEN)
    backends = {b["text_occluder"]: b["backend"] for b in left.meta["fill_backends"]}
    assert backends == {False: "solid", True: "solid"}
    assert result.meta["recomposite"]["exact"] is True


def test_flat_fill_still_denies_text_holes_on_photo_under_layers():
    """Text over a photo must not solid-fill; photo kinds always deny flat-fill."""
    flat, elements, masks = _scene()  # left/right kind=photo
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg(flat_fill_tol=6.0))
    left = result.layer("left")
    text_backends = [b["backend"] for b in left.meta["fill_backends"] if b["text_occluder"]]
    assert text_backends and all(b == "inpaint" for b in text_backends)
    assert any(c["meta"]["under_id"] == "left" and c["meta"]["text_occluder"]
               for c in spy.calls)


def test_flat_fill_defers_to_the_inpainter_on_textured_context():
    rng = np.random.default_rng(7)
    flat = rng.integers(0, 255, (H, W, 3), np.uint8)       # loud texture everywhere
    under = _rect_mask((20, 20, 220, 320))
    top = _circle_mask()
    elements = [SceneElement(id="under", mask=under, z=0.0, kind="shape"),
                SceneElement(id="top", mask=top, z=1.0)]
    spy = SpyInpaint()
    peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg(flat_fill_tol=6.0))
    assert any(c["meta"]["under_id"] == "under" for c in spy.calls)


def test_flat_fill_skips_thin_rim_oversized_holes():
    """Thin flat margins must not paint a whole plate solid (benchmark 016)."""
    flat = np.full((H, W, 3), BG, np.uint8)
    # Almost-full under "card" with a tiny visible rim — thin-rim guard fires.
    under = _rect_mask((10, 10, W - 10, H - 10))
    top = _rect_mask((30, 30, W - 30, H - 30))
    flat[under] = (250, 230, 160)
    flat[top] = RED
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="blob", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(flat_fill_tol=8.0, flat_fill_min_visible_frac=0.12,
                 flat_fill_max_area=5000, flat_fill_max_frac=0.05))
    assert not result.skipped
    card = result.layer("card")
    backends = [b["backend"] for b in card.meta["fill_backends"] if not b["text_occluder"]]
    assert backends and all(b == "inpaint" for b in backends)
    assert any(c["meta"]["under_id"] == "card" for c in spy.calls)


def test_background_flat_fill_per_cc_on_uniform_plate():
    """Peel background on solid chrome should solid-fill (002 orange/white plates)."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((40, 40, 200, 200))
    top = _rect_mask((80, 80, 160, 160))
    flat[under] = (200, 40, 10)
    flat[top] = RED
    # Leave most of the canvas as BG so background flat-fill has a clean ring.
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="blob", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(flat_fill_tol=8.0, flat_fill_allow_background=True,
                 flat_fill_min_visible_frac=0.0, fill_cc_split=True))
    assert not result.skipped
    bg_backends = [b["backend"] for b in (result.meta.get("fill_backends") or [])
                   if not b.get("text_occluder")]
    assert bg_backends and "solid" in bg_backends

    # product_on_flat archetype enables background solid without the explicit knob.
    cfg2 = _cfg(flat_fill_tol=8.0, flat_fill_min_visible_frac=0.0)
    cfg2["scene"] = {"archetype": "product_on_flat"}
    result2 = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=cfg2)
    bg2 = [b["backend"] for b in (result2.meta.get("fill_backends") or [])
           if not b.get("text_occluder")]
    assert bg2 and "solid" in bg2


def test_wordmark_does_not_activate_peel_alone():
    under = _rect_mask((40, 40, 200, 200))
    logo = _rect_mask((80, 80, 140, 140))
    elements = [
        SceneElement(id="card", mask=under, z=0.0, kind="shape"),
        SceneElement(id="logo", mask=logo, z=1.0, kind="icon", meta={"role": "logo"}),
    ]
    report = peel_scene.overlap_report(elements, _cfg(min_overlap_area=64))
    assert report["eligibility"]["logo"]["reason"] == "artwork-wordmark"
    assert report["needed"] is False


def test_fail_closed_to_flat_when_generative_smears():
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((20, 20, 220, 320))
    top = _circle_mask()
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="blob", mask=top, z=1.0, kind="product")]

    class SmearInpaint:
        def __call__(self, rgb, mask, meta=None):
            out = rgb.copy()
            noise = np.random.default_rng(0).integers(0, 255, rgb.shape, np.uint8)
            out[mask] = noise[mask]
            return out

    # Deny first solid via area cap; fail-closed re-samples and keeps solid blue.
    cfg = _cfg(flat_fill_tol=8.0, fail_closed_to_flat=True, fail_closed_residue=1.0,
               flat_fill_max_area=1, flat_fill_min_visible_frac=0.0,
               context_shadow_px=0, inpaint_feather_px=0)
    result = peel_scene.peel_scene(flat, elements, inpaint=SmearInpaint(), cfg=cfg)
    card = result.layer("card")
    hole = top & under
    assert np.all(card.rgba[:, :, :3][hole] == BLUE)
    assert any(b.get("backend") == "solid" for b in card.meta["fill_backends"])


def test_max_components_and_peel_inpaint_mode():
    frag = _fragmented_mask((20, 20, 220, 220), step=4)
    assert peel_scene.mask_integrity(frag)["components"] > 24
    elig = peel_scene.element_eligibility(
        SceneElement(id="panel", mask=frag, z=0.0, kind="photo-fragment"),
        _cfg(max_components=24))
    assert elig["eligible"] is False and "components" in elig["reason"]
    assert peel_scene.peel_inpaint_mode(
        {"inpaint": {"mode": "flux"}, "scene": {"archetype": "lifestyle_overlay"}},
        {"under_kind": "photo"}) == "lama"


def test_text_holes_on_shape_prefer_solid_fill():
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((20, 20, 220, 320))
    text = _rect_mask((60, 80, 180, 120))
    flat[under] = BLUE
    flat[text] = INK
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="t", mask=text, z=1.0, kind="text", is_text=True),
                # Need an object pair to activate peel.
                SceneElement(id="icon", mask=_rect_mask((40, 200, 100, 260)), z=2.0, kind="icon")]
    flat[elements[2].mask] = RED
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(flat_fill_tol=6.0, flat_fill_text=True, flat_fill_min_visible_frac=0.0))
    card = result.layer("card")
    text_backends = [b["backend"] for b in card.meta["fill_backends"] if b["text_occluder"]]
    assert text_backends and all(b == "solid" for b in text_backends)
    # Text on photo must not punch when punch_text_into_photos is false.
    photo = SceneElement(id="photo", mask=under, z=0.0, kind="photo-fragment")
    els2 = [photo,
            SceneElement(id="t", mask=text, z=1.0, kind="text", is_text=True),
            SceneElement(id="icon", mask=_rect_mask((40, 200, 100, 260)), z=2.0, kind="icon")]
    flat2 = flat.copy()
    spy2 = SpyInpaint()
    result2 = peel_scene.peel_scene(
        flat2, els2, inpaint=spy2,
        cfg=_cfg(punch_text_into_photos=False, flat_fill_tol=6.0))
    ph = result2.layer("photo")
    assert not any(b.get("text_occluder") for b in (ph.meta.get("fill_backends") or []))


def test_abandon_oversized_photo_holes_leaves_transparent():
    """Half-covered photo under-layers: don't LaMa-haze — punch alpha when configured."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((20, 20, 420, 320))
    # Large occluder covering >30% of under.
    top = _rect_mask((20, 100, 420, 320))
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="panel", mask=under, z=0.0, kind="photo-fragment"),
                SceneElement(id="cover", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(abandon_hole_frac=0.30, per_occluder_area=1, large_photo_hole="abandon",
                 allow_flux=False))
    panel = result.layer("panel")
    hole = top & under
    assert np.all(panel.rgba[:, :, 3][hole] == 0)
    assert panel.meta.get("abandoned_fill") is True
    assert any(b["backend"] == "abandoned" for b in panel.meta["fill_backends"])
    # No inpaint call for the abandoned element-class hole.
    assert not any(c["meta"].get("under_id") == "panel"
                   and not c["meta"].get("text_occluder") for c in spy.calls)


def test_large_photo_holes_bake_only_when_explicitly_enabled():
    """Past Flux max (or Flux off): bake keeps original pixels — but ONLY with
    bake_under_layers, because baking ghosts the occluder into the layer."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((10, 10, 450, 350))
    top = _rect_mask((40, 40, 280, 280))  # ~57k px hole
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="panel", mask=under, z=0.0, kind="photo-fragment"),
                SceneElement(id="cover", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(allow_flux=False, max_generative_photo_hole_px=8000,
                 abandon_photo_min_area=12000, large_photo_hole="bake",
                 bake_under_layers=True,
                 per_occluder_area=1, flat_fill_tol=0.0))
    panel = result.layer("panel")
    hole = top & under
    assert np.all(panel.rgba[:, :, 3][hole] == 255)
    assert np.all(panel.rgba[:, :, :3][hole] == RED)
    assert panel.meta.get("baked_large_photo_hole") is True
    assert panel.meta.get("peel_quality") == "incomplete-photo"
    assert any(b["backend"] == "baked" for b in panel.meta["fill_backends"])
    assert not any(c["meta"].get("under_id") == "panel"
                   and not c["meta"].get("text_occluder") for c in spy.calls)


def test_under_layer_never_bakes_occluder_by_default():
    """Ownership contract (H10): a chart/photo slice must NOT carry the product
    that overlapped it. Default: unfillable large holes in an UNDER-LAYER go
    transparent (abandon) even when large_photo_hole says bake — the occluder
    covers the hole in composite, and moving the layer reveals honesty, not a ghost."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((10, 10, 450, 350))
    top = _rect_mask((40, 40, 280, 280))
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="chart", mask=under, z=0.0, kind="chart"),
                SceneElement(id="product", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(allow_flux=False, max_generative_photo_hole_px=8000,
                 abandon_photo_min_area=12000, large_photo_hole="bake",
                 per_occluder_area=1, flat_fill_tol=0.0))
    chart = result.layer("chart")
    hole = top & under
    # No RED occluder pixel survives in the chart layer's visible RGBA.
    assert np.all(chart.rgba[:, :, 3][hole] == 0)
    assert chart.meta.get("abandoned_fill") is True
    assert not np.any((chart.rgba[:, :, 3] > 0)
                      & np.all(chart.rgba[:, :, :3] == RED, axis=2))


def test_flux_band_photo_holes_call_inpaint_not_bake():
    """With allow_flux, mid-size photo holes stay generative (adapter may pick Flux)."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((10, 10, 450, 350))
    top = _rect_mask((40, 40, 280, 280))  # ~57k — inside default flux_max 220k
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="panel", mask=under, z=0.0, kind="photo-fragment"),
                SceneElement(id="cover", mask=top, z=1.0, kind="product")]
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(allow_flux=True, flux_min_hole_px=4000, flux_max_hole_px=220000,
                 per_occluder_area=1, flat_fill_tol=0.0))
    panel = result.layer("panel")
    assert not panel.meta.get("baked_large_photo_hole")
    assert any(c["meta"].get("under_id") == "panel"
               and not c["meta"].get("text_occluder") for c in spy.calls)


def test_peel_inpaint_mode_routes_photo_band_to_flux():
    cfg = {"peel": {"allow_flux": True, "flux_min_hole_px": 4000, "flux_max_hole_px": 100000}}
    assert peel_scene.peel_inpaint_mode(
        cfg, {"under_kind": "photo-fragment", "hole_px": 20000}) == "flux_comfy"
    assert peel_scene.peel_inpaint_mode(
        cfg, {"under_kind": "photo-fragment", "hole_px": 500}) == "lama"
    # Non-photo unders (shape/chrome/wash plates) route to opencv, NOT LaMa: LaMa's
    # texture synthesis hallucinates blotchy bands on smooth plates (013's headline
    # smudge). opencv mode tries the analytic gradient fill first, then Telea.
    assert peel_scene.peel_inpaint_mode(
        cfg, {"under_kind": "shape", "hole_px": 20000}) == "opencv"
    assert peel_scene.peel_inpaint_mode(
        cfg, {"under_kind": "photo", "hole_px": 20000, "text_occluder": True}) == "lama"


# ── H7/H13: text / overlays sitting DIRECTLY on a busy or DARK photo (make-or-break) ─
# The risk this guards: a locally-flat ring on a dark photo passes the solid-median
# test and leaves a flat painted patch under the peeled text.  The background plate
# must be classified "photo" so its holes route to the injected inpainter, never a
# solid patch.

def _photo_bg(seed=7):
    """A photographic (high local-variance) background plate."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_background_plate_kind_flags_photo_and_rejects_flat_chrome():
    opts = dict(peel_scene.SCENE_DEFAULTS)
    vis = np.ones((H, W), bool)
    flat_gray = np.full((H, W, 3), (200, 200, 200), np.uint8)
    dark_flat = np.full((H, W, 3), (8, 8, 10), np.uint8)          # near-black chrome
    dark_photo = np.random.default_rng(3).integers(0, 90, (H, W, 3), dtype=np.uint8)
    assert peel_scene.background_plate_kind(flat_gray, vis, opts) == "background"
    assert peel_scene.background_plate_kind(dark_flat, vis, opts) == "background"
    assert peel_scene.background_plate_kind(_photo_bg(), vis, opts) == "photo"
    # The specific H7 sleep-mask risk: a DARK but textured photo is still a photo.
    assert peel_scene.background_plate_kind(dark_photo, vis, opts) == "photo"


def test_flat_fill_is_denied_on_photo_under_kind():
    """The H7 guard at the routing level: no solid patch may ever land on a photo
    plate, regardless of policy/caps."""
    write = _rect_mask((40, 40, 120, 120))
    element_mask = np.ones((H, W), bool)
    visible = ~write
    opts = dict(peel_scene.SCENE_DEFAULTS, flat_fill_tol=8.0,
                flat_fill_allow_background=True, flat_fill_text=True)
    assert not peel_scene._flat_fill_allowed(
        write, element_mask, visible,
        {"under_kind": "photo", "background": True, "text_occluder": True}, opts)
    # Same hole on a flat chrome background IS eligible (proves the gate is kind-driven).
    assert peel_scene._flat_fill_allowed(
        write, element_mask, visible,
        {"under_kind": "background", "background": True, "text_occluder": True}, opts)


def _h7_scene():
    """Photographic background + an activating element pair (card+badge) + a text
    headline sitting DIRECTLY on the photo (no backing plate) — the H7 construct."""
    flat = _photo_bg()
    card = _rect_mask((240, 40, 420, 180))
    badge = _rect_mask((360, 120, 420, 180))       # sits on the card → activates peel
    text = _rect_mask((30, 250, 200, 285))         # on the photo background, not the card
    flat[card] = (20, 120, 60)
    flat[badge] = (200, 40, 50)
    flat[text] = (250, 250, 250)
    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="card"),
        SceneElement(id="badge", mask=badge, z=1.0, kind="icon"),
        SceneElement(id="hdr", mask=text, z=2.0, kind="text", is_text=True),
    ]
    return flat, elements, {"card": card, "badge": badge, "text": text}


def test_h7_text_on_photo_plate_routes_to_inpaint_not_solid():
    flat, elements, masks = _h7_scene()
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(flat_fill_allow_background=True, flat_fill_text=True))
    assert not result.skipped                       # the element pair activated peel
    assert result.meta.get("plate_kind") == "photo"

    # Every background hole (both the element-class card/badge underfill and the
    # text-class headline hole) was routed to the injected inpainter — NEVER a solid
    # patch on the photo.
    backends = {(b["text_occluder"], b["backend"])
                for b in result.meta.get("fill_backends", [])}
    assert (True, "inpaint") in backends            # text hole → inpaint, not solid
    assert not any(b["backend"] == "solid" for b in result.meta.get("fill_backends", []))

    # The inpaint call for the text hole carried the correct ownership meta.
    bg_text = [c["meta"] for c in spy.calls
               if c["meta"].get("background") and c["meta"].get("text_occluder")]
    assert bg_text and bg_text[0]["under_kind"] == "photo"
    assert bg_text[0]["under_id"] == "background"

    # Out-of-mask invariant: any background pixel with nothing over it is byte-identical.
    union = masks["card"] | masks["badge"] | masks["text"]
    assert np.array_equal(result.background[~union], flat[~union])

    # Recomposite reproduces the input exactly (text footprint excluded — native text
    # renders on top; no ghost text is baked into the plate).
    rc = result.meta["recomposite"]
    assert rc["exact"] is True and rc["max_abs_diff"] == 0
    assert rc["text_excluded_px"] == int(masks["text"].sum())


def test_small_photo_holes_still_call_inpaint():
    """Tiny photo overlaps remain eligible for generative/Telea fill."""
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((20, 20, 400, 300))
    top = _rect_mask((100, 100, 130, 130))  # 900 px — under generative cap
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="panel", mask=under, z=0.0, kind="photo-fragment"),
                SceneElement(id="cover", mask=top, z=1.0, kind="icon")]
    spy = SpyInpaint()
    peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(max_generative_photo_hole_px=8000, abandon_photo_min_area=12000,
                 abandon_hole_frac=0.50, per_occluder_area=1, flat_fill_tol=0.0))
    assert any(c["meta"].get("under_id") == "panel"
               and not c["meta"].get("text_occluder") for c in spy.calls)


def test_perforated_top_still_activates_peel_over_solid_under():
    """Hollow occluder footprints must not block peel when the under-layer is solid
    (benchmark 009 skipped the whole scene because E011 was perforated)."""
    under = _rect_mask((40, 40, 200, 200))
    # Donut top: solid ring with large interior hole → high hole_frac (>25%).
    outer = _rect_mask((60, 60, 180, 180))
    inner = _rect_mask((85, 85, 155, 155))
    top = outer & ~inner
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[under] = BLUE
    flat[top] = RED
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="ring", mask=top, z=1.0, kind="icon")]
    report = peel_scene.overlap_report(elements, _cfg(min_overlap_area=64))
    assert report["eligibility"]["ring"]["as_top"] is True
    assert report["eligibility"]["ring"]["as_under"] is False
    assert report["eligibility"]["ring"]["eligible"] is False
    assert report["eligibility"]["card"]["eligible"] is True
    assert report["needed"] is True
    pair = next(p for p in report["pairs"] if p["qualifies"]
                and p["top"] == "ring" and p["under"] == "card")
    assert pair["eligible"] is True
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(),
                                   cfg=_cfg(min_overlap_area=64))
    assert not result.skipped
    assert result.layer("card") is not None


def test_large_element_holes_are_filled_per_occluder():
    flat = np.full((H, W, 3), BG, np.uint8)
    under = _rect_mask((10, 10, 430, 330))
    a = _rect_mask((20, 20, 120, 120))
    b = _rect_mask((200, 20, 300, 120))
    flat[under] = (200, 200, 200)
    flat[a] = RED
    flat[b] = GREEN
    elements = [SceneElement(id="card", mask=under, z=0.0, kind="shape"),
                SceneElement(id="A", mask=a, z=1.0, kind="icon"),
                SceneElement(id="B", mask=b, z=2.0, kind="icon")]
    spy = SpyInpaint()
    peel_scene.peel_scene(flat, elements, inpaint=spy,
                          cfg=_cfg(per_occluder_area=100, flat_fill_tol=0.0))
    under_calls = [c for c in spy.calls
                   if c["meta"].get("under_id") == "card"
                   and not c["meta"].get("text_occluder")]
    # Two separate element-class calls (one per large occluder), not one batch.
    assert len(under_calls) >= 2
    ids = {tuple(c["meta"].get("occluder_ids") or []) for c in under_calls}
    assert ("A",) in ids or ["A"] in [list(x) for x in ids]
    assert any(set(c["meta"].get("occluder_ids") or []) == {"A"} for c in under_calls)
    assert any(set(c["meta"].get("occluder_ids") or []) == {"B"} for c in under_calls)


# ── artifacts ──────────────────────────────────────────────────────────────────────

def test_write_outputs_manifest(tmp_path):
    flat, elements, _ = _scene()
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    manifest = peel_scene.write_outputs(result, str(tmp_path))
    with open(os.path.join(str(tmp_path), "peel_scene_manifest.json"), encoding="utf-8") as f:
        assert json.load(f) == manifest
    assert manifest["mode"] == "scene" and manifest["skipped"] is False
    assert manifest["background"]["z"] == 0
    assert [e["id"] for e in manifest["layers"]] == ["left", "right", "circle"]
    assert [e["z"] for e in manifest["layers"]] == [1, 2, 3]
    left = manifest["layers"][0]
    assert left["occluded_by"] == ["text", "circle"]
    assert {f["occluder"] for f in left["fills"]} == {"circle", "text"}
    assert left["filled_area"] == sum(f["area"] for f in left["fills"])
    for entry in manifest["layers"]:
        assert os.path.exists(os.path.join(str(tmp_path), entry["file"]))
    assert os.path.exists(os.path.join(str(tmp_path), "background.png"))


def test_write_pipeline_layers_shape(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    flat, elements, _ = _scene()
    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    layers = peel_scene.write_pipeline_layers(result, str(tmp_path))
    assert [l["id"] for l in layers] == ["P0", "P1", "P2"]          # back-to-front
    assert [l["fused_id"] for l in layers] == ["left", "right", "circle"]
    for layer in layers:
        assert set(layer) == {"id", "png", "box", "kind_hint", "fused_id"}
        path = os.path.join(str(tmp_path), layer["png"])
        assert os.path.exists(path)
        assert Image.open(path).size == (W, H)                       # full canvas RGBA
        assert Image.open(path).mode == "RGBA"
    assert layers[2]["kind_hint"] == "product"


# ── §10 top-down peel discipline (docs/PEEL-DECOMPOSITION.md §10) ────────────────────

def test_text_parallel_track_never_punches_text_by_default():
    """Rule 3: with the module default (text_parallel_track=True) OCR ink never
    enters a peel punch mask — layers keep original ink under text, no text fills
    are recorded, and recomposite stays exact."""
    flat, elements, masks = _scene()
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy,
                                   cfg=_cfg(text_parallel_track=True))
    assert not result.skipped
    left = result.layer("left")
    text_left = masks["text"] & masks["left"]
    # Original ink stays (no punch), the element hole still fills.
    assert np.all(left.rgba[:, :, :3][text_left] == INK)
    assert np.all(left.rgba[:, :, :3][masks["circle"] & masks["left"]]
                  == SENTINELS["left"])
    assert all(not f.text_occluder for f in left.fills)
    assert all(not f.text_occluder for f in result.background_fills)
    assert all(not call["meta"].get("text_occluder") for call in spy.calls)
    assert result.meta["text_parallel_track"] is True
    # Composite still reproduces the input byte-exactly (ink is in the layers).
    assert result.meta["recomposite"]["exact"]


def test_text_boxes_still_blind_context_on_the_parallel_track():
    """Rule 3 nuance: text never punches, but its box still masks the inpaint
    CONTEXT so ink cannot bleed into neighbouring fills."""
    flat, elements, masks = _scene()
    spy = SpyInpaint()
    peel_scene.peel_scene(flat, elements, inpaint=spy,
                          cfg=_cfg(text_parallel_track=True))
    left_calls = [c for c in spy.calls if c["meta"]["under_id"] == "left"]
    assert left_calls
    for call in left_calls:
        assert INK not in call["context_colors"]
        assert call["context_colors"] <= {BLUE}


def test_occlusion_levels_top_down_depth():
    a = SceneElement(id="a", mask=_rect_mask((10, 10, 200, 200)), z=0.0, kind="photo")
    b = SceneElement(id="b", mask=_rect_mask((20, 20, 150, 150)), z=1.0, kind="card")
    c = SceneElement(id="c", mask=_rect_mask((30, 30, 100, 100)), z=2.0, kind="product")
    t = SceneElement(id="t", mask=_rect_mask((15, 15, 60, 40)), z=3.0,
                     kind="text", is_text=True)
    lonely = SceneElement(id="lonely", mask=_rect_mask((300, 250, 400, 320)), z=0.5)
    levels = peel_scene.occlusion_levels([a, b, c, t, lonely])
    assert levels == {"c": 0, "b": 1, "a": 2, "lonely": 0}   # text never counts


def test_plan_max_iterations_refuses_deep_strata():
    a = SceneElement(id="a", mask=_rect_mask((10, 10, 200, 200)), z=0.0, kind="photo")
    b = SceneElement(id="b", mask=_rect_mask((20, 20, 150, 150)), z=1.0, kind="card")
    c = SceneElement(id="c", mask=_rect_mask((30, 30, 100, 100)), z=2.0, kind="product")
    kept, plan = peel_scene.plan_peel_iterations(
        [a, b, c], {"w": W, "h": H}, _cfg(max_iterations=2))
    assert [e.id for e in kept] == ["b", "c"]
    assert "max-iterations" in plan["refused"]["a"]
    kept3, plan3 = peel_scene.plan_peel_iterations(
        [a, b, c], {"w": W, "h": H}, _cfg(max_iterations=3))
    assert plan3["refused"] == {} and len(kept3) == 3


def test_plan_plate_band_and_punch_cap_stay_in_plate():
    band = SceneElement(id="band", mask=_rect_mask((0, 100, W, 160)), z=0.0,
                        kind="photo-fragment")           # full-width stratum, 18% area
    giant = SceneElement(id="giant", mask=_rect_mask((30, 10, 400, 330)), z=0.5,
                         kind="photo")                    # 79% of the canvas
    prod = SceneElement(id="prod", mask=_rect_mask((60, 120, 160, 200)), z=1.0,
                        kind="product")
    kept, plan = peel_scene.plan_peel_iterations(
        [band, giant, prod], {"w": W, "h": H},
        _cfg(plate_band_span_frac=0.95, plate_band_min_area_frac=0.05,
             max_punch_canvas_frac=0.30))
    assert [e.id for e in kept] == ["prod"]
    assert "plate-band" in plan["refused"]["band"]
    assert "punch-cap" in plan["refused"]["giant"]
    assert plan["punched_canvas_frac"] < 0.10


def test_plan_iteration_budget_admits_smaller_first():
    big = SceneElement(id="big", mask=_rect_mask((0, 0, 300, 200)), z=0.0)
    small = SceneElement(id="small", mask=_rect_mask((320, 0, 420, 60)), z=0.5)
    kept, plan = peel_scene.plan_peel_iterations(
        [big, small], {"w": W, "h": H}, _cfg(iter_mask_budget_frac=0.10))
    assert [e.id for e in kept] == ["small"]
    assert "iteration-budget" in plan["refused"]["big"]
    assert plan["per_iteration_punched_px"]["0"] == int(np.count_nonzero(small.mask))


def test_refused_band_dissolves_into_the_plate_and_holes_reattribute():
    """A refused stratum is not emitted and not punched; a kept occluder's hole
    over it attributes to the BACKGROUND plate instead."""
    flat, elements, masks = _scene()
    # Make the left portrait a full-width band so the plan refuses it.
    band = _rect_mask((0, 20, W, 320))
    flat[band] = BLUE
    flat[masks["right"]] = GREEN
    flat[masks["circle"]] = RED
    elements[0] = SceneElement(id="left", mask=band, z=0.0, kind="photo")
    spy = SpyInpaint()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg=_cfg(plate_band_span_frac=0.95, plate_band_min_area_frac=0.05,
                 text_parallel_track=True))
    assert not result.skipped
    assert result.layer("left") is None                     # not emitted
    assert "plate-band" in result.meta["iteration_plan"]["refused"]["left"]
    # The circle's left-side hole now belongs to the background plate.
    bg_ids = {f.occluder_id for f in result.background_fills}
    assert "circle" in bg_ids
    assert result.meta["recomposite"]["exact"]


def test_fill_backends_write_each_plate_pixel_at_most_once():
    """Rule 4 (dilate once, inpaint once): with a dilation ring that makes
    neighbouring punches overlap, the recorded write areas still sum to the area
    of their union — no pixel is ever re-inpainted."""
    flat, elements, masks = _scene()
    result = peel_scene.peel_scene(
        flat, elements, inpaint=SpyInpaint(),
        cfg=_cfg(hole_dilate_px=6, text_hole_dilate_px=8,
                 text_parallel_track=False, fill_cc_split=False))
    assert not result.skipped
    punched = int(result.meta.get("plate_punched_px") or 0)
    recorded = sum(entry["area"] for entry in result.meta.get("fill_backends") or [])
    assert punched > 0 and recorded == punched


def test_flux_budget_caps_calls_and_reroutes_overflow_to_lama():
    cfg = _cfg(allow_flux=True, flux_min_hole_px=100, flux_max_hole_px=10 ** 6,
               flux_budget=2, flux_max_hole_frac=0.0)
    state = {"used": 0, "overflow": 0, "budget": 2, "canvas_px": 10 ** 6}
    meta = {"under_kind": "photo", "hole_px": 5000, "flux_state": state}
    assert peel_scene.peel_inpaint_mode(cfg, meta) == "flux_comfy"
    assert peel_scene.peel_inpaint_mode(cfg, meta) == "flux_comfy"
    assert peel_scene.peel_inpaint_mode(cfg, meta) == "lama"     # budget spent
    assert state == {"used": 2, "overflow": 1, "budget": 2, "canvas_px": 10 ** 6}


def test_flux_per_hole_canvas_frac_cap():
    cfg = _cfg(allow_flux=True, flux_min_hole_px=100, flux_max_hole_px=10 ** 9,
               flux_budget=0, flux_max_hole_frac=0.25)
    meta = {"under_kind": "photo", "hole_px": 400_000, "canvas_px": 1_000_000}
    assert peel_scene.peel_inpaint_mode(cfg, meta) == "lama"     # 40% > 25% cap
    meta["hole_px"] = 100_000
    assert peel_scene.peel_inpaint_mode(cfg, meta) == "flux_comfy"


# ── printed-lockup lift (013 grüns bag) ────────────────────────────────────────────


def test_printed_lockup_product_over_plain_background_activates_peel():
    """A hero product flagged by fusion (printed_lockup) lifts off the plate even
    with no object-over-object pair — 013's bag after its printed logos were
    absorbed back into the raster."""
    bag = _rect_mask((60, 60, 180, 260))
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[bag] = RED
    elements = [SceneElement(id="bag", mask=bag, z=0.0, kind="photo-fragment",
                             meta={"role": "product", "printed_lockup": True})]
    report = peel_scene.overlap_report(elements, _cfg())
    assert report["needed"] is True
    assert report["lifted_products"] == ["bag"]

    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    assert not result.skipped
    assert [layer.id for layer in result.layers] == ["bag"]


def test_printed_lockup_lift_requires_a_solid_eligible_mask():
    """The granularity guard still gates the lift — a fragmented 'product' cannot
    switch peel on just because fusion flagged it."""
    frag = _fragmented_mask((20, 20, 220, 320))
    elements = [SceneElement(id="bag", mask=frag, z=0.0, kind="photo-fragment",
                             meta={"role": "product", "printed_lockup": True})]
    report = peel_scene.overlap_report(elements, _cfg())
    assert report["needed"] is False
    assert report["lifted_products"] == []


def test_unflagged_product_over_plain_background_still_skips():
    """Without the fusion flag, elements over plain background stay the plate's job."""
    bag = _rect_mask((60, 60, 180, 260))
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[bag] = RED
    elements = [SceneElement(id="bag", mask=bag, z=0.0, kind="photo-fragment",
                             meta={"role": "product"})]
    report = peel_scene.overlap_report(elements, _cfg())
    assert report["needed"] is False

    result = peel_scene.peel_scene(flat, elements, inpaint=SpyInpaint(), cfg=_cfg())
    assert result.skipped and result.skip_reason == "no-overlap"


# ── never punch OCR/artwork out of product interiors ───────────────────────────────


def test_text_never_punches_product_role_photo_fragment():
    """Label OCR sitting on a product cutout must leave the product alpha intact."""
    card = _rect_mask((20, 20, 300, 360))
    product = _rect_mask((40, 40, 220, 280))
    label = _rect_mask((80, 100, 180, 140))
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[card] = BLUE
    flat[product] = RED
    flat[label] = INK
    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="shape"),
        SceneElement(id="prod", mask=product, z=1.0, kind="photo-fragment",
                     meta={"role": "product"}),
        SceneElement(id="t", mask=label, z=2.0, kind="text", is_text=True),
    ]
    # Even with punch_text_into_photos forced ON, product-role ink is a hard deny.
    result = peel_scene.peel_scene(
        flat, elements, inpaint=SpyInpaint(),
        cfg=_cfg(punch_text_into_photos=True, punch_artwork_into_photos=True,
                 flat_fill_tol=6.0))
    assert not result.skipped
    prod = result.layer("prod")
    assert prod is not None
    assert not any(b.get("text_occluder") for b in (prod.meta.get("fill_backends") or []))
    # Flat already has label ink on the product — peel must keep those pixels
    # (alpha 255, original INK), not punch a hole or inpaint them away.
    assert np.all(prod.rgba[:, :, :3][label & product] == INK)
    assert np.all(prod.rgba[:, :, 3][label & product] == 255)


def test_artwork_never_punches_printed_lockup_product():
    """Absorbed on-pack logo must not carve a hole out of the product raster."""
    product = _rect_mask((40, 40, 220, 280))
    logo = _rect_mask((90, 90, 150, 130))
    card = _rect_mask((20, 20, 300, 360))
    flat = np.full((H, W, 3), BG, np.uint8)
    flat[card] = BLUE
    flat[product] = RED
    flat[logo] = INK
    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="shape"),
        SceneElement(id="prod", mask=product, z=1.0, kind="photo-fragment",
                     meta={"role": "product", "printed_lockup": True}),
        SceneElement(id="logo", mask=logo, z=2.0, kind="icon",
                     meta={"role": "logo"}),
    ]
    result = peel_scene.peel_scene(
        flat, elements, inpaint=SpyInpaint(),
        cfg=_cfg(punch_artwork_into_photos=True, flat_fill_tol=6.0))
    prod = result.layer("prod")
    assert prod is not None
    # Artwork occluder was denied — no fill backends attributed to the logo on prod.
    assert not any(
        "logo" in (b.get("occluder_ids") or [])
        for b in (prod.meta.get("fill_backends") or [])
    )
    assert np.all(prod.rgba[:, :, 3][logo & product] == 255)


def test_may_punch_into_hard_denies_product_ink():
    text = SceneElement(id="t", mask=_rect_mask((0, 0, 10, 10)), z=1.0,
                        kind="text", is_text=True)
    logo = SceneElement(id="l", mask=_rect_mask((0, 0, 10, 10)), z=1.0,
                        kind="icon", meta={"role": "logo"})
    opts = {"punch_text_into_photos": True, "punch_artwork_into_photos": True}
    assert peel_scene._may_punch_into(
        text, "photo-fragment", opts, under_meta={"role": "product"}) is False
    assert peel_scene._may_punch_into(
        logo, "photo-fragment", opts,
        under_meta={"role": "product", "printed_lockup": True}) is False
    # Generic photo panel (no product role) still honours the punch flag.
    assert peel_scene._may_punch_into(
        text, "photo-fragment", opts, under_meta={"role": "photo"}) is True
