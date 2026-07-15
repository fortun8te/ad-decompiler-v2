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
    flat, elements, masks = _scene()
    spy = SpyInpaint()
    result = peel_scene.peel_scene(flat, elements, inpaint=spy,
                                   cfg=_cfg(flat_fill_tol=6.0))
    # Element-class holes are all flat here — only TEXT-class holes reach the
    # inpainter (flat-fill is element-class only; text stays with the router).
    assert all(c["meta"]["text_occluder"] for c in spy.calls)
    left = result.layer("left")
    hole = masks["circle"] & masks["left"]
    assert np.all(left.rgba[:, :, :3][hole] == BLUE)       # exact, no Telea blur
    right = result.layer("right")
    assert np.all(right.rgba[:, :, :3][masks["circle"] & masks["right"]] == GREEN)
    backends = {b["text_occluder"]: b["backend"] for b in left.meta["fill_backends"]}
    assert backends == {False: "solid", True: "inpaint"}
    assert result.meta["recomposite"]["exact"] is True


def test_flat_fill_defers_to_the_inpainter_on_textured_context():
    rng = np.random.default_rng(7)
    flat = rng.integers(0, 255, (H, W, 3), np.uint8)       # loud texture everywhere
    under = _rect_mask((20, 20, 220, 320))
    top = _circle_mask()
    elements = [SceneElement(id="under", mask=under, z=0.0),
                SceneElement(id="top", mask=top, z=1.0)]
    spy = SpyInpaint()
    peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=_cfg(flat_fill_tol=6.0))
    assert any(c["meta"]["under_id"] == "under" for c in spy.calls)


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
