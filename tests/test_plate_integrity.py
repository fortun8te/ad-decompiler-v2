"""Regression tests for the 002 plate-integrity failure mode.

002 wiped ~85% of the canvas with Big-LaMa/Flux on a giant background union, erased
product cutouts into generative smear, and left glyph haze on flat chrome. These
CPU-only synthetics lock the peel / reconstruct / QA contracts that prevent that class.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import inpaint, peel_scene, pixel_diff, reconstruct  # noqa: E402
from src.peel_scene import SceneElement  # noqa: E402
from src.reconstruct import _is_background_plate  # noqa: E402

GENERATIVE = frozenset({"flux", "flux_comfy", "flux-comfy", "lama", "big-lama", "big_lama"})


def _frac_cfg(**peel):
    """Enable canvas-frac budget even on small synthetic canvases."""
    peel.setdefault("min_canvas_for_frac_px", 0)
    peel.setdefault("allow_flux", True)
    return {"peel": peel}


# ── 1) Giant background hole → never Big-LaMa / Flux ───────────────────────────────


def test_background_hole_over_40pct_canvas_does_not_select_generative():
    """Synthetic: hole_px > 40% of canvas must bake — never big-lama / flux."""
    canvas_px = 100_000
    hole_px = 41_000  # 41% of canvas
    assert hole_px / canvas_px > 0.40
    cfg = _frac_cfg(max_generative_canvas_frac=0.40)

    assert peel_scene.exceeds_generative_hole_budget(
        hole_px, canvas_px, "background", cfg=cfg) is True

    backend = peel_scene.resolve_fill_backend(
        hole_px, "background", canvas_px=canvas_px, cfg=cfg,
    )
    assert backend in ("bake", "abandon")
    assert backend not in GENERATIVE

    mode = peel_scene.peel_inpaint_mode(
        cfg,
        {"under_kind": "background", "hole_px": hole_px, "canvas_px": canvas_px},
    )
    assert mode == "bake"
    assert mode not in GENERATIVE


def test_background_hole_past_absolute_bg_ceiling_bakes():
    """Absolute bg ceiling (default 80k) also blocks generative — independent of frac."""
    hole_px = 90_000  # > default max_generative_bg_hole_px 80k
    canvas_px = 500_000
    assert peel_scene.exceeds_generative_hole_budget(
        hole_px, canvas_px, "background") is True
    backend = peel_scene.resolve_fill_backend(
        hole_px, "background", canvas_px=canvas_px,
        cfg={"peel": {"allow_flux": True}},
    )
    assert backend in ("bake", "abandon")
    mode = peel_scene.peel_inpaint_mode(
        {"peel": {"allow_flux": True}},
        {"under_kind": "background", "hole_px": hole_px, "canvas_px": canvas_px},
    )
    assert mode == "bake"


def test_giant_background_union_bakes_in_peel_scene_without_inpaint_call():
    """End-to-end: >40% canvas punch into the background plate → bake, no LaMa/Flux.

    Peel only activates on an eligible object pair; the true background plate is then
    filled for chrome that may punch. A large non-product chrome hole past the 40%
    budget must bake — never a generative spy call under=background.
    """
    h, w = 200, 200
    canvas_px = h * w
    flat = np.full((h, w, 3), 240, np.uint8)
    card = np.zeros((h, w), bool)
    card[10:70, 10:70] = True
    icon = np.zeros((h, w), bool)
    icon[20:50, 20:50] = True
    # Large chrome badge (not product) covering >40% — allowed to attempt bg punch,
    # then the generative budget must bake instead of LaMa/Flux.
    badge = np.zeros((h, w), bool)
    badge[20:180, 20:180] = True
    assert badge.sum() / canvas_px > 0.40
    flat[card] = (200, 200, 200)
    flat[icon] = (40, 80, 200)
    flat[badge] = (200, 40, 40)

    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="shape"),
        SceneElement(id="icon", mask=icon, z=1.0, kind="icon"),
        SceneElement(id="badge", mask=badge, z=2.0, kind="icon",
                     meta={"role": "badge"}),
    ]
    calls = []

    def spy(rgb, mask, meta=None):
        calls.append(dict(meta or {}))
        out = rgb.copy()
        out[mask] = (1, 2, 3)
        return out

    # Raise punch-area cap so the badge is eligible to attempt a bg punch; the
    # generative frac budget (40%) must still force bake.
    cfg = _frac_cfg(
        flat_fill_tol=0.0, flat_fill_allow_background=False,
        per_occluder_area=1, hole_dilate_px=0, fail_closed_to_flat=False,
        max_generative_canvas_frac=0.40, max_bg_punch_area_frac=0.70,
    )
    result = peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=cfg)
    assert not result.skipped
    # Large badge hole on the background plate must bake — never generative.
    assert not any(
        c.get("under_id") == "background"
        and "badge" in (c.get("occluder_ids") or [])
        for c in calls
    )
    bg_fills = [b for b in (result.meta.get("fill_backends") or [])
                if not b.get("text_occluder") and "badge" in (b.get("occluder_ids") or [])]
    assert bg_fills and all(b["backend"] == "baked" for b in bg_fills)
    assert result.meta.get("baked_large_bg_hole") is True


# ── 2) Product-role region is not background generative fill ───────────────────────


def test_product_role_must_not_punch_background_generative():
    """Product cutouts never become background inpaint holes (002 wipe class)."""
    product = {"box": {"x": 10, "y": 10, "w": 700, "h": 700},
               "meta": {"role": "product"}}
    assert not _is_background_plate(product, 1000, 1000)

    mask = np.zeros((200, 200), bool)
    mask[20:180, 20:180] = True
    occluder = SceneElement(id="prod", mask=mask, z=1.0, kind="product",
                            meta={"role": "product"})
    opts = peel_scene._options({})
    assert peel_scene._may_punch_background(occluder, opts, canvas_px=200 * 200) is False

    # Mid-band product under-layer may use Flux (photo path) — never as background.
    mid = peel_scene.resolve_fill_backend(
        20_000, "product", canvas_px=200_000,
        cfg={"peel": {"allow_flux": True, "flux_min_hole_px": 4000,
                     "flux_max_hole_px": 220000}},
    )
    assert mid == "flux"
    assert mid != "lama"

    # Past absolute generative ceiling: bake.
    huge = peel_scene.resolve_fill_backend(
        230_000, "product", canvas_px=1_000_000,
        cfg={"peel": {"allow_flux": True}},
    )
    assert huge in ("bake", "abandon")
    assert huge not in GENERATIVE


def test_product_is_skipped_as_background_occluder_in_peel():
    """Peel records product as skipped_bg_occluder — no generative bg fill for it."""
    h, w = 200, 200
    flat = np.full((h, w, 3), 240, np.uint8)
    card = np.zeros((h, w), bool)
    card[10:70, 10:70] = True
    icon = np.zeros((h, w), bool)
    icon[20:50, 20:50] = True
    product = np.zeros((h, w), bool)
    product[40:180, 40:180] = True  # large product mass
    flat[card] = (200, 200, 200)
    flat[icon] = (40, 80, 200)
    flat[product] = (180, 60, 40)

    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="shape"),
        SceneElement(id="icon", mask=icon, z=1.0, kind="icon"),
        SceneElement(id="product", mask=product, z=2.0, kind="product",
                     meta={"role": "product"}),
    ]
    calls = []

    def spy(rgb, mask, meta=None):
        calls.append(dict(meta or {}))
        return rgb.copy()

    result = peel_scene.peel_scene(
        flat, elements, inpaint=spy,
        cfg={"peel": {"flat_fill_tol": 0.0, "hole_dilate_px": 0,
                     "fail_closed_to_flat": False, "allow_flux": True}},
    )
    assert not result.skipped
    skipped = result.meta.get("skipped_bg_occluders") or []
    assert any(s.get("id") == "product" for s in skipped)
    assert not any(c.get("under_id") == "background"
                   and "product" in (c.get("occluder_ids") or [])
                   for c in calls)


def test_product_cutout_reconstruct_stays_image_not_background_generative(tmp_path):
    """Reconstruct keeps a product as a swappable image; plate under it is not Flux."""
    source = tmp_path / "source.png"
    img = Image.new("RGB", (160, 120), (238, 232, 220))
    ImageDraw.Draw(img).ellipse((40, 20, 120, 100), fill=(180, 60, 40))
    img.save(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (80, 80), 255).save(masks / "product.png")
    candidates = [{
        "id": "product", "target": "image", "z": 1,
        "box": {"x": 40, "y": 20, "w": 80, "h": 80},
        "mask": {"src": "masks/product.png"},
        "meta": {"role": "product", "confidence": 0.9},
    }]
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, candidates, str(tmp_path),
        {"inpaint": {"mode": "opencv"}, "scene": {"archetype": "product_on_flat"}},
    )
    by_id = {c["id"]: c for c in result["candidates"]}
    prod = by_id["product"]
    assert prod["target"] == "image"
    assert (prod.get("meta") or {}).get("role") == "product"
    assert not _is_background_plate(prod, 160, 120)
    # Product remains a foreground image layer; plate pixels are kept (no wipe).
    assert (prod.get("meta") or {}).get("keep_in_background") is True
    assert (prod.get("meta") or {}).get("plate_keep_reason") == "product-photo-cutout"
    # Product remains a foreground image layer — never a Flux background wipe.
    backend = str((result["stats"].get("inpaint") or {}).get("backend") or "").lower()
    assert "flux" not in backend


# ── 3) Text hole on flat plate → solid / telea, never Flux ─────────────────────────


def test_text_hole_on_flat_plate_routes_to_telea_or_lama_not_flux():
    """Text occluders never select Flux; flat-plate policy prefers solid upstream."""
    backend = peel_scene.resolve_fill_backend(
        800, "shape", text_occluder=True, canvas_px=50_000,
        cfg={"peel": {"allow_flux": True}, "scene": {"archetype": "product_on_flat"}},
    )
    # Router returns lama; pipeline adapter may Telea text holes before that.
    assert backend in ("lama", "opencv", "telea", "solid")
    assert backend not in {"flux", "flux_comfy", "flux-comfy"}

    mode = peel_scene.peel_inpaint_mode(
        {"peel": {"allow_flux": True}, "scene": {"archetype": "product_on_flat"}},
        {"under_kind": "shape", "hole_px": 800, "text_occluder": True, "canvas_px": 50_000},
    )
    assert mode != "flux_comfy"

    policy = peel_scene.resolve_peel_fill_policy(
        {"scene": {"archetype": "product_on_flat"}},
        under_kind="shape", text_occluder=True,
    )
    assert policy["prefer_flat"] is True
    assert policy["backend"] == "flat"


def test_text_hole_on_flat_plate_solid_fills_in_peel_and_inpaint():
    """Peel solid-fills text on chrome; inpaint_array solid-flats flat UI plates."""
    h, w = 200, 240
    flat = np.full((h, w, 3), (18, 18, 18), np.uint8)
    card = np.zeros((h, w), bool)
    card[20:180, 20:220] = True
    text = np.zeros((h, w), bool)
    text[70:100, 60:180] = True
    flat[text] = (240, 240, 240)
    icon = np.zeros((h, w), bool)
    icon[130:170, 40:80] = True
    flat[icon] = (200, 40, 40)

    elements = [
        SceneElement(id="card", mask=card, z=0.0, kind="shape"),
        SceneElement(id="t", mask=text, z=1.0, kind="text", is_text=True),
        SceneElement(id="icon", mask=icon, z=2.0, kind="icon"),
    ]
    calls = []

    def spy(rgb, mask, meta=None):
        calls.append(dict(meta or {}))
        return rgb.copy()

    cfg = {"peel": {
        "flat_fill_tol": 8.0, "flat_fill_text": True, "flat_fill_min_visible_frac": 0.0,
        "hole_dilate_px": 0, "text_hole_dilate_px": 0, "fail_closed_to_flat": False,
        "allow_flux": True,
    }, "scene": {"archetype": "product_on_flat"}}
    result = peel_scene.peel_scene(flat, elements, inpaint=spy, cfg=cfg)
    card_layer = result.layer("card")
    text_backends = [b["backend"] for b in card_layer.meta["fill_backends"]
                     if b.get("text_occluder")]
    assert text_backends and all(b == "solid" for b in text_backends)
    assert not any(c.get("text_occluder") and c.get("under_id") == "card" for c in calls)

    rgb = np.full((80, 120, 3), (18, 18, 18), dtype=np.uint8)
    rgb[30:50, 40:80] = (240, 240, 240)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 40:80] = 255
    out, used, diag = inpaint.inpaint_array(
        rgb, mask,
        {"scene": {"archetype": "product_on_flat"}, "inpaint": {"mode": "flux"}},
        return_diagnostics=True,
    )
    assert used == "solid-flat"
    assert "flux" not in used
    assert diag["solid_flat"]["flat_filled"] >= 1
    assert np.abs(out[30:50, 40:80].astype(int) - 18).max() <= 6


# ── 4) QA: excessive-plate-destruction still fires on a destroyed plate ────────────


def _qa_rules(result):
    return {item["rule"] for item in result.get("hard_fails", [])}


def test_excessive_plate_destruction_fires_on_synthetic_destroyed_plate(tmp_path):
    """Optional F3 gate: synthetic 002-class wipe must hard-fail QA."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    removal = tmp_path / "removal_mask.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 94, 94), fill=(20, 40, 90))
    image.save(source)
    image.save(render)
    Image.new("RGB", (100, 100), (128, 128, 128)).save(background)
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((5, 5, 94, 94), fill=255)
    mask.save(removal)
    design = {"layers": [
        {"id": "background", "type": "image", "src": "background_clean.png",
         "meta": {"role": "background", "source": "inpaint"}},
        {"id": "panel", "type": "shape", "box": {"x": 5, "y": 5, "w": 90, "h": 90}},
    ], "meta": {"editable_ratio": 0.5}}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "excessive-plate-destruction" in _qa_rules(result)
    assert result["structural"]["background"]["changed_canvas_ratio"] > 0.55
