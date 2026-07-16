"""CPU tests for removal-mask over-destruction and product wipeouts (postfix-benchmark-4).

Three defects from the benchmark-4 forensics (101 Craft Cadence tubes, 066 mascara
comparison, 104/107 empty product groups):

  * OVERSIZED RESIDUAL SHELLS PUNCH THE WHOLE PLATE — merge rejects a giant residual
    "text shell" (``meta.text_shell_rejected == "oversized-residual-shell"``) but the
    shell was still admitted as a removal observation covering its ENTIRE footprint. On
    101 the two half-canvas card shells owned ~75% of the removal union: the whole plate
    was inpainted, the flat cards became Big-LaMa mush, and changed_canvas_ratio hit 61%.
    A rejected shell re-renders as its own flat plate slice and every element on top of
    it (text/product/icon) removes its OWN ink, so the shell must not inpaint anything.
  * FULL-CANVAS SHELL SHIPS OVER THE PRODUCTS — the same rejected shell at full-canvas
    size also shipped as a top-of-stack "Photo" image whose product regions were white.
    It painted over the real product cutouts beneath it, wiping 101's tubes to outlines.
    Such a plate duplicate must drop to a plate passthrough (no src, no re-render).
  * UNCLAIMED REMOVALS ARE NEVER RESTORED — a removal-ledger region whose owner does not
    re-render (a plate passthrough, or an "emitted" image layer whose asset is blank —
    104/107 shipped 8KB empty product PNGs) leaves the plate as the ONLY visible surface,
    so it must hold the ORIGINAL pixels rather than an inpaint hole.

All CPU-only (opencv inpaint); no GPU backends are exercised.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import merge_layers, reconstruct  # noqa: E402


# ── oversized residual shells: never inpaint, never ship over a product ───────────────

def _shell_scene(tmp_path, shell_box=(0, 0, 60, 100)):
    """A flat card shell with a product cutout STRADDLING its edge.

    The product must straddle: a cutout wholly inside the shell mask is suppressed by the
    ordinary foreground-ownership audit ("fully-contained-in-foreground-owner"), which
    would mask the behaviour under test.
    """
    source = tmp_path / "source.png"
    img = Image.new("RGB", (120, 100), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((0, 0, 59, 99), fill=(0, 160, 176))  # flat teal card
    ImageDraw.Draw(img).rectangle((40, 30, 89, 69), fill=(120, 220, 40))  # green product
    img.save(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    x, y, w, h = shell_box
    Image.new("L", (w, h), 255).save(masks / "shell.png")
    Image.new("L", (50, 40), 255).save(masks / "product.png")
    candidates = [
        {"id": "shell", "target": "shape", "z": 0,
         "box": {"x": x, "y": y, "w": w, "h": h},
         "mask": {"src": "masks/shell.png"},
         "meta": {"role": "card", "confidence": 1.0,
                  "text_shell_rejected": "oversized-residual-shell"}},
        {"id": "product", "target": "image", "z": 3,
         "box": {"x": 40, "y": 30, "w": 50, "h": 40},
         "mask": {"src": "masks/product.png"},
         "meta": {"role": "product", "confidence": 0.9}},
    ]
    return source, candidates


def test_oversized_residual_shell_never_inpaints_its_footprint(tmp_path):
    """101: E001/E002 half-card shells owned ~75% of the union and mushed the plate.

    The shell re-renders as its own flat plate and its overlays remove their own ink, so
    admitting the whole shell footprint as a removal hole is redundant AND destructive.
    """
    source, candidates = _shell_scene(tmp_path)
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    shell = next(c for c in result["candidates"] if c["id"] == "shell")
    assert shell["meta"]["removal_skipped"] == "oversized-residual-shell"
    # The shell owns no removal ledger region ...
    assert "shell" not in set((result["removal_owner_index"] or {}).values())
    # ... so the flat teal card survives in the plate byte-identically (no mush).
    plate = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    src = np.asarray(Image.open(source).convert("RGB"))
    card_only = (slice(75, 99), slice(0, 39))  # teal card, away from the product hole
    assert np.array_equal(plate[card_only], src[card_only])
    # The product still gets its own clean removal hole (it stays a swappable cutout).
    assert "product" in set((result["removal_owner_index"] or {}).values())


def test_residual_shell_raster_drops_to_plate_passthrough(tmp_path):
    """101: c_E000 shipped as a top "Photo" whose product regions were white, painting
    over the tube cutouts underneath. A plate-duplicate shell must not re-render.

    A rejected shell with no flat fill resolves to an opaque raster, which is exactly the
    plate duplicate that wipes products — it must drop, losing src and mask so no
    downstream stage can re-emit those pixels over the real cutouts.
    """
    source, candidates = _shell_scene(tmp_path, shell_box=(0, 0, 120, 100))
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    shell = next(c for c in result["candidates"] if c["id"] == "shell")
    assert shell["target"] == "drop"
    assert shell["meta"]["plate_passthrough"] is True
    assert shell["meta"]["raster_fallback"] == "oversized-residual-shell-plate-passthrough"
    assert not shell.get("src") and not shell.get("mask")


# ── unclaimed-removal restore ────────────────────────────────────────────────────────

def test_removal_owner_rerenders_rejects_blank_assets(tmp_path):
    """104/107: an image layer whose asset is a blank PNG re-renders nothing, so the
    plate under it is the only visible surface — it must count as an absent owner."""
    blank = tmp_path / "blank.png"
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(blank)
    solid = tmp_path / "solid.png"
    Image.new("RGBA", (40, 40), (10, 200, 10, 255)).save(solid)

    empty = {"id": "e", "target": "image", "src": "blank.png", "meta": {"role": "product"}}
    filled = {"id": "f", "target": "image", "src": "solid.png", "meta": {"role": "product"}}
    assert reconstruct._removal_owner_rerenders(empty, str(tmp_path)) is False
    assert reconstruct._removal_owner_rerenders(filled, str(tmp_path)) is True
    # Native text and flat shapes always re-render; a plate passthrough never does.
    assert reconstruct._removal_owner_rerenders({"target": "text", "meta": {}}, str(tmp_path))
    assert reconstruct._removal_owner_rerenders({"target": "shape", "meta": {}}, str(tmp_path))
    assert not reconstruct._removal_owner_rerenders(
        {"target": "image", "meta": {"plate_passthrough": True}}, str(tmp_path))


def test_restore_unclaimed_removal_returns_source_pixels_and_shrinks_union(tmp_path):
    """An unclaimed removal region is restored to source AND leaves the union, so the
    plate-integrity invariant (out-of-mask == source) still holds afterwards."""
    src = np.zeros((20, 20, 3), dtype=np.uint8)
    src[:, :] = (200, 30, 40)
    plate_path = tmp_path / "background_clean.png"
    plate = np.zeros_like(src)  # a fully "inpainted" (black) plate
    Image.fromarray(plate).save(plate_path)

    region = np.zeros((20, 20), dtype=np.uint8)
    region[4:12, 4:12] = 255
    union = region.copy()
    ledger = (region > 0).astype(np.uint16) * 7
    removal = [{"id": "ghost", "mask_array": region}]
    candidates = [{"id": "ghost", "target": "drop", "meta": {"plate_passthrough": True}}]

    out = reconstruct._restore_unclaimed_removals(
        src, str(plate_path), removal, candidates, str(tmp_path), union, ledger, {},
    )
    assert out["regions"] == 1 and out["restored_px"] == 64 and out["ids"] == ["ghost"]
    restored = np.asarray(Image.open(plate_path).convert("RGB"))
    assert np.array_equal(restored[4:12, 4:12], src[4:12, 4:12])  # source pixels back
    assert not union[4:12, 4:12].any()      # cleared from the removal union
    assert not ledger[4:12, 4:12].any()     # and from the ownership ledger


def test_restore_keeps_inpaint_for_owners_that_rerender(tmp_path):
    """Conservative: a product that DOES ship a real asset keeps its clean plate hole."""
    src = np.full((20, 20, 3), 200, dtype=np.uint8)
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(np.zeros_like(src)).save(plate_path)
    Image.new("RGBA", (8, 8), (10, 200, 10, 255)).save(tmp_path / "asset.png")

    region = np.zeros((20, 20), dtype=np.uint8)
    region[4:12, 4:12] = 255
    union = region.copy()
    removal = [{"id": "prod", "mask_array": region}]
    candidates = [{"id": "prod", "target": "image", "src": "asset.png",
                   "meta": {"role": "product"}}]

    out = reconstruct._restore_unclaimed_removals(
        src, str(plate_path), removal, candidates, str(tmp_path), union, None, {},
    )
    assert out["regions"] == 0 and out["restored_px"] == 0
    assert union[4:12, 4:12].all()  # union untouched — the hole stays inpainted


def test_restore_config_gate(tmp_path):
    src = np.full((10, 10, 3), 90, dtype=np.uint8)
    plate_path = tmp_path / "background_clean.png"
    Image.fromarray(np.zeros_like(src)).save(plate_path)
    region = np.zeros((10, 10), dtype=np.uint8)
    region[2:6, 2:6] = 255
    union = region.copy()
    out = reconstruct._restore_unclaimed_removals(
        src, str(plate_path), [{"id": "g", "mask_array": region}],
        [{"id": "g", "target": "drop", "meta": {}}], str(tmp_path), union, None,
        {"reconstruct": {"restore_unclaimed_removals": False}},
    )
    assert out["regions"] == 0
    assert union[2:6, 2:6].all()  # gate off → nothing restored


# ── 066: an oversized loose residual may not own printed text ────────────────────────

def _bullet_ocr(box):
    return {"lines": [{"id": "L0", "text": "Tubing technology", "conf": 0.95,
                       "box": box, "role": "body"}]}


def test_oversized_low_confidence_photo_is_not_a_product_cutout_owner():
    """066: a conf-0.405 photo-fragment spanning 75% of canvas claimed every white-card
    checklist bullet as "text-inside-product-cutout"; the plate inpaint then erased all
    of them. Product FACES are bounded/confident — card interiors are not."""
    canvas = {"w": 1440, "h": 1440}
    loose = [{"id": "E000", "box": {"x": 20, "y": 277, "w": 1400, "h": 1117},
              "kind": "photo-fragment", "area": 1563800, "coverage": 0.754,
              "source": "residual-cc", "role": "photo", "score": 0.405}]
    out = merge_layers.merge(_bullet_ocr({"x": 200, "y": 900, "w": 400, "h": 40}),
                             loose, [], canvas, {})
    bullet = next(c for c in out if c["id"] == "c_L0")
    assert bullet["meta"].get("suppression_reason") != "text-inside-product-cutout"
    assert not bullet.get("kept_in_photo")  # stays an editable native TEXT owner
    residual = next(c for c in out if c["id"] == "c_E000")
    assert residual["meta"].get("oversized_loose_residual") is True


def test_bounded_confident_product_still_owns_its_printed_label():
    """Counter-case (135/067): a real, bounded, confident product cutout MUST keep owning
    the text printed on its face — nutrition/ingredient copy on a pack is correctly baked
    and must NOT be freed into a native layer by the 066 guard."""
    canvas = {"w": 1000, "h": 1000}
    product = [{"id": "E000", "box": {"x": 300, "y": 300, "w": 250, "h": 400},
                "kind": "photo-fragment", "area": 100000, "coverage": 0.10,
                "source": "sam3", "role": "product", "score": 0.95}]
    out = merge_layers.merge(_bullet_ocr({"x": 320, "y": 500, "w": 160, "h": 30}),
                             product, [], canvas, {})
    label = next(c for c in out if c["id"] == "c_L0")
    assert label.get("kept_in_photo") is True
    assert label["meta"].get("baked_owner_id") == "c_E000"
    residual = next(c for c in out if c["id"] == "c_E000")
    assert not residual["meta"].get("oversized_loose_residual")


def test_residual_shell_passthrough_declares_its_suppression_reason(tmp_path):
    """A reasoned plate-passthrough drop must be legible to scene_intent reconciliation.

    The shell drop is DELIBERATE (the clean plate owns those pixels), but it only recorded
    `removal_skipped`/`plate_passthrough` -- neither of which `scene_intent.
    _is_explicit_suppression` reads (it keys on keep_in_background / suppression_reason /
    removal_required). So a PLANNED id that took this path looked like an unexplained drop,
    raised SceneIntentError ("planned ids became drop: c_E003"), and the whole
    structure-first tree was discarded for the legacy layout -- a hard `structure-unavailable`
    on 002 and 101. 101's legacy fallback then shipped 20 raster leaves
    (native_leaf_ratio 0.375 vs 0.80 structure-first): SSIM propped up by copied pixels.
    """
    source, candidates = _shell_scene(tmp_path, shell_box=(0, 0, 120, 100))
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    shell = next(c for c in result["candidates"] if c["id"] == "shell")

    assert shell["target"] == "drop"
    assert shell["meta"]["plate_passthrough"] is True
    # The reason is stated in the vocabulary the reconciler actually reads.
    assert shell["meta"]["suppression_reason"] == "oversized-residual-shell"
    assert shell["meta"]["keep_in_background"] is True

    from src import scene_intent
    assert scene_intent._is_explicit_suppression(shell) is True
