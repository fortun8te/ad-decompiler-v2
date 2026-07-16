"""CPU tests for the text-ghost / double-render and plate over-destruction fixes.

Covers the three mechanisms the benchmark forensics exposed:

  * smear-aware residue detection — a re-inpaint that *smears* leftover glyph ink
    (067 "WE'RE SAYING GOODBYE" stayed red, 021 sticky-note copy) shifts the pixel
    value out of source-match range but is visibly still inky; the old source-match
    metric read that as "resolved" and shipped a double render;
  * iterate-then-solid-fill — the audit runs several re-inpaint passes and, if
    residue survives, paints a deterministic local plate colour under the glyphs
    (Codia: keep native TEXT; never bake OCR into a raster slice for SSIM);
  * per-candidate removal cap — one low-confidence backdrop blob (002 c_E003 claimed
    34.5% of the canvas) must not inpaint most of the plate.

All CPU-only; inpaint backends are monkeypatched so behaviour is deterministic.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import build_design_json, reconstruct, render_preview  # noqa: E402
from src.schema import dump, load  # noqa: E402

PLATE = (238, 232, 220)
INK = (20, 20, 20)


def _source(path, size=(200, 140)):
    image = Image.new("RGB", size, PLATE)
    ImageDraw.Draw(image).rectangle((45, 48, 135, 65), fill=INK)  # glyph-like ink bar
    image.save(path)
    return image


def _text_candidate(**overrides):
    candidate = {
        "id": "c_B0", "target": "text", "text": "SALE", "z": 4,
        "box": {"x": 45, "y": 48, "w": 91, "h": 18},
        "visible_box": {"x": 45, "y": 48, "w": 91, "h": 18},
        "style": {"fontSize": 16, "fontFamily": "Arial", "color": "#141414"},
        "meta": {"source": "ocr", "role": "headline", "line_ids": ["L0"]},
    }
    candidate.update(overrides)
    return candidate


# ── smear-aware residue detection + iterate-then-force-raster ────────────────────────


def test_smeared_ghost_is_flagged_then_solid_filled(tmp_path, monkeypatch):
    """A smear that the OLD source-match metric would miss must (a) be flagged, and
    (b) after a short re-inpaint ladder fails, be cleared with a plate-colour fill while
    the layer stays native TEXT (never force-rasterized by default)."""
    source = tmp_path / "source.png"
    _source(source)
    # 38 is 18 away from the ink colour (20): outside the source-match tolerance (12)
    # but well inside ink_tolerance (40) — i.e. "still visibly inky", a real ghost.
    smear = np.array([38, 38, 38], dtype=np.uint8)
    calls = {"once": 0, "repair": 0}

    def fake_once(image_path, mask, output_path, cfg=None):
        calls["once"] += 1
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        img[np.asarray(reconstruct.inpaint.solidify_mask(mask)) > 0] = smear
        Image.fromarray(img).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-smear",
                "backend_class": "active"}

    def fake_role_aware(image_path, masks, output_path, cfg=None):
        calls["repair"] += 1  # every pass re-smears: the ghost never clears
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        img[np.asarray(masks["text"]) > 0] = smear
        Image.fromarray(img).save(output_path)
        return {"ok": True, "backend": "fake-smear-repair"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)
    monkeypatch.setattr(reconstruct.inpaint, "inpaint_role_aware", fake_role_aware)

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        # Non-flat archetype + explicit reinpaint budget so we exercise the ladder,
        # then solid-fill closer (Codia: keep native TEXT).
        {"inpaint": {"mode": "opencv"},
         "scene": {"archetype": "lifestyle_overlay"},
         "reconstruct": {"text_residual": {"reinpaint_max_passes": 1}}},
    )

    residual = result["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] and residual["flagged"][0]["id"] == "c_B0"
    assert residual["passes"] == 1
    assert calls["repair"] == 1
    # Codia: solid plate fill clears the ghost; native TEXT stays (no force_raster).
    assert residual["flagged"][0]["resolved"] is True
    assert residual["flagged"][0].get("resolved_by") == "solid-plate-fill"
    assert "c_B0" in (residual.get("solid_filled_ids") or [])
    assert residual.get("force_raster_ids", []) == []


def test_flat_ui_ghost_skips_reinpaint_and_solid_fills(tmp_path, monkeypatch):
    """social_screenshot / product_on_flat: no generative reinpaint — solid fill first."""
    source = tmp_path / "source.png"
    _source(source)
    calls = {"repair": 0}

    def fake_once(image_path, mask, output_path, cfg=None):
        Image.open(image_path).save(output_path)  # leave full residue
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    def fake_role_aware(image_path, masks, output_path, cfg=None):
        calls["repair"] += 1
        Image.open(image_path).save(output_path)
        return {"ok": True, "backend": "fake-repair"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)
    monkeypatch.setattr(reconstruct.inpaint, "inpaint_role_aware", fake_role_aware)

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"},
         "scene": {"archetype": "social_screenshot"}},
    )
    residual = result["stats"]["text_residual"]
    assert residual["passes"] == 0
    assert calls["repair"] == 0
    assert residual.get("solid_fill_first") is True
    assert residual["flagged"][0]["resolved"] is True
    assert residual["flagged"][0].get("resolved_by") == "solid-plate-fill"
    assert residual.get("force_raster_ids", []) == []


def test_residue_resolved_on_a_later_pass_is_not_forced(tmp_path, monkeypatch):
    """Thin residue that a second pass cleans must resolve without a raster slice."""
    source = tmp_path / "source.png"
    _source(source)
    calls = {"repair": 0}

    def fake_once(image_path, mask, output_path, cfg=None):
        Image.open(image_path).save(output_path)  # leave every glyph (full residue)
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    def fake_role_aware(image_path, masks, output_path, cfg=None):
        calls["repair"] += 1
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        fill = np.array([30, 30, 30]) if calls["repair"] == 1 else np.array(PLATE)
        img[np.asarray(masks["text"]) > 0] = fill  # pass 1 still inky, pass 2 clean
        Image.fromarray(img).save(output_path)
        return {"ok": True, "backend": "fake-repair"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)
    monkeypatch.setattr(reconstruct.inpaint, "inpaint_role_aware", fake_role_aware)

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"},
         "scene": {"archetype": "lifestyle_overlay"},
         "reconstruct": {"text_residual": {
             "solid_fill_first": False,
             "reinpaint_max_passes": 3,
             "solid_fill_residue": False,
         }}},
    )

    residual = result["stats"]["text_residual"]
    assert residual["passes"] == 2
    assert calls["repair"] == 2
    assert residual["flagged"][0]["resolved"] is True
    assert residual.get("force_raster_ids", []) == []


def test_reinpaint_can_be_disabled(tmp_path, monkeypatch):
    """With reinpaint off the audit still flags residue but performs no repair pass."""
    source = tmp_path / "source.png"
    _source(source)

    def fake_once(image_path, mask, output_path, cfg=None):
        Image.open(image_path).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"},
         "reconstruct": {"text_residual": {
             "reinpaint": False, "solid_fill_residue": False,
         }}},
    )
    residual = result["stats"]["text_residual"]
    assert residual["flagged"] and residual["flagged"][0]["resolved"] is False
    assert residual["reinpainted"] is False
    assert residual["passes"] == 0


# ── end-to-end: readable TEXT is never force-sliced (Codia) ──────────────────────────


def _build_text_run(tmp_path):
    source = tmp_path / "source.png"
    _source(source)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv", "opencv_radius": 4}},
    )
    tree = [c for c in result["candidates"] if c.get("target") != "drop"]
    dump(tree, os.path.join(str(tmp_path), "layout.json"))
    build_design_json.build(
        tree, {"w": 200, "h": 140}, str(tmp_path),
        base_src=os.path.join(str(tmp_path), result["background"]),
        doc_id="t", name="t",
    )
    render_preview.render(os.path.join(str(tmp_path), "design.json"), str(tmp_path))
    return source


def test_force_raster_ids_cannot_slice_readable_text_by_default(tmp_path):
    """Even a legacy force_raster_ids entry must NOT bake readable TEXT unless the
    forensic text_slice_gate_enabled flag is on (wrong Inter beats baked pixels)."""
    source = _build_text_run(tmp_path)
    recon_path = os.path.join(str(tmp_path), "reconstruction.json")
    recon = load(recon_path)
    recon["stats"]["text_residual"] = {
        "enabled": True, "checked": 1, "reinpainted": True, "passes": 3,
        "flagged": [{"id": "c_B0", "residual_px": 900, "residual_ratio": 0.9,
                     "resolved": False, "hard_fail": True}],
        "force_raster_ids": ["c_B0"],
    }
    dump(recon, recon_path)

    permissive = {"region_ssim_min": 0.0, "region_color_min": 0.0,
                  "text_ink_iou_min": 0.0, "text_ink_excess_max": 1e9}
    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"}, "fallback": dict(permissive)},
    )

    assert report["slices"] == []
    assert any(s.get("reason") == "codia-never-slice-readable-text"
               for s in report.get("skipped") or [])
    # Forensic opt-in still available for tooling.
    forensic = dict(permissive)
    forensic["text_slice_gate_enabled"] = True
    report2 = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"}, "fallback": forensic},
    )
    assert [entry["id"] for entry in report2["slices"]] == ["c_B0"]
    assert report2.get("residue_resolved_by_slice") == ["c_B0"]


# ── per-candidate removal cap (plate over-destruction) ───────────────────────────────


def _blob_source(path, size=(120, 100)):
    Image.new("RGB", size, PLATE).save(path)
    return path


def test_low_confidence_giant_blob_is_capped_out_of_removal(tmp_path):
    """A big, low-confidence, generic-role raster (002 c_E003 signature) must stay in
    the plate rather than inpaint a third of the canvas; a genuine foreground cutout of
    the same size is still removed so it remains a swappable asset."""
    source = tmp_path / "source.png"
    _blob_source(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (100, 40), 255).save(masks / "blob.png")
    Image.new("L", (100, 40), 255).save(masks / "product.png")
    candidates = [
        # ~33% of the canvas, generic "photo" role, low detector confidence → capped.
        {"id": "blob", "target": "image", "z": 0,
         "box": {"x": 5, "y": 5, "w": 100, "h": 40},
         "mask": {"src": "masks/blob.png"},
         "meta": {"role": "photo", "confidence": 0.40}},
        # Same size, but a semantic foreground role → exempt, still removed.
        {"id": "product", "target": "image", "z": 0,
         "box": {"x": 5, "y": 55, "w": 100, "h": 40},
         "mask": {"src": "masks/product.png"},
         "meta": {"role": "product", "confidence": 0.40}},
    ]

    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    by_id = {c["id"]: c for c in result["candidates"]}

    assert result["stats"]["removal_capped"] == 1
    assert by_id["blob"]["meta"].get("keep_in_background") is True
    assert by_id["blob"]["meta"]["removal_capped"]["reason"].startswith("low-confidence")
    assert by_id["product"]["meta"].get("keep_in_background") is not True

    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L")) > 0
    # The capped blob's footprint (top band) is NOT inpainted…
    assert removal[5:45, 5:105].mean() < 0.02
    # …while the genuine product cutout (bottom band) IS removed from the plate.
    assert removal[55:95, 5:105].mean() > 0.5


def test_high_confidence_giant_raster_is_not_capped(tmp_path):
    """The cap is confidence-gated: a big generic raster the detector is SURE about is
    removed normally (missing/high confidence must never trigger the cap)."""
    source = tmp_path / "source.png"
    _blob_source(source)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (100, 40), 255).save(masks / "sure.png")
    candidates = [
        {"id": "sure", "target": "image", "z": 0,
         "box": {"x": 5, "y": 30, "w": 100, "h": 40},
         "mask": {"src": "masks/sure.png"},
         "meta": {"role": "photo", "confidence": 0.95}},
    ]
    result = reconstruct.reconstruct(str(source), {"lines": []}, candidates,
                                     str(tmp_path), {"inpaint": {"mode": "opencv"}})
    by_id = {c["id"]: c for c in result["candidates"]}
    assert result["stats"]["removal_capped"] == 0
    assert by_id["sure"]["meta"].get("keep_in_background") is not True


# ── crowded / edge glyph closer (002 ESSENTIALS, 025, 131 class) ──────────────────────


def _crowded_edge_source(path, size=(180, 80)):
    """Dark display ink crammed against the top edge — ring sampling is hostile."""
    image = Image.new("RGB", size, PLATE)
    draw = ImageDraw.Draw(image)
    # Top-edge headline bar (131 / 002 class): ink touches y=0 so a tight ring is tiny.
    draw.rectangle((8, 0, 170, 22), fill=INK)
    # Crowded sibling ink just below — contaminates a naive ring median with black.
    draw.rectangle((8, 26, 170, 44), fill=INK)
    image.save(path)
    return image


def _edge_headline_candidate(**overrides):
    candidate = {
        "id": "c_ESSENTIALS", "target": "text", "text": "ALLE ESSENTIALS", "z": 4,
        "box": {"x": 8, "y": 0, "w": 162, "h": 22},
        "visible_box": {"x": 8, "y": 0, "w": 162, "h": 22},
        "style": {"fontSize": 20, "fontFamily": "Arial", "color": "#141414"},
        "meta": {"source": "ocr", "role": "headline", "line_ids": ["L0"]},
    }
    candidate.update(overrides)
    return candidate


def test_crowded_edge_headline_solid_fills_with_plate_not_ink(tmp_path, monkeypatch):
    """002/131-class: edge + crowded ink must solid-fill with PLATE colour, not black.

    A naive ring median samples neighbouring glyph ink and "repairs" by repainting the
    ghost — QA would then mark resolved while the double-render remains. The closer must
    reject ink-like fills and sample farther / outside the box for plate colour.
    """
    source = tmp_path / "source.png"
    _crowded_edge_source(source)

    def fake_once(image_path, mask, output_path, cfg=None):
        # Leave the source glyphs fully in the plate (worst-case inpaint miss).
        Image.open(image_path).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)
    monkeypatch.setattr(
        reconstruct.inpaint, "inpaint_role_aware",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reinpaint on flat")),
    )

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_edge_headline_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"},
         "scene": {"archetype": "product_on_flat"}},
    )
    residual = result["stats"]["text_residual"]
    assert residual["flagged"], residual
    assert residual["flagged"][0]["resolved"] is True
    assert residual["flagged"][0].get("resolved_by") == "solid-plate-fill"
    assert "c_ESSENTIALS" in (residual.get("solid_filled_ids") or [])

    plate = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    # Top headline band must be plate-coloured, not leftover black ink.
    band = plate[0:22, 8:170]
    assert float(band.mean()) > 180.0, f"ghost ink remains: mean={band.mean():.1f}"
    assert float(np.abs(band.astype(np.int16) - np.array(INK)).mean()) > 80.0


def test_inky_ring_cannot_greenwash_residue_as_resolved(tmp_path, monkeypatch):
    """When every ring sample is glyph-coloured, closer must leave unresolved + hard_fail.

    Harness/QA greenwash path: mark resolved after painting black into the hole.
    """
    source = tmp_path / "source.png"
    # Full-frame ink: no plate-coloured exterior exists at any ring radius.
    Image.new("RGB", (60, 40), INK).save(source)

    def fake_once(image_path, mask, output_path, cfg=None):
        Image.open(image_path).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)

    cand = _text_candidate(
        id="c_INK", box={"x": 10, "y": 10, "w": 40, "h": 16},
        visible_box={"x": 10, "y": 10, "w": 40, "h": 16},
    )
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [cand], str(tmp_path),
        {"inpaint": {"mode": "opencv"},
         "scene": {"archetype": "product_on_flat"},
         "reconstruct": {"text_residual": {
             "solid_fill_ring": 4, "solid_fill_dilate": 1, "min_px": 8, "min_ratio": 0.1,
         }}},
    )
    residual = result["stats"]["text_residual"]
    flagged = residual.get("flagged") or []
    assert flagged, residual
    assert flagged[0]["resolved"] is False
    assert flagged[0].get("hard_fail") is True
    assert "c_INK" not in (residual.get("solid_filled_ids") or [])


def test_unresolved_glyph_residue_hard_fails_qa_and_contract(tmp_path):
    """pixel_diff must hard-fail + contract_pass=false; harness cannot greenwash."""
    from src import pixel_diff
    from src.schema import dump as schema_dump

    source = tmp_path / "source.png"
    render = tmp_path / "preview.png"
    Image.new("RGB", (40, 30), "white").save(source)
    Image.new("RGB", (40, 30), "white").save(render)
    schema_dump({
        "stats": {"text_residual": {
            "enabled": True, "checked": 1, "reinpainted": False,
            "flagged": [{"id": "c_ESSENTIALS", "residual_px": 400,
                         "resolved": False, "hard_fail": True}],
        }},
    }, str(tmp_path / "reconstruction.json"))

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0.5}},
    )
    rules = {f.get("rule") for f in (result.get("hard_fails") or [])}
    assert "glyph-residue" in rules
    assert result["structural"].get("glyph_residue_unresolved") == 1
    assert result["contract"]["glyph_residue_clean"] is False
    assert result["contract_pass"] is False

    # Greenwash attempt: resolved=True but hard_fail still set → still fails.
    schema_dump({
        "stats": {"text_residual": {
            "enabled": True, "checked": 1,
            "flagged": [{"id": "c_ESSENTIALS", "residual_px": 400,
                         "resolved": True, "hard_fail": True}],
        }},
    }, str(tmp_path / "reconstruction.json"))
    washed = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0.5}},
    )
    assert "glyph-residue" in {f.get("rule") for f in (washed.get("hard_fails") or [])}
    assert washed["contract_pass"] is False


def test_display_edge_text_coverage_is_dilated(tmp_path):
    """_ensure_text_removal_coverage dilates display/edge headlines past sparse ink."""
    rgb = np.asarray(_crowded_edge_source(tmp_path / "src.png").convert("RGB"))
    # Sparse 1px ink strip — clears a tiny area floor but misses the stem fringe.
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[8:14, 20:160] = 255
    cand = _edge_headline_candidate()
    out = reconstruct._ensure_text_removal_coverage(
        cand, mask, rgb, {"reconstruct": {"force_text_removal_coverage": True}},
    )
    assert int(np.count_nonzero(out)) > int(np.count_nonzero(mask))
    assert (cand.get("meta") or {}).get("removal_coverage_dilated", {}).get("edge") is True
