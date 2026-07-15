"""CPU tests for the text-ghost / double-render and plate over-destruction fixes.

Covers the three mechanisms the benchmark forensics exposed:

  * smear-aware residue detection — a re-inpaint that *smears* leftover glyph ink
    (067 "WE'RE SAYING GOODBYE" stayed red, 021 sticky-note copy) shifts the pixel
    value out of source-match range but is visibly still inky; the old source-match
    metric read that as "resolved" and shipped a double render;
  * iterate-then-honestly-fail — the audit now runs several re-inpaint passes and, if
    residue survives, hands the layer to the raster-slice floor via force_raster_ids
    instead of stamping native text over the ghost;
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


def test_smeared_ghost_is_flagged_and_forced_to_raster(tmp_path, monkeypatch):
    """A smear that the OLD source-match metric would miss must (a) be flagged, and
    (b) after every re-inpaint pass fails, be handed to the raster-slice floor."""
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
        {"inpaint": {"mode": "opencv"}},
    )

    residual = result["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] and residual["flagged"][0]["id"] == "c_B0"
    # Ink-colour proximity caught it even though it never matches source.
    assert residual["flagged"][0]["resolved"] is False
    assert residual["flagged"][0]["hard_fail"] is True
    # It iterated (default 3 passes) instead of giving up after one.
    assert residual["passes"] == 3
    assert calls["repair"] == 3
    # …and the unresolved editable text layer is queued for the raster-slice floor.
    assert residual["force_raster_ids"] == ["c_B0"]


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
        {"inpaint": {"mode": "opencv"}},
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
         "reconstruct": {"text_residual": {"reinpaint": False}}},
    )
    residual = result["stats"]["text_residual"]
    assert residual["flagged"] and residual["flagged"][0]["resolved"] is False
    assert residual["reinpainted"] is False
    assert residual["passes"] == 0


# ── end-to-end: force_raster_ids drives the raster-slice floor ────────────────────────


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


def test_audit_force_raster_ids_are_sliced_and_marked_resolved(tmp_path):
    """The unresolved-ghost ids the audit persists must force a slice even when the
    per-layer QA gate would otherwise pass, and the slice marks the flag resolved."""
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

    # Fully permissive per-layer gate: nothing fails on its own merits.
    permissive = {"region_ssim_min": 0.0, "region_color_min": 0.0,
                  "text_ink_iou_min": 0.0, "text_ink_excess_max": 1e9}
    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"}, "fallback": dict(permissive)},
    )

    assert [entry["id"] for entry in report["slices"]] == ["c_B0"]
    assert report.get("residue_resolved_by_slice") == ["c_B0"]
    # The persisted audit flag is now resolved (the slice IS the resolution), so QA's
    # glyph-residue gate and repair's rebuild-clean-plate won't re-fire on it.
    flag = load(recon_path)["stats"]["text_residual"]["flagged"][0]
    assert flag["resolved"] is True
    assert flag["resolved_by"] == "raster-slice"


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
