"""CPU tests for the Codia-style confidence-gated raster-slice fallback (Task 1)
and the ghost/duplicate text leak-through fixes (Task 2).

Covers: per-layer region scoring, slice creation for a synthetically corrupted
region, slice/ownership/removal-mask consistency (no double rendering), text
removal-mask coverage enforcement, the post-inpaint residual audit, and the
repair.assess wiring the harness uses to trigger fallback as a repair action.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import build_design_json, pixel_diff, reconstruct, render_preview, repair  # noqa: E402
from src.schema import (  # noqa: E402
    RASTER_SLICE_FALLBACK_DEFAULTS,
    dump,
    load,
    raster_slice_failures,
)

PLATE = (238, 232, 220)


def _source(path, size=(200, 140)):
    image = Image.new("RGB", size, PLATE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 48, 135, 65), fill=(20, 20, 20))  # glyph-like ink bar
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


# ── per-layer region scoring (pixel_diff) ───────────────────────────────────────────


def test_score_layer_regions_flags_only_the_corrupted_region(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (220, 150), PLATE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 99, 59), fill=(30, 80, 180))    # button
    draw.ellipse((140, 30, 199, 89), fill=(210, 45, 30))    # dot
    image.save(source)
    wrong = Image.new("RGB", (220, 150), PLATE)
    draw = ImageDraw.Draw(wrong)
    draw.rectangle((20, 20, 99, 59), fill=(240, 240, 240))  # button lost its fill
    draw.ellipse((140, 30, 199, 89), fill=(210, 45, 30))    # dot still correct
    wrong.save(render)
    design = {
        "canvas": {"w": 220, "h": 150},
        "layers": [
            {"id": "btn", "type": "shape", "box": {"x": 20, "y": 20, "w": 80, "h": 40},
             "meta": {"role": "button"}},
            {"id": "dot", "type": "image", "box": {"x": 140, "y": 30, "w": 60, "h": 60},
             "meta": {"role": "icon"}},
        ],
    }

    rows = {row["id"]: row for row in
            pixel_diff.score_layer_regions(str(source), str(render), design, str(tmp_path))}

    assert rows["btn"]["region_ssim"] < rows["dot"]["region_ssim"]
    assert raster_slice_failures(rows["btn"])
    assert not raster_slice_failures(rows["dot"])


def test_text_ink_iou_separates_ghost_double_text_from_clean_render(tmp_path):
    source = tmp_path / "source.png"
    ghost = tmp_path / "ghost.png"
    clean = tmp_path / "clean.png"
    base = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(base).text((30, 30), "05:00 PM - 121K", fill=(60, 60, 60))
    base.save(source)
    base.save(clean)  # identical render: text drawn once, right place
    doubled = base.copy()
    # Ghost: original not removed + re-render on top, offset a few px (009 failure).
    ImageDraw.Draw(doubled).text((34, 36), "05:00 PM - 121K", fill=(60, 60, 60))
    doubled.save(ghost)
    design = {
        "canvas": {"w": 200, "h": 80},
        "layers": [{"id": "ts", "type": "text", "text": "05:00 PM - 121K",
                    "box": {"x": 28, "y": 28, "w": 110, "h": 16}, "meta": {"role": "caption"}}],
    }

    ghost_row = pixel_diff.score_layer_regions(str(source), str(ghost), design, str(tmp_path))[0]
    clean_row = pixel_diff.score_layer_regions(str(source), str(clean), design, str(tmp_path))[0]

    assert clean_row["ink_iou"] > ghost_row["ink_iou"]
    assert ghost_row["ink_excess"] > clean_row["ink_excess"]
    # Default policy: text never fails the slice gate ("wrong Inter beats baked
    # pixels"); the ink separation stays measurable behind the forensic flag.
    assert raster_slice_failures(ghost_row) == []
    forensic = {"text_slice_gate_enabled": True}
    assert raster_slice_failures(ghost_row, forensic)
    assert not raster_slice_failures(clean_row, forensic)


def test_compare_merges_region_scores_into_per_layer(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (128, 96), PLATE)
    ImageDraw.Draw(image).rectangle((10, 10, 69, 49), fill=(30, 80, 180))
    image.save(source)
    broken = Image.new("RGB", (128, 96), PLATE)
    broken.save(render)  # the shape simply did not render
    design = {
        "canvas": {"w": 128, "h": 96},
        "layers": [{"id": "panel", "type": "shape", "box": {"x": 10, "y": 10, "w": 60, "h": 40},
                    "meta": {"role": "panel"}}],
        "meta": {"editable_ratio": 1.0},
    }

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)

    rows = {row["id"]: row for row in result["per_layer"]}
    assert "panel" in rows
    assert rows["panel"]["region_ssim"] < RASTER_SLICE_FALLBACK_DEFAULTS["region_ssim_min"]
    assert result["structural"]["raster_slices"] == {"count": 0, "ids": []}


# ── slice application end-to-end (reconstruct.apply_raster_slice_fallback) ──────────


def _build_failed_text_run(tmp_path, sabotage_color="#eee8dc"):
    """Reconstruct a text ad, then compile+render it so the text is invisible.

    Painting the editable text in the plate colour reproduces the 'render does not
    match the source region' failure deterministically on CPU.
    """
    source = tmp_path / "source.png"
    _source(source)
    candidate = _text_candidate()
    candidate["style"]["color"] = sabotage_color
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [candidate], str(tmp_path),
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
    return source, result


def test_fallback_replaces_failed_text_layer_with_pixel_exact_slice(tmp_path):
    source, _ = _build_failed_text_run(tmp_path)

    # Default policy first: text is NEVER sliced without the forensic flag.
    default_report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source), {"inpaint": {"mode": "opencv"}},
    )
    assert default_report["slices"] == []

    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"},
         "fallback": {"text_slice_gate_enabled": True}},
    )

    assert [entry["id"] for entry in report["slices"]] == ["c_B0"]
    assert report["slices"][0]["covered_by_removal_mask"] is True
    assert os.path.exists(tmp_path / "fallback.json")

    design = load(os.path.join(str(tmp_path), "design.json"))
    layers = {layer["id"]: layer for layer in design["layers"]}
    slice_layer = layers["c_B0"]
    assert slice_layer["type"] == "image"
    assert slice_layer["meta"]["fallback"] == "raster-slice"
    # The failed editable attempt is preserved for future repair.
    assert slice_layer["meta"]["fallback_editable"]["text"] == "SALE"
    assert slice_layer["meta"]["fallback_editable"]["kind"] == "text"
    assert slice_layer["meta"]["source_text"] == "SALE"
    assert os.path.exists(tmp_path / slice_layer["src"])

    # Honest accounting: slice reduces editability but is an explained raster.
    accounting = design["meta"]["leaf_accounting"]
    assert accounting["raster_slice_count"] == 1
    assert accounting["unexplained_raster_count"] == 0
    assert design["meta"]["editable_ratio"] < 1.0

    # The re-rendered preview now shows the source ink at the failed region.
    preview = np.asarray(Image.open(tmp_path / "preview.png").convert("RGB"))
    assert preview[55, 80, 0] < 100  # dark glyph bar restored by the slice

    # reconstruction.json candidate carries the same contract (lineage survives).
    recon = load(os.path.join(str(tmp_path), "reconstruction.json"))
    cand = next(c for c in recon["candidates"] if c["id"] == "c_B0")
    assert cand["target"] == "image"
    assert cand["meta"]["fallback"] == "raster-slice"


def test_slice_alpha_is_exactly_ledger_pixels_no_double_rendering(tmp_path):
    source, _ = _build_failed_text_run(tmp_path)

    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"},
         "fallback": {"text_slice_gate_enabled": True}},
    )

    entry = report["slices"][0]
    slice_rgba = np.asarray(Image.open(tmp_path / entry["src"]).convert("RGBA"))
    removal = np.asarray(Image.open(tmp_path / "removal_mask.png").convert("L")) > 0
    box = entry["box"]
    y0, x0 = int(box["y"]), int(box["x"])
    alpha = slice_rgba[:, :, 3] > 0
    canvas_alpha = np.zeros(removal.shape, dtype=bool)
    canvas_alpha[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] = alpha
    # Every slice pixel was inpainted out of the plate: covered by the removal mask,
    # so the pixels do NOT live in the background AND the slice (no double render).
    assert not np.any(canvas_alpha & ~removal)
    # And the slice restores real source pixels at the glyph positions.
    src_rgb = np.asarray(Image.open(source).convert("RGB"))
    ys, xs = np.nonzero(canvas_alpha)
    local = slice_rgba[ys - y0, xs - x0, :3]
    assert np.array_equal(local, src_rgb[ys, xs])


def test_fallback_is_idempotent_and_skips_existing_slices(tmp_path):
    source, _ = _build_failed_text_run(tmp_path)
    forensic = {"inpaint": {"mode": "opencv"},
                "fallback": {"text_slice_gate_enabled": True}}
    first = reconstruct.apply_raster_slice_fallback(str(tmp_path), str(source), forensic)
    assert first["slices"]

    second = reconstruct.apply_raster_slice_fallback(str(tmp_path), str(source), forensic)

    assert second["slices"] == []
    assert second["dropped"] == []


def test_fallback_drops_failed_layer_whose_pixels_were_never_removed(tmp_path):
    source = tmp_path / "inset.png"
    Image.new("RGB", (120, 90), (40, 110, 150)).save(source)
    Image.new("L", (30, 30), 255).save(tmp_path / "inset-mask.png")
    candidate = {
        "id": "inset", "target": "image", "box": {"x": 78, "y": 8, "w": 30, "h": 30},
        "mask": {"src": "inset-mask.png"},
        "meta": {"role": "photo", "keep_underlay": True},
    }
    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})
    tree = [c for c in result["candidates"] if c.get("target") != "drop"]
    dump(tree, os.path.join(str(tmp_path), "layout.json"))
    build_design_json.build(
        tree, {"w": 120, "h": 90}, str(tmp_path),
        base_src=os.path.join(str(tmp_path), result["background"]), doc_id="t", name="t",
    )
    # Corrupt the staged asset so the rendered region disagrees with the source.
    design = load(os.path.join(str(tmp_path), "design.json"))
    inset = next(layer for layer in design["layers"] if layer["id"] == "inset")
    Image.new("RGB", (30, 30), (255, 0, 0)).save(tmp_path / inset["src"])
    render_preview.render(os.path.join(str(tmp_path), "design.json"), str(tmp_path))

    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source), {"inpaint": {"mode": "opencv"}},
    )

    assert [entry["id"] for entry in report["dropped"]] == ["inset"]
    assert report["slices"] == []
    design = load(os.path.join(str(tmp_path), "design.json"))
    assert all(layer["id"] != "inset" for layer in design["layers"])
    # The plate (which keep_underlay preserved) shows the original pixels again.
    preview = np.asarray(Image.open(tmp_path / "preview.png").convert("RGB"))
    assert tuple(preview[20, 90]) == (40, 110, 150)


def test_fallback_never_slices_a_near_canvas_region(tmp_path):
    source, _ = _build_failed_text_run(tmp_path)
    # Shrink the allowed slice size below the text region: gate must refuse.
    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"},
         "fallback": {"max_layer_canvas_fraction": 0.02,
                      "text_slice_gate_enabled": True}},
    )
    assert report["slices"] == []
    assert any(item["reason"] == "region-too-large-for-slice" for item in report["skipped"])


def test_fallback_disabled_by_config_is_a_noop(tmp_path):
    source, _ = _build_failed_text_run(tmp_path)
    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source), {"fallback": {"enabled": False}},
    )
    assert report["enabled"] is False
    design = load(os.path.join(str(tmp_path), "design.json"))
    assert any(layer["type"] == "text" for layer in design["layers"])


def test_forced_slice_ids_from_repair_patch_are_honored(tmp_path):
    # With fully permissive thresholds nothing fails on its own, but the harness
    # repair patch (reconstruct.focus_regions with layer_id) must still force the slice.
    source, _ = _build_failed_text_run(tmp_path)
    permissive = {"region_ssim_min": 0.0, "region_color_min": 0.0,
                  "text_ink_iou_min": 0.0, "text_ink_excess_max": 1e9}

    unforced = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"}, "fallback": dict(permissive)},
    )
    assert unforced["slices"] == []  # permissive gate really passes everything

    report = reconstruct.apply_raster_slice_fallback(
        str(tmp_path), str(source),
        {"inpaint": {"mode": "opencv"}, "fallback": dict(permissive),
         "reconstruct": {"focus_regions": [{"layer_id": "c_B0", "region_ssim": 0.2}]}},
    )

    assert [entry["id"] for entry in report["slices"]] == ["c_B0"]
    assert "forced by repair (reconstruct.focus_regions)" in report["slices"][0]["reasons"]


# ── Task 2: ghost/duplicate text leak-through ────────────────────────────────────────


def test_emitted_text_with_no_ink_contrast_still_gets_removal_coverage(tmp_path):
    source = tmp_path / "flat.png"
    Image.new("RGB", (180, 120), PLATE).save(source)  # no contrast under the text box
    candidate = _text_candidate(box={"x": 10, "y": 10, "w": 45, "h": 18},
                                visible_box={"x": 10, "y": 10, "w": 45, "h": 18})

    result = reconstruct.reconstruct(str(source), {"lines": []}, [candidate], str(tmp_path),
                                     {"inpaint": {"mode": "opencv"}})

    emitted = next(c for c in result["candidates"] if c["id"] == "c_B0")
    assert emitted["target"] == "text"
    assert emitted["meta"]["removal_coverage_forced"]["mask_px_after"] >= 12
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert removal[12:26, 12:53].any()


def test_text_removal_dilate_floor_covers_antialiased_fringe(tmp_path):
    source = tmp_path / "source.png"
    _source(source)  # ink bar x=45..135, y=48..65

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},  # no explicit dilate config anywhere
    )

    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    # Default text floor is 4px (config default reconstruct.mask_dilate=2 is too
    # small for AA fringes): mask must extend beyond the ink bar's right edge.
    assert removal[55, 138] > 0
    assert removal[55, 145] == 0  # but not smear far into the plate


def test_post_inpaint_residual_audit_flags_and_reinpaints_ghost_text(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _source(source)
    calls = {"once": 0, "repair": 0}

    def fake_once(image_path, mask, output_path, cfg=None):
        calls["once"] += 1
        Image.open(image_path).save(output_path)  # "inpaint" that leaves every glyph
        return {"ok": True, "path": output_path, "backend": "fake-noop",
                "backend_class": "active"}

    def fake_role_aware(image_path, masks, output_path, cfg=None):
        calls["repair"] += 1
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        img[np.asarray(masks["text"]) > 0] = PLATE
        Image.fromarray(img).save(output_path)
        return {"ok": True, "backend": "fake-plate"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)
    monkeypatch.setattr(reconstruct.inpaint, "inpaint_role_aware", fake_role_aware)

    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv"}},
    )

    residual = result["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] and residual["flagged"][0]["id"] == "c_B0"
    assert residual["reinpainted"] is True
    assert residual["flagged"][0]["resolved"] is True
    assert calls == {"once": 1, "repair": 1}
    # The expanded region joined the persisted removal mask (single-union contract).
    removal = np.asarray(Image.open(tmp_path / result["removal_mask"]).convert("L"))
    assert int(np.count_nonzero(removal)) >= residual["expanded_px"] > 0
    # And the plate is actually clean now.
    clean = np.asarray(Image.open(tmp_path / result["background"]).convert("RGB"))
    assert clean[55, 80, 0] > 150


def test_clean_inpaint_produces_no_residual_flags(tmp_path):
    source = tmp_path / "source.png"
    _source(source)
    result = reconstruct.reconstruct(
        str(source), {"lines": []}, [_text_candidate()], str(tmp_path),
        {"inpaint": {"mode": "opencv", "opencv_radius": 4}},
    )
    residual = result["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] == []


# ── repair.assess wiring (harness trigger) ──────────────────────────────────────────


def test_assess_emits_actionable_raster_slice_repair_for_failing_regions():
    from src import harness

    qa = {
        "per_layer": [
            {"id": "headline", "type": "text", "role": "headline",
             "region_ssim": 0.30, "ink_iou": 0.10,
             "abs_box": {"x": 10, "y": 10, "w": 80, "h": 20}},
            {"id": "photo", "type": "image", "role": "photo", "region_ssim": 0.95},
        ],
    }
    # Default policy: a failing TEXT region no longer produces a slice repair.
    assert not any((item.get("params") or {}).get("raster_slice")
                   for item in repair.assess({}, qa, {"lines": []}, {}))
    repairs = repair.assess({}, qa, {"lines": []},
                            {"fallback": {"text_slice_gate_enabled": True}})

    slice_repairs = [item for item in repairs
                     if (item.get("params") or {}).get("raster_slice")]
    assert len(slice_repairs) == 1
    item = slice_repairs[0]
    assert (item["stage"], item["action"]) == ("reconstruct", "inspect-worst-regions")
    assert item["severity"] == "high"
    assert item["params"]["regions"][0]["layer_id"] == "headline"
    # The harness can act on it without any new ACTIONABLE registration…
    assert harness.is_actionable(item)
    choice = harness.recommended_resume([item])
    assert choice["resume"] == "reconstruct"
    # …and the config patch carries the failing regions to apply_raster_slice_fallback.
    assert choice["patches"]["reconstruct"]["focus_regions"][0]["layer_id"] == "headline"
    # Generic per-layer suggestions are superseded for the sliced region.
    assert not any(item.get("target_id") == "headline" and item["action"] == "review"
                   for item in repairs)


def test_assess_skips_layers_already_sliced_and_respects_disable():
    qa = {"per_layer": [{"id": "headline", "type": "text", "region_ssim": 0.2,
                         "ink_iou": 0.05, "fallback": "raster-slice"}]}
    assert not any((item.get("params") or {}).get("raster_slice")
                   for item in repair.assess({}, qa, {"lines": []}, {}))

    qa2 = {"per_layer": [{"id": "headline", "type": "text", "region_ssim": 0.2}]}
    disabled = repair.assess({}, qa2, {"lines": []}, {"fallback": {"enabled": False}})
    assert not any((item.get("params") or {}).get("raster_slice") for item in disabled)


def test_assess_turns_unresolved_text_residue_into_inpaint_repair(tmp_path):
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [],
        "stats": {"text_residual": {
            "enabled": True, "checked": 2, "reinpainted": False,
            "flagged": [{"id": "c_B0", "residual_px": 210, "resolved": False}],
        }},
    }), encoding="utf-8")

    repairs = repair.assess({}, {}, {"lines": []}, {"run_dir": str(tmp_path)})

    assert any(item["stage"] == "inpaint" and item["action"] == "rebuild-clean-plate"
               and "glyph residue" in item["reason"] for item in repairs)


# ── QA reporting stays honest ────────────────────────────────────────────────────────


def test_structural_audit_reports_slices_and_counts_slice_text_as_non_editable(tmp_path):
    # F4: a raster slice of a detected text line is reported AND counts as NON-editable.
    # Rasterizing failed overlay copy is a quality loss, never a way to score 1.0 on
    # editability while the text is pixels.
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _source(source)
    image.save(render)
    image.save(tmp_path / "background_clean.png")
    slice_asset = tmp_path / "slice.png"
    Image.new("RGBA", (91, 18), (20, 20, 20, 255)).save(slice_asset)
    design = {
        "canvas": {"w": 200, "h": 140},
        "layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "box": {"x": 0, "y": 0, "w": 200, "h": 140},
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "c_B0", "type": "image", "src": "slice.png",
             "box": {"x": 45, "y": 48, "w": 91, "h": 18},
             "meta": {"role": "headline", "fallback": "raster-slice",
                      "fallback_scores": {"region_ssim": 0.3},
                      "source_text": "SALE", "line_ids": ["L0"],
                      "layer_disposition": "foreground_raster"}},
        ],
        "kept_in_photo": [],
        "meta": {"editable_ratio": 0.5, "warnings": []},
    }
    ocr = {"lines": [{"id": "L0", "text": "SALE", "conf": 0.99,
                      "box": {"x": 45, "y": 48, "w": 91, "h": 18}}]}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=ocr, design=design)

    # The slice is still honestly reported...
    assert result["structural"]["raster_slices"] == {"count": 1, "ids": ["c_B0"]}
    # ...but now it LOWERS editable_text_recall instead of vanishing from the denominator,
    # and shows up as rasterized text (visible in benchmark.md, not hidden inside a 1.0).
    assert result["structural"]["editable_text_recall"] == 0.0
    assert result["structural"]["rasterized_text_count"] == 1
    assert result["structural"]["rasterized_text_ratio"] == 1.0
    assert result["rasterized_text_count"] == 1
    # The only text line ships as pixels, so the honest metric now trips the gate.
    assert "missing-editable-text" in {f["rule"] for f in result["hard_fails"]}


def test_editable_text_recall_mixes_native_slice_and_kept_in_photo(tmp_path):
    # F4 honest denominator: native TEXT counts as correct, a slice counts against, and
    # kept_in_photo scene text is excluded from the denominator (legitimately by-design).
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (200, 160), PLATE)
    image.save(source)
    image.save(render)
    image.save(tmp_path / "background_clean.png")
    Image.new("RGBA", (80, 16), (20, 20, 20, 255)).save(tmp_path / "slice.png")
    design = {
        "canvas": {"w": 200, "h": 160},
        "layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "box": {"x": 0, "y": 0, "w": 200, "h": 160},
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "headline", "type": "text", "text": "BIG SALE",
             "box": {"x": 10, "y": 10, "w": 90, "h": 20},
             "style": {"fontFamily": "Inter"}, "meta": {"role": "headline"}},
            {"id": "sub", "type": "image", "src": "slice.png",
             "box": {"x": 10, "y": 40, "w": 80, "h": 16},
             "meta": {"role": "subhead", "fallback": "raster-slice",
                      "source_text": "LIMITED TIME", "line_ids": ["L1"]}},
        ],
        "kept_in_photo": ["INGREDIENTS LIST"],
        "meta": {"editable_ratio": 0.5, "warnings": []},
    }
    ocr = {"lines": [
        {"id": "L0", "text": "BIG SALE", "conf": 0.99},       # native editable -> correct
        {"id": "L1", "text": "LIMITED TIME", "conf": 0.99},   # sliced -> rasterized
        {"id": "L2", "text": "INGREDIENTS LIST", "conf": 0.99},  # kept_in_photo -> excluded
    ]}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=ocr, design=design)
    structural = result["structural"]
    # Denominator excludes the baked scene-text line: 1 correct / (3 total - 1 kept) = 0.5.
    assert structural["editable_text_recall"] == 0.5
    assert structural["rasterized_text_count"] == 1
