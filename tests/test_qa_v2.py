"""CPU-only tests for local visual metrics and structural QA gates."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import pixel_diff, repair  # noqa: E402
from src.qa_config import DEFAULT_VISUAL_PASS_SSIM, pixel_diff_thresholds, visual_pass_ssim  # noqa: E402


def _scene(path, size=(128, 96)):
    image = Image.new("RGB", size, (242, 238, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 8, 62, 43), fill=(30, 80, 180))
    draw.ellipse((78, 18, 112, 52), fill=(210, 45, 30))
    draw.line((8, 75, 118, 75), fill=(20, 20, 20), width=3)
    image.save(path)
    return image


def _rules(result):
    return {item["rule"] for item in result.get("hard_fails", [])}


def test_identical_image_scores_perfect_across_all_metrics(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    image.save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert result["global_ssim"] == 1.0
    assert result["ssim"] == 1.0
    assert result["local_ssim"] == {"mean": 1.0, "p10": 1.0, "min": 1.0}
    assert result["edge_f1"] == 1.0
    assert result["color_similarity"] == 1.0
    assert result["delta_e_mean"] == 0.0
    assert result["visual_score"] == 1.0
    assert result["hard_fails"] == []
    assert result["per_layer"] == []
    assert os.path.exists(result["diff_png"])


def test_quality_flags_are_promoted_into_hard_fails(tmp_path):
    """SSIM is dominated by luminance/structure, so a render can score high multiscale
    SSIM while its edges or colors are badly wrong. quality_flags are computed for exactly
    this case — they must not be inert diagnostics; they must gate acceptance via hard_fails.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    changed = image.copy()
    ImageDraw.Draw(changed).rectangle((44, 69, 60, 80), fill=(0, 255, 0))
    changed.save(render)

    # Thresholds tightened just above what this modest corruption scores, so we know a
    # quality gate is crossed without depending on a knife's-edge fixture image.
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        thresholds={"local_ssim_min": 0.999, "edge_f1_min": 0.999, "color_similarity_min": 0.999},
    )

    assert result["quality_flags"], "expected the corrupted render to trip at least one quality gate"
    rules = _rules(result)
    assert {"local-ssim", "edge-fidelity", "color-fidelity"} & rules, (
        "quality_flags were computed but never merged into hard_fails"
    )
    # The structural sub-report must agree with the top-level hard_fails (single source of truth).
    assert result["structural"]["hard_fails"] == result["hard_fails"]


def test_local_corruption_is_penalized_more_than_global_ssim(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    changed = image.copy()
    # Local colour corruption that also erases a segment of a strong source edge.
    ImageDraw.Draw(changed).rectangle((44, 69, 60, 80), fill=(0, 255, 0))
    changed.save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert result["global_ssim"] > result["ssim"]
    assert result["local_ssim"]["min"] < result["local_ssim"]["mean"]
    assert result["edge_f1"] < 1.0
    assert result["color_similarity"] < 1.0
    assert result["per_region"]["worst"][0]["mean_delta"] > 0


def test_perfect_pixels_still_fail_missing_assets_fonts_and_ownership(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    image.save(render)
    image.save(tmp_path / "clean.png")
    shared = {"observations": [{"key": "sam3:S0", "source": "sam3", "id": "S0"}]}
    design = {
        "layers": [
            {"id": "background", "type": "image", "src": "clean.png",
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "missing", "type": "image", "src": None, "meta": {}},
            {"id": "text", "type": "text", "text": "SALE", "style": {},
             "meta": {"provenance": shared}},
            {"id": "shape", "type": "shape", "meta": {"provenance": shared}},
        ],
        "kept_in_photo": [],
        "meta": {"editable_ratio": 0.0, "warnings": []},
    }
    ocr = {"lines": [{"id": "L0", "text": "SALE", "conf": 0.99}]}

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path), source_ocr=ocr, design=design
    )

    assert result["visual_score"] == 1.0
    assert {"missing-assets", "missing-fonts", "low-editable-ratio", "duplicate-ownership"} <= _rules(result)
    assert result["structural"]["editable_text_recall"] == 1.0
    assert result["structural"]["duplicate_ownership"]


def test_untouched_background_is_detected_inside_removal_mask(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    removal = tmp_path / "removal_mask.png"
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((25, 30, 74, 47), fill="black")
    image.save(source)
    image.save(render)
    image.save(background)  # deliberately gameable old behavior
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((22, 27, 77, 50), fill=255)
    mask.save(removal)
    design = {
        "layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "headline", "type": "text", "text": "SALE",
             "style": {"fontFamily": "Inter"}, "meta": {}},
        ],
        "meta": {"editable_ratio": 0.5},
    }

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        source_ocr={"lines": [{"text": "SALE", "conf": 1}]}, design=design,
    )

    assert result["ssim"] == 1.0  # visual fidelity alone is no longer sufficient
    assert "background-leakage" in _rules(result)
    assert result["structural"]["background"]["exact_match_ratio"] == 1.0
    assert result["structural"]["background"]["changed_ratio"] == 0.0


def test_cleaned_background_passes_leakage_gate(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    removal = tmp_path / "removal_mask.png"
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((25, 30, 74, 47), fill="black")
    image.save(source)
    image.save(render)
    Image.new("RGB", image.size, "white").save(background)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((22, 27, 77, 50), fill=255)
    mask.save(removal)
    design = {
        "layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "shape", "type": "shape", "meta": {}},
        ],
        "meta": {"editable_ratio": 0.5},
    }
    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "background-leakage" not in _rules(result)
    assert result["structural"]["background"]["changed_ratio"] > 0.1


def test_inpaint_changes_outside_removal_mask_are_hard_failure(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    removal = tmp_path / "removal_mask.png"
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((40, 30, 59, 49), fill="black")
    image.save(source)
    image.save(render)
    cleaned = image.copy()
    ImageDraw.Draw(cleaned).rectangle((0, 0, 19, 19), fill="red")  # illegal exterior damage
    ImageDraw.Draw(cleaned).rectangle((40, 30, 59, 49), fill="white")
    cleaned.save(background)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((38, 28, 61, 51), fill=255)
    mask.save(removal)

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "meta": {"role": "background", "source": "inpaint"}},
            {"id": "object", "type": "shape", "box": {"x": 40, "y": 30, "w": 20, "h": 20}},
        ], "meta": {"editable_ratio": 0.5}},
    )

    assert "inpaint-outside-mask" in _rules(result)
    assert result["structural"]["background"]["outside_changed_ratio"] > 0.01


def test_product_alpha_internal_hole_is_detected_and_repairable(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (100, 80), "white")
    image.save(source)
    image.save(render)
    asset = Image.new("RGBA", (40, 40), (20, 80, 160, 255))
    ImageDraw.Draw(asset).rectangle((12, 12, 27, 27), fill=(0, 0, 0, 0))
    asset.save(tmp_path / "product.png")
    design = {"layers": [
        {"id": "product", "type": "image", "src": "product.png",
         "box": {"x": 20, "y": 20, "w": 40, "h": 40}, "meta": {"role": "product"}},
    ], "meta": {"editable_ratio": 0.0}}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)

    assert "layer-alpha-holes" in _rules(result)
    row = result["structural"]["layer_alpha"][0]
    assert row["internal_hole_count"] == 1
    repairs = repair.assess(design, result, {}, {"run_dir": str(tmp_path)})
    mask_repair = next(item for item in repairs if item.get("target_id") == "product")
    assert (mask_repair["stage"], mask_repair["action"]) == ("sam3", "rerun-detection")
    assert mask_repair["params"]["reject_internal_holes"] is True


def test_icon_alpha_holes_are_not_false_positive(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (80, 80), "white")
    image.save(source)
    image.save(render)
    icon = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    ImageDraw.Draw(icon).ellipse((8, 8, 23, 23), fill=(0, 0, 0, 0))
    icon.save(tmp_path / "ring.png")
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [{"id": "ring", "type": "image", "src": "ring.png",
                             "box": {"x": 10, "y": 10, "w": 32, "h": 32},
                             "meta": {"role": "icon"}}], "meta": {}},
    )
    assert "layer-alpha-holes" not in _rules(result)


def test_detected_elements_dropped_before_reconstruction_are_hard_failure(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (80, 60), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "elements.json").write_text(json.dumps([
        {"id": "E0", "role": "product"},
        {"id": "E1", "role": "badge"},
        {"id": "E2", "role": "icon"},
        {"id": "E3", "role": "shape"},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [{"id": "E0", "target": "image"}],
        "stats": {},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert "low-element-recall" in _rules(result)
    assert result["structural"]["element_recall"] == 0.25
    assert result["structural"]["element_survival"]["missing_ids"] == ["E1", "E2", "E3"]
    repairs = repair.assess({}, result, {}, {"run_dir": str(tmp_path)})
    assert any(item.get("action") == "rerun-detection" for item in repairs)


def test_element_survival_follows_canonical_ids_and_provenance(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (80, 60), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "fused_elements.json").write_text(json.dumps([
        {"id": "E000", "role": "logo"},
        {"id": "E004", "role": "badge"},
        {"id": "E013", "role": "icon"},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [
            # The reconstruction canonicalizes/prefixes ids.
            {"id": "c_E004", "target": "image"},
            # A renamed owner can still prove survival through explicit canonical provenance.
            {"id": "owner-1", "target": "shape", "meta": {"provenance": {"observations": [
                {"source": "fused_elements", "id": "E0"},
                # A residual id with the same spelling is a different namespace.
                {"source": "residual", "id": "E13"},
            ]}}},
        ],
        "stats": {},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    survival = result["structural"]["element_survival"]
    assert survival == {
        "proposed": 3, "kept": 2, "recall": 0.66667, "missing_ids": ["E013"]
    }
    assert "low-element-recall" in _rules(result)


def test_flattened_protected_elements_are_not_expected_as_standalone_layers(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (80, 60), "white").save(source)
    Image.new("RGB", (80, 60), "white").save(render)
    (tmp_path / "elements.json").write_text(json.dumps([
        {"id": "E0", "role": "product"},
        {"id": "E1", "role": "badge"},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [
            {"id": "c_E0", "target": "drop", "meta": {"keep_in_background": True}},
            {"id": "c_E1", "target": "drop", "meta": {"keep_in_background": True}},
        ],
        "stats": {"flattened_scene_artwork": 2},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    survival = result["structural"]["element_survival"]
    assert survival["recall"] is None
    assert survival["not_applicable"] is True
    assert survival["expected_standalone"] == 0
    assert "low-element-recall" not in _rules(result)


def _write_scene_baked_run(tmp_path, archetype, layers):
    """A run where merge said all source text is photographic and every OCR line is kept."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (128, 96), (200, 200, 200))
    image.save(source)
    image.save(render)
    (tmp_path / "merge_report.json").write_text(
        json.dumps({"photographic_scene_text": True}), encoding="utf-8")
    (tmp_path / "archetype.json").write_text(
        json.dumps({"archetype": archetype}), encoding="utf-8")
    source_ocr = {"lines": [{"id": "L0", "text": "SITE WIDE SALE", "conf": 0.99}]}
    design = {"layers": layers,
              "kept_in_photo": ["SITE WIDE SALE"],
              "meta": {"editable_ratio": 0.0,
                       "leaf_accounting": {"foreground_leaf_count": 3,
                                           "native_leaf_count": 0,
                                           "native_leaf_ratio": 0.0}}}
    return pixel_diff.compare(str(source), str(render), str(tmp_path),
                              source_ocr=source_ocr, design=design)


def test_scene_baked_exemption_denied_for_screenshot_archetype(tmp_path):
    # 009: a social_screenshot that baked all its copy is a failure, not a contract-correct
    # single-photo answer — the exemption must not shield it from the editability floors.
    result = _write_scene_baked_run(
        tmp_path, "social_screenshot",
        layers=[{"id": "bg", "type": "image", "meta": {"role": "background"}}])
    assert "low-editable-ratio" in _rules(result)


def test_scene_baked_exemption_denied_when_design_has_empty_groups(tmp_path):
    # 021: three empty wrapper groups and no real leaves is junk, never a legitimate photo.
    result = _write_scene_baked_run(
        tmp_path, "caption_over_photo",
        layers=[{"id": "bg", "type": "image", "meta": {"role": "background"}},
                {"id": "g0", "type": "group", "children": []},
                {"id": "g1", "type": "group", "children": []}])
    assert "low-editable-ratio" in _rules(result)


def test_scene_baked_exemption_still_applies_to_a_genuine_photo(tmp_path):
    # A real photographic scene (caption_over_photo) whose text is printed in the image and
    # whose tree is not empty junk keeps the exemption: the editability floor is waived.
    result = _write_scene_baked_run(
        tmp_path, "caption_over_photo",
        layers=[{"id": "bg", "type": "image", "meta": {"role": "background"}},
                {"id": "photo", "type": "image", "meta": {"role": "photo"}}])
    assert "low-editable-ratio" not in _rules(result)


# ── F-vacuous: 0 editable text must never read as a perfect editability score (021) ─────


def test_all_source_text_baked_reads_undefined_not_a_vacuous_perfect_score(tmp_path):
    """021's mechanism: every readable source line is kept_in_photo, so the editable-text
    denominator is 0. The old convention returned 1.0 (a 0/0 pass) — "we baked the whole ad"
    scored identically to "we perfectly decompiled everything". The honest value is undefined:
    editable_text_recall / native_text_ratio / true_text_coverage are None, never 1.0."""
    result = _write_scene_baked_run(
        tmp_path, "caption_over_photo",
        layers=[{"id": "bg", "type": "image", "meta": {"role": "background"}},
                {"id": "photo", "type": "image", "meta": {"role": "photo"}}])
    assert result["editable_text_recall"] is None
    assert result["native_text_ratio"] is None
    assert result["true_text_coverage"] is None
    # ...and the distinct, machine-readable "nothing to decompile" status is set, so a reader
    # can tell it apart from an actual perfect decompile.
    assert result["nothing_to_decompile"] is True
    assert result["all_source_text_baked"] is True
    # A genuine photographic scene bake is correct-by-design: it must NOT hard-fail.
    assert "no-editable-content" not in _rules(result)
    assert "missing-editable-text" not in _rules(result)


def test_all_baked_without_a_scene_verdict_hard_fails_no_editable_content(tmp_path):
    """The other side of the vacuous fix: when the denominator empties out but the bake is
    NOT a legitimate photographic scene (junk wrapper groups, denied exemption), the run
    rasterized every line of copy and delivered zero editable text. Because recall is now
    undefined (not a false 1.0), the recall/coverage gates skip it — so an explicit
    no-editable-content hard-fail must catch it instead. A photocopy is never CLEAN."""
    result = _write_scene_baked_run(
        tmp_path, "caption_over_photo",
        layers=[{"id": "bg", "type": "image", "meta": {"role": "background"}},
                {"id": "g0", "type": "group", "children": []},
                {"id": "g1", "type": "group", "children": []}])
    assert result["editable_text_recall"] is None
    assert result["nothing_to_decompile"] is False
    assert "no-editable-content" in _rules(result)


def test_editable_text_fraction_is_ink_weighted_not_line_counted(tmp_path):
    """editable_text_fraction answers "how much of THIS ad can I edit?" over ALL source text
    (kept_in_photo INCLUDED in the denominator), weighted by measured text ink. One big
    editable headline plus many tiny baked package lines is highly batch-editable even though
    only a fraction of the *lines* are editable — so the ink-dominant editable line must lift
    the fraction well above its 1-of-N line share (013: 0.20 by count, 0.63 by ink)."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    # Ink is a MINORITY of each box (dark strokes on a light plate), exactly like real text:
    # the box's local median is the background, and the dark stroke pixels are the ink weight.
    # One ink-heavy editable headline (a wide, tall stroke band)...
    draw.rectangle((10, 25, 190, 45), fill=(10, 10, 10))
    # ...and four tiny baked package lines (a few stroke pixels each).
    for i in range(4):
        draw.rectangle((12, 102 + i * 12, 27, 105 + i * 12), fill=(10, 10, 10))
    image.save(source)
    image.save(render)
    source_ocr = {"lines": [
        {"id": "H", "text": "HUGE HEADLINE", "conf": 0.99, "box": {"x": 10, "y": 10, "w": 180, "h": 60}},
        {"id": "P0", "text": "tiny package one", "conf": 0.99, "box": {"x": 10, "y": 100, "w": 60, "h": 8}},
        {"id": "P1", "text": "tiny package two", "conf": 0.99, "box": {"x": 10, "y": 112, "w": 60, "h": 8}},
        {"id": "P2", "text": "tiny package three", "conf": 0.99, "box": {"x": 10, "y": 124, "w": 60, "h": 8}},
        {"id": "P3", "text": "tiny package four", "conf": 0.99, "box": {"x": 10, "y": 136, "w": 60, "h": 8}},
    ]}
    design = {
        "layers": [{"id": "h", "type": "text", "text": "HUGE HEADLINE",
                    "style": {"fontFamily": "Inter"}, "meta": {}}],
        "kept_in_photo": ["tiny package one", "tiny package two",
                          "tiny package three", "tiny package four"],
        "meta": {"editable_ratio": 1.0},
    }
    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=source_ocr, design=design)
    frac = result["editable_text_fraction"]
    # 1 of 5 lines is editable (line-count would be 0.20); the headline's ink dominates.
    assert frac is not None and frac > 0.5, f"expected ink-weighted >0.5, got {frac}"


def test_single_background_without_removal_work_can_legitimately_match_source(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    image = Image.new("RGB", (80, 60), (120, 140, 160))
    image.save(source)
    image.save(render)
    image.save(background)
    design = {
        "layers": [
            {"id": "background", "type": "image", "src": "background_clean.png",
             "meta": {"role": "background", "source": "inpaint"}},
        ],
        "meta": {"editable_ratio": 0.0},
    }
    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "background-leakage" not in _rules(result)
    assert result["structural"]["background"]["mask_supplied"] is False


def test_figma_import_report_becomes_a_qa_failure_when_compiler_rejects_layers(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    image.save(render)
    (tmp_path / "figma_report.json").write_text(__import__("json").dumps({
        "doc_id": "demo",
        "report": {
            "ok": False,
            "assets": {"missing": 1},
            "errors": [{"title": "Layer failed", "detail": "Missing image asset: assets/product.png"}],
        },
    }))
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"id": "demo", "layers": [], "meta": {}},
    )
    assert {"missing-assets", "figma-compiler-errors"} <= _rules(result)


def test_visual_pass_ssim_unifies_pixel_diff_repair_and_pipeline_gate():
    """qa.visual_pass_ssim is the single acceptance bar for pixel_diff hard-fails,
    repair ssim/visual_score suggestions, and run_pipeline qa.ok.
    """
    cfg = {"qa": {"visual_pass_ssim": 0.9}}
    assert visual_pass_ssim(cfg) == 0.9
    assert visual_pass_ssim({}) == DEFAULT_VISUAL_PASS_SSIM
    assert pixel_diff_thresholds(cfg)["local_ssim_min"] == 0.9
    assert pixel_diff.DEFAULT_THRESHOLDS["local_ssim_min"] == DEFAULT_VISUAL_PASS_SSIM

    archetype_cfg = {
        "qa": {"visual_pass_ssim": 0.9, "archetype_thresholds": {"visual_pass_ssim_min": 0.65}},
    }
    assert visual_pass_ssim(archetype_cfg) == 0.65
    assert pixel_diff_thresholds(archetype_cfg)["local_ssim_min"] == 0.65

    below = repair.assess({}, {"ssim": 0.88, "visual_score": 0.88}, {}, cfg)
    assert any("ssim 0.88 < 0.9" in item["reason"] for item in below)
    assert any("visual score 0.88 < 0.90" in item["reason"] for item in below)

    relaxed = {"qa": {"visual_pass_ssim": 0.85}}
    assert not any(
        item.get("action") == "retry" and "ssim" in item.get("reason", "")
        for item in repair.assess({}, {"ssim": 0.88, "visual_score": 0.88}, {}, relaxed)
    )


def test_per_layer_populated_from_reconstruction_stats_for_text_layers(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (100, 80), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "stats": {
            "per_layer": [
                {"id": "headline", "type": "text", "role": "headline", "ssim": 0.72, "recall": 0.65},
                {"id": "logo", "type": "icon", "score": 0.71},
            ],
        },
        "candidates": [
            {"id": "headline", "target": "text", "text": "SALE", "meta": {"role": "headline"}},
        ],
    }), encoding="utf-8")
    design = {
        "layers": [
            {"id": "headline", "type": "text", "text": "SALE",
             "style": {"fontFamily": "Inter"}, "meta": {"role": "headline"}},
        ],
        "meta": {"editable_ratio": 1.0},
    }

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)

    assert len(result["per_layer"]) == 1
    row = result["per_layer"][0]
    assert row["id"] == "headline"
    assert row["ssim"] == 0.72
    assert row["recall"] == 0.65
    assert row["score"] == 0.65


def test_per_layer_reads_candidate_meta_qa_for_design_text_layers(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (100, 80), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "stats": {"canonical_entities": 1},
        "candidates": [
            {
                "id": "cta",
                "target": "text",
                "text": "BUY NOW",
                "meta": {"role": "cta", "qa": {"ssim": 0.81, "recall": 0.9}},
            },
        ],
    }), encoding="utf-8")
    design = {
        "layers": [
            {"id": "cta", "type": "text", "text": "BUY NOW",
             "style": {"fontFamily": "Inter"}, "meta": {"role": "cta", "source_id": "cta"}},
        ],
        "meta": {"editable_ratio": 1.0},
    }

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)

    assert result["per_layer"] == [{
        "id": "cta",
        "type": "text",
        "role": "cta",
        "ssim": 0.81,
        "recall": 0.9,
        "score": 0.81,
    }]


def test_repair_consumes_per_layer_text_scores():
    repairs = repair.assess(
        {},
        {"per_layer": [{"id": "headline", "type": "text", "role": "headline", "score": 0.55}]},
        {"lines": []},
        {},
    )
    assert any(
        item["stage"] == "build"
        and item["action"] == "review"
        and item["target_id"] == "headline"
        for item in repairs
    )


def test_repair_consumes_nested_structural_failures_and_new_metrics():
    qa = {
        "ssim": 0.7,
        "visual_score": 0.6,
        "edge_f1": 0.4,
        "color_similarity": 0.5,
        "structural": {
            "editable_ratio": 0.0,
            "duplicate_ownership": ["sam3:S0 owned twice"],
            "hard_fails": [
                {"rule": "background-leakage", "detail": "foreground remains"},
                {"rule": "missing-assets", "detail": "asset x missing"},
                {"rule": "missing-fonts", "detail": "font x missing"},
            ],
        },
    }

    repairs = repair.assess({}, qa, {"lines": []}, {})
    actions = {item["action"] for item in repairs}

    assert {
        "rebuild-clean-plate",
        "restage-assets",
        "resolve-fonts",
        "restore-native-nodes",
        "enforce-single-owner",
        "inspect-worst-regions",
        "refit-geometry",
        "refit-colors-effects",
    } <= actions
    assert repairs[0]["severity"] == "high"


def test_strict_acceptance_requires_native_leaf_accounting(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0}},
        structural={"require_native_accounting": True},
    )
    assert "native-accounting-missing" in _rules(result)


def test_strict_acceptance_rejects_unexplained_raster_fallback(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    accounting = {
        "foreground_leaf_count": 1, "native_leaf_count": 0, "raster_leaf_count": 1,
        "intentional_raster_cluster_count": 0, "fallback_raster_count": 1,
        "unexplained_raster_count": 1, "unexplained_raster_ids": ["mystery"],
        "native_leaf_ratio": 0.0,
    }
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0, "leaf_accounting": accounting,
                                            "native_leaf_ratio": 0.0}},
        structural={"require_native_accounting": True},
    )
    assert "unexplained-raster-fallback" in _rules(result)
    assert result["structural"]["native_leaf_ratio"] == 0.0
    assert result["structural"]["leaf_accounting"] == accounting


def test_low_native_leaf_ratio_is_a_hard_fail_with_real_foreground(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    accounting = {
        "foreground_leaf_count": 4, "native_leaf_count": 0, "raster_leaf_count": 4,
        "intentional_raster_cluster_count": 0, "fallback_raster_count": 0,
        "unexplained_raster_count": 0, "unexplained_raster_ids": [],
        "native_leaf_ratio": 0.0,
    }
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0, "leaf_accounting": accounting,
                                            "native_leaf_ratio": 0.0}},
        structural={"require_native_accounting": True},
    )
    assert "low-native-leaf-ratio" in _rules(result)


def test_low_native_leaf_ratio_does_not_fire_with_single_foreground_leaf(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    accounting = {
        "foreground_leaf_count": 1, "native_leaf_count": 0, "raster_leaf_count": 1,
        "intentional_raster_cluster_count": 1, "fallback_raster_count": 0,
        "unexplained_raster_count": 0, "unexplained_raster_ids": [],
        "native_leaf_ratio": 0.0,
    }
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0, "leaf_accounting": accounting,
                                            "native_leaf_ratio": 0.0}},
        structural={"require_native_accounting": True},
    )
    assert "low-native-leaf-ratio" not in _rules(result)


def test_opencv_inpaint_fallback_is_a_hard_fail_under_acceptance(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    (tmp_path / "reconstruction.json").write_text(
        json.dumps({"stats": {"opencv_fallback_used": True}}), encoding="utf-8")
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0}},
        structural={"require_native_accounting": True},
    )
    assert "inpaint-degraded-opencv" in _rules(result)

    # Same reconstruction.json must not fail an ordinary diagnostic (non-acceptance) run.
    result2 = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        design={"layers": [], "meta": {"editable_ratio": 0}},
    )
    assert "inpaint-degraded-opencv" not in _rules(result2)


def test_pipeline_degradations_in_design_meta_are_a_hard_fail_under_acceptance(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    design = {
        "layers": [],
        "meta": {
            "editable_ratio": 0,
            "degradations": [{"stage": "vectorize", "reason": "gave up on complex path"}],
        },
    }
    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path), design=design,
        structural={"require_native_accounting": True},
    )
    assert "pipeline-degraded" in _rules(result)

    result2 = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "pipeline-degraded" not in _rules(result2)


# ── F2: anti-rasterization gates fire without a Figma acceptance run ─────────────────


def test_unexplained_raster_fallback_fails_ordinary_qa_without_require_native(tmp_path):
    # F2: these honesty gates used to be keyed to figma.require_export (false everywhere),
    # so 052 passed with an unexplained raster. They now evaluate whenever leaf accounting
    # exists — no acceptance opt-in required.
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (40, 30), "white").save(source)
    Image.new("RGB", (40, 30), "white").save(render)
    accounting = {
        "foreground_leaf_count": 1, "native_leaf_count": 0, "raster_leaf_count": 1,
        "intentional_raster_cluster_count": 0, "fallback_raster_count": 1,
        "unexplained_raster_count": 1, "unexplained_raster_ids": ["c_B0"],
        "native_leaf_ratio": 0.0,
    }
    design = {"layers": [], "meta": {"editable_ratio": 0, "leaf_accounting": accounting,
                                     "native_leaf_ratio": 0.0}}
    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "unexplained-raster-fallback" in _rules(result)


def test_low_native_leaf_ratio_fails_ordinary_qa_without_require_native(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (40, 30), "white").save(source)
    Image.new("RGB", (40, 30), "white").save(render)
    accounting = {
        "foreground_leaf_count": 5, "native_leaf_count": 1, "raster_leaf_count": 4,
        "intentional_raster_cluster_count": 0, "fallback_raster_count": 0,
        "unexplained_raster_count": 0, "unexplained_raster_ids": [],
        "native_leaf_ratio": 0.2,
    }
    design = {"layers": [], "meta": {"editable_ratio": 0.2, "leaf_accounting": accounting,
                                     "native_leaf_ratio": 0.2}}
    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "low-native-leaf-ratio" in _rules(result)

    # Config can still turn the gate off explicitly (sane default is ON).
    off = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design,
                             thresholds={"enforce_native_leaf_accounting": False})
    assert "low-native-leaf-ratio" not in _rules(off)


# ── F3: plate-destruction ceiling ───────────────────────────────────────────────────


def test_excessive_plate_destruction_is_a_hard_fail(tmp_path):
    # F3: the removal/inpaint erased most of the canvas (the 002 class). There is a gate for
    # an untouched plate and a no-op removal, but none for the opposite — a plate that was
    # almost entirely rebuilt, which means real content was erased.
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    background = tmp_path / "background_clean.png"
    removal = tmp_path / "removal_mask.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 94, 94), fill=(20, 40, 90))  # a big product panel
    image.save(source); image.save(render)
    # Clean plate erased the whole panel to gray -> ~80% of the canvas changed.
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
    assert "excessive-plate-destruction" in _rules(result)
    assert result["structural"]["background"]["changed_canvas_ratio"] > 0.55

    # A normal small removal does not trip it.
    small_mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(small_mask).rectangle((40, 40, 55, 55), fill=255)
    small_mask.save(removal)
    Image.new("RGB", (100, 100), "white").save(background)  # tiny change area
    result2 = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)
    assert "excessive-plate-destruction" not in _rules(result2)


# ── F8: archetype text_recall_min threads through and gates ──────────────────────────


def test_archetype_text_recall_min_threads_through_and_gates(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    source_ocr = {"lines": [{"id": "L0", "text": "HELLO WORLD", "conf": 0.99}]}
    render_ocr = {"lines": []}  # nothing recognized in the render -> text_recall 0.0

    # No threshold supplied: a bare compare keeps its old behaviour (no text-recall gate).
    base = pixel_diff.compare(str(source), str(render), str(tmp_path),
                              source_ocr=source_ocr, render_ocr=render_ocr)
    assert base["text_recall"] == 0.0
    assert "low-text-recall" not in _rules(base)

    # Archetype preset strictness threaded in by the caller now gates.
    gated = pixel_diff.compare(str(source), str(render), str(tmp_path),
                               source_ocr=source_ocr, render_ocr=render_ocr,
                               thresholds={"text_recall_min": 0.90})
    assert "low-text-recall" in _rules(gated)


# ── F15: unresolved glyph residue blocks a clean QA verdict ──────────────────────────


def test_unresolved_glyph_residue_is_a_hard_fail(tmp_path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (40, 30), "white")
    image.save(source); image.save(render)
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "stats": {"text_residual": {
            "enabled": True, "checked": 1, "reinpainted": False,
            "flagged": [{"id": "c_B1", "residual_px": 180, "resolved": False}],
        }},
    }), encoding="utf-8")
    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                design={"layers": [], "meta": {"editable_ratio": 0.5}})
    assert "glyph-residue" in _rules(result)

    # A resolved (reinpainted) residue does NOT fail.
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "stats": {"text_residual": {
            "enabled": True, "checked": 1, "reinpainted": True,
            "flagged": [{"id": "c_B1", "residual_px": 0, "resolved": True}],
        }},
    }), encoding="utf-8")
    ok = pixel_diff.compare(str(source), str(render), str(tmp_path),
                            design={"layers": [], "meta": {"editable_ratio": 0.5}})
    assert "glyph-residue" not in _rules(ok)


# ── element_recall must be readable at the qa.json top level, not only nested ───────


def test_element_recall_is_surfaced_at_the_top_level(tmp_path):
    """The honesty refactor added a top-level mirror of editable_text_recall (and
    rasterized_text_count/ratio) alongside their nested structural.* copies, but never
    extended that mirroring to element_recall/element_survival even though the nested
    computation (_element_survival_audit) was always correct. Any caller reading
    qa.get("element_recall") directly -- the first place a human or script would look --
    always saw null even on a run where structural.element_recall was a real number.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (80, 60), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "elements.json").write_text(json.dumps([
        {"id": "E0", "role": "product"},
        {"id": "E1", "role": "badge"},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [
            {"id": "E0", "target": "image"},
            {"id": "E1", "target": "image"},
        ],
        "stats": {},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert result["element_recall"] == 1.0
    assert result["element_recall"] == result["structural"]["element_recall"]
    assert result["element_survival"] == result["structural"]["element_survival"]
    assert result["element_survival"] == {
        "proposed": 2, "kept": 2, "recall": 1.0, "missing_ids": [],
    }


def test_element_recall_top_level_reflects_a_real_drop(tmp_path):
    """Same top-level mirror, but for the failing case: a dropped element must be visible
    at qa["element_recall"] (not just buried in qa["structural"]) and the low-element-recall
    hard fail must still fire.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = Image.new("RGB", (80, 60), "white")
    image.save(source)
    image.save(render)
    (tmp_path / "elements.json").write_text(json.dumps([
        {"id": "E0", "role": "product"},
        {"id": "E1", "role": "badge"},
        {"id": "E2", "role": "icon"},
        {"id": "E3", "role": "shape"},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [{"id": "E0", "target": "image"}],
        "stats": {},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert result["element_recall"] == 0.25
    assert "low-element-recall" in _rules(result)


# ── true_text_coverage: editable_text_recall must not lie via its own denominator ───


def test_true_text_coverage_catches_the_editable_text_recall_denominator_lie(tmp_path):
    """The denominator lie, expressed through the vector that can actually tell it.

    editable_text_recall drops every ``kept_in_photo`` line from its DENOMINATOR (scene text
    baked into a photo is legitimately not-editable-by-design). Claim every line as
    kept_in_photo and the denominator empties out: recall reads a perfect 1.0 while only one
    of six lines was actually delivered. true_text_coverage = text_recall *
    editable_text_recall must not be fooled, and missing-editable-text must fire on the
    combined signal.

    text_recall is not fooled because its own exclusion is evidence-based, not declared: a
    line only leaves its denominator when `_line_baked_in_asset` finds those pixels verbatim
    in the owning photo asset. Here nothing is baked anywhere, so all six lines stay in the
    denominator and the five we never delivered count against us.

    (This test previously staged the same "1 of 6" numbers by having render-OCR read back
    only one of six correctly-delivered lines. That premise was retired with the render-OCR
    round-trip itself: ad 131 ships 'BLACK FRIDAY SALE' character-perfect and doctr reads it
    back as 'BLACKERIDAYSALE', so "OCR could not re-read it" was never evidence that the text
    did not ship. See _text_recall_detail. The lie this test names is real; only the vector
    used to demonstrate it changed.)
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    lines = [
        "HEADLINE COPY", "SUPPORTING LINE ONE", "SUPPORTING LINE TWO",
        "SUPPORTING LINE THREE", "SUPPORTING LINE FOUR", "BUY NOW",
    ]
    source_ocr = {"lines": [
        {"id": f"L{i}", "text": text, "conf": 0.99} for i, text in enumerate(lines)
    ]}
    render_ocr = {"lines": [{"id": "R0", "text": "BUY NOW", "conf": 0.99}]}
    # Only "BUY NOW" is actually delivered; the other five are merely CLAIMED as baked
    # scene text, with no asset anywhere carrying their pixels.
    design = {
        "layers": [
            {"id": "cta", "type": "text", "text": "BUY NOW",
             "style": {"fontFamily": "Inter"}, "meta": {}},
        ],
        "kept_in_photo": lines[:5],
        "meta": {"editable_ratio": 1.0},
    }

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        source_ocr=source_ocr, render_ocr=render_ocr, design=design,
    )

    assert abs(result["text_recall"] - 1 / 6) < 1e-3
    assert result["editable_text_recall"] == 1.0
    assert abs(result["true_text_coverage"] - 1 / 6) < 1e-3
    assert "missing-editable-text" in _rules(result)
    detail = next(f["detail"] for f in result["hard_fails"] if f["rule"] == "missing-editable-text")
    assert "true text coverage" in detail


def test_true_text_coverage_does_not_pile_on_when_coverage_is_still_reasonable(tmp_path):
    """025-style case in reverse: editable_text_recall is ALREADY low enough to fail on
    its own (half the detected text was rasterized, not made editable), but OCR found
    all of the source text (text_recall 1.0), so the combined true_text_coverage (0.5)
    is still comfortably above the 0.20 floor. The gate must fire for the existing
    editable_text_recall reason only -- true_text_coverage must not also claim to be low
    when it isn't, and there must be exactly one missing-editable-text entry.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    source_ocr = {"lines": [
        {"id": "L0", "text": "HEADLINE COPY", "conf": 0.99},
        {"id": "L1", "text": "SUPPORTING LINE", "conf": 0.99},
    ]}
    render_ocr = {"lines": [
        {"id": "R0", "text": "HEADLINE COPY", "conf": 0.99},
        {"id": "R1", "text": "SUPPORTING LINE", "conf": 0.99},
    ]}
    design = {
        "layers": [
            {"id": "headline", "type": "text", "text": "HEADLINE COPY",
             "style": {"fontFamily": "Inter"}, "meta": {}},
            {"id": "supporting", "type": "image", "src": None,
             "meta": {"layer_disposition": "foreground_raster", "source_text": "SUPPORTING LINE"}},
        ],
        "meta": {"editable_ratio": 0.5},
    }

    result = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        source_ocr=source_ocr, render_ocr=render_ocr, design=design,
    )

    assert result["text_recall"] == 1.0
    assert result["editable_text_recall"] == 0.5
    assert result["true_text_coverage"] == 0.5

    fails = [f for f in result["hard_fails"] if f["rule"] == "missing-editable-text"]
    assert len(fails) == 1
    assert "editable text recall" in fails[0]["detail"]
    assert "true text coverage" not in fails[0]["detail"]


# ── local-ssim-worst-region: a catastrophic window must not hide under a good aggregate ──


def test_worst_local_window_hard_fails_even_when_the_aggregate_score_is_high(tmp_path):
    """009/016-style case: the multiscale/local-ssim aggregate is deliberately
    mean-dominated (0.72 mean + 0.26 p10 + 0.02 min per scale) so one badly corrupted
    window barely dents it when the other ~99% of windows are perfect. That let a
    genuinely catastrophic region (measured worst-window SSIM ~0.03-0.04 on 009/016)
    hide under an aggregate score that stays comfortably above the pass bar. The
    worst-region gate must fire independently of that aggregate, and report the
    offending window's bbox as evidence.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    size = (320, 320)
    base = Image.new("RGB", size, (140, 150, 160))
    draw = ImageDraw.Draw(base)
    draw.rectangle((20, 20, 120, 90), fill=(60, 90, 180))
    draw.ellipse((180, 200, 260, 270), fill=(200, 60, 40))
    base.save(source)

    corrupted = base.copy()
    cdraw = ImageDraw.Draw(corrupted)
    # Corrupt exactly one local window (the grid cell at x=256..288, y=0..32) with
    # high-frequency noise uncorrelated with the smooth source patch underneath it.
    # Everything else in the 320x320 canvas is untouched.
    bx, by, bw = 256, 0, 32
    for yy in range(by, by + bw, 2):
        for xx in range(bx, bx + bw, 2):
            fill = (0, 0, 0) if (xx + yy) % 4 == 0 else (255, 255, 255)
            cdraw.rectangle((xx, yy, xx + 1, yy + 1), fill=fill)
    corrupted.save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    # The aggregate stays high -- the ordinary local-ssim gate does not fire.
    assert result["ssim"] >= 0.9
    assert "local-ssim" not in _rules(result)
    # But the worst single window is a near-total collapse, and it must hard fail with
    # a locatable bbox even though the aggregate said everything was fine.
    assert result["local_ssim_worst_window"]["ssim"] < 0.10
    assert result["local_ssim_worst_window"]["bbox"] == {"x": 256, "y": 0, "w": 32, "h": 32}
    assert "local-ssim-worst-region" in _rules(result)
    detail = next(f["detail"] for f in result["hard_fails"] if f["rule"] == "local-ssim-worst-region")
    assert "x=256 y=0 w=32 h=32" in detail


# ── per-archetype color/edge floors: pixel_diff reads archetype.json itself ─────────


def test_archetype_edge_and_color_floors_are_read_from_archetype_json(tmp_path):
    """F-per-archetype-floor: most archetype presets don't define edge_f1_min/
    color_similarity_min at all (checked against src/archetype.py), so pixel_diff must
    not depend on every caller manually forwarding those keys the way text_recall_min is
    threaded today -- it reads the run's own archetype.json (written by every run) and
    applies the preset's floor itself, the same way editable_text_recall_min etc. are
    already archetype-aware. Caller-supplied ``thresholds=`` still wins when present.
    """
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    image = _scene(source)
    changed = image.copy()
    ImageDraw.Draw(changed).rectangle((44, 69, 60, 80), fill=(0, 255, 0))
    changed.save(render)
    (tmp_path / "archetype.json").write_text(json.dumps({
        "archetype": "comparison_grid",
        "preset": {"thresholds": {"edge_f1_min": 0.999, "color_similarity_min": 0.999}},
    }), encoding="utf-8")

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert {"edge-fidelity", "color-fidelity"} <= _rules(result)

    # An explicit caller threshold still overrides the archetype-file-derived floor.
    lenient = pixel_diff.compare(
        str(source), str(render), str(tmp_path),
        thresholds={"edge_f1_min": 0.0, "color_similarity_min": 0.0},
    )
    assert "edge-fidelity" not in _rules(lenient)
    assert "color-fidelity" not in _rules(lenient)


# ── word-split text nodes must not read as MISSING (false missing-editable-text) ──────


def test_editable_recall_regroups_word_split_nodes_from_one_source_block():
    """One OCR line emitted as several sibling word nodes is EDITABLE, not missing.

    002 ships "EUR63 -> EUR49" as c_B5__w0 ("EUR63 ->") + c_B5__w1 ("EUR49"); 066 ships
    "Smudges on upper lid" as c_B5__w0/w1/w2. Both are real, editable TEXT nodes. _norm
    strips separators, so the source line ("6349") can never be a substring of a
    space-joined per-node blob ("63 49") -- the line scored as MISSING and fired a false
    `missing-editable-text` (002: 0.75 vs a 0.84 bar) on work that shipped correctly.
    Fragments of the SAME block must rejoin before matching.
    """
    ocr = {"lines": [{"id": "L31", "text": "€63 → €49", "conf": 0.9},
                     {"id": "L10", "text": "Smudges on upper lid", "conf": 0.9}]}
    layers = [
        {"id": "c_B5__w0", "type": "text", "text": "€63 →"},
        {"id": "c_B5__w1", "type": "text", "text": "€49"},
        {"id": "c_B9__w0", "type": "text", "text": "Smudges"},
        {"id": "c_B9__w1", "type": "text", "text": "on"},
        {"id": "c_B9__w2", "type": "text", "text": "upper lid"},
    ]
    design = {"layers": layers, "kept_in_photo": []}

    result = pixel_diff._text_editability(ocr, design, layers)

    assert result["editable_text_correct"] == 2
    assert result["editable_text_recall"] == 1.0


def test_editable_recall_does_not_fuse_unrelated_blocks_into_phantom_matches():
    """Regrouping is per-block: separate blocks must never concatenate into a match.

    The split fix must not become a greenwash -- a line that genuinely never shipped
    stays missing even when its characters exist across unrelated sibling nodes.
    """
    ocr = {"lines": [{"id": "L1", "text": "ALPHA BETA", "conf": 0.9}]}
    layers = [
        {"id": "c_B0", "type": "text", "text": "ALPHA"},
        {"id": "c_B1", "type": "text", "text": "BETA"},
    ]
    design = {"layers": layers, "kept_in_photo": []}

    result = pixel_diff._text_editability(ocr, design, layers)

    assert result["editable_text_correct"] == 0
    assert result["editable_text_recall"] == 0.0


# ── F-honesty: the four metrics that used to mislead in both directions ──────────────
#
# Each test below locks in the property that made a metric honest: the gate stays SILENT on
# the benign phenomenon it used to punish, and still FIRES on the real defect it must never
# miss. See the docstrings in pixel_diff for the measured per-fixture evidence.


def _text_page(size=(640, 640), shift=0, bg="white"):
    """Text-like glyph runs whose TAILS land near a window boundary — the exact geometry of
    every real worst window (009/104/107): a drift walks the tail out of the window, leaving
    it blank in the render and glyph-edged in the source, which is what collapses a 64px
    window's SSIM to ~0.02 on text a designer cannot tell apart."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for row in range(4):
        y = 100 + row * 120
        for col in range(6):
            x = 64 + col * 60 + shift
            draw.rectangle([x, y, x + 44, y + 56], fill="black")
    return img


def test_worst_region_does_not_hard_fail_a_faithful_render_that_only_drifted(tmp_path):
    """The bench-10 lie: 13 of 16 fixtures hard-failed local-ssim-worst-region, and every
    window was glyphs drifted by font substitution (009 "We zien je", 104 "ration", 107
    "EEK 4" — all visually identical). A pure translation is not damage."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    _text_page().save(source)
    _text_page(shift=20).save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    # The window really does score below the floor — this is the artifact, not a soft case.
    assert result["local_ssim_worst_window"]["ssim"] < 0.10
    assert "local-ssim-worst-region" not in _rules(result)
    # ...and it is REPORTED, never silently dropped.
    assert any(w["rule"] == "local-ssim-worst-region-shifted"
               for w in result["quality_warnings"])


def test_worst_region_still_hard_fails_content_that_vanished(tmp_path):
    """025's missing emoji: content in the source and NOWHERE in the render. No translation
    can explain it, so it must stay a hard fail."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    _text_page().save(source)
    erased = _text_page()
    ImageDraw.Draw(erased).rectangle([20, 20, 180, 180], fill="white")
    erased.save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert "local-ssim-worst-region" in _rules(result)


def test_worst_region_still_hard_fails_a_fabricated_ghost(tmp_path):
    """The reverse of a deletion, and the reason the translation test is symmetric: a
    one-directional search would excuse an ADDED ghost (some nearby render patch is also
    blank). A fabricated double has no counterpart in the source and must stay hard."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    clean = _text_page()
    clean.save(source)
    ghosted = _text_page()
    ghosted.paste(clean.crop((20, 30, 180, 60)), (20, 250))
    ghosted.save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert "local-ssim-worst-region" in _rules(result)


def test_text_recall_scores_the_deliverable_not_an_ocr_of_our_own_render(tmp_path):
    """131: we ship 'BLACK FRIDAY SALE' character-perfect, doctr reads our own correct render
    back as 'BLACKERIDAYSALE', and recall scored 0.556 — worst in the run. Every text fix
    looked like a regression. Recall must score what we DELIVER."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    source_ocr = {"lines": [
        {"id": "L0", "text": "BLACK FRIDAY SALE", "conf": 0.99},
        {"id": "L1", "text": "BUY 2, GET 1 FREE + FREE SHIPPING $100+", "conf": 0.99},
    ]}
    # What the engine actually reads back off our own correct glyphs.
    render_ocr = {"lines": [
        {"id": "R0", "text": "BLACKERIDAYSALE", "conf": 0.9},
        {"id": "R1", "text": "BUY2. GETIFRE: + FREE SHPPINC $100+", "conf": 0.9},
    ]}
    design = {"layers": [
        {"id": "a", "type": "text", "text": "BLACK FRIDAY SALE",
         "style": {"fontFamily": "Inter"}, "meta": {}},
        {"id": "b", "type": "text", "text": "BUY 2, GET 1 FREE + FREE SHIPPING $100+",
         "style": {"fontFamily": "Inter"}, "meta": {}},
    ], "meta": {"editable_ratio": 1.0}}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=source_ocr, render_ocr=render_ocr, design=design)

    assert result["text_recall"] == 1.0
    assert result["text_recall_detail"]["basis"] == "delivered-vs-source-truth"
    assert "low-text-recall" not in _rules(result)
    # The round-trip is not discarded — it is reported under a name that means what it says.
    assert result["render_text_legibility"]["ratio"] < 1.0


def test_text_recall_still_falls_when_copy_is_not_delivered(tmp_path):
    """The metric must still discriminate, or a 1.0 means nothing."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    source_ocr = {"lines": [
        {"id": "L0", "text": "HEADLINE COPY", "conf": 0.99},
        {"id": "L1", "text": "DROPPED ENTIRELY", "conf": 0.99},
    ]}
    design = {"layers": [
        {"id": "a", "type": "text", "text": "HEADLINE COPY",
         "style": {"fontFamily": "Inter"}, "meta": {}},
    ], "meta": {"editable_ratio": 1.0}}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=source_ocr, render_ocr={"lines": []}, design=design)

    assert result["text_recall"] == 0.5
    assert "DROPPED ENTIRELY" in result["text_recall_detail"]["missing_lines"]


def test_text_recall_without_a_design_scores_the_render_and_says_so(tmp_path):
    """figma_verify compares a REAL Figma export with no design.json in hand. There the
    rendered pixels ARE the deliverable, so OCRing them is right — but the report must say
    which question it answered rather than passing off one for the other."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    source_ocr = {"lines": [{"id": "L0", "text": "Hello World", "conf": 0.9},
                            {"id": "L1", "text": "Buy Now", "conf": 0.9}]}
    render_ocr = {"lines": [{"id": "R0", "text": "Hello World", "conf": 0.9}]}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path),
                                source_ocr=source_ocr, render_ocr=render_ocr)

    assert result["text_recall"] == 0.5
    assert result["text_recall_detail"]["basis"] == "render-ocr (no design supplied)"


def test_color_fidelity_does_not_hard_fail_when_only_glyph_coverage_differs(tmp_path):
    """067: every hue exact, yet color-fidelity failed because 20% of text pixels sat at
    dE>50 (black vs white) purely from font substitution. A colour gate must not fire when
    no colour is wrong."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    _text_page().save(source)
    _text_page(shift=20).save(render)   # same ink, same hues, different coverage

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert result["color_similarity"] < 0.82        # the whole-image number still dips...
    assert "color-fidelity" not in _rules(result)   # ...but nothing is the wrong colour
    assert any(w["rule"] == "color-fidelity-coverage-artifact"
               for w in result["quality_warnings"])


def test_color_fidelity_still_hard_fails_a_real_hue_error(tmp_path):
    """A tint is a real colour defect and must still be caught."""
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    _text_page().save(source)
    tinted = np.asarray(_text_page()).copy()
    tinted[..., 2] = np.clip(tinted[..., 2].astype(int) + 90, 0, 255)
    Image.fromarray(tinted).save(render)

    result = pixel_diff.compare(str(source), str(render), str(tmp_path))

    assert "color-fidelity" in _rules(result)


def test_declared_font_style_disagreeing_with_the_drawn_file_is_a_hard_fail(tmp_path):
    """013: declared fontStyle Bold while carrying an ITALIC file. The preview resolved the
    FILE and drew italic (SSIM happy); Figma resolves the STYLE NAME and ships it UPRIGHT.
    Invisible to every pixel metric by construction."""
    import glob as _glob

    import pytest

    italic = _glob.glob(os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                                     "Fonts", "ariali.ttf"))
    if not italic:
        pytest.skip("no italic system font available to stage the mismatch")
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    Image.new("RGB", (60, 40), "white").save(source)
    Image.new("RGB", (60, 40), "white").save(render)
    # NB: the declared family must not be "Arial" — _text_font special-cases that name and
    # resolves Windows' canonical arialbd.ttf directly, which would never reach the candidate.
    design = {"layers": [
        {"id": "c_B0", "type": "text", "text": "We NEVER do this",
         "style": {"fontFamily": "Inter", "fontStyle": "Bold", "fontWeight": 700,
                   "fontSize": 32,
                   "fontCandidates": [{"family": "Inter", "style": "Bold", "weight": 700,
                                       "path": italic[0], "family_resolved": True}]},
         "meta": {}},
    ], "meta": {"editable_ratio": 1.0}}

    result = pixel_diff.compare(str(source), str(render), str(tmp_path), design=design)

    assert "font-style-mismatch" in _rules(result)
    audit = result["deliverable_font_consistency"]
    assert audit["italic_drawn_not_declared"] == 1
    assert audit["preview_matches_declaration_ratio"] < 1.0


def test_ink_ownership_ledger_flags_verbatim_double_and_respects_geometry(tmp_path):
    """The invariant tripwire: a native node OVER verbatim source ink in the plate is a
    DOUBLE; the same text elsewhere (no box overlap) must NOT alias into one (088's
    ribbon 'BLACK' vs the headline node was exactly that false positive)."""
    import numpy as np
    from PIL import Image
    from src import pixel_diff as pd

    src = np.full((200, 400), 255.0, dtype=np.float32)
    src[20:60, 20:200] = 0.0          # ink for line A (stays in plate -> double)
    src[120:150, 20:150] = 0.0        # ink for line B (cleaned from plate)
    plate = src.copy()
    plate[120:150, 20:150] = 255.0    # B cleaned; A untouched
    Image.fromarray(plate.astype('uint8')).save(tmp_path / 'background_clean.png')

    ocr = {"lines": [
        {"text": "HELLO WORLD", "conf": 0.9, "box": {"x": 20, "y": 20, "w": 180, "h": 40}},
        {"text": "GOODBYE", "conf": 0.9, "box": {"x": 20, "y": 120, "w": 130, "h": 30}},
    ]}
    design = {"layers": [
        {"id": "t1", "type": "text", "text": "HELLO WORLD",
         "box": {"x": 20, "y": 20, "w": 180, "h": 40}},
        {"id": "t2", "type": "text", "text": "GOODBYE",
         "box": {"x": 20, "y": 120, "w": 130, "h": 30}},
        # Same text FAR AWAY: must not alias line A's ownership.
        {"id": "t3", "type": "text", "text": "HELLO WORLD",
         "box": {"x": 300, "y": 170, "w": 90, "h": 20}},
    ]}
    led = pd._ink_ownership_ledger(ocr, design, str(tmp_path), src)
    states = {l["text"]: l["state"] for l in led["lines"]}
    assert states["HELLO WORLD"] == "DOUBLE"      # node overlaps + plate holds ink
    assert states["GOODBYE"] == "native-clean"    # node overlaps + plate cleaned
    assert led["doubles"] == 1
