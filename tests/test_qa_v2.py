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
