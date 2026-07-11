"""CPU-only tests for local visual metrics and structural QA gates."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import pixel_diff, repair  # noqa: E402


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
