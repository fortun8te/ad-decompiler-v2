"""Tests for repair.assess() coverage of QA failure modes."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, repair


def _actions(repairs):
    return {(item["stage"], item["action"]) for item in repairs}


def test_repair_suggests_rerun_detection_for_low_element_recall():
    repairs = repair.assess(
        {},
        {"element_recall": 0.55},
        {"lines": []},
        {},
    )
    assert ("sam3", "rerun-detection") in _actions(repairs)
    choice = harness.recommended_resume(repairs)
    assert choice is not None
    assert choice["resume"] == "sam"
    assert choice["patches"]["sam3"]["enabled"] is True
    assert choice["patches"]["sam3"]["confidence"] == 0.38
    assert choice["patches"]["sam3"]["box_refine_confidence"] == 0.30
    assert choice["patches"]["vlm"]["element_propose"]["enabled"] is True
    assert choice["patches"]["vlm"]["element_propose"]["lightweight_grid"] is True


def test_repair_computes_element_recall_from_artifacts(tmp_path):
    (tmp_path / "elements.json").write_text(json.dumps([
        {"id": "E0", "meta": {"role": "product"}},
        {"id": "E1", "meta": {"role": "icon"}},
        {"id": "E2", "meta": {"role": "product"}},
    ]), encoding="utf-8")
    (tmp_path / "reconstruction.json").write_text(json.dumps({
        "candidates": [
            {"id": "E0", "target": "image"},
            {"id": "E9", "target": "image"},
        ],
    }), encoding="utf-8")

    repairs = repair.assess({}, {}, {"lines": []}, {"run_dir": str(tmp_path)})
    assert any("element recall 0.33" in item["reason"] for item in repairs)


def test_repair_suggests_revalidate_for_vlm_rejected_segments():
    design = {
        "layers": [
            {"id": "product", "type": "image", "meta": {"vlm_rejected": True}},
        ],
    }
    repairs = repair.assess(design, {}, {"lines": []}, {})
    assert ("sam3", "revalidate-rejected") in _actions(repairs)
    choice = harness.recommended_resume(repairs)
    assert choice["resume"] == "sam"
    assert choice["patches"]["vlm"]["segment_filter"]["enabled"] is False


def test_repair_suggests_restage_inbox_for_staging_failure():
    repairs = repair.assess(
        {},
        {"staged": False, "staging_error": "inbox.json was not written"},
        {"lines": []},
        {},
    )
    assert ("figma", "restage-inbox") in _actions(repairs)
    choice = harness.recommended_resume(repairs)
    assert choice["resume"] == "figma"
    assert choice["patches"]["figma"]["enabled"] is True


def test_repair_hard_fails_map_to_actionable_repairs():
    qa = {
        "hard_fails": [
            {"rule": "missing-assets", "detail": "product.png missing"},
            {"rule": "unclean-background", "detail": "plate not inpaint"},
            {"rule": "staging-failed", "detail": "bridge write failed"},
            {"rule": "low-element-recall", "detail": "0.40 < 0.75"},
            {"rule": "vlm-rejected", "detail": "2 junk segments marked"},
            {"rule": "invalid-schema", "detail": "layers[0] missing id"},
            {"rule": "sam3-unavailable", "detail": "checkpoint missing"},
        ],
    }
    repairs = repair.assess({}, qa, {"lines": []}, {})
    actions = {item["action"] for item in repairs}
    assert {
        "restage-assets",
        "rebuild-clean-plate",
        "restage-inbox",
        "rerun-detection",
        "revalidate-rejected",
        "rebuild-schema",
    } <= actions


def test_repair_unclean_background_from_design_root_without_hard_fail():
    design = {
        "layers": [
            {"id": "bg", "type": "image", "meta": {"role": "background", "source": "normalized"}},
        ],
    }
    repairs = repair.assess(design, {}, {"lines": []}, {})
    assert ("inpaint", "rebuild-clean-plate") in _actions(repairs)


def test_repair_per_layer_element_recall_targets_sam3(tmp_path):
    repairs = repair.assess(
        {},
        {"per_layer": [{"id": "logo", "role": "icon", "recall": 0.4}]},
        {"lines": []},
        {},
    )
    assert any(
        item["stage"] == "sam3"
        and item["action"] == "rerun-detection"
        and item["target_id"] == "logo"
        for item in repairs
    )
