import copy
import json

import pytest
from PIL import Image

import run_pipeline
from src import scene_intent


def test_scene_intent_freezes_button_structure_before_paint_hydration():
    merged = [
        {
            "id": "button-shell",
            "target": "shape",
            "box": {"x": 20, "y": 30, "w": 160, "h": 52},
            "meta": {"role": "button"},
        },
        {
            "id": "button-label",
            "target": "text",
            "text": "Buy now",
            "box": {"x": 60, "y": 44, "w": 80, "h": 24},
            "meta": {"role": "cta"},
        },
    ]
    intent = scene_intent.plan(merged, {"w": 400, "h": 300})

    planned = copy.deepcopy(intent["tree"])
    frame = planned[0]
    assert intent["planned_source_ids"] == ["button-shell", "button-label"]
    assert frame["id"] == "button-shell"
    assert frame["target"] == "group"
    assert frame["children"][0]["id"] == "button-label"
    assert frame["children"][0]["box"] == {"x": 40, "y": 14, "w": 80, "h": 24}

    hydrated = scene_intent.hydrate(intent, {
        "candidates": [
            {
                "id": "button-shell",
                "target": "shape",
                # Deliberately different geometry: this is paint material, not a new
                # structural instruction.
                "box": {"x": 1, "y": 2, "w": 3, "h": 4},
                "fill": {"kind": "flat", "color": "#111111"},
                "radius": 12,
                "meta": {"style_extraction": {"source": "fixture"}},
            },
            {
                "id": "button-label",
                "target": "text",
                "text": "Buy now",
                "box": {"x": 1, "y": 2, "w": 3, "h": 4},
                "style": {"fontFamily": "Inter", "fontSize": 18},
                "meta": {"mask_approval": {"accepted": True}},
            },
        ],
    })

    frame = hydrated[0]
    assert frame["target"] == "group"
    assert frame["fill"]["color"] == "#111111"
    assert frame["radius"] == 12
    assert frame["children"][0]["id"] == "button-label"
    assert frame["children"][0]["box"] == planned[0]["children"][0]["box"]
    assert frame["layout"] == planned[0]["layout"]
    assert frame["meta"]["style_extraction"] == {"source": "fixture"}
    assert frame["meta"]["scene_intent_id"] == "button-shell"
    assert frame["children"][0]["meta"]["scene_intent_id"] == "button-label"


def test_scene_intent_rejects_unplanned_reconstruction_layers():
    intent = scene_intent.plan([
        {"id": "copy", "target": "text", "text": "Hi",
         "box": {"x": 10, "y": 10, "w": 30, "h": 12}},
    ], {"w": 100, "h": 80})

    with pytest.raises(scene_intent.SceneIntentError, match="unplanned reconstructed ids"):
        scene_intent.hydrate(intent, {
            "candidates": [
                {"id": "copy", "target": "text", "text": "Hi",
                 "box": {"x": 10, "y": 10, "w": 30, "h": 12}},
                {"id": "derived", "target": "image",
                 "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
            ],
        })


def test_scene_intent_keeps_declared_comparison_splits_inside_the_planned_photo():
    intent = scene_intent.plan([
        {"id": "photo", "target": "image", "box": {"x": 20, "y": 30, "w": 200, "h": 100},
         "meta": {"role": "photo"}},
    ], {"w": 300, "h": 180})

    hydrated = scene_intent.hydrate(intent, {
        "candidates": [
            {"id": "photo", "target": "drop", "box": {"x": 20, "y": 30, "w": 200, "h": 100},
             "meta": {"removal_required": True, "suppression_reason": "split-before-after"}},
            {"id": "photo-before", "target": "image", "src": "assets/before.png",
             "box": {"x": 20, "y": 30, "w": 100, "h": 100},
             "meta": {"role": "comparison-column", "comparison_side": "before", "parent_id": "photo"}},
            {"id": "photo-after", "target": "image", "src": "assets/after.png",
             "box": {"x": 120, "y": 30, "w": 100, "h": 100},
             "meta": {"role": "comparison-column", "comparison_side": "after", "parent_id": "photo"}},
        ],
    })

    frame = hydrated[0]
    assert frame["id"] == "photo"
    assert frame["target"] == "group"
    assert frame["meta"]["scene_intent_derived_group"] is True
    assert [child["id"] for child in frame["children"]] == ["photo-before", "photo-after"]
    assert frame["children"][0]["box"] == {"x": 0.0, "y": 0.0, "w": 100, "h": 100}
    assert frame["children"][1]["box"] == {"x": 100.0, "y": 0.0, "w": 100, "h": 100}
    assert all(child["meta"]["scene_intent_derived_from"] == "photo"
               for child in frame["children"])


def test_scene_intent_accepts_explicit_clean_plate_comparison_columns_at_root():
    intent = scene_intent.plan([
        {"id": "copy", "target": "text", "text": "Before / After",
         "box": {"x": 10, "y": 10, "w": 100, "h": 20}},
    ], {"w": 200, "h": 100})

    hydrated = scene_intent.hydrate(intent, {
        "candidates": [
            {"id": "copy", "target": "text", "text": "Before / After",
             "box": {"x": 10, "y": 10, "w": 100, "h": 20}},
            {"id": "comparison-plate-before", "target": "image", "src": "assets/before.png",
             "box": {"x": 0, "y": 0, "w": 100, "h": 100},
             "meta": {"role": "comparison-column", "comparison_side": "before",
                      "source": "clean-plate-column"}},
            {"id": "comparison-plate-after", "target": "image", "src": "assets/after.png",
             "box": {"x": 100, "y": 0, "w": 100, "h": 100},
             "meta": {"role": "comparison-column", "comparison_side": "after",
                      "source": "clean-plate-column"}},
        ],
    })

    roots = {node["id"]: node for node in hydrated}
    assert set(roots) == {"copy", "comparison-plate-before", "comparison-plate-after"}
    assert roots["comparison-plate-before"]["meta"]["scene_intent_derived_root"] is True
    assert roots["comparison-plate-after"]["meta"]["scene_intent_derived_root"] is True


def test_scene_intent_keeps_a_suppressed_shell_as_a_transparent_structural_frame():
    intent = scene_intent.plan([
        {"id": "shell", "target": "shape", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
         "fill": {"kind": "flat", "color": "#111111"}, "meta": {"role": "button"}},
        {"id": "label", "target": "text", "text": "Buy now",
         "box": {"x": 60, "y": 44, "w": 80, "h": 24}, "meta": {"role": "cta"}},
    ], {"w": 300, "h": 150})

    hydrated = scene_intent.hydrate(intent, {
        "candidates": [
            {"id": "shell", "target": "drop", "box": {"x": 20, "y": 30, "w": 160, "h": 52},
             "meta": {"keep_in_background": True, "suppression_reason": "fully-contained"}},
            {"id": "label", "target": "text", "text": "Buy now",
             "box": {"x": 60, "y": 44, "w": 80, "h": 24}},
        ],
    })

    frame = hydrated[0]
    assert frame["target"] == "group"
    assert "fill" not in frame
    assert frame["meta"]["scene_intent_material_suppressed"] is True
    assert [child["id"] for child in frame["children"]] == ["label"]


def test_scene_intent_fingerprint_rejects_changed_merge_or_layout_config():
    merged = [{"id": "copy", "target": "text", "text": "Hi",
               "box": {"x": 10, "y": 10, "w": 30, "h": 12}}]
    canvas = {"w": 100, "h": 80}
    intent = scene_intent.plan(merged, canvas, {"layout": {"min_container_frac": 0.01}})

    assert scene_intent.is_current(intent, merged, canvas, {"layout": {"min_container_frac": 0.01}})
    changed_merge = copy.deepcopy(merged)
    changed_merge[0]["box"]["x"] = 11
    assert not scene_intent.is_current(intent, changed_merge, canvas, {"layout": {"min_container_frac": 0.01}})
    assert not scene_intent.is_current(intent, merged, canvas, {"layout": {"min_container_frac": 0.02}})


def test_structure_stage_is_between_merge_and_reconstruct():
    assert run_pipeline.STAGES.index("merge") < run_pipeline.STAGES.index("structure")
    assert run_pipeline.STAGES.index("structure") < run_pipeline.STAGES.index("reconstruct")
    assert run_pipeline.STAGES.index("reconstruct") < run_pipeline.STAGES.index("layout")


def test_resume_replans_a_stale_scene_intent(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (80, 60), "white").save(source)
    monkeypatch.setattr(
        run_pipeline.ocr,
        "run_ocr",
        lambda path, cfg, run_dir=None: {"engine": "fixture", "lines": []},
    )
    cfg = {
        "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
        "inpaint": {"mode": "opencv"}, "figma": {"enabled": False}, "qa_ocr": False,
    }
    run_dir = tmp_path / "run"
    assert run_pipeline.run_one(str(source), str(run_dir), cfg)["ok"]

    intent_path = run_dir / "scene_intent.json"
    stale = json.loads(intent_path.read_text(encoding="utf-8"))
    stale["planning_fingerprint"] = "stale"
    intent_path.write_text(json.dumps(stale), encoding="utf-8")
    calls = []
    real_plan = run_pipeline.scene_intent.plan
    monkeypatch.setattr(
        run_pipeline.scene_intent, "plan",
        lambda *args, **kwargs: calls.append(1) or real_plan(*args, **kwargs),
    )

    resumed = run_pipeline.run_one(str(source), str(run_dir), cfg, "reconstruct")
    assert resumed["ok"]
    assert calls == [1]


def test_pipeline_uses_legacy_layout_as_a_required_degradation(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (80, 60), "white").save(source)
    monkeypatch.setattr(
        run_pipeline.scene_intent,
        "hydrate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            scene_intent.SceneIntentError("fixture mismatch")
        ),
    )
    monkeypatch.setattr(
        run_pipeline.ocr,
        "run_ocr",
        lambda path, cfg, run_dir=None: {"engine": "fixture", "lines": []},
    )

    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"}, "figma": {"enabled": False}, "qa_ocr": False,
        },
    )

    report = json.loads((run_dir / "runtime_report.json").read_text(encoding="utf-8"))
    qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["runtime_ok"] is False
    assert any(item["component"] == "structure" and item["required"]
               for item in report["degraded"])
    assert "structure-unavailable" in {item["rule"] for item in qa["hard_fails"]}
