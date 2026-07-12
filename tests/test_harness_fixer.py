"""Tests for harness_fixer expanded automatic fixes."""
import json
import os

from src import harness, harness_fixer


def _minimal_design(tmp_path):
    design = {"id": "demo-run", "canvas": {"w": 10, "h": 10}, "layers": []}
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return path


def test_fix_ocr_stack_enables_vlm_judge_and_easyocr():
    cfg, applied = harness_fixer.fix_ocr_stack({})

    assert "boost-ocr-stack" in applied
    assert cfg["vlm"]["enabled"] is True
    assert cfg["vlm"]["ocr_judge"]["enabled"] is True
    assert "easyocr" in cfg["ocr"]["challengers"]


def test_fix_ocr_stack_is_idempotent():
    cfg = {
        "vlm": {"enabled": True, "ocr_judge": {"enabled": True}},
        "ocr": {"challengers": ["easyocr"]},
    }
    _, applied = harness_fixer.fix_ocr_stack(cfg)
    assert applied == []


def test_fix_vlm_stack_enables_segment_filter_for_sam_issues():
    cfg, applied = harness_fixer.fix_vlm_stack({}, {"category": "sam"})

    assert "boost-vlm-stack" in applied
    assert cfg["vlm"]["segment_filter"]["enabled"] is True
    assert "scene_text" not in cfg["vlm"] or not cfg["vlm"].get("scene_text", {}).get("enabled")


def test_fix_vlm_stack_enables_scene_text_for_text_issues():
    cfg, applied = harness_fixer.fix_vlm_stack({}, {"category": "text"})

    assert "boost-vlm-stack" in applied
    assert cfg["vlm"]["scene_text"]["enabled"] is True
    assert "segment_filter" not in cfg["vlm"] or not cfg["vlm"].get("segment_filter", {}).get("enabled")


def test_fix_inpaint_forces_big_lama_and_widens_button_mask():
    cfg, applied = harness_fixer.fix_inpaint({"inpaint": {"mode": "auto", "mask_dilate": {"button": 4}}})

    assert "force-lama-inpaint" in applied
    assert cfg["inpaint"]["mode"] == "big-lama"
    assert cfg["inpaint"]["mask_dilate"]["button"] >= 6
    assert cfg["inpaint"]["mask_dilate"]["text"] >= 3
    assert cfg["inpaint"]["multipass_fraction"] == 0.08


def test_fix_layout_tightens_container_inference():
    cfg, applied = harness_fixer.fix_layout({"layout": {"min_container_frac": 0.002}})

    assert "tighten-containers" in applied
    assert cfg["layout"]["min_container_frac"] == 0.001


def test_fix_staging_reruns_figma_import_when_inbox_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _minimal_design(run_dir)
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    calls = []

    def fake_import(design_path, rd, cfg):
        calls.append((design_path, rd, cfg))
        manifest = {
            "doc_id": "demo-run",
            "staged_dir": "runs/demo-run",
            "run_dir": str(run_dir),
        }
        staged = inbox / "runs" / "demo-run"
        staged.mkdir(parents=True)
        (staged / "design.json").write_text("{}", encoding="utf-8")
        (inbox / "inbox.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr("src.figma_import.import_design", fake_import)

    cfg = {"figma": {"inbox": str(inbox), "mode": "plugin"}}
    assert harness_fixer.staging_needs_fix(str(run_dir), cfg) is True

    patched, applied = harness_fixer.fix_staging(str(run_dir), cfg)

    assert applied == ["restage-inbox"]
    assert len(calls) == 1
    assert calls[0][1] == str(run_dir)
    assert patched["figma"]["enabled"] is True
    assert harness_fixer.staging_needs_fix(str(run_dir), cfg) is False


def test_apply_fixer_round_applies_critic_suggestions(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _minimal_design(run_dir)

    monkeypatch.setattr(
        "src.figma_import.import_design",
        lambda *_a, **_k: {"ok": True},
    )

    critic_output = {
        "issues": [
            {"category": "ocr", "score": 0.9, "suggested_fix_ids": ["ocr_stack"]},
            {"category": "inpaint", "score": 0.7, "suggested_fix_ids": ["inpaint"]},
        ],
        "suggested_fix_ids": ["layout"],
    }

    cfg, fixes = harness_fixer.apply_fixer_round(str(run_dir), {}, critic_output)

    assert "boost-ocr-stack" in fixes
    assert "force-lama-inpaint" in fixes
    assert "tighten-containers" in fixes
    assert cfg["vlm"]["ocr_judge"]["enabled"] is True
    assert cfg["inpaint"]["mode"] == "big-lama"
    assert cfg["layout"]["min_container_frac"] == 0.001


def test_harness_actionable_includes_fixer_actions():
    for pair in [
        ("figma", "restage-inbox"),
        ("ocr", "boost-stack"),
        ("vlm", "boost-stack"),
        ("inpaint", "force-lama"),
        ("layout", "tighten-containers"),
    ]:
        assert pair in harness.ACTIONABLE


def test_harness_maps_fixer_repairs_to_resume_stages():
    assert harness.resume_stage_for({"stage": "figma", "action": "restage-inbox"}) == "figma"
    assert harness.resume_stage_for({"stage": "ocr", "action": "boost-stack"}) == "ocr"
    assert harness.resume_stage_for({
        "stage": "vlm", "action": "boost-stack", "params": {"focus": "text"},
    }) == "text"
    assert harness.resume_stage_for({
        "stage": "vlm", "action": "boost-stack", "params": {"focus": "elements"},
    }) == "elements"
    assert harness.resume_stage_for({"stage": "inpaint", "action": "force-lama"}) == "reconstruct"
    assert harness.resume_stage_for({"stage": "layout", "action": "tighten-containers"}) == "layout"


def test_harness_config_patches_for_fixer_actions():
    ocr = harness.config_patches_for({
        "stage": "ocr", "action": "boost-stack",
    })
    assert ocr["vlm"]["ocr_judge"]["enabled"] is True
    assert "easyocr" in ocr["ocr"]["challengers"]

    inpaint = harness.config_patches_for({
        "stage": "inpaint", "action": "force-lama",
    })
    assert inpaint["inpaint"]["mode"] == "big-lama"
    assert inpaint["inpaint"]["mask_dilate"]["button"] >= 6

    layout = harness.config_patches_for({
        "stage": "layout", "action": "tighten-containers",
    })
    assert layout["layout"]["min_container_frac"] == 0.001


def test_repair_for_fix_maps_fix_ids():
    repair = harness_fixer.repair_for_fix("ocr_stack", {"detail": "low recall", "severity": "high"})
    assert repair["stage"] == "ocr"
    assert repair["action"] == "boost-stack"
    assert harness.is_actionable(repair)
