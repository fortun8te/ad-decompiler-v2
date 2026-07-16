"""Tests for deterministic harness critic analysis."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import agent_debug, harness_critic


def _write_run(run_dir, *, qa, repairs=None, runtime=None, log_lines=None, debug_lines=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    if repairs is not None:
        (run_dir / "repairs.json").write_text(json.dumps(repairs), encoding="utf-8")
    if runtime is not None:
        (run_dir / "runtime_report.json").write_text(json.dumps(runtime), encoding="utf-8")
    if log_lines is not None:
        (run_dir / "pipeline.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if debug_lines is not None:
        sid = agent_debug.session_id() or "test-session"
        path = run_dir / f"debug-{sid}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for entry in debug_lines:
                handle.write(json.dumps(entry) + "\n")


def test_analyze_scores_ocr_and_writes_critic_json(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        qa={
            "ok": False,
            "text_recall": 0.55,
            "hard_fails": [],
            "per_layer": [],
        },
        repairs=[
            {
                "stage": "ocr",
                "action": "rerun",
                "reason": "text_recall 0.55",
                "severity": "high",
                "params": {"upscale": True},
            },
            {
                "stage": "pipeline",
                "action": "review",
                "reason": "manual",
                "severity": "low",
            },
        ],
        runtime={"status": "ok", "degraded": [], "violations": []},
    )

    output = harness_critic.analyze(str(run_dir))

    assert output["qa_ok"] is False
    assert output["scores"]["ocr"]["score"] > 0.3
    assert output["scores"]["ocr"]["severity"] in ("medium", "high", "critical")
    assert "ocr:rerun" in output["suggested_fix_ids"]
    assert any(item["reason"] == "manual review required" for item in output["blockers"])
    assert output["prioritized_issues"][0]["category"] == "ocr"
    assert (run_dir / "critic.json").exists()
    saved = json.loads((run_dir / "critic.json").read_text(encoding="utf-8"))
    assert saved["scores"]["ocr"]["score"] == output["scores"]["ocr"]["score"]


def test_analyze_detects_inpaint_and_layout_from_hard_fails(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        qa={
            "ok": False,
            "edge_f1": 0.5,
            "hard_fails": [
                {"rule": "background-leakage", "detail": "foreground in clean plate"},
            ],
        },
        repairs=[
            {"stage": "inpaint", "action": "rebuild-clean-plate", "reason": "leakage", "severity": "high"},
            {"stage": "layout", "action": "refit-geometry", "reason": "edge fidelity", "severity": "medium"},
        ],
    )

    output = harness_critic.analyze(str(run_dir))

    assert output["scores"]["inpaint"]["score"] > output["scores"]["layout"]["score"]
    assert "inpaint:rebuild-clean-plate" in output["suggested_fix_ids"]


def test_analyze_uses_runtime_violations_and_log_tail(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        qa={"ok": False, "hard_fails": []},
        repairs=[],
        runtime={
            "status": "degraded",
            "violations": [
                {
                    "rule": "sam3-unavailable",
                    "detail": "required sam3 did not complete: worker offline",
                    "hard": True,
                }
            ],
        },
        log_lines=[
            "[12:00:00] normalize → 1080x1080",
            "[12:00:05] sam3[worker] failed: connection refused",
        ],
    )

    output = harness_critic.analyze(str(run_dir))

    assert output["scores"]["sam"]["score"] > 0.5
    assert any(item["category"] == "sam" for item in output["blockers"])


def test_critic_review_removes_blocked_and_low_confidence_repairs(tmp_path):
    repairs = [
        {"stage": "ocr", "action": "rerun", "reason": "text_recall low", "severity": "high"},
        {"stage": "pipeline", "action": "review", "reason": "manual", "severity": "low"},
        {"stage": "ocr", "action": "review", "reason": "disagreement", "severity": "low"},
    ]
    critic_output = harness_critic.analyze(
        str(tmp_path),
        write=False,
    )
    critic_output["scores"] = {
        cat: {"score": 0.0, "severity": "none", "evidence": []}
        for cat in harness_critic.CATEGORIES
    }
    critic_output["blockers"] = [
        {
            "category": "ocr",
            "reason": "infrastructure failure prevents automatic repair",
            "detail": "cudnn not found",
            "auto_fixable": False,
        },
        {
            "category": "staging",
            "reason": "manual review required",
            "detail": "manual",
            "fix_id": "pipeline:review",
            "auto_fixable": False,
        },
    ]

    filtered = harness_critic.critic_review(repairs, critic_output)

    assert all(r.get("action") != "review" for r in filtered)
    assert all(harness_critic._category_for_stage(r.get("stage")) != "ocr" for r in filtered)


def test_critic_review_drops_contradictory_inpaint_vs_layout(tmp_path):
    repairs = [
        {"stage": "inpaint", "action": "rebuild-clean-plate", "reason": "leakage", "severity": "high"},
        {"stage": "layout", "action": "refit-geometry", "reason": "edge fidelity", "severity": "medium"},
    ]
    critic_output = {
        "scores": {
            "ocr": {"score": 0.0},
            "text": {"score": 0.0},
            "sam": {"score": 0.0},
            "inpaint": {"score": 0.8},
            "layout": {"score": 0.3},
            "staging": {"score": 0.0},
        },
        "blockers": [],
    }

    filtered = harness_critic.critic_review(repairs, critic_output)
    fix_ids = [harness_critic.fix_id(r) for r in filtered]

    assert "inpaint:rebuild-clean-plate" in fix_ids
    assert "layout:refit-geometry" not in fix_ids


def test_critic_review_keeps_highest_confidence_duplicate_stage_action(tmp_path):
    repairs = [
        {"stage": "ocr", "action": "rerun", "reason": "low recall", "severity": "medium",
         "params": {"upscale": True}},
        {"stage": "ocr", "action": "rerun", "reason": "low recall duplicate", "severity": "high",
         "params": {"upscale": True}},
    ]
    critic_output = {
        "scores": {
            "ocr": {"score": 0.7},
            "text": {"score": 0.0},
            "sam": {"score": 0.0},
            "inpaint": {"score": 0.0},
            "layout": {"score": 0.0},
            "staging": {"score": 0.0},
        },
        "blockers": [],
    }

    filtered = harness_critic.critic_review(repairs, critic_output)

    assert len(filtered) == 1
    assert filtered[0]["severity"] == "high"
