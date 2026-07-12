"""harness_loop.py — failure-proof orchestrator for QA repair rounds.

Flow per round:
  run_pipeline.run_one → if QA not ok → execute_repairs → critic → fixer → repeat

Never lowers QA thresholds. Writes ``harness_loop.json`` with a full audit trail.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Optional

from src.harness import (
    _flag,
    _qa_accepts,
    execute_repairs,
    harness_enabled,
    harness_max_rounds,
    load_repairs,
    recommended_resume,
    _load_json,
    _write_json,
)
from src.qa_config import visual_pass_ssim


def max_harness_rounds(cfg: Optional[dict] = None, default: int = 3) -> int:
    rounds = harness_max_rounds(cfg or {})
    return max(1, rounds if rounds else default)


def repair_iterations(cfg: Optional[dict] = None, default: int = 2) -> int:
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    return max(1, int(harness.get("repair_iterations", default)))


def in_harness_loop(cfg: Optional[dict] = None) -> bool:
    return bool((((cfg or {}).get("runtime") or {}).get("harness") or {}).get("_in_loop"))


def _threshold_snapshot(cfg: dict) -> dict:
    return {
        "visual_pass_ssim": visual_pass_ssim(cfg),
        "repair": copy.deepcopy(cfg.get("repair") or {}),
    }


def _assert_thresholds_unchanged(snapshot: dict, cfg: dict) -> None:
    current_ssim = visual_pass_ssim(cfg)
    if current_ssim < snapshot["visual_pass_ssim"]:
        raise ValueError(
            f"harness must not lower QA thresholds "
            f"({current_ssim} < {snapshot['visual_pass_ssim']})"
        )
    repair = cfg.get("repair") or {}
    for key, baseline in (snapshot.get("repair") or {}).items():
        if key.endswith("_min") and key in repair:
            if float(repair[key]) < float(baseline):
                raise ValueError(f"harness must not lower repair threshold {key}")


def _cfg_for_pipeline(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("runtime", {})["auto_repair"] = False
    out.setdefault("runtime", {}).setdefault("harness", {})["_in_loop"] = True
    return out


def _qa_summary(qa: dict) -> dict:
    qa = qa or {}
    return {
        "ok": _flag(qa.get("ok")),
        "ssim": qa.get("ssim"),
        "text_recall": qa.get("text_recall"),
        "hard_fails": len(qa.get("hard_fails") or []),
        "repairs": len(qa.get("repairs") or []),
    }


def _run_critic_pass(run_dir: str, cfg: dict) -> dict:
    try:
        from src.harness_critic import analyze, critic_review
        critic_output = analyze(run_dir, write=False, cfg=cfg)
        repairs = load_repairs(run_dir, cfg)
        critic_output["filtered_repairs"] = critic_review(repairs, critic_output)
        return critic_output
    except ImportError:
        return _fallback_critic(run_dir, cfg)


def _fallback_critic(run_dir: str, cfg: dict) -> dict:
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    repairs = load_repairs(run_dir, cfg)
    issues = []
    categories = {
        "ocr": ["text_recall", "ocr"],
        "text": ["text-analysis", "editable_text"],
        "sam": ["sam", "element", "missing-assets"],
        "inpaint": ["inpaint", "background", "leakage"],
        "layout": ["layout", "container"],
        "staging": ["figma", "staging", "inbox"],
    }
    hard_fails = qa.get("hard_fails") or []
    for fail in hard_fails:
        rule = str((fail or {}).get("rule", "")).lower()
        detail = str((fail or {}).get("detail", "")).lower()
        for category, needles in categories.items():
            if any(n in rule or n in detail for n in needles):
                issues.append({"category": category, "rule": fail.get("rule"), "detail": fail.get("detail")})
                break
    if qa.get("text_recall") is not None and qa.get("text_recall", 1) < 0.85:
        issues.append({"category": "ocr", "rule": "text_recall", "detail": qa.get("text_recall")})
    filtered = repairs
    try:
        from src.harness_critic import critic_review
        filtered = critic_review(repairs, {"issues": issues, "repairs": repairs})
    except ImportError:
        pass
    return {
        "issues": issues,
        "suggested_fix_ids": [f"{r.get('stage')}:{r.get('action')}" for r in filtered[:5]],
        "blockers": [r for r in repairs if r not in filtered],
        "filtered_repairs": filtered,
        "repairs_considered": len(repairs),
    }


def _run_fixer_pass(run_dir: str, cfg: dict, critic_output: dict) -> dict:
    try:
        from src.harness_fixer import apply_fixer_round
        patched_cfg, fixes = apply_fixer_round(run_dir, cfg, critic_output)
        return {"cfg": patched_cfg, "fixes": fixes}
    except ImportError:
        return {"cfg": copy.deepcopy(cfg), "fixes": []}


def _resume_after_fixer(run_dir: str, cfg: dict, critic_output: dict) -> str:
    repairs = critic_output.get("filtered_repairs") or load_repairs(run_dir, cfg)
    choice = recommended_resume(repairs)
    return choice["resume"] if choice else "normalize"


def _run_round(
    *,
    round_num: int,
    image_path: str,
    run_dir: str,
    working_cfg: dict,
    threshold_snapshot: dict,
    start_from: str,
    skip_pipeline: bool,
    run_one: Callable[..., dict],
    execute_repairs_fn: Callable[..., dict],
    repair_iters: int,
) -> tuple[dict, dict, bool]:
    """Execute one harness round. Returns (round_record, updated_cfg, should_stop)."""
    round_record: dict[str, Any] = {"round": round_num}

    if not skip_pipeline:
        loop_cfg = _cfg_for_pipeline(working_cfg)
        pipeline_result = run_one(image_path, run_dir, loop_cfg, start_from)
        round_record["pipeline"] = {
            "ok": bool(pipeline_result.get("ok")),
            "start_from": start_from,
        }
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        round_record["qa"] = _qa_summary(qa)
        if _qa_accepts(qa, allow_summary=True):
            round_record["stopped"] = "qa_ok"
            return round_record, working_cfg, True

    repair_cfg = _cfg_for_pipeline(working_cfg)
    repair_summary = execute_repairs_fn(
        run_dir, repair_cfg, max_iterations=repair_iters, run_one=run_one,
    )
    round_record["repairs"] = repair_summary
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    round_record["qa_after_repairs"] = _qa_summary(qa)
    if _qa_accepts(qa, allow_summary=True):
        round_record["stopped"] = "qa_ok_after_repairs"
        return round_record, working_cfg, True

    critic_output = _run_critic_pass(run_dir, working_cfg)
    _write_json(os.path.join(run_dir, "critic.json"), critic_output)
    round_record["critic"] = {
        "issues": len(critic_output.get("prioritized_issues") or critic_output.get("issues") or []),
        "suggested_fix_ids": critic_output.get("suggested_fix_ids") or [],
        "blockers": len(critic_output.get("blockers") or []),
    }

    fixer_result = _run_fixer_pass(run_dir, working_cfg, critic_output)
    _write_json(os.path.join(run_dir, "fixer.json"), {
        "fixes": fixer_result.get("fixes") or [],
        "fix_count": len(fixer_result.get("fixes") or []),
    })
    round_record["fixer"] = {
        "fixes": fixer_result.get("fixes") or [],
        "fix_count": len(fixer_result.get("fixes") or []),
    }

    patched_cfg = fixer_result.get("cfg") or working_cfg
    _assert_thresholds_unchanged(threshold_snapshot, patched_cfg)
    round_record["next_resume"] = _resume_after_fixer(run_dir, patched_cfg, critic_output)

    no_progress = (
        repair_summary.get("stopped") == "no_actionable_repairs"
        and not (fixer_result.get("fixes") or [])
    )
    if no_progress:
        round_record["stopped"] = "no_progress"
        return round_record, patched_cfg, True

    return round_record, patched_cfg, False


def run_until_acceptable(
    image_path: str,
    run_dir: str,
    cfg: Optional[dict] = None,
    max_rounds: int = 3,
    *,
    start_from: str = "normalize",
    pipeline_already_ran: bool = False,
    run_one: Optional[Callable[..., dict]] = None,
    execute_repairs_fn: Optional[Callable[..., dict]] = None,
) -> dict:
    """Run the pipeline and repair loop until QA passes or *max_rounds* is exhausted."""
    if run_one is None:
        import run_pipeline
        run_one = run_pipeline.run_one
    if execute_repairs_fn is None:
        execute_repairs_fn = execute_repairs

    run_dir = os.path.abspath(run_dir)
    cfg = copy.deepcopy(cfg or {})
    max_rounds = max(1, int(max_rounds or max_harness_rounds(cfg)))
    repair_iters = repair_iterations(cfg)
    threshold_snapshot = _threshold_snapshot(cfg)
    working_cfg = copy.deepcopy(cfg)
    rounds: list[dict] = []
    stopped = "max_rounds"

    next_resume = start_from
    for round_num in range(1, max_rounds + 1):
        skip_pipeline = pipeline_already_ran and round_num == 1
        resume = next_resume

        round_record, working_cfg, should_stop = _run_round(
            round_num=round_num,
            image_path=image_path,
            run_dir=run_dir,
            working_cfg=working_cfg,
            threshold_snapshot=threshold_snapshot,
            start_from=resume,
            skip_pipeline=skip_pipeline,
            run_one=run_one,
            execute_repairs_fn=execute_repairs_fn,
            repair_iters=repair_iters,
        )
        next_resume = round_record.get("next_resume") or next_resume
        rounds.append(round_record)
        if should_stop:
            stopped = round_record.get("stopped", stopped)
            break

    final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    summary = {
        "run_dir": run_dir,
        "rounds": rounds,
        "rounds_completed": len(rounds),
        "max_rounds": max_rounds,
        "qa_ok": _qa_accepts(final_qa, allow_summary=True),
        "stopped": stopped,
        "thresholds": threshold_snapshot,
    }
    _write_json(os.path.join(run_dir, "harness_loop.json"), summary)
    return summary


def run_harness_after_pipeline(
    image_path: str,
    run_dir: str,
    cfg: Optional[dict] = None,
    max_rounds: Optional[int] = None,
    *,
    start_from: str = "normalize",
    run_one: Optional[Callable[..., dict]] = None,
    execute_repairs_fn: Optional[Callable[..., dict]] = None,
) -> dict:
    """Continue the harness loop after ``run_one`` already completed the pipeline."""
    cfg = copy.deepcopy(cfg or {})
    rounds_cap = max_rounds if max_rounds is not None else max_harness_rounds(cfg)
    return run_until_acceptable(
        image_path,
        run_dir,
        cfg,
        rounds_cap,
        start_from=start_from,
        pipeline_already_ran=True,
        run_one=run_one,
        execute_repairs_fn=execute_repairs_fn,
    )
