"""harness_loop.py — failure-proof orchestrator for QA repair rounds.

Flow per round:
  run_pipeline.run_one → if QA not ok → execute_repairs → critic → fixer → repeat

Never lowers QA thresholds. Writes ``harness_loop.json`` with a full audit trail.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import os
from typing import Any, Callable, Optional


def _invoke_run_one(run_one: Callable[..., dict], image_path: str, run_dir: str,
                    cfg: dict, start_from: str) -> dict:
    """Call production and lightweight test/extension runners without guessing arity."""
    try:
        parameters = inspect.signature(run_one).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "start_from" in parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        return run_one(image_path, run_dir, cfg, start_from=start_from)
    return run_one(image_path, run_dir, cfg)

from src.harness import (
    _flag,
    _qa_progress,
    _qa_accepts,
    _repair_id,
    execute_repairs,
    harness_enabled,
    harness_max_rounds,
    is_actionable,
    load_repairs,
    recommended_resume,
    _load_json,
    _write_json,
)
from src.qa_config import visual_pass_ssim

# Artifacts that together define "the design produced this round". Snapshotting these lets
# the loop keep the best-scoring design and roll back a regressing round.
_SNAPSHOT_FILES = ("design.json", "qa.json", "preview.png", "layout.json", "figma_export.png")

# Metrics folded into the harness-local round score. This is a read-only aggregation of QA
# fields already written by pixel_diff — the loop never computes a new visual metric here.
_SCORE_KEYS = (
    "ssim", "visual_score", "text_recall", "editable_text_recall",
    "edge_f1", "color_similarity",
)

_DEFAULT_EPSILON = 0.005
_DEFAULT_PLATEAU_ROUNDS = 2


def convergence_epsilon(cfg: Optional[dict] = None, default: float = _DEFAULT_EPSILON) -> float:
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    try:
        return float(harness.get("epsilon", default))
    except (TypeError, ValueError):
        return default


def plateau_round_limit(cfg: Optional[dict] = None, default: int = _DEFAULT_PLATEAU_ROUNDS) -> int:
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    try:
        return max(1, int(harness.get("plateau_rounds", default)))
    except (TypeError, ValueError):
        return default


def _qa_score(qa: Optional[dict]) -> Optional[float]:
    """Scalar round quality from existing QA fields. Higher is better; None if unknown."""
    qa = qa or {}
    if _qa_accepts(qa, allow_summary=True):
        return 1.0
    values = [float(qa[key]) for key in _SCORE_KEYS if isinstance(qa.get(key), (int, float))]
    if not values:
        return None
    base = sum(values) / len(values)
    hard_fails = qa.get("hard_fails")
    penalty = 0.05 * len(hard_fails) if isinstance(hard_fails, list) else 0.0
    return round(max(0.0, base - penalty), 6)


def _snapshot_artifacts(run_dir: str) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name in _SNAPSHOT_FILES:
        path = os.path.join(run_dir, name)
        try:
            with open(path, "rb") as handle:
                snapshot[name] = handle.read()
        except OSError:
            continue
    return snapshot


def _restore_artifacts(run_dir: str, snapshot: dict[str, bytes]) -> None:
    for name, blob in (snapshot or {}).items():
        path = os.path.join(run_dir, name)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            temporary = path + ".tmp"
            with open(temporary, "wb") as handle:
                handle.write(blob)
            os.replace(temporary, path)
        except OSError:
            continue


def _invoke_execute_repairs(fn, run_dir, cfg, *, max_iterations, run_one, blocked_repairs):
    """Pass ``blocked_repairs`` only to runners that accept it (test fakes may not)."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_blocked = "blocked_repairs" in parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    if accepts_blocked:
        return fn(run_dir, cfg, max_iterations=max_iterations, run_one=run_one,
                  blocked_repairs=set(blocked_repairs or ()))
    return fn(run_dir, cfg, max_iterations=max_iterations, run_one=run_one)


def _round_repair_signatures(repair_summary: dict) -> tuple[list, set]:
    """From a repair summary's attempts, return (applied, non_improving) signatures."""
    applied: list = []
    non_improving: set = set()
    for attempt in (repair_summary or {}).get("attempts") or []:
        repair = attempt.get("repair") or {}
        signature = (repair.get("stage"), repair.get("action"), repair.get("target_id"))
        if signature == (None, None, None):
            continue
        applied.append(signature)
        if not attempt.get("qa_improved") or not attempt.get("pipeline_ok", True):
            non_improving.add(signature)
    return applied, non_improving


def _actionable_signatures(run_dir: str, cfg: dict, blocked: set) -> list:
    """Signatures of currently actionable repairs not already blocked."""
    out: list = []
    for repair in load_repairs(run_dir, cfg) or []:
        if not is_actionable(repair):
            continue
        signature = _repair_id(repair)
        if signature in blocked or signature in out:
            continue
        out.append(signature)
    return out


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
        "visual_score": qa.get("visual_score"),
        "edge_f1": qa.get("edge_f1"),
        "hard_fails": len(qa.get("hard_fails") or []),
        "repairs": len(qa.get("repairs") or []),
    }


def _artifact_fingerprint(path: str) -> str | None:
    """Content fingerprint: mtimes can change even when a runner rewrites stale QA."""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _run_critic_pass(run_dir: str, cfg: dict) -> dict:
    try:
        from src.harness_critic import analyze, critic_review
        critic_output = analyze(run_dir, write=False, cfg=cfg)
        if not isinstance(critic_output, dict):
            raise TypeError("critic returned malformed output")
        repairs = load_repairs(run_dir, cfg)
        critic_output["filtered_repairs"] = critic_review(repairs, critic_output)
        return critic_output
    except (ImportError, TypeError, ValueError, KeyError) as exc:
        fallback = _fallback_critic(run_dir, cfg)
        fallback["critic_error"] = str(exc)
        return fallback


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
        if not isinstance(patched_cfg, dict) or not isinstance(fixes, list):
            raise TypeError("fixer returned malformed output")
        return {"cfg": patched_cfg, "fixes": fixes}
    except (ImportError, TypeError, ValueError, KeyError) as exc:
        return {"cfg": copy.deepcopy(cfg), "fixes": [], "error": str(exc)}


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
    blocked_repairs: Optional[set] = None,
) -> tuple[dict, dict, bool]:
    """Execute one harness round. Returns (round_record, updated_cfg, should_stop)."""
    round_record: dict[str, Any] = {"round": round_num}
    qa_before_repairs = _load_json(os.path.join(run_dir, "qa.json"), {})

    if not skip_pipeline:
        loop_cfg = _cfg_for_pipeline(working_cfg)
        qa_path = os.path.join(run_dir, "qa.json")
        qa_before = _artifact_fingerprint(qa_path)
        try:
            pipeline_result = _invoke_run_one(run_one, image_path, run_dir, loop_cfg, start_from)
            if not isinstance(pipeline_result, dict):
                raise TypeError(
                    f"pipeline runner returned {type(pipeline_result).__name__}, expected dict"
                )
        except Exception as exc:
            round_record["pipeline"] = {
                "ok": False, "start_from": start_from,
                "error": str(exc), "exception": type(exc).__name__,
            }
            round_record["stopped"] = "pipeline_exception"
            return round_record, working_cfg, True
        round_record["pipeline"] = {
            "ok": bool(pipeline_result.get("ok")),
            "start_from": start_from,
            "error": pipeline_result.get("error"),
        }
        if not pipeline_result.get("ok"):
            round_record["stopped"] = "pipeline_failed"
            return round_record, working_cfg, True
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        qa_after = _artifact_fingerprint(qa_path)
        qa_fresh = qa_after is not None and qa_after != qa_before
        round_record["pipeline"]["qa_fresh"] = qa_fresh
        round_record["qa"] = _qa_summary(qa)
        production_result = "qa_ok" in pipeline_result or "runtime_ok" in pipeline_result
        if not qa_fresh and production_result:
            round_record["stopped"] = "qa_not_refreshed"
            return round_record, working_cfg, True
        if _qa_accepts(qa, allow_summary=True):
            round_record["stopped"] = "qa_ok"
            return round_record, working_cfg, True

    qa_before_repairs = _load_json(os.path.join(run_dir, "qa.json"), {})
    repair_cfg = _cfg_for_pipeline(working_cfg)
    try:
        repair_summary = _invoke_execute_repairs(
            execute_repairs_fn, run_dir, repair_cfg,
            max_iterations=repair_iters, run_one=run_one,
            blocked_repairs=blocked_repairs,
        )
        if not isinstance(repair_summary, dict):
            raise TypeError("repair executor returned malformed output")
    except Exception as exc:
        repair_summary = {
            "qa_ok": False, "stopped": "repair_exception", "attempts": [],
            "error": str(exc), "exception": type(exc).__name__,
        }
    round_record["repairs"] = repair_summary
    qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    repair_improved, repair_deltas = _qa_progress(qa_before_repairs, qa)
    round_record["repair_progress"] = {
        "improved": repair_improved,
        "metric_deltas": repair_deltas,
    }
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
        "error": fixer_result.get("error"),
    }

    patched_cfg = fixer_result.get("cfg") or working_cfg
    _assert_thresholds_unchanged(threshold_snapshot, patched_cfg)
    round_record["next_resume"] = _resume_after_fixer(run_dir, patched_cfg, critic_output)

    no_progress = (
        repair_summary.get("stopped") in {
            "no_actionable_repairs", "all_repairs_failed", "missing_input",
        }
        or not repair_improved
    ) and not (fixer_result.get("fixes") or [])
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
    epsilon = convergence_epsilon(cfg)
    plateau_limit = plateau_round_limit(cfg)
    threshold_snapshot = _threshold_snapshot(cfg)
    working_cfg = copy.deepcopy(cfg)
    rounds: list[dict] = []
    stopped = "max_rounds"

    # Convergence state — the direct fix for the observed oscillation.
    blocked: set = set()                       # repairs proven non-improving (no-repeat)
    convergence: list[dict] = []               # per-round (repair, before, after, kept/rolled)
    before_score = _qa_score(_load_json(os.path.join(run_dir, "qa.json"), {}))
    best_score = before_score if before_score is not None else float("-inf")
    best_round = 0
    best_snapshot = _snapshot_artifacts(run_dir) if before_score is not None else {}
    plateau = 0
    rolled_back_rounds = 0

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
            blocked_repairs=blocked,
        )
        next_resume = round_record.get("next_resume") or next_resume
        round_stopped = round_record.get("stopped")

        # ── convergence bookkeeping ────────────────────────────────────────────────
        after_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        after_score = _qa_score(after_qa)
        applied, non_improving = _round_repair_signatures(round_record.get("repairs") or {})
        blocked |= non_improving
        record: dict[str, Any] = {
            "round": round_num,
            "repair": [list(sig) for sig in applied],
            "before_score": before_score,
            "after_score": after_score,
            "delta": None,
            "kept": False,
            "rolled_back": False,
            "blocked": [list(sig) for sig in sorted(non_improving)],
        }

        if after_score is not None:
            delta = after_score - (before_score if before_score is not None else after_score)
            record["delta"] = round(delta, 6)
            if after_score > best_score:
                best_score, best_round = after_score, round_num
                best_snapshot = _snapshot_artifacts(run_dir)
                record["kept"] = True
            elif after_score < best_score - epsilon and best_snapshot and not should_stop:
                # Regression: restore the best design and force a different repair next round.
                _restore_artifacts(run_dir, best_snapshot)
                rolled_back_rounds += 1
                record["rolled_back"] = True
                blocked |= set(applied)
                after_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
                after_score = best_score
                steer = recommended_resume(
                    [r for r in load_repairs(run_dir, working_cfg)
                     if is_actionable(r) and _repair_id(r) not in blocked]
                )
                if steer:
                    next_resume = steer["resume"]
            elif applied and abs(delta) <= epsilon:
                # No-op repair — never retry it.
                blocked |= set(applied)
            plateau = plateau + 1 if abs(delta) < epsilon else 0
            before_score = after_score
        convergence.append(record)
        rounds.append(round_record)

        if should_stop:
            stopped = round_stopped or stopped
            # A "no_progress" stop is premature when the current design still has untried,
            # non-blocked repairs and the round did not exhaust the repair space. Steer to a
            # different operator and keep going rather than oscillating on the same one.
            repair_stopped = (round_record.get("repairs") or {}).get("stopped")
            exhausted_space = repair_stopped in {
                "no_actionable_repairs", "all_repairs_failed", "missing_input",
            }
            alternatives = _actionable_signatures(run_dir, working_cfg, blocked)
            if (
                stopped == "no_progress"
                and not exhausted_space
                and alternatives
                and plateau < plateau_limit
                and round_num < max_rounds
            ):
                choice = recommended_resume(
                    [r for r in load_repairs(run_dir, working_cfg)
                     if is_actionable(r) and _repair_id(r) not in blocked]
                )
                if choice:
                    next_resume = choice["resume"]
                    stopped = "max_rounds"  # not actually stopping — keep converging
                    continue
            break

        if plateau >= plateau_limit:
            stopped = "plateau"
            break

    # Emit the BEST design seen, not merely the last one.
    final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
    live_score = _qa_score(final_qa)
    if best_snapshot and best_score != float("-inf") and (
        live_score is None or live_score < best_score
    ):
        _restore_artifacts(run_dir, best_snapshot)
        final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})

    success_stop = stopped in {"qa_ok", "qa_ok_after_repairs"}
    summary = {
        "run_dir": run_dir,
        "rounds": rounds,
        "rounds_completed": len(rounds),
        "max_rounds": max_rounds,
        "qa_ok": success_stop and _qa_accepts(final_qa, allow_summary=True),
        "stopped": stopped,
        "thresholds": threshold_snapshot,
        "convergence": {
            "epsilon": epsilon,
            "plateau_rounds": plateau_limit,
            "best_round": best_round,
            "best_score": None if best_score == float("-inf") else round(best_score, 6),
            "rolled_back_rounds": rolled_back_rounds,
            "trail": convergence,
        },
    }
    _write_json(os.path.join(run_dir, "harness_loop.json"), summary)
    _patch_runtime_report(run_dir, summary["convergence"], stopped, summary["qa_ok"])
    return summary


def _patch_runtime_report(run_dir: str, convergence: dict, stopped: str, qa_ok: bool) -> None:
    """Surface convergence behaviour in runtime_report.json when it exists (visible/testable)."""
    path = os.path.join(run_dir, "runtime_report.json")
    if not os.path.exists(path):
        return
    report = _load_json(path, None)
    if not isinstance(report, dict):
        return
    report["harness_convergence"] = {
        "stopped": stopped,
        "qa_ok": bool(qa_ok),
        "best_round": convergence.get("best_round"),
        "best_score": convergence.get("best_score"),
        "rolled_back_rounds": convergence.get("rolled_back_rounds"),
        "epsilon": convergence.get("epsilon"),
        "plateau_rounds": convergence.get("plateau_rounds"),
        "trail": convergence.get("trail"),
    }
    try:
        _write_json(path, report)
    except OSError:
        pass


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
