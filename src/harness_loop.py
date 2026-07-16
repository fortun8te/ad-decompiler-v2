"""harness_loop.py — failure-proof orchestrator for QA repair rounds.

Flow per round:
  run_pipeline.run_one → if QA not ok → execute_repairs → critic → fixer → repeat

Never lowers QA thresholds. Writes ``harness_loop.json`` with a full audit trail.

Reward (``runtime.harness.reward``): ``phase2`` (default) scores each round with the
src.qa_reward metric ladder — per-element local SSIM + LPIPS perceptual + text recall —
and uses one capped VLM critique per round as the primary repair driver; ``legacy``
keeps the old composite/mean-of-metrics score. Acceptance hard-fail semantics are
identical in both modes: structural fails can never be bought by a good score, and the
phase2 gate is strictly additional (it can only refuse, never grant, acceptance).
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
    load_repair_candidates,
    rank_repairs,
    recommended_resume,
    _load_json,
    _write_json,
)
from src import qa_reward
from src.qa_config import visual_pass_ssim

# Artifacts that together define "the design produced this round". Snapshotting these lets
# the loop keep the best-scoring design and roll back a regressing round.
#
# This must cover every artifact a report treats as authoritative per-round evidence, not
# just the ones the reward math itself reads. benchmark.py's row builder reads
# reconstruction.json directly (vectorized/backend/route counts, archetype/preset
# fallback), and runtime_report.json embeds its own qa_evidence/qa_ok/stage-detail mirror
# of qa.json. A repair round that resumes from "text" (or earlier) regenerates all of
# these, so leaving them out of the snapshot let a rollback restore qa.json/design.json to
# the best round while reconstruction.json/fallback.json/runtime_report.json kept whatever
# the last (regressed) round's pipeline run had written -- a "mixed rounds" state where the
# shipped qa.json disagreed with the shipped runtime_report.json (observed in
# runs/benchmark-final/016_attached_ac1eeeabce759396: qa.json restored to the round-0 ssim
# 0.874 while runtime_report.json.qa_evidence still showed round-1's ssim 0.7586 / "editable
# text recall 0.33 < 0.88").
# GB4: snapshot the metadata AND the pixel assets that reconstruction.json / design.json
# reference. Restoring only the JSON on a rollback used to leave a regressed round's bad
# pixels (a botched inpaint / removal mask) on disk under rolled-back metadata, so the
# reverted "best" design shipped with the worse round's background. Keep raster outputs in
# lockstep with the metadata that points at them.
_SNAPSHOT_FILES = (
    "design.json", "qa.json", "preview.png", "layout.json", "figma_export.png",
    "reconstruction.json", "fallback.json", "runtime_report.json",
    "background_clean.png", "removal_mask.png", "ownership.png",
    "removal_ownership.png", "normalized.png", "layers_contact.png",
)

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
    """Legacy scalar round quality from existing QA fields (composite / metric mean)."""
    qa = qa or {}
    if _qa_accepts(qa, allow_summary=True):
        return 1.0
    composite = qa.get("composite")
    if isinstance(composite, (int, float)):
        base = float(composite) / 100.0
    else:
        values = [float(qa[key]) for key in _SCORE_KEYS if isinstance(qa.get(key), (int, float))]
        if not values:
            return None
        base = sum(values) / len(values)
    hard_fails = qa.get("hard_fails")
    penalty = 0.12 * len(hard_fails) if isinstance(hard_fails, list) else 0.0
    return round(max(0.0, base - penalty), 6)


def _score_round(run_dir: str, cfg: Optional[dict], qa: Optional[dict]) -> tuple[Optional[float], Optional[dict]]:
    """Round score for best-kept/rollback/plateau: phase2 metric ladder, else legacy.

    Returns ``(score, reward)``; ``reward`` is the full qa_reward record when the
    phase2 ladder produced the score, ``None`` otherwise. Accepted QA scores 1.0 in
    both modes (acceptance is decided by ``_qa_accepts``, never by the reward).
    """
    qa = qa or {}
    if _qa_accepts(qa, allow_summary=True):
        return 1.0, None
    if qa_reward.reward_mode(cfg) == "phase2":
        try:
            reward = qa_reward.compute_reward(run_dir, cfg, qa=qa)
            score = reward.get("score")
            if isinstance(score, (int, float)):
                return float(score), reward
        except Exception:
            pass
    return _qa_score(qa), None


def _reward_gate(run_dir: str, cfg: Optional[dict], qa: Optional[dict]) -> dict:
    """Anti-degenerate acceptance gate (phase2 only). Fails CLOSED on error.

    A gate that cannot evaluate must not grant acceptance (critic A GA1); a swallowed
    exception here used to default ok:True and silently flip a degenerate round green.
    """
    try:
        return qa_reward.acceptance_gate(run_dir, cfg, qa=qa)
    except Exception as exc:
        return {"ok": False, "skipped": f"gate_error:{type(exc).__name__}", "error": True}


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


def _round_made_progress(round_record: dict) -> bool:
    """True if a round moved QA metrics or applied a fixer patch (F13c plateau input).

    Used only when no scalar reward score is available (legacy mode or a run with no QA
    metrics); the reward-delta path handles the phase2 case. A round that neither improved
    metrics nor applied a fix is a no-op and should push the loop toward a plateau stop.
    """
    progress = (round_record or {}).get("repair_progress") or {}
    if progress.get("improved"):
        return True
    if (round_record.get("fixer") or {}).get("fixes"):
        return True
    return False


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
    for repair in load_repair_candidates(run_dir, cfg) or []:
        if not is_actionable(repair):
            continue
        signature = _repair_id(repair)
        if signature in blocked or signature in out:
            continue
        out.append(signature)
    return out


def max_harness_rounds(cfg: Optional[dict] = None, default: int = 2) -> int:
    rounds = harness_max_rounds(cfg or {})
    return max(1, rounds if rounds else default)


def repair_iterations(cfg: Optional[dict] = None, default: int = 1) -> int:
    harness = ((cfg or {}).get("runtime") or {}).get("harness") or {}
    return max(1, int(harness.get("repair_iterations", default)))


def in_harness_loop(cfg: Optional[dict] = None) -> bool:
    return bool((((cfg or {}).get("runtime") or {}).get("harness") or {}).get("_in_loop"))


def _threshold_snapshot(cfg: dict) -> dict:
    return {
        "visual_pass_ssim": visual_pass_ssim(cfg),
        "repair": copy.deepcopy(cfg.get("repair") or {}),
        "reward_gate": dict(qa_reward.gate_thresholds(cfg)),
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
    current_gate = qa_reward.gate_thresholds(cfg)
    for key, baseline in (snapshot.get("reward_gate") or {}).items():
        current = current_gate.get(key)
        if current is not None and float(current) < float(baseline):
            raise ValueError(f"harness must not lower reward gate threshold {key}")


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


def _apply_vlm_critique(run_dir: str, cfg: dict, critic_output: dict) -> dict:
    """Fold the phase2 VLM critique into the critic output as a TIEBREAKER repair source.

    Critique repairs (already mapped onto the existing repair-action vocabulary by
    qa_reward.critique_to_repairs) are merged AFTER the metric/tool repairs and the
    combined list is re-ranked by ``harness.rank_repairs``: deterministic, measured
    evidence (per-layer SSIM/ink IoU, hard failures) outranks VLM opinions at equal
    severity, and the worst measured layer sorts first. (Previously critique repairs
    were prepended as the "primary driver", which let a vague medium VLM opinion outrank
    HIGH measured failures — the 002 no-op.) Failure-proof: any error leaves the critic
    output untouched.
    """
    try:
        if qa_reward.reward_mode(cfg) != "phase2" or not qa_reward.critique_enabled(cfg):
            return critic_output
        critique = qa_reward.run_critique(run_dir, cfg)
        items = critique.get("items") or []
        critique_error = critique.get("error")
        # GB2: a VLM timeout / transport error (vlm_client default 20s, nondeterministic
        # under GPU contention) must NOT read as "the VLM inspected the render and found
        # nothing". Empty-with-an-error is INCONCLUSIVE; empty-without-error is a clean
        # opinion. Surface the distinction so downstream never treats a flickering timeout
        # as convergence evidence.
        if items:
            status = "ok"
        elif critique_error:
            status = "error"
        else:
            status = "empty"
        critic_output["vlm_critique"] = {
            "items": items,
            "count": len(items),
            "model": critique.get("model"),
            "error": critique_error,
            "status": status,
            "inconclusive": status == "error",
        }
        if not items:
            return critic_output
        design = _load_json(os.path.join(run_dir, "design.json"), None)
        critique_repairs = qa_reward.critique_to_repairs(
            items, design if isinstance(design, dict) else None)
        if not critique_repairs:
            return critic_output
        merged: list = []
        seen: set[tuple] = set()
        admission_rejected = []
        for repair in list(critic_output.get("filtered_repairs") or []) + critique_repairs:
            params = repair.get("params") or {}
            # A crop critique that already resolves to concrete layer IDs is a local
            # reconstruction/text problem, not evidence that the whole SAM proposal
            # stage missed an object. Broad element-propose here caused dozens of nearly
            # identical full-card peel masks for 002's one missing arrow.
            if (repair.get("stage") == "sam3"
                    and repair.get("action") == "rerun-detection"
                    and params.get("source") == "vlm_critique"
                    and params.get("layer_ids")):
                admission_rejected.append({
                    "repair": repair,
                    "reason": "localized-vlm-issue-must-not-trigger-broad-redetection",
                })
                continue
            if (repair.get("stage") == "layout"
                    and repair.get("action") == "refit-geometry"
                    and params.get("source") == "vlm_critique"
                    and not any(key in params for key in (
                        "measured_dx", "measured_dy", "min_container_frac",
                        "max_container_frac", "tighten_containers",
                    ))):
                admission_rejected.append({
                    "repair": repair,
                    "reason": "vlm-layout-opinion-has-no-measured-geometry-or-patch",
                })
                continue
            signature = (repair.get("stage"), repair.get("action"), repair.get("target_id"))
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(repair)
        qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        critic_output["filtered_repairs"] = rank_repairs(merged, qa)
        if admission_rejected:
            critic_output.setdefault("admission_control", {})["rejected"] = admission_rejected
    except Exception:
        pass
    return critic_output


def _fixer_proposals(critic_output: dict, fixes: list) -> list[dict]:
    """Rank applied fixer ids against the critic's prioritized issues (F10 honesty).

    Every proposal is concrete (it names the config key it changed) and starts
    ``unvalidated`` — fixer.json must never present untested config changes as fixes."""
    prefix_to_category = (
        ("inpaint", "inpaint"), ("force-lama", "inpaint"),
        ("ocr", "ocr"), ("boost-ocr", "ocr"),
        ("vlm", "sam"), ("boost-vlm", "sam"),
        ("layout", "layout"), ("tighten-containers", "layout"),
        ("restage", "staging"), ("figma", "staging"), ("staging", "staging"),
    )
    issue_rank: dict[str, tuple] = {}
    issues = (critic_output or {}).get("prioritized_issues") or (critic_output or {}).get("issues") or []
    for position, issue in enumerate(issues):
        category = str((issue or {}).get("category") or "").lower()
        if category and category not in issue_rank:
            issue_rank[category] = (position, issue.get("severity"))

    records = []
    for index, fix in enumerate(fixes or []):
        fid = str(fix)
        category = next((cat for prefix, cat in prefix_to_category
                         if fid.startswith(prefix)), "staging")
        position, severity = issue_rank.get(category, (len(issues), None))
        records.append({
            "fix": fid,
            "category": category,
            "severity": severity,
            "status": "unvalidated",
            "_sort": (position, index),
        })
    records.sort(key=lambda record: record.pop("_sort"))
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def _patch_fixer_validation(run_dir: str, fixer_round: Optional[int],
                            convergence_trail: list[dict], epsilon: float) -> None:
    """After the loop, mark fixer.json proposals validated only if a LATER round improved."""
    path = os.path.join(run_dir, "fixer.json")
    if fixer_round is None or not os.path.exists(path):
        return
    report = _load_json(path, None)
    if not isinstance(report, dict):
        return
    validated = any(
        isinstance(record, dict)
        and (record.get("round") or 0) > fixer_round
        and isinstance(record.get("delta"), (int, float))
        and record["delta"] > epsilon
        for record in convergence_trail or []
    )
    report["validated"] = bool(validated)
    for proposal in report.get("proposals") or []:
        if isinstance(proposal, dict):
            proposal["status"] = "validated" if validated else "unvalidated"
    if not validated:
        report["note"] = ("config proposals were applied but no later round improved the "
                          "reward — they remain unvalidated, not fixes")
    try:
        _write_json(path, report)
    except OSError:
        pass


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
            round_record["pipeline"]["qa_stale"] = True
        elif _qa_accepts(qa, allow_summary=True) and qa_fresh:
            gate = _reward_gate(run_dir, working_cfg, qa)
            round_record["reward_gate"] = gate
            if gate.get("ok", True):
                round_record["stopped"] = "qa_ok"
                return round_record, working_cfg, True
            # Anti-degenerate: metric QA passed but the perceptual/per-element gate did
            # not — keep repairing instead of accepting a bought-looking score.

    qa_before_repairs = _load_json(os.path.join(run_dir, "qa.json"), {})
    # Run critic before repairs so execute_repairs can use filtered_repairs this round.
    # The phase2 VLM critique is folded in afterwards as the primary repair driver.
    critic_output = _run_critic_pass(run_dir, working_cfg)
    critic_output = _apply_vlm_critique(run_dir, working_cfg, critic_output)
    _write_json(os.path.join(run_dir, "critic.json"), critic_output)
    round_record["critic"] = {
        "issues": len(critic_output.get("prioritized_issues") or critic_output.get("issues") or []),
        "suggested_fix_ids": critic_output.get("suggested_fix_ids") or [],
        "blockers": len(critic_output.get("blockers") or []),
    }
    if critic_output.get("vlm_critique"):
        round_record["critic"]["vlm_critique"] = critic_output["vlm_critique"].get("count", 0)
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
    pipeline_stale = bool((round_record.get("pipeline") or {}).get("qa_stale"))
    if _qa_accepts(qa, allow_summary=True) and not pipeline_stale:
        gate = _reward_gate(run_dir, working_cfg, qa)
        round_record["reward_gate"] = gate
        if gate.get("ok", True):
            round_record["stopped"] = "qa_ok_after_repairs"
            return round_record, working_cfg, True

    fixer_result = _run_fixer_pass(run_dir, working_cfg, critic_output)
    fixer_fixes = fixer_result.get("fixes") or []
    _write_json(os.path.join(run_dir, "fixer.json"), {
        "fixes": fixer_fixes,
        "fix_count": len(fixer_fixes),
        # Honesty contract: these are config PROPOSALS applied to the next round's
        # config; none is a proven fix until a rerun improves the reward. The loop
        # patches ``validated`` after observing the following round.
        "proposals": _fixer_proposals(critic_output, fixer_fixes),
        "validated": False,
        "note": ("ranked config proposals for the next round; unvalidated until a "
                 "rerun demonstrates an improved reward"),
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
    max_rounds: int = 2,
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
    before_score, _ = _score_round(run_dir, cfg, _load_json(os.path.join(run_dir, "qa.json"), {}))
    best_score = before_score if before_score is not None else float("-inf")
    best_round = 0
    best_snapshot = _snapshot_artifacts(run_dir) if before_score is not None else {}
    plateau = 0
    rolled_back_rounds = 0
    last_fixer_round: Optional[int] = None
    # Which round's artifacts are actually on disk right now (0 = the pre-loop pipeline
    # result). This is the single source of truth the final "emit best design" step and
    # the harness reports below use to say what shipped -- distinct from ``best_round``,
    # which only advances on a *strict* score improvement and can otherwise lag behind a
    # tied/no-op round that was legitimately left on disk (see the ``abs(delta) <= epsilon``
    # branch below).
    disk_round = 0

    next_resume = start_from
    last_repair_refreshed_qa = False
    for round_num in range(1, max_rounds + 1):
        skip_pipeline = (
            (pipeline_already_ran and round_num == 1)
            or (round_num > 1 and last_repair_refreshed_qa)
        )
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
        attempts = (round_record.get("repairs") or {}).get("attempts") or []
        last_repair_refreshed_qa = any(attempt.get("qa_fresh") for attempt in attempts)
        # A round only counts as convergence EVIDENCE when something was actually
        # evaluated: an admitted repair executed, or the pipeline wrote fresh QA. Rounds
        # whose every candidate was rejected/skipped at admission are not proof the run
        # converged — counting them caused the observed premature plateau (002: one
        # rejected no-op ended the loop before any high-severity repair was tried).
        round_evaluated = bool(
            [a for a in attempts
             if not a.get("admission_skipped") and not a.get("admission_rejected")]
        ) or bool((round_record.get("pipeline") or {}).get("qa_fresh"))
        if (round_record.get("fixer") or {}).get("fixes"):
            last_fixer_round = round_num
        if round_stopped not in {"pipeline_exception", "pipeline_failed"}:
            # The pipeline and/or repairs actually ran and wrote fresh artifacts this
            # round; it is what's on disk until the scoring below proves otherwise. The
            # two exceptions above return from _run_round before touching run_dir at all.
            disk_round = round_num

        # ── convergence bookkeeping ────────────────────────────────────────────────
        after_qa = _load_json(os.path.join(run_dir, "qa.json"), {})
        after_score, after_reward = _score_round(run_dir, working_cfg, after_qa)
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
            # None-safe sort: signatures may carry None parts (e.g. critique repairs
            # without a stage/param), and tuple comparison raises on None < str.
            "blocked": [list(sig) for sig in sorted(
                non_improving, key=lambda sig: tuple(str(part) for part in sig))],
        }
        reward_record = qa_reward.reward_evidence(after_reward)
        if reward_record:
            record["reward"] = reward_record

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
                disk_round = best_round
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
                # No-op repair — never retry it (F13a admission control: a repair whose
                # observed reward delta is ~0 must not re-run a full pipeline next round).
                blocked |= set(applied)
            # A rolled-back round left the design at the previous best: net progress is
            # zero, so it counts toward the plateau. This is what makes a reward that
            # flips between two states (the observed 0.87/0.5 ↔ 0.37/0.79 oscillation)
            # plateau-stop instead of bouncing until max_rounds.
            #
            # Rejected/admission-skipped repairs are NOT convergence evidence: a round
            # that evaluated nothing (every candidate rejected before a rerun) leaves the
            # plateau counter untouched so the loop moves on to the remaining repairs
            # instead of "giving up" on the strength of a no-op it refused to run.
            record["evaluated"] = round_evaluated
            if record["rolled_back"] or (abs(delta) < epsilon and round_evaluated):
                plateau += 1
            elif abs(delta) >= epsilon:
                plateau = 0
            before_score = after_score
        elif round_evaluated and not _round_made_progress(round_record):
            # F13c: even with no scalar reward (legacy / metrics unavailable), a round that
            # neither improved QA metrics nor applied a fix is a no-op — count it toward the
            # plateau and block its repairs so the loop stops instead of chasing OCR noise.
            blocked |= set(applied)
            plateau += 1
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
    live_score, _ = _score_round(run_dir, working_cfg, final_qa)
    if best_snapshot and best_score != float("-inf") and (
        live_score is None or live_score < best_score
    ):
        _restore_artifacts(run_dir, best_snapshot)
        disk_round = best_round
        final_qa = _load_json(os.path.join(run_dir, "qa.json"), {})

    # Final reward + gate evidence. The gate is strictly additional: it can only turn a
    # metric-accepted result into "not ok" (anti-degenerate), never the other way round.
    reward_mode = qa_reward.reward_mode(cfg)
    final_reward = None
    final_gate: dict = {"ok": True, "skipped": "legacy"}
    if reward_mode == "phase2":
        final_reward = qa_reward.compute_reward(run_dir, working_cfg, qa=final_qa)
        final_gate = _reward_gate(run_dir, working_cfg, final_qa)
    qa_ok = _qa_accepts(final_qa, allow_summary=True) and bool(final_gate.get("ok", True))

    summary = {
        "run_dir": run_dir,
        "rounds": rounds,
        "rounds_completed": len(rounds),
        "max_rounds": max_rounds,
        "qa_ok": qa_ok,
        "stopped": stopped,
        # Which round's artifacts are provably on disk right now (0 = pre-loop pipeline
        # result). Recorded explicitly so a reader never has to infer "what shipped" from
        # best_round/rolled_back_rounds alone -- see the _SNAPSHOT_FILES note above.
        "shipped_round": disk_round,
        "thresholds": threshold_snapshot,
        "reward": {
            "mode": reward_mode,
            "final": qa_reward.reward_evidence(final_reward),
            "gate": final_gate,
        },
        "convergence": {
            "epsilon": epsilon,
            "plateau_rounds": plateau_limit,
            "best_round": best_round,
            "best_score": None if best_score == float("-inf") else round(best_score, 6),
            "rolled_back_rounds": rolled_back_rounds,
            "shipped_round": disk_round,
            "trail": convergence,
        },
    }
    _write_json(os.path.join(run_dir, "harness_loop.json"), summary)
    _patch_fixer_validation(run_dir, last_fixer_round, convergence, epsilon)
    _patch_harness_report(run_dir, summary["reward"], shipped_round=summary["shipped_round"])
    _patch_runtime_report(run_dir, summary["convergence"], stopped, summary["qa_ok"],
                          reward=summary["reward"])
    return summary


def _patch_harness_report(run_dir: str, reward: dict, shipped_round: Optional[int] = None) -> None:
    """Attach the final reward evidence and shipped round to harness.json.

    harness.json (written by execute_repairs) is a per-attempt audit trail and is
    intentionally left as whatever the last repair attempt wrote -- it documents what was
    *tried*, including attempts that were later rolled back. ``shipped_round`` answers the
    separate question "which round's artifacts are actually on disk", so a reader does not
    have to infer that from the (possibly rolled-back) attempt list.
    """
    path = os.path.join(run_dir, "harness.json")
    if not os.path.exists(path):
        return
    report = _load_json(path, None)
    if not isinstance(report, dict):
        return
    report["reward"] = reward
    if shipped_round is not None:
        report["shipped_round"] = shipped_round
    try:
        _write_json(path, report)
    except OSError:
        pass


def _patch_runtime_report(run_dir: str, convergence: dict, stopped: str, qa_ok: bool,
                          reward: Optional[dict] = None) -> None:
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
        "shipped_round": convergence.get("shipped_round"),
        "epsilon": convergence.get("epsilon"),
        "plateau_rounds": convergence.get("plateau_rounds"),
        "trail": convergence.get("trail"),
    }
    if reward is not None:
        report["harness_convergence"]["reward"] = reward
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
