"""Replay of runs/codex-targeted-002a through the harness decision brain.

These tests pin the fixes for the 002 harness-decision audit against the ACTUAL saved
qa/repair artifacts of that run (no GPU needed). Each maps to a numbered audit finding:

  1/4  deterministic measured evidence outranks a medium VLM opinion; worst layer first
  2    an un-actionable plan (only harness.target_id) is rejected at admission
  3    (see test_harness_loop) rejected/no-op repairs are not convergence evidence
  5    the reward gate goes RED via the worst-local floor + hard-fail/contract checks

Plus the critic follow-ups folded in afterwards (GA1 fail-closed gate, GA3 construction
clamp, GB3 admission rejection advancing to the next candidate within one iteration).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, qa_reward

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "codex-targeted-002a", "002_attached_5885519ba4359843")
PHASE2 = {"qa": {"reward": {"mode": "phase2"}}}


def _load(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as handle:
        return json.load(handle)


# ── Finding 2: un-actionable plan rejected at admission ──────────────────────────────

def test_002_target_only_plan_rejected_at_admission():
    # The exact 002 no-op: a VLM layout opinion whose ONLY config change is
    # harness.target_id = c_B5__w0 (no dx/dy, box, font, or measurable delta).
    noop = {
        "stage": "layout", "action": "refit-geometry", "target_id": "c_B5__w0",
        "resume": "layout", "patches": {"harness": {"target_id": "c_B5__w0"}},
    }
    assert harness.plan_is_concrete(noop) is False

    # A plan carrying a real, checkable geometry delta stays admissible.
    concrete = dict(noop, patches={"harness": {"target_id": "c_B5__w0"},
                                   "layout": {"measured_dx": 12, "measured_dy": -4}})
    assert harness.plan_is_concrete(concrete) is True


# ── Findings 1 + 4: deterministic worst-measured evidence outranks VLM opinion ───────

def test_002_ranking_puts_worst_measured_deterministic_first():
    qa = _load("qa.json")
    repairs = _load("repairs.json")
    # The medium VLM opinion the old harness acted on first.
    vlm_opinion = {
        "stage": "layout", "action": "refit-geometry", "target_id": "c_B5__w0",
        "severity": "medium", "reason": "looks misaligned",
        "params": {"source": "vlm_critique"},
    }
    ranked = harness.rank_repairs(repairs + [vlm_opinion], qa)

    top = ranked[0]
    # A HIGH-severity deterministic (measured) repair leads, not the medium VLM opinion.
    assert str(top.get("severity")).lower() == "high"
    assert (top.get("params") or {}).get("source") != "vlm_critique"
    # The medium VLM opinion sinks to the tail (tiebreaker only), never the top slot.
    assert ranked.index(vlm_opinion) > len(ranked) // 2

    # Finding 4: the leading repair steers at the worst MEASURED evidence, not c_B5__w0
    # (which sits at region_ssim 0.31, far from the worst). The worst window is 0.009.
    worst_window = qa["local_ssim_worst_window"]
    regions = (top.get("params") or {}).get("regions") or []
    boxed = [r.get("box") for r in regions if isinstance(r, dict)]
    assert worst_window["bbox"] in boxed


# ── Finding 5: reward gate goes RED on the 002 run ───────────────────────────────────

def test_002_reward_gate_is_red():
    qa = _load("qa.json")
    reward = qa_reward.compute_reward("", PHASE2, qa=qa)
    gate = qa_reward.acceptance_gate("", PHASE2, qa=qa, reward=reward)

    assert gate["ok"] is False
    checks = gate["checks"]
    # The giant low-confidence raster cannot buy green: the worst-local floor fails even
    # though aggregate local SSIM would clear the mean gate.
    assert checks["worst_local_ssim"]["ok"] is False
    assert checks["worst_local_ssim"]["value"] < checks["worst_local_ssim"]["min"]
    # Standing hard failures and a failed construction contract each force RED.
    assert checks["hard_fails"]["ok"] is False
    assert checks["hard_fails"]["value"] == 4
    assert checks["contract"]["ok"] is False


# ── Critic GA1: the gate fails CLOSED, never open, on an evaluation error ─────────────

def test_gate_fails_closed_on_evaluation_error():
    class Exploding(dict):
        def get(self, *a, **k):  # any access inside the gate raises
            raise RuntimeError("boom")

    reward = {"components": Exploding(local_ssim={}), "hard_fails": 0}
    gate = qa_reward.acceptance_gate("", PHASE2, qa={}, reward=reward)
    assert gate["ok"] is False
    assert gate.get("skipped", "").startswith("gate_error:")


# ── Critic GA3: a failed contract cannot buy a top construction score ────────────────

def test_construction_clamped_when_contract_fails():
    # Erased-imagery round: high raw native_text_ratio, but the contract explicitly failed.
    qa = {"contract": {"pass": False, "native_text_ratio": 0.95}}
    detail = qa_reward.construction_component(qa)
    assert detail is not None
    assert detail["source"] == "native_text_ratio"
    assert detail["score"] <= qa_reward._CONTRACT_FAIL_CONSTRUCTION_CEIL
    assert detail["clamped_contract_fail"] is True

    # A passing contract with the same ratio is NOT clamped.
    ok = qa_reward.construction_component({"contract": {"pass": True, "native_text_ratio": 0.95}})
    assert ok["score"] > qa_reward._CONTRACT_FAIL_CONSTRUCTION_CEIL


# ── Critic GB3: an admission rejection advances to the next candidate, same iteration ─

def test_admission_rejection_tries_next_candidate_within_iteration(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    # Top-ranked repair is a non-concrete no-op (target-id only); a concrete OCR repair
    # follows. With repair_iterations=1 the old loop burned the iteration on the rejected
    # no-op and never ran the concrete repair.
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "layout", "action": "refit-geometry", "target_id": "c_X",
         "severity": "high", "reason": "vague no-op"},
        {"stage": "ocr", "action": "rerun", "severity": "high", "reason": "text low",
         "params": {"upscale": True}},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": False, "repairs": []}), encoding="utf-8")

    calls = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        calls.append(start_from)
        (run_dir / "qa.json").write_text(
            json.dumps({"ok": True, "repairs": []}), encoding="utf-8")
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(
        str(run_dir), {}, max_iterations=1, run_one=fake_run_one)

    # The concrete OCR repair actually ran despite the no-op consuming the top slot.
    assert calls == ["ocr"]
    kinds = [a for a in summary["attempts"]]
    assert any(a.get("admission_rejected") for a in kinds)
    assert any(a.get("repair", {}).get("action") == "rerun" and a.get("qa_fresh")
               for a in kinds)


def test_admission_only_round_consumes_no_iteration_budget(tmp_path):
    # Finding 3: a round whose ONLY candidate is rejected at admission must not be counted
    # as an evaluated iteration (no real rerun happened). The pipeline runner must never be
    # invoked, and the summary reports zero iterations so plateau logic sees no evidence.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "repairs.json").write_text(json.dumps([
        {"stage": "layout", "action": "refit-geometry", "target_id": "c_X",
         "severity": "high", "reason": "vague no-op"},
    ]), encoding="utf-8")
    (run_dir / "qa.json").write_text(
        json.dumps({"ok": False, "repairs": []}), encoding="utf-8")

    ran = []

    def fake_run_one(path, rd, cfg, start_from="normalize"):
        ran.append(start_from)
        return {"ok": True, "run_dir": rd}

    summary = harness.execute_repairs(
        str(run_dir), {}, max_iterations=1, run_one=fake_run_one)

    assert ran == []                       # no pipeline rerun spent
    assert summary["iterations"] == 0      # the rejected no-op did not count as a round
    assert summary["stopped"] in {"no_actionable_repairs", "all_repairs_failed"}
    assert any(a.get("admission_rejected") for a in summary["attempts"])
