"""Phase-2 harness wiring tests: metric-ladder reward, gate, critique repair driver.

CPU-only, no VLM, no LPIPS model — qa_reward internals are monkeypatched where a real
model would be needed; everything else exercises the real loop code.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness_loop, qa_reward


def _stub_critic(monkeypatch, filtered=None):
    monkeypatch.setattr(
        harness_loop, "_run_critic_pass",
        lambda rd, cfg: {"prioritized_issues": [], "suggested_fix_ids": [],
                         "blockers": [], "filtered_repairs": list(filtered or [])})


def _stub_fixer(monkeypatch, fixes=("keep-going",)):
    monkeypatch.setattr(harness_loop, "_run_fixer_pass",
                        lambda rd, cfg, c: {"cfg": cfg, "fixes": list(fixes)})


def _seed(tmp_path, qa):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    (run_dir / "runtime_report.json").write_text(
        json.dumps({"input": str(input_path)}), encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps({"layers": []}), encoding="utf-8")
    (run_dir / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    return str(input_path), str(run_dir)


def _write_qa(run_dir, qa):
    with open(os.path.join(run_dir, "qa.json"), "w", encoding="utf-8") as fh:
        json.dump(qa, fh)


def _repairing_exec(sig=("ocr", "rerun", None)):
    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": [
            {"repair": {"stage": sig[0], "action": sig[1], "target_id": sig[2]},
             "qa_improved": False, "pipeline_ok": True}]}
    return exec_repairs


# ── oscillation: a reward flipping between two states must plateau-stop ───────────────

def test_oscillating_reward_plateau_stops_and_keeps_best(tmp_path, monkeypatch):
    """The observed ad9 case: rounds alternate ssim 0.87/text 0.5 ↔ 0.37/0.79.

    Round 1 regresses → rollback (counts toward plateau); round 2 lands back on the
    best score → zero delta. plateau_rounds=2 ends the loop right there instead of
    bouncing until max_rounds, and the BEST artifacts are what remains on disk.
    """
    state_a = {"ok": False, "ssim": 0.87, "text_recall": 0.5, "hard_fails": [],
               "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high"}]}
    state_b = {"ok": False, "ssim": 0.37, "text_recall": 0.79, "hard_fails": [],
               "repairs": [{"stage": "ocr", "action": "rerun", "severity": "high"}]}
    input_path, run_dir = _seed(tmp_path, state_a)
    flips = {"n": 0}

    def run_one(path, rd, cfg, start_from="normalize"):
        flips["n"] += 1
        _write_qa(rd, state_b if flips["n"] % 2 else state_a)
        return {"ok": True}

    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch)

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=6,
        run_one=run_one, execute_repairs_fn=_repairing_exec())

    assert summary["stopped"] == "plateau"
    assert summary["rounds_completed"] < 6
    trail = summary["convergence"]["trail"]
    assert any(entry["rolled_back"] for entry in trail)
    # Best design (state A, the higher phase2 score) is what the run dir holds.
    final_qa = json.loads((tmp_path / "run" / "qa.json").read_text(encoding="utf-8"))
    assert final_qa["ssim"] == 0.87
    # Per-round reward evidence is in the trail.
    assert any("reward" in entry for entry in trail)


def test_noop_repair_is_blocked_and_not_rerun_next_round(tmp_path, monkeypatch):
    """F13a admission control: a repair whose round produced no QA/reward gain is added to
    the blocked set and threaded into the NEXT execute_repairs call as blocked_repairs, so
    it is not re-executed (no wasted full-pipeline pass on a known no-op)."""
    qa = {"ok": False, "ssim": 0.5, "text_recall": 0.5, "hard_fails": [], "repairs": []}
    input_path, run_dir = _seed(tmp_path, qa)
    sig = ("ocr", "rerun", None)
    received: list[set] = []

    def exec_repairs(rd, cfg, max_iterations=1, run_one=None, blocked_repairs=None):
        blocked = set(blocked_repairs or ())
        received.append(blocked)
        if sig in blocked:  # honor admission control the way real execute_repairs does
            return {"stopped": "no_actionable_repairs", "qa_ok": False,
                    "iterations": 0, "attempts": []}
        return {"stopped": "max_iterations", "qa_ok": False, "iterations": 1, "attempts": [
            {"repair": {"stage": sig[0], "action": sig[1], "target_id": sig[2]},
             "qa_improved": False, "pipeline_ok": True}]}

    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=("keep-going",))   # keep the loop advancing past round 1

    def run_one(path, rd, c, start_from="normalize"):
        _write_qa(rd, qa)                              # nothing improves, ever
        return {"ok": True}

    # Need plateau_rounds≥2 to observe the blocked handoff into round 2; production
    # default is 1 (workstream E / config.example).
    cfg = {"runtime": {"harness": {"plateau_rounds": 2}}}
    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, cfg, max_rounds=3,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    assert len(received) >= 2
    assert sig not in received[0]                       # round 1 is free to try it
    assert sig in received[1]                           # round 2 has it blocked
    # and round 2 did NOT re-run it — the executor short-circuited on the blocked signature
    assert summary["rounds"][1]["repairs"]["stopped"] == "no_actionable_repairs"


def test_plateau_stops_well_within_budget_on_no_progress(tmp_path, monkeypatch):
    """F13c: a run that never improves must stop fast, not burn the whole round budget."""
    qa = {"ok": False, "ssim": 0.5, "text_recall": 0.5, "hard_fails": [], "repairs": []}
    input_path, run_dir = _seed(tmp_path, qa)
    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=())

    def run_one(path, rd, c, start_from="normalize"):
        _write_qa(rd, qa)
        return {"ok": True}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=8,
        run_one=run_one, execute_repairs_fn=_repairing_exec(("ocr", "rerun", None)))

    assert summary["stopped"] in {"plateau", "no_progress"}
    assert summary["rounds_completed"] <= 2            # not 8


def test_config_budget_reverted_to_example_values():
    """F13b: live config.yaml harness budget must be the example's 2/1/1, not 3/2/2."""
    import os
    import yaml

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    harness = cfg["runtime"]["harness"]
    assert harness["max_rounds"] == 2
    assert harness["repair_iterations"] == 1
    assert harness["plateau_rounds"] == 1


def test_phase2_score_comes_from_reward_ladder_not_composite(tmp_path):
    qa = {"ok": False, "hard_fails": [],
          "composite": 10.0,                      # legacy would score 0.10
          "ssim": 0.8, "text_recall": 0.6,
          "per_layer": [{"id": "a", "region_ssim": 0.9, "region_px": 100}]}
    score, reward = harness_loop._score_round(str(tmp_path), {}, qa)
    assert reward is not None and reward["mode"] == "phase2"
    assert score == reward["score"]
    assert score != pytest.approx(0.10)
    assert reward["components"]["local_ssim"]["source"] == "per_layer"

    legacy_cfg = {"runtime": {"harness": {"reward": "legacy"}}}
    legacy_score, legacy_reward = harness_loop._score_round(str(tmp_path), legacy_cfg, qa)
    assert legacy_reward is None
    assert legacy_score == pytest.approx(0.10)


# ── acceptance: hard fails cannot be bought; the gate can only refuse ─────────────────

def test_hard_fails_cannot_be_bought_by_a_high_reward(tmp_path, monkeypatch):
    qa = {"ok": False, "ssim": 0.99, "text_recall": 0.99,
          "hard_fails": [{"rule": "missing-assets", "detail": "1 unresolved"}],
          "repairs": []}
    input_path, run_dir = _seed(tmp_path, qa)

    def run_one(path, rd, cfg, start_from="normalize"):
        _write_qa(rd, qa)
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        return {"stopped": "no_actionable_repairs", "qa_ok": False, "iterations": 0,
                "attempts": []}

    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=())

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    assert summary["qa_ok"] is False
    assert summary["stopped"] not in {"qa_ok", "qa_ok_after_repairs"}
    # The reward itself was high — structure still wins.
    best = summary["convergence"]["best_score"]
    assert best is not None and best > 0.8


def test_reward_gate_blocks_acceptance_on_degenerate_lpips(tmp_path, monkeypatch):
    """qa.ok true but LPIPS says the render is perceptually unrelated → do not accept."""
    passing = {"ok": True, "ssim": 0.95, "text_recall": 1.0, "hard_fails": [], "repairs": []}
    input_path, run_dir = _seed(tmp_path, dict(passing, ok=False))
    (tmp_path / "run" / "normalized.png").write_bytes(b"png")
    (tmp_path / "run" / "preview.png").write_bytes(b"png")

    monkeypatch.setattr(qa_reward, "lpips_score",
                        lambda *a, **k: {"distance": 0.95, "similarity": 0.05,
                                         "net": "squeeze", "max_edge": 256})

    def run_one(path, rd, cfg, start_from="normalize"):
        _write_qa(rd, passing)
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        return {"stopped": "no_actionable_repairs", "qa_ok": False, "iterations": 0,
                "attempts": []}

    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=())

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    assert summary["stopped"] not in {"qa_ok", "qa_ok_after_repairs"}
    assert summary["qa_ok"] is False
    assert summary["reward"]["gate"]["ok"] is False
    assert summary["reward"]["gate"]["checks"]["lpips_similarity"]["ok"] is False
    gate_records = [r.get("reward_gate") for r in summary["rounds"] if r.get("reward_gate")]
    assert gate_records and gate_records[0]["ok"] is False


def test_gate_passes_where_lpips_unavailable_keeps_legacy_behaviour(tmp_path, monkeypatch):
    """No images / no lpips → the gate must not change acceptance at all."""
    input_path, run_dir = _seed(tmp_path, {"ok": False, "ssim": 0.5, "hard_fails": [],
                                           "repairs": []})

    def run_one(path, rd, cfg, start_from="normalize"):
        _write_qa(rd, {"ok": True, "ssim": 0.95, "hard_fails": [], "repairs": []})
        return {"ok": True}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=2,
        run_one=run_one, execute_repairs_fn=lambda *a, **k: {"attempts": []})

    assert summary["stopped"] == "qa_ok"
    assert summary["qa_ok"] is True
    assert summary["reward"]["gate"]["ok"] is True


# ── critique → tiebreaker, never the primary driver (finding 1) ──────────────────────

def test_deterministic_high_severity_outranks_vlm_critique(tmp_path, monkeypatch):
    # Finding 1: a HIGH-severity deterministic (metric) repair must stay ahead of a VLM
    # critique opinion. The old harness PREPENDED critique as the "primary driver", which
    # let a vague medium VLM opinion outrank measured HIGH failures (the 002 no-op).
    input_path, run_dir = _seed(tmp_path, {"ok": False, "ssim": 0.5, "hard_fails": [],
                                           "repairs": []})
    cfg = {"qa": {"reward": {"critique": {"enabled": True}}}}
    monkeypatch.setattr(qa_reward, "run_critique", lambda rd, c, **k: {
        "items": [{"element": "SHOP NOW", "issue": "text clipped at edge",
                   "suggested_fix": "widen box", "layer_ids": ["c_B18"]}],
        "model": "google/gemma-4-e4b"})

    critic_output = {"prioritized_issues": [], "suggested_fix_ids": [], "blockers": [],
                     "filtered_repairs": [{"stage": "ocr", "action": "rerun",
                                           "severity": "high", "reason": "metric"}]}
    merged = harness_loop._apply_vlm_critique(run_dir, cfg, dict(critic_output))

    assert merged["vlm_critique"]["count"] == 1
    # Deterministic HIGH metric repair wins the top slot; the VLM critique is a tiebreaker.
    first = merged["filtered_repairs"][0]
    assert (first["stage"], first["action"]) == ("ocr", "rerun")
    # The critique repair is still present (VLM contributes, but never as sole authority).
    assert any(r.get("params", {}).get("source") == "vlm_critique"
               for r in merged["filtered_repairs"])


def test_vlm_critique_status_distinguishes_timeout_from_empty(tmp_path, monkeypatch):
    # GB2: a VLM timeout/transport error must not read as "inspected and found nothing".
    input_path, run_dir = _seed(tmp_path, {"ok": False, "repairs": []})
    cfg = {"qa": {"reward": {"critique": {"enabled": True}}}}

    monkeypatch.setattr(qa_reward, "run_critique",
                        lambda rd, c, **k: {"items": [], "error": "vlm_error"})
    errored = harness_loop._apply_vlm_critique(run_dir, cfg, {"filtered_repairs": []})
    assert errored["vlm_critique"]["status"] == "error"
    assert errored["vlm_critique"]["inconclusive"] is True

    monkeypatch.setattr(qa_reward, "run_critique",
                        lambda rd, c, **k: {"items": [], "error": None})
    empty = harness_loop._apply_vlm_critique(run_dir, cfg, {"filtered_repairs": []})
    assert empty["vlm_critique"]["status"] == "empty"
    assert empty["vlm_critique"]["inconclusive"] is False


def test_critique_disabled_or_legacy_leaves_critic_output_untouched(tmp_path, monkeypatch):
    input_path, run_dir = _seed(tmp_path, {"ok": False})
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        return {"items": []}

    monkeypatch.setattr(qa_reward, "run_critique", boom)
    base = {"filtered_repairs": []}
    assert harness_loop._apply_vlm_critique(run_dir, {}, dict(base)) == base
    legacy = {"runtime": {"harness": {"reward": "legacy"}},
              "qa": {"reward": {"critique": {"enabled": True}}}}
    assert harness_loop._apply_vlm_critique(run_dir, legacy, dict(base)) == base
    assert called["n"] == 0


def test_round_uses_critique_repairs_for_resume(tmp_path, monkeypatch):
    """End-to-end round: critique repair lands in critic.json and steers next_resume."""
    input_path, run_dir = _seed(tmp_path, {"ok": False, "ssim": 0.5, "hard_fails": [],
                                           "repairs": []})
    cfg = {"qa": {"reward": {"critique": {"enabled": True}}}}
    monkeypatch.setattr(qa_reward, "run_critique", lambda rd, c, **k: {
        "items": [{"element": "UPFRONT", "issue": "duplicated text, appears twice",
                   "suggested_fix": "drop the ghost copy", "layer_ids": ["c_B3"]}],
        "model": "google/gemma-4-e4b"})
    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=())

    def run_one(path, rd, c, start_from="normalize"):
        _write_qa(rd, {"ok": False, "ssim": 0.5, "hard_fails": [], "repairs": []})
        return {"ok": True}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, cfg, max_rounds=1,
        run_one=run_one, execute_repairs_fn=_repairing_exec(("merge", "dedup", "c_B3")))

    critic = json.loads((tmp_path / "run" / "critic.json").read_text(encoding="utf-8"))
    assert critic["vlm_critique"]["count"] == 1
    first = critic["filtered_repairs"][0]
    assert (first["stage"], first["action"], first["target_id"]) == ("merge", "dedup", "c_B3")
    assert summary["rounds"][0]["critic"]["vlm_critique"] == 1
    assert summary["rounds"][0]["next_resume"] == "merge"


# ── evidence + never-lower invariant ─────────────────────────────────────────────────

def test_reward_evidence_lands_in_loop_summary_harness_and_runtime_report(tmp_path, monkeypatch):
    qa = {"ok": False, "ssim": 0.6, "text_recall": 0.7, "hard_fails": [], "repairs": []}
    input_path, run_dir = _seed(tmp_path, qa)
    (tmp_path / "run" / "harness.json").write_text(
        json.dumps({"stopped": "max_iterations", "attempts": []}), encoding="utf-8")
    _stub_critic(monkeypatch)
    _stub_fixer(monkeypatch, fixes=())

    def run_one(path, rd, cfg, start_from="normalize"):
        _write_qa(rd, qa)
        return {"ok": True}

    def exec_repairs(rd, cfg, max_iterations=2, run_one=None, blocked_repairs=None):
        return {"stopped": "no_actionable_repairs", "qa_ok": False, "iterations": 0,
                "attempts": []}

    summary = harness_loop.run_until_acceptable(
        input_path, run_dir, {}, max_rounds=1,
        run_one=run_one, execute_repairs_fn=exec_repairs)

    assert summary["reward"]["mode"] == "phase2"
    assert isinstance(summary["reward"]["final"]["score"], float)
    loop = json.loads((tmp_path / "run" / "harness_loop.json").read_text(encoding="utf-8"))
    assert loop["reward"]["final"]["score"] == summary["reward"]["final"]["score"]
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text(encoding="utf-8"))
    assert report["harness_convergence"]["reward"]["mode"] == "phase2"
    harness_json = json.loads((tmp_path / "run" / "harness.json").read_text(encoding="utf-8"))
    assert harness_json["reward"]["final"]["score"] == summary["reward"]["final"]["score"]


def test_threshold_guard_rejects_lowered_reward_gate(tmp_path, monkeypatch):
    qa = {"ok": False, "ssim": 0.5, "hard_fails": [], "repairs": []}
    input_path, run_dir = _seed(tmp_path, qa)
    cfg = {"qa": {"reward": {"lpips_similarity_min": 0.5}}}

    def bad_fixer(rd, c, critic_output):
        patched = json.loads(json.dumps(c))
        patched.setdefault("qa", {}).setdefault("reward", {})["lpips_similarity_min"] = 0.05
        return {"cfg": patched, "fixes": ["bad"]}

    _stub_critic(monkeypatch)
    monkeypatch.setattr(harness_loop, "_run_fixer_pass", bad_fixer)

    def run_one(path, rd, c, start_from="normalize"):
        _write_qa(rd, qa)
        return {"ok": True}

    with pytest.raises(ValueError, match="must not lower reward gate threshold"):
        harness_loop.run_until_acceptable(
            input_path, run_dir, cfg, max_rounds=1,
            run_one=run_one, execute_repairs_fn=_repairing_exec())
