"""Tests for the VLM anomaly pass and its repair wiring.

The VLM itself is always mocked (same approach as the other vlm_* stage tests) so the
suite is CPU-only and never touches LM Studio.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import harness, repair, vlm_anomaly


def _cfg(**anomaly):
    base = {"enabled": True}
    base.update(anomaly)
    return {"vlm": {"anomaly": base}}


def _seed_preview(tmp_path):
    (tmp_path / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)


def test_disabled_by_default_makes_no_call(tmp_path, monkeypatch):
    _seed_preview(tmp_path)
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("VLM must not be called when disabled")

    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer", boom)
    assert vlm_anomaly.detect_anomalies(str(tmp_path), {}) == []
    assert called["n"] == 0


def test_detect_parses_structured_anomalies(tmp_path, monkeypatch):
    _seed_preview(tmp_path)
    payload = json.dumps({"anomalies": [
        {"type": "duplicate_text", "text": "UPFRONT", "detail": "appears twice"},
        {"type": "clipped_text", "text": "30% KORTING", "detail": "cut at right edge"},
        {"type": "wrong_glyphs", "text": "ingrediÃ«nten", "detail": "mojibake"},
        {"type": "not_a_type", "text": "ignore me"},
        {"type": "duplicate_text", "text": ""},
    ]})
    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer",
                        lambda *a, **k: (payload, None))

    anomalies = vlm_anomaly.detect_anomalies(str(tmp_path), _cfg())

    kinds = [a["type"] for a in anomalies]
    assert kinds == ["duplicate_text", "clipped_text", "wrong_glyphs"]
    assert (tmp_path / "anomalies.json").exists()
    saved = json.loads((tmp_path / "anomalies.json").read_text(encoding="utf-8"))
    assert len(saved["anomalies"]) == 3


def test_detect_never_raises_on_vlm_error(tmp_path, monkeypatch):
    _seed_preview(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("LM Studio offline")

    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer", boom)
    assert vlm_anomaly.detect_anomalies(str(tmp_path), _cfg()) == []


def test_detect_returns_empty_on_disagreement_note(tmp_path, monkeypatch):
    _seed_preview(tmp_path)
    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer",
                        lambda *a, **k: (None, "vlm_disagreement"))
    assert vlm_anomaly.detect_anomalies(str(tmp_path), _cfg()) == []


def test_detect_returns_empty_when_preview_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer",
                        lambda *a, **k: (json.dumps({"anomalies": []}), None))
    assert vlm_anomaly.detect_anomalies(str(tmp_path), _cfg()) == []


def test_detect_attaches_layer_ids_from_design(tmp_path, monkeypatch):
    _seed_preview(tmp_path)
    design = {"layers": [
        {"id": "c_B3", "type": "text", "text": "UPFRONT"},
        {"id": "c_B19", "type": "text", "text": "UPFRONT"},
        {"id": "bg", "type": "image"},
    ]}
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(
        vlm_anomaly.vlm_client, "multi_pass_answer",
        lambda *a, **k: (json.dumps({"anomalies": [
            {"type": "duplicate_text", "text": "UPFRONT", "detail": "ghosted"}]}), None),
    )

    anomalies = vlm_anomaly.detect_anomalies(str(tmp_path), _cfg())
    assert anomalies[0]["layer_ids"] == ["c_B3", "c_B19"]


def test_passes_capped_by_max_calls(tmp_path, monkeypatch):
    _seed_preview(tmp_path)
    seen = {}

    def capture(image, prompt, *, passes, **k):
        seen["passes"] = passes
        return json.dumps({"anomalies": []}), None

    monkeypatch.setattr(vlm_anomaly.vlm_client, "multi_pass_answer", capture)
    vlm_anomaly.detect_anomalies(str(tmp_path), _cfg(passes=5, max_calls=1))
    assert seen["passes"] == 1


# ── repair wiring ──────────────────────────────────────────────────────────────────────

def _actions(repairs):
    return {(item["stage"], item["action"]) for item in repairs}


def test_duplicate_text_anomaly_becomes_merge_dedup_repair():
    anomalies = [{"type": "duplicate_text", "text": "UPFRONT",
                  "detail": "ghosted", "layer_ids": ["c_B3", "c_B19"]}]
    repairs = repair.repairs_from_anomalies(anomalies)
    assert len(repairs) == 1
    item = repairs[0]
    assert (item["stage"], item["action"]) == ("merge", "dedup")
    assert item["target_id"] == "c_B3"
    assert item["params"]["duplicate_text"] == ["UPFRONT"]
    choice = harness.recommended_resume(repairs)
    assert choice["resume"] == "merge"
    assert choice["patches"]["merge"]["dedup_text"] is True
    assert choice["patches"]["merge"]["layer_ids"] == ["c_B3", "c_B19"]


def test_clipped_text_anomaly_becomes_text_refit_repair():
    anomalies = [{"type": "clipped_text", "text": "30% KORTING",
                  "detail": "cut off", "layer_ids": ["c_B18"]}]
    repairs = repair.repairs_from_anomalies(anomalies)
    item = repairs[0]
    assert (item["stage"], item["action"]) == ("text-analysis", "refit-text-box")
    # refit-text-box's ``text_analysis.fit`` patch has NO consumer in the text stage:
    # postfix-benchmark-4 burned ~70 byte-identical full-pipeline reruns on it. It is
    # a suggestion for human review, not an auto-actionable repair, until a text-stage
    # consumer exists — so the harness must NOT pick it as the resume choice.
    assert harness.is_actionable(item) is False
    choice = harness.recommended_resume(repairs)
    assert choice is None


def test_wrong_glyphs_anomaly_becomes_resolve_fonts_repair():
    anomalies = [{"type": "wrong_glyphs", "text": "UPERONT", "detail": "garbled"}]
    repairs = repair.repairs_from_anomalies(anomalies)
    assert ("text-analysis", "resolve-fonts") in _actions(repairs)
    choice = harness.recommended_resume(repairs)
    assert choice["resume"] == "text"


def test_assess_reads_anomalies_json_from_run_dir(tmp_path):
    (tmp_path / "anomalies.json").write_text(json.dumps({"anomalies": [
        {"type": "duplicate_text", "text": "UPFRONT", "layer_ids": ["c_B3", "c_B19"]},
    ]}), encoding="utf-8")
    repairs = repair.assess({}, {"ok": False}, {"lines": []}, {"run_dir": str(tmp_path)})
    assert ("merge", "dedup") in _actions(repairs)
    # ad2 (ghosted duplicate text) is now catchable and resumes at the merge stage.
    choice = harness.recommended_resume(repairs)
    assert choice is not None
    assert choice["stage"] == "merge"


def test_assess_reads_anomalies_embedded_in_qa():
    qa = {"ok": False, "anomalies": [
        {"type": "clipped_text", "text": "SHOP NOW", "layer_ids": ["cta"]}]}
    repairs = repair.assess({}, qa, {"lines": []}, {})
    assert ("text-analysis", "refit-text-box") in _actions(repairs)


def test_refit_text_box_is_not_auto_actionable():
    # The action stays in the repair vocabulary (resume mapping intact) but its config
    # patch (text_analysis.fit) reaches no pipeline stage, so the harness must refuse
    # to spend a full-pipeline rerun on it (postfix-benchmark-4: ~70 no-op reruns,
    # including 091's full peel-stack replay with 12 Flux inpaints).
    r = {"stage": "text-analysis", "action": "refit-text-box",
         "params": {"widen": True, "shrink_to_fit": True}}
    assert harness.resume_stage_for(r) == "text"
    assert harness.is_actionable(r) is False
