"""Rotation corroboration: lone small OCR quad angles snap to 0, and a rotated
line/block rotates as one rigid body.

Regression cover for the P0 "spurious small rotations" defect:
  * ad 025 card text emitted at 2.35 deg — merge_layers._text_sources averaged a
    mixed [0, 4.7] block instead of trusting text_analysis's corroborated consensus;
  * ad 088 "OFF" emitted at 4.42 deg — a single uncorroborated quad past the 2.5 deg
    snap with no block-mate, neighbour or carrier to agree with it;
  * genuinely rotated lines whose WORDS each rendered at their own slightly
    different angle instead of the whole line rotating as one rigid line.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import build_design_json, merge_layers, text_analysis  # noqa: E402


_CFG = {"text_analysis": {"font_matching": {"enabled": False}}}


def _line(line_id, text, bbox, conf=0.98):
    x0, y0, x1, y1 = bbox
    box = {"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)}
    return {"id": line_id, "text": text, "conf": conf, "box": box,
            "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "words": []}


def _tilted_line(line_id, text, x0, y0, width, height, angle):
    """An OCR line whose quad baseline runs at ``angle`` degrees (OCR wobble shape)."""
    radians = math.radians(angle)
    dx, dy = math.cos(radians) * width, math.sin(radians) * width
    line = _line(line_id, text, (x0, y0, x0 + width, y0 + height))
    line["quad"] = [[x0, y0], [x0 + dx, y0 + dy],
                    [x0 + dx, y0 + dy + height], [x0, y0 + height]]
    return line


def _stack(texts_with_angles, x0=40.0, top=80.0, width=300.0, height=26.0, gap=8.0):
    lines = []
    y = top
    for index, (text, angle) in enumerate(texts_with_angles):
        lines.append(_tilted_line(f"L{index}", text, x0, y, width, height, angle))
        y += height + gap
    return lines


def _analyze(tmp_path, lines, name="rot.png", size=(700, 320), cfg=None):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return text_analysis.analyze_text(
        str(path), {"source": {"w": size[0], "h": size[1]}, "lines": lines},
        cfg if cfg is not None else _CFG,
    )


# ---------------------------------------------------------------------------
# Block consensus: mixed members emit 0, agreeing members rotate as ONE rigid body


def test_mixed_block_emits_zero_not_the_member_mean(tmp_path):
    # 025's c_B7: one horizontal line + one 4.7 deg wobble quad in the same block.
    # The corroborated consensus is 0 — the emitted block must never carry the
    # 2.35 deg mean that merge_layers._text_sources used to compute.
    result = _analyze(tmp_path, _stack([
        ("Card copy line one, painted horizontal", 0.0),
        ("Card copy line two with a wobbly quad", 4.7),
    ]))
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 2)
    assert block["rotation_deg"] == 0.0
    assert all(line["rotation_deg"] == 0.0 for line in result["lines"])

    sources = merge_layers._text_sources(result)
    emitted = next(candidate for candidate in sources if candidate.get("id") == block["id"])
    assert emitted["rotation"] == 0.0  # not 2.35


def test_text_sources_trusts_block_consensus_over_member_angles():
    # Direct unit cover for the merge-side fix: member line rotations [0, 4.7]
    # under a block whose corroborated rotation_deg is 0 must emit 0.
    lines = [
        {"id": "L0", "text": "horizontal", "conf": 0.9, "rotation": 0.0,
         "box": {"x": 40, "y": 80, "w": 300, "h": 26}},
        {"id": "L1", "text": "wobbly quad", "conf": 0.9, "rotation": 4.7,
         "box": {"x": 40, "y": 114, "w": 300, "h": 26}},
    ]
    block = {"id": "B0", "type": "paragraph", "line_ids": ["L0", "L1"],
             "text": "horizontal\nwobbly quad", "rotation": 0.0, "rotation_deg": 0.0,
             "box": {"x": 40, "y": 80, "w": 300, "h": 60},
             "painted_box": {"x": 40, "y": 80, "w": 300, "h": 60},
             "alignment": "left", "line_height": 34.0, "role": "body",
             "hierarchy": {"level": 0, "parent_id": None}, "style_id": None, "meta": {}}
    sources = merge_layers._text_sources({"lines": lines, "blocks": [block], "styles": []})
    emitted = next(candidate for candidate in sources if candidate.get("id") == "B0")
    assert emitted["rotation"] == 0.0


def test_agreeing_block_members_rotate_as_one_rigid_line(tmp_path):
    # Two stacked lines at 4.4/4.5 deg: under the 6 deg corroboration threshold
    # individually, but block-mate agreement corroborates them — and BOTH must
    # carry the one consensus angle exactly (no per-line drift inside a paragraph).
    result = _analyze(tmp_path, _stack([
        ("Tilted pair line one of the banner", 4.4),
        ("Tilted pair line two of the banner", 4.5),
    ]))
    block = next(block for block in result["blocks"] if len(block["line_ids"]) == 2)
    assert block["rotation_deg"] == pytest.approx(4.45, abs=0.05)
    angles = [line["rotation_deg"] for line in result["lines"]]
    assert angles[0] == pytest.approx(4.45, abs=0.05)
    assert angles[0] == angles[1]  # rigid: identical, not each line's own quad


def test_adjacent_agreeing_lines_corroborate_each_other():
    # Signal (ii) without a shared block: two single-line blocks stacked close
    # together whose raw angles agree within 1.5 deg both survive the gate.
    def _gate_line(line_id, y, angle):
        return {"id": line_id, "text": line_id, "rotation": angle, "rotation_deg": angle,
                "box": {"x": 40.0, "y": y, "w": 300.0, "h": 26.0}, "meta": {}}

    def _solo_block(line_id, block_id):
        return {"id": block_id, "line_ids": [line_id], "rotation": 0.0, "rotation_deg": 0.0}

    lines = [_gate_line("L0", 80.0, 4.4), _gate_line("L1", 114.0, 4.5)]
    blocks = [_solo_block("L0", "B0"), _solo_block("L1", "B1")]
    for block, line in zip(blocks, lines):
        block["rotation"] = block["rotation_deg"] = line["rotation_deg"]
    text_analysis._apply_rotation_corroboration(lines, blocks, {})
    assert lines[0]["rotation_deg"] == pytest.approx(4.4, abs=0.01)
    assert lines[1]["rotation_deg"] == pytest.approx(4.5, abs=0.01)


# ---------------------------------------------------------------------------
# The corroboration gate for lone lines


def test_single_uncorroborated_small_angle_snaps_to_zero(tmp_path):
    # 088's "OFF": one lone quad at 4.42 deg, nothing nearby to agree with it.
    result = _analyze(tmp_path, [_tilted_line("L0", "OFF", 60.0, 90.0, 120.0, 40.0, 4.42)])
    line = result["lines"][0]
    assert line["rotation_deg"] == 0.0
    assert line["meta"]["rotation_raw_deg"] == pytest.approx(4.42, abs=0.05)
    assert result["blocks"][0]["rotation_deg"] == 0.0


def test_genuine_angle_above_threshold_needs_no_corroboration(tmp_path):
    result = _analyze(tmp_path, [_tilted_line("L0", "TILTED", 60.0, 90.0, 200.0, 34.0, 8.5)])
    assert result["lines"][0]["rotation_deg"] == pytest.approx(8.5, abs=0.05)
    assert result["blocks"][0]["rotation_deg"] == pytest.approx(8.5, abs=0.05)


def test_small_angle_kept_when_carrier_median_agrees(tmp_path):
    # 013-style badge: a lone smallish angle on a rotated carrier whose cluster
    # median was injected into meta by the badge/seal read.
    line = _tilted_line("L0", "61% OFF", 60.0, 90.0, 220.0, 60.0, 5.4)
    line["meta"] = {"carrier_rotation_deg": 7.9}
    result = _analyze(tmp_path, [line])
    assert result["lines"][0]["rotation_deg"] == pytest.approx(5.4, abs=0.05)


def test_corroborate_threshold_is_configurable(tmp_path):
    cfg = {"text_analysis": {"font_matching": {"enabled": False},
                             "rotation_corroborate_deg": 3.0}}
    result = _analyze(tmp_path, [_tilted_line("L0", "OFF", 60.0, 90.0, 120.0, 40.0, 4.42)],
                      cfg=cfg)
    # 4.42 clears a lowered 3 deg corroboration bar on its own.
    assert result["lines"][0]["rotation_deg"] == pytest.approx(4.42, abs=0.05)


# ---------------------------------------------------------------------------
# Word-level: weight-run siblings inherit the line's final angle, never a word's


def test_weight_run_siblings_share_the_lines_final_rotation():
    candidate = {
        "id": "c_L0", "target": "text", "text": "hello bold world",
        "box": {"x": 0, "y": 0, "w": 220, "h": 24},
        "rotation": 4.45,  # the line's final post-gate angle
        "style": {"fontSize": 16.0, "fontWeight": 400},
        "text_runs": [{"start": 6, "end": 10, "style": {"fontWeight": 700}}],
        "meta": {},
    }
    pieces = build_design_json._split_weight_run_siblings(candidate)
    assert len(pieces) > 1
    assert [piece["text"] for piece in pieces] == ["hello", "bold", "world"]
    # Every sibling rotates with its line — identical angles, no per-word drift.
    assert {piece["rotation"] for piece in pieces} == {4.45}
