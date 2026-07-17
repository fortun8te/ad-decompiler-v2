"""CPU-safe tests for src/icon_detect.py (✓/✗/? glyphs + chart region).

Synthetic canvases only — no models, no GPU, no benchmark artifacts.
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from src import icon_detect as ic


def _canvas(w=800, h=400, color=(255, 255, 255)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = color
    return img


def _draw_check_chip(img, x, y, size=30, fill=(60, 180, 60)):
    cv2.rectangle(img, (x, y), (x + size, y + size), fill, -1)
    s = size / 30.0
    pts = np.asarray([(x + 6 * s, y + 16 * s), (x + 14 * s, y + 24 * s),
                      (x + 25 * s, y + 6 * s)], np.int32)
    cv2.polylines(img, [pts], False, (255, 255, 255), max(2, int(4 * s)))


def _draw_cross(img, x, y, size=28, color=(220, 40, 40), t=5):
    cv2.line(img, (x, y), (x + size, y + size), color, t)
    cv2.line(img, (x + size, y), (x, y + size), color, t)


def _rows(n, x_text, y0=80, step=70, h=30, prefix="L"):
    return [{"id": f"{prefix}{i}", "text": "row copy text",
             "box": {"x": float(x_text), "y": float(y0 + i * step),
                     "w": 200.0, "h": float(h)}} for i in range(n)]


def test_row_icons_check_and_cross_columns():
    img = _canvas()
    lines = []
    for i in range(3):
        y = 80 + i * 70
        _draw_check_chip(img, 40, y)
        _draw_cross(img, 440, y + 2)
    lines += _rows(3, 84, prefix="L")
    lines += _rows(3, 480, prefix="R")
    dets = ic.detect_row_icons(img, lines, {"w": 800, "h": 400}, ic.DEFAULTS)
    checks = [d for d in dets if d["glyph"] == "check"]
    crosses = [d for d in dets if d["glyph"] == "cross"]
    assert len(checks) == 3
    assert len(crosses) == 3
    # every detection is attached to its row line
    assert all(d["row_text_id"] for d in dets)
    # color robustness is structural: red cross classified as cross, not by hue
    assert all(d["glyph"] == "cross" for d in crosses)


def test_row_icons_no_false_positives_on_plain_rows():
    img = _canvas()
    lines = _rows(3, 84)
    dets = ic.detect_row_icons(img, lines, {"w": 800, "h": 400}, ic.DEFAULTS)
    assert dets == []


def test_solid_plate_not_classified_as_glyph():
    # a solid rounded plate must not pass as a fat cross (107 packshot cap)
    m = np.zeros((48, 48), bool)
    cv2.rectangle(m.view(np.uint8).reshape(48, 48), (4, 4), (43, 43), 1, -1)
    glyph, score = ic.classify_glyph(m)
    assert glyph != "cross" or score < 0.4


def test_classify_cross_and_check_templates():
    m = np.zeros((60, 60), np.uint8)
    cv2.line(m, (8, 8), (52, 52), 255, 7)
    cv2.line(m, (52, 8), (8, 52), 255, 7)
    glyph, score = ic.classify_glyph(m > 0)
    assert glyph == "cross" and score > 0.7
    m = np.zeros((60, 60), np.uint8)
    cv2.polylines(m, [np.asarray([(6, 34), (22, 50), (54, 10)], np.int32)],
                  False, 255, 7)
    glyph, score = ic.classify_glyph(m > 0)
    assert glyph == "check" and score > 0.55


def test_chart_region_gridlines():
    img = _canvas(1000, 1000, (245, 245, 245))
    for i in range(5):
        y = 300 + i * 100
        cv2.line(img, (150, y), (850, y), (190, 190, 190), 2)
    lines = [{"id": "W1", "text": "WEEK 1",
              "box": {"x": 180.0, "y": 730.0, "w": 90.0, "h": 24.0}}]
    chart = ic.detect_chart_region(img, lines, {"w": 1000, "h": 1000}, ic.DEFAULTS)
    assert chart is not None
    b = chart["box"]
    assert chart["gridlines"] >= 4
    # plot area covered, axis label row excluded
    assert b["y"] < 300 and b["y"] + b["h"] <= 730
    assert b["x"] <= 150 and b["x"] + b["w"] >= 850


def test_chart_region_absent_on_two_lines():
    img = _canvas(1000, 1000)
    cv2.line(img, (100, 200), (900, 200), (120, 120, 120), 2)
    cv2.line(img, (100, 800), (900, 800), (120, 120, 120), 2)
    assert ic.detect_chart_region(img, [], {"w": 1000, "h": 1000},
                                  ic.DEFAULTS) is None


def test_refine_is_noop_without_run_dir():
    fused = [{"id": "E000", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
              "kind": "shape", "role": "shape"}]
    out = ic.refine(list(fused), canvas={"w": 100, "h": 100}, cfg={}, run_dir=None)
    assert out == fused


def test_refine_adds_missing_icons(tmp_path):
    img = _canvas()
    lines_json = {"lines": []}
    for i in range(3):
        y = 80 + i * 70
        _draw_cross(img, 440, y + 2)
        lines_json["lines"].append(
            {"id": f"R{i}", "text": "row copy",
             "box": {"x": 480, "y": y, "w": 200, "h": 30}})
    cv2.imwrite(str(tmp_path / "normalized.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    import json
    (tmp_path / "ocr.json").write_text(json.dumps(lines_json), encoding="utf-8")
    fused = []
    out = ic.refine(fused, canvas={"w": 800, "h": 400}, cfg={},
                    run_dir=str(tmp_path))
    crosses = [e for e in out if e["role"] == "cross"]
    assert len(crosses) == 3
    for e in crosses:
        assert e["kind"] == "icon"
        assert e["source"] == "icon-cv"
        assert (e["meta"].get("row") or {}).get("text_id")
        assert e["meta"].get("vector_template")
        assert (tmp_path / "fused_elements" / f"{e['id']}.png").exists()
    assert (tmp_path / "icon_detect.json").exists()


def test_refine_absorbs_stacked_duplicates(tmp_path):
    img = _canvas()
    _draw_cross(img, 440, 82)
    import json
    cv2.imwrite(str(tmp_path / "normalized.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    (tmp_path / "ocr.json").write_text(json.dumps(
        {"lines": [{"id": "R0", "text": "row copy",
                    "box": {"x": 480, "y": 80, "w": 200, "h": 30}}]}),
        encoding="utf-8")
    # two stacked fused fragments over the same glyph (066 failure shape)
    fused = [
        {"id": "E000", "box": {"x": 438, "y": 80, "w": 32, "h": 32},
         "kind": "icon", "role": "icon", "score": 0.7, "area": 500.0,
         "coverage": 0.001, "source": "fused", "parent_id": None,
         "relationships": []},
        {"id": "E001", "box": {"x": 440, "y": 82, "w": 14, "h": 30},
         "kind": "icon", "role": "icon", "score": 0.8, "area": 250.0,
         "coverage": 0.001, "source": "fused", "parent_id": None,
         "relationships": []},
    ]
    out = ic.refine(fused, canvas={"w": 800, "h": 400}, cfg={},
                    run_dir=str(tmp_path))
    icons = [e for e in out if e.get("kind") == "icon"]
    assert len(icons) == 1
    assert icons[0]["role"] == "cross"
    assert icons[0]["meta"].get("absorbed_ids")


def test_text_clip_signal_for_swallowed_glyph():
    """066 L10/L15: OCR box starts ON the icon column → publish text_clip_x after it."""
    dets = [
        # overlap row: text box starts at 820, icon spans 824..857
        {"box": {"x": 824, "y": 1291, "w": 33, "h": 33},
         "row_box": {"x": 820, "y": 1286, "w": 318, "h": 45}, "row_text_id": "L15"},
        # clean sibling rows in the same column: text starts well after the icon
        {"box": {"x": 824, "y": 950, "w": 33, "h": 33},
         "row_box": {"x": 873, "y": 943, "w": 403, "h": 53}, "row_text_id": "L9"},
        {"box": {"x": 824, "y": 1120, "w": 33, "h": 33},
         "row_box": {"x": 872, "y": 1111, "w": 309, "h": 60}, "row_text_id": "L12"},
    ]
    ic._annotate_text_clip(dets, ic.DEFAULTS)
    over = dets[0]
    assert over["overlaps_text"] is True
    # clip x is past the icon's right edge (857) and lands in the clean column (872-880)
    assert 857 < over["text_clip_x"] <= 885
    # clean rows are never annotated
    assert not dets[1].get("overlaps_text")
    assert not dets[2].get("overlaps_text")


def _lockup_lines_and_img(two_tone=True):
    img = _canvas(w=1000, h=1000)
    # line 1 "craft" teal, line 2 "cadence" black (or teal when single-tone), isolated
    c1 = (90, 190, 195)
    c2 = (30, 30, 30) if two_tone else (90, 190, 195)
    img[374:395, 104:210] = 255
    img[374:395, 104:180] = c1
    img[406:424, 72:238] = 255
    img[406:424, 72:200] = c2
    lines = [
        {"id": "L2", "text": "craft", "box": {"x": 104, "y": 374, "w": 106, "h": 21}},
        {"id": "L3", "text": "cadence", "box": {"x": 72, "y": 406, "w": 166, "h": 18}},
        # far-away body copy so the block stays isolated
        {"id": "L9", "text": "typical tube", "box": {"x": 628, "y": 647, "w": 240, "h": 17}},
    ]
    return img, lines


def test_brand_lockup_two_tone_detected():
    img, lines = _lockup_lines_and_img(two_tone=True)
    out = ic.detect_brand_lockups(img, lines, {"w": 1000, "h": 1000}, [], ic.DEFAULTS)
    assert len(out) == 1
    lk = out[0]
    assert set(lk["text_ids"]) == {"L2", "L3"}
    assert lk["box"]["x"] <= 80 and lk["box"]["w"] >= 150


def test_brand_lockup_single_tone_rejected():
    img, lines = _lockup_lines_and_img(two_tone=False)
    out = ic.detect_brand_lockups(img, lines, {"w": 1000, "h": 1000}, [], ic.DEFAULTS)
    assert out == []


def test_brand_lockup_skips_when_owned_by_burst():
    img, lines = _lockup_lines_and_img(two_tone=True)
    covering = [{"id": "S0", "role": "sale_burst",
                 "box": {"x": 60, "y": 360, "w": 200, "h": 80}}]
    out = ic.detect_brand_lockups(img, lines, {"w": 1000, "h": 1000}, covering, ic.DEFAULTS)
    assert out == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
