"""test_element_detect.py — synthetic-image tests for the residual-CC detector.

CPU-only. Skips cleanly if numpy/opencv/scipy aren't installed. Builds a flat
background with two colored rectangles + a text block, and asserts the detector
returns the two rectangles as elements while excluding the OCR text region.
"""
import os
import sys
import tempfile

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import element_detect  # noqa: E402


def _make_ad():
    img = np.full((400, 600, 3), 240, np.uint8)  # light gray background
    cv2.rectangle(img, (40, 40), (160, 160), (30, 90, 200), -1)     # rect A
    cv2.rectangle(img, (400, 220), (540, 340), (40, 180, 60), -1)   # rect B
    # a "text" block region (dark strokes) that OCR will report
    cv2.putText(img, "SALE", (250, 205), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (20, 20, 20), 6)
    return img


def _write(img):
    d = tempfile.mkdtemp(prefix="eltest_")
    p = os.path.join(d, "ad.png")
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return p, d


def _contains(box, x, y):
    return box["x"] <= x <= box["x"] + box["w"] and box["y"] <= y <= box["y"] + box["h"]


def test_detects_two_rects_not_text():
    img = _make_ad()
    path, _ = _write(img)
    ocr = {"lines": [{"id": "L0", "text": "SALE",
                      "box": {"x": 245, "y": 165, "w": 160, "h": 50}}]}
    els = element_detect.detect(path, ocr, {})

    # exactly the two rectangles survive
    assert len(els) == 2, [e["box"] for e in els]

    # both rectangle centers are covered by a detected element
    assert any(_contains(e["box"], 100, 100) for e in els), "rect A missing"
    assert any(_contains(e["box"], 470, 280) for e in els), "rect B missing"

    # no element is centered on the text region
    assert not any(_contains(e["box"], 300, 190) for e in els), "text leaked as element"

    # solid rectangles classify as 'shape'
    assert all(e["kind"] == "shape" for e in els)
    assert all(e["area"] >= 24 for e in els)


def test_empty_ocr_still_excludes_nothing_extra():
    """With no OCR boxes the text strokes may form their own CC, but the two
    rectangles must always be present."""
    img = _make_ad()
    path, _ = _write(img)
    els = element_detect.detect(path, {"lines": []}, {})
    assert any(_contains(e["box"], 100, 100) for e in els)
    assert any(_contains(e["box"], 470, 280) for e in els)


def test_flat_image_no_elements():
    img = np.full((120, 120, 3), 200, np.uint8)
    path, _ = _write(img)
    els = element_detect.detect(path, {"lines": []}, {})
    assert els == []


def test_writes_artifacts(tmp_path):
    img = _make_ad()
    path, _ = _write(img)
    ocr = {"lines": [{"id": "L0", "text": "SALE",
                      "box": {"x": 245, "y": 165, "w": 160, "h": 50}}]}
    els = element_detect.detect(path, ocr, {}, run_dir=str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "elements.json"))
    # per-element masks saved by id convention
    for e in els:
        assert os.path.exists(os.path.join(str(tmp_path), "elements", f"{e['id']}.png"))
