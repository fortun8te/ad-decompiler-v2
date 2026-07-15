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


def test_edge_gradient_is_clamped_not_wrapped():
    """Regression: the border gradient must be edge-clamped, not circular (np.roll).

    With np.roll, column 0's "left" neighbor wraps to the LAST column and column -1's
    "right" neighbor wraps to the FIRST column, so a bright value on one edge leaks into
    the gradient of the opposite edge. Edge-clamped (replicate) padding keeps each
    border's gradient local to its own side.
    """
    row = np.array([100.0, 50.0, 50.0, 200.0])
    gray_x = np.tile(row, (2, 1))  # shape (2, 4): flat along axis 0, varying along axis 1
    gx_gy = element_detect._edge_gradient_magnitude(gray_x)
    # Edge-clamped forward/backward differences at the boundaries:
    # col0: |col1 - col0| = |50-100| = 50   (NOT the wrapped |col1 - col_last| = |50-200| = 150)
    # col3: |col_last - col2| = |200-50| = 150 (NOT the wrapped |col0 - col2| = |100-50| = 50)
    expected_row = np.array([50.0, 50.0, 150.0, 150.0])
    for row_result in gx_gy:
        assert row_result.tolist() == expected_row.tolist()

    col = np.array([[100.0], [50.0], [50.0], [200.0]])
    gray_y = np.tile(col, (1, 2))  # shape (4, 2): flat along axis 1, varying along axis 0
    grad_y = element_detect._edge_gradient_magnitude(gray_y)
    expected_col = np.array([50.0, 50.0, 150.0, 150.0])
    for col_idx in range(2):
        assert grad_y[:, col_idx].tolist() == expected_col.tolist()


def test_border_touching_icon_is_not_misclassified_by_wrapped_gradient():
    """A high-contrast icon-like shape that touches the canvas edge must not have its
    edge_density skewed by wraparound diffing against the opposite border."""
    img = np.full((200, 200, 3), 240, np.uint8)
    # Small checkerboard-ish icon-like patch touching the LEFT edge (x=0), with a
    # deliberately different color sitting at the RIGHT edge so a circular-wrap bug
    # would leak that unrelated color into this patch's edge-density measurement.
    cv2.rectangle(img, (0, 80), (30, 110), (10, 200, 10), -1)
    cv2.rectangle(img, (198, 0), (200, 200), (0, 0, 250), -1)  # unrelated bright right border
    path, _ = _write(img)
    els = element_detect.detect(path, {"lines": []}, {})
    assert any(_contains(e["box"], 10, 95) for e in els), "border-touching element missing"


def test_adaptive_recovers_low_contrast_small_element_on_flat_bg():
    """On a flat, clean background the residual noise sigma is ~0, so the adaptive pass
    lowers the luma bar (clamped at adaptiveScaleMin) and a low-contrast element that the
    fixed 14.0 threshold silently dropped is recovered. With adaptive disabled the
    historical constants apply unchanged."""
    img = np.full((400, 600, 3), 240, np.uint8)
    img[100:130, 200:230, :] = 228  # gray-on-gray square: |dY| = 12, chroma 0
    path, _ = _write(img)

    fixed = element_detect.detect(path, {"lines": []}, {"element_detect": {"adaptive": False}})
    assert not any(_contains(e["box"], 215, 115) for e in fixed), "12-level delta above 14 bar?"

    adaptive = element_detect.detect(path, {"lines": []}, {})
    assert any(_contains(e["box"], 215, 115) for e in adaptive), "low-contrast square missed"


def test_adaptive_opts_scales_thresholds_and_min_area():
    ref_area = element_detect.DEFAULTS["adaptiveRefArea"]

    # flat residual -> sigma 0 -> scale clamps at adaptiveScaleMin
    flat = np.zeros((100, 100), dtype=np.float64)
    opts = element_detect._adaptive_opts(dict(element_detect.DEFAULTS), flat, ref_area)
    assert opts["lumaThresh"] == 14.0 * 0.6
    assert opts["chromaThresh"] == 20.0 * 0.6
    assert opts["minArea"] == 24  # reference canvas keeps the historical default

    # noisy residual -> big sigma -> scale clamps at adaptiveScaleMax
    noisy = np.zeros((100, 100), dtype=np.float64)
    noisy[::2, :] = 60.0  # median 30, MAD 30 -> sigma ~44.5
    opts = element_detect._adaptive_opts(dict(element_detect.DEFAULTS), noisy, ref_area)
    assert opts["lumaThresh"] == 14.0 * 1.6
    assert opts["chromaThresh"] == 20.0 * 1.6

    # reference sigma reproduces the historical constants exactly
    at_ref = np.zeros((100, 100), dtype=np.float64)
    at_ref[::2, :] = 2 * element_detect.DEFAULTS["adaptiveRefSigma"] / 1.4826
    opts = element_detect._adaptive_opts(dict(element_detect.DEFAULTS), at_ref, ref_area)
    assert abs(opts["lumaThresh"] - 14.0) < 1e-9
    assert abs(opts["chromaThresh"] - 20.0) < 1e-9

    # minArea follows canvas area: 4x reference canvas hits the 4x cap, tiny canvas floors
    opts = element_detect._adaptive_opts(dict(element_detect.DEFAULTS), flat, 5 * ref_area)
    assert opts["minArea"] == 24 * 4
    opts = element_detect._adaptive_opts(dict(element_detect.DEFAULTS), flat, 100 * 100)
    assert opts["minArea"] == element_detect.DEFAULTS["adaptiveMinAreaFloor"]


def test_adaptive_flat_image_still_yields_no_elements():
    img = np.full((120, 120, 3), 200, np.uint8)
    path, _ = _write(img)
    assert element_detect.detect(path, {"lines": []}, {}) == []


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
