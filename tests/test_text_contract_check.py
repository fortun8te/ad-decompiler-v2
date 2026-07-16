"""Smoke coverage for the text-contract verification sweep.

This does NOT assert zero violations (benchmark runs legitimately contain defects
the sweep is meant to surface); it verifies the checker executes end-to-end and
returns the documented structure, so the "never again" gate itself stays working.
"""
import glob
import os

import numpy as np
import pytest

from scripts import text_contract_check as tcc


def _a_run_dir():
    for base in ("runs/postfix-benchmark-6", "runs/postfix-benchmark-5"):
        for d in sorted(glob.glob(os.path.join(base, "*_attached_*"))):
            if os.path.isfile(os.path.join(d, "design.json")):
                return d
    return None


def test_ink_mask_handles_high_ink_crops():
    # A tightly-cropped headline (large ink fraction) must not vanish under Otsu.
    crop = np.full((40, 200, 3), 255, np.uint8)
    crop[10:30, 10:190] = 0  # ~45% ink
    mask = tcc._ink_mask(crop)
    assert mask.sum() > 100


def test_contrast_and_luminance_monotone():
    assert tcc._contrast((0, 0, 0), (255, 255, 255)) > 15.0
    assert tcc._contrast((255, 255, 255), (255, 255, 255)) == pytest.approx(1.0)


def test_check_run_returns_structure():
    run = _a_run_dir()
    if not run:
        pytest.skip("no benchmark run available")
    rep = tcc.check_run(run)
    assert set(("fixture", "violations", "nodes", "source_lines")).issubset(rep)
    assert isinstance(rep["violations"], list)
    for v in rep["violations"]:
        assert v["severity"] in ("HARD", "WARN", "ERROR")
        assert set(("rule", "detail", "text")).issubset(v)
