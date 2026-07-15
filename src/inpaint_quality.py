"""Deterministic, CPU-safe signals for ranking inpaint candidates.

These metrics deliberately inspect only the proposed fill and the retained source
context.  They do not claim to recognise every object; they make obvious bad
candidates (hard seams, texture discontinuities, and text-like residue) less
likely to win a seed/backend comparison without requiring a GPU model.
"""
from __future__ import annotations


def _deps():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError("inpaint quality scoring requires numpy and opencv-python") from exc
    return cv2, np


def _gray(image, cv2, np):
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def candidate_metrics(source, candidate, mask) -> dict:
    """Return lower-is-better continuity/residue metrics for one candidate.

    ``candidate`` may include generated pixels outside the mask.  The scorer
    always composes it onto ``source`` inside the requested hole first, so an
    overeager backend cannot affect scoring through unrelated pixels.
    """
    cv2, np = _deps()
    source = np.asarray(source, dtype=np.uint8)
    candidate = np.asarray(candidate, dtype=np.uint8)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return {"texture": 0.0, "structure": 0.0, "residue": 0.0}

    composed = source.copy()
    composed[binary > 0] = candidate[binary > 0]
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(binary, kernel, iterations=1) > 0
    # Tiny masks may lose their interior after erosion; the full mask is still
    # useful for texture/structure scoring in that case.
    if not inner.any():
        inner = binary > 0
    outer = (cv2.dilate(binary, kernel, iterations=2) > 0) & (binary == 0)
    if not outer.any():
        outer = binary == 0

    gray = _gray(composed, cv2, np)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    # Sobel alone can cancel on a one-pixel checker/text stroke.  A small high-pass
    # term catches that texture while remaining deterministic and CPU-cheap.
    detail = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))

    inner_gradient = float(np.median(gradient[inner])) if inner.any() else 0.0
    outer_gradient = float(np.median(gradient[outer])) if outer.any() else 0.0
    inner_detail = float(np.median(detail[inner])) if inner.any() else 0.0
    outer_detail = float(np.median(detail[outer])) if outer.any() else 0.0
    # Texture continuity compares robust gradient energy, so one antialiased
    # boundary pixel cannot dominate the score.
    texture = abs(inner_gradient - outer_gradient) + 0.5 * abs(inner_detail - outer_detail)

    # Structural discontinuity combines the difference in local intensity and
    # the directional gradient vector.  It is intentionally a soft ranking
    # signal: a legitimate photo edge may score high, but a flat fill over a
    # panel boundary should not automatically lose to text-shaped residue.
    inner_gray = float(np.median(gray[inner])) if inner.any() else 0.0
    outer_gray = float(np.median(gray[outer])) if outer.any() else 0.0
    inner_vec = np.array([
        float(np.median(gx[inner])) if inner.any() else 0.0,
        float(np.median(gy[inner])) if inner.any() else 0.0,
    ])
    outer_vec = np.array([
        float(np.median(gx[outer])) if outer.any() else 0.0,
        float(np.median(gy[outer])) if outer.any() else 0.0,
    ])
    structure = abs(inner_gray - outer_gray) + 0.35 * float(np.linalg.norm(inner_vec - outer_vec))

    # Text/object leftovers tend to be compact, high-frequency islands in the
    # middle of an otherwise inferred plate.  Compare the candidate's local
    # detail to the retained ring, then count only interior components so the
    # real boundary does not become a false residue hit.
    reference = float(np.percentile(detail[outer], 90)) if outer.any() else 0.0
    threshold = max(10.0, reference * 1.8)
    residue_map = ((detail >= threshold) & inner).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(residue_map, connectivity=8)
    components = 0
    residue_pixels = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 2:
            components += 1
            residue_pixels += area
    residue = 100.0 * residue_pixels / max(1, int(np.count_nonzero(inner))) + components * 0.5

    return {
        "texture": round(float(texture), 6),
        "structure": round(float(structure), 6),
        "residue": round(float(residue), 6),
    }
