import numpy as np

from src import inpaint


def test_opencv_auto_chooses_lower_seam_candidate_and_preserves_unmasked_pixels(monkeypatch):
    source = np.full((12, 12, 3), 120, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 255
    telea = np.full_like(source, 20)
    ns = np.full_like(source, 220)

    def fake_inpaint(rgb, inner_mask, radius, method):
        return ns if method == "ns" else telea

    monkeypatch.setattr(inpaint, "_opencv_inpaint", fake_inpaint)
    monkeypatch.setattr(inpaint, "_seam_energy", lambda image, inner_mask: 8.0 if image is telea else 2.0)

    output, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "opencv", "opencv_method": "auto"}},
        return_diagnostics=True,
    )

    assert backend == "opencv-ns"
    assert diagnostics == {"telea_seam": 8.0, "ns_seam": 2.0}
    assert np.all(output[mask == 0] == source[mask == 0])
    assert np.all(output[mask > 0] == 220)


def test_inpaint_array_keeps_legacy_two_value_return_by_default():
    source = np.full((4, 4, 3), 100, dtype=np.uint8)
    output, backend = inpaint.inpaint_array(source, np.zeros((4, 4), dtype=np.uint8), {})

    assert backend == "none"
    assert np.array_equal(output, source)


def test_build_union_mask_excludes_kept_regions_so_they_are_never_erased_or_regenerated():
    """Guards the "editable regions vs. real background" invariant: an entity that is
    staying visible in the photo (keep_in_background / is_background) must never be
    added to the inpaint removal hole -- otherwise the plate would erase real content and
    the inpainter would hallucinate a duplicate/ghosted object back into the gap."""
    canvas = (10, 10)
    removed_mask = np.zeros((10, 10), dtype=np.uint8)
    removed_mask[2:5, 2:5] = 255
    kept_mask = np.zeros((10, 10), dtype=np.uint8)
    kept_mask[6:9, 6:9] = 255

    observations = [
        {"box": {"x": 2, "y": 2, "w": 3, "h": 3}, "mask_array": removed_mask, "dilate": 0},
        {"box": {"x": 6, "y": 6, "w": 3, "h": 3}, "mask_array": kept_mask, "dilate": 0,
         "keep_in_background": True},
    ]

    union = inpaint.build_union_mask(canvas, observations, default_dilate=0)

    assert np.any(union[2:5, 2:5])
    assert not np.any(union[6:9, 6:9])


def test_build_union_mask_excludes_is_background_flagged_entities():
    canvas = (10, 10)
    plate_mask = np.full((10, 10), 255, dtype=np.uint8)
    observations = [
        {"box": {"x": 0, "y": 0, "w": 10, "h": 10}, "mask_array": plate_mask, "dilate": 0,
         "is_background": True},
    ]

    union = inpaint.build_union_mask(canvas, observations, default_dilate=0)

    assert not np.any(union)
