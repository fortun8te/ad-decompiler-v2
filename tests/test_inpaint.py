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
    assert diagnostics["telea_seam"] == 8.0
    assert diagnostics["ns_seam"] == 2.0
    assert diagnostics["backend_choice"] == "opencv-ns"
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


def test_resolve_mask_dilate_maps_buttons_photos_and_text():
    cfg = {"inpaint": {"mask_dilate": {
        "default": 2, "button": 5, "shape": 4, "text": 2, "photo": 0, "image": 2,
    }}}
    assert inpaint.resolve_mask_dilate({"target": "shape", "meta": {"role": "button"}}, cfg) == 5
    assert inpaint.resolve_mask_dilate({"target": "shape", "meta": {"role": "card"}}, cfg) == 5
    assert inpaint.resolve_mask_dilate({"target": "shape", "meta": {"role": "sticker"}}, cfg) == 4
    assert inpaint.resolve_mask_dilate({"target": "text"}, cfg) == 2
    assert inpaint.resolve_mask_dilate({"target": "image", "meta": {"role": "product"}}, cfg) == 0
    assert inpaint.resolve_mask_dilate({"target": "image", "meta": {"role": "logo"}}, cfg) == 2


def test_build_union_mask_solidifies_soft_alpha_before_dilate():
    canvas = (20, 20)
    soft = np.zeros((20, 20), dtype=np.uint8)
    soft[5:15, 5:15] = 40
    observations = [{"box": {"x": 5, "y": 5, "w": 10, "h": 10}, "mask_array": soft, "dilate": 0}]

    union = inpaint.build_union_mask(canvas, observations, default_dilate=0)

    assert np.count_nonzero(union) == 100
    assert set(np.unique(union)).issubset({0, 255})


def test_build_union_mask_applies_minimal_feather_from_cfg():
    canvas = (16, 16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    observations = [{"box": {"x": 4, "y": 4, "w": 8, "h": 8}, "mask_array": mask, "dilate": 0}]
    cfg = {"inpaint": {"mask_feather": 1}}

    union = inpaint.build_union_mask(canvas, observations, default_dilate=0, cfg=cfg)

    assert union[8, 8] == 255
    rim = union[3, 8]
    assert 0 < rim < 255


def test_auto_prefers_big_lama_when_comfyui_healthy(monkeypatch):
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    hallucination = np.zeros_like(source)
    monkeypatch.setattr(inpaint, "comfyui_healthy", lambda cfg, probe=None: True)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_simple_lama", lambda rgb, inner_mask: hallucination)

    output, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto"}}, return_diagnostics=True,
    )

    assert backend == "big-lama"
    assert diagnostics["comfyui_healthy"] is True
    assert diagnostics["backend_choice"] == "big-lama"
    assert np.all(output[mask == 0] == source[mask == 0])


def test_auto_skips_big_lama_when_comfyui_down(monkeypatch):
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    telea = np.full_like(source, 11)

    monkeypatch.setattr(inpaint, "comfyui_healthy", lambda cfg, probe=None: False)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_opencv_inpaint", lambda rgb, inner_mask, radius, method: telea)

    output, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto", "opencv_method": "telea"}},
        return_diagnostics=True,
    )

    assert backend == "opencv-telea"
    assert diagnostics["auto_skip_reason"] == "comfyui_down"
    assert diagnostics["comfyui_healthy"] is False


def test_multipass_inpaint_runs_coarse_then_fine(monkeypatch):
    source = np.full((40, 40, 3), 100, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, :] = 255
    calls = []

    def fake_single(rgb, inner_mask, cfg=None):
        calls.append(rgb.shape[:2])
        return np.full_like(rgb, len(calls) * 10), f"pass-{len(calls)}", {}

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)

    _, backend, diagnostics = inpaint.inpaint_array(
        source, mask,
        {"inpaint": {"mode": "opencv", "multipass_fraction": 0.10}},
        return_diagnostics=True,
    )

    assert diagnostics["inpaint_passes"] == 2
    assert calls[0] == (20, 20)
    assert calls[1] == (40, 40)
    assert backend == "pass-2"


def test_big_lama_compositing_only_replaces_masked_pixels(monkeypatch):
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    hallucination = np.zeros_like(source)

    monkeypatch.setattr(inpaint, "_simple_lama", lambda rgb, inner_mask: hallucination)

    output, backend = inpaint.inpaint_array(source, mask, {"inpaint": {"mode": "lama"}})

    assert backend == "big-lama"
    assert np.all(output[mask == 0] == source[mask == 0])
    assert np.all(output[mask > 0] == 0)
