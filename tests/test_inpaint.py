import numpy as np
import pytest

from src import inpaint


def test_check_backends_reports_opencv_fallback(monkeypatch):
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: False)
    status = inpaint.check_backends({"inpaint": {"mode": "auto"}})
    assert status["big_lama"]["ok"] is False
    assert status["fallback_ready"] is True
    assert status["ready"] is False


def test_active_model_requirement_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_simple_lama", lambda *_args: (_ for _ in ()).throw(RuntimeError("model failed")))
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    with pytest.raises(RuntimeError, match="model failed"):
        inpaint.inpaint_array(source, mask, {"runtime": {"require_active_models": True}})


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


def test_text_ink_mask_does_not_fallback_to_a_solid_body_text_rectangle():
    """High-coverage contrast on a textured plate must stay pixel-shaped."""
    rgb = np.zeros((24, 48, 3), dtype=np.uint8)
    for x in range(48):
        rgb[:, x] = (150 + x * 2, 150 + x * 2, 150 + x * 2)
    rgb[8:16, 18:30] = 20

    mask = inpaint.text_ink_mask(rgb, {"x": 8, "y": 4, "w": 32, "h": 16})

    local = mask[4:20, 8:40]
    assert np.count_nonzero(local) > 0
    assert np.count_nonzero(local) < local.size


def test_overlay_text_mask_fails_closed_without_a_glyph_signal():
    rgb = np.full((20, 40, 3), 128, dtype=np.uint8)
    mask = inpaint.text_ink_mask(
        rgb, {"x": 5, "y": 4, "w": 30, "h": 12}, allow_box_fallback=False,
    )
    assert np.count_nonzero(mask) == 0


def test_overlay_text_mask_is_constrained_to_quad_not_rectangle():
    rgb = np.full((30, 50, 3), 220, dtype=np.uint8)
    rgb[10:16, 15:35] = 20
    quad = [[15, 10], [35, 10], [35, 16], [15, 16]]
    mask = inpaint.text_ink_mask(
        rgb, {"x": 5, "y": 5, "w": 40, "h": 16}, quad, allow_box_fallback=False,
    )
    assert np.count_nonzero(mask[5:10, 5:45]) == 0
    assert np.count_nonzero(mask[10:16, 15:35]) > 0


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


def test_resolve_mask_dilate_has_separate_overlay_text_halo():
    cfg = {"inpaint": {"mask_dilate": {"default": 1, "text": 2, "overlay_text": 4}}}
    assert inpaint.resolve_mask_dilate(
        {"target": "text", "meta": {"overlay_text": True}}, cfg
    ) == 4


def test_resolve_mask_dilate_defaults_are_role_aware_for_overlay_text():
    assert inpaint.resolve_mask_dilate(
        {"target": "text", "meta": {"overlay_text": True, "role": "headline"}},
        {"reconstruct": {"mask_dilate": 2}},
    ) == 5
    assert inpaint.resolve_mask_dilate(
        {"target": "text", "meta": {"overlay_text": True, "role": "body"}},
        {"reconstruct": {"mask_dilate": 2}},
    ) == 4
    assert inpaint.resolve_mask_dilate(
        {"target": "text", "meta": {"role": "body"}},
        {"reconstruct": {"mask_dilate": 2}},
    ) == 2


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


def test_auto_uses_big_lama_even_when_comfyui_down(monkeypatch):
    # Big-LaMa is a local pip package; ComfyUI (Qwen's advisory backend) being offline
    # must never downgrade the background plate to OpenCV. The old comfy-gated behavior
    # silently failed all 16 real benchmark images as "inpaint-unavailable" runtime
    # violations on the RTX box, where ComfyUI is routinely off.
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    lama_out = np.full_like(source, 42)

    monkeypatch.setattr(inpaint, "comfyui_healthy", lambda cfg, probe=None: False)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_simple_lama", lambda rgb, inner_mask: lama_out)

    output, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto"}},
        return_diagnostics=True,
    )

    assert backend == "big-lama"
    assert "auto_skip_reason" not in diagnostics
    assert diagnostics["comfyui_healthy"] is False  # recorded, but not load-bearing


def test_auto_skips_big_lama_only_when_lama_itself_is_missing(monkeypatch):
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    telea = np.full_like(source, 11)

    monkeypatch.setattr(inpaint, "comfyui_healthy", lambda cfg, probe=None: True)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: False)
    monkeypatch.setattr(inpaint, "_opencv_inpaint", lambda rgb, inner_mask, radius, method: telea)

    output, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto", "opencv_method": "telea"}},
        return_diagnostics=True,
    )

    assert backend == "opencv-telea"
    assert diagnostics["auto_skip_reason"] == "big_lama_missing"


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


def test_big_lama_size_mismatch_is_resized_not_crashed(monkeypatch):
    # Big-LaMa pads to a multiple of 8 and can return a plate a few px off from the input;
    # compositing that against the original mask previously raised IndexError and crashed
    # the run (3/16 real benchmark images). The output must be snapped back to input HxW.
    source = np.full((338, 344, 3), 100, dtype=np.uint8)
    mask = np.zeros((338, 344), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    oversized = np.full((344, 352, 3), 50, dtype=np.uint8)  # padded to mult of 8

    monkeypatch.setattr(inpaint, "comfyui_healthy", lambda cfg, probe=None: True)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_simple_lama", lambda rgb, m: oversized)

    out, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto"}}, return_diagnostics=True,
    )

    assert backend == "big-lama"
    assert out.shape == (338, 344, 3)
    assert diagnostics.get("resized_from") == "352x344"
    # pixels outside the mask must stay byte-identical to the source
    assert np.array_equal(out[0, 0], source[0, 0])


# ── flux_comfy backend selection ──────────────────────────────────────────────────────
def _flux_scene():
    source = np.full((8, 8, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    return source, mask


def test_flux_comfy_backend_used_when_available(monkeypatch):
    source, mask = _flux_scene()
    filled = np.full_like(source, 7)
    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", lambda rgb, m, cfg: filled)

    out, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "flux_comfy", "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )

    assert backend == "flux-comfy"
    assert diagnostics["backend_choice"] == "flux-comfy"
    assert np.all(out[mask == 0] == source[mask == 0])  # unmasked pixels untouched
    assert np.all(out[mask > 0] == 7)


def test_flux_comfy_used_in_auto_only_when_opted_in(monkeypatch):
    source, mask = _flux_scene()
    filled = np.full_like(source, 9)
    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", lambda rgb, m, cfg: filled)

    out, backend, _ = inpaint.inpaint_array(
        source, mask,
        {"inpaint": {"mode": "auto", "comfy": {"enabled": True}, "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )
    assert backend == "flux-comfy"


def test_flux_comfy_not_attempted_in_plain_auto(monkeypatch):
    source, mask = _flux_scene()
    called = []

    def spy(*args, **kwargs):
        called.append(1)
        return None

    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", spy)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: False)
    monkeypatch.setattr(inpaint, "_opencv_inpaint", lambda rgb, m, r, method: np.full_like(source, 5))

    inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "auto", "opencv_method": "telea",
                                   "multipass_fraction": 1.0}},
    )
    assert called == []  # flux is opt-in; plain auto must not touch ComfyUI


def test_flux_comfy_falls_back_to_big_lama_when_comfy_down(monkeypatch):
    source, mask = _flux_scene()
    lama = np.full_like(source, 42)
    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", lambda *a, **k: None)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: True)
    monkeypatch.setattr(inpaint, "_simple_lama", lambda rgb, m: lama)

    out, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "flux_comfy", "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )

    assert backend == "big-lama"
    assert diagnostics["flux_comfy"] == "unavailable"
    assert np.all(out[mask > 0] == 42)


def test_flux_comfy_falls_back_to_opencv_when_comfy_and_lama_down(monkeypatch):
    source, mask = _flux_scene()
    telea = np.full_like(source, 3)
    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", lambda *a, **k: None)
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: False)
    monkeypatch.setattr(inpaint, "_opencv_inpaint", lambda rgb, m, r, method: telea)

    out, backend, _ = inpaint.inpaint_array(
        source, mask,
        {"inpaint": {"mode": "flux_comfy", "opencv_method": "telea", "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )
    assert backend == "opencv-telea"
    assert np.all(out[mask > 0] == 3)


def test_flux_comfy_required_without_fallback_raises(monkeypatch):
    source, mask = _flux_scene()
    monkeypatch.setattr(inpaint, "_flux_comfy_inpaint", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="flux_comfy"):
        inpaint.inpaint_array(
            source, mask,
            {"inpaint": {"mode": "flux_comfy", "allow_fallback": False,
                         "comfy": {"required": True}, "multipass_fraction": 1.0}},
        )


def test_role_aware_overlap_is_owned_by_large_object_pass(tmp_path, monkeypatch):
    from PIL import Image
    source = np.full((20, 30, 3), 100, dtype=np.uint8)
    Image.fromarray(source).save(tmp_path / "source.png")
    text = np.zeros((20, 30), dtype=np.uint8); text[8:12, 12:18] = 255
    large = np.zeros((20, 30), dtype=np.uint8); large[5:15, 5:25] = 255
    calls = []

    def fake(rgb, mask, cfg=None, return_diagnostics=False):
        calls.append(mask.copy())
        out = rgb.copy(); out[mask > 0] = 7
        result = (out, "fake", {})
        return result if return_diagnostics else result[:2]

    monkeypatch.setattr(inpaint, "inpaint_array", fake)
    result = inpaint.inpaint_role_aware(
        str(tmp_path / "source.png"), {"text": text, "large": large},
        str(tmp_path / "out.png"), {},
    )
    assert [part["role"] for part in result["parts"]] == ["large"]
    assert np.array_equal(calls[0], large)


def test_regional_groups_contained_text_with_product_and_preserves_union():
    product = np.zeros((80, 100), dtype=np.uint8)
    product[20:70, 30:80] = 255
    text = np.zeros_like(product)
    text[35:43, 42:68] = 255
    observations = [
        {"id": "product", "target": "image", "role": "product", "mask_array": product},
        {"id": "label", "target": "text", "role": "body", "mask_array": text},
    ]
    union = np.maximum(product, text)

    regions = inpaint.build_inpaint_regions((100, 80), observations, union)

    assert len(regions) == 1
    assert set(regions[0]["ids"]) == {"product", "label"}
    assert np.array_equal(regions[0]["mask"], union)


def test_regional_does_not_bridge_nearby_unrelated_candidates():
    left = np.zeros((80, 120), dtype=np.uint8); left[20:40, 10:45] = 255
    right = np.zeros_like(left); right[20:40, 52:90] = 255  # seven-pixel gap
    union = np.maximum(left, right)
    observations = [
        {"id": "headline", "target": "text", "role": "headline", "mask_array": left},
        {"id": "product", "target": "image", "role": "product", "mask_array": right},
    ]

    regions = inpaint.build_inpaint_regions((120, 80), observations, union)

    assert len(regions) == 2
    rebuilt = np.zeros_like(union)
    for region in regions:
        rebuilt = np.maximum(rebuilt, region["mask"])
    assert np.array_equal(rebuilt, union)


def test_regional_flat_plate_uses_analytic_fill_and_keeps_exterior_exact(tmp_path):
    from PIL import Image
    source = np.full((96, 128, 3), (18, 24, 30), dtype=np.uint8)
    source[35:60, 45:85] = (240, 30, 10)
    mask = np.zeros((96, 128), dtype=np.uint8); mask[35:60, 45:85] = 255
    Image.fromarray(source).save(tmp_path / "source.png")
    cfg = {"inpaint": {"regional": {"enabled": True, "min_context": 12,
                                      "max_context": 16, "min_crop": 64}}}

    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"),
        [{"id": "badge", "target": "shape", "role": "badge", "mask_array": mask}],
        mask, str(tmp_path / "out.png"), cfg,
    )
    output = np.asarray(Image.open(tmp_path / "out.png").convert("RGB"))

    assert result["backend_counts"] == {"analytic-affine": 1}
    assert np.array_equal(output[mask == 0], source[mask == 0])
    assert np.max(np.abs(output[mask > 0].astype(int) - np.array([18, 24, 30]))) <= 1


def test_regional_ui_chrome_prefers_analytic_over_flux_on_flat_archetype(tmp_path, monkeypatch):
    """009-style dark UI: text/badge holes must not spend Flux even with a noisier ring."""
    from PIL import Image
    source = np.full((120, 160, 3), 12, dtype=np.uint8)
    source[40:70, 50:110] = (240, 240, 240)  # text ink
    # Mild ring noise (panel edge) that used to push residual over the strict flat gate.
    source[36:40, 48:112] = 28
    mask = np.zeros((120, 160), dtype=np.uint8); mask[40:70, 50:110] = 255
    Image.fromarray(source).save(tmp_path / "source.png")
    calls = {"flux": 0}

    def fake_single(rgb, inner_mask, cfg):
        calls["flux"] += 1
        return np.full_like(rgb, 7), "flux-comfy", {"backend_choice": "flux-comfy"}

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)
    cfg = {
        "scene": {"archetype": "social_screenshot"},
        "inpaint": {
            "mode": "auto", "comfy": {"enabled": True},
            "regional": {
                "enabled": True, "min_context": 12, "max_context": 16, "min_crop": 64,
                "flat_residual_p90": 4, "flat_gradient_p90": 4,
                "ui_analytic_residual_p90": 40, "ui_analytic_dominant_fraction": 0.35,
                "flux_residual_p90": -1, "flux_gradient_p90": -1,
                "flux_max_canvas_fraction": 0.5,
            },
        },
    }
    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"),
        [{"id": "t0", "target": "text", "role": "headline", "mask_array": mask}],
        mask, str(tmp_path / "out.png"), cfg,
    )
    assert calls["flux"] == 0
    assert result["regions"][0]["route"] == "analytic-affine"
    assert result["backend_counts"].get("analytic-affine") == 1


def test_regional_dominant_flat_plate_ignores_foreground_ring_contamination(tmp_path):
    from PIL import Image
    source = np.full((100, 120, 3), 245, dtype=np.uint8)
    source[25:75, 40:80] = (90, 35, 15)
    # Deliberately incomplete product mask leaves a contaminated strip in the ring.
    mask = np.zeros((100, 120), dtype=np.uint8); mask[30:70, 45:75] = 255
    Image.fromarray(source).save(tmp_path / "source.png")
    cfg = {"inpaint": {"regional": {"enabled": True, "min_context": 16,
        "max_context": 20, "min_crop": 64, "dominant_plate_fraction": 0.55}}}

    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"),
        [{"id": "product", "target": "image", "role": "product", "mask_array": mask}],
        mask, str(tmp_path / "out.png"), cfg,
    )
    record = result["regions"][0]

    assert record["complexity"]["model"] == "dominant-flat-rgb"
    assert record["route"] == "analytic-affine"


def test_regional_flux_is_crop_local_aligned_and_single_pass(tmp_path, monkeypatch):
    from PIL import Image
    yy, xx = np.mgrid[0:123, 0:157]
    source = np.stack([(xx * 11) % 255, (yy * 17) % 255, ((xx + yy) * 7) % 255], axis=-1).astype(np.uint8)
    mask = np.zeros((123, 157), dtype=np.uint8); mask[42:73, 61:96] = 255
    Image.fromarray(source).save(tmp_path / "source.png")
    calls = []

    def fake_single(rgb, inner_mask, cfg):
        calls.append((rgb.shape, cfg["inpaint"]["mode"]))
        return np.full_like(rgb, 7), "flux-comfy", {"backend_choice": "flux-comfy"}

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)
    cfg = {"inpaint": {"mode": "auto", "comfy": {"enabled": True}, "regional": {
        "enabled": True, "min_context": 10, "max_context": 12, "min_crop": 64,
        "flat_residual_p90": -1, "flat_gradient_p90": -1,
        "flux_residual_p90": -1, "flux_gradient_p90": -1,
        "flux_max_canvas_fraction": 0.10,
    }}}

    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"),
        [{"id": "photo-object", "target": "image", "role": "product", "mask_array": mask}],
        mask, str(tmp_path / "out.png"), cfg,
    )
    output = np.asarray(Image.open(tmp_path / "out.png").convert("RGB"))

    assert len(calls) == 1
    assert calls[0][1] == "flux-comfy"
    assert calls[0][0][0] % 16 == 0 and calls[0][0][1] % 16 == 0
    assert result["regions"][0]["route"] == "flux-comfy"
    assert np.array_equal(output[mask == 0], source[mask == 0])
    assert np.all(output[mask > 0] == 7)


def test_regional_large_complex_hole_routes_to_lama_not_flux(tmp_path, monkeypatch):
    from PIL import Image
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    source[:, ::2] = 255
    mask = np.zeros((100, 100), dtype=np.uint8); mask[20:80, 20:80] = 255
    Image.fromarray(source).save(tmp_path / "source.png")
    calls = []

    def fake_single(rgb, inner_mask, cfg):
        calls.append(cfg["inpaint"]["mode"])
        return np.full_like(rgb, 127), "big-lama", {"backend_choice": "big-lama"}

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)
    cfg = {"inpaint": {"mode": "auto", "comfy": {"enabled": True}, "regional": {
        "enabled": True, "min_context": 4, "max_context": 8, "min_crop": 32,
        "flat_residual_p90": -1, "flat_gradient_p90": -1,
        "flux_residual_p90": -1, "flux_gradient_p90": -1,
        "flux_max_canvas_fraction": 0.025,
    }}}

    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"),
        [{"id": "product", "target": "image", "role": "product", "mask_array": mask}],
        mask, str(tmp_path / "out.png"), cfg,
    )

    assert calls == ["big-lama"]
    assert result["backend_counts"] == {"big-lama": 1}


def test_regional_mixed_text_and_icon_routes_to_lama_not_flux(tmp_path, monkeypatch):
    from PIL import Image
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    source[:, ::2] = 255
    text = np.zeros((100, 120), dtype=np.uint8); text[35:55, 30:85] = 255
    icon = np.zeros_like(text); icon[37:53, 35:47] = 255
    union = np.maximum(text, icon)
    Image.fromarray(source).save(tmp_path / "source.png")
    calls = []

    def fake_single(rgb, inner_mask, cfg):
        calls.append(cfg["inpaint"]["mode"])
        return np.full_like(rgb, 127), "big-lama", {"backend_choice": "big-lama"}

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)
    cfg = {"inpaint": {"mode": "auto", "comfy": {"enabled": True}, "regional": {
        "enabled": True, "min_context": 4, "max_context": 8, "min_crop": 32,
        "flat_residual_p90": -1, "flat_gradient_p90": -1,
        "flux_residual_p90": -1, "flux_gradient_p90": -1,
        "flux_max_canvas_fraction": 0.10,
    }}}
    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"), [
            {"id": "copy", "target": "text", "role": "body", "mask_array": text},
            {"id": "bullet", "target": "icon", "role": "icon", "mask_array": icon},
        ], union, str(tmp_path / "out.png"), cfg,
    )
    assert calls == ["big-lama"]
    assert result["regions"][0]["route"] == "big-lama"


def test_opencv_candidate_ranking_uses_residue_when_seams_tie(monkeypatch):
    source = np.full((16, 16, 3), 120, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8); mask[4:12, 4:12] = 255
    telea = np.full_like(source, 90)
    ns = np.full_like(source, 150)

    monkeypatch.setattr(
        inpaint, "_opencv_inpaint",
        lambda _rgb, _mask, _radius, method: telea if method == "telea" else ns,
    )
    monkeypatch.setattr(inpaint, "_seam_energy", lambda *_args: 1.0)

    def fake_metrics(_source, candidate, _mask):
        # Same seam, but Telea leaves a text/object-like high-frequency residue.
        return {"texture": 0.0, "structure": 0.0,
                "residue": 10.0 if candidate[6, 6, 0] == 90 else 0.0}

    monkeypatch.setattr(inpaint, "candidate_metrics", fake_metrics)
    _out, backend, diagnostics = inpaint.inpaint_array(
        source, mask, {"inpaint": {"mode": "opencv", "opencv_method": "auto"}},
        return_diagnostics=True,
    )

    assert backend == "opencv-ns"
    assert diagnostics["telea_quality"]["seam"] == diagnostics["ns_quality"]["seam"]
    assert diagnostics["telea_quality"]["residue"] > diagnostics["ns_quality"]["residue"]


def test_cpu_quality_metrics_penalize_compact_high_frequency_residue():
    source = np.full((24, 24, 3), 120, dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8); mask[5:19, 5:19] = 255
    clean = source.copy()
    noisy = source.copy()
    yy, xx = np.mgrid[5:19, 5:19]
    noisy[5:19, 5:19] = np.where(((xx + yy) % 2)[..., None] > 0, 20, 220)

    clean_metrics = inpaint.candidate_metrics(source, clean, mask)
    noisy_metrics = inpaint.candidate_metrics(source, noisy, mask)

    assert noisy_metrics["residue"] > clean_metrics["residue"]
    assert noisy_metrics["texture"] > clean_metrics["texture"]


def test_strict_acceptance_blocks_automatic_opencv_fallback(monkeypatch):
    source = np.full((12, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8); mask[3:9, 3:9] = 255
    monkeypatch.setattr(inpaint, "_big_lama_available", lambda: False)

    with pytest.raises(RuntimeError, match="blocks the OpenCV fallback"):
        inpaint.inpaint_array(
            source, mask,
            {"inpaint": {"mode": "auto", "strict_acceptance": True,
                         "multipass_fraction": 1.0}},
        )


def test_explicit_opencv_has_clear_routing_metadata_even_in_strict_mode():
    source = np.full((12, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8); mask[3:9, 3:9] = 255

    _out, backend, diagnostics = inpaint.inpaint_array(
        source, mask,
        {"inpaint": {"mode": "opencv", "strict_acceptance": True,
                     "opencv_method": "telea", "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )

    assert backend == "opencv-telea"
    assert diagnostics["backend_class"] == "fallback"
    assert diagnostics["backend_route"] == {
        "requested": "opencv", "selected": "opencv-telea",
        "selected_class": "deterministic-fallback", "strict_acceptance": True,
        "opencv_fallback_used": False, "fallback_reason": None,
    }


def test_powerpaint_status_is_honest_about_unvalidated_runtime():
    status = inpaint.check_backends({"inpaint": {"mode": "powerpaint"}})

    assert status["ready"] is False
    assert status["powerpaint"]["runtime_validated"] is False
    assert status["powerpaint"]["importable"] is False


def test_powerpaint_adapter_seam_routes_and_records_backend(monkeypatch):
    source = np.full((12, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8); mask[3:9, 3:9] = 255
    monkeypatch.setattr(inpaint, "_powerpaint_inpaint", lambda rgb, _mask, _cfg: np.full_like(rgb, 17))

    out, backend, diagnostics = inpaint.inpaint_array(
        source, mask,
        {"inpaint": {"mode": "powerpaint", "multipass_fraction": 1.0}},
        return_diagnostics=True,
    )

    assert backend == "powerpaint"
    assert diagnostics["backend_route"]["selected_class"] == "active-model"
    assert np.all(out[mask > 0] == 17)
    assert np.all(out[mask == 0] == source[mask == 0])


def test_regional_later_region_uses_original_source_not_previous_generated_plate(tmp_path, monkeypatch):
    from PIL import Image

    source = np.full((96, 96, 3), 100, dtype=np.uint8)
    first = np.zeros((96, 96), dtype=np.uint8); first[10:34, 10:34] = 255
    second = np.zeros((96, 96), dtype=np.uint8); second[56:70, 56:70] = 255
    source[first > 0] = (220, 40, 30)
    source[second > 0] = (30, 40, 220)
    Image.fromarray(source).save(tmp_path / "source.png")
    calls = []

    def fake_single(rgb, inner_mask, _cfg):
        calls.append(rgb.copy())
        return np.full_like(rgb, 7), "fake-active", {
            "backend_route": {"selected": "fake-active"}, "backend_class": "active",
        }

    monkeypatch.setattr(inpaint, "_inpaint_single_pass", fake_single)
    cfg = {"inpaint": {"mode": "lama", "regional": {
        "enabled": True, "min_context": 1, "max_context": 1, "min_crop": 96,
        "flat_residual_p90": -1, "flat_gradient_p90": -1,
    }}}
    union = np.maximum(first, second)
    result = inpaint.inpaint_regional(
        str(tmp_path / "source.png"), [
            {"id": "first", "target": "shape", "mask_array": first},
            {"id": "second", "target": "shape", "mask_array": second},
        ], union, str(tmp_path / "out.png"), cfg,
    )

    assert len(calls) == 2
    # The first region has already been composited to 7 in the destination plate. The
    # second backend input must still contain the original red first region.
    assert np.all(calls[1][first > 0] == np.array([220, 40, 30]))
    assert all(record["context_source"] == "original-source-only" for record in result["regions"])


def test_regional_global_context_uses_full_original_canvas():
    mask = np.zeros((60, 80), dtype=np.uint8); mask[20:30, 35:45] = 255

    bounds, _padding, context = inpaint._regional_crop(
        mask, {"inpaint": {"regional": {"context_mode": "global", "alignment": 16}}},
    )

    assert bounds == (0, 0, 80, 60)
    assert context == 80
