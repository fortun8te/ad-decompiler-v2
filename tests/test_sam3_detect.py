"""CPU-safe tests for the SAM 3 image proposal wrapper.

The official 848M-parameter model is replaced by a tiny injected backend.  These tests cover
the integration contract, prompt sweep, per-residual box refinement, masks, and fallback.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import sam3_detect


def _image(tmp_path):
    path = tmp_path / "ad.png"
    Image.new("RGB", (120, 100), "white").save(path)
    return str(path)


class FakeSam3:
    name = "fake-facebookresearch/sam3"

    def __init__(self):
        self.text_calls = []
        self.box_calls = []
        self.size = None

    def set_image(self, image):
        self.size = image.size

    def predict_text(self, prompt):
        self.text_calls.append(prompt)
        mask = np.zeros((100, 120), dtype=bool)
        if prompt == "logo":
            mask[5:20, 8:38] = True
            return [{"mask": mask, "box": [8, 5, 38, 20], "score": 0.93}]
        return []

    def predict_box(self, box):
        self.box_calls.append(dict(box))
        mask = np.zeros((100, 120), dtype=bool)
        x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
        mask[y + 1 : y + h - 1, x + 1 : x + w - 1] = True
        return [{"mask": mask, "box": [x + 1, y + 1, x + w - 1, y + h - 1], "score": 0.88}]


def test_prompt_sweep_and_box_refines_every_residual(tmp_path):
    backend = FakeSam3()
    residual = [
        {"id": "R0", "kind": "shape", "box": {"x": 10, "y": 30, "w": 30, "h": 20}},
        {"id": "R1", "kind": "icon", "box": {"x": 70, "y": 55, "w": 18, "h": 18}},
    ]
    cfg = {
        "sam3": {
            "prompts": [
                {"prompt": "logo", "role": "logo", "kind": "icon"},
                {"prompt": "product", "role": "product", "kind": "photo-fragment"},
            ],
            "confidence": 0.4,
        }
    }

    result = sam3_detect.detect(
        _image(tmp_path), residual, cfg, run_dir=str(tmp_path), backend=backend
    )

    assert result["status"] == "ok"
    assert backend.text_calls == ["logo", "product"]
    assert len(backend.box_calls) == len(residual)
    assert len(result["elements"]) == 3  # one text proposal + one refinement per residual
    assert {e["provenance"]["mode"] for e in result["elements"]} == {
        "text-prompt",
        "box-refine",
    }
    assert next(e for e in result["elements"] if e["provenance"]["mode"] == "text-prompt")[
        "role"
    ] == "logo"
    for element in result["elements"]:
        assert element["score"] > 0
        assert os.path.exists(element["mask_path"])
        assert Image.open(element["mask_path"]).size == (120, 100)
    assert os.path.exists(tmp_path / "sam3.json")


def test_empty_backend_predictions_are_partial_not_false_success(tmp_path):
    class EmptySam3:
        name = "empty-sam3"

        def set_image(self, image):
            pass

        def predict_text(self, prompt):
            return []

        def predict_box(self, box):
            return []

    result = sam3_detect.detect(
        _image(tmp_path), [], {"sam3": {"prompts": ["product"]}}, backend=EmptySam3()
    )

    assert result["status"] == "partial"
    assert result["diagnostics"]["empty_model_evidence"] is True
    assert result["diagnostics"]["text_prompts_succeeded"] == 1
    assert "no accepted segmentation masks" in result["note"]


def test_no_model_falls_back_to_residual_and_still_saves_mask(tmp_path):
    residual = [
        {"id": "R0", "kind": "icon", "box": {"x": 20, "y": 25, "w": 12, "h": 10}}
    ]
    result = sam3_detect.detect(
        _image(tmp_path),
        residual,
        {"sam3": {"enabled": False, "prompts": []}},
        run_dir=str(tmp_path),
    )

    assert result["status"] == "fallback"
    assert result["engine"] == "residual-fallback"
    assert len(result["elements"]) == 1
    element = result["elements"][0]
    assert element["role"] == "icon"
    assert element["source"] == "residual-fallback"
    assert element["provenance"]["mode"] == "residual-fallback"
    assert os.path.exists(element["mask_path"])
    mask = np.asarray(Image.open(element["mask_path"])) > 0
    assert int(mask.sum()) == 120


def test_backend_failure_is_a_manifest_not_an_exception(tmp_path):
    class BrokenBackend:
        name = "broken"

        def set_image(self, image):
            raise RuntimeError("checkpoint incompatible")

    residual = [
        {"id": "R0", "kind": "shape", "box": {"x": 1, "y": 2, "w": 5, "h": 6}}
    ]
    result = sam3_detect.detect(
        _image(tmp_path), residual, {"sam3": {"prompts": []}}, str(tmp_path), BrokenBackend()
    )
    assert result["status"] == "fallback"
    assert "checkpoint incompatible" in result["note"]
    assert len(result["elements"]) == 1


def test_box_only_model_output_does_not_claim_rectangular_ownership(tmp_path):
    class BoxOnlyBackend:
        name = "box-only"

        def set_image(self, image):
            pass

        def predict_text(self, prompt):
            return [{"box": [20, 25, 32, 35], "score": 0.99}]

        def predict_box(self, box):
            return [{"box": [20, 25, 32, 35], "score": 0.99}]

    result = sam3_detect.detect(
        _image(tmp_path),
        [],
        {"sam3": {"prompts": ["icon"]}},
        run_dir=str(tmp_path),
        backend=BoxOnlyBackend(),
    )
    assert result["elements"] == []


def test_official_processor_state_shapes_are_normalized_to_predictions():
    """The real SAM 3 processor returns state with N x 1 x H x W masks."""
    state = {
        "masks": np.asarray([[[[1, 0], [0, 1]]]], dtype=bool),
        "boxes": np.asarray([[10.0, 20.0, 30.0, 40.0]]),
        "scores": np.asarray([0.91]),
    }

    predictions = sam3_detect._prediction_dicts(state, 2, 2)

    assert len(predictions) == 1
    assert predictions[0]["box"] == {"x": 10.0, "y": 20.0, "w": 20.0, "h": 20.0}
    assert predictions[0]["score"] == 0.91
    assert predictions[0]["mask"].tolist() == [[True, False], [False, True]]


def test_clip_box_off_canvas_origin_does_not_inflate_size():
    """A box with a negative/off-canvas origin must be clipped from its ORIGINAL x+w/y+h,
    not from the already-clipped x0/y0 plus the original w/h -- otherwise the box gets
    wider/taller than the true on-canvas extent."""
    # Spans x in [-20, -10): entirely off-canvas to the left. The true clipped width is 0
    # (the buggy version inflated this to w=10 by adding the original w back to x0=0).
    fully_off = sam3_detect._clip_box({"x": -20, "y": 0, "w": 10, "h": 10}, 100, 100)
    assert fully_off == {"x": 0, "y": 0, "w": 0, "h": 10}

    # Spans x in [-5, 15): true on-canvas extent is [0, 15) -> width 15, not 20.
    partly_off = sam3_detect._clip_box({"x": -5, "y": -5, "w": 20, "h": 20}, 100, 100)
    assert partly_off == {"x": 0, "y": 0, "w": 15, "h": 15}

    # Spans y in [90, 130) against height 100: true on-canvas extent is [90, 100) -> height 10.
    bottom_off = sam3_detect._clip_box({"x": 0, "y": 90, "w": 10, "h": 40}, 100, 100)
    assert bottom_off == {"x": 0, "y": 90, "w": 10, "h": 10}

    # Fully on-canvas box is unaffected.
    inside = sam3_detect._clip_box({"x": 5, "y": 5, "w": 10, "h": 10}, 100, 100)
    assert inside == {"x": 5, "y": 5, "w": 10, "h": 10}


def test_default_prompts_include_ui_ad_roles():
    roles = {spec["role"] for spec in sam3_detect._prompt_specs(None)}
    assert {"badge", "button", "card", "logo", "product", "icon",
            "avatar", "verified"} <= roles


def test_small_square_icon_accepted_below_generic_min_score(tmp_path):
    """A small, roughly-square avatar/badge/logo prediction is accepted below the generic
    text-prompt bar; a large one at the same score is not."""

    class SmallSquareBackend(FakeSam3):
        def predict_text(self, prompt):
            self.text_calls.append(prompt)
            if prompt == "verified badge":
                mask = np.zeros((100, 120), dtype=bool)
                mask[10:30, 40:60] = True  # 20x20 square, ~0.3% of canvas
                return [{"mask": mask, "box": [40, 10, 60, 30], "score": 0.34}]
            if prompt == "person":
                mask = np.zeros((100, 120), dtype=bool)
                mask[0:90, 0:110] = True  # large region at the same low score
                return [{"mask": mask, "box": [0, 0, 110, 90], "score": 0.34}]
            return []

    backend = SmallSquareBackend()
    cfg = {"sam3": {"prompts": [
        {"prompt": "verified badge", "role": "verified", "kind": "icon"},
        {"prompt": "person", "role": "person", "kind": "photo-fragment"},
    ], "confidence": 0.45}}
    result = sam3_detect.detect(_image(tmp_path), [], cfg, run_dir=str(tmp_path), backend=backend)
    roles = {e["role"] for e in result["elements"] if e["provenance"]["mode"] == "text-prompt"}
    assert "verified" in roles          # small square badge rescued at 0.34
    assert "person" not in roles        # large low-score region still rejected


def test_small_icon_pass_can_be_disabled(tmp_path):
    class SmallSquareBackend(FakeSam3):
        def predict_text(self, prompt):
            if prompt == "badge":
                mask = np.zeros((100, 120), dtype=bool)
                mask[10:30, 40:60] = True
                return [{"mask": mask, "box": [40, 10, 60, 30], "score": 0.34}]
            return []

    cfg = {"sam3": {"prompts": [{"prompt": "badge", "role": "badge", "kind": "icon"}],
                    "confidence": 0.45, "small_icon": {"enabled": False}}}
    result = sam3_detect.detect(_image(tmp_path), [], cfg, run_dir=str(tmp_path),
                                backend=SmallSquareBackend())
    assert [e for e in result["elements"] if e["provenance"]["mode"] == "text-prompt"] == []


def test_box_refine_accepts_lower_score_when_residuals_exist(tmp_path):
    class LowScoreBoxBackend(FakeSam3):
        def predict_box(self, box):
            self.box_calls.append(dict(box))
            mask = np.zeros((100, 120), dtype=bool)
            x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
            mask[y : y + h, x : x + w] = True
            return [{"mask": mask, "box": [x, y, x + w, y + h], "score": 0.34}]

    backend = LowScoreBoxBackend()
    residual = [{"id": "R0", "kind": "shape", "box": {"x": 10, "y": 30, "w": 30, "h": 20}}]
    result = sam3_detect.detect(
        _image(tmp_path),
        residual,
        {"sam3": {"prompts": [], "confidence": 0.45}},
        run_dir=str(tmp_path),
        backend=backend,
    )
    refined = next(e for e in result["elements"] if e["provenance"].get("residual_id") == "R0")
    assert refined["provenance"]["mode"] == "box-refine"
    assert result["thresholds"]["box_refine_min_score"] == 0.32


def test_union_residual_guarantees_missing_box_refine(tmp_path):
    class SkipBoxBackend(FakeSam3):
        def predict_box(self, box):
            return []

    backend = SkipBoxBackend()
    residual = [
        {"id": "R0", "kind": "shape", "box": {"x": 10, "y": 30, "w": 30, "h": 20}},
        {"id": "R1", "kind": "icon", "box": {"x": 70, "y": 55, "w": 18, "h": 18}},
    ]
    result = sam3_detect.detect(
        _image(tmp_path),
        residual,
        {"sam3": {"prompts": []}},
        run_dir=str(tmp_path),
        backend=backend,
    )
    covered = {e["provenance"].get("residual_id") for e in result["elements"]}
    assert covered == {"R0", "R1"}
    assert all(e["provenance"].get("residual_id") for e in result["elements"])


def test_rgba_residual_fallback_uses_alpha_not_transparent_white_rgb(tmp_path):
    rgba = np.full((100, 120, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = 0
    rgba[20:30, 10:25, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "cutout.png")
    residual = [{"id": "R0", "kind": "icon", "box": {"x": 0, "y": 0, "w": 120, "h": 100},
                 "mask_path": str(tmp_path / "cutout.png")}]
    result = sam3_detect.detect(_image(tmp_path), residual,
                                {"sam3": {"enabled": False}}, str(tmp_path))
    mask = np.asarray(Image.open(result["elements"][0]["mask_path"])) > 0
    assert int(mask.sum()) == 150


def test_box_refine_rejects_mask_that_snaps_to_unrelated_large_region(tmp_path):
    class BadSnap(FakeSam3):
        def predict_box(self, box):
            mask = np.zeros((100, 120), dtype=bool)
            mask[:, :80] = True
            return [{"mask": mask, "score": .99}]

    residual = [{"id": "R0", "kind": "icon", "box": {"x": 95, "y": 70, "w": 10, "h": 10}}]
    result = sam3_detect.detect(_image(tmp_path), residual,
                                {"sam3": {"prompts": []}}, str(tmp_path), BadSnap())
    element = result["elements"][0]
    assert element["source"] == "residual-fallback"
    assert element["area"] == 100
