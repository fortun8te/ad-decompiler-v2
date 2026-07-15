import json

import pytest

from src import schema


def test_dump_replaces_checkpoint_atomically(tmp_path):
    path = tmp_path / "stage.json"
    schema.dump({"version": 1}, str(path))
    schema.dump({"version": 2, "items": [1, 2]}, str(path))

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2, "items": [1, 2]}
    assert not list(tmp_path.glob(".stage.json.*.tmp"))


def test_dump_keeps_previous_checkpoint_when_serialization_fails(tmp_path):
    path = tmp_path / "stage.json"
    schema.dump({"version": 1}, str(path))

    with pytest.raises(TypeError):
        schema.dump({"bad": object()}, str(path))

    assert schema.load(str(path)) == {"version": 1}
    assert not list(tmp_path.glob(".stage.json.*.tmp"))


# ── F11: canonical meta.fallback contract ──────────────────────────────────────────
def test_fallback_kind_normalizes_every_legacy_spelling():
    assert schema.fallback_kind({"fallback": "raster-slice"}) == "raster-slice"
    assert schema.fallback_kind({"fallback": "raster_slice"}) == "raster-slice"
    assert schema.fallback_kind({"fallback": "plate-passthrough"}) == "plate-passthrough"
    # Legacy True (a text->image / masked-pixel substitution) and any other explicit
    # string both normalize to the explained-but-non-native "fidelity-image".
    assert schema.fallback_kind({"fallback": True}) == "fidelity-image"
    assert schema.fallback_kind({"fallback": "masked-pixel"}) == "fidelity-image"
    # No marker at all -> a normal native layer.
    assert schema.fallback_kind({}) is None
    assert schema.fallback_kind({"fallback": False}) is None
    assert schema.fallback_kind({"fallback": None}) is None
    assert schema.fallback_kind(None) is None


def test_is_raster_slice_only_true_for_slice():
    assert schema.is_raster_slice({"fallback": "raster-slice"}) is True
    assert schema.is_raster_slice({"fallback": True}) is False
    assert schema.is_raster_slice({"fallback": "plate-passthrough"}) is False
    assert schema.is_raster_slice({}) is False


def test_is_editable_leaf_rejects_groups_images_and_fallbacks():
    assert schema.is_editable_leaf({"type": "text"}) is True
    assert schema.is_editable_leaf({"type": "shape"}) is True
    # A native type carrying a fallback marker only looks native.
    assert schema.is_editable_leaf({"type": "text", "meta": {"fallback": True}}) is False
    # Groups are containers; images are never editable leaves.
    assert schema.is_editable_leaf({"type": "group"}) is False
    assert schema.is_editable_leaf({"type": "image"}) is False
    # Works off `target` too (pre-compile candidate dicts).
    assert schema.is_editable_leaf({"target": "text"}) is True


# ── F6: broken-text-render combined slice gate ─────────────────────────────────────
def test_f6_broken_text_render_slices_but_plausible_font_stays_editable():
    # CODIA-PARITY POLICY: by default text NEVER fails the slice gate ("wrong Inter
    # beats baked pixels"); the F6 forensic gates below live behind the
    # text_slice_gate_enabled flag for tooling only. Default path first:
    assert schema.raster_slice_failures(
        {"type": "text", "region_ssim": 0.137, "ink_iou": 0.389, "ink_excess": 0.715},
        schema.raster_slice_thresholds({}),
    ) == []

    thresholds = schema.raster_slice_thresholds(
        {"fallback": {"text_slice_gate_enabled": True}})

    def is_text(**row):
        row.setdefault("type", "text")
        return schema.raster_slice_failures(row, thresholds)

    # Benchmark 009 c_B4 / c_B7: wrong-class fonts, genuinely garbage renders.
    assert is_text(region_ssim=0.137, ink_iou=0.389, ink_excess=0.715)
    assert is_text(region_ssim=0.261, ink_iou=0.378, ink_excess=0.728)

    # Plausible same-class font, imperfect but acceptable: must STAY editable text.
    # (a) structurally close render, roughly-aligned ink (low excess) -> keep.
    assert not is_text(region_ssim=0.55, ink_iou=0.62, ink_excess=0.30)
    # (b) low crop SSIM but ink still roughly aligned (repairable offset) -> keep.
    assert not is_text(region_ssim=0.14, ink_iou=0.45, ink_excess=0.40)
    # (c) 009 body copy that should remain editable (c_B0/c_B1/c_B3, c_B13/c_B14).
    assert not is_text(region_ssim=0.206, ink_iou=0.460, ink_excess=0.308)
    assert not is_text(region_ssim=0.127, ink_iou=0.352, ink_excess=0.511)
    assert not is_text(region_ssim=0.120, ink_iou=0.350, ink_excess=0.614)
