"""Always-raster chrome cutouts + the empty-asset materialization ban.

Policy (locked): badges / seals / chips / pills / starbursts ship as exact IMAGE
cutouts — no native ellipse/pill/star shells, no editable badge copy.

Still covered from postfix-benchmark-4:
  * 104/107/021 empty asset groups + blank 8KB ghost PNGs
  * 088 c_E011 "unexplained-raster-fallback" on a nested plate-passthrough drop
"""
import numpy as np
import pytest
from PIL import Image

from src import build_design_json


def _plate(tmp_path, size=(400, 400), color="white"):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", size, color).save(background)
    return background


def _source(tmp_path, size=(400, 400), color=(240, 240, 240)):
    """A normalized.png the materialization gate can read."""
    src = tmp_path / "normalized.png"
    Image.new("RGB", size, color).save(src)
    return src


def _opaque_cutout(tmp_path, name, size, color):
    path = tmp_path / name
    Image.new("RGBA", size, (*color, 255)).save(path)
    return str(path)


# ── Always-raster chrome: no native shell, exact source crop only ───────────────────
def test_circular_badge_keeps_exact_raster_not_native_ellipse(tmp_path):
    _plate(tmp_path)
    _source(tmp_path)
    cutout = _opaque_cutout(tmp_path, "badge.png", (87, 87), (12, 162, 177))
    tree = [{
        "id": "c_E005", "target": "group", "box": {"x": 288, "y": 408, "w": 87, "h": 87},
        "shape_kind": "ellipse", "fill": {"kind": "flat", "color": "#0ca2b1"},
        "src": cutout,
        "meta": {
            "role": "badge", "shell_raster_chip": True, "baked_badge_text": True,
        },
        "children": [],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    group = next(layer for layer in doc.layers if layer.id == "c_E005")
    assert not any(c.meta.get("rebuilt_from") for c in group.children), \
        "chrome-as-raster badges must not emit native ellipse/pill shells"
    hosts = [c for c in group.children if c.meta.get("preserved_host_raster")]
    assert len(hosts) == 1
    assert hosts[0].type == "image"
    assert group.radius is None


def test_shell_raster_chip_skips_native_shell_shape_helper():
    assert build_design_json._native_shell_shape(
        {"id": "b", "shape_kind": "ellipse",
         "fill": {"kind": "flat", "color": "#0ca2b1"},
         "meta": {"role": "badge", "shell_raster_chip": True, "baked_badge_text": True}},
        {"x": 0, "y": 0, "w": 80, "h": 80}, 1.0) is None
    assert build_design_json._native_shell_shape(
        {"id": "b2", "shape_kind": "rect",
         "fill": {"kind": "flat", "color": "#1a1a1a"},
         "meta": {"role": "badge", "chrome_as_raster": True, "plate_shell": True}},
        {"x": 0, "y": 0, "w": 114, "h": 50}, 1.0) is None


def test_wide_chip_keeps_exact_raster_not_native_pill(tmp_path):
    _plate(tmp_path, (800, 900))
    _source(tmp_path, (800, 900))
    cutout = _opaque_cutout(tmp_path, "chip.png", (114, 50), (26, 26, 26))
    tree = [{
        "id": "c_E006", "target": "group", "box": {"x": 413, "y": 719, "w": 114, "h": 50},
        "shape_kind": "rect", "fill": {"kind": "flat", "color": "#1a1a1a"},
        "src": cutout,
        "meta": {"role": "badge", "shell_raster_chip": True, "baked_badge_text": True},
        "children": [],
    }]
    doc = build_design_json.build(tree, {"w": 800, "h": 900}, str(tmp_path),
                                  base_src=str(_plate(tmp_path, (800, 900))))
    group = next(layer for layer in doc.layers if layer.id == "c_E006")
    assert not any("shell" in str(c.id) for c in group.children)
    assert any(c.meta.get("preserved_host_raster") for c in group.children)


def test_scalloped_seal_keeps_exact_raster_not_starburst_path(tmp_path):
    _plate(tmp_path, (400, 400))
    cutout = _opaque_cutout(tmp_path, "seal.png", (252, 252), (46, 181, 126))
    tree = [{
        "id": "asset-group-c_E000", "target": "group",
        "box": {"x": 50, "y": 50, "w": 320, "h": 320}, "meta": {"role": "card"},
        "children": [{
            "id": "c_E014", "target": "group",
            "box": {"x": 24, "y": 24, "w": 252, "h": 252},
            "shape_kind": "ellipse", "fill": {"kind": "flat", "color": "#2eb57e"},
            "src": cutout,
            "meta": {
                "role": "badge", "shell_raster_chip": True, "baked_badge_text": True,
            },
            "children": [],
        }],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path, (400, 400))))
    group = next(l for l in doc.layers if l.id == "asset-group-c_E000")
    badge = next(c for c in group.children if c.id == "c_E014")
    assert not any("shell" in str(c.id) for c in badge.children)
    assert any(c.meta.get("preserved_host_raster") for c in badge.children)
    assert doc.meta["asset_materialization"].get("starburst_seals", []) == []


def test_button_shell_still_emits_native_shape_without_chrome_raster_flags():
    """Buttons stay editable native primitives (not chrome-as-raster)."""
    shell = build_design_json._native_shell_shape(
        {"id": "c_cta", "shape_kind": "rect",
         "fill": {"kind": "flat", "color": "#111111"}, "radius": 12,
         "meta": {"role": "button", "plate_shell": True, "text_bearing_shell": True,
                  "button_shell": True}},
        {"x": 0, "y": 0, "w": 160, "h": 48}, 1.0,
    )
    assert shell is not None
    assert shell.type == "shape" and shell.shape_kind == "rect"
    assert shell.fill == {"kind": "flat", "color": "#111111"}


# ── Guards: photographic/gradient hosts keep their raster ─
def test_non_flat_and_non_shell_hosts_keep_their_raster(tmp_path):
    _plate(tmp_path)
    _source(tmp_path)
    # A gradient badge is not flat chrome → no native shell.
    assert build_design_json._native_shell_shape(
        {"id": "g", "shape_kind": "ellipse",
         "fill": {"kind": "linear", "stops": [{"color": "#fff"}, {"color": "#000"}]},
         "meta": {"role": "badge", "plate_shell": True}},
        {"x": 0, "y": 0, "w": 80, "h": 80}, 1.0) is None
    # A product photo host is not a shell → no native shell (F1 raster must survive).
    assert build_design_json._native_shell_shape(
        {"id": "p", "shape_kind": "rect", "fill": {"kind": "flat", "color": "#123456"},
         "meta": {"role": "product"}},
        {"x": 0, "y": 0, "w": 80, "h": 80}, 1.0) is None
    # An irregular (path) shell keeps the raster path too (088 c_E011 class).
    assert build_design_json._native_shell_shape(
        {"id": "i", "shape_kind": "path", "fill": {"kind": "flat", "color": "#303030"},
         "meta": {"role": "badge", "plate_shell": True}},
        {"x": 0, "y": 0, "w": 80, "h": 80}, 1.0) is None


# ── 088: a nested plate-passthrough drop must not re-emit as a blank image leaf ─────
def test_nested_drop_child_is_not_re_emitted_as_unexplained_raster(tmp_path):
    _plate(tmp_path)
    _source(tmp_path)
    tree = [{
        "id": "asset-group-c_E010", "target": "group",
        "box": {"x": 0, "y": 0, "w": 400, "h": 400}, "meta": {"role": "card"},
        "children": [
            # The confidence fallback retired this badge to the plate (it already holds
            # the source pixels). It must be DROPPED, not compiled into an image leaf.
            {"id": "c_E011", "target": "drop", "box": {"x": 10, "y": 10, "w": 60, "h": 30},
             "meta": {"role": "badge", "fallback": "plate-passthrough",
                      "fallback_reasons": ["region_ssim 0.200 < 0.58"],
                      "fallback_scores": {"region_ssim": 0.1999},
                      "keep_in_background": True}},
            {"id": "keep", "target": "text", "z": 40, "text": "hi",
             "box": {"x": 5, "y": 60, "w": 40, "h": 16}, "style": {"fontSize": 12}},
        ],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))

    def ids(layers):
        out = []
        for layer in layers:
            out.append(layer.id)
            out.extend(ids(layer.children or []))
        return out

    assert "c_E011" not in ids(doc.layers), "a nested target=drop must not be re-emitted"
    accounting = doc.meta["leaf_accounting"]
    assert accounting["unexplained_raster_count"] == 0
    assert accounting["unexplained_raster_ids"] == []


# ── 104 / 107 / 021: ban empty asset groups + blank ghost rasters ───────────────────
def test_empty_asset_group_materializes_real_source_pixels(tmp_path):
    # The subject must live in the PLATE: materialization crops background_clean, not
    # the original (original ink is owned by emitted layers — 013's materialized band
    # re-painted the headline under the native text). 104's burned-in phones are in
    # the plate precisely because they were never removed.
    plate = tmp_path / "background_clean.png"
    arr = np.full((400, 400, 3), 240, dtype=np.uint8)
    arr[100:300, 50:250] = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(plate)
    tree = [{
        "id": "asset-group-c_E002", "target": "group",
        "box": {"x": 50, "y": 100, "w": 200, "h": 200},
        "meta": {"role": "product"}, "children": [],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate))
    group = next(layer for layer in doc.layers if layer.id == "asset-group-c_E002")
    assert group.children, "an empty product asset-group must never ship empty"
    # ONE owner per band: either the per-group plate slice claims the group (making it
    # non-empty so materialization skips) or the materialized crop does — never both
    # (013 shipped two stacked bands, the second re-painting original headline ink).
    images = [c for c in group.children if c.type == "image" and c.src]
    assert len(images) == 1, f"exactly one pixel child expected, got {[c.id for c in group.children]}"
    child = images[0]
    staged = tmp_path / child.src
    assert staged.exists()
    with Image.open(staged) as im:
        assert im.size == (200, 200)          # pixel-exact, cropped from its own box
        assert np.asarray(im.convert("RGBA"))[..., 3].min() == 255   # opaque, no ghost


def test_blank_ghost_raster_is_materialized_from_source(tmp_path):
    # Subject lives in the PLATE (see test above): a ghost raster's real pixels are
    # recovered from background_clean, never from the original.
    plate = tmp_path / "background_clean.png"
    arr = np.full((400, 400, 3), 240, dtype=np.uint8)
    arr[100:300, 50:250] = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(plate)
    # The 8KB blank PNG class: correct size, alpha ~= 0 → renders nothing.
    ghost = tmp_path / "pack.png"
    Image.new("RGBA", (200, 200), (0, 0, 0, 0)).save(ghost)
    tree = [{
        "id": "c_E004", "target": "image", "z": 20,
        "box": {"x": 50, "y": 100, "w": 200, "h": 200},
        "src": str(ghost), "meta": {"role": "product"},
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate))
    layer = next(l for l in doc.layers if l.id == "c_E004")
    assert layer.meta["materialized_reason"] == "blank-ghost-raster"
    with Image.open(tmp_path / layer.src) as im:
        assert np.asarray(im.convert("RGBA"))[..., 3].min() == 255
    assert doc.meta["asset_materialization"]["materialized"][0]["id"] == "c_E004"


def test_blank_group_over_featureless_source_is_dropped_with_a_reason(tmp_path):
    """No subject to materialize → drop with a RECORDED reason, never ship silently."""
    _plate(tmp_path)
    _source(tmp_path)  # perfectly flat source: nothing to materialize
    tree = [{
        "id": "asset-group-c_E000", "target": "group",
        "box": {"x": 10, "y": 10, "w": 120, "h": 120},
        "meta": {"role": "photo"}, "children": [],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    assert not [l for l in doc.layers if l.id == "asset-group-c_E000"]
    dropped = doc.meta["asset_materialization"]["dropped"]
    assert dropped and dropped[0]["id"] == "asset-group-c_E000"
    assert dropped[0]["reason"] == "empty-asset-group-no-subject"


def test_unverifiable_blank_layer_is_kept_not_erased(tmp_path):
    """A blank IMAGE whose pixels are NOT in the plate (uniform crop) is KEPT + warned:
    erasing what we merely failed to rebuild is content erasure (F1). An EMPTY GROUP
    over the same featureless plate has nothing anywhere and is dropped with a
    recorded reason (that is the empty-asset-group ban working)."""
    plate = _plate(tmp_path)  # uniform white plate: no subject anywhere
    tree = [
        {"id": "avatar0", "target": "image", "z": 20,
         "box": {"x": 10, "y": 10, "w": 60, "h": 60},
         "mask": {"kind": "ellipse"}, "meta": {"role": "avatar"}},
        {"id": "asset-group-x", "target": "group", "box": {"x": 100, "y": 10, "w": 80, "h": 80},
         "meta": {"role": "photo"}, "children": []},
    ]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(plate))
    ids = {layer.id for layer in doc.layers}
    assert "avatar0" in ids, "an unverifiable layer must never be silently erased"
    assert "asset-group-x" not in ids
    dropped = doc.meta["asset_materialization"]["dropped"]
    assert [d["id"] for d in dropped] == ["asset-group-x"]
    codes = {w.get("code") for w in doc.meta["warnings"]}
    assert "blank-raster-unverified" in codes


def test_materialization_never_regates_a_confidence_slice(tmp_path):
    """A raster-slice IS pixel-exact source already — the gate must leave it alone."""
    _plate(tmp_path)
    _source(tmp_path)
    # A legitimately sparse slice (thin arrow) with low alpha coverage.
    slice_png = tmp_path / "slice.png"
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[48:52, :, :] = 255
    Image.fromarray(arr, "RGBA").save(slice_png)
    tree = [{
        "id": "c_E007", "target": "image", "z": 20,
        "box": {"x": 10, "y": 10, "w": 100, "h": 100}, "src": str(slice_png),
        "meta": {"role": "photo", "fallback": "raster-slice",
                 "fallback_scores": {"region_ssim": 0.2}},
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    layer = next(l for l in doc.layers if l.id == "c_E007")
    assert "materialized_reason" not in layer.meta
    assert doc.meta["asset_materialization"]["materialized"] == []
