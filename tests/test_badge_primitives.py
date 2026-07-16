"""Badge/pill/seal native primitives + the empty-asset materialization ban.

Covers the audit findings on runs/postfix-benchmark-4:
  * 101 "BOGO badge -> square"      — a circular badge shipped as a square raster
  * 013 61%-OFF ellipse / "snacks" pill
  * 016 "45% Off" scalloped starburst seal shipped as a teal square
  * 104/107/021 empty asset groups + blank 8KB ghost PNGs
  * 088 c_E011 "unexplained-raster-fallback" on a nested plate-passthrough drop
"""
import math

import numpy as np
import pytest
from PIL import Image

from src import build_design_json


def _plate(tmp_path, size=(400, 400), color="white"):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", size, color).save(background)
    return background


def _source(tmp_path, size=(400, 400), color=(240, 240, 240)):
    """A normalized.png the materialization gate / starburst fitter can read."""
    src = tmp_path / "normalized.png"
    Image.new("RGB", size, color).save(src)
    return src


# ── 101 / 013: flat circular badge → native ellipse, never a raster hostbg ──────────
def test_circular_flat_badge_emits_native_ellipse_not_raster_square(tmp_path):
    _plate(tmp_path)
    _source(tmp_path)
    # The matte for a small saturated plate routinely comes back near-empty: this is the
    # exact asset that shipped 101's BOGO badge as an invisible ghost inside a teal SQUARE.
    ghost = tmp_path / "badge.png"
    Image.new("RGBA", (87, 87), (12, 162, 177, 0)).save(ghost)
    tree = [{
        "id": "c_E005", "target": "group", "box": {"x": 288, "y": 408, "w": 87, "h": 87},
        "shape_kind": "ellipse", "fill": {"kind": "flat", "color": "#0ca2b1"},
        "src": str(ghost),
        "meta": {"role": "badge", "plate_shell": True, "text_bearing_shell": True},
        "children": [
            {"id": "c_B4", "target": "text", "z": 40, "text": "BUY 3, GET 1",
             "box": {"x": 8, "y": 20, "w": 70, "h": 20}, "style": {"fontSize": 12}},
        ],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    group = next(layer for layer in doc.layers if layer.id == "c_E005")
    shells = [c for c in group.children if c.meta.get("rebuilt_from") == "flat-ellipse-shell"]
    assert len(shells) == 1, "a flat circular badge must be rebuilt as a native ellipse"
    shell = shells[0]
    assert shell.type == "shape" and shell.shape_kind == "ellipse"
    assert shell.fill == {"kind": "flat", "color": "#0ca2b1"}
    assert shell.box == {"x": 0.0, "y": 0.0, "w": 87.0, "h": 87.0}
    # The blank hostbg raster must NOT be emitted alongside it.
    assert not any(c.meta.get("preserved_host_raster") for c in group.children)
    # The FRAME must not paint the flat fill as a square behind the ellipse (101's bug).
    assert group.fill is None and group.radius is None
    # The badge's copy stays native TEXT.
    assert any(c.type == "text" for c in group.children)
    # ...and the ellipse sits behind the text.
    assert shell.z_index < min(c.z_index for c in group.children if c.id != shell.id)


# ── 013: a wide flat chip → native stadium/pill rrect, not a hard-cornered box ──────
def test_wide_flat_chip_emits_native_pill_rrect(tmp_path):
    _plate(tmp_path)
    _source(tmp_path)
    tree = [{
        "id": "c_E006", "target": "group", "box": {"x": 413, "y": 719, "w": 114, "h": 50},
        "shape_kind": "rect", "fill": {"kind": "flat", "color": "#1a1a1a"},
        "meta": {"role": "badge", "plate_shell": True, "text_bearing_shell": True},
        "children": [
            {"id": "c_B9", "target": "text", "z": 40, "text": "snacks",
             "box": {"x": 20, "y": 12, "w": 70, "h": 24}, "style": {"fontSize": 18}},
        ],
    }]
    doc = build_design_json.build(tree, {"w": 800, "h": 900}, str(tmp_path),
                                  base_src=str(_plate(tmp_path, (800, 900))))
    group = next(layer for layer in doc.layers if layer.id == "c_E006")
    shell = next(c for c in group.children if c.meta.get("rebuilt_from") == "flat-rect-shell")
    assert shell.type == "shape" and shell.shape_kind == "rect"
    # aspect 2.28 with no explicit radius reads as a stadium pill → radius == h/2.
    assert shell.radius == pytest.approx(25.0)
    assert shell.fill == {"kind": "flat", "color": "#1a1a1a"}


# ── 016: a scalloped seal → analytic star polygon (vectorize's verified fitter) ─────
def test_scalloped_seal_emits_native_starburst_path(tmp_path):
    _plate(tmp_path, (400, 400))
    # Paint a real 26-point scalloped seal into the source so the fitter has a silhouette.
    src = tmp_path / "normalized.png"
    img = Image.new("RGB", (400, 400), (240, 240, 240))
    arr = np.asarray(img).copy()
    cy, cx, points = 200.0, 200.0, 26
    yy, xx = np.mgrid[0:400, 0:400]
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(yy - cy, xx - cx)
    # r(theta) oscillates between 100 and 125 → a clearly scalloped (not round) edge.
    r_theta = 112.5 + 12.5 * np.cos(points * ang)
    arr[rad <= r_theta] = (46, 181, 126)  # #2eb57e
    Image.fromarray(arr).save(src)

    _plate(tmp_path, (400, 400))
    # NEST the badge inside a parent so its compiled box is PARENT-RELATIVE — the seal
    # sits at abs (74,74) but rel (24,24). The star fitter must sample absolute pixels;
    # sampling the relative box reads the wrong region and silently degrades to an ellipse
    # (the bug that shipped 016's seal as an ellipse on the first pass).
    tree = [{
        "id": "asset-group-c_E000", "target": "group",
        "box": {"x": 50, "y": 50, "w": 320, "h": 320}, "meta": {"role": "card"},
        "children": [{
            "id": "c_E014", "target": "group",
            "box": {"x": 24, "y": 24, "w": 252, "h": 252},
            "shape_kind": "ellipse", "fill": {"kind": "flat", "color": "#2eb57e"},
            "meta": {"role": "badge", "plate_shell": True, "text_bearing_shell": True},
            "children": [
                {"id": "c_B2", "target": "text", "z": 40, "text": "45% Off",
                 "box": {"x": 80, "y": 110, "w": 90, "h": 30}, "style": {"fontSize": 16}},
            ],
        }],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path, (400, 400))))
    group = next(l for l in doc.layers if l.id == "asset-group-c_E000")
    badge = next(c for c in group.children if c.id == "c_E014")
    shell = next(c for c in badge.children if "shell" in str(c.id))
    assert shell.meta.get("rebuilt_from") == "starburst-seal", (
        "a scalloped seal must fit an analytic star, not fall back to an ellipse")
    assert shell.shape_kind == "path" and shell.path and shell.path.startswith("M")
    prim = shell.meta["star_primitive"]
    assert prim["kind"] == "star" and prim["points"] == 26
    assert prim["iou"] >= 0.90
    assert shell.fill == {"kind": "flat", "color": "#2eb57e"}
    assert doc.meta["asset_materialization"]["starburst_seals"][0]["points"] == 26


def test_plain_disc_is_an_ellipse_not_a_starburst(tmp_path):
    """A solid circle must NOT be mistaken for a scalloped seal (no false stars)."""
    src = tmp_path / "normalized.png"
    img = Image.new("RGB", (400, 400), (240, 240, 240))
    arr = np.asarray(img).copy()
    yy, xx = np.mgrid[0:400, 0:400]
    arr[np.hypot(yy - 200, xx - 200) <= 110] = (12, 162, 177)
    Image.fromarray(arr).save(src)
    tree = [{
        "id": "c_E005", "target": "group", "box": {"x": 90, "y": 90, "w": 220, "h": 220},
        "shape_kind": "ellipse", "fill": {"kind": "flat", "color": "#0ca2b1"},
        "meta": {"role": "badge", "plate_shell": True},
        "children": [{"id": "t", "target": "text", "z": 40, "text": "hi",
                      "box": {"x": 90, "y": 100, "w": 40, "h": 16},
                      "style": {"fontSize": 12}}],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path, (400, 400))))
    badge = next(l for l in doc.layers if l.id == "c_E005")
    shell = next(c for c in badge.children if "shell" in str(c.id))
    assert shell.shape_kind == "ellipse"
    assert shell.meta.get("rebuilt_from") == "flat-ellipse-shell"
    assert doc.meta["asset_materialization"]["starburst_seals"] == []


# ── Guards: only flat chrome is rebuilt; photographic/gradient hosts keep their raster ─
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
    _plate(tmp_path)
    # A source whose box region carries a real subject (high variance), not flat bg.
    src = tmp_path / "normalized.png"
    arr = np.full((400, 400, 3), 240, dtype=np.uint8)
    arr[100:300, 50:250] = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src)
    tree = [{
        "id": "asset-group-c_E002", "target": "group",
        "box": {"x": 50, "y": 100, "w": 200, "h": 200},
        "meta": {"role": "product"}, "children": [],
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    group = next(layer for layer in doc.layers if layer.id == "asset-group-c_E002")
    assert group.children, "an empty product asset-group must never ship empty"
    child = group.children[0]
    assert child.type == "image" and child.src
    assert child.meta["materialized_reason"] == "empty-asset-group"
    staged = tmp_path / child.src
    assert staged.exists()
    with Image.open(staged) as im:
        assert im.size == (200, 200)          # pixel-exact, cropped from its own box
        assert np.asarray(im.convert("RGBA"))[..., 3].min() == 255   # opaque, no ghost
    report = doc.meta["asset_materialization"]
    assert [m["id"] for m in report["materialized"]] == ["asset-group-c_E002"]


def test_blank_ghost_raster_is_materialized_from_source(tmp_path):
    _plate(tmp_path)
    src = tmp_path / "normalized.png"
    arr = np.full((400, 400, 3), 240, dtype=np.uint8)
    arr[100:300, 50:250] = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src)
    # The 8KB blank PNG class: correct size, alpha ~= 0 → renders nothing.
    ghost = tmp_path / "pack.png"
    Image.new("RGBA", (200, 200), (0, 0, 0, 0)).save(ghost)
    tree = [{
        "id": "c_E004", "target": "image", "z": 20,
        "box": {"x": 50, "y": 100, "w": 200, "h": 200},
        "src": str(ghost), "meta": {"role": "product"},
    }]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
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
    """No readable source → KEEP + warn. Erasing what we merely failed to VERIFY is
    content erasure (F1), a worse bug than the ghost it would fix."""
    _plate(tmp_path)   # deliberately NO normalized.png / original.png in the run dir
    tree = [
        {"id": "avatar0", "target": "image", "z": 20,
         "box": {"x": 10, "y": 10, "w": 60, "h": 60},
         "mask": {"kind": "ellipse"}, "meta": {"role": "avatar"}},
        {"id": "asset-group-x", "target": "group", "box": {"x": 100, "y": 10, "w": 80, "h": 80},
         "meta": {"role": "photo"}, "children": []},
    ]
    doc = build_design_json.build(tree, {"w": 400, "h": 400}, str(tmp_path),
                                  base_src=str(_plate(tmp_path)))
    ids = {layer.id for layer in doc.layers}
    assert "avatar0" in ids, "an unverifiable layer must never be silently erased"
    assert "asset-group-x" in ids
    assert doc.meta["asset_materialization"]["dropped"] == []
    codes = {w.get("code") for w in doc.meta["warnings"]}
    assert "blank-raster-unverified" in codes
    assert "empty-asset-group-unverified" in codes


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
