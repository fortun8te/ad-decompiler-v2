"""Tests for the figma_import compiler preflight (mirror of figma-plugin/code.js).

Each new preflight rule gets a synthetic design.json fixture. The preflight is the only
automated guard before a human clicks Import in Figma desktop, so every case where the
plugin would reject or silently degrade a layer must surface as a named finding here.
"""
import json

import pytest
from PIL import Image

from src import figma_import


def _codes(preflight):
    return {w["code"] for w in preflight["warnings"]}


def _by_code(preflight, code):
    return [w for w in preflight["warnings"] if w["code"] == code]


def _design(layers, **extra):
    doc = {"id": "doc", "name": "Doc", "canvas": {"w": 100, "h": 100},
           "schema_version": 2, "layers": layers}
    doc.update(extra)
    return doc


def _run_dir_with_asset(tmp_path, name, image):
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    image.save(assets / name)
    return tmp_path


# ── geometry ─────────────────────────────────────────────────────────────────────────
def test_nan_geometry_is_an_error_class_finding(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "n", "type": "shape", "fill": "#fff",
         "box": {"x": 0, "y": float("nan"), "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "invalid-geometry")[0]
    assert finding["severity"] == "error"
    assert "y" in finding["fields"]


def test_negative_dimension_is_invalid_geometry(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "n", "type": "shape", "fill": "#fff",
         "box": {"x": 0, "y": 0, "w": -5, "h": 10}},
    ]), str(tmp_path), {})
    assert "invalid-geometry" in _codes(pf)
    assert pf["error_count"] >= 1


def test_zero_size_layer_is_a_warning_not_error(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "z", "type": "shape", "fill": "#fff",
         "box": {"x": 0, "y": 0, "w": 0, "h": 10}},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "zero-size-layer")[0]
    assert finding["severity"] == "warn"
    assert "invalid-geometry" not in _codes(pf)


# ── fonts ──────────────────────────────────────────────────────────────────────────
def test_text_without_font_family_flags_empty_font_candidates(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "t", "type": "text", "text": "HELLO",
         "box": {"x": 0, "y": 0, "w": 40, "h": 12}, "style": {"fontSize": 12}},
    ]), str(tmp_path), {})
    assert "empty-font-candidates" in _codes(pf)


def test_text_with_font_family_or_candidates_is_not_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "a", "type": "text", "text": "HELLO", "box": {"x": 0, "y": 0, "w": 40, "h": 12},
         "style": {"fontFamily": "Inter", "fontSize": 12}},
        {"id": "b", "type": "text", "text": "WORLD", "box": {"x": 0, "y": 20, "w": 40, "h": 12},
         "style": {"font_candidates": [{"family": "Roboto"}], "fontSize": 12}},
    ]), str(tmp_path), {})
    assert "empty-font-candidates" not in _codes(pf)


def test_empty_text_layer_does_not_trip_font_rule(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "t", "type": "text", "text": "", "box": {"x": 0, "y": 0, "w": 40, "h": 12}},
    ]), str(tmp_path), {})
    assert "empty-font-candidates" not in _codes(pf)


# ── vector / svg ─────────────────────────────────────────────────────────────────────
def test_path_shape_without_geometry_is_empty_vector_error(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "v", "type": "shape", "shape_kind": "path", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "empty-vector")[0]
    assert finding["severity"] == "error"


def test_svg_with_empty_path_d_is_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "v", "type": "shape", "shape_kind": "path", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d=""/></svg>'},
    ]), str(tmp_path), {})
    assert "svg-empty-path" in _codes(pf)


def test_svg_filter_feature_is_flagged_as_unsupported(tmp_path):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><filter id="f">'
           '<feGaussianBlur stdDeviation="2"/></filter><path d="M0 0L4 0L0 4Z"/></svg>')
    pf = figma_import.compiler_preflight(_design([
        {"id": "v", "type": "shape", "shape_kind": "path", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "svg": svg},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "svg-unsupported-feature")[0]
    assert "filter" in finding["features"]
    assert "filter-primitive" in finding["features"]


def test_svg_too_many_paths_respects_configured_limit(tmp_path):
    paths = [{"d": "M0 0L1 0L0 1Z"}, {"d": "M0 0L2 0L0 2Z"}]
    pf = figma_import.compiler_preflight(
        _design([{"id": "v", "type": "shape", "shape_kind": "path",
                  "box": {"x": 0, "y": 0, "w": 10, "h": 10}, "paths": paths}]),
        str(tmp_path), {"figma": {"preflight": {"svg_max_paths": 1}}})
    finding = _by_code(pf, "svg-too-many-paths")[0]
    assert finding["path_count"] == 2 and finding["limit"] == 1


def test_valid_single_path_vector_is_clean(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "v", "type": "shape", "shape_kind": "path", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L4 0L0 4Z"/></svg>'},
    ]), str(tmp_path), {})
    assert not (_codes(pf) & {"empty-vector", "svg-empty-path", "svg-too-many-paths",
                              "svg-unsupported-feature"})


# ── gradients ────────────────────────────────────────────────────────────────────────
def test_single_stop_gradient_is_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "g", "type": "shape", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "fill": {"kind": "linear", "stops": [{"color": "#000", "offset": 0}]}},
    ]), str(tmp_path), {})
    assert _by_code(pf, "gradient-stops")[0]["stops"] == 1


def test_two_stop_gradient_is_clean(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "g", "type": "shape", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "fill": {"kind": "linear", "stops": [{"color": "#000", "offset": 0},
                                              {"color": "#fff", "offset": 1}]}},
    ]), str(tmp_path), {})
    assert "gradient-stops" not in _codes(pf)


def test_gradient_in_style_fills_is_inspected(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "g", "type": "shape", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "style": {"fills": [{"kind": "radial-gradient", "stops": []}]}},
    ]), str(tmp_path), {})
    assert "gradient-stops" in _codes(pf)


# ── effects ──────────────────────────────────────────────────────────────────────────
def test_negative_effect_radius_flags_param_range(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "e", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "effects": [{"type": "drop-shadow", "radius": -4}]},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "effect-param-range")[0]
    assert finding["param"] == "radius" and finding["severity"] == "warn"


def test_unknown_effect_type_is_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "e", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "effects": [{"type": "glow", "radius": 4}]},
    ]), str(tmp_path), {})
    assert "unsupported-effect" in _codes(pf)


def test_reasonable_shadow_is_clean(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "e", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "effects": [{"type": "drop-shadow", "radius": 8, "spread": 0,
                      "offset": {"x": 0, "y": 3}}]},
    ]), str(tmp_path), {})
    assert not (_codes(pf) & {"effect-param-range", "unsupported-effect"})


# ── blend / shape paint ──────────────────────────────────────────────────────────────
def test_unsupported_blend_mode_is_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "s", "type": "shape", "fill": "#fff", "blend_mode": "glitter",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert "unsupported-blend-mode" in _codes(pf)


def test_known_blend_mode_is_clean(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "s", "type": "shape", "fill": "#fff", "blend_mode": "multiply",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert "unsupported-blend-mode" not in _codes(pf)


def test_shape_without_paint_is_flagged(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "s", "type": "shape", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert "shape-no-paint" in _codes(pf)


# ── unknown layer type ───────────────────────────────────────────────────────────────
def test_unknown_layer_type_is_error(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "u", "type": "hologram", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "unknown-layer-type")[0]
    assert finding["severity"] == "error"


# ── image asset integrity ────────────────────────────────────────────────────────────
def test_missing_image_asset_is_error(tmp_path):
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/nope.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    finding = _by_code(pf, "missing-asset")[0]
    assert finding["severity"] == "error" and finding["path"] == "assets/nope.png"


def test_corrupt_image_asset_is_error(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "broken.png").write_bytes(b"not really a png")
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/broken.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert _by_code(pf, "corrupt-asset")[0]["severity"] == "error"


def test_image_over_4096px_is_too_large_error(tmp_path):
    run = _run_dir_with_asset(tmp_path, "big.png", Image.new("RGBA", (4097, 8), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/big.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(run), {})
    finding = _by_code(pf, "image-too-large")[0]
    assert finding["severity"] == "error" and finding["width"] == 4097
    assert finding["limit"] == figma_import.FIGMA_MAX_IMAGE_DIM


def test_image_exactly_4096px_is_allowed(tmp_path):
    run = _run_dir_with_asset(tmp_path, "ok.png", Image.new("RGBA", (4096, 4096), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/ok.png", "meta": {"role": "photo"},
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(run), {})
    assert "image-too-large" not in _codes(pf)


def test_cutout_without_alpha_flags_alpha_channel_loss(tmp_path):
    run = _run_dir_with_asset(tmp_path, "flat.png", Image.new("RGB", (10, 10), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/flat.png", "meta": {"role": "product"},
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(run), {})
    assert "alpha-channel-loss" in _codes(pf)


def test_background_without_alpha_is_not_flagged(tmp_path):
    run = _run_dir_with_asset(tmp_path, "bg.png", Image.new("RGB", (100, 100), "white"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "background", "type": "image", "src": "assets/bg.png", "meta": {"role": "background"},
         "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ]), str(run), {})
    assert "alpha-channel-loss" not in _codes(pf)
    assert pf["assets"] and pf["assets"][0]["sha256"]  # checksum recorded at staging


def test_cmyk_image_flags_color_profile(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("CMYK", (10, 10)).save(assets / "cmyk.jpg", "JPEG")
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/cmyk.jpg", "meta": {"role": "photo"},
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert "color-profile" in _codes(pf)


def test_non_srgb_icc_profile_flags_color_profile(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGBA", (10, 10), "red").save(
        assets / "icc.png", icc_profile=b"MADEUP-non-standard-profile-bytes")
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/icc.png", "meta": {"role": "photo"},
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    assert "color-profile" in _codes(pf)


# ── masks ────────────────────────────────────────────────────────────────────────────
def test_alpha_mask_missing_source_is_warned(tmp_path):
    run = _run_dir_with_asset(tmp_path, "photo.png", Image.new("RGBA", (10, 10), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/photo.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "mask": {"kind": "alpha", "src": "assets/missing_mask.png"}},
    ]), str(run), {})
    finding = _by_code(pf, "alpha-mask-missing")[0]
    assert finding["severity"] == "warn"  # plugin degrades, does not reject


def test_path_mask_without_geometry_is_warned(tmp_path):
    run = _run_dir_with_asset(tmp_path, "photo.png", Image.new("RGBA", (10, 10), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": "assets/photo.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}, "mask": {"kind": "path"}},
    ]), str(run), {})
    assert "mask-geometry-empty" in _codes(pf)


# ── tolerance of unknown optional layer fields (raster-slice fallback, etc.) ──────────
def test_unknown_optional_layer_fields_are_never_rejected(tmp_path):
    run = _run_dir_with_asset(tmp_path, "photo.png", Image.new("RGBA", (10, 10), "red"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "future", "type": "image", "src": "assets/photo.png", "meta": {"role": "photo"},
         "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         # fields another agent may add to the schema concurrently:
         "raster_slice": {"enabled": True, "alpha": "assets/photo.png"},
         "fallback_editable": {"type": "text"}, "brand_new_flag": 123},
    ]), str(run), {})
    assert not any(w.get("layer_id") == "future" for w in pf["warnings"])
    assert pf["ok"] is True


def test_clean_design_has_no_findings(tmp_path):
    run = _run_dir_with_asset(tmp_path, "bg.png", Image.new("RGB", (100, 100), "white"))
    pf = figma_import.compiler_preflight(_design([
        {"id": "background", "type": "image", "src": "assets/bg.png",
         "meta": {"role": "background"}, "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
        {"id": "t", "type": "text", "text": "SALE", "box": {"x": 5, "y": 5, "w": 40, "h": 12},
         "style": {"fontFamily": "Inter", "fontSize": 12}},
    ]), str(run), {})
    assert pf["ok"] is True
    assert pf["warnings"] == []


# ── merge with build_design_json's structural preflight ──────────────────────────────
def test_build_preflight_warnings_are_merged_and_not_restated(tmp_path):
    # build_design_json already flagged this layer's asset as missing (and nulled src).
    (tmp_path / "design_preflight.json").write_text(json.dumps({
        "ok": False, "layer_count": 1,
        "warnings": [{"code": "missing-asset", "layer_id": "img", "path": "assets/gone.png"}],
    }), encoding="utf-8")
    pf = figma_import.compiler_preflight(_design([
        {"id": "img", "type": "image", "src": None, "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
        {"id": "s", "type": "shape", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]), str(tmp_path), {})
    missing = _by_code(pf, "missing-asset")
    assert len(missing) == 1  # build's finding kept, compiler mirror does not restate it
    assert missing[0]["path"] == "assets/gone.png"
    assert "shape-no-paint" in _codes(pf)  # compiler-only finding still added
    assert pf["layer_count"] == 1


# ── end-to-end staging with strict gating ────────────────────────────────────────────
def _write_design(run_dir, design):
    path = run_dir / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return path


def test_strict_mode_blocks_staging_on_error_class_finding(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    design_path = _write_design(run, _design([
        {"id": "img", "type": "image", "src": "assets/absent.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ], id="strict-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run),
        {"figma": {"mode": "plugin", "inbox": str(inbox), "strict": True}})
    assert result["ok"] is False
    assert result["blocked"] is True
    assert any(f["code"] == "missing-asset" for f in result["errors"])
    assert not (inbox / "inbox.json").exists()  # nothing published for the human to import


def test_strict_kwarg_overrides_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    design_path = _write_design(run, _design([
        {"id": "img", "type": "image", "src": "assets/absent.png",
         "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ], id="kwarg-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run),
        {"figma": {"mode": "plugin", "inbox": str(inbox)}}, strict=True)
    assert result["ok"] is False and result["blocked"] is True


def test_warn_class_findings_do_not_block_default_staging(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    design_path = _write_design(run, _design([
        {"id": "s", "type": "shape", "box": {"x": 0, "y": 0, "w": 0, "h": 10}, "fill": "#fff"},
    ], id="warn-demo"))
    inbox = tmp_path / "inbox"
    # strict True: zero-size is warn-class only, so staging still succeeds.
    result = figma_import.import_design(
        str(design_path), str(run),
        {"figma": {"mode": "plugin", "inbox": str(inbox), "strict": True}})
    assert result["ok"] is True
    assert result["preflight"]["errors"] == 0
    assert result["preflight"]["warnings"] >= 1


def test_staged_design_preflight_contains_merged_findings(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "design_preflight.json").write_text(json.dumps({
        "ok": False, "layer_count": 1,
        "warnings": [{"code": "text-fidelity-fallback", "layer_id": "t"}],
    }), encoding="utf-8")
    design_path = _write_design(run, _design([
        {"id": "t", "type": "text", "text": "HI", "box": {"x": 0, "y": 0, "w": 20, "h": 12}},
    ], id="merge-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run), {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    assert result["ok"] is True
    staged = json.loads(
        (inbox / "runs" / "merge-demo" / "design_preflight.json").read_text(encoding="utf-8"))
    codes = {w["code"] for w in staged["warnings"]}
    assert "text-fidelity-fallback" in codes   # from build_design_json
    assert "empty-font-candidates" in codes     # from the compiler mirror
    manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
    assert manifest["preflight"]["strict"] is False
    assert manifest["preflight"]["warnings"] == staged["warn_count"]


def test_manifest_summary_surfaces_preflight_warnings(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    design_path = _write_design(run, _design([
        {"id": "s", "type": "shape", "shape_kind": "path", "box": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ], id="summary-demo"))
    inbox = tmp_path / "inbox"
    figma_import.import_design(str(design_path), str(run),
                              {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
    assert any(w["code"] == "empty-vector" for w in manifest["summary"]["warnings"])


# ── "ship proof" screenshot sibling ───────────────────────────────────────────────────
def _run_with_screenshot(tmp_path, name="original.png", size=(100, 100)):
    run = tmp_path / "run"
    run.mkdir()
    Image.new("RGB", size, "blue").save(run / name)
    return run


def test_screenshot_sibling_present_by_default(tmp_path):
    run = _run_with_screenshot(tmp_path)
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="ship-proof-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run), {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    assert result["ok"] is True
    assert result["screenshot_sibling"]["ok"] is True

    staged = json.loads(
        (inbox / "runs" / "ship-proof-demo" / "design.json").read_text(encoding="utf-8"))
    layers = staged["layers"]
    shot = next(l for l in layers if l["name"] == "Screenshot - original.png")
    assert shot["type"] == "frame"
    assert shot["box"] == {"x": 0, "y": 0, "w": 100, "h": 100}
    assert shot["meta"]["role"] == "qa-ignore"
    image_child = shot["children"][0]
    assert image_child["type"] == "image"
    assert image_child["src"] == "assets/_screenshot_proof.png"

    # Canvas widened to fit both regions side by side; original layer shifted right by
    # exactly (canvas.w + gap) so nothing is clipped by the plugin's single clipsContent
    # root frame.
    assert staged["canvas"] == {"w": 250, "h": 100}  # 2*100 + 50 gap
    rebuilt = next(l for l in layers if l["id"] == "bg")
    assert rebuilt["box"]["x"] == 150  # 100 (canvas w) + 50 (gap)

    # Asset actually staged (not just referenced) and registered in preflight + manifest.
    asset_path = inbox / "runs" / "ship-proof-demo" / "assets" / "_screenshot_proof.png"
    assert asset_path.is_file()
    preflight = json.loads(
        (inbox / "runs" / "ship-proof-demo" / "design_preflight.json").read_text(encoding="utf-8"))
    assert not any(w["code"] == "missing-asset" for w in preflight["warnings"])
    assert any(a["layer_id"] == "screenshot-proof-image" for a in preflight["assets"])
    manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
    assert manifest["screenshot_sibling"]["ok"] is True
    assert any(f["path"] == "assets/_screenshot_proof.png" for f in manifest["files"])


def test_screenshot_sibling_absent_when_config_off(tmp_path):
    run = _run_with_screenshot(tmp_path)
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="off-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run),
        {"figma": {"mode": "plugin", "inbox": str(inbox), "stage_screenshot_sibling": False}})
    assert result["ok"] is True
    assert result["screenshot_sibling"] == {"ok": False, "reason": "disabled"}
    staged = json.loads(
        (inbox / "runs" / "off-demo" / "design.json").read_text(encoding="utf-8"))
    assert staged["canvas"] == {"w": 100, "h": 100}
    assert [l["id"] for l in staged["layers"]] == ["bg"]
    assert staged["layers"][0]["box"]["x"] == 0
    assert not (inbox / "runs" / "off-demo" / "assets" / "_screenshot_proof.png").exists()


def test_screenshot_sibling_skips_gracefully_without_source(tmp_path):
    run = tmp_path / "run"
    run.mkdir()  # no original.png / normalized.png present
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="no-source-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run), {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    assert result["ok"] is True
    assert result["screenshot_sibling"] == {"ok": False, "reason": "no-screenshot-source"}
    staged = json.loads(
        (inbox / "runs" / "no-source-demo" / "design.json").read_text(encoding="utf-8"))
    assert staged["canvas"] == {"w": 100, "h": 100}
    assert len(staged["layers"]) == 1


def test_screenshot_sibling_falls_back_to_normalized_png(tmp_path):
    run = _run_with_screenshot(tmp_path, name="normalized.png")
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="normalized-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run), {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    assert result["screenshot_sibling"]["ok"] is True
    staged = json.loads(
        (inbox / "runs" / "normalized-demo" / "design.json").read_text(encoding="utf-8"))
    assert any(l["name"] == "Screenshot - normalized.png" for l in staged["layers"])


def test_screenshot_sibling_never_mutates_the_run_design_json(tmp_path):
    """Proves the sibling is QA-invisible: preview.render/pixel_diff.compare always read
    run_dir/design.json *before* staging runs (run_pipeline.py calls render_preview.render
    on it, then figma_import.import_design later) — so as long as staging never writes
    back to that file, the sibling structurally cannot affect pixel_diff scoring."""
    run = _run_with_screenshot(tmp_path)
    design = _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="untouched-demo")
    design_path = _write_design(run, design)
    original_bytes = design_path.read_bytes()
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run), {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    assert result["screenshot_sibling"]["ok"] is True
    assert design_path.read_bytes() == original_bytes
    on_disk = json.loads(original_bytes)
    assert on_disk["canvas"] == {"w": 100, "h": 100}
    assert len(on_disk["layers"]) == 1


def test_screenshot_sibling_custom_gap_and_source_override(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    custom = tmp_path / "custom_shot.png"
    Image.new("RGB", (100, 100), "green").save(custom)
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="custom-demo"))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(
        str(design_path), str(run),
        {"figma": {"mode": "plugin", "inbox": str(inbox),
                   "screenshot_source": str(custom), "screenshot_gap": 10}})
    assert result["screenshot_sibling"] == {
        "ok": True, "source": str(custom), "asset": "assets/_screenshot_proof.png",
        "layer_id": "screenshot-proof", "gap": 10.0, "shift": 110.0,
    }
    staged = json.loads(
        (inbox / "runs" / "custom-demo" / "design.json").read_text(encoding="utf-8"))
    assert staged["canvas"] == {"w": 210, "h": 100}
    rebuilt = next(l for l in staged["layers"] if l["id"] == "bg")
    assert rebuilt["box"]["x"] == 110


def test_screenshot_sibling_idempotent_asset_copy_on_restage(tmp_path):
    run = _run_with_screenshot(tmp_path)
    design_path = _write_design(run, _design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="restage-demo"))
    inbox = tmp_path / "inbox"
    cfg = {"figma": {"mode": "plugin", "inbox": str(inbox)}}
    first = figma_import.import_design(str(design_path), str(run), cfg)
    second = figma_import.import_design(str(design_path), str(run), cfg)
    assert first["ok"] is True and second["ok"] is True
    # Restaging must not have re-shifted the already-widened original design.json on disk,
    # and must not fail even though the asset copy already exists from the first run.
    assert design_path.read_bytes() == json.dumps(_design([
        {"id": "bg", "type": "shape", "fill": "#fff", "box": {"x": 0, "y": 0, "w": 100, "h": 100}},
    ], id="restage-demo")).encode("utf-8")
