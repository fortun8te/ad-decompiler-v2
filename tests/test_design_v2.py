import json

import pytest
from PIL import Image

from src import build_design_json, figma_import


def test_untouched_source_cannot_be_used_as_rebuilt_background(tmp_path):
    source = tmp_path / "normalized.png"
    Image.new("RGB", (10, 10)).save(source)
    with pytest.raises(ValueError, match="untouched source"):
        build_design_json.build(
            [{"id": "t", "target": "text", "text": "x", "box": {"x": 0, "y": 0, "w": 5, "h": 5}}],
            {"w": 10, "h": 10}, str(tmp_path), base_src=str(source),
        )


def test_svg_z_and_nested_frame_survive_design_compile(tmp_path):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (100, 100), "white").save(background)
    tree = [{
        "id": "frame", "target": "group", "z": 2, "box": {"x": 5, "y": 5, "w": 80, "h": 60},
        "fill": {"kind": "flat", "color": "#eeeeee"}, "children": [{
            "id": "icon", "target": "icon", "z": 7, "box": {"x": 10, "y": 10, "w": 20, "h": 20},
            "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L10 0L0 10Z" fill="#f00"/></svg>',
            "paths": [{"d": "M0 0L10 0L0 10Z", "fill": "#ff0000"}],
        }],
    }]
    doc = build_design_json.build(tree, {"w": 100, "h": 100}, str(tmp_path),
                                   base_src=str(background))
    assert doc.schema_version == 2
    assert doc.layers[1].type == "group"
    assert doc.layers[1].children[0].svg.startswith("<svg")
    assert doc.layers[1].children[0].z_index == 7


def test_text_with_fusion_z_one_paints_above_its_native_button_shell(tmp_path):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (100, 100), "white").save(background)
    tree = [{
        "id": "button", "target": "group", "box": {"x": 0, "y": 0, "w": 90, "h": 40},
        "children": [
            {"id": "label", "target": "text", "z": 1, "text": "SHOP NOW",
             "box": {"x": 10, "y": 10, "w": 60, "h": 15}, "style": {"fontSize": 12}},
            {"id": "shell", "target": "shape", "z": 0,
             "box": {"x": 0, "y": 0, "w": 90, "h": 40}},
        ],
    }]
    doc = build_design_json.build(tree, {"w": 100, "h": 100}, str(tmp_path),
                                   base_src=str(background))
    children = doc.layers[1].children
    assert children[-1].id == "label"
    assert children[-1].z_index > children[0].z_index


def test_unannotated_scene_uses_background_gradient_image_icon_text_stack(tmp_path):
    background = tmp_path / "background_clean.png"
    asset = tmp_path / "photo.png"
    Image.new("RGB", (100, 100), "white").save(background)
    Image.new("RGBA", (40, 40), "red").save(asset)
    doc = build_design_json.build([
        {"id": "gradient", "target": "shape", "box": {"x": 0, "y": 0, "w": 100, "h": 100},
         "fill": {"kind": "linear", "stops": [{"color": "#112233", "pos": 0},
                                                    {"color": "#445566", "pos": 1}]}},
        {"id": "photo", "target": "image", "box": {"x": 30, "y": 20, "w": 40, "h": 40},
         "src": str(asset), "meta": {"role": "photo"}},
        {"id": "icon", "target": "icon", "box": {"x": 70, "y": 10, "w": 12, "h": 12},
         "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h12v12z"/></svg>'},
        {"id": "copy", "target": "text", "text": "SALE",
         "box": {"x": 10, "y": 70, "w": 40, "h": 15}, "style": {"fontSize": 12}},
    ], {"w": 100, "h": 100}, str(tmp_path), base_src=str(background))

    assert [layer.id for layer in doc.layers] == ["background", "gradient", "photo", "icon", "copy"]
    assert [layer.z_index for layer in doc.layers] == [-1_000_000, 20.0, 30.0, 35.0, 40.0]


def test_shape_style_paints_are_preserved_for_native_figma_mapping(tmp_path):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (100, 100), "white").save(background)
    style = {
        "fills": [
            {"kind": "linear-gradient", "angle": 90, "stops": [
                {"color": "#ff2200", "offset": 0}, {"color": "#0044ff", "offset": 100},
            ]},
            {"kind": "flat", "color": "#ffffff", "opacity": 0.15},
        ],
        "strokes": [{"color": "#ffffff", "width": 2, "align": "inside", "dash": [4, 2]}],
        "effects": [{"type": "drop-shadow", "color": "#00000066", "x": 0, "y": 3, "blur": 8}],
    }
    doc = build_design_json.build([{
        "id": "card", "target": "shape", "box": {"x": 5, "y": 5, "w": 80, "h": 50}, "style": style,
    }], {"w": 100, "h": 100}, str(tmp_path), base_src=str(background))

    layer = doc.layers[1]
    saved = json.loads((tmp_path / "design.json").read_text(encoding="utf-8"))["layers"][1]
    assert layer.fill is None  # Let Figma consume all style fills rather than only the first.
    assert layer.style["fills"] == style["fills"]
    assert layer.style["strokes"] == style["strokes"]
    assert layer.effects == style["effects"]
    assert saved["style"] == style


def test_semantic_z_band_survives_design_compile_for_placeholder_z(tmp_path):
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (100, 100), "white").save(background)
    doc = build_design_json.build([
        {"id": "content", "target": "image", "box": {"x": 0, "y": 0, "w": 80, "h": 80},
         "meta": {"z_band": "content"}},
        {"id": "chrome", "target": "shape", "box": {"x": 20, "y": 20, "w": 20, "h": 20},
         "meta": {"z_band": "chrome"}},
    ], {"w": 100, "h": 100}, str(tmp_path), base_src=str(background))

    assert [layer.id for layer in doc.layers] == ["background", "content", "chrome"]
    assert [layer.z_index for layer in doc.layers] == [-1_000_000, 20.0, 50.0]


def test_vector_layer_keeps_raster_preview_fallback(tmp_path):
    from PIL import Image
    asset = tmp_path / "icon.png"
    Image.new("RGBA", (4, 4), (20, 30, 40, 255)).save(asset)
    doc = build_design_json.build([{
        "id": "icon", "target": "icon", "box": {"x": 0, "y": 0, "w": 4, "h": 4},
        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L4 0L0 4Z"/></svg>',
        "src": str(asset),
    }], {"w": 4, "h": 4}, str(tmp_path))
    assert doc.layers[0].src.replace("\\", "/") == "assets/icon_icon.png"


def test_semantic_asset_name_survives_design_compile(tmp_path):
    asset = tmp_path / "avatar.png"
    Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(asset)
    doc = build_design_json.build([{
        "id": "avatar", "target": "image", "box": {"x": 0, "y": 0, "w": 8, "h": 8},
        "src": str(asset), "meta": {"role": "avatar", "semantic_name": "Creator avatar"},
    }], {"w": 8, "h": 8}, str(tmp_path))
    assert doc.layers[0].name == "Creator avatar"


def test_corrupt_raster_is_rejected_before_design_compile(tmp_path):
    asset = tmp_path / "broken.png"
    asset.write_bytes(b"not a png")
    doc = build_design_json.build([{
        "id": "photo", "target": "image", "box": {"x": 0, "y": 0, "w": 4, "h": 4},
        "src": str(asset),
    }], {"w": 4, "h": 4}, str(tmp_path))
    assert doc.layers[0].src is None
    assert doc.layers[0].meta["compiler_error"] == "missing image asset"
    assert "corrupt-asset" in {warning["code"] for warning in doc.meta["warnings"]}


def test_atomic_figma_staging_contains_manifest_and_assets(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assets = run / "assets"
    assets.mkdir()
    Image.new("RGBA", (5, 5), "red").save(assets / "thing.png")
    design = {
        "schema_version": 2, "id": "demo", "name": "Demo", "canvas": {"w": 10, "h": 10},
        "layers": [{"id": "i", "type": "image", "src": "assets/thing.png"}],
        "meta": {"layer_count": 1, "editable_ratio": 0},
    }
    design_path = run / "design.json"
    design_path.write_text(encoding="utf-8", data=json.dumps(design))
    inbox = tmp_path / "inbox"
    result = figma_import.import_design(str(design_path), str(run),
                                        {"figma": {"mode": "plugin", "inbox": str(inbox)}})
    manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert manifest["schema_version"] == 2
    assert (inbox / manifest["staged_dir"] / "assets" / "thing.png").exists()
    assert any(item["path"] == "assets/thing.png" and item["sha256"] for item in manifest["files"])


def test_removed_clipboard_mode_fails_clearly_instead_of_advertising_missing_bridge(tmp_path):
    result = figma_import.import_design(
        str(tmp_path / "design.json"), str(tmp_path), {"figma": {"mode": "clipboard"}},
    )

    assert result["ok"] is False
    assert result["mode"] == "clipboard"
    assert result["error"] == "unsupported Figma mode: clipboard"


def test_leaf_accounting_does_not_call_an_image_only_wrapper_editable(tmp_path):
    background = tmp_path / "background_clean.png"
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (100, 100), "white").save(background)
    Image.new("RGBA", (80, 60), "blue").save(screenshot)
    doc = build_design_json.build([{
        "id": "screen-group", "target": "group", "box": {"x": 10, "y": 10, "w": 80, "h": 60},
        "children": [{
            "id": "screen", "target": "image", "src": str(screenshot),
            "box": {"x": 0, "y": 0, "w": 80, "h": 60},
            "meta": {"role": "screenshot", "intentional_raster_cluster": True},
        }],
    }], {"w": 100, "h": 100}, str(tmp_path), base_src=str(background))

    accounting = doc.meta["leaf_accounting"]
    assert doc.meta["editable_ratio"] > 0  # legacy wrapper-counting metric remains compatible
    assert accounting["foreground_leaf_count"] == 1
    assert accounting["native_leaf_count"] == 0
    assert accounting["raster_leaf_count"] == 1
    assert accounting["intentional_raster_cluster_count"] == 1
    assert doc.meta["native_leaf_ratio"] == 0.0


def test_leaf_accounting_flags_only_unexplained_generic_fallbacks(tmp_path):
    asset = tmp_path / "asset.png"
    Image.new("RGBA", (10, 10), "red").save(asset)
    doc = build_design_json.build([
        {"id": "unknown", "target": "image", "src": str(asset),
         "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "meta": {"role": "unknown", "fallback": True}},
        {"id": "photo", "target": "image", "src": str(asset),
         "box": {"x": 12, "y": 0, "w": 10, "h": 10},
         "meta": {"role": "photo", "fallback": True}},
    ], {"w": 30, "h": 10}, str(tmp_path))

    accounting = doc.meta["leaf_accounting"]
    assert accounting["fallback_raster_count"] == 2
    assert accounting["unexplained_raster_count"] == 1
    assert accounting["unexplained_raster_ids"] == ["unknown"]
