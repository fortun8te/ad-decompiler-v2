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


def test_f1_promoted_image_host_preserves_its_raster_as_background_child(tmp_path):
    # A pixel-carrying candidate (the 002 product panel) that reconstruction promoted to
    # a container GROUP must not lose its raster: the compiler re-emits it as a background
    # image child behind the group's other children, so the products survive.
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (200, 300), "white").save(background)
    panel = tmp_path / "panel.png"
    Image.new("RGBA", (150, 250), (10, 20, 30, 255)).save(panel)
    product = tmp_path / "product.png"
    Image.new("RGBA", (60, 80), (200, 50, 50, 255)).save(product)
    tree = [{
        "id": "c_E003", "target": "group", "box": {"x": 20, "y": 30, "w": 150, "h": 250},
        # dangling raster material left on the promoted container by reconstruction
        "src": str(panel), "meta": {"role": "shape"},
        "children": [
            {"id": "c_E006", "target": "image", "z": 30,
             "box": {"x": 5, "y": 100, "w": 60, "h": 80},
             "src": str(product), "meta": {"role": "product"}},
            {"id": "label", "target": "text", "z": 40, "text": "PRE PRO",
             "box": {"x": 10, "y": 10, "w": 80, "h": 20}, "style": {"fontSize": 14}},
        ],
    }]
    doc = build_design_json.build(tree, {"w": 200, "h": 300}, str(tmp_path),
                                   base_src=str(background))
    group = next(layer for layer in doc.layers if layer.id == "c_E003")
    assert group.type == "group"
    host = [c for c in group.children if c.meta.get("preserved_host_raster")]
    assert len(host) == 1, "promoted image host must re-emit its raster as a child"
    host = host[0]
    assert host.type == "image" and host.src is not None
    # It fills the frame in local coords and sits behind every real child.
    assert host.box == {"x": 0.0, "y": 0.0, "w": 150, "h": 250}
    assert host.z_index < min(c.z_index for c in group.children if c.id != host.id)
    # The staged asset exists and matches the panel raster dimensions (pixels survived).
    staged = tmp_path / host.src
    assert staged.exists()
    with Image.open(staged) as im:
        assert im.size == (150, 250)


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
    assert [layer.z_index for layer in doc.layers] == [-1_000_000, 20.0, 30.0, 35.0, 60.0]


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


def test_f11_fidelity_image_with_substitution_is_explained_not_unexplained(tmp_path):
    # A documented text->image substitution (052 c_B0) carries WHY it gave up. It is
    # explained-but-non-native: it must NOT be counted as an "unexplained" quiet
    # give-up, but it still costs native_leaf_ratio. A bare give-up with no evidence
    # stays unexplained (F4 anti-laundering, preserved).
    asset = tmp_path / "asset.png"
    Image.new("RGBA", (10, 10), "red").save(asset)
    doc = build_design_json.build([
        {"id": "headline", "target": "image", "src": str(asset),
         "box": {"x": 0, "y": 0, "w": 10, "h": 10},
         "meta": {"role": "headline", "fallback": True,
                  "substitution": {"from": "text", "reason": "low-fidelity-font"}}},
        {"id": "bare", "target": "image", "src": str(asset),
         "box": {"x": 12, "y": 0, "w": 10, "h": 10},
         "meta": {"role": "unknown", "fallback": True}},
    ], {"w": 30, "h": 10}, str(tmp_path))

    accounting = doc.meta["leaf_accounting"]
    assert accounting["fallback_raster_count"] == 2
    # Only the bare, evidence-free give-up is unexplained.
    assert accounting["unexplained_raster_count"] == 1
    assert accounting["unexplained_raster_ids"] == ["bare"]
    # Both are still non-native (the substitution image did not launder into native).
    assert accounting["native_leaf_count"] == 0


def test_text_stack_group_lifts_z_above_chrome_host(tmp_path):
    """002 regression: group scope must not bury editable headline children."""
    background = tmp_path / "background_clean.png"
    Image.new("RGB", (200, 400), "white").save(background)
    tree = [{
        "id": "host", "target": "shape", "box": {"x": 0, "y": 200, "w": 200, "h": 200},
        "meta": {"role": "badge", "z_band": "chrome"}, "z": 0,
    }, {
        "id": "text-stack", "target": "group", "z": 8,
        "box": {"x": 10, "y": 40, "w": 180, "h": 80},
        "meta": {"role": "text-stack"},
        "children": [{
            "id": "c_B2", "target": "text", "text": "KRACHTSPORT BUNDEL", "z": 8,
            "box": {"x": 0, "y": 40, "w": 180, "h": 30},
            "style": {"fontSize": 18, "color": "#000000"},
            "meta": {"role": "subheadline"},
        }],
    }]
    doc = build_design_json.build(
        tree, {"w": 200, "h": 400}, str(tmp_path), base_src=str(background),
    )
    by_id = {layer.id: layer for layer in doc.layers}
    assert by_id["text-stack"].z_index > by_id["host"].z_index
    assert by_id["text-stack"].z_index >= 60
    child = by_id["text-stack"].children[0]
    assert child.box["x"] >= 0 and child.box["y"] >= 0
    assert child.box["x"] + child.box["w"] <= by_id["text-stack"].box["w"]
    assert child.box["y"] + child.box["h"] <= by_id["text-stack"].box["h"]


def test_decoration_reanchors_to_owner_across_group_and_translation():
    """002 regression: a root-level strike must follow its owner word even when the
    owner is nested in a group AND was translated (box != visible_box) by layout.

    The owner sits inside a group at (100, 200) and was moved +20,+40 from its
    pre-layout frame (visible_box). The tight ink (prefit_ink_box) is recorded in
    the pre-move frame, so the reanchor must (a) find the owner across the whole tree
    and (b) shift the ink by (box - visible_box) into absolute space. Endpoint
    fractions 0..1 span the glyph ink exactly.
    """
    from src.schema import Layer

    owner = Layer(
        id="c_B5__w0", type="text", name="price", text="€63",
        box={"x": 50, "y": 60, "w": 120, "h": 140},          # final (translated)
        visible_box={"x": 30, "y": 20, "w": 120, "h": 140},  # pre-move frame
        meta={"prefit_ink_box": {"x": 32, "y": 70, "w": 110, "h": 50}},
    )
    group = Layer(
        id="text-stack", type="group", name="stack",
        box={"x": 100, "y": 200, "w": 300, "h": 300}, children=[owner],
    )
    deco = Layer(
        id="c_B5__decoration_0", type="shape", name="strike",
        box={"x": 0, "y": 0, "w": 1, "h": 1},
        stroke={"color": "#e1491b", "width": 3},
        meta={
            "native_decoration": True, "role": "strikethrough",
            "line": {"x0": 999, "y0": 999, "x1": 1099, "y1": 999, "thickness": 3.0},
            "anchor": {"owner_id": "c_B5", "word_text": "€63",
                       "fx0": 0.0, "fy0": 0.5, "fx1": 1.0, "fy1": 0.5},
        },
    )
    layers = [deco, group]
    moved = build_design_json._reanchor_decorations(layers)
    assert moved == 1
    assert deco.meta["reanchored_to"] == "c_B5__w0"
    # Owner final ink in absolute space: group(100,200)+prefit(32,70)+delta(20,40)
    #   x: 100+32+20 = 152 .. 152+110 = 262 ; y-center: 200+70+40 + 25 = 335
    line = deco.meta["line"]
    assert abs(line["x0"] - 152.0) <= 0.6
    assert abs(line["x1"] - 262.0) <= 0.6
    assert abs(line["y0"] - 335.0) <= 0.6 and abs(line["y1"] - 335.0) <= 0.6
    # The stale source line is preserved for provenance.
    assert deco.meta["source_line"]["x0"] == 999
