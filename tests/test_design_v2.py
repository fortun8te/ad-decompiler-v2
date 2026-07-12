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
