"""stage_test_run.py — stages a hand-built design.json into the Figma-plugin bridge inbox
so the plugin's Refresh/Create-copy flow can be exercised without a real GPU pipeline run.

Usage:
    python stage_test_run.py [inbox_dir]

Defaults to C:\\figma-inbox on Windows (matches what figma_bridge.py prints on startup),
or ~/figma-inbox otherwise. Point --inbox at the same directory the bridge server is
watching (whatever path it printed when it started).
"""
from __future__ import annotations
import json
import os
import sys

from PIL import Image, ImageDraw


def main() -> None:
    inbox = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\figma-inbox" if os.name == "nt" else os.path.expanduser("~/figma-inbox")
    )
    assets = os.path.join(inbox, "assets")
    os.makedirs(assets, exist_ok=True)

    img = Image.new("RGB", (360, 440), (200, 210, 230))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 320, 400], fill=(120, 150, 200))
    img.save(os.path.join(assets, "photo.png"))

    design = {
        "schema_version": 2,
        "id": "manual-smoke-test",
        "name": "Manual smoke test",
        "canvas": {"w": 600, "h": 600},
        "layers": [
            {"id": "background", "type": "shape", "name": "Background", "box": {"x": 0, "y": 0, "w": 600, "h": 600},
             "fill": {"kind": "flat", "color": "#fafafa"}, "z_index": 0},
            {"id": "copy-stack", "type": "frame", "name": "Copy stack", "box": {"x": 40, "y": 30, "w": 300, "h": 180},
             "layout": {"mode": "vertical", "gap": 12, "padding": 16}, "z_index": 1,
             "children": [
                {"id": "headline", "type": "text", "name": "Headline", "box": {"x": 16, "y": 16, "w": 268, "h": 60},
                 "text": "A better\nheadline",
                 "style": {"fontFamily": "Missing Sans", "fontStyle": "Bold", "fontWeight": 700, "color": "#111111",
                            "lineCount": 2,
                            "fontCandidates": [{"family": "Missing Sans", "style": "Bold", "score": 0.9},
                                                {"family": "Inter", "style": "Bold", "weight": 700, "score": 0.82}]},
                 "meta": {"role": "Headline"}}
             ]},
            {"id": "gradient-copy", "type": "text", "name": "Gradient copy", "box": {"x": 40, "y": 230, "w": 220, "h": 34},
             "text": "COLOUR TYPE",
             "style": {"fontFamily": "Inter", "fontSize": 20, "color": "#111111",
                        "fills": [{"kind": "linear", "angle": 0, "stops": [{"offset": 0, "color": "#ff3366"},
                                                                             {"offset": 1, "color": "#6633ff"}]}],
                        "strokes": [{"color": "#111111", "width": 1}]},
             "z_index": 1.5},
            {"id": "photo", "type": "image", "name": "Photo", "box": {"x": 360, "y": 80, "w": 180, "h": 220},
             "src": "assets/photo.png", "mask": {"kind": "ellipse"}, "z_index": 2},
            {"id": "badge", "type": "group", "name": "Badge", "box": {"x": 50, "y": 420, "w": 160, "h": 60},
             "children": [
                {"id": "pill", "type": "shape", "name": "Pill", "box": {"x": 0, "y": 0, "w": 160, "h": 60},
                 "fill": "#111111", "radius": 30},
                {"id": "label", "type": "text", "name": "Label", "box": {"x": 20, "y": 14, "w": 120, "h": 28},
                 "text": "BUY NOW", "style": {"fontFamily": "Inter", "fontSize": 18, "color": "#ffffff", "align": "center"}}
             ]},
            {"id": "styled-gradient-card", "type": "shape", "name": "Styled gradient card",
             "box": {"x": 40, "y": 510, "w": 220, "h": 64},
             "style": {"fills": [{"kind": "linear-gradient", "angle": 90,
                                    "stops": [{"color": "#ff2200", "offset": 0}, {"color": "#0044ff", "offset": 100}]}],
                        "strokes": [{"color": "#ffffff", "width": 2, "align": "inside", "dash": [4, 2]}],
                        "effects": [{"type": "drop-shadow", "color": "#00000066", "x": 0, "y": 3, "blur": 8, "spread": 1}]},
             "radius": {"topLeft": 12, "topRight": 8, "bottomRight": 4, "bottomLeft": 0}, "z_index": 5},
            {"id": "masked-photo", "type": "image", "name": "Masked photo", "box": {"x": 300, "y": 490, "w": 200, "h": 90},
             "src": "assets/photo.png", "mask": {"kind": "rounded-rect", "box": {"x": 324, "y": 500, "w": 140, "h": 70}, "radius": 14},
             "z_index": 6},
        ],
    }
    with open(os.path.join(inbox, "design.json"), "w") as f:
        json.dump(design, f, indent=2)

    inbox_manifest = {
        "schema_version": 2,
        "doc_id": "manual-smoke-test",
        "design": "design.json",
        "staged_dir": ".",
        "staged_at": "2026-07-12T00:00:00Z",
        "summary": {"layers": len(design["layers"]), "note": "hand-built fixture for manual Figma-plugin testing, not a real ad run"},
        "export_to": os.path.join(inbox, "figma_export.png"),
        "run_dir": inbox,
    }
    with open(os.path.join(inbox, "inbox.json"), "w") as f:
        json.dump(inbox_manifest, f, indent=2)

    print("staged test run into", inbox)


if __name__ == "__main__":
    main()
