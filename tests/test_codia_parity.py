"""CPU tests for scripts/codia_parity.py — the Codia ground-truth parity gate.

Synthetic template/design pairs mirror the two real teardowns:
  * grouped UI screenshot (runs/codia-teardown-009.json): weight-split text runs,
    pill button, emoji cutouts, solid plates;
  * flat photo ad (runs/codia-teardown-2.json): 9 nodes, zero groups, display
    serif headline + Inter body, non-text as image cutouts.
"""
import json
import os

import pytest

from scripts import codia_parity


# --------------------------------------------------------------------------- builders

def _figma_color(rgb):
    return {"r": rgb[0] / 255.0, "g": rgb[1] / 255.0, "b": rgb[2] / 255.0, "a": 1.0}


def _t(node_id, chars, x, y, w, h, size=35, weight=400, family="Inter", rgb=(204, 204, 204)):
    return {
        "id": node_id, "name": chars[:16], "type": "TEXT",
        "characters": chars,
        "fills": [{"type": "SOLID", "color": _figma_color(rgb)}],
        "absoluteBoundingBox": {"x": x, "y": y, "width": w, "height": h},
        "style": {"fontFamily": family, "fontWeight": weight, "fontSize": size,
                  "letterSpacing": 0.0, "lineHeightPx": size * 1.21},
    }


def _img(node_id, x, y, w, h):
    return {
        "id": node_id, "name": "Image", "type": "RECTANGLE",
        "fills": [{"type": "IMAGE", "scaleMode": "FILL", "imageRef": "ref" + node_id}],
        "absoluteBoundingBox": {"x": x, "y": y, "width": w, "height": h},
    }


def _rect(node_id, x, y, w, h, rgb, radius=0):
    return {
        "id": node_id, "name": "Background", "type": "RECTANGLE",
        "fills": [{"type": "SOLID", "color": _figma_color(rgb)}],
        "cornerRadius": radius,
        "absoluteBoundingBox": {"x": x, "y": y, "width": w, "height": h},
    }


def _frame(node_id, name, x, y, w, h, children):
    return {
        "id": node_id, "name": name, "type": "FRAME", "children": children,
        "absoluteBoundingBox": {"x": x, "y": y, "width": w, "height": h},
    }


def _template_doc(root):
    return {"name": "t", "nodes": {"1:1": {"document": root}}}


def grouped_template():
    """Miniature of the 009 construction: plate + grouped rows + pill button."""
    button = _frame("1:20", "Button", 820, 130, 210, 70, [
        _rect("1:21", 830, 133, 202, 67, (238, 242, 243), radius=33),
        _t("1:22", "Volgend", 860, 145, 143, 44, size=35, weight=600, rgb=(29, 30, 31)),
    ])
    row = _frame("1:30", "Groups", 0, 900, 1000, 100, [
        _img("1:31", 30, 920, 50, 50),
        _t("1:32", "257", 100, 925, 60, 44),
        _t("1:33", "121K", 430, 921, 82, 44, weight=700),
        _t("1:34", "weergaven", 520, 923, 182, 46, weight=300, rgb=(110, 111, 114)),
    ])
    root = _frame("1:0", "Figma design - x.png", 0, 0, 1000, 1000, [
        _frame("1:1", "Root", 0, 0, 1000, 1000, [
            _rect("1:2", 0, 0, 1000, 1000, (0, 0, 0)),
            _t("1:3", "LAATSTE SALE VAN 2026", 48, 318, 651, 47, size=37, rgb=(218, 218, 218)),
            _img("1:4", 711, 322, 26, 38),   # emoji cutout
            button, row,
        ]),
    ])
    return _template_doc(root)


def flat_template():
    """Miniature of the 041 construction: flat tree, serif display headline."""
    root = _frame("2:0", "Figma design - y.webp", 0, 0, 1000, 1000, [
        _frame("2:1", "Root", 0, 0, 1000, 1000, [
            _img("2:2", 0, 0, 1000, 1000),     # full-canvas photo plate (not a cutout)
            _img("2:3", 72, 771, 217, 217),    # product cutout
            _t("2:4", "One Step to\nBeach-Ready Waves.", 76, 85, 900, 203,
               size=90, weight=700, family="Playfair Display"),
            _t("2:5", "All-day hold", 72, 390, 218, 47, size=36, weight=700),
        ]),
    ])
    return _template_doc(root)


def _leaf_text(node_id, text, x, y, w, h, size=35, weight=400, family="Inter",
               spacing=0.0, color="#cccccc"):
    return {"id": node_id, "type": "text", "name": f'Text "{text[:12]}"',
            "box": {"x": x, "y": y, "w": w, "h": h}, "text": text,
            "style": {"fontFamily": family, "fontWeight": weight, "fontSize": size,
                      "letterSpacing": spacing, "color": color}}


def perfect_grouped_design():
    return {"canvas": {"w": 1000, "h": 1000}, "layers": [
        {"id": "bg", "type": "shape", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "shape_kind": "rect", "fill": {"kind": "flat", "color": "#000000"}},
        _leaf_text("t1", "LAATSTE SALE VAN 2026", 48, 318, 651, 47, size=37),
        {"id": "emoji", "type": "image", "box": {"x": 711, "y": 322, "w": 26, "h": 38},
         "src": "assets/emoji.png"},
        {"id": "btn", "type": "group", "box": {"x": 830, "y": 133, "w": 202, "h": 67},
         "radius": 33, "fill": {"kind": "flat", "color": "#eef2f3"},
         "children": [_leaf_text("t2", "Volgend", 30, 12, 143, 44, size=35, weight=600,
                                 color="#1d1e1f")]},
        {"id": "row", "type": "group", "box": {"x": 0, "y": 900, "w": 1000, "h": 100},
         "children": [
             {"id": "i1", "type": "image", "box": {"x": 30, "y": 20, "w": 50, "h": 50},
              "src": "assets/icon.png"},
             _leaf_text("t3", "257", 100, 25, 60, 44),
             _leaf_text("t4", "121K", 430, 21, 82, 44, weight=700),
             _leaf_text("t5", "weergaven", 520, 23, 182, 46, weight=300),
         ]},
    ]}


def degraded_grouped_design():
    """What our pipeline used to ship: raster slices, wrong family, tracking noise."""
    return {"canvas": {"w": 1000, "h": 1000}, "layers": [
        {"id": "bg", "type": "image", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "src": "assets/plate.png"},
        {"id": "s1", "type": "image", "box": {"x": 48, "y": 318, "w": 651, "h": 47},
         "name": "LAATSTE SALE VAN 2026 — raster slice (low confidence)",
         "meta": {"raster_slice": True}},
        {"id": "s2", "type": "image", "box": {"x": 20, "y": 920, "w": 682, "h": 46},
         "name": "05:00 PM . 121K weergaver — raster slice",
         "meta": {"raster_slice": True}},
        _leaf_text("t2", "Volgend", 860, 145, 143, 44, size=37, weight=700,
                   family="Arimo", spacing=-1.8),
        _leaf_text("t3", "257", 100, 925, 60, 44, family="Caladea", spacing=1.4),
    ]}


# --------------------------------------------------------------------------- tests

def test_normalize_text_strips_emoji_and_unifies_dots():
    assert codia_parity.normalize_text("LAATSTE SALE 2026 ⌛") == "laatste sale 2026"
    assert (codia_parity.normalize_text("05:00 PM · 12-05-2026 ·")
            == codia_parity.normalize_text("05:00 PM . 12-05-2026 ."))


def test_similarity_containment_matches_line_inside_slice():
    slice_text = "05:00 PM . 12-05-2026 - 121K weergaver"
    assert codia_parity._similarity("121K", slice_text) > 0.9
    assert codia_parity._similarity("weergaven", slice_text) > 0.8
    assert codia_parity._similarity("Volgend", slice_text) < 0.4


def test_template_extraction_grouped():
    template = codia_parity.load_codia_template(grouped_template())
    assert len(template["texts"]) == 5
    assert len(template["cutouts"]) == 2          # emoji + row icon (plate rect is solid)
    assert template["button"] is not None
    assert template["button"]["pill"]["color"] == (238, 242, 243)
    assert template["complexity"] == "complex"
    assert template["group_count"] == 2           # Button + engagement row
    assert not any(t["display"] for t in template["texts"])


def test_template_extraction_flat_photo_ad():
    template = codia_parity.load_codia_template(flat_template())
    assert template["complexity"] == "simple"
    assert template["group_count"] == 0
    # The full-canvas photo is the plate, not a cutout; the product is a cutout.
    assert len(template["cutouts"]) == 1
    headline = [t for t in template["texts"] if t["display"]]
    assert len(headline) == 1 and headline[0]["fontFamily"] == "Playfair Display"


def test_perfect_design_scores_high():
    template = codia_parity.load_codia_template(grouped_template())
    ours = codia_parity.load_our_design(perfect_grouped_design())
    report = codia_parity.compare(template, ours)
    assert report["scores"]["native_text_ratio"] == 1.0
    assert report["scores"]["font_family"] == 1.0
    assert report["scores"]["font_weight"] == 1.0
    assert report["scores"]["letter_spacing"] == 1.0
    assert report["scores"]["icon_cutouts"] == 1.0
    assert report["scores"]["button"] == 1.0
    assert report["overall"] >= 95.0


def test_degraded_design_scores_low_and_flags_slices():
    template = codia_parity.load_codia_template(grouped_template())
    ours = codia_parity.load_our_design(degraded_grouped_design())
    report = codia_parity.compare(template, ours)
    perfect = codia_parity.compare(template,
                                   codia_parity.load_our_design(perfect_grouped_design()))
    assert report["overall"] < perfect["overall"] - 20
    # Raster slices matched their lines but are not native text.
    assert report["scores"]["native_text_ratio"] == pytest.approx(2 / 5)
    # Tracking noise and the wrong family are called out.
    assert report["scores"]["letter_spacing"] == 0.0
    assert report["scores"]["font_family"] == 0.0
    missing = report["detail"]["native_text"]["missing_or_raster"]
    assert any("LAATSTE" in text for text in missing)


def test_weight_split_runs_must_be_separate_nodes():
    """A single merged node cannot satisfy both the 700 and 300 runs."""
    template = codia_parity.load_codia_template(grouped_template())
    merged = perfect_grouped_design()
    row = merged["layers"][-1]
    row["children"] = [child for child in row["children"]
                       if child["id"] not in ("t4", "t5")]
    row["children"].append(_leaf_text("t45", "121K weergaven", 430, 21, 272, 46,
                                      weight=400))
    ours = codia_parity.load_our_design(merged)
    report = codia_parity.compare(template, ours)
    # The merged node is claimed by one run; the sibling run counts as missing, so
    # both native coverage and the weight table register the failure.
    assert report["scores"]["native_text_ratio"] < 1.0
    weight_rows = {row["text"]: row["match"] for row in report["detail"]["font_weight"]}
    claimed = [key for key in ("121K", "weergaven") if key in weight_rows]
    assert claimed and all(weight_rows[key] is False for key in claimed)
    missing = report["detail"]["native_text"]["missing_or_raster"]
    assert any(text in ("121K", "weergaven") for text in missing)


def test_flat_scene_penalizes_groups_and_bloat():
    template = codia_parity.load_codia_template(flat_template())
    flat = {"canvas": {"w": 1000, "h": 1000}, "layers": [
        {"id": "bg", "type": "image", "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
         "src": "assets/plate.png"},
        {"id": "prod", "type": "image", "box": {"x": 72, "y": 771, "w": 217, "h": 217},
         "src": "assets/prod.png"},
        _leaf_text("h", "One Step to\nBeach-Ready Waves.", 76, 85, 900, 203,
                   size=90, weight=700, family="Playfair Display"),
        _leaf_text("c", "All-day hold", 72, 390, 218, 47, size=36, weight=700),
    ]}
    report = codia_parity.compare(template, codia_parity.load_our_design(flat))
    assert report["scores"]["flatness"] == 1.0
    assert report["scores"]["node_budget"] == 1.0
    assert report["scores"]["headline_font"] == 1.0

    wrapped = dict(flat)
    wrapped["layers"] = [flat["layers"][0],
                         {"id": "g1", "type": "group",
                          "box": {"x": 0, "y": 0, "w": 1000, "h": 1000},
                          "children": flat["layers"][1:]}]
    report2 = codia_parity.compare(template, codia_parity.load_our_design(wrapped))
    assert report2["scores"]["flatness"] == 0.5

    # Serif class fallback: Georgia instead of Playfair scores half.
    georgia = json.loads(json.dumps(flat))
    georgia["layers"][2]["style"]["fontFamily"] = "Georgia"
    report3 = codia_parity.compare(template, codia_parity.load_our_design(georgia))
    assert report3["scores"]["headline_font"] == 0.5


def test_complexity_override_forces_flat_expectation():
    template = codia_parity.load_codia_template(grouped_template())
    ours = codia_parity.load_our_design(perfect_grouped_design())
    default = codia_parity.compare(template, ours)
    forced = codia_parity.compare(template, ours, complexity="simple")
    assert forced["scores"]["flatness"] < default["scores"]["flatness"]
    assert forced["detail"]["flatness"]["complexity"] == "simple"


def test_paragraph_integrity_reported():
    template = codia_parity.load_codia_template(flat_template())
    split = {"canvas": {"w": 1000, "h": 1000}, "layers": [
        _leaf_text("h1", "One Step to", 76, 85, 900, 100, size=90, weight=700,
                   family="Playfair Display"),
        _leaf_text("c", "All-day hold", 72, 390, 218, 47, size=36, weight=700),
    ]}
    report = codia_parity.compare(template, codia_parity.load_our_design(split))
    rows = report["detail"]["paragraph_integrity"]
    assert rows and rows[0]["single_node"] is False


REAL_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "runs",
                             "codia-teardown-009.json")
REAL_DESIGN = os.path.join(os.path.dirname(__file__), "..", "runs", "benchmark-final",
                           "009_attached_885c19be02ccf229", "design.json")


@pytest.mark.skipif(not (os.path.exists(REAL_TEMPLATE) and os.path.exists(REAL_DESIGN)),
                    reason="real 009 teardown/run artifacts not present")
def test_real_009_smoke():
    report = codia_parity.run(REAL_DESIGN, REAL_TEMPLATE)
    assert 0.0 < report["overall"] < 100.0
    # Codia ships 16 text lines and 11 cutouts on this fixture.
    assert report["detail"]["native_text"]["total"] == 16
    assert report["detail"]["icon_cutouts"]["of"] == 11
    assert report["detail"]["node_budget"]["codia_nodes"] == 38
