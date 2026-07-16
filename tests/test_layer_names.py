"""Fast local Figma layer naming — no VLM, GPU, or network."""
from __future__ import annotations

import time

from PIL import Image

from src import build_design_json


def test_role_and_text_produce_codia_style_names():
    assert build_design_json._name({
        "id": "c_B0", "target": "text", "text": "Volgend",
        "meta": {"role": "cta"},
    }) == "CTA / Volgend"
    assert build_design_json._name({
        "id": "c_B1", "target": "text",
        "text": "LAATSTE SITE VAN HET JAAR",
        "meta": {"role": "headline"},
    }) == "Headline / LAATSTE SITE VAN HET JAAR"
    assert build_design_json._name({
        "id": "c_B2", "target": "text",
        "text": "De Vakantie\nBegint Hier",
        "meta": {"semantic_role": "body-copy", "role": "body"},
    }) == "Body / De Vakantie Begint Hier"
    assert build_design_json._name({
        "id": "c_E006", "target": "icon", "meta": {"role": "arrow"},
    }) == "Arrow"
    assert build_design_json._name({
        "id": "c_L1", "target": "icon", "meta": {"role": "callout_leader"},
    }) == "Arrow"
    assert build_design_json._name({
        "id": "c_E013", "target": "image", "meta": {"role": "product"},
    }) == "Product"
    assert build_design_json._name({
        "id": "c_co", "target": "text", "text": "Vitamin D3 for immune health",
        "meta": {"role": "callout"},
    }) == "Callout / Vitamin D3 for immune health"
    assert build_design_json._name({
        "id": "c_fda", "target": "text",
        "text": "*These statements have not been evaluated by the FDA.",
        "meta": {"role": "disclaimer"},
    }).startswith("Disclaimer /")
    assert build_design_json._name({
        "id": "c_E011", "target": "image", "meta": {"role": "photo"},
    }) == "Photo"
    assert build_design_json._name({
        "id": "c_E008", "target": "icon", "meta": {"role": "icon"},
    }) == "Icon"
    assert build_design_json._name({
        "id": "bg", "target": "image", "meta": {"role": "background"},
    }) == "Background"
    assert build_design_json._name({
        "id": "mb", "target": "group", "meta": {"role": "message-bubble"},
        "children": [],
    }) == "Message"
    assert build_design_json._name({
        "id": "sp", "target": "group", "meta": {"role": "stat-pill"},
        "text": "200%", "children": [],
    }) in {"Stat", "Stat / 200%"}
    assert build_design_json._name({
        "id": "hc", "target": "group", "meta": {"role": "header-cluster"},
        "children": [],
    }) == "Header"


def test_machine_and_vlm_names_are_ignored():
    # Reconstruct fallback leftover must not leak into Figma.
    assert build_design_json._name({
        "id": "c_E014", "target": "image",
        "name": "c_E014 — raster slice (low confidence)",
        "meta": {"role": "logo"},
    }) == "Logo"
    # Old quote-style / technical suffixes are rewritten from role.
    assert build_design_json._name({
        "id": "c_E013", "target": "image",
        "name": "Product — swappable crop",
        "meta": {"role": "product"},
    }) == "Product"
    assert build_design_json._name({
        "id": "c_E006", "target": "icon",
        "name": "Arrow — vector",
        "meta": {"role": "arrow"},
    }) == "Arrow"
    # vlm_name is not on the design hot path.
    assert build_design_json._name({
        "id": "x", "target": "image",
        "meta": {"role": "avatar", "vlm_name": "VLM said cool person"},
    }) == "Avatar"


def test_clean_semantic_name_survives():
    assert build_design_json._name({
        "id": "avatar", "target": "image",
        "meta": {"role": "avatar", "semantic_name": "Creator avatar"},
    }) == "Creator avatar"


def test_text_bearing_banner_and_badge_names():
    assert build_design_json._name({
        "id": "c_ban", "target": "shape",
        "meta": {
            "role": "banner", "text_bearing_shell": True, "plate_shell": True,
            "shell_text_snippet": "ALMOST SOLD OUT...",
        },
    }) == "Banner / ALMOST SOLD OUT..."
    assert build_design_json._name({
        "id": "c_seal", "target": "shape",
        "meta": {
            "role": "badge", "text_bearing_shell": True, "plate_shell": True,
            "shell_text_snippet": "LIMITED TIME OFFER",
        },
    }) == "Badge / LIMITED TIME OFFER"
    assert build_design_json._name({
        "id": "c_chip", "target": "image",
        "meta": {
            "role": "banner", "shell_raster_chip": True,
            "shell_text_snippet": "ALMOST SOLD OUT...",
        },
    }) == "Banner / ALMOST SOLD OUT..."
    assert build_design_json._name({
        "id": "c_out", "target": "shape",
        "meta": {
            "role": "callout", "text_bearing_shell": True, "plate_shell": True,
            "stroke_outline_shell": True,
            "shell_text_snippet": "Daily digestive support",
        },
    }) == "Callout / Daily digestive support"
    assert build_design_json._name({
        "id": "c_save", "target": "shape",
        "meta": {
            "role": "seal", "text_bearing_shell": True, "plate_shell": True,
            "shell_text_snippet": "SAVE 10%",
        },
    }) == "Badge / SAVE 10%"
    assert build_design_json._name({
        "id": "c_cta", "target": "text", "text": "Start for £1.00/day",
        "meta": {"role": "cta"},
    }) == "CTA / Start for £1.00/day"


def test_text_snippet_truncates_and_collapses_whitespace():
    long = "A" * 40
    name = build_design_json._name({
        "id": "t", "target": "text", "text": f"  Hello\n\t{long}  ",
        "meta": {"role": "body"},
    })
    assert name.startswith("Body / Hello A")
    assert "\n" not in name and "\t" not in name
    assert name.endswith("…")
    assert len(name) <= len("Body / ") + 28


def test_group_roles_use_layout_labels():
    assert build_design_json._name({
        "id": "text-stack-1", "target": "group",
        "meta": {"role": "text-stack", "semantic_name": "Text Stack"},
    }) == "Text Stack"
    assert build_design_json._name({
        "id": "btn", "target": "group", "text": "Volgend",
        "meta": {"role": "button"},
    }) == "Button / Volgend"


def test_sibling_dedupe_suffix_only_when_needed(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(a)
    Image.new("RGBA", (4, 4), (4, 5, 6, 255)).save(b)
    doc = build_design_json.build([
        {"id": "p1", "target": "image", "box": {"x": 0, "y": 0, "w": 4, "h": 4},
         "src": str(a), "meta": {"role": "product"}, "z": 1},
        {"id": "p2", "target": "image", "box": {"x": 10, "y": 0, "w": 4, "h": 4},
         "src": str(b), "meta": {"role": "product"}, "z": 2},
        {"id": "h", "target": "text", "text": "Hi", "box": {"x": 0, "y": 10, "w": 20, "h": 10},
         "meta": {"role": "headline"}, "z": 3},
    ], {"w": 40, "h": 40}, str(tmp_path))
    names = [layer.name for layer in doc.layers]
    assert names.count("Product") == 1
    assert "Product / 2" in names
    assert any(n.startswith("Headline") for n in names)


def test_background_layer_is_simply_background(tmp_path):
    plate = tmp_path / "background_clean.png"
    Image.new("RGB", (8, 8), "white").save(plate)
    doc = build_design_json.build([], {"w": 8, "h": 8}, str(tmp_path), base_src=str(plate))
    assert doc.layers[0].name == "Background"


def test_ui_label_role_gets_readable_label():
    # wordmark._UI_LABEL / _SOCIAL_HANDLE classify social-chrome text as role
    # "ui-label"; before the _ROLE_LABELS entry this fell through to naive
    # title-casing ("Ui Label"), which is not a name a designer would write.
    assert build_design_json._name({
        "id": "c_h1", "target": "text", "text": "@UpfrontFood",
        "meta": {"role": "ui-label"},
    }) == "Label / @UpfrontFood"


def test_generic_role_group_gets_content_derived_name():
    # Benchmark 002: an element-fusion residual group wrapping 3 products, 2
    # prices, an arrow and underline/strikethrough decorations had meta.role
    # "shape" (a low-confidence catch-all), so it named itself "Shape" and
    # collided with a sibling image also named "Shape" -> "Shape / 2". A
    # designer opening the file could not tell what that group was for. The
    # name must instead describe the group's dominant content.
    candidate = {
        "id": "c_E003", "target": "group", "meta": {"role": "shape"},
        "children": [
            {"id": "c1", "target": "image", "meta": {"role": "shape"}},
            {"id": "c2", "target": "image", "meta": {"role": "product"}},
            {"id": "c3", "target": "image", "meta": {"role": "product"}},
            {"id": "c4", "target": "image", "meta": {"role": "product"}},
            {"id": "c5", "target": "image", "meta": {"role": "arrow"}},
            {"id": "c6", "target": "shape", "meta": {"role": "underline"}},
            {"id": "c7", "target": "shape", "meta": {"role": "strikethrough"}},
            {"id": "c8", "target": "text", "meta": {"role": "price"}, "text": "€63"},
            {"id": "c9", "target": "text", "meta": {"role": "price"}, "text": "€49"},
            {"id": "c10", "target": "text", "meta": {"role": "subheadline"},
             "text": "KOOP NU VIA UPFRONTNL"},
        ],
    }
    assert build_design_json._name(candidate) == "Product + Price"


def test_generic_group_with_single_child_takes_child_label():
    candidate = {
        "id": "g1", "target": "group", "meta": {"role": "shape"},
        "children": [{"id": "t1", "target": "text", "meta": {"role": "subheadline"},
                      "text": "KRACHTSPORT BUNDEL"}],
    }
    assert build_design_json._name(candidate) == "Subheadline"


def test_decoration_only_group_falls_back_to_decoration_labels():
    candidate = {
        "id": "g3", "target": "group", "meta": {"role": "shape"},
        "children": [
            {"id": "d1", "target": "shape", "meta": {"role": "underline"}},
            {"id": "d2", "target": "shape", "meta": {"role": "strikethrough"}},
        ],
    }
    assert build_design_json._name(candidate) == "Underline + Strikethrough"


def test_named_group_roles_are_unaffected_by_content_derivation():
    # text-stack (and other roles already in _ROLE_LABELS) must keep their
    # existing, deliberately-chosen names -- the content-derivation fallback
    # only applies to the generic/low-confidence roles.
    candidate = {
        "id": "g2", "target": "group", "meta": {"role": "text-stack"},
        "children": [{"id": "t1", "target": "text", "meta": {"role": "headline"}, "text": "X"}],
    }
    assert build_design_json._name(candidate) == "Text Stack"


def test_empty_generic_group_keeps_group_fallback():
    assert build_design_json._name({"id": "g4", "target": "group", "meta": {}}) == "Group"


def test_text_fidelity_fallback_is_flagged_in_name():
    # A text candidate demoted to raster because OCR/fit confidence was too low
    # must say so in the name -- otherwise a designer can't tell a genuine
    # non-editable pixel fallback apart from an intentionally-image layer.
    name = build_design_json._name({
        "id": "L6", "target": "image", "text": "muddled copy",
        "meta": {"low_fidelity": True, "substitution": {"from": "text", "to": "image"}},
    })
    assert "fallback" in name.lower()


def test_long_role_label_snippet_stays_within_forty_chars():
    # Disclaimer/legal copy has a long role label; the "Label / snippet" name must
    # still fit the ~40-char designer budget instead of growing past it the way a
    # fixed 28-char snippet cap would for longer labels.
    long_text = "These statements have not been evaluated by the FDA " * 3
    name = build_design_json._name({
        "id": "c_fda", "target": "text", "text": long_text,
        "meta": {"role": "disclaimer"},
    })
    assert len(name) <= 40
    assert name.startswith("Disclaimer / ")


def test_naming_is_sync_local_and_fast():
    candidates = [
        {"id": f"c_E{i:03d}", "target": "text", "text": f"Line {i} copy here",
         "meta": {"role": "body" if i % 2 else "headline"}}
        for i in range(500)
    ]
    t0 = time.perf_counter()
    names = [build_design_json._name(c) for c in candidates]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(names) == 500
    assert all("/" in n or n in {"Headline", "Body"} for n in names)
    # Pure string work over 500 nodes must stay well under a millisecond budget on CI.
    assert elapsed_ms < 50.0, f"naming too slow: {elapsed_ms:.2f}ms"
