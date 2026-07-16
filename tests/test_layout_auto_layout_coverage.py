"""CPU tests for the widened Auto Layout coverage in ``src.layout``.

Every box below is real geometry lifted from ``runs/benchmark-final`` (the run id is
named in each test) so the correctness gates are exercised against the exact scenes
that motivated the change.  The guiding rule is Codia's: a wrong Auto Layout is worse
than an absolute layer, so each "should fire" case is paired with a "must reproduce
geometry" assertion, and each loosening is paired with a "must NOT fire" guard.
"""
from src import layout


def _all_groups(nodes):
    for node in nodes:
        if node.get("target") == "group":
            yield node
        yield from _all_groups(node.get("children") or [])


def _text_node(node_id, text, box, role="body"):
    return {"id": node_id, "target": "text", "text": text, "box": box, "meta": {"role": role}}


# --------------------------------------------------------------------------- #
# Widened vertical text stacks (Fix A: role filter -> geometry gate)
# --------------------------------------------------------------------------- #

def test_offer_and_label_copy_lines_now_form_a_vertical_stack():
    """Roles outside the old allowlist (offer/label/body-copy) still stack when the
    geometry agrees.  Before, only headline/subhead/body were eligible."""
    candidates = [
        _text_node("l1", "50% thicker for better", {"x": 111, "y": 705, "w": 247, "h": 19}, "offer"),
        _text_node("l2", "durability", {"x": 112, "y": 733, "w": 105, "h": 20}, "label"),
        _text_node("l3", "Aluminium valve built to last", {"x": 111, "y": 781, "w": 329, "h": 40}, "label"),
    ]
    tree = layout.infer(candidates, {"w": 1000, "h": 1000}, {})
    stack = next(n for n in tree if n.get("meta", {}).get("role") == "text-stack")
    assert stack["layout"]["mode"] == "VERTICAL"
    assert [c["id"] for c in stack["children"]] == ["l1", "l2", "l3"]


def test_axis_label_column_forms_stack_with_geometry_true_spacing():
    """094: the '30 / 20 / 10' scale labels (role=label) — a real evenly spaced column.

    itemSpacing must equal the measured median gap so the column keeps its spacing
    when the frame is resized."""
    candidates = [
        _text_node("n30", "30", {"x": 55.9, "y": 1222.5, "w": 31.6, "h": 26.2}, "label"),
        _text_node("n20", "20", {"x": 55.9, "y": 1282.5, "w": 31.6, "h": 26.2}, "label"),
        _text_node("n10", "10", {"x": 58.0, "y": 1344.0, "w": 28.0, "h": 26.0}, "label"),
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, {})
    stack = next(n for n in tree if n.get("meta", {}).get("role") == "text-stack")
    assert stack["layout"]["mode"] == "VERTICAL"
    # gaps: 1282.5-(1222.5+26.2)=33.8 and 1344-(1282.5+26.2)=35.3 -> median 34.55
    assert abs(stack["layout"]["gap"] - 34.55) < 0.5
    # shared centre column, not a left edge -> CENTER counter alignment
    assert stack["layout"]["counterAlign"] == "CENTER"


def test_centered_headline_pair_forms_stack_reproducing_gap():
    """101: 'NOT ALL TPU TUBES' / 'ARE BUILT THE SAME!' — a centred two-line headline."""
    candidates = [
        _text_node("h1", "NOT ALL TPU TUBES", {"x": 203.9, "y": 56.6, "w": 592.0, "h": 45.6}, "headline"),
        _text_node("h2", "ARE BUILT THE SAME!", {"x": 177.7, "y": 123.0, "w": 640.6, "h": 43.0}, "headline"),
    ]
    tree = layout.infer(candidates, {"w": 1000, "h": 1000}, {})
    stack = next(n for n in tree if n.get("meta", {}).get("role") == "text-stack")
    assert stack["layout"]["mode"] == "VERTICAL"
    # single gap = 123.0 - (56.6 + 45.6) = 20.8, reproduced exactly for two items
    assert abs(stack["layout"]["gap"] - 20.8) < 0.1
    assert stack["layout"]["counterAlign"] == "CENTER"


# --------------------------------------------------------------------------- #
# Correctness guard for the widened alignment test (Fix A regression)
# --------------------------------------------------------------------------- #

def test_wide_headline_does_not_absorb_a_mid_canvas_cta_into_a_stack():
    """025: a full-bleed headline must not swallow a narrower CTA that merely sits
    within its horizontal span.  This is the exact false stack the max-width overlap
    denominator removes; the headline stays an independent layer."""
    candidates = [
        _text_node("why", "Why Everyone's Switching", {"x": 205.6, "y": 272.5, "w": 663.6, "h": 159.7}, "subheadline"),
        _text_node("cta", "$100+ in savings! Shop Now", {"x": 739.3, "y": 607.1, "w": 148.7, "h": 128.3}, "price"),
        _text_node("after", "AFTER", {"x": 730.9, "y": 858.8, "w": 149.8, "h": 37.5}, "subheadline"),
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, {})
    # The wide headline is never a stack member with the far, misaligned CTA.
    for group in _all_groups(tree):
        ids = {c.get("id") for c in group.get("children") or []}
        assert not ({"why"} & ids and {"cta"} & ids), "wide headline wrongly stacked with CTA"
    assert any(n.get("id") == "why" and n.get("target") == "text" for n in tree)


# --------------------------------------------------------------------------- #
# Horizontal peer rows (Fix B)
# --------------------------------------------------------------------------- #

def test_icon_and_label_form_a_horizontal_row_with_true_item_spacing():
    """025: a check icon + 'Industrial-grade' label — a labelled feature row.

    The emitted itemSpacing must place the label exactly where it sits so a resize
    keeps the icon/label pairing intact."""
    candidates = [
        {"id": "ico", "target": "icon", "box": {"x": 75, "y": 1191, "w": 34, "h": 35},
         "meta": {"role": "verified"}},
        _text_node("lab", "Industrial-grade", {"x": 111, "y": 1192, "w": 251, "h": 42}, "label"),
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, {})
    row = next(n for n in tree if n.get("meta", {}).get("role") == "text-row")
    assert row["layout"]["mode"] == "HORIZONTAL"
    # gap = 111 - (75 + 34) = 2
    assert row["layout"]["gap"] == 2
    assert row["layout"]["itemSpacing"] == 2
    children = {c["id"]: c for c in row["children"]}
    # relative geometry: icon at x=0, label reachable from icon.w + itemSpacing
    icon, label = children["ico"], children["lab"]
    assert icon["box"]["x"] == 0
    assert icon["box"]["x"] + icon["box"]["w"] + row["layout"]["itemSpacing"] == label["box"]["x"]


def test_three_evenly_spaced_labels_form_a_row():
    """A genuine three-item promo bar: consistent spacing, similar height."""
    candidates = [
        _text_node("a", "FREE SHIPPING", {"x": 100, "y": 500, "w": 180, "h": 40}, "offer"),
        _text_node("b", "EASY RETURNS", {"x": 320, "y": 500, "w": 180, "h": 40}, "offer"),
        _text_node("c", "24/7 SUPPORT", {"x": 540, "y": 500, "w": 180, "h": 40}, "offer"),
    ]
    tree = layout.infer(candidates, {"w": 900, "h": 900}, {})
    row = next(n for n in tree if n.get("meta", {}).get("role") == "text-row")
    assert row["layout"]["mode"] == "HORIZONTAL"
    assert [c["id"] for c in row["children"]] == ["a", "b", "c"]
    # gaps: 320-280=40 and 540-500=40 -> itemSpacing 40 reproduces both columns
    assert row["layout"]["itemSpacing"] == 40


def test_far_apart_display_fragments_do_not_form_a_row():
    """088-style: two wide display fragments far apart share a baseline but are not a
    row.  Width-scaled tolerance would fuse them; line-height tolerance must not."""
    candidates = [
        _text_node("f1", "OFF BLACK FRIDAY", {"x": 4, "y": 1226, "w": 377, "h": 72}, "offer"),
        _text_node("f2", "OFF", {"x": 615, "y": 1241, "w": 94, "h": 90}, "offer"),
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, {})
    assert not any(n.get("meta", {}).get("role") == "text-row" for n in _all_groups(tree))


def test_two_text_only_items_do_not_form_a_row_without_an_icon():
    """A bare two text pair is weak evidence; a row needs an icon or 3+ items."""
    candidates = [
        _text_node("p1", "$49", {"x": 100, "y": 300, "w": 60, "h": 40}, "price"),
        _text_node("p2", "$99", {"x": 180, "y": 300, "w": 60, "h": 40}, "price"),
    ]
    tree = layout.infer(candidates, {"w": 600, "h": 600}, {})
    assert not any(n.get("meta", {}).get("role") == "text-row" for n in _all_groups(tree))


# --------------------------------------------------------------------------- #
# Padded card single-child HUG (Fix C)
# --------------------------------------------------------------------------- #

def test_padded_card_single_text_hugs_with_measured_padding():
    """A surfaced plate wrapping one substantial inset text becomes a HUG frame whose
    four-side padding reconstructs the plate box exactly (resize-safe)."""
    candidates = [
        {"id": "plate", "target": "shape", "box": {"x": 40, "y": 60, "w": 320, "h": 140},
         "fill": {"kind": "flat", "color": "#101010"}, "style": {"radius": 16},
         "meta": {"role": "card"}},
        _text_node("copy", "Limited time only", {"x": 72, "y": 84, "w": 256, "h": 92}, "body"),
    ]
    tree = layout.infer(candidates, {"w": 600, "h": 400}, {})
    assert len(tree) == 1
    frame = tree[0]
    assert frame["target"] == "group"
    assert frame["layout"]["mode"] in ("HORIZONTAL", "VERTICAL")
    assert frame["layout"]["primarySizing"] == "HUG"
    assert frame["layout"]["counterSizing"] == "HUG"
    # padding measured from geometry: left 72-40=32, right 360-328=32,
    # top 84-60=24, bottom 200-176=24 -> hugging reproduces the 320x140 plate.
    assert frame["layout"]["padding"] == {"left": 32, "right": 32, "top": 24, "bottom": 24}
    pad = frame["layout"]["padding"]
    child = frame["children"][0]
    assert child["box"]["w"] + pad["left"] + pad["right"] == 320
    assert child["box"]["h"] + pad["top"] + pad["bottom"] == 140


def test_tiny_label_on_a_large_plate_does_not_hug():
    """A speck of text on a big backdrop must stay absolute — HUG would collapse the
    plate to the label and destroy the layout."""
    candidates = [
        {"id": "bg", "target": "shape", "box": {"x": 0, "y": 0, "w": 400, "h": 400},
         "fill": {"kind": "flat", "color": "#ffffff"}, "meta": {"role": "card"}},
        _text_node("tag", "new", {"x": 20, "y": 20, "w": 40, "h": 18}, "label"),
    ]
    tree = layout.infer(candidates, {"w": 800, "h": 800}, {})
    frame = tree[0]
    assert frame["layout"]["mode"] == "NONE"


# --------------------------------------------------------------------------- #
# IG Caption pills (Figma 15510: coral plate + two centered caption-paragraphs)
# --------------------------------------------------------------------------- #

_FIGMA_15510_CFG = {
    "layout": {"scene_grouping": {"pair_text_with_backplate": True}},
}


def _figma_15510_caption_candidates(*, shell_role="card"):
    """Exact local geometry from work/_figma_15510.json (frame origin subtracted)."""
    return [
        {
            "id": "shell0",
            "target": "shape",
            "box": {"x": 19.0, "y": 672.0, "w": 1041.6, "h": 116.4},
            "fill": {"kind": "flat", "color": "#ffffff"},
            "radius": 58.2,
            "meta": {"role": shell_role},
        },
        {
            "id": "text0",
            "target": "text",
            "text": "Didn't know I needed this",
            "box": {"x": 69.2, "y": 684.0, "w": 942.0, "h": 93.0},
            "style": {
                "fontFamily": "Inter",
                "fontSize": 76.8,
                "fontWeight": 700,
                "letterSpacing": 0.0,
                "align": "CENTER",
            },
            "meta": {"role": "headline"},
        },
        {
            "id": "shell1",
            "target": "shape",
            "box": {"x": 208.6, "y": 843.6, "w": 662.4, "h": 116.4},
            "fill": {"kind": "flat", "color": "#000000"},
            "radius": 58.2,
            "meta": {"role": shell_role},
        },
        {
            "id": "text1",
            "target": "text",
            "text": "...Until I ran out",
            "box": {"x": 256.8, "y": 855.6, "w": 567.0, "h": 93.0},
            "style": {
                "fontFamily": "Inter",
                "fontSize": 76.8,
                "fontWeight": 700,
                "letterSpacing": 0.0,
                "align": "CENTER",
            },
            "meta": {"role": "body"},
        },
    ]


def test_centered_caption_plate_hugs_with_center_align():
    """Centered text on a pill plate must HUG with CENTER/CENTER (not padded-card MIN)."""
    candidates = [
        {
            "id": "plate",
            "target": "shape",
            "box": {"x": 19.0, "y": 672.0, "w": 1041.6, "h": 116.4},
            "fill": {"kind": "flat", "color": "#ffffff"},
            "meta": {"role": "card"},
        },
        {
            "id": "copy",
            "target": "text",
            "text": "Didn't know I needed this",
            "box": {"x": 69.2, "y": 684.0, "w": 942.0, "h": 93.0},
            "style": {"align": "CENTER", "fontFamily": "Inter", "fontWeight": 700,
                      "fontSize": 76.8, "letterSpacing": 0.0},
            "meta": {"role": "headline"},
        },
    ]
    tree = layout.infer(candidates, {"w": 1080, "h": 1920}, _FIGMA_15510_CFG)
    assert len(tree) == 1
    frame = tree[0]
    assert frame["layout"]["mode"] in ("HORIZONTAL", "VERTICAL")
    assert frame["layout"]["align"] == "CENTER"
    assert frame["layout"]["counterAlign"] == "CENTER"
    assert frame["layout"]["primarySizing"] == "HUG"
    assert frame["layout"]["counterSizing"] == "HUG"
    pad = frame["layout"]["padding"]
    assert abs(pad["left"] - 50.2) < 0.5
    assert abs(pad["right"] - 49.4) < 0.5
    assert abs(pad["top"] - 12.0) < 0.5
    assert abs(pad["bottom"] - 11.4) < 0.5


def test_figma_15510_caption_pills_stack_as_ig_caption():
    """Two sibling caption-paragraph plates become one VERTICAL IG Caption frame.

    Ground truth (Figma 15510): layoutMode=VERTICAL, itemSpacing≈55.2,
    counterAxisAlignItems=CENTER, HUG×HUG; second pill narrower and centered.
    """
    tree = layout.infer(
        _figma_15510_caption_candidates(shell_role="card"),
        {"w": 1080, "h": 1920},
        _FIGMA_15510_CFG,
    )
    assert len(tree) == 1
    stack = tree[0]
    assert stack["meta"]["role"] == "caption-stack"
    assert stack["layout"]["mode"] == "VERTICAL"
    assert abs(stack["layout"]["gap"] - 55.2) < 0.5
    assert stack["layout"]["itemSpacing"] == stack["layout"]["gap"]
    assert stack["layout"]["counterAlign"] == "CENTER"
    assert stack["layout"]["primarySizing"] == "HUG"
    assert stack["layout"]["counterSizing"] == "HUG"
    assert len(stack["children"]) == 2
    for pill in stack["children"]:
        assert pill["layout"]["align"] == "CENTER"
        assert pill["layout"]["counterAlign"] == "CENTER"
        assert pill["layout"]["primarySizing"] == "HUG"
    # Narrower second pill stays centered under the first (relative x > 0).
    wide, narrow = stack["children"]
    assert narrow["box"]["w"] < wide["box"]["w"]
    assert narrow["box"]["x"] > 0


def test_figma_15510_shape_shells_pair_when_backplate_flag_set():
    """SAM often labels pill chrome as role=shape; pair_text_with_backplate must still fire."""
    tree = layout.infer(
        _figma_15510_caption_candidates(shell_role="shape"),
        {"w": 1080, "h": 1920},
        _FIGMA_15510_CFG,
    )
    assert len(tree) == 1
    stack = tree[0]
    assert stack["meta"]["role"] == "caption-stack"
    assert stack["layout"]["counterAlign"] == "CENTER"
    assert abs(stack["layout"]["gap"] - 55.2) < 0.5


def test_left_aligned_padded_card_keeps_min_align():
    """A left-biased label on a plate must not be forced into CENTER caption treatment."""
    candidates = [
        {"id": "plate", "target": "shape", "box": {"x": 40, "y": 60, "w": 320, "h": 140},
         "fill": {"kind": "flat", "color": "#101010"}, "meta": {"role": "card"}},
        _text_node("copy", "Limited time only", {"x": 56, "y": 84, "w": 256, "h": 92}, "body"),
    ]
    candidates[1]["style"] = {"align": "LEFT"}
    tree = layout.infer(candidates, {"w": 600, "h": 400}, _FIGMA_15510_CFG)
    frame = tree[0]
    assert frame["layout"]["primarySizing"] == "HUG"
    assert frame["layout"]["align"] == "MIN"
    assert frame["layout"]["counterAlign"] == "MIN"


def test_left_column_stat_pills_pair_and_stack():
    """Hears-style left-column stats: each plate+text becomes a stat-pill; siblings stack."""
    candidates = [
        {"id": "p1", "target": "shape", "box": {"x": 40, "y": 200, "w": 220, "h": 72},
         "fill": {"kind": "flat", "color": "#ffffff40"}, "radius": 16,
         "meta": {"role": "shape"}},
        {"id": "t1", "target": "text", "text": "200%",
         "box": {"x": 56, "y": 216, "w": 160, "h": 40},
         "style": {"align": "LEFT", "fontSize": 28}, "meta": {"role": "offer"}},
        {"id": "p2", "target": "shape", "box": {"x": 40, "y": 292, "w": 220, "h": 72},
         "fill": {"kind": "flat", "color": "#ffffff40"}, "radius": 16,
         "meta": {"role": "shape"}},
        {"id": "t2", "target": "text", "text": "Noise reduction",
         "box": {"x": 56, "y": 308, "w": 180, "h": 40},
         "style": {"align": "LEFT", "fontSize": 22}, "meta": {"role": "body"}},
    ]
    # offer is in _BUTTON_TEXT_ROLES — use label/body for stats
    candidates[1]["meta"]["role"] = "label"
    tree = layout.infer(candidates, {"w": 1080, "h": 1350}, _FIGMA_15510_CFG)
    assert len(tree) == 1
    stack = tree[0]
    assert stack["meta"]["role"] == "stat-stack"
    assert stack["layout"]["mode"] == "VERTICAL"
    assert stack["layout"]["counterAlign"] == "MIN"
    assert len(stack["children"]) == 2
    for pill in stack["children"]:
        assert pill["meta"]["role"] == "stat-pill"
        assert pill["layout"]["align"] == "MIN"
        assert pill["layout"]["primarySizing"] == "HUG"
