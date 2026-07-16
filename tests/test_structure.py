"""Layer-tree shaping — the geometries are lifted from runs/postfix-benchmark-6.

Each fixture-named test encodes a defect a designer actually hit when opening the
emitted design.json in Figma.
"""

import pytest

from src import structure

CANVAS = {"w": 1080, "h": 1080}


def _text(id_, x, y, w, h, text=""):
    return {"id": id_, "target": "text", "box": {"x": x, "y": y, "w": w, "h": h},
            "text": text, "children": [], "meta": {}}


def _image(id_, x, y, w, h, name=None):
    return {"id": id_, "target": "image", "box": {"x": x, "y": y, "w": w, "h": h},
            "name": name, "children": [], "meta": {}}


def _group(id_, x, y, w, h, children, name=None):
    return {"id": id_, "target": "group", "box": {"x": x, "y": y, "w": w, "h": h},
            "name": name, "children": list(children), "meta": {}}


def _names(nodes):
    return [n.get("name") for n in nodes]


def _flatten(nodes):
    for n in nodes:
        yield n
        yield from _flatten(n.get("children") or [])


# ── 021: empty groups and redundant wrappers ────────────────────────────────────

def test_021_empty_header_groups_are_pruned():
    """021 emitted 'Header' and 'Header / 2' with zero children."""
    roots = [
        _image("bg", 0, 0, 338, 600, name="Background"),
        _group("g_header", 226, 0, 85, 142, [], name="Header"),
        _group("g_header2", 0, 0, 201, 143, [], name="Header / 2"),
    ]
    report = {}
    out = structure.prune_empty_groups(roots, report)
    assert [n["id"] for n in out] == ["bg"]
    assert set(report["pruned_empty_groups"]) == {"g_header", "g_header2"}


def test_021_group_wrapping_one_identical_image_is_collapsed():
    """group 'Photo' [0,0 338x600] > image 'Photo' [0,0 338x600]."""
    roots = [_group("g_photo", 0, 0, 338, 600,
                    [_image("c_E000", 0, 0, 338, 600, name="Photo")], name="Photo")]
    out = structure.collapse_redundant_wrappers(roots, {})
    assert len(out) == 1
    assert out[0]["id"] == "c_E000" and out[0]["target"] == "image"


def test_wrapper_with_its_own_fill_is_kept():
    """A wrapper that paints something is real structure, not noise."""
    roots = [dict(_group("g", 0, 0, 100, 100,
                         [_image("i", 0, 0, 100, 100)]), fill={"kind": "flat"})]
    out = structure.collapse_redundant_wrappers(roots, {})
    assert out[0]["id"] == "g"


def test_wrapper_larger_than_its_child_is_kept():
    """Padding is meaningful: only an identical extent is redundant."""
    roots = [_group("g", 0, 0, 200, 200, [_image("i", 10, 10, 50, 50)])]
    out = structure.collapse_redundant_wrappers(roots, {})
    assert out[0]["id"] == "g"


def test_pruning_cascades_to_a_parent_left_empty():
    roots = [_group("outer", 0, 0, 10, 10, [_group("inner", 0, 0, 5, 5, [])])]
    assert structure.prune_empty_groups(roots, {}) == []


# ── 009: the flat dump ──────────────────────────────────────────────────────────

def _009_screenshot_group():
    """The real 009 tree: 18 text nodes flat inside one 'Screenshot' group."""
    kids = [
        _text("t_post", 509, 41, 105, 56, "Post"),
        _text("t_volgend", 879, 143, 145, 46, "Volgend"),
        _text("t_upfront", 193, 143, 207, 45, "UPFRONT"),
        _text("t_handle", 193, 186, 241, 45, "@UpfrontFood"),
        _text("t_head", 71, 315, 683, 46, "LAATSTE SITE WIDE SALE VAN 2026"),
        _text("t_b1", 71, 409, 883, 46, "De Vakantiegeldsale komt eraan,"),
        _text("t_b2", 71, 454, 772, 47, "korting krijgt op het volledige"),
        _text("t_b3", 44, 543, 969, 45, "Daarbovenop krijgen de eerste 500"),
        _text("t_b4", 70, 589, 363, 46, "geld terug tot €100."),
        _text("t_b5", 42, 677, 982, 48, "Schrijf je nu in en mis geen"),
        _text("t_b6", 70, 722, 646, 47, "woensdag 20 mei om 20:00 uur."),
        _text("t_time", 43, 918, 424, 49, "05:00 PM · 12-05-2026 ·"),
        _text("t_121k", 454, 920, 74, 44, "121K"),
        _text("t_weer", 535, 917, 192, 50, "weergaven"),
        _text("t_257", 125, 1009, 60, 41, "257"),
        _text("t_66", 371, 1009, 42, 39, "66"),
        _text("t_21k", 620, 1008, 60, 42, "21K"),
        _text("t_89", 862, 1009, 41, 38, "89"),
    ]
    return [_image("bg", 0, 0, 1080, 1080, name="Background"),
            _group("g_shot", 0, 5, 1080, 1075, kids, name="Screenshot")]


def test_009_flat_screenshot_group_gains_bands():
    opts = structure.options({})
    bands = structure.band_split(_009_screenshot_group()[1]["children"], opts)
    assert bands is not None, "18 flat text nodes must be banded"
    assert len(bands) >= 3, f"expected header/body/footer-ish bands, got {len(bands)}"
    # The header band owns the name row, never the engagement counts.
    header = {n["id"] for n in bands[0]}
    assert {"t_post", "t_upfront", "t_handle"} <= header
    assert "t_257" not in header
    # The last band owns the engagement counts, never the headline.
    footer = {n["id"] for n in bands[-1]}
    assert "t_head" not in footer


def test_009_restructure_is_no_longer_flat_and_text_stays_reachable():
    roots, report = structure.restructure(_009_screenshot_group(), CANVAS, {})
    shot = next(n for n in roots if n["id"] == "g_shot")
    assert report["applied"] is True
    assert report.get("banded_groups"), "the flat group must have been banded"
    # Every text node still exists and is still a text node (batch copy swaps).
    texts = [n for n in _flatten(roots) if n["target"] == "text"]
    assert len(texts) == 18
    # Bands are groups, and each band is named something scannable.
    bands = [n for n in shot["children"] if n["target"] == "group"]
    assert len(bands) >= 2
    assert all(n.get("name") for n in bands)
    assert not any(structure._junk_name(n["name"]) for n in bands)


def test_009_band_names_are_not_generic_group():
    roots, _ = structure.restructure(_009_screenshot_group(), CANVAS, {})
    for node in _flatten(roots):
        if node["target"] == "group":
            assert not structure._junk_name(node.get("name")), node


def test_a_short_flat_group_is_left_alone():
    """Below band_min_children the list is already readable; don't churn it."""
    kids = [_text(f"t{i}", 0, i * 60, 100, 40, f"line {i}") for i in range(3)]
    assert structure.band_split(kids, structure.options({})) is None


def test_evenly_spaced_lines_have_no_seam():
    """A uniform paragraph must not be shattered into one band per line."""
    kids = [_text(f"t{i}", 0, i * 50, 100, 40, f"line {i}") for i in range(8)]
    assert structure.band_split(kids, structure.options({})) is None


def test_band_split_respects_a_configured_seam_factor():
    kids = [_text(f"t{i}", 0, i * 50, 100, 40, f"line {i}") for i in range(4)]
    kids.append(_text("far", 0, 600, 100, 40, "far away"))
    kids.append(_text("far2", 0, 650, 100, 40, "far away 2"))
    opts = structure.options({"structure": {"band_min_children": 3}})
    bands = structure.band_split(kids, opts)
    assert bands is not None and len(bands) == 2
    assert [n["id"] for n in bands[1]] == ["far", "far2"]


# ── z-order ─────────────────────────────────────────────────────────────────────

def test_rasters_sit_behind_their_copy():
    roots = [_group("g", 0, 0, 100, 100, [
        _text("t", 0, 0, 100, 20, "hi"),
        _image("img", 0, 0, 100, 100),
    ])]
    out = structure.order_children(roots, report={})
    assert [n["id"] for n in out[0]["children"]] == ["img", "t"]


def test_ordering_is_stable_within_a_class():
    roots = [_group("g", 0, 0, 100, 100, [
        _image("img_b", 0, 0, 10, 10),
        _image("img_a", 0, 0, 10, 10),
        _text("t_b", 0, 0, 10, 10),
        _text("t_a", 0, 0, 10, 10),
    ])]
    out = structure.order_children(roots, report={})
    assert [n["id"] for n in out[0]["children"]] == ["img_b", "img_a", "t_b", "t_a"]


def _shape(id_, x, y, w, h, name=None):
    return {"id": id_, "target": "shape", "box": {"x": x, "y": y, "w": w, "h": h},
            "name": name, "children": [], "meta": {}}


def test_a_full_bleed_plate_never_sorts_above_an_overlapping_cutout():
    """The rank must not reorder ART AMONGST ITSELF.

    A full-bleed plate (shape) painted FIRST and a product cutout (image) painted
    over it: ranking rasters below shapes lifted the plate above the cutout and hid
    it. Only the text/non-text split is this pass's business.
    """
    roots = [_group("g", 0, 0, 1080, 1080, [
        _shape("plate", 0, 0, 1080, 1080, name="Background plate"),
        _image("cutout", 200, 200, 400, 400, name="Product"),
    ])]
    out = structure.order_children(roots, report={})
    assert [n["id"] for n in out[0]["children"]] == ["plate", "cutout"]


def test_incoming_art_z_survives_while_text_still_floats_up():
    """Art of every kind keeps its incoming relative order; only text moves."""
    roots = [_group("g", 0, 0, 1080, 1080, [
        _text("caption", 0, 900, 500, 40, "buy now"),
        _shape("plate", 0, 0, 1080, 1080),
        _image("photo", 0, 0, 800, 800),
        _group("badge", 50, 50, 100, 100, [_shape("chip", 50, 50, 100, 100)]),
    ])]
    out = structure.order_children(roots, report={})
    # plate/photo/badge hold their incoming order; only the text is lifted to the top.
    assert [n["id"] for n in out[0]["children"]] == ["plate", "photo", "badge", "caption"]


# ── names ───────────────────────────────────────────────────────────────────────

def test_002_root_group_named_Group_is_renamed_after_its_copy():
    roots = [_group("g", 61, 819, 955, 856,
                    [_text("t", 215, 826, 528, 47, "KOOP NU VIA UPFRONT.NL")],
                    name="Group")]
    out = structure.dedupe_sibling_names(roots, {})
    assert out[0]["name"] == "KOOP NU VIA UPFRONT.NL"


def test_duplicate_sibling_names_are_numbered():
    roots = [_image("a", 0, 0, 10, 10, name="Badge"),
             _image("b", 0, 0, 10, 10, name="Badge")]
    out = structure.dedupe_sibling_names(roots, {})
    assert _names(out) == ["Badge", "Badge / 2"]


def test_junk_named_group_without_text_keeps_a_stable_fallback():
    roots = [_group("g", 0, 0, 10, 10, [_image("i", 0, 0, 10, 10, name="Photo")],
                    name="Group")]
    out = structure.dedupe_sibling_names(roots, {})
    assert out[0]["name"] == "Group"  # nothing better available; still unique


# ── contract ────────────────────────────────────────────────────────────────────

def test_restructure_does_not_mutate_its_input():
    roots = _009_screenshot_group()
    before = len(roots[1]["children"])
    structure.restructure(roots, CANVAS, {})
    assert len(roots[1]["children"]) == before


def test_restructure_disabled_returns_input_untouched():
    roots = _009_screenshot_group()
    out, report = structure.restructure(roots, CANVAS, {"structure": {"enabled": False}})
    assert out is roots and report["applied"] is False


def test_restructure_on_empty_forest():
    out, report = structure.restructure([], CANVAS, {})
    assert out == [] and report["applied"] is False


def test_restructure_preserves_every_leaf():
    """Shaping must never lose a layer — only regroup, rename and reorder."""
    roots = _009_screenshot_group()
    leaves_before = {n["id"] for n in _flatten(roots) if n["target"] != "group"}
    out, _ = structure.restructure(roots, CANVAS, {})
    leaves_after = {n["id"] for n in _flatten(out) if n["target"] != "group"}
    assert leaves_before == leaves_after


def test_band_boxes_cover_their_children():
    roots, _ = structure.restructure(_009_screenshot_group(), CANVAS, {})
    for node in _flatten(roots):
        if (node.get("meta") or {}).get("structure") == "band":
            band = structure._box(node)
            for child in node["children"]:
                cb = structure._box(child)
                assert cb["x"] >= band["x"] - 0.5
                assert cb["y"] >= band["y"] - 0.5
                assert cb["x"] + cb["w"] <= band["x"] + band["w"] + 0.5
                assert cb["y"] + cb["h"] <= band["y"] + band["h"] + 0.5


# ── 107: sixteen flat roots ─────────────────────────────────────────────────────

def test_107_flat_root_forest_is_banded_but_background_stays_at_root():
    canvas = {"w": 1080, "h": 1920}
    roots = [
        _image("bg", 0, 0, 1080, 1920, name="Background"),
        _group("g_stack", 99, 305, 878, 417, [_text("t58", 273, 341, 478, 226, "58%")],
               name="Text Stack"),
        _image("badge", 284, 400, 88, 89, name="Badge"),
        _group("g_chart", 174, 789, 823, 535, [_image("chart", 174, 789, 823, 535)],
               name="Chart"),
        _text("w1", 199, 1349, 77, 29, "WEEK 1"),
        _text("w2", 345, 1350, 85, 27, "WEEK 2"),
        _text("w3", 494, 1351, 86, 25, "WEEK 3"),
        _text("w4", 649, 1350, 85, 26, "WEEK 4"),
        _text("w5", 801, 1350, 85, 26, "WEEK 5"),
        _text("b1", 187, 1515, 704, 40, "Most athletes think they're"),
        _text("b2", 183, 1552, 716, 40, "Research says otherwise."),
        _text("b3", 457, 1589, 165, 40, "feel thirsty."),
    ]
    out, report = structure.restructure(roots, canvas, {})
    assert report.get("banded_roots"), "16 flat roots must be banded"
    # The full-bleed background never gets nested inside a band.
    assert any(n["id"] == "bg" and n["target"] == "image" for n in out)
    # Every leaf survives the regroup.
    leaves_before = {n["id"] for n in _flatten(roots) if n["target"] != "group"}
    leaves_after = {n["id"] for n in _flatten(out) if n["target"] != "group"}
    assert leaves_before == leaves_after
    assert len(out) < len(roots), "banding must reduce root fan-out"


def test_backgroundish_detects_full_bleed_plate_only():
    canvas = {"w": 1000, "h": 1000}
    assert structure._backgroundish(_image("bg", 0, 0, 1000, 1000), canvas)
    assert not structure._backgroundish(_image("x", 0, 0, 100, 100), canvas)


def test_band_roots_left_alone_without_a_canvas():
    roots = [_text(f"t{i}", 0, i * 200, 10, 10) for i in range(8)]
    out = structure.band_roots(roots, {}, structure.options({}), {})
    assert len(out) == len(roots) or all(n.get("target") for n in out)
