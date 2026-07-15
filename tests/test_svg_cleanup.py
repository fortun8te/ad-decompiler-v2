import numpy as np
import pytest
from PIL import Image

from src import svg_cleanup, vectorize


def test_parse_and_serialize_roundtrip():
    d = "M1.00 2.00L11.00 2.00C11.00 8.00 9.00 10.00 1.00 10.00Z"
    subs = svg_cleanup.parse_d(d)
    assert len(subs) == 1
    assert subs[0]["start"] == (1.0, 2.0)
    assert subs[0]["closed"] is True
    assert svg_cleanup.serialize_subpaths(subs) == d


def test_parse_rejects_relative_and_arc_commands():
    with pytest.raises(ValueError):
        svg_cleanup.parse_d("m1 1 l2 2")
    with pytest.raises(ValueError):
        svg_cleanup.parse_d("M0 0A5 5 0 0 1 10 10Z")


def test_rdp_removes_collinear_points():
    # 10-point staircase-free straight edge collapses to its 2 endpoints.
    pts = "".join(f"L{i}.00 0.00" for i in range(1, 10))
    d = f"M0.00 0.00{pts}L9.00 5.00L0.00 5.00Z"
    out = svg_cleanup.cleanup_paths(
        [{"d": d, "fill": "#112233"}], min_area=1.0, tolerance=0.5,
    )
    assert len(out) == 1
    cleaned = svg_cleanup.parse_d(out[0]["d"])[0]
    # M + L9,0 + L9,5 + L0,5 (Z closes back to the start)
    assert len(cleaned["segs"]) == 3
    assert cleaned["segs"][0] == ("L", 9.0, 0.0)


def test_straight_cubic_demoted_to_line():
    d = "M0.00 0.00C3.00 0.10 6.00 0.10 10.00 0.00L10.00 6.00L0.00 6.00Z"
    out = svg_cleanup.cleanup_paths([{"d": d, "fill": "#000000"}],
                                    min_area=1.0, tolerance=0.5)
    cleaned = svg_cleanup.parse_d(out[0]["d"])[0]
    assert all(seg[0] == "L" for seg in cleaned["segs"])
    assert svg_cleanup.count_points(out) < svg_cleanup.count_points([{"d": d}])


def test_curved_cubic_is_preserved():
    d = "M0.00 10.00C0.00 4.00 4.00 0.00 10.00 0.00L10.00 10.00Z"
    out = svg_cleanup.cleanup_paths([{"d": d, "fill": "#000000"}],
                                    min_area=1.0, tolerance=0.5)
    cleaned = svg_cleanup.parse_d(out[0]["d"])[0]
    assert cleaned["segs"][0][0] == "C"


def _banded_paths():
    """Six near-identical blue bands tiling one rect + two sub-pixel speckles."""
    paths = []
    for i in range(6):
        x0 = 5 + i * 50 // 6
        x1 = 5 + (i + 1) * 50 // 6
        fill = "#%02x%02x%02x" % (51 + i, 102 + i, 204 + (i % 3))
        paths.append({"d": f"M{x0} 8L{x1} 8L{x1} 32L{x0} 32Z", "fill": fill})
    paths.append({"d": "M1 1L2 1L2 2L1 2Z", "fill": "#336699"})
    paths.append({"d": "M58 38L59 38L59 39L58 39Z", "fill": "#336688"})
    return paths


def test_banded_trace_collapses_to_one_path_and_still_passes_render_gate(tmp_path):
    source = tmp_path / "band.png"
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[8:32, 5:55] = (51, 102, 204, 255)
    Image.fromarray(rgba).save(source)

    paths = _banded_paths()
    before_points = svg_cleanup.count_points(paths)
    cleaned = svg_cleanup.cleanup_paths(paths, min_area=2.0, tolerance=0.6,
                                        fill_tolerance=10, merge=True)

    assert len(cleaned) == 1  # 6 bands merged, 2 speckles dropped
    assert svg_cleanup.count_points(cleaned) < before_points
    score = vectorize._score_render(svg_cleanup.serialize_svg(cleaned, 60, 40), str(source))
    assert score >= 0.85  # cleaned output still clears the default render-back gate


def test_fills_outside_tolerance_are_not_merged():
    paths = [
        {"d": "M0 0L10 0L10 10L0 10Z", "fill": "#336699"},
        {"d": "M10 0L20 0L20 10L10 10Z", "fill": "#993333"},
    ]
    cleaned = svg_cleanup.cleanup_paths(paths, min_area=1.0, fill_tolerance=10)
    assert len(cleaned) == 2
    assert {p["fill"] for p in cleaned} == {"#336699", "#993333"}


def test_evenodd_and_nonzero_neighbours_are_not_merged():
    paths = [
        {"d": "M0 0L10 0L10 10L0 10Z", "fill": "#336699", "windingRule": "EVENODD"},
        {"d": "M10 0L20 0L20 10L10 10Z", "fill": "#336699"},
    ]
    cleaned = svg_cleanup.cleanup_paths(paths, min_area=1.0)
    assert len(cleaned) == 2
    assert cleaned[0]["windingRule"] == "EVENODD"


def test_unparsable_and_stroked_paths_pass_through_untouched():
    arc = {"d": "M0 0A5 5 0 0 1 10 10Z", "fill": "#123456"}
    stroked = {"d": "M0 0L10 10", "fill": "none",
               "stroke": {"color": "#654321", "width": 2.0}}
    big = {"d": "M0 0L30 0L30 30L0 30Z", "fill": "#123456"}
    cleaned = svg_cleanup.cleanup_paths([arc, stroked, big], min_area=2.0)
    assert cleaned[0] == arc
    assert cleaned[1] == stroked
    svg = svg_cleanup.serialize_svg(cleaned, 30, 30)
    assert 'stroke="#654321"' in svg and 'stroke-width="2.0"' in svg


def test_all_noise_input_is_returned_unchanged():
    tiny = [{"d": "M0 0L1 0L1 1L0 1Z", "fill": "#000000"}]
    assert svg_cleanup.cleanup_paths(tiny, min_area=5.0) == tiny


def test_tiny_hole_subpath_is_dropped_but_real_counter_kept():
    d = ("M0 0L20 0L20 20L0 20Z"          # 400 px outer
         "M5 5L15 5L15 15L5 15Z"          # 100 px counter: keep
         "M1 1L2 1L2 2L1 2Z")             # 1 px speck: drop
    out = svg_cleanup.cleanup_paths([{"d": d, "fill": "#000000"}],
                                    min_area=2.0, tolerance=0.0)
    subs = svg_cleanup.parse_d(out[0]["d"])
    assert len(subs) == 2
