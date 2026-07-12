import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src import vectorize


def test_gate_limits_role_based_defaults():
    score, paths = vectorize._gate_limits("badge", {})
    assert score == 0.80 and paths == 35
    score, paths = vectorize._gate_limits("logo", {})
    assert score == 0.82 and paths == 50
    score, paths = vectorize._gate_limits(None, {})
    assert score == 0.85 and paths == 40


def test_gate_limits_respects_config_override():
    cfg = {"vectorize": {"score_min": {"badge": 0.75}, "max_paths": {"badge": 80}}}
    score, paths = vectorize._gate_limits("badge", cfg)
    assert score == 0.75 and paths == 80


def test_preprocess_upscales_tiny_icons(tmp_path):
    tiny = tmp_path / "tiny.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(tiny)
    out, cleanup = vectorize._preprocess_crop(str(tiny), {}, role="icon")
    try:
        with Image.open(out) as im:
            assert min(im.size) >= 48
    finally:
        if cleanup:
            os.unlink(out)


def test_check_binaries_reports_paths(monkeypatch):
    monkeypatch.setattr(vectorize.shutil, "which", lambda name: f"/bin/{name}")
    status = vectorize.check_binaries({})
    assert status["vtracer"]["ok"] is True
    assert status["potrace"]["ok"] is True
    assert "path" in status["vtracer"]


def test_evaluate_trace_applies_role_gate(monkeypatch):
    svg = '<svg><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000"/></svg>'
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.81)
    result, _ = vectorize._evaluate_trace(svg, "x.png", "vtracer", {}, "badge", 3)
    assert result["ok"] is True
    assert result["gate"]["score_min"] == 0.80


def test_vtracer_tries_multiple_presets(tmp_path, monkeypatch):
    src = tmp_path / "icon.png"
    Image.new("RGBA", (32, 32), (0, 128, 255, 255)).save(src)
    calls = []

    def fake_vtracer(path, cfg, preset=None):
        calls.append(dict(preset or {}))
        if len(calls) < 2:
            return None, "fail"
        svg = '<svg><path d="M0 0 L32 0 L32 32 Z" fill="#0080ff"/></svg>'
        return svg, None

    monkeypatch.setattr(vectorize.shutil, "which", lambda _: "vtracer")
    monkeypatch.setattr(vectorize, "_run_vtracer", fake_vtracer)
    monkeypatch.setattr(vectorize, "_run_potrace", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_run_contour_simplify", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.95)

    result = vectorize.vectorize_crop(str(src), {"vectorize": {"vtracer_presets": [
        {"mode": "spline", "colormode": "color"},
        {"mode": "polygon", "colormode": "binary", "filter_speckle": 2},
    ]}}, role="icon")
    assert result["ok"] is True
    assert result["engine"] == "vtracer"
    assert len(calls) == 2


def test_potrace_threshold_chain_for_monochrome(tmp_path, monkeypatch):
    src = tmp_path / "mono.png"
    Image.new("RGBA", (24, 24), (255, 255, 255, 0)).save(src)
    thresholds = []

    def fake_potrace(path, cfg, alpha_threshold=8, lum_threshold=128):
        thresholds.append(alpha_threshold)
        if alpha_threshold < 16:
            return None, "empty"
        svg = (
            '<svg><g transform="translate(0,24) scale(0.1,-0.1)">'
            '<path d="M0 0L100 0L100 100Z" fill="#000"/></g></svg>'
        )
        return svg, None

    monkeypatch.setattr(vectorize.shutil, "which", lambda _: "potrace")
    monkeypatch.setattr(vectorize, "_run_vtracer", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_run_potrace", fake_potrace)
    monkeypatch.setattr(vectorize, "_run_contour_simplify", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.92)
    monkeypatch.setattr(vectorize, "_count_colors", lambda _p: 1)

    result = vectorize.vectorize_crop(str(src), {"vectorize": {"potrace_thresholds": [8, 16, 32]}},
                                      role="arrow")
    assert result["ok"] is True
    assert result["engine"] == "potrace"
    assert thresholds == [8, 16]


def test_contour_fallback_builds_paths(tmp_path, monkeypatch):
    src = tmp_path / "flat.png"
    icon = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
    ImageDraw.Draw(icon).rectangle((4, 4, 16, 16), fill=(10, 20, 30, 255))
    icon.save(src)

    monkeypatch.setattr(vectorize, "_run_vtracer", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_run_potrace", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.88)
    monkeypatch.setattr(vectorize, "_count_colors", lambda _p: 2)

    result = vectorize.vectorize_crop(str(src), {}, role="badge")
    if result.get("engine") == "contour":
        assert result["ok"] is True
        assert result["paths"]
    else:
        assert result.get("note")


def test_parse_svg_paths_applies_enclosing_g_transform_to_path_coordinates():
    # Mirrors potrace's `-s` backend: it wraps every <path> in a <g transform="translate(..)
    # scale(..)"> that rescales its internal (10x, Y-flipped) trace units into real pixel
    # space. Consumers read paths[]["d"] directly (src/reconstruct.py, the single-path
    # fallback in src/build_design_json.py, the raw-path mask fallback in
    # src/render_preview.py) -- if the transform isn't baked into `d`, every potrace-traced
    # icon comes out ~10x mis-scaled and vertically mirrored.
    svg = (
        '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" width="64pt" height="64pt" '
        'viewBox="0 0 64 64">'
        '<g transform="translate(0.000000,64.000000) scale(0.100000,-0.100000)" '
        'fill="#000000" stroke="none">'
        '<path d="M100 100L100 500L500 500L500 100Z" fill="#112233"/>'
        '</g></svg>'
    )
    paths = vectorize._parse_svg_paths(svg)
    assert len(paths) == 1
    d = paths[0]["d"]

    # Expected: each raw (x, y) is scaled by (0.1, -0.1) then translated by (0, 64) —
    # e.g. raw (100, 100) -> (10, 64 - 10) = (10, 54).
    def expect(x, y):
        return f"{x * 0.1:.2f} {64.0 - y * 0.1:.2f}"

    assert f"M{expect(100, 100)}" in d
    assert f"L{expect(100, 500)}" in d
    assert f"L{expect(500, 500)}" in d
    assert f"L{expect(500, 100)}" in d

    # The raw (untransformed) coordinates must NOT leak through -- that's exactly the ~10x
    # mis-scaled, unflipped bug this test guards against.
    assert "M100.00 100.00" not in d
    assert "L500.00 500.00" not in d


def test_parse_svg_paths_nested_groups_compose_transforms_and_pop_on_close():
    svg = (
        '<svg><g transform="translate(10,20)">'
        '<g transform="scale(2)"><path d="M1 1L2 2Z" fill="#fff"/></g>'
        '<path d="M0 0Z" fill="#fff"/>'
        '</g></svg>'
    )
    paths = vectorize._parse_svg_paths(svg)
    assert len(paths) == 2
    # Inner path: scale(2) then translate(10,20) -> (1,1)->(12,22), (2,2)->(14,24).
    assert "M12.00 22.00" in paths[0]["d"]
    assert "L14.00 24.00" in paths[0]["d"]
    # Outer path (after the inner </g> pops scale off the stack): only translate(10,20).
    assert "M10.00 20.00" in paths[1]["d"]


def test_abs_path_arc_advances_current_point_for_subsequent_relative_segments():
    # An unsupported "A" (arc) command must still move (cx, cy) to its endpoint, otherwise a
    # following relative command is anchored to the wrong origin.
    d = "M0 0A5 5 0 0 1 10 10l1 1"
    out = vectorize._abs_path(d)
    assert out.endswith("L11.00 11.00")


def test_potrace_leaks_no_temp_pbm_when_bitmap_save_fails(tmp_path, monkeypatch):
    source = tmp_path / "icon.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(source)
    monkeypatch.setattr(vectorize.shutil, "which", lambda _: "potrace")

    created = {}
    real_mkstemp = vectorize.tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        if path.endswith(".pbm"):
            created["pbm"] = path
        return fd, path

    monkeypatch.setattr(vectorize.tempfile, "mkstemp", tracking_mkstemp)

    class ExplodingBitmap:
        def convert(self, _mode):
            return self

        def save(self, _path):
            raise OSError("disk full")

    # vectorize.py imports PIL.Image locally inside _run_potrace, so patch PIL.Image itself
    # (there is no module-level `vectorize.Image` to intercept).
    import PIL.Image as pil_image
    monkeypatch.setattr(pil_image, "fromarray", lambda *a, **k: ExplodingBitmap())

    svg, err = vectorize._run_potrace(str(source), {})

    assert svg is None
    assert "potrace preprocess failed" in err
    assert "pbm" in created
    assert not os.path.exists(created["pbm"])


def test_potrace_uses_transparency_as_the_silhouette_and_recolors_white_icons(tmp_path, monkeypatch):
    icon = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
    ImageDraw.Draw(icon).polygon([(3, 10), (12, 3), (12, 7), (17, 7), (17, 13), (12, 13), (12, 17)],
                                 fill=(248, 248, 248, 255))
    source = tmp_path / "arrow.png"
    icon.save(source)
    observed = {}

    monkeypatch.setattr(vectorize.shutil, "which", lambda _: "potrace")

    def fake_run(command, **_kwargs):
        bitmap = Image.open(command[-1]).convert("L")
        observed["transparent"] = bitmap.getpixel((0, 0))
        observed["foreground"] = bitmap.getpixel((10, 10))
        Path(command[3]).write_text('<svg><path d="M0 0L1 0L1 1Z"/></svg>', encoding="utf-8")

    monkeypatch.setattr(vectorize.subprocess, "run", fake_run)
    svg, error = vectorize._run_potrace(str(source), {})
    assert error is None
    assert observed == {"transparent": 255, "foreground": 0}
    recolored = vectorize._recolor_potrace_svg(svg, vectorize._opaque_fill(str(source)))
    assert vectorize._parse_svg_paths(recolored)[0]["fill"] == "#f8f8f8"
