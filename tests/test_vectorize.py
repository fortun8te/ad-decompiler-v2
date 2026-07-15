import os
import sys
import types
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


def test_trace_colour_count_ignores_transparent_matte_rgb(tmp_path):
    rng = np.random.default_rng(4)
    rgba = rng.integers(0, 255, size=(20, 20, 4), dtype=np.uint8)
    rgba[:, :, 3] = 0
    rgba[5:15, 5:15] = (12, 90, 200, 255)
    source = tmp_path / "icon.png"
    Image.fromarray(rgba).save(source)

    count, strategy = vectorize._trace_color_count(str(source))

    assert count == 1
    assert strategy == "alpha"


def test_normalize_trace_size_restores_original_crop_coordinates(tmp_path):
    source = tmp_path / "source.png"
    traced = tmp_path / "traced.png"
    Image.new("RGBA", (16, 8), (0, 0, 0, 0)).save(source)
    Image.new("RGBA", (96, 48), (0, 0, 0, 0)).save(traced)
    svg = (
        '<svg viewBox="0 0 96 48">'
        '<path d="M12 6L84 6L84 42Z" fill="#123456"/></svg>'
    )

    normalized = vectorize._normalize_trace_size(svg, str(traced), str(source))
    paths = vectorize._parse_svg_paths(normalized)

    assert 'viewBox="0 0 16 8"' in normalized
    assert "M2.00 1.00" in paths[0]["d"]
    assert "L14.00 7.00" in paths[0]["d"]
    assert paths[0]["fill"] == "#123456"


def test_vectorize_scores_and_returns_original_size_after_upscale(tmp_path, monkeypatch):
    source = tmp_path / "tiny.png"
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(source)
    seen_sizes = []

    monkeypatch.setattr(vectorize, "_count_colors", lambda _p: 3)
    monkeypatch.setattr(vectorize, "_run_vtracer", lambda *_a, **_k: (
        '<svg viewBox="0 0 96 96"><path d="M0 0L96 0L96 96Z" fill="#0a141e"/></svg>', None
    ))
    monkeypatch.setattr(vectorize, "_run_potrace", lambda *_a, **_k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_run_contour_simplify", lambda *_a, **_k: (None, "skip"))

    def score(_svg, path):
        seen_sizes.append(Image.open(path).size)
        return 0.99

    monkeypatch.setattr(vectorize, "_score_render", score)
    result = vectorize.vectorize_crop(str(source), {}, role="icon")

    assert result["ok"] is True
    assert seen_sizes == [(16, 16)]
    assert "L16.00 16.00" in result["paths"][0]["d"]


def test_check_binaries_reports_paths(monkeypatch):
    monkeypatch.setattr(vectorize.shutil, "which", lambda name: f"/bin/{name}")
    status = vectorize.check_binaries({})
    assert status["vtracer"]["ok"] is True
    assert status["potrace"]["ok"] is True
    assert "path" in status["vtracer"]


def test_check_backends_reports_trace_and_gate_health(monkeypatch):
    monkeypatch.setattr(vectorize, "check_binaries", lambda _cfg: {
        "vtracer": {"ok": False}, "potrace": {"ok": False},
        "contour": {"ok": True}, "cairosvg": {"ok": True}, "resvg": {"ok": False},
    })
    status = vectorize.check_backends({})
    assert status["ready"] is True
    assert status["fallback_ready"] is True


def test_check_binaries_reports_resvg(monkeypatch):
    monkeypatch.setitem(sys.modules, "resvg_py", types.SimpleNamespace())
    status = vectorize.check_binaries({})
    assert status["resvg"] == {"ok": True, "path": "python:resvg_py"}


def test_score_render_falls_back_to_resvg(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(source)
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        padded = Image.new("RGBA", (kwargs["width"], kwargs["height"]), (255, 0, 0, 255))
        buf = __import__("io").BytesIO()
        padded.save(buf, format="PNG")
        return buf.getvalue()

    fake_resvg = types.SimpleNamespace(svg_to_bytes=render)
    monkeypatch.setitem(sys.modules, "cairosvg", None)
    monkeypatch.setitem(sys.modules, "resvg_py", fake_resvg)

    score = vectorize._score_render(
        '<svg width="10" height="10"><rect width="10" height="10" fill="red"/></svg>',
        str(source),
    )
    assert score == 1.0
    assert calls == [{
        "svg_string": '<svg width="10" height="10"><rect width="10" height="10" fill="red"/></svg>',
        "width": 10,
        "height": 10,
    }]


def test_resvg_safe_api_is_thread_safe_and_legacy_tree_is_never_used(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    source = tmp_path / "source.png"
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(source)
    calls = []

    def safe_render(**kwargs):
        calls.append((type(kwargs["svg_string"]).__name__, kwargs["width"], kwargs["height"]))
        buf = __import__("io").BytesIO()
        Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(buf, format="PNG")
        return buf.getvalue()

    class UnsendableTree:
        @staticmethod
        def from_str(*_args):
            raise RuntimeError("tree is unsendable")

    monkeypatch.setitem(sys.modules, "cairosvg", None)
    monkeypatch.setitem(sys.modules, "resvg_py", types.SimpleNamespace(svg_to_bytes=safe_render))
    monkeypatch.setitem(sys.modules, "resvg", types.SimpleNamespace(
        usvg=types.SimpleNamespace(Tree=UnsendableTree),
        render=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("must not run")),
    ))
    svg = '<svg width="8" height="8"><rect width="8" height="8" fill="#0a141e"/></svg>'
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda _: vectorize._score_render(svg, str(source)), range(12)))

    assert scores == [1.0] * 12
    assert calls == [("str", 8, 8)] * 12


def test_resolve_binary_finds_repo_bin_windows_executable(tmp_path, monkeypatch):
    exe = Path(vectorize.__file__).parent.parent / ".bin" / "vtracer.exe"
    exe.parent.mkdir(exist_ok=True)
    try:
        exe.write_bytes(b"stub")
        monkeypatch.setattr(vectorize.shutil, "which", lambda _name: None)
        assert vectorize._resolve_binary({}, "color_engine", "vtracer") == str(exe)
    finally:
        if exe.exists():
            exe.unlink()
        if exe.parent.exists() and not any(exe.parent.iterdir()):
            exe.parent.rmdir()


def test_check_binaries_reports_contour_availability(monkeypatch):
    monkeypatch.setattr(vectorize.shutil, "which", lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    status = vectorize.check_binaries({})
    assert status["contour"]["ok"] is False
    assert "opencv" in status["contour"]["path"]


def test_evaluate_trace_applies_role_gate(monkeypatch):
    svg = '<svg><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000"/></svg>'
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.81)
    result, _ = vectorize._evaluate_trace(svg, "x.png", "vtracer", {}, "badge", 3)
    assert result["ok"] is True
    assert result["gate"]["score_min"] == 0.80


def test_evaluate_trace_rejects_lost_transparent_logo_counter(monkeypatch):
    svg = '<svg><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000"/></svg>'
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.99)
    monkeypatch.setattr(vectorize, "_transparent_hole_recall", lambda _s, _p: {
        "source_hole_pixels": 16, "trace_hole_pixels": 0, "recall": 0.0,
    })

    result, _ = vectorize._evaluate_trace(svg, "x.png", "vtracer", {}, "logo", 1)

    assert result["ok"] is False
    assert result["gate"]["hole_recall"]["recall"] == 0.0


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


def test_vtracer_python_api_is_used_when_binary_is_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "icon.png"
    Image.new("RGBA", (12, 12), (20, 40, 60, 255)).save(source)
    calls = []

    def convert(image_path, out_path, **kwargs):
        calls.append((image_path, out_path, kwargs))
        Path(out_path).write_text(
            '<svg><path d="M0 0L12 0L12 12Z" fill="#14283c"/></svg>',
            encoding="utf-8",
        )

    monkeypatch.setattr(vectorize, "_resolve_binary", lambda *args: None)
    monkeypatch.setitem(sys.modules, "vtracer", types.SimpleNamespace(
        convert_image_to_svg_py=convert,
    ))

    svg, error = vectorize._run_vtracer(str(source), {}, {
        "mode": "polygon", "colormode": "color", "filter_speckle": 3,
    })

    assert error is None
    assert "<path" in svg
    assert calls == [(str(source), calls[0][1], {
        "mode": "polygon", "colormode": "color", "filter_speckle": 3,
    })]
    assert not os.path.exists(calls[0][1])


def test_backend_probe_accepts_python_vtracer(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(vectorize, "_resolve_binary", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "vtracer", types.SimpleNamespace(
        convert_image_to_svg_py=lambda *_args, **_kwargs: None,
    ))

    status = vectorize.check_binaries({})

    assert status["vtracer"] == {"ok": True, "path": "python:vtracer"}


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


def test_contour_fallback_keeps_transparent_holes_with_evenodd_fill(tmp_path):
    source = tmp_path / "ring.png"
    icon = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.ellipse((3, 3, 28, 28), fill=(10, 20, 30, 255))
    draw.ellipse((10, 10, 21, 21), fill=(0, 0, 0, 0))
    icon.save(source)

    svg, error = vectorize._run_contour_simplify(str(source), {})

    assert error is None
    assert 'fill-rule="evenodd"' in svg
    paths = vectorize._parse_svg_paths(svg)
    assert paths[0]["windingRule"] == "EVENODD"


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


def test_analytic_rule_becomes_one_stroked_path(tmp_path):
    source = tmp_path / "rule.png"
    rgba = np.zeros((40, 160, 4), dtype=np.uint8)
    rgba[19:22, 5:155] = (20, 30, 40, 255)
    Image.fromarray(rgba).save(source)
    paths = vectorize._parse_svg_paths(
        vectorize._analytic_straight_line_svg(str(source), "divider")
    )
    assert len(paths) == 1
    assert paths[0]["fill"] == "none"
    assert paths[0]["stroke"]["color"] == "#141e28"
    assert paths[0]["stroke"]["width"] == 3.0


def test_arrowhead_is_not_simplified_to_a_plain_line(tmp_path):
    source = tmp_path / "arrow.png"
    image = Image.new("RGBA", (120, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((5, 25, 105, 25), fill=(0, 0, 0, 255), width=3)
    draw.polygon([(105, 12), (119, 25), (105, 38)], fill=(0, 0, 0, 255))
    image.save(source)
    assert vectorize._analytic_straight_line_svg(str(source), "arrow") is None


def test_parse_svg_paths_bakes_path_level_transform():
    # VTracer's python package emits translate() on each <path> itself (no wrapping <g>).
    # Dropping it displaced every traced shape to the origin, so upscaled traces always
    # failed the render gate and exported paths[] were wrong even when it passed.
    svg = ('<svg width="40" height="40">'
           '<path d="M0 0L8 0L8 8L0 8Z" transform="translate(16,4)" fill="#0a141e"/></svg>')
    paths = vectorize._parse_svg_paths(svg)
    assert "M16.00 4.00" in paths[0]["d"]
    assert "L24.00 12.00" in paths[0]["d"]


def _linear_gradient_crop(path, w=64, h=40):
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)[..., None]
    c0 = np.array([20, 50, 100], np.float32)
    c1 = np.array([150, 200, 255], np.float32)
    rgb = (c0[None, None] * (1 - xx) + c1[None, None] * xx).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, np.full((h, w), 255, np.uint8)])).save(path)


def test_gradient_crop_emits_one_silhouette_and_native_gradient_fill(tmp_path):
    source = tmp_path / "grad.png"
    _linear_gradient_crop(source)

    result = vectorize.vectorize_crop(str(source), {}, role="icon")

    assert result["ok"] is True
    assert result["engine"] == "analytic-gradient"
    assert len(result["paths"]) == 1  # not ten stacked colour bands
    assert result["paths"][0]["fill"].startswith("#")  # flat hex for existing consumers
    fill = result["gradient_fill"]
    assert fill["kind"] == "linear"
    assert abs(fill["angle"]) < 6.0
    assert [s["position"] for s in fill["stops"]] == [0, 1]
    assert all(s["color"].startswith("#") for s in fill["stops"])
    assert "<linearGradient" in result["svg"]  # true paint for SVG-capable importers
    assert fill["meta"]["flat_score"] >= vectorize._gate_limits("icon", {})[0]


def test_gradient_detection_is_config_gated(tmp_path):
    source = tmp_path / "grad_off.png"
    _linear_gradient_crop(source)
    cfg = {"vectorize": {"gradient": {"enabled": False}}}
    assert vectorize._detect_gradient(str(source), cfg) is None


def test_stepped_flag_bands_are_not_a_gradient(tmp_path):
    source = tmp_path / "flag.png"
    flag = np.zeros((40, 60, 4), np.uint8)
    flag[:, :20] = (0, 85, 164, 255)
    flag[:, 20:40] = (255, 255, 255, 255)
    flag[:, 40:] = (239, 65, 53, 255)
    Image.fromarray(flag).save(source)
    assert vectorize._detect_gradient(str(source), {}) is None


def test_radial_gradient_disk_gets_ellipse_silhouette_and_radial_fill(tmp_path):
    source = tmp_path / "radial.png"
    yy, xx = np.mgrid[0:60, 0:60].astype(np.float32)
    rr = np.hypot(xx - 29.5, yy - 29.5)
    t = np.clip(rr / 28.0, 0, 1)[..., None]
    rgb = (np.array([32, 32, 32], np.float32)[None, None] * (1 - t)
           + np.array([224, 224, 224], np.float32)[None, None] * t).astype(np.uint8)
    alpha = np.where(rr <= 28.0, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha])).save(source)

    result = vectorize.vectorize_crop(str(source), {}, role="icon")

    assert result["ok"] is True
    assert result["engine"] == "analytic-gradient"
    assert result["gradient_fill"]["kind"] == "radial"
    assert result["primitive"]["kind"] == "ellipse"
    assert "<radialGradient" in result["svg"]


def test_near_circle_icon_prefers_analytic_ellipse_primitive(tmp_path):
    source = tmp_path / "disk.png"
    icon = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    ImageDraw.Draw(icon).ellipse((4, 4, 43, 43), fill=(208, 64, 16, 255))
    icon.save(source)

    result = vectorize.vectorize_crop(str(source), {}, role="icon")

    assert result["ok"] is True
    assert result["engine"] == "analytic-primitive"
    prim = result["primitive"]
    assert prim["kind"] == "ellipse"
    assert prim["iou"] >= 0.94
    assert len(result["paths"]) == 1
    assert result["paths"][0]["d"].count("C") == 4  # the clean 4-curve ellipse
    assert result["score"] >= result["gate"]["score_min"]


def test_cross_shape_is_not_forced_into_a_primitive(tmp_path):
    source = tmp_path / "cross_prim.png"
    rgba = np.zeros((140, 140, 4), np.uint8)
    rgba[56:84, 8:132] = (10, 20, 30, 255)
    rgba[8:132, 56:84] = (10, 20, 30, 255)
    Image.fromarray(rgba).save(source)
    assert vectorize._flat_primitive_result(str(source), {}, "icon", 1) is None


def test_icon_midsize_upscale_traces_2x_and_restores_coordinates(tmp_path, monkeypatch):
    source = tmp_path / "two_tone.png"
    rgba = np.zeros((60, 60, 4), np.uint8)
    rgba[:, :30] = (200, 40, 40, 255)
    rgba[:, 30:] = (40, 40, 200, 255)
    Image.fromarray(rgba).save(source)
    traced_sizes = []

    def fake_vtracer(path, cfg, preset=None):
        traced_sizes.append(Image.open(path).size)
        return ('<svg viewBox="0 0 120 120">'
                '<path d="M0 0L120 0L120 120L0 120Z" fill="#c82828"/></svg>'), None

    monkeypatch.setattr(vectorize, "_run_vtracer", fake_vtracer)
    monkeypatch.setattr(vectorize, "_run_potrace", lambda *a, **k: (None, "skip"))
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.95)

    result = vectorize.vectorize_crop(str(source), {}, role="icon")

    assert result["ok"] is True
    assert traced_sizes[0] == (120, 120)  # sharp 2x conditioning before the trace
    assert "L60.00 60.00" in result["paths"][0]["d"]  # restored to crop coordinates

    traced_sizes.clear()
    cfg = {"vectorize": {"preprocess": {"icon_upscale_factor": 1.0}}}
    vectorize.vectorize_crop(str(source), cfg, role="icon")
    assert traced_sizes[0] == (60, 60)  # config-gated off


def test_fringe_alpha_is_snapped_but_translucent_design_kept(tmp_path):
    cfg = {"vectorize": {"preprocess": {"icon_upscale_factor": 1.0}}}
    fringe = tmp_path / "fringe.png"
    rgba = np.zeros((60, 60, 4), np.uint8)
    rgba[10:50, 10:50] = (10, 20, 30, 255)
    rgba[9, 10:50] = (10, 20, 30, 100)  # one-pixel AA fringe
    Image.fromarray(rgba).save(fringe)
    out, cleanup = vectorize._preprocess_crop(str(fringe), cfg, role="icon")
    try:
        alpha = np.asarray(Image.open(out).convert("RGBA"))[:, :, 3]
        assert set(np.unique(alpha)) <= {0, 255}
    finally:
        if cleanup:
            os.unlink(out)

    translucent = tmp_path / "translucent.png"
    rgba2 = np.zeros((60, 60, 4), np.uint8)
    rgba2[10:50, 10:50] = (10, 20, 30, 120)  # a real translucent panel
    Image.fromarray(rgba2).save(translucent)
    out2, cleanup2 = vectorize._preprocess_crop(str(translucent), cfg, role="icon")
    try:
        alpha2 = np.asarray(Image.open(out2).convert("RGBA"))[:, :, 3]
        assert 120 in np.unique(alpha2)  # mid-alpha kept: not a fringe
    finally:
        if cleanup2:
            os.unlink(out2)


def test_missing_potrace_degrades_once_to_binarized_vtracer(tmp_path, monkeypatch):
    source = tmp_path / "cross.png"
    rgba = np.zeros((140, 140, 4), np.uint8)
    rgba[56:84, 8:132] = (10, 20, 30, 255)
    rgba[8:132, 56:84] = (10, 20, 30, 255)
    Image.fromarray(rgba).save(source)
    observed = {}

    def fake_vtracer(path, cfg, preset=None):
        arr = np.asarray(Image.open(path).convert("RGBA"))
        observed["alpha"] = set(np.unique(arr[:, :, 3]).tolist())
        observed["colors"] = len(np.unique(arr[arr[:, :, 3] > 0][:, :3], axis=0))
        return ('<svg viewBox="0 0 140 140">'
                '<path d="M8 56L132 56L132 84L8 84Z" fill="#0a141e"/></svg>'), None

    def no_potrace(*_a, **_k):
        raise AssertionError("potrace must not be invoked when the binary is missing")

    real_resolve = vectorize._resolve_binary
    monkeypatch.setattr(
        vectorize, "_resolve_binary",
        lambda cfg, key, default: None if key == "binary_engine"
        else real_resolve(cfg, key, default),
    )
    monkeypatch.setattr(vectorize, "_run_vtracer", fake_vtracer)
    monkeypatch.setattr(vectorize, "_run_potrace", no_potrace)
    monkeypatch.setattr(vectorize, "_score_render", lambda _s, _p: 0.9)
    monkeypatch.setattr(vectorize, "_BINARY_DEGRADE_NOTED", False)

    first = vectorize.vectorize_crop(str(source), {}, role="icon")
    assert first["ok"] is True
    assert first["engine"] == "vtracer"
    assert observed["alpha"] <= {0, 255}  # traced a truly binarized crop
    assert observed["colors"] == 1
    assert "potrace" in first["note"] and "binarized" in first["note"]

    second = vectorize.vectorize_crop(str(source), {}, role="icon")
    assert "not installed" not in second["note"]  # degradation is noted once, not spammed
    assert "binarized" in second["note"]


def test_cleanup_is_rolled_back_when_render_gate_fails(tmp_path):
    source = tmp_path / "two_squares.png"
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[2:22, 2:22] = (16, 32, 64, 255)
    rgba[28:58, 28:58] = (16, 32, 64, 255)
    Image.fromarray(rgba).save(source)
    paths = [
        {"d": "M2 2L22 2L22 22L2 22Z", "fill": "#102040"},
        {"d": "M28 28L58 28L58 58L28 58Z", "fill": "#102040"},
    ]
    result = {"ok": True, "engine": "vtracer", "score": 0.99, "note": "synthetic",
              "paths": paths, "svg": "", "gate": {}}
    # min_area chosen to drop a REAL 400px piece of the icon: the re-gated render must
    # fail and the whole cleanup must be rolled back for this crop.
    cfg = {"vectorize": {"cleanup": {"min_area": 450, "merge_fills": False}}}

    out = vectorize._apply_cleanup(result, str(source), cfg, "icon", 1)

    assert out is result
    assert "cleanup" not in out
    assert len(out["paths"]) == 2


def test_cleanup_rescues_trace_that_only_failed_path_budget(tmp_path):
    source = tmp_path / "plate.png"
    rgba = np.zeros((40, 64, 4), np.uint8)
    rgba[5:35, 2:62] = (51, 102, 204, 255)
    Image.fromarray(rgba).save(source)
    strips = []
    for i in range(50):
        x0 = 2 + i * 60 / 50.0
        x1 = 2 + (i + 1) * 60 / 50.0
        strips.append({"d": f"M{x0:.2f} 5L{x1:.2f} 5L{x1:.2f} 35L{x0:.2f} 35Z",
                       "fill": "#3366cc"})
    over_budget = {"ok": False, "engine": "vtracer", "score": 0.99,
                   "note": "paths=50 over budget", "paths": strips, "svg": "", "gate": {}}

    out = vectorize._apply_cleanup(over_budget, str(source), {}, "icon", 1)

    assert out["ok"] is True
    assert out["cleanup"]["paths"] == [50, 1]
    assert len(out["paths"]) == 1
    assert out["score"] >= vectorize._gate_limits("icon", {})[0]
