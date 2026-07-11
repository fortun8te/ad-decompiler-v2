import os
from pathlib import Path

from PIL import Image, ImageDraw

from src import vectorize


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
