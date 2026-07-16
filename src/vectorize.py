"""vectorize.py — stage: crop -> SVG paths for icons / simple graphics / masks.

vectorize_crop(png_path_or_array, cfg, role=None) traces a small raster crop into absolute
M/L/C/Z path d-strings with fills:

  * VTracer (multiple presets: color/cutout/binary, varying filter_speckle) is primary.
  * Potrace (binary/monochrome, multiple alpha thresholds) handles 1-color icons + masks.
  * OpenCV contour simplify is a last-resort fallback for flat single-color icons.

Output: {'ok', 'paths':[{'d','fill'}], 'engine', 'score', 'gate'}. Role-based quality gate
rasterizes the traced result and compares alpha to the source; ok=False when score or path
count exceeds role-specific limits (caller keeps the raster crop instead).

Additive optional result fields (backward compatible; absent unless earned):
  * 'gradient_fill' — a flat silhouette whose paint is a simple linear/radial gradient is
    emitted as ONE path (flat hex fill for existing consumers) plus this native paint
    description ({kind, angle?, stops, meta}); the returned 'svg' carries the real
    <linearGradient>/<radialGradient> so SVG-capable importers get the true paint.
  * 'primitive' — a near-circular / rounded-rect silhouette detected analytically
    ({kind: ellipse|rrect, geometry..., iou}); the paths are the clean 4-curve ellipse or
    rounded-rect instead of a wobbly trace.
  * 'cleanup' — post-trace svg_cleanup.py stats ({paths: [before, after], points: [...]});
    cleanup that fails the same render-back gate is rolled back per-crop.

NEVER throws: on a missing binary / trace failure it returns ok=False with a note.
Binaries: vtracer (`cargo install vtracer` or download release), potrace (`choco
install potrace` / `brew install potrace`). A missing potrace degrades to VTracer on a
binarized crop (single degradation note, no per-crop warning spam).
"""
from __future__ import annotations
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

# One long install-hint per process; later crops carry only the short "binarized" note.
_BINARY_DEGRADE_NOTED = False

_KAPPA = 0.5522847498307936  # cubic Bézier quarter-circle constant


def _fail(engine, note):
    return {"ok": False, "paths": [], "engine": engine, "score": 0.0, "note": note}


def _require_np():
    try:
        import numpy as np
        return np
    except ImportError as e:  # pragma: no cover
        raise ImportError("vectorize requires numpy.  pip install numpy") from e


def _to_png_path(png_path_or_array):
    """Return (path, cleanup_bool). Accepts a path or an HxWx{3,4} numpy array."""
    if isinstance(png_path_or_array, str):
        return png_path_or_array, False
    np = _require_np()
    from PIL import Image

    arr = np.asarray(png_path_or_array)
    mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr, mode=mode).save(tmp.name)
    return tmp.name, True


# ── SVG path normalization to absolute M/L/C/Z ────────────────────────────────────────
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_CMD = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])")

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose(m1, m2):
    """Return the matrix for 'apply m2 to the point, then m1' (i.e. m1 . m2)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_matrix(matrix, x, y):
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _parse_transform(transform_str):
    """Parse an SVG ``transform`` attribute (translate/scale/matrix/rotate/skew) into
    a single 2D affine matrix. Unknown/unsupported functions are ignored (identity)."""
    if not transform_str:
        return IDENTITY
    result = IDENTITY
    for name, args in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", transform_str):
        vals = [float(v) for v in _NUM.findall(args)]
        name = name.lower()
        if name == "translate":
            tx = vals[0] if vals else 0.0
            ty = vals[1] if len(vals) > 1 else 0.0
            m = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = vals[0] if vals else 1.0
            sy = vals[1] if len(vals) > 1 else sx
            m = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "matrix" and len(vals) == 6:
            m = tuple(vals)
        elif name == "rotate" and vals:
            import math
            ang = math.radians(vals[0])
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            rot = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(vals) >= 3:
                cx, cy = vals[1], vals[2]
                m = _compose(_compose((1.0, 0.0, 0.0, 1.0, cx, cy), rot),
                              (1.0, 0.0, 0.0, 1.0, -cx, -cy))
            else:
                m = rot
        elif name == "skewx" and vals:
            import math
            m = (1.0, 0.0, math.tan(math.radians(vals[0])), 1.0, 0.0, 0.0)
        elif name == "skewy" and vals:
            import math
            m = (1.0, math.tan(math.radians(vals[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue
        result = _compose(result, m)
    return result


def _tokenize_path(d):
    tokens = []
    for part in _CMD.split(d):
        part = part.strip()
        if not part:
            continue
        if part in "MmLlHhVvCcSsQqTtAaZz":
            tokens.append(("cmd", part))
        else:
            for m in _NUM.findall(part):
                tokens.append(("num", float(m)))
    return tokens


def _abs_path(d, matrix=None):
    """Convert a possibly-relative SVG d-string to absolute M/L/C/Z only.
    S/Q/T/A/H/V are expanded to their absolute L/C forms where reasonable.

    ``matrix`` (a 2D affine (a,b,c,d,e,f) tuple, see ``_parse_transform``) is applied to
    every *emitted* coordinate — e.g. the accumulated transform of any enclosing SVG <g>
    elements — while relative-coordinate bookkeeping (cx/cy/sx/sy) stays in the path's own
    local coordinate space, matching how relative commands are defined in the SVG spec.
    """
    m = matrix if matrix is not None else IDENTITY
    tokens = _tokenize_path(d)
    out = []
    i = 0
    cx = cy = 0.0
    sx = sy = 0.0
    prev_ctrl = None
    cur = None

    def nums(k):
        nonlocal i
        vals = []
        while len(vals) < k and i < len(tokens) and tokens[i][0] == "num":
            vals.append(tokens[i][1])
            i += 1
        return vals

    def pt(x, y):
        return _apply_matrix(m, x, y)

    while i < len(tokens):
        t = tokens[i]
        if t[0] == "cmd":
            cur = t[1]
            i += 1
        cmd = cur
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            x, y = nums(2)
            if rel:
                x += cx; y += cy
            cx, cy = x, y
            sx, sy = x, y
            ox, oy = pt(x, y)
            out.append(f"M{ox:.2f} {oy:.2f}")
            cur = "l" if rel else "L"  # subsequent implicit pairs = lineto
            prev_ctrl = None
        elif C == "L":
            x, y = nums(2)
            if rel:
                x += cx; y += cy
            cx, cy = x, y
            ox, oy = pt(x, y)
            out.append(f"L{ox:.2f} {oy:.2f}")
            prev_ctrl = None
        elif C == "H":
            (x,) = nums(1)
            if rel:
                x += cx
            cx = x
            ox, oy = pt(x, cy)
            out.append(f"L{ox:.2f} {oy:.2f}")
            prev_ctrl = None
        elif C == "V":
            (y,) = nums(1)
            if rel:
                y += cy
            cy = y
            ox, oy = pt(cx, y)
            out.append(f"L{ox:.2f} {oy:.2f}")
            prev_ctrl = None
        elif C == "C":
            x1, y1, x2, y2, x, y = nums(6)
            if rel:
                x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
            ox1, oy1 = pt(x1, y1)
            ox2, oy2 = pt(x2, y2)
            ox, oy = pt(x, y)
            out.append(f"C{ox1:.2f} {oy1:.2f} {ox2:.2f} {oy2:.2f} {ox:.2f} {oy:.2f}")
            prev_ctrl = (x2, y2)
            cx, cy = x, y
        elif C == "S":
            x2, y2, x, y = nums(4)
            if rel:
                x2 += cx; y2 += cy; x += cx; y += cy
            if prev_ctrl:
                x1 = 2 * cx - prev_ctrl[0]
                y1 = 2 * cy - prev_ctrl[1]
            else:
                x1, y1 = cx, cy
            ox1, oy1 = pt(x1, y1)
            ox2, oy2 = pt(x2, y2)
            ox, oy = pt(x, y)
            out.append(f"C{ox1:.2f} {oy1:.2f} {ox2:.2f} {oy2:.2f} {ox:.2f} {oy:.2f}")
            prev_ctrl = (x2, y2)
            cx, cy = x, y
        elif C == "Q":
            x1, y1, x, y = nums(4)
            if rel:
                x1 += cx; y1 += cy; x += cx; y += cy
            # quadratic -> cubic
            c1x = cx + 2 / 3 * (x1 - cx)
            c1y = cy + 2 / 3 * (y1 - cy)
            c2x = x + 2 / 3 * (x1 - x)
            c2y = y + 2 / 3 * (y1 - y)
            oc1x, oc1y = pt(c1x, c1y)
            oc2x, oc2y = pt(c2x, c2y)
            ox, oy = pt(x, y)
            out.append(f"C{oc1x:.2f} {oc1y:.2f} {oc2x:.2f} {oc2y:.2f} {ox:.2f} {oy:.2f}")
            prev_ctrl = (x1, y1)
            cx, cy = x, y
        elif C == "Z":
            out.append("Z")
            cx, cy = sx, sy
            prev_ctrl = None
        else:
            # unsupported (A arcs): SVG arc params are rx,ry,x-rot,large-arc,sweep,x,y —
            # skip the first 5 flags/radii but still advance the current point to the arc's
            # endpoint so subsequent relative-coordinate segments stay correctly anchored.
            skipped = nums(7)
            if len(skipped) == 7:
                x, y = skipped[5], skipped[6]
                if rel:
                    x += cx; y += cy
                cx, cy = x, y
            prev_ctrl = None
    return "".join(out)


_TAG_RE = re.compile(
    r"<g\b([^>]*?)(/?)>|(</g\s*>)|<path\b([^>]*?)/?>", re.DOTALL
)


def _parse_svg_paths(svg_text):
    """Extract [{d, fill}] from an SVG string, normalizing d to absolute pixel coordinates.

    Potrace's ``-s`` backend wraps every path in an enclosing ``<g transform="translate(..)
    scale(..)">`` that rescales/flips its internal trace units into real pixel space — that
    transform must be applied to each path's own coordinates, not just to the document as a
    whole (which is all a full-SVG raster/render would need). This walks tags in document
    order, maintaining a stack of cumulative transform matrices for nested <g> elements, and
    bakes the matrix in effect at each <path> into that path's absolute ``d`` string.
    """
    paths = []
    stack = [IDENTITY]
    for m in _TAG_RE.finditer(svg_text):
        g_attrs, g_selfclose, g_close, path_attrs = m.groups()
        if g_close is not None:
            if len(stack) > 1:
                stack.pop()
            continue
        if path_attrs is not None:
            attrs = path_attrs
            dm = re.search(r'\bd\s*=\s*"([^"]*)"', attrs)
            if not dm:
                continue
            fm = re.search(r'\bfill\s*=\s*"([^"]*)"', attrs)
            fill = fm.group(1) if fm else "#000000"
            style = re.search(r'\bstyle\s*=\s*"([^"]*)"', attrs)
            if style and "fill:" in style.group(1):
                fm2 = re.search(r"fill:\s*([^;]+)", style.group(1))
                if fm2:
                    fill = fm2.group(1).strip()
            stroke_match = re.search(r'\bstroke\s*=\s*"([^"]*)"', attrs)
            stroke = stroke_match.group(1) if stroke_match else None
            if fill.lower() in ("none",) and (not stroke or stroke.lower() == "none"):
                continue
            rule = None
            rm = re.search(r'\bfill-rule\s*=\s*"([^"]*)"', attrs)
            if rm:
                rule = rm.group(1).strip().lower()
            if style and "fill-rule:" in style.group(1):
                rm2 = re.search(r"fill-rule:\s*([^;]+)", style.group(1))
                if rm2:
                    rule = rm2.group(1).strip().lower()
            # VTracer (python package >= 0.6.x) puts translate() on each <path> itself
            # rather than an enclosing <g>. Dropping it displaced every traced shape to
            # the origin: re-serialized SVGs failed the render gate after upscaling, and
            # exported paths[] were silently wrong even when the raw SVG text passed.
            matrix = stack[-1]
            ptm = re.search(r'\btransform\s*=\s*"([^"]*)"', attrs)
            if ptm:
                matrix = _compose(matrix, _parse_transform(ptm.group(1)))
            try:
                d_abs = _abs_path(dm.group(1), matrix)
            except Exception:
                d_abs = dm.group(1)
            if d_abs:
                item = {"d": d_abs, "fill": fill}
                if stroke and stroke.lower() != "none":
                    width_match = re.search(r'\bstroke-width\s*=\s*"([^"]*)"', attrs)
                    cap_match = re.search(r'\bstroke-linecap\s*=\s*"([^"]*)"', attrs)
                    opacity_match = re.search(r'\bstroke-opacity\s*=\s*"([^"]*)"', attrs)
                    stroke_spec = {"color": stroke}
                    if width_match:
                        try:
                            stroke_spec["width"] = float(width_match.group(1))
                        except ValueError:
                            pass
                    if cap_match:
                        stroke_spec["cap"] = cap_match.group(1).upper()
                    if opacity_match:
                        try:
                            stroke_spec["opacity"] = float(opacity_match.group(1))
                        except ValueError:
                            pass
                    item["stroke"] = stroke_spec
                if rule == "evenodd":
                    item["windingRule"] = "EVENODD"
                paths.append(item)
            continue
        # <g ...> open tag (possibly self-closing, in which case it has no children/effect).
        tm = re.search(r'\btransform\s*=\s*"([^"]*)"', g_attrs or "")
        local = _parse_transform(tm.group(1)) if tm else IDENTITY
        cumulative = _compose(stack[-1], local)
        if g_selfclose == "/":
            continue
        stack.append(cumulative)
    return paths


# ── config helpers ───────────────────────────────────────────────────────────────────
_DEFAULT_VTRACER_PRESETS = [
    {"mode": "spline", "colormode": "color", "hierarchical": "stacked", "filter_speckle": 4},
    {"mode": "spline", "colormode": "color", "hierarchical": "cutout", "filter_speckle": 2},
    {"mode": "polygon", "colormode": "color", "hierarchical": "stacked", "filter_speckle": 8},
    {"mode": "spline", "colormode": "binary", "hierarchical": "stacked", "filter_speckle": 2},
]

_DEFAULT_SCORE_MIN = {
    "default": 0.85, "icon": 0.82, "logo": 0.82, "badge": 0.80,
    "arrow": 0.82, "mask": 0.78, "chip": 0.80, "shape": 0.88,
    # Hand-drawn annotation strokes (marker underlines, strikethroughs, connector/leader
    # arrows). Slightly looser than filled icons so thin white 014-style leaders survive
    # the render-back gate; a genuinely failed trace still falls back to raster honestly.
    "underline": 0.78, "strikethrough": 0.78, "strike_through": 0.78, "annotation": 0.78,
    "connector": 0.78, "leader": 0.78, "callout_leader": 0.78, "leader_line": 0.78,
    "string": 0.78, "thread": 0.78, "string_leader": 0.78, "leader_string": 0.78,
    "callout_string": 0.78,
    # Hand-drawn "weird" decoration marks (H11/H17): scribbled-out strokes crossing text,
    # organic brush-stroke banners, arrows. These are irregular by construction, so the
    # render-back gate is loosened to the same tier as connector strokes — a genuinely
    # failed trace still honestly falls back to a raster/alpha chip.
    "scribble": 0.76, "scribble_strike": 0.76, "scribble_strikethrough": 0.76,
    "scratch_out": 0.76, "redaction_scribble": 0.76, "hand_strike": 0.76,
    "brush": 0.78, "brush_stroke": 0.78, "brushstroke": 0.78, "brush_banner": 0.78,
    "paint_stroke": 0.78, "marker_stroke": 0.78,
    # Starburst / sunburst seal badges (H11) fit to a real regular-star polygon; a clean
    # analytic star always beats a wobbly trace, so the fitter gates separately and this
    # entry only bounds the fallback trace when the star fit is rejected.
    "starburst": 0.80, "star": 0.80, "sunburst": 0.80, "seal": 0.80, "star_badge": 0.80,
    "circle_badge": 0.80, "badge_circle": 0.80, "price_badge": 0.80,
    "line": 0.80, "divider": 0.80,
    # Chart/diagram strokes and markers: same tier as connectors — prefer editable
    # vectors when the render-back gate passes; never force a photo-style reject bar.
    "axis": 0.80, "axis_line": 0.80, "axis-line": 0.80, "gridline": 0.80,
    "plot_line": 0.80, "plot-line": 0.80, "data_line": 0.80, "data-line": 0.80,
    "data_point": 0.80, "data-point": 0.80, "marker": 0.80,
    "chart_bar": 0.82, "chart-bar": 0.82, "bar": 0.82,
}
_DEFAULT_MAX_PATHS = {
    "default": 40, "icon": 60, "logo": 50, "badge": 35,
    "arrow": 30, "mask": 20, "chip": 35, "shape": 45,
    # A single authored annotation stroke is one or two paths; a scribble a handful more.
    "underline": 14, "strikethrough": 14, "strike_through": 14,
    "connector": 24, "leader": 24, "callout_leader": 24, "leader_line": 24,
    "string": 24, "thread": 24, "string_leader": 24, "leader_string": 24,
    "callout_string": 24,
    # A scribble is a small handful of overlapping strokes; a brush banner one organic
    # blob; a starburst one star polygon (fallback trace can be a bit richer).
    "scribble": 24, "scribble_strike": 24, "scribble_strikethrough": 24,
    "scratch_out": 24, "redaction_scribble": 24, "hand_strike": 24,
    "brush": 20, "brush_stroke": 20, "brushstroke": 20, "brush_banner": 20,
    "paint_stroke": 20, "marker_stroke": 20,
    "starburst": 30, "star": 30, "sunburst": 30, "seal": 30, "star_badge": 30,
    "circle_badge": 12, "badge_circle": 12, "price_badge": 12,
    "line": 14, "divider": 14,
    "axis": 8, "axis_line": 8, "axis-line": 8, "gridline": 8,
    "plot_line": 16, "plot-line": 16, "data_line": 16, "data-line": 16,
    "data_point": 12, "data-point": 12, "marker": 12,
    "chart_bar": 8, "chart-bar": 8, "bar": 8,
}


def _vz_cfg(cfg):
    return (cfg or {}).get("vectorize") or {}


def _gate_limits(role, cfg):
    vz = _vz_cfg(cfg)
    role_key = str(role or "default").lower()
    score_cfg = vz.get("score_min") or _DEFAULT_SCORE_MIN
    paths_cfg = vz.get("max_paths") or _DEFAULT_MAX_PATHS
    score_min = float(score_cfg.get(role_key, score_cfg.get("default", 0.85)))
    max_paths = int(paths_cfg.get(role_key, paths_cfg.get("default", 40)))
    return score_min, max_paths


def _vtracer_presets(cfg):
    presets = _vz_cfg(cfg).get("vtracer_presets")
    return presets if presets else _DEFAULT_VTRACER_PRESETS


def _potrace_thresholds(cfg):
    thresholds = _vz_cfg(cfg).get("potrace_thresholds")
    return thresholds if thresholds else [8, 16, 32, 64]


def _resolve_binary(cfg, key, default):
    """Resolve a configured tracer executable on all supported host platforms.

    ``shutil.which`` does not search the repository-local ``.bin`` directory and,
    on Windows, a configured bare name is not necessarily spelled with ``.exe``.
    The example configuration explicitly promises ``.bin/vtracer(.exe)`` support,
    so keep that lookup here rather than making every caller know about it.
    """
    configured = _vz_cfg(cfg).get(key, default)
    if configured is None:
        return None
    binary = os.fspath(configured)
    candidates = [binary]
    if not os.path.splitext(binary)[1]:
        candidates.extend(binary + ext for ext in (".exe", ".cmd", ".bat"))

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # The documented local install location is relative to the repository, not
    # the process working directory (which is often a benchmark output folder).
    repo_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".bin"))
    for name in candidates:
        local = os.path.join(repo_bin, os.path.basename(name))
        if os.path.isfile(local):
            return local
    return None


def check_binaries(cfg=None):
    """Report vtracer/potrace/cairosvg availability for doctor / health probes."""
    cfg = cfg or {}
    vtracer = _resolve_binary(cfg, "color_engine", "vtracer")
    potrace = _resolve_binary(cfg, "binary_engine", "potrace")
    try:
        from vtracer import convert_image_to_svg_py as _vtracer_python  # noqa: F401
        python_vtracer = True
    except Exception:
        python_vtracer = False
    try:
        import cairosvg  # noqa: F401
        gate = True
    except ImportError:
        gate = False
    except Exception:
        # cairosvg is pip-installed but its native libcairo binding (cairocffi) raises
        # OSError, not ImportError, when the system libcairo-2 DLL/so/dylib is missing
        # (common on a bare Windows box). check_binaries() feeds doctor.inspect(), and an
        # unhandled exception here previously crashed readiness reporting entirely instead
        # of just marking this one optional gate unavailable.
        gate = False
    try:
        import cv2  # noqa: F401
        contour = True
    except Exception:
        contour = False
    try:
        import resvg_py  # noqa: F401
        resvg_path = "python:resvg_py"
    except Exception:
        resvg_cli = shutil.which("resvg")
        resvg_path = resvg_cli or None
    return {
        "vtracer": {
            "ok": bool(vtracer or python_vtracer),
            "path": vtracer or ("python:vtracer" if python_vtracer else "pip install vtracer"),
        },
        "potrace": {"ok": bool(potrace), "path": potrace or "not on PATH (brew/choco install potrace)"},
        "contour": {"ok": contour, "path": "opencv-python" if contour else "not installed (pip install opencv-python)"},
        "cairosvg": {"ok": gate, "path": "installed" if gate else "pip install cairosvg (quality gate)"},
        # The legacy `resvg` Python package exposes a native usvg.Tree. That tree is
        # thread-affine and recent builds fail with "tree is unsendable" when render()
        # moves work internally. resvg_py deliberately accepts SVG text and returns
        # PNG bytes in one call, so no native object crosses a thread/process boundary.
        "resvg": {"ok": bool(resvg_path), "path": resvg_path or "not installed (pip install resvg_py)"},
    }


def check_backends(cfg=None):
    """Return JSON-safe vector backend health, including usable fallbacks."""
    status = check_binaries(cfg)
    trace_ready = bool(status["vtracer"]["ok"] or status["potrace"]["ok"] or status["contour"]["ok"])
    gate_ready = bool(status["cairosvg"]["ok"] or status["resvg"]["ok"])
    return {
        "trace": status,
        "ready": trace_ready and gate_ready,
        "fallback_ready": bool(status["contour"]["ok"] and gate_ready),
    }


# ── preprocess ───────────────────────────────────────────────────────────────────────
def _preprocess_crop(png_path, cfg, role=None):
    """Return (processed_png_path, cleanup_bool). Quantize, de-fringe, upscale small icons."""
    np = _require_np()
    from PIL import Image

    pre = _vz_cfg(cfg).get("preprocess") or {}
    if pre.get("enabled", True) is False:
        return png_path, False

    with Image.open(png_path) as im:
        rgba = im.convert("RGBA")
        original_arr = np.array(rgba, dtype=np.uint8, copy=True)
        arr = original_arr.copy()
        h, w = arr.shape[:2]

        # Upscale small icons with a sharp kernel so tracers see smooth, well-sampled
        # edges instead of a handful of aliased pixels. Coordinates are restored to the
        # original crop bounds by _normalize_trace_size after tracing.
        min_dim = min(w, h)
        upscale_below = int(pre.get("upscale_below", 48))
        upscale_target = int(pre.get("upscale_target", 96))
        kernel = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }.get(str(pre.get("upscale_kernel", "lanczos")).lower(), Image.Resampling.LANCZOS)
        scale = 1.0
        if 0 < min_dim < upscale_below:
            scale = upscale_target / float(min_dim)
        else:
            icon_below = int(pre.get("icon_upscale_below", 128))
            icon_factor = float(pre.get("icon_upscale_factor", 2.0))
            role_key0 = str(role or "").lower()
            if (icon_factor > 1.0 and 0 < min_dim < icon_below
                    and role_key0 in ("icon", "logo", "badge", "arrow", "chip", "mask")):
                scale = icon_factor
        if scale > 1.0:
            # Keep the traced raster bounded; a marginal upscale is not worth re-encoding.
            scale = min(scale, 512.0 / float(max(w, h)))
        if scale > 1.15:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            rgba = rgba.resize((nw, nh), kernel)
            arr = np.array(rgba, dtype=np.uint8, copy=True)
            h, w = arr.shape[:2]

        # Remove weak anti-alias fringe while preserving meaningful edge coverage. A soft
        # one-pixel fringe makes VTracer invent translucent sliver paths, so when the
        # semi-transparent band is fringe-sized it is snapped hard to 0/255; a genuinely
        # translucent design element (large mid-alpha fraction) keeps its soft alpha.
        if pre.get("denoise_fringe", True):
            alpha = arr[:, :, 3].astype(np.float32)
            hard = np.where(alpha >= 200, 255, np.where(alpha <= 40, 0, alpha))
            if pre.get("fringe_snap", True) is not False:
                visible = alpha > 40
                middle = visible & (alpha < 200)
                fringe_frac = (float(middle.sum()) / float(visible.sum())) if visible.any() else 0.0
                if 0.0 < fringe_frac <= float(pre.get("fringe_max_fraction", 0.35)):
                    snap = float(pre.get("alpha_snap", 128))
                    hard = np.where(alpha >= snap, 255, 0)
            arr[:, :, 3] = hard.astype(np.uint8)

        # Quantize colors for cleaner multi-color traces (skip for near-monochrome).
        if pre.get("quantize_colors", True):
            opaque = arr[:, :, 3] > 8
            if opaque.any():
                rgb = arr[:, :, :3].reshape(-1, 3)
                mask = opaque.reshape(-1)
                pixels = rgb[mask]
                n_unique = len(np.unique((pixels // 16) * 16, axis=0))
                max_colors = int(pre.get("max_colors", 16))
                role_key = str(role or "").lower()
                if role_key in ("icon", "badge", "logo", "arrow", "chip"):
                    max_colors = min(max_colors, 12)
                if n_unique > 2 and n_unique <= max_colors * 4:
                    step = max(8, 256 // max_colors)
                    quant = (arr[:, :, :3].astype(np.uint16) // step) * step
                    arr[:, :, :3] = quant.astype(np.uint8)

        # Fully transparent RGB is often a matte/background colour. Keeping it lets
        # VTracer invent fringe paths even though it is visually absent.
        arr[arr[:, :, 3] == 0, :3] = 0
        if np.array_equal(arr, original_arr):
            return png_path, False

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr).save(tmp.name)
        return tmp.name, True


# ── engines ──────────────────────────────────────────────────────────────────────────
def _foreground_mask(rgba, alpha_threshold=8, lum_threshold=128):
    """Choose a binary foreground from a transparent icon or an opaque logo crop.

    A preprocessed RGB crop is often saved as RGBA with alpha=255 everywhere.
    Treating that alpha as a silhouette makes Potrace trace the entire crop.  Use
    true transparency when it exists; otherwise isolate colour contrast against
    the border and only then fall back to a luminance bitmap.
    """
    np = _require_np()
    alpha = rgba[:, :, 3]
    visible = alpha > alpha_threshold
    if bool(visible.any()) and bool((~visible).any()):
        return visible, "alpha"
    rgb = rgba[:, :, :3].astype(np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    reference = np.median(border, axis=0)
    contrast = np.max(np.abs(rgb - reference), axis=2) >= 16
    fraction = float(contrast.mean())
    if 0.002 <= fraction <= 0.98:
        return contrast, "border-contrast"
    luminance = (299 * rgb[:, :, 0] + 587 * rgb[:, :, 1] + 114 * rgb[:, :, 2]) / 1000.0
    return luminance <= lum_threshold, "luminance"


def _analytic_straight_line_svg(png_path, role):
    """Fit a single authored rule/leader as stroke geometry.

    This is intentionally stricter than tracing.  Arrowheads, dots, elbows, handwriting,
    and decorative endpoints increase the perpendicular residual and are rejected; the
    ordinary render-back gate then lets VTracer/Potrace preserve those details instead.
    """
    role_key = str(role or "").lower().replace("-", "_")
    if role_key not in {
        "line", "divider", "rule", "separator", "underline", "strikethrough",
        "strike_through", "callout_leader", "leader", "leader_line", "connector",
        "string", "thread", "string_leader", "leader_string", "callout_string",
        "axis", "axis_line", "gridline", "plot_line", "data_line",
    }:
        return None
    try:
        import numpy as np
        from PIL import Image

        rgba = np.asarray(Image.open(png_path).convert("RGBA"), dtype=np.uint8)
        h, w = rgba.shape[:2]
        if min(w, h) < 1 or max(w, h) < 6:
            return None
        mask, _ = _foreground_mask(rgba)
        ys, xs = np.nonzero(mask)
        if len(xs) < 6:
            return None
        points = np.column_stack((xs, ys)).astype(np.float64)
        centre = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
        major, minor = axes[0], axes[1]
        along = (points - centre) @ major
        across = np.abs((points - centre) @ minor)
        length = float(np.percentile(along, 99) - np.percentile(along, 1))
        half_width = float(np.percentile(across, 92))
        thickness = max(1.0, 2.0 * half_width + 1.0)
        # A true rule is strongly one-dimensional. Endpoint ornaments and arrowheads
        # deliberately fail this test and continue to the exact gated tracer.
        if length < 4.0 * thickness or float(np.percentile(across, 98)) > thickness * 0.78:
            return None
        lo, hi = float(np.percentile(along, 1)), float(np.percentile(along, 99))
        p0, p1 = centre + major * lo, centre + major * hi
        visible = rgba[mask]
        colour = np.median(visible[:, :3], axis=0).astype(int)
        alpha = float(np.median(visible[:, 3])) / 255.0
        cap = "round" if role_key in {
            "callout_leader", "leader", "leader_line", "connector",
            "string", "thread", "string_leader", "leader_string", "callout_string",
        } else "butt"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
            f'<path d="M{p0[0]:.3f} {p0[1]:.3f} L{p1[0]:.3f} {p1[1]:.3f}" '
            f'fill="none" stroke="#{colour[0]:02x}{colour[1]:02x}{colour[2]:02x}" '
            f'stroke-width="{thickness:.3f}" stroke-linecap="{cap}" '
            f'stroke-opacity="{alpha:.4f}"/></svg>'
        )
    except Exception:
        return None


def _rgb_hex(rgb):
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in list(rgb)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_rgb(value):
    s = str(value or "").strip().lower()
    if not s.startswith("#"):
        return None
    s = s[1:]
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _mix_hex(a, b):
    ca, cb = _hex_rgb(a), _hex_rgb(b)
    if not ca or not cb:
        return a if ca else "#808080"
    return _rgb_hex([(va + vb) / 2.0 for va, vb in zip(ca, cb)])


def _ellipse_d(cx, cy, rx, ry):
    """Four-cubic ellipse path (the same clean geometry a designer would draw)."""
    k = _KAPPA
    x0, x1 = cx - rx, cx + rx
    y0, y1 = cy - ry, cy + ry
    return (
        f"M{x1:.2f} {cy:.2f}"
        f"C{x1:.2f} {cy + k * ry:.2f} {cx + k * rx:.2f} {y1:.2f} {cx:.2f} {y1:.2f}"
        f"C{cx - k * rx:.2f} {y1:.2f} {x0:.2f} {cy + k * ry:.2f} {x0:.2f} {cy:.2f}"
        f"C{x0:.2f} {cy - k * ry:.2f} {cx - k * rx:.2f} {y0:.2f} {cx:.2f} {y0:.2f}"
        f"C{cx + k * rx:.2f} {y0:.2f} {x1:.2f} {cy - k * ry:.2f} {x1:.2f} {cy:.2f}Z"
    )


def _rrect_d(x, y, w, h, r):
    """Axis-aligned rounded-rect path; r=0 degenerates to a plain 4-point rect."""
    r = max(0.0, min(float(r), w / 2.0, h / 2.0))
    right, bottom = x + w, y + h
    if r < 0.5:
        return (f"M{x:.2f} {y:.2f}L{right:.2f} {y:.2f}L{right:.2f} {bottom:.2f}"
                f"L{x:.2f} {bottom:.2f}Z")
    k = _KAPPA * r
    return (
        f"M{x + r:.2f} {y:.2f}"
        f"L{right - r:.2f} {y:.2f}"
        f"C{right - r + k:.2f} {y:.2f} {right:.2f} {y + r - k:.2f} {right:.2f} {y + r:.2f}"
        f"L{right:.2f} {bottom - r:.2f}"
        f"C{right:.2f} {bottom - r + k:.2f} {right - r + k:.2f} {bottom:.2f} "
        f"{right - r:.2f} {bottom:.2f}"
        f"L{x + r:.2f} {bottom:.2f}"
        f"C{x + r - k:.2f} {bottom:.2f} {x:.2f} {bottom - r + k:.2f} {x:.2f} {bottom - r:.2f}"
        f"L{x:.2f} {y + r:.2f}"
        f"C{x:.2f} {y + r - k:.2f} {x + r - k:.2f} {y:.2f} {x + r:.2f} {y:.2f}Z"
    )


def _star_d(cx, cy, r_outer, r_inner, points, rotation=0.0):
    """Regular N-point star polygon path (outer tip / inner valley alternating).

    ``rotation`` is the angle (radians) of the first outer tip, measured from +x. The
    same clean vertex geometry a designer would draw with Figma's star tool — a real
    star beats a dozen-Bézier trace of a starburst seal.
    """
    n = max(3, int(points))
    verts = []
    for i in range(n):
        a_out = rotation + 2.0 * math.pi * i / n
        a_in = rotation + 2.0 * math.pi * (i + 0.5) / n
        verts.append((cx + r_outer * math.cos(a_out), cy + r_outer * math.sin(a_out)))
        verts.append((cx + r_inner * math.cos(a_in), cy + r_inner * math.sin(a_in)))
    body = "".join(f"L{x:.2f} {y:.2f}" for x, y in verts[1:])
    return f"M{verts[0][0]:.2f} {verts[0][1]:.2f}{body}Z"


def _primitive_d(prim):
    if prim["kind"] == "ellipse":
        return _ellipse_d(prim["cx"], prim["cy"], prim["rx"], prim["ry"])
    if prim["kind"] == "star":
        return _star_d(prim["cx"], prim["cy"], prim["r_outer"], prim["r_inner"],
                       prim["points"], prim.get("rotation", 0.0))
    return _rrect_d(prim["x"], prim["y"], prim["w"], prim["h"], prim.get("radius", 0.0))


def _fit_primitive(mask, min_iou=0.94, allow_plain_rect=False):
    """Fit an ellipse or rounded-rect to a boolean silhouette; None unless IoU clears.

    A near-circle traced by VTracer is a dozen wobbly Béziers; the analytic primitive is
    the clean geometry a designer would author.  Sharp-cornered plain rects are excluded
    by default (radius < 1.5px) so ordinary tracing keeps handling them.
    """
    np = _require_np()
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if min(bw, bh) < 6:
        return None
    local = mask[y0:y1 + 1, x0:x1 + 1]
    area = float(local.sum())
    if area / float(bw * bh) < 0.5:
        return None  # sparse marks (crosses, arrows, glyphs) are not primitives
    yy, xx = np.mgrid[0:bh, 0:bw]
    cx, cy = (bw - 1) / 2.0, (bh - 1) / 2.0

    def iou(candidate):
        union = float(np.logical_or(candidate, local).sum())
        return float(np.logical_and(candidate, local).sum()) / union if union else 0.0

    best = None
    ellipse = (((xx - cx) / (bw / 2.0)) ** 2 + ((yy - cy) / (bh / 2.0)) ** 2) <= 1.0
    ell_iou = iou(ellipse)
    if ell_iou >= min_iou:
        best = {"kind": "ellipse", "cx": round(x0 + cx, 2), "cy": round(y0 + cy, 2),
                "rx": round(bw / 2.0, 2), "ry": round(bh / 2.0, 2),
                "iou": round(ell_iou, 4)}
    limit = min(bw, bh) / 2.0
    r_est = math.sqrt(max(0.0, (bw * bh - area) / (4.0 - math.pi)))
    radii = {min(limit, max(0.0, r_est * f)) for f in (0.75, 1.0, 1.25)}
    if allow_plain_rect:
        radii.add(0.0)
    for r in sorted(radii):
        if r < 1.5 and not allow_plain_rect:
            continue
        # Distance from each pixel centre to the radius-inset core rectangle.
        px = np.clip(xx, r - 0.5, bw - 0.5 - r)
        py = np.clip(yy, r - 0.5, bh - 0.5 - r)
        rrect = ((xx - px) ** 2 + (yy - py) ** 2) <= r * r
        rr_iou = iou(rrect)
        # A circle is also a maximally-rounded rect; prefer the semantic ellipse unless
        # the rounded-rect is a materially better fit, not a float-noise winner.
        beat = best["iou"] + (0.005 if best["kind"] == "ellipse" else 0.0) if best else 0.0
        if rr_iou >= min_iou and (best is None or rr_iou > beat):
            best = {"kind": "rrect", "x": float(x0), "y": float(y0),
                    "w": float(bw), "h": float(bh), "radius": round(float(r), 2),
                    "iou": round(rr_iou, 4)}
    return best


def _fit_star_polygon(mask, min_iou=0.90, min_points=5, max_points=24):
    """Fit a regular N-point star (starburst seal) to a spiky silhouette.

    A starburst is exactly what ``_fit_primitive`` rejects (sparse area fraction, no
    ellipse/rrect match), yet it is one of the cleanest shapes to author natively — a
    Figma star with a point count and two radii. The fit casts rays from the silhouette
    centroid to build a radial profile r(theta), counts the evenly-spaced tips, then
    refines the geometry to maximize IoU. Returns a primitive dict (kind="star") or None.
    """
    np = _require_np()
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if min(bw, bh) < 12:
        return None
    local = mask[y0:y1 + 1, x0:x1 + 1]
    area = float(local.sum())
    fill_frac = area / float(bw * bh)
    # A star's convex hull is far larger than its ink: a near-solid blob (disc, rrect)
    # or a near-empty ring is not a star. Bracket the ink fraction to the spiky regime.
    if not (0.30 <= fill_frac <= 0.80):
        return None
    ys_l, xs_l = np.nonzero(local)
    cx = float(xs_l.mean())
    cy = float(ys_l.mean())

    n_ang = 360
    angles = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
    # Max foreground radius along each ray (0 where the ray leaves the silhouette early).
    dxs = np.cos(angles)
    dys = np.sin(angles)
    max_r = float(math.hypot(bw, bh))
    steps = np.arange(0.0, max_r, 0.5)
    profile = np.zeros(n_ang, dtype=np.float32)
    h_l, w_l = local.shape
    for k, (dx, dy) in enumerate(zip(dxs, dys)):
        px = cx + steps * dx
        py = cy + steps * dy
        inside = (px >= 0) & (px < w_l) & (py >= 0) & (py < h_l)
        if not inside.any():
            continue
        pxi = px[inside].astype(np.int32)
        pyi = py[inside].astype(np.int32)
        fg = local[pyi, pxi]
        hit = np.nonzero(fg)[0]
        if len(hit):
            profile[k] = float(steps[inside][hit[-1]])
    if float(profile.max()) <= 0.0:
        return None

    # Circularly smooth, then count peaks (tips) as evenly-spaced maxima above the mean.
    kern = np.ones(5, dtype=np.float32) / 5.0
    sm = np.convolve(np.concatenate([profile[-4:], profile, profile[:4]]), kern, "same")[4:-4]
    thresh = 0.5 * (float(sm.max()) + float(sm.min()))
    peaks = []
    for k in range(n_ang):
        v = sm[k]
        if v < thresh:
            continue
        if v >= sm[(k - 1) % n_ang] and v >= sm[(k + 1) % n_ang]:
            peaks.append(k)
    # Merge adjacent plateau indices into single peaks.
    merged = []
    for k in peaks:
        if merged and (k - merged[-1][-1]) <= 3:
            merged[-1].append(k)
        else:
            merged.append([k])
    if merged and len(merged) > 1 and (n_ang - merged[-1][-1] + merged[0][0]) <= 3:
        merged[0] = merged.pop() + merged[0]
    tips = [grp[int(np.argmax(sm[grp]))] for grp in merged]
    n_pts = len(tips)
    if not (min_points <= n_pts <= max_points):
        return None
    # Tips must be roughly evenly spaced (regular star), else it is an organic blob.
    tip_ang = np.sort(np.array([angles[t] for t in tips]))
    gaps = np.diff(np.concatenate([tip_ang, [tip_ang[0] + 2 * math.pi]]))
    if float(gaps.std()) > (2.0 * math.pi / n_pts) * 0.45:
        return None

    r_outer = float(np.percentile(profile[profile > 0], 88))
    valleys = [profile[(t + n_ang // (2 * n_pts)) % n_ang] for t in tips]
    r_inner = float(np.median([v for v in valleys if v > 0]) if any(valleys) else r_outer * 0.5)
    r_inner = max(1.0, min(r_inner, r_outer * 0.92))
    rotation = float(angles[tips[int(np.argmax([profile[t] for t in tips]))]])

    yy, xx = np.mgrid[0:h_l, 0:w_l]

    def star_iou(ro, ri, rot):
        # Point-in-polygon via matplotlib-free ray param: rebuild vertices, test winding.
        verts = []
        for i in range(n_pts):
            ao = rot + 2 * math.pi * i / n_pts
            ai = rot + 2 * math.pi * (i + 0.5) / n_pts
            verts.append((cx + ro * math.cos(ao), cy + ro * math.sin(ao)))
            verts.append((cx + ri * math.cos(ai), cy + ri * math.sin(ai)))
        vx = np.array([p[0] for p in verts])
        vy = np.array([p[1] for p in verts])
        inside = np.zeros((h_l, w_l), dtype=bool)
        j = len(vx) - 1
        px = xx.astype(np.float32)
        py = yy.astype(np.float32)
        for i in range(len(vx)):
            cond = ((vy[i] > py) != (vy[j] > py)) & (
                px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-9) + vx[i])
            inside ^= cond
            j = i
        union = float(np.logical_or(inside, local).sum())
        return float(np.logical_and(inside, local).sum()) / union if union else 0.0

    best_iou = 0.0
    best = (r_outer, r_inner, rotation)
    for ro_f in (0.95, 1.0, 1.05):
        for ri_f in (0.8, 1.0, 1.2):
            cand = star_iou(r_outer * ro_f, min(r_outer * 0.92, r_inner * ri_f), rotation)
            if cand > best_iou:
                best_iou = cand
                best = (r_outer * ro_f, min(r_outer * 0.92, r_inner * ri_f), rotation)
    if best_iou < min_iou:
        return None
    ro, ri, rot = best
    return {"kind": "star", "cx": round(x0 + cx, 2), "cy": round(y0 + cy, 2),
            "r_outer": round(ro, 2), "r_inner": round(ri, 2),
            "points": n_pts, "rotation": round(rot, 4), "iou": round(best_iou, 4)}


def _detect_gradient(png_path, cfg):
    """Fit a simple linear / centred-radial paint over the crop's visible pixels.

    Mirrors reconstruct.py's style_extraction regression (colour vs position, R² gate,
    radial-preference margin) so both stages agree about what a design gradient is.  A
    binned smoothness check rejects stepped multi-band fills (flags, colour bars) that a
    plane can numerically explain but that are authored as discrete bands.
    """
    gz = _vz_cfg(cfg).get("gradient") or {}
    if gz.get("enabled", True) is False:
        return None
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    h, w = rgba.shape[:2]
    if min(w, h) < 10:
        return None
    visible = rgba[:, :, 3] > 8
    if not visible.any():
        return None
    mask = visible if bool((~visible).any()) else np.ones((h, w), dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 80:
        return None
    min_range = float(gz.get("min_range", 18.0))
    min_r2 = float(gz.get("min_r2", 0.86))
    radial_min_r2 = float(gz.get("radial_min_r2", 0.91))
    margin = float(gz.get("radial_r2_margin", 0.035))
    band_tol = float(gz.get("smoothness_tolerance", 10.0))

    sx, sy = max(1.0, (w - 1) / 2.0), max(1.0, (h - 1) / 2.0)
    x = (xs.astype(np.float32) - (w - 1) / 2.0) / sx
    y = (ys.astype(np.float32) - (h - 1) / 2.0) / sy
    colors = rgba[ys, xs, :3].astype(np.float32)
    if len(colors) > 12000:
        pick = np.linspace(0, len(colors) - 1, 12000).astype(int)
        x, y, colors = x[pick], y[pick], colors[pick]
    spread = np.percentile(colors, 95, axis=0) - np.percentile(colors, 5, axis=0)
    range_norm = float(np.linalg.norm(spread))
    if range_norm < min_range:
        return None
    total = float(np.square(colors - colors.mean(axis=0)).sum())
    if total <= 1e-6:
        return None

    def smooth_enough(projection, prediction):
        lo, hi = float(projection.min()), float(projection.max())
        if hi - lo <= 1e-6:
            return False
        edges = np.linspace(lo, hi, 13)
        idx = np.clip(np.digitize(projection, edges) - 1, 0, 11)
        worst = 0.0
        for b in range(12):
            sel = idx == b
            if int(sel.sum()) < 20:
                continue
            dev = float(np.abs(colors[sel].mean(axis=0) - prediction[sel].mean(axis=0)).max())
            worst = max(worst, dev)
        return worst <= band_tol

    def monotonic_enough(projection):
        """Accept a soft radial wash whose colour fades monotonically to a plateau.

        The 2-stop linear-in-radius model overshoots a heavy-blur blob's flat tail, so
        ``smooth_enough`` mis-flags a genuine radial as banded. A real blob's binned
        colour means are *monotonic* along the radius (each channel only rises or only
        falls, allowing a small wobble); a stepped flag reverses direction. This accepts
        the former and still rejects the latter, so an H8-style soft blob with no hard
        edge is emitted as a native radial instead of falling through to a raster/trace.
        """
        lo, hi = float(projection.min()), float(projection.max())
        if hi - lo <= 1e-6:
            return False
        edges = np.linspace(lo, hi, 13)
        idx = np.clip(np.digitize(projection, edges) - 1, 0, 11)
        means, populated = [], 0
        for b in range(12):
            sel = idx == b
            if int(sel.sum()) < 20:
                means.append(None)
                continue
            means.append(colors[sel].mean(axis=0))
            populated += 1
        seq = [m for m in means if m is not None]
        if populated < 6 or len(seq) < 6:
            return False
        wobble = float(gz.get("monotonic_wobble", 6.0))
        for ch in range(3):
            vals = [float(m[ch]) for m in seq]
            up = all(vals[i + 1] - vals[i] >= -wobble for i in range(len(vals) - 1))
            down = all(vals[i + 1] - vals[i] <= wobble for i in range(len(vals) - 1))
            if not (up or down):
                return False
        return True

    linear = None
    design = np.column_stack((np.ones(len(x), dtype=np.float32), x, y))
    coeff, _, _, _ = np.linalg.lstsq(design, colors, rcond=None)
    prediction = design @ coeff
    linear_r2 = 1.0 - float(np.square(colors - prediction).sum()) / total
    if linear_r2 >= min_r2:
        _, _, vh = np.linalg.svd(colors - colors.mean(axis=0), full_matrices=False)
        principal = vh[0]
        dx, dy = float(coeff[1] @ principal), float(coeff[2] @ principal)
        magnitude = (dx * dx + dy * dy) ** 0.5
        if magnitude >= 0.5:
            dx, dy = dx / magnitude, dy / magnitude
            projection = x * dx + y * dy
            low, high = np.percentile(projection, (2, 98))
            if high - low >= 0.25 and smooth_enough(projection, prediction):
                def endpoint(value):
                    return coeff[0] + coeff[1] * (value * dx) + coeff[2] * (value * dy)
                cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
                linear = {
                    "kind": "linear",
                    "angle": round(math.degrees(math.atan2(dy, dx)), 2),
                    "stops": [
                        {"position": 0, "color": _rgb_hex(endpoint(low))},
                        {"position": 1, "color": _rgb_hex(endpoint(high))},
                    ],
                    # Pixel-space gradient axis so the emitted SVG paint is exact.
                    "meta": {
                        "r2": round(linear_r2, 4), "range": round(range_norm, 2),
                        "x1": round(float(cx0 + low * dx * sx), 2),
                        "y1": round(float(cy0 + low * dy * sy), 2),
                        "x2": round(float(cx0 + high * dx * sx), 2),
                        "y2": round(float(cy0 + high * dy * sy), 2),
                    },
                }

    radial = None
    normalizer = max(1.0, float(math.hypot((w - 1) / 2.0, (h - 1) / 2.0)))
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0

    # A soft radial blob (H8 orange/yellow wash) is rarely centred on the crop. Search a
    # coarse grid of candidate centres (in half-extent units) and keep the one whose
    # radial regression best explains the colour field; fall back to the crop centre.
    # ``fit_center`` (default on) can be disabled for parity with the old fixed centre.
    fit_center = gz.get("fit_center", True) is not False

    def _radial_at(ox, oy):
        rad = (np.hypot((x - ox) * sx, (y - oy) * sy) / normalizer).astype(np.float32)
        dz = np.column_stack((np.ones(len(rad), dtype=np.float32), rad))
        cf, _, _, _ = np.linalg.lstsq(dz, colors, rcond=None)
        pr = dz @ cf
        r2 = 1.0 - float(np.square(colors - pr).sum()) / total
        return r2, (ox, oy, rad, cf, pr)

    best_r2, best = _radial_at(0.0, 0.0)
    if fit_center:
        # Coarse grid, then a finer local refinement around the winner so an off-centre
        # blob (H8) whose true centre lies between coarse nodes still fits tightly.
        for oy in (-0.6, -0.3, 0.0, 0.3, 0.6):
            for ox in (-0.6, -0.3, 0.0, 0.3, 0.6):
                r2, cand = _radial_at(ox, oy)
                if r2 > best_r2:
                    best_r2, best = r2, cand
        bx, by = best[0], best[1]
        for _ in range(2):
            step = 0.15 if _ == 0 else 0.07
            improved = False
            for oy in (by - step, by, by + step):
                for ox in (bx - step, bx, bx + step):
                    r2, cand = _radial_at(ox, oy)
                    if r2 > best_r2 + 1e-6:
                        best_r2, best, bx, by, improved = r2, cand, ox, oy, True
            if not improved:
                break
    ox, oy, radius, coeff_r, prediction_r = best
    radial_r2 = best_r2
    if (radial_r2 >= radial_min_r2
            and float(np.linalg.norm(coeff_r[1])) >= min_range * 0.55):
        low_r, high_r = np.percentile(radius, (2, 98))
        if high_r - low_r >= 0.35 and (
                smooth_enough(radius, prediction_r) or monotonic_enough(radius)):
            # Pixel-space centre + radius so the emitted SVG paint is exact even when the
            # blob is off-centre; normalized centre is kept for reconstruct parity. The
            # regression maps normalized radius 1.0 -> ``normalizer`` px from the fitted
            # centre, so the SVG gradient ``r`` MUST equal ``normalizer`` for the stop
            # positions to align with the fitted colours.
            cx_px = cx0 + ox * sx
            cy_px = cy0 + oy * sy
            r_px = normalizer
            radial = {
                "kind": "radial",
                "stops": [
                    {"position": 0, "color": _rgb_hex(coeff_r[0])},
                    {"position": 1, "color": _rgb_hex(coeff_r[0] + coeff_r[1])},
                ],
                "meta": {"r2": round(radial_r2, 4), "range": round(range_norm, 2),
                         "center": [round(0.5 + ox / 2.0, 4), round(0.5 + oy / 2.0, 4)],
                         "cx": round(cx_px, 2), "cy": round(cy_px, 2),
                         "r": round(max(1.0, r_px), 2)},
            }

    # Prefer radial only when it explains materially more variance (reconstruct parity).
    linear_r2_eff = float(((linear or {}).get("meta") or {}).get("r2", -1))
    radial_r2_eff = float(((radial or {}).get("meta") or {}).get("r2", -1))
    return radial if radial and radial_r2_eff >= linear_r2_eff + margin else linear


def _gradient_svg(d, w, h, grad, winding=None):
    """One silhouette path painted with the detected native gradient."""
    stops = "".join(
        f'<stop offset="{float(s.get("position", i)):g}" stop-color="{s["color"]}"/>'
        for i, s in enumerate(grad["stops"])
    )
    meta = grad.get("meta") or {}
    if grad["kind"] == "linear":
        defs = (
            f'<linearGradient id="vg" gradientUnits="userSpaceOnUse" '
            f'x1="{meta.get("x1", 0)}" y1="{meta.get("y1", 0)}" '
            f'x2="{meta.get("x2", w)}" y2="{meta.get("y2", 0)}">{stops}</linearGradient>'
        )
    else:
        cx = float(meta.get("cx", (w - 1) / 2.0))
        cy = float(meta.get("cy", (h - 1) / 2.0))
        radius = float(meta.get("r", math.hypot((w - 1) / 2.0, (h - 1) / 2.0)))
        defs = (
            f'<radialGradient id="vg" gradientUnits="userSpaceOnUse" '
            f'cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}">'
            f"{stops}</radialGradient>"
        )
    rule = ' fill-rule="evenodd"' if winding == "EVENODD" else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}"><defs>{defs}</defs><path d="{d}" fill="url(#vg)"{rule}/></svg>'
    )


def _trace_color_count(png_path):
    """Count quantized foreground colours, ignoring transparent matte RGB."""
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        mask, strategy = _foreground_mask(rgba)
        pixels = rgba[:, :, :3][mask]
        if not len(pixels):
            return 0, strategy
        quantized = (pixels.astype(np.uint16) // 16) * 16
        return int(len(np.unique(quantized.reshape(-1, 3), axis=0))), strategy
    except Exception:
        return 100000, "unknown"


def _run_vtracer(png_path, cfg, preset=None):
    exe = _resolve_binary(cfg, "color_engine", "vtracer")
    preset = preset or {}
    out_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False).name
    try:
        if exe:
            cmd = [
                exe, "--input", png_path, "--output", out_svg,
                "--mode", str(preset.get("mode", "spline")),
                "--colormode", str(preset.get("colormode", "color")),
                "--hierarchical", str(preset.get("hierarchical", "stacked")),
            ]
            if "filter_speckle" in preset:
                cmd.extend(["--filter_speckle", str(preset["filter_speckle"])])
            if "color_precision" in preset:
                cmd.extend(["--color_precision", str(preset["color_precision"])])
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        else:
            try:
                from vtracer import convert_image_to_svg_py
            except ImportError as e:
                raise RuntimeError("Python vtracer package not found") from e
            kwargs = {
                key: preset[key]
                for key in (
                    "colormode", "hierarchical", "mode", "filter_speckle",
                    "color_precision", "layer_difference", "corner_threshold",
                    "length_threshold", "max_iterations", "splice_threshold",
                    "path_precision",
                )
                if key in preset
            }
            convert_image_to_svg_py(png_path, out_svg, **kwargs)
    except Exception as e:
        try:
            os.unlink(out_svg)
        except OSError:
            pass
        backend = "vtracer" if exe else "Python vtracer"
        return None, f"{backend} failed: {e}"
    try:
        with open(out_svg, encoding="utf-8") as f:
            text = f.read()
        os.unlink(out_svg)
        return text, None
    except Exception as e:
        return None, f"vtracer output unreadable: {e}"


def _run_potrace(png_path, cfg, alpha_threshold=8, lum_threshold=128):
    exe = _resolve_binary(cfg, "binary_engine", "potrace")
    if not exe:
        return None, "potrace binary not found (choco/brew install potrace)"
    pbm = None
    out_svg = None
    try:
        try:
            import numpy as np
            from PIL import Image
            with Image.open(png_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                foreground, _ = _foreground_mask(rgba, alpha_threshold, lum_threshold)
                bitmap = Image.fromarray(np.where(foreground, 0, 255).astype(np.uint8))
            fd, pbm = tempfile.mkstemp(suffix=".pbm")
            os.close(fd)
            bitmap.convert("1").save(pbm)
        except Exception as e:
            return None, f"potrace preprocess failed: {e}"
        try:
            fd, out_svg = tempfile.mkstemp(suffix=".svg")
            os.close(fd)
            subprocess.run(
                [exe, "-s", "-o", out_svg, pbm], check=True, capture_output=True, timeout=120
            )
            with open(out_svg, encoding="utf-8") as f:
                return f.read(), None
        except Exception as e:
            return None, f"potrace failed: {e}"
    finally:
        for path in (pbm, out_svg):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _run_vtracer_binarized(png_path, cfg):
    """Silhouette trace without potrace: paint the foreground mask in its representative
    colour and run VTracer on the clean two-level image.  This is the documented
    degradation path when the configured ``binary_engine`` is not installed."""
    np = _require_np()
    tmp = None
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        mask, _ = _foreground_mask(rgba)
        if not mask.any():
            return None, "binarize: empty foreground"
        rgb = _hex_rgb(_opaque_fill(png_path)) or (0, 0, 0)
        flat = np.zeros_like(rgba)
        flat[mask] = (rgb[0], rgb[1], rgb[2], 255)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        Image.fromarray(flat).save(tmp)
        return _run_vtracer(tmp, cfg, {
            "mode": "spline", "colormode": "color",
            "hierarchical": "cutout", "filter_speckle": 2,
        })
    except Exception as e:
        return None, f"binarized vtracer failed: {e}"
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _contour_paths_to_svg(paths, w, h):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
    ]
    for item in paths:
        rule = ' fill-rule="evenodd"' if item.get("windingRule") == "EVENODD" else ""
        lines.append(f'<path d="{item["d"]}" fill="{item["fill"]}"{rule}/>')
    lines.append("</svg>")
    return "".join(lines)


def _normalize_trace_size(svg_text, traced_png, source_png):
    """Bake preprocessing resize into path coordinates and restore the source viewBox.

    Tiny crops are intentionally enlarged before tracing.  Returning those enlarged path
    coordinates made the editable icon overflow its real crop even though the trace gate
    passed.  The gate and exported paths must always live in original crop coordinates.
    """
    try:
        from PIL import Image
        with Image.open(traced_png) as traced, Image.open(source_png) as source:
            tw, th = traced.size
            sw, sh = source.size
        if (tw, th) == (sw, sh) or min(tw, th, sw, sh) <= 0:
            return svg_text
        matrix = (sw / float(tw), 0.0, 0.0, sh / float(th), 0.0, 0.0)
        paths = _parse_svg_paths(svg_text)
        scaled = [
            {**item, "d": _abs_path(item["d"], matrix), "fill": item.get("fill", "#000000")}
            for item in paths
        ]
        return _contour_paths_to_svg(scaled, sw, sh) if scaled else svg_text
    except Exception:
        return svg_text


def _run_contour_simplify(png_path, cfg):
    """OpenCV contour + approxPolyDP fallback for flat single-color icons."""
    if not _vz_cfg(cfg).get("contour_fallback", True):
        return None, "contour fallback disabled"
    np = _require_np()
    try:
        import cv2
    except ImportError:
        return None, "opencv not available (pip install opencv-python-headless)"
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        h, w = rgba.shape[:2]
        foreground, _ = _foreground_mask(rgba, alpha_threshold=32)
        mask = foreground.astype(np.uint8) * 255
        if not mask.any():
            return None, "contour: empty mask"
        fill = _opaque_fill(png_path)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "contour: no contours"
        epsilon_frac = float(_vz_cfg(cfg).get("contour_epsilon", 0.02))
        subpaths = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 2:
                continue
            eps = epsilon_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            d_parts = [f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"]
            for x, y in pts[1:]:
                d_parts.append(f"L{x:.1f} {y:.1f}")
            d_parts.append("Z")
            subpaths.append("".join(d_parts))
        if not subpaths:
            return None, "contour: no simplified paths"
        # One even-odd path keeps counters (camera lenses, ring logos, letters) as
        # transparent holes rather than filling them as RETR_EXTERNAL did.
        return _contour_paths_to_svg(
            [{"d": "".join(subpaths), "fill": fill, "windingRule": "EVENODD"}], w, h
        ), None
    except Exception as e:
        return None, f"contour failed: {e}"


def _opaque_fill(png_path):
    """Return the representative foreground fill for a monochrome Potrace result."""
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        foreground, _ = _foreground_mask(rgba)
        pixels = rgba[:, :, :3][foreground]
        if not len(pixels):
            return "#000000"
        # Ignore a near-white opaque background when an actual darker foreground exists.
        lum = pixels.astype(np.float32).mean(axis=1)
        foreground = pixels[lum < 245] if np.any(lum < 245) else pixels
        quant = (foreground.astype(np.uint16) // 8) * 8
        colors, counts = np.unique(quant.reshape(-1, 3), axis=0, return_counts=True)
        r, g, b = colors[int(counts.argmax())]
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    except Exception:
        return "#000000"


def _recolor_potrace_svg(svg_text, fill):
    """Potrace emits black paths; preserve the original icon colour before scoring/export."""
    def replace(match):
        attrs = match.group(2)
        if re.search(r"\bfill\s*=", attrs):
            attrs = re.sub(r'\bfill\s*=\s*(["\'])[^"\']*\1', f'fill="{fill}"', attrs)
        else:
            attrs += f' fill="{fill}"'
        return match.group(1) + attrs + match.group(3)

    return re.sub(r"(<path\b)([^>]*)(/?>)", replace, svg_text, flags=re.IGNORECASE)


# ── quality gate ─────────────────────────────────────────────────────────────────────
def _count_colors(png_path):
    return _trace_color_count(png_path)[0]


def _render_resvg_bytes(svg_text, width, height):
    """Render without ever exporting a native resvg/usvg tree from its owner thread.

    Prefer resvg_py's single-call API. A standalone CLI is the isolation fallback: it
    exchanges only files/bytes with a child process. The legacy Python Tree API is
    intentionally not called because its objects are not safely sendable.
    """
    try:
        import resvg_py
        data = resvg_py.svg_to_bytes(
            svg_string=svg_text,
            width=int(width),
            height=int(height),
        )
        return bytes(data), "resvg_py"
    except Exception as safe_error:
        executable = shutil.which("resvg")
        if not executable:
            return None, f"resvg_py unavailable/failed: {safe_error}"
        svg_path = tempfile.NamedTemporaryFile(suffix=".svg", delete=False).name
        png_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        try:
            with open(svg_path, "w", encoding="utf-8") as handle:
                handle.write(svg_text)
            subprocess.run(
                [executable, "--width", str(int(width)), "--height", str(int(height)), svg_path, png_path],
                check=True, capture_output=True, timeout=30,
            )
            with open(png_path, "rb") as handle:
                return handle.read(), "resvg-cli"
        except Exception as cli_error:
            return None, f"resvg_py failed: {safe_error}; resvg CLI failed: {cli_error}"
        finally:
            for path in (svg_path, png_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _rasterize_svg(svg_text, width, height):
    """Return an RGBA PIL rendering, or ``None`` when no safe renderer is available."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"), output_width=width, output_height=height
        )
        import io
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        try:
            png_bytes, _renderer = _render_resvg_bytes(svg_text, width, height)
            if not png_bytes:
                return None
            import io
            return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        except Exception:
            return None


def _enclosed_transparent_holes(alpha):
    """Find meaningful transparent counters, excluding the outer canvas background."""
    np = _require_np()
    try:
        import cv2
    except ImportError:  # pragma: no cover - contour fallback itself needs OpenCV
        return np.zeros_like(alpha, dtype=bool)
    background = (~np.asarray(alpha, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    holes = np.zeros_like(background, dtype=bool)
    height, width = background.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if x == 0 or y == 0 or x + w >= width or y + h >= height or area < 2:
            continue
        holes |= labels == label
    return holes


def _transparent_hole_recall(svg_text, png_path):
    """Return counter preservation for source icons that actually have counters."""
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as image:
            source = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    source_holes = _enclosed_transparent_holes(source[:, :, 3] > 8)
    source_count = int(source_holes.sum())
    if not source_count:
        return None
    rendered = _rasterize_svg(svg_text, source.shape[1], source.shape[0])
    if rendered is None:
        return {"source_hole_pixels": source_count, "trace_hole_pixels": 0, "recall": 0.0}
    traced_holes = _enclosed_transparent_holes(np.asarray(rendered, dtype=np.uint8)[:, :, 3] > 8)
    overlap = int(np.logical_and(source_holes, traced_holes).sum())
    return {
        "source_hole_pixels": source_count,
        "trace_hole_pixels": int(traced_holes.sum()),
        "recall": round(float(overlap / source_count), 4),
    }


def _score_render(svg_text, png_path):
    """Rasterize the trace and score both silhouette and colour fidelity."""
    np = _require_np()
    try:
        from PIL import Image
    except ImportError:
        return 0.0
    with Image.open(png_path) as im:
        w, h = im.size
        src = im.convert("RGBA")
    src_arr = np.asarray(src).astype(np.float32)
    src_a = src_arr[:, :, 3] > 8
    ras = _rasterize_svg(svg_text, w, h)
    if ras is None:
        return 0.0  # no rasterizer -> caller treats as ungated (score 0 -> not ok)
    ras_arr = np.asarray(ras.resize((w, h))).astype(np.float32)
    ras_a = ras_arr[:, :, 3] > 8
    union = np.logical_or(src_a, ras_a)
    alpha_iou = (float(np.logical_and(src_a, ras_a).sum()) / float(union.sum())
                 if union.any() else 1.0)
    shared = np.logical_and(src_a, ras_a)
    if shared.any():
        colour = 1.0 - float(np.abs(src_arr[:, :, :3][shared] - ras_arr[:, :, :3][shared]).mean()) / 255.0
    else:
        colour = 0.0
    return round(max(0.0, min(1.0, 0.7 * alpha_iou + 0.3 * colour)), 4)


def _gradient_silhouette_result(png_path, cfg, role, n_colors, grad):
    """Ship a detected gradient as ONE silhouette path + native gradient paint.

    The returned ``paths`` keep a flat hex fill (the existing downstream contract:
    preview tints the SVG alpha with the layer fill), while ``svg`` carries the true
    <linearGradient>/<radialGradient> for SVG-capable consumers (Figma import). Both
    representations must pass the same render-back gate; otherwise the crop falls
    through to the ordinary tracers and stays honest.
    """
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    h, w = rgba.shape[:2]
    alpha_mask = rgba[:, :, 3] > 8
    if not alpha_mask.any():
        return None
    pz = _vz_cfg(cfg).get("primitives") or {}
    min_iou = float(pz.get("min_iou", 0.94))
    candidates = []
    if bool(alpha_mask.all()):
        candidates.append((
            _rrect_d(0.0, 0.0, float(w), float(h), 0.0), None,
            {"kind": "rrect", "x": 0.0, "y": 0.0, "w": float(w), "h": float(h),
             "radius": 0.0, "iou": 1.0},
        ))
    else:
        prim = None
        if pz.get("enabled", True) is not False:
            prim = _fit_primitive(alpha_mask, min_iou, allow_plain_rect=True)
        if prim:
            candidates.append((_primitive_d(prim), None, prim))
        svg, _err = _run_contour_simplify(png_path, cfg)
        if svg:
            parsed = _parse_svg_paths(svg)
            if parsed:
                candidates.append((parsed[0]["d"], parsed[0].get("windingRule"), None))
    score_min, _ = _gate_limits(role, cfg)
    for d, winding, prim in candidates:
        svg_grad = _gradient_svg(d, w, h, grad, winding)
        result, _ = _evaluate_trace(
            svg_grad, png_path, "analytic-gradient", cfg, role, n_colors,
            preset_note=f"gradient={grad['kind']} r2={grad['meta'].get('r2')}",
        )
        if not result or not result["ok"]:
            continue
        mid = _mix_hex(grad["stops"][0]["color"], grad["stops"][-1]["color"])
        flat_paths = [dict(p, fill=mid) for p in result["paths"]]
        flat_score = _score_render(_contour_paths_to_svg(flat_paths, w, h), png_path)
        if flat_score < score_min:
            continue
        result["paths"] = flat_paths
        fill = dict(grad)
        fill["meta"] = dict(grad.get("meta") or {}, flat_score=flat_score,
                            silhouette="primitive" if prim else "contour")
        result["gradient_fill"] = fill
        if prim:
            result["primitive"] = prim
        result["note"] += f" flat_score={flat_score}"
        return result
    return None


def _flat_primitive_result(png_path, cfg, role, n_colors):
    """Prefer a clean analytic silhouette (circle / rounded-rect) for flat marks.

    Only fires for genuinely flat one-paint silhouettes and only when the ordinary
    render-back gate passes, so a textured or multi-colour icon still gets the faithful
    tracer treatment.  Full-bleed rectangles stay with the ordinary tracers.
    """
    pz = _vz_cfg(cfg).get("primitives") or {}
    if pz.get("enabled", True) is False:
        return None
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    h, w = rgba.shape[:2]
    mask, _strategy = _foreground_mask(rgba)
    if not mask.any() or bool(mask.all()):
        return None
    pixels = rgba[:, :, :3][mask].astype("float32")
    if pixels.shape[0] < 40:
        return None
    if float(pixels.std(axis=0).max()) > float(pz.get("max_color_std", 14.0)):
        return None
    prim = _fit_primitive(mask, float(pz.get("min_iou", 0.94)), allow_plain_rect=False)
    if not prim:
        return None
    fill = _rgb_hex(np.median(pixels, axis=0))
    svg = _contour_paths_to_svg([{"d": _primitive_d(prim), "fill": fill}], w, h)
    result, _ = _evaluate_trace(
        svg, png_path, "analytic-primitive", cfg, role, n_colors,
        preset_note=f"primitive={prim['kind']} iou={prim['iou']}",
    )
    if result and result["ok"]:
        result["primitive"] = prim
        return result
    return None


# Roles whose crop should be tried as a regular-star polygon before ordinary tracing.
_STAR_ROLES = frozenset({
    "starburst", "star", "sunburst", "seal", "star_badge", "star-badge", "badge_star",
})


def _star_primitive_result(png_path, cfg, role, n_colors):
    """Prefer a clean regular-star polygon for starburst / sunburst seal crops.

    Fires only for star-ish roles and only when the fitted star passes the same
    render-back gate as every trace; otherwise the caller falls through to VTracer so a
    genuinely irregular badge still reconstructs faithfully. The star is emitted as one
    flat-fill path (native star geometry) plus a ``primitive`` dict carrying point count
    and radii so a downstream Figma emitter can build a real STAR node.
    """
    pz = _vz_cfg(cfg).get("primitives") or {}
    if pz.get("enabled", True) is False:
        return None
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    h, w = rgba.shape[:2]
    mask, _strategy = _foreground_mask(rgba)
    if not mask.any() or bool(mask.all()):
        return None
    pixels = rgba[:, :, :3][mask].astype("float32")
    if pixels.shape[0] < 60:
        return None
    # A seal is a solid one-colour shape; a photographic/gradient badge is not a star fit.
    if float(pixels.std(axis=0).max()) > float(pz.get("star_max_color_std", 40.0)):
        return None
    prim = _fit_star_polygon(mask, float(pz.get("star_min_iou", 0.90)))
    if not prim:
        return None
    fill = _rgb_hex(np.median(pixels, axis=0))
    svg = _contour_paths_to_svg([{"d": _primitive_d(prim), "fill": fill}], w, h)
    result, _ = _evaluate_trace(
        svg, png_path, "analytic-star", cfg, role, n_colors,
        preset_note=f"star points={prim['points']} iou={prim['iou']}",
    )
    if result and result["ok"]:
        result["primitive"] = prim
        return result
    return None


_CLEANUP_ENGINES = ("vtracer", "potrace", "contour")


def _apply_cleanup(result, png_path, cfg, role, n_colors):
    """Post-trace path cleanup, arbitrated by the same render-back gate as the trace.

    Tiered: a full pass (with same-fill merging) first, then a conservative pass without
    merging; whichever first passes the gate wins.  A cleanup that fails the gate — or
    costs more than ``max_score_drop`` fidelity — is rolled back per-crop.  Can also
    rescue an over-budget trace whose only gate failure was path count.
    """
    if not isinstance(result, dict) or not result.get("paths"):
        return result
    if result.get("engine") not in _CLEANUP_ENGINES:
        return result
    cl = _vz_cfg(cfg).get("cleanup") or {}
    if cl.get("enabled", True) is False:
        return result
    try:
        from . import svg_cleanup
    except ImportError:
        try:
            import svg_cleanup  # script execution without package context
        except ImportError:
            return result
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            w, h = im.size
    except Exception:
        return result
    before_paths = len(result["paths"])
    before_points = svg_cleanup.count_points(result["paths"])
    min_area = float(cl.get("min_area", 2.0))
    tolerance = float(cl.get("simplify_tolerance", 0.6))
    fill_tol = int(cl.get("fill_tolerance", 10))
    max_drop = float(cl.get("max_score_drop", 0.03))
    merge_enabled = cl.get("merge_fills", True) is not False
    for merge in ((True, False) if merge_enabled else (False,)):
        try:
            cleaned = svg_cleanup.cleanup_paths(
                result["paths"], min_area=min_area, tolerance=tolerance,
                fill_tolerance=fill_tol, merge=merge,
            )
        except Exception:
            return result
        if not cleaned:
            continue
        after_paths = len(cleaned)
        after_points = svg_cleanup.count_points(cleaned)
        if after_paths >= before_paths and after_points >= before_points:
            return result  # nothing to gain; skip the extra rasterization
        svg = svg_cleanup.serialize_svg(cleaned, w, h)
        candidate, _ = _evaluate_trace(
            svg, png_path, result["engine"], cfg, role, n_colors, preset_note="cleanup",
        )
        if not candidate or not candidate["ok"]:
            continue
        if result.get("ok") and candidate["score"] < result["score"] - max_drop:
            continue
        candidate["cleanup"] = {"paths": [before_paths, after_paths],
                                "points": [before_points, after_points],
                                "merged": merge}
        candidate["note"] = (
            f"{result.get('note', '')} | cleanup paths {before_paths}->{after_paths} "
            f"points {before_points}->{after_points}"
        ).strip()
        return candidate
    return result


# ── public API ───────────────────────────────────────────────────────────────────────
def _evaluate_trace(svg, png_path, engine, cfg, role, n_colors, preset_note=""):
    score_min, max_paths = _gate_limits(role, cfg)
    paths = _parse_svg_paths(svg)
    if not paths:
        return None, f"{engine}: no paths parsed"
    score = _score_render(svg, png_path)
    note = f"paths={len(paths)} colors={n_colors}"
    if preset_note:
        note = f"{preset_note} {note}"
    hole = _transparent_hole_recall(svg, png_path)
    hole_min = float(_vz_cfg(cfg).get("hole_recall_min", 0.75))
    holes_ok = hole is None or hole["recall"] >= hole_min
    if hole is not None:
        note += f" hole_recall={hole['recall']:.3f}"
    return {
        "ok": score >= score_min and len(paths) <= max_paths and holes_ok,
        "paths": paths,
        "svg": svg,
        "engine": engine,
        "score": score,
        "note": note,
        "gate": {"score_min": score_min, "max_paths": max_paths,
                 "hole_recall_min": hole_min if hole is not None else None,
                 "hole_recall": hole},
    }, None


def vectorize_crop(png_path_or_array, cfg: Optional[dict] = None, role: Optional[str] = None):
    cfg = cfg or {}
    if _vz_cfg(cfg).get("force_raster_fallback"):
        return _fail("none", "force_raster_fallback")

    try:
        png_path, cleanup = _to_png_path(png_path_or_array)
    except ImportError as e:
        return _fail("none", str(e))
    except Exception as e:
        return _fail("none", f"bad input: {e}")

    pre_path, pre_cleanup = None, False
    try:
        pre_path, pre_cleanup = _preprocess_crop(png_path, cfg, role)
        work_path = pre_path
        n_colors = _count_colors(work_path)
        # Potrace is a silhouette tracer: use it first only for a genuinely
        # monochrome mark. Two-colour logos need VTracer to preserve their paint.
        prefer_potrace = n_colors <= 1

        best = None
        note = "no engine produced output"

        def consider(result):
            nonlocal best
            if result is None:
                return False
            if result["ok"]:
                best = result
                return True
            if best is None or result["score"] > best["score"]:
                best = result
            return False

        # Prefer compact, deterministic stroke geometry only when it independently
        # passes the same raster comparison as every traced vector. A detailed endpoint
        # or arrowhead therefore cannot be simplified away by this optimization.
        analytic_svg = _analytic_straight_line_svg(png_path, role)
        if analytic_svg:
            result, _ = _evaluate_trace(
                analytic_svg, png_path, "analytic-line", cfg, role, n_colors,
                preset_note="single-stroke",
            )
            if result and consider(result):
                return result

        # A flat shape with a simple gradient paint must become one silhouette + native
        # gradient, never ten stacked colour bands; a flat near-circle / rounded-rect
        # becomes the clean analytic primitive.  Both are gated exactly like traces.
        # A starburst / sunburst seal fits a real regular-star polygon before we ever
        # trace it — a clean N-point star beats a wobbly VTracer path. Only star-ish
        # roles pay the fitting cost, and the star is gated exactly like every trace.
        if str(role or "").lower().replace("-", "_") in _STAR_ROLES:
            result = _star_primitive_result(png_path, cfg, role, n_colors)
            if result is not None:
                return result

        # A flat shape with a simple gradient paint must become one silhouette + native
        # gradient, never ten stacked colour bands; a flat near-circle / rounded-rect
        # becomes the clean analytic primitive.  Both are gated exactly like traces.
        gradient_note = None
        grad = _detect_gradient(png_path, cfg)
        if grad:
            result = _gradient_silhouette_result(png_path, cfg, role, n_colors, grad)
            if result is not None:
                return result
            gradient_note = (f"gradient={grad['kind']} r2={grad['meta'].get('r2')} "
                             "detected but silhouette failed the gate")
        else:
            result = _flat_primitive_result(png_path, cfg, role, n_colors)
            if result is not None:
                return result

        def finalize(result):
            result = _apply_cleanup(result, png_path, cfg, role, n_colors)
            if gradient_note and isinstance(result, dict):
                result["note"] = f"{result.get('note', '')}; {gradient_note}".strip("; ")
            return result

        potrace_missing = _resolve_binary(cfg, "binary_engine", "potrace") is None

        def try_vtracer():
            nonlocal note
            for i, preset in enumerate(_vtracer_presets(cfg)):
                svg, err = _run_vtracer(work_path, cfg, preset)
                if not svg:
                    note = err
                    continue
                svg = _normalize_trace_size(svg, work_path, png_path)
                result, _ = _evaluate_trace(
                    svg, png_path, "vtracer", cfg, role, n_colors,
                    preset_note=f"preset={i}",
                )
                if result and consider(result):
                    return True
            return False

        def try_vtracer_binarized():
            # Single-shot potrace replacement: binarize the crop ourselves, trace it with
            # VTracer, and stamp one process-wide degradation note instead of repeating
            # a "potrace not found" warning on every monochrome crop.
            nonlocal note
            global _BINARY_DEGRADE_NOTED
            svg, err = _run_vtracer_binarized(work_path, cfg)
            if not svg:
                note = err or note
                return False
            svg = _normalize_trace_size(svg, work_path, png_path)
            result, fail = _evaluate_trace(
                svg, png_path, "vtracer", cfg, role, n_colors, preset_note="binarized",
            )
            if result is None:
                note = fail or note
                return False
            if not _BINARY_DEGRADE_NOTED:
                result["note"] += (" [binary_engine 'potrace' not installed; binarized "
                                   "crops degrade to vtracer -- choco install potrace]")
                _BINARY_DEGRADE_NOTED = True
            return consider(result)

        def try_potrace():
            nonlocal note
            if potrace_missing:
                return try_vtracer_binarized()
            for thr in _potrace_thresholds(cfg):
                svg, err = _run_potrace(work_path, cfg, alpha_threshold=thr, lum_threshold=thr)
                if not svg:
                    note = err
                    continue
                svg = _recolor_potrace_svg(svg, _opaque_fill(work_path))
                svg = _normalize_trace_size(svg, work_path, png_path)
                result, _ = _evaluate_trace(
                    svg, png_path, "potrace", cfg, role, n_colors,
                    preset_note=f"thr={thr}",
                )
                if result and consider(result):
                    return True
            return False

        if prefer_potrace:
            if try_potrace():
                return finalize(best)
            if try_vtracer():
                return finalize(best)
        else:
            if try_vtracer():
                return finalize(best)
            if try_potrace():
                return finalize(best)

        if n_colors <= 1:
            svg, err = _run_contour_simplify(work_path, cfg)
            if svg:
                svg = _normalize_trace_size(svg, work_path, png_path)
                result, _ = _evaluate_trace(svg, png_path, "contour", cfg, role, n_colors)
                if result and consider(result):
                    return finalize(result)
            else:
                note = err or note
        else:
            note = f"{note}; contour skipped for {n_colors}-colour crop"

        # finalize() may still rescue a trace whose only failure was path-count budget:
        # cleanup merges the banding down and re-runs the same gate.
        result = finalize(best or _fail("none", note))
        if not result.get("ok") and bool((cfg.get("runtime") or {}).get("require_active_models")):
            result["active_model_required"] = True
            result["note"] = f"active vector backend required: {result.get('note', note)}"
        return result
    finally:
        for path, do_cleanup in ((pre_path, pre_cleanup), (png_path, cleanup)):
            if do_cleanup and path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


if __name__ == "__main__":  # CPU-safe smoke: exercises parsing without a binary
    d_rel = "m10 10 h20 v20 h-20 z"
    print("abs:", _abs_path(d_rel))
    svg = '<svg><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000"/></svg>'
    print("parsed:", _parse_svg_paths(svg))
    # no vtracer/potrace on this box -> graceful failure dict
    try:
        import numpy as np
        arr = np.zeros((16, 16, 4), np.uint8)
        arr[4:12, 4:12] = [255, 0, 0, 255]
        r = vectorize_crop(arr, {})
        print("vectorize_crop ->", {k: r[k] for k in ("ok", "engine", "score", "note")})
    except ImportError as e:
        print("numpy missing:", e)
