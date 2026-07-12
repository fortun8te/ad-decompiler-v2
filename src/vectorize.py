"""vectorize.py — stage: crop -> SVG paths for icons / simple graphics / masks.

vectorize_crop(png_path_or_array, cfg, role=None) traces a small raster crop into absolute
M/L/C/Z path d-strings with fills:

  * VTracer (multiple presets: color/cutout/binary, varying filter_speckle) is primary.
  * Potrace (binary/monochrome, multiple alpha thresholds) handles 1-color icons + masks.
  * OpenCV contour simplify is a last-resort fallback for flat single-color icons.

Output: {'ok', 'paths':[{'d','fill'}], 'engine', 'score', 'gate'}. Role-based quality gate
rasterizes the traced result and compares alpha to the source; ok=False when score or path
count exceeds role-specific limits (caller keeps the raster crop instead).

NEVER throws: on a missing binary / trace failure it returns ok=False with a note.
Binaries: vtracer (`cargo install vtracer` or download release), potrace (`choco
install potrace` / `brew install potrace`).
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional


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
            if fill.lower() in ("none",):
                continue
            try:
                d_abs = _abs_path(dm.group(1), stack[-1])
            except Exception:
                d_abs = dm.group(1)
            if d_abs:
                paths.append({"d": d_abs, "fill": fill})
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
}
_DEFAULT_MAX_PATHS = {
    "default": 40, "icon": 60, "logo": 50, "badge": 35,
    "arrow": 30, "mask": 20, "chip": 35, "shape": 45,
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
    binary = _vz_cfg(cfg).get(key, default)
    return shutil.which(binary) or (binary if os.path.exists(binary) else None)


def check_binaries(cfg=None):
    """Report vtracer/potrace/cairosvg availability for doctor / health probes."""
    cfg = cfg or {}
    vtracer = _resolve_binary(cfg, "color_engine", "vtracer")
    potrace = _resolve_binary(cfg, "binary_engine", "potrace")
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
    return {
        "vtracer": {"ok": bool(vtracer), "path": vtracer or "not on PATH (cargo install vtracer)"},
        "potrace": {"ok": bool(potrace), "path": potrace or "not on PATH (brew/choco install potrace)"},
        "cairosvg": {"ok": gate, "path": "installed" if gate else "pip install cairosvg (quality gate)"},
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

        # Upscale tiny icons so tracers have enough pixels to work with.
        min_dim = min(w, h)
        upscale_below = int(pre.get("upscale_below", 48))
        upscale_target = int(pre.get("upscale_target", 96))
        if min_dim < upscale_below and min_dim > 0:
            scale = upscale_target / float(min_dim)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            rgba = rgba.resize((nw, nh), Image.Resampling.NEAREST)
            arr = np.array(rgba, dtype=np.uint8, copy=True)
            h, w = arr.shape[:2]

        # Remove anti-alias fringe: tighten alpha then restore RGB at hard edge.
        if pre.get("denoise_fringe", True):
            alpha = arr[:, :, 3].astype(np.float32)
            hard = np.where(alpha >= 200, 255, np.where(alpha <= 40, 0, alpha)).astype(np.uint8)
            arr[:, :, 3] = hard

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

        if np.array_equal(arr, original_arr):
            return png_path, False

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr, mode="RGBA").save(tmp.name)
        return tmp.name, True


# ── engines ──────────────────────────────────────────────────────────────────────────
def _run_vtracer(png_path, cfg, preset=None):
    exe = _resolve_binary(cfg, "color_engine", "vtracer")
    if not exe:
        return None, "vtracer binary not found (cargo install vtracer)"
    preset = preset or {}
    out_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False).name
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
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        try:
            os.unlink(out_svg)
        except OSError:
            pass
        return None, f"vtracer failed: {e}"
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
                has_alpha = "A" in image.getbands() or (
                    image.mode == "P" and "transparency" in image.info
                )
                if has_alpha:
                    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3]
                    bitmap = Image.fromarray(
                        np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
                    )
                else:
                    bitmap = image.convert("L").point(
                        lambda p, t=lum_threshold: 255 if p > t else 0
                    )
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


def _contour_paths_to_svg(paths, w, h):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
    ]
    for item in paths:
        lines.append(f'<path d="{item["d"]}" fill="{item["fill"]}"/>')
    lines.append("</svg>")
    return "".join(lines)


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
        alpha = rgba[:, :, 3]
        mask = (alpha > 32).astype(np.uint8) * 255
        if not mask.any():
            return None, "contour: empty mask"
        fill = _opaque_fill(png_path)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "contour: no contours"
        epsilon_frac = float(_vz_cfg(cfg).get("contour_epsilon", 0.02))
        paths = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 4:
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
            paths.append({"d": "".join(d_parts), "fill": fill})
        if not paths:
            return None, "contour: no simplified paths"
        return _contour_paths_to_svg(paths, w, h), None
    except Exception as e:
        return None, f"contour failed: {e}"


def _opaque_fill(png_path):
    """Return the representative foreground fill for a monochrome Potrace result."""
    np = _require_np()
    try:
        from PIL import Image
        with Image.open(png_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            has_alpha = "A" in image.getbands() or (
                image.mode == "P" and "transparency" in image.info
            )
        pixels = rgba[:, :, :3][rgba[:, :, 3] > 8] if has_alpha else rgba[:, :, :3].reshape(-1, 3)
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
    try:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        colors = im.getcolors(maxcolors=100000)
        return len(colors) if colors else 100000
    except Exception:
        return 100000


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
    ras = None
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"), output_width=w, output_height=h
        )
        import io
        ras = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return 0.0  # cairosvg missing -> caller treats as ungated (score 0 -> not ok)
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
    return {
        "ok": score >= score_min and len(paths) <= max_paths,
        "paths": paths,
        "svg": svg,
        "engine": engine,
        "score": score,
        "note": note,
        "gate": {"score_min": score_min, "max_paths": max_paths},
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
        prefer_potrace = n_colors <= 2

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

        def try_vtracer():
            nonlocal note
            for i, preset in enumerate(_vtracer_presets(cfg)):
                svg, err = _run_vtracer(work_path, cfg, preset)
                if not svg:
                    note = err
                    continue
                result, _ = _evaluate_trace(
                    svg, work_path, "vtracer", cfg, role, n_colors,
                    preset_note=f"preset={i}",
                )
                if result and consider(result):
                    return True
            return False

        def try_potrace():
            nonlocal note
            for thr in _potrace_thresholds(cfg):
                svg, err = _run_potrace(work_path, cfg, alpha_threshold=thr, lum_threshold=thr)
                if not svg:
                    note = err
                    continue
                svg = _recolor_potrace_svg(svg, _opaque_fill(work_path))
                result, _ = _evaluate_trace(
                    svg, work_path, "potrace", cfg, role, n_colors,
                    preset_note=f"thr={thr}",
                )
                if result and consider(result):
                    return True
            return False

        if prefer_potrace:
            if try_potrace():
                return best
            if try_vtracer():
                return best
        else:
            if try_vtracer():
                return best
            if try_potrace():
                return best

        svg, err = _run_contour_simplify(work_path, cfg)
        if svg:
            result, _ = _evaluate_trace(svg, work_path, "contour", cfg, role, n_colors)
            if result and consider(result):
                return result
        else:
            note = err or note

        return best or _fail("none", note)
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
