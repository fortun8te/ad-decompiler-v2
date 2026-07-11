"""vectorize.py — stage: crop -> SVG paths for icons / simple graphics / masks.

vectorize_crop(png_path_or_array, cfg) traces a small raster crop into absolute
M/L/C/Z path d-strings with fills:

  * VTracer (color, stacked mode) is primary — the `vtracer` binary via subprocess.
  * Potrace (binary/monochrome) handles 1-color icons + masks.

Output: {'ok', 'paths':[{'d','fill'}], 'engine', 'score'}. A quality gate rasterizes
the traced result and compares alpha to the source; ok=False when score < 0.90 or
paths > 40 (too complex to be a clean vector — caller keeps the raster crop instead).

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


# ── engines ──────────────────────────────────────────────────────────────────────────
def _run_vtracer(png_path, cfg):
    binary = (cfg.get("vectorize") or {}).get("color_engine", "vtracer")
    exe = shutil.which(binary) or (binary if os.path.exists(binary) else None)
    if not exe:
        return None, "vtracer binary not found (cargo install vtracer)"
    out_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False).name
    cmd = [
        exe, "--input", png_path, "--output", out_svg,
        "--mode", "spline", "--colormode", "color", "--hierarchical", "stacked",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        return None, f"vtracer failed: {e}"
    try:
        with open(out_svg, encoding="utf-8") as f:
            return f.read(), None
    except Exception as e:
        return None, f"vtracer output unreadable: {e}"


def _run_potrace(png_path, cfg):
    binary = (cfg.get("vectorize") or {}).get("binary_engine", "potrace")
    exe = shutil.which(binary) or (binary if os.path.exists(binary) else None)
    if not exe:
        return None, "potrace binary not found (choco/brew install potrace)"
    # Potrace needs a bitmap (PBM/PGM).  For a transparent icon, alpha is the silhouette:
    # converting RGBA directly to L makes transparent white pixels become foreground and traces
    # the entire crop.  Keep the old luminance route only for genuinely opaque source images.
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
                    # PBM convention: black (0) is foreground, white (255) background.
                    bitmap = Image.fromarray(np.where(alpha > 8, 0, 255).astype(np.uint8))
                else:
                    bitmap = image.convert("L").point(lambda p: 255 if p > 128 else 0)
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
        # One finally around both stages guarantees the .pbm is removed even when the
        # bitmap save (first stage) throws after mkstemp already created the file on disk.
        for path in (pbm, out_svg):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


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
def vectorize_crop(png_path_or_array, cfg: Optional[dict] = None):
    cfg = cfg or {}
    try:
        png_path, cleanup = _to_png_path(png_path_or_array)
    except ImportError as e:
        return _fail("none", str(e))
    except Exception as e:
        return _fail("none", f"bad input: {e}")

    try:
        n_colors = _count_colors(png_path)
        prefer_potrace = n_colors <= 2
        order = (["potrace", "vtracer"] if prefer_potrace else ["vtracer", "potrace"])

        best = None
        for eng in order:
            svg, err = (_run_vtracer if eng == "vtracer" else _run_potrace)(png_path, cfg)
            if not svg:
                note = err
                continue
            if eng == "potrace":
                svg = _recolor_potrace_svg(svg, _opaque_fill(png_path))
            paths = _parse_svg_paths(svg)
            if not paths:
                note = f"{eng}: no paths parsed"
                continue
            score = _score_render(svg, png_path)
            result = {
                "ok": score >= 0.90 and len(paths) <= 40,
                "paths": paths,
                "svg": svg,
                "engine": eng,
                "score": score,
                "note": f"paths={len(paths)} colors={n_colors}",
            }
            if result["ok"]:
                return result
            # remember the best-scoring attempt even if it fails the gate
            if best is None or score > best["score"]:
                best = result
        return best or _fail(order[0], locals().get("note", "no engine produced output"))
    finally:
        if cleanup:
            try:
                os.unlink(png_path)
            except OSError:
                pass


if __name__ == "__main__":  # CPU-safe smoke: exercises parsing without a binary
    d_rel = "m10 10 h20 v20 h-20 z"
    print("abs:", _abs_path(d_rel))
    svg = '<svg><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000"/></svg>'
    print("parsed:", _parse_svg_paths(svg))
    import numpy as _np  # noqa
    # no vtracer/potrace on this box -> graceful failure dict
    try:
        import numpy as np
        arr = np.zeros((16, 16, 4), np.uint8)
        arr[4:12, 4:12] = [255, 0, 0, 255]
        r = vectorize_crop(arr, {})
        print("vectorize_crop ->", {k: r[k] for k in ("ok", "engine", "score", "note")})
    except ImportError as e:
        print("numpy missing:", e)
