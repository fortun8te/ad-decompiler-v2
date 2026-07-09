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


def _abs_path(d):
    """Convert a possibly-relative SVG d-string to absolute M/L/C/Z only.
    S/Q/T/A/H/V are expanded to their absolute L/C forms where reasonable."""
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
            out.append(f"M{x:.2f} {y:.2f}")
            cur = "l" if rel else "L"  # subsequent implicit pairs = lineto
            prev_ctrl = None
        elif C == "L":
            x, y = nums(2)
            if rel:
                x += cx; y += cy
            cx, cy = x, y
            out.append(f"L{x:.2f} {y:.2f}")
            prev_ctrl = None
        elif C == "H":
            (x,) = nums(1)
            if rel:
                x += cx
            cx = x
            out.append(f"L{x:.2f} {cy:.2f}")
            prev_ctrl = None
        elif C == "V":
            (y,) = nums(1)
            if rel:
                y += cy
            cy = y
            out.append(f"L{cx:.2f} {y:.2f}")
            prev_ctrl = None
        elif C == "C":
            x1, y1, x2, y2, x, y = nums(6)
            if rel:
                x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
            out.append(f"C{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x:.2f} {y:.2f}")
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
            out.append(f"C{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x:.2f} {y:.2f}")
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
            out.append(f"C{c1x:.2f} {c1y:.2f} {c2x:.2f} {c2y:.2f} {x:.2f} {y:.2f}")
            prev_ctrl = (x1, y1)
            cx, cy = x, y
        elif C == "Z":
            out.append("Z")
            cx, cy = sx, sy
            prev_ctrl = None
        else:
            # unsupported (A arcs) -> skip the numbers to avoid infinite loop
            nums(7)
            prev_ctrl = None
    return "".join(out)


def _parse_svg_paths(svg_text):
    """Extract [{d, fill}] from an SVG string, normalizing d to absolute."""
    paths = []
    for m in re.finditer(r"<path\b([^>]*)/?>", svg_text, re.DOTALL):
        attrs = m.group(1)
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
            d_abs = _abs_path(dm.group(1))
        except Exception:
            d_abs = dm.group(1)
        if d_abs:
            paths.append({"d": d_abs, "fill": fill})
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
    # potrace needs a bitmap (PBM/PGM). Convert via PIL.
    try:
        from PIL import Image
        pbm = tempfile.NamedTemporaryFile(suffix=".pbm", delete=False).name
        Image.open(png_path).convert("L").point(lambda p: 255 if p > 128 else 0).convert(
            "1"
        ).save(pbm)
    except Exception as e:
        return None, f"potrace preprocess failed: {e}"
    out_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False).name
    try:
        subprocess.run(
            [exe, "-s", "-o", out_svg, pbm], check=True, capture_output=True, timeout=120
        )
        with open(out_svg, encoding="utf-8") as f:
            return f.read(), None
    except Exception as e:
        return None, f"potrace failed: {e}"


# ── quality gate ─────────────────────────────────────────────────────────────────────
def _count_colors(png_path):
    try:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        colors = im.getcolors(maxcolors=100000)
        return len(colors) if colors else 100000
    except Exception:
        return 100000


def _score_alpha(svg_text, png_path):
    """Rasterize SVG and compare alpha coverage to source. 1 - meanAbsDiff/255."""
    np = _require_np()
    try:
        from PIL import Image
    except ImportError:
        return 0.0
    with Image.open(png_path) as im:
        w, h = im.size
        src = im.convert("RGBA")
    src_a = (np.asarray(src)[:, :, 3] > 8).astype(np.float32) * 255.0
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
    ras_a = (np.asarray(ras.resize((w, h)))[:, :, 3] > 8).astype(np.float32) * 255.0
    mad = float(np.abs(src_a - ras_a).mean())
    return round(1.0 - mad / 255.0, 4)


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
            paths = _parse_svg_paths(svg)
            if not paths:
                note = f"{eng}: no paths parsed"
                continue
            score = _score_alpha(svg, png_path)
            result = {
                "ok": score >= 0.90 and len(paths) <= 40,
                "paths": paths,
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
