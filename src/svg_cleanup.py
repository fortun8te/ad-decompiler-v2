"""svg_cleanup.py — post-trace SVG path cleanup for Figma-editable vectors.

Traced icon SVGs (VTracer especially) arrive as "traced spaghetti": many stacked
colour-banded paths with redundant points, sub-pixel speckle paths, and near-identical
fills split across siblings. This module reduces them to something a designer can edit:

  * drop sub-pixel noise paths / subpaths (tiny absolute area),
  * quantize near-identical fill colours (area-weighted representative wins),
  * merge consecutive same-paint paths into one multi-subpath path,
  * reduce point count (straight-cubic demotion + Douglas-Peucker within a px tolerance).

It only understands the absolute M/L/C/Z d-strings that vectorize.py emits; anything it
cannot parse (or that carries a stroke) passes through byte-identical in its original
z-order. This module never decides fidelity: vectorize._apply_cleanup re-runs the normal
render-back gate on the cleaned SVG and rolls the whole cleanup back per-crop on failure.
"""
from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# ── d-string parsing / serialization (absolute M/L/C/Z only) ─────────────────────────


def parse_d(d):
    """Parse an absolute M/L/C/Z d-string into subpaths.

    Returns [{"start": (x, y), "segs": [("L", x, y) | ("C", x1, y1, x2, y2, x, y)],
    "closed": bool}].  Raises ValueError for relative/unsupported commands so callers
    can pass the original path through untouched.
    """
    tokens = _TOKEN.findall(d or "")
    subs = []
    cur = None
    last_cmd = None
    pos = 0
    n = len(tokens)

    def take():
        nonlocal pos
        if pos >= n:
            raise ValueError("truncated path data")
        value = float(tokens[pos])  # raises ValueError on a command letter
        pos += 1
        return value

    while pos < n:
        tok = tokens[pos]
        if tok.isalpha():
            if tok not in ("M", "L", "C", "Z"):
                raise ValueError(f"unsupported path command {tok!r}")
            cmd = tok
            pos += 1
        elif last_cmd in ("M", "L"):
            cmd = "L"  # implicit lineto per SVG spec
        elif last_cmd == "C":
            cmd = "C"
        else:
            raise ValueError("path data before any command")
        if cmd == "M":
            cur = {"start": (take(), take()), "segs": [], "closed": False}
            subs.append(cur)
        elif cmd == "Z":
            if cur is not None:
                cur["closed"] = True
            cur = None
        elif cur is None:
            raise ValueError(f"{cmd} command before M")
        elif cmd == "L":
            cur["segs"].append(("L", take(), take()))
        else:  # C
            cur["segs"].append(("C", take(), take(), take(), take(), take(), take()))
        last_cmd = cmd
    return subs


def _serialize_subpath(sub):
    x, y = sub["start"]
    parts = [f"M{x:.2f} {y:.2f}"]
    for seg in sub["segs"]:
        if seg[0] == "L":
            parts.append(f"L{seg[1]:.2f} {seg[2]:.2f}")
        else:
            parts.append("C" + " ".join(f"{v:.2f}" for v in seg[1:]))
    if sub["closed"]:
        parts.append("Z")
    return "".join(parts)


def serialize_subpaths(subs):
    return "".join(_serialize_subpath(sub) for sub in subs)


def serialize_svg(paths, width, height):
    """Rebuild an SVG document (fill/fill-rule/stroke aware) from [{d, fill, ...}]."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    for item in paths:
        rule = ' fill-rule="evenodd"' if item.get("windingRule") == "EVENODD" else ""
        stroke = ""
        spec = item.get("stroke")
        if isinstance(spec, dict) and spec.get("color"):
            stroke = f' stroke="{spec["color"]}"'
            if spec.get("width") is not None:
                stroke += f' stroke-width="{spec["width"]}"'
            if spec.get("cap"):
                stroke += f' stroke-linecap="{str(spec["cap"]).lower()}"'
            if spec.get("opacity") is not None:
                stroke += f' stroke-opacity="{spec["opacity"]}"'
        parts.append(
            f'<path d="{item["d"]}" fill="{item.get("fill", "#000000")}"{rule}{stroke}/>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ── geometry helpers ─────────────────────────────────────────────────────────────────


def _flatten_subpath(sub, samples=6):
    pts = [sub["start"]]
    x0, y0 = sub["start"]
    for seg in sub["segs"]:
        if seg[0] == "L":
            pts.append((seg[1], seg[2]))
            x0, y0 = seg[1], seg[2]
        else:
            x1, y1, x2, y2, x3, y3 = seg[1:]
            for k in range(1, samples + 1):
                t = k / samples
                mt = 1.0 - t
                pts.append((
                    mt ** 3 * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t ** 3 * x3,
                    mt ** 3 * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t ** 3 * y3,
                ))
            x0, y0 = x3, y3
    return pts


def _signed_area(points):
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _subpath_area(sub):
    return _signed_area(_flatten_subpath(sub))


def _paint_area(subs):
    """Approximate painted area: |sum of signed areas| keeps counter-holes negative."""
    return abs(sum(_subpath_area(sub) for sub in subs))


def _point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rdp(points, tol):
    """Douglas-Peucker; keeps first/last points, iterative to avoid recursion limits."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        dist_max, index = 0.0, -1
        for k in range(a + 1, b):
            dd = _point_segment_distance(points[k], points[a], points[b])
            if dd > dist_max:
                dist_max, index = dd, k
        if dist_max > tol and index > 0:
            keep[index] = True
            stack.append((a, index))
            stack.append((index, b))
    return [p for p, kept in zip(points, keep) if kept]


def _simplify_subpath(sub, tol):
    """Demote straight cubics to lines, drop zero-length segments, RDP the line runs."""
    segs = []
    prev = sub["start"]
    for seg in sub["segs"]:
        if seg[0] == "C":
            end = (seg[5], seg[6])
            c1, c2 = (seg[1], seg[2]), (seg[3], seg[4])
            # The curve lies inside the convex hull of its control points: if both
            # controls are within tol of the chord, so is every curve point.
            if (_point_segment_distance(c1, prev, end) <= tol
                    and _point_segment_distance(c2, prev, end) <= tol):
                seg = ("L", end[0], end[1])
        if seg[0] == "L" and math.hypot(seg[1] - prev[0], seg[2] - prev[1]) < 0.05:
            continue  # zero-length
        segs.append(seg)
        prev = (seg[-2], seg[-1])
    out = []
    anchor = sub["start"]
    i = 0
    while i < len(segs):
        if segs[i][0] != "L":
            out.append(segs[i])
            anchor = (segs[i][-2], segs[i][-1])
            i += 1
            continue
        run = [anchor]
        j = i
        while j < len(segs) and segs[j][0] == "L":
            run.append((segs[j][1], segs[j][2]))
            j += 1
        for pt in _rdp(run, tol)[1:]:
            out.append(("L", pt[0], pt[1]))
        anchor = run[-1]
        i = j
    # A closed subpath does not need an explicit return-to-start line.
    if (sub["closed"] and out and out[-1][0] == "L"
            and math.hypot(out[-1][1] - sub["start"][0],
                           out[-1][2] - sub["start"][1]) < 0.05):
        out.pop()
    return {"start": sub["start"], "segs": out, "closed": sub["closed"]}


# ── fill colour helpers ──────────────────────────────────────────────────────────────


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


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _quantize_fills(items, tol):
    """Snap near-identical hex fills to an area-weighted representative colour."""
    weights = {}
    for it in items:
        rgb = it.get("rgb")
        if rgb is None:
            continue
        weights[rgb] = weights.get(rgb, 0.0) + max(1.0, it["area"])
    if not weights:
        return
    reps = []
    mapping = {}
    for rgb in sorted(weights, key=lambda c: (-weights[c], c)):
        target = None
        for rep in reps:
            if max(abs(a - b) for a, b in zip(rgb, rep)) <= tol:
                target = rep
                break
        if target is None:
            reps.append(rgb)
            target = rgb
        mapping[rgb] = target
    for it in items:
        rgb = it.get("rgb")
        if rgb is not None:
            it["path"]["fill"] = _rgb_to_hex(mapping[rgb])


# ── public API ───────────────────────────────────────────────────────────────────────


def count_points(paths):
    """Anchor+control point count across [{d}] entries (parse-tolerant)."""
    total = 0
    for path in paths or []:
        d = (path or {}).get("d") or ""
        try:
            subs = parse_d(d)
        except (ValueError, IndexError):
            total += len(_NUM.findall(d)) // 2
            continue
        for sub in subs:
            total += 1
            for seg in sub["segs"]:
                total += 1 if seg[0] == "L" else 3
    return total


def cleanup_paths(paths, min_area=2.0, tolerance=0.6, fill_tolerance=10, merge=True):
    """Return a NEW cleaned [{d, fill, ...}] list (never mutates its inputs).

    Unparsable or stroked entries pass through untouched in their original z-order, and
    consecutive-only merging preserves paint order for everything else. If every path
    would be dropped as noise the original list is returned unchanged — the caller's
    render-back gate stays the sole fidelity arbiter.
    """
    items = []
    for path in paths or []:
        entry = dict(path)
        subs = None
        if not entry.get("stroke"):
            try:
                parsed = parse_d(entry.get("d") or "")
                subs = parsed if parsed else None
            except (ValueError, IndexError):
                subs = None
        area = _paint_area(subs) if subs else 0.0
        items.append({
            "path": entry,
            "subs": subs,
            "area": area,
            "rgb": _hex_rgb(entry.get("fill")) if subs else None,
        })
    if not any(it["subs"] is not None for it in items):
        return [it["path"] for it in items]

    # 1) drop sub-pixel noise: whole paths and individual tiny subpaths.
    if min_area > 0:
        survivors = []
        for it in items:
            if it["subs"] is None:
                survivors.append(it)
                continue
            subs = [s for s in it["subs"] if abs(_subpath_area(s)) >= min_area]
            if not subs:
                continue
            it["subs"] = subs
            it["area"] = _paint_area(subs)
            survivors.append(it)
        if not any(it["subs"] is not None for it in survivors):
            # The whole shape is below the noise floor: nothing meaningful to clean.
            return [it["path"] for it in items]
        items = survivors

    # 2) quantize near-identical fills so banding collapses into mergeable runs.
    if fill_tolerance and fill_tolerance > 0:
        _quantize_fills(items, int(fill_tolerance))

    # 3) merge consecutive same-paint paths (paint order preserved by adjacency).
    if merge:
        merged = []
        for it in items:
            prev = merged[-1] if merged else None
            if (prev is not None and prev["subs"] is not None and it["subs"] is not None
                    and not prev["path"].get("stroke") and not it["path"].get("stroke")
                    and prev["path"].get("fill") == it["path"].get("fill")
                    and (prev["path"].get("windingRule") or "NONZERO")
                    == (it["path"].get("windingRule") or "NONZERO")):
                prev["subs"] = prev["subs"] + it["subs"]
                prev["area"] += it["area"]
                continue
            merged.append(it)
        items = merged

    # 4) point reduction within the pixel tolerance.
    out = []
    for it in items:
        if it["subs"] is None:
            out.append(it["path"])
            continue
        subs = it["subs"]
        if tolerance and tolerance > 0:
            subs = [_simplify_subpath(s, float(tolerance)) for s in subs]
        entry = dict(it["path"])
        entry["d"] = serialize_subpaths(subs)
        out.append(entry)
    return out
