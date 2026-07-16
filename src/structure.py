"""structure.py — semantic shaping of the layer tree a designer opens in Figma.

The pipeline's deterministic layout pass is good at *geometry* (what contains
what) and bad at *legibility* (what a human calls a group).  Benchmark-6 shows
both failure modes in the emitted design.json:

  * 009 — eighteen text nodes dumped flat into one 'Screenshot' group: no header,
    no name row, no meta row.  Swapping the copy of a batch job means hunting
    through a flat list.
  * 021 — 'Header' and 'Header / 2' are groups with ZERO children, and a 'Photo'
    group wraps a single 'Photo' image of the identical box.
  * 002 — the root group is literally named 'Group'.
  * 107 — sixteen roots, WEEK 1..5 and every body line top-level siblings.

This module owns the *policy* that fixes those, as pure functions over a plain
node contract so it can be unit-tested without running a pipeline:

    node = {"id": str, "target": "text"|"image"|"group"|..., "box": {x,y,w,h},
            "children": [node, ...], "meta": {...}, "name": str|None}

Boxes are ABSOLUTE while this module runs (layout re-relativizes afterwards).
Nothing here reads the disk, calls a VLM, or mutates its input: every entry point
returns a new forest.  ``restructure`` composes the passes in dependency order.
"""

from __future__ import annotations

import copy
from statistics import median
from typing import Callable, Iterable, Optional

# A band is only worth creating when the whitespace above it is decisively larger
# than the rhythm of the copy inside it. Tuned against the 009/107 geometries: a
# 1.6x-median-gap seam splits header/body/meta without shattering a paragraph
# into one band per line.
DEFAULTS = {
    "enabled": True,
    "prune_empty_groups": True,
    "collapse_redundant_wrappers": True,
    "band_split": True,
    "band_min_children": 6,      # below this a flat group is already readable
    "band_seam_gap_factor": 1.6,  # seam if gap >= factor * median inter-node gap
    "band_seam_min_px": 24.0,     # never treat sub-pixel rhythm as a seam
    "band_min_members": 1,
    "band_min_bands": 2,          # a single band is just the old group renamed
    "text_above_rasters": True,
    "max_band_depth": 1,
}

_TEXTISH = {"text"}
# NOTE: there is deliberately no raster/shape/group class here. ``order_children``
# splits siblings into text vs everything-else and nothing finer; sub-classifying art
# re-sorts it against itself and hides overlapping cutouts. See order_children.
# Names that carry no information for a designer scanning the layer list.
_JUNK_NAMES = {"", "group", "layer", "frame", "rect", "rectangle", "vector",
               "node", "element", "shape", "untitled"}


def options(cfg: Optional[dict] = None) -> dict:
    """Merge caller config over DEFAULTS (``cfg['structure']`` wins)."""
    merged = dict(DEFAULTS)
    for key, value in ((cfg or {}).get("structure") or {}).items():
        if key in merged and value is not None:
            merged[key] = value
    return merged


def enabled(cfg: Optional[dict] = None) -> bool:
    return bool(options(cfg)["enabled"])


# ── small geometry helpers ───────────────────────────────────────────────────────

def _box(node: dict) -> dict:
    raw = node.get("box") or {}
    return {k: float(raw.get(k) or 0.0) for k in ("x", "y", "w", "h")}


def _union(nodes: Iterable[dict]) -> dict:
    boxes = [_box(n) for n in nodes]
    boxes = [b for b in boxes if b["w"] > 0 and b["h"] > 0]
    if not boxes:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _same_box(a: dict, b: dict, tol: float = 2.0) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("x", "y", "w", "h"))


def _is_group(node: dict) -> bool:
    return str(node.get("target") or "").lower() == "group"


def _children(node: dict) -> list:
    return list(node.get("children") or [])


def _walk(nodes: Iterable[dict]):
    for node in nodes:
        yield node
        yield from _walk(_children(node))


# ── 1. empty / redundant group hygiene ──────────────────────────────────────────

def prune_empty_groups(roots: list[dict], report: Optional[dict] = None) -> list[dict]:
    """Drop groups that contain nothing.

    021 emits 'Header' [226,0 85x142] and 'Header / 2' [0,0 201x143] with no
    children at all: pure noise in the layer list, and a QA 'empty_group' hard
    fail waiting to happen.  A group whose children were all suppressed upstream
    carries no pixels, so removing it cannot change the render.
    """
    dropped = []

    def _rec(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            if _is_group(node):
                node["children"] = _rec(_children(node))
                if not node["children"]:
                    dropped.append(node.get("id") or node.get("name"))
                    continue
            out.append(node)
        return out

    kept = _rec(copy.deepcopy(roots))
    if report is not None and dropped:
        report.setdefault("pruned_empty_groups", []).extend(dropped)
    return kept


def collapse_redundant_wrappers(roots: list[dict], report: Optional[dict] = None) -> list[dict]:
    """Collapse a group that wraps exactly one child of the same extent.

    021 emits group 'Photo' [0,0 338x600] whose only child is image 'Photo'
    [0,0 338x600].  The wrapper adds a click to reach the pixels and duplicates
    the name.  Only collapsed when the wrapper contributes nothing itself: no
    fill/effects/component role of its own.
    """
    collapsed = []

    def _informative(group: dict) -> bool:
        meta = group.get("meta") or {}
        return bool(group.get("fill") or group.get("effects") or group.get("component")
                    or meta.get("component") or meta.get("plate")
                    or meta.get("scrim") or group.get("mask"))

    def _rec(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            if _is_group(node):
                node["children"] = _rec(_children(node))
                kids = node["children"]
                if len(kids) == 1 and not _informative(node) \
                        and _same_box(_box(node), _box(kids[0])):
                    child = kids[0]
                    # Keep whichever name is more informative.
                    if _junk_name(child.get("name")) and not _junk_name(node.get("name")):
                        child["name"] = node.get("name")
                    collapsed.append(node.get("id") or node.get("name"))
                    out.append(child)
                    continue
            out.append(node)
        return out

    kept = _rec(copy.deepcopy(roots))
    if report is not None and collapsed:
        report.setdefault("collapsed_wrappers", []).extend(collapsed)
    return kept


def _junk_name(value) -> bool:
    return str(value or "").strip().lower() in _JUNK_NAMES


# ── 2. z-order: rasters behind their copy ───────────────────────────────────────

def order_children(roots: list[dict], z_key: Optional[Callable] = None,
                   report: Optional[dict] = None) -> list[dict]:
    """Within every group, put rasters behind and text in front.

    The mandate's editability contract: "rasters grouped behind their copy" and
    "text nodes reachable and editable at the top of their groups".  Ordering is
    STABLE within each class, so an upstream z that already separates overlapping
    art is preserved; this only fixes the text/raster relationship.

    Exactly TWO classes exist — text and everything-else — because that is the
    only distinction this pass is entitled to make.  Ranking art into sub-classes
    (image vs shape vs group) would re-sort ART AMONGST ITSELF and break the
    stability promise above: a full-bleed plate would sort above an overlapping
    cutout and hide it.  Layout already owns the paint order within the art
    (``_node_z``); all this pass adds is that copy floats above it.
    """
    def _rank(node: dict) -> int:
        return 1 if str(node.get("target") or "").lower() in _TEXTISH else 0

    reordered = []

    def _rec(nodes: list[dict]) -> list[dict]:
        for node in nodes:
            if _children(node):
                node["children"] = _rec(_children(node))
        before = [n.get("id") for n in nodes]
        # Stable: only the class rank moves anything.
        ordered = sorted(nodes, key=_rank)
        if [n.get("id") for n in ordered] != before:
            reordered.append(before)
        return ordered

    out = _rec(copy.deepcopy(roots))
    if report is not None and reordered:
        report.setdefault("reordered_groups", len(reordered))
    return out


# ── 3. band splitting: the 009 flat dump ────────────────────────────────────────

def _seams(nodes: list[dict], opts: dict) -> list[int]:
    """Indices (into y-sorted nodes) after which a horizontal seam falls."""
    ordered = sorted(nodes, key=lambda n: (_box(n)["y"], _box(n)["x"]))
    gaps = []
    running_bottom = None
    for index, node in enumerate(ordered):
        box = _box(node)
        if running_bottom is not None:
            gaps.append((index, box["y"] - running_bottom))
        running_bottom = max(running_bottom or 0.0, box["y"] + box["h"])
    positive = [g for _, g in gaps if g > 0]
    if not positive:
        return []
    typical = median(positive)
    threshold = max(float(opts["band_seam_min_px"]),
                    float(opts["band_seam_gap_factor"]) * typical)
    return [index for index, gap in gaps if gap >= threshold]


def band_split(nodes: list[dict], opts: dict) -> Optional[list[list[dict]]]:
    """Split a flat run of nodes into horizontal bands at whitespace seams.

    Returns None when the run is already readable or has no decisive seam, so a
    caller can leave the tree exactly as it was.
    """
    if len(nodes) < int(opts["band_min_children"]):
        return None
    ordered = sorted(nodes, key=lambda n: (_box(n)["y"], _box(n)["x"]))
    cuts = _seams(nodes, opts)
    if not cuts:
        return None
    bands, start = [], 0
    for cut in cuts + [len(ordered)]:
        band = ordered[start:cut]
        if band:
            bands.append(band)
        start = cut
    bands = [b for b in bands if len(b) >= int(opts["band_min_members"])]
    if len(bands) < int(opts["band_min_bands"]):
        return None
    return bands


def _backgroundish(node: dict, canvas: dict) -> bool:
    """A full-bleed plate is the stage, not a band member."""
    cw = float((canvas or {}).get("w") or 0) or 0.0
    ch = float((canvas or {}).get("h") or 0) or 0.0
    if cw <= 0 or ch <= 0:
        return False
    box = _box(node)
    return (box["w"] * box["h"]) >= 0.85 * cw * ch


def band_roots(roots: list[dict], canvas: dict, opts: dict,
               report: Optional[dict] = None) -> list[dict]:
    """Band the ROOT forest, which ``split_flat_groups`` cannot reach.

    107 emits sixteen top-level siblings — WEEK 1..5, three body lines, badges
    and arrows all peers of the background.  Banding them gives the designer the
    same header/chart/footer seams the eye already sees.  Full-bleed plates stay
    at the root: wrapping the background inside a band would nest the stage
    inside the scenery.
    """
    movable = [n for n in roots if not _backgroundish(n, canvas)]
    if len(movable) < int(opts["band_min_children"]):
        return roots
    bands = band_split(movable, opts)
    if not bands:
        return roots
    fixed = [n for n in roots if _backgroundish(n, canvas)]
    wrapped = []
    for index, band in enumerate(bands):
        if len(band) == 1:
            wrapped.append(band[0])
            continue
        wrapped.append({
            "id": f"root__band{index}",
            "target": "group",
            "box": _union(band),
            "name": _band_name(band, index, len(bands)),
            "children": band,
            "meta": {"structure": "band", "band_index": index, "derived_from": "root"},
        })
    if report is not None:
        report.setdefault("banded_roots", len(bands))
    return fixed + wrapped


def _band_name(band: list[dict], index: int, total: int) -> str:
    """A name a designer can scan: prefer the band's own copy, else its position."""
    texts = [str(n.get("text") or "").strip() for n in band
             if str(n.get("target") or "").lower() in _TEXTISH]
    texts = [t for t in texts if t]
    if len(band) == 1 and texts:
        return _snippet(texts[0])
    if index == 0:
        base = "Header"
    elif index == total - 1:
        base = "Footer"
    else:
        base = "Body"
    if texts:
        return f"{base} / {_snippet(texts[0])}"
    return base


def _snippet(text: str, length: int = 28) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= length else text[: length - 1] + "…"


def split_flat_groups(roots: list[dict], opts: dict,
                      report: Optional[dict] = None, _depth: int = 0) -> list[dict]:
    """Give every over-flat group a band structure.

    009's 'Screenshot' group holds 18 flat text nodes; after this it holds a
    Header band (Post / UPFRONT / @UpfrontFood / Volgend), a Body band (headline
    + paragraphs) and a Footer band (timestamp + engagement counts) — which is
    what a designer means by "the layering should be proper".
    """
    out = []
    for node in copy.deepcopy(roots):
        if _children(node):
            node["children"] = split_flat_groups(node["children"], opts, report, _depth + 1)
        if _is_group(node) and _depth <= int(opts["max_band_depth"]):
            bands = band_split(_children(node), opts)
            if bands:
                wrapped = []
                for index, band in enumerate(bands):
                    if len(band) == 1:
                        wrapped.append(band[0])
                        continue
                    wrapped.append({
                        "id": f"{node.get('id') or 'band'}__band{index}",
                        "target": "group",
                        "box": _union(band),
                        "name": _band_name(band, index, len(bands)),
                        "children": band,
                        "meta": {"structure": "band", "band_index": index,
                                 "derived_from": node.get("id")},
                    })
                node["children"] = wrapped
                if report is not None:
                    report.setdefault("banded_groups", []).append({
                        "id": node.get("id"), "bands": len(bands),
                    })
        out.append(node)
    return out


# ── 4. names ────────────────────────────────────────────────────────────────────

def dedupe_sibling_names(roots: list[dict], report: Optional[dict] = None) -> list[dict]:
    """Make sibling names unique and replace junk names with something scannable.

    002 emits a root group named 'Group'; 021 emits 'Header' twice. Figma shows
    the layer list by name, so duplicates and 'Group' are the difference between
    a navigable file and a guessing game.
    """
    renamed = []

    def _rec(nodes: list[dict]) -> list[dict]:
        seen: dict[str, int] = {}
        for node in nodes:
            if _children(node):
                node["children"] = _rec(_children(node))
            name = str(node.get("name") or "").strip()
            if _junk_name(name):
                better = _content_name(node)
                if better:
                    renamed.append({"from": name, "to": better, "id": node.get("id")})
                    name = better
            base = name or "Layer"
            count = seen.get(base.lower(), 0) + 1
            seen[base.lower()] = count
            node["name"] = base if count == 1 else f"{base} / {count}"
        return nodes

    out = _rec(copy.deepcopy(roots))
    if report is not None and renamed:
        report.setdefault("renamed", []).extend(renamed)
    return out


def _content_name(node: dict) -> Optional[str]:
    """Name a junk-named group after the copy it contains."""
    text = str(node.get("text") or "").strip()
    if text:
        return _snippet(text)
    texts = [str(n.get("text") or "").strip() for n in _walk(_children(node))
             if str(n.get("target") or "").lower() in _TEXTISH]
    texts = [t for t in texts if t]
    if texts:
        return _snippet(texts[0])
    return None


# ── 5. entry point ──────────────────────────────────────────────────────────────

def restructure(roots: list[dict], canvas: dict, cfg: Optional[dict] = None,
                z_key: Optional[Callable] = None) -> tuple[list[dict], dict]:
    """Shape a layout forest into a tree a designer can work in.

    Returns ``(roots, report)``.  Never raises on a well-formed forest and never
    mutates the input.  Order matters: prune before collapse (pruning can orphan a
    wrapper), band before naming (bands need names), order last so nothing
    re-sorts after.
    """
    opts = options(cfg)
    report: dict = {"applied": False}
    if not opts["enabled"] or not roots:
        report["reason"] = "disabled" if not opts["enabled"] else "empty"
        return roots, report
    out = copy.deepcopy(roots)
    if opts["prune_empty_groups"]:
        out = prune_empty_groups(out, report)
    if opts["collapse_redundant_wrappers"]:
        out = collapse_redundant_wrappers(out, report)
    if opts["band_split"]:
        out = split_flat_groups(out, opts, report)
        out = band_roots(out, canvas, opts, report)
    out = dedupe_sibling_names(out, report)
    if opts["text_above_rasters"]:
        out = order_children(out, z_key, report)
    report["applied"] = True
    report["root_count"] = len(out)
    report["layer_count"] = sum(1 for _ in _walk(out))
    return out, report


__all__ = [
    "DEFAULTS", "options", "enabled", "restructure", "prune_empty_groups",
    "collapse_redundant_wrappers", "order_children", "band_split",
    "split_flat_groups", "dedupe_sibling_names",
]
