"""codia_parity.py — score how closely a design.json reproduces Codia AI's construction.

Ground truth is a Codia-produced Figma node tree fetched from the Figma REST API
(e.g. runs/codia-teardown-009.json).  The comparator extracts what Codia actually
built — 16/16 native Inter text nodes, weight-split runs, image cutouts for
icons/emoji, a pill button, solid plates — and scores our design.json against it
per dimension.  See docs/CODIA-PARITY-SPEC.md for the teardown and the scoring
contract (section 8).

CPU-only, stdlib-only.  Usage:

    python scripts/codia_parity.py --design runs/<run>/design.json \
        --template runs/codia-teardown-009.json [--json report.json] [--fail-under 80]
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
import unicodedata

# Dimension weights (docs/CODIA-PARITY-SPEC.md section 8).
WEIGHTS = {
    "native_text_ratio": 0.25,
    "font_family": 0.08,      # body/UI lines: exact family match (Inter on both templates)
    "headline_font": 0.07,    # display lines: matched family (full) or serif/sans class (half)
    "font_weight": 0.12,
    "font_size": 0.08,
    "text_position": 0.08,
    "letter_spacing": 0.04,
    "icon_cutouts": 0.08,
    "button": 0.05,
    "node_budget": 0.10,      # radical minimalism: 1.0 at <=1x Codia's count, 0.0 at >=2.5x
    "flatness": 0.05,         # groups only where Codia groups (flat tree for simple scenes)
}

# Families whose name marks them as serif display/text faces (class fallback for
# headline_font scoring when the exact family differs).
_SERIF_HINTS = (
    "playfair", "georgia", "times", "garamond", "merriweather", "lora", "baskerville",
    "bodoni", "didot", "caladea", "cambria", "cormorant", "prata", "abril", "rozha",
    "crimson", "spectral", "tinos", "freight", "charter", "literata", "domine",
)


def _family_class(name) -> str:
    token = str(name or "").casefold()
    if "sans" in token:
        return "sans"
    if "serif" in token or any(hint in token for hint in _SERIF_HINTS):
        return "serif"
    return "sans"

_DOT_VARIANTS = re.compile(r"[·•‧・\.\-‐-―−]")
_WS = re.compile(r"\s+")


def normalize_text(value) -> str:
    """Casefolded comparison key: emoji stripped, dot/dash variants unified."""
    text = str(value or "")
    out = []
    for char in text:
        code = ord(char)
        # Strip emoji / pictographs / variation selectors / ZWJ.
        if code >= 0x1F000 or code in (0x200D, 0xFE0E, 0xFE0F, 0x20E3):
            continue
        if unicodedata.category(char) == "So":
            continue
        out.append(char)
    text = "".join(out)
    text = _DOT_VARIANTS.sub(".", text)
    text = _WS.sub(" ", text).strip().casefold()
    return text


def _similarity(codia_text: str, our_text: str) -> float:
    """Similarity of a template line vs a candidate node's text.

    Containment counts: a raster slice covering three template lines must still
    match each of them, so the longest common block relative to the template
    line is blended with the plain ratio.
    """
    a, b = normalize_text(codia_text), normalize_text(our_text)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    match = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    containment = match.size / len(a)
    return max(ratio, containment * 0.95)


# --------------------------------------------------------------------------- geometry

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a, b) -> float:
    return math.hypot((a["x"] + a["w"] / 2) - (b["x"] + b["w"] / 2),
                      (a["y"] + a["h"] / 2) - (b["y"] + b["h"] / 2))


def _hex_to_rgb(value):
    value = str(value or "").lstrip("#")
    if len(value) >= 6:
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- template

def _figma_rgb(color) -> tuple:
    return tuple(round(255 * float((color or {}).get(ch, 0))) for ch in ("r", "g", "b"))


def load_codia_template(path_or_doc, node_id=None):
    """Extract the comparison contract from a Figma REST /nodes payload.

    ``node_id`` selects one reconstruction when the payload carries several
    (runs/codia-teardown-3.json holds both 052 @ 15489:111 and 076 @ 15489:133);
    without it the largest FRAME document wins (single-teardown files).
    """
    doc = path_or_doc
    if isinstance(doc, str):
        with open(doc, encoding="utf-8") as fh:
            doc = json.load(fh)

    # Pick the node document that is a FRAME with children (the reconstruction).
    # Codia also parks a plain RECTANGLE with the source image next to it — skip it.
    frames = []
    for key, entry in (doc.get("nodes") or {}).items():
        node = entry.get("document") or {}
        if node_id and str(key) != str(node_id) and str(node.get("id")) != str(node_id):
            continue
        if node.get("type") == "FRAME" and node.get("children"):
            frames.append(node)
    if not frames:
        raise ValueError("template contains no FRAME document with children"
                         + (f" (node {node_id})" if node_id else ""))
    root = max(frames, key=lambda n: len(json.dumps(n)))
    origin = root.get("absoluteBoundingBox") or {"x": 0, "y": 0}
    ox, oy = float(origin.get("x", 0)), float(origin.get("y", 0))
    canvas = {"w": float(origin.get("width", 0) or 0), "h": float(origin.get("height", 0) or 0)}

    texts, cutouts, plates, frames_out = [], [], [], []
    stats = {"node_count": 0, "max_depth": 0}
    button = None

    def image_box(node):
        bb = node.get("absoluteBoundingBox") or {}
        return {"x": float(bb.get("x", 0)) - ox, "y": float(bb.get("y", 0)) - oy,
                "w": float(bb.get("width", 0)), "h": float(bb.get("height", 0))}

    def walk(node, depth=0):
        stats["node_count"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        ntype = node.get("type")
        box = image_box(node)
        if ntype == "TEXT":
            style = node.get("style") or {}
            fills = [f for f in node.get("fills") or [] if f.get("visible") is not False]
            color = _figma_rgb(fills[0].get("color")) if fills else None
            texts.append({
                "id": node.get("id"),
                "characters": node.get("characters") or "",
                "fontFamily": style.get("fontFamily"),
                "fontWeight": style.get("fontWeight"),
                "fontSize": style.get("fontSize"),
                "lineHeightPx": style.get("lineHeightPx"),
                "letterSpacing": style.get("letterSpacing", 0),
                "color": color,
                "box": box,
                "multiline": "\n" in (node.get("characters") or ""),
            })
        elif ntype == "RECTANGLE":
            fills = [f for f in node.get("fills") or [] if f.get("visible") is not False]
            kind = (fills[0].get("type") if fills else None)
            area_frac = (box["w"] * box["h"]) / max(1.0, canvas["w"] * canvas["h"])
            if kind == "IMAGE" and area_frac < 0.5:
                cutouts.append({"id": node.get("id"), "box": box})
            elif kind == "SOLID":
                plates.append({"id": node.get("id"), "box": box,
                               "color": _figma_rgb(fills[0].get("color")),
                               "cornerRadius": node.get("cornerRadius", 0)})
        elif ntype == "FRAME":
            frames_out.append({"id": node.get("id"), "name": node.get("name"), "box": box})
        for child in node.get("children") or []:
            walk(child, depth + 1)

    walk(root)

    # Button: a FRAME named Button, or any solid rect with a pill radius plus a
    # text node inside its box.
    for frame in frames_out:
        if str(frame.get("name", "")).strip().lower() != "button":
            continue
        pill = next((p for p in plates if _iou(p["box"], frame["box"]) > 0.5), None)
        label = next((t for t in texts if _iou(t["box"], frame["box"]) > 0.1), None)
        if pill and label:
            button = {"box": frame["box"], "pill": pill, "text": label}
            break
    if button is None:
        for pill in plates:
            if float(pill.get("cornerRadius") or 0) >= 0.25 * max(1.0, pill["box"]["h"]):
                label = next((t for t in texts if _iou(t["box"], pill["box"]) > 0.1), None)
                if label:
                    button = {"box": pill["box"], "pill": pill, "text": label}
                    break

    families = [t["fontFamily"] for t in texts if t.get("fontFamily")]
    dominant = max(set(families), key=families.count) if families else None

    # Display vs body classification: a display line is much larger than the median.
    sizes = sorted(float(t["fontSize"]) for t in texts if t.get("fontSize"))
    median = sizes[(len(sizes) - 1) // 2] if sizes else 0.0
    for text in texts:
        size = float(text.get("fontSize") or 0)
        text["display"] = bool(size and median and size >= 1.6 * median and size >= 48)

    # Groups beyond the outer document frame and its "Root" wrapper. Zero means
    # Codia judged the scene simple enough for a completely flat tree.
    group_count = max(0, len(frames_out) - 2)
    complexity = "simple" if group_count == 0 else "complex"
    return {
        "canvas": canvas,
        "texts": texts,
        "cutouts": cutouts,
        "plates": plates,
        "button": button,
        "node_count": stats["node_count"],
        "max_depth": stats["max_depth"],
        "group_count": group_count,
        "complexity": complexity,
        "dominant_family": dominant,
    }


# --------------------------------------------------------------------------- ours

_SLICE_NAME = re.compile(r"\s*(?:—|--|-)\s*(?:raster\s*slice|ras\b).*$", re.IGNORECASE)


def load_our_design(path_or_doc):
    """Flatten design.json to absolute-space leaves (coordinate_space is local)."""
    doc = path_or_doc
    if isinstance(doc, str):
        with open(doc, encoding="utf-8") as fh:
            doc = json.load(fh)
    canvas = {"w": float(doc.get("canvas", {}).get("w", 0)),
              "h": float(doc.get("canvas", {}).get("h", 0))}
    leaves = []
    stats = {"node_count": 0, "max_depth": 0}

    def absolute(box, off):
        box = box or {}
        return {"x": float(box.get("x", 0)) + off[0], "y": float(box.get("y", 0)) + off[1],
                "w": float(box.get("w", 0)), "h": float(box.get("h", 0))}

    def walk(node, off=(0.0, 0.0), depth=0):
        stats["node_count"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        box = absolute(node.get("box"), off)
        children = node.get("children") or []
        meta = node.get("meta") or {}
        record = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name") or "",
            "box": box,
            "style": node.get("style") or {},
            "text": node.get("text"),
            "text_runs": node.get("text_runs") or [],
            "fill": node.get("fill"),
            "radius": node.get("radius"),
            "effects": [e for e in node.get("effects") or []
                        if isinstance(e, dict) and e.get("visible") is not False],
            "meta": meta,
            "is_group": bool(children) or node.get("type") == "group",
            "raster_slice": bool(meta.get("raster_slice")
                                 or str(meta.get("layer_kind", "")).replace("_", "-") == "raster-slice"
                                 or "raster slice" in str(node.get("name", "")).lower()),
        }
        leaves.append(record)
        for child in children:
            walk(child, (box["x"], box["y"]), depth + 1)

    for layer in doc.get("layers") or []:
        walk(layer)

    for record in leaves:
        if record["type"] != "text" and not record.get("text"):
            # Raster slices carry the source line in their layer name.
            cleaned = _SLICE_NAME.sub("", str(record["name"]))
            cleaned = re.sub(r"\s*\(low confidence\)\s*$", "", cleaned, flags=re.IGNORECASE)
            record["name_text"] = cleaned
    return {"canvas": canvas, "leaves": leaves,
            "node_count": stats["node_count"], "max_depth": stats["max_depth"]}


# --------------------------------------------------------------------------- matching

def match_text_lines(template, ours):
    """Assign each Codia text line its best counterpart among our leaves.

    TEXT leaves are exclusive (one line each); image/raster leaves may cover
    several template lines (a slice can contain a whole row).
    """
    candidates = []
    for leaf in ours["leaves"]:
        if leaf["is_group"]:
            continue
        if leaf["type"] == "text" and leaf.get("text"):
            candidates.append((leaf, str(leaf["text"]), True))
        else:
            name_text = leaf.get("name_text") or ""
            if len(normalize_text(name_text)) >= 2:
                candidates.append((leaf, name_text, False))

    matches = []
    used_text_ids = set()
    for line in sorted(template["texts"], key=lambda t: -(t["box"]["w"] * t["box"]["h"])):
        best, best_score = None, 0.0
        for leaf, text, exclusive in candidates:
            if exclusive and id(leaf) in used_text_ids:
                continue
            sim = _similarity(line["characters"], text)
            if sim < 0.35:
                continue
            iou = _iou(line["box"], leaf["box"])
            dist = _center_dist(line["box"], leaf["box"]) / max(1.0, template["canvas"]["h"])
            geo = max(iou, 1.0 - min(1.0, dist / 0.25))
            score = 0.75 * sim + 0.25 * geo
            if sim < 0.55 and iou < 0.05:
                continue
            if score > best_score:
                best, best_score = (leaf, exclusive), score
        if best is not None:
            leaf, exclusive = best
            if exclusive:
                used_text_ids.add(id(leaf))
            matches.append({"template": line, "leaf": leaf, "score": round(best_score, 3),
                            "native": leaf["type"] == "text"})
        else:
            matches.append({"template": line, "leaf": None, "score": 0.0, "native": False})
    return matches


def _weight_bucket(value):
    try:
        return int(round(float(value) / 100.0)) * 100
    except (TypeError, ValueError):
        return None


def _button_score(template, ours):
    spec = template.get("button")
    if not spec:
        return 1.0, {"note": "template has no button"}
    pill_box, pill_color = spec["pill"]["box"], spec["pill"]["color"]
    label = spec["text"]
    best = None
    for leaf in ours["leaves"]:
        radius = leaf.get("radius")
        if isinstance(radius, dict):
            radius = max((v for v in radius.values() if isinstance(v, (int, float))), default=0)
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            continue
        box = leaf["box"]
        if box["h"] <= 0 or radius < 0.25 * box["h"]:
            continue
        if _iou(box, pill_box) < 0.3:
            continue
        best = leaf
        break
    if best is None:
        return 0.0, {"found": False}
    detail = {"found": True, "id": best.get("id")}
    score = 0.4  # pill geometry at the right place
    fill = best.get("fill")
    color = _hex_to_rgb(fill.get("color") if isinstance(fill, dict) else fill)
    if color and pill_color:
        delta = max(abs(a - b) for a, b in zip(color, pill_color))
        detail["fill_delta"] = delta
        score += 0.3 * max(0.0, 1.0 - delta / 40.0)
    label_leaf = None
    for leaf in ours["leaves"]:
        if leaf["type"] == "text" and _similarity(label["characters"], leaf.get("text")) >= 0.8:
            label_leaf = leaf
            break
    if label_leaf is not None:
        want = _weight_bucket(label["fontWeight"])
        got = _weight_bucket((label_leaf.get("style") or {}).get("fontWeight"))
        detail["text_weight"] = {"codia": want, "ours": got}
        score += 0.2 if (want is not None and got is not None and abs(want - got) <= 100) else 0.1
    if not best.get("effects"):
        score += 0.1
    else:
        detail["extra_effects"] = [e.get("type") for e in best["effects"]]
    return min(1.0, score), detail


def compare(template, ours, complexity="auto"):
    """Score ours vs template. ``complexity``: "auto" derives the expectation from
    the template itself; "simple" forces the flat-tree expectation (target: zero
    groups); "complex" allows the template's own group count."""
    if complexity not in (None, "auto"):
        template = dict(template)
        template["complexity"] = complexity
        if complexity == "simple":
            template["group_count"] = 0
    matches = match_text_lines(template, ours)
    n_lines = max(1, len(template["texts"]))
    native = [m for m in matches if m["native"]]
    dominant = template.get("dominant_family") or "Inter"

    def frac(values):
        return sum(values) / len(values) if values else 0.0

    scores, detail = {}, {}
    scores["native_text_ratio"] = len(native) / n_lines
    detail["native_text"] = {"native": len(native), "total": n_lines,
                             "missing_or_raster": [m["template"]["characters"][:40]
                                                   for m in matches if not m["native"]]}

    body_native = [m for m in native if not m["template"].get("display")]
    display_lines = [m for m in matches if m["template"].get("display")]

    fam = [1.0 if str((m["leaf"].get("style") or {}).get("fontFamily", "")).strip().casefold()
           == str(m["template"].get("fontFamily") or dominant).strip().casefold() else 0.0
           for m in body_native]
    scores["font_family"] = frac(fam) * scores["native_text_ratio"]
    detail["font_family"] = {"template_family": dominant,
                             "ours": sorted({str((m["leaf"].get("style") or {}).get("fontFamily"))
                                             for m in body_native})}

    # Display type: exact family = 1.0, same serif/sans class = 0.5 (Codia matches a
    # real Google display face here — Playfair Display on template 041).
    if display_lines:
        marks, rows = [], []
        for m in display_lines:
            want = str(m["template"].get("fontFamily") or "")
            if not m["native"]:
                marks.append(0.0)
                rows.append({"text": m["template"]["characters"][:30], "codia": want,
                             "ours": None, "score": 0.0})
                continue
            got = str((m["leaf"].get("style") or {}).get("fontFamily") or "")
            if want.strip().casefold() == got.strip().casefold():
                mark = 1.0
            elif _family_class(want) == _family_class(got):
                mark = 0.5
            else:
                mark = 0.0
            marks.append(mark)
            rows.append({"text": m["template"]["characters"][:30], "codia": want,
                         "ours": got, "score": mark})
        scores["headline_font"] = frac(marks)
        detail["headline_font"] = rows
    else:
        # No display type in the template: keep pressure on body family instead.
        scores["headline_font"] = scores["font_family"]
        detail["headline_font"] = {"note": "template has no display-class line"}

    wt = []
    weight_rows = []
    for m in native:
        want = _weight_bucket(m["template"]["fontWeight"])
        got = _weight_bucket((m["leaf"].get("style") or {}).get("fontWeight"))
        ok = want is not None and got == want
        wt.append(1.0 if ok else 0.0)
        weight_rows.append({"text": m["template"]["characters"][:30], "codia": want,
                            "ours": got, "match": ok})
    scores["font_weight"] = frac(wt) * scores["native_text_ratio"]
    detail["font_weight"] = weight_rows

    sizes = []
    for m in native:
        want, got = m["template"].get("fontSize"), (m["leaf"].get("style") or {}).get("fontSize")
        if want and got:
            sizes.append(abs(float(got) - float(want)) / float(want))
    scores["font_size"] = max(0.0, 1.0 - frac(sizes) / 0.15) if sizes else 0.0
    detail["font_size"] = {"mean_rel_dev": round(frac(sizes), 4) if sizes else None}

    dists = [(_center_dist(m["template"]["box"], m["leaf"]["box"])
              / max(1.0, template["canvas"]["h"])) for m in native]
    scores["text_position"] = max(0.0, 1.0 - frac(dists) / 0.05) if dists else 0.0
    detail["text_position"] = {"mean_center_offset_frac": round(frac(dists), 4) if dists else None}

    ls = [1.0 if abs(float((m["leaf"].get("style") or {}).get("letterSpacing") or 0.0)) <= 0.5
          else 0.0 for m in native]
    scores["letter_spacing"] = frac(ls) * scores["native_text_ratio"]
    detail["letter_spacing"] = {"within_half_px": int(sum(ls)), "of": len(ls)}

    covered, by_type = 0, {"image": 0, "shape": 0, "other": 0}
    cutout_rows = []
    max_area = 0.25 * template["canvas"]["w"] * template["canvas"]["h"]
    for cut in template["cutouts"]:
        hit = None
        for leaf in ours["leaves"]:
            if leaf["is_group"] or leaf["type"] == "text":
                continue
            if leaf["box"]["w"] * leaf["box"]["h"] > max_area:
                continue
            if _iou(cut["box"], leaf["box"]) >= 0.25:
                hit = leaf
                break
        cutout_rows.append({"codia_box": cut["box"], "covered": bool(hit),
                            "ours_type": hit["type"] if hit else None,
                            "ours_id": hit.get("id") if hit else None})
        if hit:
            covered += 1
            by_type[hit["type"] if hit["type"] in by_type else "other"] += 1
    scores["icon_cutouts"] = covered / max(1, len(template["cutouts"]))
    detail["icon_cutouts"] = {"covered": covered, "of": len(template["cutouts"]),
                              "by_type": by_type, "rows": cutout_rows}

    scores["button"], detail["button"] = _button_score(template, ours)

    # Radical minimalism budget: full marks at <= 1x Codia's node count, zero at
    # >= 2.5x. Codia ships the fewest nodes that reproduce the ad (9 for a flat
    # photo ad, 38 for a grouped UI screenshot); we tolerate up to 2.5x while
    # steering toward 1x.
    codia_n, our_n = template["node_count"], max(1, ours["node_count"])
    scores["node_budget"] = max(0.0, min(1.0, (2.5 * codia_n - our_n) / (1.5 * codia_n)))
    detail["node_budget"] = {"codia_nodes": codia_n,
                             "our_nodes": ours["node_count"],
                             "ratio": round(our_n / max(1, codia_n), 2),
                             "codia_depth": template["max_depth"],
                             "our_depth": ours["max_depth"]}

    # Flatness: groups only where Codia groups. A simple scene (Codia group_count
    # 0) expects a completely flat tree; each extra wrapper of ours costs.
    our_groups = sum(1 for leaf in ours["leaves"] if leaf["is_group"])
    codia_groups = template.get("group_count", 0)
    scores["flatness"] = min(1.0, (codia_groups + 1) / (our_groups + 1))
    detail["flatness"] = {"codia_groups": codia_groups, "our_groups": our_groups,
                          "complexity": template.get("complexity")}

    # Paragraph integrity (reported, not weighted): multi-line template blocks
    # should stay ONE text node with \n rather than per-line splits.
    para_rows = []
    for m in matches:
        if m["template"].get("multiline"):
            intact = bool(m["native"] and "\n" in str(m["leaf"].get("text") or ""))
            para_rows.append({"text": m["template"]["characters"][:30], "single_node": intact})
    detail["paragraph_integrity"] = para_rows

    overall = 100.0 * sum(WEIGHTS[key] * scores[key] for key in WEIGHTS)
    return {"overall": round(overall, 2),
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "weights": WEIGHTS,
            "detail": detail,
            "matches": [{"codia": m["template"]["characters"][:40],
                         "ours": (m["leaf"] or {}).get("id"),
                         "native": m["native"], "score": m["score"]} for m in matches]}


# ------------------------------------------------------ template-free construction score
#
# The dimensions above compare our design.json against ONE recorded Codia tree. But the
# Codia teardown (docs/CODIA-PARITY-SPEC.md) is a *policy*, not a single output, and QA
# runs on ads that have no matching Codia teardown. ``score_construction`` scores the same
# construction CONTRACT — native text, Inter/display two-tier font policy, letterSpacing 0,
# single-weight nodes (weight runs split), emoji-as-image, node budget, flatness — from our
# design.json alone (plus, optionally, an OCR-accurate native_text_ratio the metrics layer
# already computed). No template is required, so pixel_diff can embed it in every qa.json.

# The construction contract leads with native text: everything else is secondary to
# "every string is native editable TEXT".
CONSTRUCTION_WEIGHTS = {
    "native_text_ratio": 0.40,   # native editable TEXT lines / all readable lines
    "font_policy": 0.20,         # Inter body / real display face, letterSpacing 0
    "weight_split": 0.12,        # mixed-weight lines split into single-weight nodes
    "emoji_as_image": 0.10,      # emoji are pixel cutouts, never glyphs/vectors
    "node_budget": 0.12,         # radical minimalism vs the scene's complexity
    "flatness": 0.06,            # groups only where content genuinely clusters
}

# Node budgets from the teardown (§9): Codia ships 9 nodes for a flat photo ad, 38 for a
# grouped UI screenshot. Owners' recommendation: <= ~12 simple, <= ~45 complex.
_SIMPLE_NODE_BUDGET = 12
_COMPLEX_NODE_BUDGET = 45
_COMPLEX_ARCHETYPES = frozenset({"social_screenshot", "comparison_grid"})

_EMOJI_RE = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "]"
)


def _has_emoji(text) -> bool:
    text = str(text or "")
    return bool(_EMOJI_RE.search(text)) or any(ord(ch) >= 0x1F000 for ch in text)


def _leaf_weights(leaf) -> list:
    """Distinct font weights a leaf paints (a mixed-weight node should have been split)."""
    weights = []
    for run_ in leaf.get("text_runs") or []:
        if not isinstance(run_, dict):
            continue
        style = run_.get("style") if isinstance(run_.get("style"), dict) else run_
        bucket = _weight_bucket(style.get("fontWeight"))
        if bucket is not None:
            weights.append(bucket)
    if not weights:
        bucket = _weight_bucket((leaf.get("style") or {}).get("fontWeight"))
        if bucket is not None:
            weights.append(bucket)
    return sorted(set(weights))


def _is_display_text(leaf, median_size) -> bool:
    size = float((leaf.get("style") or {}).get("fontSize") or 0)
    return bool(size and median_size and size >= 1.6 * median_size and size >= 48)


def _text_leaves(leaves) -> list:
    return [l for l in leaves if not l["is_group"] and l["type"] == "text" and l.get("text")]


def _baked_text_leaves(leaves) -> list:
    """Non-text leaves that still carry a text line — raster slices / baked headlines."""
    out = []
    for leaf in leaves:
        if leaf["is_group"] or leaf["type"] == "text":
            continue
        meta = leaf.get("meta") or {}
        carries_text = (
            leaf.get("raster_slice") or meta.get("wordmark") or meta.get("platform_lockup")
            or meta.get("line_ids") or leaf.get("text")
            or len(normalize_text(leaf.get("name_text") or "")) >= 3
        )
        if carries_text and not (meta.get("emoji") or _has_emoji(leaf.get("name_text"))):
            out.append(leaf)
    return out


def score_construction(design, *, native_text_ratio=None, archetype=None,
                       complexity=None, ocr=None) -> dict:
    """Score a design.json against the Codia CONSTRUCTION CONTRACT with no template.

    ``native_text_ratio`` overrides the design-internal proxy with the metrics layer's
    OCR-accurate value (native TEXT lines / all readable OCR lines) when available.
    ``ocr`` (optional) supplies readable line count as the denominator otherwise.
    Returns ``{"score": 0..100, "scores": {...}, "detail": {...}}``.
    """
    ours = design if isinstance(design, dict) and "leaves" in design else load_our_design(design)
    leaves = ours["leaves"]
    text_leaves = _text_leaves(leaves)
    baked = _baked_text_leaves(leaves)
    scores, detail = {}, {}

    # 1. native text ratio — the objective. Prefer the OCR-accurate value; else the
    # design-internal native/(native+baked) proxy, with the readable-line count folding in
    # any lines that are simply MISSING (present in OCR, absent from the tree).
    if isinstance(native_text_ratio, (int, float)):
        ntr = max(0.0, min(1.0, float(native_text_ratio)))
        detail["native_text_ratio"] = {"value": round(ntr, 4), "source": "metrics"}
    else:
        native_n = len(text_leaves)
        readable = 0
        if isinstance(ocr, dict):
            for line in ocr.get("lines") or []:
                text = normalize_text(line.get("text"))
                if float(line.get("conf", 1) or 0) >= 0.5 and len(text) >= 3:
                    readable += 1
        denom = max(native_n + len(baked), readable, 1)
        ntr = native_n / denom
        detail["native_text_ratio"] = {"value": round(ntr, 4), "native": native_n,
                                       "baked": len(baked), "readable_lines": readable,
                                       "source": "design"}
    scores["native_text_ratio"] = ntr

    # 2. font policy: Inter for body (display type may use a real display face) AND
    # letterSpacing snapped to 0 on every native line.
    sizes = sorted(float((l.get("style") or {}).get("fontSize") or 0) for l in text_leaves)
    median = sizes[(len(sizes) - 1) // 2] if sizes else 0.0
    fam_marks, ls_marks, fam_rows = [], [], []
    for leaf in text_leaves:
        style = leaf.get("style") or {}
        family = str(style.get("fontFamily") or "").strip()
        display = _is_display_text(leaf, median)
        if display:
            fam_ok = 1.0 if family else 0.0   # any real display face is acceptable
        else:
            fam_ok = 1.0 if family.casefold() == "inter" else 0.0
        try:
            ls = abs(float(style.get("letterSpacing") or 0.0))
        except (TypeError, ValueError):
            ls = 0.0
        ls_ok = 1.0 if ls <= 0.5 else 0.0
        fam_marks.append(fam_ok)
        ls_marks.append(ls_ok)
        fam_rows.append({"text": str(leaf.get("text"))[:24], "family": family,
                         "display": display, "family_ok": bool(fam_ok),
                         "letter_spacing_ok": bool(ls_ok)})

    def _frac(values):
        return sum(values) / len(values) if values else 1.0

    scores["font_policy"] = round(0.6 * _frac(fam_marks) + 0.4 * _frac(ls_marks), 4)
    detail["font_policy"] = {"family_compliant": round(_frac(fam_marks), 4),
                             "letter_spacing_clean": round(_frac(ls_marks), 4),
                             "rows": fam_rows}

    # 3. weight split: a mixed-weight line must be split into single-weight sibling nodes.
    single = [1.0 if len(_leaf_weights(leaf)) <= 1 else 0.0 for leaf in text_leaves]
    scores["weight_split"] = round(_frac(single), 4)
    detail["weight_split"] = {"single_weight": int(sum(single)), "of": len(single)}

    # 4. emoji as image: emoji are pixel cutouts, never glyphs on a text node or vectors.
    emoji_img = emoji_bad = 0
    for leaf in leaves:
        if leaf["is_group"]:
            continue
        meta = leaf.get("meta") or {}
        is_emoji = bool(meta.get("emoji")) or _has_emoji(leaf.get("text")) or _has_emoji(leaf.get("name"))
        if not is_emoji:
            continue
        if leaf["type"] == "image":
            emoji_img += 1
        else:
            emoji_bad += 1   # emoji baked on a text node, or a vectorized emoji glyph
    total_emoji = emoji_img + emoji_bad
    scores["emoji_as_image"] = round(emoji_img / total_emoji, 4) if total_emoji else 1.0
    detail["emoji_as_image"] = {"as_image": emoji_img, "as_glyph_or_vector": emoji_bad}

    # 5. node budget: radical minimalism against the scene's complexity.
    if complexity in (None, "auto"):
        leaf_count = sum(1 for l in leaves if not l["is_group"])
        complexity = ("complex" if (archetype in _COMPLEX_ARCHETYPES or leaf_count > 15)
                      else "simple")
    budget = _COMPLEX_NODE_BUDGET if complexity == "complex" else _SIMPLE_NODE_BUDGET
    node_count = ours["node_count"]
    scores["node_budget"] = round(max(0.0, min(1.0, (2.5 * budget - node_count) / (1.5 * budget))), 4)
    detail["node_budget"] = {"nodes": node_count, "budget": budget, "complexity": complexity}

    # 6. flatness: a simple scene should be flat; a complex one groups only where content
    # genuinely clusters.
    our_groups = sum(1 for l in leaves if l["is_group"])
    expected_groups = 0 if complexity == "simple" else 4
    scores["flatness"] = round(min(1.0, (expected_groups + 1) / (our_groups + 1)), 4)
    detail["flatness"] = {"groups": our_groups, "expected": expected_groups}

    overall = 100.0 * sum(CONSTRUCTION_WEIGHTS[k] * scores[k] for k in CONSTRUCTION_WEIGHTS)
    return {"score": round(overall, 2), "scores": scores,
            "weights": CONSTRUCTION_WEIGHTS, "detail": detail, "complexity": complexity}


# --------------------------------------------------------------------------- CLI

def format_table(report) -> str:
    rows = [("dimension", "score", "weight")]
    for key in WEIGHTS:
        rows.append((key, f"{report['scores'][key]:.3f}", f"{WEIGHTS[key]:.2f}"))
    rows.append(("OVERALL", f"{report['overall']:.1f} / 100", ""))
    width = max(len(r[0]) for r in rows) + 2
    lines = [f"{name:<{width}}{score:>14}  {weight}" for name, score, weight in rows]
    lines.insert(1, "-" * (width + 20))
    lines.insert(-1, "-" * (width + 20))
    return "\n".join(lines)


def run(design_path, template_path, complexity="auto", template_node=None):
    template = load_codia_template(template_path, node_id=template_node)
    ours = load_our_design(design_path)
    return compare(template, ours, complexity=complexity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", required=True, help="our design.json")
    parser.add_argument("--template", required=True, help="Codia Figma /nodes JSON")
    parser.add_argument("--template-node", default=None,
                        help="node id inside the template payload when it holds several "
                             "reconstructions (e.g. 15489:111 = 052 in codia-teardown-3.json)")
    parser.add_argument("--complexity", choices=("auto", "simple", "complex"),
                        default="auto", help="scene complexity expectation "
                        "(auto = derive from the template's own structure)")
    parser.add_argument("--json", dest="json_out", help="write full report JSON here")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="exit 1 when overall score is below this")
    args = parser.parse_args(argv)

    report = run(args.design, args.template, complexity=args.complexity,
                 template_node=args.template_node)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(format_table(report))
    missing = report["detail"]["native_text"]["missing_or_raster"]
    if missing:
        print(f"\nnon-native / unmatched template lines ({len(missing)}):")
        for text in missing:
            print(f"  - {text}")
    weight_misses = [r for r in report["detail"]["font_weight"] if not r["match"]]
    if weight_misses:
        print(f"\nweight mismatches ({len(weight_misses)}):")
        for row in weight_misses:
            print(f"  - {row['text']!r}: codia={row['codia']} ours={row['ours']}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print(f"\nreport -> {args.json_out}")
    if args.fail_under is not None and report["overall"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
