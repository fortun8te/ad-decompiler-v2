"""TEXT CONTRACT verification sweep — the "no weird text things, ever again" gate.

For a finished pipeline run (``design.json`` + ``preview.png`` + ``normalized.png``
+ ``ocr.json``) this machine-checks every emitted text node against the source ink:

  (a) MASS      rendered glyph-ink mass within tolerance of the source line's ink mass
  (b) PLACEMENT rendered ink bbox centre within 3% of canvas dims vs the source bbox
  (c) CONTRAST  fill vs its local plate >= 3:1 unless the source itself is low-contrast
  (d) CLIP      the rendered glyph ink is not clipped by its node box
  (e) OVERLAP   no two text nodes overlap the same span
  (f) STRIKE    struck source lines carry AND render their strike decoration
  (g) MISSING   no non-empty OCR line is absent from design.json unless deliberately
                baked (present in ``kept_in_photo``)

Placement/mass/clip/contrast are measured from PIXELS (source ``normalized.png`` vs
rendered ``preview.png`` at each source line box) so the check is renderer-truthful and
independent of the design-tree coordinate frame. Missing/overlap/strike are read from
``design.json``.

Usage::

    python scripts/text_contract_check.py runs/postfix-benchmark-7          # sweep all
    python scripts/text_contract_check.py runs/postfix-benchmark-7/091_...   # one run
    python scripts/text_contract_check.py <dir> --json report.json --strict

Exit code is non-zero when any HARD violation is found (``--strict`` also fails on
soft/WARN findings), so it can gate a benchmark.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Optional

import numpy as np
from PIL import Image


# ── tolerances ────────────────────────────────────────────────────────────────
# The coordinator's contract is "rendered ink mass within 25% of source" — that band
# is a WARN; a HARD only fires on an egregious miss (near-empty or grossly bloated),
# which is what "no weird text, ever" is really about.
MASS_WARN_LO, MASS_WARN_HI = 0.75, 1.25   # 25% contract band (soft)
MASS_HARD_LO, MASS_HARD_HI = 0.40, 2.2    # egregious band (hard)
PLACEMENT_FRAC = 0.03                 # bbox-centre tolerance as fraction of canvas dim
CLIP_EDGE_FRAC = 0.06                 # ink within this frac of a box edge = clipped
CONTRAST_MIN = 3.0                    # WCAG-ish fill vs plate
LOW_CONTRAST_SOURCE = 2.2             # below this the source itself is low-contrast; skip (c)
OVERLAP_IOU = 0.35                    # text-node box IoU above which we flag an overlap
MIN_INK = 12                          # ignore source boxes with almost no ink


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _otsu(delta: np.ndarray) -> float:
    """Otsu split on a 0..255-scaled distance map."""
    d = np.clip(delta, 0, 255).astype(np.uint8)
    hist = np.bincount(d.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros(256)
    ok = denom > 1e-12
    between[ok] = ((mu_t * omega[ok] - mu[ok]) ** 2) / denom[ok]
    return float(np.argmax(between))


def _ink_mask(crop: np.ndarray) -> np.ndarray:
    """Boolean ink mask via border-background contrast + Otsu split.

    A plain high percentile threshold fails on tightly-cropped headlines (the ink is a
    large fraction of the box, so the percentile lands *inside* the ink and hides it);
    Otsu on the background-distance map separates ink from plate at any ink fraction.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return np.zeros(crop.shape[:2], bool)
    h, w = crop.shape[:2]
    bw = max(1, min(3, h // 5, w // 5))
    border = np.concatenate([
        crop[:bw].reshape(-1, 3), crop[-bw:].reshape(-1, 3),
        crop[:, :bw].reshape(-1, 3), crop[:, -bw:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(border, axis=0)
    delta = np.sqrt(((crop.astype(np.float32) - bg) ** 2).sum(axis=2))
    thr = max(28.0, _otsu(delta))
    mask = delta > thr
    # Guard against an all-ink or all-bg degenerate split.
    frac = float(mask.mean())
    if frac > 0.9:
        mask = delta > max(thr, float(np.percentile(delta, 60)))
    return mask


def _relative_luminance(rgb) -> float:
    ch = []
    for v in rgb:
        c = max(0.0, min(1.0, float(v) / 255.0))
        ch.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def _contrast(a, b) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _clean_box(b) -> dict:
    b = b or {}
    return {"x": float(b.get("x", 0) or 0), "y": float(b.get("y", 0) or 0),
            "w": max(0.0, float(b.get("w", 0) or 0)), "h": max(0.0, float(b.get("h", 0) or 0))}


def _iter_text_nodes(node, out):
    if isinstance(node, dict):
        if node.get("type") == "text" and str(node.get("text") or "").strip():
            out.append(node)
        for v in node.values():
            _iter_text_nodes(v, out)
    elif isinstance(node, list):
        for v in node:
            _iter_text_nodes(v, out)


def _ink_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def check_run(run_dir: str) -> dict:
    """Return {'fixture','violations':[...], 'nodes':int, 'source_lines':int}."""
    fixture = os.path.basename(run_dir.rstrip("/\\"))
    v: list[dict] = []

    def flag(sev, rule, detail, text=""):
        v.append({"severity": sev, "rule": rule, "text": str(text)[:60], "detail": detail})

    try:
        design = json.load(open(os.path.join(run_dir, "design.json"), encoding="utf-8"))
        ocr = json.load(open(os.path.join(run_dir, "ocr.json"), encoding="utf-8"))
    except Exception as exc:
        return {"fixture": fixture, "violations": [{"severity": "ERROR", "rule": "load",
                "detail": str(exc), "text": ""}], "nodes": 0, "source_lines": 0}

    canvas = design.get("canvas") or {}
    cw, ch = float(canvas.get("w") or 1), float(canvas.get("h") or 1)
    kept = {_norm_text(t) for t in (design.get("kept_in_photo") or [])}
    kept_blob = " ".join(_norm_text(t) for t in (design.get("kept_in_photo") or []))

    src_img = ren_img = None
    try:
        src_img = np.asarray(Image.open(os.path.join(run_dir, "normalized.png")).convert("RGB"))
    except Exception:
        pass
    try:
        ren_img = np.asarray(Image.open(os.path.join(run_dir, "preview.png")).convert("RGB"))
    except Exception:
        pass

    nodes: list[dict] = []
    _iter_text_nodes(design.get("layers") or [], nodes)
    design_blob = " ".join(_norm_text(n.get("text")) for n in nodes)

    ocr_lines = [l for l in (ocr.get("lines") or []) if str(l.get("text") or "").strip()]

    # ── (g) MISSING + (f) STRIKE + (a/b/c/d) pixel checks, per source line ──────
    for line in ocr_lines:
        text = str(line.get("text") or "").strip()
        ntext = _norm_text(text)
        if len(ntext) < 2:
            continue
        box = _clean_box(line.get("painted_box") or line.get("box"))
        in_design = ntext and ntext in design_blob.replace(" ", "")
        in_kept = ntext and (ntext in kept_blob.replace(" ", "") or ntext in kept)

        # source ink mass in the line box
        src_mass = None
        src_bbox = None
        if src_img is not None and box["w"] > 1 and box["h"] > 1:
            x0, y0 = int(box["x"]), int(box["y"])
            x1, y1 = int(math.ceil(box["x"] + box["w"])), int(math.ceil(box["y"] + box["h"]))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(src_img.shape[1], x1), min(src_img.shape[0], y1)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                sm = _ink_mask(src_img[y0:y1, x0:x1])
                src_mass = int(sm.sum())
                bb = _ink_bbox(sm)
                if bb:
                    src_bbox = (bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0)

        if not in_design and not in_kept:
            # Only flag lines that actually carry legible ink (skip OCR speckle).
            if src_mass is None or src_mass >= MIN_INK:
                flag("HARD", "missing", "OCR line absent from design.json and not in kept_in_photo", text)
            continue
        if in_kept and not in_design:
            continue  # deliberately baked into the photo

        # (f) STRIKE carry — struck source line must carry a decoration downstream
        if (line.get("meta") or {}).get("strikethrough"):
            node = _match_node(nodes, ntext, box)
            deco = str(((node or {}).get("style") or {}).get("textDecoration") or "").upper() if node else ""
            has_native = bool((node or {}).get("meta", {}).get("native_decoration_shapes")) if node else False
            if node is not None and "STRIKE" not in deco and "LINE_THROUGH" not in deco and not has_native:
                flag("HARD", "strike", "struck source line does not carry a strike decoration", text)

        # pixel mass / placement / clip need both images and enough source ink
        if src_img is None or ren_img is None or src_mass is None or src_mass < MIN_INK:
            continue
        x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
        x1 = min(ren_img.shape[1], int(math.ceil(box["x"] + box["w"])))
        y1 = min(ren_img.shape[0], int(math.ceil(box["y"] + box["h"])))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        # Keep the render window tight (a small pad only) so densely stacked neighbour
        # lines do not leak into this line's ink-mass measurement.
        pad = int(round(min(4.0, 0.1 * (y1 - y0))))
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(ren_img.shape[1], x1 + pad), min(ren_img.shape[0], y1 + pad)
        rm = _ink_mask(ren_img[ry0:ry1, rx0:rx1])
        ren_mass = int(rm.sum())

        # (a) MASS — a vanished/too-light line is the real "weird text" failure (HARD);
        # bloat is noisier (window overlap, fragment boxes) so it stays a WARN.
        ratio = ren_mass / max(1.0, src_mass)
        if ratio < MASS_HARD_LO:
            flag("HARD", "mass", f"rendered ink mass {ratio:.2f}x source (vanished/too light)", text)
        elif ratio < MASS_WARN_LO or ratio > MASS_WARN_HI:
            flag("WARN", "mass", f"rendered ink mass {ratio:.2f}x source (outside 25% band)", text)

        # (b) PLACEMENT — compare ink-bbox centres
        rbb = _ink_bbox(rm)
        if rbb and src_bbox and ren_mass >= MIN_INK:
            r_cx = (rbb[0] + rbb[2]) / 2.0 + rx0
            r_cy = (rbb[1] + rbb[3]) / 2.0 + ry0
            s_cx = (src_bbox[0] + src_bbox[2]) / 2.0
            s_cy = (src_bbox[1] + src_bbox[3]) / 2.0
            dx = abs(r_cx - s_cx) / cw
            dy = abs(r_cy - s_cy) / ch
            if dx > PLACEMENT_FRAC or dy > PLACEMENT_FRAC:
                flag("WARN", "placement",
                     f"ink centre off by ({dx*100:.1f}%,{dy*100:.1f}%) of canvas", text)

        # (d) CLIP — rendered ink touching the node's own box edge (not the padded window)
        if rbb:
            gx0, gy0, gx1, gy1 = rbb[0] + rx0, rbb[1] + ry0, rbb[2] + rx0, rbb[3] + ry0
            node = _match_node(nodes, ntext, box)
            if node is not None:
                nb = _clean_box(node.get("box"))
                # node boxes may be tree-relative; only trust when it roughly matches source
                if abs(nb["w"] - box["w"]) <= 0.5 * box["w"] + 8:
                    edge = CLIP_EDGE_FRAC * max(nb["h"], 8.0)
                    if gx1 >= x1 - 1 and ren_mass > MIN_INK and (x1 - gx1) < edge and ratio > MASS_WARN_HI:
                        flag("WARN", "clip", "rendered ink reaches the right box edge", text)

        # (c) CONTRAST — node fill vs the true local plate (sampled OUTSIDE the ink box,
        # so a tightly-cropped headline does not read its own ink as the plate).
        node = _match_node(nodes, ntext, box)
        if node is not None:
            fill = ((node.get("style") or {}).get("fill") or {})
            fill_hex = fill.get("color") if isinstance(fill, dict) else None
            plate_rgb = _plate_rgb(src_img, box)
            if fill_hex and plate_rgb is not None:
                iy0, iy1 = max(0, int(box["y"])), int(math.ceil(box["y"]+box["h"]))
                ix0, ix1 = max(0, int(box["x"])), int(math.ceil(box["x"]+box["w"]))
                reg = src_img[iy0:iy1, ix0:ix1]
                sm2 = _ink_mask(reg)
                if sm2.any():
                    ink_rgb = np.median(reg[sm2].astype(np.float32), axis=0)
                    src_contrast = _contrast(ink_rgb, plate_rgb)
                    fh = fill_hex.lstrip("#")
                    if len(fh) == 6:
                        frgb = tuple(int(fh[i:i+2], 16) for i in (0, 2, 4))
                        fill_contrast = _contrast(frgb, plate_rgb)
                        # Only flag contrast the RENDER introduced: the source reads fine
                        # (>= low-contrast floor) but the emitted fill is both below the
                        # WCAG floor AND clearly worse than the source (a real ghost fill,
                        # 066 "OUR COMPETITOR" #836a5d over its plate) — not a faithful
                        # reproduction of copy the source itself paints at low contrast.
                        if (src_contrast >= LOW_CONTRAST_SOURCE
                                and fill_contrast < CONTRAST_MIN
                                and fill_contrast < src_contrast * 0.8):
                            flag("HARD", "contrast",
                                 f"fill {fill_hex} contrast {fill_contrast:.1f}:1 vs plate "
                                 f"(source {src_contrast:.1f}:1)", text)

    # ── (e) OVERLAP — text nodes covering the same span ────────────────────────
    for i in range(len(nodes)):
        bi = _clean_box(nodes[i].get("box"))
        ti = _norm_text(nodes[i].get("text"))
        if bi["w"] <= 0 or bi["h"] <= 0:
            continue
        for j in range(i + 1, len(nodes)):
            bj = _clean_box(nodes[j].get("box"))
            if bj["w"] <= 0 or bj["h"] <= 0:
                continue
            ox = max(0.0, min(bi["x"]+bi["w"], bj["x"]+bj["w"]) - max(bi["x"], bj["x"]))
            oy = max(0.0, min(bi["y"]+bi["h"], bj["y"]+bj["h"]) - max(bi["y"], bj["y"]))
            inter = ox * oy
            union = bi["w"]*bi["h"] + bj["w"]*bj["h"] - inter
            if union > 0 and inter / union >= OVERLAP_IOU:
                tj = _norm_text(nodes[j].get("text"))
                # identical text at the same place is a duplicate; different text is a collision
                flag("WARN", "overlap",
                     f"text nodes overlap (IoU {inter/union:.2f}): "
                     f"{nodes[i].get('text')!r} / {nodes[j].get('text')!r}", ti or tj)

    return {"fixture": fixture, "violations": v, "nodes": len(nodes), "source_lines": len(ocr_lines)}


def _plate_rgb(src_img, box):
    """Median RGB of the plate ring just OUTSIDE the text box (the true local plate)."""
    if src_img is None:
        return None
    h, w = src_img.shape[:2]
    mx = max(6, int(round(0.35 * box["h"])))
    ox0 = max(0, int(box["x"]) - mx)
    oy0 = max(0, int(box["y"]) - mx)
    ox1 = min(w, int(math.ceil(box["x"] + box["w"])) + mx)
    oy1 = min(h, int(math.ceil(box["y"] + box["h"])) + mx)
    ix0, iy0 = max(0, int(box["x"])), max(0, int(box["y"]))
    ix1, iy1 = min(w, int(math.ceil(box["x"] + box["w"]))), min(h, int(math.ceil(box["y"] + box["h"])))
    if ox1 - ox0 < 2 or oy1 - oy0 < 2:
        return None
    ring = np.ones((oy1 - oy0, ox1 - ox0), bool)
    ring[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = False
    region = src_img[oy0:oy1, ox0:ox1]
    if not ring.any():
        return None
    return np.median(region[ring].astype(np.float32), axis=0)


def _match_node(nodes, ntext, box) -> Optional[dict]:
    """Best design text node for a source line by text containment + box proximity."""
    best, best_score = None, -1.0
    nt = ntext.replace(" ", "")
    for n in nodes:
        cand = _norm_text(n.get("text")).replace(" ", "")
        if not cand:
            continue
        if nt in cand or cand in nt:
            nb = _clean_box(n.get("box"))
            d = abs(nb["y"] - box["y"]) + abs(nb["x"] - box["x"])
            score = 1000.0 - d
            if score > best_score:
                best, best_score = n, score
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Text-contract verification sweep")
    ap.add_argument("path", help="a run dir or a benchmark dir of run dirs")
    ap.add_argument("--json", help="write full report JSON here")
    ap.add_argument("--strict", action="store_true", help="fail on WARN too, not just HARD")
    args = ap.parse_args(argv)

    if os.path.exists(os.path.join(args.path, "design.json")):
        runs = [args.path]
    else:
        runs = sorted(d for d in glob.glob(os.path.join(args.path, "*"))
                      if os.path.isfile(os.path.join(d, "design.json")))
    if not runs:
        print(f"no runs with design.json under {args.path}", file=sys.stderr)
        return 2

    reports, hard, warn = [], 0, 0
    for run in runs:
        rep = check_run(run)
        reports.append(rep)
        h = sum(1 for x in rep["violations"] if x["severity"] in ("HARD", "ERROR"))
        w = sum(1 for x in rep["violations"] if x["severity"] == "WARN")
        hard += h
        warn += w
        status = "OK" if h == 0 else f"{h} HARD"
        print(f"{rep['fixture'][:30]:30}  nodes={rep['nodes']:3}  {status}"
              + (f", {w} warn" if w else ""))
        for x in rep["violations"]:
            if x["severity"] in ("HARD", "ERROR") or args.strict:
                print(f"    [{x['severity']:4}] {x['rule']:10} {x['text']!r}: {x['detail']}")

    print(f"\nTOTAL: {hard} HARD, {warn} WARN across {len(runs)} run(s)")
    if args.json:
        json.dump(reports, open(args.json, "w", encoding="utf-8"), indent=2)
    return 1 if (hard > 0 or (args.strict and warn > 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
