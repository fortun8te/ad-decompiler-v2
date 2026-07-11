"""merge_layers.py — stage 6: fuse OCR + element_detect + Qwen into routed candidates.

merge(ocr, elements, qwen, canvas, cfg) turns the three evidence streams into ONE
candidate list that build_design_json can consume:

  * every OCR line          -> a text candidate (crisp text + box)
  * every detected element  -> a shape / icon / photo candidate (crisp box + mask)
  * every Qwen layer        -> supplies back-to-front z-order + clean RGBA alpha,
                               matched to overlapping OCR/element candidates by IoU

Each candidate is passed through routing.route(candidate, canvas, cfg), which sets
`target` in {text, shape, image, icon, drop}. routing.py is owned by another builder;
if it is not importable yet we fall back to a conservative inline router (used ONLY at
runtime, never written to disk).

Post-routing cleanup:
  * dedup: a shape/icon candidate that is essentially an OCR text box is dropped
    (prefer the editable text candidate)
  * scene text: an OCR line sitting inside a photo region is flagged
    meta.kept_in_photo=True and routed to `drop` (it stays baked into the base)

Candidates carry everything downstream needs: box, target, z, and either
text/style, or src(alpha png)+mask, or source_crop, plus meta{source,role,confidence}.
"""
from __future__ import annotations
import importlib
import os
from typing import Optional


def _load_routing():
    """Return the real routing.route, or a conservative inline fallback."""
    for name in ("src.routing", "routing"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "route"):
                return mod.route, True
        except ImportError:
            continue
    return _fallback_route, False


def _iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def _inside_frac(a, b):
    """Fraction of a inside b."""
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1.0, a["w"] * a["h"])


# ── conservative inline router (fallback only) ───────────────────────────────────────
def _fallback_route(candidate: dict, canvas: dict, cfg: dict) -> dict:
    """Minimal port of routing intent, used ONLY if src.routing is unimportable.
    Reads the same top-level fields the real router does. Returns a copy."""
    c = dict(candidate)
    meta = c.setdefault("meta", {})
    kind = c.get("kind")
    if c.get("text") is not None and kind in (None, "text"):
        if c.get("kept_in_photo") or meta.get("origin") == "scene":
            c["target"] = "drop"
            meta["kept_in_photo"] = True
        else:
            c["target"] = "text"
    elif kind == "photo-fragment" or meta.get("role") in ("photo", "product", "person"):
        c["target"] = "image"
    elif kind == "icon" or meta.get("role") in ("icon", "badge", "logo", "arrow"):
        c["target"] = "icon"
    elif kind == "shape":
        c["target"] = "shape"
    else:
        c["target"] = "image"
        meta["fallback"] = True
    return c


# ── candidate builders ───────────────────────────────────────────────────────────────
def _text_candidate(line):
    meta = dict(line.get("meta") or {})
    meta.update({
        "source": "ocr",
        "role": line.get("role") or meta.get("role") or "text",
        "confidence": round(float(line.get("conf", meta.get("confidence", 1.0))), 4),
        "ocr_id": line["id"],
        "line_ids": line.get("line_ids") or [line["id"]],
        "quad": line.get("quad"),
        "hierarchy": line.get("hierarchy"),
        "style_id": line.get("repeated_style_id") or line.get("style_id"),
    })
    return {
        "id": f"c_{line['id']}",
        "box": dict(line["box"]),
        "z": 0,
        "text": line.get("text", ""),
        "style": dict(line.get("style") or {}),
        "visible_box": dict(line.get("ink_box") or line.get("painted_box") or line["box"]),
        "rotation": float(line.get("rotation", 0.0) or 0.0),
        "quad": line.get("quad"),
        "meta": meta,
    }


def _text_sources(ocr):
    if not isinstance(ocr, dict):
        return ocr or []
    blocks = ocr.get("blocks") or []
    if not blocks:
        return ocr.get("lines", [])
    styles = {style.get("id"): style for style in (ocr.get("styles") or [])}
    lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    out = []
    for raw in blocks:
        block = dict(raw)
        members = [lines[line_id] for line_id in block.get("line_ids", []) if line_id in lines]
        style_id = block.get("style_id")
        block_style = dict(block.get("style") or styles.get(style_id) or
                           (members[0].get("style") if members else {}) or {})
        if members:
            representative = members[0].get("style") or {}
            for key in ("fontCandidates", "fontSizeCandidates", "fontWeightCandidates",
                        "fontStyleCandidates", "confidence"):
                if key in representative:
                    block_style.setdefault(key, representative[key])
        block["style"] = block_style
        block["ink_box"] = block.get("painted_box") or block.get("box")
        block["conf"] = (sum(float(line.get("conf", 1)) for line in members) / len(members)
                         if members else float(block.get("conf", 1)))
        block["rotation"] = (sum(float(line.get("rotation", 0)) for line in members) / len(members)
                             if members else float(block.get("rotation", 0)))
        block["repeated_style_id"] = style_id
        out.append(block)
    return out


def _element_candidate(el):
    kind = el.get("kind", "shape")  # top-level kind -> routing.route reads this
    role = el.get("role") or {"shape": "shape", "icon": "icon", "photo-fragment": "photo"}.get(kind, "shape")
    raw_mask = el.get("mask")
    mask_src = (raw_mask.get("src") if isinstance(raw_mask, dict) else raw_mask)
    mask_src = mask_src or el.get("mask_src") or el.get("mask_path")
    return {
        "id": f"c_{el['id']}",
        "box": dict(el["box"]),
        "z": 0,
        "kind": kind,
        # box-local mask written by element_detect at elements/<id>.png (by convention)
        "mask": {
            "kind": "alpha",
            "src": mask_src or os.path.join("elements", f"{el['id']}.png"),
        },
        "src": el.get("asset_src"),
        "source_crop": {"element_id": el["id"]},
        "meta": {
            "source": "element",
            "role": role,
            "kind": kind,
            "confidence": round(float(el.get("score", el.get("coverage", 0.0))), 4),
            "element_id": el["id"],
            "area": el.get("area"),
            "prompt": el.get("prompt"),
            "parent_id": el.get("parent_id"),
            "observations": el.get("observation_ids") or [],
            "provenance": el.get("provenance") or {},
        },
    }


# ── public API ───────────────────────────────────────────────────────────────────────
def merge(ocr, elements, qwen, canvas, cfg: Optional[dict] = None, run_dir=None):
    cfg = cfg or {}
    route, real = _load_routing()
    if run_dir is None:
        run_dir = cfg.get("run_dir")
    dedup_iou = float((cfg.get("merge") or {}).get("dedup_iou", 0.6))
    match_iou = float((cfg.get("merge") or {}).get("qwen_match_iou", 0.3))
    photo_inside = float((cfg.get("merge") or {}).get("photo_inside_frac", 0.82))
    scene_roles = set((cfg.get("merge") or {}).get(
        "scene_text_roles", ["product", "package", "bottle", "jar", "tube", "device", "sign"]
    ))
    overlay_text_roles = {"headline", "title", "subtitle", "subheadline", "eyebrow",
                          "cta", "button", "price", "offer"}

    # text_analysis emits paragraph/headline blocks. Prefer those over one Figma node per
    # OCR line so wrapping, hierarchy, and repeated text styles survive downstream.
    ocr_lines = _text_sources(ocr)
    elements = elements or []
    qwen = qwen or []

    text_cands = [_text_candidate(l) for l in ocr_lines]
    elem_cands = [_element_candidate(e) for e in elements]

    # ── qwen z-order + alpha: match each qwen layer to overlapping candidates ──────────
    # qwen list is back-to-front; index = z (lower index = further back).
    scene_regions = []  # product/package regions where printed text should stay baked in
    for zi, ql in enumerate(qwen):
        qbox = ql.get("box", {})
        if not (qbox.get("w", 0) > 0 and qbox.get("h", 0) > 0):
            continue
        # best-matching element candidate gets this layer's clean alpha + z
        best = None
        best_iou = match_iou
        for c in elem_cands:
            iou = _iou(c["box"], qbox)
            if iou > best_iou:
                best_iou, best = iou, c
        if best is not None:
            best["z"] = zi
            best["src"] = ql.get("png")  # clean RGBA alpha from qwen
            best["meta"]["qwen_id"] = ql.get("id")
            best["meta"]["source"] = "element+qwen"
            if best["meta"].get("role") in scene_roles:
                scene_regions.append(qbox)
        else:
            # qwen-only layer with no element match -> raster image candidate.
            # meta.role='photo' makes routing.route pick target='image' + alpha mask.
            elem_cands.append(
                {
                    "id": f"c_{ql['id']}",
                    "box": dict(qbox),
                    "z": zi,
                    "src": ql.get("png"),
                    "meta": {
                        "source": "qwen",
                        "role": "photo",
                        "confidence": 0.5,
                        "qwen_id": ql.get("id"),
                    },
                }
            )
    # Only product/package-like rasters own printed scene text. Generic photos/backgrounds
    # often have intentional editable overlay copy, so containment alone is not enough.
    for c in elem_cands:
        if c["meta"].get("role") in scene_roles or c["meta"].get("contains_scene_text"):
            scene_regions.append(c["box"])

    # ── assemble + route ──────────────────────────────────────────────────────────────
    candidates = text_cands + elem_cands

    # scene text: OCR line inside a photo region -> keep baked in the base.
    # Flag with top-level kept_in_photo + meta.origin='scene' so routing.route drops it.
    for c in text_cands:
        if c["meta"].get("role") in overlay_text_roles:
            continue
        for pr in scene_regions:
            if _inside_frac(c["box"], pr) >= photo_inside:
                c["kept_in_photo"] = True
                c["meta"]["origin"] = "scene"
                c["meta"]["role"] = "scene-text"
                break

    # routing.route returns a NEW dict (shallow copy) — capture it.
    routed = []
    for c in candidates:
        try:
            rc = route(c, canvas, cfg)
        except Exception as e:  # a bad router must not sink the whole stage
            print(f"[merge] routing error on {c.get('id')}: {e}; defaulting")
            rc = _fallback_route(c, canvas, cfg)
        # enforce scene-text -> drop regardless of router (contract hard rule)
        if rc.get("kept_in_photo") or rc.get("meta", {}).get("kept_in_photo"):
            rc["target"] = "drop"
        routed.append(rc)
    candidates = routed
    text_cands = [c for c in candidates if c["meta"].get("source") == "ocr"]

    # ── dedup: shape/icon that is really an OCR text box -> drop (prefer text) ─────────
    kept = []
    for c in candidates:
        if c["meta"].get("source") in ("element", "element+qwen") and c.get("target") in (
            "shape",
            "icon",
        ):
            covered = any(
                t.get("target") != "drop"
                and _inside_frac(t["box"], c["box"]) >= 0.9
                and _iou(t["box"], c["box"]) >= dedup_iou
                for t in text_cands
            )
            if covered:
                continue  # the "shape" is just the text's bounding box
        kept.append(c)

    # stable z: keep qwen-derived z, then order remaining by area (large=back)
    def _area(c):
        return c["box"]["w"] * c["box"]["h"]

    max_z = max((c["z"] for c in kept), default=0)
    for c in kept:
        if c["z"] == 0 and c["meta"].get("source") == "ocr":
            c["z"] = max_z + 1  # text sits above shapes by default

    kept.sort(key=lambda c: (c["z"], -_area(c)))

    if run_dir:
        try:
            schema = importlib.import_module("src.schema")
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            schema = importlib.import_module("schema")
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(kept, os.path.join(run_dir, "merged.json"))

    return kept


if __name__ == "__main__":  # CPU-safe smoke with fixtures
    ocr = {
        "lines": [
            {"id": "L0", "text": "BIG SALE", "conf": 0.98,
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}},
            {"id": "L1", "text": "Model wears watch", "conf": 0.8,
             "box": {"x": 300, "y": 400, "w": 180, "h": 24}},
        ]
    }
    elements = [
        {"id": "E0", "box": {"x": 20, "y": 20, "w": 260, "h": 90}, "kind": "shape",
         "area": 20000, "coverage": 0.1, "source": "residual-cc"},
        {"id": "E1", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "kind": "photo-fragment", "area": 40000, "coverage": 0.2,
         "source": "residual-cc"},
    ]
    qwen = [
        {"id": "Q0", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "png": "qwen_layers/Q0.png", "kind_hint": "photo"},
    ]
    cands = merge(ocr, elements, qwen, {"w": 600, "h": 600}, {})
    for c in cands:
        print(c["id"], "target=", c.get("target"), "z=", c["z"],
              "role=", c["meta"].get("role"), "kip=", c["meta"].get("kept_in_photo"))
