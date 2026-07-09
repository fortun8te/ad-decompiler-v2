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
    """Minimal port of the routing intent. Real routing.py supersedes this."""
    src = candidate.get("meta", {}).get("source")
    kind = candidate.get("meta", {}).get("kind")
    if src == "ocr":
        candidate["target"] = "text"
    elif kind == "shape":
        candidate["target"] = "shape"
    elif kind == "icon":
        candidate["target"] = "icon"
    elif kind in ("photo-fragment", "photo", "image"):
        candidate["target"] = "image"
    else:
        candidate["target"] = candidate.get("target", "image")
    return candidate


# ── candidate builders ───────────────────────────────────────────────────────────────
def _text_candidate(line):
    return {
        "id": f"c_{line['id']}",
        "box": dict(line["box"]),
        "z": 0,
        "text": line.get("text", ""),
        "style": {},
        "meta": {
            "source": "ocr",
            "role": "text",
            "confidence": round(float(line.get("conf", 1.0)), 4),
            "ocr_id": line["id"],
        },
    }


def _element_candidate(el):
    kind = el.get("kind", "shape")
    role = {"shape": "shape", "icon": "icon", "photo-fragment": "photo"}.get(kind, "shape")
    return {
        "id": f"c_{el['id']}",
        "box": dict(el["box"]),
        "z": 0,
        # box-local mask written by element_detect at elements/<id>.png (by convention)
        "mask": {"kind": "alpha", "src": os.path.join("elements", f"{el['id']}.png")},
        "source_crop": {"element_id": el["id"]},
        "meta": {
            "source": "element",
            "role": role,
            "kind": kind,
            "confidence": round(float(el.get("coverage", 0.0)), 4),
            "element_id": el["id"],
            "area": el.get("area"),
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
    photo_inside = float((cfg.get("merge") or {}).get("photo_inside_frac", 0.6))

    ocr_lines = ocr.get("lines", []) if isinstance(ocr, dict) else (ocr or [])
    elements = elements or []
    qwen = qwen or []

    text_cands = [_text_candidate(l) for l in ocr_lines]
    elem_cands = [_element_candidate(e) for e in elements]

    # ── qwen z-order + alpha: match each qwen layer to overlapping candidates ──────────
    # qwen list is back-to-front; index = z (lower index = further back).
    photo_regions = []  # regions where scene text should stay baked in
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
            if best["meta"].get("role") == "photo":
                photo_regions.append(qbox)
        else:
            # qwen-only layer with no element match -> raw image candidate
            elem_cands.append(
                {
                    "id": f"c_{ql['id']}",
                    "box": dict(qbox),
                    "z": zi,
                    "src": ql.get("png"),
                    "meta": {
                        "source": "qwen",
                        "role": "image",
                        "kind": "image",
                        "confidence": 0.5,
                        "qwen_id": ql.get("id"),
                    },
                }
            )
    # also treat detected photo-fragments as photo regions for scene-text detection
    for c in elem_cands:
        if c["meta"].get("role") == "photo":
            photo_regions.append(c["box"])

    # ── assemble + route ──────────────────────────────────────────────────────────────
    candidates = text_cands + elem_cands

    # scene text: OCR line inside a photo region -> keep baked in the base
    for c in text_cands:
        for pr in photo_regions:
            if _inside_frac(c["box"], pr) >= photo_inside:
                c["meta"]["kept_in_photo"] = True
                c["meta"]["role"] = "scene-text"
                break

    for c in candidates:
        try:
            route(c, canvas, cfg)
        except Exception as e:  # a bad router must not sink the whole stage
            print(f"[merge] routing error on {c.get('id')}: {e}; defaulting")
            _fallback_route(c, canvas, cfg)
        # enforce scene-text -> drop regardless of router (contract hard rule)
        if c["meta"].get("kept_in_photo"):
            c["target"] = "drop"

    # ── dedup: shape/icon that is really an OCR text box -> drop (prefer text) ─────────
    kept = []
    for c in candidates:
        if c["meta"].get("source") in ("element", "element+qwen") and c.get("target") in (
            "shape",
            "icon",
        ):
            covered = any(
                t.get("target") == "text"
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
