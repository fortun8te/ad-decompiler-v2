"""ocr.py — stage 2: text detection + recognition, unified into schema.OcrResult.

Primary backend: PaddleOCR PP-OCRv6 (paddleocr>=3, angle classification on).
Challengers (selectable via cfg.ocr): Surya, docTR. Tesseract is an explicit
baseline/fallback only. Every backend is normalized into schema.OcrLine/OcrWord:

  * quad -> axis-aligned box
  * reading order: rows grouped at half the median line height (top-to-bottom),
    left-to-right within a row; line ids "L0".. in that order
  * when cfg.ocr.challengers is set, challengers run too and lines are reconciled
    by IoU clustering: the highest-confidence line per cluster wins; clusters where
    backends disagree on text are confidence-penalized and flagged in line.meta

Results are cached by image sha256 under runs/.cache/ocr/<sha>.<engine>.json.

All heavy OCR deps are imported lazily with a helpful ImportError. Device comes
from cfg.device ('cuda'|'cpu').
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
import time
from typing import Optional


def _load_schema():
    for name in ("src.schema", "schema"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("schema")


# ── geometry helpers ────────────────────────────────────────────────────────────────
def _quad_to_box(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = min(xs), min(ys)
    return {
        "x": float(x0),
        "y": float(y0),
        "w": float(max(xs) - x0),
        "h": float(max(ys) - y0),
    }


def _iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def _rect_quad(box):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


# ── reading order + id assignment ────────────────────────────────────────────────────
def _order_lines(raw_lines):
    """raw_lines: [{text,conf,box,quad,words,meta?}] -> ordered + id'd list."""
    if not raw_lines:
        return []
    heights = sorted(l["box"]["h"] for l in raw_lines)
    med_h = heights[len(heights) // 2] or 1.0
    row_tol = med_h * 0.5
    # sort by y first to form rows greedily
    by_y = sorted(raw_lines, key=lambda l: l["box"]["y"])
    rows = []
    for l in by_y:
        cy = l["box"]["y"] + l["box"]["h"] / 2.0
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= row_tol:
                row["items"].append(l)
                # rolling centroid keeps rows stable
                row["cy"] = (row["cy"] * (len(row["items"]) - 1) + cy) / len(
                    row["items"]
                )
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "items": [l]})
    rows.sort(key=lambda r: r["cy"])
    ordered = []
    for row in rows:
        for l in sorted(row["items"], key=lambda x: x["box"]["x"]):
            ordered.append(l)
    for i, l in enumerate(ordered):
        l["id"] = f"L{i}"
    return ordered


# ── caching ───────────────────────────────────────────────────────────────────────
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_paths(img_path, engine, cfg):
    root = (cfg.get("cache_dir")
            or os.path.join(cfg.get("runs_dir", "runs"), ".cache", "ocr"))
    os.makedirs(root, exist_ok=True)
    sha = _sha256(img_path)
    return os.path.join(root, f"{sha}.{engine}.json"), sha


# ── backends: each returns a list of raw line dicts {text,conf,box,quad,words} ────────
def _paddle(img_path, cfg):
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PP-OCRv6 backend requires paddleocr>=3 and paddlepaddle-gpu.\n"
            "  pip install paddleocr>=3 paddlepaddle-gpu"
        ) from e
    ocr_cfg = cfg.get("ocr") or {}
    lang = ocr_cfg.get("lang", "en")
    use_gpu = str(cfg.get("device", "cpu")).startswith("cuda")
    # paddleocr>=3 auto-selects PP-OCRv5/v6 pipelines; pass device explicitly.
    engine = PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        device="gpu" if use_gpu else "cpu",
        show_log=False,
    )
    result = engine.ocr(img_path, cls=True)
    lines = []
    # paddleocr returns [[ [quad, (text, conf)], ... ]] (per-image list)
    pages = result if result and isinstance(result[0], list) else [result]
    for page in pages:
        if not page:
            continue
        for det in page:
            try:
                quad, (text, conf) = det[0], det[1]
            except (ValueError, TypeError):
                continue
            quad = [[float(p[0]), float(p[1])] for p in quad]
            lines.append(
                {
                    "text": text,
                    "conf": float(conf),
                    "box": _quad_to_box(quad),
                    "quad": quad,
                    "words": [],
                }
            )
    return lines, "ppocr-v6"


def _surya(img_path, cfg):
    try:
        from PIL import Image
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Surya backend requires surya-ocr.  pip install surya-ocr"
        ) from e
    langs = [(cfg.get("ocr") or {}).get("lang", "en")]
    image = Image.open(img_path).convert("RGB")
    det = DetectionPredictor()
    rec = RecognitionPredictor()
    preds = rec([image], [langs], det)
    lines = []
    for page in preds:
        for tl in page.text_lines:
            bbox = tl.bbox  # [x0,y0,x1,y1]
            box = {
                "x": float(bbox[0]),
                "y": float(bbox[1]),
                "w": float(bbox[2] - bbox[0]),
                "h": float(bbox[3] - bbox[1]),
            }
            lines.append(
                {
                    "text": tl.text,
                    "conf": float(getattr(tl, "confidence", 1.0) or 1.0),
                    "box": box,
                    "quad": _rect_quad(box),
                    "words": [],
                }
            )
    return lines, "surya"


def _doctr(img_path, cfg):
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "docTR backend requires python-doctr.  pip install python-doctr[torch]"
        ) from e
    doc = DocumentFile.from_images(img_path)
    predictor = ocr_predictor(pretrained=True)
    out = predictor(doc)
    lines = []
    for page in out.pages:
        ph, pw = page.dimensions  # (h, w), relative coords -> absolute
        for block in page.blocks:
            for line in block.lines:
                words = line.words
                if not words:
                    continue
                confs = [w.confidence for w in words]
                (x0, y0), (x1, y1) = line.geometry
                box = {
                    "x": x0 * pw,
                    "y": y0 * ph,
                    "w": (x1 - x0) * pw,
                    "h": (y1 - y0) * ph,
                }
                wds = []
                for w_ in words:
                    (wx0, wy0), (wx1, wy1) = w_.geometry
                    wbox = {
                        "x": wx0 * pw,
                        "y": wy0 * ph,
                        "w": (wx1 - wx0) * pw,
                        "h": (wy1 - wy0) * ph,
                    }
                    wds.append(
                        {
                            "text": w_.value,
                            "conf": float(w_.confidence),
                            "box": wbox,
                            "quad": _rect_quad(wbox),
                        }
                    )
                lines.append(
                    {
                        "text": " ".join(w_.value for w_ in words),
                        "conf": float(sum(confs) / len(confs)),
                        "box": box,
                        "quad": _rect_quad(box),
                        "words": wds,
                    }
                )
    return lines, "doctr"


def _tesseract(img_path, cfg):
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Tesseract backend requires pytesseract + the tesseract binary.\n"
            "  pip install pytesseract   (and: brew/choco install tesseract)"
        ) from e
    lang = (cfg.get("ocr") or {}).get("lang", "en")
    data = pytesseract.image_to_data(
        Image.open(img_path), lang=lang, output_type=pytesseract.Output.DICT
    )
    # group words into lines by (block,par,line)
    groups = {}
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        if not txt or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        wbox = {
            "x": float(data["left"][i]),
            "y": float(data["top"][i]),
            "w": float(data["width"][i]),
            "h": float(data["height"][i]),
        }
        groups.setdefault(key, []).append((txt, conf / 100.0, wbox))
    lines = []
    for key, words in groups.items():
        xs0 = min(w[2]["x"] for w in words)
        ys0 = min(w[2]["y"] for w in words)
        xs1 = max(w[2]["x"] + w[2]["w"] for w in words)
        ys1 = max(w[2]["y"] + w[2]["h"] for w in words)
        box = {"x": xs0, "y": ys0, "w": xs1 - xs0, "h": ys1 - ys0}
        wds = [
            {"text": t, "conf": c, "box": b, "quad": _rect_quad(b)}
            for t, c, b in words
        ]
        lines.append(
            {
                "text": " ".join(t for t, _, _ in words),
                "conf": sum(c for _, c, _ in words) / len(words),
                "box": box,
                "quad": _rect_quad(box),
                "words": wds,
            }
        )
    return lines, "tesseract"


_BACKENDS = {
    "ppocr-v6": _paddle,
    "ppocr": _paddle,
    "surya": _surya,
    "doctr": _doctr,
    "tesseract": _tesseract,
}


def _run_backend(name, img_path, cfg, use_cache=True):
    fn = _BACKENDS.get(name)
    if fn is None:
        raise ValueError(f"ocr: unknown backend '{name}'")
    cache_path, _ = _cache_paths(img_path, name, cfg) if use_cache else (None, None)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    lines, engine = fn(img_path, cfg)
    payload = {"engine": engine, "lines": lines}
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    return payload


# ── challenger reconciliation ────────────────────────────────────────────────────────
def _reconcile(primary_lines, challenger_sets, iou_thresh=0.5):
    """IoU-cluster primary + challenger lines; keep highest-conf per cluster,
    penalize + flag clusters where the winning text disagrees with a challenger."""
    clusters = []  # each: {lines:[...]}
    all_lines = list(primary_lines)
    for cs in challenger_sets:
        all_lines += cs
    for l in all_lines:
        placed = False
        for cl in clusters:
            if _iou(cl["rep"]["box"], l["box"]) >= iou_thresh:
                cl["members"].append(l)
                placed = True
                break
        if not placed:
            clusters.append({"rep": l, "members": [l]})
    out = []
    for cl in clusters:
        best = max(cl["members"], key=lambda m: m["conf"])
        texts = {m["text"].strip().lower() for m in cl["members"]}
        line = dict(best)
        if len(texts) > 1:
            line["conf"] = round(best["conf"] * 0.8, 4)
            line.setdefault("meta", {})["disagreement"] = sorted(
                {m["text"] for m in cl["members"]}
            )
        out.append(line)
    return out


# ── public API ───────────────────────────────────────────────────────────────────────
def run_ocr(img_path: str, cfg: Optional[dict] = None, run_dir: Optional[str] = None):
    schema = _load_schema()
    cfg = cfg or {}
    ocr_cfg = cfg.get("ocr") or {}
    primary = ocr_cfg.get("primary", "ppocr-v6")
    challengers = ocr_cfg.get("challengers") or []

    t0 = time.time()
    primary_payload = _run_backend(primary, img_path, cfg)
    engine = primary_payload["engine"]
    primary_lines = primary_payload["lines"]

    if challengers:
        challenger_sets = []
        for name in challengers:
            if name == primary:
                continue
            try:
                challenger_sets.append(_run_backend(name, img_path, cfg)["lines"])
            except ImportError as e:
                print(f"[ocr] challenger '{name}' unavailable, skipping: {e}")
        if challenger_sets:
            merged = _reconcile(primary_lines, challenger_sets)
            engine = f"{engine}+" + "+".join(challengers)
        else:
            merged = primary_lines
    else:
        merged = primary_lines

    ordered = _order_lines(merged)

    # build schema-shaped result
    lines_out = []
    for l in ordered:
        line = {
            "id": l["id"],
            "text": l["text"],
            "conf": round(float(l["conf"]), 4),
            "box": l["box"],
            "quad": l["quad"],
            "words": l.get("words", []),
        }
        if l.get("meta"):
            line["meta"] = l["meta"]
        lines_out.append(line)

    # source dims
    w = h = 0
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
    except Exception:
        pass

    result = {
        "engine": engine,
        "source": {"path": img_path, "w": w, "h": h},
        "ms": round((time.time() - t0) * 1000, 1),
        "lines": lines_out,
    }

    if run_dir is None:
        run_dir = cfg.get("run_dir")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(result, os.path.join(run_dir, "ocr.json"))
    return result


if __name__ == "__main__":  # CPU-safe smoke: exercises ordering/geometry, no models
    print("[ocr] reading-order self-test (no OCR model needed)")
    raw = [
        {"text": "world", "conf": 0.9, "box": {"x": 200, "y": 10, "w": 80, "h": 20},
         "quad": _rect_quad({"x": 200, "y": 10, "w": 80, "h": 20}), "words": []},
        {"text": "hello", "conf": 0.9, "box": {"x": 20, "y": 12, "w": 80, "h": 20},
         "quad": _rect_quad({"x": 20, "y": 12, "w": 80, "h": 20}), "words": []},
        {"text": "second row", "conf": 0.8, "box": {"x": 20, "y": 60, "w": 160, "h": 22},
         "quad": _rect_quad({"x": 20, "y": 60, "w": 160, "h": 22}), "words": []},
    ]
    for l in _order_lines(raw):
        print(l["id"], "->", l["text"])
    assert [l["text"] for l in _order_lines(raw)] == ["hello", "world", "second row"]
    print("iou check:", round(_iou({"x": 0, "y": 0, "w": 10, "h": 10},
                                   {"x": 5, "y": 0, "w": 10, "h": 10}), 3))
    print("ok")
