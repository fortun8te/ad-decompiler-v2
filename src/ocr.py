"""OCR adapters, targeted small-text retry, and calibrated ensemble fusion.

The public contract is still ``run_ocr(image_path, cfg, run_dir)`` and the
returned mapping still matches ``schema.OcrResult``.  Backend-specific objects
are normalized into line dictionaries with stable word/quad geometry and
provenance before reading order is assigned.

Supported result generations:

* PaddleOCR 3 / PP-OCRv6: ``PaddleOCR.predict`` Result objects exposing
  ``rec_texts``, ``rec_scores`` and ``rec_polys`` through ``.json``.
* Legacy PaddleOCR: ``ocr.ocr`` nested ``[quad, (text, confidence)]`` tuples.
* Surya v2: ``PageOCRResult.blocks`` through ``SuryaInferenceManager``.
* Surya v1: ``text_lines`` through recognition + detection predictors.
* Current and legacy docTR page/block/line/word objects.

Heavy libraries are imported only inside their adapters.  The parsing,
reconciliation and retry tests need only standard CPU-side dependencies.
"""
from __future__ import annotations

import copy
from difflib import SequenceMatcher
import hashlib
import html
import importlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from typing import Any, Callable, Iterable, Optional

from src.agent_debug import log as _agent_log


_PADDLE_ENGINES: dict[tuple, tuple[Any, str]] = {}
_SURYA_ENGINES: dict[tuple, tuple[Any, str, Any]] = {}
_DOCTR_ENGINES: dict[tuple, Any] = {}
_EASYOCR_ENGINES: dict[tuple, Any] = {}

_DEFAULT_CALIBRATION = {
    "ppocr-v6": 1.00,
    "ppocr": 0.98,
    "surya": 0.97,
    "doctr": 0.95,
    "easyocr": 0.88,
    "tesseract": 0.82,
}


def _load_schema():
    for name in ("src.schema", "schema"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("schema")


# ---------------------------------------------------------------------------
# Generic object and geometry normalization


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(value: Any) -> Any:
    """Convert arrays/model containers to ordinary Python values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, dict, list, tuple)):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    try:
        return list(value)
    except (TypeError, ValueError):
        return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _quad_to_box(quad):
    quad = _normalize_quad(quad)
    if not quad:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0, y0 = min(xs), min(ys)
    return {
        "x": float(x0),
        "y": float(y0),
        "w": float(max(xs) - x0),
        "h": float(max(ys) - y0),
    }


def _rect_quad(box):
    x = _float(_get(box, "x"))
    y = _float(_get(box, "y"))
    w = max(0.0, _float(_get(box, "w")))
    h = max(0.0, _float(_get(box, "h")))
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _normalize_quad(value: Any) -> list:
    value = _plain(value)
    if value is None:
        return []
    try:
        # Rectangular [x0, y0, x1, y1] representation.
        if len(value) == 4 and all(not isinstance(item, (list, tuple)) for item in value):
            x0, y0, x1, y1 = map(float, value)
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        points = []
        for point in value:
            point = _plain(point)
            if point is None or len(point) < 2:
                continue
            points.append([float(point[0]), float(point[1])])
        if len(points) >= 4:
            return points[:4]
        if len(points) == 2:
            return _normalize_quad([points[0][0], points[0][1], points[1][0], points[1][1]])
    except (TypeError, ValueError, IndexError):
        return []
    return []


def _relative_quad(value: Any, width: float, height: float) -> list:
    quad = _normalize_quad(value)
    if not quad:
        return []
    maximum = max(abs(coordinate) for point in quad for coordinate in point)
    if maximum <= 2.0:  # docTR normalized coordinates
        return [[point[0] * width, point[1] * height] for point in quad]
    return quad


def _clean_box(box: Any) -> dict:
    return {
        "x": _float(_get(box, "x")),
        "y": _float(_get(box, "y")),
        "w": max(0.0, _float(_get(box, "w"))),
        "h": max(0.0, _float(_get(box, "h"))),
    }


def _union_boxes(boxes: Iterable[dict]) -> dict:
    values = [_clean_box(box) for box in boxes]
    if not values:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    x0 = min(box["x"] for box in values)
    y0 = min(box["y"] for box in values)
    x1 = max(box["x"] + box["w"] for box in values)
    y1 = max(box["y"] + box["h"] for box in values)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _iou(a, b):
    a, b = _clean_box(a), _clean_box(b)
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _vertical_overlap(a: dict, b: dict) -> float:
    a, b = _clean_box(a), _clean_box(b)
    overlap = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return overlap / max(1.0, min(a["h"], b["h"]))


def _normalize_word(word: dict, engine: str) -> Optional[dict]:
    text = str(word.get("text") or word.get("value") or "").strip()
    if not text:
        return None
    quad = _normalize_quad(word.get("quad") or word.get("polygon"))
    box = _clean_box(word.get("box")) if word.get("box") else _quad_to_box(quad)
    if not quad:
        quad = _rect_quad(box)
    out = {
        "text": text,
        "conf": round(max(0.0, min(1.0, _float(word.get("conf", word.get("confidence", 1.0))))), 4),
        "box": box,
        "quad": quad,
    }
    meta = copy.deepcopy(word.get("meta") or {})
    meta.setdefault("engine", engine)
    if meta:
        out["meta"] = meta
    return out


def _make_line(text: Any, confidence: Any, quad: Any = None, box: Any = None,
               words: Optional[list] = None, engine: str = "unknown",
               meta: Optional[dict] = None) -> Optional[dict]:
    text = str(text or "").strip()
    if not text:
        return None
    normalized_quad = _normalize_quad(quad)
    normalized_box = _clean_box(box) if box is not None else _quad_to_box(normalized_quad)
    if not normalized_quad:
        normalized_quad = _rect_quad(normalized_box)
    normalized_words = []
    for word in words or []:
        candidate = _normalize_word(word, engine)
        if candidate:
            normalized_words.append(candidate)
    line_meta = copy.deepcopy(meta or {})
    line_meta.setdefault("engine", engine)
    return {
        "text": text,
        "conf": round(max(0.0, min(1.0, _float(confidence, 0.0))), 4),
        "box": normalized_box,
        "quad": normalized_quad,
        "words": normalized_words,
        "meta": line_meta,
    }


# ---------------------------------------------------------------------------
# Reading order and cache


def _line_rotation(line: dict) -> float:
    """Return the baseline angle of an OCR line, normalized to ``[-90, 90]``."""
    quad = _normalize_quad(line.get("quad"))
    if len(quad) < 2:
        return 0.0
    angle = math.degrees(math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def _rotation_delta(a: float, b: float) -> float:
    """Smallest angle between two text baselines (baseline direction is unsigned)."""
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def _column_order(lines: list[dict], median_height: float) -> Optional[list[dict]]:
    """Return a conservative column-major order, or ``None`` for normal artwork.

    OCR engines generally return rows in y/x order.  That interleaves real two-column
    copy (left-1, right-1, left-2, right-2), which is especially harmful when a later
    stage uses the order to infer paragraphs.  We only switch to columns when there are
    at least two stable, multi-line left-edge tracks separated by a large gutter.  Small
    labels, CTAs, and cards deliberately do *not* meet that confidence bar.
    """
    if len(lines) < 4:
        return None
    horizontal = [line for line in lines if _rotation_delta(_line_rotation(line), 0.0) <= 18.0]
    if len(horizontal) < 4:
        return None
    tolerance = max(14.0, median_height * 1.8)
    tracks: list[dict] = []
    for line in sorted(horizontal, key=lambda item: item["box"]["x"]):
        left = float(line["box"]["x"])
        candidates = [track for track in tracks if abs(left - track["anchor"]) <= tolerance]
        if candidates:
            track = min(candidates, key=lambda item: abs(left - item["anchor"]))
            count = len(track["items"])
            track["anchor"] = (track["anchor"] * count + left) / (count + 1)
            track["items"].append(line)
        else:
            tracks.append({"anchor": left, "items": [line]})
    tracks = [track for track in tracks if len(track["items"]) >= 2]
    tracks.sort(key=lambda track: track["anchor"])
    if len(tracks) < 2:
        return None

    # Use only the two most populous non-overlapping tracks.  A third incidental
    # label should not make an ad look like a newspaper.
    pair = None
    for left_index, left in enumerate(tracks):
        for right in tracks[left_index + 1:]:
            gutter = right["anchor"] - left["anchor"]
            if gutter < max(median_height * 4.0, tolerance * 1.6):
                continue
            left_span = max(item["box"]["y"] + item["box"]["h"] for item in left["items"]) - min(item["box"]["y"] for item in left["items"])
            right_span = max(item["box"]["y"] + item["box"]["h"] for item in right["items"]) - min(item["box"]["y"] for item in right["items"])
            if min(left_span, right_span) < median_height * 1.35:
                continue
            score = len(left["items"]) + len(right["items"])
            if pair is None or score > pair[0]:
                pair = (score, left, right)
    if pair is None:
        return None

    _, left, right = pair
    members = {id(item) for item in left["items"] + right["items"]}
    # Lines spanning both tracks (e.g. a headline above two columns) remain in
    # normal y/x order.  Put them before/between/after columns by their y position.
    spanning = [item for item in lines if id(item) not in members]
    ordered: list[dict] = []
    columns = [left["items"], right["items"]]
    first_column_top = min(item["box"]["y"] for column in columns for item in column)
    ordered.extend(sorted((item for item in spanning if item["box"]["y"] < first_column_top),
                          key=lambda item: (item["box"]["y"], item["box"]["x"])))
    for column in columns:
        ordered.extend(sorted(column, key=lambda item: (item["box"]["y"], item["box"]["x"])))
    ordered.extend(sorted((item for item in spanning if item["box"]["y"] >= first_column_top),
                          key=lambda item: (item["box"]["y"], item["box"]["x"])))
    return ordered


def _order_lines(raw_lines):
    """Order lines without flattening word children; preserve real text columns."""
    if not raw_lines:
        return []
    heights = sorted(max(1.0, _float(line["box"].get("h"), 1.0)) for line in raw_lines)
    median_height = heights[len(heights) // 2]
    row_tolerance = median_height * 0.5
    rows = []
    for line in sorted(raw_lines, key=lambda item: (item["box"]["y"], item["box"]["x"])):
        center_y = line["box"]["y"] + line["box"]["h"] / 2.0
        for row in rows:
            if abs(center_y - row["center_y"]) <= row_tolerance:
                row["items"].append(line)
                count = len(row["items"])
                row["center_y"] = (row["center_y"] * (count - 1) + center_y) / count
                break
        else:
            rows.append({"center_y": center_y, "items": [line]})
    row_ordered = []
    for row in sorted(rows, key=lambda item: item["center_y"]):
        row_ordered.extend(sorted(row["items"], key=lambda item: item["box"]["x"]))
    ordered = _column_order(row_ordered, median_height) or row_ordered
    for index, line in enumerate(ordered):
        line["id"] = f"L{index}"
    return ordered


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_paths(img_path, engine, cfg):
    root = cfg.get("cache_dir") or os.path.join(cfg.get("runs_dir", "runs"), ".cache", "ocr")
    os.makedirs(root, exist_ok=True)
    digest = _sha256(img_path)
    return os.path.join(root, f"{digest}.{engine}.json"), digest


def _tag_lines(lines: Iterable[dict], engine: str) -> list:
    tagged = []
    for raw in lines:
        line = copy.deepcopy(raw)
        line.setdefault("words", [])
        line.setdefault("quad", _rect_quad(line.get("box") or {}))
        line.setdefault("box", _quad_to_box(line["quad"]))
        line.setdefault("meta", {})
        line["meta"].setdefault("engine", engine)
        tagged.append(line)
    return tagged


# ---------------------------------------------------------------------------
# PaddleOCR 3 and legacy parsing


def _unwrap_paddle_mapping(value: Any) -> Optional[dict]:
    current = value
    seen = set()
    for _ in range(8):
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        if isinstance(current, dict):
            if "rec_texts" in current and ("rec_polys" in current or "rec_boxes" in current):
                return current
            next_value = None
            for key in ("res", "result", "data", "overall_ocr_res", "ocr_res"):
                candidate = current.get(key)
                if candidate is not None:
                    next_value = candidate
                    break
            if next_value is None:
                break
            current = next_value
            continue
        json_value = getattr(current, "json", None)
        if callable(json_value):
            try:
                json_value = json_value()
            except Exception:
                json_value = None
        if json_value is not None:
            current = json_value
            continue
        for key in ("res", "result", "data"):
            candidate = getattr(current, key, None)
            if candidate is not None:
                current = candidate
                break
        else:
            break
    return None


def _iter_result_items(result: Any) -> list:
    if result is None:
        return []
    if isinstance(result, dict) or _unwrap_paddle_mapping(result) is not None:
        return [result]
    try:
        return list(result)
    except TypeError:
        return [result]


def _parse_paddle_v3(result: Any) -> list:
    lines = []
    for item in _iter_result_items(result):
        mapping = _unwrap_paddle_mapping(item)
        if not mapping:
            continue
        texts = list(_plain(mapping.get("rec_texts")) or [])
        scores = list(_plain(mapping.get("rec_scores")) or [])
        polygons = list(_plain(mapping.get("rec_polys")) or [])
        boxes = list(_plain(mapping.get("rec_boxes")) or [])
        orientations = list(_plain(mapping.get("textline_orientation_angles")) or [])
        detection_scores = list(_plain(mapping.get("dt_scores")) or [])
        for index, text in enumerate(texts):
            polygon = polygons[index] if index < len(polygons) else None
            box_value = boxes[index] if index < len(boxes) else None
            quad = _normalize_quad(polygon)
            if not quad:
                quad = _normalize_quad(box_value)
            if not quad:
                continue
            score = scores[index] if index < len(scores) else 0.0
            meta = {
                "engine": "ppocr-v6",
                "backend_api": "paddle-v3-predict",
                "source_kind": "line",
            }
            if index < len(orientations):
                meta["orientation"] = _plain(orientations[index])
            # dt_scores can be unfiltered while rec_* arrays are filtered; only
            # attach it when lengths prove positional alignment.
            if len(detection_scores) == len(texts):
                meta["detection_confidence"] = round(_float(detection_scores[index]), 4)
            line = _make_line(text, score, quad=quad, engine="ppocr-v6", meta=meta)
            if line:
                lines.append(line)
    return lines


def _legacy_detection(value: Any) -> Optional[tuple]:
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    quad = _normalize_quad(value[0])
    recognition = _plain(value[1])
    if not quad or not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
        return None
    if isinstance(recognition[0], (list, tuple, dict)):
        return None
    return quad, recognition[0], recognition[1]


def _walk_legacy_paddle(value: Any):
    detection = _legacy_detection(value)
    if detection:
        yield detection
        return
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_legacy_paddle(child)


def _parse_paddle_legacy(result: Any) -> list:
    lines = []
    for quad, text, confidence in _walk_legacy_paddle(result):
        line = _make_line(
            text, confidence, quad=quad, engine="ppocr-v6",
            meta={"engine": "ppocr-v6", "backend_api": "paddle-legacy-ocr", "source_kind": "line"},
        )
        if line:
            lines.append(line)
    return lines


def _parse_paddle_result(result: Any) -> list:
    current = _parse_paddle_v3(result)
    return current if current else _parse_paddle_legacy(result)


def _paddle_engine(cfg: dict, *, device_override: Optional[str] = None):
    ocr_cfg = cfg.get("ocr") or {}
    lang = ocr_cfg.get("lang", "en")
    device_key = device_override if device_override is not None else str(cfg.get("device", "cpu"))
    device = "gpu" if device_key.startswith("cuda") else "cpu"
    key = (
        lang, device, ocr_cfg.get("text_detection_model_name"),
        ocr_cfg.get("text_recognition_model_name"), bool(ocr_cfg.get("textline_orientation", True)),
    )
    if key in _PADDLE_ENGINES:
        return _PADDLE_ENGINES[key]
    try:
        from paddleocr import PaddleOCR
    except ImportError as error:  # pragma: no cover - exercised on GPU host
        # #region agent log
        _agent_log("ocr.py:_paddle_engine", "paddleocr import failed", data={"error": str(error)}, hypothesis_id="H1", cfg=cfg)
        # #endregion
        raise ImportError(
            "PP-OCRv6 requires paddleocr>=3 and a matching paddlepaddle build.\n"
            "  pip install paddleocr>=3 paddlepaddle-gpu"
        ) from error

    kwargs = {
        "lang": lang,
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": bool(ocr_cfg.get("textline_orientation", True)),
    }
    for option in ("text_detection_model_name", "text_recognition_model_name"):
        if ocr_cfg.get(option):
            kwargs[option] = ocr_cfg[option]
    try:
        engine = PaddleOCR(**kwargs)
        api = "v3"
    except TypeError:
        # PaddleOCR 2.x / early 3.x compatibility.
        legacy_kwargs = {
            "use_angle_cls": True,
            "lang": lang,
            "use_gpu": device == "gpu",
            "show_log": False,
        }
        engine = PaddleOCR(**legacy_kwargs)
        api = "legacy"
    _PADDLE_ENGINES[key] = (engine, api)
    return engine, api


def _is_paddle_gpu_failure(error: Exception) -> bool:
    """True when Paddle likely failed because of CUDA/cuDNN, not bad input."""
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    if error_type in {"oserror", "runtimeerror", "importerror"} and any(
        token in message for token in ("cudnn", "cuda", "cublas", "gpu", "winerror 127", ".dll")
    ):
        return True
    return any(token in message for token in (
        "cudnn_cnn", "could not load", "no cuda gpus", "cuda driver",
        "cudnn version", "gpu is not supported",
    ))


def _paddle(img_path, cfg, *, device_override: Optional[str] = None):
    device = device_override or str(cfg.get("device", "cpu"))
    # #region agent log
    _agent_log("ocr.py:_paddle", "paddle backend start", data={"device": device, "path": os.path.basename(img_path)}, hypothesis_id="H1", cfg=cfg)
    # #endregion
    try:
        engine, api = _paddle_engine(cfg, device_override=device_override)
        if api == "v3" and hasattr(engine, "predict"):
            result = engine.predict(img_path)
        else:
            result = engine.ocr(img_path, cls=True)
        lines = _parse_paddle_result(result)
        # #region agent log
        _agent_log("ocr.py:_paddle", "paddle backend ok", data={"api": api, "lines": len(lines), "device": device}, hypothesis_id="H1", cfg=cfg)
        # #endregion
        return lines, "ppocr-v6"
    except Exception as error:
        # #region agent log
        _agent_log("ocr.py:_paddle", "paddle backend failed", data={"device": device, "error": str(error), "error_type": type(error).__name__}, hypothesis_id="H1", cfg=cfg)
        # #endregion
        configured = str(cfg.get("device", "cpu"))
        if (
            device_override is None
            and configured.startswith("cuda")
            and _is_paddle_gpu_failure(error)
        ):
            print(f"[ocr] paddle GPU failed ({error}); retrying once on CPU")
            # #region agent log
            _agent_log(
                "ocr.py:_paddle", "paddle gpu failed; retrying cpu",
                data={"error": str(error), "error_type": type(error).__name__},
                hypothesis_id="H1",
                cfg=cfg,
            )
            # #endregion
            try:
                return _paddle(img_path, cfg, device_override="cpu")
            except Exception as cpu_error:
                raise RuntimeError(
                    f"Paddle GPU failed ({error}) and CPU retry also failed ({cpu_error})"
                ) from cpu_error
        raise


# ---------------------------------------------------------------------------
# Surya v2 and v1 parsing


def _strip_html(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<\s*(?:br|/p|/div|/tr|/li|/h[1-6])\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?t[dh][^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def _parse_surya_predictions(predictions: Any) -> list:
    lines = []
    pages = list(predictions or [])
    for page in pages:
        blocks = _get(page, "blocks")
        if blocks is not None:  # Surya v2
            for block in list(blocks or []):
                if bool(_get(block, "skipped", False)) or bool(_get(block, "error", False)):
                    continue
                raw_html = _get(block, "html", "")
                text = _strip_html(raw_html or _get(block, "text", ""))
                quad = _normalize_quad(_get(block, "polygon"))
                if not quad:
                    quad = _normalize_quad(_get(block, "bbox"))
                meta = {
                    "engine": "surya",
                    "backend_api": "surya-v2-blocks",
                    "source_kind": "block",
                    "layout_label": _get(block, "label"),
                    "raw_label": _get(block, "raw_label"),
                    "reading_order": _get(block, "reading_order"),
                    "html": str(raw_html or ""),
                }
                line = _make_line(text, _get(block, "confidence", 0.0), quad=quad,
                                  engine="surya", meta=meta)
                if line:
                    lines.append(line)
            continue

        # Surya v1 RecognitionPredictor output.
        text_lines = _get(page, "text_lines", []) or []
        for value in text_lines:
            bbox = _plain(_get(value, "bbox"))
            quad = _normalize_quad(_get(value, "polygon"))
            if not quad:
                quad = _normalize_quad(bbox)
            text = _get(value, "text", _get(value, "value", ""))
            confidence = _get(value, "confidence", _get(value, "conf", 1.0))
            line = _make_line(
                text, confidence, quad=quad, engine="surya",
                meta={"engine": "surya", "backend_api": "surya-v1-text-lines", "source_kind": "line"},
            )
            if line:
                lines.append(line)
    return lines


def _surya(img_path, cfg):
    # #region agent log
    _agent_log("ocr.py:_surya", "surya backend start", data={"device": str(cfg.get("device", "cpu"))}, hypothesis_id="H2", cfg=cfg)
    # #endregion
    try:
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
    except ImportError as error:  # pragma: no cover
        # #region agent log
        _agent_log("ocr.py:_surya", "surya import failed", data={"error": str(error)}, hypothesis_id="H2", cfg=cfg)
        # #endregion
        raise ImportError("Surya requires Pillow and surya-ocr.") from error

    try:
        key = (str(cfg.get("device", "cpu")),)
        if key in _SURYA_ENGINES:
            predictor, api, detector = _SURYA_ENGINES[key]
        else:
            try:
                from surya.inference import SuryaInferenceManager
                from surya.recognition import RecognitionPredictor

                manager = SuryaInferenceManager()
                predictor = RecognitionPredictor(manager)
                detector = None
                api = "v2"
            except (ImportError, TypeError):
                try:
                    from surya.recognition import RecognitionPredictor
                    from surya.detection import DetectionPredictor
                except ImportError as error:  # pragma: no cover
                    raise ImportError("Surya backend requires surya-ocr.  pip install surya-ocr") from error
                predictor = RecognitionPredictor()
                detector = DetectionPredictor()
                api = "v1"
            _SURYA_ENGINES[key] = (predictor, api, detector)

        if api == "v2":
            predictions = predictor([image])
        else:
            language = (cfg.get("ocr") or {}).get("lang", "en")
            predictions = predictor([image], [[language]], detector)
        lines = _parse_surya_predictions(predictions)
        # #region agent log
        _agent_log("ocr.py:_surya", "surya backend ok", data={"api": api, "lines": len(lines)}, hypothesis_id="H2", cfg=cfg)
        # #endregion
        return lines, "surya"
    except Exception as error:
        # #region agent log
        _agent_log("ocr.py:_surya", "surya backend failed", data={"error": str(error), "error_type": type(error).__name__}, hypothesis_id="H2", cfg=cfg)
        # #endregion
        raise


# ---------------------------------------------------------------------------
# docTR parsing


def _doctr_word(word: Any, page_width: float, page_height: float) -> Optional[dict]:
    quad = _relative_quad(_get(word, "geometry"), page_width, page_height)
    if not quad:
        return None
    orientation = _get(word, "crop_orientation")
    meta = {"engine": "doctr"}
    if orientation is not None:
        meta["crop_orientation"] = _plain(orientation)
    objectness = _get(word, "objectness_score")
    if objectness is not None:
        meta["objectness_score"] = _float(objectness)
    return {
        "text": str(_get(word, "value", _get(word, "text", "")) or ""),
        "conf": _float(_get(word, "confidence", _get(word, "conf", 0.0))),
        "box": _quad_to_box(quad),
        "quad": quad,
        "meta": meta,
    }


def _parse_doctr_document(document: Any) -> list:
    lines = []
    for page in list(_get(document, "pages", []) or []):
        dimensions = _plain(_get(page, "dimensions", (0, 0))) or (0, 0)
        page_height = _float(dimensions[0] if len(dimensions) > 0 else 0)
        page_width = _float(dimensions[1] if len(dimensions) > 1 else 0)
        for block_index, block in enumerate(list(_get(page, "blocks", []) or [])):
            for line_index, source_line in enumerate(list(_get(block, "lines", []) or [])):
                words = []
                for source_word in list(_get(source_line, "words", []) or []):
                    word = _doctr_word(source_word, page_width, page_height)
                    if word and word["text"].strip():
                        words.append(word)
                if not words:
                    continue
                line_quad = _relative_quad(_get(source_line, "geometry"), page_width, page_height)
                line_box = _quad_to_box(line_quad) if line_quad else _union_boxes(word["box"] for word in words)
                if not line_quad:
                    line_quad = _rect_quad(line_box)
                confidence = sum(word["conf"] for word in words) / len(words)
                meta = {
                    "engine": "doctr",
                    "backend_api": "doctr-document",
                    "source_kind": "line",
                    "block_index": block_index,
                    "line_index": line_index,
                }
                objectness = _get(source_line, "objectness_score")
                if objectness is not None:
                    meta["objectness_score"] = _float(objectness)
                line = _make_line(
                    " ".join(word["text"] for word in words), confidence,
                    quad=line_quad, box=line_box, words=words, engine="doctr", meta=meta,
                )
                if line:
                    lines.append(line)
    return lines


def _torch_device_name(cfg: dict) -> str:
    """Map config device string to a torch device name."""
    device_key = str(cfg.get("device", "cpu")).lower()
    return "cuda" if device_key.startswith("cuda") else "cpu"


def _doctr(img_path, cfg):
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("docTR backend requires python-doctr.  pip install python-doctr[torch]") from error
    device = _torch_device_name(cfg)
    if device == "cuda" and not torch.cuda.is_available():
        print("[ocr] doctr CUDA requested but torch cannot see a GPU; using CPU")
        device = "cpu"
    key = (device,)
    predictor = _DOCTR_ENGINES.get(key)
    if predictor is None:
        try:
            predictor = ocr_predictor(
                pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True,
                resolve_lines=True, resolve_blocks=True,
            )
        except TypeError:
            predictor = ocr_predictor(pretrained=True)
        if hasattr(predictor, "to"):
            predictor = predictor.to(torch.device(device))
        _DOCTR_ENGINES[key] = predictor
    document = DocumentFile.from_images(img_path)
    return _parse_doctr_document(predictor(document)), "doctr"


# ---------------------------------------------------------------------------
# EasyOCR


def _parse_easyocr_results(results: Any) -> list:
    lines = []
    for item in results or []:
        item = _plain(item)
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        quad = _normalize_quad(item[0])
        if not quad:
            continue
        line = _make_line(
            item[1], item[2], quad=quad, engine="easyocr",
            meta={"engine": "easyocr", "backend_api": "easyocr-readtext", "source_kind": "line"},
        )
        if line:
            lines.append(line)
    return lines


def _easyocr_engine(cfg: dict):
    ocr_cfg = cfg.get("ocr") or {}
    lang = str(ocr_cfg.get("lang", "en"))
    gpu = str(cfg.get("device", "cpu")).lower().startswith("cuda")
    key = (lang, gpu)
    reader = _EASYOCR_ENGINES.get(key)
    if reader is not None:
        return reader
    try:
        import easyocr
    except ImportError as error:  # pragma: no cover
        raise ImportError("EasyOCR backend requires easyocr.  pip install easyocr") from error
    reader = easyocr.Reader([lang], gpu=gpu)
    _EASYOCR_ENGINES[key] = reader
    return reader


def _easyocr(img_path, cfg):
    reader = _easyocr_engine(cfg)
    return _parse_easyocr_results(reader.readtext(img_path)), "easyocr"


# ---------------------------------------------------------------------------
# Tesseract baseline


def _tesseract(img_path, cfg):
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Tesseract backend requires pytesseract and the tesseract binary.\n"
            "  pip install pytesseract"
        ) from error
    language = (cfg.get("ocr") or {}).get("lang", "en")
    data = pytesseract.image_to_data(
        Image.open(img_path), lang=language, output_type=pytesseract.Output.DICT
    )
    groups = {}
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text or "").strip()
        confidence = _float(data["conf"][index], -1.0)
        if not text or confidence < 0:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        box = {
            "x": _float(data["left"][index]), "y": _float(data["top"][index]),
            "w": _float(data["width"][index]), "h": _float(data["height"][index]),
        }
        groups.setdefault(key, []).append({
            "text": text, "conf": confidence / 100.0,
            "box": box, "quad": _rect_quad(box),
            "meta": {"engine": "tesseract"},
        })
    lines = []
    for words in groups.values():
        box = _union_boxes(word["box"] for word in words)
        confidence = sum(word["conf"] for word in words) / len(words)
        line = _make_line(
            " ".join(word["text"] for word in words), confidence,
            box=box, words=words, engine="tesseract",
            meta={"engine": "tesseract", "backend_api": "image-to-data", "source_kind": "line"},
        )
        if line:
            lines.append(line)
    return lines, "tesseract"


_BACKENDS = {
    "ppocr-v6": _paddle,
    "ppocr": _paddle,
    "surya": _surya,
    "doctr": _doctr,
    "easyocr": _easyocr,
    "tesseract": _tesseract,
}


def ensemble_disagreement_lines(lines: Iterable[dict], cfg: Optional[dict] = None) -> list[dict]:
    """Return fused OCR lines where challengers disagreed and the winner looks confident."""
    ocr_cfg = (cfg or {}).get("ocr") or {}
    setting = ocr_cfg.get("ensemble_disagreement")
    if not setting:
        return []
    min_confidence = 0.85
    if isinstance(setting, dict):
        if not setting.get("enabled", True):
            return []
        min_confidence = _float(setting.get("min_confidence"), min_confidence)
    output = []
    for line in lines or []:
        meta = line.get("meta") or {}
        if not meta.get("disagreement"):
            continue
        if _float(line.get("conf")) >= min_confidence and line.get("box"):
            output.append(line)
    return output


def _geometry_metrics(lines: Iterable[dict]) -> dict:
    """Return conservative geometry health metrics for the OCR evidence."""
    total = invalid = missing_quad = 0
    for line in lines or []:
        total += 1
        box = _clean_box(line.get("box") or {})
        quad = _normalize_quad(line.get("quad"))
        if not quad:
            missing_quad += 1
        if box["w"] <= 0 or box["h"] <= 0 or len(quad) < 4:
            invalid += 1
    return {
        "lines": total,
        "valid_lines": max(0, total - invalid),
        "invalid_lines": invalid,
        "missing_quad": missing_quad,
        "valid": invalid == 0,
    }


def _cross_check_metrics(lines: list[dict], configured: list[str], successful: list[str]) -> dict:
    disagreements = sum(1 for line in lines if (line.get("meta") or {}).get("disagreement"))
    required = list(dict.fromkeys(str(name) for name in configured))
    present = list(dict.fromkeys(str(name) for name in successful))
    missing = [name for name in required if name not in present]
    return {
        "required_engines": required,
        "successful_engines": present,
        "missing_engines": missing,
        "required": bool(required),
        "complete": not required or not missing,
        "lines_checked": len(lines) if len(present) > 1 else 0,
        "disagreements": disagreements,
        "fail_closed": bool(required and bool(missing)),
    }


def _tesseract_available() -> bool:
    if not shutil.which("tesseract"):
        return False
    return importlib.util.find_spec("pytesseract") is not None


def _fallback_engine_names(cfg: dict) -> list[str]:
    ocr_cfg = cfg.get("ocr") or {}
    names = [str(name).lower() for name in (ocr_cfg.get("fallback_engines") or [])]
    if ocr_cfg.get("auto_fallback_tesseract", True) and "tesseract" not in names:
        if _tesseract_available():
            names.append("tesseract")
    return names


def _format_ocr_failure(errors: list[dict], *, fallback_errors: Optional[list[dict]] = None) -> str:
    details = "; ".join(f"{item['engine']}: {item['error']}" for item in errors)
    messages = [f"no configured OCR backend completed ({details})"]
    combined = " ".join(str(item.get("error", "")) for item in errors + (fallback_errors or []))
    lowered = combined.lower()
    if any(token in lowered for token in ("cudnn", "winerror 127", "cudnn_cnn")):
        messages.append(
            "Paddle GPU could not load cuDNN — install cuDNN 9.x matching your paddlepaddle-gpu "
            "wheel, or set device: cpu in config.yaml (paddle will auto-retry CPU once)."
        )
    elif "cuda" in lowered and any(item.get("engine", "").startswith("ppocr") for item in errors):
        messages.append("Paddle CUDA failed — verify GPU drivers or set device: cpu in config.yaml.")
    if fallback_errors:
        fb = "; ".join(f"{item['engine']}: {item['error']}" for item in fallback_errors)
        messages.append(f"fallback OCR also failed ({fb})")
    elif not _tesseract_available():
        messages.append(
            "For automatic recovery, install the tesseract binary on PATH and pip install pytesseract, "
            "or set ocr.fallback_engines in config.yaml."
        )
    else:
        messages.append("tesseract is installed but fallback did not run or also failed.")
    return " ".join(messages)


def _run_backend(name, img_path, cfg, use_cache=True):
    function = _BACKENDS.get(name)
    if function is None:
        raise ValueError(f"ocr: unknown backend '{name}'")
    cache_path = None
    if use_cache:
        cache_path, _ = _cache_paths(img_path, name, cfg)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["lines"] = _tag_lines(payload.get("lines", []), payload.get("engine", name))
            return payload
    lines, engine = function(img_path, cfg)
    payload = {"engine": engine, "lines": _tag_lines(lines, engine)}
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    return payload


# ---------------------------------------------------------------------------
# Targeted 2x retry


def _retry_options(cfg: dict) -> dict:
    raw = (cfg.get("ocr") or {}).get("retry_2x", True)
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, dict):
        options = dict(raw)
        options.setdefault("enabled", True)
        return options
    return {"enabled": False}


def _transform_word(word: dict, offset_x: float, offset_y: float, scale: float) -> dict:
    transformed = copy.deepcopy(word)
    box = _clean_box(transformed.get("box") or {})
    transformed["box"] = {
        "x": offset_x + box["x"] / scale,
        "y": offset_y + box["y"] / scale,
        "w": box["w"] / scale,
        "h": box["h"] / scale,
    }
    quad = _normalize_quad(transformed.get("quad"))
    transformed["quad"] = [
        [offset_x + point[0] / scale, offset_y + point[1] / scale] for point in quad
    ] if quad else _rect_quad(transformed["box"])
    return transformed


def _transform_line(line: dict, offset_x: float, offset_y: float, scale: float) -> dict:
    transformed = copy.deepcopy(line)
    box = _clean_box(transformed.get("box") or {})
    transformed["box"] = {
        "x": offset_x + box["x"] / scale,
        "y": offset_y + box["y"] / scale,
        "w": box["w"] / scale,
        "h": box["h"] / scale,
    }
    quad = _normalize_quad(transformed.get("quad"))
    transformed["quad"] = [
        [offset_x + point[0] / scale, offset_y + point[1] / scale] for point in quad
    ] if quad else _rect_quad(transformed["box"])
    transformed["words"] = [
        _transform_word(word, offset_x, offset_y, scale) for word in transformed.get("words", [])
    ]
    return transformed


def _collapse_retry_lines(lines: list, engine: str) -> Optional[dict]:
    if not lines:
        return None
    ordered = sorted(lines, key=lambda line: (line["box"]["y"], line["box"]["x"]))
    if len(ordered) == 1:
        result = copy.deepcopy(ordered[0])
        result.setdefault("meta", {})["engine"] = f"{engine}@2x"
        return result
    words = []
    for line in ordered:
        if line.get("words"):
            words.extend(copy.deepcopy(line["words"]))
        else:
            words.append({
                "text": line["text"], "conf": line["conf"],
                "box": copy.deepcopy(line["box"]), "quad": copy.deepcopy(line["quad"]),
                "meta": {"engine": f"{engine}@2x", "source_kind": "retry-fragment"},
            })
    weights = [max(1, len(line.get("text", "").strip())) for line in ordered]
    confidence = sum(line["conf"] * weight for line, weight in zip(ordered, weights)) / sum(weights)
    box = _union_boxes(line["box"] for line in ordered)
    return _make_line(
        " ".join(line["text"].strip() for line in ordered), confidence,
        box=box, words=words, engine=f"{engine}@2x",
        meta={"engine": f"{engine}@2x", "backend_api": "targeted-2x", "source_kind": "line"},
    )


def _text_key(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _text_similarity(a: Any, b: Any) -> float:
    aa, bb = _text_key(a), _text_key(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


def _targeted_retry(img_path: str, lines: list, engine: str, cfg: dict,
                    runner: Optional[Callable] = None) -> list:
    options = _retry_options(cfg)
    if not options.get("enabled") or not lines:
        return copy.deepcopy(lines)
    supported = options.get("engines", ["ppocr-v6", "ppocr", "surya", "doctr"])
    if engine not in supported:
        return copy.deepcopy(lines)
    try:
        from PIL import Image
        source = Image.open(img_path).convert("RGB")
    except Exception:
        return copy.deepcopy(lines)

    runner = runner or _run_backend
    scale = max(1.1, min(4.0, _float(options.get("scale"), 2.0)))
    small_height = max(4.0, _float(options.get("small_height"), 26.0))
    low_confidence = max(0.0, min(1.0, _float(options.get("low_confidence"), 0.72)))
    max_regions = max(0, min(32, int(options.get("max_regions", 6))))
    min_gain = max(0.0, _float(options.get("min_confidence_gain"), 0.025))

    eligible = []
    for index, line in enumerate(lines):
        reasons = []
        if _float(line.get("conf")) < low_confidence:
            reasons.append("low-confidence")
        if _float(line.get("box", {}).get("h")) <= small_height:
            reasons.append("small-text")
        if reasons:
            priority = (_float(line.get("conf")), _float(line.get("box", {}).get("h")))
            eligible.append((priority, index, reasons))
    eligible.sort(key=lambda item: item[0])
    eligible = eligible[:max_regions]
    if not eligible:
        return copy.deepcopy(lines)

    output = copy.deepcopy(lines)
    width, height = source.size
    with tempfile.TemporaryDirectory(prefix="ocr_retry_") as directory:
        for retry_index, (_, index, reasons) in enumerate(eligible):
            original = output[index]
            box = _clean_box(original.get("box") or {})
            padding = max(2, int(round(box["h"] * _float(options.get("padding_factor"), 0.35))))
            x0 = max(0, int(math.floor(box["x"] - padding)))
            y0 = max(0, int(math.floor(box["y"] - padding)))
            x1 = min(width, int(math.ceil(box["x"] + box["w"] + padding)))
            y1 = min(height, int(math.ceil(box["y"] + box["h"] + padding)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = source.crop((x0, y0, x1, y1))
            resized = crop.resize(
                (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            crop_path = os.path.join(directory, f"region_{retry_index}.png")
            resized.save(crop_path)
            try:
                payload = runner(engine, crop_path, cfg, use_cache=False)
            except Exception as error:
                original.setdefault("meta", {})["retry_2x"] = {
                    "attempted": True, "selected": False, "reasons": reasons,
                    "error": str(error),
                }
                continue
            transformed = [
                _transform_line(line, x0, y0, scale) for line in payload.get("lines", [])
            ]
            candidate = _collapse_retry_lines(transformed, engine)
            if candidate is None:
                original.setdefault("meta", {})["retry_2x"] = {
                    "attempted": True, "selected": False, "reasons": reasons,
                    "candidate_count": 0,
                }
                continue
            agreement = _text_similarity(original.get("text"), candidate.get("text"))
            retry_conf = _float(candidate.get("conf"))
            original_conf = _float(original.get("conf"))
            selected = (
                (agreement >= 0.96 and retry_conf >= original_conf)
                or retry_conf >= original_conf + min_gain
                or (not str(original.get("text") or "").strip() and retry_conf > 0)
            )
            retry_meta = {
                "attempted": True,
                "selected": selected,
                "scale": scale,
                "reasons": reasons,
                "candidate_text": candidate["text"],
                "candidate_confidence": round(retry_conf, 4),
                "text_agreement": round(agreement, 4),
                "candidate_box": copy.deepcopy(candidate["box"]),
            }
            if selected:
                # Full-image detection owns line placement.  The retry owns the
                # transcription/confidence and its useful word-level geometry.
                replacement = copy.deepcopy(candidate)
                replacement["box"] = copy.deepcopy(original["box"])
                replacement["quad"] = copy.deepcopy(original.get("quad") or _rect_quad(original["box"]))
                replacement.setdefault("meta", {}).update(copy.deepcopy(original.get("meta") or {}))
                replacement["meta"]["engine"] = engine
                replacement["meta"]["retry_2x"] = retry_meta
                output[index] = replacement
            else:
                original.setdefault("meta", {})["retry_2x"] = retry_meta
    return output


# ---------------------------------------------------------------------------
# Calibrated ensemble reconciliation


def _base_engine(line: dict) -> str:
    engine = str((line.get("meta") or {}).get("engine") or "unknown")
    return engine.split("@")[0]


def _calibrated_confidence(line: dict, cfg: Optional[dict]) -> float:
    engine = _base_engine(line)
    calibration = dict(_DEFAULT_CALIBRATION)
    calibration.update(((cfg or {}).get("ocr") or {}).get("engine_calibration") or {})
    setting = calibration.get(engine, 0.90)
    confidence = _float(line.get("conf"))
    if isinstance(setting, dict):
        confidence = confidence * _float(setting.get("scale"), 1.0) + _float(setting.get("offset"), 0.0)
    else:
        confidence *= _float(setting, 1.0)
    return max(0.0, min(1.0, confidence))


def _same_region(a: dict, b: dict, threshold: float) -> bool:
    if _iou(a["box"], b["box"]) >= threshold:
        return True
    if _vertical_overlap(a["box"], b["box"]) < 0.65:
        return False
    acx = a["box"]["x"] + a["box"]["w"] / 2.0
    bcx = b["box"]["x"] + b["box"]["w"] / 2.0
    return abs(acx - bcx) <= max(a["box"]["h"], b["box"]["h"]) * 1.2


def _reconcile(primary_lines, challenger_sets, iou_thresh=0.42, cfg=None):
    """Fuse overlapping line observations using calibration and text support.

    Word children remain attached to their parent line and are never inserted
    into the cluster input, preventing the common line+word duplicate failure.
    """
    observations = [copy.deepcopy(line) for line in primary_lines]
    for challenger in challenger_sets:
        observations.extend(copy.deepcopy(challenger))
    clusters = []
    for line in observations:
        for cluster in clusters:
            if any(_same_region(member, line, iou_thresh) for member in cluster):
                cluster.append(line)
                break
        else:
            clusters.append([line])

    agreement_threshold = _float(((cfg or {}).get("ocr") or {}).get("text_agreement_threshold"), 0.92)
    output = []
    for members in clusters:
        normalized = [_text_key(member.get("text")) for member in members]
        engines = [_base_engine(member) for member in members]
        scores = []
        for index, member in enumerate(members):
            similarities = [
                _text_similarity(member.get("text"), other.get("text"))
                for other_index, other in enumerate(members) if other_index != index
            ]
            near_agreement = sum(similarities) / len(similarities) if similarities else 1.0
            exact_engines = {
                engines[other_index] for other_index, value in enumerate(normalized)
                if value and value == normalized[index]
            }
            exact_support = len(exact_engines) / max(1, len(set(engines)))
            score = (
                0.62 * _calibrated_confidence(member, cfg)
                + 0.20 * near_agreement
                + 0.18 * exact_support
            )
            scores.append(score)
        winner_index = max(range(len(members)), key=lambda index: (scores[index],
                                                                    _calibrated_confidence(members[index], cfg)))
        winner = copy.deepcopy(members[winner_index])
        agreeing_indices = [
            index for index, member in enumerate(members)
            if _text_similarity(winner.get("text"), member.get("text")) >= agreement_threshold
        ]
        supporting_engines = sorted({engines[index] for index in agreeing_indices})

        # Prefer word geometry from an agreeing engine when the selected line
        # has none; do not promote those words to top-level lines.
        if not winner.get("words"):
            word_sources = [
                members[index] for index in agreeing_indices if members[index].get("words")
            ]
            if word_sources:
                source = max(word_sources, key=lambda line: _calibrated_confidence(line, cfg))
                winner["words"] = copy.deepcopy(source["words"])

        best_calibrated = max(_calibrated_confidence(members[index], cfg) for index in agreeing_indices)
        support_bonus = min(0.16, max(0, len(supporting_engines) - 1) * 0.055)
        winner["conf"] = round(min(1.0, best_calibrated + (1.0 - best_calibrated) * support_bonus), 4)
        provenance = []
        for index, member in enumerate(members):
            provenance.append({
                "engine": engines[index],
                "text": member.get("text", ""),
                "confidence": round(_float(member.get("conf")), 4),
                "calibrated_confidence": round(_calibrated_confidence(member, cfg), 4),
                "box": copy.deepcopy(member.get("box")),
                "quad": copy.deepcopy(member.get("quad")),
                "word_count": len(member.get("words") or []),
                "source_kind": (member.get("meta") or {}).get("source_kind", "line"),
                "selected": index == winner_index,
            })
        meta = copy.deepcopy(winner.get("meta") or {})
        meta.update({
            "engine": engines[winner_index],
            "selected_engine": engines[winner_index],
            "support_engines": supporting_engines,
            "agreement": round(sum(
                _text_similarity(winner.get("text"), members[index].get("text"))
                for index in range(len(members))
            ) / len(members), 4),
            "provenance": provenance,
        })
        unique_texts = sorted({member.get("text", "") for member in members})
        if len({_text_key(value) for value in unique_texts}) > 1:
            meta["disagreement"] = unique_texts
        winner["meta"] = meta
        output.append(winner)
    return output


# ---------------------------------------------------------------------------
# Conservative detector-fragment repair


def _fragment_options(cfg: Optional[dict]) -> dict:
    raw = ((cfg or {}).get("ocr") or {}).get("recombine_fragments", True)
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, dict):
        options = dict(raw)
        options.setdefault("enabled", True)
        return options
    return {"enabled": False}


def _oriented_union_quad(lines: list[dict], angle: float) -> list:
    """Create a tight enclosing rotated rectangle for adjacent OCR fragments."""
    radians = math.radians(angle)
    ux, uy = math.cos(radians), math.sin(radians)
    vx, vy = -uy, ux
    points = []
    for line in lines:
        quad = _normalize_quad(line.get("quad")) or _rect_quad(line.get("box") or {})
        points.extend(quad)
    if not points:
        return []
    along = [point[0] * ux + point[1] * uy for point in points]
    normal = [point[0] * vx + point[1] * vy for point in points]
    a0, a1, n0, n1 = min(along), max(along), min(normal), max(normal)
    return [
        [a0 * ux + n0 * vx, a0 * uy + n0 * vy],
        [a1 * ux + n0 * vx, a1 * uy + n0 * vy],
        [a1 * ux + n1 * vx, a1 * uy + n1 * vy],
        [a0 * ux + n1 * vx, a0 * uy + n1 * vy],
    ]


def _is_fragment_pair(left: dict, right: dict, options: dict) -> bool:
    """Whether two same-row detections are almost certainly one text line.

    This is intentionally stricter than generic proximity: merging two distinct buttons
    or price/CTA labels is worse than leaving a rare OCR split alone.  Fragment repair
    therefore requires a tight gap, matching angle/height/engine, and no sentence end.
    """
    if _base_engine(left) != _base_engine(right):
        return False
    left_box, right_box = _clean_box(left.get("box")), _clean_box(right.get("box"))
    left_h, right_h = max(1.0, left_box["h"]), max(1.0, right_box["h"])
    if _rotation_delta(_line_rotation(left), _line_rotation(right)) > _float(
            options.get("max_rotation_delta"), 7.5):
        return False
    if _vertical_overlap(left_box, right_box) < _float(options.get("min_vertical_overlap"), 0.72):
        return False
    if max(left_h, right_h) / min(left_h, right_h) > _float(options.get("max_height_ratio"), 1.55):
        return False
    gap = right_box["x"] - (left_box["x"] + left_box["w"])
    max_gap = max(left_h, right_h) * _float(options.get("max_gap_factor"), 0.9)
    if gap < -min(left_h, right_h) * 0.16 or gap > max_gap:
        return False
    left_text = str(left.get("text") or "").rstrip()
    right_text = str(right.get("text") or "").lstrip()
    if not left_text or not right_text:
        return False
    if re.search(r"[.!?;:]$", left_text):
        return False
    if re.match(r"^[,.;:!?\)\]\}]", right_text):
        return False
    # Full word geometry already tells us this is a line; joining two multi-word
    # detections tends to merge neighboring UI labels.  Permit a single-word
    # fragment on either side, which is the common detector-split case.
    left_words = left.get("words") or []
    right_words = right.get("words") or []
    if len(left_words) > 1 and len(right_words) > 1:
        return False
    return True


def _join_fragment_text(left: str, right: str) -> str:
    left, right = str(left or "").rstrip(), str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    # A visible same-line hyphen is normally authored ("high-quality"), not a
    # line-wrap hyphen.  Preserve it while avoiding a bogus extra space.
    if left.endswith(("-", "‐", "‑", "/")):
        return left + right
    return left + " " + right


def _merge_fragment_chain(lines: list[dict], options: dict) -> dict:
    fragments = sorted(lines, key=lambda line: line["box"]["x"])
    angle = sum(_line_rotation(line) for line in fragments) / len(fragments)
    quad = _oriented_union_quad(fragments, angle)
    text = ""
    words = []
    weights = []
    original = []
    for line in fragments:
        text = _join_fragment_text(text, line.get("text", ""))
        words.extend(copy.deepcopy(line.get("words") or []))
        weights.append(max(1, len(str(line.get("text") or "").strip())))
        original.append({
            "text": line.get("text", ""), "box": copy.deepcopy(line.get("box") or {}),
            "quad": copy.deepcopy(line.get("quad") or []), "confidence": _float(line.get("conf")),
        })
    result = copy.deepcopy(fragments[0])
    result["text"] = text
    result["conf"] = round(sum(_float(line.get("conf")) * weight
                              for line, weight in zip(fragments, weights)) / sum(weights), 4)
    result["box"] = _union_boxes(line.get("box") or {} for line in fragments)
    result["quad"] = quad or _rect_quad(result["box"])
    result["words"] = words
    meta = result.setdefault("meta", {})
    meta["source_kind"] = "recombined-fragments"
    meta["fragments"] = original
    meta["fragment_recombine"] = {"count": len(fragments), "angle": round(angle, 3)}
    return result


def _recombine_fragments(lines: list[dict], cfg: Optional[dict] = None) -> list[dict]:
    """Repair short same-row OCR detector splits without collapsing UI neighbors."""
    options = _fragment_options(cfg)
    if not options.get("enabled") or len(lines) < 2:
        return copy.deepcopy(lines)
    max_chain = max(2, min(12, int(options.get("max_chain", 6))))
    # Row bands keep the operation local.  A line is consumed only once, preserving
    # the original z/reading-order evidence for every non-fragment observation.
    heights = sorted(_clean_box(line.get("box"))["h"] for line in lines)
    median_height = heights[len(heights) // 2] if heights else 16.0
    tolerance = max(3.0, median_height * 0.48)
    rows: list[dict] = []
    for line in sorted(lines, key=lambda item: (item["box"]["y"] + item["box"]["h"] / 2.0,
                                                item["box"]["x"])):
        center = line["box"]["y"] + line["box"]["h"] / 2.0
        match = next((row for row in rows if abs(center - row["center"]) <= tolerance), None)
        if match is None:
            rows.append({"center": center, "items": [line]})
        else:
            count = len(match["items"])
            match["center"] = (match["center"] * count + center) / (count + 1)
            match["items"].append(line)
    output = []
    for row in rows:
        items = sorted(row["items"], key=lambda item: item["box"]["x"])
        chain = []
        for line in items:
            if chain and len(chain) < max_chain and _is_fragment_pair(chain[-1], line, options):
                chain.append(line)
                continue
            if chain:
                output.append(_merge_fragment_chain(chain, options) if len(chain) > 1 else copy.deepcopy(chain[0]))
            chain = [line]
        if chain:
            output.append(_merge_fragment_chain(chain, options) if len(chain) > 1 else copy.deepcopy(chain[0]))
    return output


# ---------------------------------------------------------------------------
# Public pipeline


def run_ocr(img_path: str, cfg: Optional[dict] = None, run_dir: Optional[str] = None):
    schema = _load_schema()
    cfg = cfg or {}
    ocr_cfg = cfg.get("ocr") or {}
    primary_name = ocr_cfg.get("primary", "ppocr-v6")
    challenger_names = ocr_cfg.get("challengers") or []
    started = time.time()
    # #region agent log
    _agent_log(
        "ocr.py:run_ocr", "ocr run start",
        data={"primary": primary_name, "challengers": challenger_names, "device": str(cfg.get("device", "cpu"))},
        hypothesis_id="H3", run_dir=run_dir, cfg=cfg,
    )
    # #endregion

    errors = []
    fallback_errors: list[dict] = []
    primary_has_evidence = False
    try:
        primary_payload = _run_backend(primary_name, img_path, cfg)
        primary_engine = primary_payload.get("engine", primary_name)
        primary_lines = _targeted_retry(
            img_path, primary_payload.get("lines", []), primary_name, cfg
        )
        primary_has_evidence = bool(primary_lines)
    except Exception as error:
        # A configured challenger can still produce a usable observation set.  Do not hide
        # the failed primary: runtime_report.json turns this status into a benchmark failure
        # when the production policy requires active models.
        primary_engine = primary_name
        primary_lines = []
        errors.append({"engine": primary_name, "error": str(error), "role": "primary"})
        # #region agent log
        _agent_log(
            "ocr.py:run_ocr", "primary backend failed",
            data={"engine": primary_name, "error": str(error), "error_type": type(error).__name__},
            hypothesis_id="H1", run_dir=run_dir, cfg=cfg,
        )
        # #endregion
        print(f"[ocr] primary '{primary_name}' unavailable: {error}")
    challenger_sets = []
    successful_challengers = []
    engines_used = [] if errors else [primary_engine]
    for name in challenger_names:
        if name == primary_name:
            continue
        try:
            payload = _run_backend(name, img_path, cfg)
        except Exception as error:
            print(f"[ocr] challenger '{name}' unavailable, skipping: {error}")
            errors.append({"engine": name, "error": str(error), "role": "challenger"})
            # #region agent log
            _agent_log(
                "ocr.py:run_ocr", "challenger backend failed",
                data={"engine": name, "error": str(error), "error_type": type(error).__name__},
                hypothesis_id="H2", run_dir=run_dir, cfg=cfg,
            )
            # #endregion
            continue
        challenger_sets.append(payload.get("lines", []))
        engines_used.append(payload.get("engine", name))
        if payload.get("lines"):
            successful_challengers.append(name)

    if not engines_used:
        configured = {primary_name, *challenger_names}
        for name in _fallback_engine_names(cfg):
            if name in configured:
                continue
            try:
                payload = _run_backend(name, img_path, cfg)
            except Exception as error:
                print(f"[ocr] fallback '{name}' unavailable: {error}")
                fallback_errors.append({"engine": name, "error": str(error), "role": "fallback"})
                # #region agent log
                _agent_log(
                    "ocr.py:run_ocr", "fallback backend failed",
                    data={"engine": name, "error": str(error), "error_type": type(error).__name__},
                    hypothesis_id="H3", run_dir=run_dir, cfg=cfg,
                )
                # #endregion
                continue
            primary_engine = payload.get("engine", name)
            primary_lines = _targeted_retry(
                img_path, payload.get("lines", []), name, cfg
            )
            engines_used = [primary_engine]
            errors.append({
                "engine": name,
                "detail": "recovered via fallback after configured engines failed",
                "role": "fallback",
            })
            print(f"[ocr] using fallback engine '{primary_engine}' after configured backends failed")
            # #region agent log
            _agent_log(
                "ocr.py:run_ocr", "fallback backend recovered ocr",
                data={"engine": primary_engine, "lines": len(primary_lines)},
                hypothesis_id="H3", run_dir=run_dir, cfg=cfg,
            )
            # #endregion
            break

    if not engines_used:
        message = _format_ocr_failure(errors, fallback_errors=fallback_errors)
        # #region agent log
        _agent_log(
            "ocr.py:run_ocr", "all ocr backends failed",
            data={"errors": errors, "engines_used": engines_used},
            hypothesis_id="H3", run_dir=run_dir, cfg=cfg,
        )
        # #endregion
        raise RuntimeError(message)

    merged = _reconcile(primary_lines, challenger_sets, cfg=cfg) if challenger_sets else _reconcile(
        primary_lines, [], cfg=cfg
    )
    repaired = _recombine_fragments(merged, cfg=cfg)
    ordered = _order_lines(repaired)
    lines_out = []
    for line in ordered:
        output = {
            "id": line["id"],
            "text": line["text"],
            "conf": round(_float(line["conf"]), 4),
            "box": _clean_box(line["box"]),
            "quad": _normalize_quad(line.get("quad")) or _rect_quad(line["box"]),
            "words": copy.deepcopy(line.get("words") or []),
        }
        if line.get("meta"):
            output["meta"] = copy.deepcopy(line["meta"])
        lines_out.append(output)

    width = height = 0
    try:
        from PIL import Image
        with Image.open(img_path) as image:
            width, height = image.size
    except Exception:
        pass
    configured_engines = [primary_name, *challenger_names]
    successful_engines = ([primary_name] if primary_has_evidence else []) + successful_challengers
    cross_check = _cross_check_metrics(lines_out, configured_engines, successful_engines)
    geometry = _geometry_metrics(lines_out)
    # A configured challenger is evidence, not a best-effort hint.  Empty output
    # is indistinguishable from a failed cross-check and must remain visible.
    if cross_check["fail_closed"]:
        status = "partial"
    else:
        status = "ok" if not errors else "partial"
    result = {
        "engine": "+".join(dict.fromkeys(engines_used)),
        "status": status,
        "errors": errors,
        "source": {"path": img_path, "w": width, "h": height},
        "ms": round((time.time() - started) * 1000, 1),
        "lines": lines_out,
        "metrics": {
            "cross_check": cross_check,
            "geometry": geometry,
        },
    }
    if run_dir is None:
        run_dir = cfg.get("run_dir")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(result, os.path.join(run_dir, "ocr.json"))
    return result


def clear_engine_caches() -> None:
    """Release cached OCR backends so CUDA memory can be reclaimed between stages."""
    _PADDLE_ENGINES.clear()
    _DOCTR_ENGINES.clear()
    _SURYA_ENGINES.clear()
    _EASYOCR_ENGINES.clear()


if __name__ == "__main__":
    sample = [
        _make_line("world", 0.9, box={"x": 200, "y": 10, "w": 80, "h": 20}, engine="fixture"),
        _make_line("hello", 0.9, box={"x": 20, "y": 12, "w": 80, "h": 20}, engine="fixture"),
    ]
    assert [line["text"] for line in _order_lines(sample)] == ["hello", "world"]
    print("ocr normalization self-test: ok")
