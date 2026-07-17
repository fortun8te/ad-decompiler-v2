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

def _parse_langs(cfg: dict) -> list[str]:
    """Return OCR language preference list from config.

    Supports strings like "en", "en+nl", "en,nl" (order matters).
    """
    ocr_cfg = cfg.get("ocr") or {}
    raw = str(ocr_cfg.get("lang", "en") or "en").strip().lower()
    if raw in {"auto", "multi", "*"}:
        return ["en", "nl"]
    parts = [p.strip() for p in re.split(r"[+,]", raw) if p.strip()]
    return parts or ["en"]


def _primary_lang(cfg: dict) -> str:
    return _parse_langs(cfg)[0]


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


# OCR sometimes emits a bullet/mid-dot glyph that is already close to ·.
_INTERPUNCT_BULLET = re.compile(r"[•∙⋅‧]")
_INTERPUNCT_SEP = re.compile(r"(?<=[\w%€$£)\]]) [.\-] (?=[\w(€$£])")
_INTERPUNCT_TAIL = re.compile(r"(?<=[\w%€$£)\]]) [.\-]$")


def _restore_interpuncts(text: str) -> str:
    """Restore interpunct separators ('·') that OCR reads as '.'/'-'/bullets.

    UI metadata lines separate tokens with a centered dot ("05:00 PM · 12-05-2026 ·
    121K weergaven"); OCR engines emit '.' or '-' for that glyph. Only a SEPARATOR
    pattern is rewritten — a space-surrounded lone '.'/'-' between word characters,
    or trailing after a token — and only on lines that contain a digit (UI/meta
    lines), so prose punctuation and real hyphenated words are never touched.
    Decimals like ``1.2M`` stay intact because their '.' has no surrounding spaces.
    """
    if _INTERPUNCT_BULLET.search(text):
        text = _INTERPUNCT_BULLET.sub("·", text)
    if not any(ch.isdigit() for ch in text):
        return text
    text = _INTERPUNCT_SEP.sub(" · ", text)
    text = _INTERPUNCT_TAIL.sub(" ·", text)
    return text


def _make_line(text: Any, confidence: Any, quad: Any = None, box: Any = None,
               words: Optional[list] = None, engine: str = "unknown",
               meta: Optional[dict] = None) -> Optional[dict]:
    text = _restore_interpuncts(str(text or "").strip())
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
        # Interpunct restoration lives in _make_line, but cached payloads (written
        # by an older code path) and the single-line targeted-retry collapse both
        # bypass _make_line, so a UI meta line's centered '·' can survive as '.'/'-'
        # all the way to the render (benchmark 009's timestamp row). Re-run the
        # separator restore here — it is idempotent, so freshly parsed lines that
        # already carry '·' are unaffected — guaranteeing every engine line gets it.
        if "text" in line:
            line["text"] = _restore_interpuncts(str(line.get("text") or ""))
        line.setdefault("words", [])
        line.setdefault("quad", _rect_quad(line.get("box") or {}))
        line.setdefault("box", _quad_to_box(line["quad"]))
        # An emoji an engine read as a junk token ('GC' for 👀) is dropped from the
        # text but still stretches the box over the emoji's pixels, hiding it from
        # element detection.  Idempotent: a line without an orphan is untouched.
        orphan = _excise_orphan_edge_word(line)
        if orphan is not None:
            line.setdefault("meta", {})["orphan_edge_word_excised"] = orphan
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
    lang = _primary_lang(cfg)
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
    langs = _parse_langs(cfg)
    gpu = str(cfg.get("device", "cpu")).lower().startswith("cuda")
    key = (tuple(langs), gpu)
    reader = _EASYOCR_ENGINES.get(key)
    if reader is not None:
        return reader
    try:
        import easyocr
    except ImportError as error:  # pragma: no cover
        raise ImportError("EasyOCR backend requires easyocr.  pip install easyocr") from error
    reader = easyocr.Reader(langs, gpu=gpu)
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
    language = _primary_lang(cfg)
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


# ---------------------------------------------------------------------------
# Canonical disagreement set + product-region ownership
#
# One source of truth for "which OCR lines disagree across backends".  Every
# consumer (cross-check metrics, repair triggers, QA) must derive its count from
# these helpers so the numbers can never drift apart again.
#
# Two independent forces used to produce three different counts:
#   * ``metrics.cross_check.disagreements`` was computed once in run_ocr and never
#     refreshed, so it stayed at the pre-judge value while the live ``meta.disagreement``
#     flags dropped as vlm_ocr_judge confirmed lines (see refresh_cross_check).
#   * repair/QA counted every live flag, including text baked into product rasters
#     (packaging labels, nutrition panels) that is not part of the editable ad
#     contract (see disagreement_lines / product_region_boxes).
_PRODUCT_ELEMENT_ROLES = {"product", "image", "photo", "photo-fragment", "packshot", "packaging"}
_PRODUCT_ELEMENT_KINDS = {"photo-fragment", "photo", "raster", "image"}
_OWNED_MIN_CONTAINMENT = 0.6


def product_region_boxes(elements) -> list[dict]:
    """Bounding boxes of element-owned raster/product regions.

    These rasters carry text (labels, ingredient panels) that is intentionally
    baked into a swappable product image; OCR of that text must not feed editable
    -ad QA.  Accepts the pipeline's ``elements.json`` list (role/kind at the top
    level or under ``meta``)."""
    boxes: list[dict] = []
    for element in elements or []:
        if not isinstance(element, dict):
            continue
        meta = element.get("meta") or {}
        role = str(element.get("role") or meta.get("role") or "").lower()
        kind = str(element.get("kind") or meta.get("kind") or "").lower()
        if role not in _PRODUCT_ELEMENT_ROLES and kind not in _PRODUCT_ELEMENT_KINDS:
            continue
        box = element.get("box") or element.get("bbox")
        if isinstance(box, dict) and _float(box.get("w")) > 0 and _float(box.get("h")) > 0:
            boxes.append(box)
    return boxes


def _containment_fraction(inner: dict, outer: dict) -> float:
    """Fraction of ``inner``'s area contained inside ``outer`` (0..1)."""
    try:
        ix0 = max(_float(outer["x"]), _float(inner["x"]))
        iy0 = max(_float(outer["y"]), _float(inner["y"]))
        ix1 = min(_float(outer["x"]) + _float(outer["w"]), _float(inner["x"]) + _float(inner["w"]))
        iy1 = min(_float(outer["y"]) + _float(outer["h"]), _float(inner["y"]) + _float(inner["h"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = _float(inner.get("w")) * _float(inner.get("h"))
    return inter / area if area > 0 else 0.0


def line_in_product_region(line: dict, product_boxes: list[dict],
                           min_frac: float = _OWNED_MIN_CONTAINMENT) -> bool:
    box = line.get("box") if isinstance(line, dict) else None
    if not isinstance(box, dict) or not product_boxes:
        return False
    return any(_containment_fraction(box, pb) >= min_frac for pb in product_boxes)


def disagreement_lines(ocr, *, product_boxes: Optional[list[dict]] = None,
                       elements: Optional[list] = None, exclude_owned: bool = True) -> list[dict]:
    """Canonical editable-ad disagreement set: live ``meta.disagreement`` flags minus
    lines that sit inside an element-owned product/raster region.

    ``ocr`` may be the OCR result dict or a raw list of lines.  Pass ``product_boxes``
    (or ``elements``) to enable the ownership exclusion; with neither, every live flag
    is returned (still a single, consistent definition)."""
    lines = (ocr.get("lines") if isinstance(ocr, dict) else ocr) or []
    flagged = [ln for ln in lines
               if isinstance(ln, dict) and (ln.get("meta") or {}).get("disagreement")]
    if not exclude_owned:
        return flagged
    if product_boxes is None:
        product_boxes = product_region_boxes(elements or [])
    if not product_boxes:
        return flagged
    return [ln for ln in flagged if not line_in_product_region(ln, product_boxes)]


def refresh_cross_check(ocr: dict) -> dict:
    """Recompute ``metrics.cross_check`` counts from the live lines.

    run_ocr computes cross-check once; later stages (vlm_ocr_judge, vlm_proofread)
    pop ``meta.disagreement`` as they confirm/correct readings.  Call this after any
    such mutation so the stored metric equals the live flag count instead of drifting."""
    if not isinstance(ocr, dict):
        return ocr
    metrics = ocr.get("metrics")
    cross = metrics.get("cross_check") if isinstance(metrics, dict) else None
    if not isinstance(cross, dict):
        return ocr
    lines = ocr.get("lines") or []
    cross["disagreements"] = sum(
        1 for ln in lines if isinstance(ln, dict) and (ln.get("meta") or {}).get("disagreement"))
    cross["consensus_lines"] = sum(
        1 for ln in lines if len((ln.get("meta") or {}).get("support_engines") or []) > 1)
    agreements = [_float((ln.get("meta") or {}).get("agreement", 1.0))
                  for ln in lines if isinstance(ln, dict)]
    cross["mean_text_agreement"] = round(sum(agreements) / len(agreements), 4) if agreements else None
    present = cross.get("successful_engines") or []
    cross["consensus_ratio"] = (
        round(cross["consensus_lines"] / len(lines), 4) if lines and len(present) > 1 else None)
    return ocr


def _cross_check_metrics(lines: list[dict], configured: list[str], successful: list[str]) -> dict:
    disagreements = sum(1 for line in lines if (line.get("meta") or {}).get("disagreement"))
    agreements = [float((line.get("meta") or {}).get("agreement", 1.0)) for line in lines]
    supported = sum(1 for line in lines if len((line.get("meta") or {}).get("support_engines") or []) > 1)
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
        "consensus_lines": supported,
        "consensus_ratio": round(supported / len(lines), 4) if lines and len(present) > 1 else None,
        "mean_text_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
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


def _restore_dropped_punctuation(text: str, alternatives: Iterable[str]) -> str:
    """Splice back interior punctuation the selected reading dropped.

    OCR engines fuse per-region: the line ``text`` comes from the highest-scoring
    engine while the word tokens (and the peer readings) may come from another. When
    the winning engine drops a separator that a closely-agreeing engine preserved —
    easyocr reading ``UPFRONT.NL`` as ``UPFRONTNL`` while doctr kept the ``.`` — the
    period vanishes from the emitted line even though a peer clearly saw it.

    This aligns ``text`` against the most-similar alternative reading and restores
    *only* characters the winner is missing that are pure punctuation (never letters
    or digits, so a peer's own misreads — e.g. doctr's ``UPERONT`` — are not adopted).
    The winner's spelling wins every conflict; only dropped separators are re-inserted.
    General across ``.``/``-``/``/``/``:`` etc.; not a period special-case.
    """
    base = str(text or "")
    if not base:
        return base
    best_alt, best_ratio = None, 0.0
    for alt in alternatives:
        alt = str(alt or "")
        if not alt or _text_key(alt) == _text_key(base):
            continue
        ratio = SequenceMatcher(None, base, alt).ratio()
        if ratio > best_ratio:
            best_ratio, best_alt = ratio, alt
    # Require a genuinely close reading so unrelated peers can never inject glyphs.
    if best_alt is None or best_ratio < 0.85:
        return base
    out: list[str] = []
    matcher = SequenceMatcher(None, base, best_alt)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            segment = best_alt[j1:j2]
            # Only restore a separator the winner dropped, and only when it sits
            # between kept content (an interior/trailing separator, not a leading one).
            if (segment and out and i1 < len(base)
                    and all(not ch.isalnum() and not ch.isspace() for ch in segment)):
                out.append(segment)
        else:
            # equal / replace / delete: always keep the winner's own characters.
            out.append(base[i1:i2])
    return "".join(out)


# Glyph pairs display-font OCR reliably swaps, grouped into equivalence classes.
# Each class maps to one canonical member so two readings of the SAME line can be
# compared for "same ink, different glyph guesses" without the guesses themselves
# deciding the comparison.  Benchmark 7 / ad 131: doctr read ``FREE SHIPPING`` as
# ``FREE SHIPPINC`` (G->C) and ``GET 1 FREE`` as ``GETIFREE`` (1->I), while easyocr
# read both correctly — the classes below make those two readings normalize equal.
# Deliberately conservative: only pairs with a shared skeleton at display weight.
# V/Y, E/F, P/R etc. are NOT included — they are real letter differences that would
# let a peer overwrite a correct brand token (067's ``PROYK``).
_CONFUSABLE_CLASSES = (
    ("c", "g"),          # 131 SHIPPING->SHIPPINC, 067 SAYING->SAYINC / GOODBYE->COODBYE
    ("0", "o"),          # 131 $100+ -> $1OO+
    ("1", "i", "l", "|", "ı"),   # 131 GET 1 FREE -> GETIFREE
    ("5", "s"),
    ("8", "b"),
    ("2", "z"),
)
_CONFUSABLE_CANON = {
    member: klass[0] for klass in _CONFUSABLE_CLASSES for member in klass
}


def _confusable_key(text: Any) -> str:
    """Canonical form of ``text`` under known OCR glyph confusion.

    Case-folds, drops every non-alphanumeric character (spacing and punctuation are
    exactly what the confused engine gets wrong), then collapses each confusable
    class to its canonical member.  ``BUY2. GETIFREE + FREE SHIPPINC +OOLS`` and
    ``BUY 2, GET 1 FREE + FREE SHIPPING $1OO+`` share a ~0.87 ratio under this key
    while reading very differently as plain text.
    """
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(_CONFUSABLE_CANON.get(ch, ch) for ch in value if ch.isalnum())


def _confusable_similarity(a: Any, b: Any) -> float:
    """``_text_similarity`` computed on ``_confusable_key`` — "is this the same line?"."""
    aa, bb = _confusable_key(a), _confusable_key(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


# Ad-copy lexicon for *arbitration only*: it never rewrites text, it only decides
# which of two engine readings of the same line is more plausible, and both readings
# are scored against the same list — so a word missing here costs both equally.
# Absence is therefore "unknown", never "wrong" (brands, wordmarks and foreign copy
# live outside any list and must survive: 067 ``frøya``, 013 ``grüns``).
_LEXICON = frozenset("""
a about all also always an and any are as at available back be because been before best
better big black book both browse business but buy by call can cart check checkout claim
clean clear click code collection come comfort comfortable complete contact cover
customer customers cut daily day days deal deals design designed discount discover do
does dont down download each easy end ends enjoy every everyone exclusive experience
extra fast feel first fits for free fresh friday from full get gift gifts give glow go
good goodbye great grade guarantee had has have healthy hello help here high home hours
how in included ingredients inside instant into is it its join just keep key kit know
last learn left less life like limited live long look love low made make many market
may me minutes money month more most much must my natural need never new no not now
of off offer offers on once one online only or order orders organic our out over pack
packs per pick plus power premium price prices pro product products pure quality
quick ready real reduce refund return reusable review reviews right sale sales save
saving savings say saying see sell set shipping shop shopping single site size skin
small smart so soft sold some soon start stock stop store style subscribe such support
sure switch take taste team than that the their them then there these they this those
time to today top total try two up us use used value very view visit wait want was
way we wear week weeks well what when where which while who why will with within
without work world worth would year years you your yours
""".split())

# Glued display-copy phrases OCR emits as one token because the ad's letter-spacing
# is tighter than the word gap.  Curated, NOT lexicon-derived: a generic "split when
# both halves are words" rule also splits authored compounds (013's ``Superfoods`` ->
# ``Super foods``, ``grüns``' ``COMPREHENSIVE``), which is a worse error than the one
# it fixes.  Only fixed multi-word proper phrases that no brand would author solid.
# Ad 131: both engines read ``BLACK FRIDAY SALE`` glued (doctr ``BLACKFRIDAYSALE``,
# easyocr ``BLACKFRIDAY SALE``) — there is no peer evidence to recover the space from.
_GLUED_PHRASE_MAP = {
    "blackfriday": "black friday",
    "cybermonday": "cyber monday",
    "boxingday": "boxing day",
}


# Glyphs OCR engines confuse with a trailing exclamation mark on display fonts.
# Ad 013: doctr read ``do this!`` as ``do1 this`` (the '!' became a '1' and was
# relocated into the first token) while easyocr read ``do thisı`` (trailing
# dotless-i).  Both engines saw one stray thin glyph; neither kept the '!'.
_EXCLAIM_CONFUSABLE_TOKEN = re.compile(r"^([^\W\d_]{2,})([1lIı|!])$")
_PLAIN_ALPHA_TOKEN = re.compile(r"^[^\W\d_]+[!.,]?$")


def _exclaim_candidate(text: str) -> Optional[tuple[str, int, bool, bool]]:
    """Parse a reading as *alpha words + exactly one '!'-confusable glyph*.

    Returns ``(cleaned_text, glyph_token_index, at_line_end, saw_bang)`` when the
    line consists of alphabetic tokens plus exactly one token that ends in a
    '!'-confusable glyph (``do1``, ``thisı``, ``this!``); ``None`` otherwise.
    """
    tokens = str(text or "").split()
    if len(tokens) < 2:
        return None
    hits = [(index, _EXCLAIM_CONFUSABLE_TOKEN.match(token))
            for index, token in enumerate(tokens)]
    hits = [(index, match) for index, match in hits if match]
    if len(hits) != 1:
        return None
    index, match = hits[0]
    for other_index, token in enumerate(tokens):
        if other_index != index and not _PLAIN_ALPHA_TOKEN.match(token):
            return None
    cleaned = list(tokens)
    cleaned[index] = match.group(1)
    return (" ".join(cleaned), index, index == len(tokens) - 1, match.group(2) == "!")


def _fix_exclamation_confusion(winner_text: str, peer_texts: Iterable[str]) -> Optional[str]:
    """Deterministic trailing-'!' recovery when engines relocate/confuse the glyph.

    Fires only when the winner and a peer read the *same letters* but each carry a
    single stray '!'-confusable glyph in *different* token positions (relocation is
    the classic '!'-misread signature — ad 013 ``do1 this`` vs ``do thisı``), or the
    peer literally saw a trailing ``!``.  One of the readings must place the glyph
    at line end.  Returns the cleaned letters plus ``!``, or ``None``.
    """
    winner = _exclaim_candidate(winner_text)
    if winner is None:
        return None
    w_clean, w_index, w_end, w_bang = winner
    if w_bang:
        return None  # winner already carries the '!'
    for peer_text in peer_texts:
        peer_text = str(peer_text or "")
        if not peer_text.strip() or peer_text.strip() == str(winner_text or "").strip():
            continue
        peer = _exclaim_candidate(peer_text)
        if peer is None:
            continue
        p_clean, p_index, p_end, p_bang = peer
        if _text_key(p_clean) != _text_key(w_clean):
            continue
        if p_bang and p_end:
            return w_clean + "!"
        if not p_bang and p_index != w_index and (w_end or p_end):
            return w_clean + "!"
    return None


# --------------------------------------------------------------------------------
# Lexical arbitration between disagreeing engine readings of one line.
#
# `_reconcile` picks a winner mostly on calibrated confidence, which is a *glyph*
# confidence: it says nothing about whether the letters spell anything.  Benchmark 7
# ad 131 line 0 is the failure mode in full — both readings were present and the
# wrong one won on confidence alone:
#
#   doctr   0.765  "BUY2. GETIFREE + FREE SHIPPINC +OOLS"   <- selected
#   easyocr 0.519  "BUY 2, GET 1 FREE + FREE SHIPPING $1OO+" <- correct, discarded
#
# The VLM judge exists to arbitrate exactly this (meta.disagreement was set), but it
# is nondeterministic and errored on all 11 lines of that run, so the corruption
# shipped.  These backstops make the recovery deterministic.

_STRIP_EDGE_PUNCT = re.compile(r"^[^\w$€£¥]+|[^\w%+]+$")


def _token_core(token: str) -> str:
    """Token without decorative edge punctuation (keeps currency/percent affixes)."""
    return _STRIP_EDGE_PUNCT.sub("", str(token or ""))


def _classify_token(token: str) -> str:
    """One of ``punct`` / ``numeric`` / ``word`` / ``unknown`` / ``mixed``.

    ``numeric`` and ``word`` are *plausible* readings; ``mixed`` (letters spliced into
    a digit run, e.g. ``BUY2.``) is a positive corruption signal; ``unknown`` is an
    alphabetic token absent from the lexicon — neither credit nor evidence of damage.
    """
    core = _token_core(token)
    if not core:
        return "punct"
    body = core.strip("$€£¥%+")
    if not body:
        return "punct"
    has_alpha = any(ch.isalpha() for ch in body)
    has_digit = any(ch.isdigit() for ch in body)
    if has_digit and not has_alpha:
        return "numeric"
    if has_digit and has_alpha:
        return "mixed"
    if not has_alpha:
        return "punct"
    return "word" if body.casefold() in _LEXICON else "unknown"


def _lexical_plausibility(text: str) -> float:
    """Share of a line's content tokens that read as real words or clean numbers."""
    classes = [_classify_token(tok) for tok in str(text or "").split()]
    scored = [klass for klass in classes if klass != "punct"]
    if not scored:
        return 0.0
    plausible = sum(1 for klass in scored if klass in ("word", "numeric"))
    return plausible / len(scored)


def _corruption_signals(winner_text: str, peer_text: str) -> list[str]:
    """Positive evidence that ``winner_text`` is damaged (not merely unrecognised).

    A low lexical score alone must never promote a peer: an out-of-lexicon brand
    wordmark scores 0 while being perfectly correct.  A peer only wins when the
    winner shows a *signature* of engine damage.
    """
    signals: list[str] = []
    winner_tokens = str(winner_text or "").split()
    peer_tokens = str(peer_text or "").split()
    peer_words = {_token_core(tok).casefold() for tok in peer_tokens
                  if _classify_token(tok) == "word"}
    for token in winner_tokens:
        klass = _classify_token(token)
        core = _token_core(token)
        if klass == "mixed":
            signals.append(f"mixed-alnum:{token}")
            continue
        if klass != "unknown":
            continue
        # A non-word the peer reads as a real word using the same glyph skeleton is
        # the G->C / 1->I signature (SHIPPINC vs SHIPPING).
        key = _confusable_key(core)
        if any(_confusable_key(word) == key for word in peer_words):
            signals.append(f"confusable-of-word:{token}")
        elif len(core) >= 7:
            # Long unknown run where the peer spells the same skeleton as several
            # words — the space-collapse signature (GETIFREE vs GET 1 FREE).
            signals.append(f"glued-run:{token}")
    return signals


def _fix_confusable_against_peers(winner_text: str, peer_texts: Iterable[str]) -> Optional[str]:
    """Repair single tokens a peer spells as a real word with the same glyph skeleton.

    Token-local and order-free: for each winner token that is *not* a known word, look
    for any peer token that (a) IS a known word and (b) has the same ``_confusable_key``.
    ``FREE SHIPPINC`` + peer ``FREE SHIPPING`` -> ``FREE SHIPPING``.  The peer supplies
    only the glyph choice within a confusable class, never new letters, so a peer's own
    misreads cannot be adopted.  Returns ``None`` when nothing is confidently fixable.
    """
    tokens = str(winner_text or "").split()
    if not tokens:
        return None
    candidates: list[str] = []
    for peer_text in peer_texts:
        candidates.extend(str(peer_text or "").split())
    peer_words = [_token_core(tok) for tok in candidates
                  if _classify_token(tok) == "word"]
    if not peer_words:
        return None
    out = list(tokens)
    changed = False
    for index, token in enumerate(tokens):
        if _classify_token(token) != "unknown":
            continue
        core = _token_core(token)
        if len(core) < 3:
            continue
        key = _confusable_key(core)
        match = next((word for word in peer_words if _confusable_key(word) == key), None)
        if match is None or match.casefold() == core.casefold():
            continue
        out[index] = token.replace(core, _preserve_word_case(core, match), 1)
        changed = True
    return " ".join(out) if changed else None


def _lexicon_by_confusable_key() -> dict:
    """``{confusable_key: {words}}`` over the lexicon, built once."""
    global _LEXICON_KEYS
    if _LEXICON_KEYS is None:
        index: dict = {}
        for word in _LEXICON:
            index.setdefault(_confusable_key(word), set()).add(word)
        _LEXICON_KEYS = index
    return _LEXICON_KEYS


_LEXICON_KEYS: Optional[dict] = None


def _fix_confusable_against_lexicon(text: str) -> Optional[str]:
    """Repair display-caps glyph confusion with no peer reading to lean on.

    ``_fix_confusable_against_peers`` needs another engine to have spelled the word;
    when both engines agree on the misread — 131's bottom marquee reads ``SHIPPINC``
    twice, 067 reads ``COODBYE`` — there is no peer, and the VLM judge is the only
    thing standing between the ad and a corrupted headline. It errored on all 11 of
    131's lines, so this closes the gap deterministically.

    Fires only when an ALL-CAPS, purely alphabetic non-word maps to *exactly one*
    lexicon word under ``_confusable_key``. Uppercase-only because that is where the
    confusion is documented (display type has no x-height cues); purely alphabetic so
    prices and contractions (``DON'T``, ``$100+``) can never be touched; unique-match
    so an ambiguous skeleton is left for a human.
    """
    tokens = str(text or "").split()
    if not tokens:
        return None
    index = _lexicon_by_confusable_key()
    out = list(tokens)
    changed = False
    for position, token in enumerate(tokens):
        core = _token_core(token)
        if len(core) < 4 or not core.isalpha() or not core.isupper():
            continue
        if core.casefold() in _LEXICON:
            continue
        matches = index.get(_confusable_key(core)) or set()
        if len(matches) != 1:
            continue
        match = next(iter(matches))
        if match.casefold() == core.casefold():
            continue
        out[position] = token.replace(core, _preserve_word_case(core, match), 1)
        changed = True
    return " ".join(out) if changed else None


def _pick_lexical_peer(members: list[dict], winner_index: int) -> Optional[tuple[int, dict]]:
    """Promote a peer reading of the same line that is decisively more plausible.

    Every gate must hold, because overriding the confidence winner is the riskiest
    move in reconciliation:
      * both readings are >= 3 content tokens (never gamble on a short label),
      * they are the SAME line under ``_confusable_key`` (>= 0.6) — otherwise the
        cluster holds two genuinely different texts and this is not our call,
      * they are NOT already near-identical as plain text (nothing to arbitrate),
      * the winner carries at least one positive corruption signal,
      * the peer's plausibility beats the winner's by a decisive margin,
      * the peer splits into at least as many tokens (a fix must not lose words).
    """
    winner_text = str(members[winner_index].get("text") or "")
    winner_score = _lexical_plausibility(winner_text)
    if len(winner_text.split()) < 3:
        return None
    best: Optional[tuple[float, int, dict]] = None
    for index, member in enumerate(members):
        if index == winner_index:
            continue
        peer_text = str(member.get("text") or "")
        if len(peer_text.split()) < 3:
            continue
        if _text_key(peer_text) == _text_key(winner_text):
            continue
        if _confusable_similarity(peer_text, winner_text) < 0.6:
            continue
        if _text_similarity(peer_text, winner_text) >= 0.92:
            continue
        peer_score = _lexical_plausibility(peer_text)
        if peer_score - winner_score < 0.34:
            continue
        if len(peer_text.split()) < len(winner_text.split()):
            continue
        if not _corruption_signals(winner_text, peer_text):
            continue
        if best is None or peer_score > best[0]:
            best = (peer_score, index, member)
    if best is None:
        return None
    _, index, member = best
    return index, member


# Isolated dash-like tokens (`` - ``) that one engine hallucinates mid-line.
# Ad 066: doctr emitted ``SO YOU - DON'T`` (dash word conf 0.50) while easyocr
# read the same headline without any dash.
_STRAY_PUNCT_TOKEN = re.compile(r"^[-–—−|]{1,2}$")


def _strip_stray_punct_tokens(winner: dict, member_texts: list[str]) -> Optional[dict]:
    """Drop a low-evidence isolated dash token that near-identical peers lack.

    Only fires for an *interior* punctuation-only token, and only when either the
    winner's own word-level confidence for that token is weak (< 0.6) or at least
    two peer readings agree the line without it.  Legitimate dashes (one engine
    merely dropping a separator) are protected — that direction is handled by
    ``_restore_dropped_punctuation``.
    """
    text = str(winner.get("text") or "")
    tokens = text.split()
    if len(tokens) < 3:
        return None
    stray_indices = [index for index, token in enumerate(tokens)
                     if 0 < index < len(tokens) - 1 and _STRAY_PUNCT_TOKEN.match(token)]
    if not stray_indices:
        return None
    words = winner.get("words") or []
    peers = [str(value or "") for value in member_texts
             if str(value or "").strip() and _text_key(value) != _text_key(text)]
    for index in stray_indices:
        token = tokens[index]
        removed = tokens[:index] + tokens[index + 1:]
        removed_text = " ".join(removed)
        lacking = sum(1 for peer in peers if _text_key(peer) == _text_key(removed_text))
        if not lacking:
            continue
        word_conf = None
        for word in words:
            if str(word.get("text") or "").strip() == token:
                word_conf = _float(word.get("conf"), 1.0)
                break
        weak_word = word_conf is not None and word_conf < 0.6
        if not weak_word and lacking < 2:
            continue
        new_words = [word for word in words
                     if str(word.get("text") or "").strip() != token]
        return {
            "text": removed_text,
            "words": new_words,
            "dropped": token,
            "word_conf": word_conf,
            "peers_without": lacking,
        }
    return None


_STRAY_TRAILING_PUNCT = re.compile(r"^[,.;:·•]$")


def _strip_stray_trailing_punct(winner: dict, member_texts: list[str]) -> Optional[dict]:
    """Drop a trailing punctuation artifact that near-identical peers lack.

    The trailing-token sibling of ``_strip_stray_punct_tokens``: an engine can
    hallucinate a comma/period past the last glyph (benchmark 002's
    "KRACHTSPORT BUNDEL,").  Fires only when a peer reading agrees on the line
    without it AND either the winner's own word confidence for that token is weak
    or two peers lack it — so genuine sentence punctuation, which peers also read,
    is never stripped.  Attached punctuation ("uur.") is left alone: only a
    detached token is considered, mirroring the interior rule.
    """
    text = str(winner.get("text") or "")
    tokens = text.split()
    if len(tokens) < 2 or not _STRAY_TRAILING_PUNCT.match(tokens[-1]):
        return None
    words = winner.get("words") or []
    peers = [str(value or "") for value in member_texts
             if str(value or "").strip() and _text_key(value) != _text_key(text)]
    token = tokens[-1]
    removed_text = " ".join(tokens[:-1])
    lacking = sum(1 for peer in peers if _text_key(peer) == _text_key(removed_text))
    if not lacking:
        return None
    word_conf = None
    for word in words:
        if str(word.get("text") or "").strip() == token:
            word_conf = _float(word.get("conf"), 1.0)
            break
    weak_word = word_conf is not None and word_conf < 0.6
    if not weak_word and lacking < 2:
        return None
    new_words = [word for word in words
                 if str(word.get("text") or "").strip() != token]
    return {
        "text": removed_text,
        "words": new_words,
        "dropped": token,
        "word_conf": word_conf,
        "peers_without": lacking,
    }


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
    width, height = source.size
    scale = max(1.1, min(4.0, _float(options.get("scale"), 2.0)))
    small_height = max(4.0, _float(options.get("small_height"), 26.0))
    low_confidence = max(0.0, min(1.0, _float(options.get("low_confidence"), 0.72)))
    max_regions = max(0, min(32, int(options.get("max_regions", 6))))
    min_gain = max(0.0, _float(options.get("min_confidence_gain"), 0.025))
    # Text that runs to the image edge is frequently clipped: the detector box stops a
    # few px short of the last visible glyph.  Re-scan those lines on a crop that reaches
    # toward the margin so trailing/leading characters can be recovered.
    edge_recover = bool(options.get("edge_recover", True))
    edge_margin = _float(options.get("edge_margin"), 0.0)
    if edge_margin <= 0:
        edge_margin = max(4.0, width * 0.01)
    edge_extend_factor = max(1.0, _float(options.get("edge_extend_factor"), 4.0))
    edge_conf_tolerance = max(0.0, _float(options.get("edge_conf_tolerance"), 0.08))

    eligible = []
    for index, line in enumerate(lines):
        reasons = []
        if _float(line.get("conf")) < low_confidence:
            reasons.append("low-confidence")
        if _float(line.get("box", {}).get("h")) <= small_height:
            reasons.append("small-text")
        box = _clean_box(line.get("box") or {})
        if edge_recover and box["w"] > 0 and box["h"] > 0:
            if (width - (box["x"] + box["w"])) <= edge_margin:
                reasons.append("edge-right")
            if box["x"] <= edge_margin:
                reasons.append("edge-left")
        if reasons:
            priority = (_float(line.get("conf")), _float(line.get("box", {}).get("h")))
            eligible.append((priority, index, reasons))
    eligible.sort(key=lambda item: item[0])
    eligible = eligible[:max_regions]
    if not eligible:
        return copy.deepcopy(lines)

    output = copy.deepcopy(lines)
    with tempfile.TemporaryDirectory(prefix="ocr_retry_") as directory:
        for retry_index, (_, index, reasons) in enumerate(eligible):
            original = output[index]
            box = _clean_box(original.get("box") or {})
            padding = max(2, int(round(box["h"] * _float(options.get("padding_factor"), 0.35))))
            x0 = max(0, int(math.floor(box["x"] - padding)))
            y0 = max(0, int(math.floor(box["y"] - padding)))
            x1 = min(width, int(math.ceil(box["x"] + box["w"] + padding)))
            y1 = min(height, int(math.ceil(box["y"] + box["h"] + padding)))
            # Reach further toward a touched margin so a clipped glyph enters the crop.
            edge_extend = int(round(box["h"] * edge_extend_factor))
            if "edge-right" in reasons:
                x1 = min(width, int(math.ceil(box["x"] + box["w"] + padding + edge_extend)))
            if "edge-left" in reasons:
                x0 = max(0, int(math.floor(box["x"] - padding - edge_extend)))
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
            # An edge crop that reads a strict extension of the clipped line (the original
            # text as a prefix/suffix, now longer) is accepted even without a confidence
            # gain — it only ever adds the recovered glyphs, never rewrites the reading.
            original_key = _text_key(original.get("text"))
            candidate_key = _text_key(candidate.get("text"))
            recovered = (
                ("edge-right" in reasons or "edge-left" in reasons)
                and bool(original_key)
                and candidate_key != original_key
                and len(candidate_key) > len(original_key)
                and (candidate_key.startswith(original_key)
                     or candidate_key.endswith(original_key)
                     or original_key in candidate_key)
                and retry_conf >= original_conf - edge_conf_tolerance
            )
            selected = (
                (agreement >= 0.96 and retry_conf >= original_conf)
                or retry_conf >= original_conf + min_gain
                or (not str(original.get("text") or "").strip() and retry_conf > 0)
                or recovered
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
                "recovered_truncation": recovered,
            }
            if selected:
                # Full-image detection owns line placement.  The retry owns the
                # transcription/confidence and its useful word-level geometry.  A recovered
                # truncation additionally extends the box to cover the new glyphs.
                replacement = copy.deepcopy(candidate)
                if recovered:
                    union = _union_boxes([original["box"], candidate["box"]])
                    replacement["box"] = union
                    replacement["quad"] = _rect_quad(union)
                else:
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
        # Prefer an already-spaced case-split reading (``We NEVER``) over a smashed
        # primary (``WeNEVER``) when both engines saw the same tokens.
        winner_text = str(winner.get("text") or "")
        cleaned_winner = cleanup_line_text(winner_text)
        if cleaned_winner != winner_text:
            for index, member in enumerate(members):
                member_text = str(member.get("text") or "")
                if member_text == cleaned_winner or (
                    " " in member_text
                    and " " not in winner_text
                    and cleanup_line_text(member_text) == cleaned_winner
                ):
                    winner = copy.deepcopy(member)
                    winner_index = index
                    break
            else:
                winner["text"] = cleaned_winner

        # Lexical arbitration: calibrated confidence ranks glyph legibility, not
        # whether the letters spell anything. When a peer reads the same ink as real
        # words and the winner reads it as damaged nonsense, the peer is right no
        # matter how confident the winner was (131: doctr 0.765 "GETIFREE ... SHIPPINC
        # +OOLS" beat easyocr 0.519 "GET 1 FREE ... SHIPPING $1OO+"). Heavily gated —
        # see _pick_lexical_peer.
        promoted = _pick_lexical_peer(members, winner_index)
        if promoted is not None:
            peer_index, peer_member = promoted
            demoted_text = str(winner.get("text") or "")
            demoted_engine = engines[winner_index]
            winner = copy.deepcopy(peer_member)
            winner_index = peer_index
            promoted_text = str(winner.get("text") or "")
            fix_meta = winner.setdefault("meta", {})
            fix_meta["lexical_arbitration"] = {
                "from": demoted_text,
                "to": promoted_text,
                "from_engine": demoted_engine,
                "to_engine": engines[peer_index],
                "from_plausibility": round(_lexical_plausibility(demoted_text), 4),
                "to_plausibility": round(_lexical_plausibility(promoted_text), 4),
                "signals": _corruption_signals(demoted_text, promoted_text),
            }

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

        # Restore any interior separator the winning engine dropped but a closely
        # agreeing peer (or the borrowed word tokens) preserved — e.g. easyocr's
        # ``UPFRONTNL`` regains the ``.`` from doctr's ``UPFRONT.NL`` — without
        # adopting the peer's own letter misreads. Keeps the emitted line text and
        # its word tokens punctuation-consistent.
        if agreeing_indices:
            peer_texts = [
                members[index].get("text", "") for index in agreeing_indices
                if index != winner_index
            ]
            peer_texts.extend(
                " ".join(str(word.get("text") or "") for word in (winner.get("words") or []))
                for _ in (0,) if winner.get("words")
            )
            reconciled = _restore_dropped_punctuation(str(winner.get("text") or ""), peer_texts)
            if reconciled != str(winner.get("text") or ""):
                winner["text"] = reconciled

        # Deterministic backstops for engine-confusion patterns the VLM judge only
        # catches when it happens to sample the line (and never when it errors).
        all_peer_texts = [
            str(member.get("text") or "") for index, member in enumerate(members)
            if index != winner_index
        ]
        # G->C on display caps is the most repeated misread in the benchmark set
        # (067 SAYING->SAYINC / GOODBYE->COODBYE, 131 SHIPPING->SHIPPINC). When a peer
        # spelled the same skeleton as a real word, take its glyph choice — token-local,
        # so it also fires on lines the whole-reading promotion above declines.
        confusable_fixed = _fix_confusable_against_peers(
            str(winner.get("text") or ""), all_peer_texts
        )
        if confusable_fixed:
            fix_meta = winner.setdefault("meta", {})
            fix_meta["confusable_fix"] = {
                "from": str(winner.get("text") or ""),
                "to": confusable_fixed,
                "readings": [str(member.get("text") or "") for member in members],
            }
            winner["text"] = confusable_fixed

        exclaim_fixed = _fix_exclamation_confusion(str(winner.get("text") or ""), all_peer_texts)
        if exclaim_fixed:
            fix_meta = winner.setdefault("meta", {})
            fix_meta["exclamation_fix"] = {
                "from": str(winner.get("text") or ""),
                "to": exclaim_fixed,
                "readings": [str(member.get("text") or "") for member in members],
            }
            winner["text"] = exclaim_fixed
            # Keep word tokens consistent: strip the confusable from its word and
            # move the '!' to the final word so line/word text cannot diverge.
            words = winner.get("words") or []
            new_tokens = exclaim_fixed.split()
            if len(words) == len(new_tokens):
                for word, token in zip(words, new_tokens):
                    word["text"] = token
        stray = _strip_stray_punct_tokens(winner, [m.get("text", "") for m in members])
        if stray is not None:
            fix_meta = winner.setdefault("meta", {})
            fix_meta["stray_punct_dropped"] = {
                "from": str(winner.get("text") or ""),
                "to": stray["text"],
                "token": stray["dropped"],
                "word_conf": stray["word_conf"],
                "peers_without": stray["peers_without"],
            }
            winner["text"] = stray["text"]
            winner["words"] = stray["words"]
        trailing = _strip_stray_trailing_punct(winner, [m.get("text", "") for m in members])
        if trailing is not None:
            fix_meta = winner.setdefault("meta", {})
            fix_meta["stray_trailing_punct_dropped"] = {
                "from": str(winner.get("text") or ""),
                "to": trailing["text"],
                "token": trailing["dropped"],
                "word_conf": trailing["word_conf"],
                "peers_without": trailing["peers_without"],
            }
            winner["text"] = trailing["text"]
            winner["words"] = trailing["words"]

        # A singleton/orphan winner can have no peer above the agreement
        # threshold.  Treat the winner itself as the only supporting reading
        # instead of crashing the whole benchmark on max(empty).
        if agreeing_indices:
            best_calibrated = max(
                _calibrated_confidence(members[index], cfg) for index in agreeing_indices
            )
        else:
            best_calibrated = _calibrated_confidence(winner, cfg)
            if 0 <= winner_index < len(engines):
                supporting_engines = [engines[winner_index]]
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
        engine_count = len(set(engines))
        support_count = len(supporting_engines)
        agreement = float(meta["agreement"])
        # Keep detector confidence and consensus confidence separate. Downstream judges can
        # still select a high-confidence disputed line for review, while acceptance/reporting
        # sees that only one engine actually supported the chosen transcription.
        meta["consensus"] = {
            "engine_count": engine_count,
            "support_count": support_count,
            "dissent_count": max(0, engine_count - support_count),
            "unanimous": support_count == engine_count,
            "confidence": round(min(1.0, winner["conf"] * (.55 + .45 * agreement) *
                                    (support_count / max(1, engine_count)) ** .35), 4),
        }
        unique_texts = {member.get("text", "") for member in members}
        if len({_text_key(value) for value in unique_texts}) > 1:
            # Include the (possibly deterministically fixed) winner text so a later
            # VLM arbitration sees the corrected reading as a candidate instead of
            # only the raw engine misreads.
            winner_current = str(winner.get("text") or "")
            if winner_current:
                unique_texts.add(winner_current)
            meta["disagreement"] = sorted(unique_texts)
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
    gap = right_box["x"] - (left_box["x"] + left_box["w"])
    # Two same-engine detections whose boxes overlap horizontally (negative gap)
    # are almost certainly one split line.  Short-fragment baseline angles from
    # easyocr are noisy, so the strict rotation gate wrongly rejects them and the
    # halves survive as separate overlapping lines that the renderer glues — box
    # overlap eating the seam characters (benchmark 025: "Industrial-grade" +
    # "design" -> "Industrial-gradesign").  Relax the rotation tolerance once the
    # boxes actually overlap, where the overlap itself is the strong "same line"
    # signal.
    rotation_limit = _float(options.get("max_rotation_delta"), 7.5)
    if gap < 0:
        rotation_limit = max(rotation_limit,
                             _float(options.get("overlap_max_rotation_delta"), 15.0))
    if _rotation_delta(_line_rotation(left), _line_rotation(right)) > rotation_limit:
        return False
    if _vertical_overlap(left_box, right_box) < _float(options.get("min_vertical_overlap"), 0.72):
        return False
    if max(left_h, right_h) / min(left_h, right_h) > _float(options.get("max_height_ratio"), 1.55):
        return False
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


def _join_fragment_text(left: str, right: str, *, gap_em: Optional[float] = None) -> str:
    left, right = str(left or "").rstrip(), str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    # A visible same-line hyphen is normally authored ("high-quality"), not a
    # line-wrap hyphen.  Preserve it while avoiding a bogus extra space — UNLESS the
    # source boxes show a real inter-fragment gap (>= ~0.3 em): a gap that wide means
    # the glyph is a spaced dash/separator, not a joined hyphenate, so the space is
    # authored and must survive the join.
    wide_gap = gap_em is not None and gap_em >= 0.3
    if left.endswith(("-", "‐", "‑", "/")) and not wide_gap:
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
    previous_box: Optional[dict] = None
    for line in fragments:
        current_box = _clean_box(line.get("box") or {})
        gap_em = None
        if previous_box is not None:
            reference_h = max(1.0, min(previous_box["h"], current_box["h"]))
            gap_em = (current_box["x"] - (previous_box["x"] + previous_box["w"])) / reference_h
        text = _join_fragment_text(text, line.get("text", ""), gap_em=gap_em)
        previous_box = current_box
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
# Contained-duplicate suppression
#
# Ensemble reconciliation clusters observations by IoU/row alignment, which
# misses the "fragment inside a longer reading" case: one engine reads
# "geld terug tot €100." while another also emits "geld" (or a full timestamp
# row next to its own two halves).  Both survive to the compiler, paint twice,
# and leave ghost duplicate text.  A line whose tokens are a fuzzy contiguous
# subsequence of an overlapping longer line is the same observation, not new
# evidence — drop it and keep the fuller reading.


_PUNCT_TOKEN_RE = re.compile(r"^\W+$")
# lower→ALLCAPS smash only: "WeNEVER" → "We NEVER". Requires 2+ capitals so
# camelCase brands like "iPhone" are left alone.
_CASE_SMASH_RE = re.compile(r"(?<=[a-z])(?=[A-Z]{2,})")

# High-confidence UI OCR typos and glued display phrases. Keys are casefold; values
# are replaced case-preservingly, so a value may carry the space a glued key lost
# (``BLACKFRIDAY`` -> ``BLACK FRIDAY``, ad 131).
_OCR_TYPO_MAP = {
    "weergaver": "weergaven",  # X/Twitter "views" — final n misread as r
    **_GLUED_PHRASE_MAP,
}
_OCR_TYPO_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_OCR_TYPO_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _preserve_word_case(original: str, replacement: str) -> str:
    """Apply ``replacement`` while roughly preserving ALLCAPS / Title / lower casing."""
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original[:1].isupper() and original[1:].islower():
        # Title-case every word: a replacement may carry a space the misread token
        # lost ("Blackfriday" -> "Black Friday", not "Black friday"). Single-word
        # replacements are unaffected.
        return " ".join(word[:1].upper() + word[1:].lower()
                        for word in replacement.split())
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _apply_ocr_typos(text: str) -> str:
    """Word-boundary typo fixes from ``_OCR_TYPO_MAP`` (case-preserving)."""
    raw = str(text or "")
    if not raw or not _OCR_TYPO_MAP:
        return raw

    def _repl(match: re.Match) -> str:
        word = match.group(0)
        fix = _OCR_TYPO_MAP.get(word.casefold())
        return _preserve_word_case(word, fix) if fix else word

    return _OCR_TYPO_RE.sub(_repl, raw)


def _collapse_repeated_tokens(text: str) -> str:
    """Collapse consecutive duplicate tokens: ``do do this!`` → ``do this!``.

    OCR on dense display type sometimes emits the same word twice in one line.
    That ships as editable text and paints as a clear copy error (ad 013).
    Only collapses tokens that were already whitespace-separated.
    """
    parts = re.split(r"(\s+)", str(text or ""))
    if len(parts) <= 1:
        return str(text or "")
    out: list[str] = []
    prev_token = None
    for part in parts:
        if not part:
            continue
        if part.isspace():
            out.append(part)
            continue
        # Compare alnum core so "do" / "do!" still collapse, keeping the latter's punct.
        core = re.sub(r"^\W+|\W+$", "", part)
        prev_core = re.sub(r"^\W+|\W+$", "", prev_token) if prev_token else ""
        if core and prev_core and core.casefold() == prev_core.casefold():
            while out and out[-1].isspace():
                out.pop()
            if out:
                out.pop()
        out.append(part)
        prev_token = part
    return "".join(out)


def _split_case_smash(text: str) -> str:
    """Insert spaces at lower→Upper boundaries (``WeNEVER`` → ``We NEVER``)."""
    raw = str(text or "")
    if not raw:
        return raw
    tokens = re.split(r"(\s+)", raw)
    return "".join(
        _CASE_SMASH_RE.sub(" ", tok) if tok and not tok.isspace() else tok
        for tok in tokens
    )


# Letters that are the same ink as a digit at display weight. Restricted to the three
# unambiguous shapes; S/5, B/8 and Z/2 are excluded because they routinely appear as
# real letters inside alphanumeric product codes ("B2B", "5G", "SIZE10").
_DIGIT_LOOKALIKES = {"o": "0", "O": "0", "l": "1", "I": "1", "i": "1"}
# Currency/percent/sign affixes that may legitimately bracket a numeric token.
_NUMERIC_AFFIX_RE = re.compile(r"^([$€£¥]?)([^\W_]+)([%+]?[.,:;!]?)$")


def _fix_digit_run_letters(text: str) -> str:
    """Fold digit-lookalike letters back into an otherwise all-numeric token.

    Ad 131 read ``$100+`` as ``$1OO+`` (both engines) and the corrupted price shipped.
    Fires only when the token already proves it is a number — it must contain a real
    digit, and every remaining character must map to a digit — so ``6g``, ``5G``,
    ``B2B``, ``3D`` and ``100ri0`` are all left untouched.
    """
    def _fix_token(token: str) -> str:
        match = _NUMERIC_AFFIX_RE.match(token)
        if not match:
            return token
        prefix, body, suffix = match.groups()
        if len(body) < 2 or not any(ch.isdigit() for ch in body):
            return token
        if all(ch.isdigit() for ch in body):
            return token
        converted = "".join(_DIGIT_LOOKALIKES.get(ch, ch) for ch in body)
        if not converted.isdigit():
            return token
        return f"{prefix}{converted}{suffix}"

    return " ".join(_fix_token(tok) for tok in str(text or "").split())


def cleanup_line_text(text: str) -> str:
    """Deterministic OCR text hygiene: un-smash case merges, drop repeated tokens, safe typos."""
    cleaned = _split_case_smash(str(text or ""))
    cleaned = _collapse_repeated_tokens(cleaned)
    cleaned = _apply_ocr_typos(cleaned)
    cleaned = _fix_confusable_against_lexicon(cleaned) or cleaned
    cleaned = _fix_digit_run_letters(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


# Sub-glyph marks detectors emit for icon strokes, rule lines, chart ticks and
# gradient seams. Ad 016 shipped ``-``, ``- -``, ``- 6`` and ``.`` as real text lines:
# they paint as stray ink, and they inflate the text-recall denominator with strings
# no ground truth will ever contain.
_SUBGLYPH_MARK = frozenset("-–—−_.·•|,'\"’‘“”:;")


def _is_glyphless_noise(text: str) -> bool:
    """Whether a line carries no recoverable glyph content.

    Conservative on purpose — a dropped line is unrecoverable downstream:
      * any letter at all -> real text (``Off``),
      * two or more digits -> a real number (``45%``, ``$100+``),
      * a lone digit with no stray marks -> keep (131's ``8``),
      * everything else must be entirely sub-glyph marks (plus at most one digit
        alongside a mark, which is 016's ``- 6``).
    """
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if any(ch.isalpha() for ch in stripped):
        return False
    body = [ch for ch in stripped if not ch.isspace()]
    digits = [ch for ch in body if ch.isdigit()]
    if len(digits) >= 2:
        return False
    marks = [ch for ch in body if ch in _SUBGLYPH_MARK]
    if len(marks) + len(digits) != len(body):
        return False   # some other symbol ("%", "+", "€") — not ours to judge
    return bool(marks)


def _drop_glyphless_lines(lines: list[dict]) -> tuple[list[dict], list[str]]:
    """Remove sub-glyph noise lines; return the survivors and the dropped strings."""
    kept, dropped = [], []
    for line in lines:
        if _is_glyphless_noise(line.get("text")):
            dropped.append(str(line.get("text") or ""))
        else:
            kept.append(line)
    return kept, dropped


def cleanup_word_text(text: str) -> str:
    """Character-level hygiene for a single word token (never changes token count).

    Word tokens carry their own box, so only the fixes that rewrite glyphs *within* a
    token may run here — the space-inserting ones (case-smash, glued phrases) would
    put two words behind one box. Keeping this in step with the line matters: 131
    shipped a line reading ``FREE SHIPPING`` whose word_geometry still said
    ``SHIPPINC``, and build_design_json paints from word_geometry.
    """
    raw = str(text or "")
    if not raw.strip():
        return raw
    cleaned = _fix_confusable_against_lexicon(raw) or raw
    return _fix_digit_run_letters(cleaned)


def _apply_line_text_cleanup(lines: list[dict]) -> list[dict]:
    output = []
    for line in lines:
        item = copy.deepcopy(line)
        original = str(item.get("text") or "")
        cleaned = cleanup_line_text(original)
        if cleaned != original:
            item["text"] = cleaned
            meta = item.setdefault("meta", {})
            meta["ocr_text_cleanup"] = {"from": original, "to": cleaned}
            if item.get("ocr_text") is None:
                item["ocr_text"] = original
        # Keep word tokens in step with the line they belong to.
        for word in item.get("words") or []:
            word_original = str(word.get("text") or "")
            word_cleaned = cleanup_word_text(word_original)
            if word_cleaned != word_original:
                word["text"] = word_cleaned
                word.setdefault("meta", {})["ocr_text_cleanup"] = {
                    "from": word_original, "to": word_cleaned,
                }
        output.append(item)
    return output


def _dedup_options(cfg: Optional[dict]) -> dict:
    raw = ((cfg or {}).get("ocr") or {}).get("dedup_contained", True)
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, dict):
        options = dict(raw)
        options.setdefault("enabled", True)
        return options
    return {"enabled": False}


def _content_tokens(text: Any) -> list[str]:
    return [token for token in _text_key(text).split(" ")
            if token and not _PUNCT_TOKEN_RE.match(token)]


def _fuzzy_token_subsequence(inner: list[str], outer: list[str], min_similarity: float) -> bool:
    """True when ``inner`` appears (in order, contiguously, per-token fuzzy) in
    ``outer`` and is strictly shorter — i.e. a partial duplicate, not a peer."""
    if not inner or len(inner) >= len(outer):
        return False
    for start in range(len(outer) - len(inner) + 1):
        window = outer[start:start + len(inner)]
        if all(a == b or _text_similarity(a, b) >= min_similarity
               for a, b in zip(inner, window)):
            return True
    return False


def _box_containment(inner: dict, outer: dict) -> float:
    """Fraction of ``inner``'s area covered by ``outer``."""
    a, b = _clean_box(inner), _clean_box(outer)
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    area = a["w"] * a["h"]
    return (ix * iy) / area if area > 0 else 0.0


def _suppress_contained_duplicates(lines: list[dict], cfg: Optional[dict] = None) -> list[dict]:
    options = _dedup_options(cfg)
    if not options.get("enabled") or len(lines) < 2:
        return copy.deepcopy(lines)
    containment_min = _float(options.get("containment"), 0.55)
    token_similarity = _float(options.get("token_similarity"), 0.85)
    conf_tolerance = _float(options.get("conf_tolerance"), 0.15)

    tokens = [_content_tokens(line.get("text")) for line in lines]
    dropped: dict[int, int] = {}
    for i, inner in enumerate(lines):
        for j, outer in enumerate(lines):
            if i == j or j in dropped:
                continue
            if not _fuzzy_token_subsequence(tokens[i], tokens[j], token_similarity):
                continue
            if _box_containment(inner.get("box") or {}, outer.get("box") or {}) < containment_min:
                continue
            # Never let a low-confidence container absorb a clearly better
            # fragment reading; the fragment stays in the container's meta.
            if _float(inner.get("conf")) > _float(outer.get("conf")) + conf_tolerance:
                continue
            dropped[i] = j
            break

    output = []
    for index, line in enumerate(lines):
        if index in dropped:
            continue
        kept = copy.deepcopy(line)
        absorbed = [
            {
                "text": lines[k].get("text", ""),
                "conf": round(_float(lines[k].get("conf")), 4),
                "box": copy.deepcopy(lines[k].get("box") or {}),
                "engine": _base_engine(lines[k]),
            }
            for k, owner in dropped.items() if owner == index
        ]
        if absorbed:
            kept.setdefault("meta", {})["absorbed_duplicates"] = absorbed
        output.append(kept)
    return output


# ---------------------------------------------------------------------------
# High-risk numeric token verification (optional VLM pass)
#
# Short numeric tokens (counts, prices, times) carry no dictionary context, so a
# confidently wrong engine ships errors like a retweet count 66 read as 99.
# When engines disagreed on such a token — or the fused confidence is low — crop
# the region and ask the local VLM for a charset-constrained transcription.
# Bounded, config-gated, and never raises.


_NUMERIC_TOKEN_RE = re.compile(r"^[\s\d.,:%€$£+kKmM]{1,10}$")

_NUMERIC_PROMPT = (
    "This crop shows one short numeric label from an ad or app screenshot (a count, "
    "price, time, or percentage). Transcribe exactly the characters shown, digit for "
    "digit, keeping symbols (€, $, %, :, ., ,) and any K/M suffix. Do not guess a "
    "rounder or more common number — read the actual digits. Output only the "
    "characters, nothing else. If no legible number is visible, output an empty string."
)


def _edge_char_drop(original: str, answer: str) -> tuple[int, int]:
    """(leading, trailing) glyph counts an arbitration dropped from ``original``.

    Only pure edge truncations count.  A substitution ('5B%' -> '58%') leaves the
    glyph extent unchanged and must never move the box, so it reports (0, 0).
    """
    original, answer = str(original or ""), str(answer or "")
    if not answer or not original or answer == original or len(answer) >= len(original):
        return 0, 0
    if original.endswith(answer):
        return len(original) - len(answer), 0
    if original.startswith(answer):
        return 0, len(original) - len(answer)
    return 0, 0


def _peer_edge(meta: dict, box: dict, *, leading: bool) -> Optional[float]:
    """A dissenting engine's box edge for the same row, when it reads inside ours.

    The rejected reading is real geometric evidence: on 107 doctr swallowed the
    ↓-in-circle icon as a leading '0' (box x=288) while easyocr read the same row
    from x=378 — past the icon.  Only same-row peers strictly inside our own edge
    qualify, so an equally-wide or wider peer can never widen the winner.
    """
    y0, y1 = _float(box.get("y")), _float(box.get("y")) + _float(box.get("h"))
    best = None
    for reading in (meta.get("provenance") or []):
        if not isinstance(reading, dict) or reading.get("selected"):
            continue
        peer = _clean_box(reading.get("box") or {})
        if peer["w"] <= 0 or peer["h"] <= 0:
            continue
        py0, py1 = peer["y"], peer["y"] + peer["h"]
        overlap = min(y1, py1) - max(y0, py0)
        if overlap <= 0.4 * min(_float(box.get("h")), peer["h"]):
            continue
        edge = peer["x"] if leading else peer["x"] + peer["w"]
        inside = edge > _float(box.get("x")) if leading else edge < _float(box.get("x")) + _float(box.get("w"))
        if inside:
            best = edge if best is None else (max(best, edge) if leading else min(best, edge))
    return best


def _retighten_after_edge_char_drop(line: dict, original: str, answer: str) -> Optional[dict]:
    """Shrink a line box when arbitration drops glyphs off an edge.

    ``_verify_numeric_tokens`` rewrites only the TEXT: on 107 the VLM correctly
    turned '058%' into '58%' — the leading '0' was the ↓-in-circle icon — but the
    box kept spanning x=288..737, so the icon's pixels stayed owned by a text
    line.  Element detection then treats the region as text and the icon's raster
    chip is sliced against an empty alpha ledger (fallback.json alpha_px=10 for a
    88x89 box).  Re-tighten to the surviving glyphs so the freed region can be
    detected and sliced as artwork.  Estimation prefers a dissenting peer's edge
    (real ink evidence) and falls back to a proportional per-glyph estimate;
    whichever clips LESS wins, so a wrong guess can never eat live text.
    """
    lead, trail = _edge_char_drop(original, answer)
    if not (lead or trail):
        return None
    box = _clean_box(line.get("box") or {})
    if box["w"] <= 0 or box["h"] <= 0:
        return None
    per_char = box["w"] / max(1, len(str(original)))
    x0, x1 = box["x"], box["x"] + box["w"]
    meta = line.get("meta") or {}
    new_x0, new_x1 = x0, x1
    if lead:
        proportional = x0 + per_char * lead
        peer = _peer_edge(meta, box, leading=True)
        new_x0 = min(proportional, peer) if peer is not None else proportional
    if trail:
        proportional = x1 - per_char * trail
        peer = _peer_edge(meta, box, leading=False)
        new_x1 = max(proportional, peer) if peer is not None else proportional
    if new_x1 - new_x0 < 0.25 * box["w"]:
        return None
    tightened = {"x": float(new_x0), "y": box["y"],
                 "w": float(new_x1 - new_x0), "h": box["h"]}
    line["box"] = tightened
    line["quad"] = _rect_quad(tightened)
    for word in line.get("words") or []:
        wb = _clean_box(word.get("box") or {})
        if wb["w"] <= 0:
            continue
        if _text_key(word.get("text")) == _text_key(original):
            word["text"] = answer
            word["box"] = dict(tightened)
            word["quad"] = _rect_quad(tightened)
    return {"from": {k: round(v, 2) for k, v in box.items()},
            "to": {k: round(v, 2) for k, v in tightened.items()},
            "dropped_leading": lead, "dropped_trailing": trail}


_ORPHAN_GLYPH_MAX_CONF = 0.72


def _excise_orphan_edge_word(line: dict) -> Optional[dict]:
    """Free an edge word whose glyphs never reached the line text.

    An engine reads an emoji as a junk token — 009's trailing 👀 becomes 'GC'
    (conf 0.52) and its ⏳ becomes 'X' — which reconciliation drops from the TEXT
    while leaving it in ``words`` and inside the box union.  The line box then
    claims the emoji's pixels (the 👀 sits at x=620..657 and the line box ends at
    exactly 657), so element detection never gets to emit it as an image chip.
    The ⏳ only survives today by luck: its gap is wide enough that the box
    stopped short of it.  Excise the orphan and retighten to the real glyphs.

    Deliberately conservative: only a DETACHED, outermost word whose text is
    absent from the line text and which is weakly read.  Interior tokens (the
    '·' separators of a timestamp row) and any word the text actually uses are
    never touched.
    """
    words = [w for w in (line.get("words") or []) if _clean_box(w.get("box") or {})["w"] > 0]
    if len(words) < 2:
        return None
    text_tokens = {_text_key(token) for token in str(line.get("text") or "").split()}
    text_tokens.discard("")
    if not text_tokens:
        return None
    ordered = sorted(words, key=lambda w: _clean_box(w["box"])["x"])
    box = _clean_box(line.get("box") or {})
    removed = []
    for candidate, neighbour, leading in ((ordered[-1], ordered[-2], False),
                                          (ordered[0], ordered[1], True)):
        if _text_key(candidate.get("text")) in text_tokens:
            continue
        if _float(candidate.get("conf"), 1.0) > _ORPHAN_GLYPH_MAX_CONF:
            continue
        cb, nb = _clean_box(candidate["box"]), _clean_box(neighbour["box"])
        gap = (nb["x"] - (cb["x"] + cb["w"])) if leading else (cb["x"] - (nb["x"] + nb["w"]))
        if gap < 0.08 * max(1.0, box["h"]):
            continue
        removed.append(candidate)
    if not removed:
        return None
    keep = [w for w in words if w not in removed]
    if not keep or not any(_text_key(w.get("text")) in text_tokens for w in keep):
        return None
    xs0 = min(_clean_box(w["box"])["x"] for w in keep)
    xs1 = max(_clean_box(w["box"])["x"] + _clean_box(w["box"])["w"] for w in keep)
    tightened = {"x": float(min(xs0, box["x"] + box["w"])), "y": box["y"],
                 "w": float(max(0.0, xs1 - xs0)), "h": box["h"]}
    if tightened["w"] <= 0 or tightened["w"] >= box["w"]:
        return None
    line["box"] = tightened
    line["quad"] = _rect_quad(tightened)
    line["words"] = keep
    return {"from": {k: round(v, 2) for k, v in box.items()},
            "to": {k: round(v, 2) for k, v in tightened.items()},
            "excised": [{"text": str(w.get("text") or ""),
                         "conf": _float(w.get("conf"), 0.0),
                         "box": {k: round(v, 2) for k, v in _clean_box(w["box"]).items()}}
                        for w in removed]}


def _numeric_verify_options(cfg: Optional[dict]) -> dict:
    cfg = cfg or {}
    raw = (cfg.get("ocr") or {}).get("numeric_verify", True)
    if isinstance(raw, bool):
        options = {"enabled": raw}
    elif isinstance(raw, dict):
        options = dict(raw)
        options.setdefault("enabled", True)
    else:
        options = {"enabled": False}
    vlm = cfg.get("vlm") or {}
    # The pass needs a reachable local VLM; the root switch gates it exactly like
    # the other judge stages.
    if not vlm.get("enabled"):
        options["enabled"] = False
    options.setdefault("base_url", vlm.get("base_url"))
    options.setdefault("model", vlm.get("model"))
    options.setdefault("timeout_s", vlm.get("timeout_s"))
    return options


def _is_numeric_token(text: Any) -> bool:
    value = str(text or "").strip()
    return bool(value) and bool(_NUMERIC_TOKEN_RE.match(value)) and any(ch.isdigit() for ch in value)


def _risky_numeric_lines(lines: list[dict], min_conf: float) -> list[dict]:
    risky = []
    for line in lines:
        if not line.get("box") or not _is_numeric_token(line.get("text")):
            continue
        disagreement = bool((line.get("meta") or {}).get("disagreement"))
        if disagreement or _float(line.get("conf"), 1.0) < min_conf:
            risky.append((0 if disagreement else 1, _float(line.get("conf"), 1.0), id(line), line))
    risky.sort(key=lambda item: item[:2])
    return [line for *_, line in risky]


def _default_numeric_ask(crop: bytes, options: dict):
    from src import vlm_client

    return vlm_client.multi_pass_answer(
        crop,
        _NUMERIC_PROMPT,
        base_url=str(options.get("base_url") or vlm_client._DEFAULT_BASE_URL),
        model=str(options.get("model") or vlm_client._DEFAULT_MODEL),
        timeout_s=_float(options.get("timeout_s"), vlm_client._DEFAULT_TIMEOUT_S),
        max_tokens=int(options.get("max_tokens") or 200),
        passes=int(options.get("passes") or 2),
    )


def _numeric_crop_bytes(image, box: dict, padding: int = 3):
    """Crop the token and upscale small crops so tiny UI counts stay legible."""
    from src import vlm_client

    try:
        h = _float(box.get("h"))
        if 0 < h < 44:
            from PIL import Image as _Image
            import io

            raw = vlm_client.crop_box_bytes(image, box, padding)
            if raw is None:
                return None
            crop = _Image.open(io.BytesIO(raw))
            scale = max(2.0, 44.0 / max(1.0, h))
            scale = min(scale, 4.0)
            resized = crop.resize(
                (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                _Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            return buffer.getvalue()
        return vlm_client.crop_box_bytes(image, box, padding)
    except Exception:
        return None


def _verify_numeric_tokens(img_path: str, lines: list[dict], cfg: Optional[dict],
                           options: Optional[dict] = None, ask=None) -> Optional[dict]:
    """Verify high-risk numeric tokens in place.  Returns evidence or None."""
    options = options if options is not None else _numeric_verify_options(cfg)
    if not options.get("enabled") or not lines:
        return None
    try:
        from PIL import Image

        image = Image.open(img_path)
    except Exception:
        return None

    min_conf = _float(options.get("min_conf"), 0.90)
    max_regions = max(0, min(16, int(options.get("max_regions") or 4)))
    candidates = _risky_numeric_lines(lines, min_conf)[:max_regions]
    if not candidates:
        return None
    ask = ask or (lambda crop: _default_numeric_ask(crop, options))

    checked = corrected = errors = 0
    notes: list[dict] = []
    for line in candidates:
        crop = _numeric_crop_bytes(image, line["box"], int(options.get("padding") or 3))
        if crop is None:
            continue
        checked += 1
        original = str(line.get("text", "")).strip()
        try:
            answer, note = ask(crop)
        except Exception:
            answer, note = None, "vlm_error"
        if note == "vlm_error":
            errors += 1
            notes.append({"line_id": line.get("id"), "note": "vlm_error", "ocr_text": original})
            continue
        if note:
            notes.append({"line_id": line.get("id"), "note": note, "ocr_text": original})
            continue
        answer = str(answer or "").strip()
        # Charset-constrained acceptance: the judge may only produce another
        # plausible numeric token of comparable length, never free text.
        if (not answer or not _is_numeric_token(answer)
                or len(answer) > max(6, len(original) + 3)):
            notes.append({"line_id": line.get("id"), "note": "implausible_answer",
                          "answer": answer, "ocr_text": original})
            continue
        meta = copy.deepcopy(line.get("meta") or {})
        readings = meta.get("disagreement") or []
        evidence = {
            "answer": answer,
            "ocr_text": original,
            "readings": [str(value) for value in readings],
            "reason": "disagreement" if readings else "low-confidence",
        }
        if answer != original:
            line["ocr_text"] = original
            line["text"] = answer
            # The correction moved glyphs, so the geometry must follow it: a
            # dropped edge glyph was usually never text at all (107's ↓ icon read
            # as a leading '0'), and a stale box keeps a text line owning the
            # artwork's pixels.
            retightened = _retighten_after_edge_char_drop(line, original, answer)
            if retightened is not None:
                evidence["box_retightened"] = retightened
            corrected += 1
        # This *is* a VLM arbitration of the numeric reading; mark the line so
        # the later generic OCR judge does not spend budget re-arbitrating it.
        line["vlm_ocr_judged"] = True
        meta.pop("disagreement", None)
        meta["numeric_verify"] = evidence
        line["meta"] = meta
    return {
        "enabled": True,
        "model": str(options.get("model") or ""),
        "checked": checked,
        "corrected": corrected,
        "errors": errors,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public pipeline


# ---------------------------------------------------------------------------
# Deterministic strikethrough recovery (ad 091: struck-through headline word
# OCR'd as ``A900A`` because the strike ink pollutes recognition)


# A token with letters on both sides of digits (``A900A``) is essentially never a
# real word in ad copy — it is the classic strike-ink recognition signature.
_DIGIT_SANDWICH_RE = re.compile(r"(?<!\w)[^\W\d_]+\d+[^\W\d_]+(?!\w)|(?<!\w)\d+[^\W\d_]+\d+(?!\w)")


def _strikethrough_options(cfg: Optional[dict]) -> dict:
    raw = ((cfg or {}).get("ocr") or {}).get("strikethrough", True)
    if isinstance(raw, bool):
        options = {"enabled": raw}
    elif isinstance(raw, dict):
        options = dict(raw)
        options.setdefault("enabled", True)
    else:
        options = {"enabled": False}
    options.setdefault("max_lines", 4)
    options.setdefault("scale", 2.0)
    return options


def _is_strike_candidate(line: dict) -> bool:
    """Cheap gate: only lines where engines disagreed AND the selected reading
    carries a digit-sandwich token are worth the pixel-level strike check."""
    provenance = (line.get("meta") or {}).get("provenance") or []
    texts = {_text_key(entry.get("text")) for entry in provenance
             if str(entry.get("text") or "").strip()}
    if len(texts) < 2:
        return False
    return bool(_DIGIT_SANDWICH_RE.search(str(line.get("text") or "")))


def _detect_strike(crop):
    """Detect strike ink over text in an RGB crop.

    Two deterministic detectors:
    1. Foreign-color ink (a red scribble over dark text): ink pixels whose color is
       far from the dominant text ink, spanning a wide horizontal extent and
       crossing the line's vertical middle band.
    2. Same-color thin strike: a near-solid ink row band at mid-height spanning
       most of the line width (row fill far above normal glyph rows).

    Returns ``(mask, bbox)`` — a boolean pixel mask to erase and the strike's
    bounding box in crop coordinates — or ``None``.
    """
    import numpy as np

    arr = np.asarray(crop, dtype=np.int32)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 16:
        return None
    height, width = arr.shape[:2]
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    background = np.median(border, axis=0)
    dist_bg = np.abs(arr - background).sum(axis=2)
    ink = dist_bg > 120
    ink_fraction = float(ink.mean())
    if ink_fraction < 0.01 or ink_fraction > 0.9:
        return None

    # Detector 1: foreign-color strike ink.  Anti-aliased glyph edges blend
    # between the background and the text ink, so they sit ON the bg→text color
    # axis; measure each pixel's distance FROM that axis instead of a plain
    # color distance, so only genuinely foreign hues (a red scribble over dark
    # text) are flagged and glyph edges survive the mask.
    text_color = np.median(arr[ink], axis=0)
    axis = background - text_color
    axis_norm = float(np.sqrt((axis * axis).sum()))
    if axis_norm < 1.0:
        return None
    unit = axis / axis_norm
    rel = arr.astype(np.float64) - text_color
    along = (rel * unit).sum(axis=2)
    off_axis_sq = (rel * rel).sum(axis=2) - along * along
    off_axis = np.sqrt(np.clip(off_axis_sq, 0.0, None))
    foreign = ink & (off_axis > 60.0)
    foreign_fraction = float(foreign.mean())
    if 0.003 <= foreign_fraction <= 0.25:
        ys, xs = np.nonzero(foreign)
        span = (int(xs.max()) - int(xs.min()) + 1) / float(width)
        mid_band = ((ys >= 0.25 * height) & (ys <= 0.75 * height)).mean()
        if span >= 0.25 and mid_band >= 0.2:
            grown = foreign.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        grown |= np.roll(np.roll(foreign, dy, axis=0), dx, axis=1)
            bbox = {"x": int(xs.min()), "y": int(ys.min()),
                    "w": int(xs.max()) - int(xs.min()) + 1,
                    "h": int(ys.max()) - int(ys.min()) + 1}
            return grown, bbox

    # Detector 2: same-color thin horizontal strike at mid-height.
    row_fill = ink.mean(axis=1)
    rows = np.arange(height)
    band = (rows >= 0.3 * height) & (rows <= 0.7 * height)
    solid_rows = np.nonzero((row_fill >= 0.7) & band)[0]
    if solid_rows.size and solid_rows.size <= max(2.0, 0.3 * height):
        strike_ink = ink[solid_rows]
        xs = np.nonzero(strike_ink.any(axis=0))[0]
        if xs.size:
            span = (int(xs.max()) - int(xs.min()) + 1) / float(width)
            if span >= 0.7:
                mask = np.zeros_like(ink)
                mask[solid_rows] = ink[solid_rows]
                bbox = {"x": int(xs.min()), "y": int(solid_rows.min()),
                        "w": int(xs.max()) - int(xs.min()) + 1,
                        "h": int(solid_rows.max()) - int(solid_rows.min()) + 1}
                return mask, bbox
    return None


def _reocr_masked_crop(crop, mask, engine: str, cfg: dict, scale: float,
                       runner: Optional[Callable] = None) -> Optional[str]:
    """Erase the strike mask to the background color, upscale, and re-OCR."""
    import numpy as np
    from PIL import Image

    runner = runner or _run_backend
    arr = np.asarray(crop).copy()
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    background = np.median(border, axis=0).astype(arr.dtype)
    arr[mask] = background
    masked = Image.fromarray(arr)
    if scale > 1.0:
        masked = masked.resize(
            (max(1, int(round(masked.width * scale))),
             max(1, int(round(masked.height * scale)))),
            Image.LANCZOS,
        )
    with tempfile.TemporaryDirectory(prefix="ocr_strike_") as directory:
        path = os.path.join(directory, "strike_crop.png")
        masked.save(path)
        payload = runner(engine, path, cfg, use_cache=False)
    lines = payload.get("lines") or []
    if not lines:
        return None
    ordered = sorted(lines, key=lambda ln: (_float((ln.get("box") or {}).get("y")),
                                            _float((ln.get("box") or {}).get("x"))))
    text = " ".join(str(ln.get("text") or "").strip() for ln in ordered).strip()
    return text or None


def _fix_strikethrough_lines(img_path: str, lines: list[dict], cfg: Optional[dict],
                             runner: Optional[Callable] = None) -> list[dict]:
    """Recover struck-through text deterministically.

    When strike ink is detected over a disputed line, the strike is masked out and
    the crop re-OCR'd; the engine reading closest to the clean re-read wins.  If
    re-OCR is unavailable, a peer reading without the digit-sandwich pollution is
    preferred over the polluted winner.  Every strike sets ``meta.strikethrough``
    so text analysis can emit the strike decoration.  Never raises.
    """
    options = _strikethrough_options(cfg)
    if not options.get("enabled") or not lines:
        return lines
    candidates = [line for line in lines if line.get("box") and _is_strike_candidate(line)]
    if not candidates:
        return lines
    candidates = candidates[: max(0, int(options.get("max_lines", 4)))]
    try:
        import numpy as np  # noqa: F401  (detector dependency)
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        image.load()
    except Exception:
        return lines
    primary = str(((cfg or {}).get("ocr") or {}).get("primary", "doctr"))

    for line in candidates:
        box = _clean_box(line.get("box") or {})
        pad = max(2, int(round(box["h"] * 0.2)))
        x0 = max(0, int(box["x"] - pad))
        y0 = max(0, int(box["y"] - pad))
        x1 = min(image.width, int(math.ceil(box["x"] + box["w"] + pad)))
        y1 = min(image.height, int(math.ceil(box["y"] + box["h"] + pad)))
        if x1 - x0 < 16 or y1 - y0 < 8:
            continue
        crop = image.crop((x0, y0, x1, y1))
        detection = _detect_strike(crop)
        if detection is None:
            continue
        mask, strike_bbox = detection
        meta = line.setdefault("meta", {})
        meta["strikethrough"] = True
        meta["strikethrough_box"] = {
            "x": x0 + strike_bbox["x"], "y": y0 + strike_bbox["y"],
            "w": strike_bbox["w"], "h": strike_bbox["h"],
        }
        current = str(line.get("text") or "")
        readings = [
            (str(entry.get("engine") or ""), str(entry.get("text") or ""),
             _float(entry.get("calibrated_confidence"), _float(entry.get("confidence"))))
            for entry in meta.get("provenance") or []
            if str(entry.get("text") or "").strip()
        ]
        reocr_text = None
        try:
            reocr_text = _reocr_masked_crop(
                crop, mask, primary, cfg or {}, _float(options.get("scale"), 2.0),
                runner=runner,
            )
        except Exception as error:
            print(f"[ocr] strikethrough re-OCR failed: {error}")
        chosen = None
        method = None
        if reocr_text and readings:
            best = max(readings, key=lambda entry: _text_similarity(reocr_text, entry[1]))
            if (_text_similarity(reocr_text, best[1]) >= 0.6
                    and _text_key(best[1]) != _text_key(current)):
                chosen, method = best, "reocr-matches-peer"
        if chosen is None and _DIGIT_SANDWICH_RE.search(current):
            clean_peers = [entry for entry in readings
                           if _text_key(entry[1]) != _text_key(current)
                           and not _DIGIT_SANDWICH_RE.search(entry[1])]
            if clean_peers:
                chosen = max(clean_peers, key=lambda entry: entry[2])
                method = "clean-peer-preferred"
        if chosen is None:
            continue
        meta["strikethrough_fix"] = {
            "from": current,
            "to": chosen[1],
            "engine": chosen[0],
            "method": method,
            "reocr_text": reocr_text,
        }
        line["ocr_text"] = current
        line["text"] = chosen[1]
        if chosen[2] > 0:
            line["conf"] = round(min(1.0, chosen[2]), 4)
        # The old winner's word tokens no longer match the adopted reading.
        if line.get("words"):
            line["words"] = []
    return lines


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

    # A backend call that returned successfully but found zero lines is not useful OCR
    # evidence.  This distinction matters on broken GPU/runtime combinations which can
    # return an empty prediction instead of raising.  Try the configured fallback chain
    # whenever *all* configured engines are empty, not only when they threw exceptions.
    configured_has_evidence = bool(primary_lines) or any(bool(lines) for lines in challenger_sets)
    empty_configured = bool(engines_used) and not configured_has_evidence
    if not configured_has_evidence:
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
            if not primary_lines:
                fallback_errors.append({
                    "engine": name,
                    "error": "backend returned no text observations",
                    "role": "fallback",
                })
                continue
            engines_used = [primary_engine]
            challenger_sets = []
            errors.append({
                "engine": name,
                "detail": "recovered via fallback after configured engines failed or returned empty",
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

    # All engines may legitimately see no text (for example, a photo-only creative), so
    # do not invent OCR or abort.  Keep the run usable but visibly degraded: an empty model
    # response must never be reported as a fully healthy OCR stage.
    if empty_configured and not primary_lines and not any(challenger_sets):
        errors.append({
            "engine": "+".join(dict.fromkeys(engines_used)),
            "error": "all available OCR engines returned no text observations",
            "role": "empty-evidence",
        })

    merged = _reconcile(primary_lines, challenger_sets, cfg=cfg) if challenger_sets else _reconcile(
        primary_lines, [], cfg=cfg
    )
    merged = _suppress_contained_duplicates(merged, cfg=cfg)
    repaired = _recombine_fragments(merged, cfg=cfg)
    repaired = _apply_line_text_cleanup(repaired)
    # After cleanup, so a line only reduced to noise by it is caught too. Before
    # ordering/emit, so noise reaches neither design.json nor the recall denominator.
    repaired, dropped_noise = _drop_glyphless_lines(repaired)
    try:
        repaired = _fix_strikethrough_lines(img_path, repaired, cfg)
    except Exception as error:  # deterministic backstop must never sink OCR
        print(f"[ocr] strikethrough pass skipped: {error}")
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

    numeric_verify = None
    try:
        numeric_verify = _verify_numeric_tokens(img_path, lines_out, cfg)
    except Exception as error:  # the verification pass must never sink OCR
        print(f"[ocr] numeric verification skipped: {error}")

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
    if dropped_noise:
        result["metrics"]["subglyph_noise_dropped"] = {
            "count": len(dropped_noise),
            "texts": dropped_noise,
        }
    if numeric_verify and numeric_verify.get("checked"):
        result["numeric_verify"] = numeric_verify
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
