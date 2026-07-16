"""Hand-drawn / marker / script text detection and the render-back rasterization gate.

The text agent measured that cheap per-line statistics (stroke-width CV, baseline
wobble) CANNOT by themselves separate genuine handwriting from legitimate display
faces: 091's hand-marker word "Sharp" (stroke-width CV ~0.48, baseline wobble ~0.18)
overlaps clean display headlines ("Foggy" 0.60, "ALLE" 0.55).  So a single threshold
either misses handwriting or rasterizes editable typeset copy — and rasterizing
editable copy is the *worse* contract violation (HARD-CREATIVES-SPEC §11).

This module therefore uses a two-stage classifier plus a decisive render-back gate:

* **Stage A (cheap, high recall)** — ``stage_a_candidate`` flags a line as a *candidate*
  when loose ink/fit signals fire (a strong shape-match font that nonetheless fails to
  render back, a high stroke-width CV, or a positively script/decorative match).  It is
  tuned for recall: it is fine to over-flag here because Stage B and the gate are
  precise.
* **Stage B (precise)** — ``vlm_classify`` crops the line and asks gemma-4-e4b (LM Studio
  :1234, via ``vlm_client.ask_vlm``) "is this hand-drawn/marker/script or a typeset
  font?" with a strict-JSON answer.  Cached per crop (``ask_vlm`` caches by image+prompt).
* **Render-back gate (decisive)** — ``decide`` rasterizes a line as a pixel-exact chip
  ONLY when the fitted font genuinely cannot reproduce the ink (render-back IoU below
  ``renderback_weak``) AND either the VLM confirms handwriting OR — when the VLM is
  unavailable — a *strong* stroke-CV corroboration is present.  When the VLM says
  "typeset", the line stays native even if the render-back is weak (the VLM protects
  editable copy).  When uncertain, keep native.

The rasterization itself reuses the existing low-fidelity plumbing: the caller stamps
``meta.low_fidelity`` + ``meta.fallback_src`` (an alpha-matted crop of the ORIGINAL ink,
produced by ``text_analysis._save_fallback_crop``) and ``routing._text_fidelity_fallback``
turns it into an image node.  ``meta.handwriting`` / ``ocr_text`` / ``font_attempted`` /
``renderback_score`` are stamped so the chip stays greppable and editable-aware in tooling.
"""

from __future__ import annotations

import math
from typing import Any, Optional


# --- Tunables (all overridable via cfg["text_analysis"]["handwriting"]) -------------

# Render-back IoU below which the fitted font is judged unable to reproduce the ink.
# 091 "Sharp" fits Barlow Condensed at 0.20; 013 "do this!" (typeset display) fits at
# 0.52 and must stay native — so the gate sits comfortably between them.
_RENDERBACK_WEAK = 0.35
# A candidate must have had a *plausible* font (high shape-match) for the
# "looks-right-but-renders-wrong" signal to mean handwriting rather than OCR garble.
_SHAPE_SCORE_MIN = 0.60
# Stroke-width coefficient of variation that, on its own, is strong enough to flag a
# candidate for the VLM, and (VLM-off only) strong enough to corroborate a raster.
_STROKE_CV_STRONG = 0.42
# Minimum alphanumeric glyphs: a 1-2 char fragment cannot be classified reliably.
_MIN_ALNUM = 3

_VLM_PROMPT = (
    "You are shown a tight crop of ONE line of text from an advertisement. "
    "Decide whether the lettering was DRAWN BY HAND (handwriting, felt-tip/marker, "
    "brush, or casual hand-lettering) or set with a TYPESET DIGITAL FONT (any regular "
    "typeface: sans, serif, display, condensed, or a clean computer script font). "
    "Uneven strokes, wobbly baselines, and connected marker letters mean hand-drawn. "
    "Crisp, uniform, repeatable letterforms mean typeset. When unsure, answer typeset. "
    "Reply with STRICT JSON only."
)

_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "handwritten": {"type": "boolean"},
        "style": {
            "type": "string",
            "enum": ["handwriting", "marker", "brush", "script", "typeset"],
        },
        "confidence": {"type": "number"},
    },
    "required": ["handwritten", "style", "confidence"],
    "additionalProperties": False,
}


def _opts(cfg: Optional[dict]) -> dict:
    """Resolve the ``text_analysis.handwriting`` options mapping (accepts a bare dict)."""
    if not isinstance(cfg, dict):
        return {}
    ta = cfg.get("text_analysis")
    if isinstance(ta, dict) and isinstance(ta.get("handwriting"), dict):
        return ta["handwriting"]
    if isinstance(cfg.get("handwriting"), dict):
        return cfg["handwriting"]
    return {}


def _num(value: Any, default: float) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def enabled(cfg: Optional[dict]) -> bool:
    """Whether handwriting detection runs at all (default ON)."""
    opts = _opts(cfg)
    return bool(opts.get("enabled", True))


def _alnum_count(text: Any) -> int:
    return sum(1 for ch in str(text or "") if ch.isalnum())


# --- Cheap ink statistics -----------------------------------------------------------

def stroke_width_cv(mask) -> Optional[float]:
    """Coefficient of variation (std/mean) of stroke width across an ink mask.

    Stroke width is estimated from the distance transform: on the medial ridge of a
    stroke, the distance-to-background equals half the local stroke width.  Handwriting
    and marker ink vary the pen pressure/speed, so the ridge distances scatter (high
    CV); a typeset face holds a near-constant stroke width (low CV).  Returns ``None``
    when the mask is too small or the estimate cannot be computed.
    """
    try:
        import numpy as np

        m = np.asarray(mask, dtype=bool)
    except Exception:
        return None
    if m.ndim != 2 or int(m.sum()) < 40 or min(m.shape) < 4:
        return None
    try:
        import cv2

        dist = cv2.distanceTransform(m.astype("uint8"), cv2.DIST_L2, 3)
    except Exception:
        try:
            from scipy import ndimage

            dist = ndimage.distance_transform_edt(m)
        except Exception:
            return None
    import numpy as np

    d = np.asarray(dist, dtype=np.float32)
    # Ridge pixels: local maxima approximated as the top distance quartile of ink, which
    # samples stroke centres (where distance == half stroke width) and ignores the
    # anti-aliased edge falloff that would otherwise depress the mean.
    ink_d = d[m]
    ink_d = ink_d[ink_d > 0]
    if ink_d.size < 20:
        return None
    thresh = float(np.percentile(ink_d, 60.0))
    ridge = ink_d[ink_d >= max(thresh, 0.5)]
    if ridge.size < 8:
        ridge = ink_d
    mean = float(ridge.mean())
    if mean <= 1e-3:
        return None
    return round(float(ridge.std()) / mean, 4)


def baseline_wobble(mask) -> Optional[float]:
    """Relative vertical scatter of per-column ink bottoms (baseline roughness).

    A typeset line sits on a straight baseline: per-column bottom-of-ink rows barely
    move.  Hand-lettered words ride an irregular baseline.  Returned as the std of the
    per-column bottom row normalised by the ink height, so it is scale-free.  ``None``
    when the mask is too small.
    """
    try:
        import numpy as np

        m = np.asarray(mask, dtype=bool)
    except Exception:
        return None
    if m.ndim != 2 or int(m.sum()) < 40 or m.shape[1] < 8:
        return None
    import numpy as np

    cols = np.where(m.any(axis=0))[0]
    if cols.size < 6:
        return None
    bottoms = []
    for c in cols:
        rows = np.where(m[:, c])[0]
        if rows.size:
            bottoms.append(float(rows.max()))
    if len(bottoms) < 6:
        return None
    bottoms = np.asarray(bottoms, dtype=np.float32)
    ys = np.where(m.any(axis=1))[0]
    height = float(ys.max() - ys.min() + 1) if ys.size else 0.0
    if height <= 1.0:
        return None
    # Detrend a slanted (italic) baseline so oblique typeset text is not mistaken for
    # wobble: fit a line to the bottoms and measure the residual scatter.
    xs = cols.astype(np.float32)
    try:
        coef = np.polyfit(xs, bottoms, 1)
        residual = bottoms - np.polyval(coef, xs)
    except Exception:
        residual = bottoms - bottoms.mean()
    return round(float(residual.std()) / height, 4)


def ink_stats(mask) -> dict:
    """Bundle the cheap ink statistics used by Stage A (each may be ``None``)."""
    return {
        "stroke_width_cv": stroke_width_cv(mask),
        "baseline_wobble": baseline_wobble(mask),
    }


# --- Stage A: cheap candidate flagging ----------------------------------------------

def _render_fit_score(line: dict) -> Optional[float]:
    rf = (line.get("meta") or {}).get("render_fit") or {}
    score = rf.get("score")
    return None if score is None else _num(score, None) if isinstance(score, (int, float)) else None


def _top_shape_score(line: dict) -> Optional[float]:
    cands = (line.get("style") or {}).get("fontCandidates") or []
    scores = [
        _num(c.get("score"), 0.0)
        for c in cands
        if isinstance(c, dict) and c.get("source") in {"local-render", "google-cache"}
    ]
    return max(scores) if scores else None


def _chosen_font_path(line: dict) -> Optional[str]:
    cands = (line.get("style") or {}).get("fontCandidates") or []
    for c in cands:
        if isinstance(c, dict) and c.get("path"):
            return str(c["path"])
    return None


def stage_a_candidate(line: dict, stats: Optional[dict], cfg: Optional[dict]) -> tuple[bool, dict]:
    """High-recall flag: is this line worth the (precise) Stage-B VLM check?

    Signals (any one flags):
      * ``renderback_mismatch`` — a strong shape-match font (>= shape_score_min) that
        nonetheless renders back below ``renderback_weak``.  This is the 091 "Sharp"
        signal: a plausible font that does not reproduce the ink.
      * ``stroke_cv`` — stroke-width CV at/above ``stroke_cv_strong``.
      * ``script_class`` — the chosen font file classes as script/decorative with a
        non-trivially-weak fit (handwriting matched to a swash face).

    Very short fragments and empty text never flag.  Returns ``(is_candidate, signals)``.
    """
    opts = _opts(cfg)
    renderback_weak = _num(opts.get("renderback_weak"), _RENDERBACK_WEAK)
    shape_min = _num(opts.get("shape_score_min"), _SHAPE_SCORE_MIN)
    cv_strong = _num(opts.get("stroke_cv_strong"), _STROKE_CV_STRONG)
    min_alnum = int(_num(opts.get("min_alnum"), _MIN_ALNUM))

    conf_min = _num(opts.get("ocr_conf_min"), 0.50)

    text = str(line.get("text") or "")
    signals: dict = {}
    if _alnum_count(text) < min_alnum:
        return False, {"skip": "too-short"}
    # A low OCR confidence means the ink is garbled/occluded, not hand-lettered — such a
    # line already routes to the ink fallback on its own merits. Skip it so the VLM
    # budget (and any raster decision) is spent only on cleanly-read lines.
    conf = _num(line.get("conf"), _num(line.get("ink_confidence"), 1.0))
    if conf < conf_min:
        return False, {"skip": f"low-conf:{conf:.2f}"}

    rf = _render_fit_score(line)
    shape = _top_shape_score(line)
    signals["render_fit"] = rf
    signals["shape_score"] = shape

    stats = stats or {}
    cv = stats.get("stroke_width_cv")
    wobble = stats.get("baseline_wobble")
    signals["stroke_width_cv"] = cv
    signals["baseline_wobble"] = wobble

    flags = []
    if (rf is not None and rf < renderback_weak
            and shape is not None and shape >= shape_min):
        flags.append("renderback_mismatch")
    if cv is not None and cv >= cv_strong:
        flags.append("stroke_cv")

    # Script/decorative font class with a weak-ish fit: a hand-lettered word that
    # matched a swash face.  Only counts when render-back is not strong.
    path = _chosen_font_path(line)
    if path and (rf is None or rf < 0.5):
        try:
            from src import font_fit

            cls = font_fit.classify_font_file(path)
            if cls in (font_fit.SCRIPT, font_fit.DECORATIVE):
                flags.append("script_class")
        except Exception:
            pass

    signals["flags"] = flags
    return bool(flags), signals


# --- Stage B: VLM classification ----------------------------------------------------

def vlm_classify(image_bytes: bytes, cfg: Optional[dict]) -> dict:
    """Ask the VLM whether a line crop is hand-drawn vs typeset.

    Returns ``{"available", "handwritten", "style", "confidence", "note"}``.  On any
    VLM error/timeout/unavailability, ``available`` is False and the caller falls back
    to the render-back gate alone.
    """
    opts = _opts(cfg)
    vlm_opts = opts.get("vlm") if isinstance(opts.get("vlm"), dict) else {}
    if not vlm_opts.get("enabled", True):
        return {"available": False, "note": "vlm-disabled"}
    try:
        from src import vlm_client
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "note": f"import-error:{type(exc).__name__}"}

    kwargs = {}
    if vlm_opts.get("base_url"):
        kwargs["base_url"] = vlm_opts["base_url"]
    if vlm_opts.get("model"):
        kwargs["model"] = vlm_opts["model"]
    kwargs["timeout_s"] = _num(vlm_opts.get("timeout_s"), 30.0)
    kwargs["max_tokens"] = int(_num(vlm_opts.get("max_tokens"), 96))

    try:
        raw = vlm_client.ask_vlm(
            image_bytes, _VLM_PROMPT, response_schema=_VLM_SCHEMA,
            reasoning_effort="none", **kwargs,
        )
    except Exception as exc:
        try:
            note, _detail = vlm_client.classify_vlm_exception(exc)
        except Exception:
            note = type(exc).__name__
        return {"available": False, "note": str(note)}

    parsed = _parse_json(raw)
    if not isinstance(parsed, dict) or "handwritten" not in parsed:
        return {"available": False, "note": "unparseable", "raw": (raw or "")[:200]}
    return {
        "available": True,
        "handwritten": bool(parsed.get("handwritten")),
        "style": str(parsed.get("style") or ""),
        "confidence": _num(parsed.get("confidence"), 0.0),
        "note": "ok",
    }


def _parse_json(raw: Any) -> Any:
    import json
    import re

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Tolerate a fenced or prose-wrapped object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


# --- Decision -----------------------------------------------------------------------

def decide(line: dict, stats: Optional[dict], image_bytes: Optional[bytes],
           cfg: Optional[dict]) -> dict:
    """Full two-stage + render-back decision for one line.

    Returns a dict::

        {"rasterize": bool, "handwriting": bool, "reason": str,
         "renderback_score": float|None, "font_attempted": str|None,
         "stage_a": {...}, "vlm": {...}}

    ``rasterize`` is True only when the fitted font cannot reproduce the ink AND the
    VLM confirms handwriting (or, VLM-off, a strong stroke-CV corroborates).  Typeset
    copy is never rasterized: a VLM "typeset" verdict keeps the line native even at a
    weak render-back, and low-confidence cases keep native.
    """
    opts = _opts(cfg)
    renderback_weak = _num(opts.get("renderback_weak"), _RENDERBACK_WEAK)
    cv_strong = _num(opts.get("stroke_cv_strong"), _STROKE_CV_STRONG)
    vlm_conf_min = _num(opts.get("vlm_confidence_min"), 0.55)

    is_candidate, stage_a = stage_a_candidate(line, stats, cfg)
    rf = _render_fit_score(line)
    font_attempted = None
    style = line.get("style") or {}
    cands = style.get("fontCandidates") or []
    if cands and isinstance(cands[0], dict):
        font_attempted = cands[0].get("family")
    font_attempted = font_attempted or style.get("fontFamily")

    result = {
        "rasterize": False,
        "handwriting": False,
        "reason": "",
        "renderback_score": rf,
        "font_attempted": font_attempted,
        "stage_a": stage_a,
        "vlm": None,
    }

    if not is_candidate:
        result["reason"] = "not-a-candidate"
        return result

    # Render-back must be genuinely weak to consider rasterizing at all — a font that
    # reproduces the ink is kept native regardless of any hand-drawn look.
    weak_renderback = rf is not None and rf < renderback_weak

    vlm = vlm_classify(image_bytes, cfg) if image_bytes is not None else {"available": False, "note": "no-image"}
    result["vlm"] = vlm

    if vlm.get("available"):
        if vlm.get("handwritten") and vlm.get("confidence", 0.0) >= vlm_conf_min:
            result["handwriting"] = True
            if weak_renderback:
                result["rasterize"] = True
                result["reason"] = "vlm-handwritten+weak-renderback"
            else:
                # Confirmed hand-lettered but a library face *does* reproduce it
                # (e.g. a script font passes render-back) — keep native with that font.
                result["reason"] = "vlm-handwritten-but-font-reproduces"
        else:
            # VLM says typeset (or low confidence): protect editable copy, keep native
            # even when render-back is weak (weak fit here is OCR garble/occlusion, not
            # handwriting).
            result["reason"] = "vlm-typeset"
        return result

    # VLM unavailable: render-back gate alone, but strict — require a strong stroke-CV
    # corroboration so we never rasterize typeset copy whose weak fit is just OCR noise.
    cv = (stats or {}).get("stroke_width_cv")
    strong_cv = cv is not None and cv >= cv_strong
    if weak_renderback and strong_cv:
        result["handwriting"] = True
        result["rasterize"] = True
        result["reason"] = "vlm-unavailable:weak-renderback+strong-stroke-cv"
    else:
        result["reason"] = "vlm-unavailable:insufficient-corroboration"
    return result


__all__ = [
    "enabled", "stroke_width_cv", "baseline_wobble", "ink_stats",
    "stage_a_candidate", "vlm_classify", "decide",
]
