"""Hand-drawn / marker / script text detection and the render-back rasterization gate.

Cheap per-line statistics (stroke-width CV, baseline wobble) CANNOT by themselves
separate genuine handwriting from legitimate display faces.  Measured on benchmark-6
(``stroke_width_cv`` / ``baseline_wobble`` as computed below):

===========================  =====  =======  ======  ===================================
line                          CV     wobble  render  verdict
                                             -back
===========================  =====  =======  ======  ===================================
091 "Sharp"  (hand marker)    0.17    0.18    0.20   MUST rasterize
091 "Foggy and Steady"        0.26    0.16    --     MUST stay native (serif typeset)
013 "We NEVER"  (typeset)     0.22    0.20    0.33   MUST stay native
013 "do this!"  (typeset)     0.19    0.17    0.52   MUST stay native
025 "Why Everyone's"          0.18    0.23    0.41   MUST stay native
===========================  =====  =======  ======  ===================================

The hand-marker word has a LOWER stroke-width CV than the typeset lines it must be
separated from — a felt-tip pen lays a very uniform stroke.  Any stats-only threshold
that catches "Sharp" therefore also catches most typeset copy, and rasterizing editable
copy is the *worse* contract violation (HARD-CREATIVES-SPEC §11).  Render-back IoU is
likewise not self-sufficient: it is dominated by string length and font-match quality
(013's typeset "+ FREE GIFTS" renders back at 0.09, far *weaker* than "Sharp" at 0.20),
so it can gate but never decide.

This module therefore uses a two-stage classifier in which the VLM is the decider:

* **Stage A (cheap, high recall)** — ``stage_a_candidate`` flags a line as a *candidate*
  when loose ink/fit signals fire (a strong shape-match font that nonetheless fails to
  render back, a high stroke-width CV, or a positively script/decorative match).  It is
  tuned for recall: it is fine to over-flag here because Stage B is precise.  It does
  reject the lines that render back well, which is what keeps 013's "do this!" (0.52)
  and 025's headlines (0.41+) away from the gate entirely.
* **Stage B (decisive)** — ``vlm_classify`` crops the line and asks gemma-4-e4b (LM
  Studio :1234, via ``vlm_client.ask_vlm``) "is this hand-drawn/marker/script or a
  typeset font?" with a strict-JSON answer.  Cached per crop (``ask_vlm`` caches by
  image+prompt, so a line-id maps 1:1 onto a cache entry).
* **Render-back gate (necessary, not sufficient)** — ``decide`` rasterizes only when the
  fitted font genuinely cannot reproduce the ink (render-back IoU below
  ``renderback_weak``) AND the VLM confirms handwriting.  A VLM "typeset" verdict keeps
  the line native even at a weak render-back (the VLM protects editable copy); a
  confirmed hand-lettered line whose library font *does* reproduce the ink also stays
  native (a script face that passes render-back is better than a chip: still editable).

When the VLM is unavailable the module keeps every line NATIVE (``allow_stats_only``,
default off).  Stats-only rasterization is exactly the false-positive machine the table
above rules out, so the fail-safe direction is "leave it editable".

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
# Stage-B budget per image. Stage A flags ~1/3 of all lines by design; the VLM call is
# only worth spending on the lines a wrong font would actually show up on.
_MAX_VLM_LINES = 8

# Stage B asks for the LETTERFORM STYLE, not for authorship.
#
# "Was this drawn by a human hand?" is the wrong question, and measurably so. Asked that
# way (gemma-4-e4b, benchmark-6 crops) the model answers:
#     091 "Sharp"            -> handwritten=False, "typeset",     confidence 1.00
#     091 "Foggy and Steady" -> handwritten=True,  "handwriting", confidence 0.95
# i.e. exactly inverted on the two lines that matter — and the model is not being stupid:
#   * "Sharp" IS a marker *typeface* (stroke-width CV 0.17, smooth repeatable letterforms),
#     so "not drawn by a hand" is literally correct — yet no library font reproduces it
#     (best fit across the whole corpus: Rock Salt 0.27, Permanent Marker 0.25, all of
#     which render "SHARP" in caps), which is the fact that actually matters;
#   * "Foggy and Steady" is serif type with a hand-drawn red swipe struck THROUGH it, so
#     the crop does contain hand-drawing — the answer describes the crop, not the type.
#
# Authorship is therefore not the property we need. What decides whether a line can be
# reproduced is whether its FACE is an ordinary typeface (our corpus can substitute one)
# or a marker/script face (it cannot). Ask that, and tell the model to ignore both
# confounders the failure above exposed: marks struck through the text, and neighbouring
# glyphs clipped at the crop edge.
_VLM_PROMPT = (
    "This crop shows ONE line of lettering from an advertisement. "
    "Classify the STYLE OF THE LETTERFORMS themselves. "
    "IGNORE any mark scribbled or struck THROUGH the text: a marker or highlighter line "
    "drawn over the words is not part of the lettering. "
    "IGNORE any partial letters clipped at the very top or bottom edge, which belong to a "
    "neighbouring line. "
    "Answer with one of:\n"
    "  'plain_sans'  - an ordinary sans typeface (Helvetica/Arial/Inter/Roboto-like)\n"
    "  'plain_serif' - an ordinary serif typeface (Times/Georgia/Playfair-like)\n"
    "  'marker'      - casual brush/marker/felt-tip lettering with rounded or uneven "
    "strokes (Permanent Marker / Caveat / Comic-like), whether hand-drawn or a "
    "marker-style font\n"
    "  'script'      - joined-up cursive or calligraphic lettering\n"
    "  'other'       - none of the above\n"
    "When unsure, answer plain_sans. Reply with STRICT JSON only."
)

_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "style": {
            "type": "string",
            "enum": ["plain_sans", "plain_serif", "marker", "script", "other"],
        },
        "confidence": {"type": "number"},
    },
    "required": ["style", "confidence"],
    "additionalProperties": False,
}

# Styles with no ordinary-typeface substitute in the corpus.
_HAND_STYLES = frozenset({"marker", "script"})


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


def _vlm_root_enabled(cfg: Optional[dict]) -> bool:
    """Whether the pipeline's VLM is switched on at all (``vlm.enabled``).

    Stage B is a network call, so it follows the same convention as every other VLM
    stage (vlm_scene_text/vlm_font_judge): OFF unless configuration explicitly turns it
    on.  This keeps ``analyze_text`` a pure local computation for unit tests and for
    ``--no-vlm`` runs — a stage that silently dials an endpoint from a bare cfg trips
    vlm_client's circuit breaker and poisons unrelated callers.
    """
    if not isinstance(cfg, dict):
        return False
    root = cfg.get("vlm")
    return bool(root.get("enabled")) if isinstance(root, dict) else False


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


def stroke_width_mean(mask) -> Optional[float]:
    """Mean stroke width (px) across an ink mask, or ``None``.

    Same distance-transform ridge estimate as :func:`stroke_width_cv`, but returns the
    absolute width rather than its scatter: on the medial ridge the distance-to-
    background is half the local stroke width, so the ridge mean doubled is the stroke
    width.  Unlike ink DENSITY (ink pixels / box area) this does not move when a word's
    glyph composition or letter-spacing changes, which makes the ratio of a word's
    stroke width to its line's median an evidence source for a real weight change that
    is independent of density.
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
    ink_d = d[m]
    ink_d = ink_d[ink_d > 0]
    if ink_d.size < 20:
        return None
    thresh = float(np.percentile(ink_d, 60.0))
    ridge = ink_d[ink_d >= max(thresh, 0.5)]
    if ridge.size < 8:
        ridge = ink_d
    return round(float(ridge.mean()) * 2.0, 4)


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


def select_candidates(candidates: list, cfg: Optional[dict]) -> list:
    """Bound the Stage-B VLM budget, spending it on the most consequential lines.

    Stage A is deliberately loose and fires on roughly a third of all lines (26/65 on
    091), so an unbounded Stage B would cost a VLM call per line for no benefit: a
    mis-set body-copy word is a small blemish, while a mis-set hand-lettered HEADLINE is
    the failure the user is asking about.  Rank by painted ink area (hand-lettering in
    ads is prominent by construction) and keep the top ``vlm.max_lines``.

    ``candidates`` is a list of ``(key, painted_box)`` pairs; the returned list is the
    subset of keys to classify, in priority order.
    """
    opts = _opts(cfg)
    vlm_opts = opts.get("vlm") if isinstance(opts.get("vlm"), dict) else {}
    raw = vlm_opts.get("max_lines", _MAX_VLM_LINES)
    max_lines = _MAX_VLM_LINES if raw is None else int(_num(raw, _MAX_VLM_LINES))
    if max_lines <= 0:
        return []

    def area(entry) -> float:
        box = entry[1] or {}
        try:
            return float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(candidates or [], key=area, reverse=True)
    return [key for key, _box in ranked[:max_lines]]


# --- Stage B: VLM classification ----------------------------------------------------

def vlm_classify(image_bytes: bytes, cfg: Optional[dict]) -> dict:
    """Ask the VLM whether a line crop is hand-drawn vs typeset.

    Returns ``{"available", "handwritten", "style", "confidence", "note"}``.  On any
    VLM error/timeout/unavailability, ``available`` is False and the caller falls back
    to the render-back gate alone.
    """
    opts = _opts(cfg)
    vlm_opts = opts.get("vlm") if isinstance(opts.get("vlm"), dict) else {}
    # Default OFF (see _vlm_root_enabled): a VLM call must be opted into by config, and
    # only when the pipeline's VLM is enabled at the root.
    if not vlm_opts.get("enabled", False) or not _vlm_root_enabled(cfg):
        return {"available": False, "note": "vlm-disabled"}
    if image_bytes is None:
        return {"available": False, "note": "no-image"}
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
    if not isinstance(parsed, dict) or "style" not in parsed:
        return {"available": False, "note": "unparseable", "raw": (raw or "")[:200]}
    style = str(parsed.get("style") or "").strip().lower()
    return {
        "available": True,
        # "handwritten" here means "a face our corpus of ordinary typefaces cannot
        # substitute" (marker/script) — see _VLM_PROMPT on why authorship is not the
        # property being asked about.
        "handwritten": style in _HAND_STYLES,
        "style": style,
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

    # VLM unavailable. Stats cannot decide this: 091's hand-marker "Sharp" scores a
    # LOWER stroke-width CV (0.17) than the typeset lines it must be separated from
    # (013 "We NEVER" 0.22, 091 "Foggy and Steady" 0.26), and its render-back (0.20) is
    # STRONGER than typeset "+ FREE GIFTS" (0.09).  A stats-only rule that fires on
    # "Sharp" therefore rasterizes editable copy wholesale — the exact HARD constraint
    # this module exists to protect.  Fail safe: keep the line native and say why.
    if not bool(opts.get("allow_stats_only", False)):
        result["reason"] = "vlm-unavailable:keeping-native (stats cannot separate handwriting)"
        return result
    cv = (stats or {}).get("stroke_width_cv")
    strong_cv = cv is not None and cv >= cv_strong
    if weak_renderback and strong_cv:
        result["handwriting"] = True
        result["rasterize"] = True
        result["reason"] = "stats-only:weak-renderback+strong-stroke-cv"
    else:
        result["reason"] = "stats-only:insufficient-corroboration"
    return result


__all__ = [
    "enabled", "stroke_width_cv", "baseline_wobble", "ink_stats",
    "stage_a_candidate", "select_candidates", "vlm_classify", "decide",
]
