"""Render-and-fit font refinement and glyph-class gating.

Two failure modes dominate text reconstruction quality (benchmark 009/052):

* the wrong *kind* of font is matched (a swash script substituted for a plain
  geometric sans), because mask shape-matching normalizes away exactly the
  aspect/spacing evidence that separates font classes; and
* size/letter-spacing estimates come from coarse heuristics (cap-height ratio,
  expected-advance tracking), so even a correctly matched family renders with
  visibly wrong tracking ("U P F R O N T") or a paragraph-breaking size.

This module fixes both with pixel evidence instead of new heuristics:

``fit_line``
    renders the recognized string with a candidate font file and optimizes font
    size and per-glyph tracking against the source ink mask, scoring the final
    fit with an aspect-preserving IoU.  Exact fonts score ~0.7-0.9, same-class
    lookalikes ~0.35-0.6, wrong-class fonts fall below ~0.3, so the score doubles
    as a font-confidence signal for the masked-pixel fidelity fallback.

``classify_source`` / ``classify_font_file`` / ``filter_fonts_by_class``
    a serif/sans/script gate applied *before* shape matching.  Candidate font
    files are classed from OS/2 PANOSE metadata (with filename heuristics as
    fallback); the source ink mask is classed comparatively, by fitting a few
    canonical reference fonts of each class to the same text and voting — the
    same-text comparison cancels the per-glyph biases that make absolute
    stroke-feature classifiers unreliable on short ad copy.

Only Pillow + NumPy (and fontTools, already a dependency) are required; every
import is lazy and every public function fails soft (returns ``None``/input).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Iterable, Optional

SANS = "sans"
SERIF = "serif"
SCRIPT = "script"
DECORATIVE = "decorative"

# Fitted-IoU floor below which a candidate is considered evidence *against*
# itself (wrong class / wrong shape); callers should fall to the next candidate
# and, if none passes, to the masked-pixel fallback.
DEFAULT_MIN_FIT_SCORE = 0.30
DEFAULT_MAX_FIT_CANDIDATES = 3
DEFAULT_FIT_ITERATIONS = 3
# Prefer Figma-insertable Google fonts over local-only files when the fitted
# evidence is essentially tied. Applies to on-disk OFL-corpus matches
# (source == "google-cache") and to matches whose family is already a Google
# font (``google_native``); a local-only face that must be remapped to a Google
# equivalent does not earn it, so a real Google match wins equal-score ties.
DEFAULT_GOOGLE_PREFERENCE_BONUS = 0.015

# Very short glyph runs (digits, "66", "21K") carry too little shape evidence to
# certify a family: their aspect-preserving IoU is high-variance and a lucky
# alignment reads as a near-perfect match (benchmark 009: "257"/"66" reported
# ~0.94).  The published fit score is therefore discounted by a length-reliability
# factor that ramps from _MIN_LEN_RELIABILITY at one glyph to 1.0 at
# _RELIABLE_GLYPHS glyphs, so a 2-3 glyph match can never publish exact-font
# confidence.  Comparative callers (classify_source voting, consensus) see the
# same proportional discount on both sides, so their *rankings* are unchanged;
# only the absolute, downstream-consumed confidence is corrected.
_RELIABLE_GLYPHS = 6
_MIN_LEN_RELIABILITY = 0.45

# Source-ink class gate.  A plain-text source (sans OR serif) fits both text
# reference classes far better than the script/decorative reference, even when
# sans-vs-serif is itself undecidable (short caps, digits).  ``classify_source``
# reports that separation as ``text_confidence`` so the caller can exclude
# script/decorative candidates without resolving sans-vs-serif — a swash face
# over plain body/headline copy (the Gabriola failure) is what renders visibly
# wrong, and it is reliably separable where sans-vs-serif is not.
DEFAULT_TEXT_MARGIN_SPAN = 0.08
DEFAULT_TEXT_MIN_SIGNAL = 0.20

# Coarse "plain text, not script" source class.  Admits sans and serif candidates
# and rejects script/decorative — used when the source is clearly text but the
# sans-vs-serif call is not decisive enough to hard-filter to one of them.
TEXT = "text"

_MAX_NEG_TRACKING_EM = 0.15
_MAX_POS_TRACKING_EM = 0.30
_SCORE_HEIGHT = 48
_MAX_RENDER_PIXELS = 8_000_000

_GLYPH_CLASS_CACHE: dict[str, Optional[str]] = {}
_FIT_CACHE: dict[tuple, Optional[dict]] = {}
_FIT_CACHE_LIMIT = 256
_REFERENCE_FONT_CACHE: dict[tuple, dict[str, list[str]]] = {}

_NAME_SCRIPT_RE = re.compile(
    r"script|hand|brush|comic|cursive|calligra|gabriola|mistral|freestyle", re.IGNORECASE
)

# Canonical per-class reference fonts used by the comparative source classifier.
# Basenames are resolved against the platform font directories; absolute paths
# are used as-is.  Two resolvable classes are enough to vote.
_REFERENCE_CANDIDATES: dict[str, list[str]] = {
    SANS: [
        "arial.ttf", "segoeui.ttf", "Arial.ttf", "Helvetica.ttc",
        "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
    ],
    SERIF: [
        "times.ttf", "georgia.ttf", "Times New Roman.ttf",
        "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf",
    ],
    SCRIPT: [
        "segoesc.ttf", "Gabriola.ttf", "Inkfree.ttf",
        "Comic Sans MS.ttf", "comic.ttf",
    ],
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def fit_options(config: Optional[dict]) -> dict:
    """Normalize ``text_analysis.render_fit`` (bool or mapping, default ON)."""
    raw = (config or {}).get("render_fit", True)
    if isinstance(raw, bool):
        return {"enabled": raw, "score_letter_spacing_zero": True}
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("enabled", True)
        # Candidate ranking and emit-time refits render at the emitted tracking (0),
        # so their fit score must judge the natural-advance width. classify_source
        # keeps the comparative tracking-optimized score (its own options path).
        out.setdefault("score_letter_spacing_zero", True)
        return out
    return {"enabled": False}


# ---------------------------------------------------------------------------
# Mask helpers (self-contained: text_analysis imports this module, not vice versa)


def _tight_mask(mask):
    import numpy as np

    if mask is None:
        return None
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2 or arr.size == 0 or not arr.any():
        return None
    ys, xs = np.nonzero(arr)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _mask_fingerprint(tight) -> str:
    import numpy as np
    from PIL import Image

    image = Image.fromarray(tight.astype(np.uint8) * 255)
    small = image.resize((32, 12), Image.Resampling.BILINEAR)
    return hashlib.sha1(np.asarray(small).tobytes()).hexdigest()[:16]


def _render_tracked_mask(text: str, font_path: str, size: float, tracking: float):
    """Tight ink mask of ``text`` rendered char-by-char with per-glyph tracking,
    exactly the way the preview renderer and Figma apply letterSpacing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        font = ImageFont.truetype(font_path, max(1, int(round(size))))
        probe = Image.new("L", (8, 8), 0)
        bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
        height = max(1, bbox[3] - bbox[1])
        advances = [font.getlength(ch) for ch in text]
        total = sum(advances) + tracking * max(0, len(text) - 1)
        width = max(1, int(math.ceil(total)) + 16)
        if width * (height + 16) > _MAX_RENDER_PIXELS:
            return None
        canvas = Image.new("L", (width, height + 16), 0)
        draw = ImageDraw.Draw(canvas)
        x = 8.0
        for ch, advance in zip(text, advances):
            draw.text((x, 8 - bbox[1]), ch, fill=255, font=font)
            x += advance + tracking
        return _tight_mask(np.asarray(canvas) > 32)
    except Exception:
        return None


def _scaled_profile(mask, height: int):
    """Mask resampled to a fixed height, preserving its own aspect ratio."""
    import numpy as np
    from PIL import Image

    h, w = mask.shape
    width = max(1, int(round(w * height / max(1, h))))
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _aligned_iou(source, rendered, height: int = _SCORE_HEIGHT) -> float:
    """IoU of two tight ink masks at a common height, *without* width
    normalization: aspect (letter-spacing/width) mismatches lower the score
    instead of being resized away, which is what lets an aspect-blind matcher
    pick a swash script for plain sans body text."""
    import numpy as np

    a = _scaled_profile(source, height)
    b = _scaled_profile(rendered, height)
    width = max(a.shape[1], b.shape[1])
    pad_a = np.zeros((height, width), dtype=np.float32)
    pad_a[:, : a.shape[1]] = a
    pad_b = np.zeros((height, width), dtype=np.float32)
    pad_b[:, : b.shape[1]] = b
    bin_a, bin_b = pad_a >= 0.35, pad_b >= 0.35
    union = float(np.logical_or(bin_a, bin_b).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(bin_a, bin_b).sum()) / union


def _normalized_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _glyph_count(text: Any) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def _length_reliability(text: Any) -> float:
    """Confidence multiplier for the fit score based on glyph count.

    A 1-3 glyph run cannot certify a family — its IoU is high-variance and a lucky
    alignment reads as a near-perfect match — so its published score is scaled
    toward the uncertain band, reaching full confidence only at ``_RELIABLE_GLYPHS``
    glyphs.  This is a *reliability* discount, not a shape penalty: it applies
    equally to right- and wrong-family renders, so it corrects overconfidence
    without disturbing which candidate ranks first.
    """
    n = _glyph_count(text)
    if n >= _RELIABLE_GLYPHS:
        return 1.0
    return round(_MIN_LEN_RELIABILITY + (1.0 - _MIN_LEN_RELIABILITY) * (n / _RELIABLE_GLYPHS), 4)


# ---------------------------------------------------------------------------
# Render-and-fit refinement


def fit_line(text: str, font_path: str, source_mask, initial_size: float,
             options: Optional[dict] = None) -> Optional[dict]:
    """Optimize font size and tracking of ``text`` in ``font_path`` against the
    source ink mask.  Returns ``{"fontSize", "letterSpacing", "score"}`` or
    ``None`` when the fit cannot be computed (missing font/mask/text).

    Size converges by matching tight glyph-run heights (both sides render the
    same string, so ascender/descender composition cancels — unlike the
    cap-height heuristic, which overestimates lines containing both).  Tracking
    is solved in closed form from the width residual and bounded to a plausible
    em range so it can never become the "tracked-out headline" artefact.
    """
    options = options or {}
    text = _normalized_text(text)
    if not text or not font_path or not os.path.exists(str(font_path)):
        return None
    tight = _tight_mask(source_mask)
    if tight is None:
        return None
    source_h, source_w = tight.shape
    if source_h < 4 or source_w < 4:
        return None

    key = (str(font_path), text, _mask_fingerprint(tight))
    if key in _FIT_CACHE:
        cached = _FIT_CACHE[key]
        return dict(cached) if cached else None

    size = max(4.0, min(512.0, _num(initial_size, source_h) or float(source_h)))
    tracking = 0.0
    gaps = max(1, len(text) - 1)
    iterations = max(1, min(6, int(_num(options.get("iterations"), DEFAULT_FIT_ITERATIONS))))
    # The pipeline ALWAYS emits letterSpacing 0 (Codia parity, contract §2/§7): every
    # text node renders at its font's natural advance width, so that is the width the
    # fit score must judge. Solving a per-gap tracking and scoring the *tracked* render
    # rewards a wrong-WIDTH face (a wide humanist sans squeezed with negative tracking)
    # exactly as much as the right-width one — which is how 002's squared/condensed
    # KRACHTSPORT headline matched wide Lato (tracked score 0.62) over a squared display
    # grotesque (Archivo 0.56) yet rendered at ink IoU ~0.19 once tracking was forced
    # back to 0. Score every candidate at the emitted tracking (0) so natural advance
    # width is decisive; the solved ``letterSpacing`` is retained only as a diagnostic.
    score_at_zero = bool(options.get("score_letter_spacing_zero", False))
    result = None
    for _ in range(iterations):
        rendered = _render_tracked_mask(text, font_path, size, tracking)
        if rendered is None:
            break
        size = max(4.0, min(512.0, size * source_h / max(1, rendered.shape[0])))
        natural = _render_tracked_mask(text, font_path, size, 0.0)
        if natural is None:
            break
        # Source width expressed at the candidate's pixel scale (heights match
        # after the size step), minus the natural advance width, per gap.
        target_w = source_w * natural.shape[0] / max(1, source_h)
        tracking = (target_w - natural.shape[1]) / gaps
        tracking = max(-_MAX_NEG_TRACKING_EM * size, min(_MAX_POS_TRACKING_EM * size, tracking))
    else:
        score_tracking = 0.0 if score_at_zero else tracking
        final = _render_tracked_mask(text, font_path, size, score_tracking)
        if final is not None:
            result = {
                "fontSize": round(size, 2),
                "letterSpacing": round(tracking, 3),
                "score": round(_aligned_iou(tight, final) * _length_reliability(text), 4),
            }

    _FIT_CACHE[key] = dict(result) if result else None
    while len(_FIT_CACHE) > _FIT_CACHE_LIMIT:
        _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
    return result


def refine_candidates(text: str, source_mask, candidates: list[dict], estimated_size: float,
                      options: Optional[dict] = None) -> tuple[list[dict], dict]:
    """Render-and-fit the top matched candidates and re-rank by fitted evidence.

    Each refined candidate gains a ``fit`` mapping (fontSize/letterSpacing/score,
    plus ``rejected`` when the best fit still scores below ``min_score``).
    Ranking: passing fits (best first, small Google-cache preference bonus so
    Figma-insertable families win ties) → unrefined render matches → fallbacks →
    rejected fits.  A rejected fit is *evidence against* that font, so it ranks
    below even a neutral fallback family.
    """
    options = options or {}
    evidence = {"enabled": bool(options.get("enabled", True)), "fitted": 0, "rejected": 0}
    if not evidence["enabled"] or not candidates:
        return list(candidates or []), evidence

    min_score = _num(options.get("min_score"), DEFAULT_MIN_FIT_SCORE)
    max_candidates = max(1, min(8, int(_num(options.get("max_candidates"), DEFAULT_MAX_FIT_CANDIDATES))))
    google_bonus = _num(options.get("google_preference_bonus"), DEFAULT_GOOGLE_PREFERENCE_BONUS)
    evidence["min_score"] = round(min_score, 4)

    refined: list[dict] = []
    fitted_count = 0
    for candidate in candidates:
        item = dict(candidate) if isinstance(candidate, dict) else candidate
        path = item.get("path") if isinstance(item, dict) else None
        renderable = isinstance(item, dict) and item.get("source") in {"local-render", "google-cache"}
        if renderable and path and fitted_count < max_candidates:
            fit = fit_line(text, path, source_mask, estimated_size, options)
            if fit is not None:
                fitted_count += 1
                fit = dict(fit)
                fit["rejected"] = fit["score"] < min_score
                item["fit"] = fit
                if fit["rejected"]:
                    evidence["rejected"] += 1
        refined.append(item)
    evidence["fitted"] = fitted_count

    def rank(indexed):
        index, item = indexed
        if not isinstance(item, dict):
            return (4, 0.0, index)
        fit = item.get("fit")
        score = _num(item.get("score"))
        # A Figma-loadable Google match is preferred on a near-tie: an on-disk
        # OFL corpus match (source == "google-cache") OR a match whose family is
        # already a Google font (``google_native``, set by the caller for
        # locally-installed Google families like Roboto/Inter). A local-only font
        # that must be *remapped* to a Google equivalent does not earn the bonus,
        # so a genuine Google match outranks a substitution on equal evidence.
        prefers_google = item.get("source") == "google-cache" or bool(item.get("google_native"))
        gb = google_bonus if prefers_google else 0.0
        if isinstance(fit, dict):
            effective = _num(fit.get("score")) + gb
            if not fit.get("rejected"):
                return (0, -effective, index)
            return (3, -effective, index)
        if item.get("source") in {"local-render", "google-cache"}:
            return (1, -(score + gb), index)
        return (2, -score, index)

    ordered = [item for _, item in sorted(enumerate(refined), key=rank)]
    return ordered, evidence


# ---------------------------------------------------------------------------
# Glyph-class gate


def _class_from_name(path: str) -> Optional[str]:
    stem = os.path.basename(str(path or "")).lower()
    if _NAME_SCRIPT_RE.search(stem):
        return SCRIPT
    if "slab" in stem:
        return SERIF
    if "sans" in stem:
        return SANS
    if "serif" in stem:
        return SERIF
    return None


def classify_font_file(path: str) -> Optional[str]:
    """Class a candidate font file as sans/serif/script/decorative (or None).

    Primary evidence is the OS/2 PANOSE record (family type + serif style),
    which Windows/Google font files fill in reliably; the filename heuristic
    covers fonts with an empty PANOSE.  Results are cached per path.
    """
    key = str(path or "")
    if not key:
        return None
    if key in _GLYPH_CLASS_CACHE:
        return _GLYPH_CLASS_CACHE[key]
    cls: Optional[str] = None
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(key, lazy=True, fontNumber=0)
        try:
            panose = font["OS/2"].panose
            family_type = int(panose.bFamilyType)
            serif_style = int(panose.bSerifStyle)
        except Exception:
            family_type = serif_style = -1
        font.close()
        if family_type == 2:
            if 11 <= serif_style <= 15:
                cls = SANS
            elif 2 <= serif_style <= 10:
                cls = SERIF
        elif family_type == 3:
            cls = SCRIPT
        elif family_type in (4, 5):
            cls = DECORATIVE
    except Exception:
        cls = None
    if cls is None:
        cls = _class_from_name(key)
    _GLYPH_CLASS_CACHE[key] = cls
    return cls


def compatible(source_class: Optional[str], candidate_class: Optional[str]) -> bool:
    """Whether a candidate class may be shape-matched against a source class.

    Unknown classes on either side never filter (fail open) — only a font we can
    positively class as script/decorative is ever excluded.  A ``SCRIPT`` source
    accepts script/decorative candidates; a confident ``SANS``/``SERIF`` source
    requires an exact match; a coarse ``TEXT`` source (plain text, sans-vs-serif
    undecided) admits both sans and serif but rejects script/decorative — a
    swash/decorative face must never reach plain body/headline copy.
    """
    if not source_class or not candidate_class:
        return True
    if source_class == SCRIPT:
        return candidate_class in (SCRIPT, DECORATIVE)
    if source_class == TEXT:
        return candidate_class in (SANS, SERIF)
    return candidate_class == source_class


def filter_fonts_by_class(fonts: Iterable[dict], source_class: Optional[str]) -> list[dict]:
    """Hard-filter discovered font metas by class; falls open when the filter
    would leave nothing to match."""
    fonts = list(fonts or [])
    if not source_class:
        return fonts
    kept = [meta for meta in fonts
            if compatible(source_class, classify_font_file(meta.get("path")))]
    return kept if kept else fonts


def _platform_font_dirs() -> list[str]:
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        os.path.join(home, "Library", "Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/usr/share/fonts",
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/local/share/fonts",
        os.path.join(home, ".fonts"),
        os.path.join(home, ".local", "share", "fonts"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def _reference_fonts(options: Optional[dict] = None) -> dict[str, list[str]]:
    """Resolve up to two reference font files per class on this machine."""
    options = options or {}
    override = options.get("reference_fonts")
    if isinstance(override, dict):  # explicit override, even empty (disables voting)
        resolved = {}
        for cls, paths in override.items():
            if isinstance(paths, str):
                paths = [paths]
            usable = [path for path in (paths or []) if path and os.path.isfile(path)]
            if usable:
                resolved[str(cls)] = usable[:2]
        return resolved

    dirs = tuple(_platform_font_dirs())
    if dirs in _REFERENCE_FONT_CACHE:
        return {cls: list(paths) for cls, paths in _REFERENCE_FONT_CACHE[dirs].items()}
    resolved = {}
    for cls, names in _REFERENCE_CANDIDATES.items():
        found = []
        for name in names:
            if os.path.isabs(name) and os.path.isfile(name):
                found.append(name)
                continue
            for root in dirs:
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    found.append(path)
                    break
            if len(found) >= 2:
                break
        if found:
            resolved[cls] = found
    _REFERENCE_FONT_CACHE[dirs] = {cls: list(paths) for cls, paths in resolved.items()}
    return resolved


def classify_source(text: str, source_mask, estimated_size: float,
                    options: Optional[dict] = None) -> dict:
    """Class the source ink mask by fitting canonical reference fonts.

    Renders the *same recognized text* in a few known sans/serif/script faces,
    fits each to the mask, and votes: the class whose best reference fits best
    wins.  Same-text comparison makes this robust where absolute stroke-feature
    classifiers are not (short caps runs, digits, bowls vs serifs).  Returns
    ``{"class", "confidence", "scores", "text_confidence"}``; ``class`` is the
    sans-vs-serif call (None when its margin is thin), while ``text_confidence``
    separately reports how confidently the source is plain text rather than a
    script/decorative face — that separation stays reliable even when
    sans-vs-serif does not, and the caller uses it to exclude script candidates.
    """
    options = options or {}
    out: dict[str, Any] = {"class": None, "confidence": 0.0, "scores": {}, "text_confidence": 0.0}
    text = _normalized_text(text)
    if len(text) < 2:
        return out
    tight = _tight_mask(source_mask)
    if tight is None:
        return out
    references = _reference_fonts(options)
    if len(references) < 2:
        return out

    fit_opts = {"iterations": 2}
    scores: dict[str, float] = {}
    for cls, paths in references.items():
        best = 0.0
        for path in paths[:2]:
            fit = fit_line(text, path, tight, estimated_size, fit_opts)
            if fit is not None:
                best = max(best, _num(fit.get("score")))
        scores[cls] = round(best, 4)
    out["scores"] = scores
    # Text-vs-script separation: a plain-text source fits sans/serif references
    # far better than script/decorative even when sans-vs-serif is a toss-up, so
    # this margin is the reliable signal for "never a swash face for plain text".
    text_score = max(scores.get(SANS, 0.0), scores.get(SERIF, 0.0))
    script_score = max(scores.get(SCRIPT, 0.0), scores.get(DECORATIVE, 0.0))
    min_signal = _num(options.get("class_min_signal"), DEFAULT_TEXT_MIN_SIGNAL)
    span = max(0.01, _num(options.get("text_margin_span"), DEFAULT_TEXT_MARGIN_SPAN))
    if text_score >= min_signal and text_score > script_score:
        out["text_confidence"] = round(max(0.0, min(1.0, (text_score - script_score) / span)), 4)
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if not ranked or ranked[0][1] < 0.22:
        return out
    best_cls, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, (best_score - second_score) / 0.10))
    out["class"] = best_cls
    out["confidence"] = round(confidence, 4)
    return out


__all__ = [
    "SANS", "SERIF", "SCRIPT", "DECORATIVE", "TEXT",
    "fit_options", "fit_line", "refine_candidates",
    "classify_font_file", "classify_source", "compatible", "filter_fonts_by_class",
]
