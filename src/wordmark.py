"""wordmark.py — identify brand lettering before it reaches font-matching.

Ported from the Mac harness lib/wordmarks.mjs (owner-authored). A wordmark is artwork even
when OCR can read its letters: re-setting it in a nearby font changes the brand, and OCR
commonly turns an adjacent heart/star device into a bullet. Conservative on purpose — only
short, header/footer-positioned brand candidates are claimed; everything else stays editable.
"""
from __future__ import annotations
import re

GENERIC_SHORT_COPY = re.compile(
    r"^(buy now|shop now|learn more|swipe up|tap here|new|sale|save|free shipping|"
    r"limited time|order now|subscribe)$", re.I)
_UI_LABEL = re.compile(
    r"^(post|following|follow|followed|volgend|back|next|previous|menu|close|cancel|"
    r"done|share|reply|comment|like|save|bookmark|home|search|profile|settings)$",
    re.I,
)
_SOCIAL_HANDLE = re.compile(r"^@[A-Za-z0-9_.-]+$")
_PICTOGRAM = re.compile(r"[♡♥❤★☆✦✧]|^[•●◦·]\s*")
_ALLCAPS_SHORT = re.compile(r"^[A-ZÀ-ÖØ-Þ& ]+$")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_PLATFORM_WORDMARK = re.compile(r"^(?:x\.com|twitter\.com)$", re.I)
_CTA = re.compile(r"^(?:buy|shop|learn|order|get|try|sign up|subscribe|download)(?:\s+\w+){0,3}$", re.I)
# Sale / offer chrome ("45%", "Off", "Get up to") — overlay copy on badges, not packaging.
_OFFER = re.compile(
    r"^(?:"
    r"\d+\s*%|"
    r"(?:up\s+to|get\s+up\s+to|upto)\b.*|"
    r"off|"
    r"(?:save|upto)\s+\d+\s*%|"
    r"\d+\s*%\s*off"
    r")$",
    re.I,
)


def _clean(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip()


def is_platform_lockup(line: dict, canvas: dict) -> bool:
    """True for a platform lockup that must remain a separate artwork asset."""
    text = _clean(line.get("text"))
    box = line.get("box") or {}
    H = float(canvas.get("h") or 1)
    try:
        return bool(_PLATFORM_WORDMARK.match(text) and
                    float(box.get("y", 0)) + float(box.get("h", 0)) <= H * 0.32)
    except (TypeError, ValueError):
        return False


def semantic_text_role(line: dict, canvas: dict) -> str:
    """Give editable text a stable human-facing role for Figma naming/grouping.

    Ownership remains VLM-decided. This is deliberately deterministic so reruns
    do not need more VLM calls just to replace opaque ``Text — ...`` names.
    """
    text = _clean(line.get("text"))
    box = line.get("box") or {}
    W = float(canvas.get("w") or 1)
    H = float(canvas.get("h") or 1)
    try:
        y, w, h = float(box.get("y", 0)), float(box.get("w", 0)), float(box.get("h", 0))
    except (TypeError, ValueError):
        return "text"
    if is_platform_lockup(line, canvas):
        return "platform-logo"
    if _SOCIAL_HANDLE.match(text) or _UI_LABEL.match(text):
        return "ui-label"
    if _CTA.match(text) or GENERIC_SHORT_COPY.match(text):
        return "cta"
    if _OFFER.match(text) or ("%" in text and len(text) <= 12):
        return "offer"
    words = text.split()
    # Primary display headline: large type in the upper band — not mid-canvas
    # benefit callouts (014) which are shorter and sit beside a product.
    if (
        y <= H * .38
        and h >= max(28.0, H * .035)
        and len(words) <= 12
        and w >= W * .40
        and w <= W * .95
    ):
        return "headline"
    if len(words) >= 9 or (h <= H * .03 and len(words) >= 4):
        return "body-copy"
    return "label"


def is_wordmark_candidate(line: dict, canvas: dict, opts: dict | None = None) -> bool:
    opts = opts or {}
    text = _clean(line.get("text"))
    box = line.get("box") or {}
    W = float(canvas.get("w") or 1); H = float(canvas.get("h") or 1)
    x, y, w, h = box.get("x"), box.get("y"), box.get("w"), box.get("h")
    if not text or None in (x, y, w, h) or not (w > 3) or not (h > 3):
        return False
    force_text = opts.get("force_text_ids") or set()
    force_wm = opts.get("force_wordmark_ids") or set()
    if line.get("id") in force_text:
        return False
    if line.get("id") in force_wm:
        return True
    # Platform lockups combine a custom logo glyph with domain text. They are artwork,
    # including in the conventional top-right slot, and must not be approximated by a font.
    if is_platform_lockup(line, canvas):
        return True
    # Short social/UI labels occupy exactly the same header slots as brand marks.  The
    # old positional heuristic consequently rasterized ordinary editable copy such as
    # ``Post`` and ``@UpfrontFood`` in ad9.  Handles and known interface labels are text;
    # an explicit upstream ``force_wordmark_ids`` decision can still override this.
    if (len(text) > opts.get("max_chars", 28) or re.search(r"[\n.!?]", text)
            or re.search(r"\d", text) or GENERIC_SHORT_COPY.match(text)
            or _UI_LABEL.match(text) or _SOCIAL_HANDLE.match(text)):
        return False

    center_x = x + w / 2
    header = y + h <= H * 0.24
    footer = y >= H * 0.88 and len(text) <= 16
    centered = abs(center_x - W / 2) <= W * 0.22
    left_brand_slot = x <= W * 0.30
    if not (header or footer) or not (centered or left_brand_slot):
        return False

    words = [w_ for w_ in text.split(" ") if w_]
    letters = len(_LETTER.findall(text))
    pictogram = bool(_PICTOGRAM.search(text))
    compact = w / max(1, h) >= 1.15
    name_like = len(words) <= 3 and letters >= 3
    allcaps_short = bool(_ALLCAPS_SHORT.match(text)) and letters <= 12
    return compact and name_like and (pictogram or len(words) == 1 or (allcaps_short and len(words) == 1))


def partition_wordmarks(lines: list, canvas: dict, opts: dict | None = None) -> dict:
    """Split OCR lines into regular text and logo-artwork candidates."""
    wordmarks, text = [], []
    for line in (lines or []):
        (wordmarks if is_wordmark_candidate(line, canvas, opts) else text).append(line)
    return {"wordmarks": wordmarks, "text": text}
