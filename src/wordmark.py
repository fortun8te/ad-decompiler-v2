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


def _clean(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip()


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
