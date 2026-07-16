"""ig_text.py — Instagram/social-UI text & chat constructs (emission + light detect).

Covers the "text carries its own chrome" and "chat/social screenshot" families the
pipeline must reconstruct natively rather than slice:

  IG story / AMA (H4, H12):
    * build_ig_story_text  — per-line rounded background bars hugging each line's ink
    * build_ama_widget     — dark rounded-top header bar + attached white question card
    * build_answer_card    — white rounded answer card (+ optional 👇 emoji chip)

  Chat / social UI (Batch 2 — H9 DM, H14 tweet, H16 X-on-black):
    * build_dm_bubble          — asymmetric-radius bubble, flat OR gradient fill, emoji chips
    * build_new_messages_divider — line—text—line
    * build_reply_quote        — vertical bar + quoted mini-bubble + "Replied to you"
    * build_tweet              — circular avatar + name/handle/timestamp + body + engagement row
    * build_engagement_row     — glyph chips + NATIVE count text (like carries a liked/red state)
    * build_thread_line        — thin vertical rail connecting avatars
    * redaction_chip           — scribbled-out username kept as an IMAGE chip (never OCR)

  Detectors (classic CV, advisory):
    * detect_ig_story_text     — stacked text lines each on their own uniform bar
    * detect_screenshot_card   — the rounded screenshot card, separating from a white OR
                                 black page background (H9 dark-on-white, H16 photo-on-black)

CONTRACT: plates/bubbles = native SOLID/GRADIENT rounded rects; every string = native
TEXT (letterSpacing 0); emoji/glyphs/redactions = image chips; avatars/insets = ellipse-
masked images. All coordinates ABSOLUTE; layout._relativize rewrites to parent space.

Emitted candidates compile directly through build_design_json.build().
"""
from __future__ import annotations

from typing import Any, Optional

try:  # shared geometry helpers; both live in src/
    from . import overlay_detect as _ov
except Exception:  # pragma: no cover - flat import fallback (tests add repo root)
    import overlay_detect as _ov  # type: ignore


def _deps():
    return _ov._deps()


# ── paint / candidate helpers ────────────────────────────────────────────────────────
def flat_fill(color: str) -> dict:
    return {"kind": "flat", "color": color}


def gradient_fill(colors, angle: float = 90.0, kind: str = "linear") -> dict:
    """Native gradient paint. ``colors`` is an ordered list of hex strings (>=2).

    Positions are spread evenly 0..1. angle 0 == left-to-right (compiler convention).
    Used for DM outgoing bubbles (purple->blue) and gradient washes.
    """
    cols = list(colors)
    if len(cols) == 1:
        cols = [cols[0], cols[0]]
    n = len(cols)
    stops = [{"color": c, "position": round(i / (n - 1), 4)} for i, c in enumerate(cols)]
    return {"kind": kind, "stops": stops, "angle": float(angle)}


def _rect(cid, box, fill, radius=None, z=20.0, role="plate", stroke=None, meta=None):
    out = {
        "id": cid, "target": "shape", "shape_kind": "rect",
        "box": {k: float(box[k]) for k in ("x", "y", "w", "h")},
        "fill": fill, "z_index": float(z),
        "meta": {"role": role, "z": float(z), "source": "social-ui", **(meta or {})},
    }
    if radius is not None:
        out["radius"] = radius
    if stroke is not None:
        out["stroke"] = stroke
    return out


def _text(cid, text, box, style=None, z=22.0, role="text", color="#000000",
          align="LEFT", meta=None):
    st = {"color": color, "align": align, "letterSpacing": 0.0}
    st.update(style or {})
    st["letterSpacing"] = 0.0
    return {
        "id": cid, "target": "text", "text": str(text),
        "box": {k: float(box[k]) for k in ("x", "y", "w", "h")},
        "style": st, "z_index": float(z),
        "meta": {"role": role, "z": float(z), "source": "social-ui", **(meta or {})},
    }


def _chip(cid, box, src=None, z=24.0, role="chip", meta=None):
    """A raster IMAGE chip (emoji / engagement glyph / redaction blob)."""
    return {
        "id": cid, "target": "image", "src": src,
        "box": {k: float(box[k]) for k in ("x", "y", "w", "h")},
        "z_index": float(z),
        "meta": {"role": role, "z": float(z), "emoji_chip": role == "emoji",
                 "source": "social-ui", **(meta or {})},
    }


def ellipse_image(cid, box, src=None, z=21.0, role="avatar", meta=None):
    """Circular photo inset / avatar = ellipse mask + image fill.

    Sets both the explicit mask spec AND the meta hints reconstruct._image_mask_spec
    keys on, so the ellipse clip triggers reliably (spec capability item 5).
    """
    return {
        "id": cid, "target": "image", "src": src,
        "box": {k: float(box[k]) for k in ("x", "y", "w", "h")},
        "z_index": float(z),
        "mask": {"kind": "ellipse"},
        "meta": {"role": role, "circular_inset": True, "circular": True,
                 "z": float(z), "source": "social-ui", **(meta or {})},
    }


def redaction_chip(cid, box, src=None, z=24.0, meta=None):
    """A scribbled-out / redacted username kept as a chip — must NEVER be OCR'd (H16)."""
    m = {"no_ocr": True, "redaction": True}
    m.update(meta or {})
    return _chip(cid, box, src=src, z=z, role="redaction", meta=m)


def _group(cid, box, children, role, z=20.0, meta=None):
    return {
        "id": cid, "target": "group",
        "box": {k: float(box[k]) for k in ("x", "y", "w", "h")},
        "z_index": float(z),
        "children": children,
        "meta": {"role": role, "z": float(z), "source": "social-ui", **(meta or {})},
    }


def _line_height(style, box):
    if style and style.get("lineHeight"):
        return float(style["lineHeight"])
    if style and style.get("fontSize"):
        return float(style["fontSize"]) * 1.2
    return float(box["h"])


# ── IG story per-line background text ────────────────────────────────────────────────
def build_ig_story_text(lines, fill="#000000", text_color="#ffffff", radius=None,
                        pad_x=None, pad_y=None, overlap=None, group_id="igtext",
                        z=20.0):
    """Per-line rounded background bars (classic IG story text).

    ``lines`` — ordered list of {text, ink_box:{x,y,w,h}, style?}. Each wrapped line gets
    its OWN rounded background bar hugging that line's ink width + padding; bars are
    stacked and joined with a consistent vertical overlap so they read as one ribbon.

    Geometry (per spec): radius ~= ¼ line height; plate hugs ink + ~0.4em horizontal
    padding; adjacent plates overlap by ``overlap`` px (default ~0.12·lh) at a constant
    join. Returns a group candidate: per line a [rect, text] pair.
    """
    children = []
    xs, ys, xe, ye = [], [], [], []
    for i, ln in enumerate(lines):
        ink = ln.get("ink_box") or ln.get("box")
        style = ln.get("style") or {}
        lh = _line_height(style, ink)
        r = radius if radius is not None else round(lh * 0.25, 2)
        px = pad_x if pad_x is not None else round(lh * 0.40, 2)
        py = pad_y if pad_y is not None else round(lh * 0.18, 2)
        ov = overlap if overlap is not None else round(lh * 0.12, 2)
        plate = {
            "x": ink["x"] - px,
            "y": ink["y"] - py - (ov / 2 if i > 0 else 0),
            "w": ink["w"] + 2 * px,
            "h": ink["h"] + 2 * py + (ov / 2 if i > 0 else 0) +
                 (ov / 2 if i < len(lines) - 1 else 0),
        }
        children.append(_rect(f"{group_id}__bar{i}", plate, flat_fill(fill),
                              radius=r, z=z, role="ig-line-plate",
                              meta={"line_index": i}))
        children.append(_text(f"{group_id}__t{i}", ln["text"], ink, style=style,
                              z=z + 2, role="ig-line-text", color=text_color,
                              align=style.get("align", "CENTER"),
                              meta={"line_index": i}))
        xs.append(plate["x"]); ys.append(plate["y"])
        xe.append(plate["x"] + plate["w"]); ye.append(plate["y"] + plate["h"])
    box = {"x": min(xs), "y": min(ys), "w": max(xe) - min(xs), "h": max(ye) - min(ys)}
    return _group(group_id, box, children, "ig-story-text", z=z,
                  meta={"per_line_plates": len(lines)})


# ── AMA widget + answer cards ─────────────────────────────────────────────────────────
def build_ama_widget(header_text, question_text, box, header_fill="#1c1c1e",
                     card_fill="#ffffff", header_color="#ffffff",
                     question_color="#1c1c1e", radius=None, header_frac=0.42,
                     group_id="ama", z=20.0, header_style=None, question_style=None):
    """IG 'Ask me anything' widget: dark rounded-top header bar + attached white card.

    The header's TOP corners are rounded and the card's BOTTOM corners are rounded; they
    meet flush so the pair reads as a single sticker (H4/H12).
    """
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    r = radius if radius is not None else round(min(w, h) * 0.10, 2)
    hh = round(h * header_frac, 2)
    header_box = {"x": x, "y": y, "w": w, "h": hh}
    card_box = {"x": x, "y": y + hh, "w": w, "h": h - hh}
    children = [
        _rect(f"{group_id}__header", header_box, flat_fill(header_fill),
              radius={"topLeft": r, "topRight": r, "bottomLeft": 0, "bottomRight": 0},
              z=z, role="ama-header"),
        _text(f"{group_id}__header_t", header_text, header_box,
              style=header_style, z=z + 2, role="ama-header-text",
              color=header_color, align="CENTER"),
        _rect(f"{group_id}__card", card_box, flat_fill(card_fill),
              radius={"topLeft": 0, "topRight": 0, "bottomLeft": r, "bottomRight": r},
              z=z + 1, role="ama-card"),
        _text(f"{group_id}__card_t", question_text, card_box,
              style=question_style, z=z + 3, role="ama-question",
              color=question_color, align="CENTER"),
    ]
    return _group(group_id, box, children, "ama-widget", z=z)


def build_answer_card(text, box, fill="#ffffff", text_color="#1c1c1e", radius=None,
                      emoji=None, emoji_src=None, group_id="answer", z=20.0, style=None):
    """A white rounded answer card + native text, with an optional trailing emoji chip
    (H12 👇). ``emoji``/``emoji_src`` add an inline image chip at the card's end."""
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    r = radius if radius is not None else round(min(w, h) * 0.14, 2)
    children = [
        _rect(f"{group_id}__card", box, flat_fill(fill), radius=r, z=z, role="answer-card"),
        _text(f"{group_id}__t", text, box, style=style, z=z + 2,
              role="answer-text", color=text_color, align="LEFT"),
    ]
    if emoji or emoji_src:
        es = round(h * 0.34, 2)
        ebox = {"x": x + w - es - r * 0.5, "y": y + h - es - r * 0.5, "w": es, "h": es}
        children.append(_chip(f"{group_id}__emoji", ebox, src=emoji_src, z=z + 3,
                              role="emoji", meta={"char": emoji}))
    return _group(group_id, box, children, "answer-card", z=z)


# ── DM chat bubbles ──────────────────────────────────────────────────────────────────
# Standard messenger corner geometry: the corner nearest the sender's side is nipped
# small, the three others are the full bubble radius.
def _bubble_radius(r_full, r_tail, incoming):
    if incoming:  # tail bottom-left
        return {"topLeft": r_full, "topRight": r_full,
                "bottomLeft": r_tail, "bottomRight": r_full}
    return {"topLeft": r_full, "topRight": r_full,   # tail bottom-right
            "bottomLeft": r_full, "bottomRight": r_tail}


def build_dm_bubble(text, box, incoming=True, fill=None, gradient=None,
                    text_color=None, radius=None, tail_radius=None, emojis=None,
                    group_id="bubble", z=20.0, style=None):
    """A chat bubble with ASYMMETRIC corner radii and flat OR gradient fill.

    incoming grey bubble -> flat grey, dark text, tail bottom-left.
    outgoing bubble      -> gradient (e.g. purple->blue), white text, tail bottom-right.
    ``gradient`` (list of hex) wins over ``fill`` (hex). ``emojis`` = list of
    {box, src, char} inline image chips.
    """
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    r_full = radius if radius is not None else round(min(h * 0.5, 22.0), 2)
    r_tail = tail_radius if tail_radius is not None else round(r_full * 0.25, 2)
    rad = _bubble_radius(r_full, r_tail, incoming)
    if gradient:
        paint = gradient_fill(gradient, angle=45.0)
        tcolor = text_color or "#ffffff"
    else:
        paint = flat_fill(fill or ("#e9e9eb" if incoming else "#3797f0"))
        tcolor = text_color or ("#1c1c1e" if incoming else "#ffffff")
    children = [
        _rect(f"{group_id}__b", box, paint, radius=rad, z=z,
              role="dm-bubble", meta={"incoming": bool(incoming)}),
        _text(f"{group_id}__t", text, box, style=style, z=z + 2,
              role="dm-text", color=tcolor, align="LEFT"),
    ]
    for i, em in enumerate(emojis or []):
        children.append(_chip(f"{group_id}__emoji{i}", em["box"], src=em.get("src"),
                              z=z + 3, role="emoji", meta={"char": em.get("char")}))
    return _group(group_id, box, children, "dm-bubble-group", z=z,
                  meta={"incoming": bool(incoming), "gradient": bool(gradient)})


def build_new_messages_divider(box, text="New Messages", line_color="#c8c8cc",
                               text_color="#8e8e93", thickness=None, gap=None,
                               group_id="newmsg", z=20.0, style=None):
    """line—text—line divider. Two thin rects flank a centered caption."""
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    t = thickness if thickness is not None else max(1.0, round(h * 0.06, 2))
    cy = y + h / 2 - t / 2
    tw = max(40.0, min(w * 0.4, 8.0 * len(text)))
    g = gap if gap is not None else round(w * 0.03, 2)
    left_w = (w - tw) / 2 - g
    right_x = x + (w + tw) / 2 + g
    children = [
        _rect(f"{group_id}__lineL", {"x": x, "y": cy, "w": max(1.0, left_w), "h": t},
              flat_fill(line_color), radius=t / 2, z=z, role="divider-line"),
        _text(f"{group_id}__t", text, {"x": x + (w - tw) / 2, "y": y, "w": tw, "h": h},
              style=style, z=z + 2, role="divider-text", color=text_color, align="CENTER"),
        _rect(f"{group_id}__lineR",
              {"x": right_x, "y": cy, "w": max(1.0, x + w - right_x), "h": t},
              flat_fill(line_color), radius=t / 2, z=z, role="divider-line"),
    ]
    return _group(group_id, box, children, "new-messages-divider", z=z)


def build_reply_quote(caption, quote_text, box, bar_color="#8e8e93",
                      quote_fill="#f0f0f2", caption_color="#8e8e93",
                      quote_color="#3c3c43", bar_w=None, caption_frac=0.32,
                      group_id="reply", z=20.0, caption_style=None, quote_style=None):
    """Reply-quote construct: 'Replied to you' caption + a vertical bar + a quoted
    mini-bubble holding the quoted text (H9)."""
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    cap_h = round(h * caption_frac, 2)
    bw = bar_w if bar_w is not None else max(2.0, round(w * 0.012, 2))
    quote_y = y + cap_h
    quote_h = h - cap_h
    children = [
        _text(f"{group_id}__cap", caption, {"x": x, "y": y, "w": w, "h": cap_h},
              style=caption_style, z=z + 2, role="reply-caption",
              color=caption_color, align="LEFT"),
        _rect(f"{group_id}__bar", {"x": x, "y": quote_y, "w": bw, "h": quote_h},
              flat_fill(bar_color), radius=bw / 2, z=z + 1, role="reply-bar"),
        _rect(f"{group_id}__quote",
              {"x": x + bw + 6, "y": quote_y, "w": w - bw - 6, "h": quote_h},
              flat_fill(quote_fill), radius=round(quote_h * 0.28, 2), z=z,
              role="reply-quote-bubble"),
        _text(f"{group_id}__qt", quote_text,
              {"x": x + bw + 6, "y": quote_y, "w": w - bw - 6, "h": quote_h},
              style=quote_style, z=z + 3, role="reply-quote-text",
              color=quote_color, align="LEFT"),
    ]
    return _group(group_id, box, children, "reply-quote", z=z)


# ── Tweet / X thread row ──────────────────────────────────────────────────────────────
_ENGAGE_ORDER = ("reply", "retweet", "like", "views", "bookmark", "share")


def build_engagement_row(box, items, glyph_frac=0.6, group_id="engage", z=22.0,
                         count_color="#536471", liked_color="#f91880", style=None):
    """A row of engagement glyph chips + NATIVE count text (H14).

    ``items`` — ordered list of {kind, count, src?, liked?}. ``kind`` in reply/retweet/
    like/views/bookmark/share. The like glyph carries a liked/red state via meta + a red
    count colour. Counts are ALWAYS native TEXT, never baked into the glyph.
    """
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    n = max(1, len(items))
    slot = w / n
    gs = round(h * glyph_frac, 2)
    children = []
    for i, it in enumerate(items):
        sx = x + i * slot
        cy = y + (h - gs) / 2
        liked = bool(it.get("liked"))
        children.append(_chip(f"{group_id}__g{i}", {"x": sx, "y": cy, "w": gs, "h": gs},
                              src=it.get("src"), z=z, role="engagement-glyph",
                              meta={"kind": it.get("kind"), "liked": liked}))
        count = it.get("count")
        if count not in (None, ""):
            children.append(_text(
                f"{group_id}__c{i}",
                str(count),
                {"x": sx + gs + 4, "y": y, "w": max(1.0, slot - gs - 6), "h": h},
                style=style, z=z, role="engagement-count",
                color=liked_color if liked else count_color, align="LEFT",
                meta={"kind": it.get("kind"), "liked": liked}))
    return _group(group_id, box, children, "engagement-row", z=z)


def build_thread_line(box, color="#cfd9de", group_id="thread", z=18.0):
    """Thin vertical rail connecting stacked avatars in a thread (H14)."""
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    return _rect(f"{group_id}__line", box, flat_fill(color),
                 radius=round(w / 2, 2), z=z, role="thread-line")


def build_tweet(box, name, handle, body, avatar_src=None, timestamp=None,
                engagement=None, avatar_frac=0.14, group_id="tweet", z=20.0,
                name_color="#0f1419", meta_color="#536471", body_color="#0f1419",
                name_style=None, body_style=None):
    """A single tweet row: circular avatar + (bold name · grey handle · timestamp) +
    body TEXT + engagement glyph row (H14). Avatar is an ellipse-masked image."""
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    av = round(w * avatar_frac, 2)
    pad = round(av * 0.5, 2)
    content_x = x + av + pad
    content_w = x + w - content_x
    row_h = round(av * 0.5, 2)
    children = [
        ellipse_image(f"{group_id}__avatar", {"x": x, "y": y, "w": av, "h": av},
                      src=avatar_src, z=z + 1, role="avatar"),
        _text(f"{group_id}__name", name,
              {"x": content_x, "y": y, "w": content_w, "h": row_h},
              style=(name_style or {"fontWeight": 700}), z=z + 2, role="tweet-name",
              color=name_color, align="LEFT"),
    ]
    meta_line = " ".join(s for s in [handle, timestamp] if s)
    if meta_line:
        children.append(_text(
            f"{group_id}__meta", meta_line,
            {"x": content_x, "y": y + row_h, "w": content_w, "h": row_h},
            z=z + 2, role="tweet-handle", color=meta_color, align="LEFT"))
    body_y = y + 2 * row_h
    body_h = max(row_h, (y + h) - body_y - (row_h if engagement else 0))
    children.append(_text(
        f"{group_id}__body", body,
        {"x": content_x, "y": body_y, "w": content_w, "h": body_h},
        style=body_style, z=z + 2, role="tweet-body", color=body_color, align="LEFT"))
    if engagement:
        erow = {"x": content_x, "y": y + h - row_h, "w": content_w, "h": row_h}
        items = engagement if isinstance(engagement[0], dict) else [
            {"kind": k} for k in engagement]
        children.append(build_engagement_row(erow, items, group_id=f"{group_id}__eng",
                                             z=z + 2))
    return _group(group_id, box, children, "tweet", z=z)


# ══════════════════════════════════════════════════════════════════════════════════════
# Detectors (classic CV, advisory) — prove the constructs on synthetic fixtures.
# ══════════════════════════════════════════════════════════════════════════════════════
def detect_screenshot_card(rgb, canvas=None, cfg=None):
    """Find THE rounded screenshot card and its page background (H9 / H16).

    Works whether the page bg is white (dark card) OR black (photo/white card): the card
    is the largest connected region that differs from the dominant border colour and fills
    a near-rectangular rounded silhouette. Returns None or:
        {bbox, corner_radius, page_bg:"#rrggbb", card_fill:"#rrggbb"|None,
         fill_ratio, separates_clean: bool, source:"screenshot-card-cv"}
    ``separates_clean`` is True when the card's silhouette is a clean rounded rect (the
    user-explicit requirement that the card lifts off the page bg without fringing).
    """
    np, cv2 = _deps()
    if cv2 is None:
        return None
    arr = _ov._as_rgb_u8(rgb)
    H, W = arr.shape[:2]
    if canvas is None:
        canvas = {"w": W, "h": H}
    tol = float((cfg or {}).get("card_bg_tol", 26))

    border = np.concatenate([
        arr[0, :, :].reshape(-1, 3), arr[-1, :, :].reshape(-1, 3),
        arr[:, 0, :].reshape(-1, 3), arr[:, -1, :].reshape(-1, 3),
    ])
    bcolors, bcounts = np.unique(border, axis=0, return_counts=True)
    bg = bcolors[int(bcounts.argmax())]
    fg = (np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > tol)
    fg_u8 = (fg.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    best = None
    for comp in range(1, num):
        area = int(stats[comp, cv2.CC_STAT_AREA])
        if area < 0.06 * H * W:
            continue
        x = int(stats[comp, cv2.CC_STAT_LEFT]); y = int(stats[comp, cv2.CC_STAT_TOP])
        w = int(stats[comp, cv2.CC_STAT_WIDTH]); h = int(stats[comp, cv2.CC_STAT_HEIGHT])
        if w >= 0.985 * W and h >= 0.985 * H:
            continue  # that is the whole page, not a card
        comp_mask = labels[y:y + h, x:x + w] == comp
        fill_ratio = float(comp_mask.mean())
        if best is None or area > best[0]:
            best = (area, {"x": x, "y": y, "w": w, "h": h}, comp_mask, fill_ratio)
    if best is None:
        return None
    _, bbox, comp_mask, fill_ratio = best
    radius = _ov.estimate_corner_radius(comp_mask, (cfg or {}).get("overlay_detect"))
    # card fill only when the interior is (near) uniform — a photo card has no flat fill
    interior = arr[bbox["y"]:bbox["y"] + bbox["h"], bbox["x"]:bbox["x"] + bbox["w"]][comp_mask]
    median = np.median(interior, axis=0)
    uniform = float((np.abs(interior.astype(np.int16) - median.astype(np.int16)
                           ).max(axis=1) <= tol).mean())
    return {
        "bbox": bbox,
        "corner_radius": radius,
        "page_bg": _ov._hex(bg),
        "card_fill": _ov._hex(median) if uniform >= 0.8 else None,
        "fill_ratio": round(fill_ratio, 3),
        "separates_clean": fill_ratio >= 0.9 and radius is not None,
        "source": "screenshot-card-cv",
    }


def detect_ig_story_text(rgb, text_lines, cfg=None):
    """Group stacked text lines that each sit on their OWN uniform background bar.

    For each OCR line, sample the ring just outside the ink box: if it is a uniform colour
    distinct from the page, the line carries a per-line plate. Consecutive plated lines
    sharing a colour are grouped into one IG-story-text run. Returns a list of runs:
        {line_ids:[...], fill:"#rrggbb", boxes:[...]}
    """
    np, cv2 = _deps()
    if cv2 is None:
        return []
    arr = _ov._as_rgb_u8(rgb)
    H, W = arr.shape[:2]
    tol = float((cfg or {}).get("plate_tol", 22))
    plated = []
    for ln in text_lines or []:
        b = ln.get("ink_box") or ln.get("box")
        if not b:
            plated.append(None); continue
        x0 = int(max(0, b["x"] - b["h"] * 0.4)); y0 = int(max(0, b["y"] - b["h"] * 0.25))
        x1 = int(min(W, b["x"] + b["w"] + b["h"] * 0.4))
        y1 = int(min(H, b["y"] + b["h"] + b["h"] * 0.25))
        if x1 <= x0 or y1 <= y0:
            plated.append(None); continue
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        # ring = patch minus the central ink band (sample plate colour, not glyph ink)
        med = np.median(patch, axis=0)
        uniform = float((np.abs(patch.astype(np.int16) - med.astype(np.int16)
                               ).max(axis=1) <= tol).mean())
        plated.append((str(ln.get("id") or ""), _ov._hex(med), uniform, dict(b))
                      if uniform >= 0.5 else None)
    runs, cur = [], None
    for item in plated:
        if item is None:
            if cur:
                runs.append(cur); cur = None
            continue
        lid, fill, _u, box = item
        if cur and cur["fill"] == fill:
            cur["line_ids"].append(lid); cur["boxes"].append(box)
        else:
            if cur:
                runs.append(cur)
            cur = {"line_ids": [lid], "fill": fill, "boxes": [box]}
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r["line_ids"]) >= 1]
