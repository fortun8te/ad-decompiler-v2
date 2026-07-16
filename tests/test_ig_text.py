"""test_ig_text.py — IG story text, AMA widget, chat/social-UI constructs, and the
ellipse-masked circular inset trigger (CPU-only).

Every construct is compiled through the REAL build_design_json.build() and checked with
schema.validate_design so the emission contract is enforced end-to-end, not just asserted
on a hand-built dict.
"""
import os
import sys
import tempfile
from dataclasses import asdict

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import ig_text, build_design_json, schema  # noqa: E402


def _compile(candidates, w=1080, h=1080):
    run_dir = tempfile.mkdtemp(prefix="igtest_")
    doc = build_design_json.build(list(candidates), {"w": w, "h": h}, run_dir, base_src=None)
    d = asdict(doc)
    assert schema.validate_design(d) == [], schema.validate_design(d)
    return d


def _find(node, pred):
    stack = list(node.get("layers", node.get("children", [])))
    out = []
    while stack:
        n = stack.pop()
        if pred(n):
            out.append(n)
        stack.extend(n.get("children", []) or [])
    return out


# ── IG story per-line background text ────────────────────────────────────────────────
def test_ig_story_text_per_line_plates():
    lines = [
        {"text": "everyday", "ink_box": {"x": 100, "y": 200, "w": 220, "h": 60},
         "style": {"fontSize": 48, "lineHeight": 60}},
        {"text": "curl crème", "ink_box": {"x": 100, "y": 268, "w": 300, "h": 60},
         "style": {"fontSize": 48, "lineHeight": 60}},
    ]
    grp = ig_text.build_ig_story_text(lines, fill="#111111", text_color="#ffffff")
    d = _compile([grp])
    plates = _find(d, lambda n: (n.get("meta") or {}).get("role") == "ig-line-plate")
    texts = _find(d, lambda n: (n.get("meta") or {}).get("role") == "ig-line-text")
    assert len(plates) == 2 and len(texts) == 2
    # radius ~ 1/4 line height (60 -> 15)
    for p in plates:
        r = p["radius"]
        assert 12 <= float(r) <= 18, r
    # plate hugs ink + padding (wider than the ink box)
    assert plates[0]["box"]["w"] > 220


# ── AMA widget ────────────────────────────────────────────────────────────────────────
def test_ama_widget_header_and_card_radii():
    grp = ig_text.build_ama_widget(
        "Ask me anything", "How do I keep curls defined?",
        {"x": 300, "y": 300, "w": 480, "h": 260})
    d = _compile([grp])
    header = _find(d, lambda n: (n.get("meta") or {}).get("role") == "ama-header")[0]
    card = _find(d, lambda n: (n.get("meta") or {}).get("role") == "ama-card")[0]
    # header rounds TOP corners only; card rounds BOTTOM corners only (flush join)
    assert header["radius"]["topLeft"] > 0 and header["radius"]["bottomLeft"] == 0
    assert card["radius"]["bottomLeft"] > 0 and card["radius"]["topLeft"] == 0
    assert header["fill"]["kind"] == "flat" and card["fill"]["kind"] == "flat"
    qs = _find(d, lambda n: (n.get("meta") or {}).get("role") == "ama-question")
    assert qs and qs[0]["text"].startswith("How")


# ── DM bubble: asymmetric radius + gradient fill ─────────────────────────────────────
def test_dm_bubble_gradient_and_asymmetric_radius():
    out = ig_text.build_dm_bubble(
        "on its way 💜", {"x": 500, "y": 700, "w": 360, "h": 88},
        incoming=False, gradient=["#833ab4", "#4a5cf0"])
    d = _compile([out])
    bubble = _find(d, lambda n: (n.get("meta") or {}).get("role") == "dm-bubble")[0]
    assert bubble["fill"]["kind"] == "linear"
    assert len(bubble["fill"]["stops"]) == 2
    # outgoing tail = bottom-right nipped small, others full
    rad = bubble["radius"]
    assert rad["bottomRight"] < rad["topLeft"]

    incoming = ig_text.build_dm_bubble(
        "sounds good", {"x": 120, "y": 700, "w": 300, "h": 80}, incoming=True)
    di = _compile([incoming])
    b2 = _find(di, lambda n: (n.get("meta") or {}).get("role") == "dm-bubble")[0]
    assert b2["fill"]["kind"] == "flat"
    assert b2["radius"]["bottomLeft"] < b2["radius"]["topRight"]


# ── New-messages divider (line—text—line) ─────────────────────────────────────────────
def test_new_messages_divider():
    grp = ig_text.build_new_messages_divider({"x": 120, "y": 640, "w": 840, "h": 40})
    d = _compile([grp])
    lines = _find(d, lambda n: (n.get("meta") or {}).get("role") == "divider-line")
    text = _find(d, lambda n: (n.get("meta") or {}).get("role") == "divider-text")
    assert len(lines) == 2 and len(text) == 1
    assert text[0]["text"] == "New Messages"


# ── Reply-quote construct ─────────────────────────────────────────────────────────────
def test_reply_quote_construct():
    grp = ig_text.build_reply_quote(
        "Replied to you", "your original message",
        {"x": 140, "y": 500, "w": 500, "h": 140})
    d = _compile([grp])
    assert _find(d, lambda n: (n.get("meta") or {}).get("role") == "reply-bar")
    assert _find(d, lambda n: (n.get("meta") or {}).get("role") == "reply-quote-bubble")
    caps = _find(d, lambda n: (n.get("meta") or {}).get("role") == "reply-caption")
    assert caps and caps[0]["text"] == "Replied to you"


# ── Tweet row: avatar ellipse + native counts ────────────────────────────────────────
def test_tweet_engagement_counts_native_and_avatar_ellipse():
    grp = ig_text.build_tweet(
        {"x": 80, "y": 200, "w": 900, "h": 320},
        name="Wavy", handle="@wavycurls", timestamp="· 2h",
        body="one product, defined curls all day",
        engagement=[
            {"kind": "reply", "count": "12"},
            {"kind": "retweet", "count": "5"},
            {"kind": "like", "count": "203", "liked": True},
            {"kind": "views", "count": "1.2K"},
        ])
    d = _compile([grp])
    avatar = _find(d, lambda n: (n.get("meta") or {}).get("role") == "avatar")[0]
    assert avatar["type"] == "image"
    assert (avatar.get("mask") or {}).get("kind") == "ellipse"
    counts = _find(d, lambda n: (n.get("meta") or {}).get("role") == "engagement-count")
    assert {c["text"] for c in counts} >= {"12", "5", "203", "1.2K"}
    glyphs = _find(d, lambda n: (n.get("meta") or {}).get("role") == "engagement-glyph")
    liked = [g for g in glyphs if (g.get("meta") or {}).get("liked")]
    assert len(liked) == 1  # the like glyph carries the red/liked state


def test_redaction_chip_never_ocr():
    chip = ig_text.redaction_chip("red0", {"x": 100, "y": 100, "w": 180, "h": 30})
    assert chip["meta"]["no_ocr"] is True
    d = _compile([chip])
    node = _find(d, lambda n: n.get("id") == "red0")[0]
    assert node["type"] == "image"
    assert node["meta"]["no_ocr"] is True


# ── Detectors ─────────────────────────────────────────────────────────────────────────
def _rounded_card(page_rgb, card_box, card_fill, radius, noise_card=False):
    img = Image.new("RGB", (page_rgb[1], page_rgb[0]), page_rgb[2])
    draw = ImageDraw.Draw(img)
    xy = (card_box["x"], card_box["y"],
          card_box["x"] + card_box["w"] - 1, card_box["y"] + card_box["h"] - 1)
    draw.rounded_rectangle(xy, radius=radius, fill=card_fill)
    arr = np.array(img, dtype=np.uint8)  # writable copy
    if noise_card:  # a photo card: fill the interior with texture
        rng = np.random.default_rng(3)
        tex = rng.integers(40, 220, (card_box["h"], card_box["w"], 3), dtype=np.uint8)
        # mask the rounded silhouette so the corners stay page-coloured
        m = Image.new("L", (card_box["w"], card_box["h"]), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (0, 0, card_box["w"] - 1, card_box["h"] - 1), radius=radius, fill=255)
        mm = np.asarray(m) > 0
        region = arr[card_box["y"]:card_box["y"] + card_box["h"],
                     card_box["x"]:card_box["x"] + card_box["w"]]
        region[mm] = tex[mm]
    return arr


def test_detect_screenshot_card_dark_on_white():
    # H9: dark rounded chat card on white page
    arr = _rounded_card((1200, 800, (255, 255, 255)),
                        {"x": 80, "y": 120, "w": 640, "h": 960}, (28, 28, 30), 40)
    card = ig_text.detect_screenshot_card(arr)
    assert card is not None
    assert card["page_bg"] == "#ffffff"
    assert card["separates_clean"] is True
    for k, v in {"x": 80, "y": 120, "w": 640, "h": 960}.items():
        assert abs(card["bbox"][k] - v) <= 6, (k, card["bbox"])


def test_detect_screenshot_card_photo_on_black():
    # H16: photo card (textured) on black page — must still separate cleanly
    arr = _rounded_card((1400, 800, (0, 0, 0)),
                        {"x": 90, "y": 300, "w": 620, "h": 760}, (200, 190, 170), 36,
                        noise_card=True)
    card = ig_text.detect_screenshot_card(arr)
    assert card is not None
    assert card["page_bg"] == "#000000"
    assert card["separates_clean"] is True
    # a photo interior is NOT uniform -> no flat card fill is claimed
    assert card["card_fill"] is None


def test_detect_ig_story_text_runs():
    # two stacked lines on their own dark bars over a white page
    arr = _rounded_card((900, 700, (255, 255, 255)),
                        {"x": 150, "y": 200, "w": 300, "h": 70}, (17, 17, 17), 16)
    img = Image.fromarray(arr, "RGB")
    ImageDraw.Draw(img).rounded_rectangle((150, 280, 449, 349), radius=16, fill=(17, 17, 17))
    arr = np.asarray(img, dtype=np.uint8)
    lines = [
        {"id": "L0", "ink_box": {"x": 170, "y": 215, "w": 260, "h": 40}},
        {"id": "L1", "ink_box": {"x": 170, "y": 295, "w": 260, "h": 40}},
    ]
    runs = ig_text.detect_ig_story_text(arr, lines)
    assert runs, "no plated run detected"
    ids = [i for r in runs for i in r["line_ids"]]
    assert "L0" in ids and "L1" in ids


# ── Ellipse-masked circular inset (spec capability item 5) ───────────────────────────
def test_ellipse_inset_emission_and_reconstruct_trigger():
    reconstruct = pytest.importorskip("src.reconstruct")
    # (a) explicit ellipse_image emission carries the ellipse mask
    cand = ig_text.ellipse_image("inset0", {"x": 400, "y": 800, "w": 200, "h": 200})
    d = _compile([cand])
    node = _find(d, lambda n: n.get("id") == "inset0")[0]
    assert (node.get("mask") or {}).get("kind") == "ellipse"

    # (b) reconstruct's mask-spec resolver returns ellipse for our circular candidate
    box = {"x": 0, "y": 0, "w": 200, "h": 200}
    m = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(m).ellipse((0, 0, 199, 199), fill=255)
    alpha = np.asarray(m, dtype=np.uint8)
    spec_explicit = reconstruct._image_mask_spec(cand, alpha, box)
    assert spec_explicit["kind"] == "ellipse"

    # (c) INFERENCE path: no explicit mask, just a circular alpha -> ellipse (H4 auto)
    bare = {"id": "auto", "target": "image", "src": None, "box": dict(box), "meta": {}}
    spec_auto = reconstruct._image_mask_spec(bare, alpha, box)
    assert spec_auto["kind"] == "ellipse"
