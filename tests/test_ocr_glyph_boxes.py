"""OCR line boxes must not claim pixels that belong to adjacent artwork.

Both regressions here were diagnosed from runs/postfix-benchmark-6: a text line
whose box overhangs an icon/emoji makes that region text-owned, so element
detection never emits the art and its raster chip is sliced against an empty
alpha ledger.
"""

from src import ocr


def _line(text, box, words, meta=None):
    return {"text": text, "box": dict(box), "quad": ocr._rect_quad(box),
            "words": [dict(w) for w in words], "meta": dict(meta or {})}


# ── 107: '058%' -> '58%' must free the ↓-in-circle icon ──────────────────────────

def test_edge_char_drop_detects_only_edge_truncations():
    assert ocr._edge_char_drop("058%", "58%") == (1, 0)
    assert ocr._edge_char_drop("58%x", "58%") == (0, 1)
    # A substitution leaves the glyph extent alone and must not move the box.
    assert ocr._edge_char_drop("5B%", "58%") == (0, 0)
    assert ocr._edge_char_drop("58%", "58%") == (0, 0)


def test_numeric_correction_retightens_box_off_the_icon():
    """107: doctr swallowed the ↓ icon as a leading '0' (box x=288); easyocr read
    the same row from x=378. Correcting the text must pull the box past the icon
    (which ends at x=372) so the glyph can be detected as artwork."""
    box = {"x": 288.0, "y": 341.0, "w": 449.0, "h": 139.0}
    line = _line("058%", box, [{"text": "058%", "box": box, "conf": 0.989}], meta={
        "provenance": [
            {"engine": "doctr", "text": "058%", "selected": True, "box": box},
            {"engine": "easyocr", "text": "589", "selected": False,
             "box": {"x": 378.0, "y": 326.0, "w": 392.0, "h": 176.0}},
        ]
    })
    info = ocr._retighten_after_edge_char_drop(line, "058%", "58%")
    assert info is not None and info["dropped_leading"] == 1
    icon_right_edge = 372.0
    assert line["box"]["x"] >= icon_right_edge, "box still overhangs the ↓ icon"
    # The peer's edge is real ink evidence and clips less than the proportional
    # estimate (288 + 449/4 = 400), so it wins.
    assert line["box"]["x"] == 378.0
    assert line["box"]["x"] + line["box"]["w"] == 737.0, "right edge must not move"
    assert line["quad"] == ocr._rect_quad(line["box"])
    # The word carrying the stale reading follows the line.
    assert line["words"][0]["text"] == "58%"
    assert line["words"][0]["box"]["x"] == 378.0


def test_retighten_falls_back_to_proportional_without_a_peer():
    box = {"x": 288.0, "y": 341.0, "w": 449.0, "h": 139.0}
    line = _line("058%", box, [{"text": "058%", "box": box, "conf": 0.989}])
    info = ocr._retighten_after_edge_char_drop(line, "058%", "58%")
    assert info is not None
    assert line["box"]["x"] == 288.0 + 449.0 / 4  # one glyph of four
    assert line["box"]["x"] > 372.0, "still clears the icon"


def test_retighten_ignores_substitution_and_keeps_box():
    box = {"x": 100.0, "y": 10.0, "w": 200.0, "h": 40.0}
    line = _line("5B%", box, [{"text": "5B%", "box": box, "conf": 0.9}])
    assert ocr._retighten_after_edge_char_drop(line, "5B%", "58%") is None
    assert line["box"] == box


def test_retighten_refuses_to_eat_most_of_the_line():
    box = {"x": 0.0, "y": 0.0, "w": 100.0, "h": 20.0}
    line = _line("12345", box, [{"text": "12345", "box": box, "conf": 0.9}])
    # Dropping 4 of 5 glyphs would leave <25% of the box: refuse rather than guess.
    assert ocr._retighten_after_edge_char_drop(line, "12345", "5") is None
    assert line["box"] == box


def test_peer_edge_ignores_a_peer_on_another_row():
    box = {"x": 288.0, "y": 341.0, "w": 449.0, "h": 139.0}
    meta = {"provenance": [{"engine": "easyocr", "selected": False,
                            "box": {"x": 378.0, "y": 900.0, "w": 392.0, "h": 40.0}}]}
    assert ocr._peer_edge(meta, box, leading=True) is None


def test_peer_edge_never_widens_the_winner():
    box = {"x": 288.0, "y": 341.0, "w": 449.0, "h": 139.0}
    meta = {"provenance": [{"engine": "easyocr", "selected": False,
                            "box": {"x": 100.0, "y": 341.0, "w": 700.0, "h": 139.0}}]}
    assert ocr._peer_edge(meta, box, leading=True) is None


# ── 009: a junk-read emoji must not be kept inside the line box ──────────────────

def test_orphan_trailing_emoji_word_is_excised_and_box_retightened():
    """009: '👀' is read as 'GC' (conf 0.52) at x=620..657 and dropped from the
    text, but the line box's right edge is exactly 657 — so the emoji's pixels
    stay text-owned and it never ships as an image chip."""
    line = _line(
        "woensdag 20 mei om 20:00 uur.",
        {"x": 48.0, "y": 729.0, "w": 609.0, "h": 38.0},
        [
            {"text": "woensdag", "box": {"x": 49, "y": 736, "w": 172, "h": 30}, "conf": 0.923},
            {"text": "20", "box": {"x": 238, "y": 733, "w": 43, "h": 31}, "conf": 0.999},
            {"text": "mei", "box": {"x": 295, "y": 733, "w": 59, "h": 28}, "conf": 0.754},
            {"text": "om", "box": {"x": 367, "y": 739, "w": 53, "h": 23}, "conf": 0.999},
            {"text": "20:00", "box": {"x": 435, "y": 734, "w": 96, "h": 26}, "conf": 0.999},
            {"text": "uur.", "box": {"x": 546, "y": 736, "w": 62, "h": 27}, "conf": 0.862},
            {"text": "GC", "box": {"x": 620, "y": 730, "w": 37, "h": 28}, "conf": 0.521},
        ],
    )
    info = ocr._excise_orphan_edge_word(line)
    assert info is not None
    assert [w["text"] for w in info["excised"]] == ["GC"]
    emoji_left_edge = 620.0
    right = line["box"]["x"] + line["box"]["w"]
    assert right <= emoji_left_edge, f"box right {right} still covers the 👀"
    assert right == 608.0  # tightened to the last real glyph, 'uur.' (546 + 62)
    assert "GC" not in [w["text"] for w in line["words"]]


def test_orphan_excision_keeps_a_word_the_text_uses():
    """A confidently-read token that the line text contains is real copy."""
    line = _line(
        "SIZE X",
        {"x": 0.0, "y": 0.0, "w": 100.0, "h": 20.0},
        [
            {"text": "SIZE", "box": {"x": 0, "y": 0, "w": 60, "h": 20}, "conf": 0.9},
            {"text": "X", "box": {"x": 80, "y": 0, "w": 20, "h": 20}, "conf": 0.4},
        ],
    )
    assert ocr._excise_orphan_edge_word(line) is None
    assert len(line["words"]) == 2


def test_orphan_excision_spares_a_confident_edge_word():
    line = _line(
        "hello",
        {"x": 0.0, "y": 0.0, "w": 100.0, "h": 20.0},
        [
            {"text": "hello", "box": {"x": 0, "y": 0, "w": 60, "h": 20}, "conf": 0.9},
            {"text": "world", "box": {"x": 80, "y": 0, "w": 20, "h": 20}, "conf": 0.95},
        ],
    )
    assert ocr._excise_orphan_edge_word(line) is None


def test_orphan_excision_spares_an_attached_word():
    """No gap => an engine mis-split of one word, not a detached glyph."""
    line = _line(
        "hello",
        {"x": 0.0, "y": 0.0, "w": 80.0, "h": 20.0},
        [
            {"text": "hello", "box": {"x": 0, "y": 0, "w": 60, "h": 20}, "conf": 0.9},
            {"text": "o", "box": {"x": 60, "y": 0, "w": 20, "h": 20}, "conf": 0.3},
        ],
    )
    assert ocr._excise_orphan_edge_word(line) is None


def test_orphan_excision_spares_interior_interpunct_separators():
    """009's timestamp row: '·' words are interior, never the outermost token."""
    line = _line(
        "05:00 PM · 12-05-2026 · 121K weergaven",
        {"x": 20.0, "y": 922.0, "w": 674.0, "h": 42.0},
        [
            {"text": "05:00", "box": {"x": 20, "y": 922, "w": 100, "h": 40}, "conf": 0.9},
            {"text": ".", "box": {"x": 140, "y": 930, "w": 8, "h": 8}, "conf": 0.3},
            {"text": "12-05-2026", "box": {"x": 170, "y": 922, "w": 200, "h": 40}, "conf": 0.9},
            {"text": "121K", "box": {"x": 400, "y": 922, "w": 90, "h": 40}, "conf": 0.9},
            {"text": "weergaven", "box": {"x": 510, "y": 922, "w": 184, "h": 40}, "conf": 0.9},
        ],
    )
    assert ocr._excise_orphan_edge_word(line) is None
    assert any(w["text"] == "." for w in line["words"])


def test_tag_lines_applies_excision_and_records_meta():
    raw = [{
        "text": "woensdag 20 mei om 20:00 uur.",
        "box": {"x": 48.0, "y": 729.0, "w": 609.0, "h": 38.0},
        "words": [
            {"text": "uur.", "box": {"x": 546, "y": 736, "w": 62, "h": 27}, "conf": 0.862},
            {"text": "GC", "box": {"x": 620, "y": 730, "w": 37, "h": 28}, "conf": 0.521},
        ],
    }]
    tagged = ocr._tag_lines(raw, "doctr")
    assert tagged[0]["meta"]["orphan_edge_word_excised"]["excised"][0]["text"] == "GC"
    assert tagged[0]["box"]["x"] + tagged[0]["box"]["w"] <= 620.0


def test_tag_lines_excision_is_idempotent():
    raw = [{
        "text": "woensdag uur.",
        "box": {"x": 48.0, "y": 729.0, "w": 609.0, "h": 38.0},
        "words": [
            {"text": "woensdag", "box": {"x": 48, "y": 736, "w": 172, "h": 30}, "conf": 0.9},
            {"text": "uur.", "box": {"x": 546, "y": 736, "w": 62, "h": 27}, "conf": 0.862},
            {"text": "GC", "box": {"x": 620, "y": 730, "w": 37, "h": 28}, "conf": 0.521},
        ],
    }]
    once = ocr._tag_lines(raw, "doctr")
    twice = ocr._tag_lines(once, "doctr")
    assert once[0]["box"] == twice[0]["box"]
    assert len(twice[0]["words"]) == 2
