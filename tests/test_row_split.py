"""ad 025 regression: emoji-led comparison rows must never glue into ONE text node.

025 paints three comparison rows ("😍 Blocks everything" / "😐 Blocks some" /
"😢 Blocks nothing") that share one face, one colour and one left edge. Two separate
layers used to fuse them:

* text_analysis._can_join — every absolute band (gap, size, colour) passes because the
  rows ARE typographically identical, and the _same_typography role-escape then forgives
  any role label the regex handed out, so _make_blocks emitted a single block.
* layout._semantic_text_stacks — a shared block_id BYPASSED the gap test entirely, so
  even a glued upstream block re-glued at layout time no matter the row spacing.

The fix splits at both layers (emoji-led row veto + rhythm check upstream, gap band
applied unconditionally downstream) while genuine wrapped paragraphs still join.
"""

from src import layout, text_analysis


def _row(ident, text, y, height=40.0, baseline=None, size=32.0,
         role="body", color="#111111", x=120.0, w=340.0):
    """One text line with the geometry of an 025-style comparison row."""
    box = {"x": x, "y": y, "w": w, "h": height}
    return {"id": ident, "text": text, "role": role,
            "box": dict(box), "painted_box": dict(box),
            "baseline": {"y0": baseline if baseline is not None else y + height - 8.0},
            "style": {"fontFamily": "Inter", "fontSize": size,
                      "fontWeight": 500, "color": color, "align": "LEFT"},
            "hierarchy": {"level": 2, "parent_id": None}}


def _block_ids(lines):
    return {line["block_id"] for line in lines}


# 025's three rows: identical face/colour, one left edge, gaps that sit INSIDE
# _can_join's 1.25x band (32px against a 50px allowance) — exactly the geometry that
# used to glue. Row 2 carries a different role label so the _same_typography escape is
# the mechanism under test, mirroring the regex-mislabelled lines of 067/101.
def _emoji_rows():
    return [
        _row("R1", "\U0001F60D Blocks everything", 300.0, w=340.0),
        _row("R2", "\U0001F610 Blocks some", 372.0, w=260.0, role="offer"),
        _row("R3", "\U0001F622 Blocks nothing", 444.0, w=300.0),
    ]


def test_emoji_led_rows_never_join_into_one_block():
    rows = _emoji_rows()
    previous, current = rows[0], rows[1]
    # Preconditions: the absolute bands genuinely cannot tell these rows apart, and the
    # role disagreement is escapable — without the emoji veto this pair WOULD join.
    assert not text_analysis._compatible_roles(previous["role"], current["role"])
    assert text_analysis._same_typography(previous, current, {})
    assert not text_analysis._can_join(previous, current, {}), \
        "an emoji-led row restart is a new visual row, never a paragraph continuation"
    assert not text_analysis._can_join(current, rows[2], {})
    # End to end: three rows in, three blocks out, canvas order preserved.
    text_analysis._make_blocks(rows, {"w": 1080.0, "h": 1920.0}, {})
    assert len(_block_ids(rows)) == 3, \
        f"025's comparison rows are three blocks, got {_block_ids(rows)}"
    assert [line["id"] for line in rows] == ["R1", "R2", "R3"]


def test_emoji_rows_stack_as_three_ordered_layout_rows_not_one_text_node():
    rows = _emoji_rows()
    text_analysis._make_blocks(rows, {"w": 1080.0, "h": 1920.0}, {})
    nodes = [{"id": line["id"], "target": "text", "box": dict(line["box"]),
              "text": line["text"],
              "meta": {"role": "body", "block_id": line["block_id"]}}
             for line in rows]
    out = layout._semantic_text_stacks(nodes)
    stacks = [node for node in out if node.get("target") == "group"]
    assert len(stacks) <= 1, "the rows may share ONE vertical frame, never merge"
    children = stacks[0]["children"] if stacks else out
    assert [child["id"] for child in children] == ["R1", "R2", "R3"], \
        "three separate row nodes, top to bottom — not one glued text node"


def test_genuine_wrapped_paragraph_still_joins():
    lines = [
        _row("P1", "Our clean formula supports your gut", 300.0, w=380.0),
        _row("P2", "with clinically studied probiotics", 352.0, w=360.0),
    ]
    assert text_analysis._can_join(lines[0], lines[1], {}), \
        "no emoji, a small gap and a left-edge continuation: this is one paragraph"
    text_analysis._make_blocks(lines, {"w": 1080.0, "h": 1920.0}, {})
    assert len(_block_ids(lines)) == 1, \
        f"a wrapped paragraph is ONE block, got {_block_ids(lines)}"


def test_paragraph_followed_by_emoji_row_splits_at_the_row_boundary():
    lines = [
        _row("P1", "Our clean formula supports your gut", 300.0, w=380.0),
        _row("P2", "with clinically studied probiotics", 352.0, w=360.0),
        _row("E1", "\U0001F622 Blocks nothing", 424.0, w=300.0),
    ]
    text_analysis._make_blocks(lines, {"w": 1080.0, "h": 1920.0}, {})
    assert lines[0]["block_id"] == lines[1]["block_id"], "the paragraph stays intact"
    assert lines[2]["block_id"] != lines[0]["block_id"], \
        "the emoji-led row starts its own block below the paragraph"


def test_stack_split_when_shared_block_id_rows_exceed_the_gap_rhythm():
    """A glued upstream block (shared block_id) no longer bypasses the gap test."""
    def _node(ident, y):
        return {"id": ident, "target": "text",
                "box": {"x": 120.0, "y": y, "w": 300.0, "h": 40.0},
                "text": ident, "meta": {"role": "body", "block_id": "B0"}}

    far = [_node("N1", 300.0), _node("N2", 500.0)]  # 160px gap vs a 70px band
    out = layout._semantic_text_stacks(far)
    assert [node["id"] for node in out] == ["N1", "N2"], \
        "same block_id + a gap beyond the rhythm splits back into separate rows"
    assert all(node.get("target") == "text" for node in out)

    near = [_node("N1", 300.0), _node("N2", 352.0)]  # 12px gap: a genuine wrap
    out = layout._semantic_text_stacks(near)
    assert len(out) == 1 and out[0]["target"] == "group", \
        "same-paragraph wraps with a small gap still stack into one frame"
    assert [child["id"] for child in out[0]["children"]] == ["N1", "N2"]
