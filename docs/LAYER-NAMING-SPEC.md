# Layer naming spec

Governs designer-facing layer names emitted into `design.json` and the Figma
plugin. Implemented in `src/build_design_json.py` (`_name`, `_role_label`,
`_role_token`, `_group_content_name`, `_dedupe_sibling_names`). Fast,
deterministic, sync — no VLM call on the naming hot path (VLM group/element
names from `src/vlm_layout_group.py` are consumed when present, see below).

## Format

`<Role Label>` or `<Role Label> / <content snippet>`.

- **Role-first.** The role label always leads: `Headline`, `CTA`, `Product`,
  `Badge`, `Price`, `Disclaimer`, etc. (`_ROLE_LABELS` in
  `build_design_json.py` is the canonical role → label table.)
- **Snippet separator is `/`.** `Headline / ALLE ESSENTIALS`,
  `CTA / Koop nu`, `Badge / -50%`. One separator style is used for every role
  (not a different glyph per role) so names sort and scan predictably in the
  Figma layers panel. Snippets are NOT quoted — quote characters inside a
  layer name read as literal text to designers and are stripped
  (`_QUOTE_STYLE_RE`, `_is_machine_name`) if a name shows up already quoted.
- **Length budget: 40 chars max**, enforced by `_with_snippet`. The snippet
  portion shrinks for long role labels (`Disclaimer / ...`) and grows for
  short ones (`CTA / ...`), floor 8 chars, cap 28 chars, so the combined name
  never blows the budget. Overflow truncates with an ellipsis
  (`_clean_snippet` / `_truncate`).
- **No internal ids.** `_is_machine_name` filters out candidate ids,
  pipeline/VLM leftovers (`c_1`, `raster slice`, `swappable`, `clean plate`,
  `band-<hex>`, `text-stack-<hex>`, ...) so none of that vocabulary reaches a
  shipped name.
- **Non-English / mixed-script text is kept verbatim.** The snippet is the
  cleaned OCR/source text as-is (whitespace-collapsed, emoji-stripped only at
  the edges) — no translation, no case-forcing beyond what the source already
  has. `KRACHTSPORT BUNDEL`, `Koop nu`, `@UpfrontFood` all pass through
  unchanged.
- **Groups** get a role label from `_ROLE_LABELS` when the group's role is a
  real semantic one (`text-stack` → `Text Stack`, `header-cluster` →
  `Header`, `message-bubble` → `Message`, ...), or the VLM's own group name
  (see below) when one was assigned during layout grouping.

### Product / repeated-role disambiguation

`_name` does not number siblings itself — `_dedupe_sibling_names` walks each
sibling list bottom-up after compile and appends `/ 2`, `/ 3`, ... only when
two siblings would otherwise share an identical name (`Product`, `Product`,
`Product` → `Product`, `Product / 2`, `Product / 3`). This runs at every
nesting level independently, so identical names in different groups don't
collide with each other.

## Group naming (VLM-sourced names)

`src/vlm_layout_group.py` already runs a lightweight VLM pass that can name
groups (`plan["groups"][i]["name"]`) and individual ambiguous elements
(`plan["element_names"]`, e.g. renaming a generic detection to "brand logo").
`apply_spec()` writes these onto the candidate tree before `build_design_json`
ever runs:

- Group wrappers get `candidate["name"] = group["name"]` directly
  (`vlm_layout_group.py:431`) plus `meta["semantic_name"]`.
- Named elements get `meta["semantic_name"] = item["name"]`
  (`vlm_layout_group.py:404-408`), only when the element had no name yet.

`_explicit_designer_name` (the first thing `_name` checks) already reads
`candidate.get("name")` and `meta.get("semantic_name")` before falling
through to role-based derivation, filtered through `_is_machine_name` so a
degenerate VLM name (empty, an id, "group 1") is rejected rather than
shipped. **This wiring already existed** — the VLM group name flows straight
through to the emitted `Layer.name` with no extra plumbing needed. (One
naming-adjacent gap in `vlm_layout_group.py` — child-group boxes that clip
enlarged text children — was independently fixed in `build_design_json.py`'s
`text-stack` box-expansion logic by another concurrent change; not part of
this task.)

## What was actually broken (the fixed gap)

Audited real output (`runs/codex-targeted-002a/002_attached_.../design.json`,
`runs/bench-002/design.json`): most names were already good
(`Headline / ALLE ESSENTIALS`, `Product`, `Product / 2`, `Text Stack`). The
one recurring defect was **element-fusion residual groups** — containers
wrapping multiple children (product photos + a price + a CTA line, or just a
single mis-routed text node) whose own `meta.role` is a low-confidence
catch-all (`"shape"`, `"group"`, `"asset-group"`, `"band"`, missing). Before
the fix these groups inherited the literal role label `"Shape"`/`"Group"`,
which is meaningless for a container and, worse, collided with a sibling leaf
layer that legitimately *is* a shape — producing designer-visible names like
`Shape / 2` for a group that actually held 3 product photos, 2 prices, an
arrow and two decorations (benchmark 002, group `c_E003`).

Fix: `_group_content_name()` (new in `build_design_json.py`) ranks the
group's direct children by role label, preferring content-bearing roles
(`Product`, `Price`, `Headline`, ...) over purely decorative ones (`Shape`,
`Arrow`, `Underline`, `Strikethrough`, `Dot`), and joins the top one or two
distinct labels — `Product + Price`. It only fires when the group's own role
is in the generic set (`_GENERIC_GROUP_ROLES`); any group with a real
semantic role (`text-stack`, `header-cluster`, `button`, `cta`, ...) is
untouched.

## Non-goals / deliberately out of scope

- No VLM call added to the naming path itself — grouping's existing VLM pass
  is reused, not duplicated.
- No change to the `/` separator convention or the 40-char budget; both were
  already implemented and covered by 12 passing tests before this change.
