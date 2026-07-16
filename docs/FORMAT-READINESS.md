# Format readiness (without exploding presets)

The pipeline keeps **five decomposition archetypes** (`social_screenshot`,
`caption_over_photo`, `comparison_grid`, `lifestyle_overlay`, `product_on_flat`).
Those stay the coarse contracts for photo policy, grouping, and QA floors.

New creative **formats** (stories 9:16, feed 1:1, carousels, UGC, before/after,
testimonials, product+copy, UI screenshots, caption pills, diagrams-in-ads, …)
must **not** each become another named preset. Instead use:

1. **Aspect class** — frame geometry bucket (`story`, `portrait`, `square`,
   `landscape`, `wide`).
2. **Scene capabilities** — boolean flags about what the scene needs
   (`text_plates`, `ui_chrome`, `cutouts`, `diagrams`, `gradients`,
   `comparison_columns`, `caption_stack`, `flat_plate`, `icons_as_chips`).
3. **Optional tags / overrides** — batch metadata, not classifiers.

Artifacts per run:

- `archetype.json` — includes a nested `format` block.
- `format.json` — the same format profile for quick inspection.
- `cfg["scene"]["format"]` — live contract for routing / peel / inpaint / reconstruct.

Benchmark planning (`planned.json` v2) peeks each image’s size and attaches
`aspect_class` / `aspect_ratio`, plus optional tags from
`<input_dir>/format_index.json`.

## How to add a new format (no new preset)

### A. Geometry-only (stories / feed / landscape)

Nothing to code. `src/format_readiness.classify_aspect` already buckets
1080×1920 → `story`, 1080×1080 → `square`, etc. Stages that care about canvas
shape should read `scene.format.aspect_class`.

### B. Behavior you already get from an archetype

If the creative is “caption pills over a photo”, the existing
`caption_over_photo` archetype + inferred `text_plates` / `caption_stack`
capabilities are enough. Prefer capability checks:

```python
from src import format_readiness

if format_readiness.has_capability(cfg, "text_plates"):
    ...
```

instead of `if archetype == "instagram_story_caption_v3"`.

### C. Batch hint without changing code

Add or extend `<input_dir>/format_index.json`:

```json
{
  "016": {
    "tags": ["product_copy", "story"],
    "notes": "packshot + stacked offer copy"
  },
  "201": {
    "tags": ["before_after"],
    "capabilities": { "comparison_columns": true }
  },
  "155": {
    "tags": ["caption_pill", "testimonial"],
    "capabilities": { "caption_stack": true, "text_plates": true }
  }
}
```

Known soft tags (boost capabilities, do not invent presets):
`ugc`, `testimonial`, `caption_pill`, `caption_stack`, `before_after`,
`carousel`, `ui_screenshot`, `diagram`, `product_copy`, `timeline`,
`health_product`.

`benchmark.py` copies these into `planned.json` so activity grids / batch
runners can slice by aspect or tag before a run starts.

### D. Global capability override for a campaign

In `config.yaml`:

```yaml
format:
  capabilities:
    caption_stack: true
    text_plates: true
  tags: [caption_pill]
```

Overrides win over inference for listed keys only.

### E. New stage behavior for a capability

1. Add or reuse a flag in `src/format_readiness.CAPABILITIES` (only if truly new).
2. Infer it from existing `scene_facts` / preset grouping in `infer_capabilities`.
3. Gate the stage with `has_capability(cfg, "…")`, keeping archetype-name fallbacks
   for legacy runs that lack `format.json`.
4. Unit-test inference + the gate. Do **not** add a sixth archetype unless the
   photo-ownership / QA contract is genuinely new.

## What still uses archetypes (on purpose)

| Concern | Source of truth |
| --- | --- |
| QA recall / LPIPS floors, reward weights | `archetype.PRESETS[*].thresholds` |
| Comparison BEFORE/AFTER photo rebuild | `comparison_grid` + `before_after_pair` fact |
| Social header / avatar clustering | `social_screenshot` preset |
| Coarse photo retain / suppress policy | `preset.photo_regions` |

Capabilities cover **construction tactics** (solid flat fill, icon chips,
caption stacks). Archetypes cover **acceptance contracts**.

## Simpletics IG caption reference

`work/_figma_15510.png` + `work/_figma_15510.json` (1080×1920 stacked caption
pills) is the reference for `aspect_class=story` + `caption_stack` +
`text_plates`. That combination should ride `caption_over_photo` (or
`social_screenshot` when chrome is present) — not a new preset name.

## Batch multi-format without presets

```bash
# 1) Optional: tag the fixture set
#    benchmark_set/format_index.json  (or next to your input dir)

# 2) Plan + run — planned.json lists aspect_class per image
python benchmark.py --input-dir benchmark_set --output runs/multi-format --ids 016,009,002

# 3) Slice later by planned.json fields (aspect_class, format_tags)
#    Activity grid watches the run dirs; it does not need new preset enums.
```

Same `config.yaml` archetype `preset: auto` for the whole batch. Per-image
format differences come from geometry + facts + optional tags.
