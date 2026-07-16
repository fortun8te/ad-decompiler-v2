# Diagram / chart editability contract

Semi-editable diagrams are an **opt-in upgrade** over intentional raster clusters.
Default remains: one exact swappable crop for `chart` / `diagram` / `infographic` /
`graph` / `table`. Do not invent native geometry from a flat plot photo.

## What stays editable

| Role | Target | Notes |
|------|--------|--------|
| `data-label`, `axis-label`, `tick-label`, `legend-label` | **TEXT** | Native Figma text; never vectorize glyphs |
| `axis`, `axis-line`, `gridline`, `divider`, `bar`, `chart-bar` | **SHAPE** | Rect primitives; absolute geometry |
| `plot-line`, `data-line`, `connector`, `data-point`, `marker` | **VECTOR** (`icon`) | Only if vectorize render-back gate passes |

Layout may wrap a fully primitive `chart_group_id` as `native-chart` with
`layout.mode = NONE` (no Auto Layout guessing).

## What stays raster

- Whole `chart` / `diagram` / `infographic` / `graph` / `table` / `screenshot` /
  `receipt` / `product-cluster` regions (intentional raster clusters)
- Photos, products, people — never traced, even beside a diagram
- Any mark that fails the render-back gate (`vector_fallback: true`)
- Mixed groups that still contain an unexplained plot IMAGE crop
- Ambiguous `divider` / `connector` roles **without** `chart_group_id` (ordinary UI /
  leader routing wins)

## Detection / routing / merge hooks

1. **SAM/VLM** may still emit a whole-plot `chart`/`diagram` cluster (safe default).
2. **Decomposition** tags members with `chart_group_id` + primitive roles above.
3. **`routing.route`** sends primitives to shape/icon before the photo-fragment catch-all.
4. **`merge_layers.prefer_decomposed_charts`** demotes a same-group whole-plot raster when
   axis + ≥2 marks are present, so bars/labels are not double-owned.
5. **`layout._chart_is_deterministic`** builds `native-chart` only from routed
   shape/text/icon primitives — never from a residual plot IMAGE.

Code source of truth: `src/diagram_editability.py`.

## Known failure modes

1. **Partial decomposition** — axis + one bar only: native-chart is refused; whole-plot
   raster (if present) is kept. Prefer incomplete raster over invented marks.
2. **Cluster + primitives without demotion gate** — if marks lack `chart_group_id` or
   fail the axis/≥2-marks rule, both the flat crop and guessed chrome can coexist;
   demotion only runs when the gate passes.
3. **Label ghosts** — promoting `diagram_label` overlays while the parent chart IMAGE
   still contains the same ink doubles text unless removal punches the crop (today
   overlays are grouped with the owner; prefer demoting the cluster instead).
4. **Over-vectorizing photos** — photographic panels must keep `photo` /
   `photo-fragment` roles; chart routing never applies to them.
5. **Textured / anti-aliased plot lines** — gate fails → exact alpha raster (honest),
   not a bad SVG.
6. **Infographic photos with callout arrows** — arrows may vectorize via existing
   leader roles; the photo body stays IMAGE.
7. **No chart detector for primitives yet** — readiness is routing/layout/merge + gate
   config; producing `chart-bar` / `axis` observations is still upstream work.
