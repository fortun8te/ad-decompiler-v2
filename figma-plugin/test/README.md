# Mock-Figma E2E harness for the plugin compiler

Automated production-readiness gate for `figma-plugin/code.js`: proves the Figma
REBUILD works **without opening Figma**. The real, unmodified `code.js` is loaded
into Node with a faithful mock of the `figma` global, fed the exact
`{type:"build", design, assets, ...}` message `ui.html` sends, and the resulting
node tree is asserted against the `design.json` contract.

Plain Node, zero dependencies, no build step.

## Run it

```bash
# everything (5 golden runs + 11 synthetic edge cases) — CI gate, exits non-zero on any failure
node figma-plugin/test/run_e2e.js --all

# one fixture, with degradation/info detail
node figma-plugin/test/run_e2e.js --fixture kitchen-sink --verbose

# machine-readable results
node figma-plugin/test/run_e2e.js --all --json results.json

# alternate golden-runs directory
node figma-plugin/test/run_e2e.js --all --runs-dir runs/my-other-check

# or via the Python wrapper
python scripts/plugin_e2e.py
```

## What is asserted per fixture

| Check | Rule |
|---|---|
| Import completes | `figma.ui.onmessage` resolves, zero unhandled rejections, exactly one `build-result` |
| No silent drops | every design layer produced exactly **one** node (tagged with `adDecompilerLayerId` plugin data) **or** a warning/error naming it. Predicted hoist-absorbed button backgrounds and children of component instances are exempt. |
| Hierarchy | each layer's node sits inside its design parent's node |
| Geometry | canvas-absolute box within **0.5px** of design.json. Flow auto-layout children: x/y exempt (Figma lays them out). HUG-sized frames: w/h exempt. Rotated nodes: `node.rotation == -design.rotation` (design is CW-positive, Figma CCW). |
| Masked images | the inner `"<name> — image"` shape must land on the layer box (catches offset-inside-mask-frame bugs) |
| Text | `characters` match, `fontName` was actually loaded through `loadFontAsync`, and (for non-flow, unrotated text) the node stays inside its visible box with slack `max(3px, 6%)` — slack because the mock uses synthetic font metrics (`charW = 0.52 * fontSize`, auto line height `1.2 * fontSize`), not real ones. A compiler-recorded overflow warning downgrades a containment miss to acceptable. |
| Auto layout | layers with `layout.mode` got `layoutMode`, `itemSpacing` within 0.5 of `gap` |
| Report shape | warnings/errors are `{title, detail}` strings, events have ISO timestamps + level, progress `current <= total`, `exported` PNG bytes present when ok |
| Replace mode | (replace-mode fixture) importing twice leaves exactly one root and sets `report.replaced` |

## Fixtures

* **Golden** — all `runs/golden-optimized-check/*/design.json` with their real `assets/`.
* **Synthetic** (`test/fixtures/*`), each `design.json` (+ optional `design.js` generator,
  `harness.json` config):
  * `unknown-fields` — unknown optional layer fields (incl. the upcoming raster-slice
    `fallback`) must be ignored
  * `missing-asset` — missing image asset: recorded error, partial import survives
  * `empty-text`, `zero-size`, `deep-nesting` (10 levels), `rotated-text`
  * `fonts-missing` — `installedFonts: []`, `loadFontAsync` always rejects: text fails
    loudly, the rest imports
  * `huge-svg` — 2500-path SVG + one malformed SVG (must error, not crash)
  * `kitchen-sink` — gradients, effects, dash strokes, line/polygon/star, vector SVG,
    ellipse/path/alpha masks, hoisted button group, auto-layout stack with an absolute
    child, plain (non-promoted) group, mixed text runs, component + instance, opacity/
    blend/locked/hidden, constraints
  * `hostile-tokens` — plausible-but-wrong enum tokens (`line-through`, `uppercase`,
    `inner`, unknown fill/effect kinds, unknown layer type): nothing may crash or vanish
    silently
  * `replace-mode` — same design imported twice through the replace path

### harness.json

```json
{
  "installedFonts": "auto",            // "auto" = every family the design requests + Inter; [] = nothing; or a list
  "generateAssets": { "assets/x.png": { "w": 8, "h": 8 } },   // synthesized real PNGs
  "expect": {
    "ok": true,                        // expected report.ok
    "minCreated": 3,
    "failedLayers": ["id"],           // layers allowed to fail WITH a recorded error
    "errorsContain": ["Missing image asset"],
    "warningsContain": [],
    "assetsMissing": 1,
    "skipGeometry": ["id"],
    "skipTextContainment": ["id"],
    "mustBuild": ["id"],
    "rebuild": true                    // import twice (replace-path test)
  }
}
```

## The mock (`figma-mock.js`)

Covers exactly the API surface `code.js` uses: `showUI`, `ui.postMessage/onmessage`,
`currentPage` (children/`findAll`/`selection`), `viewport`, `notify`, `clientStorage`,
`listAvailableFontsAsync`, `loadFontAsync`, `createFrame/Rectangle/Ellipse/Line/Polygon/
Star/Text/Component/TextStyle`, `createImage`, `createNodeFromSvg`, `group`, plus the
full node property surface (`fills/strokes/effects/constraints/cornerRadius/layout*/
text*/setRange*/setPluginData/exportAsync/resize/remove/appendChild`).

Faithfulness rules that make it catch real bugs:

* `safeSet` in code.js gates on `"key" in node` — the mock exposes exactly the
  properties each real node type has.
* Enum/paint/effect validation throws like the real sandbox (`in set_x: ...`).
* Text mutations throw on unloaded fonts; `loadFontAsync` rejects for families not in
  the installed list.
* `remove()` kills the subtree; `figma.group()` throws on removed nodes, re-bases child
  coordinates and scales children on group resize.
* `exportAsync` returns a real PNG whose IHDR carries `round(width) x round(height)`
  (code.js parses those bytes for text fitting).
* `figma.createAutoLayout` is deliberately **absent** (not a real API — code.js
  feature-detects it, so containers route through `createFrame` like production).

Known simplifications (documented, deliberate): no real text shaping (synthetic
monospace-ish metrics), auto-layout does not re-position children (flow x/y is exempt
from assertions instead), component instances don't clone children, `fontName` never
returns `figma.mixed`.

## Keeping it honest

`replica.js` re-implements the *pure* prediction helpers from code.js (canonical type,
boxes, group promotion, hoist absorption, flow-child detection). If those semantics
change in code.js, update replica.js to match — assertion failures whose message says
"prediction" usually mean replica drift, not a compiler bug.

Failure classification when a run goes red:

1. **Harness/mock gap** — the mock or replica mispredicts real Figma/compiler
   semantics. Fix it here.
2. **Compiler bug** — real defect in code.js. Do not patch code.js from this harness;
   report it to the compiler owner with fixture + expected vs actual.
