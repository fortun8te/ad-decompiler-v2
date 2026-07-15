# CODIA PARITY SPEC

Two ground-truth templates, both Codia AI's real Figma output pulled via the REST API
(fileKey `P2JsJHzbIHoMA8jJ2lMURI`):

1. `runs/codia-teardown-009.json` — **complex UI screenshot** (dark X/Twitter post,
   `benchmark_set/009_attached_885c19be02ccf229.png`, node `15487:5`). Grouped tree,
   38 nodes. Sections 1–7 below.
2. `runs/codia-teardown-2.json` — **simple photo ad** (hair-product ad,
   `IMAGE AD INSPO/041_attached_885052038efb9603.webp`, node `15487:100`). Completely
   flat tree, **9 nodes**. Section 9 below.

The pair defines Codia's construction *policy*, not just one output. Our output under
comparison: `runs/benchmark-final/009_attached_885c19be02ccf229/` and
`runs/golden-optimized-check/041_attached_885052038efb9603/`.

All Codia coordinates below are converted to image space (root frame absolute origin is
(-689, -1289); add +689/+1289). Canvas is 1080x1080.

Automated comparator: `scripts/codia_parity.py` (gate: `tests/test_codia_parity.py`).

---

## 1. Global anatomy of Codia's tree

- **38 nodes total** for the full reconstruction: 8 FRAMEs, 3 solid RECTANGLEs,
  11 image-fill RECTANGLEs, 16 TEXT nodes. Nothing else. (Ours: 35 nodes — budget is
  comparable; the composition is what differs.)
- **Max leaf depth 5** (`root frame > Root > Groups > Button > Background/Text`).
- **One extra sibling node** (`15487:44`): a 1080x1080 RECTANGLE with the *original
  screenshot* as an image fill, parked 1130 px to the left of the recreation. Codia
  ships the source next to the rebuild as a visual reference. Not part of parity.
- **Naming is dumb and generic**: `Background`, `Groups`, `Image`, `Button`, and text
  nodes named by their first 16 characters of content (`"De Vakantiegelds"`,
  `"LAATSTE SITE WID"`, `"05:00 PM · 12-05"`). No semantic roles anywhere.
- **What Codia does NOT emit** (checked across every node):
  - No VECTOR / BOOLEAN_OPERATION / ELLIPSE nodes. Zero vectorization.
  - No masks (`isMask` never set), no alpha mattes. Icon/emoji/avatar = plain
    RECTANGLE with `fills:[{type:"IMAGE", scaleMode:"FILL", imageRef:...}]`.
  - No strokes (every `strokes: []`), no effects (every `effects: []`).
  - No auto-layout (no `layoutMode`), no meaningful constraints (children all carry the
    default `BOTTOM/RIGHT`; root is `TOP/LEFT_RIGHT`). Pure absolute positioning via
    `relativeTransform`.
  - No inpainted background plate. Background is **solid fills** (see §4).
  - No `characterStyleOverrides` — style changes are handled by *splitting nodes*.
- Frames: `clipsContent: false` everywhere except the outermost frame (`true`), and all
  Groups frames have a fully transparent fill (`opacity: 0` solid black).

### Hierarchy (image-space boxes)

```
15487:5  FRAME "Figma design - 009_...png"  (0,0 1080x1080) clipsContent=true
└ 15487:6  FRAME "Root" (0,0 1080x1080)
  ├ 15487:7  RECT "Background" (0,0 1080x133)  SOLID #060606        <- top-bar plate
  ├ 15487:8  FRAME "Groups" (0,124 1080x956)                        <- post body cluster
  │ ├ 15487:9  RECT "Background" (0,132 1080x948) SOLID #000000     <- main plate
  │ ├ 15487:10 FRAME "Groups" (0,878 1080x202)                      <- engagement row
  │ │ ├ 11 Image share (1007,1012 46x47)   ├ 12 TEXT "89"
  │ │ ├ 13 Image bookmark (765,1010 41x54) ├ 14 TEXT "21K"
  │ │ ├ 15 Image heart (516,1014 51x45)    ├ 16 TEXT "66"
  │ │ ├ 17 Image retweet (266,1015 62x42)  ├ 18 TEXT "257"
  │ │ ├ 19 Image comment (26,1010 53x51)
  │ │ ├ 20 TEXT "weergaven" (Light)  ├ 21 TEXT "121K" (Bold)
  │ │ └ 22 TEXT "05:00 PM · 12-05-2026 ·" (Light)
  │ ├ 15487:23 Image eyes-emoji 👀 (621,731 36x29)
  │ ├ 15487:24 Image hourglass-emoji ⌛ (711,322 26x38)
  │ ├ 15487:25 FRAME "Groups" (0,132 1080x155)                      <- header cluster
  │ │ ├ 26 FRAME "Button" (828,131 210x70)
  │ │ │ ├ 27 RECT "Background" (833,134 202x67) #EEF2F3 r=33
  │ │ │ └ 28 TEXT "Volgend" (864,146) Inter SemiBold 600 35px #1D1E1F
  │ │ ├ 29 TEXT "@UpfrontFood" (182,191)
  │ │ ├ 30 Image verified-badge (351,157 29x29)
  │ │ ├ 31 TEXT "UPFRONT" (182,151) Bold 700
  │ │ └ 32 Image avatar (22,131 125x125)  <- square RECT, no circle mask
  │ ├ 33..37 TEXT body copy (5 nodes, see §2)
  └ 15487:38 FRAME "Groups" (0,0 1080x133)                          <- nav cluster
    └ 15487:39 FRAME "Groups" (0,0 1080x122)
      ├ 40 Image ellipsis "..." (978,68 50x11)
      ├ 41 TEXT "Post" (491,49) Inter ExtraBold 800 47px
      └ 42 Image back-arrow (30,53 49x42)
```

Z-order = Figma child order (index 0 = backmost): plates first, content clusters above,
nav cluster last. The engagement icons sit *directly on the solid black* — their image
cutouts simply include the local black backdrop, which is why no matte is needed.

---

## 2. Text construction — the core of Codia's quality

**All 16 visible text lines are native TEXT nodes. 16/16. Zero raster slices.**
Every single node is `fontFamily: "Inter"`. Weight varies; family never does.

| id | characters | weight/style | size | lineHeightPx | color | box (x,y,w,h) |
|----|-----------|--------------|------|--------------|-------|----------------|
| 41 | `Post` | 800 Extra Bold | 47 | 56.88 (100%) | #CDCDCD | 491,49,106,46 |
| 28 | `Volgend` | 600 Semi Bold (`Inter-SemiBold`) | 35 | 42.36 (100%) | #1D1E1F | 864,146,143,44 |
| 31 | `UPFRONT` | 700 Bold | 33 | 39.94 (100%) | #ECECEC | 182,151,163,40 |
| 29 | `@UpfrontFood` | 400 Regular | 34 | 41.15 (100%) | #7D7D7D | 182,191,238,44 |
| 37 | `LAATSTE SITE WIDE SALE VAN 2026` | 400 | 37 | 44.78 (100%) | #DADADA | 48,318,651,47 |
| 36 | `De Vakantiegeldsale komt eraan, waarbij je 20%\nkorting krijgt op het volledige assortiment.` | 400 | 37 | **45.44 PIXELS** | #D5D5D5 | 48,409,842,96 |
| 35 | `Daarbovenop krijgen de eerste 500 bestellingen hun\ngeld terug tot €100.` | 400 | 37 | **44.31 PIXELS** | #D0D0D0 | 47,544,924,96 |
| 34 | `Schrijf je nu in en mis geen enkele update. We zien je` | 400 | 37 | 44.78 (100%) | #D5D5D5 | 47,680,933,47 |
| 33 | `woensdag 20 mei om 20:00 uur.` | 400 | 37 | 44.78 (100%) | #CFCFCF | 46,730,568,43 |
| 22 | `05:00 PM · 12-05-2026 ·` | **300 Light** | 34 | 41.15 (100%) | #626465 | 20,920,399,45 |
| 21 | `121K` | **700 Bold** | 35 | 42.36 (100%) | #CCCCCC | 430,921,82,44 |
| 20 | `weergaven` | **300 Light** | 35 | 42.36 (100%) | #6E6F72 | 520,923,182,46 |
| 18 | `257` | 400 | 35 | 42.36 (100%) | #6F7274 | 105,1011,65,44 |
| 16 | `66` | 400 | 37 | 44.78 (100%) | #757679 | 348,1011,50,43 |
| 14 | `21K` | 400 | 36 | 43.57 (100%) | #77787B | 597,1010,65,45 |
| 12 | `89` | 400 | 37 | 44.78 (100%) | #767A7A(#76777A) | 837,1011,49,42 |

Shared invariants across all 16 nodes:

- `letterSpacing: 0.0` — **always**. No fitted tracking, ever.
- `textAlignHorizontal: LEFT`, `textAlignVertical: CENTER` — always. Boxes are *loose*
  (detection box, ~15–30% taller than the ink; e.g. "89": box 49x42 vs rendered ink
  41x28). Vertical centering absorbs the box slop, so baseline placement never depends
  on a tight ink box. This is why Codia's text never sits visibly high or low.
- `fontSize` is sampled **per line** and NOT normalized (same visual size scores 33–37
  across lines). Deviations of ±2px against the true 35/36px UI font are tolerated.
- Colors are sampled per node (five different grays across the engagement counters:
  #6F7274/#757679/#77787B/#76777A/#6E6F72). No palette clustering.
- Single-line nodes: `lineHeightUnit: INTRINSIC_%` at 100%. Multi-line paragraphs:
  measured `PIXELS` (≈1.20–1.23 x fontSize).
- `fontPostScriptName` is null except the SemiBold (`Inter-SemiBold`) — the four
  standard weights load from the family+style, no PS-name gymnastics.

### 2a. Mixed-weight line → node split (the "121K weergaven" pattern)

The footer line `05:00 PM · 12-05-2026 · 121K weergaven` is **three sibling TEXT
nodes**, split exactly at weight boundaries:

1. `05:00 PM · 12-05-2026 ·` — Light 300, #626465 (note the trailing interpunct stays
   with the light run)
2. `121K` — Bold 700, #CCCCCC
3. `weergaven` — Light 300, #6E6F72

`characterStyleOverrides` is empty in all three: Codia *never* uses styled ranges — a
weight change always produces a new sibling node. Same rule explains `UPFRONT` (700) vs
`@UpfrontFood` (400) as separate nodes, and `Volgend` (600) standing alone.

### 2b. Emoji-adjacent lines → strip + image cutout

- `LAATSTE SITE WIDE SALE VAN 2026 ⌛` → TEXT node with characters ending at `...2026`
  (emoji **removed from the string**) + sibling RECTANGLE image cutout of the ⌛ pixels
  (26x38 at 711,322).
- `...We zien je woensdag 20 mei om 20:00 uur. 👀` → TEXT ends at `uur.` + 👀 cutout
  (36x29 at 621,731).
- The emoji cutouts are children of the *body cluster*, not of any text node. They are
  positioned independently at their pixel location.

### 2c. Paragraph granularity

Two-line paragraphs are **one TEXT node with `\n`** (nodes 35, 36) with pixel line
height. But the last paragraph (`Schrijf je nu in ... / woensdag ... uur.`) is split
into two single-line nodes (34, 33) — almost certainly because the emoji terminates the
paragraph. So: merge wrapped lines into one node *unless* an emoji/inline-image breaks
the run. Codia does not guarantee paragraph merging; per-line nodes are acceptable,
merged is preferred.

---

## 3. Icons / avatar / logo / verified badge / emoji — all image cutouts

All 11 non-text foreground elements are `RECTANGLE` + `fills: [IMAGE, scaleMode FILL]`,
cut at a tight pixel box (see §1 hierarchy for the exact boxes). Specifics:

- **Avatar** (`15487:32`, 125x125 at 22,131): a *square* rectangle — no ellipse, no
  cornerRadius, no mask. The circular look ships inside the pixels (black surround
  matches the solid plate). It has **two image fills**: index 0 `visible: false`
  (imageRef `9956ad...`) + index 1 visible (imageRef `bce617...`) — an alternate
  crop kept as a hidden fill. Swappability = "replace the fill", not "edit a mask".
- **Verified badge** (29x29): image cutout. NOT a vector, despite being a trivially
  traceable glyph.
- **Engagement icons** (comment/retweet/heart/bookmark/share): five separate cutouts,
  each paired as a sibling to its count TEXT node inside the engagement Groups frame.
- **Ellipsis "..."** in the nav (50x11) and the **back arrow** (49x42): cutouts.
- **Emoji** (⌛, 👀): cutouts (§2b).
- The `UPFRONT` brand lettering is **plain TEXT** (Inter Bold 700) — Codia did not
  treat it as a wordmark/logo image. Only the roundel avatar is pixels.

## 4. Background / plate strategy

No inpainting, no clean-plate image. Codia detected the flat scene and shipped:

- `15487:7` RECT solid `#060606` (0,0 1080x133) — the slightly-lighter top nav strip.
- `15487:9` RECT solid `#000000` (0,132 1080x948) — the main plate.

Text/icon removal is therefore *free*: paint solid rects, then set every cutout's box
tight enough that its baked-in black surround is invisible on the black plate. For a
photographic background this trick would not hold — but for UI screenshots (a huge slice
of ad creative) solid plates beat inpainting on cleanliness, node weight, and edit cost.

## 5. The "Volgend" button

```
FRAME "Button" (828,131 210x70)  fills: transparent, clipsContent false
├ RECT "Background" (833,134 202x67)  SOLID #EEF2F3, cornerRadius 33 (full pill), no stroke, no effects
└ TEXT "Volgend"   (864,146 143x44)   Inter Semi Bold 600, 35px, #1D1E1F, letterSpacing 0, align LEFT/CENTER
```

Padding: text sits +31px from the pill's left edge, +12px from its top. The frame is a
5px/3px loose wrapper around the pill. No shadow (our output invented a drop-shadow),
no stroke, and the fill is the sampled off-white `#EEF2F3` (ours: `#eff3f4` — that part
we matched almost exactly).

---

## 6. DELTA TABLE — Codia vs our 009 output, and the change required

Our reference: `runs/benchmark-final/009.../design.json` (35 nodes, depth 3,
`editable_ratio` 0.74, `native_leaf_ratio` 0.68).

| Element class | Codia | Ours (benchmark-final/009) | Required change (owner) |
|---|---|---|---|
| Native text coverage | **16/16 TEXT nodes** | **7/16** native; 7 raster slices (c_B4,B5,B6,B8,B10,B12,B15) + 2 counters as low-conf slices | Slices must stop firing for correct-class fits on UI archetypes. reconstruct.apply_raster_slice_fallback + schema.raster_slice_thresholds (A1+A4); upstream fit quality via Inter default (A2) |
| Font family | Inter x16, never anything else | Carlito, Arimo, Caladea (serif!), Open Sans mix | Platform-UI prior: archetype social/UI-screenshot ⇒ default family Inter unless render-fit strongly disagrees; numeric-only lines can never pick serif (A2 text_analysis) |
| Font weight | Per-line: 300/400/600/700/800, correct everywhere | 'Volgend' w700 (should be 600); '121K' bold lost inside raster slice; 'weergaven' light lost | Weight estimation per run + SemiBold support; split mixed-weight lines (A2), emit as separate sibling TEXT nodes (A4 build_design_json) |
| Mixed-weight line | 3 sibling nodes split at weight boundaries, no styled ranges | single OCR line → 1 raster slice c_B10 | Split at stroke-weight run boundaries: '05:00 PM · 12-05-2026 ·'(300) / '121K'(700) / 'weergaven'(300) (A2 → A4) |
| letterSpacing | 0.0 on all 16 nodes | fitted noise: +1.364, -1.833, -4.457, +1.528, +1.837 | Snap tracking to 0 when |fit| < ~2.5% of fontSize; only keep confident large tracking (A2 text_analysis/font_fit) |
| Text box + align | Loose box, textAlignVertical CENTER absorbs slop | tight ink box, top-anchored (baseline drift visible for 'Daarbovenop', box h 33 < fontSize 38.6) | Emit box ≥ lineHeight tall, verticalAlign CENTER (A4 build_design_json; preview already honors verticalAlign) |
| fontSize | per-line sample, ±2px tolerated, no normalization | comparable (34.9–44.5) — OK | none (keep) |
| Emoji | stripped from text; exact pixel cutouts (⌛ 26x38, 👀 36x29) | vectorized to flat single-color paths (#edecea, #f8f7f6) → broken preview glyphs | Route meta.emoji → image cutout, never icon/text-glyph (A2 routing.route step 2); strip emoji from line text + emit sibling image node (A2 text_analysis / vlm_scene_text) |
| Engagement icons | 5 image cutouts (black backdrop baked in) | vector paths (editable, mostly good) + one bogus ellipse 'Button' #030506 under comment icon | Acceptable divergence (vectors are richer) BUT kill the fake button shell under the comment bubble; on trace-fidelity failure fall back to cutout not flat fill (A1 reconstruct/element_fusion) |
| Verified badge | image cutout 29x29 | vector path flat #209aea | Acceptable; cutout fallback when trace fidelity < gate (A1) |
| Avatar | square RECT image fill, hidden alternate fill, NO mask | image + ellipse mask ('swappable crop') 123x124 | Ours is arguably better (true swap+mask); keep. No change |
| Button | frame + pill RECT #EEF2F3 r33 + TEXT 600 SemiBold; NO effects | group fill #eff3f4 r33.5 + TEXT w700 ls-1.833 + invented drop-shadow | Drop uncorroborated shadow (A1 reconstruct style_extraction); weight 600 (A2); ls 0 (A2) |
| Background | 2 solid rects (#060606 top bar, #000000 main) | full-canvas inpainted plate PNG | When plate is flat within tolerance, emit solid RECT(s) (banded if a UI bar exists) instead of raster plate; keep inpaint only for photographic bg (A1 inpaint/reconstruct/merge_layers) |
| Structure | 38 nodes, depth 5, generic names, plates + 4 clusters | 35 nodes, depth 3(+figma mapping), semantic names, 4 role bands | none required — ours is equal-or-better; keep budget ≤ ~45 |
| OCR literals | '66', 'weergaven', '05:00 PM · 12-05-2026 ·' | '666', 'weergaver', '05:00 PM . 12-05-2026 -' | numeric re-verify on tight crop; interpunct '·' restoration for centered-dot glyphs (A2 ocr/vlm_proofread) |
| Source reference | original image parked as sibling rect 1130px left | side_by_side.png only | Optional: plugin could add source frame next to import (A4 figma-plugin) |

### What we deliberately do NOT copy

- Vector icons: ours are editable vectors with a raster-crop fallback — strictly more
  useful than Codia's cutouts *when the trace is faithful*. We keep vectors, but the
  parity checker only demands each Codia cutout box is *covered by some leaf*.
- Semantic naming, roles, masks/swappable avatar, constraints: keep ours.
- Node 15487:44 (source copy) and the transparent 4-deep Groups nesting: ignore.

---

## 7. The five most actionable "how Codia does it" findings

1. **Inter is the answer, not font forensics.** For UI-screenshot creative, Codia ships
   Inter for every line at letterSpacing 0 and eats ±2px size error. Our per-line
   best-fit picks 4 families incl. a serif for digits, then fitted tracking (±1.8–4.5)
   makes the render *worse* than the naive choice. Default to Inter on platform-UI
   archetypes; only override on strong evidence.
2. **Native text always wins: 16/16.** Codia never rasterizes a text line. Our
   ink-IoU slice gate rasterized 7 lines + 2 low-confidence counters. Once family and
   tracking come from (1), those fits pass; the gate must also stop slicing
   correct-class fits on flat plates.
3. **Weight runs become sibling nodes.** Mixed-weight lines are split at weight
   boundaries into separate TEXT nodes (no styled ranges) — trivially safe in every
   Figma plugin. `121K`(700) / `weergaven`(300) / timestamp(300) is the template.
4. **Emoji are pixels, never glyphs or traces.** Strip emoji from the text characters,
   then place a tight image cutout at the emoji's pixel box. Vectorizing color emoji
   to a flat path (what we do today) is the single worst visual artifact in our 009.
5. **Flat plates are solid rects, not inpaints.** Two solid rectangles replace the
   whole background; icon/emoji cutouts keep the local plate color baked in, so no
   masks are needed. Reserve diffusion inpainting for photographic backgrounds.

---

## 8. Parity scoring dimensions (implemented in scripts/codia_parity.py)

| Dimension | Weight | Definition |
|---|---|---|
| native_text_ratio | 0.25 | Codia text lines matched by an editable TEXT node in ours |
| font_family | 0.08 | matched native **body** lines whose family == template's (Inter) |
| headline_font | 0.07 | display-class lines: exact family 1.0, same serif/sans class 0.5 |
| font_weight | 0.12 | matched native lines whose weight bucket == Codia's |
| font_size | 0.08 | 1 − mean(|Δsize|/size)/0.15, clamped |
| text_position | 0.08 | 1 − mean(center offset / canvas h)/0.05, clamped |
| letter_spacing | 0.04 | matched native lines with |letterSpacing| ≤ 0.5px |
| icon_cutouts | 0.08 | Codia image-cutout boxes covered by one of our leaves (IoU ≥ 0.25) |
| button | 0.05 | pill (radius ≥ 25% h, light fill ΔRGB ≤ 40) + label (weight ±100, no invented effects) |
| node_budget | 0.10 | 1.0 at ours ≤ 1x Codia's node count, linear to 0.0 at ≥ 2.5x |
| flatness | 0.05 | min(1, (codia_groups+1)/(our_groups+1)) — groups only where Codia groups |

Display-class = fontSize ≥ 1.6x the template median and ≥ 48px (041 headline). When
the template has no display line (009), headline_font mirrors font_family.

`--complexity {auto,simple,complex}`: auto derives the expectation from the template
itself (Codia group count 0 ⇒ simple/flat); `simple` forces the flat-tree expectation
(target 0 groups) regardless of template. Multi-line paragraph integrity (one node with
`\n`) is reported in `detail.paragraph_integrity`.

Overall = 100 x Σ(weight·score). Regression gate: overall must never drop vs the
recorded baseline; target ≥ 90 for "does what Codia does".

Recorded baselines (pre-fix, 2026-07-15):

- 009 (`runs/codia-parity-check`, fresh pipeline run): **45.5 / 100**
  (native 6/16, family 0, weights 3 wrong of matched, letterSpacing 0/6 clean,
  cutouts 10/11, button 0.69, budget 42 vs 38 nodes, flatness 0.47)
- 041 (`runs/golden-optimized-check/041...`): **62.0 / 100**
  (native 4/4 (!), family Arial not Inter = 0, headline Georgia = serif-class 0.5,
  weights 2/4, sizes way off (−23%), cutouts 1/2 (arrow strip lost), budget 11 vs 9,
  flatness 0.5)

---

## 9. Template 2 — the flat photo ad (041) and Codia's minimalism policy

Codia's entire reconstruction of a full-bleed photo ad is **9 nodes, depth 2, zero
groups** (`15487:100`):

```
FRAME "Figma design - 041..." (1080x1080) clipsContent=true
└ FRAME "Root"
  ├ RECT "Image"      (0,0 1080x1080)  IMAGE fill  <- the WHOLE photo as one clean plate
  ├ RECT "Image"      (72,771 217x217) IMAGE fill  <- circular product badge, square cutout
  ├ RECT "Background" (665,462 347x16) IMAGE fill  <- one leader line, rasterized strip
  ├ TEXT "Zero crunch,\nzero fuss"     Inter Bold 700 37px, RIGHT/CENTER, lh 37.6px, #D6ECEE
  ├ TEXT "Works on wet\nor dry hair"   Inter Bold 700 36px, RIGHT/CENTER, lh 38.2px, #D4EDEF
  ├ TEXT "All-day hold"                Inter Bold 700 36px, LEFT/CENTER, #CAE6E7
  └ TEXT "One Step to\nBeach-Ready Waves."  Playfair Display Bold 700 90px, CENTER, lh 101.5px (84.6%), #DDEEEF
```

Verified against the actual image fills (downloaded via `GET /v1/files/:key/images`):

- **The background plate IS inpainted** (imageRef `ec046bd0...`): headline, callouts
  and the product badge are all removed. The inpaint quality is mediocre — faint
  banding ghosts remain in the headline region — Codia tolerates soft ghosting on
  busy photo areas rather than shipping text baked in. The white callout leader lines
  are **left in the plate** (photo furniture, not content); one extra 347x16 strip
  patches the line that overlapped removed text.
- Product badge cutout keeps its baked-in surround; no mask.

Construction rules this adds (and that the checker now scores):

1. **Radical node minimalism.** Fewest nodes that reproduce the ad. Simple scene
   (≲15 elements) ⇒ completely FLAT tree — no groups at all. Groups exist only as
   spatial region containers when content genuinely clusters (009's engagement
   row/header/nav). Budget: target 1x Codia's node count, hard ceiling ~2.5x
   (`node_budget` dimension), groups only where Codia groups (`flatness`).
2. **Two-tier font policy.** Distinctive display type gets a real matched Google
   font — Playfair Display Bold 90px here, a genuine serif match for the ad's
   display face. Body/callout/UI text defaults to **Inter** with correct per-line
   weights (all three callouts Bold 700). letterSpacing 0 everywhere, again.
3. **Non-text = image-fill RECT cutouts on one clean plate.** No vectors anywhere —
   even the arrow/leader is an acceptable rasterized strip. Photo furniture that
   isn't content (leader lines) can stay in the plate.
4. **Multi-line blocks stay single TEXT nodes with `\n`** — both 2-line callouts and
   the 2-line headline. Tight pixel line heights (0.84–0.88 of intrinsic) for
   display type. `textAlignHorizontal` follows the visual anchor (RIGHT for the
   right-edge callouts, CENTER for the headline) — not everything is LEFT.

Delta vs our 041 output (`runs/golden-optimized-check/041.../design.json`, 11 nodes):

| Element | Codia | Ours | Change (owner) |
|---|---|---|---|
| Headline | Playfair Display **Bold 700, 90px** | Georgia **400, 69px** | Display-type tier: match a real Google display face + weight from stroke contrast; size from cap height (A2 text_analysis/font_fit) |
| Callouts | Inter Bold 36-37px | Arial 700 29-44px | Inter default for non-display text (A2) |
| Product badge | image cutout | vectorized 'Avatar — vector' shape | Large detailed graphic ⇒ cutout, not trace (A1/routing gate mostly exists — area 4% slipped under ICON_MAX_AREA_FRAC 6%; lower the vector ceiling for photo-role/complex-fill crops) |
| Leader lines | left in plate + 1 strip patch | dropped entirely + 4 spurious 'Photo — swappable crop' fragments | Keep linework in plate when contrast-stable; kill fragment layers with no template counterpart (A1 reconstruct/merge, A3 scene_intent) |
| Tree | 9 nodes, 0 groups | 11 nodes, 1 group | Drop the text-stack wrapper on simple scenes (A3 layout: skip grouping when root_layer_count would be < ~6 leaves) |

**Minimalism budget recommendation for owners (A1/A3/A4):** for a simple scene the
compiled design.json should land ≤ ~12 nodes (Codia 9); for a complex UI screenshot
≤ ~45 (Codia 38). Every layer that has no counterpart in the source pixels (fragment
crops, invented shells) is bloat the parity gate now prices in.
