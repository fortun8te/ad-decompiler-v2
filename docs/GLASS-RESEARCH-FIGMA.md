# Glassmorphism reconstruction research — Figma plugin, schema, emission, QA

Target case: **docs/HARD-CREATIVES-SPEC.md H18** — UPFRONT oats ad, two frosted-glass info
chips bottom-right over a lifestyle photo (rounded-rect, semi-transparent white fill + background
blur, white label/value text on top). User-locked requirement: "opacity/glass elements must be
recognized and rebuilt in Figma super closely — real fill-opacity + background-blur
reconstruction, not a raster slice."

This is a **research doc only**. No files owned by other agents were edited. All diffs below are
proposals; git recency was checked first (see each section).

---

## 1. Figma construct — verified

### 1.1 The applyCommon → applyEffects → createShapeLayer chain (CONFIRMED)

`figma-plugin/code.js` (checked out at HEAD, last touched ~9h ago — still hot, treat as
in-flux):

- `effectFromSpec()` (line 413-438) already maps a design.json effect spec with
  `type: "blur" | "layer-blur" | "background-blur"` (case/format normalized via
  `normalizedToken`) to a native Figma effect object:
  ```js
  if (type === "BLUR" || type === "LAYER_BLUR" || type === "BACKGROUND_BLUR") {
    return {
      type: type === "BACKGROUND_BLUR" ? "BACKGROUND_BLUR" : "LAYER_BLUR",
      radius: Math.max(0, finite(pick(spec, "radius", "blur"), 8)),
      visible: spec.visible !== false,
    };
  }
  ```
  So a `{"type": "background-blur", "radius": 24}` entry in a layer's `effects` array
  round-trips to a real `BACKGROUND_BLUR` Figma effect object. No plugin code change needed
  for this mapping.

- `applyEffects()` (line 440-450) reads `layer.effects` (or `layer.style.effects`, or a legacy
  `layer.shadow` shim), maps every spec through `effectFromSpec`, and does
  `safeSet(node, "effects", effects, ...)` — i.e. it assigns the *whole effects array* to the
  node's native `effects` property in one shot. It works on **any node with an `"effects"` key**,
  which includes RECTANGLE, ELLIPSE, FRAME, GROUP, TEXT, VECTOR — all Figma scene node types
  support `effects`.

- `applyCommon()` (line 496-542) is called by every node-creation path (`createShapeLayer`,
  `createTextLayer`, image/frame paths, group path) and unconditionally calls
  `applyEffects(node, layer, context)` at line 541, **after** opacity/rotation/blend/constraints
  are set.

- `createShapeLayer()` (line 1277-1293) creates the primitive (`figma.createRectangle()` /
  `createEllipse()` / etc.), sets geometry, `applyFills`, `applyStrokes`, `applyRadius`, then
  `applyCommon(node, layer, context)` at line 1290 — which is where `BACKGROUND_BLUR` gets
  attached.

**Conclusion: the predecessor's finding is correct and reproducible.** A shape layer emitted
with `effects: [{"type": "background-blur", "radius": 24}]` will, today, with zero plugin
changes, produce a native Figma rectangle with a real `BACKGROUND_BLUR` effect. This is the
single biggest reason this feature is cheap: the hard part (native Figma glass) is already done.

### 1.2 Fill opacity vs. layer opacity — CONFIRMED distinct and both already wired

Glassmorphism needs the *fill* translucent (so blur shows through) while the node itself stays
fully opaque (so a border/stroke and any child content aren't also faded). Figma's paint model
already separates these, and the plugin already respects the separation:

- `solidPaint()` (line 237-246) sets `paint.opacity` **on the SOLID paint object itself** —
  this is Figma's per-fill `opacity` property, independent of the node's `opacity`.
- `paintFromSpec()` (line 305-347) reads `spec.opacity` / `spec.alpha` from the **fill spec**
  (`pick(spec, "opacity", "alpha")`, line 310) and threads it into `solidPaint`. A fill entry
  like `{"kind": "flat", "color": "#FFFFFF", "opacity": 0.18}` therefore becomes a native Figma
  SOLID paint at 18% opacity.
- Separately, `applyCommon()` line 506 sets the **node's** `opacity` from `layer.opacity`
  (top-level, default 1). This is Figma's layer-level opacity, which would also fade any stroke
  and children — exactly what you do NOT want for a glass chip with legible white text on top.

**Conclusion: no plugin change needed here either.** To get "18% white glass fill, node itself
still 100% opaque, text on top unaffected," emission just needs to put `opacity: 0.18` inside the
`fill` dict (not the layer's top-level `opacity`). This is purely an emission-side decision.

### 1.3 Canonical glassmorphism recipe (design convention, not Figma-specific)

The widely-used recipe (Figma community files, Apple/Material glass guidance) is:

- Fill: white (or a color sampled from the underlying content) at **10–30% opacity**. Values
  around 15-20% are typical for "frosted" over a photo; higher (25-30%) reads as a more opaque
  frosted panel.
- Background blur radius: **20–40px** at typical UI scale (chips/cards). Lower (8-16px) reads as
  a subtle blur; higher (40px+) starts to look like heavy diffusion. H18's chips at their apparent
  on-canvas size would sit around 20-30px.
- Stroke: a thin (~1px) inside-aligned white stroke at ~20-30% opacity, which reads as a
  highlight/rim-light on the glass edge and sells the "glass panel" illusion. `applyStrokes()`
  already exists in code.js and (per the strokeAlign fix noted in the file's own comments)
  correctly forces text-only OUTSIDE alignment while leaving shape strokes free to use CENTER —
  no change needed for a shape stroke.
- Corner radius: matches the visual rounded-rect radius; `applyRadius()` already handles this.

### 1.4 Does Figma's own PNG/image export actually render BACKGROUND_BLUR?

This is the important round-trip question for QA (`figma_export.png` is compared against
`design.json` via `pixel_diff`/SSIM). Based on current, well-documented Figma behavior:

- **Yes.** `BACKGROUND_BLUR` is a real compositing effect in Figma's renderer (not just an
  editor-only visual aid), and it **does** show up in exported PNGs, in the "Export" preview
  thumbnails, and in `get_image`/REST image renders — Figma composites what's *underneath* the
  node (including sibling layers and, per current Figma behavior, layers within the same frame
  stacking context) before applying the blur and painting the translucent fill over it. This has
  been true since background blur shipped in the plugin API's Effect type (`BackgroundBlurEffect`
  / `type: "BACKGROUND_BLUR"`), which is a standard, non-beta effect type in `plugin-typings`
  (confirmed present in `src/figma_import.py`'s own `_KNOWN_EFFECT_TYPES` allowlist at line 96,
  which the codebase's own import-validation step already trusts as a real, renderable type).
- **Caveat worth flagging in the doc, not code:** background blur only blurs what's *below the
  node in the same rendering context*. If the "photo" the glass chip sits on is a raster IMAGE
  layer that is a **sibling** in the same frame and z-below the chip, this works exactly as
  expected. If the photo were nested in a different frame/group with different clipping, the
  blur could sample less than intended, or (rarely) nothing, if Figma treats the boundary as an
  isolation boundary. For H18's actual layer structure (chip is a sibling shape sitting directly
  above the photo plate in the same frame, per `build_design_json`'s flat z-ordered layer list)
  this caveat does not apply — the straightforward case is exactly what we have.
- **No paid-plan gating found.** `BACKGROUND_BLUR` is a standard plugin-API effect available on
  Figma's free tier for editing and export; it is not gated behind Enterprise/Organization
  features (those gates apply to things like variables/branching in some contexts, not to the
  effects list). No subtlety here beyond the sibling-stacking-context note above.

### 1.5 Subtlety check on `applyEffects`/blur radius units

- Figma's `BackgroundBlurEffect.radius` is a plain number in the node's local coordinate space
  (same px units as the rest of the plugin API — not scaled/DPI-adjusted). `effectFromSpec()`
  reads `spec.radius` (or `spec.blur`) directly with no unit conversion (`Math.max(0,
  finite(pick(spec, "radius", "blur"), 8))`), which is correct: design.json's `radius`/`blur`
  values are already expressed in the same source-image px space the plugin uses for `x`/`y`/`w`/`h`
  everywhere else (per `docs/CONTRACT.md`'s "Pixels = source-image px" rule). **No conversion bug.**
- One genuine (but out-of-scope-to-fix-now) gap: `effectFromSpec` does not read/preserve a
  `spread`/`blendMode` for BACKGROUND_BLUR (Figma's `BackgroundBlurEffect` type doesn't have a
  `spread` field, only `radius`+`visible`, so this is actually already complete — Figma's own
  effect shape for BACKGROUND_BLUR has no more fields to carry). Nothing to add here.

**Verdict on figma-plugin/code.js: genuinely production-ready for glass shapes as-is.** No plugin
edit is required or recommended. (Per the task's carve-out for "minimal additions if a genuine
small gap is found" — none was found. `figma-plugin/code.js` was left untouched.)

---

## 2. Schema / emission gap

### 2.1 `src/schema.py` — what exists today

`Layer` (schema.py line 306-350) already has the generic containers needed:

```python
opacity: float = 1.0                          # top-level LAYER opacity (line 315)
fill: Optional[dict] = None                   # {kind:flat|linear|radial,color|stops|angle} (line 325)
effects: list = field(default_factory=list)   # [{type:drop-shadow|blur,...}] (line 332)
```

These are loosely-typed dicts (no strict per-field dataclass), so **no schema.py structural
change is required** — `fill["opacity"]` and an `effects` entry of
`{"type": "background-blur", "radius": N}` are both already legal, already round-trip through
`_ALLOWED_LAYER_KEYS`-style passthrough (build_design_json.py line 21-22 lists `"opacity",
"effects", ..., "fill", "stroke"` as carried-through keys), and are already consumed correctly by
both figma-plugin/code.js (§1) and need to be added to render_preview.py (§3).

**What's actually missing is a documented, exact micro-spec for the fields inside those dicts**,
since nothing currently emits them for the glass case. Proposed (comment-only) additions to the
`Layer.fill` and `Layer.effects` doc comments in schema.py:

```python
# fill: Optional[dict] = None
#   {kind: "flat"|"linear"|"radial", color: "#RRGGBB", opacity: 0..1 (fill-only alpha,
#    distinct from layer opacity — use this for glass/translucent fills so text/stroke
#    on the same node stay fully opaque), stops: [...], angle: deg}
#
# effects: list[dict]
#   [{type: "drop-shadow"|"inner-shadow"|"blur"|"background-blur", radius: px,
#     offset: {x,y} (shadow only), spread: px (shadow only), color: "#RRGGBBAA" (shadow only),
#     opacity: 0..1 (shadow only), visible: bool}]
#   "background-blur" -> figma-plugin/code.js effectFromSpec() maps 1:1 to a native
#   Figma BACKGROUND_BLUR effect (glass/frosted panels). radius is source-image px,
#   NOT scaled. No spread/blendMode field exists for background-blur (Figma has none).
```

This is a **comment/doc-only** proposed diff — no dataclass field changes, since the dict shape
already supports what's needed. (Confirmed no strict validator rejects unknown dict keys: the
"minimal shape/type validation" noted at schema.py line 375-378 validates layer *shape*, not the
contents of `fill`/`effects` dicts.)

### 2.2 `src/build_design_json.py` — emission gap (PROPOSED DIFF, not applied)

**Git recency check:** `git log -1 --format=%cr -- src/build_design_json.py` → **9 hours ago**.
This file is still actively being edited by another agent in this session/timeframe — per the
task instructions, this section is diff-proposal only, not an edit.

Current state: `build_design_json.py` already threads `fill` and `effects` straight through from
`candidate`/`style` (lines 394-402, 710-715, 749-751, 846, 856-865, 940, 958) — i.e. if an
upstream stage (`reconstruct.py`, `element_fusion.py`) ever attaches `fill.opacity` or a
`background-blur` effect to a candidate, it already survives into `design.json` untouched. The
gap is that **no upstream stage currently detects/estimates glass and attaches those fields** —
confirmed by `grep -rn glass|frosted|background_blur src/*.py` returning zero hits outside
`figma_import.py`'s allowlist.

Proposed diff (illustrative — actual glass-detection heuristic belongs in `reconstruct.py` or
`element_detect.py`, which own low-opacity/translucency estimation; this is the
`build_design_json.py`-side hook that must exist for it to reach design.json):

```diff
--- a/src/build_design_json.py
+++ b/src/build_design_json.py
@@ (near line ~394, where `fill = candidate.get("fill")` is resolved)
-    fill = candidate.get("fill")
+    fill = candidate.get("fill")
+    # Glass/translucent shape chips: a detection stage may attach `fill_opacity`
+    # (0..1) and `background_blur_radius` (px) onto the candidate when it estimates
+    # a semi-transparent panel over busy/photo content (see reconstruct.py glass
+    # heuristic). Fold these into the fill dict's own `opacity` (NOT the layer's
+    # top-level opacity — see figma-plugin/code.js applyCommon vs applyFills) and
+    # a background-blur effect entry, respectively.
+    if isinstance(fill, dict) and candidate.get("fill_opacity") is not None:
+        fill = dict(fill)
+        fill["opacity"] = float(candidate["fill_opacity"])
+    glass_blur = candidate.get("background_blur_radius")
```

```diff
@@ (near line ~749-751, where `common["effects"]` is assembled)
     "effects": list(source_effects) if isinstance(source_effects, list) else [],
+    # Glass chips: append a background-blur effect if the candidate carries an
+    # estimated blur radius. Kept additive so existing shadow/blur effects survive.
```
```python
if glass_blur:
    common_effects = list(common.get("effects") or [])
    common_effects.append({
        "type": "background-blur",
        "radius": float(glass_blur),
        "visible": True,
    })
    common["effects"] = common_effects
```

**Estimation approach** (for whichever stage owns detection — likely `reconstruct.py`, which
already estimates drop-shadow opacity from pixel response at line 2077-2079, a nearly identical
kind of estimate): sample the mean/median color and alpha delta of the plate region vs. the
underlying inpainted background plate; a region with low local contrast, elevated luminance
uniformity, and a soft edge halo relative to the sharp photo behind it is the signal for
"detect translucency, don't just treat as solid." Confidence threshold and exact estimator design
is out of scope for this doc — this is the emission *plumbing* spec, not the CV detector.

---

## 3. QA renderer gap — `src/render_preview.py` (PROPOSED DIFF, not applied)

**Git recency check:** `git log -1 --format=%cr -- src/render_preview.py` → **9 hours ago**. Also
actively in-flux; diff proposed only, not applied.

### 3.1 Why this matters

`render_preview.py` is the Python re-render used for structural QA (SSIM/pixel_diff against
`figma_export.png`). Today:

- `_with_effects()` (line 651-681) handles `"blur"`/`"layer-blur"` (blurs the tile's **own**
  content — correct for a layer-blur effect on, say, a blurred photo) and drop/inner shadow.
  It does **not** handle `"background-blur"` at all — that kind string simply falls through
  every `if kind in (...)` check and is silently ignored (line 663, 666: neither branch matches
  `"background-blur"`, so the effect is dropped without even a fidelity warning).
- Consequence: if design.json emits a glass chip as `fill.opacity=0.18` +
  `effects: [{"type":"background-blur","radius":24}]`, `render_preview.py`'s local Python
  render would draw a barely-visible 18%-opacity flat rect (no blur under it) while the real
  Figma export shows a correctly frosted glass panel — a **false SSIM/QA failure** purely because
  the QA renderer doesn't simulate the same effect the plugin now produces. This is exactly the
  failure mode the task description warned about.

### 3.2 Proposed diff

`_with_effects()` operates on the layer's own tile in isolation, before it's composited onto the
canvas — it has no access to "what's behind this layer on the canvas" (that only exists inside
`_blend`/`_prepare_layer_draw`, where `canvas` is available). Background-blur has to be handled
at composite time, not tile-render time. Proposed approach: detect a `background-blur` effect in
`_prepare_layer_draw`/`_draw_layer` and, immediately before compositing, blur the destination
canvas region under the tile's footprint in place, then composite the (translucent) tile on top
normally — this is the standard "blur what's already been painted, then alpha-composite the glass
panel over it" approximation, and matches how Figma's own compositor behaves (blur the
already-rendered backdrop, not the panel's own pixels).

```diff
--- a/src/render_preview.py
+++ b/src/render_preview.py
@@ def _render_tile(layer, run_dir):
-    padded, effect_offset = _with_effects(tile, layer.get("effects") or [])
+    padded, effect_offset = _with_effects(tile, layer.get("effects") or [])
+    # background-blur is NOT applied to the tile itself here (it has no access to the
+    # canvas backdrop) — it's handled at composite time in _draw_layer/_blend below.
     return padded, (effect_offset[0] + text_offset[0], effect_offset[1] + text_offset[1])
```

```diff
--- a/src/render_preview.py
+++ b/src/render_preview.py
@@
+def _background_blur_radius(effects):
+    """Return the largest active background-blur radius on this layer's effects, or 0."""
+    best = 0.0
+    for effect in effects or []:
+        if not isinstance(effect, dict) or effect.get("visible") is False:
+            continue
+        kind = str(effect.get("type", effect.get("kind", ""))).lower().replace("_", "-")
+        if kind != "background-blur":
+            continue
+        best = max(best, max(0.0, _number(effect.get("radius", effect.get("blur", 8)))))
+    return best
+
+
+def _apply_backdrop_blur(canvas, region, radius):
+    """Gaussian-blur an already-composited canvas region in place (glass/frosted panel
+    approximation). `region` is (x0,y0,x1,y1) clamped to canvas bounds."""
+    from PIL import ImageFilter
+    x0, y0, x1, y1 = region
+    if x1 <= x0 or y1 <= y0 or radius <= 0:
+        return
+    patch = canvas.crop((x0, y0, x1, y1))
+    blurred = patch.filter(ImageFilter.GaussianBlur(radius))
+    canvas.paste(blurred, (x0, y0))
+
+
 def _blend(canvas, tile, point, mode, clip_rect=None):
     """Composite tile onto an opaque preview, including common non-normal Figma blends."""
     from PIL import ImageChops
     x, y = point
     if clip_rect is not None:
         tile = _clip_tile_to_rect(tile, x, y, clip_rect)
     x0, y0 = max(0, x), max(0, y)
     x1, y1 = min(canvas.width, x + tile.width), min(canvas.height, y + tile.height)
     if x1 <= x0 or y1 <= y0:
         return
     source = tile.crop((x0 - x, y0 - y, x1 - x, y1 - y))
```

```diff
--- a/src/render_preview.py
+++ b/src/render_preview.py
@@ def _draw_layer(canvas, layer, run_dir, offset=(0, 0)):
 def _draw_layer(canvas, layer, run_dir, offset=(0, 0)):
     tile, x, y, mode, clip = _prepare_layer_draw(layer, run_dir, offset)
+    blur_radius = _background_blur_radius(layer.get("effects") or [])
+    if blur_radius > 0:
+        cx0, cy0 = max(0, x), max(0, y)
+        cx1, cy1 = min(canvas.width, x + tile.width), min(canvas.height, y + tile.height)
+        _apply_backdrop_blur(canvas, (cx0, cy0, cx1, cy1), blur_radius)
     _blend(canvas, tile, (x, y), mode, clip)
```

Notes on the diff:
- Uses `canvas.crop`/`.paste` (not `alpha_composite`) since we're blurring the fully-opaque
  running composite in place, not compositing a new translucent layer.
- Placed right before `_blend` so the blur happens on exactly what's been painted so far
  (photo + any earlier layers), matching Figma's bottom-up compositing order — `design.json`
  layers are drawn in `z_index` order already (confirmed at `_render_tile`'s group-child sort,
  line 695, and the top-level `render()` loop).
- `radius` reuses the exact same px-space convention as the plugin (§1.5) — no scaling needed,
  since `render_preview.py` and `figma-plugin/code.js` both consume design.json's coordinates in
  the same source-image px space per `docs/CONTRACT.md`.
- This only approximates Figma's blur kernel (Figma uses its own GPU blur implementation); exact
  pixel match isn't the goal, SSIM tolerance is — a Gaussian blur of the same radius on the same
  backdrop content gets close enough to avoid the false-fail this doc set out to prevent.

---

## 4. Fallback policy — low detection confidence

Per `docs/CONTRACT.md` rule set (`## Routing rules`, "shape/card/button → Figma primitive
(rect/ellipse with fitted fill) when the fill is solid/gradient", and the "flat chrome = SOLID"
framing in `docs/HARD-CREATIVES-SPEC.md` item 2 — "Rounded-plate overlay family: ... as SOLID
native rects with radius, never raster slices"):

- **When glass detection confidence is high** (clear translucency signal — soft-edged luminance
  plateau, measurable contrast attenuation vs. the photo behind it, consistent with a deliberate
  UI panel): emit the full glass construct — `fill.opacity` (10-30% range per §1.3) +
  `background-blur` effect (§2.2) + optional `strokeHighlight` (thin white ~20-30% opacity inside
  stroke, using the existing `stroke` dict — no new field needed, `applyStrokes` already handles
  it).
- **When confidence is low** (ambiguous whether it's actually translucent vs. just a flat
  light-colored chip, or blur can't be reliably estimated): **do not guess a blur radius and do
  not fall back to a raster slice.** Ship a **solid rect at the estimated average color** of the
  detected region (no opacity trick, no effects) — this is the existing, already-correct
  "shape/card/button... solid fill" contract path and requires zero new plugin/schema work; it's
  editable, matches the routing rule, and degrades gracefully (a solid pastel chip is a much
  smaller visual miss than a raster slice, and stays in-contract per CONTRACT.md rule 8's spirit
  of "don't force uncertain visual structure into a rigid mechanism"). This also sidesteps a
  second failure mode: an incorrectly-estimated blur radius would make the QA renderer's new
  backdrop-blur (§3.2) diverge visibly from Figma's real blur, compounding two guesses. Solid
  average-color is the strictly safer degrade.
- No new field is needed to express this fallback — it's simply "detection stage doesn't set
  `fill_opacity`/`background_blur_radius` on the candidate," and `build_design_json.py`'s existing
  passthrough (unconditional, no glass-specific code path) already produces a plain opaque solid
  fill by default. The fallback is the *absence* of the new optional fields, not a separate code
  path — which is the simplest possible degrade and matches the "populate only when confidence is
  high" pattern schema.py already documents for `sizing` (line 341-344: "Populated only for
  layers that sit inside a real auto-layout stack... by build_design_json's geometry-evidence
  inference" — same idiom, reused for glass).

---

## 5. Summary / confirmation

| Layer | Status | Change needed |
|---|---|---|
| `figma-plugin/code.js` (applyCommon/applyEffects/createShapeLayer, BACKGROUND_BLUR mapping, fill-vs-layer opacity) | **Production-ready today.** Verified line-by-line; no gap found. | None. Left untouched. |
| `src/schema.py` (`Layer.fill`, `Layer.effects`) | Structurally sufficient (loose dicts). | Doc-comment only (proposed, not applied — file not under active edit but change is trivial/low-risk to bundle with build_design_json's owner). |
| `src/build_design_json.py` | **Gap confirmed**: no path emits `fill.opacity` or a `background-blur` effect; nothing upstream currently sets `fill_opacity`/`background_blur_radius` on candidates either. | Proposed diff in §2.2 (not applied — actively edited, 9h old, owned by another agent). |
| `src/render_preview.py` | **Gap confirmed**: `background-blur` effect type silently ignored by `_with_effects`; canvas-backdrop blur has no code path at all. This causes false SSIM/QA fails on any glass element once emission is added. | Proposed diff in §3.2 (not applied — actively edited, 9h old, owned by another agent). |
| Fallback policy | Aligned with existing CONTRACT.md routing rule ("shape/card/button... solid fill") — no new mechanism needed, low-confidence = simply omit the new optional fields. | None (policy note only). |

Net picture: the previously-flagged "critical structural finding" holds — the Figma side is
already fully wired for native glass. The real work is entirely on the Python side: (a) a
detection heuristic (not scoped here) that estimates fill opacity + blur radius for translucent
panels and attaches them to a candidate, (b) the small, additive `build_design_json.py` hook in
§2.2 to fold those into `fill.opacity` / an appended `background-blur` effect, and (c) the
`render_preview.py` backdrop-blur approximation in §3.2 so QA doesn't falsely tank glass elements
once they start being emitted.
