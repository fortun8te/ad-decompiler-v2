# Overlay Elements — rounded-plate overlay family (H1-class)

Owner module: `src/overlay_detect.py` (classic CV, CPU-only, no models).
Consumers: `src/element_detect.py` (detection, already wired), `src/element_fusion.py`
+ `src/layout.py` + `src/build_design_json.py` (emission — see integration diffs at bottom).

This documents the **detection proposal** shape and the **emission candidate** shape
that `overlay_detect.emit_overlay_group()` produces, and how each field maps to
`src/schema.py`'s `Layer` dataclass. It covers the H1 pill/banner/card/stadium family
from `docs/HARD-CREATIVES-SPEC.md` (gap #2, #3). The chat/DM/tweet UI family
(H9/H12/H14/H16) is explicitly out of scope for this module.

---

## 1. What an overlay is (contract)

A rounded-plate overlay is a **solid flat-colour rounded rectangle sitting over busy
content** (a photo, a gradient, another plate) that carries copy. Per the project
contract (`docs/CONTRACT.md`): flat chrome is emitted as a **native SOLID rect with a
corner radius**, never a raster slice. Its text is **native TEXT**, and any leading
emoji is an **inline image chip** (the sibling TEXT keeps `letterSpacing == 0` so the
chip owns the spacing).

Four geometric kinds (`overlay_detect._classify`):

| kind      | geometry                                             | H-ref | schema `role` |
|-----------|------------------------------------------------------|-------|---------------|
| `pill`    | wide short rounded bar, aspect ≥ ~2.2                | H1    | overlay-pill  |
| `card`    | blockier rounded panel, aspect < ~2.2                | H4    | overlay-card  |
| `banner`  | spans ≥ 90% of canvas width                          | H3    | banner        |
| `stadium` | capsule: radius == height/2, clearly wider than tall | H5    | stadium-row   |

---

## 2. Detection proposal (`detect_overlays` output)

`detect_overlays(rgb, elements=None, text_lines=None, canvas=None, cfg=None)` returns a
list of proposals, each a plain dict (absolute source pixels), sorted top-to-bottom:

```jsonc
{
  "id": "OV0",
  "bbox": {"x": 130, "y": 300, "w": 460, "h": 110},  // absolute source px
  "kind": "pill",                    // pill | card | banner | stadium
  "corner_radius": 16.0,             // scalar OR {topLeft,topRight,bottomRight,bottomLeft} OR null
  "fill": "#4a7a5c",                 // interior median colour, hex
  "text_ids": ["L0"],               // OCR line ids >= 60% contained in the plate
  "z_order": 20.0,                   // above bg, below its own text
  "fill_ratio": 0.98,                // how much of bbox the plate fills (rounded corners nip rest)
  "interior_uniform": 0.99,          // fraction of interior within color_tol of median
  "source": "overlay-cv"
}
```

`corner_radius` is:
- a **scalar float** when all four corners agree (`estimate_corner_radius` medians them),
- a **per-corner dict** when corners differ (asymmetric radii, e.g. IG bubbles),
- `null` when the mask does not support a clean rounded-rect model (safer to stay a plain rect).

A capsule end (radius reaching the half-side) snaps to the exact pill radius `min(h,w)/2`.

Detection is conservative by design: `require_text=True` gates pills/cards/stadiums on
OCR text containment (a bare decorative panel is left to `element_detect`); only a
full-width banner survives without attributed text.

### Tunables (`cfg["overlay_detect"]`, defaults in `overlay_detect.DEFAULTS`)

`min_area_frac` 0.004 · `max_area_frac` 0.75 · `color_tol` 20 · `min_fill_ratio` 0.80 ·
`min_interior_uniform` 0.82 · `banner_width_frac` 0.90 · `require_text` True ·
`corner_fit_min` 0.93 · `quant_step` 24. Add `"enabled": false` to disable the pass.

---

## 3. Emission candidate (`emit_overlay_group` output)

`emit_overlay_group(plate, texts=None, emojis=None, icons=None)` compiles one proposal
plus its resolved text/emoji/icon candidates into a **`target="group"` candidate** in
absolute coordinates. `build_design_json.build()` compiles it directly; `layout._relativize`
converts children to parent-relative space at the normal stage boundary (like any group).

```jsonc
{
  "id": "OV0",
  "target": "group",                          // -> schema Layer.type "group" (Figma FRAME)
  "box": {"x":130,"y":300,"w":460,"h":110},
  "z_index": 20.0,
  "meta": {
    "role": "overlay-pill",                   // overlay-pill | overlay-card | banner | stadium-row
    "overlay_kind": "pill",
    "z": 20.0,
    "source": "overlay-cv",
    "corner_radius": 16.0,
    "text_ids": ["L0"]
  },
  "children": [
    {                                          // (1) the SOLID rounded-rect surface
      "id": "OV0__plate",
      "target": "shape",                       // -> Layer.type "shape"
      "shape_kind": "rect",                    // -> Layer.shape_kind
      "box": {"x":130,"y":300,"w":460,"h":110},
      "fill": {"kind": "flat", "color": "#4a7a5c"},  // -> Layer.fill (SOLID, never a slice)
      "radius": 16.0,                          // -> Layer.radius (scalar or per-corner dict)
      "z_index": 20.0,
      "meta": {"role": "overlay-plate", "overlay_kind": "pill", "z": 20.0, "source": "overlay-cv"}
    },
    {                                          // (2) native TEXT child(ren)
      "id": "L0",
      "target": "text",                        // -> Layer.type "text"
      "text": "All-Day Weather Hold",
      "box": {"x":170,"y":335,"w":380,"h":40},
      "style": {"fontSize":22,"color":"#ffffff","letterSpacing":0.0},  // letterSpacing forced 0 (contract)
      "z_index": 22.0,
      "meta": {"role": "overlay-text", "z": 22.0}
    },
    {                                          // (3) leading emoji as an inline IMAGE chip
      "id": "OV0__emoji0",
      "target": "image",                       // -> Layer.type "image"
      "src": "emoji/weather.png",              // relative asset path (or null pre-render)
      "box": {"x":140,"y":336,"w":28,"h":28},
      "z_index": 23.0,
      "meta": {"role": "emoji", "z": 23.0, "emoji_chip": true}
    }
    // icons[] (stadium ✓/✗) emit identically with meta.role="icon"
  ]
}
```

### z-order within a group
`plate = base_z (20)` · `text = base_z + 2` · `emoji/icon chips = base_z + 3`.
The plate never covers its own text; chips ride above the text baseline.

### Field → `schema.Layer` mapping (validated: `schema.validate_design(doc) == []`)

| candidate field | schema.Layer field | notes |
|-----------------|--------------------|-------|
| `target`        | `type`             | group / shape / text / image |
| `box`           | `box`              | `{x,y,w,h}`; children relativized by layout |
| `shape_kind`    | `shape_kind`       | `"rect"` for all overlay plates |
| `fill`          | `fill`             | `{kind:"flat", color:"#rrggbb"}` — SOLID |
| `radius`        | `radius`           | scalar float or `{topLeft,…}` dict |
| `style.letterSpacing` | `style.letterSpacing` | forced `0.0` on overlay text (contract) |
| `src`           | `src`              | emoji/icon chip asset path |
| `z_index`       | `z_index`          | float; higher = front |
| `meta.role`     | `meta.role`        | drives layout/QA; not exported to Figma |

No child carries `meta.fallback == "raster-slice"` — the whole point is native emission
(the emission test asserts `schema.fallback_kind(child.meta) is None` for every child).

### `emit_all(plates, texts_by_id, emojis_by_plate, icons_by_plate)`
Convenience wrapper: emits a group for every proposal, selecting each plate's nested
texts by its `text_ids` and its chips from the per-plate maps. Missing mappings are
skipped (a plate with no resolved text still emits its native rect).

---

## 4. Validation results (2026-07-16, CPU / .venv)

`tests/test_overlay_detect.py` (5 tests, all pass): detects all four plate kinds on a
busy-noise composite; recovers radius/fill within tolerance; attributes contained OCR
text to the right plate; and compiles the emitted group through `build_design_json.build`
to a native SOLID rounded RECT + native TEXT + emoji image child with `validate_design == []`.

Ad-hoc validation (`scratch_overlay_val.py`, not committed):
- **Synthetic H1** (three muted-green rounded pills over a noisy vertical gradient,
  720×1280): **3/3 recall @ IoU ≥ 0.85** (actual IoU 0.978–0.991), corner radius 16–19
  vs drawn 16, fill `#4a7a5c` with **0 channel error** vs the drawn green, correct
  `text_ids` attribution per pill.
- **16 real benchmark ads**, no OCR supplied: **0 false-positive pills/cards**; a single
  full-width `banner` fired on `013` (the banner-width exception, which legitimately
  bypasses `require_text`). Confirms the detector does not over-fire on photographic
  content when text containment is absent.

---

## 5. Integration status & handoff

- **Detection**: already wired — `element_detect.detect()` step 8 calls
  `_overlay_candidates()` → `overlay_detect.detect_overlays()` and emits each plate as an
  element with an analytic rounded-rect mask (`_render_plate_mask`) plus
  `meta.overlay_kind/corner_radius/fill/text_ids`. This already satisfies the *native
  SOLID rect* half of the contract; contained text stays as separate native TEXT that
  layout nests by containment.
- **Group emission** (`emit_overlay_group` / `emit_all`): provides the tighter,
  self-contained grouping (rect + text + emoji chips as one FRAME with guaranteed
  z-order). Wiring this through `element_fusion` → `layout` → `build_design_json` as an
  explicit group candidate is the remaining handoff — see the integration diffs in the
  agent report. It is additive: the element-level path above keeps working unchanged if
  the group path is deferred.
