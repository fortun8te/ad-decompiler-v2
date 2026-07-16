# Peel Decomposition — occlusion-attributed layer completion

Status: implemented, tested, WIRED into `run_pipeline.py` (the §5 diff is applied and
verified: `peel` sits in STAGES between `elements` and `merge`, gated on
`peel.enabled`, and merge consumes `peel_layers or qwen`; the stage was exercised live
on a resumed real run). Fill quality validated on real overlap ads (§6: 052
before/after tube, 002 product bundle) — revealed holes confirmed clean by visual
inspection.

**Default `peel.enabled: false`, ONE wiring change from ON.** The gate + granularity
guard (§4a) already make ON safe *semantically* (flat/UI ads skip untouched). What
blocks it is *runtime*: the §5 adapter passes element-class peel holes to the entropy
ladder, which escalates to Flux Fill while SAM3 is still resident (no
`vram.stage_boundary` before peel) → Flux Q6 partially offloads and one call takes
~25 min (observed live on the 052 resume; the run was killed). Fix, in the adapter
(`run_pipeline.py` `_peel_inpaint`, pipeline-owner change): pin ALL peel holes to
LaMa — make the `route_cfg["inpaint"] = {**…, "mode": "lama"}` override unconditional
instead of text-only. LaMa is the validated quality bar (§6; flat-fill absorbs
card/plate holes before the router anyway, and peel then costs seconds, 0 VRAM).
Alternatively add `vram.stage_boundary("elements", "peel", …)` before the stage and
keep Flux — costlier, not needed for the validated quality. After either: flip
`peel.enabled: true`.
Owner artifacts: `src/peel_scene.py` (primary, element-guided), `src/peel_decompose.py`
(blind LayerD loop, standalone/fallback), `scripts/peel_scene_demo.py`,
`scripts/peel_demo.py`, `tests/test_peel_scene.py`, `tests/test_peel_decompose.py`,
this doc.

Fixes gap **P2-7** (docs/RESEARCH-CODIA-GAP-ANALYSIS.md §4, §6c): the pipeline builds
ONE union removal mask and inpaints ONCE, so an element sitting under another element
keeps a hole where its occluder was — moving it in Figma reveals the damage. Codia has
the same limit; nobody has productized peel decomposition (§6c).

## 0. The correctness contract

When a top element sits over MULTIPLE distinct underlying layers, peeling it must
inpaint its footprint ONLY into the specific underlying layer(s) directly beneath it —
and underlying layers that were NOT covered stay pristine.

The canonical case (the before/after ad): a circular product-in-hand element sits at
the SEAM covering part of a LEFT portrait and part of a RIGHT portrait.

* Both portraits come out as COMPLETE full-frame photos; each has a fill exactly where
  the circle actually overlapped it — the LEFT part of the circle's footprint is
  inpainted into the left portrait, the RIGHT part into the right portrait.
* The scene is never treated as one background with one hole.
* No hole is ever punched into a region the occluder never covered.
* A layer with nothing on top of it comes out byte-identical.

`tests/test_peel_scene.py` proves all four properties pixel-exactly on a synthetic
version of exactly this scene (two side-by-side rects + seam-straddling circle + text
block), with sentinel fills so any mis-attributed pixel fails loudly, and a
context-isolation spy proving the left portrait's fill could never even *see* the right
portrait's pixels.

## 1. Two modes

| | `peel_scene.py` (**scene**, pipeline default) | `peel_decompose.py` (**blind**) |
|---|---|---|
| Layer discovery | none — consumes the pipeline's fused elements + z-order | BiRefNet top-layer matting per iteration (LayerD) |
| Footprints | detection masks (fused_elements/*.png) | matting hard mask (`alpha > 0.005`) |
| Hole attribution | per-pixel direct-occluder split (§2) | single plate per iteration |
| Matting model | optional, EDGE refinement only | required, does the peeling |
| Use | pipeline runs (detection already happened) | standalone images, no-detection fallback, research |

The blind loop keeps LayerD's recipe verbatim (thresholds, unblending, stop
conditions, `cyberagent/layerd-birefnet` — see §7). The scene mode replaces the "what
is the top layer?" question (answered better by SAM/fusion than by matting) and keeps
LayerD's *structure*: peel topmost-first, fill what each peel reveals, repeat.

## 2. The occlusion-attributed algorithm (scene mode)

Peel order = reverse z-order (topmost first); each element's footprint is its fused
mask. Computed closed-form (provably equivalent to iterating the peels):

1. **Direct-occluder map.** For every layer L, its occluded region is
   `mask(L) ∩ ⋃ mask(higher)`. Each occluded pixel is attributed to its DIRECT
   occluder — the lowest element above L covering that pixel. (That is exactly what an
   iterative peel produces: peeling top element T attributes each footprint pixel to
   the next-lower owner at that pixel; `attribute_footprint()` exposes that per-peel
   view directly.)
2. **Routing split.** L's occluded region is split into an element-class hole and a
   text-class hole (text occluders → never Flux, see §3), one inpaint call each.
3. **Context isolation.** The inpaint call for L sees ONLY L's visible pixels as
   context: within L's padded bbox crop, every pixel that is not a visible pixel of L
   is part of the inpaint mask. A hole at a seam therefore *cannot* bleed the
   neighbouring layer's colors into L — the filler never observes them. If L is almost
   fully covered (`visible/total < peel.min_context_frac`) isolation is hopeless; the
   fill degrades to unisolated context and the layer is flagged
   `meta.low_context_fill` (honest degradation, never a crash).
4. **Write-back discipline.** Only the occluded region — optionally widened by
   `peel.hole_dilate_px` to kill anti-aliased fringes, but always clipped to
   `mask(L)` — is rewritten. Visible pixels are copied from the flat image
   byte-identical. A region that was never occluded is NEVER inpainted.
5. **Text occluders** (OCR boxes) punch and fill holes in under-layers — leaving glyph
   pixels baked into a completed portrait would double the text once the pipeline
   renders the native TEXT node on top (ghost-text invariant). They are never emitted
   as peel layers. Box footprints overfill slightly (inter-glyph gaps); accepted and
   recorded (`peel.text_occluders: off` disables).
6. **Background plate** is completed the same way (context = pixels no element
   covers), or reused verbatim when the caller passes `background_clean.png`.
7. **Invariant:** re-compositing background + layers back-to-front (+ native text on
   top) reproduces the input **byte-exactly** with `hole_dilate_px: 0`; with the
   default 2 px fringe, every diff pixel lies inside the intentional fringe ring
   (verified on the real ad9 run, §6). `result.meta["recomposite"]` reports it per run.

Output: ordered `ScenePeelLayer` list (back-to-front), each with the fused element
`id`, full-canvas RGBA, tight `bbox`, `z_index`, `occluded_by` / `occludes` id lists,
and per-occluder `fills` (attributed sub-hole areas/bboxes) — plus a manifest
(`write_outputs`) and the merge-ready form (`write_pipeline_layers`).

## 3. API

```python
from src import peel_scene

elements = peel_scene.elements_from_run(run_dir, fused_elements, canvas, cfg, ocr=ocr)
report   = peel_scene.overlap_report(elements, cfg)      # the gate (§4)
result   = peel_scene.peel_scene(
    norm_path, elements,
    inpaint=my_router,          # None → OpenCV Telea
    cfg=cfg,
    background=None,            # or run_dir/background_clean.png to reuse the plate
    matting=None,               # optional BiRefNet callable for EDGE refinement only
)
result.skipped                  # True → keep the single-plate path (gate said no)
result.layers                   # list[ScenePeelLayer], back-to-front, COMPLETE RGBAs
result.background               # completed plate
result.meta["recomposite"]      # {"max_abs_diff", "mean_abs_diff", "exact", ...}

peel_scene.write_outputs(result, out_dir)           # PNGs + peel_scene_manifest.json
peel_scene.write_pipeline_layers(result, run_dir)   # peel_layers/P<i>.png, QwenLayer shape
```

### The pluggable inpaint interface

`peel_scene` (and `peel_decompose`) never import `src/inpaint.py` or
`src/reconstruct.py` — the hole filler is an injected callable, resolved once:

```python
inpaint(rgb: HxWx3 uint8, mask: HxW bool) -> HxWx3 uint8              # minimal form
inpaint(rgb: HxWx3 uint8, mask: HxW bool, meta: dict) -> HxWx3 uint8  # router form
```

* Only `mask` pixels may be treated as unknown/rewritten. The caller copies back an
  even smaller region (the attributed hole), so a conservative filler is always safe.
* `meta` (passed iff the callable's signature accepts `meta` or `**kwargs`):
  `{"under_id", "under_kind", "occluder_ids", "text_occluder": bool,
    "isolated_context": bool}`.
* Defaults: `peel_decompose.opencv_inpaint` (deterministic Telea, test-safe) and
  `peel_decompose.make_simple_lama_inpaint()` (Big-LaMa, CPU by default).
* **Routing note (text):** holes whose occluders are text (`meta["text_occluder"]`)
  must NOT be routed to Flux — generative fills leave glyph residue. Route them to
  Big-LaMa/OpenCV. Peel holes in general are layer-shaped and large; keep Telea/LaMa
  as the default class and treat Flux as per-region opt-in (same failure class as gap
  P0-3). The pipeline adapter in §5 implements exactly this.

### Matting = edge refinement only

`refine_element_alpha(image, element, matting, cfg)` consults BiRefNet ONLY inside a
±`peel.refine_band_px` ring around the detection mask's boundary: interior stays
opaque, exterior stays transparent — the model can soften a cutout edge but can never
grow, shrink, or re-detect a layer (tested against an adversarial all-foreground
matting). Hole geometry always uses the hard detection mask. Off by default
(`peel.refine_alpha: false`); enable together with `hole_dilate_px ≥ 2`.

## 4. When peel runs (selective-use policy)

Peel is expensive relative to the single-plate path and can only help where elements
genuinely overlap. Ownership map + single-union inpaint (current, stays default) is
correct and cheap when nothing overlaps: every foreground pixel has exactly one owner.
AGENTS.md invariants ("one final removal mask", "one canonical owner") remain the law
for that case — peel adds *completed under-layers* on top, it does not replace the
canonical plate/ownership artifacts.

The gate is `overlap_report()`:

* a pair qualifies when `intersection ≥ peel.min_overlap_area` px AND
  `intersection ≥ peel.min_overlap_frac ×` the smaller element's area;
* peel is *needed* only when some qualifying pair is **element-over-element** — BOTH
  members non-text (text-over-element alone never activates; text stays native, and
  once peel runs for a real element pair the text holes get filled as a bonus), AND
  both members pass the granularity guard (§4a). Elements over plain background are
  already handled by the plate — with one exception: a **printed-lockup product**
  (fusion absorbed on-product label artwork into the cutout and set
  `meta.printed_lockup`, e.g. 013's grüns bag) lifts off the plate even without an
  object-over-object pair (`lifted_products` in the report). The single-plate path
  left seams/haze around such hero products; peel punches them from the plate and
  completes them, while the ink discipline (§4b) still keeps their printed
  text/artwork unpunched. The lift is still gated by §4a eligibility.
* No qualifying pair → `peel_scene` returns `skipped=True, skip_reason="no-overlap"`;
  qualifying pairs blocked only by the guard → `skip_reason="no-eligible-overlap: <id>:
  <why>"`. Either way the run continues exactly as today (asserted in tests).

### 4a. Detection-granularity guard (`element_eligibility`)

Peel is only as good as the elements it is fed. Fusion sometimes emits **residual**
masks — "photo panel minus persons minus product", a swiss-cheese of hundreds of
specks (052's E000: largest connected component 57% of the mask across 402 pieces).
Such a mask can neither be a trustworthy occluder footprint nor provide usable inpaint
context, so a pair involving one must not switch peel on. `mask_integrity()` computes
per element:

* `cc_frac` — largest connected component / mask area (`peel.min_cc_frac`, 0.80);
* `hole_frac` — interior holes / (mask + holes) (`peel.max_hole_frac`, 0.25).

Measured separation on real runs: genuine cutouts (persons, products, cards, icons)
score `cc_frac ≥ 0.996` with 1–16 components; the residual fragment scored 0.572 with
402 — the guard is nowhere near its own margins. Text and `background`/`plate` kinds
are never eligible pair members. `require_eligible: false` disables the guard for
research runs. **What detection must surface for peel to help more:** large photo
panels as distinct solid elements (e.g. the two before/after portraits of 052 live in
a fragmented leftover today, so a seam-straddling product attributes to that
fragment/background and the panels are not separately completed — correct per the
contract, but the win waits on panel-level detection).

### 4b. Fill quality (what makes the revealed hole clean)

**Strategy ladder (archetype-aware — not just mask tweaks):**

| Hole class | `product_on_flat` / UI / `social_screenshot` | `lifestyle_overlay` / photo |
|---|---|---|
| Shape/card/panel under objects | **Solid/analytic** ring median | Solid if ring flat, else LaMa |
| Background plate | **Solid** per-CC when ring agrees (orange/white chrome) | LaMa (or abandon) |
| Photo/product under | Abandon if hole ≫ mask; else LaMa | Same |
| Text occluder on plate | Solid when ring flat | LaMa / Telea |
| Text/logo on product cutout | **Do not punch** (printed ink stays) | Same |
| Flux | **Never** at peel (VRAM + smear) | Never |

Fail-closed: after LaMa/Telea, if the fill's mean deviation from the solid ring
candidate exceeds `peel.fail_closed_residue`, keep the solid plate instead of smear.

* **Shadow blinding** (`context_shadow_px`, 12): occluder drop shadows / AA halos live
  just OUTSIDE the detection mask, survive into the visible-context ring, and smear
  gray into any inpaint (the 002 gray-gradient ghosts). The band around the hole is
  masked unknown for the fill call — but never written back.
* **Robust flat-fill** (`flat_fill_tol`, 8): sample a ring beyond the shadow band; if
  ≥ `flat_fill_inlier_frac` of ring pixels sit within ±tol of the ring median, the
  surface is flat — fill with the inlier median. Thin-rim guard
  (`flat_fill_min_visible_frac`) blocks the 016 "paint whole plate beige" failure.
  Background flat-fill is enabled for flat-plate archetypes (or
  `flat_fill_allow_background: true`).
* **Fragmentation guard**: `min_cc_frac` (0.85) + `max_components` (24) — residual
  swiss-cheese masks cannot activate peel.
* **Peel objects only**: logos/wordmarks do not activate the gate; OCR/artwork do not
  punch product cutouts.
* **z-order role bands** (`_band_of`): fusion tags product/person cutouts
  `kind="photo-fragment"` (band 5) while the detector role-tags them `product`/`person`
  (band 20); the band is the MAX of both.
* **LaMa / Telea, never Flux**: pipeline `_peel_inpaint` pins large holes to LaMa and
  tiny/text holes to Telea.

Config (all optional; `enabled` gates the pipeline stage, the module itself ignores it):

```yaml
peel:
  enabled: false            # pipeline integration gate (§5). Flip to true after the
                            # adapter LaMa pin (see Status) — the overlap gate + the
                            # granularity guard then make it a conservative ON where
                            # only genuine element-over-element scenes actually peel
  # ── scene mode ──
  min_overlap_area: 400     # px² — gate threshold
  min_overlap_frac: 0.02    # fraction of the smaller element's area
  hole_dilate_px: 2         # anti-alias fringe ring; 0 = exact masks (synthetic tests)
  context_pad_px: 24        # inpaint context crop padding
  min_context_frac: 0.05    # below → unisolated fill + meta.low_context_fill
  text_occluders: box       # box | off
  refine_alpha: false       # BiRefNet cutout-edge refinement (needs matting callable)
  refine_band_px: 3
  # ── detection-granularity guard (§4a) ──
  require_eligible: true
  min_cc_frac: 0.80
  max_hole_frac: 0.25
  # ── fill quality (§4b) ──
  context_shadow_px: 12
  flat_fill_tol: 8.0        # 0 disables the flat-fill fast path
  flat_fill_inlier_frac: 0.60
  flat_fill_ring_px: 16
  flat_fill_min_px: 40
  # ── blind mode (LayerD defaults, see §7) ──
  max_layers: 3
  alpha_threshold: 0.005
  full_coverage_stop: 0.99
  min_coverage_stop: 0.0
  kernel_scale: 0.015
  unblend: true
  repeat_iou_stop: 0.95
  matting:
    backend: auto           # auto | birefnet | rembg
    hf_card: cyberagent/layerd-birefnet
    fallback_hf_card: ZhengPeng7/BiRefNet
    device: cpu             # cuda only when the GPU is actually free
    process_size: null
```

Demos:

```bash
# scene mode over an existing run's artifacts (honors the gate; --inpaint auto default
# uses Big-LaMa CPU when importable, else Telea; prints gate + eligibility verdicts):
.venv\Scripts\python.exe scripts\peel_scene_demo.py --run runs\parity-v2-052 \
    --output runs\peel-scene\052            # add --inpaint opencv / --force / --reuse-background

# blind LayerD loop on a raw image (downloads ~1 GB BiRefNet on first use):
.venv\Scripts\python.exe scripts\peel_demo.py --input ad.png --output out --device cpu
```

## 5. Integration seam (APPLIED — kept as the wiring reference)

The diff below is live in `run_pipeline.py` (import at line ~18, `"peel"` in STAGES,
stage body after the elements load, merge fed `peel_layers or qwen`) — verified
end-to-end with `--resume peel` on a real run. Peel failures degrade to
`report.stage("peel", "fallback", ...)` and never abort the run.

Peel becomes an optional stage between `elements` and `merge`, feeding
`merge_layers.merge` decomposed layers in the exact shape qwen layers already use
(`schema.py` `QwenLayer`: `{"id","png","box","kind_hint"}`, back-to-front; peel adds a
harmless extra `fused_id` audit key). `merge_layers.merge` IoU-matches each layer to a
fused element candidate and sets `best["src"] = layer["png"]`; `reconstruct._source_rgba`
already prefers that clean RGBA over cutting a crop from the flattened source. **No
merge/reconstruct code changes are needed** — that consumption path is live today for
qwen layers, and peel masks are the elements' own masks so every IoU match is ~1.0.

`run_pipeline.py`, hunk 1 — import (line 17–18):

```diff
 from src import (normalize, ocr, text_analysis, element_detect, sam3_detect,
-                 element_fusion, qwen_worker, merge_layers, reconstruct, layout,
+                 element_fusion, qwen_worker, peel_scene, merge_layers, reconstruct, layout,
```

Hunk 2 — STAGES (line 32):

```diff
 STAGES = ["normalize", "ocr", "text", "residual", "qwen", "sam", "elements",
-          "merge", "structure", "reconstruct", "layout", "design", "preview", "figma",
+          "peel", "merge", "structure", "reconstruct", "layout", "design", "preview", "figma",
           "export", "diff", "qa"]
```

Hunk 3 — stage body, inserted after `els = load(A("elements.json")) …` (line 390) and
before the merge stage comment (line 392):

```python
        # 5b OPTIONAL occlusion-attributed peel (docs/PEEL-DECOMPOSITION.md).
        # Completes layers that sit UNDER other layers so they stay whole when moved.
        # Gated: runs only when fused elements genuinely overlap; a peel failure
        # degrades with a note and must never abort the run.
        peel_layers = []
        if (cfg.get("peel") or {}).get("enabled"):
            if stage("peel") or dirty or not exists("peel.json"):
                current_stage = "peel"
                dirty = True
                try:
                    scene_elements = peel_scene.elements_from_run(
                        run_dir, els, canvas, cfg=cfg, ocr=ocr_res)

                    def _peel_inpaint(rgb, mask, meta=None):
                        # Route through the pipeline's entropy-routed ladder; text
                        # holes are pinned to LaMa/OpenCV (Flux leaves glyph residue).
                        from src import inpaint as inpaint_mod
                        route_cfg = cfg
                        if (meta or {}).get("text_occluder"):
                            route_cfg = dict(cfg)
                            route_cfg["inpaint"] = {**(cfg.get("inpaint") or {}),
                                                    "mode": "lama"}
                        out, _backend, _diag = inpaint_mod.inpaint_array(
                            rgb, mask.astype("uint8") * 255, route_cfg,
                            return_diagnostics=True)
                        return out

                    result = peel_scene.peel_scene(
                        norm_path, scene_elements, inpaint=_peel_inpaint, cfg=cfg)
                    if result.skipped:
                        dump([], A("peel.json"))
                        report.stage("peel", "ok",
                                     detail=f"skipped: {result.skip_reason}")
                        _log(run_dir, f"peel → skipped ({result.skip_reason})")
                    else:
                        peel_scene.write_outputs(result, os.path.join(run_dir, "peel"))
                        dump(peel_scene.write_pipeline_layers(result, run_dir),
                             A("peel.json"))
                        rc = result.meta.get("recomposite") or {}
                        report.stage("peel", "ok",
                                     detail=f"{len(result.layers)} layers, "
                                            f"recomposite max_diff={rc.get('max_abs_diff')}",
                                     artifacts=["peel.json", "peel_layers",
                                                "peel/peel_scene_manifest.json"])
                        _log(run_dir, f"peel → {len(result.layers)} complete layers")
                except Exception as exc:
                    dump([], A("peel.json"))
                    report.stage("peel", "fallback", detail=f"peel failed: {exc}")
                    _log(run_dir, f"peel fallback → {exc}")
            peel_layers = load(A("peel.json"))
```

Hunk 4 — feed merge (line 397). Peel and qwen answer the same question ("what are the
clean RGBA layers?"); peel is primary, qwen the second opinion only when peel produced
nothing (concatenating both would double-claim candidates):

```diff
-            merged = merge_layers.merge(ocr_res, els, qwen, canvas, cfg, run_dir=run_dir)
+            merged = merge_layers.merge(ocr_res, els, peel_layers or qwen, canvas, cfg,
+                                        run_dir=run_dir)
```

(`config.yaml` has `qwen.enabled: false` today, so peel simply fills the currently
empty decomposed-layer slot.)

Config: add the §4 `peel:` block to `config.yaml` / `config.example.yaml` with
`enabled: false`.

Notes for the applier:

* `inpaint_array(..., return_diagnostics=True)` returns `(image, backend, diagnostics)`;
  without the flag it returns `(image, backend)`. Either arity works for the adapter —
  keep whichever matches the local call style.
* `_peel_inpaint` accepts `meta`, so `peel_scene` auto-detects the router form.
* Resume semantics follow the stage convention: `--resume peel` re-runs from peel.
* If BiRefNet edge refinement is later enabled (`peel.refine_alpha: true`), build the
  callable via `peel_decompose.resolve_matting(cfg)` inside the stage and pass it as
  `matting=`; add `vram.stage_boundary("elements", "peel", …)` before it when
  `peel.matting.device == "cuda"` (CPU needs no boundary).

## 5c. H7/H13 make-or-break: text directly on a busy/dark photo (2026-07-16)

The user-flagged make-or-break case is **text sitting directly on a busy or DARK photo
with no backing plate** (H7 sleep-mask, H13 dark-towel callouts). The risk is subtle: a
locally-flat ring on a dark photo passes the solid-median flat-fill test and leaves a
flat painted patch under the peeled headline — a visible smear. The fix
(`background_plate_kind`, §5 diff) classifies the background plate as `photo` vs
`background` by median windowed stddev over the visible plate; a `photo` plate denies
the solid path (`_flat_fill_allowed` returns False for photo kinds) and routes every
hole — text and element — to the injected inpainter (LaMa/Flux, never a solid patch).

**Proven this session (ground-truth, exact):** `tests/test_peel_scene.py` adds three
H7 tests:

* `test_background_plate_kind_flags_photo_and_rejects_flat_chrome` — flat gray and
  near-black FLAT chrome → `background`; textured photo AND **dark textured** photo →
  `photo` (the sleep-mask risk).
* `test_flat_fill_is_denied_on_photo_under_kind` — the routing guard: no solid patch may
  land on a `photo` plate regardless of policy/caps, while the same hole on a `background`
  plate stays solid-eligible (proves the gate is kind-driven).
* `test_h7_text_on_photo_plate_routes_to_inpaint_not_solid` — full `peel_scene` on a
  photographic background + an activating card/badge element pair + a headline sitting
  directly on the photo. Asserts: not skipped, `plate_kind == "photo"`, the text-class
  background hole routes to `inpaint` (**zero solid backends**), the inpaint call carries
  `under_kind="photo", background=True, text_occluder=True`, background pixels outside the
  union-of-footprints are **byte-identical** to the input, and recomposite is **exact**
  (`max_abs_diff=0`) with the text footprint excluded (native text renders on top — no
  ghost text baked into the plate).

**Measured limitation (honest):** the stddev classifier reads a *heavily* Gaussian-blurred
smooth dark region (σ≈3 blur of noise) as `background`, because it has no local texture.
Real photos (JPEG grain, edges, gradients) clear the σ=7 threshold comfortably; a fully
out-of-focus dark backdrop is the residual edge case. Not re-tuned — lowering the
threshold would misclassify subtle-vignette flat chrome as photo. `flat_fill_tol` bounds
the downside if it ever misfires.

**Runtime safety (the original enabled blocker) is resolved.** `run_pipeline.py`
`_peel_inpaint` is now a ladder: text holes → Telea (LaMa when the under-layer is a
photo), small non-photo holes → Telea, photo-band holes → Flux **only after
`vram.ensure_flux_vram` evicts SAM**. Plus the GB1 fix in `src/qwen_worker.py`
(`_comfy_abort`): every early-return-after-`/prompt` path in `flux_inpaint` (timeout,
no-images, exception) now POSTs `/interrupt` and clears `/queue`, so a stalled Flux job
can no longer pin ~16 GB and wedge the next run's `/prompt`
(`tests/test_qwen_worker.py::test_flux_inpaint_timeout_interrupts_and_clears_queue`).

**peel.enabled decision: KEEP `true`.** Rationale: the overlap gate + granularity guard
are conservative — only genuine element-over-element pairs with solid (non-fragmented)
members activate, so flat/UI/most photo-background ads skip untouched (re-confirmed on
052: `needed=False`, all 5 pairs blocked against the 402-component E000 residual). The H7
photo-plate routing is now proven exact on ground truth, and the two runtime hazards
(Flux-while-SAM-resident, wedged-Flux-job) are both closed. **Still unproven this
session:** a full real-pipeline GPU run exercising the Flux photo-band at peel time (the
GB1 guard is tested in isolation, not against a live wedged ComfyUI); and 052-class
before/after panels still can't peel until detection surfaces the two portraits as
distinct solid elements (§9 — the standing detection-granularity blocker, not a peel bug).

## 6. Validation

1. **Synthetic before/after proof** (`tests/test_peel_scene.py`, 17 tests): the seam
   scene from §0 with sentinel fills — footprint split exact to the pixel, portraits
   byte-identical outside their true holes, untouched circle byte-identical, per-layer
   context isolation (left fill sees only blue, right only green, background only bg),
   text/element routing split, recomposite == input exactly, occlusion metadata,
   gate on/off, honest low-context degradation, adversarial-matting edge refinement,
   loader + manifests. Plus 14 blind-loop tests (`tests/test_peel_decompose.py`).
2. **Real run** (`runs/peel-scene/ad9`, from `runs/ad9_regional_final`, 1080×1080,
   21 fused elements + 16 OCR text occluders, CPU Telea): gate found 11 qualifying
   element-over-element pairs; 21 complete layers in **2.3 s**; every attributed fill
   lands under a real occluder (e.g. `E006` completed under `E005` +
   `text_L1`, nested icons `E011/E013/E015…` completing their buttons). Recomposite:
   byte-exact (`max_abs_diff=0`) with `hole_dilate_px: 0`; with the default 2 px
   fringe, **all** diff pixels lie inside the intentional fringe ring
   (17 105 px, mean 0.17/255).
3. **Real overlap ads, visual inspection of revealed holes** (runs/peel-validate/*):
   * **052 before/after** (`runs/parity-v2-052`, comparison_grid): guard blocked all 4
     pairs against the fragmented E000 residual (57 % / 402 pieces) and activated on
     E004 (logo) over E003 (product tube). LaMa + shadow blinding: the tube under the
     lifted "wavy" script logo reads as an intact tube — no smearing, no glyph
     residue, only a slightly flatter green where the specular gradient was
     synthesized. Recomposite `max_abs_diff=0` with `hole_dilate_px: 0` (byte-exact);
     the default 2 px fringe accounts for the entire diff otherwise.
   * **002 product bundle** (3 products over a white card): role bands fixed the
     inverted z (card had outranked its products); the card completes under all three
     product footprints. Robust flat-fill goes `solid` on the card — pure clean white
     where each product was, where plain LaMa had left gray-gradient shadow ghosts and
     Telea had smeared. The remaining dark sliver between the two tubs is ORIGINAL
     card pixels (their contact shadow, byte-preserved) — not a fill artifact.
   * **Negative case honestly reported:** 052's seam-straddling tube itself attributes
     mostly to background/fragment because the two portrait panels are NOT distinct
     detected elements (§4a) — the guard correctly refuses those pairs instead of
     producing the smears the earlier stale-run test showed.
4. **Still open**: move-test A/B (translate a peeled under-layer 40 px, count revealed
   hole pixels vs the ownership-crop baseline; gate ≥ 80 % reduction on overlap-heavy
   fixtures), 16-image benchmark A/B with `peel.enabled` on/off watching qa.json SSIM /
   editable-ratio / ghost-text counts, and Crello ground-truth matte quality for the
   blind mode.

## 7. Blind mode: LayerD recipe (unchanged)

Adopted verbatim from LayerD (CyberAgent, ICCV 2025, Apache-2.0,
github.com/CyberAgentAILab/LayerD, `src/layerd/models/layerd.py`):

| Piece | Value | Their name |
|---|---|---|
| hard-mask threshold | `alpha > 0.005` | `_th_alpha` |
| stop: no content | hard mask empty | `hard_mask.sum() == 0` |
| stop: no separable top layer | coverage `> 0.99` | `np.mean(hard_mask) > 0.99` |
| inpaint-mask dilation | kernel `round(dim * 0.015)` | `kernel_scale` |
| unblending | `fg = (img − (1−a)·bg) / a`, alpha snapped outside `[0, 0.95]` | `use_unblend`, `_unblend_alpha_clip` |
| iteration cap | 3 | `decompose(max_iterations=3)` |
| output order | background first, then back-to-front | our `PeelResult.stack()` |
| matting model | `cyberagent/layerd-birefnet` (top-layer fine-tune, Crello) | — |

Added beyond LayerD: repeat-matte guard (`repeat_iou_stop: 0.95`), residual floor
(`min_coverage_stop`), transformers 5.x manual-safetensors fallback loader
(`_load_birefnet_model`; 0 missing / 0 unexpected keys verified on this machine), and
CPU-default devices everywhere (SimpleLama/BiRefNet silently grab CUDA otherwise; the
RTX 5080 is contended by the main pipeline).

Deliberately not adopted: LayerD's `fg_refine`/`bg_refine` palette snapping — our
reconstruct stage owns color/style extraction and the palette snap would fight the
entropy-routed inpaint ladder.

## 8. Dependencies & VRAM

Already in the venv: torch 2.10+cu128, torchvision, transformers 5.13, timm, einops,
opencv, simple-lama-inpainting. **Scene mode needs none of the model deps** — it runs
on numpy/opencv alone with Telea (2–3 s per 1080² ad, 0 VRAM). Optional extras:

* `kornia` — imported by the BiRefNet HF remote code (blind mode / edge refinement).
  `pip install kornia` (0.8.3, depends only on torch; verified compatible in isolation).
* `rembg[cpu]>=2.0.61` — alternative ONNX BiRefNet backend.
* Weights: `cyberagent/layerd-birefnet` ≈ 0.9 GB safetensors, HF-cached on first use.

| Mode | Cost | Notes |
|---|---|---|
| scene + Telea (default) | 0 VRAM, ~2–3 s / 1080² ad | recommended; deterministic |
| scene + SimpleLama CPU | 0 VRAM, ~2–5 s per hole class per layer | quality on textured layers |
| scene + pipeline ladder (§5 adapter) | whatever inpaint_array routes | text pinned to LaMa |
| BiRefNet CPU edge refinement | 0 VRAM, ~10 s per matte call | crop-local, one call per refined element |
| BiRefNet CUDA fp32 | ~1 GB weights + ~1.5–2 GB activations | fits alongside SAM3, NOT alongside Flux Q6 + t5xxl; evict via `vram.stage_boundary` |

## 9. Known risks

* **Detection granularity bounds peel granularity (scene mode).** If the two portraits
  are not detected as elements (they weren't in `ad9_regional_final` OR
  `parity-v2-052` — they live in the plate / a fragmented residual), a seam-straddling
  occluder attributes to *background* and the portraits are not separately completed.
  Correct per the contract, and since §4a the guard SKIPS rather than peeling against
  residual fragments — but the leapfrog needs the detector to surface large photo
  panels as distinct solid elements; revisit the fusion "background plate" threshold
  for before/after archetypes (`archetype.py` already detects comparisons).
* **Flat-fill flattens genuine subtle texture** when a surface passes the inlier test
  (paper grain, soft vignettes). Bounded by `flat_fill_tol` (max-channel deviation 8)
  and only inside true holes; drop `flat_fill_tol` to 0 to force the router
  everywhere.
* **Enabled-by-default blast radius**: with `peel.enabled: true`, merge prefers peel
  layers over qwen layers whenever peel produced any (`peel_layers or qwen`);
  `qwen.enabled: false` today so peel simply fills an empty slot, but re-enabling qwen
  changes precedence — revisit then.
* **Box-footprint text occluders overfill** inter-glyph gaps on busy under-layers.
  Bounded by the text box, filled from the layer's own context, and recorded
  (`fills[].text_occluder`); switch to ink-level masks once reconstruct's
  text-removal mattes are exposed pre-merge.
* **Inpaint hallucination inside large peel holes** — same class as gap P0-3. Keep
  Telea/LaMa as the default; Flux only via the §5 router and never for text holes.
* **Peel-stage runtime when the ladder escalates to Flux.** The §5 adapter routes
  element-class peel holes through `inpaint_array`'s entropy ladder; on overlap-heavy
  ads that can mean several Flux Fill calls (minutes each) at peel time. The robust
  flat-fill absorbs card/plate holes before the router, but if wall-clock matters,
  pin peel element holes to LaMa in the adapter exactly like the text pin
  (`route_cfg["inpaint"]["mode"] = "lama"` unconditionally) — demo validation (§6)
  shows LaMa quality is already the accepted bar. Pipeline-owner decision.
* **Fringe ring vs byte-exactness:** `hole_dilate_px: 2` intentionally rewrites a 2 px
  ring inside the under-layer around every hole (kills AA ghosts) — recomposite is then
  exact *except* inside that ring (measured mean 0.17/255 on ad9). Set 0 where masks
  are trusted pixel-exact.
* **Blind-mode matting drift on non-Crello styles** — the fine-tune peels
  "product+shadow" as one layer on photographic ads; repeat-matte guard and coverage
  stops bound the damage.
* **transformers 5.x fallback loader** relies on the checkpoint's `auto_map` and
  `model.safetensors` names; pin an HF revision if CyberAgent restructures the repo.

## §10 Top-down peel discipline (external survey, 2026-07-16 — user-supplied)

Every serious layered-decomposition system (LayerD ICCV 2025: BiRefNet matte → LaMa →
iterate ~3x with flat-color refinement for design graphics; Inpaint-Anything: SAM mask →
dilate once → LaMa once; Qwen-Image-Layered; Plakat/Playyy) converges on the same
skeleton we have — the differences are discipline, not architecture:

1. **Peel unoccluded TOP layers only, iterate (~3 max)** — never punch every SAM
   fragment in one pass. Underlying layers become peelable on the next iteration's plate.
2. **LaMa/flat/analytic for design plates; Flux for photos ONLY** — LayerD literally
   applies a flat-color assumption to design graphics. (Our gradient/solid routing and
   non-photo→opencv routing implement this; the Flux budget cap enforces it.)
3. **Text is a PARALLEL track (OCR → native), never a peel target** — nobody inpaints
   text away before extracting it. Card/checklist interiors never enter a peel mask
   until their native text owners exist.
4. **Dilate once, inpaint once per layer** — 013's repeated near-identical full-canvas
   holes and 60%+ plate destruction are the anti-pattern.
5. **Budget caps as first-class limits**: iterations, per-run Flux call count, per-hole
   and per-iteration mask area.
6. Later option: LayerD's matting weights (BiRefNet-class) as a peel-matte upgrade —
   not a Qwen-layers revival (rejected for speed; qwen stage stays disabled).

### §10a Implementation (2026-07-16, src/peel_scene.py)

All five rules are live; the closed-form attribution is kept, the discipline is a
pre-pass (`plan_peel_iterations`) + per-call accounting:

1. **Top-down iteration** — `occlusion_levels()` computes each non-text element's
   stack depth (0 = unoccluded top-of-stack); `plan_peel_iterations()` refuses
   elements at depth ≥ `peel.max_iterations` (3). A refused element is
   *plate-committed*: never punched, never completed, never emitted — it dissolves
   into the plate and kept occluders' holes over it re-attribute to the background
   (exactly what an iterative peel that never reaches it produces).
2. **Per-run Flux budget** — `peel_scene` seeds a shared mutable
   `meta["flux_state"]`; `peel_inpaint_mode` → `_flux_budget_admits` spends it
   (`peel.flux_budget`, 4/run) and applies the per-hole canvas-fraction cap
   (`peel.flux_max_hole_frac`, 0.25); overflow falls through to LaMa/Telea/flat.
3. **Text parallel track** — `peel.text_parallel_track` (default ON): OCR ink
   never enters a peel punch mask, for under-layers AND the plate (structural
   `continue` on the text class). Text boxes stay in the occluder map, so they
   still blind the inpaint context; native TEXT is extracted downstream.
4. **Dilate once, inpaint once** — per-surface `done` masks: every write mask is
   `write & ~done` before filling; overlapping dilation rings can never re-inpaint
   a region (`meta.reinpaint_blocked` counts collapsed jobs).
5. **Mask-area budgets** — `peel.iter_mask_budget_frac` (0.45 of canvas per
   iteration, smaller footprints admitted first), `peel.max_punch_canvas_frac`
   (0.30 single-element cap), and the plate-band rule
   (`plate_band_span_frac` 0.95 / `plate_band_min_area_frac` 0.05): a full-span
   background stratum (013's E003/E008/E013/E000 bands) stays in the plate —
   peeling the plate out of itself was 013's 56.3% plate-destruction hard fail.
   `result.meta["iteration_plan"]` + `plate_punched_canvas_frac` feed the
   changed-canvas accounting.

Module defaults keep the synthetic contract tests exact (budgets/bands off,
`max_iterations` 3, text track ON); config.yaml enables the full discipline for
pipeline runs. Tests: `tests/test_peel_scene.py` "§10 top-down peel discipline"
section.
