# Peel Decomposition — occlusion-attributed layer completion

Status: implemented and tested; NOT wired into `run_pipeline.py` yet (exact diff in §5,
to be applied by the pipeline owner — peel files must not edit the core pipeline).
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
* peel is *needed* only when some qualifying pair covers a **non-text** under-layer
  (element-over-element). Elements over plain background are already handled by the
  plate; text under-layers stay native.
* No qualifying pair → `peel_scene` returns `skipped=True, skip_reason="no-overlap"`
  and the run continues exactly as today (asserted in tests).

Config (all optional; `enabled` gates the pipeline stage, the module itself ignores it):

```yaml
peel:
  enabled: false            # pipeline integration gate (§5)
  # ── scene mode ──
  min_overlap_area: 64      # px² — gate threshold
  min_overlap_frac: 0.02    # fraction of the smaller element's area
  hole_dilate_px: 2         # anti-alias fringe ring; 0 = exact masks (synthetic tests)
  context_pad_px: 24        # inpaint context crop padding
  min_context_frac: 0.05    # below → unisolated fill + meta.low_context_fill
  text_occluders: box       # box | off
  refine_alpha: false       # BiRefNet cutout-edge refinement (needs matting callable)
  refine_band_px: 3
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
# scene mode over an existing run's artifacts (no models; honors the gate):
.venv\Scripts\python.exe scripts\peel_scene_demo.py --run runs\ad9_regional_final \
    --output runs\peel-scene\ad9            # add --inpaint lama / --force / --reuse-background

# blind LayerD loop on a raw image (downloads ~1 GB BiRefNet on first use):
.venv\Scripts\python.exe scripts\peel_demo.py --input ad.png --output out --device cpu
```

## 5. Integration seam (exact diff, to be applied by the pipeline owner)

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
3. **Still open** (needs the integration applied): move-test A/B (translate a peeled
   under-layer 40 px, count revealed hole pixels vs the ownership-crop baseline; gate
   ≥ 80 % reduction on overlap-heavy fixtures), 16-image benchmark A/B with
   `peel.enabled` on/off watching qa.json SSIM / editable-ratio / ghost-text counts,
   and Crello ground-truth matte quality for the blind mode.

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
  are not detected as elements (they weren't in `ad9_regional_final` — they live in the
  plate), a seam-straddling occluder attributes to *background* and the portraits are
  not separately completed. Correct per the contract, but the leapfrog needs the
  detector to surface large photo panels; revisit the fusion "background plate"
  threshold for before/after archetypes (`archetype.py` already detects comparisons).
* **Box-footprint text occluders overfill** inter-glyph gaps on busy under-layers.
  Bounded by the text box, filled from the layer's own context, and recorded
  (`fills[].text_occluder`); switch to ink-level masks once reconstruct's
  text-removal mattes are exposed pre-merge.
* **Inpaint hallucination inside large peel holes** — same class as gap P0-3. Keep
  Telea/LaMa as the default; Flux only via the §5 router and never for text holes.
* **Fringe ring vs byte-exactness:** `hole_dilate_px: 2` intentionally rewrites a 2 px
  ring inside the under-layer around every hole (kills AA ghosts) — recomposite is then
  exact *except* inside that ring (measured mean 0.17/255 on ad9). Set 0 where masks
  are trusted pixel-exact.
* **Blind-mode matting drift on non-Crello styles** — the fine-tune peels
  "product+shadow" as one layer on photographic ads; repeat-matte guard and coverage
  stops bound the damage.
* **transformers 5.x fallback loader** relies on the checkpoint's `auto_map` and
  `model.safetensors` names; pin an HF revision if CyberAgent restructures the repo.
