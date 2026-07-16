# ad-decompiler — stage contract (build against this)

All stages read/write plain dicts matching `src/schema.py`. Pixels = source-image px,
origin top-left. The orchestrating agent inspects and repairs over these JSON files.

## Pipeline dataflow
```
run_pipeline.py --input ad.png --output ./runs/run_001
  1 normalize.load_normalize(input, run_dir)        -> normalized.png, {w,h}
  2 ocr.run_ocr(normalized, cfg)                     -> ocr_raw.json
  3 text_analysis.analyze_text(normalized, raw_ocr)  -> ocr.json (ink geometry/styles/blocks)
  4 element_detect.detect(normalized, ocr, cfg)      -> residual.json + elements/*.png
  5 qwen_worker.propose_layers(normalized, cfg)      -> qwen.json (optional advisory observations)
  6 sam3_detect.detect(normalized, residual, cfg)    -> sam3.json + sam3_masks/*.png
  7 element_fusion.fuse(sam3,residual,qwen,...)      -> fused_elements.json + canonical masks
  8 merge_layers.merge(...) + routing.route(...)     -> merged.json
  9 scene_intent.plan(merged,...)                    -> scene_intent.json (frozen hierarchy/layout)
 10 reconstruct.reconstruct(merged,...)              -> ownership/removal/background/assets
 11 scene_intent.hydrate(intent,reconstruction)      -> layout.json (nested scene graph)
 12 build_design_json.build(...)                     -> design.json v2 [SOURCE OF TRUTH]
 13 figma_import/import plugin                       -> native Figma nodes + figma_export.png
 14 pixel_diff + structural QA                       -> diff.png + qa.json
```

## Routing rules (routing.py — the crown jewel, ported from the validated Node harness)
- **text** → editable Figma TEXT. Never vectorize text. Scene text (printed on a
  product/photo) is NOT a layer — it goes to `design.kept_in_photo` and stays baked in the base.
- **wordmark** (brand lettering, port of `wordmark.py`) → raster crop OR vectorize; NEVER
  font-matched, NEVER re-set as editable text. Split any pictogram (♡/★) out of the line.
- **shape/card/button** → Figma primitive (rect/ellipse with fitted fill) when the fill is
  solid/gradient; only vectorize if it is a genuine complex path.
- **photo/product/person** → raster IMAGE layer with a mask (ellipse for avatars, rrect,
  or alpha). Qwen RGBA layer or matte cutout provides the alpha. NEVER vectorized.
- **icon/badge/logo/arrow/simple graphic** → vectorize (VTracer color / Potrace binary),
  gated by a re-render fidelity check; fail → keep raster crop.
- **emoji** → keep as text character (platform font) or platform PNG; NEVER vectorize.
- **chart/diagram (default)** → one intentional raster cluster (exact swappable crop).
  Semi-editable upgrade only when primitives are positively tagged with `chart_group_id`
  (labels → TEXT, bars/axes → SHAPE, plot lines/connectors → gated VECTOR). See
  `docs/DIAGRAM-EDITABILITY.md`.

## Hard rules (do not violate)
1. Do NOT vectorize the whole ad. Only cropped simple elements.
2. Text → real Figma text nodes. Photos/products/people → raster + mask.
3. Qwen is the *layer proposal engine only*. The harness/routing stays in control; the
   agent orchestrates tools over JSON and must not hallucinate visual structure.
4. Every stage is idempotent and writes its artifact; a failed stage degrades (writes a
   note) rather than aborting the run, so the agent can retry a single stage.
5. The untouched source image can never be the rebuilt background when foreground layers exist.
   Canonical entities create one union removal mask and one inpainted background plate.
6. Raw model observations never become Figma layers directly. Mask-aware fusion assigns each
   observation to one canonical entity before assets, inpainting, or grouping.
7. Hierarchy, parentage, and Auto Layout are planned from canonical merged entities before
   reconstruction. Reconstruction may attach paint/assets to that plan, but may not silently
   replace it; an unreconciled run uses legacy layout only as a recorded degradation. Explicit
   replacement assets (such as before/after splits) may occupy their planned source leaf only
   when they declare that source and remain inside its frozen bounds.
8. A complex gradient, texture, illustration, or photographic background stays a raster clean
   plate. Pixels outside the removal union are byte-identical to the source; only genuinely
   hidden pixels inside that union are synthesized. Do not force an abstract background through
   a vector tracer. A simple, strongly proven gradient surface may still become a native shape.
9. Straight divider/rule geometry stays a native thin Figma bar. Arrows and decorative lines may
   become vectors only when the SVG render-back gate passes; otherwise keep the exact alpha raster.

## Config keys (config.yaml) the modules read
```yaml
device: cuda                       # cuda | cpu
backend_url: http://127.0.0.1:8188 # ComfyUI, optional (qwen worker)
ocr:
  primary: ppocr-v6                # ppocr-v6 | surya | doctr | tesseract
  challengers: [surya]             # run + reconcile by IoU when set
qwen:
  mode: comfyui                    # comfyui | direct-diffusers
  workflow: workflows/qwen_layered_8_api.json
  layers: 8
vectorize:
  color_engine: vtracer            # vtracer binary path resolved from PATH or .bin
  binary_engine: potrace
upscale:
  enabled: true                    # Real-ESRGAN for compressed Meta Library assets
figma:
  mode: plugin                     # native-node Figma plugin (the only supported import path)
```
