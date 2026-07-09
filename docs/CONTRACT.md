# ad-decompiler — stage contract (build against this)

All stages read/write plain dicts matching `src/schema.py`. Pixels = source-image px,
origin top-left. The orchestrating agent inspects and repairs over these JSON files.

## Pipeline dataflow
```
run_pipeline.py --input ad.png --output ./runs/run_001
  1 normalize.load_normalize(input, run_dir)        -> normalized.png, {w,h}
  2 ocr.run_ocr(normalized, cfg)                     -> ocr.json     (schema.OcrResult)
  3 element_detect.detect(normalized, ocr, cfg)      -> elements.json (list[schema.Element])
  4 qwen_worker.propose_layers(normalized, cfg)      -> qwen_layers/*.png + qwen.json (list[schema.QwenLayer])
  5 (qwen outputs saved as RGBA layers)
  6 merge_layers.merge(ocr, elements, qwen, cfg)     -> merged.json  (list of routed candidates)
  7 routing.route(candidate) applied inside merge     -> each candidate tagged target:text|shape|image|icon
  8 build_design_json.build(merged, canvas, run_dir) -> design.json  (schema.DesignDoc)  [SOURCE OF TRUTH]
  9 figma_import.import_design(design.json)           -> nodes in Figma (plugin or clipboard)
 10 figma_import.export_screenshot()                  -> figma_export.png
 11 pixel_diff.compare(normalized, figma_export)      -> diff.png + partial qa
 12 repair.assess(design, qa, ocr)                    -> qa.json (schema.QaResult with repairs[])
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

## Hard rules (do not violate)
1. Do NOT vectorize the whole ad. Only cropped simple elements.
2. Text → real Figma text nodes. Photos/products/people → raster + mask.
3. Qwen is the *layer proposal engine only*. The harness/routing stays in control; the
   agent orchestrates tools over JSON and must not hallucinate visual structure.
4. Every stage is idempotent and writes its artifact; a failed stage degrades (writes a
   note) rather than aborting the run, so the agent can retry a single stage.

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
  mode: plugin                     # plugin | clipboard
```
