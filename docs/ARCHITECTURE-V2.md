# Image decompiler v2

The original three-commit scaffold could report a perfect visual score while producing no
editable reconstruction, because it used the untouched source as the background. V2 makes a
canonical scene graph and exclusive pixel ownership the center of the system.

## Non-negotiable invariants

1. One real visual object becomes one canonical entity. Raw OCR, SAM, Qwen, and CV results are
   observations, never Figma layers by themselves.
2. Every residual proposal is box-refined by SAM 3. A global prompt sweep finds objects the
   residual detector missed. Mask-aware fusion runs before any asset is created.
3. Editable text stores OCR geometry and painted-ink geometry separately. Font family is a
   ranked candidate list. Figma performs the final render-and-fit correction.
4. Foreground pixels have one owner. The background removal mask is built after deduplication,
   then inpainted once. The untouched source is forbidden as a rebuilt background.
5. Native primitives win over SVG; SVG wins over raster only when a render-back fidelity gate
   passes. Photos, products, people, and complex artwork remain alpha rasters.
6. The Figma compiler consumes a nested scene graph. It creates frames, Auto Layout only when
   evidence is strong, constraints otherwise, native text/styles, SVGs, masks, and repeat-safe
   components. Reimport updates the same generated frame instead of stacking another copy.
7. QA includes structure. A baked screenshot cannot pass because background provenance,
   editable ratio, missing assets/fonts, duplicate ownership, OCR recall, and render fidelity
   are checked separately.

## Data flow

```text
normalize
  -> OCR ensemble
  -> painted text + font retrieval + paragraphs/hierarchy
  -> residual proposals
  -> Qwen layer observations (optional/advisory)
  -> SAM 3 prompt sweep + box refinement of every residual
  -> mask-aware observation fusion
  -> semantic routing
  -> frozen scene intent: conservative frame/Auto Layout/component inference
  -> canonical asset extraction + vector fidelity gate
  -> one ownership map + one union removal mask + one inpainted plate
  -> hydrate reconstructed paint/assets into the frozen scene intent
  -> scene-graph v2
  -> recursive Figma compiler
  -> structural + visual QA
```

Important artifacts per run:

- `ocr_raw.json`: OCR engines before style/layout enrichment.
- `ocr.json`: painted boxes, baselines, font candidates, blocks, hierarchy, style IDs.
- `residual.json`: deterministic non-text proposals.
- `sam3.json` and `sam3_masks/`: prompt and box-refined observations.
- `fused_elements.json` and `fused_elements/`: canonical element masks.
- `scene_intent.json`: hierarchy, parent/child geometry, and layout decisions made before
  reconstruction changes material assets, plus a fingerprint so resumes rebuild a stale plan.
- `reconstruction.json`, `ownership.png`, `removal_mask.png`, `background_clean.png`.
- `layout.json`: hydrated nested frame tree with relative child coordinates.
- `design.json` and `design_preflight.json`: Figma scene graph and compiler warnings.

## GPU deployment

Use Python 3.12, current CUDA 12.8/PyTorch wheels, and the official SAM repository. Keep
ComfyUI/Qwen in its own process so its dependency tree cannot destabilize OCR/SAM. Configure a
local SAM checkpoint; automatic downloads are off by default. Big-LaMa is the preferred local
inpainting backend, with OpenCV as the honest low-quality fallback.

The 128 GB system RAM is useful for CPU offload, but VRAM remains the constraint. Cache model
instances and the single SAM image embedding. Do not reload a model per prompt or element.

Run `python doctor.py --config config.yaml` before the first run. It is deliberately strict for
active CUDA, OCR, and SAM dependencies, so a benchmark cannot quietly turn into a fallback-only
run on the wrong machine.

## Benchmark gate

Select a frozen subset by fixture prefix instead of relying on directory order:

```bash
python benchmark.py --input-dir "C:\\path\\to\\IMAGE AD INSPO" --ids 026,034 --ids 103 --output runs/named-benchmark
```

`--ids` and its alias `--include` are repeatable, normalize numeric IDs to three digits, and
fail if an ID is missing or resolves to multiple files. `benchmark.json` records the requested
IDs and each resolved source filename, absolute path, size, and SHA-256; the Markdown report
also prints the immutable filename/hash manifest. Use `--no-auto-repair` for a baseline run; it
explicitly disables both the harness and legacy auto-repair switches.

Lock 10-20 representative inputs before tuning. Include ads, UI screenshots, posters, product
shots, dense paragraphs, multiple fonts, repeated cards, small icons/arrows, gradients, and
overlaps. Record:

- OCR character error rate and word/line ink-box IoU.
- Font top-1/top-5 retrieval and final Figma rendered-ink IoU.
- Element recall/precision, duplicate rate, and mask boundary F-score.
- Residual OCR/object leakage in `background_clean.png`.
- Vector render score and path count; raster fallback rate.
- Parent/child and repeated-group accuracy.
- Missing font/asset count and editable-node ratio.
- Figma export SSIM/local delta and manual cleanup minutes.

No model or heuristic change ships because one screenshot looks better. It must improve the
benchmark without increasing duplicates or missing layers.

## Next GPU-only decisions

- Validate SAM 3 prompt vocabulary and thresholds on the benchmark. Add OmniParser only if
  small UI/icon recall remains materially low.
- Measure SAM edges on hair, hands, and translucent products. Add BiRefNet only for classes
  where alpha-boundary scores justify the extra model.
- Compare Big-LaMa against a heavier local fill model on large semantic holes; use the heavier
  model only for those holes.
- Train/re-index font retrieval against the actual fonts available to the Figma desktop user.
  The compiler's render-fit loop remains the final authority.

## Primary references

- Codia image-to-Figma flow: https://codia.ai/blog/image-to-figma-guide
- Codia developer package: https://developer.codia.ai/
- Meta SAM 3: https://github.com/facebookresearch/sam3
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- Surya OCR/layout: https://github.com/datalab-to/surya
- VTracer: https://github.com/visioncortex/vtracer
- LaMa: https://github.com/advimman/lama
- Figma Plugin API: https://developers.figma.com/docs/plugins/api/figma/
