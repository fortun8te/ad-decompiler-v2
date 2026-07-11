# ad-decompiler orchestrator

Goal: turn a flat image into a duplicate-free, editable Figma scene graph and prove both visual
fidelity and structure. Inspect the real artifacts and rendered images; do not infer success from
logs alone.

## Run loop

```bash
python3 run_pipeline.py --input <image> --output runs/<id>
python3 run_pipeline.py --input <image> --output runs/<id> --resume <stage>
```

Stages: `normalize ocr text residual qwen sam elements merge reconstruct layout design preview
figma export diff qa`.

Inspect `ocr.json`, `sam3.json`, `fused_elements.json`, `reconstruction.json`, `layout.json`,
`design_preflight.json`, and `qa.json`. Also inspect `background_clean.png`, `ownership.png`,
`layers_contact.png`, `preview.png`, and the Figma export. Record exact misses/duplicates instead
of saying output is vaguely better or worse.

## Invariants

- Raw OCR/SAM/Qwen/CV results are observations. Only fused canonical entities become layers.
- SAM prompt-sweeps semantic elements and box-refines every residual proposal.
- Text remains native text; wordmarks remain artwork; photos/products/people remain alpha raster.
- Native primitives beat vectors. Vectors must pass the render-back gate. Never trace a photo or
  the whole image.
- The untouched source is never a reconstructed background. Build one final removal mask and
  inpaint once after deduplication.
- Every foreground pixel has one canonical owner. Preserve meaningful nesting such as an icon in
  a button; collapse same-object observations.
- Auto Layout is enabled only with strong row/column evidence. Artistic overlaps remain absolute
  with constraints.
- Missing assets/fonts and compiler exceptions are failures, not gray placeholders or hidden
  warnings.
- A high screenshot score cannot override structural failures or a low editable ratio.

## Completion gate

Use the representative benchmark in `docs/ARCHITECTURE-V2.md`. Report OCR/ink accuracy, element
recall and duplicates, clean-background leakage, vector fallback, Figma warnings, render score,
and manual cleanup time. GPU code is not considered validated until those artifacts come from the
actual RTX workstation.
