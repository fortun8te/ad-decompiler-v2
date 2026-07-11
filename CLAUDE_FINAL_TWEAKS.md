# Claude final-tweaks plan

This repo already has the v2 pipeline. Do not redesign it or add another parallel
architecture. The job now is to validate it on the RTX machine, fix measured failures, and
leave a clean Figma handoff.

## 1. Establish a clean baseline

Run from the repo root:

```powershell
python doctor.py --config config.yaml
pytest -q
```

Do not continue to model tuning if `doctor.py` is not `READY`. Fix the environment first.
Keep the model weights outside git.

## 2. Lock the benchmark set

Use 10–20 real images covering:

- dense paragraphs and multiple columns;
- several fonts, weights, and rotated text;
- products/person cutouts;
- icons, arrows, badges, and logos;
- gradients, shadows, strokes, rounded cards;
- repeated cards and overlapping elements.

Never tune against one “hero” image. Keep the originals unchanged and record the folder path.

## 3. Run the first real batch

```powershell
python benchmark.py --input-dir C:\images\benchmark --output runs\benchmark-001
```

Inspect `benchmark.md`, `runtime_report.json`, `ocr.json`, `sam3.json`, `ownership.png`,
`background_clean.png`, `layers_contact.png`, `preview.png`, and `design_preflight.json`.

The benchmark must reject:

- missing CUDA, OCR, or SAM;
- model fallback when active models are required;
- duplicate ownership;
- leaked foreground pixels in the clean plate;
- missing assets/fonts;
- failed Figma compiler layers;
- low visual or editable-text scores.

## 4. Fix failures in this order

1. Background leakage and duplicate objects.
2. Missing or wrongly merged OCR lines.
3. Bad text geometry/font substitutions.
4. Wrong masks and product/person boundaries.
5. Icon/vector fidelity.
6. Incorrect grouping or layer order.
7. Visual effects and polish.

For every fix, add a regression fixture to `tests/` and rerun the full suite. Do not lower QA
thresholds to make a run pass.

## 4a. Typography-specific acceptance

Treat typography as its own benchmark, not as ordinary OCR. For every text sample record:

- font family top-1 and top-5 accuracy;
- bold, light, italic/oblique, and monospaced-style accuracy;
- rotation angle error;
- painted glyph-box IoU and baseline error;
- line-break and paragraph grouping accuracy;
- text color/gradient/stroke/effect fidelity;
- final Figma rendered-ink IoU.

Use a VLM only as a semantic judge for ambiguous cases such as “printed on the product” versus
“overlay copy.” It must not replace OCR geometry or become an unverified source of Figma layers.
If a text region is classified as printed scene text, keep it in the clean image and record why;
otherwise create editable text. Similar text elements should share a style ID/font candidate
unless the render-fit loop proves that a change is needed.

The compiler must preserve editable text even when color is a gradient or the text has an outline.
If a font or text effect cannot be represented faithfully, report the substitution and keep the
original pixels as a separate, masked fallback layer rather than silently producing wrong text.

## 5. Figma validation

On the RTX machine, start the bridge:

```powershell
python -m src.figma_bridge --inbox C:\figma-inbox --port 8790
```

Load `figma-plugin/manifest.json` in Figma Desktop. Use **Create copy** once, then **Replace
existing** for iterations. Confirm:

- the manifest loads without an allowed-domain error;
- text remains editable and uses the best available font candidate;
- groups/frames preserve relative positions;
- gradients, strokes, shadows, masks, SVGs, and opacity survive import;
- Replace does not stack duplicate frames;
- the plugin posts the rendered export and compiler report back to the run folder.

## 6. Final acceptance

Claude should only call this complete when:

- the locked benchmark passes with no unresolved hard failures;
- every image has a usable `runtime_report.json`;
- Figma imports the batch without compiler errors;
- manual cleanup time is recorded for the same images;
- `pytest -q`, plugin smoke, and `git diff --check` are green;
- the final report names any remaining known limitations honestly.

If the RTX or Figma Desktop is unavailable, stop at the handoff and report that as the only
remaining blocker. Do not claim Codia-level quality from Mac-only tests.
