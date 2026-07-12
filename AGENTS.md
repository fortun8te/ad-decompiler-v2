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

## Cursor Cloud specific instructions

This VM is **CPU-only (no CUDA)**. Per the README, the deterministic stages and the full test
suite run here, but real OCR/SAM/inpaint quality cannot be proven — that needs the RTX box.

- **Interpreter:** use the project venv at `.venv` (run everything as `.venv/bin/python ...`).
  The update script recreates `.venv` and installs `requirements.txt` plus `pytesseract`.
- **`config.yaml` is gitignored** and created from `config.example.yaml`. It persists in the VM
  snapshot; if missing, recreate with `cp config.example.yaml config.yaml` and apply the CPU dev
  tweaks below. Do not commit it.
- **CPU dev config tweaks** (already applied in the snapshot's `config.yaml`):
  - `device: cpu`
  - `ocr.lang: eng` — the value is passed straight to Tesseract, which needs `eng` (the installed
    `eng.traineddata`), not the example's `en`.
  - `runtime.require_active_models: false` — otherwise the enabled-but-unavailable GPU OCR/SAM
    backends turn into QA hard-fails instead of honest CPU fallbacks.
  - `qwen.enabled: false` — no ComfyUI here; avoids connection attempts to `:8188`.
- **OCR is mandatory to run the pipeline.** `requirements.txt` ships no OCR engine, so the pipeline
  aborts without one. The CPU path uses the **Tesseract** system binary (`tesseract-ocr` +
  `tesseract-ocr-eng`, installed in the snapshot) via `pytesseract`. docTR/PaddleOCR/SAM3 are
  GPU-oriented and intentionally not installed; they fall back and log "unavailable" (expected).
- **Run the pipeline (primary product):**
  `.venv/bin/python run_pipeline.py --input <image> --output runs/<id>` (see `## Run loop`).
- **Optional Figma bridge:** `.venv/bin/python -m src.figma_bridge --inbox ~/figma-inbox --port
  8790 --config config.yaml`. `GET /health` returns `ok:true` but `machine_ready:false` on this
  box — that is the doctor preflight correctly flagging missing GPU models, not a bug.
- **Tests/lint:** `.venv/bin/python -m pytest -q` (no dedicated Python linter is configured; the
  repo relies on pytest + `git diff --check`). At the current HEAD, **9 tests fail regardless of
  environment** because of stale test doubles (e.g. `tests/test_figma_bridge.py` calls a 3-arg
  fake `run_one` while the real `run_pipeline.run_one` takes 4 args), plus reconstruct/text_analysis/
  harness cases; these are pre-existing repo issues, not environment problems (383 pass).
- Dependency versions are pinned to `numpy<2.3`, `opencv-python<5`, `pillow<12` (matching
  `requirements-gpu.txt`'s numpy cap and avoiding brand-new opencv 5 behavior drift).
