# ad-decompiler

**Any flat image → a duplicate-free, editable, Figma-native scene graph.**

The old repo was an unvalidated three-commit scaffold. It kept the original image as the
background and then placed extracted elements on top, so duplicates were unavoidable and an
empty decomposition could score perfectly. The v2 pipeline removes that architecture.

```text
image
  → OCR ensemble + font/style/paragraph analysis
  → residual proposals + SAM 3 on every proposal + semantic prompt sweep
  → mask-aware canonical entities
  → vector/native/raster routing
  → one ownership map + one inpainted clean background
  → nested frames, selective Auto Layout, constraints and repeats
  → recursive Figma compiler + structural/visual QA
```

## What is implemented

- Current PaddleOCR 3/PP-OCRv6 parsing, Surya/docTR challengers, calibrated reconciliation,
  word geometry, and targeted 2× retries for small/uncertain text.
- Painted glyph boxes, baseline/rotation/color estimates, paragraph grouping, hierarchy,
  shared style IDs, and bounded local font retrieval with ranked candidates.
- Official local SAM 3 image adapter. It prompt-sweeps products, people, logos, icons, arrows,
  badges, buttons, and illustrations, then box-refines every deterministic residual proposal.
- Mask-aware SAM/residual/Qwen fusion with one canonical ID per entity and meaningful nesting.
- Asset extraction, native shape/color inference, wired VTracer/Potrace with full SVG and a
  color+silhouette render gate, plus honest raster fallback.
- One union removal mask and one background inpaint. The untouched source is explicitly
  rejected as a rebuilt background.
- Conservative frame/Auto Layout inference, relative child coordinates, constraints, and safe
  repeat/component markers.
- Recursive, idempotent Figma import with native text, SVG, masks, effects, styles, groups,
  font fitting, replace/create-copy modes, progress, warnings, and a real plugin UI.
- Atomic local Figma bridge staging with asset checksums, previews, run summaries, and honest
  compiler preflight warnings.

GPU model quality still needs to be benchmarked on the target RTX workstation. The Mac can run
the deterministic stages and full test suite, but it cannot prove SAM/OCR/inpainting quality.

## Run

```bash
cp config.example.yaml config.yaml
python3 run_pipeline.py --input image.png --output runs/example
```

Main artifacts:

```text
ocr_raw.json          raw OCR observations
ocr.json              text geometry, font candidates, blocks and hierarchy
sam3.json             SAM prompt/box observations
fused_elements.json   canonical elements after mask-aware fusion
ownership.png         exclusive pixel-owner map
removal_mask.png      the single final removal mask
background_clean.png  the only allowed reconstructed background
reconstruction.json   assets/vector/inpaint diagnostics
layout.json           nested frame tree
design.json           Figma scene graph v2
design_preflight.json missing asset/compiler warnings
preview.png / qa.json render and QA output
runtime_report.json   exact model/fallback/retry evidence and acceptance policy
```

Resume from any stage:

```bash
python3 run_pipeline.py --input image.png --output runs/example --resume sam
python3 run_pipeline.py --input image.png --output runs/example --resume reconstruct
python3 run_pipeline.py --input image.png --output runs/example --resume qa
```

## RTX 5080 setup

On Windows, the easiest path is now one file:

```powershell
Start Bridge.bat
```

Double-click it. On the first run it installs the RTX environment, creates and aligns
`config.yaml`, reports any model file still needed, then starts the bridge. Later launches are
immediate and safely reuse an already-running bridge. It never updates the repo unless you run
`Start Bridge.bat -Update`.

Figma Desktop needs `figma-plugin\manifest.json` imported once. For a full benchmark, run
`.\start_rtx.ps1 -InputDir C:\images\benchmark` after the bridge reports the machine ready.

After the model files and LM Studio are ready, run `Start Bridge.bat -SelfTest` once. It really
runs the configured OCR, SAM 3, Gemma vision, Big-LaMa, VTracer, Figma staging, and one integrated
synthetic pipeline. Proof is saved under `runs\rtx-self-test`. Normal restarts only read that
small proof; they do not reload models. Use `-ForceSelfTest` after driver/model changes.

For remote Tailscale benchmarking, use `Start Bridge.bat -Remote`. It binds only to this PC's
Tailscale address, never every network interface. Pass the shown URL to
`scripts\remote_benchmark.py --bridge <url>`.

Use Python 3.12 and current CUDA 12.8/PyTorch wheels. Keep ComfyUI/Qwen in its own process.

```bash
pip install -r requirements.txt
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

git clone https://github.com/facebookresearch/sam3.git C:\src\sam3
pip install -e C:\src\sam3
```

Download the official SAM image checkpoint once and put its local path in `config.yaml`.
The setup script installs Big-LaMa because acceptance runs require it; OpenCV remains a visibly
lower-quality emergency fallback and cannot pass the production readiness gate.

## Figma

**One click (Windows):** double-click `Start Bridge.bat` in the repo folder.

**Mac:** double-click `start_bridge.command`, or run `./start_bridge.sh`.

The launcher installs the environment on its first run, creates `config.yaml`, keeps its inbox
and port aligned with the bridge, stamps the plugin build number, and starts the bridge. Missing
models do not hide the bridge: the launcher and plugin show the exact remaining fixes, while
uploads stay blocked until required quality tools are ready.

**Build numbers:** `Start Bridge.bat` / `start_bridge.sh` auto-stamp the plugin as
`v{VERSION}+b{git-commit-count}.{sha}` (see the badge in the plugin header). The bridge
records the last seen plugin build at `GET /health` → `plugin_client` and in
`~/figma-inbox/plugin_client.json`.

```bash
./start_bridge.sh
# or manually:
python3 -m src.figma_bridge --inbox ~/figma-inbox --port 8790
```

Import `figma-plugin/manifest.json` as a Figma development plugin. It includes `figma-plugin/icon.svg`
and the UI shows the same mark. The plugin shows the staged
preview and warnings, then imports as a new copy or replaces the previous generated frame.
Reimporting no longer stacks overlapping frames at `(0,0)`.

## Verification

```bash
pytest -q
```

On the RTX machine, put representative images in one folder and run:

```bash
python3 doctor.py --config config.yaml
python3 benchmark.py --input-dir benchmark-images --output runs/benchmark
```

`doctor.py` stops a run that would silently fall back because SAM, OCR, or CUDA is missing.
The benchmark runs that doctor preflight itself, records `doctor.json`, and fails if any image
has a real reconstruction/import failure or an enabled required model fell back. Qwen remains
advisory unless `qwen.required: true`; its outage is still visible in `runtime_report.json`.

The CPU suite covers OCR normalization/retries, SAM fallback and box-refinement contracts,
mask fusion/nesting, one-time inpainting, duplicate removal, Qwen asset resolution, hierarchy,
scene-graph compilation, atomic Figma staging, and an end-to-end vertical slice.

See [docs/ARCHITECTURE-V2.md](docs/ARCHITECTURE-V2.md) for the benchmark gate and GPU-only next
decisions. Do not call the system Codia-quality until the representative benchmark passes.
