# Windows RTX 5080 handoff

The Mac validates deterministic code. The Windows workstation must validate real model output.
Do not call the rebuild successful until a representative batch passes the benchmark in
`ARCHITECTURE-V2.md`.

## Setup

**Bridge only (1 click):** double-click `Start Bridge.bat`. It bootstraps `config.yaml` +
`~/figma-inbox` and starts `http://localhost:8790`. Import `figma-plugin/manifest.json` in
Figma Desktop once.

**Full GPU setup + benchmark:**

```powershell
.\setup_rtx.ps1
.\start_rtx.ps1 -InputDir C:\images\benchmark
```

The scripts create the environment, install the dependencies, check CUDA/OCR/SAM, start the
local Figma bridge, and run the benchmark. Use the longer commands below only when you need to
change the installation manually.

Use Python 3.12 and current Blackwell-compatible CUDA wheels:

```powershell
git clone https://github.com/fortun8te/ad-decompiler
cd ad-decompiler
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

git clone https://github.com/facebookresearch/sam3.git C:\src\sam3
pip install -e C:\src\sam3

copy config.example.yaml config.yaml
python doctor.py --config config.yaml
```

Keep model weights outside git. Set the local SAM checkpoint and optional font-library paths in
`config.yaml`. Automatic SAM downloads are disabled. Keep ComfyUI/Qwen in a separate process;
it is an advisory layer/alpha source, not the element detector.

`doctor.py` must print `READY` before the first run. It blocks a benchmark if CUDA, the active
OCR engine, or SAM's local image checkpoint is missing; optional fallbacks are shown as warnings.
Run `pytest -q` after cloning only if you are changing code.

The benchmark runs the same preflight itself and writes `doctor.json` beside the scorecard.
Every run also writes `runtime_report.json`: `ok` means no model degraded, `degraded` means an
advisory path (normally Qwen) fell back, and a listed `violation` means a required OCR/SAM model
did not run and the benchmark is invalid. Do not use `--skip-doctor` for acceptance evidence.

Install VTracer and Potrace on `PATH`. Install Big-LaMa if compatible with the environment;
otherwise the run will label OpenCV as the inpaint fallback.

## First real run

```powershell
python run_pipeline.py --input C:\images\reference.png --output runs\gpu-001
```

For the actual acceptance run, put the locked representative images in one directory:

```powershell
python benchmark.py --input-dir C:\images\benchmark --output runs\benchmark
```

It writes `benchmark.json` and `benchmark.md`, and returns a failure code unless every run's QA
gate passes. Do not cherry-pick the best screenshot.

Inspect these before Figma:

- `ocr.json`: text, painted boxes, hierarchy, and font candidates.
- `sam3.json` and `sam3_masks\`: prompt coverage and box refinement.
- `fused_elements.json`: canonical elements; there should not be same-object duplicates.
- `ownership.png`: every foreground pixel should have one owner.
- `removal_mask.png` and `background_clean.png`: no leaked duplicate text/products.
- `layers_contact.png` and `preview.png`.
- `design_preflight.json`: no missing assets.

Useful resumes:

```powershell
python run_pipeline.py --input C:\images\reference.png --output runs\gpu-001 --resume ocr
python run_pipeline.py --input C:\images\reference.png --output runs\gpu-001 --resume sam
python run_pipeline.py --input C:\images\reference.png --output runs\gpu-001 --resume reconstruct
```

## Figma loop

```powershell
python -m src.figma_bridge --inbox $HOME\figma-inbox --port 8790
```

Import `figma-plugin\manifest.json` as a Figma development plugin. Review the staged preview and
warnings, choose **Create copy** for the first import, then **Replace existing** for iterations.
Run QA again after the plugin posts `figma_export.png` back to the run folder.

## Report back with evidence

For each benchmark image record OCR CER/ink IoU, element misses/duplicates, residual leakage in
the clean plate, vector fallback count, missing fonts/assets, final Figma render score, and manual
cleanup minutes. Include the actual run folders or screenshots; do not report only “it worked.”
