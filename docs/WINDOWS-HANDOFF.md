# Windows RTX 5080 handoff

The Mac validates deterministic code. The Windows workstation must validate real model output.
Do not call the rebuild successful until a representative batch passes the benchmark in
`ARCHITECTURE-V2.md`.

## Setup

**One click:** double-click `Start Bridge.bat`. The first launch installs the RTX environment,
creates `config.yaml` + `~/figma-inbox`, checks what is ready, and starts
`http://localhost:8790`. Later launches are immediate. Import `figma-plugin/manifest.json` in
Figma Desktop once.

If the bridge is already running it is reused. If another app owns port 8790, the launcher says
so instead of starting a second broken process. Advanced custom ports also require updating the
address and allowed development domain in the Figma plugin.

**Full GPU setup + benchmark:**

```powershell
.\setup_rtx.ps1
.\start_rtx.ps1 -InputDir C:\images\benchmark
```

The scripts create the environment, install the dependencies, check CUDA/OCR/SAM, start the
local Figma bridge, and run the benchmark. Use the longer commands below only when you need to
change the installation manually.

To make setup execute the backend currently selected in `config.yaml`, rather than only checking
files and ports, use:

```powershell
.\setup_rtx.ps1 -DeepDoctor
```

This is an RTX-only operation. It is deliberately not treated as having run on the Mac.

Use Python 3.12 and current Blackwell-compatible CUDA wheels:

```powershell
git clone https://github.com/fortun8te/ad-decompiler-v2
cd ad-decompiler-v2
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

## Downloads that cannot be bundled

- Python 3.12 and Git.
- A current NVIDIA driver for the RTX 5080.
- The official SAM 3 image checkpoint at `C:\models\sam3.pt` (or update its config path).
- LM Studio with `google/gemma-4-e4b` loaded and its local server enabled.
- ComfyUI plus the configured Qwen workflow only when `qwen.required: true`.
- VTracer and two SVG render checkers are installed by `setup_rtx.ps1`; `choco install potrace`
  adds the optional monochrome backup tracer.
- Recommended OCR backup: `winget install UB-Mannheim.TesseractOCR`.

Run `.\.venv\Scripts\python.exe doctor.py`. Every failed item now includes a direct `FIX:` line.
PaddleOCR and Surya are attempted separately because their Windows packages can conflict with
Blackwell. Either may warn and skip without breaking the working docTR OCR path.

## Choose and prove the inpaint backend

`inpaint.mode` is the model claim for an acceptance run. `strict_acceptance: true` makes
`doctor.py` reject a missing selected Flux or PowerPaint route instead of accepting a later
fallback as equivalent evidence.

For local Flux Fill, install the weights into the actual ComfyUI installation, then expose that
directory in the config so doctor can inspect it:

```powershell
.\scripts\setup_flux_inpaint.ps1 -ComfyDir "C:\ComfyUI"
```

```yaml
inpaint:
  mode: flux_comfy
  strict_acceptance: true
  comfy:
    enabled: true
    required: true
    comfy_dir: C:\ComfyUI
```

Flux readiness means the workflow file exists, ComfyUI answers, and the local GGUF/encoder/VAE
files are visible. It still is not proof of an actual inpaint. Run `doctor.py --deep` or
`Start Bridge.bat -SelfTest`; those submit a small masked image and require the result to report
`flux-comfy` with untouched pixels outside the mask.

PowerPaint is intentionally **not** downloaded or named by this repository. Supply your own
RTX-side adapter, then configure its import path and callable:

```yaml
inpaint:
  mode: powerpaint
  strict_acceptance: true
  allow_fallback: false
  powerpaint:
    enabled: true
    required: true
    adapter_module: your_powerpaint_adapter
    callable: inpaint
```

The adapter contract is `inpaint(rgb_uint8, mask_uint8, cfg) -> RGB image | None`. Doctor only
checks that this explicit adapter is configured and importable; it does not claim the weights or
CUDA model are ready. The selected PowerPaint runtime smoke is the proof: it invokes that adapter
with fallback disabled and records `backend=powerpaint`, mask-local change, and unchanged pixels
outside the hole. Keep the resulting `runtime_smoke.json` / `self_test.json` with the benchmark.
When `runtime.require_active_models: true` (the default acceptance config) or
`strict_acceptance` is enabled, `start_rtx.ps1` checks this cached self-test automatically before
it starts a benchmark. If the config or code changed, it reruns the real OCR/SAM/Gemma/selected-
inpaint probes rather than treating dependency checks as quality evidence.

## Prove the models actually run

`doctor.py` checks dependency readiness. It is not runtime proof. Once it says READY, run:

```powershell
Start Bridge.bat -SelfTest
```

For a final Codia-parity benchmark, use the explicit Figma-acceptance launcher flag. It prevents
a local preview from being counted as a Figma result: each screen must receive a fresh plugin
export and report from Figma Desktop.

Use named fixture IDs for a repeatable, reviewable subset instead of trusting directory order:

```powershell
.\start_rtx.ps1 -InputDir "C:\images\IMAGE AD INSPO" -Ids 002,010,017,020 -Output runs\codia-parity -RequireFigma -FigmaWaitS 120
```

Keep the plugin open. The benchmark pauses for each image until its matching Figma import posts
the export/report; it fails rather than silently using the local preview if that proof is missing
or stale. Do not combine `-RequireFigma` with `-NoBridge`.

This executes bounded real OCR, SAM 3, Gemma vision, the inpaint backend selected in config,
VTracer and Figma staging probes, then one integrated synthetic pipeline. It writes `runs\rtx-self-test\latest.json`, the input,
model artifacts, `runtime_report.json`, and a fingerprinted `self_test.json`. A normal bridge
restart only reads that cache. It asks for another test after relevant code/config changes or
seven days. Use `Start Bridge.bat -ForceSelfTest` after changing drivers, CUDA, or model files.

For a remote benchmark over Tailscale:

```powershell
Start Bridge.bat -Remote
python scripts\remote_benchmark.py --bridge http://<shown-tailscale-ip>:8790 --input-dir C:\images\benchmark
```

Remote mode binds only to the machine's Tailscale IP. Figma cross-machine access still requires
an HTTPS `tailscale serve` URL because Figma blocks mixed-content HTTP hosts.

`doctor.py` must print `READY` before the first run. It blocks a benchmark if CUDA, the active
OCR engine, or SAM's local image checkpoint is missing; optional fallbacks are shown as warnings.
Run `pytest -q` after cloning only if you are changing code.

The benchmark runs the same preflight itself and writes `doctor.json` beside the scorecard.
With the supplied acceptance config it also runs the bounded real-model smoke automatically,
even when you call `python benchmark.py` directly instead of the launcher.
Every run also writes `runtime_report.json`: `ok` means no model degraded, `degraded` means an
advisory path (normally Qwen) fell back, and a listed `violation` means a required OCR/SAM model
did not run and the benchmark is invalid. Do not use `--skip-doctor` for acceptance evidence.

VTracer, Big-LaMa, and the SVG render-back checker are installed by setup. Potrace on `PATH` is
an optional monochrome backup. OpenCV inpainting remains available for diagnosis, but strict
acceptance must use and smoke-prove the selected active route; it cannot represent Flux or
PowerPaint evidence.

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
