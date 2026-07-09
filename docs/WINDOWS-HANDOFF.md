# Windows RTX 5080 — setup + handoff prompt

Goal right now: **one ad in → recreation you can look at (`preview.png` + `layers/`) on this
machine.** Figma export is optional and comes later. The models live on this box; the code
lives in git. They meet through paths in `config.yaml`.

## ⚠️ Do not commit models to git
Qwen weights are multi-GB. Keep them in a local folder (or the HuggingFace cache) and point
`config.yaml` at them. `.gitignore` already blocks `*.safetensors/*.pth/*.onnx`.

## One-time setup
```powershell
git clone https://github.com/fortun8te/ad-decompiler
cd ad-decompiler
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# PyTorch with CUDA (see the header of requirements-gpu.txt for the exact index line):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-gpu.txt

copy config.example.yaml config.yaml   # then edit: device: cuda, model/backend paths
```

## Models
- **Qwen-Image-Layered** — HuggingFace `Qwen/Qwen-Image-Layered`. Two ways to run it:
  - **ComfyUI worker** (recommended): install ComfyUI + the `Comfy-Org/Qwen-Image-Layered_ComfyUI`
    custom node, put the weights where that node expects, start ComfyUI (default
    `http://127.0.0.1:8188`). Set `qwen.mode: comfyui`, `backend_url` to match. The graphs in
    `workflows/qwen_layered_{4,8}_api.json` are what the worker POSTs — open them once in
    ComfyUI to confirm node names match your install.
  - **Direct** (no ComfyUI): `qwen.mode: direct-diffusers`; the worker loads the HF model itself.
- **OCR**: PaddleOCR PP-OCRv6 downloads its own weights on first run. Surya/docTR similar.
- **VTracer / Potrace**: install the binaries (`cargo install vtracer`, and Potrace via choco
  or the official build); make sure they're on PATH.

## Run it
```powershell
python run_pipeline.py --input path\to\ad.png --output runs\run_001
```
Then open `runs\run_001\`:
- `preview.png` — the recreation (layers stacked back together)
- `layers\` — every layer as its own image, plus `layers_contact.png` (all of them at a glance)
- `design.json` — the machine-readable layer list (the source of truth)
- `ocr.json`, `elements.json`, `qwen.json`, `qa.json` — what each stage found

Re-run just one stage after a tweak: `--resume qwen` (or `ocr`, `elements`, `merge`, `design`,
`preview`, `diff`, `qa`). Batch a folder: `--input .\ads\ --batch`.

## What "good" looks like at this stage
Poor-but-real is fine. Success = the text pieces, the product/photo pieces, and the
icon/badge pieces come out as **separate** layers in `layers/`, the `preview.png` roughly
resembles the ad, and `design.json` has sensible layer types. Fidelity polish comes after.

---

## Copy-paste prompt for Claude Code / OpenCode / Cursor on Windows

> I'm running the `ad-decompiler` repo (already cloned) on a Windows machine with an RTX 5080.
> Read `README.md`, `docs/CONTRACT.md`, `docs/WINDOWS-HANDOFF.md`, and `AGENTS.md` first.
>
> The goal right now is ONLY: get one real Meta ad through the pipeline so it produces
> `runs/<id>/preview.png` + a `layers/` folder + `design.json` on this machine. Figma export
> stays OFF (`figma.enabled: false`) — ignore it for now.
>
> The CPU/deterministic spine (routing, wordmark, build_design_json, pixel_diff, render_preview)
> is done and tested. The GPU modules (ocr.py, qwen_worker.py, element_detect.py, merge_layers.py,
> vectorize.py) are written against real APIs but have NOT been run on a GPU yet — your job is to
> make them actually run here:
> 1. Set up `config.yaml` (device: cuda; point qwen at my local Qwen-Image-Layered model — I have
>    it at `<PATH>`; choose ComfyUI or direct-diffusers mode).
> 2. `pip install -r requirements.txt -r requirements-gpu.txt` and torch cu121; fix any dep/import
>    errors on this Windows/CUDA box.
> 3. Run `python run_pipeline.py --input <one_ad.png> --output runs/test1` and fix whatever breaks,
>    stage by stage (use `--resume <stage>`). Each stage writes JSON — inspect it, don't guess.
> 4. Do NOT change the routing rules or the design.json schema. Do NOT vectorize whole photos or
>    text — those rules in routing.py are intentional. Qwen only proposes layers; OCR + element
>    detection own text and boxes.
> 5. When it runs, show me `runs/test1/preview.png` and `layers_contact.png` and tell me which
>    stages are weak so we can improve them.
>
> Don't rebuild anything that exists. Don't add a frontend. Poor-but-real layers are the target.
