# ad-decompiler

**Flat Meta ad → editable Figma layers.** A scriptable, headless Python pipeline built to run
on a Windows RTX 5080. Deterministic CV + OCR extract the *facts*; Qwen-Image-Layered proposes
RGBA layers; the harness routes every element to the right Figma primitive; a screenshot
diff proves the reconstruction is close. An OpenCode agent orchestrates the stages over JSON.

> Milestone: **one ad in → editable Figma out → screenshot diff proves it's close.**
> This is a backend pipeline. There is no SaaS UI and none is planned here.

```
input ad → OCR + element detect → Qwen layers → merge → design.json → Figma import → screenshot → pixel-diff QA
```

## Why this shape
- **Qwen is only the layer-proposal engine.** OCR + `element_detect` own text and boxes; Qwen
  owns z-order and clean alpha. The harness stays in control so the agent can't hallucinate
  visual structure.
- **The routing rules are the crown jewel** (`src/routing.py`, ported from a validated Mac
  harness): text → real Figma text, photos/products/people → raster + mask, only
  icons/badges/logos/arrows/simple graphics → vectorized (VTracer/Potrace). Never vectorize the
  whole ad. Scene text stays baked in the base (`kept_in_photo`); wordmarks stay as artwork.
- **`design.json` is the single source of truth** (`src/schema.py`). Every stage reads/writes
  JSON so the agent can inspect and repair without touching pixels.

## Run
```bash
python run_pipeline.py --input ad.png --output ./runs/run_001
python run_pipeline.py --input ./ads/ --batch
python run_pipeline.py --input ad.png --output ./runs/run_001 --resume qa   # re-run from a stage
```
Artifacts land in `runs/<id>/`: `normalized.png ocr.json elements.json qwen_layers/ merged.json
design.json figma_export.png diff.png qa.json`.

## Windows RTX 5080 setup
```bash
# CPU-side (any machine)
pip install -r requirements.txt
# GPU backends (Windows/CUDA 12.x) — see the header of requirements-gpu.txt for the torch cu121 line
pip install -r requirements-gpu.txt
```
Default vision stack (benchmark and pick per ad):
- OCR: **PaddleOCR / PP-OCRv6** (primary), **Surya** (challenger), **docTR** (fallback), Tesseract (baseline only)
- Layers: **Qwen-Image-Layered** (RGBA proposals) — via ComfyUI worker or direct diffusers
- Vectorize: **VTracer** (color), **Potrace** (binary icons/masks)
- Upscale: **Real-ESRGAN** for compressed Meta Library assets
- Optional: **SAM2** mask refinement when Qwen edges are rough

`config.yaml` (copy from `config.example.yaml`) selects device, backends, and Figma mode.

## Figma import (the one interactive seam)
Figma can't create nodes fully headlessly, so a companion plugin builds them:
1. `python -m src.figma_bridge --inbox ~/figma-inbox --port 8790`  (serves the staged run)
2. Import `figma-plugin/` in Figma desktop (Plugins → Development → Import from manifest).
3. Run it → **Import latest from bridge** → real editable nodes appear; the exported PNG is
   POSTed back to the run dir as `figma_export.png`.
4. `python run_pipeline.py … --resume qa` scores it.

`figma.mode: clipboard` is an alternative that reuses the Mac harness's proven kiwi clipboard
encoder (80/80 roundtrip) — paste straight into Figma, no plugin.

## Orchestrator agent
`AGENTS.md` + `opencode.jsonc` define an OpenCode agent that runs stages, reads the JSON
artifacts, and drives `qa.json.repairs[]` until QA passes. It never processes images itself.

## Benchmark (10 real Meta ads)
Track per ad: OCR accuracy, bbox accuracy, layer usefulness, runtime, VRAM/RAM, SVG path count,
Figma editability, manual cleanup time. See `docs/CONTRACT.md` for the stage contract.

## Repo layout
```
run_pipeline.py            CLI orchestrator (stage runner, --resume, --batch)
config.example.yaml        copy to config.yaml
AGENTS.md / opencode.jsonc OpenCode orchestrator agent
src/                       normalize ocr element_detect qwen_worker merge_layers routing
                           wordmark vectorize build_design_json figma_import figma_bridge
                           pixel_diff repair schema
workflows/                 ComfyUI API graphs for Qwen-Image-Layered (4 / 8 layer)
figma-plugin/              design.json → editable Figma nodes + PNG export
docs/CONTRACT.md           stage dataflow + routing + config keys
runs/                      per-ad outputs (gitignored)
```
