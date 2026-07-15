# OmniSVG-4B feasibility spike (neural logo/icon vectorizer)

**Status:** research-only feasibility spike. No weights downloaded, no `src/` changes. Evaluates whether
[OmniSVG 1.1 4B](https://huggingface.co/OmniSVG/OmniSVG1.1_4B) (NeurIPS 2025) should slot next to VTracer/Potrace
in `src/vectorize.py` as a "clean compact path" vectorizer for logos and icons.

**Date:** 2026-07-15 · **Target box:** Windows 11, RTX 5080 16 GB, Python 3.12, torch 2.10 cu128, VRAM already
contended by SAM3 + Gemma-12B + Flux.

---

## TL;DR verdict: NOT NOW. Revisit later if VRAM frees up.

OmniSVG is real, good research and genuinely produces the semantically-clean compact paths the premise hoped for —
but for *this* box and *this* pipeline it loses on every axis that matters right now:

| Axis | OmniSVG 1.1 4B | Our current VTracer/Potrace | Winner for us |
| --- | --- | --- | --- |
| VRAM to load | **16–17 GB** (bf16, no official quant) | ~0 (subprocess, MBs) | tracers |
| Latency / crop | **~40 s** (icon, 2.2k tokens) to 160 s+ (illustration) | ~50–300 ms | tracers |
| Image-to-SVG fidelity (LPIPS↓) | **0.237** | VTracer **0.035** (~7× better) | tracers |
| Image-to-SVG fidelity (MSE↓) | **0.034** | VTracer **0.002** (~17× better) | tracers |
| Output tokens (compactness) | **5.8k** | VTracer 52.4k (~9× more) | **OmniSVG** |
| Gradients / strokes / opacity | **none** (flat fill only) | native `<linearGradient>`, stroke specs | tracers |
| Figma `createNodeFromSvg` subset | M/L/C/A/Z + flat fill (cleanest possible) | same + gradients/strokes | tie / OmniSVG |
| Fails on photographic crops | **yes** ("fails on natural images") | degrades gracefully | tracers |

The one thing OmniSVG wins — compactness / semantic cleanliness — is exactly what our **render-back gate cannot
reward** (the gate scores pixel fidelity, not editability), and its reinterpretive output will be **rejected by that
same gate most of the time**. So we'd pay 16 GB of VRAM we don't have and 40 s/crop for a candidate the gate throws
away in the common case. The gate does make it *safe* to try later, which is why the verdict is "not now," not "never."

Sources for every number below are cited inline.

---

## 1. VRAM, latency, quantization

**Architecture.** OmniSVG 1.1 4B is `Qwen/Qwen2.5-VL-3B-Instruct` extended with an SVG tokenizer; bf16 weights are
**7.6 GB** on disk. There is also a 3B (8.49 GB) and an 8B (17.2 GB) variant. (Base VLM, sizes:
[HF model card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B), [GitHub README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md).)

**VRAM — the deal-breaker.** The numbers cluster right at or *above* the 5080's 16 GB, before contention:
- Official HF card: **"16G"** for the 4B. ([HF model card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B))
- GitHub README: **17 GB** for the 4B/3B, **26 GB** for the 8B. ([GitHub README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md))
- Community ComfyUI node: **"17GB+ VRAM for optimal performance"**, with an explicit CUDA-OOM troubleshooting entry
  advising to "free up GPU memory from other processes." ([ComfyUI-OmniSVG](https://github.com/A043-studios/ComfyUI-OmniSVG))

The runtime footprint is weights (7.6 GB) + the ViT vision tower + a KV cache that grows with the *very long* SVG
token sequences this model emits (up to 30k+ tokens, §3). That is why the load-time figure balloons to 16–17 GB
despite 7.6 GB of weights. On a 16 GB card this fits *only if nothing else is resident*. Our box already runs
SAM3 + Gemma-12B + Flux, so co-residency is impossible without a model-swap orchestration layer (evict, load
OmniSVG, run, evict, reload the others) — a lot of plumbing for a per-crop tool.

**Latency.** Measured per SVG token by the authors (4B):

| Tokens | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- |
| Seconds | 4.08 | 8.68 | 18.07 | 37.51 | 82.70 |

([HF model card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B), [GitHub README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md).)
Typical output lengths ([arxiv v3](https://arxiv.org/html/2504.06263v3)): icons **2.2k ± 0.9k**, illustrations
**8.1k ± 3.3k**, anime characters **28k ± 7.3k** tokens. So a single icon ≈ **40 s**; an illustration comfortably
exceeds **82 s** (extrapolating past the 4096-token row). Compare our tracers at ~50–300 ms/crop. This is a
100–1000× slowdown per crop; it cannot live inline in the hot path.

**Quantization.** No official bnb/GGUF/AWQ/vLLM path is published for any OmniSVG checkpoint
([HF card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B), [README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md),
[ComfyUI node](https://github.com/A043-studios/ComfyUI-OmniSVG) all silent on it). Because the backbone is a stock
Qwen2.5-VL-3B, in principle bnb 4-bit or vLLM's `bitsandbytes_4bit` backend applies — 4-bit would drop the 4B class
from ~8 GB to **~4–6 GB** of weights
([vLLM bnb docs](https://docs.vllm.ai/en/stable/features/quantization/bnb/),
[Qwen-VL bnb walkthrough](https://medium.com/@krshranjith/quantizing-qwen-models-using-bitsandbytes-bnb-for-efficient-inference-b937286b9bbf)).
But three caveats make this a "later, unproven" mitigation, not a plan:
1. The checkpoint's embedding/head are **extended with custom SVG tokens**; a naive quant or a vanilla vLLM load must
   carry that extended tokenizer/vocab, which no one has published a recipe for.
2. Even quantized, the **KV cache for 30k-token generations** still costs multiple GB, and community reports note
   VL models still OOM on 16 GB "depending on model size and input complexity"
   ([vLLM forum thread](https://discuss.vllm.ai/t/torch-outofmemoryerror-cuda-out-of-memory/2420)).
3. 4-bit degrades exactly the coordinate/color token precision that a vectorizer depends on — untested for this task.

---

## 2. Input conditioning — how faithful is image-to-SVG, really?

**It's a reinterpretation, not a trace.** OmniSVG's image-to-SVG mode takes a raster crop and *redraws* it as clean
paths — it is not pixel-locked. The independent-metric picture (authors' own table, image-to-SVG on
MMSVG-Illustration, [arxiv v3](https://arxiv.org/html/2504.06263v3)):

| Metric | OmniSVG 4B | StarVector 8B | **VTracer** | DiffVG | LIVE |
| --- | --- | --- | --- | --- | --- |
| DINO ↑ | 0.899 | 0.877 | **0.994** | 0.945 | 0.935 |
| SSIM ↑ | 0.906 | 0.900 | **0.966** | 0.955 | 0.950 |
| LPIPS ↓ | 0.237 | 0.238 | **0.035** | 0.065 | 0.111 |
| MSE ↓ | 0.034 | 0.046 | **0.002** | 0.001 | 0.008 |

The classical tracers — including the **VTracer we already run** — are dramatically more faithful (LPIPS ~7× lower,
MSE ~17× lower). OmniSVG's win is compactness: it hits that with **5.8k tokens vs VTracer's 52.4k** for the same
target. In plain terms: OmniSVG gives you a *human-plausible clean redraw* that a designer would accept as "close
enough and editable," whereas VTracer gives you a pixel-accurate but path-heavy replica. That is the exact
"faithful-but-spaghetti vs semantically-clean" tradeoff the premise described — confirmed, and quantified.

**Input handling.** The reference community node resizes inputs to a `target_size` of **64–512 px** before processing
([ComfyUI-OmniSVG](https://github.com/A043-studios/ComfyUI-OmniSVG)); the authors fit all SVGs inside a **200×200
viewbox** and quantize coordinates to that grid ([arxiv v3](https://arxiv.org/html/2504.06263v3)). So arbitrary
raster crops are accepted but downsampled to ~200–512 px and re-gridded — fine for logos/icons, lossy for fine text
or hairline detail. Lower temperature (range 0.01–1.0) yields more faithful conversions
([omnisvg.github.io](https://omnisvg.github.io/), [ComfyUI-Wiki](https://comfyui-wiki.com/en/news/2025-04-10-omnisvg-svg-generation-model)).

**Hard limitation.** From the paper's Limitations section, verbatim: *"OmniSVG is only bounded by vector style image
prompt and fails on natural images."* ([arxiv v3](https://arxiv.org/html/2504.06263v3)). Photographic product crops —
a large fraction of what an ad decompiler sees — are out of scope; only vector-style logos/icons/illustrations are in
scope. (Our render-gate would catch these failures automatically; see §5.)

---

## 3. Output constraints & Figma compatibility

**SVG feature subset.** OmniSVG emits a deliberately minimal grammar: five path commands **M, L, C, A, Z** plus a
sixth **F** token that sets a **flat hex fill** ([arxiv v3](https://arxiv.org/html/2504.06263v3)). Coordinates are
merged into single tokens via `x·w + y` over the 200×200 grid.

What it **cannot** emit, that `src/vectorize.py` produces today:
- **No gradients.** Our `_detect_gradient` emits real `<linearGradient>`/`<radialGradient>` paint; OmniSVG would
  flatten a gradient logo to a single solid color.
- **No strokes.** Our analytic line/divider path and Potrace stroke specs have no OmniSVG equivalent.
- **No opacity / no even-odd holes as a first-class feature.** (Counters come only from path winding it happens to draw.)

**Complexity budget.** Up to **>30k tokens** (vs the ~10k ceiling of prior LLM vectorizers)
([arxiv v3](https://arxiv.org/html/2504.06263v3)). No hard path-count cap; complexity scales with the token budget
(icons ~2.2k, illustrations ~8.1k tokens).

**Figma `createNodeFromSvg`.** The M/L/C/A/Z + flat-fill subset is the *cleanest possible* input for Figma import —
no gradients, filters, clip-paths, or exotic features to choke on. So OmniSVG output is highly Figma-editable, arguably
*more* import-robust than our gradient/stroke-carrying SVGs. But "importable" isn't the same as "correct": a flattened
gradient or dropped stroke imports cleanly and *looks wrong*. The editability win is real only for marks that were
flat-fill to begin with.

---

## 4. License fine print (commercial use)

There is a **split license**, and it's the crux of any commercial decision:
- **Model weights: Apache License 2.0** — permissive, commercial use allowed.
  ([HF card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B), [README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md))
- **Training dataset (MMSVG-2M): CC BY-NC-SA 4.0 — Non-Commercial, ShareAlike.**
  ([HF card](https://huggingface.co/OmniSVG/OmniSVG1.1_4B), [README](https://github.com/OmniSVG/OmniSVG/blob/main/README.md))

The Apache-2.0 grant on the *weights* is what governs running the model, and it permits commercial use. The NC clause
attaches to the *dataset*, which we would never redistribute. The open legal gray area — unsettled industry-wide — is
whether outputs of a model trained on NC-SA data inherit any restriction. For an ad-decompiler producing commercial
deliverables, this is a **flag for legal review, not a blocker**: the model license is clean, but a cautious posture
would note the training-data provenance before shipping OmniSVG output into paid client work. VTracer/Potrace (MIT/GPL
tools operating on the client's own pixels) carry no such ambiguity.

---

## 5. Integration sketch (where it would slot, read-only view of `vectorize.py`)

Current control flow in `vectorize_crop()` (`src/vectorize.py`):

```
_preprocess_crop
  → _analytic_straight_line_svg   (rules/dividers, gated)
  → if n_colors <= 1: try_potrace then try_vtracer
       else:          try_vtracer then try_potrace
  → contour fallback (single-color)
  every candidate scored by _evaluate_trace →
       _score_render  = 0.7·alpha_IoU + 0.3·color_mean   (>= role score_min, e.g. logo/icon 0.82)
     + path count <= role max_paths
     + _transparent_hole_recall >= hole_recall_min
  first candidate that passes the gate wins; else best-scoring; else caller keeps the raster crop
```

**The natural slot.** OmniSVG would be an additional candidate `try_omnisvg()` feeding the **same
`_evaluate_trace` render-back gate**. Crucially, that gate rasterizes the returned SVG and compares alpha/color to the
source — so it **automatically catches OmniSVG's reinterpretations**: a redrawn logo that drifts scores below
`score_min` and is rejected, falling through to VTracer or the raster crop. No new safety code needed; the existing
gate is exactly the reinterpretation guardrail.

**But the gate also predicts a low hit-rate.** The logo/icon `score_min` is **0.82** on a 0.7·alpha-IoU + 0.3·color
metric. OmniSVG's LPIPS 0.237 / reinterpretive redraw means many outputs will land *below* 0.82 and get thrown away.
OmniSVG would only "win" on the narrow set of crops where (a) VTracer produces genuine spaghetti, **and** (b) OmniSVG
happens to redraw faithfully enough to clear 0.82, **and** (c) the mark was flat-fill (no gradient/stroke to lose).
That's a small intersection.

**Placement constraints:**
- It must run **after** VTracer/Potrace (cheap, deterministic, gate-passing tracers win first); OmniSVG only fires when
  they fail the gate — otherwise you pay 40 s for nothing.
- At ~40 s/crop it **cannot be inline** in the batch hot path. It would have to be an **opt-in / offline pass** (e.g.
  `vectorize.omnisvg.enabled`, run only on crops flagged `role in {logo, icon}` that failed the deterministic gate).
- Architecturally it is **not an in-process function** like the current tracers — it's a 4B VLM that must run as a
  separate service (like SAM3/Gemma/Flux), VRAM-scheduled against them. The gate would call it over IPC/HTTP, adding a
  model-lifecycle dependency the vectorizer currently doesn't have.
- Cairo is already a dependency of both OmniSVG and our gate (`cairosvg`), so no new native surface there.

---

## 6. StarVector comparison — is OmniSVG strictly superseding it?

For our use case, effectively yes:
- On image-to-SVG, **OmniSVG 4B beats StarVector 8B** on every metric (DINO 0.899 vs 0.877, SSIM 0.906 vs 0.900,
  LPIPS 0.237 vs 0.238, MSE 0.034 vs 0.046) — at **half the parameters**. ([arxiv v3](https://arxiv.org/html/2504.06263v3))
- StarVector "demonstrates strong performance … from simple icon images" but "fails to generate complex SVGs" and
  shows "limited generalization" on illustrations/characters. ([NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/file/a510f05a574d4203ef3952973672fe2f-Paper-Conference.pdf), [StarVector](https://starvector.github.io/starvector/))
- StarVector's usable model is **8B**, i.e. an even worse VRAM fit than OmniSVG-4B on our 16 GB card.

So there's no reason to prefer StarVector here. The relevant point, though, is that **neither** neural model beats the
classical tracers on fidelity — the OmniSVG-vs-StarVector contest is for second place behind VTracer for our
faithfulness-gated pipeline.

---

## 7. Verdict & recommendation

**Now: no.** Adopting OmniSVG-4B today is a net negative for this box and pipeline:
1. **VRAM (16–17 GB) can't co-reside** with SAM3 + Gemma-12B + Flux on a 16 GB 5080 — it needs a whole model-swap
   orchestration layer just to load.
2. **~40 s/crop** is 100–1000× our deterministic tracers; unusable inline, awkward even offline at batch scale.
3. **Image-to-SVG fidelity is ~7× worse (LPIPS) than the VTracer we already run** — and our render-gate scores
   fidelity, so OmniSVG's reinterpretations get **rejected in the common case** anyway.
4. **No gradients/strokes** means it can't even represent the paint that `vectorize.py` already emits natively for a
   large share of logos.
5. **NC-SA training-data provenance** adds a legal-review flag that MIT/GPL tracers don't.

The single genuine benefit — compact, Figma-clean flat-fill paths — is not something the current gate can *reward*
(it measures pixels, not editability), so there is no automatic path for OmniSVG to earn its keep in the pipeline as it
stands.

**Later: conditional maybe.** The render-gate de-risks a future trial (it auto-catches reinterpretation failures), so
revisit — as an **opt-in, offline "clean-logo" pass** — when *all three* hold:
- (a) VRAM frees up — a second GPU, or a proven 4-bit/vLLM recipe for the SVG-token-extended checkpoint, or accepted
  model-swapping;
- (b) we've **measured** that VTracer's spaghetti on genuinely flat-fill logos/icons is actually costing designer
  Figma-edit time (i.e. the problem OmniSVG solves is a problem we have); and
- (c) we add an **editability-aware acceptance signal** (e.g. path-count/compactness) alongside the pixel gate, so a
  clean-but-slightly-drifted OmniSVG redraw can win on the crops where that tradeoff is actually wanted — otherwise the
  0.82 fidelity gate will reject it and the VRAM/latency buys nothing.

**Never** overstates it — the tech is real and the gate makes it safe to experiment — but there is no case for
integrating it into the RTX-5080 pipeline in its current, VRAM-contended state.

---

### Sources
- OmniSVG 1.1 4B model card — https://huggingface.co/OmniSVG/OmniSVG1.1_4B
- OmniSVG GitHub README — https://github.com/OmniSVG/OmniSVG/blob/main/README.md
- OmniSVG paper (arxiv v3 HTML) — https://arxiv.org/html/2504.06263v3
- OmniSVG NeurIPS 2025 camera-ready — https://papers.nips.cc/paper_files/paper/2025/file/a510f05a574d4203ef3952973672fe2f-Paper-Conference.pdf
- OmniSVG project page — https://omnisvg.github.io/
- ComfyUI-OmniSVG community node — https://github.com/A043-studios/ComfyUI-OmniSVG
- ComfyUI-Wiki OmniSVG overview — https://comfyui-wiki.com/en/news/2025-04-10-omnisvg-svg-generation-model
- StarVector project page — https://starvector.github.io/starvector/
- vLLM BitsAndBytes docs — https://docs.vllm.ai/en/stable/features/quantization/bnb/
- Qwen-VL bnb quantization walkthrough — https://medium.com/@krshranjith/quantizing-qwen-models-using-bitsandbytes-bnb-for-efficient-inference-b937286b9bbf
- vLLM CUDA-OOM forum thread — https://discuss.vllm.ai/t/torch-outofmemoryerror-cuda-out-of-memory/2420
- Local reference: `src/vectorize.py` (`vectorize_crop`, `_evaluate_trace`, `_score_render`, `_detect_gradient`)
</content>
</invoke>
