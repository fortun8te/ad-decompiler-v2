# Codia Gap Analysis — Research Findings (2026-07-15)

Three-track research: full codebase audit, Codia AI pipeline teardown (sourced), and a
mid-2026 SOTA survey per pipeline stage. Local machine state verified (doctor READY,
all required models present). This doc is the source of truth for the fix roadmap.

## 1. Machine / model status (verified today)

- `doctor.py` → **READY**. CUDA 12.8 + torch 2.10 on RTX 5080, docTR GPU, SAM3 package +
  `C:/models/sam3.pt`, ComfyUI :8188 with all Flux Fill files (Q4/Q5/Q6 GGUF, t5xxl fp8,
  clip_l, ae), LM Studio :1234 with `google/gemma-4-12b` loaded (upgraded from e4b
  2026-07-15), vtracer + resvg present. Big-LaMa routes are actively used in benchmarks.
- Since resolved (2026-07-15): **tesseract** installed via winget, **cairosvg** installed
  in the venv. **potrace** deliberately NOT installed — vtracer + resvg are the
  production pair; potrace remains an optional monochrome backup only.
- Nothing critical is missing. The gaps are quality/validation, not installation.

## 2. Benchmark evidence (runs/golden-optimized-check, 5 images)

3/5 QA pass. Visual 0.587–0.979. Failures are **text-dominated**:

- **009** (dark X/Twitter post, visual 0.587): wrong font substituted (swash serif for a
  plain sans), broken letter-spacing on "UPFRONT", lost line breaks / merged paragraphs,
  visibly rotated text baselines, ghost duplicate text at the timestamp row (original not
  fully removed + re-rendered on top), "Volgend" pill lost its corner radius + white
  rectangle artifact, wrong retweet count (66→99 = OCR error shipped).
- **052** (curl-cream ad, visual 0.740): overall very close; headline tracking too wide,
  ghost "125ML" residue near the tube, edge fidelity 0.452.
- Pattern: photos/products/backgrounds reconstruct well; **text styling, text layout, and
  leak-through of un-removed originals are the failure mode**.

## 3. How Codia actually does it (teardown, high-confidence claims)

Codia is an **ensemble, not one model**: vision analysis (segmentation + type
classification) → OCR (+ dictionary post-correction) → font restoration (glyph-feature
match against a font DB, best-effort with fallback) → layout reconstruction → component
recognition. Everything is emitted into a **semantic JSON IR** (their "Visual Element
Schema": elementType, flexbox-like layoutConfig, styles) which a plugin renders to Figma
nodes. Sources: codia.ai/visual-struct, codia.ai/blog/image-to-figma-guide,
threehappyer Medium series.

What makes Codia output *feel* editable (and what reviewers confirm):

1. **Real text nodes everywhere**, even with wrong fonts — instantly fixable.
2. Semantic IR with typed elements → meaningful named frames, componentizable repeats.
3. **Conservative Auto Layout** — flex only where spacing/alignment evidence is strong;
   free-form stays absolute. (v2 already does this — parity.)
4. **Confidence-gated raster-slice fallback**: any region they can't reconstruct becomes a
   pixel-perfect image slice in the right place. Fidelity never collapses; the file always
   *looks* right. Their own troubleshooting docs admit buttons/charts fall back to images.
5. Vectorize-where-cheap: only simple flat icons get traced; photos stay raster. (Parity.)

Codia does NOT appear to do LayerD-style peel-and-inpaint layer decomposition — that is
an opening to beat them on ads, where overlap is common.

## 4. Gap list (v2 vs Codia), ranked by impact

### P0 — ship-blockers for "looks really close"

1. **No raster-slice fallback.** v2 ships wrong text/shapes rather than falling back to a
   pixel slice. This is Codia's single biggest fidelity trick and we lack it. Add a
   per-region confidence gate after QA: if a text block / shape / vector fails its local
   render-match, replace it with an image slice of the source (keeping the editable
   attempt in a hidden/secondary layer or as metadata). Fidelity floor guaranteed.
2. **Ghost/duplicate text leak-through** (009 timestamp, 052 "125ML"): removal mask misses
   + re-render on top = double text. Investigate removal-ledger coverage for low-contrast
   /small text and mask dilation on anti-aliased fringes.
3. **Flux Fill config contradicts documented-good settings.** Live config:
   `steps:8, cfg:1.0, guidance:30.0`, empty `photo_prompt`, `force_flux:true`
   (config.yaml:144-172). Documented proven settings: steps 20, guidance 3.5, no turbo
   LoRA (docs/HANDOFF-2026-07-13.md:132-134; also user memory). Empty prompt + huge
   guidance is exactly the documented "hallucinated ghosts" combo. A/B on the flux-ab run
   fixtures and lock the winner. Also reconsider `force_flux:true`: 2025-26 evidence
   (RePainter, OmniPaint) says diffusion inpainters hallucinate objects into removal
   masks while LaMa cleanly interpolates flat/gradient ad backgrounds — route by local
   texture complexity (entropy) instead of always-Flux.
4. **Font matching quality.** Current: shape-match against local system fonts +
   optional Google cache (text_analysis.py:1193-1280). Failures visible in 009. Upgrade:
   integrate **Lens** (github.com/mixfont/lens, ResNet18, matches to Google Fonts — which
   Figma can actually insert) or Storia font-classify; majority-vote per word crop; then
   **render-and-fit refinement** — render candidate text with skia/Pillow and optimize
   size/letter-spacing/line-height against the source crop via SSIM before emitting.
5. **Text block geometry**: preserve line breaks as authored (don't re-wrap), snap
   near-zero baseline rotations to 0 unless evidence is strong, fix paragraph merging.

### P1 — the reward signal (why the harness can't fix the above itself)

6. **QA reward is global SSIM** (pass ≥0.84) — authors themselves flag it misleading on
   dark/text-heavy ads (HANDOFF:110-112, ad9 oscillation). Phase-2 (render-OCR reward,
   per-archetype thresholds) is spec'd but unbuilt (docs/HARNESS-PHASE2.md). SOTA
   (UI2Code^N, ReLook): **VLM-as-judge beats CLIP/SSIM as repair signal**. Build the
   metric ladder: per-element crop SSIM (gate) → LPIPS global (gate) → Gemma/Qwen3-VL
   structured diff critique ("element, issue, fix" JSON) as the harness repair signal,
   ≤3 rounds.

### P2 — decomposition depth & editability

7. **Single union mask / one-pass inpaint** means elements under other elements are
   incomplete when moved. **LayerD** (CyberAgent, ICCV 2025, Apache-2.0,
   github.com/CyberAgentAILab/LayerD) is our exact peel loop (BiRefNet matting → LaMa →
   repeat), built for ad designs (Crello), PSD export. Integrate as the decomposition
   backbone for overlapping elements, keep current ownership map for the simple case.
   **Qwen-Image-Layered** (Apache-2.0, native ComfyUI, fp8/GGUF fits 16GB at 640px) as a
   second-opinion decomposer — the dormant `qwen` stage (config.yaml:114 disabled) is the
   natural seam.
8. **Hierarchy depth / components**: v2's tree is shallow and repeat detection only marks
   candidates. Add VLM grouping pass (image + boxes → nested groups with names +
   flex direction) and instantiate repeated cards as components.
9. **Logo/icon vector cleanliness**: vtracer is faithful but produces traced spaghetti on
   logos. **OmniSVG-4B** (Apache-2.0, fits 16GB) generates clean, compact, Figma-editable
   paths — use for icons/logos behind the existing render-back gate; keep vtracer for
   shapes. Emit detected linear gradients as native Figma gradient fills, never traced.

### P3 — hygiene

10. Install tesseract, potrace, cairosvg (doctor WARNs).
11. **End-to-end Figma desktop verification has never fully happened** (HANDOFF:98, 201 —
    image-in-mask unverified). The user's acceptance bar is "looks good IN Figma", so run
    the full bridge → plugin import → export → QA loop on the 16-image benchmark_set and
    fix what breaks. The 128-image "IMAGE AD INSPO" folder lives on the Mac
    (/Users/michael/Downloads/IMAGE AD INSPO); only 16 curated fixtures are local.
12. README/config drift: README claims PaddleOCR3/Surya primary; live config is
    doctr+easyocr. Either wire Paddle 3.x polygons (better char geometry for
    letter-spacing math) or fix the README.

## 5. Where v2 is already at or above Codia parity

- Conservative auto-layout policy (matches Codia's own approach).
- Real TEXT nodes with gradient/stroke/shadow support, font candidate lists + plugin-side
  render fitting.
- Structural QA that can't be bought by a good screenshot (ownership map, one-plate rule,
  editable-ratio hard fails) — Codia has no public equivalent; this discipline is a moat.
- Idempotent resumable pipeline, honest runtime_report fallback evidence, doctor gating.
- Text holes routed to Big-LaMa (correct per SOTA: diffusion fills regenerate glyph
  residue).

## 6. Research addendum (2026-07-15, second sweep — 4 agents)

### 6a. Codia output teardown (hands-on evidence)
Codia's real-world output is weaker than its reputation:
- Most detailed independent hands-on review scored layer STRUCTURE 1/5: "frames and
  groups don't follow good design practices", "no padding, margins, or design system —
  just a static file", "a designer could rebuild faster with cleaner logic."
- Icons/charts/progress bars routinely land as FLATTENED RASTER; vectorization is a
  manual, separately-credited user action, and they ship a manual "Tag as Image"
  toolbar because their own auto-classifier isn't trusted (dev quote on plugin page).
- No semantic layer naming — their own docs tell users to rename "Group 47" by hand.
- Font matching always picks *something* (no uncertainty flag); uninstalled fonts fall
  back to Figma defaults with only a label preserved.
- Pricing pain is loud: $29/mo (200 credits) / $59/mo (500 credits), users quote
  "$50/m is high for 20-25 images", one got charged $49/image on API overage.
- Codia self-assesses "75–90% of the way there"; 15–20 min manual cleanup per screen.

### 6b. VisualStruct schema (extracted from https://codia.ai/openapi.json — saved at
docs/reference-codia-openapi.json)
Fields our design.json IR should adopt:
- **Per-dimension sizing modes** `FIXED|FILL|FIT_CONTENT` (DimensionSpec) — this is what
  makes their auto-layout exports behave like real hug/fill frames, not pixel-locked.
- **Per-node `detectionScore` (0-1) + `surfaceArea`** as first-class metadata — feeds
  confidence-gated fallback and UI warnings.
- **Discriminated background union**: COLOR | IMAGE | LINEAR_GRADIENT(deg, stops).
- positionMode enum Normal|Absolute|Relative|Flex + flexAttributes per container.
- elementType taxonomy: Body|Layer|Image|Text|Component|Vector|Group; ComponentSpec
  carries sourceCode + configOptions inline.
- Codia treats **compositing layers** (image/layering "Magic Layers": background/
  subject/overlay RGBA stack) and the **structural tree** as two separate outputs.
- Stack reveal: their only proprietary model (`codia_image_v2`) covers erase/upscale/
  bg-remove/layering; ALL generative pixels are routed to third-party models (Gemini,
  GPT Image, Seedream, FLUX 2, Ideogram). Their moat is detection + IR, not pixels.
- No patents, no papers found. New distribution: codia-design-cli npm package + agent
  skills for Claude Code/Cursor.

### 6c. Competitive landscape (mid-2026)
- Codia is the benchmark **by incumbency, not superiority** — quality parity with
  image.to.design (divRIOTS) confirmed by Codia's own comparison blog.
- **Motiff is dead** (Figma lawsuit settlement; offline June 23, 2026).
- **Figma has NO native image→layers feature** — Make/First Draft/Design Agent/Code
  Layers are all generative or design→code (opposite direction). Risk window 6-12 mo.
- Google Stitch (free, Gemini): VLM-regenerative, exports named auto-layout Figma
  frames — improving fast but reinterprets rather than reconstructs.
- Refore.ai: claims pixel-perfect icon vectorization (our vectorize work competes here).
- Nobody has productized LayerD/Qwen-Image-Layered-style peel decomposition. That plus
  ad-specific tuning is the open leapfrog.

### 6d. Ad-niche opening (failure-modes sweep)
- Codia's own docs admit its weak points are exactly ad anatomy: "heavy photo overlays
  and stylized typography may need manual cleanup", "overlapping elements — hardest to
  separate; may need cropping and re-running", multi-stop/radial gradients only
  "approximated", product galleries become swap-me placeholders, charts flatten.
- Their flagship page's use cases are all app screens/dashboards/landing pages — no ad
  or banner example demonstrated anywhere (their galleries, reviews, Reddit, X, YT).
- **Uncontested claims available**: curved/rotated text (untested territory across all
  tools), drop shadows, dense overlapped compositions (our ownership-map + peel design
  targets exactly this), true editable product crops (we already emit masked image
  fills), batch/usage-based pricing for ad-variant workflows.
- Wishlist nobody ships: standalone credit purchase, multi-frame conversion, editable
  charts, design-system output (spacing tokens/components) instead of "a static file".

## 7. Recommended fix order

1. Flux settings A/B + entropy-based LaMa/Flux routing (small config/code change, big win)
2. Ghost-text removal-mask fix (dilation + low-contrast text coverage)
3. Raster-slice confidence fallback (the Codia fidelity floor)
4. Font pipeline: Lens Google-Fonts matcher + render-and-fit spacing/size optimization
5. Text geometry: line-break preservation, rotation snapping
6. Phase-2 QA reward (per-element SSIM + LPIPS + VLM critic) so the harness converges
7. Figma desktop E2E on all 16 fixtures
8. LayerD peel integration + Qwen-Image-Layered advisory revival
9. Components/hierarchy VLM pass; OmniSVG for logos
10. Hygiene installs + README/config reconciliation
