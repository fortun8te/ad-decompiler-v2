# Ad-Remix Agent — Research Synthesis (2026-07-16)

Five parallel researchers covered the "decompiled reference ad → OUR ad (brand/product/copy) + N variations" vision: (1) agentic architecture, (2) brand injection, (3) product image swap, (4) copy generation + fitting, (5) variation generation + ranking. This is the combined design.

## The headline

**Nobody ships this.** Every commercial tool (Canva, AdCreative, Predis, Smartly/Celtra, Recraft) dodges the hard problem by requiring pre-declared template slots or locking generation style — none remixes an *arbitrary decompiled* layout. The academic systems closest to it (BannerAgency, MIMO, GameUIAgent) generate from scratch, not from a foreign source layout. The decompiler + remix combination is genuinely first.

**And the architecture already exists in this repo.** The harness loop (`harness_critic.py` / `harness_fixer.py` / `harness_loop.py` — Generator→Critic→Editor→Verifier with best-kept/no-repeat/plateau-stop) is the exact validated loop shape (independently converged on by GameUIAgent, arXiv 2603.14724). The remix agent is the same loop with a different reward: instead of "match the source pixels," it's "match the brand kit + copy brief while preserving the source's structural intent." Mostly a new reward function + operator set, not a new system.

## Core design rules (consensus across all five reports)

1. **Deterministic-first, VLM only where code can't.** VLM tasks: role inference on layers (which hex is the CTA plate), logo/content-aware placement planning (natural-language plan → deterministic coordinate pass, per arXiv 2512.12596), pairwise variant judging. Everything else is code: color substitution (dictionary lookup once roles are tagged), contrast repair (WCAG gate + L-axis-only nudge of the brand color), font metric-fitting, text auto-fit, resize, hard QC gates.
2. **Nobody trusts a generative model to compute production coordinates.** Reflow = deterministic constraint/auto-layout engine (Celtra/Smartly/Luban pattern, and Figma Auto Layout which the plugin already emits); the LLM only decides *what* changes and catches residual overflow.
3. **Edits are JSON patches against stable layer ids** (RFC 6902 style, per JSON Whisperer) on the existing `design.json` schema — never full-document regeneration, never pixel space.
4. **e4b as judge: pairwise only, never absolute scores.** Pairwise creative preference hits F1 0.96–0.98 while absolute scoring is weak, and small 7–8B models *beat* 13B at ranking (arXiv 2503.00046). Tournament + Bradley-Terry/Elo over a 6-dim rubric (BannerAgency's, ICC >0.92 with humans) + a label-legibility axis. VLMs can flag "something's off" but can't localize aesthetic issues (<0.20) — deterministic checks are the primary gate, VLM is a tiebreaker, human is final.
5. **Cap critique loops at K≤2** — GameUIAgent measured a hard quality-ceiling effect (r=−0.96); more reflection rounds don't pay.

## The remix loop

1. **Input:** `design.json` + brand kit JSON + product cutout(s) + copy brief (or "N variations").
2. **Role inference** (1 VLM call): tag each layer — bg / surface.plate / text.headline / text.body / text.cta / accent / logo / product — from geometry + colors + text jointly, not pixels.
3. **Remix Planner** (1 LLM call, text-only): emits a JSON-Patch diff against stable layer ids, validated against `schema.py` before apply.
4. **Deterministic apply:** recolor = fill swap via brand-kit role map + WCAG repair; copy = text swap; product = composite into the existing ownership mask/bbox (geometry stays fixed — sidesteps reflow for the riskiest element).
5. **Text fit** (deterministic): precompute glyph advances (fonttools/HarfBuzz); route by Figma autoresize mode — single-line = closed-form size solve, wrapping = binary-search + greedy break; adjust within a hierarchy band (headline stays ≥ the reference's headline:body ratio) → tracking ±2-3% → regenerate shorter copy (≤2 loops) → truncate with ellipsis, surfaced as "didn't fit." Generation itself uses soft length targets + generate-then-condense (MarketingFM pattern) — never one-shot exact character counts (<30% compliance).
6. **Render** via the existing `figma_bridge.py` flow.
7. **Hard QC gates** (code): WCAG contrast on every text/bg pair, bbox overlap/clipping, logo clear-space (stored as ratio of the logo's own dimension) + safe zones (central, avoid top/bottom in 9:16 / 4:5), off-palette leakage, product shadow-direction consistency (Farid forensics check: estimate light direction for product vs scene, threshold the angular delta).
8. **VLM critique** (≤2 rounds, rubric + pairwise) → **targeted repair** via one-defect-one-operator (recolor / refit_text / reflow / re-inpaint), reusing `harness_fixer.py`'s pattern.
9. **N variations:** run the planner N times with diversity instructions along the axes that drive performance — **hook/headline first** (most variance per marketer data), then aspect ratio (1:1, 4:5, 9:16), then CTA framing; layout shuffle later (VLM-agent pattern: textual layout plan → coordinates), color permutation last. Default N = 10–30 (Meta wants 20–50/product; TikTok fatigues in ~7 days — design for a weekly loop, not one-shot).

## Product swap specifics

- Decision tree: aspect match (±20% bbox delta) → drop-in at matched scale. Mismatch → **re-layout the slot first** (cheapest, safest), then **Blender re-render at the reference's estimated angle/light** (unique lever — no SaaS competitor has a 3D source; bpy camera/light scripting is standard, the text-to-camera glue is custom), generative reshape only as last resort with a label-legibility gate.
- Complexity gate: flat/studio background → parametric drop shadow only (Photoroom pattern — more reliable than generative). Lifestyle/directional light → relight + shadow synthesis (IC-Light-class relight; GPSDiffusion/DESOBAv2-class shadow models, code public; RdSOBA shows a rendered-pairs training loop works — relevant to the Blender assets).
- Known risk everywhere: generative relight can drift printed label text. Test on Simpletics bundle boxes early; QC rubric gets an explicit label-legibility axis. VRAM: relight models run sequentially with the VLM, not co-resident.
- Business case: lifestyle-context product placement measured at ~5% sales lift / up to 40% CTR lift vs flat shots (Amazon Ads).

## Brand kit schema (adopt)

Three-tier tokens per W3C DTCG v1.0: primitives (raw hex) → semantic roles (`bg.primary`, `surface.plate`, `text.on-dark`, `accent.cta`, `status.*`) → usage. Type: families with class tags (Google Fonts FAMILY_TAGS taxonomy is free and curated) + metrics (x-height/cap-height ratios for the size-adjust formula) + scale + weight map. Logo: variants, clear_space_ratio, min_size_px. Voice: tone tags + 3–5 few-shot pairs from past ads (the evidenced cheap default for short copy; fine-tune is unproven for this task shape). Taste constraints: whitespace floor, WCAG level.

## Build order

1. **Brand kit schema + role inference + deterministic recolor/contrast repair** — smallest slice that produces a visibly "our brand" remix.
2. **Copy swap + text fitting** (slot spec + generate-then-condense + deterministic fit).
3. **Product drop-in + parametric shadow**, complexity gate stubbed to drop-in-only.
4. **Hard QC gates + e4b pairwise ranking.**
5. **Aspect-ratio variants** via constraint resize (anchors/min-max/priority per layer, saliency-crop only for photo layers).
6. Later: relight/shadow synthesis tier, Blender re-render glue, layout permutation, Meta Insights feedback loop (own account only; Creative Breakdown gives per-asset attribution → bandit reweighting of axes).

## Riskiest unknowns

- Remix Planner quality on arbitrary layouts — build an injected-defect eval set before trusting it (mirrors HARNESS-PHASE2 critic validation).
- Product silhouette mismatch (bottle→box) sometimes forces a *layout* edit, not a pixel edit — untested anywhere in the literature; the planner must know which case it's in.
- Label fidelity under any generative pass — unverified for all models; test first.
- e4b judging reliability — mitigated by pairwise-only + deterministic gates carrying the real weight.

Full per-topic reports with sources: session research archive (architecture, brand, product, copy, variations — 2026-07-16).
