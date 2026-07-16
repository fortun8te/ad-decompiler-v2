# Feature Plan — what ad-decompiler-v2 is realistically missing (2026-07-16)

Planning doc only. No code changes. Prioritizes what to build next, grounded in the
repo's own evidence (audits, critic review, parity spec, run artifacts), not vibes.
Effort is sized for the **actual constraint**: one RTX 5080 16 GB local stack already
juggling SAM3 + gemma-4-12b + Flux GGUF + Big-LaMa. No "train a model" items in the
critical path.

---

## 0. The one brutal truth this plan is built around

Two documents disagree, and the disagreement is the whole story:

- **docs/INSPO-RECREATABILITY-AUDIT.md**: 70% strict-GREEN on 128 ads. But this is a
  **capability-on-paper** grade — 5 vision agents scoring what the pipeline *should* be
  able to do given its feature set, not what it actually emitted.
- **docs/CRITIC-REVIEW-2026-07-15.md §4**: on the **four freshest full-pipeline outputs**
  (009, 052, 002, 013), the measured strict standard ("near-identical AND genuinely
  editable") is **0/4**. The main headline is editable-with-correct-look in **zero**
  cases. The parity scorer agrees: 009 = 45.5/100, 041 = 62/100 (CODIA-PARITY-SPEC §6,
  target ≥ 90).

**Implication for prioritization:** the "Now" tier is dominated by correctness, honesty,
and measurement — not new capabilities. The seductive trap (called out explicitly in
CRITIC-REVIEW §4 "Mis-prioritizations") is shipping demo-differentiators (annotation
vectorization, glassmorphism, editable charts) while the GREENs aren't actually green.
Every YELLOW→GREEN feature is deferred behind making the GREENs real.

A second hard truth for the roadmap: **the pipeline has never round-tripped through real
Figma.** Zero `figma_export.png` exists in any run (FIGMA-VERIFY.md; CRITIC §3). Every QA
number to date describes the Python `render_preview.py` simulation, not what Figma
displays. The user's acceptance bar is "looks good IN Figma." We currently have no
evidence at that surface.

### State-change flag (verify before trusting anything downstream)

Live `config.yaml` now has **`peel.enabled: true`** (LaMa-pinned). The task brief and
PEEL-DECOMPOSITION.md both describe peel as default-**off**, and CRITIC §3 explicitly says
"wire peel **after** F1+F3 land." F1 (the 002 pixel-vaporizing layout bug) is **not
recorded as fixed** anywhere in the docs. So peel may have been switched on *before* its
stated prerequisite. **First action for the GPU-owning agents: confirm F1 is fixed, or
turn peel back off, before running the acceptance benchmark** — otherwise peel is layering
on top of a hierarchy stage that can still eat whole product panels.

---

## 1. NOW — highest payoff-per-effort, ship next

Ordered. Items N0–N2 are prerequisites for *believing any future number*; N3–N7 are the
"make the GREENs real" work. Nothing here needs a new model.

### N0. Get a real baseline: run the 16-image benchmark + one figma-verify loop
- **What:** Execute the PRODUCTION-READINESS Gates 2–4 that are all still PENDING —
  pytest → mock-E2E → doctor → `benchmark.py` on `benchmark_set` with `report.html` →
  bridge + `scripts/figma_verify.py --watch` + a human doing Import/Export on ≥5 fixtures.
  (This is GPU/human work owned by other agents — flagged here as the gating dependency,
  not something this planner runs.)
- **Payoff:** Every priority below is currently steering on unmeasured or simulation-only
  numbers. Until one `figma_qa.json` says `verified`, "beats Codia" is unfalsifiable.
- **Evidence:** PRODUCTION-READINESS.md (Gates 2–6 PENDING); FIGMA-VERIFY.md ("zero
  figma_export.png exists in any run"); CRITIC §3 ("figma_verify … is now the bottleneck",
  "Cheapest path needs no code"); CRITIC §4 ("Do not use 70% externally until a benchmark
  run scores it").
- **Effort:** M (no code; bridge + human round-trip ≈ 30 min once the box is warm).
- **Risk:** Low. The risk is *not* doing it — shipping green dashboards over broken files.
- **Depends on:** F1 verification (see state-change flag).

### N1. F1 + F3 — stop vaporizing product clusters; cap plate destruction
- **What:** F1: when a candidate that owns pixels (`target=="image"`, has `src`/mask) is
  promoted to a container (`layout.py:1277`), re-emit its raster as the group's first
  child — or forbid promotion for `photo-fragment`/`product` hosts. F3: hard-fail (or
  force-review) when `masked_fraction`/`changed_ratio` exceeds ~0.5 outside a full-bleed
  photo archetype (`pixel_diff.DEFAULT_THRESHOLDS` has no upper bound today).
- **Payoff:** Kills the 002 catastrophe class — the entire product cluster (~50% of canvas)
  replaced by gray smear, which *passed* structural checks (`element_recall 1.0`) and only
  failed QA by luck of an SSIM threshold. One 002 in a 16-image demo ends the pitch.
- **Evidence:** CRITIC F1 (root-cause chain traced to `layout.py:1277` + reconstruct drop
  + 85%-canvas removal mask), F3, §4 gap #2, Quick-wins #2 & #9 ("the big one; do before
  any peel work").
- **Effort:** M (F1) + S (F3).
- **Risk:** Medium — F1 touches the layout/reconstruct interplay; needs a regression test
  (image host + children). But it is the highest structural risk in the system.
- **Depends on:** nothing. **Blocks:** peel trust (N-tier), N0 acceptance run.

### N2. F2 — turn the anti-rasterization honesty gates back on
- **What:** Enforce leaf-accounting gates (`unexplained-raster-fallback`,
  `low-native-leaf-ratio ≥ 0.30`) whenever `leaf_accounting` exists, instead of gating them
  on `require_figma_export` (false in every run to date). `run_pipeline.py:600`,
  `pixel_diff.py:1163–1187`.
- **Payoff:** The repo's stated moat — "editable-ratio hard fails a good screenshot can't
  buy off" — is currently switched **off**. 052 passed QA at `native_leaf_ratio 0.2`. This
  is a ~1-line condition change that restores the discipline the project claims as its edge.
- **Evidence:** CRITIC F2, §2 metrics audit, Quick-wins #1.
- **Effort:** S (condition change + fixture updates). Pairs with F11 enum cleanup so
  `fallback=True` substitutions aren't misclassified as "unexplained" when the gate turns
  on (CRITIC F11).
- **Risk:** Low-medium — will start *failing* runs that used to pass. That is the point,
  but it must land together with N3 (font) or the failures will be all-headline and noisy.

### N3. Editable headline with a plausible font (F5 + F6) — the single biggest perceived-quality win
- **What:** Three changes, all cheap, all proven:
  1. **Match fonts on single-word crops, not lines**, with per-block majority vote
     (word boxes already in `ocr.json`). FONT-MATCHER-EVAL measured 100% vs 29% class
     accuracy on exactly the 009/052 failures — and it applies to the **existing**
     shape-matcher, no new model.
  2. **Default to Inter** on social/UI-screenshot archetypes; numeric-only lines can never
     pick a serif; match a real **Google display face** for headlines (Playfair-class) so
     Figma can actually insert it. CODIA-PARITY §7.1: "Inter is the answer, not font
     forensics."
  3. **Prefer refit over slice for the largest text** and route near-threshold text
     (ink_excess 0.6–0.75) to refit instead of shipping the wrong font (F6); add
     `region_color` to the text gate for headline/offer/price roles.
- **Payoff:** "Real text nodes everywhere, wrong font but instantly fixable" is Codia's
  core UX win, and this pipeline currently ships headlines as pixels or wrong-font. CRITIC:
  "converts more perceived quality per engineer-hour than anything else."
- **Evidence:** CRITIC F5 (Cambria/Gabriola at 0.94 fidelity), F6, §4 gap #1;
  FONT-MATCHER-EVAL §4c, §10; CODIA-PARITY §7.1–7.2.
- **Effort:** S–M. The word-crop + majority-vote change is the free 80%; Inter-default and
  Google-display matching are small policy additions.
- **Risk:** Low. Do **not** vendor the Lens weights (non-commercial license, FONT-MATCHER
  §7) — adopt only the method.
- **Depends on:** nothing. Land with N2.

### N4. Latency quick wins (F13) — make iteration and batch actually usable
- **What:** (a) Drop `structure.vlm_grouping.timeout_s` 60→15s + cache the negative result
  by planning fingerprint across harness rounds (today it hits the full 60s timeout and the
  result is **discarded** — 68% of 052's wall time). (b) Harness admission control: don't
  re-run the tail when `qa.ok` and the only repairs are OCR-recall shortfalls attributable
  to sliced/kept-in-photo lines; revert the round budget from live 3/2/2 to the example's
  2/1/1. (c) Cap OCR VLM chatter (`ocr_judge.max_lines`, skip proofread for scene-text).
- **Payoff:** 60–160s saved per image. Makes the 16-image benchmark, the harness, and any
  batch/ad-variant workflow (the actual use case) practical rather than a coffee break.
- **Evidence:** CRITIC F13 (measured from pipeline.log: 002 = 377s with 4 no-op harness
  passes; structure VLM = exactly 60s timeout twice), Quick-wins #3 & #4.
- **Effort:** S (mostly config + small patches).
- **Risk:** Low.

### N5. F7 + F9 — before/after bake arm; pill radius decomposition
- **What:** F7: delete the `or facts.get("before_after_pair")` arm in
  `reconstruct.py:255` so labels bake into photos only on actual containment (today a
  literal BEFORE/AFTER pair force-bakes uneditable pills even when they don't overlap a
  column — and it contradicts `archetype.py:194`). F9: fix rounded-rect decomposition so a
  pill stays one shape with corner radius instead of a rectangle + two parenthesis-shaped
  end-cap fragments (009 "Volgend").
- **Payoff:** Two of the most visible per-ad defects; F9 is a named P0 symptom that
  survived the last change wave.
- **Evidence:** CRITIC F7, F9, §0 (009/052 designer-view), Quick-wins #6.
- **Effort:** S (F7, delete one arm) + M (F9, shape radius extraction in fusion/merge).
- **Risk:** Low (F7) / Medium (F9 touches element_fusion).

### N6. Reporting honesty (F4 + F14 + F10) — you can't fix what the dashboard hides
- **What:** (a) Rename `editable_text_recall` → `intended_text_recall`; add
  `editable_text_fraction` = native-TEXT-leaf chars / source-OCR chars, and a **slice-count
  column** in `benchmark.md` (today 009's 8 slices are invisible). (b) Write `archetype`
  into `qa.json`/`runtime_report.json` so `benchmark.md` stops printing "—" for every
  archetype column. (c) Record `slice-budget-exhausted` truncation (`reconstruct.py:2013`
  silently drops regions past the cap of 8).
- **Payoff:** Restores trust in the one report a human reads. `editable_text_recall = 1.0`
  currently prints on 002 (preview destroyed) and 009 (all body copy sliced) — gameable by
  construction (a failing text becomes a slice, which removes it from the metric).
- **Evidence:** CRITIC F4, F14, F10, §2 metrics-honesty audit, Quick-wins #5 & #7.
- **Effort:** S each.
- **Risk:** Low.

### N7. F8 — wire the per-archetype text thresholds that already exist
- **What:** Forward `text_recall_min` (0.90–0.93 in every preset, `archetype.py:34–80`)
  into `pixel_diff` thresholds and repair defaults. Today the social preset traded a
  lenient visual bar (0.55) for strict text bars, but **only the leniency got wired** —
  009 passes at 0.78 visual / 0.733 text_recall against a 0.90 contract.
- **Payoff:** The text-heaviest archetype currently has the *weakest* enforced QA. This
  closes the loophole that lets wrong-font tweets pass.
- **Evidence:** CRITIC F8, §2.
- **Effort:** S. **Depends on:** N3 (or it will fail everything on font alone) and N6
  (archetype written to qa.json).

---

## 2. NEXT — needs a prerequisite or more evidence

### X1. Phase-2 reward: render-OCR text metric + per-crop VLM critic
- **What:** Build HARNESS-PHASE2 §1a first — OCR the rendered preview, align to source
  OCR, score per-line Levenshtein, classify defect ∈ {ok, clipped, duplicated, garbled,
  missing}. Then feed the VLM critic **per-region crops**, not full images. Calibrate the
  LPIPS/SSIM floors from the measured 16-fixture distribution.
- **Payoff:** The harness currently "steers on OCR-recall noise" — it re-ran 002 four times
  improving text_recall 0.130→0.148 while the products stayed vaporized; the gemma-12b
  critic found 1 of ~8 defects at full-image scale and scored the 002 inpaint severity
  "none." A grounded reward is what lets the loop actually converge instead of burning
  compute.
- **Evidence:** HARNESS-PHASE2 §1a; CRITIC F12 (LPIPS passes every image by 3×; critic
  blind at full scale), §4 ("Phase-2 reward expansion before calibration" is a
  mis-prioritization — calibrate first).
- **Effort:** M–L.
- **Depends on:** N6 (honest metrics) + N0 (measured distribution to set floors). **This is
  why it's Next, not Now** — building more reward machinery before calibration is wasted.
- **Where Codia research could change this:** if the parallel teardown shows Codia ships a
  usable per-region confidence signal (`detectionScore`, RESEARCH §6b), adopt its shape.

### X2. Peel: validate the move-test A/B and surface before/after panels
- **What:** Peel is enabled but the two validations that prove it *helps* are STILL OPEN:
  the move-test A/B (translate a peeled under-layer 40px, count revealed-hole reduction,
  gate ≥ 80%) and the 16-image `peel.enabled` on/off A/B on qa/editable-ratio/ghost-text.
  Separately, the leapfrog needs the detector to surface large photo panels (052's two
  before/after portraits) as **distinct** elements — today they live in a fragmented
  residual, so the seam-straddling product attributes to background and the panels are never
  completed.
- **Payoff:** Peel is the one genuinely uncontested differentiator (RESEARCH §6c: "Nobody
  has productized LayerD/Qwen-Image-Layered-style peel decomposition"). But right now it's a
  claim, not a measured win.
- **Evidence:** PEEL-DECOMPOSITION §6.4 ("Still open"), §9 (detection-granularity bounds);
  CRITIC §3.
- **Effort:** M (validation is CPU/Telea, 0 VRAM; panel detection is a fusion threshold
  change in `archetype.py`/fusion).
- **Depends on:** N1 (F1) confirmed fixed — peel feeds the hierarchy stage that F1 breaks.

### X3. Annotation → editable vector strokes
- **What:** Route annotation-role masks (marker arrows, X's, strikethroughs, scribble
  underlines) through vtracer → editable Figma VECTOR strokes, arbitrated by the existing
  render-back fidelity gate.
- **Payoff:** ~9 YELLOW ads → GREEN (014, 015, 016, 060, 078, 079, 083, 084, 091). Nobody
  ships this — Codia can't vectorize annotations either. Demo-visible, uses tools already in
  the stack.
- **Evidence:** INSPO-AUDIT gap #1 ("best ROI" among YELLOW drivers); RESEARCH §4 P2, §6d
  ("uncontested claims available").
- **Effort:** M.
- **Risk:** Low (vtracer + render gate already exist; failures fall back to slice).
- **Depends on:** Now tier — CRITIC §4 explicitly warns against demo-differentiators "ahead
  of F1/F5." This is the **first** YELLOW→GREEN feature to build once the GREENs are real.

### X4. Validate radial / multi-stop gradient emission end-to-end
- **What:** Multi-stop/radial gradients "landed 2026-07-15" per the brief. Confirm they
  actually survive design.json → plugin → Figma (the plugin has `paintFromSpec` gradient
  handling; needs a real round-trip). Extend to vignettes/glows if cheap.
- **Payoff:** ~7 YELLOW ads (018, 019, 055, 064, 123, 131, 138). Codia only "approximates"
  these — a documented weak spot.
- **Evidence:** INSPO-AUDIT gap #2; RESEARCH §6d.
- **Effort:** S–M (mostly validation of landed work).
- **Depends on:** N0 (figma-verify loop is the only way to confirm gradients render).

### X5. 90° rotated text as native rotated TEXT nodes
- **What:** Small fix in text rotation handling — 90° rotation *is* representable in Figma
  (rotated text node), unlike perspective/curved text.
- **Payoff:** Ads 085, 088 → GREEN. Cheap.
- **Evidence:** INSPO-AUDIT gap #9.
- **Effort:** S.

### X6. Batch / ad-variant workflow (the real distribution wedge for the actual user)
- **What:** The plugin imports **one** image at a time in `replace` mode
  (`code.js:21 importMode`); `benchmark.py` already batches on the pipeline side. Close the
  gap: process a folder → review the `report.html` grid → import the accepted ones. Wire the
  plugin's existing history panel to a run queue.
- **Payoff:** The identified market opening is "batch/usage-based pricing for ad-variant
  workflows" (RESEARCH §6d) — and it's Michael's own use case (Simpletics ad variants). This
  is a more realistic adoption wedge than any single-image quality gain.
- **Evidence:** RESEARCH §6d; plugin is single-image today (confirmed in code.js/ui.html).
- **Effort:** M.
- **Depends on:** N4 (latency — batch is untenable at 377s/image).
- **Where Codia research could change this:** the parallel teardown of Codia's plugin may
  reveal multi-frame/queue UX worth matching or deliberately diverging from.

### X7. Config reconciliation + doctor unknown-key warning (F16)
- **What:** One pass to reconcile `config.yaml` vs `config.example.yaml` (materially
  diverged: harness budget 3/2/2 vs 2/1/1, `visual_pass_ssim` 0.84 vs 0.9, gemma-12b vs
  e4b, whole subsystems documented-but-dead or live-but-undocumented incl. the 60s
  `structure.vlm_grouping` timeout). Add a doctor check that warns on unknown keys. Fix the
  README drift (claims PaddleOCR3/Surya primary; live is doctr+easyocr).
- **Payoff:** The example currently misleads about what's tunable; the live file hides
  what's active (e.g. the SAM prompt list is missing "inseparable product cluster" — directly
  relevant to 002's missed cluster).
- **Evidence:** CRITIC F16; README.md vs RESEARCH §4 P3 #12.
- **Effort:** M.
- **Risk:** Low.

---

## 3. LATER / WON'T — and why (including the seductive-but-wrong)

### W1. OmniSVG / neural vectorization — WON'T (now)
Seductive: "clean, compact, semantically-editable paths" for logos, the one thing vtracer
does badly. Wrong for this box: **16–17 GB VRAM** can't co-reside with SAM3 + gemma-12b +
Flux on a 16 GB card; **~40 s/crop** (100–1000× the tracers); image-to-SVG fidelity is
**~7× worse LPIPS than the VTracer we already run**; no gradients/strokes; and the
render-back gate can't *reward* compactness, so it rejects the reinterpretive output in the
common case anyway. Revisit only if VRAM frees up AND we've measured that vtracer spaghetti
actually costs designer edit-time AND we add an editability-aware acceptance signal.
- **Evidence:** OMNISVG-FEASIBILITY.md (full spike, "NOT NOW").

### W2. Glassmorphism / BACKGROUND_BLUR — WON'T (deliberately dropped)
The INSPO audit flagged it as a differentiator (~5 ads: 018, 022, 025, 057, 129) and Figma
supports it natively. But the brief states it was **tried and dropped**. Respect that.
Reopen only after the base is strict-GREEN and only if a clean detector (local blur +
translucency over busy bg) proves reliable — a false positive bakes blur over sharp content.

### W3. Editable charts / editable tables — LATER (someday-differentiator)
Seductive (a wishlist item nobody ships, RESEARCH §6d). Wrong to build now: it's a
different problem (data extraction, not layout reconstruction), huge effort, narrow payoff,
and slice is the **correct** v1 behavior — Codia flattens charts too. Ads 094, 107, 001.
- **Evidence:** INSPO-AUDIT gap #7.

### W4. Curved / arched / text-on-path — WON'T (industry limitation, not a gap)
Figma has **no** native text-on-path. An editable recreation is impossible in-platform for
*anyone*, Codia included (they slice these). Correct output = slice or vector outlines. Not
a gap vs Codia; do not spend effort trying to beat a platform limitation.
- **Evidence:** INSPO-AUDIT gap #6.

### W5. Lens font-model *weights* — WON'T ship
Non-commercial license is a hard blocker for a commercial-intent product. The *method*
(single-word crop → Google Fonts) is free and is in N3. If the free method on the existing
matcher proves insufficient after N0 measures it, the fallback is training a ~983-class
ResNet18 over rendered Google Fonts — CRITIC calls this "a weekend of 5080 time." Keep it as
a **Later** contingency, not a plan; the free method should capture most of the win.
- **Evidence:** FONT-MATCHER-EVAL §7, §10; CRITIC §3.

### W6. Declared non-goals stay non-goals — audit does not argue for promotion
Vector icons (Codia ships raster cutouts anyway, CODIA-PARITY §3), circular masks,
components/repeat-instantiation, and brand-font perfection remain deferred. The INSPO audit
surfaces **no** evidence that ad creatives need repeated-card componentization (ads rarely
have repeated cards), and CODIA-PARITY §1 shows Codia itself emits generic names and zero
components. The one place to watch: RESEARCH §6d lists "design-system output (spacing
tokens/components)" as a wishlist item — but that's a differentiator for *app-screenshot*
inputs, not ads. Keep deferred.

### W7. Outside-designer distribution — WON'T (structurally blocked, not a feature)
Be brutally honest: **no outside designer can adopt this today**, and no single feature
changes that. Adoption requires an RTX 5080 16 GB + SAM3 + LM Studio/gemma-12b + ComfyUI +
Flux GGUF ladder + Big-LaMa + a local bridge + a manually-imported Figma **dev** plugin
(not published to Community; `manifest.json` id `ad-decompiler-import`, network access to a
local bridge only). Codia is a $29/mo cloud SaaS with one-click install. The gating item for
outside adoption is a **hosted service or a packaged installer**, both large undertakings
orthogonal to pipeline quality, and arguably out of scope for a 16 GB local stack.

**Realistic distribution posture:** treat this as a **power-user / personal tool** for
Michael's own ad-variant workflow. The near-term "distribution" work that pays off is X6
(batch) + N0 (real in-Figma evidence) + the plugin polish already done — not Community
publication, which is pointless while the backend is a multi-hour expert install. If outside
adoption becomes a real goal, that's a separate strategic bet (hosting), not a line item
here.

---

## 4. Dependency graph / sequencing

```
N0 baseline ──────────────────────────────┐ (gates all trust)
                                           │
N1 (F1+F3) ──► peel trust (X2) ──► X3 annotation, X4 gradients (YELLOW→GREEN)
   │
   └──► N0 acceptance run

N2 (F2 gates) ──┐
N3 (fonts) ─────┼──► N7 (F8 archetype thresholds)
N6 (honest metrics) ──► X1 (Phase-2 reward, needs calibration from N0)

N4 (latency) ──► X6 (batch / ad-variant workflow)
N5 (F7+F9)  — independent, ship anytime
X7 config reconcile — independent, ship anytime
```

Critical path to a defensible "beats Codia on ads" claim:
**N1 → N3 → N0 (measure) → N2/N6/N7 (lock honesty) → X2/X3 (differentiate).**

## 5. Non-goals respected; nothing promoted

Reviewed every YELLOW driver and non-goal against the audit. **No non-goal is promoted.**
The audit's own strongest YELLOW→GREEN candidates (annotation vectors, gradients) are
sequenced as **Next**, explicitly behind the Now correctness work, per CRITIC-REVIEW's
warning that demo-differentiators ahead of F1/F5 are a mis-prioritization. Glassmorphism
stays dropped by the user's own decision.

## 6. Where the parallel Codia-plugin research could move a priority

- **X1 (reward):** if Codia exposes a usable per-node `detectionScore`/`surfaceArea`
  (RESEARCH §6b), adopt its confidence shape rather than inventing one.
- **X6 (batch) & plugin UX:** Codia ships a manual "Tag as Image" toolbar and a
  source-reference frame (CODIA-PARITY §1) because its auto-classifier isn't trusted — the
  teardown may reveal plugin affordances (manual override, multi-frame) worth matching or
  deliberately diverging from.
- **N3 (fonts):** Codia "always picks *something*" with no uncertainty flag (RESEARCH §6a).
  Our honesty discipline (F2) could be a *differentiator* — surface a "font uncertain" badge
  in the plugin rather than silently shipping a wrong font. Worth a small plugin item if the
  research confirms Codia never does.
- **Distribution (W7):** if the research shows Codia's new `codia-design-cli` + agent skills
  (Claude Code/Cursor) gain traction, a CLI/agent-skill wrapper around *our* pipeline could
  be a lighter distribution path than a hosted service — reconsider W7 then.
