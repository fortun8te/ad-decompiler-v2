# Adversarial Critic Review — 2026-07-15

Scope: post-landing review of the ~21-agent change wave, judged against the repo's own
bar (AGENTS.md honesty rules, PRODUCTION-READINESS gates, "Codia-beating" goal).
Method: inspected real artifacts only — `runs/integration-smoke-6` (009, 052) and
`runs/integration-full-16` (002, 009, 013), original vs preview renders viewed at pixel
level, every qa/fallback/harness/critic/runtime artifact read, and every suspicious
metric traced to its implementation. Everything below cites a file:line, an artifact
path, or a measured number. Suspicions that did not survive verification (e.g. an
apparent mojibake in design.json names — an artifact of reading the file with cp1252,
the files are clean UTF-8) are not listed.

Verdict in one line: **the fidelity floor (raster-slice) works and QA correctly failed
the one catastrophic run, but the pipeline currently ships zero correctly-editable
headlines across all four inspected outputs while reporting `editable_text_recall = 1.0`
on every one of them, and the two honesty gates built to catch exactly that are disabled
in every run mode ever exercised.**

---

## 0. The four outputs, as a designer sees them

### 009 (dark X post) — QA pass, visual 0.830, composite 80.6
- "Volgend" pill: rounded pill became a **sharp-cornered rectangle**, bold→regular
  text, plus **two stray parenthesis-shaped fragments** at its ends (the pill's round
  end-caps were detected as separate elements: `c_E005` raster slice + `c_E006`
  vector). Fusion split one rounded rect into 3 wrong pieces.
- Headline "LAATSTE SITE WIDE SALE VAN 2026": wrong font (Candara, old-style numerals
  visible on "2026") shipped as editable text — `per_layer c_B4: region_ssim 0.137,
  ink_iou 0.389, ink_excess 0.715` (just under the 0.75 slice gate).
- Engagement row: "257" and "21K" rendered in **Cambria (a serif)** on a Twitter UI
  (`design.json c_B13/c_B14: fontFamily Cambria, claimed fidelity 0.945/0.941`),
  while "66"/"89" beside them are pixel slices — mixed fonts inside one row.
- "Post" title lost its bold weight.
- All 5 body-copy blocks (`c_B5,B6,B8,B9,B10`) are **raster slices** — the tweet text
  a user would most want to edit is pixels.
- QA `ok: true` with an outstanding **high-severity** repair: "glyph residue remains
  under 1 removed text region(s): c_B1" (qa.json `recommended_resume`), after 2 no-op
  harness rounds.

### 052 (curl-cream ad) — QA pass, visual 0.906
- Overall visually close; but the main headline is an **image fallback** (`c_B0`
  "Text (fallback)"), style says **Gabriola** (calligraphic) with render-fit winner
  Cambria at fit score 0.585 — the ad's primary copy is not editable and its recorded
  font is junk.
- Only **1 of 5 foreground leaves is native** (`native_leaf_ratio 0.2`,
  `editable_ratio 0.375`) — below the 0.30 QA floor, which did not fire (see F2).
- "Before"/"After" pill labels baked into the column photos (`kept_in_photo:
  ['After','Before',…]`) despite the archetype contract saying a literal
  BEFORE+AFTER pair must rebuild column copy (see F7).
- `leaf_accounting.unexplained_raster_count = 1` (`c_B0`) — flagged, gated nowhere.

### 002 (supplement bundle, integration-full-16, 384s) — QA fail (correctly)
- **Catastrophic**: the entire product cluster (whey bag + 2 jars, ~50% of canvas) is
  GONE — replaced by gray inpaint smear with orphan floating text ("UPFRONT",
  "WHEY MILKSHAKE", microtext slices at wrong positions). Headline "ALLE ESSENTIALS"
  is baseline-clipped with a ghost bar.
- Root cause chain (F1): `c_E003` (image candidate, 1025×1418 = the panel+products)
  was promoted to a **container group** by layout (`layout.py:1277
  host["target"] = "group"`); layout.json still carries its `src`, but the design
  compiler emits groups without raster fills, so the pixels vanish; the real product
  `c_E006` had already been dropped by reconstruct as
  `fully-contained-in-foreground-owner`; the removal mask then covered 1.77M px
  (85% of canvas, `background.changed_ratio 0.81, mean_change 80.5`) and the regional
  inpaint dutifully erased the products.
- QA failed it — but on `local-ssim 0.832 < 0.840` and `color-fidelity 0.479`, i.e.
  by luck of thresholds, not because any check knows "the largest element is missing":
  `element_recall = 1.0`, `editable_text_recall = 1.0`, `missing_assets: []`.
- The harness then spent **4 more full pipeline passes (163s)** improving
  text_recall 0.1296 → 0.1481 and never touched the actual failure; the VLM critic
  scored **inpaint severity "none"** (critic.json) and flagged only wrong glyphs.

### 013 (grüns pouch, integration-full-16) — QA pass, visual 0.941
- "snacks" lockup clipped to "…cks"; "Dietary Supplement" → "…ary Supplement";
  "Gut Health" → "…ut Health" (removal-ledger alpha crops truncated at region edges).
- "grüns" wordmark double-exposed look (masked-pixel fallback `c_B5` over imperfect
  plate, `font_confidence 0.26`).
- A circular clipping arc from the ellipse-masked badge (`c_E005 mask: ellipse`)
  cuts visibly through the pouch.
- Bottom pill text re-rendered in a visibly different font/weight — and it is the
  **only** native text leaf in the file (`native_leaf_ratio 0.444, editable_ratio 0.4`).
- Headline "We NEVER do this!" = raster slice. Again `editable_text_recall = 1.0`.

Across all four: **the main headline is genuinely editable-with-correct-look in zero
cases.** That is the gap between the dashboards and reality.

---

## 1. Findings, ranked

### SEV-0 — correctness / honesty of the system itself

**F1. Layout container promotion silently destroys host-image pixels (the 002 bug).**
- Evidence: `src/layout.py:1277` (`host["target"] = "group"` with no re-emission of the
  host's raster surface); `runs/integration-full-16/002.../layout.json` shows `c_E003`
  as `group` **with a dangling `src`**; design.json emits it as a bare group; preview
  shows the gray inpainted hole; reconstruct dropped contained product `c_E006`
  (`suppression_reason: fully-contained-in-foreground-owner`).
- Owner: layout hierarchy change (+ reconstruct `_flatten_photo_scene` interplay).
- Fix: when a candidate that owns pixels (`target == "image"`, has `src`/mask) is
  promoted to a container, re-emit its raster as the group's first child; or forbid
  promotion for `photo-fragment`/`product` hosts. Add a regression test with an image
  host + children. Add a QA ceiling on plate destruction (see F3).

**F2. The two anti-rasterization honesty gates are disabled in every run mode that has
ever been exercised.**
- `run_pipeline.py:600` sets `require_native_accounting = require_figma_export`
  (`run_pipeline.py:562` → `figma.require_export`, false in config.yaml and every run
  to date). `src/pixel_diff.py:1163–1187` wraps **both** `unexplained-raster-fallback`
  and `low-native-leaf-ratio` (min 0.30) in `if require_native_accounting…`.
- Measured consequence: 052 **passed QA** with `native_leaf_ratio 0.2` and
  `unexplained_raster_count 1`. The repo's stated moat — "editable-ratio hard fails
  that a good screenshot can't buy off" — is mostly switched off. Only the weak
  `editable_ratio_min 0.15` floor is live.
- Fix: enforce leaf-accounting gates whenever `leaf_accounting` exists (it always does
  now); keep only the figma-report gates keyed to `require_figma_export`.

**F3. No upper bound on plate destruction.**
- `pixel_diff.DEFAULT_THRESHOLDS` has `background_exact_match_max` (catches an
  untouched plate) and `background_changed_min 0.01` (catches a no-op) but **no
  maximum** on `changed_ratio`/masked fraction. 002 inpainted 85% of the canvas
  (changed_ratio 0.81) and tripped no background rule.
- Fix: hard-fail (or at minimum warn + force review) when masked_fraction or
  changed_ratio exceeds ~0.5 outside a full-bleed-photo archetype. Cheap, catches the
  entire 002 class regardless of root cause.

**F4. `editable_text_recall` is no longer a text-editability metric, and reporting
sells it as one.**
- `src/pixel_diff.py:783–830`: raster-slice text, wordmarks, lockups, and
  `foreground_raster` image text all count as "intentionally rasterized" → excluded
  from the denominator or credited via `slice_texts`. Deliberate and test-locked
  (`tests/test_raster_fallback.py:503`).
- Measured: **1.0 on all four fixtures**, including 002 (preview destroyed,
  `text_recall 0.148`) and 009 (all body copy sliced). `benchmark.md` prints it in a
  column headed "editable text" (benchmark.py:322) with **no slice-count column**
  ("raster clusters" = intentional clusters only; 009's 8 slices are invisible in the
  table).
- The metric answers "did any text we *chose* to keep editable go missing" — a fine
  regression guard, a dishonest headline number. As a QA gate
  (`editable_text_recall_min 0.80/0.86`) it is now nearly impossible to fail: any text
  the pipeline can't render well gets sliced, which *removes it from the metric*.
- Fix: (a) rename to `intended_text_recall` or similar; (b) add
  `editable_text_fraction` = chars (or ink px) in native TEXT leaves / chars in source
  OCR, report it in benchmark.md next to slice count; (c) keep the gate but on the new
  honest fraction per archetype.

### SEV-1 — visible quality, seam bugs of the `min_text_fidelity` class

**F5. Font-matcher confidence is miscalibrated and poisons every downstream gate.**
- 009: "257"/"21K" → **Cambria at claimed fidelity 0.945/0.941**; headline → Candara
  0.846; 052 headline → **Gabriola** (the exact failure documented in
  docs/FONT-MATCHER-EVAL.md as the motivating bug). With serif-for-sans scoring 0.94,
  `routing.min_text_fidelity 0.40` (archetype preset, `src/archetype.py:35` →
  `routing.py:129`) separates nothing.
- Owner: text_analysis shape-match + font_fit render-fit (landed "render-fit" change
  improved sizing/tracking — 052's tracking is fixed vs the July-13 audit — but family
  selection is still the loss driver).
- Fix (cheap, today): score on **single-word crops** with per-block majority vote —
  Lens's eval proved 100% vs 29% on exactly this corpus with word-level inputs; that
  finding applies to the current matcher too. Penalize serif/monospace candidates when
  the OCR-line stroke profile is sans (class_gate exists — `class_gate_min_confidence
  0.5` — verify why Cambria passes it on 3-digit crops).

**F6. Text ink gates ship visibly-wrong headlines that sit just under the thresholds.**
- `schema.py:75–91`: text rows with ink evidence are judged **only** on
  ink_iou/ink_excess (SSIM/color deliberately skipped). 009 `c_B4` (iou 0.389, excess
  **0.715**) and `c_B7` (0.378, **0.728**) cleared the `text_ink_excess_max 0.75` gate
  by 0.02–0.03 and shipped with the wrong font at region_ssim 0.14/0.26. Meanwhile
  `c_B5` (excess 0.81) was sliced. The gate boundary sits exactly in the middle of the
  "wrong font, same layout" cluster.
- Fix: for `role in (headline, offer, price)` add `region_color` to the text path and
  drop excess_max to ~0.60; or route near-threshold text (excess 0.6–0.75) to the
  font-judge/refit instead of shipping.

**F7. `before_after_pair` semantics are inverted between two agents' code.**
- `src/archetype.py:194–198`: a literal BEFORE+AFTER pair sets
  `suppress_descendants = False` — "rebuild all contained column copy… swappable
  columns".
- `src/reconstruct.py:255`: `if owner is not None **or facts.get("before_after_pair")`
  → the same fact **forces the labels to be baked into the photos**, even when they
  don't overlap any column. 052's Before/After pills are uneditable as a result.
- Fix: delete the `or facts.get("before_after_pair")` arm (bake only on actual
  containment), and honor the preset's rebuilt-copy contract.

**F8. Archetype preset thresholds are only half-wired (the `min_text_fidelity` class).**
- `text_recall_min` (0.90–0.93 in every preset, `archetype.py:34–80`) is enforced
  **nowhere** in pipeline QA: `qa_config.pixel_diff_thresholds` forwards only
  `visual_pass_ssim_min`/`edge_f1_min`/`editable_text_recall_min`; `repair.py:25` and
  `harness_critic.py:75` use their own hardcoded 0.85; `figma_verify.py:71` uses 0.80.
  Net effect for social_screenshot: the preset traded a lenient visual bar (0.55) for
  strict text bars — and **only the leniency got wired** (009 passed at 0.78 visual,
  0.733 text_recall vs the preset's 0.90 contract).
- Fix: forward `text_recall_min` into pixel_diff thresholds and repair defaults from
  `qa.archetype_thresholds`, same as editable_text_recall_min.

**F9. Rounded-shape decomposition (fusion) still broken — pre-landing defect not fixed.**
- 009 Volgend pill → rectangle (radius lost) + 2 end-cap fragments rendered as
  parentheses. The July-15 gap analysis listed this exact pill ("lost its corner
  radius + white rectangle artifact") as a P0 symptom; the landed detection-fusion and
  merge-dedup changes did not fix it. Owner: element_fusion/merge + shape radius
  extraction.

**F10. `max_slices` truncation is silent.**
- `src/reconstruct.py:2013`: `failing = failing[:max_slices]` — regions beyond the cap
  of 8 are neither sliced nor recorded in `skipped`/`dropped`. 009 hit exactly 8.
  PRODUCTION-READINESS Gate 3 requires "every sub-threshold region resolved by
  raster-slice fallback" — currently unauditable.
- Fix: append truncated rows to `report["skipped"]` with reason `slice-budget-exhausted`
  and surface the count in qa.structural.

**F11. `meta.fallback` is a three-state contract read three different ways.**
- Writers: `routing.py:157,317` and `merge_layers.py:357` write `True`;
  `reconstruct.py:1881` writes `"raster-slice"`; `_apply_drop_mutation` writes
  `"plate-passthrough"`.
- Readers disagree: `repair.py:559` skips slice-gating for **any truthy** value (so a
  `fallback=True` text→image substitution is never re-gated by repair), while
  `reconstruct.py:1992` re-gates everything except the literal `"raster-slice"` — the
  comment says that divergence is deliberate for reconstruct, but repair's blanket skip
  contradicts it; and leaf accounting classes `True` as **unexplained** raster
  (052 `c_B0`) even though it carries a full substitution explanation in meta.
- Fix: make `fallback` an enum (`raster-slice | plate-passthrough | fidelity-image |
  residual-crop`), update the three readers, and make leaf accounting treat
  `fidelity-image` as explained-but-non-native.

### SEV-2 — reward signal & wasted compute

**F12. The Phase-2 reward is decorative at current calibration.**
- LPIPS: 052 `lpips_similarity 0.9954` vs floor **0.30** (squeeze net at max_edge 256
  cannot discriminate ad-scale defects; every image passes by 3×). The reward's LPIPS
  weight (0.15–0.35) contributes ~zero gradient.
- `reward_local_ssim_min 0.45` vs measured 0.74 on a file with wrong-font headline.
- VLM critic (gemma-4-12b): on 009 found **1 of ~8** designer-visible defects
  (qa_critique.json: only the Volgend gray box); `anomalies.json: []` on the same
  file; on 002 scored **inpaint severity "none"** while the inpaint had erased the
  products. The gap analysis' premise "VLM-as-judge beats SSIM as repair signal" is not
  achieved by this model at this size — the harness is steering on OCR-recall noise.
- Fix: calibrate floors from the 16-fixture distribution (set floors at the failing
  runs' measured values, not round numbers); give the critic per-region crops instead
  of full images (it caught the pill when the defect was large); add a
  missing-content check that doesn't need a VLM: SAM observation boxes with no
  surviving owner AND no plate coverage → hard fail (would have caught 002 alone).

**F13. Where the seconds actually go (measured from pipeline.log timestamps).**

| run | total | breakdown |
|---|---|---|
| 009 smoke | 60s | 37.2s pipeline + 2 harness rounds (11.4s + 11.4s) that changed **nothing** (identical ssim 0.7817, QA was already ok) |
| 052 smoke | 177s | 94.0s + 83.3s full re-run with metric deltas **0.0**; of each pass, **60.0s is `structure`** (VLM grouping) |
| 002 full-16 | 377s | 221s first pass (**128s OCR**: doctr+easyocr + vlm-judged 32 lines + vlm-corrected 44; **60s structure**) + 4 harness passes (31+27+50+47s) for text_recall +0.019, final QA still fail |

Top 3 cheap latency wins:
1. **Structure VLM grouping hits its full 60s timeout and is discarded** (052 and 002:
   exactly 60s merge→structure both passes; `scene_intent.json planner:
   "layout.infer"` = the VLM result never landed; config
   `structure.vlm_grouping.timeout_s: 60`). Drop the timeout to ~15s, cache the
   negative result by planning fingerprint across harness rounds, and log the
   failure reason. Saves 60–120s on affected images — 68% of 052's wall time.
2. **Harness round admission control.** Don't re-run the tail when `qa.ok` and the
   only repairs are (a) `ocr rerun` for recall shortfalls attributable to
   kept_in_photo/sliced lines (compute attribution first — 052's 0.75 vs 0.85 is
   baked product microtext, unfixable by rerun) or (b) repairs whose patch doesn't
   change effective config. Live config also doubles the example's budget
   (max_rounds 3 / repair_iterations 2 / plateau_rounds 2 vs 2/1/1) — revert. Saves
   83s on 052, ~160s on 002, 23s on 009.
3. **Cap the OCR VLM chatter.** 002 spent ~2 min on VLM judge+proofread for 72 lines,
   31 of which ended `kept_in_photo` (product-label microtext that never becomes a
   layer). Skip proofread/judge for lines already classified scene-text, and set
   `ocr_judge.max_lines`. Also: gemma unload/load thrash costs ~9s per reconstruct
   round (5× in 002 ≈ 45s) — skip the reload when the next stage needs it evicted
   again.

### SEV-3 — reporting & config drift

**F14. benchmark.md archetype/preset columns are always "—".**
- `benchmark.py:237` reads `archetype` from qa/design-meta/reconstruction/runtime —
  no stage writes it to any of those; `archetype.json` (which exists in every run) is
  never consulted. The per-archetype QA story is invisible in the one report humans
  read. One-line fix.

**F15. QA `ok:true` coexists with outstanding high-severity repairs.**
- 009: `recommended_resume` = glyph residue under c_B1, severity high, after the
  harness declared `qa_ok_after_repairs`. Gate 3 requires "zero glyph-residue
  detections"; QA does not gate it (`reconstruction.stats.text_residual.flagged`
  → repair suggestion only). Promote unresolved glyph residue to a hard fail or a
  benchmark column.

**F16. config.yaml vs config.example.yaml have diverged materially.**
- Example documents whole subsystems the live config doesn't carry (vlm
  `segment_filter`, `element_propose`, `scene_text`, `font_judge`, ocr
  `fallback_engines`, inpaint `mask_dilate`/`quality`, vectorize `score_min`/
  `max_paths`/presets) — those stages run on **code defaults** (they visibly ran in
  the logs), so the example misleads about what's tunable and the live file hides
  what's active.
- Live-only keys the example never mentions: `structure.vlm_grouping` (the 60s
  timeout!), `small_icon`, `small_region_refine`, `numeric_verify`, `render_fit`,
  `consensus`, `fallback:` block.
- Direct conflicts: harness budget 3/2/2 vs 2/1/1; `visual_pass_ssim 0.84` vs 0.9;
  `merge.nested_max_area_ratio 0.70` vs 0.62; vlm model `gemma-4-12b` vs `e4b`;
  `qa_ocr` nested under `figma:` live but documented at root (run_pipeline reads root
  with default True — works by accident; `figma_verify.py:635` reads both).
- The live SAM prompt list is missing the example's `inseparable product cluster`,
  `product shot`, `card`, `chart`, `screenshot` prompts — directly relevant to 002's
  missed product cluster.
- Fix: one agent pass to reconcile both files and delete dead keys; add a doctor check
  that warns on unknown/undocumented keys.

---

## 2. Metrics honesty audit (direct answers)

- **editable_text_recall**: not honest as named/reported — see F4. It reported 1.0 on
  a run whose preview is garbage (002) and on runs whose entire body copy is pixels
  (009) or whose headline is an image (052, 013). Gameable by construction: failing
  text becomes a slice, which removes it from the metric.
- **native_leaf_ratio**: honestly computed, correctly designed (leaves only,
  background excluded) — but its gate never fires (F2). Currently a decorative number
  in benchmark.md.
- **unexplained_raster_fallbacks**: doubly broken — the gate is off (F2), and the
  classifier mislabels explained fidelity fallbacks (`fallback: True` with full
  substitution metadata) as "unexplained" (F11), so when it is turned on it will fail
  runs for the wrong reason.
- **local-ssim 0.550 bar (social preset)**: not calibrated — it is a floor chosen so
  dark tweets stop failing on global SSIM, with the compensating per-archetype text
  bars (`text_recall_min 0.90`) left unwired (F8). Result: the text-heaviest archetype
  has the *weakest* enforced QA of all. 009 passes at 0.78 with 7 wrong-font/degraded
  text regions. The three-source threshold stack (code default 0.9 → config 0.84 →
  preset 0.55) also means the effective bar depends on which stage asks (qa_reward's
  floor resolution differs from pixel_diff's).
- **element_recall / element_survival**: measures survival of *kept proposals*
  (16/16 in 009), not coverage of source content — a layer that was never proposed, or
  dropped as "contained", is invisible (002: recall 1.0 with the products missing).
  Needs a coverage complement (SAM/photo-region area with no owner and no plate
  match).

## 3. Unwired work (direct answers)

- **peel_decompose** — *not wiring it today was the right call*, but for a different
  reason than the docs give: the ownership/layout layer that peel would feed is
  currently eating whole photo clusters (F1). Wire peel after F1+F3 land, exactly via
  the doc's 3-hunk diff (stage between elements and merge, QwenLayer shape, zero
  merge/reconstruct changes), default `peel.enabled: false`, auto-gated on overlap
  evidence (pairwise non-text IoU > 0.05). Validate offline first on the 5
  overlap-heavy inspo fixtures (005/008/013/059/132) comparing qa deltas.
- **figma_verify** — *not running it is the wrong call and is now the bottleneck.*
  Zero `figma_export.png` exists in any run; `preview_drift` is unmeasured; and the
  native-accounting honesty gates are keyed to `figma.require_export` (F2). Cheapest
  path needs **no code**: bridge + `scripts/figma_verify.py --watch` + a human doing
  Import/Export on 5 fixtures (~30 min), then set `figma.require_export: true` for
  acceptance runs. Until this happens, every QA number is about the Python simulation.
- **qwen stage (disabled)** — correct call. 16GB VRAM is already juggling
  gemma-4-12b + SAM3 + Flux GGUF with per-stage evictions; a second diffusion
  decomposer adds contention and a second opinion nobody arbitrates. Revisit only as
  peel's cross-check, behind the example config's `failure_cooldown_s`.
- **Lens font matcher (license-blocked)** — correct not to ship the weights, but the
  eval's *findings* are free and unadopted: (1) match on single-word crops, not lines;
  (2) per-block majority vote. Both apply to the existing shape-matcher today and
  target the exact Gabriola/Cambria failures reproduced in this review (F5). In
  parallel: request commercial terms from mixfont, or synth-train an equivalent
  (983-class ResNet18 over rendered Google Fonts is a weekend of 5080 time; the
  method is not encumbered, only their weights).

## 4. The "better than Codia" claim

The INSPO audit's 70% strict-GREEN is a *capability-on-paper* grade. The measured
strict standard ("near-identical AND genuinely editable") on the four freshest full
pipeline outputs is **0/4**: 009 (wrong fonts + sliced body), 052 (headline not
editable), 013 (headline sliced, clipped lockups), 002 (broken). Do not use 70%
externally until a benchmark run scores it.

Three highest-leverage gaps, in order:

1. **Editable headline with a plausible font.** Codia's core UX win is "real text
   nodes everywhere, wrong font but instantly fixable". This pipeline's headline
   outcome today is pixels (slice/fallback) or wrong-font-shipped. Fix = F5 + F6 +
   headline-role slice aversion (prefer refit over slice for the largest text). This
   converts more perceived quality per engineer-hour than anything else.
2. **Structural integrity under the new hierarchy code** (F1 + F3 + coverage metric
   from §2). One 002 in a 16-image demo ends the "beats Codia" pitch regardless of
   the other 15.
3. **Real-Figma E2E evidence** (figma_verify loop + Gate 4/5 cleanup-time
   measurement). "Looks good in Figma" is the user's acceptance bar and there is
   currently zero evidence at that surface — including whether the plugin's font
   render-fit behaves like the Python preview's (fontCandidates are consumed by both
   `render_preview.py:254` and `code.js:588`, but nothing has ever compared them).

Mis-prioritizations in the current roadmap:
- **Phase-2 reward expansion before calibration.** The ladder exists but gates
  nothing (F12): floors pass everything, the 12B critic can't see defects at
  full-image scale. More reward machinery is wasted until floors come from measured
  distributions and the critic gets crops.
- **Demo-differentiators (annotation vectorization, glassmorphism, OmniSVG) ahead of
  F1/F5.** They convert YELLOWs on the inspo corpus, but the GREENs aren't actually
  green yet at the strict standard.
- **Peel as "the leapfrog"** is correctly deferred, but the docs should stop calling
  the ownership map "correct and cheap when elements do not overlap" while F1 can
  vaporize a non-overlapping product panel.

## 5. Quick wins list (ordered by value/effort)

1. F2 — enable leaf-accounting gates unconditionally (1-line condition change + fixture updates).
2. F3 — plate-destruction ceiling in pixel_diff (few lines).
3. F13.1 — vlm_grouping timeout 60→15s + negative cache (config + small patch).
4. F13.2 — harness admission control + revert round budget to example values (config).
5. F14 — write archetype into qa.json/runtime_report so benchmark.md stops printing "—".
6. F7 — delete the `or before_after_pair` bake arm in reconstruct.py:255.
7. F10 — report slice-budget truncation.
8. F5 (first step) — single-word crops + block majority vote in the existing matcher.
9. F1 — group-promotion raster re-emission + regression test (the big one; do before any peel work).
10. F16 — config reconciliation pass.
