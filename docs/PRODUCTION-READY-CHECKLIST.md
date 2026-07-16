# Production-Ready Checklist — ad-decompiler-v2

Single source of truth. Supersedes nothing in `PRODUCTION-READINESS.md` (gates) or
`FEATURE-PLAN-2026-07-16.md` (roadmap) — this doc translates both, plus
`CODIA-PARITY-SPEC.md` and `CRITIC-REVIEW-2026-07-15.md`, into one blocker list with
measurable acceptance criteria and an owner. Written 2026-07-16 while six fix agents
(A/B/C/E/F/G) plus perf/critic agents are in flight against a fresh 50-finding audit of
`runs/codex-targeted-002a/002_attached_5885519ba4359843`.

---

## 1. Definition of "production ready"

This is a **personal power-tool for Michael's ad-variant workflow** (W7 in the feature
plan) — not a hosted product, not public distribution. "Production ready" means:

1. **≥14/16** benchmark_set fixtures pass the strict contract standard
   (`contract_pass: true` in `qa.json`, i.e. `native_text_ratio ≥ 0.90` AND
   `glyph_residue_clean` AND `placement_ok` AND `ssim_floor_ok`) — not the loose
   `qa.ok` flag, which the critic review showed passes broken runs (F4, F12).
2. **Zero hard failures** of the named catastrophic class: background-leakage,
   duplicate-content, unclean-background, inpaint-outside-mask, or product-cluster
   vaporization (the 002 F1 bug).
3. **At least 5 of the 16** have a real `figma_export.png` round-trip with
   `figma_qa.json` verdict `"verified"` — not just the Python `render_preview.py`
   simulation. Zero exist today (FIGMA-VERIFY.md, CRITIC §3).
4. **Wall-clock budget**: ≤120s/image steady-state (harness converges, no repeat
   passes with zero metric delta). Current worst case is 377s/image with 4 no-op
   harness rounds (CRITIC F13).
5. **Harness converges without babysitting**: no round that reruns the full pipeline
   for a metric delta of 0.00–0.02 (F13.2); plateau detection stops it.
6. **Dashboard honesty**: every metric a human reads in `benchmark.md`/`qa.json`
   means what its name says (F4, F14) — no metric that is gameable by construction.
7. **CODIA-PARITY score ≥ 90** on both the complex-UI (009) and flat-photo (041)
   templates (currently 45.5 and 62.0).

A release/checkpoint claim ("beats Codia", "ready to use daily") requires all seven.
Partial progress is fine to work from, but must never be reported as done.

---

## 2. Blocker checklist

Columns: **id** | **description** | **acceptance criterion** | **evidence source** |
**owner** | **severity**

### Plate / raster integrity (Mandate A)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| A1 | Layout container promotion drops host-image pixels (002 catastrophe) | Regression test: image host w/ children promoted to group still emits raster as first child; 0 instances of `target=="group"` with dangling `src` and no raster child across the 16-set | `src/layout.py:1277`; `runs/integration-full-16/002/layout.json` (`c_E003`) | A | SEV-0 |
| A2 | No ceiling on background-plate destruction | `pixel_diff` hard-fails when `masked_fraction`/`changed_ratio` > 0.5 outside full-bleed-photo archetype; 0 runs exceed this silently | `src/pixel_diff.py` `DEFAULT_THRESHOLDS` (no max today); CRITIC F3 (002 `changed_ratio 0.81`) | A | SEV-0 |
| A3 | Seam artifacts / clean-plate escape hatch (host-raster) | Visual seam check passes on ≥14/16; no visible boundary between inpainted plate and re-emitted host raster | Fresh 002 audit mandate A | A | SEV-1 |
| A4 | Product ghosting (baked pixels under a moved element leave a residue) | Editability Gate-5 check: move any product/photo element in Figma → no ghost visible underneath, on 3 diverse fixtures | PRODUCTION-READINESS Gate 5, item 3 | A | SEV-1 |

### Geometry / structure (Mandate B)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| B1 | Price placement wrong/drifted | `placement_ink_iou ≥` contract threshold (currently 0.3343 on 002, contract wants higher) for price-role text on ≥14/16 | qa.json `contract.placement_ink_iou`; contract `placement_ok: false` on latest 002 run | B | SEV-1 |
| B2 | Decoration-follows-text not honored (decorative elements detach from the text they anchor to when text reflows) | Spot check on 3 fixtures with decoration+text pairs: decoration bbox stays anchored after any text-box height change | Fresh audit mandate B | B | SEV-1 |
| B3 | Grouping / node-budget bloat | `node_budget` dimension in CODIA-PARITY: simple scene ≤ ~12 nodes (Codia 9), complex UI ≤ ~45 (Codia 38); `flatness` — groups only where Codia groups | CODIA-PARITY-SPEC §9, §8 (`node_budget` 0.10, `flatness` 0.05 weights); current 002 contract `node_budget: 0.72`, `flatness: 0.33` | B | SEV-1 |
| B4 | Rounded-shape decomposition still broken (pill → rect + 2 end-cap fragments) | 009 "Volgend" pill (and equivalents) render as one shape with corner radius, not 3 pieces | CRITIC F9 (P0 symptom, unfixed by prior wave); `element_fusion`/merge | B | SEV-1 |

### Fonts / text (Mandate C)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| C1 | Font-matcher miscalibrated on multi-word/line crops (serif picked for sans UI text at high confidence) | Single-word-crop + per-block majority-vote matching live; family accuracy ≥ 90% vs FONT-MATCHER-EVAL's measured 100%-vs-29% gap | CRITIC F5; FONT-MATCHER-EVAL §4c/§10; CODIA-PARITY §7.1 | C | SEV-1 |
| C2 | Text-box heights wrong (tight ink box, top-anchored → baseline drift) vs Codia's loose box + vertical-CENTER | Emitted text box ≥ lineHeight tall, `verticalAlign CENTER`, on all native TEXT nodes | CODIA-PARITY §2 (delta table row "Text box + align"); `schema.py` | C | SEV-1 |
| C3 | CTA / punctuation handling (trailing interpunct, em-dash, degree signs mis-OCR'd or dropped at line-split boundaries) | Numeric/punctuation re-verify on tight crop; interpunct restoration; 0 OCR literal corruption on 16-set spot check | CODIA-PARITY §6 delta row "OCR literals" (`'666'` vs `'66'`, `'weergaver'` vs `'weergaven'`) | C | SEV-2 |
| C4 | Mixed-weight lines not split into sibling TEXT nodes at weight boundaries | Weight-boundary split matches Codia's 3-sibling-node pattern (e.g. timestamp/count/label) on all detected mixed-weight lines | CODIA-PARITY §2a; CRITIC F5 | C | SEV-1 |
| C5 | letterSpacing noise (fitted tracking instead of snapping to 0) | `|letterSpacing| ≤ 0.5px` for ≥95% of native TEXT nodes when |fit| < ~2.5% of fontSize | CODIA-PARITY §6 delta row "letterSpacing" | C | SEV-2 |

### Harness brain (Mandate E)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| E1 | Repair ranking: deterministic checks outrank VLM opinions | Repair order always tries deterministic fixes (OCR re-verify, threshold-boundary re-fit) before spending a VLM-critic round; documented in harness config | Fresh audit mandate E; CRITIC F12 (VLM missed 002's vaporized products) | E | SEV-1 |
| E2 | No actionability gate — harness reruns full pipeline for repairs that can't move the metric (e.g. baked product microtext) | Admission control: skip rerun when `qa.ok` and only outstanding repairs are OCR-recall shortfalls attributable to `kept_in_photo`/sliced lines, or a patch that doesn't change effective config | CRITIC F13.2; 052 measured 83s wasted, 002 measured ~160s wasted | E | SEV-1 |
| E3 | Plateau logic missing/weak — 052 and 002 ran passes with metric delta 0.00 | Round budget reverts to example config values (2/1/1 vs live 3/2/2); plateau detector stops after 1 no-delta round | CRITIC F13 timing table (009: 2 rounds changed nothing; 052: 2nd full pass delta 0.0) | E | SEV-1 |
| E4 | Reward gate has no worst-local floor / anti-raster-gaming | `quality_flags` worst-local-window rule wired as a hard gate, not advisory (latest 002: `local-ssim-worst-region` 0.009 at x=704,y=512 flagged but not gating) | qa.json `quality_flags` (002 codex-targeted-002a run, worst window ssim 0.00907) | E | SEV-0 |
| E5 | Reward signal decorative at current calibration (LPIPS floor 0.30 vs measured 0.9954, everything passes 3x) | Floors recalibrated from measured 16-fixture distribution, not round numbers; VLM critic fed per-region crops, not full images | CRITIC F12 | E | SEV-1 |
| E6 | Anti-raster-gaming: text sliced instead of fixed removes it from `editable_text_recall` denominator | Metric renamed `intended_text_recall`; new `editable_text_fraction` = native-TEXT chars / source-OCR chars added and gated | CRITIC F4; N6 in feature plan | E | SEV-0 |

### Figma verification (Mandate F)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| F1 | Zero real `figma_export.png` round-trips exist in any run | ≥5 fixtures produce a real bridge→plugin-import→export-back `figma_export.png` with `figma_qa.json` verdict `"verified"` | FIGMA-VERIFY.md; PRODUCTION-READINESS Gate 4; CRITIC §3 ("now the bottleneck") | F | SEV-0 |
| F2 | Preflight sRGB / alpha-mask warnings not surfaced | `design_preflight.json` warns on non-sRGB profile and lossy alpha before import; 0 silent color-space drift on the 5-fixture round trip | Fresh audit mandate F; existing `design_preflight.json` artifact present in run dir | F | SEV-1 |
| F3 | Inbox staging hygiene (plugin import queue / history panel not cleaned between runs) | Re-import (replace mode) is idempotent — second import produces no orphans/dupes | PRODUCTION-READINESS Gate 4, last bullet | F | SEV-2 |
| F4 | Native-accounting honesty gates disabled in every run mode ever exercised | `require_native_accounting` no longer keyed to `figma.require_export` (false everywhere); `unexplained-raster-fallback` and `low-native-leaf-ratio ≥0.30` gates fire whenever `leaf_accounting` exists | CRITIC F2 (`run_pipeline.py:600`, `pixel_diff.py:1163-1187`); 052 passed QA at `native_leaf_ratio 0.2` | F | SEV-0 |

### OCR sanity (Mandate G)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| G1 | Product-label text counted in QA text-recall denominator (unfixable microtext dragging the score) | Product-label/scene-text OCR lines excluded from `text_recall` denominator; documented exclusion list in qa.json | Fresh audit mandate G; CRITIC F13.3 (002: 31/72 OCR lines were `kept_in_photo` microtext) | G | SEV-1 |
| G2 | Canonical disagreement counts between OCR engines not surfaced | doctr vs easyocr disagreement count reported per run; used to flag low-confidence text regions | Fresh audit mandate G | G | SEV-2 |
| G3 | VLM OCR judge failing / not adding signal | VLM OCR judge precision/recall measured against a hand-labeled subset; judge either fixed or disabled with the compute reclaimed | Fresh audit mandate G; CRITIC F13.3 (128s of OCR VLM chatter on 002, much of it wasted) | G | SEV-2 |

### Cross-cutting (not owned by a single letter mandate)

| id | description | acceptance criterion | evidence source | owner | severity |
|---|---|---|---|---|---|
| X1 | Archetype text-recall thresholds (`text_recall_min` 0.90-0.93) not forwarded to QA gates | `qa_config.pixel_diff_thresholds` and `repair.py`/`harness_critic.py` read `archetype_thresholds.text_recall_min` instead of hardcoded 0.85/0.80 | CRITIC F8; feature plan N7 | UNCOVERED (needs N3 fonts + N6 metrics landed first) | SEV-1 |
| X2 | `before_after_pair` bake-arm inverted (labels force-baked even without column containment) | Delete `or facts.get("before_after_pair")` arm in `reconstruct.py:255`; labels bake only on actual containment | CRITIC F7 | UNCOVERED (small, independent — no mandate claims it explicitly) | SEV-2 |
| X3 | `max_slices` truncation silent (regions past cap of 8 dropped, unrecorded) | Truncated rows appended to `report["skipped"]` with reason `slice-budget-exhausted`; count surfaced in `qa.structural` | CRITIC F10; PRODUCTION-READINESS Gate 3 ("every sub-threshold region resolved") | UNCOVERED | SEV-2 |
| X4 | `meta.fallback` three-state contract read three different ways by writers/readers | `fallback` becomes an enum (`raster-slice \| plate-passthrough \| fidelity-image \| residual-crop`); all readers updated; leaf accounting treats `fidelity-image` as explained-but-non-native | CRITIC F11 | UNCOVERED | SEV-2 |
| X5 | `editable_text_recall` still misleading even after rename (E6) unless benchmark.md gets a slice-count column | `benchmark.py` prints a slice-count column; 009's 8 slices become visible instead of hidden | CRITIC F4 | E (paired with E6) but benchmark.md change itself UNCOVERED | SEV-1 |
| X6 | `benchmark.md` archetype/preset columns always print "—" | `benchmark.py` reads `archetype.json` (already written every run) instead of qa/design-meta/reconstruction/runtime, which never carry it | CRITIC F14; feature plan N6 | UNCOVERED | SEV-2 |
| X7 | QA `ok:true` coexists with unresolved high-severity repairs (glyph residue) | Unresolved glyph residue promoted to hard fail or a mandatory benchmark.md column; 0 runs pass with outstanding high-severity `recommended_resume` | CRITIC F15; PRODUCTION-READINESS Gate 3 ("zero glyph-residue detections") | UNCOVERED | SEV-1 |

---

## 3. UNCOVERED section (ranked by risk)

Nothing in the six agent mandates (A/B/C/E/F/G) or the perf/critic agents currently
addresses these. Ranked by risk to shipping a false "production ready" claim.

1. **Benchmark-wide validation after fixes land (N0).** The single biggest risk: six
   agents are fixing 002 in isolation. Nobody has scheduled the full 16-image
   `benchmark.py` + `report.html` run that proves the fixes generalize and don't
   regress the other 15. This is the #1 item in the feature plan's Now tier and it is
   not owned by any in-flight mandate. **Risk: HIGH** — without it, "production ready"
   is unfalsifiable no matter how clean the fix agents' diffs look.
2. **F1 real Figma round-trip (5-fixture verified loop).** Same story — Mandate F
   covers preflight/staging/export-plumbing but the actual human-in-the-loop
   Import→Export cycle on 5 fixtures with `figma_qa.json: verified` needs a human at
   the keyboard, not just code. **Risk: HIGH** — "looks good in Figma" is the user's
   acceptance bar and there is zero evidence at that surface today.
3. **Config/README reconciliation (F16 / X7-adjacent).** `config.yaml` vs
   `config.example.yaml` have materially diverged (harness budget 3/2/2 vs 2/1/1,
   `visual_pass_ssim` 0.84 vs 0.9, missing SAM prompts like "inseparable product
   cluster" that are directly relevant to the 002 bug class). No agent mandate claims
   this. **Risk: MEDIUM** — stale config can silently undo a fix agent's work (e.g. if
   E3's plateau fix is config-driven but example/live disagree).
4. **Per-archetype `text_recall_min` wiring (N7/X1).** Explicitly deferred behind N3
   (fonts) and N6 (metrics) landing first — correct sequencing, but no mandate above
   owns it, so it needs to be scheduled as an explicit next step once C (fonts) and
   E6 (honest metrics) land. **Risk: MEDIUM**.
5. **VLM critic crop-feeding (part of E5, but the crop-generation plumbing itself is
   unassigned).** Mandate E covers ranking/actionability/plateau/reward-floor but
   "feed the critic per-region crops instead of full images" is a concrete code change
   (crop extraction + critic prompt rework) that could fall through the cracks between
   E5's calibration work and X-tier reward work. **Risk: MEDIUM**.
6. **Peel A/B validation (move-test + 16-image on/off).** `peel.enabled: true` is live
   but its two validations are still open, and peel depends on A1 (F1 in critic
   numbering) being confirmed fixed. No mandate above owns re-validating peel after A1
   lands. **Risk: MEDIUM** — peel could be silently making the A1 fix worse on
   overlap-heavy fixtures.
7. **CODIA-PARITY score re-run after fixes.** `scripts/codia_parity.py` exists and
   scored baselines (009: 45.5, 041: 62.0) but nobody has scheduled a fixed-code
   re-run against the ≥90 target. **Risk: LOW-MEDIUM** — good proxy metric, cheap to
   re-run, just needs to be on the verification-order list (see §4).

---

## 4. Suggested verification order (cheapest regression catch first)

1. **Single 002 replay** (the fixture the fix agents targeted). Fast, GPU-warm,
   confirms A1/A2/E4 (product cluster survives, worst-local-window gate fires
   correctly, no plate over-destruction) before spending time on anything broader.
   Check `qa.json contract_pass`, `quality_flags`, and diff a fresh `figma_import.json`
   against the current one for structural sanity.
2. **3-fixture spot check** — 002 (catastrophic-class regression), 009 (font/text
   mandate C, rounded-pill mandate B), 052 (harness mandate E, before/after X2). Covers
   every mandate's fix area with the minimum fixture count. Run `codia_parity.py` on
   009 and 041-class fixtures here too — cheap and directly answers whether the
   fixes moved the ≥90 target.
3. **Full 16-image benchmark_set** with `report.html`, figma-export **off** first
   (fast pass) to catch any cross-fixture regression from the mandate-A/B/C/E/G
   changes before spending the human-in-the-loop budget.
4. **Full 16-image benchmark with figma-export ON** (or as close to it as the 5-fixture
   human loop allows) — this is Gate 4 and the #1 UNCOVERED item above. Only run this
   after step 3 is clean; it's the most expensive step (human + bridge + plugin) and
   should not be spent on code that step 1-3 would have caught for free.
5. **Config/README reconciliation pass** (X7/F16) — do this in parallel with step 3,
   not after; a stale config can silently swallow a mandate-E fix (e.g. plateau budget)
   before step 3 even runs.

Do not report "production ready" progress based on any step before 4 completing with a
`verified` figma_qa.json on ≥5 fixtures — steps 1-3 are regression-catching, not
acceptance.
