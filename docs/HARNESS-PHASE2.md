# Harness Phase 2 — External, Grounded, Self-Correcting QA Loop

Status: SPEC (execute after the 5-agent phase-1 fixes land and are integrated).
Owner: orchestrator (main). Phase-1 Agent 4 already delivers the in-loop convergence
guarantees (best-kept / rollback / no-repeat / plateau-stop) and `src/vlm_anomaly.py`;
this spec builds the *reward signal* and *diagnosis→repair* architecture on top.

## Why (grounded in observed failures + prior art)

Observed: the old loop bounced `ssim 0.42 ↔ 0.58` and "repaired" 13–14× with zero net
improvement; ad9 (a tweet) scored `ssim=0.45` while being structurally near-perfect.

Root problem: the harness is **blind** (global SSIM is a bad reward for text/dark UIs),
**forgetful** (fixed in phase 1), and **blunt** (re-runs whole stages, not the bad element).

Prior-art consensus (2026):
- Intrinsic self-correction without an *external* signal is unreliable — use an independent
  critic + tool checks. (Can-LLMs-Correct-Themselves 2510.16062; CRITIC / TACL survey.)
- Refine from **localized visual differences** across a fixed dimension set, reward =
  *reduction in discrepancy*, not absolute score. (VisRefiner 2602.05998; ReLook 2510.11498;
  UI2Code^N 2511.08195.)
- Separate roles: Generator → Critic → Editor → **Verifier** (a distinct step that confirms
  each edit reduced discrepancy). (VLM-critic T2I pipelines; IVT 2606.13156.)
- Structure the flow around **explicit tests / gates**, incl. generated ones. (AlphaCodium
  2401.08500 — 19%→44% pass@5 with ~100 calls.)
- Spend refinement budget where **confidence is low**. (CoRefine 2602.08948.)
- Guard against reward-hacking / degenerate empties (README already fears "empty
  decomposition scores perfectly"). (ReLook periodic critic-off.)

## Architecture: 4 separated roles

```
pipeline output (design.json + preview.png)
  → CRITIC   : diagnose defects, per-dimension, per-element  → defect list
  → EDITOR   : apply ONE targeted repair operator per defect → new design.json
  → VERIFIER : re-score ONLY the affected region             → keep | rollback
  → LOOP     : best-kept, no-repeat-failed, plateau-stop (phase-1 Agent 4)
```

Acceptance is decided by explicit per-element GATES, not a single global SSIM.

## 1. Reward signal redesign (the core lever)

Replace "global SSIM pass/fail" with a weighted, grounded, per-dimension score.
Touch-points (confirm after agents land): `src/pixel_diff.py`, `src/qa_config.py`,
new `src/qa_text.py`, `src/harness_critic.py`.

### 1a. Render-OCR text-correctness metric (highest value, no Figma round-trip)
- OCR the **rendered preview** (reuse `src/ocr.py`), align lines to source OCR, score each
  line by normalized Levenshtein.
- Emit: `text_score` (mean), plus per-line `{source, rendered, sim, defect}` where defect ∈
  {ok, clipped (rendered is a strict prefix / trailing words missing), duplicated (same
  string appears N>expected times), garbled (low sim, not prefix), missing}.
- This DIRECTLY catches ad9 clipping (prefix match) and ad2 duplication (repeat count)
  without the plugin export step. It becomes the dominant term in acceptance.

### 1b. Element/region-level scoring
- Reuse the existing 16×16 `per_region` deltas; additionally score per LAYER bbox:
  `region_ssim`, `region_deltaE`, coverage. Localizes a single bad element instead of
  dragging one global number.

### 1c. Per-archetype thresholds
- Classify the ad up front (`social_screenshot | product_hero | text_poster | mixed`) via a
  cheap VLM call or heuristics (dark bg + engagement icons ⇒ social). Store on the run.
- Acceptance thresholds are per-archetype: a tweet must NOT need 0.90 visual SSIM; it must
  need high `text_score` + correct structure. Config: `qa.thresholds.<archetype>`.

### 1d. Anti-degenerate guard (anti reward-hacking)
- Hard-fail acceptance if: element_count << detected_count, or any required element missing,
  or background == source (README's existing check), regardless of visual score.

### Acceptance = weighted gate
```
accept = text_gate AND structure_gate AND visual_gate AND not degenerate
  text_gate:      text_score ≥ τ_text[archetype]  AND  no line defect in {clipped,duplicated,missing}
  structure_gate: no unresolved critic defect of severity ≥ high
  visual_gate:    region-weighted visual_score ≥ τ_vis[archetype]
```

## 2. Defect taxonomy (VisRefiner 6 dimensions) + repair operators

Fixed enum shared by critic, anomaly pass, and repair. Each defect maps to ONE operator.

| dimension  | example defect                     | repair operator            |
|------------|------------------------------------|----------------------------|
| text       | clipped / duplicated / garbled     | `refit_text`, `dedupe_text`, `reocr_region` |
| color      | wrong fill / drift                 | `recolor(layer)`           |
| layout     | wrong position / overlap           | `reposition(layer)`        |
| alignment  | off-grid / baseline                | `realign(layer)`           |
| component  | missing / extra element            | `resegment(region)` / `drop(layer)` |
| image      | logo baked / bad cutout / no mask  | `remask(layer)`, `revectorize` |

Operators are discrete, **idempotent**, single-element, with a stable signature (for the
phase-1 no-repeat memory). Editor applies the highest-severity defect first (coordinate
descent), Verifier confirms, loop continues.

Defect object contract:
```json
{ "id": "d17", "layer_id": "c_B0", "dimension": "text", "defect": "clipped",
  "severity": "high", "evidence": "rendered 'waar' is prefix of source 'waarbij je 20%'",
  "operator": "refit_text", "region": [x,y,w,h] }
```

## 3. Critic (external + grounded)

New: extend `src/vlm_anomaly.py` (phase-1) into a full critic, or `src/harness_critic.py`.
- Input: source image + rendered preview + an **overlay/diff image** (source vs render) +
  the layer list. (IVT/ReLook: the model must SEE its rendered output vs target.)
- Two signal sources, fused:
  1. **Tool checks** (deterministic, cheap, trusted): render-OCR diff (§1a), region deltas
     (§1b), duplicate-string detection, bbox-overflow (rendered text width > box).
  2. **VLM judge** (interpretable): rubric-scored per dimension with reasons; only used to
     catch what tools miss and to disambiguate. Never the sole authority (intrinsic
     self-correction is unreliable).
- Output: the defect list above. Cap VLM calls; never raise; degrade to tool-only.

## 4. Loop control (mostly phase-1; add these)

- best-kept / rollback-on-regression / no-repeat-failed-repair / plateau-stop — DONE by
  Agent 4. Verify integration.
- **Per-repair verification**: Verifier re-scores only the affected region; commit only if it
  improved. (Kills "repaired 14×, no change".)
- **Confidence-guided budget**: order/allocate rounds by element confidence (low-confidence
  OCR / shaky masks first). (CoRefine.)
- Log every round `{defect, operator, before, after, kept|rolled_back}` to
  `runtime_report.json` so behavior is visible and testable.

## 5. Cross-run + archetype memory (compounding)

- Persist a small `defect_priors.json`: per-archetype recurring defects and their winning
  guard (e.g. `social_screenshot → text always auto-width`). Applied pre-emptively next run.
  (Reflexion episodic memory.)

## 6. Critic validation harness (trust but verify the critic)

Before trusting the critic in the loop, build a tiny internal eval set (VISCO-style,
2412.02172): take a few good designs, inject KNOWN defects (clip a text box, duplicate a
layer, unmask a logo, drift a color), and assert the critic flags exactly those. Gate: critic
recall on injected defects ≥ 0.9 before it drives repairs.

## Integration plan / sequencing

1. Land + merge the 5 phase-1 agents; apply their reported config/schema keys.
2. Confirm QA touch-points (`pixel_diff.py`, `qa_config.py`) post-merge.
3. Ship §1a (render-OCR text-correctness) first — biggest single win, unblocks real
   clipping/duplication detection.
4. Wire the defect taxonomy (§2) + operators into Agent-4's critic/repair.
5. Add per-archetype thresholds (§1c) + anti-degenerate guard (§1d).
6. Add per-repair verification (§4) + confidence budget.
7. Build the critic eval set (§6); only then enable the VLM judge as a repair driver.
8. Add cross-run priors (§5) last.

## Config keys introduced (add to config.example.yaml + config.yaml)

```yaml
qa:
  archetype: auto            # auto | social_screenshot | product_hero | text_poster | mixed
  text:
    enabled: true            # render-OCR text-correctness metric
    min_line_sim: 0.85
  thresholds:
    social_screenshot: { text: 0.90, visual: 0.55 }
    product_hero:      { text: 0.85, visual: 0.80 }
    text_poster:       { text: 0.92, visual: 0.70 }
    default:           { text: 0.88, visual: 0.75 }
runtime:
  harness:
    critic: { vlm: true, eval_gate_recall: 0.9 }
    verify_per_repair: true
    confidence_budget: true
    priors_path: defect_priors.json
```

## Non-goals (explicitly out of scope for now)
- Training/RL (VisRefiner/ReLook fine-tune; we stay inference-only + flow engineering).
- Replacing SSIM entirely — it stays as one term in the visual gate, just no longer the sole
  arbiter.
```
