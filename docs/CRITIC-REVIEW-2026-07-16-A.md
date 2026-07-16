# Critic Review 2026-07-16-A — Metrics Honesty & Reward Correctness

Lens: METRICS HONESTY AND REWARD CORRECTNESS. Scope: `src/qa_reward.py`,
`src/harness_loop.py`, `src/harness.py`, `src/repair.py`, `src/run_report.py`,
`benchmark.py`, plus `src/pixel_diff.py` (metric producer).

## Status of the audited 002 bug (reward gate green beside 4 hard fails / worst-local 0.0091)

Largely **CLOSED for full-metric runs**. The code now has: a worst-local floor
(`qa_reward.acceptance_gate` GA reads `local_ssim_worst_window.ssim`, emitted by
`pixel_diff.py:1852`), a single-layer weight-share cap (`_LOCAL_WEIGHT_SHARE_CAP`),
hard-fail and `contract.pass==False` checks that force the gate RED
(`qa_reward.py:720-732`), and `run_until_acceptable` folds the gate into the final
`qa_ok` (`harness_loop.py:863`). The specific 002 vector no longer buys a green gate
**when pixel_diff emits per_layer + local_ssim_worst_window + hard_fails.**

However the same *class* of bug survives through several escape hatches below. All
findings marked "demonstrated" were reproduced by feeding `qa_reward` a crafted qa dict.

---

### GA1 — Acceptance gate fails OPEN on any exception  — CRITICAL
`src/qa_reward.py:737-738`; mirrored in `src/harness_loop.py:145-147`.

Both the gate itself and the loop's `_reward_gate` wrapper swallow **every** exception
and return `{"ok": True, "skipped": "gate_error:..."}`. The gate is the anti-degenerate
safety net; making it fail-open means any bug, malformed qa row, or unexpected type in the
gate path silently converts a would-be RED into GREEN. A NaN/None in a per_layer row, a
non-dict `contract`, etc. → gate passes.

Repro (demonstrated):
```
reward whose components.get raises RuntimeError
-> acceptance_gate(...) == {'ok': True, 'skipped': 'gate_error:RuntimeError'}
```
Scenario: any partially-corrupt qa.json (frequent under the harness's own resume/rollback
churn) that throws inside the gate is scored as acceptable.

Fix: a gate that cannot evaluate must fail CLOSED (`ok: False, skipped: gate_error`) or at
minimum propagate `skipped` into `qa_ok` as a non-acceptance. Never return `ok: True` from
an exception handler in an acceptance path.

---

### GA2 — LPIPS perceptual floor silently unenforced (config-disable or missing torch) — HIGH
`src/qa_reward.py:415-419` (`lpips_enabled` defaults True but honors
`qa.reward.lpips.enabled: false`), `:474` (`lpips_score` returns None on ANY exception,
incl. torch/lpips import failure), `:620-621`, `:710-715`.

`gate_thresholds` always returns `lpips_similarity_min` (default 0.80), but the gate only
checks it `if isinstance(lpips, dict) and isinstance(lpips.get("similarity"), ...)`. When
the lpips component is None the floor is simply **not applied** — no check recorded, no
penalty. Two ways to reach None: (a) config sets `qa.reward.lpips.enabled: false`; (b) any
import/runtime error in the lpips path (torch not importable, model download failed, CPU
OOM). A degenerate render with catastrophic perceptual distance but a passable
local/aggregate score then clears the gate.

Repro (demonstrated):
```
cfg qa.reward.lpips.enabled=False, qa with one big raster region_ssim 0.55
-> lpips component None; GATE checks == ['local_ssim','worst_local_ssim']; ok: True
```
The 0.80 perceptual floor — one of the three advertised anti-degenerate gates — evaporated
via a one-line config toggle, with nothing in the gate record flagging its absence.

Fix: when lpips is expected (phase2, not explicitly disabled) but the component is None,
record a RED `lpips_similarity: {value: null, ok: false, reason: "unavailable"}` instead of
skipping, OR make `require_active_models` treat a missing lpips as a violation. Distinguish
"legitimately disabled" from "silently failed to load".

---

### GA3 — Construction score falls back to raw `native_text_ratio` and dominates the reward even when the contract FAILED — HIGH (Goodhart on best-round selection)
`src/qa_reward.py:500-517` (`construction_component` fallback chain ends at bare
`native_text_ratio`), `:639-642` (contract blended at `_CONSTRUCTION_WEIGHT = 0.55`).

When `contract.contract_score` and a construction dict are absent but `native_text_ratio`
is present, the "construction" component becomes the raw native-text ratio and then
dominates the scalar reward at 0.55 weight — with **no** residue/placement/erasure gating
in the score (those only affect the gate, not the reward number). The reward is what
`harness_loop._score_round` / best-round tracking use to pick `best_snapshot` and to
roll back (`harness_loop.py:764-804, 846-853`). So a round that erased a product cluster
but kept its headline text as native TEXT scores HIGH and can be selected as "best" and
emitted to disk, beating a round with intact imagery but slightly lower native-text ratio.

Repro (demonstrated):
```
qa: contract.pass=False, native_text_ratio=0.95, ssim 0.30, worst per_layer 0.20
-> construction source='native_text_ratio' score 0.95
-> reward score 0.6125  (ladder-only would be ~0.20)
```
The gate still blocks *acceptance* (contract.pass False → RED), but the inflated reward
mis-drives best-kept/rollback, so the shipped-on-disk artifact is the erased one.

Fix: the fallback to bare `native_text_ratio` must be gated by residue/placement/erasure
signals, or clamped when `contract.pass is False` (e.g., cap construction at the contract
floor). A failed contract must not yield a >0.9 construction term.

---

### GA4 — `benchmark._mean` drops None rows → failed/missing fixtures excluded from every quality mean — HIGH (aggregation lie)
`benchmark.py:389-391`.

```python
def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None
```
The denominator is only rows that HAVE the field. A fixture that crashed, produced no
contract score, or whose metric is null is silently removed from `mean_native_text_ratio`,
`mean_contract_score`, `mean_ssim`, `mean_text_recall`, `mean_editable_text_recall`,
`mean_native_leaf_ratio`, `mean_element_recall`, etc. (`benchmark.py:596-623`). The worst
fixtures are exactly the ones most likely to be null, so every headline mean is biased
upward by dropping the failures. A benchmark where 4/10 fixtures errored out can report a
glowing `mean_ssim` computed over only the 6 that succeeded.

Fix: report the denominator alongside each mean (`{value, n, of}`), and/or count nulls as
0 for coverage-style metrics, and surface a `fixtures_missing_<field>` count so a reader
sees the mean is partial.

---

### GA5 — Worst-local and local_ssim floors both skipped on the `multiscale_ssim`-only path — MEDIUM
`src/qa_reward.py:393-397` (`local_component` multiscale fallback returns
`{score, count:0, source:"multiscale_ssim"}` with **no** `min`/`worst_local`), `:400-409`
(`_worst_window_ssim` returns None when neither `local_ssim_worst_window` nor
`local_ssim.min` present), `:683-708`.

When a qa dict carries only a global `ssim` (no `per_layer`, no `local_ssim_worst_window`,
no `local_ssim` dict) — lightweight/bridge summaries, older artifacts, or a truncated
metrics run — the gate applies ONLY `local_ssim >= 0.50` and skips the worst-local floor
entirely. A single giant raster with global ssim 0.62 and erased content passes with one
check.

Repro (demonstrated):
```
qa = {"ok": True, "ssim": 0.62}
-> GATE {'ok': True, 'checks': {'local_ssim': {'value':0.62,'min':0.5,'ok':True}}}
```
Lower severity than GA1-GA4 because full pixel_diff output populates the backstop fields;
this is the degraded/summary path. But it is the literal 002 lie reachable whenever the
worst-window evidence is absent, so it should not be a single-check green.

Fix: when the local component is `source: "multiscale_ssim"` (no per-element/window
evidence) require the perceptual (LPIPS) gate to be present, or treat missing worst-local
evidence as non-acceptance rather than a skipped check.

---

### GA6 — `final_qa_passing` counts loose `qa.get("ok")` when harness_loop.json is absent — MEDIUM
`benchmark.py:191-195` (`_harness_telemetry.final_qa_ok` fallback chain ends at
`qa.get("ok")`), summarized at `benchmark.py:627`.

`final_qa_ok = loop.get("final_qa_ok") or loop.get("qa_ok") or qa.get("ok")`. When neither
`harness_loop.json` nor legacy `harness.json` exists (harness disabled, or crashed before
writing), it falls back to raw `qa.get("ok")` — the exact flag that can read True beside
standing hard fails and an incomplete evidence set. The strict per-run `qa_ok`
(`benchmark.py:265`, `= qa.ok and evidence_complete and not merged_hard_fails`) does NOT
gate this summary line, so `final_qa_passing` can exceed `qa_passing` and overstate success.

Fix: derive `final_qa_ok` from the same strict predicate as `_entry.qa_ok` (evidence
complete + zero merged hard fails + reward gate), never from bare `qa.ok`.

---

### GA7 — `content_penalty` is near-toothless — MEDIUM
`src/qa_reward.py:88-92`, `:552-571`.

`_RASTERIZED_TEXT_FLOOR = 0.50` means up to 50% of headlines shipped as pixels costs
nothing; `_CONTENT_PENALTY_CAP = 0.15` caps the entire penalty at ~one hard-fail's weight
even at 100% rasterization + zero native leaves. For runs with no `contract_score` (legacy
fixtures, or any run where the contract block is absent) this penalty is the ONLY thing
representing "text was rasterized instead of made editable," and it barely dents a
screenshot-plausible reward. Directly contradicts the north-star ("readable OCR text must
NEVER be raster-sliced; raster slices count as NON-editable").

Fix: lower the rasterized-text floor toward 0 and raise/remove the cap so a majority-raster
reconstruction cannot score high; or make a rasterized-text ratio above a small threshold a
hard fail rather than a bounded soft penalty.

---

### GA8 — `RunReport.acceptable` / benchmark `runtime_ok` never consult the reward gate — MEDIUM
`src/run_report.py:119-129` (`acceptable` checks violations/status/qa_ok only);
`benchmark.py:264` (`runtime_ok = bool(runtime.get("acceptable"))`), `:598`
(`runtime_accepted` sum).

`runtime_report.json.acceptable` is computed from hard violations + `qa_ok`, with no input
from the phase2 reward acceptance gate (LPIPS / worst-local / construction). A run with no
structural hard-fails but a failing perceptual/worst-local gate is still "acceptable" and
counted in `runtime_accepted`. The reward gate result is only patched into
`runtime_report.harness_convergence.reward` (`harness_loop.py:944-945`) after the fact and
is not read back by `acceptable`.

Fix: fold `harness_convergence.reward.gate.ok` into `RunReport.acceptable`, or have the
harness set a violation when the final gate is RED.

---

### GA9 — Benchmark reports only means; bimodal failure distributions read as healthy — MEDIUM
`benchmark.py:589-628` (summary block is entirely `sum`/`_mean`).

No min / worst / p10 / stddev for any per-fixture metric. A batch that is half 0.98 and
half 0.35 reports the same `mean_ssim ≈ 0.66` as a uniformly-mediocre batch, hiding the
cluster of hard failures. The per-run rows exist, but the headline (and the exit-code logic)
lean on means and pass-counts, so a reader scanning the summary misses bimodality.

Fix: add `min_`/`worst_`/`p10_` companions (at least for ssim, native_text_ratio,
contract_score, element_recall) so the worst fixture is visible in the roll-up.

---

### GA10 — Reward renormalizes over available axes → a missing dimension is dropped, not penalized — MEDIUM
`src/qa_reward.py:616-636`.

`weight_total = sum(weights[name] for name in available)` then `ladder /= weight_total`.
If `text` is absent (OCR wrote no `text_recall`/`editable_text_recall`), its weight is
redistributed onto local_ssim + lpips, so a text-heavy ad is scored purely on visual
similarity — the screenshot-similarity trap the contract explicitly rejects. Same for a
missing local component. A dropped axis should count against the run, not silently vanish.

Fix: for the axes the archetype declares relevant (esp. `text` on text-heavy/dark
archetypes), treat "missing" as 0 or as non-scoreable rather than renormalizing it away.

---

## Notes / non-findings
- `_assert_thresholds_unchanged` (`harness_loop.py:255-271`) genuinely enforces the
  never-lower-thresholds invariant across visual_pass_ssim, repair `*_min`, and reward gate
  floors — good.
- `_qa_accepts` (`harness.py:65-85`) correctly fails closed on non-empty hard_fails and
  structural hard_fails — the acceptance predicate itself is honest.
- `RunReport.finish` (`run_report.py:131-185`) cross-checks `qa_ok` against required visual
  evidence and downgrades to False on contradiction — a real honesty mechanism.
- `harness_critic` / `harness_fixer` are repair routers; the fixer only *tightens*
  thresholds (e.g. `min_container_frac` down at `harness_fixer.py:200-202`), it cannot relax
  acceptance. Not a reward-honesty risk.
- `text_component` (`qa_reward.py:523-530`) takes the strict MIN of text_recall /
  editable_text_recall — correct; the editable-recall-denominator inflation is handled by
  `true_text_coverage` in the metrics layer (though see GA10: `true_text_coverage` is not
  itself fed into the reward).

## Severity roll-up
- CRITICAL: GA1
- HIGH: GA2, GA3, GA4
- MEDIUM: GA5, GA6, GA7, GA8, GA9, GA10
