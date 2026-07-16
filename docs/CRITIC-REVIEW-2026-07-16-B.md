# Critic Review 2026-07-16-B — Harness robustness, ops correctness, loop behavior

Lens: **robustness / operational correctness / loop lifecycle** (a sibling critic covers
metrics honesty / qa_reward gaming — not repeated here). Scope, priority order:
`src/harness.py`, `src/harness_loop.py`, `src/harness_fixer.py`, `src/repair.py`,
`src/routing.py`, then `benchmark.py`, `src/background_benchmark.py`,
`run_pipeline.py` resume machinery, plus the resource-lifecycle seams
(`src/qwen_worker.py` flux, `src/inpaint.py`, `src/vlm_client.py`).

Every finding cites file:line + a concrete failure sequence. Findings numbered GB1…

Verdict summary: termination is genuinely bounded on every path (no infinite loops).
The real damage is in **resource lifecycle** (ComfyUI wedge, silent VLM timeouts),
**iteration-budget starvation** (admission rejections burn the only repair slot), and
**stale/mixed state** (rollback restores metadata but not the pixel assets it references;
VLM-class blocks persist forever with no input dependence). Several audit-reported bugs
(medium VLM opinion outranking high deterministic; the target-id-only no-op) are verified
**fixed** in the current code — noted at the end.

---

## GB1 — ComfyUI queue/job never interrupted on Flux timeout → VRAM wedge poisons the NEXT run — CRITICAL

`src/qwen_worker.py:469-485` (`flux_inpaint`) and the outer failure paths (`:370-374`,
`:461-467`, `:505+`).

On timeout the poll loop simply gives up and returns `None`:

```python
deadline = time.time() + int(params["timeout_s"])
while time.time() < deadline:
    h = requests.get(f"{base}/history/{prompt_id}", ...).json()
    if prompt_id in h: history = h[prompt_id]; break
    time.sleep(1.5)
if history is None:
    msg = f"timed out waiting for {prompt_id}"   # returns None, no /interrupt, no /queue clear
    return None
```

Nowhere in the module (grep: zero hits for `/interrupt` or `/queue`) does the code cancel
the submitted prompt. When `flux_inpaint` returns `None`, `src/inpaint.py` degrades to
Big-LaMa/OpenCV and the current run "recovers" — but the ComfyUI job the code POSTed to
`/prompt` (`:453`) **is still running on the GPU, holding all VRAM**.

Failure sequence (matches the documented operational reality):
1. Run A submits Flux, exceeds `timeout_s`, returns `None`, run A finishes on fallback.
2. The Flux job keeps executing (or sits queued) on :8188, pinning 16 GB.
3. Run B's health probe `comfyui_healthy` (`src/inpaint.py:299-319`) hits `/system_stats`
   with a **0.5 s** timeout and gets a 200 → reports healthy=True.
4. Run B's Flux `/prompt` queues behind the wedged job; its own poll loop times out →
   returns `None` again, logged as an inpaint fallback that looks like a pipeline failure.
5. The harness then spends repair rounds re-resuming `reconstruct` (each a fresh 25-min
   Flux attempt) that can never succeed until the box is manually restarted.

Fix: on every early-return-after-`/prompt` path (timeout AND the outer `except`), POST
`{base}/interrupt` and, if the job is still queued, delete it via `{base}/queue`
(`{"delete":[prompt_id]}`) before returning `None`. Also treat `history is None` as
"still running" distinctly from "failed" so the caller can decide whether a retry is even
safe. Consider making `comfyui_healthy` additionally check `/queue` depth (a non-empty
running queue owned by a dead client = wedged), not just `/system_stats` liveness.

---

## GB2 — VLM timeouts are swallowed AND nondeterministic → repair loop chases GPU-contention noise — HIGH

`src/vlm_client.py:24` (`_DEFAULT_TIMEOUT_S = 20`), `:377-392` (`multi_pass_answer`),
consumed by `qa_reward.run_critique` / the critique path folded in at
`src/harness_loop.py:355-426` (`_apply_vlm_critique`, bare `except Exception: pass`).

`multi_pass_answer` collapses **any** exception — including a 20 s `urlopen` timeout under
GPU contention — into `(None, "vlm_error")`:

```python
try:
    answers.append(ask_vlm(...))
except Exception:
    return None, "vlm_error"
```

The critique caller then treats "no items" as "the VLM had no opinion" and proceeds. Two
distinct defects:

- **Error swallowing / undiagnosable:** a timed-out VLM is indistinguishable from a VLM
  that genuinely found nothing. No stage records "critique unavailable due to timeout";
  the run silently loses its repair-steering evidence with no surfaced error. Under the
  documented GPU-contention timeout this is the common case, not the rare one.
- **Nondeterminism → the loop chases noise:** whether a critique call beats the 20 s
  budget depends on concurrent GPU load (Flux/SAM running in the same window). Same input
  + same config can therefore yield a critique on one round and none on the next, changing
  which repair is chosen and whether a round "improves". `run_until_acceptable`'s
  plateau/rollback bookkeeping (`harness_loop.py:764-810`) then makes keep/roll decisions
  off a signal that flickers with wall-clock timing — the exact "iterate-to-perfection
  chasing noise" failure mode.

Fix: distinguish timeout from empty in `multi_pass_answer` (return a typed reason and let
the caller record `critique_unavailable=timeout` into critic.json/qa evidence); raise the
critique-path timeout well above 20 s when it runs alongside Flux, or serialize critique
against the Flux/SAM VRAM boundary (`vram.stage_boundary`) so it never competes; and make
a swallowed critique failure visible in the round record rather than an implicit no-op.
Note the content cache (`:79-100`) already dedupes identical crops across rounds, so a
longer timeout does not multiply cost.

---

## GB3 — Admission rejection consumes the only repair iteration → concrete high-severity repairs are starved for a whole round — HIGH

`src/harness.py:772-841` (`execute_repairs` main loop), driven from
`src/harness_loop.py:238-240` (`repair_iterations`, default **1**) via `_run_round`
(`:586-592`).

`execute_repairs` iterates `for iteration in range(1, max_iterations+1)` and picks exactly
**one** candidate per iteration. When that candidate is rejected/skipped at admission
(`plan_is_concrete` fails `:792`, VLM-class already-failed `:814`, or unchanged plan
fingerprint `:830`), the code does `exhausted.add(...)` then **`continue`** — advancing the
*iteration counter*, not moving to the next candidate:

```python
for iteration in range(1, max(0, int(max_iterations)) + 1):
    repairs = load_repair_candidates(run_dir, working_cfg)
    candidates = [r for r in repairs if _repair_id(r) not in exhausted]
    choice = recommended_resume(candidates)
    ...
    if not plan_is_concrete(choice):
        exhausted.add(...); attempts.append({... admission_rejected ...}); continue   # burns the iteration
```

With `repair_iterations` defaulting to 1 (the value passed by the main
`run_until_acceptable` loop), a single non-concrete **top-ranked** candidate exhausts the
round's entire repair budget: no concrete repair runs, even though concrete
lower-ranked ones are sitting in `candidates`.

Failure sequence: QA has one vague VLM-sourced top candidate plus several concrete
high-severity deterministic repairs. Round 1 picks the vague one → rejected → `continue`
→ loop ends → `execute_repairs` returns having run nothing. The round is (correctly) not
counted toward plateau, so `run_until_acceptable` proceeds to round 2 and — because
`blocked_repairs` carries the rejection forward — finally runs one concrete repair. Net:
with the default `max_rounds=2`, **one of the two rounds is wasted on a rejection**, so at
most one concrete repair ever executes. This is the same "high-severity repairs never ran"
class the audit flagged, now caused by budget accounting rather than mis-ranking.

Fix: admission rejections/skips must advance to the **next candidate within the same
iteration** (loop over ranked candidates; only decrement/consume `max_iterations` when a
repair actually triggers a pipeline rerun). E.g. make the inner selection a `while` over
non-exhausted candidates and only count reruns against the budget.

---

## GB4 — Rollback restores design metadata but NOT the pixel assets it references → shipped "mixed rounds" design — HIGH

`src/harness_loop.py:71-74` (`_SNAPSHOT_FILES`), `:150-172` (snapshot/restore),
`:771-785` and `:846-853` (rollback / emit-best).

`_SNAPSHOT_FILES` snapshots `design.json`, `reconstruction.json`, `layout.json`,
`preview.png`, `runtime_report.json`, `fallback.json`, `figma_export.png`, `qa.json`. It
does **not** include the binary assets those JSONs point at:
`background_clean.png`, `removal_mask.png`, `ownership.png`, and the per-layer
`layers/*.png` / `layers_contact.png`. `reconstruction.json` explicitly stores these as
filename references (and `run_pipeline._artifact_ready:159-162` even validates them as
required sidecars); `design.json`'s background `base_src` is
`A(reconstruction.get("background", "background_clean.png"))` (`run_pipeline.py:644`).

Failure sequence:
1. Round 0 produces a good `background_clean.png`.
2. Round 1 resumes from `reconstruct` and regenerates a **worse** `background_clean.png`
   (e.g. Flux fell back to OpenCV per GB1); its score regresses.
3. `run_until_acceptable` detects the regression and calls `_restore_artifacts`
   (`:773`), restoring `reconstruction.json` + `design.json` + `preview.png` from the
   round-0 snapshot — **but `background_clean.png` on disk is still round 1's bad plate.**
4. The shipped `design.json` now says "background = background_clean.png" while that file
   holds the regressed pixels. Figma staging (`figma_import.import_design` reads
   `design.json` → the png) and any post-loop re-render ship the round-1 plate under
   round-0 metadata. `qa.json` (round 0) disagrees with the pixels actually staged.

This is the identical "shipped X disagreed with shipped Y" corruption the `_SNAPSHOT_FILES`
comment claims to have fixed for `runtime_report.json` — the fix is incomplete because the
referenced image assets were never added to the snapshot set.

Fix: snapshot/restore every artifact `reconstruction.json`/`design.json` reference by path
(`background_clean.png`, `removal_mask.png`, `ownership.png`, and the `layers/` dir), or
snapshot the whole run dir minus the harness bookkeeping files. Safest: derive the asset
list from the restored `reconstruction.json` so it stays correct as the contract evolves.

---

## GB5 — `seen_classes` VLM-opinion block is persisted with NO input dependence → a VLM class is blocked forever, across runs — MEDIUM

`src/harness.py:766-770` (loads `harness_admission.json`), `:809-827` and `:886-891`
(records `seen_classes[vlm_class] = {"qa_improved": ...}`), key = `f"vlm:{resume}:{action}"`
(`:812`).

Unlike the plan fingerprint (`_repair_plan_fingerprint:516-541`, which mixes in the SHA-256
of the stage's input artifacts so it re-opens when inputs change), the VLM-class key encodes
only the resume stage and action. Once any equivalent VLM opinion in that class is observed
non-improving, `harness_admission.json` persists `qa_improved=False`, and **every future**
opinion of that class is skipped at admission (`:814`) — with no re-evaluation when the
design later changes.

Failure sequence: round 1's `vlm:layout:refit-geometry` critique doesn't improve QA →
recorded. A later repair (or a later benchmark re-run resuming into the same `run_dir`)
materially changes reconstruction/layout so the layout critique would now be actionable —
but the class is permanently blocked because the admission file is read back at
`execute_repairs:767` and the key never changes. Because `harness_admission.json` lives in
the run dir and is never invalidated, `--resume` and repeated benchmark invocations inherit
stale blocks.

Fix: fold the same stage-input fingerprint used by `_repair_plan_fingerprint` into the
`seen_classes` key (or store the input digest alongside and re-admit when it changes).
Alternatively scope the class memory to the current `execute_repairs` invocation instead of
persisting it across runs.

---

## GB6 — Config change on `--resume` is only logged, not enforced → stale upstream artifacts silently ignore a repair's patch — MEDIUM

`run_pipeline.py:217-219` and the per-stage reuse guards (`stage()` + `dirty` + `exists`,
e.g. `:339`, `:348-366`, `:380`).

On resume, a changed `config_sha256` produces only a log line
("config changed since prior artifacts; resuming from the requested stage") and the run
proceeds to **reuse every artifact before the resume stage as-is**. The resume model is
correct only when a repair's config patch affects the resume stage or later. It is silently
wrong when a patch targets a stage *earlier* than the mapped resume stage.

Concrete case: `repair.assess` / `config_patches_for` emit `sam3 rerun-detection` with
`vlm.element_propose.enabled=True`, and `resume_stage_for` maps it to **`sam`**
(`harness.py:266-271`). But VLM element-propose enrichment runs in the residual→sam region
(`run_pipeline.py:348-376`) and operates on `residual.json`, which — resuming from `sam` —
is **not** regenerated (`stage("residual")` False, `dirty` False, artifact exists → loaded
from disk at `:345-346`). The enrichment then runs against the stale residual. More
generally there is no guard that a patch's affected stages are all ≥ the resume stage, so
some patches are partial no-ops that still consume a repair round.

Fix: when `config_sha256` differs, compute the earliest stage any patch key affects and
refuse to start later than it (or force `dirty` from that stage), instead of logging and
proceeding. At minimum, map each patch namespace to its owning stage and assert
`resume_stage <= min(affected_stage)`.

---

## GB7 — Reward computation and acceptance gate fail **open** → a degenerate result can be accepted when scoring errors — MEDIUM

`src/harness_loop.py:121-139` (`_score_round`), `:142-147` (`_reward_gate`), `:857-863`
(final gate). Also `qa_reward.compute_reward` wrapped in bare `except: pass` (`:133-138`).

`_reward_gate` returns `{"ok": True, "skipped": "gate_error:..."}` on any exception, and the
final acceptance is `qa_ok = _qa_accepts(final_qa) and bool(final_gate.get("ok", True))`.
The gate is documented as "strictly additional — can only refuse, never grant". But because
an *errored* gate defaults to `ok=True`, a metric-accepted-but-perceptually-degenerate
result that the gate was supposed to catch is accepted whenever the gate raises (missing
LPIPS weights, a malformed reward record, an OOM during scoring, etc.). Likewise
`_score_round` silently falls back to the legacy mean-of-metrics on any `compute_reward`
exception, so plateau/rollback/best-kept decisions quietly switch scoring regimes mid-loop
without any record.

Failure sequence: phase2 reward model fails to load on the GPU box → every `_reward_gate`
call returns `ok:True (gate_error)` → the anti-degenerate gate is a no-op for the whole run
→ a "bought-looking" metric pass ships as accepted. The only evidence is a `skipped` string
buried in the round record.

Fix: a gate that *errored* is not the same as a gate that *passed* — treat `gate_error` as
non-acceptance for acceptance/parity configs (fail closed), or at least surface it as a
runtime violation so the benchmark's `runtime_accepted` count reflects it. Record scoring-
regime switches explicitly.

---

## GB8 — Plan-fingerprint no-repeat memory is inert for `merge`/`qwen`/`figma`/`design` resumes — MEDIUM (thrash mitigation gap)

`src/harness.py:516-541` (`_repair_plan_fingerprint`). The `inputs` map only covers
`ocr/text/sam/elements/reconstruct/layout/design/figma` for a subset; any unmapped resume
(`.get(resume, ("qa.json",))`) fingerprints against **`qa.json`**, which is rewritten on
every pipeline pass. Its fingerprint therefore always differs round-to-round, so the
"unchanged repair plan and inputs" skip (`:830`) can never fire for those stages.

Impact is bounded (a non-improving `merge dedup` / `qwen retry` is still caught by the
`exhausted.add` on non-improvement at `:897-898` and blocked across rounds via
`blocked_repairs`), so this is thrash-mitigation degradation, not non-termination. But it
defeats the intended cross-invocation memory: an interrupted-then-resumed harness will
re-run an identical `merge`/`qwen` repair that a persisted fingerprint should have skipped.

Fix: give every resume stage a real input list (e.g. `merge → ("elements.json", "ocr.json",
"peel.json")`, `figma → ("design.json",)` already exists but `merge`/`qwen`/`structure` are
missing), and fingerprint the *design output* rather than `qa.json` for the fallback case.

---

## GB9 — Critic/critique/fixer exceptions are broadly swallowed → a fixture can "pass" or skip with no diagnosable cause — MEDIUM

`src/harness_loop.py:303-315` (`_run_critic_pass` catches ImportError/TypeError/ValueError/
KeyError → fallback), `:355-426` (`_apply_vlm_critique` bare `except Exception: pass`),
`:496-504` (`_run_fixer_pass` catch → empty fixes), and `run_pipeline.py:739-760` (primary
QA crash → minimal pixel judge with `ssim:0.0`).

Individually each is a reasonable "never crash the run" guard, but stacked they let a run
reach a terminal verdict with critical evidence silently missing. The `_apply_vlm_critique`
bare-except in particular hides *logic* bugs in the merge/re-rank (not just VLM outages):
a `KeyError` while building `critique_to_repairs` disappears and the round proceeds with
deterministic-only repairs, indistinguishable from "VLM found nothing". The QA fallback at
`run_pipeline.py:750-756` fabricates `ssim:0.0` + a `qa-primary-failed` hard fail — correct
that it can't pass, but the *reason* the primary judge crashed is only a `detail` string.

Fix: keep the guards, but record a structured `error`/`skipped` marker into
`critic.json` / the round record for each swallowed failure so `benchmark._harness_telemetry`
can surface "critic_error / critique_error / fixer_error" counts. A run that only "passed"
because three subsystems silently no-op'd should be visibly distinct from a clean pass.

---

## GB10 — `harness_admission.json` written non-atomically-consistent across early returns; partial admission state on interruption — LOW

`src/harness.py:766-841`. `admission` is loaded once, mutated, and `_write_json`-persisted at
several points (`:803`, `:826`, `:840`, `:891`). `_write_json` itself is atomic (tmp+replace,
`:450-455`), but the several early `return`s in the actionable/plateau branches
(`:776-784`, `:743-749`) leave `seen_plans`/`seen_classes` in whatever partial state the last
write captured. Combined with GB5's persistence, a harness killed mid-round can leave a
class/plan recorded as "seen" whose attempt never actually completed, permanently skipping it
on resume. Low severity (needs a kill at a specific point) but compounds GB5.

Fix: only commit `seen_*` entries after the attempt's outcome is known (move the writes to
after the `attempts.append(attempt)`), so an interrupted attempt is retried rather than
skipped.

---

## GB11 — `routing.py`: no loop/lifecycle defects (informational)

`src/routing.py` is pure, deterministic decision logic (candidate → `target`); no I/O, no
retries, no state. Reviewed for the lens; nothing to flag. (One duplicated comment block at
`:450-455` is a style nit, out of scope.)

---

## GB12 — `background_benchmark.py`: robust; one minor honesty note (LOW)

CPU-only, deterministic (seeded RNG), no GPU/loop lifecycle. `run_inpaint_bakeoff` correctly
treats a backend outage as evidence rather than a crash (`:716-717`) and fails a substituted
backend unless explicitly allowed. Only nit: `worst_inside_mae = max(row["mae"] ...)`
(`:624`, `:791`) will `ValueError` on an empty `inside_rows`, but `generate_synthetic_cases`
guarantees ≥1 case, so unreachable in practice. No action required.

---

## Verified-FIXED (audit items confirmed no longer reproducible in current code)

Documented so the fixer does not re-open them:

- **Medium VLM opinion outranking HIGH deterministic failures** — `rank_repairs`
  (`harness.py:376-392`) keys on `(-severity, vlm_flag, -badness, index)`; severity is
  primary and the `vlm_critique` flag is only a same-severity tiebreaker, so a MEDIUM VLM
  opinion can no longer precede a HIGH deterministic repair. `_apply_vlm_critique`
  (`harness_loop.py:388-421`) also merges critique repairs *after* metric/tool repairs.
- **The target-id-only no-op repair** — `plan_is_concrete` (`harness.py:305-323`) rejects a
  plan whose only patch is `harness.target_id`, and rejects VLM-critique plans that patch
  nothing; `_apply_vlm_critique:403-414` additionally drops VLM `layout/refit-geometry`
  opinions carrying no measured geometry. Admission runs *before* a rerun is spent
  (`harness.py:792-804`).
- **Per-layer scores ignored when picking the target** — `repair.assess:694-739` now emits
  worst-measured-first targeted repairs from `per_layer` local scores, and
  `_layer_local_scores`/`_repair_measured_badness` (`harness.py:326-370`) feed the ranking.
- **Plateau counting rejected/no-op repairs as convergence** — `round_evaluated`
  (`harness_loop.py:730-733`) excludes admission-skipped/rejected attempts, and the plateau
  increment (`:800-801`) is gated on `round_evaluated`; rejected-only rounds no longer count
  toward plateau. (The residual budget problem is GB3, a different mechanism.)
- **Termination** — `run_until_acceptable` (bounded `for round_num in range(1, max_rounds+1)`),
  `execute_repairs` (bounded `range(1, max_iterations+1)`), and `flux_inpaint` (deadline
  while-loop) are all bounded; the "keep converging" branch (`harness_loop.py:824-838`) only
  `continue`s while `round_num < max_rounds`. No unbounded loop exists.

---

## Priority for the fixer

1. **GB1** (CRITICAL) — interrupt/clear ComfyUI on Flux timeout; this is the root of the
   "next run's Flux times out looking like a pipeline failure" reality.
2. **GB4** (HIGH) — snapshot the pixel assets that `reconstruction.json`/`design.json`
   reference, or rollback ships a corrupt mixed-round design.
3. **GB3** (HIGH) — admission rejection must not consume the repair-iteration budget;
   otherwise at most one concrete repair runs per 2-round loop.
4. **GB2** (HIGH) — surface + de-noise VLM timeouts so the loop stops chasing GPU-contention
   noise and stops silently discarding critique evidence.
5. **GB5–GB7** (MEDIUM) — persistent-block input-dependence, resume config-staleness guard,
   fail-closed reward gate.
6. **GB8–GB10** (MEDIUM/LOW) — fingerprint coverage, structured swallowed-error markers,
   admission-write ordering.
