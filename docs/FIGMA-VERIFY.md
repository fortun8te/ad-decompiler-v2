# FIGMA-VERIFY — the "verified in real Figma" QA gate

## Problem

Pipeline QA (`run_pipeline.py` diff/qa) scores whichever render exists in the run
dir. During development that is almost always `preview.png` — our own Python-side
simulation from `src/render_preview.py`. A green `qa.json` therefore proves *our
simulation* matches the source; it proves nothing about what Figma actually
displays after the plugin import. If the simulation and Figma disagree (fonts,
gradient interpolation, effects, auto-layout), we ship green dashboards and broken
Figma files.

The companion plugin already closes the loop: after Import it exports the built
frame as PNG (`figma-plugin/code.js` → `root.exportAsync({format:"PNG"})`) and
POSTs it to the bridge (`POST /export`, `src/figma_bridge.py`), which writes it to
the staging manifest's `export_to` — set by `src/figma_import.py::_stage_for_plugin`
to **`<run_dir>/figma_export.png`**. `src/figma_verify.py` turns that export into a
first-class verdict.

## What it scores

| comparison | pair | meaning |
|---|---|---|
| `fidelity` | `figma_export.png` vs `normalized.png` (fallback `original.png`) | what Figma really displays vs the source ad — the score that matters for shipping |
| `preview_drift` | `figma_export.png` vs `preview.png` | does our simulation match Figma? When this fires, preview-based QA scores are not trustworthy — the drift heatmap + region list show WHERE |
| text recall | source `ocr.json` lines recovered from OCR of the export | did text survive the real Figma import (fonts loaded, nodes built)? |

Pixel metrics come from `src/pixel_diff.py::compare` (multiscale SSIM, edge F1,
color similarity, visual score), called into a scratch evidence dir
(`run_dir/figma_verify/{fidelity,drift}/`) so its `diff.png` never clobbers the
pipeline's. pixel_diff is consumed defensively (getattr/.get + an internal
fallback judge) because it is being extended concurrently. Drift *regions* are
scored color-aware (mean per-channel SSIM per grid cell) so a pure chroma
difference — wrong fill, gradient, color profile — cannot hide in grayscale.

Before scoring, the export is:

1. **Scale-normalized** — export dimensions vs the `design.json` canvas detect
   1x/2x/3x exports (`export.scale.detected`); non-integer or non-uniform scale is
   an anomaly that caps the verdict at `degraded`.
2. **Coarsely aligned** — phase correlation estimates a global (dx, dy) offset
   (≤ `max_align_shift_px`); the shift is applied only if it measurably improves
   alignment, and is recorded in `export.alignment`.
3. **Freshness-gated** — mirroring `run_pipeline._artifact_at_least_as_fresh`, an
   export older than `design.json` is *not evidence about the current design*:
   status `stale-export`, verdict `not-exported` (unless `--allow-stale`, which
   scores it but caps the verdict at `degraded`).

## Verdicts

| verdict | meaning |
|---|---|
| `verified` | fresh export, fidelity ≥ threshold, text recall OK (when measurable), drift immaterial |
| `degraded` | fidelity passes but evidence is weakened: material preview drift, missing preview, anomalous scale, stale export scored via allow-stale, unknown text recall under `require_text_evidence`, or fallback-judge-only scoring |
| `failed` | export does not match the original above threshold, or text recall was measured below threshold, i.e. the real Figma file is broken (fails closed on scoring crashes too) |
| `not-exported` | no usable export — nothing was verified in real Figma. This is the *default truth* for headless runs |

A pretty screenshot can never override a hard check; loss of evidence is reported,
never silently passed (see AGENTS.md invariants and `src/run_report.py`).

## Workflow (human-in-the-loop round trip)

```text
1. bridge up          Start Bridge.bat            (or: python -m src.figma_bridge
                                                   --inbox C:/Users/micha/figma-inbox --port 8790)
2. stage a run        python run_pipeline.py --input <img> --output runs/<id>
                      (figma.enabled: true stages design.json+assets into the inbox)
3. start the watcher  python scripts/figma_verify.py runs/<id> --watch --timeout 600
4. import in Figma    Figma desktop → ad-decompiler plugin → Import latest
5. export-back        the plugin auto-POSTs the frame PNG → bridge writes
                      runs/<id>/figma_export.png (atomic os.replace)
6. verdict            the watcher sees the fresh export, scores it, prints the
                      table, and writes runs/<id>/figma_qa.json
```

The bridge also re-runs pipeline QA on export/report arrival
(`_rerun_qa_for_run`, `start_from="qa"`), so `qa.json` flips to the real render at
the same time `figma_qa.json` appears.

## CLI

```text
python scripts/figma_verify.py <run_dir>                      # one run
python scripts/figma_verify.py --all runs/golden-optimized-check
python scripts/figma_verify.py <run_dir> --watch [--timeout 600] [--poll 2]
options:
  --export PATH     explicit export PNG (single run)
  --config PATH     alternate config.yaml
  --allow-stale     score an export older than design.json (capped at degraded)
  --ocr / --no-ocr  force / forbid OCR of the export (default: reuse a matching
                    render_ocr.json; run OCR only if figma.qa_ocr is enabled)
  --json            print full figma_qa.json payload(s)
  --strict          exit non-zero unless every run is 'verified'
exit codes: 0 no failures (--strict: all verified) · 1 any 'failed' ·
            2 --strict and a run not 'verified' · 3 watch timeout · 4 usage error
```

## figma_qa.json schema (v1)

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-07-15T12:34:56Z",
  "run_dir": "C:/.../runs/<suite>/<id>",
  "status": "scored | not-exported | stale-export | error",
  "verdict": "verified | degraded | failed | not-exported",
  "export": {
    "path": ".../figma_export.png", "bytes": 1234567,
    "mtime": "...Z", "fresh": true,           // mtime >= design.json mtime
    "size": [2000, 2000], "canvas": [2000, 2000],
    "scale": {"x": 1.0, "y": 1.0, "uniform": true, "detected": 1},
    "alignment": {"dx": 0, "dy": 0, "applied": false},
    "aligned_png": ".../figma_verify/exported_aligned.png"
  },
  "fidelity": {                                // exported-vs-original
    "reference": ".../normalized.png",
    "ssim": 0.9656, "global_ssim": 0.98, "edge_f1": 0.9262,
    "color_similarity": 0.9639, "visual_score": 0.9574,
    "rgb_mae": 4.1, "delta_e_mean": 1.2,
    "text_recall": null,                       // null = no OCR evidence (reported, not hidden)
    "text_evidence": {"source": "render_ocr.json", "reused": true},
    "diff_png": ".../figma_verify/fidelity/diff.png",
    "engine": "pixel_diff.compare"             // or "figma_verify-fallback"
  },
  "preview_drift": {                           // exported-vs-preview (drift detector)
    "preview": ".../preview.png",
    "ssim": 0.9142, "material": true,          // ssim < drift_ssim_min
    "grid": [12, 12],
    "cell_ssim": [[1.0, "..."]],               // color-aware per-cell SSIM matrix
    "regions": [                               // top-N worst cells, canvas pixels
      {"rank": 1, "row": 9, "col": 9, "ssim": 0.138,
       "bbox": {"x": 1500, "y": 1500, "w": 166, "h": 166}}
    ],
    "heatmap_png": ".../figma_verify/drift_heatmap.png",
    "diff_png": ".../figma_verify/drift/diff.png"
  },
  "checks": [                                  // honest per-check evidence
    {"check": "export-present",      "status": "pass", "value": null, "threshold": null, "detail": "figma_export.png"},
    {"check": "export-fresh",        "status": "pass"},
    {"check": "export-scale",        "status": "pass", "value": {"detected": 1}},
    {"check": "fidelity-ssim",       "status": "pass", "value": 0.9656, "threshold": 0.84},
    {"check": "fidelity-text-recall","status": "skip", "threshold": 0.8,
     "detail": "no OCR of the Figma export — text survival in real Figma is unproven"},
    {"check": "preview-drift-ssim",  "status": "fail", "value": 0.9142, "threshold": 0.95}
  ],
  "thresholds": {"fidelity_ssim_min": 0.84, "text_recall_min": 0.8,
                 "drift_ssim_min": 0.95, "drift_grid": 12, "drift_top_n": 8,
                 "drift_region_ssim_min": 0.85, "max_align_shift_px": 12,
                 "scale_tolerance": 0.02, "require_text_evidence": false},
  "warnings": ["degraded: material preview drift"],
  "report_path": ".../figma_qa.json"
}
```

## Configuration

All thresholds live under a `figma_verify:` block in `config.yaml` (every key
optional; defaults in `src/figma_verify.py::DEFAULT_VERIFY_THRESHOLDS`):

```yaml
figma_verify:
  fidelity_ssim_min: null      # null → qa.visual_pass_ssim (0.84 in config.yaml),
                               # so the Figma bar can never drift below the pipeline bar
  text_recall_min: 0.80
  drift_ssim_min: 0.95         # exported-vs-preview: below this the simulation lies
  drift_grid: 12               # grid cells per side for region localization
  drift_top_n: 8               # worst regions reported/outlined on the heatmap
  drift_region_ssim_min: 0.85  # a cell below this counts as a drift region
  max_align_shift_px: 12
  scale_tolerance: 0.02
  require_text_evidence: false # production sign-off should set true: without OCR
                               # of the export, "verified" is capped at degraded
```

OCR of the export: a matching `render_ocr.json` (provenance.render_path ==
figma_export.png, at least as fresh) is reused for free; otherwise OCR runs only
when `figma.qa_ocr` is enabled (or `--ocr`), writing to
`figma_verify/export_ocr.json` — never over the pipeline-owned `render_ocr.json`.

## run_pipeline.py wiring diff (for later application — NOT applied)

Two hunks make verification a real pipeline stage after `export`. Placing the
stage code after the qa block (and the stage name after `"qa"` in `STAGES`) means
the bridge's post-export re-entry (`_rerun_qa_for_run` → `start_from="qa"`)
automatically refreshes the verdict every time the plugin POSTs an export.

```diff
--- a/run_pipeline.py
+++ b/run_pipeline.py
@@ -32,4 +32,4 @@
 STAGES = ["normalize", "ocr", "text", "residual", "qwen", "sam", "elements",
           "merge", "structure", "reconstruct", "layout", "design", "preview", "figma",
-          "export", "diff", "qa"]
+          "export", "diff", "qa", "verify"]
```

```diff
@@ inside run_one(), immediately after the qa/else block and before
@@ `elapsed = round(time.time() - t0, 3)`:
+        # 13 figma verify — first-class verdict on the plugin's export-back PNG.
+        # Scores exported-vs-original (real fidelity) and exported-vs-preview
+        # (simulation drift) and writes figma_qa.json. 'not-exported' is the
+        # normal, honest outcome for headless runs; the bridge re-enters at
+        # start_from="qa" after POST /export, which refreshes this verdict.
+        if stage("verify") and cfg.get("figma", {}).get("enabled", False):
+            from src import figma_verify
+            fv = figma_verify.verify(run_dir, cfg=cfg)
+            _log(run_dir, f"figma-verify → {fv.verdict} (status={fv.status})")
```

Notes for whoever applies it:
- `--resume verify` comes for free (`--resume` choices are `STAGES`).
- Do NOT make `verify` fail the run by itself: `not-exported` is expected in
  headless/benchmark runs. Acceptance gating belongs to the benchmark consumer
  (below) and to the existing `figma.require_export` structural fail
  (`figma-export-missing`) already enforced in the qa stage.
- If a hard in-pipeline gate is later wanted for parity runs, the right place is
  a `report.degraded("figma-verify", ..., required=True)` *before*
  `report.finish` in the qa stage, keyed on `figma.require_export` — keep it out
  of the minimal wiring until that policy is decided.

## Benchmark / production sign-off consumption

`benchmark.py` currently reads `qa.json` per run. For production sign-off
(docs/PRODUCTION-READINESS.md Gate 4) it must also read `figma_qa.json`:

```python
figma_qa = _read(run_dir / "figma_qa.json", {})   # same _read helper as qa.json
row["figma_verdict"] = figma_qa.get("verdict", "not-exported")
row["figma_fidelity"] = (figma_qa.get("fidelity") or {}).get("ssim")
row["preview_drift"] = (figma_qa.get("preview_drift") or {}).get("ssim")
# production sign-off: preview-based qa.ok is NOT sufficient
row["production_ready"] = bool(qa.get("ok")) and row["figma_verdict"] == "verified"
```

Policy:
- **`verified` required** for every fixture in the sign-off set (Gate 4: ≥ 5
  representative fixtures). `not-exported` counts as *unverified*, not as a pass —
  a run that never round-tripped through Figma has proven nothing.
- **Drift budget**: track `preview_drift.ssim` across the suite. If the median
  drifts down, `render_preview.py` no longer predicts Figma and every
  preview-based dashboard number inherits that error bar — fix the simulation
  (region heatmaps say where: fonts/gradients/effects) before trusting preview QA
  again.
- Sign-off runs should set `figma_verify.require_text_evidence: true` and enable
  `figma.qa_ocr` so "verified" includes proof that text survived the import.
- CI shape: `python scripts/figma_verify.py --all runs/<suite> --strict` → exit 0
  only when every run is `verified`.

## Evidence layout (per run)

```text
runs/<id>/figma_qa.json                    ← the verdict (schema above)
runs/<id>/figma_verify/exported_aligned.png  scale-normalized/aligned export
runs/<id>/figma_verify/drift_heatmap.png     red = simulation≠Figma, worst regions outlined
runs/<id>/figma_verify/fidelity/diff.png     exported-vs-original diff grid
runs/<id>/figma_verify/drift/diff.png        exported-vs-preview diff grid
runs/<id>/figma_verify/export_ocr.json       OCR of the export (only when run)
```

## Known risks / limits

- **Stale exports**: mtime ordering is the only linkage between an export and a
  design revision. The bridge's roundtrip token prevents cross-run pollution at
  POST time, but a re-run pipeline invalidates an old export silently — hence the
  freshness gate. A future plugin build could embed the roundtrip token in the
  export filename/metadata for exact matching.
- **Export scale**: the plugin currently exports at 1x; the 2x/3x detection is
  defensive. Non-uniform scaling (letterboxed exports) is flagged, force-resized,
  and capped at `degraded`.
- **Alignment is global and coarse**: a single translation is corrected; rotation
  or local reflow is (correctly) left in the score as real drift.
- **Grayscale vs color**: the whole-image drift ssim comes from
  pixel_diff.compare (luma multiscale); region cells are color-aware. A global
  chroma-only shift smaller than the drift threshold could pass the global gate
  while regions still highlight it — read the region list, not just the number.
- **OCR cost**: `--ocr` on a big suite loads OCR models per export; default
  behavior reuses `render_ocr.json` only.
- **Concurrent extension of pixel_diff**: figma_verify tolerates signature/key
  changes (fallback judge, .get access), but if compare's `ssim` semantics are
  redefined, thresholds here inherit that redefinition by design (single bar).
