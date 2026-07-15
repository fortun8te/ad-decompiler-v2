# Production Readiness Gate

Definition of "production ready" for ad-decompiler-v2. No release claim until every
gate below passes. This extends the repo's existing honesty rules (AGENTS.md): a good
screenshot score can never buy off a structural failure.

## Gate 1 — Environment (automated: doctor.py)
- [ ] doctor.py READY with zero FAIL; WARNs only for explicitly-optional tools
- [ ] All required models present: SAM3 ckpt, Flux Fill GGUF ladder (Q4/Q5/Q6),
      t5xxl fp8, clip_l, ae.safetensors, Big-LaMa, gemma-4-12b loaded in LM Studio
- [ ] `lms` CLI available (VRAM eviction feature dependency)

## Gate 2 — Unit/contract tests (automated: pytest)
- [ ] Full CPU suite green: `.venv\Scripts\python.exe -m pytest -q`
- [ ] Plugin mock-E2E green: `node figma-plugin/test/run_e2e.js --all`
      (every golden design.json imports without silent drops in the figma mock)

## Gate 3 — Pipeline benchmark (automated: benchmark.py)
- [ ] Full 16-image benchmark_set run completes with zero crashes and zero
      required-model silent fallbacks (runtime accepted for every image)
- [ ] QA pass rate ≥ 14/16 (was 3/5 on the golden subset pre-fixes)
- [ ] No hard fails of class: background-leakage, duplicate-content, unclean-background,
      inpaint-outside-mask
- [ ] Ghost-text check: zero glyph-residue detections under emitted text layers
- [ ] Every sub-threshold region resolved by raster-slice fallback (looks-right floor),
      with editable_ratio honestly reported per run
- [ ] report.html reviewed by a human for the full set (side-by-side sanity)

## Gate 4 — Real-Figma verification (semi-automated: figma_verify)
- [ ] Bridge → plugin import → export-back loop run on ≥ 5 representative fixtures
- [ ] figma_qa.json verdict "verified" on all 5 (exported-vs-original fidelity above
      threshold AND exported-vs-preview drift below threshold)
- [ ] Font substitutions during import: listed in plugin UI, count ≤ agreed budget,
      zero layers dropped
- [ ] Re-import (replace mode) idempotent: second import produces no orphans/dupes

## Gate 5 — Editability audit (manual, per release)
On 3 diverse fixtures inside Figma:
- [ ] Every text block is a real TEXT node with correct content, authored line breaks,
      plausible font, and editable styling (change a word → looks right)
- [ ] Buttons/pills are shapes with real corner radius + fills (not slices), unless
      honestly marked fallback
- [ ] Icons/logos are vectors or cleanly-masked images; move one → background is clean
      underneath (no baked ghost)
- [ ] Layer names are semantic; hierarchy has meaningful groups; auto-layout only where
      it behaves correctly when resized
- [ ] Manual cleanup time measured and recorded (target: beat Codia's reported
      15–20 min/screen materially)

## Gate 6 — Ops
- [ ] Start Bridge.bat cold-start on a clean shell reaches ready state unattended
- [ ] -SelfTest passes post model changes (gemma-4-12b identity, VRAM eviction path)
- [ ] VRAM telemetry in runtime_report.json shows no stage exceeding budget; no OOM
      across the full benchmark
- [ ] runs/ disk management: prune tooling available, docs-referenced runs protected

## Current status (2026-07-15)
Gate 1: PASS (verified today, minor WARNs: potrace intentionally absent).
Gates 2–6: PENDING — 20 improvement agents in flight; integration pass scheduled
immediately after they land (order: pytest → mock-E2E → doctor → full benchmark with
report.html → figma-verify loop on 5 fixtures → editability audit).
