# Clean-background benchmark

This is a small CPU-only benchmark for one specific claim: after foreground removal, did the system recreate the hidden background correctly without modifying anything outside its removal mask?

It creates deterministic synthetic pairs:

- `clean_background.png` is the known target;
- `composite_input.png` is that background with a button, card, or badge pasted over it;
- `removal_mask.png` is the exact non-transparent foreground matte.

The generated background is compared with the clean target **only inside** that known mask. Separately, the generated image must be byte-for-byte identical to `composite_input.png` outside the mask. Exterior changes never improve the inside score.

## Run

Smoke-test the benchmark itself with the oracle (the oracle is not an inpainting result):

```bash
python3 scripts/background_benchmark.py \
  --output runs/background-benchmark-oracle \
  --method oracle
```

Run a deliberately modest local baseline:

```bash
python3 scripts/background_benchmark.py \
  --output runs/background-benchmark-telea \
  --method telea
```

This has no GPU or model-server dependency. A failing OpenCV result is useful baseline evidence, not a harness failure.

## RTX backend bakeoff

To test the repo's actual inpainting seam, run one backend at a time on the same pairs. The bakeoff loads the supplied YAML or JSON configuration, overrides only `inpaint.mode`, then calls `src.inpaint.inpaint_array` for every case.

```bash
python3 scripts/background_benchmark.py \
  --output runs/background-bakeoff-big-lama \
  --bakeoff-mode big-lama \
  --config config.yaml
```

The available requested modes are `big-lama`, `flux_comfy`, `powerpaint`, and `opencv`. Run each separately; this makes GPU/model availability, quality, and runtime cost directly comparable instead of letting `auto` hide the route.

`summary.json`, `metrics.json`, and `summary.md` record both the requested and selected backend. For example, a requested `flux_comfy` result selected as `big-lama` or `opencv-telea` fails with `backend-substitution`, even if its pixels meet the quality thresholds. This is intentional: a fallback is useful evidence but is not a Flux result.

Use this only when a substitution is deliberately acceptable:

```bash
python3 scripts/background_benchmark.py \
  --output runs/background-bakeoff-flux-fallback-observed \
  --bakeoff-mode flux_comfy \
  --config config.yaml \
  --allow-backend-substitution
```

The flag keeps the requested-versus-selected record; it only removes the automatic acceptance failure. Backend exceptions are recorded as `backend-error` and fail the run. The existing byte-exact outside-mask gate remains active for every route.

## Score an actual backend

First run the tool being evaluated on each saved `composite_input.png` with its paired `removal_mask.png`. Save its outputs as:

```text
candidate-output/
  flat-01.png
  gradient-01.png
  texture-01.png
  photo_like-01.png
  ...
```

Then score them untouched:

```bash
python3 scripts/background_benchmark.py \
  --output runs/background-benchmark-candidate \
  --candidate-dir candidate-output
```

The command exits `0` only when every case passes. It exits `2` for a quality or outside-mask failure.

## Artifacts and acceptance

Each case is self-contained under `cases/<id>/`:

- clean input, composite input, and removal mask;
- `generated_background.png` exactly as scored;
- `inside_diff.png` against the known clean target;
- `outside_diff.png` against the original composite;
- `metrics.json` with per-case values and reasons.

The root has `manifest.json`, `summary.json`, and a review-ready `summary.md` table. Default acceptance requires all cases to meet:

- inside-mask MAE at most `8`;
- inside-mask PSNR at least `30 dB`;
- inside-mask SSIM at least `0.90`;
- zero changed pixels outside the mask.

Thresholds are command-line options. Change them only with a documented target benchmark; do not relax the zero outside-mask gate.

## Scope

Synthetic fixtures make hidden pixels observable, which is impossible for a normal ad screenshot without source artwork. They cover flat, gradient, texture, and photo-like backgrounds and a few common foreground shapes. They do not replace a held-out, source-layer benchmark from real Figma ads; use both before accepting a production inpainting route.
