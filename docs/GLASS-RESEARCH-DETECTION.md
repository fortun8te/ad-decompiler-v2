# Glassmorphism detection/estimation research — from-pixels α, fill color, blur σ

Companion to `docs/GLASS-RESEARCH-FIGMA.md` (Figma/schema/QA-renderer emission side, already
verified — read that first, not duplicated here). This doc covers the **other half**: given a
photo region that might be a frosted-glass panel, how do we *detect* it and *estimate*
`fill.opacity` + `background-blur radius` from pixels alone, with no ground truth. Research only,
CPU only, no pipeline code changed.

Target case: `docs/HARD-CREATIVES-SPEC.md` **H18** — UPFRONT oats ad, two frosted-glass info chips
bottom-right over a lifestyle photo: rounded rect, low-opacity white fill, background blur, white
label/value text on top. User-locked requirement: "opacity/glass elements must be recognized and
rebuilt in Figma super closely — real fill-opacity + background-blur reconstruction, not a raster
slice."

---

## 1. Archaeology — what was tried, why it was dropped, what's salvageable

**Re-derived directly** (`git log --all --oneline -S glass -i`, `-S opacity -S blur`,
`git stash list`, `git fsck --lost-found`):

- No commit, dangling commit, dangling tree, or the one existing stash
  (`backup/cursor-regression-before-clean-baseline-2026-07-16`) contains any code touching
  glass/frosted/translucency/`background_blur`/`fill_opacity`. `grep -rn glass|frosted src/*.py`
  is zero hits today, and was zero hits in every recoverable historical/dangling object too.
  **There was no code attempt to recover.** The "tried" in "tried and dropped" refers to a
  planning-level decision, not an abandoned implementation.
- The actual paper trail is in `docs/CRITIC-REVIEW-2026-07-15.md` (line ~391): glassmorphism is
  named alongside annotation-vectorization and OmniSVG as a **"demo-differentiator ahead of
  F1/F5"** — i.e. flagged as a mis-prioritization: converts a few inspiration-corpus items from
  YELLOW to GREEN but the base-correctness GREENs ("F1/F5") aren't actually solid yet.
- `docs/FEATURE-PLAN-2026-07-16.md` §W2 makes the drop explicit and sets the reopen bar:
  > "The INSPO audit flagged it as a differentiator (~5 ads: 018, 022, 025, 057, 129) and Figma
  > supports it natively. But the brief states it was **tried and dropped**. Respect that. Reopen
  > only after the base is strict-GREEN and **only if a clean detector (local blur + translucency
  > over busy bg) proves reliable — a false positive bakes blur over sharp content.**"
- **What's salvageable:** everything on the Figma/schema/QA-renderer side — confirmed
  production-ready with zero code changes needed (`GLASS-RESEARCH-FIGMA.md` §1-3): `code.js`
  already round-trips a `background-blur` effect spec to a native `BACKGROUND_BLUR` Figma effect;
  fill-opacity vs. layer-opacity are already correctly separable in the schema; only
  `build_design_json.py` (thread `fill_opacity`/`background_blur_radius` off a candidate) and
  `render_preview.py` (simulate `background-blur` at composite time so QA doesn't false-fail) need
  small additive hooks — both proposed as diffs, not applied, since those files are hot. **The one
  missing piece, and the reason it's still WON'T-ship, is exactly the "clean detector" this doc
  addresses:** nothing upstream estimates `fill_opacity`/`background_blur_radius` from pixels. This
  doc's algorithm (§2-3) is a candidate for that detector; §4 defines the reliability gate the
  reopen condition demands.
- **Differentiator confirmation (independent of the drop):** `docs/RESEARCH-CODIA-GAP-2026-07-16.md`
  line 7, first-party empirical, across all 130 nodes in 4 real Codia teardowns: *"zero effects,
  zero gradient fills, **zero non-default opacity/blend modes**."* `docs/CODIA-PARITY-SPEC.md`
  confirms the only non-default opacity Codia ever emits is `opacity: 0` on inert wrapper groups
  (not real translucency). **Codia does not reconstruct transparency at all, in any observed
  output.** If this pipeline ever ships a working glass detector, it is a genuine, measured
  differentiator, not a marketing claim — but per W2, it only ships after base correctness, and
  only if the detector clears the false-positive bar below.

---

## 2. Detection / estimation algorithm

### 2.1 Setup

Precondition: the pipeline already produces an **inpainted clean background plate** for occluded
regions (this exists today — `src/inpaint.py`, Big-LaMa backend, used for peel/occlusion recovery
elsewhere in the pipeline). For a candidate glass region, we have two aligned images:

- `orig` — the original ad crop, region `[x0:x1, y0:y1]`, containing the (possibly glass) overlay.
- `bg` — the same region from the inpainted plate: LaMa's best guess at what the photo looks like
  with the overlay removed, **not blurred**.

The forward (compositing) model for an ideal frosted panel, per-pixel, per-channel:

```
observed(x,y) = alpha * fill_color + (1 - alpha) * blur_sigma(bg)(x,y)
```

`alpha` (opacity, 0-1) and `fill_color` (RGB) are assumed **constant across the region** — true
for a flat-fill rounded-rect chip, which is the only glass shape this pipeline needs to emit
(§4 of `GLASS-RESEARCH-FIGMA.md`: shape/card/button routing, not arbitrary raster). `sigma` (the
backdrop blur's Gaussian standard deviation) is also one shared scalar per region.

### 2.2 Why this is solvable without deconvolution

Naively this looks like a blind-deconvolution problem (recover the blur kernel from a single
blurred+composited observation) — hard and ill-posed. It isn't, here, because we have the
**unblurred bg plate already** (from inpainting) as ground truth for what's *underneath*. That
turns blur estimation into a 1-D search: sweep candidate `sigma` values, forward-blur the known
`bg` with each, and see which one — combined with the best-fit `alpha`/`fill_color` — reproduces
`observed`. No kernel recovery from the blurred image itself is needed.

### 2.3 Per-sigma least-squares fit (closed form, not blind)

The forward equation is **bilinear** in `(alpha, fill_color)` (alpha multiplies fill_color), which
would normally need nonlinear optimization. Reparametrize to make it linear: let
`beta_c = alpha * fill_c` (3 unknowns, one per channel) and keep `alpha` as a 4th unknown. Subtract
`blur_sigma(bg)` from both sides:

```
observed_c(x,y) - bg_blur_c(x,y) = beta_c - alpha * bg_blur_c(x,y)
```

This is linear in `[beta_r, beta_g, beta_b, alpha]` for a *fixed* candidate `sigma` (since
`bg_blur` is then a known constant per pixel). Stack every pixel × channel in the region into one
linear system and solve by ordinary least squares (`numpy.linalg.lstsq`). Recover
`fill_c = beta_c / alpha`. Compute the residual sum of squares (RSS) for this `sigma`.

Sweep `sigma` over a grid (e.g. 0.5px steps from 0.5 to 20px — covers the plausible glass range at
ad-chip scale) and take the `sigma` with minimum RSS. This is exact for the noiseless case and a
good MLE approximation under i.i.d. pixel noise; it costs one `lstsq` solve (4 unknowns) plus one
Gaussian blur per grid point — cheap, CPU-only, no GPU/ML needed.

### 2.4 Verified Figma blur-radius ↔ Gaussian-σ conversion

The algorithm above recovers `sigma` in the sense PIL's `ImageFilter.GaussianBlur(radius=sigma)`
uses it — **PIL's `radius` parameter *is* the Gaussian standard deviation directly** (per PIL's
own filter semantics), not a "visual radius." Figma's `BackgroundBlurEffect.radius` (the value
`design.json`/`figma-plugin/code.js` actually emits, per `GLASS-RESEARCH-FIGMA.md` §1.5) is a
**different, larger number** — it needs conversion before being written back into `design.json`.

Verified via Bjango's blur-radius cross-tool comparison (`bjango.com/articles/blurradiuscomparison`,
a reverse-engineered/measured comparison of blur radius definitions across CSS, iOS, Android,
Sketch, and **Figma**, covering background blur, drop shadow, and layer blur):

- CSS's own spec-defined relationship: **σ = css_blur_radius / 2** (CSS Filter Effects / SVG
  `feGaussianBlur` convention; independently confirmed via MDN/`dbaron.org`).
- Bjango's measured finding: **Figma's blur radius is a constant 1.136364× larger than the CSS
  box-shadow-equivalent radius**, and this factor is the *same* across all three Figma blur types
  (background blur, drop shadow, layer blur) — i.e. Figma doesn't need a different conversion
  per-effect-type.

Chaining these: `figma_radius = 1.136364 * css_radius = 1.136364 * (2 * sigma) = 2.272728 * sigma`.

**Conversion factor: `sigma ≈ figma_radius / 2.273`, or `figma_radius ≈ 2.273 * sigma`.**

This matters in two directions:
- **Estimation → emission:** the `sigma` this doc's algorithm recovers from pixels must be
  multiplied by ~2.273 before being written as `background_blur_radius` into `design.json`'s
  effects list (the field `GLASS-RESEARCH-FIGMA.md` §2.2 proposes threading through
  `build_design_json.py`).
- **QA-renderer fidelity:** `GLASS-RESEARCH-FIGMA.md` §3.2's proposed `render_preview.py` patch
  currently plugs the raw Figma `radius` straight into `PIL.ImageFilter.GaussianBlur(radius)`
  with **no conversion** — per that doc's own caveat ("this only approximates Figma's blur kernel
  ... exact pixel match isn't the goal, SSIM tolerance is"), that's an acknowledged approximation,
  but per this doc's finding it is not just "not exact," it is roughly **2.27× oversized**
  (PIL would blur ~2.27× more aggressively than real Figma at the same stored radius value).
  Worth a one-line fix if/when that patch is actually applied: divide by 2.273 before calling
  `GaussianBlur`. Flagging, not fixing (that file is out of scope here).

### 2.5 Text-inside-glass exclusion

H18's chips have white label/value text on top of the glass fill. Text pixels violate the
region-constant-color assumption badly (they're near-black-to-white strokes, not the translucent
fill) and would corrupt the least-squares fit if included. Fix: run OCR/text-region detection
(already available in the pipeline — `src/ocr.py`) on the candidate region *first*, and exclude
any pixel inside a detected text bounding box (plus a small dilation margin, ~2-3px, to cover
anti-aliased glyph edges) from both the `observed` and `bg` samples fed into the `lstsq` fit. The
fit only ever needs the *fill* pixels, not the *text* pixels — glass color/opacity estimation and
text-color/text-recognition are already separate concerns elsewhere in the pipeline; this is
purely "don't let text pixels leak into the glass regression."

---

## 3. Detection trigger — is this region glass, solid, or an image at all?

The algorithm above is a *fitter*, not a *detector* — it will happily return some `(alpha,
fill_color, sigma)` for any region, including ones that aren't glass. Two-part gate before
trusting its output:

### 3.1 Model comparison: glass-model RSS vs. flat-solid-model RSS

Fit a trivial competing hypothesis — "this region is a flat solid color" (`observed(x,y) = mean
color`, RSS = variance of the region) — and compare it to the best glass-model RSS from §2.3.
A real glass panel has faint-but-real residual structure from the (blurred) background showing
through; a solid chip does not, so the *ratio* `glass_rss / solid_rss` is the discriminating
signal — low ratio means the bg-dependent model explains far more variance than "it's just one
flat color," which only happens if there's genuine bg leakage (i.e. actual translucency).

### 3.2 The α→1 degeneracy is itself the "not glass" signal

As `alpha → 1`, the forward model's dependence on `bg`/`sigma` vanishes (`(1-alpha) → 0`), so
`sigma` becomes unidentifiable — the fit can return *any* `sigma` with equally-low RSS once alpha
saturates near 1 (confirmed empirically in §5, Case D below: recovered `alpha=1.000` exactly, but
`sigma` came back as an arbitrary grid value with no bearing on truth, because there is no
`sigma_true` for a fully opaque region). **This is a feature, not a bug of the algorithm — it's
the natural mathematical signature of "this is not glass."** Detection rule:

- `alpha_est >= ~0.97` (or, more robustly, since near-opacity also degrades sigma identifiability
  well before exactly 1.0: check whether RSS stays low across a *wide range* of candidate sigmas,
  not just at one minimum — a flat RSS-vs-sigma curve means sigma is unidentified, i.e. solid) →
  treat as **solid**, do not emit `background-blur`. This is exactly the CONTRACT.md-aligned
  degrade `GLASS-RESEARCH-FIGMA.md` §4 already specifies: emit a plain solid-color rect at the
  fitted `fill_color`.
- `alpha_est` well below 1 (region has real bg-mixing) **and** the glass/solid RSS ratio from
  §3.1 is low (glass model meaningfully outperforms flat-solid) **and** `sigma_est` sits at a
  clear, well-defined RSS minimum (not flat/degenerate) → treat as **glass**, emit
  `fill.opacity = alpha_est`, `background-blur radius = 2.273 * sigma_est` (§2.4).
- Anything ambiguous in between (indistinct RSS minimum, alpha in a gray zone, e.g. 0.85-0.97) →
  **low confidence**, fall back to solid per the existing CONTRACT.md degrade
  (`GLASS-RESEARCH-FIGMA.md` §4) — per the FEATURE-PLAN-2026-07-16 §W2 reopen bar, a false
  positive (baking blur over sharp content that was never actually translucent) is the failure
  mode to bias hard against, so the threshold should be conservative.
- **Not-a-panel-at-all case** (region is actually part of the photo, not an overlay): this is a
  *prior* question the detector doesn't answer — it assumes it's already been handed a candidate
  overlay region by upstream element/shape detection (rounded-rect candidate from
  `element_detect.py`/`reconstruct.py`), same as every other shape-routing decision in the
  pipeline. Out of scope for the pixel-level fitter; in scope for whatever stage proposes "here's
  a shape candidate, is its fill glass or solid."

---

## 4. Validation — synthetic composites, actual numbers

Ran on CPU (`.venv` — numpy 2.5.1, Pillow 10.4.0, scipy 1.18.0), no GPU. Synthetic "photo": smooth
low-frequency color field + Gaussian pixel noise (σ=18) + 25 random sharp-edged colored blobs (to
give a real background-blur something to visibly remove — flat backgrounds would make the
detection trivial and prove nothing). Sigma grid swept 0.5-20px at 0.5px steps.

**Recovery accuracy** (`observed = alpha*fill + (1-alpha)*GaussianBlur(bg, sigma)`, then fit via
§2.3):

| Case | alpha true | sigma true | fill true | alpha est | sigma est | alpha err | sigma err | color err (L2, 0-255) |
|---|---|---|---|---|---|---|---|---|
| A: typical H18 chip | 0.18 | 8.0 | white | 0.180 | 8.0 | 0.0001 | 0.00 | 4.51 |
| B: opaque frosted panel | 0.28 | 14.0 | white | 0.280 | 14.0 | 0.0004 | 0.00 | 3.30 |
| C: low opacity/low blur | 0.12 | 3.0 | white | 0.120 | 3.0 | 0.0002 | 0.00 | 6.44 |
| D: solid opaque (control) | 1.00 | n/a | off-white | 1.000 | **16.5 (arbitrary)** | 0.0000 | n/a | 0.00 |
| E: small H18-scale chip, tinted fill | 0.20 | 10.0 | (250,248,235) | 0.200 | 10.0 | 0.0003 | 0.00 | 3.77 |

Alpha recovered to within 0.0004 in every real-glass case; sigma recovered **exactly** to the
0.5px grid resolution in every real-glass case; color error (3-6/255) is consistent with the
injected pixel noise floor (σ=18 per channel on the underlying photo), not systematic bias. Case D
confirms the §3.2 degeneracy empirically: alpha correctly saturates to 1.000, but the "recovered"
sigma (16.5) is meaningless noise from an unidentifiable parameter, exactly as predicted.

**Detection-trigger separation** (§3.1 ratio test, `glass_rss / flat_solid_rss`):

| Case | alpha | sigma | glass RSS | solid RSS | ratio | alpha est |
|---|---|---|---|---|---|---|
| glass 18%/σ8 | 0.18 | 8.0 | 2,648 | 20,005,059 | 0.0001 | 0.180 |
| glass 28%/σ14 | 0.28 | 14.0 | 2,592 | 9,607,605 | 0.0003 | 0.280 |
| solid opaque | 1.00 | — | 0 | 0 | 0.000 | 1.000 |
| near-opaque 95% | 0.95 | 5.0 | 2,801 | 77,370 | 0.036 | 0.950 |
| near-invisible glass 5% | 0.05 | 8.0 | 2,608 | 23,360,921 | <0.0001 | 0.049 |

Real glass cases separate from solid by 3-4 orders of magnitude in ratio; even a near-invisible
5%-opacity panel is detected cleanly (ratio ~1e-4, alpha recovered to 0.049 vs. true 0.05). The
near-opaque 95% case is the interesting edge: ratio rises to 0.036 (bg leakage getting faint) but
is still two orders of magnitude below the clean solid case's 0 — meaning even at 95% opacity the
fit correctly identifies *some* residual translucency, which is arguably correct (it *is* slightly
translucent) but is exactly the alpha-near-1 zone §3.2 flags for a conservative solid-fallback
decision regardless of what the ratio says, since a 95%-opacity chip is visually indistinguishable
from solid and not worth risking a wrong blur guess over.

**Script:** validation code is in the scratchpad
(`glass_validate.py`, ~140 lines, self-contained, not committed to the repo — research artifact
only per the CPU-only/research-only mandate). Rerunning it reproduces the table above exactly
(fixed RNG seed).

### What this validation does *not* cover (real-image caveats, for a future implementer)

- Real inpainted `bg` plates are LaMa outputs, not ground truth — they'll have their own
  reconstruction error/hallucinated texture where the real photo is fully occluded by the chip.
  This validation used a perfect `bg`; a real detector's accuracy is upper-bounded by inpainting
  quality, not by this fitting algorithm.
- No JPEG/compression artifacts, no anti-aliased chip edges (only interior region pixels were
  fit), no actual OCR-based text exclusion exercised (§2.5 is a design, not benchmarked here).
- Real photo content will have real edges much sharper/more structured than the synthetic blobs;
  worth re-validating on 1-2 actual H18-style crops (real UPFRONT ad or a similar frosted-chip
  inspiration image + a real LaMa inpaint) before this graduates past "algorithm design" to
  "candidate for implementation," per the FEATURE-PLAN-2026-07-16 §W2 reopen bar's "clean detector
  ... proves reliable" requirement.

---

## 5. Summary

| Question | Answer |
|---|---|
| Was glass ever actually implemented and reverted? | No — zero code in any reachable git object (including stash/dangling). "Tried and dropped" was a planning decision (CRITIC-REVIEW-2026-07-15, FEATURE-PLAN-2026-07-16 §W2), not an abandoned patch. |
| Reopen condition | Base strict-GREEN + "a clean detector (local blur + translucency over busy bg) proves reliable." |
| Is it a real differentiator vs. Codia? | Yes, confirmed empirically: Codia emits zero non-default opacity/effects across 130 real teardown nodes (RESEARCH-CODIA-GAP-2026-07-16, CODIA-PARITY-SPEC). |
| Figma-side readiness | Fully wired today, zero plugin changes needed (GLASS-RESEARCH-FIGMA.md). |
| Missing piece | Pixel-level detector/estimator — this doc's §2-3, validated synthetically in §4. |
| Blur-radius conversion | `figma_radius ≈ 2.273 * sigma` (Bjango cross-tool measurement × CSS's σ=radius/2 spec definition). |
| Validated? | Yes, synthetically — clean separation and near-exact recovery on 5 cases. Needs 1-2 real-photo validation passes before implementation, per reopen bar. |
