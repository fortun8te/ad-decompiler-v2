# Font-Matcher Evaluation: Lens (mixfont) vs. current local shape-matcher

**Status:** Evaluation spike (INTEL ONLY -- no `src/` changes, no integration).
**Date:** 2026-07-15
**Author:** eval agent
**Verdict:** **ADOPT WITH CAVEATS** -- technically a clear win over the current
shape-matcher, but it *only* works on **single-word crops** (not text lines), and
its **non-commercial license is a hard blocker** that must be resolved before it
can ship in a commercial product.

---

## 1. TL;DR

- We evaluated **Lens** (`github.com/mixfont/lens`): a ResNet18 font-recognition
  model, **983 output classes**, trained on open-source fonts and mapping directly
  to **Google Fonts** family names (Figma can insert these natively).
- The repo is **alive and builds cleanly**. Bundled `font_classifier.pt` (46 MB)
  loads and runs on **CPU** with **zero network access** required for inference.
  (The fallback repo `Storia-AI/font-classify` was therefore *not* needed; see 9.)
- Ran it end-to-end on **28 real crops** (14 text-lines + the same 14 as
  single-word crops) pulled from the golden runs `009` and `052`.
- **The single most important finding:** Lens is a *single-word* model.
  - Fed **full text-line** crops (multiple words): **29% class accuracy** -- it
    collapses to random display/handwriting fonts. Useless.
  - Fed **single-word** crops (its documented, intended input): **100% class
    accuracy (14/14)** on this sample, with sensible family-level matches.
- On the single-word input, Lens **fixes the exact bug that motivated this eval**:
  - `009:L4` "assortiment" -- the swash mismatch. Our pipeline chose **Gabriola**
    (a calligraphic swash font) for plain sans body text. **Lens says `Inter` 1.00**
    -- correct, and Inter is the near-perfect Google-Fonts stand-in for X's Chirp.
  - `009:L14` "woensdag" -- our pipeline chose **Cascadia Code** (monospace).
    **Lens says `Inter` 1.00**.
- **Latency:** ~**8 ms per crop on CPU** (median 8 ms, max 14 ms cold). Model load
  **~1-2 s one-time**. Negligible cost. **No GPU / no VRAM needed.**
- **License:** *"personal, academic, and non-commercial use only ... Commercial
  use is strictly prohibited without written permission."* This is the gating issue.

---

## 2. What Lens is

| Property | Value |
|---|---|
| Repo | `github.com/mixfont/lens` (standalone open-weights release, Mar 2026) |
| Architecture | ResNet18 (torchvision), final FC swapped to 983 classes |
| Model file | `model/font_classifier.pt`, **46 MB**, plain state-dict checkpoint |
| Classes | **983 font families** (`model/classes.json`), Google-Fonts + a few extra OSS |
| Metadata | `model/font_metadata.json` (1.3 MB) -- per-family Google Fonts `category`, weights, styles, CDN URLs |
| Input tensor | **128 x 384** (H x W) -- a *wide* rectangle, i.e. built for word/line strips, grayscale-normalized ImageNet stats |
| Intended input | The **largest single word** in an image (its own pipeline runs Tesseract OCR to find it first) |
| Output | Top-K `{name, score, fonts:[{full_name, style, weight, url}]}`, scores are softmax probs |
| Deps | `torch`, `torchvision`, `Pillow`, `pytesseract` (+ Tesseract binary -- only for its OCR stage, which we bypass) |

Model, class map, and font metadata are all **bundled in the repo** -- inference is
fully offline. The only network dependency in the vendored code is (a) downloading
the input image from a URL and (b) the font CDN URLs in the output metadata; both
are avoidable (we feed local `PIL.Image` crops directly).

---

## 3. How it was evaluated

Script: **`scripts/eval_lens_fonts.py`** (run from repo root with the existing venv).

- Installed **nothing** and **touched no requirements file**. `torch 2.10.0+cu128`,
  `torchvision 0.25.0+cu128`, `Pillow 10.4.0`, and `pytesseract` were already in
  `.venv`. Lens pins `torch==2.10.0` -- exact match. (Its `Pillow==12.1.1` pin is
  irrelevant; 10.4.0 runs the preprocessing fine.)
- **Forced CPU** (`CUDA_VISIBLE_DEVICES=""` + monkeypatched `lens_inference.pick_device`)
  to avoid GPU contention, per the task constraint.
- **Bypassed Lens's Tesseract OCR entirely.** We already have word/line geometry in
  each run's `ocr.json`, so the script crops directly from `normalized.png` and calls
  Lens's `run_model()` on the crop. This is both faster and a fairer test of the
  *classifier* in isolation.
- For each ground-truth line we produced **two** crops: the **full line box** and the
  **largest word box** inside it (from `line.words[*].box`). This is what surfaced the
  line-vs-word cliff.
- Crops saved to `work/lens/eval_out/crops/`; full results in
  `work/lens/eval_out/lens_eval_results.json`.

### Ground truth (eyeballed -- assumptions stated)

- **`009`** (X/Twitter post, Dutch): the entire UI chrome *and* the tweet body are
  set in **Chirp**, X's proprietary grotesque **sans-serif** (Helvetica/Arial/Inter
  family). There is no serif, script, or monospace anywhere in the image. So *every*
  `009` line's ground-truth class = **sans-serif**. The Google-Fonts-correct answer is
  a clean grotesque: **Inter / Roboto / Arimo / Karla**.
- **`052`** (hair-product ad "wavy"): the headline ("No need to ruin your hair to have
  perfect curls") and green sub-headline are a **high-contrast transitional/Didone
  serif**; the "Before"/"After" pill chips are a **bold grotesque sans**. Labeled
  per-line accordingly.

"Class match" below = did the matcher pick the right **typeface class**
(serif / sans-serif / display / handwriting / monospace)? We score class rather than
exact family because there is no licensed ground-truth family for these proprietary
ad fonts -- class correctness is the honest, defensible metric, and it is exactly
the axis on which the current matcher visibly fails (sans -> swash, sans -> mono).

---

## 4. Results

### 4a. Line crops -- Lens FAILS (4/14 = 29% class accuracy)

Feeding a whole multi-word line squashes per-glyph proportions into the 128x384 input
and Lens hallucinates display/handwriting fonts with high confidence:

| Run:Line | Text | GT class | Our pipeline | Lens top-1 | Lens top-5 (score, class) |
|---|---|---|---|---|---|
| 009:L0 | Post | sans-serif | Arial `sans-serif` OK | Epunda Sans `sans-serif` OK | Epunda Sans 0.45 (sans-serif); Otomanopee One 0.27; Comme 0.09; Mulish 0.09; Libre Franklin 0.08 |
| 009:L2 | LAATSTE SITE WIDE SALE VAN 2026 | sans-serif | Candara `sans-serif` OK | Miltonian `display` **X** | Miltonian 0.77 (display); Zen Tokyo Zoo 0.15; Agu Display 0.07 |
| 009:L3 | De Vakantiegeldsale komt eraan... | sans-serif | Bahnschrift `sans-serif` OK | Ole `handwriting` **X** | Ole 0.49 (handwriting); Neonderthaw 0.34; Sassy Frass 0.08 |
| 009:L4 | korting krijgt op het volledige... | sans-serif | Gabriola `handwriting` **X** | Miltonian `display` **X** | Miltonian 0.56 (display); Bonbon 0.21; Cherry Swash 0.11 |
| 009:L5 | Daarbovenop krijgen de eerste 500... | sans-serif | Calibri `sans-serif` OK | Hanalei `display` **X** | Hanalei 0.88 (display); Neonderthaw 0.08; Sassy Frass 0.02 |
| 009:L7 | Schrijf je nu in en mis geen... | sans-serif | Arial `sans-serif` OK | Neonderthaw `handwriting` **X** | Neonderthaw 0.77 (handwriting); Tilt Prism 0.10; Sassy Frass 0.09 |
| 009:L8 | 05:00 PM . 12-05-2026 - 121K... | sans-serif | Leelawadee UI `sans-serif` OK | Butterfly Kids `handwriting` **X** | Butterfly Kids 0.66 (handwriting); Miltonian 0.10; Reenie Beanie 0.06 |
| 009:L14 | woensdag 20 mei om 20:00 uur. | sans-serif | Cascadia Code `monospace` **X** | Macondo `display` **X** | Macondo 0.55 (display); Agu Display 0.09; Miltonian 0.09 |
| 052:L0 | No need to ruin your hair | serif | Cambria `serif` OK | Sofia `handwriting` **X** | Sofia 0.78 (handwriting); Arima 0.11; Agu Display 0.04 |
| 052:L1 | to have perfect curls | serif | Georgia `serif` OK | Stardos Stencil `display` **X** | Stardos Stencil 0.82 (display); Baskervville 0.15 (serif); Emilys Candy 0.03 |
| 052:L2 | Before | sans-serif | Arial `sans-serif` OK | Schibsted Grotesk `sans-serif` OK | Schibsted Grotesk 0.78 (sans-serif); Special Gothic 0.12; Stack Sans Headline 0.05 |
| 052:L3 | After | sans-serif | Gadugi `sans-serif` OK | Schibsted Grotesk `sans-serif` OK | Schibsted Grotesk 0.96 (sans-serif); Darker Grotesque 0.02; Mona Sans 0.01 |
| 052:L8 | This natural curl cream | serif | Cambria `serif` OK | Rye `display` **X** | Rye 0.98 (display); Milonga 0.02; Lancelot 0.00 |
| 052:L9 | is all you need | serif | Georgia `serif` OK | Bacasime Antique `serif` OK | Bacasime Antique 0.78 (serif); Stardos Stencil 0.17; Baskervville 0.04 |

Note the single-word lines it *did* get right ("Before", "After") -- consistent with
the word-level story: short crops work, multi-word lines don't. Also note how
**confidently wrong** it is (0.77-0.98 on garbage), so softmax score is not a usable
line-crop reject signal.

### 4b. Single-word crops -- Lens NAILS IT (14/14 = 100% class accuracy)

Same 14 lines, but cropped to the **largest word** (Lens's intended input):

| Run:Line | Word | GT class | Our pipeline (line) | Lens top-1 | Lens top-3 (score, class) |
|---|---|---|---|---|---|
| 009:L0 | Post | sans-serif | Arial `sans-serif` OK | Epunda Sans `sans-serif` OK | Epunda Sans 0.62 (sans-serif); Mulish 0.19; Libre Franklin 0.07 |
| 009:L2 | LAATSTE | sans-serif | Candara `sans-serif` OK | Karla `sans-serif` OK | Karla 0.50 (sans-serif); Sawarabi Gothic 0.19; Golos Text 0.15 |
| 009:L3 | Vakantiegeldsale | sans-serif | Bahnschrift `sans-serif` OK | Inter `sans-serif` OK | Inter 0.99 (sans-serif); Roboto 0.01; Atkinson Hyperlegible 0.00 |
| **009:L4** | **assortiment** | **sans-serif** | **Gabriola `handwriting` X** | **Inter `sans-serif` OK** | **Inter 1.00 (sans-serif)**; Roboto 0.00; Golos Text 0.00 |
| 009:L5 | Daarbovenop | sans-serif | Calibri `sans-serif` OK | Inter `sans-serif` OK | Inter 1.00 (sans-serif); Atkinson Hyperlegible 0.00; Liter 0.00 |
| 009:L7 | Schrijf | sans-serif | Arial `sans-serif` OK | Inter `sans-serif` OK | Inter 0.84 (sans-serif); Roboto 0.12; DM Sans 0.04 |
| 009:L8 | 12-05-2026 | sans-serif | Leelawadee UI `sans-serif` OK | Inter `sans-serif` OK | Inter 1.00 (sans-serif); Roboto Mono 0.00; Funnel Sans 0.00 |
| **009:L14** | **woensdag** | **sans-serif** | **Cascadia Code `monospace` X** | **Inter `sans-serif` OK** | **Inter 1.00 (sans-serif)**; Barlow 0.00; Zalando Sans 0.00 |
| 052:L0 | your | serif | Cambria `serif` OK | Bacasime Antique `serif` OK | Bacasime Antique 0.90 (serif); Gloock 0.04; DM Serif Display 0.01 (serif) |
| 052:L1 | perfect | serif | Georgia `serif` OK | Bacasime Antique `serif` OK | Bacasime Antique 0.98 (serif); Vidaloka 0.02 (serif); Stardos Stencil 0.00 |
| 052:L2 | Before | sans-serif | Arial `sans-serif` OK | Stack Sans Headline `sans-serif` OK | Stack Sans Headline 0.58 (sans-serif); Schibsted Grotesk 0.25; Special Gothic 0.10 |
| 052:L3 | After | sans-serif | Gadugi `sans-serif` OK | Schibsted Grotesk `sans-serif` OK | Schibsted Grotesk 0.96 (sans-serif); Darker Grotesque 0.02; Mona Sans 0.01 |
| 052:L8 | natural | serif | Cambria `serif` OK | Bacasime Antique `serif` OK | Bacasime Antique 0.99 (serif); Baskervville 0.00 (serif); Stardos Stencil 0.00 |
| 052:L9 | need | serif | Georgia `serif` OK | Bacasime Antique `serif` OK | Bacasime Antique 1.00 (serif); Gloock 0.00; Oranienbaum 0.00 (serif) |

### 4c. Scoreboard

| Matcher / input | Class accuracy (this 14-line sample) |
|---|---|
| **Lens -- single-word crops** | **14/14 = 100%** |
| Current pipeline shape-matcher (line-level) | 12/14 = 86% |
| Lens -- full-line crops | 4/14 = 29% |

---

## 5. Accuracy impressions

- **Lens (word input) is genuinely good and clearly better at the *family* level than
  our shape-matcher.** For X's Chirp it repeatedly returns **Inter** (often at
  probability 1.00) and **Roboto** -- which is exactly what a human designer would pick
  as the Google-Fonts substitute. Our shape-matcher, matching against *local Windows
  fonts*, lands on Candara / Bahnschrift / Leelawadee UI -- right class, wrong feel, and
  not even a Google Font Figma can insert.
- **Both fix the serif headline** in `052` (our pipeline via Cambria/Georgia, Lens via
  Bacasime Antique / Baskervville). Where our matcher shines is that it never confuses
  serif with sans at the class level here (86%). Its failures are *within-or-near class*
  but visually jarring: **sans body text -> Gabriola swash** (`009:L4`, the exact bug in
  the task) and **sans -> Cascadia Code monospace** (`009:L14`). Lens's word path fixes
  both.
- **The catch:** Lens has **no notion of "I'm not sure."** On a bad (multi-word) crop it
  returns display/handwriting fonts at 0.55-0.98 confidence. You cannot threshold your
  way out of the line-crop failure -- you must give it clean single-word input.
- Sample size is small (2 ads, 14 lines) and ground truth is class-level, so treat the
  100% as "strong signal on the cases we care about," not a benchmark. But the mechanism
  is clear and reproducible: **word crop good, line crop bad.**

---

## 6. Latency / CPU / VRAM

Measured on this box (Ryzen 9800X3D, forced CPU, RTX 5080 untouched):

| Metric | Value |
|---|---|
| Inference **per crop** | **~8 ms** (median 8, min 5.5, max 14 cold) on CPU |
| Model load (one-time) | ~1-2 s (0.95-2.2 s across runs) |
| Peak memory | trivial -- 46 MB weights, ResNet18; runs in the pipeline's existing process |
| **VRAM** | **none required** -- CPU inference is plenty fast |
| Per-ad cost estimate | ~15-20 unique text styles x 8 ms = **<200 ms per ad**, CPU-only |

This is small enough to run inline in the pipeline without touching GPU scheduling.
Loading the bundle once and reusing it (Lens already caches via `get_model_bundle`)
keeps the 1-2 s load off the hot path.

---

## 7. License -- THE BLOCKER

From `work/lens/README.md`:

> This project is provided for **personal, academic, and non-commercial use only.**
> ... **Commercial use is strictly prohibited without written permission.**
> For commercial licensing inquiries, please contact: hello@mixfont.com.

ad-decompiler is a commercial-intent product, so **this license prohibits shipping the
bundled weights as-is.** This is independent of how good the model is. Options:

1. **Buy a commercial license / use the Mixfont commercial API** (`mixfont.com/docs`) --
   same model, but now it's a paid network dependency (latency + per-call cost + sends
   customer creatives to a third party). Kills the "local, offline, no-contention" win.
2. **Use Lens strictly as internal/offline intel** (e.g. to build our own labeled data
   or sanity-check other matchers) -- allowed, but not a shippable feature.
3. **Reject and swap to a permissively-licensed model** trained the same way
   (single-word crop -> Google Fonts). See 9.

Do **not** vendor `font_classifier.pt` into a shipped build without resolving this.

---

## 8. Integration difficulty (if licensing is cleared)

**Easy on the technical axis, given one hard requirement: feed it single words.**

- **Input format:** a `PIL.Image` word crop. We already have per-word boxes in
  `ocr.json` (`lines[*].words[*].box = {x,y,w,h}`), so the pipeline can crop the
  largest (or most representative) word per text style and call `run_model()` directly
  -- **no Tesseract, no image download, no network.** ~15 lines of glue.
- **Google Fonts coverage:** output family names *are* Google Fonts names, and each
  prediction carries `{full_name, style, weight}` plus the GF `category`. **Figma can
  insert these natively** -- this is the strategic reason Lens beats local-font shape
  matching, which yields Windows fonts Figma can't use.
- **Suggested integration shape (for the agent rewriting `text_analysis.py`, NOT done
  here):**
  1. Per text style/line, pick the **largest confident word box** from OCR.
  2. Crop it, run Lens top-5.
  3. Take top-1 as the family; keep top-5 as fallback candidates.
  4. Map style/weight from the OCR-measured weight (Lens's own weight guess is coarse).
  5. **Reconcile with the serif/sans class** our current pipeline already computes -- if
     Lens's top-1 class disagrees with a high-confidence class estimate, fall to the
     highest-ranked Lens candidate whose class agrees. (This guards the residual
     line-vs-word risk and the confidently-wrong failure mode.)
- **Risks to watch:** (a) very short words / all-caps / stylized ligatures can still
  wobble; (b) no built-in confidence gate; (c) `Pillow` version drift (repo pins 12.1.1,
  we run 10.4.0 -- fine today, pin-check on upgrade); (d) `torch.load` of a third-party
  `.pt` (trusted here, but it's a pickle -- keep it vendored, don't auto-update from the
  internet).

---

## 9. Fallback: `Storia-AI/font-classify` (not needed, but noted)

The task said to evaluate the fallback only *if Lens is dead/unbuildable*. **Lens builds
and runs**, so no full eval was done. For the record, from a repo read:

- ResNet50 (timm), **~3,000 Google Fonts** classes, includes Google-Fonts name mappings,
  HuggingFace checkpoint available. **Larger family coverage than Lens.**
- Trained/served on **full images with text** (not single-word crops) -- so it may be the
  better fit for our *line*-oriented crops, and worth a head-to-head **if** licensing
  forces a swap.
- **License: none stated in the repo** -> defaults to all-rights-reserved, so it is
  **not** obviously safer than Lens for commercial use. Would need to confirm with Storia
  before relying on it. (It does *not* automatically solve the license problem.)

If licensing blocks Lens, the real options are: (a) commercial license from mixfont,
(b) confirm/obtain a license for font-classify and benchmark it head-to-head, or
(c) train a small in-house ResNet on Google Fonts renders (the architecture here is
trivial to reproduce; the value is the training data + label->GF mapping).

---

## 10. Recommendation

**ADOPT WITH CAVEATS.**

- **Technically:** adopt. Lens on **single-word crops** is a clear, demonstrated upgrade
  over the current local-font shape-matcher: it fixes the swash (`Gabriola`) and
  monospace (`Cascadia Code`) mismatches, returns Chirp -> `Inter`/`Roboto` like a human
  would, is **Google-Fonts-native (Figma-insertable)**, runs **CPU-only at ~8 ms/crop**,
  and is fully offline.
- **Two non-negotiable caveats:**
  1. **Feed it single words, never full lines.** Line crops score 29% and fail
     *confidently*. Use the per-word boxes already in `ocr.json`, plus a class-agreement
     guard against our existing serif/sans estimate.
  2. **Resolve the non-commercial license before shipping.** As-is, the bundled weights
     are legal only for internal/offline use. Get a commercial license from mixfont, use
     their paid API, or swap to a permissively-licensed equivalent (confirm
     font-classify's license, or train in-house).
- **If the license cannot be cleared:** treat this eval as proof that the *approach*
  (CNN word-classifier -> Google Fonts) is the right direction, and **reject the Lens
  weights specifically** in favor of a license-clean model of the same shape.

**Bottom line:** the method is right and the intel is strong -- wire it in as a
single-word matcher behind a class-agreement guard, but only once the license is sorted.

---

## 11. Reproduce

```bash
# from repo root, using the existing venv (CPU forced inside the script)
.venv\Scripts\python.exe scripts\eval_lens_fonts.py --save-crops
# outputs:
#   work/lens/eval_out/lens_eval_results.json   (full top-5 + latency + accuracy)
#   work/lens/eval_out/lens_eval_table.md       (line-crop table)
#   work/lens/eval_out/crops/*.png              (every crop fed to the model)
```

Clone + model live in `work/lens/` (bundled `model/font_classifier.pt`, 46 MB).
No requirements files were modified; nothing was installed.
