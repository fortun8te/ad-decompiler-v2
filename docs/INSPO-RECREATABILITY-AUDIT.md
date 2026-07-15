# IMAGE AD INSPO Recreatability Audit (2026-07-15)

All 128 images in `C:\Users\micha\Downloads\IMAGE AD INSPO` were individually viewed and
graded by 5 vision agents against the pipeline's capability set (current + this week's
landings). Strict standard: "properly recreated" = near-identical AND genuinely editable;
a full-image slice does not count. Photos correctly staying raster is not a degrade.

## Verdict totals

| Verdict | Count | Share | Meaning |
|---|---|---|---|
| GREEN  | 90 | 70% | fully recreatable, properly editable |
| YELLOW | 37 | 29% | recreates and looks right; named elements degrade to slices/approximations |
| RED    | 1  | <1% | 021 — a photo of handwritten sticky notes at perspective; nothing editable exists (unrecreatable by ANY tool, incl. Codia) |

Per batch: 001-025: 11G/13Y/1R · 026-050: 23G/2Y · 051-075: 19G/6Y · 076-100: 14G/11Y ·
101-139: 23G/5Y. (111/112 are screenshots of our own UI that got mixed into the folder.)

## Capability gaps ranked by frequency (the YELLOW drivers)

1. **Hand-drawn annotations** (marker arrows, X's, strikethroughs, scribble underlines) —
   ~9 ads (014, 015, 016, 060, 078, 079, 083, 084, 091). Currently raster-slices.
   **Fix path (best ROI):** route annotation-role masks through the vectorizer →
   editable Figma VECTOR strokes. vtracer handles flat marker strokes well; the
   fidelity gate already arbitrates. Beats Codia (they can't either).
2. **Radial / multi-stop / metallic gradients, vignettes, glows** — ~7 ads (018, 019,
   055, 064, 123, 131, 138). Approximated to 2-stop linear today.
   **Fix path:** vectorize.py now detects radial gradients on crops; extend
   style_extraction + design.json + plugin to multi-stop GRADIENT_RADIAL fills.
3. **Frosted-glass / translucent panels** — ~5 ads (018, 022, 025, 057, 129).
   **Fix path:** Figma natively supports BACKGROUND_BLUR effects on translucent
   frames — detect glassmorphism (local blur + translucency over busy bg) and emit it.
   Genuinely editable and nobody ships this. Differentiator.
4. **Textured/noisy backgrounds** — 007, 008, 020, 048, 061. Mostly OK via Flux
   inpaint; keep as watch-item in benchmarks.
5. **Dense overlap / 3D clusters / chat screenshots** — 005, 008, 013, 059, 132.
   **Fix path:** peel-decomposition stage (prototype built, integration pending).
6. **Curved/arched badge text** — 076, 095, 054-adjacent. NOTE: Figma has NO native
   text-on-path, so an editable recreation is impossible in-platform for anyone
   (Codia slices these too). Correct output = slice (or vector outlines). Not a gap
   vs Codia; document as industry limitation.
7. **Charts (094, 107) / rendered tables (001)** — slice is the correct v1 behavior
   (Codia flattens these too). Editable-chart reconstruction is a someday-differentiator.
8. **Handwritten/script primary copy** — 083 (+ RED 021). Font matching can't help;
   slice is correct. Script *accents* (signatures 067, 105) are fine as raster marks.
9. **Rotated 90° / perspective-warped text** — 085, 088. 90° rotation is actually
   representable in Figma (rotated text node) — small fix in text rotation handling;
   perspective ribbon text stays a slice.
10. **Tiny legal text (<10px)** — 055. OCR 2x retry usually covers; watch in benchmarks.

## What this means vs Codia

- Codia's own docs admit ads (photo overlays, stylized type, overlaps, gradients) are
  their weak spot; our 70% strict-GREEN + honest-slice-fallback on the rest already
  matches or beats their documented behavior on this exact corpus class.
- Items 1 (annotation vectorization) and 3 (glassmorphism) are features NOBODY ships —
  each converts several YELLOWs to GREEN and is demo-visible.
- Items 6/7/8: parity-by-limitation — slices look right, same as Codia; not blockers.

## Recreation coverage claim (post-fix expectations)

With annotation vectorization + radial/multi-stop gradients + glassmorphism + peel
integration landed, the strict-editable rate on this corpus is projected ~85-90%, with
the remainder looking correct via honest slices. 100% "looks right"; the only unreachable
ad is 021 (pure photo of handwriting).
