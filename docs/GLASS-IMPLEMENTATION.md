# Glass / frosted-panel reconstruction — implementation status

Closes out the glass mandate. Companion to `GLASS-RESEARCH-DETECTION.md` (algorithm) and
`GLASS-RESEARCH-FIGMA.md` (Figma/schema/QA side). Target case: `HARD-CREATIVES-SPEC.md` **H18**
(two frosted info chips over a lifestyle photo).

## What shipped (applied)

| Piece | File | Status |
|---|---|---|
| α / fill / σ estimator + detection trigger | `src/glass_detect.py` (new) | **Applied** |
| Unit tests (5 §4 recovery cases + trigger table + text-exclusion + conversion + fallback) | `tests/test_glass_detect.py` (new) | **Applied**, 11/11 pass |
| `fill.opacity` + `background-blur` field spec | `src/schema.py` (doc-comment, additive) | **Applied** |
| Emission hook (candidate → fill.opacity + effect) | `src/build_design_json.py` | **Diff only** — file actively edited by another agent (mtime moved every few min during this session) |
| QA backdrop-blur approximation | `src/render_preview.py` | **Diff only** — same reason |

`figma-plugin/code.js` needs **zero** changes (confirmed in `GLASS-RESEARCH-FIGMA.md` §1).

## `src/glass_detect.py` — API

- `detect_glass(orig, bg, region, sigma_grid=…, exclude_boxes=…) -> GlassResult`
  — full detect+classify. `GlassResult.classification` ∈ {`"glass"`,`"solid"`,`"low-confidence"`};
  `.is_glass`, `.fill_color` (always a usable solid RGB), `.fill_opacity`, `.background_blur_radius`
  (**Figma-space**), `.sigma` (PIL-space).
- `estimate_glass(...) -> GlassFit` — raw least-squares fit (α, fill, σ, RSS, ratio, flatness).
- `sigma_to_figma_radius(σ)` / `figma_radius_to_sigma(r)` — the **2.272728** conversion, exposed
  so emission multiplies and the QA renderer divides. Never feed a raw Figma radius into
  `PIL.GaussianBlur` (over-blurs ~2.27×).

Detection gates (bias hard against false positives per FEATURE-PLAN §W2):
`alpha ≥ 0.97` → solid · flat RSS-vs-σ curve (`flatness ≥ 0.90`) → solid · flat single color
(`solid_rss ≈ 0`) → solid · `alpha ∈ [0.85,0.97)` → low-confidence → solid ·
`glass_rss/solid_rss > 0.30` → low-confidence → solid · else **glass**. Text boxes are excluded
from the fit (`exclude_boxes`, dilated 3px).

Contract for the upstream detector: attach `meta.fill_opacity` + `meta.background_blur_radius`
onto a shape candidate when `is_glass`; **omit both** otherwise (absence == solid fallback,
no separate code path).

## Verification

Unit tests: `.venv/Scripts/python.exe -m pytest tests/test_glass_detect.py -q` → **11 passed**.

End-to-end (`scratchpad/glass_e2e.py`, detect → emit → QA-render on H18-style composites):

- H18 chip (α 0.20 / σ 9.0 / white, white text excluded): recovered **α 0.20 (err 0.0000),
  σ 9.0 (err 0.00), figma_radius 20.455, color_err 3.46**. Emitted
  `fill={color:#FDFDFD, opacity:0.2}`, `effects=[{background-blur, radius:20.455}]`.
- QA SSIM over the glass-fill region: **glass-aware render 0.9961** vs **naive no-blur flat-rect
  0.3707** (+0.6254). Confirms the backdrop-blur approximation stops glass from false-failing QA,
  and that the naive path (what render_preview does today) *would* tank it.
- Solid fallback: opaque flat chip → `classification=solid`, plain rect, no effect, color exact.
- Second glass case (α 0.28 / σ 13): recovered α 0.2799, σ 13.0.

## Pending diffs (apply after the concurrent edit to these two files settles)

### `src/build_design_json.py`

In `_compile`, right after `source_effects` is resolved (~line 930):

```python
    # Glass/translucent chips: glass_detect.detect_glass may attach fill_opacity (0..1)
    # and background_blur_radius (FIGMA-space px) onto the candidate's meta. Fold into the
    # fill's own opacity (NOT layer opacity) + an appended background-blur effect. Absence
    # of these == solid fallback (no separate code path).
    glass_fill_opacity = meta.get("fill_opacity", candidate.get("fill_opacity"))
    glass_blur_radius = meta.get("background_blur_radius", candidate.get("background_blur_radius"))
    if glass_blur_radius is not None:
        source_effects = list(source_effects or [])
        source_effects.append({"type": "background-blur",
                               "radius": float(glass_blur_radius), "visible": True})
```

Add a module-level helper (near `_surface_fill`):

```python
def _apply_glass_fill(fill, fill_opacity):
    """Fold a glass fill-opacity into the fill dict (fill-only alpha, not layer opacity)."""
    if fill_opacity is None or not isinstance(fill, dict):
        return fill
    fill = dict(fill)
    fill["opacity"] = float(fill_opacity)
    return fill
```

In the `target == "shape"` branch (~line 1155) and the group branch (~line 1061), wrap the fill:

```python
        fill=_apply_glass_fill(candidate.get("fill"), glass_fill_opacity),
```

`source_effects` already flows into `common["effects"]` at line 966, so the appended effect
survives with no further change.

### `src/render_preview.py`

Use the FIGMA-doc §3.2 diff **with one correction** — convert the Figma radius to a PIL σ before
blurring (the raw diff over-blurs ~2.27×):

```python
def _background_blur_radius(effects):
    best = 0.0
    for effect in effects or []:
        if not isinstance(effect, dict) or effect.get("visible") is False:
            continue
        kind = str(effect.get("type", effect.get("kind", ""))).lower().replace("_", "-")
        if kind != "background-blur":
            continue
        best = max(best, max(0.0, _number(effect.get("radius", effect.get("blur", 8)))))
    return best


def _apply_backdrop_blur(canvas, region, figma_radius):
    """Blur an already-composited canvas region in place (glass backdrop)."""
    from PIL import ImageFilter
    from src.glass_detect import figma_radius_to_sigma   # 2.273 conversion
    x0, y0, x1, y1 = region
    sigma = figma_radius_to_sigma(figma_radius)
    if x1 <= x0 or y1 <= y0 or sigma <= 0:
        return
    patch = canvas.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(sigma))
    canvas.paste(patch, (x0, y0))
```

In `_draw_layer`, immediately before `_blend`:

```python
    blur_radius = _background_blur_radius(layer.get("effects") or [])
    if blur_radius > 0:
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(canvas.width, x + tile.width), min(canvas.height, y + tile.height)
        _apply_backdrop_blur(canvas, (cx0, cy0, cx1, cy1), blur_radius)
```

(`figma_radius_to_sigma` is a pure-python constant divide — safe to import from `glass_detect`.)
