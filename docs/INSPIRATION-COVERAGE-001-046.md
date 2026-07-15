# Inspiration coverage audit: 001–046

Scope: visual inspection of every source file whose name begins `001` through `046` in
`/Users/michael/Downloads/IMAGE AD INSPO`. This is a static capability audit of the
current source tree, not a claim that an RTX acceptance run has passed any of these ads.

Status keys:

- **Native when evidenced** — the compiler can make the layer editable, provided the
  detector/ownership pass identifies it correctly.
- **Safe raster** — source pixels should be retained as a clean plate or swappable crop;
  it is deliberately not claimed to be fully editable.
- **Gap** — the current route has no matching semantic detector or representation.

| Construction pattern observed | Source IDs | Current route and evidence | Status |
| --- | --- | --- | --- |
| Plain/gradient offer plate, product packshot, price/CTA shell | 001–006, 008, 012–013, 016, 019, 023–024, 030–031, 034, 036–037, 039, 043 | SAM prompts products/packages/buttons/cards (`src/sam3_detect.py:37-74`); routing sends products to masked images and buttons/cards to shapes (`src/routing.py:216-262`); primitive style extraction recovers fill, radius, stroke, gradient and shadow (`src/reconstruct.py:977-1009`). | Native when evidenced; physical product rendering stays a swappable raster. |
| Full-bleed lifestyle/UGC photo with type over it | 007, 014, 021–022, 027–029, 035, 040–042, 044–046 | Lifestyle preset retains the scene photograph and supports overlay copy (`src/archetype.py:50-57`). The text ownership pass is deliberately conservative about text on a photo/product/card (`src/vlm_scene_text.py:29-41`). | Photo is safe raster; proven marketing overlay text can be native. |
| Product microcopy, label artwork and wordmarks | 001–008, 012–013, 016, 019, 023–024, 030–031, 034, 036–037, 039, 043 | Printed product copy is kept in its raster owner (`src/routing.py:150-180`); wordmarks default to an exact masked raster rather than a guessed font (`src/routing.py:98-100`, `154-180`). | Safe raster by design. |
| Styled marketing type, outlined/gradient text, mixed text lines | 002–008, 012–014, 016–020, 023–046 | Text compiler supports native fills, strokes and ranges (`figma-plugin/code.js:1064-1073`, `707-753`). Per-line mixed styles are retained as text runs (`src/merge_layers.py`). Low-confidence type is rasterized rather than guessed (`src/routing.py:103-134`). | Native when font/ownership evidence is good; safe raster otherwise. |
| Rounded cards, pills, chips, circles, masked photo insets | 003–004, 009–012, 015–016, 019, 024, 026, 030, 032, 034–037, 041, 046 | Figma importer supports rounded/ellipse/path/alpha masks (`figma-plugin/code.js:1219-1323`); reconstruction infers mask geometry and may recover a proven inside frame stroke (`src/reconstruct.py:1012-1030`, `1091-1127`). | Native when evidenced. |
| Social, browser, Notes, phone and video screenshot chrome | 009–011, 015, 017, 032, 035, 038, 042 | Only `social_screenshot` is a dedicated UI-like archetype; the available archetype set has no browser/Notes/phone/video parser (`src/archetype.py:13-16`). UI/card/screenshot text is intentionally eligible for `raster_keep` (`src/vlm_scene_text.py:37-41`). | Safe raster card; **gap** for native UI components. |
| Before/after and two-column comparison layouts | 025, 026, 033 | Comparison preset preserves columns and aligned rows (`src/archetype.py:40-48`). The actual independent image split requires literal `before` **and** `after` evidence and emits exactly two halves (`src/reconstruct.py:601-670`). | 025/033: native two-column treatment when OCR sees both labels. 026: visually safe but **gap** for independent columns because `vs` alone does not authorize the split. |
| Multi-panel / horizontal band storytelling | 018 | No archetype or panel splitter exists beyond the verified two-way before/after path (`src/archetype.py:13-16`, `src/reconstruct.py:601-670`). | Safe raster; **gap** for editable band/panel structure. |
| Curved arrows, straight/orthogonal leaders, endpoint dots | 014, 016, 041, 044 | Arrows and leaders route through gated vectorization with alpha-raster fallback (`src/routing.py:21-28`). The residual detector only labels shape/icon/photo-fragment (`src/element_detect.py:326-331`), so a dedicated leader proposal is required. `callout leader line` is now in default SAM prompts and example config (`src/sam3_detect.py:57-63`, `config.example.yaml`). | Improved; still needs an actual corpus run to prove SAM masks the thin lines. |
| Rules/dividers | 018, 026, 041, 044 | A confirmed divider becomes a native thin rectangle (`src/routing.py:224-230`), but no deterministic divider detector currently creates that role. | **Gap** for guaranteed native rules; vector/raster fallback remains safe. |
| Jagged price bursts, distressed underline, sticker-like ornaments | 008, 013, 016, 028, 037, 039 | Importer can create a STAR if supplied (`figma-plugin/code.js:1080-1105`), but the detector has no `starburst` semantic prompt and the schema pipeline does not infer star geometry. | Safe vector/raster fallback; **gap** for native editable starbursts/brush textures. |
| Collage, paper stack, perspective screen/packaging, 3-D compositing | 020, 021, 023, 031, 042, 046 | Compiler applies ordinary rotation but no perspective/perspective-warp transform (`figma-plugin/code.js:459-478`). | Safe raster cluster; **gap** for independently editable perspective pieces. |
| Reflections, glass/soft shadows, product-on-stage lighting | 003–004, 006, 019, 024, 030–031, 034, 036, 039, 043 | Simple primitive shadows can be recovered only on a flat surrounding field (`src/reconstruct.py:915-974`); product photo lighting is intentionally preserved in the raster asset. | Safe raster for product lighting; native only for proven simple shadow. |

## Visual index

| ID | Primary construction | ID | Primary construction | ID | Primary construction |
| --- | --- | --- | --- | --- | --- |
| 001 | isolated label/pack | 002 | packshot offer card | 003 | product, price chip, review card |
| 004 | product/package + CTA | 005 | chocolate-pour packshot | 006 | product + long copy |
| 007 | angled product-photo crop | 008 | dark offer, burst, repeated type | 009 | X post screenshot |
| 010 | Notes-style editorial card | 011 | Notes app/product placement | 012 | product + benefits list |
| 013 | gummies offer, burst, CTA | 014 | lifestyle hero + curved arrows | 015 | mobile browser/annotation screen |
| 016 | product infographic + arrows | 017 | long white-letter layout | 018 | three-band dark product ad |
| 019 | dark product hero + award | 020 | magazine/paper collage | 021 | laptop/UGC photo |
| 022 | ear photo + circular callouts | 023 | apology letter + rotated packaging | 024 | call-screen style product offer |
| 025 | before/after portraits | 026 | Silk-vs-Satin comparison | 027 | photo + translucent CTA |
| 028 | lifestyle sale + textured underline | 029 | fashion photo + CTA pill | 030 | two-product offer + cloud shapes |
| 031 | stylized pillow product scene | 032 | phone/app mockup cards | 033 | before/after skin panels |
| 034 | product + review/CTA | 035 | UGC question-card screenshot | 036 | clinical product/dropper ad |
| 037 | bundle + neon coupon/starburst | 038 | long founder letter | 039 | product photo + sale burst |
| 040 | testimonial over water photo | 041 | lifestyle photo + leader callouts | 042 | X/video screenshot |
| 043 | product in hand + icon claims | 044 | product + orthogonal leaders | 045 | lifestyle CTA with inset portrait |
| 046 | product-use photo + pills/insets |  |  |  |  |

## Concrete next benchmark

Run a real RTX/Figma acceptance corpus on the representative hard set:
`009, 015, 018, 020, 025, 026, 033, 041, 044, 046`.
Inspect each resulting `design.json`, Figma export and QA result. Do not call the tool
better than a competitor until this set has actual rendered evidence.
