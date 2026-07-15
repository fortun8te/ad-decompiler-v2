# Ad inspiration coverage: 097–139

Visual audit of the 31 files that actually exist in
`/Users/michael/Downloads/IMAGE AD INSPO` for IDs 097–139. IDs 109–110 and
113–121 are not present. This is a code-and-image audit, not an RTX run.

## What is already native when the scene is identified

The current compiler creates editable text, shapes, strokes, gradients, rounded
corners, drop/inner shadows, layer/background blur, masks, auto-layout, and
exact repeated components. Photos, products, people, phone maps, and any
unreliable graphic remain swappable raster layers by design.

| Visual family | IDs | Current result | Real boundary |
| --- | --- | --- | --- |
| Clean product/editorial layouts | 097, 099, 100, 105, 128, 130, 133, 136, 139 | Text, dividers, cards, badges, CTA outlines and product placement can be native. | Product packs and realistic reflections remain raster. |
| Lifestyle/photo with marketing overlay | 098, 102, 106, 108, 122, 124, 125, 127, 129, 131, 134, 138 | Source photo stays raster; separately identified copy, pills and CTAs can be native. | Separating the copy/card from the photo is an image-understanding decision. |
| Social/testimonial shell | 103, 123, 126 | Safe result is an editable outer layout with the social card/image retained as raster. | Rebuilding X/social chrome and avatars as native, meaningful components requires a screenshot-template model. |
| Phone/product-app screenshot | 104 | Phone/map stays a swappable raster. | Map, device chrome and in-app metrics are not reliably recoverable as real UI from pixels today. |
| Charts, glass and visual effects | 107, 129, 131, 138, 139 | The compiler supports gradients, shadows and background blur **if supplied in design.json**. | The pipeline does not deterministically infer charts, glass blur, soft field gradients or reflected product lighting; those are model-only recognition cases. |
| Repeated UI patterns | 100, 101, 104, 112, 130, 132, 134, 135 | Exact repeated cards can become Figma components; aligned text/card stacks can use auto-layout. | Rows with different copy (stats, messages, comparison columns) remain individual editable frames. Converting them into reusable components with correct per-instance text overrides is not implemented. |
| Text-first / typography posters | 099, 100, 105, 107, 128, 130, 131, 136 | OCR text is the correct editable path, and the compiler preserves fills, strokes and text gradients. | Exact font family, signature handwriting, distorted/outlined type and text baked into packaging need a font/style model or stay raster. |
| Asset rather than full ad | 111, 112, 137 | 111/112 are UI references; 137 is a tiny avatar crop. They can be retained as raster assets. | None is a sensible full-ad benchmark by itself. |

## Exact pattern notes

| IDs | Distinct pattern | Coverage call |
| --- | --- | --- |
| 097 | Multi-product bundle, crossed-out price, code line | Native copy and price treatment; product stack is model-led cutout placement. |
| 098 | Dark lifestyle hero with benefit chips | Native chips/text only after overlay segmentation; lifestyle photo remains raster. |
| 099–100 | Editorial feature list / large metric stack | Strong native target; variable metric rows should be editable frames, not cloned components. |
| 101 | Two-column comparison | Native columns/checks possible; column semantics must come from the vision model. |
| 102 | Dark product hero with reflective floor and CTA | Copy/CTA native; product, highlight and reflection are model/raster territory. |
| 103, 123, 126 | X-style testimonial | Preserve source card as raster unless a social-card template model is explicitly added. |
| 104 | Two phone/map panels | Treat the phone screens as raster; only surrounding explanatory copy is a safe native target. |
| 105 | Founder letter with signature | Text body native; signature is not trustworthy OCR text. |
| 106 | Hand-held product and sticky note | Photo is raster; sticky note can be native only if separated confidently. |
| 107 | Line chart with gradient area | Native primitives can render it, but extracting chart geometry/data is model-only. |
| 108 | Quote with highlighted phrases | Text and highlight plates are native if identified; product/reflection stays raster. |
| 111–112 | Product UI reference / Layers inspector | Screenshot parsing and reusable UI components are model-only. |
| 122, 124, 125 | UGC image with stacked black/white text pills | High-value native overlay target; requires robust pill/copy segmentation. |
| 127 | Large overlapping product packs and urgent copy | Copy native; pack occlusion/perspective remains raster. |
| 128 | Single-product offer poster | Strong native typography + asset placement target. |
| 129–130 | Product-plus-stat cards | Native cards/text when found; glass blur in 129 is not currently inferred. |
| 131 | Full-bleed sale promotion with very large type | Native text/gradient if recognized; person and product remain raster. |
| 132, 135 | Chat/message screenshots | Safe raster source; semantic chat reconstruction needs a dedicated template model. |
| 133 | Product comparison pair | Native headline/layout; product images raster. |
| 134 | Sale badge and repeated ticker strip | Native shapes/type possible; ticker/repetition semantics are model-only. |
| 136 | Repeated background headline behind product | Compiler can render repeated text; detector has no pattern-replicator. |
| 137 | Avatar crop | Asset only, not a layout benchmark. |
| 138–139 | Soft radial/field-gradient product posters | Native gradient support exists; deciding it is a gradient rather than a baked photo field is model-only. |

## Practical result

The blocker for these ads is not Figma paint support. It is deciding which pixels
are editable structure and which must remain an honest raster layer. The next
useful benchmark is a real RTX run on: 099 (clean metric layout), 122 (UGC
pills), 129 (glass stats), 107 (chart), and 132 (chat UI). That five-image set
spans the actual unsupported recognition cases without pretending a compiler
change solves them.
