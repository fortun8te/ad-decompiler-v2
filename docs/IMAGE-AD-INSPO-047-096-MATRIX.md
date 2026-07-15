# IMAGE AD INSPO 047-096 coverage matrix

Visual audit of the actual source images in `/Users/michael/Downloads/IMAGE AD INSPO`.
This is a coverage set, not a claimed quality benchmark: model-quality results still require
an RTX run with the configured OCR, SAM, vector, and inpaint services active.

| ID | Distinct pattern | Current treatment / acceptance contract |
| --- | --- | --- |
| 047 | Packshot with stacked answer captions | Product remains a cutout; caption text needs positive overlay ownership. |
| 048 | Hand-held product with benefit chips | Product/hand is raster; chips are native only when separately verified. |
| 049 | Product in hand with editorial headline | Keep hand/product photographic ownership; headline has a font-fidelity gate. |
| 050 | Social-post card with portrait | Intentional `screenshot` cluster; platform lockup remains a separate artwork exception. |
| 051 | Product-in-hand FAQ caption treatment | Photo remains raster; repeated caption plates require inpainting only when proven overlays. |
| 052 | Before/after product comparison | `comparison_grid`; split only with literal Before+After evidence. |
| 053 | Lifestyle offer with multiple labels | Lifestyle raster plus external labels; avoid extracting incidental scene copy. |
| 054 | Cinematic restaurant photo with editorial type | Photo plate plus overlay text; serif/font mismatch must use exact raster fallback. |
| 055 | Black product reflection ad | Product/reflection remains one raster cluster; no fake separate reflection layer. |
| 056 | Four-column product comparison | Intentional `diagram`/`ui-panel` cluster unless all columns are independently proven. |
| 057 | Sleep lifestyle photo with glassy CTA | Photo raster; translucent UI only native when its boundary/effect is verified. |
| 058 | Clinical product sale sheet | Flat plate, native text/card where OCR/font evidence passes; bottle stays raster. |
| 059 | 3D Prime Day multi-product composition | Intentional `product-cluster`; do not split objects, coupons, confetti, or 3D lettering. |
| 060 | Custom type, scribble arrow, two product renders | Products raster; arrow uses gated vector/raster fallback; custom type may be raster. |
| 061 | Receipt, perforations, tiny products, barcode | Intentional `receipt` cluster; barcode and paper texture stay exact pixels. |
| 062 | Toothpaste before/after close-up | `comparison_grid`; photos remain separate only with strong pair evidence. |
| 063 | Multi-item mini-bundle sales card | Intentional `ui-panel`/product cluster; do not over-segment packshots. |
| 064 | Cosmetic bottle with liquid/reflection | Intentional product/reflection cluster; editorial text independently gated. |
| 065 | UGC drink photo with black caption bubbles | Lifestyle raster; bubbles/text only escape with positive external-overlay evidence. |
| 066 | Clinical eye before/after benefits matrix | Intentional `diagram`/comparison cluster; tiny checkmarks and copy stay raster by default. |
| 067 | Product discontinuation announcement | Flat editorial layout; product strip raster, copy native only if font matches. |
| 068 | Dense X/Twitter thread | Intentional `screenshot` cluster; never rebuild its small chrome line-by-line. |
| 069 | Lime/product word-art composition | Product/lime photography raster; custom display lettering falls back exactly if needed. |
| 070 | Dark fragrance testimonial | Bottle/glow/photo remain raster; quote is overlay only with ownership evidence. |
| 071 | Shower product/arm lifestyle | Product/skin is one photo region; labels must not erase source pixels blindly. |
| 072 | Hand-held fragrance product array | Product-in-hand photo cluster; printed package text stays baked. |
| 073 | Product with flower and reflection | Product/reflection raster cluster; simple gradient plate can be native. |
| 074 | Bathroom product with quote backplate | Lifestyle photo plus proven quote overlay; tile scene remains background. |
| 075 | Rotated gel package and liquid | Intentional product cluster; rotated packaging/gel are not vectors. |
| 076 | Review/testimonial cards with product | Intentional `ui-panel`/diagram cluster until individual cards are proven. |
| 077 | Side-by-side product choice with arrows | Simple arrows use vector gate; product imagery remains raster. |
| 078 | Ingredient explainer with arrows | Intentional `diagram` or gated arrows; hand/product assets remain raster. |
| 079 | Long-run cramps annotated packshot | Product raster; leader arrows use vector gate with raster fallback. |
| 080 | Pile-of-cans UGC creative | Inseparable product cluster/photo; only external caption copy escapes. |
| 081 | Technical warning/product callout sheet | Intentional `diagram`/`nutrition-panel`; rules and tiny annotations stay raster. |
| 082 | Editorial product claim layout | Product raster plus text; brand lockup and custom art remain exact artwork. |
| 083 | Handwritten budget/product concept | Intentional infographic/product cluster; handwriting is not editable unless explicitly supplied. |
| 084 | Pill warning with red strikeout | Product raster and custom strike artwork; avoid treating it as a generic table. |
| 085 | Ingredient/claim chart with product | Intentional `diagram`/`chart` cluster; pill rows should not become guessed controls. |
| 086 | Product shelf photo with caption | Lifestyle raster; black caption can be external only with positive evidence. |
| 087 | Laptop/packshot testimonial photo | Photo/product cluster; display type needs an exact-font or raster decision. |
| 088 | Black Friday packshot with tickets | Product cluster; tickets are vectors only if a render-back gate passes. |
| 089 | Editorial meeting/product sale layout | Flat card + product raster; decorative wordmark stays artwork. |
| 090 | Black Friday bento grid and stickers | Intentional `ui-panel`/`diagram` cluster unless card boundaries are independently verified. |
| 091 | Sharp Focus product feature grid | Intentional `nutrition-panel`/diagram cluster; tiny labels and rules remain source-backed. |
| 092 | Dark hand-held sachet photograph | Lifestyle/product raster; centered text is an overlay only when positively classified. |
| 093 | Blur-backed warning card | Intentional `ui-panel`; blurred source and type treatment remain an atomic crop. |
| 094 | Caffeine-free claim with small graph | Intentional `chart`/diagram cluster; product is a separate raster only with a reliable matte. |
| 095 | Cadence dark campaign collage | Inseparable product/person cluster; overlapping photo treatment stays source-backed. |
| 096 | White-on-white product bundle and price map | Intentional `product-cluster`/diagram; shadows, pack overlaps, and strikes remain exact pixels. |

## What the matrix changes

When detector/VLM evidence assigns `screenshot`, `ui-panel`, `receipt`, `chart`, `graph`,
`table`, `nutrition-panel`, `diagram`, `infographic`, or `product-cluster`, the pipeline now
exports one named swappable raster crop using its full rectangular/rounded-rect source region.
Contained OCR stays baked by default. Only a positive `recreate`/`overlay_copy`/explicit
promotion can create editable overlay text, and that text is kept in the same Figma asset group.
