# Codia Research Synthesis — 2026-07-16

Three parallel researchers ran on 2026-07-16: (1) plugin UX teardown, (2) output-tech capability audit — including a fresh empirical scan of all 4 real Codia teardown files in `runs/` (130 nodes across fixtures 009/041/052/076), (3) ecosystem/pricing/sentiment. This doc is the synthesis; it supersedes nothing but extends `RESEARCH-CODIA-GAP-ANALYSIS.md` with newer, harder evidence.

## Headline findings

1. **Codia's auto-layout / constraints / component claims are marketing, not shipped output.** Across all 130 nodes in our 4 real Codia teardowns: zero `layoutMode`, zero `COMPONENT`/`INSTANCE`, zero masks, zero effects, zero gradient fills, zero non-default opacity/blend modes; constraints are inert Figma defaults. First-party empirical, safe to state as fact in positioning material.
2. **CORRECTION to `RESEARCH-CODIA-GAP-ANALYSIS.md` §6d:** rotated text is NOT untested territory — Codia emits at least axis-aligned rotation as native rotated TEXT (`runs/codia-teardown-3.json` node 15489:130, rotation −90°, fontSize 70, Inter). Arbitrary-angle and text-on-path remain unverified. Do not repeat the old "untested across all tools" claim.
3. **Codia is moving into our niche:** "Magic Layers" launched June 20, 2026 — a flat-image → semantic-layer decomposer explicitly targeting posters, social creatives, and product photos (not UI screenshots). Lives in Codia Studio, method undisclosed, output is an opaque `dsl` string via `/v2/open/image/layering`. Watch this product.
4. **No credible open-source competitor exists.** Closest literal match is a 1-star GitHub stub; abi/screenshot-to-code (71k stars) never outputs Figma layers. The building blocks exist in OSS separately; nobody shipped the glue. We would be the only one.
5. **Pricing/privacy are structural wins for us:** free tier is 5 credits/month at 1440×1440px cap (a top user complaint); heavy ad use realistically costs ~$59/mo Pro. Users call $49/mo "too expensive" in plugin comments; vendor blames GPU cost. Zero organic community trust exists (no Reddit threads, no G2/Capterra, PH "No reviews yet" despite ~319k installs). Local = confidential client IP never uploaded + zero marginal cost + community trust winnable by default.

## What Codia's plugin UX has that we lack (ranked by ad-creative impact)

1. **"Tag as Image" pre-generation region control** — user marks a photographic/illustrated region so the AI treats it as one flat image instead of fragmenting it. Ads are dominated by product photography; this is the single most ad-relevant UX affordance to adopt.
2. **Per-element "Vectorize" post-generation** — scoped re-trace of one icon/button/logo without a full re-run. A scoped-regenerate affordance generally (re-run one element/region) is the pattern to copy.
3. **Text localization workflow** — 108-language OCR, translate/rewrite copy in place while keeping layout + 1:1 text style (a named ad/campaign use case on their site).
4. **Multi-source import breadth** — paste from clipboard, select image node on canvas, PSD/PDF direct, Canva links, live URL. For us: clipboard-paste + select-node-on-canvas are cheap, high-value plugin wins; the rest is low priority.
5. **Side-by-side original-vs-rebuild canvas placement** — we already do this (contract: "screenshot parked beside rebuild") — parity, keep it.
6. **Trust signals** — 177 plugin versions, visible weekly changelog. OSS equivalent: public changelog + release cadence in the repo.

Notable Codia UX facts: multi-step wizard (upload → crop/tag → explicit "AI Design" click → result), ~10s–1min generation, single-image only in the plugin (batch is Enterprise-API-only — a 2-year unresolved user request), no prototyping/interaction output, recurring signup device-limit bug (2026, unresolved).

## What we have that Codia lacks (positioning ammo — all evidence-backed)

- **Integrated per-element vectorization** — their OpenAPI schema marks Vector as PDF-only; their SVG converter is a disconnected whole-image product. We vectorize per-element with render-back gating.
- **True ellipse masks** (avatars/badges) — they emit zero masks, ever; square rects with baked surrounds.
- **Honest editability QA** — native_text_ratio/residue/placement contract metrics; their dominant independent complaint is exactly "looks fine at first glance, falls apart in real use / a designer could rebuild faster."
- **Local/private/free** — vs cloud SaaS + credits.
- **Ad specialization** — Codia's own blog: "Banners and ad creatives... heavy photo overlays and stylized typography may need manual cleanup." Their pipeline treats ads as a generic edge case.

## Open ground (nobody ships it — leapfrog lanes)

1. **Occlusion/overlap decomposition** — Codia's #1 self-admitted weakness ("overlapping elements are the hardest... may need cropping and re-running"); our peel lane. Highest-leverage build.
2. **Native gradient emission for image input** — their schema supports gradients but real output contains zero; ads lean on gradient heroes + drop shadows. Finishing our P2 #9 gradient work wins ad-relevant ground outright.
3. **Real component/instance detection** — their output built the same star-rating 3× from scratch (fixture 076). Both sides missing; lower urgency for single-frame ads.
4. **Responsive/multi-breakpoint reconstruction** — confirmed broken market-wide (Figma forum complaint + their own admission).
5. **In-flow AI-chat refinement after conversion** (Banani's pattern) and **MCP/agent-native export** — relevant to our agentic ad-remix direction.

## Pricing reference (2026)

| Tier | Price | Credits/mo | Caps |
|---|---|---|---|
| Free | $0 | 5 | 1440×1440px, 5MB |
| Starter | $29/mo annual ($49 list) | 200 | 4096×4096, 20MB, batch ≤5 |
| Pro | $59/mo annual ($99 list) | 500 | 10k×10k, 50MB, batch ≤20 |
| Enterprise | custom | custom | API, batch ≤1000/call, SSO |

Credits are universal across their whole product suite (vendor-confirmed). Per-conversion cost unpublished; ~13 credits/call appears in the image_to_design OpenAPI doc, 1 credit ≈ 1 image by NoteSlide analogy (inferred).

## Recommended actions

1. **Plugin:** add Tag-as-Image region control + scoped per-element re-run (feeds the plugin-overhaul work).
2. **Pipeline:** finish native gradient emission (P2 #9) — cheap, provable differentiator on ad content.
3. **Keep peel/occlusion as the strategic leapfrog** (see PEEL-INPAINT-DECISION doc when it lands).
4. **README/positioning:** state the empirical claims — "Codia's real output contains no auto-layout, no components, no masks, no gradients (130-node scan of 4 real conversions)" — plus local/private/free and ad specialization.
5. **Correct** the rotated-text claim in `RESEARCH-CODIA-GAP-ANALYSIS.md` §6d before it's used in any pitch.
6. **Watch Magic Layers** (their entry into our category, June 2026).

Key sources: codia.ai pricing/docs/releases/blog (screenshot-to-figma-guide, codia-vs-image-to-design, introducing-codia-magic-layers, image-text-to-editable-text), Figma Community plugin pages + creator comment replies (plugins 1329812760871373657, 1301565000406306598, 1406878522260473502), forum.figma.com thread 17993, medium.com/@yuxuan-o-o teardown, github abi/screenshot-to-code, local `runs/codia-teardown-{2,3,009}.json` scans, `docs/reference-codia-openapi.json`.
