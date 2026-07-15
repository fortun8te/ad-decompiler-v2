# RTX PC final-check prompt

Paste this into Codex on the Windows RTX computer:

> Work in the existing `ad-decompiler-v2` checkout. Preserve local work: never reset, stash, or
> overwrite a dirty checkout. First run `git status --short`, then run
> `.\scripts\windows_sync.ps1`; if it reports DIRTY, stop the sync and report the changed files.
> Install automatic safe sync once with `.\scripts\install_windows_sync_task.ps1`.
>
> Run `.\setup_rtx.ps1 -DeepDoctor`, then `Start Bridge.bat -SelfTest`. Do not continue if CUDA,
> OCR, SAM 3, Gemma, the selected inpaint backend, vectorization, or Figma staging lacks real
> runtime proof. Keep Figma Desktop open with the Ad Decompiler development plugin loaded.
>
> Run the four-image smoke set:
> `.\start_rtx.ps1 -InputDir "C:\images\IMAGE AD INSPO" -Ids 002,010,017,020 -Output runs\codia-smoke -RequireFigma -FigmaWaitS 180`.
> Click **Import into Figma** once for each exact ad shown by the plugin.
>
> If all four pass, run the locked 26-image hard set:
> `.\start_rtx.ps1 -InputDir "C:\images\IMAGE AD INSPO" -Ids 009,015,018,020,025,026,033,041,044,046,056,059,060,061,065,068,075,081,090,096,099,107,122,129,132,138 -Output runs\codia-hard-set -RequireFigma -FigmaWaitS 180`.
> These are intentionally difficult: screenshot UI, multi-panel/comparison layouts, perspective
> collage, arrows/leaders, dense product grids, receipts/barcodes, caption pills, overlapping 3D
> product scenes, charts, glass/blur, soft gradients, and difficult typography.
>
> If the locked set passes, run the entire reference folder with no `-Ids` filter:
> `.\start_rtx.ps1 -InputDir "C:\images\IMAGE AD INSPO" -Output runs\image-ad-inspo-all`.
> This folder currently contains 128 real ad images. Confirm the benchmark manifest also says 128;
> hidden files such as `.DS_Store` do not count. Do not skip, reorder, or silently retry failed ads.
> Keep the generated per-ad previews, design JSON, routing/ownership reports, and QA rows. Then open
> every failed or degraded ad in Figma through the plugin and record the exact failure category.
>
> Do not call it Codia-level or push anything unless every row has runtime accepted, QA passed,
> no hard failures, a fresh matching `figma_export.png` and `figma_report.json`, no missing assets,
> and no unexplained font substitution. Inspect the real Figma layers for editable text, correct
> order/groups, clean background, masks, rounded image strokes, and usable vectors. Record manual
> cleanup minutes for each ad. Review `native_leaf_ratio`, `leaf_accounting`, intentional raster
> cluster count, and every raster fallback; zero unexplained raster fallbacks are allowed. A wrapper
> frame around one image is not an editable result. Fix only measured failures, add a regression test for every code
> fix, rerun the full Python suite and plugin compiler smoke, then show the complete evidence.

The four-image run is a wiring check. The locked 26-image run is the minimum real-Figma release
gate. The full 128-image run is the coverage gate: it must expose every failure, including cases
that use a faithful named raster fallback because the source pixels cannot safely become native
Figma vectors or text.
