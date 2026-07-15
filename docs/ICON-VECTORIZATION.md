# Icon and logo vectorization

Small icons are traced only when their render-back gate passes. The source crop remains a transparent raster asset either way, so a rejected trace is exported as an alpha-preserving image instead of a flattened opaque rectangle.

- VTracer is preferred for multi-colour marks.
- Potrace is tried first only for a true one-colour silhouette.
- OpenCV contours are the final single-colour fallback; they use even-odd fill so counters such as a camera lens, ring logo, or letter hole stay transparent.

The gate rasterizes the generated SVG at the original crop size and checks alpha/colour fidelity, path count, and enclosed transparent-hole recall. A trace with a filled counter falls back to the original alpha raster. Tiny crops are enlarged for tracing but their SVG coordinates are restored to the original bounds before export.

Useful output lives in `reconstruction.json` under each icon's `meta.vectorize` (`ok`, engine, score, and note). `vector_fallback: true` means the alpha raster was retained deliberately.

Thin dividers and rules bypass tracing and remain native bars. Arrows use the same render-back
gate as icons: a clean trace becomes an editable SVG/vector; a complex, textured, or low-scoring
arrow remains an exact transparent raster instead of a malformed path.
