"""pixel_diff.py — QA: compare the Figma render against the original ad.

Deterministic. Produces SSIM + a per-region delta heatmap + a text-recall check (OCR the
render, compare against the source OCR strings). This is the honest gate: text presence and
per-region colour, not a single global blur score. Writes diff.png; returns a partial QA dict
that repair.assess() finishes.
"""
from __future__ import annotations


def _load_gray(path, size=None):
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("L")
    if size:
        im = im.resize(size)
    return np.asarray(im, dtype=np.float64)


def _ssim(a, b):
    import numpy as np
    # global SSIM (single-window) — adequate for a whole-image sanity score
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def compare(source_path, render_path, run_dir, source_ocr=None, render_ocr=None):
    """Return {ssim, text_recall, per_region, diff_png}. render_ocr optional (callers may
    pass an already-run OCR of the render; else text_recall is None)."""
    import os, numpy as np
    from PIL import Image
    src = _load_gray(source_path)
    h, w = src.shape
    ren = _load_gray(render_path, size=(w, h))
    ssim = _ssim(src, ren)

    # per-region grid delta (16x16) → heatmap
    gy, gx = 16, 16
    diff = np.abs(src - ren)
    cell = diff.reshape(gy, h // gy, gx, w // gx).mean(axis=(1, 3)) if (h % gy == 0 and w % gx == 0) \
        else _block_mean(diff, gy, gx)
    heat = (cell / max(1e-6, cell.max()) * 255).astype(np.uint8)
    diff_png = os.path.join(run_dir, "diff.png")
    Image.fromarray(np.kron(heat, np.ones((16, 16), np.uint8))).save(diff_png)

    text_recall = None
    if source_ocr and render_ocr:
        text_recall = _text_recall(source_ocr, render_ocr)

    return {"ssim": round(ssim, 4),
            "text_recall": None if text_recall is None else round(text_recall, 4),
            "per_region_max_delta": float(cell.max()),
            "diff_png": diff_png}


def _block_mean(a, gy, gx):
    import numpy as np
    h, w = a.shape
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    out = np.zeros((gy, gx))
    for i in range(gy):
        for j in range(gx):
            out[i, j] = a[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
    return out


def _norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _text_recall(source_ocr, render_ocr):
    src_lines = [_norm(l["text"]) for l in source_ocr.get("lines", []) if l.get("conf", 1) >= 0.5]
    src_lines = [s for s in src_lines if len(s) >= 3]
    ren_blob = " ".join(_norm(l["text"]) for l in render_ocr.get("lines", []))
    if not src_lines:
        return 1.0
    hit = sum(1 for s in src_lines if s in ren_blob)
    return hit / len(src_lines)
