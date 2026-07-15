"""figma_verify.py — first-class "verified in real Figma" QA verdict.

Pipeline QA (run_pipeline diff/qa) scores whichever render exists — during
development that is usually the local ``preview.png`` simulation from
render_preview.py.  That proves our simulation matches the source; it proves
nothing about what Figma actually displays after the plugin import.  The
companion plugin already POSTs a PNG of the imported frame back through the
bridge (``POST /export`` → written to the staging manifest's ``export_to``,
which figma_import._stage_for_plugin points at ``run_dir/figma_export.png``).
This module turns that export into a first-class verdict with two independent
comparisons:

``fidelity``       exported-vs-original — what Figma really displays vs the
                   source ad (the score that matters for shipping).
``preview_drift``  exported-vs-preview — does our Python-side simulation match
                   what Figma renders?  When this fires, green preview
                   dashboards are lying about Figma, and the region breakdown
                   plus ``figma_verify/drift_heatmap.png`` show WHERE the
                   plugin renders differently (fonts, gradients, effects...).

Verdicts (written atomically to ``run_dir/figma_qa.json``):

``verified``      fresh export exists, fidelity passes, drift immaterial.
``degraded``      fidelity passes but the evidence is weakened: material
                  preview drift, missing preview (no drift evidence), unknown
                  text recall under ``require_text_evidence``, anomalous
                  export scale, or a stale export scored via ``allow_stale``.
``failed``        the export does not match the original above threshold (or
                  text visibly went missing in the real Figma render, or the
                  scoring itself crashed — fail closed, never open).
``not-exported``  no usable export PNG: nothing was verified in real Figma.
                  A stale export (older than design.json) is treated the same
                  way run_pipeline treats it — it is not evidence about the
                  current design (status ``stale-export``).

Honest-evidence rules mirrored from run_report/pixel_diff: every check records
value/threshold/status, loss of evidence is reported instead of silently
passing, and a pretty screenshot can never override a hard check.

This module is read-only with respect to pipeline artifacts.  Its outputs are
``run_dir/figma_qa.json`` and the ``run_dir/figma_verify/`` evidence folder.
pixel_diff is consumed defensively (getattr/.get with a minimal internal
fallback judge) because it is being extended concurrently.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src import pixel_diff
from src.qa_config import visual_pass_ssim

FIGMA_QA_SCHEMA_VERSION = 1
EXPORT_BASENAME = "figma_export.png"
REPORT_NAME = "figma_qa.json"
EVIDENCE_DIR = "figma_verify"

VERDICT_VERIFIED = "verified"
VERDICT_DEGRADED = "degraded"
VERDICT_FAILED = "failed"
VERDICT_NOT_EXPORTED = "not-exported"

DEFAULT_VERIFY_THRESHOLDS = {
    # None → resolved from qa_config.visual_pass_ssim(cfg) so the "verified in
    # real Figma" bar can never silently drift below the pipeline's visual bar.
    "fidelity_ssim_min": None,
    "text_recall_min": 0.80,
    # exported-vs-preview: below this the simulation materially disagrees with
    # what Figma displays and preview-based QA scores stop being trustworthy.
    "drift_ssim_min": 0.95,
    "drift_grid": 12,
    "drift_top_n": 8,
    "drift_region_ssim_min": 0.85,
    # Coarse global alignment (exports can be off by a few device pixels).
    "max_align_shift_px": 12,
    # |scale - round(scale)| tolerance when detecting 1x/2x/3x exports.
    "scale_tolerance": 0.02,
    # Production sign-off should set this true: without OCR of the export we
    # cannot prove text survived the Figma import, so "verified" is capped.
    "require_text_evidence": False,
}


# --------------------------------------------------------------------------- result


@dataclass
class FigmaVerifyResult:
    """Machine-readable verdict; serialized to run_dir/figma_qa.json."""

    run_dir: str
    status: str = "not-exported"      # scored | not-exported | stale-export | error
    verdict: str = VERDICT_NOT_EXPORTED
    export: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    preview_drift: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    report_path: str = ""
    generated_at: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_VERIFIED

    def to_dict(self) -> dict:
        payload = {"schema_version": FIGMA_QA_SCHEMA_VERSION}
        payload.update(asdict(self))
        return payload


# --------------------------------------------------------------------------- io helpers


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: str, value: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_default_cfg(config_path: Optional[str] = None) -> dict:
    """Light copy of run_pipeline.load_cfg (that module must not be imported here)."""
    path = config_path or os.path.join(_repo_root(), "config.yaml")
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle) or {}
        except Exception:
            return {}


def resolve_thresholds(cfg: Optional[dict]) -> dict:
    merged = dict(DEFAULT_VERIFY_THRESHOLDS)
    overrides = (cfg or {}).get("figma_verify")
    if isinstance(overrides, dict):
        for key in merged:
            if overrides.get(key) is not None:
                merged[key] = overrides[key]
    if merged.get("fidelity_ssim_min") is None:
        merged["fidelity_ssim_min"] = float(visual_pass_ssim(cfg))
    return merged


def find_export(run_dir: str) -> Optional[str]:
    """Locate the plugin-exported PNG.

    The bridge's POST /export handler writes to the staging manifest's
    ``export_to``, which figma_import sets to ``<run_dir>/figma_export.png``.
    A glob fallback tolerates future suffixed variants (e.g. retries)."""
    exact = os.path.join(run_dir, EXPORT_BASENAME)
    if os.path.exists(exact) and os.path.getsize(exact) > 0:
        return exact
    candidates = [p for p in glob.glob(os.path.join(run_dir, "figma_export*.png"))
                  if os.path.getsize(p) > 0]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _export_is_fresh(export_path: str, run_dir: str) -> Optional[bool]:
    """Mirror run_pipeline._artifact_at_least_as_fresh(figma_export, design.json)."""
    design_path = os.path.join(run_dir, "design.json")
    if not os.path.exists(design_path):
        return None
    try:
        return os.path.getmtime(export_path) >= os.path.getmtime(design_path)
    except OSError:
        return None


def _canvas_size(run_dir: str) -> Optional[tuple]:
    design = _load_json(os.path.join(run_dir, "design.json")) or {}
    canvas = design.get("canvas") or {}
    try:
        w, h = int(canvas.get("w", 0)), int(canvas.get("h", 0))
        if w > 0 and h > 0:
            return (w, h)
    except (TypeError, ValueError):
        pass
    for name in ("normalized.png", "original.png"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            try:
                from PIL import Image

                with Image.open(path) as im:
                    return im.size
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- metrics


def _ssim_pair(a, b) -> float:
    """Whole-array SSIM; prefers pixel_diff's implementation when present."""
    fn = getattr(pixel_diff, "_ssim", None)
    if callable(fn):
        try:
            return float(fn(a, b))
        except Exception:
            pass
    mu_a, mu_b = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / den) if den else 1.0


def _gray(image):
    import numpy as np

    return np.asarray(image.convert("L"), dtype=np.float64)


def _local_text_recall(source_ocr, render_ocr):
    """Fallback copy of pixel_diff's recall semantics (conf≥.5, ≥3 chars, contains)."""

    def norm(s):
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    src = [norm(l.get("text", "")) for l in (source_ocr or {}).get("lines", [])
           if l.get("conf", 1) >= 0.5]
    src = [s for s in src if len(s) >= 3]
    blob = " ".join(norm(l.get("text", "")) for l in (render_ocr or {}).get("lines", []))
    if not src:
        return 1.0
    return sum(1 for s in src if s in blob) / len(src)


def _text_recall(source_ocr, render_ocr) -> Optional[float]:
    if not source_ocr or not render_ocr:
        return None
    fn = getattr(pixel_diff, "_text_recall", None)
    if callable(fn):
        try:
            return float(fn(source_ocr, render_ocr))
        except Exception:
            pass
    return float(_local_text_recall(source_ocr, render_ocr))


def _fnum(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_metrics(source_path: str, render_path: str, out_dir: str,
                      error: Optional[str] = None) -> dict:
    """Minimal independent judge so a pixel_diff crash cannot hide the export."""
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    with Image.open(source_path) as src_im:
        source = src_im.convert("L")
        with Image.open(render_path) as ren_im:
            render = ren_im.convert("L").resize(source.size, Image.Resampling.LANCZOS)
        a = np.asarray(source, dtype=np.float64)
        b = np.asarray(render, dtype=np.float64)
    ssim = max(0.0, min(1.0, _ssim_pair(a, b)))
    mae = float(np.abs(a - b).mean())
    diff_png = os.path.join(out_dir, "diff.png")
    Image.fromarray(np.clip(np.abs(a - b) * 3, 0, 255).astype(np.uint8)).save(diff_png)
    metrics = {
        "ssim": round(ssim, 4),
        "global_ssim": round(ssim, 4),
        "visual_score": round(ssim, 4),
        "rgb_mae": round(mae, 4),
        "edge_f1": None,
        "color_similarity": None,
        "delta_e_mean": None,
        "text_recall": None,
        "diff_png": diff_png,
        "engine": "figma_verify-fallback",
    }
    if error:
        metrics["engine_error"] = error
    return metrics


def _score_pair(source_path: str, render_path: str, out_dir: str,
                source_ocr: Optional[dict] = None, render_ocr: Optional[dict] = None,
                removal_mask: Optional[str] = None) -> dict:
    """Score one image pair via pixel_diff.compare, consumed defensively.

    ``out_dir`` is a scratch evidence dir inside run_dir/figma_verify/ so
    compare's own artifacts (diff.png) never clobber the pipeline's diff.png.
    """
    os.makedirs(out_dir, exist_ok=True)
    compare = getattr(pixel_diff, "compare", None)
    raw, error = None, None
    if callable(compare):
        try:
            raw = compare(source_path, render_path, out_dir,
                          source_ocr=source_ocr, render_ocr=render_ocr,
                          removal_mask=removal_mask)
        except TypeError:
            # Signature is being extended concurrently — retry the stable
            # backward-compatible positional core.
            try:
                raw = compare(source_path, render_path, out_dir, source_ocr, render_ocr)
            except Exception as exc:  # noqa: BLE001 — never let QA scoring crash the verdict
                error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return _fallback_metrics(source_path, render_path, out_dir, error=error)
    ssim = _fnum(raw.get("multiscale_ssim", raw.get("ssim")))
    metrics = {
        "ssim": ssim,
        "global_ssim": _fnum(raw.get("global_ssim")),
        "edge_f1": _fnum(raw.get("edge_f1")),
        "color_similarity": _fnum(raw.get("color_similarity")),
        "visual_score": _fnum(raw.get("visual_score")),
        "rgb_mae": _fnum(raw.get("rgb_mae")),
        "delta_e_mean": _fnum(raw.get("delta_e_mean")),
        "text_recall": _fnum(raw.get("text_recall")),
        "diff_png": raw.get("diff_png"),
        "engine": "pixel_diff.compare",
    }
    if metrics["text_recall"] is None and source_ocr and render_ocr:
        metrics["text_recall"] = _fnum(_text_recall(source_ocr, render_ocr))
    if metrics["ssim"] is None:
        return _fallback_metrics(source_path, render_path, out_dir,
                                 error="pixel_diff.compare returned no ssim")
    return metrics


# --------------------------------------------------------------------------- scale + alignment


def _detect_scale(export_size, canvas, tolerance: float) -> dict:
    ew, eh = export_size
    cw, ch = canvas
    sx = ew / float(cw) if cw else 0.0
    sy = eh / float(ch) if ch else 0.0
    uniform = abs(sx - sy) <= max(tolerance, 0.02) * max(sx, sy, 1e-6)
    detected = None
    mean_scale = (sx + sy) / 2.0
    nearest = round(mean_scale)
    if uniform and nearest >= 1 and abs(mean_scale - nearest) <= tolerance * nearest:
        detected = int(nearest)
    return {"x": round(sx, 4), "y": round(sy, 4), "uniform": bool(uniform),
            "detected": detected}


def _estimate_shift(ref_gray, mov_gray, max_shift: int) -> tuple:
    """Coarse global (dx, dy) such that mov ≈ ref translated by (dx, dy).

    Phase correlation on (optionally downsampled) grayscale; returns (0, 0) on
    any numerical trouble.  Full-image content dominates, so a locally shifted
    block does not fool the estimate.
    """
    import numpy as np

    try:
        factor = max(1, int(np.ceil(max(ref_gray.shape) / 512.0)))
        a = ref_gray[::factor, ::factor]
        b = mov_gray[::factor, ::factor]
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h < 8 or w < 8:
            return (0, 0)
        a, b = a[:h, :w], b[:h, :w]
        fa, fb = np.fft.fft2(a), np.fft.fft2(b)
        spectrum = fa * np.conj(fb)
        spectrum /= np.maximum(np.abs(spectrum), 1e-9)
        corr = np.fft.ifft2(spectrum).real
        py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
        if py > h // 2:
            py -= h
        if px > w // 2:
            px -= w
        # peak (py, px) recovers (-dy, -dx) — see the roll convention.
        dx, dy = -px * factor, -py * factor
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return (0, 0)
        return (int(dx), int(dy))
    except Exception:
        return (0, 0)


def _shift_image(image, dx: int, dy: int):
    """Translate image content by (+dx right, +dy down); white fill at edges."""
    from PIL import Image

    return image.transform(image.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                           resample=Image.Resampling.NEAREST,
                           fillcolor=(255, 255, 255))


def _normalize_export(export_path: str, canvas, reference_path: Optional[str],
                      evidence_dir: str, thresholds: dict) -> tuple:
    """Scale-normalize (and coarsely align) the export to canvas pixels.

    Returns (aligned_png_path, info_dict).  The aligned copy is written into
    the evidence dir; the raw export is never modified.
    """
    import numpy as np
    from PIL import Image

    os.makedirs(evidence_dir, exist_ok=True)
    with Image.open(export_path) as raw:
        raw = raw.convert("RGB")
        export_size = raw.size
        scale = _detect_scale(export_size, canvas, float(thresholds["scale_tolerance"]))
        work = raw if raw.size == tuple(canvas) else raw.resize(tuple(canvas), Image.Resampling.LANCZOS)

        alignment = {"dx": 0, "dy": 0, "applied": False}
        if reference_path and os.path.exists(reference_path):
            try:
                with Image.open(reference_path) as ref_im:
                    ref_gray = _gray(ref_im.resize(tuple(canvas), Image.Resampling.LANCZOS))
                mov_gray = _gray(work)
                dx, dy = _estimate_shift(ref_gray, mov_gray, int(thresholds["max_align_shift_px"]))
                if dx or dy:
                    shifted = _shift_image(work, -dx, -dy)
                    before = float(np.abs(ref_gray - mov_gray).mean())
                    after = float(np.abs(ref_gray - _gray(shifted)).mean())
                    if after < before:
                        work = shifted
                        alignment = {"dx": int(dx), "dy": int(dy), "applied": True}
                    else:
                        alignment = {"dx": int(dx), "dy": int(dy), "applied": False}
            except Exception:
                alignment = {"dx": 0, "dy": 0, "applied": False}

        aligned_path = os.path.join(evidence_dir, "exported_aligned.png")
        work.save(aligned_path)
    info = {"size": list(export_size), "canvas": list(canvas),
            "scale": scale, "alignment": alignment}
    return aligned_path, info


# --------------------------------------------------------------------------- drift regions


def drift_regions(preview_path: str, aligned_export_path: str, grid: int,
                  top_n: int, region_ssim_min: float) -> dict:
    """Grid-cell SSIM between preview and Figma export → worst regions.

    Localizes WHERE the plugin renders differently than our simulation so a
    single drift number becomes an actionable pointer (fonts? gradients?
    effects?).  Cells are scored color-aware (mean per-channel SSIM): a pure
    chroma difference — wrong fill color, gradient interpolation, color
    profile — is real drift that grayscale SSIM would be blind to.  bboxes are
    in canvas pixels.
    """
    import numpy as np
    from PIL import Image

    with Image.open(preview_path) as pv:
        preview = pv.convert("RGB")
        with Image.open(aligned_export_path) as ex:
            export = ex.convert("RGB").resize(preview.size, Image.Resampling.LANCZOS)
        a = np.asarray(preview, dtype=np.float64)
        b = np.asarray(export, dtype=np.float64)
        width, height = preview.size
    grid = max(2, int(grid))
    ys = np.linspace(0, height, grid + 1).astype(int)
    xs = np.linspace(0, width, grid + 1).astype(int)
    cells = np.ones((grid, grid), dtype=np.float64)
    rows = []
    for i in range(grid):
        for j in range(grid):
            pa = a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            pb = b[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if not pa.size:
                continue
            value = float(np.mean([_ssim_pair(pa[..., c], pb[..., c])
                                   for c in range(pa.shape[-1])]))
            value = max(-1.0, min(1.0, value))
            cells[i, j] = value
            rows.append({
                "row": i, "col": j, "ssim": round(float(value), 4),
                "bbox": {"x": int(xs[j]), "y": int(ys[i]),
                         "w": int(xs[j + 1] - xs[j]), "h": int(ys[i + 1] - ys[i])},
            })
    worst = [dict(row, rank=rank + 1) for rank, row in enumerate(
        sorted((r for r in rows if r["ssim"] < region_ssim_min),
               key=lambda r: r["ssim"])[:max(0, int(top_n))]
    )]
    return {"grid": [grid, grid], "cell_ssim": [[round(float(v), 4) for v in line] for line in cells],
            "regions": worst}


def save_drift_heatmap(preview_path: str, cell_ssim, regions, out_path: str) -> str:
    """Red-overlay heatmap of per-cell drift with the worst regions outlined."""
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(preview_path) as pv:
        base = pv.convert("RGB")
        width, height = base.size
        cells = np.asarray(cell_ssim, dtype=np.float64)
        drift = np.clip(1.0 - cells, 0.0, 1.0)
        alpha_small = (drift * 200).astype(np.uint8)
        alpha = Image.fromarray(alpha_small, mode="L").resize((width, height),
                                                              Image.Resampling.NEAREST)
        overlay = Image.new("RGB", (width, height), (255, 32, 32))
        heat = base.copy()
        heat.paste(overlay, (0, 0), alpha)
    draw = ImageDraw.Draw(heat)
    for region in regions or []:
        box = region.get("bbox") or {}
        x, y = int(box.get("x", 0)), int(box.get("y", 0))
        w, h = int(box.get("w", 0)), int(box.get("h", 0))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 0, 255), width=2)
        draw.text((x + 3, y + 2), f"#{region.get('rank')}: {region.get('ssim')}",
                  fill=(255, 0, 255))
    heat.save(out_path)
    return out_path


# --------------------------------------------------------------------------- export OCR


def _resolve_export_ocr(run_dir: str, export_path: str, cfg: dict,
                        allow_ocr: bool, evidence_dir: str, warnings: list) -> tuple:
    """Return (render_ocr_dict_or_None, evidence_dict).

    Reuses the pipeline's render_ocr.json only when its provenance says it was
    OCR'd from the Figma export (run_pipeline records provenance.render_path)
    and it is at least as fresh as the export.  Otherwise, optionally runs OCR
    on the export directly — output goes to figma_verify/export_ocr.json, never
    over the pipeline-owned render_ocr.json.
    """
    render_ocr_path = os.path.join(run_dir, "render_ocr.json")
    data = _load_json(render_ocr_path)
    if data:
        provenance = data.get("provenance") or {}
        render_path = str(provenance.get("render_path") or "")
        same_file = os.path.basename(render_path).lower() == os.path.basename(export_path).lower()
        try:
            fresh = os.path.getmtime(render_ocr_path) >= os.path.getmtime(export_path)
        except OSError:
            fresh = False
        if same_file and fresh:
            return data, {"source": "render_ocr.json", "reused": True}
        if same_file and not fresh:
            warnings.append("render_ocr.json was OCR'd from an older figma_export.png — ignored")
    if not allow_ocr:
        return None, {"source": None, "reused": False,
                      "note": "no export OCR available (qa_ocr disabled or reuse mismatch)"}
    try:
        from src import ocr as ocr_module

        result = ocr_module.run_ocr(export_path, cfg, run_dir="")
        if isinstance(result, dict):
            result = dict(result)
            result["provenance"] = {"kind": "figma-verify",
                                    "render_path": os.path.abspath(export_path)}
            os.makedirs(evidence_dir, exist_ok=True)
            out_path = os.path.join(evidence_dir, "export_ocr.json")
            _atomic_json(out_path, result)
            return result, {"source": os.path.join(EVIDENCE_DIR, "export_ocr.json"),
                            "reused": False}
    except Exception as exc:  # noqa: BLE001 — OCR loss is evidence, not a crash
        warnings.append(f"export OCR unavailable: {type(exc).__name__}: {exc}")
    return None, {"source": None, "reused": False, "note": "export OCR failed or unavailable"}


# --------------------------------------------------------------------------- verdict


def _check(checks: list, name: str, status: str, value=None, threshold=None,
           detail: str = "") -> dict:
    entry = {"check": name, "status": status, "value": value,
             "threshold": threshold, "detail": detail}
    checks.append(entry)
    return entry


def _finish(result: FigmaVerifyResult, write: bool) -> FigmaVerifyResult:
    result.generated_at = _utc_now()
    if write:
        report_path = os.path.join(result.run_dir, REPORT_NAME)
        result.report_path = report_path
        _atomic_json(report_path, result.to_dict())
    return result


def verify(run_dir: str, exported_png_path: Optional[str] = None,
           cfg: Optional[dict] = None, *, allow_ocr: Optional[bool] = None,
           allow_stale: bool = False, write: bool = True) -> FigmaVerifyResult:
    """Score the plugin-exported PNG and emit run_dir/figma_qa.json.

    ``allow_ocr=None`` derives from cfg figma.qa_ocr (reuse of a matching
    render_ocr.json is always attempted first and never runs models).
    ``allow_stale=True`` scores an export older than design.json anyway, but
    the verdict is capped at ``degraded`` because a stale export is not
    evidence about the current design.
    """
    run_dir = os.path.abspath(str(run_dir or ""))
    cfg = cfg if cfg is not None else load_default_cfg()
    thresholds = resolve_thresholds(cfg)
    if allow_ocr is None:
        allow_ocr = bool(((cfg.get("figma") or {}).get("qa_ocr", False)) or cfg.get("qa_ocr", False))

    result = FigmaVerifyResult(run_dir=run_dir, thresholds=thresholds)
    checks, warnings = result.checks, result.warnings

    if not os.path.isdir(run_dir):
        result.status, result.verdict = "error", VERDICT_NOT_EXPORTED
        _check(checks, "run-dir", "fail", detail=f"not a directory: {run_dir}")
        return _finish(result, write=False)

    # -- 1. locate the export --------------------------------------------------
    export_path = exported_png_path or find_export(run_dir)
    if exported_png_path and not os.path.exists(exported_png_path):
        warnings.append(f"explicit export path missing: {exported_png_path}")
        export_path = None
    if not export_path:
        result.status, result.verdict = "not-exported", VERDICT_NOT_EXPORTED
        _check(checks, "export-present", "fail",
               detail="no figma_export.png in run dir — run the plugin's Import "
                      "(the bridge writes POST /export to the run dir)")
        return _finish(result, write)
    export_path = os.path.abspath(export_path)
    result.export = {
        "path": export_path,
        "bytes": os.path.getsize(export_path),
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(export_path))),
    }
    _check(checks, "export-present", "pass", detail=os.path.basename(export_path))

    # -- 2. freshness (stale exports are not evidence — run_pipeline semantics) -
    fresh = _export_is_fresh(export_path, run_dir)
    result.export["fresh"] = fresh
    stale_scored = False
    if fresh is False and not allow_stale:
        result.status, result.verdict = "stale-export", VERDICT_NOT_EXPORTED
        _check(checks, "export-fresh", "fail",
               detail="figma_export.png is older than design.json — re-import in "
                      "Figma (or pass allow_stale to score it anyway, capped at degraded)")
        return _finish(result, write)
    if fresh is False:
        stale_scored = True
        _check(checks, "export-fresh", "fail",
               detail="stale export scored via allow_stale — verdict capped at degraded")
    elif fresh is None:
        _check(checks, "export-fresh", "skip", detail="no design.json mtime to compare against")
    else:
        _check(checks, "export-fresh", "pass")

    # -- 3. reference images ----------------------------------------------------
    reference = None
    for name in ("normalized.png", "original.png"):
        candidate = os.path.join(run_dir, name)
        if os.path.exists(candidate):
            reference = candidate
            break
    if reference is None:
        result.status, result.verdict = "error", VERDICT_FAILED
        _check(checks, "reference-present", "fail",
               detail="neither normalized.png nor original.png exists — cannot judge fidelity")
        return _finish(result, write)
    preview = os.path.join(run_dir, "preview.png")
    preview = preview if os.path.exists(preview) else None

    canvas = _canvas_size(run_dir)
    if canvas is None:
        from PIL import Image

        with Image.open(export_path) as im:
            canvas = im.size
        warnings.append("no design.json canvas — using export dimensions as canvas")

    # -- 4. scale-normalize + align --------------------------------------------
    evidence_dir = os.path.join(run_dir, EVIDENCE_DIR)
    try:
        aligned_path, info = _normalize_export(export_path, canvas, reference,
                                               evidence_dir, thresholds)
        result.export.update(info)
        result.export["aligned_png"] = aligned_path
    except Exception as exc:  # noqa: BLE001
        result.status, result.verdict = "error", VERDICT_FAILED
        _check(checks, "export-decode", "fail",
               detail=f"cannot read/normalize export: {type(exc).__name__}: {exc}")
        return _finish(result, write)
    scale = result.export.get("scale") or {}
    scale_ok = bool(scale.get("uniform")) and scale.get("detected") is not None
    _check(checks, "export-scale", "pass" if scale_ok else "fail",
           value=scale, detail="" if scale_ok else
           "export dimensions are not a clean 1x/2x/3x multiple of the design canvas")

    # -- 5. text evidence --------------------------------------------------------
    source_ocr = _load_json(os.path.join(run_dir, "ocr.json"))
    export_ocr, ocr_evidence = _resolve_export_ocr(run_dir, export_path, cfg,
                                                   allow_ocr, evidence_dir, warnings)

    # -- 6. fidelity: exported-vs-original (the real number) ---------------------
    removal_mask = os.path.join(run_dir, "removal_mask.png")
    removal_mask = removal_mask if os.path.exists(removal_mask) else None
    fidelity = _score_pair(reference, aligned_path,
                           os.path.join(evidence_dir, "fidelity"),
                           source_ocr=source_ocr, render_ocr=export_ocr,
                           removal_mask=removal_mask)
    fidelity["reference"] = reference
    fidelity["text_evidence"] = ocr_evidence
    result.fidelity = fidelity

    fid_ssim = _fnum(fidelity.get("ssim")) or 0.0
    fid_min = float(thresholds["fidelity_ssim_min"])
    _check(checks, "fidelity-ssim", "pass" if fid_ssim >= fid_min else "fail",
           value=round(fid_ssim, 4), threshold=fid_min,
           detail="Figma's rendered frame vs the original source")
    recall = _fnum(fidelity.get("text_recall"))
    recall_min = float(thresholds["text_recall_min"])
    if recall is None:
        _check(checks, "fidelity-text-recall", "skip", threshold=recall_min,
               detail="no OCR of the Figma export — text survival in real Figma is unproven")
    else:
        _check(checks, "fidelity-text-recall",
               "pass" if recall >= recall_min else "fail",
               value=round(recall, 4), threshold=recall_min,
               detail="source OCR lines recovered from the Figma export")

    # -- 7. preview drift: exported-vs-preview (the drift detector) --------------
    drift_material = False
    if preview:
        drift = _score_pair(preview, aligned_path, os.path.join(evidence_dir, "drift"))
        drift["preview"] = preview
        drift_ssim = _fnum(drift.get("ssim")) or 0.0
        drift_min = float(thresholds["drift_ssim_min"])
        drift_material = drift_ssim < drift_min
        drift["material"] = drift_material
        _check(checks, "preview-drift-ssim",
               "pass" if not drift_material else "fail",
               value=round(drift_ssim, 4), threshold=drift_min,
               detail="does Figma render what render_preview.py simulated?")
        try:
            regions = drift_regions(preview, aligned_path,
                                    grid=int(thresholds["drift_grid"]),
                                    top_n=int(thresholds["drift_top_n"]),
                                    region_ssim_min=float(thresholds["drift_region_ssim_min"]))
            drift.update(regions)
            heatmap = save_drift_heatmap(preview, regions["cell_ssim"],
                                         regions["regions"],
                                         os.path.join(evidence_dir, "drift_heatmap.png"))
            drift["heatmap_png"] = heatmap
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"drift region breakdown failed: {type(exc).__name__}: {exc}")
        result.preview_drift = drift
    else:
        _check(checks, "preview-drift-ssim", "skip",
               detail="no preview.png — drift between simulation and Figma is unknown")
        result.preview_drift = {"material": None,
                                "note": "preview.png missing; drift not measurable"}

    # -- 8. verdict ---------------------------------------------------------------
    result.status = "scored" if not stale_scored else "stale-export"
    fidelity_failed = fid_ssim < fid_min
    text_failed = recall is not None and recall < recall_min
    engine_broken = fidelity.get("engine") == "figma_verify-fallback" and fidelity.get("engine_error")
    if engine_broken:
        warnings.append(f"primary QA engine unavailable: {fidelity.get('engine_error')}")

    if fidelity_failed or text_failed:
        result.verdict = VERDICT_FAILED
    else:
        degrade_reasons = []
        if drift_material:
            degrade_reasons.append("material preview drift")
        if preview is None:
            degrade_reasons.append("no preview to measure drift against")
        if not scale_ok:
            degrade_reasons.append("anomalous export scale")
        if stale_scored:
            degrade_reasons.append("stale export scored via allow_stale")
        if recall is None and bool(thresholds.get("require_text_evidence")):
            degrade_reasons.append("text recall unproven (require_text_evidence)")
        if engine_broken:
            degrade_reasons.append("scored by fallback judge only")
        if degrade_reasons:
            result.verdict = VERDICT_DEGRADED
            warnings.extend(f"degraded: {reason}" for reason in degrade_reasons)
        else:
            result.verdict = VERDICT_VERIFIED
    return _finish(result, write)
