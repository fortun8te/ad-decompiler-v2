#!/usr/bin/env python3
"""Flux Fill inpaint settings A/B on real golden-ad removal holes.

Crop-local probe (no full pipeline re-run): for each fixture run dir it reuses
``normalized.png`` + ``removal_mask.png`` + the recorded inpaint regions, crops the
holes Flux actually handles (recorded ``flux-comfy`` regions) plus the single largest
non-text hole as a hallucination stress case, then runs Flux Fill with each settings
variant and scores continuation quality.

Axes that matter (docs/RESEARCH-CODIA-GAP-ANALYSIS.md §4 P0-3):
  guidance {3.5, 30} x steps {8, 20} x prompt {empty, descriptive}

Scored per crop: seam energy (lower = cleaner continuation), hole MAE (large =
possible hallucination — inspect the montage), preserve MSE (outside-mask drift),
elapsed s. A montage PNG per crop (source | mask | each variant) is written for visual
inspection, since a hallucinated object can pass a low seam score.

Usage:
  python scripts/flux_probe.py --run-dirs runs/golden-optimized-check/009_* runs/.../041_* \
      --output runs/flux-settings-ab
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import load_cfg  # noqa: E402
from src import inpaint, qwen_worker  # noqa: E402

_DESCRIPTIVE = "clean plain seamless background, empty, no text, no objects"

# One-factor-at-a-time from the current live baseline (g30/s8/empty), plus the
# documented proven-good combination.
VARIANTS: dict[str, dict] = {
    "g30_s8_empty":    {"guidance": 30.0, "steps": 8,  "prompt": ""},
    "g3.5_s8_empty":   {"guidance": 3.5,  "steps": 8,  "prompt": ""},
    "g30_s20_empty":   {"guidance": 30.0, "steps": 20, "prompt": ""},
    "g30_s8_prompt":   {"guidance": 30.0, "steps": 8,  "prompt": _DESCRIPTIVE},
    "g3.5_s20_prompt": {"guidance": 3.5,  "steps": 20, "prompt": _DESCRIPTIVE},
}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _seam_score(source, filled, mask) -> float:
    preview = source.copy()
    preview[mask > 0] = filled[mask > 0]
    return float(inpaint._seam_energy(preview, mask))


def _preserve_mse(source, filled, mask) -> float:
    keep = mask == 0
    if not np.any(keep):
        return 0.0
    diff = np.abs(source.astype(np.float32) - filled.astype(np.float32))
    return float(diff[keep].mean())


def _hole_mae(source, filled, mask) -> float:
    hole = mask > 0
    if not np.any(hole):
        return 0.0
    return float(np.abs(source.astype(np.float32) - filled.astype(np.float32))[hole].mean())


def _crop(source, union, bbox, cfg):
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    x0, y0 = int(bbox["x"]), int(bbox["y"])
    x1, y1 = x0 + int(bbox["w"]), y0 + int(bbox["h"])
    mask[y0:y1, x0:x1] = union[y0:y1, x0:x1]
    spec = inpaint._regional_crop(mask, cfg)
    if spec is None:
        return None
    (cx0, cy0, cx1, cy1), padding, _ctx = spec
    pr, pb, _, _ = padding
    crop_rgb = source[cy0:cy1, cx0:cx1].copy()
    crop_mask = mask[cy0:cy1, cx0:cx1].copy()
    if pr or pb:
        import cv2
        crop_rgb = cv2.copyMakeBorder(crop_rgb, 0, pb, 0, pr, cv2.BORDER_REPLICATE)
        crop_mask = cv2.copyMakeBorder(crop_mask, 0, pb, 0, pr, cv2.BORDER_CONSTANT, value=0)
    return crop_rgb, crop_mask


def _select_crops(run_dir: Path, cfg: dict) -> list[dict]:
    """Recorded flux-comfy regions + the largest non-text hole (hallucination stress)."""
    recon = json.loads((run_dir / "reconstruction.json").read_text(encoding="utf-8"))
    regions = (recon.get("stats") or {}).get("inpaint", {}).get("regions") or []
    chosen: list[dict] = []
    for reg in regions:
        if reg.get("backend") == "flux-comfy":
            chosen.append({"tag": f"r{reg['index']}-flux", "bbox": reg["bbox"],
                           "targets": reg.get("targets"), "note": "recorded-flux"})
    chosen = chosen[:2]  # cap flux crops per fixture to bound runtime
    # Stress case: the largest hole that is not pure text (mixed product/image holes on a
    # plain-ish background are where diffusion Fill is most tempted to hallucinate an object).
    stress = [r for r in regions if set(r.get("targets") or []) != {"text"}]
    if stress:
        biggest = max(stress, key=lambda r: r.get("masked_fraction_canvas", 0))
        if not any(c["bbox"] == biggest["bbox"] for c in chosen):
            chosen.append({"tag": f"r{biggest['index']}-stress", "bbox": biggest["bbox"],
                           "targets": biggest.get("targets"), "note": "largest-non-pure-text"})
    return chosen


def _montage(crops: list[np.ndarray], labels: list[str], out_path: Path) -> None:
    import cv2
    h = max(c.shape[0] for c in crops)
    tiles = []
    for img, label in zip(crops, labels):
        pad = cv2.copyMakeBorder(img, 0, h - img.shape[0] + 24, 0, 8,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(pad, label, (4, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(pad)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Golden run dirs with normalized.png + removal_mask.png + reconstruction.json")
    parser.add_argument("--output", default="runs/flux-settings-ab")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()

    out_root = ROOT / args.output
    out_root.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    base_cfg = load_cfg(str(ROOT / "config.yaml"))
    # Deterministic single attempt; adaptive quant off so every variant uses the same GGUF.
    base_cfg.setdefault("inpaint", {}).setdefault("comfy", {})
    base_cfg["inpaint"]["comfy"]["attempts"] = 1
    base_cfg["inpaint"]["comfy"]["vram_adaptive_quant"] = False

    summary: dict = {"variants": names, "fixtures": {}}
    for raw in args.run_dirs:
        matches = sorted(Path().glob(raw)) or [Path(raw)]
        run_dir = matches[0]
        if not (run_dir / "reconstruction.json").exists():
            print(f"skip {run_dir}: no reconstruction.json")
            continue
        source = _load_rgb(run_dir / "normalized.png")
        union = np.asarray(Image.open(run_dir / "removal_mask.png").convert("L"), dtype=np.uint8)
        crops = _select_crops(run_dir, base_cfg)
        fx = run_dir.name.split("_")[0]
        print(f"\n=== fixture {fx} ({run_dir.name}): {len(crops)} crops ===")
        fixture_out = out_root / fx
        fixture_out.mkdir(parents=True, exist_ok=True)
        summary["fixtures"][fx] = {}
        for spec in crops:
            crop = _crop(source, union, spec["bbox"], base_cfg)
            if crop is None:
                continue
            crop_rgb, crop_mask = crop
            tag = spec["tag"]
            montage_imgs = [crop_rgb, np.dstack([crop_mask] * 3)]
            montage_labels = ["source", "mask"]
            rows = []
            for name in names:
                v = VARIANTS[name]
                cfg = copy.deepcopy(base_cfg)
                cfg["inpaint"]["comfy"].update(
                    {"guidance": v["guidance"], "steps": v["steps"], "prompt": v["prompt"]}
                )
                t0 = time.monotonic()
                out = qwen_worker.flux_inpaint(crop_rgb, crop_mask, cfg)
                elapsed = time.monotonic() - t0
                if out is None:
                    print(f"  {tag} {name}: FAIL ({elapsed:.1f}s) {getattr(qwen_worker.flux_inpaint, '__dict__', {}).get('last_error','')}")
                    rows.append({"variant": name, "ok": False, "elapsed_s": round(elapsed, 1)})
                    continue
                out = np.asarray(out, dtype=np.uint8)
                if out.shape[:2] != crop_rgb.shape[:2]:
                    import cv2
                    out = cv2.resize(out, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                composite = crop_rgb.copy()
                composite[crop_mask > 0] = out[crop_mask > 0]
                row = {
                    "variant": name, "ok": True, "elapsed_s": round(elapsed, 1),
                    "seam": round(_seam_score(crop_rgb, out, crop_mask), 3),
                    "hole_mae": round(_hole_mae(crop_rgb, out, crop_mask), 3),
                    "preserve_mse": round(_preserve_mse(crop_rgb, out, crop_mask), 3),
                    **{k: v[k] for k in ("guidance", "steps", "prompt")},
                }
                rows.append(row)
                montage_imgs.append(composite)
                montage_labels.append(f"{name} seam{row['seam']}")
                print(f"  {tag} {name}: seam={row['seam']} hole_mae={row['hole_mae']} "
                      f"preserve_mse={row['preserve_mse']} ({elapsed:.1f}s)")
            montage_path = fixture_out / f"{tag}_montage.png"
            try:
                _montage(montage_imgs, montage_labels, montage_path)
            except Exception as exc:
                print(f"  (montage failed: {exc})")
            summary["fixtures"][fx][tag] = {"note": spec["note"], "targets": spec.get("targets"),
                                            "montage": str(montage_path), "rows": rows}

    # Aggregate mean seam per variant across all ok crops.
    agg: dict = {}
    for fx in summary["fixtures"].values():
        for crop in fx.values():
            for row in crop["rows"]:
                if row.get("ok"):
                    agg.setdefault(row["variant"], []).append(row)
    ranking = []
    for name in names:
        rws = agg.get(name, [])
        if not rws:
            continue
        ranking.append({
            "variant": name,
            "mean_seam": round(float(np.mean([r["seam"] for r in rws])), 3),
            "mean_hole_mae": round(float(np.mean([r["hole_mae"] for r in rws])), 3),
            "mean_preserve_mse": round(float(np.mean([r["preserve_mse"] for r in rws])), 3),
            "mean_elapsed_s": round(float(np.mean([r["elapsed_s"] for r in rws])), 1),
            "n": len(rws),
        })
    ranking.sort(key=lambda r: r["mean_seam"])
    summary["ranking"] = ranking
    (out_root / "flux_settings_ab.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== ranking (lower mean_seam = cleaner continuation) ===")
    for r in ranking:
        print(f"  {r['variant']:16} seam={r['mean_seam']:<7} hole_mae={r['mean_hole_mae']:<7} "
              f"preserve_mse={r['mean_preserve_mse']:<6} {r['mean_elapsed_s']}s (n={r['n']})")
    print(f"\nWrote {out_root / 'flux_settings_ab.json'} + montages under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
