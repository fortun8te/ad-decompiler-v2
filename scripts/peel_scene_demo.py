#!/usr/bin/env python3
"""Occlusion-attributed scene peel demo over an existing pipeline run directory.

Consumes the artifacts a normal run already produced — normalized.png,
fused_elements.json (or elements.json), canvas.json, ocr.json — and writes into
--output:

    layer_<z>_<id>.png          full-canvas RGBA, ORIGINAL pixels + attributed fills
    background.png              completed plate (skipped when --reuse-background)
    peel_scene_manifest.json    per-layer z / occluded_by / occludes / fills
    composite_check.png         background + layers recomposited (text excluded)

No models load by default: the hole filler is OpenCV Telea (pass --inpaint lama for
Big-LaMa via simple-lama-inpainting, CPU).  The overlap gate is honored — if nothing
overlaps, the demo reports "skipped: no-overlap" and writes only the manifest, exactly
what the pipeline integration would do (single-plate path keeps ownership semantics).

Usage:
  python scripts/peel_scene_demo.py --run runs/ad9_regional_final --output runs/peel-scene/ad9
  python scripts/peel_scene_demo.py --run runs/x --output out --inpaint lama --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import peel_decompose, peel_scene  # noqa: E402
from src.console_io import configure_stdio  # noqa: E402

configure_stdio()


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="existing pipeline run directory")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--inpaint", choices=["auto", "opencv", "lama"], default="auto",
                        help="auto = Big-LaMa when importable, else OpenCV Telea")
    parser.add_argument("--force", action="store_true",
                        help="bypass the overlap gate (peel even a flat scene)")
    parser.add_argument("--no-text-occluders", action="store_true",
                        help="ignore OCR lines (only element-over-element holes)")
    parser.add_argument("--reuse-background", action="store_true",
                        help="reuse the run's background_clean.png instead of refilling")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = _load_cfg(Path(args.config))
    cfg.setdefault("peel", {})
    if args.no_text_occluders:
        cfg["peel"]["text_occluders"] = "off"

    canvas = _load_json(run / "canvas.json")
    elements_path = run / "elements.json"
    if not elements_path.exists():
        elements_path = run / "fused_elements.json"
    fused = _load_json(elements_path)
    ocr = _load_json(run / "ocr.json") if (run / "ocr.json").exists() else None

    elements = peel_scene.elements_from_run(str(run), fused, canvas, cfg=cfg, ocr=ocr)
    print(f"[peel-scene] {len(fused)} fused elements → {len(elements)} scene layers "
          f"({sum(1 for e in elements if e.is_text)} text occluders)")

    report = peel_scene.overlap_report(elements, cfg)
    qualifying = [p for p in report["pairs"] if p["qualifies"]
                  and not p["under_is_text"] and not p.get("top_is_text")]
    print(f"[peel-scene] overlap gate: needed={report['needed']} "
          f"({len(qualifying)} qualifying element-over-element pairs, "
          f"{report.get('blocked_qualifying', 0)} blocked by the granularity guard)")
    for pair in qualifying[:12]:
        tag = "" if pair.get("eligible") else "  [BLOCKED: ineligible member]"
        print(f"    {pair['top']} over {pair['under']}: {pair['area']} px "
              f"({pair['frac']:.1%} of the smaller){tag}")
    for eid, e in sorted((report.get("eligibility") or {}).items()):
        if not e["eligible"]:
            print(f"    ineligible {eid}: {e['reason']}")

    if args.inpaint == "lama":
        inpaint = peel_decompose.make_simple_lama_inpaint()
        chosen = "lama"
    elif args.inpaint == "auto":
        try:
            inpaint = peel_decompose.make_simple_lama_inpaint()
            chosen = "lama (auto)"
        except Exception as exc:
            inpaint = peel_decompose.opencv_inpaint
            chosen = f"opencv (auto; lama unavailable: {exc})"
    else:
        inpaint = peel_decompose.opencv_inpaint
        chosen = "opencv"
    print(f"[peel-scene] hole filler: {chosen}")
    background = None
    if args.reuse_background and (run / "background_clean.png").exists():
        background = str(run / "background_clean.png")

    started = time.time()
    result = peel_scene.peel_scene(str(run / "normalized.png"), elements,
                                   inpaint=inpaint, cfg=cfg, background=background,
                                   force=args.force)
    elapsed = time.time() - started

    out = Path(args.output)
    manifest = peel_scene.write_outputs(result, str(out))
    if result.skipped:
        print(f"[peel-scene] skipped: {result.skip_reason} "
              "(single-plate path keeps this scene) — manifest written")
        return 0

    import numpy as np
    from PIL import Image
    plate = np.asarray(result.background, np.float64)
    for layer in sorted(result.layers, key=lambda l: l.z_index):
        rgba = np.asarray(layer.rgba, np.float64)
        a = rgba[:, :, 3:4] / 255.0
        plate = rgba[:, :, :3] * a + plate * (1.0 - a)
    Image.fromarray(plate.round().astype(np.uint8)).save(out / "composite_check.png")

    print(f"[peel-scene] {len(result.layers)} complete layers in {elapsed:.1f}s")
    for entry in manifest["layers"]:
        note = ""
        if entry["fills"]:
            filled = ", ".join(f"{f['occluder']}({f['area']}px"
                               f"{',text' if f['text_occluder'] else ''})"
                               for f in entry["fills"])
            note = f"  filled-under: {filled}"
        print(f"  z={entry['z']:>2} {entry['id']:<10} {entry['kind']:<14} "
              f"occluded_by={entry['occluded_by'] or '-'}{note}")
    rc = manifest["meta"].get("recomposite") or {}
    print(f"[peel-scene] recomposite: max_abs_diff={rc.get('max_abs_diff')} "
          f"mean={rc.get('mean_abs_diff')} (text px excluded: {rc.get('text_excluded_px')})")
    print(f"[peel-scene] wrote {out / 'peel_scene_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
