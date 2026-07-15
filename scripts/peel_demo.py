#!/usr/bin/env python3
"""LayerD-style peel decomposition demo on a single flattened image.

Runs src/peel_decompose.py end to end and writes into the output dir:

    layer_00.png … layer_NN.png   full-canvas RGBA, layer_00 = TOPMOST peeled layer
    background.png                residual plate after the last peel
    manifest.json                 bbox / area / coverage / peel_order / z per layer
    composite_check.png           background + layers recomposited (sanity overlay)

Matting downloads ~1 GB of BiRefNet weights on first use (HF cache).  Default device is
CPU on purpose — the RTX 5080 is usually contended by the main pipeline; pass
``--device cuda`` only when the GPU is actually free.

Usage:
  python scripts/peel_demo.py --input benchmark_set/052.png --output runs/peel-demo/052
  python scripts/peel_demo.py --input ad.png --output out --inpaint lama --max-layers 4
  python scripts/peel_demo.py --input ad.png --output out --matting rembg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import peel_decompose  # noqa: E402
from src.console_io import configure_stdio  # noqa: E402

configure_stdio()


def load_cfg(path: str) -> dict:
    """Standalone config loader (same YAML-then-JSON ladder as run_pipeline.load_cfg,
    duplicated here so the demo never imports the pipeline's module graph)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        with open(p, encoding="utf-8") as f:
            return json.load(f)


def _composite_check(result, out_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    plate = np.asarray(result.background, dtype=np.float64)
    for layer in reversed(result.layers):  # back-to-front
        rgba = np.asarray(layer.rgba, dtype=np.float64)
        a = rgba[:, :, 3:4] / 255.0
        plate = rgba[:, :, :3] * a + plate * (1.0 - a)
    Image.fromarray(plate.round().astype(np.uint8)).save(out_dir / "composite_check.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="flattened design image (png/jpg)")
    parser.add_argument("--output", required=True, help="output directory for layers + manifest")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"),
                        help="pipeline config; only the optional 'peel' block is read")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="peel iteration cap (default: cfg or 3, LayerD's default)")
    parser.add_argument("--matting", choices=["auto", "birefnet", "rembg"], default=None,
                        help="matting backend override (default: cfg peel.matting.backend)")
    parser.add_argument("--hf-card", default=None,
                        help="HF checkpoint for birefnet matting "
                             "(default cyberagent/layerd-birefnet)")
    parser.add_argument("--device", default=None,
                        help="matting device, cpu (default) or cuda — GPU is usually "
                             "contended by the main pipeline")
    parser.add_argument("--inpaint", choices=["opencv", "lama"], default="opencv",
                        help="hole filler: deterministic OpenCV Telea or Big-LaMa "
                             "(simple-lama-inpainting, LayerD-faithful)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    peel_cfg = dict(cfg.get("peel") or {})
    matting_cfg = dict(peel_cfg.get("matting") or {})
    if args.matting:
        matting_cfg["backend"] = args.matting
    if args.hf_card:
        matting_cfg["hf_card"] = args.hf_card
    if args.device:
        matting_cfg["device"] = args.device
    peel_cfg["matting"] = matting_cfg
    cfg["peel"] = peel_cfg

    inpaint = (peel_decompose.make_simple_lama_inpaint() if args.inpaint == "lama"
               else peel_decompose.opencv_inpaint)

    print(f"[peel] matting backend={matting_cfg.get('backend', 'auto')} "
          f"device={matting_cfg.get('device', 'cpu')} inpaint={args.inpaint}")
    print("[peel] loading matting model (first run downloads ~1 GB from HF)…")
    started = time.time()
    matting = peel_decompose.resolve_matting(cfg)
    print(f"[peel] matting ready in {time.time() - started:.1f}s")

    started = time.time()
    result = peel_decompose.peel(args.input, max_layers=args.max_layers, cfg=cfg,
                                 matting=matting, inpaint=inpaint)
    elapsed = time.time() - started

    out_dir = Path(args.output)
    manifest = peel_decompose.write_outputs(result, str(out_dir))
    _composite_check(result, out_dir)

    print(f"[peel] {len(result.layers)} layer(s) in {elapsed:.1f}s "
          f"→ stop: {result.stop_reason}")
    for entry in manifest["layers"]:
        box = entry["bbox"]
        print(f"  {entry['file']}  z={entry['z']}  "
              f"bbox=({box['x']},{box['y']} {box['w']}x{box['h']})  "
              f"coverage={entry['coverage']:.3f}")
    print(f"  background.png  z=0")
    print(f"[peel] wrote {out_dir / 'manifest.json'} (+ composite_check.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
