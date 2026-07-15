#!/usr/bin/env python3
"""Run inpaint A/B variants on golden inspo fixtures and compare benchmark metrics."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_IDS = "009,041,042,050,052"

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "q4ks": {
        "inpaint": {
            "comfy": {
                "models": {"unet_gguf": "flux1-fill-dev-Q4_K_S.gguf"},
            }
        }
    },
    "q5ks": {
        "inpaint": {
            "comfy": {
                "models": {"unet_gguf": "flux1-fill-dev-Q5_K_S.gguf"},
            }
        }
    },
    "q6k": {
        "inpaint": {
            "comfy": {
                "models": {"unet_gguf": "flux1-fill-dev-Q6_K.gguf"},
            }
        }
    },
    "flux_aggressive": {
        "inpaint": {
            "regional": {
                "flux_max_canvas_fraction": 0.12,
                "flux_residual_p90": 12,
                "flux_gradient_p90": 12,
            }
        }
    },
    "flux_force": {
        "inpaint": {
            "regional": {
                "force_flux": True,
                "flux_max_canvas_fraction": 0.25,
            }
        }
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder with inspo ad images")
    parser.add_argument("--output", default="runs/flux-ab", help="Output root for all variants")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="Comma-separated fixture IDs")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Variant names to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from run_pipeline import load_cfg

    base_cfg = load_cfg(str(ROOT / "config.yaml"))
    names = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    summary = {}
    for name in names:
        cfg_path = Path(args.output) / name / "config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = _deep_merge(base_cfg, VARIANTS[name])
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        out_dir = Path(args.output) / name
        cmd = [
            sys.executable,
            str(ROOT / "benchmark.py"),
            "--input-dir",
            args.input_dir,
            "--output",
            str(out_dir),
            "--ids",
            args.ids,
            "--config",
            str(cfg_path),
        ]
        print(f"\n=== variant {name} ===", flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        proc = subprocess.run(cmd, cwd=str(ROOT))
        report_path = out_dir / "benchmark.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary[name] = report.get("summary") or {}
        if proc.returncode not in (0, 2):
            return proc.returncode

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "flux_ab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
