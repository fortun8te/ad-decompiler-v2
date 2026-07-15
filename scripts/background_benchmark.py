#!/usr/bin/env python3
"""Run the known-ground-truth clean-background benchmark and RTX bakeoff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.background_benchmark import (  # noqa: E402
    BACKGROUND_FAMILIES,
    BAKEOFF_MODES,
    AcceptanceThresholds,
    run_inpaint_bakeoff,
    run_synthetic_benchmark,
)


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, for example 384x256") from exc
    if width < 96 or height < 96:
        raise argparse.ArgumentTypeError("size must be at least 96x96")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score clean-background generation against synthetic known ground truth."
    )
    parser.add_argument("--output", required=True, help="directory for paired cases and reports")
    parser.add_argument("--candidate-dir", help="external output directory containing <case-id>.png files")
    parser.add_argument("--method", choices=("telea", "ns", "copy-input", "oracle"), default="telea",
                        help="CPU baseline; oracle only smoke-tests the benchmark")
    parser.add_argument("--bakeoff-mode", choices=BAKEOFF_MODES,
                        help="run this actual src.inpaint backend using --config on every synthetic case")
    parser.add_argument("--config", help="YAML or JSON RTX config; required with --bakeoff-mode")
    parser.add_argument("--allow-backend-substitution", action="store_true",
                        help="record but do not reject a backend selected differently from --bakeoff-mode")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--size", type=_parse_size, default=(384, 256))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--families", nargs="+", choices=BACKGROUND_FAMILIES, default=list(BACKGROUND_FAMILIES))
    parser.add_argument("--max-inside-mae", type=float, default=8.0)
    parser.add_argument("--min-inside-psnr", type=float, default=30.0)
    parser.add_argument("--min-inside-ssim", type=float, default=0.90)
    args = parser.parse_args()
    if args.cases_per_family < 1:
        parser.error("--cases-per-family must be at least 1")
    if args.max_inside_mae < 0 or args.min_inside_psnr < 0 or not -1 <= args.min_inside_ssim <= 1:
        parser.error("acceptance thresholds must be non-negative (SSIM must be between -1 and 1)")

    thresholds = AcceptanceThresholds(
        max_inside_mae=args.max_inside_mae,
        min_inside_psnr_db=args.min_inside_psnr,
        min_inside_ssim=args.min_inside_ssim,
    )
    if args.bakeoff_mode:
        if not args.config:
            parser.error("--config is required with --bakeoff-mode")
        if args.candidate_dir:
            parser.error("--candidate-dir cannot be combined with --bakeoff-mode")
        if args.method != "telea":
            parser.error("--method cannot be combined with --bakeoff-mode")
        report = run_inpaint_bakeoff(
            args.output,
            config=args.config,
            requested_backend=args.bakeoff_mode,
            cases_per_family=args.cases_per_family,
            size=args.size,
            seed=args.seed,
            thresholds=thresholds,
            families=args.families,
            allow_backend_substitution=args.allow_backend_substitution,
        )
    else:
        if args.config:
            parser.error("--config is only valid with --bakeoff-mode")
        if args.allow_backend_substitution:
            parser.error("--allow-backend-substitution is only valid with --bakeoff-mode")
        report = run_synthetic_benchmark(
            args.output,
            cases_per_family=args.cases_per_family,
            size=args.size,
            seed=args.seed,
            method=args.method,
            candidate_dir=args.candidate_dir,
            thresholds=thresholds,
            families=args.families,
        )
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(0 if report["summary"]["accepted"] else 2)


if __name__ == "__main__":
    main()
