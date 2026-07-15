#!/usr/bin/env python3
"""CLI for the "verified in real Figma" QA gate (src/figma_verify.py).

Usage:
    # one run dir
    python scripts/figma_verify.py runs/golden-final/009_attached_885c19be02ccf229

    # every run dir under a parent (benchmark suite)
    python scripts/figma_verify.py --all runs/golden-optimized-check

    # human-in-the-loop: block until the plugin's export lands, then print the verdict
    python scripts/figma_verify.py runs/my-run --watch --timeout 600

Exit codes: 0 = no run failed (with --strict: every run verified),
1 = at least one 'failed' verdict, 2 = --strict and a run was not 'verified',
3 = --watch timed out waiting for an export, 4 = usage/environment error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import figma_verify  # noqa: E402
from src.console_io import safe_print  # noqa: E402

RUN_MARKERS = ("design.json", "input_manifest.json", "normalized.png", "preview.png")


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, marker)) for marker in RUN_MARKERS
    )


def _discover_runs(parent: str) -> list:
    if _is_run_dir(parent):
        return [parent]
    found = []
    try:
        entries = sorted(os.listdir(parent))
    except OSError:
        return []
    for name in entries:
        child = os.path.join(parent, name)
        if _is_run_dir(child):
            found.append(child)
    return found


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _note(result) -> str:
    for entry in result.checks:
        if entry.get("status") == "fail":
            return f"{entry.get('check')}: {entry.get('detail') or entry.get('value')}"
    if result.warnings:
        return str(result.warnings[0])
    return ""


def _print_table(results: list) -> None:
    headers = ("run", "verdict", "status", "fid_ssim", "drift_ssim", "recall", "note")
    rows = []
    for result in results:
        rows.append((
            os.path.basename(result.run_dir.rstrip("/\\")),
            result.verdict,
            result.status,
            _fmt((result.fidelity or {}).get("ssim")),
            _fmt((result.preview_drift or {}).get("ssim")),
            _fmt((result.fidelity or {}).get("text_recall"), 2),
            _note(result)[:70],
        ))
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows
              else len(headers[i]) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    safe_print(line)
    safe_print("  ".join("-" * w for w in widths))
    for row in rows:
        safe_print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    tally = {}
    for result in results:
        tally[result.verdict] = tally.get(result.verdict, 0) + 1
    summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(tally.items()))
    safe_print(f"\n{len(results)} run(s): {summary}")


def _watch(run_dir: str, cfg: dict, args) -> object:
    """Poll until a fresh export appears (the bridge writes it atomically)."""
    deadline = time.time() + max(1.0, float(args.timeout))
    announced = False
    while True:
        export = figma_verify.find_export(run_dir)
        if export is not None:
            fresh = figma_verify._export_is_fresh(export, run_dir)
            if args.allow_stale or fresh is not False:
                return figma_verify.verify(run_dir, exported_png_path=export, cfg=cfg,
                                           allow_ocr=args.ocr, allow_stale=args.allow_stale)
            if not announced:
                safe_print(f"[watch] stale export in {run_dir} — waiting for a fresh one "
                      "(re-import in Figma)", flush=True)
                announced = True
        elif not announced:
            safe_print(f"[watch] waiting for figma_export.png in {run_dir} "
                  "(Figma: run plugin → Import latest)", flush=True)
            announced = True
        if time.time() >= deadline:
            return None
        time.sleep(max(0.05, float(args.poll)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score plugin-exported Figma PNGs into a verified/degraded/"
                    "failed/not-exported verdict (writes figma_qa.json per run).")
    parser.add_argument("run_dir", nargs="?", help="a single run directory")
    parser.add_argument("--all", metavar="PARENT",
                        help="verify every run dir directly under PARENT")
    parser.add_argument("--export", help="explicit export PNG path (single run only)")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--watch", action="store_true",
                        help="poll the run dir until an export appears, then verify")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="--watch timeout in seconds (default 600)")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="--watch poll interval in seconds (default 2)")
    parser.add_argument("--allow-stale", action="store_true",
                        help="score an export older than design.json (verdict capped "
                             "at degraded)")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--ocr", dest="ocr", action="store_true", default=None,
                           help="force OCR of the export for text recall")
    ocr_group.add_argument("--no-ocr", dest="ocr", action="store_false",
                           help="never run OCR (still reuses a matching render_ocr.json)")
    parser.add_argument("--json", action="store_true",
                        help="print full figma_qa.json payload(s) instead of the table")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero unless every run is 'verified' "
                             "(production sign-off mode)")
    args = parser.parse_args(argv)

    if bool(args.run_dir) == bool(args.all):
        parser.print_usage()
        safe_print("error: pass exactly one of <run_dir> or --all PARENT", file=sys.stderr)
        return 4
    if args.watch and args.all:
        safe_print("error: --watch works on a single run dir", file=sys.stderr)
        return 4

    cfg = figma_verify.load_default_cfg(args.config)
    results = []
    if args.all:
        run_dirs = _discover_runs(args.all)
        if not run_dirs:
            safe_print(f"error: no run dirs found under {args.all}", file=sys.stderr)
            return 4
        for run_dir in run_dirs:
            results.append(figma_verify.verify(run_dir, cfg=cfg, allow_ocr=args.ocr,
                                               allow_stale=args.allow_stale))
    elif args.watch:
        result = _watch(args.run_dir, cfg, args)
        if result is None:
            safe_print(f"[watch] timed out after {args.timeout:.0f}s — verdict: not-exported")
            return 3
        results.append(result)
    else:
        results.append(figma_verify.verify(args.run_dir, exported_png_path=args.export,
                                           cfg=cfg, allow_ocr=args.ocr,
                                           allow_stale=args.allow_stale))

    if args.json:
        for result in results:
            safe_print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_table(results)

    if any(result.verdict == figma_verify.VERDICT_FAILED for result in results):
        return 1
    if args.strict and any(result.verdict != figma_verify.VERDICT_VERIFIED
                           for result in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
