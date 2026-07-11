#!/usr/bin/env python3
"""Run and score a reproducible image-decompiler benchmark on the GPU machine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from doctor import inspect as inspect_machine
from run_pipeline import STAGES, load_cfg, run_one


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _read(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _entry(run_dir: Path, result: dict) -> dict:
    qa = _read(run_dir / "qa.json", {})
    reconstruction = _read(run_dir / "reconstruction.json", {})
    design = _read(run_dir / "design.json", {})
    runtime = _read(run_dir / "runtime_report.json", {})
    structure = qa.get("structural") or {}
    return {
        "id": run_dir.name,
        "pipeline_ok": bool(result.get("ok")),
        "runtime_ok": bool(runtime.get("acceptable", result.get("runtime_ok", False))),
        "runtime_status": runtime.get("status", result.get("runtime_status")),
        "runtime_degraded": runtime.get("degraded") or [],
        "runtime_violations": runtime.get("violations") or [],
        "duration_s": result.get("duration_s"),
        "qa_ok": bool(qa.get("ok")),
        "visual_score": qa.get("visual_score"),
        "ssim": qa.get("ssim"),
        "text_recall": qa.get("text_recall"),
        "editable_text_recall": qa.get("editable_text_recall"),
        "edge_f1": qa.get("edge_f1"),
        "color_similarity": qa.get("color_similarity"),
        "hard_fails": qa.get("hard_fails") or [],
        "duplicate_observations_removed": (reconstruction.get("stats") or {}).get("duplicates_removed", 0),
        "vectorized": (reconstruction.get("stats") or {}).get("vectorized", 0),
        "vector_fallback": (reconstruction.get("stats") or {}).get("vector_fallback", 0),
        "background_leakage": bool(any(item.get("rule") == "background-leakage" for item in qa.get("hard_fails") or [])),
        "missing_assets": structure.get("missing_assets") or [],
        "missing_fonts": structure.get("missing_fonts") or [],
        "font_substitutions": structure.get("font_substitutions") or [],
        "editable_ratio": (design.get("meta") or {}).get("editable_ratio"),
        "run_dir": str(run_dir),
    }


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Image decompiler benchmark",
        "",
        f"Images: {summary['images']}  |  QA passing: {summary['qa_passing']}  |  Runtime accepted: {summary.get('runtime_accepted', '—')}",
        "",
        "| image | QA | runtime | seconds | visual | text | edge | duplicates removed | hard fails |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["runs"]:
        fails = ", ".join(item.get("rule", "unknown") for item in row["hard_fails"]) or "—"
        def metric(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {row['id']} | {'pass' if row['qa_ok'] else 'fail'} | {row.get('runtime_status') or ('ok' if row.get('runtime_ok') else 'unknown')} | {metric('duration_s')} | {metric('visual_score')} | "
            f"{metric('text_recall')} | {metric('edge_f1')} | {row['duplicate_observations_removed']} | {fails} |"
        )
    lines.extend([
        "",
        "A benchmark is not complete until every hard fail has a deliberate disposition and the"
        "manual-cleanup time has been measured against the same inputs.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible image-to-Figma benchmark")
    parser.add_argument("--input-dir", required=True, help="directory containing benchmark images")
    parser.add_argument("--output", default="runs/benchmark", help="benchmark report/run root")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default="normalize", choices=STAGES)
    parser.add_argument("--skip-doctor", action="store_true",
                        help="development only; benchmark normally refuses an unready model machine")
    args = parser.parse_args()
    source_dir = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    if not args.skip_doctor:
        preflight = inspect_machine(cfg, Path(__file__).resolve().parent)
        (output / "doctor.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        if not preflight.get("ok"):
            print(json.dumps({"benchmark": "blocked", "doctor": preflight.get("blockers")}, indent=2))
            raise SystemExit(2)
    images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {source_dir}")

    runs = []
    for image in images:
        run_dir = output / image.stem
        print(f"\n=== {image.name} ===", flush=True)
        result = run_one(str(image), str(run_dir), cfg, args.resume)
        runs.append(_entry(run_dir, result))

    report = {
        "version": 1,
        "input_dir": str(source_dir.resolve()),
        "output": str(output.resolve()),
        "runs": runs,
        "summary": {
            "images": len(runs),
            "pipeline_passing": sum(1 for row in runs if row["pipeline_ok"]),
            "qa_passing": sum(1 for row in runs if row["qa_ok"]),
            "runtime_accepted": sum(1 for row in runs if row["runtime_ok"]),
            "degraded_runs": sum(1 for row in runs if row["runtime_degraded"]),
            "runtime_violation_runs": sum(1 for row in runs if row["runtime_violations"]),
            "mean_visual_score": _mean(runs, "visual_score"),
            "mean_ssim": _mean(runs, "ssim"),
            "mean_text_recall": _mean(runs, "text_recall"),
            "mean_edge_f1": _mean(runs, "edge_f1"),
            "background_leakage_runs": sum(1 for row in runs if row["background_leakage"]),
            "runs_with_hard_fails": sum(1 for row in runs if row["hard_fails"]),
        },
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "benchmark.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    passing = (report["summary"]["qa_passing"] == len(runs)
               and report["summary"]["runtime_accepted"] == len(runs))
    raise SystemExit(0 if passing else 2)


if __name__ == "__main__":
    main()
