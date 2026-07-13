#!/usr/bin/env python3
"""Run and score a reproducible image-decompiler benchmark on the GPU machine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from doctor import inspect as inspect_machine
from run_pipeline import STAGES, load_cfg, run_one
from src.harness import harness_enabled


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
REQUIRED_ARTIFACTS = (
    "input_manifest.json", "normalized.png", "ocr_raw.json", "ocr.json",
    "residual.json", "qwen.json", "sam3.json", "fused_elements.json",
    "elements.json", "merged.json", "reconstruction.json", "layout.json",
    "design.json", "design_preflight.json", "background_clean.png", "removal_mask.png",
    "ownership.png", "layers_contact.png", "preview.png", "diff.png",
    "runtime_report.json", "qa.json",
)

VISUAL_FAILURE_RULES = frozenset({
    "background-leakage", "unclean-background", "inpaint-outside-mask",
    "layer-alpha-holes", "empty-layer-alpha", "low-element-recall",
})


def _normalize_fixture_id(value: str) -> str:
    """Normalize a requested fixture prefix (``26`` and ``026`` both become ``026``)."""
    value = str(value).strip()
    if not value:
        raise ValueError("fixture ID cannot be empty")
    return value.zfill(3) if value.isdigit() else value.lower()


def parse_fixture_ids(values: list[str] | None) -> list[str]:
    """Parse repeatable/comma-separated CLI values and reject duplicate requests."""
    requested = [
        _normalize_fixture_id(item)
        for value in (values or [])
        for item in str(value).split(",")
        if item.strip()
    ]
    duplicates = sorted({item for item in requested if requested.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate fixture IDs requested: {', '.join(duplicates)}")
    return requested


def _file_fixture_id(path: Path) -> str:
    return _normalize_fixture_id(path.stem.split("_", 1)[0])


def select_images(source_dir: Path, max_images: int | None = None,
                  fixture_ids: list[str] | None = None) -> list[Path]:
    """Return a stable benchmark selection, limited after sorting by filename."""
    images = sorted(path for path in source_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    requested = parse_fixture_ids(fixture_ids)
    if requested:
        if max_images is not None and max_images != len(requested):
            raise ValueError("--max-images cannot truncate named fixtures; omit it or match the ID count")
        by_id: dict[str, list[Path]] = {}
        for path in images:
            by_id.setdefault(_file_fixture_id(path), []).append(path)
        missing = [item for item in requested if not by_id.get(item)]
        ambiguous = {item: by_id[item] for item in requested if len(by_id.get(item, [])) > 1}
        if missing:
            raise ValueError(f"missing fixture IDs: {', '.join(missing)}")
        if ambiguous:
            detail = "; ".join(
                f"{item}: {', '.join(path.name for path in paths)}"
                for item, paths in ambiguous.items()
            )
            raise ValueError(f"duplicate files for fixture IDs: {detail}")
        images = [by_id[item][0] for item in requested]
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be at least 1")
        images = images[:max_images]
    return images


def _source_manifest(images: list[Path], requested_ids: list[str]) -> dict:
    resolved = []
    for path in images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        resolved.append({
            "id": _file_fixture_id(path),
            "filename": path.name,
            "path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        })
    return {"requested_ids": requested_ids, "resolved": resolved}


def configure_auto_repair(cfg: dict, enabled: bool) -> None:
    """Set both repair switches so an explicit CLI disable cannot leave the harness active."""
    runtime = cfg.setdefault("runtime", {})
    runtime["auto_repair"] = bool(enabled)
    runtime.setdefault("harness", {})["enabled"] = bool(enabled)


def _read(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _harness_telemetry(run_dir: Path) -> dict:
    """Summarize harness_loop.json (or legacy harness.json) for benchmark rows."""
    loop_path = run_dir / "harness_loop.json"
    legacy_path = run_dir / "harness.json"
    loop = _read(loop_path, None)
    if loop is None:
        loop = _read(legacy_path, {})

    qa = _read(run_dir / "qa.json", {})
    rounds = loop.get("rounds") or loop.get("attempts") or []
    round_count = loop.get("round_count")
    if round_count is None:
        round_count = loop.get("iterations")
    if round_count is None:
        round_count = len(rounds)

    final_qa_ok = loop.get("final_qa_ok")
    if final_qa_ok is None:
        final_qa_ok = loop.get("qa_ok")
    if final_qa_ok is None:
        final_qa_ok = qa.get("ok")

    auto_fixed = loop.get("auto_fixed")
    if auto_fixed is None:
        initial_qa_ok = loop.get("initial_qa_ok")
        if initial_qa_ok is None and rounds:
            first = rounds[0] if isinstance(rounds[0], dict) else {}
            initial_qa_ok = first.get("qa_ok_before")
        if initial_qa_ok is None:
            stopped = loop.get("stopped")
            if stopped == "already_ok":
                initial_qa_ok = True
            elif stopped == "qa_ok" and int(round_count or 0) > 0:
                initial_qa_ok = False
        auto_fixed = bool(initial_qa_ok is False and final_qa_ok)
    elif loop.get("stopped") == "already_ok":
        auto_fixed = False

    telemetry = {
        "auto_fixed": bool(auto_fixed),
        "harness_rounds": int(round_count or 0),
        "final_qa_ok": bool(final_qa_ok),
    }
    if loop:
        telemetry["harness"] = {
            "stopped": loop.get("stopped"),
            "round_count": int(round_count or 0),
            "auto_fixed": bool(auto_fixed),
            "final_qa_ok": bool(final_qa_ok),
        }
        if rounds:
            telemetry["harness"]["rounds"] = rounds

    for key, name in (
        ("harness_loop_path", "harness_loop.json"),
        ("critic_path", "critic.json"),
        ("fixer_path", "fixer.json"),
    ):
        path = run_dir / name
        if path.exists():
            telemetry[key] = str(path)
    return telemetry


def _entry(run_dir: Path, result: dict) -> dict:
    qa = _read(run_dir / "qa.json", {})
    reconstruction = _read(run_dir / "reconstruction.json", {})
    design = _read(run_dir / "design.json", {})
    runtime = _read(run_dir / "runtime_report.json", {})
    structure = qa.get("structural") or {}
    hard_fails = qa.get("hard_fails") if isinstance(qa.get("hard_fails"), list) else None
    structural_hard_fails = structure.get("hard_fails") if isinstance(structure.get("hard_fails"), list) else None
    merged_hard_fails = list(hard_fails or [])
    seen_failures = {(item.get("rule"), item.get("detail")) for item in merged_hard_fails if isinstance(item, dict)}
    for item in structural_hard_fails or []:
        key = (item.get("rule"), item.get("detail")) if isinstance(item, dict) else None
        if key and key not in seen_failures:
            merged_hard_fails.append(item)
            seen_failures.add(key)
    qa_evidence_complete = bool(
        isinstance(qa, dict)
        and hard_fails is not None
        and isinstance(qa.get("structural"), dict)
        and structural_hard_fails is not None
        and all(key in structure for key in ("background", "layer_alpha", "element_recall"))
    )
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS
                         if not (run_dir / name).is_file()]
    complete = not missing_artifacts
    runtime_ok = bool(runtime.get("acceptable")) if runtime else False
    qa_ok = bool(qa.get("ok")) and qa_evidence_complete and not merged_hard_fails
    visual_failure_rules = [
        item.get("rule") for item in merged_hard_fails
        if isinstance(item, dict) and item.get("rule") in VISUAL_FAILURE_RULES
    ]
    meta = design.get("meta") or {}
    archetype = (qa.get("archetype") or meta.get("archetype")
                 or reconstruction.get("archetype") or runtime.get("archetype"))
    preset = (qa.get("preset") or meta.get("preset")
              or reconstruction.get("preset") or runtime.get("preset"))
    inpaint = (reconstruction.get("stats") or {}).get("inpaint") or {}
    regions = inpaint.get("regions") or reconstruction.get("inpaint_regions") or []
    route_counts: dict[str, int] = {}
    fallback_dispositions: list[dict] = []
    for region in regions if isinstance(regions, list) else []:
        if not isinstance(region, dict):
            continue
        route = str(region.get("route") or region.get("backend") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        if region.get("fallback") or region.get("fallback_reason") or region.get("disposition"):
            fallback_dispositions.append({
                key: region.get(key) for key in
                ("ids", "route", "backend", "fallback", "fallback_reason", "disposition")
                if region.get(key) is not None
            })
    return {
        "id": run_dir.name,
        "pipeline_ok": bool(result.get("ok")) and complete,
        "complete": complete,
        "missing_artifacts": missing_artifacts,
        "runtime_ok": runtime_ok and complete,
        "runtime_status": runtime.get("status", result.get("runtime_status")),
        "runtime_degraded": runtime.get("degraded") or [],
        "runtime_violations": runtime.get("violations") or [],
        "duration_s": result.get("duration_s"),
        "qa_ok": qa_ok and complete,
        "qa_evidence_complete": qa_evidence_complete,
        "visual_score": qa.get("visual_score"),
        "ssim": qa.get("ssim"),
        "text_recall": qa.get("text_recall"),
        "editable_text_recall": qa.get("editable_text_recall"),
        "archetype": archetype,
        "preset": preset,
        "edge_f1": qa.get("edge_f1"),
        "color_similarity": qa.get("color_similarity"),
        "hard_fails": merged_hard_fails,
        "visual_failure_rules": visual_failure_rules,
        "duplicate_observations_removed": (reconstruction.get("stats") or {}).get("duplicates_removed", 0),
        "vectorized": (reconstruction.get("stats") or {}).get("vectorized", 0),
        "vector_fallback": (reconstruction.get("stats") or {}).get("vector_fallback", 0),
        "regional_inpaint_backend": inpaint.get("backend"),
        "regional_inpaint_routes": route_counts,
        "fallback_dispositions": fallback_dispositions,
        "background_leakage": "background-leakage" in visual_failure_rules,
        "inpaint_outside_mask": "inpaint-outside-mask" in visual_failure_rules,
        "layer_alpha_holes": "layer-alpha-holes" in visual_failure_rules,
        "empty_layer_alpha": "empty-layer-alpha" in visual_failure_rules,
        "low_element_recall": "low-element-recall" in visual_failure_rules,
        "element_recall": structure.get("element_recall"),
        "element_survival": structure.get("element_survival"),
        "background_audit": structure.get("background"),
        "layer_alpha_audit": structure.get("layer_alpha") or [],
        "missing_assets": structure.get("missing_assets") or [],
        "missing_fonts": structure.get("missing_fonts") or [],
        "font_substitutions": structure.get("font_substitutions") or [],
        "editable_ratio": (design.get("meta") or {}).get("editable_ratio"),
        "run_dir": str(run_dir),
        **_harness_telemetry(run_dir),
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
        "| image | archetype | preset | QA | evidence | runtime | seconds | visual | text | editable text | edge | element recall | regional routes | hard fails |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["runs"]:
        fails = ", ".join(item.get("rule", "unknown") for item in row["hard_fails"]) or "—"
        def metric(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {row['id']} | {row.get('archetype') or '—'} | {row.get('preset') or '—'} | {'pass' if row['qa_ok'] else 'fail'} | {'complete' if row.get('qa_evidence_complete') else 'missing'} | {row.get('runtime_status') or ('ok' if row.get('runtime_ok') else 'unknown')} | {metric('duration_s')} | {metric('visual_score')} | "
            f"{metric('text_recall')} | {metric('editable_text_recall')} | {metric('edge_f1')} | {metric('element_recall')} | "
            f"{', '.join(f'{k}:{v}' for k, v in row.get('regional_inpaint_routes', {}).items()) or '—'} | {fails} |"
        )
    fixture_manifest = report.get("fixture_manifest") or {}
    if fixture_manifest.get("resolved"):
        lines.extend(["", "## Fixture manifest", "", "| id | filename | sha256 |", "| --- | --- | --- |"])
        for item in fixture_manifest["resolved"]:
            lines.append(f"| {item['id']} | {item['filename']} | `{item['sha256']}` |")
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
    parser.add_argument(
        "--auto-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable runtime.auto_repair / harness loop (default: on when harness enabled in config)",
    )
    parser.add_argument("--max-images", type=int, default=None,
                        help="benchmark only the first N images after stable filename sorting (for example, 5)")
    parser.add_argument("--ids", "--include", action="append", default=[],
                        help="fixture ID/prefix to include; repeat or comma-separate (for example 26,034)")
    parser.add_argument("--skip-doctor", action="store_true",
                        help="development only; benchmark normally refuses an unready model machine")
    parser.add_argument("--deep-smoke", action="store_true",
                        help="run bounded actual OCR/SAM/VLM/Big-LaMa/vector/Figma probes before images")
    parser.add_argument("--probe-timeout", type=float, default=120,
                        help="maximum seconds for each isolated deep-smoke probe")
    args = parser.parse_args()
    source_dir = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    auto_repair = args.auto_repair if args.auto_repair is not None else harness_enabled(cfg)
    configure_auto_repair(cfg, auto_repair)
    from src.runtime_bootstrap import ensure_services
    startup = ensure_services(cfg)
    (output / "startup.json").write_text(json.dumps(startup, indent=2), encoding="utf-8")
    if not startup.get("ok"):
        print(json.dumps({"benchmark": "blocked", "startup": startup.get("checks")}, indent=2))
        raise SystemExit(2)
    if not args.skip_doctor:
        preflight = inspect_machine(cfg, Path(__file__).resolve().parent)
        (output / "doctor.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        if not preflight.get("ok"):
            print(json.dumps({"benchmark": "blocked", "doctor": preflight.get("blockers")}, indent=2))
            raise SystemExit(2)
    if args.deep_smoke:
        from runtime_smoke import run_all
        smoke = run_all(cfg, output / "runtime-smoke", timeout_s=args.probe_timeout)
        if not smoke.get("ok"):
            print(json.dumps({"benchmark": "blocked", "runtime_smoke": smoke.get("checks")}, indent=2))
            raise SystemExit(2)
    startup_probes = tuple(((cfg.get("runtime") or {}).get("startup_smoke") or []))
    if startup_probes and not args.deep_smoke:
        from runtime_smoke import run_all
        smoke = run_all(cfg, output / "runtime-smoke", probes=startup_probes,
                        timeout_s=args.probe_timeout)
        if not smoke.get("ok"):
            print(json.dumps({"benchmark": "blocked", "runtime_smoke": smoke.get("checks")}, indent=2))
            raise SystemExit(2)
    try:
        requested_ids = parse_fixture_ids(args.ids)
        images = select_images(source_dir, args.max_images, requested_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
        "fixture_manifest": _source_manifest(images, requested_ids),
        "runs": runs,
        "summary": {
            "images": len(runs),
            "complete_runs": sum(1 for row in runs if row["complete"]),
            "pipeline_passing": sum(1 for row in runs if row["pipeline_ok"]),
            "qa_passing": sum(1 for row in runs if row["qa_ok"]),
            "runtime_accepted": sum(1 for row in runs if row["runtime_ok"]),
            "degraded_runs": sum(1 for row in runs if row["runtime_degraded"]),
            "runtime_violation_runs": sum(1 for row in runs if row["runtime_violations"]),
            "mean_visual_score": _mean(runs, "visual_score"),
            "mean_ssim": _mean(runs, "ssim"),
            "mean_text_recall": _mean(runs, "text_recall"),
            "mean_editable_text_recall": _mean(runs, "editable_text_recall"),
            "mean_edge_f1": _mean(runs, "edge_f1"),
            "background_leakage_runs": sum(1 for row in runs if row["background_leakage"]),
            "inpaint_outside_mask_runs": sum(1 for row in runs if row["inpaint_outside_mask"]),
            "layer_alpha_hole_runs": sum(1 for row in runs if row["layer_alpha_holes"]),
            "empty_layer_alpha_runs": sum(1 for row in runs if row["empty_layer_alpha"]),
            "low_element_recall_runs": sum(1 for row in runs if row["low_element_recall"]),
            "qa_evidence_complete_runs": sum(1 for row in runs if row["qa_evidence_complete"]),
            "mean_element_recall": _mean(runs, "element_recall"),
            "runs_with_hard_fails": sum(1 for row in runs if row["hard_fails"]),
            "auto_fixed_runs": sum(1 for row in runs if row.get("auto_fixed")),
            "harness_rounds_total": sum(row.get("harness_rounds", 0) for row in runs),
            "final_qa_passing": sum(1 for row in runs if row.get("final_qa_ok")),
        },
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "benchmark.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    passing = (report["summary"]["images"] > 0
               and report["summary"]["complete_runs"] == len(runs)
               and report["summary"]["qa_passing"] == len(runs)
               and report["summary"]["runtime_accepted"] == len(runs))
    raise SystemExit(0 if passing else 2)


if __name__ == "__main__":
    main()
