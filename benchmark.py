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

# Codia construction CONTRACT pass bar (docs/CODIA-PARITY-SPEC.md): a per-run PASS requires
# native text everywhere (no handwriting archetype exists, so 0.90 is universal), a clean
# plate (zero unresolved glyph residue), and placement within tolerance. SSIM is NOT part of
# the contract pass — it is a floor gate only.
CONTRACT_NATIVE_TEXT_MIN = 0.90


def contract_verdict(row: dict) -> dict:
    """Per-run Codia-contract verdict for the --contract summary.

    Prefers the qa.json contract block; recomputes from the row's fields when a run predates
    it. native_text_ratio >= 0.90, zero glyph residue, placement within tolerance.
    """
    ntr = row.get("native_text_ratio")
    native_ok = ntr is not None and float(ntr) >= CONTRACT_NATIVE_TEXT_MIN
    residue_clean = row.get("glyph_residue_clean")
    placement_ok = row.get("placement_ok")
    reasons = []
    if ntr is None:
        reasons.append("native_text_ratio unknown")
    elif not native_ok:
        reasons.append(f"native text {float(ntr):.0%} < {CONTRACT_NATIVE_TEXT_MIN:.0%}")
    if residue_clean is False:
        reasons.append("unresolved glyph residue")
    if placement_ok is False:
        reasons.append("placement out of tolerance")
    reported = row.get("contract_pass")
    passed = bool(native_ok and residue_clean is not False and placement_ok is not False)
    if reported is not None:
        passed = bool(reported) and passed
    return {"id": row.get("id"), "pass": passed, "native_text_ratio": ntr,
            "glyph_residue_clean": residue_clean, "placement_ok": placement_ok,
            "reasons": reasons}


def requires_runtime_smoke(cfg: dict) -> bool:
    """Whether this config is making a real-model acceptance claim.

    A direct ``python benchmark.py`` invocation must not be weaker than the Windows
    launcher. Development configs can still opt out by leaving both settings false.
    """
    runtime = cfg.get("runtime") or {}
    inpaint = cfg.get("inpaint") or {}
    return bool(runtime.get("require_active_models", False) or
                inpaint.get("strict_acceptance", False))


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


def configure_figma_acceptance(cfg: dict, wait_s: int) -> None:
    """Turn a benchmark into a real Figma round-trip acceptance run.

    Preview QA remains useful for local iteration. This explicit switch is for final
    evidence: it stages the design, waits for the plugin's fresh export, and rejects a
    local-preview-only result or a missing compiler report.
    """
    figma = cfg.setdefault("figma", {})
    figma["enabled"] = True
    figma["require_export"] = True
    cfg["export_wait_s"] = max(1, int(wait_s))


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
    leaf_accounting = meta.get("leaf_accounting") or structure.get("leaf_accounting") or {}
    archetype = (qa.get("archetype") or meta.get("archetype")
                 or reconstruction.get("archetype") or runtime.get("archetype"))
    preset = (qa.get("preset") or meta.get("preset")
              or reconstruction.get("preset") or runtime.get("preset"))
    # F14: no stage writes archetype/preset into any of the sources above, so the columns
    # were always "—". archetype.json IS produced for every run — consult it as a fallback.
    # The archetype selects a preset of the same name, so it fills the preset column too.
    if archetype is None or preset is None:
        arch_decision = _read(run_dir / "archetype.json", {})
        archetype = archetype or arch_decision.get("archetype")
        preset = preset or arch_decision.get("archetype")
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
        # ── CODIA CONSTRUCTION CONTRACT (the objective — leads the benchmark) ─────────
        # native text %, ghost-free (clean plate), and the contract score come FIRST; the
        # visual/ssim columns follow as a report. See docs/CODIA-PARITY-SPEC.md.
        "native_text_ratio": qa.get("native_text_ratio",
                                    (qa.get("contract") or {}).get("native_text_ratio")),
        "contract_score": qa.get("contract_score", (qa.get("contract") or {}).get("contract_score")),
        "contract_pass": qa.get("contract_pass", (qa.get("contract") or {}).get("pass")),
        "glyph_residue_clean": (qa.get("contract") or {}).get("glyph_residue_clean"),
        "placement_ok": (qa.get("contract") or {}).get("placement_ok"),
        "construction_score": (qa.get("construction") or (qa.get("contract") or {}).get("construction") or {}).get("score"),
        "text_recall": qa.get("text_recall"),
        "editable_text_recall": qa.get("editable_text_recall"),
        # F-honesty: editable_text_recall's own denominator is only the text OCR detected,
        # never the ad's full source text -- a run where OCR missed most of the copy can
        # still read a perfect 1.0 if the sliver it did find is all editable (021: text_recall
        # 0.17, editable_text_recall 1.0). true_text_coverage = text_recall *
        # editable_text_recall is the honest share of ALL source text that ended up correct
        # AND editable.
        "true_text_coverage": qa.get("true_text_coverage", structure.get("true_text_coverage")),
        # F4: detected text lines shipped as raster (slice / wordmark / foreground_raster)
        # instead of editable TEXT. Surfaced so slices are visible, not hidden inside a 1.0.
        "rasterized_text_count": qa.get("rasterized_text_count",
                                        structure.get("rasterized_text_count")),
        "rasterized_text_ratio": qa.get("rasterized_text_ratio",
                                        structure.get("rasterized_text_ratio")),
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
        # F-honesty: prefer the top-level qa.json mirror (pixel_diff now hoists it there the
        # same way editable_text_recall already is); fall back to the nested structural copy
        # for older run artifacts written before that mirror existed.
        "element_recall": qa.get("element_recall", structure.get("element_recall")),
        "element_survival": qa.get("element_survival", structure.get("element_survival")),
        "background_audit": structure.get("background"),
        "layer_alpha_audit": structure.get("layer_alpha") or [],
        "missing_assets": structure.get("missing_assets") or [],
        "missing_fonts": structure.get("missing_fonts") or [],
        "font_substitutions": structure.get("font_substitutions") or [],
        "editable_ratio": (design.get("meta") or {}).get("editable_ratio"),
        "native_leaf_ratio": meta.get("native_leaf_ratio", structure.get("native_leaf_ratio")),
        "leaf_accounting": leaf_accounting,
        "intentional_raster_clusters": int(
            leaf_accounting.get("intentional_raster_cluster_count", 0) or 0
        ),
        "unexplained_raster_fallbacks": int(
            leaf_accounting.get("unexplained_raster_count", 0) or 0
        ),
        "run_dir": str(run_dir),
        **_harness_telemetry(run_dir),
    }


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _markdown(report: dict) -> str:
    summary = report["summary"]
    contract_pass = summary.get("contract_passing")
    lines = [
        "# Image decompiler benchmark",
        "",
        f"Images: {summary['images']}  |  Contract passing: {contract_pass if contract_pass is not None else '—'}  "
        f"|  QA passing: {summary['qa_passing']}  |  Runtime accepted: {summary.get('runtime_accepted', '—')}",
        "",
        "Columns lead with the Codia construction CONTRACT (native text %, ghost-free clean "
        "plate, contract score) — the objective — then the visual/ssim REPORT. "
        "See docs/CODIA-PARITY-SPEC.md.",
        "",
        "| image | archetype | native text | ghost-free | contract | contract score | construction | QA | visual | ssim | evidence | runtime | seconds | text | editable text | true text coverage | raster text | native leaves | raster clusters | edge | element recall | regional routes | hard fails |",
        "| --- | --- | ---: | :---: | :---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["runs"]:
        fails = ", ".join(item.get("rule", "unknown") for item in row["hard_fails"]) or "—"
        def metric(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.3f}"
        def pct(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.0%}"
        def flag(key):
            value = row.get(key)
            return "—" if value is None else ("yes" if value else "NO")
        # raster text: how much detected copy shipped as pixels rather than editable TEXT —
        # the column that keeps editable_text_recall honest instead of a hidden 1.0 (F4).
        raster_text = "—"
        if row.get("rasterized_text_count") is not None:
            raster_text = str(int(row["rasterized_text_count"]))
            if row.get("rasterized_text_ratio") is not None:
                raster_text += f" ({float(row['rasterized_text_ratio']):.0%})"
        construction = row.get("construction_score")
        construction = "—" if construction is None else f"{float(construction):.0f}"
        lines.append(
            f"| {row['id']} | {row.get('archetype') or '—'} | {pct('native_text_ratio')} | {flag('glyph_residue_clean')} | {flag('contract_pass')} | {metric('contract_score')} | {construction} | "
            f"{'pass' if row['qa_ok'] else 'fail'} | {metric('visual_score')} | {metric('ssim')} | {'complete' if row.get('qa_evidence_complete') else 'missing'} | {row.get('runtime_status') or ('ok' if row.get('runtime_ok') else 'unknown')} | {metric('duration_s')} | "
            f"{metric('text_recall')} | {metric('editable_text_recall')} | {metric('true_text_coverage')} | {raster_text} | {metric('native_leaf_ratio')} | {row.get('intentional_raster_clusters', 0)} | {metric('edge_f1')} | {metric('element_recall')} | "
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


def _emit_html_report(output: Path) -> None:
    """Best-effort visual report.html hook; never fails the benchmark run."""
    import sys
    scripts_dir = Path(__file__).resolve().parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import report_html
        path = report_html.generate_report(output)
        print(f"HTML report: {path}")
    except Exception as exc:  # pragma: no cover - reporting must not break a benchmark
        print(f"report.html generation skipped: {exc}")


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
    parser.add_argument("--require-figma-export", action="store_true",
                        help="final acceptance: require a fresh plugin export and compiler report for every image")
    parser.add_argument("--figma-wait-s", type=int, default=120,
                        help="seconds to wait per image for its Figma plugin export when --require-figma-export is set")
    parser.add_argument("--contract", action="store_true",
                        help="print the per-image Codia construction-contract verdict "
                        "(native text >= 90%%, zero glyph residue, placement in tolerance)")
    args = parser.parse_args()
    source_dir = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    if args.require_figma_export:
        configure_figma_acceptance(cfg, args.figma_wait_s)
        print(f"Figma acceptance: waiting up to {cfg['export_wait_s']}s per image for a fresh plugin export.")
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
    acceptance_smoke = requires_runtime_smoke(cfg)
    if args.deep_smoke or acceptance_smoke:
        if acceptance_smoke and not args.deep_smoke:
            print("Acceptance config: running required real-model smoke before images.")
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
            # Codia contract: the objective, ahead of qa/runtime in the headline.
            "contract_passing": sum(1 for row in runs if contract_verdict(row)["pass"]),
            "mean_native_text_ratio": _mean(runs, "native_text_ratio"),
            "mean_contract_score": _mean(runs, "contract_score"),
            "runtime_accepted": sum(1 for row in runs if row["runtime_ok"]),
            "degraded_runs": sum(1 for row in runs if row["runtime_degraded"]),
            "runtime_violation_runs": sum(1 for row in runs if row["runtime_violations"]),
            "mean_visual_score": _mean(runs, "visual_score"),
            "mean_ssim": _mean(runs, "ssim"),
            "mean_text_recall": _mean(runs, "text_recall"),
            "mean_editable_text_recall": _mean(runs, "editable_text_recall"),
            "mean_true_text_coverage": _mean(runs, "true_text_coverage"),
            "rasterized_text_total": sum(
                int(row.get("rasterized_text_count") or 0) for row in runs
            ),
            "mean_native_leaf_ratio": _mean(runs, "native_leaf_ratio"),
            "intentional_raster_clusters_total": sum(
                row.get("intentional_raster_clusters", 0) for row in runs
            ),
            "unexplained_raster_fallbacks_total": sum(
                row.get("unexplained_raster_fallbacks", 0) for row in runs
            ),
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
    _emit_html_report(output)
    # --contract: the Codia construction-contract verdict is the objective, printed ahead of
    # the raw summary. Always emit the one-line roll-up; --contract adds the per-image detail.
    verdicts = [contract_verdict(row) for row in runs]
    passing = sum(1 for v in verdicts if v["pass"])
    print(f"\nCONTRACT: {passing}/{len(runs)} pass "
          f"(native text >= {CONTRACT_NATIVE_TEXT_MIN:.0%}, zero glyph residue, placement in tolerance)")
    if args.contract:
        for verdict in verdicts:
            ntr = verdict["native_text_ratio"]
            ntr_str = "—" if ntr is None else f"{float(ntr):.0%}"
            status = "PASS" if verdict["pass"] else "fail"
            why = "" if verdict["pass"] else "  <- " + "; ".join(verdict["reasons"])
            print(f"  [{status}] {verdict['id']}  native_text={ntr_str}{why}")
    print(json.dumps(report["summary"], indent=2))
    passing = (report["summary"]["images"] > 0
               and report["summary"]["complete_runs"] == len(runs)
               and report["summary"]["qa_passing"] == len(runs)
               and report["summary"]["runtime_accepted"] == len(runs))
    raise SystemExit(0 if passing else 2)


if __name__ == "__main__":
    main()
