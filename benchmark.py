#!/usr/bin/env python3
"""Run and score a reproducible image-decompiler benchmark on the GPU machine."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

from doctor import inspect as inspect_machine
from run_pipeline import STAGES, load_cfg, run_one
from src.harness import harness_enabled
from src import format_readiness


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

# ── per-fixture watchdog ────────────────────────────────────────────────────────
# postfix-benchmark-6 lost all 16 fixtures when 094 wedged on a VLM call: benchmark.py
# invoked run_one() directly in-process (see run_pipeline.run_one), so once that call
# stalled -- LM Studio had dropped its loaded model, and every VLM call it touched
# retried into TimeoutErrors indefinitely (bench_stderr.log: repeated
# "vlm_timeout ... TimeoutError: timed out" with no end) -- there was no way to
# interrupt it and no later fixture (101, 104, 107, 131, 135) ever ran.
#
# signal.alarm does not exist on win32, so a same-process timeout cannot forcibly
# reclaim a blocked call. The fix mirrors the isolation runtime_smoke.py already uses
# for its GPU probes: run each fixture in its own child process (multiprocessing,
# spawn context -- the only context win32 supports) so the parent can actually
# TerminateProcess() it on timeout, record the failure, and move on to the next
# fixture. 091 legitimately took 2018s (heavy Flux inpaint + 2 harness rounds) in the
# same run, so the default timeout must clear genuine slow-but-working fixtures by a
# wide margin.
DEFAULT_FIXTURE_TIMEOUT_S = 3600.0

# REQUIRED_ARTIFACTS is already the ordered list of files each stage of run_pipeline
# produces; reuse it (rather than re-deriving from run_pipeline.STAGES, which does not
# map 1:1 onto artifact names) to guess which stage a killed fixture was stuck in: the
# first artifact still missing on disk when we kill it.
_ARTIFACT_STAGE = {
    "input_manifest.json": "normalize", "normalized.png": "normalize",
    "ocr_raw.json": "ocr", "ocr.json": "text",
    "residual.json": "residual", "qwen.json": "qwen", "sam3.json": "sam",
    "fused_elements.json": "elements", "elements.json": "elements",
    "merged.json": "merge", "reconstruction.json": "reconstruct",
    "background_clean.png": "reconstruct", "removal_mask.png": "reconstruct",
    "ownership.png": "reconstruct", "layout.json": "layout",
    "design.json": "design", "design_preflight.json": "design",
    "layers_contact.png": "preview", "preview.png": "preview",
    "diff.png": "qa", "runtime_report.json": "qa", "qa.json": "qa",
}


def _infer_wedged_stage(run_dir: Path) -> str:
    """Best-effort "which stage was this fixture stuck in" from artifact presence."""
    run_dir = Path(run_dir)
    for name in REQUIRED_ARTIFACTS:
        if not (run_dir / name).is_file():
            return _ARTIFACT_STAGE.get(name, name)
    return "qa"  # every artifact present; the harness loop after run_one() is the culprit


def _fixture_worker(image_path: str, run_dir: str, cfg: dict, resume: str, output_queue) -> None:
    """Child-process entry point (see ``run_bounded``): run one fixture end to end.

    Runs in its own OS process so the parent can forcibly kill it if it wedges;
    ``run_one`` already catches broadly, but a second net costs nothing here.
    """
    try:
        result = run_one(image_path, run_dir, cfg, resume)
    except Exception as exc:  # pragma: no cover - run_one already catches broadly
        result = {"ok": False, "run_dir": run_dir, "runtime_ok": False,
                  "error": f"{type(exc).__name__}: {exc}"}
    try:
        output_queue.put(result)
    except Exception:
        pass


def run_bounded(target, args: tuple, run_dir: Path, timeout_s: float, *, cfg: dict | None = None) -> dict:
    """Run ``target(*args, output_queue)`` in an isolated child process with a hard
    wall-clock timeout, killing it (and best-effort cancelling any orphaned remote
    ComfyUI job) if it outlives ``timeout_s``.

    ``target`` must be a module-level (picklable) callable; ``multiprocessing``'s
    ``spawn`` context is used unconditionally because it is the only context win32
    supports (``fork`` does not exist there), matching ``runtime_smoke._run_bounded``.
    """
    run_dir = Path(run_dir)
    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=1)
    process = context.Process(target=target, args=(*args, output_queue))
    started = time.monotonic()
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        stage = _infer_wedged_stage(run_dir)
        process.terminate()
        process.join(5)
        if process.is_alive():  # stubborn child; TerminateProcess again via kill()
            process.kill()
            process.join(5)
        if cfg is not None:
            # A killed worker does not cancel a remote job it started (Flux via
            # ComfyUI keeps sampling after we vanish) -- same caveat runtime_smoke.py
            # documents for its own probes; reuse its cleanup rather than reinventing it.
            try:
                from runtime_smoke import _cancel_remote_comfy_job
                _cancel_remote_comfy_job(cfg)
            except Exception:
                pass
        elapsed = round(time.monotonic() - started, 3)
        reason = f"watchdog: fixture exceeded {timeout_s:.0f}s timeout while in stage '{stage}'"
        record = {"ok": False, "run_dir": str(run_dir), "duration_s": elapsed,
                  "runtime_ok": False, "runtime_status": "timeout",
                  "error": reason, "failed_stage": stage, "timed_out": True,
                  "timeout_s": timeout_s}
        try:
            (run_dir / "watchdog_timeout.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError:
            pass
        return record
    try:
        result = output_queue.get(timeout=5)
    except queue.Empty:
        elapsed = round(time.monotonic() - started, 3)
        return {"ok": False, "run_dir": str(run_dir), "duration_s": elapsed,
                "runtime_ok": False, "runtime_status": "failed",
                "error": f"fixture process exited {process.exitcode} without a result",
                "failed_stage": _infer_wedged_stage(run_dir)}
    result.setdefault("run_dir", str(run_dir))
    return result


def _run_fixture(image: Path, run_dir: Path, cfg: dict, resume: str, timeout_s: float) -> dict:
    """Run one benchmark fixture with the per-fixture watchdog applied."""
    return run_bounded(_fixture_worker, (str(image), str(run_dir), cfg, resume),
                       run_dir, timeout_s, cfg=cfg)

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
    # A fully-baked plate reports native_text_ratio 1.0 vacuously (0 native / 0 emitted text),
    # which would otherwise clear the native-text bar. A run that QA hard-failed for near-total
    # rasterization (009: no-editable-content / low-editable-ratio / low-native-leaf-ratio) is
    # never contract-correct, whatever the vacuous ratio says.
    _bake_rules = {"no-editable-content", "low-editable-ratio", "low-native-leaf-ratio"}
    _bake_fail = next((str(item.get("rule")) for item in (row.get("hard_fails") or [])
                       if isinstance(item, dict) and item.get("rule") in _bake_rules), None)
    if _bake_fail:
        passed = False
        reasons.append(f"near-total rasterization ({_bake_fail})")
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
    # Per-fixture watchdog (run_bounded): a killed/crashed fixture has no qa.json to carry
    # its failure, so surface it here as a hard fail -- the stage name + timeout/error
    # reason then rides the existing hard_fails column instead of vanishing silently.
    if result.get("error") and not result.get("ok"):
        rule = "fixture-timeout" if result.get("timed_out") else "fixture-error"
        key = (rule, result.get("error"))
        if key not in seen_failures:
            merged_hard_fails.append({
                "rule": rule, "detail": result.get("error"), "hard": True,
                "stage": result.get("failed_stage"),
            })
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
    # Figma export is a distinct, human-gated step (the plugin must be launched in Figma
    # desktop). Track it as its own pending state so an unattended benchmark neither counts
    # a missing export as failure nor silently treats the run as fully export-verified.
    figma_export_present = (run_dir / "figma_export.png").is_file()
    figma_export_status = "exported" if figma_export_present else "awaiting-manual-import"
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
        "figma_export_present": figma_export_present,
        "figma_export_status": figma_export_status,
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
        "aspect_class": (
            (qa.get("aspect_class") or meta.get("aspect_class")
             or (_read(run_dir / "format.json", {}) or {}).get("aspect_class")
             or (_read(run_dir / "archetype.json", {}) or {}).get("format", {}).get("aspect_class"))
        ),
        "enabled_capabilities": (
            qa.get("enabled_capabilities")
            or meta.get("enabled_capabilities")
            or (_read(run_dir / "format.json", {}) or {}).get("enabled_capabilities")
            or (_read(run_dir / "archetype.json", {}) or {}).get("format", {}).get("enabled_capabilities")
        ),
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
        "failure_reason": result.get("error"),
        "failed_stage": result.get("failed_stage"),
        "timed_out": bool(result.get("timed_out")),
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
    scripts_dir = Path(__file__).resolve().parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import report_html
        path = report_html.generate_report(output)
        print(f"HTML report: {path}")
    except Exception as exc:  # pragma: no cover - reporting must not break a benchmark
        print(f"report.html generation skipped: {exc}")


def _point_activity_grid_at(output: Path) -> Path:
    """Point scripts/activity_grid.py's dashboard at this run (it follows ``.activity_current``).

    Written unconditionally, before the per-fixture loop starts (not after it finishes),
    so a RESUMED run (re-invoked against the same/an existing --output) re-asserts the
    same pointer, and a PARTIAL/ABORTED run (killed or crashed mid-loop) still leaves the
    pointer correctly naming this run dir -- it was written before anything could go wrong.
    """
    pointer = output.resolve().parent / ".activity_current"
    try:
        pointer.write_text(str(output.resolve()), encoding="utf-8")
        print(f"Activity grid pointer -> {pointer} ({output.name})", flush=True)
    except OSError as exc:
        print(f"Activity grid pointer skipped: {exc}", flush=True)
    return pointer


def _build_report(source_dir: Path, output: Path, images: list[Path], requested_ids: list[str],
                  runs: list[dict], *, wall_s: float, partial: bool,
                  aborted_reason: str | None = None) -> dict:
    """Assemble the benchmark.json/.md payload from whatever ``runs`` has so far.

    Called after every fixture (not just once at the end) so a run that dies mid-loop --
    killed fixture, Ctrl-C, an unexpected exception -- still leaves a summary behind
    instead of nothing at all (postfix-benchmark-6 produced no benchmark.json/.md because
    the only write was after the full loop completed).
    """
    return {
        "version": 1,
        "input_dir": str(source_dir.resolve()),
        "output": str(output.resolve()),
        "fixture_manifest": _source_manifest(images, requested_ids),
        "runs": runs,
        # Partial/abort bookkeeping: additive fields, so a fully-completed run's report
        # is unchanged in shape from before other than these new keys.
        "partial": partial,
        "aborted_reason": aborted_reason,
        "wall_time_s": wall_s,
        "fixtures_planned": len(images),
        "fixtures_completed": len(runs),
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
            "figma_exports_present": sum(1 for row in runs if row.get("figma_export_present")),
            "figma_exports_awaiting_manual_import": sum(
                1 for row in runs if not row.get("figma_export_present")
            ),
            "mean_element_recall": _mean(runs, "element_recall"),
            "runs_with_hard_fails": sum(1 for row in runs if row["hard_fails"]),
            "auto_fixed_runs": sum(1 for row in runs if row.get("auto_fixed")),
            "harness_rounds_total": sum(row.get("harness_rounds", 0) for row in runs),
            "final_qa_passing": sum(1 for row in runs if row.get("final_qa_ok")),
            "timed_out_runs": sum(1 for row in runs if row.get("timed_out")),
        },
    }


def _run_text_contract_check(run_dir: Path) -> dict | None:
    """One fixture's text-contract sweep (scripts/text_contract_check.check_run).

    Returns None for a fixture that never reached design.json (for example, one the
    watchdog killed before the design stage) -- there is nothing to check yet, the same
    way scripts/text_contract_check.py's own CLI skips run dirs without design.json.
    """
    if not (Path(run_dir) / "design.json").is_file():
        return None
    from scripts.text_contract_check import check_run
    try:
        return check_run(str(run_dir))
    except Exception as exc:  # never let the contract sweep take down the benchmark
        return {"fixture": Path(run_dir).name, "nodes": 0, "source_lines": 0,
                "violations": [{"severity": "ERROR", "rule": "contract-check-crashed",
                                "detail": str(exc), "text": ""}]}


def _build_contract_report(contract_reports: list[dict]) -> dict:
    """Aggregate scripts/text_contract_check.py's per-fixture reports (see main()'s
    per-image loop, which appends one entry per fixture as soon as it reaches design.json)."""
    hard = sum(1 for r in contract_reports for v in r.get("violations", [])
              if v.get("severity") in ("HARD", "ERROR"))
    warn = sum(1 for r in contract_reports for v in r.get("violations", [])
              if v.get("severity") == "WARN")
    return {
        "version": 1,
        "checked": len(contract_reports),
        "hard_total": hard,
        "warn_total": warn,
        "runs": contract_reports,
    }


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
    parser.add_argument("--fixture-timeout", type=float, default=DEFAULT_FIXTURE_TIMEOUT_S,
                        help="max seconds a single fixture may run before its stuck stage is "
                        "killed and it is recorded as a timeout failure, so one wedged "
                        "fixture (a stuck VLM/HTTP call) cannot take down the whole "
                        f"benchmark (default: {DEFAULT_FIXTURE_TIMEOUT_S:.0f}s)")
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

    # Resolve the image roster BEFORE smoke so activity grids can pre-seed all dots.
    try:
        requested_ids = parse_fixture_ids(args.ids)
        images = select_images(source_dir, args.max_images, requested_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not images:
        raise SystemExit(f"No images found in {source_dir}")
    # Format readiness metadata for activity-grid / batch slicing: aspect class +
    # optional tags from <input_dir>/format_index.json (not new named presets).
    format_index = format_readiness.resolve_format_index(source_dir)
    planned_images = [
        format_readiness.planned_image_entry(
            path,
            fixture_id=_file_fixture_id(path),
            format_index=format_index,
        )
        for path in images
    ]
    aspect_counts: dict[str, int] = {}
    for row in planned_images:
        key = str(row.get("aspect_class") or "unknown")
        aspect_counts[key] = aspect_counts.get(key, 0) + 1
    planned = {
        "version": 2,
        "input_dir": str(source_dir.resolve()),
        "output": str(output.resolve()),
        "format_index_entries": len(format_index),
        "aspect_class_counts": aspect_counts,
        "images": planned_images,
    }
    (output / "planned.json").write_text(json.dumps(planned, indent=2), encoding="utf-8")
    for path in images:
        (output / path.stem).mkdir(parents=True, exist_ok=True)
    # Point activity grid at this bench (scripts/activity_grid.py follows .activity_current).
    _point_activity_grid_at(output)
    print(f"Planned {len(images)} ads -> {output / 'planned.json'}", flush=True)

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

    runs: list[dict] = []
    contract_reports: list[dict] = []
    aborted_reason: str | None = None
    bench_t0 = time.time()

    def _flush_reports(*, partial: bool) -> dict:
        # Written after EVERY fixture (not only once at the end) so a run killed from the
        # outside (task manager, power loss) -- not just a clean Python exception -- still
        # leaves the most current summary + contract-check evidence on disk.  See (b).
        rep = _build_report(source_dir, output, images, requested_ids, runs,
                            wall_s=round(time.time() - bench_t0, 3), partial=partial,
                            aborted_reason=aborted_reason)
        (output / "benchmark.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        (output / "benchmark.md").write_text(_markdown(rep), encoding="utf-8")
        (output / "text_contract_report.json").write_text(
            json.dumps(_build_contract_report(contract_reports), indent=2), encoding="utf-8")
        return rep

    try:
        for image in images:
            run_dir = output / image.stem
            print(f"\n=== {image.name} ===", flush=True)
            result = _run_fixture(image, run_dir, cfg, args.resume, args.fixture_timeout)
            if result.get("timed_out"):
                print(f"  !! fixture watchdog: exceeded {args.fixture_timeout:.0f}s in stage "
                      f"'{result.get('failed_stage')}' -- killed, recorded as a failure, "
                      "continuing to the next fixture", flush=True)
            runs.append(_entry(run_dir, result))
            contract_row = _run_text_contract_check(run_dir)
            if contract_row is not None:
                contract_reports.append(contract_row)
            _flush_reports(partial=True)
    except BaseException as exc:
        aborted_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = _flush_reports(partial=(len(runs) < len(images)))
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
