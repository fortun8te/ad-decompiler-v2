#!/usr/bin/env python3
"""Cached, evidence-producing runtime smoke test for the Windows RTX worker.

doctor.py proves that dependencies appear ready. This command goes further: it runs CUDA,
Gemma vision, OCR, SAM 3, the inpaint backend selected by config, and VTracer, then runs one
synthetic image through the integrated pipeline. Evidence is kept under runs/rtx-self-test and
can be checked cheaply on every bridge start without loading any model again.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


VERSION = 2
DEFAULT_OUTPUT = Path("runs/rtx-self-test")
CACHE_MAX_AGE_S = 7 * 24 * 60 * 60


def _read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def fingerprint(config_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"rtx-self-test-v{VERSION}\n".encode())
    digest.update(config_path.resolve().as_posix().encode())
    if config_path.is_file():
        digest.update(config_path.read_bytes())
    # Changes to these execution boundaries invalidate old runtime proof.
    root = Path(__file__).resolve().parent
    for name in ("rtx_self_test.py", "runtime_smoke.py", "run_pipeline.py", "doctor.py",
                 "src/ocr.py", "src/sam3_detect.py",
                 "src/inpaint.py", "src/inpaint_quality.py", "src/vectorize.py", "src/vlm_client.py"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def cache_status(output: Path, config_path: Path, now: float | None = None) -> dict:
    latest = _read_json(output / "latest.json", {}) or {}
    expected = fingerprint(config_path)
    age_s = max(0, int((now or time.time()) - float(latest.get("finished_at") or 0)))
    valid = bool(
        latest.get("ok")
        and latest.get("fingerprint") == expected
        and age_s <= CACHE_MAX_AGE_S
        and Path(str(latest.get("evidence_path") or "")).is_file()
    )
    reason = "passed" if valid else (
        "never_run" if not latest else
        "config_or_code_changed" if latest.get("fingerprint") != expected else
        "expired" if age_s > CACHE_MAX_AGE_S else
        "last_run_failed"
    )
    return {"valid": valid, "reason": reason, "fingerprint": expected,
            "age_s": age_s if latest else None, "latest": latest or None}


def _synthetic_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 480), "#f3eadb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((42, 34, 598, 446), radius=30, fill="#fffaf0", outline="#181818", width=4)
    draw.ellipse((68, 86, 238, 256), fill="#f06435", outline="#181818", width=4)
    draw.rectangle((392, 86, 548, 268), fill="#4c72b0", outline="#181818", width=4)
    draw.polygon([(436, 126), (500, 126), (522, 226), (414, 226)], fill="#f7d154")
    draw.rounded_rectangle((78, 354, 344, 420), radius=18, fill="#181818")
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    try:
        title = ImageFont.truetype(str(font_path), 46)
        body = ImageFont.truetype(str(font_path), 28)
    except OSError:
        title = body = ImageFont.load_default()
    draw.text((268, 82), "RTX SMOKE", font=title, fill="#181818")
    draw.text((270, 148), "CODE 7429", font=body, fill="#181818")
    draw.text((108, 370), "TEST MODEL", font=body, fill="#ffffff")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _check(name: str, ok: bool, detail: str, **evidence) -> dict:
    row = {"name": name, "ok": bool(ok), "detail": str(detail)}
    if evidence:
        row["evidence"] = evidence
    return row


def evaluate_pipeline(run_dir: Path, result: dict, cfg: dict) -> list[dict]:
    raw = _read_json(run_dir / "ocr_raw.json", {}) or {}
    sam = _read_json(run_dir / "sam3.json", {}) or {}
    reconstruction = _read_json(run_dir / "reconstruction.json", {}) or {}
    runtime = _read_json(run_dir / "runtime_report.json", {}) or {}
    primary = str((cfg.get("ocr") or {}).get("primary", "doctr"))
    successful = ((raw.get("metrics") or {}).get("cross_check") or {}).get("successful_engines") or []
    ocr_text = " ".join(str(line.get("text", "")) for line in raw.get("lines") or [])
    ocr_ok = primary in successful and bool(raw.get("lines"))
    diag = sam.get("diagnostics") or {}
    sam_ok = (sam.get("engine") != "residual-fallback" and int(diag.get("model_elements") or 0) > 0
              and int(diag.get("text_prompts_succeeded") or 0) > 0)
    inpaint_stats = (reconstruction.get("stats") or {}).get("inpaint") or {}
    qwen_required = bool((cfg.get("qwen") or {}).get("required", False))
    qwen_stage = next((row for row in runtime.get("stages") or [] if row.get("name") == "qwen"), {})
    checks = [
        _check("pipeline_completed", bool(result.get("ok")) and runtime.get("status") != "failed",
               str(result.get("error") or runtime.get("status") or "unknown"), run_dir=str(run_dir)),
        _check("ocr_runtime", ocr_ok, f"primary={primary}; successful={successful}; text={ocr_text[:160]}",
               primary=primary, successful_engines=successful, text=ocr_text[:500]),
        _check("sam3_runtime", sam_ok,
               f"engine={sam.get('engine')}; model elements={diag.get('model_elements')}",
               engine=sam.get("engine"), diagnostics=diag),
    ]
    if qwen_required:
        checks.append(_check("qwen_runtime", qwen_stage.get("status") == "ok",
                             f"stage={qwen_stage.get('status')}", stage=qwen_stage))
    checks.append(_check("pipeline_inpaint_evidence", bool(inpaint_stats),
                         f"backend={inpaint_stats.get('backend')}", stats=inpaint_stats))
    return checks


def run(config_path: Path, output: Path, force: bool = False) -> dict:
    fp = fingerprint(config_path)
    cached = cache_status(output, config_path)
    if cached["valid"] and not force:
        return {**cached["latest"], "ok": True, "cached": True}
    from doctor import inspect, load_cfg
    root = Path(__file__).resolve().parent
    cfg = load_cfg(str(config_path))
    run_dir = (output / fp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = int(time.time())
    preflight = inspect(cfg, root)
    _atomic_json(run_dir / "dependency_preflight.json", preflight)
    if not preflight.get("ok"):
        checks = [_check("dependency_readiness", False, "doctor has blockers",
                         blockers=preflight.get("blockers") or [])]
    else:
        image_path = run_dir / "self-test-input.png"
        _synthetic_image(image_path)
        checks = [_check("dependency_readiness", True, "doctor passed")]
        # runtime_smoke owns the bounded per-model probes. Keep this wrapper focused on
        # cache policy plus the integrated end-to-end run so doctor, benchmark and the
        # launcher cannot drift into three different definitions of "model executed".
        from runtime_smoke import run_all
        smoke = run_all(cfg, run_dir / "runtime-smoke", timeout_s=120)
        checks.extend(smoke.get("checks") or [])
        pipeline_cfg = copy.deepcopy(cfg)
        pipeline_cfg.setdefault("runtime", {})["auto_repair"] = False
        pipeline_cfg["runtime"].setdefault("harness", {})["enabled"] = False
        pipeline_cfg.setdefault("figma", {})["enabled"] = False
        # Gemma was already exercised above with a deterministic vision read. Avoid dozens
        # of optional judging calls in what should remain a compact runtime smoke.
        pipeline_cfg.setdefault("vlm", {})["enabled"] = False
        for key in ("segment_filter", "font_judge", "scene_text", "ocr_judge", "element_propose"):
            pipeline_cfg["vlm"].setdefault(key, {})["enabled"] = False
        try:
            from run_pipeline import run_one
            pipeline_result = run_one(str(image_path), str(run_dir), pipeline_cfg)
            checks.extend(evaluate_pipeline(run_dir, pipeline_result, cfg))
        except Exception as exc:
            checks.append(_check("pipeline_completed", False, str(exc)))
    finished = int(time.time())
    report = {
        "version": VERSION,
        "ok": bool(checks) and all(row.get("ok") for row in checks),
        "cached": False,
        "fingerprint": fp,
        "dependency_ready": bool(preflight.get("ok")),
        "runtime_smoke_passed": bool(preflight.get("ok")) and len(checks) > 1
        and all(row.get("ok") for row in checks[1:]),
        "started_at": started,
        "finished_at": finished,
        "duration_s": finished - started,
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir),
        "checks": checks,
    }
    evidence_path = run_dir / "self_test.json"
    report["evidence_path"] = str(evidence_path)
    _atomic_json(evidence_path, report)
    _atomic_json(output / "latest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or inspect the cached RTX model smoke test")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status-json", action="store_true", help="cheap cache check; never loads a model")
    args = parser.parse_args()
    config_path = Path(args.config)
    output = Path(args.output)
    if args.status_json:
        print(json.dumps(cache_status(output, config_path)))
        return 0
    report = run(config_path, output, force=args.force)
    print("RTX SELF-TEST PASSED" if report.get("ok") else "RTX SELF-TEST FAILED")
    print(f"Evidence: {report.get('evidence_path')}")
    for item in report.get("checks") or []:
        print(f"{'OK' if item.get('ok') else 'FAIL':4} {item.get('name')}: {item.get('detail')}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
