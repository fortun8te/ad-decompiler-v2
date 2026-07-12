#!/usr/bin/env python3
"""run_pipeline.py — headless orchestrator for the ad → editable Figma pipeline.

    python run_pipeline.py --input ad.png --output ./runs/run_001
    python run_pipeline.py --input ./ads/ --batch
    python run_pipeline.py --input ad.png --output ./runs/run_001 --resume qa   # re-run from a stage

Each stage writes its artifact into the run dir and is idempotent, so the OpenCode agent
(see AGENTS.md) can re-run any single stage, inspect the JSON, and trigger repairs without
restarting. Stages degrade on failure (write a note) rather than aborting the whole run.
"""
from __future__ import annotations
import argparse, copy, hashlib, os, sys, time, glob, traceback, json

sys.path.insert(0, os.path.dirname(__file__))
from src.console_io import configure_stdio, safe_print
from src import (normalize, ocr, text_analysis, element_detect, sam3_detect,
                 element_fusion, qwen_worker, merge_layers, reconstruct, layout,
                 build_design_json, figma_import, pixel_diff, repair, render_preview)
from src.run_report import RunReport, qwen_degradation
from src.schema import dump, load

configure_stdio()

STAGES = ["normalize", "ocr", "text", "residual", "qwen", "sam", "elements",
          "merge", "reconstruct", "layout", "design", "preview", "figma",
          "export", "diff", "qa"]


def load_cfg(path):
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            with open(path) as f:
                return json.load(f)
    return {}


def _log(run_dir, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    safe_print(line, flush=True)
    with open(os.path.join(run_dir, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _sam_with_safe_retry(image_path, residual, cfg, run_dir, report):
    """Retry only the known low-risk SAM compile failure mode once.

    Re-running a complete 848M segmentation pass after any arbitrary miss is costly and can
    overwrite useful evidence.  Disabling ``torch.compile`` is different: it is a deterministic
    compatibility retry for a model-load/graph-compile failure, and is recorded in the report.
    """
    result = sam3_detect.detect(image_path, residual=residual, cfg=cfg, run_dir=run_dir)
    sam_cfg = cfg.get("sam3") or {}
    policy = (cfg.get("runtime") or {}).get("auto_retry") or {}
    if result.get("status") not in ("fallback", "partial") or not sam_cfg.get("compile"):
        return result
    if not policy.get("sam_disable_compile", True):
        return result
    retry_cfg = copy.deepcopy(cfg)
    retry_cfg.setdefault("sam3", {})["compile"] = False
    # A cached compiled backend would otherwise be reused on the second attempt.
    cache = getattr(sam3_detect, "_BACKEND_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    report.retry("sam3", result.get("note") or result.get("status"), "retrying once with compile=false")
    retried = sam3_detect.detect(image_path, residual=residual, cfg=retry_cfg, run_dir=run_dir)
    if retried.get("status") == "ok":
        retried["retry"] = {"strategy": "compile-disabled", "first_status": result.get("status")}
        dump(retried, os.path.join(run_dir, "sam3.json"))
        report.retry("sam3", "compile=true attempt did not complete", "recovered with compile=false")
    else:
        report.retry("sam3", "compile=true attempt did not complete", f"retry remained {retried.get('status')}")
    return retried


def _model_health(raw_ocr, sam, run_dir, cfg, report):
    """Convert intentional fallback contracts into visible run evidence and QA failures."""
    status = str((raw_ocr or {}).get("status") or "ok")
    if status != "ok":
        errors = (raw_ocr or {}).get("errors") or []
        reason = "; ".join(str(item.get("error", item)) for item in errors[:2]) or status
        report.degraded("ocr", reason)
    sam_status = str((sam or {}).get("status") or "ok")
    if sam_status != "ok":
        report.degraded("sam3", str((sam or {}).get("note") or sam_status))
    qwen_cfg = cfg.get("qwen") or {}
    qwen_note = qwen_degradation(run_dir, bool(qwen_cfg.get("enabled", True)))
    if qwen_note:
        report.degraded("qwen", qwen_note)
    return report.violations


def _input_fingerprint(path, cfg):
    """Bind resumable artifacts to their exact source instead of a folder name."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    config_payload = json.dumps(cfg or {}, sort_keys=True, default=str).encode("utf-8")
    return {
        "source_path": os.path.abspath(path),
        "source_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
    }


def _write_json_atomic(path, data):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(temporary, path)


def run_one(input_path, run_dir, cfg, start_from="normalize"):
    os.makedirs(run_dir, exist_ok=True)
    cfg = copy.deepcopy(cfg or {})
    cfg["run_dir"] = os.path.abspath(run_dir)
    report = RunReport(run_dir, input_path, cfg, start_from)
    A = lambda n: os.path.join(run_dir, n)          # artifact path
    exists = lambda n: os.path.exists(A(n))
    begin = STAGES.index(start_from) if start_from in STAGES else 0
    canvas = None
    t0 = time.time()

    try:
        fingerprint = _input_fingerprint(input_path, cfg)
    except Exception as exc:
        error = f"cannot fingerprint input: {exc}"
        report.stage("input", "failed", detail=error)
        report.finish(error=error)
        return {"ok": False, "run_dir": run_dir, "error": error, "runtime_ok": False}
    manifest_path = A("input_manifest.json")
    previous = load(manifest_path) if os.path.exists(manifest_path) else None
    if previous and previous.get("source_sha256") != fingerprint["source_sha256"] and start_from != "normalize":
        message = ("input image changed since this run folder was created; rerun with "
                   "--resume normalize or choose a new output folder")
        _log(run_dir, f"ERROR: {message}")
        report.stage("input", "failed", detail=message)
        report.finish(error=message)
        return {"ok": False, "run_dir": run_dir, "error": message, "runtime_ok": False}
    if previous and previous.get("config_sha256") != fingerprint["config_sha256"] and start_from != "normalize":
        _log(run_dir, "config changed since prior artifacts; resuming from the requested stage")
    _write_json_atomic(manifest_path, fingerprint)

    def stage(name):
        return STAGES.index(name) >= begin

    current_stage = "normalize"
    try:
        # 1 normalize
        if stage("normalize") or not exists("normalized.png"):
            current_stage = "normalize"
            norm_path, canvas = normalize.load_normalize(input_path, run_dir, cfg)
            _log(run_dir, f"normalize → {canvas['w']}x{canvas['h']}")
        else:
            canvas = load(A("canvas.json"))
        dump({"w": canvas["w"], "h": canvas["h"]}, A("canvas.json"))
        norm_path = A("normalized.png")

        # 2 OCR facts, followed by painted-text/style/hierarchy analysis.
        if stage("ocr") or not exists("ocr_raw.json"):
            current_stage = "ocr"
            raw_ocr = ocr.run_ocr(norm_path, cfg, run_dir=run_dir)
            dump(raw_ocr, A("ocr_raw.json"))
            _log(run_dir, f"ocr[{raw_ocr.get('engine')}] → {len(raw_ocr.get('lines', []))} lines")
        raw_ocr = load(A("ocr_raw.json")) if exists("ocr_raw.json") else load(A("ocr.json"))
        if stage("text") or not exists("ocr.json"):
            current_stage = "text"
            ocr_res = text_analysis.analyze_text(norm_path, raw_ocr, cfg)
            dump(ocr_res, A("ocr.json"))
            _log(run_dir, f"text analysis → {len(ocr_res.get('blocks', []))} blocks, {len(ocr_res.get('styles', []))} styles")
        ocr_res = load(A("ocr.json"))

        # 3 deterministic residual proposals. This also writes box-local masks.
        if stage("residual") or not exists("residual.json"):
            current_stage = "residual"
            residual = element_detect.detect(norm_path, ocr_res, cfg, run_dir=run_dir)
            dump(residual, A("residual.json"))
            _log(run_dir, f"residual proposals → {len(residual)}")
        residual = load(A("residual.json"))

        # 4 optional Qwen layers are advisory observations/assets, never the scene graph.
        if stage("qwen") or not exists("qwen.json"):
            current_stage = "qwen"
            qwen = qwen_worker.propose_layers(norm_path, run_dir, cfg)
            dump(qwen, A("qwen.json")); _log(run_dir, f"qwen → {len(qwen)} layers")
        qwen = load(A("qwen.json"))

        # 5 SAM 3 image prompt sweep + box-refine every residual, then mask-aware fusion.
        if stage("sam") or not exists("sam3.json"):
            current_stage = "sam"
            sam = _sam_with_safe_retry(norm_path, residual, cfg, run_dir, report)
            _log(run_dir, f"sam3[{sam.get('status')}] → {len(sam.get('elements', []))} observations")
        sam = load(A("sam3.json"))
        report.stage("ocr", str(raw_ocr.get("status") or "ok"),
                     detail="; ".join(str(item.get("error", item)) for item in raw_ocr.get("errors", [])[:2]) or None,
                     artifacts=["ocr_raw.json"])
        qwen_note = qwen_degradation(run_dir, bool((cfg.get("qwen") or {}).get("enabled", True)))
        report.stage("qwen", "fallback" if qwen_note else "ok", detail=qwen_note, artifacts=["qwen.json"])
        report.stage("sam", str(sam.get("status") or "ok"), detail=sam.get("note"),
                     artifacts=["sam3.json", "sam3_masks"])
        runtime_violations = _model_health(raw_ocr, sam, run_dir, cfg, report)
        if stage("elements") or not exists("fused_elements.json"):
            els = element_fusion.fuse(sam3=sam, residual=residual, qwen=qwen,
                                      canvas=canvas, cfg=cfg, run_dir=run_dir)
            dump(els, A("elements.json"))
            _log(run_dir, f"element fusion → {len(els)} canonical elements")
        els = load(A("fused_elements.json")) if exists("fused_elements.json") else load(A("elements.json"))

        # 6 merge/routing creates semantic candidates; reconstruction gives pixels one owner.
        if stage("merge") or not exists("merged.json"):
            merged = merge_layers.merge(ocr_res, els, qwen, canvas, cfg, run_dir=run_dir)
            dump(merged, A("merged.json")); _log(run_dir, f"merge → {len(merged)} candidates")
        merged = load(A("merged.json"))

        if stage("reconstruct") or not exists("reconstruction.json"):
            reconstruction = reconstruct.reconstruct(norm_path, ocr_res, merged, run_dir, cfg)
            _log(run_dir, "reconstruct → "
                 f"{reconstruction['stats']['canonical_entities']} entities, "
                 f"background={reconstruction['stats']['inpaint']['backend']}")
        reconstruction = load(A("reconstruction.json"))
        inpaint_backend = str((reconstruction.get("stats") or {}).get("inpaint", {}).get("backend") or "")
        if inpaint_backend and inpaint_backend != "big-lama":
            report.stage("inpaint", "fallback", detail=f"backend={inpaint_backend}")
            report.degraded("inpaint", f"Big-LaMa unavailable; used {inpaint_backend} fallback for background plate")
        elif inpaint_backend:
            report.stage("inpaint", "ok", detail=f"backend={inpaint_backend}")

        if stage("layout") or not exists("layout.json"):
            tree = layout.infer(reconstruction.get("candidates", []), canvas, cfg)
            dump(tree, A("layout.json"))
            _log(run_dir, f"layout → {len(tree)} root layers")
        tree = load(A("layout.json"))

        # 8 schema-v2 Figma scene graph (source of truth)
        if stage("design") or not exists("design.json"):
            kept = [c.get("text") for c in reconstruction.get("candidates", [])
                    if c.get("target") == "drop" and c.get("text")]
            doc = build_design_json.build(
                tree, canvas, run_dir, base_src=A(reconstruction.get("background", "background_clean.png")),
                doc_id=os.path.basename(run_dir), name=os.path.basename(run_dir), kept_in_photo=kept,
            )
            _log(run_dir, f"design.json → {len(doc.layers)} layers, kept_in_photo={len(doc.kept_in_photo)}")

        # 8.5 LOCAL PREVIEW — see the layers without Figma (default on)
        if stage("preview") and cfg.get("preview", True):
            pv = render_preview.render(A("design.json"), run_dir)
            _log(run_dir, f"preview → {pv['preview']} ({pv['count']} layers in layers/, see layers_contact.png)")

        # 9 figma import (optional — Figma export can come later)
        if stage("figma") and cfg.get("figma", {}).get("enabled", False):
            imp = figma_import.import_design(A("design.json"), run_dir, cfg)
            dump(imp, A("figma_import.json")); _log(run_dir, f"figma import: {imp.get('action', imp)}")

        # 10 export screenshot (plugin writes it; may need the manual click)
        if stage("export"):
            exp = figma_import.export_screenshot(run_dir, cfg, wait_s=cfg.get("export_wait_s", 0))
            _log(run_dir, f"export: {exp.get('note', exp.get('path'))}")

        # 11 diff + 12 qa — QA against the Figma render if present, else the local preview
        qa_render = A("figma_export.png") if exists("figma_export.png") else \
            (A("preview.png") if exists("preview.png") else None)
        if (stage("diff") or stage("qa")) and qa_render:
            ren_ocr = ocr.run_ocr(qa_render, cfg, run_dir=run_dir) if cfg.get("qa_ocr", True) else None
            qa_partial = pixel_diff.compare(norm_path, qa_render, run_dir,
                                            source_ocr=ocr_res, render_ocr=ren_ocr)
            design_data = load(A("design.json"))
            structural_fails = []
            warnings = (design_data.get("meta") or {}).get("warnings") or []
            if any(warning.get("code") == "missing-asset" for warning in warnings):
                structural_fails.append({"rule": "missing-assets", "detail": "one or more image assets are unresolved"})
            roots = design_data.get("layers") or []
            if not roots or (roots[0].get("meta") or {}).get("source") != "inpaint":
                structural_fails.append({"rule": "unclean-background", "detail": "background is not the reconstructed plate"})
            if (design_data.get("meta") or {}).get("editable_ratio", 0) <= 0 and ocr_res.get("lines"):
                structural_fails.append({"rule": "no-editable-content", "detail": "source text exists but output has no editable nodes"})
            combined_fails = list(qa_partial.get("hard_fails") or [])
            # Required model fallback is an acceptance failure.  The full reason is persisted
            # in runtime_report.json even where a render is unavailable.
            combined_fails.extend(runtime_violations)
            seen_rules = {(item.get("rule"), item.get("detail")) for item in combined_fails}
            for item in structural_fails:
                key = (item.get("rule"), item.get("detail"))
                if key not in seen_rules:
                    combined_fails.append(item); seen_rules.add(key)
            qa_partial["hard_fails"] = combined_fails
            recall = qa_partial.get("text_recall")
            visual = qa_partial.get("visual_score", qa_partial.get("ssim", 0))
            qa_partial["composite"] = round(100 * (0.75 * visual +
                                                     0.25 * (recall if recall is not None else 1)), 2)
            reps = repair.assess(design_data, qa_partial, ocr_res, cfg)
            structural_report = dict(qa_partial.get("structural") or {})
            structural_report.setdefault("warnings", warnings)
            structural_report.setdefault(
                "duplicates_removed", reconstruction.get("stats", {}).get("duplicates_removed", 0)
            )
            qa = {**qa_partial, "repairs": reps,
                  "structural": structural_report,
                  "ok": qa_partial.get("ssim", 0) >= 0.9 and not combined_fails}
            dump(qa, A("qa.json"))
            report.stage("qa", "ok" if qa.get("ok") else "failed",
                         detail=((qa.get("hard_fails") or [{}])[0].get("detail")),
                         artifacts=["qa.json", "diff.png"])
            report.finish(qa_ok=bool(qa.get("ok")))
            _log(run_dir, f"qa → ssim={qa.get('ssim')} text_recall={qa.get('text_recall')} repairs={len(reps)}")
        elif stage("diff") or stage("qa"):
            report.stage("qa", "skipped", detail="no render found")
            report.finish(qa_ok=None)
            _log(run_dir, "diff/qa skipped — no render found (preview.png or figma_export.png)")

        elapsed = round(time.time() - t0, 3)
        _log(run_dir, f"done in {elapsed:.1f}s")
        if report.data.get("status") == "running":
            report.finish(qa_ok=None)
        return {"ok": True, "run_dir": run_dir, "duration_s": elapsed,
                "runtime_ok": report.acceptable, "runtime_status": report.data.get("status")}
    except Exception as e:
        _log(run_dir, f"ERROR: {e}\n{traceback.format_exc()}")
        report.stage("pipeline", "failed", detail=str(e))
        report.finish(error=str(e))
        return {"ok": False, "run_dir": run_dir, "error": str(e), "failed_stage": current_stage,
                "duration_s": round(time.time() - t0, 3), "runtime_ok": False,
                "runtime_status": "failed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or a directory (with --batch)")
    ap.add_argument("--output", help="run dir (single mode); default runs/<name>")
    ap.add_argument("--batch", action="store_true", help="treat --input as a dir of ads")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", default="normalize", choices=STAGES,
                    help="re-run starting from this stage (uses existing artifacts before it)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.batch:
        imgs = sorted([p for p in glob.glob(os.path.join(args.input, "*"))
                       if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
        report = []
        for p in imgs:
            rd = os.path.join("runs", os.path.splitext(os.path.basename(p))[0])
            report.append(run_one(p, rd, cfg, args.resume))
        print(json.dumps({"batch": len(imgs),
                          "ok": sum(1 for r in report if r["ok"])}, indent=2))
    else:
        rd = args.output or os.path.join("runs", os.path.splitext(os.path.basename(args.input))[0])
        r = run_one(args.input, rd, cfg, args.resume)
        sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
