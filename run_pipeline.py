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
import argparse, os, sys, time, glob, traceback, json

sys.path.insert(0, os.path.dirname(__file__))
from src import (normalize, ocr, element_detect, qwen_worker, merge_layers,
                 build_design_json, figma_import, pixel_diff, repair, render_preview)
from src.schema import dump, load

STAGES = ["normalize", "ocr", "elements", "qwen", "merge", "design",
          "preview", "figma", "export", "diff", "qa"]


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
    print(line, flush=True)
    with open(os.path.join(run_dir, "pipeline.log"), "a") as f:
        f.write(line + "\n")


def run_one(input_path, run_dir, cfg, start_from="normalize"):
    os.makedirs(run_dir, exist_ok=True)
    A = lambda n: os.path.join(run_dir, n)          # artifact path
    exists = lambda n: os.path.exists(A(n))
    begin = STAGES.index(start_from) if start_from in STAGES else 0
    canvas = None
    t0 = time.time()

    def stage(name):
        return STAGES.index(name) >= begin

    try:
        # 1 normalize
        if stage("normalize") or not exists("normalized.png"):
            norm_path, canvas = normalize.load_normalize(input_path, run_dir)
            _log(run_dir, f"normalize → {canvas['w']}x{canvas['h']}")
        else:
            canvas = load(A("canvas.json"))
        dump({"w": canvas["w"], "h": canvas["h"]}, A("canvas.json"))
        norm_path = A("normalized.png")

        # 2 ocr
        if stage("ocr") or not exists("ocr.json"):
            ocr_res = ocr.run_ocr(norm_path, cfg); dump(ocr_res, A("ocr.json"))
            _log(run_dir, f"ocr[{ocr_res.get('engine')}] → {len(ocr_res.get('lines', []))} lines")
        ocr_res = load(A("ocr.json"))

        # 3 element detect
        if stage("elements") or not exists("elements.json"):
            els = element_detect.detect(norm_path, ocr_res, cfg)
            dump(els, A("elements.json")); _log(run_dir, f"elements → {len(els)}")
        els = load(A("elements.json"))

        # 4 qwen layers
        if stage("qwen") or not exists("qwen.json"):
            qwen = qwen_worker.propose_layers(norm_path, run_dir, cfg)
            dump(qwen, A("qwen.json")); _log(run_dir, f"qwen → {len(qwen)} layers")
        qwen = load(A("qwen.json"))

        # 6 merge + route
        if stage("merge") or not exists("merged.json"):
            merged = merge_layers.merge(ocr_res, els, qwen, canvas, cfg)
            dump(merged, A("merged.json")); _log(run_dir, f"merge → {len(merged)} candidates")
        merged = load(A("merged.json"))

        # 8 design.json (source of truth)
        if stage("design") or not exists("design.json"):
            base = A("normalized.png")
            doc = build_design_json.build(merged, canvas, run_dir, base_src=base,
                                          doc_id=os.path.basename(run_dir), name=os.path.basename(run_dir))
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
        if stage("diff") and qa_render:
            ren_ocr = ocr.run_ocr(qa_render, cfg) if cfg.get("qa_ocr", True) else None
            qa_partial = pixel_diff.compare(norm_path, qa_render, run_dir,
                                            source_ocr=ocr_res, render_ocr=ren_ocr)
            reps = repair.assess(load(A("design.json")), qa_partial, ocr_res, cfg)
            qa = {**qa_partial, "repairs": reps,
                  "ok": qa_partial.get("ssim", 0) >= 0.9 and not any(r.get("hard") for r in reps)}
            dump(qa, A("qa.json"))
            _log(run_dir, f"qa → ssim={qa.get('ssim')} text_recall={qa.get('text_recall')} repairs={len(reps)}")
        elif stage("diff"):
            _log(run_dir, "diff/qa skipped — no render found (preview.png or figma_export.png)")

        _log(run_dir, f"done in {time.time()-t0:.1f}s")
        return {"ok": True, "run_dir": run_dir}
    except Exception as e:
        _log(run_dir, f"ERROR: {e}\n{traceback.format_exc()}")
        return {"ok": False, "run_dir": run_dir, "error": str(e)}


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
