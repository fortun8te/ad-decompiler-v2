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
                 build_design_json, figma_import, pixel_diff, repair, render_preview,
                 vlm_proofread, vlm_ocr_judge, vlm_font_judge, vlm_scene_text,
                 vlm_segment_filter, vlm_element_propose, vram)
from src import archetype
from src.run_report import RunReport, qwen_degradation
from src.schema import dump, load, validate_design
from src.harness import harness_enabled, harness_max_rounds, recommended_resume
from src.harness_loop import in_harness_loop, run_harness_after_pipeline
from src.qa_config import pixel_diff_thresholds, visual_pass_ssim

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


def _artifact_ready(path: str) -> bool:
    """Return false for missing, corrupt, or structurally unusable checkpoints."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
        lower = path.lower()
        if lower.endswith(".json"):
            value = load(path)
            name = os.path.basename(path)
            if value is None:
                return False
            if name == "canvas.json":
                return isinstance(value, dict) and all(
                    isinstance(value.get(key), (int, float)) and value[key] > 0
                    for key in ("w", "h")
                )
            if name in ("ocr_raw.json", "ocr.json"):
                return isinstance(value, dict) and isinstance(value.get("lines", []), list)
            if name in ("residual.json", "qwen.json", "fused_elements.json",
                        "elements.json", "merged.json", "layout.json"):
                return isinstance(value, list)
            if name == "sam3.json":
                return isinstance(value, dict) and isinstance(value.get("elements", []), list)
            if name == "reconstruction.json":
                if not (isinstance(value, dict) and isinstance(value.get("candidates"), list)
                        and isinstance(value.get("stats"), dict)):
                    return False
                for key in ("background", "removal_mask", "ownership"):
                    asset = value.get(key)
                    if not asset or not _artifact_ready(os.path.join(os.path.dirname(path), asset)):
                        return False
                return True
            if name == "design.json":
                return not validate_design(value)
            if name == "qa.json":
                return isinstance(value, dict) and isinstance(value.get("hard_fails", []), list)
            return True
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")):
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            return True
        return True
    except (OSError, ValueError, TypeError, SyntaxError, json.JSONDecodeError):
        return False


def _artifact_at_least_as_fresh(path: str, reference: str) -> bool:
    """True only when a render was produced for the current design checkpoint."""
    if not (_artifact_ready(path) and _artifact_ready(reference)):
        return False
    try:
        return os.path.getmtime(path) >= os.path.getmtime(reference)
    except OSError:
        return False


def run_one(input_path, run_dir, cfg, start_from="normalize"):
    os.makedirs(run_dir, exist_ok=True)
    cfg = copy.deepcopy(cfg or {})
    cfg["run_dir"] = os.path.abspath(run_dir)
    report = RunReport(run_dir, input_path, cfg, start_from)
    A = lambda n: os.path.join(run_dir, n)          # artifact path
    exists = lambda n: _artifact_ready(A(n))
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
    dirty = False
    design_updated = False
    try:
        # 1 normalize
        if stage("normalize") or dirty or not exists("normalized.png"):
            current_stage = "normalize"
            norm_path, canvas = normalize.load_normalize(input_path, run_dir, cfg)
            dirty = True
            _log(run_dir, f"normalize → {canvas['w']}x{canvas['h']}")
        else:
            canvas = load(A("canvas.json"))
        dump({"w": canvas["w"], "h": canvas["h"]}, A("canvas.json"))
        norm_path = A("normalized.png")

        # Cheap initial scene contract. OCR/VLM observations refine this below before
        # text analysis and decomposition consume the preset.
        configured_archetype = str((cfg.get("archetype") or {}).get("preset", "auto"))

        # 2 OCR facts, followed by painted-text/style/hierarchy analysis.
        if stage("ocr") or dirty or not exists("ocr_raw.json"):
            current_stage = "ocr"
            dirty = True
            raw_ocr = ocr.run_ocr(norm_path, cfg, run_dir=run_dir)
            ocr_judge = ((cfg.get("vlm") or {}).get("ocr_judge") or {})
            if ocr_judge.get("enabled"):
                vram.stage_boundary("ocr", "vlm-ocr-judge", cfg, run_dir,
                                    log_fn=lambda msg: _log(run_dir, msg))
                raw_ocr = vlm_ocr_judge.judge_ocr_lines(norm_path, raw_ocr, cfg)
            if (cfg.get("vlm") or {}).get("enabled"):
                vram.stage_boundary("ocr", "vlm-proofread", cfg, run_dir,
                                    log_fn=lambda msg: _log(run_dir, msg))
            raw_ocr = vlm_proofread.proofread_lines(norm_path, raw_ocr, cfg)
            dump(raw_ocr, A("ocr_raw.json"))
            oj = raw_ocr.get("vlm_ocr_judge")
            oj_note = ""
            if oj:
                oj_note = f", vlm-judged {oj['lines_corrected']}/{oj['lines_checked']}"
                if oj.get("ocr_read_added"):
                    oj_note += f", vlm-read +{oj['ocr_read_added']}"
            vp = raw_ocr.get("vlm_proofread")
            vp_note = ""
            if vp:
                vp_note = f", vlm-corrected {vp['lines_corrected']}/{vp['lines_checked']}"
                if vp.get("ensemble_disagreement_checked"):
                    vp_note += f", ensemble-proofread {vp['ensemble_disagreement_checked']}"
            _log(run_dir, f"ocr[{raw_ocr.get('engine')}] → {len(raw_ocr.get('lines', []))} lines{oj_note}{vp_note}")
        raw_ocr = load(A("ocr_raw.json")) if exists("ocr_raw.json") else load(A("ocr.json"))
        # Persist a filename-independent scene decision and make its preset visible to
        # every remaining stage. Optional semantic observations can come from an
        # upstream/VLM caller; OCR and canvas facts are always available locally.
        archetype_cfg = cfg.get("archetype") or {}
        facts = archetype.scene_facts(
            canvas, raw_ocr, observations=archetype_cfg.get("observations") or {},
        )
        facts.update(archetype.image_facts(norm_path))
        scene_decision = archetype.classify(facts, configured=configured_archetype)
        scene_decision["facts"] = facts
        dump(scene_decision, A("archetype.json"))
        cfg = archetype.apply_preset(cfg, scene_decision)
        _log(run_dir, f"archetype → {scene_decision['archetype']}")
        if stage("text") or dirty or not exists("ocr.json"):
            current_stage = "text"
            dirty = True
            ocr_res = text_analysis.analyze_text(norm_path, raw_ocr, cfg)
            scene_text = ((cfg.get("vlm") or {}).get("scene_text") or {})
            if scene_text.get("enabled"):
                vram.stage_boundary("text", "vlm-scene-text", cfg, run_dir,
                                    log_fn=lambda msg: _log(run_dir, msg))
                ocr_res = vlm_scene_text.classify_scene_text(norm_path, ocr_res, cfg)
            font_judge = ((cfg.get("vlm") or {}).get("font_judge") or {})
            if vlm_font_judge.should_judge_fonts(ocr_res, cfg):
                vram.stage_boundary("text", "vlm-font-judge", cfg, run_dir,
                                    log_fn=lambda msg: _log(run_dir, msg))
                ocr_res = vlm_font_judge.judge_fonts(norm_path, ocr_res, cfg)
            dump(ocr_res, A("ocr.json"))
            st = ocr_res.get("vlm_scene_text")
            st_note = ""
            if st:
                st_note = (f", vlm-scene-text {st.get('lines_classified', 0)}"
                           f"/{st.get('lines_checked', 0)}")
            fj = ocr_res.get("vlm_font_judge")
            fj_note = ""
            if fj:
                fj_note = f", vlm-font-judge promoted {fj.get('styles_promoted', 0)}/{fj.get('styles_checked', 0)}"
            _log(run_dir, f"text analysis → {len(ocr_res.get('blocks', []))} blocks, "
                 f"{len(ocr_res.get('styles', []))} styles{st_note}{fj_note}")
        ocr_res = load(A("ocr.json"))

        # 3 deterministic residual proposals. This also writes box-local masks.
        if stage("residual") or dirty or not exists("residual.json"):
            current_stage = "residual"
            dirty = True
            residual = element_detect.detect(norm_path, ocr_res, cfg, run_dir=run_dir)
            dump(residual, A("residual.json"))
            _log(run_dir, f"residual proposals → {len(residual)}")
        residual = load(A("residual.json"))

        ep_cfg = ((cfg.get("vlm") or {}).get("element_propose") or {})
        sam_element_count = None
        if exists("sam3.json"):
            prev_sam = load(A("sam3.json")) or {}
            sam_element_count = len(prev_sam.get("elements") or [])
        lightweight_due_to_sam = (
            sam_element_count is not None
            and ep_cfg.get("lightweight_grid_below_sam_count") is not None
            and sam_element_count < int(ep_cfg.get("lightweight_grid_below_sam_count"))
        )
        should_enrich = ep_cfg.get("enabled") and (
            dirty
            or
            stage("sam")
            or not exists("sam3.json")
            or ep_cfg.get("lightweight_grid")
            or lightweight_due_to_sam
        )
        if should_enrich:
            vram.stage_boundary("residual", "vlm-element-propose", cfg, run_dir,
                                log_fn=lambda msg: _log(run_dir, msg))
            before = len(residual)
            residual = vlm_element_propose.enrich_residual(
                norm_path, residual, cfg, sam_element_count=sam_element_count
            )
            added = len(residual) - before
            if added:
                dump(residual, A("residual.json"))
            _log(run_dir, f"vlm element propose → +{added} proposals ({len(residual)} total)")

        # 4 optional Qwen layers are advisory observations/assets, never the scene graph.
        if stage("qwen") or dirty or not exists("qwen.json"):
            current_stage = "qwen"
            dirty = True
            qwen = qwen_worker.propose_layers(norm_path, run_dir, cfg)
            dump(qwen, A("qwen.json")); _log(run_dir, f"qwen → {len(qwen)} layers")
        qwen = load(A("qwen.json"))

        # 5 SAM 3 image prompt sweep + box-refine every residual, then mask-aware fusion.
        if stage("sam") or dirty or not exists("sam3.json"):
            current_stage = "sam"
            dirty = True
            vram.stage_boundary("qwen", "sam", cfg, run_dir, log_fn=lambda msg: _log(run_dir, msg))
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
        if stage("elements") or dirty or not exists("elements.json"):
            current_stage = "elements"
            dirty = True
            fused = element_fusion.fuse(sam3=sam, residual=residual, qwen=qwen,
                                        canvas=canvas, cfg=cfg, run_dir=run_dir)
            dump(fused, A("fused_elements.json"))
            seg_filter = ((cfg.get("vlm") or {}).get("segment_filter") or {})
            if seg_filter.get("enabled"):
                vram.stage_boundary("fusion", "vlm-segment-filter", cfg, run_dir,
                                    log_fn=lambda msg: _log(run_dir, msg))
            els = vlm_segment_filter.filter_elements(norm_path, fused, cfg)
            dump(els, A("elements.json"))
            _log(run_dir, f"element fusion → {len(fused)} canonical → {len(els)} after filter")
        els = load(A("elements.json")) if exists("elements.json") else load(A("fused_elements.json"))

        # 6 merge/routing creates semantic candidates; reconstruction gives pixels one owner.
        if stage("merge") or dirty or not exists("merged.json"):
            current_stage = "merge"
            dirty = True
            merged = merge_layers.merge(ocr_res, els, qwen, canvas, cfg, run_dir=run_dir)
            dump(merged, A("merged.json")); _log(run_dir, f"merge → {len(merged)} candidates")
        merged = load(A("merged.json"))

        if stage("reconstruct") or dirty or not exists("reconstruction.json"):
            current_stage = "reconstruct"
            dirty = True
            vram.stage_boundary("merge", "reconstruct", cfg, run_dir, log_fn=lambda msg: _log(run_dir, msg))
            reconstruction = reconstruct.reconstruct(norm_path, ocr_res, merged, run_dir, cfg)
            _log(run_dir, "reconstruct → "
                 f"{reconstruction['stats']['canonical_entities']} entities, "
                 f"background={reconstruction['stats']['inpaint']['backend']}")
        reconstruction = load(A("reconstruction.json"))
        inpaint_stats = (reconstruction.get("stats") or {}).get("inpaint", {}) or {}
        inpaint_backend = str(inpaint_stats.get("backend") or "")
        inpaint_class = str(inpaint_stats.get("backend_class") or "").lower()
        backend_counts = inpaint_stats.get("backend_counts") or {}
        inpaint_diag = inpaint_stats.get("diagnostics") or {}
        comfy_note = ""
        if inpaint_diag.get("comfyui_healthy") is True:
            comfy_note = "; comfyui=ok"
        elif inpaint_diag.get("comfyui_healthy") is False:
            comfy_note = "; comfyui=down"
        fallback_backend = (
            inpaint_class in {"fallback", "degraded", "unavailable"}
            or inpaint_backend.lower().startswith("opencv")
            or any(str(name).lower().startswith("opencv") for name in backend_counts)
        )
        if inpaint_backend and fallback_backend:
            report.stage("inpaint", "fallback", detail=f"backend={inpaint_backend}{comfy_note}")
            skip = inpaint_diag.get("auto_skip_reason")
            reason = f"Big-LaMa unavailable; used {inpaint_backend} fallback for background plate"
            if skip:
                reason = f"{reason} ({skip})"
            report.degraded("inpaint", reason)
        elif inpaint_backend:
            passes = inpaint_diag.get("inpaint_passes")
            pass_note = f"; passes={passes}" if passes else ""
            routes = ",".join(f"{name}:{count}" for name, count in sorted(backend_counts.items()))
            route_note = f"; routes={routes}" if routes else ""
            report.stage("inpaint", "ok", detail=f"backend={inpaint_backend}{route_note}{comfy_note}{pass_note}")

        if stage("layout") or dirty or not exists("layout.json"):
            current_stage = "layout"
            dirty = True
            try:
                tree = layout.infer(reconstruction.get("candidates", []), canvas, cfg)
            except Exception as exc:
                # Layout is an optimization, not the source of visual pixels. Preserve the
                # canonical flat scene and mark it failed instead of losing the whole design.
                tree = []
                for candidate in reconstruction.get("candidates", []):
                    if not isinstance(candidate, dict) or candidate.get("target") == "drop":
                        continue
                    item = copy.deepcopy(candidate)
                    item.setdefault("constraints", {"horizontal": "LEFT", "vertical": "TOP"})
                    item.setdefault("meta", {})["layout_fallback"] = True
                    tree.append(item)
                detail = f"layout inference failed; preserved {len(tree)} flat layer(s): {exc}"
                report.stage("layout", "fallback", detail=detail, artifacts=["layout.json"])
                report.degraded("layout", detail, required=True)
                runtime_violations = report.violations
            dump(tree, A("layout.json"))
            _log(run_dir, f"layout → {len(tree)} root layers")
        tree = load(A("layout.json"))

        # 8 schema-v2 Figma scene graph (source of truth)
        if stage("design") or dirty or not exists("design.json"):
            current_stage = "design"
            dirty = True
            design_updated = True
            kept = [c.get("text") for c in reconstruction.get("candidates", [])
                    if c.get("target") == "drop" and c.get("text")]
            doc = build_design_json.build(
                tree, canvas, run_dir, base_src=A(reconstruction.get("background", "background_clean.png")),
                doc_id=os.path.basename(run_dir), name=os.path.basename(run_dir), kept_in_photo=kept,
            )
            _log(run_dir, f"design.json → {len(doc.layers)} layers, kept_in_photo={len(doc.kept_in_photo)}")

        # 8.5 LOCAL PREVIEW — see the layers without Figma (default on)
        need_local_preview = bool(cfg.get("preview", True)) or (
            (stage("diff") or stage("qa")) and not exists("figma_export.png")
        )
        if need_local_preview and (stage("preview") or dirty or not exists("preview.png")):
            current_stage = "preview"
            pv = render_preview.render(A("design.json"), run_dir)
            _log(run_dir, f"preview → {pv['preview']} ({pv['count']} layers in layers/, see layers_contact.png)")
            if pv.get("errors"):
                detail = f"{len(pv['errors'])} preview layer(s) failed; first: {pv['errors'][0].get('detail')}"
                report.stage("preview", "partial", detail=detail, artifacts=["preview.png"])
                report.degraded("preview", detail, required=True)
                runtime_violations = report.violations

        # 9 figma import (optional — Figma export can come later)
        if stage("figma") and cfg.get("figma", {}).get("enabled", False):
            imp = figma_import.import_design(A("design.json"), run_dir, cfg)
            dump(imp, A("figma_import.json")); _log(run_dir, f"figma import: {imp.get('action', imp)}")

        # 10 export screenshot (plugin writes it; may need the manual click)
        if stage("export"):
            exp = figma_import.export_screenshot(run_dir, cfg, wait_s=cfg.get("export_wait_s", 0))
            _log(run_dir, f"export: {exp.get('note', exp.get('path'))}")

        # 11 diff + 12 qa — QA against the Figma render if present, else the local preview
        # A stale Figma export must not judge a newly rebuilt design. Until this run exports
        # again, the freshly generated local preview is the only matching render.
        use_figma_export = _artifact_at_least_as_fresh(
            A("figma_export.png"), A("design.json")
        )
        qa_render = A("figma_export.png") if use_figma_export else \
            (A("preview.png") if exists("preview.png") else None)
        if (stage("diff") or stage("qa")) and qa_render:
            current_stage = "qa"
            ren_ocr = None
            if cfg.get("qa_ocr", True):
                try:
                    # QA OCR is an observation of the rendered output, not a new source
                    # observation.  Do not let run_ocr's default artifact writer replace
                    # the canonical source-derived ocr.json used by resumed stages.
                    ren_ocr = ocr.run_ocr(qa_render, cfg, run_dir="")
                    ren_ocr["provenance"] = {
                        "kind": "render-qa",
                        "render_path": os.path.abspath(qa_render),
                        "source_ocr_path": os.path.abspath(A("ocr.json")),
                    }
                    dump(ren_ocr, A("render_ocr.json"))
                except Exception as exc:
                    # Pixel/structural judges remain useful when render OCR is down. Keep
                    # text_recall unknown and make that loss of evidence explicit.
                    detail = f"render OCR judge failed: {exc}"
                    report.degraded("qa-ocr", detail, required=True)
                    runtime_violations = report.violations
            try:
                qa_partial = pixel_diff.compare(norm_path, qa_render, run_dir,
                                                source_ocr=ocr_res, render_ocr=ren_ocr,
                                                thresholds=pixel_diff_thresholds(cfg))
            except Exception as exc:
                # A minimal independent pixel judge keeps the run inspectable, but can never
                # turn a primary QA crash into success.
                import numpy as np
                from PIL import Image
                source_image = Image.open(norm_path).convert("RGB")
                render_image = Image.open(qa_render).convert("RGB").resize(source_image.size)
                delta = np.abs(np.asarray(source_image, dtype=np.float32) -
                               np.asarray(render_image, dtype=np.float32))
                mae = float(delta.mean())
                Image.fromarray(np.clip(delta.mean(axis=2) * 3, 0, 255).astype(np.uint8)).save(A("diff.png"))
                qa_partial = {
                    "ssim": 0.0, "visual_score": round(max(0.0, 1.0 - mae / 255.0), 4),
                    "rgb_mae": round(mae, 4), "text_recall": None,
                    "quality_flags": [{"rule": "qa-primary-failed", "detail": str(exc)}],
                    "structural": {},
                    "hard_fails": [{"rule": "qa-primary-failed", "detail": str(exc), "hard": True}],
                }
                detail = f"primary QA failed; independent pixel judge recorded MAE={mae:.2f}: {exc}"
                report.stage("qa", "fallback", detail=detail, artifacts=["diff.png"])
                report.degraded("qa", detail, required=True)
                runtime_violations = report.violations
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
                  "recommended_resume": recommended_resume(reps),
                  "structural": structural_report,
                  "ok": qa_partial.get("ssim", 0) >= visual_pass_ssim(cfg) and not combined_fails}
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

        repair_summary = None
        if not in_harness_loop(cfg) and harness_enabled(cfg):
            needs_repair = False
            if os.path.exists(A("qa.json")):
                needs_repair = not load(A("qa.json")).get("ok")
            elif not report.acceptable:
                needs_repair = True
            if needs_repair:
                max_rounds = harness_max_rounds(cfg)
                repair_summary = run_harness_after_pipeline(
                    input_path, run_dir, cfg, max_rounds=max_rounds)
                _log(run_dir, f"harness → {repair_summary.get('stopped')} "
                     f"after {repair_summary.get('rounds_completed', 0)}/{max_rounds} round(s)")

        result = {"ok": True, "run_dir": run_dir, "duration_s": elapsed,
                  "runtime_ok": report.acceptable, "runtime_status": report.data.get("status")}
        if os.path.exists(A("qa.json")):
            result["qa_ok"] = bool(load(A("qa.json")).get("ok"))
        if repair_summary is not None:
            result["repair"] = repair_summary
            result["harness"] = repair_summary if harness_enabled(cfg) else None
            result["qa_ok"] = bool(repair_summary.get("qa_ok"))
            result["harness_rounds"] = repair_summary.get("rounds_completed")
            result["harness_stopped"] = repair_summary.get("stopped")
        return result
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
