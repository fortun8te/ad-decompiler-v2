#!/usr/bin/env python3
"""Bounded, actual-runtime smoke probes for the GPU worker.

Unlike doctor.py's cheap readiness checks, these probes execute one tiny real inference or
staging operation. Every probe runs in an isolated process with a hard timeout so a wedged
model/backend cannot hang a benchmark preflight.
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path


PROBES = ("ocr", "sam3", "vlm", "big_lama", "flux_comfy", "vectorization", "figma_staging")


def _fixture(directory: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    draw.text((18, 18), "GPU SMOKE", font=font, fill="black", stroke_width=1)
    draw.rounded_rectangle((28, 82, 168, 146), radius=12, fill="#ef573f")
    draw.polygon(((196, 80), (232, 114), (196, 148)), fill="#2457d6")
    image_path = directory / "smoke.png"
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((28, 82, 168, 146), radius=12, fill=255)
    mask_path = directory / "mask.png"
    mask.save(mask_path)
    return image_path, mask_path


def _probe_ocr(cfg: dict, work: Path) -> dict:
    from src import ocr
    image, _ = _fixture(work)
    result = ocr.run_ocr(str(image), cfg, run_dir=str(work))
    lines = result.get("lines") or []
    primary = str((cfg.get("ocr") or {}).get("primary", "ppocr-v6")).lower()
    engine = str(result.get("engine") or "").lower()
    ok = result.get("status", "ok") == "ok" and bool(lines) and primary in engine
    return {"ok": ok, "detail": f"engine={engine}; lines={len(lines)}", "evidence": {
        "engine": engine, "texts": [line.get("text") for line in lines[:3]],
    }}


def _probe_sam3(cfg: dict, work: Path) -> dict:
    from src import sam3_detect
    image, _ = _fixture(work)
    residual = [{
        "id": "smoke-arrow", "kind": "icon", "role": "icon",
        "box": {"x": 188, "y": 72, "w": 52, "h": 84}, "score": .95,
    }]
    result = sam3_detect.detect(str(image), residual=residual, cfg=cfg, run_dir=str(work))
    elements = result.get("elements") or []
    ok = result.get("status") == "ok" and bool(elements)
    return {"ok": ok, "detail": f"status={result.get('status')}; elements={len(elements)}",
            "evidence": {"status": result.get("status"), "elements": len(elements)}}


def _probe_vlm(cfg: dict, work: Path) -> dict:
    from src.vlm_client import ask_vlm
    image, _ = _fixture(work)
    vlm = cfg.get("vlm") or {}
    schema = {"type": "object", "properties": {"label": {"type": "string"}},
              "required": ["label"], "additionalProperties": False}
    answer = ask_vlm(
        image.read_bytes(),
        "Return JSON only. The label must be exactly gpu-smoke.",
        base_url=str(vlm.get("base_url", "http://127.0.0.1:1234/v1")),
        model=str(vlm.get("model", "google/gemma-4-e4b")),
        timeout_s=float(vlm.get("timeout_s", 45)), max_tokens=500,
        response_schema=schema,
    )
    payload = json.loads(answer)
    ok = str(payload.get("label", "")).strip().lower() == "gpu-smoke"
    return {"ok": ok, "detail": f"model={vlm.get('model', 'google/gemma-4-e4b')}; answer={payload}",
            "evidence": payload}


def _probe_big_lama(cfg: dict, work: Path) -> dict:
    import numpy as np
    from PIL import Image
    from src import inpaint
    image, mask = _fixture(work)
    output = work / "inpainted.png"
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.setdefault("inpaint", {})["mode"] = "big-lama"
    result = inpaint.inpaint_once(str(image), str(mask), str(output), probe_cfg)
    before = np.asarray(Image.open(image).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    region = np.asarray(Image.open(mask).convert("L")) > 0
    change = float(np.abs(before.astype(float) - after.astype(float))[region].mean())
    ok = result.get("backend") == "big-lama" and output.is_file() and change > 1
    return {"ok": ok, "detail": f"backend={result.get('backend')}; masked_mae={change:.2f}",
            "evidence": {"backend": result.get("backend"), "masked_mae": round(change, 3)}}


def _probe_flux_comfy(cfg: dict, work: Path) -> dict:
    """Submit a real Flux crop; liveness alone does not prove the workflow can run."""
    import numpy as np
    from PIL import Image
    from src import inpaint
    image, mask = _fixture(work)
    output = work / "flux-inpainted.png"
    probe_cfg = copy.deepcopy(cfg)
    inpaint_cfg = probe_cfg.setdefault("inpaint", {})
    inpaint_cfg["mode"] = "flux_comfy"
    inpaint_cfg["allow_fallback"] = False
    inpaint_cfg.setdefault("comfy", {})["enabled"] = True
    inpaint_cfg["comfy"]["required"] = True
    result = inpaint.inpaint_once(str(image), str(mask), str(output), probe_cfg)
    before = np.asarray(Image.open(image).convert("RGB"))
    after = np.asarray(Image.open(output).convert("RGB"))
    region = np.asarray(Image.open(mask).convert("L")) > 0
    outside_identical = bool(np.array_equal(before[~region], after[~region]))
    masked_mae = float(np.abs(before.astype(float) - after.astype(float))[region].mean())
    same_shape = before.shape == after.shape
    backend = str(result.get("backend") or "")
    ok = backend == "flux-comfy" and same_shape and outside_identical and masked_mae > 1
    return {
        "ok": ok,
        "detail": (f"backend={backend}; shape_ok={same_shape}; "
                   f"outside_identical={outside_identical}; masked_mae={masked_mae:.2f}"),
        "evidence": {"backend": backend, "shape_ok": same_shape,
                     "outside_identical": outside_identical,
                     "masked_mae": round(masked_mae, 3)},
    }


def _probe_vectorization(cfg: dict, work: Path) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw
    from src import vectorize
    icon = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.polygon(((12, 48), (58, 12), (58, 34), (84, 34),
                  (84, 62), (58, 62), (58, 84)), fill="#17233d")
    draw.ellipse((30, 34, 50, 54), fill="#ef573f")
    draw.rectangle((58, 40, 76, 56), fill="#f2c94c")
    result = vectorize.vectorize_crop(np.asarray(icon), cfg, role="icon")
    ok = bool(result.get("ok")) and result.get("engine") == "vtracer" and bool(result.get("paths"))
    return {"ok": ok, "detail": f"engine={result.get('engine')}; score={result.get('score')}; paths={len(result.get('paths') or [])}",
            "evidence": {key: result.get(key) for key in ("engine", "score", "gate", "note")}}


def _probe_figma_staging(cfg: dict, work: Path) -> dict:
    from PIL import Image
    from src.figma_import import import_design
    run = work / "figma-run"
    assets = run / "assets"
    assets.mkdir(parents=True)
    Image.new("RGBA", (8, 8), "red").save(assets / "dot.png")
    design = {"schema_version": 2, "id": "gpu-smoke", "name": "GPU smoke",
              "canvas": {"w": 32, "h": 32}, "layers": [{"id": "dot", "type": "image",
              "name": "Dot", "box": {"x": 4, "y": 4, "w": 8, "h": 8}, "src": "assets/dot.png"}]}
    design_path = run / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    inbox = work / "figma-inbox"
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["figma"] = {"mode": "plugin", "inbox": str(inbox)}
    result = import_design(str(design_path), str(run), probe_cfg)
    manifest = json.loads((inbox / "inbox.json").read_text(encoding="utf-8")) if result.get("ok") else {}
    files = manifest.get("files") or []
    ok = bool(result.get("ok")) and any(row.get("path") == "assets/dot.png" and row.get("sha256") for row in files)
    return {"ok": ok, "detail": f"staged={result.get('ok')}; files={len(files)}",
            "evidence": {"doc_id": manifest.get("doc_id"), "files": len(files)}}


_IMPLEMENTATIONS = {"ocr": _probe_ocr, "sam3": _probe_sam3, "vlm": _probe_vlm,
                    "big_lama": _probe_big_lama, "flux_comfy": _probe_flux_comfy,
                    "vectorization": _probe_vectorization,
                    "figma_staging": _probe_figma_staging}


def _worker(name: str, cfg: dict, work: str, output) -> None:
    started = time.monotonic()
    try:
        result = _IMPLEMENTATIONS[name](cfg, Path(work))
    except Exception as exc:
        result = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    result.update({"name": name, "duration_s": round(time.monotonic() - started, 3)})
    output.put(result)


def _run_bounded(name: str, cfg: dict, work: Path, timeout_s: float) -> dict:
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    # Some OCR/model runtimes create helper processes of their own; a daemonic Python
    # process is forbidden from doing that. The parent still owns the hard timeout below.
    process = context.Process(target=_worker, args=(name, cfg, str(work), output))
    started = time.monotonic()
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate(); process.join(5)
        return {"name": name, "ok": False, "timeout": True,
                "duration_s": round(time.monotonic() - started, 3),
                "detail": f"probe exceeded {timeout_s:.1f}s timeout"}
    try:
        return output.get(timeout=1)
    except queue.Empty:
        return {"name": name, "ok": False, "duration_s": round(time.monotonic() - started, 3),
                "detail": f"probe process exited {process.exitcode} without evidence"}


def run_all(cfg: dict, output_dir: str | Path, *, probes=PROBES, timeout_s: float = 120) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for name in probes:
        if name not in _IMPLEMENTATIONS:
            checks.append({"name": name, "ok": False, "detail": "unknown probe"})
            continue
        checks.append(_run_bounded(name, cfg, output_dir / name, timeout_s))
    report = {"version": 1, "ok": all(item.get("ok") for item in checks),
              "checks": checks, "timeout_s": timeout_s}
    (output_dir / "runtime_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
