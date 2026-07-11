"""qwen_worker.py — stage 4: Qwen-Image-Layered RGBA layer proposals.

propose_layers(img_path, run_dir, cfg) decomposes the ad into stacked RGBA layers
and returns a back-to-front list[schema.QwenLayer]. Two backends (cfg.qwen.mode):

  'comfyui'          -> POST the API workflow (cfg.qwen.workflow) to cfg.backend_url
                        /prompt, poll /history, download the RGBA PNGs. requests only.
  'direct-diffusers' -> load QwenImageLayeredPipeline (diffusers) on cfg.device and run.

Layers are saved to <run_dir>/qwen_layers/Q<i>.png and their tight non-transparent
bbox recorded. On backend-down / model-missing / any failure the worker logs a note
and returns [] — it NEVER throws (the pipeline degrades gracefully per the contract).

Model: Qwen/Qwen-Image-Layered (HF). ComfyUI weights: Comfy-Org/Qwen-Image-Layered_ComfyUI.
See workflows/qwen_layered_{4,8}_api.json for the graph + install notes.
"""
from __future__ import annotations
import importlib
import json
import os
import time
import uuid
from typing import Optional


def _load_schema():
    for name in ("src.schema", "schema"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("schema")


# layer count -> ComfyUI EmptyHunyuanLatentVideo.length (benjiyaya mapper formula)
def _length_for_layers(n):
    return 5 + (max(1, int(n)) - 1) * 4


def _tight_bbox(rgba):
    """Tight bbox of non-transparent content. rgba: numpy HxWx4 -> {x,y,w,h}."""
    import numpy as np
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:  # opaque layer
        h, w = rgba.shape[:2]
        return {"x": 0, "y": 0, "w": int(w), "h": int(h)}
    ys, xs = np.where(alpha > 0)
    if xs.size == 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def _finalize(layer_pngs, run_dir):
    """layer_pngs: ordered list of absolute PNG paths (back-to-front).
    Move/record into qwen_layers/Q<i>.png and build QwenLayer dicts."""
    import numpy as np
    from PIL import Image

    out_dir = os.path.join(run_dir, "qwen_layers")
    os.makedirs(out_dir, exist_ok=True)
    layers = []
    for i, src in enumerate(layer_pngs):
        rel = os.path.join("qwen_layers", f"Q{i}.png")
        dst = os.path.join(run_dir, rel)
        img = Image.open(src).convert("RGBA")
        img.save(dst)
        box = _tight_bbox(np.asarray(img))
        layers.append(
            {"id": f"Q{i}", "png": rel, "box": box, "kind_hint": "unknown"}
        )
    return layers


def _write_manifest(schema, layers, run_dir, note=None):
    os.makedirs(run_dir, exist_ok=True)
    schema.dump(layers, os.path.join(run_dir, "qwen.json"))
    if note:
        with open(os.path.join(run_dir, "qwen.note.txt"), "w", encoding="utf-8") as f:
            f.write(note + "\n")


# ── ComfyUI backend ──────────────────────────────────────────────────────────────────
def _run_comfyui(img_path, run_dir, cfg, schema):
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise ImportError("comfyui backend requires requests.  pip install requests") from e

    qcfg = cfg.get("qwen") or {}
    base = cfg.get("backend_url", "http://127.0.0.1:8188").rstrip("/")
    wf_path = qcfg.get("workflow", "workflows/qwen_layered_8_api.json")
    layers_n = int(qcfg.get("layers", 8))

    if not os.path.exists(wf_path):
        note = f"qwen(comfyui): workflow not found: {wf_path}"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    with open(wf_path, encoding="utf-8") as f:
        workflow = json.load(f)
    workflow.pop("_note", None)

    # upload the input image to ComfyUI so LoadImage can find it
    try:
        with open(img_path, "rb") as fh:
            up = requests.post(
                f"{base}/upload/image",
                files={"image": (os.path.basename(img_path), fh, "image/png")},
                data={"overwrite": "true"},
                timeout=30,
            )
        up.raise_for_status()
        uploaded_name = up.json().get("name", os.path.basename(img_path))
    except Exception as e:
        note = f"qwen(comfyui): image upload failed ({e}); backend likely down"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    # patch the graph: LoadImage image + layer count length
    for node in workflow.values():
        ct = node.get("class_type")
        if ct == "LoadImage":
            node["inputs"]["image"] = uploaded_name
        elif ct == "EmptyHunyuanLatentVideo":
            node["inputs"]["length"] = _length_for_layers(layers_n)

    client_id = str(uuid.uuid4())
    try:
        resp = requests.post(
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except Exception as e:
        note = f"qwen(comfyui): /prompt failed ({e})"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    # poll /history/<id>
    timeout_s = int(qcfg.get("timeout_s", 600))
    deadline = time.time() + timeout_s
    history = None
    while time.time() < deadline:
        try:
            h = requests.get(f"{base}/history/{prompt_id}", timeout=15).json()
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in h:
            history = h[prompt_id]
            break
        time.sleep(2)
    if history is None:
        note = f"qwen(comfyui): timed out after {timeout_s}s waiting for {prompt_id}"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    # collect output images (SaveImage nodes) in node order
    tmp_dir = os.path.join(run_dir, "qwen_layers", "_raw")
    os.makedirs(tmp_dir, exist_ok=True)
    downloaded = []
    outputs = history.get("outputs", {})
    for node_id in sorted(outputs.keys(), key=lambda k: str(k)):
        for img in outputs[node_id].get("images", []):
            params = {
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            }
            try:
                r = requests.get(f"{base}/view", params=params, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"[qwen] download failed for {img['filename']}: {e}")
                continue
            dst = os.path.join(tmp_dir, img["filename"])
            with open(dst, "wb") as fh:
                fh.write(r.content)
            downloaded.append(dst)

    if not downloaded:
        note = "qwen(comfyui): run completed but produced no images"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    # ComfyUI batch is saved in order = layer order (back-to-front as decoded)
    layers = _finalize(downloaded, run_dir)
    _write_manifest(schema, layers, run_dir)
    print(f"[qwen] comfyui produced {len(layers)} layers")
    return layers


# ── direct diffusers backend ─────────────────────────────────────────────────────────
def _run_diffusers(img_path, run_dir, cfg, schema):
    try:
        import torch
        from diffusers import QwenImageLayeredPipeline
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "direct-diffusers backend requires diffusers (from git), transformers>=4.51.3, "
            "torch.\n  pip install git+https://github.com/huggingface/diffusers "
            "transformers accelerate"
        ) from e

    qcfg = cfg.get("qwen") or {}
    device = cfg.get("device", "cuda")
    model_id = qcfg.get("model", "Qwen/Qwen-Image-Layered")
    layers_n = int(qcfg.get("layers", 8))

    try:
        pipe = QwenImageLayeredPipeline.from_pretrained(model_id)
        pipe = pipe.to(device, torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        note = f"qwen(diffusers): model load failed ({e})"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    image = Image.open(img_path).convert("RGBA")
    gen = torch.Generator(device=device).manual_seed(int(qcfg.get("seed", 777)))
    inputs = {
        "image": image,
        "generator": gen,
        "true_cfg_scale": float(qcfg.get("true_cfg_scale", 4.0)),
        "negative_prompt": qcfg.get("negative_prompt", " "),
        "num_inference_steps": int(qcfg.get("steps", 50)),
        "num_images_per_prompt": 1,
        "layers": layers_n,
        "resolution": int(qcfg.get("resolution", 640)),
        "cfg_normalize": bool(qcfg.get("cfg_normalize", True)),
        "use_en_prompt": bool(qcfg.get("use_en_prompt", True)),
    }
    try:
        with torch.inference_mode():
            output = pipe(**inputs)
        # output.images[0] is a list of RGBA layer PIL images (back-to-front)
        layer_imgs = output.images[0]
    except Exception as e:
        note = f"qwen(diffusers): inference failed ({e})"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    tmp_dir = os.path.join(run_dir, "qwen_layers", "_raw")
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i, im in enumerate(layer_imgs):
        p = os.path.join(tmp_dir, f"raw_{i}.png")
        im.convert("RGBA").save(p)
        paths.append(p)
    layers = _finalize(paths, run_dir)
    _write_manifest(schema, layers, run_dir)
    print(f"[qwen] diffusers produced {len(layers)} layers")
    return layers


# ── public API ───────────────────────────────────────────────────────────────────────
def propose_layers(img_path: str, run_dir: str, cfg: Optional[dict] = None):
    schema = _load_schema()
    cfg = cfg or {}
    os.makedirs(run_dir, exist_ok=True)
    qcfg = cfg.get("qwen") or {}
    if qcfg.get("enabled", True) is False:
        note = "qwen: disabled (SAM/residual pipeline remains active)"
        _write_manifest(schema, [], run_dir, note)
        return []
    mode = qcfg.get("mode", "comfyui")
    try:
        if mode == "comfyui":
            return _run_comfyui(img_path, run_dir, cfg, schema)
        elif mode in ("direct-diffusers", "diffusers", "direct"):
            return _run_diffusers(img_path, run_dir, cfg, schema)
        else:
            note = f"qwen: unknown mode '{mode}'"
            print("[qwen]", note)
            _write_manifest(schema, [], run_dir, note)
            return []
    except ImportError as e:
        # heavy dep missing -> degrade, don't throw
        note = f"qwen: backend deps missing -> {e}"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []
    except Exception as e:  # pragma: no cover - defensive; contract says never throw
        note = f"qwen: unexpected error -> {e}"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []


if __name__ == "__main__":  # CPU-safe smoke: no backend -> graceful [] + note
    import tempfile
    from PIL import Image

    run_dir = tempfile.mkdtemp(prefix="qwen_smoke_")
    src = os.path.join(run_dir, "in.png")
    Image.new("RGB", (256, 256), (120, 140, 200)).save(src)
    # point at a dead backend to exercise the degrade path
    cfg = {"qwen": {"mode": "comfyui", "workflow": "workflows/qwen_layered_4_api.json",
                    "layers": 4, "timeout_s": 5},
           "backend_url": "http://127.0.0.1:65535"}
    layers = propose_layers(src, run_dir, cfg)
    print("layers:", layers)
    print("length_for(4) =", _length_for_layers(4), " length_for(8) =",
          _length_for_layers(8))
    assert layers == []
    print("ok (degraded gracefully)")
