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
        # Validate before publishing. A zero-alpha image is not a layer and used to pass
        # through as a successful Qwen observation, only to fail much later in fusion.
        img = Image.open(src).convert("RGBA")
        box = _tight_bbox(np.asarray(img))
        if box["w"] <= 0 or box["h"] <= 0:
            continue
        index = len(layers)
        rel = os.path.join("qwen_layers", f"Q{index}.png")
        dst = os.path.join(run_dir, rel)
        img.save(dst)
        layers.append(
            {"id": f"Q{index}", "png": rel, "box": box, "kind_hint": "unknown"}
        )
    return layers


def _write_manifest(schema, layers, run_dir, note=None):
    os.makedirs(run_dir, exist_ok=True)
    schema.dump(layers, os.path.join(run_dir, "qwen.json"))
    if note:
        with open(os.path.join(run_dir, "qwen.note.txt"), "w", encoding="utf-8") as f:
            f.write(note + "\n")
    else:
        # A recovered retry must clear an older failure marker.  Leaving this file behind
        # makes qwen_degradation() report a false failure even though fresh layers exist.
        try:
            os.remove(os.path.join(run_dir, "qwen.note.txt"))
        except FileNotFoundError:
            pass


def _last_note(run_dir: str) -> str:
    path = os.path.join(run_dir, "qwen.note.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _recent_deterministic_failure(run_dir: str, qcfg: dict) -> str:
    """Return a cached non-transient failure during the harness cooldown window."""
    if qcfg.get("force_retry"):
        return ""
    path = os.path.join(run_dir, "qwen.note.txt")
    note = _last_note(run_dir)
    markers = ("/prompt failed", "prompt_outputs_failed_validation", "validation=")
    if not note or not any(marker in note.lower() for marker in markers):
        return ""
    try:
        age = max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        return ""
    cooldown = max(0.0, float(qcfg.get("failure_cooldown_s", 900)))
    return note if age < cooldown else ""


def _http_error_detail(response, limit: int = 1800) -> str:
    """Return bounded ComfyUI validation detail for an unsuccessful request.

    ComfyUI's useful explanation (missing model, invalid node input, absent custom node)
    lives in the JSON response body.  Logging only ``400 Bad Request`` made a configured
    but incomplete backend indistinguishable from a broken workflow.
    """
    if response is None:
        return ""
    try:
        payload = response.json()
        detail = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        detail = str(getattr(response, "text", "") or "")
    detail = " ".join(detail.split())
    if not detail:
        return ""
    return detail[: max(0, int(limit))]


def _comfy_auth_headers(section: Optional[dict] = None) -> dict[str, str]:
    """Return Comfy Cloud authentication headers without persisting secrets.

    Local ComfyUI ignores the empty header set. Remote providers compatible with the
    official Comfy Cloud API can opt in with ``api_key`` or, preferably,
    ``api_key_env`` (default ``COMFY_CLOUD_API_KEY``).
    """
    section = section or {}
    key = str(section.get("api_key") or "").strip()
    if not key:
        env_name = str(section.get("api_key_env") or "COMFY_CLOUD_API_KEY").strip()
        key = str(os.environ.get(env_name, "") or "").strip()
    return {"X-API-Key": key} if key else {}


# ── ComfyUI backend ──────────────────────────────────────────────────────────────────
def _run_comfyui(img_path, run_dir, cfg, schema):
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise ImportError("comfyui backend requires requests.  pip install requests") from e

    qcfg = cfg.get("qwen") or {}
    base = str(qcfg.get("base_url") or cfg.get("backend_url", "http://127.0.0.1:8188")).rstrip("/")
    headers = _comfy_auth_headers(qcfg)
    wf_path = qcfg.get("workflow", "workflows/qwen_layered_8_api.json")
    layers_n = int(qcfg.get("layers", 8))

    # Fail fast when the separately hosted ComfyUI service is not running. The old
    # upload-first path could waste thirty seconds on every harness round.
    try:
        probe = requests.get(f"{base}/system_stats", headers=headers,
                             timeout=float(qcfg.get("probe_timeout_s", 2)))
        probe.raise_for_status()
    except Exception as e:
        note = f"qwen(comfyui): backend offline ({e})"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

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
                headers=headers,
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
    resp = None
    try:
        resp = requests.post(
            f"{base}/prompt",
            headers=headers,
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except Exception as e:
        detail = _http_error_detail(resp)
        suffix = f"; validation={detail}" if detail else ""
        note = f"qwen(comfyui): /prompt failed ({e}){suffix}"
        print("[qwen]", note)
        _write_manifest(schema, [], run_dir, note)
        return []

    # poll /history/<id>
    timeout_s = int(qcfg.get("timeout_s", 600))
    deadline = time.time() + timeout_s
    history = None
    while time.time() < deadline:
        try:
            h = requests.get(f"{base}/history/{prompt_id}", headers=headers, timeout=15).json()
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
                r = requests.get(f"{base}/view", headers=headers, params=params, timeout=30)
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


# ── Flux Fill inpaint backend (ComfyUI) ──────────────────────────────────────────────
def _requests():
    """Indirection so tests can inject a fake HTTP client without a live ComfyUI."""
    import requests
    return requests


_FLUX_DEFAULTS = {
    "base_url": "http://127.0.0.1:8188",
    "workflow": "workflows/flux_fill_inpaint_api.json",
    "steps": 8,
    "cfg": 1.0,
    # guidance 3.5 won the 2026-07-15 settings A/B (runs/flux-settings-ab) on the
    # crop-local photo/complex holes this backend actually receives; see config.yaml.
    "guidance": 3.5,
    "denoise": 1.0,
    "seed": 0,
    "prompt": "",
    "negative_prompt": "",
    "probe_timeout_s": 2.0,
    "timeout_s": 300,
}


def _resolve_workflow_path(wf_path: str) -> Optional[str]:
    """Find the workflow JSON relative to cwd or the repo root (parent of src/)."""
    if os.path.isabs(wf_path):
        return wf_path if os.path.exists(wf_path) else None
    if os.path.exists(wf_path):
        return wf_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt = os.path.join(repo_root, wf_path)
    return alt if os.path.exists(alt) else None


def _comfy_abort(base, headers, prompt_id=None):
    """Interrupt the running job and clear the ComfyUI queue (GB1).

    A Flux Fill job that we stop waiting on (timeout / decode error / no images)
    keeps running on the GPU holding ~16 GB. The NEXT run's /prompt then queues
    behind the wedged job and times out too — looking like an unrelated pipeline
    failure. Every early-return-after-/prompt path must call this so we never
    leave a job pinning the GPU. Best-effort: swallows all errors (the caller is
    already degrading to Big-LaMa and must never crash on cleanup).
    """
    try:
        requests = _requests()
    except Exception:  # pragma: no cover - requests always importable if we got here
        return
    for attempt in ("interrupt", "queue"):
        try:
            if attempt == "interrupt":
                requests.post(f"{base}/interrupt", headers=headers, timeout=10)
            else:
                # Clear any pending items; also delete the specific job if we have its id.
                payload = {"clear": True}
                if prompt_id:
                    payload["delete"] = [prompt_id]
                requests.post(f"{base}/queue", headers=headers, json=payload, timeout=10)
        except Exception as exc:  # pragma: no cover - network best-effort
            print(f"[flux-inpaint] comfy {attempt} cleanup failed ({exc})")


def flux_inpaint(rgb, mask, cfg: Optional[dict] = None):
    """Inpaint a background plate with FLUX.1 Fill dev on ComfyUI.

    ``rgb`` is an HxWx3 uint8 array, ``mask`` an HxW array where non-zero pixels are the
    region to regenerate.  Returns an HxWx3 uint8 array on success, or ``None`` when
    ComfyUI is unreachable / the workflow is missing / anything fails.  This NEVER raises
    for backend problems: src/inpaint.py degrades to Big-LaMa/OpenCV so a downed GPU box
    can never crash the pipeline.

    Config is read from ``cfg['inpaint']['comfy']`` (see workflows/flux_fill_inpaint_api.json
    _note for the keys).  ``base_url`` falls back to the top-level ``backend_url``.
    """
    import io
    import tempfile

    import numpy as np
    from PIL import Image

    try:
        requests = _requests()
    except ImportError:
        return None

    cfg = cfg or {}
    icfg = cfg.get("inpaint") or {}
    comfy = icfg.get("comfy") or {}
    params = dict(_FLUX_DEFAULTS)
    params.update({k: v for k, v in comfy.items() if k in _FLUX_DEFAULTS})
    base = str(comfy.get("base_url") or cfg.get("backend_url") or _FLUX_DEFAULTS["base_url"]).rstrip("/")
    headers = _comfy_auth_headers(comfy)

    # Expose last backend failure to src/inpaint.py without raising.
    flux_inpaint.__dict__["last_error"] = ""
    try:
        # Fail fast when the separately hosted ComfyUI service is not running.
        try:
            probe = requests.get(f"{base}/system_stats", headers=headers,
                                 timeout=float(params["probe_timeout_s"]))
            probe.raise_for_status()
        except Exception as exc:
            msg = f"ComfyUI offline at {base} ({exc})"
            flux_inpaint.__dict__["last_error"] = msg
            print(f"[flux-inpaint] {msg}; caller will fall back")
            return None

        # Pre-flight queue hygiene (§10 discipline / GB1 companion): a killed
        # previous run can leave stale queued prompts that this call would queue
        # behind and time out on. Clear pending items before submitting ours.
        # Best-effort — never blocks the real work.
        try:
            queue_state = requests.get(f"{base}/queue", headers=headers, timeout=10)
            pending = (queue_state.json() or {}).get("queue_pending") or []
            if pending:
                requests.post(f"{base}/queue", headers=headers,
                              json={"clear": True}, timeout=10)
                print(f"[flux-inpaint] cleared {len(pending)} stale queued job(s)")
        except Exception:
            pass

        wf_path = _resolve_workflow_path(str(params["workflow"]))
        if not wf_path:
            msg = f"workflow not found: {params['workflow']}"
            flux_inpaint.__dict__["last_error"] = msg
            print(f"[flux-inpaint] {msg}; caller will fall back")
            return None
        with open(wf_path, encoding="utf-8") as fh:
            workflow = json.load(fh)
        workflow.pop("_note", None)

        # Stage the source + mask as PNGs and upload them so LoadImage can find them.
        tmp = tempfile.mkdtemp(prefix="flux_inpaint_")
        src_png = os.path.join(tmp, "flux_source.png")
        mask_png = os.path.join(tmp, "flux_mask.png")
        Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), "RGB").save(src_png)
        binary_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(binary_mask), "L").save(mask_png)

        def _upload(path):
            with open(path, "rb") as handle:
                resp = requests.post(
                    f"{base}/upload/image",
                    headers=headers,
                    files={"image": (os.path.basename(path), handle, "image/png")},
                    data={"overwrite": "true"},
                    timeout=30,
                )
            resp.raise_for_status()
            return resp.json().get("name", os.path.basename(path))

        try:
            src_name = _upload(src_png)
            mask_name = _upload(mask_png)
        except Exception as exc:
            msg = f"image/mask upload failed ({exc})"
            flux_inpaint.__dict__["last_error"] = msg
            print(f"[flux-inpaint] {msg}; caller will fall back")
            return None

        # Patch the graph. Match by _meta.title first (robust to node-id changes), then by
        # class_type. Optional model filename overrides keep the graph easy to re-point.
        models = comfy.get("models") or {}
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            title = str((node.get("_meta") or {}).get("title", ""))
            ins = node.setdefault("inputs", {})
            if ct == "LoadImage":
                is_mask = title == "mask_image" or "mask" in str(ins.get("image", "")).lower()
                ins["image"] = mask_name if is_mask else src_name
            elif ct == "CLIPTextEncode":
                if title == "negative_prompt":
                    ins["text"] = str(params["negative_prompt"])
                elif title == "positive_prompt":
                    ins["text"] = str(params["prompt"])
            elif ct == "FluxGuidance":
                ins["guidance"] = float(params["guidance"])
            elif ct == "KSampler":
                ins["steps"] = int(params["steps"])
                ins["cfg"] = float(params["cfg"])
                ins["denoise"] = float(params["denoise"])
                ins["seed"] = int(params["seed"])
            elif ct in ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced") and models.get("unet_gguf"):
                ins["unet_name"] = str(models["unet_gguf"])
            elif ct == "DualCLIPLoader":
                if models.get("t5xxl"):
                    ins["clip_name1"] = str(models["t5xxl"])
                if models.get("clip_l"):
                    ins["clip_name2"] = str(models["clip_l"])
            elif ct == "VAELoader" and models.get("vae"):
                ins["vae_name"] = str(models["vae"])
            elif ct in ("LoraLoaderModelOnly", "LoraLoader") and models.get("lora"):
                ins["lora_name"] = str(models["lora"])

        client_id = str(uuid.uuid4())
        try:
            resp = requests.post(
                f"{base}/prompt",
                headers=headers,
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
        except Exception as exc:
            detail = _http_error_detail(resp) if "resp" in locals() else ""
            suffix = f"; validation={detail}" if detail else ""
            msg = f"/prompt failed ({exc}){suffix}"
            flux_inpaint.__dict__["last_error"] = msg
            print(f"[flux-inpaint] {msg}; caller will fall back")
            return None

        deadline = time.time() + int(params["timeout_s"])
        history = None
        while time.time() < deadline:
            try:
                h = requests.get(f"{base}/history/{prompt_id}", headers=headers, timeout=15).json()
            except Exception:
                time.sleep(1.5)
                continue
            if prompt_id in h:
                history = h[prompt_id]
                break
            time.sleep(1.5)
        if history is None:
            msg = f"timed out waiting for {prompt_id}"
            flux_inpaint.__dict__["last_error"] = msg
            print(f"[flux-inpaint] {msg}; caller will fall back")
            # GB1: the job is still running on the GPU — interrupt + clear queue so
            # it does not pin ~16 GB and wedge the next run's /prompt.
            _comfy_abort(base, headers, prompt_id)
            return None

        outputs = history.get("outputs", {})
        for node_id in sorted(outputs.keys(), key=lambda k: str(k)):
            for img in outputs[node_id].get("images", []):
                if img.get("type") == "temp":
                    continue
                try:
                    view = requests.get(
                        f"{base}/view",
                        headers=headers,
                        params={
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        },
                        timeout=60,
                    )
                    view.raise_for_status()
                    arr = np.asarray(Image.open(io.BytesIO(view.content)).convert("RGB"), dtype=np.uint8)
                except Exception as exc:
                    msg = f"output download/decode failed ({exc})"
                    flux_inpaint.__dict__["last_error"] = msg
                    print(f"[flux-inpaint] {msg}")
                    continue
                print(f"[flux-inpaint] produced {arr.shape[1]}x{arr.shape[0]} plate")
                return arr
        msg = "run completed but produced no images"
        flux_inpaint.__dict__["last_error"] = msg
        print(f"[flux-inpaint] {msg}; caller will fall back")
        # History existed but yielded nothing usable — best-effort clear (job is
        # done in this case, but clearing keeps the queue clean for the next run).
        _comfy_abort(base, headers, prompt_id)
        return None
    except Exception as exc:  # pragma: no cover - last-ditch guard, must never crash the run
        msg = f"unexpected error ({exc})"
        flux_inpaint.__dict__["last_error"] = msg
        print(f"[flux-inpaint] {msg}; caller will fall back")
        # A job may have been submitted before the exception — clear it if we can.
        _lv = locals()
        if "base" in _lv and "headers" in _lv:
            try:
                _comfy_abort(_lv["base"], _lv["headers"], _lv.get("prompt_id"))
            except Exception:
                pass
        return None


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
    # Layer diffusion is strictly opt-in. It is a separately served optional
    # capability, so an absent config must never make an offline ComfyUI request.
    if qcfg.get("enabled", False) is False:
        note = "qwen: disabled (SAM/residual pipeline remains active)"
        _write_manifest(schema, [], run_dir, note)
        return []
    cached_failure = _recent_deterministic_failure(run_dir, qcfg)
    if cached_failure:
        print("[qwen] skipping cached deterministic workflow failure during cooldown")
        _write_manifest(schema, [], run_dir, cached_failure)
        return []
    primary_mode = str(qcfg.get("mode", "comfyui"))
    configured_fallbacks = qcfg.get("fallback_modes") or []
    if isinstance(configured_fallbacks, str):
        configured_fallbacks = [configured_fallbacks]
    modes = list(dict.fromkeys([primary_mode, *[str(item) for item in configured_fallbacks]]))
    failures = []
    for mode in modes:
        attempt_cfg = dict(cfg)
        attempt_cfg["qwen"] = {**qcfg, "mode": mode}
        try:
            if mode == "comfyui":
                layers = _run_comfyui(img_path, run_dir, attempt_cfg, schema)
            elif mode in ("direct-diffusers", "diffusers", "direct"):
                layers = _run_diffusers(img_path, run_dir, attempt_cfg, schema)
            else:
                layers = []
                _write_manifest(schema, [], run_dir, f"qwen: unknown mode '{mode}'")
        except ImportError as exc:
            layers = []
            _write_manifest(schema, [], run_dir, f"qwen({mode}): backend deps missing -> {exc}")
        except Exception as exc:  # pragma: no cover - defensive boundary around each backend
            layers = []
            _write_manifest(schema, [], run_dir, f"qwen({mode}): unexpected error -> {exc}")
        if layers:
            # Backends normally write the manifest themselves. Re-write on recovery so a
            # stale note from the failed first attempt cannot survive.
            _write_manifest(schema, layers, run_dir)
            if failures:
                print(f"[qwen] recovered with {mode} after {len(failures)} failed backend(s)")
            return layers
        failures.append(_last_note(run_dir) or f"qwen({mode}): produced no usable layers")

    note = " | ".join(failures) if failures else "qwen: no backend modes configured"
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
