"""peel_decompose.py — LayerD-style iterative peel decomposition for overlapping elements.

The single-union-mask pipeline inpaints ONCE, so any element sitting under another element
keeps a hole where its occluder used to be: moving it in Figma reveals the damage.  This
module owns the opposite invariant for dense/overlapping compositions:

    flattened image -> peel topmost layer (matte) -> unblend RGBA -> inpaint hole -> repeat

Each round the matting model predicts the alpha of the *topmost unoccluded* layer, that
layer is extracted as a full-canvas RGBA, the revealed hole is inpainted, and the loop
continues on the inpainted result until only background remains.  Elements that were
occluded become complete and independently movable.

Recipe follows LayerD (CyberAgent, ICCV 2025, Apache-2.0,
github.com/CyberAgentAILab/LayerD — src/layerd/models/layerd.py).  Adopted verbatim:

  * hard mask       = alpha > 0.005          (their ``_th_alpha``)
  * stop            when hard mask is empty ("no content") or covers > 0.99 of the
                     canvas ("full content" — the matting sees no separable top layer)
  * inpaint mask    = hard mask dilated by a kernel of ``round(dim * 0.015)`` per axis
                     (their ``kernel_scale``) so anti-aliased fringes never leak ghosts
  * unblending      fg = (image - (1 - a) * bg) / a with alpha snapped to {0,1} outside
                     [0, 0.95] (their ``_unblend_alpha_clip``) so soft edges keep the true
                     foreground color instead of a bg-contaminated blend
  * max iterations  3 by default (their ``decompose(max_iterations=3)``)
  * matting model   their fine-tuned BiRefNet (HF ``cyberagent/layerd-birefnet``) —
                     vanilla BiRefNet predicts *salient objects*; the LayerD fine-tune
                     predicts the *top layer*, which is the whole trick

Added beyond LayerD: a repeat-matte guard (two consecutive near-identical mattes mean the
matting is stuck and the inpainter failed to remove the layer — abort instead of emitting
duplicates) and an optional residual floor.

Zero import coupling to the rest of ``src``: matting and inpainting are injected
callables.  Defaults are OpenCV Telea (deterministic, test-safe) and, when installed,
``simple-lama-inpainting`` — the same ladder ``src/inpaint.py`` uses.  See
docs/PEEL-DECOMPOSITION.md for the adapter that routes holes through the existing
``src.inpaint.inpaint_array`` backends (Big-LaMa / Flux) instead.

Contracts:
  matting(rgb: HxWx3 uint8) -> HxW float alpha in [0, 1]
  inpaint(rgb: HxWx3 uint8, mask: HxW bool)  -> HxWx3 uint8 (only mask pixels may change)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional


def _deps():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("peel_decompose requires numpy, pillow and opencv-python") from exc
    return cv2, np, Image


class PeelDependencyError(ImportError):
    """A requested matting/inpaint backend is not installed. Message carries install steps."""


# ── configuration ──────────────────────────────────────────────────────────────────

#: LayerD defaults (github.com/CyberAgentAILab/LayerD, src/layerd/models/layerd.py).
DEFAULTS = {
    "max_layers": 3,             # LayerD decompose(max_iterations=3)
    "alpha_threshold": 0.005,    # LayerD _th_alpha — hard mask = alpha > this
    "full_coverage_stop": 0.99,  # LayerD: mean(hard_mask) > 0.99 → no separable top layer
    "min_coverage_stop": 0.0,    # LayerD stops only on an exactly-empty matte; raise this
                                 # (e.g. 0.0005) to also stop on speck-sized residue
    "kernel_scale": 0.015,       # LayerD: dilate inpaint mask by round(dim * scale)
    "unblend": True,             # LayerD use_unblend — recover true fg color at soft edges
    "unblend_alpha_clip": (0.0, 0.95),  # LayerD _unblend_alpha_clip
    "repeat_iou_stop": 0.95,     # ours: consecutive mattes this similar → matting is stuck
    "matting": {
        "backend": "auto",       # auto | birefnet | rembg (callable injection wins over all)
        "hf_card": "cyberagent/layerd-birefnet",   # LayerD's top-layer fine-tune
        "fallback_hf_card": "ZhengPeng7/BiRefNet",  # generic salient-object BiRefNet
        "device": "cpu",         # keep OFF the contended GPU unless explicitly granted
        "process_size": None,    # None → the model's trained size (1024 for BiRefNet)
    },
}


def _options(cfg: Optional[dict]) -> dict:
    """Merge ``cfg['peel']`` over DEFAULTS (one level deep for the matting block)."""
    peel_cfg = dict((cfg or {}).get("peel") or {})
    opts = {**DEFAULTS, **peel_cfg}
    opts["matting"] = {**DEFAULTS["matting"], **dict(peel_cfg.get("matting") or {})}
    return opts


# ── results ────────────────────────────────────────────────────────────────────────

@dataclass
class PeeledLayer:
    """One peeled foreground layer. ``peel_order`` 0 is the TOPMOST layer."""
    peel_order: int
    rgba: object                 # HxWx4 uint8 full-canvas RGBA (np.ndarray)
    bbox: dict                   # tight {x,y,w,h} of the hard mask
    area: int                    # hard-mask pixel count
    coverage: float              # area / canvas pixels
    meta: dict = field(default_factory=dict)


@dataclass
class PeelResult:
    layers: list                 # list[PeeledLayer], topmost first (peel order)
    background: object           # HxWx3 uint8 residual background plate (np.ndarray)
    stop_reason: str             # why the loop ended (see peel())
    canvas: dict                 # {"w": int, "h": int}
    steps: list = field(default_factory=list)   # per-iteration diagnostics

    def stack(self) -> list:
        """LayerD output order: background first, then foregrounds back-to-front."""
        return [self.background] + [layer.rgba for layer in reversed(self.layers)]


# ── mask / color helpers (LayerD helpers.py equivalents) ──────────────────────────

def _to_rgb(image):
    """Accept a path, PIL image, or HxWx3/HxWx4 array; return HxWx3 uint8."""
    _, np, Image = _deps()
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if hasattr(image, "convert"):  # PIL
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3/4 image, got shape {arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)


def _expand_mask(mask, kernel_size):
    """LayerD expand_mask: dilate a bool mask with an all-ones kernel."""
    cv2, np, _ = _deps()
    kh, kw = kernel_size
    if kh < 1 and kw < 1:
        return mask.copy()
    kernel = np.ones((max(1, kh), max(1, kw)), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _mask_iou(a, b) -> float:
    _, np, _ = _deps()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _tight_bbox(mask) -> dict:
    _, np, _ = _deps()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def estimate_fg_color(image, bg, alpha, clip_range=(0.0, 0.95)):
    """LayerD unblending: solve fg from  image = a*fg + (1-a)*bg  per pixel.

    Alpha is snapped to 0 below ``clip_range[0]`` and to 1 above ``clip_range[1]`` first,
    so nearly-opaque pixels keep their exact source color and only genuine soft edges are
    unblended.  Result is clipped to [0, 255] (LayerD ``clip_way="clip"``).
    """
    _, np, _ = _deps()
    image_f = np.asarray(image, dtype=np.float64)
    bg_f = np.asarray(bg, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64).copy()
    a[a <= clip_range[0]] = 0.0
    a[a >= clip_range[1]] = 1.0
    a3 = a[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        solved = (image_f - bg_f * (1.0 - a3)) / np.where(a3 > 0, a3, 1.0)
    fg = np.where(a3 > 0, solved, image_f)
    return np.clip(fg, 0, 255).astype(np.uint8)


# ── inpaint backends (defaults; the caller may inject anything) ────────────────────

def opencv_inpaint(rgb, mask, radius: int = 5):
    """Deterministic OpenCV Telea fill. Test-safe default; smears on textured bg."""
    cv2, np, _ = _deps()
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return cv2.inpaint(np.asarray(rgb, dtype=np.uint8), mask_u8, radius, cv2.INPAINT_TELEA)


def make_simple_lama_inpaint(device: str = "cpu") -> Callable:
    """Big-LaMa via ``simple-lama-inpainting`` (LayerD's own inpainter is LaMa).

    Weights download on first call. Raises PeelDependencyError when the package is
    missing.  Device defaults to CPU deliberately: SimpleLama's own default silently
    grabs CUDA whenever it is available, and the RTX 5080 is usually contended by the
    main pipeline — pass ``device="cuda"`` only as an explicit opt-in.
    """
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise PeelDependencyError(
            "simple-lama-inpainting is not installed: pip install simple-lama-inpainting"
        ) from exc
    _, np, Image = _deps()
    lama = SimpleLama(device=torch.device(device))

    def _inpaint(rgb, mask):
        mask_img = Image.fromarray(((np.asarray(mask) > 0) * 255).astype(np.uint8))
        out = lama(Image.fromarray(np.asarray(rgb, dtype=np.uint8)), mask_img)
        out = np.asarray(out.convert("RGB"), dtype=np.uint8)
        # LaMa pads to a multiple of 8 and can return a padded size (same failure mode
        # src/inpaint.py documents); snap back so compositing stays aligned.
        if out.shape[:2] != np.asarray(rgb).shape[:2]:
            cv2, _, _ = _deps()
            h, w = np.asarray(rgb).shape[:2]
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    return _inpaint


# ── matting backends ───────────────────────────────────────────────────────────────

def _load_birefnet_model(hf_card: str):
    """Load a BiRefNet checkpoint, surviving the transformers 4→5 break.

    The BiRefNet remote code on HF targets transformers 4.x; under 5.x its
    ``from_pretrained`` dies inside the meta-device machinery
    (``'BiRefNet' object has no attribute 'all_tied_weights_keys'``).  Fallback:
    resolve the remote class via ``get_class_from_dynamic_module``, instantiate it
    directly, and load the safetensors state dict ourselves (verified 0 missing /
    0 unexpected keys against cyberagent/layerd-birefnet).
    """
    from transformers import AutoModelForImageSegmentation
    try:
        return AutoModelForImageSegmentation.from_pretrained(hf_card, trust_remote_code=True)
    except AttributeError:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(hf_card, trust_remote_code=True)
        class_ref = (getattr(config, "auto_map", None) or {}).get(
            "AutoModelForImageSegmentation", "birefnet.BiRefNet")
        model_cls = get_class_from_dynamic_module(class_ref, hf_card)
        model = model_cls(config)
        state = load_file(hf_hub_download(hf_card, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise PeelDependencyError(
                f"BiRefNet manual load mismatch for {hf_card}: "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys")
        return model


def make_birefnet_matting(hf_card: str = "cyberagent/layerd-birefnet",
                          device: str = "cpu",
                          process_size: Optional[int] = None) -> Callable:
    """BiRefNet matting via ``transformers`` (trust_remote_code), LayerD preprocessing.

    ``cyberagent/layerd-birefnet`` is LayerD's fine-tune trained to matte the TOPMOST
    layer of a flattened design (Crello); ``ZhengPeng7/BiRefNet`` is the generic
    salient-object checkpoint (still useful, but it peels "the most salient thing",
    not strictly the top of the z-stack).  Weights (~1 GB) download on first use.
    The remote code additionally imports ``kornia`` — see requirements-gpu.txt.
    """
    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise PeelDependencyError(
            "BiRefNet matting needs torch, torchvision and transformers: "
            "pip install torch torchvision transformers"
        ) from exc
    cv2, np, Image = _deps()

    try:
        model = _load_birefnet_model(hf_card)
    except ImportError as exc:
        # The HF remote code imports kornia at module import time.
        raise PeelDependencyError(
            f"BiRefNet remote code for {hf_card} failed to import: {exc}\n"
            "most commonly missing: pip install kornia"
        ) from exc
    model.to(device)
    model.eval()

    side = process_size
    if side is None:
        side = getattr(getattr(model, "config", None), "size", None) or 1024
        if isinstance(side, (list, tuple)):
            side = side[0]
    # LayerD birefnet_matting.py: Resize → ToTensor → ImageNet mean/std normalize.
    transform = transforms.Compose([
        transforms.Resize((int(side), int(side))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def _matte(rgb):
        h, w = np.asarray(rgb).shape[:2]
        tensor = transform(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor)[0][-1].sigmoid().cpu()
        alpha = preds[0].squeeze().numpy()
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha.astype(np.float64), 0.0, 1.0)

    return _matte


def make_rembg_matting(model_name: str = "birefnet-general") -> Callable:
    """BiRefNet matting via the ``rembg`` package (ONNX runtime; alternative install)."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise PeelDependencyError(
            "rembg is not installed: pip install \"rembg[cpu]\"  "
            "(models download on first use; birefnet-general ≈ 1 GB)"
        ) from exc
    _, np, Image = _deps()
    session = new_session(model_name)

    def _matte(rgb):
        mask = remove(Image.fromarray(np.asarray(rgb, dtype=np.uint8)),
                      session=session, only_mask=True)
        return np.asarray(mask.convert("L"), dtype=np.float64) / 255.0

    return _matte


def resolve_matting(cfg: Optional[dict] = None) -> Callable:
    """Build the configured matting callable; ``auto`` tries LayerD BiRefNet → generic
    BiRefNet → rembg and raises PeelDependencyError with install steps if none load."""
    opts = _options(cfg)["matting"]
    backend = str(opts.get("backend") or "auto").lower()
    device = str(opts.get("device") or "cpu")
    size = opts.get("process_size")
    if backend == "birefnet":
        return make_birefnet_matting(opts["hf_card"], device, size)
    if backend == "rembg":
        return make_rembg_matting()
    if backend != "auto":
        raise PeelDependencyError(f"unknown peel.matting.backend: {backend!r}")
    errors = []
    for card in (opts["hf_card"], opts.get("fallback_hf_card")):
        if not card:
            continue
        try:
            return make_birefnet_matting(card, device, size)
        except Exception as exc:  # noqa: BLE001 - report every backend's failure below
            errors.append(f"transformers[{card}]: {exc}")
    try:
        return make_rembg_matting()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rembg: {exc}")
    raise PeelDependencyError(
        "no matting backend available:\n  " + "\n  ".join(errors)
        + "\ninstall one of:\n"
        "  (a) transformers path — already in requirements-gpu.txt; weights fetch on "
        "first use from HF (cyberagent/layerd-birefnet, ~1 GB)\n"
        "  (b) pip install \"rembg[cpu]\""
    )


# ── the peel loop ──────────────────────────────────────────────────────────────────

def peel_step(rgb, matting: Callable, inpaint: Callable, opts: dict,
              prev_mask=None):
    """One LayerD ``_decompose_step``: matte the top layer, extract RGBA, inpaint hole.

    Returns ``(layer_dict | None, new_background, stop_reason | None)``. ``layer_dict``
    carries rgba/alpha/hard_mask/bbox/area/coverage; ``None`` means a stop condition hit
    and ``new_background`` is the unchanged input.
    """
    _, np, _ = _deps()
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]

    alpha = np.clip(np.asarray(matting(rgb), dtype=np.float64), 0.0, 1.0)
    if alpha.shape != (h, w):
        raise ValueError(f"matting returned {alpha.shape}, expected {(h, w)}")

    hard = alpha > float(opts["alpha_threshold"])
    coverage = float(hard.mean())
    if hard.sum() == 0:                                   # LayerD: "No content"
        return None, rgb, "empty-matte"
    if coverage > float(opts["full_coverage_stop"]):      # LayerD: "Full content"
        return None, rgb, "full-coverage-matte"
    if coverage < float(opts["min_coverage_stop"]):       # ours: speck-sized residue
        return None, rgb, "residual-below-threshold"
    if prev_mask is not None and _mask_iou(hard, prev_mask) >= float(opts["repeat_iou_stop"]):
        # The inpainter failed to actually remove the previous layer, so the matting
        # keeps proposing the same region. Emitting it again would duplicate pixels.
        return None, rgb, "repeat-matte"

    # LayerD _calc_kernel_size: dilate the hole past anti-aliased fringes before inpaint.
    scale = float(opts["kernel_scale"])
    kernel = (max(1, round(h * scale)), max(1, round(w * scale)))
    inpaint_mask = _expand_mask(hard, kernel)

    bg = np.asarray(inpaint(rgb, inpaint_mask), dtype=np.uint8)
    if bg.shape != rgb.shape:
        raise ValueError(f"inpaint returned {bg.shape}, expected {rgb.shape}")

    if opts["unblend"]:
        fg_rgb = estimate_fg_color(rgb, bg, alpha, tuple(opts["unblend_alpha_clip"]))
    else:
        fg_rgb = rgb.copy()
    rgba = np.dstack([fg_rgb, np.round(alpha * 255).astype(np.uint8)])

    layer = {
        "rgba": rgba,
        "alpha": alpha,
        "hard_mask": hard,
        "bbox": _tight_bbox(hard),
        "area": int(hard.sum()),
        "coverage": coverage,
    }
    return layer, bg, None


def peel(image, max_layers: Optional[int] = None, cfg: Optional[dict] = None,
         matting: Optional[Callable] = None,
         inpaint: Optional[Callable] = None) -> PeelResult:
    """Iteratively peel top layers off ``image`` until only background remains.

    Args:
        image: path, PIL image, or HxWx3/4 uint8 array (flattened design).
        max_layers: cap on peel iterations; defaults to ``cfg.peel.max_layers`` (3,
            LayerD's default).
        cfg: full pipeline config dict; only the optional ``peel`` block is read.
        matting: ``rgb -> float alpha``; default resolves BiRefNet per cfg (downloads
            ~1 GB of weights on first use — inject a callable in tests).
        inpaint: ``(rgb, bool_mask) -> rgb``; default is deterministic OpenCV Telea.
            Pass ``make_simple_lama_inpaint()`` for LayerD-faithful quality, or the
            ``src.inpaint`` adapter from docs/PEEL-DECOMPOSITION.md for the pipeline's
            routed Big-LaMa/Flux backends.

    Returns:
        PeelResult with ``layers`` topmost-first, the residual ``background`` plate, and
        ``stop_reason`` in {"empty-matte", "full-coverage-matte",
        "residual-below-threshold", "repeat-matte", "max-layers"}.
    """
    _, np, _ = _deps()
    opts = _options(cfg)
    rounds = int(max_layers if max_layers is not None else opts["max_layers"])
    if rounds < 1:
        raise ValueError(f"max_layers must be >= 1, got {rounds}")
    if matting is None:
        matting = resolve_matting(cfg)
    if inpaint is None:
        inpaint = opencv_inpaint

    current = _to_rgb(image)
    h, w = current.shape[:2]
    layers: list[PeeledLayer] = []
    steps: list[dict] = []
    prev_mask = None
    stop_reason = "max-layers"

    for index in range(rounds):
        layer, current, reason = peel_step(current, matting, inpaint, opts, prev_mask)
        if layer is None:
            stop_reason = reason
            steps.append({"iteration": index, "stop": reason})
            break
        prev_mask = layer["hard_mask"]
        layers.append(PeeledLayer(
            peel_order=index,
            rgba=layer["rgba"],
            bbox=layer["bbox"],
            area=layer["area"],
            coverage=round(layer["coverage"], 6),
        ))
        steps.append({"iteration": index, "stop": None,
                      "coverage": round(layer["coverage"], 6),
                      "bbox": layer["bbox"]})

    return PeelResult(layers=layers, background=current, stop_reason=stop_reason,
                      canvas={"w": w, "h": h}, steps=steps)


# ── artifacts ──────────────────────────────────────────────────────────────────────

def write_outputs(result: PeelResult, out_dir: str) -> dict:
    """Write layer_00.png (topmost) … layer_NN.png, background.png and manifest.json.

    Manifest layers carry both ``peel_order`` (0 = topmost, the order they were peeled)
    and ``z`` (bottom-to-top compositing index over the background: re-compositing
    background + layers sorted by ascending ``z`` reproduces the input, minus inpaint
    error).  Returns the manifest dict.
    """
    _, np, Image = _deps()
    os.makedirs(out_dir, exist_ok=True)
    total = len(result.layers)
    entries = []
    for layer in result.layers:
        name = f"layer_{layer.peel_order:02d}.png"
        Image.fromarray(np.asarray(layer.rgba, dtype=np.uint8)).save(os.path.join(out_dir, name))
        entries.append({
            "file": name,
            "peel_order": layer.peel_order,
            "z": total - layer.peel_order,   # background is z=0; topmost peel is z=total
            "bbox": layer.bbox,
            "area": layer.area,
            "coverage": layer.coverage,
        })
    Image.fromarray(np.asarray(result.background, dtype=np.uint8)).save(
        os.path.join(out_dir, "background.png"))
    manifest = {
        "version": 1,
        "canvas": result.canvas,
        "stop_reason": result.stop_reason,
        "background": {"file": "background.png", "z": 0},
        "layers": entries,
        "steps": result.steps,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_pipeline_layers(result: PeelResult, run_dir: str,
                          subdir: str = "peel_layers") -> list:
    """Publish peeled layers in the pipeline's decomposed-layer shape (see QwenLayer in
    src/schema.py): back-to-front ``{"id": "P<i>", "png": relpath, "box", "kind_hint"}``.

    This is the merge seam — ``merge_layers.merge`` already matches such layers to fused
    element candidates by IoU and hands their clean RGBA to reconstruct via ``src`` (the
    identical path qwen_layers/Q<i>.png takes today).  The background plate is *not*
    published as a layer; reconstruct still owns the canonical background artifact.
    """
    _, np, Image = _deps()
    out_dir = os.path.join(run_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    published = []
    # merge_layers expects back-to-front (lower index = further back): reverse peel order.
    for layer in reversed(result.layers):
        if layer.bbox["w"] <= 0 or layer.bbox["h"] <= 0:
            continue
        index = len(published)
        rel = os.path.join(subdir, f"P{index}.png")
        Image.fromarray(np.asarray(layer.rgba, dtype=np.uint8)).save(
            os.path.join(run_dir, rel))
        published.append({"id": f"P{index}", "png": rel, "box": dict(layer.bbox),
                          "kind_hint": "unknown"})
    return published
