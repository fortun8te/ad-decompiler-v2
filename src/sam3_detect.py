"""SAM 3-first element proposals for flat image decompilation.

This module deliberately sits beside the existing pipeline until its contract has been
validated on the RTX worker.  It uses Meta's official ``facebookresearch/sam3`` image API:

* one shared image embedding;
* an open-vocabulary text-prompt sweep for likely ad elements; and
* a positive geometric (box) prompt for *every* residual proposal.

The public :func:`detect` function is safe to import on machines without torch/SAM.  All
heavy imports are lazy and a missing checkpoint/package returns residual-backed proposals
with saved masks instead of failing the run.

Runtime config (under ``cfg['sam3']``)::

    enabled: true
    checkpoint: C:/models/sam3.pt               # image checkpoint, or SAM3_CHECKPOINT
    bpe_path: C:/src/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
    device: cuda
    load_from_hf: false               # local by default; opt in to downloads
    confidence: 0.45
    resolution: 1008
    compile: false
    prompts:                           # strings or {prompt, role, kind}
      - {prompt: logo, role: logo, kind: icon}

``detect(...)`` returns a manifest with ``elements``.  Every element has a box, semantic
role, score, provenance, and a full-canvas mask saved under ``sam3_masks/``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


DEFAULT_PROMPTS = [
    {"prompt": "person", "role": "person", "kind": "photo-fragment"},
    {"prompt": "face", "role": "person", "kind": "photo-fragment"},
    {"prompt": "hand", "role": "person", "kind": "photo-fragment"},
    {"prompt": "product", "role": "product", "kind": "photo-fragment"},
    {"prompt": "product package", "role": "product", "kind": "photo-fragment"},
    {"prompt": "product shot", "role": "product", "kind": "photo-fragment"},
    {"prompt": "bottle", "role": "product", "kind": "photo-fragment"},
    {"prompt": "jar", "role": "product", "kind": "photo-fragment"},
    {"prompt": "tube", "role": "product", "kind": "photo-fragment"},
    {"prompt": "box", "role": "product", "kind": "photo-fragment"},
    {"prompt": "phone", "role": "product", "kind": "photo-fragment"},
    {"prompt": "logo", "role": "logo", "kind": "icon"},
    {"prompt": "brand logo", "role": "logo", "kind": "icon"},
    {"prompt": "circular logo", "role": "logo", "kind": "icon"},
    {"prompt": "icon", "role": "icon", "kind": "icon"},
    {"prompt": "app icon", "role": "icon", "kind": "icon"},
    {"prompt": "profile picture", "role": "avatar", "kind": "icon"},
    {"prompt": "profile photo", "role": "avatar", "kind": "icon"},
    {"prompt": "avatar", "role": "avatar", "kind": "icon"},
    {"prompt": "arrow", "role": "arrow", "kind": "icon"},
    {"prompt": "badge", "role": "badge", "kind": "icon"},
    {"prompt": "verified badge", "role": "verified", "kind": "icon"},
    {"prompt": "verified checkmark", "role": "verified", "kind": "icon"},
    {"prompt": "price badge", "role": "badge", "kind": "icon"},
    {"prompt": "sale badge", "role": "badge", "kind": "icon"},
    {"prompt": "sticker", "role": "sticker", "kind": "shape"},
    {"prompt": "button", "role": "button", "kind": "shape"},
    {"prompt": "cta button", "role": "button", "kind": "shape"},
    {"prompt": "call to action", "role": "button", "kind": "shape"},
    {"prompt": "card", "role": "card", "kind": "shape"},
    {"prompt": "offer card", "role": "card", "kind": "shape"},
    {"prompt": "illustration", "role": "illustration", "kind": "photo-fragment"},
]

_BACKEND_CACHE = {}


class Sam3Unavailable(RuntimeError):
    """Raised internally when the official local backend cannot be loaded."""


def _sam_cfg(cfg: Optional[dict]) -> dict:
    cfg = cfg or {}
    if not cfg.get("sam3"):
        return dict(cfg)
    out = dict(cfg.get("sam3") or {})
    # Match the repo's existing convention where device lives at the config root.
    if "device" not in out and cfg.get("device") is not None:
        out["device"] = cfg["device"]
    return out


def _prompt_specs(raw) -> list[dict]:
    raw = DEFAULT_PROMPTS if raw is None else raw
    specs = []
    for item in raw or []:
        if isinstance(item, str):
            role = item.strip().lower().replace(" ", "-")
            specs.append({"prompt": item, "role": role, "kind": _kind_for_role(role)})
            continue
        if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
            continue
        prompt = str(item["prompt"]).strip()
        role = str(item.get("role") or prompt).strip().lower().replace(" ", "-")
        specs.append(
            {
                "prompt": prompt,
                "role": role,
                "kind": item.get("kind") or _kind_for_role(role),
            }
        )
    return specs


def _kind_for_role(role: str) -> str:
    role = str(role or "").lower()
    if role in {"logo", "icon", "arrow", "badge", "symbol", "pictogram",
                "avatar", "profile", "profile-picture", "verified",
                "verified-badge", "checkmark"}:
        return "icon"
    if role in {"shape", "button", "card", "container", "sticker", "background"}:
        return "shape"
    return "photo-fragment"


def _role_from_residual(item: dict) -> str:
    if item.get("role"):
        return str(item["role"])
    return {
        "icon": "icon",
        "shape": "shape",
        "photo-fragment": "photo",
    }.get(item.get("kind"), "object")


def _valid_box(box: Any) -> bool:
    return bool(
        isinstance(box, dict)
        and float(box.get("w", 0) or 0) > 0
        and float(box.get("h", 0) or 0) > 0
    )


def _clip_box(box: dict, width: int, height: int) -> dict:
    x_raw = float(box.get("x", 0))
    y_raw = float(box.get("y", 0))
    w_raw = float(box.get("w", 0))
    h_raw = float(box.get("h", 0))
    x0 = max(0.0, min(float(width), x_raw))
    y0 = max(0.0, min(float(height), y_raw))
    # x1/y1 must derive from the ORIGINAL x+w/y+h, not from the already-clipped x0/y0 --
    # otherwise an off-canvas origin (e.g. x=-20) gets clipped to x0=0 and then the box
    # keeps its full original width added back on top, inflating w/h for any box with a
    # negative or off-canvas origin.
    x1 = max(x0, min(float(width), x_raw + w_raw))
    y1 = max(y0, min(float(height), y_raw + h_raw))
    return {
        "x": int(round(x0)),
        "y": int(round(y0)),
        "w": max(0, int(round(x1 - x0))),
        "h": max(0, int(round(y1 - y0))),
    }


def _as_numpy(value):
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        try:
            value = value.float()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _mask_stack(value, width: int, height: int):
    import numpy as np

    arr = _as_numpy(value)
    if arr is None or arr.size == 0:
        return np.zeros((0, height, width), dtype=bool)
    while arr.ndim > 3 and 1 in arr.shape[1:]:
        # Official Sam3Processor currently returns N x 1 x H x W.
        axis = next(i for i in range(1, arr.ndim) if arr.shape[i] == 1)
        arr = np.squeeze(arr, axis=axis)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return np.zeros((0, height, width), dtype=bool)
    if arr.shape[1:] != (height, width):
        from PIL import Image

        resized = []
        for m in arr:
            u8 = ((m > 0.5) * 255).astype(np.uint8)
            resized.append(
                np.asarray(Image.fromarray(u8).resize((width, height), Image.Resampling.NEAREST))
                > 0
            )
        arr = np.stack(resized) if resized else np.zeros((0, height, width), bool)
    return arr > 0.5


def _boxes(value) -> list[dict]:
    arr = _as_numpy(value)
    if arr is None or arr.size == 0:
        return []
    arr = arr.reshape(-1, 4)
    out = []
    for x0, y0, x1, y1 in arr.tolist():
        out.append({"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)})
    return out


def _scores(value) -> list[float]:
    arr = _as_numpy(value)
    if arr is None or arr.size == 0:
        return []
    return [float(v) for v in arr.reshape(-1).tolist()]


def _mask_box(mask) -> dict:
    import numpy as np

    ys, xs = np.where(mask)
    if not xs.size:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def _rect_mask(box: dict, width: int, height: int):
    import numpy as np

    mask = np.zeros((height, width), dtype=bool)
    b = _clip_box(box, width, height)
    if b["w"] and b["h"]:
        mask[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]] = True
    return mask


def _prediction_dicts(raw, width: int, height: int) -> list[dict]:
    """Normalize an official processor state or a test/backend list into predictions."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            mask = item.get("mask")
            if mask is None and item.get("masks") is not None:
                stack = _mask_stack(item.get("masks"), width, height)
                mask = stack[0] if len(stack) else None
            if mask is not None:
                stack = _mask_stack(mask, width, height)
                mask = stack[0] if len(stack) else None
            box = item.get("box")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x0, y0, x1, y1 = [float(v) for v in box]
                box = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            out.append({"mask": mask, "box": box, "score": float(item.get("score", 1.0))})
        return out
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get("elements"), list):
        return _prediction_dicts(raw["elements"], width, height)
    masks = _mask_stack(raw.get("masks"), width, height)
    boxes = _boxes(raw.get("boxes"))
    scores = _scores(raw.get("scores"))
    n = max(len(masks), len(boxes), len(scores))
    out = []
    for i in range(n):
        out.append(
            {
                "mask": masks[i] if i < len(masks) else None,
                "box": boxes[i] if i < len(boxes) else None,
                "score": scores[i] if i < len(scores) else 1.0,
            }
        )
    return out


class _OfficialSam3Backend:
    """Thin adapter around Meta's official image processor API."""

    name = "facebookresearch/sam3"

    def __init__(self, cfg: dict):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise Sam3Unavailable(
                "official SAM 3 package is not installed (clone facebookresearch/sam3 and pip install -e .)"
            ) from exc

        checkpoint = (
            cfg.get("checkpoint")
            or cfg.get("checkpoint_path")
            or os.environ.get("SAM3_CHECKPOINT")
        )
        load_from_hf = bool(cfg.get("load_from_hf", False))
        if not checkpoint and load_from_hf:
            try:
                from huggingface_hub import hf_hub_download

                checkpoint = hf_hub_download(
                    repo_id=str(cfg.get("repo_id", "facebook/sam3")),
                    filename=str(cfg.get("checkpoint_filename", "sam3.pt")),
                )
            except Exception as exc:
                raise Sam3Unavailable(f"SAM 3 checkpoint download failed: {exc}") from exc
        if checkpoint and not os.path.exists(os.path.expanduser(str(checkpoint))):
            raise Sam3Unavailable(f"SAM 3 checkpoint not found: {checkpoint}")
        if not checkpoint and not load_from_hf:
            raise Sam3Unavailable(
                "no local SAM 3 image checkpoint configured (set sam3.checkpoint or SAM3_CHECKPOINT)"
            )
        checkpoint = os.path.expanduser(str(checkpoint)) if checkpoint else None
        device = str(cfg.get("device", "cuda"))
        kwargs = {
            "device": device,
            "eval_mode": True,
            "checkpoint_path": checkpoint,
            # Resolve the image checkpoint explicitly so production runs stay local-only.
            "load_from_HF": False,
            "enable_segmentation": True,
            "compile": bool(cfg.get("compile", False)),
        }
        if cfg.get("bpe_path"):
            kwargs["bpe_path"] = os.path.expanduser(str(cfg["bpe_path"]))
        try:
            model = build_sam3_image_model(**kwargs)
            self.processor = Sam3Processor(
                model,
                resolution=int(cfg.get("resolution", 1008)),
                device=device,
                confidence_threshold=float(cfg.get("confidence", 0.45)),
            )
        except Exception as exc:
            raise Sam3Unavailable(f"SAM 3 model load failed: {exc}") from exc
        self.device = device
        self.state = None
        self.size = None

    def _autocast(self):
        # SAM 3's vision backbone runs its activations in bfloat16 while the
        # checkpoint weights load as float32; without an autocast context the two
        # meet in an F.linear and raise "mat1 and mat2 must have the same dtype"
        # (BFloat16 vs Float). Every official example enters
        # torch.autocast("cuda", dtype=bfloat16) before set_image/predict — mirror
        # that here so the matmuls agree. No-op (nullcontext) on CPU.
        import torch
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def set_image(self, image):
        self.size = image.size
        with self._autocast():
            self.state = self.processor.set_image(image)

    def predict_text(self, prompt: str):
        self.processor.reset_all_prompts(self.state)
        with self._autocast():
            return self.processor.set_text_prompt(state=self.state, prompt=prompt)

    def predict_box(self, box: dict):
        if not self.size:
            raise RuntimeError("set_image must be called first")
        width, height = self.size
        cx = (float(box["x"]) + float(box["w"]) / 2.0) / max(1, width)
        cy = (float(box["y"]) + float(box["h"]) / 2.0) / max(1, height)
        bw = float(box["w"]) / max(1, width)
        bh = float(box["h"]) / max(1, height)
        self.processor.reset_all_prompts(self.state)
        with self._autocast():
            return self.processor.add_geometric_prompt(
                state=self.state, box=[cx, cy, bw, bh], label=True
            )


def unload_backend() -> None:
    """Release cached SAM3 backends so CUDA memory can be reclaimed between stages."""
    _BACKEND_CACHE.clear()


def _cached_official_backend(cfg: dict):
    """Reuse the 848M model/processor across images; only the image state is replaced."""
    key = (
        os.path.abspath(os.path.expanduser(str(cfg.get("checkpoint") or cfg.get("checkpoint_path")
                                              or os.environ.get("SAM3_CHECKPOINT") or "hf"))),
        str(cfg.get("device", "cuda")),
        int(cfg.get("resolution", 1008)),
        bool(cfg.get("compile", False)),
        str(cfg.get("bpe_path") or ""),
    )
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        backend = _OfficialSam3Backend(cfg)
        _BACKEND_CACHE[key] = backend
    return backend


def _mask_iou_box(mask, box: dict) -> float:
    rect = _rect_mask(box, mask.shape[1], mask.shape[0])
    inter = int((mask & rect).sum())
    union = int((mask | rect).sum())
    return inter / union if union else 0.0


def _load_residual_mask(item: dict, width: int, height: int, base_dir: Optional[str]):
    import numpy as np

    if item.get("_mask") is not None:
        arr = _as_numpy(item["_mask"])
        if arr is not None:
            if arr.shape == (height, width):
                return arr > 0
            b = _clip_box(item.get("box") or {}, width, height)
            if arr.shape == (b["h"], b["w"]):
                full = np.zeros((height, width), dtype=bool)
                full[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]] = arr > 0
                return full
    candidates = [item.get("mask_path"), item.get("mask_src")]
    if isinstance(item.get("mask"), dict):
        candidates.append(item["mask"].get("src"))
    for raw in candidates:
        if not raw:
            continue
        path = str(raw)
        if not os.path.isabs(path) and base_dir:
            path = os.path.join(base_dir, path)
        if not os.path.exists(path):
            continue
        try:
            from PIL import Image

            with Image.open(path) as image:
                # Residual cutouts are frequently RGBA. Their transparent RGB is
                # undefined (often solid white), so luminance would turn the whole
                # canvas into an object. Match the inpaint boundary: alpha owns
                # transparency whenever it exists.
                has_alpha = "A" in image.getbands() or (
                    image.mode == "P" and "transparency" in image.info
                )
                arr = np.asarray(
                    image.convert("RGBA").getchannel("A") if has_alpha else image.convert("L")
                ) > 0
            if arr.shape == (height, width):
                return arr
            b = _clip_box(item.get("box") or {}, width, height)
            if b["w"] and b["h"]:
                arr = np.asarray(
                    Image.fromarray((arr * 255).astype(np.uint8)).resize(
                        (b["w"], b["h"]), Image.Resampling.NEAREST
                    )
                ) > 0
                full = np.zeros((height, width), dtype=bool)
                full[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]] = arr
                return full
        except Exception:
            continue
    return _rect_mask(item.get("box") or {}, width, height)


def _write_mask(mask, path: str) -> None:
    import numpy as np
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def _make_element(
    idx: int,
    mask,
    role: str,
    kind: str,
    score: float,
    width: int,
    height: int,
    run_dir: Optional[str],
    provenance: dict,
    source: str = "sam3",
) -> Optional[dict]:
    area = int(mask.sum())
    if area <= 0:
        return None
    box = _mask_box(mask)
    sid = f"S{idx:03d}"
    rel = os.path.join("sam3_masks", f"{sid}.png")
    path = os.path.join(run_dir, rel) if run_dir else None
    if path:
        _write_mask(mask, path)
    return {
        "id": sid,
        "box": box,
        "kind": kind or _kind_for_role(role),
        "role": role,
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "area": float(area),
        "coverage": round(area / max(1, width * height), 6),
        "source": source,
        "mask": {"kind": "alpha", "src": rel} if run_dir else {"kind": "alpha"},
        "mask_src": rel if run_dir else None,
        "mask_path": os.path.abspath(path) if path else None,
        "provenance": provenance,
    }


def _text_min_score(scfg: dict) -> float:
    return float(scfg.get("min_score", scfg.get("confidence", 0.45)))


# Small circular logos, avatars, and verified badges routinely land just below the
# generic text-prompt bar (the ad9 verified badge scored 0.539).  For roughly-square,
# small (sub-canvas) predictions coming from an icon/logo/badge/avatar prompt we accept
# a lower score.  Large regions keep the full bar, so this cannot rescue a prompt that
# snapped to a whole photograph.
_SMALL_ICON_ROLES = {
    "avatar", "profile", "profile-picture", "logo", "badge", "icon",
    "verified", "verified-badge", "checkmark", "symbol",
}


def _small_icon_cfg(scfg: dict) -> dict:
    raw = scfg.get("small_icon")
    if raw is None:
        return {"enabled": True}
    if isinstance(raw, bool):
        return {"enabled": raw}
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("enabled", True)
        return out
    return {"enabled": False}


def _text_pred_threshold(scfg: dict, role: str, mask, width: int, height: int,
                         base: float) -> float:
    cfg = _small_icon_cfg(scfg)
    if not cfg.get("enabled", True):
        return base
    roles = cfg.get("roles")
    roles = set(roles) if roles else _SMALL_ICON_ROLES
    if str(role) not in roles:
        return base
    box = _mask_box(mask)
    if box["w"] <= 0 or box["h"] <= 0:
        return base
    coverage = (box["w"] * box["h"]) / max(1, width * height)
    if coverage > float(cfg.get("max_coverage", 0.05)):
        return base
    aspect = box["w"] / max(1, box["h"])
    if not (float(cfg.get("min_aspect", 0.6)) <= aspect <= float(cfg.get("max_aspect", 1.7))):
        return base
    return min(base, float(cfg.get("min_score", 0.30)))


def _box_refine_min_score(scfg: dict, residual: list) -> float:
    """Box-refine keeps a lower acceptance bar when deterministic residuals exist."""
    if scfg.get("box_refine_confidence") is not None:
        return float(scfg["box_refine_confidence"])
    if residual:
        return 0.32
    return float(scfg.get("confidence", 0.45))


def _acceptable_refinement(mask, box: dict, scfg: dict) -> bool:
    """Reject confident but geometrically unrelated box-prompt masks.

    SAM confidence alone is not enough for ad decomposition: a prompt can snap to a
    nearby large photograph.  Box overlap and area expansion keep the deterministic
    residual as the safe fallback without fabricating ownership.
    """
    import numpy as np

    area = int(np.count_nonzero(mask))
    box_area = max(1.0, float(box.get("w", 0)) * float(box.get("h", 0)))
    if area <= 0:
        return False
    if area / box_area > float(scfg.get("max_box_area_ratio", 4.0)):
        return False
    return _mask_iou_box(mask, box) >= float(scfg.get("min_box_iou", 0.12))


def _residual_ids_in_elements(elements: list) -> set[str]:
    covered = set()
    for element in elements or []:
        prov = element.get("provenance") or {}
        residual_id = prov.get("residual_id")
        if residual_id is not None:
            covered.add(str(residual_id))
    return covered


def _union_residual_guarantees(
    elements: list,
    residual: list,
    width: int,
    height: int,
    run_dir: Optional[str],
    note: str,
) -> list[dict]:
    """Append residual-backed observations for any proposal SAM did not emit."""
    out = list(elements or [])
    covered = _residual_ids_in_elements(out)
    for item in residual:
        residual_id = str(item.get("id")) if item.get("id") is not None else None
        if residual_id is not None and residual_id in covered:
            continue
        if not _valid_box(item.get("box")):
            continue
        role = _role_from_residual(item)
        mask = _load_residual_mask(item, width, height, run_dir)
        el = _make_element(
            len(out),
            mask,
            role,
            item.get("kind") or _kind_for_role(role),
            float(item.get("score", item.get("confidence", 0.35)) or 0.35),
            width,
            height,
            run_dir,
            {
                "model": "sam3",
                "mode": "residual-fallback",
                "residual_id": item.get("id"),
                "reason": note,
            },
            source="residual-fallback",
        )
        if el:
            out.append(el)
            if residual_id is not None:
                covered.add(residual_id)
    return out


def _fallback_elements(
    residual: list,
    width: int,
    height: int,
    run_dir: Optional[str],
    note: str,
) -> list[dict]:
    out = []
    for item in residual:
        if not _valid_box(item.get("box")):
            continue
        role = _role_from_residual(item)
        mask = _load_residual_mask(item, width, height, run_dir)
        el = _make_element(
            len(out),
            mask,
            role,
            item.get("kind") or _kind_for_role(role),
            float(item.get("score", item.get("confidence", 0.35)) or 0.35),
            width,
            height,
            run_dir,
            {
                "model": "sam3",
                "mode": "residual-fallback",
                "residual_id": item.get("id"),
                "reason": note,
            },
            source="residual-fallback",
        )
        if el:
            out.append(el)
    return out


def detect(
    img_path: str,
    residual: Optional[list] = None,
    cfg: Optional[dict] = None,
    run_dir: Optional[str] = None,
    backend=None,
) -> dict:
    """Run SAM text proposals plus box refinement for every residual proposal.

    ``backend`` is an intentionally small injection seam used by CPU tests and alternate
    workers.  It must provide ``set_image(image)``, ``predict_text(prompt)`` and
    ``predict_box(box_dict)``.
    """
    from PIL import Image

    residual = list(residual or [])
    scfg = _sam_cfg(cfg)
    with Image.open(img_path) as src:
        image = src.convert("RGB")
    width, height = image.size
    prompts = _prompt_specs(scfg.get("prompts"))
    errors = []

    if run_dir:
        os.makedirs(os.path.join(run_dir, "sam3_masks"), exist_ok=True)

    if not scfg.get("enabled", True):
        note = "SAM 3 disabled; residual fallback used"
        elements = _fallback_elements(residual, width, height, run_dir, note)
        result = {
            "engine": "residual-fallback",
            "status": "fallback",
            "note": note,
            "source": {"path": img_path, "w": width, "h": height},
            "prompts": [p["prompt"] for p in prompts],
            "elements": elements,
        }
        _write_manifest(result, run_dir)
        return result

    try:
        backend = backend or _cached_official_backend(scfg)
        backend.set_image(image)
    except Exception as exc:
        note = str(exc)
        elements = _fallback_elements(residual, width, height, run_dir, note)
        result = {
            "engine": "residual-fallback",
            "status": "fallback",
            "note": note,
            "source": {"path": img_path, "w": width, "h": height},
            "prompts": [p["prompt"] for p in prompts],
            "elements": elements,
        }
        _write_manifest(result, run_dir)
        return result

    elements = []
    min_score = _text_min_score(scfg)
    box_min_score = _box_refine_min_score(scfg, residual)
    text_prompt_successes = 0
    text_prompt_predictions = 0
    box_prompt_successes = 0
    box_prompt_predictions = 0

    # Open-vocabulary sweep. Duplicates are intentionally retained as observations and
    # resolved once, mask-aware, by element_fusion.fuse().
    for spec in prompts:
        try:
            preds = _prediction_dicts(backend.predict_text(spec["prompt"]), width, height)
            text_prompt_successes += 1
            text_prompt_predictions += len(preds)
        except Exception as exc:
            errors.append(f"text:{spec['prompt']}: {exc}")
            continue
        for pred in preds:
            # A SAM score/box without a segmentation mask is not ownership evidence.
            # Do not promote the model box into a fabricated rectangular owner.
            mask = pred.get("mask")
            if mask is None:
                continue
            # Small square avatars/badges/logos get a lower acceptance bar; every other
            # prediction keeps the generic min_score.
            threshold = _text_pred_threshold(scfg, spec["role"], mask, width, height, min_score)
            if float(pred.get("score", 0)) < threshold:
                continue
            el = _make_element(
                len(elements),
                mask,
                spec["role"],
                spec["kind"],
                pred.get("score", 0),
                width,
                height,
                run_dir,
                {
                    "model": "sam3",
                    "api": "facebookresearch/sam3 Sam3Processor.set_text_prompt",
                    "mode": "text-prompt",
                    "prompt": spec["prompt"],
                    "model_box": pred.get("box"),
                },
            )
            if el:
                elements.append(el)

    # Positive box-refine every deterministic residual proposal. One observation is emitted
    # for every valid residual even when SAM finds nothing, preserving recall and provenance.
    for item in residual:
        box = item.get("box") or {}
        if not _valid_box(box):
            continue
        role = _role_from_residual(item)
        best = None
        try:
            preds = _prediction_dicts(backend.predict_box(box), width, height)
            box_prompt_successes += 1
            box_prompt_predictions += len(preds)
            for pred in preds:
                if float(pred.get("score", 0)) < box_min_score:
                    continue
                # Box refinement is accepted only when SAM actually returned a mask;
                # a box-only response must fall through to the deterministic residual
                # observation below.
                mask = pred.get("mask")
                if mask is None:
                    continue
                if not _acceptable_refinement(mask, box, scfg):
                    continue
                quality = 0.7 * float(pred.get("score", 0)) + 0.3 * _mask_iou_box(mask, box)
                if best is None or quality > best[0]:
                    best = (quality, pred, mask)
        except Exception as exc:
            errors.append(f"box:{item.get('id')}: {exc}")

        if best is None:
            mask = _load_residual_mask(item, width, height, run_dir)
            score = float(item.get("score", item.get("confidence", 0.35)) or 0.35)
            mode = "box-refine-fallback"
            model_box = None
        else:
            _, pred, mask = best
            score = float(pred.get("score", 0))
            mode = "box-refine"
            model_box = pred.get("box")
        el = _make_element(
            len(elements),
            mask,
            role,
            item.get("kind") or _kind_for_role(role),
            score,
            width,
            height,
            run_dir,
            {
                "model": "sam3",
                "api": "facebookresearch/sam3 Sam3Processor.add_geometric_prompt",
                "mode": mode,
                "residual_id": item.get("id"),
                "input_box": _clip_box(box, width, height),
                "model_box": model_box,
            },
            source="sam3" if best is not None else "residual-fallback",
        )
        if el:
            elements.append(el)

    elements = _union_residual_guarantees(
        elements,
        residual,
        width,
        height,
        run_dir,
        "residual proposal missing from SAM observations",
    )

    # A backend that accepts calls but emits no masks for every text prompt used to look
    # perfectly healthy.  Preserve the residual observations, but make the empty model
    # evidence explicit so acceptance policy and repair logic can react to it.
    model_element_count = sum(1 for item in elements if item.get("source") == "sam3")
    empty_model_evidence = model_element_count == 0
    status = "ok" if not errors and not empty_model_evidence else "partial"
    notes = list(errors[:8])
    if empty_model_evidence:
        notes.append("SAM 3 returned no accepted segmentation masks; residual observations preserved")
    result = {
        "engine": getattr(backend, "name", "sam3"),
        "status": status,
        "note": "; ".join(notes) if notes else None,
        "source": {"path": img_path, "w": width, "h": height},
        "prompts": [p["prompt"] for p in prompts],
        "elements": elements,
        "thresholds": {
            "text_min_score": min_score,
            "box_refine_min_score": box_min_score,
        },
        "diagnostics": {
            "text_prompts_attempted": len(prompts),
            "text_prompts_succeeded": text_prompt_successes,
            "text_predictions": text_prompt_predictions,
            "box_prompts_attempted": sum(1 for item in residual if _valid_box(item.get("box"))),
            "box_prompts_succeeded": box_prompt_successes,
            "box_predictions": box_prompt_predictions,
            "model_elements": model_element_count,
            "residual_fallback_elements": sum(
                1 for item in elements if item.get("source") == "residual-fallback"
            ),
            "empty_model_evidence": empty_model_evidence,
        },
    }
    _write_manifest(result, run_dir)
    return result


def _write_manifest(result: dict, run_dir: Optional[str]) -> None:
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sam3.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


propose_elements = detect
