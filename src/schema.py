"""schema.py — the JSON contracts that flow between pipeline stages.

design.json is the SOURCE OF TRUTH. Every stage reads/writes plain dicts matching
these dataclasses so the orchestrating agent can inspect and repair over JSON without
touching pixels. Coordinates are ALWAYS source-image pixels, origin top-left, unless a
field name ends in `Pct`.

Stage artifacts (written under runs/<run_id>/):
  normalized.png   ocr.json        elements.json    qwen_layers/*.png + qwen.json
  design.json      figma_export.png diff.png         qa.json
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import json
import os
import tempfile

SCHEMA_VERSION = 2


# ── OCR (ocr.py) ───────────────────────────────────────────────────────────────────
@dataclass
class OcrWord:
    text: str
    conf: float
    box: dict          # {x,y,w,h}
    quad: list         # [[x,y]*4] TL,TR,BR,BL
    meta: dict = field(default_factory=dict)

@dataclass
class OcrLine:
    id: str            # "L0".. reading order
    text: str
    conf: float
    box: dict
    quad: list
    words: list = field(default_factory=list)
    # The OCR detector box is not the same thing as the visible painted glyph box.
    # Keeping both is essential for accurate Figma text fitting.
    ink_box: Optional[dict] = None
    baseline: Optional[float] = None
    rotation: float = 0.0
    style: dict = field(default_factory=dict)
    block_id: Optional[str] = None
    role: str = "text"
    meta: dict = field(default_factory=dict)


@dataclass
class OcrBlock:
    id: str
    text: str
    box: dict
    line_ids: list = field(default_factory=list)
    style: dict = field(default_factory=dict)
    role: str = "body"
    hierarchy: int = 0
    repeated_style_id: Optional[str] = None
    meta: dict = field(default_factory=dict)

@dataclass
class OcrResult:
    engine: str        # "ppocr-v6" | "surya" | "doctr" | "tesseract"
    source: dict       # {path,w,h}
    ms: float
    lines: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    styles: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ── Element detection (element_detect.py) ─────────────────────────────────────────
ElementKind = Literal["shape", "icon", "photo-fragment"]

@dataclass
class Element:
    id: str            # "E0"..
    box: dict          # {x,y,w,h}
    kind: ElementKind
    area: float
    coverage: float = 0.0
    source: str = "residual-cc"   # or "qwen"
    mask: Optional[str] = None
    score: float = 0.0
    role: str = "unknown"
    prompt: Optional[str] = None
    parent_id: Optional[str] = None
    observation_ids: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ── Qwen-Image-Layered (qwen_worker.py) ───────────────────────────────────────────
@dataclass
class QwenLayer:
    id: str            # "Q0".. back-to-front
    png: str           # relative path to RGBA png under qwen_layers/
    box: dict          # tight bbox of non-transparent content
    kind_hint: str = "unknown"   # qwen's semantic guess, advisory only


# ── design.json (build_design_json.py) — THE SOURCE OF TRUTH ──────────────────────
# Layer.type routes to Figma:
#   text  -> Figma TEXT node (editable)
#   shape -> Figma primitive (RECT/ELLIPSE) or VECTOR (shapeKind=path with `path` d-string)
#   image -> raster fill, optionally clipped by `mask` (ellipse/rrect/path)
#   group -> Figma FRAME/GROUP (children carry PARENT-RELATIVE ("local") coords —
#            layout.py::_relativize subtracts the parent's absolute box before writing;
#            render_preview.py and figma-plugin/code.js both consume "local" coords,
#            and design.json's meta stamps coordinate_space="local".)
LayerType = Literal["text", "shape", "image", "group"]

@dataclass
class Layer:
    id: str
    type: LayerType
    name: str                      # semantic, e.g. 'Headline — "THE 10 SEC…"', 'Product — cutout'
    box: dict                      # {x,y,w,h}
    z_index: float = 0.0
    visible_box: Optional[dict] = None
    rotation: float = 0.0
    opacity: float = 1.0
    blend_mode: str = "NORMAL"
    # text
    text: Optional[str] = None
    style: dict = field(default_factory=dict)   # fontFamily,fontSize,fontWeight,color,align,lineHeight,uppercase,letterSpacing
    text_runs: list = field(default_factory=list)
    # shape / vector
    shape_kind: Optional[str] = None            # rect|ellipse|path
    path: Optional[str] = None                  # SVG d-string when shape_kind=path (VTracer/Potrace)
    svg: Optional[str] = None                   # complete multi-path SVG; preferred for icons/logos
    fill: Optional[dict] = None                 # {kind:flat|linear|radial,color|stops|angle}
    stroke: Optional[dict] = None
    radius: Any = None
    # image
    src: Optional[str] = None                   # relative asset path (cutout/qwen png)
    mask: Optional[dict] = None                 # {kind:ellipse|rrect|path, radius?, path?}
    # effects (shared)
    effects: list = field(default_factory=list) # [{type:drop-shadow|blur,...}]
    # provenance (agent + QA read this; NOT exported to Figma)
    meta: dict = field(default_factory=dict)    # {source, role, confidence, wordmark, kept_in_photo,...}
    children: list = field(default_factory=list)
    layout: dict = field(default_factory=dict)  # Figma auto-layout intent, only when confidence is high
    constraints: dict = field(default_factory=dict)
    component: dict = field(default_factory=dict)

@dataclass
class DesignDoc:
    id: str
    name: str
    canvas: dict                   # {w,h}
    schema_version: int = SCHEMA_VERSION
    layers: list = field(default_factory=list)
    kept_in_photo: list = field(default_factory=list)  # scene-text strings left baked in the base
    meta: dict = field(default_factory=dict)


# ── QA (pixel_diff.py / repair.py) ────────────────────────────────────────────────
@dataclass
class QaResult:
    ok: bool
    composite: float               # 0..100
    ssim: float
    text_recall: float             # fraction of source OCR strings present in the render
    hard_fails: list = field(default_factory=list)   # [{rule,detail}]
    per_layer: list = field(default_factory=list)
    repairs: list = field(default_factory=list)      # agent-actionable suggestions


# ── minimal shape/type validation at the design.json write boundary ───────────────
# Not a full JSON-schema validator. design.json is consumed by every downstream stage
# (figma_import, render_preview, pixel_diff, repair) with zero validation on read, so a
# structurally broken document (missing ids, bad layer types, non-numeric geometry) would
# otherwise propagate silently until something dereferences a bad field far from the cause.
_REQUIRED_LAYER_FIELDS = ("id", "type", "box")
_VALID_LAYER_TYPES = {"text", "shape", "image", "group"}
_BOX_KEYS = ("x", "y", "w", "h")


def _validate_layers(layers: Any, path: str, errors: list) -> None:
    if not isinstance(layers, list):
        errors.append(f"'{path}' must be a list")
        return
    for i, layer in enumerate(layers):
        here = f"{path}[{i}]"
        if not isinstance(layer, dict):
            errors.append(f"{here} is not an object")
            continue
        for field_name in _REQUIRED_LAYER_FIELDS:
            if layer.get(field_name) in (None, ""):
                errors.append(f"{here} missing '{field_name}'")
        ltype = layer.get("type")
        if ltype is not None and ltype not in _VALID_LAYER_TYPES:
            errors.append(f"{here} has unknown type {ltype!r}")
        box = layer.get("box")
        if box is not None:
            if not isinstance(box, dict) or any(
                not isinstance(box.get(k), (int, float)) for k in _BOX_KEYS
            ):
                errors.append(f"{here}.box must have numeric x/y/w/h")
        children = layer.get("children")
        if children:
            _validate_layers(children, f"{here}.children", errors)


def validate_design(doc: Any) -> list[str]:
    """Minimal structural check for a design.json document (dict or DesignDoc).

    Returns a list of human-readable error strings; an empty list means the document
    passed the shape/type check. This intentionally does not validate style/effect
    payloads or Figma-specific semantics — just enough to catch a broken document
    before it is written and silently trusted by every downstream consumer.
    """
    if hasattr(doc, "__dataclass_fields__"):
        doc = asdict(doc)
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["design.json root must be an object"]
    if not doc.get("id"):
        errors.append("missing top-level 'id'")
    canvas = doc.get("canvas")
    if not isinstance(canvas, dict) or any(
        not isinstance(canvas.get(k), (int, float)) or canvas.get(k) <= 0 for k in ("w", "h")
    ):
        errors.append("canvas must be an object with positive numeric w/h")
    _validate_layers(doc.get("layers"), "layers", errors)
    return errors


def dump(obj: Any, path: str) -> None:
    """Atomically write any dataclass (or dict/list of them) to pretty JSON.

    Pipeline artifacts are resume checkpoints. A killed worker must leave either the prior
    complete checkpoint or the new complete checkpoint, never half a JSON document.
    """
    def enc(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=enc)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
