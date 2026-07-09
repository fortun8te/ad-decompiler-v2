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


# ── OCR (ocr.py) ───────────────────────────────────────────────────────────────────
@dataclass
class OcrWord:
    text: str
    conf: float
    box: dict          # {x,y,w,h}
    quad: list         # [[x,y]*4] TL,TR,BR,BL

@dataclass
class OcrLine:
    id: str            # "L0".. reading order
    text: str
    conf: float
    box: dict
    quad: list
    words: list = field(default_factory=list)

@dataclass
class OcrResult:
    engine: str        # "ppocr-v6" | "surya" | "doctr" | "tesseract"
    source: dict       # {path,w,h}
    ms: float
    lines: list = field(default_factory=list)


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
#   group -> Figma FRAME/GROUP (children carry ABSOLUTE canvas coords)
LayerType = Literal["text", "shape", "image", "group"]

@dataclass
class Layer:
    id: str
    type: LayerType
    name: str                      # semantic, e.g. 'Headline — "THE 10 SEC…"', 'Product — cutout'
    box: dict                      # {x,y,w,h}
    # text
    text: Optional[str] = None
    style: dict = field(default_factory=dict)   # fontFamily,fontSize,fontWeight,color,align,lineHeight,uppercase,letterSpacing
    # shape / vector
    shape_kind: Optional[str] = None            # rect|ellipse|path
    path: Optional[str] = None                  # SVG d-string when shape_kind=path (VTracer/Potrace)
    fill: Optional[dict] = None                 # {kind:flat|linear|radial,color|stops|angle}
    # image
    src: Optional[str] = None                   # relative asset path (cutout/qwen png)
    mask: Optional[dict] = None                 # {kind:ellipse|rrect|path, radius?, path?}
    # effects (shared)
    effects: list = field(default_factory=list) # [{type:drop-shadow|blur,...}]
    # provenance (agent + QA read this; NOT exported to Figma)
    meta: dict = field(default_factory=dict)    # {source, role, confidence, wordmark, kept_in_photo,...}
    children: list = field(default_factory=list)

@dataclass
class DesignDoc:
    id: str
    name: str
    canvas: dict                   # {w,h}
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


def dump(obj: Any, path: str) -> None:
    """Write any dataclass (or dict/list of them) to pretty JSON."""
    def enc(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=enc)


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
