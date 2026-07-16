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


# ── Confidence-gated raster-slice fallback (Codia-style fidelity floor) ────────────
# A layer whose rendered region no longer matches the source is replaced by a
# pixel-exact image slice of the ORIGINAL pixels (alpha = exactly the pixels the
# removal ledger inpainted out for that layer, so the slice covers the hole and
# nothing else — no double rendering, no background leakage). The failed editable
# attempt is preserved under meta["fallback_editable"] for future repair, and the
# slice is marked meta["fallback"] = "raster-slice". Slices honestly reduce
# editable_ratio; QA reports them instead of shipping a wrong reconstruction.
# Shared by reconstruct.apply_raster_slice_fallback, pixel_diff per-layer region
# scoring, and repair.assess (stdlib-only on purpose: repair.py must stay
# import-safe anywhere).
RASTER_SLICE_FALLBACK_DEFAULTS = {
    "enabled": True,
    # Local crop SSIM gate for every emitted foreground leaf (text/shape/image),
    # plus a local colour gate (grayscale SSIM is blind to hue swaps on flat fills).
    "region_ssim_min": 0.58,
    "region_color_min": 0.75,
    # Text-only ink gates: hard IoU floor, plus a ghost gate (moderate IoU with a
    # large amount of extra rendered ink = double text / leak-through).
    "text_ink_iou_min": 0.30,
    "text_ink_iou_soft": 0.55,
    "text_ink_excess_max": 0.75,
    # F6 — combined "broken text render" gate. Text is normally judged on ink metrics
    # only (a plausible font at the right size/weight/spacing renders with a few-pixel
    # offset that would fail crop SSIM but is trivially repairable in Figma). But a
    # render that is BOTH structurally wrong (very low region_ssim) AND carries a large
    # amount of extra rendered ink is a wrong-CLASS font producing visible garbage, not
    # an offset. Both conditions must hold together, and the bar is deliberately
    # conservative: a same-class font that renders reasonably (decent region_ssim OR
    # roughly-aligned ink) stays editable text. See raster_slice_failures.
    "text_broken_region_ssim_max": 0.35,
    "text_broken_ink_excess_min": 0.65,
    # Never slice regions too small to matter or so large that the "slice" would
    # approach an untouched source copy (the repo's core forbidden failure mode).
    "min_region_px": 120,
    "max_layer_canvas_fraction": 0.40,
    # Real ads routinely carry >8 sub-threshold regions (benchmark 009 hit exactly the
    # old cap of 8, silently dropping the rest — F10). Cap high enough that a genuine ad
    # is auditable; the min/max region-size gates above still bound each individual slice.
    "max_slices": 16,
}


def raster_slice_thresholds(cfg) -> dict:
    """Effective fallback thresholds: config ``fallback:`` section over defaults."""
    out = dict(RASTER_SLICE_FALLBACK_DEFAULTS)
    section = (cfg or {}).get("fallback")
    if isinstance(section, dict):
        out.update(section)
    return out


def raster_slice_failures(row, thresholds=None) -> list:
    """Reasons a per-layer QA row fails the raster-slice confidence gate.

    ``row`` is a per-layer entry produced by pixel_diff (region_ssim, and for text
    ink_iou/ink_excess). Returns [] when the region passes. Pure and deterministic.
    """
    t = dict(RASTER_SLICE_FALLBACK_DEFAULTS)
    t.update(thresholds or {})
    reasons = []
    is_text = str(row.get("type") or "") == "text"
    # CODIA-PARITY POLICY (authoritative construction brief): readable text is NEVER
    # raster-sliced to protect a pixel metric — "wrong Inter beats baked pixels".
    # Native TEXT always ships; the raster-slice fallback applies only to non-text
    # regions (icons/chrome/pills/cards). Ghost ink under native text is resolved by
    # reconstruct's solid plate-fill residue path, NOT by baking OCR into a slice.
    # ``force_raster_ids`` / focus-region forcing also cannot slice TEXT unless the
    # forensic ``text_slice_gate_enabled`` flag is on (apply_raster_slice_fallback).
    # Legacy ink gates below stay behind that flag for tooling only.
    if is_text and not bool(t.get("text_slice_gate_enabled", False)):
        return []
    iou = row.get("ink_iou") if is_text else None
    # Text with ink evidence is judged on translation-aligned ink metrics only:
    # the local preview's exact glyph placement is a proxy (Figma render-fits),
    # so raw crop SSIM/colour would slice repairable few-pixel offsets.
    if not (is_text and isinstance(iou, (int, float))):
        ssim = row.get("region_ssim")
        if isinstance(ssim, (int, float)) and ssim < float(t["region_ssim_min"]):
            reasons.append(f"region_ssim {ssim:.3f} < {float(t['region_ssim_min']):.2f}")
        color = row.get("region_color")
        if isinstance(color, (int, float)) and color < float(t["region_color_min"]):
            reasons.append(f"region_color {color:.3f} < {float(t['region_color_min']):.2f}")
    if is_text and isinstance(iou, (int, float)):
        excess = row.get("ink_excess")
        ssim = row.get("region_ssim")
        color = row.get("region_color")
        if iou < float(t["text_ink_iou_min"]):
            # A low ink-IoU with the RIGHT colour and no extra ink is a positional /
            # kerning offset of an otherwise-plausible, same-weight render — Figma
            # re-fits glyph placement, so this is repairable editable text, not a
            # broken render. The user's standing priority is that editable
            # properly-styled text beats a pixel slice; only a genuinely wrong
            # render (off-colour OR ink-heavy) is sliced here. The broken/ghost
            # rules below still catch a low-IoU render that is also heavy.
            plausible_offset = (
                isinstance(color, (int, float))
                and color >= float(t.get("text_offset_keep_color_min", 0.80))
                and isinstance(excess, (int, float))
                and excess <= float(t.get("text_offset_keep_excess_max", 0.55))
            )
            if not plausible_offset:
                reasons.append(f"ink_iou {iou:.3f} < {float(t['text_ink_iou_min']):.2f}")
        elif (isinstance(excess, (int, float))
              and iou < float(t["text_ink_iou_soft"])
              and excess > float(t["text_ink_excess_max"])):
            reasons.append(
                f"ghost ink: iou {iou:.3f} with {excess:.2f}x extra rendered ink"
            )
        # F6 combined rule: catch a genuinely-broken text render (wrong-class font) that
        # slips just under the ink_excess ceiling above. It fires ONLY when the region is
        # both structurally wrong AND ink-heavy, so a plausible same-class font with an
        # acceptable render (either metric alone in range) is left editable.
        if (isinstance(ssim, (int, float)) and isinstance(excess, (int, float))
                and ssim < float(t["text_broken_region_ssim_max"])
                and excess > float(t["text_broken_ink_excess_min"])):
            reasons.append(
                f"broken text render: region_ssim {ssim:.3f} < "
                f"{float(t['text_broken_region_ssim_max']):.2f} with "
                f"{excess:.2f}x extra rendered ink"
            )
    return reasons


# ── meta.fallback CANONICAL CONTRACT (F11) ─────────────────────────────────────────
# `meta["fallback"]` is a historically tri-state marker written by several stages, and
# was read three inconsistent ways (repair skipped ANY truthy value; reconstruct
# re-gated everything except the literal "raster-slice"; leaf accounting called a bare
# True "unexplained"). These helpers are the SINGLE source of truth for what a fallback
# marker means. Every stage — reconstruct, build_design_json, pixel_diff, repair —
# should import and use them so the readings can never diverge again.
#
# `fallback_kind(meta)` normalizes every legacy spelling to exactly one canonical name:
#
#   "raster-slice"      A confidence-gated slice of the ORIGINAL source pixels that
#                       replaced a failed editable layer (reconstruct.apply_raster_slice_fallback).
#                       Carries meta["fallback_scores"] + meta["fallback_editable"].
#                       Documented give-up; must NOT be re-gated by the slice fallback.
#   "plate-passthrough" The failed layer was dropped because the clean plate already
#                       shows its source pixels — no slice needed (target becomes "drop").
#   "fidelity-image"    An upstream give-up that substituted a raster/masked-pixel IMAGE
#                       for a native layer (text->image substitution, masked-pixel
#                       wordmark, vector/raster fallback). Explained-but-non-native: it
#                       records WHY (meta["substitution"]/low_fidelity), so it is NOT an
#                       "unexplained" quiet give-up, but it still costs native_leaf_ratio.
#   None                No fallback marker at all — a normal, fully-native layer.

_RASTER_SLICE_ALIASES = frozenset({"raster-slice", "raster_slice", "slice"})
_PLATE_PASSTHROUGH_ALIASES = frozenset({"plate-passthrough", "plate_passthrough", "passthrough"})


def fallback_kind(meta) -> Optional[str]:
    """Canonical fallback disposition of a layer's meta, or None (see contract above)."""
    if not isinstance(meta, dict):
        return None
    raw = meta.get("fallback")
    if raw in (None, False, "", 0):
        return None
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in _RASTER_SLICE_ALIASES:
            return "raster-slice"
        if value in _PLATE_PASSTHROUGH_ALIASES:
            return "plate-passthrough"
        # Any other explicit string is an upstream substitution label.
        return "fidelity-image"
    # A truthy non-string (legacy ``fallback = True``) is a substitution image.
    return "fidelity-image"


def is_raster_slice(meta) -> bool:
    """True iff this layer is a confidence-gated source-pixel slice."""
    return fallback_kind(meta) == "raster-slice"


def is_editable_leaf(layer) -> bool:
    """True iff a design.json/candidate leaf is a genuinely-editable native node.

    A leaf is editable only when it is a native TEXT or SHAPE node that is not itself a
    fidelity fallback masquerading as native. Groups are containers (not leaves) and
    image leaves are never editable. This is the honest per-leaf editability predicate
    the leaf-accounting and QA gates should share, rather than counting every FRAME.
    """
    if hasattr(layer, "__dataclass_fields__"):
        layer = asdict(layer)
    if not isinstance(layer, dict):
        return False
    ltype = layer.get("type") or layer.get("target")
    if ltype not in ("text", "shape"):
        return False
    return fallback_kind(layer.get("meta") or {}) is None


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
    # meta["fallback"] == "raster-slice" marks a confidence-gated source-pixel slice;
    # meta["fallback_editable"] keeps the failed editable attempt for future repair.
    meta: dict = field(default_factory=dict)    # {source, role, confidence, wordmark, kept_in_photo,...}
    children: list = field(default_factory=list)
    layout: dict = field(default_factory=dict)  # Figma auto-layout intent, only when confidence is high
    constraints: dict = field(default_factory=dict)
    component: dict = field(default_factory=dict)
    # Per-dimension auto-layout sizing (Codia DimensionSpec parity). Optional; an EMPTY
    # dict means "fixed on both axes" — today's pixel-frozen behavior, so absence is a
    # pure no-op and nothing regresses. Populated only for layers that sit inside a real
    # auto-layout stack (and for the stack container itself) by build_design_json's
    # geometry-evidence inference. Per axis one of:
    #   "fixed" exact size · "fill" stretch to fill the parent (auto-layout CHILDREN only)
    #   · "hug"  shrink-wrap the content (auto-layout FRAMES & TEXT only).
    # figma-plugin/code.js maps w->layoutSizingHorizontal, h->layoutSizingVertical with
    # the same legality guards; an illegal combo is skipped/guarded, never thrown.
    sizing: dict = field(default_factory=dict)  # {"w": fixed|fill|hug, "h": fixed|hug|fill}

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
