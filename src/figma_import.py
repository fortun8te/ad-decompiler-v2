"""figma_import.py — bridge design.json into Figma and export a screenshot back.

Figma has no fully-headless "create arbitrary nodes" API (REST is read-only for node
creation). The reliable path is the companion plugin in figma-plugin/, which reads a
design.json + assets from a shared inbox folder and builds real, editable nodes.

The supported mode is ``plugin``: stage design.json + assets into FIGMA_INBOX; the
plugin's Import action builds nodes and writes figma_export.png back to the run dir.
The old clipboard mode depended on an unshipped kiwi bridge and has been removed rather
than advertised as a path that cannot run.

Compiler preflight (``compiler_preflight``)
    Staging is the last automated gate before a human clicks "Import" in Figma desktop,
    and the whole desktop side is never exercised end-to-end by the CI pipeline. So the
    preflight mirrors, in Python, every case where the plugin (figma-plugin/code.js) will
    *reject* a layer (skip it) or *silently degrade* it (substitute a font, clamp a size,
    fabricate a gradient, drop an effect, downscale/refuse an image). Each mismatch between
    what build_design_json can emit and what code.js actually handles is a NAMED finding in
    design_preflight.json, merged with the structural warnings build_design_json already
    records (missing-asset, corrupt-asset, text-fidelity-fallback, layer-compile-error,
    invalid-schema). Findings are additive metadata: with strict mode off (the default)
    they never block staging, they just surface in the manifest the human reads. With
    ``figma.strict: true`` (or ``import_design(..., strict=True)``) an error-class finding
    fails staging before the atomic swap so a broken document never reaches Figma.

    The preflight is deliberately tolerant of unknown optional layer fields: it only ever
    inspects fields it knows about and never rejects a layer for carrying an unfamiliar key
    (e.g. a raster-slice fallback flag another stage may add to the schema).

"Ship proof" screenshot sibling (``figma.stage_screenshot_sibling``, default on)
    Every staged import also parks a flat copy of the original screenshot next to the
    rebuild, mirroring Codia's teardown (see runs/codia-teardown-*.json: every
    "Figma design - X" node has a sibling "Screenshot - X" frame holding one full-size
    image rectangle) so a designer gets instant eye-QA of the delta. This is a
    STAGING-TIME transform only: ``_stage_screenshot_sibling`` mutates the in-memory
    design dict written into the inbox copy, never the run's own design.json.

    figma-plugin/code.js's buildDocument() only ever creates ONE root frame per import
    (figma.createFrame(); root.clipsContent = true; every doc.layers entry is appended as
    a descendant of that single frame — code.js ~2298-2317). There is no way for
    design.json to ask for a second, independent top-level frame the way Codia's own
    "Screenshot - X" sibling is a *separate* node from "Figma design - X" — that would
    need a code.js change, which is out of scope here. A child placed outside
    [0, canvas.w] x [0, canvas.h] does not survive either: clipsContent clips to the
    frame's OWN bounds regardless of what a child's own clip setting says, so parking the
    screenshot at a literal negative x (Codia's on-canvas gap of ~1130px) would just make
    the plugin discard it silently.

    So this widens the single root frame to fit both: canvas becomes
    ``2*w + gap`` wide, every pre-existing top-level layer is shifted right by
    ``w + gap`` (only top-level boxes need shifting — schema v2's coordinate space is
    local-to-parent, so nested descendants move for free with their parent), and the
    screenshot frame is inserted as the new first top-level layer at x=0, sized to the
    canvas. That is the "in-root, first child, position accounted for" fallback the spec
    calls for when a true sibling frame isn't reachable.

    This can never leak into pixel_diff/QA scoring because render_preview.render() and
    pixel_diff.compare() run on run_dir/design.json *before* staging even happens
    (run_pipeline.py: render_preview.render(...) at the QA step precedes
    figma_import.import_design(...) later in the same run), and staging never writes back
    to that file — only to the inbox's own copy. The injected layers additionally carry
    ``meta.role == "qa-ignore"`` for any future direct consumer to recognise and skip.
"""
from __future__ import annotations
import hashlib, math, os, re, shutil, json, time, tempfile, uuid

DEFAULT_INBOX = os.environ.get("FIGMA_INBOX", os.path.expanduser("~/figma-inbox"))

# ── Figma compiler limits (mirror figma-plugin/code.js + documented API constraints) ──
# figma.createImage throws "Image is too large" above 4096px in either dimension.
# https://developers.figma.com/docs/plugins/api/properties/figma-createimage/
FIGMA_MAX_IMAGE_DIM = 4096

_PREFLIGHT_DEFAULTS = {
    # createNodeFromSvg becomes very slow / fails on pathological path counts.
    "svg_max_paths": 2000,
    # Hard Figma image-fill ceiling; oversized fills are rejected by createImage.
    "image_max_dim": FIGMA_MAX_IMAGE_DIM,
    # Figma requires at least two gradient stops; the plugin fabricates the rest.
    "gradient_min_stops": 2,
    # A sane upper bound; hundreds of stops are almost always extraction noise.
    "gradient_max_stops": 64,
    # Sane upper bound for a shadow/blur radius before it is certainly a bug.
    "effect_max_radius": 1000.0,
    # boxOf() clamps w/h to 0.01px; anything at/under this renders invisibly.
    "zero_size_epsilon": 0.01,
}

# Renderable canonical types the plugin's compileLayer() dispatches on. Anything else
# hits its "Unsupported layer type" throw and is skipped.
_KNOWN_TYPES = frozenset({"text", "image", "shape", "vector", "frame", "group"})

# Effect types effectFromSpec() can represent; everything else is dropped as a fidelity
# fallback in the plugin.
_KNOWN_EFFECT_TYPES = frozenset({
    "DROP_SHADOW", "SHADOW", "INNER_SHADOW", "BLUR", "LAYER_BLUR", "BACKGROUND_BLUR",
})

_FIGMA_BLEND_MODES = frozenset({
    "PASS_THROUGH", "NORMAL", "DARKEN", "MULTIPLY", "LINEAR_BURN", "COLOR_BURN",
    "LIGHTEN", "SCREEN", "LINEAR_DODGE", "COLOR_DODGE", "OVERLAY", "SOFT_LIGHT",
    "HARD_LIGHT", "DIFFERENCE", "EXCLUSION", "HUE", "SATURATION", "COLOR", "LUMINOSITY",
})

_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
})

# Foreground/cutout roles whose raster is expected to carry transparency; a flattened
# (alpha-free) asset in one of these roles is a visible regression, not a rectangle by
# intent. Kept in sync with build_design_json's raster taxonomy.
_CUTOUT_ROLES = frozenset({
    "cutout", "product", "product-cluster", "person", "people", "face", "hand",
    "avatar", "profile", "profile-photo", "logo", "wordmark", "brand", "logotype",
    "icon", "sticker", "subject", "foreground",
})

# Findings whose presence means at least one layer will be missing/wrong in Figma (not
# merely degraded). Strict mode blocks staging when any of these are present. This spans
# both the structural codes build_design_json already emits and the compiler-mirror codes
# added here.
_ERROR_CLASS_CODES = frozenset({
    "missing-asset", "corrupt-asset", "invalid-schema", "layer-compile-error",
    "invalid-geometry", "empty-vector", "image-too-large", "unknown-layer-type",
})

# SVG features figma.createNodeFromSvg either drops silently or chokes on. Matched
# case-insensitively against the SVG payload the plugin would hand to createNodeFromSvg.
_SVG_FEATURE_CHECKS = (
    ("filter", re.compile(r"<\s*filter\b|\bfilter\s*=\s*[\"']?\s*url\(", re.I)),
    ("filter-primitive", re.compile(
        r"<\s*fe(GaussianBlur|ColorMatrix|DropShadow|Blend|Image|Turbulence|Morphology"
        r"|Offset|Composite|Flood|Merge|Tile|DisplacementMap|ComponentTransfer"
        r"|ConvolveMatrix|DiffuseLighting|SpecularLighting)\b", re.I)),
    ("foreignObject", re.compile(r"<\s*foreignObject\b", re.I)),
    ("script", re.compile(r"<\s*script\b", re.I)),
    ("style-block", re.compile(r"<\s*style\b", re.I)),
    ("animation", re.compile(r"<\s*(animate|animateTransform|animateMotion|set)\b", re.I)),
    ("pattern", re.compile(r"<\s*pattern\b", re.I)),
    ("embedded-image", re.compile(r"<\s*image\b", re.I)),
    ("use-ref", re.compile(r"<\s*use\b", re.I)),
    ("svg-text", re.compile(r"<\s*text\b", re.I)),
)

_PATH_TAG_RE = re.compile(r"<\s*path\b[^>]*/?>", re.I)
_D_ATTR_RE = re.compile(r"""\bd\s*=\s*("([^"]*)"|'([^']*)')""", re.I)


# ── small helpers mirroring code.js's pick()/finite()/normalizedToken() ──────────────
def _pick(d, *keys):
    if not isinstance(d, dict):
        return None
    for key in keys:
        value = d.get(key)
        if value is not None:
            return value
    return None


def _norm(value):
    return re.sub(r"[\s\-]+", "_", str(value if value is not None else "").strip()).upper()


def _finite(value):
    """Return value as a finite float, or None if it is not a finite number.

    Mirrors code.js finite(): numeric strings coerce, NaN/±Inf and non-numerics fail.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (ValueError, TypeError):
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _jsonsafe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _finding(code, severity, layer_id=None, **extra):
    finding = {"code": code, "severity": severity}
    if layer_id is not None:
        finding["layer_id"] = layer_id
    finding.update({key: _jsonsafe(val) for key, val in extra.items()})
    return finding


def _is_error_class(finding):
    if not isinstance(finding, dict):
        return False
    if str(finding.get("severity")) == "error":
        return True
    return finding.get("code") in _ERROR_CLASS_CODES


def _layer_id(layer):
    return str(_pick(layer, "id", "source_id", "sourceId") or "layer")


def _canon_type(layer):
    raw = _norm(_pick(layer, "type", "node_type", "nodeType", "kind"))
    shape_kind = _norm(_pick(layer, "shape_kind", "shapeKind"))
    if raw == "TEXT":
        return "text"
    if raw in ("IMAGE", "PHOTO", "RASTER"):
        return "image"
    if raw in ("FRAME", "CONTAINER", "SECTION"):
        return "frame"
    if raw == "GROUP":
        return "group"
    if raw in ("VECTOR", "SVG", "ICON", "PATH"):
        return "vector"
    if raw == "SHAPE":
        return "vector" if shape_kind == "PATH" else "shape"
    if raw in ("RECT", "RECTANGLE", "ELLIPSE", "CIRCLE"):
        return "shape"
    kids = layer.get("children")
    if not isinstance(kids, list):
        kids = layer.get("layers")
    if isinstance(kids, list) and kids:
        return "frame"
    return raw.lower() if raw else "unknown"


def _fill_specs(layer):
    many = _pick(layer, "fills", "paints")
    if isinstance(many, list):
        return many
    one = _pick(layer, "fill", "background")
    if one is not None:
        return [one]
    style = layer.get("style") or {}
    style_many = _pick(style, "fills", "paints")
    if isinstance(style_many, list):
        return style_many
    style_one = _pick(style, "fill", "background", "color")
    return [] if style_one is None else [style_one]


def _stroke_specs(layer):
    many = layer.get("strokes")
    if isinstance(many, list):
        return many
    one = layer.get("stroke")
    if one is not None:
        return [one]
    style = layer.get("style") or {}
    style_many = style.get("strokes")
    if isinstance(style_many, list):
        return style_many
    style_one = style.get("stroke")
    return [] if style_one is None else [style_one]


def _is_gradient(spec):
    if not isinstance(spec, dict):
        return False
    kind = _norm(_pick(spec, "kind", "type"))
    return "GRADIENT" in kind or kind in ("LINEAR", "RADIAL", "ANGULAR", "DIAMOND")


def _iter_layers(layers, _depth=0):
    if _depth > 256 or not isinstance(layers, list):
        return
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        yield layer
        kids = layer.get("children")
        if not isinstance(kids, list):
            kids = layer.get("layers")
        if isinstance(kids, list):
            yield from _iter_layers(kids, _depth + 1)


def _resolve_asset(run_dir, src):
    if not src:
        return None
    text = str(src).replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if parts:
        candidate = os.path.normpath(os.path.join(run_dir, *parts))
        if os.path.isfile(candidate):
            return candidate
    if os.path.isabs(str(src)) and os.path.isfile(str(src)):
        return str(src)
    if os.path.isfile(text):
        return os.path.abspath(text)
    return None


def _asset_meta(run_dir, src, cache):
    """Resolve, checksum, and (for images) probe an asset once, memoised by real path."""
    resolved = _resolve_asset(run_dir, src)
    if not resolved:
        return None
    key = os.path.abspath(resolved)
    if key in cache:
        return cache[key]
    info = {"path": str(src),
            "resolved": os.path.relpath(resolved, run_dir).replace(os.sep, "/")}
    try:
        with open(resolved, "rb") as handle:
            data = handle.read()
        info["bytes"] = len(data)
        info["sha256"] = hashlib.sha256(data).hexdigest()
        if not data:
            info["corrupt"] = True
            info["error"] = "empty file"
    except OSError as exc:
        info["corrupt"] = True
        info["error"] = str(exc)
        cache[key] = info
        return info
    if not info.get("corrupt") and os.path.splitext(resolved)[1].lower() in _IMAGE_EXTS:
        try:
            from PIL import Image
            with Image.open(resolved) as probe:
                probe.verify()
            with Image.open(resolved) as probe:
                info["width"], info["height"] = int(probe.width), int(probe.height)
                info["mode"] = probe.mode
                info["has_alpha"] = bool(
                    probe.mode in ("RGBA", "LA", "PA") or "transparency" in probe.info
                )
                info["cmyk"] = probe.mode == "CMYK"
                icc = probe.info.get("icc_profile")
                info["nonsrgb_icc"] = bool(icc) and (b"srgb" not in bytes(icc).lower())
        except Exception as exc:  # PIL raises a menagerie of types on bad rasters
            info["corrupt"] = True
            info["error"] = str(exc)
    cache[key] = info
    return info


def _expects_alpha(layer):
    meta = layer.get("meta") or {}
    role = str(_pick(meta, "role") or _pick(layer, "role") or "").strip().lower().replace("_", "-")
    if role == "background":
        return False
    if isinstance(layer.get("mask"), dict) and layer.get("mask"):
        return True
    if meta.get("cutout") or meta.get("transparent") or meta.get("alpha"):
        return True
    return role in _CUTOUT_ROLES


# ── per-concern checks (each appends 0+ findings; none raise on well-formed input) ───
def _check_geometry(layer, lid, findings, thresholds):
    box = layer.get("box")
    if not isinstance(box, dict):
        return
    eps = float(thresholds["zero_size_epsilon"])
    bad, zero = [], []
    for axis in ("x", "y", "w", "h"):
        if axis not in box:
            continue
        raw = box.get(axis)
        value = _finite(raw)
        if value is None:
            bad.append(axis)
            continue
        if axis in ("w", "h"):
            if value < 0:
                bad.append(axis)
            elif value <= eps:
                zero.append(axis)
    if bad:
        fields = sorted(set(bad))
        findings.append(_finding(
            "invalid-geometry", "error", lid, fields=fields,
            detail="non-finite or negative geometry in box: " + ", ".join(fields)
                   + "; the plugin substitutes fallbacks and mis-sizes/mis-places the layer"))
    if zero:
        fields = sorted(set(zero))
        findings.append(_finding(
            "zero-size-layer", "warn", lid, fields=fields,
            detail="box has ~zero " + "/".join(fields)
                   + "; Figma clamps to 0.01px so the layer renders invisibly"))


def _check_unknown_type(layer, lid, findings):
    if _canon_type(layer) not in _KNOWN_TYPES:
        findings.append(_finding(
            "unknown-layer-type", "error", lid,
            type=_pick(layer, "type", "node_type", "nodeType", "kind"),
            detail="layer type is not one the plugin can compile; it will be skipped"))


def _check_fonts(layer, lid, findings):
    style = {}
    style.update(layer.get("typography") or {})
    style.update(layer.get("style") or {})
    content = layer.get("text")
    runs = _pick(layer, "text_runs", "textRuns", "runs")
    if content is None and isinstance(runs, list):
        content = "".join(str(r.get("text", "")) for r in runs if isinstance(r, dict))
    if not str(content or "").strip():
        return
    if _pick(style, "fontFamily", "font_family", "family"):
        return
    candidates = _pick(style, "fontCandidates", "font_candidates", "candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return
            if isinstance(candidate, dict) and _pick(candidate, "fontFamily", "font_family", "family"):
                return
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                run_style = {}
                run_style.update(run.get("typography") or {})
                run_style.update(run.get("style") or {})
                if (_pick(run_style, "fontFamily", "font_family", "family")
                        or _pick(run, "fontFamily", "font_family", "family")):
                    return
    findings.append(_finding(
        "empty-font-candidates", "warn", lid,
        detail="text has no font family or candidate list; Figma silently substitutes Inter"))


def _svg_string_path_stats(svg):
    tags = _PATH_TAG_RE.findall(svg)
    empty = 0
    for tag in tags:
        match = _D_ATTR_RE.search(tag)
        value = None
        if match:
            value = match.group(2) if match.group(2) is not None else match.group(3)
        if not (value and value.strip()):
            empty += 1
    return len(tags), empty


def _svg_unsupported_features(svg):
    return [name for name, pattern in _SVG_FEATURE_CHECKS if pattern.search(svg)]


def _check_svg(layer, lid, findings, thresholds):
    max_paths = int(thresholds["svg_max_paths"])
    raw = _pick(layer, "svg", "svg_string", "svgString")
    if isinstance(raw, str) and "<svg" in raw:
        count, empty = _svg_string_path_stats(raw)
        features = _svg_unsupported_features(raw)
        if features:
            findings.append(_finding(
                "svg-unsupported-feature", "warn", lid, features=features,
                detail="SVG uses features createNodeFromSvg drops/chokes on: " + ", ".join(features)))
        if empty:
            findings.append(_finding(
                "svg-empty-path", "warn", lid, empty_paths=empty,
                detail="%d <path> element(s) have an empty 'd' attribute" % empty))
        if count > max_paths:
            findings.append(_finding(
                "svg-too-many-paths", "warn", lid, path_count=count, limit=max_paths,
                detail="%d paths exceeds %d; createNodeFromSvg may be very slow or fail"
                       % (count, max_paths)))
        return
    paths = _pick(layer, "vector_paths", "vectorPaths", "paths")
    if not isinstance(paths, list) or not paths:
        meta_paths = _pick(layer.get("meta") or {}, "vector_paths", "vectorPaths")
        if isinstance(meta_paths, list) and meta_paths:
            paths = meta_paths
    if not isinstance(paths, list) or not paths:
        single = _pick(layer, "path", "d")
        paths = [{"d": single}] if single else []
    if not paths and not (isinstance(raw, str) and raw.strip()):
        findings.append(_finding(
            "empty-vector", "error", lid,
            detail="vector layer has no SVG string and no path geometry; the plugin rejects it"))
        return
    empty = 0
    for entry in paths:
        if isinstance(entry, str):
            d = entry
        elif isinstance(entry, dict):
            d = _pick(entry, "d", "path")
        else:
            d = None
        if not (isinstance(d, str) and d.strip()):
            empty += 1
    if empty:
        findings.append(_finding(
            "svg-empty-path", "warn", lid, empty_paths=empty,
            detail="%d vector path entry(ies) have empty geometry" % empty))
    if len(paths) > max_paths:
        findings.append(_finding(
            "svg-too-many-paths", "warn", lid, path_count=len(paths), limit=max_paths,
            detail="%d paths exceeds %d; createNodeFromSvg may be very slow or fail"
                   % (len(paths), max_paths)))


def _check_shape_paint(layer, lid, findings):
    if _fill_specs(layer) or _stroke_specs(layer):
        return
    findings.append(_finding(
        "shape-no-paint", "warn", lid,
        detail="shape has no fill or stroke; the plugin builds an unpainted rectangle"))


def _check_gradients(layer, lid, findings, thresholds):
    low = int(thresholds["gradient_min_stops"])
    high = int(thresholds["gradient_max_stops"])
    specs = list(_fill_specs(layer))
    for stroke in _stroke_specs(layer):
        if isinstance(stroke, dict) and isinstance(stroke.get("paint"), dict):
            specs.append(stroke["paint"])
        else:
            specs.append(stroke)
    for spec in specs:
        if not _is_gradient(spec):
            continue
        stops = _pick(spec, "stops", "gradientStops", "gradient_stops")
        count = len(stops) if isinstance(stops, list) else 0
        if count < low:
            findings.append(_finding(
                "gradient-stops", "warn", lid, stops=count,
                detail="gradient has %d stop(s); Figma needs >=%d so the plugin fabricates the rest"
                       % (count, low)))
        elif count > high:
            findings.append(_finding(
                "gradient-stops", "warn", lid, stops=count,
                detail="gradient has %d stops (> %d); likely extraction noise" % (count, high)))


def _check_effects(layer, lid, findings, thresholds):
    style = layer.get("style") or {}
    if isinstance(layer.get("effects"), list):
        specs = layer["effects"]
    elif isinstance(style.get("effects"), list):
        specs = style["effects"]
    elif isinstance(layer.get("shadow"), dict):
        specs = [dict(layer["shadow"], type="drop-shadow")]
    else:
        return
    max_radius = float(thresholds["effect_max_radius"])
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        etype = _norm(_pick(spec, "type", "kind"))
        if etype not in _KNOWN_EFFECT_TYPES:
            findings.append(_finding(
                "unsupported-effect", "warn", lid, effect=(etype or None),
                detail="effect type %r is not a Figma shadow/blur and will be dropped" % (etype or "")))
            continue
        raw_radius = _pick(spec, "radius", "blur")
        if raw_radius is not None:
            value = _finite(raw_radius)
            if value is None:
                findings.append(_finding(
                    "effect-param-range", "warn", lid, param="radius", value=raw_radius,
                    detail="effect radius is not a finite number; the plugin substitutes a default"))
            elif value < 0:
                findings.append(_finding(
                    "effect-param-range", "warn", lid, param="radius", value=value,
                    detail="negative effect radius (%s) is clamped to 0; the blur/shadow is lost" % value))
            elif value > max_radius:
                findings.append(_finding(
                    "effect-param-range", "warn", lid, param="radius", value=value,
                    detail="effect radius %s exceeds the sane maximum %s" % (value, max_radius)))
        spread = _pick(spec, "spread")
        if spread is not None and _finite(spread) is None:
            findings.append(_finding(
                "effect-param-range", "warn", lid, param="spread", value=spread,
                detail="effect spread is not a finite number"))
        offset = _pick(spec, "offset")
        if isinstance(offset, dict):
            for axis in ("x", "y"):
                if offset.get(axis) is not None and _finite(offset.get(axis)) is None:
                    findings.append(_finding(
                        "effect-param-range", "warn", lid, param="offset." + axis, value=offset.get(axis),
                        detail="effect offset.%s is not a finite number" % axis))


def _check_blend(layer, lid, findings):
    blend = _pick(layer, "blend_mode", "blendMode")
    if blend is None:
        return
    norm = _norm(blend)
    if norm and norm not in _FIGMA_BLEND_MODES:
        findings.append(_finding(
            "unsupported-blend-mode", "warn", lid, blend_mode=str(blend),
            detail="blend mode %r is not a Figma blend mode and will be ignored" % str(blend)))


def _check_image_asset(layer, lid, run_dir, findings, assets_meta, cache, thresholds):
    src = _pick(layer, "src", "source", "asset", "asset_path", "assetPath")
    if not src:
        if _canon_type(layer) == "image":
            findings.append(_finding(
                "missing-asset", "error", lid, path=None,
                detail="image layer has no asset source; the plugin throws 'Missing image asset'"))
        return
    meta = _asset_meta(run_dir, src, cache)
    if meta is None:
        findings.append(_finding(
            "missing-asset", "error", lid, path=str(src),
            detail="image asset not found under the run directory"))
        return
    assets_meta.append(dict(meta, layer_id=lid))
    if meta.get("corrupt"):
        findings.append(_finding(
            "corrupt-asset", "error", lid, path=str(src),
            detail=meta.get("error") or "asset could not be decoded"))
        return
    width, height = meta.get("width"), meta.get("height")
    limit = int(thresholds["image_max_dim"])
    if isinstance(width, int) and isinstance(height, int) and max(width, height) > limit:
        findings.append(_finding(
            "image-too-large", "error", lid, path=str(src), width=width, height=height, limit=limit,
            detail="image is %dx%d; Figma rejects image fills larger than %dpx per side"
                   % (width, height, limit)))
    if meta.get("cmyk"):
        findings.append(_finding(
            "color-profile", "warn", lid, path=str(src),
            detail="image is CMYK; Figma treats fills as sRGB so colors will shift"))
    elif meta.get("nonsrgb_icc"):
        findings.append(_finding(
            "color-profile", "warn", lid, path=str(src),
            detail="image carries a non-sRGB ICC profile Figma ignores; colors may shift"))
    if _expects_alpha(layer) and meta.get("has_alpha") is False:
        findings.append(_finding(
            "alpha-channel-loss", "warn", lid, path=str(src),
            detail="cutout/masked image has no alpha channel; it will paint as an opaque rectangle"))


def _check_mask(layer, lid, run_dir, findings, assets_meta, cache, thresholds):
    mask = layer.get("mask")
    if not isinstance(mask, dict) or not mask:
        return
    kind = _norm(_pick(mask, "kind", "type"))
    if kind == "ALPHA":
        msrc = _pick(mask, "src", "source", "asset", "asset_path", "assetPath")
        if not msrc:
            # An alpha mask with no separate source means "clip to the image's own alpha".
            # For a SAM3/cutout RGBA image that already carries a real alpha channel this is
            # the intended, lossless representation: figma-plugin/code.js renders the image
            # with its embedded transparency (needsMaskGroup is false for ALPHA-without-src),
            # so nothing is lost — it is not a degradation and must not warn. Only warn when
            # the host image ALSO lacks an alpha channel, i.e. there is genuinely no mask
            # data anywhere and the layer would paint as an opaque rectangle.
            lsrc = _pick(layer, "src", "source", "asset", "asset_path", "assetPath")
            lmeta = _asset_meta(run_dir, lsrc, cache) if lsrc else None
            if lmeta and not lmeta.get("corrupt") and lmeta.get("has_alpha"):
                return
            findings.append(_finding(
                "alpha-mask-missing", "warn", lid,
                detail="alpha mask has no source and the image carries no alpha channel; "
                       "it will paint as an opaque rectangle"))
            return
        meta = _asset_meta(run_dir, msrc, cache)
        if meta is None:
            findings.append(_finding(
                "alpha-mask-missing", "warn", lid, path=str(msrc),
                detail="alpha mask asset not found; the image's own transparency is used instead"))
            return
        assets_meta.append(dict(meta, layer_id=lid + "/mask"))
        if meta.get("corrupt"):
            findings.append(_finding(
                "alpha-mask-missing", "warn", lid, path=str(msrc),
                detail="alpha mask asset could not be decoded"))
            return
        width, height = meta.get("width"), meta.get("height")
        limit = int(thresholds["image_max_dim"])
        if isinstance(width, int) and isinstance(height, int) and max(width, height) > limit:
            findings.append(_finding(
                "image-too-large", "error", lid, path=str(msrc), width=width, height=height, limit=limit,
                detail="alpha mask is %dx%d; Figma rejects images larger than %dpx per side"
                       % (width, height, limit)))
        return
    if kind == "PATH" or mask.get("path") or mask.get("svg"):
        if not (mask.get("svg") or mask.get("path")
                or _pick(mask, "vector_paths", "vectorPaths", "paths")):
            findings.append(_finding(
                "mask-geometry-empty", "warn", lid,
                detail="path mask has no geometry; the plugin cannot build it and the image is left unclipped"))


def _count_layers(layers, _depth=0):
    if _depth > 256 or not isinstance(layers, list):
        return 0
    total = 0
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        total += 1
        kids = layer.get("children")
        if not isinstance(kids, list):
            kids = layer.get("layers")
        total += _count_layers(kids, _depth + 1)
    return total


def compiler_preflight(design: dict, run_dir: str, cfg: dict | None = None) -> dict:
    """Mirror figma-plugin/code.js: enumerate every reject/degrade case as a finding.

    Returns a design_preflight.json-shaped dict whose ``warnings`` list merges the
    structural warnings build_design_json already recorded (seeded from the run's
    design_preflight.json, or design.meta.warnings) with the compiler-mirror findings
    added here. Unknown optional layer fields are never a reason to reject a layer.
    """
    cfg = cfg or {}
    figma_cfg = cfg.get("figma") or {}
    thresholds = dict(_PREFLIGHT_DEFAULTS)
    overrides = figma_cfg.get("preflight")
    if isinstance(overrides, dict):
        thresholds.update({k: v for k, v in overrides.items() if k in _PREFLIGHT_DEFAULTS})

    # Seed with the structural warnings the compiler already produced.
    build_findings: list = []
    build_layer_count = None
    preflight_path = os.path.join(run_dir, "design_preflight.json")
    if os.path.isfile(preflight_path):
        try:
            with open(preflight_path, encoding="utf-8") as handle:
                build_pf = json.load(handle)
            if isinstance(build_pf, dict):
                if isinstance(build_pf.get("warnings"), list):
                    build_findings = [w for w in build_pf["warnings"] if isinstance(w, dict)]
                build_layer_count = build_pf.get("layer_count")
        except (OSError, ValueError):
            pass
    if not build_findings:
        meta_warnings = (design.get("meta") or {}).get("warnings")
        if isinstance(meta_warnings, list):
            build_findings = [w for w in meta_warnings if isinstance(w, dict)]

    findings: list = []
    assets_meta: list = []
    asset_cache: dict = {}

    for layer in _iter_layers(design.get("layers") or []):
        lid = _layer_id(layer)
        try:
            _check_geometry(layer, lid, findings, thresholds)
            _check_unknown_type(layer, lid, findings)
            canon = _canon_type(layer)
            if canon == "text":
                _check_fonts(layer, lid, findings)
            elif canon == "vector":
                _check_svg(layer, lid, findings, thresholds)
            elif canon == "shape":
                _check_shape_paint(layer, lid, findings)
            _check_gradients(layer, lid, findings, thresholds)
            _check_effects(layer, lid, findings, thresholds)
            _check_blend(layer, lid, findings)
            if canon == "image" or _pick(layer, "src", "source", "asset", "asset_path", "assetPath"):
                _check_image_asset(layer, lid, run_dir, findings, assets_meta, asset_cache, thresholds)
            _check_mask(layer, lid, run_dir, findings, assets_meta, asset_cache, thresholds)
        except Exception as exc:  # a malformed layer must never crash staging
            findings.append(_finding(
                "preflight-error", "warn", lid,
                detail="preflight rule raised on this layer: " + str(exc)))

    # Merge structural + compiler-mirror findings. build_design_json's own warnings win:
    # when it already flagged that a layer will be missing/wrong (identity codes below),
    # the compiler mirror must not restate the same layer's problem, and exact duplicates
    # are dropped either way.
    _identity_codes = {"missing-asset", "corrupt-asset", "image-too-large",
                       "empty-vector", "unknown-layer-type", "invalid-geometry"}
    build_identity = {(f.get("code"), f.get("layer_id"))
                      for f in build_findings if f.get("code") in _identity_codes}
    merged: list = []
    seen = set()
    for finding in list(build_findings) + findings:
        code = finding.get("code")
        lid = finding.get("layer_id")
        if finding not in build_findings and code in _identity_codes and (code, lid) in build_identity:
            continue
        key = (code, lid, finding.get("path"), finding.get("detail"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(finding)

    error_count = sum(1 for f in merged if _is_error_class(f))
    layer_count = build_layer_count
    if not isinstance(layer_count, int):
        layer_count = _count_layers(design.get("layers") or [])
    return {
        "ok": not merged,
        "warnings": merged,
        "layer_count": layer_count,
        "error_count": error_count,
        "warn_count": len(merged) - error_count,
        "assets": assets_meta,
        "thresholds": thresholds,
        "generated_by": "figma_import.compiler_preflight",
    }


# ── "ship proof" screenshot sibling (staging-time only) ───────────────────────────────
_SCREENSHOT_SOURCE_NAMES = ("original.png", "normalized.png")
_SCREENSHOT_ASSET_STEM = "_screenshot_proof"


def _file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _copy_as_srgb(source, dest):
    """Copy ``source`` -> ``dest``, converting a non-sRGB ICC image to sRGB.

    Figma ignores embedded ICC profiles and treats every image fill as sRGB, so an
    asset carrying (say) a Display-P3 or untagged-wide profile shows shifted colors in
    Figma versus the pixels our own preview/QA scored. Baking the profile into sRGB at
    stage time makes what Figma displays match what we measured, and removes the
    ``color-profile`` preflight warning at the source rather than merely reporting it.

    Returns True when a colour conversion was applied; falls back to a byte copy (never
    raises) so a missing ImageCms / odd profile can never break staging.
    """
    try:
        from PIL import Image
        with Image.open(source) as im:
            icc = im.info.get("icc_profile")
            if icc and b"srgb" not in bytes(icc).lower():
                import io
                from PIL import ImageCms
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                srgb_profile = ImageCms.createProfile("sRGB")
                has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                out_mode = "RGBA" if has_alpha else "RGB"
                converted = ImageCms.profileToProfile(
                    im, src_profile, srgb_profile, outputMode=out_mode)
                # Save WITHOUT any embedded icc_profile: an untagged PNG is assumed sRGB by
                # Figma, and our probe treats "no profile" as sRGB. (profileToProfile leaves
                # the freshly-created sRGB profile in .info, whose description does not
                # contain the literal "srgb" token the probe looks for, so it must be
                # stripped or it would re-trip the same warning.)
                converted.info.pop("icc_profile", None)
                converted.save(dest, format="PNG", icc_profile=None)
                return True
    except Exception:
        pass
    shutil.copyfile(source, dest)
    return False


def _screenshot_source_path(run_dir, figma_cfg):
    """Pick the flat, original-looking screenshot to park next to the rebuild.

    Config override wins; otherwise prefer the run's own ``original.png`` (the raw
    screenshot before any cleanup) and fall back to ``normalized.png``.
    """
    override = figma_cfg.get("screenshot_source")
    candidates = [override] if override else []
    candidates += [os.path.join(run_dir, name) for name in _SCREENSHOT_SOURCE_NAMES]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _stage_screenshot_sibling(design: dict, run_dir: str, figma_cfg: dict) -> dict:
    """Mutate ``design`` in place to add the flat screenshot "ship proof" layer.

    See the module docstring for why this widens the single root frame and shifts the
    existing top-level layers rather than placing the screenshot at a literal negative x.
    Never raises: any failure here must degrade to "no sibling added", not a broken stage.
    """
    canvas = design.get("canvas")
    if not isinstance(canvas, dict):
        return {"ok": False, "reason": "invalid-canvas"}
    width = _finite(_pick(canvas, "w", "width"))
    height = _finite(_pick(canvas, "h", "height"))
    if not width or not height or width <= 0 or height <= 0:
        return {"ok": False, "reason": "invalid-canvas"}

    source = _screenshot_source_path(run_dir, figma_cfg)
    if not source:
        return {"ok": False, "reason": "no-screenshot-source"}

    gap = _finite(figma_cfg.get("screenshot_gap"))
    gap = gap if gap is not None and gap >= 0 else 50.0

    # Copy into run_dir/assets so compiler_preflight (which resolves image sources against
    # run_dir, not the eventual staged copy) and the existing assets copytree below both
    # pick it up automatically, using the exact same on-disk convention every other image
    # layer's `src` already relies on.
    ext = os.path.splitext(source)[1] or ".png"
    asset_name = _SCREENSHOT_ASSET_STEM + ext
    assets_dir = os.path.join(run_dir, "assets")
    dest = os.path.join(assets_dir, asset_name)
    try:
        os.makedirs(assets_dir, exist_ok=True)
        # Always convert-through sRGB (cheap): the proof is the raw source screenshot,
        # which frequently carries a non-sRGB capture profile Figma would ignore.
        _copy_as_srgb(source, dest)
    except OSError as exc:
        return {"ok": False, "reason": "asset-copy-failed", "error": str(exc)}
    src_rel = "assets/" + asset_name

    layers = design.get("layers")
    if not isinstance(layers, list):
        layers = []
        design["layers"] = layers

    shift = width + gap
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        box = layer.get("box")
        if not isinstance(box, dict):
            box = {}
            layer["box"] = box
        current_x = _finite(box.get("x"))
        box["x"] = (current_x if current_x is not None else 0.0) + shift

    source_name = os.path.basename(source)
    layer_name = f"Screenshot - {source_name}"
    screenshot_layer = {
        "id": "screenshot-proof",
        "type": "frame",
        "name": layer_name,
        "box": {"x": 0, "y": 0, "w": width, "h": height},
        "clips_content": True,
        "meta": {"role": "qa-ignore", "screenshot_proof": True, "source": source_name},
        "children": [
            {
                "id": "screenshot-proof-image",
                "type": "image",
                "name": layer_name,
                "box": {"x": 0, "y": 0, "w": width, "h": height},
                "src": src_rel,
                "meta": {"role": "qa-ignore", "screenshot_proof": True},
            }
        ],
    }
    layers.insert(0, screenshot_layer)
    design["canvas"] = dict(canvas, w=2 * width + gap, h=height)
    meta = design.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        design["meta"] = meta
    meta["screenshot_proof"] = {"added": True, "source": source_name, "asset": src_rel}
    return {"ok": True, "source": source, "asset": src_rel, "layer_id": "screenshot-proof",
            "gap": gap, "shift": shift}


def import_design(design_path: str, run_dir: str, cfg: dict | None = None,
                  strict: bool | None = None) -> dict:
    cfg = cfg or {}
    mode = (cfg.get("figma") or {}).get("mode", "plugin")
    try:
        if mode != "plugin":
            return {"ok": False, "mode": mode, "error": f"unsupported Figma mode: {mode}"}
        return _stage_for_plugin(design_path, run_dir, cfg, strict=strict)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "mode": mode, "error": str(exc),
                "exception": type(exc).__name__}


def _stage_for_plugin(design_path, run_dir, cfg, strict=None) -> dict:
    figma_cfg = cfg.get("figma") or {}
    inbox = figma_cfg.get("inbox", DEFAULT_INBOX)
    os.makedirs(inbox, exist_ok=True)
    if not os.path.isfile(design_path):
        raise FileNotFoundError(f"design.json not found: {design_path}")
    with open(design_path, encoding="utf-8") as fh:
        design = json.load(fh)
    if not isinstance(design, dict) or not isinstance(design.get("layers", []), list):
        raise ValueError("design.json must be an object with a layers list")
    doc_id = "".join(c if c.isalnum() or c in "-_" else "-"
                     for c in str(design.get("id") or os.path.basename(run_dir)))[:80] or "run"

    # "Ship proof" screenshot sibling: staging-time only, mutates this in-memory `design`
    # dict (never design_path on disk), so it must run before preflight/asset registration
    # below but can never affect anything that reads design_path itself (preview.png/QA).
    screenshot_sibling = {"ok": False, "reason": "disabled"}
    if figma_cfg.get("stage_screenshot_sibling", True):
        try:
            screenshot_sibling = _stage_screenshot_sibling(design, run_dir, figma_cfg)
        except Exception as exc:  # never let the ship-proof enhancement break staging
            screenshot_sibling = {"ok": False, "reason": "error", "error": str(exc)}

    # Compiler preflight BEFORE the atomic swap so --strict can refuse a broken document
    # without ever publishing a manifest a human would act on.
    preflight = compiler_preflight(design, run_dir, cfg)
    strict_mode = bool(figma_cfg.get("strict", False)) if strict is None else bool(strict)
    error_findings = [f for f in preflight["warnings"] if _is_error_class(f)]
    if strict_mode and error_findings:
        return {
            "ok": False, "mode": "plugin", "blocked": True, "doc_id": doc_id,
            "error": "strict preflight blocked staging: %d error-class finding(s)"
                     % len(error_findings),
            "errors": error_findings,
            "preflight": preflight,
        }

    staged_root = os.path.join(inbox, "runs", doc_id)
    runs_root = os.path.join(inbox, "runs")
    os.makedirs(runs_root, exist_ok=True)
    temp_root = tempfile.mkdtemp(prefix=f".{doc_id}-", dir=runs_root)
    # Write the (possibly screenshot-sibling-mutated) in-memory `design` dict rather than
    # byte-copying design_path, so the staged copy reflects the staging-time transform
    # above while design_path itself is never touched.
    with open(os.path.join(temp_root, "design.json"), "w", encoding="utf-8") as fh:
        json.dump(design, fh, indent=2)
    assets = os.path.join(run_dir, "assets")
    pruned_assets = []
    if os.path.isdir(assets):
        # Inbox hygiene: stage ONLY the assets the current design.json references, not the
        # whole run_dir/assets pile. Across harness rounds the run's assets dir accumulates
        # superseded slices/masks/host variants (old arrow/decoration/host files) that the
        # final design no longer points at; copying them verbatim left the plugin inbox full
        # of stale files beside the current ones. Keep a file iff its basename appears in the
        # (screenshot-sibling-mutated) design text — this cannot drop a referenced asset
        # (the name is always present when referenced) and purges everything else, so the
        # staged folder is self-evidently the current attempt's asset set.
        design_text = json.dumps(design)
        staged_assets = os.path.join(temp_root, "assets")
        os.makedirs(staged_assets, exist_ok=True)
        for root, _dirs, names in os.walk(assets):
            for name in names:
                src_path = os.path.join(root, name)
                rel = os.path.relpath(src_path, assets)
                if name in design_text:
                    dest_path = os.path.join(staged_assets, rel)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copyfile(src_path, dest_path)
                else:
                    pruned_assets.append(rel.replace(os.sep, "/"))
    for filename in ("preview.png", "qa.json"):
        source = os.path.join(run_dir, filename)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(temp_root, filename))
    # The plugin reads the STAGED design_preflight.json — write the merged (structural +
    # compiler-mirror) report there rather than copying the build-only one.
    with open(os.path.join(temp_root, "design_preflight.json"), "w", encoding="utf-8") as fh:
        json.dump(preflight, fh, indent=2)
    shutil.rmtree(staged_root, ignore_errors=True)
    os.replace(temp_root, staged_root)

    files = []
    for root, _, names in os.walk(staged_root):
        for filename in sorted(names):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, staged_root).replace(os.sep, "/")
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            files.append({"path": rel, "sha256": digest, "bytes": os.path.getsize(path)})
    manifest = {
        "schema_version": design.get("schema_version", design.get("schemaVersion", 1)),
        "doc_id": doc_id,
        # The plugin returns both its compiler report and Figma PNG through the bridge.
        # Scope those callbacks to this exact staged revision so a late callback from an
        # older import can never overwrite the newest run after another upload finishes.
        "roundtrip_token": uuid.uuid4().hex,
        "design": "design.json",
        "staged_dir": os.path.relpath(staged_root, inbox).replace(os.sep, "/"),
        "assets": "assets",
        "files": files,
        "preview": "preview.png" if os.path.exists(os.path.join(staged_root, "preview.png")) else None,
        "export_to": os.path.abspath(os.path.join(run_dir, "figma_export.png")),
        "run_dir": os.path.abspath(run_dir),
        "staged_at": int(time.time()),
        "preflight": {
            "ok": preflight["ok"],
            "errors": preflight["error_count"],
            "warnings": preflight["warn_count"],
            "strict": strict_mode,
        },
        "screenshot_sibling": screenshot_sibling,
        "pruned_assets": pruned_assets,
        "summary": {
            "name": design.get("name"),
            "canvas": design.get("canvas"),
            "layers": (design.get("meta") or {}).get("layer_count", len(design.get("layers") or [])),
            "editable_ratio": (design.get("meta") or {}).get("editable_ratio"),
            "warnings": preflight.get("warnings") or (design.get("meta") or {}).get("warnings") or [],
        },
    }
    manifest_path = os.path.join(inbox, "inbox.json")
    temp_manifest = manifest_path + ".tmp"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_manifest, manifest_path)
    return {"ok": True, "mode": "plugin", "inbox": inbox,
            "doc_id": doc_id, "files": len(files),
            "preflight": {"ok": preflight["ok"], "errors": preflight["error_count"],
                          "warnings": preflight["warn_count"], "strict": strict_mode},
            "screenshot_sibling": screenshot_sibling,
            "pruned_assets": len(pruned_assets),
            "action": "In Figma desktop: run the ad-decompiler plugin → Import latest."}


def export_screenshot(run_dir: str, cfg: dict | None = None, wait_s: int = 0) -> dict:
    """Return path to figma_export.png once the plugin has written it.

    This may poll briefly; the pipeline can also run --resume after the manual import click.
    """
    target = os.path.join(run_dir, "figma_export.png")
    deadline = time.time() + wait_s
    while True:
        if os.path.exists(target):
            return {"ok": True, "path": target}
        if time.time() >= deadline:
            return {"ok": False, "path": target,
                    "note": "figma_export.png not found yet — run the plugin's Import+Export, then re-run QA with --resume"}
        time.sleep(1)
