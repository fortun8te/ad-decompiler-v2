"""Render the canonical scene graph locally, with the same visual primitives as Figma.

This renderer is deliberately independent of the plugin.  It is used by QA, so accepting a
preview that drops a mask, gradient, shadow, opacity or rotation would make the visual score
meaningless.  PIL cannot reproduce every Figma blend operation perfectly, but the supported
subset below has the same geometry and paint semantics and is deterministic on CPU.
"""
from __future__ import annotations

import math
import os


def _color(value, alpha=255):
    """Return RGBA for #rgb, #rrggbb, #rrggbbaa or a small safe fallback."""
    value = str(value or "#cccccc").strip().replace("#", "")
    if len(value) in (3, 4):
        value = "".join(char * 2 for char in value)
    try:
        rgb = tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))
        embedded = int(value[6:8], 16) if len(value) >= 8 else 255
        return rgb + (round(embedded * max(0, min(255, alpha)) / 255),)
    except (TypeError, ValueError):
        return (204, 204, 204, alpha)


def _rgba(size, value, alpha=255):
    from PIL import Image
    return Image.new("RGBA", size, _color(value, alpha))


def _color_close(rgba_a, rgba_b, tol=30.0):
    """True when two RGBA colours are within ``tol`` Euclidean RGB distance."""
    if rgba_a is None or rgba_b is None:
        return False
    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(rgba_a[:3], rgba_b[:3]))) <= tol
    except (TypeError, ValueError):
        return False


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stops(fill):
    raw = list((fill or {}).get("stops") or [])
    if not raw:
        return [(0.0, _color((fill or {}).get("color", "#cccccc"))),
                (1.0, _color((fill or {}).get("color", "#cccccc")))]
    count = len(raw)
    stops = []
    for index, stop in enumerate(raw):
        if not isinstance(stop, dict):
            stop = {"color": stop}
        position = _number(stop.get("position", stop.get("offset", index / max(1, count - 1))),
                           index / max(1, count - 1))
        opacity = _number(stop.get("opacity", stop.get("alpha", 1)), 1)
        if opacity <= 1:
            opacity *= 255
        stops.append((max(0.0, min(1.0, position)), _color(stop.get("color", "#000000"), opacity)))
    return sorted(stops, key=lambda item: item[0])


def _gradient(size, fill):
    """Multi-stop linear/radial paint.  Angles follow the compiler: 0 is left-to-right."""
    import numpy as np
    from PIL import Image

    width, height = size
    stops = _stops(fill)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    kind = str(fill.get("kind", fill.get("type", "linear"))).lower()
    if "radial" in kind:
        cx, cy = (width - 1) / 2, (height - 1) / 2
        radius = max(1.0, math.hypot(cx, cy))
        position = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    else:
        angle = math.radians(_number(fill.get("angle", fill.get("rotation", 0))))
        direction_x, direction_y = math.cos(angle), math.sin(angle)
        cx, cy = (width - 1) / 2, (height - 1) / 2
        extent = max(1.0, abs(direction_x) * max(0, width - 1) + abs(direction_y) * max(0, height - 1))
        position = ((xx - cx) * direction_x + (yy - cy) * direction_y) / extent + 0.5
    position = np.clip(position, 0, 1)
    out = np.empty((height, width, 4), dtype=np.uint8)
    for index in range(len(stops) - 1):
        start_at, start = stops[index]
        end_at, end = stops[index + 1]
        selector = (position >= start_at) & (position <= end_at if index == len(stops) - 2 else position < end_at)
        ratio = np.clip((position - start_at) / max(1e-6, end_at - start_at), 0, 1)
        for channel in range(4):
            values = np.rint(start[channel] + (end[channel] - start[channel]) * ratio).astype(np.uint8)
            out[:, :, channel][selector] = values[selector]
    out[position <= stops[0][0]] = stops[0][1]
    out[position >= stops[-1][0]] = stops[-1][1]
    return Image.fromarray(out)


def _fill_tile(size, fill):
    fill = fill or {}
    if isinstance(fill, str):
        fill = {"kind": "flat", "color": fill}
    kind = str(fill.get("kind", fill.get("type", "flat"))).lower()
    if kind in ("linear", "radial", "gradient_linear", "gradient_radial") or fill.get("stops"):
        return _gradient(size, fill)
    opacity = _number(fill.get("opacity", fill.get("alpha", 1)), 1)
    if opacity <= 1:
        opacity *= 255
    return _rgba(size, fill.get("color", "#cccccc"), opacity)


def _layer_fill(layer):
    """Match the Figma compiler's paint precedence for style-only native layers."""
    fill = layer.get("fill")
    if fill is not None:
        return fill
    style = layer.get("style") or {}
    fills = style.get("fills") or style.get("paints")
    if isinstance(fills, list) and fills:
        return fills[0]
    return style.get("fill", style.get("background", style.get("color")))


def _radius_mask(size, radius=0, ellipse=False):
    from PIL import Image, ImageDraw
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if ellipse:
        draw.ellipse((0, 0, width - 1, height - 1), fill=255)
        return mask
    if isinstance(radius, dict):
        # PIL has no per-corner primitive.  The most faithful safe preview is the largest
        # shared radius; native Figma still receives all four values through the compiler.
        radius = max((_number(value) for value in radius.values()), default=0)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=max(0, _number(radius)), fill=255)
    return mask


def _multiply_alpha(image, mask):
    import numpy as np
    from PIL import Image
    # Single owned buffer (avoid asarray-view + .copy()).
    array = np.array(image.convert("RGBA"), dtype=np.uint16, copy=True)
    alpha = np.asarray(mask.convert("L"), dtype=np.uint16)
    array[:, :, 3] = (array[:, :, 3] * alpha + 127) // 255
    return Image.fromarray(array.astype(np.uint8))


def _svg_or_path_mask(layer, size):
    """Rasterize a vector clipping path when CairoSVG is present; otherwise no-op."""
    try:
        import io
        import cairosvg
        from PIL import Image
        width, height = size
        svg = layer.get("svg")
        if not svg and layer.get("path"):
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                   f'viewBox="0 0 {width} {height}"><path d="{layer["path"]}" fill="#fff"/></svg>')
        if not svg:
            return None
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=width, output_height=height)
        return Image.open(io.BytesIO(png)).convert("RGBA").getchannel("A")
    except Exception:
        return None


def _mask_for_image(layer, size, run_dir):
    from PIL import Image
    mask = layer.get("mask") or {}
    if not isinstance(mask, dict):
        return None
    kind = str(mask.get("kind", mask.get("type", ""))).lower()
    if kind in ("ellipse", "circle"):
        return _radius_mask(size, ellipse=True)
    if kind in ("rrect", "rounded_rect"):
        return _radius_mask(size, mask.get("radius", layer.get("radius", 16)))
    if kind == "path" or mask.get("path") or mask.get("svg"):
        return _svg_or_path_mask(mask, size)
    source = mask.get("src") or mask.get("source") or mask.get("asset")
    if source:
        path = source if os.path.isabs(source) else os.path.join(run_dir, source)
        if os.path.exists(path):
            return Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS).getchannel("A")
    return None


def _stroke_spec(layer):
    stroke = layer.get("stroke")
    if not stroke:
        strokes = layer.get("strokes") or []
        stroke = strokes[0] if strokes else None
    if not stroke:
        style = layer.get("style") or {}
        strokes = style.get("strokes") or []
        stroke = strokes[0] if strokes else style.get("stroke")
    if isinstance(stroke, str):
        return stroke, 1
    if isinstance(stroke, dict):
        return stroke.get("color", stroke.get("paint", "#000000")), max(1, round(_number(
            stroke.get("width", stroke.get("weight", layer.get("stroke_width", 1))), 1)))
    return None, 0


def _shape_tile(layer, size, run_dir=None):
    from PIL import Image, ImageDraw
    width, height = size
    meta = layer.get("meta") or {}
    if meta.get("native_decoration") and isinstance(meta.get("line"), dict):
        # The production Figma node keeps the authored SVG. Windows benchmark workers
        # may not have libcairo, so render this evidence-backed one-segment path directly
        # for preview/QA instead of silently omitting it.
        line = meta["line"]
        absolute = meta.get("absolute_box") or layer.get("box") or {}
        x0 = _number(line.get("x0")) - _number(absolute.get("x"))
        y0 = _number(line.get("y0")) - _number(absolute.get("y"))
        x1 = _number(line.get("x1")) - _number(absolute.get("x"))
        y1 = _number(line.get("y1")) - _number(absolute.get("y"))
        color, stroke_width = _stroke_spec(layer)
        stroke_width = max(1, round(_number(line.get("thickness"), stroke_width or 2)))
        tile = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(tile).line(
            (x0, y0, x1, y1), fill=_color(color or _layer_fill(layer) or "#000000"),
            width=stroke_width,
        )
        return tile
    star = meta.get("star_primitive")
    if star and layer.get("shape_kind") == "path":
        # A starburst seal (badge shells rebuilt by build_design_json) keeps its authored
        # star PATH for Figma, which draws vectors natively. Preview must not depend on
        # cairosvg for it: libcairo is routinely absent on the Windows benchmark workers,
        # and _svg_or_path_mask would then return None and silently omit the whole seal
        # (016's "45% Off" seal vanished, leaving bare plate). The primitive is exact
        # geometry, so draw the polygon directly — same precedent as native_decoration.
        try:
            box = layer.get("box") or {}
            scale_x = width / max(1.0, _number(box.get("w"), 1) or 1)
            scale_y = height / max(1.0, _number(box.get("h"), 1) or 1)
            points = max(3, int(star.get("points") or 5))
            rotation = _number(star.get("rotation"))
            cx, cy = _number(star.get("cx")), _number(star.get("cy"))
            r_outer, r_inner = _number(star.get("r_outer")), _number(star.get("r_inner"))
            verts = []
            for index in range(points):
                a_out = rotation + 2.0 * math.pi * index / points
                a_in = rotation + 2.0 * math.pi * (index + 0.5) / points
                verts.append(((cx + r_outer * math.cos(a_out)) * scale_x,
                              (cy + r_outer * math.sin(a_out)) * scale_y))
                verts.append(((cx + r_inner * math.cos(a_in)) * scale_x,
                              (cy + r_inner * math.sin(a_in)) * scale_y))
            # Build a polygon MASK and paint the layer's real fill through it (the fill
            # may be flat or a gradient) — same composition the vector branch uses.
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).polygon(verts, fill=255)
            if mask.getbbox() is not None:
                tile = _fill_tile(size, _layer_fill(layer))
                tile.putalpha(_multiply_alpha(tile, mask).getchannel("A"))
                return tile
        except Exception:
            pass  # fall through to the generic vector/raster handling below
    svg = layer.get("svg")
    is_vector_path = bool(svg or (layer.get("shape_kind") == "path" and layer.get("path")))
    if is_vector_path:
        mask = _svg_or_path_mask(layer, size)
        if mask is not None and mask.getbbox() is not None:
            tile = _fill_tile(size, _layer_fill(layer))
            tile.putalpha(_multiply_alpha(tile, mask).getchannel("A"))
            return tile
        # SVG is the editable representation, but it is not a reliable preview
        # representation (malformed paths and transparent SVGs are both common).
        # Reconstructed vector layers carry the source crop as a lossless fallback.
        fallback = layer.get("src")
        if fallback and run_dir:
            path = fallback if os.path.isabs(fallback) else os.path.join(run_dir, fallback)
            if os.path.exists(path):
                return Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        # No source pixels exist: omission is safer than inventing an opaque shape.
        return Image.new("RGBA", size, (0, 0, 0, 0))
    tile = _fill_tile(size, _layer_fill(layer))
    kind = str(layer.get("shape_kind", "rect")).lower()
    ellipse = kind in ("ellipse", "circle")
    mask = _radius_mask(size, layer.get("radius", 0), ellipse)
    tile = _multiply_alpha(tile, mask)
    color, weight = _stroke_spec(layer)
    if color and weight:
        # Stroke falls half outside Figma's CENTER alignment.  Keeping it in the tile is a
        # stable local approximation and prevents cards/buttons looking borderless in QA.
        draw = ImageDraw.Draw(tile)
        inset = max(0, weight // 2)
        bounds = (inset, inset, max(inset, width - 1 - inset), max(inset, height - 1 - inset))
        if ellipse:
            draw.ellipse(bounds, outline=_color(color), width=weight)
        else:
            radius = layer.get("radius", 0)
            if isinstance(radius, dict):
                radius = max((_number(value) for value in radius.values()), default=0)
            draw.rounded_rectangle(bounds, radius=max(0, _number(radius) - inset), outline=_color(color), width=weight)
    return tile


def _normalize_align(value, default="left"):
    token = str(value or default).strip().lower()
    if token in ("center", "centre", "middle"):
        return "center"
    if token in ("right", "end"):
        return "right"
    if token in ("left", "start", "justify", "justified"):
        return "left"
    return default


_TEXT_FONT_CACHE = {}


def _text_font(style, font_size):
    from PIL import ImageFont
    candidates = style.get("fontCandidates") or []
    family = str(style.get("fontFamily") or "").strip().casefold()
    try:
        target_weight = float(style.get("fontWeight") or 400)
    except (TypeError, ValueError):
        target_weight = 400.0
    usable = [candidate for candidate in candidates if isinstance(candidate, dict)
              and candidate.get("path") and os.path.exists(candidate["path"])]

    def _candidate_weight(candidate):
        try:
            return float(candidate.get("weight") or 400)
        except (TypeError, ValueError):
            return 400.0

    # A VLM/font-arbitration pass can promote ``fontFamily`` while retaining the
    # original candidates for provenance. Preview must render the selected family
    # first; otherwise it may draw Comic Sans even though the exported Figma node
    # correctly says Arial.  Within a family, the candidate closest to the node's
    # fontWeight wins: Codia-style weight-split runs (121K bold / weergaven light)
    # only read correctly when Bold text is drawn with the Bold file.
    usable.sort(key=lambda candidate: (
        0 if str(candidate.get("family") or "").strip().casefold() == family else 1,
        abs(_candidate_weight(candidate) - target_weight)))
    selected_paths = []
    # The font matcher intentionally stores only the top few candidates. A
    # later semantic/font decision can select a common family not present in
    # that shortlist; resolve Windows' canonical Arial directly before falling
    # back to a visually unrelated candidate.
    if family in {"arial", "arial mt"}:
        windir = os.environ.get("WINDIR", r"C:\\Windows")
        selected_paths.append(os.path.join(windir, "Fonts",
                                           "arialbd.ttf" if target_weight >= 600 else "arial.ttf"))
    # Prefer weight-matched files; if every candidate is Regular for a Bold node,
    # try a system bold face BEFORE loading the mismatched Regular path.
    # A path's weight is driven ONLY when its candidate was resolved by family name
    # (`family_resolved`): that file was picked for a declared weight, so a variable
    # face must be dialled to it. A matcher-chosen face keeps None — it was fitted at
    # its file's own default instance and must render as that same instance here
    # (see _apply_line_render_fit).
    def _axis_weight(candidate):
        return _candidate_weight(candidate) if candidate.get("family_resolved") else None

    matched = [
        (candidate["path"], _axis_weight(candidate)) for candidate in usable
        if abs(_candidate_weight(candidate) - target_weight) <= 150
    ]
    matched_paths = {path for path, _ in matched}
    mismatched = [
        (candidate["path"], _axis_weight(candidate)) for candidate in usable
        if candidate["path"] not in matched_paths
    ]
    paths = [(path, target_weight) for path in selected_paths] + matched
    if target_weight >= 600:
        paths += [("arialbd.ttf", target_weight)]
    paths += mismatched
    paths += [(name, None) for name in
              ("arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf")]
    size_key = max(1, int(round(_number(font_size, 12))))
    for path, path_weight in paths:
        # The weight picks the VARIABLE INSTANCE, so it must key the cache too — a
        # variable face (Inter[opsz,wght].ttf) renders Regular unless wght is driven,
        # which is how a fontWeight-700 node drew light, narrow ink (font_fit.load_font).
        weight_key = None if path_weight is None else int(round(path_weight))
        cache_key = (path, size_key, weight_key)
        cached = _TEXT_FONT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from src import font_fit

            font = font_fit.load_font(path, size_key, weight_key)
        except Exception:
            continue
        _TEXT_FONT_CACHE[cache_key] = font
        return font
    try:
        return ImageFont.load_default(size_key)
    except Exception:
        return ImageFont.load_default()


# Zero-width joiners/variation selectors are invisible formatting characters; drawing
# them with a text font produces tofu boxes (the "broken emoji" artifact in QA).
_ZERO_WIDTH = {0x200D, 0xFE0E, 0xFE0F}
_EMOJI_FONT_CACHE = {}


def _is_emoji_char(char):
    code = ord(char)
    return (code >= 0x1F000 or 0x2600 <= code <= 0x27BF or 0x2B00 <= code <= 0x2BFF
            or code in (0x231A, 0x231B) or 0x23E9 <= code <= 0x23FA)


def _emoji_font(size):
    """Color-emoji face at ``size`` (Segoe UI Emoji on Windows), or None."""
    size = max(1, int(round(size)))
    if size in _EMOJI_FONT_CACHE:
        return _EMOJI_FONT_CACHE[size]
    from PIL import ImageFont
    font = None
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for path in (os.path.join(windir, "Fonts", "seguiemj.ttf"),
                 "/System/Library/Fonts/Apple Color Emoji.ttc",
                 "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"):
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    _EMOJI_FONT_CACHE[size] = font
    return font


def _char_font(char, font):
    """Face used for one glyph: the emoji face for pictographs when available."""
    if _is_emoji_char(char):
        emoji = _emoji_font(getattr(font, "size", 16))
        if emoji is not None:
            return emoji, True
    return font, False


def _line_advance(font, line, tracking):
    """Rendered width of one line, honoring per-glyph tracking (letterSpacing).

    Measured exactly the way :func:`_draw_tracked_line` advances the pen, so a tile
    sized to this width can never clip the glyphs it is about to draw.  PIL's own
    ``multiline_text`` ignores letter spacing entirely, which is why authored text
    with negative tracking used to overrun its box and clip on the right.
    """
    if not line:
        return 0.0
    width = 0.0
    drawn = 0
    for char in line:
        if ord(char) in _ZERO_WIDTH:
            continue
        glyph_font, _ = _char_font(char, font)
        width += glyph_font.getlength(char)
        drawn += 1
    return width + tracking * max(0, drawn - 1)


def _draw_tracked_line(draw, origin, line, font, fill, tracking, ascent,
                       stroke_fill=None, stroke_width=0):
    x, baseline = origin
    # Draw outline first (Pillow strokes under the fill when both are set), so a
    # CENTER-style stroke never plates opaque ink over the glyph interior.
    # OUTSIDE-align strokes are approximated by a slightly thicker Pillow stroke
    # under the fill so the glyph interior stays readable (Codia / Figma parity).
    stroke_kwargs = {}
    if stroke_fill is not None and stroke_width and stroke_width > 0:
        stroke_kwargs = {
            "stroke_width": max(1, int(round(stroke_width))),
            "stroke_fill": stroke_fill,
        }
    for char in line:
        if ord(char) in _ZERO_WIDTH:
            continue
        glyph_font, embedded = _char_font(char, font)
        try:
            draw.text((x, baseline), char, font=glyph_font, fill=fill, anchor="ls",
                      embedded_color=embedded, **stroke_kwargs)
        except (ValueError, TypeError, OSError):
            try:
                draw.text((x, baseline - ascent), char, font=glyph_font, fill=fill,
                          **stroke_kwargs)
            except (ValueError, TypeError, OSError):
                continue
        x += glyph_font.getlength(char) + tracking


def _text_decoration_kind(style):
    raw = style.get("textDecoration", style.get("text_decoration", style.get("decoration")))
    kind = str(raw or "NONE").upper().replace(" ", "_").replace("-", "_")
    if kind in {"UNDERLINE", "UNDERLINE_SOLID"}:
        return "UNDERLINE"
    if kind in {"STRIKETHROUGH", "LINE_THROUGH", "LINETHROUGH"}:
        return "STRIKETHROUGH"
    return None


def _paint_text_decoration(draw, kind, x0, x1, baseline, ascent, descent, colour, font_size):
    """Draw a native underline / strike so QA preview matches Figma textDecoration."""
    if kind is None or x1 <= x0:
        return
    thickness = max(1, int(round(font_size * 0.06)))
    if kind == "UNDERLINE":
        y = baseline + max(1.0, descent * 0.35)
    else:
        y = baseline - ascent * 0.35
    y0 = int(round(y))
    draw.line((x0, y0, x1, y0), fill=colour, width=thickness)


def _run_segments(layer, text):
    """Validated ``text_runs`` spans covering ``text`` (gaps filled with the base
    style), or None when runs are absent/malformed/overlapping."""
    runs = layer.get("text_runs") or []
    if not isinstance(runs, list) or not runs:
        return None
    spans = []
    for run in runs:
        if not isinstance(run, dict):
            return None
        try:
            start, end = int(run.get("start")), int(run.get("end"))
        except (TypeError, ValueError):
            return None
        if start < 0 or end > len(text) or end <= start:
            return None
        spans.append((start, end, run.get("style") or {}))
    spans.sort(key=lambda item: item[0])
    covered, cursor = [], 0
    for start, end, run_style in spans:
        if start < cursor:
            return None
        if start > cursor:
            covered.append((cursor, start, {}))
        covered.append((start, end, run_style))
        cursor = end
    if cursor < len(text):
        covered.append((cursor, len(text), {}))
    return covered


# Only weight and color honor per-run overrides.  Upstream analysis attaches noisy
# per-word fontSize/letterSpacing/family *evidence* to runs; rendering those verbatim
# jumbles a paragraph (each word at its own sampled size).  The Codia construction
# this mirrors (weight-split runs like "121K" bold inside a Light footer line) only
# ever changes weight and color mid-line — size, family and tracking stay uniform.
_RUN_STYLE_KEYS = ("fontWeight", "color")


def _style_is_italic(style):
    """Whether a (run or node) style is italic/oblique.

    Italic is a real per-run axis that ``_RUN_STYLE_KEYS`` deliberately excludes
    (it only carries weight/colour): a run's italic-ness is encoded by the font
    FILE its ``fontCandidates`` resolve to. This predicate lets the styled path
    activate when runs differ in slant even at the same weight/colour, so a mixed
    headline ("We NEVER" italic / "do this!" upright, 013) renders each run with
    its own upright/italic candidate instead of forcing the node's slant on all.
    """
    token = str((style or {}).get("fontStyle") or "").lower()
    if "italic" in token or "oblique" in token:
        return True
    shear = (style or {}).get("italicShearDeg")
    try:
        return shear is not None and abs(float(shear)) > 0.5
    except (TypeError, ValueError):
        return False


def _text_tile(layer, size):
    """Rasterize a text layer without ever clipping its glyphs.

    The tile is sized to the *measured* rendered text (with letterSpacing applied),
    not to ``box.w``/``box.h``.  The returned offset places that block so its
    alignment anchor (left/center/right, top/middle/bottom) coincides with the box,
    letting text spill outside a too-small box instead of being cut off.

    When ``text_runs`` carry styles that differ from the layer style (Codia-style
    weight-split runs such as "121K"(700) inside a Light line), each run segment is
    measured and drawn with its own font/color so the preview shows the same mixed
    weights the exported Figma text will.
    """
    from PIL import Image, ImageDraw
    style = layer.get("style", {}) or {}
    box_w, box_h = size
    text = str(layer.get("text", ""))
    font_size = max(1, int(round(_number(style.get("fontSize", max(12, box_h)), max(12, box_h)))))
    font = _text_font(style, font_size)

    def _metrics(of_font, of_size):
        try:
            return of_font.getmetrics()
        except Exception:
            return int(of_size * 0.8), int(of_size * 0.2)

    ascent, descent = _metrics(font, font_size)
    line_height = _number(style.get("lineHeight", font_size * 1.2), font_size * 1.2)
    if line_height <= 0:
        line_height = font_size * 1.2
    tracking = _number(style.get("letterSpacing", 0), 0)
    align = _normalize_align(style.get("align", "left"))
    vertical = str(style.get("verticalAlign", style.get("vertical_align", "top"))).lower()
    fill = style.get("color")
    if fill is None:
        fill_spec = layer.get("fill")
        if isinstance(fill_spec, str):
            fill = fill_spec
        elif isinstance(fill_spec, dict):
            fill = fill_spec.get("color", "#111111")
    colour = _color(fill or "#111111")

    stroke_colour = None
    stroke_width = 0.0
    stroke_spec = layer.get("stroke") or style.get("stroke")
    if not stroke_spec:
        strokes = layer.get("strokes") or style.get("strokes") or []
        stroke_spec = strokes[0] if strokes else None
    if isinstance(stroke_spec, str):
        stroke_colour, stroke_width = _color(stroke_spec), max(1.0, font_size * 0.04)
    elif isinstance(stroke_spec, dict):
        raw = stroke_spec.get("color", stroke_spec.get("paint"))
        if isinstance(raw, dict):
            raw = raw.get("color")
        if raw:
            stroke_colour = _color(raw)
            stroke_width = max(0.0, _number(
                stroke_spec.get("width", stroke_spec.get("weight", 1)), 1))
            align_stroke = str(
                stroke_spec.get("strokeAlign", stroke_spec.get("align", "OUTSIDE"))
            ).upper()
            # Pillow grows stroke both ways; bias OUTSIDE so fill stays readable.
            if align_stroke in {"OUTSIDE", "CENTER", "CENTRE", ""} and stroke_width > 0:
                stroke_width = max(stroke_width, stroke_width * 1.15)

    # A stroke whose colour matches the glyph fill contributes no visible outline —
    # it only fattens the ink. Sampling contamination from an adjacent element (009:
    # the blue verified badge produced a white #fefefe stroke over white "UPFRONT")
    # then bloats same-colour glyphs until they merge into an unreadable blob that
    # even OCR-of-render can't recover. Drop it; the fitted weight already carries mass.
    if stroke_colour is not None and _color_close(stroke_colour, colour, 30.0):
        stroke_colour, stroke_width = None, 0.0

    decoration = _text_decoration_kind(style)
    lines = text.split("\n")
    segments = _run_segments(layer, text)
    base_italic = _style_is_italic(style)

    def _segment_italic(run_style):
        # Only an explicit per-run slant signal overrides the node's italic-ness;
        # gap-filled segments ({}) inherit the base style and must not count as a
        # difference.
        if run_style and ("fontStyle" in run_style or run_style.get("italicShearDeg") is not None):
            return _style_is_italic(run_style)
        return base_italic

    styled = bool(segments) and (
        any(run_style.get(key) is not None and run_style.get(key) != style.get(key)
            for _, _, run_style in segments for key in _RUN_STYLE_KEYS)
        or any(_segment_italic(run_style) != base_italic
               for _, _, run_style in segments))

    # Per line: [(segment_text, font, colour, tracking, ascent, descent)]
    line_specs = []
    if styled:
        offset = 0
        for line in lines:
            line_start, line_end = offset, offset + len(line)
            spec = []
            for start, end, run_style in segments:
                seg_start, seg_end = max(start, line_start), min(end, line_end)
                if seg_end <= seg_start:
                    continue
                merged = dict(style)
                for key in _RUN_STYLE_KEYS:
                    value = run_style.get(key)
                    if value is not None:
                        merged[key] = value
                # Run-level candidates may carry the weight-specific font file.
                if run_style.get("fontCandidates"):
                    merged["fontCandidates"] = run_style["fontCandidates"]
                seg_size = font_size
                seg_font = _text_font(merged, seg_size)
                seg_ascent, seg_descent = _metrics(seg_font, seg_size)
                seg_colour = _color(merged.get("color") or fill or "#111111")
                seg_tracking = _number(merged.get("letterSpacing", tracking), tracking)
                spec.append((text[seg_start:seg_end], seg_font, seg_colour,
                             seg_tracking, seg_ascent, seg_descent))
            if not spec:
                spec = [("", font, colour, tracking, ascent, descent)]
            line_specs.append(spec)
            offset = line_end + 1
    else:
        line_specs = [[(line, font, colour, tracking, ascent, descent)] for line in lines]

    widths = [sum(_line_advance(seg_font, seg_text, seg_tracking)
                  for seg_text, seg_font, _, seg_tracking, _, _ in spec)
              for spec in line_specs]
    content_w = max(widths + [0.0])
    line_count = max(1, len(lines))
    max_ascent = max([ascent] + [seg[4] for spec in line_specs for seg in spec])
    max_descent = max([descent] + [seg[5] for spec in line_specs for seg in spec])
    deco_extra = font_size * 0.22 if decoration else 0.0
    content_h = (line_count - 1) * line_height + max_ascent + max_descent + deco_extra

    # A small margin guards side bearings, negative-tracking overshoot, descenders
    # and outside strokes so measured content never touches the tile edge.
    pad = max(2, int(math.ceil(font_size * 0.12 + stroke_width + (deco_extra * 0.5))))
    tile_w = max(1, int(math.ceil(content_w)) + 2 * pad)
    tile_h = max(1, int(math.ceil(content_h)) + 2 * pad)
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    y = float(pad)
    for spec, width in zip(line_specs, widths):
        if align == "center":
            x = pad + (content_w - width) / 2.0
        elif align == "right":
            x = pad + (content_w - width)
        else:
            x = float(pad)
        baseline = y + max_ascent
        line_x0 = x
        for seg_text, seg_font, seg_colour, seg_tracking, seg_ascent, _ in spec:
            _draw_tracked_line(draw, (x, baseline), seg_text, seg_font, seg_colour,
                               seg_tracking, seg_ascent,
                               stroke_fill=stroke_colour, stroke_width=stroke_width)
            x += _line_advance(seg_font, seg_text, seg_tracking)
        if decoration and width > 0:
            deco_colour = colour
            deco_x0, deco_x1 = line_x0, line_x0 + width
            if decoration == "STRIKETHROUGH":
                # Hand-drawn strikes (091) are a foreign ink over the glyphs, not the
                # text colour; honour the sampled decorationColor when analysis carried
                # it. A partial decorationSpan strikes only the struck words (091's
                # "Foggy", not "and Steady"); applied only on single-line nodes where
                # one span is unambiguous.
                override = style.get("decorationColor")
                if override:
                    deco_colour = _color(override)
                span = style.get("decorationSpan")
                if (isinstance(span, (list, tuple)) and len(span) == 2
                        and len(lines) == 1):
                    f0 = max(0.0, min(1.0, _number(span[0], 0.0)))
                    f1 = max(0.0, min(1.0, _number(span[1], 1.0)))
                    if f1 > f0:
                        deco_x0 = line_x0 + f0 * width
                        deco_x1 = line_x0 + f1 * width
            _paint_text_decoration(
                draw, decoration, deco_x0, deco_x1, baseline,
                max_ascent, max_descent, deco_colour, font_size,
            )
        y += line_height

    if align == "center":
        anchor_x = (box_w - content_w) / 2.0
    elif align == "right":
        anchor_x = box_w - content_w
    else:
        anchor_x = 0.0
    if vertical in ("center", "middle"):
        anchor_y = (box_h - content_h) / 2.0
    elif vertical in ("bottom", "end"):
        anchor_y = box_h - content_h
    else:
        anchor_y = 0.0
    return tile, (anchor_x - pad, anchor_y - pad)


def _image_tile(layer, size, run_dir):
    from PIL import Image
    source = layer.get("src")
    path = source if source and os.path.isabs(source) else os.path.join(run_dir, source or "")
    if not source or not os.path.exists(path):
        # A gray rectangle looks like a valid layer and can inflate visual QA. Missing
        # assets must stay visibly absent and be rejected by structural QA instead.
        return Image.new("RGBA", size, (0, 0, 0, 0))
    try:
        tile = Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
    except (OSError, ValueError, SyntaxError):
        return Image.new("RGBA", size, (0, 0, 0, 0))
    clip = _mask_for_image(layer, size, run_dir)
    return _multiply_alpha(tile, clip) if clip is not None else tile


def _effect_padding(effects):
    left = top = right = bottom = 0
    for effect in effects or []:
        if not isinstance(effect, dict) or effect.get("visible") is False:
            continue
        kind = str(effect.get("type", effect.get("kind", ""))).lower().replace("_", "-")
        if kind not in ("drop-shadow", "shadow", "inner-shadow"):
            continue
        offset = effect.get("offset") or {}
        dx = _number(offset.get("x", effect.get("x", effect.get("offsetX", 0))))
        dy = _number(offset.get("y", effect.get("y", effect.get("offsetY", 4))))
        spread = max(0, _number(effect.get("spread", 0)))
        blur = max(0, _number(effect.get("radius", effect.get("blur", 8))))
        extent = math.ceil(spread + blur * 2)
        left = max(left, int(math.ceil(extent - min(0, dx))))
        right = max(right, int(math.ceil(extent + max(0, dx))))
        top = max(top, int(math.ceil(extent - min(0, dy))))
        bottom = max(bottom, int(math.ceil(extent + max(0, dy))))
    return left, top, right, bottom


def _with_effects(tile, effects):
    from PIL import Image, ImageFilter
    active = [
        effect for effect in (effects or [])
        if isinstance(effect, dict) and effect.get("visible") is not False
    ]
    if not active:
        return tile, (0, 0)
    left, top, right, bottom = _effect_padding(active)
    padded = Image.new("RGBA", (tile.width + left + right, tile.height + top + bottom), (0, 0, 0, 0))
    for effect in active:
        kind = str(effect.get("type", effect.get("kind", ""))).lower().replace("_", "-")
        if kind in ("blur", "layer-blur"):
            tile = tile.filter(ImageFilter.GaussianBlur(max(0, _number(effect.get("radius", effect.get("blur", 8))))))
            continue
        if kind not in ("drop-shadow", "shadow"):
            continue
        offset = effect.get("offset") or {}
        dx = round(_number(offset.get("x", effect.get("x", effect.get("offsetX", 0)))))
        dy = round(_number(offset.get("y", effect.get("y", effect.get("offsetY", 4)))))
        spread = max(0, round(_number(effect.get("spread", 0))))
        blur = max(0, _number(effect.get("radius", effect.get("blur", 8))))
        alpha = tile.getchannel("A")
        if spread:
            alpha = alpha.filter(ImageFilter.MaxFilter(max(3, spread * 2 + 1)))
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
        shadow = _rgba(tile.size, effect.get("color", "#00000040"), _number(effect.get("opacity", 1), 1) * 255)
        shadow.putalpha(_multiply_alpha(shadow, alpha).getchannel("A"))
        padded.alpha_composite(shadow, (left + dx, top + dy))
    padded.alpha_composite(tile, (left, top))
    return padded, (-left, -top)


def _layer_effects(layer):
    """Merge layer-level and style-level effects (design.json may keep either)."""
    style = layer.get("style") or {}
    effects = list(layer.get("effects") or [])
    for effect in style.get("effects") or []:
        if effect not in effects:
            effects.append(effect)
    return effects


def _group_text_overflow_pad(children):
    """Extra (L,T,R,B) so a clipsContent group tile does not shear top-line glyphs."""
    left = top = right = bottom = 0
    for child in children or []:
        if child.get("type") != "text":
            continue
        style = child.get("style") or {}
        box = child.get("box") or {}
        font_size = _number(style.get("fontSize", box.get("h", 12)), max(12, _number(box.get("h", 12))))
        stroke_spec = child.get("stroke") or style.get("stroke")
        stroke_w = 0.0
        if isinstance(stroke_spec, dict):
            stroke_w = max(0.0, _number(stroke_spec.get("width", stroke_spec.get("weight", 0)), 0))
        pad = max(2, int(math.ceil(font_size * 0.22 + stroke_w)))
        if _text_decoration_kind(style):
            pad = max(pad, int(math.ceil(font_size * 0.30)))
        el, et, er, eb = _effect_padding(_layer_effects(child))
        cx = _number(box.get("x", 0))
        cy = _number(box.get("y", 0))
        # Only pad the sides where the child sits against / past the group edge.
        if cy < pad:
            top = max(top, int(math.ceil(pad - min(0.0, cy) + et)))
        if cx < pad:
            left = max(left, int(math.ceil(pad - min(0.0, cx) + el)))
        right = max(right, er)
        bottom = max(bottom, int(math.ceil(pad * 0.5 + eb)))
    return left, top, right, bottom


def _render_tile(layer, run_dir):
    from PIL import Image
    box = layer.get("box", {}) or {}
    size = max(1, round(_number(box.get("w", 1), 1))), max(1, round(_number(box.get("h", 1), 1)))
    kind = layer.get("type")
    text_offset = (0.0, 0.0)
    if kind == "group":
        children = layer.get("children") or []
        pad_l, pad_t, pad_r, pad_b = _group_text_overflow_pad(children)
        tile_w = size[0] + pad_l + pad_r
        tile_h = size[1] + pad_t + pad_b
        tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        if _layer_fill(layer) is not None:
            base = _shape_tile({**layer, "shape_kind": "rect"}, size, run_dir)
            tile.alpha_composite(base, (pad_l, pad_t))
        child_offset = (pad_l, pad_t)
        for child in sorted(children, key=lambda entry: _number(entry.get("z_index", entry.get("z", 0)))):
            _draw_layer(tile, child, run_dir, offset=child_offset)
        text_offset = (-float(pad_l), -float(pad_t))
    elif kind == "image":
        tile = _image_tile(layer, size, run_dir)
    elif kind == "text":
        tile, text_offset = _text_tile(layer, size)
    else:
        tile = _shape_tile(layer, size, run_dir)
    padded, effect_offset = _with_effects(tile, _layer_effects(layer))
    return padded, (effect_offset[0] + text_offset[0], effect_offset[1] + text_offset[1])


def _layer_paint_clip(layer, offset=(0, 0)):
    box = layer.get("box") or {}
    vis = layer.get("visible_box") or {}
    ox, oy = _number(offset[0]), _number(offset[1])
    top_bound = oy + min(_number(box.get("y", 0)), _number(vis.get("y", box.get("y", 0))))
    bottom_bound = oy + max(
        _number(box.get("y", 0)) + _number(box.get("h", 0)),
        _number(vis.get("y", 0)) + _number(vis.get("h", 0)),
    )
    if layer.get("type") == "text":
        # Text tiles intentionally paint past box edges (ascenders, OUTSIDE stroke,
        # underline, drop-shadow). Clipping to the nominal box was the 067/131
        # off-frame class; keep horizontal unbounded and give vertical room for
        # stroke + decoration + shadow padding.
        style = layer.get("style") or {}
        font_size = _number(style.get("fontSize", box.get("h", 12)), max(12, _number(box.get("h", 12))))
        stroke_spec = layer.get("stroke") or style.get("stroke")
        stroke_w = 0.0
        if isinstance(stroke_spec, dict):
            stroke_w = max(0.0, _number(stroke_spec.get("width", stroke_spec.get("weight", 0)), 0))
        pad = max(2, int(math.ceil(font_size * 0.35 + stroke_w)))
        if _text_decoration_kind(style):
            pad = max(pad, int(math.ceil(font_size * 0.45)))
        left_pad, top_pad, right_pad, bottom_pad = _effect_padding(_layer_effects(layer))
        top_bound -= pad + top_pad
        bottom_bound += pad + bottom_pad
        # Unbounded horizontal (letterSpacing / overflow runs); vertical padded only.
        return (
            -1_000_000,
            int(math.floor(top_bound - 1)),
            1_000_000,
            int(math.ceil(bottom_bound + 1)),
        )
    left = ox + min(_number(box.get("x", 0)), _number(vis.get("x", box.get("x", 0))))
    right = ox + max(
        _number(box.get("x", 0)) + _number(box.get("w", 0)),
        _number(vis.get("x", 0)) + _number(vis.get("w", 0)),
    )
    return (
        int(math.floor(left - 1)),
        int(math.floor(top_bound - 1)),
        int(math.ceil(right + 1)),
        int(math.ceil(bottom_bound + 1)),
    )


def _clip_tile_to_rect(tile, dest_x, dest_y, clip_rect):
    """Zero alpha outside the layer's declared paint bounds on the canvas."""
    from PIL import Image, ImageDraw
    cx0, cy0, cx1, cy1 = clip_rect
    tx0, ty0 = dest_x, dest_y
    tx1, ty1 = dest_x + tile.width, dest_y + tile.height
    ix0, iy0 = max(cx0, tx0), max(cy0, ty0)
    ix1, iy1 = min(cx1, tx1), min(cy1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return Image.new("RGBA", tile.size, (0, 0, 0, 0))
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rectangle(
        (ix0 - tx0, iy0 - ty0, ix1 - tx0 - 1, iy1 - ty0 - 1),
        fill=255,
    )
    return _multiply_alpha(tile, mask)


def _blend(canvas, tile, point, mode, clip_rect=None):
    """Composite tile onto an opaque preview, including common non-normal Figma blends."""
    from PIL import ImageChops
    x, y = point
    if clip_rect is not None:
        tile = _clip_tile_to_rect(tile, x, y, clip_rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.width, x + tile.width), min(canvas.height, y + tile.height)
    if x1 <= x0 or y1 <= y0:
        return
    source = tile.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    if str(mode or "NORMAL").upper() in ("NORMAL", "PASS_THROUGH"):
        canvas.alpha_composite(source, (x0, y0))
        return
    destination = canvas.crop((x0, y0, x1, y1)).convert("RGBA")
    token = str(mode).upper()
    if token == "MULTIPLY":
        mixed = ImageChops.multiply(destination.convert("RGB"), source.convert("RGB"))
    elif token == "SCREEN":
        mixed = ImageChops.screen(destination.convert("RGB"), source.convert("RGB"))
    elif token in ("LIGHTEN", "LIGHTEN_COLOR"):
        mixed = ImageChops.lighter(destination.convert("RGB"), source.convert("RGB"))
    elif token in ("DARKEN", "DARKEN_COLOR"):
        mixed = ImageChops.darker(destination.convert("RGB"), source.convert("RGB"))
    elif token in ("ADD", "LINEAR_DODGE"):
        mixed = ImageChops.add(destination.convert("RGB"), source.convert("RGB"), scale=1.0, offset=0)
    else:
        canvas.alpha_composite(source, (x0, y0))
        return
    mixed.putalpha(source.getchannel("A"))
    canvas.alpha_composite(mixed, (x0, y0))


def _prepare_layer_draw(layer, run_dir, offset=(0, 0)):
    """Rasterize a layer once (tile + pose). Caller may blend onto multiple canvases."""
    from PIL import Image
    box = layer.get("box", {}) or {}
    tile, local_offset = _render_tile(layer, run_dir)
    x = round(_number(offset[0]) + _number(box.get("x", 0)) + local_offset[0])
    y = round(_number(offset[1]) + _number(box.get("y", 0)) + local_offset[1])
    rotation = _number(layer.get("rotation", 0))
    if rotation:
        before = tile.size
        tile = tile.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
        x -= (tile.width - before[0]) // 2
        y -= (tile.height - before[1]) // 2
    opacity = max(0.0, min(1.0, _number(layer.get("opacity", 1), 1)))
    if opacity < 1:
        alpha = tile.getchannel("A").point(lambda value: round(value * opacity))
        tile.putalpha(alpha)
    mode = layer.get("blend_mode", layer.get("blendMode", "NORMAL"))
    clip = _layer_paint_clip(layer, offset)
    return tile, x, y, mode, clip


def _background_blur_radius(effects):
    """Figma-space radius of a visible background-blur effect, if any (0 if none)."""
    best = 0.0
    for effect in effects or []:
        if not isinstance(effect, dict) or effect.get("visible") is False:
            continue
        kind = str(effect.get("type", effect.get("kind", ""))).lower().replace("_", "-")
        if kind != "background-blur":
            continue
        best = max(best, max(0.0, _number(effect.get("radius", effect.get("blur", 8)))))
    return best


def _apply_backdrop_blur(canvas, region, figma_radius):
    """Blur an already-composited canvas region in place (glass backdrop approximation).

    Converts the Figma-space blur radius to a PIL Gaussian sigma via the measured
    2.272728 scale factor (see src/glass_detect.py) before filtering — feeding the raw
    Figma radius straight into GaussianBlur over-blurs by ~2.27x.
    """
    from PIL import ImageFilter
    from src.glass_detect import figma_radius_to_sigma
    x0, y0, x1, y1 = region
    sigma = figma_radius_to_sigma(figma_radius)
    if x1 <= x0 or y1 <= y0 or sigma <= 0:
        return
    patch = canvas.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(sigma))
    canvas.paste(patch, (x0, y0))


def _draw_layer(canvas, layer, run_dir, offset=(0, 0)):
    tile, x, y, mode, clip = _prepare_layer_draw(layer, run_dir, offset)
    blur_radius = _background_blur_radius(_layer_effects(layer))
    if blur_radius > 0:
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(canvas.width, x + tile.width), min(canvas.height, y + tile.height)
        _apply_backdrop_blur(canvas, (cx0, cy0, cx1, cy1), blur_radius)
    _blend(canvas, tile, (x, y), mode, clip)


def render(design_or_path, run_dir, out_name="preview.png"):
    """Composite ``design.json`` back into a local preview and per-root-layer swatches."""
    from PIL import Image
    from .schema import load
    doc = load(design_or_path) if isinstance(design_or_path, str) else design_or_path
    width, height = int(doc["canvas"]["w"]), int(doc["canvas"]["h"])
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    layers_dir = os.path.join(run_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    # A resumed render can contain fewer layers. Remove stale swatches so the contact
    # sheet remains evidence for this design rather than a mixture of two revisions.
    for filename in os.listdir(layers_dir):
        if filename.endswith(".png"):
            try:
                os.unlink(os.path.join(layers_dir, filename))
            except OSError:
                pass
    errors = []
    swatches = []
    for index, layer in enumerate(sorted(doc.get("layers", []), key=lambda item: _number(item.get("z_index", item.get("z", 0))))):
        prepared = None
        try:
            prepared = _prepare_layer_draw(layer, run_dir)
        except Exception as exc:
            # Keep rendering independent layers, but make the omission explicit to the
            # orchestrator. The caller turns these diagnostics into a QA failure.
            errors.append({"layer_id": layer.get("id"), "detail": str(exc)})
        if prepared is not None:
            try:
                _blend(canvas, prepared[0], (prepared[1], prepared[2]), prepared[3], prepared[4])
            except Exception as exc:
                errors.append({"layer_id": layer.get("id"), "detail": str(exc)})
        one = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if prepared is not None:
            try:
                # Same prepared tile as the preview composite — avoids a second full
                # rasterize (image decode / text / shape) per root layer.
                _blend(one, prepared[0], (prepared[1], prepared[2]), prepared[3], prepared[4])
            except Exception:
                pass
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in (layer.get("name") or layer.get("id")))[:40]
        one.save(os.path.join(layers_dir, f"{index:02d}_{layer.get('type')}_{safe}.png"))
        swatches.append(one)
    out = os.path.join(run_dir, out_name)
    canvas.convert("RGB").save(out)
    _contact_sheet_images(swatches, os.path.join(run_dir, "layers_contact.png"))
    def count(items):
        return sum(1 + count(item.get("children") or []) for item in items)
    return {"preview": out, "layers_dir": layers_dir,
            "count": count(doc.get("layers", [])), "errors": errors}


def _contact_sheet_images(images, out, cols=4, thumb=260):
    """Build the contact sheet from in-memory swatches (no re-decode of layer PNGs)."""
    from PIL import Image
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (240, 240, 240))
    for index, image in enumerate(images):
        rgba = image.convert("RGBA") if image.mode != "RGBA" else image
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        background.thumbnail((thumb - 8, thumb - 8))
        sheet.paste(background.convert("RGB"), ((index % cols) * thumb + 4, (index // cols) * thumb + 4))
    sheet.save(out)


def _contact_sheet(layers_dir, out, cols=4, thumb=260):
    from PIL import Image
    files = sorted(name for name in os.listdir(layers_dir) if name.endswith(".png"))
    if not files:
        return
    images = [Image.open(os.path.join(layers_dir, name)).convert("RGBA") for name in files]
    _contact_sheet_images(images, out, cols=cols, thumb=thumb)
