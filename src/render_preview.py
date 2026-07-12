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
    array = np.asarray(image.convert("RGBA"), dtype=np.uint16).copy()
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
    if isinstance(stroke, str):
        return stroke, 1
    if isinstance(stroke, dict):
        return stroke.get("color", stroke.get("paint", "#000000")), max(1, round(_number(
            stroke.get("width", stroke.get("weight", layer.get("stroke_width", 1))), 1)))
    return None, 0


def _shape_tile(layer, size, run_dir=None):
    from PIL import Image, ImageDraw
    width, height = size
    svg = layer.get("svg")
    is_vector_path = bool(svg or (layer.get("shape_kind") == "path" and layer.get("path")))
    if is_vector_path:
        mask = _svg_or_path_mask(layer, size)
        if mask is not None and mask.getbbox() is not None:
            tile = _fill_tile(size, layer.get("fill"))
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
    tile = _fill_tile(size, layer.get("fill"))
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


def _text_tile(layer, size):
    from PIL import Image, ImageDraw, ImageFont
    style = layer.get("style", {}) or {}
    width, height = size
    tile = Image.new("RGBA", size, (0, 0, 0, 0))
    font_size = max(1, int(round(_number(style.get("fontSize", max(12, height)), max(12, height)))))
    candidates = style.get("fontCandidates") or []
    font_path = next((candidate.get("path") for candidate in candidates
                      if isinstance(candidate, dict) and candidate.get("path") and os.path.exists(candidate["path"])), None)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    line_height = _number(style.get("lineHeight", font_size * 1.2), font_size * 1.2)
    spacing = max(0, round(line_height - font_size))
    align = str(style.get("align", "left")).lower()
    vertical = str(style.get("verticalAlign", style.get("vertical_align", "top"))).lower()
    fill = style.get("color")
    if fill is None:
        fill_spec = layer.get("fill")
        if isinstance(fill_spec, str):
            fill = fill_spec
        elif isinstance(fill_spec, dict):
            fill = fill_spec.get("color", "#111111")
    draw = ImageDraw.Draw(tile)
    text = str(layer.get("text", ""))
    align = align if align in ("left", "center", "right") else "left"
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
    painted_w, painted_h = max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])
    x = 0 if align == "left" else (width - painted_w) / 2 if align == "center" else width - painted_w
    y = 0 if vertical == "top" else (height - painted_h) / 2 if vertical in ("center", "middle") else height - painted_h
    draw.multiline_text(
        (x - bbox[0], y - bbox[1]), text, fill=_color(fill or "#111111"),
        font=font, spacing=spacing, align=align,
    )
    return tile


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
    left, top, right, bottom = _effect_padding(effects)
    padded = Image.new("RGBA", (tile.width + left + right, tile.height + top + bottom), (0, 0, 0, 0))
    for effect in effects or []:
        if not isinstance(effect, dict) or effect.get("visible") is False:
            continue
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


def _render_tile(layer, run_dir):
    from PIL import Image
    box = layer.get("box", {}) or {}
    size = max(1, round(_number(box.get("w", 1), 1))), max(1, round(_number(box.get("h", 1), 1)))
    kind = layer.get("type")
    if kind == "group":
        tile = Image.new("RGBA", size, (0, 0, 0, 0))
        if layer.get("fill"):
            base = _shape_tile({**layer, "shape_kind": "rect"}, size, run_dir)
            tile.alpha_composite(base)
        for child in sorted(layer.get("children") or [], key=lambda entry: _number(entry.get("z_index", entry.get("z", 0)))):
            _draw_layer(tile, child, run_dir)
    elif kind == "image":
        tile = _image_tile(layer, size, run_dir)
    elif kind == "text":
        tile = _text_tile(layer, size)
    else:
        tile = _shape_tile(layer, size, run_dir)
    return _with_effects(tile, layer.get("effects") or [])


def _blend(canvas, tile, point, mode):
    """Composite tile onto an opaque preview, including common non-normal Figma blends."""
    from PIL import ImageChops
    x, y = point
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


def _draw_layer(canvas, layer, run_dir, offset=(0, 0)):
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
    _blend(canvas, tile, (x, y), layer.get("blend_mode", layer.get("blendMode", "NORMAL")))


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
    for index, layer in enumerate(sorted(doc.get("layers", []), key=lambda item: _number(item.get("z_index", item.get("z", 0))))):
        try:
            _draw_layer(canvas, layer, run_dir)
        except Exception as exc:
            # Keep rendering independent layers, but make the omission explicit to the
            # orchestrator. The caller turns these diagnostics into a QA failure.
            errors.append({"layer_id": layer.get("id"), "detail": str(exc)})
        one = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        try:
            _draw_layer(one, layer, run_dir)
        except Exception:
            pass
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in (layer.get("name") or layer.get("id")))[:40]
        one.save(os.path.join(layers_dir, f"{index:02d}_{layer.get('type')}_{safe}.png"))
    out = os.path.join(run_dir, out_name)
    canvas.convert("RGB").save(out)
    _contact_sheet(layers_dir, os.path.join(run_dir, "layers_contact.png"))
    def count(items):
        return sum(1 + count(item.get("children") or []) for item in items)
    return {"preview": out, "layers_dir": layers_dir,
            "count": count(doc.get("layers", [])), "errors": errors}


def _contact_sheet(layers_dir, out, cols=4, thumb=260):
    from PIL import Image
    files = sorted(name for name in os.listdir(layers_dir) if name.endswith(".png"))
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (240, 240, 240))
    for index, name in enumerate(files):
        image = Image.open(os.path.join(layers_dir, name)).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        background.thumbnail((thumb - 8, thumb - 8))
        sheet.paste(background.convert("RGB"), ((index % cols) * thumb + 4, (index // cols) * thumb + 4))
    sheet.save(out)
