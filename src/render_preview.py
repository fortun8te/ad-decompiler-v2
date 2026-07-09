"""render_preview.py — SEE the layers without Figma.

Composites design.json back into a single preview.png (so you can eyeball how close the
reconstruction is) and writes every layer out as its own PNG under layers/ plus a contact
sheet, so you can open a folder and look at the pieces. Pure PIL — runs anywhere, needs no
Figma, no browser, no GPU. This is what closes the loop day-to-day; Figma export is optional.
"""
from __future__ import annotations
import os


def _rgba(size, hexstr, alpha=255):
    from PIL import Image
    h = str(hexstr or "#cccccc").replace("#", "")
    if len(h) == 3:
        h = "".join(c + c for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return Image.new("RGBA", size, (r, g, b, alpha))


def _draw_layer(canvas, L, run_dir, draw_labels=False):
    """Paint one design.json layer onto the canvas (best-effort, 'poor but real')."""
    from PIL import Image, ImageDraw, ImageFont
    box = L.get("box", {})
    x, y = int(box.get("x", 0)), int(box.get("y", 0))
    w, h = max(1, int(box.get("w", 1))), max(1, int(box.get("h", 1)))
    t = L.get("type")

    if t == "image" and L.get("src"):
        p = os.path.join(run_dir, L["src"])
        if os.path.exists(p):
            im = Image.open(p).convert("RGBA").resize((w, h))
            mask = L.get("mask") or {}
            if mask.get("kind") == "ellipse":
                m = Image.new("L", (w, h), 0)
                ImageDraw.Draw(m).ellipse([0, 0, w, h], fill=255)
                im.putalpha(m)
            canvas.alpha_composite(im, (x, y))
            return
        canvas.alpha_composite(_rgba((w, h), "#dddddd", 120), (x, y))
        return

    if t == "shape":
        fill = L.get("fill") or {}
        color = fill.get("color") or (fill.get("stops", [{}])[0].get("color") if fill.get("stops") else "#cccccc")
        tile = _rgba((w, h), color, 255)
        if fill.get("kind") in ("linear", "radial") and fill.get("stops"):
            tile = _gradient((w, h), fill)
        if L.get("shape_kind") == "ellipse":
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).ellipse([0, 0, w, h], fill=255)
            tile.putalpha(m)
        canvas.alpha_composite(tile, (x, y))
        return

    if t == "text":
        st = L.get("style", {})
        color = st.get("color", "#111111")
        size = int(st.get("fontSize", max(12, h)))
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
            except Exception:
                font = ImageFont.load_default()
        d = ImageDraw.Draw(canvas)
        h_ = str(color).replace("#", "")
        if len(h_) == 3:
            h_ = "".join(c + c for c in h_)
        rgb = (int(h_[0:2], 16), int(h_[2:4], 16), int(h_[4:6], 16)) if len(h_) >= 6 else (17, 17, 17)
        d.multiline_text((x, y), L.get("text", ""), fill=rgb + (255,), font=font)
        return


def _gradient(size, fill):
    from PIL import Image
    w, h = size
    stops = fill.get("stops", [])
    if len(stops) < 2:
        return _rgba(size, stops[0].get("color") if stops else "#cccccc")
    c0 = stops[0]["color"].replace("#", ""); c1 = stops[-1]["color"].replace("#", "")
    a = tuple(int(c0[i:i+2], 16) for i in (0, 2, 4))
    b = tuple(int(c1[i:i+2], 16) for i in (0, 2, 4))
    vertical = 45 <= (fill.get("angle", 0) % 180) <= 135
    img = Image.new("RGBA", size)
    px = img.load()
    n = h if vertical else w
    for i in range(n):
        f = i / max(1, n - 1)
        col = tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3)) + (255,)
        if vertical:
            for xx in range(w):
                px[xx, i] = col
        else:
            for yy in range(h):
                px[i, yy] = col
    return img


def render(design_or_path, run_dir, out_name="preview.png"):
    """Composite design.json -> preview.png; also dump each layer + a contact sheet."""
    from PIL import Image
    from .schema import load
    doc = load(design_or_path) if isinstance(design_or_path, str) else design_or_path
    W, H = int(doc["canvas"]["w"]), int(doc["canvas"]["h"])
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    layers_dir = os.path.join(run_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    for i, L in enumerate(doc.get("layers", [])):
        _draw_layer(canvas, L, run_dir)
        # per-layer swatch on transparent canvas, so you can look at each piece alone
        one = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        _draw_layer(one, L, run_dir)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (L.get("name") or L.get("id")))[:40]
        one.crop((0, 0, W, H)).save(os.path.join(layers_dir, f"{i:02d}_{L.get('type')}_{safe}.png"))

    out = os.path.join(run_dir, out_name)
    canvas.convert("RGB").save(out)
    _contact_sheet(layers_dir, os.path.join(run_dir, "layers_contact.png"))
    return {"preview": out, "layers_dir": layers_dir, "count": len(doc.get("layers", []))}


def _contact_sheet(layers_dir, out, cols=4, thumb=260):
    from PIL import Image
    files = sorted(f for f in os.listdir(layers_dir) if f.endswith(".png"))
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (240, 240, 240))
    for i, f in enumerate(files):
        im = Image.open(os.path.join(layers_dir, f)).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        bg.thumbnail((thumb - 8, thumb - 8))
        sheet.paste(bg.convert("RGB"), ((i % cols) * thumb + 4, (i // cols) * thumb + 4))
    sheet.save(out)
