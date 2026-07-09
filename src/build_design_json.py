"""build_design_json.py — assemble routed candidates into design.json (the source of truth).

Deterministic. Consumes the merged+routed candidate list and emits a schema.DesignDoc:
paint order back→front, semantic layer names, scene text collected into kept_in_photo.
No model, no pixels beyond copying asset crops into the run dir.
"""
from __future__ import annotations
import os, shutil
from .schema import DesignDoc, Layer, dump

# z-order bands (lower = further back)
_Z = {"base": 0, "scrim": 1, "image": 2, "shape": 3, "icon": 4, "text": 5, "badge": 6}


def _truncate(s, n=24):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _name(c: dict) -> str:
    meta = c.get("meta", {})
    role = (meta.get("role") or "").strip()
    t = c.get("target")
    if t == "text":
        return f'{(role or "Text").title()} — "{_truncate(c.get("text"))}"'
    if t == "image":
        if meta.get("wordmark"):
            return f'Logo — {_truncate(c.get("text") or "wordmark")}'
        return f'{(role or "Image").title()} — cutout'
    if t == "icon":
        return f'{(role or "Icon").title()} — vector'
    if t == "shape":
        return f'{(role or "Shape").title()}'
    return role.title() or "Layer"


def build(candidates: list, canvas: dict, run_dir: str, base_src: str | None = None,
          doc_id: str = "doc", name: str = "ad") -> dict:
    layers: list[Layer] = []
    kept: list[str] = []
    assets_dir = os.path.join(run_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # base layer (full-bleed photo/flat) first
    if base_src:
        dst = os.path.join(assets_dir, "base.png")
        if os.path.abspath(base_src) != os.path.abspath(dst):
            shutil.copyfile(base_src, dst)
        layers.append(Layer(id="base", type="image", name="Background — base",
                            box={"x": 0, "y": 0, "w": canvas["w"], "h": canvas["h"]},
                            src="assets/base.png", meta={"source": "base"}))

    for i, c in enumerate(candidates):
        t = c.get("target")
        if t == "drop":
            if c.get("text"):
                kept.append(str(c["text"]).strip())
            continue
        lid = c.get("id") or f"n{i}"
        box = c.get("box", {})
        meta = dict(c.get("meta", {}))
        meta.setdefault("source", c.get("source"))
        meta.setdefault("confidence", c.get("confidence"))
        z = _Z.get("text" if t == "text" else "icon" if t == "icon"
                   else "shape" if t == "shape" else "image")

        if t == "text":
            layers.append(Layer(id=lid, type="text", name=_name(c), box=box,
                                text=c.get("text"), style=c.get("style", {}),
                                effects=c.get("effects", []), meta={**meta, "z": z}))
        elif t == "shape":
            layers.append(Layer(id=lid, type="shape", name=_name(c), box=box,
                                shape_kind=c.get("shape_kind", "rect"),
                                fill=c.get("fill"), path=c.get("path"),
                                effects=c.get("effects", []), meta={**meta, "z": z}))
        elif t == "icon":
            # icon carries vector paths in c['paths']; store first as path d-string list joined
            paths = c.get("paths") or []
            d = " ".join(p.get("d", "") for p in paths) if paths else c.get("path")
            layers.append(Layer(id=lid, type="shape", name=_name(c), box=box,
                                shape_kind="path", path=d,
                                fill=c.get("fill") or (paths[0].get("fill") if paths else None),
                                meta={**meta, "z": z, "vector_paths": paths}))
        else:  # image
            src = c.get("src")
            rel = None
            if src and os.path.exists(src):
                base = os.path.basename(src)
                dst = os.path.join(assets_dir, f"{lid}_{base}")
                shutil.copyfile(src, dst)
                rel = os.path.relpath(dst, run_dir)
            layers.append(Layer(id=lid, type="image", name=_name(c), box=box,
                                src=rel, mask=c.get("mask"),
                                effects=c.get("effects", []), meta={**meta, "z": z}))

    # stable paint order: base first, then by z band, then original order
    ordered = sorted(layers, key=lambda L: (0 if L.id == "base" else 1, L.meta.get("z", 3)))
    doc = DesignDoc(id=doc_id, name=name, canvas={"w": canvas["w"], "h": canvas["h"]},
                    layers=ordered, kept_in_photo=sorted(set(kept)),
                    meta={"layer_count": len(ordered)})
    dump(doc, os.path.join(run_dir, "design.json"))
    return doc
