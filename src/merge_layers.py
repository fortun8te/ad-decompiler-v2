"""merge_layers.py — stage 6: fuse OCR + element_detect + Qwen into routed candidates.

merge(ocr, elements, qwen, canvas, cfg) turns the three evidence streams into ONE
candidate list that build_design_json can consume:

  * every OCR line          -> a text candidate (crisp text + box)
  * every detected element  -> a shape / icon / photo candidate (crisp box + mask)
  * every Qwen layer        -> supplies back-to-front z-order + clean RGBA alpha,
                               matched to overlapping OCR/element candidates by IoU

Each candidate is passed through routing.route(candidate, canvas, cfg), which sets
`target` in {text, shape, image, icon, drop}. routing.py is owned by another builder;
if it is not importable yet we fall back to a conservative inline router (used ONLY at
runtime, never written to disk).

Post-routing cleanup:
  * dedup: a shape/icon candidate that is essentially an OCR text box is dropped
    (prefer the editable text candidate)
  * scene text: an OCR line sitting inside a photo region is flagged
    meta.kept_in_photo=True and routed to `drop` (it stays baked into the base)

Candidates carry everything downstream needs: box, target, z, and either
text/style, or src(alpha png)+mask, or source_crop, plus meta{source,role,confidence}.
"""
from __future__ import annotations
import importlib
import os
import copy
import json
import re
from typing import Optional
from .wordmark import is_platform_lockup, semantic_text_role
from .raster_clusters import is_intentional_raster_cluster


def _load_routing():
    """Return the real routing.route, or a conservative inline fallback."""
    for name in ("src.routing", "routing"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "route"):
                return mod.route, True
        except ImportError:
            continue
    return _fallback_route, False


def _iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def _inside_frac(a, b):
    """Fraction of a inside b."""
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return (ix * iy) / max(1.0, a["w"] * a["h"])


def _raster_cluster_owner(box: dict, owners: list[dict], threshold: float):
    """Return the smallest positive intentional-raster owner around ``box``."""
    area = float(box.get("w", 0) or 0) * float(box.get("h", 0) or 0)
    matches = [
        owner for owner in owners
        if (float((owner.get("box") or {}).get("w", 0) or 0)
            * float((owner.get("box") or {}).get("h", 0) or 0)) >= area
        and _inside_frac(box, owner.get("box") or {}) >= threshold
    ]
    if not matches:
        return None
    return min(matches, key=lambda owner: (
        owner["box"]["w"] * owner["box"]["h"], owner["id"],
    ))


def _normalize_text_key(value):
    return " ".join(str(value or "").strip().upper().split())


def _dedup_text_candidates(candidates, merge_cfg: dict, dedup_iou: float):
    """Collapse ghosted/duplicate OCR text layers flagged by the harness or VLM critic."""
    if not merge_cfg.get("dedup_text"):
        return candidates
    drop_ids = set()
    layer_ids = [str(layer_id) for layer_id in (merge_cfg.get("layer_ids") or []) if layer_id]
    if layer_ids:
        drop_ids.update(layer_ids[1:])
    duplicate_texts = {
        _normalize_text_key(text)
        for text in (merge_cfg.get("duplicate_text") or [])
        if _normalize_text_key(text)
    }
    if duplicate_texts:
        texts = [
            c for c in candidates
            if c.get("target") == "text" and c.get("id") not in drop_ids
        ]
        by_text = {}
        for candidate in texts:
            key = _normalize_text_key(candidate.get("text"))
            if key not in duplicate_texts:
                continue
            by_text.setdefault(key, []).append(candidate)
        for group in by_text.values():
            if len(group) < 2:
                continue
            group.sort(
                key=lambda c: (
                    float((c.get("meta") or {}).get("confidence", 0) or 0),
                    -_inside_frac(c.get("box", {}), c.get("box", {})),
                    str(c.get("id", "")),
                ),
                reverse=True,
            )
            keeper = group[0]
            for duplicate in group[1:]:
                if _iou(keeper.get("box", {}), duplicate.get("box", {})) >= dedup_iou:
                    drop_ids.add(duplicate.get("id"))
                    keeper.setdefault("meta", {}).setdefault("deduped_text_ids", []).append(
                        duplicate.get("id")
                    )
    if not drop_ids:
        return candidates
    return [c for c in candidates if c.get("id") not in drop_ids]


# ── conservative inline router (fallback only) ───────────────────────────────────────
def _fallback_route(candidate: dict, canvas: dict, cfg: dict) -> dict:
    """Minimal port of routing intent, used ONLY if src.routing is unimportable.
    Reads the same top-level fields the real router does. Returns a copy."""
    c = dict(candidate)
    meta = c.setdefault("meta", {})
    kind = c.get("kind")
    if c.get("text") is not None and kind in (None, "text"):
        if c.get("kept_in_photo") or meta.get("origin") == "scene":
            c["target"] = "drop"
            meta["kept_in_photo"] = True
        else:
            c["target"] = "text"
    elif kind == "photo-fragment" or meta.get("role") in ("photo", "product", "person"):
        c["target"] = "image"
    elif kind == "icon" or meta.get("role") in ("icon", "badge", "logo", "arrow"):
        c["target"] = "icon"
    elif kind == "shape":
        c["target"] = "shape"
    else:
        c["target"] = "image"
        meta["fallback"] = True
    return c


# ── candidate builders ───────────────────────────────────────────────────────────────
def _word_aligned_text_runs(line: dict) -> list[dict]:
    """Map positively evidenced OCR-word styles to exact character ranges.

    OCR engines disagree about punctuation and whitespace, so matching is sequential
    and fail-closed. A word without ``style_evidence`` is deliberately left in the
    node's base style.
    """
    text = str(line.get("text") or "")
    base_signature = _run_style_signature(dict(line.get("style") or {}))
    cursor = 0
    runs = []
    for word in line.get("words") or []:
        if not isinstance(word, dict) or not word.get("style_evidence"):
            continue
        token = str(word.get("text") or "").strip()
        style = dict(word.get("style") or {})
        if not token or not style or _run_style_signature(style) == base_signature:
            continue
        # Exact search first. Whitespace-tolerant matching supports OCR tokens such
        # as ``30 %`` without allowing punctuation/content to drift.
        start = text.find(token, cursor)
        end = start + len(token) if start >= 0 else -1
        if start < 0 and any(ch.isspace() for ch in token):
            pattern = r"\s+".join(re.escape(part) for part in token.split())
            match = re.search(pattern, text[cursor:])
            if match:
                start, end = cursor + match.start(), cursor + match.end()
        if start < 0:
            continue
        runs.append({"start": start, "end": end, "style": copy.deepcopy(style)})
        cursor = end
    return runs


def _text_candidate(line):
    meta = dict(line.get("meta") or {})
    if line.get("baseline"):
        meta["baseline"] = dict(line["baseline"])
    declared_role = line.get("role") or meta.get("role")
    # OCR normally labels every observation merely "text". Replace only that
    # placeholder with a semantic role so Figma layers are easy to work with.
    if not declared_role or str(declared_role).strip().lower() in {"text", "body"}:
        declared_role = meta.get("semantic_role") or "text"
    meta.update({
        "source": "ocr",
        "role": declared_role,
        "confidence": round(float(line.get("conf", meta.get("confidence", 1.0))), 4),
        "ocr_id": line["id"],
        "line_ids": line.get("line_ids") or [line["id"]],
        "quad": line.get("quad"),
        "hierarchy": line.get("hierarchy"),
        "style_id": line.get("repeated_style_id") or line.get("style_id"),
    })
    return {
        "id": f"c_{line['id']}",
        "box": dict(line["box"]),
        "z": 0,
        "text": line.get("text", ""),
        "style": dict(line.get("style") or {}),
        # Paragraph assembly can preserve distinct styles for each source line.  Keep
        # those exact character ranges all the way to design.json; otherwise Figma sees
        # only the block's representative style and silently flattens bold/colour/font
        # changes across its lines.
        "text_runs": copy.deepcopy(line.get("text_runs") or _word_aligned_text_runs(line)),
        "visible_box": dict(line.get("ink_box") or line.get("painted_box") or line["box"]),
        "rotation": float(line.get("rotation", 0.0) or 0.0),
        "quad": line.get("quad"),
        "meta": meta,
    }


_RUN_VISUAL_STYLE_KEYS = (
    "fontFamily", "font_family", "family", "fontStyle", "font_style",
    "fontWeight", "font_weight", "weight", "fontSize", "font_size", "size",
    "italic", "color", "colorRGB", "fill", "fills", "stroke", "strokes",
    "letterSpacing", "letter_spacing", "tracking", "lineHeight", "line_height",
    "leading", "textDecoration", "text_decoration", "decoration", "textCase",
    "text_case", "opacity",
)


def _run_style_signature(style: dict) -> str:
    """Return a stable visual-style key without letting evidence metadata create runs."""
    visible = {key: style[key] for key in _RUN_VISUAL_STYLE_KEYS if key in style}
    return json.dumps(visible, sort_keys=True, separators=(",", ":"), default=str)


def _line_aligned_text_runs(block_text: str, members: list[dict]) -> list[dict]:
    """Build per-line Figma ranges only when a paragraph really changes visual style.

    ``text_analysis`` builds a paragraph by joining OCR lines with literal newlines.  Check
    that invariant before emitting ranges so a malformed or post-corrected block cannot style
    the wrong characters.  Newline separators intentionally receive the node's base style.
    """
    texts = [str(member.get("text") or "") for member in members]
    if block_text != "\n".join(texts):
        return []
    styles = [dict(member.get("style") or {}) for member in members]
    line_styles_differ = len({_run_style_signature(style) for style in styles}) > 1
    runs = []
    cursor = 0
    for index, (text, style) in enumerate(zip(texts, styles)):
        end = cursor + len(text)
        word_runs = _word_aligned_text_runs(members[index])
        if end > cursor and line_styles_differ:
            runs.append({"start": cursor, "end": end, "style": copy.deepcopy(style)})
        if word_runs:
            # Preserve only the exceptional word ranges; the paragraph node's base
            # style already covers the rest of a same-style line.
            for run in word_runs:
                shifted = copy.deepcopy(run)
                shifted["start"] += cursor
                shifted["end"] += cursor
                runs.append(shifted)
        cursor = end + (1 if index < len(texts) - 1 else 0)
    return runs


def _inline_repeat_pattern(text: str) -> Optional[dict]:
    """Recognise an explicit ticker/marquee phrase repeated three or more times."""
    parts = [part.strip() for part in re.split(r"\s*(?:[•·|]|—|–)\s*", text) if part.strip()]
    if len(parts) < 3:
        return None
    normalized = [_normalize_text_key(part) for part in parts]
    if not normalized[0] or any(part != normalized[0] for part in normalized[1:]):
        return None
    return {"phrase": parts[0], "count": len(parts), "source": "exact-ocr-sequence"}


def _annotate_native_text_repetitions(candidates: list[dict]) -> None:
    """Label only exact, style-consistent repetition; never synthesize missing copies."""
    groups: dict[tuple, list[dict]] = {}
    for candidate in candidates:
        inline = _inline_repeat_pattern(str(candidate.get("text") or ""))
        if inline:
            candidate.setdefault("meta", {})["native_repeat"] = inline
            candidate["meta"]["repeat_signature"] = (
                "ticker:" + _normalize_text_key(inline["phrase"])
            )
        key = (
            _normalize_text_key(candidate.get("text")),
            _run_style_signature(dict(candidate.get("style") or {})),
        )
        if key[0]:
            groups.setdefault(key, []).append(candidate)
    for (text_key, _style), matches in groups.items():
        if len(matches) < 2:
            continue
        xs = [float(item["box"].get("x", 0)) for item in matches]
        ys = [float(item["box"].get("y", 0)) for item in matches]
        hs = [max(1.0, float(item["box"].get("h", 1))) for item in matches]
        row = max(ys) - min(ys) <= max(hs) * .65
        column = max(xs) - min(xs) <= max(hs) * .9
        if not (row or column):
            continue
        signature = "text-repeat:" + text_key
        ordered = sorted(matches, key=lambda item: (
            float(item["box"].get("x", 0)) if row else float(item["box"].get("y", 0)),
            str(item.get("id") or ""),
        ))
        for index, item in enumerate(ordered):
            item.setdefault("meta", {}).update({
                "repeat_signature": signature,
                "native_repeat": {
                    "count": len(ordered), "index": index,
                    "axis": "horizontal" if row else "vertical",
                    "source": "exact-ocr-observations",
                },
            })


def _text_sources(ocr):
    if not isinstance(ocr, dict):
        return ocr or []
    blocks = ocr.get("blocks") or []
    if not blocks:
        return ocr.get("lines", [])
    styles = {style.get("id"): style for style in (ocr.get("styles") or [])}
    lines = {line.get("id"): line for line in (ocr.get("lines") or [])}
    out = []
    represented_line_ids = set()
    for raw in blocks:
        block = dict(raw)
        members = [lines[line_id] for line_id in block.get("line_ids", []) if line_id in lines]
        represented_line_ids.update(line.get("id") for line in members if line.get("id"))
        style_id = block.get("style_id")
        block_style = dict(block.get("style") or styles.get(style_id) or
                           (members[0].get("style") if members else {}) or {})
        if members:
            representative = members[0].get("style") or {}
            for key in ("fontCandidates", "fontSizeCandidates", "fontWeightCandidates",
                        "fontStyleCandidates", "confidence"):
                if key in representative:
                    block_style.setdefault(key, representative[key])
            # Font judging runs after block construction and updates OCR lines. Carry
            # its promoted winner into the actual downstream block instead of leaving
            # the stale pre-judge family on the editable Figma node.
            judged_member = next((line for line in members if line.get("vlm_font_judged")), None)
            if judged_member:
                judged_style = judged_member.get("style") or {}
                for key in ("fontFamily", "fontStyle", "fontWeight", "fontCandidates"):
                    if key in judged_style:
                        block_style[key] = copy.deepcopy(judged_style[key])
        # Blocks are the actual downstream text nodes. Preserve paragraph facts on the
        # block itself rather than relying on the first OCR line's one-line style.
        block_style.setdefault("align", block.get("alignment") or
                               (members[0].get("style") or {}).get("align") if members else "LEFT")
        if block.get("line_height") is not None:
            block_style["lineHeight"] = block["line_height"]
        block_style["lineCount"] = max(1, len(members), str(block.get("text") or "").count("\n") + 1)
        if members and members[0].get("baseline"):
            block.setdefault("meta", {})["baseline_first"] = dict(members[0]["baseline"])
        if members and members[-1].get("baseline"):
            block.setdefault("meta", {})["baseline_last"] = dict(members[-1]["baseline"])
        block["style"] = block_style
        text_runs = _line_aligned_text_runs(str(block.get("text") or ""), members)
        if text_runs:
            block["text_runs"] = text_runs
        else:
            # A raw/resumed block can carry stale ranges from a prior version.  Never let
            # those ranges address a newly assembled paragraph.
            block.pop("text_runs", None)
        block["ink_box"] = block.get("painted_box") or block.get("box")
        block["conf"] = (sum(float(line.get("conf", 1)) for line in members) / len(members)
                         if members else float(block.get("conf", 1)))
        block["rotation"] = (sum(float(line.get("rotation", 0)) for line in members) / len(members)
                             if members else float(block.get("rotation", 0)))
        block["repeated_style_id"] = style_id
        out.append(block)
    # A partial/malformed block list must never delete otherwise valid OCR observations.
    # This happened in a live run where OCR saw text but only the surviving blocks reached
    # merge/design, producing text_recall=0. Preserve every orphan as its own text node.
    out.extend(copy.deepcopy(line) for line_id, line in lines.items()
               if line_id and line_id not in represented_line_ids)
    return out


def _element_candidate(el):
    kind = el.get("kind", "shape")  # top-level kind -> routing.route reads this
    role = el.get("role") or {"shape": "shape", "icon": "icon", "photo-fragment": "photo"}.get(kind, "shape")
    raw_mask = el.get("mask")
    mask_src = (raw_mask.get("src") if isinstance(raw_mask, dict) else raw_mask)
    mask_src = mask_src or el.get("mask_src") or el.get("mask_path")
    structural_meta = {
        key: el.get(key)
        for key in (
            "structure_group_id", "repeat_group_id", "panel_set_id", "grid_group_id",
            "comparison_group_id", "chart_group_id", "row_index", "column_index",
        )
        if el.get(key) not in (None, "")
    }
    metadata = copy.deepcopy(el.get("meta") or {})
    metadata.update({
        "source": "element", "role": role, "kind": kind,
        "confidence": round(float(el.get("score", el.get("coverage", 0.0))), 4),
        "element_id": el["id"], "area": el.get("area"), "prompt": el.get("prompt"),
        "parent_id": el.get("parent_id") or metadata.get("parent_id"),
        "observations": el.get("observation_ids") or metadata.get("observations") or [],
        "provenance": el.get("provenance") or metadata.get("provenance") or {},
        **structural_meta,
    })
    return {
        "id": f"c_{el['id']}",
        "box": dict(el["box"]),
        "z": 0,
        "kind": kind,
        # box-local mask written by element_detect at elements/<id>.png (by convention)
        "mask": {
            "kind": "alpha",
            "src": mask_src or os.path.join("elements", f"{el['id']}.png"),
        },
        "src": el.get("asset_src"),
        "source_crop": {"element_id": el["id"]},
        "meta": metadata,
    }


# ── public API ───────────────────────────────────────────────────────────────────────
def merge(ocr, elements, qwen, canvas, cfg: Optional[dict] = None, run_dir=None):
    cfg = cfg or {}
    # Ownership aggregation is cheap and deterministic. Recompute it at the merge
    # boundary so a routing-policy refinement can resume from merge without paying for
    # every VLM call again.
    if isinstance(ocr, dict) and ocr.get("lines") and ocr.get("blocks"):
        ocr = copy.deepcopy(ocr)
        try:
            from src.vlm_scene_text import _propagate_to_blocks
            _propagate_to_blocks(ocr["lines"], ocr["blocks"])
        except Exception:
            pass
    route, real = _load_routing()
    if run_dir is None:
        run_dir = cfg.get("run_dir")
    dedup_iou = float((cfg.get("merge") or {}).get("dedup_iou", 0.6))
    match_iou = float((cfg.get("merge") or {}).get("qwen_match_iou", 0.3))
    photo_inside = float((cfg.get("merge") or {}).get("photo_inside_frac", 0.82))
    scene_roles = set((cfg.get("merge") or {}).get(
        "scene_text_roles", ["product", "package", "bottle", "jar", "tube", "device", "sign"]
    ))
    overlay_text_roles = {"headline", "title", "subtitle", "subheadline", "eyebrow",
                          "cta", "button", "price", "offer"}

    # text_analysis emits paragraph/headline blocks. Prefer those over one Figma node per
    # OCR line so wrapping, hierarchy, and repeated text styles survive downstream.
    ocr_lines = _text_sources(ocr)
    elements = elements or []
    qwen = qwen or []

    text_cands = [_text_candidate(l) for l in ocr_lines]
    for source, candidate in zip(ocr_lines, text_cands):
        meta = candidate["meta"]
        # OCR commonly reads a single typographic em dash as two separated ASCII
        # hyphens.  Normalise that unambiguous separator before font fitting so a
        # CTA remains both readable and editable instead of inheriting a visible
        # double-hyphen artefact.
        text_value = str(candidate.get("text") or "")
        if "- -" in text_value:
            candidate["text"] = text_value.replace("- -", "—")
            meta["ocr_normalized"] = "double-hyphen-to-emdash"
        if str(meta.get("role") or "").lower() in {"", "text", "body"}:
            meta["role"] = semantic_text_role(source, canvas)
        meta.setdefault("semantic_role", meta["role"])
    _annotate_native_text_repetitions(text_cands)
    elem_cands = [_element_candidate(e) for e in elements]

    # ── qwen z-order + alpha: match each qwen layer to overlapping candidates ──────────
    # qwen list is back-to-front; index = z (lower index = further back).
    scene_regions = []  # product/package regions where printed text should stay baked in
    for zi, ql in enumerate(qwen):
        qbox = ql.get("box", {})
        if not (qbox.get("w", 0) > 0 and qbox.get("h", 0) > 0):
            continue
        # best-matching element candidate gets this layer's clean alpha + z
        best = None
        best_iou = match_iou
        for c in elem_cands:
            iou = _iou(c["box"], qbox)
            if iou > best_iou:
                best_iou, best = iou, c
        if best is not None:
            best["z"] = zi
            best["src"] = ql.get("png")  # clean RGBA alpha from qwen
            best["meta"]["qwen_id"] = ql.get("id")
            best["meta"]["source"] = "element+qwen"
            if best["meta"].get("role") in scene_roles:
                scene_regions.append(qbox)
        else:
            # qwen-only layer with no element match -> raster image candidate.
            # meta.role='photo' makes routing.route pick target='image' + alpha mask.
            elem_cands.append(
                {
                    "id": f"c_{ql['id']}",
                    "box": dict(qbox),
                    "z": zi,
                    "src": ql.get("png"),
                    "meta": {
                        "source": "qwen",
                        "role": "photo",
                        "confidence": 0.5,
                        "qwen_id": ql.get("id"),
                    },
                }
            )
    # Only product/package-like rasters own printed scene text. Generic photos/backgrounds
    # often have intentional editable overlay copy, so containment alone is not enough.
    for c in elem_cands:
        if c["meta"].get("role") in scene_roles or c["meta"].get("contains_scene_text"):
            scene_regions.append(c["box"])
    raster_cluster_owners = [
        c for c in elem_cands
        if is_intentional_raster_cluster(c.get("meta", {}).get("role"))
    ]
    # Do not rebuild buttons, icons and other internal chrome detected *inside* a
    # screenshot/UI crop. The screenshot is the exact, swappable visual owner. Only a
    # separately corroborated external overlay may escape; a surrounding source-evidenced
    # card/shell remains eligible because it contains (rather than sits inside) the crop.
    for child in elem_cands:
        if child in raster_cluster_owners:
            child["meta"].setdefault("decomposition_policy", {
                "outer_shell": "source-evidenced-only",
                "internal_chrome": "baked-in-raster-owner",
            })
            continue
        owner = _raster_cluster_owner(child.get("box") or {}, raster_cluster_owners, photo_inside)
        if owner is None:
            continue
        meta = child.setdefault("meta", {})
        ownership = meta.get("ownership_decision") or {}
        positive_external = bool(
            meta.get("external_overlay") or meta.get("extract_from_cluster")
            or meta.get("separate_layer") or ownership.get("action") == "recreate"
        )
        if positive_external:
            meta["parent_id"] = owner["id"]
            meta["raster_cluster_owner"] = owner["id"]
            continue
        meta.update({
            "layer_disposition": "plate",
            "keep_in_background": True,
            "baked_owner_id": owner["id"],
            "suppression_reason": "internal-chrome-contained-in-raster-cluster",
            "raster_cluster_owner": owner["id"],
        })

    # ── assemble + route ──────────────────────────────────────────────────────────────
    candidates = text_cands + elem_cands

    # scene text: OCR line inside a photo region -> keep baked in the base.
    # VLM scene_text_role overrides geometry when confident; geometry remains fallback.
    for c in text_cands:
        ownership = c["meta"].get("ownership_decision") or {}
        ownership_action = ownership.get("action")
        # Platform lockups are individual artwork layers even when they sit on a
        # social-card raster.  The VLM correctly says "raster_keep" (do not
        # recreate glyphs), but that must mean a cropped/logo asset, not baking
        # the X.com lockup into the whole screenshot.
        if is_platform_lockup(c, canvas):
            c["meta"].update({
                "wordmark": True,
                "platform_lockup": True,
                "role": "platform-logo",
                "semantic_role": "platform-logo",
                "ownership_enforced": True,
                "preserve_underlay": True,
            })
            continue
        # A recognised UI/receipt/chart/table/diagram/product cluster owns its internal
        # pixels as one exact source crop. A generic text role is not enough to extract
        # it: only positive external-overlay evidence can make a contained text layer
        # editable. The parent link keeps such an overlay selectable with the raster.
        owner = _raster_cluster_owner(c["box"], raster_cluster_owners, photo_inside)
        if owner is not None:
            owner_id = owner["id"]
            meta = c["meta"]
            positive_overlay = bool(
                ownership_action == "recreate"
                or meta.get("scene_text_role") == "overlay_copy"
                or meta.get("overlay_text") or meta.get("promote_text")
                or meta.get("editable_text") or meta.get("text_promoted")
            )
            meta["raster_cluster_owner"] = owner_id
            if positive_overlay:
                meta["overlay_text"] = True
                meta["removal_required"] = True
                meta["parent_id"] = owner_id
                meta["external_overlay"] = True
                meta["ownership_enforced"] = True
                continue
            c["kept_in_photo"] = True
            meta["origin"] = "scene"
            meta["role"] = "scene-text"
            meta["baked_owner_id"] = owner_id
            meta["suppression_reason"] = "text-contained-in-intentional-raster-cluster"
            meta["ownership_enforced"] = True
            continue
        # A failed/disagreeing VLM response deliberately becomes raster_keep so it
        # can never cause destructive inpainting by itself.  But that uncertainty
        # must not flatten an otherwise clear high-confidence CTA/headline: OCR
        # hierarchy already proves it is intentional overlay copy.  Promote only
        # explicit overlay roles, never generic body/product text.
        ownership_failed_closed = (
            ownership_action == "raster_keep"
            and str(ownership.get("reason") or "") in {
                "vlm_disagreement", "vlm_error", "vlm_parse_error",
            }
            and float(ownership.get("confidence") or 0) <= 0.01
            and str(c["meta"].get("role") or "").lower() in overlay_text_roles
        )
        if ownership_failed_closed:
            c["meta"].update({
                "overlay_text": True,
                "removal_required": True,
                "ownership_enforced": True,
                "ownership_recovery": "explicit-overlay-role-after-vlm-failure",
            })
            continue
        if ownership_action != "recreate" and ownership:
            c["kept_in_photo"] = True
            c["meta"]["origin"] = "scene"
            c["meta"]["role"] = "scene-text"
            c["meta"]["ownership_enforced"] = True
            continue
        if ownership_action == "recreate":
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
            c["meta"]["ownership_enforced"] = True
            continue
        scene_text_role = c["meta"].get("scene_text_role")
        if scene_text_role == "printed_on_product":
            # A VLM crop classification is advisory. It can easily see a product near
            # overlay copy and label every word "printed". Only bake/drop text when a
            # separately detected product/package region geometrically corroborates it.
            corroborated = any(_inside_frac(c["box"], region) >= photo_inside
                               for region in scene_regions)
            if corroborated:
                c["kept_in_photo"] = True
                c["meta"]["origin"] = "scene"
                c["meta"]["role"] = "scene-text"
                c["meta"]["scene_text_corroborated"] = True
                continue
            c["meta"]["scene_text_uncorroborated"] = True
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
        if scene_text_role == "wordmark":
            c["meta"]["wordmark"] = True
            c["meta"]["role"] = "logo"
            continue
        if scene_text_role == "overlay_copy":
            continue
        if c["meta"].get("role") in overlay_text_roles:
            # Overlay copy must be painted back as editable text, so its original
            # glyphs have to be removed from the Big-LaMa plate first.  Keep this
            # explicit because a downstream router may conservatively return drop.
            c["meta"]["overlay_text"] = True
            c["meta"]["removal_required"] = True
            continue
        for pr in scene_regions:
            if _inside_frac(c["box"], pr) >= photo_inside:
                c["kept_in_photo"] = True
                c["meta"]["origin"] = "scene"
                c["meta"]["role"] = "scene-text"
                break

    # routing.route returns a NEW dict (shallow copy) — capture it.
    routed = []
    for c in candidates:
        try:
            rc = route(c, canvas, cfg)
        except Exception as e:  # a bad router must not sink the whole stage
            print(f"[merge] routing error on {c.get('id')}: {e}; defaulting")
            rc = _fallback_route(c, canvas, cfg)
        # enforce scene-text -> drop regardless of router (contract hard rule)
        if rc.get("kept_in_photo") or rc.get("meta", {}).get("kept_in_photo"):
            rc["target"] = "drop"
        routed.append(rc)
    candidates = routed
    text_cands = [c for c in candidates if c["meta"].get("source") == "ocr"]

    # ── dedup: shape/icon that is really an OCR text box -> drop (prefer text) ─────────
    kept = []
    for c in candidates:
        if c["meta"].get("source") in ("element", "element+qwen") and c.get("target") in (
            "shape",
            "icon",
        ):
            role = c["meta"].get("role")
            if role in ("button", "badge", "chip"):
                kept.append(c)
                continue
            covered = any(
                t.get("target") != "drop"
                and _inside_frac(t["box"], c["box"]) >= 0.9
                and _iou(t["box"], c["box"]) >= dedup_iou
                for t in text_cands
            )
            if covered:
                continue  # the "shape" is just the text's bounding box
            # Button shells are larger than the CTA label — keep the painted backdrop.
            if role in ("shape", "card", "container", None) and any(
                t.get("target") != "drop"
                and (t.get("meta") or {}).get("role") in ("cta", "button", "offer", "price")
                and 0.55 <= _inside_frac(t["box"], c["box"]) < 0.98
                for t in text_cands
            ):
                c.setdefault("meta", {})["role"] = "button"
                kept.append(c)
                continue
        kept.append(c)

    kept = _dedup_text_candidates(kept, cfg.get("merge") or {}, dedup_iou)

    # stable z: keep qwen-derived z, then order remaining by area (large=back)
    def _area(c):
        return c["box"]["w"] * c["box"]["h"]

    max_z = max((c["z"] for c in kept), default=0)
    for c in kept:
        if c["z"] == 0 and c["meta"].get("source") == "ocr":
            c["z"] = max_z + 1  # text sits above shapes by default

    kept.sort(key=lambda c: (c["z"], -_area(c)))

    if run_dir:
        try:
            schema = importlib.import_module("src.schema")
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            schema = importlib.import_module("schema")
        os.makedirs(run_dir, exist_ok=True)
        schema.dump(kept, os.path.join(run_dir, "merged.json"))

    return kept


if __name__ == "__main__":  # CPU-safe smoke with fixtures
    ocr = {
        "lines": [
            {"id": "L0", "text": "BIG SALE", "conf": 0.98,
             "box": {"x": 40, "y": 30, "w": 220, "h": 60}},
            {"id": "L1", "text": "Model wears watch", "conf": 0.8,
             "box": {"x": 300, "y": 400, "w": 180, "h": 24}},
        ]
    }
    elements = [
        {"id": "E0", "box": {"x": 20, "y": 20, "w": 260, "h": 90}, "kind": "shape",
         "area": 20000, "coverage": 0.1, "source": "residual-cc"},
        {"id": "E1", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "kind": "photo-fragment", "area": 40000, "coverage": 0.2,
         "source": "residual-cc"},
    ]
    qwen = [
        {"id": "Q0", "box": {"x": 280, "y": 300, "w": 240, "h": 260},
         "png": "qwen_layers/Q0.png", "kind_hint": "photo"},
    ]
    cands = merge(ocr, elements, qwen, {"w": 600, "h": 600}, {})
    for c in cands:
        print(c["id"], "target=", c.get("target"), "z=", c["z"],
              "role=", c["meta"].get("role"), "kip=", c["meta"].get("kept_in_photo"))
