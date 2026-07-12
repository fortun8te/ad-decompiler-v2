"""vlm_font_judge.py — optional VLM visual font ranking after local font matching.

For each style cluster with local-render font candidates, crops the painted ink region,
renders candidate font previews, and asks a local vision model to rate the match (0-10)
or pick the closer preview (A/B). Two agreeing passes reorder ``fontCandidates``;
scores below a threshold try the next candidate (max 3). Disabled by default and never
raises.
"""
from __future__ import annotations

import copy
import io
import json
import re

from src import vlm_client
from src.text_analysis import _style_cluster_key

_DEFAULT_PASSES = 2
_DEFAULT_MAX_TOKENS = 120
_DEFAULT_PADDING = 4
_DEFAULT_MAX_STYLES = 12
_DEFAULT_MAX_CANDIDATES = 3
_DEFAULT_SCORE_THRESHOLD = 7

_SINGLE_PROMPT = (
    "The image shows two panels side by side. LEFT is the original ad text ink crop. "
    "RIGHT is a font preview rendered with a candidate typeface. "
    "Rate how closely the RIGHT panel matches the original letterforms on the LEFT "
    "(0 = no match, 10 = perfect). "
    "If the original crop is illegible or not text, set reject to true. "
    'Reply with JSON only: {"score": <0-10>, "reject": <bool>}'
)

_COMPARE_PROMPT = (
    "The image shows three panels: LEFT is the original ad text ink crop, "
    "MIDDLE is font preview A, RIGHT is font preview B. "
    "Which preview (A or B) matches the original letterforms better? "
    "If illegible, set reject to true. "
    'Reply with JSON only: {"choice": "A"|"B"|null, "score": <0-10 for the better one>, '
    '"reject": <bool>}'
)


def _font_judge_cfg(cfg: dict) -> dict:
    root = (cfg or {}).get("vlm") or {}
    judge = root.get("font_judge") or {}
    merged = {
        "base_url": root.get("base_url"),
        "model": root.get("model"),
        "timeout_s": root.get("timeout_s"),
        "max_tokens": root.get("max_tokens"),
        "passes": root.get("passes"),
    }
    merged.update({k: v for k, v in judge.items() if k != "enabled"})
    return merged


def _font_matching_enabled(cfg: dict) -> bool:
    fm = ((cfg or {}).get("text_analysis") or {}).get("font_matching")
    if isinstance(fm, bool):
        return fm
    if isinstance(fm, dict):
        return bool(fm.get("enabled"))
    return False


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None


def _parse_single(text: str) -> dict | None:
    data = _extract_json(text)
    if not data:
        return None
    if data.get("reject"):
        return {"reject": True}
    score = data.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if not 0 <= score <= 10:
        return None
    return {"score": score, "reject": False}


def _parse_compare(text: str) -> dict | None:
    data = _extract_json(text)
    if not data:
        return None
    if data.get("reject"):
        return {"reject": True}
    choice = str(data.get("choice") or "").strip().upper()
    if choice not in {"A", "B"}:
        return None
    score = data.get("score")
    try:
        score = float(score) if score is not None else 5.0
    except (TypeError, ValueError):
        score = 5.0
    return {"choice": choice, "score": max(0.0, min(10.0, score)), "reject": False}


def _compose_panels(panels: list):
    from PIL import Image

    valid = [p for p in panels if p is not None]
    if not valid:
        return None
    height = max(p.height for p in valid)
    gap = 6
    resized = []
    for panel in valid:
        if panel.height != height:
            ratio = height / max(1, panel.height)
            resized.append(panel.resize((max(1, int(panel.width * ratio)), height), Image.Resampling.LANCZOS))
        else:
            resized.append(panel)
    width = sum(p.width for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (width, height), (240, 240, 240))
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _hex_colour(style: dict) -> tuple[int, int, int]:
    rgb = style.get("colorRGB")
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    colour = str(style.get("color") or "#141414")
    if colour.startswith("#") and len(colour) >= 7:
        return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
    return 20, 20, 20


def _render_candidate(text: str, candidate: dict, size: float, colour: tuple[int, int, int], target_h: int):
    path = candidate.get("path")
    if not path:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(path, max(8, int(round(size))))
        probe = Image.new("RGB", (8, 8), (255, 255, 255))
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        canvas = Image.new("RGB", (width + 12, height + 12), (255, 255, 255))
        ImageDraw.Draw(canvas).text((6 - bbox[0], 6 - bbox[1]), text, fill=colour, font=font)
        target_w = max(32, int(target_h * max(1.0, width / max(1, height))))
        return canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception:
        return None


def _local_candidates(candidates: list[dict], max_n: int) -> list[dict]:
    out = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        if item.get("source") != "local-render" or not item.get("path"):
            continue
        out.append(item)
        if len(out) >= max_n:
            break
    return out


def _cluster_key(line: dict) -> str:
    style_id = line.get("style_id")
    if style_id:
        return str(style_id)
    style = line.get("style") or {}
    geo = {
        "font_size": style.get("fontSize", 16),
        "weight": style.get("fontWeight", 400),
        "shear_angle": style.get("italicShearDeg"),
    }
    return "cluster:" + repr(_style_cluster_key(geo, style.get("color", "#000000")))


def _promote_candidate(candidates: list[dict], winner: dict, score: float) -> list[dict]:
    key = (str(winner.get("family", "")).lower(), str(winner.get("style", "")).lower())
    rest = []
    promoted = None
    for item in candidates:
        if not isinstance(item, dict):
            rest.append(item)
            continue
        item_key = (str(item.get("family", "")).lower(), str(item.get("style", "")).lower())
        enriched = dict(item)
        if item_key == key:
            enriched["vlm_score"] = round(score, 2)
            promoted = enriched
        else:
            rest.append(enriched)
    if promoted is None:
        promoted = dict(winner)
        promoted["vlm_score"] = round(score, 2)
    return [promoted] + rest


def _apply_style(cluster_lines: list[dict], ranked: list[dict], winner: dict) -> None:
    for line in cluster_lines:
        style = line.setdefault("style", {})
        style["fontCandidates"] = [dict(item) for item in ranked]
        style["fontFamily"] = winner.get("family", style.get("fontFamily"))
        style["fontStyle"] = winner.get("style", style.get("fontStyle"))
        if winner.get("weight") is not None:
            style["fontWeight"] = int(winner["weight"])
        line["vlm_font_judged"] = True


def _judge_cluster(rep: dict, candidates: list[dict], source_crop: bytes, options: dict) -> dict | None:
    style = rep.get("style") or {}
    text = str(rep.get("text") or "").strip() or "Ag"
    size = float(style.get("fontSize") or 24)
    colour = _hex_colour(style)
    target_h = max(32, min(96, int(round((rep.get("painted_box") or rep.get("box") or {}).get("h", 32)))))
    from PIL import Image

    original = Image.open(io.BytesIO(source_crop)).convert("RGB")
    original = original.resize((max(32, target_h * 3), target_h), Image.Resampling.LANCZOS)

    base_url = options["base_url"]
    model = options["model"]
    timeout_s = options["timeout_s"]
    max_tokens = options["max_tokens"]
    passes = options["passes"]
    threshold = options["score_threshold"]
    max_attempts = min(options["max_candidates"], len(candidates))

    if len(candidates) >= 2 and max_attempts >= 2:
        panel_a = _render_candidate(text, candidates[0], size, colour, target_h)
        panel_b = _render_candidate(text, candidates[1], size, colour, target_h)
        composed = _compose_panels([original, panel_a, panel_b])
        if composed:
            answer, note = vlm_client.multi_pass_answer(
                composed, _COMPARE_PROMPT,
                base_url=base_url, model=model, timeout_s=timeout_s,
                max_tokens=max_tokens, passes=passes,
            )
            if note:
                return {"status": note}
            parsed = _parse_compare(answer or "")
            if parsed and parsed.get("reject"):
                return {"status": "rejected"}
            if parsed and parsed.get("choice") in {"A", "B"}:
                idx = 0 if parsed["choice"] == "A" else 1
                score = float(parsed.get("score", 5.0))
                if score >= threshold:
                    return {
                        "status": "ok",
                        "winner": candidates[idx],
                        "score": score,
                        "method": "compare",
                        "attempts": 1,
                    }

    attempts = 0
    for candidate in candidates[:max_attempts]:
        preview = _render_candidate(text, candidate, size, colour, target_h)
        composed = _compose_panels([original, preview])
        if not composed:
            continue
        attempts += 1
        answer, note = vlm_client.multi_pass_answer(
            composed, _SINGLE_PROMPT,
            base_url=base_url, model=model, timeout_s=timeout_s,
            max_tokens=max_tokens, passes=passes,
        )
        if note:
            return {"status": note, "attempts": attempts}
        parsed = _parse_single(answer or "")
        if parsed and parsed.get("reject"):
            return {"status": "rejected", "attempts": attempts}
        if parsed and float(parsed.get("score", 0)) >= threshold:
            return {
                "status": "ok",
                "winner": candidate,
                "score": float(parsed["score"]),
                "method": "single",
                "attempts": attempts,
            }
    return {"status": "below_threshold", "attempts": attempts}


def judge_fonts(image_path: str, ocr_result: dict, cfg: dict) -> dict:
    """Re-rank font candidates per style cluster using optional VLM judging. Never raises."""
    judge = ((cfg or {}).get("vlm") or {}).get("font_judge") or {}
    if not judge.get("enabled", False) or not _font_matching_enabled(cfg):
        return ocr_result

    lines = list(ocr_result.get("lines") or [])
    if not lines:
        return ocr_result

    vcfg = _font_judge_cfg(cfg)
    options = {
        "base_url": str(vcfg.get("base_url") or vlm_client._DEFAULT_BASE_URL),
        "model": str(vcfg.get("model") or vlm_client._DEFAULT_MODEL),
        "timeout_s": float(vcfg.get("timeout_s") or vlm_client._DEFAULT_TIMEOUT_S),
        "max_tokens": int(vcfg.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "padding": int(vcfg.get("padding", _DEFAULT_PADDING)),
        "passes": int(vcfg.get("passes", _DEFAULT_PASSES)),
        "max_candidates": int(vcfg.get("max_candidates", vcfg.get("top_candidates", _DEFAULT_MAX_CANDIDATES))),
        "score_threshold": float(vcfg.get("score_threshold", vcfg.get("min_score", _DEFAULT_SCORE_THRESHOLD))),
        "max_styles": int(vcfg.get("max_styles", _DEFAULT_MAX_STYLES)),
    }

    try:
        from PIL import Image

        image = Image.open(image_path)
    except Exception:
        return ocr_result

    clusters: dict[str, list[dict]] = {}
    for line in lines:
        style = line.get("style") or {}
        if not _local_candidates(style.get("fontCandidates") or [], 1):
            continue
        clusters.setdefault(_cluster_key(line), []).append(line)

    if not clusters:
        return ocr_result

    judged = 0
    updated = 0
    rejected = 0
    disagreements = 0
    errors = 0
    notes: list[dict] = []

    for cluster_key, cluster_lines in list(clusters.items())[: options["max_styles"]]:
        rep = max(cluster_lines, key=lambda ln: float(ln.get("ink_confidence", ln.get("conf", 0))))
        style = rep.get("style") or {}
        candidates = _local_candidates(style.get("fontCandidates") or [], options["max_candidates"])
        if not candidates:
            continue
        box = rep.get("painted_box") or rep.get("box")
        if not box:
            continue
        source_crop = vlm_client.crop_box_bytes(image, box, options["padding"])
        if source_crop is None:
            continue

        judged += 1
        outcome = _judge_cluster(rep, candidates, source_crop, options)
        if not outcome:
            continue
        status = outcome.get("status")
        if status == "ok":
            winner = outcome["winner"]
            ranked = _promote_candidate(style.get("fontCandidates") or candidates, winner, outcome["score"])
            _apply_style(cluster_lines, ranked, winner)
            updated += 1
            notes.append({
                "style_id": cluster_key,
                "winner": winner.get("family"),
                "score": outcome.get("score"),
                "method": outcome.get("method"),
                "attempts": outcome.get("attempts"),
            })
        elif status == "rejected":
            rejected += 1
            notes.append({"style_id": cluster_key, "note": "illegible"})
        elif status == "vlm_disagreement":
            disagreements += 1
            notes.append({"style_id": cluster_key, "note": "vlm_disagreement"})
        elif status == "vlm_error":
            errors += 1
        elif status == "below_threshold":
            notes.append({"style_id": cluster_key, "note": "below_threshold", "attempts": outcome.get("attempts")})

    result = copy.deepcopy(ocr_result)
    result["lines"] = lines
    result["vlm_font_judge"] = {
        "enabled": True,
        "model": options["model"],
        "score_threshold": options["score_threshold"],
        "clusters_judged": judged,
        "clusters_updated": updated,
        "clusters_rejected": rejected,
        "clusters_disagreed": disagreements,
        "clusters_errored": errors,
        "notes": notes,
    }
    return result


__all__ = ["judge_fonts"]
