"""Integration tests for VLM + layout pipeline wiring (mocked — no LM Studio)."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

import run_pipeline


def _test_font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _make_ad(tmp_path):
    source = tmp_path / "input.png"
    image = Image.new("RGB", (200, 140), (245, 240, 230))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "SALE", font=_test_font(28), fill=(20, 20, 20))
    draw.rounded_rectangle((30, 90, 120, 125), radius=8, fill=(200, 70, 40))
    image.save(source)
    return source


def test_pipeline_order_applies_vlm_stages_when_enabled(monkeypatch, tmp_path):
    source = _make_ad(tmp_path)
    calls = {"ocr_judge": 0, "proofread": 0, "scene_text": 0, "font_judge": 0, "segment_filter": 0}
    order = []

    def fake_ocr(path, cfg, run_dir=None):
        order.append("ocr")
        return {
            "engine": "fixture", "status": "ok", "errors": [],
            "source": {"path": path, "w": 200, "h": 140}, "ms": 1,
            "lines": [{
                "id": "L0", "text": "SALE", "conf": 0.55,
                "box": {"x": 20, "y": 20, "w": 80, "h": 36},
                "quad": [[20, 20], [100, 20], [100, 56], [20, 56]], "words": [],
            }],
        }

    def fake_ocr_judge(path, raw, cfg):
        calls["ocr_judge"] += 1
        order.append("vlm-ocr-judge")
        return raw

    def fake_proofread(path, raw, cfg):
        calls["proofread"] += 1
        order.append("vlm-proofread")
        out = dict(raw)
        out["vlm_proofread"] = {"lines_corrected": 1, "lines_checked": 1}
        out["lines"][0]["text"] = "SALE!"
        return out

    def fake_analyze(path, raw, cfg):
        order.append("text")
        line = dict(raw["lines"][0])
        line["style"] = {
            "fontFamily": "Inter",
            "fontSize": 28,
            "fontWeight": 400,
            "fontCandidates": [
                {"family": "Inter", "source": "local-render", "score": 0.7, "path": "/tmp/a.ttf"},
                {"family": "Arial", "source": "local-render", "score": 0.6, "path": "/tmp/b.ttf"},
            ],
            "styleId": "S0",
        }
        line["painted"] = line["box"]
        return {**raw, "lines": [line], "blocks": [], "styles": []}

    def fake_scene_text(path, ocr_res, cfg):
        calls["scene_text"] += 1
        order.append("vlm-scene-text")
        out = dict(ocr_res)
        out["vlm_scene_text"] = {"lines_classified": 1, "lines_checked": 1}
        return out

    def fake_font_judge(path, ocr_res, cfg):
        calls["font_judge"] += 1
        order.append("vlm-font-judge")
        out = dict(ocr_res)
        out["vlm_font_judge"] = {"styles_promoted": 1, "styles_checked": 1}
        return out

    def fake_fuse(**kwargs):
        order.append("fusion")
        return [{"id": "E0", "box": {"x": 30, "y": 90, "w": 90, "h": 35}, "meta": {"role": "badge"}}]

    def fake_segment_filter(path, elements, cfg):
        calls["segment_filter"] += 1
        order.append("vlm-segment-filter")
        return elements[:0]

    def fake_merge(ocr_res, els, qwen, canvas, cfg, run_dir=None):
        order.append("merge")
        return [{"id": "T0", "target": "text", "text": "SALE!", "box": ocr_res["lines"][0]["box"]}]

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    monkeypatch.setattr(run_pipeline.vlm_ocr_judge, "judge_ocr_lines", fake_ocr_judge)
    monkeypatch.setattr(run_pipeline.vlm_proofread, "proofread_lines", fake_proofread)
    monkeypatch.setattr(run_pipeline.text_analysis, "analyze_text", fake_analyze)
    monkeypatch.setattr(run_pipeline.vlm_scene_text, "classify_scene_text", fake_scene_text)
    monkeypatch.setattr(run_pipeline.vlm_font_judge, "judge_fonts", fake_font_judge)
    monkeypatch.setattr(run_pipeline.element_fusion, "fuse", fake_fuse)
    monkeypatch.setattr(run_pipeline.vlm_segment_filter, "filter_elements", fake_segment_filter)
    monkeypatch.setattr(run_pipeline.merge_layers, "merge", fake_merge)
    monkeypatch.setattr(run_pipeline.vram, "stage_boundary", lambda *a, **k: order.append(f"vram:{a[0]}->{a[1]}"))

    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu",
            "vlm": {
                "enabled": True,
                "ocr_judge": {"enabled": True},
                "scene_text": {"enabled": True},
                "font_judge": {"enabled": True},
                "segment_filter": {"enabled": True},
            },
            "text_analysis": {"font_matching": {"enabled": True}},
            "qwen": {"enabled": False},
            "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"},
            "figma": {"enabled": False},
            "qa_ocr": False,
        },
    )

    assert result["ok"] is True
    assert calls == {"ocr_judge": 1, "proofread": 1, "scene_text": 1, "font_judge": 1, "segment_filter": 1}
    assert order.index("ocr") < order.index("vlm-ocr-judge") < order.index("vlm-proofread") < order.index("text")
    assert order.index("text") < order.index("vlm-scene-text") < order.index("vlm-font-judge") < order.index("fusion")
    assert order.index("fusion") < order.index("vlm-segment-filter") < order.index("merge")
    assert (run_dir / "fused_elements.json").exists()
    assert (run_dir / "elements.json").exists()
    intent = json.loads((run_dir / "scene_intent.json").read_text(encoding="utf-8"))
    assert intent["kind"] == "scene-intent"
    assert intent["planned_source_ids"] == ["T0"]
    assert json.loads((run_dir / "elements.json").read_text(encoding="utf-8")) == []
    assert "vram:ocr->vlm-ocr-judge" in order
    assert "vram:ocr->vlm-proofread" in order
    assert "vram:text->vlm-scene-text" in order
    assert "vram:text->vlm-font-judge" in order
    assert "vram:fusion->vlm-segment-filter" in order


def test_vlm_stages_skipped_when_disabled(monkeypatch, tmp_path):
    source = _make_ad(tmp_path)
    font_judge_calls = []

    monkeypatch.setattr(
        run_pipeline.ocr, "run_ocr",
        lambda path, cfg, run_dir=None: {
            "engine": "fixture", "lines": [{
                "id": "L0", "text": "SALE", "conf": 0.99,
                "box": {"x": 20, "y": 20, "w": 80, "h": 36},
            }],
        },
    )
    monkeypatch.setattr(
        run_pipeline.vlm_font_judge, "judge_fonts",
        lambda *a, **k: font_judge_calls.append(1) or a[1],
    )

    run_dir = tmp_path / "run"
    run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu",
            "vlm": {"enabled": False, "segment_filter": {"enabled": False}, "font_judge": {"enabled": False}},
            "qwen": {"enabled": False},
            "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"},
            "text_analysis": {"font_matching": {"enabled": False}},
            "figma": {"enabled": False},
            "qa_ocr": False,
        },
    )

    raw = json.loads((run_dir / "ocr_raw.json").read_text(encoding="utf-8"))
    ocr = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    fused = json.loads((run_dir / "fused_elements.json").read_text(encoding="utf-8"))
    elements = json.loads((run_dir / "elements.json").read_text(encoding="utf-8"))
    assert "vlm_proofread" not in raw
    assert "vlm_font_judge" not in ocr
    assert len(elements) == len(fused)
    assert font_judge_calls == []
