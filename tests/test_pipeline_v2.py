import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

import run_pipeline


def _test_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    candidates += glob.glob("/usr/share/fonts/**/*DejaVuSans.ttf", recursive=True)[:1]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def test_end_to_end_cpu_vertical_slice_uses_clean_plate(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    image = Image.new("RGB", (240, 180), (244, 239, 228))
    draw = ImageDraw.Draw(image)
    # Real glyph ink (not a flat block) so text_analysis's ink/fidelity signals are
    # representative of actual painted text rather than a solid rectangle.
    draw.text((30, 28), "SALE", font=_test_font(34), fill=(25, 25, 25))
    draw.rounded_rectangle((35, 115, 150, 160), radius=10, fill=(210, 70, 45))
    image.save(source)

    def fake_ocr(path, cfg, run_dir=None):
        return {
            "engine": "fixture", "source": {"path": path, "w": 240, "h": 180}, "ms": 1,
            "lines": [{
                "id": "L0", "text": "SALE", "conf": .99,
                "box": {"x": 25, "y": 25, "w": 90, "h": 40},
                "quad": [[25, 25], [115, 25], [115, 65], [25, 65]], "words": [],
            }],
        }

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu",
            "qwen": {"enabled": False},
            "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"},
            "text_analysis": {"font_matching": {"enabled": False}},
            "figma": {"enabled": False},
            "qa_ocr": False,
        },
    )
    assert result["ok"] is True
    assert result["duration_s"] >= 0
    assert (run_dir / "input_manifest.json").exists()
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    assert design["schema_version"] == 2
    assert design["layers"][0]["name"] == "Background — clean plate"
    assert design["layers"][0]["meta"]["source"] == "inpaint"
    assert (run_dir / "background_clean.png").exists()
    assert (run_dir / "ownership.png").exists()
    assert (run_dir / "removal_mask.png").exists()
    assert any(layer["type"] == "text" and layer["style"].get("fontSize")
               for layer in design["layers"])


def test_required_sam_fallback_is_a_real_qa_failure(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    image = Image.new("RGB", (120, 80), (244, 239, 228))
    ImageDraw.Draw(image).rectangle((15, 15, 75, 45), fill=(25, 25, 25))
    image.save(source)

    def fake_ocr(path, cfg, run_dir=None):
        return {
            "engine": "fixture", "status": "ok", "errors": [],
            "source": {"path": path, "w": 120, "h": 80}, "ms": 1,
            "lines": [{"id": "L0", "text": "SALE", "conf": .99,
                       "box": {"x": 15, "y": 15, "w": 60, "h": 30},
                       "quad": [[15, 15], [75, 15], [75, 45], [15, 45]], "words": []}],
        }

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {"device": "cpu", "runtime": {"require_active_models": True},
         "qwen": {"enabled": False}, "sam3": {"enabled": True, "checkpoint": str(tmp_path / "missing.pt")},
         "inpaint": {"mode": "opencv"}, "text_analysis": {"font_matching": {"enabled": False}},
         "figma": {"enabled": False}, "qa_ocr": False},
    )

    qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "runtime_report.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["runtime_ok"] is False
    assert "sam3-unavailable" in {item["rule"] for item in qa["hard_fails"]}
    assert report["violations"][0]["rule"] == "sam3-unavailable"


def test_qa_ok_is_false_when_edge_or_color_fidelity_fails_even_at_high_ssim(monkeypatch, tmp_path):
    """qa['ok'] must not be gated by multiscale SSIM alone. A run with badly wrong edges
    or colors but ssim >= 0.9 (very plausible — SSIM is dominated by luminance/structure)
    has to be rejected via hard_fails, not silently pass because only ssim was checked.
    """
    source = tmp_path / "input.png"
    image = Image.new("RGB", (120, 80), (244, 239, 228))
    ImageDraw.Draw(image).rectangle((15, 15, 75, 45), fill=(25, 25, 25))
    image.save(source)

    def fake_ocr(path, cfg, run_dir=None):
        return {"engine": "fixture", "source": {"path": path, "w": 120, "h": 80}, "ms": 1, "lines": []}

    def fake_compare(*args, **kwargs):
        # Mirrors what pixel_diff.compare now returns once quality_flags are merged into
        # hard_fails: high ssim, but a hard fail from a failed visual-quality gate.
        return {
            "ssim": 0.95, "visual_score": 0.95, "edge_f1": 0.2, "color_similarity": 0.3,
            "quality_flags": [{"rule": "edge-fidelity", "detail": "0.2 < 0.68"}],
            "text_recall": None,
            "hard_fails": [{"rule": "edge-fidelity", "detail": "0.2 < 0.68", "hard": True}],
            "structural": {"hard_fails": [{"rule": "edge-fidelity", "detail": "0.2 < 0.68", "hard": True}],
                           "editable_text_recall": None},
        }

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    monkeypatch.setattr(run_pipeline.pixel_diff, "compare", fake_compare)
    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu",
            "qwen": {"enabled": False},
            "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"},
            "text_analysis": {"font_matching": {"enabled": False}},
            "figma": {"enabled": False},
            "qa_ocr": False,
        },
    )

    assert result["ok"] is True  # the pipeline run itself completed
    qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    assert qa["ssim"] >= 0.9
    assert qa["ok"] is False, "high ssim alone should not be sufficient to pass qa"
    assert "edge-fidelity" in {item["rule"] for item in qa["hard_fails"]}


def test_render_qa_ocr_is_separate_from_canonical_source_ocr(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (120, 80), "white").save(source)
    calls = []

    def fake_ocr(path, cfg, run_dir=None):
        calls.append((path, run_dir))
        text = "SOURCE COPY" if os.path.basename(path) == "normalized.png" else "RENDER COPY"
        return {
            "engine": "fixture", "status": "ok", "errors": [],
            "source": {"path": path, "w": 120, "h": 80}, "ms": 1,
            "lines": [{"id": "L0", "text": text, "conf": .99,
                       "box": {"x": 5, "y": 5, "w": 80, "h": 20},
                       "quad": [[5, 5], [85, 5], [85, 25], [5, 25]], "words": []}],
        }

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    run_dir = tmp_path / "run"
    result = run_pipeline.run_one(
        str(source), str(run_dir),
        {
            "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
            "inpaint": {"mode": "opencv"},
            "text_analysis": {"font_matching": {"enabled": False}},
            "figma": {"enabled": False}, "qa_ocr": True,
        },
    )

    assert result["ok"] is True
    canonical = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    rendered = json.loads((run_dir / "render_ocr.json").read_text(encoding="utf-8"))
    assert canonical["source"]["path"] == str(run_dir / "normalized.png")
    assert canonical["lines"][0]["text"] == "SOURCE COPY"
    assert rendered["source"]["path"] == str(run_dir / "preview.png")
    assert rendered["lines"][0]["text"] == "RENDER COPY"
    assert rendered["provenance"] == {
        "kind": "render-qa",
        "render_path": str((run_dir / "preview.png").resolve()),
        "source_ocr_path": str((run_dir / "ocr.json").resolve()),
    }
    assert calls[-1][1] == "", "render QA OCR must disable run_ocr's ocr.json writer"


def test_resume_refuses_stale_artifacts_from_a_different_source(monkeypatch, tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (48, 32), "white").save(image_a)
    Image.new("RGB", (48, 32), "black").save(image_b)
    run_dir = tmp_path / "run"
    cfg = {
        "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
        "inpaint": {"mode": "opencv"}, "qa_ocr": False,
    }
    monkeypatch.setattr(
        run_pipeline.ocr, "run_ocr",
        lambda path, cfg, run_dir=None: {"engine": "fixture", "lines": []},
    )

    first = run_pipeline.run_one(str(image_a), str(run_dir), cfg)
    second = run_pipeline.run_one(str(image_b), str(run_dir), cfg, "qa")

    assert first["ok"]
    assert not second["ok"]
    assert "input image changed" in second["error"]


def test_resume_rebuilds_a_truncated_json_checkpoint(monkeypatch, tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 48), "white").save(source)
    run_dir = tmp_path / "run"
    cfg = {
        "device": "cpu", "qwen": {"enabled": False}, "sam3": {"enabled": False},
        "inpaint": {"mode": "opencv"}, "qa_ocr": False,
        "text_analysis": {"font_matching": {"enabled": False}},
    }
    calls = {"ocr": 0}

    def fake_ocr(path, cfg, run_dir=None):
        calls["ocr"] += 1
        return {"engine": "fixture", "status": "ok", "errors": [], "lines": []}

    monkeypatch.setattr(run_pipeline.ocr, "run_ocr", fake_ocr)
    assert run_pipeline.run_one(str(source), str(run_dir), cfg)["ok"]
    (run_dir / "ocr_raw.json").write_text('{"broken":', encoding="utf-8")

    resumed = run_pipeline.run_one(str(source), str(run_dir), cfg, "qa")

    assert resumed["ok"]
    assert json.loads((run_dir / "ocr_raw.json").read_text(encoding="utf-8"))["engine"] == "fixture"
    assert calls["ocr"] >= 2
