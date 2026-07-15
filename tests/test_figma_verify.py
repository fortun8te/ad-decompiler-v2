"""Tests for the "verified in real Figma" QA gate (src/figma_verify.py).

All images are synthetic: a fake plugin 'export' is produced by perturbing the
preview (shifted text block, wrong-color rect) and the tests assert that the
verdict ladder and the drift-region localization behave."""
import json
import os
import threading
import time

from PIL import Image, ImageDraw

from src import figma_verify
from scripts import figma_verify as figma_verify_cli

CANVAS = (240, 180)
TEXT_BLOCK = (30, 118, 110, 152)   # fake text lines (x0, y0, x1, y1)
COLOR_RECT = (150, 30, 210, 70)    # the rect whose color the "plugin" gets wrong

CFG = {
    "qa": {"visual_pass_ssim": 0.60},
    "figma_verify": {
        "drift_ssim_min": 0.98,
        "drift_grid": 12,
        # top-N must cover both perturbations: the shifted text block owns the
        # ~20 worst cells, the recolored rect the next ~12.
        "drift_top_n": 24,
        "drift_region_ssim_min": 0.90,
    },
}


def _base_image():
    image = Image.new("RGB", CANVAS, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 80, 60), fill=(20, 60, 200))          # stable blue rect
    draw.rectangle(COLOR_RECT, fill=(220, 30, 30))                # red rect
    draw.rectangle((30, 118, 110, 130), fill=(10, 10, 10))        # "text" line 1
    draw.rectangle((30, 138, 110, 150), fill=(10, 10, 10))        # "text" line 2
    return image


def _make_run(tmp_path, with_preview=True, name="run"):
    run = tmp_path / name
    run.mkdir(exist_ok=True)
    image = _base_image()
    image.save(run / "normalized.png")
    image.save(run / "original.png")
    if with_preview:
        image.save(run / "preview.png")
    (run / "design.json").write_text(json.dumps(
        {"id": "test", "canvas": {"w": CANVAS[0], "h": CANVAS[1]}, "layers": []}
    ), encoding="utf-8")
    return run


def _write_export(run, image):
    path = run / "figma_export.png"
    image.save(path)
    # exports must be at least as fresh as design.json to count as evidence
    future = time.time() + 5
    os.utime(path, (future, future))
    return path


def _perturbed_export():
    """The 'plugin got it wrong' export: text block shifted, rect recolored."""
    image = _base_image()
    block = image.crop(TEXT_BLOCK)
    draw = ImageDraw.Draw(image)
    draw.rectangle(TEXT_BLOCK, fill=(255, 255, 255))       # erase original text
    image.paste(block, (TEXT_BLOCK[0] + 8, TEXT_BLOCK[1] + 6))  # shifted re-paste
    draw.rectangle(COLOR_RECT, fill=(30, 200, 30))         # red → green
    return image


def _intersects(bbox, area):
    x0, y0 = bbox["x"], bbox["y"]
    x1, y1 = x0 + bbox["w"], y0 + bbox["h"]
    return not (x1 <= area[0] or x0 >= area[2] or y1 <= area[1] or y0 >= area[3])


def test_missing_export_reports_not_exported(tmp_path):
    run = _make_run(tmp_path)
    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.status == "not-exported"
    assert result.verdict == "not-exported"
    report = json.loads((run / "figma_qa.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not-exported"
    assert report["schema_version"] == figma_verify.FIGMA_QA_SCHEMA_VERSION
    assert any(c["check"] == "export-present" and c["status"] == "fail"
               for c in report["checks"])


def test_pixel_perfect_export_is_verified(tmp_path):
    run = _make_run(tmp_path)
    _write_export(run, _base_image())
    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.status == "scored"
    assert result.verdict == "verified"
    assert result.fidelity["ssim"] > 0.99
    assert result.preview_drift["ssim"] > 0.99
    assert result.preview_drift["material"] is False
    report = json.loads((run / "figma_qa.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "verified"


def test_drift_is_material_and_localized_to_perturbed_regions(tmp_path):
    run = _make_run(tmp_path)
    _write_export(run, _perturbed_export())
    # This test is about drift *localization*: lower the fidelity bar so the
    # perturbed export still passes fidelity (multiscale SSIM ≈ 0.57) and the
    # verdict is driven by the preview-drift detector alone.
    cfg = json.loads(json.dumps(CFG))
    cfg["qa"]["visual_pass_ssim"] = 0.45
    result = figma_verify.verify(str(run), cfg=cfg, allow_ocr=False)
    # fidelity clears the (test-lowered) bar, but the simulation disagreed with
    # "Figma" → drift detector fires and the verdict degrades.
    assert result.verdict == "degraded"
    drift = result.preview_drift
    assert drift["material"] is True
    assert drift["ssim"] < 0.98
    regions = drift["regions"]
    assert regions, "worst-region breakdown must not be empty"
    assert any(_intersects(r["bbox"], TEXT_BLOCK) for r in regions), \
        "shifted text block must appear in the worst drift regions"
    assert any(_intersects(r["bbox"], COLOR_RECT) for r in regions), \
        "recolored rect must appear in the worst drift regions"
    heatmap = drift["heatmap_png"]
    assert os.path.exists(heatmap)
    assert Image.open(heatmap).size == CANVAS


def test_broken_export_fails_fidelity(tmp_path):
    run = _make_run(tmp_path)
    _write_export(run, Image.new("RGB", CANVAS, (255, 255, 255)))  # blank frame
    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.verdict == "failed"
    assert result.fidelity["ssim"] < CFG["qa"]["visual_pass_ssim"]
    assert any(c["check"] == "fidelity-ssim" and c["status"] == "fail"
               for c in result.checks)


def test_2x_export_is_scale_normalized(tmp_path):
    run = _make_run(tmp_path)
    big = _base_image().resize((CANVAS[0] * 2, CANVAS[1] * 2), Image.Resampling.NEAREST)
    _write_export(run, big)
    cfg = json.loads(json.dumps(CFG))
    cfg["figma_verify"]["drift_ssim_min"] = 0.85  # resampling softness is not drift
    result = figma_verify.verify(str(run), cfg=cfg, allow_ocr=False)
    assert result.export["scale"]["detected"] == 2
    assert result.export["size"] == [CANVAS[0] * 2, CANVAS[1] * 2]
    assert result.verdict == "verified"
    assert result.fidelity["ssim"] > 0.85


def test_stale_export_is_not_evidence(tmp_path):
    run = _make_run(tmp_path)
    export = _write_export(run, _base_image())
    past = time.time() - 3600
    os.utime(export, (past, past))  # design.json is now newer than the export
    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.status == "stale-export"
    assert result.verdict == "not-exported"

    scored = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False, allow_stale=True)
    assert scored.status == "stale-export"
    assert scored.verdict == "degraded"  # scored, but capped: stale ≠ current design
    assert any("stale" in w for w in scored.warnings)


def test_text_recall_reuses_matching_render_ocr(tmp_path):
    run = _make_run(tmp_path)
    export = _write_export(run, _base_image())
    (run / "ocr.json").write_text(json.dumps({
        "lines": [{"text": "Hello World", "conf": 0.9},
                  {"text": "Buy Now", "conf": 0.9}],
    }), encoding="utf-8")
    render_ocr = run / "render_ocr.json"
    render_ocr.write_text(json.dumps({
        "lines": [{"text": "Hello World", "conf": 0.9}],
        "provenance": {"kind": "render-qa", "render_path": str(export)},
    }), encoding="utf-8")
    future = time.time() + 10
    os.utime(render_ocr, (future, future))  # OCR at least as fresh as the export

    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.fidelity["text_recall"] == 0.5   # 1 of 2 source lines recovered
    assert result.verdict == "failed"              # text went missing in real Figma
    assert any(c["check"] == "fidelity-text-recall" and c["status"] == "fail"
               for c in result.checks)
    assert result.fidelity["text_evidence"]["reused"] is True


def test_missing_preview_degrades_instead_of_lying(tmp_path):
    run = _make_run(tmp_path, with_preview=False)
    _write_export(run, _base_image())
    result = figma_verify.verify(str(run), cfg=CFG, allow_ocr=False)
    assert result.verdict == "degraded"
    assert any(c["check"] == "preview-drift-ssim" and c["status"] == "skip"
               for c in result.checks)


def test_cli_all_prints_verdict_table_and_exit_codes(tmp_path, capsys):
    parent = tmp_path / "suite"
    parent.mkdir()
    good = _make_run(parent, name="good")
    _write_export(good, _base_image())
    _make_run(parent, name="empty")

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(CFG), encoding="utf-8")

    code = figma_verify_cli.main(["--all", str(parent), "--no-ocr",
                                  "--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert code == 0  # nothing failed (one verified, one not-exported)
    assert "verified" in out and "not-exported" in out
    assert "2 run(s)" in out

    strict = figma_verify_cli.main(["--all", str(parent), "--no-ocr",
                                    "--config", str(cfg_path), "--strict"])
    assert strict == 2  # strict sign-off requires every run verified


def test_cli_watch_verifies_once_export_lands(tmp_path, capsys):
    run = _make_run(tmp_path)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(CFG), encoding="utf-8")

    def late_export():
        time.sleep(0.4)
        _write_export(run, _base_image())

    thread = threading.Thread(target=late_export)
    thread.start()
    try:
        code = figma_verify_cli.main([str(run), "--watch", "--timeout", "15",
                                      "--poll", "0.1", "--no-ocr",
                                      "--config", str(cfg_path)])
    finally:
        thread.join()
    out = capsys.readouterr().out
    assert code == 0
    assert "verified" in out
    assert (run / "figma_qa.json").exists()


def test_cli_watch_times_out_cleanly(tmp_path, capsys):
    run = _make_run(tmp_path)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(CFG), encoding="utf-8")
    code = figma_verify_cli.main([str(run), "--watch", "--timeout", "0.3",
                                  "--poll", "0.1", "--no-ocr",
                                  "--config", str(cfg_path)])
    assert code == 3
    assert "not-exported" in capsys.readouterr().out
