"""CPU tests for the gradient-tolerant smear detector (025 'Blocks everything' row).

The flat smear detector gates on the ring around the glyphs being locally SMOOTH
(median high-freq energy <= smear_max_plate_hf) and used to fail closed on anything
else.  A dimmed LaMa smear over a soft card gradient / vignette / panel wash therefore
shipped baked source ink UNDER the emitted native TEXT — the ghost double QA saw on
025's 'Blocks everything' row — while three sibling rows on smooth plates were cleaned.

The gradient-tolerant variant reuses the per-pixel fill estimator
(``_gradient_plate_fill``) as the background model and scores surviving glyph contrast
against it.  Genuine photographic texture still fails closed: when the smooth model
cannot even explain the ring itself, the estimate is noise and a plate fill would paint
a slab over a real photo.

All CPU-only; inpaint backends are monkeypatched so behaviour is deterministic.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import reconstruct  # noqa: E402

W, H = 240, 160
BOX = {"x": 60, "y": 70, "w": 110, "h": 18}  # glyph bar, same shape as the ghost tests
INK = (30, 30, 30)


def _gradient_bg(seed=11, noise=14, texture=False):
    """A soft panel gradient (150→210 luma across the row) + mild grain.

    noise=±14 pushes the ring's high-freq energy past smear_max_plate_hf (3.0) while
    staying explainable by a smooth per-pixel fit — 025's card gradient.  texture=True
    (±42) is genuine photo grain the smooth model must refuse to own.
    """
    rng = np.random.default_rng(seed)
    xgrad = np.linspace(150, 210, W, dtype=np.float32)[None, :, None]
    ygrad = np.linspace(-8, 8, H, dtype=np.float32)[:, None, None]
    bg = np.repeat(xgrad + ygrad, 3, axis=2)
    amp = 42 if texture else noise
    bg = bg + rng.integers(-amp, amp + 1, size=bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _source_with_bar(path, bg, box=BOX, ink=INK):
    img = bg.copy()
    img[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]] = ink
    Image.fromarray(img).save(path)
    return img


def _text_candidate(**overrides):
    candidate = {
        "id": "c_B0", "target": "text", "text": "Blocks everything", "z": 4,
        "box": dict(BOX), "visible_box": dict(BOX),
        "style": {"fontSize": 16, "fontFamily": "Arial", "color": "#1e1e1e"},
        "meta": {"source": "ocr", "role": "headline", "line_ids": ["L0"]},
    }
    candidate.update(overrides)
    return candidate


def _dim_glyphs_keep_gradient(monkeypatch, bg, keep=0.55):
    """Model 025's LaMa smear over a gradient: the halo comes back clean (gradient +
    grain intact) but the glyph pixels only DIM toward the local backdrop — they match
    neither 'still source' nor 'still ink', and the ring is not smooth, so the flat
    smear detector used to wave the row through.
    """
    def fake_once(image_path, mask, output_path, cfg=None):
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        m = np.asarray(reconstruct.inpaint.solidify_mask(mask)) > 0
        dark = img.mean(axis=2) < 100  # glyph pixels inside the removal mask
        img[m & ~dark] = bg[m & ~dark]  # halo: honestly cleaned gradient
        dim = np.clip(keep * bg.astype(np.float32)
                      + (1.0 - keep) * np.array(INK, dtype=np.float32), 0, 255)
        img[m & dark] = dim[m & dark].astype(np.uint8)  # glyphs: dimmed, not removed
        Image.fromarray(img).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-gradient-smear",
                "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_once)


def _run(tmp_path, archetype="comparison_grid", cfg_extra=None):
    cfg = {"inpaint": {"mode": "opencv"}, "scene": {"archetype": archetype}}
    if cfg_extra:
        cfg.update(cfg_extra)
    return reconstruct.reconstruct(
        str(tmp_path / "source.png"), {"lines": []}, [_text_candidate()],
        str(tmp_path), cfg,
    )


def test_gradient_plate_smear_is_flagged_cleaned_and_gradient_preserved(tmp_path, monkeypatch):
    """025's row: a dimmed smear over a soft gradient must be flagged by the GRADIENT
    variant, cleaned through the same solid-fill path as a flat smear, and the fill must
    follow the gradient — not flatten it into a highlighter bar."""
    bg = _gradient_bg()
    source = tmp_path / "source.png"
    _source_with_bar(source, bg)
    _dim_glyphs_keep_gradient(monkeypatch, bg)

    result = _run(tmp_path)
    residual = result["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"], "a dimmed smear over a gradient must not ship doubled"
    flag = residual["flagged"][0]
    assert flag["id"] == "c_B0"
    assert flag["kind"] == "smear", "must be attributed to the smear detector"
    assert flag["smear_variant"] == "gradient", "the flat path cannot see a gradient ring"
    assert flag["smear_ratio"] >= 0.14
    # Same remedy as a flat smear: trial-paint, re-detect, commit, expand union/ledger.
    assert flag["resolved"] is True
    assert flag["resolved_by"] == "gradient-plate-fill"
    assert "c_B0" in (residual.get("solid_filled_ids") or [])

    # The fill rides the gradient: the row's cleaned pixels still drift with the panel
    # (left vs right end of the bar differ) instead of collapsing to one flat median.
    plate = np.asarray(Image.open(tmp_path / "background_clean.png").convert("RGB"),
                       dtype=np.float32)
    bar = plate[BOX["y"] + 4:BOX["y"] + BOX["h"] - 4, BOX["x"] + 4:BOX["x"] + BOX["w"] - 4]
    span = float(bar[:, -6:].mean() - bar[:, :6].mean())
    assert span > 15.0, "the gradient must survive the fill (not be flattened)"
    truth = bg[BOX["y"] + 4:BOX["y"] + BOX["h"] - 4, BOX["x"] + 4:BOX["x"] + BOX["w"] - 4]
    assert float(np.abs(bar - truth.astype(np.float32)).mean()) < 14.0, \
        "the fill must track the local plate, not invent a colour"

    # The committed fill expanded the removal union (plate-integrity contract).
    union = np.asarray(Image.open(tmp_path / "removal_mask.png").convert("L"))
    assert int(np.count_nonzero(
        union[BOX["y"]:BOX["y"] + BOX["h"], BOX["x"]:BOX["x"] + BOX["w"]])) > 1000


def test_flat_plate_smear_still_takes_the_flat_path(tmp_path, monkeypatch):
    """No behaviour change on a smooth plate: same verdict, same remedy as before."""
    bg = np.full((H, W, 3), (238, 232, 220), dtype=np.uint8)
    source = tmp_path / "source.png"
    _source_with_bar(source, bg)
    _dim_glyphs_keep_gradient(monkeypatch, bg, keep=0.55)

    residual = _run(tmp_path)["stats"]["text_residual"]
    assert residual["flagged"], "the flat-path smear verdict must be unchanged"
    flag = residual["flagged"][0]
    assert flag["kind"] == "smear"
    assert flag["smear_variant"] == "flat"
    assert flag["smear_ratio"] >= 0.18
    assert flag["resolved"] is True
    assert "c_B0" in (residual.get("solid_filled_ids") or [])


def test_photographic_texture_under_kept_raster_is_not_flagged(tmp_path, monkeypatch):
    """A genuinely clean plate over real photo grain must NOT be flagged: the smooth
    model cannot explain the ring, so the gradient variant abstains and nothing is
    repainted over the texture (a false fill there is worse than a miss)."""
    bg = _gradient_bg(texture=True)
    source = tmp_path / "source.png"
    _source_with_bar(source, bg)

    def fake_clean(image_path, mask, output_path, cfg=None):
        # An honest inpaint: the whole mask comes back as clean textured backdrop.
        img = np.asarray(Image.open(image_path).convert("RGB")).copy()
        m = np.asarray(reconstruct.inpaint.solidify_mask(mask)) > 0
        img[m] = bg[m]
        Image.fromarray(img).save(output_path)
        return {"ok": True, "path": output_path, "backend": "fake-clean",
                "backend_class": "active"}

    monkeypatch.setattr(reconstruct.inpaint, "inpaint_once", fake_clean)

    residual = _run(tmp_path)["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] == [], \
        "genuine photographic texture must not trip the smear detector"


def test_textured_smear_still_fails_closed(tmp_path, monkeypatch):
    """Even with a real smear present, fine texture stays out of scope: the ring-fit
    gate abstains and the row is left to the absolute tests (same contract as the
    photographic test in test_ghost_and_overerase.py)."""
    bg = _gradient_bg(texture=True)
    source = tmp_path / "source.png"
    _source_with_bar(source, bg)
    _dim_glyphs_keep_gradient(monkeypatch, bg)

    residual = _run(tmp_path)["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert [f for f in residual["flagged"] if f.get("kind") == "smear"] == []


def test_few_strong_pixels_abstains(tmp_path, monkeypatch):
    """A 4x4 glyph sliver has fewer strong pixels than the abstention floor: the
    variant must return None (as the flat path does today), not flag noise.

    The sliver carries a single ALNUM glyph with confident OCR meta: punctuation-only
    text (".") never reaches the audit at all — _flatten_photo_scene drops it as
    invalid-photo-scene-ocr debris before any mask exists, which is intentional and
    NOT the behaviour under test here."""
    bg = _gradient_bg()
    tiny = {"x": 100, "y": 76, "w": 4, "h": 4}
    source = tmp_path / "source.png"
    _source_with_bar(source, bg, box=tiny)
    _dim_glyphs_keep_gradient(monkeypatch, bg, keep=0.5)

    residual = reconstruct.reconstruct(
        str(source), {"lines": []},
        [_text_candidate(box=dict(tiny), visible_box=dict(tiny), text="i",
                         meta={"source": "ocr", "role": "headline",
                               "line_ids": ["L0"], "confidence": 0.9})],
        str(tmp_path),
        {"inpaint": {"mode": "opencv"}, "scene": {"archetype": "comparison_grid"}},
    )["stats"]["text_residual"]
    assert residual["checked"] == 1
    assert residual["flagged"] == [], \
        "too few strong glyph pixels must abstain, not guess"
