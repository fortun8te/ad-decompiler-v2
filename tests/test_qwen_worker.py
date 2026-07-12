from pathlib import Path

import numpy as np
from PIL import Image

from src import qwen_worker


class _Schema:
    @staticmethod
    def dump(value, path):
        import json

        Path(path).write_text(json.dumps(value), encoding="utf-8")


def test_absent_qwen_config_is_offline_safe(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    called = []
    monkeypatch.setattr(qwen_worker, "_load_schema", lambda: _Schema)
    monkeypatch.setattr(qwen_worker, "_run_comfyui", lambda *args: called.append(args))

    assert qwen_worker.propose_layers(str(source), str(tmp_path), {}) == []
    assert not called
    assert "disabled" in (tmp_path / "qwen.note.txt").read_text(encoding="utf-8")


def test_successful_retry_clears_stale_failure_note(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(source)
    layer = {"id": "Q0", "png": "qwen_layers/Q0.png", "box": {"x": 0, "y": 0, "w": 20, "h": 20}}

    def fail(*args):
        qwen_worker._write_manifest(_Schema, [], str(tmp_path), "comfy failed")
        return []

    def recover(*args):
        return [layer]

    monkeypatch.setattr(qwen_worker, "_load_schema", lambda: _Schema)
    monkeypatch.setattr(qwen_worker, "_run_comfyui", fail)
    monkeypatch.setattr(qwen_worker, "_run_diffusers", recover)

    result = qwen_worker.propose_layers(
        str(source), str(tmp_path),
        {"qwen": {"enabled": True, "mode": "comfyui", "fallback_modes": ["direct-diffusers"]}},
    )

    assert result == [layer]
    assert not (tmp_path / "qwen.note.txt").exists()


def test_finalize_rejects_fully_transparent_fake_layers(tmp_path):
    transparent = tmp_path / "transparent.png"
    Image.fromarray(np.zeros((16, 16, 4), dtype=np.uint8), "RGBA").save(transparent)

    layers = qwen_worker._finalize([str(transparent)], str(tmp_path))

    assert layers == []
    assert not (tmp_path / "qwen_layers" / "Q0.png").exists()


# ── Flux Fill inpaint backend ─────────────────────────────────────────────────────────
def _mask_pair():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    return rgb, mask


def test_flux_inpaint_returns_none_when_comfyui_offline(monkeypatch):
    class _Dead:
        def get(self, *args, **kwargs):
            raise OSError("connection refused")

        def post(self, *args, **kwargs):
            raise OSError("connection refused")

    monkeypatch.setattr(qwen_worker, "_requests", lambda: _Dead())
    rgb, mask = _mask_pair()
    # A downed ComfyUI must degrade gracefully to None, never raise.
    assert qwen_worker.flux_inpaint(rgb, mask, {"inpaint": {"comfy": {}}}) is None


def test_flux_inpaint_returns_none_when_workflow_missing(monkeypatch):
    class _R:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class _OkProbe:
        def get(self, *args, **kwargs):
            return _R()

        def post(self, *args, **kwargs):
            return _R()

    monkeypatch.setattr(qwen_worker, "_requests", lambda: _OkProbe())
    rgb, mask = _mask_pair()
    cfg = {"inpaint": {"comfy": {"workflow": "workflows/__does_not_exist__.json"}}}
    assert qwen_worker.flux_inpaint(rgb, mask, cfg) is None


def test_flux_inpaint_full_cycle_returns_decoded_plate(monkeypatch):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (11, 22, 33)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class _Resp:
        def __init__(self, json_data=None, content=b""):
            self._json = json_data
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    class _Fake:
        def get(self, url, **kwargs):
            if url.endswith("/system_stats"):
                return _Resp(json_data={"ok": 1})
            if "/history/" in url:
                return _Resp(json_data={"pid123": {"outputs": {
                    "14": {"images": [{"filename": "flux_inpaint_00001_.png",
                                       "subfolder": "", "type": "output"}]}
                }}})
            if "/view" in url:
                return _Resp(content=png_bytes)
            return _Resp(json_data={})

        def post(self, url, **kwargs):
            if url.endswith("/upload/image"):
                return _Resp(json_data={"name": "flux_uploaded.png"})
            if url.endswith("/prompt"):
                return _Resp(json_data={"prompt_id": "pid123"})
            return _Resp(json_data={})

    monkeypatch.setattr(qwen_worker, "_requests", lambda: _Fake())
    rgb, mask = _mask_pair()
    cfg = {"inpaint": {"comfy": {
        "workflow": "workflows/flux_fill_inpaint_api.json", "timeout_s": 5,
    }}}
    out = qwen_worker.flux_inpaint(rgb, mask, cfg)
    assert out is not None
    assert out.shape == (8, 8, 3)
