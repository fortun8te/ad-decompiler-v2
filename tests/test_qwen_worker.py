from pathlib import Path

import numpy as np
from PIL import Image

from src import qwen_worker


class _Schema:
    @staticmethod
    def dump(value, path):
        import json

        Path(path).write_text(json.dumps(value), encoding="utf-8")


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
        {"qwen": {"mode": "comfyui", "fallback_modes": ["direct-diffusers"]}},
    )

    assert result == [layer]
    assert not (tmp_path / "qwen.note.txt").exists()


def test_finalize_rejects_fully_transparent_fake_layers(tmp_path):
    transparent = tmp_path / "transparent.png"
    Image.fromarray(np.zeros((16, 16, 4), dtype=np.uint8), "RGBA").save(transparent)

    layers = qwen_worker._finalize([str(transparent)], str(tmp_path))

    assert layers == []
    assert not (tmp_path / "qwen_layers" / "Q0.png").exists()

