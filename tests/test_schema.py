import json

import pytest

from src import schema


def test_dump_replaces_checkpoint_atomically(tmp_path):
    path = tmp_path / "stage.json"
    schema.dump({"version": 1}, str(path))
    schema.dump({"version": 2, "items": [1, 2]}, str(path))

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2, "items": [1, 2]}
    assert not list(tmp_path.glob(".stage.json.*.tmp"))


def test_dump_keeps_previous_checkpoint_when_serialization_fails(tmp_path):
    path = tmp_path / "stage.json"
    schema.dump({"version": 1}, str(path))

    with pytest.raises(TypeError):
        schema.dump({"bad": object()}, str(path))

    assert schema.load(str(path)) == {"version": 1}
    assert not list(tmp_path.glob(".stage.json.*.tmp"))
