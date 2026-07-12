import json
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stamp_plugin_build", ROOT / "scripts" / "stamp_plugin_build.py")
stamp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stamp)


def test_stamp_plugin_build_writes_build_info_and_manifest(tmp_path, monkeypatch):
    plugin = tmp_path / "figma-plugin"
    plugin.mkdir()
    (plugin / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    (plugin / "code.js").write_text(
        "// header\n// It accepts the legacy flat design.json contract and scene-graph v2 documents.\n\nfigma.showUI();\n",
        encoding="utf-8",
    )
    (plugin / "ui.html").write_text(
        "<script>\n    const $ = function (id) { return document.getElementById(id); };\n</script>\n",
        encoding="utf-8",
    )
    (plugin / "manifest.json").write_text('{"name":"Test"}\n', encoding="utf-8")

    monkeypatch.setattr(stamp, "ROOT", tmp_path)
    monkeypatch.setattr(stamp, "PLUGIN", plugin)
    monkeypatch.setattr(stamp, "VERSION_FILE", plugin / "VERSION")
    monkeypatch.setattr(stamp, "COUNTER_FILE", plugin / ".build-counter")
    monkeypatch.setattr(stamp, "BUILD_INFO", plugin / "build-info.json")
    monkeypatch.setattr(stamp, "CODE_JS", plugin / "code.js")
    monkeypatch.setattr(stamp, "UI_HTML", plugin / "ui.html")
    monkeypatch.setattr(stamp, "MANIFEST", plugin / "manifest.json")
    monkeypatch.setattr(stamp, "_git_info", lambda: {"build": 99, "commit": "deadbeef", "dirty": False, "source": "git"})

    info = stamp.stamp_files()
    assert info["version"] == "1.2.3"
    assert info["build"] == 99
    assert info["label"].startswith("v1.2.3+b99.deadbeef")
    assert "const PLUGIN_BUILD =" in (plugin / "code.js").read_text(encoding="utf-8")
    manifest = json.loads((plugin / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build"] == 99


def test_stamp_plugin_build_cli_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stamp_plugin_build.py"), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["version"]
    assert isinstance(info["build"], int)
