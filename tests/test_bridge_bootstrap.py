from pathlib import Path

from src.bridge_bootstrap import prepare


def test_bridge_bootstrap_creates_config_and_enables_figma(tmp_path):
    example = tmp_path / "config.example.yaml"
    example.write_text(
        "device: cuda\nfigma:\n  enabled: false\n  mode: plugin\n",
        encoding="utf-8",
    )
    root = tmp_path
    (root / "config.example.yaml").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    inbox = tmp_path / "inbox"
    status = prepare(config_path=root / "config.yaml", inbox=str(inbox), root=str(root))
    assert status["created_config"] is True
    assert status["patched_figma"] is True
    assert (root / "config.yaml").exists()
    assert inbox.is_dir()
    text = (root / "config.yaml").read_text(encoding="utf-8")
    assert "enabled: true" in text


def test_bridge_bootstrap_resolves_config_relative_to_root_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "config.example.yaml").write_text("figma:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    status = prepare(config_path="config.yaml", inbox=str(root / "inbox"), root=str(root))
    assert (root / "config.yaml").exists()
    assert not (tmp_path / "config.yaml").exists()
    assert status["config_path"] == str(root / "config.yaml")
