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


def test_bridge_bootstrap_reports_cuda_cudnn_warnings(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "config.yaml").write_text("device: cuda\nfigma:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.bridge_bootstrap._cuda_cudnn_warnings",
        lambda cfg: ["CUDA is available but cuDNN is missing — PaddleOCR GPU on Windows often fails"],
    )
    status = prepare(config_path=root / "config.yaml", inbox=str(root / "inbox"), root=str(root))
    assert status["gpu_warnings"]
    assert "cuDNN" in status["gpu_warnings"][0]


def test_bridge_bootstrap_keeps_launcher_inbox_and_port_in_sync(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    config = root / "config.yaml"
    config.write_text(
        "figma:\n  enabled: true\n  inbox: C:/old-inbox\n  bridge_port: 9999\n",
        encoding="utf-8",
    )
    inbox = tmp_path / "current-inbox"

    status = prepare(config_path=config, inbox=inbox, port=8791, root=root)

    assert status["patched_figma"] is True
    assert status["port"] == 8791
    text = config.read_text(encoding="utf-8")
    assert str(inbox).replace("\\", "/") in text
    assert "bridge_port: 8791" in text
    assert "C:/old-inbox" not in text
