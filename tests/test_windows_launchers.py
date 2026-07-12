from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_one_click_launcher_runs_complete_setup_and_tracks_it():
    launcher = _read("start_bridge.ps1")
    setup = _read("setup_rtx.ps1")

    assert "setup_rtx.ps1\" -SkipDoctor" in launcher
    assert ".rtx-setup-v3" in launcher
    assert "[switch]$SkipDoctor" in setup
    assert ".rtx-setup-v3" in setup


def test_optional_ocr_backends_cannot_break_core_gpu_install():
    requirements = _read("requirements-gpu.txt")
    setup = _read("setup_rtx.ps1")

    assert "paddlepaddle-gpu>=" not in requirements
    assert "surya-ocr>=" not in requirements
    assert 'Invoke-OptionalPip "PaddleOCR fallback"' in setup
    assert 'Invoke-OptionalPip "Surya OCR fallback"' in setup


def test_launcher_does_not_update_code_unless_asked_and_syncs_port():
    launcher = _read("start_bridge.ps1")

    assert "if ($Update" in launcher
    assert "--inbox $Inbox --port $Port" in launcher
    assert "Test-PortOpen" in launcher
    assert 'service -eq "ad-decompiler-bridge"' in launcher


def test_launcher_exposes_cached_runtime_self_test_and_safe_remote_mode():
    launcher = _read("start_bridge.ps1")

    assert "[switch]$SelfTest" in launcher
    assert "[switch]$ForceSelfTest" in launcher
    assert "rtx_self_test.py" in launcher
    assert "--status-json" in launcher
    assert "tailscale ip -4" in launcher
    assert "--host $BridgeHost" in launcher
