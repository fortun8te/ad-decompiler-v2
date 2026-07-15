from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_one_click_launcher_runs_complete_setup_and_tracks_it():
    launcher = _read("start_bridge.ps1")
    setup = _read("setup_rtx.ps1")

    assert "setup_rtx.ps1\" -SkipDoctor" in launcher
    assert ".rtx-setup-v4" in launcher
    assert "[switch]$SkipDoctor" in setup
    assert ".rtx-setup-v4" in setup


def test_setup_exposes_explicit_selected_model_smoke_without_claiming_powerpaint_install():
    setup = _read("setup_rtx.ps1")

    assert "[switch]$DeepDoctor" in setup
    assert '"--deep", "--deep-output", "runs\\runtime-smoke"' in setup
    assert "does not install or claim a PowerPaint model" in setup


def test_flux_setup_wrapper_default_matches_checked_in_flux_workflow():
    wrapper = _read("scripts/setup_flux_inpaint.ps1")

    assert '[string]$Quant = "Q6_K"' in wrapper
    assert "checked-in workflow/doctor default" in wrapper


def test_acceptance_benchmark_requires_cached_runtime_evidence_for_default_active_models():
    launcher = _read("start_rtx.ps1")

    assert "[string[]]$Ids" in launcher
    assert '"--ids", "$id"' in launcher
    assert "require_active_models" in launcher
    assert "inpaint_strict_acceptance" in launcher
    assert "rtx_self_test.py" in launcher
    assert "acceptance runtime self-test failed" in launcher
    assert "[switch]$RequireFigma" in launcher
    assert '"--require-figma-export", "--figma-wait-s"' in launcher
    assert "Figma acceptance needs the bridge" in launcher


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
    assert "scripts\\sync_update.py" in launcher
    assert "--update --notify --json" in launcher
    assert "Remove-Item -Force $SetupStamp" in launcher
    assert "Code updated — refreshing the local RTX environment" in launcher
    assert 'git -C $Root pull' not in launcher
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


def test_platform_sync_wrappers_share_the_safe_python_updater():
    windows = _read("scripts/windows_sync.ps1")
    macos = _read("scripts/mac_sync_update.sh")

    assert "scripts\\sync_update.py" in windows
    assert '& git pull' not in windows
    assert "sync_update.py" in macos
    assert "--update --notify" in macos


def test_platform_sync_installers_schedule_login_and_hourly_without_destructive_git():
    windows = _read("scripts/install_windows_sync_task.ps1")
    macos = _read("scripts/install_macos_sync_launchd.sh")
    plist = _read("scripts/com.fortun8te.addecompiler.sync.plist.template")

    assert "AdDecompilerSyncAtLogon" in windows
    assert "/SC ONLOGON" in windows
    assert "AdDecompilerSyncHourly" in windows
    assert "/SC HOURLY /MO 1" in windows
    assert "windows_sync.ps1" in windows
    assert "git reset" not in windows and "git pull" not in windows
    assert "launchctl bootstrap" in macos
    assert "launchctl kickstart" in macos
    assert "mac_sync_update.sh" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert "<key>StartInterval</key>" in plist
    assert "<integer>3600</integer>" in plist
