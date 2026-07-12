param(
  [int]$Port = $(if ($env:BRIDGE_PORT) { [int]$env:BRIDGE_PORT } else { 8790 }),
  [string]$Inbox = $(if ($env:FIGMA_INBOX) { $env:FIGMA_INBOX } else { "$HOME\figma-inbox" }),
  [switch]$FullSetup
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Ensure-Venv {
  if (Test-Path $Python) { return }
  Write-Host ""
  Write-Host "First run — setting up Python environment..."
  if ($FullSetup -and (Test-Path ".\setup_rtx.ps1")) {
    & powershell -ExecutionPolicy Bypass -File ".\setup_rtx.ps1" -SkipDoctor
    if (-not (Test-Path $Python)) { throw "Setup finished but .venv\Scripts\python.exe is still missing." }
    return
  }
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python is missing. Install Python 3.12, or run .\setup_rtx.ps1 once."
  }
  & py -3.12 -m venv .venv
  & $Python -m pip install --upgrade pip
  & $Python -m pip install -r requirements.txt
}

function Test-BridgeHealth([int]$TargetPort) {
  try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:$TargetPort/health" -UseBasicParsing -TimeoutSec 2
    return $response.StatusCode -eq 200
  } catch {
    return $false
  }
}

Ensure-Venv

# Sync repo before starting so the bridge always runs the latest code.
if (Test-Path (Join-Path $Root ".git")) {
  try {
    Write-Host "Syncing repo (git pull newrepo main)..."
    & git -C $Root fetch newrepo main 2>$null
    & git -C $Root pull newrepo main
    if (Test-Path $Python) {
      & $Python "$Root\scripts\stamp_plugin_build.py" --quiet
    }
    $head = (& git -C $Root rev-parse --short HEAD).Trim()
    Write-Host "Repo at $head"
  } catch {
    Write-Host "WARNING: git pull failed — continuing with current checkout: $_"
  }
}

& $Python "$Root\scripts\stamp_plugin_build.py" --quiet
& $Python -m src.bridge_bootstrap --config config.yaml --inbox $Inbox

if (Test-BridgeHealth $Port) {
  Write-Host ""
  Write-Host "================================================"
  Write-Host "  Bridge is already running"
  Write-Host "  http://localhost:$Port"
  Write-Host "================================================"
  Write-Host ""
  Write-Host "Code was synced above. Restart this script (or kill python.exe)"
  Write-Host "so the bridge loads the new checkout."
  Write-Host ""
  Read-Host "Press Enter to close"
  exit 0
}

Write-Host ""
Write-Host "================================================"
Write-Host "  Ad Decompiler Bridge"
Write-Host "  http://localhost:$Port"
Write-Host "================================================"
Write-Host "Inbox:  $Inbox"
Write-Host "Config: $Root\config.yaml"
Write-Host ""
Write-Host "1. Import figma-plugin\manifest.json in Figma (once)"
Write-Host "2. Open the plugin and upload an image"
Write-Host ""
Write-Host "Press Ctrl+C to stop the bridge."
Write-Host ""

& $Python -m src.figma_bridge --inbox $Inbox --port $Port --host 127.0.0.1 --config config.yaml --no-bootstrap
