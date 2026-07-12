param(
  [int]$Port = $(if ($env:BRIDGE_PORT) { [int]$env:BRIDGE_PORT } else { 8790 }),
  [string]$Inbox = $(if ($env:FIGMA_INBOX) { $env:FIGMA_INBOX } else { "$HOME\figma-inbox" }),
  [switch]$FullSetup
)

$ErrorActionPreference = "Stop"
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
& $Python "$Root\scripts\stamp_plugin_build.py" --quiet
& $Python -m src.bridge_bootstrap --config config.yaml --inbox $Inbox

if (Test-BridgeHealth $Port) {
  Write-Host ""
  Write-Host "================================================"
  Write-Host "  Bridge is already running"
  Write-Host "  http://localhost:$Port"
  Write-Host "================================================"
  Write-Host ""
  Write-Host "Open Figma Desktop and run the Ad Decompiler plugin."
  Write-Host "You can close this window — the bridge keeps running."
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
