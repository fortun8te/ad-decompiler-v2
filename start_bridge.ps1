param(
  [int]$Port = $(if ($env:BRIDGE_PORT) { [int]$env:BRIDGE_PORT } else { 8790 }),
  [string]$Inbox = $(if ($env:FIGMA_INBOX) { $env:FIGMA_INBOX } else { "$HOME\figma-inbox" }),
  [switch]$Update,
  [switch]$SkipSetup,
  [switch]$SelfTest,
  [switch]$ForceSelfTest,
  [switch]$Remote
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$SetupStamp = Join-Path $Root ".venv\.rtx-setup-v4"
$BridgeHost = "127.0.0.1"

function Ensure-ToolOnPath([string]$Tool, [string[]]$KnownDirs) {
  if (Get-Command $Tool -ErrorAction SilentlyContinue) { return }
  foreach ($dir in $KnownDirs) {
    if ($dir -and (Test-Path (Join-Path $dir "$Tool.exe"))) {
      $env:PATH = "$dir;$env:PATH"
      return
    }
  }
}

# Tesseract (winget UB-Mannheim build) and the LM Studio `lms` CLI are frequently not on PATH
# until the user logs out and back in. Prepend their known install dirs to this process's PATH
# so doctor.py finds tesseract and the VLM-eviction feature (runtime.vram.evict_vlm_for_inpaint)
# can shell out to `lms` without a shell restart.
Ensure-ToolOnPath "tesseract" @("$env:ProgramFiles\Tesseract-OCR", "${env:ProgramFiles(x86)}\Tesseract-OCR")
Ensure-ToolOnPath "lms" @("$env:USERPROFILE\.lmstudio\bin")

function Ensure-Venv {
  if ((Test-Path $Python) -and (Test-Path $SetupStamp)) { return }
  Write-Host ""
  if ($SkipSetup) {
    throw "The app is not installed yet. Run setup_rtx.ps1, or start again without -SkipSetup."
  }
  Write-Host "First run (or setup upgrade) — installing the RTX app. This can take a while..."
  & powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_rtx.ps1" -SkipDoctor
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python) -or -not (Test-Path $SetupStamp)) {
    throw "First-run setup did not finish. See the message above, then run Start Bridge.bat again."
  }
}

function Get-BridgeHealth([string]$TargetHost, [int]$TargetPort) {
  try {
    $response = Invoke-RestMethod -Uri "http://${TargetHost}:$TargetPort/health" -TimeoutSec 4
    if ($response.ok -and $response.service -eq "ad-decompiler-bridge") { return $response }
  } catch {
  }
  return $null
}

function Test-PortOpen([string]$TargetHost, [int]$TargetPort) {
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $wait = $client.BeginConnect($TargetHost, $TargetPort, $null, $null)
    return $wait.AsyncWaitHandle.WaitOne(350) -and $client.Connected
  } catch { return $false } finally { $client.Close() }
}

Ensure-Venv

if ($Remote) {
  if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
    throw "Tailscale is not installed. Install and sign in to Tailscale, then try -Remote again."
  }
  $BridgeHost = (& tailscale ip -4 | Select-Object -First 1).Trim()
  if (-not $BridgeHost) { throw "Tailscale is not connected, so remote mode cannot start safely." }
  Write-Host "Remote mode: binding only to the Tailscale address $BridgeHost."
}

# Updating is explicit. The shared updater refuses dirty/diverged worktrees and only
# fast-forwards a clean checkout, then reports its decision in this terminal.
if ($Update -and (Test-Path (Join-Path $Root ".git"))) {
  Write-Host "Checking for a safe update..."
  $updateText = (& $Python "$Root\scripts\sync_update.py" --repo $Root --update --notify --json | Out-String)
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: update check failed — continuing with the current checkout."
  } else {
    try {
      $updateResult = $updateText | ConvertFrom-Json
      if ($updateResult.action -eq "updated") {
        # Setup is versioned by a stamp. A pulled requirements/model-adapter change must
        # invalidate that old stamp before the bridge starts, or the new code can run in
        # a stale environment despite a successful fast-forward.
        Remove-Item -Force $SetupStamp -ErrorAction SilentlyContinue
        Write-Host "Code updated — refreshing the local RTX environment before startup..."
        Ensure-Venv
      }
    } catch {
      Write-Host "WARNING: update status could not be read — continuing with the current environment."
    }
  }
}

& $Python "$Root\scripts\stamp_plugin_build.py" --quiet
& $Python -m src.bridge_bootstrap --config config.yaml --inbox $Inbox --port $Port

$existing = Get-BridgeHealth $BridgeHost $Port
if ($existing -and -not ($SelfTest -or $ForceSelfTest)) {
  Write-Host ""
  Write-Host "================================================"
  Write-Host "  Bridge is already running"
  Write-Host "  http://${BridgeHost}:$Port"
  Write-Host "================================================"
  Write-Host ""
  Write-Host "Nothing else to do — open the plugin in Figma."
  Write-Host ""
  Read-Host "Press Enter to close"
  exit 0
}

if (-not $existing -and (Test-PortOpen $BridgeHost $Port)) {
  throw "Port $Port is already used by another app. Close that app, then run Start Bridge.bat again."
}

# Show setup gaps, but keep the bridge available so the Figma plugin can display
# the same blockers and the user can fix them without losing the launcher.
$doctorText = (& $Python doctor.py --config config.yaml --json 2>$null | Out-String)
$doctor = $null
try { $doctor = $doctorText | ConvertFrom-Json } catch { }

if ($SelfTest -or $ForceSelfTest) {
  Write-Host ""
  Write-Host "Running the real RTX model self-test (evidence is cached)..."
  $selfTestArgs = @("rtx_self_test.py", "--config", "config.yaml")
  if ($ForceSelfTest) { $selfTestArgs += "--force" }
  & $Python @selfTestArgs
  if ($LASTEXITCODE -ne 0) {
    throw "The runtime self-test failed. Open runs\rtx-self-test\latest.json for exact evidence."
  }
} elseif ($doctor -and $doctor.ok) {
  $selfTestText = (& $Python rtx_self_test.py --config config.yaml --status-json | Out-String)
  try {
    $selfTestStatus = $selfTestText | ConvertFrom-Json
    if (-not $selfTestStatus.valid) {
      Write-Host ""
      Write-Host "Dependencies are ready, but real model execution is not yet proven."
      Write-Host "Run once: Start Bridge.bat -SelfTest"
    }
  } catch { }
}

if ($existing) {
  Write-Host ""
  Write-Host "Self-test finished. The existing bridge is still running at http://${BridgeHost}:$Port."
  Read-Host "Press Enter to close"
  exit 0
}

Write-Host ""
Write-Host "================================================"
Write-Host "  Ad Decompiler Bridge"
Write-Host "  http://${BridgeHost}:$Port"
Write-Host "================================================"
Write-Host "Inbox:  $Inbox"
Write-Host "Config: $Root\config.yaml"
Write-Host ""
Write-Host "1. Import figma-plugin\manifest.json in Figma (once)"
Write-Host "2. Open the plugin and upload an image"
if ($doctor -and -not $doctor.ok) {
  Write-Host ""
  Write-Host "Processing is not ready yet:"
  foreach ($item in $doctor.blockers) {
    Write-Host " - $($item.name): $($item.detail)"
    if ($item.fix) { Write-Host "   Fix: $($item.fix)" }
  }
  Write-Host "Run .\.venv\Scripts\python.exe doctor.py for the full fix list."
}
Write-Host ""
Write-Host "Press Ctrl+C to stop the bridge."
Write-Host ""

& $Python -m src.figma_bridge --inbox $Inbox --port $Port --host $BridgeHost --config config.yaml --no-bootstrap
