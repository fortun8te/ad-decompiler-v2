param(
  [string]$InputDir,
  [string]$Output = "runs\benchmark",
  [string[]]$Ids,
  [switch]$RequireFigma,
  [int]$FigmaWaitS = 120,
  [switch]$NoBridge,
  [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  Write-Host "The environment is not installed yet. Running setup..."
  & powershell -ExecutionPolicy Bypass -File ".\setup_rtx.ps1"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
    throw "Setup did not finish. Fix the messages above, then run this launcher again."
  }
}

if (-not $SkipDoctor) {
  & $Python doctor.py --config config.yaml
  if ($LASTEXITCODE -ne 0) { throw "The machine is not ready. Fix the doctor output first." }

  # An acceptance benchmark must have actual model-execution evidence, not merely static
  # dependencies. This covers the default active OCR/SAM/Gemma/Big-LaMa route as well as
  # strict Flux/PowerPaint. rtx_self_test is cached, so normal restarts stay cheap.
  $doctorJsonText = (& $Python doctor.py --config config.yaml --json | Out-String)
  try {
    $doctorReport = $doctorJsonText | ConvertFrom-Json
    $needsRuntimeEvidence = $doctorReport.policy.require_active_models -or $doctorReport.policy.inpaint_strict_acceptance
    if ($needsRuntimeEvidence) {
      Write-Host "Acceptance benchmark: checking cached real-model runtime evidence..."
      & $Python rtx_self_test.py --config config.yaml
      if ($LASTEXITCODE -ne 0) {
        throw "The acceptance runtime self-test failed. Fix its evidence before running a benchmark."
      }
    }
  } catch {
    if ($_.Exception.Message -like "*acceptance runtime self-test failed*") { throw }
    throw "Could not read doctor policy for acceptance runtime evidence: $_"
  }
}

$bridgeProcess = $null
$bridgeWasAlreadyRunning = $false
if (-not $NoBridge) {
  & $Python "$((Get-Location).Path)\scripts\stamp_plugin_build.py" --quiet
  & $Python -m src.bridge_bootstrap --config config.yaml --inbox "$HOME\figma-inbox" | Out-Null
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8790/health" -TimeoutSec 3
    $bridgeWasAlreadyRunning = $health.ok -and $health.service -eq "ad-decompiler-bridge"
  } catch { }
  if ($bridgeWasAlreadyRunning) {
    Write-Host "Using the bridge that is already running on http://localhost:8790."
  } else {
    Write-Host "Starting the local Figma bridge on http://localhost:8790..."
    $bridgeProcess = Start-Process -FilePath $Python -ArgumentList @(
      "-m", "src.figma_bridge", "--inbox", "$HOME\figma-inbox", "--port", "8790", "--config", "config.yaml", "--no-bootstrap"
    ) -PassThru
    Start-Sleep -Seconds 1
    if ($bridgeProcess.HasExited) {
      throw "The bridge could not start. Port 8790 may be used by another app; close it or double-click Start Bridge.bat for a clearer check."
    }
  }
}

function Stop-Bridge {
  if ($bridgeProcess -and -not $bridgeProcess.HasExited) {
    Write-Host "Stopping the Figma bridge (PID $($bridgeProcess.Id))..."
    Stop-Process -Id $bridgeProcess.Id -Force -ErrorAction SilentlyContinue
  }
}

if ($InputDir) {
  if ($RequireFigma -and $NoBridge) {
    throw "Figma acceptance needs the bridge. Remove -NoBridge so the plugin can fetch the staged run and post its export."
  }
  Write-Host "Running the benchmark..."
  $benchmarkArgs = @("benchmark.py", "--input-dir", $InputDir, "--output", $Output, "--config", "config.yaml")
  # Name fixtures instead of relying on directory ordering. This makes the Codia-parity
  # corpus repeatable and lets a small representative batch run before the full library.
  foreach ($id in @($Ids)) {
    if ($null -ne $id -and "$id".Trim()) { $benchmarkArgs += @("--ids", "$id") }
  }
  if ($RequireFigma) {
    $benchmarkArgs += @("--require-figma-export", "--figma-wait-s", "$FigmaWaitS")
  }
  # Only skip the benchmark's own doctor check (and its doctor.json evidence) when the caller
  # explicitly asked to via -SkipDoctor. Acceptance runs must keep it so doctor.json is written.
  if ($SkipDoctor) { $benchmarkArgs += "--skip-doctor" }
  & $Python @benchmarkArgs
  $benchmarkExit = $LASTEXITCODE
  Stop-Bridge
  exit $benchmarkExit
}

Write-Host ""
Write-Host "Bridge is running. Import figma-plugin\manifest.json in Figma Desktop."
Write-Host "To run images: .\start_rtx.ps1 -InputDir C:\images\benchmark"
Write-Host "Press Ctrl+C to stop this launcher."
try {
  while ($true) { Start-Sleep -Seconds 2 }
} finally {
  Stop-Bridge
}
