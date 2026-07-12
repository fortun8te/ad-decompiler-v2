param(
  [string]$InputDir,
  [string]$Output = "runs\benchmark",
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
}

if (-not $SkipDoctor) {
  & $Python doctor.py --config config.yaml
  if ($LASTEXITCODE -ne 0) { throw "The machine is not ready. Fix the doctor output first." }
}

$bridgeProcess = $null
if (-not $NoBridge) {
  & $Python "$((Get-Location).Path)\scripts\stamp_plugin_build.py" --quiet
  & $Python -m src.bridge_bootstrap --config config.yaml --inbox "$HOME\figma-inbox" | Out-Null
  Write-Host "Starting the local Figma bridge on http://localhost:8790..."
  $bridgeProcess = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "src.figma_bridge", "--inbox", "$HOME\figma-inbox", "--port", "8790"
  ) -PassThru
  Start-Sleep -Seconds 1
  if ($bridgeProcess.HasExited) {
    throw "The Figma bridge exited immediately (exit code $($bridgeProcess.ExitCode)). Port 8790 may already be in use by a previous run -- check Task Manager for a stray python.exe, or rerun with -NoBridge."
  }
}

function Stop-Bridge {
  if ($bridgeProcess -and -not $bridgeProcess.HasExited) {
    Write-Host "Stopping the Figma bridge (PID $($bridgeProcess.Id))..."
    Stop-Process -Id $bridgeProcess.Id -Force -ErrorAction SilentlyContinue
  }
}

if ($InputDir) {
  Write-Host "Running the benchmark..."
  $benchmarkArgs = @("benchmark.py", "--input-dir", $InputDir, "--output", $Output, "--config", "config.yaml")
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
