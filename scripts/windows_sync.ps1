# Sync ad-decompiler on the Windows RTX box (git pull + stamp build).
# Can run locally or be fetched over HTTPS from the Mac:
#   irm https://raw.githubusercontent.com/fortun8te/ad-decompiler-v2/main/scripts/windows_sync.ps1 | iex
param(
  [string]$Repo = $(if ($env:AD_DECOMPILER_ROOT) { $env:AD_DECOMPILER_ROOT } else { "$HOME\ad-decompiler" }),
  [string]$Remote = "newrepo",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path (Join-Path $Repo ".git"))) {
  throw "Not a git repo: $Repo (set AD_DECOMPILER_ROOT or clone first)"
}

Set-Location $Repo
Write-Host "==> $Repo"
Write-Host "==> git fetch $Remote $Branch"
& git fetch $Remote $Branch
Write-Host "==> git pull $Remote $Branch"
& git pull $Remote $Branch

$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (Test-Path $Python) {
  & $Python (Join-Path $Repo "scripts\stamp_plugin_build.py") --quiet
  Write-Host "==> plugin build stamped"
} else {
  Write-Host "==> skip stamp (no .venv yet)"
}

$head = (& git rev-parse --short HEAD).Trim()
Write-Host "==> now at $head"
Write-Host "Restart Start Bridge.bat if the bridge is already running."
