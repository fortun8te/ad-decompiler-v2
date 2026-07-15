# Safely sync ad-decompiler on the Windows RTX box.
# The Python helper refuses dirty/diverged worktrees and only uses git pull --ff-only.
param(
  [string]$Repo = $(if ($env:AD_DECOMPILER_ROOT) { $env:AD_DECOMPILER_ROOT } else { (Split-Path -Parent $PSScriptRoot) }),
  [string]$Remote = "origin",
  [string]$Branch = "main",
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path (Join-Path $Repo ".git"))) {
  throw "Not a git repo: $Repo (set AD_DECOMPILER_ROOT or clone first)"
}

Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { $Python = "py" }
  elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = "python" }
  else { throw "No Python found. Install Python 3.12 or run setup_rtx.ps1 first." }
}

$syncArgs = @((Join-Path $Repo "scripts\sync_update.py"), "--repo", $Repo, "--remote", $Remote, "--branch", $Branch, "--notify")
if (-not $CheckOnly) { $syncArgs += "--update" }
& $Python @syncArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Restart Start Bridge.bat if an update was applied and the bridge is already running."
