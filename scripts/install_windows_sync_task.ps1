# Install per-user Windows update tasks. The updater only fast-forwards clean checkouts.
param(
  [string]$Repo = $(if ($env:AD_DECOMPILER_ROOT) { $env:AD_DECOMPILER_ROOT } else { (Split-Path -Parent $PSScriptRoot) })
)

$ErrorActionPreference = "Stop"
$runner = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Repo\scripts\windows_sync.ps1`""

& schtasks.exe /Create /TN "AdDecompilerSyncAtLogon" /TR $runner /SC ONLOGON /F | Out-Null
& schtasks.exe /Create /TN "AdDecompilerSyncHourly" /TR $runner /SC HOURLY /MO 1 /F | Out-Null
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$Repo\scripts\windows_sync.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Installed Ad Decompiler update tasks (at logon and hourly). Dirty checkouts are never changed."
