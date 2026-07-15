<#
.SYNOPSIS
  Download the FLUX.1 Fill (quantized GGUF) inpaint stack into a ComfyUI install.

.DESCRIPTION
  Thin wrapper over scripts/setup_flux_inpaint.py. Uses the repo's .venv Python when
  present so huggingface_hub is already available. No tokens are embedded; a gated repo
  falls back to your cached `huggingface-cli login`.

.PARAMETER ComfyDir
  Path to the ComfyUI install (the folder that contains models/). If omitted, the script
  prints the plan and the exact target subfolders, then exits without downloading.

.PARAMETER Quant
  GGUF quant of Flux Fill. Q6_K is the checked-in workflow/doctor default. Choose a smaller
  quant only when you also point inpaint.comfy.models.unet_gguf at that exact filename.

.PARAMETER List
  Print the download plan and exit.

.EXAMPLE
  .\scripts\setup_flux_inpaint.ps1 -ComfyDir "C:\ComfyUI"

  # After setting inpaint.mode: flux_comfy and inpaint.comfy.comfy_dir in config.yaml:
  .\.venv\Scripts\python.exe doctor.py --config config.yaml --deep

.EXAMPLE
  .\scripts\setup_flux_inpaint.ps1 -List
#>
param(
  [string]$ComfyDir = "",
  [string]$Quant = "Q6_K",
  [switch]$List
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

# Prefer the project venv so huggingface_hub is already installed; fall back to py/python.
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }
  elseif (Get-Command python -ErrorAction SilentlyContinue) { $python = "python" }
  else { throw "No Python found. Create the venv (setup_rtx.ps1) or install Python 3.12." }
}

$pyScript = Join-Path $scriptDir "setup_flux_inpaint.py"
$argList = @($pyScript)
if ($List) { $argList += "--list" }
if ($ComfyDir -ne "") { $argList += @("--comfy-dir", $ComfyDir) }
$argList += @("--quant", $Quant)

Write-Host "Running: $python $($argList -join ' ')"
& $python @argList
exit $LASTEXITCODE
