param(
  [string]$SamPath = "C:\src\sam3",
  [switch]$SkipSamClone,
  [switch]$SkipGpuPackages
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "Python Launcher is missing. Install Python 3.12 first."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "Creating the Python environment..."
  Invoke-Checked "py" @("-3.12", "-m", "venv", ".venv")
}
$Python = (Resolve-Path ".venv\Scripts\python.exe").Path
Invoke-Checked $Python @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $Python @("-m", "pip", "install", "-r", "requirements.txt")

if (-not $SkipGpuPackages) {
  Write-Host "Installing RTX/CUDA packages..."
  Invoke-Checked $Python @("-m", "pip", "install", "torch==2.10.0", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")
  Invoke-Checked $Python @("-m", "pip", "install", "-r", "requirements-gpu.txt")
}

if (-not $SkipSamClone) {
  if (-not (Test-Path $SamPath)) {
    Write-Host "Installing the official SAM 3 package..."
    $samParent = Split-Path -Parent $SamPath
    New-Item -ItemType Directory -Force -Path $samParent | Out-Null
    Invoke-Checked "git" @("clone", "https://github.com/facebookresearch/sam3.git", $SamPath)
  }
  Invoke-Checked $Python @("-m", "pip", "install", "-e", $SamPath)
}

if (-not (Test-Path "config.yaml")) {
  Copy-Item "config.example.yaml" "config.yaml"
  Write-Host "Created config.yaml. Set the local SAM checkpoint path before running."
}

Write-Host ""
Write-Host "Setup finished. Checking the machine..."
& $Python doctor.py --config config.yaml
if ($LASTEXITCODE -ne 0) {
  throw "Setup is installed, but the machine is not ready yet. Fix the doctor output and rerun start_rtx.ps1."
}
Write-Host ""
Write-Host "Next: edit config.yaml if needed, then run .\start_rtx.ps1 -InputDir C:\images\benchmark"
