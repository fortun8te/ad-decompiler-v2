param(
  [string]$SamPath = "C:\src\sam3",
  [switch]$SkipSamClone,
  [switch]$SkipGpuPackages,
  [switch]$SkipDoctor,
  [switch]$DeepDoctor
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}

function Invoke-OptionalPip([string]$Label, [string[]]$Packages) {
  Write-Host "Trying optional $Label..."
  & $Python -m pip install @Packages
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: $Label could not be installed. The main pipeline will use docTR instead."
  }
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
  Write-Host "Installing the high-quality inpainting fallback..."
  Invoke-Checked $Python @("-m", "pip", "install", "simple-lama-inpainting", "pytesseract")
  # Vectorization render-back gate (VTracer trace -> raster -> SSIM fidelity check). CairoSVG
  # is the primary rasterizer; resvg_py is the pure-Python Windows fallback. This is production
  # vectorization -- Potrace is only an optional monochrome fallback and is NOT required.
  Write-Host "Installing the vectorization render-back gate (CairoSVG + resvg)..."
  Invoke-Checked $Python @("-m", "pip", "install", "cairosvg", "resvg_py", "vtracer")
  # These are deliberately isolated: Paddle's CUDA libraries and Surya's pins have
  # both broken otherwise healthy Blackwell environments in all-or-nothing installs.
  Invoke-OptionalPip "PaddleOCR fallback" @("paddleocr>=3.0", "paddlepaddle-gpu>=3.0")
  Invoke-OptionalPip "Surya OCR fallback" @("surya-ocr>=0.6")
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

# Lets the one-click launcher distinguish a complete RTX setup from an old bridge-only venv.
New-Item -ItemType File -Force -Path ".venv\.rtx-setup-v4" | Out-Null

if (-not $SkipDoctor) {
  Write-Host ""
  Write-Host "Setup finished. Checking the machine..."
  $doctorArgs = @("doctor.py", "--config", "config.yaml")
  if ($DeepDoctor) {
    Write-Host "Running selected inpaint backend smoke too (this executes real RTX models)..."
    $doctorArgs += @("--deep", "--deep-output", "runs\runtime-smoke")
  }
  & $Python @doctorArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The software is installed, but model files/services still need attention."
    Write-Host "Fix the BLOCK lines above, then double-click Start Bridge.bat again."
    exit 2
  }
}
Write-Host ""
Write-Host "Flux Fill weights are installed separately with scripts\setup_flux_inpaint.ps1."
Write-Host "PowerPaint is an optional external adapter: this setup does not install or claim a PowerPaint model."
Write-Host "Vectorization uses VTracer + CairoSVG/resvg (installed above). Potrace is an optional"
Write-Host "monochrome-only fallback and is NOT required for a complete setup."
Write-Host "The VLM is google/gemma-4-12b in LM Studio; enable the lms CLI (run: lms bootstrap) if you"
Write-Host "use runtime.vram.evict_vlm_for_inpaint so the VLM can be unloaded during Flux inpaint."
Write-Host "Setup finished. Next: double-click Start Bridge.bat."
