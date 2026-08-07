# Local-only rogii install (easy come / easy go)
#   cd D:\sandbox\rogii-wellbore-topology
#   .\install-local.ps1
# Optional CUDA torch (RTX):  .\install-local.ps1 -Cuda
# Teardown: Remove-Item -Recurse -Force .venv

param(
  [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Root = (Resolve-Path $Root).Path
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Host "RWT_ROOT=$Root"
$sysPy = Get-Command python -ErrorAction SilentlyContinue
if (-not $sysPy) { throw "python 3.10+ required on PATH" }

if (-not (Test-Path $Py)) {
  & python -m venv $Venv
}
$env:Path = "$(Join-Path $Venv 'Scripts');$env:Path"

& $Py -m pip install -U pip setuptools wheel
& $Py -m pip install -r (Join-Path $Root "requirements.txt")
& $Py -m pip install -e $Root
if ($Cuda) {
  Write-Host "== torch cu128 =="
  & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  & $Py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
}

Write-Host "== unit tests =="
& $Py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== synthetic smoke =="
& $Py (Join-Path $Root "scripts\smoke_synthetic.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "LOCAL RWT OK. Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Kaggle: place token at %USERPROFILE%\.kaggle\kaggle.json then scripts\download_kaggle.py"
