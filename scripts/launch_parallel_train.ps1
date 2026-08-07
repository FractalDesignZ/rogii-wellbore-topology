# Launch multi-core RWT train + live HTML dashboard (venv).
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts/launch_parallel_train.ps1
#   powershell -File scripts/launch_parallel_train.ps1 -MaxWells 40 -Workers 6

param(
  [string]$Data = "data/raw",
  [string]$Out = "artifacts/full_train_parallel",
  [int]$Folds = 5,
  [int]$MaxWells = 0,
  [int]$Workers = 0,
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Venv = Join-Path $Root ".venv-dash"
$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
  Write-Host "Creating dashboard venv at $Venv ..."
  python -m venv $Venv
  & $Py -m pip install --upgrade pip
  & $Py -m pip install -r requirements.txt matplotlib
}

# Stop previous single-thread full_train if still chewing one core
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
  if ($_.CommandLine -and ($_.CommandLine -match "rwt\.train" -or $_.CommandLine -match "full_train")) {
    Write-Host "Stopping prior train PID $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:PYTHONPATH = (Join-Path $Root "src")
$env:RWT_WORKERS = if ($Workers -gt 0) { "$Workers" } else { "" }
$env:RWT_LGBM_THREADS = "8"
# Per-worker BLAS when multi-process assemble
$env:RWT_BLAS_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$wArg = @()
if ($Workers -gt 0) { $wArg = @("--workers", "$Workers") }
$mwArg = @()
if ($MaxWells -gt 0) { $mwArg = @("--max-wells", "$MaxWells") }

Write-Host "Starting dashboard on http://127.0.0.1:$Port/"
$dash = Start-Process -FilePath $Py -ArgumentList @(
  "scripts/dashboard_server.py", "--out", $Out, "--port", "$Port"
) -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 1
Start-Process "http://127.0.0.1:$Port/"

Write-Host "Starting parallel train → $Out"
$trainArgs = @(
  "-m", "rwt.train",
  "--data", $Data,
  "--out", $Out,
  "--folds", "$Folds"
) + $wArg + $mwArg

# Foreground train so this window shows logs; dashboard is separate process
& $Py @trainArgs
$code = $LASTEXITCODE
Write-Host "Train exit=$code  dashboard PID=$($dash.Id) still serving $Out"
exit $code
