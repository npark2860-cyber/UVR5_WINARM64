param(
    [string]$UVRPython = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $UVRPython) {
    $UVRPython = Join-Path $RepoRoot ".venv-arm64\Scripts\python.exe"
}
if (-not (Test-Path $UVRPython)) {
    throw "UVR ARM64 Python not found: $UVRPython"
}

& $UVRPython -m pip uninstall -y torch
if ($LASTEXITCODE -ne 0) { throw "Failed to uninstall torch." }

& $UVRPython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"
if ($LASTEXITCODE -ne 0) { throw "Failed to restore official torch wheel." }

& $UVRPython -c "import torch; print(torch.__version__); print(torch.__config__.show())"
if ($LASTEXITCODE -ne 0) { throw "Restored torch import failed." }

Write-Host "Official PyTorch 2.14.0+cpu restored."
