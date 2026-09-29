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

& $UVRPython -m pip uninstall -y onnxruntime-qnn onnxruntime
if ($LASTEXITCODE -ne 0) { throw "Failed to remove QNN ONNX Runtime." }

& $UVRPython -m pip install --only-binary=:all: "onnxruntime==1.30.0"
if ($LASTEXITCODE -ne 0) { throw "Failed to restore onnxruntime 1.30.0." }

& $UVRPython -c "import onnxruntime as ort; print('onnxruntime=',ort.__version__); print('providers=',ort.get_available_providers())"
if ($LASTEXITCODE -ne 0) { throw "Restored ONNX Runtime import failed." }

Write-Host "Standard ONNX Runtime restored."
