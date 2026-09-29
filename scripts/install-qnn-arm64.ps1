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

& $UVRPython -m pip uninstall -y onnxruntime onnxruntime-qnn
if ($LASTEXITCODE -ne 0) { throw "Failed to remove existing ONNX Runtime package." }

& $UVRPython -m pip install --only-binary=:all: "onnxruntime-qnn==2.6.0"
if ($LASTEXITCODE -ne 0) { throw "Failed to install onnxruntime-qnn 2.6.0." }

& $UVRPython -c "import platform,onnxruntime as ort; print('machine=',platform.machine()); print('onnxruntime=',ort.__version__); print('providers=',ort.get_available_providers()); assert platform.machine().upper() in ('ARM64','AARCH64'); assert 'QNNExecutionProvider' in ort.get_available_providers()"
if ($LASTEXITCODE -ne 0) { throw "QNN provider verification failed." }

Write-Host ""
Write-Host "QNN Execution Provider is ready."
Write-Host "Run:"
Write-Host ".\scripts\run-mdx23c-qnn-benchmark.ps1"
