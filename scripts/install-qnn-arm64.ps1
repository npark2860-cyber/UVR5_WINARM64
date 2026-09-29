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

# Plugin QNN EP 2.x is a separate package. Keep the regular ORT runtime and
# install the plugin alongside it.
& $UVRPython -m pip install --upgrade --only-binary=:all: "onnxruntime==1.30.0" "onnxruntime-qnn==2.6.0"
if ($LASTEXITCODE -ne 0) { throw "Failed to install ONNX Runtime + QNN plugin." }

$verify = @'
import platform
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

print("machine=", platform.machine())
print("onnxruntime=", ort.__version__)
print("onnxruntime_qnn=", qnn_ep.__version__)
print("qnn_plugin=", qnn_ep.get_library_path())
print("qnn_htp=", qnn_ep.get_qnn_htp_path())

assert platform.machine().upper() in ("ARM64", "AARCH64")

name = "QNNExecutionProvider"
ort.register_execution_provider_library(name, qnn_ep.get_library_path())
devices = ort.get_ep_devices()
qnn_devices = [d for d in devices if d.ep_name == name]
print("qnn_devices=", qnn_devices)
assert qnn_devices, "QNN EP registered but no QNN device was exposed"
ort.unregister_execution_provider_library(name)
'@

& $UVRPython -c $verify
if ($LASTEXITCODE -ne 0) { throw "QNN plugin registration/device verification failed." }

Write-Host ""
Write-Host "QNN plugin and Snapdragon device are ready."
Write-Host "Run:"
Write-Host ".\scripts\run-mdx23c-qnn-benchmark.ps1"
