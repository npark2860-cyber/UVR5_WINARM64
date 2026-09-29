param(
    [int]$Runs = 3,
    [switch]$ForceExport
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-arm64\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "UVR ARM64 Python not found: $Python"
}

Set-Location $RepoRoot

& $Python -c "import onnxruntime as ort,sys; sys.exit(0 if 'QNNExecutionProvider' in ort.get_available_providers() else 7)"
if ($LASTEXITCODE -eq 7) {
    Write-Host "QNNExecutionProvider is not installed. Installing onnxruntime-qnn 2.6.0..."
    & (Join-Path $PSScriptRoot "install-qnn-arm64.ps1") -UVRPython $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic QNN installation failed."
    }
}
elseif ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect ONNX Runtime providers."
}

$argsList = @(
    ".\scripts\benchmark_mdx23c_qnn.py",
    "--runs", "$Runs"
)
if ($ForceExport) {
    $argsList += "--force-export"
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MDX23C QNN NPU benchmark failed."
}
