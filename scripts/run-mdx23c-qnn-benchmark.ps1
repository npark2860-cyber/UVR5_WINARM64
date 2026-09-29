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

& $Python -c "import onnxruntime_qnn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "onnxruntime-qnn plugin is not installed. Installing..."
    & (Join-Path $PSScriptRoot "install-qnn-arm64.ps1") -UVRPython $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic QNN installation failed."
    }
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
