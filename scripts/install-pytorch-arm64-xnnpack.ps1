param(
    [string]$UVRPython = "",
    [string]$Wheel = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $UVRPython) {
    $UVRPython = Join-Path $RepoRoot ".venv-arm64\Scripts\python.exe"
}
if (-not (Test-Path $UVRPython)) {
    throw "UVR ARM64 Python not found: $UVRPython"
}

if (-not $Wheel) {
    $candidate = Get-ChildItem -Path (Join-Path $RepoRoot "wheelhouse\xnnpack") -Filter "torch-*.whl" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate) {
        $Wheel = $candidate.FullName
    }
}

if (-not $Wheel -or -not (Test-Path $Wheel)) {
    throw "Custom XNNPACK torch wheel not found. Build it first."
}

Write-Host "Installing: $Wheel"
& $UVRPython -m pip uninstall -y torch
if ($LASTEXITCODE -ne 0) { throw "Failed to uninstall current torch." }

& $UVRPython -m pip install --no-deps --force-reinstall $Wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to install custom torch wheel." }

& $UVRPython -c "import platform,torch; print('machine=',platform.machine()); print('torch=',torch.__version__); print(torch.__config__.show()); print('threads=',torch.get_num_threads()); print('interop=',torch.get_num_interop_threads())"
if ($LASTEXITCODE -ne 0) { throw "Custom torch import failed." }

Write-Host ""
Write-Host "Installed custom ARM64 XNNPACK PyTorch."
Write-Host "Benchmark:"
Write-Host ".\.venv-arm64\Scripts\python.exe .\scripts\benchmark_mdx23c_torch.py"
