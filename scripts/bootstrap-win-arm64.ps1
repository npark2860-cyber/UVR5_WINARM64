param(
    [string]$PythonExe = "python",
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Run-Python {
    param([string[]]$Args)
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonExe $($Args -join ' ')"
    }
}

Write-Host "[1/6] Checking host Python architecture"
Run-Python @("-c", "import platform,sys; print(sys.version); print(platform.machine()); assert platform.machine().upper() in ('ARM64','AARCH64'), 'Native ARM64 Python required'")

$Venv = Join-Path $RepoRoot ".venv-arm64"
if (-not (Test-Path $Venv)) {
    Write-Host "[2/6] Creating ARM64 virtual environment"
    Run-Python @("-m", "venv", $Venv)
} else {
    Write-Host "[2/6] Reusing $Venv"
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"

Write-Host "[3/6] Updating pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "[4/6] Installing native Windows ARM64 PyTorch CPU"
& $VenvPython -m pip install "torch==2.12.1+cpu" --index-url "https://download.pytorch.org/whl/cpu"
if ($LASTEXITCODE -ne 0) { throw "PyTorch ARM64 installation failed" }

Write-Host "[5/6] Installing ARM64 bootstrap dependencies"
& $VenvPython -m pip install -r "requirements-win-arm64.txt"
if ($LASTEXITCODE -ne 0) { throw "ARM64 requirements installation failed" }

Write-Host "[6/6] Running architecture/import probe"
& $VenvPython "scripts\probe_win_arm64.py"
if ($LASTEXITCODE -ne 0) { throw "ARM64 probe failed" }

Write-Host ""
Write-Host "Bootstrap PASS."
Write-Host "Run: .\.venv-arm64\Scripts\python.exe UVR.py"

if ($Launch) {
    & $VenvPython "UVR.py"
}
