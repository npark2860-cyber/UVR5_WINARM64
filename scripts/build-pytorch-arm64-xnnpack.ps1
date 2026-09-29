param(
    [string]$BuildRoot = "C:\pytorch-arm64-xnnpack",
    [string]$PythonExe = "python",
    [string]$TorchCommit = "08187d9e0fba026dc8217405802ab5381dc88d90",
    [int]$MaxJobs = 8,
    [string]$ArmPLDir = $env:ARMPL_DIR
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $RepoRoot "wheelhouse\xnnpack"
$SourceDir = Join-Path $BuildRoot "pytorch"
$BuildVenv = Join-Path $BuildRoot ".venv"

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$File,
        [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
    )
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')"
    }
}

function Import-VcVarsArm64 {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio Installer vswhere.exe was not found."
    }

    $vsPath = (& $vswhere -latest -products * -property installationPath).Trim()
    if (-not $vsPath) {
        throw "Visual Studio 2022 / Build Tools was not found."
    }

    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvarsall.bat was not found: $vcvars"
    }

    $tempCmd = Join-Path $env:TEMP "uvr5-vcvars-arm64.cmd"
    @"
@echo off
call "$vcvars" arm64 >nul
if errorlevel 1 exit /b %errorlevel%
set
"@ | Set-Content -Path $tempCmd -Encoding ASCII

    $envDump = & $env:ComSpec /d /c $tempCmd
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Visual Studio ARM64 native toolchain."
    }

    foreach ($line in $envDump) {
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    Remove-Item $tempCmd -Force -ErrorAction SilentlyContinue
    Write-Host "MSVC ARM64 environment: $vsPath"
}

Write-Host "[1/9] Checking native ARM64 Python"
Invoke-Checked $PythonExe -c "import platform,sys; print(sys.version); print(platform.machine()); assert platform.machine().upper() in ('ARM64','AARCH64'); assert sys.version_info[:2] in ((3,12),(3,13))"

Write-Host "[2/9] Checking build prerequisites"
foreach ($command in @("git", "cmake")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $command"
    }
}

if (-not $ArmPLDir) {
    throw "ARMPL_DIR is not set. Install Arm Performance Libraries for Windows first, then reopen PowerShell."
}
if (-not (Test-Path $ArmPLDir)) {
    throw "ARMPL_DIR does not exist: $ArmPLDir"
}
$env:ARMPL_DIR = $ArmPLDir
Write-Host "ARMPL_DIR=$ArmPLDir"

Write-Host "[3/9] Loading Visual Studio ARM64 native compiler environment"
Import-VcVarsArm64

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[4/9] Preparing PyTorch source at exact official-wheel commit"
if (-not (Test-Path (Join-Path $SourceDir ".git"))) {
    New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
    Invoke-Checked git -C $SourceDir init
    Invoke-Checked git -C $SourceDir remote add origin https://github.com/pytorch/pytorch.git
}

Invoke-Checked git -C $SourceDir fetch --depth 1 origin $TorchCommit
Invoke-Checked git -C $SourceDir checkout --force --detach FETCH_HEAD
Invoke-Checked git -C $SourceDir submodule sync
Invoke-Checked git -C $SourceDir submodule update --init --recursive --jobs $MaxJobs

Write-Host "[5/9] Preparing isolated build environment"
if (-not (Test-Path $BuildVenv)) {
    Invoke-Checked $PythonExe -m venv $BuildVenv
}
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"

Invoke-Checked $BuildPython -m pip install --upgrade pip setuptools wheel cmake ninja
Invoke-Checked $BuildPython -m pip install -r (Join-Path $SourceDir "requirements.txt")

Write-Host "[6/9] Configuring APL + XNNPACK build"
$env:BLAS = "APL"
$env:USE_LAPACK = "1"
$env:MSSdk = "1"
$env:DISTUTILS_USE_SDK = "1"
$env:USE_CUDA = "0"
$env:USE_CUDNN = "0"
$env:USE_DISTRIBUTED = "0"
$env:USE_GLOO = "0"
$env:USE_NCCL = "0"
$env:USE_KINETO = "0"
$env:BUILD_TEST = "0"
$env:USE_OPENMP = "1"
$env:USE_XNNPACK = "1"
$env:USE_NNPACK = "0"
$env:USE_PYTORCH_QNNPACK = "0"
$env:USE_MKLDNN = "0"
$env:USE_MKLDNN_ACL = "0"
$env:CMAKE_GENERATOR = "Ninja"
$env:MAX_JOBS = "$MaxJobs"
$env:PYTORCH_BUILD_VERSION = "2.14.0+xnnpack"
$env:PYTORCH_BUILD_NUMBER = "1"

Write-Host "USE_XNNPACK=$env:USE_XNNPACK"
Write-Host "USE_MKLDNN=$env:USE_MKLDNN"
Write-Host "BLAS=$env:BLAS"
Write-Host "MAX_JOBS=$env:MAX_JOBS"

Write-Host "[7/9] Building PyTorch wheel"
Push-Location $SourceDir
try {
    Invoke-Checked $BuildPython setup.py bdist_wheel
}
finally {
    Pop-Location
}

Write-Host "[8/9] Collecting wheel"
$wheel = Get-ChildItem -Path (Join-Path $SourceDir "dist") -Filter "torch-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $wheel) {
    throw "Build completed without a torch wheel in $SourceDir\dist"
}
if ($wheel.Name -notmatch "win_arm64") {
    throw "Unexpected wheel architecture: $($wheel.Name)"
}

$dest = Join-Path $OutputDir $wheel.Name
Copy-Item $wheel.FullName $dest -Force

$cache = Get-ChildItem -Path (Join-Path $SourceDir "build") -Filter "CMakeCache.txt" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($cache) {
    $summary = Join-Path $OutputDir "cmake-backend-summary.txt"
    Select-String -Path $cache.FullName -Pattern "USE_XNNPACK|USE_MKLDNN|USE_OPENMP|BLAS" |
        ForEach-Object { $_.Line } |
        Set-Content $summary
    Write-Host "CMake backend summary: $summary"
}

Write-Host "[9/9] Build complete"
Write-Host "Wheel: $dest"
Write-Host ""
Write-Host "Install into UVR with:"
Write-Host ".\scripts\install-pytorch-arm64-xnnpack.ps1"
