param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Output) {
    $Output = Join-Path $RepoRoot "UVR5_ARM64.exe"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio Installer / vswhere.exe was not found."
}

$vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath).Trim()
if (-not $vsPath) {
    $vsPath = (& $vswhere -latest -products * -property installationPath).Trim()
}
if (-not $vsPath) {
    throw "Visual Studio 2022 or Build Tools was not found."
}

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat was not found: $vcvars"
}

$buildDir = Join-Path $RepoRoot "build\launcher-arm64"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$resPath = Join-Path $buildDir "UVR5_ARM64.res"
$cppPath = Join-Path $RepoRoot "launcher\UVR5_ARM64.cpp"
$rcPath = Join-Path $RepoRoot "launcher\UVR5_ARM64.rc"

$tempCmd = Join-Path $env:TEMP "build-uvr5-arm64-launcher.cmd"

@"
@echo off
call "$vcvars" arm64 >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "$RepoRoot"
rc.exe /nologo /fo "$resPath" "$rcPath"
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /O2 /EHsc /DUNICODE /D_UNICODE /Fe:"$Output" "$cppPath" "$resPath" user32.lib shell32.lib /link /SUBSYSTEM:WINDOWS /MACHINE:ARM64
exit /b %errorlevel%
"@ | Set-Content -Path $tempCmd -Encoding ASCII

try {
    & $env:ComSpec /d /c $tempCmd
    if ($LASTEXITCODE -ne 0) {
        throw "ARM64 launcher build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Remove-Item $tempCmd -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $Output)) {
    throw "Build reported success but output EXE was not created: $Output"
}

$bytes = [System.IO.File]::ReadAllBytes($Output)
if ($bytes.Length -lt 0x100) {
    throw "Output EXE is unexpectedly small."
}

$peOffset = [BitConverter]::ToInt32($bytes, 0x3C)
$machine = [BitConverter]::ToUInt16($bytes, $peOffset + 4)

if ($machine -ne 0xAA64) {
    throw ("Output is not ARM64 PE. Machine=0x{0:X4}" -f $machine)
}

Write-Host ""
Write-Host "PASS: Native Windows ARM64 launcher built."
Write-Host "EXE: $Output"
Write-Host "Machine: ARM64 (0xAA64)"
Write-Host ""
Write-Host "You can now start UVR by double-clicking UVR5_ARM64.exe."
