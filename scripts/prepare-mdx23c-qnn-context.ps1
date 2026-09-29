param(
    [ValidateSet("0","1","2","3")]
    [string]$OptimizationMode = "1",
    [ValidateRange(1,8)]
    [int]$BatchSize = 1,
    [int]$Frames = 256,
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-arm64\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "UVR ARM64 Python not found: $Python"
}

Set-Location $RepoRoot

$argsList = @(
    ".\scripts\prepare_mdx23c_qnn_context.py",
    "--optimization-mode", $OptimizationMode,
    "--batch-size", "$BatchSize",
    "--frames", "$Frames"
)

if ($Model) {
    $argsList += @("--model", $Model)
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MDX23C QNN context preparation failed."
}
