# UVR5 ARM64 launcher EXE

The launcher is a tiny native Windows ARM64 executable. It does not bundle or
copy the Python/AI runtime. Instead, it starts the already-tested environment:

    .venv-arm64\Scripts\pythonw.exe UVR.py

This preserves the current PyTorch, ONNX Runtime, QNN plugin, NPU contexts,
models, and runtime files exactly as tested.

## Build once

From the repository root:

    .\scripts\build-uvr5-arm64-launcher.ps1

The output is:

    UVR5_ARM64.exe

It embeds the existing UVR icon and runs as a Windows GUI subsystem
application, so no PowerShell or console window is shown.

## Run

Double-click:

    UVR5_ARM64.exe

The launcher expects to remain in the repository root beside UVR.py and the
.venv-arm64 folder.

This is intentionally a launcher EXE, not a single-file frozen Python build.
A portable folder build can be added later after the native ARM64/QNN path is
fully stabilized.
