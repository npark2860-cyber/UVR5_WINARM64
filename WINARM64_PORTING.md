# Windows ARM64 Port - Phase 1 Bootstrap

## Source of truth

- Repository: `npark2860-cyber/UVR5_WINARM64`
- Upstream baseline commit: `ad64d7f4c000ed4b62e17c82790039dd290ffb4f`
- Upstream UVR version: `v5.6.0`
- Port branch: `exp/win-arm64-bootstrap`

The `main` branch remains the untouched baseline.

## Phase 1 goal

Start the UVR 5.6 GUI under **native CPython 3.12 Windows ARM64** without
Prism/x64 emulation.

This phase does not claim full processing parity. It establishes the native
runtime and identifies the next real blockers with minimal source changes.

## Architecture-sensitive packages selected

- PyTorch: `2.12.1+cpu`, official Windows ARM64 CPU wheel
- NumPy: `2.3.5`, Windows ARM64 wheel
- SciPy: `1.16.3`, Windows ARM64 wheel
- ONNX: `1.23.0`, Windows ARM64 wheel
- ONNX Runtime: `1.30.0`, Windows ARM64 wheel
- SoundFile: `0.14.0`, Windows ARM64 wheel
- Pillow: `12.1.1`, Windows ARM64 wheel
- psutil: `7.2.2`, Windows ARM64 wheel
- PyYAML: `6.0.3`, Windows ARM64 wheel
- cffi: `2.1.1`, Windows ARM64 wheel

## Dependencies intentionally deferred

### librosa / numba / llvmlite

UVR 5.6 uses librosa for loading, resampling, STFT and ISTFT. Current
CPython 3.12 Windows ARM64 availability does not provide a clean matching
numba/llvmlite path. Phase 1 therefore lazy-loads librosa instead of importing
it at process startup.

Phase 2 must replace the required librosa subset with ARM64-native primitives
or validate another compatible implementation before processing parity is
claimed.

### TkDND

The repository contains only the x64 Windows tkdnd binary. Native Windows ARM64
therefore disables drag and drop and falls back to ordinary Tk. File picker and
manual path entry remain available.

### cryptography / Matchering

These are feature-specific and are lazy-loaded. They must not prevent the main
GUI from starting.

### onnx2pytorch

Used at one processing path only and is lazy-loaded to avoid pulling additional
startup dependencies.

### PyTorch Lightning

`lib_v5/mdxnet.py` only used `LightningModule` as a base class and contains
no Lightning training hooks. The ARM64 branch uses `torch.nn.Module`
directly.

## Bootstrap

From native Windows ARM64 PowerShell:

```powershell
.\scripts\bootstrap-win-arm64.ps1 -Launch
```

The script rejects an x64 Python interpreter. A passing probe is therefore
evidence that the Python runtime and core numerical/ML dependencies are
actually native ARM64.

## Phase 1 PASS condition

1. Probe prints `Machine: ARM64` or `AARCH64`.
2. Core imports pass.
3. `UVR.py` opens the GUI.
4. No x64 Python/PyTorch/ONNX Runtime process is involved.

Processing-model validation belongs to Phase 2.
