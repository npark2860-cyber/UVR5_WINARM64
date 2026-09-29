import importlib
import platform
import sys

EXPECTED = "ARM64"
machine = platform.machine().upper()
print(f"Python: {sys.version.split()[0]}")
print(f"Machine: {machine}")
print(f"Executable: {sys.executable}")

if machine not in {"ARM64", "AARCH64"}:
    raise SystemExit(f"FAIL: native ARM64 Python required, got {machine}")

modules = [
    "numpy",
    "scipy",
    "soundfile",
    "torch",
    "onnx",
    "onnxruntime",
    "PIL",
    "psutil",
    "yaml",
    "tkinter",
]

for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "stdlib")
    print(f"PASS import {name}: {version}")

import torch
if torch.cuda.is_available():
    print("INFO: CUDA is available")
else:
    print("INFO: CPU backend active")

print("PASS: Windows ARM64 bootstrap imports")
