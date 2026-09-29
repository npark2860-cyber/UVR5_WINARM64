import importlib
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

uvr = importlib.import_module("UVR")
print("PASS import UVR module")
if getattr(uvr, "is_dnd_compatible", True):
    raise SystemExit("FAIL: x64 TkDND must be disabled on native Windows ARM64")
print("PASS: x64 TkDND disabled for native ARM64")

print("PASS: Windows ARM64 bootstrap imports")
