# PyTorch Windows ARM64 XNNPACK experiment

## Goal

The official PyTorch 2.14 Windows ARM64 CPU wheel reports APL BLAS but no
oneDNN and no XNNPACK. On the Snapdragon X Elite test machine, native ARM64
MDX23C inference is roughly an order of magnitude slower than the x64/Prism
build.

This experiment keeps the native APL setup and enables XNNPACK only.

## Experiment A

Build configuration:

- PyTorch source commit: 08187d9e0fba026dc8217405802ab5381dc88d90
- BLAS=APL
- USE_LAPACK=1
- USE_XNNPACK=1
- USE_MKLDNN=0
- USE_OPENMP=1
- USE_CUDA=0
- USE_DISTRIBUTED=0
- BUILD_TEST=0

oneDNN is deliberately excluded from this first build so performance changes
can be attributed to XNNPACK.

## Required system components

- Native Windows ARM64 Python 3.12 or 3.13
- Visual Studio 2022 C++ ARM64 build tools
- Arm Performance Libraries for Windows with ARMPL_DIR set
- Git
- CMake

## Build

From the UVR repository root:

    .\scripts\build-pytorch-arm64-xnnpack.ps1

The source/build tree is placed under:

    C:\pytorch-arm64-xnnpack

The resulting wheel is copied to:

    wheelhouse\xnnpack\

## Install into UVR

    .\scripts\install-pytorch-arm64-xnnpack.ps1

## Benchmark the exact UVR MDX23C architecture

    .\.venv-arm64\Scripts\python.exe .\scripts\benchmark_mdx23c_torch.py

The benchmark instantiates UVR's TFC_TDF_net using the 8K full-band MDX23C
configuration and measures one real CPU forward pass.

## Restore official PyTorch

    .\scripts\restore-official-pytorch-arm64.ps1

This restores torch 2.14.0+cpu from the official PyTorch CPU wheel index.
