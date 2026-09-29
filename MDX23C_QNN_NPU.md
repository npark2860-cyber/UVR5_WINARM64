# MDX23C Snapdragon NPU probe

This is a strict feasibility test for running the MDX23C neural core on the
Snapdragon X NPU through ONNX Runtime QNN Execution Provider.

The test disables CPU fallback. A PASS therefore means the entire exported
MDX23C core was accepted and executed by QNN HTP.

## Enable QNN package

    .\scripts\install-qnn-arm64.ps1

This replaces the standard ONNX Runtime Python package in the existing
Windows ARM64 Python 3.13 UVR environment with onnxruntime-qnn 2.6.0.

## Run

    .\scripts\run-mdx23c-qnn-benchmark.ps1

The script exports a compact dynamic ONNX graph, fixes the time axis to the
real UVR size of 256 frames, then compiles it for QNN HTP with:

- backend_type=htp
- htp_performance_mode=burst
- htp_graph_finalization_optimization_mode=3
- enable_htp_fp16_precision=1
- CPU fallback disabled

## Restore standard ONNX Runtime

    .\scripts\restore-onnxruntime-arm64.ps1

Do this if QNN causes compatibility problems with the existing CPU ORT path.
