# MDX23C ONNX Runtime ARM64 benchmark

This benchmark tests the expensive neural-network core of UVR's MDX23C model
without waiting for a full song.

It deliberately excludes STFT/ISTFT from the ONNX graph. The exported core
contains the Conv2d, ConvTranspose2d, InstanceNorm, GELU, Linear, reshape,
concatenation, and elementwise operations that dominate MDX23C inference.

No downloaded UVR model is required for this speed test. Random weights are
used because weight values do not affect operator/backend throughput.

The ONNX model is exported using 32 time frames with a dynamic time axis, then
benchmarked at the real UVR inference size of 256 time frames.

Run:

    .\scripts\run-mdx23c-ort-benchmark.ps1

Useful output:

    ORT run=1 elapsed=...
    output_shape=...
    best=...

The output can be compared directly with the first-batch timing printed by the
native PyTorch MDX23C diagnostic.
