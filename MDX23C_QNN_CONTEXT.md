# MDX23C QNN context cache

The UVR GUI must not JIT-compile a large QNN HTP graph. Real MDX23C model
preparation is therefore done once in a separate console process.

Run:

    .\scripts\prepare-mdx23c-qnn-context.ps1

The default uses QNN HTP graph finalization optimization mode 1 to prioritize
shorter preparation time. The generated embedded EP-context model is stored
next to the static ARM64 QNN ONNX model with the suffix:

    .ctx.onnx

After this file exists, UVR loads the precompiled context directly. If the
context is absent, UVR falls back to CPU ONNX Runtime rather than compiling in
the GUI process.

For a slower, more aggressively optimized context:

    .\scripts\prepare-mdx23c-qnn-context.ps1 -OptimizationMode 3
