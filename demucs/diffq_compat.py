"""Optional diffq bridge for Windows ARM64.

diffq does not ship a CPython 3.12 Windows ARM64 wheel. UVR only needs it for
quantized Demucs model handling/training-oriented paths, so the base GUI and
non-quantized paths must not fail at import time when diffq is absent.
"""

try:
    from diffq import DiffQuantizer, UniformQuantizer, restore_quantized_state
    DIFFQ_AVAILABLE = True
except ImportError:
    DIFFQ_AVAILABLE = False

    class _MissingDiffQ:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "This Demucs path requires diffq, which is not available in "
                "the current native Windows ARM64 bootstrap environment."
            )

    DiffQuantizer = _MissingDiffQ
    UniformQuantizer = _MissingDiffQ

    def restore_quantized_state(*args, **kwargs):
        raise RuntimeError(
            "Loading this quantized Demucs state requires diffq, which is not "
            "available in the current native Windows ARM64 bootstrap environment."
        )
