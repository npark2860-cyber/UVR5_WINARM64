"""Lazy optional dependencies used by the Windows ARM64 bootstrap.

The original UVR 5.6 source imports several packages at process start even though
they are only needed by specific tools. Some of those packages do not currently
ship a CPython 3.12 Windows ARM64 wheel. Keeping them lazy allows the native
ARM64 GUI and supported inference backends to start without emulation.

When an optional package becomes available, installing it is enough; callers do
not need to change.
"""

from importlib import import_module


class LazyModule:
    def __init__(self, module_name: str, feature_name: str):
        self._module_name = module_name
        self._feature_name = feature_name
        self._module = None

    def _load(self):
        if self._module is None:
            try:
                self._module = import_module(self._module_name)
            except Exception as exc:
                raise RuntimeError(
                    f"{self._feature_name} requires optional dependency "
                    f"'{self._module_name}', which is not installed in the "
                    "current Windows ARM64 bootstrap environment."
                ) from exc
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __dir__(self):
        return dir(self._load())


librosa = LazyModule("librosa", "Legacy librosa audio helpers")
matchering = LazyModule("matchering", "Matchering")
