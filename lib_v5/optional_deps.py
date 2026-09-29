"""Optional dependencies used by the Windows ARM64 port."""

from importlib import import_module
import platform


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
                    "current Windows ARM64 environment."
                ) from exc
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __dir__(self):
        return dir(self._load())


_is_windows_arm64 = (
    platform.system() == "Windows"
    and platform.machine().lower() in {"arm64", "aarch64"}
)

if _is_windows_arm64:
    librosa = import_module("lib_v5.librosa_compat")
else:
    librosa = LazyModule("librosa", "Legacy librosa audio helpers")

matchering = LazyModule("matchering", "Matchering")
