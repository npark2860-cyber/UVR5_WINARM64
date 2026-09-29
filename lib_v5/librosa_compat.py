"""Small librosa-compatible subset for native Windows ARM64 UVR.

UVR 5.6 only uses five librosa entry points at runtime:
load, resample, stft, istft, and get_duration.

This module implements those operations with NumPy, SciPy, and SoundFile so
the Windows ARM64 port does not depend on numba/llvmlite.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample_poly


def _pad_center(data: np.ndarray, size: int) -> np.ndarray:
    if data.shape[-1] > size:
        raise ValueError("window is longer than n_fft")
    total = size - data.shape[-1]
    left = total // 2
    right = total - left
    return np.pad(data, (left, right))


def _fix_length(data: np.ndarray, size: int, axis: int = -1) -> np.ndarray:
    current = data.shape[axis]
    if current == size:
        return data
    if current > size:
        index = [slice(None)] * data.ndim
        index[axis] = slice(0, size)
        return data[tuple(index)]

    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, size - current)
    return np.pad(data, pad_width)


def resample(
    y: np.ndarray,
    orig_sr: float,
    target_sr: float,
    *,
    res_type: str | None = None,
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
    **_: Any,
) -> np.ndarray:
    """Resample along the requested axis."""
    del res_type

    y = np.asarray(y)
    if float(orig_sr) <= 0 or float(target_sr) <= 0:
        raise ValueError("sample rates must be positive")
    if float(orig_sr) == float(target_sr):
        return np.array(y, copy=True)

    ratio = Fraction(float(target_sr) / float(orig_sr)).limit_denominator(1_000_000)
    output = resample_poly(y, ratio.numerator, ratio.denominator, axis=axis)

    if fix:
        expected = int(np.ceil(y.shape[axis] * float(target_sr) / float(orig_sr)))
        output = _fix_length(output, expected, axis=axis)

    if scale:
        output = output / np.sqrt(float(target_sr) / float(orig_sr))

    return output.astype(y.dtype, copy=False) if np.issubdtype(y.dtype, np.floating) else output


def load(
    path: str | Path,
    sr: float | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    dtype=np.float32,
    res_type: str | None = None,
    **_: Any,
):
    """Load audio using SoundFile with librosa-compatible channel layout."""
    with sf.SoundFile(path) as audio:
        native_sr = int(audio.samplerate)
        start = max(0, int(float(offset) * native_sr))
        audio.seek(min(start, len(audio)))

        frames = -1
        if duration is not None:
            frames = max(0, int(float(duration) * native_sr))

        data = audio.read(frames=frames, dtype="float32", always_2d=True)

    data = np.asarray(data, dtype=dtype).T

    if mono:
        data = np.mean(data, axis=0, dtype=dtype)

    target_sr = native_sr if sr is None else sr
    if sr is not None and float(native_sr) != float(sr):
        data = resample(
            data,
            orig_sr=native_sr,
            target_sr=sr,
            res_type=res_type,
            axis=-1,
        )

    return np.asarray(data, dtype=dtype), target_sr


def stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | tuple | float | np.ndarray = "hann",
    center: bool = True,
    dtype=None,
    pad_mode: str = "constant",
    **_: Any,
) -> np.ndarray:
    """Short-time Fourier transform matching librosa's default layout."""
    y = np.asarray(y)
    if y.ndim < 1:
        raise ValueError("audio must have at least one dimension")

    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("n_fft, hop_length and win_length must be positive")

    if isinstance(window, np.ndarray):
        fft_window = np.asarray(window)
    else:
        fft_window = get_window(window, win_length, fftbins=True)
    fft_window = _pad_center(fft_window, n_fft)

    if center:
        pad = n_fft // 2
        padding = [(0, 0)] * y.ndim
        padding[-1] = (pad, pad)
        y = np.pad(y, padding, mode=pad_mode)

    if y.shape[-1] < n_fft:
        raise ValueError("n_fft is too large for uncentered input")

    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft, axis=-1)
    frames = frames[..., ::hop_length, :]
    spectrum = np.fft.rfft(frames * fft_window, n=n_fft, axis=-1)
    spectrum = np.swapaxes(spectrum, -2, -1)

    if dtype is None:
        dtype = np.complex64 if y.dtype == np.float32 else np.complex128

    return np.asfortranarray(spectrum.astype(dtype, copy=False))


def istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: str | tuple | float | np.ndarray = "hann",
    center: bool = True,
    dtype=None,
    length: int | None = None,
    **_: Any,
) -> np.ndarray:
    """Inverse STFT with librosa-style overlap-add/window normalization."""
    matrix = np.asarray(stft_matrix)
    if matrix.ndim < 2:
        raise ValueError("stft_matrix must have frequency and frame axes")

    if n_fft is None:
        n_fft = 2 * (matrix.shape[-2] - 1)
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    if isinstance(window, np.ndarray):
        ifft_window = np.asarray(window)
    else:
        ifft_window = get_window(window, win_length, fftbins=True)
    ifft_window = _pad_center(ifft_window, n_fft)

    frames = np.fft.irfft(np.swapaxes(matrix, -2, -1), n=n_fft, axis=-1)
    n_frames = matrix.shape[-1]
    expected_len = n_fft + hop_length * max(0, n_frames - 1)

    leading_shape = matrix.shape[:-2]
    y = np.zeros(leading_shape + (expected_len,), dtype=np.float64)
    window_sum = np.zeros(expected_len, dtype=np.float64)
    window_sq = np.asarray(ifft_window, dtype=np.float64) ** 2

    for frame_index in range(n_frames):
        start = frame_index * hop_length
        stop = start + n_fft
        y[..., start:stop] += frames[..., frame_index, :] * ifft_window
        window_sum[start:stop] += window_sq

    nonzero = window_sum > np.finfo(window_sum.dtype).tiny
    y[..., nonzero] /= window_sum[nonzero]

    if center:
        trim = n_fft // 2
        y = y[..., trim:] if length is not None else y[..., trim:-trim if trim else None]

    if length is not None:
        y = _fix_length(y, int(length), axis=-1)

    if dtype is None:
        dtype = np.float32 if matrix.dtype == np.complex64 else np.float64
    return y.astype(dtype, copy=False)


def get_duration(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    path: str | Path | None = None,
    filename: str | Path | None = None,
    **_: Any,
) -> float:
    """Return duration from an array or audio file."""
    if y is not None:
        return float(np.asarray(y).shape[-1]) / float(sr)

    target = path if path is not None else filename
    if target is None:
        raise ValueError("provide either y or path")

    info = sf.info(target)
    return float(info.frames) / float(info.samplerate)
