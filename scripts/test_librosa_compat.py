"""Fixed-work audio compatibility checks for the Windows ARM64 port."""

from pathlib import Path
import tempfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from lib_v5 import librosa_compat as lc


def main():
    sr_in = 48000
    t = np.arange(sr_in, dtype=np.float64) / sr_in
    left = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.20 * np.sin(2 * np.pi * 660.0 * t)
    stereo = np.column_stack([left, right]).astype(np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "compat_48k.wav"
        sf.write(wav_path, stereo, sr_in, subtype="PCM_16")

        y, sr = lc.load(wav_path, sr=44100, mono=False)
        assert sr == 44100
        assert y.shape == (2, 44100), y.shape

        duration = lc.get_duration(y=y, sr=sr)
        assert abs(duration - 1.0) < 1e-6, duration

        d = lc.stft(y[0], n_fft=2048, hop_length=512)
        reconstructed = lc.istft(
            d,
            n_fft=2048,
            hop_length=512,
            length=y.shape[-1],
        )

        error = float(np.max(np.abs(reconstructed - y[0])))
        assert error < 1e-5, error

        half_rate = lc.resample(y[0], 44100, 22050)
        assert half_rate.shape[-1] == 22050

    print("PASS: 48kHz stereo WAV -> 44.1kHz load")
    print("PASS: STFT/ISTFT round-trip")
    print("PASS: resample/get_duration")


if __name__ == "__main__":
    main()
