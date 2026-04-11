"""
audio.py — raw PCM loading, no media libraries
------------------------------------------------
Loads WAV directly via stdlib wave module.
Non-WAV formats are decoded to PCM via ffmpeg subprocess.
Returns (samples: np.ndarray float32 mono, sample_rate: int)
"""

import wave
import struct
import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Unpack PCM bytes
    fmt = {1: "B", 2: "h", 4: "i"}.get(sampwidth)
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.array(struct.unpack(f"{n_frames * n_channels}{fmt}", raw), dtype=np.float32)

    # Normalise to [-1, 1]
    if sampwidth == 1:
        samples = (samples - 128) / 128.0
    else:
        samples /= float(2 ** (8 * sampwidth - 1))

    # Downmix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


def load(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Load any audio/video file as mono float32 PCM at target_sr.
    Uses stdlib wave for WAV, ffmpeg subprocess for everything else.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".wav":
        samples, sr = _load_wav(path)
    else:
        # Decode to temporary WAV via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", path,
                    "-ac", "1",
                    "-ar", str(target_sr),
                    "-f", "wav", tmp_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            samples, sr = _load_wav(tmp_path)
        finally:
            os.unlink(tmp_path)

    # Resample if needed (integer ratio, simple)
    if sr != target_sr:
        samples = _resample(samples, sr, target_sr)
        sr = target_sr

    return samples, sr


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Linear interpolation resample.
    Good enough for analysis; not for mastering.
    """
    ratio = target_sr / orig_sr
    n_out = int(len(y) * ratio)
    x_orig = np.linspace(0, len(y) - 1, len(y))
    x_new = np.linspace(0, len(y) - 1, n_out)
    return np.interp(x_new, x_orig, y).astype(np.float32)
