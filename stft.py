"""
stft.py — Short-Time Fourier Transform from scratch
-----------------------------------------------------
No scipy, no librosa. Pure numpy.

Returns magnitude spectrogram shape (n_frames, n_fft//2 + 1).
"""

import numpy as np


def hann_window(n: int) -> np.ndarray:
    """Hann window: 0.5 * (1 - cos(2π k / N))"""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute magnitude spectrogram.

    Parameters
    ----------
    y          : float32 mono signal
    n_fft      : FFT window size
    hop_length : hop between frames

    Returns
    -------
    S : (n_frames, n_fft//2 + 1) float32 magnitude spectrogram
    """
    window = hann_window(n_fft)

    # Pad so every sample is covered
    pad = n_fft // 2
    y_padded = np.pad(y, pad, mode="reflect")

    n_frames = 1 + (len(y_padded) - n_fft) // hop_length

    # Stack frames: (n_frames, n_fft)
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop_length
    frames = y_padded[idx] * window  # apply Hann window

    # FFT of each frame, take magnitude of positive frequencies
    S = np.abs(np.fft.rfft(frames, n=n_fft)).astype(np.float32)
    return S


def frames_to_time(frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Convert frame indices to time in seconds."""
    return (frames * hop_length / sr).astype(np.float64)


def time_to_frames(times: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Convert times in seconds to frame indices."""
    return (np.asarray(times) * sr / hop_length).astype(int)
