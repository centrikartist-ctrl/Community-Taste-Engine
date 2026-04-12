"""
onset.py — Spectral flux onset strength from scratch
------------------------------------------------------
Algorithm:
  1. Compute magnitude spectrogram (STFT)
  2. Log-compress to perceptually weight frequency bins
  3. First-order frame difference → half-wave rectify (keep increases only)
  4. Sum across frequency → onset strength envelope (OSE)
  5. Peak-pick with adaptive threshold

Reference: Ellis 2007 §2.1, Müller FMP §6.2
"""

import numpy as np
from stft import stft, frames_to_time


# ── 1. Onset Strength Envelope ──────────────────────────────────────────────

def onset_strength(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    log_scale: float = 1000.0,
) -> np.ndarray:
    """
    Compute onset strength envelope (OSE).

    Log-compressed spectral flux — half-wave rectified sum of
    frame-to-frame spectral differences.

    Parameters
    ----------
    y          : float32 mono signal
    sr         : sample rate
    n_fft      : FFT size
    hop_length : hop size
    log_scale  : C in log(1 + C * |S|), controls compression strength

    Returns
    -------
    ose : (n_frames - 1,) float32 onset strength per frame
    """
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    if S.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)

    # Log magnitude compression — perceptually weights quiet onsets
    S_log = np.log1p(log_scale * S)

    # First-order difference across frames
    diff = np.diff(S_log, axis=0)  # (n_frames-1, bins)

    # Half-wave rectify: keep only increases (onsets, not offsets)
    diff = np.maximum(0.0, diff)

    # Sum across frequency bins → scalar per frame
    ose = diff.sum(axis=1).astype(np.float32)
    if len(ose) == 0:
        return ose

    # Normalise to [0, 1] for stability across inputs
    peak = ose.max()
    if peak > 0:
        ose /= peak

    return ose


# ── 2. Peak Picking ──────────────────────────────────────────────────────────

def pick_peaks(
    ose: np.ndarray,
    sr: int,
    hop_length: int,
    pre_avg: int = 3,
    post_avg: int = 3,
    pre_max: int = 3,
    post_max: int = 3,
    delta: float = 0.07,
    wait: int = 1,
) -> np.ndarray:
    """
    Adaptive threshold peak picker.

    A frame i is a peak if:
      ose[i] == local_max(ose[i-pre_max : i+post_max])
      AND
      ose[i] >= local_mean(ose[i-pre_avg : i+post_avg]) + delta

    Parameters
    ----------
    ose        : onset strength envelope
    pre_avg    : frames before i for mean threshold
    post_avg   : frames after i for mean threshold
    pre_max    : frames before i for local max check
    post_max   : frames after i for local max check
    delta      : minimum height above local mean
    wait       : minimum gap between peaks (frames)

    Returns
    -------
    onset_frames : integer frame indices of detected onsets
    """
    n = len(ose)
    if n == 0:
        return np.zeros(0, dtype=int)
    peaks = []
    last_peak = -wait - 1

    for i in range(n):
        # Local max window
        lo_max = max(0, i - pre_max)
        hi_max = min(n, i + post_max + 1)
        if ose[i] != ose[lo_max:hi_max].max():
            continue

        # Local mean threshold
        lo_avg = max(0, i - pre_avg)
        hi_avg = min(n, i + post_avg + 1)
        threshold = ose[lo_avg:hi_avg].mean() + delta

        if ose[i] >= threshold and (i - last_peak) > wait:
            peaks.append(i)
            last_peak = i

    return np.array(peaks, dtype=int)


def onset_times(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    **kwargs,
) -> np.ndarray:
    """
    Convenience: return onset times in seconds.
    """
    ose = onset_strength(y, sr, hop_length=hop_length)
    frames = pick_peaks(ose, sr, hop_length, **kwargs)
    return frames_to_time(frames, sr, hop_length)
