"""
vad.py — Voice Activity Detection from scratch
-----------------------------------------------
Three features per frame, combined into a speech/silence decision:

  1. Short-time energy (STE) — speech has higher energy than silence
  2. Zero Crossing Rate (ZCR) — voiced speech has low ZCR,
                                 unvoiced consonants have high ZCR,
                                 silence is variable but typically low
  3. Spectral entropy — speech has lower entropy than silence/noise
                        (energy concentrated in harmonic bands)

Decision: simple per-frame thresholds + hysteresis to prevent chattering.
Outputs a list of (start_sec, end_sec) speech regions.
"""

import numpy as np
from stft import stft, frames_to_time


# ── Frame-level features ──────────────────────────────────────────────────────

def short_time_energy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """RMS energy per frame."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    n = len(y)
    if n < frame_length:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (n - frame_length) // hop_length
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * hop_length: i * hop_length + frame_length]
        out[i] = float(np.sqrt(np.mean(frame ** 2)))
    return out


def zero_crossing_rate(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """ZCR per frame: number of sign changes / frame_length."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    n = len(y)
    if n < frame_length:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (n - frame_length) // hop_length
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * hop_length: i * hop_length + frame_length]
        signs = np.sign(frame)
        signs[signs == 0] = 1  # treat zero as positive
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        out[i] = float(crossings / frame_length)
    return out


def spectral_entropy(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    n_fft: int = 512,
) -> np.ndarray:
    """
    Shannon entropy of the normalised power spectrum per frame.
    Low entropy = energy concentrated (speech / tonal).
    High entropy = energy spread (noise / silence).
    """
    if frame_length <= 0 or hop_length <= 0 or n_fft <= 0:
        raise ValueError("frame_length, hop_length, and n_fft must be positive")
    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    if S.size == 0:
        return np.zeros(0, dtype=np.float32)
    power = S ** 2
    # Normalise each frame to a probability distribution
    row_sums = power.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    p = power / row_sums
    # Shannon entropy: -sum(p log p), guard against log(0)
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1).astype(np.float32)
    return entropy


# ── Decision logic ────────────────────────────────────────────────────────────

def _normalise(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def detect_speech(
    y: np.ndarray,
    sr: int,
    frame_length: int = 512,
    hop_length: int = 256,
    energy_thresh: float = 0.1,
    entropy_thresh: float = 0.7,
    min_speech_frames: int = 8,
    min_silence_frames: int = 4,
) -> list[tuple[float, float]]:
    """
    Detect speech regions. Returns list of (start_sec, end_sec) tuples.

    Thresholds operate on normalised [0,1] features:
      energy_thresh  : minimum normalised RMS to be considered active
      entropy_thresh : maximum normalised entropy to be considered speech
                       (speech is low-entropy; noise/silence is high-entropy)
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if len(y) == 0 or len(y) < frame_length:
        return []

    ste = _normalise(short_time_energy(y, frame_length, hop_length))
    entropy = _normalise(spectral_entropy(y, sr, frame_length, hop_length))

    # Align lengths (spectral entropy may differ by a frame)
    min_len = min(len(ste), len(entropy))
    if min_len == 0:
        return []
    ste = ste[:min_len]
    entropy = entropy[:min_len]

    # Per-frame speech decision
    is_speech = (ste >= energy_thresh) & (entropy <= entropy_thresh)

    # Hysteresis: avoid flicker
    # Expand speech by min_speech_frames, shrink silence by min_silence_frames
    smoothed = _hysteresis(is_speech, min_speech_frames, min_silence_frames)

    # Convert frame runs to (start_sec, end_sec)
    regions = []
    in_speech = False
    start_frame = 0

    for i, val in enumerate(smoothed):
        if val and not in_speech:
            in_speech = True
            start_frame = i
        elif not val and in_speech:
            in_speech = False
            t_start = start_frame * hop_length / sr
            t_end = i * hop_length / sr
            regions.append((t_start, t_end))

    if in_speech:
        regions.append((start_frame * hop_length / sr, len(smoothed) * hop_length / sr))

    return regions


def _hysteresis(
    mask: np.ndarray,
    min_on: int,
    min_off: int,
) -> np.ndarray:
    """
    Suppress short speech bursts and short silences.
    min_on  : minimum consecutive True frames to count as speech
    min_off : minimum consecutive False frames to count as silence
    """
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            # Find end of this True run
            j = i
            while j < n and out[j]:
                j += 1
            run_len = j - i
            if run_len < min_on:
                out[i:j] = False  # too short — kill it
            i = j
        else:
            j = i
            while j < n and not out[j]:
                j += 1
            run_len = j - i
            if run_len < min_off:
                out[i:j] = True  # silence too short — fill it
            i = j
    return out
