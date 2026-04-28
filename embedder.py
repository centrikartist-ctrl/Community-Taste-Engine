"""
embedder.py — cross-modal embedding from scratch
-------------------------------------------------
ImageBind's key insight: map audio and video into the same embedding space
so you can measure compatibility via cosine similarity.

We build a lightweight version of that idea without a neural network:

  Audio embedding:
    — Mel-scaled filterbank energies (perceptual frequency weighting)
    — Spectral centroid (brightness)
    — Spectral rolloff (energy distribution)
    — RMS energy (loudness)
    — Zero crossing rate proxy (roughness)
    → Concatenated, L2-normalised → 40-dim float32 vector

  Visual embedding (from video frames via ffmpeg extraction):
    — Per-channel mean brightness
    — Brightness variance (motion proxy)
    — Colour saturation mean
    — Frame difference magnitude (motion energy)
    → L2-normalised → 8-dim float32 vector

  Pairing score:
    Cosine similarity between audio and visual vectors gives a
    [0, 1] compatibility measure. Higher = better match.

This is explicitly weaker than ImageBind (no trained cross-modal alignment),
but it captures real signal: high-energy audio tends to pair with
high-motion, high-saturation visuals.

The agent can improve this over time by:
  1. Logging pairing scores alongside human-accepted / rejected cuts
  2. Learning weights for each feature dimension from that feedback
"""

from __future__ import annotations

import subprocess
import tempfile
import os
import struct
import numpy as np
from audio import load as load_audio
from stft import stft


# ─── Mel filterbank ──────────────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 80.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Build a (n_filters, n_fft//2 + 1) mel filterbank matrix.
    Each row is a triangular filter in frequency space.
    """
    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)

    # n_filters + 2 equally spaced mel points
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_left = bin_points[m - 1]
        f_centre = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_centre):
            if f_centre != f_left:
                filters[m - 1, k] = (k - f_left) / (f_centre - f_left)
        for k in range(f_centre, f_right):
            if f_right != f_centre:
                filters[m - 1, k] = (f_right - k) / (f_right - f_centre)

    return filters


# ─── Audio embedding ─────────────────────────────────────────────────────────

_FILTERBANK_CACHE: dict[tuple, np.ndarray] = {}

N_MELS = 32
N_FFT = 2048
HOP = 512


def audio_embedding(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a fixed-length audio embedding vector.

    Returns
    -------
    embedding : (N_MELS + 4,) float32, L2-normalised
    """
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0)
    S = stft(y, n_fft=N_FFT, hop_length=HOP)  # (n_frames, bins)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    S = np.maximum(S, 0.0)

    # Mel filterbank
    key = (N_MELS, N_FFT, sr)
    if key not in _FILTERBANK_CACHE:
        _FILTERBANK_CACHE[key] = mel_filterbank(N_MELS, N_FFT, sr)
    fb = _FILTERBANK_CACHE[key]

    # Mel spectrogram: (n_frames, N_MELS)
    with np.errstate(over="ignore", invalid="ignore"):
        mel_spec = S @ fb.astype(np.float64, copy=False).T
    mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
    mel_spec = np.maximum(mel_spec, 0.0)
    # Mean over time → (N_MELS,)
    mel_mean = mel_spec.mean(axis=0)

    bins = np.arange(S.shape[1], dtype=np.float32)
    freqs = bins * sr / N_FFT

    total_energy = S.sum(axis=1) + 1e-8  # (n_frames,)

    # Spectral centroid: mean frequency weighted by magnitude, averaged over frames
    centroid = np.mean((S * freqs[None, :]).sum(axis=1) / total_energy)

    # Spectral rolloff: frequency below which 85% of energy is contained
    cumsum = np.cumsum(S, axis=1)
    threshold = 0.85 * cumsum[:, -1:]
    rolloff_bins = np.argmax(cumsum >= threshold, axis=1)
    rolloff = np.mean(rolloff_bins * sr / N_FFT)

    # RMS energy
    rms = float(np.sqrt(np.mean(y ** 2)))

    # High-frequency energy ratio (roughness proxy)
    hf_ratio = float(S[:, N_FFT // 4:].mean() / (S.mean() + 1e-8))

    extra = np.array([centroid / 4000.0, rolloff / 8000.0, rms * 10.0, hf_ratio],
                     dtype=np.float32)

    vec = np.concatenate([mel_mean, extra])
    return _l2_norm(vec)


# ─── Visual embedding ─────────────────────────────────────────────────────────

def _extract_frames(
    video_path: str,
    t_start: float,
    t_end: float,
    n_frames: int = 8,
) -> list[np.ndarray]:
    """
    Extract n evenly spaced frames from [t_start, t_end] as RGB uint8 arrays.
    Uses ffmpeg subprocess — outputs raw RGB bytes.
    """
    duration = max(t_end - t_start, 0.1)
    fps = n_frames / duration

    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%04d.raw")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(t_start),
                "-t", str(duration),
                "-i", video_path,
                "-vf", f"fps={fps:.3f},scale=64:64",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                pattern.replace("_%04d.raw", "_out.raw"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        raw_path = pattern.replace("_%04d.raw", "_out.raw")
        if not os.path.exists(raw_path):
            return []

        with open(raw_path, "rb") as f:
            raw = f.read()

    # Each frame = 64 * 64 * 3 bytes
    frame_bytes = 64 * 64 * 3
    frames = []
    for i in range(len(raw) // frame_bytes):
        chunk = raw[i * frame_bytes: (i + 1) * frame_bytes]
        arr = np.frombuffer(chunk, dtype=np.uint8).reshape(64, 64, 3).astype(np.float32)
        frames.append(arr)

    return frames


def visual_embedding(
    video_path: str,
    t_start: float,
    t_end: float,
) -> np.ndarray | None:
    """
    Compute a fixed-length visual embedding for a video segment.

    Returns
    -------
    embedding : (8,) float32 L2-normalised, or None if extraction failed
    """
    frames = _extract_frames(video_path, t_start, t_end)
    if not frames:
        return None

    brightnesses = [f.mean(axis=(0, 1)) for f in frames]  # (n_frames, 3) RGB means
    br_array = np.stack(brightnesses)  # (n_frames, 3)

    # Mean brightness per channel
    mean_brightness = br_array.mean(axis=0) / 255.0  # (3,)

    # Brightness variance over time (motion proxy)
    brightness_variance = br_array.var(axis=0) / (255.0 ** 2)  # (3,)

    # Colour saturation: max(RGB) - min(RGB), averaged
    saturations = np.array([f.max(axis=2) - f.min(axis=2) for f in frames])
    mean_saturation = np.array([saturations.mean()])

    # Frame-to-frame difference (motion energy)
    if len(frames) > 1:
        diffs = [np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
                 for i in range(1, len(frames))]
        motion = np.array([np.mean(diffs) / 255.0])
    else:
        motion = np.array([0.0])

    vec = np.concatenate([mean_brightness, brightness_variance, mean_saturation, motion])
    return _l2_norm(vec.astype(np.float32))


# ─── Pairing score ────────────────────────────────────────────────────────────

def pairing_score(
    audio_emb: np.ndarray,
    visual_emb: np.ndarray,
) -> float:
    """
    Cosine similarity between audio and visual embeddings.

    Because the dimensions differ, we project audio down to visual size
    via simple mean-pooling of audio's mel features, then compare.

    Returns float in [-1, 1]; typical range for matched content: [0.3, 0.9].
    """
    # Project audio to same dim as visual by mean-pooling mel features.
    # np.array_split handles cases where n_aud < n_vis by producing empty
    # sub-arrays; we replace their mean with 0.0 to avoid NaN.
    n_vis = len(visual_emb)
    parts = np.array_split(audio_emb, n_vis)
    a = np.array(
        [p.mean() if len(p) > 0 else 0.0 for p in parts],
        dtype=np.float32,
    )
    a = _l2_norm(a)

    dot = float(np.dot(a, visual_emb))
    return max(-1.0, min(1.0, dot))


def _l2_norm(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return (v / norm).astype(np.float32)
