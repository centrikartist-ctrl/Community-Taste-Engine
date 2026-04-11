"""
beat_tracker.py — Ellis (2007) beat tracking from scratch
----------------------------------------------------------
Two stages:

  Stage 1 — Tempo estimation
    Autocorrelation of the onset strength envelope (OSE).
    Weight autocorrelation by a Gaussian centred on expected BPM range.
    Peak of weighted AC → estimated inter-beat interval (IBI) in frames.

  Stage 2 — Beat sequence optimisation (Dynamic Programming)
    Objective: maximise sum of OSE values at beat times subject to
    a tempo-consistency penalty.

    Score function (Ellis eq. 1):
      C(i) = OSE(i) + max_{j} [ C(j) - alpha * (log(t_i - t_j) - log(period))^2 ]

    alpha controls tightness: higher = stricter tempo adherence.
    Backtrace from the highest-scoring frame to recover beat sequence.

Reference: D. P. W. Ellis, "Beat Tracking by Dynamic Programming,"
           J. New Music Research 36(1), 51–60, 2007.
"""

import numpy as np
from stft import frames_to_time
from onset import onset_strength


# ── 1. Tempo estimation via autocorrelation ──────────────────────────────────

def estimate_period(
    ose: np.ndarray,
    sr: int,
    hop_length: int,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
) -> int:
    """
    Estimate the dominant inter-beat interval (IBI) in frames.

    Uses full autocorrelation of the OSE, then restricts to lags
    corresponding to [min_bpm, max_bpm] and weights by a Gaussian
    centred on 120 BPM (most common musical tempo).

    Returns
    -------
    period_frames : int — estimated IBI in frames
    """
    n = len(ose)

    # Full autocorrelation (only positive lags needed)
    ac_full = np.correlate(ose, ose, mode="full")
    ac = ac_full[n - 1:]  # lags 0, 1, 2, ...

    # Convert BPM bounds to lag bounds (frames)
    lag_min = max(1, int(60.0 / max_bpm * sr / hop_length))
    lag_max = int(60.0 / min_bpm * sr / hop_length)
    lag_max = min(lag_max, len(ac) - 1)

    if lag_min >= lag_max:
        return max(lag_min, 1)

    # Gaussian prior centred on 120 BPM
    target_lag = 60.0 / 120.0 * sr / hop_length
    lags = np.arange(lag_min, lag_max + 1, dtype=float)
    prior = np.exp(-0.5 * ((lags - target_lag) / (target_lag * 0.4)) ** 2)

    weighted = ac[lag_min: lag_max + 1] * prior
    best = int(np.argmax(weighted)) + lag_min

    return best


# ── 2. Dynamic programming beat tracker ─────────────────────────────────────

def _dp_beats(
    ose: np.ndarray,
    period: int,
    alpha: float = 400.0,
) -> np.ndarray:
    """
    Core DP. Returns array of beat frame indices.

    Parameters
    ----------
    ose    : onset strength envelope
    period : estimated IBI in frames
    alpha  : tempo tightness (400 = Ellis default)
    """
    n = len(ose)

    # Cumulative score and backtrace pointer
    score = ose.copy().astype(np.float64)
    prev = np.arange(n, dtype=int)  # default: point to self

    # Search window: Ellis looks back 0.5 to 2 × period
    half = max(1, int(0.5 * period))
    double = int(2.0 * period)

    for i in range(1, n):
        lo = max(0, i - double)
        hi = max(0, i - half)

        if lo >= hi:
            continue

        candidates = np.arange(lo, hi)
        lags = i - candidates  # always > 0

        # Penalty: alpha * (log(lag / period))^2
        penalty = alpha * (np.log(lags.astype(float) / period)) ** 2

        values = score[candidates] - penalty
        best_idx = int(np.argmax(values))

        score[i] += values[best_idx]
        prev[i] = candidates[best_idx]

    # Backtrace from global max
    beats = []
    i = int(np.argmax(score))
    while True:
        beats.append(i)
        p = prev[i]
        if p == i:
            break
        i = p

    beats.reverse()
    return np.array(beats, dtype=int)


# ── 3. Public API ────────────────────────────────────────────────────────────

def track_beats(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    alpha: float = 400.0,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
) -> tuple[np.ndarray, float]:
    """
    Full beat tracking pipeline.

    Returns
    -------
    beat_times : np.ndarray of beat positions in seconds
    bpm        : estimated tempo in BPM
    """
    ose = onset_strength(y, sr, n_fft=n_fft, hop_length=hop_length)
    period = estimate_period(ose, sr, hop_length, min_bpm=min_bpm, max_bpm=max_bpm)

    beat_frames = _dp_beats(ose, period, alpha=alpha)
    beat_times = frames_to_time(beat_frames, sr, hop_length)

    bpm = 60.0 / (period * hop_length / sr)

    return beat_times, bpm
