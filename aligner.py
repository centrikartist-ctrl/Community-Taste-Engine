"""
aligner.py — syllable boundary detection and semantic chunking
---------------------------------------------------------------
Without a neural forced-aligner (like whisperX's CTC model), we recover
approximate word/syllable boundaries from the acoustic signal:

  1. Within each VAD speech region, compute short-time energy envelope
  2. Find local energy minima below the region mean — these are
     inter-syllabic valleys (consonant clusters / brief pauses)
  3. Tag each chunk with duration, energy profile, and position in region

This gives the agent timestamp-anchored chunks it can reason about for
cut decisions — selecting "after the sentence ends" not "mid-word".

A text transcript (from any external ASR) can optionally be passed in
and will be aligned to the detected boundary timestamps.
"""

import numpy as np
from dataclasses import dataclass, field
from vad import detect_speech
from stft import frames_to_time


@dataclass
class Chunk:
    start: float             # seconds
    end: float
    duration: float
    mean_energy: float
    peak_energy: float
    is_speech: bool
    boundary_type: str       # "syllable", "word_gap", "sentence_end", "silence"
    transcript: str = ""     # filled in if ASR output provided
    tags: list[str] = field(default_factory=list)


def _frame_energy(y: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    if len(y) < frame_len:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (len(y) - frame_len) // hop
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        f = y[i * hop: i * hop + frame_len]
        energy[i] = float(np.sqrt(np.mean(f ** 2)))
    return energy


def _find_valleys(
    energy: np.ndarray,
    min_depth: float = 0.3,
    min_gap_frames: int = 3,
) -> list[int]:
    """
    Find local energy minima that fall below
    (mean - min_depth * std) of the region.
    These are inter-syllabic valleys.
    """
    if len(energy) < 3:
        return []

    threshold = energy.mean() - min_depth * energy.std()
    valleys = []
    last = -min_gap_frames - 1

    for i in range(1, len(energy) - 1):
        if (
            energy[i] < energy[i - 1]
            and energy[i] < energy[i + 1]
            and energy[i] < threshold
            and (i - last) > min_gap_frames
        ):
            valleys.append(i)
            last = i

    return valleys


def _classify_boundary(
    gap_duration: float,
    energy_at_valley: float,
    region_mean_energy: float,
) -> str:
    """
    Heuristic boundary classification based on gap duration and energy depth.
    """
    relative_depth = 1.0 - (energy_at_valley / max(region_mean_energy, 1e-6))

    if gap_duration > 0.3:
        return "sentence_end"
    elif gap_duration > 0.08 and relative_depth > 0.5:
        return "word_gap"
    else:
        return "syllable"


def build_chunks(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    frame_length: int = 512,
    asr_segments: list[dict] | None = None,
) -> list[Chunk]:
    """
    Build a list of Chunks covering the full audio.

    Parameters
    ----------
    y            : float32 mono signal
    sr           : sample rate
    hop_length   : analysis hop
    frame_length : analysis frame
    asr_segments : optional list of {start, end, text} dicts from any ASR

    Returns
    -------
    List of Chunk objects ordered by start time.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if len(y) == 0:
        return []

    speech_regions = detect_speech(y, sr, frame_length=frame_length, hop_length=hop_length)

    # Build a set of boundary times: start/end of each speech region
    # plus intra-region syllable valleys
    boundary_times = [0.0]

    for (t_start, t_end) in speech_regions:
        s_start = int(t_start * sr)
        s_end = min(int(t_end * sr), len(y))
        region_audio = y[s_start:s_end]

        if len(region_audio) < frame_length:
            continue

        region_energy = _frame_energy(region_audio, frame_length, hop_length)
        valley_frames = _find_valleys(region_energy)

        # Add region start
        boundary_times.append(t_start)

        # Add valley times (converted from local frame → global seconds)
        for vf in valley_frames:
            t_valley = t_start + vf * hop_length / sr
            boundary_times.append(t_valley)

        # Add region end
        boundary_times.append(t_end)

    boundary_times.append(len(y) / sr)
    boundary_times = sorted(set(round(t, 4) for t in boundary_times))
    if len(boundary_times) < 2:
        boundary_times = [0.0, len(y) / sr]

    # Build speech lookup: which (start, end) pairs cover a given time
    def in_speech(t: float) -> bool:
        for (s, e) in speech_regions:
            if s <= t < e:
                return True
        return False

    # Build Chunk list from boundary pairs
    chunks = []
    global_energy = _frame_energy(y, frame_length, hop_length)
    if len(global_energy) == 0:
        dur = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        return [
            Chunk(
                start=0.0,
                end=dur,
                duration=dur,
                mean_energy=rms,
                peak_energy=rms,
                is_speech=False,
                boundary_type="silence",
                tags=["low_energy"] if rms < 0.01 else [],
            )
        ]

    for i in range(len(boundary_times) - 1):
        t0 = boundary_times[i]
        t1 = boundary_times[i + 1]
        if t1 - t0 < 0.01:
            continue

        # Energy for this chunk
        f0 = max(0, min(len(global_energy) - 1, int(t0 * sr / hop_length)))
        f1 = max(f0 + 1, min(len(global_energy), int(t1 * sr / hop_length)))
        chunk_energy = global_energy[f0:f1] if f1 > f0 else np.array([0.0])

        mean_e = float(chunk_energy.mean())
        peak_e = float(chunk_energy.max())
        is_sp = in_speech((t0 + t1) / 2)

        # Boundary type
        if not is_sp:
            btype = "silence"
        else:
            valley_energy = float(global_energy[f0]) if f0 < len(global_energy) else 0.0
            region_mean = mean_e
            gap = t1 - t0
            btype = _classify_boundary(gap, valley_energy, region_mean)

        tags = []
        if is_sp:
            tags.append("speech")
        if mean_e > 0.05:
            tags.append("high_energy")
        if mean_e < 0.01:
            tags.append("low_energy")
        if btype == "sentence_end":
            tags.append("safe_cut_point")

        chunks.append(Chunk(
            start=t0,
            end=t1,
            duration=t1 - t0,
            mean_energy=mean_e,
            peak_energy=peak_e,
            is_speech=is_sp,
            boundary_type=btype,
            tags=tags,
        ))

    # Align ASR transcript if provided
    if asr_segments:
        chunks = _align_transcript(chunks, asr_segments)

    return chunks


def _align_transcript(
    chunks: list[Chunk],
    asr_segments: list[dict],
) -> list[Chunk]:
    """
    Assign ASR text to overlapping chunks.
    asr_segments: [{"start": float, "end": float, "text": str}, ...]
    """
    for chunk in chunks:
        texts = []
        for seg in asr_segments:
            # Overlap check
            overlap = min(chunk.end, seg["end"]) - max(chunk.start, seg["start"])
            if overlap > 0:
                texts.append(seg["text"].strip())
        chunk.transcript = " ".join(texts)
    return chunks
