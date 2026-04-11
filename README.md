# Judgement Pipeline

An agentic cut decision engine for video editing. Not a wrapper around existing tools — the algorithms are written from scratch.

Most tools automate the mechanics of editing. This one tries to learn the judgment.

## What it does

Analyses a video and proposes cut points based on three things:

- **Rhythm** — is the cut on the beat?
- **Speech integrity** — does it break someone mid-sentence?
- **Energy coherence** — does the energy make sense across the cut?

Every decision is logged with its score. The planner reads that log on the next run and adjusts its own weights. It gets better with use.

## How it works

| File | Algorithm |
|------|-----------|
| `audio.py` | PCM loading via stdlib `wave` + ffmpeg |
| `stft.py` | Hann-windowed FFT from scratch |
| `onset.py` | Log-compressed spectral flux, half-wave rectified |
| `beat_tracker.py` | Ellis (2007) — autocorrelation tempo + DP backtrace |
| `vad.py` | Voice activity detection — energy, ZCR, spectral entropy |
| `aligner.py` | Syllable boundary detection via energy valleys |
| `embedder.py` | Mel filterbank + cosine similarity for audio-visual pairing |
| `pipeline.py` | Planner → Critic → Logger orchestrator |

## Dependencies

```
numpy
ffmpeg (system)
```

That's it.

## Usage

```bash
pip install numpy

# dry run — logs decisions, no cuts made
python pipeline.py myvideo.mp4

# live — calls capcut-cli
python pipeline.py myvideo.mp4 --live

# custom log
python pipeline.py myvideo.mp4 --log project.jsonl
```

## Output

```
[1/4] loading audio...
      47.3s @ 22050Hz
[2/4] building semantic index...
      83 chunks, 91 beats, 124.0 BPM
[3/4] planning cuts...
      12 candidates
[4/4] executing, critiquing, logging...
  ✓  12.440s  [0.78]  sentence_end|beat_aligned
  ~  23.190s  [0.51]  energy_spike
  ✗  31.020s  [0.31]  silence_gap

[rule performance across all runs]
  0.821 ████████████████    sentence_end (n=47)
  0.714 ██████████████      beat_aligned (n=83)
  0.502 ██████████          energy_spike (n=61)
```

## Adding a transcript

Any ASR output can be passed in to improve boundary detection:

```python
from aligner import build_chunks

segments = [
    {"start": 0.0, "end": 2.3, "text": "so the thing about this"},
    {"start": 2.5, "end": 4.1, "text": "is that it changes everything"},
]
chunks = build_chunks(y, sr, asr_segments=segments)
```

## Licence

MIT — take it apart, build on it, just keep the credit.
```
