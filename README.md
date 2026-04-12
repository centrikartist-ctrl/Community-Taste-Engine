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
capcut-cli (system, external)
```

Optional for development:

```
pytest
```

## Usage

```bash
pip install numpy

# dry run — logs decisions, no clips generated
python pipeline.py myvideo.mp4

# live — executes capcut-cli compose
python pipeline.py myvideo.mp4 --live --sound-id sound_123 --clip-id clip_a --clip-id clip_b

# custom log
python pipeline.py myvideo.mp4 --log project.jsonl

# tune threshold and output directory
python pipeline.py myvideo.mp4 --min-confidence 0.45 --output-dir outputs
```

In live mode, provide a CapCut sound + one or more clip IDs either by CLI args
or by environment variables:

```bash
set CAPCUT_SOUND_ID=sound_123
set CAPCUT_CLIP_ID=clip_a
# or multiple:
set CAPCUT_CLIP_IDS=clip_a,clip_b,clip_c
```

This repository is downstream of upstream capcut-cli discovery/import flows.
If your agent stack already resolved IDs (for example through
`pashpashpash/capcut-cli`), pass those IDs directly here.

Compose duration is configurable:

```bash
python pipeline.py myvideo.mp4 --live --duration-seconds 30
```

## Licensing and dependency policy

This repository does not bundle capcut-cli source or binaries.
It integrates with a user-installed external tool via subprocess calls.

As of this writing, upstream `pashpashpash/capcut-cli` does not declare a
license in GitHub metadata. Treat it as an external dependency and verify
upstream licensing terms before redistribution.

## Agent integration

The pipeline is designed to be called programmatically by agents:

```python
from pipeline import run

decisions = run(
      "myvideo.mp4",
      log_path="decisions.jsonl",
      dry_run=True,
)
```

For progress streaming, provide a callback:

```python
def on_progress(stage, payload):
      print(stage, payload)

run("myvideo.mp4", progress_callback=on_progress)

# live compose path for agents
run(
      "myvideo.mp4",
      dry_run=False,
      sound_id="sound_123",
      clip_ids=["clip_a", "clip_b"],
      duration_seconds=30,
)
```

For headless agent operations, use the JSON shim:

```bash
# machine-readable environment/auth/dependency check
python agent_capcut.py preflight

# machine-readable compose invocation
python agent_capcut.py compose --sound-id sound_123 --clip-id clip_a --clip-id clip_b --duration-seconds 30
```

OpenClaw integration details are documented in `OPENCLAW_AGENT_PLAN.md`.

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
