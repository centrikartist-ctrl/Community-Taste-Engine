# Community Taste Engine

A judgement layer that scores candidate media, posts, and ideas and explains why they deserve attention.

The core question is simple:

> Given a pile of candidate clips, posts, ideas, memes, or claims, what deserves attention, and why?

Video scoring is still here, but it is now one module inside the Community Taste Engine instead of the whole identity.

Non-goal:
this repo does not import source URLs, compose clips, or render final MP4 batches.
That manual-URL media execution path belongs in tools like `capcut-cli`, not here.

## Trust pass

For a fresh clone, the one-command verification path is:

```bash
python scripts/bootstrap_verify.py
```

That command creates or reuses `.venv`, installs `requirements.txt`, checks for `ffmpeg`,
then runs the public judgement command, checked-in evals, both trust passes, and the full pytest suite.

If you are on a machine without `ffmpeg` and only want the judgement-layer checks:

```bash
python scripts/bootstrap_verify.py --skip-media
```

For the human-facing trust pass, run:

```bash
python scripts/trust_judgement_pass.py
```

This generates:

- `trust/judgement_pass_report.json`
- `trust/judgement_pass_report.md`

That pass reruns the canonical checked-in five-candidate example in `examples/candidates.json`
and confirms the same top two described in `examples/judgements.json`.

For the lower-level media-module trust pass, run:

```bash
python scripts/trust_ugly_pass.py
```

This generates:

- `trust/ugly_success_report.json`
- `trust/ugly_success_report.md`

The trust pass verifies one end-to-end dry-run path:
source clip generation -> judged pairing evaluation -> report output.

## Core contract

Input:

- `candidates.json`

Output:

- `judgements.json`

Run it like this:

```bash
python judge.py candidates.json --output judgements.json
```

The main contract filenames are `candidates.json` and `judgements.json`.

Canonical example files live in `examples/candidates.json` and `examples/judgements.json`.

The checked-in canonical story lives under `examples/`.
The root-level `candidates.json` and `judgements.json` files are convenience working copies for local runs.

The public contract is also described in `schemas/candidates.schema.json` and `schemas/judgements.schema.json`.

The checked-in example is portable on purpose: it uses community-style candidates and does not require a local media file.
The judgement trust pass uses that same checked-in example so the public story stays aligned.

Try it directly:

```bash
python judge.py candidates.json --output judgements.json
```

Stable CLI flags:

- `--output` writes `judgements.json`
- `--summary-output` writes only the summary block
- `--work-dir` controls temporary scoring state
- `--min-confidence` tunes media scoring threshold
- `--quiet` suppresses stdout summary output
- `--skip-schema-validation` allows local extended payloads outside the strict checked-in schema

Exit behavior:

- exit `0` on success
- exit `1` on input, schema, or file I/O failure

Each judgement contains:

- `candidate_id`
- `kind`
- `score`
- `status`
- `reasons`
- `risks`
- `recommended_action`

It may also include:

- `title`

Status buckets are:

- `strong_signal`
- `needs_work`
- `unclear`
- `probably_noise`

Recommended actions are:

- `feature`
- `refine`
- `clarify`
- `repair_input`
- `pass`

## What it does

Ranks mixed candidate batches and explains the call in plain language.

That includes:

- local video clips
- links
- ideas
- claims
- memes
- community submissions

For local video candidates, the pipeline uses the existing rhythm, speech-boundary, energy, and pairing analysis.
For non-media candidates, it uses structured community/taste signals you provide in `signals`.

## Community mode

This repo is now useful even when there is no editing step yet.

You can feed it a community batch from Discord or anywhere else, for example:

- ideas from a brainstorm thread
- links people think are worth posting about
- memes to test for relevance
- clips that might deserve production attention
- claims that need ranking before anyone builds content around them

The judgement layer tells you which items are strong signal, which need work, which are unclear, and which are probably noise.

The Discord adapter also does a first pass of community cleanup before ranking:

- collapses obvious duplicate posts around the same external source
- carries thread/reply context into the candidate description
- records simple author-history signals inside the candidate metadata

## Video scoring module

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
| `judge.py` | Candidate ranking + explained judgements |

## Dependencies

Python:

- Python 3.9+ for the judgement layer
- `judge.py` lazy-loads the media pipeline, so pure community batches do not require NumPy or ffmpeg just to rank ideas/posts/claims
- local video/media scoring still requires `numpy` and system `ffmpeg`

```
numpy
ffmpeg (system, only required for local video/media scoring)
```

The media modules also use postponed annotations so the checked-in code stays compatible with Python 3.9+.

## Evaluation

Run the checked-in ranking evals with:

```bash
python scripts/evaluate_judgements.py
```

The eval suite includes a Discord-style room batch that checks the intended ordering:
brand-risk flag, tooling review, and concrete community idea should beat pure price chatter and vague hype.

It also includes a room-derived anonymized batch under `evals/room_slice_redacted.eval.json`.
That batch was distilled from a real Discord CSV export, but usernames, message IDs, links, and raw message bodies were not checked in.
The committed fixture keeps only paraphrased candidate summaries plus redacted `signals` metadata such as `source`, `has_receipts`, `actionable`, and `risk_type`.

If you want to keep raw room messages for local calibration, store them under `evals/private/`.
That path is gitignored so public CI only runs the redacted/paraphrased evals.

Required for validation and test runs:

```
pytest
jsonschema
```

## Usage

### Judge a batch

```bash
python judge.py candidates.json --output judgements.json
```

### Score a local video candidate directly

```bash
pip install numpy

# dry run — logs decisions, no clips generated
python pipeline.py myvideo.mp4

# custom log
python pipeline.py myvideo.mp4 --log project.jsonl

# tune threshold
python pipeline.py myvideo.mp4 --min-confidence 0.45
```

`pipeline.py` remains the richer media-analysis module.
`judge.py` is the upstream decision layer that tells you what deserves to become something.

## Agent integration

For open-source consumers, the stable public surface is:

- `judge.py`
- `candidates.json`
- `judgements.json`
- `COMMUNITY_TASTE_ENGINE_GUIDE.md`
- `discord_adapter.py`
- `scripts/discord_to_candidates.py`
- `examples/`
- `schemas/`
- `scripts/trust_judgement_pass.py`

The lower-level media module remains available, but the repo no longer ships downstream compose helpers. The public story is judgement first.

The judgement layer is designed to be called programmatically by agents:

```python
from judge import judge_candidates

payload = judge_candidates([
        {
              "id": "candidate_1",
              "title": "Claim with receipts",
              "text": "Concrete evidence and a clear audience hook.",
              "signals": {"credibility": 0.9, "clarity": 0.85}
        }
])
```

The media module is still available directly:

```python
from pipeline import run

decisions = run(
      "myvideo.mp4",
      log_path="decisions.jsonl",
)
```

For progress streaming, provide a callback:

```python
def on_progress(stage, payload):
      print(stage, payload)

run("myvideo.mp4", progress_callback=on_progress)
```

To convert a Discord export into the main `candidates.json` contract:

```bash
python scripts/discord_to_candidates.py examples/discord_export.json --output candidates.json
```

If you are preparing a public eval fixture from a real-room export, use `--redact-public` so source identifiers and links are stripped before anything is committed:

```bash
python scripts/discord_to_candidates.py room_export.json --output out/redacted_candidates.json --redact-public
```

The neutral usage and contract notes live in `COMMUNITY_TASTE_ENGINE_GUIDE.md`.

## Evaluation

Offline ranking evaluation is checked in so the repo can prove more than one happy-path story:

```bash
python scripts/evaluate_judgements.py
```

That suite currently covers:

- the canonical five-candidate example
- the Discord-derived community batch
- a more ambiguous batch with `needs_work` and `unclear` cases

For human-review drift checks against an existing `judgements.json`, run:

```bash
python scripts/review_judgements.py examples/judgements.json examples/judgement_feedback.json
```

That gives the repo a judgement-layer feedback artifact even before live production traffic exists.

## Output

`judge.py` output is plain JSON for downstream agents and humans.

`pipeline.py` still prints a media-analysis summary like this:

```
[1/4] loading audio...
      47.3s @ 22050Hz
[2/4] building semantic index...
      83 chunks, 91 beats, 124.0 BPM
[3/4] planning cuts...
      12 candidates
[4/4] critiquing and logging...
       OK  12.440s  [0.78]  sentence_end|beat_aligned
 WARN  23.190s  [0.51]  energy_spike
      BAD  31.020s  [0.31]  silence_gap

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
