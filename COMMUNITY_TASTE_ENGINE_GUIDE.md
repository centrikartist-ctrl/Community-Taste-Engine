# Community Taste Engine Guide

Community Taste Engine is a judgement layer for ranking candidate clips, posts, ideas, memes, and claims.

It is not the repo that imports source URLs, composes clips, or renders final MP4 batches.
That execution path belongs outside the Community Taste Engine.

## Core question

"Given a pile of candidate clips, posts, ideas, memes, or claims, what deserves attention, and why?"

## Canonical files

Input:

- `candidates.json`

Output:

- `judgements.json`

Main command:

```bash
python judge.py candidates.json --output judgements.json
```

The checked-in canonical examples live under `examples/`.
Root-level `candidates.json` and `judgements.json` are convenience working files for local runs.

## Expected judgement shape

Every judgement must include:

- `candidate_id`
- `kind`
- `score`
- `status`
- `reasons`
- `risks`
- `recommended_action`

It may also include:

- `title`

Status buckets:

- `strong_signal`
- `needs_work`
- `unclear`
- `probably_noise`

Recommended actions:

- `feature`
- `refine`
- `clarify`
- `repair_input`
- `pass`

## Preparing input

### Option 1: Write `candidates.json` directly

Populate the root `candidates.json` file with candidate items and signals.

### Option 2: Start from a community export

Convert a Discord-style export first:

```bash
python scripts/discord_to_candidates.py examples/discord_export.json --output candidates.json
```

Then run judgement:

```bash
python judge.py candidates.json --output judgements.json
```

The Discord conversion pass also collapses obvious duplicate posts, preserves thread context, and carries simple author-history metadata into the candidate batch.

## CLI contract

Stable `judge.py` flags:

- `--output`
- `--summary-output`
- `--work-dir`
- `--min-confidence`
- `--quiet`
- `--skip-schema-validation`

Exit behavior:

- `0` means success
- `1` means contract, validation, or file I/O failure

## Stable public surface

- `judge.py`
- `candidates.json`
- `judgements.json`
- `COMMUNITY_TASTE_ENGINE_GUIDE.md`
- `schemas/`
- `examples/`
- `scripts/trust_judgement_pass.py`

The judgement trust pass reruns `examples/candidates.json` and verifies the expected top two from `examples/judgements.json`.

Offline evaluation beyond the single trust pass lives in `scripts/evaluate_judgements.py` and `evals/*.eval.json`.
Human review comparisons live in `scripts/review_judgements.py` and `schemas/judgement_feedback.schema.json`.

## Secondary surface

- `pipeline.py` as the deeper media-analysis module.
- `scripts/trust_ugly_pass.py` as a lower-level systems check.

## Working rule

If a component helps decide what deserves attention, it is core.

If a component only helps make something after that decision, it is secondary.
