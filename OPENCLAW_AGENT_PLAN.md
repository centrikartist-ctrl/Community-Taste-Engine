# OpenClaw CapCut Agent Plan

This document defines how AI agents (including OpenClaw) should run the Judgement Pipeline with capcut-cli in fully headless mode.

## Goals

- Use capcut-cli as the only live executor path.
- Keep this repo legally clean (no vendorized capcut-cli source/binaries).
- Provide deterministic, machine-readable orchestration suitable for agent loops.

## Implemented Components

1. `capcut_automation.py`
- Preflight report (`capcut-cli` presence, ffmpeg presence, optional auth probe when token is configured).
- Deterministic compose execution wrapper.
- ID resolution from arguments or env vars.
- Structured error classes for agent decision logic.

2. `agent_capcut.py`
- Machine-readable command shim for agents.
- Commands:
  - `preflight`
  - `compose`
- Emits JSON responses with success/error payloads.

3. `pipeline.py`
- Live mode executes through the capcut automation layer.
- Requires sound and clip IDs for compose.
- Supports programmatic usage for agent pipelines.

## Required Inputs for Live Runs

- `sound_id` (or `CAPCUT_SOUND_ID`)
- One or more `clip_ids` (or `CAPCUT_CLIP_ID` / `CAPCUT_CLIP_IDS`)

## Suggested OpenClaw Execution Flow

1. Preflight
- Run: `python agent_capcut.py preflight`
- If `ok=false`, stop and repair environment.

2. ID Resolution
- Resolve IDs in upstream capcut-cli discovery/library workflow.
- Store in OpenClaw state:
  - `sound_id`
  - `clip_ids[]`

3. Judgement
- Run Judgement Pipeline dry:
  - `python pipeline.py <video> --min-confidence 0.35`
- Evaluate cut decisions and score outputs.

4. Compose
- Run headless compose via shim:
  - `python agent_capcut.py compose --sound-id <id> --clip-id <id1> --clip-id <id2> --duration-seconds 30`

5. Observe + Retry
- If command fails with JSON error payload, classify and retry only transient failures.

## Error Handling Policy

- Dependency errors (`capcut-cli` missing): non-retryable until environment is fixed.
- Input errors (missing IDs): non-retryable until planner provides valid IDs.
- Command failures (non-zero exit): retryable depending on stderr class and OpenClaw policy.
- Pairing failures (ffmpeg/embedding errors): fail-fast; treat as non-retryable until media/env issue is resolved.

## Security and Legal Notes

- Do not commit access tokens.
- Do not copy external capcut-cli source/binaries into this repository.
- Upstream capcut-cli license should be verified before redistribution workflows.

## CI Recommendations

- Unit tests (already implemented).
- Add a non-auth smoke check step:
  - `python agent_capcut.py preflight`
- Add explicit-ID compose integration smoke job.
