# Release Checklist

Use this checklist before every public release.

## 1) Code health

- [ ] `python -m pytest -q` passes.
- [ ] `python agent_capcut.py preflight` returns `capcut_cli_available=true`.
- [ ] `ffmpeg` is available in `PATH`.

## 2) Live execution readiness

- [ ] Explicit-ID live path works:
  - `python pipeline.py <video> --live --sound-id <sound_id> --clip-id <clip_id>`
- [ ] Explicit-ID JSON shim path works:
  - `python agent_capcut.py compose --sound-id <sound_id> --clip-id <clip_id>`

## 3) Open-source hygiene

- [ ] No vendorized external capcut-cli source/binaries in repo.
- [ ] README setup steps match current CLI behavior.
- [ ] Changelog/commit message explains new behavior and migration notes.

## 4) Runtime observability

- [ ] Error JSON includes actionable details (`returncode`, `stderr`, `stdout`).
- [ ] Decision log path is documented and writable.
- [ ] One end-to-end smoke run on a sample video is recorded.

## Current status snapshot (2026-04-12)

- PASS: tests (`45 passed`)
- PASS: explicit-ID live path (`pipeline --live --sound-id ... --clip-id ...`)
- PASS: downstream contract (IDs provided by upstream discovery/import pipeline)

## Go/No-Go Rule

- GO for open-source release as "judgement + explicit-ID compose".
- NO-GO if this repo advertises its own discovery/import/auth path (that belongs upstream).
