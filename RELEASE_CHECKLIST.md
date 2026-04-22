# Release Checklist

Use this checklist before every public release.

## 1) Judgement contract

- [ ] `python -m pytest -q` passes.
- [ ] `python judge.py candidates.json --output judgements.json` is the main documented command.
- [ ] The README leads with the judgement-layer promise, not editing automation.
- [ ] Judgement examples clearly show `candidate_id`, `kind`, `score`, `status`, `reasons`, `risks`, and `recommended_action`.
- [ ] CLI flags and exit behavior are documented for `judge.py`.

## 2) Trust and ranking quality

- [ ] `python scripts/trust_judgement_pass.py` passes.
- [ ] The trust pass reruns the canonical checked-in five-candidate example and proves the same top 2 surface with reasons.
- [ ] The `trust/` artifacts match the current judgement contract wording and checked-in example story.
- [ ] `python scripts/evaluate_judgements.py` passes across all checked-in datasets.
- [ ] At least one checked-in dataset includes ambiguous `needs_work` or `unclear` cases.
- [ ] `python scripts/review_judgements.py ...` works against a checked-in feedback sample.

## 3) Media-module health

- [ ] `python scripts/trust_ugly_pass.py` still passes if the media module is being shipped.
- [ ] `ffmpeg` requirements for media scoring are documented.
- [ ] `pipeline.py` is described as a secondary media-analysis module, not the repo identity.

## 4) Repo focus

- [ ] CapCut-first or compose-first messaging does not lead any primary docs.
- [ ] Primary docs explicitly say this repo does not own manual-URL import/compose/render workflows.
- [ ] Stale planning material that conflicts with the community taste engine direction is removed.
- [ ] Unneeded downstream execution paths are removed outright.
- [ ] Discord export ingestion is documented and works against the checked-in example export.
- [ ] Discord export ingestion preserves obvious duplicate, thread, and author-history context.
- [ ] Local runtime folders like `.venv/`, `out/`, `outputs/`, and `smoke_output/` are ignored.

## Go/No-Go Rule

- GO when the repo clearly ships as a judgement layer that ranks what deserves attention.
- NO-GO if the public story drifts back toward "another video tool" or CapCut-first automation.
