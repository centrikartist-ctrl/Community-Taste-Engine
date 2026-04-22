# Community Taste Engine Plan

## Original Review

> Tally, yes, keep it. But I'd stop thinking of it as "another video tool" and make it the community taste engine.
>
> What you already have is not nothing: there's a Python pipeline with rhythm, speech-boundary, energy, pairing score, decision logs, weight feedback, tests, and a reproducible trust pass. That's a solid weird little machine. The problem is the positioning: right now it sounds like "agentic cut decisions for video editing," which puts it next to Niner's capcut-cli work. I'd move it one layer up.
>
> Make Community Taste Engine answer this:
>
> "Given a pile of candidate clips/posts/ideas, what deserves attention, and why?"
>
> Practical next direction:
>
> 1. Rename the core promise in the README.
>    Not "CapCut automation," more like:
>    "A judgement layer that scores candidate media and explains why it is worth using."
>
> 2. Add a simple input/output contract:
>    - input: `candidates.json`
>    - output: `judgements.json`
>
>    each judgement has score, reasons, risks, recommended_action
>
> Keep video scoring as one module.
> Your rhythm/speech/energy stuff becomes one kind of judgement, not the whole identity.
>
> Add a Discord/community mode.
> People submit ideas, links, memes, clips, or claims. The pipeline ranks them and says:
> strong signal
> needs work
> unclear
> probably noise
>
> Make the trust pass more human.
> Right now "ugly successful pass" proves the machinery runs. Next trust pass should prove: "I gave it 5 candidates and it correctly surfaced the best 2 with reasons."
>
> If I were you, I'd build one command next:
>
> `python judge.py candidates.json`
>
> And make the output dead simple. That becomes useful immediately in here. Niner can build the thing that makes clips. You build the thing that decides what deserves to become one.

## North Star

Community Taste Engine should rank candidate media, links, ideas, memes, and claims and explain why they deserve attention.

The core product question is:

"Given a pile of candidate clips/posts/ideas, what deserves attention, and why?"

## Scope

### In scope now

- Reposition the repo around judgement instead of editing automation.
- Make `python judge.py candidates.json` the main command.
- Standardize a small `candidates.json` -> `judgements.json` contract.
- Keep video analysis as one scoring module under the judgement layer.
- Add a community-mode framing for ideas, links, memes, clips, and claims.
- Add a human-facing trust pass that proves ranking quality on a small batch.

### Out of scope now

- Building a full Discord bot.
- Replacing or expanding Niner's CapCut CLI work.
- Building a remote service, database, or dashboard.
- Adding heavy new dependencies unless they directly support the judgement contract.

## Implementation Plan

### 1. Reframe the repo promise

Goal:
Change the language of the project so it reads as a judgement engine first and a video-analysis module second.

Work:

- Rewrite the README opening promise around judgement and attention ranking.
- Move CapCut/live execution details lower in the README or into a secondary section.
- Make the first visible command `python judge.py candidates.json`.
- Describe video scoring as one module inside the larger judgement system.

Success condition:
Someone reading the README should immediately understand that this repo decides what deserves attention, not how to mechanically edit a video.

### 2. Define the simple contract

Goal:
Make the repo useful with one dead-simple input and output shape.

Work:

- Accept `candidates.json` as input.
- Emit `judgements.json` as output.
- Ensure each judgement contains:
  - `score`
  - `reasons`
  - `risks`
  - `recommended_action`
- Keep the output small and machine-readable.

Success condition:
An upstream tool or human can hand the repo a candidate batch and receive ranked judgements without needing to understand the internal media pipeline.

### 3. Keep video scoring, but shrink its identity footprint

Goal:
Preserve the current rhythm, speech, energy, pairing, and logging system without letting it define the whole repo.

Work:

- Treat the media pipeline as a submodule used when a candidate is a local video/media item.
- Reuse existing scoring and logging where it helps the new contract.
- Avoid presenting cut generation or CapCut execution as the lead story.

Success condition:
The repo keeps its strongest technical machinery but presents it as one source of judgement rather than the entire product.

### 4. Add community mode

Goal:
Support community-submitted items without requiring media analysis for every candidate.

Work:

- Support candidates such as ideas, links, memes, clips, and claims.
- Add status buckets:
  - `strong_signal`
  - `needs_work`
  - `unclear`
  - `probably_noise`
- Accept lightweight structured signals for non-media candidates.
- Keep the framing centered on ranking and explanation.

Success condition:
The repo can rank a mixed batch of community submissions and explain the calls in human language.

### 5. Replace the main trust story

Goal:
Make the trust pass prove useful judgement, not just pipeline execution.

Work:

- Add a trust pass with 5 candidates.
- Assert that the best 2 candidates surface at the top with reasons.
- Keep the older ugly-success pass as a low-level systems check only if it still adds value.

Success condition:
The main trust artifact shows that the repo can make a sensible ranking decision that a human can inspect quickly.

### 6. Remove or de-emphasize what is not needed now

Goal:
Reduce positioning confusion and keep the repo focused on the judgement layer.

Remove or de-emphasize now:

- CapCut-first wording in primary docs.
- Any framing that makes the repo sound like it competes with Niner's CLI work.
- Agent-plan language that assumes live compose is the main outcome.
- README sections that foreground editing automation ahead of judgement.

Keep:

- `pipeline.py` as the deeper media-analysis module.
- The ugly-success trust pass as a secondary systems check.

Remove outright when they no longer serve the judgement layer:

- Downstream compose helpers that belong to Niner's side of the stack.
- Old CapCut-specific planning docs that no longer describe the product direction.
- README examples that lead with live compose instead of judgement.

Success condition:
The repo has one clear story, and anything outside that story is either secondary or removed.

## Order of Execution

1. Lock the wording and contract in docs.
2. Implement `judge.py` as the main entry point.
3. Reuse the current media machinery behind that new entry point.
4. Add community-mode scoring for non-media candidates.
5. Add the human-facing trust pass.
6. Remove or demote misleading CapCut-first messaging and stale planning material.

## Working Rule

When deciding between features, prefer the thing that makes this repo better at answering:

"What deserves attention, and why?"

If a component only helps with mechanical execution after that decision, it should stay secondary.