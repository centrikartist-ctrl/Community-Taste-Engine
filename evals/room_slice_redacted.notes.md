# Room Slice Redacted

This evaluation batch was derived from a real Discord CSV export supplied locally during development.

What was kept:

- the ranking shape of the room slice
- the candidate categories that actually appeared in the export
- enough semantic detail for the judgement layer to reason about receipts, tooling, spill-risk, price chatter, and vague hype

What was removed:

- raw usernames
- raw full message bodies
- direct quoting of room text beyond short generic framing in titles
- any unnecessary personal or contextual detail

Why it is redacted:

- the CSV export did not include rich Discord metadata like channel threads or reactions
- the repo does not need private room text committed to prove the taste engine works
- the student still needs to know that this is a real-room-derived batch, but not at the cost of publishing the room verbatim

Important limitation:

- this slice did not contain a clean community-idea post in the same quality as the tooling and receipts material
- it also did not contain a classic marketing-style brand-risk item, so the batch uses a disclosure/spill-risk discussion as the closest real-room substitute
- because the CSV export lacks richer Discord metadata like reactions, channel structure, and thread context, the strongest items in this batch currently grade as `needs_work` rather than `strong_signal`