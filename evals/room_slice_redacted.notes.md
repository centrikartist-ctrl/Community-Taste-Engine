# Room Slice Redacted

This evaluation batch was derived from a real Discord CSV export supplied locally during development.

What was kept:

- the ranking shape of the room slice
- the public categories the student asked for: brand-risk flag, repo review, community idea, price chatter, and vague hype
- enough semantic detail for the judgement layer to reason about those patterns without raw Discord context

What was removed:

- raw usernames
- raw full message bodies
- message IDs
- raw links
- direct quoting of room text beyond short generic framing in titles
- unnecessary personal or contextual detail

Why it is redacted:

- the CSV export did not include rich Discord metadata like channel threads or reliable reaction structure
- the repo does not need private room text committed to prove the taste engine works
- the public batch keeps only paraphrased text plus redacted `signals` metadata like `source`, `has_receipts`, `actionable`, and `risk_type`

Important limitation:

- the underlying room export did not hand us a perfect verbatim five-item lesson set, so the public categories are paraphrased from adjacent real-room messages rather than copied line-for-line
- the redacted batch is meant to encode the room's taste lessons in public, not preserve the room as evidence