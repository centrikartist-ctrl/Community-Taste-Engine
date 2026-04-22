# Judgement Trust Pass

This pass reruns the canonical checked-in example batch from examples/candidates.json
and confirms the same top two shown in examples/judgements.json.

## Result

- status: PASS
- candidate count: 5
- source example: examples/candidates.json
- expected top 2: claim_with_receipts, community_clip
- actual top 2: claim_with_receipts, community_clip

## Re-run

```bash
python scripts/trust_judgement_pass.py
```
