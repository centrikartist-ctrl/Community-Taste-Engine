"""Convert a Discord export into candidates.json for the judgement layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from discord_adapter import discord_export_to_candidates, load_discord_export, write_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Discord export into candidates.json")
    parser.add_argument("discord_export", help="Path to a Discord export JSON file")
    parser.add_argument("--output", default="candidates.json", help="Path to write candidates.json")
    parser.add_argument(
        "--redact-public",
        action="store_true",
        help="Strip source identifiers and links so the output is safer to commit as a public fixture",
    )
    args = parser.parse_args()

    payload = discord_export_to_candidates(load_discord_export(args.discord_export), redact_public=args.redact_public)
    write_candidates(args.output, payload)
    print(
        json.dumps(
            {
                "candidate_count": len(payload["candidates"]),
                "output": str(Path(args.output)),
                "redact_public": args.redact_public,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())