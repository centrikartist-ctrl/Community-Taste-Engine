"""Compare machine judgements against human review feedback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge import load_json_schema
from jsonschema import Draft202012Validator


def load_feedback(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(load_json_schema("judgement_feedback.schema.json"))
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(bit) for bit in first.absolute_path) or "<root>"
        raise ValueError(f"feedback file failed schema validation at {location}: {first.message}")
    return payload


def review_judgements(judgements_path: Path, feedback_path: Path) -> dict[str, Any]:
    judgements_payload = json.loads(judgements_path.read_text(encoding="utf-8"))
    feedback_payload = load_feedback(feedback_path)
    by_id = {item["candidate_id"]: item for item in judgements_payload["judgements"]}

    reviewed_items = []
    for item in feedback_payload["items"]:
        candidate_id = item["candidate_id"]
        if candidate_id not in by_id:
            raise ValueError(f"feedback references unknown candidate_id: {candidate_id}")

        machine = by_id[candidate_id]
        status_match = machine["status"] == item["human_status"]
        action_match = machine["recommended_action"] == item["human_recommended_action"]
        reviewed_items.append(
            {
                "candidate_id": candidate_id,
                "machine_status": machine["status"],
                "human_status": item["human_status"],
                "machine_action": machine["recommended_action"],
                "human_action": item["human_recommended_action"],
                "status_match": status_match,
                "action_match": action_match,
                "notes": item.get("notes"),
            }
        )

    agreement_rate = _rate([item["status_match"] and item["action_match"] for item in reviewed_items])
    status_agreement_rate = _rate([item["status_match"] for item in reviewed_items])
    action_agreement_rate = _rate([item["action_match"] for item in reviewed_items])

    return {
        "batch_name": feedback_payload.get("batch_name"),
        "reviewed_count": len(reviewed_items),
        "agreement_rate": round(agreement_rate, 4),
        "status_agreement_rate": round(status_agreement_rate, 4),
        "action_agreement_rate": round(action_agreement_rate, 4),
        "reviewed_items": reviewed_items,
    }


def _rate(values: list[bool]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare machine judgements against human review feedback")
    parser.add_argument("judgements", help="Path to judgements.json")
    parser.add_argument("feedback", help="Path to judgement feedback JSON")
    parser.add_argument("--output", default=None, help="Optional path to write the review report JSON")
    args = parser.parse_args()

    try:
        report = review_judgements(Path(args.judgements), Path(args.feedback))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())