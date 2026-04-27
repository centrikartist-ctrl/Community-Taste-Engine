"""Run checked-in ranking evaluations for the judgement layer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge import judge_candidates, load_candidates_payload, validate_candidates_payload, validate_judgements_payload


EVALS_DIR = REPO_ROOT / "evals"
DEFAULT_WORK_ROOT = REPO_ROOT / ".judgement-temp"


def load_eval_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "candidate_batch", "expected_top_candidate_ids")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Evaluation manifest {path} is missing required fields: {', '.join(missing)}")
    return manifest


def evaluate_manifest(manifest_path: Path, *, work_root: Path) -> dict[str, Any]:
    manifest = load_eval_manifest(manifest_path)
    candidates_path = REPO_ROOT / manifest["candidate_batch"]
    payload = load_candidates_payload(str(candidates_path))
    validate_candidates_payload(payload)

    dataset_work = work_root / manifest_path.stem
    dataset_work.mkdir(parents=True, exist_ok=True)
    judged = judge_candidates(payload["candidates"], work_dir=str(dataset_work))
    validate_judgements_payload(judged)

    expected_top = list(manifest.get("expected_top_candidate_ids", []))
    actual_top = judged["summary"]["top_candidate_ids"][: len(expected_top)]
    positions = {item["candidate_id"]: index for index, item in enumerate(judged["judgements"])}

    expected_statuses = dict(manifest.get("expected_statuses", {}))
    status_checks = []
    for candidate_id, expected_status in expected_statuses.items():
        actual_status = next((item["status"] for item in judged["judgements"] if item["candidate_id"] == candidate_id), None)
        status_checks.append(
            {
                "candidate_id": candidate_id,
                "expected": expected_status,
                "actual": actual_status,
                "ok": actual_status == expected_status,
            }
        )

    pair_checks = []
    for pair in manifest.get("must_rank_above", []):
        higher = pair[0]
        lower = pair[1]
        ok = positions.get(higher, 10**9) < positions.get(lower, 10**9)
        pair_checks.append({"higher": higher, "lower": lower, "ok": ok})

    top_two_exact = actual_top == expected_top
    status_match_rate = _rate([item["ok"] for item in status_checks])
    pair_match_rate = _rate([item["ok"] for item in pair_checks])
    passed = top_two_exact and all(item["ok"] for item in status_checks) and all(item["ok"] for item in pair_checks)

    return {
        "name": manifest["name"],
        "description": manifest.get("description"),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "candidate_batch": manifest["candidate_batch"],
        "expected_top_candidate_ids": expected_top,
        "actual_top_candidate_ids": actual_top,
        "top_two_exact": top_two_exact,
        "status_match_rate": round(status_match_rate, 4),
        "pair_match_rate": round(pair_match_rate, 4),
        "status_checks": status_checks,
        "pair_checks": pair_checks,
        "passed": passed,
    }


class _ManagedWorkDir:
    def __init__(self, prefix: str, work_root: Path) -> None:
        self.work_root = work_root
        self.prefix = prefix
        self.path: str | None = None

    def __enter__(self) -> str:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.path = tempfile.mkdtemp(prefix=self.prefix, dir=str(self.work_root))
        return self.path

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


def _temporary_work_dir(prefix: str, work_root: Path) -> _ManagedWorkDir:
    return _ManagedWorkDir(prefix, work_root)


def evaluate_manifests(manifest_paths: list[Path], *, work_root: Path = DEFAULT_WORK_ROOT) -> dict[str, Any]:
    with _temporary_work_dir("judgement_evals_", work_root) as tmp:
        work_root = Path(tmp)
        reports = [evaluate_manifest(path, work_root=work_root) for path in manifest_paths]

    return {
        "dataset_count": len(reports),
        "passed_count": sum(1 for item in reports if item["passed"]),
        "top_two_exact_rate": round(_rate([item["top_two_exact"] for item in reports]), 4),
        "average_status_match_rate": round(_average(item["status_match_rate"] for item in reports), 4),
        "average_pair_match_rate": round(_average(item["pair_match_rate"] for item in reports), 4),
        "datasets": reports,
    }


def _rate(values: list[bool]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value) / len(values)


def _average(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 1.0


def _default_manifest_paths() -> list[Path]:
    return sorted(EVALS_DIR.glob("*.eval.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run checked-in judgement evaluations")
    parser.add_argument("manifests", nargs="*", help="Optional evaluation manifest paths")
    parser.add_argument("--output", default=None, help="Optional path to write the evaluation report JSON")
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_WORK_ROOT),
        help="Directory where temporary evaluation work directories are created",
    )
    args = parser.parse_args()

    manifest_paths = [REPO_ROOT / path for path in args.manifests] if args.manifests else _default_manifest_paths()
    if not manifest_paths:
        print("No evaluation manifests found.", file=sys.stderr)
        return 1

    report = evaluate_manifests(manifest_paths, work_root=Path(args.work_root))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["passed_count"] == report["dataset_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
