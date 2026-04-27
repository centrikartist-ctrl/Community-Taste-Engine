"""Run a human-facing trust pass for the new judgement contract."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
CANONICAL_CANDIDATES_PATH = EXAMPLES_DIR / "candidates.json"
CANONICAL_JUDGEMENTS_PATH = EXAMPLES_DIR / "judgements.json"
DEFAULT_WORK_ROOT = REPO_ROOT / ".judgement-temp"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_top_candidate_ids() -> list[str]:
    canonical_output = _load_json(CANONICAL_JUDGEMENTS_PATH)
    top_ids = canonical_output.get("summary", {}).get("top_candidate_ids", [])
    if len(top_ids) < 2:
        raise RuntimeError(
            f"canonical example output must define at least two top candidates: {CANONICAL_JUDGEMENTS_PATH}"
        )
    return top_ids[:2]


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


def run_trust_pass(report_dir: Path, *, work_root: Path = DEFAULT_WORK_ROOT) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)

    with _temporary_work_dir("judgement_contract_pass_", work_root) as tmp:
        tmp_dir = Path(tmp)
        output_path = tmp_dir / "judgements.json"

        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "judge.py"),
                str(CANONICAL_CANDIDATES_PATH),
                "--output",
                str(output_path),
                "--work-dir",
                str(tmp_dir / "work"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or "judge.py failed")

        payload = json.loads(output_path.read_text(encoding="utf-8"))

    ranked_ids = [item["candidate_id"] for item in payload["judgements"]]
    expected_top = _expected_top_candidate_ids()
    actual_top = ranked_ids[: len(expected_top)]
    if actual_top != expected_top:
        raise RuntimeError(f"trust pass failed: expected top 2 {expected_top}, got {actual_top}")

    report = {
        "ok": True,
        "generated_at_unix": int(time.time()),
        "source_example": {
            "candidates": "examples/candidates.json",
            "judgements": "examples/judgements.json",
        },
        "candidate_count": payload["summary"]["candidate_count"],
        "expected_top_candidate_ids": expected_top,
        "actual_top_candidate_ids": actual_top,
        "top_judgements": payload["judgements"][:2],
        "artifacts": {
            "report_json": "trust/judgement_pass_report.json",
            "report_md": "trust/judgement_pass_report.md",
        },
    }

    json_path = report_dir / "judgement_pass_report.json"
    md_path = report_dir / "judgement_pass_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Judgement Trust Pass\n\n"
        "This pass reruns the canonical checked-in example batch from examples/candidates.json\n"
        "and confirms the same top two shown in examples/judgements.json.\n\n"
        "## Result\n\n"
        f"- status: {'PASS' if report['ok'] else 'FAIL'}\n"
        f"- candidate count: {report['candidate_count']}\n"
        f"- source example: {report['source_example']['candidates']}\n"
        f"- expected top 2: {', '.join(report['expected_top_candidate_ids'])}\n"
        f"- actual top 2: {', '.join(report['actual_top_candidate_ids'])}\n\n"
        "## Re-run\n\n"
        "```bash\n"
        "python scripts/trust_judgement_pass.py\n"
        "```\n",
        encoding="utf-8",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the judgement contract trust pass")
    parser.add_argument("--report-dir", default="trust", help="Directory where trust report files are written")
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_WORK_ROOT),
        help="Directory where temporary trust-pass work directories are created",
    )
    args = parser.parse_args()

    report = run_trust_pass(Path(args.report_dir), work_root=Path(args.work_root))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
