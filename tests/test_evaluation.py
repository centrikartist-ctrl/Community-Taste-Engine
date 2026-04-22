import json
import subprocess
import sys
from pathlib import Path


def test_evaluation_script_passes_checked_in_datasets(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "evaluation_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_judgements.py",
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset_count"] == 3
    assert report["passed_count"] == 3
    assert report["top_two_exact_rate"] == 1.0


def test_review_judgements_script_reports_human_overrides(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "review_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/review_judgements.py",
            "examples/judgements.json",
            "examples/judgement_feedback.json",
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reviewed_count"] == 5
    assert report["status_agreement_rate"] == 0.8
    assert report["action_agreement_rate"] == 0.8