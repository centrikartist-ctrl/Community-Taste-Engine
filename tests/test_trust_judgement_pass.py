import json
import subprocess
import sys
from pathlib import Path


def test_judgement_trust_pass_generates_report(tmp_path):
    report_dir = tmp_path / "trust"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/trust_judgement_pass.py",
            "--report-dir",
            str(report_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    report_path = report_dir / "judgement_pass_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source_example"] == {
        "candidates": "examples/candidates.json",
        "judgements": "examples/judgements.json",
    }
    assert report["candidate_count"] == 5
    assert report["actual_top_candidate_ids"] == ["claim_with_receipts", "community_clip"]


def test_checked_in_judgement_trust_report_matches_canonical_example():
    repo_root = Path(__file__).resolve().parents[1]
    report = json.loads((repo_root / "trust" / "judgement_pass_report.json").read_text(encoding="utf-8"))
    canonical_output = json.loads((repo_root / "examples" / "judgements.json").read_text(encoding="utf-8"))

    expected_top = canonical_output["summary"]["top_candidate_ids"][:2]
    assert report["expected_top_candidate_ids"] == expected_top
    assert report["actual_top_candidate_ids"] == expected_top
    assert report["top_judgements"][0]["candidate_id"] == expected_top[0]
    assert report["top_judgements"][1]["candidate_id"] == expected_top[1]