from __future__ import annotations

from scripts.evaluate_judgements import REPO_ROOT, evaluate_manifests
from scripts.trust_judgement_pass import run_trust_pass


def test_run_trust_pass_uses_explicit_work_root(tmp_path):
    report_dir = tmp_path / "report"
    work_root = tmp_path / "work-root"

    report = run_trust_pass(report_dir, work_root=work_root)

    assert report["ok"] is True
    assert (report_dir / "judgement_pass_report.json").exists()
    assert work_root.exists()


def test_evaluate_manifests_uses_explicit_work_root(tmp_path):
    manifest = REPO_ROOT / "evals" / "canonical.eval.json"
    work_root = tmp_path / "eval-work-root"

    report = evaluate_manifests([manifest], work_root=work_root)

    assert report["dataset_count"] == 1
    assert report["passed_count"] == 1
    assert work_root.exists()
