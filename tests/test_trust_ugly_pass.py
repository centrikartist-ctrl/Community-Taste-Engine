import json
import shutil
import subprocess
import sys

import pytest


def test_ugly_success_pass_script_generates_report(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    report_dir = tmp_path / "trust"
    cmd = [
        sys.executable,
        "scripts/trust_ugly_pass.py",
        "--report-dir",
        str(report_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr
    report_path = report_dir / "ugly_success_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["pipeline"]["scored_count"] >= 1
    assert report["pipeline"]["pairing_min"] >= 0.0
