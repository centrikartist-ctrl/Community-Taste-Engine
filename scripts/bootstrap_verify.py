"""Bootstrap a fresh checkout and run the full verification stack."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _using_repo_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == _venv_python().resolve()
    except OSError:
        return False


def _run(cmd: list[str], *, label: str, timeout: int = 300) -> None:
    print(f"\n==> {label}", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _bootstrap_venv(args: argparse.Namespace) -> int | None:
    if args.no_venv or _using_repo_venv():
        return None

    if not _venv_python().exists():
        _run([sys.executable, "-m", "venv", str(VENV_DIR)], label="Create .venv", timeout=180)

    cmd = [str(_venv_python()), str(Path(__file__).resolve()), "--no-venv"]
    if args.no_install:
        cmd.append("--no-install")
    if args.skip_media:
        cmd.append("--skip-media")
    if args.report_dir != "trust":
        cmd.extend(["--report-dir", args.report_dir])

    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _install_requirements() -> None:
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], label="Upgrade pip", timeout=300)
    _run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        label="Install Python requirements",
        timeout=900,
    )


def _check_ffmpeg(skip_media: bool) -> None:
    if skip_media:
        return
    if shutil.which("ffmpeg") is not None:
        return

    message = (
        "\nffmpeg is required for the media trust pass.\n"
        "Install ffmpeg, then rerun this command. If you only want the judgement-layer checks, "
        "rerun with --skip-media.\n"
    )
    raise SystemExit(message)


def run_verification(args: argparse.Namespace) -> None:
    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9+ is required for the judgement verification stack.")

    if not args.no_install:
        _install_requirements()

    _check_ffmpeg(args.skip_media)

    _run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "sys.modules.pop('judge', None); "
                "sys.modules.pop('pipeline', None); "
                "importlib.import_module('judge'); "
                "assert 'pipeline' not in sys.modules, 'judge import should not load pipeline'"
            ),
        ],
        label="Verify judge import stays decoupled from media pipeline",
    )
    _run(
        [
            sys.executable,
            "judge.py",
            "examples/candidates.json",
            "--output",
            "out/bootstrap-judgements.json",
            "--work-dir",
            ".judgement-ci",
            "--quiet",
        ],
        label="Run public judgement command",
    )
    _run([sys.executable, "scripts/evaluate_judgements.py"], label="Run checked-in evaluation suite")
    _run(
        [sys.executable, "scripts/trust_judgement_pass.py", "--report-dir", args.report_dir],
        label="Run judgement trust pass",
    )
    if not args.skip_media:
        _run(
            [sys.executable, "scripts/trust_ugly_pass.py", "--report-dir", args.report_dir],
            label="Run media trust pass",
            timeout=300,
        )
    _run([sys.executable, "-m", "pytest", "-q"], label="Run test suite", timeout=600)

    print("\nFull verification stack passed.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local Python environment, install dependencies, and verify the repo."
    )
    parser.add_argument("--no-venv", action="store_true", help="Use the current Python instead of .venv")
    parser.add_argument("--no-install", action="store_true", help="Skip pip installation")
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Skip ffmpeg-dependent media trust checks",
    )
    parser.add_argument("--report-dir", default="trust", help="Directory where trust reports are written")
    args = parser.parse_args()

    delegated = _bootstrap_venv(args)
    if delegated is not None:
        return delegated

    run_verification(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
