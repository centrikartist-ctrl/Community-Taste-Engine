import sys
from pathlib import Path


def test_using_repo_venv_checks_active_prefix(monkeypatch):
    import scripts.bootstrap_verify as bootstrap

    monkeypatch.setattr(sys, "prefix", str(bootstrap.VENV_DIR))

    assert bootstrap._using_repo_venv() is True


def test_using_repo_venv_does_not_trust_executable_symlink(monkeypatch):
    import scripts.bootstrap_verify as bootstrap

    monkeypatch.setattr(sys, "prefix", str(bootstrap.REPO_ROOT / "not-the-repo-venv"))
    monkeypatch.setattr(sys, "executable", str(bootstrap._venv_python()))

    assert bootstrap._using_repo_venv() is False


def test_venv_python_path_is_inside_repo_venv():
    import scripts.bootstrap_verify as bootstrap

    path = Path(bootstrap._venv_python())

    assert bootstrap.VENV_DIR in path.parents
