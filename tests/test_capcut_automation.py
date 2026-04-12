import pytest

from capcut_automation import (
    CapCutAutomation,
    CapCutDependencyError,
    CapCutInputError,
    ComposeRequest,
)


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_preflight_detects_missing_cli(monkeypatch):
    monkeypatch.setattr("capcut_automation.shutil.which", lambda _name: None)
    automation = CapCutAutomation()
    report = automation.preflight()
    assert report.capcut_cli_available is False
    assert any(i["code"] == "missing_capcut_cli" for i in report.issues)


def test_preflight_without_auth_token_is_not_warning(monkeypatch):
    def _exists(self):
        return True

    def _run(self, args, timeout=60, cwd=None, check=True):
        if args == ["--version"]:
            return _Proc(returncode=0, stdout="capcut-cli 0.1.0", stderr="")
        raise AssertionError("unexpected command")

    monkeypatch.delenv("CAPCUT_CLI_APIFY_TOKEN", raising=False)
    monkeypatch.setattr("capcut_automation.shutil.which", lambda name: "C:/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr("capcut_automation.CapCutAutomation._exists", _exists)
    monkeypatch.setattr("capcut_automation.CapCutAutomation._run", _run)

    report = CapCutAutomation().preflight()
    assert report.capcut_cli_available is True
    assert report.auth_ok is False
    assert not any(i["code"] == "capcut_auth_token_missing" for i in report.issues)


def test_resolve_ids_from_arguments():
    sound, clips = CapCutAutomation.resolve_ids("sound_1", ["clip_a", "clip_b"])
    assert sound == "sound_1"
    assert clips == ["clip_a", "clip_b"]


def test_resolve_ids_from_env(monkeypatch):
    monkeypatch.setenv("CAPCUT_SOUND_ID", "sound_env")
    monkeypatch.setenv("CAPCUT_CLIP_IDS", "clip_1, clip_2")
    sound, clips = CapCutAutomation.resolve_ids(None, None)
    assert sound == "sound_env"
    assert clips == ["clip_1", "clip_2"]


def test_resolve_ids_missing_raises(monkeypatch):
    monkeypatch.delenv("CAPCUT_SOUND_ID", raising=False)
    monkeypatch.delenv("CAPCUT_CLIP_ID", raising=False)
    monkeypatch.delenv("CAPCUT_CLIP_IDS", raising=False)
    with pytest.raises(CapCutInputError):
        CapCutAutomation.resolve_ids(None, None)


def test_compose_builds_command(monkeypatch):
    calls = []

    def _exists(self):
        return True

    def _run(self, args, timeout=60, cwd=None, check=True):
        calls.append((args, timeout, cwd, check))
        return _Proc(returncode=0)

    monkeypatch.setattr("capcut_automation.CapCutAutomation._exists", _exists)
    monkeypatch.setattr("capcut_automation.CapCutAutomation._run", _run)

    automation = CapCutAutomation()
    req = ComposeRequest(
        sound_id="sound_1",
        clip_ids=["clip_a", "clip_b"],
        duration_seconds=18,
        output_dir="out",
    )
    result = automation.compose(req)

    assert result.sound_id == "sound_1"
    assert len(calls) == 1
    assert calls[0][0][:3] == ["compose", "--sound", "sound_1"]
    assert calls[0][2] == "out"


def test_compose_rejects_non_positive_duration():
    automation = CapCutAutomation()
    req = ComposeRequest(sound_id="sound_1", clip_ids=["clip_a"], duration_seconds=0, output_dir=".")
    with pytest.raises(CapCutInputError):
        automation.compose(req)
