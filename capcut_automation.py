"""
capcut_automation.py
-------------------
Agent-safe wrapper around capcut-cli with deterministic preflight,
structured errors, and compose execution helpers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Optional


class CapCutAutomationError(RuntimeError):
    """Base exception for capcut automation failures."""


class CapCutDependencyError(CapCutAutomationError):
    """Raised when capcut-cli is not available."""


class CapCutInputError(CapCutAutomationError):
    """Raised for missing or invalid compose inputs."""


class CapCutCommandError(CapCutAutomationError):
    """Raised when capcut-cli returns a non-zero status."""

    def __init__(self, message: str, *, returncode: int, stdout: str, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class PreflightReport:
    capcut_cli_available: bool
    ffmpeg_available: bool
    capcut_cli_version: Optional[str]
    auth_ok: bool
    issues: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComposeRequest:
    sound_id: str
    clip_ids: list[str]
    duration_seconds: int = 30
    output_dir: str = "."


@dataclass
class ComposeResult:
    command: list[str]
    cwd: str
    duration_seconds: int
    sound_id: str
    clip_ids: list[str]
    started_at: float
    finished_at: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class CapCutAutomation:
    def __init__(self, cli_path: Optional[str] = None):
        self.cli_path = cli_path or os.getenv("CAPCUT_CLI_PATH", "capcut-cli")

    def _exists(self) -> bool:
        if os.path.isabs(self.cli_path):
            return os.path.exists(self.cli_path)
        return shutil.which(self.cli_path) is not None

    def _run(
        self,
        args: list[str],
        *,
        timeout: int = 60,
        cwd: Optional[str] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        if not self._exists():
            raise CapCutDependencyError("capcut-cli not found in PATH")

        cmd = [self.cli_path, *args]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
        if check and proc.returncode != 0:
            raise CapCutCommandError(
                "capcut-cli command failed",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        return proc

    def preflight(self) -> PreflightReport:
        issues: list[dict] = []

        capcut_ok = self._exists()
        if not capcut_ok:
            issues.append({"code": "missing_capcut_cli", "severity": "error", "message": "capcut-cli not found"})

        ffmpeg_ok = shutil.which("ffmpeg") is not None
        if not ffmpeg_ok:
            issues.append({"code": "missing_ffmpeg", "severity": "warning", "message": "ffmpeg not found"})

        version = None
        auth_ok = False
        if capcut_ok:
            try:
                v = self._run(["--version"], timeout=15)
                version = (v.stdout or v.stderr).strip() or None
            except CapCutAutomationError as exc:
                issues.append({"code": "capcut_version_failed", "severity": "warning", "message": str(exc)})

            apify_token = os.getenv("CAPCUT_CLI_APIFY_TOKEN", "").strip()
            if apify_token:
                try:
                    self._run(["auth", "--from-env"], timeout=20)
                    auth_ok = True
                except CapCutCommandError as exc:
                    issues.append(
                        {
                            "code": "capcut_auth_check_failed",
                            "severity": "warning",
                            "message": f"auth check failed (exit {exc.returncode})",
                        }
                    )

        return PreflightReport(
            capcut_cli_available=capcut_ok,
            ffmpeg_available=ffmpeg_ok,
            capcut_cli_version=version,
            auth_ok=auth_ok,
            issues=issues,
        )

    @staticmethod
    def resolve_ids(
        sound_id: Optional[str] = None,
        clip_ids: Optional[list[str]] = None,
    ) -> tuple[str, list[str]]:
        sound = (sound_id or os.getenv("CAPCUT_SOUND_ID", "")).strip()

        resolved_clips: list[str] = []
        if clip_ids:
            resolved_clips = [c.strip() for c in clip_ids if c and c.strip()]
        if not resolved_clips:
            csv_ids = os.getenv("CAPCUT_CLIP_IDS", "").strip()
            if csv_ids:
                resolved_clips = [c.strip() for c in csv_ids.split(",") if c.strip()]
        if not resolved_clips:
            single = os.getenv("CAPCUT_CLIP_ID", "").strip()
            if single:
                resolved_clips = [single]

        if not sound:
            raise CapCutInputError("Missing sound_id. Provide argument or CAPCUT_SOUND_ID.")
        if not resolved_clips:
            raise CapCutInputError("Missing clip_ids. Provide arguments or CAPCUT_CLIP_ID(S).")

        return sound, resolved_clips

    def compose(self, request: ComposeRequest, *, timeout: int = 120) -> ComposeResult:
        if request.duration_seconds <= 0:
            raise CapCutInputError("duration_seconds must be positive")

        os.makedirs(request.output_dir, exist_ok=True)

        cmd = ["compose", "--sound", request.sound_id]
        for clip_id in request.clip_ids:
            cmd.extend(["--clip", clip_id])
        cmd.extend(["--duration-seconds", str(request.duration_seconds)])

        t0 = time.time()
        self._run(cmd, timeout=timeout, cwd=request.output_dir, check=True)
        t1 = time.time()

        return ComposeResult(
            command=[self.cli_path, *cmd],
            cwd=request.output_dir,
            duration_seconds=request.duration_seconds,
            sound_id=request.sound_id,
            clip_ids=request.clip_ids,
            started_at=round(t0, 3),
            finished_at=round(t1, 3),
            elapsed_seconds=round(t1 - t0, 3),
        )

    def compose_from_env(
        self,
        *,
        sound_id: Optional[str] = None,
        clip_ids: Optional[list[str]] = None,
        duration_seconds: int = 30,
        output_dir: str = ".",
        timeout: int = 120,
    ) -> ComposeResult:
        resolved_sound, resolved_clips = self.resolve_ids(sound_id=sound_id, clip_ids=clip_ids)
        req = ComposeRequest(
            sound_id=resolved_sound,
            clip_ids=resolved_clips,
            duration_seconds=duration_seconds,
            output_dir=output_dir,
        )
        return self.compose(req, timeout=timeout)


def emit_json(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
