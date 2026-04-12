"""
agent_capcut.py
---------------
Machine-readable CLI shim for AI agents to use capcut-cli headlessly.
"""

from __future__ import annotations

import argparse
import json
import sys

from capcut_automation import (
    CapCutAutomation,
    CapCutAutomationError,
    CapCutCommandError,
    ComposeRequest,
    emit_json,
)


def _ok(payload: dict) -> int:
    print(emit_json({"ok": True, **payload}))
    return 0


def _err(code: str, message: str, details: dict | None = None, exit_code: int = 1) -> int:
    body = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    print(emit_json(body))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless capcut-cli adapter for agents")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Run dependency and auth checks")

    compose = sub.add_parser("compose", help="Run capcut-cli compose")
    compose.add_argument("--sound-id", default=None)
    compose.add_argument("--clip-id", action="append", default=None)
    compose.add_argument("--duration-seconds", type=int, default=30)
    compose.add_argument("--output-dir", default=".")
    compose.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()
    automation = CapCutAutomation()

    try:
        if args.command == "preflight":
            report = automation.preflight().to_dict()
            exit_code = 0 if report["capcut_cli_available"] else 70
            print(json.dumps({"ok": report["capcut_cli_available"], "preflight": report}))
            return exit_code

        if args.command == "compose":
            sound_id, clip_ids = automation.resolve_ids(args.sound_id, args.clip_id)
            req = ComposeRequest(
                sound_id=sound_id,
                clip_ids=clip_ids,
                duration_seconds=args.duration_seconds,
                output_dir=args.output_dir,
            )
            result = automation.compose(req, timeout=args.timeout)
            return _ok({"compose": result.to_dict()})

        return _err("invalid_command", "Unknown command", exit_code=2)

    except CapCutCommandError as exc:
        return _err(
            "capcut_command_error",
            str(exc),
            details={
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
            exit_code=3,
        )
    except CapCutAutomationError as exc:
        return _err("capcut_automation_error", str(exc), exit_code=3)
    except Exception as exc:  # noqa: BLE001
        return _err("unexpected_error", str(exc), exit_code=99)


if __name__ == "__main__":
    sys.exit(main())
