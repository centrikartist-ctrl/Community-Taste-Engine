import json
import subprocess
import sys
from pathlib import Path

from discord_adapter import discord_export_to_candidates, load_discord_export
from judge import judge_candidates


def test_discord_export_to_candidates_extracts_real_signals(tmp_path):
    export = {
        "messages": [
            {
                "id": "2001",
                "channel_name": "research-drops",
                "content": "Receipts here: source https://example.com/report with proof attached.",
                "author": {"display_name": "Tally", "roles": ["core-team"]},
                "attachments": [{"filename": "proof.png"}],
                "embeds": [{"url": "https://example.com/report"}],
                "reactions": [{"count": 5}],
                "reply_count": 3,
            },
            {
                "id": "2002",
                "channel_name": "research-drops",
                "content": "Same report, same claim: https://example.com/report",
                "author": {"display_name": "Another Researcher", "roles": ["member"]},
                "embeds": [{"url": "https://example.com/report"}],
                "reactions": [{"count": 2}],
                "reply_count": 1,
            }
        ]
    }
    export_path = tmp_path / "discord_export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    payload = discord_export_to_candidates(load_discord_export(str(export_path)))

    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["id"] == "discord_2001"
    assert candidate["source"]["platform"] == "discord"
    assert candidate["community"]["reaction_count"] == 7
    assert candidate["community"]["duplicate_count"] == 2
    assert candidate["community"]["unique_author_count"] == 2
    assert candidate["signals"]["community_support"] > 0.5
    assert candidate["signals"]["credibility"] > 0.6


def test_discord_candidates_rank_strong_messages_above_vague_ones(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    export_path = repo_root / "examples" / "discord_export.json"
    payload = discord_export_to_candidates(load_discord_export(str(export_path)))

    judged = judge_candidates(payload["candidates"], work_dir=str(tmp_path / "work"))
    assert judged["summary"]["top_candidate_ids"][:2] == ["discord_1001", "discord_1002"]
    assert judged["judgements"][0]["status"] == "strong_signal"


def test_discord_adapter_cli_writes_candidates_file(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    export_path = repo_root / "examples" / "discord_export.json"
    output_path = tmp_path / "candidates.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/discord_to_candidates.py",
            str(export_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidates"][0]["source"]["platform"] == "discord"


def test_discord_export_to_candidates_redacts_public_identifiers(tmp_path):
    export = {
        "messages": [
            {
                "id": "4001",
                "channel_name": "private-room",
                "thread_name": "sensitive-thread",
                "content": "Receipt thread https://example.com/private with a follow-up.",
                "jump_url": "https://discord.com/channels/private/4001",
                "timestamp": "2026-04-23T12:00:00Z",
                "author": {"display_name": "Private Person", "roles": ["member"]},
                "embeds": [{"url": "https://example.com/private"}],
                "reactions": [{"count": 2}],
                "reply_count": 1,
            }
        ]
    }
    export_path = tmp_path / "discord_export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    payload = discord_export_to_candidates(load_discord_export(str(export_path)), redact_public=True)

    candidate = payload["candidates"][0]
    assert candidate["source"] == {"platform": "discord", "redacted": True}
    assert candidate["signals"]["source"] == "discord_redacted"
    assert "url" not in candidate
    assert "Private Person" not in candidate["description"]
    assert "private-room" not in candidate["description"]


def test_discord_adapter_cli_redact_public_writes_redacted_candidates(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    export_path = repo_root / "examples" / "discord_export.json"
    output_path = tmp_path / "candidates.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/discord_to_candidates.py",
            str(export_path),
            "--output",
            str(output_path),
            "--redact-public",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]
    assert candidate["source"] == {"platform": "discord", "redacted": True}
    assert candidate["signals"]["source"] == "discord_redacted"
    assert "url" not in candidate


def test_discord_html_export_loads_messages_and_can_redact_publicly(tmp_path):
    export_path = tmp_path / "discord_export.html"
    export_path.write_text(
        """
        <!DOCTYPE html>
        <html>
          <body>
            <ul class="chatContent">
              <li>
                <div class="titleInfo">
                  <p class="timeInfo"><span class="chatName">Private Person</span><span class="time">Sat Apr 25 2026 08:00:00 GMT+0100</span></p>
                  <p><span style="color: skyblue;"></span> Bring the artifact, not the trailer. One concrete repo note beats vague hype.</p>
                </div>
              </li>
              <li>
                <div class="titleInfo">
                  <p class="timeInfo"><span class="chatName">Another Member</span><span class="time">Sat Apr 25 2026 08:01:00 GMT+0100</span></p>
                  <p><span style="color: skyblue;"></span> Price soon?</p>
                </div>
              </li>
            </ul>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    messages = load_discord_export(str(export_path))
    payload = discord_export_to_candidates(messages, redact_public=True)

    assert len(messages) == 2
    assert messages[0]["author"]["display_name"] == "Private Person"
    assert payload["candidates"][0]["source"] == {"platform": "discord", "redacted": True}
    assert "Private Person" not in payload["candidates"][0]["description"]


def test_discord_adapter_keeps_thread_context_and_penalises_attachment_only_noise(tmp_path):
    export = {
        "messages": [
            {
                "id": "3001",
                "channel_name": "clips",
                "thread_name": "payoff-lab",
                "content": "Clip idea with a clear payoff and direct quotes in the replies.",
                "author": {"display_name": "Clip Poster", "roles": ["member"]},
                "attachments": [{"filename": "clip.mp4"}],
                "reactions": [{"count": 3}],
                "reply_count": 2,
                "referenced_message": {"id": "2999", "content": "People keep quoting the ending."}
            },
            {
                "id": "3002",
                "channel_name": "memes",
                "content": "",
                "author": {"display_name": "Meme Poster", "roles": ["member"]},
                "attachments": [{"filename": "reaction.png"}],
                "reactions": [{"count": 1}],
                "reply_count": 0
            }
        ]
    }

    export_path = tmp_path / "discord_export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    payload = discord_export_to_candidates(load_discord_export(str(export_path)))
    clip_candidate = next(item for item in payload["candidates"] if item["id"] == "discord_3001")
    meme_candidate = next(item for item in payload["candidates"] if item["id"] == "discord_3002")

    assert clip_candidate["source"]["thread_name"] == "payoff-lab"
    assert "Reply context" in clip_candidate["description"]
    assert meme_candidate["signals"]["uncertainty"] > 0.45
    assert meme_candidate["signals"]["relevance"] < 0.4
