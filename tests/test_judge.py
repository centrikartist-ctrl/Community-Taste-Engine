import json
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import validate

from judge import judge_candidates


def test_importing_judge_does_not_import_pipeline(monkeypatch):
    sys.modules.pop("judge", None)
    sys.modules.pop("pipeline", None)

    imported = importlib.import_module("judge")

    assert imported is not None
    assert "pipeline" not in sys.modules


def test_judge_candidates_ranks_and_emits_contract(tmp_path):
    candidates = [
        {
            "id": "noise_meme",
            "kind": "meme",
            "title": "Out-of-context meme",
            "text": "Funny, but disconnected from the community theme.",
            "signals": {
                "novelty": 0.2,
                "clarity": 0.2,
                "uncertainty": 0.9,
            },
        },
        {
            "id": "strong_claim",
            "kind": "claim",
            "title": "Claim with receipts",
            "text": "A concrete claim with sources, momentum, and clear upside.",
            "signals": {
                "novelty": 0.8,
                "credibility": 0.95,
                "clarity": 0.9,
                "community_support": 0.8,
            },
        },
        {
            "id": "rough_idea",
            "kind": "idea",
            "title": "Promising but undercooked",
            "text": "Interesting angle, but still missing concrete supporting detail.",
            "signals": {
                "novelty": 0.65,
                "clarity": 0.45,
                "uncertainty": 0.4,
            },
        },
    ]

    payload = judge_candidates(candidates, work_dir=str(tmp_path / "judge-work"))

    assert payload["summary"]["candidate_count"] == 3
    assert payload["summary"]["top_candidate_ids"][0] == "strong_claim"

    top = payload["judgements"][0]
    bottom = payload["judgements"][-1]
    assert top["candidate_id"] == "strong_claim"
    assert top["status"] == "strong_signal"
    assert top["recommended_action"] == "feature"
    assert top["reasons"]
    assert top["risks"]

    assert bottom["candidate_id"] == "noise_meme"
    assert bottom["status"] == "probably_noise"


def test_judge_candidates_emit_more_concrete_taste_language(tmp_path):
    candidates = [
        {
            "id": "brand_risk_flag",
            "kind": "claim",
            "title": "Brand-risk flag: the borrowed frame will read as copycat",
            "text": "We have receipts, screenshots, and source links showing the current framing borrows a bad external frame.",
            "description": "Discord-style note from #brand-risk with concrete examples.",
            "source": {
                "platform": "discord",
                "channel_name": "brand-risk",
                "external_urls": ["https://example.com/thread"],
            },
            "community": {
                "reaction_count": 6,
                "reply_count": 4,
                "attachment_count": 1,
                "link_count": 1,
                "trusted_submitter": True,
            },
        },
        {
            "id": "tooling_review",
            "kind": "idea",
            "title": "Tooling review: lazy-import pipeline so pure judgement runs stay clean",
            "text": "This is actionable today: the PR isolates media imports, keeps non-video batches off NumPy and ffmpeg, and includes a focused runtime test.",
            "description": "Discord-style review thread from #tooling-review.",
            "source": {
                "platform": "discord",
                "channel_name": "tooling-review",
                "external_urls": ["https://example.com/pr/17"],
            },
            "community": {
                "reaction_count": 5,
                "reply_count": 4,
                "attachment_count": 0,
                "link_count": 1,
                "trusted_submitter": True,
            },
        },
        {
            "id": "price_chatter",
            "kind": "meme",
            "title": "Price is ripping again",
            "text": "Green candles everywhere, moon soon, bullish again. No doc, no repo, no clip, just price energy.",
            "description": "Discord-style price chatter from #price-chat.",
            "source": {
                "platform": "discord",
                "channel_name": "price-chat",
                "external_urls": [],
            },
            "community": {
                "reaction_count": 7,
                "reply_count": 1,
                "attachment_count": 0,
                "link_count": 0,
                "trusted_submitter": False,
            },
        },
    ]

    payload = judge_candidates(candidates, work_dir=str(tmp_path / "judge-work"))
    by_id = {item["candidate_id"]: item for item in payload["judgements"]}

    assert "Strong because it has receipts and a clean hook." in by_id["brand_risk_flag"]["reasons"]
    assert "Useful because builders can act on it today." in by_id["tooling_review"]["reasons"]
    assert "Noise because it only has price energy, no artifact path." in by_id["price_chatter"]["risks"]
    assert by_id["price_chatter"]["status"] == "probably_noise"


def test_redacted_signal_metadata_drives_room_style_reasons(tmp_path):
    payload = judge_candidates(
        [
            {
                "id": "room_brand_risk_flag",
                "kind": "community_signal",
                "title": "Brand-risk flag with clear cleanup action",
                "text": "A community member flags that a public spill created reputational risk and asks for cleanup rather than debate.",
                "signals": {
                    "source": "discord_redacted",
                    "has_receipts": True,
                    "actionable": True,
                    "risk_type": "brand_frame",
                    "clarity": 0.78,
                    "credibility": 0.82,
                    "source_quality": 0.76,
                    "relevance": 0.88,
                    "uncertainty": 0.18,
                },
            },
            {
                "id": "room_price_chatter",
                "kind": "community_signal",
                "title": "Price prompt without any artifact path",
                "text": "A member asks whether price moves soon, with no document or implementation detail.",
                "signals": {
                    "source": "discord_redacted",
                    "no_artifact_path": True,
                    "price_chatter": True,
                    "risk_type": "price_chatter",
                    "relevance": 0.18,
                    "uncertainty": 0.82,
                    "credibility": 0.08,
                    "source_quality": 0.0,
                },
            },
        ],
        work_dir=str(tmp_path / "judge-work"),
    )

    by_id = {item["candidate_id"]: item for item in payload["judgements"]}
    assert "Strong because it has receipts and a clean hook." in by_id["room_brand_risk_flag"]["reasons"]
    assert "Important because it carries brand risk and needs a clear call." in by_id["room_brand_risk_flag"]["reasons"]
    assert "Risky because it surfaces brand damage and needs careful framing." in by_id["room_brand_risk_flag"]["risks"]
    assert "Noise because it only has price energy, no artifact path." in by_id["room_price_chatter"]["risks"]
    assert by_id["room_price_chatter"]["status"] == "probably_noise"
    assert all("No artifact path is strong." != reason for reason in by_id["room_price_chatter"]["reasons"])
    assert all("Actionable is strong." != reason for reason in by_id["room_brand_risk_flag"]["reasons"])


def test_reaction_only_candidate_gets_non_flattering_reason(tmp_path):
    payload = judge_candidates(
        [
            {
                "id": "pure_emoji_reaction",
                "kind": "community_signal",
                "title": "Pure emoji reaction without context",
                "text": "A member posts only a custom emoji reaction with no explanation, artifact path, or next step.",
                "signals": {
                    "source": "discord_redacted",
                    "no_artifact_path": True,
                    "risk_type": "reaction_only",
                    "clarity": 0.12,
                    "credibility": 0.08,
                    "source_quality": 0.0,
                    "relevance": 0.2,
                    "community_support": 0.08,
                    "uncertainty": 0.86,
                },
            }
        ],
        work_dir=str(tmp_path / "judge-work"),
    )

    judgement = payload["judgements"][0]
    assert judgement["status"] == "probably_noise"
    assert judgement["reasons"][0] == "Low-context reaction, useful as mood but not direction."
    assert "The hook is easy to understand." not in judgement["reasons"]
    assert "Low-context reaction only; it should not drive direction by itself." in judgement["risks"]


def test_judge_cli_writes_output_file(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "nested" / "reports" / "judgements.json"
    summary_path = tmp_path / "nested" / "reports" / "summary.json"
    candidates_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate_1",
                        "title": "Useful link",
                        "text": "A link with clear context and strong community response.",
                        "signals": {
                            "clarity": 0.9,
                            "community_support": 0.75,
                            "credibility": 0.8,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(candidates_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path.parent if (tmp_path.parent / "judge.py").exists() else "."),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["judgements"][0]["candidate_id"] == "candidate_1"
    assert "score" in payload["judgements"][0]
    assert "recommended_action" in payload["judgements"][0]
    assert summary["candidate_count"] == 1


def test_judge_cli_rejects_contract_invalid_top_level_payload(tmp_path):
    bad_contract_path = tmp_path / "bad_contract.json"
    bad_contract_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate_1",
                        "text": "Valid candidate body.",
                    }
                ],
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(bad_contract_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )

    assert proc.returncode == 1
    assert "failed schema validation" in proc.stderr


def test_judge_cli_can_skip_schema_validation_for_extended_top_level_payload(tmp_path):
    extended_payload_path = tmp_path / "extended_candidates.json"
    output_path = tmp_path / "judgements.json"
    extended_payload_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate_1",
                        "text": "Valid candidate body.",
                        "signals": {"clarity": 0.7},
                    }
                ],
                "unexpected": {"notes": "local extension"},
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(extended_payload_path),
            "--skip-schema-validation",
            "--quiet",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1


def test_checked_in_example_still_produces_expected_top_two(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    candidates_path = repo_root / "examples" / "candidates.json"
    output_path = tmp_path / "judgements.json"

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(candidates_path),
            "--output",
            str(output_path),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 5
    assert payload["summary"]["top_candidate_ids"][:2] == [
        "claim_with_receipts",
        "community_clip",
    ]

    top = payload["judgements"][0]
    assert top["candidate_id"] == "claim_with_receipts"
    assert top["status"] == "strong_signal"
    assert set(("score", "reasons", "risks", "recommended_action")).issubset(top)


def test_judge_cli_reports_invalid_json_cleanly(tmp_path):
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{bad json}", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(bad_json_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )

    assert proc.returncode == 1
    assert "invalid JSON" in proc.stderr


def test_judge_cli_reports_missing_video_path_as_repairable_candidate(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "reports" / "judgements.json"
    candidates_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "broken_video",
                        "kind": "clip",
                        "path": "missing.mp4",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(candidates_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    judgement = payload["judgements"][0]
    assert judgement["candidate_id"] == "broken_video"
    assert judgement["recommended_action"] == "repair_input"
    assert "Missing local video file" in judgement["risks"][0]


def test_judge_candidates_degrades_gracefully_when_video_processing_fails(tmp_path, monkeypatch):
    import judge
    import types

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video")

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("media processing blew up")

    class _FakeLogger:
        def __init__(self, _path: str):
            self._path = _path

        def load(self):
            return []

    fake_pipeline = types.SimpleNamespace(Logger=_FakeLogger, run=_raise_runtime_error)

    original_import_module = judge.importlib.import_module

    def _fake_import_module(name: str):
        if name == "pipeline":
            return fake_pipeline
        return original_import_module(name)

    monkeypatch.setattr(judge.importlib, "import_module", _fake_import_module)

    payload = judge_candidates(
        [
            {
                "id": "video_candidate",
                "kind": "clip",
                "path": str(video_path),
            }
        ],
        work_dir=str(tmp_path / "judge-work"),
    )

    judgement = payload["judgements"][0]
    assert judgement["candidate_id"] == "video_candidate"
    assert judgement["status"] == "probably_noise"
    assert judgement["recommended_action"] == "pass"
    assert "Media analysis could not complete" in judgement["reasons"][0]
    assert "RuntimeError" in judgement["risks"][0]


def test_checked_in_examples_conform_to_schemas(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    candidates_schema = json.loads((repo_root / "schemas" / "candidates.schema.json").read_text(encoding="utf-8"))
    judgements_schema = json.loads((repo_root / "schemas" / "judgements.schema.json").read_text(encoding="utf-8"))
    candidates_payload = json.loads((repo_root / "examples" / "candidates.json").read_text(encoding="utf-8"))

    validate(instance=candidates_payload, schema=candidates_schema)

    output_path = tmp_path / "judgements.json"
    proc = subprocess.run(
        [
            sys.executable,
            "judge.py",
            str(repo_root / "examples" / "candidates.json"),
            "--output",
            str(output_path),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    judgements_payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate(instance=judgements_payload, schema=judgements_schema)
