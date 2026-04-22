import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import validate

from judge import judge_candidates


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


def test_judge_cli_writes_output_file(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "judgements.json"
    summary_path = tmp_path / "summary.json"
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


def test_judge_cli_reports_missing_video_path_cleanly(tmp_path):
    candidates_path = tmp_path / "candidates.json"
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
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )

    assert proc.returncode == 1
    assert "Video path does not exist" in proc.stderr


def test_judge_candidates_degrades_gracefully_when_video_processing_fails(tmp_path, monkeypatch):
    import judge

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video")

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("media processing blew up")

    monkeypatch.setattr(judge, "run", _raise_runtime_error)

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