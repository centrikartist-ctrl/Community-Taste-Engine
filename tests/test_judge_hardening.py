from __future__ import annotations

from judge import judge_candidates


def test_missing_local_video_path_becomes_failed_candidate_not_batch_error(tmp_path):
    payload = [
        {
            "id": "video_missing",
            "kind": "video",
            "path": str(tmp_path / "missing.mp4"),
            "title": "Missing local clip",
        },
        {
            "id": "idea_ok",
            "kind": "idea",
            "title": "Concrete idea",
            "text": "A concrete idea with receipts and a clear hook.",
            "signals": {
                "credibility": 0.9,
                "clarity": 0.85,
                "relevance": 0.8,
                "community_support": 0.75,
            },
        },
    ]

    result = judge_candidates(payload, work_dir=str(tmp_path / "work"))

    judgement_map = {item["candidate_id"]: item for item in result["judgements"]}
    assert judgement_map["video_missing"]["recommended_action"] == "repair_input"
    assert "Missing local video file" in judgement_map["video_missing"]["risks"][0]
    assert judgement_map["idea_ok"]["status"] in {"strong_signal", "needs_work"}
