"""Rank candidate media and ideas into simple, explained judgements."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
NEGATIVE_SIGNAL_TOKENS = (
    "risk",
    "uncertain",
    "tox",
    "flag",
    "dependency",
    "cost",
    "legal",
    "effort",
)
ENGAGEMENT_SIGNAL_TOKENS = (
    "vote",
    "like",
    "share",
    "comment",
    "save",
    "reaction",
    "engagement",
)
COMMUNITY_TEXT_SIGNAL_TOKENS = (
    "quote",
    "thread",
    "source",
    "proof",
    "receipts",
    "claim",
    "hook",
    "idea",
)
HEDGE_TOKENS = ("maybe", "might", "idk", "not sure", "unclear", "perhaps", "guess")
RECEIPT_TOKENS = (
    "receipt",
    "receipts",
    "source link",
    "source links",
    "proof",
    "evidence",
    "report",
    "reports",
    "screenshot",
    "screenshots",
    "docs",
    "data",
    "bench",
    "benchmark",
    "trace",
    "log",
    "logs",
)
ACTIONABLE_TOKENS = (
    "actionable",
    "act on it today",
    "builder",
    "builders",
    "build",
    "fix",
    "review",
    "merge",
    "ship",
    "tool",
    "tooling",
    "repo",
    "pull request",
    "prototype",
    "deploy",
    "test",
    "runtime",
)
BRAND_RISK_TOKENS = (
    "brand risk",
    "reputational",
    "backlash",
    "off-brand",
    "off brand",
    "trust hit",
    "copycat",
)
BAD_FRAME_TOKENS = (
    "bad external frame",
    "borrowed frame",
    "wrong frame",
    "copycat",
)
PRICE_CHATTER_TOKENS = (
    "price",
    "prices",
    "chart",
    "charts",
    "candle",
    "candles",
    "moon",
    "bullish",
    "breakout",
    "ath",
)
HYPE_TOKENS = (
    "huge",
    "massive",
    "soon",
    "alpha",
    "big announcement",
    "game changer",
    "exploding",
)
REACTION_ONLY_TOKENS = (
    "emoji reaction",
    "custom emoji",
    "reaction only",
    "reaction-only",
    "emoji only",
    "only emoji",
    "standalone reaction",
)
ARTIFACT_TOKENS = (
    "repo",
    "pull request",
    "prototype",
    "demo",
    "doc",
    "docs",
    "screenshot",
    "report",
    "clip",
    "artifact",
    "diff",
    "commit",
)
NEGATED_ARTIFACT_TOKENS = (
    "no doc",
    "no docs",
    "no repo",
    "no clip",
    "no artifact",
    "no prototype",
    "no screenshot",
    "no receipt",
    "no receipts",
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass
class Judgement:
    candidate_id: str
    kind: str
    score: float
    status: str
    reasons: list[str]
    risks: list[str]
    recommended_action: str
    title: str | None = None


def load_candidates_payload(path: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"candidates.json not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidates.json has invalid JSON: {exc}") from exc

    if isinstance(data, list):
        payload = {"candidates": data}
    elif isinstance(data, dict) and isinstance(data.get("candidates"), list):
        payload = data
    else:
        raise ValueError("candidates.json must be a list or an object with a 'candidates' list")

    if not payload["candidates"]:
        raise ValueError("candidates.json must contain at least one candidate")
    return payload


def load_json_schema(name: str) -> dict[str, Any]:
    schema_path = SCHEMAS_DIR / name
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Schema file not found: {schema_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema file is invalid JSON: {schema_path}: {exc}") from exc


def validate_candidates_payload(payload: dict[str, Any]) -> None:
    _validate_payload(payload, load_json_schema("candidates.schema.json"), "candidates.json")


def validate_judgements_payload(payload: dict[str, Any]) -> None:
    _validate_payload(payload, load_json_schema("judgements.schema.json"), "judgements.json")


def _validate_payload(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if not errors:
        return

    first = errors[0]
    path_bits = [str(bit) for bit in first.absolute_path]
    location = ".".join(path_bits) if path_bits else "<root>"
    raise ValueError(f"{label} failed schema validation at {location}: {first.message}")


def load_candidates(path: str) -> list[dict[str, Any]]:
    return load_candidates_payload(path)["candidates"]


def write_judgements(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def judge_candidates(
    candidates: list[dict[str, Any]],
    *,
    work_dir: str = ".judgement",
    min_confidence: float = 0.35,
) -> dict[str, Any]:
    work_root = Path(work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    judgements: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("id") or f"candidate_{index}")
        kind = str(candidate.get("kind") or _infer_kind(candidate))

        if _is_local_video_candidate(candidate):
            judgement = _judge_video_candidate(
                candidate_id,
                kind,
                candidate,
                work_root=work_root,
                min_confidence=min_confidence,
            )
        else:
            judgement = _judge_generic_candidate(candidate_id, kind, candidate)

        judgements.append(asdict(judgement))

    judgements.sort(key=lambda item: item["score"], reverse=True)
    top_ids = [item["candidate_id"] for item in judgements[:2]]
    status_counts: dict[str, int] = {}
    for item in judgements:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    return {
        "generated_at_unix": int(time.time()),
        "summary": {
            "candidate_count": len(candidates),
            "top_candidate_ids": top_ids,
            "status_counts": status_counts,
        },
        "judgements": judgements,
    }


def _infer_kind(candidate: dict[str, Any]) -> str:
    path_value = candidate.get("path")
    if isinstance(path_value, str) and Path(path_value).suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    return "idea"


def _is_local_video_candidate(candidate: dict[str, Any]) -> bool:
    path_value = candidate.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    candidate_path = Path(path_value)
    if candidate_path.suffix.lower() in VIDEO_SUFFIXES and not candidate_path.exists():
        raise ValueError(f"Video path does not exist: {path_value}")
    return candidate_path.exists() and candidate_path.suffix.lower() in VIDEO_SUFFIXES


def _judge_video_candidate(
    candidate_id: str,
    kind: str,
    candidate: dict[str, Any],
    *,
    work_root: Path,
    min_confidence: float,
) -> Judgement:
    candidate_path = Path(str(candidate["path"]))
    pipeline_module = importlib.import_module("pipeline")
    logger_class = pipeline_module.Logger
    run_pipeline = pipeline_module.run

    try:
        with tempfile.TemporaryDirectory(prefix=f"judge_{candidate_id}_", dir=str(work_root)) as tmp:
            tmp_dir = Path(tmp)
            log_path = tmp_dir / "decisions.jsonl"
            decisions = run_pipeline(
                str(candidate_path),
                log_path=str(log_path),
                min_confidence=min_confidence,
            )
            records = logger_class(str(log_path)).load()
    except Exception as exc:
        return Judgement(
            candidate_id=candidate_id,
            kind=kind,
            score=0.15,
            status=_status_for_score(0.15),
            reasons=["Media analysis could not complete for this candidate."],
            risks=[f"Video processing failed: {type(exc).__name__}."],
            recommended_action=_action_for_score(0.15),
            title=_candidate_title(candidate),
        )

    scored = [record for record in records if record.get("score") and record.get("decision")]
    if not scored:
        return Judgement(
            candidate_id=candidate_id,
            kind=kind,
            score=0.18,
            status=_status_for_score(0.18),
            reasons=["Media analysis found no strong moments worth elevating."],
            risks=["The clip produced no scored decisions, so the signal is thin."],
            recommended_action=_action_for_score(0.18),
            title=_candidate_title(candidate),
        )

    top_scores = sorted((float(record["score"]["final_score"]) for record in scored), reverse=True)
    top_mean = sum(top_scores[: min(3, len(top_scores))]) / min(3, len(top_scores))
    confidence_mean = sum(float(record["decision"]["confidence"]) for record in scored) / len(scored)
    density = min(len(decisions) / 6.0, 1.0)
    score = _clamp(0.65 * top_mean + 0.2 * confidence_mean + 0.15 * density)

    reasons = _video_reasons(scored, top_mean)
    risks = _video_risks(scored, confidence_mean)

    return Judgement(
        candidate_id=candidate_id,
        kind=kind,
        score=round(score, 4),
        status=_status_for_score(score),
        reasons=reasons,
        risks=risks,
        recommended_action=_action_for_score(score),
        title=_candidate_title(candidate),
    )


def _video_reasons(scored: list[dict[str, Any]], top_mean: float) -> list[str]:
    rule_counts: dict[str, int] = {}
    for record in scored:
        reason_value = str(record["decision"].get("reason") or "")
        for rule in reason_value.split("|"):
            cleaned = rule.split("(")[0].strip()
            if cleaned:
                rule_counts[cleaned] = rule_counts.get(cleaned, 0) + 1

    ordered_rules = sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)
    reasons: list[str] = []
    if ordered_rules:
        dominant = ", ".join(rule for rule, _count in ordered_rules[:2])
        reasons.append(f"Strong edit signals repeat across the clip: {dominant}.")
    if top_mean >= 0.72:
        reasons.append("Multiple moments scored strongly enough to justify attention now.")
    elif top_mean >= 0.58:
        reasons.append("The best moments are usable, even if the signal is not dominant throughout.")
    else:
        reasons.append("There is at least one usable moment, but the upside is limited.")
    return reasons[:3]


def _video_risks(scored: list[dict[str, Any]], confidence_mean: float) -> list[str]:
    risks: list[str] = []
    low_scores = [float(record["score"]["final_score"]) for record in scored if float(record["score"]["final_score"]) < 0.45]
    if low_scores:
        risks.append("Quality is uneven outside the best moments.")
    if len(scored) < 2:
        risks.append("The clip only produced a small amount of decisive evidence.")
    if confidence_mean < 0.45:
        risks.append("Planner confidence stayed modest, so ranking confidence is limited.")
    if not risks:
        risks.append("No major structural risk surfaced in the media pass.")
    return risks[:3]


def _judge_generic_candidate(candidate_id: str, kind: str, candidate: dict[str, Any]) -> Judgement:
    signals = _merged_generic_signals(candidate)
    text_value = _candidate_text(candidate)
    community = candidate.get("community") if isinstance(candidate.get("community"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    link_count = _safe_int(community.get("link_count"), len(URL_RE.findall(text_value)))
    attachment_count = _safe_int(community.get("attachment_count"))
    flags = _candidate_theme_flags(candidate, text_value, community, source, link_count, attachment_count)

    positive_entries: list[tuple[str, float]] = []
    negative_entries: list[tuple[str, float]] = []
    for key, value in signals.items():
        normalised = _normalise_signal(key, value)
        if normalised is None:
            continue
        if _is_negative_signal(key):
            negative_entries.append((key, normalised))
        else:
            positive_entries.append((key, normalised))

    completeness = _completeness_score(candidate)
    positive_mean = _average(value for _key, value in positive_entries) if positive_entries else 0.35
    negative_mean = _average(value for _key, value in negative_entries) if negative_entries else 0.25
    core_signal = _average(
        signals.get(name, 0.0)
        for name in ("community_support", "credibility", "clarity", "source_quality", "relevance")
        if name in signals
    )
    score = _clamp(0.45 * positive_mean + 0.3 * core_signal + 0.15 * (1.0 - negative_mean) + 0.1 * completeness)
    strong_positive_count = sum(1 for _key, value in positive_entries if value >= 0.75)
    if strong_positive_count >= 3:
        score = _clamp(score + 0.08)
    if flags["price_chatter"] and not flags["artifact_path"]:
        score = _clamp(score - 0.18)
    if flags["vague_hype"]:
        score = _clamp(score - 0.12)
    if not flags["artifact_path"] and signals.get("community_support", 0.0) < 0.25 and signals.get("credibility", 0.0) < 0.4:
        score = _clamp(score - 0.08)

    reasons = _generic_reasons(candidate, positive_entries, completeness, signals)
    risks = _generic_risks(candidate, negative_entries, completeness, score, signals)

    return Judgement(
        candidate_id=candidate_id,
        kind=kind,
        score=round(score, 4),
        status=_status_for_score(score),
        reasons=reasons,
        risks=risks,
        recommended_action=_action_for_score(score),
        title=_candidate_title(candidate),
    )


def _generic_reasons(
    candidate: dict[str, Any],
    positive_entries: list[tuple[str, float]],
    completeness: float,
    signals: dict[str, float],
) -> list[str]:
    community = candidate.get("community") if isinstance(candidate.get("community"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    text_value = _candidate_text(candidate)
    link_count = _safe_int(community.get("link_count"), len(URL_RE.findall(text_value)))
    attachment_count = _safe_int(community.get("attachment_count"))
    flags = _candidate_theme_flags(candidate, text_value, community, source, link_count, attachment_count)
    low_context_reaction = flags["reaction_only"] or (
        flags["no_artifact_path"]
        and signals.get("source_quality", 0.0) <= 0.15
        and signals.get("credibility", 0.0) <= 0.2
    )
    reasons: list[str] = []

    if flags["has_receipts"] and flags["clean_hook"] and signals.get("credibility", 0.0) >= 0.65:
        _append_unique(reasons, "Strong because it has receipts and a clean hook.")
    if flags["actionable"] and signals.get("relevance", 0.0) >= 0.65:
        _append_unique(reasons, "Useful because builders can act on it today.")
    if flags["brand_risk"] and signals.get("relevance", 0.0) >= 0.65:
        _append_unique(reasons, "Important because it carries brand risk and needs a clear call.")
    if low_context_reaction:
        _append_unique(reasons, "Low-context reaction, useful as mood but not direction.")
    if community.get("reply_count", 0) >= 2 and signals.get("community_support", 0.0) >= 0.65 and flags["concrete_angle"] and not flags["price_chatter"]:
        _append_unique(reasons, "The room is engaging with a concrete angle, not just reacting.")
    if flags["artifact_path"] and not flags["has_receipts"] and signals.get("source_quality", 0.0) >= 0.6:
        _append_unique(reasons, "It points to an artifact path instead of only a vibe.")

    for key, value in sorted(positive_entries, key=lambda item: item[1], reverse=True):
        if value < 0.6:
            continue
        if low_context_reaction:
            continue
        if flags["price_chatter"] and key in {"community_support", "clarity"}:
            continue
        if flags["vague_hype"] and key == "clarity":
            continue
        _append_unique(reasons, _positive_signal_reason(key))
        if len(reasons) == 2:
            break

    if completeness >= 0.66 and not (flags["price_chatter"] or flags["vague_hype"] or low_context_reaction):
        _append_unique(reasons, "There is enough context here to make a confident call.")
    if not reasons:
        if low_context_reaction:
            reasons.append("Low-context reaction, useful as mood but not direction.")
        elif text_value:
            reasons.append("The submission is concrete enough to evaluate, even without rich signal data.")
        else:
            reasons.append("The candidate has some directional signal, but it is still sparse.")
    return reasons[:3]


def _generic_risks(
    candidate: dict[str, Any],
    negative_entries: list[tuple[str, float]],
    completeness: float,
    score: float,
    signals: dict[str, float],
) -> list[str]:
    risks: list[str] = []
    community = candidate.get("community") if isinstance(candidate.get("community"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    text_value = _candidate_text(candidate)
    link_count = _safe_int(community.get("link_count"), len(URL_RE.findall(text_value)))
    attachment_count = _safe_int(community.get("attachment_count"))
    flags = _candidate_theme_flags(candidate, text_value, community, source, link_count, attachment_count)

    if flags["brand_risk"]:
        _append_unique(risks, "Risky because it surfaces brand damage and needs careful framing.")
    if community.get("reaction_count", 0) + community.get("reply_count", 0) == 0 and signals.get("community_support", 0.0) < 0.3:
        _append_unique(risks, "The room has not really validated it yet.")
    if flags["price_chatter"] and not flags["artifact_path"]:
        _append_unique(risks, "Noise because it only has price energy, no artifact path.")
    if flags["bad_external_frame"]:
        _append_unique(risks, "Risky because it borrows a bad external frame.")
    if flags["reaction_only"]:
        _append_unique(risks, "Low-context reaction only; it should not drive direction by itself.")
    elif flags["no_artifact_path"] and not flags["price_chatter"]:
        _append_unique(risks, "Risky because there is still no artifact path or clear next step.")
    if flags["vague_hype"]:
        _append_unique(risks, "Risky because it asks for attention without enough substance.")

    for key, value in sorted(negative_entries, key=lambda item: item[1], reverse=True):
        if value < 0.45:
            continue
        _append_unique(risks, _negative_signal_risk(key))
        if len(risks) == 2:
            break

    if completeness < 0.5:
        _append_unique(risks, "The submission needs more context before it should drive action.")
    if score < 0.55 and not flags["price_chatter"]:
        _append_unique(risks, "Signal is weak relative to the rest of the batch.")
    if not risks:
        if not text_value:
            risks.append("There is not much explanation attached to this candidate yet.")
        else:
            risks.append("No major risk surfaced, but it still needs human taste review.")
    return risks[:3]


def _candidate_title(candidate: dict[str, Any]) -> str | None:
    for key in ("title", "headline", "name"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _completeness_score(candidate: dict[str, Any]) -> float:
    fields = (
        candidate.get("title"),
        candidate.get("text"),
        candidate.get("description"),
        candidate.get("path"),
        candidate.get("url"),
    )
    present = sum(1 for value in fields if isinstance(value, str) and value.strip())
    community = candidate.get("community") if isinstance(candidate.get("community"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    metadata_bonus = 0.1 * int(bool(community)) + 0.1 * int(bool(source))
    return _clamp((present / len(fields)) + metadata_bonus)


def _merged_generic_signals(candidate: dict[str, Any]) -> dict[str, float]:
    explicit = candidate.get("signals") if isinstance(candidate.get("signals"), dict) else {}
    merged = {key: value for key, value in ((_humanise_signal_key(k), _normalise_signal(k, v)) for k, v in explicit.items()) if value is not None}

    derived = _derived_generic_signals(candidate)
    for key, value in derived.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = round(max(merged[key], value), 4)
    return merged


def _derived_generic_signals(candidate: dict[str, Any]) -> dict[str, float]:
    text_value = _candidate_text(candidate)
    lowered = text_value.lower()
    community = candidate.get("community") if isinstance(candidate.get("community"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}

    reaction_count = _safe_int(community.get("reaction_count"))
    reply_count = _safe_int(community.get("reply_count"))
    attachment_count = _safe_int(community.get("attachment_count"))
    link_count = _safe_int(community.get("link_count"), len(URL_RE.findall(text_value)))
    trusted_submitter = bool(community.get("trusted_submitter"))
    word_count = len(text_value.split())
    hedge_hits = sum(1 for token in HEDGE_TOKENS if token in lowered)
    evidence_hits = sum(1 for token in COMMUNITY_TEXT_SIGNAL_TOKENS if token in lowered)
    channel_name = str(source.get("channel_name") or "").lower()
    flags = _candidate_theme_flags(candidate, text_value, community, source, link_count, attachment_count)

    clarity = _clamp(
        0.15
        + min(word_count / 18.0, 1.0) * 0.65
        + (0.12 if any(mark in text_value for mark in ".:?!") else 0.0)
        + (0.1 if flags["clean_hook"] else 0.0)
        + (0.08 if flags["actionable"] else 0.0)
        - (0.1 if flags["price_chatter"] and not flags["artifact_path"] else 0.0)
        - (0.18 if flags["vague_hype"] else 0.0)
    )
    community_support = _clamp((reaction_count + reply_count * 1.4 + attachment_count * 0.5) / 12.0)
    credibility = _clamp(
        0.18
        + min(link_count, 2) * 0.2
        + min(evidence_hits, 2) * 0.16
        + (0.2 if trusted_submitter else 0.0)
        + (0.18 if flags["has_receipts"] else 0.0)
        + (0.08 if flags["actionable"] else 0.0)
        - (0.16 if flags["price_chatter"] and not flags["artifact_path"] else 0.0)
        - (0.12 if flags["vague_hype"] else 0.0)
    )
    source_quality = _clamp(
        0.12
        + min(link_count, 2) * 0.22
        + min(attachment_count, 2) * 0.1
        + (0.18 if trusted_submitter else 0.0)
        + (0.14 if flags["artifact_path"] else 0.0)
        - (0.18 if flags["price_chatter"] and not flags["artifact_path"] else 0.0)
    )
    relevance = _clamp(
        0.42
        + (0.12 if any(token in lowered for token in COMMUNITY_TEXT_SIGNAL_TOKENS) else 0.0)
        + (0.12 if any(token in channel_name for token in ("ideas", "clips", "research", "memes")) else 0.0)
        + (0.16 if flags["brand_risk"] else 0.0)
        + (0.14 if flags["actionable"] else 0.0)
        + (0.08 if flags["artifact_path"] else 0.0)
        - (0.12 if flags["price_chatter"] and not flags["artifact_path"] else 0.0)
    )
    novelty = _clamp(
        0.35
        + min(attachment_count, 2) * 0.08
        + (0.14 if "hot take" in lowered or "unexpected" in lowered else 0.0)
        + (0.08 if flags["brand_risk"] else 0.0)
        + (0.06 if flags["actionable"] else 0.0)
    )
    uncertainty = _clamp(
        0.04
        + hedge_hits * 0.14
        + (0.18 if word_count < 6 else 0.0)
        + (0.12 if reaction_count + reply_count == 0 else 0.0)
        + (0.16 if flags["price_chatter"] and not flags["artifact_path"] else 0.0)
        + (0.18 if flags["vague_hype"] else 0.0)
        - (0.08 if flags["has_receipts"] else 0.0)
        - (0.06 if flags["actionable"] else 0.0)
    )

    return {
        "community_support": round(community_support, 4),
        "clarity": round(clarity, 4),
        "credibility": round(credibility, 4),
        "source_quality": round(source_quality, 4),
        "relevance": round(relevance, 4),
        "novelty": round(novelty, 4),
        "uncertainty": round(uncertainty, 4),
    }


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        part.strip()
        for part in (
            str(candidate.get("title") or ""),
            str(candidate.get("text") or ""),
            str(candidate.get("description") or ""),
        )
        if part.strip()
    )


def _candidate_theme_flags(
    candidate: dict[str, Any],
    text_value: str,
    community: dict[str, Any],
    source: dict[str, Any],
    link_count: int,
    attachment_count: int,
) -> dict[str, bool]:
    explicit = candidate.get("signals") if isinstance(candidate.get("signals"), dict) else {}
    explicit_risk_type = _signal_text(explicit, "risk_type")
    explicit_source = _signal_text(explicit, "source")
    combined = " ".join(
        part.strip().lower()
        for part in (
            str(candidate.get("title") or ""),
            text_value,
            str(candidate.get("description") or ""),
            str(source.get("channel_name") or ""),
            explicit_source,
            explicit_risk_type,
            " ".join(str(item) for item in source.get("external_urls", []) if item),
        )
        if part and part.strip()
    )

    explicit_artifact_path = _signal_bool(explicit, "artifact_path")
    explicit_no_artifact_path = _signal_bool(explicit, "no_artifact_path")
    artifact_path = (
        explicit_artifact_path
        or (
            not explicit_no_artifact_path
            and (
                link_count > 0
                or attachment_count > 0
                or (
                    _contains_any(combined, ARTIFACT_TOKENS)
                    and not _contains_any(combined, NEGATED_ARTIFACT_TOKENS)
                )
            )
        )
    )
    no_artifact_path = explicit_no_artifact_path or (not artifact_path and _contains_any(combined, NEGATED_ARTIFACT_TOKENS))
    has_receipts = _signal_bool(explicit, "has_receipts") or link_count > 0 or _contains_any(combined, RECEIPT_TOKENS)
    actionable = _signal_bool(explicit, "actionable") or _contains_any(combined, ACTIONABLE_TOKENS)
    brand_risk = _signal_bool(explicit, "brand_risk") or explicit_risk_type in {"brand_frame", "brand_risk", "disclosure_spill"} or _contains_any(combined, BRAND_RISK_TOKENS)
    bad_external_frame = _signal_bool(explicit, "bad_external_frame") or explicit_risk_type == "brand_frame" or _contains_any(combined, BAD_FRAME_TOKENS)
    price_chatter = _signal_bool(explicit, "price_chatter") or explicit_risk_type == "price_chatter" or _contains_any(combined, PRICE_CHATTER_TOKENS)
    vague_hype = _signal_bool(explicit, "vague_hype") or explicit_risk_type == "vague_hype" or (_contains_any(combined, HYPE_TOKENS) and not has_receipts and not artifact_path)
    reaction_only = explicit_risk_type == "reaction_only" or (no_artifact_path and _contains_any(combined, REACTION_ONLY_TOKENS))
    clean_hook = bool(str(candidate.get("title") or "").strip()) and (
        ":" in str(candidate.get("title") or "")
        or has_receipts
        or actionable
        or brand_risk
        or len(str(candidate.get("title") or "").split()) >= 4
    )
    concrete_angle = has_receipts or actionable or brand_risk or artifact_path or len(text_value.split()) >= 12 or _safe_int(community.get("reply_count")) >= 2

    return {
        "artifact_path": artifact_path,
        "no_artifact_path": no_artifact_path,
        "has_receipts": has_receipts,
        "actionable": actionable,
        "brand_risk": brand_risk,
        "bad_external_frame": bad_external_frame,
        "price_chatter": price_chatter,
        "vague_hype": vague_hype,
        "reaction_only": reaction_only,
        "clean_hook": clean_hook,
        "concrete_angle": concrete_angle,
    }


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for token in tokens:
        pattern = rf"(?<!\w){re.escape(token.lower())}(?!\w)"
        if re.search(pattern, lowered):
            return True
    return False


def _signal_bool(signals: dict[str, Any], key: str) -> bool:
    value = signals.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _signal_text(signals: dict[str, Any], key: str) -> str:
    value = signals.get(key)
    return value.strip().lower() if isinstance(value, str) else ""


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _positive_signal_reason(key: str) -> str:
    phrases = {
        "community_support": "The room is already giving it real attention.",
        "credibility": "The claim is grounded enough to trust the premise.",
        "clarity": "The hook is easy to understand.",
        "source_quality": "There is an artifact path behind the claim.",
        "relevance": "It maps cleanly to what the community should pay attention to.",
        "novelty": "There is a fresh angle here worth tracking.",
    }
    return phrases.get(key, f"{_humanise_key(key).capitalize()} is strong.")


def _negative_signal_risk(key: str) -> str:
    phrases = {
        "uncertainty": "Risky because the claim is still vague and under-evidenced.",
        "community_support": "The room has not validated it yet.",
        "source_quality": "Risky because there is no artifact path yet.",
        "credibility": "Risky because the claim still lacks enough evidence.",
    }
    return phrases.get(key, f"{_humanise_key(key).capitalize()} is high.")


def _normalise_signal(key: str, value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric < 0:
        numeric = 0.0
    key_l = key.lower()
    if 0.0 <= numeric <= 1.0:
        return numeric
    if any(token in key_l for token in ENGAGEMENT_SIGNAL_TOKENS):
        return numeric / (numeric + 25.0)
    if numeric <= 10.0:
        return numeric / 10.0
    return min(numeric / 100.0, 1.0)


def _humanise_signal_key(key: str) -> str:
    return key.strip().lower()


def _is_negative_signal(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in NEGATIVE_SIGNAL_TOKENS)


def _humanise_key(key: str) -> str:
    return key.replace("_", " ").replace("-", " ")


def _average(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _status_for_score(score: float) -> str:
    if score >= 0.65:
        return "strong_signal"
    if score >= 0.55:
        return "needs_work"
    if score >= 0.35:
        return "unclear"
    return "probably_noise"


def _action_for_score(score: float) -> str:
    if score >= 0.65:
        return "feature"
    if score >= 0.55:
        return "refine"
    if score >= 0.35:
        return "clarify"
    return "pass"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge candidate media, posts, and ideas")
    parser.add_argument("candidates", help="Path to candidates.json")
    parser.add_argument("--output", default="judgements.json", help="Path to write judgements.json")
    parser.add_argument("--summary-output", default=None, help="Optional path to write the summary JSON")
    parser.add_argument("--work-dir", default=".judgement", help="Temporary work directory for scoring")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Media planner threshold in [0,1]")
    parser.add_argument("--skip-schema-validation", action="store_true", help="Skip candidates/judgements schema validation")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary output")
    args = parser.parse_args()

    try:
        candidate_payload = load_candidates_payload(args.candidates)
        if not args.skip_schema_validation:
            validate_candidates_payload(candidate_payload)

        payload = judge_candidates(
            candidate_payload["candidates"],
            work_dir=args.work_dir,
            min_confidence=args.min_confidence,
        )
        if not args.skip_schema_validation:
            validate_judgements_payload(payload)

        write_judgements(args.output, payload)
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload["summary"], indent=2) + "\n", encoding="utf-8")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to read or write judgement files: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())