"""Convert Discord exports into candidates.json batches for judgement."""

from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from copy import deepcopy
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
TRUSTED_ROLE_TOKENS = ("mod", "admin", "trusted", "core", "team", "research")
EVIDENCE_TOKENS = ("source", "receipt", "receipts", "proof", "evidence", "data", "study", "report")
IDEA_TOKENS = ("what if", "should we", "idea", "maybe we", "could we")
HEDGE_TOKENS = ("maybe", "might", "idk", "not sure", "unclear", "perhaps", "guess")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def load_discord_export(path: str) -> list[dict[str, Any]]:
    export_path = Path(path)
    raw = export_path.read_text(encoding="utf-8")
    if export_path.suffix.lower() in {".html", ".htm"} or raw.lstrip().lower().startswith(("<!doctype html", "<html")):
        messages = _load_discord_html_export(raw)
        if not messages:
            raise ValueError("Discord HTML export produced no messages")
        return messages

    data = json.loads(raw)
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and isinstance(data.get("messages"), list):
        messages = data["messages"]
    else:
        raise ValueError("Discord export must be a list or an object with a 'messages' list")

    if not messages:
        raise ValueError("Discord export must contain at least one message")
    return messages


class _DiscordHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[dict[str, Any]] = []
        self._in_message = False
        self._li_depth = 0
        self._in_chat_name = False
        self._in_time = False
        self._paragraph_depth = 0
        self._paragraph_parts: list[str] = []
        self._current: dict[str, Any] = {}
        self._content_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        classes = set(attrs_dict.get("class", "").split())

        if tag == "li":
            self._in_message = True
            self._li_depth += 1
            self._current = {}
            self._content_parts = []
            return

        if not self._in_message:
            return

        if tag == "li":
            self._li_depth += 1
        elif tag == "span" and "chatName" in classes:
            self._in_chat_name = True
        elif tag == "span" and "time" in classes:
            self._in_time = True
        elif tag == "p" and "timeInfo" not in classes:
            self._paragraph_depth += 1
            self._paragraph_parts = []
        elif tag == "br" and self._paragraph_depth:
            self._paragraph_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._in_message:
            return

        if tag == "span":
            self._in_chat_name = False
            self._in_time = False
        elif tag == "p" and self._paragraph_depth:
            text = _clean_text(unescape("".join(self._paragraph_parts)))
            if text:
                self._content_parts.append(text)
            self._paragraph_depth -= 1
            self._paragraph_parts = []
        elif tag == "li":
            self._li_depth -= 1
            if self._li_depth <= 0:
                self._finish_message()

    def handle_data(self, data: str) -> None:
        if not self._in_message:
            return
        if self._in_chat_name:
            self._current["author_name"] = (self._current.get("author_name", "") + data).strip()
        elif self._in_time:
            self._current["timestamp"] = (self._current.get("timestamp", "") + data).strip()
        elif self._paragraph_depth:
            self._paragraph_parts.append(data)

    def _finish_message(self) -> None:
        content = _clean_text("\n".join(self._content_parts))
        author_name = str(self._current.get("author_name") or "unknown member").strip()
        timestamp = str(self._current.get("timestamp") or "").strip()
        if content:
            self.messages.append(
                {
                    "id": f"html_{len(self.messages) + 1}",
                    "channel_name": "html-export",
                    "content": content,
                    "timestamp": timestamp,
                    "author": {"display_name": author_name, "roles": []},
                }
            )
        self._in_message = False
        self._li_depth = 0
        self._current = {}
        self._content_parts = []


def _load_discord_html_export(raw: str) -> list[dict[str, Any]]:
    parser = _DiscordHtmlParser()
    parser.feed(raw)
    return parser.messages


def discord_export_to_candidates(messages: list[dict[str, Any]], *, redact_public: bool = False) -> dict[str, Any]:
    author_stats = _author_message_counts(messages)
    candidates: list[dict[str, Any]] = []
    for message in messages:
        candidate = _message_to_candidate(message, author_stats=author_stats)
        if candidate is not None:
            candidates.append(candidate)

    candidates = _collapse_duplicate_candidates(candidates)
    if redact_public:
        candidates = [_redact_public_candidate(candidate) for candidate in candidates]

    if not candidates:
        raise ValueError("Discord export produced no candidate messages")

    return {"candidates": candidates}


def write_candidates(path: str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _message_to_candidate(message: dict[str, Any], *, author_stats: dict[str, int]) -> dict[str, Any] | None:
    content = _clean_text(str(message.get("content") or ""))
    attachments = [item for item in message.get("attachments", []) if isinstance(item, dict)]
    embeds = [item for item in message.get("embeds", []) if isinstance(item, dict)]
    author = message.get("author") if isinstance(message.get("author"), dict) else {}

    if not content and not attachments and not embeds:
        return None

    reaction_count = sum(_safe_int(item.get("count"), 1) for item in message.get("reactions", []) if isinstance(item, dict))
    reply_count = _safe_int(message.get("reply_count"), len(message.get("replies", [])) if isinstance(message.get("replies"), list) else 0)
    attachment_count = len(attachments)
    link_count = len(URL_RE.findall(content)) + sum(1 for item in embeds if item.get("url"))
    image_count = sum(1 for item in attachments if _suffix(item.get("filename")) in IMAGE_SUFFIXES)
    video_count = sum(1 for item in attachments if _suffix(item.get("filename")) in VIDEO_SUFFIXES)
    roles = [str(role).lower() for role in author.get("roles", []) if role]
    author_name = str(author.get("display_name") or author.get("username") or "").strip() or "unknown member"
    author_message_count = author_stats.get(author_name.lower(), 1)
    external_urls = _external_urls(content, embeds)
    referenced = message.get("referenced_message") if isinstance(message.get("referenced_message"), dict) else {}
    thread_name = str(message.get("thread_name") or "").strip()
    parent_message_id = str(message.get("parent_message_id") or referenced.get("id") or "").strip()

    candidate_id = f"discord_{message.get('id') or len(content)}"
    kind = _guess_kind(content, attachment_count, image_count, video_count, link_count)
    title = _message_title(content, kind, author)
    description = _message_description(
        message,
        author_name=author_name,
        reaction_count=reaction_count,
        reply_count=reply_count,
        thread_name=thread_name,
        referenced=referenced,
        author_message_count=author_message_count,
        has_content=bool(content),
        attachment_count=attachment_count,
    )
    source_url = str(message.get("jump_url") or "").strip() or None
    trusted_submitter = any(any(token in role for token in TRUSTED_ROLE_TOKENS) for role in roles)

    signals = {
        "community_support": _clamp((reaction_count + reply_count * 1.6 + attachment_count * 0.5) / 12.0),
        "clarity": _clarity_score(content),
        "credibility": _credibility_score(content, link_count, roles, author_message_count),
        "source_quality": _source_quality_score(link_count, attachment_count, roles, bool(content), author_message_count),
        "novelty": _novelty_score(content, attachment_count, image_count, video_count),
        "relevance": _relevance_score(content, message, attachment_count=attachment_count, image_count=image_count, video_count=video_count),
        "uncertainty": _uncertainty_score(content, reaction_count, reply_count, attachment_count=attachment_count, image_count=image_count),
    }

    return {
        "id": candidate_id,
        "kind": kind,
        "title": title,
        "text": content or _attachment_summary(attachments, image_count=image_count, video_count=video_count),
        "description": description,
        "url": source_url,
        "source": {
            "platform": "discord",
            "message_id": str(message.get("id") or ""),
            "channel_name": str(message.get("channel_name") or message.get("channel") or "").strip(),
            "author_name": author_name,
            "timestamp": str(message.get("timestamp") or "").strip(),
            "thread_name": thread_name,
            "parent_message_id": parent_message_id or None,
            "external_urls": external_urls,
            "related_message_ids": [str(message.get("id") or "")],
        },
        "community": {
            "reaction_count": reaction_count,
            "reply_count": reply_count,
            "attachment_count": attachment_count,
            "link_count": link_count,
            "trusted_submitter": trusted_submitter,
            "author_message_count": author_message_count,
            "duplicate_count": 1,
            "unique_author_count": 1,
        },
        "signals": signals,
    }


def _author_message_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        author = message.get("author") if isinstance(message.get("author"), dict) else {}
        author_name = str(author.get("display_name") or author.get("username") or "").strip().lower()
        if author_name:
            counts[author_name] = counts.get(author_name, 0) + 1
    return counts


def _collapse_duplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for candidate in candidates:
        key = _dedupe_key(candidate)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(candidate)

    collapsed: list[dict[str, Any]] = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        collapsed.append(_merge_duplicate_group(group))
    return collapsed


def _dedupe_key(candidate: dict[str, Any]) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    external_urls = [str(item).strip().lower() for item in source.get("external_urls", []) if str(item).strip()]
    if external_urls:
        return f"url:{external_urls[0]}"

    text = _clean_text(str(candidate.get("text") or "")).lower()
    if text:
        return f"text:{text[:120]}"

    title = _clean_text(str(candidate.get("title") or "")).lower()
    return f"title:{title[:120]}"


def _merge_duplicate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    primary = deepcopy(
        max(
            group,
            key=lambda item: (
                _safe_int(item.get("community", {}).get("reaction_count")),
                _safe_int(item.get("community", {}).get("reply_count")),
                len(str(item.get("text") or "")),
            ),
        )
    )
    community = primary.get("community") if isinstance(primary.get("community"), dict) else {}
    source = primary.get("source") if isinstance(primary.get("source"), dict) else {}

    reaction_count = sum(_safe_int(item.get("community", {}).get("reaction_count")) for item in group)
    reply_count = sum(_safe_int(item.get("community", {}).get("reply_count")) for item in group)
    attachment_count = sum(_safe_int(item.get("community", {}).get("attachment_count")) for item in group)
    trusted_submitter = any(bool(item.get("community", {}).get("trusted_submitter")) for item in group)
    author_names = sorted({str(item.get("source", {}).get("author_name") or "").strip() for item in group if str(item.get("source", {}).get("author_name") or "").strip()})
    related_message_ids = [str(item.get("source", {}).get("message_id") or "").strip() for item in group if str(item.get("source", {}).get("message_id") or "").strip()]
    external_urls = sorted({str(url).strip() for item in group for url in item.get("source", {}).get("external_urls", []) if str(url).strip()})

    community["reaction_count"] = reaction_count
    community["reply_count"] = reply_count
    community["attachment_count"] = attachment_count
    community["link_count"] = len(external_urls)
    community["trusted_submitter"] = trusted_submitter
    community["duplicate_count"] = len(group)
    community["unique_author_count"] = len(author_names) or 1
    community["author_message_count"] = max(_safe_int(item.get("community", {}).get("author_message_count"), 1) for item in group)

    source["related_message_ids"] = related_message_ids
    source["external_urls"] = external_urls
    source["author_names"] = author_names

    primary["description"] = (
        f"{primary['description']} Aggregated from {len(group)} similar Discord messages "
        f"across {community['unique_author_count']} authors."
    )
    primary["signals"] = {
        "community_support": _clamp((reaction_count + reply_count * 1.6 + attachment_count * 0.5) / 12.0),
        "clarity": max(float(item.get("signals", {}).get("clarity", 0.0)) for item in group),
        "credibility": max(float(item.get("signals", {}).get("credibility", 0.0)) for item in group),
        "source_quality": max(float(item.get("signals", {}).get("source_quality", 0.0)) for item in group),
        "novelty": max(float(item.get("signals", {}).get("novelty", 0.0)) for item in group),
        "relevance": max(float(item.get("signals", {}).get("relevance", 0.0)) for item in group),
        "uncertainty": min(float(item.get("signals", {}).get("uncertainty", 1.0)) for item in group),
    }
    return primary


def _redact_public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    redacted = deepcopy(candidate)
    redacted.pop("url", None)

    kind = str(redacted.get("kind") or "post").strip() or "post"
    text = str(redacted.get("text") or "").strip()
    community = redacted.get("community") if isinstance(redacted.get("community"), dict) else {}
    duplicate_count = max(_safe_int(community.get("duplicate_count"), 1), 1)
    unique_author_count = max(_safe_int(community.get("unique_author_count"), 1), 1)
    reaction_count = _safe_int(community.get("reaction_count"))
    reply_count = _safe_int(community.get("reply_count"))
    attachment_count = _safe_int(community.get("attachment_count"))

    if not text and attachment_count:
        description = "Redacted attachment-only Discord post prepared for a public judgement fixture."
    else:
        description = (
            "Redacted Discord submission prepared for a public judgement fixture "
            f"with {reaction_count} reactions and {reply_count} replies."
        )
    if duplicate_count > 1:
        description = (
            f"{description} Aggregated from {duplicate_count} similar messages across "
            f"{unique_author_count} authors."
        )

    redacted["description"] = description
    if not text:
        redacted["title"] = f"{kind.capitalize()} from redacted Discord"

    redacted["source"] = {
        "platform": "discord",
        "redacted": True,
    }

    signals = redacted.get("signals") if isinstance(redacted.get("signals"), dict) else None
    if signals is not None:
        signals["source"] = "discord_redacted"

    return redacted


def _guess_kind(content: str, attachment_count: int, image_count: int, video_count: int, link_count: int) -> str:
    lowered = content.lower()
    if video_count:
        return "clip"
    if image_count and len(lowered.split()) < 18:
        return "meme"
    if link_count and any(token in lowered for token in EVIDENCE_TOKENS):
        return "claim"
    if any(token in lowered for token in IDEA_TOKENS):
        return "idea"
    if attachment_count:
        return "post"
    if link_count:
        return "link"
    return "idea"


def _message_title(content: str, kind: str, author: dict[str, Any]) -> str:
    text = content.strip()
    if text:
        words = text.split()
        short = " ".join(words[:8]).strip()
        if len(words) > 8:
            short += "..."
        return short
    author_name = str(author.get("display_name") or author.get("username") or "community").strip()
    return f"{kind.capitalize()} from {author_name}"


def _message_description(
    message: dict[str, Any],
    *,
    author_name: str,
    reaction_count: int,
    reply_count: int,
    thread_name: str,
    referenced: dict[str, Any],
    author_message_count: int,
    has_content: bool,
    attachment_count: int,
) -> str:
    channel_name = str(message.get("channel_name") or message.get("channel") or "discord").strip()
    if not has_content and attachment_count:
        base = f"Attachment-only Discord post from {author_name} in #{channel_name}."
    else:
        base = f"Discord submission from {author_name} in #{channel_name} with {reaction_count} reactions and {reply_count} replies."
    details: list[str] = []
    if thread_name:
        details.append(f"Thread: {thread_name}.")
    if referenced:
        parent_excerpt = _clean_text(str(referenced.get("content") or ""))
        if parent_excerpt:
            details.append(f"Reply context: {parent_excerpt[:80]}")
    if author_message_count > 1:
        details.append(f"Submitter has {author_message_count} messages in this export.")
    return " ".join([base, *details]).strip()


def _attachment_summary(attachments: list[dict[str, Any]], *, image_count: int, video_count: int) -> str:
    if image_count and not video_count:
        return ""
    names = [str(item.get("filename") or "attachment").strip() for item in attachments[:3]]
    return ", ".join(name for name in names if name)


def _clarity_score(content: str) -> float:
    words = content.split()
    count = len(words)
    if count == 0:
        return 0.1
    length_score = min(count / 18.0, 1.0)
    punctuation_bonus = 0.15 if any(mark in content for mark in ".:?!") else 0.0
    return _clamp(0.2 + 0.65 * length_score + punctuation_bonus)


def _credibility_score(content: str, link_count: int, roles: list[str], author_message_count: int) -> float:
    lowered = content.lower()
    trusted = any(any(token in role for token in TRUSTED_ROLE_TOKENS) for role in roles)
    evidence_hits = sum(1 for token in EVIDENCE_TOKENS if token in lowered)
    repeat_bonus = 0.05 if author_message_count > 1 else 0.0
    return _clamp(0.2 + min(link_count, 2) * 0.2 + min(evidence_hits, 2) * 0.18 + (0.2 if trusted else 0.0) + repeat_bonus)


def _source_quality_score(link_count: int, attachment_count: int, roles: list[str], has_text: bool, author_message_count: int) -> float:
    trusted = any(any(token in role for token in TRUSTED_ROLE_TOKENS) for role in roles)
    attachment_bonus = min(attachment_count, 2) * (0.12 if has_text else 0.05)
    repeat_bonus = 0.05 if author_message_count > 1 else 0.0
    return _clamp(0.12 + min(link_count, 2) * 0.25 + attachment_bonus + (0.2 if trusted else 0.0) + repeat_bonus)


def _novelty_score(content: str, attachment_count: int, image_count: int, video_count: int) -> float:
    lowered = content.lower()
    hook_bonus = 0.2 if any(token in lowered for token in ("new", "unexpected", "wild", "hot take", "nobody")) else 0.0
    media_bonus = min(attachment_count, 2) * 0.08 + image_count * 0.05 + video_count * 0.1
    return _clamp(0.35 + hook_bonus + media_bonus)


def _relevance_score(content: str, message: dict[str, Any], *, attachment_count: int, image_count: int, video_count: int) -> float:
    lowered = content.lower()
    channel_name = str(message.get("channel_name") or message.get("channel") or "").lower()
    topical = any(token in lowered for token in ("claim", "clip", "idea", "post", "thread", "community"))
    channel_bonus = 0.15 if any(token in channel_name for token in ("ideas", "clips", "research", "memes")) else 0.0
    base = 0.45
    if not content and image_count and not video_count:
        base = 0.12
    elif attachment_count and not content:
        base = 0.3
    return _clamp(base + (0.12 if topical else 0.0) + channel_bonus)


def _uncertainty_score(content: str, reaction_count: int, reply_count: int, *, attachment_count: int, image_count: int) -> float:
    lowered = content.lower()
    hedge_hits = sum(1 for token in HEDGE_TOKENS if token in lowered)
    sparse_penalty = 0.2 if len(content.split()) < 6 else 0.0
    community_penalty = 0.15 if reaction_count + reply_count == 0 else 0.0
    attachment_only_penalty = 0.2 if not content and attachment_count else 0.0
    meme_penalty = 0.12 if not content and image_count else 0.0
    return _clamp(0.05 + hedge_hits * 0.12 + sparse_penalty + community_penalty + attachment_only_penalty + meme_penalty)


def _external_urls(content: str, embeds: list[dict[str, Any]]) -> list[str]:
    urls = list(URL_RE.findall(content))
    urls.extend(str(item.get("url") or "").strip() for item in embeds if str(item.get("url") or "").strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        lowered = url.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(url)
    return ordered


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _suffix(filename: Any) -> str:
    if not isinstance(filename, str):
        return ""
    return Path(filename).suffix.lower()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
