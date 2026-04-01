"""Message creation and normalization utilities."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


def create_user_message(
    content: str | list[dict[str, Any]],
    uuid_str: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a user message for the conversation."""
    if isinstance(content, str):
        content_blocks = [{"type": "text", "text": content}]
    else:
        content_blocks = content

    return {
        "type": "user",
        "role": "user",
        "content": content_blocks,
        "uuid": uuid_str or str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now().isoformat(),
    }


def create_assistant_message(
    content: list[dict[str, Any]],
    usage: dict[str, int] | None = None,
    model: str = "",
    stop_reason: str = "",
) -> dict[str, Any]:
    """Create an assistant message for the conversation."""
    return {
        "type": "assistant",
        "role": "assistant",
        "content": content,
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "usage": usage or {},
        "model": model,
        "stop_reason": stop_reason,
    }


def normalize_messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize messages for the Anthropic API.

    Ensures alternating user/assistant roles and merges same-role messages.
    """
    if not messages:
        return []

    normalized: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        api_msg = {"role": role, "content": content}

        # Merge consecutive same-role messages
        if normalized and normalized[-1]["role"] == role:
            existing_content = normalized[-1]["content"]
            if isinstance(existing_content, str):
                existing_content = [{"type": "text", "text": existing_content}]
            existing_content.extend(content)
            normalized[-1]["content"] = existing_content
        else:
            normalized.append(api_msg)

    # Ensure messages start with user
    if normalized and normalized[0]["role"] != "user":
        normalized.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})

    # Ensure alternating roles by inserting filler messages
    result: list[dict[str, Any]] = []
    for i, msg in enumerate(normalized):
        if i > 0 and result[-1]["role"] == msg["role"]:
            filler_role = "user" if msg["role"] == "assistant" else "assistant"
            filler_content = "Continue." if filler_role == "user" else "I understand."
            result.append({"role": filler_role, "content": [{"type": "text", "text": filler_content}]})
        result.append(msg)

    return result


def strip_images_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove image content blocks from messages to save tokens."""
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = {**msg}
        content = msg.get("content", [])
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    new_content.append({"type": "text", "text": "[Image removed for context compaction]"})
                else:
                    new_content.append(block)
            new_msg["content"] = new_content
        stripped.append(new_msg)
    return stripped


def extract_text_from_content(content: str | list[dict[str, Any]]) -> str:
    """Extract text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    texts.append(result_content)
                elif isinstance(result_content, list):
                    for rb in result_content:
                        if isinstance(rb, dict) and rb.get("type") == "text":
                            texts.append(rb.get("text", ""))
    return "\n".join(texts)


def create_compact_boundary_message() -> dict[str, Any]:
    """Create a system message marking a compaction boundary."""
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "timestamp": datetime.now().isoformat(),
    }


def truncate_text(text: str, max_chars: int = 100000) -> str:
    """Truncate text to max_chars with an indicator."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [truncated {len(text) - max_chars} characters] ...\n\n" + text[-half:]
