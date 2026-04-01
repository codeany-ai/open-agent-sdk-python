"""Auto-compaction logic for managing context window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from open_agent_sdk.utils.messages import extract_text_from_content, strip_images_from_messages
from open_agent_sdk.utils.tokens import estimate_messages_tokens, get_auto_compact_threshold


@dataclass
class AutoCompactState:
    compacted: bool = False
    turn_counter: int = 0
    consecutive_failures: int = 0


def create_auto_compact_state() -> AutoCompactState:
    return AutoCompactState()


def should_auto_compact(
    messages: list[dict[str, Any]],
    model: str,
    state: AutoCompactState,
) -> bool:
    """Check if conversation should be auto-compacted."""
    if state.consecutive_failures >= 3:
        return False

    estimated_tokens = estimate_messages_tokens(messages)
    threshold = get_auto_compact_threshold(model)
    return estimated_tokens >= threshold


async def compact_conversation(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict[str, Any]],
    state: AutoCompactState,
) -> dict[str, Any]:
    """Compact conversation by summarizing it.

    Returns dict with compacted_messages, summary, and updated state.
    """
    try:
        # Strip images before summarizing
        stripped = strip_images_from_messages(messages)

        # Build summary prompt
        conversation_text = ""
        for msg in stripped:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text = extract_text_from_content(content)
            if text:
                conversation_text += f"\n{role}: {text[:5000]}\n"

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key decisions, code changes, and context needed to continue:\n\n"
            + conversation_text[:50000]
        )

        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": summary_prompt}],
        )

        summary = ""
        for block in response.content:
            if hasattr(block, "text"):
                summary += block.text

        # Build compacted messages
        compacted_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[Previous conversation summary]\n\n{summary}",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I understand the context. Let me continue from where we left off.",
                    }
                ],
            },
        ]

        new_state = AutoCompactState(
            compacted=True,
            turn_counter=0,
            consecutive_failures=0,
        )

        return {
            "compacted_messages": compacted_messages,
            "summary": summary,
            "state": new_state,
        }

    except Exception:
        new_state = AutoCompactState(
            compacted=state.compacted,
            turn_counter=state.turn_counter,
            consecutive_failures=state.consecutive_failures + 1,
        )
        return {
            "compacted_messages": messages,
            "summary": "",
            "state": new_state,
        }


def micro_compact_messages(messages: list[dict[str, Any]], max_tool_result_chars: int = 50000) -> list[dict[str, Any]]:
    """Truncate large tool results in messages to reduce context usage."""
    compacted: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            compacted.append(msg)
            continue

        new_content = []
        modified = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str) and len(result_content) > max_tool_result_chars:
                    half = max_tool_result_chars // 2
                    truncated = (
                        result_content[:half]
                        + f"\n\n... [truncated {len(result_content) - max_tool_result_chars} chars] ...\n\n"
                        + result_content[-half:]
                    )
                    new_content.append({**block, "content": truncated})
                    modified = True
                else:
                    new_content.append(block)
            else:
                new_content.append(block)

        if modified:
            compacted.append({**msg, "content": new_content})
        else:
            compacted.append(msg)

    return compacted
