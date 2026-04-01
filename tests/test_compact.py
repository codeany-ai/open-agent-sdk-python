"""Tests for auto-compaction logic."""

import pytest
from open_agent_sdk.utils.compact import (
    AutoCompactState,
    create_auto_compact_state,
    micro_compact_messages,
    should_auto_compact,
)


class TestAutoCompactState:
    def test_defaults(self):
        state = create_auto_compact_state()
        assert state.compacted is False
        assert state.turn_counter == 0
        assert state.consecutive_failures == 0


class TestShouldAutoCompact:
    def test_small_conversation_no_compact(self):
        messages = [{"role": "user", "content": "hello"}]
        state = create_auto_compact_state()
        assert should_auto_compact(messages, "claude-sonnet-4-5-20250514", state) is False

    def test_large_conversation_triggers_compact(self):
        # Create a message that would exceed the threshold
        large_text = "x" * 1_000_000  # ~250k tokens at 4 chars/token
        messages = [{"role": "user", "content": large_text}]
        state = create_auto_compact_state()
        assert should_auto_compact(messages, "claude-sonnet-4-5-20250514", state) is True

    def test_too_many_failures_disables(self):
        large_text = "x" * 1_000_000
        messages = [{"role": "user", "content": large_text}]
        state = AutoCompactState(consecutive_failures=3)
        assert should_auto_compact(messages, "claude-sonnet-4-5-20250514", state) is False


class TestMicroCompact:
    def test_truncates_large_tool_results(self):
        large_result = "x" * 100_000
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": large_result},
                ],
            }
        ]
        result = micro_compact_messages(messages, max_tool_result_chars=1000)
        content = result[0]["content"][0]["content"]
        assert len(content) < len(large_result)
        assert "truncated" in content

    def test_small_results_unchanged(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": "small"},
                ],
            }
        ]
        result = micro_compact_messages(messages)
        assert result[0]["content"][0]["content"] == "small"

    def test_non_tool_result_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        result = micro_compact_messages(messages)
        assert result[0]["content"][0]["text"] == "Hello"

    def test_string_content_unchanged(self):
        messages = [{"role": "user", "content": "plain text"}]
        result = micro_compact_messages(messages)
        assert result[0]["content"] == "plain text"
