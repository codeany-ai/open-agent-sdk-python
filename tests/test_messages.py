"""Tests for message utilities."""

import pytest
from open_agent_sdk.utils.messages import (
    create_user_message,
    create_assistant_message,
    normalize_messages_for_api,
    strip_images_from_messages,
    extract_text_from_content,
    truncate_text,
)


class TestCreateUserMessage:
    def test_string_content(self):
        msg = create_user_message("Hello")
        assert msg["role"] == "user"
        assert msg["content"] == [{"type": "text", "text": "Hello"}]
        assert "uuid" in msg
        assert "timestamp" in msg

    def test_list_content(self):
        blocks = [{"type": "text", "text": "Hi"}, {"type": "image", "source": {}}]
        msg = create_user_message(blocks)
        assert msg["content"] == blocks

    def test_custom_uuid(self):
        msg = create_user_message("Test", uuid_str="custom-id")
        assert msg["uuid"] == "custom-id"


class TestCreateAssistantMessage:
    def test_basic(self):
        msg = create_assistant_message([{"type": "text", "text": "Response"}])
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Response"}]

    def test_with_usage(self):
        msg = create_assistant_message(
            [{"type": "text", "text": "Hi"}],
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert msg["usage"]["input_tokens"] == 100


class TestNormalizeMessages:
    def test_empty(self):
        assert normalize_messages_for_api([]) == []

    def test_single_user(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = normalize_messages_for_api(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_ensures_starts_with_user(self):
        msgs = [{"role": "assistant", "content": "Hi"}]
        result = normalize_messages_for_api(msgs)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_merges_consecutive_same_role(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        result = normalize_messages_for_api(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_alternating_roles(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = normalize_messages_for_api(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["user", "assistant", "user"]

    def test_inserts_filler_for_non_alternating(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
        ]
        result = normalize_messages_for_api(msgs)
        roles = [m["role"] for m in result]
        # Should alternate
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1]


class TestStripImages:
    def test_removes_images(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image", "source": {"type": "base64"}},
                ],
            }
        ]
        result = strip_images_from_messages(msgs)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1]["type"] == "text"
        assert "removed" in result[0]["content"][1]["text"].lower()

    def test_preserves_non_image(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        result = strip_images_from_messages(msgs)
        assert result[0]["content"][0]["text"] == "Hello"


class TestExtractText:
    def test_string_content(self):
        assert extract_text_from_content("Hello") == "Hello"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        result = extract_text_from_content(content)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_tool_result_content(self):
        content = [
            {"type": "tool_result", "content": "Result text"},
        ]
        result = extract_text_from_content(content)
        assert "Result text" in result

    def test_empty(self):
        assert extract_text_from_content("") == ""
        assert extract_text_from_content([]) == ""


class TestTruncateText:
    def test_short_text(self):
        assert truncate_text("hello", 100) == "hello"

    def test_long_text(self):
        text = "a" * 200
        result = truncate_text(text, 100)
        assert len(result) < 200
        assert "truncated" in result

    def test_exact_limit(self):
        text = "a" * 100
        assert truncate_text(text, 100) == text
