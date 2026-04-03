"""Tests for LLM provider abstraction."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from open_agent_sdk.providers.types import (
    CreateMessageParams,
    CreateMessageResponse,
    LLMProvider,
)
from open_agent_sdk.providers.anthropic_provider import AnthropicProvider
from open_agent_sdk.providers.openai_provider import OpenAIProvider
from open_agent_sdk.providers.factory import create_provider


class TestCreateProvider:
    def test_anthropic(self):
        provider = create_provider("anthropic-messages", api_key="test")
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_type == "anthropic-messages"

    def test_openai(self):
        provider = create_provider("openai-completions", api_key="test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_type == "openai-completions"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_provider("invalid-type", api_key="test")


class TestAnthropicProvider:
    def test_api_type(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_type == "anthropic-messages"

    def test_has_client(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.client is not None


class TestOpenAIProvider:
    def test_api_type(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_type == "openai-completions"

    def test_default_base_url(self):
        provider = OpenAIProvider(api_key="test")
        assert "openai.com" in provider._base_url

    def test_custom_base_url(self):
        provider = OpenAIProvider(api_key="test", base_url="https://custom.api/v1")
        assert provider._base_url == "https://custom.api/v1"

    def test_strip_trailing_slash(self):
        provider = OpenAIProvider(api_key="test", base_url="https://api.example.com/v1/")
        assert not provider._base_url.endswith("/")

    def test_convert_messages_system(self):
        provider = OpenAIProvider(api_key="test")
        msgs = provider._convert_messages(
            "You are helpful",
            [{"role": "user", "content": "Hello"}],
        )
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        assert msgs[1]["role"] == "user"

    def test_convert_messages_tool_use(self):
        provider = OpenAIProvider(api_key="test")
        msgs = provider._convert_messages("", [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tc_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Let me check."
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["id"] == "tc_1"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "Bash"

    def test_convert_messages_tool_result(self):
        provider = OpenAIProvider(api_key="test")
        msgs = provider._convert_messages("", [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tc_1", "content": "file1.py\nfile2.py"},
                ],
            },
        ])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "tc_1"

    def test_convert_tools(self):
        from open_agent_sdk.providers.types import NormalizedTool
        provider = OpenAIProvider(api_key="test")
        tools = provider._convert_tools([
            NormalizedTool(name="Bash", description="Run shell", input_schema={
                "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"],
            }),
        ])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "Bash"

    def test_convert_response(self):
        provider = OpenAIProvider(api_key="test")
        data = {
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = provider._convert_response(data)
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello!"
        assert result.stop_reason == "end_turn"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    def test_convert_response_tool_calls(self):
        provider = OpenAIProvider(api_key="test")
        data = {
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command":"ls"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = provider._convert_response(data)
        assert len(result.content) == 1
        assert result.content[0]["type"] == "tool_use"
        assert result.content[0]["name"] == "Bash"
        assert result.content[0]["input"] == {"command": "ls"}
        assert result.stop_reason == "tool_use"

    def test_finish_reason_mapping(self):
        provider = OpenAIProvider(api_key="test")
        assert provider._map_finish_reason("stop") == "end_turn"
        assert provider._map_finish_reason("length") == "max_tokens"
        assert provider._map_finish_reason("tool_calls") == "tool_use"
        assert provider._map_finish_reason("content_filter") == "content_filter"
