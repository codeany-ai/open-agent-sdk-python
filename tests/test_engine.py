"""Tests for QueryEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from open_agent_sdk.engine import QueryEngine, QueryEngineConfig
from open_agent_sdk.types import (
    BaseTool,
    SDKMessageType,
    SDKResultStatus,
    TokenUsage,
    ToolContext,
    ToolInputSchema,
    ToolResult,
)


class EchoTool(BaseTool):
    _name = "Echo"
    _description = "Echoes input"
    _input_schema = ToolInputSchema(
        properties={"text": {"type": "string"}},
        required=["text"],
    )

    def is_read_only(self, input=None):
        return True

    def is_concurrency_safe(self, input=None):
        return True

    async def call(self, input, context):
        return ToolResult(tool_use_id="", content=f"Echo: {input.get('text', '')}")


def make_mock_client():
    """Create a mock Anthropic client."""
    client = AsyncMock()
    return client


def make_api_response(text="Hello!", tool_use=None, stop_reason="end_turn"):
    """Create a mock API response."""
    response = MagicMock()
    response.stop_reason = stop_reason
    response.model = "claude-sonnet-4-5-20250514"

    content = []
    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        content.append(text_block)

    if tool_use:
        for tu in tool_use:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tu["id"]
            block.name = tu["name"]
            block.input = tu["input"]
            content.append(block)

    response.content = content

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    response.usage = usage

    return response


class TestQueryEngine:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        client = make_mock_client()
        client.messages.create = AsyncMock(return_value=make_api_response("Hello world!"))

        config = QueryEngineConfig(client=client, model="test", tools=[])
        engine = QueryEngine(config)

        events = []
        async for event in engine.submit_message("Hi"):
            events.append(event)

        # Should have: system(init), assistant, result
        types = [e.type for e in events]
        assert SDKMessageType.SYSTEM in types
        assert SDKMessageType.ASSISTANT in types
        assert SDKMessageType.RESULT in types

        result_event = [e for e in events if e.type == SDKMessageType.RESULT][0]
        assert result_event.status == SDKResultStatus.SUCCESS
        assert "Hello world!" in result_event.text

    @pytest.mark.asyncio
    async def test_tool_call_loop(self):
        client = make_mock_client()

        # First call returns tool use, second returns text
        tool_response = make_api_response(
            text="Let me echo that.",
            tool_use=[{"id": "tu_1", "name": "Echo", "input": {"text": "test"}}],
        )
        final_response = make_api_response("Done! The echo said: Echo: test")

        client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

        echo = EchoTool()
        config = QueryEngineConfig(client=client, model="test", tools=[echo])
        engine = QueryEngine(config)

        events = []
        async for event in engine.submit_message("Echo something"):
            events.append(event)

        types = [e.type for e in events]
        assert SDKMessageType.TOOL_RESULT in types
        assert SDKMessageType.RESULT in types

        # Check tool result
        tool_events = [e for e in events if e.type == SDKMessageType.TOOL_RESULT]
        assert len(tool_events) == 1
        assert "Echo: test" in tool_events[0].result_content

        result = [e for e in events if e.type == SDKMessageType.RESULT][0]
        assert result.status == SDKResultStatus.SUCCESS
        assert result.num_turns == 2

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        client = make_mock_client()

        # Always returns a tool call
        tool_response = make_api_response(
            text="Calling echo...",
            tool_use=[{"id": "tu_1", "name": "Echo", "input": {"text": "loop"}}],
        )
        client.messages.create = AsyncMock(return_value=tool_response)

        echo = EchoTool()
        config = QueryEngineConfig(client=client, model="test", tools=[echo], max_turns=3)
        engine = QueryEngine(config)

        events = []
        async for event in engine.submit_message("Keep going"):
            events.append(event)

        result = [e for e in events if e.type == SDKMessageType.RESULT][0]
        assert result.status == SDKResultStatus.ERROR_MAX_TURNS
        assert result.num_turns == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        client = make_mock_client()

        tool_response = make_api_response(
            text="Using unknown tool...",
            tool_use=[{"id": "tu_1", "name": "NonexistentTool", "input": {}}],
        )
        final_response = make_api_response("OK I see the error")
        client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

        config = QueryEngineConfig(client=client, model="test", tools=[])
        engine = QueryEngine(config)

        events = []
        async for event in engine.submit_message("Use nonexistent"):
            events.append(event)

        tool_events = [e for e in events if e.type == SDKMessageType.TOOL_RESULT]
        assert len(tool_events) == 1
        assert tool_events[0].is_error

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        client = make_mock_client()
        client.messages.create = AsyncMock(return_value=make_api_response("OK"))

        config = QueryEngineConfig(client=client, model="test", tools=[])
        engine = QueryEngine(config)

        async for _ in engine.submit_message("Hi"):
            pass

        assert engine.total_usage.input_tokens == 100
        assert engine.total_usage.output_tokens == 50
        assert engine.total_cost > 0

    @pytest.mark.asyncio
    async def test_messages_preserved(self):
        client = make_mock_client()
        client.messages.create = AsyncMock(return_value=make_api_response("Response"))

        config = QueryEngineConfig(client=client, model="test", tools=[])
        engine = QueryEngine(config)

        async for _ in engine.submit_message("Hello"):
            pass

        messages = engine.messages
        assert len(messages) >= 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_permission_deny(self):
        client = make_mock_client()

        tool_response = make_api_response(
            text="Calling echo...",
            tool_use=[{"id": "tu_1", "name": "Echo", "input": {"text": "test"}}],
        )
        final_response = make_api_response("Permission denied, OK")
        client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

        from open_agent_sdk.types import CanUseToolResult, PermissionBehavior

        async def deny_all(tool, input):
            return CanUseToolResult(behavior=PermissionBehavior.DENY, message="Not allowed")

        echo = EchoTool()
        config = QueryEngineConfig(client=client, model="test", tools=[echo], can_use_tool=deny_all)
        engine = QueryEngine(config)

        events = []
        async for event in engine.submit_message("Echo"):
            events.append(event)

        tool_events = [e for e in events if e.type == SDKMessageType.TOOL_RESULT]
        assert len(tool_events) == 1
        assert tool_events[0].is_error
        assert "Not allowed" in tool_events[0].result_content
