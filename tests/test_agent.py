"""Tests for Agent high-level API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from open_agent_sdk.agent import Agent, create_agent
from open_agent_sdk.types import (
    AgentOptions,
    PermissionMode,
    SDKMessageType,
    SDKResultStatus,
    ThinkingConfig,
)


def make_mock_response(text="Hello!", stop_reason="end_turn"):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.model = "claude-sonnet-4-5-20250514"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response.content = [text_block]
    usage = MagicMock()
    usage.input_tokens = 50
    usage.output_tokens = 25
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    response.usage = usage
    return response


class TestAgent:
    def test_create_agent(self):
        agent = create_agent()
        assert isinstance(agent, Agent)

    def test_create_with_options(self):
        agent = create_agent(AgentOptions(model="claude-opus-4-6", max_turns=20))
        assert agent._options.model == "claude-opus-4-6"
        assert agent._options.max_turns == 20

    def test_session_id(self):
        agent = Agent()
        sid = agent.get_session_id()
        assert sid
        assert len(sid) > 10  # UUID

    def test_custom_session_id(self):
        agent = Agent(AgentOptions(session_id="my-session"))
        assert agent.get_session_id() == "my-session"

    def test_clear(self):
        agent = Agent()
        agent._history = [{"role": "user", "content": "test"}]
        agent.clear()
        assert agent.get_messages() == []

    def test_get_messages_returns_copy(self):
        agent = Agent()
        agent._history = [{"role": "user", "content": "test"}]
        msgs = agent.get_messages()
        msgs.clear()
        assert len(agent._history) == 1

    @pytest.mark.asyncio
    async def test_set_model(self):
        agent = Agent()
        await agent.set_model("claude-opus-4-6")
        assert agent._options.model == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_set_permission_mode(self):
        agent = Agent()
        await agent.set_permission_mode(PermissionMode.DEFAULT)
        assert agent._options.permission_mode == PermissionMode.DEFAULT

    @pytest.mark.asyncio
    async def test_set_thinking_tokens(self):
        agent = Agent()
        await agent.set_max_thinking_tokens(5000)
        assert agent._options.thinking is not None
        assert agent._options.thinking.budget_tokens == 5000

        await agent.set_max_thinking_tokens(None)
        assert agent._options.thinking is None

    @pytest.mark.asyncio
    async def test_query_streaming(self):
        agent = Agent(AgentOptions(
            api_key="test-key",
            allowed_tools=["Bash"],
        ))

        mock_response = make_mock_response("Test response")

        with patch("open_agent_sdk.engine.QueryEngine.submit_message") as mock_submit:
            from open_agent_sdk.types import SDKMessage

            async def fake_submit(prompt):
                yield SDKMessage(type=SDKMessageType.SYSTEM, subtype=None)
                yield SDKMessage(type=SDKMessageType.ASSISTANT, text="Test response")
                yield SDKMessage(
                    type=SDKMessageType.RESULT,
                    status=SDKResultStatus.SUCCESS,
                    text="Test response",
                    num_turns=1,
                )

            mock_submit.return_value = fake_submit("test")

            events = []
            async for event in agent.query("Hello"):
                events.append(event)

            assert len(events) == 3
            types = [e.type for e in events]
            assert SDKMessageType.RESULT in types

    @pytest.mark.asyncio
    async def test_prompt_convenience(self):
        agent = Agent(AgentOptions(api_key="test-key", allowed_tools=[]))

        with patch("open_agent_sdk.engine.QueryEngine.submit_message") as mock_submit:
            from open_agent_sdk.types import SDKMessage, TokenUsage

            async def fake_submit(prompt):
                yield SDKMessage(
                    type=SDKMessageType.ASSISTANT,
                    text="Response text",
                )
                yield SDKMessage(
                    type=SDKMessageType.RESULT,
                    status=SDKResultStatus.SUCCESS,
                    text="Response text",
                    num_turns=1,
                    total_usage=TokenUsage(input_tokens=10, output_tokens=5),
                    total_cost=0.001,
                )

            mock_submit.return_value = fake_submit("test")

            result = await agent.prompt("Hello")
            assert result.text == "Response text"
            assert result.num_turns == 1
            assert result.cost == 0.001

    @pytest.mark.asyncio
    async def test_close(self):
        agent = Agent()
        await agent.close()  # Should not raise


class TestAgentOptions:
    def test_default_tools_empty(self):
        opts = AgentOptions()
        assert opts.tools == []

    def test_environment_vars(self):
        opts = AgentOptions(env={"MY_VAR": "value"})
        assert opts.env["MY_VAR"] == "value"

    def test_mcp_servers(self):
        from open_agent_sdk.types import McpStdioConfig

        opts = AgentOptions(
            mcp_servers={
                "test": McpStdioConfig(command="echo", args=["hi"]),
            }
        )
        assert "test" in opts.mcp_servers
