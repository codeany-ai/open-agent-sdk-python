"""Tests for MCP client."""

import pytest
from open_agent_sdk.mcp.client import MCPToolWrapper, close_all_connections
from open_agent_sdk.types import MCPConnection, ToolContext


class TestMCPToolWrapper:
    @pytest.mark.asyncio
    async def test_name_prefix(self):
        async def call_fn(input):
            return "result"

        wrapper = MCPToolWrapper(
            server_name="myserver",
            tool_name="mytool",
            tool_desc="A tool",
            tool_schema={"properties": {"x": {"type": "string"}}, "required": ["x"]},
            call_fn=call_fn,
        )
        assert wrapper.name == "mcp__myserver__mytool"
        assert wrapper.description == "A tool"
        assert "x" in wrapper.input_schema.properties

    @pytest.mark.asyncio
    async def test_call(self):
        async def call_fn(input):
            return f"Got: {input.get('x')}"

        wrapper = MCPToolWrapper("s", "t", "desc", {"properties": {}}, call_fn)
        result = await wrapper.call({"x": "hello"}, ToolContext())
        assert "Got: hello" in result.content

    @pytest.mark.asyncio
    async def test_call_error(self):
        async def call_fn(input):
            raise RuntimeError("boom")

        wrapper = MCPToolWrapper("s", "t", "desc", {"properties": {}}, call_fn)
        result = await wrapper.call({}, ToolContext())
        assert result.is_error
        assert "boom" in result.content


class TestMCPConnection:
    def test_defaults(self):
        conn = MCPConnection(name="test")
        assert conn.name == "test"
        assert conn.status == "disconnected"
        assert conn.tools == []


class TestCloseAllConnections:
    @pytest.mark.asyncio
    async def test_close_multiple(self):
        closed = []

        async def close1():
            closed.append("conn1")

        async def close2():
            closed.append("conn2")

        connections = [
            MCPConnection(name="a", close=close1),
            MCPConnection(name="b", close=close2),
        ]
        await close_all_connections(connections)
        assert "conn1" in closed
        assert "conn2" in closed

    @pytest.mark.asyncio
    async def test_handles_close_error(self):
        async def bad_close():
            raise RuntimeError("close failed")

        connections = [MCPConnection(name="bad", close=bad_close)]
        await close_all_connections(connections)  # Should not raise
