"""Tests for in-process SDK MCP server."""

import pytest
from open_agent_sdk.sdk_mcp_server import (
    McpSdkServerConfig,
    SdkMcpToolWrapper,
    create_sdk_mcp_server,
    is_sdk_server_config,
)
from open_agent_sdk.tools import BashTool, FileReadTool
from open_agent_sdk.types import ToolContext


class TestCreateSdkMcpServer:
    def test_creates_config(self):
        config = create_sdk_mcp_server("test-server", tools=[BashTool()])
        assert isinstance(config, McpSdkServerConfig)
        assert config.name == "test-server"
        assert config.type == "sdk"
        assert len(config.tools) == 1

    def test_tool_name_prefix(self):
        config = create_sdk_mcp_server("myserver", tools=[BashTool(), FileReadTool()])
        names = [t.name for t in config.tools]
        assert "mcp__myserver__Bash" in names
        assert "mcp__myserver__Read" in names

    def test_empty_tools(self):
        config = create_sdk_mcp_server("empty")
        assert config.tools == []


class TestSdkMcpToolWrapper:
    @pytest.mark.asyncio
    async def test_delegates_to_inner(self):
        inner = BashTool()
        wrapper = SdkMcpToolWrapper("server", inner)
        assert wrapper.name == "mcp__server__Bash"
        assert wrapper.description == inner.description

    @pytest.mark.asyncio
    async def test_call_delegates(self, tmp_path):
        inner = FileReadTool()
        wrapper = SdkMcpToolWrapper("server", inner)

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\n")

        result = await wrapper.call(
            {"file_path": str(test_file)},
            ToolContext(cwd=str(tmp_path)),
        )
        assert "hello" in result.content


class TestIsSdkServerConfig:
    def test_true_for_config(self):
        config = create_sdk_mcp_server("test")
        assert is_sdk_server_config(config) is True

    def test_false_for_dict(self):
        assert is_sdk_server_config({"type": "stdio"}) is False

    def test_false_for_string(self):
        assert is_sdk_server_config("not a config") is False
