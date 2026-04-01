"""In-process MCP server factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from open_agent_sdk.types import BaseTool, MCPConnection, ToolContext, ToolInputSchema, ToolResult


@dataclass
class McpSdkServerConfig:
    type: str = "sdk"
    name: str = ""
    version: str = "0.1.0"
    tools: list[BaseTool] = field(default_factory=list)


class SdkMcpToolWrapper(BaseTool):
    """Wraps an SDK tool as an MCP-namespaced tool."""

    def __init__(self, server_name: str, tool: BaseTool):
        self._name = f"mcp__{server_name}__{tool.name}"
        self._description = tool.description
        self._input_schema = tool.input_schema
        self._inner_tool = tool

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return self._inner_tool.is_read_only(input)

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return self._inner_tool.is_concurrency_safe(input)

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        return await self._inner_tool.call(input, context)


def create_sdk_mcp_server(
    name: str,
    version: str = "0.1.0",
    tools: list[BaseTool] | None = None,
) -> McpSdkServerConfig:
    """Create an in-process MCP server config.

    Tools will be prefixed with mcp__{name}__ when registered.
    """
    wrapped_tools = []
    for t in (tools or []):
        wrapped_tools.append(SdkMcpToolWrapper(name, t))

    return McpSdkServerConfig(
        name=name,
        version=version,
        tools=wrapped_tools,
    )


def is_sdk_server_config(config: Any) -> bool:
    """Check if a config is an SDK MCP server config."""
    return isinstance(config, McpSdkServerConfig)
