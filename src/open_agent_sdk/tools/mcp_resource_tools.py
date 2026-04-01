"""MCP resource management tools."""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class ListMcpResourcesTool(BaseTool):
    _name = "ListMcpResources"
    _description = "List resources from MCP servers."
    _input_schema = ToolInputSchema(
        properties={
            "server": {"type": "string", "description": "MCP server name"},
        },
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _mcp_connections

        server_name = input.get("server", "")
        resources = []

        for conn in _mcp_connections:
            if server_name and getattr(conn, "name", "") != server_name:
                continue
            if hasattr(conn, "list_resources"):
                try:
                    res = await conn.list_resources()
                    resources.extend(res)
                except Exception:
                    pass

        if not resources:
            return ToolResult(tool_use_id="", content="No MCP resources found.")

        return ToolResult(tool_use_id="", content=json.dumps(resources, indent=2, default=str))


class ReadMcpResourceTool(BaseTool):
    _name = "ReadMcpResource"
    _description = "Read an MCP resource by URI."
    _input_schema = ToolInputSchema(
        properties={
            "uri": {"type": "string", "description": "Resource URI"},
            "server": {"type": "string", "description": "MCP server name"},
        },
        required=["uri"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        uri = input.get("uri", "")
        if not uri:
            return ToolResult(tool_use_id="", content="Error: uri is required", is_error=True)

        from open_agent_sdk.tools import _mcp_connections

        for conn in _mcp_connections:
            if hasattr(conn, "read_resource"):
                try:
                    content = await conn.read_resource(uri)
                    return ToolResult(tool_use_id="", content=str(content))
                except Exception:
                    continue

        return ToolResult(tool_use_id="", content=f"Resource not found: {uri}", is_error=True)
