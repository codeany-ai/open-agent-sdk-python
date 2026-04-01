"""MCP client - connect to MCP servers and discover tools."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from open_agent_sdk.types import (
    BaseTool,
    MCPConnection,
    McpHttpConfig,
    McpServerConfig,
    McpSseConfig,
    McpStdioConfig,
    ToolContext,
    ToolInputSchema,
    ToolResult,
)


class MCPToolWrapper(BaseTool):
    """Wraps an MCP tool as a standard ToolDefinition."""

    def __init__(self, server_name: str, tool_name: str, tool_desc: str, tool_schema: dict[str, Any], call_fn: Any):
        self._name = f"mcp__{server_name}__{tool_name}"
        self._description = tool_desc
        self._input_schema = ToolInputSchema(
            properties=tool_schema.get("properties", {}),
            required=tool_schema.get("required", []),
        )
        self._call_fn = call_fn

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            result = await self._call_fn(input)
            if isinstance(result, str):
                return ToolResult(tool_use_id="", content=result)
            return ToolResult(tool_use_id="", content=json.dumps(result, default=str))
        except Exception as e:
            return ToolResult(tool_use_id="", content=f"MCP tool error: {e}", is_error=True)


class StdioMCPConnection:
    """MCP connection via stdio transport."""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str]):
        self.name = name
        self._command = command
        self._args = args
        self._env = {**os.environ, **env}
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1

    async def connect(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self._command, *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )

        # Send initialize
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "open-agent-sdk-python", "version": "0.1.0"},
        })

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # Extract text from content
        content = result.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else json.dumps(result)

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("MCP connection not established")

        request_id = self._next_id
        self._next_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        line = json.dumps(request) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

        # Read response
        response_line = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=30
        )
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        line = json.dumps(notification) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    async def close(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()


async def connect_mcp_server(name: str, config: McpServerConfig) -> MCPConnection:
    """Connect to an MCP server and discover its tools."""
    tools: list[BaseTool] = []

    if isinstance(config, McpStdioConfig) or (isinstance(config, dict) and config.get("type") == "stdio"):
        command = config.command if isinstance(config, McpStdioConfig) else config.get("command", "")
        args = config.args if isinstance(config, McpStdioConfig) else config.get("args", [])
        env = config.env if isinstance(config, McpStdioConfig) else config.get("env", {})

        conn = StdioMCPConnection(name, command, args, env)
        await conn.connect()

        mcp_tools = await conn.list_tools()
        for mt in mcp_tools:
            tool_name = mt.get("name", "")
            tool_desc = mt.get("description", "")
            tool_schema = mt.get("inputSchema", {})

            async def make_call_fn(tn: str):
                async def call_fn(input_data: dict[str, Any]):
                    return await conn.call_tool(tn, input_data)
                return call_fn

            call_fn = await make_call_fn(tool_name)
            wrapper = MCPToolWrapper(name, tool_name, tool_desc, tool_schema, call_fn)
            tools.append(wrapper)

        async def close_fn():
            await conn.close()

        return MCPConnection(
            name=name,
            status="connected",
            tools=tools,
            close=close_fn,
        )

    elif isinstance(config, (McpSseConfig, McpHttpConfig)):
        # HTTP/SSE transport - simplified implementation
        url = config.url
        headers = config.headers

        return MCPConnection(
            name=name,
            status="connected",
            tools=[],
        )

    else:
        return MCPConnection(name=name, status="error", tools=[])


async def close_all_connections(connections: list[MCPConnection]) -> None:
    """Gracefully close all MCP connections."""
    for conn in connections:
        if conn.close:
            try:
                await conn.close()
            except Exception:
                pass
