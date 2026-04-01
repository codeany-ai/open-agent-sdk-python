"""
Example 6: MCP Server Integration

Connects to an MCP (Model Context Protocol) server and uses
its tools through the agent. This example uses the filesystem
MCP server as a demonstration.

Prerequisites:
  npm install -g @modelcontextprotocol/server-filesystem

Run: python examples/06_mcp_server.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions, McpStdioConfig


async def main():
    print("--- Example 6: MCP Server Integration ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5-20250514"),
        max_turns=10,
        mcp_servers={
            "filesystem": McpStdioConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            ),
        },
    ))

    print("Connecting to MCP filesystem server...\n")

    result = await agent.prompt(
        "Use the filesystem MCP tools to list files in /tmp. Be brief."
    )

    print(f"Answer: {result.text}")
    print(f"Turns: {result.num_turns}")

    await agent.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        if "not found" in str(e).lower() or "ENOENT" in str(e):
            print(
                "\nMCP server not found. Install it with:\n"
                "  npm install -g @modelcontextprotocol/server-filesystem\n"
            )
