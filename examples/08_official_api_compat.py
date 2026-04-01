"""
Example 8: Official SDK-Compatible API

Demonstrates the query() function with the same API pattern
as open-agent-sdk. Drop-in compatible.

Run: python examples/08_official_api_compat.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import query, AgentOptions, SDKMessageType


async def main():
    print("--- Example 8: Official SDK-Compatible API ---\n")

    # Standard SDK query pattern
    async for message in query(
        prompt="What files are in this directory? Be brief.",
        options=AgentOptions(
            allowed_tools=["Bash", "Glob"],
            permission_mode="bypassPermissions",
        ),
    ):
        if message.type == SDKMessageType.ASSISTANT:
            msg = message.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            print(block["text"])
                        elif block.get("type") == "tool_use":
                            print(f'Tool: {block["name"]}')

        elif message.type == SDKMessageType.RESULT:
            print(f"\nDone: {message.status}")


if __name__ == "__main__":
    asyncio.run(main())
