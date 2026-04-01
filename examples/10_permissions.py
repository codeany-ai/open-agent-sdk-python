"""
Example 10: Permissions and Allowed Tools

Shows how to restrict which tools the agent can use.
Creates a read-only agent that can analyze but not modify code.

Run: python examples/10_permissions.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import query, AgentOptions, SDKMessageType


async def main():
    print("--- Example 10: Read-Only Agent ---\n")

    # Read-only agent: can only use Read, Glob, Grep
    async for message in query(
        prompt="Review the code in src/open_agent_sdk/agent.py for best practices. Be concise.",
        options=AgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
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
                            print(f'[{block["name"]}]')

        elif message.type == SDKMessageType.RESULT:
            print(f"\n--- {message.status} ---")


if __name__ == "__main__":
    asyncio.run(main())
