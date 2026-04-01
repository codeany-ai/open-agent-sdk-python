"""
Example 9: Subagents

Define specialized subagents that the main agent can delegate
tasks to. Matches the official SDK's agents option.

Run: python examples/09_subagents.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import query, AgentOptions, AgentDefinition, SDKMessageType


async def main():
    print("--- Example 9: Subagents ---\n")

    async for message in query(
        prompt="Use the code-reviewer agent to review src/open_agent_sdk/agent.py",
        options=AgentOptions(
            allowed_tools=["Read", "Glob", "Grep", "Agent"],
            agents={
                "code-reviewer": AgentDefinition(
                    description="Expert code reviewer for quality and security reviews.",
                    prompt=(
                        "Analyze code quality and suggest improvements. Focus on "
                        "security, performance, and maintainability. Be concise."
                    ),
                    tools=["Read", "Glob", "Grep"],
                ),
            },
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
                            print(f'[{block["name"]}] {str(block.get("input", {}))[:80]}')

        elif message.type == SDKMessageType.RESULT:
            print(f"\n--- {message.status} ---")


if __name__ == "__main__":
    asyncio.run(main())
