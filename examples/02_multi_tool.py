"""
Example 2: Multi-Tool Orchestration

Demonstrates the agent using Glob, Bash, and Read tools autonomously
to accomplish a multi-step task.

Run: python examples/02_multi_tool.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions, SDKMessageType


async def main():
    print("--- Example 2: Multi-Tool Orchestration ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5"),
        max_turns=10,
        allowed_tools=["Glob", "Bash", "Read"],
    ))

    async for event in agent.query(
        "Find all Python files in this project, count how many there are, "
        "and show the first 5 lines of pyproject.toml. Be brief."
    ):
        if event.type == SDKMessageType.ASSISTANT:
            msg = event.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            print(block["text"])
                        elif block.get("type") == "tool_use":
                            print(f'[{block["name"]}] {str(block.get("input", {}))[:80]}')

        elif event.type == SDKMessageType.TOOL_RESULT:
            content = event.result_content or ""
            print(f"  → {content[:150]}{'...' if len(content) > 150 else ''}")

        elif event.type == SDKMessageType.RESULT:
            print(f"\n--- {event.status} ---")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
