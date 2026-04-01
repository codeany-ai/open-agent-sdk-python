"""
Example 5: Custom System Prompt

Shows how to customize the agent's behavior with a system prompt.

Run: python examples/05_custom_system_prompt.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions


async def main():
    print("--- Example 5: Custom System Prompt ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5-20250514"),
        max_turns=5,
        system_prompt=(
            "You are a senior code reviewer. When asked to review code, focus on: "
            "1) Security issues, 2) Performance concerns, 3) Maintainability. "
            "Be concise and use bullet points."
        ),
    ))

    result = await agent.prompt("Read src/open_agent_sdk/agent.py and give a brief code review.")
    print(result.text)

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
