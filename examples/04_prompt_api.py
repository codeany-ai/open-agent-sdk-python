"""
Example 4: Simple Prompt API

Uses the blocking prompt() method for quick one-shot queries.
No need to iterate over streaming events.

Run: python examples/04_prompt_api.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions


async def main():
    print("--- Example 4: Simple Prompt API ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5-20250514"),
        max_turns=5,
    ))

    result = await agent.prompt(
        'Use Bash to run `python --version` and `pip --version`, then tell me the versions.'
    )

    print(f"Answer: {result.text}")
    print(f"Turns: {result.num_turns}")
    print(f"Tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
    print(f"Duration: {result.duration_ms}ms")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
