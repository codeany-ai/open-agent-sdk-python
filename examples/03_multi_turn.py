"""
Example 3: Multi-Turn Conversation

Shows session persistence across multiple turns where the agent
remembers context from previous interactions.

Run: python examples/03_multi_turn.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions


async def main():
    print("--- Example 3: Multi-Turn Conversation ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5-20250514"),
        max_turns=5,
    ))

    # Turn 1: Create a file
    r1 = await agent.prompt('Create a file /tmp/hello.txt with "Hello World"')
    print(f"Turn 1: {r1.text}\n")

    # Turn 2: Read back the file
    r2 = await agent.prompt("Read back the file you just created")
    print(f"Turn 2: {r2.text}\n")

    # Turn 3: Modify the file
    r3 = await agent.prompt('Append " from Open Agent SDK" to that file')
    print(f"Turn 3: {r3.text}\n")

    # Turn 4: Verify
    r4 = await agent.prompt("Show me the final contents of the file")
    print(f"Turn 4: {r4.text}\n")

    print(f"Session messages: {len(agent.get_messages())}")
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
