"""Multi-tool example - agent using multiple tools."""

import asyncio
from open_agent_sdk import Agent, AgentOptions, SDKMessageType


async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        cwd=".",
        allowed_tools=["Bash", "Read", "Glob", "Grep"],
    ))

    async for event in agent.query("List all Python files in the current directory and show a summary."):
        if event.type == SDKMessageType.ASSISTANT:
            print(f"Assistant: {event.text[:200]}")
        elif event.type == SDKMessageType.TOOL_RESULT:
            print(f"  Tool [{event.tool_name}]: {event.result_content[:100]}...")
        elif event.type == SDKMessageType.RESULT:
            print(f"\nDone! Turns: {event.num_turns}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
