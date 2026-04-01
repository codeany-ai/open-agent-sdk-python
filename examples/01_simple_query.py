"""Simple query example - basic agent usage."""

import asyncio
from open_agent_sdk import Agent, AgentOptions, SDKMessageType


async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        # api_key="your-api-key",  # or set ANTHROPIC_API_KEY env var
    ))

    # Streaming mode
    print("=== Streaming Mode ===")
    async for event in agent.query("What is 2 + 2? Reply briefly."):
        if event.type == SDKMessageType.ASSISTANT:
            print(f"Assistant: {event.text}")
        elif event.type == SDKMessageType.RESULT:
            print(f"\nResult: {event.text}")
            print(f"Turns: {event.num_turns}, Cost: ${event.total_cost:.4f}")

    # Convenience prompt mode
    print("\n=== Prompt Mode ===")
    result = await agent.prompt("What is the capital of France? Reply in one word.")
    print(f"Answer: {result.text}")
    print(f"Turns: {result.num_turns}, Duration: {result.duration_ms}ms")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
