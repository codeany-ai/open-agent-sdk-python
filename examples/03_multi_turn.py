"""Multi-turn conversation example."""

import asyncio
from open_agent_sdk import Agent, AgentOptions, SDKMessageType


async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        allowed_tools=[],
    ))

    prompts = [
        "Remember the number 42.",
        "What number did I ask you to remember?",
        "Multiply it by 2.",
    ]

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await agent.prompt(prompt)
        print(f"Assistant: {result.text}")

    print(f"\nTotal messages in conversation: {len(agent.get_messages())}")
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
