"""Session persistence example."""

import asyncio
from open_agent_sdk import (
    Agent,
    AgentOptions,
    save_session,
    load_session,
    list_sessions,
    delete_session,
)


async def main():
    # Create agent with session persistence
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        persist_session=True,
        session_id="demo-session",
        allowed_tools=[],
    ))

    # First conversation
    result = await agent.prompt("Remember: the secret code is ALPHA-7.")
    print(f"Turn 1: {result.text}")

    # Close (persists session)
    await agent.close()

    # Resume session
    agent2 = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        resume="demo-session",
        allowed_tools=[],
    ))

    result = await agent2.prompt("What was the secret code?")
    print(f"Turn 2 (resumed): {result.text}")
    await agent2.close()

    # List sessions
    sessions = await list_sessions()
    print(f"\nSaved sessions: {len(sessions)}")

    # Cleanup
    await delete_session("demo-session")


if __name__ == "__main__":
    asyncio.run(main())
