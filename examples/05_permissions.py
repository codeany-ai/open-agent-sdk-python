"""Permission system example."""

import asyncio
from open_agent_sdk import (
    Agent,
    AgentOptions,
    CanUseToolResult,
    PermissionBehavior,
    SDKMessageType,
)


async def my_permission_handler(tool, input):
    """Custom permission handler - deny destructive bash commands."""
    if tool.name == "Bash":
        command = input.get("command", "")
        dangerous = ["rm ", "dd ", "mkfs", "> /dev/"]
        for d in dangerous:
            if d in command:
                return CanUseToolResult(
                    behavior=PermissionBehavior.DENY,
                    message=f"Dangerous command blocked: {command}",
                )
    return CanUseToolResult(behavior=PermissionBehavior.ALLOW)


async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        can_use_tool=my_permission_handler,
        allowed_tools=["Bash", "Read"],
    ))

    async for event in agent.query("Run: echo hello && rm -rf /"):
        if event.type == SDKMessageType.TOOL_RESULT:
            status = "ERROR" if event.is_error else "OK"
            print(f"  [{status}] {event.tool_name}: {event.result_content[:100]}")
        elif event.type == SDKMessageType.RESULT:
            print(f"\nResult: {event.text[:200]}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
