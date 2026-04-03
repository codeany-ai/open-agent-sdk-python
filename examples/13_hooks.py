"""
Example 13: Hooks

Shows how to use lifecycle hooks to intercept agent behavior.
Hooks fire at key points: session start/end, before/after tool use,
compaction, etc.

Run: python examples/13_hooks.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import (
    create_agent,
    AgentOptions,
    SDKMessageType,
    create_hook_registry,
    HookEvent,
    HookDefinition,
    HookInput,
    HookOutput,
)


async def main():
    print("--- Example 13: Hooks ---\n")

    # Create a hook registry with custom handlers
    registry = create_hook_registry()

    async def on_pre_tool(input: HookInput) -> HookOutput:
        print(f"[Hook] About to use tool: {input.tool_name}")
        # You can block a tool by returning HookOutput(block=True)
        return HookOutput()

    async def on_post_tool(input: HookInput) -> HookOutput:
        output_preview = str(input.tool_output)[:100] if input.tool_output else ""
        print(f"[Hook] Tool {input.tool_name} completed: {output_preview}...")
        return HookOutput()

    async def on_stop(input: HookInput) -> HookOutput:
        print("[Hook] Agent loop completed")
        return HookOutput()

    registry.register(HookEvent.PRE_TOOL_USE, HookDefinition(handler=on_pre_tool))
    registry.register(HookEvent.POST_TOOL_USE, HookDefinition(handler=on_post_tool))
    registry.register(HookEvent.STOP, HookDefinition(handler=on_stop))

    # Create agent
    agent = create_agent(AgentOptions(max_turns=5))

    async for event in agent.query("What files are in the current directory? Be brief."):
        if event.type == SDKMessageType.ASSISTANT:
            msg = event.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            print(f"\nAssistant: {block['text'][:200]}")
        elif event.type == SDKMessageType.RESULT:
            print(f"\n--- {event.status} ---")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
