"""
Example 1: Simple Query with Streaming

Demonstrates the basic create_agent() + query() flow with real-time event streaming.

Run: python examples/01_simple_query.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions, SDKMessageType


async def main():
    print("--- Example 1: Simple Query with Streaming ---\n")

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5"),
        max_turns=10,
    ))

    async for event in agent.query("What files are in this directory? Use Glob to find them. Be brief."):
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
            print(f"  → {content[:120]}{'...' if len(content) > 120 else ''}")

        elif event.type == SDKMessageType.RESULT:
            usage = event.total_usage
            print(f"\n--- {event.status} ---")
            if usage:
                print(f"Turns: {event.num_turns}, "
                      f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out, "
                      f"Cost: ${event.total_cost:.4f}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
