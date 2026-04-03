"""
Example 14: OpenAI-Compatible Models

Shows how to use the SDK with OpenAI's API or any OpenAI-compatible
endpoint (e.g., DeepSeek, Qwen, vLLM, Ollama).

Environment variables:
  CODEANY_API_KEY=sk-...                     # Your API key
  CODEANY_BASE_URL=https://api.openai.com/v1 # Optional
  CODEANY_API_TYPE=openai-completions         # Optional, auto-detected

Run: python examples/14_openai_compat.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, AgentOptions, SDKMessageType


async def main():
    print("--- Example 14: OpenAI-Compatible Models ---\n")

    # Option 1: Explicit api_type
    agent = create_agent(AgentOptions(
        api_type="openai-completions",
        model=os.environ.get("CODEANY_MODEL", "gpt-4o"),
        api_key=os.environ.get("CODEANY_API_KEY", ""),
        base_url=os.environ.get("CODEANY_BASE_URL", "https://api.openai.com/v1"),
        max_turns=5,
    ))

    print(f"API Type: {agent.get_api_type()}")
    print(f"Model: {os.environ.get('CODEANY_MODEL', 'gpt-4o')}\n")

    # Option 2: Auto-detected from model name (uncomment to try)
    # agent = create_agent(AgentOptions(model="gpt-4o"))

    # Option 3: DeepSeek example (uncomment to try)
    # agent = create_agent(AgentOptions(
    #     model="deepseek-chat",
    #     api_key=os.environ.get("CODEANY_API_KEY", ""),
    #     base_url="https://api.deepseek.com/v1",
    # ))

    # Option 4: Via environment variables only
    # CODEANY_API_TYPE=openai-completions
    # CODEANY_MODEL=gpt-4o
    # CODEANY_API_KEY=sk-...
    # agent = create_agent()

    async for event in agent.query("What is 2+2? Reply in one sentence."):
        if event.type == SDKMessageType.ASSISTANT:
            msg = event.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            print(f"Assistant: {block['text']}")
                        elif block.get("type") == "tool_use":
                            print(f'[Tool: {block["name"]}] {block.get("input", {})}')
        elif event.type == SDKMessageType.RESULT:
            usage = event.total_usage
            tokens = f"{usage.input_tokens}+{usage.output_tokens}" if usage else "?"
            print(f"\n--- {event.status} ({tokens} tokens) ---")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
