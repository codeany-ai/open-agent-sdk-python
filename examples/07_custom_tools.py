"""
Example 7: Custom Tools

Shows how to define and use custom tools alongside built-in tools.

Run: python examples/07_custom_tools.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import create_agent, get_all_base_tools, AgentOptions, SDKMessageType
from open_agent_sdk.tool_helper import define_tool
from open_agent_sdk.types import ToolResult, ToolContext


async def weather_handler(input: dict, ctx: ToolContext) -> ToolResult:
    temps = {"tokyo": 22, "london": 14, "beijing": 25, "new york": 18, "paris": 16}
    city = input.get("city", "")
    temp = temps.get(city.lower(), 20)
    return ToolResult(tool_use_id="", content=f"Weather in {city}: {temp}°C, partly cloudy")


weather_tool = define_tool(
    name="GetWeather",
    description="Get current weather for a city. Returns temperature and conditions.",
    input_schema={
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
    handler=weather_handler,
    read_only=True,
    concurrency_safe=True,
)


async def calculator_handler(input: dict, ctx: ToolContext) -> ToolResult:
    try:
        expression = input.get("expression", "")
        result = eval(expression, {"__builtins__": {}})
        return ToolResult(tool_use_id="", content=f"{expression} = {result}")
    except Exception as e:
        return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)


calculator_tool = define_tool(
    name="Calculator",
    description="Evaluate a mathematical expression. Use ** for exponentiation.",
    input_schema={
        "properties": {"expression": {"type": "string", "description": "Math expression"}},
        "required": ["expression"],
    },
    handler=calculator_handler,
    read_only=True,
    concurrency_safe=True,
)


async def main():
    print("--- Example 7: Custom Tools ---\n")

    builtin_tools = get_all_base_tools()
    all_tools = [*builtin_tools, weather_tool, calculator_tool]

    agent = create_agent(AgentOptions(
        model=os.environ.get("CODEANY_MODEL", "claude-sonnet-4-5-20250514"),
        max_turns=10,
        tools=all_tools,
    ))

    print(f"Loaded {len(all_tools)} tools ({len(builtin_tools)} built-in + 2 custom)\n")

    async for event in agent.query(
        "What is the weather in Tokyo and London? Also calculate 2**10 * 3. Be brief."
    ):
        if event.type == SDKMessageType.ASSISTANT:
            msg = event.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            print(f'[{block["name"]}] {block.get("input", {})}')
                        elif block.get("type") == "text" and block.get("text", "").strip():
                            print(f"\n{block['text']}")

        elif event.type == SDKMessageType.RESULT:
            print(f"\n--- {event.status} ---")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
