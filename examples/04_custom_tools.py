"""Custom tools example - define tools with Pydantic models."""

import asyncio
from pydantic import BaseModel

from open_agent_sdk import Agent, AgentOptions, SDKMessageType
from open_agent_sdk.tool_helper import CallToolResult, ToolAnnotations, tool


# Define a custom tool with Pydantic
class CalculateInput(BaseModel):
    expression: str


async def calculate_handler(input: CalculateInput, ctx) -> CallToolResult:
    try:
        # WARNING: eval is unsafe in production! Use a proper math parser.
        result = eval(input.expression, {"__builtins__": {}})
        return CallToolResult(content=[{"type": "text", "text": f"Result: {result}"}])
    except Exception as e:
        return CallToolResult(content=[{"type": "text", "text": f"Error: {e}"}], is_error=True)


calculator = tool(
    "Calculator",
    "Evaluate a mathematical expression",
    CalculateInput,
    calculate_handler,
    annotations=ToolAnnotations(read_only_hint=True),
)


async def main():
    agent = Agent(AgentOptions(
        model="claude-sonnet-4-5-20250514",
        tools=[calculator],
        allowed_tools=["Calculator"],
    ))

    result = await agent.prompt("What is (17 * 23) + (45 / 9)?")
    print(f"Answer: {result.text}")
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
