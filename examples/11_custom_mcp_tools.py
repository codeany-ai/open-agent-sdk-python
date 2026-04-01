"""
Example 11: Custom Tools with tool() + create_sdk_mcp_server()

Shows the Pydantic-based tool() helper and in-process MCP server creation.
This is the recommended way to add custom tools.

Run: python examples/11_custom_mcp_tools.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from open_agent_sdk import query, create_sdk_mcp_server, AgentOptions, SDKMessageType
from open_agent_sdk.tool_helper import tool, CallToolResult, ToolAnnotations


# Define tools using Pydantic models for type-safe input validation

class TemperatureInput(BaseModel):
    city: str = Field(description="City name")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")


async def temperature_handler(input: TemperatureInput, ctx) -> CallToolResult:
    temps = {"tokyo": 22, "london": 14, "paris": 16, "new york": 18, "beijing": 25}
    temp_c = temps.get(input.city.lower(), 20)
    if input.unit == "fahrenheit":
        temp = temp_c * 9 / 5 + 32
        symbol = "°F"
    else:
        temp = temp_c
        symbol = "°C"
    return CallToolResult(
        content=[{"type": "text", "text": f"Temperature in {input.city}: {temp}{symbol}"}]
    )


get_temperature = tool(
    "get_temperature",
    "Get the current temperature at a location",
    TemperatureInput,
    temperature_handler,
    annotations=ToolAnnotations(read_only_hint=True),
)


class ConvertInput(BaseModel):
    value: float = Field(description="Value to convert")
    from_unit: str = Field(description="Source unit")
    to_unit: str = Field(description="Target unit")


async def convert_handler(input: ConvertInput, ctx) -> CallToolResult:
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
    }
    fn = conversions.get((input.from_unit, input.to_unit))
    if not fn:
        return CallToolResult(
            content=[{"type": "text", "text": f"Cannot convert from {input.from_unit} to {input.to_unit}"}],
            is_error=True,
        )
    result = fn(input.value)
    return CallToolResult(
        content=[{"type": "text", "text": f"{input.value} {input.from_unit} = {result:.2f} {input.to_unit}"}]
    )


convert_units = tool("convert_units", "Convert between measurement units", ConvertInput, convert_handler)

# Bundle tools into an in-process MCP server
utility_server = create_sdk_mcp_server("utilities", version="1.0.0", tools=[get_temperature, convert_units])


async def main():
    print("--- Example 11: Custom MCP Tools (tool + create_sdk_mcp_server) ---\n")

    async for message in query(
        prompt="What is the temperature in Tokyo and Paris? Also convert 10 km to miles. Be brief.",
        options=AgentOptions(
            # Use tools from the in-process MCP server
            tools=list(utility_server.tools),
            allowed_tools=[t.name for t in utility_server.tools],
            permission_mode="bypassPermissions",
        ),
    ):
        if message.type == SDKMessageType.ASSISTANT:
            msg = message.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            print(block["text"])
                        elif block.get("type") == "tool_use":
                            print(f'[{block["name"]}] {block.get("input", {})}')

        elif message.type == SDKMessageType.RESULT:
            print(f"\nDone: {message.status} (cost: ${message.total_cost:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
