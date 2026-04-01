"""Tests for tool helper (Pydantic-based tool creation)."""

import pytest
from pydantic import BaseModel

from open_agent_sdk.tool_helper import (
    CallToolResult,
    PydanticTool,
    ToolAnnotations,
    define_tool,
    tool,
    tool_to_api_schema,
)
from open_agent_sdk.types import ToolContext, ToolResult


class AddInput(BaseModel):
    a: int
    b: int


class TestPydanticTool:
    @pytest.mark.asyncio
    async def test_create_and_call(self):
        async def handler(input: AddInput, ctx) -> CallToolResult:
            return CallToolResult(
                content=[{"type": "text", "text": str(input.a + input.b)}]
            )

        t = tool("add", "Add two numbers", AddInput, handler)
        assert t.name == "add"
        assert t.description == "Add two numbers"
        assert "a" in t.input_schema.properties
        assert "b" in t.input_schema.properties

        result = await t.call({"a": 3, "b": 4}, ToolContext())
        assert "7" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_validation_error(self):
        async def handler(input: AddInput, ctx) -> CallToolResult:
            return CallToolResult(content=[{"type": "text", "text": "ok"}])

        t = tool("add", "Add", AddInput, handler)
        result = await t.call({"a": "not_a_number", "b": 4}, ToolContext())
        assert result.is_error

    def test_annotations(self):
        async def handler(input: AddInput, ctx) -> CallToolResult:
            return CallToolResult(content=[])

        t = tool(
            "readonly_add", "Add", AddInput, handler,
            annotations=ToolAnnotations(read_only_hint=True),
        )
        assert t.is_read_only() is True
        assert t.is_concurrency_safe() is True


class TestDefineTool:
    @pytest.mark.asyncio
    async def test_create_and_call(self):
        async def handler(input: dict, ctx: ToolContext) -> ToolResult:
            return ToolResult(tool_use_id="", content=f"Hello {input.get('name', '')}")

        t = define_tool(
            name="greet",
            description="Greet someone",
            input_schema={"properties": {"name": {"type": "string"}}, "required": ["name"]},
            handler=handler,
        )
        assert t.name == "greet"
        result = await t.call({"name": "World"}, ToolContext())
        assert "Hello World" in result.content

    def test_read_only_flag(self):
        async def handler(input: dict, ctx: ToolContext) -> ToolResult:
            return ToolResult(tool_use_id="", content="ok")

        t = define_tool("ro", "Read only", {}, handler, read_only=True)
        assert t.is_read_only() is True


class TestToolToApiSchema:
    def test_conversion(self):
        async def handler(input: AddInput, ctx) -> CallToolResult:
            return CallToolResult(content=[])

        t = tool("calc", "Calculate", AddInput, handler)
        schema = tool_to_api_schema(t)
        assert schema["name"] == "calc"
        assert schema["description"] == "Calculate"
        assert "properties" in schema["input_schema"]
        assert "a" in schema["input_schema"]["properties"]
