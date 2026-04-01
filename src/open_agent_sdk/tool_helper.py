"""Tool helper - Pydantic-based tool creation (analogous to Zod-based in TS)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Type

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult

try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


@dataclass
class ToolAnnotations:
    read_only_hint: bool = False
    destructive_hint: bool = False
    idempotent_hint: bool = False
    open_world_hint: bool = False


@dataclass
class CallToolResult:
    content: list[dict[str, str]]
    is_error: bool = False


class PydanticTool(BaseTool):
    """A tool defined using a Pydantic model for input validation."""

    def __init__(
        self,
        name: str,
        description: str,
        input_model: Type[BaseModel],
        handler: Callable[[Any, Any], Awaitable[CallToolResult]],
        annotations: ToolAnnotations | None = None,
    ):
        self._name = name
        self._description = description
        self._input_model = input_model
        self._handler = handler
        self._annotations = annotations or ToolAnnotations()

        # Convert Pydantic model to JSON Schema
        schema = input_model.model_json_schema()
        self._input_schema = ToolInputSchema(
            properties=schema.get("properties", {}),
            required=schema.get("required", []),
        )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return self._annotations.read_only_hint

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return self._annotations.read_only_hint

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            # Validate input with Pydantic
            validated = self._input_model.model_validate(input)
            result = await self._handler(validated, context)

            # Convert CallToolResult to ToolResult
            texts = []
            for block in result.content:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            content = "\n".join(texts) if texts else json.dumps(result.content, default=str)

            return ToolResult(
                tool_use_id="",
                content=content,
                is_error=result.is_error,
            )
        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Tool error: {e}",
                is_error=True,
            )


def tool(
    name: str,
    description: str,
    input_model: Type[BaseModel],
    handler: Callable[[Any, Any], Awaitable[CallToolResult]],
    annotations: ToolAnnotations | None = None,
) -> PydanticTool:
    """Create a tool from a Pydantic model (analogous to Zod-based tool() in TS)."""
    if not HAS_PYDANTIC:
        raise ImportError("pydantic is required for tool(). Install with: pip install pydantic")
    return PydanticTool(name, description, input_model, handler, annotations)


def define_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]],
    read_only: bool = False,
    concurrency_safe: bool = False,
) -> BaseTool:
    """Create a tool from a raw JSON schema dict (simpler alternative to tool())."""

    class CustomTool(BaseTool):
        _name = name
        _description = description
        _input_schema = ToolInputSchema(
            properties=input_schema.get("properties", {}),
            required=input_schema.get("required", []),
        )

        def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
            return read_only

        def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
            return concurrency_safe

        async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
            return await handler(input, context)

    return CustomTool()


def tool_to_api_schema(t: BaseTool) -> dict[str, Any]:
    """Convert a tool to Anthropic API tool schema."""
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema.to_dict(),
    }
