"""ToolSearch tool - discover available tools."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class ToolSearchTool(BaseTool):
    _name = "ToolSearch"
    _description = "Find available tools by name or description keywords."
    _input_schema = ToolInputSchema(
        properties={
            "query": {"type": "string", "description": "Search query for tools"},
            "max_results": {"type": "number", "description": "Max results (default 5)"},
        },
        required=["query"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _deferred_tools

        query = input.get("query", "").lower()
        max_results = int(input.get("max_results", 5))

        matches = []
        for tool_info in _deferred_tools:
            name = tool_info.get("name", "").lower()
            desc = tool_info.get("description", "").lower()
            if query in name or query in desc:
                matches.append(tool_info)

        matches = matches[:max_results]

        if not matches:
            return ToolResult(tool_use_id="", content="No matching tools found.")

        output_parts = []
        for t in matches:
            output_parts.append(f"- {t.get('name', '')}: {t.get('description', '')}")

        return ToolResult(tool_use_id="", content="\n".join(output_parts))
