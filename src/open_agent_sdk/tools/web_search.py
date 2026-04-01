"""WebSearch tool - web search integration."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult

# Pluggable search function
SearchFn = Callable[[str, int], Awaitable[list[dict[str, str]]]]

_search_fn: SearchFn | None = None


def set_search_fn(fn: SearchFn | None) -> None:
    global _search_fn
    _search_fn = fn


class WebSearchTool(BaseTool):
    """Search the web for information."""

    _name = "WebSearch"
    _description = "Search the web for information. Returns search results with titles, URLs, and snippets."
    _input_schema = ToolInputSchema(
        properties={
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10)",
            },
        },
        required=["query"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = input.get("query", "")
        max_results = input.get("max_results", 10)

        if not query:
            return ToolResult(tool_use_id="", content="Error: query is required", is_error=True)

        if _search_fn is None:
            return ToolResult(
                tool_use_id="",
                content="Error: web search is not configured. Set a search function with set_search_fn().",
                is_error=True,
            )

        try:
            results = await _search_fn(query, max_results)
            if not results:
                return ToolResult(tool_use_id="", content="No results found.")

            output_parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                output_parts.append(f"{i}. {title}\n   {url}\n   {snippet}")

            return ToolResult(tool_use_id="", content="\n\n".join(output_parts))

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error searching: {e}", is_error=True)
