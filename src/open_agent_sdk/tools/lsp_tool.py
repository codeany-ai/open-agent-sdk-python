"""LSP tool - Language Server Protocol integration."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class LSPTool(BaseTool):
    _name = "LSP"
    _description = (
        "Language Server Protocol integration. "
        "Operations: goToDefinition, findReferences, hover, documentSymbol, workspaceSymbol."
    )
    _input_schema = ToolInputSchema(
        properties={
            "operation": {
                "type": "string",
                "description": "LSP operation",
                "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol", "workspaceSymbol"],
            },
            "file_path": {"type": "string", "description": "File path"},
            "line": {"type": "integer", "description": "Line number (0-based)"},
            "character": {"type": "integer", "description": "Character position (0-based)"},
            "query": {"type": "string", "description": "Search query (for workspaceSymbol)"},
        },
        required=["operation"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        operation = input.get("operation", "")
        return ToolResult(
            tool_use_id="",
            content=f"LSP operation '{operation}' is not yet implemented. Use Grep/Glob as alternatives.",
            is_error=True,
        )
