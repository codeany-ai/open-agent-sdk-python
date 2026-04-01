"""Glob tool - file pattern matching."""

from __future__ import annotations

import glob as glob_module
import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class GlobTool(BaseTool):
    """Fast file pattern matching using glob patterns."""

    _name = "Glob"
    _description = (
        "Fast file pattern matching tool. "
        "Supports glob patterns like '**/*.py' or 'src/**/*.ts'. "
        "Returns matching file paths sorted by modification time."
    )
    _input_schema = ToolInputSchema(
        properties={
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against",
            },
            "path": {
                "type": "string",
                "description": "The directory to search in (defaults to cwd)",
            },
        },
        required=["pattern"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        search_path = input.get("path", context.cwd)

        if not pattern:
            return ToolResult(tool_use_id="", content="Error: pattern is required", is_error=True)

        if not os.path.isabs(search_path):
            search_path = os.path.join(context.cwd, search_path)

        try:
            full_pattern = os.path.join(search_path, pattern)
            matches = glob_module.glob(full_pattern, recursive=True)

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

            # Limit results
            max_results = 500
            truncated = len(matches) > max_results
            matches = matches[:max_results]

            if not matches:
                return ToolResult(tool_use_id="", content="No files found matching pattern.")

            output = "\n".join(matches)
            if truncated:
                output += f"\n\n... (results limited to {max_results} files)"

            return ToolResult(tool_use_id="", content=output)

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error in glob search: {e}", is_error=True)
