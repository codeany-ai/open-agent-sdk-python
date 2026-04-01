"""Write tool - create or overwrite files."""

from __future__ import annotations

import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class FileWriteTool(BaseTool):
    """Write content to a file, creating directories as needed."""

    _name = "Write"
    _description = "Writes a file to the local filesystem. Will overwrite existing files."
    _input_schema = ToolInputSchema(
        properties={
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        required=["file_path", "content"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        content = input.get("content", "")

        if not file_path:
            return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)

        try:
            # Create parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            size = len(content.encode("utf-8"))

            return ToolResult(
                tool_use_id="",
                content=f"Successfully wrote {lines} lines ({size} bytes) to {file_path}",
            )

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error writing file: {e}", is_error=True)
