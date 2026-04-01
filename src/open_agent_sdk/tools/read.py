"""Read tool - read files with line numbers."""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class FileReadTool(BaseTool):
    """Read files with line numbers, supporting offset/limit."""

    _name = "Read"
    _description = (
        "Reads a file from the local filesystem. "
        "Results are returned with line numbers starting at 1."
    )
    _input_schema = ToolInputSchema(
        properties={
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "The line number to start reading from (0-based)",
                "minimum": 0,
            },
            "limit": {
                "type": "integer",
                "description": "The number of lines to read",
                "minimum": 1,
            },
        },
        required=["file_path"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        offset = input.get("offset", 0)
        limit = input.get("limit", 2000)

        if not file_path:
            return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

        # Resolve relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)

        if not os.path.exists(file_path):
            return ToolResult(tool_use_id="", content=f"Error: file not found: {file_path}", is_error=True)

        if os.path.isdir(file_path):
            return ToolResult(
                tool_use_id="",
                content=f"Error: {file_path} is a directory, not a file. Use Bash with ls instead.",
                is_error=True,
            )

        # Check if it's an image
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("image/"):
            try:
                with open(file_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                return ToolResult(
                    tool_use_id="",
                    content=[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": data,
                            },
                        }
                    ],
                )
            except Exception as e:
                return ToolResult(tool_use_id="", content=f"Error reading image: {e}", is_error=True)

        # Read text file
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            if not all_lines:
                return ToolResult(tool_use_id="", content="(empty file)")

            # Apply offset and limit
            selected = all_lines[offset : offset + limit]

            # Format with line numbers (1-based)
            numbered_lines = []
            for i, line in enumerate(selected, start=offset + 1):
                numbered_lines.append(f"{i}\t{line.rstrip()}")

            output = "\n".join(numbered_lines)

            if offset + limit < len(all_lines):
                remaining = len(all_lines) - (offset + limit)
                output += f"\n\n... ({remaining} more lines not shown)"

            return ToolResult(tool_use_id="", content=output)

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error reading file: {e}", is_error=True)
