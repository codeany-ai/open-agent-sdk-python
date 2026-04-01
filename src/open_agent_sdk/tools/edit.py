"""Edit tool - exact string replacement in files."""

from __future__ import annotations

import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class FileEditTool(BaseTool):
    """Perform exact string replacements in files."""

    _name = "Edit"
    _description = (
        "Performs exact string replacements in files. "
        "The edit will fail if old_string is not unique unless replace_all is true."
    )
    _input_schema = ToolInputSchema(
        properties={
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to modify",
            },
            "old_string": {
                "type": "string",
                "description": "The text to replace",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false)",
                "default": False,
            },
        },
        required=["file_path", "old_string", "new_string"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")
        replace_all = input.get("replace_all", False)

        if not file_path:
            return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)
        if not old_string:
            return ToolResult(tool_use_id="", content="Error: old_string is required", is_error=True)
        if old_string == new_string:
            return ToolResult(tool_use_id="", content="Error: old_string and new_string must differ", is_error=True)

        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)

        if not os.path.exists(file_path):
            return ToolResult(tool_use_id="", content=f"Error: file not found: {file_path}", is_error=True)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_string)

            if count == 0:
                return ToolResult(
                    tool_use_id="",
                    content=f"Error: old_string not found in {file_path}",
                    is_error=True,
                )

            if count > 1 and not replace_all:
                return ToolResult(
                    tool_use_id="",
                    content=(
                        f"Error: old_string found {count} times in {file_path}. "
                        "Use replace_all=true or provide more context to make it unique."
                    ),
                    is_error=True,
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(
                tool_use_id="",
                content=f"Successfully edited {file_path} ({count} replacement{'s' if count > 1 else ''})",
            )

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error editing file: {e}", is_error=True)
