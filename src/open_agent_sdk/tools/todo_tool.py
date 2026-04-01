"""Todo tool - todo/task tracking."""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class TodoWriteTool(BaseTool):
    _name = "TodoWrite"
    _description = "Write/manage todo items."
    _input_schema = ToolInputSchema(
        properties={
            "action": {
                "type": "string",
                "description": "Action: 'add', 'complete', 'remove', 'list'",
                "enum": ["add", "complete", "remove", "list"],
            },
            "text": {"type": "string", "description": "Todo text (for add)"},
            "index": {"type": "integer", "description": "Todo index (for complete/remove)"},
        },
        required=["action"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return input.get("action") == "list" if input else False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _todos

        action = input.get("action", "")

        if action == "add":
            text = input.get("text", "")
            if not text:
                return ToolResult(tool_use_id="", content="Error: text is required", is_error=True)
            _todos.append({"text": text, "completed": False})
            return ToolResult(tool_use_id="", content=f"Added todo: {text}")

        elif action == "complete":
            index = input.get("index", -1)
            if index < 0 or index >= len(_todos):
                return ToolResult(tool_use_id="", content="Error: invalid index", is_error=True)
            _todos[index]["completed"] = True
            return ToolResult(tool_use_id="", content=f"Completed todo: {_todos[index]['text']}")

        elif action == "remove":
            index = input.get("index", -1)
            if index < 0 or index >= len(_todos):
                return ToolResult(tool_use_id="", content="Error: invalid index", is_error=True)
            removed = _todos.pop(index)
            return ToolResult(tool_use_id="", content=f"Removed todo: {removed['text']}")

        elif action == "list":
            if not _todos:
                return ToolResult(tool_use_id="", content="No todos.")
            lines = []
            for i, todo in enumerate(_todos):
                marker = "[x]" if todo["completed"] else "[ ]"
                lines.append(f"{i}. {marker} {todo['text']}")
            return ToolResult(tool_use_id="", content="\n".join(lines))

        return ToolResult(tool_use_id="", content=f"Unknown action: {action}", is_error=True)
