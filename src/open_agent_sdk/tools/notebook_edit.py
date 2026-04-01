"""NotebookEdit tool - edit Jupyter notebooks."""

from __future__ import annotations

import json
import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class NotebookEditTool(BaseTool):
    """Edit Jupyter notebook cells."""

    _name = "NotebookEdit"
    _description = "Edit Jupyter notebook cells - modify content, add or remove cells."
    _input_schema = ToolInputSchema(
        properties={
            "notebook_path": {
                "type": "string",
                "description": "Path to the Jupyter notebook",
            },
            "action": {
                "type": "string",
                "description": "Action: 'edit', 'add', 'remove'",
                "enum": ["edit", "add", "remove"],
            },
            "cell_index": {
                "type": "integer",
                "description": "Index of the cell to modify (0-based)",
            },
            "content": {
                "type": "string",
                "description": "New cell content (for edit/add)",
            },
            "cell_type": {
                "type": "string",
                "description": "Cell type: 'code' or 'markdown'",
                "enum": ["code", "markdown"],
            },
        },
        required=["notebook_path", "action", "cell_index"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        notebook_path = input.get("notebook_path", "")
        action = input.get("action", "")
        cell_index = input.get("cell_index", 0)
        content = input.get("content", "")
        cell_type = input.get("cell_type", "code")

        if not os.path.isabs(notebook_path):
            notebook_path = os.path.join(context.cwd, notebook_path)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            cells = notebook.get("cells", [])

            if action == "edit":
                if cell_index < 0 or cell_index >= len(cells):
                    return ToolResult(tool_use_id="", content=f"Error: cell_index {cell_index} out of range", is_error=True)
                cells[cell_index]["source"] = content.split("\n")
                if cell_type:
                    cells[cell_index]["cell_type"] = cell_type

            elif action == "add":
                new_cell = {
                    "cell_type": cell_type,
                    "source": content.split("\n"),
                    "metadata": {},
                    "outputs": [] if cell_type == "code" else None,
                }
                if cell_type == "code":
                    new_cell["execution_count"] = None
                    new_cell["outputs"] = []
                cells.insert(cell_index, new_cell)

            elif action == "remove":
                if cell_index < 0 or cell_index >= len(cells):
                    return ToolResult(tool_use_id="", content=f"Error: cell_index {cell_index} out of range", is_error=True)
                cells.pop(cell_index)

            notebook["cells"] = cells

            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                f.write("\n")

            return ToolResult(
                tool_use_id="",
                content=f"Successfully {action}ed cell at index {cell_index} in {notebook_path}",
            )

        except Exception as e:
            return ToolResult(tool_use_id="", content=f"Error editing notebook: {e}", is_error=True)
