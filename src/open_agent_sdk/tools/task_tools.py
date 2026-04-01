"""Task management tools."""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class TaskCreateTool(BaseTool):
    _name = "TaskCreate"
    _description = "Create a new task for tracking work."
    _input_schema = ToolInputSchema(
        properties={
            "subject": {"type": "string", "description": "Task subject/title"},
            "description": {"type": "string", "description": "Task description"},
            "owner": {"type": "string", "description": "Task owner"},
            "status": {
                "type": "string",
                "description": "Initial status",
                "enum": ["pending", "in_progress"],
            },
        },
        required=["subject"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks, _task_counter
        import open_agent_sdk.tools as tools_mod

        tools_mod._task_counter += 1
        task_id = f"task_{tools_mod._task_counter}"
        task = {
            "id": task_id,
            "subject": input.get("subject", ""),
            "description": input.get("description", ""),
            "owner": input.get("owner", ""),
            "status": input.get("status", "pending"),
            "output": "",
            "blocked_by": [],
            "blocks": [],
        }
        _tasks[task_id] = task
        return ToolResult(tool_use_id="", content=f"Created task {task_id}: {task['subject']}")


class TaskListTool(BaseTool):
    _name = "TaskList"
    _description = "List tasks with optional filters."
    _input_schema = ToolInputSchema(
        properties={
            "status": {"type": "string", "description": "Filter by status"},
            "owner": {"type": "string", "description": "Filter by owner"},
        },
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks

        status_filter = input.get("status")
        owner_filter = input.get("owner")

        tasks = list(_tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t["status"] == status_filter]
        if owner_filter:
            tasks = [t for t in tasks if t["owner"] == owner_filter]

        if not tasks:
            return ToolResult(tool_use_id="", content="No tasks found.")

        return ToolResult(tool_use_id="", content=json.dumps(tasks, indent=2))


class TaskUpdateTool(BaseTool):
    _name = "TaskUpdate"
    _description = "Update task status, metadata, or dependencies."
    _input_schema = ToolInputSchema(
        properties={
            "task_id": {"type": "string", "description": "Task ID to update"},
            "status": {
                "type": "string",
                "description": "New status",
                "enum": ["pending", "in_progress", "completed", "cancelled"],
            },
            "output": {"type": "string", "description": "Task output/result"},
            "blocked_by": {"type": "array", "description": "Task IDs this task is blocked by"},
            "blocks": {"type": "array", "description": "Task IDs this task blocks"},
        },
        required=["task_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks

        task_id = input.get("task_id", "")
        if task_id not in _tasks:
            return ToolResult(tool_use_id="", content=f"Error: task {task_id} not found", is_error=True)

        task = _tasks[task_id]
        for key in ["status", "output", "blocked_by", "blocks"]:
            if key in input:
                task[key] = input[key]

        return ToolResult(tool_use_id="", content=f"Updated task {task_id}")


class TaskGetTool(BaseTool):
    _name = "TaskGet"
    _description = "Get a single task by ID."
    _input_schema = ToolInputSchema(
        properties={"task_id": {"type": "string", "description": "Task ID"}},
        required=["task_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks

        task_id = input.get("task_id", "")
        task = _tasks.get(task_id)
        if not task:
            return ToolResult(tool_use_id="", content=f"Error: task {task_id} not found", is_error=True)
        return ToolResult(tool_use_id="", content=json.dumps(task, indent=2))


class TaskStopTool(BaseTool):
    _name = "TaskStop"
    _description = "Cancel/stop a task."
    _input_schema = ToolInputSchema(
        properties={"task_id": {"type": "string", "description": "Task ID to stop"}},
        required=["task_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks

        task_id = input.get("task_id", "")
        if task_id not in _tasks:
            return ToolResult(tool_use_id="", content=f"Error: task {task_id} not found", is_error=True)
        _tasks[task_id]["status"] = "cancelled"
        return ToolResult(tool_use_id="", content=f"Stopped task {task_id}")


class TaskOutputTool(BaseTool):
    _name = "TaskOutput"
    _description = "Get task output/results."
    _input_schema = ToolInputSchema(
        properties={"task_id": {"type": "string", "description": "Task ID"}},
        required=["task_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _tasks

        task_id = input.get("task_id", "")
        task = _tasks.get(task_id)
        if not task:
            return ToolResult(tool_use_id="", content=f"Error: task {task_id} not found", is_error=True)
        return ToolResult(tool_use_id="", content=task.get("output", "(no output)"))
