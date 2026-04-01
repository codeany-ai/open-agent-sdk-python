"""Cron scheduling and remote trigger tools."""

from __future__ import annotations

import json
import uuid
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class CronCreateTool(BaseTool):
    _name = "CronCreate"
    _description = "Create a scheduled cron job."
    _input_schema = ToolInputSchema(
        properties={
            "schedule": {"type": "string", "description": "Cron expression (5-field: min hour dom mon dow)"},
            "command": {"type": "string", "description": "Command or prompt to execute"},
            "name": {"type": "string", "description": "Job name"},
        },
        required=["schedule", "command"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _cron_jobs

        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "schedule": input.get("schedule", ""),
            "command": input.get("command", ""),
            "name": input.get("name", f"job_{job_id}"),
        }
        _cron_jobs[job_id] = job
        return ToolResult(tool_use_id="", content=f"Created cron job {job_id}: {job['name']}")


class CronDeleteTool(BaseTool):
    _name = "CronDelete"
    _description = "Delete a scheduled cron job."
    _input_schema = ToolInputSchema(
        properties={"job_id": {"type": "string", "description": "Job ID to delete"}},
        required=["job_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _cron_jobs

        job_id = input.get("job_id", "")
        if job_id not in _cron_jobs:
            return ToolResult(tool_use_id="", content=f"Error: job {job_id} not found", is_error=True)
        del _cron_jobs[job_id]
        return ToolResult(tool_use_id="", content=f"Deleted cron job {job_id}")


class CronListTool(BaseTool):
    _name = "CronList"
    _description = "List all cron jobs."
    _input_schema = ToolInputSchema()

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _cron_jobs

        if not _cron_jobs:
            return ToolResult(tool_use_id="", content="No cron jobs.")
        return ToolResult(tool_use_id="", content=json.dumps(list(_cron_jobs.values()), indent=2))


class RemoteTriggerTool(BaseTool):
    _name = "RemoteTrigger"
    _description = "Manage remote triggers and scheduled agents."
    _input_schema = ToolInputSchema(
        properties={
            "action": {"type": "string", "description": "Action: 'create', 'delete', 'list'"},
            "trigger_id": {"type": "string", "description": "Trigger ID (for delete)"},
            "config": {"type": "object", "description": "Trigger configuration (for create)"},
        },
        required=["action"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return input.get("action") == "list" if input else False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        action = input.get("action", "")
        return ToolResult(
            tool_use_id="",
            content=f"Remote trigger action '{action}' acknowledged. (Stub implementation)",
        )
