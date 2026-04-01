"""Planning mode tools."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class EnterPlanModeTool(BaseTool):
    _name = "EnterPlanMode"
    _description = "Enter structured planning workflow."
    _input_schema = ToolInputSchema(
        properties={
            "plan": {"type": "string", "description": "The plan content"},
            "session_id": {"type": "string", "description": "Session ID"},
        },
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _plan_mode, _current_plan

        session_id = input.get("session_id", "default")
        plan = input.get("plan", "")

        _plan_mode[session_id] = True
        _current_plan[session_id] = plan

        return ToolResult(tool_use_id="", content="Entered plan mode.")


class ExitPlanModeTool(BaseTool):
    _name = "ExitPlanMode"
    _description = "Exit plan mode."
    _input_schema = ToolInputSchema(
        properties={
            "session_id": {"type": "string", "description": "Session ID"},
        },
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _plan_mode, _current_plan

        session_id = input.get("session_id", "default")
        _plan_mode[session_id] = False
        _current_plan.pop(session_id, None)

        return ToolResult(tool_use_id="", content="Exited plan mode.")
