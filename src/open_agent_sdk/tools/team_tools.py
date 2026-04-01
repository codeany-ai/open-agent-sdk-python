"""Team coordination tools."""

from __future__ import annotations

import json
import uuid
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class TeamCreateTool(BaseTool):
    _name = "TeamCreate"
    _description = "Create a team for multi-agent coordination."
    _input_schema = ToolInputSchema(
        properties={
            "name": {"type": "string", "description": "Team name"},
            "description": {"type": "string", "description": "Team purpose"},
            "members": {"type": "array", "description": "List of agent names"},
        },
        required=["name"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _teams

        team_id = str(uuid.uuid4())[:8]
        team = {
            "id": team_id,
            "name": input.get("name", ""),
            "description": input.get("description", ""),
            "members": input.get("members", []),
        }
        _teams[team_id] = team
        return ToolResult(tool_use_id="", content=f"Created team {team_id}: {team['name']}")


class TeamDeleteTool(BaseTool):
    _name = "TeamDelete"
    _description = "Delete a team and cleanup resources."
    _input_schema = ToolInputSchema(
        properties={"team_id": {"type": "string", "description": "Team ID to delete"}},
        required=["team_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from open_agent_sdk.tools import _teams

        team_id = input.get("team_id", "")
        if team_id not in _teams:
            return ToolResult(tool_use_id="", content=f"Error: team {team_id} not found", is_error=True)
        del _teams[team_id]
        return ToolResult(tool_use_id="", content=f"Deleted team {team_id}")
