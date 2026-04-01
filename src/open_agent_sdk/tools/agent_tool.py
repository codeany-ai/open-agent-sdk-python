"""Agent tool - spawn subagents for delegated work."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class AgentTool(BaseTool):
    """Spawn subagents to handle complex, multi-step tasks."""

    _name = "Agent"
    _description = (
        "Launch a new agent to handle complex, multi-step tasks autonomously. "
        "Subagents can run in the background or foreground."
    )
    _input_schema = ToolInputSchema(
        properties={
            "prompt": {
                "type": "string",
                "description": "The task for the agent to perform",
            },
            "description": {
                "type": "string",
                "description": "A short (3-5 word) description of the task",
            },
            "subagent_type": {
                "type": "string",
                "description": "Type of specialized agent (e.g., 'Explore', 'Plan')",
            },
            "model": {
                "type": "string",
                "description": "Optional model override for this agent",
            },
            "name": {
                "type": "string",
                "description": "Name for the spawned agent",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Set to true to run this agent in the background",
            },
        },
        required=["prompt", "description"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        # Subagent execution is handled by the engine
        # This is a placeholder that should be intercepted by the engine
        return ToolResult(
            tool_use_id="",
            content="Agent tool must be executed by the engine's subagent handler.",
            is_error=True,
        )
