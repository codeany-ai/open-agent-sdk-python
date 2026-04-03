"""Skill Tool.

Allows the model to invoke registered skills by name.
Skills are prompt templates that provide specialized capabilities.
"""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult
from open_agent_sdk.skills.registry import get_skill, get_user_invocable_skills


class SkillTool(BaseTool):
    """Execute a skill within the current conversation.

    Skills provide specialized capabilities and domain knowledge.
    Use this tool with the skill name and optional arguments.
    Available skills are listed in system-reminder messages.
    """

    _name = "Skill"
    _description = (
        "Execute a skill within the current conversation. "
        "Skills provide specialized capabilities and domain knowledge. "
        "Use this tool with the skill name and optional arguments. "
        "Available skills are listed in system-reminder messages."
    )
    _input_schema = ToolInputSchema(
        properties={
            "skill": {
                "type": "string",
                "description": 'The skill name to execute (e.g., "commit", "review", "simplify")',
            },
            "args": {
                "type": "string",
                "description": "Optional arguments for the skill",
            },
        },
        required=["skill"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_enabled(self) -> bool:
        return len(get_user_invocable_skills()) > 0

    async def get_prompt(self, context: ToolContext) -> str | None:
        skills = get_user_invocable_skills()
        if not skills:
            return None

        lines = []
        for s in skills:
            desc = (
                s.description[:200] + "..."
                if len(s.description) > 200
                else s.description
            )
            lines.append(f"- {s.name}: {desc}")

        return (
            "Execute a skill within the main conversation.\n\n"
            "Available skills:\n"
            + "\n".join(lines)
            + "\n\nWhen a skill matches the user's request, invoke it using the Skill tool."
        )

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        skill_name: str = input.get("skill", "")
        args: str = input.get("args", "")

        if not skill_name:
            return ToolResult(
                tool_use_id="",
                content="Error: skill name is required",
                is_error=True,
            )

        skill = get_skill(skill_name)
        if skill is None:
            available = ", ".join(
                s.name for s in get_user_invocable_skills()
            )
            return ToolResult(
                tool_use_id="",
                content=f'Error: Unknown skill "{skill_name}". Available skills: {available or "none"}',
                is_error=True,
            )

        # Check if skill is enabled
        if skill.is_enabled is not None and not skill.is_enabled():
            return ToolResult(
                tool_use_id="",
                content=f'Error: Skill "{skill_name}" is currently disabled',
                is_error=True,
            )

        try:
            # Get skill prompt
            if skill.get_prompt is None:
                return ToolResult(
                    tool_use_id="",
                    content=f'Error: Skill "{skill_name}" has no prompt generator',
                    is_error=True,
                )

            content_blocks = await skill.get_prompt(args, context)

            # Convert content blocks to text
            prompt_text = "\n\n".join(
                b["text"] for b in content_blocks if b.get("type") == "text"
            )

            # Build result with metadata
            result: dict[str, Any] = {
                "success": True,
                "commandName": skill.name,
                "status": "forked" if skill.context == "fork" else "inline",
                "prompt": prompt_text,
            }

            if skill.allowed_tools:
                result["allowedTools"] = skill.allowed_tools

            if skill.model:
                result["model"] = skill.model

            return ToolResult(
                tool_use_id="",
                content=json.dumps(result),
            )

        except Exception as exc:
            return ToolResult(
                tool_use_id="",
                content=f'Error executing skill "{skill_name}": {exc}',
                is_error=True,
            )
