"""Skill System Types.

Skills are reusable prompt templates that extend agent capabilities.
They can be invoked by the model via the Skill tool or by users via /skillname.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

from open_agent_sdk.types import ToolContext
from open_agent_sdk.hooks import HookConfig


# Content block types for skill prompts (compatible with Anthropic API).
SkillContentBlock = dict[str, Any]
"""Either {'type': 'text', 'text': str} or {'type': 'image', 'source': {...}}."""


@dataclass
class SkillDefinition:
    """Bundled skill definition.

    Inspired by Claude Code's skill system. Skills provide specialized
    capabilities by injecting context-specific prompts with optional
    tool restrictions and model overrides.
    """

    # Unique skill name (e.g., 'simplify', 'commit')
    name: str = ""

    # Human-readable description
    description: str = ""

    # Alternative names for the skill
    aliases: list[str] = field(default_factory=list)

    # When the model should invoke this skill (used in system prompt)
    when_to_use: str = ""

    # Hint for expected arguments
    argument_hint: str = ""

    # Tools the skill is allowed to use (empty = all tools)
    allowed_tools: list[str] = field(default_factory=list)

    # Model override for this skill
    model: str = ""

    # Whether the skill can be invoked by users via /command
    user_invocable: bool = True

    # Runtime check for availability
    is_enabled: Callable[[], bool] | None = None

    # Hook overrides while skill is active
    hooks: HookConfig | None = None

    # Execution context: 'inline' runs in current context, 'fork' spawns a subagent
    context: Literal["inline", "fork"] = "inline"

    # Subagent type for forked execution
    agent: str = ""

    # Generate the prompt content blocks for this skill.
    get_prompt: Callable[[str, ToolContext], Awaitable[list[SkillContentBlock]]] | None = None


@dataclass
class SkillResult:
    """Result of executing a skill."""

    # Whether execution succeeded
    success: bool = False

    # Skill name that was executed
    skill_name: str = ""

    # Execution status
    status: Literal["inline", "forked"] = "inline"

    # Allowed tools override (for inline execution)
    allowed_tools: list[str] = field(default_factory=list)

    # Model override
    model: str = ""

    # Result text (for forked execution)
    result: str = ""

    # Error message
    error: str = ""
