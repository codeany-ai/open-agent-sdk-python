"""Skill Registry.

Central registry for managing skill definitions.
Skills can be registered programmatically or loaded from bundled definitions.
"""

from __future__ import annotations

from open_agent_sdk.skills.types import SkillDefinition


# Internal skill store
_skills: dict[str, SkillDefinition] = {}

# Alias -> skill name mapping
_aliases: dict[str, str] = {}


def register_skill(definition: SkillDefinition) -> None:
    """Register a skill definition."""
    _skills[definition.name] = definition

    # Register aliases
    for alias in definition.aliases:
        _aliases[alias] = definition.name


def get_skill(name: str) -> SkillDefinition | None:
    """Get a skill by name or alias."""
    # Direct lookup
    direct = _skills.get(name)
    if direct is not None:
        return direct

    # Alias lookup
    resolved = _aliases.get(name)
    if resolved is not None:
        return _skills.get(resolved)

    return None


def get_all_skills() -> list[SkillDefinition]:
    """Get all registered skills."""
    return list(_skills.values())


def get_user_invocable_skills() -> list[SkillDefinition]:
    """Get all user-invocable skills (for /command listing)."""
    return [
        s for s in get_all_skills()
        if s.user_invocable is not False
        and (s.is_enabled is None or s.is_enabled())
    ]


def has_skill(name: str) -> bool:
    """Check if a skill exists."""
    return name in _skills or name in _aliases


def unregister_skill(name: str) -> bool:
    """Remove a skill."""
    skill = _skills.get(name)
    if skill is None:
        return False

    # Remove aliases
    for alias in skill.aliases:
        _aliases.pop(alias, None)

    del _skills[name]
    return True


def clear_skills() -> None:
    """Clear all skills (for testing)."""
    _skills.clear()
    _aliases.clear()


def format_skills_for_prompt(context_window_tokens: int | None = None) -> str:
    """Format skills listing for system prompt injection.

    Uses a budget system: skills listing gets a limited character budget
    to avoid bloating the context window.
    """
    invocable = get_user_invocable_skills()
    if not invocable:
        return ""

    # Budget: 1% of context window in characters (4 chars per token)
    CHARS_PER_TOKEN = 4
    DEFAULT_BUDGET = 8000
    MAX_DESC_CHARS = 250
    budget = (
        int(context_window_tokens * 0.01 * CHARS_PER_TOKEN)
        if context_window_tokens
        else DEFAULT_BUDGET
    )

    lines: list[str] = []
    used = 0

    for skill in invocable:
        desc = (
            skill.description[:MAX_DESC_CHARS] + "..."
            if len(skill.description) > MAX_DESC_CHARS
            else skill.description
        )

        trigger = f" TRIGGER when: {skill.when_to_use}" if skill.when_to_use else ""

        line = f"- {skill.name}: {desc}{trigger}"

        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)

    return "\n".join(lines)
