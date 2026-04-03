"""Bundled Skill: commit

Create a git commit with a well-crafted message based on staged changes.
"""

from __future__ import annotations

from open_agent_sdk.skills.registry import register_skill
from open_agent_sdk.skills.types import SkillContentBlock, SkillDefinition

COMMIT_PROMPT = """Create a git commit for the current changes. Follow these steps:

1. Run `git status` and `git diff --cached` to understand what's staged
2. If nothing is staged, run `git diff` to see unstaged changes and suggest what to stage
3. Analyze the changes and draft a concise commit message that:
   - Uses imperative mood ("Add feature" not "Added feature")
   - Summarizes the "why" not just the "what"
   - Keeps the first line under 72 characters
   - Adds a body with details if the change is complex
4. Create the commit

Do NOT push to remote unless explicitly asked."""


async def _get_prompt(args: str, context: object) -> list[SkillContentBlock]:
    prompt = COMMIT_PROMPT
    if args.strip():
        prompt += f"\n\nAdditional instructions: {args}"
    return [{"type": "text", "text": prompt}]


def register_commit_skill() -> None:
    """Register the commit skill."""
    register_skill(SkillDefinition(
        name="commit",
        description="Create a git commit with a well-crafted message based on staged changes.",
        aliases=["ci"],
        allowed_tools=["Bash", "Read", "Glob", "Grep"],
        user_invocable=True,
        get_prompt=_get_prompt,
    ))
