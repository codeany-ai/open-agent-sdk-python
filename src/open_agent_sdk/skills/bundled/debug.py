"""Bundled Skill: debug

Systematic debugging of an issue using structured investigation.
"""

from __future__ import annotations

from open_agent_sdk.skills.registry import register_skill
from open_agent_sdk.skills.types import SkillContentBlock, SkillDefinition

DEBUG_PROMPT = """Debug the described issue using a systematic approach:

1. **Reproduce**: Understand and reproduce the issue
   - Read relevant error messages or logs
   - Identify the failing component

2. **Investigate**: Trace the root cause
   - Read the relevant source code
   - Add logging or use debugging tools if needed
   - Check recent changes that might have introduced the issue (`git log --oneline -20`)

3. **Hypothesize**: Form a theory about the cause
   - State your hypothesis clearly before attempting a fix

4. **Fix**: Implement the minimal fix
   - Make the smallest change that resolves the issue
   - Don't refactor unrelated code

5. **Verify**: Confirm the fix works
   - Run relevant tests
   - Check for regressions"""


async def _get_prompt(args: str, context: object) -> list[SkillContentBlock]:
    prompt = DEBUG_PROMPT
    if args.strip():
        prompt += f"\n\n## Issue Description\n{args}"
    else:
        prompt += "\n\nAsk the user to describe the issue they're experiencing."
    return [{"type": "text", "text": prompt}]


def register_debug_skill() -> None:
    """Register the debug skill."""
    register_skill(SkillDefinition(
        name="debug",
        description="Systematic debugging of an issue using structured investigation.",
        aliases=["investigate", "diagnose"],
        user_invocable=True,
        get_prompt=_get_prompt,
    ))
