"""Bundled Skill: test

Run tests and analyze failures.
"""

from __future__ import annotations

from open_agent_sdk.skills.registry import register_skill
from open_agent_sdk.skills.types import SkillContentBlock, SkillDefinition

TEST_PROMPT = """Run the project's test suite and analyze the results:

1. **Discover**: Find the test runner configuration
   - Look for package.json scripts, jest.config, vitest.config, pytest.ini, etc.
   - Identify the appropriate test command

2. **Execute**: Run the tests
   - Run the full test suite or specific tests if specified
   - Capture output including failures and errors

3. **Analyze**: If tests fail:
   - Read the failing test to understand what it expects
   - Read the source code being tested
   - Identify why the test is failing
   - Fix the issue (in tests or source as appropriate)

4. **Re-verify**: Run the failing tests again to confirm the fix"""


async def _get_prompt(args: str, context: object) -> list[SkillContentBlock]:
    prompt = TEST_PROMPT
    if args.strip():
        prompt += f"\n\nSpecific test target: {args}"
    return [{"type": "text", "text": prompt}]


def register_test_skill() -> None:
    """Register the test skill."""
    register_skill(SkillDefinition(
        name="test",
        description="Run tests and analyze failures, fixing any issues found.",
        aliases=["run-tests"],
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        user_invocable=True,
        get_prompt=_get_prompt,
    ))
