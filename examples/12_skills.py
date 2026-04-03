"""
Example 12: Skills

Shows how to use the skill system: bundled skills, custom skills,
and invoking skills programmatically.

Run: python examples/12_skills.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_agent_sdk import (
    create_agent,
    AgentOptions,
    SDKMessageType,
)
from open_agent_sdk.skills import (
    register_skill,
    get_all_skills,
    get_user_invocable_skills,
    get_skill,
    init_bundled_skills,
    SkillDefinition,
)
from open_agent_sdk.types import ToolContext


async def main():
    print("--- Example 12: Skills ---\n")

    # Bundled skills are auto-initialized when creating an Agent,
    # but you can also init them explicitly:
    init_bundled_skills()

    # List all registered skills
    all_skills = get_all_skills()
    print(f"Registered skills ({len(all_skills)}):")
    for skill in all_skills:
        print(f"  - {skill.name}: {skill.description[:80]}...")

    # Register a custom skill
    async def explain_prompt(args: str, ctx: ToolContext):
        return [{
            "type": "text",
            "text": (
                "Explain the following in simple, clear terms that a beginner "
                "could understand. Use analogies where helpful.\n\n"
                f"Topic: {args or 'Ask the user what they want explained.'}"
            ),
        }]

    register_skill(SkillDefinition(
        name="explain",
        description="Explain a concept or piece of code in simple terms.",
        aliases=["eli5"],
        user_invocable=True,
        get_prompt=explain_prompt,
    ))

    print(f"\nAfter registering custom skill: {len(get_all_skills())} total")
    print(f"User-invocable: {len(get_user_invocable_skills())}")

    # Get a specific skill
    commit_skill = get_skill("commit")
    if commit_skill:
        blocks = await commit_skill.get_prompt("", ToolContext())
        text = blocks[0]["text"] if blocks and blocks[0].get("type") == "text" else "(non-text)"
        print(f"\nCommit skill prompt (first 200 chars):")
        print(text[:200] + "...")

    # Use skills with an agent — the model can invoke them via the Skill tool
    print("\n--- Using skills with an agent ---\n")

    agent = create_agent(AgentOptions(max_turns=5))

    async for event in agent.query('Use the "explain" skill to explain what git rebase does.'):
        if event.type == SDKMessageType.ASSISTANT:
            msg = event.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            print(f'[Tool: {block["name"]}] {block.get("input", {})}')
                        elif block.get("type") == "text" and block.get("text", "").strip():
                            print(block["text"])
        elif event.type == SDKMessageType.RESULT:
            print(f"\n--- {event.status} ---")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
