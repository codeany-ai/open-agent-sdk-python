"""Tests for skill system."""

import pytest

from open_agent_sdk.skills.types import SkillDefinition
from open_agent_sdk.skills.registry import (
    register_skill,
    get_skill,
    get_all_skills,
    get_user_invocable_skills,
    has_skill,
    unregister_skill,
    clear_skills,
    format_skills_for_prompt,
)
from open_agent_sdk.skills.bundled import init_bundled_skills
from open_agent_sdk.tools.skill_tool import SkillTool
from open_agent_sdk.types import ToolContext


@pytest.fixture(autouse=True)
def reset_skills():
    clear_skills()
    import open_agent_sdk.skills.bundled as bundled_mod
    bundled_mod._initialized = False
    yield
    clear_skills()
    bundled_mod._initialized = False


def _make_skill(name="test_skill", description="A test skill", **kwargs) -> SkillDefinition:
    async def default_prompt(args, ctx):
        return [{"type": "text", "text": f"Test prompt: {args}"}]

    return SkillDefinition(
        name=name,
        description=description,
        get_prompt=kwargs.pop("get_prompt", default_prompt),
        **kwargs,
    )


class TestSkillRegistry:
    def test_register_and_get(self):
        skill = _make_skill()
        register_skill(skill)
        assert get_skill("test_skill") is not None
        assert get_skill("test_skill").name == "test_skill"

    def test_get_by_alias(self):
        skill = _make_skill(aliases=["ts", "tskill"])
        register_skill(skill)
        assert get_skill("ts") is not None
        assert get_skill("tskill").name == "test_skill"

    def test_get_nonexistent(self):
        assert get_skill("nonexistent") is None

    def test_get_all_skills(self):
        register_skill(_make_skill("a", "Skill A"))
        register_skill(_make_skill("b", "Skill B"))
        all_skills = get_all_skills()
        assert len(all_skills) == 2

    def test_get_user_invocable(self):
        register_skill(_make_skill("pub", user_invocable=True))
        register_skill(_make_skill("priv", user_invocable=False))
        invocable = get_user_invocable_skills()
        assert len(invocable) == 1
        assert invocable[0].name == "pub"

    def test_has_skill(self):
        register_skill(_make_skill(aliases=["alias1"]))
        assert has_skill("test_skill") is True
        assert has_skill("alias1") is True
        assert has_skill("nope") is False

    def test_unregister(self):
        register_skill(_make_skill(aliases=["a1"]))
        assert unregister_skill("test_skill") is True
        assert get_skill("test_skill") is None
        assert get_skill("a1") is None

    def test_unregister_nonexistent(self):
        assert unregister_skill("nope") is False

    def test_clear(self):
        register_skill(_make_skill("x"))
        register_skill(_make_skill("y"))
        clear_skills()
        assert len(get_all_skills()) == 0

    def test_format_for_prompt(self):
        register_skill(_make_skill("commit", "Create git commit", user_invocable=True))
        register_skill(_make_skill("review", "Review code", user_invocable=True))
        text = format_skills_for_prompt()
        assert "commit" in text
        assert "review" in text

    def test_format_empty(self):
        assert format_skills_for_prompt() == ""

    def test_format_respects_budget(self):
        for i in range(100):
            register_skill(_make_skill(f"skill_{i}", "x" * 200, user_invocable=True))
        text = format_skills_for_prompt(context_window_tokens=1000)
        # With 1000 tokens * 0.01 * 4 = 40 chars budget, should be truncated
        assert len(text) < 200 * 100

    def test_disabled_skill_excluded(self):
        register_skill(_make_skill("disabled", is_enabled=lambda: False, user_invocable=True))
        register_skill(_make_skill("enabled", user_invocable=True))
        invocable = get_user_invocable_skills()
        assert len(invocable) == 1
        assert invocable[0].name == "enabled"


class TestBundledSkills:
    def test_init(self):
        init_bundled_skills()
        skills = get_all_skills()
        names = [s.name for s in skills]
        assert "simplify" in names
        assert "commit" in names
        assert "review" in names
        assert "debug" in names
        assert "test" in names

    def test_idempotent(self):
        init_bundled_skills()
        count1 = len(get_all_skills())
        init_bundled_skills()
        count2 = len(get_all_skills())
        assert count1 == count2

    @pytest.mark.asyncio
    async def test_commit_skill_prompt(self):
        init_bundled_skills()
        skill = get_skill("commit")
        assert skill is not None
        blocks = await skill.get_prompt("", ToolContext())
        assert len(blocks) >= 1
        assert blocks[0]["type"] == "text"
        assert "git" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_commit_alias(self):
        init_bundled_skills()
        assert get_skill("ci") is not None

    @pytest.mark.asyncio
    async def test_review_skill_prompt(self):
        init_bundled_skills()
        skill = get_skill("review")
        blocks = await skill.get_prompt("focus on security", ToolContext())
        assert "security" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_debug_skill_prompt(self):
        init_bundled_skills()
        skill = get_skill("debug")
        blocks = await skill.get_prompt("app crashes on startup", ToolContext())
        assert "app crashes" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_simplify_skill_prompt(self):
        init_bundled_skills()
        skill = get_skill("simplify")
        blocks = await skill.get_prompt("", ToolContext())
        assert "reuse" in blocks[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_test_skill_prompt(self):
        init_bundled_skills()
        skill = get_skill("test")
        blocks = await skill.get_prompt("tests/unit/", ToolContext())
        assert "tests/unit/" in blocks[0]["text"]


class TestSkillTool:
    @pytest.mark.asyncio
    async def test_invoke_skill(self):
        register_skill(_make_skill("greet", "Greet someone", user_invocable=True))
        tool = SkillTool()
        result = await tool.call({"skill": "greet", "args": "World"}, ToolContext())
        assert not result.is_error
        import json
        data = json.loads(result.content)
        assert data["success"] is True
        assert data["commandName"] == "greet"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent(self):
        tool = SkillTool()
        result = await tool.call({"skill": "nonexistent"}, ToolContext())
        assert result.is_error
        assert "Unknown skill" in result.content

    @pytest.mark.asyncio
    async def test_invoke_disabled_skill(self):
        register_skill(_make_skill("off", is_enabled=lambda: False))
        tool = SkillTool()
        result = await tool.call({"skill": "off"}, ToolContext())
        assert result.is_error
        assert "disabled" in result.content.lower()

    @pytest.mark.asyncio
    async def test_missing_skill_name(self):
        tool = SkillTool()
        result = await tool.call({"skill": ""}, ToolContext())
        assert result.is_error

    def test_is_enabled_with_skills(self):
        register_skill(_make_skill("x", user_invocable=True))
        assert SkillTool().is_enabled() is True

    def test_is_enabled_without_skills(self):
        assert SkillTool().is_enabled() is False

    @pytest.mark.asyncio
    async def test_get_prompt(self):
        register_skill(_make_skill("skill1", "Do thing 1", user_invocable=True))
        tool = SkillTool()
        prompt = await tool.get_prompt(ToolContext())
        assert "skill1" in prompt

    @pytest.mark.asyncio
    async def test_allowed_tools_in_result(self):
        async def prompt_fn(args, ctx):
            return [{"type": "text", "text": "do stuff"}]

        register_skill(_make_skill("limited", allowed_tools=["Bash", "Read"], get_prompt=prompt_fn))
        tool = SkillTool()
        result = await tool.call({"skill": "limited"}, ToolContext())
        import json
        data = json.loads(result.content)
        assert data["allowedTools"] == ["Bash", "Read"]
