"""Tests for context utilities."""

import os
import pytest
from open_agent_sdk.utils.context import (
    clear_context_cache,
    get_git_status,
    get_system_context,
    get_user_context,
)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_context_cache()
    yield
    clear_context_cache()


class TestGetGitStatus:
    @pytest.mark.asyncio
    async def test_git_repo(self, tmp_path):
        # Initialize a git repo
        os.system(f"cd {tmp_path} && git init && git config user.name test && git config user.email test@test.com")
        status = await get_git_status(str(tmp_path))
        assert status.get("is_git") == "true"
        assert "branch" in status

    @pytest.mark.asyncio
    async def test_non_git_dir(self, tmp_path):
        status = await get_git_status(str(tmp_path))
        assert "is_git" not in status


class TestGetSystemContext:
    @pytest.mark.asyncio
    async def test_includes_cwd(self, tmp_path):
        context = await get_system_context(str(tmp_path))
        assert str(tmp_path) in context

    @pytest.mark.asyncio
    async def test_includes_platform(self, tmp_path):
        context = await get_system_context(str(tmp_path))
        assert "Platform" in context


class TestGetUserContext:
    @pytest.mark.asyncio
    async def test_includes_date(self, tmp_path):
        context = await get_user_context(str(tmp_path))
        assert "Current date" in context

    @pytest.mark.asyncio
    async def test_reads_agent_md(self, tmp_path):
        agent_md = tmp_path / "AGENT.md"
        agent_md.write_text("# Custom Agent Instructions\nDo stuff.")

        context = await get_user_context(str(tmp_path))
        assert "Custom Agent Instructions" in context

    @pytest.mark.asyncio
    async def test_reads_claude_md(self, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Claude Config\nBe helpful.")

        context = await get_user_context(str(tmp_path))
        assert "Claude Config" in context
