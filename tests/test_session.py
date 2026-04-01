"""Tests for session management."""

import os
import pytest
from open_agent_sdk.session import (
    save_session,
    load_session,
    list_sessions,
    fork_session,
    get_session_messages,
    get_session_info,
    rename_session,
    tag_session,
    append_to_session,
    delete_session,
    SESSION_DIR,
)


@pytest.fixture
def session_id():
    return "test-session-001"


@pytest.fixture(autouse=True)
async def cleanup(session_id):
    yield
    await delete_session(session_id)
    await delete_session("forked-session")


class TestSaveAndLoad:
    @pytest.mark.asyncio
    async def test_save_and_load(self, session_id):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await save_session(session_id, messages, {"id": session_id, "model": "test"})

        data = await load_session(session_id)
        assert data is not None
        assert len(data["messages"]) == 2
        assert data["metadata"]["id"] == session_id

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        data = await load_session("nonexistent-session")
        assert data is None


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_includes_saved(self, session_id):
        await save_session(session_id, [{"role": "user", "content": "Test"}])
        sessions = await list_sessions()
        ids = [s.get("id") for s in sessions]
        assert session_id in ids


class TestForkSession:
    @pytest.mark.asyncio
    async def test_fork(self, session_id):
        await save_session(session_id, [{"role": "user", "content": "Original"}])
        new_id = await fork_session(session_id, "forked-session")
        assert new_id == "forked-session"

        data = await load_session(new_id)
        assert data is not None
        assert len(data["messages"]) == 1
        assert data["metadata"]["forkedFrom"] == session_id

    @pytest.mark.asyncio
    async def test_fork_nonexistent(self):
        result = await fork_session("nonexistent")
        assert result is None


class TestSessionOperations:
    @pytest.mark.asyncio
    async def test_get_messages(self, session_id):
        await save_session(session_id, [{"role": "user", "content": "Hi"}])
        messages = await get_session_messages(session_id)
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_get_info(self, session_id):
        await save_session(session_id, [], {"id": session_id, "model": "sonnet"})
        info = await get_session_info(session_id)
        assert info is not None
        assert info["id"] == session_id

    @pytest.mark.asyncio
    async def test_rename(self, session_id):
        await save_session(session_id, [])
        await rename_session(session_id, "My Session")
        info = await get_session_info(session_id)
        assert info["title"] == "My Session"

    @pytest.mark.asyncio
    async def test_tag(self, session_id):
        await save_session(session_id, [])
        await tag_session(session_id, ["debug", "test"])
        info = await get_session_info(session_id)
        assert "debug" in info["tags"]

    @pytest.mark.asyncio
    async def test_append(self, session_id):
        await save_session(session_id, [{"role": "user", "content": "First"}])
        await append_to_session(session_id, {"role": "assistant", "content": "Second"})
        messages = await get_session_messages(session_id)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_delete(self, session_id):
        await save_session(session_id, [])
        assert await delete_session(session_id) is True
        assert await load_session(session_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        assert await delete_session("nonexistent") is False
