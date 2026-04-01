"""Tests for built-in tools."""

import os
import tempfile

import pytest

from open_agent_sdk.tools import (
    BashTool,
    FileReadTool,
    FileWriteTool,
    FileEditTool,
    GlobTool,
    GrepTool,
    TaskCreateTool,
    TaskListTool,
    TaskUpdateTool,
    TaskGetTool,
    TaskStopTool,
    TeamCreateTool,
    TeamDeleteTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    AskUserQuestionTool,
    ConfigTool,
    TodoWriteTool,
    CronCreateTool,
    CronListTool,
    CronDeleteTool,
    get_all_base_tools,
    filter_tools,
    assemble_tool_pool,
    clear_tasks,
    clear_teams,
    clear_cron_jobs,
    clear_config,
    clear_todos,
)
from open_agent_sdk.types import ToolContext


@pytest.fixture
def context(tmp_path):
    return ToolContext(cwd=str(tmp_path))


@pytest.fixture(autouse=True)
def cleanup_state():
    """Reset global state between tests."""
    yield
    clear_tasks()
    clear_teams()
    clear_cron_jobs()
    clear_config()
    clear_todos()


# ─── Tool Registry ──────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_get_all_base_tools(self):
        tools = get_all_base_tools()
        assert len(tools) >= 26
        names = [t.name for t in tools]
        assert "Bash" in names
        assert "Read" in names
        assert "Write" in names
        assert "Edit" in names
        assert "Glob" in names
        assert "Grep" in names
        assert "WebFetch" in names
        assert "WebSearch" in names
        assert "Agent" in names
        assert "TaskCreate" in names
        assert "TeamCreate" in names

    def test_filter_allowed(self):
        tools = get_all_base_tools()
        filtered = filter_tools(tools, allowed_tools=["Bash", "Read"])
        assert len(filtered) == 2
        names = [t.name for t in filtered]
        assert "Bash" in names
        assert "Read" in names

    def test_filter_disallowed(self):
        tools = get_all_base_tools()
        filtered = filter_tools(tools, disallowed_tools=["Bash"])
        names = [t.name for t in filtered]
        assert "Bash" not in names
        assert "Read" in names

    def test_assemble_tool_pool(self):
        pool = assemble_tool_pool(allowed_tools=["Bash", "Read", "Glob"])
        assert len(pool) == 3

    def test_tool_properties(self):
        for tool in get_all_base_tools():
            assert tool.name
            assert tool.description
            assert tool.input_schema is not None
            schema = tool.input_schema.to_dict()
            assert "type" in schema
            assert "properties" in schema


# ─── Bash Tool ───────────────────────────────────────────────────────────────

class TestBashTool:
    @pytest.mark.asyncio
    async def test_echo(self, context):
        tool = BashTool()
        result = await tool.call({"command": "echo hello"}, context)
        assert "hello" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_exit_code(self, context):
        tool = BashTool()
        result = await tool.call({"command": "exit 1"}, context)
        assert result.is_error
        assert "exit code: 1" in result.content

    @pytest.mark.asyncio
    async def test_stderr(self, context):
        tool = BashTool()
        result = await tool.call({"command": "echo error >&2"}, context)
        assert "error" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self, context):
        tool = BashTool()
        result = await tool.call({"command": "sleep 10", "timeout": 500}, context)
        assert result.is_error
        assert "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_empty_command(self, context):
        tool = BashTool()
        result = await tool.call({"command": ""}, context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_cwd(self, context):
        tool = BashTool()
        result = await tool.call({"command": "pwd"}, context)
        assert context.cwd in result.content

    def test_is_read_only(self):
        tool = BashTool()
        assert tool.is_read_only({"command": "ls"}) is True
        assert tool.is_read_only({"command": "rm -rf /"}) is False
        assert tool.is_read_only({"command": "git status"}) is True

    def test_is_not_concurrency_safe(self):
        assert BashTool().is_concurrency_safe() is False


# ─── File Read Tool ──────────────────────────────────────────────────────────

class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_file(self, context):
        filepath = os.path.join(context.cwd, "test.txt")
        with open(filepath, "w") as f:
            f.write("line 1\nline 2\nline 3\n")

        tool = FileReadTool()
        result = await tool.call({"file_path": filepath}, context)
        assert "1\tline 1" in result.content
        assert "2\tline 2" in result.content
        assert "3\tline 3" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_offset_and_limit(self, context):
        filepath = os.path.join(context.cwd, "test.txt")
        with open(filepath, "w") as f:
            for i in range(10):
                f.write(f"line {i}\n")

        tool = FileReadTool()
        result = await tool.call({"file_path": filepath, "offset": 2, "limit": 3}, context)
        assert "3\tline 2" in result.content
        assert "4\tline 3" in result.content
        assert "5\tline 4" in result.content

    @pytest.mark.asyncio
    async def test_file_not_found(self, context):
        tool = FileReadTool()
        result = await tool.call({"file_path": "/nonexistent/file.txt"}, context)
        assert result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_read_directory(self, context):
        tool = FileReadTool()
        result = await tool.call({"file_path": context.cwd}, context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_empty_file(self, context):
        filepath = os.path.join(context.cwd, "empty.txt")
        with open(filepath, "w") as f:
            pass
        tool = FileReadTool()
        result = await tool.call({"file_path": filepath}, context)
        assert "empty" in result.content.lower()

    def test_is_read_only(self):
        assert FileReadTool().is_read_only() is True

    def test_is_concurrency_safe(self):
        assert FileReadTool().is_concurrency_safe() is True


# ─── File Write Tool ─────────────────────────────────────────────────────────

class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_write_file(self, context):
        filepath = os.path.join(context.cwd, "output.txt")
        tool = FileWriteTool()
        result = await tool.call({"file_path": filepath, "content": "hello\nworld\n"}, context)
        assert not result.is_error
        assert "Successfully wrote" in result.content

        with open(filepath) as f:
            assert f.read() == "hello\nworld\n"

    @pytest.mark.asyncio
    async def test_creates_directories(self, context):
        filepath = os.path.join(context.cwd, "sub", "dir", "file.txt")
        tool = FileWriteTool()
        result = await tool.call({"file_path": filepath, "content": "nested"}, context)
        assert not result.is_error
        assert os.path.exists(filepath)

    @pytest.mark.asyncio
    async def test_overwrites(self, context):
        filepath = os.path.join(context.cwd, "overwrite.txt")
        with open(filepath, "w") as f:
            f.write("old content")

        tool = FileWriteTool()
        await tool.call({"file_path": filepath, "content": "new content"}, context)

        with open(filepath) as f:
            assert f.read() == "new content"


# ─── File Edit Tool ──────────────────────────────────────────────────────────

class TestFileEditTool:
    @pytest.mark.asyncio
    async def test_single_replacement(self, context):
        filepath = os.path.join(context.cwd, "edit.txt")
        with open(filepath, "w") as f:
            f.write("Hello World\nGoodbye World\n")

        tool = FileEditTool()
        result = await tool.call({
            "file_path": filepath,
            "old_string": "Hello World",
            "new_string": "Hi World",
        }, context)
        assert not result.is_error

        with open(filepath) as f:
            content = f.read()
        assert "Hi World" in content
        assert "Goodbye World" in content

    @pytest.mark.asyncio
    async def test_non_unique_error(self, context):
        filepath = os.path.join(context.cwd, "dup.txt")
        with open(filepath, "w") as f:
            f.write("foo bar foo\n")

        tool = FileEditTool()
        result = await tool.call({
            "file_path": filepath,
            "old_string": "foo",
            "new_string": "baz",
        }, context)
        assert result.is_error
        assert "found 2 times" in result.content

    @pytest.mark.asyncio
    async def test_replace_all(self, context):
        filepath = os.path.join(context.cwd, "replall.txt")
        with open(filepath, "w") as f:
            f.write("aaa bbb aaa\n")

        tool = FileEditTool()
        result = await tool.call({
            "file_path": filepath,
            "old_string": "aaa",
            "new_string": "ccc",
            "replace_all": True,
        }, context)
        assert not result.is_error

        with open(filepath) as f:
            assert f.read() == "ccc bbb ccc\n"

    @pytest.mark.asyncio
    async def test_not_found(self, context):
        filepath = os.path.join(context.cwd, "nf.txt")
        with open(filepath, "w") as f:
            f.write("hello\n")

        tool = FileEditTool()
        result = await tool.call({
            "file_path": filepath,
            "old_string": "xyz",
            "new_string": "abc",
        }, context)
        assert result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_same_string_error(self, context):
        tool = FileEditTool()
        result = await tool.call({
            "file_path": "/tmp/test.txt",
            "old_string": "same",
            "new_string": "same",
        }, context)
        assert result.is_error


# ─── Glob Tool ───────────────────────────────────────────────────────────────

class TestGlobTool:
    @pytest.mark.asyncio
    async def test_find_files(self, context):
        for name in ["a.py", "b.py", "c.txt"]:
            with open(os.path.join(context.cwd, name), "w") as f:
                f.write("content")

        tool = GlobTool()
        result = await tool.call({"pattern": "*.py"}, context)
        assert "a.py" in result.content
        assert "b.py" in result.content
        assert "c.txt" not in result.content

    @pytest.mark.asyncio
    async def test_no_matches(self, context):
        tool = GlobTool()
        result = await tool.call({"pattern": "*.xyz"}, context)
        assert "no files" in result.content.lower()

    @pytest.mark.asyncio
    async def test_recursive(self, context):
        subdir = os.path.join(context.cwd, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "deep.py"), "w") as f:
            f.write("x")

        tool = GlobTool()
        result = await tool.call({"pattern": "**/*.py"}, context)
        assert "deep.py" in result.content

    def test_is_read_only(self):
        assert GlobTool().is_read_only() is True

    def test_is_concurrency_safe(self):
        assert GlobTool().is_concurrency_safe() is True


# ─── Grep Tool ───────────────────────────────────────────────────────────────

class TestGrepTool:
    @pytest.mark.asyncio
    async def test_search(self, context):
        filepath = os.path.join(context.cwd, "search.txt")
        with open(filepath, "w") as f:
            f.write("line 1 hello\nline 2 world\nline 3 hello world\n")

        tool = GrepTool()
        result = await tool.call({"pattern": "hello", "path": context.cwd}, context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_no_matches(self, context):
        filepath = os.path.join(context.cwd, "nope.txt")
        with open(filepath, "w") as f:
            f.write("nothing here\n")

        tool = GrepTool()
        result = await tool.call({"pattern": "zzzzzzz", "path": context.cwd}, context)
        assert "no matches" in result.content.lower()

    def test_is_read_only(self):
        assert GrepTool().is_read_only() is True


# ─── Task Tools ──────────────────────────────────────────────────────────────

class TestTaskTools:
    @pytest.mark.asyncio
    async def test_create_and_list(self, context):
        create_tool = TaskCreateTool()
        result = await create_tool.call({"subject": "Test task", "status": "pending"}, context)
        assert "task_1" in result.content

        list_tool = TaskListTool()
        result = await list_tool.call({}, context)
        assert "Test task" in result.content

    @pytest.mark.asyncio
    async def test_update(self, context):
        create_tool = TaskCreateTool()
        await create_tool.call({"subject": "Update me"}, context)

        update_tool = TaskUpdateTool()
        result = await update_tool.call({"task_id": "task_1", "status": "completed"}, context)
        assert "Updated" in result.content

        get_tool = TaskGetTool()
        result = await get_tool.call({"task_id": "task_1"}, context)
        assert "completed" in result.content

    @pytest.mark.asyncio
    async def test_stop(self, context):
        create_tool = TaskCreateTool()
        await create_tool.call({"subject": "Stop me"}, context)

        stop_tool = TaskStopTool()
        result = await stop_tool.call({"task_id": "task_1"}, context)
        assert "Stopped" in result.content

    @pytest.mark.asyncio
    async def test_not_found(self, context):
        get_tool = TaskGetTool()
        result = await get_tool.call({"task_id": "nonexistent"}, context)
        assert result.is_error


# ─── Team Tools ──────────────────────────────────────────────────────────────

class TestTeamTools:
    @pytest.mark.asyncio
    async def test_create_and_delete(self, context):
        create_tool = TeamCreateTool()
        result = await create_tool.call({"name": "Test Team"}, context)
        assert "Created team" in result.content
        team_id = result.content.split()[2].rstrip(":")

        delete_tool = TeamDeleteTool()
        result = await delete_tool.call({"team_id": team_id}, context)
        assert "Deleted" in result.content

    @pytest.mark.asyncio
    async def test_delete_not_found(self, context):
        delete_tool = TeamDeleteTool()
        result = await delete_tool.call({"team_id": "nonexistent"}, context)
        assert result.is_error


# ─── Plan Tools ──────────────────────────────────────────────────────────────

class TestPlanTools:
    @pytest.mark.asyncio
    async def test_enter_exit(self, context):
        enter = EnterPlanModeTool()
        result = await enter.call({"plan": "My plan", "session_id": "s1"}, context)
        assert "Entered" in result.content

        from open_agent_sdk.tools import is_plan_mode_active
        assert is_plan_mode_active("s1") is True

        exit_tool = ExitPlanModeTool()
        result = await exit_tool.call({"session_id": "s1"}, context)
        assert "Exited" in result.content
        assert is_plan_mode_active("s1") is False


# ─── Config Tool ─────────────────────────────────────────────────────────────

class TestConfigTool:
    @pytest.mark.asyncio
    async def test_set_get_clear(self, context):
        tool = ConfigTool()

        result = await tool.call({"action": "set", "key": "theme", "value": "dark"}, context)
        assert "Set" in result.content

        result = await tool.call({"action": "get", "key": "theme"}, context)
        assert "dark" in result.content

        result = await tool.call({"action": "clear", "key": "theme"}, context)
        assert "Cleared" in result.content

        result = await tool.call({"action": "get", "key": "theme"}, context)
        assert "not set" in result.content.lower()

    @pytest.mark.asyncio
    async def test_list(self, context):
        tool = ConfigTool()
        await tool.call({"action": "set", "key": "a", "value": 1}, context)
        result = await tool.call({"action": "list"}, context)
        assert "a" in result.content


# ─── Todo Tool ───────────────────────────────────────────────────────────────

class TestTodoTool:
    @pytest.mark.asyncio
    async def test_add_list_complete_remove(self, context):
        tool = TodoWriteTool()

        result = await tool.call({"action": "add", "text": "Buy milk"}, context)
        assert "Added" in result.content

        result = await tool.call({"action": "list"}, context)
        assert "Buy milk" in result.content
        assert "[ ]" in result.content

        result = await tool.call({"action": "complete", "index": 0}, context)
        assert "Completed" in result.content

        result = await tool.call({"action": "list"}, context)
        assert "[x]" in result.content

        result = await tool.call({"action": "remove", "index": 0}, context)
        assert "Removed" in result.content

        result = await tool.call({"action": "list"}, context)
        assert "No todos" in result.content


# ─── Cron Tools ──────────────────────────────────────────────────────────────

class TestCronTools:
    @pytest.mark.asyncio
    async def test_create_list_delete(self, context):
        create = CronCreateTool()
        result = await create.call({"schedule": "*/5 * * * *", "command": "backup"}, context)
        assert "Created" in result.content
        job_id = result.content.split()[3].rstrip(":")

        list_tool = CronListTool()
        result = await list_tool.call({}, context)
        assert "backup" in result.content

        delete = CronDeleteTool()
        result = await delete.call({"job_id": job_id}, context)
        assert "Deleted" in result.content


# ─── AskUser Tool ────────────────────────────────────────────────────────────

class TestAskUserTool:
    @pytest.mark.asyncio
    async def test_no_handler(self, context):
        tool = AskUserQuestionTool()
        result = await tool.call({"question": "What?"}, context)
        assert result.is_error
        assert "no question handler" in result.content.lower()
