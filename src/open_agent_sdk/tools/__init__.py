"""Built-in tools for the Open Agent SDK."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolDefinition

from open_agent_sdk.tools.bash import BashTool
from open_agent_sdk.tools.read import FileReadTool
from open_agent_sdk.tools.write import FileWriteTool
from open_agent_sdk.tools.edit import FileEditTool
from open_agent_sdk.tools.glob_tool import GlobTool
from open_agent_sdk.tools.grep import GrepTool
from open_agent_sdk.tools.notebook_edit import NotebookEditTool
from open_agent_sdk.tools.web_fetch import WebFetchTool
from open_agent_sdk.tools.web_search import WebSearchTool
from open_agent_sdk.tools.agent_tool import AgentTool
from open_agent_sdk.tools.send_message import SendMessageTool
from open_agent_sdk.tools.task_tools import (
    TaskCreateTool,
    TaskListTool,
    TaskUpdateTool,
    TaskGetTool,
    TaskStopTool,
    TaskOutputTool,
)
from open_agent_sdk.tools.team_tools import TeamCreateTool, TeamDeleteTool
from open_agent_sdk.tools.worktree_tools import EnterWorktreeTool, ExitWorktreeTool
from open_agent_sdk.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from open_agent_sdk.tools.ask_user import AskUserQuestionTool
from open_agent_sdk.tools.tool_search import ToolSearchTool
from open_agent_sdk.tools.mcp_resource_tools import ListMcpResourcesTool, ReadMcpResourceTool
from open_agent_sdk.tools.cron_tools import CronCreateTool, CronDeleteTool, CronListTool, RemoteTriggerTool
from open_agent_sdk.tools.lsp_tool import LSPTool
from open_agent_sdk.tools.config_tool import ConfigTool
from open_agent_sdk.tools.todo_tool import TodoWriteTool
from open_agent_sdk.tools.skill_tool import SkillTool

# In-memory stores for task/team/plan/config/cron state
_tasks: dict[str, dict[str, Any]] = {}
_task_counter: int = 0
_teams: dict[str, dict[str, Any]] = {}
_mailboxes: dict[str, list[dict[str, Any]]] = {}
_plan_mode: dict[str, bool] = {}
_current_plan: dict[str, str] = {}
_agents: dict[str, Any] = {}
_question_handler: Any = None
_deferred_tools: list[dict[str, Any]] = []
_mcp_connections: list[Any] = []
_cron_jobs: dict[str, dict[str, Any]] = {}
_config_store: dict[str, Any] = {}
_todos: list[dict[str, Any]] = []


def get_all_base_tools() -> list[BaseTool]:
    """Get all built-in tools."""
    return [
        BashTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GlobTool(),
        GrepTool(),
        NotebookEditTool(),
        WebFetchTool(),
        WebSearchTool(),
        AgentTool(),
        SendMessageTool(),
        TaskCreateTool(),
        TaskListTool(),
        TaskUpdateTool(),
        TaskGetTool(),
        TaskStopTool(),
        TaskOutputTool(),
        TeamCreateTool(),
        TeamDeleteTool(),
        EnterWorktreeTool(),
        ExitWorktreeTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        AskUserQuestionTool(),
        ToolSearchTool(),
        ListMcpResourcesTool(),
        ReadMcpResourceTool(),
        CronCreateTool(),
        CronDeleteTool(),
        CronListTool(),
        RemoteTriggerTool(),
        LSPTool(),
        ConfigTool(),
        TodoWriteTool(),
        SkillTool(),
    ]


def filter_tools(
    tools: list[BaseTool],
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
) -> list[BaseTool]:
    """Filter tools by allowed/disallowed lists."""
    result = tools
    if allowed_tools is not None:
        result = [t for t in result if t.name in allowed_tools]
    if disallowed_tools is not None:
        result = [t for t in result if t.name not in disallowed_tools]
    return [t for t in result if t.is_enabled()]


def assemble_tool_pool(
    base_tools: list[BaseTool] | None = None,
    mcp_tools: list[Any] | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
) -> list[BaseTool]:
    """Assemble the complete tool pool from base + MCP + custom tools."""
    tools = base_tools if base_tools is not None else get_all_base_tools()
    filtered = filter_tools(tools, allowed_tools, disallowed_tools)
    if mcp_tools:
        filtered.extend(mcp_tools)
    return filtered


# ─── State management functions ──────────────────────────────────────────────

def get_all_tasks() -> dict[str, dict[str, Any]]:
    return _tasks


def get_task(task_id: str) -> dict[str, Any] | None:
    return _tasks.get(task_id)


def clear_tasks() -> None:
    global _task_counter
    _tasks.clear()
    _task_counter = 0


def get_all_teams() -> dict[str, dict[str, Any]]:
    return _teams


def get_team(team_id: str) -> dict[str, Any] | None:
    return _teams.get(team_id)


def clear_teams() -> None:
    _teams.clear()


def read_mailbox(agent_name: str) -> list[dict[str, Any]]:
    msgs = _mailboxes.get(agent_name, [])
    _mailboxes[agent_name] = []
    return msgs


def write_to_mailbox(agent_name: str, message: dict[str, Any]) -> None:
    if agent_name not in _mailboxes:
        _mailboxes[agent_name] = []
    _mailboxes[agent_name].append(message)


def clear_mailboxes() -> None:
    _mailboxes.clear()


def is_plan_mode_active(session_id: str = "default") -> bool:
    return _plan_mode.get(session_id, False)


def get_current_plan(session_id: str = "default") -> str:
    return _current_plan.get(session_id, "")


def register_agents(agents: dict[str, Any]) -> None:
    _agents.update(agents)


def clear_agents() -> None:
    _agents.clear()


def set_question_handler(handler: Any) -> None:
    global _question_handler
    _question_handler = handler


def clear_question_handler() -> None:
    global _question_handler
    _question_handler = None


def set_deferred_tools(tools: list[dict[str, Any]]) -> None:
    global _deferred_tools
    _deferred_tools = tools


def set_mcp_connections(connections: list[Any]) -> None:
    global _mcp_connections
    _mcp_connections = connections


def get_all_cron_jobs() -> dict[str, dict[str, Any]]:
    return _cron_jobs


def clear_cron_jobs() -> None:
    _cron_jobs.clear()


def get_config() -> dict[str, Any]:
    return _config_store


def set_config(key: str, value: Any) -> None:
    _config_store[key] = value


def clear_config() -> None:
    _config_store.clear()


def get_todos() -> list[dict[str, Any]]:
    return _todos


def clear_todos() -> None:
    _todos.clear()


__all__ = [
    # Tools
    "BashTool",
    "FileReadTool",
    "FileWriteTool",
    "FileEditTool",
    "GlobTool",
    "GrepTool",
    "NotebookEditTool",
    "WebFetchTool",
    "WebSearchTool",
    "AgentTool",
    "SendMessageTool",
    "TaskCreateTool",
    "TaskListTool",
    "TaskUpdateTool",
    "TaskGetTool",
    "TaskStopTool",
    "TaskOutputTool",
    "TeamCreateTool",
    "TeamDeleteTool",
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "AskUserQuestionTool",
    "ToolSearchTool",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "CronCreateTool",
    "CronDeleteTool",
    "CronListTool",
    "RemoteTriggerTool",
    "LSPTool",
    "ConfigTool",
    "TodoWriteTool",
    "SkillTool",
    # Registry functions
    "get_all_base_tools",
    "filter_tools",
    "assemble_tool_pool",
    # State management
    "get_all_tasks",
    "get_task",
    "clear_tasks",
    "get_all_teams",
    "get_team",
    "clear_teams",
    "read_mailbox",
    "write_to_mailbox",
    "clear_mailboxes",
    "is_plan_mode_active",
    "get_current_plan",
    "register_agents",
    "clear_agents",
    "set_question_handler",
    "clear_question_handler",
    "set_deferred_tools",
    "set_mcp_connections",
    "get_all_cron_jobs",
    "clear_cron_jobs",
    "get_config",
    "set_config",
    "clear_config",
    "get_todos",
    "clear_todos",
]
