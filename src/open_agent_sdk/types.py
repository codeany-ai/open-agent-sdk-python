"""Core type definitions for the Open Agent SDK."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence, runtime_checkable


# ─── Message Types ───────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ContentBlockType(str, Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    IMAGE = "image"


@dataclass
class ContentBlock:
    type: ContentBlockType
    text: str = ""
    # tool_use fields
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    # tool_result fields
    tool_use_id: str = ""
    content: str | list[ContentBlock] = ""
    is_error: bool = False
    # thinking fields
    thinking: str = ""
    # image fields
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


@dataclass
class ConversationMessage:
    role: MessageRole
    content: str | list[dict[str, Any]]


@dataclass
class UserMessage:
    type: Literal["user"] = "user"
    message: ConversationMessage | None = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AssistantMessage:
    type: Literal["assistant"] = "assistant"
    message: ConversationMessage | None = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    usage: TokenUsage | None = None
    cost: float = 0.0


# ─── SDK Event Types (streaming) ─────────────────────────────────────────────

class SDKMessageType(str, Enum):
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    RESULT = "result"
    PARTIAL = "partial"
    SYSTEM = "system"


class SDKResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR_MAX_TURNS = "error_max_turns"
    ERROR_MAX_BUDGET = "error_max_budget_usd"
    ERROR_DURING_EXECUTION = "error_during_execution"


class SDKSystemSubtype(str, Enum):
    INIT = "init"
    COMPACT_BOUNDARY = "compact_boundary"
    STATUS = "status"
    TASK_NOTIFICATION = "task_notification"
    RATE_LIMIT = "rate_limit"


@dataclass
class SDKMessage:
    """Base streaming event yielded by agent loop."""
    type: SDKMessageType
    # For assistant messages
    message: ConversationMessage | None = None
    content_blocks: list[ContentBlock] = field(default_factory=list)
    usage: TokenUsage | None = None
    stop_reason: str = ""
    model: str = ""
    # For tool results
    tool_use_id: str = ""
    tool_name: str = ""
    result_content: str = ""
    is_error: bool = False
    # For final result
    status: SDKResultStatus | None = None
    text: str = ""
    num_turns: int = 0
    duration_ms: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_usage: TokenUsage | None = None
    total_cost: float = 0.0
    # For partial messages
    delta_text: str = ""
    delta_tool_use: dict[str, Any] | None = None
    # For system messages
    subtype: SDKSystemSubtype | None = None
    system_data: dict[str, Any] = field(default_factory=dict)


# ─── Tool Types ──────────────────────────────────────────────────────────────

@dataclass
class ToolInputSchema:
    type: str = "object"
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    additional_properties: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }
        if not self.additional_properties:
            result["additionalProperties"] = False
        return result


@dataclass
class ToolContext:
    cwd: str = "."
    abort_signal: Any = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@runtime_checkable
class ToolDefinition(Protocol):
    """Interface that all tools must implement."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> ToolInputSchema: ...

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult: ...

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool: ...

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool: ...

    def is_enabled(self) -> bool: ...


class BaseTool:
    """Base class for tool implementations."""

    _name: str = ""
    _description: str = ""
    _input_schema: ToolInputSchema = field(default_factory=ToolInputSchema)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> ToolInputSchema:
        return self._input_schema

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_enabled(self) -> bool:
        return True

    async def get_prompt(self, context: ToolContext) -> str | None:
        return None


# ─── Permission Types ────────────────────────────────────────────────────────

class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"
    PLAN = "plan"
    DONT_ASK = "dontAsk"
    AUTO = "auto"


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class CanUseToolResult:
    behavior: PermissionBehavior
    updated_input: Any = None
    message: str = ""


CanUseToolFn = Callable[["ToolDefinition", Any], Awaitable[CanUseToolResult]]


# ─── MCP Types ───────────────────────────────────────────────────────────────

class MCPTransportType(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class McpStdioConfig:
    type: Literal["stdio"] = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class McpSseConfig:
    type: Literal["sse"] = "sse"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class McpHttpConfig:
    type: Literal["http"] = "http"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


McpServerConfig = McpStdioConfig | McpSseConfig | McpHttpConfig


@dataclass
class MCPConnection:
    name: str
    status: str = "disconnected"  # connected | disconnected | error
    tools: list[ToolDefinition] = field(default_factory=list)
    close: Callable[[], Awaitable[None]] | None = None


# ─── Agent Types ─────────────────────────────────────────────────────────────

@dataclass
class ThinkingConfig:
    type: str = "enabled"
    budget_tokens: int = 10000


@dataclass
class AgentDefinition:
    description: str = ""
    prompt: str = ""
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    model: str | None = None
    mcp_servers: list[str | dict[str, Any]] | None = None
    max_turns: int | None = None


@dataclass
class AgentOptions:
    model: str = "claude-sonnet-4-5-20250514"
    api_key: str = ""
    base_url: str = ""
    cwd: str = ""
    system_prompt: str = ""
    append_system_prompt: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    max_turns: int = 10
    max_budget_usd: float | None = None
    max_tokens: int = 16000
    permission_mode: PermissionMode = PermissionMode.BYPASS_PERMISSIONS
    can_use_tool: CanUseToolFn | None = None
    abort_signal: Any = None
    include_partial_messages: bool = False
    env: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    agents: dict[str, AgentDefinition] = field(default_factory=dict)
    thinking: ThinkingConfig | None = None
    json_schema: dict[str, Any] | None = None
    persist_session: bool = False
    session_id: str = ""
    continue_session: bool = False
    resume: str = ""
    fork_session: str = ""
    sandbox: bool = False
    debug: bool = False
    debug_file: str = ""
    hooks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    custom_headers: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, Any] = field(default_factory=dict)
    betas: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    num_turns: int = 0
    duration_ms: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
