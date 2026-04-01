"""Tests for core type definitions."""

import pytest
from open_agent_sdk.types import (
    AgentOptions,
    BaseTool,
    CanUseToolResult,
    ContentBlock,
    ContentBlockType,
    ConversationMessage,
    MCPConnection,
    MessageRole,
    PermissionBehavior,
    PermissionMode,
    QueryResult,
    SDKMessage,
    SDKMessageType,
    SDKResultStatus,
    SDKSystemSubtype,
    ThinkingConfig,
    TokenUsage,
    ToolContext,
    ToolInputSchema,
    ToolResult,
)


class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_addition(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=75, cache_read_input_tokens=10)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 125
        assert c.cache_read_input_tokens == 10

    def test_custom_values(self):
        u = TokenUsage(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=100)
        assert u.input_tokens == 1000
        assert u.cache_creation_input_tokens == 100


class TestToolInputSchema:
    def test_to_dict(self):
        schema = ToolInputSchema(
            properties={"name": {"type": "string"}},
            required=["name"],
        )
        d = schema.to_dict()
        assert d["type"] == "object"
        assert d["properties"] == {"name": {"type": "string"}}
        assert d["required"] == ["name"]
        assert d["additionalProperties"] is False

    def test_empty_schema(self):
        schema = ToolInputSchema()
        d = schema.to_dict()
        assert d["properties"] == {}
        assert d["required"] == []


class TestToolResult:
    def test_to_dict(self):
        r = ToolResult(tool_use_id="id123", content="hello", is_error=False)
        d = r.to_dict()
        assert d["type"] == "tool_result"
        assert d["tool_use_id"] == "id123"
        assert d["content"] == "hello"
        assert d["is_error"] is False

    def test_error_result(self):
        r = ToolResult(tool_use_id="id456", content="failed", is_error=True)
        assert r.is_error is True


class TestToolContext:
    def test_defaults(self):
        ctx = ToolContext()
        assert ctx.cwd == "."
        assert ctx.env == {}

    def test_custom(self):
        ctx = ToolContext(cwd="/tmp", env={"KEY": "val"})
        assert ctx.cwd == "/tmp"
        assert ctx.env["KEY"] == "val"


class TestEnums:
    def test_message_role(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"

    def test_content_block_type(self):
        assert ContentBlockType.TEXT == "text"
        assert ContentBlockType.TOOL_USE == "tool_use"
        assert ContentBlockType.TOOL_RESULT == "tool_result"

    def test_sdk_message_type(self):
        assert SDKMessageType.ASSISTANT == "assistant"
        assert SDKMessageType.TOOL_RESULT == "tool_result"
        assert SDKMessageType.RESULT == "result"
        assert SDKMessageType.PARTIAL == "partial"
        assert SDKMessageType.SYSTEM == "system"

    def test_sdk_result_status(self):
        assert SDKResultStatus.SUCCESS == "success"
        assert SDKResultStatus.ERROR_MAX_TURNS == "error_max_turns"
        assert SDKResultStatus.ERROR_MAX_BUDGET == "error_max_budget_usd"

    def test_permission_mode(self):
        assert PermissionMode.DEFAULT == "default"
        assert PermissionMode.BYPASS_PERMISSIONS == "bypassPermissions"
        assert PermissionMode.AUTO == "auto"

    def test_permission_behavior(self):
        assert PermissionBehavior.ALLOW == "allow"
        assert PermissionBehavior.DENY == "deny"


class TestAgentOptions:
    def test_defaults(self):
        opts = AgentOptions()
        assert opts.model == "claude-sonnet-4-5"
        assert opts.max_turns == 10
        assert opts.max_tokens == 16000
        assert opts.permission_mode == PermissionMode.BYPASS_PERMISSIONS
        assert opts.tools == []
        assert opts.mcp_servers == {}

    def test_custom(self):
        opts = AgentOptions(
            model="claude-opus-4-6",
            max_turns=20,
            api_key="sk-test",
        )
        assert opts.model == "claude-opus-4-6"
        assert opts.max_turns == 20
        assert opts.api_key == "sk-test"


class TestQueryResult:
    def test_defaults(self):
        r = QueryResult()
        assert r.text == ""
        assert r.num_turns == 0
        assert r.cost == 0.0


class TestCanUseToolResult:
    def test_allow(self):
        r = CanUseToolResult(behavior=PermissionBehavior.ALLOW)
        assert r.behavior == PermissionBehavior.ALLOW
        assert r.message == ""

    def test_deny_with_message(self):
        r = CanUseToolResult(behavior=PermissionBehavior.DENY, message="Not allowed")
        assert r.behavior == PermissionBehavior.DENY
        assert r.message == "Not allowed"


class TestThinkingConfig:
    def test_defaults(self):
        tc = ThinkingConfig()
        assert tc.type == "enabled"
        assert tc.budget_tokens == 10000


class TestSDKMessage:
    def test_assistant_message(self):
        msg = SDKMessage(
            type=SDKMessageType.ASSISTANT,
            text="Hello",
            model="claude-sonnet-4-5",
        )
        assert msg.type == SDKMessageType.ASSISTANT
        assert msg.text == "Hello"

    def test_result_message(self):
        msg = SDKMessage(
            type=SDKMessageType.RESULT,
            status=SDKResultStatus.SUCCESS,
            num_turns=3,
            total_cost=0.05,
        )
        assert msg.status == SDKResultStatus.SUCCESS
        assert msg.num_turns == 3

    def test_system_message(self):
        msg = SDKMessage(
            type=SDKMessageType.SYSTEM,
            subtype=SDKSystemSubtype.INIT,
            system_data={"model": "test"},
        )
        assert msg.subtype == SDKSystemSubtype.INIT
