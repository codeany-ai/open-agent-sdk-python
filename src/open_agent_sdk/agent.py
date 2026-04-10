"""Agent - high-level API for the Open Agent SDK."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, AsyncGenerator

import anthropic

from open_agent_sdk.providers.types import LLMProvider
from open_agent_sdk.providers.factory import create_provider
from open_agent_sdk.engine import QueryEngine, QueryEngineConfig
from open_agent_sdk.session import load_session, save_session
from open_agent_sdk.tools import get_all_base_tools, filter_tools
from open_agent_sdk.types import (
    AgentOptions,
    BaseTool,
    MCPConnection,
    PermissionMode,
    QueryResult,
    SDKMessage,
    SDKMessageType,
    SDKResultStatus,
    TokenUsage,
)


class Agent:
    """High-level agent API with multi-turn conversation support."""

    def __init__(self, options: AgentOptions | None = None):
        self._options = options or AgentOptions()
        self._session_id = self._options.session_id or str(uuid.uuid4())
        self._history: list[dict[str, Any]] = []
        self._tool_pool: list[BaseTool] = []
        self._mcp_connections: list[MCPConnection] = []
        self._client: anthropic.AsyncAnthropic | None = None
        self._provider: LLMProvider | None = None
        self._engine: QueryEngine | None = None
        self._initialized = False

    # OpenAI-compatible model prefixes
    _OPENAI_MODEL_PREFIXES = ("gpt-", "deepseek-", "qwen-", "o1-", "o3-", "o4-")

    def _resolve_model(self) -> str:
        """Resolve model from options or CODEANY_MODEL env var."""
        return (
            self._options.model
            or os.environ.get("CODEANY_MODEL", "")
            or "claude-sonnet-4-5"
        )

    def get_api_type(self) -> str:
        """Detect the API type from explicit config, env var, or model name.

        Returns 'anthropic-messages' or 'openai-completions'.
        """
        # Explicit option
        if self._options.api_type:
            return self._options.api_type

        # Environment variable
        env_type = os.environ.get("CODEANY_API_TYPE", "")
        if env_type:
            return env_type

        # Auto-detect from model name
        model = self._resolve_model()
        for prefix in self._OPENAI_MODEL_PREFIXES:
            if model.startswith(prefix):
                return "openai-completions"

        return "anthropic-messages"

    def _ensure_provider(self) -> LLMProvider:
        """Create or return the LLM provider."""
        if self._provider is not None:
            return self._provider

        api_type = self.get_api_type()

        # Resolve API key: options > CODEANY_API_KEY > ANTHROPIC_API_KEY / OPENAI_API_KEY
        api_key = (
            self._options.api_key
            or os.environ.get("CODEANY_API_KEY", "")
        )
        if not api_key:
            if api_type == "openai-completions":
                api_key = os.environ.get("OPENAI_API_KEY", "")
            else:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        # Resolve base URL: options > CODEANY_BASE_URL
        base_url = (
            self._options.base_url
            or os.environ.get("CODEANY_BASE_URL", "")
        )

        kwargs: dict[str, Any] = {}
        if api_type == "anthropic-messages" and self._options.custom_headers:
            kwargs["default_headers"] = self._options.custom_headers

        self._provider = create_provider(
            api_type,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        return self._provider

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        """Backward-compatible client accessor.

        For Anthropic provider, returns the underlying client.
        For other providers, creates a dummy client (not used in API calls).
        """
        if self._client is None:
            provider = self._ensure_provider()
            if hasattr(provider, "client"):
                self._client = provider.client
            else:
                # Create a client for backward compat (e.g., compact_conversation)
                api_key = (
                    self._options.api_key
                    or os.environ.get("CODEANY_API_KEY", "")
                    or os.environ.get("ANTHROPIC_API_KEY", "")
                )
                base_url = (
                    self._options.base_url
                    or os.environ.get("CODEANY_BASE_URL", "")
                )
                kwargs: dict[str, Any] = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                if self._options.custom_headers:
                    kwargs["default_headers"] = self._options.custom_headers
                self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def _initialize(self) -> None:
        """Initialize tool pool and MCP connections."""
        if self._initialized:
            return

        # Build tool pool
        base_tools = get_all_base_tools()
        if self._options.tools:
            base_tools.extend(self._options.tools)

        self._tool_pool = filter_tools(
            base_tools,
            self._options.allowed_tools,
            self._options.disallowed_tools,
        )

        # Connect MCP servers
        if self._options.mcp_servers:
            from open_agent_sdk.mcp.client import connect_mcp_server

            for name, config in self._options.mcp_servers.items():
                try:
                    conn = await connect_mcp_server(name, config)
                    self._mcp_connections.append(conn)
                    self._tool_pool.extend(conn.tools)
                except Exception as e:
                    if self._options.debug:
                        print(f"Failed to connect MCP server '{name}': {e}")

        # Resume session if needed
        if self._options.resume or self._options.continue_session:
            session_id = self._options.resume or self._session_id
            session_data = await load_session(session_id)
            if session_data:
                self._history = session_data.get("messages", [])
                self._session_id = session_id

        self._initialized = True

    async def query(
        self,
        prompt: str,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncGenerator[SDKMessage, None]:
        """Main agentic loop with streaming. Yields SDKMessage events."""
        await self._initialize()
        provider = self._ensure_provider()
        client = self._ensure_client()

        opts = self._options
        if overrides:
            # Apply overrides (shallow merge)
            opts = AgentOptions(**{**vars(opts), **overrides})

        config = QueryEngineConfig(
            client=client,
            provider=provider,
            model=opts.model or os.environ.get("CODEANY_MODEL", "") or "claude-sonnet-4-5",
            system_prompt=opts.system_prompt,
            append_system_prompt=opts.append_system_prompt,
            tools=self._tool_pool,
            max_turns=opts.max_turns,
            max_budget_usd=opts.max_budget_usd,
            max_tokens=opts.max_tokens,
            can_use_tool=opts.can_use_tool,
            cwd=opts.cwd or os.getcwd(),
            env=opts.env,
            include_partial_messages=opts.include_partial_messages,
            thinking=opts.thinking,
            json_schema=opts.json_schema,
            abort_signal=opts.abort_signal,
            debug=opts.debug,
            extra_args=opts.extra_args,
            betas=opts.betas,
            custom_headers=opts.custom_headers,
        )

        engine = QueryEngine(config)
        engine.messages = list(self._history)
        self._engine = engine

        async for event in engine.submit_message(prompt):
            yield event

        # Update history from engine
        self._history = list(engine.messages)

    async def prompt(
        self,
        text: str,
        overrides: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Convenience: run query and collect final result."""
        start_time = time.monotonic()
        result = QueryResult()
        messages: list[dict[str, Any]] = []

        async for event in self.query(text, overrides):
            if event.type == SDKMessageType.RESULT:
                result.text = event.text
                result.usage = event.total_usage or TokenUsage()
                result.num_turns = event.num_turns
                result.cost = event.total_cost
                result.messages = event.messages
            elif event.type == SDKMessageType.ASSISTANT:
                if event.text:
                    result.text = event.text

        result.duration_ms = int((time.monotonic() - start_time) * 1000)
        return result

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all conversation messages."""
        return list(self._history)

    def clear(self) -> None:
        """Reset conversation history."""
        self._history.clear()

    async def interrupt(self) -> None:
        """Abort current query by setting the abort signal event."""
        signal = self._options.abort_signal
        if signal is not None and hasattr(signal, "set"):
            signal.set()

    async def set_model(self, model: str) -> None:
        """Switch model during session."""
        self._options.model = model

    async def set_permission_mode(self, mode: PermissionMode) -> None:
        """Change permission policy."""
        self._options.permission_mode = mode

    async def set_max_thinking_tokens(self, max_tokens: int | None) -> None:
        """Enable/disable/adjust extended thinking."""
        if max_tokens is None:
            self._options.thinking = None
        else:
            from open_agent_sdk.types import ThinkingConfig

            self._options.thinking = ThinkingConfig(budget_tokens=max_tokens)

    def get_session_id(self) -> str:
        return self._session_id

    async def close(self) -> None:
        """Close MCP connections and optionally persist session."""
        if self._options.persist_session:
            await save_session(
                self._session_id,
                self._history,
                {
                    "id": self._session_id,
                    "cwd": self._options.cwd or os.getcwd(),
                    "model": self._options.model,
                },
            )

        for conn in self._mcp_connections:
            if conn.close:
                try:
                    await conn.close()
                except Exception:
                    pass

        if self._client:
            await self._client.close()
            self._client = None


def create_agent(options: AgentOptions | None = None) -> Agent:
    """Factory function to create an Agent instance."""
    return Agent(options)


async def query(
    prompt: str,
    options: AgentOptions | None = None,
) -> AsyncGenerator[SDKMessage, None]:
    """Standalone query function - creates a temporary agent."""
    agent = Agent(options)
    try:
        async for event in agent.query(prompt):
            yield event
    finally:
        await agent.close()
