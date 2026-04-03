"""QueryEngine - core agentic loop with streaming."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import anthropic

from open_agent_sdk.providers.types import CreateMessageParams, CreateMessageResponse, LLMProvider, NormalizedTool
from open_agent_sdk.providers.anthropic_provider import AnthropicProvider
from open_agent_sdk.types import (
    AgentOptions,
    BaseTool,
    CanUseToolResult,
    PermissionBehavior,
    SDKMessage,
    SDKMessageType,
    SDKResultStatus,
    SDKSystemSubtype,
    TokenUsage,
    ToolContext,
    ToolResult,
)
from open_agent_sdk.utils.compact import (
    AutoCompactState,
    compact_conversation,
    create_auto_compact_state,
    micro_compact_messages,
    should_auto_compact,
)
from open_agent_sdk.utils.context import get_system_context, get_user_context
from open_agent_sdk.utils.messages import (
    extract_text_from_content,
    normalize_messages_for_api,
)
from open_agent_sdk.utils.retry import with_retry, format_api_error, is_auth_error
from open_agent_sdk.utils.tokens import estimate_cost

MAX_CONCURRENCY = int(os.environ.get("AGENT_SDK_MAX_TOOL_CONCURRENCY", "10"))
MAX_OUTPUT_RECOVERY_ATTEMPTS = 3


@dataclass
class QueryEngineConfig:
    client: anthropic.AsyncAnthropic | None = None
    provider: LLMProvider | None = None
    model: str = "claude-sonnet-4-5"
    system_prompt: str = ""
    append_system_prompt: str = ""
    tools: list[BaseTool] = field(default_factory=list)
    max_turns: int = 10
    max_budget_usd: float | None = None
    max_tokens: int = 16000
    can_use_tool: Any = None  # CanUseToolFn
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    include_partial_messages: bool = False
    thinking: Any = None
    json_schema: dict[str, Any] | None = None
    abort_signal: Any = None
    debug: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)
    betas: list[str] = field(default_factory=list)
    custom_headers: dict[str, str] = field(default_factory=dict)


class _ContentBlockAdapter:
    """Adapts a dict content block to look like an object with attributes."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def type(self) -> str:
        return self._data.get("type", "")

    @property
    def text(self) -> str:
        return self._data.get("text", "")

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def input(self) -> Any:
        return self._data.get("input", {})

    @property
    def thinking(self) -> str:
        return self._data.get("thinking", "")


class _UsageAdapter:
    """Adapts a dict usage to look like an object with attributes."""

    def __init__(self, data: dict[str, int]):
        self.input_tokens = data.get("input_tokens", 0)
        self.output_tokens = data.get("output_tokens", 0)
        self.cache_creation_input_tokens = data.get("cache_creation_input_tokens", 0)
        self.cache_read_input_tokens = data.get("cache_read_input_tokens", 0)


class _ProviderResponseAdapter:
    """Adapts CreateMessageResponse to match the duck-typed interface
    that the rest of QueryEngine expects (same shape as anthropic SDK response)."""

    def __init__(self, response: CreateMessageResponse, model: str):
        self.content = [_ContentBlockAdapter(b) for b in response.content]
        self.stop_reason = response.stop_reason
        self.model = model
        self.usage = _UsageAdapter(response.usage)


class QueryEngine:
    """Core agentic loop with streaming events."""

    def __init__(self, config: QueryEngineConfig):
        self._config = config

        # Auto-create provider from client for backward compatibility
        if config.provider is None and config.client is not None:
            config.provider = AnthropicProvider(client=config.client)

        self._messages: list[dict[str, Any]] = []
        self._total_usage = TokenUsage()
        self._total_cost: float = 0.0
        self._turn_count: int = 0
        self._compact_state = create_auto_compact_state()
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in config.tools}

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self._messages = value

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    @property
    def total_cost(self) -> float:
        return self._total_cost

    async def submit_message(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> AsyncGenerator[SDKMessage, None]:
        """Main agentic loop. Yields SDKMessage events."""
        config = self._config
        start_time = time.monotonic()

        # Add user message
        if isinstance(prompt, str):
            user_content = [{"type": "text", "text": prompt}]
        else:
            user_content = prompt

        self._messages.append({"role": "user", "content": user_content})

        # Build system prompt
        system_prompt = await self._build_system_prompt()

        # Emit init event
        yield SDKMessage(
            type=SDKMessageType.SYSTEM,
            subtype=SDKSystemSubtype.INIT,
            system_data={
                "model": config.model,
                "cwd": config.cwd,
                "tools": [t.name for t in config.tools],
            },
        )

        turns_remaining = config.max_turns
        output_recovery_attempts = 0

        while turns_remaining > 0:
            turns_remaining -= 1
            self._turn_count += 1

            # Check budget
            if config.max_budget_usd and self._total_cost >= config.max_budget_usd:
                yield self._make_result_event(
                    SDKResultStatus.ERROR_MAX_BUDGET, start_time
                )
                return

            # Auto-compact if needed
            if should_auto_compact(self._messages, config.model, self._compact_state):
                result = await compact_conversation(
                    config.client, config.model, self._messages, self._compact_state
                )
                self._messages = result["compacted_messages"]
                self._compact_state = result["state"]
                yield SDKMessage(
                    type=SDKMessageType.SYSTEM,
                    subtype=SDKSystemSubtype.COMPACT_BOUNDARY,
                )

            # Micro-compact large tool results
            api_messages = micro_compact_messages(self._messages)
            api_messages = normalize_messages_for_api(api_messages)

            # Build API request
            try:
                response = await self._call_api(system_prompt, api_messages)
            except Exception as e:
                if is_auth_error(e):
                    yield SDKMessage(
                        type=SDKMessageType.RESULT,
                        status=SDKResultStatus.ERROR_DURING_EXECUTION,
                        text=f"Authentication error: {format_api_error(e)}",
                    )
                    return
                raise

            # Extract response data
            usage = self._extract_usage(response)
            self._total_usage = self._total_usage + usage
            self._total_cost += estimate_cost(config.model, usage)

            # Build content blocks
            content_blocks = []
            tool_use_blocks = []
            assistant_text = ""

            for block in response.content:
                if block.type == "text":
                    content_blocks.append({"type": "text", "text": block.text})
                    assistant_text += block.text
                elif block.type == "tool_use":
                    tool_block = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                    content_blocks.append(tool_block)
                    tool_use_blocks.append(tool_block)
                elif block.type == "thinking":
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": getattr(block, "thinking", ""),
                    })

            # Add assistant message to history
            self._messages.append({"role": "assistant", "content": content_blocks})

            # Yield assistant message
            yield SDKMessage(
                type=SDKMessageType.ASSISTANT,
                message={"role": "assistant", "content": content_blocks},
                content_blocks=[],
                usage=usage,
                stop_reason=response.stop_reason or "",
                model=response.model or config.model,
                text=assistant_text,
            )

            # If no tool calls, we're done
            if not tool_use_blocks:
                # Handle max_tokens recovery
                if response.stop_reason == "max_tokens" and output_recovery_attempts < MAX_OUTPUT_RECOVERY_ATTEMPTS:
                    output_recovery_attempts += 1
                    self._messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "Continue from where you left off."}],
                    })
                    turns_remaining += 1
                    continue
                break

            output_recovery_attempts = 0

            # Execute tools
            tool_results = await self._execute_tools(tool_use_blocks)

            # Build tool_result message
            result_content = []
            for tr in tool_results:
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_use_id,
                    "content": tr.content,
                }
                if tr.is_error:
                    result_block["is_error"] = True
                result_content.append(result_block)

                # Yield individual tool result events
                yield SDKMessage(
                    type=SDKMessageType.TOOL_RESULT,
                    tool_use_id=tr.tool_use_id,
                    tool_name=self._find_tool_name(tr.tool_use_id, tool_use_blocks),
                    result_content=tr.content if isinstance(tr.content, str) else json.dumps(tr.content),
                    is_error=tr.is_error,
                )

            self._messages.append({"role": "user", "content": result_content})

        else:
            # Exhausted all turns
            yield self._make_result_event(SDKResultStatus.ERROR_MAX_TURNS, start_time)
            return

        # Success
        yield self._make_result_event(SDKResultStatus.SUCCESS, start_time)

    async def _build_system_prompt(self) -> str | list[dict[str, Any]]:
        """Build the system prompt with context."""
        config = self._config
        parts: list[str] = []

        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append("You are a helpful AI assistant with access to tools.")

        # Add system context
        try:
            sys_context = await get_system_context(config.cwd)
            if sys_context:
                parts.append(f"\n# Environment\n{sys_context}")
        except Exception:
            pass

        # Add user context
        try:
            user_context = await get_user_context(config.cwd)
            if user_context:
                parts.append(f"\n# User Context\n{user_context}")
        except Exception:
            pass

        # Add tool descriptions
        if config.tools:
            tool_names = [t.name for t in config.tools]
            parts.append(f"\n# Available Tools\n{', '.join(tool_names)}")

        if config.append_system_prompt:
            parts.append(config.append_system_prompt)

        return "\n\n".join(parts)

    async def _call_api(
        self,
        system_prompt: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> Any:
        """Call LLM API via provider with retry."""
        config = self._config
        provider = config.provider

        # Build tool definitions for API
        tools_api: list[NormalizedTool] = []
        for tool in config.tools:
            schema = tool.input_schema
            tools_api.append(NormalizedTool(
                name=tool.name,
                description=tool.description,
                input_schema=schema.to_dict(),
            ))

        system_str = system_prompt if isinstance(system_prompt, str) else ""

        thinking_dict = None
        if config.thinking:
            thinking_dict = {
                "type": config.thinking.type,
                "budget_tokens": config.thinking.budget_tokens,
            }

        params = CreateMessageParams(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_str,
            messages=messages,
            tools=tools_api if tools_api else [],
            thinking=thinking_dict,
        )

        async def _do_call():
            return await provider.create_message(params)

        response = await with_retry(_do_call)

        # Wrap CreateMessageResponse in a duck-typed object compatible with
        # the rest of the engine (which expects response.content as list of
        # objects with .type, .text, .id, .name, .input attributes, and
        # response.stop_reason, response.model, response.usage)
        return _ProviderResponseAdapter(response, config.model)

    async def _execute_tools(
        self,
        tool_use_blocks: list[dict[str, Any]],
    ) -> list[ToolResult]:
        """Execute tool calls, concurrent for read-only, serial for mutations."""
        config = self._config
        context = ToolContext(cwd=config.cwd, env=config.env)

        # Partition into read-only (concurrent) and mutations (serial)
        read_only: list[dict[str, Any]] = []
        mutations: list[dict[str, Any]] = []

        for block in tool_use_blocks:
            tool_name = block.get("name", "")
            tool = self._tool_map.get(tool_name)
            if tool and tool.is_read_only(block.get("input")) and tool.is_concurrency_safe(block.get("input")):
                read_only.append(block)
            else:
                mutations.append(block)

        results: list[ToolResult] = []

        # Execute read-only tools concurrently (in batches)
        if read_only:
            for i in range(0, len(read_only), MAX_CONCURRENCY):
                batch = read_only[i : i + MAX_CONCURRENCY]
                batch_results = await asyncio.gather(
                    *[self._execute_single_tool(b, context) for b in batch],
                    return_exceptions=True,
                )
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        results.append(ToolResult(
                            tool_use_id=batch[j].get("id", ""),
                            content=f"Error: {result}",
                            is_error=True,
                        ))
                    else:
                        results.append(result)

        # Execute mutations serially
        for block in mutations:
            result = await self._execute_single_tool(block, context)
            results.append(result)

        return results

    async def _execute_single_tool(
        self,
        block: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a single tool call with permission checking."""
        tool_use_id = block.get("id", "")
        tool_name = block.get("name", "")
        tool_input = block.get("input", {})

        tool = self._tool_map.get(tool_name)
        if not tool:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=f"Error: unknown tool '{tool_name}'",
                is_error=True,
            )

        # Permission check
        if self._config.can_use_tool:
            try:
                permission = await self._config.can_use_tool(tool, tool_input)
                if permission.behavior == PermissionBehavior.DENY:
                    msg = permission.message or f"Permission denied for tool '{tool_name}'"
                    return ToolResult(
                        tool_use_id=tool_use_id,
                        content=msg,
                        is_error=True,
                    )
                if permission.updated_input is not None:
                    tool_input = permission.updated_input
            except Exception as e:
                return ToolResult(
                    tool_use_id=tool_use_id,
                    content=f"Permission check error: {e}",
                    is_error=True,
                )

        # Execute tool
        try:
            result = await tool.call(tool_input, context)
            result.tool_use_id = tool_use_id
            return result
        except Exception as e:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=f"Tool execution error: {e}",
                is_error=True,
            )

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from API response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
        )

    def _find_tool_name(self, tool_use_id: str, blocks: list[dict[str, Any]]) -> str:
        for b in blocks:
            if b.get("id") == tool_use_id:
                return b.get("name", "")
        return ""

    def _make_result_event(self, status: SDKResultStatus, start_time: float) -> SDKMessage:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        text = ""
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant":
                text = extract_text_from_content(msg.get("content", ""))
                if text:
                    break

        return SDKMessage(
            type=SDKMessageType.RESULT,
            status=status,
            text=text,
            num_turns=self._turn_count,
            duration_ms=duration_ms,
            messages=self._messages,
            total_usage=self._total_usage,
            total_cost=self._total_cost,
        )
