"""Anthropic Messages API Provider.

Wraps the anthropic AsyncAnthropic client. Since our internal format is
Anthropic-like, this is mostly a thin pass-through.
"""

from __future__ import annotations

from typing import Any

import anthropic

from open_agent_sdk.providers.types import (
    ApiType,
    CreateMessageParams,
    CreateMessageResponse,
    LLMProvider,
)


class AnthropicProvider:
    """LLM provider for Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        client: anthropic.AsyncAnthropic | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            if default_headers:
                kwargs["default_headers"] = default_headers
            self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def api_type(self) -> ApiType:
        return "anthropic-messages"

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        return self._client

    async def create_message(self, params: CreateMessageParams) -> CreateMessageResponse:
        kwargs: dict[str, Any] = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": params.messages,
        }

        if params.system:
            kwargs["system"] = params.system

        if params.tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in params.tools
            ]

        if params.thinking and params.thinking.get("type") == "enabled":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking.get("budget_tokens", 10000),
            }

        response = await self._client.messages.create(**kwargs)

        # Convert response content to normalized dicts
        content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                content.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                })

        usage = response.usage
        return CreateMessageResponse(
            content=content,
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
            },
        )
