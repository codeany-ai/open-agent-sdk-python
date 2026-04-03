"""OpenAI Chat Completions API Provider.

Converts between the SDK's internal Anthropic-like message format
and OpenAI's Chat Completions API format.

Uses urllib.request with asyncio.get_event_loop().run_in_executor()
to avoid adding new dependencies (same approach as WebFetchTool).
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
import urllib.error
from typing import Any

from open_agent_sdk.providers.types import (
    ApiType,
    CreateMessageParams,
    CreateMessageResponse,
    NormalizedTool,
)


class OpenAIProvider:
    """LLM provider for OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
    ):
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    @property
    def api_type(self) -> ApiType:
        return "openai-completions"

    async def create_message(self, params: CreateMessageParams) -> CreateMessageResponse:
        messages = self._convert_messages(params.system, params.messages)
        tools = self._convert_tools(params.tools) if params.tools else None

        body: dict[str, Any] = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": messages,
        }

        if tools:
            body["tools"] = tools

        data = await self._post_chat_completions(body)
        return self._convert_response(data)

    # --------------------------------------------------------------------------
    # HTTP
    # --------------------------------------------------------------------------

    async def _post_chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=300),
            )
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"OpenAI API error: {e.code} {e.reason}: {err_body}"
            ) from e

    # --------------------------------------------------------------------------
    # Message Conversion: Internal -> OpenAI
    # --------------------------------------------------------------------------

    def _convert_messages(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            if role == "user":
                self._convert_user_message(msg, result)
            elif role == "assistant":
                self._convert_assistant_message(msg, result)

        return result

    def _convert_user_message(
        self,
        msg: dict[str, Any],
        result: list[dict[str, Any]],
    ) -> None:
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": "user", "content": content})
            return

        # Content blocks may contain text and/or tool_result blocks
        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_result":
                tool_results.append({
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": block.get("content", ""),
                })

        # Tool results become separate tool messages
        for tr in tool_results:
            content_val = tr["content"]
            if not isinstance(content_val, str):
                content_val = json.dumps(content_val)
            result.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": content_val,
            })

        # Text parts become a user message
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})

    def _convert_assistant_message(
        self,
        msg: dict[str, Any],
        result: list[dict[str, Any]],
    ) -> None:
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": "assistant", "content": content})
            return

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                input_val = block.get("input", {})
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": input_val if isinstance(input_val, str) else json.dumps(input_val),
                    },
                })

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        result.append(assistant_msg)

    # --------------------------------------------------------------------------
    # Tool Conversion: Internal -> OpenAI
    # --------------------------------------------------------------------------

    def _convert_tools(self, tools: list[NormalizedTool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    # --------------------------------------------------------------------------
    # Response Conversion: OpenAI -> Internal
    # --------------------------------------------------------------------------

    def _convert_response(self, data: dict[str, Any]) -> CreateMessageResponse:
        choices = data.get("choices", [])
        if not choices:
            return CreateMessageResponse(
                content=[{"type": "text", "text": ""}],
                stop_reason="end_turn",
                usage={"input_tokens": 0, "output_tokens": 0,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            )

        choice = choices[0]
        message = choice.get("message", {})
        content: list[dict[str, Any]] = []

        # Text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Tool calls
        for tc in (message.get("tool_calls") or []):
            func = tc.get("function", {})
            try:
                input_val = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                input_val = func.get("arguments", "")

            content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": input_val,
            })

        if not content:
            content.append({"type": "text", "text": ""})

        # Map finish_reason
        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = self._map_finish_reason(finish_reason)

        usage_data = data.get("usage", {})
        return CreateMessageResponse(
            content=content,
            stop_reason=stop_reason,
            usage={
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return mapping.get(reason, reason)
