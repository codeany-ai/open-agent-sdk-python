"""LLM Provider Abstraction Types.

Defines a provider interface that normalizes API differences between
Anthropic Messages API and OpenAI Chat Completions API.

Internally the SDK uses Anthropic-like message format as the canonical
representation. Providers convert to/from their native API format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# API Type
# --------------------------------------------------------------------------

ApiType = Literal["anthropic-messages", "openai-completions"]

# --------------------------------------------------------------------------
# Normalized Request
# --------------------------------------------------------------------------


@dataclass
class NormalizedTool:
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class CreateMessageParams:
    model: str = ""
    max_tokens: int = 16000
    system: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[NormalizedTool] = field(default_factory=list)
    thinking: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Normalized Response
# --------------------------------------------------------------------------


@dataclass
class CreateMessageResponse:
    """Normalized response from any LLM provider."""
    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    })


# --------------------------------------------------------------------------
# Provider Interface
# --------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that all LLM providers must implement."""

    @property
    def api_type(self) -> ApiType: ...

    async def create_message(self, params: CreateMessageParams) -> CreateMessageResponse: ...
