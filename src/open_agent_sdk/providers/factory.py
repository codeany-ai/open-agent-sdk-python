"""LLM Provider Factory.

Creates the appropriate provider based on API type configuration.
"""

from __future__ import annotations

from typing import Any

from open_agent_sdk.providers.types import ApiType, LLMProvider
from open_agent_sdk.providers.anthropic_provider import AnthropicProvider
from open_agent_sdk.providers.openai_provider import OpenAIProvider


def create_provider(
    api_type: ApiType,
    *,
    api_key: str = "",
    base_url: str = "",
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider based on the API type.

    Args:
        api_type: 'anthropic-messages' or 'openai-completions'
        api_key: API key for the provider
        base_url: Base URL override
        **kwargs: Additional provider-specific options
    """
    if api_type == "anthropic-messages":
        return AnthropicProvider(api_key=api_key, base_url=base_url, **kwargs)
    elif api_type == "openai-completions":
        return OpenAIProvider(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(
            f"Unsupported API type: {api_type}. "
            "Use 'anthropic-messages' or 'openai-completions'."
        )
