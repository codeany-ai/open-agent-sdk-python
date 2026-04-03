"""Token estimation and cost calculation utilities."""

from __future__ import annotations

import json
from typing import Any

from open_agent_sdk.types import TokenUsage

# Buffer tokens reserved for auto-compaction
AUTOCOMPACT_BUFFER_TOKENS = 13_000

# Model pricing per million tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0},
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-latest": {"input": 0.8, "output": 4.0},
    "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
    # OpenAI models
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "o1-preview": {"input": 15.0, "output": 60.0},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o4-mini": {"input": 1.1, "output": 4.4},
    # DeepSeek models
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Qwen models
    "qwen-turbo": {"input": 0.3, "output": 0.6},
    "qwen-plus": {"input": 0.8, "output": 2.0},
    "qwen-max": {"input": 2.0, "output": 6.0},
}

# Context window sizes
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-latest": 200_000,
    "claude-3-opus-latest": 200_000,
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # DeepSeek models
    "deepseek-chat": 128_000,
    "deepseek-coder": 128_000,
    "deepseek-reasoner": 128_000,
    # Qwen models
    "qwen-turbo": 131_072,
    "qwen-plus": 131_072,
    "qwen-max": 32_768,
}

DEFAULT_CONTEXT_WINDOW = 200_000


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (~4 chars per token, conservative)."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total += estimate_tokens(text)
                    # For tool_use, estimate the JSON input
                    input_data = block.get("input")
                    if input_data:
                        total += estimate_tokens(json.dumps(input_data))
                    # For tool_result content
                    result_content = block.get("content", "")
                    if isinstance(result_content, str) and result_content:
                        total += estimate_tokens(result_content)
        # Overhead per message
        total += 4
    return total


def estimate_system_prompt_tokens(system_prompt: str | list[dict[str, Any]]) -> int:
    """Estimate tokens in system prompt."""
    if isinstance(system_prompt, str):
        return estimate_tokens(system_prompt)
    total = 0
    for block in system_prompt:
        if isinstance(block, dict):
            total += estimate_tokens(block.get("text", ""))
    return total


def get_token_count_from_usage(usage: TokenUsage) -> int:
    """Get total token count from usage object."""
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )


def get_context_window_size(model: str) -> int:
    """Get context window size for a model."""
    # Check exact match
    if model in _CONTEXT_WINDOWS:
        return _CONTEXT_WINDOWS[model]
    # Check prefix match
    for key, size in _CONTEXT_WINDOWS.items():
        if model.startswith(key):
            return size
    # Check for 1M context models
    if "opus-4-6" in model:
        return 1_000_000
    return DEFAULT_CONTEXT_WINDOW


def get_auto_compact_threshold(model: str) -> int:
    """Get auto-compaction threshold (context window - buffer)."""
    return get_context_window_size(model) - AUTOCOMPACT_BUFFER_TOKENS


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Estimate cost in USD for a given model and usage."""
    pricing = None
    for key, p in MODEL_PRICING.items():
        if model.startswith(key) or key.startswith(model):
            pricing = p
            break
    if pricing is None:
        pricing = {"input": 3.0, "output": 15.0}

    input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]

    # Cache pricing: creation = same as input, read = 10% of input
    cache_create_cost = (usage.cache_creation_input_tokens / 1_000_000) * pricing["input"]
    cache_read_cost = (usage.cache_read_input_tokens / 1_000_000) * pricing["input"] * 0.1

    return input_cost + output_cost + cache_create_cost + cache_read_cost
