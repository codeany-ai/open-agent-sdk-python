"""Tests for token estimation and cost calculation."""

import pytest
from open_agent_sdk.types import TokenUsage
from open_agent_sdk.utils.tokens import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MODEL_PRICING,
    estimate_cost,
    estimate_messages_tokens,
    estimate_system_prompt_tokens,
    estimate_tokens,
    get_auto_compact_threshold,
    get_context_window_size,
    get_token_count_from_usage,
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens <= 10

    def test_longer_text(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100  # ~4 chars per token


class TestEstimateMessagesTokens:
    def test_empty(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0

    def test_message_with_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response text here"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            }
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0

    def test_overhead_per_message(self):
        one_msg = estimate_messages_tokens([{"role": "user", "content": "Hi"}])
        two_msgs = estimate_messages_tokens([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        assert two_msgs > one_msg


class TestEstimateSystemPromptTokens:
    def test_string(self):
        tokens = estimate_system_prompt_tokens("You are a helpful assistant.")
        assert tokens > 0

    def test_list(self):
        tokens = estimate_system_prompt_tokens([
            {"type": "text", "text": "System prompt"},
            {"type": "text", "text": "More context"},
        ])
        assert tokens > 0


class TestGetTokenCountFromUsage:
    def test_basic(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert get_token_count_from_usage(usage) == 150

    def test_with_cache(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=10,
        )
        assert get_token_count_from_usage(usage) == 180


class TestContextWindowSize:
    def test_known_models(self):
        assert get_context_window_size("claude-opus-4-6") == 1_000_000
        assert get_context_window_size("claude-sonnet-4-5") == 200_000

    def test_unknown_model(self):
        assert get_context_window_size("unknown-model") == 200_000

    def test_prefix_match(self):
        assert get_context_window_size("claude-sonnet-4-5") == 200_000


class TestAutoCompactThreshold:
    def test_basic(self):
        threshold = get_auto_compact_threshold("claude-sonnet-4-5")
        expected = 200_000 - AUTOCOMPACT_BUFFER_TOKENS
        assert threshold == expected

    def test_opus_1m(self):
        threshold = get_auto_compact_threshold("claude-opus-4-6")
        assert threshold == 1_000_000 - AUTOCOMPACT_BUFFER_TOKENS


class TestEstimateCost:
    def test_sonnet(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = estimate_cost("claude-sonnet-4-5", usage)
        assert cost == pytest.approx(3.0 + 15.0, rel=0.01)

    def test_opus(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = estimate_cost("claude-opus-4-6", usage)
        assert cost == pytest.approx(15.0 + 75.0, rel=0.01)

    def test_with_cache(self):
        usage = TokenUsage(
            input_tokens=500_000,
            output_tokens=100_000,
            cache_creation_input_tokens=200_000,
            cache_read_input_tokens=300_000,
        )
        cost = estimate_cost("claude-sonnet-4-5", usage)
        assert cost > 0

    def test_unknown_model_defaults(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("some-unknown-model", usage)
        assert cost > 0


class TestModelPricing:
    def test_all_models_have_input_output(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert pricing["input"] > 0
            assert pricing["output"] > 0
