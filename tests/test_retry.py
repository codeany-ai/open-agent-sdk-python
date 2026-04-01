"""Tests for retry logic."""

import pytest
from open_agent_sdk.utils.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    format_api_error,
    get_retry_delay,
    is_auth_error,
    is_prompt_too_long_error,
    is_rate_limit_error,
    is_retryable_error,
    with_retry,
)


class TestRetryConfig:
    def test_defaults(self):
        config = DEFAULT_RETRY_CONFIG
        assert config.max_retries == 3
        assert config.base_delay_ms == 2000
        assert config.max_delay_ms == 30000
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes
        assert 529 in config.retryable_status_codes


class FakeError(Exception):
    def __init__(self, status_code=None, message="error"):
        self.status_code = status_code
        super().__init__(message)


class TestIsRetryableError:
    def test_429_is_retryable(self):
        assert is_retryable_error(FakeError(status_code=429)) is True

    def test_500_is_retryable(self):
        assert is_retryable_error(FakeError(status_code=500)) is True

    def test_400_not_retryable(self):
        assert is_retryable_error(FakeError(status_code=400)) is False

    def test_connection_error(self):
        class ConnectionError(Exception):
            pass
        assert is_retryable_error(ConnectionError()) is True


class TestIsAuthError:
    def test_401(self):
        assert is_auth_error(FakeError(status_code=401)) is True

    def test_403(self):
        assert is_auth_error(FakeError(status_code=403)) is True

    def test_200(self):
        assert is_auth_error(FakeError(status_code=200)) is False


class TestIsRateLimitError:
    def test_429(self):
        assert is_rate_limit_error(FakeError(status_code=429)) is True

    def test_500(self):
        assert is_rate_limit_error(FakeError(status_code=500)) is False


class TestIsPromptTooLong:
    def test_prompt_too_long(self):
        assert is_prompt_too_long_error(Exception("prompt is too long")) is True

    def test_context_length(self):
        assert is_prompt_too_long_error(Exception("context_length_exceeded")) is True

    def test_other(self):
        assert is_prompt_too_long_error(Exception("some other error")) is False


class TestGetRetryDelay:
    def test_increases_with_attempt(self):
        d0 = get_retry_delay(0)
        d1 = get_retry_delay(1)
        d2 = get_retry_delay(2)
        # Due to jitter, compare base values
        assert d0 < d2  # Generally true, but jitter makes it non-deterministic

    def test_respects_max(self):
        config = RetryConfig(base_delay_ms=1000, max_delay_ms=5000)
        delay = get_retry_delay(10, config)  # Very high attempt
        assert delay <= 6.25  # max_delay_ms/1000 + 25% jitter

    def test_returns_seconds(self):
        delay = get_retry_delay(0)
        assert delay > 0
        assert delay < 10  # Should be reasonable


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(fn)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeError(status_code=500)
            return "recovered"

        config = RetryConfig(max_retries=3, base_delay_ms=10, max_delay_ms=50)
        result = await with_retry(fn, config)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        async def fn():
            raise FakeError(status_code=400)

        with pytest.raises(FakeError):
            await with_retry(fn, RetryConfig(max_retries=3, base_delay_ms=10))

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        async def fn():
            raise FakeError(status_code=500)

        config = RetryConfig(max_retries=2, base_delay_ms=10, max_delay_ms=50)
        with pytest.raises(FakeError):
            await with_retry(fn, config)


class TestFormatApiError:
    def test_with_status(self):
        result = format_api_error(FakeError(status_code=500, message="server error"))
        assert "500" in result
        assert "server error" in result

    def test_without_status(self):
        result = format_api_error(Exception("generic error"))
        assert "generic error" in result
        assert "API Error" in result
