"""Comprehensive error handling tests for Ollama provider.

Tests cover:
- Network layer: ConnectError, TimeoutException
- HTTP layer: retryable vs non-retryable status codes
- JSON parsing: invalid JSON in streaming
- Response validation: missing keys and empty content
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_pipeline.core.exceptions import LLMProviderError, RetryableError
from rag_pipeline.llm_providers import OllamaConfig, OllamaProvider
from rag_pipeline.llm_providers.ollama_provider import ChatMessage


class TestNetworkErrors:
    """Test network-level error handling."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        provider = OllamaProvider(config=OllamaConfig(max_retries=2))
        provider._started = True
        provider._client = MagicMock()
        return provider

    @pytest.mark.asyncio
    async def test_connect_error_triggers_retry(self, provider: OllamaProvider) -> None:
        """ConnectError should trigger retry with exponential backoff."""
        call_count = 0

        async def failing_request(*args: object, **kwargs: object) -> httpx.Response:
            nonlocal call_count
            _ = (args, kwargs)
            call_count += 1
            request = httpx.Request("POST", "http://localhost")
            raise httpx.ConnectError("Connection refused", request=request)

        provider._client = MagicMock()
        provider._client.request = AsyncMock(side_effect=failing_request)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RetryableError) as exc_info,
        ):
            await provider._request_with_retry("POST", "/api/chat", json={})

        assert call_count == 2
        assert "retry attempts failed" in str(exc_info.value).lower()
        assert exc_info.value.details["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_timeout_error_triggers_retry(self, provider: OllamaProvider) -> None:
        """TimeoutException should trigger retry."""

        async def timeout_request(*args: object, **kwargs: object) -> httpx.Response:
            _ = (args, kwargs)
            request = httpx.Request("POST", "http://localhost")
            raise httpx.TimeoutException("Request timed out", request=request)

        provider._client = MagicMock()
        provider._client.request = AsyncMock(side_effect=timeout_request)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RetryableError) as exc_info,
        ):
            await provider._request_with_retry("POST", "/api/chat", json={})

        assert exc_info.value.retry_after > 0


class TestHTTPErrors:
    """Test HTTP status code error handling."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        provider = OllamaProvider(config=OllamaConfig(max_retries=2))
        provider._started = True
        provider._client = MagicMock()
        return provider

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 500])
    async def test_non_retryable_status_raises_immediately(
        self, provider: OllamaProvider, status_code: int
    ) -> None:
        """Non-retryable status codes should raise LLMProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = "Bad request"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=mock_response,
        )

        provider._client = MagicMock()
        provider._client.request = AsyncMock(return_value=mock_response)

        with pytest.raises(LLMProviderError) as exc_info:
            await provider._request_with_retry("POST", "/api/chat", json={})

        assert exc_info.value.details["status_code"] == status_code

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [429, 503, 504])
    async def test_retryable_status_codes_trigger_retry(
        self, provider: OllamaProvider, status_code: int
    ) -> None:
        """Retryable status codes should trigger retry and raise RetryableError."""
        call_count = 0

        async def retryable_request(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            _ = (args, kwargs)
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.text = "Service unavailable"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Retryable",
                request=MagicMock(),
                response=mock_response,
            )
            return mock_response

        provider._client = MagicMock()
        provider._client.request = AsyncMock(side_effect=retryable_request)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RetryableError),
        ):
            await provider._request_with_retry("POST", "/api/chat", json={})

        assert call_count == 2


class TestJSONParsingErrors:
    """Test JSON parsing error handling."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        provider = OllamaProvider()
        provider._started = True
        return provider

    @pytest.mark.asyncio
    async def test_invalid_json_in_stream_logs_warning(self, provider: OllamaProvider) -> None:
        """Invalid JSON lines should log warning and continue."""

        async def mock_lines():
            yield "not valid json"
            yield '{"message": {"content": "hello"}, "done": false}'
            yield '{"message": {"content": " world"}, "done": true}'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        mock_client.stream.return_value.__aexit__.return_value = None
        provider._client = mock_client

        with patch("rag_pipeline.llm_providers.ollama_provider.logger") as logger_mock:
            chunks = [chunk async for chunk in provider.generate_stream("test")]

        assert "".join(chunks) == "hello world"
        logger_mock.warning.assert_called()

    @pytest.mark.asyncio
    async def test_empty_lines_in_stream_ignored(self, provider: OllamaProvider) -> None:
        """Empty lines in stream should be ignored."""

        async def mock_lines():
            yield ""
            yield '{"message": {"content": "test"}, "done": true}'
            yield ""

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        mock_client.stream.return_value.__aexit__.return_value = None
        provider._client = mock_client

        chunks = [chunk async for chunk in provider.generate_stream("test")]

        assert "".join(chunks) == "test"


class TestResponseValidation:
    """Test response validation and edge cases."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        provider = OllamaProvider()
        provider._started = True
        provider._client = MagicMock()
        return provider

    @pytest.mark.asyncio
    async def test_missing_message_key_in_response(self, provider: OllamaProvider) -> None:
        """Missing 'message' key should raise."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"model": "mistral", "done": True}
        mock_response.raise_for_status = MagicMock()

        provider._client = MagicMock()
        provider._client.request = AsyncMock(return_value=mock_response)

        with pytest.raises(KeyError):
            await provider.chat([ChatMessage(role="user", content="test")])

    @pytest.mark.asyncio
    async def test_empty_content_in_response(self, provider: OllamaProvider) -> None:
        """Empty content should return an empty string."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        provider._client = MagicMock()
        provider._client.request = AsyncMock(return_value=mock_response)

        result = await provider.chat([ChatMessage(role="user", content="test")])

        assert result == ""
