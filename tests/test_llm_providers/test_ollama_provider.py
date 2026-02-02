"""Tests for OllamaProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_pipeline.core.exceptions import LLMProviderError, RetryableError
from rag_pipeline.llm_providers import ChatMessage, OllamaProvider


class TestOllamaProvider:
    """Test suite for OllamaProvider."""

    @pytest.fixture
    def provider(self, ollama_config):
        """Create provider instance."""
        return OllamaProvider(config=ollama_config)

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, provider) -> None:
        """Test async context manager lifecycle."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch(
            "rag_pipeline.llm_providers.ollama_provider.httpx.AsyncClient",
            return_value=mock_client,
        ):
            async with provider:
                assert provider._started is True
                assert provider._client is mock_client

        assert provider._started is False
        assert provider._client is None
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_requires_start(self, provider) -> None:
        """Test that generate raises if not started."""
        with pytest.raises(LLMProviderError, match="Provider not started"):
            await provider.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider, mock_ollama_response) -> None:
        """Test successful generation."""
        with patch.object(
            provider,
            "_request_with_retry",
            new_callable=AsyncMock,
            return_value=mock_ollama_response,
        ):
            response = await provider.generate("Hello")

        assert response == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, provider, mock_ollama_response) -> None:
        """Test generation with system prompt."""
        with patch.object(
            provider,
            "_request_with_retry",
            new_callable=AsyncMock,
            return_value=mock_ollama_response,
        ) as mock_request:
            await provider.generate("Hello", system_prompt="You are helpful.")

        payload = mock_request.call_args.kwargs.get("json", {})
        messages = payload.get("messages", [])

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_context(self, provider) -> None:
        """Test RAG context generation."""
        with patch.object(
            provider,
            "generate",
            new_callable=AsyncMock,
            return_value="Based on the context...",
        ) as mock_generate:
            response = await provider.generate_with_context(
                query="What is RAG?",
                context="[1] RAG stands for Retrieval-Augmented Generation.",
            )

        assert "Based on" in response
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == provider.DEFAULT_RAG_SYSTEM_PROMPT
        assert "Context:" in call_kwargs["prompt"]
        assert "Question: What is RAG?" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_streaming_generation(self, provider, mock_stream_chunks) -> None:
        """Test streaming response."""

        async def mock_stream():
            for chunk in mock_stream_chunks:
                yield json.dumps(chunk)

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream
        mock_response.raise_for_status = MagicMock()

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_context

        provider._client = mock_client
        provider._started = True

        chunks = []
        async for chunk in provider.generate_stream("Hello"):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello there!"

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, provider) -> None:
        """Test retry logic on connection errors with exponential backoff."""
        call_count = 0
        request = httpx.Request("POST", "http://localhost")
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"message": {"content": "Success"}}

        async def mock_request(method: str, url: str, **request_kwargs: object):
            nonlocal call_count
            assert method == "POST"
            assert url == "/api/chat"
            assert "json" in request_kwargs
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed", request=request)
            return response

        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=mock_request)
        provider._client = mock_client
        provider._started = True
        provider.config.max_retries = 3

        with patch(
            "rag_pipeline.llm_providers.ollama_provider.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            result = await provider._request_with_retry("POST", "/api/chat", json={})

        assert result["message"]["content"] == "Success"
        assert call_count == 3
        assert [call.args[0] for call in mock_sleep.call_args_list] == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, provider) -> None:
        """Test that exhausted retries raise RetryableError."""
        request = httpx.Request("POST", "http://localhost")
        mock_client = MagicMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.TimeoutException("Timed out", request=request)
        )
        provider._client = mock_client
        provider._started = True
        provider.config.max_retries = 2

        sleep_patch = patch(
            "rag_pipeline.llm_providers.ollama_provider.asyncio.sleep",
            new_callable=AsyncMock,
        )
        with (
            sleep_patch as mock_sleep,
            pytest.raises(RetryableError, match="All retry attempts failed"),
        ):
            await provider._request_with_retry("POST", "/api/chat", json={})

        mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider, mock_models_response) -> None:
        """Test health check with available model."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_models_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        provider._started = True

        result = await provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self, provider) -> None:
        """Test health check when model is not available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama2:latest"}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        provider._started = True

        result = await provider.health_check()

        assert result is False

    def test_get_model_info(self, provider) -> None:
        """Test model info retrieval."""
        info = provider.get_model_info()

        assert info["name"] == "mistral"
        assert info["provider"] == "ollama"
        assert info["base_url"] == "http://localhost:11434"
        assert "chat" in info["capabilities"]

    @pytest.mark.asyncio
    async def test_chat_messages(self, provider, mock_ollama_response) -> None:
        """Test chat with message list."""
        with patch.object(
            provider,
            "_request_with_retry",
            new_callable=AsyncMock,
            return_value=mock_ollama_response,
        ) as mock_request:
            messages = [
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!"),
            ]
            await provider.chat(messages)

        payload = mock_request.call_args.kwargs.get("json", {})
        assert payload["messages"] == [message.to_dict() for message in messages]

    @pytest.mark.asyncio
    async def test_non_retryable_http_error_raises(self, provider) -> None:
        """Test non-retryable HTTPStatusError raises LLMProviderError."""
        request = httpx.Request("POST", "http://localhost")
        response = httpx.Response(400, request=request, text="Bad Request")
        http_error = httpx.HTTPStatusError("Bad Request", request=request, response=response)

        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=http_error)
        provider._client = mock_client
        provider._started = True

        with pytest.raises(LLMProviderError, match="Ollama API error: 400"):
            await provider._request_with_retry("POST", "/api/chat", json={})


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.to_dict() == {"role": "user", "content": "Hello"}
