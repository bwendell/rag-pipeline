"""Async lifecycle and streaming cancellation tests for Ollama provider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_pipeline.core.exceptions import LLMProviderError
from rag_pipeline.llm_providers import OllamaProvider
from rag_pipeline.llm_providers.ollama_provider import ChatMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class TestAsyncLifecycle:
    """Test async context manager and lifecycle methods."""

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self) -> None:
        """Calling start() twice should be safe and reuse client."""
        provider = OllamaProvider()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        with patch(
            "rag_pipeline.llm_providers.ollama_provider.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await provider.start()
            client_first = provider._client
            await provider.start()
            client_second = provider._client

        assert client_first is client_second
        assert provider._started is True

        await provider.stop()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_double_stop_is_idempotent(self) -> None:
        """Calling stop() twice should be safe."""
        provider = OllamaProvider()

        with patch(
            "rag_pipeline.llm_providers.ollama_provider.httpx.AsyncClient",
            return_value=MagicMock(aclose=AsyncMock()),
        ):
            await provider.start()
        await provider.stop()
        await provider.stop()

        assert provider._started is False
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self) -> None:
        """Calling stop() without start() should be safe."""
        provider = OllamaProvider()

        await provider.stop()

        assert provider._started is False

    @pytest.mark.asyncio
    async def test_use_after_stop_raises(self) -> None:
        """Using provider after stop should raise LLMProviderError."""
        provider = OllamaProvider()

        with patch(
            "rag_pipeline.llm_providers.ollama_provider.httpx.AsyncClient",
            return_value=MagicMock(aclose=AsyncMock()),
        ):
            await provider.start()
        await provider.stop()

        with pytest.raises(LLMProviderError, match="not started"):
            await provider.generate("test")

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self) -> None:
        """Context manager should cleanup even on exception."""
        provider = OllamaProvider()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        try:
            with patch(
                "rag_pipeline.llm_providers.ollama_provider.httpx.AsyncClient",
                return_value=mock_client,
            ):
                async with provider:
                    assert provider._started is True
                    raise ValueError("Test exception")
        except ValueError:
            pass

        assert provider._started is False
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_generate_before_start_raises(self) -> None:
        """Using generate before start should raise LLMProviderError."""
        provider = OllamaProvider()

        with pytest.raises(LLMProviderError, match="not started"):
            await provider.generate("test")

    @pytest.mark.asyncio
    async def test_chat_before_start_raises(self) -> None:
        """Using chat before start should raise LLMProviderError."""
        provider = OllamaProvider()

        with pytest.raises(LLMProviderError, match="not started"):
            await provider.chat([ChatMessage(role="user", content="test")])


class TestStreamingCancellation:
    """Test streaming cancellation and cleanup."""

    @pytest.mark.asyncio
    async def test_streaming_cancellation(self) -> None:
        """Streaming generator can be cancelled mid-stream."""
        provider = OllamaProvider()
        provider._started = True

        async def mock_lines() -> AsyncIterator[str]:
            for i in range(5):
                yield f'{{"message": {{"content": "chunk{i}"}}, "done": false}}'
                await asyncio.sleep(0)
            yield '{"message": {"content": "done"}, "done": true}'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        mock_client.stream.return_value.__aexit__.return_value = None
        provider._client = mock_client

        async def consume() -> None:
            async for _ in provider.generate_stream("test"):
                await asyncio.sleep(0)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_stream_timeout_during_iteration(self) -> None:
        """Timeout during stream iteration should raise."""
        provider = OllamaProvider()
        provider._started = True

        async def timeout_lines() -> AsyncIterator[str]:
            yield '{"message": {"content": "start"}, "done": false}'
            raise httpx.ReadTimeout("timeout", request=MagicMock())

        mock_response = MagicMock()
        mock_response.aiter_lines = timeout_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        mock_client.stream.return_value.__aexit__.return_value = None
        provider._client = mock_client

        with pytest.raises(httpx.ReadTimeout):
            async for _ in provider.generate_stream("test"):
                pass
