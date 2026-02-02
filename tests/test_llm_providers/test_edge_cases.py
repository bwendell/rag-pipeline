"""Edge case tests for LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_pipeline.llm_providers import OllamaProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class TestPromptEdgeCases:
    """Edge cases for prompt inputs and parameters."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        provider = OllamaProvider()
        provider._started = True
        provider._client = MagicMock()
        return provider

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, provider: OllamaProvider) -> None:
        """Very long prompts should be accepted without errors."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "ok"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        prompt = "x" * 10000
        response = await provider.generate(prompt)

        assert response == "ok"

    @pytest.mark.asyncio
    async def test_unicode_content(self, provider: OllamaProvider) -> None:
        """Unicode prompts should be handled correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        prompt = "こんにちは 🌍 Привет мир"
        response = await provider.generate(prompt)

        assert response == "Response"

    @pytest.mark.asyncio
    async def test_empty_prompt(self, provider: OllamaProvider) -> None:
        """Empty prompt should still call the provider."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": ""},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        response = await provider.generate("")

        assert response == ""

    @pytest.mark.asyncio
    async def test_special_characters(self, provider: OllamaProvider) -> None:
        """Special characters should be preserved."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "ok"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        prompt = 'Line1\nLine2\t"quoted"'
        response = await provider.generate(prompt)

        assert response == "ok"

    @pytest.mark.asyncio
    async def test_max_tokens_parameter(self, provider: OllamaProvider) -> None:
        """max_tokens should be passed to payload as num_predict."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "ok"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        await provider.generate("test", max_tokens=12)

        payload = client.request.call_args.kwargs.get("json", {})
        assert payload["options"]["num_predict"] == 12

    @pytest.mark.asyncio
    async def test_stop_sequences_parameter(self, provider: OllamaProvider) -> None:
        """stop_sequences should be passed to payload as stop list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "ok"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        stops = ["END", "STOP"]
        await provider.generate("test", stop_sequences=stops)

        payload = client.request.call_args.kwargs.get("json", {})
        assert payload["options"]["stop"] == stops

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temperature", [0.0, 1.0])
    async def test_temperature_edge_values(
        self, provider: OllamaProvider, temperature: float
    ) -> None:
        """Temperature edge values should be passed through."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "ok"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        client = MagicMock()
        client.request = AsyncMock(return_value=mock_response)
        provider._client = client

        await provider.generate("test", temperature=temperature)

        payload = client.request.call_args.kwargs.get("json", {})
        assert payload["options"]["temperature"] == temperature


class TestStreamingUnicode:
    """Edge cases for streaming content."""

    @pytest.mark.asyncio
    async def test_unicode_streaming_content(self) -> None:
        """Streaming should handle Unicode chunks correctly."""
        provider = OllamaProvider()
        provider._started = True

        async def mock_lines() -> AsyncIterator[str]:
            yield '{"message": {"content": "こん"}, "done": false}'
            yield '{"message": {"content": "にちは"}, "done": false}'
            yield '{"message": {"content": " 🌍"}, "done": true}'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_lines
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        mock_client.stream.return_value.__aexit__.return_value = None
        provider._client = mock_client

        chunks = [chunk async for chunk in provider.generate_stream("test")]

        assert "".join(chunks) == "こんにちは 🌍"
