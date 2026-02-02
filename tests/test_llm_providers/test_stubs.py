"""Tests for stub LLM providers."""

from __future__ import annotations

import pytest

from rag_pipeline.llm_providers import AnthropicProvider, OCIGenAIProvider, OpenAIProvider


class TestOCIGenAIProvider:
    """Tests for OCI GenAI provider stub."""

    @pytest.fixture
    def provider(self) -> OCIGenAIProvider:
        """Create OCI GenAI stub provider."""
        return OCIGenAIProvider()

    @pytest.mark.asyncio
    async def test_start_raises(self, provider) -> None:
        """Test start raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.start()

    @pytest.mark.asyncio
    async def test_generate_raises(self, provider) -> None:
        """Test generate raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_stream_raises(self, provider) -> None:
        """Test generate_stream raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            async for _ in provider.generate_stream("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_health_check_returns_false(self, provider) -> None:
        """Test health check returns False for stub."""
        result = await provider.health_check()

        assert result is False

    def test_get_model_info_has_status(self, provider) -> None:
        """Test model info contains status field."""
        info = provider.get_model_info()

        assert info["status"] == "stub"


class TestOpenAIProvider:
    """Tests for OpenAI provider stub."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        """Create OpenAI stub provider."""
        return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_start_raises(self, provider) -> None:
        """Test start raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.start()

    @pytest.mark.asyncio
    async def test_generate_raises(self, provider) -> None:
        """Test generate raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.generate("Hello")


class TestAnthropicProvider:
    """Tests for Anthropic provider stub."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        """Create Anthropic stub provider."""
        return AnthropicProvider()

    @pytest.mark.asyncio
    async def test_start_raises(self, provider) -> None:
        """Test start raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.start()

    @pytest.mark.asyncio
    async def test_generate_raises(self, provider) -> None:
        """Test generate raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="stub"):
            await provider.generate("Hello")
