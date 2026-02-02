"""Tests for LLM provider factory."""

from __future__ import annotations

import pytest

from rag_pipeline.core.exceptions import ConfigurationError
from rag_pipeline.llm_providers import LLMProviderFactory, OllamaProvider, create_llm_provider
from rag_pipeline.llm_providers.stubs import OCIGenAIProvider


class TestLLMProviderFactory:
    """Test suite for LLMProviderFactory."""

    @pytest.fixture
    def factory(self) -> LLMProviderFactory:
        """Create a fresh factory instance."""
        return LLMProviderFactory()

    def test_create_ollama_provider(self, factory) -> None:
        """Test creating Ollama provider."""
        provider = factory.create_provider("ollama")

        assert isinstance(provider, OllamaProvider)

    def test_create_with_config_kwargs(self, factory) -> None:
        """Test creating provider with config kwargs."""
        provider = factory.create_provider("ollama", model="codellama")

        assert provider.config.model == "codellama"

    def test_create_oci_provider(self, factory) -> None:
        """Test creating OCI provider (stub)."""
        provider = factory.create_provider("oci_genai")

        assert isinstance(provider, OCIGenAIProvider)

    def test_create_with_alias(self, factory) -> None:
        """Test creating provider with alias."""
        provider = factory.create_provider("oci")

        assert isinstance(provider, OCIGenAIProvider)

    def test_unknown_provider_raises(self, factory) -> None:
        """Test that unknown provider type raises."""
        with pytest.raises(ConfigurationError, match="Unknown LLM provider type"):
            factory.create_provider("unknown_provider")

    def test_list_providers(self, factory) -> None:
        """Test listing available providers."""
        providers = factory.list_providers()

        assert "ollama" in providers
        assert "oci_genai" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_provider_caching(self, factory) -> None:
        """Test that providers are cached."""
        provider_one = factory.get_provider("ollama")
        provider_two = factory.get_provider("ollama")

        assert provider_one is provider_two

    def test_cache_bypass(self, factory) -> None:
        """Test bypassing cache."""
        provider_one = factory.get_provider("ollama", cache=False)
        provider_two = factory.get_provider("ollama", cache=False)

        assert provider_one is not provider_two

    def test_singleton_instance(self) -> None:
        """Test singleton factory instance."""
        factory_one = LLMProviderFactory.get_instance()
        factory_two = LLMProviderFactory.get_instance()

        assert factory_one is factory_two

    def test_clear_cache(self, factory) -> None:
        """Test cache clearing."""
        provider_one = factory.get_provider("ollama")
        factory.clear_cache()
        provider_two = factory.get_provider("ollama")

        assert provider_one is not provider_two


class TestCreateLLMProvider:
    """Test convenience function."""

    def test_create_default(self) -> None:
        """Test creating default provider."""
        provider = create_llm_provider()

        assert isinstance(provider, OllamaProvider)

    def test_create_with_type(self) -> None:
        """Test creating specific type."""
        provider = create_llm_provider("ollama", model="mistral")

        assert provider.config.model == "mistral"
