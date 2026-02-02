"""Unit tests for embedding provider factory."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider
from rag_pipeline.embedding_providers.factory import (
    create_embedding_provider,
    get_available_providers,
    get_default_model,
    get_provider_class,
    register_provider,
    PROVIDER_REGISTRY,
)


class TestCreateEmbeddingProvider:
    """Tests for create_embedding_provider function."""

    def test_create_provider_default(self):
        """Test creating provider with all defaults."""
        with patch("sentence_transformers.SentenceTransformer"):
            provider = create_embedding_provider()

            assert provider is not None
            assert provider.model_name == "all-MiniLM-L6-v2"

    def test_create_provider_with_model_name(self):
        """Test creating provider with custom model name."""
        with patch("sentence_transformers.SentenceTransformer"):
            provider = create_embedding_provider(
                provider_type="sentence-transformers", model_name="all-mpnet-base-v2"
            )

            assert provider.model_name == "all-mpnet-base-v2"

    def test_create_provider_with_kwargs(self):
        """Test kwargs are passed to provider constructor."""
        with patch("sentence_transformers.SentenceTransformer"):
            provider = create_embedding_provider(
                provider_type="sentence-transformers", normalize_embeddings=False
            )

            assert provider is not None
            assert isinstance(provider, AbstractEmbeddingProvider)
            assert provider.model_name == "all-MiniLM-L6-v2"

    def test_create_provider_openai(self):
        """Test creating OpenAI provider."""
        provider = create_embedding_provider(provider_type="openai")

        assert provider is not None
        assert provider.model_name == "text-embedding-ada-002"

    def test_create_provider_oci(self):
        """Test creating OCI provider."""
        provider = create_embedding_provider(provider_type="oci")

        assert provider is not None
        assert provider.model_name == "cohere.embed-english-v3.0"

    def test_create_provider_unknown_type_raises(self):
        """Test unknown provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embedding_provider(provider_type="unknown-provider")


class TestGetDefaultModel:
    """Tests for get_default_model function."""

    def test_get_default_model_sentence_transformers(self):
        """Test default model for sentence-transformers."""
        model = get_default_model("sentence-transformers")
        assert model == "all-MiniLM-L6-v2"

    def test_get_default_model_openai(self):
        """Test default model for openai."""
        model = get_default_model("openai")
        assert model == "text-embedding-ada-002"

    def test_get_default_model_oci(self):
        """Test default model for oci."""
        model = get_default_model("oci")
        assert model == "cohere.embed-english-v3.0"

    def test_get_default_model_unknown_raises(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            get_default_model("unknown")


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    def test_get_available_providers_returns_list(self):
        """Test returns list of providers."""
        providers = get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) >= 3  # At least sentence-transformers, openai, oci

    def test_get_available_providers_contains_builtin(self):
        """Test contains all built-in providers."""
        providers = get_available_providers()

        assert "sentence-transformers" in providers
        assert "openai" in providers
        assert "oci" in providers

    def test_get_available_providers_sorted(self):
        """Test providers are sorted alphabetically."""
        providers = get_available_providers()
        assert providers == sorted(providers)


class TestGetProviderClass:
    """Tests for get_provider_class function."""

    def test_get_provider_class_sentence_transformers(self):
        """Test getting sentence-transformers provider class."""
        provider_class = get_provider_class("sentence-transformers")

        assert provider_class is not None
        assert issubclass(provider_class, AbstractEmbeddingProvider)

    def test_get_provider_class_openai(self):
        """Test getting openai provider class."""
        provider_class = get_provider_class("openai")

        assert provider_class is not None
        assert issubclass(provider_class, AbstractEmbeddingProvider)

    def test_get_provider_class_oci(self):
        """Test getting oci provider class."""
        provider_class = get_provider_class("oci")

        assert provider_class is not None
        assert issubclass(provider_class, AbstractEmbeddingProvider)

    def test_get_provider_class_unknown_raises(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            get_provider_class("unknown-provider")


class TestRegisterProvider:
    """Tests for register_provider function."""

    def test_register_custom_provider(self):
        """Test registering a custom provider."""

        # Create a mock provider class
        class CustomProvider(AbstractEmbeddingProvider):
            def _get_provider_name(self) -> str:
                return "custom"

            async def embed(self, text):
                return [0.0] * 384

            async def embed_batch(self, texts, *, batch_size=None, show_progress=False):
                return [[0.0] * 384 for _ in texts]

        # Register it
        register_provider("custom-test", CustomProvider)

        # Verify it's available
        assert "custom-test" in get_available_providers()

        # Cleanup
        del PROVIDER_REGISTRY["custom-test"]

    def test_register_non_subclass_raises(self):
        """Test registering non-AbstractEmbeddingProvider raises TypeError."""

        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="must be a subclass of AbstractEmbeddingProvider"):
            register_provider("invalid", NotAProvider)  # type: ignore


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_registry_initializes_with_lazy_loaders(self):
        """Test that registry initializes with lazy loaders."""
        providers = get_available_providers()

        # Should have at least the built-in providers
        assert len(providers) >= 3
        assert "sentence-transformers" in providers
        assert "openai" in providers
        assert "oci" in providers

    def test_lazy_loader_converts_to_class(self):
        """Test that lazy loader is converted to class on first use."""
        # Get the provider class twice
        class1 = get_provider_class("openai")
        class2 = get_provider_class("openai")

        # Should be the same class (cached)
        assert class1 is class2
        assert isinstance(class1, type)


class TestProviderInstantiation:
    """Tests for proper provider instantiation."""

    def test_created_provider_is_instance_of_abstract(self):
        """Test that created provider is instance of AbstractEmbeddingProvider."""
        provider = create_embedding_provider("openai")
        assert isinstance(provider, AbstractEmbeddingProvider)

    def test_created_provider_has_model_name(self):
        """Test that created provider has model_name attribute."""
        provider = create_embedding_provider("openai", model_name="text-embedding-3-small")
        assert hasattr(provider, "model_name")
        assert provider.model_name == "text-embedding-3-small"

    def test_created_provider_has_get_embedding_dimension(self):
        """Test that created provider has get_embedding_dimension method."""
        provider = create_embedding_provider("openai")
        assert hasattr(provider, "get_embedding_dimension")
        assert callable(provider.get_embedding_dimension)
