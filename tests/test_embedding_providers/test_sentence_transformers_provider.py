"""Unit tests for SentenceTransformersProvider.

All tests use mocked SentenceTransformer to avoid loading actual models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_pipeline.core.exceptions import EmbeddingError
from rag_pipeline.embedding_providers import SentenceTransformersProvider
from rag_pipeline.embedding_providers.base import create_zero_vector


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for provider initialization."""

    def test_initialization_with_defaults(self):
        """Test provider initializes with default values."""
        provider = SentenceTransformersProvider()

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.device is None
        assert provider.normalize_embeddings is True
        assert provider._model is None  # Lazy loading - not loaded yet

    def test_initialization_with_custom_model(self):
        """Test provider accepts custom model name."""
        provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")
        assert provider.model_name == "all-mpnet-base-v2"

    def test_initialization_with_device(self):
        """Test provider accepts device parameter."""
        provider = SentenceTransformersProvider(device="cpu")
        assert provider.device == "cpu"

    def test_initialization_with_device_cuda(self):
        """Test provider accepts CUDA device."""
        provider = SentenceTransformersProvider(device="cuda")
        assert provider.device == "cuda"

    def test_initialization_with_device_mps(self):
        """Test provider accepts Apple Silicon MPS device."""
        provider = SentenceTransformersProvider(device="mps")
        assert provider.device == "mps"

    def test_initialization_with_normalize_false(self):
        """Test provider accepts normalize_embeddings=False."""
        provider = SentenceTransformersProvider(normalize_embeddings=False)
        assert provider.normalize_embeddings is False

    def test_initialization_with_normalize_true(self):
        """Test provider accepts normalize_embeddings=True."""
        provider = SentenceTransformersProvider(normalize_embeddings=True)
        assert provider.normalize_embeddings is True

    def test_initialization_with_empty_model_name_raises(self):
        """Test empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            SentenceTransformersProvider(model_name="")

    def test_initialization_with_all_custom_params(self):
        """Test initialization with all custom parameters."""
        provider = SentenceTransformersProvider(
            model_name="custom-model",
            device="cuda",
            normalize_embeddings=False,
        )

        assert provider.model_name == "custom-model"
        assert provider.device == "cuda"
        assert provider.normalize_embeddings is False
        assert provider._model is None


# =============================================================================
# Lazy Loading Tests
# =============================================================================


class TestLazyLoading:
    """Tests for lazy model loading behavior."""

    def test_model_not_loaded_until_first_embed(self):
        """Test that model is not loaded on initialization."""
        provider = SentenceTransformersProvider()
        assert provider._model is None

    @pytest.mark.anyio
    async def test_model_loaded_on_first_embed(self, mock_sentence_transformer):
        """Test model is loaded when embed() is called."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            assert provider._model is None

            await provider.embed("test text")

            assert provider._model is not None
            mock_cls.assert_called_once()

    @pytest.mark.anyio
    async def test_model_loaded_only_once(self, mock_sentence_transformer):
        """Test model is loaded only once even with multiple calls."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()

            await provider.embed("first")
            await provider.embed("second")
            await provider.embed("third")

            mock_cls.assert_called_once()

    @pytest.mark.anyio
    async def test_dimension_set_after_model_load(self, mock_sentence_transformer):
        """Test embedding dimension is set after model loads."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.get_sentence_embedding_dimension.return_value = 384

            provider = SentenceTransformersProvider()
            await provider.embed("test")

            assert provider.get_embedding_dimension() == 384

    @pytest.mark.anyio
    async def test_dimension_persists_across_calls(self, mock_sentence_transformer):
        """Test dimension is correctly returned after loading."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.get_sentence_embedding_dimension.return_value = 384

            provider = SentenceTransformersProvider()
            dim1 = await provider.embed("text1")
            dim2 = await provider.embed("text2")

            assert len(dim1) == 384
            assert len(dim2) == 384


# =============================================================================
# Single Embedding Tests
# =============================================================================


class TestEmbed:
    """Tests for single text embedding."""

    @pytest.mark.anyio
    async def test_embed_single_text(self, mock_sentence_transformers_provider):
        """Test embedding a single text returns vector."""
        embedding = await mock_sentence_transformers_provider.embed("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, (float, np.floating)) for x in embedding)

    @pytest.mark.anyio
    async def test_embed_empty_string_returns_zero_vector(
        self, mock_sentence_transformers_provider
    ):
        """Test empty string returns zero vector."""
        embedding = await mock_sentence_transformers_provider.embed("")

        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.anyio
    async def test_embed_whitespace_only_treated_as_normal_text(
        self, mock_sentence_transformers_provider
    ):
        """Test whitespace-only string is treated as normal text (not zero vector)."""
        embedding = await mock_sentence_transformers_provider.embed("   ")

        assert len(embedding) == 384
        # Whitespace-only strings are NOT treated as empty, so should generate embedding
        assert embedding != [0.0] * 384

    @pytest.mark.anyio
    async def test_embed_calls_model_encode(self, mock_sentence_transformer):
        """Test embed calls model.encode with correct parameters."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed("test text")

            # Verify encode was called
            mock_sentence_transformer.encode.assert_called()

    @pytest.mark.anyio
    async def test_embed_passes_normalize_embeddings_param(self, mock_sentence_transformer):
        """Test that normalize_embeddings parameter is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(normalize_embeddings=True)
            await provider.embed("test text")

            # Check kwargs passed to encode
            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("normalize_embeddings") is True

    @pytest.mark.anyio
    async def test_embed_long_text(self, mock_sentence_transformers_provider, long_text):
        """Test embedding a long text."""
        embedding = await mock_sentence_transformers_provider.embed(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, (float, np.floating)) for x in embedding)

    @pytest.mark.anyio
    async def test_embed_short_text(self, mock_sentence_transformers_provider, short_text):
        """Test embedding a short text."""
        embedding = await mock_sentence_transformers_provider.embed(short_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, (float, np.floating)) for x in embedding)

    @pytest.mark.anyio
    async def test_embed_special_characters(self, mock_sentence_transformers_provider):
        """Test embedding text with special characters."""
        text = "Hello! @#$%^&*() [bracket] {brace} 你好 🎉"
        embedding = await mock_sentence_transformers_provider.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    @pytest.mark.anyio
    async def test_embed_multiline_text(self, mock_sentence_transformers_provider):
        """Test embedding multiline text."""
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        embedding = await mock_sentence_transformers_provider.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384


# =============================================================================
# Batch Embedding Tests
# =============================================================================


class TestEmbedBatch:
    """Tests for batch text embedding."""

    @pytest.mark.anyio
    async def test_embed_batch(self, mock_sentence_transformers_provider, sample_texts):
        """Test batch embedding multiple texts."""
        embeddings = await mock_sentence_transformers_provider.embed_batch(sample_texts)

        assert len(embeddings) == len(sample_texts)
        for embedding in embeddings:
            assert len(embedding) == 384
            assert all(isinstance(x, (float, np.floating)) for x in embedding)

    @pytest.mark.anyio
    async def test_embed_batch_with_empty_strings(
        self, mock_sentence_transformers_provider, texts_with_empty_strings
    ):
        """Test batch embedding with empty strings returns zero vectors at correct positions."""
        embeddings = await mock_sentence_transformers_provider.embed_batch(texts_with_empty_strings)

        assert len(embeddings) == 5

        # Positions 1 and 3 should be zero vectors (empty strings)
        assert all(x == 0.0 for x in embeddings[1])
        assert all(x == 0.0 for x in embeddings[3])

        # Other positions should NOT be zero vectors
        assert any(x != 0.0 for x in embeddings[0])
        assert any(x != 0.0 for x in embeddings[2])
        assert any(x != 0.0 for x in embeddings[4])

    @pytest.mark.anyio
    async def test_embed_batch_all_empty_strings(self, mock_sentence_transformers_provider):
        """Test batch with all empty strings returns all zero vectors."""
        texts = ["", "", ""]
        embeddings = await mock_sentence_transformers_provider.embed_batch(texts)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert all(x == 0.0 for x in embedding)

    @pytest.mark.anyio
    async def test_embed_batch_empty_list_raises_valueerror(
        self, mock_sentence_transformers_provider
    ):
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            await mock_sentence_transformers_provider.embed_batch([])

    @pytest.mark.anyio
    async def test_embed_batch_custom_batch_size(self, mock_sentence_transformer, sample_texts):
        """Test custom batch_size is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed_batch(sample_texts, batch_size=16)

            # Verify batch_size was passed
            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("batch_size") == 16

    @pytest.mark.anyio
    async def test_embed_batch_default_batch_size(self, mock_sentence_transformer, sample_texts):
        """Test default batch_size is used when not specified."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed_batch(sample_texts)

            # Verify default batch_size (32) was used
            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("batch_size") == 32

    @pytest.mark.anyio
    async def test_embed_batch_show_progress(self, mock_sentence_transformer, sample_texts):
        """Test show_progress parameter is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed_batch(sample_texts, show_progress=True)

            # Verify show_progress was passed
            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("show_progress_bar") is True

    @pytest.mark.anyio
    async def test_embed_batch_single_text(self, mock_sentence_transformers_provider):
        """Test batch embedding with single text."""
        embeddings = await mock_sentence_transformers_provider.embed_batch(["Single text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    @pytest.mark.anyio
    async def test_embed_batch_large_batch(self, mock_sentence_transformers_provider):
        """Test batch embedding with large number of texts."""
        texts = [f"Text {i}" for i in range(100)]
        embeddings = await mock_sentence_transformers_provider.embed_batch(texts)

        assert len(embeddings) == 100
        for embedding in embeddings:
            assert len(embedding) == 384

    @pytest.mark.anyio
    async def test_embed_batch_preserves_order(self, mock_sentence_transformers_provider):
        """Test that embeddings are returned in same order as input texts."""
        texts = ["First", "Second", "Third", "Fourth", "Fifth"]
        embeddings = await mock_sentence_transformers_provider.embed_batch(texts)

        assert len(embeddings) == len(texts)
        # Verify all embeddings are different (stochastic generation)
        for i in range(len(embeddings) - 1):
            assert embeddings[i] != embeddings[i + 1]


# =============================================================================
# Provider Info Tests
# =============================================================================


class TestProviderInfo:
    """Tests for provider information methods."""

    def test_get_provider_name(self):
        """Test provider name is correct."""
        provider = SentenceTransformersProvider()
        assert provider._get_provider_name() == "sentence-transformers"

    @pytest.mark.anyio
    async def test_get_model_info(self, mock_sentence_transformers_provider):
        """Test get_model_info returns correct structure."""
        info = mock_sentence_transformers_provider.get_model_info()

        assert "name" in info
        assert "provider" in info
        assert "dimension" in info
        assert info["name"] == "all-MiniLM-L6-v2"
        assert info["provider"] == "sentence-transformers"
        assert info["dimension"] == 384

    def test_get_embedding_dimension_before_load(self):
        """Test get_embedding_dimension returns default dimension."""
        provider = SentenceTransformersProvider()
        # Model not loaded, but dimension is set to default 384 in __init__
        # So this should work since DEFAULT_DIMENSION is set
        dim = provider.get_embedding_dimension()
        assert dim == 384

    @pytest.mark.anyio
    async def test_get_embedding_dimension_after_load(self, mock_sentence_transformers_provider):
        """Test get_embedding_dimension returns correct dimension after load."""
        dim = mock_sentence_transformers_provider.get_embedding_dimension()
        assert dim == 384

    def test_get_model_info_with_custom_model(self):
        """Test get_model_info with custom model name."""
        provider = SentenceTransformersProvider(model_name="custom-model-v1")
        info = provider.get_model_info()

        assert info["name"] == "custom-model-v1"
        assert info["provider"] == "sentence-transformers"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.anyio
    async def test_health_check_success(self, mock_sentence_transformers_provider):
        """Test health check returns True when healthy."""
        is_healthy = await mock_sentence_transformers_provider.health_check()
        assert is_healthy is True

    @pytest.mark.anyio
    async def test_health_check_failure_on_error(self, mock_sentence_transformer):
        """Test health check returns False when embed fails."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.side_effect = RuntimeError("Model failed")

            provider = SentenceTransformersProvider()
            is_healthy = await provider.health_check()

            assert is_healthy is False

    @pytest.mark.anyio
    async def test_health_check_caches_successful_result(self, mock_sentence_transformers_provider):
        """Test that multiple health checks don't reload model."""
        initial_model = mock_sentence_transformers_provider._model

        await mock_sentence_transformers_provider.health_check()
        await mock_sentence_transformers_provider.health_check()
        await mock_sentence_transformers_provider.health_check()

        # Model should be same instance
        assert mock_sentence_transformers_provider._model is initial_model


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.anyio
    async def test_embed_error_wrapped_in_embedding_error(self):
        """Test that embed errors are wrapped in EmbeddingError."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.side_effect = RuntimeError("Encode failed")
            mock_cls.return_value = mock_model

            provider = SentenceTransformersProvider()

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed("test")

            assert "Failed to generate embedding" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_model_load_error_wrapped_in_embedding_error(self):
        """Test that model loading errors are wrapped in EmbeddingError."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.side_effect = ImportError("Cannot load model")

            provider = SentenceTransformersProvider()

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed("test")

            assert "Failed to load sentence-transformers model" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_model_load_error_details(self):
        """Test that model loading error includes details."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.side_effect = ImportError("Cannot import")

            provider = SentenceTransformersProvider(model_name="test-model")

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed("test")

            error = exc_info.value
            assert error.details["model"] == "test-model"
            assert "Cannot import" in error.details["error"]

    @pytest.mark.anyio
    async def test_batch_error_wrapped_in_embedding_error(self, mock_sentence_transformer):
        """Test that batch embed errors are wrapped in EmbeddingError."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.side_effect = RuntimeError("Batch failed")

            provider = SentenceTransformersProvider()

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed_batch(["text1", "text2"])

            assert "Failed to generate batch embeddings" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_batch_error_includes_details(self, mock_sentence_transformer):
        """Test that batch error includes relevant details."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.side_effect = RuntimeError("Batch failed")

            provider = SentenceTransformersProvider()

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed_batch(["text1", "text2"], batch_size=16)

            error = exc_info.value
            assert error.details["text_count"] == 2
            assert error.details["batch_size"] == 16

    @pytest.mark.anyio
    async def test_embed_error_includes_text_length(self, mock_sentence_transformer):
        """Test that embed error includes text length information."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.side_effect = RuntimeError("Failed")

            provider = SentenceTransformersProvider()

            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed("This is a test text")

            error = exc_info.value
            assert error.details["text_length"] == 19


# =============================================================================
# Embed Query Tests
# =============================================================================


class TestEmbedQuery:
    """Tests for embed_query method."""

    @pytest.mark.anyio
    async def test_embed_query_delegates_to_embed(self, mock_sentence_transformers_provider):
        """Test embed_query calls embed() internally."""
        embedding = await mock_sentence_transformers_provider.embed_query("search query")

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    @pytest.mark.anyio
    async def test_embed_query_empty_string(self, mock_sentence_transformers_provider):
        """Test embed_query with empty string returns zero vector."""
        embedding = await mock_sentence_transformers_provider.embed_query("")

        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.anyio
    async def test_embed_query_long_query(self, mock_sentence_transformers_provider):
        """Test embed_query with long query string."""
        long_query = "How do I configure the system for optimal performance? " * 10
        embedding = await mock_sentence_transformers_provider.embed_query(long_query)

        assert isinstance(embedding, list)
        assert len(embedding) == 384


# =============================================================================
# Device Configuration Tests
# =============================================================================


class TestDeviceConfiguration:
    """Tests for device configuration handling."""

    @pytest.mark.anyio
    async def test_device_passed_to_model_initialization(self, mock_sentence_transformer):
        """Test device parameter is passed to SentenceTransformer."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(device="cuda")
            await provider.embed("test")

            # Verify device was passed to constructor
            call_args = mock_cls.call_args
            assert call_args[1].get("device") == "cuda"

    @pytest.mark.anyio
    async def test_none_device_passed_to_model_initialization(self, mock_sentence_transformer):
        """Test None device is passed to SentenceTransformer for auto-detection."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(device=None)
            await provider.embed("test")

            # Verify device=None was passed
            call_args = mock_cls.call_args
            assert call_args[1].get("device") is None


# =============================================================================
# Normalize Embeddings Tests
# =============================================================================


class TestNormalizeEmbeddings:
    """Tests for normalize_embeddings configuration."""

    @pytest.mark.anyio
    async def test_normalize_true_passed_to_encode(self, mock_sentence_transformer):
        """Test normalize_embeddings=True is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(normalize_embeddings=True)
            await provider.embed("test")

            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("normalize_embeddings") is True

    @pytest.mark.anyio
    async def test_normalize_false_passed_to_encode(self, mock_sentence_transformer):
        """Test normalize_embeddings=False is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(normalize_embeddings=False)
            await provider.embed("test")

            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("normalize_embeddings") is False

    @pytest.mark.anyio
    async def test_normalize_batch_false_passed_to_encode(
        self, mock_sentence_transformer, sample_texts
    ):
        """Test normalize_embeddings=False is passed to batch encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider(normalize_embeddings=False)
            await provider.embed_batch(sample_texts)

            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("normalize_embeddings") is False


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    @pytest.mark.anyio
    async def test_embed_unicode_text(self, mock_sentence_transformers_provider):
        """Test embedding unicode text."""
        texts = [
            "Hello 世界",
            "مرحبا بالعالم",
            "Привет мир",
            "مرحبا العالم",
        ]

        for text in texts:
            embedding = await mock_sentence_transformers_provider.embed(text)
            assert len(embedding) == 384

    @pytest.mark.anyio
    async def test_embed_repeated_text_gives_same_embedding(self, mock_sentence_transformer):
        """Test that same text gives consistent embeddings across calls."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            # Make encode return fixed value for same input
            def fixed_encode(text, **kwargs):
                if isinstance(text, str):
                    return np.array([0.1] * 384, dtype=np.float32)
                else:
                    return np.array([[0.1] * 384] * len(text), dtype=np.float32)

            mock_sentence_transformer.encode.side_effect = fixed_encode
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()

            # Embed same text twice
            embed1 = await provider.embed("same text")
            embed2 = await provider.embed("same text")

            # Should get same embedding
            assert embed1 == embed2

    @pytest.mark.anyio
    async def test_batch_with_single_non_empty_and_many_empty(
        self, mock_sentence_transformers_provider
    ):
        """Test batch with one non-empty text among many empty ones."""
        texts = ["", "", "Only this text", "", ""]
        embeddings = await mock_sentence_transformers_provider.embed_batch(texts)

        assert len(embeddings) == 5
        # All except index 2 should be zero
        for i in [0, 1, 3, 4]:
            assert all(x == 0.0 for x in embeddings[i])

        # Index 2 should not be all zeros
        assert any(x != 0.0 for x in embeddings[2])

    @pytest.mark.anyio
    async def test_very_large_batch_with_mixed_empty_and_non_empty(
        self, mock_sentence_transformers_provider
    ):
        """Test large batch with mixture of empty and non-empty texts."""
        texts = []
        for i in range(100):
            if i % 3 == 0:
                texts.append("")
            else:
                texts.append(f"Text {i}")

        embeddings = await mock_sentence_transformers_provider.embed_batch(texts)

        assert len(embeddings) == 100

        # Check zero vectors are at correct positions
        for i in range(100):
            if i % 3 == 0:
                assert all(x == 0.0 for x in embeddings[i])
            else:
                assert any(x != 0.0 for x in embeddings[i])


# =============================================================================
# Convert to Numpy Tests
# =============================================================================


class TestConvertToNumpy:
    """Tests for numpy conversion in encode calls."""

    @pytest.mark.anyio
    async def test_convert_to_numpy_passed_to_encode(self, mock_sentence_transformer):
        """Test that convert_to_numpy=True is passed to encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed("test")

            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("convert_to_numpy") is True

    @pytest.mark.anyio
    async def test_convert_to_numpy_batch_passed_to_encode(
        self, mock_sentence_transformer, sample_texts
    ):
        """Test that convert_to_numpy=True is passed to batch encode."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_sentence_transformer

            provider = SentenceTransformersProvider()
            await provider.embed_batch(sample_texts)

            call_kwargs = mock_sentence_transformer.encode.call_args[1]
            assert call_kwargs.get("convert_to_numpy") is True
