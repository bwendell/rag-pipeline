"""Integration tests for embedding providers.

These tests use real models and are marked with @pytest.mark.slow and
@pytest.mark.integration to allow skipping in CI for fast feedback loops.

Run these tests specifically with:
    pytest tests/test_embedding_providers/test_integration.py -v

Skip these in CI with:
    pytest -m "not slow"
"""

from __future__ import annotations

import pytest

from rag_pipeline.embedding_providers import (
    SentenceTransformersProvider,
    create_embedding_provider,
)

from .helpers import (
    assert_embedding_valid,
    assert_embeddings_batch_valid,
    assert_embeddings_consistent_dimension,
    assert_embeddings_different,
    assert_embeddings_similar,
    assert_not_zero_vector,
    cosine_similarity,
)


# Use a small, fast model for integration tests
# all-MiniLM-L6-v2 is ~80MB and produces 384-dimensional embeddings
INTEGRATION_TEST_MODEL = "all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


@pytest.fixture(scope="module")
def real_provider() -> SentenceTransformersProvider:
    """Create a real SentenceTransformers provider for integration tests.

    Uses module scope to avoid loading the model multiple times.
    """
    return SentenceTransformersProvider(model_name=INTEGRATION_TEST_MODEL)


@pytest.mark.slow
@pytest.mark.integration
class TestRealEmbeddingGeneration:
    """Test real embedding generation with actual model."""

    @pytest.mark.anyio
    async def test_real_embedding_generation(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that real embeddings are generated correctly."""
        text = "This is a test sentence for embedding generation."

        embedding = await real_provider.embed(text)

        assert_embedding_valid(embedding, expected_dimension=EXPECTED_DIMENSION)
        assert_not_zero_vector(embedding)

    @pytest.mark.anyio
    async def test_real_batch_embedding(self, real_provider: SentenceTransformersProvider) -> None:
        """Test batch embedding generation with real model."""
        texts = [
            "First sentence for batch testing.",
            "Second sentence with different content.",
            "Third sentence about technology and AI.",
        ]

        embeddings = await real_provider.embed_batch(texts)

        assert_embeddings_batch_valid(
            embeddings,
            expected_count=len(texts),
            expected_dimension=EXPECTED_DIMENSION,
        )

        # Each embedding should be non-zero
        for embedding in embeddings:
            assert_not_zero_vector(embedding)

    @pytest.mark.anyio
    async def test_embedding_dimension_matches(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that embedding dimension matches provider's reported dimension."""
        text = "Testing dimension consistency."

        embedding = await real_provider.embed(text)
        reported_dimension = real_provider.get_embedding_dimension()

        assert len(embedding) == reported_dimension
        assert reported_dimension == EXPECTED_DIMENSION


@pytest.mark.slow
@pytest.mark.integration
class TestSemanticSimilarity:
    """Test semantic similarity properties of embeddings."""

    @pytest.mark.anyio
    async def test_similar_texts_have_similar_embeddings(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that semantically similar texts produce similar embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."

        embedding1 = await real_provider.embed(text1)
        embedding2 = await real_provider.embed(text2)

        # These texts are semantically similar, so embeddings should be similar
        # Using a threshold of 0.7 to account for phrasing differences
        assert_embeddings_similar(embedding1, embedding2, threshold=0.7)

    @pytest.mark.anyio
    async def test_different_texts_have_different_embeddings(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that semantically different texts produce different embeddings."""
        text1 = "The weather is sunny and warm today."
        text2 = "Machine learning algorithms process data efficiently."

        embedding1 = await real_provider.embed(text1)
        embedding2 = await real_provider.embed(text2)

        # These texts are semantically different, so embeddings should differ
        assert_embeddings_different(embedding1, embedding2, threshold=0.7)

    @pytest.mark.anyio
    async def test_identical_texts_have_identical_embeddings(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that identical texts produce identical embeddings."""
        text = "This is the exact same text."

        embedding1 = await real_provider.embed(text)
        embedding2 = await real_provider.embed(text)

        # Identical texts should produce identical embeddings
        similarity = cosine_similarity(embedding1, embedding2)
        assert similarity > 0.9999, (
            f"Identical texts should have identical embeddings, got {similarity}"
        )

    @pytest.mark.anyio
    async def test_question_answer_similarity(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that questions and their answers have reasonable similarity."""
        question = "What is the capital of France?"
        answer = "Paris is the capital of France."
        unrelated = "The Python programming language was created by Guido van Rossum."

        q_embedding = await real_provider.embed(question)
        a_embedding = await real_provider.embed(answer)
        u_embedding = await real_provider.embed(unrelated)

        # Question and answer should be more similar than question and unrelated
        qa_similarity = cosine_similarity(q_embedding, a_embedding)
        qu_similarity = cosine_similarity(q_embedding, u_embedding)

        assert qa_similarity > qu_similarity, (
            f"Q-A similarity ({qa_similarity:.4f}) should be greater than "
            f"Q-unrelated similarity ({qu_similarity:.4f})"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestBatchConsistency:
    """Test batch embedding consistency properties."""

    @pytest.mark.anyio
    async def test_batch_vs_single_embedding_consistency(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that batch and single embeddings produce same results."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
        ]

        # Get batch embeddings
        batch_embeddings = await real_provider.embed_batch(texts)

        # Get single embeddings
        single_embeddings = [await real_provider.embed(text) for text in texts]

        # They should be identical (within floating point tolerance)
        for i, (batch_emb, single_emb) in enumerate(zip(batch_embeddings, single_embeddings)):
            similarity = cosine_similarity(batch_emb, single_emb)
            assert similarity > 0.9999, (
                f"Batch and single embeddings for text {i} should be identical, "
                f"got similarity {similarity}"
            )

    @pytest.mark.anyio
    async def test_batch_dimension_consistency(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that all batch embeddings have consistent dimensions."""
        texts = [
            "Short text.",
            "This is a medium length text for testing.",
            "This is a much longer text that contains more words and should still produce embeddings of the same dimension as shorter texts.",
        ]

        embeddings = await real_provider.embed_batch(texts)

        dimension = assert_embeddings_consistent_dimension(embeddings)
        assert dimension == EXPECTED_DIMENSION


@pytest.mark.slow
@pytest.mark.integration
class TestFactoryIntegration:
    """Test factory function integration with real providers."""

    @pytest.mark.anyio
    async def test_factory_creates_working_provider(self) -> None:
        """Test that factory-created provider works correctly."""
        provider = create_embedding_provider(
            provider_type="sentence-transformers",
            model_name=INTEGRATION_TEST_MODEL,
        )

        text = "Testing factory-created provider."
        embedding = await provider.embed(text)

        assert_embedding_valid(embedding, expected_dimension=EXPECTED_DIMENSION)
        assert_not_zero_vector(embedding)

    @pytest.mark.anyio
    async def test_factory_with_default_model(self) -> None:
        """Test that factory with default model works."""
        provider = create_embedding_provider(provider_type="sentence-transformers")

        text = "Testing default model."
        embedding = await provider.embed(text)

        # Just verify it produces a valid embedding (dimension may vary by model)
        assert_embedding_valid(embedding)
        assert_not_zero_vector(embedding)


@pytest.mark.slow
@pytest.mark.integration
class TestEdgeCasesWithRealModel:
    """Test edge cases with real model."""

    @pytest.mark.anyio
    async def test_empty_string_returns_zero_vector(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that empty string returns zero vector."""
        embedding = await real_provider.embed("")

        # Empty strings should return zero vectors
        assert len(embedding) == EXPECTED_DIMENSION
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.anyio
    async def test_unicode_text(self, real_provider: SentenceTransformersProvider) -> None:
        """Test embedding of unicode text."""
        texts = [
            "Hello, world!",  # English
            "Bonjour le monde!",  # French
            "你好世界！",  # Chinese
            "こんにちは世界！",  # Japanese
            "مرحبا بالعالم!",  # Arabic
        ]

        embeddings = await real_provider.embed_batch(texts)

        assert_embeddings_batch_valid(
            embeddings,
            expected_count=len(texts),
            expected_dimension=EXPECTED_DIMENSION,
        )

        # Each should be non-zero
        for embedding in embeddings:
            assert_not_zero_vector(embedding)

    @pytest.mark.anyio
    async def test_very_long_text(self, real_provider: SentenceTransformersProvider) -> None:
        """Test embedding of very long text."""
        # Create a long text (models typically truncate)
        long_text = "This is a test sentence. " * 100

        embedding = await real_provider.embed(long_text)

        assert_embedding_valid(embedding, expected_dimension=EXPECTED_DIMENSION)
        assert_not_zero_vector(embedding)

    @pytest.mark.anyio
    async def test_special_characters(self, real_provider: SentenceTransformersProvider) -> None:
        """Test embedding of text with special characters."""
        text = "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?`~"

        embedding = await real_provider.embed(text)

        assert_embedding_valid(embedding, expected_dimension=EXPECTED_DIMENSION)
        assert_not_zero_vector(embedding)


@pytest.mark.slow
@pytest.mark.integration
class TestProviderInfo:
    """Test provider information methods."""

    @pytest.mark.anyio
    async def test_model_info_after_loading(
        self, real_provider: SentenceTransformersProvider
    ) -> None:
        """Test that model info is correct after loading."""
        # Trigger model loading
        await real_provider.embed("trigger loading")

        info = real_provider.get_model_info()

        assert info["provider"] == "sentence-transformers"
        assert info["name"] == INTEGRATION_TEST_MODEL
        assert info["dimension"] == EXPECTED_DIMENSION

    @pytest.mark.anyio
    async def test_health_check_passes(self, real_provider: SentenceTransformersProvider) -> None:
        """Test that health check passes for working provider."""
        is_healthy = await real_provider.health_check()

        assert is_healthy is True
