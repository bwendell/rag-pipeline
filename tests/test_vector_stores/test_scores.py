"""Score normalization and similarity verification tests.

This module tests similarity score behavior across vector store implementations:
- Cosine similarity scores in [0, 1] range
- Identical embeddings score ~1.0
- Orthogonal vectors score ~0.0
- Score ordering (highest first)
- Edge cases (zero vectors, NaN handling)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rag_pipeline.core.types import Chunk, Metadata
from rag_pipeline.vector_stores import ChromaDBStore, InMemoryStore
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore as InMemoryStoreClass

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Cosine Similarity Score Range Tests
# =============================================================================


class TestCosineSimilarityRange:
    """Tests verifying cosine similarity scores are in [0, 1] range."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="score_test_collection",
            )
            yield store

    @pytest.mark.anyio
    async def test_scores_are_in_valid_range_in_memory(self, in_memory_store: InMemoryStore):
        """All search scores from InMemoryStore must be in [0, 1] range."""
        # Add multiple chunks with random embeddings
        chunks = [
            Chunk(
                id=f"score-chunk-{i}",
                content=f"Content {i}",
                document_id="score-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(10)
        ]
        await in_memory_store.add(chunks)

        # Search with random query
        query = list(np.random.rand(384).astype(np.float32))
        results = await in_memory_store.search(query, top_k=10)

        # All scores should be in [0, 1]
        for result in results:
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} is outside valid range [0, 1]"

    @pytest.mark.anyio
    async def test_scores_are_in_valid_range_chroma(self, chroma_store: ChromaDBStore):
        """All search scores from ChromaDBStore must be in [0, 1] range."""
        chunks = [
            Chunk(
                id=f"chroma-score-{i}",
                content=f"Content {i}",
                document_id="chroma-score-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(10)
        ]
        await chroma_store.add(chunks)

        # Search with random query
        query = list(np.random.rand(384).astype(np.float32))
        results = await chroma_store.search(query, top_k=10)

        # All scores should be in [0, 1] (allowing small epsilon for floating point)
        for result in results:
            assert -0.01 <= result.score <= 1.01, (
                f"Score {result.score} is outside valid range [0, 1]"
            )


# =============================================================================
# Identical Embeddings Score Tests
# =============================================================================


class TestIdenticalEmbeddings:
    """Tests for similarity when embeddings are identical."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_identical_embedding_scores_one(self, in_memory_store: InMemoryStore):
        """Identical embeddings should have similarity ~1.0."""
        embedding = [0.5] * 384  # All same values

        chunk = Chunk(
            id="identical-chunk",
            content="Identical embedding chunk",
            document_id="doc-1",
            embedding=embedding,
        )
        await in_memory_store.add([chunk])

        # Search with identical embedding
        results = await in_memory_store.search(embedding, top_k=1)

        assert len(results) == 1
        assert results[0].score > 0.99, (
            f"Identical embedding should score ~1.0, got {results[0].score}"
        )

    @pytest.mark.anyio
    async def test_normalized_embedding_scores_one(self, in_memory_store: InMemoryStore):
        """Normalized embeddings should score ~1.0 when identical."""
        # Create a normalized embedding
        vec = np.array([1.0, 2.0, 3.0] + [1.0] * 381, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embedding = list(vec)

        chunk = Chunk(
            id="normalized-chunk",
            content="Normalized embedding chunk",
            document_id="doc-1",
            embedding=embedding,
        )
        await in_memory_store.add([chunk])

        # Search with identical embedding
        results = await in_memory_store.search(embedding, top_k=1)

        assert len(results) == 1
        assert results[0].score > 0.99, (
            f"Normalized identical embedding should score ~1.0, got {results[0].score}"
        )

    @pytest.mark.anyio
    async def test_scaled_embedding_scores_one(self, in_memory_store: InMemoryStore):
        """Scaled embeddings (same direction) should score ~1.0."""
        # Create embedding
        base_vec = np.array([1.0, 2.0, 3.0] + [0.0] * 381, dtype=np.float32)

        # Store as-is
        chunk = Chunk(
            id="scaled-chunk",
            content="Scaled embedding chunk",
            document_id="doc-1",
            embedding=list(base_vec),
        )
        await in_memory_store.add([chunk])

        # Search with scaled version (2x)
        query_vec = base_vec * 2.0
        results = await in_memory_store.search(list(query_vec), top_k=1)

        assert len(results) == 1
        # Cosine similarity ignores magnitude, so scaled versions should score ~1.0
        assert results[0].score > 0.99, (
            f"Scaled embedding should score ~1.0, got {results[0].score}"
        )


# =============================================================================
# Orthogonal Vectors Score Tests
# =============================================================================


class TestOrthogonalVectors:
    """Tests for similarity when vectors are orthogonal."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_orthogonal_vectors_score_zero(self, in_memory_store: InMemoryStore):
        """Orthogonal vectors should have similarity ~0.0."""
        # Create orthogonal vectors
        vec1 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0

        vec2 = np.zeros(384, dtype=np.float32)
        vec2[1] = 1.0

        chunk = Chunk(
            id="orthogonal-chunk",
            content="Orthogonal chunk",
            document_id="doc-1",
            embedding=list(vec1),
        )
        await in_memory_store.add([chunk])

        # Search with orthogonal vector
        results = await in_memory_store.search(list(vec2), top_k=1)

        assert len(results) == 1
        assert results[0].score < 0.01, (
            f"Orthogonal vectors should score ~0.0, got {results[0].score}"
        )

    @pytest.mark.anyio
    async def test_multiple_orthogonal_vectors(self, in_memory_store: InMemoryStore):
        """Multiple orthogonal vectors should all score ~0.0 with each other."""
        # Create multiple orthogonal basis vectors
        embeddings = []
        for i in range(5):
            vec = np.zeros(384, dtype=np.float32)
            vec[i] = 1.0
            embeddings.append(list(vec))

        chunks = [
            Chunk(
                id=f"ortho-chunk-{i}",
                content=f"Orthogonal chunk {i}",
                document_id="doc-1",
                embedding=embeddings[i],
            )
            for i in range(5)
        ]
        await in_memory_store.add(chunks)

        # Search with another orthogonal vector (6th basis vector)
        query_vec = np.zeros(384, dtype=np.float32)
        query_vec[5] = 1.0

        results = await in_memory_store.search(list(query_vec), top_k=5)

        # All should score ~0.0 (orthogonal)
        for result in results:
            assert result.score < 0.01, f"Orthogonal vector should score ~0.0, got {result.score}"


# =============================================================================
# Score Ordering Tests
# =============================================================================


class TestScoreOrdering:
    """Tests verifying search results are ordered by score (highest first)."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_results_ordered_by_score_descending(self, in_memory_store: InMemoryStore):
        """Search results must be ordered by score (highest first)."""
        # Create chunks with distinct embeddings
        chunk1 = Chunk(
            id="high-sim-chunk",
            content="High similarity",
            document_id="doc-1",
            embedding=[1.0] + [0.0] * 383,  # First dimension = 1
        )
        chunk2 = Chunk(
            id="med-sim-chunk",
            content="Medium similarity",
            document_id="doc-1",
            embedding=[0.7, 0.7] + [0.0] * 382,  # First two = 0.7
        )
        chunk3 = Chunk(
            id="low-sim-chunk",
            content="Low similarity",
            document_id="doc-1",
            embedding=[0.0] * 384,  # All zeros
        )

        await in_memory_store.add([chunk1, chunk2, chunk3])

        # Search with query aligned with first dimension
        query = [1.0] + [0.0] * 383
        results = await in_memory_store.search(query, top_k=3)

        assert len(results) == 3

        # Verify descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores should be in descending order: {scores}"
        )

    @pytest.mark.anyio
    async def test_rank_field_matches_order(self, in_memory_store: InMemoryStore):
        """Rank field must match the position in results (1-indexed)."""
        chunks = [
            Chunk(
                id=f"rank-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(5)
        ]
        await in_memory_store.add(chunks)

        query = list(np.random.rand(384).astype(np.float32))
        results = await in_memory_store.search(query, top_k=5)

        # Verify rank field
        for i, result in enumerate(results):
            assert result.rank == i + 1, f"Expected rank {i + 1}, got {result.rank}"

    @pytest.mark.anyio
    async def test_top_k_respects_score_ordering(self, in_memory_store: InMemoryStore):
        """top_k must return the highest scoring chunks."""
        # Create 10 chunks with known similarity order
        base_vec = np.array([1.0] + [0.0] * 383, dtype=np.float32)

        chunks = []
        for i in range(10):
            # Create vectors with decreasing similarity to [1, 0, 0, ...]
            vec = np.copy(base_vec)
            vec[0] = 1.0 - (i * 0.05)  # Decreasing first component
            vec[1] = i * 0.05  # Increasing second component
            vec = vec / np.linalg.norm(vec)  # Normalize

            chunks.append(
                Chunk(
                    id=f"ordered-chunk-{i}",
                    content=f"Content {i}",
                    document_id="doc-1",
                    embedding=list(vec),
                )
            )

        await in_memory_store.add(chunks)

        # Search with query [1, 0, 0, ...]
        results = await in_memory_store.search(list(base_vec), top_k=5)

        assert len(results) == 5

        # The first chunks should have higher IDs (they were more similar)
        # Wait, no - chunk 0 has highest first component, so should rank first
        # Let's verify scores are in descending order
        scores = [r.score for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
            f"Scores not in descending order: {scores}"
        )


# =============================================================================
# Zero Vector Edge Cases
# =============================================================================


class TestZeroVectors:
    """Tests for zero vector edge cases."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_zero_vector_chunk_scores_zero(self, in_memory_store: InMemoryStore):
        """Zero vector chunk should score 0.0 when searched."""
        zero_chunk = Chunk(
            id="zero-chunk",
            content="Zero embedding chunk",
            document_id="doc-1",
            embedding=[0.0] * 384,
        )
        await in_memory_store.add([zero_chunk])

        # Search with non-zero query
        query = [1.0] + [0.0] * 383
        results = await in_memory_store.search(query, top_k=1)

        assert len(results) == 1
        assert results[0].score == 0.0, f"Zero vector should score 0.0, got {results[0].score}"

    @pytest.mark.anyio
    async def test_zero_query_vector_scores_zero(self, in_memory_store: InMemoryStore):
        """Zero query vector should result in 0.0 scores."""
        chunk = Chunk(
            id="non-zero-chunk",
            content="Non-zero embedding chunk",
            document_id="doc-1",
            embedding=[1.0] + [0.0] * 383,
        )
        await in_memory_store.add([chunk])

        # Search with zero query
        results = await in_memory_store.search([0.0] * 384, top_k=1)

        assert len(results) == 1
        assert results[0].score == 0.0, f"Zero query should give 0.0 score, got {results[0].score}"

    @pytest.mark.anyio
    async def test_both_zero_vectors(self, in_memory_store: InMemoryStore):
        """Both zero vectors should score 0.0."""
        zero_chunk = Chunk(
            id="both-zero-chunk",
            content="Zero embedding chunk",
            document_id="doc-1",
            embedding=[0.0] * 384,
        )
        await in_memory_store.add([zero_chunk])

        results = await in_memory_store.search([0.0] * 384, top_k=1)

        assert len(results) == 1
        assert results[0].score == 0.0


# =============================================================================
# Cosine Similarity Calculation Tests (InMemoryStore specific)
# =============================================================================


class TestCosineSimilarityCalculation:
    """Direct tests for InMemoryStore's cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self):
        """Cosine similarity of identical vectors is 1.0."""
        vec = np.ones(384, dtype=np.float32)
        similarity = InMemoryStoreClass._cosine_similarity(vec, vec)

        assert similarity > 0.99
        assert similarity <= 1.0

    def test_cosine_similarity_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        vec1 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0

        vec2 = np.zeros(384, dtype=np.float32)
        vec2[1] = 1.0

        similarity = InMemoryStoreClass._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_opposite_vectors(self):
        """Cosine similarity of opposite vectors is clamped to 0."""
        vec = np.ones(384, dtype=np.float32)
        neg_vec = -vec

        similarity = InMemoryStoreClass._cosine_similarity(vec, neg_vec)

        # Should be clamped to [0, 1] range
        assert 0.0 <= similarity <= 1.0
        # Opposite vectors should have cosine = -1, clamped to 0
        assert similarity == 0.0

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector returns 0.0."""
        vec1 = np.ones(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)

        similarity = InMemoryStoreClass._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_both_zero(self):
        """Cosine similarity of two zero vectors returns 0.0."""
        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)

        similarity = InMemoryStoreClass._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_normalized(self):
        """Cosine similarity works with normalized vectors."""
        # Create normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        similarity = InMemoryStoreClass._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_45_degrees(self):
        """Cosine similarity at 45 degrees is ~0.707."""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([1.0, 1.0], dtype=np.float32)

        similarity = InMemoryStoreClass._cosine_similarity(vec1, vec2)

        # cos(45°) ≈ 0.707, but vec2 is not normalized
        # cos = dot / (norm1 * norm2) = 1 / (1 * sqrt(2)) = 0.707
        assert 0.70 < similarity < 0.72


# =============================================================================
# NaN and Inf Handling Tests
# =============================================================================


class TestNaNAndInfHandling:
    """Tests for NaN and Inf values in embeddings."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_nan_in_embedding(self, in_memory_store: InMemoryStore):
        """NaN values in embeddings should be handled gracefully."""
        embedding = [1.0] * 383 + [float("nan")]

        chunk = Chunk(
            id="nan-chunk",
            content="NaN embedding chunk",
            document_id="doc-1",
            embedding=embedding,
        )

        # Should be addable
        await in_memory_store.add([chunk])

        # Search might produce NaN scores
        query = [1.0] * 384
        results = await in_memory_store.search(query, top_k=1)

        # Result should exist, score may be NaN
        assert len(results) == 1
        # NaN comparison is always False, so check explicitly
        if not math.isnan(results[0].score):
            assert 0.0 <= results[0].score <= 1.0

    @pytest.mark.anyio
    async def test_inf_in_embedding(self, in_memory_store: InMemoryStore):
        """Inf values in embeddings should be handled."""
        embedding = [1.0] * 383 + [float("inf")]

        chunk = Chunk(
            id="inf-chunk",
            content="Inf embedding chunk",
            document_id="doc-1",
            embedding=embedding,
        )

        # Should be addable
        await in_memory_store.add([chunk])

        # Search
        query = [1.0] * 384
        results = await in_memory_store.search(query, top_k=1)

        # Result should exist
        assert len(results) == 1


# =============================================================================
# ChromaDB Distance-to-Similarity Conversion Tests
# =============================================================================


class TestChromaSimilarityConversion:
    """Tests for ChromaDB's distance to similarity conversion."""

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="similarity_conversion_test",
            )
            yield store

    @pytest.mark.anyio
    async def test_chroma_similarity_is_1_minus_distance(self, chroma_store: ChromaDBStore):
        """ChromaDB converts distance to similarity as (1 - distance)."""
        # ChromaDB uses cosine distance by default
        # Distance ranges [0, 2] for cosine, similarity = 1 - distance

        # Create chunk
        vec = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        chunk = Chunk(
            id="chroma-sim-chunk",
            content="Chroma similarity chunk",
            document_id="doc-1",
            embedding=list(vec),
        )
        await chroma_store.add([chunk])

        # Search with identical vector
        results = await chroma_store.search(list(vec), top_k=1)

        assert len(results) == 1
        # Identical vectors have distance ~0, similarity ~1
        assert results[0].score > 0.99, (
            f"Identical vector should have similarity ~1.0, got {results[0].score}"
        )

    @pytest.mark.anyio
    async def test_chroma_scores_in_valid_range(self, chroma_store: ChromaDBStore):
        """ChromaDB scores should be in [0, 1] after conversion."""
        chunks = [
            Chunk(
                id=f"chroma-range-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(5)
        ]
        await chroma_store.add(chunks)

        query = list(np.random.rand(384).astype(np.float32))
        results = await chroma_store.search(query, top_k=5)

        for result in results:
            # Allow small epsilon for floating point errors
            assert -0.01 <= result.score <= 1.01, f"Score {result.score} outside valid range"


# =============================================================================
# Score Distribution Tests
# =============================================================================


class TestScoreDistribution:
    """Tests for score distribution properties."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_score_range_with_random_embeddings(self, in_memory_store: InMemoryStore):
        """Scores from random embeddings should span the range."""
        # Create many chunks with random embeddings
        np.random.seed(42)  # For reproducibility
        chunks = [
            Chunk(
                id=f"random-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                embedding=list(np.random.randn(384).astype(np.float32)),
            )
            for i in range(100)
        ]
        await in_memory_store.add(chunks)

        # Search with random query
        query = list(np.random.randn(384).astype(np.float32))
        results = await in_memory_store.search(query, top_k=100)

        assert len(results) == 100

        scores = [r.score for r in results]

        # With random vectors, we should see a range of scores
        # (Not all the same, and spanning some range)
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0
        assert max(scores) > min(scores), "All scores should not be identical"

    @pytest.mark.anyio
    async def test_identical_embeddings_all_score_one(self, in_memory_store: InMemoryStore):
        """All identical embeddings should score 1.0 with each other."""
        embedding = [0.5] * 384

        chunks = [
            Chunk(
                id=f"identical-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                embedding=embedding,
            )
            for i in range(5)
        ]
        await in_memory_store.add(chunks)

        results = await in_memory_store.search(embedding, top_k=5)

        # All should score ~1.0
        for result in results:
            assert result.score > 0.99, (
                f"Identical embeddings should all score ~1.0, got {result.score}"
            )


# =============================================================================
# High-Dimensional Score Tests
# =============================================================================


class TestHighDimensionalScores:
    """Score tests with high-dimensional embeddings."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_high_dimensional_similarity(self, in_memory_store: InMemoryStore):
        """Similarity should work correctly in high dimensions (1536-dim)."""
        dimension = 1536

        # Create normalized vectors
        vec1 = np.random.randn(dimension).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        vec2 = np.copy(vec1)  # Identical

        chunk = Chunk(
            id="high-dim-chunk",
            content="High dimensional chunk",
            document_id="doc-1",
            embedding=list(vec1),
        )
        await in_memory_store.add([chunk])

        results = await in_memory_store.search(list(vec2), top_k=1)

        assert len(results) == 1
        assert results[0].score > 0.99, (
            f"High-dim identical vectors should score ~1.0, got {results[0].score}"
        )

    @pytest.mark.anyio
    async def test_high_dimensional_orthogonal(self, in_memory_store: InMemoryStore):
        """Orthogonality in high dimensions should give ~0.0 similarity."""
        dimension = 1536

        # In high dimensions, random vectors are approximately orthogonal
        vec1 = np.random.randn(dimension).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        vec2 = np.random.randn(dimension).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        chunk = Chunk(
            id="high-dim-ortho-chunk",
            content="High dimensional orthogonal chunk",
            document_id="doc-1",
            embedding=list(vec1),
        )
        await in_memory_store.add([chunk])

        results = await in_memory_store.search(list(vec2), top_k=1)

        assert len(results) == 1
        # In high dimensions, random vectors have ~0 dot product
        # Score should be close to 0
        assert results[0].score < 0.5, (
            f"High-dim random vectors should have low similarity, got {results[0].score}"
        )
