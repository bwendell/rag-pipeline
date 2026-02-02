"""Test helper functions for vector store tests.

Provides utility functions for creating test data, validating search results,
and comparing embeddings. These helpers ensure consistent testing patterns
across vector store tests.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from rag_pipeline.core.types import Chunk, Metadata, SearchResult, SourceType

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Data Factory Functions
# =============================================================================


def create_test_chunk(
    content: str,
    document_id: str,
    embedding: list[float],
    source_path: str = "/test/path.py",
    chunk_index: int = 0,
    start_index: int = 0,
    end_index: int | None = None,
) -> Chunk:
    """Create a test chunk with specified parameters.

    Factory function for creating chunks with custom content and metadata
    for use in tests.

    Args:
        content: The text content of the chunk.
        document_id: ID of the parent document.
        embedding: The embedding vector for the chunk.
        source_path: File path in metadata (default: "/test/path.py").
        chunk_index: Position within the document (default: 0).
        start_index: Character offset in original document (default: 0).
        end_index: End character offset (default: len(content)).

    Returns:
        Chunk: A new chunk with the specified parameters.

    Example:
        >>> chunk = create_test_chunk(
        ...     content="Test content",
        ...     document_id="doc-1",
        ...     embedding=[0.1, 0.2, 0.3],
        ...     source_path="/code/test.py"
        ... )
        >>> assert chunk.content == "Test content"
    """
    if end_index is None:
        end_index = start_index + len(content)

    return Chunk(
        id=str(uuid4()),
        content=content,
        document_id=document_id,
        metadata=Metadata(
            source_path=source_path,
            source_type=SourceType.FILESYSTEM,
            language="python",
            file_extension=".py",
        ),
        embedding=embedding,
        start_index=start_index,
        end_index=end_index,
        chunk_index=chunk_index,
    )


def create_test_chunks(
    count: int,
    document_id: str = "doc-1",
    embedding_dimension: int = 384,
    source_path: str = "/test/path.py",
) -> list[Chunk]:
    """Create multiple test chunks with random embeddings.

    Factory function for bulk creation of chunks with varying content
    and random embeddings.

    Args:
        count: Number of chunks to create.
        document_id: ID of the parent document (default: "doc-1").
        embedding_dimension: Dimension of embeddings (default: 384).
        source_path: File path in metadata (default: "/test/path.py").

    Returns:
        list[Chunk]: List of test chunks with unique content and embeddings.

    Example:
        >>> chunks = create_test_chunks(count=10, embedding_dimension=384)
        >>> assert len(chunks) == 10
        >>> assert all(len(c.embedding) == 384 for c in chunks)
    """
    chunks = []
    content_templates = [
        "Sample chunk content number {} with unique text.",
        "This is chunk {} about various topics.",
        "Information chunk {} with relevant details.",
        "Data chunk {} containing structured information.",
        "Context chunk {} for vector search testing.",
    ]

    for i in range(count):
        content_template = content_templates[i % len(content_templates)]
        content = content_template.format(i + 1)
        embedding = list(np.random.rand(embedding_dimension).astype(np.float32))

        chunk = create_test_chunk(
            content=content,
            document_id=document_id,
            embedding=embedding,
            source_path=source_path,
            chunk_index=i,
            start_index=i * 100,
            end_index=(i + 1) * 100,
        )
        chunks.append(chunk)

    return chunks


# =============================================================================
# Embedding Generation Functions
# =============================================================================


def generate_random_embedding(dimension: int) -> list[float]:
    """Generate a random embedding vector.

    Creates a random embedding with values between 0 and 1.

    Args:
        dimension: The dimension of the embedding.

    Returns:
        list[float]: A random embedding vector.

    Example:
        >>> embedding = generate_random_embedding(384)
        >>> assert len(embedding) == 384
        >>> assert all(0 <= x <= 1 for x in embedding)
    """
    return list(np.random.rand(dimension).astype(np.float32))


def generate_similar_embedding(
    base_embedding: list[float],
    noise_level: float = 0.01,
) -> list[float]:
    """Generate an embedding similar to a base embedding.

    Creates a noisy version of the base embedding for testing similarity
    calculations and near-duplicate detection.

    Args:
        base_embedding: The base embedding to add noise to.
        noise_level: Standard deviation of noise (default: 0.01).
            Lower values create more similar embeddings.

    Returns:
        list[float]: An embedding similar but not identical to the base.

    Example:
        >>> base = [0.1, 0.2, 0.3]
        >>> similar = generate_similar_embedding(base, noise_level=0.05)
        >>> assert len(similar) == len(base)
    """
    noise = np.random.normal(0, noise_level, len(base_embedding))
    return list((np.array(base_embedding) + noise).astype(np.float32))


# =============================================================================
# Search Result Validation Functions
# =============================================================================


def assert_search_results_valid(
    results: list[SearchResult],
    expected_count: int,
) -> None:
    """Validate that search results have correct structure and count.

    Checks that results contain the expected number of SearchResult objects,
    each with valid chunks and scores.

    Args:
        results: List of search results to validate.
        expected_count: Expected number of results.

    Raises:
        AssertionError: If validation fails.

    Example:
        >>> results = await store.search([0.1, 0.2], top_k=5)
        >>> assert_search_results_valid(results, expected_count=5)
    """
    assert results is not None, "Results cannot be None"
    assert isinstance(results, list), f"Results must be list, got {type(results)}"
    assert len(results) == expected_count, f"Expected {expected_count} results, got {len(results)}"

    for i, result in enumerate(results):
        assert isinstance(result, SearchResult), (
            f"Result {i} must be SearchResult, got {type(result)}"
        )
        assert result.chunk is not None, f"Result {i} has no chunk"
        assert isinstance(result.score, (int, float)), (
            f"Result {i} score must be numeric, got {type(result.score)}"
        )
        assert 0 <= result.score <= 1, f"Result {i} score {result.score} not in range [0, 1]"
        assert result.rank == i + 1, f"Result {i} rank should be {i + 1}, got {result.rank}"


def assert_search_results_ordered_by_score(
    results: list[SearchResult],
) -> None:
    """Validate that search results are ordered by score (descending).

    Checks that each result has a lower or equal score than the previous one.

    Args:
        results: List of search results to validate.

    Raises:
        AssertionError: If results are not properly ordered.

    Example:
        >>> results = await store.search([0.1, 0.2], top_k=5)
        >>> assert_search_results_ordered_by_score(results)
    """
    assert results is not None, "Results cannot be None"
    assert isinstance(results, list), f"Results must be list, got {type(results)}"

    if len(results) <= 1:
        return  # Single or no results are trivially ordered

    for i in range(len(results) - 1):
        current_score = results[i].score
        next_score = results[i + 1].score
        assert current_score >= next_score, (
            f"Results not ordered: result {i} has score {current_score}, "
            f"but result {i + 1} has score {next_score}"
        )


def assert_search_results_contain_chunk(
    results: list[SearchResult],
    chunk_id: str,
) -> SearchResult:
    """Assert that search results contain a specific chunk.

    Checks that the chunk with the given ID is present in the results
    and returns the corresponding SearchResult.

    Args:
        results: List of search results.
        chunk_id: The ID of the chunk to find.

    Returns:
        SearchResult: The search result containing the chunk.

    Raises:
        AssertionError: If the chunk is not found in results.

    Example:
        >>> results = await store.search([0.1, 0.2], top_k=5)
        >>> result = assert_search_results_contain_chunk(results, "chunk-1")
        >>> print(result.score)
    """
    for result in results:
        if result.chunk.id == chunk_id:
            return result

    chunk_ids = [r.chunk.id for r in results]
    raise AssertionError(f"Chunk {chunk_id} not found in results. Found: {chunk_ids}")


def assert_search_results_empty(results: list[SearchResult]) -> None:
    """Assert that search results are empty.

    Checks that the results list is empty, useful for testing no-match scenarios.

    Args:
        results: List of search results.

    Raises:
        AssertionError: If results is not empty.

    Example:
        >>> results = await store.search([0.0, 0.0], top_k=5)
        >>> assert_search_results_empty(results)
    """
    assert results is not None, "Results cannot be None"
    assert isinstance(results, list), f"Results must be list, got {type(results)}"
    assert len(results) == 0, f"Expected no results, got {len(results)}"


# =============================================================================
# Embedding Comparison Functions
# =============================================================================


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Computes the cosine of the angle between two vectors. The result is
    in the range [-1, 1] where 1 indicates identical direction, 0 indicates
    orthogonal vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        float: Cosine similarity in range [-1, 1].

    Raises:
        AssertionError: If vectors have different dimensions.

    Example:
        >>> vec1 = [1, 0, 0]
        >>> vec2 = [1, 0, 0]
        >>> cosine_similarity(vec1, vec2)
        1.0
    """
    assert len(vec1) == len(vec2), "Vectors must have same dimension"

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def assert_embeddings_similar(
    vec1: list[float],
    vec2: list[float],
    threshold: float = 0.8,
) -> None:
    """Assert two embeddings have high cosine similarity.

    Useful for testing that similar chunks produce similar embeddings
    in search results.

    Args:
        vec1: First embedding.
        vec2: Second embedding.
        threshold: Minimum similarity threshold (default 0.8).

    Raises:
        AssertionError: If similarity is below threshold.

    Example:
        >>> embedding1 = [1.0, 0.0, 0.0]
        >>> embedding2 = [0.99, 0.1, 0.0]
        >>> assert_embeddings_similar(embedding1, embedding2, threshold=0.9)
    """
    similarity = cosine_similarity(vec1, vec2)
    assert similarity >= threshold, (
        f"Embeddings are not similar enough: {similarity:.4f} < {threshold}"
    )


def assert_embeddings_different(
    vec1: list[float],
    vec2: list[float],
    threshold: float = 0.95,
) -> None:
    """Assert two embeddings are not too similar.

    Useful for testing that dissimilar chunks produce dissimilar embeddings.

    Args:
        vec1: First embedding.
        vec2: Second embedding.
        threshold: Maximum similarity threshold (default 0.95).

    Raises:
        AssertionError: If similarity is above threshold.

    Example:
        >>> embedding1 = [1.0, 0.0, 0.0]
        >>> embedding2 = [0.0, 1.0, 0.0]
        >>> assert_embeddings_different(embedding1, embedding2, threshold=0.1)
    """
    similarity = cosine_similarity(vec1, vec2)
    assert similarity < threshold, f"Embeddings are too similar: {similarity:.4f} >= {threshold}"
