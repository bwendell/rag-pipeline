"""Test helper functions for embedding provider tests.

Provides utility functions for validating embeddings and comparing similarity.
These helpers ensure consistent testing patterns across embedding provider tests.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.core.types import EmbeddingVector


# =============================================================================
# Embedding Validation Functions
# =============================================================================


def assert_embedding_valid(
    embedding: EmbeddingVector,
    expected_dimension: int | None = None,
) -> None:
    """Validate that an embedding has a valid structure.

    Checks that the embedding is a list of numbers with the correct dimensions.

    Args:
        embedding: The embedding vector to validate.
        expected_dimension: Optional expected dimension to check.

    Raises:
        AssertionError: If validation fails.

    Example:
        >>> embedding = [0.1, 0.2, 0.3]
        >>> assert_embedding_valid(embedding, expected_dimension=3)
    """
    assert embedding is not None, "Embedding cannot be None"
    assert isinstance(embedding, list), f"Embedding must be list, got {type(embedding)}"
    assert len(embedding) > 0, "Embedding cannot be empty"

    # Check all elements are floats/numbers
    assert all(isinstance(x, (int, float)) for x in embedding), (
        "All embedding elements must be numbers"
    )

    if expected_dimension is not None:
        assert len(embedding) == expected_dimension, (
            f"Embedding dimension mismatch: expected {expected_dimension}, got {len(embedding)}"
        )


def assert_is_zero_vector(embedding: EmbeddingVector) -> None:
    """Assert that an embedding is a zero vector.

    Checks that all elements in the embedding are exactly 0.0.

    Args:
        embedding: The embedding to check.

    Raises:
        AssertionError: If embedding is not all zeros.

    Example:
        >>> embedding = [0.0, 0.0, 0.0]
        >>> assert_is_zero_vector(embedding)
    """
    assert all(x == 0.0 for x in embedding), "Embedding is not a zero vector"


def assert_not_zero_vector(embedding: EmbeddingVector) -> None:
    """Assert that an embedding is NOT a zero vector.

    Checks that at least one element in the embedding is non-zero.

    Args:
        embedding: The embedding to check.

    Raises:
        AssertionError: If embedding is all zeros.

    Example:
        >>> embedding = [0.1, 0.0, 0.0]
        >>> assert_not_zero_vector(embedding)
    """
    assert any(x != 0.0 for x in embedding), "Embedding should not be a zero vector"


# =============================================================================
# Similarity Calculation Functions
# =============================================================================


def cosine_similarity(vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
    """Calculate cosine similarity between two vectors.

    Computes the cosine of the angle between two vectors. The result is
    in the range [-1, 1] where 1 indicates identical direction, 0 indicates
    orthogonal vectors, and -1 indicates opposite direction.

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


def euclidean_distance(vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
    """Calculate Euclidean distance between two vectors.

    Computes the straight-line distance between two points in embedding space.
    Lower values indicate more similar embeddings.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        float: Euclidean distance (always non-negative).

    Raises:
        AssertionError: If vectors have different dimensions.

    Example:
        >>> vec1 = [0, 0, 0]
        >>> vec2 = [3, 4, 0]
        >>> euclidean_distance(vec1, vec2)
        5.0
    """
    assert len(vec1) == len(vec2), "Vectors must have same dimension"

    sum_squared_diff = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    return math.sqrt(sum_squared_diff)


# =============================================================================
# Embedding Comparison Functions
# =============================================================================


def assert_embeddings_similar(
    vec1: EmbeddingVector,
    vec2: EmbeddingVector,
    threshold: float = 0.8,
) -> None:
    """Assert two embeddings have high cosine similarity.

    Useful for testing that similar texts produce similar embeddings.

    Args:
        vec1: First embedding.
        vec2: Second embedding.
        threshold: Minimum similarity threshold (default 0.8).

    Raises:
        AssertionError: If similarity is below threshold.

    Example:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [0.99, 0.1, 0.0]
        >>> assert_embeddings_similar(vec1, vec2, threshold=0.9)
    """
    similarity = cosine_similarity(vec1, vec2)
    assert similarity >= threshold, (
        f"Embeddings are not similar enough: {similarity:.4f} < {threshold}"
    )


def assert_embeddings_different(
    vec1: EmbeddingVector,
    vec2: EmbeddingVector,
    threshold: float = 0.95,
) -> None:
    """Assert two embeddings are not too similar (different texts).

    Useful for testing that dissimilar texts produce dissimilar embeddings.

    Args:
        vec1: First embedding.
        vec2: Second embedding.
        threshold: Maximum similarity threshold (default 0.95).

    Raises:
        AssertionError: If similarity is above threshold.

    Example:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [0.0, 1.0, 0.0]
        >>> assert_embeddings_different(vec1, vec2, threshold=0.1)
    """
    similarity = cosine_similarity(vec1, vec2)
    assert similarity < threshold, f"Embeddings are too similar: {similarity:.4f} >= {threshold}"


def assert_embedding_magnitude(
    embedding: EmbeddingVector,
    expected: float | None = None,
    tolerance: float = 0.01,
) -> None:
    """Assert embedding has expected magnitude (norm).

    Useful for testing normalization and magnitude constraints.

    Args:
        embedding: The embedding to check.
        expected: Expected magnitude. If None, just verify non-zero.
        tolerance: Tolerance for magnitude difference (default 0.01).

    Raises:
        AssertionError: If magnitude check fails.

    Example:
        >>> embedding = [0.6, 0.8, 0.0]  # magnitude = 1.0
        >>> assert_embedding_magnitude(embedding, expected=1.0)
    """
    magnitude = math.sqrt(sum(x * x for x in embedding))

    if expected is None:
        assert magnitude > 0, "Embedding magnitude should be non-zero"
    else:
        assert abs(magnitude - expected) <= tolerance, (
            f"Embedding magnitude {magnitude:.4f} "
            f"does not match expected {expected:.4f} (tolerance: {tolerance})"
        )


# =============================================================================
# Batch Validation Functions
# =============================================================================


def assert_embeddings_batch_valid(
    embeddings: list[EmbeddingVector],
    expected_count: int,
    expected_dimension: int | None = None,
) -> None:
    """Validate a batch of embeddings.

    Checks that the batch has the expected number of embeddings and that
    each embedding is valid with consistent dimensionality.

    Args:
        embeddings: List of embedding vectors.
        expected_count: Expected number of embeddings.
        expected_dimension: Optional expected dimension for each embedding.

    Raises:
        AssertionError: If validation fails.

    Example:
        >>> embeddings = [[0.1, 0.2], [0.3, 0.4]]
        >>> assert_embeddings_batch_valid(embeddings, expected_count=2, expected_dimension=2)
    """
    assert len(embeddings) == expected_count, (
        f"Expected {expected_count} embeddings, got {len(embeddings)}"
    )

    for i, embedding in enumerate(embeddings):
        try:
            assert_embedding_valid(embedding, expected_dimension)
        except AssertionError as e:
            raise AssertionError(f"Embedding at index {i} is invalid: {e}") from e


def assert_all_embeddings_different(
    embeddings: list[EmbeddingVector],
    threshold: float = 0.95,
) -> None:
    """Assert that all embeddings in a batch are different from each other.

    Useful for testing that different inputs produce different embeddings.

    Args:
        embeddings: List of embedding vectors.
        threshold: Maximum similarity threshold (default 0.95).

    Raises:
        AssertionError: If any pair of embeddings is too similar.
    """
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            assert similarity < threshold, (
                f"Embeddings at indices {i} and {j} are too similar: "
                f"{similarity:.4f} >= {threshold}"
            )


def assert_embeddings_consistent_dimension(
    embeddings: list[EmbeddingVector],
) -> int:
    """Assert all embeddings have the same dimension and return it.

    Useful for batch validation that requires consistent embedding size.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        int: The dimension of the embeddings.

    Raises:
        AssertionError: If embeddings have inconsistent dimensions.

    Example:
        >>> embeddings = [[0.1, 0.2], [0.3, 0.4]]
        >>> dim = assert_embeddings_consistent_dimension(embeddings)
        >>> dim
        2
    """
    if not embeddings:
        raise AssertionError("Cannot check dimension of empty embedding list")

    first_dimension = len(embeddings[0])

    for i, embedding in enumerate(embeddings):
        assert len(embedding) == first_dimension, (
            f"Embedding at index {i} has dimension {len(embedding)}, expected {first_dimension}"
        )

    return first_dimension
