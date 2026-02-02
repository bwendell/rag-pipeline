"""Pytest fixtures for embedding provider tests.

This module provides embedding provider-specific fixtures that are automatically
loaded when running tests in the test_embedding_providers directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def embedding_dimension() -> int:
    """Return default embedding dimension for tests.

    Standard dimension for embedding model testing (384-dimensional vectors).
    """
    return 384


@pytest.fixture
def mock_sentence_transformer(embedding_dimension: int):
    """Create a mocked SentenceTransformer model.

    Returns a MagicMock that simulates SentenceTransformer behavior
    including encode() and get_sentence_embedding_dimension() methods.
    The mock generates random embeddings for any input text.

    Args:
        embedding_dimension: The dimension of generated embeddings.

    Returns:
        MagicMock: A mock SentenceTransformer instance.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = embedding_dimension

    # Single text encoding returns 1D array, batch returns 2D array
    def mock_encode(text_or_texts, **kwargs):
        if isinstance(text_or_texts, str):
            return np.random.rand(embedding_dimension).astype(np.float32)
        else:
            return np.random.rand(len(text_or_texts), embedding_dimension).astype(np.float32)

    mock_model.encode.side_effect = mock_encode
    return mock_model


@pytest.fixture
def mock_sentence_transformers_provider(mock_sentence_transformer, embedding_dimension: int):
    """Create a SentenceTransformersProvider with mocked model.

    Patches the SentenceTransformer class to return mock_sentence_transformer
    and creates a provider instance.

    Args:
        mock_sentence_transformer: The mocked SentenceTransformer model.
        embedding_dimension: The embedding dimension.

    Yields:
        SentenceTransformersProvider: A provider with mocked model.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = mock_sentence_transformer

        from rag_pipeline.embedding_providers import SentenceTransformersProvider

        provider = SentenceTransformersProvider()
        # Force model loading to populate dimension
        provider._ensure_model_loaded()
        yield provider


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_texts() -> list[str]:
    """Return sample texts for testing batch embedding.

    A list of diverse text samples with varying lengths for testing
    batch embedding functionality.

    Returns:
        list[str]: List of 5 sample text strings.
    """
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "The third text is a bit longer and contains more words.",
        "Short text.",
        "Final sample for testing purposes.",
    ]


@pytest.fixture
def sample_text() -> str:
    """Return a single sample text for testing.

    A simple text string for testing single text embedding.

    Returns:
        str: A sample text string.
    """
    return "This is a sample text for embedding."


@pytest.fixture
def empty_string() -> str:
    """Return an empty string for edge case testing.

    Returns:
        str: An empty string.
    """
    return ""


@pytest.fixture
def texts_with_empty_strings() -> list[str]:
    """Return texts including empty strings for testing.

    A list of texts mixed with empty strings to test handling of
    edge cases in batch processing.

    Returns:
        list[str]: List of texts with empty string entries.
    """
    return [
        "First text",
        "",
        "Third text",
        "",
        "Fifth text",
    ]


@pytest.fixture
def long_text() -> str:
    """Return a long text for testing handling of large inputs.

    A lengthy text string that tests embedding performance and
    memory handling with longer documents.

    Returns:
        str: A long multi-paragraph text.
    """
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"This is paragraph {i + 1} of a long document. "
            f"It contains enough text to simulate realistic document content. "
            f"Each paragraph adds more context and information to the overall text. "
            f"This helps test how embedding providers handle larger inputs."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def short_text() -> str:
    """Return a very short text for testing minimal input.

    A single short phrase for testing edge cases with minimal input.

    Returns:
        str: A short text string.
    """
    return "Hello"


# =============================================================================
# Embedding Result Fixtures
# =============================================================================


@pytest.fixture
def sample_embedding(embedding_dimension: int) -> list[float]:
    """Return a sample embedding vector.

    A random embedding vector with the standard dimension for testing
    embedding validation and similarity calculations.

    Args:
        embedding_dimension: The dimension of the embedding.

    Returns:
        list[float]: A list of random floats representing an embedding.
    """
    return list(np.random.rand(embedding_dimension).astype(np.float32))


@pytest.fixture
def zero_embedding(embedding_dimension: int) -> list[float]:
    """Return a zero embedding vector.

    An all-zeros embedding for testing edge cases like null embeddings
    or zero vectors.

    Args:
        embedding_dimension: The dimension of the embedding.

    Returns:
        list[float]: A list of zeros.
    """
    return [0.0] * embedding_dimension


@pytest.fixture
def normalized_embedding(embedding_dimension: int) -> list[float]:
    """Return a normalized embedding vector.

    An embedding vector with unit norm (magnitude = 1) for testing
    normalized embedding operations.

    Args:
        embedding_dimension: The dimension of the embedding.

    Returns:
        list[float]: A normalized embedding vector.
    """
    vec = np.random.rand(embedding_dimension).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1.0
    return list(vec / norm)


@pytest.fixture
def similar_embedding(sample_embedding: list[float]) -> list[float]:
    """Return an embedding similar to sample_embedding.

    Creates a slightly noisy version of the sample embedding to test
    similarity calculations and near-duplicate detection.

    Args:
        sample_embedding: The base embedding to create a similar one from.

    Returns:
        list[float]: An embedding similar but not identical to the sample.
    """
    noise = np.random.normal(0, 0.01, len(sample_embedding))
    return list((np.array(sample_embedding) + noise).astype(np.float32))


@pytest.fixture
def different_embedding(embedding_dimension: int) -> list[float]:
    """Return an embedding different from typical samples.

    A randomly generated embedding that should be dissimilar to others
    for testing difference detection.

    Args:
        embedding_dimension: The dimension of the embedding.

    Returns:
        list[float]: A random embedding vector.
    """
    return list(np.random.rand(embedding_dimension).astype(np.float32))
