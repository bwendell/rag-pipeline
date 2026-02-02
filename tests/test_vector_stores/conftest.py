"""Pytest fixtures for vector store tests.

This module provides vector store-specific fixtures that are automatically
loaded when running tests in the test_vector_stores directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rag_pipeline.core.types import Chunk, Metadata, SourceType
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore

if TYPE_CHECKING:
    pass


# =============================================================================
# Dimension Fixtures
# =============================================================================


@pytest.fixture
def embedding_dimension() -> int:
    """Return default embedding dimension for tests.

    Standard dimension for vector store testing (384-dimensional vectors).
    """
    return 384


# =============================================================================
# Embedding Fixtures
# =============================================================================


@pytest.fixture
def sample_embedding(embedding_dimension: int) -> list[float]:
    """Return a sample embedding vector.

    A random embedding vector with the standard dimension for testing
    vector store operations and similarity calculations.

    Args:
        embedding_dimension: The dimension of the embedding.

    Returns:
        list[float]: A list of random floats representing an embedding.
    """
    return [float(x) for x in np.random.rand(embedding_dimension).astype(np.float32)]


# =============================================================================
# Chunk Fixtures
# =============================================================================


@pytest.fixture
def sample_chunk(sample_embedding: list[float]) -> Chunk:
    """Return a single sample chunk with embedding.

    A complete chunk with realistic content, metadata, and embedding for
    testing basic vector store operations.

    Args:
        sample_embedding: The embedding vector for the chunk.

    Returns:
        Chunk: A sample chunk with id, content, document_id, metadata, and embedding.
    """
    return Chunk(
        id="chunk-1",
        content="This is a sample chunk content about Python functions.",
        document_id="doc-1",
        metadata=Metadata(
            source_path="/path/to/sample.py",
            source_type=SourceType.FILESYSTEM,
            language="python",
            file_extension=".py",
        ),
        embedding=sample_embedding,
        start_index=0,
        end_index=50,
        chunk_index=0,
    )


@pytest.fixture
def sample_chunks(embedding_dimension: int) -> list[Chunk]:
    """Return multiple sample chunks with different content and embeddings.

    Creates 5 chunks with varying content, metadata, and embeddings for
    testing batch operations and search functionality.

    Args:
        embedding_dimension: The dimension of embeddings.

    Returns:
        list[Chunk]: List of 5 sample chunks.
    """
    chunks = []
    contents = [
        "This is the first chunk about Python decorators and their usage.",
        "Here is information about context managers in Python.",
        "This chunk discusses async/await and asynchronous programming.",
        "Database migration strategies and best practices.",
        "Error handling patterns and exception management.",
    ]

    for i, content in enumerate(contents):
        embedding = list(np.random.rand(embedding_dimension).astype(np.float32))
        chunk = Chunk(
            id=f"chunk-{i + 1}",
            content=content,
            document_id=f"doc-{(i // 2) + 1}",  # 2-3 chunks per document
            metadata=Metadata(
                source_path=f"/path/to/file{(i // 2) + 1}.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
            embedding=embedding,
            start_index=i * 100,
            end_index=(i + 1) * 100,
            chunk_index=i,
        )
        chunks.append(chunk)

    return chunks


# =============================================================================
# Vector Store Fixtures
# =============================================================================


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    """Return a fresh InMemoryStore instance.

    Creates a new, empty vector store for testing. Each test gets its own
    isolated store instance.

    Returns:
        InMemoryStore: A new empty vector store instance.
    """
    return InMemoryStore()


@pytest.fixture
async def populated_store(
    in_memory_store: InMemoryStore, sample_chunks: list[Chunk]
) -> InMemoryStore:
    """Return an InMemoryStore pre-populated with sample chunks.

    Creates a store and adds sample chunks to it, useful for testing
    search, retrieval, and filtering operations.

    This is an async fixture that can be used with pytest-asyncio.

    Args:
        in_memory_store: A fresh empty store.
        sample_chunks: Sample chunks to add to the store.

    Returns:
        InMemoryStore: Store populated with sample chunks.
    """
    await in_memory_store.add(sample_chunks)
    return in_memory_store
