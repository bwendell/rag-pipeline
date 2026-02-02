"""Tests for ChromaDB vector store implementation.

This module provides comprehensive test coverage for the ChromaDBStore class,
including all VectorStore protocol methods, error handling, and edge cases.

Tests use a temporary directory to avoid interfering with any existing ChromaDB data.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, Metadata, SearchResult, SourceType
from rag_pipeline.vector_stores.chroma_store import ChromaDBStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest_asyncio.fixture
async def chroma_store() -> AsyncGenerator[ChromaDBStore, None]:
    """Create a temporary ChromaDBStore for testing.

    Yields:
        ChromaDBStore instance with a temporary persist directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        store = ChromaDBStore(
            persist_path=temp_dir,
            collection_name="test_collection",
        )
        yield store


def create_test_chunk(
    content: str = "test content",
    document_id: str = "doc-123",
    embedding: list[float] | None = None,
    source_path: str = "/path/to/file.py",
    language: str = "python",
    chunk_index: int = 0,
) -> Chunk:
    """Create a test chunk with the given parameters.

    Args:
        content: Chunk content.
        document_id: Parent document ID.
        embedding: Optional embedding vector (default: 384-dim zeros).
        source_path: Source file path.
        language: Programming language.
        chunk_index: Position within document.

    Returns:
        Chunk instance with all fields populated.
    """
    if embedding is None:
        embedding = [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

    return Chunk(
        content=content,
        document_id=document_id,
        embedding=embedding,
        metadata=Metadata(
            source_path=source_path,
            source_type=SourceType.FILESYSTEM,
            language=language,
            file_extension=".py",
        ),
        chunk_index=chunk_index,
        start_index=chunk_index * 100,
        end_index=chunk_index * 100 + len(content),
    )


class TestChromaDBStoreInitialization:
    """Tests for ChromaDBStore initialization."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        store = ChromaDBStore()

        assert store.persist_path == "./data/chroma"
        assert store.collection_name == "rag_chunks"
        assert store._client is None
        assert store._collection is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        store = ChromaDBStore(
            persist_path="/custom/path",
            collection_name="custom_collection",
        )

        assert store.persist_path == "/custom/path"
        assert store.collection_name == "custom_collection"

    def test_initialization_from_env(self, monkeypatch):
        """Test that RAG_CHROMA_PATH environment variable is respected."""
        monkeypatch.setenv("RAG_CHROMA_PATH", "/env/path")
        store = ChromaDBStore()

        assert store.persist_path == "/env/path"

    def test_env_overrides_default(self, monkeypatch):
        """Test that env var overrides default but not explicit parameter."""
        monkeypatch.setenv("RAG_CHROMA_PATH", "/env/path")
        store = ChromaDBStore(persist_path="/explicit/path")

        assert store.persist_path == "/explicit/path"


class TestChromaDBStoreAdd:
    """Tests for the add() method."""

    @pytest.mark.asyncio
    async def test_add_single_chunk(self, chroma_store: ChromaDBStore):
        """Test adding a single chunk."""
        chunk = create_test_chunk(content="Hello, world!")

        ids = await chroma_store.add([chunk])

        assert len(ids) == 1
        assert ids[0] == chunk.id

    @pytest.mark.asyncio
    async def test_add_multiple_chunks(self, chroma_store: ChromaDBStore):
        """Test adding multiple chunks."""
        chunks = [
            create_test_chunk(content="Chunk 1", chunk_index=0),
            create_test_chunk(content="Chunk 2", chunk_index=1),
            create_test_chunk(content="Chunk 3", chunk_index=2),
        ]

        ids = await chroma_store.add(chunks)

        assert len(ids) == 3
        assert ids[0] == chunks[0].id
        assert ids[1] == chunks[1].id
        assert ids[2] == chunks[2].id

    @pytest.mark.asyncio
    async def test_add_empty_list(self, chroma_store: ChromaDBStore):
        """Test adding an empty list returns empty list."""
        ids = await chroma_store.add([])

        assert ids == []

    @pytest.mark.asyncio
    async def test_add_chunk_without_embedding_raises(self, chroma_store: ChromaDBStore):
        """Test that adding a chunk without embedding raises ValueError."""
        chunk = Chunk(
            content="No embedding",
            document_id="doc-123",
            embedding=None,
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await chroma_store.add([chunk])

    @pytest.mark.asyncio
    async def test_add_upserts_existing(self, chroma_store: ChromaDBStore):
        """Test that adding a chunk with existing ID updates it."""
        chunk = create_test_chunk(content="Original content")

        # Add first time
        ids1 = await chroma_store.add([chunk])

        # Modify and add again with same ID
        chunk.content = "Updated content"
        ids2 = await chroma_store.add([chunk])

        assert ids1[0] == ids2[0]

        # Verify content was updated
        retrieved = await chroma_store.get([chunk.id])
        assert len(retrieved) == 1
        assert retrieved[0].content == "Updated content"

    @pytest.mark.asyncio
    async def test_add_preserves_document_id(self, chroma_store: ChromaDBStore):
        """Test that document_id is preserved in metadata for later lookup."""
        chunk = create_test_chunk(document_id="my-doc-123")

        await chroma_store.add([chunk])

        # Retrieve and verify document_id is preserved
        retrieved = await chroma_store.get([chunk.id])
        assert retrieved[0].document_id == "my-doc-123"


class TestChromaDBStoreSearch:
    """Tests for the search() method."""

    @pytest.mark.asyncio
    async def test_search_empty_store(self, chroma_store: ChromaDBStore):
        """Test search on empty store returns empty list."""
        results = await chroma_store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self, chroma_store: ChromaDBStore):
        """Test search returns results after adding chunks."""
        # Add some chunks
        chunks = [
            create_test_chunk(
                content="Python programming guide",
                embedding=[0.9] * 384,
            ),
            create_test_chunk(
                content="Java programming guide",
                embedding=[0.1] * 384,
            ),
        ]
        await chroma_store.add(chunks)

        # Search with embedding closer to first chunk
        results = await chroma_store.search(
            query_embedding=[0.95] * 384,
            top_k=2,
        )

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        # Score can slightly exceed 1.0 due to floating point precision
        assert 0.0 <= results[0].score <= 1.01
        assert results[0].rank == 1
        assert results[1].rank == 2

    @pytest.mark.asyncio
    async def test_search_with_filters(self, chroma_store: ChromaDBStore):
        """Test search with metadata filters."""
        # Add chunks with different languages
        py_chunk = create_test_chunk(
            content="Python code",
            language="python",
            embedding=[0.9] * 384,
        )
        js_chunk = create_test_chunk(
            content="JavaScript code",
            source_path="/path/to/file.js",
            language="javascript",
            embedding=[0.9] * 384,
        )
        await chroma_store.add([py_chunk, js_chunk])

        # Search with language filter
        results = await chroma_store.search(
            query_embedding=[0.9] * 384,
            top_k=5,
            filters={"language": "python"},
        )

        # Should only return Python chunk
        assert len(results) == 1
        assert results[0].chunk.metadata.language == "python"

    @pytest.mark.asyncio
    async def test_search_top_k_limit(self, chroma_store: ChromaDBStore):
        """Test that top_k limits the number of results."""
        # Add multiple chunks
        chunks = [create_test_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(10)]
        await chroma_store.add(chunks)

        results = await chroma_store.search(
            query_embedding=[0.5] * 384,
            top_k=3,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_score_is_similarity(self, chroma_store: ChromaDBStore):
        """Test that search returns similarity scores (higher is better)."""
        chunk = create_test_chunk(
            content="Test content",
            embedding=[1.0] * 384,
        )
        await chroma_store.add([chunk])

        # Search with identical embedding should return score close to 1
        results = await chroma_store.search(
            query_embedding=[1.0] * 384,
            top_k=1,
        )

        assert len(results) == 1
        # Similarity should be high (close to 1) for identical embeddings
        assert results[0].score > 0.9


class TestChromaDBStoreDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_single_chunk(self, chroma_store: ChromaDBStore):
        """Test deleting a single chunk."""
        chunk = create_test_chunk()
        await chroma_store.add([chunk])

        deleted = await chroma_store.delete([chunk.id])

        assert deleted == 1

        # Verify it's gone
        retrieved = await chroma_store.get([chunk.id])
        assert retrieved == []

    @pytest.mark.asyncio
    async def test_delete_multiple_chunks(self, chroma_store: ChromaDBStore):
        """Test deleting multiple chunks."""
        chunks = [create_test_chunk(chunk_index=i) for i in range(3)]
        await chroma_store.add(chunks)

        deleted = await chroma_store.delete([c.id for c in chunks[:2]])

        assert deleted == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id(self, chroma_store: ChromaDBStore):
        """Test deleting a non-existent ID returns 0."""
        deleted = await chroma_store.delete(["nonexistent-id"])

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, chroma_store: ChromaDBStore):
        """Test deleting an empty list returns 0."""
        deleted = await chroma_store.delete([])

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_by_document(self, chroma_store: ChromaDBStore):
        """Test deleting all chunks for a document."""
        # Add chunks for different documents
        doc1_chunks = [create_test_chunk(document_id="doc-1", chunk_index=i) for i in range(3)]
        doc2_chunks = [create_test_chunk(document_id="doc-2", chunk_index=i) for i in range(2)]
        await chroma_store.add(doc1_chunks + doc2_chunks)

        deleted = await chroma_store.delete_by_document("doc-1")

        assert deleted == 3

        # Verify doc-1 chunks are gone but doc-2 remains
        assert await chroma_store.count() == 2

    @pytest.mark.asyncio
    async def test_delete_by_document_nonexistent(self, chroma_store: ChromaDBStore):
        """Test deleting by non-existent document returns 0."""
        deleted = await chroma_store.delete_by_document("nonexistent-doc")

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_by_source(self, chroma_store: ChromaDBStore):
        """Test deleting all chunks for a source path."""
        # Add chunks from different sources
        chunks1 = [create_test_chunk(source_path="/path/file1.py", chunk_index=i) for i in range(2)]
        chunks2 = [create_test_chunk(source_path="/path/file2.py", chunk_index=i) for i in range(3)]
        await chroma_store.add(chunks1 + chunks2)

        deleted = await chroma_store.delete_by_source("/path/file1.py")

        assert deleted == 2

        # Verify file1 chunks are gone but file2 remains
        assert await chroma_store.count() == 3

    @pytest.mark.asyncio
    async def test_delete_by_source_nonexistent(self, chroma_store: ChromaDBStore):
        """Test deleting by non-existent source returns 0."""
        deleted = await chroma_store.delete_by_source("/nonexistent/file.py")

        assert deleted == 0


class TestChromaDBStoreGet:
    """Tests for retrieval operations."""

    @pytest.mark.asyncio
    async def test_get_single_chunk(self, chroma_store: ChromaDBStore):
        """Test retrieving a single chunk."""
        chunk = create_test_chunk(content="Hello, world!")
        await chroma_store.add([chunk])

        retrieved = await chroma_store.get([chunk.id])

        assert len(retrieved) == 1
        assert retrieved[0].id == chunk.id
        assert retrieved[0].content == "Hello, world!"
        assert retrieved[0].document_id == chunk.document_id

    @pytest.mark.asyncio
    async def test_get_multiple_chunks(self, chroma_store: ChromaDBStore):
        """Test retrieving multiple chunks."""
        chunks = [create_test_chunk(chunk_index=i) for i in range(3)]
        await chroma_store.add(chunks)

        retrieved = await chroma_store.get([c.id for c in chunks])

        assert len(retrieved) == 3

    @pytest.mark.asyncio
    async def test_get_missing_id_skipped(self, chroma_store: ChromaDBStore):
        """Test that missing IDs are silently skipped."""
        chunk = create_test_chunk()
        await chroma_store.add([chunk])

        retrieved = await chroma_store.get([chunk.id, "nonexistent-id"])

        assert len(retrieved) == 1
        assert retrieved[0].id == chunk.id

    @pytest.mark.asyncio
    async def test_get_empty_list(self, chroma_store: ChromaDBStore):
        """Test getting an empty list returns empty list."""
        retrieved = await chroma_store.get([])

        assert retrieved == []

    @pytest.mark.asyncio
    async def test_get_empty_store(self, chroma_store: ChromaDBStore):
        """Test getting from empty store returns empty list."""
        retrieved = await chroma_store.get(["nonexistent-id"])

        assert retrieved == []

    @pytest.mark.asyncio
    async def test_get_by_source(self, chroma_store: ChromaDBStore):
        """Test retrieving chunks by source path."""
        # Add chunks from different sources
        chunks1 = [create_test_chunk(source_path="/path/file1.py", chunk_index=i) for i in range(2)]
        chunks2 = [create_test_chunk(source_path="/path/file2.py", chunk_index=i) for i in range(3)]
        await chroma_store.add(chunks1 + chunks2)

        retrieved = await chroma_store.get_by_source("/path/file1.py")

        assert len(retrieved) == 2
        for chunk in retrieved:
            assert chunk.metadata.source_path == "/path/file1.py"

    @pytest.mark.asyncio
    async def test_get_by_source_sorted(self, chroma_store: ChromaDBStore):
        """Test that chunks are sorted by chunk_index."""
        chunks = [
            create_test_chunk(source_path="/path/file.py", chunk_index=2),
            create_test_chunk(source_path="/path/file.py", chunk_index=0),
            create_test_chunk(source_path="/path/file.py", chunk_index=1),
        ]
        await chroma_store.add(chunks)

        retrieved = await chroma_store.get_by_source("/path/file.py")

        assert len(retrieved) == 3
        assert retrieved[0].chunk_index == 0
        assert retrieved[1].chunk_index == 1
        assert retrieved[2].chunk_index == 2

    @pytest.mark.asyncio
    async def test_get_by_source_limit(self, chroma_store: ChromaDBStore):
        """Test that limit parameter works."""
        chunks = [create_test_chunk(source_path="/path/file.py", chunk_index=i) for i in range(10)]
        await chroma_store.add(chunks)

        retrieved = await chroma_store.get_by_source("/path/file.py", limit=3)

        assert len(retrieved) <= 3

    @pytest.mark.asyncio
    async def test_get_by_source_nonexistent(self, chroma_store: ChromaDBStore):
        """Test getting by non-existent source returns empty list."""
        retrieved = await chroma_store.get_by_source("/nonexistent/file.py")

        assert retrieved == []

    @pytest.mark.asyncio
    async def test_get_all_source_paths(self, chroma_store: ChromaDBStore):
        """Test getting all unique source paths."""
        chunks = [
            create_test_chunk(source_path="/path/file1.py"),
            create_test_chunk(source_path="/path/file1.py"),  # Duplicate path
            create_test_chunk(source_path="/path/file2.py"),
            create_test_chunk(source_path="/path/file3.py"),
        ]
        await chroma_store.add(chunks)

        paths = await chroma_store.get_all_source_paths()

        assert len(paths) == 3
        assert "/path/file1.py" in paths
        assert "/path/file2.py" in paths
        assert "/path/file3.py" in paths
        # Verify sorted
        assert paths == sorted(paths)

    @pytest.mark.asyncio
    async def test_get_all_source_paths_empty_store(self, chroma_store: ChromaDBStore):
        """Test getting source paths from empty store."""
        paths = await chroma_store.get_all_source_paths()

        assert paths == []


class TestChromaDBStoreStats:
    """Tests for statistics operations."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_store(self, chroma_store: ChromaDBStore):
        """Test stats for empty store."""
        stats = await chroma_store.get_stats()

        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0
        assert stats["embedding_dimension"] == 0
        assert stats["index_type"] == "hnsw"
        assert stats["store_type"] == "chroma"
        assert stats["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_get_stats_with_chunks(self, chroma_store: ChromaDBStore):
        """Test stats with chunks."""
        # Add chunks from different documents
        chunks = [
            create_test_chunk(document_id="doc-1", chunk_index=0),
            create_test_chunk(document_id="doc-1", chunk_index=1),
            create_test_chunk(document_id="doc-2", chunk_index=0),
        ]
        await chroma_store.add(chunks)

        stats = await chroma_store.get_stats()

        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert stats["embedding_dimension"] == 384

    @pytest.mark.asyncio
    async def test_count_empty_store(self, chroma_store: ChromaDBStore):
        """Test count on empty store."""
        count = await chroma_store.count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_with_chunks(self, chroma_store: ChromaDBStore):
        """Test count with chunks."""
        chunks = [create_test_chunk(chunk_index=i) for i in range(5)]
        await chroma_store.add(chunks)

        count = await chroma_store.count()

        assert count == 5


class TestChromaDBStoreClear:
    """Tests for clear operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_chunks(self, chroma_store: ChromaDBStore):
        """Test that clear removes all chunks."""
        chunks = [create_test_chunk(chunk_index=i) for i in range(5)]
        await chroma_store.add(chunks)
        assert await chroma_store.count() == 5

        await chroma_store.clear()

        assert await chroma_store.count() == 0

    @pytest.mark.asyncio
    async def test_clear_empty_store(self, chroma_store: ChromaDBStore):
        """Test clearing an empty store doesn't error."""
        await chroma_store.clear()

        assert await chroma_store.count() == 0


class TestChromaDBStoreHealthCheck:
    """Tests for health check operation."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, chroma_store: ChromaDBStore):
        """Test health check returns True for working store."""
        is_healthy = await chroma_store.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_after_operations(self, chroma_store: ChromaDBStore):
        """Test health check works after various operations."""
        # Add, delete, search operations
        chunk = create_test_chunk()
        await chroma_store.add([chunk])
        await chroma_store.search([0.5] * 384)
        await chroma_store.delete([chunk.id])

        is_healthy = await chroma_store.health_check()

        assert is_healthy is True


class TestChromaDBStoreErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_persist_path(self):
        """Test error handling for invalid persist path."""
        # Use a path that can't be created (e.g., a file as directory)
        with tempfile.NamedTemporaryFile() as temp_file:
            store = ChromaDBStore(persist_path=temp_file.name)

            # This should raise VectorStoreError when trying to create client
            with pytest.raises(VectorStoreError):
                await store.count()

    @pytest.mark.asyncio
    async def test_add_with_invalid_content(self, chroma_store: ChromaDBStore):
        """Test add operation with edge case content."""
        chunk = create_test_chunk(content="test content")
        ids = await chroma_store.add([chunk])
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_search_with_chromadb_error(self, chroma_store: ChromaDBStore):
        """Test error handling when ChromaDB raises an exception during search."""
        original_collection = chroma_store._ensure_collection()

        with patch.object(original_collection, "query", side_effect=RuntimeError("Query failed")):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.search([0.0] * 384, top_k=5)

            assert "Failed to search" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_with_chromadb_error(self, chroma_store: ChromaDBStore):
        """Test error handling when ChromaDB raises an exception during delete."""
        original_collection = chroma_store._ensure_collection()

        with patch.object(original_collection, "delete", side_effect=RuntimeError("Delete failed")):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.delete(["id-1"])

            assert "Failed to delete" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_with_chromadb_error(self, chroma_store: ChromaDBStore):
        """Test error handling when ChromaDB raises an exception during get."""
        original_collection = chroma_store._ensure_collection()

        with patch.object(original_collection, "get", side_effect=RuntimeError("Get failed")):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.get(["id-1"])

            assert "Failed to get chunks" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clear_with_existing_chunks(self, chroma_store: ChromaDBStore):
        """Test clearing store with existing chunks."""
        chunk = create_test_chunk()
        await chroma_store.add([chunk])

        count_before = await chroma_store.count()
        assert count_before == 1

        await chroma_store.clear()
        count_after = await chroma_store.count()
        assert count_after == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_chromadb_error(self, chroma_store: ChromaDBStore):
        """Test error handling when ChromaDB raises an exception during get_stats."""
        original_collection = chroma_store._ensure_collection()

        with patch.object(original_collection, "count", side_effect=RuntimeError("Count failed")):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.get_stats()

            assert "Failed to get stats" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_count_with_chromadb_error(self, chroma_store: ChromaDBStore):
        """Test error handling when ChromaDB raises an exception during count."""
        original_collection = chroma_store._ensure_collection()

        with patch.object(original_collection, "count", side_effect=RuntimeError("Count failed")):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.count()

            assert "Failed to count" in str(exc_info.value)


class TestChromaDBStoreEdgeCases:
    """Tests for edge case handling."""

    @pytest.mark.asyncio
    async def test_add_empty_list(self, chroma_store: ChromaDBStore):
        """Test adding an empty list of chunks."""
        result = await chroma_store.add([])
        assert result == []

    @pytest.mark.asyncio
    async def test_search_zero_top_k_returns_empty(self, chroma_store: ChromaDBStore):
        """Test search with top_k=0 returns empty list."""
        chunk = create_test_chunk()
        await chroma_store.add([chunk])

        results = await chroma_store.search([0.0] * 384, top_k=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_none_filters(self, chroma_store: ChromaDBStore):
        """Test search with None filters (should use all chunks)."""
        chunk = create_test_chunk()
        await chroma_store.add([chunk])

        results = await chroma_store.search([0.0] * 384, top_k=10, filters=None)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, chroma_store: ChromaDBStore):
        """Test deleting an empty list of IDs."""
        # Should not raise an error
        await chroma_store.delete([])

    @pytest.mark.asyncio
    async def test_get_empty_list(self, chroma_store: ChromaDBStore):
        """Test getting an empty list of IDs."""
        results = await chroma_store.get([])
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_ids(self, chroma_store: ChromaDBStore):
        """Test deleting IDs that don't exist."""
        # Should not raise an error (ChromaDB allows this)
        await chroma_store.delete(["nonexistent-1", "nonexistent-2"])

    @pytest.mark.asyncio
    async def test_get_nonexistent_ids(self, chroma_store: ChromaDBStore):
        """Test getting IDs that don't exist."""
        results = await chroma_store.get(["nonexistent-1", "nonexistent-2"])
        assert results == []

    @pytest.mark.asyncio
    async def test_large_batch_add(self, chroma_store: ChromaDBStore):
        """Test adding a large batch of chunks."""
        chunks = [
            create_test_chunk(
                content=f"Content {i}",
                document_id=f"doc-{i}",
                chunk_index=i,
            )
            for i in range(100)
        ]

        ids = await chroma_store.add(chunks)
        assert len(ids) == 100

        count = await chroma_store.count()
        assert count == 100

    @pytest.mark.asyncio
    async def test_very_long_content(self, chroma_store: ChromaDBStore):
        """Test adding chunks with very long content."""
        long_content = "a" * 10000  # 10k character content
        chunk = create_test_chunk(content=long_content)

        ids = await chroma_store.add([chunk])
        assert len(ids) == 1

        results = await chroma_store.get(ids)
        assert len(results) == 1
        assert len(results[0].content) == 10000

    @pytest.mark.asyncio
    async def test_metadata_with_special_characters(self, chroma_store: ChromaDBStore):
        """Test adding chunks with special characters in metadata."""
        chunk = create_test_chunk(source_path="/path/with spaces/and-dashes/file.py")

        ids = await chroma_store.add([chunk])
        assert len(ids) == 1

        results = await chroma_store.get(ids)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_empty_embedding(self, chroma_store: ChromaDBStore):
        """Test search with all-zeros embedding (legitimate edge case)."""
        chunk = create_test_chunk(embedding=[0.0] * 384)
        await chroma_store.add([chunk])

        # Query with zeros should still work
        results = await chroma_store.search([0.0] * 384, top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, chroma_store: ChromaDBStore):
        """Test that health_check returns True for a healthy store."""
        is_healthy = await chroma_store.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_delete_by_source_with_nonexistent_source(self, chroma_store: ChromaDBStore):
        """Test delete_by_source with a source path that doesn't exist."""
        # Should not raise an error
        deleted_count = await chroma_store.delete_by_source("/nonexistent/path")
        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_get_by_source_with_nonexistent_source(self, chroma_store: ChromaDBStore):
        """Test get_by_source with a source path that doesn't exist."""
        results = await chroma_store.get_by_source("/nonexistent/path")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_all_source_paths_empty_store(self, chroma_store: ChromaDBStore):
        """Test get_all_source_paths on an empty store."""
        paths = await chroma_store.get_all_source_paths()
        assert paths == []

    @pytest.mark.asyncio
    async def test_unicode_content(self, chroma_store: ChromaDBStore):
        """Test adding chunks with unicode content."""
        unicode_content = "测试内容 🚀 Тестовое содержание"
        chunk = create_test_chunk(content=unicode_content)

        ids = await chroma_store.add([chunk])
        assert len(ids) == 1

        results = await chroma_store.get(ids)
        assert len(results) == 1
        assert results[0].content == unicode_content


class TestChromaDBStoreMetadataFiltering:
    """Tests for metadata filtering."""

    def test_filter_metadata_primitives(self, chroma_store: ChromaDBStore):
        """Test that primitive types are preserved."""
        metadata = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
        }

        filtered = chroma_store._filter_metadata(metadata)

        assert filtered["string"] == "value"
        assert filtered["integer"] == 42
        assert filtered["float"] == 3.14
        assert filtered["boolean"] is True

    def test_filter_metadata_nested_dict(self, chroma_store: ChromaDBStore):
        """Test that nested dicts are flattened."""
        metadata = {
            "nested": {"key1": "value1", "key2": 42},
        }

        filtered = chroma_store._filter_metadata(metadata)

        assert filtered["nested.key1"] == "value1"
        assert filtered["nested.key2"] == 42

    def test_filter_metadata_list_of_strings(self, chroma_store: ChromaDBStore):
        """Test that lists of strings are converted to comma-separated."""
        metadata = {
            "tags": ["python", "async", "fastapi"],
        }

        filtered = chroma_store._filter_metadata(metadata)

        assert filtered["tags"] == "python,async,fastapi"

    def test_filter_metadata_excludes_complex(self, chroma_store: ChromaDBStore):
        """Test that complex objects are excluded."""
        metadata = {
            "string": "value",
            "object": {"nested": "dict"},
            "list_of_ints": [1, 2, 3],  # Non-string lists should be excluded
        }

        filtered = chroma_store._filter_metadata(metadata)

        assert "string" in filtered
        # Complex objects in nested dicts are flattened if primitives
        assert "object.nested" in filtered
        # Non-string lists should be excluded
        assert "list_of_ints" not in filtered


class TestChromaDBStoreProtocolCompliance:
    """Tests that ChromaDBStore implements VectorStore protocol."""

    def test_isinstance_vector_store(self):
        """Test that ChromaDBStore is recognized as VectorStore."""
        from rag_pipeline.core.base_vector_store import VectorStore

        store = ChromaDBStore()

        assert isinstance(store, VectorStore)

    @pytest.mark.asyncio
    async def test_all_protocol_methods_implemented(self, chroma_store: ChromaDBStore):
        """Test that all protocol methods are callable."""
        chunk = create_test_chunk()

        # Test all required methods exist and are callable
        assert callable(chroma_store.add)
        assert callable(chroma_store.search)
        assert callable(chroma_store.delete)
        assert callable(chroma_store.delete_by_document)
        assert callable(chroma_store.delete_by_source)
        assert callable(chroma_store.get)
        assert callable(chroma_store.get_by_source)
        assert callable(chroma_store.get_all_source_paths)
        assert callable(chroma_store.get_stats)
        assert callable(chroma_store.count)
        assert callable(chroma_store.clear)
        assert callable(chroma_store.health_check)

        # Quick smoke test that they work
        await chroma_store.add([chunk])
        await chroma_store.search([0.5] * 384)
        await chroma_store.count()
        await chroma_store.get([chunk.id])
        await chroma_store.get_by_source("/path/to/file.py")
        await chroma_store.get_all_source_paths()
        await chroma_store.get_stats()
        await chroma_store.health_check()
