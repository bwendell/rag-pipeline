"""Contract compliance tests for VectorStore protocol implementations.

This module provides a comprehensive test suite that enforces VectorStore
Protocol compliance across all implementations (InMemoryStore, ChromaDBStore).

The VectorStoreContractTests base class defines tests that any VectorStore
implementation must pass. Concrete test classes inherit from this base and
provide the specific store instance via the store fixture.

Example:
    To test a new vector store implementation, simply inherit from
    VectorStoreContractTests and implement the store fixture:

    class TestMyNewStore(VectorStoreContractTests):
        @pytest_asyncio.fixture
        async def store(self):
            store = MyNewStore()
            yield store
            await store.clear()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, Metadata, SearchResult, SourceType
from rag_pipeline.vector_stores import ChromaDBStore, InMemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Chunk Factory Functions
# =============================================================================


def create_test_chunk(
    chunk_id: str = "chunk-1",
    content: str = "test content",
    document_id: str = "doc-1",
    embedding: list[float] | None = None,
    source_path: str = "/path/to/file.py",
    language: str = "python",
    chunk_index: int = 0,
    dimension: int = 384,
) -> Chunk:
    """Create a test chunk with the given parameters.

    Args:
        chunk_id: Unique identifier for the chunk.
        content: Text content of the chunk.
        document_id: Parent document ID.
        embedding: Optional embedding vector. If None, creates random vector.
        source_path: Source file path.
        language: Programming language.
        chunk_index: Position within document.
        dimension: Embedding dimension if creating random embedding.

    Returns:
        Chunk instance with all fields populated.
    """
    if embedding is None:
        embedding = [float(x) for x in np.random.rand(dimension).astype(np.float32)]

    return Chunk(
        id=chunk_id,
        content=content,
        document_id=document_id,
        metadata=Metadata(
            source_path=source_path,
            source_type=SourceType.FILESYSTEM,
            language=language,
            file_extension=".py",
        ),
        embedding=embedding,
        chunk_index=chunk_index,
        start_index=chunk_index * 100,
        end_index=chunk_index * 100 + len(content),
    )


def create_test_chunks(
    count: int,
    document_id: str = "doc-1",
    dimension: int = 384,
    base_content: str = "Content",
) -> list[Chunk]:
    """Create multiple test chunks.

    Args:
        count: Number of chunks to create.
        document_id: Document ID for all chunks.
        dimension: Embedding dimension.
        base_content: Base content string (will be suffixed with index).

    Returns:
        List of Chunk instances.
    """
    chunks = []
    for i in range(count):
        embedding = [float(x) for x in np.random.rand(dimension).astype(np.float32)]
        chunk = Chunk(
            id=f"chunk-{i}",
            content=f"{base_content} {i}",
            document_id=document_id,
            metadata=Metadata(
                source_path=f"/path/to/file{i}.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
            embedding=embedding,
            chunk_index=i,
            start_index=i * 100,
            end_index=(i + 1) * 100,
        )
        chunks.append(chunk)
    return chunks


# =============================================================================
# VectorStoreContractTests Base Class
# =============================================================================


class VectorStoreContractTests:
    """Base class for VectorStore protocol compliance tests.

    This class defines tests that any VectorStore implementation must pass.
    Subclasses must provide a 'store' fixture that yields a VectorStore instance.

    Usage:
        class TestMyStore(VectorStoreContractTests):
            @pytest_asyncio.fixture
            async def store(self):
                store = MyStore()
                yield store
                await store.clear()
    """

    # -------------------------------------------------------------------------
    # add() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_add_returns_ids_in_order(self, store: Any):
        """add() must return IDs in the same order as input chunks."""
        chunks = [
            create_test_chunk(chunk_id="chunk-a", chunk_index=0),
            create_test_chunk(chunk_id="chunk-b", chunk_index=1),
            create_test_chunk(chunk_id="chunk-c", chunk_index=2),
        ]

        ids = await store.add(chunks)

        assert ids == ["chunk-a", "chunk-b", "chunk-c"]

    @pytest.mark.anyio
    async def test_add_overwrites_existing_ids(self, store: Any):
        """add() must overwrite existing chunks with the same ID."""
        chunk1 = create_test_chunk(chunk_id="same-id", content="Original content")
        await store.add([chunk1])

        chunk2 = create_test_chunk(chunk_id="same-id", content="Updated content")
        ids = await store.add([chunk2])

        # Should return same ID
        assert ids == ["same-id"]

        # Verify content was updated
        retrieved = await store.get(["same-id"])
        assert len(retrieved) == 1
        assert retrieved[0].content == "Updated content"

    @pytest.mark.anyio
    async def test_add_handles_empty_list(self, store: Any):
        """add() must handle empty list and return empty list."""
        ids = await store.add([])

        assert ids == []

    @pytest.mark.anyio
    async def test_add_validates_embeddings_exist(self, store: Any):
        """add() must raise ValueError for chunks without embeddings."""
        chunk = Chunk(
            id="no-embedding",
            content="Content without embedding",
            document_id="doc-1",
            embedding=None,
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await store.add([chunk])

    @pytest.mark.anyio
    async def test_add_validates_embeddings_not_empty(self, store: Any):
        """add() must raise ValueError for chunks with empty embeddings."""
        chunk = Chunk(
            id="empty-embedding",
            content="Content with empty embedding",
            document_id="doc-1",
            embedding=[],
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await store.add([chunk])

    # -------------------------------------------------------------------------
    # search() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_search_returns_ranked_results(self, store: Any):
        """search() must return results ordered by score (highest first)."""
        # Create chunks with different embeddings
        chunks = [
            create_test_chunk(chunk_id="chunk-1", chunk_index=0),
            create_test_chunk(chunk_id="chunk-2", chunk_index=1),
            create_test_chunk(chunk_id="chunk-3", chunk_index=2),
        ]
        await store.add(chunks)

        # Search with random query
        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=5)

        # Results should be ordered by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.anyio
    async def test_search_returns_search_result_objects(self, store: Any):
        """search() must return list of SearchResult objects."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=5)

        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.anyio
    async def test_search_results_have_rank(self, store: Any):
        """search() results must have correct rank (1-indexed)."""
        chunks = create_test_chunks(5)
        await store.add(chunks)

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=3)

        for i, result in enumerate(results):
            assert result.rank == i + 1

    @pytest.mark.anyio
    async def test_search_respects_top_k(self, store: Any):
        """search() must respect top_k parameter."""
        chunks = create_test_chunks(10)
        await store.add(chunks)

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=3)

        assert len(results) <= 3

    @pytest.mark.anyio
    async def test_search_top_k_zero_returns_empty(self, store: Any):
        """search() with top_k=0 must return empty list."""
        chunks = create_test_chunks(5)
        await store.add(chunks)

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=0)

        assert results == []

    @pytest.mark.anyio
    async def test_search_top_k_exceeds_count_returns_all(self, store: Any):
        """search() with top_k > chunk count must return all chunks."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=100)

        assert len(results) == 3

    @pytest.mark.anyio
    async def test_search_respects_filters(self, store: Any):
        """search() must respect metadata filters."""
        # Add chunks with different metadata
        py_chunk = create_test_chunk(
            chunk_id="py-chunk",
            content="Python code",
            language="python",
        )
        js_chunk = create_test_chunk(
            chunk_id="js-chunk",
            content="JavaScript code",
            source_path="/path/to/file.js",
            language="javascript",
        )
        await store.add([py_chunk, js_chunk])

        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(
            query_embedding,
            top_k=5,
            filters={"language": "python"},
        )

        # All results should match the filter
        for result in results:
            assert result.chunk.metadata.language == "python"

    @pytest.mark.anyio
    async def test_search_empty_store_returns_empty(self, store: Any):
        """search() on empty store must return empty list."""
        query_embedding = [float(x) for x in np.random.rand(384).astype(np.float32)]
        results = await store.search(query_embedding, top_k=5)

        assert results == []

    # -------------------------------------------------------------------------
    # delete() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_delete_returns_count(self, store: Any):
        """delete() must return number of chunks actually deleted."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        deleted = await store.delete(["chunk-0", "chunk-1"])

        assert deleted == 2

    @pytest.mark.anyio
    async def test_delete_ignores_missing_ids(self, store: Any):
        """delete() must ignore non-existent IDs and return actual count."""
        chunks = create_test_chunks(2)
        await store.add(chunks)

        deleted = await store.delete(["chunk-0", "nonexistent", "chunk-1", "also-missing"])

        assert deleted == 2

    @pytest.mark.anyio
    async def test_delete_mixed_valid_invalid(self, store: Any):
        """delete() must handle mix of valid and invalid IDs."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        deleted = await store.delete(["chunk-0", "missing-1", "chunk-2", "missing-2"])

        assert deleted == 2

    @pytest.mark.anyio
    async def test_delete_is_idempotent(self, store: Any):
        """delete() must be idempotent (deleting same ID twice is safe)."""
        chunks = create_test_chunks(1)
        await store.add(chunks)

        # First delete
        deleted1 = await store.delete(["chunk-0"])
        assert deleted1 == 1

        # Second delete of same ID
        deleted2 = await store.delete(["chunk-0"])
        assert deleted2 == 0

    @pytest.mark.anyio
    async def test_delete_empty_list_returns_zero(self, store: Any):
        """delete() with empty list must return 0."""
        deleted = await store.delete([])

        assert deleted == 0

    @pytest.mark.anyio
    async def test_delete_actually_removes_chunks(self, store: Any):
        """delete() must actually remove chunks from store."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        await store.delete(["chunk-0"])

        # Verify chunk is gone
        retrieved = await store.get(["chunk-0"])
        assert len(retrieved) == 0

        # Verify count decreased
        count = await store.count()
        assert count == 2

    # -------------------------------------------------------------------------
    # delete_by_document() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_delete_by_document_removes_all_chunks(self, store: Any):
        """delete_by_document() must remove all chunks for a document."""
        # Create chunks for document
        chunks = [
            create_test_chunk(chunk_id="doc-chunk-1", document_id="doc-to-delete", chunk_index=0),
            create_test_chunk(chunk_id="doc-chunk-2", document_id="doc-to-delete", chunk_index=1),
            create_test_chunk(chunk_id="doc-chunk-3", document_id="doc-to-delete", chunk_index=2),
        ]
        await store.add(chunks)

        deleted = await store.delete_by_document("doc-to-delete")

        assert deleted == 3

        # Verify all chunks are gone
        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_delete_by_document_handles_missing(self, store: Any):
        """delete_by_document() must return 0 for non-existent document."""
        deleted = await store.delete_by_document("nonexistent-doc")

        assert deleted == 0

    @pytest.mark.anyio
    async def test_delete_by_document_only_affects_target(self, store: Any):
        """delete_by_document() must only affect specified document."""
        chunks_doc1 = [
            create_test_chunk(chunk_id="d1-c1", document_id="doc-1", chunk_index=0),
            create_test_chunk(chunk_id="d1-c2", document_id="doc-1", chunk_index=1),
        ]
        chunks_doc2 = [
            create_test_chunk(chunk_id="d2-c1", document_id="doc-2", chunk_index=0),
            create_test_chunk(chunk_id="d2-c2", document_id="doc-2", chunk_index=1),
        ]
        await store.add(chunks_doc1 + chunks_doc2)

        deleted = await store.delete_by_document("doc-1")

        assert deleted == 2

        # Verify doc-2 chunks remain
        count = await store.count()
        assert count == 2

        retrieved = await store.get(["d2-c1", "d2-c2"])
        assert len(retrieved) == 2

    # -------------------------------------------------------------------------
    # get() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_get_returns_matching_chunks(self, store: Any):
        """get() must return chunks for requested IDs."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        retrieved = await store.get(["chunk-0", "chunk-2"])

        assert len(retrieved) == 2
        retrieved_ids = {c.id for c in retrieved}
        assert retrieved_ids == {"chunk-0", "chunk-2"}

    @pytest.mark.anyio
    async def test_get_skips_missing(self, store: Any):
        """get() must skip missing IDs without error."""
        chunks = create_test_chunks(2)
        await store.add(chunks)

        retrieved = await store.get(["chunk-0", "missing", "chunk-1", "also-missing"])

        assert len(retrieved) == 2

    @pytest.mark.anyio
    async def test_get_preserves_embeddings(self, store: Any):
        """get() must preserve embeddings in returned chunks."""
        # Create chunk with specific embedding
        embedding = [1.0, 2.0, 3.0] + [0.0] * 381
        chunk = create_test_chunk(chunk_id="test-chunk", embedding=embedding)
        await store.add([chunk])

        retrieved = await store.get(["test-chunk"])

        assert len(retrieved) == 1
        assert retrieved[0].embedding is not None
        assert len(retrieved[0].embedding) == 384

    @pytest.mark.anyio
    async def test_get_empty_list_returns_empty(self, store: Any):
        """get() with empty list must return empty list."""
        retrieved = await store.get([])

        assert retrieved == []

    @pytest.mark.anyio
    async def test_get_all_missing_returns_empty(self, store: Any):
        """get() with all missing IDs must return empty list."""
        retrieved = await store.get(["missing-1", "missing-2"])

        assert retrieved == []

    # -------------------------------------------------------------------------
    # get_stats() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_get_stats_counts_chunks_correctly(self, store: Any):
        """get_stats() must count chunks correctly."""
        chunks = create_test_chunks(5)
        await store.add(chunks)

        stats = await store.get_stats()

        assert stats["total_chunks"] == 5

    @pytest.mark.anyio
    async def test_get_stats_counts_documents_correctly(self, store: Any):
        """get_stats() must count unique documents correctly."""
        chunks = [
            create_test_chunk(chunk_id="c1", document_id="doc-a"),
            create_test_chunk(chunk_id="c2", document_id="doc-a"),
            create_test_chunk(chunk_id="c3", document_id="doc-b"),
            create_test_chunk(chunk_id="c4", document_id="doc-c"),
        ]
        await store.add(chunks)

        stats = await store.get_stats()

        assert stats["total_documents"] == 3

    @pytest.mark.anyio
    async def test_get_stats_reflects_deletions(self, store: Any):
        """get_stats() must reflect deletions."""
        chunks = create_test_chunks(5)
        await store.add(chunks)

        # Delete some chunks
        await store.delete(["chunk-0", "chunk-1"])

        stats = await store.get_stats()

        assert stats["total_chunks"] == 3

    @pytest.mark.anyio
    async def test_get_stats_empty_store(self, store: Any):
        """get_stats() on empty store must have zeros."""
        stats = await store.get_stats()

        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0

    @pytest.mark.anyio
    async def test_get_stats_has_required_keys(self, store: Any):
        """get_stats() must have required keys."""
        chunks = create_test_chunks(1)
        await store.add(chunks)

        stats = await store.get_stats()

        required_keys = {"store_type", "total_chunks", "total_documents", "embedding_dimension"}
        assert required_keys.issubset(stats.keys())

    # -------------------------------------------------------------------------
    # clear() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_clear_removes_all_data(self, store: Any):
        """clear() must remove all chunks from store."""
        chunks = create_test_chunks(5)
        await store.add(chunks)

        await store.clear()

        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_clear_allows_re_adding(self, store: Any):
        """clear() must allow re-adding chunks after clearing."""
        chunks = create_test_chunks(3)
        await store.add(chunks)
        await store.clear()

        # Should be able to add again
        new_chunks = create_test_chunks(2)
        ids = await store.add(new_chunks)

        assert len(ids) == 2
        count = await store.count()
        assert count == 2

    @pytest.mark.anyio
    async def test_clear_empty_store_is_safe(self, store: Any):
        """clear() on empty store must not raise error."""
        # Should not raise
        await store.clear()

        count = await store.count()
        assert count == 0

    # -------------------------------------------------------------------------
    # health_check() Contract Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_health_check_returns_boolean(self, store: Any):
        """health_check() must return a boolean."""
        result = await store.health_check()

        assert isinstance(result, bool)

    @pytest.mark.anyio
    async def test_health_check_returns_true_when_healthy(self, store: Any):
        """health_check() must return True when store is operational."""
        result = await store.health_check()

        assert result is True

    @pytest.mark.anyio
    async def test_health_check_with_data(self, store: Any):
        """health_check() must work with populated store."""
        chunks = create_test_chunks(3)
        await store.add(chunks)

        result = await store.health_check()

        assert result is True

    # -------------------------------------------------------------------------
    # Additional Protocol Method Tests
    # -------------------------------------------------------------------------

    @pytest.mark.anyio
    async def test_count_returns_zero_for_empty(self, store: Any):
        """count() must return 0 for empty store."""
        count = await store.count()

        assert count == 0

    @pytest.mark.anyio
    async def test_count_returns_correct_number(self, store: Any):
        """count() must return correct chunk count."""
        chunks = create_test_chunks(7)
        await store.add(chunks)

        count = await store.count()

        assert count == 7

    @pytest.mark.anyio
    async def test_delete_by_source_removes_matching(self, store: Any):
        """delete_by_source() must remove chunks with matching source_path."""
        chunks = [
            create_test_chunk(chunk_id="s1-c1", source_path="/path/file1.py"),
            create_test_chunk(chunk_id="s1-c2", source_path="/path/file1.py"),
            create_test_chunk(chunk_id="s2-c1", source_path="/path/file2.py"),
        ]
        await store.add(chunks)

        deleted = await store.delete_by_source("/path/file1.py")

        assert deleted == 2

        # Verify remaining
        count = await store.count()
        assert count == 1

    @pytest.mark.anyio
    async def test_get_by_source_returns_ordered_chunks(self, store: Any):
        """get_by_source() must return chunks ordered by chunk_index."""
        chunks = [
            create_test_chunk(chunk_id="c3", chunk_index=3, source_path="/path/file.py"),
            create_test_chunk(chunk_id="c1", chunk_index=1, source_path="/path/file.py"),
            create_test_chunk(chunk_id="c2", chunk_index=2, source_path="/path/file.py"),
        ]
        await store.add(chunks)

        retrieved = await store.get_by_source("/path/file.py")

        assert len(retrieved) == 3
        assert retrieved[0].chunk_index == 1
        assert retrieved[1].chunk_index == 2
        assert retrieved[2].chunk_index == 3

    @pytest.mark.anyio
    async def test_get_all_source_paths_returns_unique_sorted(self, store: Any):
        """get_all_source_paths() must return unique sorted paths."""
        chunks = [
            create_test_chunk(chunk_id="c1", source_path="/path/b.py"),
            create_test_chunk(chunk_id="c2", source_path="/path/a.py"),
            create_test_chunk(chunk_id="c3", source_path="/path/b.py"),  # Duplicate
        ]
        await store.add(chunks)

        paths = await store.get_all_source_paths()

        assert paths == ["/path/a.py", "/path/b.py"]


# =============================================================================
# Concrete Test Classes
# =============================================================================


class TestInMemoryStoreContract(VectorStoreContractTests):
    """Contract tests for InMemoryStore implementation."""

    @pytest.fixture
    def store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance for each test.

        Yields:
            InMemoryStore: A new empty vector store instance.
        """
        return InMemoryStore()


class TestChromaDBStoreContract(VectorStoreContractTests):
    """Contract tests for ChromaDBStore implementation."""

    @pytest.fixture
    async def store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance for each test.

        Uses a temporary directory to avoid interfering with existing data.
        Cleans up by clearing all data after each test.

        Yields:
            ChromaDBStore: A new ChromaDBStore with temporary storage.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="contract_test_collection",
            )
            yield store
            # Cleanup: clear all data
            try:
                await store.clear()
            except Exception:
                pass  # Ignore cleanup errors
