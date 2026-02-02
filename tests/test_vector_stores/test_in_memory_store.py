"""Comprehensive unit tests for InMemoryStore.

Tests all 12 VectorStore Protocol methods with comprehensive coverage including:
- Basic functionality (happy path for each method)
- Edge cases (empty inputs, missing IDs, no results)
- Error handling (missing embeddings raises ValueError)
- Search with filters (metadata filtering)
- Ordering (search results by score, get_by_source by chunk_index)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, Metadata, SearchResult, SourceType
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore


# =============================================================================
# TestInMemoryStoreAdd - Test add() method
# =============================================================================


class TestInMemoryStoreAdd:
    """Tests for add() method - adding chunks to store."""

    @pytest.mark.anyio
    async def test_add_single_chunk(self, sample_chunk: Chunk):
        """Test adding a single chunk returns its ID."""
        store = InMemoryStore()
        ids = await store.add([sample_chunk])

        assert len(ids) == 1
        assert ids[0] == sample_chunk.id

    @pytest.mark.anyio
    async def test_add_multiple_chunks(self, sample_chunks: list[Chunk]):
        """Test adding multiple chunks returns all IDs in order."""
        store = InMemoryStore()
        ids = await store.add(sample_chunks)

        assert len(ids) == len(sample_chunks)
        for i, chunk in enumerate(sample_chunks):
            assert ids[i] == chunk.id

    @pytest.mark.anyio
    async def test_add_chunks_stored_in_store(self, sample_chunks: list[Chunk]):
        """Test chunks are actually stored after add()."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        for chunk in sample_chunks:
            assert chunk.id in store._chunks
            assert store._chunks[chunk.id] == chunk

    @pytest.mark.anyio
    async def test_add_chunk_without_embedding_raises_valueerror(self):
        """Test adding chunk without embedding raises ValueError."""
        store = InMemoryStore()
        chunk = Chunk(
            id="no-embedding",
            content="Content without embedding",
            document_id="doc-1",
            embedding=None,  # No embedding
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await store.add([chunk])

    @pytest.mark.anyio
    async def test_add_chunk_with_empty_embedding_raises_valueerror(self):
        """Test adding chunk with empty embedding list raises ValueError."""
        store = InMemoryStore()
        chunk = Chunk(
            id="empty-embedding",
            content="Content with empty embedding",
            document_id="doc-1",
            embedding=[],  # Empty embedding
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await store.add([chunk])

    @pytest.mark.anyio
    async def test_add_mixed_valid_and_invalid_chunks(self, sample_chunk: Chunk):
        """Test adding mixed valid/invalid chunks fails on first invalid."""
        store = InMemoryStore()
        invalid_chunk = Chunk(
            id="invalid",
            content="No embedding",
            document_id="doc-1",
            embedding=None,
        )

        # Should fail on invalid chunk
        with pytest.raises(ValueError):
            await store.add([sample_chunk, invalid_chunk])

    @pytest.mark.anyio
    async def test_add_duplicate_ids_overwrites(self, sample_chunk: Chunk):
        """Test adding chunk with duplicate ID overwrites previous."""
        store = InMemoryStore()

        # Add first chunk
        await store.add([sample_chunk])
        count1 = await store.count()

        # Add another chunk with same ID but different content
        duplicate = Chunk(
            id=sample_chunk.id,
            content="Different content",
            document_id="doc-2",
            embedding=sample_chunk.embedding,
        )
        await store.add([duplicate])
        count2 = await store.count()

        # Count should remain same (overwrite)
        assert count1 == count2 == 1
        # But content should be updated
        retrieved = await store.get([sample_chunk.id])
        assert retrieved[0].content == "Different content"

    @pytest.mark.anyio
    async def test_add_empty_list(self):
        """Test adding empty list returns empty list."""
        store = InMemoryStore()
        ids = await store.add([])
        assert ids == []


# =============================================================================
# TestInMemoryStoreSearch - Test search() method
# =============================================================================


class TestInMemoryStoreSearch:
    """Tests for search() method - searching with embeddings."""

    @pytest.mark.anyio
    async def test_search_returns_search_results(self, populated_store, sample_embedding):
        """Test search returns list of SearchResult objects."""
        results = await populated_store.search(sample_embedding, top_k=5)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.anyio
    async def test_search_results_ordered_by_score(self, populated_store, sample_embedding):
        """Test search results are ordered by score (highest first)."""
        results = await populated_store.search(sample_embedding, top_k=5)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.anyio
    async def test_search_results_have_correct_rank(self, populated_store, sample_embedding):
        """Test search results have correct rank values (1-indexed)."""
        results = await populated_store.search(sample_embedding, top_k=5)

        for i, result in enumerate(results):
            assert result.rank == i + 1

    @pytest.mark.anyio
    async def test_search_respects_top_k(self, populated_store, sample_embedding):
        """Test search respects top_k parameter."""
        results_3 = await populated_store.search(sample_embedding, top_k=3)
        results_5 = await populated_store.search(sample_embedding, top_k=5)

        assert len(results_3) <= 3
        assert len(results_5) <= 5

    @pytest.mark.anyio
    async def test_search_empty_store_returns_empty_list(self, sample_embedding):
        """Test searching empty store returns empty list."""
        store = InMemoryStore()
        results = await store.search(sample_embedding, top_k=5)
        assert results == []

    @pytest.mark.anyio
    async def test_search_cosine_similarity_exact_match(self, embedding_dimension):
        """Test cosine similarity finds exact embeddings with score 1.0."""
        store = InMemoryStore()

        # Create embedding and chunk with same embedding
        embedding = list(np.ones(embedding_dimension).astype(np.float32))
        chunk = Chunk(
            id="chunk-1",
            content="Test content",
            document_id="doc-1",
            embedding=embedding,
        )

        await store.add([chunk])
        results = await store.search(embedding, top_k=1)

        assert len(results) == 1
        assert results[0].score > 0.99  # Should be ~1.0 (cosine similarity)
        assert results[0].chunk.id == "chunk-1"

    @pytest.mark.anyio
    async def test_search_with_filter_metadata(self, populated_store, sample_embedding):
        """Test search with metadata filters."""
        filters = {"language": "python"}
        results = await populated_store.search(sample_embedding, top_k=5, filters=filters)

        # All results should match filter
        for result in results:
            assert result.chunk.metadata.language == "python"

    @pytest.mark.anyio
    async def test_search_with_filter_source_path(self, populated_store, sample_embedding):
        """Test search filtering by source_path."""
        filters = {"source_path": "/path/to/file1.py"}
        results = await populated_store.search(sample_embedding, top_k=5, filters=filters)

        for result in results:
            assert result.chunk.metadata.source_path == "/path/to/file1.py"

    @pytest.mark.anyio
    async def test_search_with_filter_list_value(self, populated_store, sample_embedding):
        """Test search with filter using list of acceptable values."""
        filters = {"language": ["python", "javascript"]}
        results = await populated_store.search(sample_embedding, top_k=5, filters=filters)

        for result in results:
            assert result.chunk.metadata.language in ["python", "javascript"]

    @pytest.mark.anyio
    async def test_search_with_nonmatching_filter(self, populated_store, sample_embedding):
        """Test search with filter that matches no chunks returns empty."""
        filters = {"language": "rust"}  # No chunks with rust language
        results = await populated_store.search(sample_embedding, top_k=5, filters=filters)

        assert results == []

    @pytest.mark.anyio
    async def test_search_no_filters_param(self, populated_store, sample_embedding):
        """Test search without filters parameter works."""
        results = await populated_store.search(sample_embedding, top_k=5)
        assert isinstance(results, list)

    @pytest.mark.anyio
    async def test_search_with_none_filters(self, populated_store, sample_embedding):
        """Test search with filters=None works."""
        results = await populated_store.search(sample_embedding, top_k=5, filters=None)
        assert isinstance(results, list)


# =============================================================================
# TestInMemoryStoreDelete - Test delete() method
# =============================================================================


class TestInMemoryStoreDelete:
    """Tests for delete() method - deleting chunks by ID."""

    @pytest.mark.anyio
    async def test_delete_single_chunk(self, sample_chunks):
        """Test deleting a single chunk by ID."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        chunk_id = sample_chunks[0].id
        deleted = await store.delete([chunk_id])

        assert deleted == 1
        assert chunk_id not in store._chunks

    @pytest.mark.anyio
    async def test_delete_multiple_chunks(self, sample_chunks):
        """Test deleting multiple chunks."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        ids_to_delete = [c.id for c in sample_chunks[:3]]
        deleted = await store.delete(ids_to_delete)

        assert deleted == 3
        for chunk_id in ids_to_delete:
            assert chunk_id not in store._chunks

    @pytest.mark.anyio
    async def test_delete_nonexistent_id_returns_zero(self, sample_chunks):
        """Test deleting nonexistent ID returns 0."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        deleted = await store.delete(["nonexistent-id"])
        assert deleted == 0

    @pytest.mark.anyio
    async def test_delete_mixed_existing_nonexisting_ids(self, sample_chunks):
        """Test deleting mix of existing and nonexisting IDs."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        ids_to_delete = [sample_chunks[0].id, "nonexistent", sample_chunks[1].id]
        deleted = await store.delete(ids_to_delete)

        assert deleted == 2  # Only 2 exist

    @pytest.mark.anyio
    async def test_delete_empty_list(self, sample_chunks):
        """Test deleting empty list of IDs."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        deleted = await store.delete([])
        assert deleted == 0

    @pytest.mark.anyio
    async def test_delete_updates_count(self, sample_chunks):
        """Test delete reduces store count."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        count_before = await store.count()
        await store.delete([sample_chunks[0].id])
        count_after = await store.count()

        assert count_after == count_before - 1


# =============================================================================
# TestInMemoryStoreDeleteByDocument - Test delete_by_document() method
# =============================================================================


class TestInMemoryStoreDeleteByDocument:
    """Tests for delete_by_document() method - deleting by document_id."""

    @pytest.mark.anyio
    async def test_delete_by_document_single_document(self, sample_chunks):
        """Test deleting all chunks for a document."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        # sample_chunks[0] is from doc-1
        deleted = await store.delete_by_document("doc-1")

        assert deleted > 0
        # Verify chunks from doc-1 are gone
        remaining = await store.count()
        assert remaining == len(sample_chunks) - deleted

    @pytest.mark.anyio
    async def test_delete_by_document_all_chunks_removed(self, sample_chunk):
        """Test all chunks for document are removed."""
        store = InMemoryStore()

        # Add multiple chunks with same document_id
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="test-doc",
                embedding=sample_chunk.embedding,
            )
            for i in range(3)
        ]
        await store.add(chunks)

        deleted = await store.delete_by_document("test-doc")
        assert deleted == 3

        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_delete_by_document_nonexistent_returns_zero(self, sample_chunks):
        """Test deleting nonexistent document returns 0."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        deleted = await store.delete_by_document("nonexistent-doc")
        assert deleted == 0

    @pytest.mark.anyio
    async def test_delete_by_document_partial_removal(self, sample_chunks):
        """Test only specified document chunks are deleted."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        initial_count = await store.count()
        deleted = await store.delete_by_document("doc-1")
        final_count = await store.count()

        assert final_count == initial_count - deleted
        # Verify doc-1 chunks don't exist
        for chunk in sample_chunks:
            if chunk.document_id == "doc-1":
                assert chunk.id not in store._chunks


# =============================================================================
# TestInMemoryStoreDeleteBySource - Test delete_by_source() method
# =============================================================================


class TestInMemoryStoreDeleteBySource:
    """Tests for delete_by_source() method - deleting by source_path."""

    @pytest.mark.anyio
    async def test_delete_by_source_removes_chunks(self, sample_chunks):
        """Test deleting chunks by source_path."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        source_path = sample_chunks[0].metadata.source_path
        deleted = await store.delete_by_source(source_path)

        assert deleted > 0
        # Verify chunks from source are gone
        remaining = await store.count()
        assert remaining == len(sample_chunks) - deleted

    @pytest.mark.anyio
    async def test_delete_by_source_all_from_source_removed(self, sample_chunk):
        """Test all chunks from source are removed."""
        store = InMemoryStore()

        source = "/path/to/file.py"
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id=f"doc-{i}",
                metadata=Metadata(source_path=source),
                embedding=sample_chunk.embedding,
            )
            for i in range(3)
        ]
        await store.add(chunks)

        deleted = await store.delete_by_source(source)
        assert deleted == 3

    @pytest.mark.anyio
    async def test_delete_by_source_nonexistent_returns_zero(self, sample_chunks):
        """Test deleting nonexistent source returns 0."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        deleted = await store.delete_by_source("/nonexistent/path.py")
        assert deleted == 0


# =============================================================================
# TestInMemoryStoreGet - Test get() method
# =============================================================================


class TestInMemoryStoreGet:
    """Tests for get() method - retrieving chunks by ID."""

    @pytest.mark.anyio
    async def test_get_single_chunk(self, sample_chunks):
        """Test retrieving a single chunk by ID."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        chunk_id = sample_chunks[0].id
        retrieved = await store.get([chunk_id])

        assert len(retrieved) == 1
        assert retrieved[0].id == chunk_id
        assert retrieved[0].content == sample_chunks[0].content

    @pytest.mark.anyio
    async def test_get_multiple_chunks(self, sample_chunks):
        """Test retrieving multiple chunks."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        ids = [c.id for c in sample_chunks[:3]]
        retrieved = await store.get(ids)

        assert len(retrieved) == 3
        retrieved_ids = [c.id for c in retrieved]
        assert set(retrieved_ids) == set(ids)

    @pytest.mark.anyio
    async def test_get_nonexistent_id_skipped(self, sample_chunks):
        """Test nonexistent IDs are skipped."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        ids = [sample_chunks[0].id, "nonexistent"]
        retrieved = await store.get(ids)

        assert len(retrieved) == 1
        assert retrieved[0].id == sample_chunks[0].id

    @pytest.mark.anyio
    async def test_get_all_nonexistent_returns_empty(self, sample_chunks):
        """Test getting all nonexistent IDs returns empty list."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        retrieved = await store.get(["fake1", "fake2", "fake3"])
        assert retrieved == []

    @pytest.mark.anyio
    async def test_get_empty_list(self, sample_chunks):
        """Test getting empty list of IDs returns empty list."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        retrieved = await store.get([])
        assert retrieved == []

    @pytest.mark.anyio
    async def test_get_preserves_embedding(self, sample_chunks):
        """Test retrieved chunks have embeddings."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        retrieved = await store.get([sample_chunks[0].id])

        assert len(retrieved) == 1
        assert retrieved[0].embedding is not None
        assert len(retrieved[0].embedding) > 0


# =============================================================================
# TestInMemoryStoreGetBySource - Test get_by_source() method
# =============================================================================


class TestInMemoryStoreGetBySource:
    """Tests for get_by_source() method - retrieving chunks by source_path."""

    @pytest.mark.anyio
    async def test_get_by_source_returns_chunks(self, sample_chunks):
        """Test getting chunks by source_path."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        source_path = sample_chunks[0].metadata.source_path
        retrieved = await store.get_by_source(source_path)

        assert len(retrieved) > 0
        for chunk in retrieved:
            assert chunk.metadata.source_path == source_path

    @pytest.mark.anyio
    async def test_get_by_source_ordered_by_chunk_index(self, sample_chunk):
        """Test chunks are returned ordered by chunk_index."""
        store = InMemoryStore()

        source = "/path/to/file.py"
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(source_path=source),
                embedding=sample_chunk.embedding,
                chunk_index=i * 10,  # Non-sequential to test ordering
            )
            for i in [3, 1, 2, 0, 4]
        ]
        await store.add(chunks)

        retrieved = await store.get_by_source(source)

        # Should be ordered by chunk_index
        for i in range(len(retrieved) - 1):
            assert retrieved[i].chunk_index <= retrieved[i + 1].chunk_index

    @pytest.mark.anyio
    async def test_get_by_source_nonexistent_returns_empty(self, sample_chunks):
        """Test getting nonexistent source returns empty list."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        retrieved = await store.get_by_source("/nonexistent/path.py")
        assert retrieved == []

    @pytest.mark.anyio
    async def test_get_by_source_respects_limit(self, sample_chunk):
        """Test get_by_source respects limit parameter."""
        store = InMemoryStore()

        source = "/path/to/file.py"
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(source_path=source),
                embedding=sample_chunk.embedding,
                chunk_index=i,
            )
            for i in range(10)
        ]
        await store.add(chunks)

        retrieved = await store.get_by_source(source, limit=3)
        assert len(retrieved) == 3

    @pytest.mark.anyio
    async def test_get_by_source_default_limit(self, sample_chunk):
        """Test get_by_source default limit is 100."""
        store = InMemoryStore()

        source = "/path/to/file.py"
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(source_path=source),
                embedding=sample_chunk.embedding,
                chunk_index=i,
            )
            for i in range(50)
        ]
        await store.add(chunks)

        # Should return all 50 without hitting default limit
        retrieved = await store.get_by_source(source)
        assert len(retrieved) == 50


# =============================================================================
# TestInMemoryStoreGetAllSourcePaths - Test get_all_source_paths() method
# =============================================================================


class TestInMemoryStoreGetAllSourcePaths:
    """Tests for get_all_source_paths() method - getting unique source paths."""

    @pytest.mark.anyio
    async def test_get_all_source_paths_returns_list(self, sample_chunks):
        """Test get_all_source_paths returns list of strings."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        paths = await store.get_all_source_paths()

        assert isinstance(paths, list)
        assert all(isinstance(p, str) for p in paths)

    @pytest.mark.anyio
    async def test_get_all_source_paths_sorted(self, sample_chunks):
        """Test source paths are returned sorted."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        paths = await store.get_all_source_paths()

        assert paths == sorted(paths)

    @pytest.mark.anyio
    async def test_get_all_source_paths_unique(self, sample_chunk):
        """Test only unique paths are returned."""
        store = InMemoryStore()

        # Add multiple chunks with same source path
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(source_path="/path/to/file.py"),
                embedding=sample_chunk.embedding,
            )
            for i in range(3)
        ]
        await store.add(chunks)

        paths = await store.get_all_source_paths()

        assert len(paths) == 1
        assert paths[0] == "/path/to/file.py"

    @pytest.mark.anyio
    async def test_get_all_source_paths_empty_store(self):
        """Test empty store returns empty list."""
        store = InMemoryStore()
        paths = await store.get_all_source_paths()
        assert paths == []

    @pytest.mark.anyio
    async def test_get_all_source_paths_ignores_empty_source_path(self, sample_chunk):
        """Test chunks with empty source_path are ignored."""
        store = InMemoryStore()

        chunk_with_source = Chunk(
            id="chunk-1",
            content="Has source",
            document_id="doc-1",
            metadata=Metadata(source_path="/path/to/file.py"),
            embedding=sample_chunk.embedding,
        )

        chunk_without_source = Chunk(
            id="chunk-2",
            content="No source",
            document_id="doc-1",
            metadata=Metadata(source_path=""),  # Empty source path
            embedding=sample_chunk.embedding,
        )

        await store.add([chunk_with_source, chunk_without_source])
        paths = await store.get_all_source_paths()

        assert len(paths) == 1
        assert paths[0] == "/path/to/file.py"


# =============================================================================
# TestInMemoryStoreGetStats - Test get_stats() method
# =============================================================================


class TestInMemoryStoreGetStats:
    """Tests for get_stats() method - getting store statistics."""

    @pytest.mark.anyio
    async def test_get_stats_returns_dict(self, sample_chunks):
        """Test get_stats returns dictionary with stats."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        assert isinstance(stats, dict)

    @pytest.mark.anyio
    async def test_get_stats_has_required_keys(self, sample_chunks):
        """Test get_stats has required keys."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        required_keys = {
            "store_type",
            "total_chunks",
            "total_documents",
            "embedding_dimension",
            "index_type",
        }
        assert required_keys.issubset(stats.keys())

    @pytest.mark.anyio
    async def test_get_stats_correct_total_chunks(self, sample_chunks):
        """Test total_chunks is correct."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        assert stats["total_chunks"] == len(sample_chunks)

    @pytest.mark.anyio
    async def test_get_stats_correct_total_documents(self, sample_chunks):
        """Test total_documents count is correct."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        # sample_chunks has chunks from multiple documents
        unique_docs = len(set(c.document_id for c in sample_chunks))
        stats = await store.get_stats()

        assert stats["total_documents"] == unique_docs

    @pytest.mark.anyio
    async def test_get_stats_correct_embedding_dimension(self, sample_chunks, embedding_dimension):
        """Test embedding_dimension is correct."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        assert stats["embedding_dimension"] == embedding_dimension

    @pytest.mark.anyio
    async def test_get_stats_empty_store(self):
        """Test get_stats on empty store."""
        store = InMemoryStore()

        stats = await store.get_stats()

        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0
        assert stats["embedding_dimension"] == 0
        assert stats["store_type"] == "in_memory"

    @pytest.mark.anyio
    async def test_get_stats_store_type_is_in_memory(self, sample_chunks):
        """Test store_type is 'in_memory'."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        assert stats["store_type"] == "in_memory"

    @pytest.mark.anyio
    async def test_get_stats_index_type_is_flat(self, sample_chunks):
        """Test index_type is 'flat'."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        stats = await store.get_stats()

        assert stats["index_type"] == "flat"


# =============================================================================
# TestInMemoryStoreCount - Test count() method
# =============================================================================


class TestInMemoryStoreCount:
    """Tests for count() method - getting total chunk count."""

    @pytest.mark.anyio
    async def test_count_empty_store(self):
        """Test count on empty store returns 0."""
        store = InMemoryStore()
        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_count_after_add(self, sample_chunks):
        """Test count returns correct number after add."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        count = await store.count()
        assert count == len(sample_chunks)

    @pytest.mark.anyio
    async def test_count_after_delete(self, sample_chunks):
        """Test count decreases after delete."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        count_before = await store.count()
        await store.delete([sample_chunks[0].id])
        count_after = await store.count()

        assert count_after == count_before - 1

    @pytest.mark.anyio
    async def test_count_after_delete_by_document(self, sample_chunks):
        """Test count decreases after delete_by_document."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        count_before = await store.count()
        await store.delete_by_document("doc-1")
        count_after = await store.count()

        assert count_after < count_before

    @pytest.mark.anyio
    async def test_count_after_clear(self, sample_chunks):
        """Test count is 0 after clear."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        await store.clear()
        count = await store.count()

        assert count == 0


# =============================================================================
# TestInMemoryStoreClear - Test clear() method
# =============================================================================


class TestInMemoryStoreClear:
    """Tests for clear() method - clearing all data."""

    @pytest.mark.anyio
    async def test_clear_removes_all_chunks(self, sample_chunks):
        """Test clear removes all chunks."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        await store.clear()

        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_clear_empty_store(self):
        """Test clear on already empty store."""
        store = InMemoryStore()

        await store.clear()  # Should not raise

        count = await store.count()
        assert count == 0

    @pytest.mark.anyio
    async def test_clear_allows_re_adding(self, sample_chunks):
        """Test chunks can be re-added after clear."""
        store = InMemoryStore()
        await store.add(sample_chunks)
        await store.clear()

        # Should be able to add again
        ids = await store.add([sample_chunks[0]])
        assert len(ids) == 1

    @pytest.mark.anyio
    async def test_clear_removes_from_source_paths(self, sample_chunks):
        """Test clear removes all source paths."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        await store.clear()

        paths = await store.get_all_source_paths()
        assert paths == []


# =============================================================================
# TestInMemoryStoreHealthCheck - Test health_check() method
# =============================================================================


class TestInMemoryStoreHealthCheck:
    """Tests for health_check() method - checking store health."""

    @pytest.mark.anyio
    async def test_health_check_always_returns_true(self):
        """Test health_check always returns True."""
        store = InMemoryStore()
        is_healthy = await store.health_check()
        assert is_healthy is True

    @pytest.mark.anyio
    async def test_health_check_with_populated_store(self, sample_chunks):
        """Test health_check returns True even with data."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        is_healthy = await store.health_check()
        assert is_healthy is True

    @pytest.mark.anyio
    async def test_health_check_after_clear(self, sample_chunks):
        """Test health_check returns True after clear."""
        store = InMemoryStore()
        await store.add(sample_chunks)
        await store.clear()

        is_healthy = await store.health_check()
        assert is_healthy is True

    @pytest.mark.anyio
    async def test_health_check_multiple_calls(self):
        """Test health_check can be called multiple times."""
        store = InMemoryStore()

        for _ in range(5):
            is_healthy = await store.health_check()
            assert is_healthy is True


# =============================================================================
# TestCosineSimilarity - Test cosine similarity computation
# =============================================================================


class TestCosineSimilarity:
    """Tests for _cosine_similarity method."""

    def test_cosine_similarity_identical_vectors(self, embedding_dimension):
        """Test cosine similarity of identical vectors is close to 1.0."""
        vec = np.ones(embedding_dimension, dtype=np.float32)
        similarity = InMemoryStore._cosine_similarity(vec, vec)

        assert similarity > 0.99

    def test_cosine_similarity_orthogonal_vectors(self, embedding_dimension):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec1 = np.zeros(embedding_dimension, dtype=np.float32)
        vec1[0] = 1.0

        vec2 = np.zeros(embedding_dimension, dtype=np.float32)
        vec2[1] = 1.0

        similarity = InMemoryStore._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_opposite_vectors(self, embedding_dimension):
        """Test cosine similarity of opposite vectors is clamped to 0."""
        vec = np.ones(embedding_dimension, dtype=np.float32)
        neg_vec = -vec

        similarity = InMemoryStore._cosine_similarity(vec, neg_vec)

        # Should be clamped to [0, 1]
        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_zero_vector(self, embedding_dimension):
        """Test cosine similarity with zero vector returns 0.0."""
        vec1 = np.ones(embedding_dimension, dtype=np.float32)
        vec2 = np.zeros(embedding_dimension, dtype=np.float32)

        similarity = InMemoryStore._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_both_zero_vectors(self, embedding_dimension):
        """Test cosine similarity of two zero vectors is 0.0."""
        vec1 = np.zeros(embedding_dimension, dtype=np.float32)
        vec2 = np.zeros(embedding_dimension, dtype=np.float32)

        similarity = InMemoryStore._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_cosine_similarity_normalized_vectors(self):
        """Test cosine similarity calculation with normalized vectors."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.5, 0.5, 0.707], dtype=np.float32) / np.linalg.norm(
            np.array([0.5, 0.5, 0.707])
        )

        similarity = InMemoryStore._cosine_similarity(vec1, vec2)

        assert 0.0 <= similarity <= 1.0


# =============================================================================
# TestFiltering - Test _matches_filters method
# =============================================================================


class TestFiltering:
    """Tests for _matches_filters method."""

    def test_matches_filters_empty_filters(self, sample_chunk):
        """Test empty filters match all chunks."""
        result = InMemoryStore._matches_filters(sample_chunk, {})
        assert result is True

    def test_matches_filters_single_matching_filter(self, sample_chunk):
        """Test chunk matches single matching filter."""
        filters = {"language": "python"}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is True

    def test_matches_filters_single_nonmatching_filter(self, sample_chunk):
        """Test chunk doesn't match nonmatching filter."""
        filters = {"language": "javascript"}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is False

    def test_matches_filters_multiple_all_match(self, sample_chunk):
        """Test chunk matches when all filters match."""
        filters = {"language": "python", "source_type": SourceType.FILESYSTEM.value}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is True

    def test_matches_filters_multiple_one_fails(self, sample_chunk):
        """Test chunk doesn't match if any filter fails."""
        filters = {"language": "python", "file_extension": ".js"}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is False

    def test_matches_filters_list_value_matching(self, sample_chunk):
        """Test chunk matches filter with list of acceptable values."""
        filters = {"language": ["python", "javascript"]}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is True

    def test_matches_filters_list_value_nonmatching(self, sample_chunk):
        """Test chunk doesn't match filter with list if not in list."""
        filters = {"language": ["javascript", "go"]}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is False

    def test_matches_filters_source_path(self, sample_chunk):
        """Test matching by source_path."""
        filters = {"source_path": "/path/to/sample.py"}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is True

    def test_matches_filters_nonexistent_key(self, sample_chunk):
        """Test filter with nonexistent key doesn't match."""
        filters = {"nonexistent_key": "value"}
        result = InMemoryStore._matches_filters(sample_chunk, filters)
        assert result is False


# =============================================================================
# TestErrorHandling - Test error handling and exceptions
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and VectorStoreError."""

    @pytest.mark.anyio
    async def test_add_missing_embedding_raises_valueerror(self):
        """Test add with missing embedding raises ValueError (not VectorStoreError)."""
        store = InMemoryStore()
        chunk = Chunk(
            id="no-embed",
            content="No embedding",
            document_id="doc-1",
            embedding=None,
        )

        with pytest.raises(ValueError):
            await store.add([chunk])

    @pytest.mark.anyio
    async def test_search_with_valid_store(self, sample_chunks):
        """Test search handles various inputs without error."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        results = await store.search([0.1] * 384, top_k=5)
        assert isinstance(results, list)

    @pytest.mark.anyio
    async def test_delete_respects_return_count(self, sample_chunks):
        """Test delete correctly reports number of deleted items."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        deleted = await store.delete([sample_chunks[0].id, sample_chunks[1].id])
        assert deleted == 2

    @pytest.mark.anyio
    async def test_search_error_wrapped_in_vector_store_error(self, sample_chunks):
        """Test search errors are wrapped in VectorStoreError."""
        store = InMemoryStore()
        await store.add(sample_chunks)

        with patch("numpy.array", side_effect=RuntimeError("Array failed")):
            with pytest.raises(VectorStoreError, match="Search failed"):
                await store.search([0.1, 0.2, 0.3], top_k=5)


# =============================================================================
# TestIntegration - Integration tests combining multiple methods
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple methods."""

    @pytest.mark.anyio
    async def test_add_search_delete_workflow(self, sample_chunks, sample_embedding):
        """Test typical workflow: add, search, delete."""
        store = InMemoryStore()

        # Add chunks
        ids = await store.add(sample_chunks)
        assert len(ids) == len(sample_chunks)

        # Search
        results = await store.search(sample_embedding, top_k=3)
        assert len(results) <= 3

        # Delete one result
        deleted = await store.delete([results[0].chunk.id])
        assert deleted == 1

        # Verify count decreased
        count = await store.count()
        assert count == len(sample_chunks) - 1

    @pytest.mark.anyio
    async def test_full_lifecycle(self, sample_chunks):
        """Test complete lifecycle of store operations."""
        store = InMemoryStore()

        # Empty store checks
        assert await store.count() == 0
        assert await store.get_all_source_paths() == []
        assert await store.health_check() is True

        # Add chunks
        await store.add(sample_chunks)
        assert await store.count() == len(sample_chunks)

        # Get source paths
        paths = await store.get_all_source_paths()
        assert len(paths) > 0

        # Get by source
        if paths:
            chunks = await store.get_by_source(paths[0])
            assert len(chunks) > 0

        # Get stats
        stats = await store.get_stats()
        assert stats["total_chunks"] == len(sample_chunks)

        # Clear
        await store.clear()
        assert await store.count() == 0

    @pytest.mark.anyio
    async def test_multiple_documents_operations(self, sample_chunk):
        """Test operations across multiple documents."""
        store = InMemoryStore()

        # Create chunks for multiple documents
        chunks = []
        for doc_id in ["doc-1", "doc-2", "doc-3"]:
            for i in range(3):
                chunks.append(
                    Chunk(
                        id=f"{doc_id}-chunk-{i}",
                        content=f"Content for {doc_id}",
                        document_id=doc_id,
                        metadata=Metadata(source_path=f"/docs/{doc_id}.py"),
                        embedding=sample_chunk.embedding,
                    )
                )

        await store.add(chunks)

        # Test delete by document
        deleted = await store.delete_by_document("doc-1")
        assert deleted == 3

        # Test remaining
        assert await store.count() == 6

        # Test source paths
        paths = await store.get_all_source_paths()
        assert "/docs/doc-1.py" not in paths
        assert len(paths) == 2

    @pytest.mark.anyio
    async def test_search_after_modifications(self, sample_chunks, sample_embedding):
        """Test search works correctly after add/delete operations."""
        store = InMemoryStore()

        # Add initial chunks
        await store.add(sample_chunks[:3])
        results1 = await store.search(sample_embedding, top_k=5)

        # Add more chunks
        await store.add(sample_chunks[3:])
        results2 = await store.search(sample_embedding, top_k=5)

        # Should have more candidates now
        assert len(results2) >= len(results1)

        # Delete some chunks
        await store.delete([sample_chunks[0].id])
        results3 = await store.search(sample_embedding, top_k=5)

        # Should have fewer results now
        assert len(results3) <= len(results2)
