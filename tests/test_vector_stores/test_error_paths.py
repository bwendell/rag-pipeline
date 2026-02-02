"""Error path tests for vector store implementations.

This module tests exception handling and error wrapping for all VectorStore
operations. It verifies that internal errors are properly wrapped in
VectorStoreError exceptions and that error handling is consistent across
implementations.

Tests use mocking to simulate error conditions without breaking real stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, Metadata, SourceType
from rag_pipeline.vector_stores import ChromaDBStore, InMemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# ChromaDB Error Handling Tests
# =============================================================================


class TestChromaDBErrorHandling:
    """Error path tests for ChromaDBStore.

    These tests verify that ChromaDBStore properly wraps exceptions
    in VectorStoreError and handles error conditions gracefully.
    """

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore for testing.

        Yields:
            ChromaDBStore instance with temporary storage.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="error_test_collection",
            )
            yield store

    @pytest.mark.anyio
    async def test_add_wraps_collection_error(self, chroma_store: ChromaDBStore):
        """add() must wrap collection errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.upsert to raise error
        with patch.object(
            chroma_store._collection,
            "upsert",
            side_effect=RuntimeError("ChromaDB error"),
        ):
            chunk = Chunk(
                id="test-chunk",
                content="Test content",
                document_id="doc-1",
                embedding=[0.1] * 384,
            )

            with pytest.raises(VectorStoreError, match="Failed to add"):
                await chroma_store.add([chunk])

    @pytest.mark.anyio
    async def test_search_wraps_query_error(self, chroma_store: ChromaDBStore):
        """search() must wrap query errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.query to raise error
        with patch.object(
            chroma_store._collection,
            "query",
            side_effect=RuntimeError("Query failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to search"):
                await chroma_store.search([0.1] * 384, top_k=5)

    @pytest.mark.anyio
    async def test_delete_wraps_delete_error(self, chroma_store: ChromaDBStore):
        """delete() must wrap deletion errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Add a chunk first
        chunk = Chunk(
            id="delete-test",
            content="Test",
            document_id="doc-1",
            embedding=[0.1] * 384,
        )
        await chroma_store.add([chunk])

        # Mock collection.delete to raise error
        with patch.object(
            chroma_store._collection,
            "delete",
            side_effect=RuntimeError("Delete failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to delete"):
                await chroma_store.delete(["delete-test"])

    @pytest.mark.anyio
    async def test_delete_by_document_wraps_error(self, chroma_store: ChromaDBStore):
        """delete_by_document() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to delete"):
                await chroma_store.delete_by_document("doc-1")

    @pytest.mark.anyio
    async def test_delete_by_source_wraps_error(self, chroma_store: ChromaDBStore):
        """delete_by_source() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to delete"):
                await chroma_store.delete_by_source("/path/to/file.py")

    @pytest.mark.anyio
    async def test_get_wraps_get_error(self, chroma_store: ChromaDBStore):
        """get() must wrap get errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to get"):
                await chroma_store.get(["chunk-id"])

    @pytest.mark.anyio
    async def test_get_by_source_wraps_error(self, chroma_store: ChromaDBStore):
        """get_by_source() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to get"):
                await chroma_store.get_by_source("/path/to/file.py")

    @pytest.mark.anyio
    async def test_get_all_source_paths_wraps_error(self, chroma_store: ChromaDBStore):
        """get_all_source_paths() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to get"):
                await chroma_store.get_all_source_paths()

    @pytest.mark.anyio
    async def test_get_stats_wraps_error(self, chroma_store: ChromaDBStore):
        """get_stats() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.count to raise error
        with patch.object(
            chroma_store._collection,
            "count",
            side_effect=RuntimeError("Count failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to get stats"):
                await chroma_store.get_stats()

    @pytest.mark.anyio
    async def test_count_wraps_error(self, chroma_store: ChromaDBStore):
        """count() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.count to raise error
        with patch.object(
            chroma_store._collection,
            "count",
            side_effect=RuntimeError("Count failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to count"):
                await chroma_store.count()

    @pytest.mark.anyio
    async def test_clear_wraps_error(self, chroma_store: ChromaDBStore):
        """clear() must wrap errors in VectorStoreError."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        # Mock collection.get to raise error
        with patch.object(
            chroma_store._collection,
            "get",
            side_effect=RuntimeError("Get failed"),
        ):
            with pytest.raises(VectorStoreError, match="Failed to clear"):
                await chroma_store.clear()

    @pytest.mark.anyio
    async def test_health_check_returns_false_on_error(self, chroma_store: ChromaDBStore):
        """health_check() must return False when store has errors."""
        # Mock _ensure_collection to raise error
        with patch.object(
            chroma_store,
            "_ensure_collection",
            side_effect=RuntimeError("Connection failed"),
        ):
            result = await chroma_store.health_check()

            assert result is False

    @pytest.mark.anyio
    async def test_health_check_returns_true_when_healthy(self, chroma_store: ChromaDBStore):
        """health_check() must return True when store is operational."""
        result = await chroma_store.health_check()

        assert result is True


# =============================================================================
# InMemoryStore Error Handling Tests
# =============================================================================


class TestInMemoryStoreErrorHandling:
    """Error path tests for InMemoryStore.

    These tests verify that InMemoryStore properly handles error conditions
    and wraps exceptions in VectorStoreError where appropriate.
    """

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore for testing."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_add_missing_embedding_raises_valueerror(self, in_memory_store: InMemoryStore):
        """add() must raise ValueError for missing embedding."""
        chunk = Chunk(
            id="no-embedding",
            content="Content without embedding",
            document_id="doc-1",
            embedding=None,
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await in_memory_store.add([chunk])

    @pytest.mark.anyio
    async def test_add_empty_embedding_raises_valueerror(self, in_memory_store: InMemoryStore):
        """add() must raise ValueError for empty embedding list."""
        chunk = Chunk(
            id="empty-embedding",
            content="Content with empty embedding",
            document_id="doc-1",
            embedding=[],
        )

        with pytest.raises(ValueError, match="missing embedding"):
            await in_memory_store.add([chunk])

    @pytest.mark.anyio
    async def test_add_wraps_unexpected_error(self, in_memory_store: InMemoryStore):
        """add() must wrap unexpected errors in VectorStoreError."""
        chunk = Chunk(
            id="test-chunk",
            content="Test",
            document_id="doc-1",
            embedding=[0.1] * 384,
        )

        mock_chunks = MagicMock()
        mock_chunks.__setitem__ = MagicMock(side_effect=MemoryError("Out of memory"))
        in_memory_store._chunks = mock_chunks

        with pytest.raises(VectorStoreError, match="Failed to add"):
            await in_memory_store.add([chunk])

    @pytest.mark.anyio
    async def test_search_wraps_error(self, in_memory_store: InMemoryStore):
        """search() must wrap errors in VectorStoreError."""
        # Add some chunks first
        chunks = [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                embedding=[0.1] * 384,
            )
            for i in range(3)
        ]
        await in_memory_store.add(chunks)

        # Mock numpy operation to fail
        with patch("numpy.array", side_effect=RuntimeError("NumPy error")):
            with pytest.raises(VectorStoreError, match="Search failed"):
                await in_memory_store.search([0.1] * 384, top_k=5)

    @pytest.mark.anyio
    async def test_delete_wraps_error(self, in_memory_store: InMemoryStore):
        """delete() must wrap errors in VectorStoreError."""
        chunk = Chunk(
            id="delete-test",
            content="Test",
            document_id="doc-1",
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        mock_chunks = MagicMock()
        mock_chunks.__contains__ = MagicMock(return_value=True)
        mock_chunks.__delitem__ = MagicMock(side_effect=RuntimeError("Dict error"))
        in_memory_store._chunks = mock_chunks

        with pytest.raises(VectorStoreError, match="Failed to delete"):
            await in_memory_store.delete(["delete-test"])

    @pytest.mark.anyio
    async def test_get_wraps_error(self, in_memory_store: InMemoryStore):
        """get() must wrap errors in VectorStoreError."""
        mock_chunks = MagicMock()
        mock_chunks.__contains__ = MagicMock(return_value=True)
        mock_chunks.__getitem__ = MagicMock(side_effect=RuntimeError("Dict error"))
        in_memory_store._chunks = mock_chunks

        with pytest.raises(VectorStoreError, match="Failed to get"):
            await in_memory_store.get(["chunk-id"])

    @pytest.mark.anyio
    async def test_health_check_returns_false_on_error(self, in_memory_store: InMemoryStore):
        """health_check() must return False on internal error."""
        # Mock len() to raise error
        with patch("builtins.len", side_effect=RuntimeError("Error")):
            result = await in_memory_store.health_check()

            assert result is False

    @pytest.mark.anyio
    async def test_health_check_returns_true_when_healthy(self, in_memory_store: InMemoryStore):
        """health_check() must return True when store is operational."""
        result = await in_memory_store.health_check()

        assert result is True


# =============================================================================
# Error Context Tests
# =============================================================================


class TestErrorContext:
    """Tests for error context and details."""

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore for testing."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="error_context_test",
            )
            yield store

    @pytest.mark.anyio
    async def test_error_includes_details(self, chroma_store: ChromaDBStore):
        """VectorStoreError must include relevant details."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        with patch.object(
            chroma_store._collection,
            "upsert",
            side_effect=RuntimeError("ChromaDB error"),
        ):
            chunk = Chunk(
                id="test-chunk",
                content="Test",
                document_id="doc-1",
                embedding=[0.1] * 384,
            )

            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.add([chunk])

            error = exc_info.value
            assert isinstance(error.details, dict)
            assert "chunk_count" in error.details

    @pytest.mark.anyio
    async def test_search_error_includes_filters(self, chroma_store: ChromaDBStore):
        """Search error must include filter details."""
        # Pre-initialize collection
        _ = chroma_store._ensure_collection()

        with patch.object(
            chroma_store._collection,
            "query",
            side_effect=RuntimeError("Query failed"),
        ):
            with pytest.raises(VectorStoreError) as exc_info:
                await chroma_store.search(
                    [0.1] * 384,
                    top_k=5,
                    filters={"language": "python"},
                )

            error = exc_info.value
            assert "filters" in error.details
            assert error.details["top_k"] == 5


# =============================================================================
# Dimension Mismatch Tests
# =============================================================================


class TestDimensionMismatch:
    """Tests for embedding dimension mismatches.

    Note: Current implementations may not validate dimension mismatches
    at the store level. These tests document current behavior.
    """

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore for testing."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_in_memory_allows_different_dimensions(self, in_memory_store: InMemoryStore):
        """InMemoryStore currently allows mixed dimension embeddings."""
        chunk_384 = Chunk(
            id="chunk-384",
            content="384 dim chunk",
            document_id="doc-1",
            embedding=[0.1] * 384,
        )
        chunk_768 = Chunk(
            id="chunk-768",
            content="768 dim chunk",
            document_id="doc-1",
            embedding=[0.2] * 768,
        )

        # Both should be added without error (current behavior)
        ids = await in_memory_store.add([chunk_384, chunk_768])
        assert len(ids) == 2

    @pytest.mark.anyio
    async def test_search_with_different_dimension_query(self, in_memory_store: InMemoryStore):
        """Search with different dimension query may fail or return 0 results."""
        chunk = Chunk(
            id="chunk-384",
            content="384 dim chunk",
            document_id="doc-1",
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Search with different dimension - behavior varies by implementation
        # InMemoryStore uses numpy which may broadcast or error
        try:
            results = await in_memory_store.search([0.1] * 768, top_k=5)
            # If it doesn't error, it might return empty or partial results
            assert isinstance(results, list)
        except VectorStoreError:
            # Error is also acceptable behavior
            pass


# =============================================================================
# Edge Case Error Tests
# =============================================================================


class TestEdgeCaseErrors:
    """Tests for edge cases that might cause errors."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore for testing."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_add_with_none_in_embedding_list(self, in_memory_store: InMemoryStore):
        """Adding chunk with None values in embedding should be handled."""
        chunk = Chunk(
            id="test-chunk",
            content="Test",
            document_id="doc-1",
            embedding=[0.1, None, 0.3],  # type: ignore[list-item]
        )

        # Current behavior: may raise TypeError when computing similarity
        # This test documents that embedding validation is minimal
        try:
            await in_memory_store.add([chunk])
            # If added, should be retrievable
            retrieved = await in_memory_store.get(["test-chunk"])
            assert len(retrieved) == 1
        except (TypeError, ValueError):
            pass  # Also acceptable

    @pytest.mark.anyio
    async def test_search_with_empty_query_embedding(self, in_memory_store: InMemoryStore):
        """Search with empty query embedding should not crash."""
        chunks = [
            Chunk(
                id="chunk-1",
                content="Content",
                document_id="doc-1",
                embedding=[0.1] * 384,
            )
        ]
        await in_memory_store.add(chunks)

        # Empty query should be handled gracefully
        results = await in_memory_store.search([], top_k=5)

        # Should return empty results (zero similarity)
        assert isinstance(results, list)

    @pytest.mark.anyio
    async def test_delete_by_document_with_special_chars(self, in_memory_store: InMemoryStore):
        """delete_by_document should handle special characters in document ID."""
        chunk = Chunk(
            id="chunk-1",
            content="Content",
            document_id="doc-with-special-chars-!@#$%",
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Should handle special characters
        deleted = await in_memory_store.delete_by_document("doc-with-special-chars-!@#$%")

        assert deleted == 1

    @pytest.mark.anyio
    async def test_get_by_source_with_special_chars(self, in_memory_store: InMemoryStore):
        """get_by_source should handle special characters in source path."""
        chunk = Chunk(
            id="chunk-1",
            content="Content",
            document_id="doc-1",
            metadata=Metadata(source_path="/path/with spaces and special!@#chars.py"),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Should handle special characters
        results = await in_memory_store.get_by_source("/path/with spaces and special!@#chars.py")

        assert len(results) == 1
