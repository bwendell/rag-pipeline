"""Performance and stress tests for vector store implementations.

This module contains tests that verify vector stores can handle:
- Large numbers of chunks (10,000+)
- High-dimensional embeddings (1536-dim like OpenAI)
- Concurrent operations (add/search/delete)
- Sustained load over time

These tests are marked with @pytest.mark.slow and should be run separately
from the main test suite.

Example:
    # Run only stress tests
    pytest tests/test_vector_stores/test_stress.py -v

    # Run all tests including slow ones
    pytest tests/test_vector_stores/ --run-slow
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rag_pipeline.core.types import Chunk, Metadata, SourceType
from rag_pipeline.vector_stores import ChromaDBStore, InMemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Large Dataset Tests
# =============================================================================


@pytest.mark.slow
class TestInMemoryStoreStress:
    """Stress tests for InMemoryStore.

    These tests verify that InMemoryStore can handle large datasets
    and perform within acceptable time limits.
    """

    @pytest.fixture
    def store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_handles_10000_chunks(self, store: InMemoryStore):
        """InMemoryStore must handle 10,000 chunks efficiently.

        This test verifies that adding and searching 10,000 chunks
        completes within a reasonable time frame.
        """
        # Create 10,000 chunks
        num_chunks = 10_000
        chunks = []
        for i in range(num_chunks):
            embedding = list(np.random.rand(384).astype(np.float32))
            chunk = Chunk(
                id=f"chunk-{i:05d}",
                content=f"Content for chunk {i}",
                document_id=f"doc-{i // 10:04d}",  # 10 chunks per document
                metadata=Metadata(
                    source_path=f"/path/file{i // 100:03d}.py",
                    source_type=SourceType.FILESYSTEM,
                    language="python",
                ),
                embedding=embedding,
                chunk_index=i % 10,
            )
            chunks.append(chunk)

        # Measure add time
        start_time = time.perf_counter()
        ids = await store.add(chunks)
        add_time = time.perf_counter() - start_time

        assert len(ids) == num_chunks
        assert add_time < 30.0, f"Adding {num_chunks} chunks took {add_time:.2f}s"

        # Verify count
        count = await store.count()
        assert count == num_chunks

        # Measure search time
        query_embedding = list(np.random.rand(384).astype(np.float32))
        start_time = time.perf_counter()
        results = await store.search(query_embedding, top_k=10)
        search_time = time.perf_counter() - start_time

        assert len(results) == 10
        assert search_time < 5.0, f"Search took {search_time:.2f}s"

    @pytest.mark.anyio
    async def test_handles_high_dimensional_embeddings(self, store: InMemoryStore):
        """InMemoryStore must handle 1536-dimensional embeddings (OpenAI size).

        Many embedding models (e.g., OpenAI text-embedding-ada-002)
        produce 1536-dimensional vectors.
        """
        dimension = 1536
        num_chunks = 100

        chunks = []
        for i in range(num_chunks):
            embedding = list(np.random.rand(dimension).astype(np.float32))
            chunk = Chunk(
                id=f"hd-chunk-{i}",
                content=f"High dimensional content {i}",
                document_id="hd-doc",
                embedding=embedding,
            )
            chunks.append(chunk)

        ids = await store.add(chunks)
        assert len(ids) == num_chunks

        # Search with high-dimensional embedding
        query_embedding = list(np.random.rand(dimension).astype(np.float32))
        results = await store.search(query_embedding, top_k=5)

        assert len(results) <= 5
        assert all(0.0 <= r.score <= 1.0 for r in results)

    @pytest.mark.anyio
    async def test_memory_usage_with_large_chunks(self, store: InMemoryStore):
        """Test handling of chunks with large content."""
        num_chunks = 1000
        content_size = 10_000  # 10KB per chunk

        chunks = []
        for i in range(num_chunks):
            embedding = list(np.random.rand(384).astype(np.float32))
            content = f"Large content {i}: " + "x" * content_size
            chunk = Chunk(
                id=f"large-chunk-{i}",
                content=content,
                document_id="large-doc",
                embedding=embedding,
            )
            chunks.append(chunk)

        ids = await store.add(chunks)
        assert len(ids) == num_chunks

        # Verify retrieval works
        retrieved = await store.get(["large-chunk-0", "large-chunk-500"])
        assert len(retrieved) == 2
        assert len(retrieved[0].content) > content_size


@pytest.mark.slow
class TestChromaDBStress:
    """Stress tests for ChromaDBStore."""

    @pytest.fixture
    async def store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="stress_test_collection",
            )
            yield chroma_store
            await chroma_store.clear()

    @pytest.mark.anyio
    async def test_handles_large_dataset(self, store: ChromaDBStore):
        """ChromaDBStore must handle 1000 chunks efficiently.

        Note: ChromaDB is disk-based, so we use a smaller number
        than InMemoryStore to keep tests reasonably fast.
        """
        num_chunks = 1000
        chunks = []

        for i in range(num_chunks):
            embedding = list(np.random.rand(384).astype(np.float32))
            chunk = Chunk(
                id=f"chroma-chunk-{i:04d}",
                content=f"ChromaDB test content {i}",
                document_id=f"chroma-doc-{i // 5:03d}",
                metadata=Metadata(
                    source_path=f"/chroma/file{i // 50:02d}.py",
                    source_type=SourceType.FILESYSTEM,
                    language="python",
                ),
                embedding=embedding,
                chunk_index=i % 5,
            )
            chunks.append(chunk)

        # Measure add time
        start_time = time.perf_counter()
        ids = await store.add(chunks)
        add_time = time.perf_counter() - start_time

        assert len(ids) == num_chunks
        assert add_time < 60.0, f"Adding {num_chunks} chunks took {add_time:.2f}s"

        # Measure search time
        query_embedding = list(np.random.rand(384).astype(np.float32))
        start_time = time.perf_counter()
        results = await store.search(query_embedding, top_k=10)
        search_time = time.perf_counter() - start_time

        assert len(results) <= 10
        assert search_time < 5.0, f"Search took {search_time:.2f}s"

    @pytest.mark.anyio
    async def test_handles_high_dimensional_embeddings(self, store: ChromaDBStore):
        """ChromaDBStore must handle 1536-dimensional embeddings."""
        dimension = 1536
        num_chunks = 50

        chunks = []
        for i in range(num_chunks):
            embedding = list(np.random.rand(dimension).astype(np.float32))
            chunk = Chunk(
                id=f"chroma-hd-{i}",
                content=f"High dimensional chroma content {i}",
                document_id="chroma-hd-doc",
                embedding=embedding,
            )
            chunks.append(chunk)

        ids = await store.add(chunks)
        assert len(ids) == num_chunks

        # Verify stats show correct dimension
        stats = await store.get_stats()
        assert stats["embedding_dimension"] == dimension

        # Search should work
        query_embedding = list(np.random.rand(dimension).astype(np.float32))
        results = await store.search(query_embedding, top_k=5)
        assert len(results) <= 5


# =============================================================================
# Concurrency Tests
# =============================================================================


@pytest.mark.slow
class TestInMemoryStoreConcurrency:
    """Concurrency tests for InMemoryStore."""

    @pytest.fixture
    def store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_concurrent_adds(self, store: InMemoryStore):
        """Multiple concurrent add operations should complete correctly."""

        async def add_batch(batch_id: int, num_chunks: int) -> list[str]:
            chunks = []
            for i in range(num_chunks):
                embedding = list(np.random.rand(384).astype(np.float32))
                chunk = Chunk(
                    id=f"batch{batch_id}-chunk{i}",
                    content=f"Batch {batch_id} chunk {i}",
                    document_id=f"batch{batch_id}-doc",
                    embedding=embedding,
                )
                chunks.append(chunk)
            return await store.add(chunks)

        # Run 5 concurrent add operations
        tasks = [add_batch(i, 50) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all chunks were added
        total_added = sum(len(ids) for ids in results)
        assert total_added == 250

        count = await store.count()
        assert count == 250

    @pytest.mark.anyio
    async def test_concurrent_searches(self, store: InMemoryStore):
        """Multiple concurrent search operations should complete correctly."""
        # Add some chunks first
        chunks = [
            Chunk(
                id=f"concurrent-chunk-{i}",
                content=f"Content {i}",
                document_id="concurrent-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(100)
        ]
        await store.add(chunks)

        async def search() -> list[Any]:
            query = list(np.random.rand(384).astype(np.float32))
            return await store.search(query, top_k=5)

        # Run 20 concurrent searches
        tasks = [search() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All searches should return results
        assert all(len(r) <= 5 for r in results)

    @pytest.mark.anyio
    async def test_concurrent_add_and_search(self, store: InMemoryStore):
        """Concurrent adds and searches should not interfere."""

        async def add_chunks(start: int, count: int) -> list[str]:
            chunks = [
                Chunk(
                    id=f"mixed-chunk-{start + i}",
                    content=f"Content {start + i}",
                    document_id="mixed-doc",
                    embedding=list(np.random.rand(384).astype(np.float32)),
                )
                for i in range(count)
            ]
            return await store.add(chunks)

        async def search_repeatedly(n: int) -> int:
            count = 0
            for _ in range(n):
                query = list(np.random.rand(384).astype(np.float32))
                results = await store.search(query, top_k=3)
                count += len(results)
            return count

        # Run adds and searches concurrently
        tasks = [
            add_chunks(0, 50),
            add_chunks(50, 50),
            search_repeatedly(10),
            search_repeatedly(10),
        ]
        results = await asyncio.gather(*tasks)

        # Verify adds completed
        assert len(results[0]) == 50
        assert len(results[1]) == 50

        # Verify final count
        count = await store.count()
        assert count == 100

    @pytest.mark.anyio
    async def test_concurrent_deletes(self, store: InMemoryStore):
        """Concurrent delete operations should complete correctly."""
        # Add chunks first
        chunks = [
            Chunk(
                id=f"delete-chunk-{i}",
                content=f"Content {i}",
                document_id="delete-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(100)
        ]
        await store.add(chunks)

        async def delete_range(start: int, end: int) -> int:
            ids = [f"delete-chunk-{i}" for i in range(start, end)]
            return await store.delete(ids)

        # Run concurrent deletes
        tasks = [
            delete_range(0, 25),
            delete_range(25, 50),
            delete_range(50, 75),
            delete_range(75, 100),
        ]
        results = await asyncio.gather(*tasks)

        # Verify all chunks were deleted
        assert sum(results) == 100

        count = await store.count()
        assert count == 0


@pytest.mark.slow
class TestChromaDBConcurrency:
    """Concurrency tests for ChromaDBStore."""

    @pytest.fixture
    async def store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="concurrency_test",
            )
            yield chroma_store
            await chroma_store.clear()

    @pytest.mark.anyio
    async def test_concurrent_adds_stability(self, store: ChromaDBStore):
        """ChromaDB should handle concurrent adds without corruption."""

        async def add_batch(batch_id: int) -> list[str]:
            chunks = [
                Chunk(
                    id=f"concurrent-chroma-{batch_id}-{i}",
                    content=f"Batch {batch_id} chunk {i}",
                    document_id=f"concurrent-chroma-doc-{batch_id}",
                    embedding=list(np.random.rand(384).astype(np.float32)),
                )
                for i in range(20)
            ]
            return await store.add(chunks)

        # Run concurrent adds
        tasks = [add_batch(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        total_added = sum(len(ids) for ids in results)
        assert total_added == 100

        # Verify count
        count = await store.count()
        assert count == 100

    @pytest.mark.anyio
    async def test_concurrent_searches(self, store: ChromaDBStore):
        """Multiple concurrent searches should complete successfully."""
        # Add test data
        chunks = [
            Chunk(
                id=f"search-chroma-{i}",
                content=f"Search content {i}",
                document_id="search-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(50)
        ]
        await store.add(chunks)

        async def search() -> list[Any]:
            query = list(np.random.rand(384).astype(np.float32))
            return await store.search(query, top_k=5)

        # Run concurrent searches
        tasks = [search() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(isinstance(r, list) for r in results)


# =============================================================================
# Performance Baseline Tests
# =============================================================================


@pytest.mark.slow
class TestPerformanceBaselines:
    """Tests to establish performance baselines."""

    @pytest.mark.anyio
    async def test_in_memory_search_performance(self):
        """Establish baseline for InMemoryStore search performance."""
        store = InMemoryStore()

        # Add chunks
        num_chunks = 1000
        chunks = [
            Chunk(
                id=f"perf-chunk-{i}",
                content=f"Performance test content {i}",
                document_id="perf-doc",
                embedding=list(np.random.rand(384).astype(np.float32)),
            )
            for i in range(num_chunks)
        ]
        await store.add(chunks)

        # Benchmark search
        num_searches = 100
        query = list(np.random.rand(384).astype(np.float32))

        start_time = time.perf_counter()
        for _ in range(num_searches):
            await store.search(query, top_k=10)
        elapsed = time.perf_counter() - start_time

        avg_time_ms = (elapsed / num_searches) * 1000
        print(f"\nInMemoryStore avg search time: {avg_time_ms:.2f}ms")

        # Should complete 100 searches in less than 10 seconds
        assert elapsed < 10.0, f"100 searches took {elapsed:.2f}s"

    @pytest.mark.anyio
    async def test_chroma_search_performance(self):
        """Establish baseline for ChromaDBStore search performance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="perf_test",
            )

            # Add chunks
            num_chunks = 500
            chunks = [
                Chunk(
                    id=f"chroma-perf-{i}",
                    content=f"Chroma performance test {i}",
                    document_id="chroma-perf-doc",
                    embedding=list(np.random.rand(384).astype(np.float32)),
                )
                for i in range(num_chunks)
            ]
            await store.add(chunks)

            # Benchmark search
            num_searches = 50
            query = list(np.random.rand(384).astype(np.float32))

            start_time = time.perf_counter()
            for _ in range(num_searches):
                await store.search(query, top_k=10)
            elapsed = time.perf_counter() - start_time

            avg_time_ms = (elapsed / num_searches) * 1000
            print(f"\nChromaDBStore avg search time: {avg_time_ms:.2f}ms")

            # Should complete 50 searches in less than 10 seconds
            assert elapsed < 10.0, f"50 searches took {elapsed:.2f}s"


# =============================================================================
# Edge Case Performance Tests
# =============================================================================


@pytest.mark.slow
class TestEdgeCasePerformance:
    """Performance tests for edge cases."""

    @pytest.mark.anyio
    async def test_many_small_embeddings(self):
        """Test performance with many small (64-dim) embeddings."""
        store = InMemoryStore()

        num_chunks = 5000
        dimension = 64

        chunks = [
            Chunk(
                id=f"small-{i}",
                content=f"Small embedding {i}",
                document_id="small-doc",
                embedding=list(np.random.rand(dimension).astype(np.float32)),
            )
            for i in range(num_chunks)
        ]

        start_time = time.perf_counter()
        await store.add(chunks)
        add_time = time.perf_counter() - start_time

        assert add_time < 15.0, f"Adding {num_chunks} small chunks took {add_time:.2f}s"

        # Search should be fast with small dimensions
        query = list(np.random.rand(dimension).astype(np.float32))
        start_time = time.perf_counter()
        results = await store.search(query, top_k=10)
        search_time = time.perf_counter() - start_time

        assert search_time < 2.0, f"Search took {search_time:.2f}s"

    @pytest.mark.anyio
    async def test_single_large_document(self):
        """Test performance with a single document having many chunks."""
        store = InMemoryStore()

        num_chunks = 2000
        document_id = "large-single-doc"

        chunks = [
            Chunk(
                id=f"large-doc-chunk-{i}",
                content=f"Chunk {i} of large document",
                document_id=document_id,
                embedding=list(np.random.rand(384).astype(np.float32)),
                chunk_index=i,
            )
            for i in range(num_chunks)
        ]

        await store.add(chunks)

        # delete_by_document should handle large documents efficiently
        start_time = time.perf_counter()
        deleted = await store.delete_by_document(document_id)
        delete_time = time.perf_counter() - start_time

        assert deleted == num_chunks
        assert delete_time < 5.0, f"Deleting {num_chunks} chunks took {delete_time:.2f}s"

    @pytest.mark.anyio
    async def test_repeated_add_and_delete_cycles(self):
        """Test stability over repeated add/delete cycles."""
        store = InMemoryStore()

        for cycle in range(10):
            # Add chunks
            chunks = [
                Chunk(
                    id=f"cycle{cycle}-chunk{i}",
                    content=f"Cycle {cycle} chunk {i}",
                    document_id=f"cycle{cycle}-doc",
                    embedding=list(np.random.rand(384).astype(np.float32)),
                )
                for i in range(100)
            ]
            await store.add(chunks)

            # Search
            query = list(np.random.rand(384).astype(np.float32))
            results = await store.search(query, top_k=5)
            assert len(results) <= 5

            # Delete
            await store.delete([c.id for c in chunks[:50]])

            count = await store.count()
            expected = (cycle + 1) * 100 - (cycle + 1) * 50
            assert count == expected, f"Cycle {cycle}: expected {expected}, got {count}"

        # Final cleanup
        await store.clear()
        assert await store.count() == 0
