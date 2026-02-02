"""Performance tests for end-to-end RAG pipeline integration.

These tests measure the performance characteristics of the complete pipeline
from document ingestion through embedding and vector storage, providing
end-to-end performance baselines.

Test Categories:
1. Pipeline Throughput: Documents processed per second
2. Latency Measurements: Time for complete pipeline steps
3. Memory Efficiency: Memory usage during processing
4. Scalability: Performance with increasing data volume
5. Query Performance: Search and retrieval speed

Run with:
    pytest tests/test_pipeline_performance.py -v
    pytest tests/test_pipeline_performance.py -v --benchmark  # With benchmarks
    pytest tests/test_pipeline_performance.py -v -m slow  # Include slow tests
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import numpy as np
import pytest

from rag_pipeline.chunkers import CodeChunker, MarkdownChunker, create_chunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import (
    Chunk,
    Document,
    DocumentType,
    Metadata,
    SourceType,
)
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore

if TYPE_CHECKING:
    from rag_pipeline.embedding_providers.base import EmbeddingProvider


# =============================================================================
# Performance Test Fixtures & Helpers
# =============================================================================


class PerformanceMetrics:
    """Container for performance measurements."""

    def __init__(self) -> None:
        """Initialize metrics."""
        self.chunking_time: float = 0.0
        self.embedding_time: float = 0.0
        self.storage_time: float = 0.0
        self.search_time: float = 0.0
        self.chunks_produced: int = 0
        self.memory_used: int = 0

    def __repr__(self) -> str:
        """Format metrics as string."""
        return (
            f"PerformanceMetrics("
            f"chunks={self.chunks_produced}, "
            f"chunk_time={self.chunking_time:.3f}s, "
            f"embed_time={self.embedding_time:.3f}s, "
            f"storage_time={self.storage_time:.3f}s, "
            f"search_time={self.search_time:.3f}s"
            f")"
        )

    @property
    def total_time(self) -> float:
        """Get total pipeline time."""
        return self.chunking_time + self.embedding_time + self.storage_time

    @property
    def chunks_per_second(self) -> float:
        """Calculate chunks processed per second."""
        if self.total_time == 0:
            return 0.0
        return self.chunks_produced / self.total_time


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Standard chunking configuration for performance tests."""
    return ChunkingConfig(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def embedding_dimension() -> int:
    """Standard embedding dimension."""
    return 384


@pytest.fixture
def mock_embedding_provider(embedding_dimension: int):
    """Create a mock embedding provider with deterministic behavior."""
    provider = AsyncMock()
    provider.dimension = embedding_dimension

    async def mock_embed(text: str) -> list[float]:
        """Generate a deterministic embedding."""
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        embedding = np.random.randn(embedding_dimension).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.tolist()

    async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch."""
        return [await mock_embed(text) for text in texts]

    provider.embed = mock_embed
    provider.embed_batch = mock_embed_batch
    return provider


# =============================================================================
# Sample Documents for Performance Testing
# =============================================================================


def create_sample_python_documents(count: int) -> list[Document]:
    """Create sample Python documents of varying sizes."""
    documents = []
    base_code = '''"""Module {}.

This module provides authentication functionality.
"""

import hashlib
from typing import Optional


class User:
    """Represents a system user."""
    
    def __init__(self, username: str, password: str):
        """Initialize user with credentials."""
        self.username = username
        self._password_hash = self._hash_password(password)
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return self._hash_password(password) == self._password_hash


class AuthenticationManager:
    """Manages user authentication."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.users: dict[str, User] = {{}}
    
    def register_user(self, username: str, password: str) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        self.users[username] = User(username, password)
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self.users.get(username)
        if user and user.verify_password(password):
            return user
        return None
'''

    for i in range(count):
        # Vary document size slightly
        content = base_code.format(i) * (1 + (i % 3))

        documents.append(
            Document(
                content=content,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path=f"/src/module_{i:04d}.py",
                    source_type=SourceType.FILESYSTEM,
                    language="python",
                    file_extension=".py",
                ),
            )
        )

    return documents


# =============================================================================
# Performance Test Classes
# =============================================================================


class TestChunkingPerformance:
    """Test chunking performance with various document sizes."""

    def test_chunk_small_documents(self, chunking_config: ChunkingConfig) -> None:
        """Measure chunking performance on small documents."""
        docs = create_sample_python_documents(10)
        chunker = CodeChunker(chunking_config)

        start_time = time.perf_counter()
        total_chunks = 0
        for doc in docs:
            chunks = chunker.chunk(doc)
            total_chunks += len(chunks)
        elapsed = time.perf_counter() - start_time

        # Performance assertion
        assert elapsed < 1.0, f"Chunking 10 docs took {elapsed:.3f}s"
        assert total_chunks > 0

    def test_chunk_medium_documents(self, chunking_config: ChunkingConfig) -> None:
        """Measure chunking performance on medium documents."""
        docs = create_sample_python_documents(50)
        chunker = CodeChunker(chunking_config)

        start_time = time.perf_counter()
        total_chunks = 0
        for doc in docs:
            chunks = chunker.chunk(doc)
            total_chunks += len(chunks)
        elapsed = time.perf_counter() - start_time

        # Should handle 50 docs quickly
        assert elapsed < 5.0, f"Chunking 50 docs took {elapsed:.3f}s"
        assert total_chunks > 0

    def test_chunk_consistency_across_runs(self, chunking_config: ChunkingConfig) -> None:
        """Verify chunking is consistent and repeatable."""
        doc = create_sample_python_documents(1)[0]
        chunker = CodeChunker(chunking_config)

        # Run chunking multiple times
        results = []
        for _ in range(3):
            chunks = chunker.chunk(doc)
            results.append(len(chunks))

        # All runs should produce same result
        assert len(set(results)) == 1, "Chunking results inconsistent"


class TestEmbeddingPerformance:
    """Test embedding performance for chunks."""

    @pytest.mark.asyncio
    async def test_embed_small_batch(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure embedding time for small batch."""
        docs = create_sample_python_documents(5)
        chunker = CodeChunker(chunking_config)

        # Chunk documents
        chunks = []
        for doc in docs:
            chunks.extend(chunker.chunk(doc))

        # Measure embedding time
        start_time = time.perf_counter()
        for chunk in chunks:
            chunk.embedding = await mock_embedding_provider.embed(chunk.content)
        elapsed = time.perf_counter() - start_time

        # Verify completion
        assert all(c.has_embedding for c in chunks)
        # Embedding should be fast with mock
        assert elapsed < 2.0, f"Embedding {len(chunks)} chunks took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_embed_batch_processing(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure batch embedding processing."""
        docs = create_sample_python_documents(10)
        chunker = CodeChunker(chunking_config)

        # Chunk all documents
        chunks = []
        for doc in docs:
            chunks.extend(chunker.chunk(doc))

        # Measure batch embedding
        start_time = time.perf_counter()
        texts = [c.content for c in chunks]
        embeddings = await mock_embedding_provider.embed_batch(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        elapsed = time.perf_counter() - start_time

        # Verify all embedded
        assert all(c.has_embedding for c in chunks)


class TestVectorStorePerformance:
    """Test vector store insertion and search performance."""

    @pytest.mark.asyncio
    async def test_add_small_batch(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure time to add small batch to vector store."""
        docs = create_sample_python_documents(5)
        chunker = CodeChunker(chunking_config)
        store = InMemoryStore()

        # Prepare chunks with embeddings
        chunks = []
        for doc in docs:
            doc_chunks = chunker.chunk(doc)
            for chunk in doc_chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)
            chunks.extend(doc_chunks)

        # Measure add time
        start_time = time.perf_counter()
        await store.add(chunks)
        elapsed = time.perf_counter() - start_time

        # Verify storage
        count = await store.count()
        assert count == len(chunks)
        # Should be very fast for small batch
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_search_performance(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure search performance on populated store."""
        docs = create_sample_python_documents(20)
        chunker = CodeChunker(chunking_config)
        store = InMemoryStore()

        # Populate store
        chunks = []
        for doc in docs:
            doc_chunks = chunker.chunk(doc)
            for chunk in doc_chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)
            chunks.extend(doc_chunks)

        await store.add(chunks)

        # Measure search time
        query_embedding = await mock_embedding_provider.embed("authentication password")

        start_time = time.perf_counter()
        results = await store.search(query_embedding, top_k=10)
        elapsed = time.perf_counter() - start_time

        # Verify results
        assert len(results) > 0
        # Search should be reasonably fast
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_add_large_batch(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure time to add larger batch to vector store."""
        docs = create_sample_python_documents(50)
        chunker = CodeChunker(chunking_config)
        store = InMemoryStore()

        # Prepare chunks
        chunks = []
        for doc in docs:
            doc_chunks = chunker.chunk(doc)
            for chunk in doc_chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)
            chunks.extend(doc_chunks)

        # Measure add time
        start_time = time.perf_counter()
        await store.add(chunks)
        elapsed = time.perf_counter() - start_time

        # Verify storage
        count = await store.count()
        assert count == len(chunks)
        # Should handle reasonable size efficiently
        assert elapsed < 10.0


class TestEndToEndPipelinePerformance:
    """Test complete pipeline performance from document to search."""

    @pytest.mark.asyncio
    async def test_full_pipeline_throughput(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure throughput of complete pipeline."""
        docs = create_sample_python_documents(20)
        store = InMemoryStore()
        metrics = PerformanceMetrics()

        # Process documents through full pipeline
        start_time = time.perf_counter()

        # 1. Chunk all documents
        chunking_start = time.perf_counter()
        all_chunks = []
        for doc in docs:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        metrics.chunking_time = time.perf_counter() - chunking_start
        metrics.chunks_produced = len(all_chunks)

        # 2. Embed all chunks
        embedding_start = time.perf_counter()
        for chunk in all_chunks:
            chunk.embedding = await mock_embedding_provider.embed(chunk.content)
        metrics.embedding_time = time.perf_counter() - embedding_start

        # 3. Store chunks
        storage_start = time.perf_counter()
        await store.add(all_chunks)
        metrics.storage_time = time.perf_counter() - storage_start

        total_time = time.perf_counter() - start_time

        # 4. Perform searches
        search_start = time.perf_counter()
        query_embedding = await mock_embedding_provider.embed("authentication")
        search_results = await store.search(query_embedding, top_k=5)
        metrics.search_time = time.perf_counter() - search_start

        # Verify results
        assert len(all_chunks) > 0
        assert await store.count() == len(all_chunks)
        assert len(search_results) > 0

        # Performance assertions
        assert total_time < 30.0, f"Full pipeline took {total_time:.2f}s"
        assert metrics.chunks_per_second > 10.0, (
            f"Pipeline throughput too low: {metrics.chunks_per_second:.1f} chunks/sec"
        )

    @pytest.mark.asyncio
    async def test_mixed_language_pipeline_performance(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Measure performance with mixed programming languages."""
        # This would require multilanguage documents - using Python as proxy
        docs = create_sample_python_documents(30)
        store = InMemoryStore()

        all_chunks = []
        for doc in docs:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        # Embed
        for chunk in all_chunks:
            chunk.embedding = await mock_embedding_provider.embed(chunk.content)

        # Store
        start_time = time.perf_counter()
        await store.add(all_chunks)
        storage_time = time.perf_counter() - start_time

        # Should handle mixed content efficiently
        assert storage_time < 10.0
        assert await store.count() == len(all_chunks)


class TestScalabilityAndLimits:
    """Test pipeline behavior at various scales."""

    @pytest.mark.asyncio
    async def test_progressive_document_loading(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test loading increasing numbers of documents."""
        store = InMemoryStore()
        sizes = [10, 25, 50]

        for size in sizes:
            docs = create_sample_python_documents(size)
            chunker = CodeChunker(chunking_config)

            # Chunk and embed
            for doc in docs:
                chunks = chunker.chunk(doc)
                for chunk in chunks:
                    chunk.embedding = await mock_embedding_provider.embed(chunk.content)
                # Add to store
                await store.add(chunks)

            # Verify
            count = await store.count()
            assert count > 0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_scale_document_handling(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test handling of larger document sets."""
        docs = create_sample_python_documents(100)
        chunker = CodeChunker(chunking_config)
        store = InMemoryStore()

        # Measure end-to-end time for 100 documents
        start_time = time.perf_counter()

        all_chunks = []
        for doc in docs:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)
            all_chunks.extend(chunks)

        await store.add(all_chunks)
        elapsed = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert elapsed < 60.0, f"Processing 100 docs took {elapsed:.2f}s"
        assert await store.count() == len(all_chunks)


# =============================================================================
# Performance Summary
# =============================================================================

__all__ = [
    "TestChunkingPerformance",
    "TestEmbeddingPerformance",
    "TestVectorStorePerformance",
    "TestEndToEndPipelinePerformance",
    "TestScalabilityAndLimits",
]
