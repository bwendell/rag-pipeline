"""In-memory vector store implementation for testing and development.

This module provides a simple in-memory implementation of the VectorStore
protocol, suitable for testing, development, and small-scale applications.

All data is stored in memory and will be lost when the program terminates.
For production use, consider using ChromaDB, Qdrant, or OCI Vector Search.

Example:
    >>> from rag_pipeline.vector_stores.in_memory_store import InMemoryStore
    >>> from rag_pipeline.core.types import Chunk, Metadata
    >>>
    >>> store = InMemoryStore()
    >>> chunks = [
    ...     Chunk(
    ...         id="chunk-1",
    ...         content="Hello world",
    ...         document_id="doc-1",
    ...         embedding=[0.1, 0.2, 0.3],
    ...     )
    ... ]
    >>> ids = await store.add(chunks)
    >>> results = await store.search([0.1, 0.2, 0.3], top_k=5)
"""

from typing import Any

import numpy as np

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, SearchResult


class InMemoryStore:
    """In-memory vector store for testing and development.

    This implementation stores all data in memory using simple Python
    data structures. It's suitable for testing, prototyping, and
    development, but not recommended for production use.

    Attributes:
        _chunks: Dictionary mapping chunk IDs to Chunk objects.
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self._chunks: dict[str, Chunk] = {}

    async def add(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the store.

        Args:
            chunks: List of chunks with embeddings to store.

        Returns:
            List of chunk IDs in the same order as input.

        Raises:
            ValueError: If any chunk is missing an embedding.
            VectorStoreError: If storage fails.

        Example:
            >>> ids = await store.add([chunk1, chunk2])
            >>> assert len(ids) == 2
        """
        try:
            ids = []
            for chunk in chunks:
                if not chunk.has_embedding:
                    raise ValueError(
                        f"Chunk {chunk.id} is missing embedding. "
                        "All chunks must have embeddings before adding to store."
                    )
                self._chunks[chunk.id] = chunk
                ids.append(chunk.id)
            return ids
        except ValueError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to add chunks: {e!s}") from e

    async def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector embedding.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by similarity (highest first).

        Raises:
            VectorStoreError: If search fails.

        Example:
            >>> results = await store.search(
            ...     query_embedding=[0.1, 0.2, 0.3],
            ...     top_k=5,
            ...     filters={"language": "python"}
            ... )
            >>> for result in results:
            ...     print(f"{result.rank}. {result.score:.3f}")
        """
        try:
            if not self._chunks:
                return []

            # Convert query embedding to numpy array for efficient computation
            query_vec = np.array(query_embedding, dtype=np.float32)

            # Calculate similarities for all chunks
            candidates: list[tuple[str, float]] = []

            for chunk_id, chunk in self._chunks.items():
                # Apply filters if provided
                if filters and not self._matches_filters(chunk, filters):
                    continue

                # Calculate cosine similarity
                if chunk.embedding is None:
                    continue

                chunk_vec = np.array(chunk.embedding, dtype=np.float32)
                similarity = self._cosine_similarity(query_vec, chunk_vec)
                candidates.append((chunk_id, similarity))

            # Sort by similarity (highest first) and take top_k
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_k]

            # Build SearchResult objects
            results = []
            for rank, (chunk_id, score) in enumerate(top_candidates, 1):
                chunk = self._chunks[chunk_id]
                results.append(SearchResult(chunk=chunk, score=score, rank=rank))

            return results

        except Exception as e:
            raise VectorStoreError(f"Search failed: {e!s}") from e

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            ids: List of chunk IDs to delete.

        Returns:
            Number of chunks actually deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete(["id1", "id2"])
            >>> assert deleted == 2
        """
        try:
            deleted_count = 0
            for chunk_id in ids:
                if chunk_id in self._chunks:
                    del self._chunks[chunk_id]
                    deleted_count += 1
            return deleted_count
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks: {e!s}") from e

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: ID of the parent document.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete_by_document("doc-123")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            chunk_ids_to_delete = [
                chunk_id
                for chunk_id, chunk in self._chunks.items()
                if chunk.document_id == document_id
            ]
            return await self.delete(chunk_ids_to_delete)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks by document: {e!s}") from e

    async def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a source path.

        Args:
            source_path: Path from metadata.source_path.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete_by_source("/path/to/file.py")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            chunk_ids_to_delete = [
                chunk_id
                for chunk_id, chunk in self._chunks.items()
                if chunk.metadata.source_path == source_path
            ]
            return await self.delete(chunk_ids_to_delete)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks by source: {e!s}") from e

    async def get(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by ID.

        Args:
            ids: List of chunk IDs to retrieve.

        Returns:
            List of found chunks (missing IDs are skipped).

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> chunks = await store.get(["id1", "id2"])
            >>> assert len(chunks) <= 2
        """
        try:
            chunks = []
            for chunk_id in ids:
                if chunk_id in self._chunks:
                    chunks.append(self._chunks[chunk_id])
            return chunks
        except Exception as e:
            raise VectorStoreError(f"Failed to get chunks: {e!s}") from e

    async def get_by_source(self, source_path: str, limit: int = 100) -> list[Chunk]:
        """Retrieve all chunks from a source path.

        Args:
            source_path: Path from metadata.source_path.
            limit: Maximum number of chunks to return.

        Returns:
            List of chunks from source, ordered by chunk_index.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> chunks = await store.get_by_source("/path/to/file.py")
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.chunk_index}: {chunk.content[:50]}")
        """
        try:
            chunks = [
                chunk
                for chunk in self._chunks.values()
                if chunk.metadata.source_path == source_path
            ]
            # Sort by chunk_index
            chunks.sort(key=lambda c: c.chunk_index)
            return chunks[:limit]
        except Exception as e:
            raise VectorStoreError(f"Failed to get chunks by source: {e!s}") from e

    async def get_all_source_paths(self) -> list[str]:
        """Get all unique source paths in the store.

        Returns:
            List of unique source_path values.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> paths = await store.get_all_source_paths()
            >>> print(f"Store contains {len(paths)} indexed files")
        """
        try:
            source_paths = set()
            for chunk in self._chunks.values():
                if chunk.metadata.source_path:
                    source_paths.add(chunk.metadata.source_path)
            return sorted(source_paths)
        except Exception as e:
            raise VectorStoreError(f"Failed to get source paths: {e!s}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the store.

        Returns:
            Dictionary with store statistics.

        Raises:
            VectorStoreError: If stats retrieval fails.

        Example:
            >>> stats = await store.get_stats()
            >>> print(f"Total chunks: {stats['total_chunks']}")
        """
        try:
            total_chunks = len(self._chunks)
            total_documents = len({c.document_id for c in self._chunks.values()})

            # Calculate embedding dimension from first chunk
            embedding_dimension = 0
            if self._chunks:
                first_chunk = next(iter(self._chunks.values()))
                if first_chunk.embedding:
                    embedding_dimension = len(first_chunk.embedding)

            return {
                "store_type": "in_memory",
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "embedding_dimension": embedding_dimension,
                "index_type": "flat",
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to get stats: {e!s}") from e

    async def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Total chunk count.

        Raises:
            VectorStoreError: If count retrieval fails.

        Example:
            >>> total = await store.count()
            >>> print(f"Store has {total} chunks")
        """
        try:
            return len(self._chunks)
        except Exception as e:
            raise VectorStoreError(f"Failed to count chunks: {e!s}") from e

    async def clear(self) -> None:
        """Remove all data from the store.

        WARNING: This operation is destructive and cannot be undone.

        Raises:
            VectorStoreError: If clearing fails.

        Example:
            >>> await store.clear()
            >>> assert await store.count() == 0
        """
        try:
            self._chunks.clear()
        except Exception as e:
            raise VectorStoreError(f"Failed to clear store: {e!s}") from e

    async def health_check(self) -> bool:
        """Check if the store is healthy.

        Returns:
            True if the store is operational.

        Example:
            >>> if await store.health_check():
            ...     print("Store is healthy")
        """
        try:
            # In-memory store is always healthy if we can access it
            _ = len(self._chunks)
            return True
        except Exception:
            return False

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector (numpy array).
            vec2: Second vector (numpy array).

        Returns:
            Cosine similarity score between 0 and 1.
        """
        # Compute norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity: dot(a,b) / (norm(a) * norm(b))
        dot_product = float(np.dot(vec1, vec2))
        similarity = dot_product / (float(norm1) * float(norm2))

        # Clamp to [0, 1] range to handle numerical precision issues
        clamped = max(0.0, min(1.0, similarity))
        return float(clamped)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check if a chunk matches the provided metadata filters.

        Args:
            chunk: Chunk to check.
            filters: Dictionary of filter key-value pairs.

        Returns:
            True if chunk matches all filters, False otherwise.
        """
        if not filters:
            return True

        # Get metadata as dictionary
        metadata_dict = chunk.metadata.to_dict()

        # Check each filter
        for key, expected_value in filters.items():
            if key not in metadata_dict:
                return False

            actual_value = metadata_dict[key]

            # Handle list values (any match is OK)
            if isinstance(expected_value, (list, tuple)):
                if actual_value not in expected_value:
                    return False
            else:
                # Exact match required
                if actual_value != expected_value:
                    return False

        return True
