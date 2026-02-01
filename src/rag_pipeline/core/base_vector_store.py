"""VectorStore Protocol definition.

This module defines the abstract interface for vector store implementations.
All vector stores (ChromaDB, Qdrant, OCI Vector Search, etc.) must conform
to this protocol.

Example usage:
    >>> from rag_pipeline.core import VectorStore
    >>>
    >>> class MyVectorStore:
    ...     async def add(self, chunks: list[Chunk]) -> list[str]:
    ...         # Store chunks and return their IDs
    ...         ...
    ...
    ...     async def search(
    ...         self,
    ...         query_embedding: list[float],
    ...         top_k: int = 5,
    ...         filters: dict | None = None
    ...     ) -> list[SearchResult]:
    ...         # Search for similar chunks
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyVectorStore(), VectorStore)
"""

from typing import Any, Protocol, runtime_checkable

from rag_pipeline.core.types import Chunk, SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store implementations.

    Vector stores are responsible for:
    1. Storing chunks with their embeddings
    2. Performing similarity searches
    3. Managing chunk lifecycle (add, update, delete)
    4. Providing statistics about stored data

    All methods are async to support both local and remote stores.
    Implementations should handle their own connection management.

    Implementations:
        - ChromaDBStore: Local embedded vector store
        - InMemoryStore: In-memory store for testing
        - OCIVectorStore (stub): OCI Vector Search
        - QdrantStore (stub): Qdrant vector database
    """

    async def add(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the vector store.

        Chunks should already have embeddings populated. If a chunk
        with the same ID already exists, it should be updated.

        Args:
            chunks: List of chunks to store. Each chunk must have:
                - id: Unique identifier
                - content: Text content
                - embedding: Vector embedding (required)
                - metadata: Associated metadata

        Returns:
            List of IDs for the stored chunks, in the same order
            as the input chunks.

        Raises:
            VectorStoreError: If storage fails.
            ValueError: If any chunk is missing an embedding.

        Example:
            >>> ids = await store.add([chunk1, chunk2])
            >>> print(f"Stored {len(ids)} chunks")
        """
        ...

    async def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.

        Performs approximate nearest neighbor search to find chunks
        most similar to the query embedding.

        Args:
            query_embedding: Vector embedding of the query.
            top_k: Maximum number of results to return (default: 5).
            filters: Optional metadata filters to narrow search.
                Example: {"language": "python", "source_type": "code"}

        Returns:
            List of SearchResult objects, ordered by similarity
            (highest score first). Each result contains:
                - chunk: The matched chunk
                - score: Similarity score (0-1, higher is better)
                - rank: Position in results (1-indexed)

        Raises:
            VectorStoreError: If search fails.

        Example:
            >>> results = await store.search(
            ...     query_embedding=embedding,
            ...     top_k=10,
            ...     filters={"language": "python"}
            ... )
            >>> for r in results:
            ...     print(f"{r.rank}. {r.score:.3f}: {r.chunk.content[:50]}")
        """
        ...

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by their IDs.

        Removes chunks from the store. IDs that don't exist are
        silently ignored.

        Args:
            ids: List of chunk IDs to delete.

        Returns:
            Number of chunks actually deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete(["id1", "id2", "id3"])
            >>> print(f"Deleted {deleted} chunks")
        """
        ...

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Useful for re-indexing a document - delete old chunks first,
        then add new ones.

        Args:
            document_id: ID of the parent document.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete_by_document("doc-123")
            >>> print(f"Deleted {deleted} chunks for document")
        """
        ...

    async def get(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs to retrieve.

        Returns:
            List of chunks found. Missing IDs are silently skipped,
            so the result may have fewer items than requested.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> chunks = await store.get(["id1", "id2"])
            >>> for chunk in chunks:
            ...     print(chunk.content)
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics. Common keys:
                - total_chunks: Total number of stored chunks
                - total_documents: Number of unique documents
                - embedding_dimension: Dimension of stored embeddings
                - index_type: Type of index (e.g., "hnsw", "flat")
                - store_type: Implementation name (e.g., "chroma")

        Example:
            >>> stats = await store.get_stats()
            >>> print(f"Store has {stats['total_chunks']} chunks")
        """
        ...

    async def clear(self) -> None:
        """Remove all data from the store.

        WARNING: This is destructive and cannot be undone.

        Raises:
            VectorStoreError: If clearing fails.

        Example:
            >>> await store.clear()
            >>> stats = await store.get_stats()
            >>> assert stats["total_chunks"] == 0
        """
        ...

    async def health_check(self) -> bool:
        """Check if the vector store is healthy and accessible.

        Returns:
            True if the store is operational, False otherwise.

        Example:
            >>> if await store.health_check():
            ...     print("Store is healthy")
            ... else:
            ...     print("Store is not responding")
        """
        ...

    async def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a specific source path.

        Useful for re-indexing a file - delete old chunks first,
        then add new ones. This is different from delete_by_document
        as it uses the source_path metadata field.

        Args:
            source_path: Path of the source document (from metadata.source_path).

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete_by_source("/path/to/file.py")
            >>> print(f"Deleted {deleted} chunks from file")
        """
        ...

    async def get_by_source(self, source_path: str, limit: int = 100) -> list[Chunk]:
        """Retrieve all chunks from a specific source path.

        Args:
            source_path: Path of the source document.
            limit: Maximum number of chunks to return.

        Returns:
            List of chunks from the source, ordered by chunk_index.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> chunks = await store.get_by_source("/path/to/file.py")
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.chunk_index}: {chunk.content[:50]}")
        """
        ...

    async def get_all_source_paths(self) -> list[str]:
        """Get all unique source paths in the store.

        Useful for incremental indexing to compare against
        current source state.

        Returns:
            List of unique source_path values.

        Example:
            >>> paths = await store.get_all_source_paths()
            >>> print(f"Store contains {len(paths)} indexed files")
        """
        ...

    async def count(self) -> int:
        """Get the total number of chunks in the store.

        Returns:
            Total chunk count.

        Example:
            >>> total = await store.count()
            >>> print(f"Store has {total} chunks")
        """
        ...
