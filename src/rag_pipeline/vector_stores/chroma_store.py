"""ChromaDB vector store implementation.

This module provides a persistent vector store implementation using ChromaDB,
an open-source embedding database. It supports all VectorStore protocol methods
with efficient similarity search and metadata filtering.

Example usage:
    >>> from rag_pipeline.vector_stores import ChromaDBStore
    >>> from rag_pipeline.core.types import Chunk, Metadata
    >>>
    >>> # Initialize store
    >>> store = ChromaDBStore(
    ...     persist_path="./data/chroma",
    ...     collection_name="rag_chunks"
    ... )
    >>>
    >>> # Add chunks
    >>> chunk = Chunk(
    ...     content="Example content",
    ...     document_id="doc-123",
    ...     metadata=Metadata(source_path="/path/to/file.py")
    ... )
    >>> ids = await store.add([chunk])
    >>>
    >>> # Search
    >>> results = await store.search(
    ...     query_embedding=[0.1, 0.2, 0.3],
    ...     top_k=5,
    ...     filters={"language": "python"}
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from rag_pipeline.core.exceptions import VectorStoreError
from rag_pipeline.core.types import Chunk, Metadata, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_PATH = "./data/chroma"
DEFAULT_COLLECTION_NAME = "rag_chunks"


class ChromaDBStore:
    """Vector store implementation using ChromaDB.

    This store provides persistent storage of chunk embeddings with
    metadata filtering and similarity search capabilities. Data is
    stored on disk using ChromaDB's PersistentClient.

    Attributes:
        persist_path: Directory path where ChromaDB data is persisted.
        collection_name: Name of the ChromaDB collection.
        _client: The ChromaDB PersistentClient instance (lazy-loaded).
        _collection: The ChromaDB collection instance (lazy-loaded).

    Example:
        >>> store = ChromaDBStore(
        ...     persist_path="./data/chroma",
        ...     collection_name="my_chunks"
        ... )
        >>> # Add chunks with embeddings
        >>> chunk = Chunk(
        ...     content="Hello world",
        ...     document_id="doc-123",
        ...     embedding=[0.1, 0.2, 0.3],
        ...     metadata=Metadata(source_path="hello.txt")
        ... )
        >>> ids = await store.add([chunk])
    """

    def __init__(
        self,
        persist_path: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            persist_path: Directory path for ChromaDB persistence.
                Defaults to RAG_CHROMA_PATH env var or "./data/chroma".
            collection_name: Name of the collection. Defaults to "rag_chunks".

        Raises:
            VectorStoreError: If the store cannot be initialized.

        Example:
            >>> store = ChromaDBStore(
            ...     persist_path="./chroma_data",
            ...     collection_name="my_app"
            ... )
        """
        self.persist_path: str = (
            persist_path or os.getenv("RAG_CHROMA_PATH") or DEFAULT_PERSIST_PATH
        )
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_client(self) -> Any:
        """Lazy-load the ChromaDB client.

        Returns:
            The PersistentClient instance.

        Raises:
            VectorStoreError: If client initialization fails.
        """
        if self._client is None:
            try:
                import chromadb

                logger.debug(f"Initializing ChromaDB client at {self.persist_path}")

                # Ensure parent directory exists
                Path(self.persist_path).mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(path=self.persist_path)
            except Exception as e:
                error_msg = f"Failed to initialize ChromaDB client at {self.persist_path}"
                logger.error(error_msg, exc_info=True)
                raise VectorStoreError(
                    error_msg,
                    details={"persist_path": self.persist_path, "error": str(e)},
                ) from e

        return self._client

    def _ensure_collection(self) -> Any:
        """Lazy-load or create the ChromaDB collection.

        Returns:
            The Collection instance.

        Raises:
            VectorStoreError: If collection access fails.
        """
        if self._collection is None:
            try:
                client = self._ensure_client()
                logger.debug(f"Accessing collection: {self.collection_name}")

                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                error_msg = f"Failed to access collection '{self.collection_name}'"
                logger.error(error_msg, exc_info=True)
                raise VectorStoreError(
                    error_msg,
                    details={"collection_name": self.collection_name, "error": str(e)},
                ) from e

        return self._collection

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Filter metadata to only include primitive types.

        ChromaDB only supports primitive types (str, int, float, bool) in metadata.
        This method recursively flattens nested dicts and filters out non-primitive values.

        Args:
            metadata: The metadata dictionary to filter.

        Returns:
            Dictionary containing only primitive values.
        """
        result: dict[str, Any] = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, list) and value:
                # ChromaDB doesn't support lists; convert to comma-separated string
                # Only include if all items are strings
                if all(isinstance(item, str) for item in value):
                    result[key] = ",".join(value)
            elif isinstance(value, dict):
                # Flatten nested dicts with dot notation
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        result[f"{key}.{nested_key}"] = nested_value

        return result

    async def add(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the vector store.

        Chunks must have embeddings populated. If a chunk with the same ID
        already exists, it will be updated.

        Args:
            chunks: List of chunks to store. Each chunk must have an embedding.

        Returns:
            List of IDs for the stored chunks.

        Raises:
            VectorStoreError: If storage fails.
            ValueError: If any chunk is missing an embedding.

        Example:
            >>> chunk = Chunk(
            ...     content="Example",
            ...     document_id="doc-123",
            ...     embedding=[0.1, 0.2, 0.3],
            ...     metadata=Metadata(source_path="example.txt")
            ... )
            >>> ids = await store.add([chunk])
            >>> print(f"Stored chunk with ID: {ids[0]}")
        """
        if not chunks:
            return []

        # Validate all chunks have embeddings
        for chunk in chunks:
            if not chunk.has_embedding:
                raise ValueError(
                    f"Chunk {chunk.id} has missing embedding. "
                    "All chunks must have embeddings before adding to store."
                )

        try:
            collection = self._ensure_collection()

            ids = [chunk.id for chunk in chunks]
            # Convert embeddings to native Python floats for ChromaDB compatibility
            # (ChromaDB doesn't accept numpy.float32 scalars in lists)
            embeddings = [
                [float(v) for v in chunk.embedding]  # type: ignore[union-attr]
                for chunk in chunks
            ]
            documents = [chunk.content for chunk in chunks]
            metadatas = [self._filter_metadata(chunk.metadata.to_dict()) for chunk in chunks]

            # Add document_id to metadata for lookup operations
            for i, chunk in enumerate(chunks):
                metadatas[i]["document_id"] = chunk.document_id
                metadatas[i]["chunk_index"] = chunk.chunk_index
                metadatas[i]["start_index"] = chunk.start_index
                metadatas[i]["end_index"] = chunk.end_index

            # Run sync ChromaDB operation in thread pool
            await asyncio.to_thread(
                collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.debug(f"Added {len(chunks)} chunks to ChromaDB")
            return ids

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to add {len(chunks)} chunks to ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"chunk_count": len(chunks), "error": str(e)},
            ) from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.

        Performs approximate nearest neighbor search to find chunks
        most similar to the query embedding.

        Args:
            query_embedding: Vector embedding of the query.
            top_k: Maximum number of results to return (default: 5).
            filters: Optional metadata filters to narrow search.

        Returns:
            List of SearchResult objects, ordered by similarity
            (highest score first).

        Raises:
            VectorStoreError: If search fails.

        Example:
            >>> results = await store.search(
            ...     query_embedding=[0.1, 0.2, 0.3],
            ...     top_k=10,
            ...     filters={"language": "python"}
            ... )
        """
        if top_k <= 0:
            return []

        try:
            collection = self._ensure_collection()

            where_clause = self._filter_metadata(filters) if filters else None
            normalized_query = [float(v) for v in query_embedding]

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[normalized_query],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )

            # Convert to SearchResult objects
            search_results: list[SearchResult] = []

            if not results["ids"] or not results["ids"][0]:
                return search_results

            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score (1 - distance for cosine)
                distance = distances[i] if i < len(distances) else 0.0
                similarity = 1.0 - float(distance)

                # Reconstruct metadata
                metadata_raw = metadatas[i] if i < len(metadatas) else {}
                metadata_dict: dict[str, Any] = dict(metadata_raw) if metadata_raw else {}
                metadata = Metadata.from_dict(metadata_dict)

                # Create chunk
                chunk = Chunk(
                    id=str(chunk_id),
                    content=str(documents[i]) if i < len(documents) else "",
                    document_id=str(metadata_dict.get("document_id", "")),
                    metadata=metadata,
                    start_index=int(metadata_dict.get("start_index", 0)),
                    end_index=int(metadata_dict.get("end_index", 0)),
                    chunk_index=int(metadata_dict.get("chunk_index", 0)),
                )

                search_results.append(SearchResult(chunk=chunk, score=similarity, rank=i + 1))

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to search ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"top_k": top_k, "filters": filters, "error": str(e)},
            ) from e

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete.

        Returns:
            Number of chunks actually deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete(["id1", "id2"])
            >>> print(f"Deleted {deleted} chunks")
        """
        if not ids:
            return 0

        try:
            collection = self._ensure_collection()

            # Get count before deletion (ChromaDB doesn't return deleted count)
            # We'll count how many existed
            existing = await asyncio.to_thread(
                collection.get,
                ids=ids,
                include=[],
            )
            count_before = len(existing["ids"]) if existing.get("ids") else 0

            # Delete the chunks
            await asyncio.to_thread(collection.delete, ids=ids)

            logger.debug(f"Deleted {count_before} chunks from ChromaDB")
            return count_before

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to delete chunks from ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"id_count": len(ids), "error": str(e)},
            ) from e

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

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
        try:
            collection = self._ensure_collection()

            # Find all chunks for this document
            results = await asyncio.to_thread(
                collection.get,
                where={"document_id": document_id},
                include=[],
            )

            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                return 0

            # Delete them
            await asyncio.to_thread(collection.delete, ids=ids_to_delete)

            logger.debug(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
            return len(ids_to_delete)

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to delete chunks for document {document_id}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"document_id": document_id, "error": str(e)},
            ) from e

    async def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a specific source path.

        Args:
            source_path: Path of the source document.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.

        Example:
            >>> deleted = await store.delete_by_source("/path/to/file.py")
            >>> print(f"Deleted {deleted} chunks from file")
        """
        try:
            collection = self._ensure_collection()

            # Find all chunks for this source
            results = await asyncio.to_thread(
                collection.get,
                where={"source_path": source_path},
                include=[],
            )

            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                return 0

            # Delete them
            await asyncio.to_thread(collection.delete, ids=ids_to_delete)

            logger.debug(f"Deleted {len(ids_to_delete)} chunks for source {source_path}")
            return len(ids_to_delete)

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to delete chunks for source {source_path}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"source_path": source_path, "error": str(e)},
            ) from e

    async def get(self, ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs.

        Args:
            ids: List of chunk IDs to retrieve.

        Returns:
            List of chunks found. Missing IDs are silently skipped.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> chunks = await store.get(["id1", "id2"])
            >>> for chunk in chunks:
            ...     print(chunk.content)
        """
        if not ids:
            return []

        try:
            collection = self._ensure_collection()

            results = await asyncio.to_thread(
                collection.get,
                ids=ids,
                include=["metadatas", "documents", "embeddings"],
            )

            chunks: list[Chunk] = []
            result_ids = results.get("ids") if results.get("ids") is not None else []
            documents = results.get("documents") if results.get("documents") is not None else []
            metadatas = results.get("metadatas") if results.get("metadatas") is not None else []
            raw_embeddings = results.get("embeddings")
            embeddings_list = list(raw_embeddings) if raw_embeddings is not None else []

            for i, chunk_id in enumerate(result_ids):
                metadata_raw = metadatas[i] if i < len(metadatas) else {}
                metadata_dict: dict[str, Any] = dict(metadata_raw) if metadata_raw else {}
                metadata = Metadata.from_dict(metadata_dict)

                embedding: list[float] | None = None
                if i < len(embeddings_list) and embeddings_list[i] is not None:
                    emb = embeddings_list[i]
                    embedding = list(emb) if hasattr(emb, "__iter__") else None

                chunk = Chunk(
                    id=str(chunk_id),
                    content=str(documents[i]) if i < len(documents) else "",
                    document_id=str(metadata_dict.get("document_id", "")),
                    metadata=metadata,
                    embedding=embedding,
                    start_index=int(metadata_dict.get("start_index", 0)),
                    end_index=int(metadata_dict.get("end_index", 0)),
                    chunk_index=int(metadata_dict.get("chunk_index", 0)),
                )
                chunks.append(chunk)

            return chunks

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to get chunks from ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"id_count": len(ids), "error": str(e)},
            ) from e

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
        try:
            collection = self._ensure_collection()

            results = await asyncio.to_thread(
                collection.get,
                where={"source_path": source_path},
                limit=limit,
                include=["metadatas", "documents"],
            )

            chunks: list[Chunk] = []
            result_ids = results.get("ids") or []
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            for i, chunk_id in enumerate(result_ids):
                metadata_raw = metadatas[i] if i < len(metadatas) else {}
                metadata_dict: dict[str, Any] = dict(metadata_raw) if metadata_raw else {}
                metadata = Metadata.from_dict(metadata_dict)

                chunk = Chunk(
                    id=str(chunk_id),
                    content=str(documents[i]) if i < len(documents) else "",
                    document_id=str(metadata_dict.get("document_id", "")),
                    metadata=metadata,
                    start_index=int(metadata_dict.get("start_index", 0)),
                    end_index=int(metadata_dict.get("end_index", 0)),
                    chunk_index=int(metadata_dict.get("chunk_index", 0)),
                )
                chunks.append(chunk)

            # Sort by chunk_index
            chunks.sort(key=lambda c: c.chunk_index)

            return chunks

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to get chunks for source {source_path}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"source_path": source_path, "limit": limit, "error": str(e)},
            ) from e

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
            collection = self._ensure_collection()

            # Get all metadata
            results = await asyncio.to_thread(
                collection.get,
                include=["metadatas"],
            )

            paths: set[str] = set()
            metadatas_raw = results.get("metadatas") or []

            for metadata_raw in metadatas_raw:
                if metadata_raw:
                    metadata_dict: dict[str, Any] = dict(metadata_raw)
                    if "source_path" in metadata_dict:
                        source_path = metadata_dict["source_path"]
                        if isinstance(source_path, str):
                            paths.add(source_path)

            return sorted(paths)

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to get source paths from ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"error": str(e)},
            ) from e

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> stats = await store.get_stats()
            >>> print(f"Store has {stats['total_chunks']} chunks")
        """
        try:
            collection = self._ensure_collection()

            # Get collection count
            count = await asyncio.to_thread(collection.count)

            # Get unique document count
            results = await asyncio.to_thread(
                collection.get,
                include=["metadatas"],
            )

            document_ids: set[str] = set()
            metadatas_raw = results.get("metadatas") or []

            for metadata_raw in metadatas_raw:
                if metadata_raw:
                    metadata_dict: dict[str, Any] = dict(metadata_raw)
                    if "document_id" in metadata_dict:
                        doc_id = metadata_dict["document_id"]
                        if isinstance(doc_id, str):
                            document_ids.add(doc_id)

            # Get embedding dimension if we have any chunks
            embedding_dimension = 0
            if count > 0:
                peek = await asyncio.to_thread(
                    collection.peek,
                    limit=1,
                )
                peek_embeddings = peek.get("embeddings")
                if peek_embeddings is not None and len(peek_embeddings) > 0:
                    embedding_dimension = len(peek_embeddings[0])

            return {
                "total_chunks": count,
                "total_documents": len(document_ids),
                "embedding_dimension": embedding_dimension,
                "index_type": "hnsw",
                "store_type": "chroma",
                "collection_name": self.collection_name,
                "persist_path": self.persist_path,
            }

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to get stats from ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"error": str(e)},
            ) from e

    async def count(self) -> int:
        """Get the total number of chunks in the store.

        Returns:
            Total chunk count.

        Raises:
            VectorStoreError: If retrieval fails.

        Example:
            >>> total = await store.count()
            >>> print(f"Store has {total} chunks")
        """
        try:
            collection = self._ensure_collection()
            return await asyncio.to_thread(collection.count)

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to count chunks in ChromaDB"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"error": str(e)},
            ) from e

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
        try:
            collection = self._ensure_collection()

            # Get all IDs and delete them
            results = await asyncio.to_thread(
                collection.get,
                include=[],
            )

            ids = results.get("ids", [])
            if ids:
                await asyncio.to_thread(collection.delete, ids=ids)

            logger.info(f"Cleared all {len(ids)} chunks from ChromaDB")

        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = "Failed to clear ChromaDB collection"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(
                error_msg,
                details={"error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """Check if the vector store is healthy and accessible.

        Returns:
            True if the store is operational, False otherwise.
        """
        try:
            # Try to get client and perform a simple operation
            collection = self._ensure_collection()
            await asyncio.to_thread(collection.count)
            return True

        except Exception:
            return False
