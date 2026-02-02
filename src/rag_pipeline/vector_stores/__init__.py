"""Vector stores module for embedding storage and retrieval.

This module provides vector store implementations for persisting chunk
embeddings and performing similarity search operations.

Available stores:
- ChromaDBStore: ChromaDB persistent store (default, production-ready)
- InMemoryStore: In-memory store for testing and development
- OCIVectorStore: Oracle Database 23ai Vector Search (stub)
- QdrantStore: Qdrant vector database (stub)

Example usage:
    >>> from rag_pipeline.vector_stores import (
    ...     create_vector_store,
    ...     ChromaDBStore,
    ...     InMemoryStore,
    ... )
    >>>
    >>> # Using factory (recommended)
    >>> store = create_vector_store()  # Uses ChromaDB by default
    >>> await store.health_check()
    >>>
    >>> # Using factory with in-memory store for testing
    >>> store = create_vector_store(store_type="in_memory")
    >>>
    >>> # Using class directly
    >>> store = ChromaDBStore(persist_path="./data/vectors")
    >>> await store.add(chunks)
    >>> results = await store.search(query_embedding, top_k=5)
    >>>
    >>> # In-memory for unit tests
    >>> store = InMemoryStore()
    >>> await store.add(chunks)
    >>> assert await store.count() == len(chunks)
"""

from rag_pipeline.vector_stores.chroma_store import ChromaDBStore
from rag_pipeline.vector_stores.factory import (
    create_vector_store,
    get_available_stores,
    get_store_class,
    register_store,
)
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore
from rag_pipeline.vector_stores.stubs import OCIVectorStore, QdrantStore

__all__ = [
    "ChromaDBStore",
    "InMemoryStore",
    "OCIVectorStore",
    "QdrantStore",
    "create_vector_store",
    "get_available_stores",
    "get_store_class",
    "register_store",
]
