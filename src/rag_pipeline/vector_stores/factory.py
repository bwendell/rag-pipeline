"""Factory for creating vector store instances.

This module provides factory functions to instantiate vector stores
for storing and retrieving chunk embeddings. It supports multiple store
backends (ChromaDB, in-memory, Qdrant, OCI) with automatic store
selection and configuration.

Features:
    - Registry-based store management for easy extensibility
    - Lazy imports to avoid loading heavy dependencies (chromadb)
    - Default configurations for each store type
    - Helper functions to inspect available stores
    - Type-safe store instantiation

Example usage:
    >>> from rag_pipeline.vector_stores.factory import (
    ...     create_vector_store,
    ...     get_available_stores,
    ... )
    >>>
    >>> # Create store with defaults (ChromaDB)
    >>> store = create_vector_store()
    >>> await store.health_check()
    >>>
    >>> # Create in-memory store for testing
    >>> store = create_vector_store(store_type="in_memory")
    >>>
    >>> # Create ChromaDB with custom path
    >>> store = create_vector_store(
    ...     store_type="chroma",
    ...     persist_path="./my_data",
    ...     collection_name="my_chunks"
    ... )
    >>>
    >>> # List available stores
    >>> stores = get_available_stores()
    >>> print(f"Available: {stores}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from rag_pipeline.core.types import Chunk, SearchResult


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the vector store interface.

    This is a minimal protocol for type checking purposes. The actual
    VectorStore protocol with all 12 methods is in core/base_vector_store.py.
    """

    async def add(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the store."""
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        ...

    async def health_check(self) -> bool:
        """Check store health."""
        ...


# Store registry - uses lazy loading to avoid importing heavy dependencies
# Maps store names to either classes or lazy loader callables
STORE_REGISTRY: dict[str, type[VectorStoreProtocol] | Any] = {}


def _get_chroma_store() -> type[VectorStoreProtocol]:
    """Lazily import and return ChromaDBStore class.

    Returns:
        The ChromaDBStore class.

    Raises:
        ImportError: If chromadb package is not installed.
    """
    from rag_pipeline.vector_stores.chroma_store import ChromaDBStore

    return ChromaDBStore


def _get_in_memory_store() -> type[VectorStoreProtocol]:
    """Lazily import and return InMemoryStore class.

    Returns:
        The InMemoryStore class.
    """
    from rag_pipeline.vector_stores.in_memory_store import InMemoryStore

    return InMemoryStore


def _get_qdrant_store() -> type[VectorStoreProtocol]:
    """Lazily import and return QdrantStore class.

    Returns:
        The QdrantStore class (stub).

    Raises:
        ImportError: If qdrant-client package is not installed.
    """
    from rag_pipeline.vector_stores.stubs import QdrantStore

    return QdrantStore


def _get_oci_store() -> type[VectorStoreProtocol]:
    """Lazily import and return OCIVectorStore class.

    Returns:
        The OCIVectorStore class (stub).

    Raises:
        ImportError: If oci package is not installed.
    """
    from rag_pipeline.vector_stores.stubs import OCIVectorStore

    return OCIVectorStore


def create_vector_store(
    store_type: str = "chroma",
    **kwargs: Any,
) -> VectorStoreProtocol:
    """Create a vector store instance.

    Instantiates a vector store of the specified type with optional
    configuration. Uses lazy imports to avoid loading heavy dependencies
    (like chromadb) unless actually needed.

    The store is responsible for persisting chunk embeddings and performing
    similarity search in the RAG pipeline.

    Args:
        store_type: Store type identifier. Supported values:
            - "chroma": ChromaDB persistent store (default)
            - "in_memory": In-memory store for testing/development
            - "qdrant": Qdrant vector database (stub)
            - "oci": OCI Vector Search (stub)
        **kwargs: Additional arguments passed to the store constructor.
            These depend on the specific store type. Common examples:
            - persist_path: Directory for data persistence (chroma)
            - collection_name: Name of the collection (chroma, qdrant)
            - url: API endpoint URL (qdrant)
            - compartment_id: OCI compartment OCID (oci)

    Returns:
        A VectorStoreProtocol instance configured with the specified settings.

    Raises:
        ValueError: If store_type is not registered.
        ImportError: If required dependencies for the store are not installed.

    Example:
        >>> # Use default ChromaDB store
        >>> store = create_vector_store()
        >>>
        >>> # Use in-memory store for testing
        >>> store = create_vector_store(store_type="in_memory")
        >>>
        >>> # Use ChromaDB with custom configuration
        >>> store = create_vector_store(
        ...     store_type="chroma",
        ...     persist_path="./data/vectors",
        ...     collection_name="my_rag_app"
        ... )
    """
    # Get the store class
    store_class = get_store_class(store_type)

    # Instantiate the store with provided kwargs
    return store_class(**kwargs)


def get_store_class(store_type: str) -> type[VectorStoreProtocol]:
    """Get the store class registered for a store type.

    Retrieves the vector store class for the specified type. If the
    store is not yet loaded in the registry, it uses lazy loading to
    import the class on-demand.

    Args:
        store_type: The store type identifier to look up.

    Returns:
        The store class registered for this type.

    Raises:
        ValueError: If store_type is not recognized/supported.
        ImportError: If required dependencies are not installed.

    Example:
        >>> store_class = get_store_class("chroma")
        >>> store = store_class(persist_path="./data")
    """
    _ensure_registry_initialized()

    if store_type not in STORE_REGISTRY:
        available = ", ".join(get_available_stores())
        raise ValueError(f"Unknown store type: {store_type}. Available stores: {available}")

    store_entry = STORE_REGISTRY[store_type]

    # If it's callable (lazy loader), call it to get the class
    if callable(store_entry) and not isinstance(store_entry, type):
        store_class = cast("type[VectorStoreProtocol]", store_entry())
        # Cache the class for future use
        STORE_REGISTRY[store_type] = store_class
        return store_class

    # Otherwise it's already a class
    return cast("type[VectorStoreProtocol]", store_entry)


def register_store(
    name: str,
    store_class: type[VectorStoreProtocol],
) -> None:
    """Register a custom vector store class.

    Allows extending the factory with custom vector store implementations.
    The registered store will be available through create_vector_store()
    and other factory functions.

    Custom stores should implement the VectorStore protocol from
    rag_pipeline.core.base_vector_store.

    Args:
        name: The identifier to register the store under. This will be used
            as the store_type argument to create_vector_store().
        store_class: The store class to register.

    Example:
        >>> class CustomVectorStore:
        ...     async def add(self, chunks: list[Chunk]) -> list[str]:
        ...         # Custom implementation
        ...         pass
        ...
        ...     async def search(
        ...         self,
        ...         query_embedding: list[float],
        ...         top_k: int = 5,
        ...         filters: dict[str, Any] | None = None
        ...     ) -> list[SearchResult]:
        ...         # Custom implementation
        ...         pass
        ...
        ...     # ... implement all 12 Protocol methods
        >>>
        >>> register_store("custom", CustomVectorStore)
        >>>
        >>> # Now you can use it
        >>> store = create_vector_store(store_type="custom")
    """
    STORE_REGISTRY[name] = store_class


def get_available_stores() -> list[str]:
    """Get all available vector store types.

    Returns a list of store type identifiers that can be used with
    create_vector_store(). Both built-in and registered custom
    stores are included.

    Returns:
        List of available store type names, sorted alphabetically.

    Example:
        >>> stores = get_available_stores()
        >>> print(f"Available: {stores}")
        Available: ['chroma', 'in_memory', 'oci', 'qdrant']
    """
    # Initialize registry on first access
    _ensure_registry_initialized()
    return sorted(STORE_REGISTRY.keys())


def _ensure_registry_initialized() -> None:
    """Initialize the store registry with built-in stores.

    This is called lazily the first time the registry is accessed to avoid
    importing heavy dependencies at module import time. Only the minimal
    necessary code is executed during module import.

    Built-in stores are:
    - "chroma": ChromaDB persistent vector store
    - "in_memory": Simple in-memory store for testing
    - "qdrant": Qdrant vector database (stub)
    - "oci": OCI Vector Search (stub)
    """
    # Only initialize if registry is empty
    if STORE_REGISTRY:
        return

    # Register built-in stores with lazy loaders (as callables, not called yet)
    STORE_REGISTRY["chroma"] = _get_chroma_store
    STORE_REGISTRY["in_memory"] = _get_in_memory_store
    STORE_REGISTRY["qdrant"] = _get_qdrant_store
    STORE_REGISTRY["oci"] = _get_oci_store
