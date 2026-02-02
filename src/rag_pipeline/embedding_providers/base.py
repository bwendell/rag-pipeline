"""Base embedding provider implementation with common functionality.

This module provides the AbstractEmbeddingProvider base class that implements
shared functionality for all embedding provider implementations, including:
- Configuration validation and storage
- Embedding dimension management
- Model information retrieval
- Health checking
- Helper functions for text validation and zero vectors

Example usage:
    >>> from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider
    >>>
    >>> class MySentenceTransformerProvider(AbstractEmbeddingProvider):
    ...     async def embed(self, text: str) -> EmbeddingVector:
    ...         # Implementation using sentence-transformers
    ...         pass
    ...
    ...     async def embed_batch(
    ...         self,
    ...         texts: list[str],
    ...         *,
    ...         batch_size: int | None = None,
    ...         show_progress: bool = False
    ...     ) -> list[EmbeddingVector]:
    ...         # Implementation for batch embedding
    ...         pass
    ...
    ...     def _get_provider_name(self) -> str:
    ...         return "sentence-transformers"
    ...
    ...
    >>> provider = MySentenceTransformerProvider(
    ...     model_name="all-MiniLM-L6-v2",
    ...     dimension=384
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_pipeline.core.types import EmbeddingVector


class AbstractEmbeddingProvider(ABC):
    """Abstract base class for embedding provider implementations.

    Provides common functionality shared across all embedding providers, including
    configuration management, dimension tracking, model information retrieval, and
    health checking. Concrete implementations must override the abstract methods
    `embed()`, `embed_batch()`, and `_get_provider_name()`.

    Attributes:
        model_name: Name of the embedding model (e.g., "all-MiniLM-L6-v2").
        _dimension: Dimension of embedding vectors (can be None if not pre-configured).

    Example:
        >>> provider = ConcreteEmbeddingProvider(
        ...     model_name="all-MiniLM-L6-v2",
        ...     dimension=384
        ... )
        >>> embedding = await provider.embed("Hello, world!")
        >>> print(f"Got embedding with dimension {len(embedding)}")
        Got embedding with dimension 384
    """

    def __init__(self, model_name: str, dimension: int | None = None) -> None:
        """Initialize the embedding provider.

        Args:
            model_name: Name of the embedding model. Should be a valid identifier
                for the specific provider (e.g., "all-MiniLM-L6-v2" for
                sentence-transformers, "text-embedding-ada-002" for OpenAI).
            dimension: Dimension of the embedding vectors. If None, the provider
                should determine this from the model (may require loading the model).

        Raises:
            ValueError: If model_name is empty or invalid.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")

        self.model_name = model_name
        self._dimension = dimension

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns the embedding dimension that was configured or inferred from
        the model. Must be positive.

        Returns:
            The dimension of embedding vectors as a positive integer.

        Raises:
            RuntimeError: If dimension is not yet known (model not loaded).

        Example:
            >>> dim = provider.get_embedding_dimension()
            >>> print(f"Embeddings have dimension {dim}")
            Embeddings have dimension 384
        """
        if self._dimension is None:
            raise RuntimeError(
                f"Embedding dimension not yet available for {self.model_name}. "
                "The model may not be loaded yet."
            )
        return self._dimension

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model.

        Returns comprehensive information about the embedding model including
        its name, provider, dimension, and other metadata useful for debugging
        and monitoring.

        Returns:
            Dictionary containing model information:
                - name: Model name (e.g., "all-MiniLM-L6-v2")
                - provider: Provider name (e.g., "sentence-transformers")
                - dimension: Embedding dimension
                - other provider-specific metadata

        Example:
            >>> info = provider.get_model_info()
            >>> print(f"Using {info['name']} ({info['dimension']}d)")
            Using all-MiniLM-L6-v2 (384d)
        """
        return {
            "name": self.model_name,
            "provider": self._get_provider_name(),
            "dimension": self.get_embedding_dimension(),
        }

    async def health_check(self) -> bool:
        """Check if the embedding provider is healthy and operational.

        Performs a basic health check by attempting to embed a test string.
        Returns True if successful, False if the provider is unavailable or
        encounters an error.

        Returns:
            True if the provider is operational, False otherwise.

        Example:
            >>> is_healthy = await provider.health_check()
            >>> if is_healthy:
            ...     print("Embedding provider is ready")
            ... else:
            ...     print("Embedding provider is unavailable")
        """
        try:
            _ = await self.embed("test")
            return True
        except Exception:
            return False

    async def embed_query(self, query: str) -> EmbeddingVector:
        """Generate an embedding optimized for queries.

        Some embedding models use different encoding strategies for queries
        versus documents. This method is used for search queries. By default,
        it delegates to `embed()`, but subclasses can override it for models
        with specific query encoding requirements.

        Args:
            query: The search query text to embed.

        Returns:
            A query-optimized embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If query is empty.

        Example:
            >>> query_embedding = await provider.embed_query("How does auth work?")
            >>> results = await store.search(query_embedding, top_k=5)
        """
        return await self.embed(query)

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingVector:
        """Generate an embedding for a single text.

        Converts input text into a dense vector representation suitable for
        similarity search. Concrete implementations must provide the actual
        embedding logic.

        Args:
            text: The text to embed. Can be any string, including empty strings
                which should return zero vectors.

        Returns:
            A list of floats representing the embedding vector. The dimension
            matches `get_embedding_dimension()`.

        Raises:
            EmbeddingError: If embedding generation fails.

        Example:
            >>> embedding = await provider.embed("Hello, world!")
            >>> print(f"Embedding dimension: {len(embedding)}")
            Embedding dimension: 384
        """
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts efficiently.

        More efficient than calling `embed()` multiple times, as it can batch
        API calls or utilize GPU parallelism. Concrete implementations should
        provide optimized batching logic.

        Args:
            texts: List of texts to embed. Empty strings are allowed and
                should produce zero vectors. Empty list is an error.
            batch_size: Optional batch size for processing. If None, uses
                the provider's default or sensible default. Larger batches
                are typically more efficient but use more memory.
            show_progress: If True, display a progress bar during embedding
                (useful for large batches). Ignored if the provider doesn't
                support progress reporting.

        Returns:
            List of embedding vectors, one per input text, in the same order
            as the input. Each vector has dimension matching `get_embedding_dimension()`.

        Raises:
            EmbeddingError: If batch embedding generation fails.
            ValueError: If texts list is empty.

        Example:
            >>> texts = ["Hello", "World", "How are you?"]
            >>> embeddings = await provider.embed_batch(texts, batch_size=32)
            >>> print(f"Generated {len(embeddings)} embeddings")
            Generated 3 embeddings
        """
        ...

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the name of this embedding provider.

        Used for identification and logging. Concrete implementations should
        return a consistent provider identifier.

        Returns:
            Provider name as a string (e.g., "sentence-transformers", "openai",
            "oci", "cohere").

        Example:
            >>> name = provider._get_provider_name()
            >>> print(f"Provider: {name}")
            Provider: sentence-transformers
        """
        ...


def validate_texts(texts: list[str]) -> None:
    """Validate a list of texts for embedding.

    Ensures that the texts list is not empty. Empty strings within the list
    are valid and will produce zero vectors.

    Args:
        texts: List of texts to validate.

    Raises:
        ValueError: If texts list is empty.

    Example:
        >>> validate_texts(["hello", "world"])  # OK
        >>> validate_texts(["", "text"])  # OK (empty strings are allowed)
        >>> validate_texts([])  # Raises ValueError
    """
    if not texts:
        raise ValueError("texts list cannot be empty")


def create_zero_vector(dimension: int) -> EmbeddingVector:
    """Create a zero vector of specified dimension.

    Returns a vector of zeros with the given dimension. Used for empty
    strings or when an embedding cannot be generated.

    Args:
        dimension: The dimension of the zero vector to create. Must be positive.

    Returns:
        A list of `dimension` zeros as floats.

    Raises:
        ValueError: If dimension is not positive.

    Example:
        >>> zero_vec = create_zero_vector(384)
        >>> print(len(zero_vec))
        384
        >>> print(all(x == 0.0 for x in zero_vec))
        True
    """
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    return [0.0] * dimension
