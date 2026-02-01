"""EmbeddingProvider Protocol definition.

This module defines the abstract interface for embedding provider implementations.
All embedding providers (sentence-transformers, OpenAI, OCI, etc.) must conform
to this protocol.

Example usage:
    >>> from rag_pipeline.core import EmbeddingProvider
    >>>
    >>> class MyEmbeddingProvider:
    ...     async def embed(self, text: str) -> list[float]:
    ...         # Generate embedding for single text
    ...         ...
    ...
    ...     async def embed_batch(self, texts: list[str]) -> list[list[float]]:
    ...         # Generate embeddings for multiple texts
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyEmbeddingProvider(), EmbeddingProvider)
"""

from typing import Any, Protocol, runtime_checkable

from rag_pipeline.core.types import EmbeddingVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding provider implementations.

    Embedding providers are responsible for:
    1. Converting text into dense vector representations
    2. Supporting both single and batch embedding
    3. Providing information about embedding dimensions

    All embedding methods are async to support both local models
    (which may use thread pools) and remote APIs.

    Implementations:
        - SentenceTransformersProvider: Local sentence-transformers models
        - OpenAIEmbeddingProvider (stub): OpenAI embeddings API
        - OCIEmbeddingProvider (stub): OCI Generative AI embeddings
    """

    async def embed(self, text: str) -> EmbeddingVector:
        """Generate an embedding for a single text.

        Converts the input text into a dense vector representation
        suitable for similarity search.

        Args:
            text: The text to embed. Should be non-empty.

        Returns:
            A list of floats representing the embedding vector.
            The dimension depends on the model (e.g., 384 for
            all-MiniLM-L6-v2, 1536 for OpenAI ada-002).

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.

        Example:
            >>> embedding = await provider.embed("Hello, world!")
            >>> print(f"Dimension: {len(embedding)}")
            Dimension: 384
        """
        ...

    async def embed_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False
    ) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts.

        More efficient than calling embed() multiple times, as it
        can batch API calls or utilize GPU parallelism.

        Args:
            texts: List of texts to embed. Empty strings are allowed
                but will produce zero vectors.
            batch_size: Optional batch size for processing. If None,
                uses the provider's default batch size.
            show_progress: If True, show progress bar (for large batches).

        Returns:
            List of embedding vectors, in the same order as input texts.
            Each vector has the same dimension.

        Raises:
            EmbeddingError: If embedding generation fails.

        Example:
            >>> texts = ["Hello", "World", "How are you?"]
            >>> embeddings = await provider.embed_batch(texts)
            >>> print(f"Generated {len(embeddings)} embeddings")
        """
        ...

    async def embed_query(self, query: str) -> EmbeddingVector:
        """Generate an embedding optimized for queries.

        Some embedding models use different encoding strategies for
        queries vs documents. This method should be used for search
        queries, while embed() is used for documents.

        Args:
            query: The search query to embed.

        Returns:
            A query-optimized embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.

        Note:
            For most models, this is identical to embed(). Override
            only if the model requires different query encoding.

        Example:
            >>> query_embedding = await provider.embed_query("How does auth work?")
            >>> results = await store.search(query_embedding, top_k=5)
        """
        ...

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns:
            Integer dimension of embedding vectors.

        Example:
            >>> dim = provider.get_embedding_dimension()
            >>> print(f"Embeddings have dimension {dim}")
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model.

        Returns:
            Dictionary containing model information:
                - name: Model name (e.g., "all-MiniLM-L6-v2")
                - provider: Provider name (e.g., "sentence-transformers")
                - dimension: Embedding dimension
                - max_tokens: Maximum input tokens
                - normalized: Whether embeddings are L2-normalized

        Example:
            >>> info = provider.get_model_info()
            >>> print(f"Using {info['name']} ({info['dimension']}d)")
        """
        ...

    async def health_check(self) -> bool:
        """Check if the embedding provider is healthy.

        Returns:
            True if the provider is operational, False otherwise.

        Example:
            >>> if await provider.health_check():
            ...     print("Embedding provider is ready")
        """
        ...


@runtime_checkable
class CachedEmbeddingProvider(EmbeddingProvider, Protocol):
    """Extended protocol for embedding providers with caching.

    Adds cache management capabilities for providers that cache
    embeddings to avoid redundant computation.
    """

    async def get_cached(self, text: str) -> EmbeddingVector | None:
        """Get a cached embedding if available.

        Args:
            text: The text to look up.

        Returns:
            Cached embedding vector, or None if not cached.
        """
        ...

    async def clear_cache(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of cache entries cleared.
        """
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats:
                - size: Number of cached embeddings
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate (0.0 - 1.0)
        """
        ...
