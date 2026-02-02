"""Sentence-Transformers embedding provider implementation.

This module provides a local embedding provider using the sentence-transformers
library. It supports lazy model loading, batch processing, and efficient async
operations using thread pools for CPU-bound inference.

The default model is "all-MiniLM-L6-v2" which produces 384-dimensional embeddings
and offers a good balance of quality and speed for most RAG use cases.

Example usage:
    >>> from rag_pipeline.embedding_providers import SentenceTransformersProvider
    >>>
    >>> # Initialize provider with default model
    >>> provider = SentenceTransformersProvider()
    >>>
    >>> # Generate single embedding
    >>> embedding = await provider.embed("Hello, world!")
    >>> print(f"Embedding dimension: {len(embedding)}")
    Embedding dimension: 384
    >>>
    >>> # Generate batch embeddings
    >>> texts = ["First text", "Second text", "Third text"]
    >>> embeddings = await provider.embed_batch(texts, batch_size=32)
    >>> print(f"Generated {len(embeddings)} embeddings")
    Generated 3 embeddings
    >>>
    >>> # Check model info
    >>> info = provider.get_model_info()
    >>> print(f"Using {info['name']} ({info['dimension']}d)")
    Using all-MiniLM-L6-v2 (384d)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anyio

from rag_pipeline.core.exceptions import EmbeddingError
from rag_pipeline.embedding_providers.base import (
    AbstractEmbeddingProvider,
    create_zero_vector,
    validate_texts,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rag_pipeline.core.types import EmbeddingVector

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class SentenceTransformersProvider(AbstractEmbeddingProvider):
    """Embedding provider using sentence-transformers library.

    This provider loads and runs sentence-transformers models locally, making
    it ideal for offline use and scenarios where data privacy is important.
    The model is loaded lazily on first use to avoid slow initialization.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
        device: Device to run inference on ("cpu", "cuda", "mps", or None for auto).
        normalize_embeddings: Whether to L2-normalize embeddings after generation.
        _model: The loaded SentenceTransformer model (loaded lazily).

    Example:
        >>> provider = SentenceTransformersProvider(
        ...     model_name="all-MiniLM-L6-v2",
        ...     device="cpu",
        ...     normalize_embeddings=True
        ... )
        >>> embedding = await provider.embed("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
        Dimension: 384
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the sentence-transformers embedding provider.

        Args:
            model_name: Name of the sentence-transformers model to load.
                Defaults to "all-MiniLM-L6-v2" (384 dimensions).
            device: Device to run inference on. Options:
                - "cpu": Use CPU
                - "cuda": Use CUDA GPU
                - "mps": Use Apple Silicon MPS
                - None: Auto-detect best available device
            normalize_embeddings: Whether to L2-normalize embeddings.
                Normalized embeddings are required for cosine similarity.

        Raises:
            ValueError: If model_name is empty or invalid.

        Example:
            >>> provider = SentenceTransformersProvider(
            ...     model_name="all-MiniLM-L6-v2",
            ...     device="cpu"
            ... )
        """
        super().__init__(model_name=model_name, dimension=DEFAULT_DIMENSION)
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None

    def _get_provider_name(self) -> str:
        """Get the name of this embedding provider.

        Returns:
            Provider identifier string "sentence-transformers".

        Example:
            >>> name = provider._get_provider_name()
            >>> print(name)
            sentence-transformers
        """
        return "sentence-transformers"

    def _ensure_model_loaded(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model on first use.

        This method loads the model if it hasn't been loaded yet and caches
        it for subsequent calls. Loading happens in a thread to avoid blocking
        the event loop.

        Returns:
            The loaded SentenceTransformer model instance.

        Raises:
            EmbeddingError: If model loading fails.

        Example:
            >>> model = provider._ensure_model_loaded()
            >>> print(f"Model loaded: {model is not None}")
            Model loaded: True
        """
        if self._model is None:
            try:
                logger.info(f"Loading sentence-transformers model: {self.model_name}")

                # Import here to avoid loading at module import time
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )

                # Update dimension from the loaded model
                self._dimension = self._model.get_sentence_embedding_dimension()

                logger.info(f"Loaded model '{self.model_name}' with dimension {self._dimension}")
            except Exception as e:
                error_msg = f"Failed to load sentence-transformers model '{self.model_name}'"
                logger.error(error_msg, exc_info=True)
                raise EmbeddingError(
                    error_msg,
                    details={"model": self.model_name, "error": str(e)},
                ) from e

        return self._model

    async def embed(self, text: str) -> EmbeddingVector:
        """Generate an embedding for a single text.

        Converts the input text into a dense vector representation using the
        sentence-transformers model. Empty strings return zero vectors.
        The actual inference runs in a thread pool to avoid blocking.

        Args:
            text: The text to embed. Empty strings return zero vectors.

        Returns:
            A list of floats representing the embedding vector. The dimension
            matches `get_embedding_dimension()` (typically 384 for the default
            all-MiniLM-L6-v2 model).

        Raises:
            EmbeddingError: If embedding generation fails.

        Example:
            >>> embedding = await provider.embed("Hello, world!")
            >>> print(f"Dimension: {len(embedding)}")
            Dimension: 384
            >>>
            >>> # Empty string returns zero vector
            >>> zero = await provider.embed("")
            >>> print(all(x == 0.0 for x in zero))
            True
        """
        # Handle empty string - return zero vector
        if not text:
            return create_zero_vector(self.get_embedding_dimension())

        try:
            model = self._ensure_model_loaded()

            # Run CPU-bound inference in thread pool
            embedding = await anyio.to_thread.run_sync(
                lambda: model.encode(
                    text,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                ).tolist(),
            )

            return embedding
        except EmbeddingError:
            # Re-raise EmbeddingError directly
            raise
        except Exception as e:
            error_msg = "Failed to generate embedding for text"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingError(
                error_msg,
                details={
                    "model": self.model_name,
                    "text_length": len(text),
                    "error": str(e),
                },
            ) from e

    async def embed_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts efficiently.

        More efficient than calling embed() multiple times as it batches the
        inference and can utilize GPU parallelism. Handles empty strings by
        returning zero vectors at the correct positions.

        Args:
            texts: List of texts to embed. Empty strings are allowed and will
                produce zero vectors. Empty list raises ValueError.
            batch_size: Number of texts to process in each batch. If None,
                uses the default of 32. Larger batches are more efficient but
                use more memory.
            show_progress: If True, displays a progress bar during processing
                (requires tqdm to be installed). Useful for large batches.

        Returns:
            List of embedding vectors, one per input text, in the same order
            as the input. Each vector has dimension matching
            `get_embedding_dimension()`.

        Raises:
            ValueError: If texts list is empty.
            EmbeddingError: If batch embedding generation fails.

        Example:
            >>> texts = ["First text", "", "Third text"]
            >>> embeddings = await provider.embed_batch(texts, batch_size=32)
            >>> print(f"Generated {len(embeddings)} embeddings")
            Generated 3 embeddings
            >>> # Empty string at index 1 returns zero vector
            >>> print(all(x == 0.0 for x in embeddings[1]))
            True
        """
        validate_texts(texts)

        effective_batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

        try:
            model = self._ensure_model_loaded()

            # Track which texts are empty (need zero vectors)
            empty_indices: set[int] = set()
            non_empty_texts: list[str] = []
            non_empty_indices: list[int] = []

            for i, text in enumerate(texts):
                if not text:
                    empty_indices.add(i)
                else:
                    non_empty_texts.append(text)
                    non_empty_indices.append(i)

            # If all texts are empty, return all zero vectors
            if not non_empty_texts:
                zero_vector = create_zero_vector(self.get_embedding_dimension())
                return [zero_vector.copy() for _ in texts]

            # Generate embeddings for non-empty texts in thread pool
            logger.debug(
                f"Embedding batch of {len(non_empty_texts)} texts "
                f"(batch_size={effective_batch_size}, show_progress={show_progress})"
            )

            embeddings_array = await anyio.to_thread.run_sync(
                lambda: model.encode(
                    non_empty_texts,
                    batch_size=effective_batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                ),
            )

            # Convert to list of lists
            non_empty_embeddings: list[list[float]] = embeddings_array.tolist()

            # Reconstruct full result with zero vectors for empty strings
            result: list[EmbeddingVector] = []
            non_empty_iter = iter(non_empty_embeddings)

            for i in range(len(texts)):
                if i in empty_indices:
                    result.append(create_zero_vector(self.get_embedding_dimension()))
                else:
                    result.append(next(non_empty_iter))

            logger.debug(f"Successfully generated {len(result)} embeddings")
            return result

        except EmbeddingError:
            # Re-raise EmbeddingError directly
            raise
        except Exception as e:
            error_msg = "Failed to generate batch embeddings"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingError(
                error_msg,
                details={
                    "model": self.model_name,
                    "text_count": len(texts),
                    "batch_size": effective_batch_size,
                    "error": str(e),
                },
            ) from e
