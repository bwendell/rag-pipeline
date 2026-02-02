"""Embedding providers module for text-to-vector conversion.

This module provides embedding provider implementations for converting text into
dense vector representations suitable for similarity search and retrieval.

Available providers:
- SentenceTransformersProvider: Local sentence-transformers models (default)
- OpenAIEmbeddingProvider: OpenAI API embeddings (stub)
- OCIEmbeddingProvider: OCI Generative AI embeddings (stub)

Example usage:
    >>> from rag_pipeline.embedding_providers import (
    ...     create_embedding_provider,
    ...     SentenceTransformersProvider,
    ... )
    >>>
    >>> # Using factory (recommended)
    >>> provider = create_embedding_provider()  # Uses sentence-transformers by default
    >>> embedding = await provider.embed("Hello, world!")
    >>>
    >>> # Using class directly
    >>> provider = SentenceTransformersProvider()
    >>> embedding = await provider.embed("Hello, world!")
    >>> print(f"Embedding dimension: {len(embedding)}")
    384
    >>>
    >>> # Use base classes for typing
    >>> from rag_pipeline.embedding_providers import AbstractEmbeddingProvider
    >>> def process(provider: AbstractEmbeddingProvider, texts: list[str]) -> None:
    ...     pass
"""

from rag_pipeline.embedding_providers.base import (
    AbstractEmbeddingProvider,
    create_zero_vector,
    validate_texts,
)
from rag_pipeline.embedding_providers.factory import (
    create_embedding_provider,
    get_available_providers,
    get_default_model,
    get_provider_class,
    register_provider,
)
from rag_pipeline.embedding_providers.sentence_transformers_provider import (
    SentenceTransformersProvider,
)
from rag_pipeline.embedding_providers.stubs import (
    OCIEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

__all__ = [
    "AbstractEmbeddingProvider",
    "OCIEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformersProvider",
    "create_embedding_provider",
    "create_zero_vector",
    "get_available_providers",
    "get_default_model",
    "get_provider_class",
    "register_provider",
    "validate_texts",
]
