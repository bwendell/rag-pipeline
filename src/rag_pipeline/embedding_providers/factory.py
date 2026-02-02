"""Factory for creating embedding provider instances.

This module provides factory functions to instantiate embedding providers
for generating vector representations of text. It supports multiple provider
backends (local sentence-transformers, OpenAI API, OCI GenAI) with automatic
provider selection and configuration.

Features:
    - Registry-based provider management for easy extensibility
    - Lazy imports to avoid loading heavy dependencies (sentence-transformers)
    - Default model configurations for each provider
    - Helper functions to inspect available providers
    - Type-safe provider instantiation

Example usage:
    >>> from rag_pipeline.embedding_providers.factory import (
    ...     create_embedding_provider,
    ...     get_available_providers,
    ...     get_default_model,
    ... )
    >>>
    >>> # Create provider with defaults
    >>> provider = create_embedding_provider()
    >>> embedding = await provider.embed("Hello, world!")
    >>>
    >>> # Create provider with custom model
    >>> provider = create_embedding_provider(
    ...     provider_type="sentence-transformers",
    ...     model_name="all-mpnet-base-v2"
    ... )
    >>>
    >>> # List available providers
    >>> providers = get_available_providers()
    >>> print(f"Available: {providers}")
    >>>
    >>> # Get default model for a provider
    >>> default_model = get_default_model("openai")
    >>> print(f"OpenAI default: {default_model}")
"""

from __future__ import annotations

from typing import Any, cast

from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider

# Default model configurations for each provider
DEFAULT_MODELS: dict[str, str] = {
    "sentence-transformers": "all-MiniLM-L6-v2",
    "openai": "text-embedding-ada-002",
    "oci": "cohere.embed-english-v3.0",
}

# Provider registry - uses lazy loading to avoid importing heavy dependencies
# Maps provider names to either classes or lazy loader callables
PROVIDER_REGISTRY: dict[str, type[AbstractEmbeddingProvider] | Any] = {}


def _get_sentence_transformers_provider() -> type[AbstractEmbeddingProvider]:
    """Lazily import and return SentenceTransformersProvider class.

    Returns:
        The SentenceTransformersProvider class.

    Raises:
        ImportError: If sentence-transformers package is not installed.
    """
    from rag_pipeline.embedding_providers.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    return SentenceTransformersProvider


def _get_openai_provider() -> type[AbstractEmbeddingProvider]:
    """Lazily import and return OpenAIEmbeddingProvider class.

    Returns:
        The OpenAIEmbeddingProvider class.

    Raises:
        ImportError: If openai package is not installed.
    """
    from rag_pipeline.embedding_providers.stubs import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider


def _get_oci_provider() -> type[AbstractEmbeddingProvider]:
    """Lazily import and return OCIEmbeddingProvider class.

    Returns:
        The OCIEmbeddingProvider class.

    Raises:
        ImportError: If oci package is not installed.
    """
    from rag_pipeline.embedding_providers.stubs import OCIEmbeddingProvider

    return OCIEmbeddingProvider


def get_default_model(provider_type: str) -> str:
    """Get the default model name for a provider type.

    Returns the recommended default model for each provider. These defaults
    balance quality and performance for typical RAG use cases.

    Args:
        provider_type: The provider type identifier ("sentence-transformers", "openai", "oci").

    Returns:
        The default model name for the provider.

    Raises:
        ValueError: If provider_type is not recognized.

    Example:
        >>> model = get_default_model("sentence-transformers")
        >>> print(model)
        all-MiniLM-L6-v2
    """
    if provider_type not in DEFAULT_MODELS:
        available = ", ".join(DEFAULT_MODELS.keys())
        raise ValueError(
            f"Unknown provider type: {provider_type}. Available providers: {available}"
        )
    return DEFAULT_MODELS[provider_type]


def create_embedding_provider(
    provider_type: str = "sentence-transformers",
    model_name: str | None = None,
    **kwargs: Any,
) -> AbstractEmbeddingProvider:
    """Create an embedding provider instance.

    Instantiates an embedding provider of the specified type with optional
    model name override. Uses lazy imports to avoid loading heavy dependencies
    (like sentence-transformers) unless actually needed.

    The provider is responsible for generating vector embeddings of text for
    similarity search and semantic analysis in the RAG pipeline.

    Args:
        provider_type: Provider type identifier. Supported values:
            - "sentence-transformers": Local transformers-based embeddings (default)
            - "openai": OpenAI API-based embeddings
            - "oci": OCI Generative AI Service embeddings
        model_name: Optional model name override. If None, uses the provider's default model.
            Examples:
            - For "sentence-transformers": "all-MiniLM-L6-v2", "all-mpnet-base-v2"
            - For "openai": "text-embedding-ada-002", "text-embedding-3-small"
            - For "oci": "cohere.embed-english-v3.0"
        **kwargs: Additional arguments passed to the provider constructor.
            These depend on the specific provider type. Common examples:
            - cache_dir: Directory for caching downloaded models (sentence-transformers)
            - api_key: API key for cloud providers (openai, oci)
            - compartment_id: OCI compartment OCID (oci)

    Returns:
        An AbstractEmbeddingProvider instance configured with the specified settings.

    Raises:
        ValueError: If provider_type is not registered.
        ImportError: If required dependencies for the provider are not installed.

    Example:
        >>> # Use default sentence-transformers provider
        >>> provider = create_embedding_provider()
        >>>
        >>> # Use OpenAI with API key from environment
        >>> provider = create_embedding_provider(
        ...     provider_type="openai",
        ...     api_key="sk-..."
        ... )
        >>>
        >>> # Use different sentence-transformers model
        >>> provider = create_embedding_provider(
        ...     provider_type="sentence-transformers",
        ...     model_name="all-mpnet-base-v2",
        ...     cache_dir="/path/to/cache"
        ... )
    """
    # Get the provider class
    provider_class = get_provider_class(provider_type)

    # Use provider default if model_name not specified
    if model_name is None:
        model_name = get_default_model(provider_type)

    # Instantiate the provider with model name and additional kwargs
    return provider_class(model_name=model_name, **kwargs)


def get_provider_class(provider_type: str) -> type[AbstractEmbeddingProvider]:
    """Get the provider class registered for a provider type.

    Retrieves the embedding provider class for the specified type. If the
    provider is not yet loaded in the registry, it uses lazy loading to
    import the class on-demand.

    Args:
        provider_type: The provider type identifier to look up.

    Returns:
        The provider class registered for this type.

    Raises:
        ValueError: If provider_type is not recognized/supported.
        ImportError: If required dependencies are not installed.

    Example:
        >>> provider_class = get_provider_class("sentence-transformers")
        >>> provider = provider_class(model_name="all-MiniLM-L6-v2")
    """
    _ensure_registry_initialized()

    if provider_type not in PROVIDER_REGISTRY:
        available = ", ".join(get_available_providers())
        raise ValueError(
            f"Unknown provider type: {provider_type}. Available providers: {available}"
        )

    provider_entry = PROVIDER_REGISTRY[provider_type]

    # If it's callable (lazy loader), call it to get the class
    if callable(provider_entry) and not isinstance(provider_entry, type):
        provider_class = cast("type[AbstractEmbeddingProvider]", provider_entry())
        # Cache the class for future use
        PROVIDER_REGISTRY[provider_type] = provider_class
        return provider_class

    # Otherwise it's already a class
    return cast("type[AbstractEmbeddingProvider]", provider_entry)


def register_provider(
    name: str,
    provider_class: type[AbstractEmbeddingProvider],
) -> None:
    """Register a custom embedding provider class.

    Allows extending the factory with custom embedding provider implementations.
    The registered provider will be available through create_embedding_provider()
    and other factory functions.

    Custom providers must implement the AbstractEmbeddingProvider interface.

    Args:
        name: The identifier to register the provider under. This will be used
            as the provider_type argument to create_embedding_provider().
        provider_class: The provider class to register. Must be a subclass of
            AbstractEmbeddingProvider.

    Raises:
        TypeError: If provider_class is not a subclass of AbstractEmbeddingProvider.

    Example:
        >>> class CustomEmbeddingProvider(AbstractEmbeddingProvider):
        ...     def _get_provider_name(self) -> str:
        ...         return "custom"
        ...
        ...     async def embed(self, text: str) -> EmbeddingVector:
        ...         # Custom implementation
        ...         pass
        ...
        ...     async def embed_batch(
        ...         self,
        ...         texts: list[str],
        ...         *,
        ...         batch_size: int | None = None,
        ...         show_progress: bool = False
        ...     ) -> list[EmbeddingVector]:
        ...         # Custom batch implementation
        ...         pass
        >>>
        >>> register_provider("custom", CustomEmbeddingProvider)
        >>>
        >>> # Now you can use it
        >>> provider = create_embedding_provider(provider_type="custom")
    """
    if not issubclass(provider_class, AbstractEmbeddingProvider):
        raise TypeError(
            f"provider_class must be a subclass of AbstractEmbeddingProvider, got {provider_class}"
        )
    PROVIDER_REGISTRY[name] = provider_class


def get_available_providers() -> list[str]:
    """Get all available embedding provider types.

    Returns a list of provider type identifiers that can be used with
    create_embedding_provider(). Both built-in and registered custom
    providers are included.

    Returns:
        List of available provider type names, sorted alphabetically.

    Example:
        >>> providers = get_available_providers()
        >>> print(f"Available: {providers}")
        Available: ['oci', 'openai', 'sentence-transformers']
    """
    # Initialize registry on first access
    _ensure_registry_initialized()
    return sorted(PROVIDER_REGISTRY.keys())


def _ensure_registry_initialized() -> None:
    """Initialize the provider registry with built-in providers.

    This is called lazily the first time the registry is accessed to avoid
    importing heavy dependencies at module import time. Only the minimal
    necessary code is executed during module import.

    Built-in providers are:
    - "sentence-transformers": Local sentence-transformers based embeddings
    - "openai": OpenAI API embeddings
    - "oci": OCI Generative AI Service embeddings
    """
    # Only initialize if registry is empty
    if PROVIDER_REGISTRY:
        return

    # Register built-in providers with lazy loaders (as callables, not called yet)
    PROVIDER_REGISTRY["sentence-transformers"] = _get_sentence_transformers_provider
    PROVIDER_REGISTRY["openai"] = _get_openai_provider
    PROVIDER_REGISTRY["oci"] = _get_oci_provider
